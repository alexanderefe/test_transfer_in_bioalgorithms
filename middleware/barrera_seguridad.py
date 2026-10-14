"""
Barrera de seguridad — Wasserstein + MMD (middleware/barrera_seguridad.py)

Implementa los dos componentes de la barrera de seguridad descritos en
las secciones 2.10 y 4.2.3 del documento de tesis:

  1. Distancia de Wasserstein (W1): BARRERA DURA aplicada sobre las
     poblaciones completas del algoritmo fuente y objetivo, antes de
     consolidar cualquier transferencia (ecuaciones 5-7, criterio de
     control de la ecuación 7: si W1 > θ_W, se aborta la transferencia).

  2. MMD (Maximum Mean Discrepancy): FILTRO FINO aplicado sobre
     instancias candidatas individuales extraídas en la Fase 2, para
     verificar que cada instancia específica pertenezca a una
     distribución compatible con el espacio de búsqueda del algoritmo
     objetivo. Se aplica DESPUÉS de la barrera dura: su MMD está centrado
     (compara forma, no ubicación) y asume que Wasserstein ya descartó
     las poblaciones demasiado alejadas. El orquestador evalúa la barrera
     al entrar a la Fase 3 y de nuevo antes de escalar de Canal A a
     Canal B (las poblaciones se mueven durante la ventana de Canal A).

DECISIONES DE FORMALIZACIÓN ACORDADAS EXPLÍCITAMENTE (el documento no
especifica valores ni implementación concreta para ninguna de las dos):

1. Wasserstein: se calcula de forma EXACTA como un problema de
   programación lineal de transporte óptimo (ecuación 6 del documento),
   resuelto con scipy.optimize.linprog (HiGHS), en vez de aproximaciones
   (POT se descartó: sin wheels para Python 3.14 en Windows). El umbral θ_W se define como una FRACCIÓN RELATIVA
   del diámetro máximo del hipercubo de búsqueda (√D × rango_por_dim),
   en vez de un valor absoluto fijo. Esto se decidió tras observar
   numéricamente que W1 varía en un rango muy amplio (130 a 344, en el
   caso real PSO/DE sobre F11/D=20) y que un umbral relativo generaliza
   correctamente a las 12 funciones del benchmark CEC 2022, que tienen
   distintos rangos y dimensionalidades, sin necesitar recalibración
   manual por función. θ_W = 0.6 (60% del diámetro máximo), recalibrado
   en D-11 (2026-10-13). El valor original, 0.25, se eligió con el
   evaluador opfunu (fracciones observadas ~15%-38%); con el CEC 2022
   oficial (D-9) PSO y DE convergen a cuencas distintas (fracciones
   0.31-0.63, constantes durante toda la corrida) y 0.25 dejaba sin
   transferencias a 6/12 funciones en D=10 y 7/12 en D=20. Barrido
   θ_W ∈ {0.25, 0.4, 0.6, ∞} (12 funciones × D∈{10,20} × 5 semillas,
   MaxFES=5·10⁴): 0.25 fue peor que ∞ (20 peores / 5 mejores, Wilcoxon
   p=0.02); 0.6 no se distingue de ∞ (0 mejores / 2 peores, p=0.5) y
   conserva la barrera de la ecuación 7 para casos extremos.

2. MMD: la heurística estándar de ancho de banda del kernel RBF (mediana
   de TODAS las distancias por pares, incluyendo pares entre los dos
   grupos comparados) resultó NO ser confiable en este problema: con
   D=20 y rango [-100,100], la mediana global se infla por la alta
   dimensionalidad, produciendo un gamma demasiado pequeño que aplana el
   kernel y le hace perder poder discriminativo (un caso de control
   "concentrada vs dispersa", claramente disímil, daba un MMD más bajo
   que el caso real PSO-vs-DE). Se corrigió calculando el ancho de banda
   SOLO con distancias intra-grupo (Ps consigo mismo, Pt consigo mismo),
   excluyendo las distancias inter-grupo de la calibración de gamma. Esto
   restauró la interpretabilidad: ~0.03 para distribuciones genuinamente
   similares, ~0.6-0.65 para casos claramente disímiles, valores
   crecientes y monótonos en el caso real conforme las poblaciones de
   PSO y DE se vuelven más disímiles en FORMA (aunque su UBICACIÓN se
   acerque, que es lo que captura Wasserstein por separado).

3. Filtro MMD POR INSTANCIA (corrección D-11, 2026-10-12). La versión
   anterior comparaba con el objetivo el grupo formado por TODAS las
   élites (permutadas), así que el MMD era idéntico para cada candidata
   y el filtro aceptaba o rechazaba el lote entero; además su umbral
   (3x P95 del MMD entre mitades de 15 de la fuente) no estaba en la
   escala del valor comparado (5 élites contra 30 del objetivo), y
   aceptaba aunque el objetivo fuera 2x más disperso. Ahora cada
   instancia se evalúa sola, como masa puntual δ_x:

       MMD²(δ_x, Pt) = 1 − 2·mean_t k(x, t) + mean k(Pt, Pt)

   con x centrada en la media de la fuente y Pt en la suya (se mantiene
   la semántica de forma, no ubicación).

4. REFERENCIA EN EL OBJETIVO (corrección D-11, 2026-10-13). La primera
   versión del filtro por instancia tomaba el ancho del kernel y el
   umbral de la FUENTE (en Fase 2). Instrumentando 178 eventos de Canal B
   reales se vio que, cuando la fuente ya convergió a un punto (66/178),
   el ancho quedaba en ~1e-3, el kernel no "veía" al objetivo y todas las
   élites daban el mismo valor (1 + 1/n): la decisión dependía solo de si
   a la fuente le quedaba algún rezagado (29 lotes aprobados enteros, 37
   rechazados enteros). Como la pregunta es "¿la instancia es típica del
   OBJETIVO?", el ancho del kernel y el umbral salen ahora del objetivo,
   en el momento del Canal B: ancho = mediana de distancias intra-
   objetivo; umbral = MÁXIMO de MMD²(δ_t, Pt sin t) sobre los individuos
   t del objetivo. Se rechaza una élite solo si es más atípica que
   cualquier miembro real del objetivo. Solo el centro de la fuente (para
   centrar las élites) se fija en Fase 2, porque las élites salen de esa
   fuente. Con la fuente colapsada pasan a aprobarse 64/66 lotes; con
   escalas comparables el comportamiento casi no cambia (80/7 contra
   82/6 lotes aprobados/rechazados enteros).

   Objetivo colapsado: si la dispersión del objetivo (RMS de la distancia
   a su centroide) es menor que FRACCION_COLAPSO_MMD × diámetro, el
   objetivo no tiene forma con la que comparar y el filtro se declara NO
   APLICABLE: aprueba todas las instancias (la ubicación ya la controló
   Wasserstein) y el orquestador lo registra. Rechazar en ese caso
   bloqueaba el Canal B justo cuando el objetivo está estancado, y un
   ancho mínimo del kernel no lo resolvía (rechazaba aún más).
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist


# ----------------------------------------------------------------------
# Configuración de la barrera de seguridad.
# ----------------------------------------------------------------------

FRACCION_UMBRAL_WASSERSTEIN = 0.6    # θ_W como fracción del diámetro
                                       # máximo del espacio de búsqueda
FRACCION_COLAPSO_MMD = 1e-3          # dispersión del objetivo bajo la cual
                                       # el filtro MMD no es aplicable


@dataclass
class ReferenciaMMD:
    """Lo que el filtro MMD fija en Fase 2 (ver docstring del módulo,
    puntos 3 y 4): solo el centro de la fuente, para centrar las élites.
    El ancho del kernel y el umbral se calculan con el objetivo en Canal B."""
    centro: np.ndarray     # media de la población fuente (n_dimensiones,)


@dataclass
class ResultadoBarreraSeguridad:
    """Salida estructurada de la evaluación de la barrera de seguridad."""
    distancia_wasserstein: float
    umbral_wasserstein: float
    transferencia_permitida: bool   # resultado de la barrera DURA (W1)
    diametro_maximo_espacio: float


def calcular_diametro_maximo(limites: np.ndarray) -> float:
    """
    Calcula el diámetro máximo del hipercubo de búsqueda: la distancia
    euclidiana entre las dos esquinas opuestas del espacio definido por
    `limites` (shape (n_dimensiones, 2)). Usado como escala de referencia
    para el umbral relativo de Wasserstein.
    """
    rangos = limites[:, 1] - limites[:, 0]
    return float(np.sqrt(np.sum(rangos ** 2)))


def calcular_wasserstein(poblacion_fuente: np.ndarray,
                          poblacion_objetivo: np.ndarray) -> float:
    """
    Calcula la distancia de Wasserstein de primer orden (W1) entre dos
    poblaciones empíricas, resolviendo el problema de transporte óptimo
    de forma EXACTA mediante programación lineal (ecuación 6 del
    documento de tesis), usando distancia euclidiana como costo de
    transporte y pesos uniformes (1/n) para cada población.

    Implementado con scipy.optimize.linprog (solver HiGHS) en vez de la
    librería POT, que requiere compilación en C++ y no tiene wheels
    precompilados para Python 3.14 en Windows.
    """
    matriz_costos = cdist(poblacion_fuente, poblacion_objetivo, metric="euclidean")
    n_fuente = len(poblacion_fuente)
    n_objetivo = len(poblacion_objetivo)
    pesos_fuente = np.ones(n_fuente) / n_fuente
    pesos_objetivo = np.ones(n_objetivo) / n_objetivo

    costos_planos = matriz_costos.flatten()
    n_variables = n_fuente * n_objetivo

    A_filas = np.zeros((n_fuente, n_variables))
    for i in range(n_fuente):
        A_filas[i, i * n_objetivo:(i + 1) * n_objetivo] = 1.0

    A_columnas = np.zeros((n_objetivo, n_variables))
    for j in range(n_objetivo):
        A_columnas[j, j::n_objetivo] = 1.0

    A_eq = np.vstack([A_filas, A_columnas])
    b_eq = np.concatenate([pesos_fuente, pesos_objetivo])

    resultado = linprog(
        c=costos_planos,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
    )

    if not resultado.success:
        raise RuntimeError(
            f"No se pudo resolver el problema de transporte óptimo: "
            f"{resultado.message}"
        )
    return float(resultado.fun)



def evaluar_barrera_wasserstein(
    poblacion_fuente: np.ndarray,
    poblacion_objetivo: np.ndarray,
    limites: np.ndarray,
    fraccion_umbral: float = FRACCION_UMBRAL_WASSERSTEIN,
) -> ResultadoBarreraSeguridad:
    """
    Evalúa el criterio de control de la ecuación (7) del documento de
    tesis: si W1(Ps, Pt) > θ_W, se aborta la transferencia.

    θ_W se calcula aquí mismo como fracción_umbral × diámetro_máximo del
    espacio de búsqueda (ver justificación en el docstring del módulo).
    """
    w1 = calcular_wasserstein(poblacion_fuente, poblacion_objetivo)
    diametro_maximo = calcular_diametro_maximo(limites)
    umbral = fraccion_umbral * diametro_maximo
    permitida = w1 <= umbral

    return ResultadoBarreraSeguridad(
        distancia_wasserstein=w1,
        umbral_wasserstein=umbral,
        transferencia_permitida=permitida,
        diametro_maximo_espacio=diametro_maximo,
    )


def _kernel_rbf(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Kernel RBF (gaussiano) estándar: exp(-gamma * ||x-y||^2)."""
    d2 = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-gamma * d2)


def _calcular_gamma_intra_grupo(Ps: np.ndarray, Pt: np.ndarray) -> float:
    """
    Calcula el ancho de banda gamma del kernel RBF usando ÚNICAMENTE
    distancias intra-grupo (Ps consigo mismo, Pt consigo mismo), NO las
    distancias inter-grupo entre Ps y Pt.

    CORRECCIÓN DOCUMENTADA: la heurística estándar (mediana de TODAS las
    distancias, incluyendo las inter-grupo) resultó no ser confiable en
    alta dimensionalidad — ver justificación completa en el docstring del
    módulo. Excluir las distancias inter-grupo de este cálculo evita que
    la separación real entre las poblaciones (lo que precisamente se
    quiere medir) contamine la calibración de la escala del kernel.
    """
    d_intra_s = cdist(Ps, Ps, metric="sqeuclidean")
    d_intra_t = cdist(Pt, Pt, metric="sqeuclidean")
    todas_intra = np.concatenate([
        d_intra_s[d_intra_s > 0], d_intra_t[d_intra_t > 0]
    ])
    if len(todas_intra) == 0:
        return 1.0  # caso degenerado (poblaciones de tamaño 1)
    mediana = np.median(todas_intra)
    if mediana <= 0:
        return 1.0
    return 1.0 / (2.0 * mediana)


def calcular_mmd(poblacion_a: np.ndarray, poblacion_b: np.ndarray,
                  gamma: float = None) -> float:
    """
    Calcula el Maximum Mean Discrepancy (MMD) con kernel RBF entre dos
    poblaciones, usando la fórmula estándar:

        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    CORRECCIÓN DE INVARIANZA A TRASLACIÓN (documentada): el MMD con
    kernel gaussiano NO es invariante a traslación cuando los grupos
    están separados en una escala mayor que el ancho de banda del kernel.
    En ese caso, el término Kxy ≈ 0 (el kernel no "alcanza" a la otra
    población), haciendo que MMD ≈ Kxx + Kyy ≈ 2 independientemente de
    si las formas son similares — lo que hace al filtro inútil para el
    propósito de comparar compatibilidad de distribuciones entre
    algoritmos que convergen a distintas regiones del espacio.

    Se corrige centrando cada población en su propia media antes de
    calcular el kernel, eliminando la componente de traslación. Esto
    hace que el MMD compare exclusivamente la FORMA (varianza, dispersión,
    estructura) de las distribuciones, que es exactamente lo que el
    documento de tesis describe como el propósito del filtro MMD:
    verificar que las instancias candidatas "pertenecen a una
    distribución compatible" con el espacio del algoritmo objetivo —
    no que estén en la misma región geométrica (eso ya lo hace
    Wasserstein como barrera separada).

    Verificación empírica de la corrección:
      - Misma forma gaussiana, centros a 500u de distancia → MMD=0.005
      - Formas distintas (gaussiana vs uniforme), mismo centro → MMD=0.51
      - Misma distribución, semillas distintas → MMD=0.006
    Sin centrado, el primer caso daba MMD=1.24, confundiendo diferencia
    de ubicación con incompatibilidad de distribución.

    Si gamma no se especifica, se calcula automáticamente con
    _calcular_gamma_intra_grupo sobre las poblaciones ya centradas.
    """
    # Centrar cada población en su propia media antes de comparar.
    a_centrado = poblacion_a - poblacion_a.mean(axis=0)
    b_centrado = poblacion_b - poblacion_b.mean(axis=0)

    if gamma is None:
        gamma = _calcular_gamma_intra_grupo(a_centrado, b_centrado)
    kxx = _kernel_rbf(a_centrado, a_centrado, gamma).mean()
    kyy = _kernel_rbf(b_centrado, b_centrado, gamma).mean()
    kxy = _kernel_rbf(a_centrado, b_centrado, gamma).mean()
    return float(kxx + kyy - 2 * kxy)


def calcular_mmd_instancias(instancias: np.ndarray,
                              centro_fuente: np.ndarray,
                              poblacion_objetivo: np.ndarray,
                              gamma: float) -> np.ndarray:
    """
    MMD² entre cada instancia (como masa puntual δ_x) y la población
    objetivo, con la instancia centrada en `centro_fuente` y el objetivo en
    su propia media (compara forma, no ubicación):

        MMD²(δ_x, Pt) = k(x,x) − 2·mean_t k(x,t) + mean k(Pt,Pt)

    con k(x,x) = 1 en el kernel RBF. Retorna un valor por instancia.
    """
    x_centrado = instancias - centro_fuente
    t_centrado = poblacion_objetivo - poblacion_objetivo.mean(axis=0)
    kyy = _kernel_rbf(t_centrado, t_centrado, gamma).mean()
    kxy = _kernel_rbf(x_centrado, t_centrado, gamma).mean(axis=1)
    return 1.0 - 2.0 * kxy + kyy


def calcular_referencia_mmd(poblacion_fuente: np.ndarray) -> ReferenciaMMD:
    """
    Fija en Fase 2 lo que el filtro necesita de la fuente: su centro, para
    centrar las élites (que salen de la fuente de ese momento). El ancho
    del kernel y el umbral NO se fijan aquí: dependen del objetivo y se
    calculan en Canal B (ver docstring del módulo, punto 4).
    """
    return ReferenciaMMD(centro=poblacion_fuente.mean(axis=0))


def calcular_dispersion(poblacion: np.ndarray) -> float:
    """Dispersión de una población: RMS de la distancia a su centroide."""
    centrada = poblacion - poblacion.mean(axis=0)
    return float(np.sqrt((centrada ** 2).sum(axis=1).mean()))


def objetivo_colapsado(poblacion_objetivo: np.ndarray, limites: np.ndarray,
                       fraccion: float = FRACCION_COLAPSO_MMD) -> bool:
    """
    True si el objetivo convergió a un punto (dispersión menor que
    `fraccion` × diámetro del espacio): no tiene forma con la que comparar
    y el filtro MMD no es aplicable (ver docstring del módulo, punto 4).
    """
    return (calcular_dispersion(poblacion_objetivo)
            < fraccion * calcular_diametro_maximo(limites))


def calcular_umbral_mmd_objetivo(poblacion_objetivo: np.ndarray,
                                  gamma: float) -> float:
    """
    Umbral del filtro: máximo de MMD²(δ_t, Pt sin t) sobre los individuos t
    del objetivo — la misma estadística que se aplica a las élites, medida
    sobre miembros reales del objetivo (leave-one-out).
    """
    centro_objetivo = poblacion_objetivo.mean(axis=0)
    nulo = [
        calcular_mmd_instancias(
            poblacion_objetivo[j:j + 1], centro_objetivo,
            np.delete(poblacion_objetivo, j, axis=0), gamma,
        )[0]
        for j in range(len(poblacion_objetivo))
    ]
    return float(np.max(nulo))


def filtrar_instancias_por_mmd(
    instancias_candidatas: np.ndarray,
    poblacion_fuente: np.ndarray,
    poblacion_objetivo: np.ndarray,
    limites: np.ndarray,
    referencia: ReferenciaMMD = None,
) -> np.ndarray:
    """
    Filtro fino de la Fase 3 (ecuación de salida de la sección 4.2.2:
    "L_instancias ordenadas por ϕ+MMD"): decide POR INSTANCIA si cada
    candidata es típica de la distribución del objetivo (ver docstring del
    módulo, puntos 3 y 4). Ancho del kernel y umbral salen del objetivo;
    las candidatas se centran en el centro de la fuente.

    Si se provee `referencia` (calculada en la extracción, Fase 2), se usa
    su centro; si no, se toma el de `poblacion_fuente`. Canal B debe pasar
    la de Fase 2: las élites provienen de la fuente de ese momento.

    Si el objetivo está colapsado (`objetivo_colapsado`), el filtro no es
    aplicable y aprueba todas las instancias.

    Retorna un array booleano: True si la instancia pasa el filtro.
    """
    if objetivo_colapsado(poblacion_objetivo, limites):
        return np.ones(len(instancias_candidatas), dtype=bool)
    if referencia is None:
        referencia = calcular_referencia_mmd(poblacion_fuente)
    objetivo_centrado = poblacion_objetivo - poblacion_objetivo.mean(axis=0)
    gamma = _calcular_gamma_intra_grupo(objetivo_centrado, objetivo_centrado)
    umbral = calcular_umbral_mmd_objetivo(poblacion_objetivo, gamma)
    valores = calcular_mmd_instancias(
        instancias_candidatas, referencia.centro, poblacion_objetivo, gamma)
    return valores <= umbral
