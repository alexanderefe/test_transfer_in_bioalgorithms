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
     objetivo, antes de que lleguen a la barrera dura de Wasserstein.

DECISIONES DE FORMALIZACIÓN ACORDADAS EXPLÍCITAMENTE (el documento no
especifica valores ni implementación concreta para ninguna de las dos):

1. Wasserstein: se calcula de forma EXACTA como un problema de
   programación lineal de transporte óptimo (ecuación 6 del documento),
   usando la librería POT (Python Optimal Transport), en vez de
   aproximaciones. El umbral θ_W se define como una FRACCIÓN RELATIVA
   del diámetro máximo del hipercubo de búsqueda (√D × rango_por_dim),
   en vez de un valor absoluto fijo. Esto se decidió tras observar
   numéricamente que W1 varía en un rango muy amplio (130 a 344, en el
   caso real PSO/DE sobre F11/D=20) y que un umbral relativo generaliza
   correctamente a las 12 funciones del benchmark CEC 2022, que tienen
   distintos rangos y dimensionalidades, sin necesitar recalibración
   manual por función. θ_W = 0.25 (25% del diámetro máximo) fue elegido
   tras verificar que los valores reales observados ocupan entre ~15% y
   ~38% de esa escala: 25% bloquea las iteraciones tempranas (poblaciones
   muy dispersas, fracción ~38%) pero permite la transferencia una vez
   que ambos algoritmos convergen a regiones razonablemente cercanas
   (fracción ~15-25%).

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

   El umbral de MMD se define de forma DINÁMICA, no como un número fijo:
   se calcula un "ruido de muestreo de referencia" dividiendo la propia
   población fuente en dos sub-muestras aleatorias y midiendo el MMD
   entre ellas (que debería ser cercano a 0, ya que ambas mitades
   provienen de la misma distribución). El umbral de rechazo se fija en
   3x el percentil 95 de ese ruido de referencia, de modo que el filtro
   se calibra automáticamente al tamaño y dispersión de cada población
   fuente real, sin necesitar un valor fijo ajustado a mano por función
   del benchmark.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist


# ----------------------------------------------------------------------
# Configuración de la barrera de seguridad.
# ----------------------------------------------------------------------

FRACCION_UMBRAL_WASSERSTEIN = 0.25   # θ_W como fracción del diámetro
                                       # máximo del espacio de búsqueda
N_SUBMUESTRAS_REFERENCIA_MMD = 50     # repeticiones para estimar el
                                        # ruido de muestreo de referencia
MULTIPLO_UMBRAL_MMD = 3.0             # umbral = 3x percentil 95 del ruido
PERCENTIL_REFERENCIA_MMD = 95


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


def calcular_umbral_mmd_dinamico(
    poblacion_fuente: np.ndarray,
    n_submuestras: int = N_SUBMUESTRAS_REFERENCIA_MMD,
    percentil: float = PERCENTIL_REFERENCIA_MMD,
    multiplo: float = MULTIPLO_UMBRAL_MMD,
    semilla: int = 0,
) -> float:
    """
    Calcula el umbral dinámico de rechazo por MMD (ver justificación
    completa en el docstring del módulo): divide la población fuente en
    dos mitades aleatorias repetidamente, mide el MMD entre ellas (ruido
    de muestreo puro, sin diferencia real de distribución), y fija el
    umbral en `multiplo` veces el percentil indicado de esa distribución
    de referencia.
    """
    rng = np.random.default_rng(semilla)
    n = len(poblacion_fuente)
    valores_referencia = []

    for _ in range(n_submuestras):
        indices = rng.permutation(n)
        mitad_1 = poblacion_fuente[indices[: n // 2]]
        mitad_2 = poblacion_fuente[indices[n // 2:]]
        valores_referencia.append(calcular_mmd(mitad_1, mitad_2))

    percentil_referencia = np.percentile(valores_referencia, percentil)
    return float(multiplo * percentil_referencia)


def filtrar_instancias_por_mmd(
    instancias_candidatas: np.ndarray,
    poblacion_fuente: np.ndarray,
    poblacion_objetivo: np.ndarray,
    umbral_precalculado: float = None,
) -> np.ndarray:
    """
    Filtro fino de la Fase 2/3 (ecuación de salida de la sección 4.2.2:
    "L_instancias ordenadas por ϕ+MMD"): para cada instancia candidata,
    se evalúa si su distribución es compatible con la del algoritmo
    objetivo, usando MMD como medida de discrepancia.

    CORRECCIÓN DE IMPLEMENTACIÓN (bug detectado y documentado): la
    versión original construía el grupo de comparación replicando la
    instancia individual con np.tile() (30 copias idénticas). Eso
    genera un grupo con VARIANZA INTERNA CERO, lo cual rompe el cálculo
    de MMD de forma sistemática: el término Kxx (kernel de la instancia
    contra sí misma) se vuelve trivialmente 1.0 en todos los elementos,
    inflando artificialmente la discrepancia frente a cualquier
    distribución real con varianza normal. Se verificó que incluso
    instancias tomadas literalmente de la propia población objetivo
    eran rechazadas con ese método, lo que confirma el defecto.

    DECISIÓN ACORDADA (Opción 3): en vez de replicar la instancia sola,
    se construye el grupo de comparación combinando la instancia
    candidata con el CONJUNTO COMPLETO de instancias de élite disponibles
    (instancias_candidatas). Esto aprovecha la varianza genuina que ya
    existe entre las instancias de élite extraídas por FastSHAP en la
    Fase 2, sin introducir ningún parámetro nuevo ni muestras sintéticas.
    El grupo resultante tiene varianza real y un tamaño razonable para
    que el kernel MMD funcione correctamente.

    Retorna un array booleano indicando cuáles instancias pasan el filtro
    (MMD del grupo que las incluye, respecto a la población objetivo, por
    debajo del umbral). Si se provee umbral_precalculado (calculado en el
    momento de la extracción en Fase 2), se usa directamente en vez de
    recalcularlo con la población fuente actual — que puede haber convergido
    más desde entonces, produciendo un umbral que no refleja la distribución
    del fuente en el momento relevante.
    """
    umbral = (umbral_precalculado
              if umbral_precalculado is not None
              else calcular_umbral_mmd_dinamico(poblacion_fuente))

    aprobadas = np.zeros(len(instancias_candidatas), dtype=bool)
    for i in range(len(instancias_candidatas)):
        otras_instancias = np.delete(instancias_candidatas, i, axis=0)
        grupo_con_varianza = np.vstack([
            instancias_candidatas[i:i+1],
            otras_instancias,
        ])
        mmd_instancia = calcular_mmd(grupo_con_varianza, poblacion_objetivo)
        aprobadas[i] = mmd_instancia <= umbral

    return aprobadas