"""
Fase 1 — Detección (middleware/deteccion.py)

Implementa el Score compuesto S definido en la ecuación (4) del documento
de tesis:

    S = W_FIR · FIR + W_DWD · DWD + W_HDF · HDF

con sus tres componentes:

- FIR  (Fitness Improvement Rate)
- DWD  (Dimension-Wise Diversity)      — ecuación (2)
- HDF  (Hamming Diversity by Frequency) — ecuación (3)

DECISIONES DE FORMALIZACIÓN (acordadas explícitamente antes de implementar,
porque el documento las deja ambiguas o abiertas a sintonización empírica):

1. FIR: el documento describe este indicador en prosa de forma ambigua
   ("dividiendo el valor de la función objetivo del mejor individuo... 
   frente a su histórico"). Una razón directa f_actual/f_pasado no es
   consistente con el rango [0,1] que sí tienen DWD y HDF, ni con la
   semántica deseada (FIR ≈ 0 si no hay progreso, FIR ≈ 1 si el progreso
   es sustancial). Se formaliza como mejora relativa normalizada con
   saturación exponencial:

       FIR_t = 1 - exp(-λ · max(0, f_best(t-w) - f_best(t)) / (|f_best(t-w)| + ε))

   Ventana w = 20 iteraciones (consistente con el diagnóstico usado en
   Fase 0). λ controla la sensibilidad de saturación.

2. HDF: está diseñado en el documento para codificaciones binarias o
   discretas. PSO y DE operan en el dominio continuo del benchmark CEC
   2022, por lo que HDF no tiene datos sobre los que operar en este
   contexto. Se deja implementado como placeholder inactivo (retorna
   None) y su peso correspondiente se fija en W_HDF = 0.0, manteniéndolo
   explícito en la fórmula del Score por fidelidad a la ecuación (4).
   Cuando exista un caso de uso con codificación discreta/binaria, se
   implementa la lógica real aquí mismo.

3. Pesos: el documento indica sintonización empírica. Para esta primera
   versión, se prioriza la diversidad poblacional (DWD) sobre la mejora
   de fitness (FIR), bajo el argumento de que la pérdida de diversidad
   es la señal más temprana y confiable de convergencia prematura, antes
   de que el estancamiento se refleje en el fitness:

       W_FIR = 0.3 | W_DWD = 0.7 | W_HDF = 0.0 (inactivo)
"""

from dataclasses import dataclass
import numpy as np


# ----------------------------------------------------------------------
# Constantes de configuración del Score compuesto (Fase 1).
# Centralizadas aquí para facilitar la sintonización empírica futura
# sin tener que modificar la lógica de cálculo.
# ----------------------------------------------------------------------

VENTANA_FIR = 20          # iteraciones hacia atrás para medir mejora de fitness
LAMBDA_FIR = 3.0          # sensibilidad de saturación exponencial de FIR.
                          # Calibrado para que mejoras >=50% den FIR>=0.78 y
                          # mejoras >=90% den FIR>=0.93 (con lambda=1.0 una
                          # mejora del 99% solo alcanzaba FIR=0.63, lo cual
                          # sesgaba el Score hacia abajo incluso con progreso
                          # excelente; ver validación numérica registrada).
EPSILON = 1e-12           # evita división por cero

PESO_FIR = 0.3
PESO_DWD = 0.7
PESO_HDF = 0.0            # inactivo: dominio continuo (CEC 2022)

UMBRAL_ESTANCAMIENTO = 0.2          # S < 0.2 => estado de alto estancamiento
FRACCION_ACTIVACION = 0.15          # Score se computa solo tras 15% de iteraciones


@dataclass
class ResultadoDeteccion:
    """Salida estructurada de una evaluación del Score compuesto."""
    iteracion: int
    fir: float
    dwd: float
    hdf: float | None     # None mientras el indicador esté inactivo
    score: float
    estancado: bool
    activo: bool           # False si aún no se alcanzó el 15% de iteraciones


def calcular_fir(fitness_pasado: float, fitness_actual: float,
                  lambda_sat: float = LAMBDA_FIR) -> float:
    """
    Fitness Improvement Rate, formalizado como mejora relativa normalizada
    con saturación exponencial. Ver justificación en el docstring del módulo.

    Retorna 0.0 si no hubo mejora o si el fitness empeoró (truncado, ya que
    el Score no debe premiar retrocesos, solo penalizar la falta de avance).
    Retorna valores en (0, 1), acercándose a 1 cuanto mayor sea la mejora
    relativa respecto a la magnitud del fitness pasado.
    """
    mejora = fitness_pasado - fitness_actual  # minimización: mejora si es positivo
    mejora = max(mejora, 0.0)
    denominador = abs(fitness_pasado) + EPSILON
    return float(1.0 - np.exp(-lambda_sat * mejora / denominador))


def calcular_dwd(poblacion: np.ndarray) -> float:
    """
    Dimension-Wise Diversity, ecuación (2) del documento de tesis:

        DWD = (1 / (m·n)) · Σ_i Σ_j |median(x_j) - x_j_i|

    donde n = cantidad de individuos, m = cantidad de dimensiones,
    median(x_j) = mediana de la dimensión j sobre toda la población,
    x_j_i = valor de la dimensión j para el individuo i.

    A diferencia de la versión simplificada usada en Fase 0 (que
    normalizaba por el rango poblacional para fines de diagnóstico
    exploratorio), esta es la fórmula EXACTA del documento: una distancia
    absoluta promedio, SIN normalizar a [0,1] por rango.

    Nota importante: el documento describe el resultado como normalizado
    en [0,1], pero la fórmula tal como está escrita (ecuación 2) no
    normaliza por ninguna escala — es una distancia absoluta promedio,
    cuya magnitud depende directamente de la escala del espacio de
    búsqueda (en CEC 2022, [-100, 100]). Para que el resultado sea
    comparable entre funciones y consistente con el rango [0,1] que
    exige la ecuación (4) del Score, se normaliza aquí dividiendo por el
    máximo recorrido posible del espacio de búsqueda (rango de los
    límites), preservando la fórmula original como el numerador.
    """
    n_individuos, n_dimensiones = poblacion.shape
    mediana_por_dimension = np.median(poblacion, axis=0)  # shape (n_dimensiones,)
    distancia_absoluta = np.abs(poblacion - mediana_por_dimension)  # (n, m)
    dwd_bruto = float(np.sum(distancia_absoluta) / (n_individuos * n_dimensiones))
    return dwd_bruto


def normalizar_dwd(dwd_bruto: float, limites: np.ndarray) -> float:
    """
    Normaliza el DWD bruto (distancia absoluta promedio) al rango [0,1],
    usando como escala el rango máximo del espacio de búsqueda. Esto es
    necesario porque la ecuación (2) tal como está escrita no normaliza,
    pero la ecuación (4) del Score exige que cada componente esté en [0,1].
    """
    rango_promedio = float(np.mean(limites[:, 1] - limites[:, 0]))
    if rango_promedio <= 0:
        return 0.0
    # División por 4 como aproximación de la dispersión máxima esperada
    # de |x - mediana| respecto al rango total (la mitad del rango es el
    # máximo posible, por lo que el promedio esperado bajo dispersión
    # uniforme máxima es del orden de rango/4).
    return float(np.clip(dwd_bruto / (rango_promedio / 4.0), 0.0, 1.0))


def calcular_hdf(poblacion: np.ndarray) -> float | None:
    """
    Diversidad de Hamming por Frecuencia, ecuación (3) del documento.

    INACTIVO en esta implementación: está diseñado para codificaciones
    binarias/discretas. PSO y DE operan en R^m continuo sobre el
    benchmark CEC 2022, por lo que no hay frecuencias de valores binarios
    que calcular. Se retorna None explícitamente para distinguir este
    caso de un valor 0.0 real (que significaría "ausencia total de
    diversidad binaria", una afirmación que no aplica aquí).
    """
    return None


def fuente_en_condiciones_optimas(fitness_fuente: float, fitness_objetivo: float) -> bool:
    """
    Determina si el algoritmo fuente "funciona en condiciones óptimas",
    condición exigida por la Fase 2 del documento de tesis (sección 4.2.2)
    para activar la extracción de conocimiento, junto con
    S_objetivo < UMBRAL_ESTANCAMIENTO.

    VACÍO DE ESPECIFICACIÓN DEL DOCUMENTO: el texto no define
    matemáticamente qué significa "condiciones óptimas". Se descubrió
    durante la validación de Fase 1 que el Score compuesto S por sí solo
    NO sirve como ese criterio: un algoritmo que converge exitosamente
    hacia el óptimo también pierde diversidad poblacional y cae bajo el
    umbral de estancamiento (S < 0.2), igual que un algoritmo realmente
    estancado. El propio documento reconoce esta ambigüedad en la
    sección 2.10 ("baja diversidad... puede ser indicio de estancamiento
    O convergencia prematura"), pero no la resuelve en la fórmula del
    Score.

    DECISIÓN ADOPTADA (acordada explícitamente, no inferida del texto):
    el fuente se considera en condiciones óptimas si su fitness actual es
    estrictamente mejor que el del algoritmo objetivo en el mismo
    instante. Se prefirió esta opción (comparativa, agnóstica) sobre dos
    alternativas consideradas:
      - Comparar contra el óptimo global conocido del benchmark: se
        descartó porque no siempre existe un óptimo conocido en
        problemas reales, contradiciendo el agnosticismo al algoritmo
        y al problema que el documento declara como pilar de la
        propuesta.
      - Combinar S_fuente alto Y fitness mejor que el objetivo: se
        descartó por redundante una vez aceptado que S no distingue
        convergencia de estancamiento; agregar S como condición extra
        no aporta información adicional confiable.

    Nota: esta función NO usa el Score S en absoluto. Es deliberado.
    """
    return fitness_fuente < fitness_objetivo  # minimización


def asignar_roles_dinamicos(fitness_algoritmo_a: float, fitness_algoritmo_b: float) -> tuple[str, str]:
    """
    Asigna dinámicamente los roles "fuente" y "objetivo" entre dos
    algoritmos en ejecución paralela, según cuál tenga mejor fitness en
    el instante actual.

    DECISIÓN DE DISEÑO (acordada explícitamente): los roles fuente/
    objetivo NO se fijan de antemano por identidad del algoritmo (p.ej.
    "PSO siempre es fuente, DE siempre es objetivo"). Esto se decidió
    tras observar, durante la integración de Fase 1 sobre datos reales,
    que en la función F11/D=20 (semilla=42) DE terminó con mejor fitness
    que PSO desde la iteración 60 en adelante — es decir, el algoritmo
    "destinado" a ser objetivo resultó ser el de mejor desempeño real en
    ese caso. Fijar los roles de antemano habría sido inconsistente con
    el propio objetivo general del documento de tesis: "tomar dos
    algoritmos, identificar cuál es óptimo y transferir su conocimiento
    al de menor rendimiento" — lo cual ya implica una asignación
    dinámica, no estática, de roles.

    Retorna una tupla ("fuente", "objetivo") con las etiquetas 'a' o 'b'
    indicando qué algoritmo (de los dos parámetros recibidos, en ese
    orden) ocupa cada rol en este instante.

    Caso de empate exacto (fitness_a == fitness_b): se asigna 'b' como
    fuente por defecto (rama else). Es una decisión arbitraria menor,
    poco relevante en la práctica dado que el fitness es de punto
    flotante y un empate exacto es estadísticamente improbable, pero se
    deja documentado para que no quede como comportamiento implícito.
    """
    if fitness_algoritmo_a < fitness_algoritmo_b:  # minimización
        return ("a", "b")  # a es fuente, b es objetivo
    return ("b", "a")  # b es fuente, a es objetivo



def calcular_score(
    historial_fitness: list[float],
    poblacion_actual: np.ndarray,
    limites: np.ndarray,
    iteracion_actual: int,
    max_iteraciones: int,
    ventana: int = VENTANA_FIR,
    peso_fir: float = PESO_FIR,
    peso_dwd: float = PESO_DWD,
    peso_hdf: float = PESO_HDF,
) -> ResultadoDeteccion:
    """
    Calcula el Score compuesto S según la ecuación (4), integrando FIR,
    DWD (normalizado) y HDF (inactivo). Aplica la regla de activación:
    el Score solo se computa de forma "activa" tras el 15% de las
    iteraciones totales (estabilidad estocástica inicial, según el
    documento). Antes de eso, se retorna igualmente el cálculo pero
    marcado como `activo=False`, para que el middleware pueda decidir
    explícitamente ignorarlo en ese período.
    """
    activo = iteracion_actual >= FRACCION_ACTIVACION * max_iteraciones

    # FIR requiere comparar contra `ventana` iteraciones atrás. Si el
    # historial todavía no tiene suficiente profundidad, no hay mejora
    # que medir todavía: se asume FIR=0 (conservador: sin evidencia de
    # progreso, no se premia).
    if len(historial_fitness) > ventana:
        fitness_pasado = historial_fitness[-ventana - 1]
        fitness_actual = historial_fitness[-1]
        fir = calcular_fir(fitness_pasado, fitness_actual)
    else:
        fir = 0.0

    dwd_bruto = calcular_dwd(poblacion_actual)
    dwd = normalizar_dwd(dwd_bruto, limites)

    hdf = calcular_hdf(poblacion_actual)
    hdf_para_score = hdf if hdf is not None else 0.0

    score = peso_fir * fir + peso_dwd * dwd + peso_hdf * hdf_para_score
    score = float(np.clip(score, 0.0, 1.0))

    estancado = activo and (score < UMBRAL_ESTANCAMIENTO)

    return ResultadoDeteccion(
        iteracion=iteracion_actual,
        fir=fir,
        dwd=dwd,
        hdf=hdf,
        score=score,
        estancado=estancado,
        activo=activo,
    )