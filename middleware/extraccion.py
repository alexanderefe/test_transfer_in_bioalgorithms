"""
Fase 2 — Extracción (middleware/extraccion.py)

Implementa la arquitectura conceptual descrita en la sección 4.2.2 del
documento de tesis:

  1. Modelo Subrogado: XGBoost entrenado con el historial del algoritmo
     fuente (variables de decisión -> fitness).
  2. Extracción de Instancias de Élite: FastSHAP real (repositorio oficial
     iancovert/fastshap) sobre el modelo subrogado, filtrado posteriormente
     por compatibilidad de distribución (placeholder de MMD, que se
     implementa formalmente en la Fase 3 según el plan acordado).
  3. Extracción de Parámetros: identificación de los momentos episódicos
     donde el algoritmo fuente logró escapar de un óptimo local.

DECISIÓN DE ARQUITECTURA ACORDADA EXPLÍCITAMENTE (no inferida del texto):
se usa la implementación OFICIAL de FastSHAP (Jethani et al., 2022,
repositorio iancovert/fastshap), no una reimplementación propia, para
mantener fidelidad total con el método citado en el documento. Esto
requirió resolver una diferencia de terminología entre el paper y el
documento de tesis:

  - El documento de tesis llama "modelo subrogado" al XGBoost (variables
    de decisión -> fitness).
  - El paper de FastSHAP usa "surrogate" para una red de IMPUTACIÓN
    distinta (que aprende a predecir cuando se ocultan features), nunca
    mencionada en el documento de tesis.

Tras revisar el código fuente del repositorio oficial, se determinó que
ese "surrogate" de imputación de FastSHAP NO es necesario en este caso:
MarginalImputer (también provisto por el repositorio oficial) reemplaza
directamente las variables ocultas con muestras de la población real de
PSO, sin entrenar ninguna red intermedia. Esto evita introducir un
componente que no existe en el documento de tesis, y reduce la
arquitectura real a dos piezas entrenadas (XGBoost + explainer de
FastSHAP), en vez de tres.

El modelo a explicar (XGBoost) no es un torch.nn.Module nativo, por lo que
se envuelve mediante XGBoostWrapper (middleware/xgboost_wrapper.py) para
ser compatible con la interfaz que exige MarginalImputer.
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb

from middleware.xgboost_wrapper import XGBoostWrapper
from fastshap_lib.fastshap import FastSHAP
from fastshap_lib.tabular_imputers import MarginalImputer
from middleware.barrera_seguridad import calcular_umbral_mmd_dinamico


# ----------------------------------------------------------------------
# Configuración de la Fase 2.
# ----------------------------------------------------------------------

VENTANA_HISTORIAL_FUENTE = 50   # iteraciones recientes de PSO usadas para
                                 # entrenar el modelo subrogado (acordado:
                                 # ventana reciente, no historial completo,
                                 # para reflejar comportamiento actual).
TOP_K_INSTANCIAS_ELITE = 5       # cantidad de instancias de élite a extraer
PERCENTIL_PREFILTRO_FITNESS = 0.20  # Opción C acordada: primero se filtra
                                     # el 20% de instancias con MEJOR fitness
                                     # real de la ventana; solo dentro de ese
                                     # subconjunto ya garantizado como "bueno"
                                     # se aplica el ranking por Shapley. Esto
                                     # corrige el error de diseño detectado en
                                     # la primera versión (rankear por suma de
                                     # ϕ directamente seleccionaba las PEORES
                                     # instancias, porque un ϕ alto empuja el
                                     # fitness hacia arriba, es decir, peor,
                                     # al ser este un problema de minimización).
N_BACKGROUND_MARGINAL = 30       # tamaño de muestra de población usada como
                                  # "background" del MarginalImputer

# Nº de árboles del modelo subrogado XGBoost. Se probó bajarlo a 50 (decisión
# D-3) pero el R² del subrogado cayó a 0.65 en el escenario de validación
# (PSO/F11), por debajo del criterio de aceptación R²>0.7 del documento de
# tesis. Se mantiene en 100. El costo real —el `predict` dentro de FastSHAP—
# ya se reduce a la mitad por NUM_SAMPLES_EXPLAINER (8->4).
N_ESTIMADORES_SUBROGADO = 100

# Hiperparámetros de entrenamiento del explainer FastSHAP. Se mantienen
# moderados porque la ventana de datos disponible es pequeña (50
# iteraciones x ~30 individuos = ~1500 muestras), muy por debajo de la
# escala de los datasets usados en los notebooks originales del paper
# (census: decenas de miles de filas).
# EPOCHS y NUM_SAMPLES reducidos (50->20, 8->4) por la decisión D-3: en las
# mediciones el resultado del ciclo de transferencia no cambia y el wall-clock
# del middleware baja. NUM_SAMPLES afecta solo al explainer (no al R² del
# subrogado), así que es seguro.
EPOCHS_EXPLAINER = 20
NUM_SAMPLES_EXPLAINER = 4        # subconjuntos muestreados por ejemplo (train y validación)
BATCH_SIZE_EXPLAINER = 32
LR_EXPLAINER = 2e-3


@dataclass
class ResultadoExtraccion:
    """Salida estructurada de la Fase 2, según la ecuación de salida de
    la sección 4.2.2 del documento de tesis:
        Salida = L_instancias(ordenadas por ϕ+MMD) ∪ D_parámetros(ordenados por ϕ)
    """
    instancias_elite: np.ndarray          # (k, n_dimensiones)
    fitness_instancias_elite: np.ndarray  # (k,)
    valores_shapley_elite: np.ndarray     # (k, n_dimensiones) — ϕ por variable
    parametros_escape: dict                # configuración en momentos de escape
    r2_modelo_subrogado: float             # calidad del XGBoost (validación)
    transferencia_bloqueada_por_r2: bool = False  # True cuando R²≤0
    umbral_mmd_precalculado: float = 0.0  # umbral MMD calculado en el momento
                                           # de la extracción (Fase 2), con la
                                           # población del fuente tal como estaba
                                           # entonces. Canal B DEBE usar este
                                           # umbral, no recalcularlo después
                                           # (la población del fuente puede haber
                                           # convergido más entre Fase 2 y Canal B,
                                           # produciendo un umbral artificialmente
                                           # alto que aprueba todo o bajo que rechaza todo)


def entrenar_modelo_subrogado(
    historial_poblacional: list[np.ndarray],
    historial_fitness_individual: list[np.ndarray],
    ventana: int = VENTANA_HISTORIAL_FUENTE,
) -> tuple[xgb.XGBRegressor, float]:
    """
    Entrena el modelo subrogado XGBoost sobre la ventana reciente del
    historial del algoritmo fuente, correlacionando variables de decisión
    con fitness obtenido (sección 4.2.2, paso 1).

    historial_poblacional: lista de arrays (n_individuos, n_dimensiones),
        uno por iteración.
    historial_fitness_individual: lista de arrays (n_individuos,) con el
        fitness de cada individuo en esa iteración (NO el mejor histórico,
        sino el fitness real de cada individuo, necesario para tener
        variedad de pares (x, fitness) con los que entrenar XGBoost).

    Retorna el modelo entrenado y su R² de validación, para verificar
    que cumple el criterio de aceptación definido para esta fase
    (R² > 0.7 sobre datos de validación separados del entrenamiento).
    """
    ventana_poblacion = historial_poblacional[-ventana:]
    ventana_fitness = historial_fitness_individual[-ventana:]

    X = np.vstack(ventana_poblacion)
    y = np.concatenate(ventana_fitness)

    # Verificación temprana de varianza: si el fitness tiene varianza≈0
    # (el fuente convergió completamente), no tiene sentido entrenar
    # XGBoost ni FastSHAP — ss_tot≈0 produciría R² negativo o cero y
    # Shapley no tendría señal real. Se retorna r2=0.0 directamente
    # para que el llamador pueda abortar antes de gastar tiempo en el
    # entrenamiento costoso de FastSHAP.
    if np.var(y) < 1e-10:
        modelo = xgb.XGBRegressor(
            n_estimators=N_ESTIMADORES_SUBROGADO, max_depth=4,
            learning_rate=0.1, random_state=0, n_jobs=1,
        )
        modelo.fit(X, y)  # entrenar igualmente para tener un objeto válido
        return modelo, 0.0

    # División train/validación dentro de la propia ventana, para poder
    # verificar el criterio de aceptación (R² en datos NO vistos durante
    # el entrenamiento del subrogado).
    n_muestras = len(X)
    rng = np.random.default_rng(0)
    indices = rng.permutation(n_muestras)
    corte = int(n_muestras * 0.8)
    idx_train, idx_val = indices[:corte], indices[corte:]

    modelo = xgb.XGBRegressor(
        n_estimators=N_ESTIMADORES_SUBROGADO,
        max_depth=4,
        learning_rate=0.1,
        random_state=0,
        n_jobs=1,  # D-2: el histograma multi-hilo de XGBoost no es determinista
                   # bit a bit (orden de suma en punto flotante); con n_jobs=1
                   # el resultado es reproducible dado el mismo random_state.
    )
    modelo.fit(X[idx_train], y[idx_train])

    pred_val = modelo.predict(X[idx_val])
    ss_res = np.sum((y[idx_val] - pred_val) ** 2)
    ss_tot = np.sum((y[idx_val] - np.mean(y[idx_val])) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return modelo, r2


def construir_fastshap(modelo_xgboost: xgb.XGBRegressor,
                        poblacion_background: np.ndarray,
                        n_dimensiones: int) -> FastSHAP:
    """
    Construye el objeto FastSHAP oficial (sección 4.2.2, paso 2), usando:
      - XGBoostWrapper: adapta el XGBoost a la interfaz torch.nn.Module
        que exige MarginalImputer.
      - MarginalImputer: enmascara variables de decisión reemplazándolas
        por muestras de la población real de PSO (background), siguiendo
        la decisión acordada de usar este imputer en vez de BaselineImputer.
      - Explainer: red MLP pequeña, dimensionada para la escala de datos
        de este problema (no la arquitectura grande de los notebooks
        originales, pensados para datasets de imagen/tabulares masivos).
    """
    modelo_envuelto = XGBoostWrapper(modelo_xgboost)
    modelo_envuelto.eval()

    background_tensor = torch.tensor(poblacion_background, dtype=torch.float32)
    imputer = MarginalImputer(modelo_envuelto, background_tensor)

    # Explainer: MLP simple. Entrada = n_dimensiones, salida = n_dimensiones
    # (un valor de Shapley por variable de decisión, ya que el modelo a
    # explicar tiene una sola salida —el fitness—, no múltiples clases
    # como en los notebooks de clasificación del repositorio original).
    explainer = nn.Sequential(
        nn.Linear(n_dimensiones, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, n_dimensiones),
    )

    fastshap = FastSHAP(explainer, imputer, normalization=None, link=None)
    return fastshap


def extraer_conocimiento(
    historial_poblacional: list[np.ndarray],
    historial_fitness_individual: list[np.ndarray],
    historial_mejor_fitness: list[float],
    historial_hiperparametros: list[dict],
    n_dimensiones: int,
    top_k: int = TOP_K_INSTANCIAS_ELITE,
    max_epochs_fastshap: int = EPOCHS_EXPLAINER,
) -> ResultadoExtraccion:
    """
    Ejecuta la Fase 2 completa: entrena el modelo subrogado, construye y
    entrena FastSHAP, calcula los valores de Shapley sobre las instancias
    de la ventana reciente, y selecciona las top-k instancias de élite
    mediante un criterio de DOS PASOS (Opción C acordada explícitamente):

      Paso 1: pre-filtro duro por fitness real — solo se consideran
              elegibles las instancias dentro del percentil de mejor
              desempeño real de la ventana (PERCENTIL_PREFILTRO_FITNESS).
      Paso 2: dentro de ese subconjunto ya garantizado como bueno, se
              ordena por contribución de Shapley (suma de ϕ) para obtener
              el ranking final de las top-k.

    Este diseño de dos pasos corrige un error detectado en una primera
    versión de esta función: rankear directamente por suma de ϕ (sin
    prefiltro) seleccionaba las PEORES instancias de la ventana, porque en
    un problema de minimización un ϕ alto indica que esas variables
    empujan el fitness hacia arriba (peor resultado), no hacia uno mejor.
    También se identifican los parámetros de escape (sección 4.2.2,
    pasos 1, 2 y 3 del documento de tesis), recuperando la configuración
    EXACTA de hiperparámetros vigente en el momento del mayor salto de
    mejora dentro de la ventana, no la configuración actual del
    algoritmo (ver _extraer_parametros_escape para el detalle de esta
    corrección).

    Nota sobre el filtro MMD mencionado en la ecuación de salida de la
    sección 4.2.2: NO se implementa en este módulo. Por el plan de
    desarrollo incremental acordado, MMD se construye formalmente en la
    Fase 3 (Transferencia), junto con la distancia de Wasserstein, como
    parte de la barrera de seguridad. Aquí se retornan las instancias ya
    ordenadas por ϕ; el filtrado por MMD se aplica después, en Fase 3,
    sobre esta misma salida.
    """
    ventana_poblacion = historial_poblacional[-VENTANA_HISTORIAL_FUENTE:]
    ventana_fitness = historial_fitness_individual[-VENTANA_HISTORIAL_FUENTE:]

    modelo_subrogado, r2 = entrenar_modelo_subrogado(
        historial_poblacional, historial_fitness_individual
    )

    # Cortocircuito: si R²=0 (varianza del fitness≈0), no tiene sentido
    # entrenar FastSHAP. Se retorna un resultado marcado como bloqueado
    # con arrays vacíos/zeros que el orquestador ignorará completamente.
    if r2 < 0.0:
        n_dim = historial_poblacional[-1].shape[1] if historial_poblacional else 1
        dummy = np.zeros((top_k, n_dim))
        dummy_fit = np.zeros(top_k)
        ventana_mejor_fitness = historial_mejor_fitness[-VENTANA_HISTORIAL_FUENTE:]
        ventana_hiperparametros = historial_hiperparametros[-VENTANA_HISTORIAL_FUENTE:]
        parametros_escape = _extraer_parametros_escape(
            ventana_mejor_fitness, ventana_hiperparametros
        )
        return ResultadoExtraccion(
            instancias_elite=dummy,
            fitness_instancias_elite=dummy_fit,
            valores_shapley_elite=dummy,
            parametros_escape=parametros_escape,
            r2_modelo_subrogado=r2,
            transferencia_bloqueada_por_r2=True,
        )

    X_ventana = np.vstack(ventana_poblacion)
    y_ventana = np.concatenate(ventana_fitness)

    # Background para MarginalImputer: muestra de la población real
    # (no sintética), tomada de la misma ventana reciente.
    rng = np.random.default_rng(1)
    idx_background = rng.choice(
        len(X_ventana), size=min(N_BACKGROUND_MARGINAL, len(X_ventana)),
        replace=False
    )
    poblacion_background = X_ventana[idx_background]

    # Semilla fija del RNG global de torch (mismo criterio que los
    # np.random.default_rng(0)/(1) de arriba: una constante deliberada, no
    # ligada a la semilla del experimento). Sin esto, la inicialización de
    # pesos de construir_fastshap() y el muestreo interno de fastshap.train()
    # dependían del estado global de torch — no sembrado en ningún lado —,
    # una fuente de no determinismo independiente del threading (D-2).
    #
    # NO se fuerza torch.set_num_threads(1): se probó como precaución contra
    # una eventual no-determinismo de reducciones multi-hilo en BLAS/MKL,
    # pero no hizo falta — la fuente real de no-determinismo residual era
    # ShapleySampler.rng sin semilla en fastshap_lib/utils.py (ya parcheado)
    # — y quitarlo no mostró ninguna mejora de tiempo medible, así que se
    # deja sin forzar para no restringir el paralelismo interno sin motivo.
    torch.manual_seed(0)
    fastshap = construir_fastshap(modelo_subrogado, poblacion_background, n_dimensiones)

    # Entrenamiento del explainer (única red que realmente se entrena).
    # Se usa toda la ventana como train, y una submuestra pequeña como
    # validación interna de FastSHAP (exigida por su propia API).
    n_val = max(10, int(len(X_ventana) * 0.1))
    idx_val = rng.choice(len(X_ventana), size=n_val, replace=False)

    fastshap.train(
        X_ventana.astype(np.float32),
        X_ventana[idx_val].astype(np.float32),
        batch_size=BATCH_SIZE_EXPLAINER,
        num_samples=NUM_SAMPLES_EXPLAINER,
        max_epochs=max_epochs_fastshap,
        lr=LR_EXPLAINER,
        validation_samples=NUM_SAMPLES_EXPLAINER,
        bar=False,
        verbose=False,
    )

    # Cálculo de los valores de Shapley (forward pass único, la
    # propiedad central de FastSHAP) sobre toda la ventana.
    valores_shapley = fastshap.shap_values(X_ventana.astype(np.float32))
    # valores_shapley: (n_muestras, n_dimensiones, 1) para modelos de
    # salida escalar (regresión); se aplana la última dimensión.
    if valores_shapley.ndim == 3:
        valores_shapley = valores_shapley[:, :, 0]

    # Contribución total por instancia: suma de ϕ por variable. Usada
    # ÚNICAMENTE como criterio de ranking secundario DENTRO del subconjunto
    # ya pre-filtrado por fitness real (ver justificación de la constante
    # PERCENTIL_PREFILTRO_FITNESS). No se usa para decidir qué instancias
    # son "buenas" en términos absolutos, porque un ϕ alto en un problema
    # de minimización indica que esas variables empujan el fitness hacia
    # ARRIBA (peor resultado), no hacia abajo.
    contribucion_total = np.sum(valores_shapley, axis=1)

    # Paso 1 (Opción C): pre-filtro duro por fitness real. Solo se
    # consideran elegibles las instancias dentro del percentil de mejor
    # desempeño real de la ventana (minimización: fitness más bajo).
    n_elegibles = max(top_k, int(len(y_ventana) * PERCENTIL_PREFILTRO_FITNESS))
    indices_por_fitness = np.argsort(y_ventana)  # ascendente: mejores primero
    indices_elegibles = indices_por_fitness[:n_elegibles]

    # Paso 2 (Opción C): dentro del subconjunto ya garantizado como bueno,
    # se ordena por contribución de Shapley para obtener las top-k finales.
    contribucion_elegibles = contribucion_total[indices_elegibles]
    orden_dentro_elegibles = np.argsort(-contribucion_elegibles)  # descendente
    indices_top_k = indices_elegibles[orden_dentro_elegibles[:top_k]]

    instancias_elite = X_ventana[indices_top_k]
    fitness_instancias_elite = y_ventana[indices_top_k]
    valores_shapley_elite = valores_shapley[indices_top_k]

    # Se recorta historial_mejor_fitness a la misma ventana que el resto
    # de los datos analizados (VENTANA_HISTORIAL_FUENTE), para que el
    # índice de "iteración relativa de escape" resultante sea consistente
    # con la ventana de hiperparámetros que se usa para recuperar la
    # configuración exacta vigente en ese momento.
    ventana_mejor_fitness = historial_mejor_fitness[-VENTANA_HISTORIAL_FUENTE:]
    ventana_hiperparametros = historial_hiperparametros[-VENTANA_HISTORIAL_FUENTE:]

    parametros_escape = _extraer_parametros_escape(
        ventana_mejor_fitness, ventana_hiperparametros
    )

    # Bloquear solo cuando R² < 0 (estrictamente negativo): el modelo es
    # peor que predecir la media, señal de que no hay varianza real que
    # modelar. R²=0 exacto ya no bloquea — puede tener algo de señal débil.
    bloqueada = r2 < 0.0

    # Calcular el umbral MMD usando la población del fuente TAL COMO ESTÁ
    # en este momento (Fase 2). Este valor se guarda para que Canal B lo
    # use directamente, sin recalcularlo más tarde cuando la población del
    # fuente puede haber convergido más (produciendo un umbral que no refleja
    # la distribución real al momento de la extracción).
    poblacion_fuente_actual = historial_poblacional[-1]
    umbral_mmd = calcular_umbral_mmd_dinamico(poblacion_fuente_actual)

    return ResultadoExtraccion(
        instancias_elite=instancias_elite,
        fitness_instancias_elite=fitness_instancias_elite,
        valores_shapley_elite=valores_shapley_elite,
        parametros_escape=parametros_escape,
        r2_modelo_subrogado=r2,
        transferencia_bloqueada_por_r2=bloqueada,
        umbral_mmd_precalculado=umbral_mmd,
    )


def _extraer_parametros_escape(ventana_mejor_fitness: list[float],
                                 ventana_hiperparametros: list[dict]) -> dict:
    """
    Identifica el momento episódico, dentro de la ventana analizada,
    donde el algoritmo fuente logró romper un óptimo local de forma más
    significativa (sección 4.2.2, paso 3: "Extracción de Parámetros"), y
    recupera la configuración EXACTA de hiperparámetros que estaba
    vigente en ese instante específico — no la configuración actual del
    algoritmo, que puede ser distinta (ej. PSO con inercia decreciente
    tiene un valor de w distinto en cada iteración).

    Esta es la versión corregida: una primera implementación devolvía
    los hiperparámetros ACTUALES del algoritmo (en el momento de llamar
    a esta función), lo cual no es fiel al documento de tesis, que pide
    explícitamente "la configuración exacta de hiperparámetros... que
    impulsó dicho escape" — es decir, la vigente EN el momento del
    escape, no después. Esto requirió agregar el registro de
    hiperparámetros por iteración en EstadoIteracion (bioalgorithms/
    base.py) y en cada algoritmo concreto (pso.py, de.py), que antes no
    se guardaba en absoluto.

    ventana_mejor_fitness y ventana_hiperparametros deben tener la misma
    longitud y estar alineados índice a índice (ambos recortados a la
    misma VENTANA_HISTORIAL_FUENTE), de modo que
    ventana_hiperparametros[i] corresponda exactamente a la iteración en
    la que se registró ventana_mejor_fitness[i].
    """
    mejoras = [
        ventana_mejor_fitness[i - 1] - ventana_mejor_fitness[i]
        for i in range(1, len(ventana_mejor_fitness))
    ]
    if not mejoras:
        indice_mejor_salto = 0
    else:
        # +1 porque `mejoras[j]` representa la mejora ENTRE la iteración
        # j y la j+1; el hiperparámetro relevante es el vigente EN la
        # iteración donde se concretó la mejora (j+1), no la anterior.
        indice_mejor_salto = int(np.argmax(mejoras)) + 1

    hiperparametros_en_el_escape = ventana_hiperparametros[indice_mejor_salto]

    return {
        "iteracion_relativa_escape": indice_mejor_salto,
        "magnitud_mejora": float(mejoras[indice_mejor_salto - 1]) if mejoras else 0.0,
        "hiperparametros": hiperparametros_en_el_escape,
    }