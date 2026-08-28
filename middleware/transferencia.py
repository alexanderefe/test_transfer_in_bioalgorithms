"""
Fase 3 — Transferencia (middleware/transferencia.py)

Implementa el protocolo de transferencia jerárquico descrito en la
sección 4.2.3 del documento de tesis:

  Canal A: Transferencia Paramétrica Gradual (RMP, adaptado de MFEA-II).
  Canal B: Inyección de Instancias de Élite (si Canal A no es suficiente).
  Barrera de seguridad: Wasserstein (dura, sobre poblaciones) + MMD
    (fina, sobre instancias) — implementadas en barrera_seguridad.py.

DECISIONES DE FORMALIZACIÓN ACORDADAS EXPLÍCITAMENTE (el documento deja
estos valores abiertos a "sintonización empírica" o simplemente no los
especifica):

1. RMP: valor inicial = 0.05 (conservador), tope = 0.6 (NO se permite
   llegar a 1.0 deliberadamente: un RMP=1.0 significaría que el
   algoritmo objetivo adopta COMPLETAMENTE la configuración del fuente,
   lo cual contradice el principio de "baja invasividad" que el propio
   documento atribuye al Canal A, y erosiona la heterogeneidad entre
   metaheurísticas que es el tema central de la tesis). El incremento
   tras una evaluación exitosa es proporcional al ΔScore obtenido, con
   factor de escala 0.5x (un factor 1.0x permitiría que una sola mejora
   grande del Score saturara el RMP casi de inmediato, lo cual es
   inconsistente con haber elegido deliberadamente un tope conservador:
   el camino hacia ese tope también debe ser gradual).

2. Criterio de evaluación del Canal A ("número predeterminado de
   iteraciones" sin cambios significativos, según el documento): se fija
   en N=30 iteraciones, midiendo ΔScore (la misma métrica formalizada en
   Fase 1/4) en vez del fitness crudo. Se eligió 30 en vez de reutilizar
   la ventana de FIR (20, Fase 1) porque el RMP creciente necesita
   tiempo para acumular varios incrementos antes de que el Canal A
   pueda mostrar su efecto real; evaluarlo demasiado pronto penalizaría
   injustamente un mecanismo diseñado para ser gradual. Tampoco se usó
   la ventana del modelo subrogado (50, Fase 2) para mantener capacidad
   de reacción rápida del middleware ante estancamiento real. El umbral
   de "cambio significativo" se fija en ΔS > 0.02 (no cualquier mejora
   positiva), para no escalar a Canal B por ruido estadístico menor.

3. Canal B: el número de individuos a reemplazar en la población
   objetivo es exactamente el número de instancias de élite ya
   extraídas en Fase 2 (TOP_K_INSTANCIAS_ELITE=5), sin un porcentaje
   adicional de la población a sustituir definido por separado.

4. Cierre de ciclo (PROVISIONAL, pendiente de reemplazo en Fase 4): este
   módulo solo implementa el inicio y escalamiento del ciclo de
   transferencia (NINGUNO -> CANAL_A -> CANAL_B), pero el documento de
   tesis define el cierre formal del ciclo en la Fase 4 (Comprobación,
   sección 4.2.4), que mide ΔScore y realiza auditoría SHAP de
   asimilación de instancias — fase aún no construida. Mientras tanto,
   se usa un criterio mínimo y explícitamente temporal: si el algoritmo
   objetivo deja de estar estancado (según el Score de Fase 1) mientras
   un canal está activo, el ciclo se considera cerrado con éxito y el
   estado vuelve a NINGUNO. Este mecanismo DEBE ser reemplazado, no
   simplemente extendido, cuando se implemente la Fase 4 real.

HALLAZGO DOCUMENTADO (validado en tests/test_fase3_transferencia.py):
en el escenario de prueba (F11/D=20/semilla=42, roles DE=fuente/
PSO=objetivo), el Canal A NUNCA logró un ΔScore significativo y siempre
escaló a Canal B tras la primera ventana de evaluación. La causa raíz es
la misma ambigüedad del Score documentada desde la Fase 1: PSO cae bajo
el umbral de estancamiento por CONVERGENCIA EXITOSA (colapso natural de
diversidad al acercarse al óptimo), no por una patología de balance
exploración-explotación que ajustar hiperparámetros vía RMP pueda
resolver. Sin embargo, esto no invalida el Canal B subsecuente: el
criterio fuente_en_condiciones_optimas() garantiza que solo se transfiere
conocimiento cuando el fuente tiene fitness objetivamente mejor, por lo
que las instancias inyectadas en Canal B siguen siendo información
genuinamente útil, incluso cuando el diagnóstico que disparó el ciclo
(Score bajo) no correspondía a una patología real. Queda como limitación
conocida que, en escenarios como este, el Canal A puede representar
iteraciones "desperdiciadas" antes de llegar a un Canal B efectivo.
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from middleware.deteccion import ResultadoDeteccion
from middleware.barrera_seguridad import (
    evaluar_barrera_wasserstein,
    filtrar_instancias_por_mmd,
)


# ----------------------------------------------------------------------
# Configuración del Canal A (RMP).
# ----------------------------------------------------------------------

RMP_INICIAL = 0.05
RMP_TOPE = 0.6                  # tope conservador deliberado (ver docstring)
FACTOR_INCREMENTO_RMP = 0.5      # incremento = factor * ΔScore (ver docstring)
FRACCION_VENTANA_CANAL_A = 0.10  # la ventana de evaluación del Canal A es el
                                  # 10% del total de iteraciones del experimento.
                                  # DECISIÓN ACORDADA: se pasó de un valor fijo
                                  # (30 iteraciones absolutas) a un porcentaje
                                  # del total, porque 30 iteraciones fijas sobre
                                  # 2000 representan solo el 1.5% — demasiado
                                  # poco para que el RMP tenga efecto real antes
                                  # de que el middleware evalúe si Canal A funcionó.
                                  # Con 10%, la ventana escala con el experimento:
                                  # 200 iter en 2000, 30 iter en 300, etc.
UMBRAL_DELTA_SCORE_SIGNIFICATIVO = 0.02


class CanalActivo(Enum):
    NINGUNO = "ninguno"
    CANAL_A = "canal_a_rmp"
    CANAL_B = "canal_b_elite"


@dataclass
class EstadoTransferencia:
    """
    Estado persistente del proceso de transferencia para un par
    (fuente, objetivo) en ejecución paralela. Se mantiene a lo largo de
    múltiples ciclos del middleware, ya que el Canal A requiere
    acumular iteraciones antes de evaluarse.
    """
    canal_activo: CanalActivo = CanalActivo.NINGUNO
    rmp_actual: float = RMP_INICIAL
    iteracion_inicio_canal_a: int = 0
    score_al_iniciar_canal_a: float = 0.0
    en_backoff: bool = False
    iteracion_fin_backoff: int = 0
    log_eventos: list = field(default_factory=list)


@dataclass
class ResultadoTransferencia:
    """Salida de un ciclo de transferencia (Tabla 2 del documento)."""
    canal_utilizado: CanalActivo
    score_pre_transferencia: float
    distancia_wasserstein: float
    transferencia_abortada: bool
    payload_inyectado: dict
    rmp_resultante: float


def _registrar_log(estado: EstadoTransferencia, iteracion: int, evento: str,
                    detalles: dict) -> None:
    """
    Registra un evento en el log de auditoría (Tabla 2 del documento de
    tesis: Timestamp, Canal Utilizado, Score Pre-Transf., Payload
    Inyectado, Dist. Wasserstein). Se usa la iteración como timestamp
    lógico, ya que el middleware opera sobre iteraciones discretas, no
    sobre tiempo de reloj real.
    """
    estado.log_eventos.append({
        "iteracion": iteracion,
        "evento": evento,
        **detalles,
    })


def aplicar_canal_a(algoritmo_objetivo, parametros_fuente: dict,
                     rmp_actual: float) -> dict:
    """
    Canal A: Transferencia Paramétrica Gradual. Interpola los
    hiperparámetros actuales del algoritmo objetivo hacia los del
    algoritmo fuente, con intensidad controlada por rmp_actual
    (0 = sin cambio, 1 = adopción total del valor del fuente).

    Solo se interpolan las claves de hiperparámetros que EXISTEN en
    ambos diccionarios (intersección), preservando el agnosticismo: el
    middleware no necesita saber qué significan "F", "CR", "w", etc.,
    solo que son números que pueden interpolarse linealmente. Claves
    presentes solo en el fuente (sin equivalente en el objetivo) se
    ignoran, ya que no hay un hiperparámetro correspondiente al cual
    aplicarlas en el algoritmo objetivo.
    """
    hiperparametros_objetivo_actuales = algoritmo_objetivo.obtener_hiperparametros_actuales()
    nuevos_parametros = {}

    for clave in hiperparametros_objetivo_actuales:
        valor_objetivo = hiperparametros_objetivo_actuales[clave]
        if clave in parametros_fuente:
            valor_fuente = parametros_fuente[clave]
            valor_interpolado = (
                (1 - rmp_actual) * valor_objetivo + rmp_actual * valor_fuente
            )
            nuevos_parametros[clave] = valor_interpolado
        # Si la clave no existe en el fuente, no se modifica: el
        # algoritmo objetivo conserva su valor actual para ese
        # hiperparámetro específico.

    algoritmo_objetivo.recibir_parametros_transferidos(nuevos_parametros)
    return nuevos_parametros


def aplicar_canal_b(algoritmo_objetivo, instancias_elite: np.ndarray,
                     fitness_instancias_elite: np.ndarray) -> dict:
    """
    Canal B: Inyección de Instancias de Élite. Sustituye directamente
    los peores individuos de la población objetivo por las instancias
    de élite ya extraídas y filtradas (Fase 2 + filtro MMD), usando el
    método ya implementado en la interfaz común (bioalgorithms/base.py).
    """
    algoritmo_objetivo.recibir_instancias_elite(instancias_elite, fitness_instancias_elite)
    return {
        "n_instancias_inyectadas": len(instancias_elite),
        "fitness_instancias": fitness_instancias_elite.tolist(),
    }


def ejecutar_ciclo_transferencia(
    estado: EstadoTransferencia,
    algoritmo_fuente,
    algoritmo_objetivo,
    resultado_deteccion_objetivo: ResultadoDeteccion,
    parametros_fuente: dict,
    instancias_elite: np.ndarray,
    fitness_instancias_elite: np.ndarray,
    limites: np.ndarray,
    iteracion_actual: int,
    ventana_canal_a: int = 30,
) -> ResultadoTransferencia:
    """
    Orquesta un ciclo completo de la Fase 3: aplica la barrera de
    seguridad (Wasserstein sobre poblaciones, ecuación 7), filtra las
    instancias candidatas por MMD, y decide si activar/mantener Canal A
    o escalar a Canal B, según el protocolo jerárquico de la sección
    4.2.3 del documento.

    ventana_canal_a: número de ITERACIONES REALES de los algoritmos que
        deben transcurrir desde el inicio del Canal A antes de evaluar
        si tuvo éxito (ΔS > UMBRAL_DELTA_SCORE_SIGNIFICATIVO) o escalar
        a Canal B. Se recibe como parámetro (en vez de constante fija)
        porque se calcula como fracción del total de iteraciones del
        experimento (FRACCION_VENTANA_CANAL_A × n_iteraciones_total),
        de modo que la ventana escala con el tamaño del experimento.
    """
    # --- Barrera dura: Wasserstein sobre las poblaciones completas ---
    resultado_wasserstein = evaluar_barrera_wasserstein(
        algoritmo_fuente.poblacion, algoritmo_objetivo.poblacion, limites
    )

    if not resultado_wasserstein.transferencia_permitida:
        _registrar_log(estado, iteracion_actual, "transferencia_abortada_wasserstein", {
            "distancia_wasserstein": resultado_wasserstein.distancia_wasserstein,
            "umbral": resultado_wasserstein.umbral_wasserstein,
        })
        estado.en_backoff = True
        estado.iteracion_fin_backoff = iteracion_actual + ventana_canal_a
        return ResultadoTransferencia(
            canal_utilizado=CanalActivo.NINGUNO,
            score_pre_transferencia=resultado_deteccion_objetivo.score,
            distancia_wasserstein=resultado_wasserstein.distancia_wasserstein,
            transferencia_abortada=True,
            payload_inyectado={},
            rmp_resultante=estado.rmp_actual,
        )

    # --- Si el middleware está en backoff, no se interviene aún ---
    if estado.en_backoff:
        if iteracion_actual < estado.iteracion_fin_backoff:
            return ResultadoTransferencia(
                canal_utilizado=CanalActivo.NINGUNO,
                score_pre_transferencia=resultado_deteccion_objetivo.score,
                distancia_wasserstein=resultado_wasserstein.distancia_wasserstein,
                transferencia_abortada=False,
                payload_inyectado={"en_backoff": True},
                rmp_resultante=estado.rmp_actual,
            )
        estado.en_backoff = False  # backoff expirado, se reanuda evaluación

    # --- Decisión de canal: iniciar/continuar Canal A, o evaluar escalar ---
    if estado.canal_activo == CanalActivo.NINGUNO:
        estado.canal_activo = CanalActivo.CANAL_A
        estado.iteracion_inicio_canal_a = iteracion_actual
        estado.score_al_iniciar_canal_a = resultado_deteccion_objetivo.score

    # MECANISMO PROVISIONAL DE CIERRE DE CICLO (acordado explícitamente
    # como temporal): si el algoritmo objetivo deja de estar estancado
    # mientras hay un canal activo, se considera el ciclo cerrado con
    # éxito y se regresa a NINGUNO (monitoreo pasivo). Esta es una
    # simplificación deliberada: el documento de tesis define el cierre
    # real del ciclo en la Fase 4 (Comprobación), mediante ΔScore y
    # auditoría SHAP de asimilación de instancias (sección 4.2.4), que
    # AÚN NO ESTÁ CONSTRUIDA. Cuando se implemente la Fase 4, este bloque
    # debe ser reemplazado por la lógica formal de esa fase, no
    # simplemente extendido — el criterio real de cierre exitoso es más
    # específico que "ya no está estancado" (incluye verificar asimilación
    # de instancias inyectadas, no solo el valor de S).
    if estado.canal_activo != CanalActivo.NINGUNO and not resultado_deteccion_objetivo.estancado:
        _registrar_log(estado, iteracion_actual, "ciclo_cerrado_provisional", {
            "canal_al_cerrar": estado.canal_activo.value,
            "score_actual": resultado_deteccion_objetivo.score,
        })
        estado.canal_activo = CanalActivo.NINGUNO
        estado.rmp_actual = RMP_INICIAL
        return ResultadoTransferencia(
            canal_utilizado=CanalActivo.NINGUNO,
            score_pre_transferencia=resultado_deteccion_objetivo.score,
            distancia_wasserstein=resultado_wasserstein.distancia_wasserstein,
            transferencia_abortada=False,
            payload_inyectado={"ciclo_cerrado": True},
            rmp_resultante=estado.rmp_actual,
        )

    iteraciones_en_canal_a = iteracion_actual - estado.iteracion_inicio_canal_a

    if estado.canal_activo == CanalActivo.CANAL_A:
        payload = aplicar_canal_a(
            algoritmo_objetivo, parametros_fuente, estado.rmp_actual
        )
        canal_usado = CanalActivo.CANAL_A

        if iteraciones_en_canal_a >= ventana_canal_a:
            delta_score = resultado_deteccion_objetivo.score - estado.score_al_iniciar_canal_a
            if delta_score > UMBRAL_DELTA_SCORE_SIGNIFICATIVO:
                # Canal A funciona: incrementar RMP proporcionalmente.
                incremento = FACTOR_INCREMENTO_RMP * delta_score
                estado.rmp_actual = float(np.clip(
                    estado.rmp_actual + incremento, RMP_INICIAL, RMP_TOPE
                ))
                estado.iteracion_inicio_canal_a = iteracion_actual
                estado.score_al_iniciar_canal_a = resultado_deteccion_objetivo.score
                _registrar_log(estado, iteracion_actual, "canal_a_exitoso", {
                    "delta_score": delta_score, "rmp_nuevo": estado.rmp_actual,
                })
            else:
                # Canal A insuficiente tras la ventana de evaluación:
                # escalar a Canal B.
                estado.canal_activo = CanalActivo.CANAL_B
                _registrar_log(estado, iteracion_actual, "escalado_a_canal_b", {
                    "delta_score": delta_score,
                })

    if estado.canal_activo == CanalActivo.CANAL_B:
        # Filtro MMD sobre las instancias candidatas antes de inyectar.
        aprobadas = filtrar_instancias_por_mmd(
            instancias_elite, algoritmo_fuente.poblacion, algoritmo_objetivo.poblacion
        )
        instancias_filtradas = instancias_elite[aprobadas]
        fitness_filtrado = fitness_instancias_elite[aprobadas]

        payload = aplicar_canal_b(
            algoritmo_objetivo, instancias_filtradas, fitness_filtrado
        )
        payload["instancias_rechazadas_por_mmd"] = int((~aprobadas).sum())
        canal_usado = CanalActivo.CANAL_B
        _registrar_log(estado, iteracion_actual, "canal_b_aplicado", payload)

    _registrar_log(estado, iteracion_actual, "ciclo_transferencia", {
        "canal": canal_usado.value,
        "score_pre": resultado_deteccion_objetivo.score,
        "distancia_wasserstein": resultado_wasserstein.distancia_wasserstein,
    })

    return ResultadoTransferencia(
        canal_utilizado=canal_usado,
        score_pre_transferencia=resultado_deteccion_objetivo.score,
        distancia_wasserstein=resultado_wasserstein.distancia_wasserstein,
        transferencia_abortada=False,
        payload_inyectado=payload,
        rmp_resultante=estado.rmp_actual,
    )