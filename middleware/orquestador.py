"""
Orquestador con threading — Solución A (middleware/orquestador.py)

Implementa la Solución A acordada: tres hilos corriendo en el mismo
proceso con memoria compartida.

  Hilo 1 — Algoritmo A (PSO o DE)
  Hilo 2 — Algoritmo B (PSO o DE)
  Hilo 3 — Middleware (detección + extracción + transferencia)

MECANISMO DE PAUSA (acordado explícitamente):
  Cuando el middleware detecta que debe entrar a Fase 2 o Fase 3
  (operaciones costosas: XGBoost + FastSHAP + Wasserstein + MMD),
  señaliza a ambos hilos de algoritmo que se detengan. Los hilos
  verifican esta señal ENTRE ITERACIONES (nunca a mitad de una), de
  modo que no se interrumpe ningún paso de cálculo interno. Solo cuando
  ambos confirman estar pausados, el middleware ejecuta la extracción
  y transferencia. Al terminar, reanuda ambos hilos y el ciclo continúa.

MECANISMO DE SINCRONIZACIÓN (threading.Event):
  - evento_pausa: cuando está SET, los hilos de algoritmo deben detenerse
    y esperar. Cuando está CLEAR, pueden avanzar.
  - evento_algoritmo_a_pausado / evento_algoritmo_b_pausado: cada hilo
    confirma al middleware que está efectivamente pausado antes de que
    este empiece a trabajar con los datos.

LOG:
  - Consola en tiempo real: mensajes con timestamp para cada evento
    relevante (activación de Fase 2/3, resultado de barrera, canal
    utilizado, ΔScore, etc.).
  - Archivo JSON al finalizar: registro completo de todas las
    activaciones con sus métricas, guardado en el directorio que
    especifique el usuario.

ROLES FUENTE/OBJETIVO:
  Al volver de Fase 3 a Fase 1, el middleware re-evalúa dinámicamente
  cuál de los dos algoritmos es el fuente y cuál el objetivo, según su
  fitness actual en ese momento. El algoritmo que recibió la
  transferencia puede haber mejorado, cambiando los roles respecto al
  ciclo anterior.
"""

import threading
import time
import json
import datetime
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from middleware.deteccion import (
    calcular_score, fuente_en_condiciones_optimas,
    asignar_roles_dinamicos, UMBRAL_ESTANCAMIENTO,
)
from middleware.extraccion import extraer_conocimiento, VENTANA_HISTORIAL_FUENTE
from middleware.transferencia import (
    EstadoTransferencia, ejecutar_ciclo_transferencia, CanalActivo,
    FRACCION_VENTANA_CANAL_A, aplicar_canal_a, aplicar_canal_b,
    RMP_INICIAL, RMP_TOPE, FACTOR_INCREMENTO_RMP,
    UMBRAL_DELTA_SCORE_SIGNIFICATIVO,
)
from middleware.barrera_seguridad import (
    evaluar_barrera_wasserstein, filtrar_instancias_por_mmd,
)


# ─── Estructuras del log ─────────────────────────────────────────────────────

@dataclass
class EntradaLogFase1:
    iteracion: int
    rol_fuente: str
    rol_objetivo: str
    fitness_fuente: float
    fitness_objetivo: float
    score_objetivo: float
    estancado: bool


@dataclass
class EntradaLogFase2:
    iteracion: int
    r2_modelo_subrogado: float
    n_instancias_elite: int
    fitness_instancias_elite: list
    iteracion_relativa_escape: int
    magnitud_mejora_escape: float
    hiperparametros_escape: dict


@dataclass
class EntradaLogFase3:
    iteracion: int
    distancia_wasserstein: float
    umbral_wasserstein: float
    transferencia_permitida: bool
    canal_utilizado: str
    n_instancias_aprobadas_mmd: int
    n_instancias_rechazadas_mmd: int
    rmp_resultante: float
    delta_score_posterior: Optional[float] = None


@dataclass
class ResumenEjecucion:
    n_iteraciones_totales: int = 0
    n_activaciones_fase2: int = 0
    n_reentrenamientos_por_cambio_fuente: int = 0  # veces que FastSHAP
                                                     # se re-entrenó porque
                                                     # el fuente cambió a
                                                     # mitad de un ciclo activo
    n_activaciones_fase3: int = 0
    n_transferencias_abortadas_wasserstein: int = 0
    n_transferencias_canal_a: int = 0
    n_transferencias_canal_b: int = 0
    n_instancias_inyectadas_total: int = 0
    fitness_final_algoritmo_a: float = float("inf")
    fitness_final_algoritmo_b: float = float("inf")
    tiempo_total_segundos: float = 0.0
    tiempo_pausado_en_middleware_segundos: float = 0.0
    log_fase1: list = field(default_factory=list)
    log_fase2: list = field(default_factory=list)
    log_fase3: list = field(default_factory=list)


# ─── Hilo de algoritmo ───────────────────────────────────────────────────────

class HiloAlgoritmo(threading.Thread):
    """
    Ejecuta un algoritmo bioinspirado iteración a iteración, verificando
    entre cada iteración si debe pausarse. La pausa ocurre SOLO entre
    iteraciones completas, nunca a mitad de un paso de cálculo interno.
    """

    def __init__(self, algoritmo, nombre: str,
                 evento_pausa: threading.Event,
                 evento_pausado: threading.Event,
                 n_iteraciones: int):
        super().__init__(daemon=True)
        self.algoritmo = algoritmo
        self.nombre = nombre
        self.evento_pausa = evento_pausa
        self.evento_pausado = evento_pausado
        self.n_iteraciones = n_iteraciones
        self.terminado = False
        self._lock_historial = threading.Lock()

    def run(self):
        for _ in range(self.n_iteraciones):
            # Verificar si el middleware solicitó una pausa.
            # Se verifica ENTRE iteraciones completas para no interrumpir
            # ningún paso de cálculo interno del algoritmo.
            if self.evento_pausa.is_set():
                self.evento_pausado.set()  # confirmar al middleware que estoy pausado

                # Esperar hasta que el middleware limpie el evento de pausa
                # (evento_pausa.clear()). No se puede usar evento_pausa.wait()
                # directamente porque wait() espera hasta que el evento esté
                # SET — y ya lo está. Se necesita un evento de "permiso para
                # reanudar" separado, implementado aquí como espera activa
                # con sleep corto para no consumir CPU innecesariamente.
                while self.evento_pausa.is_set():
                    time.sleep(0.005)

                self.evento_pausado.clear()  # confirmar que reanudé

            # Ejecutar una iteración (operación atómica desde el punto de
            # vista del middleware: no se interrumpe a mitad)
            with self._lock_historial:
                self.algoritmo.ejecutar_iteracion()

        self.terminado = True

    def obtener_historial_seguro(self):
        """Lee el historial del algoritmo de forma thread-safe."""
        with self._lock_historial:
            return list(self.algoritmo.historial)

    def obtener_mejor_fitness(self) -> float:
        with self._lock_historial:
            return self.algoritmo.mejor_fitness_historico


# ─── Orquestador principal ───────────────────────────────────────────────────

class Orquestador:
    """
    Orquesta la ejecución paralela de dos algoritmos y el ciclo del
    middleware (Fases 1 → 2 → 3 → vuelta a 1) con hilos.
    """

    def __init__(self, algoritmo_a, algoritmo_b, limites: np.ndarray,
                 n_iteraciones: int, frecuencia_monitoreo: int = 5,
                 directorio_log: str = ".", nombre_log: str = "middleware_log",
                 max_epochs_fastshap: int = 50,
                 cooldown_post_ciclo: int = None):
        """
        algoritmo_a, algoritmo_b: instancias de AlgoritmoBioinspirado.
        frecuencia_monitoreo: cada cuántas iteraciones el hilo del
            middleware verifica el Score de Fase 1.
        max_epochs_fastshap: épocas de entrenamiento del explainer
            FastSHAP. Reducir para tests rápidos (ej. 10).
        cooldown_post_ciclo: iteraciones mínimas de espera tras cerrar
            un ciclo antes de poder abrir uno nuevo. Evita que el sistema
            re-detecte estancamiento inmediatamente después de una
            intervención y abra ciclos en ráfaga. Si None, se calcula
            automáticamente como ventana_canal_a (10% del total),
            igualando el período de prueba de hiperparámetros del Canal A
            con el período de espera post-intervención — ambos usan la
            misma escala temporal para mantener consistencia.
        """
        self.algoritmo_a = algoritmo_a
        self.algoritmo_b = algoritmo_b
        self.limites = limites
        self.n_iteraciones = n_iteraciones
        self.frecuencia_monitoreo = frecuencia_monitoreo
        self.directorio_log = directorio_log
        self.nombre_log = nombre_log
        self.max_epochs_fastshap = max_epochs_fastshap

        # Eventos de sincronización
        self._evento_pausa = threading.Event()       # SET = pausar algoritmos
        self._pausado_a = threading.Event()          # SET = A confirmó pausa
        self._pausado_b = threading.Event()          # SET = B confirmó pausa

        self.resumen = ResumenEjecucion()
        self._estado_transferencia = EstadoTransferencia()

        # Ventana de evaluación del Canal A: 10% del total de iteraciones.
        # Se calcula aquí una sola vez para que el valor sea consistente
        # durante toda la ejecución y aparezca claramente en el log.
        self._ventana_canal_a = max(
            10,
            int(FRACCION_VENTANA_CANAL_A * n_iteraciones)
        )

        # Enfriamiento post-ciclo: iteraciones mínimas entre el cierre
        # de un ciclo y la apertura del siguiente. Evita que el sistema
        # detecte estancamiento inmediatamente tras Canal B (antes de que
        # el efecto de la inyección se refleje en el Score) y abra ciclos
        # en ráfaga re-entrenando FastSHAP innecesariamente.
        self._cooldown_post_ciclo = (
            cooldown_post_ciclo if cooldown_post_ciclo is not None
            else self._ventana_canal_a  # 10% del total, igual que la ventana de Canal A
        )
        self._iteracion_fin_cooldown = 0
        self._iteracion_ultima_transferencia = 0
        self._hilo_a_ref = None  # se asigna en ejecutar()
        self._hilo_b_ref = None

        self._conocimiento_ciclo_actual = None
        self._rol_fuente_activo = None
        self._rol_objetivo_activo = None
        self._rol_fuente_ultimo_ciclo = None

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, mensaje: str, nivel: str = "INFO"):
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefijos = {"INFO": "ℹ", "FASE1": "①", "FASE2": "②",
                    "FASE3": "③", "OK": "✅", "WARN": "⚠", "ERR": "❌"}
        prefijo = prefijos.get(nivel, "·")
        print(f"[{ts}] {prefijo} {mensaje}")

    # ── Pausa / reanudación ──────────────────────────────────────────────────

    def _pausar_algoritmos(self):
        """
        Señaliza a ambos hilos que deben pausarse y espera confirmación
        de que ambos llegaron a un punto seguro (entre iteraciones).
        Si un hilo ya terminó (completó todas sus iteraciones), no puede
        confirmar la pausa — se omite la espera para ese hilo.
        """
        self._evento_pausa.set()
        # Solo esperar confirmación de hilos que aún están corriendo.
        # Un hilo terminado nunca llamará a evento_pausado.set(), así que
        # esperar su confirmación produciría un timeout de 30s innecesario.
        if not self._hilo_a_ref.terminado:
            self._pausado_a.wait(timeout=5)
        if not self._hilo_b_ref.terminado:
            self._pausado_b.wait(timeout=5)

    def _reanudar_algoritmos(self):
        """Limpia la señal de pausa para que ambos hilos continúen."""
        self._evento_pausa.clear()

    # ── Ciclo del middleware ─────────────────────────────────────────────────

    def _ejecutar_ciclo_middleware(self, iteracion_actual: int,
                                    hilo_a: HiloAlgoritmo,
                                    hilo_b: HiloAlgoritmo) -> bool:
        """
        Ejecuta el ciclo del middleware para la iteración actual.

        Tres estados posibles:

        1. MONITOREO PASIVO (self._conocimiento_ciclo_actual is None):
           Solo Fase 1. Si detecta estancamiento + fuente óptimo,
           ejecuta Fase 2 (extracción costosa) y abre un ciclo activo.

        2. CICLO ACTIVO (self._conocimiento_ciclo_actual is not None):
           Ejecuta Fase 3 con el conocimiento ya extraído, sin repetir
           Fase 2. Mientras el objetivo sigue estancado, el ciclo
           continúa aplicando Canal A iteración a iteración (ventana
           de 30 para evaluar si Canal A funciona o escalar a Canal B).

        3. CIERRE DE CICLO:
           Cuando el objetivo deja de estar estancado (S ≥ 0.2), el
           ciclo se considera cerrado. El estado se reinicia y el
           middleware vuelve a modo monitoreo pasivo (estado 1), listo
           para detectar un nuevo estancamiento —que puede ser en el
           mismo algoritmo u otro, con roles potencialmente distintos.

        Retorna True si se activó Fase 2 o Fase 3 en esta llamada.
        """
        hist_a = hilo_a.obtener_historial_seguro()
        hist_b = hilo_b.obtener_historial_seguro()

        if not hist_a or not hist_b:
            return False

        fitness_a = self.algoritmo_a.mejor_fitness_historico
        fitness_b = self.algoritmo_b.mejor_fitness_historico

        # ── Determinar roles actuales ─────────────────────────────────────
        # Los roles se evalúan siempre con el fitness real actual.
        rol_fuente_actual, rol_objetivo_actual = asignar_roles_dinamicos(
            fitness_a, fitness_b)

        if self._conocimiento_ciclo_actual is not None:
            if rol_fuente_actual != self._rol_fuente_activo:
                self._log(
                    f"it={iteracion_actual:4d} | Cambio de fuente durante ciclo "
                    f"activo: {self._rol_fuente_activo} → {rol_fuente_actual}. "
                    f"Invalidando conocimiento → re-entrenará FastSHAP.",
                    nivel="WARN",
                )
                self.resumen.n_reentrenamientos_por_cambio_fuente += 1
                self._conocimiento_ciclo_actual = None
                self._rol_fuente_activo = None
                self._rol_objetivo_activo = None
            else:
                rol_fuente_actual = self._rol_fuente_activo
                rol_objetivo_actual = self._rol_objetivo_activo
        elif (self._rol_fuente_ultimo_ciclo is not None
              and rol_fuente_actual != self._rol_fuente_ultimo_ciclo):
            # Cambio de fuente ENTRE ciclos: el ciclo anterior ya cerró
            # (_conocimiento_ciclo_actual es None) pero el fuente cambió.
            # Se contabiliza en el log; FastSHAP se re-entrenará sobre el
            # nuevo fuente cuando empiece el próximo ciclo (Fase 2).
            self._log(
                f"it={iteracion_actual:4d} | Cambio de fuente entre ciclos: "
                f"{self._rol_fuente_ultimo_ciclo} → {rol_fuente_actual}. "
                f"FastSHAP se re-entrenará con el nuevo fuente al iniciar ciclo.",
                nivel="WARN",
            )
            self.resumen.n_reentrenamientos_por_cambio_fuente += 1
            self._rol_fuente_ultimo_ciclo = rol_fuente_actual

        rol_fuente = rol_fuente_actual
        rol_objetivo = rol_objetivo_actual

        alg_fuente = self.algoritmo_a if rol_fuente == "a" else self.algoritmo_b
        alg_objetivo = self.algoritmo_a if rol_objetivo == "a" else self.algoritmo_b
        hist_fuente = hist_a if rol_fuente == "a" else hist_b
        hist_objetivo = hist_a if rol_objetivo == "a" else hist_b
        nombre_fuente = f"{'A' if rol_fuente == 'a' else 'B'}({type(alg_fuente).__name__})"
        nombre_objetivo = f"{'A' if rol_objetivo == 'a' else 'B'}({type(alg_objetivo).__name__})"

        # ── Fase 1: Detección ─────────────────────────────────────────────
        historial_fitness_obj_completo = [
            s.mejor_fitness_historico for s in hist_objetivo
        ]

        # Recorte para FIR post-transferencia (Opción 2 acordada):
        # Si hubo una transferencia previa, se recorta el historial para
        # que FIR no pueda mirar más atrás que el punto de transferencia.
        # Esto evita que el FIR compare el fitness actual contra valores
        # anteriores a la intervención, que contaminarían la medición de
        # si el algoritmo objetivo mejoró GRACIAS a la transferencia.
        # El historial COMPLETO se sigue usando para la activación del 15%
        # (iteracion_actual / max_iteraciones), que no depende del contenido
        # del historial sino solo del número de iteración actual.
        VENTANA_FIR = 20  # mismo valor que deteccion.py
        if self._iteracion_ultima_transferencia > 0:
            # Calcular cuántos snapshots corresponden a partir del punto
            # de transferencia (el snapshot en esa iteración es el índice
            # len - (iteracion_actual - iteracion_ultima_transferencia))
            n_desde_transferencia = (
                iteracion_actual - self._iteracion_ultima_transferencia
            )
            # Dejar como mínimo VENTANA_FIR+1 entradas para que FIR
            # pueda calcularse correctamente, pero sin retroceder más
            # allá del punto de transferencia.
            n_entradas_a_usar = max(1, min(
                n_desde_transferencia + 1,
                len(historial_fitness_obj_completo)
            ))
            historial_fitness_obj = historial_fitness_obj_completo[
                -n_entradas_a_usar:
            ]
        else:
            historial_fitness_obj = historial_fitness_obj_completo

        resultado_deteccion = calcular_score(
            historial_fitness=historial_fitness_obj,
            poblacion_actual=alg_objetivo.poblacion,
            limites=self.limites,
            iteracion_actual=iteracion_actual,
            max_iteraciones=self.n_iteraciones,
        )
        fuente_ok = fuente_en_condiciones_optimas(
            alg_fuente.mejor_fitness_historico,
            alg_objetivo.mejor_fitness_historico,
        )

        historial_fitness_fuente = [s.mejor_fitness_historico for s in hist_fuente]
        resultado_deteccion_fuente = calcular_score(
            historial_fitness=historial_fitness_fuente,
            poblacion_actual=alg_fuente.poblacion,
            limites=self.limites,
            iteracion_actual=iteracion_actual,
            max_iteraciones=self.n_iteraciones,
        )

        self.resumen.log_fase1.append(asdict(EntradaLogFase1(
            iteracion=iteracion_actual,
            rol_fuente=nombre_fuente,
            rol_objetivo=nombre_objetivo,
            fitness_fuente=float(alg_fuente.mejor_fitness_historico),
            fitness_objetivo=float(alg_objetivo.mejor_fitness_historico),
            score_objetivo=float(resultado_deteccion.score),
            estancado=resultado_deteccion.estancado,
        )))

        self._log(
            f"it={iteracion_actual:4d} | "
            f"fuente={nombre_fuente} fit={alg_fuente.mejor_fitness_historico:.2f} S={resultado_deteccion_fuente.score:.4f} | "
            f"objetivo={nombre_objetivo} fit={alg_objetivo.mejor_fitness_historico:.2f} S={resultado_deteccion.score:.4f} "
            f"estancado={resultado_deteccion.estancado}",
            nivel="FASE1",
        )

        # ── Estado 3: VERIFICACIÓN DE CIERRE ─────────────────────────────
        # Si había un ciclo activo y el objetivo ya no está estancado,
        # es porque el efecto de una transferencia anterior ya tuvo
        # resultado — pero el cierre formal ocurre en _aplicar_canal_b()
        # o en _evaluar_post_canal_a(), no aquí. Si llegamos a este punto
        # con ciclo activo y objetivo no estancado, es una condición
        # inesperada: registrar y limpiar como precaución.
        if self._conocimiento_ciclo_actual is not None and not resultado_deteccion.estancado:
            self._log(
                f"it={iteracion_actual:4d} | Ciclo activo pero objetivo ya no "
                f"estancado (S={resultado_deteccion.score:.4f}). "
                f"Limpiando estado como precaución.",
                nivel="WARN",
            )
            self._conocimiento_ciclo_actual = None
            self._rol_fuente_activo = None
            self._rol_objetivo_activo = None
            self._estado_transferencia = EstadoTransferencia()
            return False

        # ── Condición de activación ───────────────────────────────────────
        condicion = (
            resultado_deteccion.activo
            and resultado_deteccion.estancado
            and fuente_ok
            and len(hist_fuente) > VENTANA_HISTORIAL_FUENTE
        )

        if not condicion:
            return False

        # Respetar el período de enfriamiento post-ciclo (solo loguear una vez)
        if iteracion_actual < self._iteracion_fin_cooldown:
            return False

        # ── Estado 1 → 2: INICIO DE CICLO con Fase 2 ─────────────────────
        # Solo se ejecuta Fase 2 (costosa) al INICIO de un nuevo ciclo,
        # no en cada verificación mientras el ciclo está activo.
        if self._conocimiento_ciclo_actual is None:
            self._log(
                f"it={iteracion_actual:4d} | Nuevo ciclo detectado. "
                f"Activando Fase 2 (extracción)...",
                nivel="FASE2",
            )
            t0 = time.time()

            self._conocimiento_ciclo_actual = extraer_conocimiento(
                historial_poblacional=[s.poblacion for s in hist_fuente],
                historial_fitness_individual=[s.fitness for s in hist_fuente],
                historial_mejor_fitness=[s.mejor_fitness_historico for s in hist_fuente],
                historial_hiperparametros=[s.hiperparametros for s in hist_fuente],
                n_dimensiones=self.limites.shape[0],
                max_epochs_fastshap=self.max_epochs_fastshap,
            )
            self._rol_fuente_activo = rol_fuente
            self._rol_objetivo_activo = rol_objetivo
            self._rol_fuente_ultimo_ciclo = rol_fuente
            self.resumen.n_activaciones_fase2 += 1

            t_fase2 = time.time() - t0
            self.resumen.log_fase2.append(asdict(EntradaLogFase2(
                iteracion=iteracion_actual,
                r2_modelo_subrogado=float(
                    self._conocimiento_ciclo_actual.r2_modelo_subrogado),
                n_instancias_elite=len(
                    self._conocimiento_ciclo_actual.instancias_elite),
                fitness_instancias_elite=(
                    self._conocimiento_ciclo_actual.fitness_instancias_elite.tolist()),
                iteracion_relativa_escape=(
                    self._conocimiento_ciclo_actual.parametros_escape["iteracion_relativa_escape"]),
                magnitud_mejora_escape=float(
                    self._conocimiento_ciclo_actual.parametros_escape["magnitud_mejora"]),
                hiperparametros_escape=(
                    self._conocimiento_ciclo_actual.parametros_escape["hiperparametros"]),
            )))
            self._log(
                f"it={iteracion_actual:4d} | Fase 2 OK ({t_fase2:.1f}s) | "
                f"R²={self._conocimiento_ciclo_actual.r2_modelo_subrogado:.4f} | "
                f"élite: {[f'{v:.1f}' for v in self._conocimiento_ciclo_actual.fitness_instancias_elite]}",
                nivel="FASE2",
            )

            # Verificar R²: si ≤0 el fuente convergió completamente
            # (varianza de fitness≈0) y Shapley no tiene señal real.
            # Abortar el ciclo sin transferir nada y aplicar cooldown
            # para no volver a intentarlo inmediatamente.
            if self._conocimiento_ciclo_actual.transferencia_bloqueada_por_r2:
                self._log(
                    f"it={iteracion_actual:4d} | ⛔ Transferencia bloqueada: "
                    f"R²={self._conocimiento_ciclo_actual.r2_modelo_subrogado:.4f} < 0 "
                    f"(modelo peor que predecir la media). "
                    f"Shapley sin señal real → no se transfiere nada.",
                    nivel="WARN",
                )
                self._conocimiento_ciclo_actual = None
                self._rol_fuente_activo = None
                self._rol_objetivo_activo = None
                self._iteracion_fin_cooldown = (
                    iteracion_actual + self._cooldown_post_ciclo
                )
                return False

        # ── Estado 2: CICLO ACTIVO — aplicar Canal A o Canal B ───────────
        # Canal A: solo aplica RMP y retorna "canal_a_iniciado" para que
        # el bucle principal reanude los algoritmos y espere ventana_canal_a
        # iteraciones reales antes de evaluar el efecto.
        #
        # Canal B: intervención puntual (algoritmos permanecen pausados),
        # se delega a _aplicar_canal_b() y se retorna "canal_b_aplicado".
        #
        # Abort (Wasserstein): ya manejado arriba, nunca llega aquí.

        if (self._estado_transferencia.canal_activo == CanalActivo.NINGUNO
                or self._estado_transferencia.canal_activo == CanalActivo.CANAL_A):

            # Intentar Canal A: interpolar hiperparámetros del objetivo
            # hacia los del fuente usando el RMP actual.
            payload = aplicar_canal_a(
                alg_objetivo,
                self._conocimiento_ciclo_actual.parametros_escape["hiperparametros"],
                self._estado_transferencia.rmp_actual,
            )

            # OPCIÓN 1 acordada: si el payload es vacío (ninguna clave en
            # común entre fuente y objetivo, ej. PSO {w,c1,c2} vs DE {F,CR}),
            # Canal A es estructuralmente inerte — no modificaría nada al
            # algoritmo objetivo durante las 200 iteraciones de la ventana.
            # En ese caso se escala DIRECTAMENTE a Canal B sin esperar la
            # ventana, evitando desperdiciar 200 iteraciones y el ΔScore
            # negativo producido por la convergencia natural del objetivo
            # (que se confundía erróneamente con un efecto de Canal A).
            if not payload:
                self._log(
                    f"it={iteracion_actual:4d} | Canal A inerte: sin hiperparámetros "
                    f"comunes entre fuente ({nombre_fuente}) y objetivo ({nombre_objetivo}). "
                    f"Escalando directamente a Canal B.",
                    nivel="WARN",
                )
                self._estado_transferencia.canal_activo = CanalActivo.CANAL_B
                return self._aplicar_canal_b(
                    iteracion_actual, alg_fuente, alg_objetivo, resultado_deteccion
                )

            # Payload no vacío: Canal A tiene hiperparámetros que transferir.
            if self._estado_transferencia.canal_activo == CanalActivo.NINGUNO:
                self._estado_transferencia.canal_activo = CanalActivo.CANAL_A
                self._estado_transferencia.iteracion_inicio_canal_a = iteracion_actual
                self._estado_transferencia.score_al_iniciar_canal_a = (
                    resultado_deteccion.score)
            self.resumen.n_activaciones_fase3 += 1
            self.resumen.n_transferencias_canal_a += 1
            self.resumen.log_fase3.append(asdict(EntradaLogFase3(
                iteracion=iteracion_actual,
                distancia_wasserstein=float(
                    evaluar_barrera_wasserstein(
                        alg_fuente.poblacion, alg_objetivo.poblacion,
                        self.limites).distancia_wasserstein),
                umbral_wasserstein=0.0,
                transferencia_permitida=True,
                canal_utilizado=CanalActivo.CANAL_A.value,
                n_instancias_aprobadas_mmd=0,
                n_instancias_rechazadas_mmd=0,
                rmp_resultante=float(self._estado_transferencia.rmp_actual),
            )))
            self._log(
                f"it={iteracion_actual:4d} | Canal A: RMP={self._estado_transferencia.rmp_actual:.3f} "
                f"aplicado. Hiperparámetros objetivo → {payload}",
                nivel="FASE3",
            )
            return "canal_a_iniciado"

        # Canal B: inyección puntual, algoritmos permanecen pausados
        return self._aplicar_canal_b(
            iteracion_actual, alg_fuente, alg_objetivo, resultado_deteccion
        )

    def _evaluar_post_canal_a(self, iteracion_actual: int,
                               hilo_a: HiloAlgoritmo,
                               hilo_b: HiloAlgoritmo) -> None:
        """
        Evalúa si Canal A tuvo efecto tras ventana_canal_a iteraciones
        reales. Si ΔScore > umbral → éxito, RMP sube. Si no → escala a
        Canal B inmediatamente (algoritmos siguen pausados).
        """
        hist_a = hilo_a.obtener_historial_seguro()
        hist_b = hilo_b.obtener_historial_seguro()
        hist_objetivo = (hist_a if self._rol_objetivo_activo == "a" else hist_b)
        alg_fuente = (self.algoritmo_a if self._rol_fuente_activo == "a"
                      else self.algoritmo_b)
        alg_objetivo = (self.algoritmo_a if self._rol_objetivo_activo == "a"
                        else self.algoritmo_b)

        historial_fitness_obj = [s.mejor_fitness_historico for s in hist_objetivo]
        resultado_deteccion = calcular_score(
            historial_fitness=historial_fitness_obj,
            poblacion_actual=alg_objetivo.poblacion,
            limites=self.limites,
            iteracion_actual=iteracion_actual,
            max_iteraciones=self.n_iteraciones,
        )

        delta_score = (resultado_deteccion.score
                       - self._estado_transferencia.score_al_iniciar_canal_a)

        if delta_score > UMBRAL_DELTA_SCORE_SIGNIFICATIVO:
            incremento = FACTOR_INCREMENTO_RMP * delta_score
            self._estado_transferencia.rmp_actual = float(np.clip(
                self._estado_transferencia.rmp_actual + incremento,
                RMP_INICIAL, RMP_TOPE
            ))
            self._log(
                f"it={iteracion_actual:4d} | Canal A exitoso: ΔS={delta_score:.4f} "
                f"RMP→{self._estado_transferencia.rmp_actual:.3f}. "
                f"Cerrando ciclo, cooldown hasta it={iteracion_actual + self._cooldown_post_ciclo}.",
                nivel="OK",
            )
            self._conocimiento_ciclo_actual = None
            self._rol_fuente_activo = None
            self._rol_objetivo_activo = None
            self._estado_transferencia = EstadoTransferencia()
            self._iteracion_fin_cooldown = iteracion_actual + self._cooldown_post_ciclo
            self._iteracion_ultima_transferencia = iteracion_actual
        else:
            self._log(
                f"it={iteracion_actual:4d} | Canal A insuficiente (ΔS={delta_score:.4f}). "
                f"Escalando a Canal B (algoritmos siguen pausados)...",
                nivel="WARN",
            )
            self._estado_transferencia.canal_activo = CanalActivo.CANAL_B
            self._aplicar_canal_b(iteracion_actual, alg_fuente, alg_objetivo,
                                   resultado_deteccion)

    def _aplicar_canal_b(self, iteracion_actual: int,
                          alg_fuente, alg_objetivo,
                          resultado_deteccion) -> str:
        """
        Canal B: inyección puntual de instancias de élite. Los algoritmos
        permanecen pausados. Filtra por MMD, inyecta, evalúa inmediatamente
        (sin correr iteraciones adicionales).

        Tras ejecutar Canal B, el ciclo se cierra SIEMPRE — ya sea que
        haya inyectado instancias o no. Canal B es una intervención final:
        después de ella el middleware vuelve a monitoreo pasivo y los
        algoritmos corren con el estado resultante de la inyección.
        """
        aprobadas = filtrar_instancias_por_mmd(
            self._conocimiento_ciclo_actual.instancias_elite,
            alg_fuente.poblacion,
            alg_objetivo.poblacion,
            umbral_precalculado=self._conocimiento_ciclo_actual.umbral_mmd_precalculado,
        )
        instancias_filtradas = (
            self._conocimiento_ciclo_actual.instancias_elite[aprobadas])
        fitness_filtrado = (
            self._conocimiento_ciclo_actual.fitness_instancias_elite[aprobadas])

        n_aprobadas = int(aprobadas.sum())
        n_rechazadas = int((~aprobadas).sum())

        if n_aprobadas > 0:
            aplicar_canal_b(alg_objetivo, instancias_filtradas, fitness_filtrado)
            self.resumen.n_instancias_inyectadas_total += n_aprobadas

        self.resumen.n_activaciones_fase3 += 1
        self.resumen.n_transferencias_canal_b += 1
        self.resumen.log_fase3.append(asdict(EntradaLogFase3(
            iteracion=iteracion_actual,
            distancia_wasserstein=0.0,
            umbral_wasserstein=0.0,
            transferencia_permitida=True,
            canal_utilizado=CanalActivo.CANAL_B.value,
            n_instancias_aprobadas_mmd=n_aprobadas,
            n_instancias_rechazadas_mmd=n_rechazadas,
            rmp_resultante=float(self._estado_transferencia.rmp_actual),
        )))
        self._log(
            f"it={iteracion_actual:4d} | Canal B: MMD ok={n_aprobadas} "
            f"rechaz={n_rechazadas}. Evaluación inmediata (sin iterar).",
            nivel="FASE3",
        )

        # Cierre del ciclo: Canal B es siempre la última intervención.
        # Se reinicia el estado para volver a monitoreo pasivo, dejando
        # que los algoritmos corran y el Score refleje el efecto de la
        # inyección en la siguiente verificación de Fase 1.
        self._conocimiento_ciclo_actual = None
        self._rol_fuente_activo = None
        self._rol_objetivo_activo = None
        self._estado_transferencia = EstadoTransferencia()
        self._iteracion_fin_cooldown = iteracion_actual + self._cooldown_post_ciclo
        self._iteracion_ultima_transferencia = iteracion_actual
        self._log(
            f"it={iteracion_actual:4d} | Ciclo cerrado tras Canal B. "
            f"Cooldown activo hasta it={self._iteracion_fin_cooldown} "
            f"({self._cooldown_post_ciclo} iter).",
            nivel="OK",
        )
        return "canal_b_aplicado"

    # ── Máquina de estados del bucle principal ───────────────────────────────

    # Estados del bucle de control:
    # MONITOREO   → esperando que pasen frecuencia_monitoreo iteraciones
    # PAUSADO_F12 → algoritmos pausados, ejecutando Fase 1/2 + barrera
    # CANAL_A     → algoritmos corriendo con RMP aplicado, esperando
    #               que pasen ventana_canal_a iteraciones reales
    # PAUSADO_EVAL→ algoritmos pausados, evaluando ΔScore post-Canal A
    # CANAL_B     → algoritmos pausados, inyectando + evaluando MMD
    # REANUDAR    → reanudando después de completar un ciclo

    # ── Ejecución principal ──────────────────────────────────────────────────

    def ejecutar(self) -> ResumenEjecucion:
        """
        Lanza los dos hilos de algoritmo y el bucle de monitoreo del
        middleware. La pausa/reanudación de los algoritmos ahora sigue
        la máquina de estados descrita en el docstring del módulo:

        - Fase 2 completa, Wasserstein, MMD: algoritmos pausados.
        - Canal A: algoritmos se REANUDAN para correr exactamente
          ventana_canal_a iteraciones bajo el efecto del RMP, luego
          se vuelven a pausar para evaluar el ΔScore.
        - Canal B: intervención puntual, algoritmos pausados durante
          la inyección y evaluación inmediata (sin iterar adicional).
        """
        self._log(f"Iniciando ejecución paralela ({self.n_iteraciones} iteraciones, "
                  f"monitoreo cada {self.frecuencia_monitoreo} iter. | "
                  f"ventana Canal A: {self._ventana_canal_a} iter "
                  f"({FRACCION_VENTANA_CANAL_A*100:.0f}% de {self.n_iteraciones}) | "
                  f"cooldown post-ciclo: {self._cooldown_post_ciclo} iter)")
        t_inicio = time.time()

        hilo_a = HiloAlgoritmo(
            self.algoritmo_a, "A",
            self._evento_pausa, self._pausado_a,
            self.n_iteraciones,
        )
        hilo_b = HiloAlgoritmo(
            self.algoritmo_b, "B",
            self._evento_pausa, self._pausado_b,
            self.n_iteraciones,
        )
        # Guardar referencias para que _pausar_algoritmos pueda verificar
        # si cada hilo ya terminó antes de esperar su confirmación de pausa.
        self._hilo_a_ref = hilo_a
        self._hilo_b_ref = hilo_b

        hilo_a.start()
        hilo_b.start()

        iteracion_monitoreada = 0
        iteracion_inicio_canal_a_real = None  # iteración REAL donde Canal A empezó
        pendiente_registrar_inicio_canal_a = False  # flag: registrar en próxima lectura
        tiempo_en_middleware = 0.0

        while not (hilo_a.terminado and hilo_b.terminado):

            len_a = len(self.algoritmo_a.historial)
            len_b = len(self.algoritmo_b.historial)
            iteracion_actual = min(len_a, len_b)

            # Capturar la iteración real de inicio de Canal A: se registra
            # en el primer ciclo del bucle DESPUÉS de haber reanudado los
            # algoritmos, garantizando que medimos desde donde realmente
            # empezaron a correr con el RMP aplicado — no desde la iteración
            # donde se aplicó el RMP (que es anterior a la reanudación por
            # el tiempo que tardó Fase 2).
            if pendiente_registrar_inicio_canal_a:
                iteracion_inicio_canal_a_real = iteracion_actual
                pendiente_registrar_inicio_canal_a = False
                self._log(
                    f"it={iteracion_actual:4d} | Canal A: inicio real registrado "
                    f"(ventana finaliza en it≈{iteracion_actual + self._ventana_canal_a}).",
                    nivel="FASE3",
                )

            # ── ESTADO: Canal A activo ────────────────────────────────────
            # Los algoritmos están corriendo con RMP aplicado.
            # Esperamos hasta que hayan completado ventana_canal_a
            # iteraciones reales desde que se inició Canal A.
            if iteracion_inicio_canal_a_real is not None:
                iteraciones_transcurridas = (
                    iteracion_actual - iteracion_inicio_canal_a_real
                )
                if iteraciones_transcurridas < self._ventana_canal_a:
                    time.sleep(0.02)
                    continue

                # Ventana de Canal A completa → pausar y evaluar
                self._log(
                    f"it={iteracion_actual:4d} | Canal A: ventana de "
                    f"{self._ventana_canal_a} iter completada "
                    f"({iteracion_inicio_canal_a_real} → {iteracion_actual}). "
                    f"Pausando para evaluar ΔScore...",
                    nivel="FASE3",
                )
                t_pausa = time.time()
                self._pausar_algoritmos()
                iteracion_inicio_canal_a_real = None  # limpiar estado

                # Evaluar si Canal A tuvo efecto y decidir si escalar a B
                self._evaluar_post_canal_a(iteracion_actual, hilo_a, hilo_b)
                tiempo_en_middleware += time.time() - t_pausa
                iteracion_monitoreada = iteracion_actual
                self._reanudar_algoritmos()
                continue

            # ── ESTADO: Monitoreo pasivo ──────────────────────────────────
            # Esperar que pasen frecuencia_monitoreo iteraciones
            if iteracion_actual < iteracion_monitoreada + self.frecuencia_monitoreo:
                time.sleep(0.02)
                continue

            # Pausar y ejecutar Fase 1/2 + Wasserstein + primera acción
            t_pausa = time.time()
            self._pausar_algoritmos()

            resultado = self._ejecutar_ciclo_middleware(
                iteracion_actual, hilo_a, hilo_b
            )

            if resultado == "canal_a_iniciado":
                # Canal A aplicado → reanudar para que corran ventana_canal_a iter
                pendiente_registrar_inicio_canal_a = True  # registrar en próxima lectura
                tiempo_en_middleware += time.time() - t_pausa
                self._reanudar_algoritmos()
                self._log(
                    f"it={iteracion_actual:4d} | Canal A: RMP aplicado, "
                    f"algoritmos reanudados. Inicio real se registrará en "
                    f"próxima lectura del historial.",
                    nivel="FASE3",
                )
            else:
                # Cualquier otro resultado (monitoreo pasivo, Canal B,
                # abort): reanudamos normalmente
                if resultado:
                    tiempo_en_middleware += time.time() - t_pausa
                iteracion_monitoreada = iteracion_actual
                self._reanudar_algoritmos()

        hilo_a.join()
        hilo_b.join()

        t_total = time.time() - t_inicio
        self.resumen.n_iteraciones_totales = self.n_iteraciones
        self.resumen.tiempo_total_segundos = round(t_total, 3)
        self.resumen.tiempo_pausado_en_middleware_segundos = round(
            tiempo_en_middleware, 3)
        self.resumen.fitness_final_algoritmo_a = float(
            self.algoritmo_a.mejor_fitness_historico)
        self.resumen.fitness_final_algoritmo_b = float(
            self.algoritmo_b.mejor_fitness_historico)

        self._imprimir_resumen()
        self._guardar_log()
        return self.resumen

    # ── Resumen y guardado ───────────────────────────────────────────────────

    def _imprimir_resumen(self):
        r = self.resumen
        self._log("=" * 60)
        self._log("RESUMEN FINAL")
        self._log(f"  Iteraciones totales:          {r.n_iteraciones_totales}")
        self._log(f"  Ventana Canal A (10%):        {self._ventana_canal_a} iter")
        self._log(f"  Activaciones Fase 2:          {r.n_activaciones_fase2}")
        self._log(f"  Re-entrenamientos (cambio fuente): {r.n_reentrenamientos_por_cambio_fuente}")
        self._log(f"  Activaciones Fase 3:          {r.n_activaciones_fase3}")
        self._log(f"  Transferencias abortadas (W1):{r.n_transferencias_abortadas_wasserstein}")
        self._log(f"  Transferencias Canal A:       {r.n_transferencias_canal_a}")
        self._log(f"  Transferencias Canal B:       {r.n_transferencias_canal_b}")
        self._log(f"  Instancias inyectadas total:  {r.n_instancias_inyectadas_total}")
        self._log(f"  Fitness final algoritmo A:    {r.fitness_final_algoritmo_a:.4f}")
        self._log(f"  Fitness final algoritmo B:    {r.fitness_final_algoritmo_b:.4f}")
        self._log(f"  Tiempo total:                 {r.tiempo_total_segundos:.2f}s")
        self._log(f"  Tiempo pausado en middleware: {r.tiempo_pausado_en_middleware_segundos:.2f}s")
        self._log("=" * 60)

    def _guardar_log(self):
        os.makedirs(self.directorio_log, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta = os.path.join(self.directorio_log, f"{self.nombre_log}_{ts}.json")
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(asdict(self.resumen), f, indent=2, ensure_ascii=False)
        self._log(f"Log guardado en: {ruta}", nivel="OK")