"""
Interfaz común para algoritmos bioinspirados (Fase 0).

Esta clase abstracta define el contrato que todo algoritmo metaheurístico
debe cumplir para poder ser monitoreado y controlado por el middleware sin
que este conozca su lógica interna. Esto es lo que en el informe se describe
como "agnosticismo al algoritmo".

El middleware solo necesita poder leer el HISTORIAL del algoritmo (posiciones,
fitness por iteración) y, en el caso del algoritmo objetivo, poder INYECTARLE
conocimiento (parámetros o instancias). El algoritmo no necesita saber que
está siendo observado ni que el middleware existe.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EstadoIteracion:
    """
    Snapshot del estado de la población en una iteración específica.

    Es la unidad mínima de información que el algoritmo le entrega al
    middleware. El Score compuesto (Fase 1) y el modelo subrogado (Fase 2)
    se construyen a partir de una secuencia de estos snapshots.
    """
    iteracion: int
    poblacion: np.ndarray       # shape (n_individuos, n_dimensiones)
    fitness: np.ndarray         # shape (n_individuos,)
    mejor_fitness_historico: float
    mejor_individuo_historico: np.ndarray
    hiperparametros: dict       # configuración vigente del algoritmo en
                                 # esta iteración (claves específicas de
                                 # cada algoritmo concreto, ej. {"w":...,
                                 # "c1":..., "c2":...} para PSO, o
                                 # {"F":..., "CR":...} para DE). Necesario
                                 # para la Fase 2 (extracción de
                                 # parámetros de escape), que debe poder
                                 # recuperar la configuración EXACTA
                                 # vigente en el momento en que el
                                 # algoritmo fuente rompió un óptimo
                                 # local, no solo la configuración actual.


class AlgoritmoBioinspirado(ABC):
    """
    Clase base abstracta para cualquier metaheurística bioinspirada que
    participe en el esquema de transferencia (ya sea como fuente u objetivo).

    Cualquier algoritmo nuevo que se quiera integrar al middleware (PSO, DE,
    GWO, etc.) debe heredar de esta clase e implementar sus métodos abstractos.
    Esto garantiza que el middleware pueda operar sobre él sin modificaciones.
    """

    def __init__(self, funcion_objetivo, n_individuos: int, n_dimensiones: int,
                 limites: np.ndarray, max_iteraciones: int, semilla: int = None):
        """
        funcion_objetivo: callable que recibe un vector x (n_dimensiones,) y
                           retorna un escalar (fitness). Para este proyecto
                           proviene de opfunu (benchmark CEC 2022).
        limites: array (n_dimensiones, 2) con [min, max] por dimensión.
        semilla: semilla aleatoria para reproducibilidad (obligatoria en el
                 protocolo experimental CEC 2022, donde cada corrida debe ser
                 reproducible).
        """
        self.funcion_objetivo = funcion_objetivo
        self.n_individuos = n_individuos
        self.n_dimensiones = n_dimensiones
        self.limites = limites
        self.max_iteraciones = max_iteraciones
        self.rng = np.random.default_rng(semilla)

        self.iteracion_actual = 0
        self.historial: list[EstadoIteracion] = []

        self.poblacion: np.ndarray = None
        self.fitness: np.ndarray = None
        self.mejor_fitness_historico: float = np.inf
        self.mejor_individuo_historico: np.ndarray = None

    # ------------------------------------------------------------------
    # Métodos que cada algoritmo concreto (PSO, DE) DEBE implementar.
    # ------------------------------------------------------------------

    @abstractmethod
    def inicializar_poblacion(self) -> None:
        """Genera la población inicial dentro de los límites definidos."""
        raise NotImplementedError

    @abstractmethod
    def un_paso(self) -> None:
        """
        Ejecuta UNA iteración completa del algoritmo (una generación).
        Debe actualizar self.poblacion, self.fitness, mejor histórico, etc.
        No debe encargarse de registrar el historial: eso lo hace
        ejecutar_iteracion() para mantener esa responsabilidad centralizada.
        """
        raise NotImplementedError

    @abstractmethod
    def obtener_hiperparametros_actuales(self) -> dict:
        """
        Retorna un diccionario con los hiperparámetros vigentes del
        algoritmo en el instante actual (justo después de ejecutar
        un_paso()). Cada algoritmo concreto define sus propias claves
        (ej. PSO: {"w", "c1", "c2"}; DE: {"F", "CR"}), manteniendo el
        agnosticismo del middleware: este solo maneja un dict genérico,
        sin conocer la semántica interna de cada algoritmo.

        Es necesario para la Fase 2 (extracción de parámetros de
        escape), que debe poder recuperar la configuración EXACTA
        vigente en el momento en que el algoritmo fuente rompió un
        óptimo local, no solo su configuración en el instante actual.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Métodos comunes (ya implementados) que el middleware usa para leer
    # el estado del algoritmo, SIN conocer su lógica interna.
    # ------------------------------------------------------------------

    def ejecutar_iteracion(self) -> EstadoIteracion:
        """
        Ejecuta una iteración y registra el snapshot correspondiente.
        Este es el único método que el middleware invoca para avanzar
        el algoritmo un paso. Retorna el EstadoIteracion resultante.
        """
        if self.poblacion is None:
            self.inicializar_poblacion()
            self._evaluar_poblacion()
            self._actualizar_mejor_historico()
        else:
            self.un_paso()
            self._actualizar_mejor_historico()

        self.iteracion_actual += 1
        snapshot = EstadoIteracion(
            iteracion=self.iteracion_actual,
            poblacion=self.poblacion.copy(),
            fitness=self.fitness.copy(),
            mejor_fitness_historico=self.mejor_fitness_historico,
            mejor_individuo_historico=self.mejor_individuo_historico.copy(),
            hiperparametros=self.obtener_hiperparametros_actuales(),
        )
        self.historial.append(snapshot)
        return snapshot

    def obtener_historial_reciente(self, ventana: int) -> list[EstadoIteracion]:
        """
        Retorna los últimos `ventana` snapshots. Usado por el middleware
        en Fase 1 (Score compuesto, que necesita ventana de iteraciones
        recientes para calcular FIR) y Fase 2 (extracción, que necesita
        el historial completo del algoritmo fuente).
        """
        return self.historial[-ventana:]

    def aplicar_correccion_limites(self, poblacion: np.ndarray) -> np.ndarray:
        """
        Aplica clipping a los límites definidos. Método compartido porque
        tanto PSO como DE (y la inyección de instancias del middleware)
        necesitan garantizar soluciones factibles dentro de los bounds del
        benchmark CEC 2022 ([-100, 100] por dimensión).
        """
        return np.clip(poblacion, self.limites[:, 0], self.limites[:, 1])

    def _evaluar_poblacion(self) -> None:
        """Evalúa self.poblacion completa contra la función objetivo."""
        self.fitness = np.array([self.funcion_objetivo(ind) for ind in self.poblacion])

    def _actualizar_mejor_historico(self) -> None:
        idx_mejor = np.argmin(self.fitness)
        if self.fitness[idx_mejor] < self.mejor_fitness_historico:
            self.mejor_fitness_historico = float(self.fitness[idx_mejor])
            self.mejor_individuo_historico = self.poblacion[idx_mejor].copy()

    # ------------------------------------------------------------------
    # Punto de entrada para la Fase 3 (Transferencia) del middleware.
    # Por defecto NO hace nada: cada algoritmo concreto puede sobrescribir
    # este método si quiere interpretar parámetros transferidos de forma
    # específica (p.ej. DE interpretando una tasa de mutación F).
    # ------------------------------------------------------------------

    def recibir_parametros_transferidos(self, parametros: dict) -> None:
        """
        Canal A de la Fase 3: aplica parámetros transferidos desde el
        algoritmo fuente. La implementación base no hace nada; los
        algoritmos concretos deben sobrescribir este método para mapear
        los parámetros recibidos a sus propios hiperparámetros internos.
        """
        pass

    def recibir_instancias_elite(self, instancias: np.ndarray, fitness_instancias: np.ndarray) -> None:
        """
        Canal B de la Fase 3: sustituye los peores individuos de la
        población actual por las instancias de élite recibidas del
        algoritmo fuente. Implementación genérica válida para cualquier
        algoritmo poblacional (no necesita sobrescritura en la mayoría
        de los casos).
        """
        n_inyectar = len(instancias)
        if n_inyectar == 0:
            return
        idx_peores = np.argsort(self.fitness)[-n_inyectar:]
        instancias_corregidas = self.aplicar_correccion_limites(instancias)
        self.poblacion[idx_peores] = instancias_corregidas
        self.fitness[idx_peores] = fitness_instancias
        self._actualizar_mejor_historico()