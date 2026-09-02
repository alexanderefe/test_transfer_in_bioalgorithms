"""
PSO (Particle Swarm Optimization) — Algoritmo FUENTE.

Se usa como algoritmo fuente porque es una metaheurística consolidada,
ampliamente validada en la literatura, con buen desempeño general en
benchmarks de optimización continua. El middleware extraerá conocimiento
desde este algoritmo cuando el algoritmo objetivo (DE) se estanque.

Implementación estándar de PSO con coeficientes de inercia, cognitivo y
social (Kennedy & Eberhart, 1995; variante con inercia de Shi & Eberhart).
"""

import numpy as np
from bioalgorithms.base import AlgoritmoBioinspirado


class PSO(AlgoritmoBioinspirado):

    def __init__(self, funcion_objetivo, n_individuos, n_dimensiones, limites,
                 max_iteraciones=None, semilla=None,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0,
                 *, max_fes=None, fraccion_presupuesto=0.7):
        super().__init__(funcion_objetivo, n_individuos, n_dimensiones,
                          limites, max_iteraciones, semilla,
                          max_fes=max_fes,
                          fraccion_presupuesto=fraccion_presupuesto)
        self.w_max = w_max          # inercia inicial (favorece exploración)
        self.w_min = w_min          # inercia final (favorece explotación)
        self.c1 = c1                # coeficiente cognitivo (atracción a pbest)
        self.c2 = c2                # coeficiente social (atracción a gbest)

        # Inicializada en w_max: antes de la primera llamada a un_paso()
        # (es decir, durante la iteración de inicialización de población),
        # la inercia vigente es la inicial por definición del esquema de
        # decrecimiento lineal.
        self.w_actual = w_max

        self.velocidades: np.ndarray = None
        self.pbest: np.ndarray = None
        self.pbest_fitness: np.ndarray = None

    def inicializar_poblacion(self) -> None:
        rango = self.limites[:, 1] - self.limites[:, 0]
        self.poblacion = self.limites[:, 0] + self.rng.random(
            (self.n_individuos, self.n_dimensiones)) * rango

        # Velocidad inicial acotada a un 10% del rango, práctica estándar
        # para evitar que las partículas salgan disparadas del espacio
        # de búsqueda en las primeras iteraciones.
        self.velocidades = (self.rng.random(
            (self.n_individuos, self.n_dimensiones)) - 0.5) * 0.1 * rango

        self.pbest = self.poblacion.copy()

    def un_paso(self) -> None:
        # Inercia lineal decreciente: empieza explorando, termina explotando.
        # Se guarda como self.w_actual (no solo variable local) para que
        # obtener_hiperparametros_actuales() pueda exponerla al middleware
        # en cada snapshot (necesario para la Fase 2, extracción de
        # parámetros de escape).
        #
        # El "progreso" [0, 1] se mide contra el presupuesto que corresponde
        # a este algoritmo:
        #   - Vía MaxFES: contra su CUOTA estimada del presupuesto compartido
        #     (max_fes * fraccion_presupuesto). El setup experimental exige
        #     que la inercia decreciente se calcule contra el MaxFES total del
        #     experimento; como aquí dos algoritmos comparten ese total, se usa
        #     la fracción que se estima consumirá PSO (~0.7 por defecto: el
        #     reparto medido PSO/DE fue ~70/30; decisión provisional, ver
        #     docs/configuracion_experimental.md).
        #   - Vía clásica: contra el número total de generaciones.
        if self.max_fes is not None:
            presupuesto_propio = max(self.max_fes * self.fraccion_presupuesto, 1)
            progreso = min(self.fes_propias / presupuesto_propio, 1.0)
        else:
            progreso = self.iteracion_actual / max(self.max_iteraciones or 1, 1)
        self.w_actual = self.w_max - (self.w_max - self.w_min) * progreso

        r1 = self.rng.random((self.n_individuos, self.n_dimensiones))
        r2 = self.rng.random((self.n_individuos, self.n_dimensiones))

        gbest = self.mejor_individuo_historico

        self.velocidades = (
            self.w_actual * self.velocidades
            + self.c1 * r1 * (self.pbest - self.poblacion)
            + self.c2 * r2 * (gbest - self.poblacion)
        )

        self.poblacion = self.aplicar_correccion_limites(
            self.poblacion + self.velocidades
        )

        self._evaluar_poblacion()
        self._actualizar_pbest()

    def _actualizar_pbest(self) -> None:
        if self.pbest_fitness is None:
            self.pbest_fitness = self.fitness.copy()
            return
        mejora = self.fitness < self.pbest_fitness
        self.pbest[mejora] = self.poblacion[mejora]
        self.pbest_fitness[mejora] = self.fitness[mejora]

    def _evaluar_poblacion(self) -> None:
        super()._evaluar_poblacion()
        if self.pbest_fitness is None:
            self.pbest_fitness = self.fitness.copy()

    def obtener_hiperparametros_actuales(self) -> dict:
        """
        Hiperparámetros vigentes de PSO: inercia actual (ya decreciente
        según el esquema lineal implementado en un_paso()), y los
        coeficientes cognitivo/social (constantes durante la ejecución
        en esta implementación).
        """
        return {"w": self.w_actual, "c1": self.c1, "c2": self.c2}

    # ------------------------------------------------------------------
    # Sobrescritura del canal A (Fase 3) para PSO: si PSO fuera alguna vez
    # algoritmo objetivo (no es el caso en esta tesis, pero se deja
    # implementado por completitud y extensibilidad futura), interpretaría
    # los parámetros transferidos como ajustes a w, c1 o c2.
    # ------------------------------------------------------------------
    def recibir_parametros_transferidos(self, parametros: dict) -> None:
        if "w_max" in parametros:
            self.w_max = float(parametros["w_max"])
        if "w_min" in parametros:
            self.w_min = float(parametros["w_min"])
        if "c1" in parametros:
            self.c1 = float(parametros["c1"])
        if "c2" in parametros:
            self.c2 = float(parametros["c2"])