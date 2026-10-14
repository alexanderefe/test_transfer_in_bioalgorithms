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
from bioalgorithms.base import AlgoritmoBioinspirado, DIM_ADOPCION, DIM_EXPLORACION


class PSO(AlgoritmoBioinspirado):

    # Rango de referencia de la inercia para el espacio común del Canal A
    # (D-10): el calendario lineal clásico 0.9 → 0.4 (Shi & Eberhart, 1998),
    # el mismo de la configuración congelada. w = 0.4 ↔ exploración 0,
    # w = 0.9 ↔ exploración 1.
    W_REF = (0.4, 0.9)
    # Cotas del calendario tras un desplazamiento por Canal A: w ≥ 0 y
    # w ≤ 1 (con w > 1 las velocidades crecen sin límite).
    W_COTAS = (0.0, 1.0)

    def __init__(self, funcion_objetivo, n_individuos, n_dimensiones, limites,
                 max_iteraciones=None, semilla=None,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0,
                 *, max_fes=None, fraccion_presupuesto=0.5):
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
        #   - Vía MaxFES: contra su CUOTA del presupuesto compartido
        #     (max_fes * fraccion_presupuesto). El setup experimental exige
        #     que la inercia decreciente se calcule contra el MaxFES total del
        #     experimento; como aquí dos algoritmos comparten ese total, se usa
        #     la fracción que le corresponde a PSO (0.5 por defecto: con el
        #     planificador cooperativo determinista — D-2 — A y B alternan
        #     generación por generación, así que con poblaciones iguales el
        #     reparto real es EXACTAMENTE 50/50, no una estimación).
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
    # Canal A (Fase 3). Con roles dinámicos PSO puede ser objetivo (de DE o
    # de otro PSO), así que interpreta los parámetros transferidos.
    # ------------------------------------------------------------------
    def recibir_parametros_transferidos(self, parametros: dict) -> None:
        if "w" in parametros:
            # D-10: la inercia vigente la fija el calendario lineal en cada
            # paso, así que asignar w directamente se perdería en la
            # generación siguiente. Se DESPLAZA el calendario completo por
            # Δ = w_transferida − w_actual: sigue decreciendo con la misma
            # pendiente, en el nuevo nivel (cambio persistente, igual que
            # F/CR en DE). Cada extremo se acota a W_COTAS.
            delta = float(parametros["w"]) - self.w_actual
            lo, hi = self.W_COTAS
            self.w_max = float(np.clip(self.w_max + delta, lo, hi))
            self.w_min = float(np.clip(self.w_min + delta, lo, hi))
            self.w_actual = float(np.clip(self.w_actual + delta, lo, hi))
        if "w_max" in parametros:
            self.w_max = float(parametros["w_max"])
        if "w_min" in parametros:
            self.w_min = float(parametros["w_min"])
        if "c1" in parametros:
            self.c1 = float(parametros["c1"])
        if "c2" in parametros:
            self.c2 = float(parametros["c2"])

    def a_espacio_comun(self, hiperparametros: dict) -> dict:
        """
        Espacio común (D-10): exploración = w normalizada contra W_REF
        (afín, sin recorte); adopción = peso social relativo c2/(c1+c2).
        """
        latente = {}
        if "w" in hiperparametros:
            lo, hi = self.W_REF
            latente[DIM_EXPLORACION] = (float(hiperparametros["w"]) - lo) / (hi - lo)
        if "c1" in hiperparametros and "c2" in hiperparametros:
            suma = float(hiperparametros["c1"]) + float(hiperparametros["c2"])
            if suma > 0:
                latente[DIM_ADOPCION] = float(hiperparametros["c2"]) / suma
        return latente

    def desde_espacio_comun(self, latente: dict) -> dict:
        """
        Inversa: exploración → w (luego aplicada como desplazamiento del
        calendario); adopción → reparto de c1 + c2 entre ambos
        coeficientes, conservando su suma actual (la intensidad total de
        atracción no cambia, solo cuánto pesa lo social frente a lo propio).
        """
        hiperparametros = {}
        if DIM_EXPLORACION in latente:
            lo, hi = self.W_REF
            hiperparametros["w"] = lo + float(latente[DIM_EXPLORACION]) * (hi - lo)
        if DIM_ADOPCION in latente:
            suma = self.c1 + self.c2
            adopcion = float(np.clip(latente[DIM_ADOPCION], 0.0, 1.0))
            hiperparametros["c1"] = (1.0 - adopcion) * suma
            hiperparametros["c2"] = adopcion * suma
        return hiperparametros