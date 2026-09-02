"""
DE (Differential Evolution) — Algoritmo OBJETIVO.

Se usa como algoritmo objetivo porque, en su variante clásica DE/rand/1/bin
con parámetros fijos y conservadores (F y CR bajos), tiende a perder
diversidad poblacional rápidamente y estancarse en óptimos locales en
funciones multimodales — exactamente el comportamiento patológico que el
middleware debe detectar y corregir.

Esta es una elección deliberada: NO se implementa una variante adaptativa
de DE (como SHADE o L-SHADE) porque el objetivo experimental es que el
algoritmo objetivo presente estancamiento real y observable, para poder
medir si el middleware logra desbloquearlo. Implementación basada en
Storn & Price (1997).
"""

import numpy as np
from bioalgorithms.base import AlgoritmoBioinspirado


class DE(AlgoritmoBioinspirado):

    def __init__(self, funcion_objetivo, n_individuos, n_dimensiones, limites,
                 max_iteraciones=None, semilla=None, F=0.5, CR=0.7,
                 *, max_fes=None, fraccion_presupuesto=0.7):
        super().__init__(funcion_objetivo, n_individuos, n_dimensiones,
                          limites, max_iteraciones, semilla,
                          max_fes=max_fes,
                          fraccion_presupuesto=fraccion_presupuesto)
        self.F = F     # factor de escala de la mutación diferencial
        self.CR = CR   # probabilidad de cruce (recombinación binomial)

        # Valores conservadores fijos por defecto: F y CR bajos favorecen
        # la explotación sobre la exploración, lo que en funciones
        # multimodales del CEC 2022 produce convergencia prematura.

    def inicializar_poblacion(self) -> None:
        rango = self.limites[:, 1] - self.limites[:, 0]
        self.poblacion = self.limites[:, 0] + self.rng.random(
            (self.n_individuos, self.n_dimensiones)) * rango

    def un_paso(self) -> None:
        nueva_poblacion = self.poblacion.copy()
        nuevo_fitness = self.fitness.copy()

        for i in range(self.n_individuos):
            candidatos = [idx for idx in range(self.n_individuos) if idx != i]
            r1, r2, r3 = self.rng.choice(candidatos, size=3, replace=False)

            # Mutación: DE/rand/1
            vector_mutado = (
                self.poblacion[r1]
                + self.F * (self.poblacion[r2] - self.poblacion[r3])
            )
            vector_mutado = self.aplicar_correccion_limites(
                vector_mutado.reshape(1, -1)
            ).flatten()

            # Cruce binomial
            mascara_cruce = self.rng.random(self.n_dimensiones) < self.CR
            # Garantiza al menos una dimensión del mutado (regla estándar DE)
            j_rand = self.rng.integers(0, self.n_dimensiones)
            mascara_cruce[j_rand] = True

            vector_prueba = np.where(mascara_cruce, vector_mutado, self.poblacion[i])

            fitness_prueba = self.funcion_objetivo(vector_prueba)

            # Selección: el hijo reemplaza al padre solo si es mejor o igual.
            if fitness_prueba <= self.fitness[i]:
                nueva_poblacion[i] = vector_prueba
                nuevo_fitness[i] = fitness_prueba

        self.poblacion = nueva_poblacion
        self.fitness = nuevo_fitness

    def obtener_hiperparametros_actuales(self) -> dict:
        """
        Hiperparámetros vigentes de DE: F (factor de escala) y CR
        (probabilidad de cruce). En esta implementación son constantes
        durante la ejecución, salvo que el middleware los modifique
        mediante recibir_parametros_transferidos() (Canal A, Fase 3).
        """
        return {"F": self.F, "CR": self.CR}

    # ------------------------------------------------------------------
    # Canal A de la Fase 3: DE interpreta los parámetros transferidos
    # ajustando F y CR. Este es el mecanismo de "baja invasividad" que
    # describe el documento de la tesis (RMP-like, aplicado aquí como
    # ajuste directo de hiperparámetros mientras se define el mapeo
    # formal vía RMP en la Fase 3 del middleware).
    # ------------------------------------------------------------------
    def recibir_parametros_transferidos(self, parametros: dict) -> None:
        if "F" in parametros:
            self.F = float(np.clip(parametros["F"], 0.0, 2.0))
        if "CR" in parametros:
            self.CR = float(np.clip(parametros["CR"], 0.0, 1.0))