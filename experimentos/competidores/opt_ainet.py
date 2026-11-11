"""
opt-aiNet — red inmune artificial para optimización (de Castro y Timmis,
CEC 2002), con las correcciones de Timmis y Edmonds (GECCO 2004, «A comment
on opt-aiNET»); familia de sistemas inmunes artificiales.

No hay implementación en librerías de Python. Se implementa desde Timmis y
Edmonds (2004), que describe el algoritmo, revisó el código MATLAB original
de los autores y publica sus parámetros:

  1. Población inicial aleatoria de N células.
  2. Mientras no se cumpla el criterio de parada:
     I-V. Cada célula genera Nc clones; cada clon muta según
          c' = c + α·N(0, 1),  α = (1/β)·exp(−f*),
          con f* la fitness de la célula padre normalizada a [0, 1] en la
          red actual (1 = la mejor). Un clon fuera del dominio se descarta
          (no se evalúa). De la célula y sus clones sobrevive el mejor
          (elitista).
     VI. Cada 5 iteraciones (constante del código original) se compara la
          fitness media con la de 5 iteraciones antes; si se estabilizó
          (diferencia relativa < 0,0001), se pasa a 3 y 4.
  3. Supresión: se eliminan las células a distancia euclidiana < σs de una
     célula mejor.
  4. Se agrega un d % de células aleatorias (sobre el tamaño de la red).

Correcciones de Timmis y Edmonds que se aplican:
  - Criterio de estabilidad: el código original usaba
    |1 − avefitold/avefit| < 0,0001 suponiendo valores positivos y
    crecientes; aquí |avefit − avefitold| < 0,0001·|avefit|, válido para
    cualquier signo.
  - Supresión elitista: el original suprimía la primera célula del par sin
    mirar su fitness; aquí se ordena por fitness y siempre se elimina la
    peor.

Parámetros (los del paper original, tabla de Timmis y Edmonds): N = 20,
Nc = 10, β = 100, σs = 0,2, d = 40 %.

Adaptación al dominio: esos valores están definidos para funciones en
[−2, 2] (ancho 4). En CEC 2022 ([−100, 100], ancho 200) el algoritmo se
aplica en coordenadas reescaladas al ancho original: el paso de mutación y
el umbral de supresión se multiplican por (ub − lb)/4 en cada dimensión. En
[−2, 2] el factor es 1 y el algoritmo es el publicado (así se valida en
tests/test_opt_ainet_reproduccion.py).

Criterio de parada: solo el presupuesto. El original terminaba si una
supresión no eliminaba ninguna célula o a las 500 iteraciones; aquí la red
sigue iterando hasta agotar MaxFES.

Determinismo: `numpy.random.Generator` sembrado con la semilla de la corrida.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from experimentos.competidores.base import Presupuesto, PresupuestoAgotado

VERSION = "impl-opt-aiNet@deCastroTimmis2002+TimmisEdmonds2004"
N_INICIAL = 20
N_CLONES = 10
BETA = 100.0
SIGMA_S = 0.2
FRACCION_ALEATORIAS = 0.4
CADA_ITERACIONES = 5
TOLERANCIA_ESTABILIDAD = 1e-4
ANCHO_ORIGINAL = 4.0


class OptAiNet:
    """opt-aiNet para minimización. `evaluar(x) -> float`; los parámetros
    por defecto son los del paper."""

    def __init__(self, evaluar: Callable[[np.ndarray], float], lb, ub,
                 rng: np.random.Generator, n_inicial=N_INICIAL, n_clones=N_CLONES,
                 beta=BETA, sigma_s=SIGMA_S, fraccion_aleatorias=FRACCION_ALEATORIAS):
        self.evaluar = evaluar
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.rng = rng
        self.n_clones = n_clones
        self.beta = beta
        self.sigma_s = sigma_s
        self.fraccion_aleatorias = fraccion_aleatorias
        self.escala = (self.ub - self.lb) / ANCHO_ORIGINAL
        self.iteracion = 0
        self._media_previa = None
        self.X = np.empty((0, len(self.lb)))
        self.f = np.empty(0)
        self._agregar_aleatorias(n_inicial)

    def _agregar_aleatorias(self, n: int) -> None:
        nuevas = self.lb + (self.ub - self.lb) * self.rng.random((n, len(self.lb)))
        f_nuevas = np.array([self.evaluar(x) for x in nuevas])
        self.X = np.vstack([self.X, nuevas])
        self.f = np.concatenate([self.f, f_nuevas])

    def _fitness_normalizada(self) -> np.ndarray:
        rango = self.f.max() - self.f.min()
        if rango == 0 or not np.isfinite(rango):
            return np.ones_like(self.f)
        return (self.f.max() - self.f) / rango          # 1 = mejor, 0 = peor

    def paso_clonal(self) -> None:
        """Pasos I-V: clonación, hipermutación y selección del mejor clon."""
        alfa = (1.0 / self.beta) * np.exp(-self._fitness_normalizada())
        for i in range(len(self.f)):
            clones = (self.X[i] + alfa[i] * self.escala
                      * self.rng.standard_normal((self.n_clones, len(self.lb))))
            validos = np.all((clones >= self.lb) & (clones <= self.ub), axis=1)
            for c in clones[validos]:
                fc = self.evaluar(c)
                if fc < self.f[i]:
                    self.X[i], self.f[i] = c, fc
        self.iteracion += 1

    def estabilizada(self) -> bool:
        """Paso VI, cada CADA_ITERACIONES iteraciones (corrección de signo
        de Timmis y Edmonds)."""
        if self.iteracion % CADA_ITERACIONES:
            return False
        media = float(np.mean(self.f))
        previa, self._media_previa = self._media_previa, media
        if previa is None:
            return False
        return abs(media - previa) < TOLERANCIA_ESTABILIDAD * abs(media) or media == previa

    def suprimir(self) -> int:
        """Paso 3, elitista: recorre por fitness y elimina toda célula a
        distancia < σs de una célula mejor ya conservada."""
        # Misma semántica que recorrer por fitness y conservar una célula si
        # está a ≥ σs de todas las conservadas, pero con KD-tree: la red
        # puede llegar a decenas de miles de células con MaxFES grandes.
        orden = np.argsort(self.f, kind="stable")
        Z = self.X / self.escala
        arbol = cKDTree(Z)
        eliminada = np.zeros(len(self.f), dtype=bool)
        conservadas = []
        for i in orden:
            if eliminada[i]:
                continue
            conservadas.append(i)
            for j in arbol.query_ball_point(Z[i], r=self.sigma_s):
                if j != i and np.linalg.norm(Z[j] - Z[i]) < self.sigma_s:
                    eliminada[j] = True
        eliminadas = len(self.f) - len(conservadas)
        self.X, self.f = self.X[conservadas], self.f[conservadas]
        return eliminadas

    def insertar_aleatorias(self) -> None:
        """Paso 4: d % de células nuevas sobre el tamaño actual de la red."""
        self._agregar_aleatorias(max(1, math.ceil(self.fraccion_aleatorias * len(self.f))))

    def iterar(self) -> int | None:
        """Una iteración completa. Devuelve None si la red no se estabilizó;
        si se estabilizó, el número de células suprimidas (el original
        terminaba cuando era 0)."""
        self.paso_clonal()
        if self.estabilizada():
            eliminadas = self.suprimir()
            self.tamano_tras_supresion = len(self.f)
            self.insertar_aleatorias()
            self._media_previa = None
            return eliminadas
        return None


def correr(problema, max_fes: int, semilla: int) -> None:
    pres = Presupuesto(problema, max_fes)
    rng = np.random.default_rng(semilla)
    try:
        red = OptAiNet(pres.evaluar, problema.limites[:, 0], problema.limites[:, 1], rng)
        while True:
            red.iterar()
    except PresupuestoAgotado:
        return
