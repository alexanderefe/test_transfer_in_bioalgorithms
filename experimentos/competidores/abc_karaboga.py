"""
ABC — Artificial Bee Colony (Karaboga 2005; Karaboga y Basturk 2007),
familia de enjambre.

`ArtificialBeeColonyAlgorithm` de NiaPy 2.7.1 corregido para reproducir el
código oficial de los autores (`ABC.C`, abc.erciyes.edu.tr/pub/ABC.C,
Copyright 2009 Erciyes University). Se mantiene la infraestructura de NiaPy
(Task, RNG, Individual) y se reescriben `init_population` y
`run_iteration`.

Errores y desviaciones de NiaPy que se corrigen:
  1. Fase de observadoras: el chequeo de exploradora usaba `trials[i]`
     (índice que sobra del ciclo de empleadas) en lugar de `trials[s]`.
  2. `init_population` creaba NP fuentes pero solo usaba NP/2: gastaba
     NP/2 evaluaciones en fuentes que nunca participaban.
  3. En la fase de empleadas el vecino k no excluía a i (ABC.C sí).
  4. La exploradora se aplicaba dentro de la fase de empleadas, para
     cualquier fuente; ABC.C aplica UNA exploradora por ciclo, al final, a
     la fuente con más intentos fallidos si llega a `limit`.
  5. Probabilidades 1/(f + 0,01) normalizadas por la suma; ABC.C usa la
     fitness fit = 1/(1 + f) (f ≥ 0) o 1 + |f| (f < 0) y
     p_i = 0,9·fit_i / max(fit) + 0,1.
  6. La selección codiciosa de NiaPy compara f; ABC.C compara fitness
     (equivalente para f ≥ 0, que es el caso de CEC 2022).

Como en ABC.C: φ ∈ [−1, 1) sobre UNA dimensión elegida al azar, recorte al
límite de la dimensión modificada, y las probabilidades se calculan una
vez por ciclo, después de la fase de empleadas.

Parámetros:
  - Tamaño de colonia NP = 40 (ABC.C), es decir, 20 fuentes de alimento.
  - limit = SN·D (Karaboga y Akay 2009; Karaboga y Basturk 2008). ABC.C
    trae limit = 100 para su ejemplo en D = 50; la regla SN·D es la que
    recomiendan los autores para elegirlo según el problema.
"""

from __future__ import annotations

import numpy as np
from niapy.algorithms.algorithm import default_individual_init
from niapy.algorithms.basic.abc import ArtificialBeeColonyAlgorithm, SolutionABC

from experimentos.competidores.adaptador_niapy import correr_niapy, version_niapy

VERSION = f"{version_niapy()}+abc-karaboga-ABC.C"
COLONIA = 40


def fitness_abc(f: float) -> float:
    """CalculateFitness de ABC.C."""
    return 1.0 / (f + 1.0) if f >= 0 else 1.0 + abs(f)


class ABCKaraboga(ArtificialBeeColonyAlgorithm):
    """ABC de NiaPy corregido según ABC.C (ver docstring del módulo)."""

    Name = ["ABCKaraboga"]

    def __init__(self, population_size=COLONIA, limit=None, *args, **kwargs):
        super().__init__(population_size=population_size, limit=limit, *args, **kwargs)

    def init_population(self, task):
        foods, fpop = default_individual_init(task=task, population_size=self.food_number,
                                              rng=self.rng, individual_type=SolutionABC)
        self._limite = (self.limit if self.limit is not None
                        else self.food_number * task.dimension)
        trials = np.zeros(self.food_number, dtype=np.int64)
        fit = np.array([fitness_abc(x.f) for x in foods])
        return foods, fpop, {"trials": trials, "fit": fit}

    def _explotar(self, foods, fit, trials, i, task):
        """Vecino de la fuente i en una dimensión + selección codiciosa
        (cuerpo común de SendEmployedBees y SendOnlookerBees)."""
        D, sn = task.dimension, self.food_number
        j = int(self.random() * D)
        k = int(self.random() * sn)
        while k == i:
            k = int(self.random() * sn)
        sol = foods[i].x.copy()
        phi = (self.random() - 0.5) * 2.0
        sol[j] = foods[i].x[j] + (foods[i].x[j] - foods[k].x[j]) * phi
        sol[j] = min(max(sol[j], task.lower[j]), task.upper[j])
        f_sol = task.eval(sol)
        fit_sol = fitness_abc(f_sol)
        if fit_sol > fit[i]:
            foods[i].x, foods[i].f, fit[i] = sol, f_sol, fit_sol
            trials[i] = 0
        else:
            trials[i] += 1

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        trials, fit = params.pop("trials"), params.pop("fit")
        sn = self.food_number

        for i in range(sn):                                  # SendEmployedBees
            self._explotar(population, fit, trials, i, task)

        prob = 0.9 * (fit / np.max(fit)) + 0.1               # CalculateProbabilities

        i = t = 0                                            # SendOnlookerBees
        while t < sn:
            if self.random() < prob[i]:
                t += 1
                self._explotar(population, fit, trials, i, task)
            i = (i + 1) % sn

        fpop = np.array([x.f for x in population])
        best_x, best_fitness = self.get_best(population, fpop, best_x, best_fitness)

        idx = int(np.argmax(trials))                         # SendScoutBees (una)
        if trials[idx] >= self._limite:
            population[idx] = SolutionABC(task=task, rng=self.rng)
            fit[idx] = fitness_abc(population[idx].f)
            trials[idx] = 0
            fpop[idx] = population[idx].f
            best_x, best_fitness = self.get_best(population, fpop, best_x, best_fitness)

        return population, fpop, best_x, best_fitness, {"trials": trials, "fit": fit}


def correr(problema, max_fes: int, semilla: int) -> None:
    correr_niapy(ABCKaraboga, problema, max_fes, semilla)
