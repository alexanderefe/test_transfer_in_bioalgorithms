"""
jDE — DE autoadaptativo (Brest, Greiner, Bošković, Mernik y Žumer, IEEE
TEC 2006), familia evolutiva.

`SelfAdaptiveDifferentialEvolution` de NiaPy 2.7.1 (lógica revisada y
fiel: DE/rand/1/bin con F y CR propios de cada individuo, regenerados con
probabilidad τ₁ y τ₂ antes de crear el vector de prueba), con los
parámetros del paper en lugar de los defaults de NiaPy:

  | Parámetro    | Paper          | Default NiaPy |
  |--------------|----------------|---------------|
  | NP           | 100            | 50            |
  | τ₁, τ₂       | 0,1 y 0,1      | 0,4 y 0,2     |
  | F nuevo      | 0,1 + 0,9·rand | 0 + 1·rand    |
  | F₀, CR₀      | 0,5 y 0,9      | 2 y 0,5       |

Además, la selección del paper es «u reemplaza a x si f(u) ≤ f(x)»; NiaPy
usa `<` estricto, y se corrige.
Límites: recorte (default de NiaPy; el paper no lo especifica).
"""

from __future__ import annotations

from functools import partial

import numpy as np
from niapy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from niapy.algorithms.modified.jde import SolutionJDE
from niapy.util import objects_to_array

from experimentos.competidores.adaptador_niapy import correr_niapy, version_niapy

VERSION = f"{version_niapy()}+jde-brest2006"
NP = 100
F_INICIAL, CR_INICIAL = 0.5, 0.9
F_L, F_U = 0.1, 0.9          # F = F_l + rand·F_u  →  [0,1, 1,0]
TAU1 = TAU2 = 0.1


class JDEBrest(SelfAdaptiveDifferentialEvolution):
    Name = ["JDEBrest"]

    def selection(self, population, new_population, best_x, best_fitness, task, **kwargs):
        arr = objects_to_array([e if e.f <= population[i].f else population[i]
                                for i, e in enumerate(new_population)])
        best_x, best_fitness = self.get_best(arr, np.asarray([e.f for e in arr]),
                                             best_x, best_fitness)
        return arr, best_x, best_fitness


def correr(problema, max_fes: int, semilla: int) -> None:
    correr_niapy(
        JDEBrest, problema, max_fes, semilla,
        population_size=NP,
        f_lower=F_L, f_upper=F_L + F_U,       # NiaPy: F = f_lower + rand·(f_upper − f_lower)
        tao1=TAU1, tao2=TAU2,
        individual_type=partial(SolutionJDE, differential_weight=F_INICIAL,
                                crossover_probability=CR_INICIAL),
    )
