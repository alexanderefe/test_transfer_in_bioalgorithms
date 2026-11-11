"""
GWO — Grey Wolf Optimizer (Mirjalili, Mirjalili y Lewis, Advances in
Engineering Software 2014), familia de enjambre.

`GreyWolfOptimizer` de NiaPy 2.7.1, revisado contra el código MATLAB
original: la posición de cada lobo es el promedio de X₁, X₂, X₃ guiados por
α, β y δ, con A = 2a·r₁ − a y C = 2·r₂ por dimensión; los líderes se
actualizan igual que en el original (sin desplazar α → β → δ cuando aparece
un nuevo α). En NiaPy `a` baja linealmente de 2 a 0 según las evaluaciones
consumidas (en el original, según las iteraciones; con población fija es
equivalente).

Parámetros: 30 lobos (los experimentos del paper). Límites: recorte, como
el original.
"""

from __future__ import annotations

from niapy.algorithms.basic import GreyWolfOptimizer

from experimentos.competidores.adaptador_niapy import correr_niapy, version_niapy

VERSION = f"{version_niapy()}+gwo-mirjalili2014"
LOBOS = 30


def correr(problema, max_fes: int, semilla: int) -> None:
    correr_niapy(GreyWolfOptimizer, problema, max_fes, semilla, population_size=LOBOS)
