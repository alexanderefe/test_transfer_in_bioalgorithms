"""
CLONALG — algoritmo de selección clonal (de Castro y Von Zuben, IEEE TEC
2002), familia de sistemas inmunes artificiales, sobre `aisp` (`csa.Clonalg`).

Se eligió aisp porque las versiones de NiaPy y pyMetaheuristic tienen
errores en la tasa de hipermutación y rinden peor que una búsqueda
aleatoria en CEC 2022 (docs/competidores.md §8).

Lo que hace aisp en cada iteración (versión de optimización de CLONALG):
  - selecciona los n mejores anticuerpos;
  - clona cada uno en proporción a su afinidad normalizada en [0, 1]
    (máx(1, ⌊β·afinidad⌋) clones);
  - hipermuta cada clon con tasa exp(−ρ·afinidad) por variable
    (reinicio uniforme de la variable en su rango): los mejores mutan menos;
  - se queda con los N − d mejores entre seleccionados y clones, y agrega
    d anticuerpos aleatorios (diversidad).

Parámetros: los defaults de aisp 0.5.5, sin re-tuning: N = 50, n = 5,
β (`rate_clonal`) = 10, ρ (`rate_hypermutation`) = 1, d = 5. Diferencia con
el paper: el número de clones es lineal en la afinidad normalizada (paper:
según el ranking, round(β·N/i)).

Cambios respecto del uso directo de aisp:
  - Sin detención temprana: `n_iter_no_change` y `max_iters` muy altos; la
    corrida termina solo por presupuesto (`Presupuesto.evaluar`).
  - Sin historial: `Clonalg._record_best` agrega la mejor solución y su
    costo a dos listas en cada iteración (más de un millón de vectores con
    MaxFES = 5·10⁷); aquí solo se guarda el mejor.

Determinismo: aisp siembra `np.random` global y el RNG de Numba con
`seed` (sus mutaciones están compiladas con Numba).
"""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
from aisp.csa import Clonalg

from experimentos.competidores.base import Presupuesto, PresupuestoAgotado

VERSION = f"aisp-{version('aisp')}"
_SIN_TOPE = 10**12


class _ClonalgSinHistorial(Clonalg):
    def _record_best(self, cost, best_solution):
        if self._best_cost is None or cost < self._best_cost:
            self._best_cost = cost
            self._best_solution = best_solution


def correr(problema, max_fes: int, semilla: int) -> None:
    pres = Presupuesto(problema, max_fes)
    lb = problema.limites[:, 0].astype(float)
    ub = problema.limites[:, 1].astype(float)
    algo = _ClonalgSinHistorial(
        problem_size=len(lb),
        affinity_function=pres.evaluar,
        bounds={"low": lb.tolist(), "high": ub.tolist()},
        mode="min",
        seed=int(semilla) % 2**32,
    )
    try:
        algo.optimize(max_iters=_SIN_TOPE, n_iter_no_change=_SIN_TOPE, verbose=False)
    except PresupuestoAgotado:
        return
    raise RuntimeError("CLONALG terminó sin agotar MaxFES")
