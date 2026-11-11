"""
GA de codificación real (familia evolutiva) sobre `pymoo`
(Blank y Deb, 2020): el GA canónico para variables continuas de la línea
de Deb, con cruce SBX y mutación polinomial.

Se eligió pymoo porque el GA de NiaPy tiene errores que lo convierten en
otro algoritmo y el `BaseGA` de mealpy no es el GA real canónico
(docs/competidores.md §8).

Parámetros: los defaults de `pymoo.algorithms.soo.nonconvex.ga.GA`, sin
re-tuning:
  - Población: 100, muestreo uniforme en el dominio.
  - Selección: torneo binario por fitness.
  - Cruce: SBX, probabilidad 0,9 por pareja, η_c = 15, probabilidad 0,5
    por variable.
  - Mutación: polinomial, η_m = 20, probabilidad 1/D por variable.
  - Supervivencia: (μ + λ) elitista por fitness; se eliminan duplicados.

Diferencias con `pymoo.optimize.minimize`:
  - Terminación: solo por presupuesto. El GA de pymoo trae por defecto
    `DefaultSingleObjectiveTermination` (tolerancias de f y x), que
    detendría la corrida antes de MaxFES; se reemplaza por ("n_eval", ·)
    y el corte exacto lo hace `Presupuesto.evaluar` (lanza
    `PresupuestoAgotado` a mitad de generación si hace falta).
  - Sin historial (`save_history=False`, default) ni salida en pantalla.

Determinismo: `setup(seed=semilla)` crea el `numpy.random.Generator` que
pymoo pasa a todos sus operadores.
"""

from __future__ import annotations

import numpy as np
import pymoo
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem

from experimentos.competidores.base import Presupuesto, PresupuestoAgotado

VERSION = f"pymoo-{pymoo.__version__}"
POBLACION = 100


class _ProblemaPymoo(Problem):
    def __init__(self, presupuesto: Presupuesto, lb: np.ndarray, ub: np.ndarray):
        super().__init__(n_var=len(lb), n_obj=1, xl=lb, xu=ub)
        self._presupuesto = presupuesto

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self._presupuesto.evaluar_lote(X)


def correr(problema, max_fes: int, semilla: int) -> None:
    pres = Presupuesto(problema, max_fes)
    lb = problema.limites[:, 0].astype(float)
    ub = problema.limites[:, 1].astype(float)
    algo = GA(pop_size=POBLACION)
    algo.setup(_ProblemaPymoo(pres, lb, ub), termination=("n_eval", int(max_fes)),
               seed=int(semilla), verbose=False)
    try:
        while algo.has_next():
            algo.next()
    except PresupuestoAgotado:
        return
