"""
Adaptador común para correr algoritmos de NiaPy como competidores
(protocolo de `experimentos/competidores/base.py`).

Uso, desde el módulo de un algoritmo concreto:

    from niapy.algorithms.basic import GreyWolfOptimizer
    from experimentos.competidores.adaptador_niapy import correr_niapy, version_niapy

    VERSION = version_niapy()

    def correr(problema, max_fes, semilla):
        correr_niapy(GreyWolfOptimizer, problema, max_fes, semilla,
                     population_size=...)

Qué resuelve el adaptador:
  - Corte exacto en MaxFES: `Task(max_evals=max_fes)` deja de llamar a la
    función objetivo en cuanto se alcanza el presupuesto (devuelve `inf`
    sin evaluar), así que la corrida consume exactamente `max_fes`. Además,
    toda evaluación pasa por `Presupuesto.evaluar` como segunda barrera.
  - Excepciones: `Algorithm.run` de NiaPy relanza las excepciones solo en el
    hilo y proceso principal; en un worker de la parrilla las guarda en
    `algo.exception` y devuelve `None` en silencio. El adaptador las relanza
    siempre, para que el fallo quede registrado en `fallos/`.
  - Memoria: la `Task` de NiaPy agrega el mejor fitness a una lista en cada
    iteración y en cada mejora. Con MaxFES = 5·10⁷ eso son millones de
    entradas por corrida; `_TaskSinHistorial` no guarda nada (la traza de
    convergencia la lleva `ProblemaCEC2022` con sus checkpoints).
  - Determinismo: la semilla de la corrida va al `seed` del algoritmo, que
    crea su propio `numpy.random.Generator`.
  - Límites: reparación por recorte (`niapy.util.repair.limit`, default de
    NiaPy), igual que PSO y DE del proyecto.
"""

from __future__ import annotations

import niapy
import numpy as np
from niapy.problems import Problem
from niapy.task import Task

from experimentos.competidores.base import Presupuesto, PresupuestoAgotado


def version_niapy() -> str:
    return f"niapy-{niapy.__version__}"


class _ProblemaNiaPy(Problem):
    """`ProblemaCEC2022` visto como `niapy.problems.Problem`."""

    def __init__(self, presupuesto: Presupuesto, lb: np.ndarray, ub: np.ndarray):
        super().__init__(dimension=len(lb), lower=lb, upper=ub)
        self._presupuesto = presupuesto

    def _evaluate(self, x):
        return self._presupuesto.evaluar(x)


class _TaskSinHistorial(Task):
    """`Task` que no guarda la traza de convergencia de NiaPy.

    Reproduce `Task.eval` y `Task.next_iter` (niapy 2.7.1) sin las listas
    `fitness_iters`, `n_evals` y `fitness_evals`."""

    def next_iter(self):
        self.iters += 1

    def eval(self, x):
        if self.stopping_condition():
            return np.inf
        self.evals += 1
        x_f = self.problem.evaluate(x) * self.optimization_type.value
        if x_f < self.x_f * self.optimization_type.value:
            self.x_f = x_f * self.optimization_type.value
        return x_f


def correr_niapy(clase, problema, max_fes: int, semilla: int, **parametros) -> None:
    """Corre `clase(**parametros)` de NiaPy sobre `problema` hasta agotar
    exactamente `max_fes`. El resultado queda en el contador y los
    checkpoints de `problema`."""
    pres = Presupuesto(problema, max_fes)
    lb = problema.limites[:, 0].astype(float)
    ub = problema.limites[:, 1].astype(float)
    task = _TaskSinHistorial(problem=_ProblemaNiaPy(pres, lb, ub),
                             max_evals=int(max_fes))
    algo = clase(seed=int(semilla), **parametros)
    try:
        algo.run(task)
    except PresupuestoAgotado:
        return
    if algo.exception is not None and not isinstance(algo.exception, PresupuestoAgotado):
        raise algo.exception
