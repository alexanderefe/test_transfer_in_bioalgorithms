"""
Adaptador común para correr algoritmos de mealpy como competidores
(protocolo de `experimentos/competidores/base.py`).

Uso, desde el módulo de un algoritmo concreto:

    from mealpy.evolutionary_based.DE import JADE
    from experimentos.competidores.adaptador_mealpy import correr_mealpy, version_mealpy

    VERSION = version_mealpy()

    def correr(problema, max_fes, semilla):
        correr_mealpy(JADE, problema, max_fes, semilla, pop_size=...)

Qué resuelve el adaptador:
  - Corte exacto en MaxFES: mealpy revisa `max_fe` solo al final de cada
    época y puede pasarse hasta en una población. Toda evaluación pasa por
    `Presupuesto.evaluar`, que lanza `PresupuestoAgotado` en la evaluación
    exacta; mealpy no captura excepciones, así que la corrida termina ahí.
  - Número de épocas: el validador de mealpy limita `epoch` a 10⁵, menos de
    lo necesario con MaxFES = 5·10⁷ (p. ej. 5·10⁵ épocas con población 100).
    El adaptador fija `epoch` muy alto después de construir el algoritmo y
    deja que corte el presupuesto. Por eso SOLO admite algoritmos cuyo
    `evolve` no use `self.epoch` (calendarios como w(t) = f(t / T)); se
    verifica al llamar.
  - Memoria y salida: mealpy guarda por época cuatro copias de agentes en
    `history` y, si no registra en consola, dibuja una barra `tqdm`. El
    adaptador recorta el historial al último elemento (lo único que usa
    `update_global_best_agent`) y silencia el logger.
  - Determinismo: la semilla va a `solve(seed=...)` (generador propio de
    mealpy) y también a `np.random.seed`, porque algunos algoritmos usan
    `scipy.stats` sin `random_state` (p. ej. `cauchy.rvs` en JADE), que
    toma el estado global de numpy. Se usa el modo 'single' (secuencial),
    el único reproducible.
  - Límites: `correct_solution` de mealpy (recorte a [lb, ub]).
"""

from __future__ import annotations

import inspect
import logging

import mealpy
import numpy as np
from mealpy import FloatVar

from experimentos.competidores.base import Presupuesto, PresupuestoAgotado

# Commit de GitHub fijado en requirements.txt (la 3.1.0 no está en PyPI).
COMMIT_MEALPY = "46b9850"
_EPOCAS_SIN_TOPE = 10**12
_LISTAS_HISTORIAL = ("list_global_best", "list_current_best",
                     "list_global_worst", "list_current_worst")


def version_mealpy() -> str:
    return f"mealpy-{mealpy.__version__}+{COMMIT_MEALPY}"


def _usa_epoch(clase) -> bool:
    """True si el `evolve` de la clase (o de una base) depende de `self.epoch`."""
    for c in inspect.getmro(clase):
        if "evolve" in c.__dict__ and "self.epoch" in inspect.getsource(c.__dict__["evolve"]):
            return True
    return False


def _paso_sin_historial(self, pop=None, epoch=None, runtime=None):
    """Reemplazo de `Optimizer.track_optimize_step`: no guarda ni imprime
    nada y recorta las listas que `update_global_best_agent` hace crecer."""
    for nombre in _LISTAS_HISTORIAL:
        lista = getattr(self.history, nombre)
        if len(lista) > 1:
            del lista[:-1]


def correr_mealpy(clase, problema, max_fes: int, semilla: int, **parametros) -> None:
    """Corre `clase(**parametros)` de mealpy sobre `problema` hasta agotar
    exactamente `max_fes`. El resultado queda en el contador y los
    checkpoints de `problema`."""
    if _usa_epoch(clase):
        raise ValueError(
            f"{clase.__name__}.evolve usa self.epoch: su calendario depende del "
            f"número de épocas y este adaptador no lo fija. Requiere un adaptador "
            f"que traduzca MaxFES a épocas.")
    pres = Presupuesto(problema, max_fes)
    lb = problema.limites[:, 0].astype(float)
    ub = problema.limites[:, 1].astype(float)
    problema_mealpy = {
        "obj_func": pres.evaluar,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "min",
        "log_to": "console",   # evita tqdm; el logger se silencia abajo
    }
    # El constructor de mealpy registra sus parámetros con logger.info.
    nivel_previo = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        algo = clase(**parametros)
    finally:
        logging.disable(nivel_previo)
    algo.epoch = _EPOCAS_SIN_TOPE
    algo.track_optimize_step = _paso_sin_historial.__get__(algo)
    # Si MaxFES se agota justo al final de una época, mealpy termina por su
    # propio `max_fe` (sin PresupuestoAgotado) y llama a
    # track_optimize_process, que resume el historial que aquí no se guarda.
    algo.track_optimize_process = lambda: None
    original_check_problem = algo.check_problem

    def _check_problem_silencioso(problem, seed):
        # check_problem crea el logger y registra los parámetros con info.
        previo = logging.root.manager.disable
        logging.disable(logging.INFO)
        try:
            original_check_problem(problem, seed)
        finally:
            logging.disable(previo)
        algo.logger.disabled = True
        algo.problem.logger.disabled = True
    algo.check_problem = _check_problem_silencioso

    np.random.seed(int(semilla) % 2**32)
    try:
        algo.solve(problema_mealpy, mode="single",
                   termination={"max_fe": int(max_fes)}, seed=int(semilla))
    except PresupuestoAgotado:
        return
