"""
IPOP-CMA-ES (Auger & Hansen, 2005) sobre la librería `cma` de N. Hansen.

CMA-ES con reinicios y población creciente: cada vez que una corrida de
CMA-ES cumple un criterio de parada interno de `cma` (tolfun, tolx,
estancamiento, condicionamiento...), se reinicia desde un punto uniforme
nuevo con el doble de población. Se reinicia sin tope hasta agotar MaxFES,
para que los presupuestos grandes (5×10⁷) no se desperdicien en una
corrida ya convergida.

Parámetros (los recomendados por los autores, sin re-tuning):
  - λ inicial: el default de `cma`, 4 + ⌊3·ln D⌋ (10 en D=10, 12 en D=20).
  - Factor de crecimiento de λ por reinicio: 2 (IPOP).
  - σ₀: 1/4 del ancho del dominio (50 en [-100, 100]), la recomendación de
    la documentación de `cma` ("about 1/4 of the search domain width").
  - x₀: uniforme en el dominio, nuevo en cada reinicio.
  - Restricciones de caja: `BoundTransform` de `cma` (default con `bounds`).
  - El resto de opciones y criterios de parada: defaults de `cma`.

Determinismo: `cma` muestrea con `np.random` global. Cada reinicio recibe
una semilla derivada de la semilla de la corrida (opción `seed`, que
re-siembra `np.random`), así que la misma semilla da la misma corrida.
"""

from __future__ import annotations

import math

import cma
import numpy as np

from experimentos.competidores.base import Presupuesto, PresupuestoAgotado

VERSION = f"cma-{cma.__version__}"
FACTOR_POBLACION = 2


def correr(problema, max_fes: int, semilla: int) -> None:
    pres = Presupuesto(problema, max_fes)
    rng = np.random.default_rng(semilla)
    lb, ub = problema.limites[:, 0], problema.limites[:, 1]
    dim = len(lb)
    sigma0 = 0.25 * float(np.max(ub - lb))
    popsize = int(4 + 3 * math.log(dim))

    try:
        while True:
            x0 = lb + rng.random(dim) * (ub - lb)
            opciones = {
                "seed": int(rng.integers(1, 2**31 - 1)),
                "popsize": popsize,
                "bounds": [lb.tolist(), ub.tolist()],
                "verbose": -9,
                "verb_log": 0,
                "verb_disp": 0,
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, opciones)
            while not es.stop():
                X = es.ask()
                es.tell(X, [pres.evaluar(x) for x in X])
            popsize *= FACTOR_POBLACION
    except PresupuestoAgotado:
        return
