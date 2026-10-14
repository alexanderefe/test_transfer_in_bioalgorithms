"""
MadDE — Multiple adaptation DE (Biswas, Saha, De, Cobb, Das & Jalaian,
"Improving Differential Evolution through Bayesian Hyperparameter
Optimization", IEEE CEC 2021).

Port a Python del código MATLAB oficial de los autores
(github.com/subhodipbiswas/MadDE, archivo `MadDE/MadDE.m` y auxiliares
`gnR1R2.m`, `updateArchive.m`, `boundConstraint.m`; commit 13f0582, GPL-3.0).
Se porta línea a línea; los hiperparámetros son los de la entrega a la
competición CEC 2021 (sin re-tuning):

  q_cr_rate = 0.01, p_best_rate = 0.18, arc_rate = 2.3,
  memory_size = 10·D, pop_size = 2·D² (200 en D=10, 800 en D=20),
  min_pop_size = 4, memorias de F y CR iniciadas en 0.2.

Mecanismos: tres operadores de mutación con probabilidad adaptativa
(current-to-pbest/1 con archivo, current-to-rand/1 con archivo,
weighted-rand-to-qbest/1 con atracción), cruce binomial "q-best",
memorias de éxito tipo SHADE y reducción lineal de población (LPSR)
contra MaxFES.

Comportamientos del original que se REPRODUCEN tal cual (mismo criterio
que D-5: no se "corrige" el código de referencia):
  1. Al inicio de cada generación se reordenan `pop` y `fitness`, pero no
     `popold`; el archivo recibe `popold(I == 1)` con `I` en el orden ya
     ordenado, es decir, no necesariamente los padres reemplazados.
  2. `probDE` solo se actualiza si los tres operadores tuvieron éxito
     medio > 0 en la generación (`if count_S ~= 0` sobre un vector en
     MATLAB exige todos no nulos); si no, vuelve a 1/3 cada uno.

Diferencias inevitables con el original:
  - Inicialización: Sobol aleatorizado (scipy, scrambling LMS + desplazamiento
    digital) con skip 10⁴, en vez de `scramble(sobolset, 'MatousekAffineOwen')`
    con skip 10⁴ y leap 10³ — scipy no ofrece leap. Ambas son secuencias de
    baja discrepancia aleatorizadas; la diferencia es de detalle.
  - RNG: numpy en vez del Mersenne Twister de MATLAB (las distribuciones
    son las mismas; los números concretos no).
  - Corte exacto en MaxFES (ver `base.py`).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import qmc

from experimentos.competidores.base import (
    Presupuesto, PresupuestoAgotado, round_half_up,
)

VERSION = "port-MadDE.m@13f0582"

Q_CR_RATE = 0.01
P_BEST_RATE = 0.18
ARC_RATE = 2.3
MIN_POP_SIZE = 4


def _gn_r1r2r3(rng, np1: int, np2: int):
    """gnR1R2.m: r1 ∈ pop ≠ r0; r2 ∈ pop∪archivo ≠ r0, r1;
    r3 ∈ pop ≠ r0, r1, r2 (r0 = índice propio)."""
    r0 = np.arange(np1)

    def regenerar(r, n, prohibidos):
        for _ in range(1000):
            pos = np.zeros(np1, dtype=bool)
            for p in prohibidos:
                pos |= r == p
            if not pos.any():
                return r
            r[pos] = rng.integers(n, size=int(pos.sum()))
        raise RuntimeError("gnR1R2: no se pudo generar índices distintos en 1000 iteraciones")

    r1 = regenerar(rng.integers(np1, size=np1), np1, (r0,))
    r2 = regenerar(rng.integers(np2, size=np1), np2, (r1, r0))
    r3 = regenerar(rng.integers(np1, size=np1), np1, (r0, r1, r2))
    return r1, r2, r3


def _actualizar_archivo(rng, arch_pop, arch_fun, arch_np, nuevos, fit_nuevos):
    """updateArchive.m: agrega, elimina filas duplicadas y, si sobra,
    conserva un subconjunto aleatorio de tamaño `arch_np`."""
    if arch_np == 0:
        return arch_pop, arch_fun
    pop_all = np.vstack([arch_pop, nuevos])
    fun_all = np.concatenate([arch_fun, fit_nuevos])
    _, ix = np.unique(pop_all, axis=0, return_index=True)
    if len(ix) < len(pop_all):
        pop_all, fun_all = pop_all[ix], fun_all[ix]
    if len(pop_all) <= arch_np:
        return pop_all, fun_all
    pos = rng.permutation(len(pop_all))[:arch_np]
    return pop_all[pos], fun_all[pos]


def _poblacion_inicial(rng, n: int, lb, ub) -> np.ndarray:
    sobol = qmc.Sobol(d=len(lb), scramble=True, rng=rng)
    sobol.fast_forward(10_000)
    with warnings.catch_warnings():
        # Sobol avisa cuando n no es potencia de 2 (pierde balance); el
        # original MATLAB usa el mismo n = 2·D² sin esa restricción.
        warnings.simplefilter("ignore", UserWarning)
        rand0 = sobol.random(n)
    return lb + rand0 * (ub - lb)


def correr(problema, max_fes: int, semilla: int) -> None:
    pres = Presupuesto(problema, max_fes)
    rng = np.random.default_rng(semilla)
    lb, ub = problema.limites[:, 0], problema.limites[:, 1]
    dim = len(lb)

    memory_size = 10 * dim
    pop_size = 2 * dim ** 2
    max_pop_size = pop_size

    try:
        popold = _poblacion_inicial(rng, pop_size, lb, ub)
        fitness = pres.evaluar_lote(popold)

        prob_de = np.full(3, 1.0 / 3)
        memory_sf = np.full(memory_size, 0.2)
        memory_cr = np.full(memory_size, 0.2)
        memory_pos = 0

        arch_np = round_half_up(ARC_RATE * pop_size)
        arch_pop = np.zeros((0, dim))
        arch_fun = np.zeros(0)

        while pres.fes < max_fes:
            nfes = pres.fes
            orden = np.argsort(fitness, kind="stable")
            fitness = fitness[orden]
            pop = popold[orden]

            idx_mem = rng.integers(memory_size, size=pop_size)
            mu_sf, mu_cr = memory_sf[idx_mem], memory_cr[idx_mem]

            # ── Crossover rate ───────────────────────────────────────
            cr = rng.normal(mu_cr, 0.1)
            cr[mu_cr == -1] = 0
            cr = np.clip(cr, 0.0, 1.0)

            # ── Scaling factor (Cauchy, re-muestreo si ≤ 0) ──────────
            sf = mu_sf + 0.1 * np.tan(np.pi * (rng.random(pop_size) - 0.5))
            pos = sf <= 0
            while pos.any():
                sf[pos] = mu_sf[pos] + 0.1 * np.tan(np.pi * (rng.random(int(pos.sum())) - 0.5))
                pos = sf <= 0
            sf = np.minimum(sf, 1.0)

            pop_all = np.vstack([pop, arch_pop])
            r1, r2, r3 = _gn_r1r2r3(rng, pop_size, len(pop_all))
            vi = np.zeros((pop_size, dim))

            # ── Mutación: tres operadores ────────────────────────────
            bb = rng.random(pop_size)
            l2 = prob_de[0] + prob_de[1]
            de_1 = bb <= prob_de[0]
            de_2 = (bb > prob_de[0]) & (bb <= l2)
            de_3 = (bb > l2) & (bb <= 1.0)

            p_np = max(round_half_up(P_BEST_RATE * pop_size), 2)
            pbest = pop[rng.integers(p_np, size=pop_size)]
            f = sf[:, None]

            # DE/current-to-p-best/1 con archivo
            vi[de_1] = pop[de_1] + f[de_1] * (
                pbest[de_1] - pop[de_1] + pop[r1[de_1]] - pop_all[r2[de_1]])
            # DE/current-to-rand/1 con archivo
            vi[de_2] = pop[de_2] + f[de_2] * (pop[r1[de_2]] - pop_all[r2[de_2]])
            # DE/weighted-rand-to-q-best/1 con atracción
            progreso = nfes / max_fes
            q_best_rate = 2 * P_BEST_RATE - P_BEST_RATE * progreso
            q_np = max(round_half_up(q_best_rate * pop_size), 2)
            qbest = pop[rng.integers(q_np, size=pop_size)]
            atraccion = 0.5 + 0.5 * progreso
            vi[de_3] = f[de_3] * (pop[r1[de_3]] + atraccion * (qbest[de_3] - pop[r3[de_3]]))

            # boundConstraint.m: punto medio entre el padre y el límite
            bajo = vi < lb
            vi[bajo] = ((pop + lb) / 2)[bajo]
            alto = vi > ub
            vi[alto] = ((pop + ub) / 2)[alto]

            # ── Cruce binomial q-best ────────────────────────────────
            mask = rng.random((pop_size, dim)) > cr[:, None]
            mask[np.arange(pop_size), rng.integers(dim, size=pop_size)] = False
            q_np_all = max(round_half_up(q_best_rate * len(pop_all)), 2)
            pop_all_best = pop_all[rng.integers(q_np_all, size=pop_size)]
            usa_qbest = rng.random(pop_size) <= Q_CR_RATE
            base = pop.copy()
            base[usa_qbest] = pop_all_best[usa_qbest]
            ui = vi.copy()
            ui[mask] = base[mask]

            hijos = pres.evaluar_lote(ui)

            # ── Selección ────────────────────────────────────────────
            dif = np.abs(fitness - hijos)
            mejora = fitness > hijos
            good_cr, good_f, dif_val = cr[mejora], sf[mejora], dif[mejora]

            # Quirk 1 del original: popold sin reordenar, I ya ordenado.
            arch_pop, arch_fun = _actualizar_archivo(
                rng, arch_pop, arch_fun, arch_np, popold[mejora], fitness[mejora])

            diff2 = np.maximum(0.0, fitness - hijos) / np.abs(fitness)
            count_s = np.array([
                max(0.0, diff2[m].mean()) if m.any() else 0.0
                for m in (de_1, de_2, de_3)
            ])
            if np.all(count_s != 0):   # quirk 2: los tres deben ser no nulos
                prob_de = np.clip(count_s / count_s.sum(), 0.1, 0.9)
            else:
                prob_de = np.full(3, 1.0 / 3)

            fitness = np.where(mejora, hijos, fitness)
            popold = pop.copy()
            popold[mejora] = ui[mejora]

            # ── Memorias de F y CR (media de Lehmer ponderada) ───────
            if len(good_cr) > 0:
                w = dif_val / dif_val.sum()
                memory_sf[memory_pos] = (w @ good_f ** 2) / (w @ good_f)
                if good_cr.max() == 0 or memory_cr[memory_pos] == -1:
                    memory_cr[memory_pos] = -1
                else:
                    memory_cr[memory_pos] = (w @ good_cr ** 2) / (w @ good_cr)
                memory_pos = (memory_pos + 1) % memory_size
            else:
                memory_cr[memory_pos] = 0.5
                memory_sf[memory_pos] = 0.5

            # ── Reducción lineal de población ────────────────────────
            plan_pop_size = round_half_up(
                ((MIN_POP_SIZE - max_pop_size) / max_fes) * pres.fes + max_pop_size)
            if pop_size > plan_pop_size:
                reduccion = pop_size - plan_pop_size
                if pop_size - reduccion < MIN_POP_SIZE:
                    reduccion = pop_size - MIN_POP_SIZE
                pop_size -= reduccion
                # Elimina uno a uno el peor (último entre empates, como
                # `sort` estable de MATLAB): equivale a quitar los últimos
                # `reduccion` del orden estable ascendente.
                quedan = np.sort(np.argsort(fitness, kind="stable")[:pop_size])
                popold, fitness = popold[quedan], fitness[quedan]

                arch_np = round_half_up(ARC_RATE * pop_size)
                if len(arch_pop) > arch_np:
                    pos = rng.permutation(len(arch_pop))[:arch_np]
                    arch_pop, arch_fun = arch_pop[pos], arch_fun[pos]
    except PresupuestoAgotado:
        return
