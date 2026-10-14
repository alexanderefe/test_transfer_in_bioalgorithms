"""
NL-SHADE-RSP (Stanovov, Akhmedova & Semenkin, "NL-SHADE-RSP Algorithm with
Adaptive Archive and Selective Pressure for CEC 2021 Numerical
Optimization", IEEE CEC 2021) — ganador de CEC 2021.

Port a Python del C++ oficial de los autores (`nl_shade_rsp.cpp`, entregado a
la competición CEC 2021; distribuido en el archivo "Codes-of-top-methods" de
github.com/P-N-Suganthan/2021-SO-BCO). Parámetros de la entrega, sin
re-tuning:

  NInds = 30·D (300 / 600), NIndsMin = 4, MemorySize = 20·D,
  ArchiveSizeParam = 2.1, memorias de F y Cr iniciadas en 0.2.

Mecanismos: current-to-pbest/1 con pbest ∈ [0.2, 0.4] creciente; presión
selectiva por rango (prob. ∝ exp(-i/NInds)) para r2 cuando no viene del
archivo; uso del archivo con probabilidad adaptativa; cruce binomial o
exponencial (50 % cada uno por generación) con Cr ordenado por rango del
individuo; reducción NO lineal de población contra MaxFES; reparación de
límites por re-muestreo uniforme.

Comportamientos del original que se REPRODUCEN tal cual (criterio de D-5):
  1. La memoria de F se actualiza con la variable global `F`, que al
     guardar los éxitos vale el ÚLTIMO F muestreado en la generación (sin
     recortar a 1), no el F usado por cada individuo.
  2. El cruce binomial usa `CrToUse` (0 durante la primera mitad del
     presupuesto, luego crece linealmente hasta 1), no el Cr adaptado:
     antes de 0.5·MaxFES el binomial solo copia la coordenada jrand.
  3. r1 no excluye al propio individuo (`GenerateNextRandUnif` solo evita
     repetir r0).
  4. `RemoveWorst` busca el peor sobre el tamaño viejo de la población en
     cada eliminación (no lo achica), y la tasa de éxito sin archivo se
     normaliza por `NInds - NArchUsages`; con la semántica de `std::min/max`
     para los casos 0/0.

Diferencias inevitables: RNG de numpy en vez de mt19937 (las
distribuciones son las mismas), ordenamiento estable en vez del quicksort
propio (solo cambia el orden entre empates exactos), y corte exacto en
MaxFES (ver `base.py`).
"""

from __future__ import annotations

import numpy as np

from experimentos.competidores.base import (
    Presupuesto, PresupuestoAgotado, round_half_up,
)

VERSION = "port-nl_shade_rsp.cpp@CEC2021"

N_INDS_MIN = 4
ARCHIVE_SIZE_PARAM = 2.1
REINTENTOS = 25


def _cmin(a, b):
    """std::min(a, b) de C++: (b < a) ? b : a — define el caso NaN."""
    return b if b < a else a


def _cmax(a, b):
    """std::max(a, b) de C++: (a < b) ? b : a."""
    return b if a < b else a


def _media_lehmer(valores: np.ndarray, deltas: np.ndarray) -> float:
    """MeanWL_general(g_p=2, g_m=1): media de Lehmer ponderada por mejora."""
    w = deltas / deltas.sum()
    den = np.sum(w * valores)
    if abs(den) > 1e-6:
        return float(np.sum(w * valores ** 2) / den)
    return 0.5


def _reintentar(rng, r, malos, generar):
    """Hasta REINTENTOS re-muestreos de las posiciones que violan la
    condición (como los `for(i=0; i!=25 && !Check...; i++)` del C++)."""
    pos = malos(r)
    for _ in range(REINTENTOS):
        if not pos.any():
            break
        r[pos] = generar(int(pos.sum()))
        pos = malos(r)
    return r


def correr(problema, max_fes: int, semilla: int) -> None:
    pres = Presupuesto(problema, max_fes)
    rng = np.random.default_rng(semilla)
    lb, ub = problema.limites[:, 0], problema.limites[:, 1]
    dim = len(lb)

    n_inds_max = 30 * dim
    n_inds = n_inds_max
    memory_size = 20 * dim
    archive_size = int(n_inds_max * ARCHIVE_SIZE_PARAM)
    archivo = np.zeros((n_inds_max * int(np.ceil(ARCHIVE_SIZE_PARAM)), dim))
    arch_actual = 0

    memory_cr = np.full(memory_size, 0.2)
    memory_f = np.full(memory_size, 0.2)
    memory_iter = 0
    arch_probs = 0.5

    try:
        popul = lb + rng.random((n_inds, dim)) * (ub - lb)
        fit = pres.evaluar_lote(popul)

        while pres.fes < max_fes:
            nfe0 = pres.fes
            idx = np.arange(n_inds)
            indexes = np.argsort(fit, kind="stable")
            back = np.empty(n_inds, dtype=int)
            back[indexes] = idx
            p_rango = np.exp(-idx / n_inds)
            p_rango /= p_rango.sum()
            psizeval = int(max(2.0, n_inds * (0.2 / max_fes * nfe0 + 0.2)))
            cruce_exponencial = rng.random() < 0.5

            # ── Generación de Cr y F por individuo ───────────────────
            mem = rng.integers(memory_size, size=n_inds)
            cr = np.clip(rng.normal(memory_cr[mem], 0.1), 0.0, 1.0)
            f = memory_f[mem] + 0.1 * rng.standard_cauchy(n_inds)
            malos = f <= 0
            while malos.any():
                f[malos] = memory_f[mem[malos]] + 0.1 * rng.standard_cauchy(int(malos.sum()))
                malos = f <= 0
            f_global = f[-1]                    # quirk 1
            f_gen = np.minimum(f, 1.0)
            cr_gen = np.sort(cr)
            cr_ind = cr_gen[back]               # mejor individuo → menor Cr

            # ── Índices r0 (pbest), r1, r2 ───────────────────────────
            r0 = indexes[rng.integers(psizeval, size=n_inds)]
            r0 = _reintentar(rng, r0, lambda r: r == idx,
                             lambda n: indexes[rng.integers(psizeval, size=n)])
            r1 = rng.integers(n_inds, size=n_inds)
            r1 = _reintentar(rng, r1, lambda r: r == r0,       # quirk 3
                             lambda n: rng.integers(n_inds, size=n))
            usa_arch = (rng.random(n_inds) <= arch_probs) & (arch_actual > 0)
            r2 = indexes[rng.choice(n_inds, size=n_inds, p=p_rango)]
            r2[usa_arch] = n_inds + rng.integers(max(arch_actual, 1), size=int(usa_arch.sum()))
            sin_arch = ~usa_arch

            def r2_malo(r):
                return sin_arch & ((r == idx) | (r == r0) | (r == r1))

            r2 = _reintentar(rng, r2, r2_malo,
                             lambda n: indexes[rng.choice(n_inds, size=n, p=p_rango)])

            todos = np.vstack([popul, archivo[:arch_actual]])
            fi = f_gen[:, None]
            donor = popul + fi * (todos[r0] - popul) + fi * (todos[r1] - todos[r2])

            # ── Cruce ────────────────────────────────────────────────
            will = rng.integers(dim, size=n_inds)
            nfe_i = nfe0 + idx                  # NFEval al procesar cada individuo
            cr_to_use = np.where(nfe_i > 0.5 * max_fes, (nfe_i / max_fes - 0.5) * 2, 0.0)
            cols = np.arange(dim)[None, :]
            if not cruce_exponencial:
                mask = (rng.random((n_inds, dim)) < cr_to_use[:, None]) | (cols == will[:, None])
            else:
                inicio = rng.integers(dim, size=n_inds)
                # nº de éxitos consecutivos de "Random < Cr" (geométrica),
                # acotado al final del vector (sin vuelta circular).
                extra = np.where(cr_ind >= 1.0, dim,
                                 rng.geometric(np.clip(1.0 - cr_ind, 1e-12, 1.0)) - 1)
                fin = np.minimum(inicio + 1 + extra, dim)
                mask = (cols >= inicio[:, None]) & (cols < fin[:, None])
            trial = np.where(mask, donor, popul)

            # FindLimits: fuera de rango → uniforme en el dominio
            fuera = (trial < lb) | (trial > ub)
            if fuera.any():
                trial[fuera] = (lb + rng.random((n_inds, dim)) * (ub - lb))[fuera]

            fit_trial = pres.evaluar_lote(trial)

            # ── Éxitos (estricto) para las memorias ──────────────────
            exito = fit_trial < fit
            delta_ok = np.abs(fit - fit_trial)[exito]
            cr_ok = cr_ind[exito]
            f_ok = np.full(int(exito.sum()), f_global)

            # ── Reemplazo (≤), archivo y probabilidad de archivo ─────
            reemplazo = fit_trial <= fit
            mejora_rel = (fit - fit_trial) / fit
            arch_success = float(mejora_rel[reemplazo & usa_arch].sum())
            no_arch_success = float(mejora_rel[reemplazo & ~usa_arch].sum())
            n_arch_usages = int((reemplazo & usa_arch).sum())

            padres = popul[reemplazo]
            n_nuevos = len(padres)
            entran = min(n_nuevos, max(archive_size - arch_actual, 0))
            archivo[arch_actual:arch_actual + entran] = padres[:entran]
            arch_actual += entran
            resto = padres[entran:]
            if len(resto) and archive_size > 0:
                archivo[rng.integers(archive_size, size=len(resto))] = resto

            popul[reemplazo] = trial[reemplazo]
            fit[reemplazo] = fit_trial[reemplazo]

            if n_arch_usages != 0:
                arch_success /= n_arch_usages
                with np.errstate(divide="ignore", invalid="ignore"):
                    no_arch_success = np.float64(no_arch_success) / (n_inds - n_arch_usages)
                    arch_probs = np.float64(arch_success) / (arch_success + no_arch_success)
                arch_probs = _cmax(0.1, _cmin(0.9, float(arch_probs)))
                if arch_success == 0:
                    arch_probs = 0.5
            else:
                arch_probs = 0.5

            # ── Reducción no lineal de población y archivo ───────────
            ratio = pres.fes / max_fes
            plan = round_half_up((N_INDS_MIN - n_inds_max) * ratio ** (1.0 - ratio) + n_inds_max)
            nuevo_n = min(max(plan, N_INDS_MIN), n_inds_max)
            archive_size = max(int(plan * ARCHIVE_SIZE_PARAM), N_INDS_MIN)
            arch_actual = min(arch_actual, archive_size)

            for _ in range(n_inds - nuevo_n):       # RemoveWorst (quirk 4)
                peor = int(np.argmax(fit[:n_inds]))
                popul[peor:n_inds - 1] = popul[peor + 1:n_inds].copy()
                fit[peor:n_inds - 1] = fit[peor + 1:n_inds].copy()
            popul, fit = popul[:nuevo_n].copy(), fit[:nuevo_n].copy()
            n_inds = nuevo_n

            # ── Memorias ─────────────────────────────────────────────
            if len(cr_ok) > 0:
                memory_cr[memory_iter] = _media_lehmer(cr_ok, delta_ok)
                memory_f[memory_iter] = _media_lehmer(f_ok, delta_ok)
                memory_iter = (memory_iter + 1) % memory_size
            else:
                memory_f[memory_iter] = 0.5
                memory_cr[memory_iter] = 0.5
    except PresupuestoAgotado:
        return
