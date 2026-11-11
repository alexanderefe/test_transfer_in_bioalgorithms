"""
CLPSO — Comprehensive Learning PSO (Liang, Qin, Suganthan y Baskar, IEEE
TEC 2006), familia de enjambre.

Port de `CLPSO_new_func.m`, el código MATLAB de los autores
(github.com/P-N-Suganthan/CODES, archivo 2006-IEEE-TEC-CLPSO.zip,
carpeta "TEC matlab code"), con los parámetros de su script de prueba
`TEC_test.m`. Se porta el código y no el texto del paper porque el
`CL_PSO` de mealpy tiene errores (docs/competidores.md §8) y porque el
código es la versión que produjo los resultados publicados.

Lo que hace el código original (y este port):
  - Cada partícula k aprende, dimensión por dimensión, del pbest de un
    «ejemplar» f_pbest(k, d). Con probabilidad Pc(k) ese ejemplar es el
    ganador de un torneo binario por pbest entre dos partículas al azar;
    si no, es la propia partícula. Si ninguna dimensión aprende de otra,
    se fuerza una al azar.
  - El ejemplar se reasigna cuando la partícula acumula `stay_num ≥ 5`
    evaluaciones sin mejorar su pbest (refreshing gap). El contador solo
    se reinicia al reasignar, como en el original.
  - Velocidad: v = w·v + c·r·(pbest_f − x), recortada a ±0,2·(ub − lb);
    x = x + v sin recorte. La partícula solo se evalúa (y solo actualiza
    pbest) si queda dentro del dominio; fuera de él no gasta evaluación.
  - Actualización asíncrona: gbest y los pbest se actualizan partícula
    por partícula dentro de la generación.
  - m = 0: el término de gbest (`ai`) queda apagado, como en el original.

Parámetros (del código, que difiere del texto del paper en tres puntos):
  - Pc(k) = 0,5·(e^{t_k} − 1)/(e^{5} − 1), t_k = 5·k/(ps − 1), en [0, 0,5]
    (paper: 0,05 + 0,45·(e^{10(k−1)/(ps−1)} − 1)/(e^{10} − 1)).
  - w lineal de 0,9 a 0,2 en `me` generaciones, me = MaxFES / ps; si se
    agotan las generaciones antes que las FES (partículas fuera del
    dominio no gastan evaluación), w queda en 0,2, igual que el original
    (paper: 0,9 → 0,4).
  - Refreshing gap 5 (paper: m = 7).
  - c = 1,49445; población: 10 en D = 10 (TEC_test.m) y 40 en D > 10
    (valor del paper para 30-D; el paper no da uno para 20-D).

Determinismo: `numpy.random.Generator` sembrado con la semilla de la
corrida (el original usa `rand('state', clock)`). La secuencia de números
aleatorios no es la de MATLAB, pero las distribuciones sí.
"""

from __future__ import annotations

import numpy as np

from experimentos.competidores.base import Presupuesto, PresupuestoAgotado

VERSION = "port-CLPSO_new_func.m@TEC2006"
C_ACELERACION = 1.49445
W_INICIAL, W_RANGO = 0.9, 0.7          # iwt = 0.9 - (1:me)*(0.7/me)
REFRESHING_GAP = 5
FRACCION_VMAX = 0.2


def tamano_poblacion(dim: int) -> int:
    return 10 if dim <= 10 else 40


def correr(problema, max_fes: int, semilla: int) -> None:
    pres = Presupuesto(problema, max_fes)
    rng = np.random.default_rng(semilla)
    lb = problema.limites[:, 0].astype(float)
    ub = problema.limites[:, 1].astype(float)
    D = len(lb)
    ps = tamano_poblacion(D)
    me = max(2, int(max_fes) // ps)    # TEC_test.m: Max_FES = ps*me

    t = 5.0 * np.linspace(0.0, 1.0, ps)
    Pc = 0.5 * (np.exp(t) - np.exp(t[0])) / (np.exp(t[-1]) - np.exp(t[0]))
    m = np.zeros(ps, dtype=int)
    mv = FRACCION_VMAX * (ub - lb)
    dims = np.arange(D)

    try:
        pos = lb + (ub - lb) * rng.random((ps, D))
        e = np.array([pres.evaluar(pos[k]) for k in range(ps)])
        vel = -mv + 2.0 * mv * rng.random((ps, D))
        pbest = pos.copy()
        pbestval = e.copy()
        gid = int(np.argmin(pbestval))
        gbestval, gbest = pbestval[gid], pbest[gid].copy()
        stay_num = np.zeros(ps, dtype=int)
        ai = np.zeros((ps, D))
        f_pbest = np.repeat(np.arange(ps)[:, None], D, axis=1)

        def asignar_ejemplar(k: int) -> None:
            ai[k, :] = 0.0
            f_pbest[k, :] = k
            ar = rng.permutation(D)
            ai[k, ar[:m[k]]] = 1.0
            fi1 = rng.integers(0, ps, D)
            fi2 = rng.integers(0, ps, D)
            fi = np.where(pbestval[fi1] < pbestval[fi2], fi1, fi2)
            bi = rng.random(D) > 1.0 - Pc[k]      # ceil(rand - 1 + Pc(k))
            if not bi.any():
                bi[rng.permutation(D)[0]] = True
            f_pbest[k, :] = np.where(bi, fi, f_pbest[k, :])

        for k in range(ps):
            asignar_ejemplar(k)

        gen = 1
        while True:
            gen = min(gen + 1, me)
            w = W_INICIAL - gen * (W_RANGO / me)
            for k in range(ps):
                if stay_num[k] >= REFRESHING_GAP:
                    stay_num[k] = 0
                    asignar_ejemplar(k)
                pbest_f = pbest[f_pbest[k], dims]
                aa = (C_ACELERACION * (1.0 - ai[k]) * rng.random(D) * (pbest_f - pos[k])
                      + C_ACELERACION * ai[k] * rng.random(D) * (gbest - pos[k]))
                vel[k] = np.clip(w * vel[k] + aa, -mv, mv)
                pos[k] = pos[k] + vel[k]
                if np.all(pos[k] <= ub) and np.all(pos[k] >= lb):
                    ek = pres.evaluar(pos[k])
                    if pbestval[k] <= ek:
                        stay_num[k] += 1
                    else:
                        pbest[k] = pos[k]
                        pbestval[k] = ek
                    if pbestval[k] < gbestval:
                        gbest = pbest[k].copy()
                        gbestval = pbestval[k]
    except PresupuestoAgotado:
        return
