"""
JADE — DE adaptativo con archivo externo (Zhang y Sanderson, IEEE TEC
2009), versión con archivo; familia evolutiva.

`DE.JADE` de mealpy (commit 46b9850) con `evolve` reescrito para quedar
fiel al paper. Se conserva lo que ya era fiel: current-to-pbest/1/bin,
CR_i ~ N(μ_CR, 0,1) recortado a [0, 1], F_i ~ Cauchy(μ_F, 0,1) truncado a 1
y regenerado si ≤ 0, archivo A de padres reemplazados (tamaño ≤ NP, con
eliminación aleatoria), r₂ ∈ P ∪ A, y las medias aritmética (μ_CR) y de
Lehmer (μ_F) de los parámetros exitosos.

Desviaciones de mealpy que se corrigen:
  1. Si en una generación no hubo éxitos, mealpy empuja μ_CR y μ_F hacia
     0,5; en el paper solo se actualizan con S_CR y S_F no vacíos.
  2. F se muestreaba con `scipy.stats.cauchy.rvs`, que usa el RNG global de
     numpy; ahora sale del generador sembrado de mealpy.
  3. Límites: el paper repara cada componente fuera de rango con el punto
     medio entre el límite y el padre, (lb + x_i)/2 o (ub + x_i)/2; mealpy
     recortaba al límite.
  4. Selección: el paper reemplaza x_i por u_i si f(u_i) es mejor; se
     mantiene la comparación estricta de mealpy (igual que el paper).

Parámetros del paper: p = 0,05, c = 0,1, μ_CR = μ_F = 0,5 iniciales y
NP = 30 (el paper usa 30 para D = 30 y 100 para D = 100).
"""

from __future__ import annotations

import numpy as np
from mealpy.evolutionary_based.DE import JADE

from experimentos.competidores.adaptador_mealpy import correr_mealpy, version_mealpy

VERSION = f"{version_mealpy()}+jade-zhang2009"
NP = 30
P_MEJORES = 0.05
C_ADAPTACION = 0.1


class JADEZhang(JADE):
    def _muestrear_f(self) -> float:
        while True:
            f = self.dyn_miu_f + 0.1 * self.generator.standard_cauchy()
            if f > 0:
                return min(f, 1.0)

    def _reparar(self, v: np.ndarray, x: np.ndarray) -> np.ndarray:
        lb, ub = self.problem.lb, self.problem.ub
        v = np.where(v < lb, (lb + x) / 2.0, v)
        return np.where(v > ub, (ub + x) / 2.0, v)

    def evolve(self, epoch):
        n = self.pop_size
        pop_sorted, _ = self.get_sorted_population(self.pop, self.problem.minmax)
        top = max(1, int(round(n * self.pt)))
        cr_i, f_i, hijos = np.empty(n), np.empty(n), []
        for idx in range(n):
            cr_i[idx] = np.clip(self.generator.normal(self.dyn_miu_cr, 0.1), 0.0, 1.0)
            f_i[idx] = self._muestrear_f()
            x = self.pop[idx].solution
            x_pbest = pop_sorted[self.generator.integers(0, top)].solution
            r1 = self.generator.choice([j for j in range(n) if j != idx])
            union = self.pop + self.dyn_pop_archive
            r2 = self.generator.choice([j for j in range(len(union)) if j not in (idx, r1)])
            v = x + f_i[idx] * (x_pbest - x) + f_i[idx] * (self.pop[r1].solution - union[r2].solution)
            v = self._reparar(v, x)
            u = np.where(self.generator.random(self.problem.n_dims) < cr_i[idx], v, x)
            j_rand = self.generator.integers(0, self.problem.n_dims)
            u[j_rand] = v[j_rand]
            hijo = self.generate_empty_agent(u)
            hijo.target = self.get_target(u)
            hijos.append(hijo)
        s_cr, s_f = [], []
        for idx in range(n):
            if self.compare_target(hijos[idx].target, self.pop[idx].target, self.problem.minmax):
                self.dyn_pop_archive.append(self.pop[idx].copy())
                s_cr.append(cr_i[idx])
                s_f.append(f_i[idx])
                self.pop[idx] = hijos[idx]
        exceso = len(self.dyn_pop_archive) - n
        if exceso > 0:
            quitar = set(self.generator.choice(len(self.dyn_pop_archive), exceso, replace=False).tolist())
            self.dyn_pop_archive = [a for j, a in enumerate(self.dyn_pop_archive) if j not in quitar]
        if s_cr:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * float(np.mean(s_cr))
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * self.lehmer_mean(np.array(s_f))


def correr(problema, max_fes: int, semilla: int) -> None:
    correr_mealpy(JADEZhang, problema, max_fes, semilla,
                  pop_size=NP, miu_f=0.5, miu_cr=0.5, pt=P_MEJORES, ap=C_ADAPTACION)
