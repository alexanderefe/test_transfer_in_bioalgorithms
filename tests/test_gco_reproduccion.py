"""
Prueba de reproducción de GCO (experimentos/competidores/gco.py) contra la
Tabla 2 de Villaseñor et al. (2018), «Germinal Center Optimization
Algorithm», IJCIS 12(1):13-27: dimensión 2, 40 células, 500 iteraciones
(40 + 500·40 = 20 040 evaluaciones), CR = 0,7, F = 1,25, 30 corridas.

Valores publicados para GCO (media ± desviación):
  f1  Sphere         [−5,12, 5,12]²    1,53e−55 ± 4,58e−55
  f2  Sum of squares [−5,12, 5,12]²    1,29e−53 ± 7,06e−53
  f8  Ackley         [−32,76, 32,76]²  4,44e−16 ± 2,00e−31
  f9  Griewank       [−600, 600]²      2,21e−03 ± 3,44e−03
  f11 Rastrigin      [−5,12, 5,12]²    0 ± 0
  f13 Zakharov       [−5, 10]²         6,91e−51 ± 1,73e−50
  f14 Dixon-Price    [−10, 10]²        3,86e−32 ± 4,26e−33
  f15 Rosenbrock     [−5, 10]²         1,44e−26 ± 7,25e−26

Criterio de aceptación: donde el paper llega a precisión de máquina
(media < 1e−10), esta implementación también; en Griewank, la media debe
quedar bajo la media publicada + 2 desviaciones (las corridas que no
convergen quedan en óptimos locales del orden de 1e−2).

Además, Tabla 4 del paper (dimensión 30, 40 células, 1000 iteraciones =
40 040 evaluaciones), donde GCO NO converge y los valores son intermedios
(prueba más exigente que «llega a 0»):
  f1  Sphere     4,01e+01 ± 1,37e+01
  f8  Ackley     1,92e+01 ± 8,51e−01
  f11 Rastrigin  2,37e+02 ± 4,69e+01
  f13 Zakharov   6,08e+02 ± 6,57e+01
Con 10 corridas propias, la diferencia se compara con
3·sqrt(std_pub²/30 + std_propia²/10). ESTA PARTE ES INFORMATIVA (no hace
fallar el test): la Tabla 4 NO se reproduce. El pseudocódigo no dice si
r₁, r₂, r₃ ~ 𝒞 deben ser distintos, y en 30-D eso cambia el resultado
~50 veces (Sphere: 114 ± 21 con distintos, como mealpy; 2,5 ± 2,2 con
reemplazo; publicado 40,1 ± 13,7). Los autores no publicaron su código.
Ver docs/competidores.md §8.3.
"""

import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from experimentos.competidores import gco  # noqa: E402

D, CORRIDAS = 2, 30
MAX_FES = gco.N_CELULAS + 500 * gco.N_CELULAS
D_ALTA, CORRIDAS_ALTA = 30, 10
MAX_FES_ALTA = gco.N_CELULAS + 1000 * gco.N_CELULAS


def _sphere(x): return float(np.sum(x ** 2))
def _sumsq(x): return float(np.sum(np.arange(1, len(x) + 1) * x ** 2))
def _ackley(x):
    d = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + math.e)
def _griewank(x):
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)
def _rastrigin(x): return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))
def _zakharov(x):
    s = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return float(np.sum(x ** 2) + s ** 2 + s ** 4)
def _dixon(x):
    i = np.arange(2, len(x) + 1)
    return float((x[0] - 1) ** 2 + np.sum(i * (2 * x[1:] ** 2 - x[:-1]) ** 2))
def _rosenbrock(x): return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


# nombre: (función, límite inferior, límite superior, media publicada, desviación publicada)
CASOS = {
    "f1 Sphere": (_sphere, -5.12, 5.12, 1.53e-55, 4.58e-55),
    "f2 Sum of squares": (_sumsq, -5.12, 5.12, 1.29e-53, 7.06e-53),
    "f8 Ackley": (_ackley, -32.76, 32.76, 4.44e-16, 2.00e-31),
    "f9 Griewank": (_griewank, -600.0, 600.0, 2.21e-03, 3.44e-03),
    "f11 Rastrigin": (_rastrigin, -5.12, 5.12, 0.0, 0.0),
    "f13 Zakharov": (_zakharov, -5.0, 10.0, 6.91e-51, 1.73e-50),
    "f14 Dixon-Price": (_dixon, -10.0, 10.0, 3.86e-32, 4.26e-33),
    "f15 Rosenbrock": (_rosenbrock, -5.0, 10.0, 1.44e-26, 7.25e-26),
}


# Tabla 4 (d = 30): nombre -> (función, lb, ub, media publicada, desviación publicada)
CASOS_ALTA = {
    "f1 Sphere": (_sphere, -5.12, 5.12, 4.01e+01, 1.37e+01),
    "f8 Ackley": (_ackley, -32.76, 32.76, 1.92e+01, 8.51e-01),
    "f11 Rastrigin": (_rastrigin, -5.12, 5.12, 2.37e+02, 4.69e+01),
    "f13 Zakharov": (_zakharov, -5.0, 10.0, 6.08e+02, 6.57e+01),
}


class _Problema:
    """Mínimo que exige el protocolo de competidores: limites, evaluar, fes."""

    def __init__(self, f, lb, ub, dim=D):
        self._f, self.limites = f, np.array([[lb, ub]] * dim, dtype=float)
        self.fes, self.mejor = 0, math.inf

    def evaluar(self, x):
        self.fes += 1
        v = self._f(np.asarray(x, dtype=float))
        self.mejor = min(self.mejor, v)
        return v


def main():
    ok = []
    print(f"GCO, d={D}, {MAX_FES} FES, {CORRIDAS} corridas")
    for nombre, (f, lb, ub, media_pub, std_pub) in CASOS.items():
        mejores = []
        for s in range(CORRIDAS):
            p = _Problema(f, lb, ub)
            gco.correr(p, MAX_FES, s)
            assert p.fes == MAX_FES
            mejores.append(p.mejor)
        media, std = float(np.mean(mejores)), float(np.std(mejores))
        if media_pub < 1e-10:
            bien = media < 1e-10
        else:
            bien = media <= media_pub + 2 * std_pub
        ok.append(bien)
        print(f"  {nombre:18s} {media:10.3g} ± {std:9.3g}   publicado {media_pub:9.3g} ± {std_pub:9.3g}"
              f"   {'OK' if bien else 'FUERA'}")
    assert all(ok), "GCO no reproduce algún resultado de la Tabla 2"
    print("\n[OK] GCO reproduce la Tabla 2 de Villaseñor et al. (2018)".encode("ascii", "replace").decode())

    print(f"\nGCO, d={D_ALTA}, {MAX_FES_ALTA} FES, {CORRIDAS_ALTA} corridas "
          f"(Tabla 4, INFORMATIVO: no se reproduce, ver docstring)")
    for nombre, (f, lb, ub, media_pub, std_pub) in CASOS_ALTA.items():
        mejores = []
        for s in range(CORRIDAS_ALTA):
            p = _Problema(f, lb, ub, dim=D_ALTA)
            gco.correr(p, MAX_FES_ALTA, s)
            assert p.fes == MAX_FES_ALTA
            mejores.append(p.mejor)
        media, std = float(np.mean(mejores)), float(np.std(mejores, ddof=1))
        tol = 3 * math.sqrt(std_pub ** 2 / 30 + std ** 2 / CORRIDAS_ALTA)
        bien = abs(media - media_pub) <= tol
        print(f"  {nombre:18s} {media:10.4g} +- {std:9.3g}   publicado {media_pub:9.4g} +- {std_pub:9.3g}"
              f"   |dif|={abs(media - media_pub):.3g} <= {tol:.3g}  {'OK' if bien else 'FUERA'}")


if __name__ == "__main__":
    main()
