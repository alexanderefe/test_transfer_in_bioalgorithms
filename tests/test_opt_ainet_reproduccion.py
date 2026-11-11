"""
Prueba de reproducción de opt-aiNet (experimentos/competidores/opt_ainet.py)
contra los resultados publicados por Timmis y Edmonds (GECCO 2004, «A
comment on opt-aiNET»), Tablas 1 y 2: mismas funciones, dominio [−2, 2]²
(factor de escala 1), parámetros del paper (N = 20, Nc = 10, β = 100,
σs = 0,2, d = 40 %), terminación original (una supresión que no elimina
células, o 500 iteraciones) y 50 corridas.

Valores publicados (media ± desviación; «published» = de Castro y Timmis
2002, «recorded» = código original re-ejecutado, «new» = versión corregida,
que es la que se implementa aquí):

  Multi   G(x,y) = x·sin(4πx) − y·sin(4πy + π) + 1   (máximo global único)
    Peaks: 56,10 ± 4,36 | 57,2 ± 13,83 | 56,5 ± 17
    ItG  : 53,50 ± 47,19 | 58,8 ± 45,18 | 212,8 ± 140
  Roots   g(z) = 1 / (1 + |z⁶ − 1|), z = x + iy       (seis máximos = 1)
    Peaks: 6,0 | 5,9 ± 0,32 | 5,2 ± 0,79
    ItG  : 86,89 ± 34,31 | 93,2 ± 31,99 | 149,2 ± 87,1

ItG = iteraciones hasta alcanzar el óptimo global. El paper no da la
tolerancia: aquí, a menos de 0,1 % del valor máximo. Peaks en Multi: tamaño
de la red tras la última supresión, antes de insertar las células
aleatorias (el original termina ahí; cada célula es el mejor punto de su
vecindad σs). Peaks en Roots: raíces de la unidad con una célula a
distancia < 0,1 y g ≥ 0,9.

Criterio de aceptación (amplio, por la alta varianza que reporta el propio
paper): los promedios deben caer dentro del rango [min − 1·std, max + 1·std]
de las tres columnas publicadas.
"""

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from experimentos.competidores.opt_ainet import OptAiNet  # noqa: E402

CORRIDAS, MAX_ITER = 50, 500
LB, UB = np.array([-2.0, -2.0]), np.array([2.0, 2.0])


def multi(p):
    x, y = p
    return x * np.sin(4 * np.pi * x) - y * np.sin(4 * np.pi * y + np.pi) + 1


def roots(p):
    z = complex(p[0], p[1])
    return 1.0 / (1.0 + abs(z ** 6 - 1))


RAICES = np.array([[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)])


def _maximo_multi():
    g = np.linspace(-2, 2, 4001)
    X, Y = np.meshgrid(g, g)
    return float(np.max(X * np.sin(4 * np.pi * X) - Y * np.sin(4 * np.pi * Y + np.pi) + 1))


def correr(funcion, maximo, semilla):
    """Corre con la terminación original. Devuelve (red, ItG, ItC)."""
    red = OptAiNet(lambda p: -funcion(p), LB, UB, np.random.default_rng(semilla))
    itg = None
    for _ in range(MAX_ITER):
        suprimidas = red.iterar()
        if itg is None and -red.f.min() >= maximo * (1 - 1e-3):
            itg = red.iteracion
        if suprimidas == 0:
            break
    return red, itg, red.iteracion


def picos_roots(red):
    vals = -red.f
    return sum(bool(np.any((np.linalg.norm(red.X - r, axis=1) < 0.1) & (vals >= 0.9)))
               for r in RAICES)


def rango_publicado(columnas):
    return (min(m - s for m, s in columnas), max(m + s for m, s in columnas))


def resumen(nombre, valores, columnas):
    v = np.array([x for x in valores if x is not None], dtype=float)
    lo, hi = rango_publicado(columnas)
    ok = lo <= v.mean() <= hi
    print(f"  {nombre:12s} {v.mean():7.2f} ± {v.std():6.2f}  (n={len(v)})   "
          f"publicado: [{lo:.1f}, {hi:.1f}]  {'OK' if ok else 'FUERA'}")
    return ok


def main():
    m_max = _maximo_multi()
    res = {"multi": [correr(multi, m_max, s) for s in range(CORRIDAS)],
           "roots": [correr(roots, 1.0, s) for s in range(CORRIDAS)]}

    print(f"Multi (máximo {m_max:.4f})")
    ok = [resumen("Peaks", [r.tamano_tras_supresion for r, _, _ in res["multi"]],
                  [(56.10, 4.36), (57.2, 13.83), (56.5, 17)]),
          resumen("ItG", [g for _, g, _ in res["multi"]],
                  [(53.50, 47.19), (58.8, 45.18), (212.8, 140)])]
    encontrado = sum(g is not None for _, g, _ in res["multi"])
    print(f"  óptimo global alcanzado en {encontrado}/{CORRIDAS} corridas")

    print("Roots")
    ok += [resumen("Peaks", [picos_roots(r) for r, _, _ in res["roots"]],
                   [(6.0, 0.0), (5.9, 0.32), (5.2, 0.79)]),
           resumen("ItG", [g for _, g, _ in res["roots"]],
                   [(86.89, 34.31), (93.2, 31.99), (149.2, 87.1)])]

    assert encontrado >= 0.9 * CORRIDAS, "opt-aiNet no alcanza el óptimo de Multi"
    assert all(ok), "algún promedio cae fuera del rango publicado"
    print("\n[OK] opt-aiNet reproduce los resultados publicados (Multi y Roots)")


if __name__ == "__main__":
    main()
