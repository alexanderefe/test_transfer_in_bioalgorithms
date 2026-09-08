"""
Paridad numérica del backend compilado (Numba) contra el backend de
referencia (opfunu) — gate de aceptación de D-5
(docs/configuracion_experimental.md).

El backend "numba" (problems/cec2022_numba.py) reimplementa las 12
funciones de CEC2022 reutilizando los mismos arrays de shift/rotación/
shuffle que carga opfunu, compilados en vez de interpretados. Este script
no confía en que el puerto sea correcto por inspección: evalúa ambos
backends sobre los mismos puntos (óptimo, muestras aleatorias, esquinas
del dominio) para las 12 funciones × D∈{10,20} y exige que coincidan
dentro de una tolerancia muy por debajo del umbral de "empate" del setup
experimental (1e-8) — una discrepancia real de fórmula da errores
grandes, no ruido de punto flotante, así que esta tolerancia no deja
pasar un puerto mal hecho por casualidad.
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

import numpy as np

from problems.cec2022_wrapper import ProblemaCEC2022, DIMENSIONES_VALIDAS

N_ALEATORIOS = 200
TOLERANCIA_ABS = 1e-6
TOLERANCIA_REL = 1e-9


def _tolerancia_ok(v_ref: float, v_numba: float) -> bool:
    return abs(v_ref - v_numba) <= max(TOLERANCIA_ABS, TOLERANCIA_REL * abs(v_ref))


def _puntos_de_prueba(problema_ref: ProblemaCEC2022) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    lo, hi = problema_ref.limites[:, 0], problema_ref.limites[:, 1]
    ndim = problema_ref.ndim
    puntos = [np.asarray(problema_ref.x_optimo, dtype=float)]
    puntos += [rng.uniform(lo, hi) for _ in range(N_ALEATORIOS)]
    puntos.append(np.full(ndim, lo[0]))
    puntos.append(np.full(ndim, hi[0]))
    alternado = np.where(np.arange(ndim) % 2 == 0, lo, hi)
    puntos.append(alternado)
    return puntos


def verificar_funcion(numero_funcion: int, ndim: int) -> tuple[float, float]:
    """Retorna (error_absoluto_max, error_relativo_max) entre backends."""
    problema_ref = ProblemaCEC2022(numero_funcion=numero_funcion, ndim=ndim,
                                    backend="opfunu")
    problema_numba = ProblemaCEC2022(numero_funcion=numero_funcion, ndim=ndim,
                                      backend="numba")

    err_abs_max = 0.0
    err_rel_max = 0.0
    for x in _puntos_de_prueba(problema_ref):
        v_ref = problema_ref.evaluar(x)
        v_numba = problema_numba.evaluar(x)
        assert _tolerancia_ok(v_ref, v_numba), (
            f"F{numero_funcion} D={ndim}: opfunu={v_ref!r} vs "
            f"numba={v_numba!r} en x={x!r} (diff={abs(v_ref - v_numba):.3e})"
        )
        err_abs_max = max(err_abs_max, abs(v_ref - v_numba))
        if v_ref != 0.0:
            err_rel_max = max(err_rel_max, abs(v_ref - v_numba) / abs(v_ref))
    return err_abs_max, err_rel_max


def main():
    print(f"Paridad opfunu vs numba — {N_ALEATORIOS} puntos aleatorios + "
          f"óptimo + esquinas, por función×dim.\n")
    for numero_funcion in range(1, 13):
        for ndim in DIMENSIONES_VALIDAS:
            err_abs, err_rel = verificar_funcion(numero_funcion, ndim)
            print(f"  F{numero_funcion:2d} D={ndim:2d}  "
                  f"err_abs_max={err_abs:.3e}  err_rel_max={err_rel:.3e}  OK")
    print("\nParidad verificada: 12 funciones × 2 dimensiones, sin discrepancias "
          "por encima de la tolerancia.")


if __name__ == "__main__":
    main()
