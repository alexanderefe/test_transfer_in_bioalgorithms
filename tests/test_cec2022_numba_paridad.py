"""
Fidelidad del evaluador CEC 2022 al código OFICIAL de la competición —
gate de aceptación de D-5, corregido en D-9
(docs/configuracion_experimental.md).

Hasta D-9 este test comparaba el backend Numba contra opfunu, y ambos
coincidían... porque el Numba portaba opfunu, cuya CEC 2022 difiere del
oficial en 11 de las 12 funciones (todas coinciden en el óptimo x*, por eso
no se notaba). Ahora la referencia es el código oficial, en dos niveles:

  1. ANCLAS: valores calculados con el C oficial (`cec22_test_func.cpp` de
     github.com/P-N-Suganthan/2022-SO-BO, compilado con `zig c++`) en dos
     puntos fijos por función × D: x = 0 y x = linspace(-90, 90, D).
  2. PARIDAD: el backend "referencia" (`problems/cec2022_referencia.py`,
     port Python oficial de los organizadores, verificado igual al C) en el
     óptimo, en puntos cerca del óptimo (donde están las diferencias que el
     test viejo no veía), aleatorios en el dominio y esquinas.

La tolerancia queda muy por debajo del umbral de empate del setup (1e-8):
una discrepancia de fórmula da errores grandes, no ruido de coma flotante.
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

import numpy as np

from problems.cec2022_wrapper import ProblemaCEC2022, DIMENSIONES_VALIDAS

N_ALEATORIOS = 100
TOLERANCIA_ABS = 1e-6
TOLERANCIA_REL = 1e-9

# (función, D): (f en x = 0, f en x = linspace(-90, 90, D)) — C oficial.
ANCLAS_C_OFICIAL = {
    (1, 10): (15908044999.492702, 5808610.049431784),
    (1, 20): (9558730232304.59, 377015740795.22),
    (2, 10): (11097.372890481096, 12255.139536130224),
    (2, 20): (7508.6777109481645, 23189.803584457604),
    (3, 10): (741.775494104428, 712.3393866270043),
    (3, 20): (760.3132407487321, 784.3216904227942),
    (4, 10): (911.9234884074399, 981.8243016616491),
    (4, 20): (1077.3586217236857, 1240.4670510446615),
    (5, 10): (3843.9382800867998, 18922.899548406873),
    (5, 20): (10492.485115390029, 34637.1872305956),
    (6, 10): (9850054875.054192, 26800837123.335133),
    (6, 20): (8859205369.3246, 32547151858.74569),
    (7, 10): (2929.254971040536, 2645.2810490288534),
    (7, 20): (2691.8786415840423, 2859.18326493834),
    (8, 10): (87756.64612737099, 1087322.7741276722),
    (8, 20): (225283.57615173256, 2065757.9157826875),
    (9, 10): (4768.752719488762, 5236.716414738338),
    (9, 20): (6618.138143224724, 10172.60191680483),
    (10, 10): (6852.886289733871, 3190.764183640262),
    (10, 20): (10921.290353661823, 4663.3341553743685),
    (11, 10): (5291.300260040884, 17509.785892718242),
    (11, 20): (10695.510621014344, 26989.431056053916),
    (12, 10): (4978.88844252468, 3168.9090698372806),
    (12, 20): (9228.009396206773, 5992.042087480062),
}


def _tolerancia_ok(v_ref: float, v_numba: float) -> bool:
    return abs(v_ref - v_numba) <= max(TOLERANCIA_ABS, TOLERANCIA_REL * abs(v_ref))


def _puntos_de_prueba(problema: ProblemaCEC2022) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    lo, hi = problema.limites[:, 0], problema.limites[:, 1]
    ndim = problema.ndim
    x_opt = np.asarray(problema.x_optimo, dtype=float)[:ndim]
    puntos = [x_opt]
    puntos += [x_opt + rng.normal(0.0, s, ndim) for s in (1e-3, 1e-1, 1.0, 10.0)]
    puntos += [rng.uniform(lo, hi) for _ in range(N_ALEATORIOS)]
    puntos.append(np.full(ndim, lo[0]))
    puntos.append(np.full(ndim, hi[0]))
    puntos.append(np.where(np.arange(ndim) % 2 == 0, lo, hi))
    return puntos


def verificar_funcion(numero_funcion: int, ndim: int) -> tuple[float, float]:
    """Anclas del C + paridad contra la referencia. Retorna
    (error_absoluto_max, error_relativo_max) contra la referencia."""
    problema_ref = ProblemaCEC2022(numero_funcion, ndim, backend="referencia")
    problema_numba = ProblemaCEC2022(numero_funcion, ndim, backend="numba")

    for x, v_c in zip((np.zeros(ndim), np.linspace(-90.0, 90.0, ndim)),
                      ANCLAS_C_OFICIAL[(numero_funcion, ndim)]):
        v_numba = problema_numba.evaluar(x)
        assert _tolerancia_ok(v_c, v_numba), (
            f"F{numero_funcion} D={ndim}: C oficial={v_c!r} vs numba={v_numba!r}")

    err_abs_max = 0.0
    err_rel_max = 0.0
    for x in _puntos_de_prueba(problema_ref):
        v_ref = problema_ref.evaluar(x)
        v_numba = problema_numba.evaluar(x)
        assert _tolerancia_ok(v_ref, v_numba), (
            f"F{numero_funcion} D={ndim}: referencia={v_ref!r} vs "
            f"numba={v_numba!r} en x={x!r} (diff={abs(v_ref - v_numba):.3e})"
        )
        # Ningún punto del dominio puede quedar por debajo de f(x*) (D-9).
        assert v_numba >= problema_numba.optimo_global - 1e-8, (
            f"F{numero_funcion} D={ndim}: f={v_numba} < f(x*)={problema_numba.optimo_global}")
        err_abs_max = max(err_abs_max, abs(v_ref - v_numba))
        if v_ref != 0.0:
            err_rel_max = max(err_rel_max, abs(v_ref - v_numba) / abs(v_ref))
    return err_abs_max, err_rel_max


def main():
    print(f"Fidelidad al CEC 2022 oficial — anclas del C + paridad contra la "
          f"referencia en óptimo, 4 vecindades del óptimo, {N_ALEATORIOS} "
          f"aleatorios y esquinas, por función×dim.\n")
    for numero_funcion in range(1, 13):
        for ndim in DIMENSIONES_VALIDAS:
            err_abs, err_rel = verificar_funcion(numero_funcion, ndim)
            print(f"  F{numero_funcion:2d} D={ndim:2d}  "
                  f"err_abs_max={err_abs:.3e}  err_rel_max={err_rel:.3e}  OK")
    print("\nFidelidad verificada: 12 funciones × 2 dimensiones.")


if __name__ == "__main__":
    main()
