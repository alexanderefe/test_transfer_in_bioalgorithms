"""
Test de los algoritmos de referencia por familia implementados a partir de
fuentes revisadas (docs/competidores.md §8):
  - Evolutivos: GA real (pymoo), jDE (NiaPy, parámetros de Brest 2006),
    JADE (mealpy corregido según Zhang y Sanderson 2009).
  - Enjambre: CLPSO (port de CLPSO_new_func.m), GWO (NiaPy), ABC (NiaPy
    corregido según ABC.C).
  - Inmunes: CLONALG (aisp), opt-aiNet (implementación propia), GCO (mealpy).

Verifica, en D = 10 y D = 20, que cada uno:
  - consume EXACTAMENTE MaxFES (con un MaxFES que no es múltiplo de la
    población),
  - deja una traza de convergencia monótona no creciente,
  - es determinista (misma semilla → misma corrida) y semillas distintas
    dan corridas distintas,
  - expone VERSION.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from experimentos.competidores import (  # noqa: E402
    abc_karaboga, clonalg_aisp, clpso, ga_pymoo, gco, gwo, jade, jde, opt_ainet)
from problems.cec2022_wrapper import ProblemaCEC2022  # noqa: E402

MODULOS = {"GA (pymoo)": ga_pymoo, "jDE": jde, "JADE": jade,
           "CLPSO": clpso, "GWO": gwo, "ABC (Karaboga)": abc_karaboga,
           "CLONALG (aisp)": clonalg_aisp, "opt-aiNet": opt_ainet,
           "GCO": gco}
MAX_FES = 3_001


def _correr(modulo, semilla, dim, funcion=4):
    p = ProblemaCEC2022(funcion, dim)
    p.configurar_checkpoints(MAX_FES)
    modulo.correr(p, MAX_FES, semilla)
    return p


def main():
    for dim in (10, 20):
        for nombre, mod in MODULOS.items():
            assert isinstance(mod.VERSION, str) and mod.VERSION
            p = _correr(mod, 3, dim)
            assert p.fes == MAX_FES, f"{nombre} D={dim}: {p.fes} FES, se esperaban {MAX_FES}"
            cps = p.error_checkpoints()
            assert all(cps[i] >= cps[i + 1] - 1e-9 for i in range(len(cps) - 1)), \
                f"{nombre} D={dim}: traza no monótona"
            a = _correr(mod, 11, dim).error_checkpoints()
            b = _correr(mod, 11, dim).error_checkpoints()
            c = _correr(mod, 12, dim).error_checkpoints()
            assert a == b, f"{nombre} D={dim}: misma semilla, corridas distintas"
            assert a != c, f"{nombre} D={dim}: semillas distintas, misma corrida"
            print(f"  D={dim} {nombre:15s} FES={p.fes}  error={cps[-1]:.4g}  [{mod.VERSION}]")
    print(f"\n[OK] {len(MODULOS)} algoritmos de referencia × 2 dimensiones")


if __name__ == "__main__":
    main()
