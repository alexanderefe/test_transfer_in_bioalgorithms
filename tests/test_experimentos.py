"""
Smoke test del harness experimental sobre F1/D10 (la función más barata):
  1. algoritmo propuesto ("middleware"), con presupuesto y FastSHAP mínimos.
  2. PSO y DE SOLOS (D-4: los algoritmos base sin middleware).
  3. Los competidores externos (D-8): IPOP-CMA-ES, MadDE, NL-SHADE-RSP.

Verifica que:
  - todas las corridas terminan en estado "ok",
  - la fila trae todas las columnas del esquema §11.2,
  - el error es no negativo y la traza de convergencia tiene un punto por
    fracción de FRACCIONES_CHECKPOINT,
  - la traza es monótona no creciente y su último valor == el error final,
  - la fila §11.3 (diagnósticos del middleware) existe SOLO para "middleware",
  - los algoritmos que corren solos consumen MaxFES exacto, y los
    competidores externos son deterministas (misma semilla, mismo error) y
    registran la versión de su implementación.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from experimentos.corrida import correr_corrida
from experimentos.competidores import COMPETIDORES
from experimentos.esquema import COLUMNAS_CORRIDA, COLUMNAS_PROPUESTO, FAMILIA_ALGORITMO
from problems.cec2022_wrapper import FRACCIONES_CHECKPOINT


def main():
    salida = RAIZ / "tests" / "_tmp_resultados"
    fila, fila_prop = correr_corrida(
        "middleware", funcion=1, dim=10, max_fes=3000, idx_semilla=0,
        max_epochs_fastshap=3, dir_salida=salida,
    )

    print("estado:", fila["estado"])
    print("error :", fila["error"])
    print("checkpoints:", [f"{v:.3g}" for v in fila["error_checkpoints"]])
    print("fes   :", fila["fes_consumidas"], "/ 3000")
    if fila_prop is not None:
        print("fes_a/b:", fila_prop["fes_a"], "/", fila_prop["fes_b"],
              "| fase2:", fila_prop["n_activaciones_fase2"],
              "| primera activación:", fila_prop["fes_primera_activacion"])

    assert fila["estado"] == "ok", fila["mensaje_error"]
    assert set(COLUMNAS_CORRIDA) == set(fila), "columnas §11.2 no coinciden"
    assert fila["error"] >= 0.0
    assert fila["f_opt"] == 300.0, "F1 tiene f(x*) = 300"

    cps = fila["error_checkpoints"]
    assert len(cps) == len(FRACCIONES_CHECKPOINT), \
        f"esperados {len(FRACCIONES_CHECKPOINT)} checkpoints, hay {len(cps)}"
    assert all(cps[i] >= cps[i + 1] - 1e-9 for i in range(len(cps) - 1)), \
        "la traza de convergencia debe ser monótona no creciente"
    assert abs(cps[-1] - fila["error"]) < 1e-6, \
        "el último checkpoint debe coincidir con el error final"

    assert fila_prop is not None
    assert set(COLUMNAS_PROPUESTO) == set(fila_prop)
    assert fila_prop["fes_a"] + fila_prop["fes_b"] == fila["fes_consumidas"]
    # Con MaxFES=3000 el middleware no alcanza a activarse (D-1d)
    assert fila_prop["fes_primera_activacion"] != fila_prop["fes_primera_activacion"], \
        "con MaxFES=3000 fes_primera_activacion debe ser NaN"

    print("\n[OK] smoke del algoritmo propuesto")

    # ── D-4: PSO y DE solos · D-8: competidores externos ────────────────
    for algo in ("pso", "de", *COMPETIDORES):
        f, prop = correr_corrida(algo, funcion=1, dim=10, max_fes=3000,
                                 idx_semilla=0, dir_salida=salida)
        print(f"\n[{algo}] estado={f['estado']} familia={f['familia']} "
              f"error={f['error']:.4g} fes={f['fes_consumidas']} version={f['version']}")
        assert f["estado"] == "ok", f["mensaje_error"]
        assert f["familia"] == FAMILIA_ALGORITMO[algo]
        assert f["error"] >= 0.0
        assert f["fes_consumidas"] == 3000, \
            "los algoritmos solos no comparten presupuesto: deben consumir MaxFES exacto"
        cps = f["error_checkpoints"]
        assert len(cps) == len(FRACCIONES_CHECKPOINT)
        assert all(cps[i] >= cps[i + 1] - 1e-9 for i in range(len(cps) - 1))
        assert abs(cps[-1] - f["error"]) < 1e-6
        assert prop is None, "sin middleware no hay fila §11.3"
        if algo in COMPETIDORES:
            assert f"|alg:{COMPETIDORES[algo][0].VERSION}" in f["version"]
            f2, _ = correr_corrida(algo, funcion=1, dim=10, max_fes=3000,
                                   idx_semilla=0, dir_salida=salida)
            assert f2["error"] == f["error"], f"{algo} no es determinista"

    print("\n[OK] smoke del harness experimental "
          f"(middleware + pso + de + {' + '.join(COMPETIDORES)})")


if __name__ == "__main__":
    main()
