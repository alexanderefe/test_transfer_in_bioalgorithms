"""
Test de la integración BBOB (README_BBOB.md, opción A).

Verifica:
  1. `ProblemaBBOB`: límites [-5, 5]^D, f(x*) de COCO, ningún punto por
     debajo de f(x*), contador de FES y checkpoints como `ProblemaCEC2022`.
  2. Los 12 algoritmos en las 6 dimensiones de BBOB (2, 3, 5, 10, 20, 40):
     no se pasan de MaxFES (los competidores lo consumen exacto), traza
     monótona, estado "ok".
  3. Determinismo en D = 2 (misma instancia → misma corrida).
  4. El middleware activa la Fase 2 (XGBoost + FastSHAP) en D = 2 y D = 40.
  5. Observador de COCO: escribe en la carpeta pedida y, al repetir la
     corrida, la reemplaza (no crea `-0001`, que cocopp contaría dos veces).

    python tests/test_bbob.py
"""

import shutil
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from problems.bbob_wrapper import DIMENSIONES_BBOB, ProblemaBBOB  # noqa: E402
from experimentos.corrida_bbob import correr_corrida_bbob  # noqa: E402
from experimentos.grid_bbob import ALGORITMOS_BBOB  # noqa: E402
from experimentos.competidores import COMPETIDORES  # noqa: E402

TMP = RAIZ / "tests" / "_tmp_resultados" / "bbob"


def _monotona(cps):
    return all(cps[i] >= cps[i + 1] - 1e-12 for i in range(len(cps) - 1))


def test_problema():
    rng = np.random.default_rng(0)
    for d in DIMENSIONES_BBOB:
        for f in (1, 8, 15, 24):
            p = ProblemaBBOB(f, d, 3)
            assert p.limites.shape == (d, 2)
            assert np.all(p.limites[:, 0] == -5) and np.all(p.limites[:, 1] == 5)
            assert np.isfinite(p.optimo_global)
            X = rng.uniform(-5, 5, size=(200, d))
            vals = np.array([p(x) for x in X])
            assert np.all(vals >= p.optimo_global - 1e-9), f"f{f} D={d}: punto bajo f(x*)"
            assert p.fes == 200
            p.cerrar()
    # Checkpoints: misma semántica que ProblemaCEC2022.
    p = ProblemaBBOB(1, 5, 1)
    p.configurar_checkpoints(1000)
    for x in rng.uniform(-5, 5, size=(1000, 5)):
        p(x)
    cps = p.error_checkpoints()
    assert len(cps) == 120 and _monotona(cps) and cps[-1] >= 0
    p.cerrar()
    print("  [OK] ProblemaBBOB: límites, f(x*), FES y checkpoints")


def test_algoritmos():
    for alg in ALGORITMOS_BBOB:
        for d in DIMENSIONES_BBOB:
            max_fes = 500 * d + 7            # no múltiplo de la población
            if alg == "middleware":
                max_fes = 2_000 * d + 7      # que alcance a pasar el warm-up
            fila, _ = correr_corrida_bbob(alg, 15, d, 1, max_fes, dir_salida=TMP,
                                          observador=False, max_epochs_fastshap=3)
            assert fila["estado"] == "ok", f"{alg} D={d}: {fila['mensaje_error']}"
            assert fila["fes_consumidas"] <= max_fes, f"{alg} D={d}: {fila['fes_consumidas']} > {max_fes}"
            if alg in COMPETIDORES:
                assert fila["fes_consumidas"] == max_fes, f"{alg} D={d}: no consume MaxFES exacto"
            assert _monotona(fila["error_checkpoints"]), f"{alg} D={d}: traza no monótona"
            assert fila["benchmark"] == "BBOB" and fila["run_id"].startswith("bbob__")
        print(f"  [OK] {alg:11s} D = {DIMENSIONES_BBOB}")


def test_determinismo():
    for alg in ALGORITMOS_BBOB:
        mf = 2_000 if alg != "middleware" else 20_000
        a, _ = correr_corrida_bbob(alg, 3, 2, 4, mf, dir_salida=TMP, observador=False,
                                   max_epochs_fastshap=3)
        b, _ = correr_corrida_bbob(alg, 3, 2, 4, mf, dir_salida=TMP, observador=False,
                                   max_epochs_fastshap=3)
        assert a["error_checkpoints"] == b["error_checkpoints"], f"{alg}: no determinista"
    print("  [OK] determinismo de los 12 en D=2")


def test_middleware_fase2():
    for d, mf in ((2, 20_000), (40, 200_000)):
        fila, prop = correr_corrida_bbob("middleware", 15, d, 1, mf, dir_salida=TMP,
                                         observador=False, max_epochs_fastshap=3)
        assert fila["estado"] == "ok", fila["mensaje_error"]
        assert prop is not None and prop["n_activaciones_fase2"] > 0, \
            f"D={d}: el middleware no activó la Fase 2"
        print(f"  [OK] middleware D={d}: {prop['n_activaciones_fase2']} activaciones de Fase 2, "
              f"canal A={prop['n_canal_a']}, canal B={prop['n_canal_b']}, "
              f"abortos W={prop['n_abortos_wasserstein']}, error={fila['error']:.4g}")


def test_observador():
    salida = TMP / "obs"
    if salida.exists():
        shutil.rmtree(salida)
    for _ in range(2):
        fila, _ = correr_corrida_bbob("pso", 2, 3, 1, 3_000, dir_salida=salida)
        assert fila["estado"] == "ok", fila["mensaje_error"]
    padre = salida / "coco" / "pso"
    carpetas = sorted(c.name for c in padre.iterdir())
    assert carpetas == ["f02_d03_i01_fes3000"], carpetas
    assert (padre / carpetas[0] / "bbobexp_f2.info").is_file()
    assert any((padre / carpetas[0] / "data_f2").glob("*.tdat"))
    print("  [OK] observador de COCO: carpeta exacta, sin duplicados al repetir")


def main():
    test_problema()
    test_observador()
    test_algoritmos()
    test_determinismo()
    test_middleware_fase2()
    print("\n[OK] integración BBOB: 12 algoritmos × 6 dimensiones")


if __name__ == "__main__":
    main()
