"""
Test de los adaptadores de NiaPy y mealpy
(experimentos/competidores/adaptador_niapy.py y adaptador_mealpy.py).

Usa algoritmos que pasaron la revisión del paso 2 (GWO y jDE de NiaPy,
JADE de mealpy) solo como vehículo; no fija sus parámetros finales.

Verifica que:
  - la corrida consume EXACTAMENTE MaxFES (también con un MaxFES que no es
    múltiplo de la población, donde mealpy por sí solo se pasaría),
  - misma semilla → misma corrida; semillas distintas → corridas distintas,
  - una excepción de la función objetivo se relanza aunque la corrida esté
    en un hilo secundario (NiaPy la esconde fuera del hilo principal),
  - el historial interno de mealpy no crece con las épocas,
  - el adaptador de mealpy rechaza algoritmos cuyo evolve usa self.epoch.
"""

import sys
import threading
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from mealpy.evolutionary_based.DE import JADE  # noqa: E402
from mealpy.swarm_based.PSO import CL_PSO  # noqa: E402
from niapy.algorithms.basic import GreyWolfOptimizer  # noqa: E402
from niapy.algorithms.modified import SelfAdaptiveDifferentialEvolution  # noqa: E402

from experimentos.competidores import adaptador_mealpy  # noqa: E402
from experimentos.competidores.adaptador_mealpy import correr_mealpy  # noqa: E402
from experimentos.competidores.adaptador_niapy import correr_niapy  # noqa: E402
from problems.cec2022_wrapper import ProblemaCEC2022  # noqa: E402

MAX_FES = 3_001          # no múltiplo de las poblaciones usadas (20 y 30)
CASOS = {
    "NiaPy GWO": (correr_niapy, GreyWolfOptimizer, {"population_size": 20}),
    "NiaPy jDE": (correr_niapy, SelfAdaptiveDifferentialEvolution, {"population_size": 20}),
    "mealpy JADE": (correr_mealpy, JADE, {"pop_size": 30}),
}


def _correr(caso, semilla, funcion=1, dim=10, max_fes=MAX_FES):
    adaptador, clase, params = CASOS[caso]
    problema = ProblemaCEC2022(funcion, dim)
    problema.configurar_checkpoints(max_fes)
    adaptador(clase, problema, max_fes, semilla, **params)
    return problema


class _ErrorObjetivo(RuntimeError):
    pass


class _ProblemaQueFalla(ProblemaCEC2022):
    def __init__(self, *a, falla_en=500, **k):
        super().__init__(*a, **k)
        self._falla_en = falla_en

    def evaluar(self, x):
        if self.fes >= self._falla_en:
            raise _ErrorObjetivo("falla simulada")
        return super().evaluar(x)


def test_presupuesto_exacto():
    # 3001: el presupuesto se agota a mitad de generación (PresupuestoAgotado).
    # 3000: múltiplo de las poblaciones (20 y 30), se agota justo al cerrar
    # una generación y la librería termina por su propio criterio.
    for max_fes in (MAX_FES, 3_000):
        for caso in CASOS:
            p = _correr(caso, semilla=1, max_fes=max_fes)
            assert p.fes == max_fes, f"{caso}: consumió {p.fes} FES, se esperaban {max_fes}"
            cps = p.error_checkpoints()
            assert all(cps[i] >= cps[i + 1] - 1e-9 for i in range(len(cps) - 1))
            print(f"  {caso:12s} MaxFES={max_fes}  FES={p.fes}  error final={cps[-1]:.4g}")


def test_determinismo():
    for caso in CASOS:
        a = _correr(caso, semilla=7).error_checkpoints()
        b = _correr(caso, semilla=7).error_checkpoints()
        c = _correr(caso, semilla=8).error_checkpoints()
        assert a == b, f"{caso}: misma semilla dio corridas distintas"
        assert a != c, f"{caso}: semillas distintas dieron la misma corrida"


def test_excepcion_en_hilo_secundario():
    for caso in CASOS:
        adaptador, clase, params = CASOS[caso]
        capturada = []

        def corrida():
            p = _ProblemaQueFalla(1, 10)
            p.configurar_checkpoints(MAX_FES)
            try:
                adaptador(clase, p, MAX_FES, 0, **params)
            except _ErrorObjetivo as e:
                capturada.append(e)

        h = threading.Thread(target=corrida)
        h.start()
        h.join()
        assert capturada, f"{caso}: la excepción de la función objetivo se perdió"


def test_historial_mealpy_acotado():
    adaptador, clase, params = CASOS["mealpy JADE"]
    problema = ProblemaCEC2022(1, 10)
    problema.configurar_checkpoints(MAX_FES)
    instancias = []
    original = clase.__init__

    def init_registrado(self, *a, **k):
        original(self, *a, **k)
        instancias.append(self)
    clase.__init__ = init_registrado
    try:
        adaptador(clase, problema, MAX_FES, 0, **params)
    finally:
        clase.__init__ = original
    hist = instancias[-1].history
    largos = {n: len(getattr(hist, n)) for n in adaptador_mealpy._LISTAS_HISTORIAL}
    assert max(largos.values()) <= 2, f"historial de mealpy creció: {largos}"
    assert not hist.list_global_best_fit and not hist.list_diversity


def test_mealpy_rechaza_calendario_por_epoca():
    problema = ProblemaCEC2022(1, 10)
    try:
        correr_mealpy(CL_PSO, problema, MAX_FES, 0, pop_size=20)
    except ValueError:
        return
    raise AssertionError("CL_PSO usa self.epoch en evolve y debió rechazarse")


def main():
    pruebas = [test_presupuesto_exacto, test_determinismo,
               test_excepcion_en_hilo_secundario, test_historial_mealpy_acotado,
               test_mealpy_rechaza_calendario_por_epoca]
    for prueba in pruebas:
        print(f"- {prueba.__name__}")
        prueba()
    print(f"\n[OK] {len(pruebas)} pruebas de los adaptadores NiaPy/mealpy")


if __name__ == "__main__":
    main()
