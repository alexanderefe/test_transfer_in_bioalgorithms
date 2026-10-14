"""
Barrera de Wasserstein aplicada por el orquestador (D-11,
docs/configuracion_experimental.md).

Antes de D-11 el orquestador calculaba W1 solo para el log del Canal A y
nunca la comparaba contra θ_W: ninguna transferencia se abortaba y
`n_transferencias_abortadas_wasserstein` era siempre 0. Ahora la barrera se
evalúa al entrar a la Fase 3 y otra vez antes de escalar de Canal A a
Canal B; si W1 > θ_W se aborta el ciclo sin transferir nada.

Parte 1 — unidad (_evaluar_barrera_wasserstein):
  1a. poblaciones cercanas → permitida, sin efectos sobre el estado;
  1b. poblaciones en esquinas opuestas → abortada: contador, entrada en el
      log de Fase 3 con W1 y θ_W reales, ciclo limpio y cooldown aplicado.
Parte 2 — integración (F1, F11; D=10; MaxFES = 5·10⁴):
  2a. θ_W = 0 → toda transferencia se aborta: ni Canal A ni Canal B, cero
      instancias inyectadas, y el cooldown impide re-pagar la Fase 2 en el
      siguiente monitoreo;
  2b. θ_W = ∞ → ningún aborto, y el log de Fase 3 registra W1 y θ_W reales
      también en el Canal B (antes quedaban en 0.0).
"""

import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np

import middleware.orquestador as orq_mod
from bioalgorithms.de import DE
from bioalgorithms.pso import PSO
from middleware.barrera_seguridad import evaluar_barrera_wasserstein
from middleware.transferencia import CanalActivo, EstadoTransferencia
from problems.cec2022_wrapper import ProblemaCEC2022

CONFIGS_INTEGRACION = [(1, 10), (11, 10)]
MAX_FES_INTEGRACION = 50_000


def _orquestador(funcion: int, dim: int, max_fes: int, nombre: str):
    from experimentos.semillas import SEMILLAS

    class Orquestador(orq_mod.Orquestador):
        def _log(self, mensaje, nivel="INFO"):
            pass

    problema = ProblemaCEC2022(funcion, dim)
    ss_pso, ss_de = np.random.SeedSequence(SEMILLAS[0]).spawn(2)
    pso = PSO(problema, 30, dim, problema.limites, semilla=ss_pso, max_fes=max_fes,
              w_max=0.9, w_min=0.4, c1=2.0, c2=2.0)
    de = DE(problema, 30, dim, problema.limites, semilla=ss_de, max_fes=max_fes,
            F=0.2, CR=0.3)
    dir_logs = RAIZ / "tests" / "_tmp_resultados"
    dir_logs.mkdir(parents=True, exist_ok=True)
    orq = Orquestador(
        algoritmo_a=pso, algoritmo_b=de, limites=problema.limites, problema=problema,
        max_fes=max_fes, directorio_log=str(dir_logs), nombre_log=nombre,
        max_epochs_fastshap=3, silencioso=True,
    )
    return orq, pso, de


# ── Parte 1 ─────────────────────────────────────────────────────────────

def parte_1():
    print("=" * 70)
    print("PARTE 1 — _evaluar_barrera_wasserstein (unidad)")
    print("=" * 70)
    orq, pso, de = _orquestador(1, 10, MAX_FES_INTEGRACION, "barrera_w_unidad")
    pso.ejecutar_iteracion()
    de.ejecutar_iteracion()
    rng = np.random.default_rng(0)
    lim = orq.limites

    # 1a — cercanas: ambas poblaciones alrededor del mismo punto
    centro = lim.mean(axis=1)
    pso.poblacion = centro + rng.normal(0, 1, (30, 10))
    de.poblacion = centro + rng.normal(0, 1, (30, 10))
    orq._conocimiento_ciclo_actual = "ciclo"     # centinela: no debe tocarse
    estado = orq._estado_transferencia
    log_antes = len(orq.resumen.log_fase3)
    r = orq._evaluar_barrera_wasserstein(100, pso, de)
    assert r.transferencia_permitida
    assert orq.resumen.n_transferencias_abortadas_wasserstein == 0
    assert orq._conocimiento_ciclo_actual == "ciclo"
    assert orq._estado_transferencia is estado
    assert len(orq.resumen.log_fase3) == log_antes
    print(f"  1a. cercanas: W1={r.distancia_wasserstein:.2f} ≤ θ_W="
          f"{r.umbral_wasserstein:.2f} → permitida, sin efectos ... OK")

    # 1b — esquinas opuestas del hipercubo
    pso.poblacion = lim[:, 0] + rng.uniform(0, 1, (30, 10))
    de.poblacion = lim[:, 1] - rng.uniform(0, 1, (30, 10))
    orq._estado_transferencia.canal_activo = CanalActivo.CANAL_A
    orq._rol_fuente_activo, orq._rol_objetivo_activo = "a", "b"
    reloj = orq._reloj()
    r = orq._evaluar_barrera_wasserstein(200, pso, de)
    esperado = evaluar_barrera_wasserstein(pso.poblacion, de.poblacion, lim)
    assert not r.transferencia_permitida
    assert orq.resumen.n_transferencias_abortadas_wasserstein == 1
    assert orq._conocimiento_ciclo_actual is None
    assert orq._rol_fuente_activo is None and orq._rol_objetivo_activo is None
    assert orq._estado_transferencia.canal_activo == CanalActivo.NINGUNO
    assert orq._fin_cooldown == reloj + orq._cooldown_post_ciclo
    entrada = orq.resumen.log_fase3[-1]
    assert entrada["transferencia_permitida"] is False
    assert entrada["canal_utilizado"] == CanalActivo.NINGUNO.value
    assert entrada["iteracion"] == 200
    assert np.isclose(entrada["distancia_wasserstein"], esperado.distancia_wasserstein)
    assert np.isclose(entrada["umbral_wasserstein"], esperado.umbral_wasserstein)
    print(f"  1b. opuestas: W1={r.distancia_wasserstein:.2f} > θ_W="
          f"{r.umbral_wasserstein:.2f} → abortada, ciclo limpio, cooldown ... OK")


# ── Parte 2 ─────────────────────────────────────────────────────────────

def _correr_con_fraccion(funcion: int, dim: int, fraccion: float):
    original = orq_mod.evaluar_barrera_wasserstein
    orq_mod.evaluar_barrera_wasserstein = (
        lambda ps, pt, lim: original(ps, pt, lim, fraccion_umbral=fraccion))
    try:
        orq, _, _ = _orquestador(funcion, dim, MAX_FES_INTEGRACION,
                                 f"barrera_w_f{funcion}_{fraccion}")
        return orq, orq.ejecutar()
    finally:
        orq_mod.evaluar_barrera_wasserstein = original


def parte_2():
    print("\n" + "=" * 70)
    print(f"PARTE 2 — Barrera en el orquestador (D=10, MaxFES={MAX_FES_INTEGRACION})")
    print("=" * 70)
    for funcion, dim in CONFIGS_INTEGRACION:
        # 2a — θ_W = 0: todo ciclo que llegue a la Fase 3 se aborta
        orq, r = _correr_con_fraccion(funcion, dim, 0.0)
        assert r.n_transferencias_abortadas_wasserstein > 0, \
            f"F{funcion}: con θ_W = 0 la barrera debe abortar"
        assert r.n_transferencias_canal_a == 0 and r.n_transferencias_canal_b == 0
        assert r.n_instancias_inyectadas_total == 0
        assert all(not e["transferencia_permitida"] for e in r.log_fase3)
        # Cada aborto aplica el cooldown: la Fase 2 no puede activarse más
        # veces que las ventanas de cooldown que caben en el presupuesto.
        tope = MAX_FES_INTEGRACION // orq._cooldown_post_ciclo + 1
        assert r.n_activaciones_fase2 <= tope, \
            f"F{funcion}: {r.n_activaciones_fase2} activaciones de Fase 2 > {tope}"
        print(f"  F{funcion:<2d} θ_W=0: abortos={r.n_transferencias_abortadas_wasserstein} "
              f"fase2={r.n_activaciones_fase2} (≤{tope}), canal A/B=0, inyectadas=0 ... OK")

        # 2b — θ_W = ∞: nunca se aborta y el log lleva W1/θ_W reales
        orq, r = _correr_con_fraccion(funcion, dim, np.inf)
        assert r.n_transferencias_abortadas_wasserstein == 0
        assert r.n_transferencias_canal_a + r.n_transferencias_canal_b > 0, \
            f"F{funcion}: con θ_W = ∞ debe haber transferencias"
        assert all(e["transferencia_permitida"] and e["distancia_wasserstein"] > 0
                   and e["umbral_wasserstein"] == np.inf for e in r.log_fase3)
        print(f"  F{funcion:<2d} θ_W=∞: abortos=0, canal A={r.n_transferencias_canal_a} "
              f"B={r.n_transferencias_canal_b}, W1 registrado en todos ... OK")


def main():
    parte_1()
    parte_2()
    print("\n[OK] Barrera de Wasserstein en el orquestador (D-11)")


if __name__ == "__main__":
    main()
