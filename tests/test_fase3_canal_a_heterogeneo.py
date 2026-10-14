"""
Canal A heterogéneo PSO↔DE en el espacio latente común (D-10,
docs/configuracion_experimental.md).

Antes de D-10 el Canal A solo interpolaba hiperparámetros con el MISMO
nombre en fuente y objetivo: entre PSO {w, c1, c2} y DE {F, CR} el payload
salía vacío y el orquestador escalaba siempre directo a Canal B. Ahora cada
algoritmo traduce sus hiperparámetros a un espacio común
(exploración: w ↔ F; adopción: c2/(c1+c2) ↔ CR) y el middleware interpola
ahí con el RMP, en el sentido que fijen los roles dinámicos.

Parte 1 — unidad (exacta, sin optimizar):
  1a. ida y vuelta por el espacio común = identidad (PSO y DE);
  1b. PSO→DE: F y CR se mueven exactamente rmp·(z_fuente − z_objetivo);
  1c. DE→PSO: el calendario de inercia se DESPLAZA (misma pendiente) y la
      suma c1 + c2 se conserva;
  1d. rmp = 0 no cambia nada; rmp = 1 lleva al objetivo a la posición
      latente del fuente;
  1e. PSO→PSO ahora sí transfiere w (antes se descartaba en silencio);
  1f. el desplazamiento del calendario persiste en los pasos siguientes;
  1g. un algoritmo sin espacio común deja el Canal A inerte (payload {}).
Parte 2 — integración en el orquestador (F1, F2, F5, F11; D=10;
  MaxFES = 10⁵): el Canal A ocurre en AMBOS sentidos, nunca queda inerte
  entre PSO y DE, y cada aplicación acerca al objetivo al fuente en el
  espacio común.
Parte 3 — efecto (solo se reporta, no se exige): las mismas corridas con y
  sin Canal A heterogéneo (desactivado devolviendo {} en a_espacio_comun,
  el comportamiento previo a D-10): error final y resultado de cada
  ventana de Canal A (exitoso / insuficiente → Canal B).
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np

from bioalgorithms.base import AlgoritmoBioinspirado, DIM_ADOPCION, DIM_EXPLORACION
from bioalgorithms.de import DE
from bioalgorithms.pso import PSO
from middleware.transferencia import aplicar_canal_a
from problems.cec2022_wrapper import ProblemaCEC2022

TOL = 1e-12
CONFIGS_INTEGRACION = [(1, 10), (2, 10), (5, 10), (11, 10)]
MAX_FES_INTEGRACION = 100_000


def _iniciados(semilla: int = 0):
    """PSO y DE con la configuración congelada, ya inicializados."""
    p = ProblemaCEC2022(1, 10)
    pso = PSO(p, 30, 10, p.limites, semilla=semilla, max_fes=50_000,
              w_max=0.9, w_min=0.4, c1=2.0, c2=2.0)
    de = DE(p, 30, 10, p.limites, semilla=semilla + 1, max_fes=50_000, F=0.2, CR=0.3)
    for _ in range(3):
        pso.ejecutar_iteracion()
        de.ejecutar_iteracion()
    return pso, de


# ── Parte 1 ─────────────────────────────────────────────────────────────

def parte_1():
    print("=" * 70)
    print("PARTE 1 — Mapas del espacio común e interpolación (exacto)")
    print("=" * 70)

    # 1a — ida y vuelta
    pso, de = _iniciados()
    hp_pso = {"w": 0.73, "c1": 1.2, "c2": 2.8}
    pso.c1, pso.c2 = hp_pso["c1"], hp_pso["c2"]
    vuelta = pso.desde_espacio_comun(pso.a_espacio_comun(hp_pso))
    assert all(abs(vuelta[k] - hp_pso[k]) < TOL for k in hp_pso), vuelta
    hp_de = {"F": 0.2, "CR": 0.3}
    vuelta = de.desde_espacio_comun(de.a_espacio_comun(hp_de))
    assert all(abs(vuelta[k] - hp_de[k]) < TOL for k in hp_de), vuelta
    print("  1a ida y vuelta PSO y DE = identidad ............................ OK")

    # 1b — PSO (fuente) → DE (objetivo)
    pso, de = _iniciados()
    rmp = 0.05
    escape_pso = {"w": 0.65, "c1": 2.0, "c2": 2.0}
    z_src = pso.a_espacio_comun(escape_pso)
    z_obj = de.a_espacio_comun(de.obtener_hiperparametros_actuales())
    payload = aplicar_canal_a(de, escape_pso, rmp, pso)
    assert set(payload) == {"F", "CR"}, payload
    z_nuevo = de.a_espacio_comun(de.obtener_hiperparametros_actuales())
    for dim in (DIM_EXPLORACION, DIM_ADOPCION):
        esperado = z_obj[dim] + rmp * (z_src[dim] - z_obj[dim])
        assert abs(z_nuevo[dim] - esperado) < TOL, (dim, z_nuevo[dim], esperado)
    print(f"  1b PSO→DE (rmp={rmp}): F 0.2000→{de.F:.4f}, CR 0.3000→{de.CR:.4f} "
          f"................ OK")

    # 1c — DE (fuente) → PSO (objetivo)
    pso, de = _iniciados()
    pendiente = pso.w_max - pso.w_min
    suma_c = pso.c1 + pso.c2
    w_antes = pso.w_actual
    escape_de = {"F": 0.9, "CR": 0.9}
    z_src = de.a_espacio_comun(escape_de)
    z_obj = pso.a_espacio_comun(pso.obtener_hiperparametros_actuales())
    payload = aplicar_canal_a(pso, escape_de, rmp, de)
    assert set(payload) == {"w", "c1", "c2"}, payload
    delta = payload["w"] - w_antes
    esperado_delta = rmp * (z_src[DIM_EXPLORACION] - z_obj[DIM_EXPLORACION]) * (
        PSO.W_REF[1] - PSO.W_REF[0])
    assert abs(delta - esperado_delta) < TOL, (delta, esperado_delta)
    assert abs((pso.w_max - pso.w_min) - pendiente) < TOL, "el calendario debe desplazarse, no deformarse"
    assert abs(pso.w_max - (0.9 + delta)) < TOL and abs(pso.w_min - (0.4 + delta)) < TOL
    assert abs((pso.c1 + pso.c2) - suma_c) < TOL, "c1 + c2 debe conservarse"
    esperado_adop = z_obj[DIM_ADOPCION] + rmp * (z_src[DIM_ADOPCION] - z_obj[DIM_ADOPCION])
    assert abs(pso.c2 / (pso.c1 + pso.c2) - esperado_adop) < TOL
    print(f"  1c DE→PSO (rmp={rmp}): calendario w desplazado {delta:+.4f} "
          f"(pendiente intacta), c1+c2 = {pso.c1 + pso.c2:.2f} ....... OK")

    # 1d — extremos del RMP
    pso, de = _iniciados()
    antes = de.obtener_hiperparametros_actuales()
    aplicar_canal_a(de, {"w": 0.8, "c1": 1.0, "c2": 3.0}, 0.0, pso)
    assert de.obtener_hiperparametros_actuales() == antes, "rmp=0 no debe cambiar nada"
    escape = {"w": 0.8, "c1": 1.0, "c2": 3.0}
    aplicar_canal_a(de, escape, 1.0, pso)
    z_de = de.a_espacio_comun(de.obtener_hiperparametros_actuales())
    z_ps = pso.a_espacio_comun(escape)
    assert all(abs(z_de[d] - z_ps[d]) < TOL for d in z_ps), (z_de, z_ps)
    print("  1d rmp=0 sin cambio; rmp=1 iguala la posición latente del fuente .. OK")

    # 1e — PSO → PSO transfiere w (antes la clave "w" se descartaba)
    pso_obj, _ = _iniciados(semilla=10)
    pso_src, _ = _iniciados(semilla=20)
    w_max_antes = pso_obj.w_max
    payload = aplicar_canal_a(pso_obj, {"w": 0.4, "c1": 2.0, "c2": 2.0}, 0.5, pso_src)
    assert "w" in payload and pso_obj.w_max < w_max_antes, (payload, pso_obj.w_max)
    print(f"  1e PSO→PSO: w_max {w_max_antes:.4f}→{pso_obj.w_max:.4f} (antes se ignoraba) .... OK")

    # 1f — el desplazamiento persiste en los pasos siguientes
    pso, de = _iniciados()
    aplicar_canal_a(pso, {"F": 0.2, "CR": 0.3}, 0.3, de)
    w_max, w_min = pso.w_max, pso.w_min
    for _ in range(5):
        pso.ejecutar_iteracion()
        progreso = min(pso.fes_propias / (pso.max_fes * pso.fraccion_presupuesto), 1.0)
        # w_actual se calcula al inicio del paso, con las FES previas a él
        progreso_paso = min((pso.fes_propias - pso.n_individuos)
                            / (pso.max_fes * pso.fraccion_presupuesto), 1.0)
        esperado = w_max - (w_max - w_min) * progreso_paso
        assert abs(pso.w_actual - esperado) < TOL, (pso.w_actual, esperado, progreso)
    print(f"  1f calendario desplazado persiste (w_max={w_max:.4f}, w_min={w_min:.4f}) ... OK")

    # 1g — algoritmo sin espacio común → Canal A inerte
    class SinEspacio(AlgoritmoBioinspirado):
        def inicializar_poblacion(self): pass
        def un_paso(self): pass
        def obtener_hiperparametros_actuales(self): return {"k": 1.0}
    pso, _ = _iniciados()
    otro = SinEspacio(lambda x: 0.0, 2, 10, pso.limites)
    assert aplicar_canal_a(otro, {"w": 0.5, "c1": 2.0, "c2": 2.0}, 0.5, pso) == {}
    assert aplicar_canal_a(pso, {"k": 1.0}, 0.5, otro) == {}
    print("  1g sin espacio común → payload vacío (Canal A inerte, como antes) . OK")


# ── Partes 2 y 3 (orquestador) ──────────────────────────────────────────

def _correr_orquestador(args):
    """Una corrida del propuesto (PSO=A, DE=B) espiando el Canal A.
    `heterogeneo=False` reproduce el comportamiento previo a D-10."""
    funcion, dim, max_fes, heterogeneo = args
    import middleware.orquestador as orq_mod
    from experimentos.semillas import SEMILLAS

    if not heterogeneo:
        PSO.a_espacio_comun = lambda self, hp: {}
        DE.a_espacio_comun = lambda self, hp: {}

    registros, logs = [], []
    original = orq_mod.aplicar_canal_a

    def espia(objetivo, parametros, rmp, fuente):
        z_obj = objetivo.a_espacio_comun(objetivo.obtener_hiperparametros_actuales())
        z_src = fuente.a_espacio_comun(parametros)
        payload = original(objetivo, parametros, rmp, fuente)
        z_new = objetivo.a_espacio_comun(objetivo.obtener_hiperparametros_actuales())
        acerca = all(abs(z_new[d] - z_src[d]) <= abs(z_obj[d] - z_src[d]) + 1e-12
                     for d in z_new if d in z_src)
        registros.append((type(fuente).__name__, type(objetivo).__name__,
                          bool(payload), acerca))
        return payload
    orq_mod.aplicar_canal_a = espia

    class Orquestador(orq_mod.Orquestador):
        def _log(self, mensaje, nivel="INFO"):
            logs.append(mensaje)

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
        max_fes=max_fes, directorio_log=str(dir_logs),
        nombre_log=f"canal_a_het_f{funcion}_{int(heterogeneo)}",
        max_epochs_fastshap=3, silencioso=True,
    )
    resumen = orq.ejecutar()
    return {
        "funcion": funcion, "dim": dim, "heterogeneo": heterogeneo,
        "registros": registros,
        "inerte": sum("Canal A inerte" in m for m in logs),
        "exitoso": sum("Canal A exitoso" in m for m in logs),
        "insuficiente": sum("Canal A insuficiente" in m for m in logs),
        "n_canal_a": resumen.n_transferencias_canal_a,
        "n_canal_b": resumen.n_transferencias_canal_b,
        "error": resumen.error_sistema,
    }


def partes_2_y_3():
    tareas = [(f, d, MAX_FES_INTEGRACION, het)
              for f, d in CONFIGS_INTEGRACION for het in (True, False)]
    with ProcessPoolExecutor(max_workers=min(len(tareas), os.cpu_count() or 1)) as ex:
        res = list(ex.map(_correr_orquestador, tareas))
    con = [r for r in res if r["heterogeneo"]]
    sin = {(r["funcion"], r["dim"]): r for r in res if not r["heterogeneo"]}

    print("\n" + "=" * 70)
    print(f"PARTE 2 — Canal A en el orquestador (D=10, MaxFES={MAX_FES_INTEGRACION})")
    print("=" * 70)
    direcciones = set()
    for r in con:
        dirs = [f"{s}→{t}" for s, t, _, _ in r["registros"]]
        direcciones.update(dirs)
        print(f"  F{r['funcion']:<2d}: Canal A {dirs}")
        assert r["inerte"] == 0, f"F{r['funcion']}: Canal A inerte entre PSO y DE"
        assert all(ok for _, _, ok, _ in r["registros"]), "payload vacío entre PSO y DE"
        assert all(ac for _, _, _, ac in r["registros"]), \
            "el objetivo debe acercarse al fuente en el espacio común"
    assert sum(r["n_canal_a"] for r in con) > 0, "el Canal A debe activarse"
    assert {"PSO→DE", "DE→PSO"} <= direcciones, f"faltan sentidos: {direcciones}"
    print("  Ambos sentidos observados, sin Canal A inerte, siempre acercando ... OK")

    print("\n" + "=" * 70)
    print("PARTE 3 — Efecto: con Canal A heterogéneo vs sin él (pre-D-10)")
    print("=" * 70)
    print(f"  {'F':>3} | {'error con':>11} {'error sin':>11} | {'canal A':>7} "
          f"{'exitoso':>7} {'insuf.':>6} | {'canal B con':>11} {'canal B sin':>11}")
    for r in con:
        s = sin[(r["funcion"], r["dim"])]
        assert s["n_canal_a"] == 0 and s["inerte"] > 0 or s["n_canal_b"] == 0, \
            "sin espacio común el Canal A debe quedar inerte (comportamiento pre-D-10)"
        print(f"  {r['funcion']:>3} | {r['error']:>11.4g} {s['error']:>11.4g} | "
              f"{r['n_canal_a']:>7} {r['exitoso']:>7} {r['insuficiente']:>6} | "
              f"{r['n_canal_b']:>11} {s['n_canal_b']:>11}")
    print("  (una semilla por función: indicativo, no concluyente — la comparación")
    print("   real es la parrilla de 51 semillas)")


def main():
    parte_1()
    partes_2_y_3()
    print("\n[OK] Canal A heterogéneo PSO↔DE (D-10)")


if __name__ == "__main__":
    main()
