"""
Una corrida del harness experimental: ejecuta un algoritmo sobre una función
CEC 2022 con un presupuesto MaxFES y una semilla, y devuelve las filas de los
esquemas §11.2 (todos los algoritmos) y §11.3 (solo el propuesto).

    from experimentos.corrida import correr_corrida
    fila, fila_prop = correr_corrida("middleware", funcion=11, dim=20,
                                     max_fes=120_000, idx_semilla=0)

Algoritmos disponibles (decisión D-4, docs/configuracion_experimental.md):
  - "middleware": PSO fuente + DE objetivo + orquestador (el propuesto).
  - "pso": PSO SOLO, sin middleware, con la misma configuración congelada.
  - "de":  DE SOLO, sin middleware, con la misma configuración congelada.
No hay competidores externos: el experimento se programó desde el inicio
alrededor de PSO y DE, y añadir más algoritmos dispararía aún más el tiempo
de cómputo, ya alto solo con el propuesto.
"""

from __future__ import annotations

import datetime
import socket
import subprocess
import traceback
from pathlib import Path

import numpy as np

from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from bioalgorithms.de import DE
from middleware.orquestador import Orquestador
from middleware.extraccion import EPOCHS_EXPLAINER
from experimentos.semillas import SEMILLAS
from experimentos import esquema

RAIZ = Path(__file__).resolve().parent.parent

# Configuración única del algoritmo propuesto (docs/configuracion_experimental.md).
_POP = 30
_PSO_KW = dict(w_max=0.9, w_min=0.4, c1=2.0, c2=2.0)
_DE_KW = dict(F=0.2, CR=0.3)


def _version() -> str:
    """Marca de versión para reproducibilidad (§10): commit corto de git."""
    try:
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RAIZ, capture_output=True, text=True, timeout=5,
        )
        commit = h.stdout.strip() or "sin-git"
    except Exception:
        commit = "sin-git"
    return f"git:{commit}"


def _correr_middleware(problema: ProblemaCEC2022, dim: int, max_fes: int,
                       semilla: int, max_epochs_fastshap: int,
                       dir_logs: Path, run_id: str):
    """Ejecuta el algoritmo propuesto. Devuelve (resumen, ruta_log_json)."""
    # Dos sub-semillas independientes y reproducibles a partir de la semilla
    # de la corrida (PSO y DE necesitan una cada uno).
    ss_pso, ss_de = np.random.SeedSequence(semilla).spawn(2)

    alg_a = PSO(problema, _POP, dim, problema.limites, semilla=ss_pso,
                max_fes=max_fes, **_PSO_KW)          # fraccion_presupuesto=0.5 (default, D-2)
    alg_b = DE(problema, _POP, dim, problema.limites, semilla=ss_de,
               max_fes=max_fes, **_DE_KW)

    dir_logs.mkdir(parents=True, exist_ok=True)
    orq = Orquestador(
        algoritmo_a=alg_a, algoritmo_b=alg_b, limites=problema.limites,
        problema=problema, max_fes=max_fes,
        directorio_log=str(dir_logs), nombre_log=run_id,
        max_epochs_fastshap=max_epochs_fastshap,
        silencioso=True,
    )
    resumen = orq.ejecutar()
    # El orquestador guarda <nombre_log>_<timestamp>.json; recuperamos el más
    # reciente que empiece por run_id.
    candidatos = sorted(dir_logs.glob(f"{run_id}_*.json"))
    ruta_log_json = str(candidatos[-1]) if candidatos else ""
    return resumen, ruta_log_json


def _correr_algoritmo_solo(clase_alg, kwargs_alg):
    """
    Fábrica de runners para un algoritmo SOLO (sin middleware, sin
    threading): PSO o DE consumiendo el presupuesto MaxFES completo por sí
    mismo. Mismo guard de sobrepaso que HiloAlgoritmo._presupuesto_agotado
    (no arranca una generación que no cabe en lo que queda de presupuesto).

    fraccion_presupuesto=1.0: a diferencia del propuesto (donde PSO comparte
    el presupuesto con DE y decae su inercia contra su CUOTA, 0.5 por
    default), aquí el algoritmo tiene el 100% de MaxFES para sí — su
    esquema temporal debe decaer contra el total completo.
    """
    def _runner(problema, dim, max_fes, semilla, max_epochs_fastshap,
               dir_logs, run_id):
        alg = clase_alg(problema, _POP, dim, problema.limites, semilla=semilla,
                        max_fes=max_fes, fraccion_presupuesto=1.0, **kwargs_alg)
        while problema.fes + alg.n_individuos <= max_fes:
            alg.ejecutar_iteracion()
        return None, ""  # sin ResumenEjecucion ni log JSON (no hay middleware)
    return _runner


_RUNNERS = {
    "middleware": _correr_middleware,
    "pso": _correr_algoritmo_solo(PSO, _PSO_KW),
    "de": _correr_algoritmo_solo(DE, _DE_KW),
}


def correr_corrida(algoritmo: str, funcion: int, dim: int, max_fes: int,
                   idx_semilla: int, *, max_epochs_fastshap: int = EPOCHS_EXPLAINER,
                   dir_salida: str | Path = "resultados"
                   ) -> tuple[dict, dict | None]:
    """
    Ejecuta una corrida y devuelve `(fila_corrida, fila_propuesto | None)`.

    Nunca lanza excepción por un fallo de la corrida: en ese caso devuelve la
    fila con `estado="fallo"` y deja el traceback en
    `<dir_salida>/fallos/<run_id>.txt`.
    """
    if algoritmo not in _RUNNERS:
        raise KeyError(
            f"Algoritmo '{algoritmo}' no reconocido. Disponibles: "
            f"{sorted(_RUNNERS)} (decisión D-4: sin competidores externos)."
        )

    dir_salida = Path(dir_salida)
    rid = esquema.run_id(algoritmo, funcion, dim, max_fes, idx_semilla)
    semilla = SEMILLAS[idx_semilla]
    meta = {
        "algoritmo": algoritmo, "funcion": funcion, "dim": dim,
        "max_fes": max_fes, "idx_semilla": idx_semilla, "semilla": semilla,
        "version": _version(),
    }
    host = socket.gethostname()

    t0 = datetime.datetime.now(datetime.timezone.utc)
    try:
        problema = ProblemaCEC2022(numero_funcion=funcion, ndim=dim)
        problema.configurar_checkpoints(max_fes)
        resumen, ruta_log_json = _RUNNERS[algoritmo](
            problema, dim, max_fes, semilla, max_epochs_fastshap,
            dir_salida / "logs_middleware", rid,
        )
        t1 = datetime.datetime.now(datetime.timezone.utc)
        fila = esquema.fila_corrida(
            meta, estado="ok", problema=problema,
            tiempo_seg=(t1 - t0).total_seconds(), host=host,
            timestamp_utc=t0.isoformat(), ruta_log=ruta_log_json,
        )
        # La fila §11.3 (diagnósticos del middleware) solo existe cuando hubo
        # un ResumenEjecucion real, es decir, para "middleware". PSO/DE solos
        # devuelven resumen=None (ver _correr_algoritmo_solo).
        fila_prop = None
        if resumen is not None:
            fila_prop = esquema.fila_propuesto(
                meta, resumen, ruta_log_json=ruta_log_json)
        return fila, fila_prop

    except Exception as e:  # noqa: BLE001 — se registra y se sigue
        t1 = datetime.datetime.now(datetime.timezone.utc)
        dir_fallos = dir_salida / "fallos"
        dir_fallos.mkdir(parents=True, exist_ok=True)
        (dir_fallos / f"{rid}.txt").write_text(
            f"{t0.isoformat()}  {rid}\n\n{traceback.format_exc()}")
        fila = esquema.fila_corrida(
            meta, estado="fallo",
            tiempo_seg=(t1 - t0).total_seconds(), host=host,
            timestamp_utc=t0.isoformat(),
            mensaje_error=repr(e)[:500],
        )
        return fila, None
