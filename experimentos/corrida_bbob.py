"""
Una corrida BBOB (opción A de README_BBOB.md): un algoritmo sobre una
instancia (función, dimensión, instancia) de la suite `bbob` de COCO, con
presupuesto `max_fes` evaluaciones.

    from experimentos.corrida_bbob import correr_corrida_bbob
    fila, fila_prop = correr_corrida_bbob("cmaes", funcion=15, dim=10,
                                          instancia=1, max_fes=100_000)

Reutiliza TAL CUAL los runners de `experimentos/corrida.py` (los mismos 12
algoritmos, la misma configuración congelada): el único cambio es el
problema, `ProblemaBBOB` en vez de `ProblemaCEC2022`, con la misma
interfaz.

Semilla: en COCO las instancias hacen de repeticiones independientes, así
que hay una corrida por instancia. La semilla del algoritmo es
`SEMILLAS[instancia - 1]` (la misma lista fija de 51 semillas del
protocolo CEC); en las filas, `idx_semilla = instancia - 1`.

Las filas siguen el esquema §11.2/§11.3 de `experimentos/esquema.py`, con
`benchmark="BBOB"`, `conjunto="BBOB_<D>"` y run_id `bbob__...`.
"""

from __future__ import annotations

import datetime
import shutil
import socket
import traceback
from pathlib import Path

from problems.bbob_wrapper import ProblemaBBOB, VERSION_EVALUADOR_BBOB
from middleware.extraccion import EPOCHS_EXPLAINER
from experimentos.semillas import SEMILLAS
from experimentos.corrida import _RUNNERS, _version
from experimentos.competidores import COMPETIDORES
from experimentos import esquema

# Nombre que registra el observador de COCO (leyendas de cocopp).
NOMBRE_COCO = {
    "middleware": "Middleware", "pso": "PSO", "de": "DE",
    **{k: v[1] for k, v in COMPETIDORES.items()},
}


def correr_corrida_bbob(algoritmo: str, funcion: int, dim: int, instancia: int,
                        max_fes: int, *,
                        max_epochs_fastshap: int = EPOCHS_EXPLAINER,
                        dir_salida: str | Path = "resultados_bbob",
                        observador: bool = True,
                        ) -> tuple[dict, dict | None]:
    """
    Ejecuta una corrida BBOB y devuelve `(fila_corrida, fila_propuesto | None)`.
    Igual que `correr_corrida`: nunca lanza excepción por un fallo de la
    corrida (queda `estado="fallo"` y el traceback en `<dir_salida>/fallos/`).

    observador: si True, los archivos de COCO quedan en
        `<dir_salida>/coco/<algoritmo>/fFF_dDD_iII_fesN/` (entrada de `cocopp`).
    """
    if algoritmo not in _RUNNERS:
        raise KeyError(f"Algoritmo '{algoritmo}' no reconocido: {sorted(_RUNNERS)}.")

    dir_salida = Path(dir_salida)
    rid = esquema.run_id_bbob(algoritmo, funcion, dim, instancia, max_fes)
    idx_semilla = instancia - 1
    semilla = SEMILLAS[idx_semilla]
    meta = {
        "run_id": rid, "benchmark": "BBOB", "conjunto": f"BBOB_{dim}",
        "algoritmo": algoritmo, "funcion": funcion, "dim": dim,
        "max_fes": max_fes, "idx_semilla": idx_semilla, "semilla": semilla,
        "version": _version(backend=VERSION_EVALUADOR_BBOB, algoritmo=algoritmo),
    }
    host = socket.gethostname()
    # Nombre corto (MAX_PATH de Windows, ver ProblemaBBOB): el algoritmo ya
    # va en la carpeta padre.
    carpeta_obs = (dir_salida / "coco" / algoritmo
                   / f"f{funcion:02d}_d{dim:02d}_i{instancia:02d}_fes{int(max_fes)}"
                   ) if observador else None

    t0 = datetime.datetime.now(datetime.timezone.utc)
    problema = None
    try:
        problema = ProblemaBBOB(funcion, dim, instancia,
                                carpeta_observador=carpeta_obs,
                                nombre_algoritmo=NOMBRE_COCO.get(algoritmo, algoritmo))
        problema.configurar_checkpoints(max_fes)
        resumen, ruta_log_json = _RUNNERS[algoritmo](
            problema, dim, max_fes, semilla, max_epochs_fastshap,
            dir_salida / "logs_middleware", rid,
        )
        problema.cerrar()
        t1 = datetime.datetime.now(datetime.timezone.utc)
        fila = esquema.fila_corrida(
            meta, estado="ok", problema=problema,
            tiempo_seg=(t1 - t0).total_seconds(), host=host,
            timestamp_utc=t0.isoformat(),
            ruta_log=ruta_log_json or (str(carpeta_obs) if carpeta_obs else ""),
        )
        fila_prop = None
        if resumen is not None:
            fila_prop = esquema.fila_propuesto(meta, resumen, ruta_log_json=ruta_log_json)
        return fila, fila_prop

    except Exception as e:  # noqa: BLE001 — se registra y se sigue
        if problema is not None:
            try:
                problema.cerrar()
            except Exception:  # noqa: BLE001
                pass
        # Datos parciales de COCO de una corrida fallida: fuera, para que
        # cocopp no los mezcle con las corridas completas.
        if carpeta_obs is not None and carpeta_obs.is_dir():
            shutil.rmtree(carpeta_obs, ignore_errors=True)
        t1 =datetime.datetime.now(datetime.timezone.utc)
        dir_fallos = dir_salida / "fallos"
        dir_fallos.mkdir(parents=True, exist_ok=True)
        (dir_fallos / f"{rid}.txt").write_text(
            f"{t0.isoformat()}  {rid}\n\n{traceback.format_exc()}")
        fila = esquema.fila_corrida(
            meta, estado="fallo",
            tiempo_seg=(t1 - t0).total_seconds(), host=host,
            timestamp_utc=t0.isoformat(), mensaje_error=repr(e)[:500],
        )
        return fila, None
