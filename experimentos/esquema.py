"""
Esquema de datos del harness — única fuente de verdad para lo que se captura
por corrida (setup §11). Los módulos de `analisis/` deben leer las columnas
desde aquí, no redefinirlas.

Dos tablas:
  - `corridas`           (§11.2): una fila por corrida, TODOS los algoritmos.
  - `corridas_propuesto` (§11.3): diagnósticos del middleware, SOLO el propuesto.
"""

from __future__ import annotations

import pandas as pd

# Familia de cada algoritmo (para agrupar en figuras). Decisión (D-4,
# docs/configuracion_experimental.md): los únicos "competidores" son PSO y
# DE corriendo SOLOS (sin middleware, con la misma config congelada) — el
# experimento se programó desde el inicio alrededor de estos dos algoritmos.
# No se agregan competidores externos.
FAMILIA_ALGORITMO = {
    "middleware": "propuesto",
    "pso": "baseline",
    "de": "baseline",
}

# ── §11.2 · tabla `corridas` ────────────────────────────────────────────────
# nombre -> dtype de pandas (para fijar el esquema al escribir Parquet).
COLUMNAS_CORRIDA: dict[str, str] = {
    "run_id": "string",
    "algoritmo": "string",
    "familia": "string",
    "version": "string",
    "benchmark": "string",
    "funcion": "int16",
    "dim": "int16",
    "conjunto": "string",
    "max_fes": "int64",
    "idx_semilla": "int16",
    "semilla": "int64",
    "f_mejor": "float64",
    "error": "float64",
    "f_opt": "float64",
    "error_checkpoints": "object",   # list[float] de 14 elementos
    "fes_consumidas": "int64",
    "estado": "string",              # ok | fallo | timeout
    "mensaje_error": "string",
    "tiempo_seg": "float64",
    "host": "string",
    "timestamp_utc": "string",
    "ruta_log": "string",
}

# ── §11.3 · tabla `corridas_propuesto` ─────────────────────────────────────
COLUMNAS_PROPUESTO: dict[str, str] = {
    "run_id": "string",
    "funcion": "int16",
    "dim": "int16",
    "max_fes": "int64",
    "idx_semilla": "int16",
    "f_mejor_a": "float64",
    "f_mejor_b": "float64",
    "error_a": "float64",
    "error_b": "float64",
    "fes_a": "int64",
    "fes_b": "int64",
    "n_generaciones_a": "int64",
    "n_generaciones_b": "int64",
    "n_activaciones_fase2": "int32",
    "n_activaciones_fase3": "int32",
    "n_reentrenamientos_cambio_fuente": "int32",
    "n_canal_a": "int32",
    "n_canal_b": "int32",
    "n_abortos_wasserstein": "int32",
    "n_instancias_inyectadas": "int32",
    "fes_primera_activacion": "float64",   # NaN si el middleware nunca actúa
    "frac_primera_activacion": "float64",
    "r2_fase2": "object",                  # list[float], un R² por activación
    "tiempo_pausado_middleware_seg": "float64",
    "ruta_log_json": "string",
}


def df_tipado(filas: list[dict], columnas: dict[str, str]) -> "pd.DataFrame":
    """DataFrame con las columnas de `columnas` en su orden y dtype. Las
    columnas `object` (listas) se dejan como están."""
    df = pd.DataFrame(filas, columns=list(columnas))
    for col, dt in columnas.items():
        if dt == "object":
            continue
        df[col] = df[col].astype(dt)
    return df


def run_id(algoritmo: str, funcion: int, dim: int, max_fes: int,
           idx_semilla: int) -> str:
    """Identificador legible y estable de una corrida."""
    return (f"{algoritmo}__f{funcion:02d}__d{dim:02d}"
            f"__fes{int(max_fes)}__s{idx_semilla:02d}")


def conjunto(dim: int) -> str:
    return f"CEC2022_{dim}"


def fila_corrida(meta: dict, *, estado: str, problema=None,
                 tiempo_seg: float = float("nan"), host: str = "",
                 timestamp_utc: str = "", ruta_log: str = "",
                 mensaje_error: str = "") -> dict:
    """
    Construye la fila de §11.2. `meta` trae la identidad de la corrida
    (algoritmo, funcion, dim, max_fes, idx_semilla, semilla, version).
    En fallo se pasa `estado="fallo"` y `problema` puede ser None.

    Todo sale de `problema` (agnóstico al algoritmo: sirve igual para el
    middleware, PSO solo o DE solo): `problema.optimo_global`, `problema.fes`
    y `problema.error_checkpoints()`. No depende de `ResumenEjecucion`.
    """
    fila = {c: None for c in COLUMNAS_CORRIDA}
    fila.update({
        "run_id": run_id(meta["algoritmo"], meta["funcion"], meta["dim"],
                         meta["max_fes"], meta["idx_semilla"]),
        "algoritmo": meta["algoritmo"],
        "familia": FAMILIA_ALGORITMO.get(meta["algoritmo"], "desconocida"),
        "version": meta.get("version", ""),
        "benchmark": "CEC2022",
        "funcion": int(meta["funcion"]),
        "dim": int(meta["dim"]),
        "conjunto": conjunto(meta["dim"]),
        "max_fes": int(meta["max_fes"]),
        "idx_semilla": int(meta["idx_semilla"]),
        "semilla": int(meta["semilla"]),
        "estado": estado,
        "mensaje_error": mensaje_error,
        "tiempo_seg": float(tiempo_seg),
        "host": host,
        "timestamp_utc": timestamp_utc,
        "ruta_log": ruta_log,
        "error_checkpoints": None,
        "f_mejor": float("nan"),
        "error": float("nan"),
        "f_opt": float("nan"),
        "fes_consumidas": 0,
    })
    if problema is not None:
        fila["f_opt"] = float(problema.optimo_global)
        fila["fes_consumidas"] = int(problema.fes)
        cps = list(problema.error_checkpoints())
        fila["error_checkpoints"] = cps
        # Métrica de comparación: mejor error usando <= MaxFES evaluaciones.
        # Se toma del checkpoint final (fracción 1.0), NO del último valor
        # visto por el algoritmo, para no contar las <= pop evaluaciones de
        # sobrepaso de la última generación.
        fila["error"] = float(cps[-1])
        fila["f_mejor"] = float(fila["f_opt"] + cps[-1])
    return fila


def fila_propuesto(meta: dict, resumen, *, ruta_log_json: str = "") -> dict:
    """Construye la fila de §11.3 a partir de `ResumenEjecucion`."""
    max_fes = int(meta["max_fes"])
    fpa = float(resumen.fes_primera_activacion)
    r2 = [float(e["r2_modelo_subrogado"]) for e in resumen.log_fase2]
    return {
        "run_id": run_id(meta["algoritmo"], meta["funcion"], meta["dim"],
                         max_fes, meta["idx_semilla"]),
        "funcion": int(meta["funcion"]),
        "dim": int(meta["dim"]),
        "max_fes": max_fes,
        "idx_semilla": int(meta["idx_semilla"]),
        "f_mejor_a": float(resumen.fitness_final_algoritmo_a),
        "f_mejor_b": float(resumen.fitness_final_algoritmo_b),
        "error_a": float(resumen.error_algoritmo_a),
        "error_b": float(resumen.error_algoritmo_b),
        "fes_a": int(resumen.fes_consumidas_a),
        "fes_b": int(resumen.fes_consumidas_b),
        "n_generaciones_a": int(resumen.n_generaciones_a),
        "n_generaciones_b": int(resumen.n_generaciones_b),
        "n_activaciones_fase2": int(resumen.n_activaciones_fase2),
        "n_activaciones_fase3": int(resumen.n_activaciones_fase3),
        "n_reentrenamientos_cambio_fuente": int(
            resumen.n_reentrenamientos_por_cambio_fuente),
        "n_canal_a": int(resumen.n_transferencias_canal_a),
        "n_canal_b": int(resumen.n_transferencias_canal_b),
        "n_abortos_wasserstein": int(resumen.n_transferencias_abortadas_wasserstein),
        "n_instancias_inyectadas": int(resumen.n_instancias_inyectadas_total),
        "fes_primera_activacion": fpa,
        "frac_primera_activacion": (fpa / max_fes) if fpa == fpa else float("nan"),
        "r2_fase2": r2,
        "tiempo_pausado_middleware_seg": float(
            resumen.tiempo_pausado_en_middleware_segundos),
        "ruta_log_json": ruta_log_json,
    }
