"""
Carga y utilidades base sobre los Parquet consolidados del harness
experimental (`resultados/corridas.parquet`, `resultados/corridas_propuesto.parquet`).

Punto de entrada único que reutiliza el resto de `analisis/` — evita que
cada módulo reimplemente el `groupby` del error medio. Todo el análisis
(§6-§9 de `docs/setup_experimental.md`) se calcula sin re-ejecutar nada,
a partir de estos dos archivos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Identidad del experimento (setup §1-§5) — constantes compartidas por todo
# `analisis/`, para no repetirlas en cada módulo.
ALGORITMOS = ("middleware", "pso", "de")
NOMBRE_ALGORITMO = {
    "middleware": "Propuesto",
    "pso": "PSO",
    "de": "DE",
}
DIMS = (10, 20)
MAXFES = (5_000, 50_000, 500_000, 5_000_000)
N_FUNCIONES = 12
N_SEMILLAS = 51


def cargar_corridas(ruta: str | Path = "resultados/corridas.parquet") -> pd.DataFrame:
    """
    Carga `corridas.parquet` y valida su forma antes de analizar nada:
    todas las corridas deben estar en `estado="ok"` y el total de filas
    debe coincidir exactamente con 3 algoritmos × 2 dims × 4 MaxFES ×
    12 funciones × 51 semillas = 14 688.
    """
    df = pd.read_parquet(ruta)
    n_fallo = int((df["estado"] != "ok").sum())
    if n_fallo:
        raise ValueError(
            f"{n_fallo} corridas no están en estado 'ok' en {ruta} — "
            f"resolver/reintentar antes de analizar (ver resultados/fallos/)."
        )
    esperado = len(ALGORITMOS) * len(DIMS) * len(MAXFES) * N_FUNCIONES * N_SEMILLAS
    if len(df) != esperado:
        raise ValueError(
            f"Se esperaban {esperado} filas en {ruta}, hay {len(df)}. "
            f"¿Falta consolidar (experimentos/consolidar.py) o falta correr algo?"
        )
    return df


def cargar_propuesto(
    ruta: str | Path = "resultados/corridas_propuesto.parquet",
) -> pd.DataFrame:
    """Carga los diagnósticos del middleware (§11.3) — solo existen para
    `algoritmo="middleware"`, usados en la discusión §9."""
    return pd.read_parquet(ruta)


def tabla_error_medio(df: pd.DataFrame, dim: int, max_fes: int) -> pd.DataFrame:
    """
    Tabla base de una configuración (dim × MaxFES): índice = algoritmo,
    columnas = función (1..12), valores = error medio de las 51 corridas.
    Esta es la tabla de la que salen los rankings (§6) y la matriz de
    Friedman (§7).
    """
    sub = df[(df["dim"] == dim) & (df["max_fes"] == max_fes)]
    tabla = sub.groupby(["algoritmo", "funcion"])["error"].mean().unstack("funcion")
    return tabla.reindex(index=list(ALGORITMOS), columns=range(1, N_FUNCIONES + 1))


def configuraciones():
    """Itera las 8 configuraciones (dim × MaxFES) en el orden del setup —
    nunca se promedian ni mezclan entre sí (§6-§7)."""
    for dim in DIMS:
        for max_fes in MAXFES:
            yield dim, max_fes


def id_config(dim: int, max_fes: int) -> str:
    """Identificador corto y estable de una configuración, para nombres de
    archivo (`rankings_d10_fes5000000`, etc.)."""
    return f"d{dim}_fes{max_fes}"
