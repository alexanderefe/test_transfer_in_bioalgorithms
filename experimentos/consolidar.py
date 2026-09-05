"""
Une los Parquet por corrida en un único archivo consolidado, listo para el
análisis y el material suplementario:

    resultados/corridas/*.parquet            -> resultados/corridas.parquet
    resultados/corridas_propuesto/*.parquet  -> resultados/corridas_propuesto.parquet

Es seguro correrlo cuantas veces se quiera (regenera los consolidados).

Uso:
    python experimentos/consolidar.py [--salida resultados]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experimentos import esquema


def _consolidar(dir_partes: Path, destino: Path, columnas: dict) -> int:
    partes = sorted(dir_partes.glob("*.parquet"))
    if not partes:
        print(f"  {dir_partes}: sin archivos, se omite.")
        return 0
    df = pd.concat((pd.read_parquet(p) for p in partes), ignore_index=True)
    df = df.drop_duplicates(subset="run_id", keep="last")
    df = esquema.df_tipado(df.to_dict("records"), columnas)
    df.to_parquet(destino, index=False)
    print(f"  {destino}: {len(df)} filas")
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--salida", default="resultados")
    args = ap.parse_args()
    salida = Path(args.salida)

    _consolidar(salida / "corridas", salida / "corridas.parquet",
                esquema.COLUMNAS_CORRIDA)
    _consolidar(salida / "corridas_propuesto", salida / "corridas_propuesto.parquet",
                esquema.COLUMNAS_PROPUESTO)


if __name__ == "__main__":
    main()
