"""
Figuras y tablas restantes de §8, más los insumos numéricos de la
discusión obligatoria §9 (usa `corridas_propuesto.parquet`, diagnósticos
del middleware — D-1a, D-1b, D-1d en `docs/configuracion_experimental.md`).
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analisis.datos import (
    ALGORITMOS, DIMS, MAXFES, NOMBRE_ALGORITMO, cargar_corridas, cargar_propuesto,
)
from analisis.rankings import ranking_configuracion


def barras_rangos_por_maxfes(df: pd.DataFrame, dim: int, ruta_salida: Path) -> None:
    """Barras de rango promedio × MaxFES, una serie por algoritmo, para
    una dimensionalidad (2 figuras en total, setup §8)."""
    datos = {a: [] for a in ALGORITMOS}
    for max_fes in MAXFES:
        ranking = ranking_configuracion(df, dim, max_fes)
        for a in ALGORITMOS:
            datos[a].append(ranking.loc[a, ("resumen", "rango_promedio")])

    x = range(len(MAXFES))
    ancho = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, a in enumerate(ALGORITMOS):
        pos = [xi + (i - 1) * ancho for xi in x]
        ax.bar(pos, datos[a], width=ancho, label=NOMBRE_ALGORITMO.get(a, a))
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{m:,}".replace(",", ".") for m in MAXFES])
    ax.set_xlabel("MaxFES")
    ax.set_ylabel("Rango promedio (1=mejor)")
    ax.set_title(f"Rango promedio por MaxFES — CEC2022_{dim}")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def tabla_suplementaria(df: pd.DataFrame) -> pd.DataFrame:
    """Material suplementario obligatorio (§8): media, mediana y desv.
    estándar de las 51 corridas, por (función × dim × MaxFES × algoritmo)."""
    return (df.groupby(["algoritmo", "dim", "max_fes", "funcion"])["error"]
            .agg(["mean", "median", "std"]).reset_index())


def diagnostico_d1d(df_prop: pd.DataFrame) -> pd.DataFrame:
    """
    D-1d (setup §9): con MaxFES=5000 el middleware no alcanza a
    activarse. Cuantifica el % de corridas con `fes_primera_activacion`
    NaN (el middleware nunca activó Fase 2), por MaxFES.
    """
    tabla = df_prop.assign(nunca_activo=df_prop["fes_primera_activacion"].isna())
    return (tabla.groupby("max_fes")["nunca_activo"]
            .mean().mul(100).rename("pct_nunca_activo").reset_index())


def diagnostico_canales(df_prop: pd.DataFrame) -> pd.DataFrame:
    """Uso de Canal A vs Canal B por MaxFES (§9: qué operador es
    beneficioso/contraproducente según el presupuesto)."""
    return df_prop.groupby("max_fes")[["n_canal_a", "n_canal_b"]].mean().reset_index()


def diagnostico_r2(df_prop: pd.DataFrame) -> pd.DataFrame:
    """Distribución de R² del subrogado por función (insumo D-1a)."""
    explotado = df_prop[["funcion", "r2_fase2"]].explode("r2_fase2").dropna()
    explotado["r2_fase2"] = explotado["r2_fase2"].astype(float)
    return explotado.groupby("funcion")["r2_fase2"].agg(["mean", "median", "std", "count"]).reset_index()


def generar_todos(
    ruta_corridas: str | Path = "resultados/corridas.parquet",
    ruta_propuesto: str | Path = "resultados/corridas_propuesto.parquet",
    dir_tablas: str | Path = "resultados/analisis/tablas",
    dir_figuras: str | Path = "resultados/analisis/figuras",
) -> dict[str, pd.DataFrame]:
    df = cargar_corridas(ruta_corridas)
    df_prop = cargar_propuesto(ruta_propuesto)
    dir_tablas = Path(dir_tablas)
    dir_figuras = Path(dir_figuras)
    dir_tablas.mkdir(parents=True, exist_ok=True)
    dir_figuras.mkdir(parents=True, exist_ok=True)

    for dim in DIMS:
        barras_rangos_por_maxfes(df, dim, dir_figuras / f"barras_rangos_d{dim}.png")

    suplementaria = tabla_suplementaria(df)
    suplementaria.to_csv(dir_tablas / "suplementario_media_mediana_std.csv", index=False)

    d1d = diagnostico_d1d(df_prop)
    d1d.to_csv(dir_tablas / "diagnostico_d1d_activacion.csv", index=False)

    canales = diagnostico_canales(df_prop)
    canales.to_csv(dir_tablas / "diagnostico_canales_a_b.csv", index=False)

    r2 = diagnostico_r2(df_prop)
    r2.to_csv(dir_tablas / "diagnostico_r2_fase2.csv", index=False)

    return {
        "suplementaria": suplementaria, "d1d": d1d,
        "canales": canales, "r2": r2,
    }


def main():
    salidas = generar_todos()
    print("D-1d — % de corridas donde el middleware NUNCA activó Fase 2, por MaxFES:")
    print(salidas["d1d"].to_string(index=False))
    print()
    print("Uso de Canal A vs Canal B (promedio de activaciones), por MaxFES:")
    print(salidas["canales"].to_string(index=False))
    print()
    print("R² del subrogado por función (Fase 2):")
    print(salidas["r2"].round(3).to_string(index=False))
    print(f"\nSuplementario: {len(salidas['suplementaria'])} filas "
          f"(algoritmo × dim × MaxFES × función)")

    assert salidas["d1d"].loc[salidas["d1d"]["max_fes"] == 5000, "pct_nunca_activo"].iloc[0] > 0, \
        "D-1d esperaba que a MaxFES=5000 el middleware no siempre active Fase 2"
    print("\n[OK] figuras.py")


if __name__ == "__main__":
    main()
