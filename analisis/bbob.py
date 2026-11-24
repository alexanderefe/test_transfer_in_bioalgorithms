"""
Análisis de la parrilla BBOB (opción A de README_BBOB.md) con la misma
metodología de rankings que CEC 2022 (setup §6–§8), por dimensión:

  - error medio de las 15 instancias por (algoritmo × función), con errores
    < 1e-8 colapsados a 0 (convención CEC);
  - rango por función (empate si |Δ| < 1e-8) y rango promedio sobre las 24
    funciones (y por cada uno de los 5 grupos de funciones de COCO);
  - Friedman (k algoritmos, N = 24 funciones) y Diferencia Crítica de
    Nemenyi (k = 12, N = 24 → CD ≈ 3.40), con su diagrama;
  - tasa de éxito: fracción de instancias que llegan a error < 1e-8.

El análisis estándar de COCO (ECDF de runtime a objetivos, ERT, comparación
con el archivo de datos de COCO) lo hace `cocopp` con los archivos del
observador — ver README_BBOB.md.

    python analisis/bbob.py --resultados resultados_bbob
    python analisis/bbob.py --resultados resultados_bbob --permitir-incompleto

Salida en `<resultados>/analisis_bbob/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from analisis.cd_nemenyi import diferencia_critica, dibujar_cd
from analisis.rankings import rankear_columna
from problems.bbob_wrapper import (
    DIMENSIONES_BBOB, FUNCIONES_BBOB, GRUPO_BBOB, INSTANCIAS_BBOB,
)
from problems.cec2022_wrapper import UMBRAL_ERROR_CERO

ALPHA = 0.05


def cargar(ruta: Path, permitir_incompleto: bool) -> pd.DataFrame:
    df = pd.read_parquet(ruta)
    df = df[df["benchmark"] == "BBOB"].copy()
    if df.empty:
        raise ValueError(f"{ruta}: no hay corridas BBOB.")
    fallos = df[df["estado"] != "ok"]
    if len(fallos):
        msg = f"{len(fallos)} corridas BBOB en estado distinto de 'ok'"
        if not permitir_incompleto:
            raise ValueError(msg + " (reintentarlas con grid_bbob.py)")
        print("Aviso:", msg, "— se excluyen.")
        df = df[df["estado"] == "ok"]
    presupuestos = sorted((df["max_fes"] // df["dim"]).unique())
    if len(presupuestos) > 1:
        raise ValueError(f"Mezcla de presupuestos (·D): {presupuestos}. "
                         f"Usar una carpeta de resultados por presupuesto.")
    df["instancia"] = df["idx_semilla"] + 1
    df["error_u"] = np.where(df["error"] < UMBRAL_ERROR_CERO, 0.0, df["error"])
    algoritmos = sorted(df["algoritmo"].unique())
    esperado = len(algoritmos) * len(FUNCIONES_BBOB) * len(DIMENSIONES_BBOB) * len(INSTANCIAS_BBOB)
    if len(df) != esperado:
        msg = (f"{len(df)} corridas, se esperaban {esperado} "
               f"({len(algoritmos)} algoritmos × 24 × 6 × 15)")
        if not permitir_incompleto:
            raise ValueError(msg)
        print("Aviso:", msg)
    return df


def ranking_dim(df: pd.DataFrame, dim: int) -> tuple[pd.DataFrame, pd.Series]:
    """(tabla error medio algoritmo × función, rangos algoritmo × función)."""
    sub = df[df["dim"] == dim]
    err = sub.groupby(["algoritmo", "funcion"])["error_u"].mean().unstack("funcion")
    err = err.dropna(axis=1)                 # solo funciones con todos los algoritmos
    rangos = err.apply(rankear_columna, axis=0)
    return err, rangos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", default="resultados_bbob")
    ap.add_argument("--permitir-incompleto", action="store_true")
    args = ap.parse_args()

    base = Path(args.resultados)
    df = cargar(base / "corridas.parquet", args.permitir_incompleto)
    salida = base / "analisis_bbob"
    (salida / "figuras").mkdir(parents=True, exist_ok=True)
    presupuesto = int((df["max_fes"] // df["dim"]).iloc[0])
    print(f"BBOB, presupuesto {presupuesto}·D: {len(df)} corridas, "
          f"{df['algoritmo'].nunique()} algoritmos")

    resumen = []
    for dim in sorted(df["dim"].unique()):
        err, rangos = ranking_dim(df, dim)
        k, n = rangos.shape
        if k < 2 or n < 2:
            continue
        err.to_csv(salida / f"error_medio_d{dim:02d}.csv")
        rangos.to_csv(salida / f"rangos_d{dim:02d}.csv")
        prom = rangos.mean(axis=1).sort_values()
        por_grupo = pd.DataFrame({
            g: rangos[[f for f in rangos.columns if GRUPO_BBOB[f] == g]].mean(axis=1)
            for g in dict.fromkeys(GRUPO_BBOB.values())
            if any(GRUPO_BBOB[f] == g for f in rangos.columns)
        })
        por_grupo.to_csv(salida / f"rangos_por_grupo_d{dim:02d}.csv")

        stat, p = friedmanchisquare(*[err.loc[a].values for a in err.index]) if k >= 3 else (np.nan, np.nan)
        cd = diferencia_critica(k, n)
        dibujar_cd(prom.to_dict(), cd, f"BBOB D={dim} — {presupuesto}·D",
                   salida / "figuras" / f"cd_bbob_d{dim:02d}.png")

        exito = (df[df["dim"] == dim].assign(ok=lambda d: d["error_u"] == 0)
                 .groupby("algoritmo")["ok"].mean())
        for a in prom.index:
            resumen.append({"dim": dim, "algoritmo": a, "rango_promedio": prom[a],
                            "tasa_exito": float(exito.get(a, np.nan)),
                            "friedman_p": p, "cd": cd, "k": k, "n_funciones": n})
        print(f"\nD={dim}: Friedman p={p:.3g} ({'rechaza' if p < ALPHA else 'no rechaza'} H0), "
              f"CD={cd:.2f} (k={k}, N={n})")
        for a in prom.index:
            print(f"   {a:11s} rango {prom[a]:5.2f}   éxito {exito.get(a, np.nan):5.1%}")

    pd.DataFrame(resumen).to_csv(salida / "resumen.csv", index=False)
    print(f"\n[OK] tablas y diagramas CD en {salida}")


if __name__ == "__main__":
    main()
