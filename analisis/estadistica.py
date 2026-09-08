"""
Análisis estadístico (setup §7, α = 0.05). Todo se calcula por separado
para cada una de las 8 configuraciones (dim × MaxFES) — nunca se agrupan
entre configuraciones.

- Friedman: H0 = "los 3 algoritmos tienen el mismo rango promedio" sobre
  la matriz de rangos (12 funciones × 3 algoritmos). `scipy.stats.friedmanchisquare`.
- Post-hoc de Shaffer: NO está en ninguna librería instalada (scikit-posthocs
  tiene Nemenyi/Conover/Miller/Siegel/Durbin/Quade, no Shaffer). Se
  implementa aquí — con k=3 algoritmos hay solo 3 comparaciones por pares,
  y por transitividad el número de hipótesis nulas verdaderas simultáneas
  solo puede ser 0, 1 o 3 (nunca 2: si A=B y A=C, entonces B=C se sigue),
  lo que da la secuencia estática de Shaffer S=(3,1,1) para los 3 p-valores
  ordenados ascendente — coincide con la tabla publicada (Shaffer 1986;
  García & Herrera 2008) para k=3.
- Wilcoxon de rangos con signo: propuesto vs cada competidor, por función,
  emparejado por semilla (`idx_semilla`) — veredicto better/equal/worse.
- Desviación estándar de los rangos promedio entre algoritmos.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, wilcoxon

from analisis.datos import ALGORITMOS, N_FUNCIONES, cargar_corridas, configuraciones, id_config
from analisis.rankings import ranking_configuracion, tabla_error_medio

ALPHA = 0.05

# Divisores estáticos de Shaffer para k=3 algoritmos (3 comparaciones por
# pares) — ver docstring del módulo. Si algún día k cambiara, esta
# secuencia habría que rederivarla (no es válida para otro k).
_DIVISORES_SHAFFER_K3 = (3, 1, 1)


def friedman(df: pd.DataFrame, dim: int, max_fes: int) -> dict:
    """Test de Friedman sobre las 12 funciones (bloques) para los 3
    algoritmos de esta configuración."""
    errores = tabla_error_medio(df, dim, max_fes)
    stat, p = friedmanchisquare(*[errores.loc[a].values for a in ALGORITMOS])
    return {"estadistico": float(stat), "p_valor": float(p), "rechaza_h0": p < ALPHA}


def shaffer(df: pd.DataFrame, dim: int, max_fes: int) -> pd.DataFrame:
    """
    Post-hoc de Shaffer sobre los rangos promedio de esta configuración:
    z = |R_i - R_j| / SE, SE = sqrt(k(k+1)/(6N)), p_crudo = 2(1-Φ(|z|)),
    ajuste step-down con los divisores estáticos (3,1,1).
    """
    ranking = ranking_configuracion(df, dim, max_fes)
    rangos = {a: ranking.loc[a, ("resumen", "rango_promedio")] for a in ALGORITMOS}
    k, n = len(ALGORITMOS), N_FUNCIONES
    se = np.sqrt(k * (k + 1) / (6 * n))

    pares = [(ALGORITMOS[i], ALGORITMOS[j])
             for i in range(k) for j in range(i + 1, k)]
    crudos = []
    for a, b in pares:
        z = abs(rangos[a] - rangos[b]) / se
        p_crudo = 2 * (1 - norm.cdf(z))
        crudos.append((a, b, z, p_crudo))
    crudos.sort(key=lambda t: t[3])

    filas = []
    p_prev = 0.0
    for (a, b, z, p_crudo), s in zip(crudos, _DIVISORES_SHAFFER_K3):
        p_ajustado = min(1.0, s * p_crudo)
        p_ajustado = max(p_ajustado, p_prev)
        p_prev = p_ajustado
        filas.append({
            "algoritmo_a": a, "algoritmo_b": b,
            "rango_a": rangos[a], "rango_b": rangos[b],
            "z": z, "p_crudo": p_crudo,
            "divisor_shaffer": s, "p_ajustado": p_ajustado,
            "significativo": p_ajustado < ALPHA,
        })
    return pd.DataFrame(filas)


def pct_diferencias_significativas(tabla_shaffer: pd.DataFrame, propuesto: str = "middleware") -> float:
    """% de competidores de los que el propuesto difiere significativamente
    (setup §7) — sobre las comparaciones donde participa `propuesto`."""
    del_propuesto = tabla_shaffer[
        (tabla_shaffer["algoritmo_a"] == propuesto) | (tabla_shaffer["algoritmo_b"] == propuesto)
    ]
    if len(del_propuesto) == 0:
        return float("nan")
    return 100.0 * del_propuesto["significativo"].mean()


def wilcoxon_bew(df: pd.DataFrame, dim: int, max_fes: int, propuesto: str = "middleware") -> pd.DataFrame:
    """
    Wilcoxon de rangos con signo, propuesto vs cada competidor, POR
    FUNCIÓN (12), emparejado por `idx_semilla` (51 pares). Veredicto:
    p >= 0.05 -> equal; si no, better/worse según qué media sea menor.
    """
    sub = df[(df["dim"] == dim) & (df["max_fes"] == max_fes)]
    competidores = [a for a in ALGORITMOS if a != propuesto]
    filas = []
    for competidor in competidores:
        for funcion in range(1, N_FUNCIONES + 1):
            pareado = sub[sub["funcion"] == funcion].pivot(
                index="idx_semilla", columns="algoritmo", values="error")
            a, b = pareado[propuesto].values, pareado[competidor].values
            if np.allclose(a, b):
                # wilcoxon no admite diferencias todas cero (división por N=0)
                p_valor, veredicto = 1.0, "equal"
            else:
                _, p_valor = wilcoxon(a, b)
                if p_valor >= ALPHA:
                    veredicto = "equal"
                else:
                    veredicto = "better" if a.mean() < b.mean() else "worse"
            filas.append({
                "competidor": competidor, "funcion": funcion,
                "p_valor": p_valor, "veredicto": veredicto,
            })
    return pd.DataFrame(filas)


def resumen_wilcoxon_bew(tabla_bew: pd.DataFrame) -> pd.DataFrame:
    """Conteo better/equal/worse sobre las 12 funciones, por competidor
    (una fila por competidor)."""
    return (tabla_bew.groupby("competidor")["veredicto"]
            .value_counts().unstack(fill_value=0)
            .reindex(columns=["better", "equal", "worse"], fill_value=0))


def desviacion_rangos(df: pd.DataFrame, dim: int, max_fes: int) -> float:
    """Desviación estándar de los rangos promedio entre los 3 algoritmos
    (medida de diversidad de desempeño, setup §7)."""
    ranking = ranking_configuracion(df, dim, max_fes)
    return float(ranking[("resumen", "rango_promedio")].std())


def generar_todos(
    ruta_corridas: str | Path = "resultados/corridas.parquet",
    dir_salida: str | Path = "resultados/analisis/tablas",
) -> dict[str, pd.DataFrame]:
    """Genera y guarda Friedman, Shaffer, % significativo, Wilcoxon b/e/w
    y desviación de rangos para las 8 configuraciones."""
    df = cargar_corridas(ruta_corridas)
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    friedman_filas, pct_filas, desv_filas = [], [], []
    bew_todas = []
    for dim, max_fes in configuraciones():
        cfg = id_config(dim, max_fes)

        f = friedman(df, dim, max_fes)
        friedman_filas.append({"dim": dim, "max_fes": max_fes, **f})

        tabla_shaf = shaffer(df, dim, max_fes)
        tabla_shaf.to_csv(dir_salida / f"shaffer_{cfg}.csv", index=False)
        pct_filas.append({
            "dim": dim, "max_fes": max_fes,
            "pct_significativo_vs_propuesto": pct_diferencias_significativas(tabla_shaf),
        })

        bew = wilcoxon_bew(df, dim, max_fes)
        resumen_bew = resumen_wilcoxon_bew(bew).reset_index()
        resumen_bew.insert(0, "max_fes", max_fes)
        resumen_bew.insert(0, "dim", dim)
        bew_todas.append(resumen_bew)

        desv_filas.append({"dim": dim, "max_fes": max_fes,
                           "desv_rangos": desviacion_rangos(df, dim, max_fes)})

    tabla_friedman = pd.DataFrame(friedman_filas)
    tabla_pct = pd.DataFrame(pct_filas)
    tabla_bew = pd.concat(bew_todas, ignore_index=True)
    tabla_desv = pd.DataFrame(desv_filas)

    tabla_friedman.to_csv(dir_salida / "friedman.csv", index=False)
    tabla_pct.to_csv(dir_salida / "shaffer_pct_significativo.csv", index=False)
    tabla_bew.to_csv(dir_salida / "wilcoxon_bew.csv", index=False)
    tabla_desv.to_csv(dir_salida / "desviacion_rangos.csv", index=False)

    return {
        "friedman": tabla_friedman, "shaffer_pct": tabla_pct,
        "wilcoxon_bew": tabla_bew, "desviacion_rangos": tabla_desv,
    }


def main():
    salidas = generar_todos()
    print("Friedman (8 configuraciones):")
    print(salidas["friedman"].to_string(index=False))
    print()
    print("% significativo vs propuesto (Shaffer):")
    print(salidas["shaffer_pct"].to_string(index=False))
    print()
    print("Wilcoxon better/equal/worse:")
    print(salidas["wilcoxon_bew"].to_string(index=False))
    print()
    print("Desviación de rangos:")
    print(salidas["desviacion_rangos"].to_string(index=False))

    assert (salidas["friedman"]["p_valor"].between(0, 1)).all()
    assert (salidas["shaffer_pct"]["pct_significativo_vs_propuesto"].dropna().between(0, 100)).all()
    print("\n[OK] estadistica.py")


if __name__ == "__main__":
    main()
