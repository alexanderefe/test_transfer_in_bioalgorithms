"""
Boxplots (complemento de §8): la distribución completa de las 51 semillas,
que el error medio de los rankings y la mediana + banda intercuartil de los
gráficos de convergencia resumen en uno o tres números.

Siempre por configuración (dim × MaxFES), sin mezclar configuraciones:
  - `boxplot_d{dim}_fes{max_fes}.png`: panel con las 12 funciones; en cada
    una, un boxplot por algoritmo del error final (f − f(x*)) de las 51
    semillas (8 figuras);
  - `boxplot/d{dim}_fes{max_fes}/fNN.png`: una figura por función, más
    grande (96 figuras);
  - `boxplot_rangos_d{dim}.png`: por MaxFES, un boxplot por algoritmo de
    los 12 rangos por función de los rankings (§6) — la dispersión detrás
    del rango promedio de `barras_rangos_d{dim}.png` (2 figuras);
  - `tablas/boxplot_cuartiles.csv`: los números de cada caja.

Caja = rango intercuartil (P25–P75), línea = mediana, bigotes = el dato más
lejano dentro de 1.5 × RIC, puntos = valores fuera de los bigotes. Eje Y del
error en escala log; los errores < 1e-8 se dibujan en 1e-8 (convención CEC,
`UMBRAL_ERROR_CERO`). Color por grupo (propuesto, parrilla principal, y cada
familia de los algoritmos de referencia); el nombre de cada algoritmo va en
el eje X, así que el color nunca es la única forma de identificarlo.

    python analisis/boxplots.py [--resultados resultados] [--sin-por-funcion]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analisis.datos import (
    ALGORITMOS, DIMS, GRUPO_ALGORITMO, MAXFES, N_FUNCIONES, NOMBRE_ALGORITMO,
    cargar_corridas, configuraciones, id_config,
)
from analisis.rankings import ranking_configuracion
from problems.cec2022_wrapper import UMBRAL_ERROR_CERO

# Un color por grupo (paleta categórica de referencia, slots 8, 1, 2, 3 y 7).
COLOR_GRUPO = {
    "propuesto": "#e34948",
    "principal": "#2a78d6",
    "evolutivo": "#eb6834",
    "enjambre": "#1baf7a",
    "inmune": "#6250d6",
}
NOMBRE_GRUPO = {
    "propuesto": "Propuesto",
    "principal": "Base y competidores",
    "evolutivo": "Referencia: evolutivos",
    "enjambre": "Referencia: enjambre",
    "inmune": "Referencia: inmunes",
}
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"


def _titulo(dim: int, max_fes: int) -> str:
    return f"CEC2022_{dim}, MaxFES = " + f"{max_fes:,}".replace(",", ".")


def errores_finales(df: pd.DataFrame, dim: int, max_fes: int, funcion: int) -> dict[str, np.ndarray]:
    """Error final de las semillas de cada algoritmo, con piso en 1e-8."""
    sub = df[(df["dim"] == dim) & (df["max_fes"] == max_fes) & (df["funcion"] == funcion)]
    return {a: np.maximum(g["error"].to_numpy(dtype=float), UMBRAL_ERROR_CERO)
            for a, g in sub.groupby("algoritmo")}


def _cajas(ax, datos: list[np.ndarray], algoritmos: list[str], log: bool,
           fuente_etiquetas: float, ancho_caja: float = 0.65) -> None:
    """Dibuja un boxplot por algoritmo, coloreado por grupo."""
    bp = ax.boxplot(datos, widths=ancho_caja, patch_artist=True, whis=1.5,
                    medianprops=dict(color=TINTA, linewidth=1.4),
                    whiskerprops=dict(color=TINTA_SECUNDARIA, linewidth=0.9),
                    capprops=dict(color=TINTA_SECUNDARIA, linewidth=0.9),
                    flierprops=dict(marker="o", markersize=2.5, markeredgewidth=0.5,
                                    markerfacecolor="none", alpha=0.7))
    for caja, volado, a in zip(bp["boxes"], bp["fliers"], algoritmos):
        color = COLOR_GRUPO[GRUPO_ALGORITMO.get(a, "principal")]
        caja.set(facecolor=color, alpha=0.45, edgecolor=color, linewidth=1.1)
        volado.set(markeredgecolor=color)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(1, len(algoritmos) + 1))
    ax.set_xticklabels([NOMBRE_ALGORITMO.get(a, a) for a in algoritmos],
                       rotation=90, fontsize=fuente_etiquetas, color=TINTA)
    ax.tick_params(axis="y", labelsize=fuente_etiquetas + 1, colors=TINTA_SECUNDARIA)
    ax.grid(True, axis="y", which="major", alpha=0.25)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def _leyenda_grupos(fig, algoritmos: list[str], extra: list | None = None, **kwargs) -> None:
    grupos = list(dict.fromkeys(GRUPO_ALGORITMO.get(a, "principal") for a in algoritmos))
    manejadores = [Patch(facecolor=COLOR_GRUPO[g], alpha=0.45, edgecolor=COLOR_GRUPO[g],
                         label=NOMBRE_GRUPO[g]) for g in grupos] + (extra or [])
    fig.legend(handles=manejadores, frameon=False, ncol=len(manejadores), **kwargs)


def panel_configuracion(df: pd.DataFrame, dim: int, max_fes: int, ruta: Path) -> None:
    """Las 12 funciones de una configuración en una grilla de 3 × 4."""
    algoritmos = [a for a in ALGORITMOS]
    fig, ejes = plt.subplots(3, 4, figsize=(18, 13))
    for funcion, ax in zip(range(1, N_FUNCIONES + 1), ejes.flat):
        err = errores_finales(df, dim, max_fes, funcion)
        _cajas(ax, [err[a] for a in algoritmos], algoritmos, log=True,
               fuente_etiquetas=6.5 if len(algoritmos) > 8 else 8)
        ax.set_title(f"F{funcion}", fontsize=10)
    for ax in ejes[:, 0]:
        ax.set_ylabel("Error final  f − f(x*)")
    n = df["idx_semilla"].nunique()
    fig.suptitle(f"Error final — {_titulo(dim, max_fes)} ({n} semillas por caja)")
    _leyenda_grupos(fig, algoritmos, loc="lower center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_funcion(df: pd.DataFrame, dim: int, max_fes: int, funcion: int, ruta: Path) -> None:
    """Una (dim, MaxFES, función)."""
    algoritmos = list(ALGORITMOS)
    err = errores_finales(df, dim, max_fes, funcion)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.6 * len(algoritmos) + 2), 5.2))
    _cajas(ax, [err[a] for a in algoritmos], algoritmos, log=True, fuente_etiquetas=9)
    ax.set_ylabel("Error final  f − f(x*)")
    n = len(next(iter(err.values())))
    ax.set_title(f"F{funcion} — {_titulo(dim, max_fes)} ({n} semillas por caja)", fontsize=10)
    _leyenda_grupos(fig, algoritmos, loc="upper center", fontsize=8,
                    bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def rangos_por_funcion(df: pd.DataFrame, dim: int, max_fes: int) -> pd.DataFrame:
    """Rangos (algoritmo × función) del ranking de la configuración (§6)."""
    return ranking_configuracion(df, dim, max_fes)["rango"]


def figura_rangos(df: pd.DataFrame, dim: int, ruta: Path) -> None:
    """Por MaxFES, un boxplot por algoritmo de sus 12 rangos por función."""
    algoritmos = list(ALGORITMOS)
    k = len(algoritmos)
    fig, ejes = plt.subplots(1, len(MAXFES), figsize=(4.2 * len(MAXFES), 5.2), sharey=True)
    for ax, max_fes in zip(ejes, MAXFES):
        rangos = rangos_por_funcion(df, dim, max_fes)
        _cajas(ax, [rangos.loc[a].to_numpy(dtype=float) for a in algoritmos], algoritmos,
               log=False, fuente_etiquetas=7 if k > 8 else 8.5)
        medias = [rangos.loc[a].mean() for a in algoritmos]
        marcador, = ax.plot(range(1, k + 1), medias, linestyle="none", marker="D",
                            markersize=4, color=TINTA, label="Rango promedio")
        ax.set_title("MaxFES = " + f"{max_fes:,}".replace(",", "."), fontsize=10)
        ax.set_ylim(k + 0.5, 0.5)          # 1 = mejor, arriba
        ax.set_yticks(range(1, k + 1))
    ejes[0].set_ylabel(f"Rango por función (1 = mejor, {k} = peor)")
    fig.suptitle(f"Distribución de los rangos sobre las 12 funciones — CEC2022_{dim}")
    _leyenda_grupos(fig, algoritmos, extra=[marcador], loc="lower center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def tabla_cuartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Los números de cada caja: mínimo, P25, mediana, P75, máximo y bigotes
    (1.5 × RIC), por (algoritmo × dim × MaxFES × función), sobre el error
    final sin piso."""
    def resumen(e: pd.Series) -> pd.Series:
        q1, med, q3 = np.percentile(e, [25, 50, 75])
        ric = q3 - q1
        dentro = e[(e >= q1 - 1.5 * ric) & (e <= q3 + 1.5 * ric)]
        return pd.Series({
            "minimo": e.min(), "q1": q1, "mediana": med, "q3": q3, "maximo": e.max(),
            "bigote_inf": dentro.min(), "bigote_sup": dentro.max(),
            "n_atipicos": int(len(e) - len(dentro)),
        })
    sub = df[df["algoritmo"].isin(ALGORITMOS)]
    return (sub.groupby(["algoritmo", "dim", "max_fes", "funcion"])["error"]
            .apply(resumen).unstack().reset_index())


def generar_todos(
    ruta_corridas: str | Path = "resultados/corridas.parquet",
    dir_salida: str | Path = "resultados/analisis",
    por_funcion: bool = True,
) -> None:
    df = cargar_corridas(ruta_corridas)
    dir_figuras = Path(dir_salida) / "figuras"
    dir_tablas = Path(dir_salida) / "tablas"
    dir_figuras.mkdir(parents=True, exist_ok=True)
    dir_tablas.mkdir(parents=True, exist_ok=True)

    for dim, max_fes in configuraciones():
        cfg = id_config(dim, max_fes)
        panel_configuracion(df, dim, max_fes, dir_figuras / f"boxplot_{cfg}.png")
        if por_funcion:
            dir_cfg = dir_figuras / "boxplot" / cfg
            dir_cfg.mkdir(parents=True, exist_ok=True)
            for funcion in range(1, N_FUNCIONES + 1):
                figura_funcion(df, dim, max_fes, funcion, dir_cfg / f"f{funcion:02d}.png")
        print(f"  boxplot {cfg} listo")
    for dim in DIMS:
        figura_rangos(df, dim, dir_figuras / f"boxplot_rangos_d{dim}.png")
    tabla_cuartiles(df).to_csv(dir_tablas / "boxplot_cuartiles.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--corridas", default=None)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--sin-por-funcion", action="store_true",
                    help="solo los 8 paneles y los 2 de rangos, sin las 96 figuras por función")
    args = ap.parse_args()
    args.corridas = args.corridas or str(Path(args.resultados) / "corridas.parquet")
    args.salida = args.salida or str(Path(args.resultados) / "analisis")
    generar_todos(args.corridas, args.salida, por_funcion=not args.sin_por_funcion)
    print(f"\n[OK] boxplots.py — figuras en {Path(args.salida) / 'figuras'}/")


if __name__ == "__main__":
    main()
