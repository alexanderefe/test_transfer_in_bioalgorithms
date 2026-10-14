"""
Gráficos de convergencia de los 6 algoritmos (setup §8), a partir de la
traza `error_checkpoints` de cada corrida (§11.2): el mejor valor visto
hasta cada fracción de `FRACCIONES_CHECKPOINT` (120 puntos: tramo
logarítmico 1e-4…1e-2 y un punto por cada 1 % de MaxFES).

Tres niveles de detalle, siempre por configuración (dim × MaxFES), sin
mezclar configuraciones entre sí:
  - panel por configuración: las 12 funciones en una figura, mediana de
    las 51 semillas + banda intercuartil (8 figuras);
  - por función: una figura por (dim, MaxFES, función), igual que el panel
    pero más grande (96 figuras);
  - por semilla: una figura por (dim, MaxFES, función, semilla), la curva
    de cada algoritmo en esa semilla (4 896 figuras; opcional).

Magnitud del eje Y (`magnitud`):
  - "error"   (default): f - f(x*), escala log. Los errores < 1e-8 se
    dibujan en 1e-8 (convención CEC, `UMBRAL_ERROR_CERO`).
  - "fitness": f, el valor objetivo, = f(x*) + error (exacto: la traza se
    guarda sin umbral). También en escala log; como f(x*) va de 300 a
    2700, las diferencias cerca del óptimo se ven mejor con "error".

Uso:
    python analisis/convergencia.py [--corridas resultados/corridas.parquet]
        [--salida resultados/analisis] [--magnitud error|fitness]
        [--sin-por-funcion] [--por-semilla] [--semillas 0,1,2]
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

from analisis.datos import (
    ALGORITMOS, N_FUNCIONES, NOMBRE_ALGORITMO, cargar_corridas, configuraciones,
    id_config,
)
from problems.cec2022_wrapper import FRACCIONES_CHECKPOINT, UMBRAL_ERROR_CERO

MAGNITUDES = ("error", "fitness")

# Color fijo por algoritmo en todas las figuras; el propuesto se destaca
# con línea más gruesa y por encima de los demás.
COLOR_ALGORITMO = {
    "middleware": "#d62728",
    "pso": "#1f77b4",
    "de": "#2ca02c",
    "cmaes": "#9467bd",
    "madde": "#ff7f0e",
    "nlshadersp": "#8c564b",
}


def fes_checkpoints(max_fes: int) -> np.ndarray:
    """FES de cada checkpoint, igual que `ProblemaCEC2022.configurar_checkpoints`."""
    return np.array([max(1, int(np.ceil(f * max_fes))) for f in FRACCIONES_CHECKPOINT])


def _errores_crudos(df: pd.DataFrame, dim: int, max_fes: int, funcion: int
                    ) -> tuple[dict[str, np.ndarray], float]:
    """Por algoritmo, la matriz (n_semillas × n_checkpoints) de errores tal
    como se guardaron, filas ordenadas por `idx_semilla` (fila i = semilla
    i, emparejadas entre algoritmos); y el f(x*) de la función."""
    sub = df[(df["dim"] == dim) & (df["max_fes"] == max_fes)
             & (df["funcion"] == funcion)].sort_values("idx_semilla")
    errores = {
        algoritmo: np.stack(grupo["error_checkpoints"].map(np.asarray).to_list())
        for algoritmo, grupo in sub.groupby("algoritmo", sort=False)
    }
    return errores, float(sub["f_opt"].iloc[0])


def trazas(df: pd.DataFrame, dim: int, max_fes: int, funcion: int,
           magnitud: str = "error") -> dict[str, np.ndarray]:
    """
    Trazas de convergencia de una (dim, MaxFES, función) en la magnitud
    pedida: por algoritmo, una matriz (n_semillas × n_checkpoints) con las
    filas ordenadas por `idx_semilla`.
    """
    if magnitud not in MAGNITUDES:
        raise ValueError(f"magnitud debe ser una de {MAGNITUDES}, no '{magnitud}'")
    errores, f_opt = _errores_crudos(df, dim, max_fes, funcion)
    if magnitud == "error":
        return {a: np.maximum(e, UMBRAL_ERROR_CERO) for a, e in errores.items()}
    return {a: f_opt + e for a, e in errores.items()}


def _titulo(dim: int, max_fes: int) -> str:
    """'CEC2022_10, MaxFES = 50.000.000' (separador de miles con punto)."""
    return f"CEC2022_{dim}, MaxFES = " + f"{max_fes:,}".replace(",", ".")


def _rotulo_agregado(trz: dict[str, np.ndarray]) -> str:
    n = max(len(m) for m in trz.values())
    return f"mediana de {n} semillas, banda = rango intercuartil"


def _formatear_eje(ax, max_fes: int, magnitud: str, etiquetas: bool = True) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(fes_checkpoints(max_fes)[0], max_fes)
    ax.grid(True, which="major", alpha=0.3)
    if etiquetas:
        ax.set_xlabel("FES")
        ax.set_ylabel("Error  f − f(x*)" if magnitud == "error" else "Fitness  f")


def _dibujar_agregado(ax, trz: dict[str, np.ndarray], x: np.ndarray) -> None:
    """Mediana de las semillas + banda intercuartil, un color por algoritmo."""
    for algoritmo in ALGORITMOS:
        if algoritmo not in trz:
            continue
        q1, mediana, q3 = np.percentile(trz[algoritmo], [25, 50, 75], axis=0)
        propuesto = algoritmo == "middleware"
        color = COLOR_ALGORITMO[algoritmo]
        ax.fill_between(x, q1, q3, color=color, alpha=0.15, linewidth=0,
                        zorder=3 if propuesto else 2)
        ax.plot(x, mediana, color=color, label=NOMBRE_ALGORITMO[algoritmo],
                linewidth=2.0 if propuesto else 1.2, zorder=4 if propuesto else 3)


def _leyenda_figura(fig, ax) -> None:
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               ncol=len(ALGORITMOS), fontsize=9, frameon=False)


def panel_configuracion(df: pd.DataFrame, dim: int, max_fes: int, ruta: Path,
                        magnitud: str = "error") -> None:
    """Las 12 funciones de una configuración en una grilla de 3 × 4."""
    x = fes_checkpoints(max_fes)
    fig, ejes = plt.subplots(3, 4, figsize=(16, 10.5))
    for funcion, ax in zip(range(1, N_FUNCIONES + 1), ejes.flat):
        trz = trazas(df, dim, max_fes, funcion, magnitud)
        _dibujar_agregado(ax, trz, x)
        _formatear_eje(ax, max_fes, magnitud, etiquetas=False)
        ax.set_title(f"F{funcion}", fontsize=10)
    for ax in ejes[-1]:
        ax.set_xlabel("FES")
    for ax in ejes[:, 0]:
        ax.set_ylabel("Error" if magnitud == "error" else "Fitness")
    fig.suptitle(f"Convergencia — {_titulo(dim, max_fes)} ({_rotulo_agregado(trz)})")
    _leyenda_figura(fig, ejes.flat[0])
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_funcion(df: pd.DataFrame, dim: int, max_fes: int, funcion: int,
                   ruta: Path, magnitud: str = "error") -> None:
    """Una (dim, MaxFES, función): mediana + banda intercuartil."""
    trz = trazas(df, dim, max_fes, funcion, magnitud)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    _dibujar_agregado(ax, trz, fes_checkpoints(max_fes))
    _formatear_eje(ax, max_fes, magnitud)
    ax.set_title(f"F{funcion} — {_titulo(dim, max_fes)}\n{_rotulo_agregado(trz)}",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_semilla(df: pd.DataFrame, dim: int, max_fes: int, funcion: int,
                   idx_semilla: int, ruta: Path, magnitud: str = "error") -> None:
    """Una (dim, MaxFES, función, semilla): la curva de cada algoritmo."""
    # Filtrar por idx_semilla antes de armar las trazas: cada algoritmo
    # queda con una sola fila, aunque falten otras semillas.
    trz = trazas(df[df["idx_semilla"] == idx_semilla], dim, max_fes, funcion, magnitud)
    x = fes_checkpoints(max_fes)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for algoritmo in ALGORITMOS:
        if algoritmo not in trz:
            continue
        propuesto = algoritmo == "middleware"
        ax.plot(x, trz[algoritmo][0], color=COLOR_ALGORITMO[algoritmo],
                label=NOMBRE_ALGORITMO[algoritmo],
                linewidth=2.0 if propuesto else 1.2, zorder=4 if propuesto else 3)
    _formatear_eje(ax, max_fes, magnitud)
    ax.set_title(f"F{funcion} — {_titulo(dim, max_fes)}, semilla {idx_semilla}",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def tabla_curvas_medianas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Curvas agregadas en formato largo: una fila por (dim, MaxFES, función,
    algoritmo, checkpoint) con q1/mediana/q3 del error y la mediana del
    fitness — para re-graficar fuera de Python (p. ej. pgfplots).
    """
    filas = []
    for dim, max_fes in configuraciones():
        x = fes_checkpoints(max_fes)
        for funcion in range(1, N_FUNCIONES + 1):
            errores, f_opt = _errores_crudos(df, dim, max_fes, funcion)
            for algoritmo, err in errores.items():
                q1, mediana, q3 = np.percentile(err, [25, 50, 75], axis=0)
                filas.append(pd.DataFrame({
                    "dim": dim, "max_fes": max_fes, "funcion": funcion,
                    "algoritmo": algoritmo, "fraccion": FRACCIONES_CHECKPOINT,
                    "fes": x, "error_q1": q1, "error_mediana": mediana,
                    "error_q3": q3, "fitness_mediana": f_opt + mediana,
                }))
    return pd.concat(filas, ignore_index=True)


def generar_todos(
    ruta_corridas: str | Path = "resultados/corridas.parquet",
    dir_salida: str | Path = "resultados/analisis",
    magnitud: str = "error",
    por_funcion: bool = True,
    por_semilla: bool = False,
    semillas: list[int] | None = None,
) -> None:
    """
    Genera las figuras de convergencia en `<dir_salida>/figuras/`:
      - `convergencia_d{dim}_fes{max_fes}.png` (8 paneles),
      - `convergencia/d{dim}_fes{max_fes}/f{NN}.png` si `por_funcion`,
      - `convergencia/d{dim}_fes{max_fes}/semillas/f{NN}_s{SS}.png` si
        `por_semilla` (todas las semillas, o solo `semillas`),
    y la tabla `<dir_salida>/tablas/convergencia_medianas.csv`.
    Con `magnitud="fitness"` los nombres llevan el sufijo `_fitness`.
    """
    df = cargar_corridas(ruta_corridas)
    dir_figuras = Path(dir_salida) / "figuras"
    dir_tablas = Path(dir_salida) / "tablas"
    dir_figuras.mkdir(parents=True, exist_ok=True)
    dir_tablas.mkdir(parents=True, exist_ok=True)
    sufijo = "" if magnitud == "error" else f"_{magnitud}"

    for dim, max_fes in configuraciones():
        cfg = id_config(dim, max_fes)
        panel_configuracion(df, dim, max_fes,
                            dir_figuras / f"convergencia_{cfg}{sufijo}.png", magnitud)
        dir_cfg = dir_figuras / "convergencia" / cfg
        if por_funcion:
            dir_cfg.mkdir(parents=True, exist_ok=True)
            for funcion in range(1, N_FUNCIONES + 1):
                figura_funcion(df, dim, max_fes, funcion,
                               dir_cfg / f"f{funcion:02d}{sufijo}.png", magnitud)
        if por_semilla:
            dir_sem = dir_cfg / "semillas"
            dir_sem.mkdir(parents=True, exist_ok=True)
            idxs = semillas if semillas is not None else sorted(df["idx_semilla"].unique())
            for funcion in range(1, N_FUNCIONES + 1):
                for idx in idxs:
                    figura_semilla(df, dim, max_fes, funcion, int(idx),
                                   dir_sem / f"f{funcion:02d}_s{int(idx):02d}{sufijo}.png",
                                   magnitud)
        print(f"  convergencia {cfg} listo")

    tabla_curvas_medianas(df).to_csv(dir_tablas / "convergencia_medianas.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corridas", default="resultados/corridas.parquet")
    ap.add_argument("--salida", default="resultados/analisis")
    ap.add_argument("--magnitud", choices=MAGNITUDES, default="error")
    ap.add_argument("--sin-por-funcion", action="store_true",
                    help="solo los 8 paneles, sin las 96 figuras por función")
    ap.add_argument("--por-semilla", action="store_true",
                    help="una figura por (dim, MaxFES, función, semilla)")
    ap.add_argument("--semillas", default=None,
                    help="con --por-semilla: lista de idx_semilla, p. ej. 0,1,2")
    args = ap.parse_args()
    semillas = [int(s) for s in args.semillas.split(",")] if args.semillas else None
    generar_todos(args.corridas, args.salida, args.magnitud,
                  por_funcion=not args.sin_por_funcion,
                  por_semilla=args.por_semilla, semillas=semillas)
    print(f"\n[OK] convergencia.py — figuras en {Path(args.salida) / 'figuras'}/")


if __name__ == "__main__":
    main()
