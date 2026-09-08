"""
Diagrama de Diferencia Crítica (post-hoc de Nemenyi, Demšar 2006),
setup §7-§8: visualización compacta de Friedman. Dos algoritmos son
significativamente distintos si la diferencia de sus rangos promedio
supera la Diferencia Crítica

    CD = q_alpha * sqrt( k(k+1) / (6N) )

con k=3 algoritmos, N=12 funciones, q_alpha = valor crítico del rango
studentizado (alpha=0.05) dividido por sqrt(2). Se calcula con
`scipy.stats.studentized_range` (sin tabla hardcodeada) — para k=3 da
q_alpha ≈ 2.3437, que coincide con la tabla publicada de Demšar (2006).

8 diagramas, uno por (dim × MaxFES).
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")  # sin display (Colab / headless) — guarda a archivo
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import studentized_range

from analisis.datos import (
    ALGORITMOS, N_FUNCIONES, NOMBRE_ALGORITMO, cargar_corridas, configuraciones, id_config,
)
from analisis.rankings import ranking_configuracion

ALPHA = 0.05


def q_alpha(k: int, alpha: float = ALPHA) -> float:
    """Valor crítico del rango studentizado para k tratamientos, dividido
    por sqrt(2) (Nemenyi). Calculado, no hardcodeado."""
    return float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))


def diferencia_critica(k: int, n: int, alpha: float = ALPHA) -> float:
    return q_alpha(k, alpha) * np.sqrt(k * (k + 1) / (6 * n))


def dibujar_cd(rangos_promedio: dict[str, float], cd: float, titulo: str, ruta_salida: Path) -> None:
    """
    Diagrama CD estilo Demšar: eje horizontal de rangos 1..k, un marcador
    por algoritmo en su rango promedio, y una barra uniendo los
    algoritmos cuyo rango difiere en menos de CD (grupo estadísticamente
    indistinguible).
    """
    algoritmos = sorted(rangos_promedio, key=rangos_promedio.get)
    valores = [rangos_promedio[a] for a in algoritmos]
    k = len(algoritmos)

    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="black", linewidth=1)
    for x in range(1, k + 1):
        ax.plot([x, x], [0.47, 0.53], color="black", linewidth=1)
        ax.text(x, 0.6, str(x), ha="center", fontsize=9)

    for a, v in zip(algoritmos, valores):
        ax.plot(v, 0.5, "o", color="tab:blue", markersize=6)
        ax.annotate(f"{NOMBRE_ALGORITMO.get(a, a)}\n({v:.2f})", xy=(v, 0.5),
                    xytext=(v, 0.15), ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="gray"))

    # Barras de grupo: algoritmos consecutivos (por rango) cuya diferencia
    # extremo-a-extremo es < CD quedan unidos como "indistinguibles".
    y_barra = 0.85
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (valores[j + 1] - valores[i]) < cd:
            j += 1
        if j > i:
            ax.plot([valores[i], valores[j]], [y_barra, y_barra], color="black", linewidth=2)
            y_barra -= 0.12
        i += 1

    ax.set_title(f"{titulo}  (CD = {cd:.3f})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def generar_todos(
    ruta_corridas: str | Path = "resultados/corridas.parquet",
    dir_salida: str | Path = "resultados/analisis/figuras",
) -> list[dict]:
    df = cargar_corridas(ruta_corridas)
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    k = len(ALGORITMOS)
    cd = diferencia_critica(k, N_FUNCIONES)
    resumen = []
    for dim, max_fes in configuraciones():
        ranking = ranking_configuracion(df, dim, max_fes)
        rangos = {a: ranking.loc[a, ("resumen", "rango_promedio")] for a in ALGORITMOS}
        cfg = id_config(dim, max_fes)
        ruta = dir_salida / f"cd_{cfg}.png"
        dibujar_cd(rangos, cd, f"CEC2022_{dim} — MaxFES={max_fes}", ruta)
        resumen.append({"dim": dim, "max_fes": max_fes, "cd": cd, "ruta": str(ruta), **rangos})
    return resumen


def main():
    k = len(ALGORITMOS)
    qa = q_alpha(k)
    cd = diferencia_critica(k, N_FUNCIONES)
    print(f"q_alpha(k={k}, alpha=0.05) = {qa:.4f}  (referencia publicada: 2.3437 para k=3)")
    print(f"CD (k={k}, N={N_FUNCIONES}) = {cd:.4f}\n")
    assert abs(qa - 2.3437) < 1e-3, "q_alpha no coincide con la tabla publicada de Demšar (2006)"

    resumen = generar_todos()
    for fila in resumen:
        print(f"  dim={fila['dim']:2d} MaxFES={fila['max_fes']:>9d}  -> {fila['ruta']}")
    print(f"\n[OK] cd_nemenyi.py — {len(resumen)} diagramas generados")


if __name__ == "__main__":
    main()
