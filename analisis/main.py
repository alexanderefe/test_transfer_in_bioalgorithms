"""
Corre todo el análisis (setup §6-§9) de punta a punta: rankings,
estadística (Friedman/Shaffer/Wilcoxon/desviación), diagramas CD de
Nemenyi, figuras y tablas restantes (barras por MaxFES, suplementario,
diagnósticos §9).

Uso:
    python analisis/main.py [--corridas resultados/corridas.parquet]
                            [--propuesto resultados/corridas_propuesto.parquet]
                            [--salida resultados/analisis]

Salida: `<salida>/tablas/*.csv` y `<salida>/figuras/*.png`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from analisis import cd_nemenyi, estadistica, figuras, rankings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corridas", default="resultados/corridas.parquet")
    ap.add_argument("--propuesto", default="resultados/corridas_propuesto.parquet")
    ap.add_argument("--salida", default="resultados/analisis")
    args = ap.parse_args()

    dir_tablas = Path(args.salida) / "tablas"
    dir_figuras = Path(args.salida) / "figuras"

    print("== 1/4 Rankings (§6) ==")
    rankings.generar_todos(args.corridas, dir_tablas)

    print("== 2/4 Estadística: Friedman + Shaffer + Wilcoxon + desviación (§7) ==")
    estadistica.generar_todos(args.corridas, dir_tablas)

    print("== 3/4 Diagramas de Diferencia Crítica — Nemenyi (§7-§8) ==")
    cd_nemenyi.generar_todos(args.corridas, dir_figuras)

    print("== 4/4 Barras por MaxFES + suplementario + diagnósticos §9 ==")
    figuras.generar_todos(args.corridas, args.propuesto, dir_tablas, dir_figuras)

    print(f"\nListo. Tablas en {dir_tablas}/, figuras en {dir_figuras}/")


if __name__ == "__main__":
    main()
