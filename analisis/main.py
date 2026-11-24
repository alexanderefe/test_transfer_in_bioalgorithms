"""
Corre todo el análisis (setup §6-§9) de punta a punta: rankings,
estadística (Friedman/Shaffer/Wilcoxon/desviación), diagramas CD de
Nemenyi, figuras y tablas restantes (barras por MaxFES, suplementario,
diagnósticos §9) y gráficos de convergencia.

Uso:
    python analisis/main.py [--resultados resultados]
                            [--corridas <resultados>/corridas.parquet]
                            [--propuesto <resultados>/corridas_propuesto.parquet]
                            [--salida <resultados>/analisis]
                            [--algoritmos principal|principal+referencia|a,b,...]
                            [--convergencia-por-semilla]

`--resultados` es la misma carpeta que `--salida` de `experimentos/grid.py`
y `experimentos/consolidar.py`; las otras tres rutas se derivan de ella
salvo que se pasen explícitamente.

Las figuras de convergencia por semilla (4 896) solo se generan con
`--convergencia-por-semilla`; para más opciones (magnitud fitness, un
subconjunto de semillas) usar directamente `analisis/convergencia.py`.

Salida: `<salida>/tablas/*.csv` y `<salida>/figuras/*.png`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

# Atajos de --algoritmos (mismos nombres que analisis/datos.py).
_CONJUNTOS = {
    "principal": "middleware,pso,de,cmaes,madde,nlshadersp",
    "principal+referencia": ("middleware,pso,de,cmaes,madde,nlshadersp,"
                             "ga,jade,clpso,abc,clonalg,optainet"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resultados", default="resultados",
                    help="carpeta de resultados de la parrilla (--salida de grid.py)")
    ap.add_argument("--corridas", default=None)
    ap.add_argument("--propuesto", default=None)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--algoritmos", default="principal",
                    help="algoritmos a analizar: 'principal' (los 6 de la parrilla "
                         "principal, default), 'principal+referencia' (+ los 6 de "
                         "referencia de README_MAC_REFERENCIA.md) o una lista separada "
                         "por comas")
    ap.add_argument("--convergencia-por-semilla", action="store_true")
    args = ap.parse_args()
    # Debe fijarse antes de importar analisis/: datos.ALGORITMOS se lee al importar.
    os.environ["ANALISIS_ALGORITMOS"] = _CONJUNTOS.get(args.algoritmos, args.algoritmos)
    from analisis import (boxplots, cd_nemenyi, convergencia, estadistica,
                          figuras, rankings)
    from analisis.datos import ALGORITMOS
    print(f"Algoritmos (k={len(ALGORITMOS)}): {', '.join(ALGORITMOS)}")
    resultados = Path(args.resultados)
    args.corridas = args.corridas or str(resultados / "corridas.parquet")
    args.propuesto = args.propuesto or str(resultados / "corridas_propuesto.parquet")
    args.salida = args.salida or str(resultados / "analisis")

    dir_tablas = Path(args.salida) / "tablas"
    dir_figuras = Path(args.salida) / "figuras"

    print("== 1/6 Rankings (§6) ==")
    rankings.generar_todos(args.corridas, dir_tablas)

    print("== 2/6 Estadística: Friedman + Shaffer + Wilcoxon + desviación (§7) ==")
    estadistica.generar_todos(args.corridas, dir_tablas)

    print("== 3/6 Diagramas de Diferencia Crítica — Nemenyi (§7-§8) ==")
    cd_nemenyi.generar_todos(args.corridas, dir_figuras)

    print("== 4/6 Barras por MaxFES + suplementario + diagnósticos §9 ==")
    figuras.generar_todos(args.corridas, args.propuesto, dir_tablas, dir_figuras)

    print("== 5/6 Gráficos de convergencia (§8) ==")
    convergencia.generar_todos(args.corridas, args.salida,
                               por_semilla=args.convergencia_por_semilla)

    print("== 6/6 Boxplots del error final y de los rangos ==")
    boxplots.generar_todos(args.corridas, args.salida)

    print(f"\nListo. Tablas en {dir_tablas}/, figuras en {dir_figuras}/")


if __name__ == "__main__":
    main()
