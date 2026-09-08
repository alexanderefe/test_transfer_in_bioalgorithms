"""
Construcción de rankings (setup §6): a partir del error medio de 51
corridas por (algoritmo × función), rankea los algoritmos de 1 (mejor) a
N (peor) por función, con **empate si la diferencia es < 1e-8**
(convención CEC 2017), y calcula el rango promedio de cada algoritmo
sobre las 12 funciones del conjunto.

8 rankings independientes — uno por (dim × MaxFES) — nunca se promedian
entre configuraciones.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import pandas as pd

from analisis.datos import (
    ALGORITMOS, N_FUNCIONES, cargar_corridas, configuraciones, id_config,
    tabla_error_medio,
)

TOLERANCIA_EMPATE = 1e-8


def rankear_columna(col: pd.Series, tol: float = TOLERANCIA_EMPATE) -> pd.Series:
    """
    Rango 1 (mejor)..N (peor) de una función: ordena ascendente por error,
    agrupa en bloques consecutivos con diferencia < `tol` respecto al
    primero del bloque, y asigna a todo el bloque el **rango promedio**
    de las posiciones que ocupa (convención estándar para alimentar
    Friedman: dos algoritmos empatados en 1º-2º lugar reciben rango 1.5
    cada uno, no rango 1).
    """
    orden = col.sort_values()
    vals = orden.values
    idxs = orden.index
    rangos = pd.Series(index=col.index, dtype=float)
    i = 0
    pos = 1
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and abs(vals[j + 1] - vals[i]) < tol:
            j += 1
        rango_prom = (pos + (pos + (j - i))) / 2
        for k in range(i, j + 1):
            rangos[idxs[k]] = rango_prom
        pos += (j - i + 1)
        i = j + 1
    return rangos


def ranking_configuracion(df: pd.DataFrame, dim: int, max_fes: int) -> pd.DataFrame:
    """
    Ranking completo de una configuración: error medio y rango por
    función, más el rango promedio final (columna `rango_promedio`).
    """
    errores = tabla_error_medio(df, dim, max_fes)
    rangos = errores.apply(rankear_columna, axis=0)
    resultado = errores.copy()
    resultado.columns = pd.MultiIndex.from_product([["error_medio"], resultado.columns])
    rangos.columns = pd.MultiIndex.from_product([["rango"], rangos.columns])
    tabla = pd.concat([resultado, rangos], axis=1)
    tabla[("resumen", "rango_promedio")] = rangos.mean(axis=1)
    return tabla.sort_values(("resumen", "rango_promedio"))


def tabla_posiciones_finales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla resumen pedida en §8: posición final (rango promedio) de cada
    algoritmo en cada uno de los 8 rankings — una fila por configuración,
    una columna por algoritmo.
    """
    filas = []
    for dim, max_fes in configuraciones():
        ranking = ranking_configuracion(df, dim, max_fes)
        fila = {"dim": dim, "max_fes": max_fes}
        for alg in ALGORITMOS:
            fila[alg] = ranking.loc[alg, ("resumen", "rango_promedio")]
        filas.append(fila)
    return pd.DataFrame(filas)


def generar_todos(
    ruta_corridas: str | Path = "resultados/corridas.parquet",
    dir_salida: str | Path = "resultados/analisis/tablas",
) -> dict[str, pd.DataFrame]:
    """Genera y guarda los 8 rankings + la tabla de posiciones finales."""
    df = cargar_corridas(ruta_corridas)
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    salidas = {}
    for dim, max_fes in configuraciones():
        tabla = ranking_configuracion(df, dim, max_fes)
        nombre = f"rankings_{id_config(dim, max_fes)}"
        tabla.to_csv(dir_salida / f"{nombre}.csv")
        salidas[nombre] = tabla

    posiciones = tabla_posiciones_finales(df)
    posiciones.to_csv(dir_salida / "posiciones_finales.csv", index=False)
    salidas["posiciones_finales"] = posiciones
    return salidas


def main():
    salidas = generar_todos()
    print(f"{len(salidas) - 1} rankings + tabla de posiciones finales generados "
          f"en resultados/analisis/tablas/\n")
    dim, max_fes = 10, 500_000
    ejemplo = salidas[f"rankings_{id_config(dim, max_fes)}"]
    print(f"Ejemplo (dim={dim}, MaxFES={max_fes}):")
    print(ejemplo[("resumen", "rango_promedio")].round(3).to_string())
    assert list(ejemplo.index) == sorted(
        ejemplo.index, key=lambda a: ejemplo.loc[a, ("resumen", "rango_promedio")]
    ), "la tabla debe venir ordenada por rango promedio ascendente"
    print("\n[OK] rankings.py")


if __name__ == "__main__":
    main()
