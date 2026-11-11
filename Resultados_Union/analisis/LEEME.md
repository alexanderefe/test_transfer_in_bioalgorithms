# Cómo leer `Resultados_Union/analisis/`

Todo lo de esta carpeta sale de los dos consolidados de la parrilla
completa (Ubuntu + Mac Studio), calculado por `analisis/main.py` — ver
`docs/setup_experimental.md` §6-§9 para el protocolo. Nada de esto
re-ejecuta ninguna corrida:

- `Resultados_Union/corridas.parquet` — **29 376 corridas**: 6 algoritmos ×
  2 dimensionalidades (10, 20) × 4 MaxFES (5·10⁴, 5·10⁵, 5·10⁶, 5·10⁷) ×
  12 funciones CEC2022 × 51 semillas. Todas en `estado="ok"`.
- `Resultados_Union/corridas_propuesto.parquet` — 4 896 filas con los
  diagnósticos internos del middleware (una por corrida del propuesto).

Para regenerar la carpeta completa (~1 min):

```
python analisis/main.py --resultados Resultados_Union
```

Los 6 algoritmos y su nombre en las figuras:

| `algoritmo` | En figuras | Rol |
|---|---|---|
| `middleware` | Propuesto | Middleware de transferencia sobre PSO + DE |
| `pso` | PSO | Algoritmo base solo |
| `de` | DE | Algoritmo base solo |
| `cmaes` | IPOP-CMA-ES | Competidor externo |
| `madde` | MadDE | Competidor externo |
| `nlshadersp` | NL-SHADE-RSP | Competidor externo |

**Regla general:** todo se calcula por separado para cada una de las **8
configuraciones** (2 dimensionalidades × 4 MaxFES). Los resultados de
distintas configuraciones NUNCA se promedian entre sí (setup §6-§7). Los
archivos por configuración llevan el sufijo `d{dim}_fes{max_fes}`, p. ej.
`d10_fes50000000`.

## Tablas (`tablas/`)

### `rankings_d*_fes*.csv` — setup §6

Una por configuración. Filas = algoritmos. Tres bloques de columnas:

1. `error_medio` (funciones 1-12): error medio de las 51 corridas.
2. `rango` (funciones 1-12): en cada función se ordenan los 6 algoritmos
   de menor a mayor error medio; 1 = mejor, 6 = peor. Si dos algoritmos
   difieren en menos de 1e-8 se consideran empatados y reciben el rango
   promedio de las posiciones que ocupan (empate en 1º-2º → ambos 1.5;
   convención CEC 2017).
3. `rango_promedio`: promedio de esos 12 rangos. Es un promedio de
   *posiciones*, no de errores: va de 1.0 (1º en las 12 funciones) a 6.0
   (6º en las 12).

### `posiciones_finales.csv` — tabla resumen (§8)

Una fila por configuración. Columnas `rango_promedio_<algoritmo>` (las
mismas de los `rankings_*`) y `puesto_1` … `puesto_6`: qué algoritmo
quedó en cada lugar, con su rango promedio entre paréntesis.

### `friedman.csv` — setup §7

Test de Friedman sobre la matriz de rangos (12 funciones × 6
algoritmos). H0 = "los 6 algoritmos tienen el mismo rango promedio".
`rechaza_h0` es `True` si `p_valor < 0.05`, y `conclusion` dice lo mismo
en palabras. Rechazar H0 solo dice que **al menos un algoritmo difiere**;
cuál, lo dice el post-hoc de Shaffer.

### `shaffer_d*_fes*.csv` y `shaffer_pct_significativo.csv` — setup §7

Las 15 comparaciones por pares entre los 6 algoritmos, con el ajuste
estático de Shaffer (controla el error tipo I mejor que comparar cada
par sin ajustar). Columnas:

- `rango_a`, `rango_b`: rango promedio de cada algoritmo del par.
- `z`, `p_crudo`: estadístico de la diferencia de rangos y su p-valor
  sin ajustar.
- `divisor_shaffer`: factor por el que se multiplica `p_crudo` (máximo
  nº de hipótesis que pueden ser verdaderas a la vez en ese paso).
- `p_ajustado`, `significativo`: `True` si `p_ajustado < 0.05` → ese par
  es distinguible en esa configuración.

`shaffer_pct_significativo.csv` resume, por configuración, qué % de las
5 comparaciones donde participa el propuesto salieron significativas
(20 % = 1 de 5).

### `wilcoxon_bew.csv` — setup §7

Test de Wilcoxon de rangos con signo, **emparejado por semilla** (51
pares), por función. Compara el propuesto contra cada uno de los otros 5
algoritmos, más `pso` vs `de` (para caracterizar la diferencia entre los
dos algoritmos base). Por función, el veredicto es:

- `equal` si p ≥ 0.05 (no distinguibles);
- si no, `better` o `worse` según cuál tenga menor error medio.

Se lee desde la perspectiva de `algoritmo_a`: las columnas `better`,
`equal`, `worse` cuentan, sobre las 12 funciones, cuántas veces
`algoritmo_a` fue significativamente mejor, no distinguible o
significativamente peor que `algoritmo_b`.

A diferencia de Friedman/Shaffer (que trabajan sobre el error *medio*
por función), Wilcoxon usa las 51 corridas individuales: puede detectar
diferencias en una función que el ranking no muestra.

### `desviacion_rangos.csv` — setup §7

Desviación estándar de los 6 `rango_promedio` de cada configuración: qué
tan separados quedaron los algoritmos entre sí (0 = todos empatados en
promedio; más alto = más separados).

### `suplementario_media_mediana_std.csv` — §8

Media, mediana y desviación estándar del error de las 51 corridas, por
(algoritmo × dim × MaxFES × función). Es el material suplementario
obligatorio, más fino que el error medio usado en los rankings.

### `convergencia_medianas.csv` — §8

Los datos detrás de los gráficos de convergencia: por (dim × MaxFES ×
función × algoritmo × punto de control), el cuartil 1, la mediana y el
cuartil 3 del error de las 51 semillas, y la mediana del fitness. Hay
120 puntos de control por curva: `fraccion` es la fracción de MaxFES
consumida (tramo logarítmico de 1e-4 a 1e-2 y luego uno por cada 1 %) y
`fes` las evaluaciones equivalentes.

### `diagnostico_*.csv` — insumos para la discusión §9

Salen de `corridas_propuesto.parquet` y solo describen al propuesto.

- `diagnostico_d1d_activacion.csv`: % de corridas donde el middleware
  **nunca activó la Fase 2** (`fes_primera_activacion` es NaN), por
  MaxFES. Cuantifica si el presupuesto alcanza para que el middleware
  llegue a actuar.
- `diagnostico_canales_a_b.csv`: nº promedio de usos del Canal A y del
  Canal B por corrida, por MaxFES.
- `diagnostico_r2_fase2.csv`: calidad del modelo subrogado (R² de la
  Fase 2) por función: media, mediana, desviación y nº de
  entrenamientos. **Usar la mediana:** algunos entrenamientos dan R²
  negativos enormes (subrogado peor que predecir la media) que arrastran
  la media a valores sin sentido, p. ej. −1898 en F1.

## Figuras (`figuras/`)

### `cd_d*_fes*.png` — Diagramas de Diferencia Crítica (Nemenyi)

Uno por configuración. Eje horizontal = rango promedio (1 a 6), con cada
algoritmo marcado en su posición. Una barra horizontal agrupa a los
algoritmos cuya diferencia de rango es menor que la Diferencia Crítica
(CD, en el título; con 6 algoritmos y 12 funciones, CD ≈ 2.18): esos
algoritmos **no** son distinguibles entre sí según Nemenyi. Nemenyi es
más conservador que Shaffer: puede no separar un par que Shaffer sí
marcó significativo.

### `barras_rangos_d10.png`, `barras_rangos_d20.png`

Rango promedio de los 6 algoritmos en los 4 MaxFES, una figura por
dimensionalidad. Sirve para ver de un vistazo si el propuesto mejora,
empeora o se mantiene según el presupuesto de evaluaciones.

### `convergencia_d*_fes*.png` — §8

Un panel por configuración con las 12 funciones. Cada curva es la
**mediana de las 51 semillas** del error (f − f(x*)) a medida que se
consumen evaluaciones, y la banda sombreada es el rango intercuartil.
Ambos ejes en escala logarítmica. Los errores menores que 1e-8 se dibujan
en 1e-8 (convención CEC: se consideran óptimo alcanzado). El propuesto va
en rojo y con línea más gruesa.

### `convergencia/d*_fes*/fNN.png`

Lo mismo que el panel, pero una figura más grande por función (96 en
total).

Las figuras **por semilla** (4 896) no se generan por defecto: usar
`python analisis/main.py --resultados Resultados_Union --convergencia-por-semilla`,
o `analisis/convergencia.py --por-semilla --semillas 0,1,2` para solo
algunas. `analisis/convergencia.py --magnitud fitness` genera los paneles
con el valor objetivo f en vez del error (archivos con sufijo `_fitness`).
