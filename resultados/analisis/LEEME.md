# Cómo leer `resultados/analisis/`

Todo lo de esta carpeta sale de `resultados/corridas.parquet` (14 688
corridas: 3 algoritmos × 2 dimensionalidades × 4 MaxFES × 12 funciones ×
51 semillas), calculado por `analisis/` — ver `docs/setup_experimental.md`
§6-§9 para el protocolo completo. Nada de esto re-ejecuta ninguna corrida.

## `tablas/posiciones_finales.csv` — tabla resumen (§8)

Una fila por cada una de las **8 configuraciones** (2 dimensionalidades ×
4 MaxFES). Para cada una:

1. Se calcula el **error medio** de las 51 corridas de cada algoritmo,
   por separado en cada una de las 12 funciones CEC2022.
2. En cada función, se **ordenan los 3 algoritmos** de menor a mayor
   error medio y se les asigna un rango: 1 = mejor (menor error), 3 =
   peor. Si dos algoritmos difieren en menos de 1e-8 en su error medio,
   se consideran empatados y ambos reciben el **rango promedio** de las
   posiciones que ocupan (p. ej. empate en 1º-2º → ambos con rango 1.5;
   convención CEC 2017).
3. El **rango promedio** de un algoritmo (columnas `rango_promedio_*`)
   es el promedio de esos 12 rangos (uno por función) — NO es un
   promedio de errores, es un promedio de *posiciones*. Va de 1.0 (fue
   1º en las 12 funciones) a 3.0 (fue 3º en las 12).
4. Las columnas `puesto_1`, `puesto_2`, `puesto_3` muestran directamente qué algoritmo
   quedó en cada lugar en esa configuración, con su rango promedio entre
   paréntesis.

**Importante:** los 8 rankings NUNCA se promedian entre sí — cada
configuración (dim × MaxFES) se interpreta por separado (setup §6-§7).
El detalle función por función de cada uno está en
`tablas/rankings_d{10,20}_fes{5000,50000,500000,5000000}.csv`.

## `tablas/friedman.csv` — setup §7

Por configuración: test de Friedman sobre la matriz de rangos (12
funciones × 3 algoritmos). H0 = "los 3 algoritmos tienen el mismo rango
promedio". Dos columnas dicen lo mismo de dos formas:

- `rechaza_h0` (booleano): `True` si `p_valor < 0.05`.
- `conclusion` (texto): la misma lectura en palabras — si se rechaza H0
  hay evidencia de que **al menos un algoritmo difiere** de los otros
  (no dice todavía cuál — para eso está el post-hoc de Shaffer, más
  abajo); si NO se rechaza, no hay evidencia suficiente de que los 3
  algoritmos se comporten distinto en esa configuración.

## `tablas/shaffer_d*.csv` y `shaffer_pct_significativo.csv` — setup §7

Comparación por pares (middleware-pso, middleware-de, pso-de) con el
ajuste de Shaffer (controla mejor el error tipo I que comparar cada par
sin ajustar). `significativo=True` (p_ajustado < 0.05) → esos dos
algoritmos tienen desempeño distinguible en esa configuración.
`shaffer_pct_significativo.csv` resume qué % de las comparaciones donde
participa el propuesto salieron significativas.

## `tablas/wilcoxon_bew.csv` — setup §7 (extendido)

Test de Wilcoxon de rangos con signo, **emparejado por semilla**, por
función — para las 3 comparaciones por pares (middleware vs pso,
middleware vs de, y pso vs de, esta última agregada más allá del setup
original para caracterizar también la diferencia entre los dos
algoritmos base). El veredicto (`better`/`equal`/`worse`) se lee desde
la perspectiva de `algoritmo_a`: contando sobre las 12 funciones, cuántas
veces `algoritmo_a` fue significativamente mejor, no distinguible, o
significativamente peor que `algoritmo_b`.

## `tablas/desviacion_rangos.csv` — setup §7

Desviación estándar de los 3 `rango_promedio` de esa configuración —
qué tan dispersos quedaron los algoritmos entre sí (0 = los 3 empatados
en promedio; más alto = más separados).

## `figuras/cd_d*.png` — Diagramas de Diferencia Crítica (Nemenyi)

Un diagrama por configuración. Eje horizontal = rango promedio (1 a 3).
Cada algoritmo marcado en su posición. Una barra horizontal conecta a
los algoritmos cuya diferencia de rango es menor que la Diferencia
Crítica (CD, indicada en el título) — esos algoritmos NO son
distinguibles estadísticamente entre sí según Nemenyi (más conservador
que Shaffer: puede no marcar como distinguible un par que Shaffer sí
marcó significativo).

## `figuras/barras_rangos_d{10,20}.png`

Rango promedio de los 3 algoritmos a través de los 4 valores de MaxFES,
uno por dimensionalidad — para ver de un vistazo si el propuesto mejora,
empeora o se mantiene según el presupuesto de evaluaciones.

## `tablas/suplementario_media_mediana_std.csv`

Media, mediana y desviación estándar de las 51 corridas, por (algoritmo
× dim × MaxFES × función) — material suplementario obligatorio (§8),
más fino que el error medio ya usado en los rankings.

## `tablas/diagnostico_*.csv` — insumos para la discusión §9

- `diagnostico_d1d_activacion.csv`: % de corridas donde el middleware
  nunca llegó a activar la Fase 2 (`fes_primera_activacion` es NaN), por
  MaxFES — cuantifica la D-1d (con MaxFES=5000 el middleware casi no
  alcanza a actuar).
- `diagnostico_canales_a_b.csv`: uso promedio de Canal A vs Canal B por
  MaxFES.
- `diagnostico_r2_fase2.csv`: calidad del modelo subrogado (R²) por
  función.
