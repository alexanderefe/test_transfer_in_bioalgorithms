> **Análisis con k = 12 algoritmos:** middleware, pso, de, cmaes, madde, nlshadersp, ga, jade, clpso, abc, clonalg, optainet. Donde abajo dice 6 algoritmos, rangos de 1 a 6, 15 comparaciones de Shaffer o CD ≈ 2.18, léase k = 12, rangos de 1 a 12, 66 comparaciones y la CD del título de cada diagrama. Wilcoxon compara al propuesto con los otros 11 algoritmos (+ pso vs de). Las figuras `boxplot_*` muestran la distribución de las 51 semillas (error final) y de los 12 rangos por función.

# Cómo leer `resultados/analisis/`

Todo lo de esta carpeta sale de `resultados/corridas.parquet` (29 376
corridas: 6 algoritmos × 2 dimensionalidades × 4 MaxFES × 12 funciones ×
51 semillas), calculado por `analisis/` — ver `docs/setup_experimental.md`
§6-§9 para el protocolo completo. Nada de esto re-ejecuta ninguna corrida.

Los 6 algoritmos: el propuesto (`middleware`), sus dos algoritmos base
corriendo solos (`pso`, `de`, D-4) y tres competidores externos
(`cmaes` = IPOP-CMA-ES, `madde` = MadDE, `nlshadersp` = NL-SHADE-RSP, D-8;
ver `docs/configuracion_experimental.md`).

## `tablas/posiciones_finales.csv` — tabla resumen (§8)

Una fila por cada una de las **8 configuraciones** (2 dimensionalidades ×
4 MaxFES). Para cada una:

1. Se calcula el **error medio** de las 51 corridas de cada algoritmo,
   por separado en cada una de las 12 funciones CEC2022.
2. En cada función, se **ordenan los 6 algoritmos** de menor a mayor
   error medio y se les asigna un rango: 1 = mejor (menor error), 6 =
   peor. Si dos algoritmos difieren en menos de 1e-8 en su error medio,
   se consideran empatados y ambos reciben el **rango promedio** de las
   posiciones que ocupan (p. ej. empate en 1º-2º → ambos con rango 1.5;
   convención CEC 2017).
3. El **rango promedio** de un algoritmo (columnas `rango_promedio_*`)
   es el promedio de esos 12 rangos (uno por función) — NO es un
   promedio de errores, es un promedio de *posiciones*. Va de 1.0 (fue
   1º en las 12 funciones) a 6.0 (fue 6º en las 12).
4. Las columnas `puesto_1` … `puesto_6` muestran directamente qué algoritmo
   quedó en cada lugar en esa configuración, con su rango promedio entre
   paréntesis.

**Importante:** los 8 rankings NUNCA se promedian entre sí — cada
configuración (dim × MaxFES) se interpreta por separado (setup §6-§7).
El detalle función por función de cada uno está en
`tablas/rankings_d{10,20}_fes{50000,500000,5000000,50000000}.csv`.

## `tablas/friedman.csv` — setup §7

Por configuración: test de Friedman sobre la matriz de rangos (12
funciones × 6 algoritmos). H0 = "los 6 algoritmos tienen el mismo rango
promedio". Dos columnas dicen lo mismo de dos formas:

- `rechaza_h0` (booleano): `True` si `p_valor < 0.05`.
- `conclusion` (texto): la misma lectura en palabras — si se rechaza H0
  hay evidencia de que **al menos un algoritmo difiere** de los otros
  (no dice todavía cuál — para eso está el post-hoc de Shaffer, más
  abajo); si NO se rechaza, no hay evidencia suficiente de que los 6
  algoritmos se comporten distinto en esa configuración.

## `tablas/shaffer_d*.csv` y `shaffer_pct_significativo.csv` — setup §7

Las 15 comparaciones por pares entre los 6 algoritmos, con el
ajuste de Shaffer (controla mejor el error tipo I que comparar cada par
sin ajustar). `significativo=True` (p_ajustado < 0.05) → esos dos
algoritmos tienen desempeño distinguible en esa configuración.
`shaffer_pct_significativo.csv` resume qué % de las comparaciones donde
participa el propuesto salieron significativas.

## `tablas/wilcoxon_bew.csv` — setup §7 (extendido)

Test de Wilcoxon de rangos con signo, **emparejado por semilla**, por
función — el propuesto contra cada uno de los otros 5 algoritmos
(setup §7), más pso vs de (agregada más allá del setup original para
caracterizar también la diferencia entre los dos algoritmos base). El veredicto (`better`/`equal`/`worse`) se lee desde
la perspectiva de `algoritmo_a`: contando sobre las 12 funciones, cuántas
veces `algoritmo_a` fue significativamente mejor, no distinguible, o
significativamente peor que `algoritmo_b`.

## `tablas/desviacion_rangos.csv` — setup §7

Desviación estándar de los 6 `rango_promedio` de esa configuración —
qué tan dispersos quedaron los algoritmos entre sí (0 = los 6 empatados
en promedio; más alto = más separados).

## `figuras/cd_d*.png` — Diagramas de Diferencia Crítica (Nemenyi)

Un diagrama por configuración. Eje horizontal = rango promedio (1 a 6).
Cada algoritmo marcado en su posición. Una barra horizontal conecta a
los algoritmos cuya diferencia de rango es menor que la Diferencia
Crítica (CD, indicada en el título) — esos algoritmos NO son
distinguibles estadísticamente entre sí según Nemenyi (más conservador
que Shaffer: puede no marcar como distinguible un par que Shaffer sí
marcó significativo).

## `figuras/barras_rangos_d{10,20}.png`

Rango promedio de los 6 algoritmos a través de los 4 valores de MaxFES,
uno por dimensionalidad — para ver de un vistazo si el propuesto mejora,
empeora o se mantiene según el presupuesto de evaluaciones.

## `figuras/convergencia_d*.png` y `figuras/convergencia/` — gráficos de convergencia

Mejor error visto hasta cada momento (best-so-far, f − f(x*)) contra las
evaluaciones consumidas (FES), ambos ejes en escala log, sacado de la
traza `error_checkpoints` de cada corrida (120 puntos: 20 logarítmicos
entre 10⁻⁴ y 10⁻² de MaxFES y uno por cada 1 % de MaxFES). Errores
< 1e-8 se dibujan en 1e-8.

- `convergencia_d{10,20}_fes*.png`: un panel por configuración con las 12
  funciones. Línea = mediana de las 51 semillas; banda = rango
  intercuartil (percentiles 25-75). El propuesto va en rojo y más grueso.
- `convergencia/d*_fes*/fNN.png`: lo mismo, una figura por función.
- `convergencia/d*_fes*/semillas/fNN_sSS.png` (solo con
  `--convergencia-por-semilla`): la curva de cada algoritmo en una semilla.
- Con `analisis/convergencia.py --magnitud fitness`, las mismas figuras
  en valor objetivo f (= f(x*) + error), con sufijo `_fitness`.
- `tablas/convergencia_medianas.csv`: las curvas agregadas en números
  (q1, mediana y q3 del error, y fitness mediano), por (dim × MaxFES ×
  función × algoritmo × checkpoint), para re-graficarlas fuera de Python.

## `tablas/suplementario_media_mediana_std.csv`

Media, mediana y desviación estándar de las 51 corridas, por (algoritmo
× dim × MaxFES × función) — material suplementario obligatorio (§8),
más fino que el error medio ya usado en los rankings.

## `tablas/diagnostico_*.csv` — insumos para la discusión §9

- `diagnostico_d1d_activacion.csv`: % de corridas donde el middleware
  nunca llegó a activar la Fase 2 (`fes_primera_activacion` es NaN), por
  MaxFES. Originalmente cuantificaba la D-1d (con el extinto tier
  MaxFES=5×10³ el middleware casi no alcanzaba a actuar) — ese tier se
  retiró del protocolo (ver D-6, `configuracion_experimental.md`); con
  los tiers actuales (≥5×10⁴) sigue siendo el diagnóstico relevante para
  ver si el régimen persiste en el tier más chico vigente.
- `diagnostico_canales_a_b.csv`: uso promedio de Canal A vs Canal B por
  MaxFES.
- `diagnostico_r2_fase2.csv`: calidad del modelo subrogado (R²) por
  función.
