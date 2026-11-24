# Análisis BBOB: cómo leer cada tabla y figura

Parrilla: 12 algoritmos × 24 funciones × D ∈ {2, 3, 5, 10, 20, 40} × 15
instancias = 25,920 corridas, presupuesto **10000·D**
evaluaciones (opción A de `README_BBOB.md`). Generado con
`python analisis/bbob.py --resultados <carpeta>`.

**Error:** `f − f(x*)` al agotar el presupuesto (best-so-far). Un error bajo
`1e-8` cuenta como 0 (convención CEC, y objetivo final de COCO).
**Repeticiones:** en COCO las 15 instancias reemplazan a las semillas: son
versiones desplazadas y rotadas de la misma función, una corrida por
instancia.

Todo es **por dimensión**: no se promedia entre dimensiones (igual que en
CEC 2022 no se promedia entre configuraciones dim × MaxFES).

## Tablas (`tablas/`)

| Archivo | Contenido |
|---|---|
| `rangos_promedio.csv` | Algoritmo × D: rango promedio sobre las 24 funciones (1 = mejor, k = 12 = peor). Es la tabla principal. |
| `rangos_dDD.csv` / `error_medio_dDD.csv` | Algoritmo × función: rango y error medio de las 15 instancias, en la dimensión DD. El rango se asigna por función sobre el error medio, con empate si la diferencia es < 1e-8 (rango promedio de las posiciones empatadas). |
| `rangos_por_grupo.csv` | Rango promedio dentro de cada grupo de funciones de COCO: separables (f1–f5), condicionamiento moderado (f6–f9), condicionamiento alto y unimodales (f10–f14), multimodales con estructura global adecuada (f15–f19) y débil (f20–f24). |
| `friedman.csv` | Friedman por dimensión (k algoritmos, N = 24 funciones). `rechaza_h0` = hay diferencia significativa (α = 0.05). `cd_nemenyi` = diferencia crítica de los diagramas CD. |
| `wilcoxon_bew.csv` / `wilcoxon_resumen.csv` | Wilcoxon de rangos con signo, propuesto contra cada algoritmo (y PSO contra DE), por (dim, función), emparejado por instancia (15 pares). better / equal / worse desde el primer algoritmo: `equal` si p ≥ 0.05; si no, gana el de menor error medio. Se usa el error con umbral 1e-8. El resumen cuenta de 24 funciones por dimensión. |
| `exito.csv` | Algoritmo × D: fracción de las 360 corridas (24 × 15) que llegan a error < 1e-8. |
| `exito_por_funcion.csv` | Instancias resueltas (de 15) por dim, función y algoritmo. |
| `suplementario.csv` | Media, mediana, desviación, mínimo, P25, P75 y máximo del error final (sin umbral) por dim, función y algoritmo. |
| `middleware_diagnostico.csv` | Por dimensión, media de 360 corridas del propuesto: activaciones de Fase 2 y 3, Canal A, Canal B, abortos por la barrera de Wasserstein, instancias inyectadas, fracción del presupuesto en la primera activación, corridas sin ninguna activación, y mediana y % negativo del R² del modelo subrogado (R² < 0 bloquea la transferencia; la media no se reporta porque unos pocos R² muy negativos la distorsionan). |

## Figuras (`figuras/`)

| Archivo | Cómo leerla |
|---|---|
| `cd_dDD.png` | Diagrama de diferencia crítica (Nemenyi): cada algoritmo en su rango promedio. Una barra une algoritmos cuya diferencia de rango es menor que la CD, que no se distinguen estadísticamente. |
| `mapa_rangos_dDD.png` | Rango de cada algoritmo (filas, ordenadas por rango promedio) en cada función (columnas). Verde = mejor. Las líneas blancas separan los 5 grupos de funciones. |
| `rangos_contra_dim.png` | Rango promedio de cada algoritmo en cada D (eje invertido: arriba = mejor). Muestra cómo escala cada algoritmo con la dimensión. |
| `exito_contra_dim.png` | % de corridas resueltas a 1e-8 en cada D. |
| `boxplot_dDD.png` | Panel de las 24 funciones: distribución del error final de las 15 instancias por algoritmo, en escala log con piso 1e-8. Colores por grupo de algoritmos. |
| `convergencia_dDD.png` | Panel de las 24 funciones: mediana de las 15 instancias del error best-so-far contra las evaluaciones consumidas (ejes log). |
| `cocopp/ecdf_dDD_<grupo>.png` | Si se generó con `--ppdata`: ECDF agregadas de `cocopp` (grupo `todas` = f1–f24, o uno de los 5 grupos). La curva *best 2009* es la envolvente de los mejores algoritmos de BBOB 2009 (archivo de datos de COCO). **El presupuesto real es 10⁴·D (x = log₁₀ = 4, la cruz de cada curva): a la derecha, `cocopp` dibuja reinicios simulados, no evaluaciones hechas.** Fracción de pares (función, objetivo de precisión 10² … 10⁻⁸) alcanzados contra evaluaciones / D, sobre todas las funciones o un grupo, por dimensión. Más arriba y más a la izquierda = mejor. Es la figura estándar con que se publica en BBOB. |

## Advertencias

- Los 12 algoritmos usan la **configuración congelada de CEC 2022**, sin
  re-ajustar para BBOB: BBOB mide la generalización de esa configuración.
- El presupuesto 10⁴·D es chico en D = 2–5 (2·10⁴ a 5·10⁴ evaluaciones).
  El middleware gasta el 15 % en warm-up y ventanas del 10 % por ciclo,
  así que en esas dimensiones alcanza a actuar pocas veces
  (`middleware_diagnostico.csv`).
- Los rangos de BBOB y de CEC 2022 no se comparan entre sí (otro
  protocolo, otras funciones): cada benchmark tiene su propio ranking.
