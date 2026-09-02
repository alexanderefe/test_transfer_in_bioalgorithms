# Setup experimental — adaptación a CEC 2022

> Derivado del documento *"Setup Experimental para Evaluación de Nueva
> Metaheurística"* (10 secciones). Ese documento describe un protocolo
> sobre **72 problemas** (CEC 2014 + 2017 + 2022). Por decisión de alcance
> de esta tesis, **la evaluación usa únicamente CEC 2022**. Este archivo
> fija cómo se instancia cada sección del protocolo con ese recorte.
>
> Configuración de parámetros del algoritmo propuesto: ver
> [`configuracion_experimental.md`](configuracion_experimental.md).
> Última actualización: 2026-09-01.

---

## 1. Problemas de prueba

- **Benchmark:** CEC 2022, 12 funciones (`opfunu.cec_based.cec2022`),
  vía `problems/cec2022_wrapper.py`.
  - F1: unimodal · F2–F4: multimodales básicas · F5–F7: híbridas ·
    F8–F12: composición.
  - Rango de búsqueda `[−100, 100]^D`.
  - Óptimo global `f(x*)` conocido (tabla `OPTIMO_GLOBAL_CEC2022`:
    F1→300 … F12→2700). El **error** reportado es `f_obtenido − f(x*)`
    (≥ 0, 0 en el óptimo) — equivale a "restar el óptimo conocido" del PDF.

- **Conjuntos por dimensionalidad** (reemplazan a `CEC_72_10` / `CEC_72_50`):

  | Conjunto | Contenido |
  |---|---|
  | `CEC2022_10` | las 12 funciones en **D = 10** |
  | `CEC2022_20` | las 12 funciones en **D = 20** (CEC 2022 no define dimensiones mayores) |

- **No** hay desagregación por benchmark individual (solo hay uno).

## 2. Criterio de parada: MaxFES

- Único criterio de parada: evaluaciones de la función objetivo,
  **agregadas** sobre los dos algoritmos base (contador `problema.fes`).
- Cuatro valores: **5×10³, 5×10⁴, 5×10⁵, 5×10⁶**.
- Corridas independientes: cada valor de MaxFES arranca desde cero con su
  propia semilla; **no** son continuación unas de otras.
- **8 configuraciones experimentales** = 2 dimensionalidades × 4 MaxFES.
- Implementado: `HiloAlgoritmo._presupuesto_agotado` (no arranca una
  generación que no cabe; sobrepaso ≤ `n_individuos` por hilo).

## 3. Algoritmos competidores

El algoritmo propuesto **no** es una variante de Differential Evolution
(es un middleware sobre PSO + DE heterogéneos), por lo que aplica el
criterio de "metaheurística distinta": **≥ 20–25 algoritmos diversos**
cubriendo evolutivos, swarm y basados en distribución, incluyendo
ganadores recientes de competiciones CEC.

- **Lista candidata** (a confirmar según disponibilidad de implementación
  reproducible en Python — ver nota de riesgo):
  - *DE y variantes:* DE/rand/1/bin, jDE, SADE, JADE, SHADE, L-SHADE, jSO.
  - *Swarm:* PSO (canónico), CLPSO/HCLPSO, GWO, WOA, ABC.
  - *Evolutivos / distribución:* CMA-ES, GA, GSK, AGSK, APGSK-IMODE.
  - *Ganadores CEC recientes:* EA4eig, UMOEAII, LSHADE-cnEpSin,
    ELSHADE-SPACMA, NL-SHADE-RSP, MadDE.
- Debe incluir clásicos consolidados **y** propuestas recientes; no
  seleccionar solo competidores débiles.
- **RIESGO ABIERTO:** varios ganadores CEC (EA4eig, APGSK-IMODE, UMOEAII)
  solo tienen código MATLAB oficial. Decisión pendiente: portar los 3–4
  más relevantes, usar una librería Python (mealpy / niapy) para el resto,
  o reducir la lista documentando la limitación. Registrar en
  `configuracion_experimental.md` cuando se resuelva.

## 4. Configuración de parámetros

- **Competidores:** valores recomendados por sus autores originales, sin
  re-tuning. Tamaños de población documentados en una tabla del paper.
- **Algoritmo propuesto:** una única configuración para las 8
  configuraciones experimentales, sin ajuste por MaxFES ni por problema.
  Congelada en [`configuracion_experimental.md`](configuracion_experimental.md).

## 5. Protocolo de ejecución

- **51 corridas independientes** por (función × dimensionalidad × MaxFES).
- Semillas: **lista fija de 51 valores**, versionada
  (`experimentos/semillas.json` — pendiente de generar), la misma para
  todos los algoritmos y todas las configuraciones.
- Por corrida se registra el valor objetivo de la mejor solución al
  alcanzar MaxFES. En el código: `resumen.mejor_fitness_sistema` y
  `resumen.error_sistema`.
- **Métrica de comparación:** media de las 51 corridas del error
  (`mean( f_obtenido − f(x*) )`).
- Volumen: 12 × 2 × 4 × 51 = **4 896 corridas del algoritmo propuesto**
  (más 51× por cada competidor). El coste de las corridas a MaxFES = 5×10⁶
  es el principal cuello de botella → el harness debe soportar
  paralelismo por proceso y checkpoint/reanudación.

## 6. Construcción de rankings

- Por función individual: rankear los algoritmos de 1 (mejor) a N (peor)
  según el error medio de sus 51 corridas.
- **Empates:** si dos algoritmos difieren en `< 10⁻⁸` en el error medio,
  reciben el mismo rango (criterio CEC 2017).
- Rango promedio de cada algoritmo sobre las 12 funciones del conjunto.
- **8 rankings** (uno por cada `CEC2022_{10,20} × MaxFES`). Cada uno se
  discute de forma independiente: **no** se promedian rankings entre
  valores de MaxFES ni se elige uno como "principal".

## 7. Análisis estadístico (α = 0.05)

- **Friedman + post-hoc de Shaffer** — comparaciones múltiples por pares
  entre todos los algoritmos, **por separado para cada (dimensionalidad ×
  MaxFES)** (8 análisis). Se reporta el % de competidores de los que el
  algoritmo propuesto difiere significativamente.
- **Wilcoxon de rangos con signo** — propuesto vs cada competidor
  individual, formato **better / equal / worse** (en cuántas de las 12
  funciones el propuesto es estad. superior / equivalente / inferior),
  por cada (dimensionalidad × MaxFES).
- **Desviación estándar de los rangos promedio** entre algoritmos, por
  configuración (medida de diversidad de desempeño).

## 8. Presentación de resultados

Mínimo:
- Tablas de rangos promedio por algoritmo, separadas por dimensionalidad y
  MaxFES (8 tablas, o 2 tablas de 12 columnas de MaxFES).
- Tabla resumen con la posición final de cada algoritmo en cada uno de los
  8 rankings.
- Gráficos de barras comparando rangos a través de los 4 valores de MaxFES,
  uno por dimensionalidad.
- Tabla Wilcoxon better/equal/worse del propuesto vs cada competidor, por
  MaxFES y dimensionalidad.
- Tabla de % de diferencias significativas según Friedman, por configuración.
- Material suplementario: media, mediana y desviación estándar de las 51
  corridas para cada (función × dimensionalidad × MaxFES × algoritmo).

## 9. Discusión obligatoria

- En qué valores de MaxFES el propuesto destaca / es solo competitivo /
  es inferior.
- Cómo varía su ranking entre `CEC2022_10` y `CEC2022_20`.
- Operadores/componentes beneficiosos con cierto MaxFES pero
  contraproducentes o **inertes** con otros. En particular:
  **con MaxFES = 5×10³ el middleware no alcanza a activarse** (decisión
  D-1d en `configuracion_experimental.md`) → en ese régimen el "algoritmo
  propuesto" ≈ PSO + DE en paralelo sin transferencia. Debe discutirse
  explícitamente.
- Robustez del propuesto: varianza de su posición en los 8 rankings frente
  a la de los competidores.

## 10. Reproducibilidad

Se entregará:
- Código fuente del algoritmo propuesto (este repositorio).
- `experimentos/semillas.json` — las 51 semillas.
- Versiones exactas de los códigos de los competidores (commit / release +
  parámetros) en una tabla.
- Resultados crudos de cada corrida (`resultados/`), en material
  suplementario.
- `docs/configuracion_experimental.md` con la configuración única y el
  registro de decisiones.

---

## Estructura de trabajo pendiente

| Módulo | Contenido | Estado |
|---|---|---|
| `experimentos/semillas.json` | 51 semillas fijas | ⏳ |
| `experimentos/corrida.py` | una corrida: `(func, dim, max_fes, semilla) → métricas` | ⏳ |
| `experimentos/grid.py` | recorre las 4 896 corridas del propuesto; checkpoint + paralelismo | ⏳ |
| `experimentos/competidores/` | wrappers de los 20–25 competidores | ⏳ (bloqueado por §3) |
| `analisis/rankings.py` | rangos por función, promedio, empates 10⁻⁸ | ⏳ |
| `analisis/estadistica.py` | Friedman + Shaffer, Wilcoxon b/e/w, std de rangos | ⏳ |
| `analisis/figuras.py` | barras de rangos × MaxFES; tablas resumen | ⏳ |
| `resultados/` | CSV crudo (una fila por corrida) | ⏳ |
