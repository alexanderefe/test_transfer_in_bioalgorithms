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

## 3. Algoritmos competidores — DESVIACIÓN DEL SETUP ORIGINAL (decisión D-4)

El PDF exige, para una "metaheurística distinta" (no variante de DE),
**≥ 20–25 algoritmos diversos**. Se evaluó un roster candidato de 27
(archivado en [`competidores.md`](competidores.md), superseded) y se
**descartó**: implementarlo y correrlo disparaba el cómputo a semanas, y
varios de los ganadores CEC que el PDF nombra (EA4eig, APGSK-IMODE, UMOEAII)
solo tienen código MATLAB.

**Decisión (D-4, `configuracion_experimental.md`): los únicos competidores
son PSO y DE corriendo SOLOS** (sin middleware, sin transferencia), con la
misma configuración congelada (§A.1) que usan como algoritmos base del
propuesto. Motivo: el experimento se programó desde el inicio alrededor de
estos dos algoritmos, y esa comparación —¿el middleware mejora sobre los
algoritmos base sin transferencia?— es la pregunta central de la tesis.

- **Impacto en cómputo:** sube el total de ~53 h a **~183 h (≈7.6 días) en
  5 workers** — ver la tabla de D-4. Sigue siendo muchísimo menor que el
  roster externo (semanas–meses), pero **no es gratis**: hay que tenerlo
  presente al planificar.
- **Implementado:** `experimentos/corrida.py` (`_RUNNERS["pso"]`,
  `_RUNNERS["de"]`) y `grid.py --algoritmo middleware,pso,de` (default).
- Las secciones §6–§9 (rankings, Friedman/Nemenyi, Wilcoxon, discusión) se
  aplican igual, con **3 algoritmos por ranking** (propuesto, PSO, DE) en
  vez de ~23. Los rankings y CD diagrams siguen siendo válidos con N=3,
  aunque menos informativos que con un panel amplio — se debe reconocer
  esta limitación en la redacción.

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
  `resumen.error_sistema`. El esquema completo de captura está en **§11**.
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

Todos los análisis se hacen **por separado para cada (dimensionalidad ×
MaxFES)** → 8 repeticiones de cada uno.

- **Friedman + post-hoc de Shaffer** — comparaciones múltiples por pares
  entre todos los algoritmos. Friedman contrasta H₀ = "todos los
  algoritmos tienen el mismo rango promedio" sobre la matriz de rangos
  (12 funciones × N algoritmos). Si se rechaza, el post-hoc de Shaffer da
  los *p*-valores ajustados por pares. Se reporta el **% de competidores
  de los que el algoritmo propuesto difiere significativamente**.

- **Diagrama de Diferencia Crítica (post-hoc de Nemenyi)** — visualización
  compacta del resultado de Friedman (Demšar, 2006). Dos algoritmos tienen
  rendimiento significativamente distinto si la diferencia de sus rangos
  promedio supera la **Diferencia Crítica**:

  ```
  CD = q_α · sqrt( k · (k + 1) / (6 · N) )
  ```

  con `k` = nº de algoritmos, `N` = nº de funciones del conjunto (12), y
  `q_α` = valor crítico del rango studentizado (para α = 0.05) dividido
  por √2. El **diagrama CD** dibuja el eje de rangos promedio y une con
  una barra a los grupos de algoritmos cuya diferencia de rango es `< CD`
  (estadísticamente indistinguibles). **Un diagrama por cada
  (dimensionalidad × MaxFES)** → 8 diagramas.

  Nota metodológica: Nemenyi es más conservador que Shaffer (no explota la
  lógica de las hipótesis). Se reportan ambos — Shaffer para la tabla de
  porcentajes (§8), Nemenyi para la figura CD.

- **Wilcoxon de rangos con signo** — propuesto vs cada competidor
  individual. Para **cada una de las 12 funciones**: test pareado (por
  semilla) entre las 51 corridas del propuesto y las 51 del competidor →
  veredicto *better / equal / worse* según el signo del efecto y si
  *p* < 0.05. Se reporta el conteo **better/equal/worse** sobre las 12
  funciones, por cada (dimensionalidad × MaxFES).

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
- **8 diagramas de Diferencia Crítica (Nemenyi)**, uno por
  (dimensionalidad × MaxFES).
- Tabla Wilcoxon better/equal/worse del propuesto vs cada competidor, por
  MaxFES y dimensionalidad.
- Tabla de % de diferencias significativas según Friedman/Shaffer, por
  configuración.
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
- `experimentos/semillas.json` — las 51 semillas. **La misma lista se
  reutiliza para todo algoritmo y toda configuración** (dim × MaxFES):
  cada corrida siembra su RNG desde cero con ese valor (independencia, no
  continuación entre MaxFES) y a la vez permite el emparejamiento por
  semilla del test de Wilcoxon.
- Versiones exactas de los códigos de los competidores (commit / release +
  parámetros) en una tabla.
- Resultados crudos de cada corrida (`resultados/`), en material
  suplementario.
- 8 diagramas de Diferencia Crítica (Nemenyi).
- `docs/configuracion_experimental.md` con la configuración única y el
  registro de decisiones.

---

## 11. Datos a capturar por el harness (derivado de §6–§8)

La unidad atómica es **una corrida**. Todo el análisis (rankings, Friedman/
Shaffer, diagrama CD de Nemenyi, Wilcoxon better/equal/worse, desviación de
rangos, material suplementario) debe poder calcularse **sin re-ejecutar
nada**. Por eso se guarda el **error individual de cada corrida**, no
agregados.

### 11.1 Qué necesita cada análisis

| Análisis | Dato mínimo requerido |
|---|---|
| Rankings + empates 10⁻⁸ (§6) | error **medio** de las 51 corridas, por (algoritmo × función × dim × MaxFES) |
| Friedman + Shaffer (§7) | matriz de rangos = error medio por (algoritmo × función), por configuración |
| Diagrama CD / Nemenyi (§7) | rangos promedio por algoritmo + `k` (nº algoritmos) + `N` = 12 |
| Wilcoxon better/equal/worse (§7) | los **51 errores individuales** del propuesto y del competidor, **emparejados por semilla**, por (función × dim × MaxFES) |
| Desviación de rangos promedio (§7) | vector de rangos promedio por configuración |
| Material suplementario (§8) | media, mediana y **desv. estándar** de las 51 → requiere los 51 valores |
| Barras de rangos × MaxFES (§8) | rango promedio por (algoritmo × dim × MaxFES) |
| Robustez del propuesto (§9) | posición del propuesto en los 8 rankings |
| Discusión operadores × MaxFES (§9) | diagnósticos del middleware por corrida (11.3) |

**Conclusión:** guardar el error individual por corrida cubre todo lo
demás; los agregados se calculan aguas abajo en `analisis/`.

### 11.2 Tabla `resultados/corridas.parquet` — **todas** las corridas (propuesto + competidores)

Una fila por corrida. 12 × 2 × 4 × 51 ≈ 4 896 filas por algoritmo.
**Implementación:** cada corrida escribe su propio Parquet en
`resultados/corridas/<run_id>.parquet` (robusto a interrupción, reanudable,
sin contención entre workers); `experimentos/consolidar.py` los une en
`resultados/corridas.parquet`. Esquema en `experimentos/esquema.py`
(`COLUMNAS_CORRIDA`).

| Columna | Tipo | Para qué |
|---|---|---|
| `run_id` | str | `<algoritmo>__f<NN>__d<NN>__fes<N>__s<NN>` — dedupe y reanudación |
| `algoritmo` | str | id corto (p. ej. `middleware`, `LSHADE`, `PSO`, `EA4eig`) |
| `familia` | str | `propuesto` \| `DE` \| `swarm` \| `evolutivo` \| `distribucion` — para agrupar en figuras |
| `version` | str | commit / release / versión de librería + hash de la config usada (§10) |
| `benchmark` | str | `CEC2022` (constante, explícito) |
| `funcion` | int | 1 … 12 |
| `dim` | int | 10 \| 20 |
| `conjunto` | str | `CEC2022_10` \| `CEC2022_20` |
| `max_fes` | int | 5000 \| 50000 \| 500000 \| 5000000 |
| `idx_semilla` | int | 0 … 50 (posición en `semillas.json`) |
| `semilla` | int | valor de la semilla |
| `f_mejor` | float | `f_opt + error` |
| **`error`** | float | mejor error usando **≤ MaxFES** evaluaciones = `error_checkpoints[-1]` (checkpoint de la fracción 1.0). Se toma del checkpoint, no del resumen, para no contar las ≤ pop evaluaciones de sobrepaso de la última generación. **La columna que consume todo el análisis.** |
| `f_opt` | float | f(x*) conocido de la función |
| `error_checkpoints` | list[float] (14) | best-so-far error en FES = {0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}·MaxFES. Se captura vía hook en `ProblemaCEC2022` (sirve igual para competidores). Monótona no creciente. |
| `fes_consumidas` | int | evaluaciones realmente gastadas (`max_fes` ± pop por el sobrepaso/subpaso de la última generación); detecta corridas corruptas |
| `estado` | str | `ok` \| `fallo` \| `timeout` |
| `mensaje_error` | str | vacío si `ok` |
| `tiempo_seg` | float | wall-clock de la corrida |
| `host` | str | máquina donde se corrió |
| `timestamp_utc` | str | ISO-8601 |
| `ruta_log` | str | ruta al artefacto detallado de esa corrida (auditoría sin re-correr) |

### 11.3 Tabla `resultados/corridas_propuesto.parquet` — **solo** el algoritmo propuesto (diagnósticos)

Misma identidad (`run_id`, `funcion`, `dim`, `max_fes`, `idx_semilla`) más
los campos del middleware. Alimenta la discusión §9 y las validaciones
diferidas D-1a / D-1b / D-1d.

| Columna | Origen | Para qué |
|---|---|---|
| `f_mejor_a`, `f_mejor_b` | `ResumenEjecucion.fitness_final_algoritmo_{a,b}` | desempeño de cada algoritmo base por separado |
| `error_a`, `error_b` | `ResumenEjecucion.error_algoritmo_{a,b}` | idem, ya restado el óptimo |
| `fes_a`, `fes_b` | `fes_consumidas_{a,b}` | reparto real del presupuesto (**D-1b**) |
| `n_generaciones_a`, `n_generaciones_b` | orquestador (añadir campo) | contexto |
| `n_activaciones_fase2` | `ResumenEjecucion` | cuánto actuó el middleware |
| `n_activaciones_fase3` | `ResumenEjecucion` | idem |
| `n_reentrenamientos_cambio_fuente` | `ResumenEjecucion` | inestabilidad de roles |
| `n_canal_a`, `n_canal_b` | `n_transferencias_canal_{a,b}` | qué canal domina por MaxFES (**§9**) |
| `n_abortos_wasserstein` | `n_transferencias_abortadas_wasserstein` | frecuencia de bloqueo por barrera |
| `n_instancias_inyectadas` | `n_instancias_inyectadas_total` | magnitud de Canal B |
| `fes_primera_activacion` | orquestador (añadir campo) | **NaN si el middleware nunca actúa** → clave para D-1d (MaxFES = 5×10³) |
| `frac_primera_activacion` | `fes_primera_activacion / max_fes` | comparabilidad entre MaxFES |
| `r2_fase2` | `log_fase2[*].r2_modelo_subrogado` | calidad del subrogado (**D-1a**, discusión) |
| `tiempo_pausado_middleware_seg` | `ResumenEjecucion` | sobrecoste del middleware |
| `ruta_log_json` | harness | log dual completo de la corrida (Fases 1/2/3) |

**Ajustes menores al orquestador** (no bloquean el harness, se pueden
derivar temporalmente del log): añadir a `ResumenEjecucion` los campos
`fes_primera_activacion` y `n_generaciones_a` / `n_generaciones_b`.

---

## Estructura de trabajo pendiente

| Módulo | Contenido | Estado |
|---|---|---|
| `experimentos/semillas.json` + `generar_semillas.py` + `semillas.py` | 51 semillas fijas (mismas para todo algoritmo y config) | ✅ |
| `experimentos/esquema.py` | esquemas §11.2 / §11.3, `run_id()`, builders de fila, `df_tipado()` | ✅ |
| `experimentos/corrida.py` | `correr_corrida(...) → (fila §11.2, fila §11.3\|None)`; `middleware`, `pso`, `de` (D-4) | ✅ |
| `experimentos/grid.py` | `--algoritmo middleware,pso,de` (default); per-run Parquet; reanudación; `ProcessPoolExecutor`; `--dry-run` con estimación por algoritmo | ✅ |
| `experimentos/consolidar.py` | `resultados/corridas/*.parquet` → `resultados/corridas.parquet` | ✅ |
| `problems/cec2022_wrapper.py` | hook de checkpoints (`configurar_checkpoints`, `error_checkpoints`) | ✅ |
| `middleware/orquestador.py` | campos `fes_primera_activacion`, `n_generaciones_{a,b}`; modo `silencioso` | ✅ |
| ~~`experimentos/competidores/`~~ | descartado (D-4): sin roster externo, solo PSO/DE solos | ❌ superseded |
| `analisis/rankings.py` | error medio → rangos por función, promedio, empates 10⁻⁸ (N=3: propuesto, PSO, DE) | ⏳ |
| `analisis/estadistica.py` | Friedman + Shaffer, Wilcoxon b/e/w, desv. de rangos | ⏳ |
| `analisis/cd_nemenyi.py` | rangos promedio + CD → 8 diagramas de Diferencia Crítica | ⏳ |
| `analisis/figuras.py` | barras de rangos × MaxFES; tablas resumen de posiciones | ⏳ |

### ✅ RESUELTO — reproducibilidad del algoritmo propuesto (D-2)

La misma (función, semilla) llegó a producir resultados **materialmente
distintos** entre corridas: F11/D10/MaxFES=20 000 dio errores 8.30 / 0.0003
/ 8.30 en tres ejecuciones idénticas, por el *threading* pseudo-paralelo
del orquestador (el reparto de FES entre PSO y DE dependía del scheduling
del SO).

**Resuelto (2026-09-04):** el threading se reemplazó por un planificador
cooperativo determinista (un solo hilo, round-robin A/B), y se cerraron dos
fuentes adicionales de no-determinismo que aparecieron al validar el
arreglo (RNG global de `torch` sin sembrar en `middleware/extraccion.py`,
y `ShapleySampler.rng` sin semilla en `fastshap_lib/utils.py`). Verificado
en `tests/test_determinismo_orquestador.py`: misma configuración, dos
corridas, resultado idéntico. Detalle completo en
`configuracion_experimental.md` (D-2). La parrilla ya no está bloqueada por
esto — ver D-4 para lo que sigue bloqueando el lanzamiento (selección y
sourcing de competidores, decisión sobre el presupuesto de cómputo).
