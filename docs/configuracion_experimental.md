# Configuración experimental — parámetros congelados y registro de decisiones

> **Propósito.** El setup experimental de la tesis (§4) exige **una única
> configuración de parámetros** para el algoritmo propuesto, sin ajuste por
> valor de MaxFES ni por problema. Este documento fija esa configuración y
> deja registro de las decisiones tomadas para congelarla, incluyendo las
> que quedan pendientes de una revisión más profunda.
>
> Última actualización: 2026-09-04.

---

## A. Configuración única del algoritmo propuesto

Todos los valores son fijos para las 8 configuraciones experimentales
(2 dimensionalidades × 4 valores de MaxFES) y para todos los problemas.

**Benchmark:** CEC 2022 únicamente — 12 funciones, D ∈ {10, 20}. No se
amplía a CEC 2014/2017 ni D=50. Protocolo completo en
[`setup_experimental.md`](setup_experimental.md).

### A.1 Algoritmos base (substrato del middleware)

| Parámetro | Valor | Fuente en código |
|---|---|---|
| PSO — tamaño de población | 30 | argumento de construcción |
| PSO — inercia inicial `w_max` | 0.9 | `bioalgorithms/pso.py` |
| PSO — inercia final `w_min` | 0.4 | `bioalgorithms/pso.py` |
| PSO — coef. cognitivo `c1` | 2.0 | `bioalgorithms/pso.py` |
| PSO — coef. social `c2` | 2.0 | `bioalgorithms/pso.py` |
| DE — tamaño de población | 30 | argumento de construcción |
| DE — factor de escala `F` | **0.2** | config experimental (el *default* del código es 0.5) |
| DE — prob. de cruce `CR` | **0.3** | config experimental (el *default* del código es 0.7) |
| Variante DE | DE/rand/1/bin | `bioalgorithms/de.py` |

> DE se configura deliberadamente conservador (`F`, `CR` bajos) para que
> presente estancamiento real y observable como algoritmo objetivo. No es
> una variante adaptativa (no SHADE / L-SHADE).

### A.2 Presupuesto y reloj

| Parámetro | Valor | Fuente |
|---|---|---|
| Criterio de parada | MaxFES agregado A+B | `problems/cec2022_wrapper.py` (`.fes`) |
| Valores de MaxFES | 5×10³, 5×10⁴, 5×10⁵, 5×10⁶ | setup §2 |
| Planificador | Cooperativo determinista (round-robin, un hilo) | `Orquestador.ejecutar` — D-2, **resuelto** |
| Sobrepaso tolerado | ≤ `n_individuos` por algoritmo | `Orquestador._puede_avanzar` |
| `fraccion_presupuesto` (cuota PSO para su inercia) | **0.5 exacto** (round-robin, poblaciones iguales) | `bioalgorithms/base.py` — D-1b, **resuelto** |
| `frecuencia_monitoreo_fes` (default) | `max(pop_A+pop_B, MaxFES // 500)` | `middleware/orquestador.py` |
| `max_epochs_fastshap` (producción) | 20 (D-3) | `middleware/extraccion.py` (`EPOCHS_EXPLAINER`) |

### A.3 Fase 1 — Detección (`middleware/deteccion.py`)

| Parámetro | Valor |
|---|---|
| Peso FIR `W_FIR` | 0.3 |
| Peso DWD `W_DWD` | 0.7 |
| Peso HDF `W_HDF` | 0.0 (inactivo, dominio continuo) |
| Ventana FIR | 20 generaciones del objetivo |
| `LAMBDA_FIR` (saturación) | 3.0 |
| Umbral de estancamiento (`S <`) | 0.2 |
| Warm-up (`FRACCION_ACTIVACION`) | 0.15 · MaxFES |

### A.4 Fase 2 — Extracción (`middleware/extraccion.py`)

| Parámetro | Valor |
|---|---|
| Ventana de historial de la fuente | 50 generaciones |
| Instancias de élite `TOP_K` | 5 |
| Pre-filtro por fitness real | percentil 20 % |
| Background del MarginalImputer | 30 muestras |
| Nº de árboles del subrogado XGBoost | 100 (`max_depth`=4, `lr`=0.1) |
| Épocas del explainer FastSHAP | **20** (D-3) |
| `num_samples` del explainer (train y validación) | **4** (D-3) |
| Batch / LR del explainer | 32 / 2×10⁻³ |
| Bloqueo por R² | R² < 0 (estrictamente) |

### A.5 Fase 3 — Transferencia (`middleware/transferencia.py`)

| Parámetro | Valor |
|---|---|
| RMP inicial | 0.05 |
| RMP tope | 0.6 |
| Factor de incremento del RMP | 0.5 · ΔS |
| Ventana de evaluación de Canal A | 10 % de MaxFES (evaluaciones agregadas) |
| Umbral ΔScore significativo | 0.02 |
| Cooldown post-ciclo | = ventana de Canal A (10 % de MaxFES) |
| Instancias reemplazadas en Canal B | = TOP_K (5) |

### A.6 Barrera de seguridad (`middleware/barrera_seguridad.py`)

| Parámetro | Valor |
|---|---|
| θ_W (Wasserstein) | 0.25 · diámetro máximo del espacio |
| Ancho de banda MMD | mediana de distancias **intra-grupo** |
| Centrado MMD | poblaciones centradas en su media antes del kernel RBF |
| Umbral MMD | 3 × P95 del ruido de muestreo de la fuente (precalculado en Fase 2) |
| Submuestras para el ruido de referencia | 50 |

### A.7 Protocolo

| Parámetro | Valor |
|---|---|
| Corridas independientes por (problema × dim × MaxFES) | 51 |
| Semillas | lista fija, versionada (pendiente de generar con el harness) |
| Métrica de comparación | media de las 51 corridas de `f_obtenido − f(x*)` |
| Corridas con MaxFES mayor | **no** son continuación: cada una desde cero |

---

## B. Registro de decisiones (punto 1 del plan de avance)

Formato: **decisión tomada ahora** (para no bloquear el setup de pruebas) +
**qué queda pendiente de revisar a fondo**.

### D-1a · Calibración de los pesos del Score (`W_FIR` / `W_DWD`) — RESUELTO

- **Estado previo (2026-09-01):** 0.3 / 0.7, marcados "empíricos
  provisionales", con un análisis de sensibilidad empírico planeado (3
  configuraciones — `0.2/0.8`, `0.3/0.7`, `0.5/0.5` — sobre un subconjunto
  de 12 funciones CEC 2022, D=10, MaxFES=5×10⁴, ~10 semillas) para
  confirmar que el error final del sistema no dependiera fuertemente de
  la elección.
- **Decisión final (2026-09-04): se conservan `W_FIR=0.3` / `W_DWD=0.7`
  por justificación conceptual, sin ejecutar el barrido empírico.**
  Razón: la diversidad poblacional (DWD) es una señal más informativa de
  estancamiento real que la tasa de mejora de fitness (FIR) — un
  algoritmo puede dejar de mejorar su fitness momentáneamente sin haber
  perdido diversidad (falso positivo de estancamiento si FIR pesara más),
  mientras que la pérdida de diversidad poblacional es un indicador más
  directo y difícil de revertir de que el algoritmo quedó atrapado en un
  óptimo local. Ponderar DWD por encima de FIR (0.7 vs 0.3) refleja esa
  jerarquía de evidencia.
- Esto reemplaza el plan de validación empírica: no se implementó el
  parámetro `peso_fir`/`peso_dwd` en `Orquestador.__init__` (los tres
  llamados a `calcular_score()` en `middleware/orquestador.py` siguen
  usando los defaults del módulo, `middleware/deteccion.py:68-69`) ni se
  corrió el subconjunto de sensibilidad. Si en la redacción hace falta
  robustecer este punto, la justificación a citar es la de arriba
  (jerarquía conceptual DWD > FIR), no un resultado experimental.
- Ya no bloquea el lanzamiento de la parrilla — ver README "Pendiente".

### D-1b · Cuota de presupuesto de PSO (`fraccion_presupuesto`) — RESUELTO

- **Estado previo (2026-09-01):** con el threading no determinista, el
  reparto real medido en el par PSO(fuente)/DE(objetivo) sobre F11/D=20 era
  **~70/30 a favor de PSO** (70.7 %, 70.5 %, 73.5 % en tres corridas) —
  emergente del scheduling del SO (PSO, vectorizado, "ganaba" más turnos de
  CPU que el bucle por individuo de DE). Se había fijado
  `fraccion_presupuesto = 0.7` como aproximación empírica, marcada
  provisional porque el split variaba por corrida, función y dimensión.
- **Resuelto (2026-09-04, junto con D-2):** el planificador cooperativo
  determinista alterna una generación de A y una de B — con poblaciones
  iguales (30 y 30), eso reparte el presupuesto **exactamente 50/50, por
  construcción**, no por medición. `fraccion_presupuesto` vuelve a **0.5**,
  y esta vez es el valor **correcto**, no una aproximación: confirmado en
  `tests/test_determinismo_orquestador.py` (`fes_consumidas_a ==
  fes_consumidas_b`, diferencia 0 en la corrida de prueba).
  Aplicado en `bioalgorithms/base.py`, `pso.py`, `de.py`.

### D-1d · Comportamiento con `MaxFES = 5×10³`

- **Problema:** con ese presupuesto el middleware **no alcanza a activarse**.
  La condición de Fase 2 exige `len(hist_fuente) > 50` generaciones de la
  fuente; con ~166 evaluaciones totales / (30+30 por generación) ≈ 83
  generaciones repartidas, esa condición recién se cumple pasado el ~60 %
  de la corrida, y ya no hay margen para un ciclo completo
  (extracción + ventana Canal A + Canal B + cooldown).
- **Decisión (2026-09-01):** **no** se modifica el código. Con
  `MaxFES = 5×10³` el "algoritmo propuesto" se comporta, en la práctica,
  como **PSO + DE en paralelo sin transferencia**. Es un resultado legítimo
  y es exactamente lo que la discusión obligatoria del setup (§9) pide
  analizar: "operadores o componentes beneficiosos con cierto MaxFES pero
  contraproducentes o inertes con otros".
- **Pendiente:** en la redacción, discutir explícitamente este régimen.
  Alternativa a evaluar más adelante (solo si se decide que el middleware
  *debe* actuar con presupuestos chicos): regla adaptativa
  `VENTANA_HISTORIAL_FUENTE = clamp(k · generaciones_estimadas, mín, 50)` y
  warm-up análogo — regla fija única, pero exige re-validar los 2
  escenarios y cambia un valor hoy marcado "Definitiva".

### D-1e · Test rojo de la barrera de seguridad (`tests/test_barrera_seguridad.py`, CASO 3)

- **Síntoma:** el CASO 3 espera que instancias en la "esquina extrema del
  espacio" sean rechazadas por el filtro MMD; hoy las aprueba (5/5).
- **Diagnóstico:** **no es un bug de código.** El MMD del módulo está
  **centrado** por diseño (corrección documentada en
  `barrera_seguridad.py`): mide compatibilidad de **forma**
  (varianza / estructura), no de **ubicación** — la incompatibilidad de
  ubicación la cubre la barrera de Wasserstein por separado. Además el test
  usa 5 instancias **idénticas** (varianza cero), un caso que no ocurre en
  el pipeline real, donde la élite de FastSHAP siempre es distinta.
  El CASO 3 encodea una expectativa anterior a la corrección del centrado.
- **Decisión (2026-09-01):** **no se toca la barrera de seguridad ahora.**
  Prioridad: tener el setup de pruebas. El test queda como fallo conocido y
  documentado.
- **Pendiente:** dar una vuelta más profunda. Opciones sobre la mesa:
  (i) reescribir el CASO 3 para verificar rechazo por **forma** incompatible
  en vez de "esquina extrema"; (ii) añadir en `filtrar_instancias_por_mmd`
  una guarda explícita para grupos candidatos con varianza ≈ 0; (iii) ambas.
  Revisar más adelante, sin bloquear el harness experimental.

### D-3 · Costo de FastSHAP en la Fase 2

- **Motivación:** el harness estima ~7–8 días de cómputo para la parrilla del
  propuesto; el middleware (dominado por FastSHAP) es ~30 % de una corrida
  corta. Se descartó GPU (sin hardware; además el MLP es demasiado chico y el
  costo real es el `predict` de XGBoost en CPU con batches pequeños).
- **Decisión (2026-09-03):**
  - `EPOCHS_EXPLAINER` **50 → 20**. Medido: el ciclo de transferencia produce
    el mismo comportamiento (mismas activaciones de Fase 2/3, calidad de
    instancias y atribución Shapley intactas en `test_fase2_extraccion.py`).
  - `NUM_SAMPLES_EXPLAINER` **8 → 4** (train y validación del explainer).
    Afecta solo al explainer, no al R² del subrogado. Reduce a la mitad las
    llamadas a `xgb.predict` dentro del imputer.
  - **`n_estimators` del subrogado se mantiene en 100.** Se probó bajarlo a
    50 pero el R² del subrogado cayó a 0.65 en el escenario de validación
    (PSO/F11), por debajo del criterio de aceptación **R² > 0.7** del
    documento de tesis.
- **Efecto medido:** F11/D10, MaxFES=100 000 → wall-clock **55 s → 35 s
  (−36 %)**; el overhead del middleware en sí cae ~50 %.
- **Efecto sobre la parrilla completa: pequeño (~7 %).** El estimador
  calibrado (`experimentos/grid.py --dry-run`) da **~53 h en 5 workers**
  post-D3 vs **~57 h** pre-D3. La razón: la **función objetivo de opfunu es
  el ~90 % del tiempo** (el tier MaxFES=5×10⁶ solo es 87 % del total, y ahí
  el middleware es <10 %). D-3 acelera las corridas cortas y el ciclo de
  desarrollo/sensibilidad, pero **no es la palanca de la parrilla** — esa es
  el evaluador CEC compilado.
- **Nota:** D-1a (pesos del Score) se cerró por justificación conceptual,
  sin barrido empírico — no hay una corrida de sensibilidad donde
  confirmar que `20` epochs no degrada la calidad del middleware en otras
  funciones. Queda la opción de bajar más si, al revisar los resultados
  de la parrilla completa, el R²/atribución de Fase 2 lo permiten.

### D-2 · Reproducibilidad del algoritmo propuesto — RESUELTO

- **Hallazgo (2026-09-03):** la misma `(función, semilla)` producía
  resultados **materialmente distintos** entre corridas idénticas —
  F11/D10/MaxFES=20 000: errores 8.30 / 0.0003 / 8.30 en tres ejecuciones.
  Causa: el orquestador corría PSO y DE en hilos pseudo-paralelos
  ("Threading Solución A"); el reparto de FES entre ambos dependía del
  scheduling del SO, que cambiaba la trayectoria de inercia de PSO y el
  momento en que el middleware detectaba estancamiento.
- **Impacto:** violaba la reproducibilidad del setup §10 e inflaba la
  varianza de las 51 corridas.
- **Resuelto (2026-09-04).** Se encontraron y cerraron **tres fuentes de
  no-determinismo independientes**, no solo el threading:

  1. **Scheduling de hilos** (la causa original) → reemplazado por un
     **planificador cooperativo determinista** en `middleware/orquestador.py`:
     un solo bucle alterna una generación de A, una de B; el middleware se
     ejecuta de forma síncrona (sin locks, eventos ni `time.sleep` de espera
     activa). Se eliminó la clase `HiloAlgoritmo` y todo `threading.Event`.
     Con poblaciones iguales, el reparto pasa a ser exactamente 50/50 (D-1b).
  2. **RNG global de `torch` sin sembrar**: `middleware/extraccion.py`
     construía y entrenaba el explainer de FastSHAP (inicialización de
     pesos + muestreo interno de `fastshap.train()`) sin llamar nunca a
     `torch.manual_seed(...)`. Se añadió `torch.manual_seed(0)` (mismo
     criterio que los `np.random.default_rng(0)`/`(1)` ya fijos en ese
     archivo — una constante deliberada, no ligada a la semilla del
     experimento). Se probó también forzar `torch.set_num_threads(1)`
     (precaución contra no-determinismo de reducciones multi-hilo en
     BLAS/MKL) pero no hizo falta para el determinismo y no mostró mejora
     de tiempo medible al quitarlo — se dejó sin forzar.
  3. **`fastshap_lib/utils.py` — `ShapleySampler.rng =
     np.random.default_rng()` sin semilla** (arranca desde entropía del
     SO). Esta fue la causa real de la divergencia residual que sobrevivió
     a (1) y (2): el muestreo de coaliciones de Shapley en cada llamada a
     `sample()` durante el entrenamiento de FastSHAP. **Parche** (mismo
     criterio que el parche de compatibilidad PyTorch ≥2.x que ya llevaba
     este archivo vendorizado): `default_rng(0)`.

  También se fijó `n_jobs=1` en los `xgb.XGBRegressor(...)` del modelo
  subrogado (precaución: el histograma multi-hilo de XGBoost no garantiza
  el mismo orden de suma en punto flotante entre corridas; no fue la causa
  activa del bug, pero es un riesgo documentado del mismo tipo y el costo
  es despreciable en un dataset de ~1200 filas).

- **Verificación:** `tests/test_determinismo_orquestador.py` — misma
  configuración corrida dos veces, compara `error_sistema`,
  `fitness_final_algoritmo_a/b`, `fes_consumidas_*`,
  `n_activaciones_fase2/3`, `n_transferencias_canal_a/b`: **igualdad
  exacta** en las dos corridas, `fes_consumidas_a == fes_consumidas_b`.
- Cambia la decisión de diseño "Paralelismo" del README, de *Threading
  Solución A (pseudo-paralelo, GIL)* a *Planificador cooperativo
  determinista* — pasa a **Definitiva**.
- El límite de "reproducible" sigue siendo *dentro del mismo entorno*
  (SO, versión de numpy/BLAS/torch); ver la discusión de determinismo
  entre sistemas operativos en la conversación de diseño — diferencias de
  BLAS/últimos bits de punto flotante entre SO son un riesgo residual
  menor, no relevante al umbral de empate de 10⁻⁸ del setup.

### D-4 · Competidores: solo PSO y DE solos, sin roster externo

- **Contexto:** `competidores.md` había levantado un roster candidato de 27
  algoritmos externos (mealpy/pymoo/CMA-ES + ganadores CEC vía Octave) para
  cumplir el mínimo de 20–25 competidores del setup §3.
- **Decisión (2026-09-04):** **se descarta el roster externo.** Los únicos
  "competidores" son **PSO y DE corriendo SOLOS** (sin middleware), con la
  misma configuración única congelada (§A.1) que usan como algoritmos base
  del propuesto. Motivos:
  1. El experimento se programó desde el inicio alrededor de estos dos
     algoritmos (`bioalgorithms/pso.py`, `bioalgorithms/de.py`) — es la
     comparación que responde la pregunta central de la tesis: ¿el
     middleware mejora sobre los algoritmos base sin transferencia?
  2. El tiempo de cómputo ya es alto solo con el propuesto (~53 h en 5
     workers); el roster de 20–25 competidores externos lo llevaba a
     **semanas** (`competidores.md` §6).
  - `competidores.md` queda **archivado/superseded** — no se implementa.
- **Implementación:** `experimentos/corrida.py` añade los runners `"pso"` y
  `"de"` — el algoritmo corre solo, con el **100 % de MaxFES para sí mismo**
  (`fraccion_presupuesto=1.0`, no 0.5: ese valor es específico para cuando
  PSO comparte presupuesto con DE bajo el middleware). Sin threading, sin
  orquestador: un bucle secuencial que respeta el mismo guard de sobrepaso
  (`fes + n_individuos <= max_fes`). No generan fila §11.3 (no hay
  diagnósticos de middleware que registrar). `esquema.fila_corrida` se
  simplificó para depender solo de `problema` (agnóstico al algoritmo), no
  de `ResumenEjecucion`.
- **⚠ Impacto medido en el tiempo total — sube ~3.4×, no es gratis:**

  | | secuencial | 5 workers |
  |---|---|---|
  | Solo propuesto | 265 h | ~53 h |
  | + PSO solo | +261 h | +52 h |
  | + DE solo | +386 h | +77 h |
  | **Total (los 3)** | **912 h** | **~183 h ≈ 7.6 días** |

  La razón por la que **DE solo cuesta más que el objetivo puro** (386 h
  contra ~241 h de objetivo): su `un_paso()` tiene un for-loop por individuo
  con `rng.choice()` y slicing — **~77 µs de overhead por evaluación**,
  independiente de la función, frente a solo ~11 µs/eval en PSO (vectorizado
  con numpy). Es un costo real de la implementación de DE tal como está,
  **no se modifica** (tocar su bucle interno arriesga cambiar el orden de
  llamadas al RNG — un problema de determinismo como D-2, no vale la pena
  para un baseline).
- **Sigue siendo muchísimo menor** que el roster de 20–25 externos
  (semanas–meses en esta máquina). Las mismas palancas aplican
  proporcionalmente a los tres: evaluador CEC compilado, más núcleos.
- **Pendiente:** decidir si status quo (~7.6 días en esta máquina) es
  aceptable o si hace falta más cómputo antes de lanzar. D-2 ya está
  resuelto — la parrilla se puede lanzar en cuanto se decida esto y el
  sourcing de §3 quede cerrado (D-4 ya no bloquea el lanzamiento).

### D-5 · Evaluador CEC2022 compilado (Numba) — RESUELTO

- **Motivación:** D-4 dejó la parrilla en ~183 h con 5 workers (~7.6 días).
  Del propio estimador calibrado, la función objetivo de opfunu (Python
  interpretado + overhead de numpy por llamada) era ~79% del tiempo total.
  Se evaluó envolver el código C oficial de CEC2022 vía `ctypes`, pero esta
  máquina no tiene compilador C/C++ instalado (`cl.exe`, `gcc`, `clang`
  ausentes) — instalar un toolchain nativo (MSVC Build Tools o MinGW) es
  fricción adicional evitable.
- **Decisión (2026-09-04):** reemplazar el evaluador por un backend
  compilado con **Numba** (`@njit`), que no requiere toolchain nativo
  (`llvmlite` trae wheel precompilado; confirmado `numba==0.67.0` +
  `llvmlite==0.49.0` con wheel `cp314-win_amd64`). El backend nuevo
  (`problems/cec2022_numba.py`) **no reimplementa la especificación CEC
  desde cero**: reutiliza los mismos arrays de shift/rotación/shuffle que
  `opfunu` ya carga desde `data_2022/`, portando línea a línea las 12
  funciones de `opfunu/cec_based/cec2022.py` y las funciones elementales
  de `opfunu/utils/operator.py` (incluyendo comportamientos no triviales
  como `rounder()` en F4, que no se "corrige", se reproduce tal cual).
  `ProblemaCEC2022` gana `backend: str = "numba"` (default) | `"opfunu"`
  (referencia, usado solo por el test de paridad); el resto del wrapper
  (contador de FES, checkpoints, `error()`) no cambia.
- **Validación de paridad** (`tests/test_cec2022_numba_paridad.py`, gate de
  aceptación): las 12 funciones × D∈{10,20}, evaluadas en el óptimo, 200
  puntos aleatorios y las esquinas del dominio, contra tolerancia
  `abs(diff) ≤ max(1e-6, 1e-9·|valor|)`. Resultado: **error relativo
  máximo entre 1e-14 y 1e-16 en las 24 combinaciones** — ruido de punto
  flotante por orden de suma distinto (loop secuencial vs reducción de
  numpy), no discrepancia de fórmula. `tests/test_determinismo_orquestador.py`
  (criterio de aceptación de D-2) sigue dando igualdad exacta entre
  corridas bajo el nuevo backend — la evaluación es una función pura sin
  estado ni RNG, cambiar de backend no reabre D-2.
- **Speedup medido** (µs/eval, backend numba vs opfunu, esta máquina, post
  warm-up JIT): entre **6× (F1, función barata, dominada por overhead de
  llamada) y 220× (F8, la más cara bajo opfunu)**. Las funciones híbridas
  y de composición (F7-F12, antes 85-666 µs/eval) son las que más ganan
  (20×-220×), justo las que dominaban el tier MaxFES=5×10⁶.
- **Efecto medido en la parrilla completa (primera pasada, solo evaluador):**
  con `_US_POR_EVAL` recalibrado con los valores medidos del evaluador
  Numba, pero las demás constantes del estimador (`_OVERHEAD_US_POR_EVAL`,
  costo de Fase 2) sin tocar todavía: **912 h → 206.6 h secuencial, ~183 h
  → ~41.3 h con 5 workers (~1.72 días)**. Este número resultó optimista —
  ver el piloto real abajo.

- **Corrida piloto real (143 corridas, 2 rondas) — recalibración completa
  del estimador (2026-09-04).** Antes de lanzar la parrilla completa se
  corrió un piloto (108 corridas amplias en tiers baratos + 32 puntuales
  en tiers caros para caracterizar mejor `middleware`) y se comparó tiempo
  real vs. estimado de `_estimacion_seg()`. Dos hallazgos, independientes
  de D-5 pero recién visibles porque el evaluador dejó de ser el costo
  dominante:
  1. **`_OVERHEAD_US_POR_EVAL` (pso/de) estaba desactualizado y
     sobreestimaba el costo real.** Regresión lineal sobre 37+37 corridas
     reales (R²=1.000 ambos): PSO 11.0→**5.34 µs/eval**, DE 77.0→
     **48.33 µs/eval**.
  2. **El costo de Fase 2 (FastSHAP+XGBoost) estaba subestimado ~5×**, y
     había un costo fijo de arranque por corrida (~construcción de
     XGBoost/FastSHAP/torch) invisible mientras opfunu dominaba el tiempo
     total. Regresión sobre 69 corridas reales de middleware, incluyendo
     los tiers 500 000 y 5 000 000 (R²=0.893): antes se asumía 2.5 s por
     activación de Fase 2, medido **~12.56 s/activación**. El conteo de
     activaciones **se estabiliza por función** (rango 3-9) una vez
     MaxFES ≥ 50 000 en vez de seguir creciendo — el tope de 9 de
     `min(9, MaxFES/25_000)` sigue siendo una aproximación razonable para
     maxfes ≥ 225 000 (validado: error promedio ~8% en esos tiers), pero
     subestima en el tier más barato (MaxFES=5 000, donde el conteo real
     es un valor discreto pequeño específico de cada función, no
     proporcional a MaxFES) — ese residuo agrega ~9 h secuenciales
     (~1.8 h con 5 workers) no capturadas por el modelo, pequeño frente al
     total.
  3. Las tres constantes se corrigieron en `experimentos/grid.py`
     (`_OVERHEAD_US_POR_EVAL`, y la fórmula de `_estimacion_seg` para
     `"middleware"`).

- **Efecto medido en la parrilla completa (final, validado con el
  piloto):**

  | | secuencial | 5 workers |
  |---|---|---|
  | middleware | 138.3 h | 27.7 h |
  | pso solo | 15.4 h | 3.1 h |
  | de solo | 96.5 h | 19.3 h |
  | **Total** | **250.3 h** | **~50.1 h (~2.09 días)**, ~52 h (~2.15 días) con el residuo conocido del tier barato |

  Comparado con el punto de partida pre-D5 (~183 h / ~7.6 días), sigue
  siendo una reducción de **~3.5×**, aunque menor que la primera pasada
  optimista (~41.3 h) porque esta corrigió, además del evaluador, dos
  costos de Python que estaban mal calibrados desde antes de D-5.

- **No llega a la meta orientativa de 1.5 días (36 h)** planteada al
  iniciar esta decisión. La razón, anticipada antes de medir: el
  evaluador compilado no toca el overhead de Python del *bucle* de DE
  (`bioalgorithms/de.py::un_paso()`, for-loop por individuo con
  `rng.choice()` y slicing) — con el objetivo casi gratis, ese overhead
  pasa a ser **el cuello de botella de "de solo"** y, junto con el costo
  real de Fase 2 en middleware, de toda la parrilla.
- **Decidido (2026-09-04): NO vectorizar el loop de DE.** Se evaluó como
  próxima palanca para bajar de 1.72 a ~1.5 días, pero se descarta:
  - Vectorizar `DE.un_paso()` reordena las llamadas al RNG (`rng.choice()`
    por individuo → una operación vectorizada de una sola vez), el mismo
    tipo de riesgo de reproducibilidad que motivó D-2. Con la misma
    semilla, el resultado dejaría de coincidir con el ya calibrado — no
    por un bug, sino porque el stream de números pseudoaleatorios se
    consumiría en un orden distinto.
  - Riesgo más serio que la sola reproducibilidad: `rng.choice(candidatos,
    size=3, replace=False)` garantiza 3 vecinos distintos entre sí y
    distintos de `i`, sin sesgo. Una versión vectorizada de "elegir 3
    vecinos excluyendo el propio índice, para los `n` individuos a la
    vez" es fácil de escribir con un sesgo sutil (sobre-representar
    ciertos vecinos, permitir duplicados, no excluir `i` en un caso
    límite) que **cambiaría la distribución real de la mutación
    DE/rand/1** — justo la propiedad que Fase 0 validó como la que hace
    que DE se estanque de forma confiable en las funciones multimodales
    del CEC2022 (`tests/test_fase0_algoritmos_base.py`). Esa propiedad es
    la premisa completa del experimento (el middleware necesita un
    objetivo que realmente se estanque). A diferencia de D-5 (función
    pura, paridad numérica verificable con puntos aleatorios), esto no
    admite un test de "mismo input → mismo output": la validación pasaría
    a ser un juicio estadístico/de comportamiento sobre varias semillas,
    no una verificación directa.
  - Se acepta el status quo post-D5: **~41.3 h (~1.72 días) con 5
    workers**, ~5 h por encima de la meta orientativa de 1.5 días. Ya no
    bloquea el lanzamiento de la parrilla — ver README "Pendiente".
- Nueva dependencia: `numba>=0.67` en `requirements.txt`.

---

## C. Cómo consumir esta configuración desde el harness

- Punto de entrada: `experimentos/corrida.py` (`correr_corrida`) y
  `experimentos/grid.py --algoritmo middleware,pso,de` (default: los tres).
  No re-implementar el armado de PSO+DE+Orquestador ni los bucles de PSO/DE
  solos — ya están en `corrida.py` (`_correr_middleware`,
  `_correr_algoritmo_solo`).
- `corrida.py` ya aplica la config de A: pop 30, `_PSO_KW` / `_DE_KW`.
  - `"middleware"`: `fraccion_presupuesto=0.5` (default, exacto — D-2), PSO
    y DE reciben dos sub-semillas independientes vía
    `np.random.SeedSequence(semilla).spawn(2)`, `Orquestador(...,
    silencioso=True)` con el planificador cooperativo determinista.
  - `"pso"` / `"de"` (D-4): corren solos, `fraccion_presupuesto=1.0`, con la
    semilla de la corrida directa (sin spawn — no hay un segundo algoritmo
    con quien repartir), sin orquestador (bucle secuencial simple).
- Cada corrida usa una instancia nueva de `ProblemaCEC2022` +
  `configurar_checkpoints(max_fes)`.
- Salida: fila §11.2 (todos) + fila §11.3 (diagnósticos del middleware,
  `None` para `"pso"`/`"de"`), un Parquet por corrida; `consolidar.py` los une.
