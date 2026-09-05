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

### D-1a · Calibración de los pesos del Score (`W_FIR` / `W_DWD`)

- **Estado previo:** 0.3 / 0.7, marcados "empíricos provisionales".
- **Decisión (2026-09-01):** se conservan **0.3 / 0.7** para el setup. La
  validación será un **análisis de sensibilidad**, no un grid-search de
  optimización: correr 3 configuraciones — `0.2/0.8`, `0.3/0.7`, `0.5/0.5` —
  sobre un subconjunto (12 funciones CEC 2022, D=10, MaxFES = 5×10⁴,
  ~10 semillas) y reportar si el error final del sistema es robusto a la
  elección. Si lo es, 0.3/0.7 queda justificado; si no, se revisa.
- **Bloqueado por:** el harness experimental (punto 4 del plan). No se
  puede correr hasta tenerlo.
- **Pendiente:** ejecutar el análisis de sensibilidad y añadir sus
  resultados a este documento y al paper (sección de resultados / material
  suplementario).

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
- **Pendiente:** confirmar en el análisis de sensibilidad (D-1a) que `20`
  epochs no degrada la calidad del middleware en otras funciones; queda la
  opción de bajar más si el R²/atribución lo permiten.

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
