# Configuración experimental — parámetros congelados y registro de decisiones

> **Propósito.** El setup experimental de la tesis (§4) exige **una única
> configuración de parámetros** para el algoritmo propuesto, sin ajuste por
> valor de MaxFES ni por problema. Este documento fija esa configuración y
> deja registro de las decisiones tomadas para congelarla, incluyendo las
> que quedan pendientes de una revisión más profunda.
>
> Última actualización: 2026-10-12 (D-8: competidores externos; D-9:
> evaluador CEC 2022 re-portado desde el C oficial — resuelto; D-10:
> Canal A heterogéneo PSO↔DE en espacio latente común).

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
| Valores de MaxFES | 5×10⁴, 5×10⁵, 5×10⁶, 5×10⁷ (D-6, 2026-09-20) | setup §2 |
| Planificador | Cooperativo determinista (round-robin, un hilo) | `Orquestador.ejecutar` — D-2, **resuelto** |
| Hilos por proceso (torch/BLAS/Numba) | **1** (`OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=1`), paralelismo solo por workers | `experimentos/grid.py` — D-7 |
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
| Espacio del Canal A (D-10) | latente común: exploración y adopción |
| Exploración — PSO | (w − 0.4) / (0.9 − 0.4) (afín, sin recorte) |
| Exploración — DE | (F − 0.4) / (1.0 − 0.4) (afín, sin recorte) |
| Adopción — PSO | c2 / (c1 + c2); la inversa conserva c1 + c2 |
| Adopción — DE | CR |
| `w` recibida por PSO | desplaza w_max y w_min por Δ = w_nueva − w_actual, acotados a [0, 1] |

### A.6 Barrera de seguridad (`middleware/barrera_seguridad.py`)

| Parámetro | Valor |
|---|---|
| θ_W (Wasserstein) | 0.6 · diámetro máximo del espacio (D-11) |
| Ancho de banda MMD | mediana de distancias **intra-grupo** (en el filtro: intra-**objetivo**, calculado en Canal B) |
| Centrado MMD | candidatas centradas en la media de la fuente (fijada en Fase 2), objetivo en la suya |
| Filtro MMD | **por instancia**: MMD²(δ_x, objetivo) de cada candidata (D-11) |
| Umbral MMD | máximo del nulo leave-one-out del **objetivo** (misma estadística), calculado en Canal B (D-11) |
| Objetivo colapsado | dispersión del objetivo < 10⁻³ · diámetro → filtro MMD no aplicable, aprueba todo y se registra (D-11) |

### A.7 Protocolo

| Parámetro | Valor |
|---|---|
| Corridas independientes por (problema × dim × MaxFES) | 51 |
| Semillas | lista fija, versionada (pendiente de generar con el harness) |
| Métrica de comparación | media de las 51 corridas de `f_obtenido − f(x*)` |
| Corridas con MaxFES mayor | **no** son continuación: cada una desde cero |

### A.8 Competidores externos (D-8)

No son parte del algoritmo propuesto: corren solos, con los parámetros
de sus autores (setup §4, sin re-tuning), la misma semilla de la corrida
y corte exacto en MaxFES. Código en `experimentos/competidores/`.

| Competidor (`id`) | Fuente | Parámetros |
|---|---|---|
| IPOP-CMA-ES (`cmaes`) | librería `cma` 4.5.0 (N. Hansen) + bucle IPOP en `cmaes_ipop.py` | λ₀ = 4+⌊3 ln D⌋ (10 / 12); λ ×2 por reinicio, sin tope de reinicios; σ₀ = 0.25·(ub−lb) = 50; x₀ uniforme nuevo en cada reinicio; `BoundTransform` y criterios de parada por defecto de `cma`; semilla de cada reinicio derivada de la de la corrida |
| MadDE (`madde`) | port de `MadDE.m` (github.com/subhodipbiswas/MadDE @13f0582, GPL-3.0) | NP = 2·D² → 4 (lineal); memoria 10·D; F y CR iniciales 0.2; p_best 0.18; archivo 2.3·NP; q_cr 0.01; 3 operadores con probabilidad adaptativa; inicialización Sobol aleatorizada |
| NL-SHADE-RSP (`nlshadersp`) | port de `nl_shade_rsp.cpp` (entrega CEC 2021, archivo "Codes-of-top-methods" de P-N-Suganthan/2021-SO-BCO) | NP = 30·D → 4 (no lineal); memoria 20·D; F y Cr iniciales 0.2; archivo 2.1·NP; pbest 0.2→0.4; cruce binomial/exponencial 50 %; prob. de archivo adaptativa |

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

### D-1d · Comportamiento con `MaxFES = 5×10³` — TIER RETIRADO (ver D-6)

> **2026-09-20:** el tier `MaxFES = 5×10³` que motiva este punto **ya no
> es parte del protocolo vigente** (D-6 lo retiró, junto con la
> incorporación de `5×10⁷`). Se conserva esta entrada como registro
> histórico del hallazgo y su diagnóstico (siguen siendo válidos para
> entender el comportamiento del middleware con presupuestos chicos),
> pero **no aplica a ninguna corrida futura**. Si el régimen "middleware
> casi inerte" reaparece en el nuevo tier más chico (`5×10⁴`), amerita
> una entrada nueva, no reabrir esta.

- **Problema (histórico):** con ese presupuesto el middleware **no alcanza a activarse**.
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

### D-1e · Test rojo de la barrera de seguridad (`tests/test_barrera_seguridad.py`, CASO 3) — RESUELTO (D-11)

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
- **Resolución (2026-10-13, con D-11):** el filtro MMD pasó a decidir por
  instancia (MMD²(δ_x, objetivo), centrada en la media de la fuente), así
  que las copias idénticas ya no forman un grupo de varianza cero: cada una
  se evalúa sola y la guarda (ii) no hace falta. Se aplicó la opción (i):
  el CASO 3 se reescribió con casos controlados — misma forma con el
  objetivo trasladado +300 (5/5 aprobadas), una élite en la esquina extrema
  (solo esa rechazada), cinco copias de la esquina (0/5), objetivo 5x más
  concentrado (0/5), objetivo 5x más disperso (5/5, las élites caen dentro)
  y objetivo colapsado (5/5, filtro no aplicable). Pasa.

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
    documento de tesis. *(Nota D-12: el middleware no aplica ese 0.7; solo
    bloquea con R² < 0.)*
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

> **2026-10-12 — ampliado por D-8:** PSO y DE solos siguen siendo
> competidores (lo de abajo sigue vigente), pero ya no son los únicos: D-8
> agrega tres competidores externos. Lo que D-8 revierte es solo el "sin
> roster externo".

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
- **⚠ Corregido por D-9 (2026-10-12):** la "paridad" de abajo se validó
  contra **opfunu**, cuyo CEC 2022 no es el oficial (difiere en 11 de 12
  funciones). El evaluador Numba se re-portó desde el C oficial y el test
  ahora valida contra el código oficial — ver D-9. Lo demás de D-5 (Numba
  sin toolchain C, speedup frente a un evaluador interpretado) sigue en pie.
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

### D-6 · Cambio de tiers de MaxFES y rechazo (repetido) de D=50 — RESUELTO

- **Pedido (2026-09-20):** el usuario solicitó dos cambios al setup: (1)
  quitar el tier `MaxFES = 5×10³` y agregar uno nuevo por encima de
  `5×10⁶`; (2) agregar `D = 50` como tercera dimensionalidad.
- **D=50 — rechazado, se reafirma la decisión ya tomada.** Se verificó
  directamente contra `opfunu` (no por inspección de documentación):

  ```
  >>> cec2022.F12022(ndim=50)
  ValueError: F12022 problem supports maximum 20 variables!
  ```

  El benchmark CEC2022 oficial **no define datos de shift/rotación para
  D=50** (solo D∈{2,10,20} según la función) — esto ya estaba documentado
  como decisión de alcance (`setup_experimental.md` §1: *"CEC 2022 no
  define dimensiones mayores"*; `README.md`: *"no se amplía... ni
  D=50"*), y esta verificación lo confirma con el error real, no solo por
  la nota histórica. D=50 sí existe en CEC2014/2017, pero agregarlo
  implicaría otro benchmark completo (otro wrapper, otro backend Numba
  — el de `problems/cec2022_numba.py` está atado a las 12 funciones y
  los datos de CEC2022 —, revalidar paridad, recalibrar tiempos), no un
  cambio menor. **Se mantiene D∈{10,20}.**
- **MaxFES — aplicado.** Se retira `5×10³` y se agrega `5×10⁷`,
  manteniendo la progresión geométrica ×10 ya existente:
  `{5×10⁴, 5×10⁵, 5×10⁶, 5×10⁷}`. Cambios: `experimentos/grid.py`
  (`MAXFES_VALIDOS`, default de `--maxfes`), `analisis/datos.py`
  (`MAXFES`). La D-1d (comportamiento del middleware con
  `MaxFES=5×10³`) queda marcada como **tier retirado** — se conserva como
  registro histórico, no aplica a corridas futuras (ver nota en esa
  sección).
- **Impacto en los datos ya calculados:** de las 14 688 corridas de la
  parrilla completa (D-5), **11 016 siguen siendo válidas** (los 3 tiers
  compartidos: `5×10⁴, 5×10⁵, 5×10⁶`) — no hace falta recalcularlas.
  Faltan **3 672 corridas nuevas** (12 funciones × 2 dims × 51 semillas ×
  3 algoritmos) para el tier `5×10⁷`; las 3 672 del tier retirado
  `5×10³` quedan huérfanas (archivadas en la rama
  `respaldo-2026-09-20-parrilla-analisis`, no se borran de `main` sin
  pedido explícito).
- **⚠ Impacto en cómputo — no es un agregado menor.** `experimentos/grid.py
  --dry-run` con el protocolo nuevo:

  | | secuencial | 5 workers |
  |---|---|---|
  | Solo el tier nuevo (5×10⁷, 3 672 corridas pendientes) | 1561.4 h | **~312.3 h (~13 días)** |

  El tier `5×10⁷` por sí solo cuesta más que **toda** la parrilla anterior
  completa (4 tiers, 71.3 h reales). Es la consecuencia esperada de que
  el costo escale ~linealmente con MaxFES y el nuevo techo sea 10× el
  anterior — no hay evaluador más rápido que lo evite (D-5 ya explotó esa
  palanca). **Pendiente:** decidir cuándo lanzar esas 3 672 corridas — no
  se lanzó en esta misma tanda, dado el tamaño del compromiso de cómputo.
- **Validación reforzada:** `analisis/datos.py::cargar_corridas()` ahora
  valida el conjunto exacto de `(dim, max_fes)` presentes contra el
  protocolo vigente, no solo el conteo total de filas — un Parquet del
  protocolo viejo podía tener casualmente las mismas 14 688 filas (4
  tiers × ...) y pasar inadvertido con la validación anterior, produciendo
  un análisis silenciosamente incorrecto (rankings/Friedman sobre tiers
  que ya no son el protocolo, con el tier nuevo vacío).

### D-7 · Cambio de CPU: recalibración de tiempos y 1 hilo por proceso — RESUELTO

- **Contexto (2026-10-08):** se cambió el procesador del equipo de cómputo
  (`DESKTOP-D1SJUAH`, mismo hostname) a un **Intel Xeon E5-2673 v3**
  (12 núcleos / 24 hilos, 2.4 GHz). Las constantes de `_estimacion_seg()`
  (D-5) y la estimación de D-6 (~312 h con 5 workers para el tier `5×10⁷`)
  se habían calibrado en la CPU anterior.
- **Método:** se re-corrieron `run_id` ya existentes a `MaxFES = 5×10⁶`
  (por D-2 hacen exactamente el mismo trabajo) y se comparó `tiempo_seg`
  contra `resultados/corridas.parquet`. Factor = tiempo nuevo / tiempo
  viejo por corrida:

  | Configuración | pso | de | middleware |
  |---|---|---|---|
  | 12 workers, hilos por defecto | 2.28 | 3.59 | 5.26 |
  | 3 workers, hilos por defecto | 1.68 | 1.84 | 1.87 |
  | 12 workers, 1 hilo/proceso | 1.62 | 1.65 | 1.36 |
  | 16 workers, 1 hilo/proceso | — | 1.94 | 1.67 |
  | 20 workers, 1 hilo/proceso | — | 2.28 | 2.04 |
  | **24 workers, 1 hilo/proceso** | (≈ de) | **2.54** | **2.25** |

- **Hallazgos:**
  1. Por núcleo, el Xeon es ~1.6× más lento que la CPU anterior; la
     ganancia viene de tener más núcleos e hilos.
  2. Con los pools de hilos por defecto, cada proceso abre su propio pool
     (torch/BLAS/Numba) y 12 workers se pelean los núcleos: el rendimiento
     total cae por debajo del de la CPU anterior con 5 workers. **Se fija
     1 hilo por proceso** en `experimentos/grid.py` (antes de importar
     numpy/torch; los hijos heredan el entorno).
  3. Con 1 hilo/proceso el rendimiento total (workers ÷ factor, en
     "workers de la CPU anterior") sigue subiendo hasta usar todos los
     hilos lógicos: 12 → ~7.3–8.8, 16 → ~8.3–9.6, 20 → ~8.8–9.8,
     **24 → ~9.5–10.7** (vs. 5 antes, ~2×). Hyper-Threading aporta ~+30%
     sobre 12 workers. Default de `--workers` → `os.cpu_count()` (24).
- **⚠ Reproducibilidad:** el resultado depende de la configuración de
  hilos. De 32 `run_id` distintos re-ejecutados con 1 hilo (75
  ejecuciones), **uno** difiere de la
  original: `middleware__f12__d10__fes5000000__s00` (error 102.5666 vs
  101.9588 original; probablemente reducciones de punto flotante en el
  entrenamiento torch de FastSHAP). Con 1 hilo da el mismo valor en las 4
  repeticiones, y con hilos por defecto reproduce el original: el
  determinismo de D-2 se mantiene **dada una configuración de hilos fija**.
  Por eso la configuración de hilos pasa a ser parte del protocolo (tabla
  A.2). Las 11 016 corridas existentes (tiers `5×10⁴`–`5×10⁶`) se hicieron
  con hilos por defecto; el tier `5×10⁷` se hará con 1 hilo. Ambas son
  corridas válidas e independientes, pero no son bit a bit intercambiables
  con una re-ejecución bajo la otra configuración — se documenta, no se
  re-corre.
- **Recalibración:** `_estimacion_seg()` conserva las constantes de D-5 y
  multiplica por `_FACTOR_CPU = {pso: 2.54, de: 2.54, middleware: 2.25}`
  (factor a 24 workers; pso no se midió a 24 y se asume igual a de, como a
  12 workers). Nuevo `--dry-run` para el tier pendiente:

  | | suma de corridas | 24 workers |
  |---|---|---|
  | middleware | 1244.0 h | 51.8 h |
  | pso | 352.7 h | 14.7 h |
  | de | 2209.0 h | 92.0 h |
  | **Total (3 672 corridas, `5×10⁷`)** | **3805.7 h** | **~158.6 h (~6.6 días)** |

  vs. ~312 h (~13 días) estimadas en D-6. Limitaciones: la medición fue
  a `5×10⁶` y D=10 (el escalado a `5×10⁷` sigue siendo la extrapolación
  lineal del modelo D-5); usar el equipo para otra cosa durante la corrida
  la alarga.

### D-8 · Competidores externos: IPOP-CMA-ES, MadDE y NL-SHADE-RSP — IMPLEMENTADO

- **Contexto:** con solo propuesto/PSO/DE (D-4), los rankings tenían k=3,
  válidos pero poco informativos, y sin rivales del estado del arte. El
  2026-10-08 se dejó una propuesta en evaluación (README) con criterios de
  selección y 3–4 candidatos del roster de `competidores.md`.
- **Decisión (2026-10-12, pedido del usuario):** se agregan los **3
  primeros** de esa propuesta. Los criterios (en orden de peso) fueron:
  responder la pregunta de la tesis (al menos un rival que combine
  estrategias **sin** transferencia: MadDE), credibilidad CEC 2021/2022,
  diversidad de paradigma (faltaba uno basado en distribución:
  CMA-ES), integración fiel y barata, y escalar a 5×10⁷ (reinicios o
  calendario normalizado contra MaxFES). EA4eig (ganador CEC 2022, solo
  MATLAB) y HCLPSO **no** se adoptan.
- **Rol:** corren **solos, como medida comparativa** — no se acoplan al
  middleware ni implementan la interfaz de `bioalgorithms/`. Caja negra
  `correr(problema, max_fes, semilla)` sobre el mismo `ProblemaCEC2022`
  (mismo contador de FES, checkpoints y semillas); solo llenan §11.2,
  §11.3 queda en `None` como para `pso`/`de`.
- **Implementación** (`experimentos/competidores/`, parámetros en §A.8):
  - `cmaes_ipop.py` — `cma` 4.5.0 de Hansen con bucle IPOP propio
    (reinicios sin tope, λ ×2).
  - `madde.py` — port línea a línea del **MATLAB** oficial. Corrección a la
    propuesta del README, que decía "repo Python de los autores": el repo
    es MATLAB.
  - `nlshade_rsp.py` — port línea a línea del **C++** oficial de la
    competición CEC 2021.
  - Los ports **reproducen las rarezas del código de referencia**, con el
    mismo criterio que D-5, y quedan documentadas en el docstring de cada
    módulo. En MadDE: el archivo recibe `popold` sin reordenar y `probDE`
    solo se actualiza si los 3 operadores tuvieron éxito. En NL-SHADE-RSP:
    la memoria de F usa el último F muestreado, el cruce binomial usa
    `CrToUse` en vez del Cr adaptado, r1 puede coincidir con el propio
    individuo, y se replica la semántica de `RemoveWorst`.
  - **Corte exacto en MaxFES** (`base.Presupuesto`): los originales
    terminan la generación en curso y se pasan hasta en una población.
    Aquí se corta en la evaluación exacta, que es lo que mide el
    checkpoint 1.0.
  - `experimentos/corrida.py` registra los runners desde `COMPETIDORES`;
    la columna `version` agrega `|alg:<VERSION>` (setup §10).
    `esquema.FAMILIA_ALGORITMO`: `cmaes` → `distribucion`, `madde` /
    `nlshadersp` → `DE`. `grid.py`: `ALGORITMOS_VALIDOS` y el default de
    `--algoritmo` pasan a los 6.
  - `analisis/` generalizado a k=6: `ALGORITMOS`, validación de los
    algoritmos presentes en `cargar_corridas()`, **Shaffer con la secuencia
    derivada para cualquier k** (S(k) recursivo; verificado contra k=3,4),
    Wilcoxon propuesto-vs-cada-competidor (5 pares) + pso-vs-de, q_α de
    Nemenyi contrastado con la tabla de Demšar para el k vigente (k=6:
    q_α = 2.850, **CD ≈ 2.18**), figuras ajustadas a 6 series.
  - Nueva dependencia: `cma==4.5.0` (`requirements.txt`).
- **Verificación:**
  - `tests/test_experimentos.py` cubre los 6 algoritmos: estado `ok`,
    consumo exacto de MaxFES, checkpoints monótonos, `version` y
    **determinismo** (misma semilla → mismo error).
  - Piloto de 216 corridas (3 competidores × 12 funciones × 2 dims × 3
    semillas, MaxFES = 5×10⁵, 24 workers): 216/216 `ok`, todas con
    `fes_consumidas == max_fes`.
  - Calidad coherente con el pedigrí, comparando con 3 semillas contra las
    51 existentes a 5×10⁵: MadDE y NL-SHADE-RSP superan a PSO/DE solos y
    al propuesto en la mayoría de las funciones. IPOP-CMA-ES también, salvo
    F9 y F12 en D=10, donde queda detrás del propuesto. **Esta comparación
    se hizo con el evaluador viejo (pre-D-9) y no vale como resultado**:
    parte de los errores 0 eran artefacto del bug. Con el evaluador oficial
    los tres competidores alcanzan los pisos publicados de CEC 2022 (D-9).
  - `analisis/main.py` corrido de punta a punta sobre un dataset sintético
    de 6 algoritmos.
  - El piloto destapó **D-9** (abajo).
- **Calibración de tiempos** (`grid.py::_COSTO_US_POR_EVAL`, medido bajo
  carga completa en la CPU actual, sin `_FACTOR_CPU`; medido con el
  evaluador pre-D-9 — el oficial cuesta ~1–2 µs/eval más, <10 % de estos
  valores, no se re-midió):

  | µs/eval (D=10 / D=20) | a 5×10⁵ | escala con MaxFES |
  |---|---|---|
  | IPOP-CMA-ES | 176.7 / 239.6 | ×0.81 / ×0.72 por década (medido a 5×10⁶; λ crece con los reinicios) |
  | MadDE | 44.7 / 30.0 | ~constante (confirmado a 5×10⁶) |
  | NL-SHADE-RSP | 63.2 / 49.9 | ~constante (confirmado a 5×10⁶) |

  CMA-ES es el caro: `cma` tiene un costo fijo por iteración de Python
  que con λ = 10–12 domina sobre el evaluador Numba.
- **⚠ Impacto en cómputo** (`grid.py --dry-run`, 24 workers):

  | | corridas | suma | 24 workers |
  |---|---|---|---|
  | IPOP-CMA-ES | 4 896 | 2349.6 h | 97.9 h |
  | MadDE | 4 896 | 705.4 h | 29.4 h |
  | NL-SHADE-RSP | 4 896 | 1068.1 h | 44.5 h |
  | **Competidores nuevos** | **14 688** | **4123.1 h** | **~171.8 h (~7.2 días)** |
  | ~~Pendiente total (competidores + tier 5×10⁷ de los 3 de antes)~~ | ~~18 360~~ | ~~7928.8 h~~ | ~~~330.4 h~~ — obsoleto por D-9 |
  | **Parrilla completa desde cero (los 6) — vigente tras D-9** | **29 376** | **8552.5 h** | **~356.4 h (~14.9 días)** |

  La extrapolación de CMA-ES a 5×10⁷ (85 h del total) asume que el
  decaimiento medido entre 5×10⁵ y 5×10⁶ se mantiene — es la parte más
  incierta de la estimación.
- **Pendiente:** lanzar la parrilla completa desde cero (D-9 está
  resuelto e invalidó las corridas existentes).

### D-9 · El CEC 2022 de opfunu no es el oficial — evaluador re-portado desde el C oficial — RESUELTO

- **Hallazgo (2026-10-12, piloto de D-8):** MadDE y NL-SHADE-RSP marcaban
  error 0 en F12/D=10, donde la literatura de CEC 2022 reporta un piso de
  ~160. Al registrar el valor **crudo** (sin el `np.maximum(err, 0)` de
  `ProblemaCEC2022.error`) aparecían soluciones **por debajo del óptimo
  declarado** en 9 de las 24 combinaciones (función, D): F7 y F8 en D=20,
  F10 (hasta −2409 contra f\* = 2400), F11/D=10 (−10 025) y F12. El
  backend `opfunu` daba los mismos valores: el port Numba (D-5) lo había
  heredado por paridad.
- **Primera causa encontrada:** `opfunu/utils/operator.py::modified_schwefel_func`
  (opfunu 1.0.1), rama `z > 500`, usa `(500.0 + fmod(|z|, 500))`, mientras
  que el C oficial usa `(500.0 - fmod(z, 500))`.
- **Diagnóstico completo — el problema era mucho más amplio.** Al comparar
  contra el código oficial apareció que el CEC 2022 de opfunu **difiere del
  oficial en 11 de las 12 funciones** (solo F2 coincide). Todas coinciden
  en el óptimo x\*, y por eso ni el test de paridad de D-5 (que comparaba
  contra opfunu) ni la tabla de f(x\*) lo detectaban. Diferencias
  encontradas en opfunu respecto del C oficial:
  - **Zakharov (F1):** sin el peso (i+1) en el segundo término.
  - **F3:** opfunu evalúa Expanded Schaffer F6 rotada; la oficial es
    Schaffer F7, que además opera sobre el vector desplazado **sin rotar**
    (lee el buffer global `y`).
  - **Levy (F5):** el oficial no escala por 5.12/100.
  - **Híbridas (F6–F8):** el oficial aplica a cada sub-función su propio
    factor de escala (`5/100`, `5.12/100`, `1000/100`…), y la Schaffer F7
    de F7 lee el `y` global desde el índice 0.
  - **Katsuura:** redondeo con `floor(x+0.5)` en vez de `round`.
  - **Schwefel modificado:** el signo de `fmod` (arriba).
  - **Composición (F9–F12):** opfunu transforma todos los componentes con
    el desplazamiento del primer óptimo (`x − o₁`), mientras que el oficial
    usa el de cada componente (`x − oᵢ`); además hay componentes que el
    oficial no rota (p. ej. el 5.º de F9, el Schwefel de F10).
- **Árbitro usado:** el **C oficial** (`CEC2022/C-Code/cec22_test_func.cpp`,
  github.com/P-N-Suganthan/2022-SO-BO), compilado como DLL con `zig c++`
  (paquete `ziglang` de PyPI, instalado solo en un directorio temporal; no
  es dependencia del proyecto) y llamado vía `ctypes`. Contra él:
  - el **port Python oficial** de los organizadores (`Python-CEC2022.zip`
    del mismo repo) coincide hasta ≤ 4·10⁻¹⁶ relativo en las 12 × 2;
  - opfunu (y el Numba de D-5) difería hasta 10¹⁵ relativo.
  - Los archivos de datos (M, shift, shuffle) de opfunu `data_2022/` son
    **idénticos en valor** a los `input_data/` oficiales (54/54): el
    problema estaba solo en las fórmulas.
- **Corrección (2026-10-12, pedido del usuario: la parrilla se re-corre
  completa desde cero):**
  - `problems/cec2022_numba.py` **reescrito como port literal del C
    oficial**, incluidos los efectos de los buffers globales `y`/`z`
    (detallados en su docstring). Lee los datos replicando los `fscanf` del
    C. Resultado contra el C compilado: **≤ 4.2·10⁻¹⁶ relativo en las 24
    combinaciones** (la mayoría exacto, diferencia 0).
  - `problems/cec2022_referencia.py` (nuevo): el port Python oficial
    vendorizado, con 3 cambios mínimos y marcados (ruta de datos, un
    `print` espurio silenciado y lectura secuencial del shift de F8, cuyo
    archivo trae 10 líneas en el C y 1 en el paquete Python, con los mismos
    primeros valores).
  - `ProblemaCEC2022`: el backend de referencia pasa de `"opfunu"` a
    `"referencia"`. **opfunu ya no evalúa nada**: solo aporta metadatos
    (límites, nombre, f(x\*), x\*) y los archivos de datos.
  - `tests/test_cec2022_numba_paridad.py` reescrito como gate de fidelidad:
    48 **anclas calculadas con el C oficial** (x = 0 y x = linspace(−90, 90)
    por función × D), paridad contra la referencia en el óptimo, en 4
    vecindades del óptimo (donde el test viejo era ciego), 100 puntos
    aleatorios y esquinas, más la verificación de que ningún punto quede
    bajo f(x\*).
- **Verificación de comportamiento:** con el evaluador corregido, IPOP-CMA-ES,
  MadDE y NL-SHADE-RSP (MaxFES = 5×10⁵, semilla 0) ya no bajan de f(x\*) en
  ninguna de las 24 combinaciones. Sus pisos coinciden con los reportados
  en la literatura de CEC 2022: F9 = 229.3 (D=10) / 180.8 (D=20), F12 ≈ 160
  (D=10), F10 ≈ 100, F11/D=20 = 300.
- **Consecuencias:**
  - **Todos los resultados anteriores quedan inválidos**: las 14 688
    corridas de la parrilla de 2026-09 (incluidas las 11 016 que D-6
    conservaba), los pilotos de D-8 y el `analisis/` generado con ellas.
    Se evaluaron sobre funciones que no son las del CEC 2022 oficial. No se
    borran: quedan como registro, pero no se deben usar ni mezclar con
    corridas nuevas.
  - La parrilla se re-corre **completa desde cero**: 29 376 corridas (6
    algoritmos).
  - D-5 sigue vigente en lo que importa (evaluador compilado con Numba,
    sin toolchain C), pero su "paridad validada" se refería a opfunu y
    queda reemplazada por esta.
  - Las constantes de tiempo de `grid.py` se midieron con el evaluador
    viejo — ver la recalibración abajo.
- **Resguardo contra mezclar evaluadores:** la columna `version` pasa de
  `eval:numba` a `eval:numba-c22oficial`
  (`problems/cec2022_numba.VERSION_EVALUADOR`).
  - `grid.py` solo da por hecha una corrida si su Parquet tiene esa
    etiqueta. Las 14 688 corridas viejas de `resultados/corridas/` se
    re-ejecutan y se sobrescriben al relanzar (avisa cuántas). No hace falta
    borrarlas a mano, aunque conviene archivar `resultados/` antes de lanzar
    si se quiere conservarlas.
  - `analisis/datos.py::cargar_corridas()` rechaza cualquier dataset con
    corridas de otro evaluador.
- **El middleware sigue activo con las funciones oficiales.** Chequeo de 72
  corridas (propuesto/PSO/DE × 12 funciones × 2 dims, MaxFES = 5×10⁵,
  semilla 0) contra las mismas `run_id` del evaluador viejo:

  | | activaciones Fase 2 | Fase 3 | inyecciones Canal B | abortos Wasserstein |
  |---|---|---|---|---|
  | evaluador viejo | 142 | 132 | 335 | 0 |
  | evaluador oficial | 138 | 132 | 275 | 0 |

- **Tests:** la suite completa corre.
  - `test_cec2022_numba_paridad.py` (gate de D-9), `test_experimentos.py` y
    `test_determinismo_orquestador.py` (D-2) pasan.
  - Hay **3 veredictos de escenario que antes pasaban y ahora no**. Son
    escenarios y valores calibrados a mano sobre las funciones viejas, que
    no eran las oficiales:
    - `test_barrera_seguridad.py` CASO 1: los W₁ "observados durante el
      diseño" (343.99 / 172.34 / 130.74) ahora son ~404 / 399 / 388, y la
      barrera bloquea PSO↔DE en F11/D=20 en los tres snapshots. El pipeline
      real no aborta por Wasserstein (tabla de arriba).
    - `test_fase2_extraccion.py`: R² del subrogado 0.65 en su escenario,
      bajo el criterio R² > 0.7 del documento de tesis (antes 0.72).
    - `test_fase3_transferencia_exitosa.py` (F1, D=10): con el Zakharov
      oficial, el PSO "bien afinado" ya no es mejor que el "mal afinado"
      (6901 contra 2555), así que no se activa ningún ciclo.
  - No se tocaron umbrales, escenarios ni middleware: re-calibrar esos
    escenarios sobre las funciones oficiales queda **pendiente**. El CASO 3
    de la barrera sigue siendo el fallo conocido D-1e. *(Actualización
    2026-10-13: el CASO 1 se recalibró y el CASO 3 se reescribió en D-11;
    `test_barrera_seguridad` pasa 3/3. Los otros dos escenarios se
    resolvieron en D-12.)*
  - `test_fase3_transferencia.py` no terminó en ~1 h ni con el evaluador
    nuevo ni con el viejo (corrido en paralelo sobre el commit anterior,
    igual de lento). Es un problema previo, no de D-9: el test no tiene el
    cooldown del orquestador y puede re-extraer conocimiento (XGBoost +
    FastSHAP) en muchas iteraciones.
- **Recalibración de tiempos:** el evaluador oficial cuesta 0.85–1.44×
  (mediana ~1.1×) lo que costaba el viejo por evaluación, medido alternando
  ambos en el mismo proceso. Se agruparon los argumentos del despacho
  Python→Numba (8 arrays → 4), que al principio lo hacían 1.1–1.8× más
  lento. `_US_POR_EVAL` se escaló por esa razón por función × D; el resto
  de las constantes no se tocó. **Parrilla completa desde cero, 6
  algoritmos: 29 376 corridas, ~356 h (~14.9 días) con 24 workers**
  (`grid.py --dry-run`). El chequeo de arriba, contaminado por otra carga
  en paralelo, dio real/estimado = 1.22 (pso), 1.38 (de) y 0.64
  (middleware): tomar la estimación como ±40 % por algoritmo.

### D-10 · Canal A heterogéneo PSO↔DE en un espacio latente común — IMPLEMENTADO

- **Problema:** el Canal A solo interpolaba hiperparámetros con el mismo
  nombre en fuente y objetivo. PSO {w, c1, c2} y DE {F, CR} no comparten
  ninguna clave, así que entre ellos el Canal A era estructuralmente inerte
  y el orquestador escalaba siempre directo a Canal B (`n_canal_a = 0` en
  toda la parrilla). Con los roles dinámicos de Fase 1 el par puede ir en
  cualquier sentido, y ninguno de los dos usaba el Canal A.
- **Bug encontrado de paso:** PSO exponía la inercia como `"w"`, pero
  `recibir_parametros_transferidos` solo entendía `"w_max"`/`"w_min"`: la
  `w` transferida se descartaba en silencio, también en PSO↔PSO.
- **Decisión (2026-10-12, elegida por el usuario entre 3 alternativas):**
  1. **Espacio latente común**, descartando una tabla fija en el middleware
     y un único escalar de exploración. Cada algoritmo declara
     `a_espacio_comun(hp)` / `desde_espacio_comun(z)` (`bioalgorithms/base.py`;
     por defecto `{}`, lo que deja el Canal A inerte como antes). El
     middleware interpola `z = (1 − rmp)·z_objetivo + rmp·z_fuente` en las
     dimensiones que ambos declaran, sin saber qué son `w` o `F`, así que
     **sigue siendo agnóstico**. Dos dimensiones:
     - **Exploración** (tamaño de paso): `w` normalizada en [0.4, 0.9]
       (calendario clásico, Shi & Eberhart 1998) ↔ `F` normalizado en
       [0.4, 1.0] (rango efectivo, Storn & Price 1997). Mapas afines **sin
       recorte**: el F = 0.2 congelado queda en −0.33, y así el RMP lo mueve
       de a poco en vez de saltar al borde del rango.
     - **Adopción** (cuánta información ajena toma un individuo): peso social
       relativo `c2/(c1+c2)` ↔ `CR`. La inversa en PSO conserva `c1 + c2`.
       Es la analogía más débil: hay que justificarla en la redacción.
  2. **`w` transferida a PSO = desplazar el calendario**, descartando fijar
     `w` durante la ventana o escalar el calendario: `w_max` y `w_min` se
     mueven por `Δ = w_nueva − w_actual` (misma pendiente), acotados a
     [0, 1]. El cambio persiste, igual que `F`/`CR` en DE.
- **Implementación:** `bioalgorithms/base.py` (interfaz + constantes
  `DIM_EXPLORACION`, `DIM_ADOPCION`), `bioalgorithms/pso.py` y
  `bioalgorithms/de.py` (mapas, `W_REF`, `W_COTAS`, `F_REF`; PSO interpreta
  `"w"`), `middleware/transferencia.py::aplicar_canal_a` (recibe además el
  algoritmo fuente e interpola en el espacio común) y
  `middleware/orquestador.py` (pasa el fuente). El resto del protocolo
  (RMP 0.05 → 0.6, ventana 10 % de MaxFES, ΔS > 0.02, escalado a Canal B,
  cooldown) no cambia.
- **Verificación** (`tests/test_fase3_canal_a_heterogeneo.py`, nuevo):
  - Mapas e interpolación exactos (≤ 10⁻¹²): ida y vuelta = identidad,
    PSO→DE y DE→PSO, rmp = 0 / 1, PSO→PSO ahora transfiere `w`, el
    calendario desplazado persiste en los pasos siguientes y un algoritmo
    sin espacio común deja el Canal A inerte.
  - En el orquestador (F1, F2, F5, F11; D=10; MaxFES = 10⁵; semilla 0) el
    Canal A ocurre en **ambos sentidos**, nunca queda inerte entre PSO y DE
    y siempre acerca al objetivo al fuente.
  - Regresión: `test_determinismo_orquestador` (D-2), los tres tests del
    orquestador y `test_experimentos` pasan.
- **Efecto observado, 1 semilla (indicativo, no concluyente):**

  | F (D=10) | error con Canal A | error sin (pre-D-10) | ventanas Canal A | exitosas | Canal B con / sin |
  |---|---|---|---|---|---|
  | F1 | 5.0e-10 | 2.3e-10 | 3 | 1 | 1 / 5 |
  | F2 | 7.554 | 7.554 | 4 | 0 | 4 / 8 |
  | F5 | 1.1e-13 | 2.3e-13 | 4 | 0 | 3 / 5 |
  | F11 | 12.31 | 4.5e-13 | 4 | 0 | 4 / 7 |

  El Canal A heterogéneo casi nunca supera ΔS > 0.02, la misma limitación
  que el módulo ya documentaba para PSO↔PSO. Al ocupar el 10 % de MaxFES
  por ventana, reduce a la mitad las inyecciones de Canal B; en F11 eso
  coincidió con un error final peor.
- **Pendiente:**
  - Evaluar el efecto con la parrilla de 51 semillas. Si resulta
    contraproducente, es material para la discusión §9.
  - Recalibrar la estimación del middleware en `grid.py`: con ventanas de
    Canal A hay menos ciclos y menos activaciones de Fase 2, y no se
    re-midió.
  - Actualizar el informe: la Fase 3 del LaTeX describe el Canal A inerte
    entre PSO y DE.

### D-11 · La barrera de Wasserstein no se aplicaba — CORREGIDO (θ_W = 0.6, filtro MMD por instancia)

- **Problema:** el orquestador nunca comparaba W₁ contra θ_W. No llamaba a
  `transferencia.ejecutar_ciclo_transferencia()` (la única función que
  abortaba); en el Canal A calculaba W₁ solo para el log, con
  `umbral_wasserstein=0.0` y `transferencia_permitida=True` fijos, y en el
  Canal B registraba W₁ = 0.0 sin calcularla. El comentario "Abort
  (Wasserstein): ya manejado arriba" era falso. Consecuencias: ninguna
  transferencia se abortó nunca (`n_abortos_wasserstein` = 0 en todas las
  corridas) y, como el MMD está centrado y solo compara forma (D-1e), nada
  controlaba la **ubicación** de lo que se transfería.
- **Corrección (2026-10-12):** `Orquestador._evaluar_barrera_wasserstein()`.
  - Se evalúa al entrar a la Fase 3 (antes de Canal A o de Canal B directo)
    y **otra vez antes de escalar de Canal A a Canal B**: las poblaciones se
    mueven durante la ventana de Canal A (10 % de MaxFES).
  - Si W₁ > θ_W: aborta el ciclo sin transferir nada, incrementa el
    contador, registra en `log_fase3` W₁ y θ_W reales con
    `transferencia_permitida=False` y `canal_utilizado="ninguno"`, limpia el
    estado del ciclo y aplica el **cooldown post-ciclo** (sin él se volvería
    a pagar la Fase 2 en el siguiente monitoreo para abortar otra vez, igual
    que el bloqueo por R² < 0).
  - Las entradas de Canal A y Canal B registran ahora W₁ y θ_W reales.
  - Un aborto no cuenta en `n_activaciones_fase3`, que sigue siendo
    `n_canal_a + n_canal_b`.
  - `ejecutar_ciclo_transferencia()` sigue sin usarse en el orquestador (solo
    la usan los tests de Fase 3); no se tocó.
- **Verificación:** `tests/test_fase3_barrera_wasserstein_orquestador.py`
  (nuevo) y `test_determinismo_orquestador` (D-2) pasan.
- **Efecto observado con θ_W = 0.25 (1 semilla, MaxFES = 5·10⁴, W₁ como
  fracción del diámetro en cada evaluación de la barrera):**

  | D | funciones donde la barrera bloquea **todos** los ciclos | fracciones observadas ahí | resto |
  |---|---|---|---|
  | 10 | F6, F7, F8, F9 | 0.31 – 0.63 | 0.00 – 0.27, abortos aislados en F2 y F10 |
  | 20 | F6, F7, F8, F9, F10, F11, F12 | 0.33 – 0.58 | 0.00 – 0.25, 1 aborto en F5 |

  En las funciones bloqueadas la fracción se mantiene **constante** a lo
  largo de toda la corrida (p. ej. F9/D=10 ≈ 0.63, F10/D=20 ≈ 0.57): PSO y
  DE ya convergieron, pero a cuencas distintas. W₁ entre poblaciones
  convergidas mide la distancia entre esas cuencas, no la dispersión, así
  que la barrera nunca se abre. Ese es justamente el escenario en el que
  transferir élite del fuente podría sacar al objetivo de su óptimo local.
- **Filtro MMD corregido (2026-10-13).** El filtro anterior decidía
  todo-o-nada: el grupo de cada candidata era siempre el conjunto completo
  de élites (permutado), así que el MMD era idéntico para todas. Además el
  umbral (3 × P95 del MMD entre mitades de 15 de la fuente) no estaba en la
  escala del valor comparado (5 élites contra 30 del objetivo): aceptaba
  aunque el objetivo fuera 2x más disperso. Ahora:
  - Cada candidata se evalúa sola, como masa puntual:
    MMD²(δ_x, Pt) = 1 − 2·mean_t k(x, t) + mean k(Pt, Pt), con x centrada en
    la media de la fuente y Pt en la suya (sigue comparando forma).
  - Umbral = **máximo** del nulo leave-one-out (misma estadística sobre
    individuos reales). En esta primera versión el nulo, el ancho del
    kernel y el centro salían de la fuente en Fase 2; la segunda corrección
    (abajo) pasó el ancho y el umbral al objetivo.
  - Se descartó una primera variante (contribución leave-one-out al MMD del
    grupo, umbral 3 × P95): con 5 élites no tenía poder, aceptaba un
    outlier evidente y no rechazaba ni con el objetivo 5x más disperso.
  - Verificación sintética (D=20, 200 réplicas): misma distribución → 100 %
    aceptadas; objetivo trasladado +300 → 100 %; una élite outlier → esa 0 %,
    las demás 100 %; objetivo 2x más disperso → 84 %; 5x → 0 %.
  - En corridas reales (barrido de θ_W de abajo, θ_W = ∞) rechazaba ~1/4–1/3
    de las instancias (D=10: 10.4 aprobadas / 3.2 rechazadas por corrida;
    D=20: 8.0 / 4.1), casi siempre lotes completos. Se diagnosticó como un
    artefacto (siguiente punto).
- **Filtro MMD, segunda corrección: referencia en el objetivo
  (2026-10-13).**
  - **Diagnóstico.** Se instrumentaron 72 corridas (12 funciones × D ∈
    {10, 20} × 3 semillas, θ_W = 0.6, MaxFES = 5·10⁴): 178 eventos de
    Canal B, guardando poblaciones, ancho del kernel, umbral y el valor de
    cada élite. Los rechazos totales (48) salían de poblaciones
    colapsadas, no de élites incompatibles:

    | régimen | eventos | lotes aprobados enteros / rechazados enteros |
    |---|---|---|
    | A. fuente colapsada (dispersión del objetivo > 10 × ancho del kernel) | 66 | 29 / 37 |
    | B. objetivo colapsado (dispersión del objetivo < ancho / 10) | 21 | 12 / 5 |
    | C. escalas comparables | 91 | 82 / 6 |

    En A el ancho, tomado de la fuente, quedaba en ~10⁻³: el kernel no
    "veía" al objetivo, todas las élites daban el mismo valor (1 + 1/30 =
    1.033) y la decisión dependía solo de si a la fuente le quedaba algún
    rezagado que inflara el umbral. No es un caso borde: la fuente es el
    algoritmo que convergió.
  - **Corrección.** La pregunta del filtro es "¿la instancia es típica del
    objetivo?", así que el ancho del kernel (mediana intra-objetivo) y el
    umbral (máximo de MMD²(δ_t, Pt sin t) sobre los individuos t del
    objetivo) salen ahora **del objetivo**, calculados en Canal B. Se
    rechaza una élite solo si es más atípica que cualquier miembro real
    del objetivo. Solo el centro de la fuente, para centrar las élites, se
    sigue fijando en Fase 2 (`ReferenciaMMD.centro`): las élites salen de
    esa fuente. La razón original para precalcular el umbral en Fase 2 (la
    fuente se mueve antes del Canal B) deja de aplicar porque la fuente ya
    no interviene.
  - **Objetivo colapsado.** Con el objetivo en un punto no hay forma con la
    que comparar y cualquier élite resulta atípica (en B el objetivo medía
    10⁻¹⁰–10⁻⁴ del diámetro y las élites estaban a 10⁻³–10⁻¹). Si la
    dispersión del objetivo (RMS de la distancia a su centroide) es menor
    que `FRACCION_COLAPSO_MMD` = 10⁻³ × diámetro, el filtro se declara **no
    aplicable**: aprueba todo (la ubicación ya la controló Wasserstein), el
    log de Fase 3 lo marca con `mmd_aplicable = False` y el resumen cuenta
    `n_canal_b_mmd_no_aplicable`. Rechazar ahí bloqueaba el Canal B justo
    con el objetivo estancado.
  - **Alternativas evaluadas offline sobre los 178 eventos** (lotes
    aprobados enteros / rechazados enteros por régimen):

    | variante | A | B | C |
    |---|---|---|---|
    | referencia en la fuente (anterior) | 29 / 37 | 12 / 5 | 82 / 6 |
    | **V1: referencia en el objetivo (adoptada)** | **64 / 2** | 5 / 16 | **80 / 7** |
    | V2: V1 + ancho mínimo = 1 % del diámetro | 63 / 3 | 2 / 19 | 76 / 12 |
    | V3: V1 + ancho = máx(objetivo, fuente) | 64 / 2 | 0 / 21 | 72 / 17 |

    V1 arregla A sin cambiar C y no agrega parámetros. V2 y V3 distorsionan
    C y empeoran B, así que B se resolvió con la regla de no aplicabilidad.
  - **Umbral de colapso.** La dispersión relativa de los objetivos no tiene
    un hueco limpio: el grueso está entre 10⁻² y 1 (119/178) y hay una cola
    hasta 10⁻¹². Entre 10⁻³ y 10⁻² hay 9/178 eventos, así que mover el corte
    en ese rango cambia poco. Valor **provisional**.
  - **Verificación.** `test_barrera_seguridad` CASO 3 reescrito (a–f, ver
    D-1e) pasa. Barrido θ_W ∈ {0.6, ∞} repetido con el filtro nuevo (240
    corridas, todas `ok`):
    - Por corrida, D=10: 12.6 aprobadas / 0.8 rechazadas (antes 9.9 / 3.2);
      D=20: 10.9 / 1.0 (antes 7.7 / 4.1).
    - Lotes de Canal B: rechazo total 5 % (D=10) y 8 % (D=20); mixtos 1–3 %;
      **filtro no aplicable 27–35 %**: el objetivo colapsado es frecuente,
      hay que mencionarlo en §9.
    - Error contra el filtro anterior (θ_W = 0.6, 120 pares f × D ×
      semilla): 23 mejores, 5 peores, 92 empates (Wilcoxon p < 0.001). Más
      claro en F11: D=10 mediana 0.0 contra 75.6; D=20 300.1 contra 471.6.
    - θ_W = 0.6 contra ∞ con el filtro nuevo: 0 mejores, 2 peores, 118
      empates (p = 0.5) — la decisión de θ_W se mantiene.
  - Los lotes mixtos siguen siendo raros por construcción: las 5 élites
    son las mejores de una fuente casi convergida y están juntas entre sí;
    el filtro solo las separa cuando se dispersan a la escala del objetivo
    (en el CASO 3, un objetivo a la mitad de la dispersión de la fuente da
    2/5 aprobadas).
- **θ_W recalibrado a 0.6 (decisión del usuario, 2026-10-13).** Barrido
  θ_W ∈ {0.25, 0.4, 0.6, ∞}, 12 funciones × D ∈ {10, 20} × 5 semillas,
  MaxFES = 5·10⁴, con la primera versión del filtro por instancia (más PSO
  y DE solos como referencia; 720 corridas, todas `ok`; θ_W ∈ {0.6, ∞} se
  repitió con la segunda versión, ver arriba):

  | θ_W | funciones sin ninguna transferencia (D=10 / D=20) | contra θ_W = ∞ (120 pares f × D × semilla) |
  |---|---|---|
  | 0.25 | F2, F6–F10 / F1, F6–F10, F12 | 5 mejores, **20 peores**, 95 empates (Wilcoxon p = 0.02) |
  | 0.4 | F6 / F6, F7, F8, F10, F12 | 1 mejor, 5 peores (p = 0.44) |
  | 0.6 | ninguna / ninguna | 0 mejores, 2 peores (p = 0.50) |
  | ∞ | ninguna / ninguna | — |

  - Con 0.25 bloquear nunca protegió de una transferencia negativa medible
    y empeoró significativamente (caso más claro: F11/D=10, error mediano
    118.7 contra 0.0).
  - 0.6 no se distingue de ∞ (0.15 abortos por corrida) y conserva la
    barrera de la ecuación 7 como red de seguridad para casos extremos, sin
    tener que justificar la eliminación de un componente.
  - **Limitaciones (declarar en §9):** 5 semillas, solo el tier más bajo de
    MaxFES, y la calibración usa el mismo benchmark que se reporta.
  - Hallazgo aparte, no atribuible a la barrera (igual con todo θ_W): el
    propuesto queda muy por detrás de DE solo en F1/D=10 (133 contra 0.04),
    F8/D=10 y F1, F6, F10/D=20. Revisar junto con D-10 (Canal A heterogéneo
    y reparto del presupuesto).

### D-12 · Tests alineados con los criterios que aplica el código — RESUELTO

- **Problema:** varios tests verificaban criterios que el código no aplica,
  o escenarios calibrados con el evaluador opfunu:
  - `test_fase2_extraccion.py` exigía **R² > 0.7** (plan metodológico
    original), pero el middleware solo bloquea la transferencia con
    **R² < 0** (`extraer_conocimiento`: `bloqueada = r2 < 0.0`). Con el
    evaluador oficial el R² del escenario es 0.659 y el test fallaba sin
    que el middleware hiciera nada distinto.
  - `test_fase3_transferencia.py` y `test_fase3_transferencia_exitosa.py`
    ignoraban `transferencia_bloqueada_por_r2`: con R² < 0 la Fase 2
    devuelve élites de ceros, y los tests las habrían inyectado. El
    orquestador no ejecuta el ciclo en ese caso.
  - `test_fase3_transferencia.py`, Criterio 4 ("abortos concentrados en
    iteraciones tempranas"), se aprobaba solo cuando no había abortos (con
    θ_W = 0.6 no los hay): no verificaba nada. El docstring decía que el
    ciclo se activa "desde la iteración ~80" (se activa en la 51) e
    imprimía la evolución "tras escalar a Canal B (it=116)" fija (el
    escalado real ocurre en la 81).
  - `test_fase3_transferencia_exitosa.py` usaba F1/D=10, donde con el
    Zakharov oficial la fuente "bien afinada" termina peor (6901 contra
    2555) y no se activa ningún ciclo. Además medía la mejora del objetivo
    desde la iteración 30, antes de cualquier intervención.
  - Comentarios de `middleware/extraccion.py` decían "True cuando R²≤0" y
    citaban el R² > 0.7 como criterio de la función.
- **Corrección (2026-10-13):**
  - `test_fase2_extraccion.py`: criterio **R² ≥ 0** y
    `transferencia_bloqueada_por_r2 == (R² < 0)`. El R² se sigue
    imprimiendo como referencia.
  - Ambos tests de Fase 3: si la Fase 2 quedó bloqueada por R² < 0, el
    ciclo no se ejecuta (como el orquestador) y se cuenta.
  - `test_fase3_transferencia.py`, Criterio 4: cada ciclo se aborta **si y
    solo si W₁ > θ_W** (ecuación 7, θ_W = FRACCION_UMBRAL_WASSERSTEIN ×
    diámetro). Docstring corregido; la evolución del fitness se imprime
    desde el escalado real.
  - `test_fase3_transferencia_exitosa.py`: refactorizado a
    `correr_escenario(f, d)`. Búsqueda en las 12 funciones con D ∈ {10,
    20}: premisa y criterios se cumplen en 16/24 casos (fallan F1, F4, F6,
    F10/D=10 y F1, F4, F5, F6, F9/D=20 sin ciclos; F3/D=10 con un único
    ciclo al final). Escenario elegido: **F2/D=10** (fuente 433 contra
    objetivo 469 al final, primer ciclo en la iteración 51, 30 iteraciones
    de Canal A, 600 instancias inyectadas, sin bloqueos por R² ni rechazos
    del MMD). La mejora del objetivo se mide desde el primer ciclo.
  - Comentarios de `extraccion.py` corregidos.
- **Verificación:** la suite completa (15 scripts) pasa sin ❌.

---

## C. Cómo consumir esta configuración desde el harness

- Punto de entrada: `experimentos/corrida.py` (`correr_corrida`) y
  `experimentos/grid.py --algoritmo middleware,pso,de,cmaes,madde,nlshadersp`
  (default: los seis). No re-implementar el armado de PSO+DE+Orquestador ni
  los bucles de PSO/DE solos — ya están en `corrida.py`
  (`_correr_middleware`, `_correr_algoritmo_solo`, `_correr_competidor`).
- `corrida.py` ya aplica la config de A: pop 30, `_PSO_KW` / `_DE_KW`.
  - `"middleware"`: `fraccion_presupuesto=0.5` (default, exacto — D-2), PSO
    y DE reciben dos sub-semillas independientes vía
    `np.random.SeedSequence(semilla).spawn(2)`, `Orquestador(...,
    silencioso=True)` con el planificador cooperativo determinista.
  - `"pso"` / `"de"` (D-4): corren solos, `fraccion_presupuesto=1.0`, con la
    semilla de la corrida directa (sin spawn — no hay un segundo algoritmo
    con quien repartir), sin orquestador (bucle secuencial simple).
  - `"cmaes"` / `"madde"` / `"nlshadersp"` (D-8): `modulo.correr(problema,
    max_fes, semilla)` de `experimentos/competidores/`, con la semilla de la
    corrida directa y la configuración de §A.8.
- Cada corrida usa una instancia nueva de `ProblemaCEC2022` +
  `configurar_checkpoints(max_fes)`.
- Salida: fila §11.2 (todos) + fila §11.3 (diagnósticos del middleware,
  `None` para los demás), un Parquet por corrida; `consolidar.py` los une.
