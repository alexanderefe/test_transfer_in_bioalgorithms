# Configuración experimental — parámetros congelados y registro de decisiones

> **Propósito.** El setup experimental de la tesis (§4) exige **una única
> configuración de parámetros** para el algoritmo propuesto, sin ajuste por
> valor de MaxFES ni por problema. Este documento fija esa configuración y
> deja registro de las decisiones tomadas para congelarla, incluyendo las
> que quedan pendientes de una revisión más profunda.
>
> Última actualización: 2026-09-01.

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
| Sobrepaso tolerado | ≤ `n_individuos` por hilo | `HiloAlgoritmo._presupuesto_agotado` |
| `fraccion_presupuesto` (cuota PSO para su inercia) | **0.7** | `bioalgorithms/base.py` — ver decisión D-1b |
| `frecuencia_monitoreo_fes` (default) | `max(pop_A+pop_B, MaxFES // 500)` | `middleware/orquestador.py` |
| `max_epochs_fastshap` (producción) | 50 | `middleware/extraccion.py` (`EPOCHS_EXPLAINER`) |

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
| Épocas del explainer FastSHAP | 50 |
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

### D-1b · Cuota de presupuesto de PSO (`fraccion_presupuesto`)

- **Estado previo:** 0.5 (supuesto de reparto 50/50 entre PSO y DE).
- **Observación:** el reparto real medido en el par PSO(fuente)/DE(objetivo)
  sobre F11/D=20 fue **~70/30 a favor de PSO** (70.7 %, 70.5 %, 73.5 % en
  tres corridas). Causa: las generaciones de PSO son vectorizadas y más
  rápidas que el bucle por individuo de DE, así que PSO completa más
  evaluaciones bajo el mismo presupuesto compartido y con GIL.
- **Decisión (2026-09-01):** fijar `fraccion_presupuesto = 0.7`
  (valor fijo medido). Con esto la inercia de PSO llega a `w_min` cerca del
  final real de su corrida, no al ~70 % como con 0.5.
  Aplicado en `bioalgorithms/base.py`, `pso.py`, `de.py`.
- **Pendiente:** el split exacto varía por función y dimensión (y cambiaría
  si se sustituye el par de algoritmos base). Alternativa a evaluar más
  adelante: **proyección adaptativa** — PSO estima su presupuesto final como
  `fes_propias / (fes_globales / MaxFES)` y se auto-corrige por corrida, sin
  ningún valor fijo. Medir la dispersión del split en el subconjunto de
  D-1a y decidir si 0.7 fijo es suficiente o conviene la proyección.

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

---

## C. Cómo consumir esta configuración desde el harness

- Instanciar PSO/DE con los valores de A.1 (pasar `F=0.2, CR=0.3` a DE
  explícitamente; los demás son defaults del código).
- Pasar `max_fes` a los algoritmos y `problema=` + `max_fes=` al
  `Orquestador`. No tocar `fraccion_presupuesto` (queda en 0.7 por default).
- Para cada corrida: instancia nueva de `ProblemaCEC2022` (o
  `problema.reiniciar_fes()`) y semilla nueva de la lista fija.
- Registrar por corrida: `resumen.mejor_fitness_sistema`,
  `resumen.error_sistema`, `resumen.fes_consumidas_total`, y las métricas
  de activación de Fases 2/3.
