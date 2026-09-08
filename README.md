# Middleware Adaptativo de Transferencia de Conocimiento entre Metaheurísticas Heterogéneas

**Tesis de Ingeniería Civil Informática — Universidad de Valparaíso**
**Autor:** Alexander Nicolas Farias Cifuentes
**Profesor Guía:** Rodrigo Olivares Órdenes

> Última actualización de este README: 2026-09-04. Si continúas este
> proyecto desde una sesión/ventana de IA nueva, empieza por **"Estado
> Actual del Proyecto"** más abajo — tiene el resumen de qué está hecho, qué
> falta y en qué orden, con enlaces a `docs/` para el detalle de cada decisión.

---

## Descripción

Middleware adaptativo que opera sobre dos algoritmos bioinspirados ejecutados de forma cooperativa (planificador determinista round-robin, D-2), transfiriendo conocimiento del algoritmo con mejor desempeño (**fuente**) hacia el que presenta estancamiento (**objetivo**), de forma online y sin modificar la lógica interna de los algoritmos.

El sistema detecta estancamiento mediante un Score compuesto, extrae conocimiento relevante con FastSHAP sobre un modelo subrogado XGBoost, y lo transfiere gradualmente usando dos canales jerárquicos protegidos por barreras de seguridad (Wasserstein + MMD).

---

## Estructura del Proyecto

```
Fase 3/
├── bioalgorithms/
│   ├── base.py              # AlgoritmoBioinspirado + EstadoIteracion; contador fes_propias;
│   │                        #   parámetros max_fes / fraccion_presupuesto
│   ├── pso.py               # PSO; inercia decrece contra la cuota MaxFES·0.5 (o iteraciones)
│   └── de.py                # DE/rand/1/bin (F=0.2, CR=0.3 por defecto)
│
├── problems/
│   ├── cec2022_wrapper.py   # ProblemaCEC2022: 12 funciones CEC 2022, D∈{10,20};
│   │                        #   contador FES agregado (.fes) + óptimo conocido + error()
│   └── cec2022_numba.py     # Backend compilado (Numba, D-5) — default; opfunu = referencia
│
├── middleware/
│   ├── deteccion.py         # Fase 1: Score S = 0.3·FIR + 0.7·DWD; warm-up 15% por MaxFES
│   ├── extraccion.py        # Fase 2: XGBoost + FastSHAP + selección élite + umbral MMD
│   ├── xgboost_wrapper.py   # Wrapper XGBoost como nn.Module (para MarginalImputer)
│   ├── barrera_seguridad.py # Wasserstein (scipy.linprog) + MMD centrado
│   ├── transferencia.py     # Canal A (RMP) + Canal B (inyección élite)
│   └── orquestador.py       # Planificador cooperativo determinista (D-2); parada por MaxFES o iteraciones
│
├── fastshap_lib/            # Código fuente oficial iancovert/fastshap
│   ├── fastshap.py          # FastSHAP + parche compatibilidad PyTorch ≥ 2.x
│   ├── tabular_imputers.py  # MarginalImputer
│   └── ...
│
├── tests/
│   ├── test_fase0_algoritmos_base.py
│   ├── test_fase1_deteccion.py
│   ├── test_fase1_integracion_real.py
│   ├── test_fase2_extraccion.py
│   ├── test_barrera_seguridad.py
│   ├── test_fase3_transferencia.py
│   ├── test_fase3_transferencia_exitosa.py
│   ├── test_orquestador_threading.py          # integración completa — vía iteraciones (clásica)
│   ├── test_orquestador_threading2.py         # ídem, par PSO↔PSO
│   ├── test_orquestador_threading_maxfes.py   # integración completa — vía MaxFES
│   ├── test_determinismo_orquestador.py       # criterio de aceptación de D-2
│   └── test_experimentos.py                   # smoke del harness (middleware/pso/de)
│
├── experimentos/           # harness: semillas fijas, corrida única, parrilla, consolidación
│   ├── semillas.json · semillas.py · generar_semillas.py
│   ├── esquema.py          # esquema §11 (columnas, run_id, builders)
│   ├── corrida.py          # correr_corrida(algoritmo, func, dim, max_fes, semilla)
│   │                       #   algoritmos: "middleware" | "pso" | "de" (D-4, solos)
│   ├── grid.py             # parrilla resumible + paralela (--algoritmo middleware,pso,de; --dry-run)
│   └── consolidar.py       # resultados/corridas/*.parquet -> resultados/corridas.parquet
│
├── docs/
│   ├── setup_experimental.md            # protocolo de evaluación (adaptado del PDF a CEC 2022) + §11 datos
│   ├── configuracion_experimental.md    # config única congelada + registro de decisiones (D-1a…D-4)
│   └── competidores.md                  # roster candidato (27) — DESCARTADO (D-4), archivado
│
├── .gitignore              # __pycache__/, entornos, cachés de test, logs de prueba
├── requirements.txt
└── README.md
```

---

## Instalación

**Requisitos:** Python 3.14+ (desarrollado en Python 3.14.6)

```bash
pip install -r requirements.txt
# o, explícito (comillas obligatorias: sin ellas la shell interpreta '>' como redirección):
pip install "setuptools<82" "numpy>=1.24" "scipy>=1.10" "opfunu>=1.0.0" "torch>=2.0" "xgboost>=2.0" "scikit-learn>=1.3" "tqdm>=4.65"
```

> **Nota:** `fastshap` no está disponible en PyPI para Python 3.14. Se incluye como código fuente en `fastshap_lib/` con un parche de compatibilidad para PyTorch ≥ 2.0 ya aplicado. No instalar por pip.

> **Nota:** La librería POT (Python Optimal Transport) fue descartada por incompatibilidad con Python 3.14 en Windows. La distancia de Wasserstein se calcula con `scipy.optimize.linprog` (solver HiGHS), produciendo resultados idénticos.

---

## Uso Rápido

El punto de entrada principal es el test de integración completa. El
criterio de parada es **MaxFES** (evaluaciones de la función objetivo,
agregadas sobre los dos algoritmos), tal como exige el setup experimental:

```python
from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from bioalgorithms.de import DE
from middleware.orquestador import Orquestador

problema = ProblemaCEC2022(numero_funcion=11, ndim=20)
MAX_FES = 120_000            # presupuesto total A+B (5e3 / 5e4 / 5e5 / 5e6 en el protocolo)

# semilla SIEMPRE como keyword; max_iteraciones se omite en la vía MaxFES
alg_a = PSO(problema, 30, problema.ndim, problema.limites, semilla=42,
            w_max=0.9, w_min=0.4, c1=2.0, c2=2.0,
            max_fes=MAX_FES)            # inercia decrece contra MAX_FES * 0.5 (su cuota exacta, D-2)

alg_b = DE(problema, 30, problema.ndim, problema.limites, semilla=500,
           F=0.2, CR=0.3, max_fes=MAX_FES)

orq = Orquestador(
    algoritmo_a=alg_a,
    algoritmo_b=alg_b,
    limites=problema.limites,
    problema=problema,                  # provee el contador .fes compartido
    max_fes=MAX_FES,
    frecuencia_monitoreo_fes=600,       # verificar Fase 1 cada 600 evaluaciones
    directorio_log="logs",
    nombre_log="experimento",
    max_epochs_fastshap=10,             # producción: 20 (default, D-3); 3-5 solo para tests
)

resumen = orq.ejecutar()
# resumen.mejor_fitness_sistema  → min(fitness_A, fitness_B) al alcanzar MaxFES
# resumen.error_sistema          → mejor_fitness_sistema - f(x*); 0 = óptimo alcanzado
# resumen.fes_consumidas_total   → evaluaciones realmente gastadas (≈ MaxFES ± pop)
```

### Error respecto al óptimo (métrica de comparación)

`ProblemaCEC2022` conoce el valor de la función en el óptimo global, `f(x*)`
(tabla `OPTIMO_GLOBAL_CEC2022`: F1→300, F2→400, …, F11→2600, F12→2700; igual
para D=10 y D=20). El error de convergencia es `f_obtenido − f(x*)` (≥ 0, y
`→ 0` cuanto más cerca del óptimo):

```python
problema.optimo_global            # f(x*), p. ej. 2600.0 para F11
problema.error(2637.4)            # -> 37.4   (escalar o array de numpy)
problema.error(vals, aplicar_umbral=True)   # colapsa errores < 1e-8 a 0 (convención CEC)
```

El `Orquestador` lo calcula automáticamente al terminar (`resumen.error_sistema`,
`error_algoritmo_a/b`, `optimo_conocido`) siempre que se le pase `problema=`.

El orquestador imprime el log en tiempo real y guarda un archivo JSON en `logs/` al finalizar.

> **Vía clásica (compatibilidad):** pasar `n_iteraciones=` (a los algoritmos como
> 5.º posicional y al `Orquestador`) en vez de `max_fes` + `problema`. Se
> conserva para los tests de Fase 0–3; los experimentos usan la vía MaxFES.

### Contabilidad de evaluaciones (FES)

- El contador vive en `ProblemaCEC2022` (`problema.fes`, thread-safe). Como el
  mismo objeto `problema` se entrega a los dos algoritmos, cuenta el total
  **agregado** A+B. `problema.reiniciar_fes()` lo pone a cero para corridas
  independientes.
- Cada algoritmo lleva además su propio `algoritmo.fes_propias`.
- El middleware **no consume FES**: Fase 2 entrena sobre historial ya evaluado
  y Canal B inyecta instancias con su fitness heredado (sin re-evaluar).
- **Sobrepaso:** ningún algoritmo arranca una generación que no cabe en el
  presupuesto (`fes + n_individuos > MaxFES`); el sobrepaso máximo es ~`pop_size`
  por algoritmo (convención CEC).

---

## Arquitectura — Ciclo de Transferencia

```
Algoritmo A (fuente) ──┐    Parada: MaxFES = evaluaciones agregadas A+B (contador en el problema)
                       ├──► Orquestador (planificador cooperativo determinista, D-2)
Algoritmo B (objetivo) ─┘   Round-robin: una generación de A, una de B, alternando
                         │
                    ① Fase 1 — Detección
                         │  S = 0.3·FIR + 0.7·DWD
                         │  activo tras 15% de MaxFES; S < 0.2 ∧ fuente_óptimo → activar ciclo
                         │
                    ② Fase 2 — Extracción (síncrona: no hay nada más corriendo mientras tanto)
                         │  XGBoost → R² (bloquear si R² < 0)
                         │  FastSHAP → valores Shapley ϕ
                         │  Élite: percentil 20% fitness + ranking ϕ
                         │  Escape: hiperparámetros en max(Δfitness)
                         │  Umbral MMD precalculado con población fuente actual
                         │
                    ③ Fase 3 — Transferencia
                         │
                         ├─ Barrera Wasserstein (dura, síncrona)
                         │  W₁ > θ_W → ABORTAR
                         │
                         ├─ Canal A — RMP paramétrico
                         │  RMP: 0.05→0.6, incremento 0.5·ΔS
                         │  A y B siguen alternando generaciones por ventana = 10% de MaxFES
                         │  Si payload vacío (PSO↔DE sin claves comunes) → Canal B directo
                         │  Evaluar ΔS > 0.02 → éxito / fracaso
                         │
                         └─ Canal B — Inyección élite (síncrona, sin generaciones adicionales)
                            Filtro MMD (umbral precalculado en Fase 2)
                            Sustituye peores individuos del objetivo
                            Evaluación inmediata, sin iterar
                            → CERRAR CICLO + cooldown = 10% de MaxFES
```

---

## Decisiones de Diseño Críticas

| Componente | Decisión | Estado |
|---|---|---|
| Criterio de parada | MaxFES agregado A+B (contador en `ProblemaCEC2022`); sobrepaso ≤ pop_size | Definitiva |
| Métrica de comparación | error = `f_obtenido − f(x*)`; `f(x*)` de tabla `OPTIMO_GLOBAL_CEC2022` validada contra opfunu | Definitiva |
| Progreso inercia PSO | contra su cuota `MaxFES · 0.5` — exacto, no medido (reparto 50/50 por el planificador round-robin) | Definitiva — D-1b resuelto junto con D-2 |
| Determinismo / reproducibilidad | Planificador cooperativo (no threading) + `torch.manual_seed(0)` en Fase 2 + parche de semilla en `fastshap_lib/utils.py` (`ShapleySampler`) | Definitiva — D-2 resuelto, ver `docs/configuracion_experimental.md` |
| FIR | `1 - exp(-3·max(0, f_{t-20} - f_t) / (\|f_{t-20}\| + ε))` — ventana en generaciones del objetivo | Definitiva |
| Pesos Score | W_FIR=0.3, W_DWD=0.7, W_HDF=0.0 | Provisional — validar por sensibilidad, ver `docs/configuracion_experimental.md` (D-1a) |
| Umbral estancamiento | S < 0.2 tras 15% de **MaxFES** consumido | Definitiva |
| Ventana historial Fase 2 | Últimas 50 generaciones de la fuente | Definitiva |
| Criterio élite | Percentil 20% fitness + ranking Shapley | Definitiva |
| RMP inicial / tope | 0.05 / 0.6 | Definitiva |
| Ventana Canal A | 10% de MaxFES (evaluaciones agregadas A+B) | Definitiva |
| θ_W (Wasserstein) | 25% del diámetro máximo del espacio | Definitiva |
| Gamma MMD | Solo distancias intra-grupo | Definitiva |
| Centrado MMD | Centrar poblaciones en su media antes del kernel RBF | Definitiva |
| Umbral MMD | 3×P95 ruido muestreo fuente; precalculado en Fase 2 | Definitiva |
| Bloqueo R² | Estrictamente negativo (R² < 0) | Definitiva |
| Canal A inerte | Escalar a Canal B directo si payload vacío | Definitiva |
| Cooldown post-ciclo | 10% de MaxFES (igual que ventana Canal A) | Definitiva |
| Planificador | Cooperativo determinista (round-robin, un solo hilo) — reemplazó al threading pseudo-paralelo (D-2) | Definitiva |
| FIR post-transferencia | Historial recortado desde última transferencia | Definitiva |
| Cierre de ciclo | Criterio mínimo (objetivo deja de estar estancado); Fase 4 solo se aborda en la redacción | Provisional |

---

## Bugs Resueltos Relevantes

1. ~~**Pausa threading falsa**~~: `threading.Event.wait()` retorna inmediatamente si el evento ya está SET. Solución de la época: bucle `while self.evento_pausa.is_set(): sleep(0.005)`. **Todo el mecanismo de pausa/hilos se eliminó en D-2** (planificador cooperativo determinista) — este bug y su parche ya no existen en el código, se dejan como registro histórico.

2. ~~**Timeout 30s en verificaciones**~~: cuando los algoritmos terminaban sus iteraciones durante una pausa larga (Fase 2), el orquestador esperaba confirmación de pausa de hilos ya muertos. Solución de la época: verificar `hilo.terminado` antes de `evento_pausado.wait()`. También obsoleto tras D-2 (sin hilos, sin pausas que confirmar).

3. **Umbral MMD desfasado**: el umbral recalculado en Canal B (cuando el fuente ya convergió más) difería del umbral en el momento de extracción. Solución: precalcular y guardar el umbral en `ResultadoExtraccion.umbral_mmd_precalculado`.

4. **Score contaminado post-transferencia**: FIR comparaba fitness actual contra valores anteriores a la transferencia. Solución: recortar historial desde `_iteracion_ultima_transferencia`.

5. **MMD no invariante a traslación**: kernel RBF puro mide separación geométrica cuando las poblaciones están en zonas distintas. Solución: centrar cada población en su media antes del kernel.

6. **Canal A inerte (PSO↔DE)**: PSO {w,c1,c2} y DE {F,CR} no tienen claves en común → payload vacío → el orquestador escala directamente a Canal B sin esperar la ventana de evaluación.

7. **Ciclos en ráfaga post-Canal B**: Score sigue bajo inmediatamente tras el cierre, re-detecta estancamiento. Solución: cooldown de 10% del presupuesto total (MaxFES).

8. **R²=0 bloqueaba transferencias válidas**: varianza de fitness ≈ 0 producía R²=0.0 exacto por la guarda `ss_tot > 0`. Solución: bloquear solo si R² < 0 estrictamente.

9. **`FastSHAP.train()` crasheaba con `AttributeError: 'NoneType' object has no attribute 'parameters'`** (`fastshap_lib/fastshap.py:440`), detectado en la parrilla completa: 17/14 688 corridas fallaron, todas en F8 (rango dinámico muy grande — valores ~10⁶). Causa: `best_model` arranca en `None` y solo se asigna si la pérdida de validación mejora en alguna época; si sale `NaN` en todas (posible con F8), la comparación `NaN < best_loss` es siempre `False` y `best_model` nunca se asigna. Solución: si `best_model is None` al terminar el entrenamiento, conservar el `explainer` tal como quedó en la última época en vez de copiarle pesos de un "mejor modelo" que nunca existió — mismo criterio que los otros parches de compatibilidad de ese archivo vendorizado (no altera la lógica del paper). Las 17 corridas se reintentaron tras el parche: las 14 688 corridas de la parrilla completa terminaron en `estado="ok"`.

---

## Escenarios Validados

### Caso 1: PSO fuente → PSO objetivo (transferencia homogénea)
- Canal A activo: hiperparámetros comunes {w, c1, c2}
- MMD aprueba instancias cuando distribuciones son compatibles
- Fitness B mejoró de ~4484 a ~2672 en F11/D=20

### Caso 2: PSO fuente → DE objetivo (transferencia heterogénea)
- Canal A inerte: sin hiperparámetros comunes → Canal B directo
- Validación de barrera Wasserstein + filtro MMD
- Bloqueo automático por R² < 0 cuando fuente convergió

---

## Estado Actual del Proyecto

> **Para retomar en otra sesión/ventana de IA:** este README es la fuente de
> verdad del estado general; el detalle de cada decisión (motivo, alternativas
> consideradas, qué falta) vive en `docs/configuracion_experimental.md`
> (registro D-1a…D-4) y `docs/setup_experimental.md` (protocolo de evaluación
> completo). Léelos antes de tocar `middleware/` o `experimentos/`.

### Completado ✅

**Núcleo del middleware (Fases 0–3):**
- Fase 0: interfaz común (`AlgoritmoBioinspirado`), PSO, DE, wrapper CEC 2022.
- Fase 1: detección de estancamiento (Score S), roles dinámicos fuente/objetivo, historial FIR recortado post-transferencia.
- Fase 2: XGBoost (modelo subrogado) + FastSHAP oficial (iancovert/fastshap) + selección élite + parámetros de escape + umbral MMD precalculado.
- Fase 3: barrera Wasserstein + Canal A (RMP paramétrico) + Canal B (inyección de élite filtrada por MMD).
- Tests unitarios de cada fase (`tests/test_fase0…test_fase3_transferencia_exitosa.py`).

**Criterio de parada y métrica (MaxFES):**
- Contador de evaluaciones agregado A+B en `ProblemaCEC2022.fes` (thread-safe); `error()` y tabla `OPTIMO_GLOBAL_CEC2022` para la métrica de comparación (`f_obtenido − f(x*)`).
- Warm-up 15%, ventana Canal A (10%), cooldown y decaimiento de inercia de PSO expresados como fracción de MaxFES. Vía clásica por iteraciones conservada para compatibilidad con los tests de Fase 0–3.
- Traza de convergencia: `configurar_checkpoints()` / `error_checkpoints()` — best-so-far en 14 fracciones de FES.

**Reproducibilidad — planificador cooperativo determinista (D-2, RESUELTO):**
- Se reemplazó el threading pseudo-paralelo por un bucle único (round-robin A/B, sin locks/eventos/sleeps). Misma semilla → resultado **exactamente idéntico** (`tests/test_determinismo_orquestador.py`).
- De paso se cerraron dos fuentes de no-determinismo independientes del threading: RNG global de `torch` sin sembrar (`middleware/extraccion.py`) y `ShapleySampler.rng` sin semilla en el `fastshap_lib` vendorizado.
- Efecto colateral: el reparto de FES entre PSO y DE pasa a ser **exactamente 50/50** (antes ~70/30, emergente del scheduling) → `fraccion_presupuesto=0.5` es ahora el valor correcto, no una aproximación (D-1b resuelto).

**Configuración única congelada** (`docs/configuracion_experimental.md`):
- Tabla completa de parámetros para las 8 configuraciones experimentales (2 dim × 4 MaxFES).
- Registro de decisiones D-1a (pesos Score, pendiente de validar), D-1b (✅ resuelto), D-1d (comportamiento con MaxFES bajo, aceptado), D-1e (test de barrera rojo, diagnosticado y diferido), D-2 (✅ resuelto), D-3 (costo de FastSHAP, aplicado), D-4 (alcance de competidores, decidido).

**Harness experimental** (`experimentos/`, ver `docs/setup_experimental.md` §11):
- `semillas.json` (51 semillas fijas) + `generar_semillas.py` + `semillas.py`.
- `esquema.py`: esquema de datos único (`COLUMNAS_CORRIDA`, `COLUMNAS_PROPUESTO`), `run_id()`, builders de fila.
- `corrida.py`: `correr_corrida(algoritmo, funcion, dim, max_fes, idx_semilla)` — implementa `"middleware"` (propuesto), `"pso"` y `"de"` (solos, D-4). Nunca lanza excepción: fallos quedan registrados con `estado="fallo"`.
- `grid.py`: recorre subconjuntos arbitrarios, resumible (salta corridas ya hechas), paralelo por proceso, `--dry-run` con estimación de tiempo calibrada por función.
- `consolidar.py`: une los Parquet por corrida en `resultados/corridas.parquet` / `corridas_propuesto.parquet`.
- Smoke tests: `tests/test_experimentos.py`.

### Pendiente ⏳ — en orden recomendado

**Alcance ya fijado (no reabrir sin motivo):** evaluación **solo sobre CEC 2022**
(12 funciones, D∈{10,20}) — no se amplía a CEC 2014/2017 ni D=50. La Fase 4
(cierre formal del ciclo, con ΔScore estructurado y auditoría SHAP) **no se
implementa en código**: se aborda únicamente en la redacción de la tesis.
Competidores = **solo PSO y DE corriendo solos** (D-4), no un roster externo.

**Presupuesto de cómputo — decidido (2026-09-04):** se acepta el status quo
post-D5, **~50-52 h (~2.1-2.2 días) en 5 workers** (`experimentos/grid.py
--dry-run`), bajando de los ~183 h (~7.6 días) previos gracias al evaluador
CEC2022 compilado con Numba (D-5, sin tocar la especificación, paridad
numérica validada). Este número está **validado con una corrida piloto real
de 143 corridas** (no es solo teórico): el piloto reveló que, además del
evaluador, dos costos de Python estaban mal calibrados desde antes de D-5 y
recién se hicieron visibles al dejar de estar tapados por el costo de
opfunu — el overhead del bucle de PSO/DE (antes sobreestimado) y sobre todo
el costo real de Fase 2 del middleware (FastSHAP+XGBoost, antes subestimado
~5×). Con las tres constantes recalibradas contra datos reales, el cuello
de botella restante es el `for`-loop de `DE.un_paso()` — **se decide NO
vectorizarlo**: el riesgo de reproducibilidad (reordena las llamadas al
RNG, mismo tipo de problema que D-2, y podría alterar sutilmente la
distribución de la mutación DE/rand/1 que Fase 0 validó como generadora de
estancamiento confiable) pesa más que ganar unas horas adicionales. Ya no
bloquea el lanzamiento de la parrilla — ver `docs/configuracion_experimental.md`
D-5 para el detalle de la recalibración.

**D-1a — pesos del Score — decidido (2026-09-04):** se conservan
`W_FIR=0.3` / `W_DWD=0.7` por justificación conceptual (la diversidad
poblacional, DWD, es una señal más informativa de estancamiento real que
la tasa de mejora de fitness, FIR — un algoritmo puede dejar de mejorar
momentáneamente sin haber perdido diversidad), **sin ejecutar el barrido
empírico de sensibilidad** que estaba planeado. Ya no bloquea el
lanzamiento — ver `docs/configuracion_experimental.md` D-1a.

**Parrilla completa — lanzada y terminada (2026-09-08).** `experimentos/grid.py
--algoritmo middleware,pso,de` corrió las 14 688 corridas en **71.30 h**
(por encima de la estimación de 50-52 h — el estimador de `grid.py` sigue
siendo aproximado, no una cota dura). Terminó con 14 528 `ok` / 17 `fallo`;
los 17 fallos eran el mismo bug (`fastshap_lib/fastshap.py`, `best_model`
quedaba en `None` cuando la pérdida de validación salía `NaN` en todas las
épocas — específico de F8 por su rango dinámico grande, ver "Bugs Resueltos
Relevantes" #9). Con el parche aplicado se reintentaron las 17 y
`resultados/corridas.parquet` / `corridas_propuesto.parquet` quedaron con
**14 688 / 4 896 filas, 100% `estado="ok"`**.

1. **Construir `analisis/`** (no existe todavía — siguiente paso): rankings
   con empates a 10⁻⁸ (N=3: propuesto/PSO/DE, ver nota abajo), Friedman +
   Shaffer, diagramas de Diferencia Crítica (Nemenyi), Wilcoxon
   better/equal/worse, desviación de rangos, figuras. El esquema de datos
   que consume está en `docs/setup_experimental.md` §11 y ya está
   disponible en `resultados/corridas.parquet` / `corridas_propuesto.parquet`.
2. **Redacción**: Fase 4 como apartado separado (fuera del código); sección
   de resultados con la discusión obligatoria del setup (§9) — en qué MaxFES
   destaca el propuesto, cómo varía entre dimensionalidades, qué operadores
   son beneficiosos/contraproducentes según el presupuesto (D-1d es
   material directo para esto), robustez frente a PSO/DE solos.

**Diferido, no bloqueante:**
- D-1e — `tests/test_barrera_seguridad.py` CASO 3 falla por una expectativa
  desactualizada del filtro MMD centrado (no es un bug del código, ver
  diagnóstico en `docs/configuracion_experimental.md`). Revisar en algún
  momento, no urge.
- Ablation Study: descartado por tiempo, queda como trabajo futuro.

**Nota sobre N=3 en los rankings:** con solo propuesto/PSO/DE, los rankings,
Friedman y los diagramas CD de Nemenyi son válidos pero menos informativos
que con un panel amplio de competidores — reconocerlo explícitamente en la
redacción (ya anticipado en `docs/setup_experimental.md` §3).

---

## Log de Ejecución — Formato de Referencia

> El formato del log no cambió con D-2. El ejemplo de abajo se capturó
> **antes** del planificador determinista — por eso "FES consumidas A / B"
> muestra 88200/31800 en vez del 50/50 exacto que da el código actual.

```
[HH:MM:SS.mmm] ℹ  Iniciando ejecución paralela (MaxFES=120000 evaluaciones agregadas A+B | monitoreo cada 600 FES | ventana Canal A: 12000 FES (10% de MaxFES) | cooldown post-ciclo: 12000 FES)
[HH:MM:SS.mmm] ①  it= 381 | fuente=A(PSO) fit=2891.17 S=0.1673 | objetivo=B(PSO) fit=4484.47 S=0.0000 estancado=True
[HH:MM:SS.mmm] ②  it= 381 | Nuevo ciclo detectado. Activando Fase 2 (extracción)...
[HH:MM:SS.mmm] ②  it= 381 | Fase 2 OK (6.5s) | R²=0.7902 | élite: ['3186.2', ...]
[HH:MM:SS.mmm] ③  it= 381 | Canal A: RMP=0.050 aplicado. Hiperparámetros objetivo → {'w': 0.611, 'c1': 0.86, 'c2': 0.86}
[HH:MM:SS.mmm] ③  it= 382 | Canal A: inicio real registrado (ventana finaliza en ≈34800 FES).
[HH:MM:SS.mmm] ③  it= 540 | Canal A: ventana de 12000 FES completada (22800 → 34800 FES). Pausando para evaluar ΔScore...
[HH:MM:SS.mmm] ⚠  it= 540 | Canal A insuficiente (ΔS=-0.0000). Escalando a Canal B...
[HH:MM:SS.mmm] ③  it= 540 | Canal B: MMD ok=5 rechaz=0. Evaluación inmediata (sin iterar).
[HH:MM:SS.mmm] ✅  it= 540 | Ciclo cerrado tras Canal B. Cooldown activo hasta 46800 FES (12000 FES).
...
[HH:MM:SS.mmm] ℹ  RESUMEN FINAL
[HH:MM:SS.mmm] ℹ    MaxFES (presupuesto):         120000
[HH:MM:SS.mmm] ℹ    FES consumidas (total A+B):   120000
[HH:MM:SS.mmm] ℹ    FES consumidas A / B:         88200 / 31800
[HH:MM:SS.mmm] ℹ    Mejor fitness del sistema:    2623.3425
[HH:MM:SS.mmm] ℹ    Óptimo conocido f(x*):        2600.0000
[HH:MM:SS.mmm] ℹ    Error del sistema (→0 ideal): 2.3342e+01
```

---

## Dependencias

```
setuptools<82        # restaura pkg_resources para opfunu (ver nota abajo)
numpy>=1.24
scipy>=1.10
opfunu>=1.0.4
torch>=2.0
xgboost>=2.0
scikit-learn>=1.3
tqdm>=4.65
numba>=0.67          # evaluador CEC2022 compilado (D-5); wheel cp314, sin toolchain C
# fastshap: incluido como código fuente en fastshap_lib/ (no instalar por pip)
# POT: descartado (incompatible con Python 3.14 en Windows) → Wasserstein vía scipy.linprog
```

(Ver `requirements.txt`.)

> **Python 3.14 — `pkg_resources` removido**: `opfunu` depende de `pkg_resources`, que fue eliminado de la biblioteca estándar en Python 3.14. La instalación de `setuptools<82` lo restaura. Sin esto, la importación de `opfunu` falla con `ModuleNotFoundError: No module named 'pkg_resources'`.

---

## Referencia de Archivos Clave para Continuar

### Documentos — leer primero

| Archivo | Qué contiene |
|---|---|
| `docs/setup_experimental.md` | Protocolo de evaluación completo (10 secciones, adaptado del PDF de la tesis a CEC 2022) + §11 esquema de datos del harness |
| `docs/configuracion_experimental.md` | Configuración única congelada (tabla completa) + registro de decisiones D-1a…D-5 con motivo, alternativas y qué queda pendiente |
| `docs/competidores.md` | Roster candidato de 27 competidores externos — **descartado (D-4)**, se conserva como registro de por qué |

### Código — middleware

| Archivo | Qué hace | Dónde continuar |
|---|---|---|
| `middleware/orquestador.py` | Ciclo completo (Fases 1→2→3), planificador cooperativo determinista (D-2), log dual, vía MaxFES / clásica | Fase 4 (cierre formal) queda fuera de código a propósito — no implementar aquí |
| `middleware/deteccion.py` | Score S, roles dinámicos, FIR; warm-up 15% por MaxFES | D-1a: calibrar pesos `W_FIR`/`W_DWD` por sensibilidad |
| `middleware/extraccion.py` | XGBoost (subrogado) + FastSHAP; `torch.manual_seed(0)` (D-2) | Sin cambios pendientes conocidos |
| `middleware/transferencia.py` | Canal A (RMP) y Canal B (élite), `EstadoTransferencia` | Sin cambios pendientes conocidos |
| `middleware/barrera_seguridad.py` | Wasserstein, MMD, umbral dinámico | D-1e: revisar CASO 3 de `test_barrera_seguridad.py` (no urgente) |
| `problems/cec2022_wrapper.py` | 12 funciones CEC 2022, D∈{10,20}; `.fes`, `.error()`, checkpoints de convergencia; `backend="numba"` (default, D-5) / `"opfunu"` (referencia) | — (alcance cerrado) |
| `problems/cec2022_numba.py` | Backend compilado (D-5): puerto @njit de las 12 funciones, reutiliza datos de opfunu | Sin cambios pendientes conocidos — ver test de paridad antes de tocar |
| `bioalgorithms/{base,pso,de}.py` | PSO/DE + `fraccion_presupuesto` (0.5 exacto, D-2) | — |
| `fastshap_lib/` | FastSHAP vendorizado (iancovert/fastshap) + parches (PyTorch ≥2.x, semilla D-2) | No tocar salvo nuevos parches de compatibilidad |

### Código — harness experimental (`experimentos/`)

| Archivo | Qué hace | Dónde continuar |
|---|---|---|
| `experimentos/corrida.py` | `correr_corrida(algoritmo, funcion, dim, max_fes, idx_semilla)` — `"middleware"`, `"pso"`, `"de"` | Punto de entrada para una corrida suelta |
| `experimentos/grid.py` | Parrilla resumible y paralela, `--dry-run` con estimación | **Empezar por acá** para lanzar el experimento — leer §Pendiente arriba primero |
| `experimentos/consolidar.py` | Une los Parquet por corrida | Correr después de cada lote de `grid.py` |
| `experimentos/esquema.py` | Esquema único de columnas (§11.2/§11.3) | Los módulos de `analisis/` (a crear) deben importar de acá, no redefinir |
| `analisis/` | **No existe todavía** | Crear: `rankings.py`, `estadistica.py` (Friedman/Shaffer/Wilcoxon), `cd_nemenyi.py`, `figuras.py` — spec en `setup_experimental.md` §6–§9 |

### Tests

| Archivo | Cubre |
|---|---|
| `tests/test_determinismo_orquestador.py` | **Criterio de aceptación de D-2** — correr antes de tocar `orquestador.py` |
| `tests/test_cec2022_numba_paridad.py` | **Criterio de aceptación de D-5** — paridad numérica numba vs opfunu, 12 funciones × 2 dims — correr antes de tocar `problems/cec2022_numba.py` |
| `tests/test_orquestador_threading_maxfes.py` / `test_orquestador_threading.py` / `test_orquestador_threading2.py` | Integración completa, vía MaxFES / clásica (nombres históricos: ya no usan threading) |
| `tests/test_experimentos.py` | Smoke del harness (`middleware`/`pso`/`de`) |
| `tests/test_fase0…test_fase3_transferencia_exitosa.py` | Unitarios por fase |
| `tests/test_barrera_seguridad.py` | Falla el CASO 3 (D-1e) — pre-existente, no relacionado con cambios recientes |