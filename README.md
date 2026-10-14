# Middleware Adaptativo de Transferencia de Conocimiento entre Metaheurísticas Heterogéneas

**Tesis de Ingeniería Civil Informática — Universidad de Valparaíso**
**Autor:** Alexander Nicolas Farias Cifuentes
**Profesor Guía:** Rodrigo Olivares Órdenes

> Última actualización de este README: 2026-10-13. Si continúas este
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
Fase_pruebas/
├── bioalgorithms/
│   ├── base.py              # AlgoritmoBioinspirado + EstadoIteracion; contador fes_propias;
│   │                        #   parámetros max_fes / fraccion_presupuesto
│   ├── pso.py               # PSO; inercia decrece contra la cuota MaxFES·0.5 (o iteraciones)
│   └── de.py                # DE/rand/1/bin (F=0.2, CR=0.3 por defecto)
│
├── problems/
│   ├── cec2022_wrapper.py   # ProblemaCEC2022: 12 funciones CEC 2022, D∈{10,20};
│   │                        #   contador FES agregado (.fes) + óptimo conocido + error()
│   ├── cec2022_numba.py     # Backend compilado (Numba) — port literal del C oficial de CEC 2022 (D-9)
│   └── cec2022_referencia.py # Port Python oficial de los organizadores (referencia lenta, D-9)
│
├── middleware/
│   ├── deteccion.py         # Fase 1: Score S = 0.3·FIR + 0.7·DWD; warm-up 15% por MaxFES
│   ├── extraccion.py        # Fase 2: XGBoost + FastSHAP + selección élite + referencia MMD (centro de la fuente)
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
│   ├── test_fase3_barrera_wasserstein_orquestador.py  # barrera W₁ aplicada por el orquestador (D-11)
│   ├── test_fase3_canal_a_heterogeneo.py      # Canal A PSO↔DE en espacio común (D-10)
│   ├── test_orquestador_threading.py          # integración completa — vía iteraciones (clásica)
│   ├── test_orquestador_threading2.py         # ídem, par PSO↔PSO
│   ├── test_orquestador_threading_maxfes.py   # integración completa — vía MaxFES
│   ├── test_determinismo_orquestador.py       # criterio de aceptación de D-2
│   └── test_experimentos.py                   # smoke del harness (los 6 algoritmos)
│
├── experimentos/           # harness: semillas fijas, corrida única, parrilla, consolidación
│   ├── semillas.json · semillas.py · generar_semillas.py
│   ├── esquema.py          # esquema §11 (columnas, run_id, builders)
│   ├── corrida.py          # correr_corrida(algoritmo, func, dim, max_fes, semilla)
│   │                       #   algoritmos: "middleware" | "pso" | "de" (D-4, solos)
│   │                       #   | "cmaes" | "madde" | "nlshadersp" (D-8, competidores externos)
│   ├── competidores/       # D-8: caja negra correr(problema, max_fes, semilla), corte exacto en MaxFES
│   │   ├── base.py         #   Presupuesto / PresupuestoAgotado
│   │   ├── cmaes_ipop.py   #   IPOP-CMA-ES sobre `cma` 4.5.0
│   │   ├── madde.py        #   port del MATLAB oficial de MadDE
│   │   └── nlshade_rsp.py  #   port del C++ oficial de NL-SHADE-RSP
│   ├── grid.py             # parrilla resumible + paralela (--algoritmo, default: los 6; --dry-run)
│   └── consolidar.py       # resultados/corridas/*.parquet -> resultados/corridas.parquet
│
├── analisis/               # rankings, Friedman+Shaffer, Wilcoxon, CD de Nemenyi, figuras, convergencia (k=6)
│
├── docs/
│   ├── setup_experimental.md            # protocolo de evaluación (adaptado del PDF a CEC 2022) + §11 datos
│   ├── configuracion_experimental.md    # config única congelada + registro de decisiones (D-1a…D-12)
│   ├── competidores.md                  # roster candidato (27); 3 adoptados en D-8
│   ├── cambios_informe_avance1.md       # bitácora de edición del informe LaTeX de Avance 1
│   └── cambios_presentacion_avances.md  # bitácora de edición de la presentación de avances
│
├── resultados/             # salida de grid.py/consolidar.py + resultados/analisis/ (tablas, figuras, LEEME.md)
├── Avance 1/               # informe LaTeX + presentación de avances (en .gitignore, sin historial git)
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
pip install "setuptools<82" "numpy>=1.24" "scipy>=1.10" "opfunu>=1.0.4" "torch>=2.0" "xgboost>=2.0" "scikit-learn>=1.3" "tqdm>=4.65" "pandas>=2.0" "pyarrow>=15.0" "numba>=0.67" "cma==4.5.0"
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
MAX_FES = 120_000            # presupuesto total A+B (5e4 / 5e5 / 5e6 / 5e7 en el protocolo, D-6)

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
                         │  Referencia MMD: centro de la población fuente actual (D-11)
                         │
                    ③ Fase 3 — Transferencia
                         │
                         ├─ Barrera Wasserstein (dura, síncrona) — D-11
                         │  W₁ > θ_W → ABORTAR ciclo sin transferir + cooldown
                         │  se re-evalúa antes de escalar de Canal A a Canal B
                         │
                         ├─ Canal A — RMP paramétrico en espacio latente común (D-10)
                         │  exploración (w ↔ F) y adopción (c2/(c1+c2) ↔ CR); PSO↔DE en ambos sentidos
                         │  RMP: 0.05→0.6, incremento 0.5·ΔS
                         │  A y B siguen alternando generaciones por ventana = 10% de MaxFES
                         │  Si no comparten dimensiones (algoritmo sin espacio común) → Canal B directo
                         │  Evaluar ΔS > 0.02 → éxito / fracaso
                         │
                         └─ Canal B — Inyección élite (síncrona, sin generaciones adicionales)
                            Filtro MMD por instancia (umbral leave-one-out del objetivo, D-11)
                            Sustituye peores individuos del objetivo
                            Evaluación inmediata, sin iterar
                            → CERRAR CICLO + cooldown = 10% de MaxFES
```

---

## Decisiones de Diseño Críticas

| Componente | Decisión | Estado |
|---|---|---|
| Criterio de parada | MaxFES agregado A+B (contador en `ProblemaCEC2022`); sobrepaso ≤ pop_size | Definitiva |
| Métrica de comparación | error = `f_obtenido − f(x*)`; `f(x*)` de tabla `OPTIMO_GLOBAL_CEC2022` (= bias oficial de cada función) | Definitiva |
| Progreso inercia PSO | contra su cuota `MaxFES · 0.5` — exacto, no medido (reparto 50/50 por el planificador round-robin) | Definitiva — D-1b resuelto junto con D-2 |
| Determinismo / reproducibilidad | Planificador cooperativo (no threading) + `torch.manual_seed(0)` en Fase 2 + parche de semilla en `fastshap_lib/utils.py` (`ShapleySampler`) | Definitiva — D-2 resuelto, ver `docs/configuracion_experimental.md` |
| FIR | `1 - exp(-3·max(0, f_{t-20} - f_t) / (\|f_{t-20}\| + ε))` — ventana en generaciones del objetivo | Definitiva |
| Pesos Score | W_FIR=0.3, W_DWD=0.7, W_HDF=0.0 | Provisional — validar por sensibilidad, ver `docs/configuracion_experimental.md` (D-1a) |
| Umbral estancamiento | S < 0.2 tras 15% de **MaxFES** consumido | Definitiva |
| Ventana historial Fase 2 | Últimas 50 generaciones de la fuente | Definitiva |
| Criterio élite | Percentil 20% fitness + ranking Shapley | Definitiva |
| RMP inicial / tope | 0.05 / 0.6 | Definitiva |
| Ventana Canal A | 10% de MaxFES (evaluaciones agregadas A+B) | Definitiva |
| θ_W (Wasserstein) | 60% del diámetro máximo del espacio; la aplica el orquestador al entrar a Fase 3 y antes de Canal B (D-11) | Definitiva — recalibrada con un barrido sobre las funciones oficiales (0.25 empeoraba, 0.6 no se distingue de no tener barrera; D-11) |
| Gamma MMD | Solo distancias intra-grupo; en el filtro, intra-**objetivo** (D-11) | Definitiva |
| Centrado MMD | Centrar poblaciones en su media antes del kernel RBF | Definitiva |
| Filtro MMD | Por instancia: MMD²(δ_x, objetivo) de cada candidata, centrada en la media de la fuente (D-11) | Definitiva |
| Umbral MMD | Máximo del nulo leave-one-out del **objetivo** (misma estadística), calculado en Canal B; solo el centro de la fuente se fija en Fase 2 (D-11) | Definitiva |
| Objetivo colapsado | Dispersión del objetivo < 10⁻³ · diámetro → filtro MMD no aplicable: aprueba todo y queda registrado (`mmd_aplicable`, `n_canal_b_mmd_no_aplicable`) (D-11) | Provisional — umbral de colapso sin hueco natural en los datos; 9/178 eventos entre 10⁻³ y 10⁻² |
| Bloqueo R² | Estrictamente negativo (R² < 0) | Definitiva |
| Canal A entre algoritmos heterogéneos | Interpolación en un espacio latente común (exploración: w↔F; adopción: c2/(c1+c2)↔CR) que cada algoritmo declara; la `w` transferida a PSO desplaza su calendario de inercia (D-10) | Definitiva |
| Canal A inerte | Escalar a Canal B directo si fuente y objetivo no comparten dimensiones del espacio común (desde D-10 no ocurre entre PSO y DE) | Definitiva |
| Cooldown post-ciclo | 10% de MaxFES (igual que ventana Canal A) | Definitiva |
| Planificador | Cooperativo determinista (round-robin, un solo hilo) — reemplazó al threading pseudo-paralelo (D-2) | Definitiva |
| FIR post-transferencia | Historial recortado desde última transferencia | Definitiva |
| Cierre de ciclo | Criterio mínimo (objetivo deja de estar estancado); Fase 4 solo se aborda en la redacción | Provisional |
| Competidores | PSO y DE solos (D-4) + IPOP-CMA-ES, MadDE y NL-SHADE-RSP solos, con parámetros de sus autores y corte exacto en MaxFES (D-8) | Definitiva |
| Evaluador CEC 2022 | Backend Numba portado del **C oficial** de la competición, validado contra él (≤ 4·10⁻¹⁶); opfunu solo aporta metadatos y archivos de datos (D-9) | Definitiva |

---

## Bugs Resueltos Relevantes

1. ~~**Pausa threading falsa**~~: `threading.Event.wait()` retorna inmediatamente si el evento ya está SET. Solución de la época: bucle `while self.evento_pausa.is_set(): sleep(0.005)`. **Todo el mecanismo de pausa/hilos se eliminó en D-2** (planificador cooperativo determinista) — este bug y su parche ya no existen en el código, se dejan como registro histórico.

2. ~~**Timeout 30s en verificaciones**~~: cuando los algoritmos terminaban sus iteraciones durante una pausa larga (Fase 2), el orquestador esperaba confirmación de pausa de hilos ya muertos. Solución de la época: verificar `hilo.terminado` antes de `evento_pausado.wait()`. También obsoleto tras D-2 (sin hilos, sin pausas que confirmar).

3. **Umbral MMD desfasado**: el umbral recalculado en Canal B (cuando el fuente ya convergió más) difería del umbral en el momento de extracción. Solución de la época: precalcular y guardar el umbral en `ResultadoExtraccion.umbral_mmd_precalculado`. **Superado por D-11** (bug #11): el umbral se calcula ahora en Canal B sobre el objetivo; de la Fase 2 solo se conserva el centro de la fuente (`ResultadoExtraccion.referencia_mmd`).

4. **Score contaminado post-transferencia**: FIR comparaba fitness actual contra valores anteriores a la transferencia. Solución: recortar historial desde `_iteracion_ultima_transferencia`.

5. **MMD no invariante a traslación**: kernel RBF puro mide separación geométrica cuando las poblaciones están en zonas distintas. Solución: centrar cada población en su media antes del kernel.

6. **Canal A inerte (PSO↔DE)**: PSO {w,c1,c2} y DE {F,CR} no tienen claves en común → payload vacío → el orquestador escala directamente a Canal B sin esperar la ventana de evaluación. **Superado por D-10**: el Canal A interpola ahora en un espacio latente común y funciona entre PSO y DE en ambos sentidos. De paso se corrigió que PSO descartaba en silencio la `w` transferida (solo entendía `w_max`/`w_min`), lo que dejaba a medias también el Canal A PSO↔PSO.

7. **Ciclos en ráfaga post-Canal B**: Score sigue bajo inmediatamente tras el cierre, re-detecta estancamiento. Solución: cooldown de 10% del presupuesto total (MaxFES).

8. **R²=0 bloqueaba transferencias válidas**: varianza de fitness ≈ 0 producía R²=0.0 exacto por la guarda `ss_tot > 0`. Solución: bloquear solo si R² < 0 estrictamente.

9. **`FastSHAP.train()` crasheaba con `AttributeError: 'NoneType' object has no attribute 'parameters'`** (`fastshap_lib/fastshap.py:440`), detectado en la parrilla completa: 17/14 688 corridas fallaron, todas en F8 (rango dinámico muy grande — valores ~10⁶). Causa: `best_model` arranca en `None` y solo se asigna si la pérdida de validación mejora en alguna época; si sale `NaN` en todas (posible con F8), la comparación `NaN < best_loss` es siempre `False` y `best_model` nunca se asigna. Solución: si `best_model is None` al terminar el entrenamiento, conservar el `explainer` tal como quedó en la última época en vez de copiarle pesos de un "mejor modelo" que nunca existió — mismo criterio que los otros parches de compatibilidad de ese archivo vendorizado (no altera la lógica del paper). Las 17 corridas se reintentaron tras el parche: las 14 688 corridas de la parrilla completa terminaron en `estado="ok"`.


10. **El CEC 2022 de opfunu no es el oficial (D-9)**: el backend Numba portaba `opfunu.cec_based.cec2022`, que difiere del código C oficial en 11 de las 12 funciones, aunque todas coinciden en el óptimo x\* (la paridad de D-5 y la tabla de f(x\*) no lo veían). Síntoma: soluciones con f < f(x\*) que `error()` recortaba a 0. Solución: re-portar el evaluador desde el C oficial, validarlo contra él compilado y etiquetar las corridas (`eval:numba-c22oficial`) para no mezclarlas con las viejas.

11. **La barrera de Wasserstein nunca se aplicaba (D-11)**: el orquestador no llamaba a `ejecutar_ciclo_transferencia()` (la única función que abortaba); calculaba W₁ solo para el log del Canal A, con `umbral_wasserstein=0.0` y `transferencia_permitida=True` fijos, y en el Canal B registraba W₁=0.0 sin calcularla. Ninguna transferencia se abortó nunca y `n_abortos_wasserstein` era siempre 0. Como el MMD está centrado (mira forma, no ubicación), en la práctica nada controlaba la ubicación de lo que se transfería. Solución: `Orquestador._evaluar_barrera_wasserstein()`, evaluada al entrar a la Fase 3 y otra vez antes de escalar de Canal A a Canal B; si W₁ > θ_W aborta el ciclo, lo registra y aplica el cooldown. θ_W se recalibró de 0.25 a 0.6 con un barrido sobre las funciones oficiales (con 0.25, 6–7 de 12 funciones quedaban sin transferencias y el error empeoraba significativamente frente a no tener barrera). El filtro MMD decidía todo-o-nada (el grupo de cada candidata era siempre el mismo conjunto de élites) con un umbral en otra escala; ahora evalúa cada instancia sola (MMD²(δ_x, objetivo)) contra el máximo de un nulo leave-one-out del **objetivo**, con el ancho del kernel también del objetivo. La primera versión tomaba ancho y umbral de la fuente: con la fuente ya convergida el kernel no veía al objetivo y aprobaba o rechazaba lotes enteros casi al azar. Si el objetivo colapsó a un punto, el filtro se declara no aplicable y aprueba (ocurre en ~30 % de los Canal B). Contra la primera versión, el error mejora en 23 pares y empeora en 5 de 120 (p < 0.001).

---

## Escenarios Validados

> Los casos 1 y 2 se validaron antes de D-9 (evaluador opfunu) y D-10
> (Canal A heterogéneo); el Caso 3 los reemplaza para PSO↔DE.

### Caso 1: PSO fuente → PSO objetivo (transferencia homogénea)
- Canal A activo: hiperparámetros comunes {w, c1, c2}
- MMD aprueba instancias cuando distribuciones son compatibles
- Fitness B mejoró de ~4484 a ~2672 en F11/D=20

### Caso 2: PSO fuente → DE objetivo (transferencia heterogénea, pre-D-10)
- Canal A inerte: sin hiperparámetros comunes → Canal B directo
- Validación de barrera Wasserstein + filtro MMD
- Bloqueo automático por R² < 0 cuando fuente convergió

### Caso 3: PSO ↔ DE con Canal A en espacio común (D-10)
- `tests/test_fase3_canal_a_heterogeneo.py`. La interpolación es exacta en
  el espacio común, el calendario de PSO se desplaza sin deformarse y se
  conserva c1+c2.
- En el orquestador (F1, F2, F5, F11, D=10, MaxFES=10⁵) el Canal A ocurre
  en **ambos sentidos** (PSO→DE y DE→PSO), nunca queda inerte y siempre
  acerca al objetivo al fuente.
- Efecto con una semilla (indicativo): Canal A "exitoso" 1 de 15
  ventanas; el resto escala a Canal B, que ahora se usa la mitad de veces.

---

## Estado Actual del Proyecto

> **Para retomar en otra sesión/ventana de IA:** este README es la fuente de
> verdad del estado general; el detalle de cada decisión (motivo, alternativas
> consideradas, qué falta) vive en `docs/configuracion_experimental.md`
> (registro D-1a…D-12) y `docs/setup_experimental.md` (protocolo de evaluación
> completo). Léelos antes de tocar `middleware/` o `experimentos/`.

### Completado ✅

**Núcleo del middleware (Fases 0–3):**
- Fase 0: interfaz común (`AlgoritmoBioinspirado`), PSO, DE, wrapper CEC 2022.
- Fase 1: detección de estancamiento (Score S), roles dinámicos fuente/objetivo, historial FIR recortado post-transferencia.
- Fase 2: XGBoost (modelo subrogado) + FastSHAP oficial (iancovert/fastshap) + selección élite + parámetros de escape + referencia MMD (centro de la fuente; el umbral se calcula en Canal B, D-11).
- Fase 3: barrera Wasserstein + Canal A (RMP paramétrico) + Canal B (inyección de élite filtrada por MMD).
- Tests unitarios de cada fase (`tests/test_fase0…test_fase3_transferencia_exitosa.py`).

**Criterio de parada y métrica (MaxFES):**
- Contador de evaluaciones agregado A+B en `ProblemaCEC2022.fes` (thread-safe); `error()` y tabla `OPTIMO_GLOBAL_CEC2022` para la métrica de comparación (`f_obtenido − f(x*)`).
- Warm-up 15%, ventana Canal A (10%), cooldown y decaimiento de inercia de PSO expresados como fracción de MaxFES. Vía clásica por iteraciones conservada para compatibilidad con los tests de Fase 0–3.
- Traza de convergencia: `configurar_checkpoints()` / `error_checkpoints()` — best-so-far en 120 fracciones de MaxFES (`FRACCIONES_CHECKPOINT`: 20 logarítmicas entre 10⁻⁴ y 10⁻² + una por cada 1 %), registradas evaluación a evaluación.
- Gráficos de convergencia (`analisis/convergencia.py`, 2026-10-13): los 6 algoritmos por (dim × MaxFES × función) como mediana + banda intercuartil de las 51 semillas, y opcionalmente por semilla; en error (default) o fitness. Las corridas con la grilla vieja de 14 checkpoints las rechaza `analisis/` y `grid.py` las re-ejecuta.

**Reproducibilidad — planificador cooperativo determinista (D-2, RESUELTO):**
- Se reemplazó el threading pseudo-paralelo por un bucle único (round-robin A/B, sin locks/eventos/sleeps). Misma semilla → resultado **exactamente idéntico** (`tests/test_determinismo_orquestador.py`).
- De paso se cerraron dos fuentes de no-determinismo independientes del threading: RNG global de `torch` sin sembrar (`middleware/extraccion.py`) y `ShapleySampler.rng` sin semilla en el `fastshap_lib` vendorizado.
- Efecto colateral: el reparto de FES entre PSO y DE pasa a ser **exactamente 50/50** (antes ~70/30, emergente del scheduling) → `fraccion_presupuesto=0.5` es ahora el valor correcto, no una aproximación (D-1b resuelto).

**Configuración única congelada** (`docs/configuracion_experimental.md`):
- Tabla completa de parámetros para las 8 configuraciones experimentales (2 dim × 4 MaxFES).
- Registro de decisiones D-1a (pesos Score, pendiente de validar), D-1b (✅ resuelto), D-1d (comportamiento con MaxFES bajo, aceptado), D-1e (test de barrera rojo, ✅ resuelto en D-11), D-2 (✅ resuelto), D-3 (costo de FastSHAP, aplicado), D-4 (PSO/DE solos como competidores), D-5…D-7 (evaluador Numba, tiers de MaxFES, CPU), D-8 (competidores externos, ✅ implementado), D-9 (evaluador CEC 2022 re-portado desde el C oficial, ✅ resuelto — invalida los resultados previos), D-10 (Canal A heterogéneo PSO↔DE en espacio latente común, ✅ implementado), D-11 (barrera de Wasserstein aplicada por el orquestador, θ_W = 0.6 y filtro MMD por instancia, ✅ corregido), D-12 (tests alineados con los criterios del código, ✅ resuelto).

**Harness experimental** (`experimentos/`, ver `docs/setup_experimental.md` §11):
- `semillas.json` (51 semillas fijas) + `generar_semillas.py` + `semillas.py`.
- `esquema.py`: esquema de datos único (`COLUMNAS_CORRIDA`, `COLUMNAS_PROPUESTO`), `run_id()`, builders de fila.
- `corrida.py`: `correr_corrida(algoritmo, funcion, dim, max_fes, idx_semilla)` — implementa `"middleware"` (propuesto), `"pso"` y `"de"` (solos, D-4) y los competidores externos `"cmaes"`, `"madde"`, `"nlshadersp"` (D-8). Nunca lanza excepción: fallos quedan registrados con `estado="fallo"`.
- `grid.py`: recorre subconjuntos arbitrarios, resumible (salta corridas ya hechas), paralelo por proceso, `--dry-run` con estimación de tiempo calibrada por función.
- `consolidar.py`: une los Parquet por corrida en `resultados/corridas.parquet` / `corridas_propuesto.parquet`.
- Smoke tests: `tests/test_experimentos.py`.

**Competidores externos (D-8, 2026-10-12)** — `experimentos/competidores/`:
- **IPOP-CMA-ES** (`cmaes`, `cma` 4.5.0), **MadDE** (`madde`, port del
  MATLAB oficial) y **NL-SHADE-RSP** (`nlshadersp`, port del C++ oficial
  CEC 2021). Corren solos como caja negra, con los parámetros de sus
  autores, corte exacto en MaxFES y las mismas semillas.
- Harness, esquema y `analisis/` pasan a **6 algoritmos** (k=6, CD ≈ 2.18;
  Shaffer generalizado a cualquier k).
- Validados con un piloto de 216 corridas, todas `ok`, y con el smoke test
  (que incluye determinismo).
- Costo: **+14 688 corridas, ~172 h (~7.2 días) con 24 workers**. Detalle
  en `docs/configuracion_experimental.md` D-8.

### Pendiente ⏳ — en orden recomendado

**Alcance ya fijado (no reabrir sin motivo):** evaluación **solo sobre CEC 2022**
(12 funciones, D∈{10,20}) — no se amplía a CEC 2014/2017 ni D=50 (reconfirmado
en D-6, 2026-09-20: `opfunu` rechaza D=50 con error real, el benchmark oficial
no define datos para esa dimensión). La Fase 4
(cierre formal del ciclo, con ΔScore estructurado y auditoría SHAP) **no se
implementa en código**: se aborda únicamente en la redacción de la tesis.
Competidores = **PSO y DE corriendo solos** (D-4) **+ IPOP-CMA-ES, MadDE y
NL-SHADE-RSP** (D-8) — 6 algoritmos por ranking, no el roster de ~22.

> **✅ D-9 (2026-10-12) — evaluador corregido: TODOS los resultados
> anteriores quedan inválidos.** El CEC 2022 de `opfunu`, que el backend
> Numba portaba (D-5), **no es el oficial**: difiere en 11 de las 12
> funciones (Zakharov, Schaffer F7, Levy, híbridas, Schwefel modificado,
> composiciones…). Todas coinciden solo en el óptimo, por eso la paridad de
> D-5 no lo vio. Síntoma: soluciones con f < f(x*) en 9 de 24 combinaciones.
>
> `problems/cec2022_numba.py` se re-portó desde el **C oficial** y coincide
> con él compilado hasta ≤ 4·10⁻¹⁶. Las 14 688 corridas de 2026-09, los
> pilotos de D-8 y `resultados/analisis/` se hicieron sobre funciones que no
> son las oficiales: **la parrilla se re-corre completa desde cero**.
> `grid.py` re-ejecuta sola las corridas del evaluador viejo y `analisis/`
> las rechaza. Detalle en `docs/configuracion_experimental.md` D-9.

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

> **Nota D-9:** las corridas descritas en los párrafos siguientes (2026-09)
> usaron el evaluador pre-D-9, que no es el CEC 2022 oficial. Se conservan
> como historia del proceso, **no como resultados**.

**Parrilla completa bajo el protocolo anterior — lanzada y terminada
(2026-09-08), archivada en la rama `respaldo-2026-09-20-parrilla-analisis`.**
`experimentos/grid.py --algoritmo middleware,pso,de` corrió las 14 688
corridas de los tiers `{5×10³, 5×10⁴, 5×10⁵, 5×10⁶}` en **71.30 h** (por
encima de la estimación de 50-52 h — el estimador de `grid.py` sigue siendo
aproximado, no una cota dura). Terminó con 14 528 `ok` / 17 `fallo`; los 17
fallos eran el mismo bug (`fastshap_lib/fastshap.py`, `best_model` quedaba en
`None` cuando la pérdida de validación salía `NaN` en todas las épocas —
específico de F8 por su rango dinámico grande, ver "Bugs Resueltos
Relevantes" #9). Con el parche aplicado se reintentaron las 17 y quedaron
**14 688 / 4 896 filas, 100% `estado="ok"`**. `analisis/` (rankings, Friedman
+ Shaffer, Wilcoxon, diagramas CD de Nemenyi) se construyó y validó contra
ese dataset — ver `analisis/` y `resultados/analisis/LEEME.md`.

**Protocolo actualizado (D-6, 2026-09-20):** se retiró el tier `5×10³` y se
agregó `5×10⁷` — `{5×10⁴, 5×10⁵, 5×10⁶, 5×10⁷}` — manteniendo D∈{10,20}
(D=50 reconfirmado inviable). De las 14 688 corridas ya hechas, **11 016
siguen siendo válidas** (los 3 tiers compartidos); faltan **3 672** nuevas
para `5×10⁷`, estimadas en **~312 h (~13 días) con 5 workers** — más que
toda la parrilla anterior junta. Ver `docs/configuracion_experimental.md`
D-6 para el detalle.

**Cambio de CPU (D-7, 2026-10-08):** Xeon E5-2673 v3 (12 núcleos / 24
hilos). Con 1 hilo por proceso (fijado en `grid.py`) y 24 workers rinde ~2×
la CPU anterior: tier `5×10⁷` **~159 h (~6.6 días)**; parrilla completa
desde cero **~184 h (~7.7 días)**. Ver `docs/configuracion_experimental.md`
D-7.

**Competidores externos (D-8) + evaluador oficial (D-9), 2026-10-12:** la
parrilla vigente son **6 algoritmos × 4 896 = 29 376 corridas, todas desde
cero, ~356 h (~14.9 días) con 24 workers** (`grid.py --dry-run`; tomarlo
como ±40 % por algoritmo).

1. **Lanzar la parrilla** (`python experimentos/grid.py`, default: los 6
   algoritmos y los 4 tiers; reanudable). Las corridas viejas de
   `resultados/corridas/` se re-ejecutan y se sobrescriben solas. Si se
   quieren conservar como registro, archivar `resultados/` antes de lanzar.
2. **Regenerar `analisis/`** una vez completa. `cargar_corridas()` valida
   tiers, algoritmos y evaluador: no deja analizar datos mezclados.
3. **Canal A heterogéneo (D-10) — revisar con la parrilla.** Con una
   semilla, el Canal A PSO↔DE casi nunca supera el umbral ΔS > 0.02 y, al
   ocupar el 10 % de MaxFES por ventana, reduce a la mitad las inyecciones
   de Canal B. En F11 el error final empeoró (12.3 contra ~0). Evaluarlo
   con las 51 semillas; si resulta contraproducente, es material para la
   discusión §9 (operadores beneficiosos o no según el presupuesto). La
   estimación de tiempos del middleware en `grid.py` no se recalibró para
   D-10 (menos activaciones de Fase 2 por ventana más larga). El barrido
   de θ_W (D-11, 5 semillas, MaxFES = 5·10⁴) mostró además que el
   propuesto queda muy por detrás de DE solo en F1/D=10 (133 contra 0.04),
   F8/D=10 y F1, F6, F10/D=20, con cualquier θ_W: no lo causa la barrera.
4. **Redacción**: Fase 4 como apartado separado (fuera del código); sección
   de resultados con la discusión obligatoria del setup (§9) — en qué MaxFES
   destaca el propuesto, cómo varía entre dimensionalidades, qué operadores
   son beneficiosos/contraproducentes según el presupuesto, robustez frente
   a PSO/DE solos y a los competidores externos. Tabla de competidores
   con poblaciones y versiones (`setup_experimental.md` §4/§10). Mencionar
   que el benchmark se evalúa con el código oficial (D-9) y por qué no con
   opfunu.
5. **Actualizar el Informe de Avance 1 y la presentación** (`Avance 1/`,
   editados hasta 2026-09-27; bitácoras en `docs/cambios_*.md`). Sus
   resultados, tablas y diagramas CD salen de la parrilla pre-D-9: 3
   algoritmos, tier `5×10³` y "Canal A = 0 en las 4 896 corridas"
   (anterior a D-10). Sus resultados quedan inválidos y no se pueden citar
   como definitivos. Hay que reemplazarlos con los de la nueva parrilla.

**Diferido, no bloqueante:**
- Ablation Study: descartado por tiempo, queda como trabajo futuro.

**Nota sobre k=6 en los rankings:** con propuesto + PSO/DE solos + 3
competidores externos, los rankings, Friedman y los CD de Nemenyi (CD ≈ 2.18)
siguen por debajo de los 20–25 algoritmos del PDF del setup y sin el
ganador de CEC 2022 (EA4eig, solo MATLAB). Hay que reconocerlo en la
redacción (`docs/setup_experimental.md` §3).

### Competidores externos — decidido e implementado (D-8)

> La propuesta del 2026-10-08 se adoptó con los **3 primeros** candidatos.
> EA4eig y HCLPSO no se implementan. Registro completo (criterios, ports,
> verificación y tiempos) en `docs/configuracion_experimental.md` D-8, y
> parámetros en §A.8.

**Rol de los competidores:** corren **solos, como medida comparativa**
frente al middleware — **no** se acoplan al middleware ni implementan la
interfaz de `bioalgorithms/`. Son una caja negra `correr(problema, max_fes,
semilla)` sobre `ProblemaCEC2022`, que corta exactamente en MaxFES
(`experimentos/competidores/base.py`) y deja los checkpoints en el
problema. Solo llenan §11.2; §11.3 queda en `None` (como `pso`/`de`).

**Criterios de selección (en orden de peso):**
1. **Responder a la pregunta de la tesis:** al menos un competidor que
   combine varias estrategias/algoritmos *sin* transferencia explícita, para
   descartar que la mejora del middleware venga solo de "combinar dos
   algoritmos" y no de transferir conocimiento.
2. **Credibilidad:** estado del arte en CEC 2021/2022, no solo competidores
   débiles.
3. **Diversidad de paradigma:** había swarm (PSO) y evolutivo (DE); faltaba
   basado en distribución.
4. **Integración fiel y barata:** referencia oficial portable y validable.
5. **Escalar a presupuestos grandes (`5×10⁷`):** adaptación a MaxFES o
   reinicios, para no estancarse y desperdiciar presupuesto.

| id | Algoritmo | Por qué | Integración real | µs/eval (D10/D20, 5×10⁵) |
|---|---|---|---|---|
| `cmaes` | **IPOP-CMA-ES** | Paradigma de distribución; referencia universal; los reinicios aprovechan `5×10⁷` | `cma` 4.5.0 + bucle IPOP propio (reinicios sin tope, λ ×2) | 177 / 240, baja con MaxFES |
| `madde` | **MadDE** | DE multi-estrategia adaptativo (top CEC 2021): contraste directo del criterio 1 | **Port del MATLAB oficial** (el repo de los autores es MATLAB, no Python como se suponía) | 45 / 30 |
| `nlshadersp` | **NL-SHADE-RSP** | Estado del arte de la línea L-SHADE (ganador CEC 2021); población ajustada a MaxFES | Port del C++ oficial de la entrega CEC 2021 | 63 / 50 |

Los ports reproducen tal cual las rarezas del código de referencia (mismo
criterio que D-5), documentadas en el docstring de cada módulo.

**Impacto:** rankings de 3 → 6 algoritmos (`CD ≈ 2.18`, N=12, α=0.05);
+14 688 corridas, ~172 h con 24 workers. CMA-ES es el más caro: el costo
fijo por iteración de `cma` domina con λ chico. Su extrapolación a 5×10⁷
es la parte más incierta de la estimación.

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
pandas>=2.0          # harness experimental: resultados en Parquet
pyarrow>=15.0        # backend Parquet de pandas
numba>=0.67          # evaluador CEC2022 compilado (D-5); wheel cp314, sin toolchain C
cma==4.5.0           # IPOP-CMA-ES, competidor externo (D-8)
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
| `docs/configuracion_experimental.md` | Configuración única congelada (tabla completa) + registro de decisiones D-1a…D-12 con motivo, alternativas y qué queda pendiente |
| `docs/competidores.md` | Roster candidato de 27 competidores externos; de ahí salen los 3 adoptados en D-8 (tabla de estado al inicio) |

### Código — middleware

| Archivo | Qué hace | Dónde continuar |
|---|---|---|
| `middleware/orquestador.py` | Ciclo completo (Fases 1→2→3), planificador cooperativo determinista (D-2), log dual, vía MaxFES / clásica | Fase 4 (cierre formal) queda fuera de código a propósito — no implementar aquí |
| `middleware/deteccion.py` | Score S, roles dinámicos, FIR; warm-up 15% por MaxFES | D-1a: calibrar pesos `W_FIR`/`W_DWD` por sensibilidad |
| `middleware/extraccion.py` | XGBoost (subrogado) + FastSHAP; `torch.manual_seed(0)` (D-2) | Sin cambios pendientes conocidos |
| `middleware/transferencia.py` | Canal A (RMP) y Canal B (élite), `EstadoTransferencia` | Sin cambios pendientes conocidos |
| `middleware/barrera_seguridad.py` | Wasserstein, MMD, filtro por instancia | θ_W = 0.6; filtro MMD por instancia con referencia en el objetivo y regla de objetivo colapsado (D-11) |
| `problems/cec2022_wrapper.py` | 12 funciones CEC 2022, D∈{10,20}; `.fes`, `.error()`, checkpoints de convergencia; `backend="numba"` (default) / `"referencia"` (port Python oficial, lento). opfunu solo da metadatos (D-9) | — (alcance cerrado) |
| `problems/cec2022_numba.py` | Port @njit **literal del C oficial** de CEC 2022 (D-9), incluidos los efectos de sus buffers globales `y`/`z`; `VERSION_EVALUADOR` etiqueta las corridas | No "corregir" rarezas del C: se replican a propósito — correr el test de fidelidad antes de tocar |
| `problems/cec2022_referencia.py` | Port Python oficial de los organizadores, vendorizado con 3 cambios mínimos marcados `D-9` | Solo referencia para el test; no usar en experimentos |
| `bioalgorithms/{base,pso,de}.py` | PSO/DE + `fraccion_presupuesto` (0.5 exacto, D-2) | — |
| `fastshap_lib/` | FastSHAP vendorizado (iancovert/fastshap) + parches (PyTorch ≥2.x, semilla D-2) | No tocar salvo nuevos parches de compatibilidad |

### Código — harness experimental (`experimentos/`)

| Archivo | Qué hace | Dónde continuar |
|---|---|---|
| `experimentos/corrida.py` | `correr_corrida(algoritmo, funcion, dim, max_fes, idx_semilla)` — `"middleware"`, `"pso"`, `"de"`, `"cmaes"`, `"madde"`, `"nlshadersp"` | Punto de entrada para una corrida suelta |
| `experimentos/competidores/` | IPOP-CMA-ES, MadDE, NL-SHADE-RSP (D-8): `correr(problema, max_fes, semilla)` + `VERSION` | Para agregar otro competidor: módulo nuevo + entrada en `COMPETIDORES` (`__init__.py`), `ALGORITMOS_VALIDOS` y `_COSTO_US_POR_EVAL` (`grid.py`), `FAMILIA_ALGORITMO` (`esquema.py`), `ALGORITMOS` (`analisis/datos.py`) |
| `experimentos/grid.py` | Parrilla resumible y paralela, `--dry-run` con estimación | **Empezar por acá** para lanzar el experimento — leer §Pendiente arriba primero |
| `experimentos/consolidar.py` | Une los Parquet por corrida | Correr después de cada lote de `grid.py` |
| `experimentos/esquema.py` | Esquema único de columnas (§11.2/§11.3) | Los módulos de `analisis/` importan de acá, no redefinir |
| `analisis/` | `rankings.py`, `estadistica.py` (Friedman/Shaffer/Wilcoxon), `cd_nemenyi.py`, `figuras.py`, `convergencia.py`, `main.py` — spec en `setup_experimental.md` §6–§9; generalizado a k=6 (D-8) | `python analisis/main.py` sobre la parrilla completa (valida que estén los 6 algoritmos y los 4 tiers vigentes) |

### Tests

| Archivo | Cubre |
|---|---|
| `tests/test_determinismo_orquestador.py` | **Criterio de aceptación de D-2** — correr antes de tocar `orquestador.py` |
| `tests/test_cec2022_numba_paridad.py` | **Gate de fidelidad (D-5 → D-9)** — anclas del C oficial + paridad contra la referencia oficial (óptimo, vecindades del óptimo, aleatorios, esquinas) + nada bajo f(x*), 12 funciones × 2 dims — correr antes de tocar `problems/cec2022_numba.py` |
| `tests/test_orquestador_threading_maxfes.py` / `test_orquestador_threading.py` / `test_orquestador_threading2.py` | Integración completa, vía MaxFES / clásica (nombres históricos: ya no usan threading) |
| `tests/test_experimentos.py` | Smoke del harness: los 6 algoritmos (corte exacto en MaxFES, checkpoints, determinismo y `version` de los competidores) |
| `tests/test_fase0…test_fase3_transferencia_exitosa.py` | Unitarios por fase; criterios alineados con lo que aplica el código (D-12): Fase 2 exige R² ≥ 0 (no 0.7), los de Fase 3 respetan el bloqueo por R² < 0 y verifican la regla W₁ > θ_W; `test_fase3_transferencia_exitosa` usa F2/D=10 |
| `tests/test_fase3_barrera_wasserstein_orquestador.py` | **Barrera de Wasserstein en el orquestador (D-11)**: aborto con estado limpio y cooldown, sin efectos si se permite; θ_W=0 aborta todo (sin inyecciones), θ_W=∞ nunca aborta y el log registra W₁/θ_W reales |
| `tests/test_fase3_canal_a_heterogeneo.py` | **Canal A PSO↔DE (D-10)**: mapas del espacio común, interpolación exacta, desplazamiento del calendario de PSO, ambos sentidos en el orquestador, y comparación con/sin Canal A heterogéneo (se reporta, no se exige) |
| `tests/test_barrera_seguridad.py` | Pasa 3/3: CASO 1 recalibrado al evaluador oficial, CASO 3 reescrito para el filtro por instancia (D-11, cierra D-1e) |