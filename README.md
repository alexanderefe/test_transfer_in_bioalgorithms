# Middleware Adaptativo de Transferencia de Conocimiento entre Metaheurísticas Heterogéneas

**Tesis de Ingeniería Civil Informática — Universidad de Valparaíso**
**Autor:** Alexander Nicolas Farias Cifuentes
**Profesor Guía:** Rodrigo Olivares Órdenes

---

## Descripción

Middleware adaptativo que opera sobre dos algoritmos bioinspirados ejecutados en paralelo (threading), transfiriendo conocimiento del algoritmo con mejor desempeño (**fuente**) hacia el que presenta estancamiento (**objetivo**), de forma online y sin modificar la lógica interna de los algoritmos.

El sistema detecta estancamiento mediante un Score compuesto, extrae conocimiento relevante con FastSHAP sobre un modelo subrogado XGBoost, y lo transfiere gradualmente usando dos canales jerárquicos protegidos por barreras de seguridad (Wasserstein + MMD).

---

## Estructura del Proyecto

```
Fase 3/
├── bioalgorithms/
│   ├── base.py              # Clase abstracta AlgoritmoBioinspirado + EstadoIteracion
│   ├── pso.py               # PSO con inercia decreciente (self.w_actual persistente)
│   └── de.py                # DE/rand/1/bin (F=0.2, CR=0.3 por defecto)
│
├── problems/
│   └── cec2022_wrapper.py   # Wrapper opfunu: 12 funciones CEC 2022, D∈{10,20}
│
├── middleware/
│   ├── deteccion.py         # Fase 1: Score S = 0.3·FIR + 0.7·DWD, roles dinámicos
│   ├── extraccion.py        # Fase 2: XGBoost + FastSHAP + selección élite + umbral MMD
│   ├── xgboost_wrapper.py   # Wrapper XGBoost como nn.Module (para MarginalImputer)
│   ├── barrera_seguridad.py # Wasserstein (scipy.linprog) + MMD centrado
│   ├── transferencia.py     # Canal A (RMP) + Canal B (inyección élite)
│   └── orquestador.py       # Threading Solución A: 3 hilos, pausa entre iteraciones
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
│   └── test_orquestador_threading.py   # ← Test principal de integración completa
│
├── requirements.txt
└── README.md
```

---

## Instalación

**Requisitos:** Python 3.14+ (desarrollado en Python 3.14.6)

```bash
pip install "setuptools<82" numpy>=1.24 scipy>=1.10 opfunu>=1.0.0 torch>=2.0 xgboost>=2.0 scikit-learn>=1.3 tqdm>=4.65
```

> **Nota:** `fastshap` no está disponible en PyPI para Python 3.14. Se incluye como código fuente en `fastshap_lib/` con un parche de compatibilidad para PyTorch ≥ 2.0 ya aplicado. No instalar por pip.

> **Nota:** La librería POT (Python Optimal Transport) fue descartada por incompatibilidad con Python 3.14 en Windows. La distancia de Wasserstein se calcula con `scipy.optimize.linprog` (solver HiGHS), produciendo resultados idénticos.

---

## Uso Rápido

El punto de entrada principal es el test de integración completa:

```python
from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from bioalgorithms.de import DE
from middleware.orquestador import Orquestador

problema = ProblemaCEC2022(numero_funcion=11, ndim=20)
n_iteraciones = 2000

# Caso validado: PSO fuente (bien afinado) → PSO objetivo (mal afinado)
alg_a = PSO(problema, 30, problema.ndim, problema.limites,
            n_iteraciones, semilla=42,
            w_max=0.9, w_min=0.4, c1=2.0, c2=2.0)

alg_b = PSO(problema, 30, problema.ndim, problema.limites,
            n_iteraciones, semilla=500,
            w_max=0.6, w_min=0.6, c1=0.8, c2=0.8)

orq = Orquestador(
    algoritmo_a=alg_a,
    algoritmo_b=alg_b,
    limites=problema.limites,
    n_iteraciones=n_iteraciones,
    frecuencia_monitoreo=10,   # verificar cada 10 iteraciones
    directorio_log="logs",
    nombre_log="experimento",
    max_epochs_fastshap=10,    # reducir a 10 para pruebas rápidas; 50 para producción
)

resumen = orq.ejecutar()
```

El orquestador imprime el log en tiempo real y guarda un archivo JSON en `logs/` al finalizar.

---

## Arquitectura — Ciclo de Transferencia

```
Hilo A (fuente) ──┐
                  ├──► Orquestador (middleware)
Hilo B (objetivo) ──┘
                         │
                    ① Fase 1 — Detección
                         │  S = 0.3·FIR + 0.7·DWD
                         │  S < 0.2 ∧ fuente_óptimo → activar ciclo
                         │
                    ② Fase 2 — Extracción (algoritmos PAUSADOS)
                         │  XGBoost → R² (bloquear si R² < 0)
                         │  FastSHAP → valores Shapley ϕ
                         │  Élite: percentil 20% fitness + ranking ϕ
                         │  Escape: hiperparámetros en max(Δfitness)
                         │  Umbral MMD precalculado con población fuente actual
                         │
                    ③ Fase 3 — Transferencia
                         │
                         ├─ Barrera Wasserstein (dura, algoritmos PAUSADOS)
                         │  W₁ > θ_W → ABORTAR
                         │
                         ├─ Canal A — RMP paramétrico
                         │  RMP: 0.05→0.6, incremento 0.5·ΔS
                         │  Algoritmos REANUDADOS por ventana 10% iteraciones reales
                         │  Si payload vacío (PSO↔DE sin claves comunes) → Canal B directo
                         │  Evaluar ΔS > 0.02 → éxito / fracaso
                         │
                         └─ Canal B — Inyección élite (algoritmos PAUSADOS)
                            Filtro MMD (umbral precalculado en Fase 2)
                            Sustituye peores individuos del objetivo
                            Evaluación inmediata, sin iterar
                            → CERRAR CICLO + cooldown 10% iteraciones
```

---

## Decisiones de Diseño Críticas

| Componente | Decisión | Estado |
|---|---|---|
| FIR | `1 - exp(-3·max(0, f_{t-20} - f_t) / (\|f_{t-20}\| + ε))` | Definitiva |
| Pesos Score | W_FIR=0.3, W_DWD=0.7, W_HDF=0.0 | Provisional |
| Umbral estancamiento | S < 0.2 tras 15% de iteraciones | Definitiva |
| Ventana historial Fase 2 | Últimas 50 iteraciones | Definitiva |
| Criterio élite | Percentil 20% fitness + ranking Shapley | Definitiva |
| RMP inicial / tope | 0.05 / 0.6 | Definitiva |
| Ventana Canal A | 10% del total de iteraciones reales | Definitiva |
| θ_W (Wasserstein) | 25% del diámetro máximo del espacio | Definitiva |
| Gamma MMD | Solo distancias intra-grupo | Definitiva |
| Centrado MMD | Centrar poblaciones en su media antes del kernel RBF | Definitiva |
| Umbral MMD | 3×P95 ruido muestreo fuente; precalculado en Fase 2 | Definitiva |
| Bloqueo R² | Estrictamente negativo (R² < 0) | Definitiva |
| Canal A inerte | Escalar a Canal B directo si payload vacío | Definitiva |
| Cooldown post-ciclo | 10% del total (igual que ventana Canal A) | Definitiva |
| Paralelismo | Threading Solución A (pseudo-paralelo, GIL) | Definitiva |
| Pausa threading | Bucle activo `while evento.is_set(): sleep(0.005)` | Definitiva |
| FIR post-transferencia | Historial recortado desde última transferencia | Definitiva |
| Cierre de ciclo | Provisional (sin Fase 4) | Provisional |

---

## Bugs Resueltos Relevantes

1. **Pausa threading falsa**: `threading.Event.wait()` retorna inmediatamente si el evento ya está SET. Solución: bucle `while self.evento_pausa.is_set(): sleep(0.005)`.

2. **Timeout 30s en verificaciones**: cuando los algoritmos terminan sus iteraciones durante una pausa larga (Fase 2), el orquestador esperaba confirmación de pausa de hilos ya muertos. Solución: verificar `hilo.terminado` antes de `evento_pausado.wait()`.

3. **Umbral MMD desfasado**: el umbral recalculado en Canal B (cuando el fuente ya convergió más) difería del umbral en el momento de extracción. Solución: precalcular y guardar el umbral en `ResultadoExtraccion.umbral_mmd_precalculado`.

4. **Score contaminado post-transferencia**: FIR comparaba fitness actual contra valores anteriores a la transferencia. Solución: recortar historial desde `_iteracion_ultima_transferencia`.

5. **MMD no invariante a traslación**: kernel RBF puro mide separación geométrica cuando las poblaciones están en zonas distintas. Solución: centrar cada población en su media antes del kernel.

6. **Canal A inerte (PSO↔DE)**: PSO {w,c1,c2} y DE {F,CR} no tienen claves en común → payload vacío → el orquestador escala directamente a Canal B sin esperar la ventana de evaluación.

7. **Ciclos en ráfaga post-Canal B**: Score sigue bajo inmediatamente tras el cierre, re-detecta estancamiento. Solución: cooldown de 10% del total de iteraciones.

8. **R²=0 bloqueaba transferencias válidas**: varianza de fitness ≈ 0 producía R²=0.0 exacto por la guarda `ss_tot > 0`. Solución: bloquear solo si R² < 0 estrictamente.

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

### Completado ✅
- Fase 0: interfaz común (`AlgoritmoBioinspirado`), PSO, DE, wrapper CEC 2022
- Fase 1: detección de estancamiento, roles dinámicos, historial FIR recortado
- Fase 2: XGBoost + FastSHAP + selección élite + parámetros de escape + umbral MMD precalculado
- Fase 3: Wasserstein + Canal A (RMP) + Canal B (MMD + inyección)
- Orquestador: threading Solución A con pausa real, log dual (consola + JSON)
- Tests unitarios para cada fase (Fases 0–3)

### Pendiente ⏳
- **Fase 4 / Fase Experimental**: sustituye la comprobación formal. Debe redactarse como apartado separado fuera del código. Incluiría: ΔScore estructurado, auditoría SHAP de asimilación, retorno formal al ciclo de monitoreo pasivo.
- **Protocolo experimental CEC 2022**: 30 corridas × 12 funciones × D∈{10,20}, análisis estadístico (Wilcoxon, A12 Vargha-Delaney).
- **Calibración de pesos del Score**: W_FIR y W_DWD son provisionales (empíricos).
- **Ablation Study**: descartado por tiempo, queda como trabajo futuro.

---

## Log de Ejecución — Formato de Referencia

```
[HH:MM:SS.mmm] ℹ  Iniciando ejecución (2000 iter | ventana Canal A: 200 | cooldown: 200)
[HH:MM:SS.mmm] ①  it= 381 | fuente=A(PSO) fit=2891.17 S=0.1673 | objetivo=B(PSO) fit=4484.47 S=0.0000 estancado=True
[HH:MM:SS.mmm] ②  it= 381 | Nuevo ciclo. Activando Fase 2...
[HH:MM:SS.mmm] ②  it= 381 | Fase 2 OK (6.5s) | R²=0.7902 | élite: ['3186.2', ...]
[HH:MM:SS.mmm] ③  it= 381 | Canal A: RMP=0.050 aplicado. Hiperparámetros → {'w': 0.611, 'c1': 0.86, 'c2': 0.86}
[HH:MM:SS.mmm] ③  it= 382 | Canal A: inicio real registrado (ventana finaliza en it≈582)
[HH:MM:SS.mmm] ③  it= 582 | Canal A: ventana completada. Pausando para evaluar ΔScore...
[HH:MM:SS.mmm] ⚠  it= 582 | Canal A insuficiente (ΔS=-0.0000). Escalando a Canal B...
[HH:MM:SS.mmm] ③  it= 582 | Canal B: MMD ok=5 rechaz=0. Evaluación inmediata.
[HH:MM:SS.mmm] ✅  it= 582 | Ciclo cerrado tras Canal B. Cooldown hasta it=782.
```

---

## Dependencias

```
numpy>=1.24
scipy>=1.10
opfunu>=1.0.0
torch>=2.0
xgboost>=2.0
scikit-learn>=1.3
tqdm>=4.65
setuptools<82
# fastshap: incluido como código fuente en fastshap_lib/ (no instalar por pip)
```

> **Python 3.14 — `pkg_resources` removido**: `opfunu` depende de `pkg_resources`, que fue eliminado de la biblioteca estándar en Python 3.14. La instalación de `setuptools<82` lo restaura. Sin esto, la importación de `opfunu` falla con `ModuleNotFoundError: No module named 'pkg_resources'`.

---

## Referencia de Archivos Clave para Continuar

| Archivo | Qué hace | Dónde continuar |
|---|---|---|
| `middleware/orquestador.py` | Ciclo completo, threading, log | Implementar Fase Experimental |
| `middleware/deteccion.py` | Score S, roles, FIR | Calibrar pesos W_FIR / W_DWD |
| `middleware/extraccion.py` | XGBoost + FastSHAP | Ajustar ventana historial (actual: 50 iter) |
| `middleware/transferencia.py` | Canal A y B, RMP, EstadoTransferencia | Ajustar RMP inicial si se desea |
| `middleware/barrera_seguridad.py` | Wasserstein, MMD, umbral | Sin cambios pendientes |
| `tests/test_orquestador_threading.py` | Test principal de integración | Base para protocolo experimental |