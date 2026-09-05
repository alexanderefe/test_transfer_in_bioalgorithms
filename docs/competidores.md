# Competidores — roster candidato y sourcing (setup §3) — SUPERSEDED

> **⚠ DESCARTADO (decisión D-4, 2026-09-04, ver
> `configuracion_experimental.md`).** El roster de este documento NO se
> implementa. Los únicos competidores son **PSO y DE corriendo solos**
> (sin middleware) — el experimento se programó desde el inicio alrededor
> de estos dos algoritmos, y el roster externo disparaba el cómputo a
> semanas. Se conserva este documento como registro de la exploración y
> del porqué se descartó, no como plan de trabajo vigente.
>
> Estado en el momento de escribirlo: NADA seleccionado ni implementado.
> Última actualización: 2026-09-03.

---

## 1. Requisito del setup

El algoritmo propuesto **no** es una variante de Differential Evolution
(es un middleware sobre PSO + DE heterogéneos), así que aplica el criterio
de "metaheurística distinta": **≥ 20–25 algoritmos diversos** cubriendo

- **evolutivos** (GA, DE y variantes, EAs multi-operador),
- **swarm** (PSO y derivados, GWO, WOA, ABC…),
- **basados en distribución** (CMA-ES, EDAs, GSK…),

mezclando **clásicos consolidados** y **propuestas recientes / ganadoras de
competiciones CEC**, sin caer en "solo competidores débiles". Parámetros de
cada competidor: los **recomendados por sus autores**, sin re-tuning.
Documentar tamaños de población en una tabla del paper.

---

## 2. Roster candidato (27 algoritmos)

Leyenda de disponibilidad:
`PY✓✓` implementación de referencia sólida en Python ·
`PY✓` disponible en librería Python establecida (mealpy/pymoo/…) ·
`PY~` existe port comunitario, **fidelidad a verificar** ·
`ML` código oficial solo en MATLAB.

### 2.A · Differential Evolution y variantes (referencia obligada del SOTA)

| # | Algoritmo | Qué es | Ref. | Pedigrí CEC | Disp. |
|---|---|---|---|---|---|
| 1 | **DE/rand/1/bin** | DE clásico, parámetros fijos | Storn & Price 1997 | — (baseline) | PY✓ (pymoo, scipy) |
| 2 | **jDE** | auto-adapta F y CR por individuo | Brest et al. 2006 | — | PY✓ (mealpy) |
| 3 | **SADE** | selección adaptativa de estrategia | Qin et al. 2009 | — | PY✓ (mealpy) |
| 4 | **JADE** | `current-to-pbest/1` + archivo externo | Zhang & Sanderson 2009 | — | PY✓ (mealpy) |
| 5 | **SHADE** | JADE + memoria de éxito (success-history) | Tanabe & Fukunaga 2013 | finalista CEC 2013 | PY✓ (mealpy) |
| 6 | **L-SHADE** | SHADE + reducción lineal de población | Tanabe & Fukunaga 2014 | **ganador CEC 2014** | PY~ / PY✓ parcial (mealpy) |
| 7 | **iL-SHADE** | L-SHADE con memorias mejoradas | Brest et al. 2016 | finalista CEC 2016 | PY~ |
| 8 | **jSO** | mejora de iL-SHADE (peso de mutación) | Brest et al. 2017 | **2.º CEC 2017** | ML / PY~ |
| 9 | **LSHADE-cnEpSin** | L-SHADE + ensemble sinusoidal + covarianza | Awad et al. 2017 | **co-ganador CEC 2017** | ML |
| 10 | **ELSHADE-SPACMA** | L-SHADE + semi-parámetros + CMA-ES híbrido | Hadi et al. 2018 | finalista CEC 2018 | ML |
| 11 | **NL-SHADE-RSP** | L-SHADE no lineal + selección por rango | Stanovov et al. 2021 | **ganador CEC 2021** | PY~ (C++ oficial) |
| 12 | **MadDE** | DE con múltiples estrategias adaptativas | Biswas et al. 2021 | finalista CEC 2021 | PY✓✓ (repo de autores) |

### 2.B · Swarm intelligence

| # | Algoritmo | Qué es | Ref. | Pedigrí CEC | Disp. |
|---|---|---|---|---|---|
| 13 | **PSO** (inercia) | enjambre de partículas canónico | Kennedy & Eberhart 1995; Shi & Eberhart 1998 | — | PY✓✓ (pymoo, mealpy) |
| 14 | **CLPSO** | comprehensive learning PSO | Liang et al. 2006 | — | PY✓ (mealpy/niapy) |
| 15 | **HCLPSO** | CLPSO heterogéneo (2 subpoblaciones) | Lynn & Suganthan 2015 | — | PY~ |
| 16 | **GWO** | grey wolf optimizer | Mirjalili et al. 2014 | — | PY✓ (mealpy/niapy) |
| 17 | **WOA** | whale optimization | Mirjalili & Lewis 2016 | — | PY✓ (mealpy/niapy) |
| 18 | **ABC** | artificial bee colony | Karaboga 2007 | — | PY✓ (mealpy/niapy) |
| 19 | **SaDE-swarm / opción libre** | un swarm reciente adicional (evitar solo débiles) | — | — | PY✓ (mealpy) |

### 2.C · Evolutivos generales y basados en distribución

| # | Algoritmo | Qué es | Ref. | Pedigrí CEC | Disp. |
|---|---|---|---|---|---|
| 20 | **CMA-ES** | evolución de la matriz de covarianza (distribución) | Hansen & Ostermeier 2001 | referencia universal | PY✓✓ (`cma` de Hansen) |
| 21 | **GA real-coded** (SBX + mutación polinómica) | algoritmo genético clásico | Deb 2000 | — | PY✓✓ (pymoo) |
| 22 | **GSK** | gaining–sharing knowledge | Mohamed et al. 2020 | — | PY✓ (mealpy) / ML |
| 23 | **AGSK** | GSK adaptativo | Mohamed et al. 2020 | finalista CEC 2020 | ML / PY~ |
| 24 | **APGSK-IMODE** | GSK adaptativo + IMODE (híbrido) | Mohamed et al. 2021 | **ganador CEC 2021** | ML |
| 25 | **EA4eig** | ensemble: CoBiDE + CMA-ES + jSO + IDE, con eigen-transf. | Bujok & Poláková 2022 | **ganador CEC 2022** | ML |
| 26 | **UMOEAII** | EA de multi-operador unificado | Elsayed et al. 2016 | **ganador CEC 2016** | ML |
| 27 | **HSES** | hybrid sampling evolution strategy | Zhang & Duan 2018 | **ganador CEC 2018** | ML |

*(AMALGAM, citado en el PDF, es multi-método de Vrugt et al. 2009 —
orientado a calibración; código MATLAB, algún port Python. Opcional.)*

---

## 3. Matriz de sourcing

| Grupo | Algoritmos | Acción |
|---|---|---|
| **Verde (Python directo)** — ~15 | DE/rand, jDE, SADE, JADE, SHADE, PSO, CLPSO, GWO, WOA, ABC, CMA-ES, GA, GSK, MadDE, (+1 swarm libre) | Wrapper fino sobre `mealpy` / `pymoo` / `cma`. Días de trabajo, no semanas. |
| **Amarillo (port a verificar)** — ~4 | L-SHADE, jSO, NL-SHADE-RSP, AGSK | Usar port comunitario **validando** contra resultados publicados en CEC 2017/2021 (tabla de medias). Si no cuadra, portar del MATLAB. |
| **Rojo (solo MATLAB)** — ~6 | LSHADE-cnEpSin, ELSHADE-SPACMA, APGSK-IMODE, EA4eig, UMOEAII, HSES | Decisión — ver §4. |

**Riesgo central:** los algoritmos que el PDF §3 nombra explícitamente como
ejemplo de "ganadores recientes a incluir" (**EA4eig, AGSK, APGSK-IMODE,
UMOEAII**) son casi todos del grupo Rojo. EA4eig es especialmente relevante
porque es **el ganador de CEC 2022**, el mismo benchmark que usamos.

---

## 4. Opciones para los algoritmos solo-MATLAB

| Opción | Cómo | Pro | Contra |
|---|---|---|---|
| **A. MATLAB Engine for Python** | `matlab.engine`, llamar el `.m` original desde el harness | fidelidad total (código de los autores) | requiere licencia MATLAB; overhead de arranque del engine por proceso |
| **B. Octave** (libre) | GNU Octave corre la mayoría de los `.m` de CEC con ajustes menores; `oct2py` para llamarlo desde Python | sin licencia; código original | algunos `.m` usan toolboxes o sintaxis no soportada; hay que probar uno a uno |
| **C. Portar a Python** | reimplementar y validar contra las tablas CEC publicadas | queda en el repo, reproducible, sin dependencias externas | semanas de trabajo por algoritmo complejo (EA4eig es un ensemble) |
| **D. Reducir la lista** | quedarse en ~20 (todo Verde + Amarillo validado) y documentar la ausencia de los ganadores solo-MATLAB como limitación | rápido | debilita la comparación; un revisor puede objetar que faltan los SOTA relevantes |

**Recomendación:** B (Octave) para 3–4 de los Rojos más importantes
(**EA4eig** sí o sí; APGSK-IMODE, UMOEAII, HSES si Octave los corre), C solo
si B falla en alguno crítico, D como plan de contingencia documentado.

---

## 5. Preselección propuesta (~22, cubre los 3 paradigmas + clásicos + SOTA)

| Paradigma | Algoritmos |
|---|---|
| DE clásico/adaptativo | DE/rand/1/bin, jDE, SADE, JADE, SHADE |
| DE state-of-the-art | L-SHADE, jSO, NL-SHADE-RSP, MadDE |
| Swarm | PSO, CLPSO, GWO, WOA, ABC |
| Distribución / ES | CMA-ES |
| Evolutivo general | GA (real-coded) |
| Knowledge-based | GSK, AGSK |
| Ganadores CEC (vía Octave/port) | APGSK-IMODE, EA4eig, UMOEAII, HSES |

= 22 algoritmos + el propuesto → **23 en cada ranking**. Ajusta la
Diferencia Crítica de Nemenyi: `CD = q_α·√(23·24/(6·12)) ≈ q_α·2.77`.

---

## 6. Lo que falta decidir

1. **Lista definitiva** (esta preselección u otra) — con el profesor guía.
2. **Sourcing de los Rojos**: ¿hay licencia MATLAB? ¿probamos Octave?
3. **Presupuesto de cómputo**: 22 competidores × la parrilla completa son
   ~semanas en esta máquina (no tienen FastSHAP, pero el costo objetivo es
   idéntico). Requiere evaluador CEC compilado y/o cluster — ver
   `configuracion_experimental.md`.
4. **Versión exacta** de cada implementación (commit/release) para la tabla
   de reproducibilidad (§10).

---

## 7. Estructura de implementación (pendiente)

```
experimentos/competidores/
├── __init__.py
├── base.py            # protocolo común: correr(problema, dim, max_fes, semilla) -> mejor_error
├── registro.py        # {nombre: (constructor, familia, version, params_autor)}
├── mealpy_wrap.py     # los del grupo Verde vía mealpy
├── pymoo_wrap.py      # GA, DE, PSO vía pymoo
├── cma_wrap.py        # CMA-ES vía `cma`
├── madde.py           # port/adaptación de MadDE
├── octave/            # .m originales + oct2py para EA4eig, APGSK-IMODE, UMOEAII, HSES
└── ...
```

Todos deben: (a) parar exactamente en MaxFES usando `problema.fes`, (b)
devolver el mejor valor con ≤ MaxFES evaluaciones, (c) sembrar su RNG con la
semilla de la corrida. El `corrida.py` del harness los despacha por nombre
(hoy `_RUNNERS` solo tiene `middleware`).
