# Competidores — roster candidato y sourcing (setup §3)

> **Estado vigente (D-8, 2026-10-12, ver `configuracion_experimental.md`):**
> de este roster se implementaron **3 competidores externos**, que corren
> solos junto a PSO y DE solos (D-4):
>
> | id | Algoritmo | Fila del roster | Implementación |
> |---|---|---|---|
> | `cmaes` | IPOP-CMA-ES | #20 (CMA-ES) | librería `cma` 4.5.0 (Hansen) + bucle IPOP propio |
> | `madde` | MadDE | #12 | port a Python del **MATLAB** oficial (`subhodipbiswas/MadDE`) |
> | `nlshadersp` | NL-SHADE-RSP | #11 | port a Python del **C++** oficial (entrega CEC 2021) |
>
> Código en `experimentos/competidores/` (estructura real en §7). **Corrección
> a la tabla de §2:** el repositorio de MadDE de los autores es MATLAB, no
> Python (`PY✓✓` era incorrecto); se portó igual que NL-SHADE-RSP.
>
> El resto del roster **no** se implementa. EA4eig (#25, ganador CEC 2022)
> y HCLPSO (#15) quedaron como 4.º competidor opcional no adoptado.
>
> **Historia:** la versión de 2026-09-03 de este documento quedó descartada
> por D-4 (2026-09-04): el roster completo de ~22 algoritmos disparaba el
> cómputo a semanas. El 2026-10-08 se propuso retomar 3–4 competidores con
> criterios de selección (README), y D-8 adoptó los 3 primeros. El resto
> del documento se conserva como registro de la exploración.

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

## 7. Estructura de implementación (implementada en D-8)

```
experimentos/competidores/
├── __init__.py        # COMPETIDORES = {id: (módulo, nombre, familia)}
├── base.py            # Presupuesto: corte exacto en MaxFES (PresupuestoAgotado)
├── cmaes_ipop.py      # IPOP-CMA-ES sobre `cma`
├── madde.py           # port de MadDE.m (+ gnR1R2, updateArchive, boundConstraint)
└── nlshade_rsp.py     # port de nl_shade_rsp.cpp
```

Cada módulo expone `correr(problema, max_fes, semilla) -> None` y una
constante `VERSION`. Los tres (a) paran exactamente en MaxFES (toda
evaluación pasa por `Presupuesto.evaluar`, que corta en `problema.fes ==
max_fes`), (b) dejan el mejor valor con ≤ MaxFES evaluaciones en los
checkpoints de `problema`, y (c) siembran su RNG con la semilla de la
corrida. `experimentos/corrida.py` los registra en `_RUNNERS` a partir de
`COMPETIDORES`.

## 8. Algoritmos de referencia por familia — revisión de fuentes (nov-2026)

Propuesta del trabajo futuro del Avance 2: tres algoritmos por familia
(evolutivos, enjambre, sistemas inmunes), como **referencia de
comparación**. BCA se reemplazó por opt-IA/IMMALG (mejor especificado y con
resultados publicados para validar). Antes de integrar nada se leyó el
código instalado de cada implementación de librería y se comparó con su
paper.

**Versiones**: `niapy==2.7.1` (PyPI) y mealpy 3.1.0 desde GitHub, commit
`46b9850` (la 3.0.3 de PyPI exige `numpy<=1.26`, que no compila en Python
3.14). Ver `requirements.txt`.

| Algoritmo | Fuente revisada | Veredicto | Detalle |
|---|---|---|---|
| jDE (Brest 2006) | NiaPy `SelfAdaptiveDifferentialEvolution` | **Usable con parámetros corregidos** | Lógica fiel (DE/rand/1/bin, autoadaptación de F y CR). Defaults distintos al paper: τ₁ = 0,4, τ₂ = 0,2 (paper 0,1 y 0,1); F ∈ [0, 1] (paper [0,1, 1,0]); F₀ = 2 y CR₀ = 0,5 (paper 0,5 y 0,9; hay que pasar un `individual_type` con esos valores). La selección usa `<` estricto (paper `≤`). |
| GWO (Mirjalili 2014) | NiaPy `GreyWolfOptimizer` | **Usable** | Fiel al código MATLAB original (actualización de α, β, δ igual que el original; `a` lineal 2 → 0 por FES). |
| JADE (Zhang y Sanderson 2009) | mealpy `DE.JADE` | **Usable con correcciones menores** | current-to-pbest/1 con archivo, Cauchy/Normal y media de Lehmer, fieles. Desviaciones: sin éxitos empuja μ_F y μ_CR hacia 0,5 (paper: sin cambio); `cauchy.rvs` usa el RNG global de numpy (el adaptador lo siembra); límites por recorte (paper: punto medio padre–límite). |
| GA real | NiaPy `GeneticAlgorithm` | **Descartado (con errores)** | `while io != ic` con `io = ic` nunca elige pareja (cruce consigo mismo), y `run_iteration` sobrescribe la selección con el cruce y el cruce con la mutación sobre `population[i]`: queda «padre i + reinicio aleatorio de genes», sin selección. |
| GA real | mealpy `GA.BaseGA` | No canónico | Cruce uniforme (intercambio de genes) + mutación por reinicio y supervivencia por torneo poco convencional. Alternativa propuesta: GA de pymoo (SBX + mutación polinomial), que instala en Python 3.14 sin cambiar numpy. |
| CLPSO (Liang et al. 2006) | mealpy `PSO.CL_PSO` | **Descartado (con errores)** | Inercia `w_max·(t/T)·(w_max−w_min)` **crece** de 0 a 0,45 (paper: 0,9 → 0,4); la partícula solo se mueve si mejora (PSO codicioso); el ejemplar se sortea en cada iteración y el *refreshing gap* m = 7 no se usa. NiaPy no tiene CLPSO. Alternativa: portar el código MATLAB de los autores. |
| ABC (Karaboga 2005/2007) | NiaPy `ArtificialBeeColonyAlgorithm` | Con error | Fase de observadoras revisa `trials[i]` (índice de otro ciclo) en vez de `trials[s]`; vecino no excluye a i en empleadas; exploradora dentro de empleadas; probabilidad 1/(f+0,01); `limit` = 100 (paper SN·D). |
| ABC | mealpy `ABC.OriginalABC` | No fiel | Muta todas las dimensiones con φ (el ABC original cambia una sola). Alternativa: corregir el de NiaPy o portar el código oficial de Karaboga. |
| CLONALG (de Castro y Von Zuben 2002) | NiaPy `ClonalSelectionAlgorithm` | **Descartado (con errores)** | Mutación invertida (invierte el bit si `rand > tasa`) y tasa `exp(−10·f)` con fitness sin normalizar: con f ≥ 300 cada clon invierte todos sus bits. |
| CLONALG | pyMetaheuristic `clonalg` | **Descartado** | Tasa `exp(−2,5·f)` sin normalizar: con f ≥ 300 los clones nunca mutan. |
| CLONALG | aisp `csa.Clonalg` | **Usable** | Selección de los n mejores, clones ∝ afinidad normalizada, hipermutación `exp(−ρ·afinidad)`, reemplazo de los d peores. Desviación: clones ∝ afinidad lineal (paper: por ranking). Hay que desactivar `n_iter_no_change`. |

**Contraste empírico de CLONALG** (D = 10, MaxFES = 5·10⁴, 5 semillas,
evaluador oficial CEC 2022, parámetros por defecto de cada librería; error
mediano):

| Función | NiaPy | aisp | pyMetaheuristic | Búsqueda aleatoria |
|---|---|---|---|---|
| f1 | 13 470 | **553** | 33 390 | 9 692 |
| f4 | 74,3 | **23,6** | 93,1 | 62,8 |
| f7 | 112 | **15,2** | 200 | 73,2 |

NiaPy y pyMetaheuristic quedan bajo la búsqueda aleatoria, coherente con
los errores encontrados en el código.

**Adaptadores**: `experimentos/competidores/adaptador_niapy.py` y
`adaptador_mealpy.py` (corte exacto en MaxFES vía `Presupuesto`, relanzan
las excepciones que NiaPy oculta fuera del hilo principal, historial
acotado, semilla). Probados en `tests/test_adaptadores_librerias.py`. El
de mealpy rechaza algoritmos cuyo `evolve` use `self.epoch`.

### 8.1 Fuentes elegidas e implementación (nov-2026)

| Algoritmo | Módulo | Fuente |
|---|---|---|
| GA real | `experimentos/competidores/ga_pymoo.py` | `pymoo` 0.6.2, `GA` con sus defaults (pop. 100, torneo binario, SBX η = 15 p = 0,9, mutación polinomial η = 20 p = 1/D, supervivencia (μ+λ)). Se reemplaza su terminación por defecto (tolerancias) por la de presupuesto. |
| CLPSO | `experimentos/competidores/clpso.py` | Port de `CLPSO_new_func.m` (código de los autores, github.com/P-N-Suganthan/CODES, `2006-IEEE-TEC-CLPSO.zip`) con los parámetros de `TEC_test.m`. El código difiere del texto del paper en Pc ∈ [0, 0,5] (paper [0,05, 0,5]), w 0,9 → 0,2 (paper 0,9 → 0,4) y refreshing gap 5 (paper 7); se sigue el código. Población 10 en D = 10 y 40 en D = 20 (el paper usa 40 para 30-D y no da valor para 20-D). |
| ABC | `experimentos/competidores/abc_karaboga.py` | NiaPy corregido para reproducir `ABC.C` oficial (abc.erciyes.edu.tr/pub/ABC.C): corrige el índice `trials[i]`, la inicialización de NP fuentes (usaba NP/2), el vecino k ≠ i, una exploradora por ciclo, fitness 1/(1+f) y p = 0,9·fit/max + 0,1. Colonia 40 (ABC.C), limit = SN·D (Karaboga y Akay 2009). |
| CLONALG | `experimentos/competidores/clonalg_aisp.py` | `aisp` 0.5.5 con sus defaults (N = 50, n = 5, β = 10, ρ = 1, d = 5), sin detención temprana y sin el historial por iteración. |

Pruebas: `tests/test_referencias_familia.py` (FES exactas, traza monótona y
determinismo en D = 10 y 20).

Sanidad (D = 10, MaxFES = 5·10⁴, 5 semillas, error mediano):

| Función | GA pymoo | CLPSO | ABC corregido | ABC NiaPy original | CLONALG aisp | Búsqueda aleatoria |
|---|---|---|---|---|---|---|
| f1 | 11,2 | 0,0018 | 8 660 | 7 836 | 553 | 9 692 |
| f4 | 9,95 | 9,53 | 34,5 | 25,8 | 23,6 | 62,8 |
| f7 | 1,09 | 0,024 | 1,55 | 2,83 | 15,2 | 73,2 |

ABC queda cerca de la búsqueda aleatoria en f1 (Zakharov rotado). No es un
error de implementación: en funciones sin rotar converge como reportan los
autores (esfera D = 10: 6·10⁻¹⁷ con 2·10⁴ FES; Rastrigin D = 10: 0 con
3·10⁴ FES). Es la debilidad conocida de ABC en problemas no separables, ya
que modifica una sola coordenada por vez.

### 8.2 Resto de la familia, opt-aiNet y registro (nov-2026)

| Algoritmo | Módulo | Fuente y parámetros |
|---|---|---|
| jDE | `jde.py` | NiaPy con los parámetros de Brest et al. (2006): NP = 100, τ₁ = τ₂ = 0,1, F ∈ [0,1, 1,0], F₀ = 0,5, CR₀ = 0,9; selección con `≤` (NiaPy usaba `<`). Verificado: la población inicial nace con F = 0,5 y CR = 0,9, y ~10 % de los individuos regenera F y CR en cada generación. |
| GWO | `gwo.py` | NiaPy sin cambios, 30 lobos (paper). |
| JADE | `jade.py` | mealpy con `evolve` reescrito: sin éxitos no se actualizan μ_CR ni μ_F; F con el generador sembrado (no el RNG global); reparación de límites por punto medio padre–límite. p = 0,05, c = 0,1, NP = 30. |
| opt-aiNet | `opt_ainet.py` | Implementación propia desde Timmis y Edmonds (2004), con sus dos correcciones (criterio de estabilidad válido para cualquier signo y supresión elitista) y los parámetros originales (N = 20, Nc = 10, β = 100, σs = 0,2, d = 40 %). En CEC 2022 el paso de mutación y σs se escalan por (ub − lb)/4 (el original está definido en [−2, 2]); supresión con KD-tree. |
| opt-IA / IMMALG | — | **Pendiente.** Los tres papers (Cutello et al. 2005, EA y ICARIS; Pavone et al. 2012, JOGO) son de acceso cerrado (OpenAlex: `closed`), y IRIS UniCT, CORE y academia.edu devuelven 403. No se implementa de memoria; hace falta el PDF por acceso institucional. |

**Reproducción de opt-aiNet** (`tests/test_opt_ainet_reproduccion.py`, 50 corridas,
funciones y protocolo de Timmis y Edmonds 2004):

| Indicador | Esta implementación | Publicado (paper 2002 / código original / versión corregida) |
|---|---|---|
| Multi: picos | 62,6 ± 24,9 | 56,1 ± 4,4 / 57,2 ± 13,8 / 56,5 ± 17 |
| Multi: iteraciones al óptimo | 38,0 ± 33,2 (47/50 lo alcanzan) | 53,5 ± 47,2 / 58,8 ± 45,2 / 212,8 ± 140 |
| Roots: picos (de 6) | 5,98 ± 0,14 | 6,0 / 5,9 ± 0,3 / 5,2 ± 0,8 |
| Roots: iteraciones al óptimo | 55,7 ± 20,7 | 86,9 ± 34,3 / 93,2 ± 32,0 / 149,2 ± 87,1 |

Sanidad en CEC 2022 (D = 10, MaxFES = 5·10⁴, 5 semillas, error mediano):

| Función | jDE | JADE | GWO | opt-aiNet | Búsqueda aleatoria |
|---|---|---|---|---|---|
| f1 | 1 445 | 0 | 122 | 4 977 | 9 692 |
| f4 | 19,3 | 5,02 | 13,2 | 42,8 | 62,8 |
| f7 | 8,54 | 0,043 | 24,3 | 95,5 | 73,2 |

opt-aiNet queda bajo la búsqueda aleatoria en f7 con este presupuesto: su
mutación es muy local (paso ≤ 1 % del ancho del dominio, igual que el
original) y la red no se estabiliza mientras sigue mejorando, así que casi
no inserta células nuevas. En la esfera de 10-D converge (1,6·10⁴ →
2,6·10⁻² en 10⁵ FES); es una limitación del algoritmo, diseñado para
problemas multimodales de baja dimensión.

**Registro**: los 8 ids (`ga`, `jde`, `jade`, `clpso`, `gwo`, `abc`,
`clonalg`, `optainet`) están en `COMPETIDORES` (más la tupla
`REFERENCIA`), `ALGORITMOS_VALIDOS` y `_COSTO_US_POR_EVAL` (provisional,
un solo proceso × 2,54) de `grid.py`, y `FAMILIA_ALGORITMO` de `esquema.py`
(familias `evolutivo`, `enjambre`, `inmune`). En `analisis/datos.py` van en
`ALGORITMOS_REFERENCIA`, aparte de `ALGORITMOS`, hasta decidir si entran al
mismo ranking. Estimación provisional de la parrilla de los 8: 39 168
corridas, ~1 164 h con 24 workers (≈ 90 % en el tier 5·10⁷).

### 8.3 GCO reemplaza a opt-IA (nov-2026)

Se buscó un tercer algoritmo inmune con especificación completa y de
acceso abierto, y con resultados publicados para validar:

| Candidato | Acceso | Problema |
|---|---|---|
| opt-IA / IMMALG (Cutello et al. 2005; Pavone et al. 2012) | Cerrado (OpenAlex `closed`; IRIS UniCT, CORE y academia.edu: 403) | No se puede verificar la especificación. |
| BCA (Kelsey y Timmis, GECCO 2003) | Abierto (copia en York) | Subespecificado: muta los bits crudos del double IEEE 754 sin decir cómo manejar valores fuera de rango, NaN o ±inf; no fija la probabilidad de mutación dentro de la región contigua, la distribución del largo, si el hotspot es por variable o global, ni la probabilidad de la metadinámica; su tabla usa parada por distancia al óptimo. |
| **GCO** (Villaseñor et al., IJCIS 2018) | **Abierto (CC BY-NC)** | Pseudocódigo completo (Algoritmos 1 y 2), parámetros y tablas en 2-D y 30-D, salvo un punto: no precisa si r₁, r₂, r₃ ~ 𝒞 son distintos. Ese punto no afecta en 2-D pero sí en 30-D (ver abajo). Sin código de los autores. |

GCO (*Germinal Center Optimization*) se presenta explícitamente como «an
AIS algorithm based on GC» (sección 5 del paper): modela la reacción del
centro germinal (expansión clonal y hipermutación en la zona oscura,
selección por afinidad y ayuda T en la zona clara).

**Implementación**: `experimentos/competidores/gco.py`, sobre
`system_based.GCO.OriginalGCO` de mealpy (commit 46b9850). Revisado línea a
línea contra los Algoritmos 1 y 2: coincide (señal de vida inicial 70,
contador 1, duplicar o morir con U[0, 100], mutación DE con r₁, r₂, r₃
según la distribución de clones, cruce con CR sin j_rand, +10 al mejorar,
zona clara −10 + 10·fit). Única ambigüedad del paper: no dice si r₁, r₂, r₃
deben ser distintos (mealpy los toma distintos). Parámetros del paper:
N = 40, CR = 0,7, F = 1,25.

**Reproducción** (`tests/test_gco_reproduccion.py`):

- Tabla 2 del paper (d = 2, 20 040 FES, 30 corridas): **se reproduce**. Las
  8 funciones probadas llegan a precisión de máquina donde el paper también
  (Sphere 6·10⁻⁵⁰, Ackley 4,44·10⁻¹⁶ idéntico, Rastrigin 0 idéntico,
  Zakharov 1,6·10⁻⁴⁶, Dixon-Price 8·10⁻²⁶, Rosenbrock 1,1·10⁻¹⁸) y
  Griewank queda en 0,0038 ± 0,0048 (publicado 0,0022 ± 0,0034).
- Tabla 4 (d = 30, 40 040 FES, 10 corridas): **no se reproduce**. El
  pseudocódigo dice «r₁, r₂, r₃ ~ 𝒞» sin precisar si deben ser distintos, y
  en 30-D esa decisión cambia el resultado ~50 veces:

  | Función 30-D | mealpy (distintos) | Con reemplazo | Publicado |
  |---|---|---|---|
  | Sphere | 114 ± 21 | 2,46 ± 2,19 | 40,1 ± 13,7 |
  | Ackley | 20,2 ± 0,27 | 9,37 ± 2,09 | 19,2 ± 0,85 |
  | Rastrigin | 359 ± 16 | 71,4 ± 11,9 | 237 ± 47 |
  | Zakharov | 642 ± 76 | 539 ± 111 | 608 ± 66 |

  Lo publicado queda entre ambas lecturas; el código de los autores hace
  algo que el paper no especifica, y no está publicado (solo existe la
  implementación de mealpy, que no es de los autores). La parte de la
  Tabla 4 del test es informativa.

**Sanidad en CEC 2022** (D = 10, 5·10⁴ FES, 5 semillas, error mediano):
f1 8 707, f4 53,7, f7 22,8 (búsqueda aleatoria: 9 692, 62,8, 73,2). El
rendimiento débil en dimensión ≥ 10 es el que reporta el propio paper
(Tabla 4: Sphere 30-D 40,1 ± 13,7 con 40 040 FES), no un error.

**Costo**: ~220 µs/eval en un solo proceso, el mayor de la familia (mealpy
reconstruye la distribución de clones y la lista de índices en cada
mutación). Registrado como `gco` (familia `inmune`) en `COMPETIDORES`,
`REFERENCIA`, `ALGORITMOS_VALIDOS`, `_COSTO_US_POR_EVAL`,
`FAMILIA_ALGORITMO` y `ALGORITMOS_REFERENCIA`.

**Decisión (2026-11-10)**: se usa GCO con la implementación de mealpy tal
cual (r₁, r₂, r₃ distintos), documentando que reproduce la Tabla 2 del paper
(2-D) y no la Tabla 4 (30-D) por la ambigüedad del muestreo. La familia
inmune queda con CLONALG (aisp), opt-aiNet (implementación propia) y GCO.
Pendiente opcional: pedir el código a los autores (Universidad de
Guadalajara) para resolver la ambigüedad.
