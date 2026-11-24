# Pruebas BBOB (Black-Box Optimization Benchmarking) con los 12 algoritmos

Segundo benchmark, además de CEC 2022: la suite **`bbob`** (sin ruido) de
la plataforma **COCO** (Hansen et al., 2021, *COCO: a platform for
comparing continuous optimizers in a black-box setting*, Optimization
Methods and Software 36(1)). Se corre con los **12 algoritmos** del
proyecto:

| Grupo | Algoritmos (`--algoritmo`) |
|---|---|
| Parrilla principal | `middleware`, `pso`, `de`, `cmaes`, `madde`, `nlshadersp` |
| Referencia por familia (los 6 del Mac, README_MAC_REFERENCIA.md) | `jade`, `ga`, `clpso`, `abc`, `clonalg`, `optainet` |

**Estado (2026-11-22): implementada la opción A** (protocolo nativo de
COCO), en la rama `bbob`. Las demás opciones quedan documentadas abajo por
si se decide cambiar.

> **Alcance:** el README principal fija "evaluación solo sobre CEC 2022…
> no reabrir sin motivo". Agregar BBOB reabre ese punto: hay que registrar
> el motivo como decisión nueva (D-13) en `docs/configuracion_experimental.md`.

---

## 1. La suite `bbob`

- **24 funciones** en 5 grupos: separables (f1–f5), condicionamiento bajo o
  moderado (f6–f9), condicionamiento alto y unimodales (f10–f14),
  multimodales con estructura global adecuada (f15–f19) y multimodales con
  estructura global débil (f20–f24).
- **Dimensiones:** 2, 3, 5, 10, 20 y 40.
- **Instancias:** cada función tiene versiones desplazadas y rotadas, con
  otro óptimo x\* y otro f(x\*). En COCO las instancias reemplazan a las
  semillas: cada instancia se corre una vez y cuenta como una repetición
  independiente. Se usan las 15 clásicas (1–15).
- **Dominio:** [-5, 5]^D. El óptimo está dentro de [-4, 4]^D.
- **Evaluador:** el código C oficial de COCO (paquete `coco-experiment`),
  sin ports propios. A diferencia de CEC 2022 (D-9), no hay nada que
  validar contra otra implementación.
- **Métrica estándar de COCO:** tiempo de ejecución (evaluaciones) hasta
  alcanzar objetivos de precisión de 10² a 10⁻⁸. Se reporta como ECDF,
  sin un error fijo a presupuesto fijo.

---

## 2. Opciones evaluadas

Costo calculado con el µs/evaluación de cada algoritmo medido en el Xeon
con 24 workers (pilotos de D-8 y de los algoritmos de referencia).
Evaluar una función BBOB cuesta 0,9–2,5 µs (medido), igual que CEC 2022.
El observador de COCO suma unos 15 µs por evaluación.

| Opción | Diseño | Corridas (12 alg.) | Xeon, 24 workers | Mac (≈3–7× el Xeon) |
|---|---|---|---|---|
| **A — implementada** | Protocolo COCO: 24 funciones × D{2,3,5,10,20,40} × instancias 1–15, una corrida por instancia, presupuesto **10⁴·D** | 25 920 | ≤ ~8 h (`--dry-run`, cota alta). **Piloto: ~56 h-núcleo** (§8) | ~2–3 h |
| A' | Igual que A con presupuesto **10⁵·D** | 25 920 | ≤ ~59 h (`--dry-run`); menos de 10× A, porque el costo fijo del middleware no escala | ≤ ~1 día |
| B | Protocolo de la tesis: 24 funciones × D{10,20} × 51 semillas × MaxFES {5·10⁴, 5·10⁵, 5·10⁶, 5·10⁷} | 117 504 | ~2 000 h | ~12–28 días |
| B' | B sin el tier 5·10⁷ | ~88 000 | ~200 h | ~1,5–3 días |

**Por qué A:**

1. **Costo:** cabe en horas y no compite con las parrillas CEC.
2. **Comparabilidad:** los archivos del observador de COCO se procesan con
   `cocopp`. Eso da las figuras estándar (ECDF, ERT) y la comparación
   directa contra los cientos de algoritmos publicados en el archivo de
   datos de COCO (BIPOP-CMA-ES, variantes de DE y PSO, etc.), sin correr
   nada más.
3. **Metodología:** es el protocolo con el que se publica en BBOB. Aplicar
   51 semillas y 4 tiers de MaxFES sobre BBOB (opción B) sería poco
   habitual y cuesta semanas.
4. **Rankings igual que CEC:** con las mismas corridas también salen el
   error medio, los rangos con empate a 1e-8, Friedman y la CD de Nemenyi,
   por dimensión (`analisis/bbob.py`). Con k = 12 y N = 24 funciones,
   CD ≈ 3,40.

**Diferencia principal entre A y A':** con 10⁴·D, el middleware y los
algoritmos de presupuesto adaptativo (NL-SHADE-RSP, MadDE) tienen poco
margen en D = 2–5 (2·10⁴ a 5·10⁴ evaluaciones). 10⁵·D se acerca más a los
tiers chicos de CEC (5·10⁴–5·10⁵ en D = 10–20). Se pueden correr las dos
en carpetas distintas: `--presupuesto 100000 --salida resultados_bbob_1e5`.

**Opción B, si se decide.** No está implementada. Haría falta:
- En `grid_bbob.py`, iterar sobre MaxFES fijos en vez de `presupuesto·D`.
- Pasar de instancias a semillas: 51 corridas por función, sea con
  instancias 1–51 (COCO las genera, pero no son las "oficiales") o con la
  instancia 1 y 51 semillas.
- Ajustar `analisis/bbob.py` a la configuración (dim × MaxFES).

`ProblemaBBOB` y `corrida_bbob.py` sirven tal cual.

---

## 3. Qué se implementó (opción A)

| Archivo | Qué hace |
|---|---|
| [problems/bbob_wrapper.py](problems/bbob_wrapper.py) | `ProblemaBBOB(funcion, dim, instancia, carpeta_observador=…)`: **misma interfaz que `ProblemaCEC2022`** (`evaluar`, `fes`, `limites`, `optimo_global`, `error`, checkpoints). Los 12 algoritmos corren sin cambios. Opcionalmente adjunta el observador de COCO. |
| [experimentos/corrida_bbob.py](experimentos/corrida_bbob.py) | `correr_corrida_bbob(algoritmo, funcion, dim, instancia, max_fes)`: reutiliza los runners de `corrida.py` (misma configuración congelada) y devuelve las filas §11.2/§11.3 con `benchmark="BBOB"`. |
| [experimentos/grid_bbob.py](experimentos/grid_bbob.py) | Parrilla reanudable y paralela, con `--dry-run` y estimación (como `grid.py`). Default: los 12 algoritmos, 24 funciones, 6 dimensiones, 15 instancias y 10⁴·D. |
| [analisis/bbob.py](analisis/bbob.py) | Error medio, rangos (globales y por grupo de funciones), Friedman, CD de Nemenyi con diagrama y tasa de éxito, por dimensión. |
| [tests/test_bbob.py](tests/test_bbob.py) | Los 12 algoritmos en las 6 dimensiones (MaxFES respetado, traza monótona), determinismo, Fase 2 del middleware en D = 2 y 40, y observador sin duplicados. |
| [experimentos/esquema.py](experimentos/esquema.py) | Cambio retrocompatible: `meta` puede traer `benchmark`, `conjunto` y `run_id`; sin ellos, los valores de CEC 2022 de siempre. Nuevo `run_id_bbob()`. |

**Decisiones del protocolo:**

- **Semilla:** la del algoritmo en la instancia *i* es `SEMILLAS[i-1]`, de
  la misma lista fija de 51. En las filas, `idx_semilla = instancia - 1` y
  `semilla` es la semilla usada.
- **run_id:** `bbob__<alg>__fFF__dDD__iII__fes<MaxFES>`. El prefijo impide
  chocar con las corridas CEC aunque compartan carpeta. Igual conviene
  usar una carpeta propia (`resultados_bbob/`).
- **`version`:** `git:<commit>|eval:bbob-coco2.8.4[|alg:…]`. `grid_bbob.py`
  vuelve a correr las corridas hechas con otra versión de COCO.
- **f(x\*):** sale de `cocoex.Problem._best_value`, un atributo privado
  porque COCO no lo publica para que el optimizador no lo use. Aquí solo lo
  lee el harness para calcular el error; ningún algoritmo lo usa. El test
  verifica que ningún punto quede por debajo de f(x\*).
- **Presupuesto:** `MaxFES = presupuesto · D`. PSO, DE y el middleware no
  inician una generación que no cabe en el presupuesto. Los competidores
  cortan en la evaluación exacta (`Presupuesto`), igual que en CEC.
- **Configuración de los algoritmos:** la misma congelada de CEC 2022, sin
  re-ajustar para BBOB. Los umbrales del middleware (θ_W = 60 % del
  diámetro, warm-up 15 %, ventanas de 10 % de MaxFES) son relativos al
  dominio y al presupuesto, así que escalan solos.
- **Observador de COCO:** escribe en `<salida>/coco/<alg>/fFF_dDD_iII_fesN/`,
  una carpeta por corrida para que los workers no se pisen. `cocopp`
  busca los `.info` de forma recursiva y une las instancias de cada
  algoritmo (verificado, §8). Antes de cada corrida se borra su carpeta: si existiera, COCO
  crearía `-0001` y `cocopp` contaría dos veces una corrida reanudada. Si
  una corrida falla, también se borra su carpeta.

**Detalles técnicos de COCO encontrados al integrarlo:**

- `result_folder` solo acepta rutas **relativas a `./exdata/`**. Con una
  ruta absoluta falla en Windows (`unable to create exdata\C:`). El wrapper
  le pasa la ruta relativa desde `exdata/`, por lo que **hay que correr
  siempre desde la raíz del proyecto**. La carpeta `exdata/` queda vacía y
  está en `.gitignore`.
- Si COCO no puede abrir un archivo, **termina el proceso** desde C, sin
  excepción de Python. En Windows pasa con rutas de 260 caracteres o más
  (MAX_PATH). En un worker de `grid_bbob.py` eso dejaría la parrilla
  colgada, así que el wrapper verifica el largo antes y lanza `ValueError`.
  Usar una `--salida` corta dentro del proyecto.

---

## 4. Instalación

En el entorno del proyecto (Windows o Mac):

```bash
pip install "coco-experiment==2.8.4" "cocopp==2.9.4"
```

`coco-experiment` 2.8.4 tiene wheel cp314 para **Windows x64** (instalado
y probado) y **macOS arm64** (la wheel existe, no se ha probado en el
Mac). No hace falta compilar. `cocopp` es Python puro y no cambió las
versiones de numpy, scipy, pandas ni matplotlib.

## 5. Verificación antes de lanzar

```bash
python tests/test_bbob.py          # ~2 min. Termina con "[OK] integración BBOB: 12 algoritmos × 6 dimensiones"
python tests/test_experimentos.py  # el harness CEC sigue igual tras el cambio en esquema.py
```

## 6. Lanzar

> **Mac Studio:** instructivo paso a paso (clon aparte para no tocar la
> parrilla de referencia en curso, entorno, verificación, lanzamiento,
> análisis y traslado al Windows) en [README_MAC_BBOB.md](README_MAC_BBOB.md).

```bash
# Qué se va a correr y estimación (calibrada para el Xeon)
python experimentos/grid_bbob.py --dry-run

# Windows
python experimentos/grid_bbob.py --salida resultados_bbob > grid_bbob.log 2>&1
```

En el Mac, seguir [README_MAC_BBOB.md](README_MAC_BBOB.md), sin copiar
este comando: allá se usa un clon aparte.

Opciones útiles: `--presupuesto 100000` (opción A'), `--sin-observador`
(sin archivos de COCO, unos 15 µs menos por evaluación), y `--algoritmo`,
`--funciones`, `--dims` e `--instancias` para subconjuntos. Para
**reanudar**, relanzar con la misma `--salida`: se saltan las corridas ya
completas.

Las corridas se encolan de la más cara a la más barata (D = 40 y los
algoritmos lentos primero), así la cola final queda con las cortas.

## 7. Consolidar y analizar

```bash
python experimentos/consolidar.py --salida resultados_bbob
python analisis/bbob.py --resultados resultados_bbob        # -> resultados_bbob/analisis_bbob/

# Análisis estándar de COCO (ECDF, ERT, tablas). Una carpeta por algoritmo:
python -m cocopp resultados_bbob/coco/middleware resultados_bbob/coco/cmaes resultados_bbob/coco/de
# Comparar contra el archivo de datos de COCO (lo descarga cocopp), p. ej.:
python -m cocopp resultados_bbob/coco/middleware bbob/2009/BIPOP-CMA-ES
```

`analisis/bbob.py` exige las 25 920 corridas en estado `ok` y un solo
presupuesto. `--permitir-incompleto` analiza lo que haya, avisando.

`analisis/main.py` (CEC 2022) **no** lee las corridas BBOB, y así debe
ser: son conjuntos aparte.

---

## 8. Piloto (2026-11-22)

En el portátil de desarrollo (Windows, 6 núcleos, 6 workers, no es el
Xeon): 12 algoritmos × f1, f15, f24 × D = 2, 10, 40 × instancia 1, con
10⁴·D y el observador activado. Resultado: **108/108 `ok` en 3 min**.

Segundos por corrida (media de las 3 funciones):

| Algoritmo | D=2 | D=10 | D=40 | Parrilla completa (h-núcleo, interpolando D=3, 5, 20) |
|---|---|---|---|---|
| middleware | 14.5 | 19.6 | 49.3 | 14.4 |
| jade | 2.4 | 12.7 | 52.8 | 10.4 |
| cmaes | 1.2 | 6.2 | 38.9 | 6.9 |
| ga | 1.6 | 6.8 | 34.2 | 6.4 |
| de | 1.4 | 5.8 | 26.4 | 5.1 |
| clpso | 1.2 | 5.3 | 23.0 | 4.5 |
| nlshadersp | 1.1 | 2.6 | 7.9 | 1.9 |
| madde | 2.2 | 2.0 | 6.5 | 1.9 |
| abc | 0.5 | 1.4 | 7.0 | 1.4 |
| clonalg | 0.4 | 1.0 | 5.6 | 1.1 |
| optainet | 0.4 | 0.9 | 4.8 | 0.9 |
| pso | 0.3 | 0.8 | 4.2 | 0.8 |
| **Total** | | | | **~56 h-núcleo** |

- **Parrilla completa, opción A:** ~56 h-núcleo. Son **~9 h en el
  portátil** (6 workers) y, por extrapolación, **~2–3 h en el Mac** (24
  núcleos). El `--dry-run` calibrado al Xeon da ~8 h con 24 workers, y en
  este piloto sobreestimó unas 3,5 veces lo medido en el portátil. Tomarlo
  como cota alta.
- **D = 40:** el costo por evaluación casi no crece con D (CMA-ES pasa de
  62 a 97 µs/eval). El supuesto "×1,5–2 por D = 40" del análisis de
  factibilidad sobraba.
- **Middleware:** domina el costo por la Fase 2 (XGBoost + FastSHAP), que
  tiene costo fijo por activación. Por eso su µs/eval baja al crecer D,
  porque crece el presupuesto (D=2: 724 µs/eval; D=40: 123 µs/eval).
- **`cocopp` verificado:** lee una carpeta por corrida y une las
  instancias (3 instancias en 3 carpetas → un conjunto con `nruns = 3`).
  La leyenda de cada algoritmo es la ruta de su carpeta (`coco/pso`).
- **`analisis/bbob.py` verificado** sobre el piloto con
  `--permitir-incompleto`.

---

## 9. Qué documentar en la tesis

- BBOB se evaluó con el **protocolo de COCO**: 15 instancias por
  función × dimensión y presupuesto 10⁴·D. No es el protocolo de CEC
  (51 semillas, 4 tiers de MaxFES), así que los números de las dos
  evaluaciones no se comparan directamente: cada benchmark tiene su propio
  ranking.
- Los algoritmos usan la **misma configuración congelada** de CEC 2022, sin
  re-ajustarla para BBOB. Así BBOB funciona como prueba de generalización
  de esa configuración.
- La evaluación usa el código C oficial de COCO 2.8.4.
- Las comparaciones contra el archivo de datos de COCO mezclan hardware y
  versiones de COCO, aunque las instancias 1–15 sí son las mismas que en
  BBOB 2009. Son comparaciones en evaluaciones, no en tiempo.
- Con k = 12 y N = 24 funciones por dimensión, CD ≈ 3,40 (α = 0,05). El
  ranking por dimensión no promedia entre dimensiones, igual que en CEC no
  se promedia entre configuraciones.
