# Algoritmos de referencia: parrilla completa en el Mac Studio

Procedimiento para correr en el Mac Studio (Apple M2 Ultra) **la parrilla
completa** de los **6 algoritmos de referencia por familia** seleccionados
(2 por familia), en los 4 tiers de MaxFES:

| Familia | Algoritmos (`--algoritmo`) |
|---|---|
| Evolutivos | `ga` (GA real, pymoo), `jade` (JADE, mealpy corregido) |
| Enjambre | `clpso` (CLPSO, port de los autores), `abc` (ABC, NiaPy corregido) |
| Inmunes | `clonalg` (CLONALG, aisp), `optainet` (opt-aiNet, implementación propia) |

Quedan fuera `jde`, `gwo` y `gco` (el más caro de los 9 y el único que no
reproduce todas las tablas de su paper).

6 algoritmos × 12 funciones × 2 dimensiones × 4 tiers (`5×10⁴`, `5×10⁵`,
`5×10⁶`, `5×10⁷`) × 51 semillas = **29 376 corridas**.

## Duración esperada

> **ETA: ~90 h (~3,8 días), rango 72–98 h** con 24 workers. Es una
> extrapolación: el paso 5 la confirma con un piloto en el Mac antes de
> lanzar.

| Tier | Corridas | Duración estimada |
|---|---|---|
| `5×10⁴` + `5×10⁵` + `5×10⁶` | 22 032 | ~9 h |
| `5×10⁷` | 7 344 | ~81 h |
| **Total** | 29 376 | **~90 h** |

| Algoritmo | ga | jade | clpso | abc | clonalg | optainet |
|---|---|---|---|---|---|---|
| Horas (4 tiers) | ~26 | ~36 | ~18 | ~4 | ~3 | ~3 |

Cómo se estimó:

- **Costo en el Xeon:** piloto de costo bajo carga completa (2026-11-15/17,
  Xeon E5-2673 v3 con Windows, 24 workers, MaxFES = 5×10⁵, 216 corridas por
  algoritmo, todas `ok`):

  | µs/eval (D=10 / D=20) | ga | jade | clpso | abc | clonalg | optainet |
  |---|---|---|---|---|---|---|
  | Xeon, Windows | 236.6 / 223.5 | 319.4 / 322.4 | 158.3 / 164.2 | 39.0 / 40.8 | 24.6 / 26.9 | 21.7 / 24.4 |

  Con estos valores, la parrilla completa tardaría ~630 h (~26 días) en el
  Xeon. Se supone que el costo por evaluación no cambia con MaxFES (el
  piloto solo midió 5×10⁵).
- **Factor Mac:** el Mac rinde **~7×** el Xeon con Windows (rango
  6,4–8,8×): 630 h / 7 ≈ 90 h. El factor sale de PSO, DE, MadDE y
  NL-SHADE-RSP en la parrilla principal (Mac en 5×10⁷ contra Ubuntu en
  5×10⁶: 5,1–7,9×) y de una calibración Windows contra Ubuntu en el mismo
  Xeon (1,09–1,44×). Es **indirecto**: no se ha medido ningún algoritmo de
  referencia en el Mac.
- Referencia: el tier 5×10⁷ de la parrilla principal tardó **58,5 h** en
  el Mac, contra ~311 h estimadas para el Xeon.

---

## 1. Entorno base

Se parte del entorno ya instalado para la parrilla principal (README_MAC
pasos 1–3: Xcode CLT, `libomp`, Python 3.14 arm64, `~/Fase_pruebas/.venv`).
Si el Mac es otro, hacer primero esos pasos.

## 2. Código actualizado

```bash
cd ~/Fase_pruebas
git status --short          # no debe haber cambios en archivos versionados
git pull
git log -1 --format=%h      # anotar el hash: queda en la columna `version`
```

No hacer `git pull` mientras corre la parrilla.

## 3. Dependencias nuevas, con las versiones del Windows

Los algoritmos de referencia usan cuatro librerías que el Mac no tiene
(NiaPy, pymoo, aisp y mealpy). Se fijan las versiones validadas en el
Windows (2026-11-19), incluidas sus dependencias:

```bash
cd ~/Fase_pruebas
source .venv/bin/activate

cat > /tmp/requirements-ref-mac.txt <<'EOF'
niapy==2.7.1
pymoo==0.6.2
aisp==0.5.5
mealpy @ git+https://github.com/thieu1995/mealpy.git@46b9850bdd6949a992372d03ed280b6302807d8d
autograd==1.9.1
moocore==0.3.2
matplotlib==3.11.1
alive-progress==3.3.0
Deprecated==3.0.1
EOF
pip install -r /tmp/requirements-ref-mac.txt
```

`mealpy` se instala desde GitHub con el commit fijado (necesita `git`, ya
instalado con las herramientas de Xcode). La 3.0.3 de PyPI exige
`numpy<=1.26` y bajaría el numpy del proyecto.

Comprobar las versiones:

```bash
python -c "import numpy, numba, scipy, pandas, niapy, pymoo, aisp, mealpy; \
print(numpy.__version__, numba.__version__, scipy.__version__, pandas.__version__, \
niapy.__version__, pymoo.__version__, aisp.__version__, mealpy.__version__)"
# Esperado: 2.5.2 0.67.0 1.18.1 3.0.5 2.7.1 0.6.2 0.5.5 3.1.0
```

Si `pip` cambió numpy, numba, scipy o pandas, reinstalar la versión de la
lista de README_MAC paso 3 (`pip install numpy==2.5.2`, etc.).

## 4. Verificación antes de lanzar

```bash
# Evaluador CEC 2022 fiel al C oficial.
# Termina con "Fidelidad verificada: 12 funciones × 2 dimensiones."
python tests/test_cec2022_numba_paridad.py

# Los 9 de referencia en D=10 y 20: MaxFES exacto, traza monótona, determinismo.
# Termina con "[OK] 9 algoritmos de referencia × 2 dimensiones"
python tests/test_referencias_familia.py

# Adaptadores NiaPy/mealpy.
# Termina con "[OK] N pruebas de los adaptadores NiaPy/mealpy"
python tests/test_adaptadores_librerias.py
```

Si alguno falla, **no lanzar**. Lo más probable es una librería con otra
versión (paso 3).

## 5. Piloto en el Mac (~10–20 min)

Mide el costo real de los 6 algoritmos en el Mac antes de comprometer ~4
días. Corre las mismas 1 296 corridas del piloto del Xeon (5×10⁵, semillas
0–8) **en la carpeta definitiva**: son corridas de la parrilla, así que el
paso 7 no las repite.

```bash
python experimentos/grid.py --salida resultados_ref_MAC \
    --algoritmo jade,ga,clpso,abc,clonalg,optainet \
    --maxfes 500000 --semillas 0-8 --workers $(sysctl -n hw.physicalcpu) \
    > piloto_ref_MAC.log 2>&1
tail -1 piloto_ref_MAC.log      # "Hecho: 1296 ok, 0 fallo en X h."
```

Calcular la duración esperada de la parrilla con lo medido:

```bash
python - <<'EOF'
import glob, pandas as pd
XEON = {"ga": (236.6, 223.5), "jade": (319.4, 322.4), "clpso": (158.3, 164.2),
        "abc": (39.0, 40.8), "clonalg": (24.6, 26.9), "optainet": (21.7, 24.4)}
d = pd.concat(pd.read_parquet(f) for f in glob.glob("resultados_ref_MAC/corridas/*.parquet"))
d = d[d["max_fes"] == 500_000]
print("corridas:", len(d), "| estados:", d["estado"].value_counts().to_dict())
d["us"] = d["tiempo_seg"] / d["max_fes"] * 1e6
us = d.pivot_table(index="algoritmo", columns="dim", values="us", aggfunc="mean")
# 612 corridas por dimensión y tier (12 funciones × 51 semillas), 24 workers
TIERS = {"5e4-5e6": 50_000 + 500_000 + 5_000_000, "5e7": 50_000_000}
tot = dict.fromkeys(TIERS, 0.0)
for a, (x10, x20) in XEON.items():
    m = us.loc[a, 10] + us.loc[a, 20]
    h = {t: 612 * fes * m * 1e-6 / 3600 / 24 for t, fes in TIERS.items()}
    for t in TIERS:
        tot[t] += h[t]
    print(f"{a:9s} µs/eval {us.loc[a, 10]:6.1f} / {us.loc[a, 20]:6.1f} | "
          f"Xeon/Mac {(x10 + x20) / m:4.1f}x | {sum(h.values()):5.1f} h")
print(f"Tiers 5e4-5e6: ~{tot['5e4-5e6']:.0f} h | tier 5e7: ~{tot['5e7']:.0f} h | "
      f"TOTAL: ~{sum(tot.values()):.0f} h ({sum(tot.values()) / 24:.1f} días)")
EOF
```

- Debe dar **1 296 corridas `ok`**.
- Si el total queda cerca de **~90 h** (72–98 h), la estimación se confirma.
  Si se aleja mucho, revisar que Python no corra bajo Rosetta (README_MAC
  paso 1) antes de lanzar.

## 6. Revisar qué se va a correr (dry-run)

```bash
python experimentos/grid.py --salida resultados_ref_MAC \
    --algoritmo jade,ga,clpso,abc,clonalg,optainet \
    --workers $(sysctl -n hw.physicalcpu) --dry-run
```

La primera línea debe decir
`Parrilla: 29376 corridas (jade, ga, clpso, abc, clonalg, optainet) | ya hechas: 1296 | pendientes: 28080`.
Sin `--maxfes`, `grid.py` usa los 4 tiers. Las horas que muestra **no
aplican**: están calibradas para el Xeon y además con los costos
provisionales de `grid.py`, anteriores al piloto. La cifra para el Mac es
la del paso 5.

Los algoritmos van de más caro a más barato a propósito: `grid.py` los
encola en ese orden y así la cola final queda con las corridas cortas.

## 7. Lanzar la parrilla

```bash
cd ~/Fase_pruebas
source .venv/bin/activate

WORKERS=$(sysctl -n hw.physicalcpu) && echo "Workers: $WORKERS" && \
nohup caffeinate -dims python experimentos/grid.py \
    --salida resultados_ref_MAC \
    --algoritmo jade,ga,clpso,abc,clonalg,optainet \
    --workers "$WORKERS" \
    > grid_ref_MAC.log 2>&1 &
```

- `caffeinate -dims`: impide que el Mac entre en reposo mientras dure la
  ejecución.
- `nohup … &`: corre en segundo plano; se puede cerrar la Terminal.

Antes de dejarlo varios días, aplicar la configuración de README_MAC paso 7:
sin reposo automático, sin actualizaciones automáticas, sin modo de bajo
consumo.

## 8. Monitorear, detener y retomar

```bash
tail -f ~/Fase_pruebas/grid_ref_MAC.log                 # progreso y ETA (Ctrl+C no detiene nada)
pgrep -fl grid.py                                       # ¿sigue vivo?
grep -c "  ok  " ~/Fase_pruebas/grid_ref_MAC.log        # corridas ok
grep    "  fallo  " ~/Fase_pruebas/grid_ref_MAC.log     # fallidas (traceback en resultados_ref_MAC/fallos/)
```

El ETA del log promedia lo hecho hasta el momento, y el ritmo cambia mucho
entre algoritmos y tiers: tomarlo como orientativo y fiarse de la
estimación del paso 5.

**Detener:** `pkill -f experimentos/grid.py; pkill -f multiprocessing.spawn`

**Retomar** tras una detención, un corte de luz o un reinicio: relanzar el
comando del paso 7 con la misma `--salida resultados_ref_MAC` y otro nombre
de log (`grid_ref_MAC_2.log`). `grid.py` salta las corridas ya completas;
las que estaban a medias se repiten desde cero.

## 9. Al terminar: consolidar y llevar al Windows

Repetir el dry-run del paso 6: debe decir
**`ya hechas: 29376 | pendientes: 0`**. Si faltan, relanzar el paso 7.

Consolidar en el Mac y empaquetar sin los archivos ocultos de macOS (`._*`,
`.DS_Store`). Si se copian, `consolidar.py` intentaría leer los `._*.parquet`
y fallaría:

```bash
cd ~/Fase_pruebas
python experimentos/consolidar.py --salida resultados_ref_MAC
COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' \
    -czf resultados_ref_MAC.tar.gz resultados_ref_MAC
```

El consolidado queda en `resultados_ref_MAC/corridas.parquet`. Los
algoritmos de referencia no generan `corridas_propuesto/` ni
`logs_middleware/`. En el Windows, descomprimir en la raíz del proyecto con
`tar -xzf resultados_ref_MAC.tar.gz`.

**`analisis/main.py` todavía no sirve para esta parrilla:** `cargar_corridas()`
exige exactamente los 6 algoritmos de la parrilla principal, y los de
referencia están aparte en `ALGORITMOS_REFERENCIA` (`analisis/datos.py`).
Falta decidir si entran al mismo ranking que los competidores o se analizan
por separado, y adaptar `analisis/` según eso.

## 10. Qué documentar en la tesis

- La parrilla de los algoritmos de referencia se corrió completa en el Mac
  Studio (Apple M2 Ultra, 24 núcleos, 64 GB, arm64). La parrilla principal
  se repartió entre el Xeon E5-2673 v3 (tiers 5×10⁴–5×10⁶) y el Mac
  (5×10⁷): las comparaciones de los algoritmos de referencia contra los de
  la parrilla principal en los tiers 5×10⁴–5×10⁶ cruzan hardware.
- Los resultados no son idénticos bit a bit entre arquitecturas (README_MAC
  §10). `tiempo_seg` mezcla hardware entre las dos parrillas; si se reportan
  tiempos, separarlos por la columna `host`.
- Se usaron 6 de los 9 algoritmos de referencia (2 por familia); motivo de
  la selección: costo de cómputo y, en GCO, la reproducción incompleta de su
  paper.
