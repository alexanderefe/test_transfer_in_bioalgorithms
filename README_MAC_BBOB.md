# BBOB: parrilla completa en el Mac Studio

Procedimiento para correr en el Mac Studio (Apple M2 Ultra, 24 núcleos) la
**parrilla BBOB de la opción A** con los 12 algoritmos. El contexto, las
opciones y el diseño están en [README_BBOB.md](README_BBOB.md).

| | |
|---|---|
| Algoritmos | `middleware`, `pso`, `de`, `cmaes`, `madde`, `nlshadersp`, `jade`, `ga`, `clpso`, `abc`, `clonalg`, `optainet` |
| Problemas | suite `bbob` de COCO: 24 funciones × D ∈ {2, 3, 5, 10, 20, 40} × instancias 1–15 |
| Presupuesto | 10⁴·D evaluaciones por corrida |
| Corridas | 12 × 24 × 6 × 15 = **25 920** |
| Rama | `bbob` |

## Duración esperada

> **ETA: ~2–3 h con 24 workers, rango razonable 2–5 h.** Es una
> extrapolación: el piloto en el portátil Windows midió ~56 h-núcleo para
> la parrilla completa (README_BBOB.md §8). El Mac no se ha medido.
> Confirmarlo con el ETA del log tras 15–20 min (paso 7).

Las horas que muestra `grid_bbob.py --dry-run` (~8 h) están calibradas
para el Xeon y sobreestiman: no aplican al Mac.

---

## 1. Antes de empezar: no tocar `~/Fase_pruebas`

En `~/Fase_pruebas` corre la parrilla de los algoritmos de referencia
(README_MAC_REFERENCIA.md). Esa carpeta **no se toca**: ni `git pull` ni
`git checkout bbob`. La parrilla recicla sus workers cada 20 corridas, y
cada worker nuevo vuelve a importar el código desde disco y lee
`git rev-parse` para la columna `version`. Cambiar la rama mezclaría
código y commits dentro de esa parrilla.

Por eso BBOB va en un **clon aparte, `~/Fase_pruebas_bbob`**, que se
puede preparar ahora mismo (pasos 2–5).

**Cuándo lanzar (paso 6):**

- **Recomendado: cuando termine la parrilla de referencia.** Comprobarlo
  con `pgrep -fl experimentos/grid.py`: si no imprime nada, terminó.
- Si se lanza mientras la otra sigue, las dos se reparten los 24 núcleos.
  Las dos tardan más y los ETA de ambas dejan de servir. Los resultados
  siguen siendo válidos: el determinismo no depende de la carga, solo
  cambia `tiempo_seg`.

## 2. Clonar la rama `bbob`

```bash
cd ~
git clone --branch bbob https://github.com/alexanderefe/test_transfer_in_bioalgorithms.git Fase_pruebas_bbob
cd ~/Fase_pruebas_bbob
git branch --show-current        # debe decir: bbob
git log -1 --format=%h           # anotar el hash: queda en la columna `version`
```

Clonar desde git, sin copiar carpetas: así no se arrastran `__pycache__`
ni el caché de Numba de otra arquitectura. No hacer `git pull` en este
clon mientras corre la parrilla BBOB.

## 3. Entorno: reutilizar el `.venv` de `~/Fase_pruebas`, agregando solo COCO

El entorno de `~/Fase_pruebas/.venv` ya tiene exactamente las versiones
validadas (README_MAC paso 3 + README_MAC_REFERENCIA paso 3). Solo faltan
COCO y su post-procesador. Se instalan con `--no-deps`, para que `pip` no
toque ningún paquete que esté usando la parrilla en curso:

```bash
source ~/Fase_pruebas/.venv/bin/activate
pip install --no-deps "coco-experiment==2.8.4" "cocopp==2.9.4" "platformdirs==4.14.5"
```

> **"No such file or directory" al activar:** el clon `~/Fase_pruebas_bbob`
> **no trae `.venv`**, porque está en `.gitignore`. Entonces
> `source .venv/bin/activate` dentro del clon falla. Hay que activar el
> entorno de `~/Fase_pruebas` con la ruta completa, como arriba. Si eso
> también falla, el entorno está en otro lugar. Para ver qué Python usa
> la parrilla en curso:
>
> ```bash
> ps -o command= -p "$(pgrep -f experimentos/grid.py | head -1)"
> # p. ej. /Users/<usuario>/Fase_pruebas/.venv/bin/python experimentos/grid.py ...
> ls -d ~/Fase_pruebas/.venv ~/Fase_pruebas/venv 2>/dev/null
> ```
>
> Activar `<carpeta de ese python>/activate`, es decir, la ruta que
> muestra `ps` con `python` reemplazado por `activate`. Si no queda
> ningún entorno, crear uno propio en el clon (§3b).

- `coco-experiment` trae la wheel nativa `cp314-macosx_11_0_arm64`, sin
  compilar.
- `cocopp` necesita `matplotlib` (ya instalado, 3.11.1), `numpy` y
  `platformdirs` (el único nuevo, fijado a la versión del Windows).
- Instalar paquetes nuevos no afecta a la parrilla en curso: sus procesos
  no los importan.

Comprobar versiones. **numpy, numba, scipy y pandas deben ser las mismas
de antes:**

```bash
python -c "import platform, numpy, numba, scipy, pandas, cocoex, cocopp; \
print(platform.machine(), numpy.__version__, numba.__version__, scipy.__version__, \
pandas.__version__, cocoex.__version__, cocopp.__version__)"
# Esperado: arm64 2.5.2 0.67.0 1.18.1 3.0.5 2.8.4 2.9.4
```

### 3b. Alternativa: entorno propio en el clon

Independiente del de `~/Fase_pruebas`, con las mismas versiones
(README_MAC paso 3 + README_MAC_REFERENCIA paso 3 + COCO):

```bash
cd ~/Fase_pruebas_bbob
python3.14 -m venv .venv
source .venv/bin/activate

cat > /tmp/requirements-bbob-mac.txt <<'EOF'
setuptools==81.0.0
numpy==2.5.2
scipy==1.18.1
opfunu==1.0.1
torch==2.13.0
xgboost==3.4.1
scikit-learn==1.9.0
tqdm==4.70.0
pandas==3.0.5
pyarrow==25.0.1
numba==0.67.0
llvmlite==0.49.0
cma==4.5.0
niapy==2.7.1
pymoo==0.6.2
aisp==0.5.5
mealpy @ git+https://github.com/thieu1995/mealpy.git@46b9850bdd6949a992372d03ed280b6302807d8d
autograd==1.9.1
moocore==0.3.2
matplotlib==3.11.1
alive-progress==3.3.0
Deprecated==3.0.1
coco-experiment==2.8.4
cocopp==2.9.4
platformdirs==4.14.5
EOF
pip install -r /tmp/requirements-bbob-mac.txt
```

Si `pip` cambió numpy, numba, scipy o pandas, reinstalar la versión de la
lista. Comprobar con el mismo comando de versiones del paso 3. Con esta
alternativa, en los pasos 6 y 8 activar `source .venv/bin/activate` desde
`~/Fase_pruebas_bbob`, en vez de `~/Fase_pruebas/.venv`.

## 4. Verificación antes de lanzar

Desde la raíz del clon, con el entorno activado:

```bash
cd ~/Fase_pruebas_bbob

# Los 12 algoritmos × 6 dimensiones, determinismo, Fase 2 del middleware y
# observador de COCO. ~2 min en el portátil; más si la otra parrilla sigue corriendo.
# Termina con "[OK] integración BBOB: 12 algoritmos × 6 dimensiones"
python tests/test_bbob.py

# El harness CEC sigue igual (cambio en experimentos/esquema.py).
# Termina con "[OK] smoke del harness experimental (...)"
python tests/test_experimentos.py
```

Si alguno falla, **no lanzar**. Lo más probable es una versión distinta
(paso 3) o `libomp` ausente (README_MAC paso 1).

## 5. Revisar qué se va a correr (dry-run)

```bash
cd ~/Fase_pruebas_bbob
python experimentos/grid_bbob.py --salida resultados_bbob \
    --workers $(sysctl -n hw.physicalcpu) --dry-run
```

La primera línea debe decir:

```
Parrilla BBOB: 25920 corridas (middleware, pso, de, cmaes, madde, nlshadersp, jade, ga, clpso, abc, clonalg, optainet) | presupuesto 10000*D | observador COCO: si | ya hechas: 0 | pendientes: 25920
```

`sysctl -n hw.physicalcpu` debe dar **24**.

## 6. Lanzar la parrilla

**Siempre desde la raíz del clon.** COCO escribe sus archivos relativo a
`./exdata/`, y el código calcula la ruta de salida desde el directorio
actual.

```bash
cd ~/Fase_pruebas_bbob
source ~/Fase_pruebas/.venv/bin/activate

WORKERS=$(sysctl -n hw.physicalcpu) && echo "Workers: $WORKERS" && \
nohup caffeinate -dims python experimentos/grid_bbob.py \
    --salida resultados_bbob \
    --workers "$WORKERS" \
    > grid_bbob.log 2>&1 &
```

- `caffeinate -dims` impide el reposo. `nohup … &` deja la parrilla en
  segundo plano, así que se puede cerrar la Terminal.
- La configuración del Mac para ejecuciones largas (sin reposo ni
  actualizaciones automáticas) ya está hecha (README_MAC paso 7).
- Las corridas van de la más cara a la más barata (D = 40 primero). Al
  principio el ritmo es lento y el ETA exagera; se corrige solo.

**Opcional, presupuesto 10⁵·D (opción A' de README_BBOB.md, ≤ ~1 día):**
lanzarla después, en otra carpeta y con otro log. No mezclar
presupuestos en la misma `--salida`: `analisis/bbob.py` lo rechaza.

```bash
nohup caffeinate -dims python experimentos/grid_bbob.py \
    --salida resultados_bbob_1e5 --presupuesto 100000 \
    --workers "$WORKERS" > grid_bbob_1e5.log 2>&1 &
```

## 7. Monitorear, detener y retomar

```bash
cd ~/Fase_pruebas_bbob
tail -f grid_bbob.log                       # progreso y ETA (Ctrl+C no detiene nada)
pgrep -fl experimentos/grid_bbob.py         # ¿sigue vivo?
grep -c "  ok  " grid_bbob.log              # corridas ok
grep    "  fallo  " grid_bbob.log           # fallidas (traceback en resultados_bbob/fallos/)
```

Tras 15–20 min, el ETA del log ya es la cifra realista. Si pasa de ~5 h,
revisar que Python no corra bajo Rosetta (`platform.machine()` debe decir
`arm64`, paso 3) y que no haya fallos.

**Detener solo la parrilla BBOB.** **No usar** `pkill -f
multiprocessing.spawn` de los otros README: mataría también los workers
de la parrilla de referencia si sigue corriendo.

```bash
PID=$(pgrep -f experimentos/grid_bbob.py | head -1)
pkill -P "$PID"; kill "$PID"
```

**Retomar** tras una detención, un corte de luz o un reinicio: relanzar el
comando del paso 6 con la misma `--salida resultados_bbob` y otro nombre de
log (`grid_bbob_2.log`). Se saltan las corridas completas. Las que
estaban a medias se repiten desde cero, y su carpeta de COCO se borra y se
vuelve a escribir, sin duplicados.

## 8. Al terminar: verificar, consolidar y analizar en el Mac

```bash
cd ~/Fase_pruebas_bbob
source ~/Fase_pruebas/.venv/bin/activate

# 1) Nada pendiente: debe decir "ya hechas: 25920 | pendientes: 0".
#    Si faltan, relanzar el paso 6 (corre solo esas).
python experimentos/grid_bbob.py --salida resultados_bbob --dry-run | head -1

# 2) Consolidar -> resultados_bbob/corridas.parquet (+ corridas_propuesto.parquet)
python experimentos/consolidar.py --salida resultados_bbob

# 3) Rankings, Friedman, CD y tasa de éxito por dimensión -> resultados_bbob/analisis_bbob/
python analisis/bbob.py --resultados resultados_bbob

# 4) Post-procesamiento estándar de COCO (ECDF, ERT, tablas) -> resultados_bbob/ppdata/
cd resultados_bbob
python -m cocopp -o ppdata coco/middleware coco/pso coco/de coco/cmaes coco/madde coco/nlshadersp \
    coco/jade coco/ga coco/clpso coco/abc coco/clonalg coco/optainet
cd ..
```

Abrir `resultados_bbob/ppdata/index.html` en el navegador. La leyenda de
cada algoritmo es el nombre de su carpeta (`coco/pso`, …). `cocopp` tarda
algunos minutos con 12 algoritmos.

Para comparar contra un algoritmo publicado en el archivo de datos de
COCO (necesita internet; `cocopp` lo descarga), p. ej.:
`python -m cocopp -o ppdata_vs_bipop coco/middleware coco/cmaes bbob/2009/BIPOP-CMA-ES`.

## 9. Llevar los resultados al Windows

Empaquetar sin los archivos ocultos de macOS (`._*`, `.DS_Store`). Si se
copian, `consolidar.py` intentaría leer los `._*.parquet` y fallaría:

```bash
cd ~/Fase_pruebas_bbob
COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' \
    -czf resultados_bbob.tar.gz resultados_bbob grid_bbob.log
```

En el Windows, descomprimir en la raíz del proyecto
(`tar -xzf resultados_bbob.tar.gz`). Con `corridas.parquet` ya
consolidado alcanza para `analisis/bbob.py`. `coco/` hace falta solo para
volver a correr `cocopp`. `resultados_bbob*/` está en `.gitignore`, salvo
los consolidados.

## 10. Qué documentar en la tesis

- La parrilla BBOB (25 920 corridas, COCO 2.8.4, presupuesto 10⁴·D) se
  corrió completa en el Mac Studio (M2 Ultra, 24 núcleos, 64 GB, arm64),
  con el commit anotado en el paso 2.
- Los resultados no son idénticos bit a bit a los de una corrida en x86
  (README_MAC §10). Dentro del Mac son deterministas.
- Si BBOB y la parrilla de referencia corrieron a la vez en algún momento,
  `tiempo_seg` de esas corridas no es comparable. El error y la traza de
  convergencia no se ven afectados.
