# Ejecución del tier MaxFES = 5×10⁷ en el Mac Studio (Apple M2 Ultra)

Guía para correr en el Mac Studio **el tier `5×10⁷` completo con los 6
algoritmos** (`middleware`, `pso`, `de`, `cmaes`, `madde`, `nlshadersp`):
6 algoritmos × 12 funciones × 2 dimensiones × 51 semillas = **7 344
corridas**. El computador Windows (Xeon) corre los otros tres tiers
(`5×10⁴`, `5×10⁵`, `5×10⁶`). Contexto general: sección "Ejecución de la
parrilla en dos computadores" del [README.md](README.md).

No requiere cambios de código: todo es configuración del entorno y
operación.

## Equipo

| | |
|---|---|
| Modelo | Mac Studio (Mac14,14, MQH63CI/A) |
| Chip | Apple M2 Ultra: 24 núcleos (16 de rendimiento + 8 de eficiencia), sin Hyper-Threading |
| Memoria | 64 GB |
| Sistema | macOS 26.4.1 (25E253), Darwin 25.4.0, arm64 |
| Disco | 926 GB (457 GB libres al 2026-10-16) |
| Nombre | `Mac-Studio-Server.local` (es el valor que queda en la columna `host`) |

## Duración esperada

> **ETA: ~4,5 días (~105 h), rango razonable 3,5–6 días (85–140 h)** con
> 24 workers. Es una extrapolación, no una medición: confirmarla con el ETA
> de `grid_B.log` tras 4–6 h de ejecución (paso 6).

Cómo se estimó:

- **Base:** el tier `5×10⁷` está estimado en **~311 h en el Xeon E5-2673 v3
  con 24 workers** (tabla de "Ejecución de la parrilla en dos computadores"
  del [README.md](README.md)). Esa cifra ya incluye la penalización de
  correr 24 hilos sobre 12 núcleos con Hyper-Threading (`_FACTOR_CPU` en
  `experimentos/grid.py`: cada corrida tarda ×2,25–2,54 lo de un núcleo
  libre).
- **Rendimiento relativo:** un núcleo de rendimiento del M2 es ~2–2,5× un
  núcleo Haswell a 3,1 GHz en código Python/Numba escalar, y ~3,5–4,5× un
  hilo del Xeon con la máquina cargada (frecuencia sostenida menor y núcleo
  compartido por HT). Un núcleo de eficiencia rinde ~1,3–1,5 hilos del Xeon
  cargado. En total el Mac rinde **~2,3–3,5× el Xeon (central ~3×)**:
  311 h / 3 ≈ **105 h**.
- **Cola final despreciable:** la corrida más larga (DE F12 D=20 o CMA-ES
  D=20) dura ~2 h en el Xeon y ~0,5–1,5 h en el Mac, según caiga en un
  núcleo de rendimiento o de eficiencia.
- **Memoria y disco sobran:** 24 workers con torch/XGBoost ocupan del orden
  de 10–15 GB de los 64 GB.

Qué puede mover la cifra:

1. La estimación base ya es **±40 % por algoritmo**, y la del middleware no
   se recalibró tras D-10 (menos activaciones de Fase 2): probablemente
   sobreestima.
2. torch y XGBoost en ARM a un hilo están algo menos optimizados que en x86
   con AVX2; afecta sobre todo a FastSHAP dentro del middleware.

| Computador | Trabajo | Duración estimada |
|---|---|---|
| Windows (Xeon, 24 hilos) | tiers `5×10⁴`, `5×10⁵`, `5×10⁶` (22 032 corridas) | ~45 h (~2 días) |
| Mac Studio (M2 Ultra, 24 núcleos) | tier `5×10⁷` (7 344 corridas) | ~105 h (~4,5 días) |
| **Parrilla completa** | 29 376 corridas | **~4,5 días** (vs. ~15 días con el Xeon solo) |

Rebalancear (pasarle al Xeon `pso` y `nlshadersp` del `5×10⁷`, ~55 h)
dejaría ~100 h en el Xeon y ~88 h en el Mac: medio día de ganancia, menor
que la incertidumbre de la estimación. No se hace: se mantiene cada tier
completo en un solo computador.

---

## 1. Requisitos del sistema

En la Terminal del Mac:

```bash
# Herramientas de línea de comandos de Xcode (incluye git).
# corrida.py llama a `git rev-parse` para la columna `version`; sin git queda "sin-git".
xcode-select --install

# Homebrew (si no está): https://brew.sh
# OpenMP: XGBoost lo necesita en macOS. Sin él, `import xgboost` falla y
# todas las corridas del middleware terminan en estado "fallo".
brew install libomp
```

**Python 3.14 nativo arm64.** Instalar desde python.org (instalador
"macOS 64-bit universal2") o con `brew install python@3.14`. Verificar que
**no** corre bajo Rosetta (emulación x86, mucho más lenta):

```bash
python3.14 -c "import platform, sys; print(platform.machine(), sys.version)"
# Debe decir: arm64 3.14.x
```

## 2. Código: mismo commit que el Windows

Clonar desde git (no copiar la carpeta desde Windows: arrastraría
`__pycache__` y el caché de Numba de otra arquitectura):

```bash
cd ~
git clone https://github.com/alexanderefe/test_transfer_in_bioalgorithms.git Fase_pruebas
cd ~/Fase_pruebas
git log -1 --format=%h
```

El hash debe ser **el mismo** que muestra `git log -1 --format=%h` en el
Windows. Si no coincide, alinear ambos (`git pull` / `git checkout <hash>`)
**antes** de lanzar. Durante la ejecución no hacer `git pull` en ninguno de
los dos: la columna `version` de cada corrida registra el commit.

## 3. Entorno virtual con las mismas versiones que el Windows

`requirements.txt` solo fija mínimos; para que ambos computadores corran
exactamente lo mismo se fijan las versiones instaladas en el Windows:

```bash
cd ~/Fase_pruebas
python3.14 -m venv .venv
source .venv/bin/activate

cat > /tmp/requirements-mac.txt <<'EOF'
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
EOF
pip install -r /tmp/requirements-mac.txt
```

Notas:
- `torch` en macOS se instala sin el sufijo `+cpu` del Windows: es la
  misma versión.
- `fastshap` **no** se instala con pip: va como código fuente en
  `fastshap_lib/` (ya viene en el repo).
- Las versiones son las del Windows al 2026-10-16. Si en el Windows cambian,
  actualizar esta lista (`python -m pip freeze` en el Windows).

Comprobar la instalación:

```bash
python -c "import numpy, numba, torch, xgboost, scipy, cma, opfunu; \
print(numpy.__version__, numba.__version__, torch.__version__, xgboost.__version__, scipy.__version__, cma.__version__)"
```

## 4. Verificación antes de lanzar

Con el entorno activado (`source .venv/bin/activate`), desde `~/Fase_pruebas`:

```bash
# Evaluador CEC 2022 fiel al código oficial (12 funciones × 2 dims).
# Termina con "Fidelidad verificada: 12 funciones × 2 dimensiones."
python tests/test_cec2022_numba_paridad.py

# Smoke del harness: los 6 algoritmos corren y son deterministas.
# Termina con "[OK] smoke del harness experimental (...)"
python tests/test_experimentos.py
```

Si alguno falla, **no lanzar la parrilla**. El error más probable es
`libomp` ausente (ver paso 1).

## 5. Revisar qué se va a correr (dry-run)

```bash
python experimentos/grid.py --salida resultados_B --maxfes 50000000 \
    --workers $(sysctl -n hw.physicalcpu) --dry-run
```

La primera línea debe decir
`Parrilla: 7344 corridas (middleware, pso, de, cmaes, madde, nlshadersp) | ya hechas: 0 | pendientes: 7344`
(sin `--algoritmo`, `grid.py` usa los 6). Las horas estimadas que muestra
(~311 h) están calibradas para el Xeon: **en el Mac no aplican**; la
estimación para este equipo es la de "Duración esperada" (~105 h).

`sysctl -n hw.physicalcpu` debe dar **24** en el M2 Ultra (16 núcleos de
rendimiento + 8 de eficiencia). No tiene Hyper-Threading, así que se usa un
worker por núcleo, incluidos los de eficiencia: aportan ~15 % del
rendimiento total y el pool reparte las corridas dinámicamente.

## 6. Comando final: lanzar la ejecución

```bash
cd ~/Fase_pruebas
source .venv/bin/activate

WORKERS=$(sysctl -n hw.physicalcpu) && echo "Workers: $WORKERS" && \
nohup caffeinate -dims python experimentos/grid.py \
    --salida resultados_B \
    --maxfes 50000000 \
    --workers "$WORKERS" \
    > grid_B.log 2>&1 &
```

- `caffeinate -dims`: impide que el Mac entre en reposo mientras dure la
  ejecución (se libera solo al terminar).
- `nohup … &`: corre en segundo plano; se puede cerrar la Terminal.
- `grid_B.log`: progreso y errores. Cada línea es una corrida terminada
  con su estado, su tiempo y un ETA de lo que falta.

Ya se puede cerrar la Terminal. **Tras 4–6 h, revisar el ETA del log:** es
la cifra realista para el Mac. Si queda fuera del rango 85–140 h, la
extrapolación de "Duración esperada" estaba errada (o algo anda mal: revisar
que no corra bajo Rosetta, paso 1, y que no haya fallos, paso 8).

## 7. Configuración del Mac para varios días de ejecución

- **Ajustes del Sistema → Energía:** activar "Evitar el reposo automático
  cuando la pantalla esté apagada" (refuerza `caffeinate`).
- **Ajustes del Sistema → General → Actualización de software →
  Actualizaciones automáticas:** desactivar la instalación automática
  (reinicia el equipo).
- **Modo de bajo consumo:** desactivado.
- No cerrar sesión ni apagar. Dejar el equipo ventilado.

## 8. Monitorear, detener y retomar

```bash
tail -f ~/Fase_pruebas/grid_B.log     # ver progreso (Ctrl+C sale sin detener nada)
pgrep -fl grid.py                     # ¿sigue vivo?
grep -c "  ok  " ~/Fase_pruebas/grid_B.log     # corridas ok hasta ahora
grep    "  fallo  " ~/Fase_pruebas/grid_B.log  # corridas fallidas (traceback en resultados_B/fallos/)
```

**Detener** (detiene el proceso principal y sus workers):

```bash
pkill -f experimentos/grid.py; pkill -f multiprocessing.spawn
```

**Retomar** tras una detención, un corte de luz o un reinicio: relanzar
**el mismo comando del paso 6** con la misma `--salida resultados_B`.
`grid.py` salta las corridas ya completas; las que estaban a medias se
repiten desde cero. Para no perder el log anterior, cambiar el nombre del
archivo de log al relanzar (p. ej. `grid_B_2.log`).

## 9. Al terminar: llevar los resultados al Windows

La última línea del log es `Hecho: N ok, M fallo en X h.` (cuenta solo
las corridas de ese lanzamiento). Para confirmar que no falta nada, repetir
el dry-run del paso 5: debe decir **`ya hechas: 7344 | pendientes: 0`**. Si quedan
pendientes (fallos o cortes), relanzar el paso 6: corre solo esas.

Empaquetar **sin** los archivos ocultos de macOS (`._*`, `.DS_Store`).
Si se copian, `consolidar.py` intentaría leer los `._*.parquet` como
Parquet y fallaría:

```bash
cd ~/Fase_pruebas
COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' -czf resultados_B.tar.gz resultados_B
```

Llevar `resultados_B.tar.gz` al Windows (pendrive, red o nube) y
descomprimirlo en la raíz del proyecto (`tar -xzf resultados_B.tar.gz`,
disponible en Windows 10/11). Luego unir con la carpeta del Windows
(`resultados_A`), copiando el contenido de `corridas/`,
`corridas_propuesto/`, `logs_middleware/` y `fallos/` de `resultados_B`
dentro de las mismas subcarpetas de `resultados_A`, y correr:

```bash
python experimentos/consolidar.py --salida resultados_A
python analisis/main.py --resultados resultados_A
```

`cargar_corridas()` exige las 29 376 corridas de la parrilla completa: si
falta alguna de cualquiera de los dos computadores, lo detecta.

## 10. Qué documentar en la tesis

- **El tier `5×10⁷` se corrió en un Mac Studio con Apple M2 Ultra (24
  núcleos, 64 GB, macOS 26.4.1, arm64) y el resto en Xeon E5-2673 v3
  (x86-64).** Cada algoritmo×tier quedó completo en un solo
  computador, así que las comparaciones dentro de un tier son con el mismo
  hardware.
- Los resultados **no son idénticos bit a bit entre arquitecturas** (Numba,
  torch y XGBoost generan código distinto en ARM y x86). El determinismo se
  mantiene dentro de cada computador; no reproducir en el Windows una
  corrida del Mac esperando el mismo número exacto.
- La columna `tiempo_seg` mezcla hardware. `analisis/` no la usa, pero si
  se reportan tiempos de cómputo hay que separar por la columna `host`.
