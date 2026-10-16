# Ejecución del tier MaxFES = 5×10⁷ en el Mac mini (Apple M4)

Guía para correr en el Mac mini **el tier `5×10⁷` completo con los 6
algoritmos** (`middleware`, `pso`, `de`, `cmaes`, `madde`, `nlshadersp`):
6 algoritmos × 12 funciones × 2 dimensiones × 51 semillas = **7 344
corridas**. El computador Windows (Xeon) corre los otros tres tiers
(`5×10⁴`, `5×10⁵`, `5×10⁶`). Contexto general: sección "Ejecución de la
parrilla en dos computadores" del [README.md](README.md).

No requiere cambios de código: todo es configuración del entorno y
operación.

> **Duración esperada:** el tier `5×10⁷` concentra ~87 % del cómputo de la
> parrilla (~311 h en el Xeon con 24 workers). El Windows termina su parte
> en ~2 días y el Mac sigue solo bastante más. En el Mac el tiempo real es
> desconocido hasta medirlo (ver paso 6): el `--dry-run` de `grid.py` está
> calibrado para el Xeon y en el Mac es solo orientativo.

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
están calibradas para el Xeon: en el Mac no son confiables.

`sysctl -n hw.physicalcpu` da los núcleos físicos (10 en el M4 base, 12 o
14 en el M4 Pro). El M4 no tiene Hyper-Threading, así que se usa un worker
por núcleo.

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

Ya se puede cerrar la Terminal. Después de unas horas, el ETA del log da
una idea realista de la duración en el Mac.

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

- **El tier `5×10⁷` se corrió en Apple M4 (arm64) y el resto en Xeon
  E5-2673 v3 (x86-64).** Cada algoritmo×tier quedó completo en un solo
  computador, así que las comparaciones dentro de un tier son con el mismo
  hardware.
- Los resultados **no son idénticos bit a bit entre arquitecturas** (Numba,
  torch y XGBoost generan código distinto en ARM y x86). El determinismo se
  mantiene dentro de cada computador; no reproducir en el Windows una
  corrida del Mac esperando el mismo número exacto.
- La columna `tiempo_seg` mezcla hardware. `analisis/` no la usa, pero si
  se reportan tiempos de cómputo hay que separar por la columna `host`.
