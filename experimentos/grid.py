"""
Recorre una parrilla de corridas del harness experimental (setup §5).

Ejecuta, para el producto cartesiano de funciones × dimensiones × MaxFES ×
semillas, una corrida por combinación, en paralelo por proceso. Cada corrida
escribe su propio Parquet en `<salida>/corridas/<run_id>.parquet` (y
`<salida>/corridas_propuesto/<run_id>.parquet`), donde `<salida>` es la
carpeta de `--salida` (default `resultados`), de modo que:
  - no hay contención entre workers,
  - una interrupción no corrompe nada,
  - re-lanzar el comando SALTA las corridas ya hechas (reanudación).

Ejemplos:
    # estimación, sin ejecutar nada
    python experimentos/grid.py --funciones 1-12 --dims 10,20 \
        --maxfes 50000,500000,5000000,50000000 --semillas 0-50 --dry-run

    # un lote real pequeño
    python experimentos/grid.py --funciones 1 --dims 10 --maxfes 50000 \
        --semillas 0-4 --max-epochs-fastshap 10

    # la parrilla completa en una carpeta propia (no toca `resultados/`)
    python experimentos/grid.py --salida resultados_2026-10

Después de un lote, con la misma carpeta:
    python experimentos/consolidar.py --salida resultados_2026-10
    python analisis/main.py --resultados resultados_2026-10
"""

from __future__ import annotations

import os

# Un hilo por proceso (D-7): el paralelismo va por workers, no dentro de cada
# corrida. Con los pools por defecto de torch/BLAS/Numba, N workers abren N×n
# hilos y se pelean los núcleos (medido: corridas 3-5× más lentas con 12
# workers). Debe fijarse ANTES de importar numpy/torch; los hijos de
# multiprocessing.Pool heredan el entorno.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from experimentos import esquema
from experimentos.corrida import correr_corrida
from problems.cec2022_numba import VERSION_EVALUADOR
from problems.cec2022_wrapper import FRACCIONES_CHECKPOINT
from experimentos.semillas import SEMILLAS
from middleware.extraccion import EPOCHS_EXPLAINER

MAXFES_VALIDOS = (50_000, 500_000, 5_000_000, 50_000_000)
DIMS_VALIDAS = (10, 20)

# Algoritmos implementados: el propuesto, sus dos algoritmos base corriendo
# SOLOS como baseline (D-4) y los tres competidores externos (D-8).
ALGORITMOS_VALIDOS = ("middleware", "pso", "de", "cmaes", "madde", "nlshadersp")

# Overhead constante de Python por evaluación (µs), medido en esta máquina,
# INDEPENDIENTE de la función (no es costo del objetivo, es el bucle del
# algoritmo). PSO está vectorizado con numpy (bajo); DE tiene un for-loop
# por individuo con rng.choice() y slicing (alto). No incluye FastSHAP.
# Recalibrado post-D5 con regresión lineal sobre 37+37 corridas piloto
# reales (R²=1.000 ambos): valores previos (11.0 / 77.0) estaban obsoletos
# y sobreestimaban el costo real — quedan comentados como referencia.
# anterior: {"pso": 11.0, "de": 77.0}
_OVERHEAD_US_POR_EVAL = {"pso": 5.34, "de": 48.33}

# µs/eval del evaluador CEC2022, medido en esta máquina, como (D=10, D=20).
# Backend "numba" (D-5, docs/configuracion_experimental.md): compilado con
# Numba, reemplazó al backend "opfunu" (interpretado) como default de
# ProblemaCEC2022. Speedup medido 6x (F1, función barata, dominada por
# overhead de llamada) a 220x (F8, la más cara bajo opfunu). Valores
# opfunu previos quedan comentados como referencia histórica.
# opfunu (pre-D5): 1:(9.7,9.8) 2:(15.4,15.2) 3:(34.1,34.6) 4:(56.1,57.6)
#   5:(21.7,21.4) 6:(25.7,25.6) 7:(267.4,454.6) 8:(358.8,629.0)
#   9:(85.0,87.1) 10:(91.3,92.6) 11:(159.3,159.6) 12:(172.8,172.4)
# D-9: el evaluador se re-portó desde el C oficial de CEC 2022. Los valores
# de abajo son los de D-5 × (µs/eval nuevo / µs/eval viejo), razón medida
# en esta CPU con ambos evaluadores alternados en el mismo proceso
# (0.85–1.44×, mediana ~1.1×). Valores D-5 previos, como referencia:
#   1:(1.64,1.69) 2:(1.43,1.65) 3:(1.53,1.88) 4:(1.65,1.95) 5:(2.06,2.03)
#   6:(2.20,2.04) 7:(2.27,2.89) 8:(2.49,3.03) 9:(3.30,4.76) 10:(3.12,4.33)
#   11:(3.97,5.76) 12:(4.07,6.14)
_US_POR_EVAL = {
    1: (2.03, 1.76),   2: (1.69, 1.68),  3: (2.20, 2.54),  4: (1.83, 1.66),
    5: (2.35, 2.01),   6: (2.44, 2.28),  7: (2.81, 3.73),  8: (2.79, 3.73),
    9: (3.00, 4.90),  10: (2.87, 4.46), 11: (3.81, 6.28), 12: (3.99, 7.06),
}

# Recalibración por cambio de CPU (D-7, 2026-10-08): las constantes de arriba
# se ajustaron en la CPU anterior con 5 workers. En el Xeon E5-2673 v3
# (12 núcleos / 24 hilos), con 1 hilo por proceso y 24 workers, cada corrida
# tarda este factor × lo que tardaba antes (medido re-corriendo run_ids ya
# existentes, mismo trabajo por determinismo). El factor depende de la carga:
# 12 workers → ~1.65 (de) / ~1.36 (middleware); 24 → valores de abajo. Aun
# así el rendimiento total crece hasta 24 workers (Hyper-Threading aporta
# ~+30% sobre 12). pso no se midió a 24: se asume igual que de (a 12 workers
# ambos daban ~1.62-1.65).
_FACTOR_CPU = {"pso": 2.54, "de": 2.54, "middleware": 2.25}

# Competidores externos (D-8): µs/eval TOTALES (objetivo + algoritmo) como
# (D=10, D=20), medidos directamente en la CPU actual con 24 workers y 1
# hilo por proceso (piloto de 216 corridas a MaxFES=5×10⁵, las 12
# funciones), así que NO llevan _FACTOR_CPU. MadDE y NL-SHADE-RSP
# normalizan su calendario contra MaxFES y su costo por evaluación es ~
# constante (confirmado a 5×10⁶). IPOP-CMA-ES no: el costo fijo por
# iteración de `cma` se reparte entre más evaluaciones a medida que los
# reinicios duplican λ, así que su µs/eval cae ×_DECAIMIENTO_POR_DECADA por
# cada década de MaxFES (medido 5×10⁵ → 5×10⁶; extrapolado a 5×10⁴ y 5×10⁷).
_COSTO_US_POR_EVAL = {
    "cmaes": (176.7, 239.6),
    "madde": (44.7, 30.0),
    "nlshadersp": (63.2, 49.9),
}
_DECAIMIENTO_POR_DECADA = {"cmaes": (0.81, 0.72)}


_ETIQUETA_EVALUADOR = f"|eval:{VERSION_EVALUADOR}"


def _corrida_vigente(ruta: Path) -> bool:
    """Una corrida ya hecha solo cuenta si se evaluó con el evaluador
    vigente (D-9) y trae la traza de convergencia completa: las del port de
    opfunu ("eval:numba") no son el CEC 2022 oficial, y las de la grilla
    vieja de 14 checkpoints no alcanzan para las curvas de convergencia
    (analisis/convergencia.py). Ambas se re-ejecutan, no se reanudan."""
    try:
        # ParquetFile directo: ~1 ms/archivo (pandas ~7 ms), importa con
        # ~29 000 archivos en cada reanudación.
        tabla = pq.ParquetFile(ruta).read(columns=["version", "error_checkpoints"])
        version = tabla.column("version")[0].as_py()
        checkpoints = tabla.column("error_checkpoints")[0].as_py()
    except Exception:  # noqa: BLE001 — archivo ilegible: se re-ejecuta
        return False
    # checkpoints = None: corrida en estado "fallo" (sin traza); se trata
    # como antes, por la versión del evaluador.
    return (_ETIQUETA_EVALUADOR in str(version)
            and (checkpoints is None
                 or len(checkpoints) == len(FRACCIONES_CHECKPOINT)))


def _parse_lista(texto: str, tipo=int) -> list:
    """Acepta '1-12', '10,20', '1-3,7,9-11'."""
    salida: list = []
    for parte in texto.split(","):
        parte = parte.strip()
        if "-" in parte and not parte.startswith("-"):
            a, b = parte.split("-")
            salida.extend(range(int(a), int(b) + 1))
        else:
            salida.append(tipo(parte))
    return sorted(set(salida))


def _estimacion_seg(algoritmo: str, funcion: int, dim: int, max_fes: int) -> float:
    """
    Tiempo aprox. de una corrida. Recalibrado post-D5 (2026-09-04) con
    regresión lineal sobre 111 corridas piloto reales
    (docs/configuracion_experimental.md, D-5 / piloto):

      "middleware": t ≈ intercepto_fijo + seg_por_activacion · nº activaciones
                       + us_eval_combinado · MaxFES  (ajuste: R²=0.893, n=69,
                       12 funciones × 2 dims cubiertas en maxfes ≥ 500 000)
        seg_por_activacion ≈ 12.56 s (vs. 2.5 s asumido antes — la Fase 2
          real es ~5x más cara de lo que se creía).
        us_eval_combinado ≈ 30.31 µs/eval (overhead interno de PSO+DE
          corriendo bajo el middleware, consistente con _OVERHEAD_US_POR_EVAL).
        intercepto_fijo ≈ -2.42 s (artefacto del ajuste por mínimos
          cuadrados con muestra finita; magnitud pequeña, no se interpreta
          como costo negativo real).

        nº activaciones ≈ min(9, MaxFES / 25 000): validado con el piloto
        ampliado en maxfes=500 000 y 5 000 000 — el conteo real SE ESTABILIZA
        en un valor propio de cada función (rango 3-9, la mayoría 5-9) en
        vez de seguir creciendo con MaxFES; el tope de 9 de esta fórmula
        aproxima razonablemente ese comportamiento para maxfes ≥ 225 000.

      "pso" / "de" (sin middleware, presupuesto completo para sí solos):
        t ≈ t_objetivo + overhead_us_por_eval · MaxFES — ajuste R²=1.000,
        n=37 cada uno; ver _OVERHEAD_US_POR_EVAL (recalibrado, ya no es la
        estimación pre-D5).

    Todo se multiplica por _FACTOR_CPU (D-7): tiempo de una corrida en la CPU
    actual con todos los hilos lógicos ocupados, no en un núcleo libre.

      competidores externos (D-8): t ≈ costo_us_por_eval(MaxFES) · MaxFES,
        medido ya bajo carga completa en la CPU actual (sin _FACTOR_CPU);
        ver _COSTO_US_POR_EVAL.
    """
    i_dim = 0 if dim == 10 else 1
    if algoritmo in _COSTO_US_POR_EVAL:
        decaimiento = _DECAIMIENTO_POR_DECADA.get(algoritmo, (1.0, 1.0))[i_dim]
        us = (_COSTO_US_POR_EVAL[algoritmo][i_dim]
              * decaimiento ** math.log10(max_fes / 500_000))
        return us * 1e-6 * max_fes
    us = _US_POR_EVAL.get(funcion, (150.0, 150.0))[i_dim]
    t_obj = us * 1e-6 * max_fes
    if algoritmo == "middleware":
        n_act = min(9, max_fes / 25_000)
        t = -2.4235 + 12.5617 * n_act + 30.31e-6 * max_fes
    else:
        overhead_us = _OVERHEAD_US_POR_EVAL.get(algoritmo, 0.0)
        t = t_obj + overhead_us * 1e-6 * max_fes
    return t * _FACTOR_CPU.get(algoritmo, 1.0)


def _tarea(args):
    algoritmo, funcion, dim, max_fes, idx, max_epochs, salida = args
    fila, fila_prop = correr_corrida(
        algoritmo, funcion, dim, max_fes, idx,
        max_epochs_fastshap=max_epochs, dir_salida=salida,
    )
    rid = fila["run_id"]
    dir_c = Path(salida) / "corridas"
    dir_p = Path(salida) / "corridas_propuesto"
    dir_c.mkdir(parents=True, exist_ok=True)
    esquema.df_tipado([fila], esquema.COLUMNAS_CORRIDA).to_parquet(
        dir_c / f"{rid}.parquet", index=False)
    if fila_prop is not None:
        dir_p.mkdir(parents=True, exist_ok=True)
        esquema.df_tipado([fila_prop], esquema.COLUMNAS_PROPUESTO).to_parquet(
            dir_p / f"{rid}.parquet", index=False)
    return rid, fila["estado"], fila["tiempo_seg"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algoritmo", default=",".join(ALGORITMOS_VALIDOS),
                    help=f"lista separada por comas, de {ALGORITMOS_VALIDOS}")
    ap.add_argument("--funciones", default="1-12")
    ap.add_argument("--dims", default="10,20")
    ap.add_argument("--maxfes", default="50000,500000,5000000,50000000")
    ap.add_argument("--semillas", default="0-50", help="índices 0..50 en semillas.json")
    # Default = todos los hilos lógicos (D-7): medido, el rendimiento sigue
    # subiendo hasta 24 workers en el Xeon de 24 hilos; _FACTOR_CPU está
    # calibrado a esa carga.
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--max-epochs-fastshap", type=int, default=EPOCHS_EXPLAINER)
    # Cada worker se reemplaza por uno nuevo tras N corridas: torch/XGBoost y
    # el allocator no devuelven toda la memoria entre corridas y, a lo largo
    # de ~900 corridas por worker, la deriva se acumula (ver LARGO_HISTORIAL
    # en bioalgorithms/base.py para la causa principal del OOM de 2026-10-17).
    ap.add_argument("--tareas-por-worker", type=int, default=20,
                    help="corridas antes de reciclar cada proceso worker (0 = nunca)")
    ap.add_argument("--salida", default="resultados",
                    help="carpeta de resultados de esta ejecución (se crea si no existe); "
                         "usar la misma en consolidar.py --salida y analisis/main.py --resultados")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    algoritmos = [a.strip() for a in args.algoritmo.split(",") if a.strip()]
    funciones = _parse_lista(args.funciones)
    dims = _parse_lista(args.dims)
    maxfes = _parse_lista(args.maxfes)
    idx_semillas = _parse_lista(args.semillas)

    for a in algoritmos:
        if a not in ALGORITMOS_VALIDOS:
            ap.error(f"algoritmo '{a}' no reconocido ({ALGORITMOS_VALIDOS})")
    for d in dims:
        if d not in DIMS_VALIDAS:
            ap.error(f"dim {d} no válida (CEC 2022: {DIMS_VALIDAS})")
    for m in maxfes:
        if m not in MAXFES_VALIDOS:
            ap.error(f"MaxFES {m} no está en el protocolo {MAXFES_VALIDOS}")
    for i in idx_semillas:
        if not 0 <= i < len(SEMILLAS):
            ap.error(f"índice de semilla {i} fuera de rango 0..{len(SEMILLAS) - 1}")
    if not 1 <= min(funciones) and max(funciones) <= 12:
        ap.error("funciones deben estar en 1..12")

    salida = Path(args.salida)
    dir_corridas = salida / "corridas"

    combos = list(itertools.product(algoritmos, funciones, dims, maxfes, idx_semillas))
    pendientes, hechas, obsoletas = [], 0, 0
    for algoritmo, funcion, dim, mf, idx in combos:
        rid = esquema.run_id(algoritmo, funcion, dim, mf, idx)
        ruta = dir_corridas / f"{rid}.parquet"
        if ruta.exists() and _corrida_vigente(ruta):
            hechas += 1
        else:
            obsoletas += ruta.exists()
            pendientes.append((algoritmo, funcion, dim, mf, idx,
                               args.max_epochs_fastshap, str(salida)))
    if obsoletas:
        print(f"Aviso: {obsoletas} corridas existentes se hicieron con otro evaluador "
              f"(no '{_ETIQUETA_EVALUADOR}', D-9) o con otra grilla de checkpoints "
              f"(no {len(FRACCIONES_CHECKPOINT)}) y se re-ejecutan (se sobrescriben).")

    est_por_algoritmo: dict[str, float] = {}
    for algoritmo, f, d, mf, *_ in pendientes:
        est_por_algoritmo[algoritmo] = (
            est_por_algoritmo.get(algoritmo, 0.0) + _estimacion_seg(algoritmo, f, d, mf))
    est_total = sum(est_por_algoritmo.values())

    print(f"Parrilla: {len(combos)} corridas ({', '.join(algoritmos)}) | "
          f"ya hechas: {hechas} | pendientes: {len(pendientes)}")
    # "suma": suma de los tiempos por corrida bajo carga completa (D-7), no un
    # tiempo de un solo núcleo libre; dividir por workers da el wall-clock.
    for a, seg in est_por_algoritmo.items():
        print(f"  {a:12s} suma: {seg / 3600:6.1f} h  ->  "
              f"{args.workers} workers: ~{seg / 3600 / args.workers:.1f} h")
    print(f"Estimación TOTAL (suma): {est_total / 3600:.1f} h  ->  "
          f"con {args.workers} workers: ~{est_total / 3600 / args.workers:.1f} h")
    if args.workers != (os.cpu_count() or 1):
        print(f"  (aviso: _FACTOR_CPU está calibrado con {os.cpu_count()} workers; "
              f"con menos, cada corrida es más rápida y la estimación sobreestima)")

    if args.dry_run:
        print("\n--dry-run: no se ejecuta nada.")
        for t in pendientes[:20]:
            print("  ", esquema.run_id(t[0], t[1], t[2], t[3], t[4]))
        if len(pendientes) > 20:
            print(f"   ... (+{len(pendientes) - 20})")
        return

    if not pendientes:
        print("Nada que hacer.")
        return

    t0 = time.time()
    ok = fallo = 0
    # multiprocessing.Pool y no ProcessPoolExecutor: en Python 3.14.4 el
    # max_tasks_per_child de este último se cuelga al reciclar el primer
    # worker (verificado con un ejemplo mínimo); maxtasksperchild de Pool sí
    # funciona.
    with Pool(args.workers, maxtasksperchild=args.tareas_por_worker or None) as pool:
        for n, (rid, estado, seg) in enumerate(
                pool.imap_unordered(_tarea, pendientes, chunksize=1), 1):
            ok += estado == "ok"
            fallo += estado != "ok"
            transc = time.time() - t0
            eta = transc / n * (len(pendientes) - n)
            print(f"[{n}/{len(pendientes)}] {rid}  {estado}  {seg:.0f}s  "
                  f"| ETA {eta / 3600:.1f} h", flush=True)

    print(f"\nHecho: {ok} ok, {fallo} fallo en {(time.time() - t0) / 3600:.2f} h. "
          f"Consolidá con:  python experimentos/consolidar.py --salida {args.salida}")


if __name__ == "__main__":
    main()
