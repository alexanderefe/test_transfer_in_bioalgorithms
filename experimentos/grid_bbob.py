"""
Parrilla BBOB, opción A de README_BBOB.md: protocolo nativo de COCO.

    12 algoritmos × 24 funciones × D ∈ {2,3,5,10,20,40} × instancias 1..15,
    una corrida por instancia, presupuesto = multiplicador · D evaluaciones
    (default 10⁴ · D)  →  25 920 corridas.

Mismo mecanismo que `experimentos/grid.py`: un Parquet por corrida en
`<salida>/corridas/`, reanudable (salta lo ya hecho), paralelo por proceso,
`--dry-run` con estimación. Además, con el observador activado (default),
los archivos de COCO quedan en `<salida>/coco/<algoritmo>/...` para `cocopp`.

Ejemplos:
    python experimentos/grid_bbob.py --dry-run
    python experimentos/grid_bbob.py --salida resultados_bbob
    python experimentos/grid_bbob.py --salida resultados_bbob --presupuesto 100000
    python experimentos/grid_bbob.py --algoritmo pso,cmaes --dims 2,3 --funciones 1-5

Después:
    python experimentos/consolidar.py --salida resultados_bbob
    python analisis/bbob.py --resultados resultados_bbob
    python -m cocopp resultados_bbob/coco/cmaes resultados_bbob/coco/middleware ...

Correr SIEMPRE desde la raíz del proyecto: COCO escribe relativo a
`./exdata/` y la carpeta de salida debe ser corta (MAX_PATH de Windows).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from experimentos import esquema
from experimentos.corrida_bbob import correr_corrida_bbob
from experimentos.grid import (
    ALGORITMOS_VALIDOS, _estimacion_seg, _parse_lista,
)
from problems.bbob_wrapper import (
    DIMENSIONES_BBOB, FUNCIONES_BBOB, INSTANCIAS_BBOB, VERSION_EVALUADOR_BBOB,
)
from problems.cec2022_wrapper import FRACCIONES_CHECKPOINT
from middleware.extraccion import EPOCHS_EXPLAINER

# Los 12 algoritmos: parrilla principal (6) + los 6 de referencia que se
# corren en el Mac (README_MAC_REFERENCIA.md: 2 por familia).
ALGORITMOS_BBOB = ("middleware", "pso", "de", "cmaes", "madde", "nlshadersp",
                   "jade", "ga", "clpso", "abc", "clonalg", "optainet")

PRESUPUESTO_DEFAULT = 10_000          # evaluaciones por dimensión (budget · D)

# Sobrecosto del observador de COCO, medido en el Windows (2026-11-22):
# ~15 µs por evaluación (vs. 0.9–2.5 µs de la función sola).
_US_OBSERVADOR = 15.0

_ETIQUETA_EVALUADOR = f"|eval:{VERSION_EVALUADOR_BBOB}"


def _corrida_vigente(ruta: Path, carpeta_obs: Path | None) -> bool:
    """Hecha = Parquet legible, misma versión de COCO, traza completa y, si
    se pide observador, su carpeta presente."""
    try:
        tabla = pq.ParquetFile(ruta).read(columns=["version", "error_checkpoints", "estado"])
        version = tabla.column("version")[0].as_py()
        checkpoints = tabla.column("error_checkpoints")[0].as_py()
        estado = tabla.column("estado")[0].as_py()
    except Exception:  # noqa: BLE001
        return False
    if _ETIQUETA_EVALUADOR not in str(version):
        return False
    if estado == "ok" and (checkpoints is None
                           or len(checkpoints) != len(FRACCIONES_CHECKPOINT)):
        return False
    if estado == "ok" and carpeta_obs is not None and not carpeta_obs.is_dir():
        return False
    return True


def _estimacion_bbob(algoritmo: str, funcion: int, dim: int, max_fes: int,
                     observador: bool) -> float:
    """Reutiliza la estimación calibrada de grid.py (µs/eval medidos en el
    Xeon con 24 workers, en D=10 y D=20) llevando cada D de BBOB a la más
    cercana: D ≤ 10 → 10, D > 10 → 20. La función BBOB se toma como la F1
    de CEC (barata). Es GRUESA: no captura el costo O(D²) de CMA-ES en
    D=40 ni el de FastSHAP con más variables; confirmar con un piloto."""
    d_ref = 10 if dim <= 10 else 20
    seg = _estimacion_seg(algoritmo, 1, d_ref, max_fes)
    if observador:
        seg += _US_OBSERVADOR * 1e-6 * max_fes
    return seg


def _tarea(args):
    algoritmo, funcion, dim, instancia, max_fes, max_epochs, salida, observador = args
    fila, fila_prop = correr_corrida_bbob(
        algoritmo, funcion, dim, instancia, max_fes,
        max_epochs_fastshap=max_epochs, dir_salida=salida, observador=observador,
    )
    rid = fila["run_id"]
    dir_c = Path(salida) / "corridas"
    dir_c.mkdir(parents=True, exist_ok=True)
    esquema.df_tipado([fila], esquema.COLUMNAS_CORRIDA).to_parquet(
        dir_c / f"{rid}.parquet", index=False)
    if fila_prop is not None:
        dir_p = Path(salida) / "corridas_propuesto"
        dir_p.mkdir(parents=True, exist_ok=True)
        esquema.df_tipado([fila_prop], esquema.COLUMNAS_PROPUESTO).to_parquet(
            dir_p / f"{rid}.parquet", index=False)
    return rid, fila["estado"], fila["tiempo_seg"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algoritmo", default=",".join(ALGORITMOS_BBOB),
                    help=f"lista separada por comas (default: los 12). Válidos: {ALGORITMOS_VALIDOS}")
    ap.add_argument("--funciones", default="1-24")
    ap.add_argument("--dims", default=",".join(map(str, DIMENSIONES_BBOB)))
    ap.add_argument("--instancias", default="1-15")
    ap.add_argument("--presupuesto", type=int, default=PRESUPUESTO_DEFAULT,
                    help="multiplicador del presupuesto: MaxFES = presupuesto · D (default 10⁴)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--max-epochs-fastshap", type=int, default=EPOCHS_EXPLAINER)
    ap.add_argument("--tareas-por-worker", type=int, default=20)
    ap.add_argument("--salida", default="resultados_bbob")
    ap.add_argument("--sin-observador", action="store_true",
                    help="no escribir los archivos de COCO (sin cocopp; ~15 µs/eval menos)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    algoritmos = [a.strip() for a in args.algoritmo.split(",") if a.strip()]
    funciones = _parse_lista(args.funciones)
    dims = _parse_lista(args.dims)
    instancias = _parse_lista(args.instancias)
    observador = not args.sin_observador

    for a in algoritmos:
        if a not in ALGORITMOS_VALIDOS:
            ap.error(f"algoritmo '{a}' no reconocido ({ALGORITMOS_VALIDOS})")
    for d in dims:
        if d not in DIMENSIONES_BBOB:
            ap.error(f"dim {d} no está en bbob {DIMENSIONES_BBOB}")
    for f in funciones:
        if f not in FUNCIONES_BBOB:
            ap.error(f"función {f} fuera de 1..24")
    for i in instancias:
        if i not in INSTANCIAS_BBOB:
            ap.error(f"instancia {i} fuera de 1..15")
    if args.presupuesto < 1:
        ap.error("--presupuesto debe ser ≥ 1")

    salida = Path(args.salida)
    combos = list(itertools.product(algoritmos, funciones, dims, instancias))
    pendientes, hechas = [], 0
    for algoritmo, funcion, dim, inst in combos:
        mf = args.presupuesto * dim
        rid = esquema.run_id_bbob(algoritmo, funcion, dim, inst, mf)
        carpeta_obs = (salida / "coco" / algoritmo
                       / f"f{funcion:02d}_d{dim:02d}_i{inst:02d}_fes{mf}") if observador else None
        ruta = salida / "corridas" / f"{rid}.parquet"
        if ruta.exists() and _corrida_vigente(ruta, carpeta_obs):
            hechas += 1
        else:
            pendientes.append((algoritmo, funcion, dim, inst, mf,
                               args.max_epochs_fastshap, str(salida), observador))

    est: dict[str, float] = {}
    for algoritmo, f, d, _i, mf, *_ in pendientes:
        est[algoritmo] = est.get(algoritmo, 0.0) + _estimacion_bbob(algoritmo, f, d, mf, observador)
    total = sum(est.values())

    print(f"Parrilla BBOB: {len(combos)} corridas ({', '.join(algoritmos)}) | "
          f"presupuesto {args.presupuesto}*D | observador COCO: {'si' if observador else 'no'} | "
          f"ya hechas: {hechas} | pendientes: {len(pendientes)}")
    for a, seg in est.items():
        print(f"  {a:12s} suma: {seg / 3600:7.2f} h  ->  "
              f"{args.workers} workers: ~{seg / 3600 / args.workers:.2f} h")
    print(f"Estimacion TOTAL (Xeon, gruesa): {total / 3600:.1f} h  ->  "
          f"con {args.workers} workers: ~{total / 3600 / args.workers:.1f} h")

    if args.dry_run:
        print("\n--dry-run: no se ejecuta nada.")
        for t in pendientes[:20]:
            print("  ", esquema.run_id_bbob(t[0], t[1], t[2], t[3], t[4]))
        if len(pendientes) > 20:
            print(f"   ... (+{len(pendientes) - 20})")
        return
    if not pendientes:
        print("Nada que hacer.")
        return

    # Las corridas más largas primero (D grande, algoritmos caros): la cola
    # final queda con las cortas.
    pendientes.sort(key=lambda t: -_estimacion_bbob(t[0], t[1], t[2], t[4], observador))

    t0 = time.time()
    ok = fallo = 0
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
