"""
Recorre una parrilla de corridas del harness experimental (setup §5).

Ejecuta, para el producto cartesiano de funciones × dimensiones × MaxFES ×
semillas, una corrida por combinación, en paralelo por proceso. Cada corrida
escribe su propio Parquet en `resultados/corridas/<run_id>.parquet` (y
`resultados/corridas_propuesto/<run_id>.parquet`), de modo que:
  - no hay contención entre workers,
  - una interrupción no corrompe nada,
  - re-lanzar el comando SALTA las corridas ya hechas (reanudación).

Ejemplos:
    # estimación, sin ejecutar nada
    python experimentos/grid.py --funciones 1-12 --dims 10,20 \
        --maxfes 5000,50000,500000,5000000 --semillas 0-50 --dry-run

    # un lote real pequeño
    python experimentos/grid.py --funciones 1 --dims 10 --maxfes 5000 \
        --semillas 0-4 --max-epochs-fastshap 10

Después de un lote: `python experimentos/consolidar.py`.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experimentos import esquema
from experimentos.corrida import correr_corrida
from experimentos.semillas import SEMILLAS
from middleware.extraccion import EPOCHS_EXPLAINER

MAXFES_VALIDOS = (5_000, 50_000, 500_000, 5_000_000)
DIMS_VALIDAS = (10, 20)

# Algoritmos implementados (decisión D-4: sin competidores externos, solo
# el propuesto y sus dos algoritmos base corriendo SOLOS como baseline).
ALGORITMOS_VALIDOS = ("middleware", "pso", "de")

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
_US_POR_EVAL = {
    1: (1.64, 1.69),   2: (1.43, 1.65),  3: (1.53, 1.88),  4: (1.65, 1.95),
    5: (2.06, 2.03),   6: (2.20, 2.04),  7: (2.27, 2.89),  8: (2.49, 3.03),
    9: (3.30, 4.76),  10: (3.12, 4.33), 11: (3.97, 5.76), 12: (4.07, 6.14),
}


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
    """
    us = _US_POR_EVAL.get(funcion, (150.0, 150.0))[0 if dim == 10 else 1]
    t_obj = us * 1e-6 * max_fes
    if algoritmo == "middleware":
        n_act = min(9, max_fes / 25_000)
        return -2.4235 + 12.5617 * n_act + 30.31e-6 * max_fes
    overhead_us = _OVERHEAD_US_POR_EVAL.get(algoritmo, 0.0)
    return t_obj + overhead_us * 1e-6 * max_fes


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
    ap.add_argument("--algoritmo", default="middleware,pso,de",
                    help=f"lista separada por comas, de {ALGORITMOS_VALIDOS}")
    ap.add_argument("--funciones", default="1-12")
    ap.add_argument("--dims", default="10,20")
    ap.add_argument("--maxfes", default="5000,50000,500000,5000000")
    ap.add_argument("--semillas", default="0-50", help="índices 0..50 en semillas.json")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-epochs-fastshap", type=int, default=EPOCHS_EXPLAINER)
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    algoritmos = [a.strip() for a in args.algoritmo.split(",") if a.strip()]
    funciones = _parse_lista(args.funciones)
    dims = _parse_lista(args.dims)
    maxfes = _parse_lista(args.maxfes)
    idx_semillas = _parse_lista(args.semillas)

    for a in algoritmos:
        if a not in ALGORITMOS_VALIDOS:
            ap.error(f"algoritmo '{a}' no reconocido (D-4: {ALGORITMOS_VALIDOS})")
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
    pendientes, hechas = [], 0
    for algoritmo, funcion, dim, mf, idx in combos:
        rid = esquema.run_id(algoritmo, funcion, dim, mf, idx)
        if (dir_corridas / f"{rid}.parquet").exists():
            hechas += 1
        else:
            pendientes.append((algoritmo, funcion, dim, mf, idx,
                               args.max_epochs_fastshap, str(salida)))

    est_por_algoritmo: dict[str, float] = {}
    for algoritmo, f, d, mf, *_ in pendientes:
        est_por_algoritmo[algoritmo] = (
            est_por_algoritmo.get(algoritmo, 0.0) + _estimacion_seg(algoritmo, f, d, mf))
    est_total = sum(est_por_algoritmo.values())

    print(f"Parrilla: {len(combos)} corridas ({', '.join(algoritmos)}) | "
          f"ya hechas: {hechas} | pendientes: {len(pendientes)}")
    for a, seg in est_por_algoritmo.items():
        print(f"  {a:12s} secuencial: {seg / 3600:6.1f} h  ->  "
              f"{args.workers} workers: ~{seg / 3600 / args.workers:.1f} h")
    print(f"Estimación TOTAL (secuencial): {est_total / 3600:.1f} h  ->  "
          f"con {args.workers} workers: ~{est_total / 3600 / args.workers:.1f} h")

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
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futuros = {ex.submit(_tarea, t): t for t in pendientes}
        for n, fut in enumerate(as_completed(futuros), 1):
            rid, estado, seg = fut.result()
            ok += estado == "ok"
            fallo += estado != "ok"
            transc = time.time() - t0
            eta = transc / n * (len(pendientes) - n)
            print(f"[{n}/{len(pendientes)}] {rid}  {estado}  {seg:.0f}s  "
                  f"| ETA {eta / 3600:.1f} h", flush=True)

    print(f"\nHecho: {ok} ok, {fallo} fallo en {(time.time() - t0) / 3600:.2f} h. "
          f"Consolidá con:  python experimentos/consolidar.py")


if __name__ == "__main__":
    main()
