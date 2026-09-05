"""
Genera la lista fija de 51 semillas aleatorias del protocolo experimental
(setup §5 / §10).

Es IDEMPOTENTE: si `semillas.json` ya existe, no lo toca. La lista se genera
una sola vez y se versiona, de modo que sea reproducible con independencia
de la versión de numpy instalada.

Uso:
    python experimentos/generar_semillas.py [--forzar]
"""

import argparse
import json
from pathlib import Path

import numpy as np

RUTA = Path(__file__).with_name("semillas.json")

N_SEMILLAS = 51            # regla de CEC 2017 (setup §5)
SEMILLA_MAESTRA = 20260903  # fija la generación; no cambiar
LIMITE = 2**31 - 1


def generar() -> list[int]:
    rng = np.random.default_rng(SEMILLA_MAESTRA)
    return sorted(int(v) for v in rng.integers(1, LIMITE, size=N_SEMILLAS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forzar", action="store_true",
                    help="regenera aunque el archivo ya exista (¡rompe reproducibilidad!)")
    args = ap.parse_args()

    if RUTA.exists() and not args.forzar:
        datos = json.loads(RUTA.read_text())
        print(f"{RUTA.name} ya existe con {len(datos)} semillas. Nada que hacer.")
        return

    semillas = generar()
    RUTA.write_text(json.dumps(semillas, indent=2) + "\n")
    print(f"Escritas {len(semillas)} semillas en {RUTA}")


if __name__ == "__main__":
    main()
