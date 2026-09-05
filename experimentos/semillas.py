"""
Carga la lista fija de 51 semillas del protocolo experimental.

`SEMILLAS[i]` es la semilla de la corrida con índice `i` (0..50). La MISMA
lista se usa para todo algoritmo y toda configuración (dimensionalidad ×
MaxFES): cada corrida siembra su RNG desde cero con ese valor, lo que da
independencia (no hay continuación entre MaxFES) y a la vez permite el
emparejamiento por semilla del test de Wilcoxon (setup §7, §10).
"""

import json
from pathlib import Path

_RUTA = Path(__file__).with_name("semillas.json")

if not _RUTA.exists():
    raise FileNotFoundError(
        f"No existe {_RUTA}. Generá la lista fija con:\n"
        f"    python experimentos/generar_semillas.py"
    )

SEMILLAS: list[int] = json.loads(_RUTA.read_text())

if len(SEMILLAS) != 51:
    raise ValueError(
        f"{_RUTA} tiene {len(SEMILLAS)} semillas; el protocolo exige 51."
    )
