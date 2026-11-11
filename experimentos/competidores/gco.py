"""
GCO — Germinal Center Optimization (Villaseñor, Arana-Daniel, Alanis,
López-Franco y Hernández-Vargas, International Journal of Computational
Intelligence Systems 12(1):13-27, 2018; acceso abierto), familia de
sistemas inmunes artificiales: el paper lo presenta como «an AIS algorithm
based on GC» (sección 5), inspirado en la reacción del centro germinal
(expansión clonal, hipermutación somática y selección por afinidad de los
linfocitos B).

Reemplaza a opt-IA, cuyos papers son de acceso cerrado
(docs/competidores.md §8.2-8.3).

`system_based.GCO.OriginalGCO` de mealpy (commit 46b9850), revisado línea a
línea contra los Algoritmos 1 y 2 del paper, sin diferencias:
  - Inicialización: N células B aleatorias, contador de células = 1 y
    señal de vida 𝔏 = 70 en cada una.
  - Zona oscura, para cada célula B_i: si r ~ U[0, 100] < 𝔏_i se duplica
    (contador + 1); si no, y su contador es > 1, muere una copia
    (contador − 1). Luego Mutate(B_i): r₁, r₂, r₃ ~ 𝒞 (distribución
    proporcional a los contadores), M(j) = B_r1(j) + F·(B_r2(j) − B_r3(j))
    si U[0, 1] < CR, si no B_i(j); si f(M) < f(B_i), B_i ← M y 𝔏_i += 10.
  - Zona clara: 𝔏_i −= 10 y 𝔏_i += 10·fit_i, con
    fit_i = (f(B_i) − max f) / (min f − max f) ∈ [0, 1].
Ambigüedad del paper: no dice si r₁, r₂, r₃ deben ser distintos; mealpy los
toma distintos (sin reemplazo). Límites: recorte (el paper no lo especifica).

Parámetros del paper: N = 40, CR = 0,7, F = 1,25.
Validación: tests/test_gco_reproduccion.py. Reproduce la Tabla 2 del paper
(d = 2) pero NO la Tabla 4 (d = 30): la ambigüedad de r₁, r₂, r₃ cambia el
resultado en dimensión alta y los autores no publicaron su código
(docs/competidores.md §8.3).
"""

from __future__ import annotations

from mealpy.system_based.GCO import OriginalGCO

from experimentos.competidores.adaptador_mealpy import correr_mealpy, version_mealpy

VERSION = f"{version_mealpy()}+gco-villasenor2018"
N_CELULAS = 40
CR = 0.7
F = 1.25


def correr(problema, max_fes: int, semilla: int) -> None:
    correr_mealpy(OriginalGCO, problema, max_fes, semilla,
                  pop_size=N_CELULAS, cr=CR, wf=F)
