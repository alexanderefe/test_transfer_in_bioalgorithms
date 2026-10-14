"""
Protocolo común de los competidores externos (D-8,
docs/configuracion_experimental.md).

Los competidores corren SOLOS, como caja negra `(problema, max_fes, semilla)`:
no se acoplan al middleware ni implementan la interfaz de `bioalgorithms/`.
Lo único que comparten con el resto del harness es `ProblemaCEC2022`, que
lleva el contador de FES y la traza de convergencia (checkpoints), así que
la fila §11.2 se arma igual que para PSO/DE solos.

Corte exacto en MaxFES: toda evaluación pasa por `Presupuesto.evaluar`, que
lanza `PresupuestoAgotado` en cuanto `problema.fes` llega a `max_fes`. Las
implementaciones originales (MATLAB/C++) terminan la generación en curso y
se pasan hasta en una población; aquí se corta en la evaluación exacta, que
es lo que mide el checkpoint 1.0 de todos modos (best-so-far con ≤ MaxFES).
"""

from __future__ import annotations

import numpy as np


class PresupuestoAgotado(Exception):
    """Se alcanzó MaxFES: el competidor debe terminar ya."""


class Presupuesto:
    """Envoltorio del problema que corta exactamente en `max_fes`."""

    def __init__(self, problema, max_fes: int):
        self.problema = problema
        self.max_fes = int(max_fes)

    @property
    def fes(self) -> int:
        return self.problema.fes

    def evaluar(self, x: np.ndarray) -> float:
        if self.problema.fes >= self.max_fes:
            raise PresupuestoAgotado
        return self.problema.evaluar(x)

    def evaluar_lote(self, X: np.ndarray) -> np.ndarray:
        """Evalúa fila por fila; si el presupuesto se acaba a mitad del
        lote, lanza `PresupuestoAgotado` (las evaluaciones ya hechas
        quedaron registradas en el problema)."""
        return np.array([self.evaluar(x) for x in X], dtype=float)


def round_half_up(x: float) -> int:
    """`round` de MATLAB/C++ (mitades alejándose de cero) para x ≥ 0.
    El `round` de Python redondea mitades al par, lo que cambiaría
    tamaños de población en los casos .5."""
    return int(np.floor(x + 0.5))
