"""
Wrapper de la suite `bbob` (noiseless) de COCO — Black-Box Optimization
Benchmarking (Hansen et al., 2009; Hansen, Auger, Ros, Mersmann, Tušar y
Brockhoff, 2021, "COCO: a platform for comparing continuous optimizers in a
black-box setting") — con la MISMA interfaz que `ProblemaCEC2022`, para que
los 12 algoritmos del harness corran sin cambios (ver README_BBOB.md).

Lo único que consumen los algoritmos (middleware, PSO/DE solos y los
competidores de `experimentos/competidores/`) es:

    evaluar(x) / __call__(x)   ·  fes  ·  limites  ·  optimo_global  ·  error()

más `configurar_checkpoints()` / `error_checkpoints()`, que usa el harness
para la traza de convergencia (mismas `FRACCIONES_CHECKPOINT` que CEC 2022).

La evaluación la hace el código C oficial de COCO (`cocoex`, paquete
`coco-experiment`). Opcionalmente se adjunta un observador de COCO
(`carpeta_observador=`), que escribe los archivos estándar (.info/.dat/.tdat)
para post-procesar con `cocopp` y comparar contra el archivo de datos de
COCO. El observador NO afecta la búsqueda: solo registra.

Protocolo (opción A de README_BBOB.md): 24 funciones × D ∈ {2,3,5,10,20,40}
× instancias 1..15, una corrida por instancia (en COCO las instancias hacen
el papel de las repeticiones independientes) con presupuesto
`multiplicador · D` evaluaciones.
"""

from __future__ import annotations

import math
import os
import shutil
import threading
from pathlib import Path

import numpy as np

from problems.cec2022_wrapper import FRACCIONES_CHECKPOINT, UMBRAL_ERROR_CERO

import cocoex

# Sin los "COCO INFO: Results will be output to folder ..." de cada corrida
# (25 920 líneas en el log de la parrilla); los errores se siguen mostrando.
cocoex.log_level("warning")

FUNCIONES_BBOB = tuple(range(1, 25))
DIMENSIONES_BBOB = (2, 3, 5, 10, 20, 40)
INSTANCIAS_BBOB = tuple(range(1, 16))      # instancias clásicas 2009 (1..15)

# Etiqueta del evaluador, va a la columna `version` (como VERSION_EVALUADOR
# de CEC 2022): grid_bbob.py la usa para no reanudar corridas de otra versión.
VERSION_EVALUADOR_BBOB = f"bbob-coco{cocoex.__version__}"

# Nombre de cada función (Hansen et al., 2009, "Real-parameter black-box
# optimization benchmarking 2009: noiseless functions definitions").
NOMBRES_BBOB = {
    1: "Sphere", 2: "Ellipsoidal separable", 3: "Rastrigin separable",
    4: "Bueche-Rastrigin", 5: "Linear slope",
    6: "Attractive sector", 7: "Step ellipsoidal", 8: "Rosenbrock original",
    9: "Rosenbrock rotated",
    10: "Ellipsoidal", 11: "Discus", 12: "Bent cigar", 13: "Sharp ridge",
    14: "Different powers",
    15: "Rastrigin", 16: "Weierstrass", 17: "Schaffer F7",
    18: "Schaffer F7 mal condicionada", 19: "Griewank-Rosenbrock F8F2",
    20: "Schwefel x*sin(x)", 21: "Gallagher 101 picos", 22: "Gallagher 21 picos",
    23: "Katsuura", 24: "Lunacek bi-Rastrigin",
}

# Grupos estándar de COCO, para el análisis por grupo de funciones.
GRUPO_BBOB = {
    **{f: "separables" for f in range(1, 6)},
    **{f: "condicionamiento bajo/moderado" for f in range(6, 10)},
    **{f: "condicionamiento alto, unimodales" for f in range(10, 15)},
    **{f: "multimodales, estructura global adecuada" for f in range(15, 20)},
    **{f: "multimodales, estructura global débil" for f in range(20, 25)},
}


def _suite() -> "cocoex.Suite":
    # Instancias fijadas explícitamente (no "año por defecto" de la versión
    # instalada de COCO): mismo conjunto 1..15 en cualquier versión.
    return cocoex.Suite("bbob", "instances: 1-15", "")


class ProblemaBBOB:
    """
    Una instancia (función, dimensión, instancia) de la suite `bbob`.

    Contador de FES y traza de convergencia: idénticos a `ProblemaCEC2022`
    (mismo lock, mismo best-so-far, mismas fracciones de MaxFES).

    carpeta_observador: si se da, se adjunta un `cocoex.Observer("bbob")`
        que escribe sus archivos EXACTAMENTE en esa carpeta (se borra antes
        si existe: COCO crearía `<carpeta>-0001` y una corrida reanudada
        quedaría duplicada en `cocopp`). Llamar a `cerrar()` al terminar
        para que COCO vacíe sus archivos.
    nombre_algoritmo: `algorithm_name` que registra el observador (lo que
        `cocopp` muestra en las leyendas).
    """

    def __init__(self, numero_funcion: int, ndim: int, instancia: int = 1, *,
                 carpeta_observador: str | Path | None = None,
                 nombre_algoritmo: str = "algoritmo"):
        if numero_funcion not in FUNCIONES_BBOB:
            raise ValueError(f"Función f{numero_funcion} no existe en bbob (1..24).")
        if ndim not in DIMENSIONES_BBOB:
            raise ValueError(f"Dimensión {ndim} no está en bbob {DIMENSIONES_BBOB}.")
        if instancia not in INSTANCIAS_BBOB:
            raise ValueError(f"Instancia {instancia} fuera de 1..15.")

        self._suite = _suite()
        self._p = self._suite.get_problem_by_function_dimension_instance(
            numero_funcion, ndim, instancia)
        self.numero_funcion = numero_funcion
        self.ndim = ndim
        self.instancia = instancia
        self.nombre = f"{self._p.id} ({NOMBRES_BBOB[numero_funcion]})"
        self.limites = np.column_stack(
            [np.asarray(self._p.lower_bounds, dtype=float),
             np.asarray(self._p.upper_bounds, dtype=float)])

        # f(x*) de la instancia. cocoex lo expone como atributo privado
        # `_best_value` (no lo publica para que el optimizador no lo use).
        # Aquí lo usa SOLO el harness para calcular el error; ningún
        # algoritmo lo lee (mismo uso que `optimo_global` en CEC 2022).
        self.optimo_global = float(self._p._best_value)
        self.x_optimo = None

        self._observador = None
        if carpeta_observador is not None:
            self._observador = self._adjuntar_observador(
                Path(carpeta_observador), nombre_algoritmo)

        self._fes = 0
        self._lock_fes = threading.Lock()
        self._mejor_visto = float("inf")
        self._umbrales_checkpoint: list[tuple[float, int]] = []
        self._idx_checkpoint = 0
        self._checkpoints: dict[float, float] = {}

    def _adjuntar_observador(self, carpeta: Path, nombre_algoritmo: str):
        carpeta = carpeta.resolve()
        for vieja in [carpeta, *carpeta.parent.glob(carpeta.name + "-0*")]:
            if vieja.is_dir():
                shutil.rmtree(vieja)
        carpeta.parent.mkdir(parents=True, exist_ok=True)
        # COCO solo acepta `result_folder` relativo a "exdata/" (una ruta
        # absoluta falla en Windows: "unable to create exdata\C:"). Se le da
        # la ruta relativa desde exdata/ hasta la carpeta pedida.
        base = Path(os.getcwd()) / "exdata"
        rel = os.path.relpath(carpeta, base)
        # COCO termina el PROCESO (exit desde C, sin excepción de Python) si
        # no puede abrir un archivo; en Windows pasa si la ruta SIN
        # normalizar (cwd/exdata/../../...) supera MAX_PATH = 260. En un
        # worker de grid_bbob.py eso dejaría la tarea colgada: se verifica
        # antes, con margen para "data_f24/bbobexp_f24_DIM40.tdat".
        largo = len(str(base / rel)) + len("/data_f24/bbobexp_f24_DIM40.tdat") + 2
        if os.name == "nt" and largo >= 260:
            raise ValueError(
                f"Ruta del observador de COCO demasiado larga para Windows "
                f"({largo} ≥ 260 caracteres): usar una carpeta de salida más "
                f"corta dentro del proyecto. Carpeta: {carpeta}")
        # algorithm_name no admite espacios en las opciones de COCO.
        nombre = nombre_algoritmo.replace(" ", "_")
        obs = cocoex.Observer(
            "bbob", f"result_folder: {rel} algorithm_name: {nombre}")
        self._p.observe_with(obs)
        return obs

    # ── interfaz común con ProblemaCEC2022 ────────────────────────────────

    def configurar_checkpoints(self, max_fes: int) -> None:
        self._umbrales_checkpoint = [
            (f, max(1, math.ceil(f * max_fes))) for f in FRACCIONES_CHECKPOINT
        ]
        self._idx_checkpoint = 0
        self._checkpoints = {}

    def evaluar(self, x: np.ndarray) -> float:
        valor = float(self._p(np.asarray(x, dtype=float)))
        with self._lock_fes:
            self._fes += 1
            if valor < self._mejor_visto:
                self._mejor_visto = valor
            while (self._idx_checkpoint < len(self._umbrales_checkpoint)
                   and self._fes >= self._umbrales_checkpoint[self._idx_checkpoint][1]):
                frac = self._umbrales_checkpoint[self._idx_checkpoint][0]
                self._checkpoints[frac] = self._mejor_visto
                self._idx_checkpoint += 1
        return valor

    @property
    def fes(self) -> int:
        with self._lock_fes:
            return self._fes

    def reiniciar_fes(self) -> None:
        with self._lock_fes:
            self._fes = 0
            self._mejor_visto = float("inf")
            self._idx_checkpoint = 0
            self._checkpoints = {}

    def error_checkpoints(self) -> list[float]:
        with self._lock_fes:
            checkpoints = dict(self._checkpoints)
            mejor_actual = self._mejor_visto
        salida = []
        ultimo = mejor_actual
        for f in FRACCIONES_CHECKPOINT:
            if f in checkpoints:
                ultimo = checkpoints[f]
            salida.append(self.error(ultimo))
        return salida

    def error(self, valor_objetivo, aplicar_umbral: bool = False):
        """error = f − f(x*), recortado a ≥ 0 (mismo criterio que CEC 2022)."""
        err = np.asarray(valor_objetivo, dtype=float) - self.optimo_global
        err = np.maximum(err, 0.0)
        if aplicar_umbral:
            err = np.where(err < UMBRAL_ERROR_CERO, 0.0, err)
        return float(err) if np.ndim(err) == 0 else err

    def cerrar(self) -> None:
        """Libera el problema de COCO (vacía los archivos del observador)."""
        if self._p is not None:
            self._p.free()
            self._p = None
        self._suite = None

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluar(x)

    def __repr__(self):
        return (f"ProblemaBBOB(f{self.numero_funcion}, D={self.ndim}, "
                f"i{self.instancia}, f(x*)={self.optimo_global:g})")
