"""
Wrapper de las funciones del benchmark CEC 2022 (Kumar, Price, Mohamed,
Hadi & Suganthan, 2022) sobre la librería opfunu.

Se centraliza el acceso a las 12 funciones aquí por dos razones:
1. Si en el futuro cambia la librería subyacente (o se reimplementa
   manualmente alguna función por validación cruzada), solo se modifica
   este archivo, no cada algoritmo.
2. Permite fijar la interfaz exacta que consumen PSO y DE: una función
   callable f(x) -> float, y una matriz de límites (n_dim, 2).
"""

import math
import threading

import numpy as np
from opfunu.cec_based import cec2022

# Umbral de la convención CEC 2017/2022: un error de la función objetivo por
# debajo de este valor se considera equivalente a haber alcanzado el óptimo
# (se usa, por ejemplo, para asignar rangos empatados entre algoritmos).
UMBRAL_ERROR_CERO = 1e-8

# Fracciones de MaxFES en las que se fotografía el mejor valor visto hasta el
# momento (traza de convergencia). Convención CEC + fracciones tempranas
# densas para capturar el comportamiento inicial.
FRACCIONES_CHECKPOINT = (
    0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)

# Mapeo de las 12 funciones oficiales del benchmark CEC 2022.
# F1: unimodal | F2-F4: básicas (multimodales simples)
# F5-F7: híbridas | F8-F12: composición (alta complejidad multimodal)
_FUNCIONES_CEC2022 = {
    1: cec2022.F12022,
    2: cec2022.F22022,
    3: cec2022.F32022,
    4: cec2022.F42022,
    5: cec2022.F52022,
    6: cec2022.F62022,
    7: cec2022.F72022,
    8: cec2022.F82022,
    9: cec2022.F92022,
    10: cec2022.F102022,
    11: cec2022.F112022,
    12: cec2022.F122022,
}

DIMENSIONES_VALIDAS = (10, 20)  # según especificación oficial CEC 2022

# Valor de la función objetivo en el óptimo global, f(x*), para cada una de
# las 12 funciones del benchmark CEC 2022 (Kumar et al., 2022, Tabla I).
# Es el mismo para D=10 y D=20. El error de convergencia que se reporta en
# el análisis experimental es (f_obtenido - OPTIMO_GLOBAL_CEC2022[n]), que
# es 0 en el óptimo y positivo en cualquier otro punto.
OPTIMO_GLOBAL_CEC2022 = {
    1: 300.0,    2: 400.0,    3: 600.0,    4: 800.0,
    5: 900.0,    6: 1800.0,   7: 2000.0,   8: 2200.0,
    9: 2300.0,  10: 2400.0,  11: 2600.0,  12: 2700.0,
}


class ProblemaCEC2022:
    """
    Representa una instancia de problema: una función del benchmark CEC 2022
    en una dimensionalidad específica.

    CONTADOR DE EVALUACIONES (FES): esta clase lleva un contador thread-safe
    del número de evaluaciones de la función objetivo realizadas sobre esta
    instancia. Como el orquestador entrega LA MISMA instancia de problema a
    los dos algoritmos que corren en paralelo (PSO y DE comparten el mismo
    objeto `problema` como `funcion_objetivo`), `self.fes` cuenta el total
    AGREGADO de ambos hilos. Ese total es el que se compara contra MaxFES
    como criterio de parada del sistema completo (ver setup experimental de
    la tesis: MaxFES es el único criterio de parada, con corridas
    independientes por cada valor de MaxFES).

    El middleware en sí no evalúa la función objetivo (Fase 2 entrena sobre
    historial ya evaluado; Canal B inyecta instancias con su fitness ya
    conocido, sin re-evaluar), por lo que el contador refleja únicamente el
    consumo de presupuesto de los dos algoritmos.
    """

    def __init__(self, numero_funcion: int, ndim: int, backend: str = "numba"):
        if numero_funcion not in _FUNCIONES_CEC2022:
            raise ValueError(
                f"Función F{numero_funcion} no existe en CEC 2022. "
                f"Debe ser un valor entre 1 y 12."
            )
        if ndim not in DIMENSIONES_VALIDAS:
            raise ValueError(
                f"Dimensión {ndim} no estándar para CEC 2022. "
                f"Debe ser una de {DIMENSIONES_VALIDAS}."
            )
        if backend not in ("numba", "opfunu"):
            raise ValueError(
                f"backend '{backend}' no reconocido. Debe ser 'numba' "
                f"(evaluador compilado, default) u 'opfunu' (referencia "
                f"interpretada, usado por el test de paridad)."
            )

        self._funcion = _FUNCIONES_CEC2022[numero_funcion](ndim=ndim)
        self.numero_funcion = numero_funcion
        self.ndim = ndim
        self.nombre = self._funcion.name
        self.backend = backend

        # Backend compilado (D-5, docs/configuracion_experimental.md): evita
        # el overhead de opfunu/numpy por llamada, que domina el tiempo de
        # la parrilla experimental. Reutiliza los mismos arrays de
        # shift/rotación/shuffle que opfunu ya cargó arriba — no es una
        # reimplementación independiente de la especificación CEC, es el
        # mismo cómputo compilado (paridad validada en
        # tests/test_cec2022_numba_paridad.py). Import perezoso: quien use
        # backend="opfunu" no necesita tener `numba` instalado.
        self._eval_compilado = None
        if backend == "numba":
            from problems.cec2022_numba import construir_evaluador
            self._eval_compilado = construir_evaluador(numero_funcion, self._funcion)
            self._eval_compilado(np.zeros(ndim))  # warm-up: compila aquí, no en la corrida

        # opfunu expone bounds como array (ndim, 2): [min, max] por dimensión
        self.limites = np.array(self._funcion.bounds, dtype=float)

        # Valor de la función objetivo en el óptimo global, f(x*). Se usa
        # para medir cuánto se acercan los valores obtenidos en el
        # experimento: el error reportado es (f_obtenido - optimo_global),
        # que vale 0 en el óptimo y es positivo en cualquier otro punto.
        # Se toma de opfunu (f_global) y se contrasta con la tabla oficial
        # OPTIMO_GLOBAL_CEC2022; si opfunu no lo expone, se usa la tabla.
        optimo_opfunu = getattr(self._funcion, "f_global", None)
        optimo_tabla = OPTIMO_GLOBAL_CEC2022[numero_funcion]
        if optimo_opfunu is not None and not np.isclose(
            float(optimo_opfunu), optimo_tabla, atol=1e-6
        ):
            raise ValueError(
                f"Inconsistencia en el óptimo de F{numero_funcion}: opfunu "
                f"reporta {optimo_opfunu}, la tabla oficial CEC 2022 dice "
                f"{optimo_tabla}. Revisar la versión de opfunu."
            )
        self.optimo_global = float(optimo_opfunu) if optimo_opfunu is not None \
            else optimo_tabla

        # Vector solución del óptimo global, x* (si opfunu lo expone).
        self.x_optimo = getattr(self._funcion, "x_global", None)

        # Contador de evaluaciones de la función objetivo (FES). Protegido
        # por un lock porque los dos algoritmos que comparten esta instancia
        # corren en hilos distintos y ambos incrementan el contador.
        self._fes = 0
        self._lock_fes = threading.Lock()

        # Traza de convergencia: mejor valor objetivo visto en cualquier
        # evaluación (best-so-far, convención CEC) y su "foto" en cada
        # fracción de MaxFES. Se activa con configurar_checkpoints().
        self._mejor_visto = float("inf")
        self._umbrales_checkpoint: list[tuple[float, int]] = []  # (fracción, FES)
        self._idx_checkpoint = 0
        self._checkpoints: dict[float, float] = {}

    def configurar_checkpoints(self, max_fes: int) -> None:
        """
        Prepara la captura de la traza de convergencia para una corrida de
        `max_fes` evaluaciones. Debe llamarse ANTES de iniciar la corrida
        (y después de reiniciar_fes() si se reutiliza la instancia).
        """
        self._umbrales_checkpoint = [
            (f, max(1, math.ceil(f * max_fes))) for f in FRACCIONES_CHECKPOINT
        ]
        self._idx_checkpoint = 0
        self._checkpoints = {}

    def evaluar(self, x: np.ndarray) -> float:
        """Interfaz uniforme: recibe un vector y retorna un escalar."""
        if self.backend == "numba":
            valor = float(self._eval_compilado(x))
        else:
            valor = float(self._funcion.evaluate(x))
        with self._lock_fes:
            self._fes += 1
            if valor < self._mejor_visto:
                self._mejor_visto = valor
            # Registrar todos los checkpoints cuyo umbral de FES ya se alcanzó
            # (un solo eval puede cruzar varios si MaxFES es muy bajo).
            while (self._idx_checkpoint < len(self._umbrales_checkpoint)
                   and self._fes >= self._umbrales_checkpoint[self._idx_checkpoint][1]):
                frac = self._umbrales_checkpoint[self._idx_checkpoint][0]
                self._checkpoints[frac] = self._mejor_visto
                self._idx_checkpoint += 1
        return valor

    @property
    def fes(self) -> int:
        """Número de evaluaciones de la función objetivo realizadas hasta
        ahora sobre esta instancia (agregado de todos los hilos que la
        comparten)."""
        with self._lock_fes:
            return self._fes

    def reiniciar_fes(self) -> None:
        """Pone el contador de FES a cero y descarta la traza de convergencia.
        Necesario para reutilizar la misma instancia de problema en corridas
        independientes (el setup experimental exige que cada valor de MaxFES
        sea una corrida nueva desde cero, no una continuación).
        Llamar a configurar_checkpoints() de nuevo tras reiniciar."""
        with self._lock_fes:
            self._fes = 0
            self._mejor_visto = float("inf")
            self._idx_checkpoint = 0
            self._checkpoints = {}

    def error_checkpoints(self) -> list[float]:
        """
        Traza de convergencia como error respecto al óptimo (`f − f(x*)`),
        una entrada por cada fracción de FRACCIONES_CHECKPOINT. Para una
        fracción que la corrida no alcanzó se usa el mejor valor visto hasta
        ese punto (con ~MaxFES consumidas la fracción 1.0 siempre se alcanza).
        """
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
        """
        Error de convergencia: cuánto se aleja `valor_objetivo` del óptimo
        global conocido. Es la métrica de comparación del setup experimental
        (media de las 51 corridas del mejor valor "tras sustraer el óptimo
        conocido").

            error = valor_objetivo - optimo_global   (>= 0, 0 en el óptimo)

        Acepta un escalar o un array de numpy (se vectoriza).

        aplicar_umbral: si True, todo error por debajo de UMBRAL_ERROR_CERO
            (1e-8, convención CEC) se colapsa a 0.0. Por defecto False para
            no perder información en el material suplementario; el umbral se
            aplica más adelante, al construir los rankings.
        """
        err = np.asarray(valor_objetivo, dtype=float) - self.optimo_global
        # Se recorta el ruido numérico que podría dar un error levemente
        # negativo (el óptimo es un mínimo global, no debería superarse).
        err = np.maximum(err, 0.0)
        if aplicar_umbral:
            err = np.where(err < UMBRAL_ERROR_CERO, 0.0, err)
        return float(err) if np.ndim(err) == 0 else err

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluar(x)

    def __repr__(self):
        return (f"ProblemaCEC2022(F{self.numero_funcion}, D={self.ndim}, "
                f"'{self.nombre}', f(x*)={self.optimo_global:g})")