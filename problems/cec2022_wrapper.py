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

import threading

import numpy as np
from opfunu.cec_based import cec2022

# Umbral de la convención CEC 2017/2022: un error de la función objetivo por
# debajo de este valor se considera equivalente a haber alcanzado el óptimo
# (se usa, por ejemplo, para asignar rangos empatados entre algoritmos).
UMBRAL_ERROR_CERO = 1e-8

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

    def __init__(self, numero_funcion: int, ndim: int):
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

        self._funcion = _FUNCIONES_CEC2022[numero_funcion](ndim=ndim)
        self.numero_funcion = numero_funcion
        self.ndim = ndim
        self.nombre = self._funcion.name

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

    def evaluar(self, x: np.ndarray) -> float:
        """Interfaz uniforme: recibe un vector y retorna un escalar."""
        valor = float(self._funcion.evaluate(x))
        with self._lock_fes:
            self._fes += 1
        return valor

    @property
    def fes(self) -> int:
        """Número de evaluaciones de la función objetivo realizadas hasta
        ahora sobre esta instancia (agregado de todos los hilos que la
        comparten)."""
        with self._lock_fes:
            return self._fes

    def reiniciar_fes(self) -> None:
        """Pone el contador de FES a cero. Necesario para reutilizar la
        misma instancia de problema en corridas independientes (el setup
        experimental exige que cada valor de MaxFES sea una corrida nueva
        desde cero, no una continuación)."""
        with self._lock_fes:
            self._fes = 0

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