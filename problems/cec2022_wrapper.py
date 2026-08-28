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

import numpy as np
from opfunu.cec_based import cec2022

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


class ProblemaCEC2022:
    """
    Representa una instancia de problema: una función del benchmark CEC 2022
    en una dimensionalidad específica.
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

        # Valor óptimo conocido (bias), usado para calcular el error de
        # convergencia en el análisis estadístico (Parte B del protocolo).
        self.optimo_global = getattr(self._funcion, "f_global", None)

    def evaluar(self, x: np.ndarray) -> float:
        """Interfaz uniforme: recibe un vector y retorna un escalar."""
        return float(self._funcion.evaluate(x))

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluar(x)

    def __repr__(self):
        return f"ProblemaCEC2022(F{self.numero_funcion}, D={self.ndim}, '{self.nombre}')"