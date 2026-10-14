"""
Backend compilado (Numba @njit) para las 12 funciones del benchmark CEC 2022.

Port LITERAL del código C oficial de la competición
(`CEC2022/C-Code/cec22_test_func.cpp`, github.com/P-N-Suganthan/2022-SO-BO),
función por función (D-9, docs/configuracion_experimental.md). Hasta D-9
este módulo portaba `opfunu.cec_based.cec2022`, que difiere del oficial en
11 de las 12 funciones (Zakharov sin el peso (i+1), Schwefel modificado con
el signo de `fmod` cambiado, factores de escala ausentes en las híbridas,
Levy y Schaffer F7 distintos, etc.); todas coinciden solo en el óptimo x*.

"Literal" incluye los efectos de los buffers globales `y` y `z` del C, que
aquí se pasan explícitos como arrays de trabajo:
  - `sr_func` escribe el vector desplazado y escalado en `y` (cuando rota)
    y el resultado final en `z`.
  - `schaffer_F7_func` lee `y`, no `z`: en F3 usa el vector desplazado SIN
    rotar, y dentro de la híbrida F7 (hf10) lee el `y` de la híbrida desde
    el índice 0, no el segmento que le toca.
  - `step_rastrigin_func` (F4) redondea un `y` de la llamada ANTERIOR que
    `sr_func` sobrescribe a continuación: el redondeo no tiene efecto y F4
    es, en la práctica, Rastrigin desplazado-rotado. Se omite ese bucle.
  - Las sub-funciones de las híbridas reciben `&y[G[i]]` (vista de `y`) y
    aplican su propio factor de escala con `s_flag = r_flag = 0`.

Los datos (M, shift, shuffle) se leen de los archivos `data_2022/` que trae
opfunu — idénticos byte a byte en valor a `input_data/` del paquete oficial
(verificado, D-9) — replicando cómo los lee el C con `fscanf`.

Validación: `tests/test_cec2022_numba_paridad.py` compara contra
`problems/cec2022_referencia.py` (port Python oficial de los organizadores),
que a su vez coincide con el C oficial compilado hasta ≤ 4·10⁻¹⁶ relativo
en las 12 funciones × D ∈ {10, 20} (D-9).

`construir_evaluador(numero_funcion, ndim)` es el único punto de entrada.
"""

import math
import os

import numpy as np
from numba import njit

# Etiqueta del evaluador para la columna `version` de cada corrida (§11.2).
# Las corridas hechas con el port de opfunu (antes de D-9) llevan
# "eval:numba" y NO son comparables: grid.py las re-ejecuta y analisis/ las
# rechaza.
VERSION_EVALUADOR = "numba-c22oficial"

E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029
INF = 1.0e99


def directorio_datos() -> str:
    """Carpeta `data_2022/` de opfunu (mismos valores que el `input_data/`
    oficial de CEC 2022)."""
    import opfunu
    return os.path.join(os.path.dirname(opfunu.__file__), "cec_based", "data_2022")


# ---------------------------------------------------------------------
# Desplazamiento y rotación (sr_func, shiftfunc, rotatefunc)
# ---------------------------------------------------------------------


@njit(cache=True)
def _rotar(v, out, nx, mr, off_m):
    for i in range(nx):
        out[i] = 0.0
        for j in range(nx):
            out[i] = out[i] + v[j] * mr[off_m + i * nx + j]


@njit(cache=True)
def _sr(x, nx, os_, off_s, mr, off_m, sh_rate, s_flag, r_flag, y, z):
    """sr_func: el resultado queda en `z` (en el C, `sr_x` es siempre `z`)."""
    if s_flag == 1:
        if r_flag == 1:
            for i in range(nx):
                y[i] = x[i] - os_[off_s + i]
            for i in range(nx):
                y[i] = y[i] * sh_rate
            _rotar(y, z, nx, mr, off_m)
        else:
            for i in range(nx):
                z[i] = x[i] - os_[off_s + i]
            for i in range(nx):
                z[i] = z[i] * sh_rate
    else:
        if r_flag == 1:
            for i in range(nx):
                y[i] = x[i] * sh_rate
            _rotar(y, z, nx, mr, off_m)
        else:
            for i in range(nx):
                z[i] = x[i] * sh_rate


# ---------------------------------------------------------------------
# Funciones básicas (misma firma lógica que el C; devuelven f)
# ---------------------------------------------------------------------


@njit(cache=True)
def _ellips(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    f = 0.0
    _sr(x, nx, os_, off_s, mr, off_m, 1.0, s_flag, r_flag, y, z)
    for i in range(nx):
        f += 10.0 ** (6.0 * i / (nx - 1)) * z[i] * z[i]
    return f


@njit(cache=True)
def _bent_cigar(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    _sr(x, nx, os_, off_s, mr, off_m, 1.0, s_flag, r_flag, y, z)
    f = z[0] * z[0]
    for i in range(1, nx):
        f += 1.0e6 * z[i] * z[i]
    return f


@njit(cache=True)
def _discus(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    _sr(x, nx, os_, off_s, mr, off_m, 1.0, s_flag, r_flag, y, z)
    f = 1.0e6 * z[0] * z[0]
    for i in range(1, nx):
        f += z[i] * z[i]
    return f


@njit(cache=True)
def _rosenbrock(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    f = 0.0
    _sr(x, nx, os_, off_s, mr, off_m, 2.048 / 100.0, s_flag, r_flag, y, z)
    z[0] += 1.0
    for i in range(nx - 1):
        z[i + 1] += 1.0
        tmp1 = z[i] * z[i] - z[i + 1]
        tmp2 = z[i] - 1.0
        f += 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    return f


@njit(cache=True)
def _ackley(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    sum1 = 0.0
    sum2 = 0.0
    _sr(x, nx, os_, off_s, mr, off_m, 1.0, s_flag, r_flag, y, z)
    for i in range(nx):
        sum1 += z[i] * z[i]
        sum2 += math.cos(2.0 * PI * z[i])
    sum1 = -0.2 * math.sqrt(sum1 / nx)
    sum2 /= nx
    return E - 20.0 * math.exp(sum1) - math.exp(sum2) + 20.0


@njit(cache=True)
def _griewank(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    s = 0.0
    p = 1.0
    _sr(x, nx, os_, off_s, mr, off_m, 600.0 / 100.0, s_flag, r_flag, y, z)
    for i in range(nx):
        s += z[i] * z[i]
        p *= math.cos(z[i] / math.sqrt(1.0 + i))
    return 1.0 + s / 4000.0 - p


@njit(cache=True)
def _rastrigin(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    f = 0.0
    _sr(x, nx, os_, off_s, mr, off_m, 5.12 / 100.0, s_flag, r_flag, y, z)
    for i in range(nx):
        f += z[i] * z[i] - 10.0 * math.cos(2.0 * PI * z[i]) + 10.0
    return f


@njit(cache=True)
def _fmod_pos(a, b):
    # fmod para a ≥ 0, b > 0 (los únicos casos que usa schwefel_func): ahí
    # el "%" de coma flotante coincide exactamente con fmod (math.fmod no
    # está tipado en esta versión de Numba).
    return a % b


@njit(cache=True)
def _schwefel(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    f = 0.0
    _sr(x, nx, os_, off_s, mr, off_m, 1000.0 / 100.0, s_flag, r_flag, y, z)
    for i in range(nx):
        z[i] += 4.209687462275036e+002
        if z[i] > 500:
            r = _fmod_pos(z[i], 500.0)
            f -= (500.0 - r) * math.sin((500.0 - r) ** 0.5)
            tmp = (z[i] - 500.0) / 100
            f += tmp * tmp / nx
        elif z[i] < -500:
            r = _fmod_pos(abs(z[i]), 500.0)
            f -= (-500.0 + r) * math.sin((500.0 - r) ** 0.5)
            tmp = (z[i] + 500.0) / 100
            f += tmp * tmp / nx
        else:
            f -= z[i] * math.sin(abs(z[i]) ** 0.5)
    f += 4.189828872724338e+002 * nx
    return f


@njit(cache=True)
def _grie_rosen(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    f = 0.0
    _sr(x, nx, os_, off_s, mr, off_m, 5.0 / 100.0, s_flag, r_flag, y, z)
    z[0] += 1.0
    for i in range(nx - 1):
        z[i + 1] += 1.0
        tmp1 = z[i] * z[i] - z[i + 1]
        tmp2 = z[i] - 1.0
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        f += (temp * temp) / 4000.0 - math.cos(temp) + 1.0
    tmp1 = z[nx - 1] * z[nx - 1] - z[0]
    tmp2 = z[nx - 1] - 1.0
    temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    f += (temp * temp) / 4000.0 - math.cos(temp) + 1.0
    return f


@njit(cache=True)
def _escaffer6(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    _sr(x, nx, os_, off_s, mr, off_m, 1.0, s_flag, r_flag, y, z)
    f = 0.0
    for i in range(nx - 1):
        temp1 = math.sin(math.sqrt(z[i] * z[i] + z[i + 1] * z[i + 1]))
        temp1 = temp1 * temp1
        temp2 = 1.0 + 0.001 * (z[i] * z[i] + z[i + 1] * z[i + 1])
        f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
    temp1 = math.sin(math.sqrt(z[nx - 1] * z[nx - 1] + z[0] * z[0]))
    temp1 = temp1 * temp1
    temp2 = 1.0 + 0.001 * (z[nx - 1] * z[nx - 1] + z[0] * z[0])
    f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
    return f


@njit(cache=True)
def _happycat(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    alpha = 1.0 / 8.0
    _sr(x, nx, os_, off_s, mr, off_m, 5.0 / 100.0, s_flag, r_flag, y, z)
    r2 = 0.0
    sum_z = 0.0
    for i in range(nx):
        z[i] = z[i] - 1.0
        r2 += z[i] * z[i]
        sum_z += z[i]
    return abs(r2 - nx) ** (2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5


@njit(cache=True)
def _hgbat(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    alpha = 1.0 / 4.0
    _sr(x, nx, os_, off_s, mr, off_m, 5.0 / 100.0, s_flag, r_flag, y, z)
    r2 = 0.0
    sum_z = 0.0
    for i in range(nx):
        z[i] = z[i] - 1.0
        r2 += z[i] * z[i]
        sum_z += z[i]
    return abs(r2 ** 2.0 - sum_z ** 2.0) ** (2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5


@njit(cache=True)
def _schaffer_f7(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    # Lee `y` (no `z`), como el C — ver docstring del módulo.
    f = 0.0
    _sr(x, nx, os_, off_s, mr, off_m, 1.0, s_flag, r_flag, y, z)
    for i in range(nx - 1):
        z[i] = (y[i] * y[i] + y[i + 1] * y[i + 1]) ** 0.5
        tmp = math.sin(50.0 * z[i] ** 0.2)
        f += z[i] ** 0.5 + z[i] ** 0.5 * tmp * tmp
    return f * f / (nx - 1) / (nx - 1)


@njit(cache=True)
def _levy(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    _sr(x, nx, os_, off_s, mr, off_m, 1.0, s_flag, r_flag, y, z)
    w = np.empty(nx)
    for i in range(nx):
        w[i] = 1.0 + (z[i] - 0.0) / 4.0
    term1 = math.sin(PI * w[0]) ** 2
    term3 = (w[nx - 1] - 1) ** 2 * (1 + math.sin(2 * PI * w[nx - 1]) ** 2)
    s = 0.0
    for i in range(nx - 1):
        wi = w[i]
        s = s + (wi - 1) ** 2 * (1 + 10 * math.sin(PI * wi + 1) ** 2)
    return term1 + s + term3


@njit(cache=True)
def _zakharov(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    _sr(x, nx, os_, off_s, mr, off_m, 1.0, s_flag, r_flag, y, z)
    sum1 = 0.0
    sum2 = 0.0
    for i in range(nx):
        xi = z[i]
        sum1 = sum1 + xi ** 2
        sum2 = sum2 + 0.5 * (i + 1) * xi
    return sum1 + sum2 ** 2 + sum2 ** 4


@njit(cache=True)
def _katsuura(x, nx, os_, off_s, mr, off_m, s_flag, r_flag, y, z):
    f = 1.0
    tmp3 = (1.0 * nx) ** 1.2
    _sr(x, nx, os_, off_s, mr, off_m, 5.0 / 100.0, s_flag, r_flag, y, z)
    for i in range(nx):
        temp = 0.0
        for j in range(1, 33):
            tmp1 = 2.0 ** j
            tmp2 = tmp1 * z[i]
            temp += abs(tmp2 - math.floor(tmp2 + 0.5)) / tmp1
        f *= (1.0 + (i + 1) * temp) ** (10.0 / tmp3)
    tmp1 = 10.0 / nx / nx
    return f * tmp1 - tmp1


# ---------------------------------------------------------------------
# Híbridas (hf02 = F6, hf10 = F7, hf06 = F8)
# ---------------------------------------------------------------------


@njit(cache=True)
def _permutar(x, nx, os_, mr, ss, y, z):
    """sr_func(x, z, ..., 1.0, 1, 1) seguido de y[i] = z[S[i]-1]."""
    _sr(x, nx, os_, 0, mr, 0, 1.0, 1, 1, y, z)
    for i in range(nx):
        y[i] = z[ss[i] - 1]


@njit(cache=True)
def _hf02(x, nx, os_, mr, ss, g, gn, y, z):
    _permutar(x, nx, os_, mr, ss, y, z)
    f0 = _bent_cigar(y[g[0]:], gn[0], os_, 0, mr, 0, 0, 0, y, z)
    f1 = _hgbat(y[g[1]:], gn[1], os_, 0, mr, 0, 0, 0, y, z)
    f2 = _rastrigin(y[g[2]:], gn[2], os_, 0, mr, 0, 0, 0, y, z)
    f = 0.0
    f += f0
    f += f1
    f += f2
    return f


@njit(cache=True)
def _hf10(x, nx, os_, mr, ss, g, gn, y, z):
    _permutar(x, nx, os_, mr, ss, y, z)
    f0 = _hgbat(y[g[0]:], gn[0], os_, 0, mr, 0, 0, 0, y, z)
    f1 = _katsuura(y[g[1]:], gn[1], os_, 0, mr, 0, 0, 0, y, z)
    f2 = _ackley(y[g[2]:], gn[2], os_, 0, mr, 0, 0, 0, y, z)
    f3 = _rastrigin(y[g[3]:], gn[3], os_, 0, mr, 0, 0, 0, y, z)
    f4 = _schwefel(y[g[4]:], gn[4], os_, 0, mr, 0, 0, 0, y, z)
    f5 = _schaffer_f7(y[g[5]:], gn[5], os_, 0, mr, 0, 0, 0, y, z)
    f = 0.0
    f += f0
    f += f1
    f += f2
    f += f3
    f += f4
    f += f5
    return f


@njit(cache=True)
def _hf06(x, nx, os_, mr, ss, g, gn, y, z):
    _permutar(x, nx, os_, mr, ss, y, z)
    f0 = _katsuura(y[g[0]:], gn[0], os_, 0, mr, 0, 0, 0, y, z)
    f1 = _happycat(y[g[1]:], gn[1], os_, 0, mr, 0, 0, 0, y, z)
    f2 = _grie_rosen(y[g[2]:], gn[2], os_, 0, mr, 0, 0, 0, y, z)
    f3 = _schwefel(y[g[3]:], gn[3], os_, 0, mr, 0, 0, 0, y, z)
    f4 = _ackley(y[g[4]:], gn[4], os_, 0, mr, 0, 0, 0, y, z)
    f = 0.0
    f += f0
    f += f1
    f += f2
    f += f3
    f += f4
    return f


# ---------------------------------------------------------------------
# Composición (cf01 = F9, cf02 = F10, cf06 = F11, cf07 = F12)
# ---------------------------------------------------------------------


@njit(cache=True)
def _cf_cal(x, nx, os_, delta, bias, fit, cf_num):
    w = np.empty(cf_num)
    w_max = 0.0
    w_sum = 0.0
    for i in range(cf_num):
        fit[i] += bias[i]
        w[i] = 0.0
        for j in range(nx):
            w[i] += (x[j] - os_[i * nx + j]) ** 2.0
        if w[i] != 0:
            w[i] = (1.0 / w[i]) ** 0.5 * math.exp(-w[i] / 2.0 / nx / delta[i] ** 2.0)
        else:
            w[i] = INF
        if w[i] > w_max:
            w_max = w[i]
    for i in range(cf_num):
        w_sum = w_sum + w[i]
    if w_max == 0:
        for i in range(cf_num):
            w[i] = 1.0
        w_sum = cf_num
    f = 0.0
    for i in range(cf_num):
        f = f + w[i] / w_sum * fit[i]
    return f


@njit(cache=True)
def _cf01(x, nx, os_, mr, y, z):
    fit = np.empty(5)
    m = nx * nx
    fit[0] = _rosenbrock(x, nx, os_, 0 * nx, mr, 0 * m, 1, 1, y, z)
    fit[0] = 10000 * fit[0] / 1e+4
    fit[1] = _ellips(x, nx, os_, 1 * nx, mr, 1 * m, 1, 1, y, z)
    fit[1] = 10000 * fit[1] / 1e+10
    fit[2] = _bent_cigar(x, nx, os_, 2 * nx, mr, 2 * m, 1, 1, y, z)
    fit[2] = 10000 * fit[2] / 1e+30
    fit[3] = _discus(x, nx, os_, 3 * nx, mr, 3 * m, 1, 1, y, z)
    fit[3] = 10000 * fit[3] / 1e+10
    fit[4] = _ellips(x, nx, os_, 4 * nx, mr, 4 * m, 1, 0, y, z)
    fit[4] = 10000 * fit[4] / 1e+10
    delta = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    bias = np.array([0.0, 200.0, 300.0, 100.0, 400.0])
    return _cf_cal(x, nx, os_, delta, bias, fit, 5)


@njit(cache=True)
def _cf02(x, nx, os_, mr, y, z):
    fit = np.empty(3)
    m = nx * nx
    fit[0] = _schwefel(x, nx, os_, 0 * nx, mr, 0 * m, 1, 0, y, z)
    fit[1] = _rastrigin(x, nx, os_, 1 * nx, mr, 1 * m, 1, 1, y, z)
    fit[2] = _hgbat(x, nx, os_, 2 * nx, mr, 2 * m, 1, 1, y, z)
    delta = np.array([20.0, 10.0, 10.0])
    bias = np.array([0.0, 200.0, 100.0])
    return _cf_cal(x, nx, os_, delta, bias, fit, 3)


@njit(cache=True)
def _cf06(x, nx, os_, mr, y, z):
    fit = np.empty(5)
    m = nx * nx
    fit[0] = _escaffer6(x, nx, os_, 0 * nx, mr, 0 * m, 1, 1, y, z)
    fit[0] = 10000 * fit[0] / 2e+7
    fit[1] = _schwefel(x, nx, os_, 1 * nx, mr, 1 * m, 1, 1, y, z)
    fit[2] = _griewank(x, nx, os_, 2 * nx, mr, 2 * m, 1, 1, y, z)
    fit[2] = 1000 * fit[2] / 100
    fit[3] = _rosenbrock(x, nx, os_, 3 * nx, mr, 3 * m, 1, 1, y, z)
    fit[4] = _rastrigin(x, nx, os_, 4 * nx, mr, 4 * m, 1, 1, y, z)
    fit[4] = 10000 * fit[4] / 1e+3
    delta = np.array([20.0, 20.0, 30.0, 30.0, 20.0])
    bias = np.array([0.0, 200.0, 300.0, 400.0, 200.0])
    return _cf_cal(x, nx, os_, delta, bias, fit, 5)


@njit(cache=True)
def _cf07(x, nx, os_, mr, y, z):
    fit = np.empty(6)
    m = nx * nx
    fit[0] = _hgbat(x, nx, os_, 0 * nx, mr, 0 * m, 1, 1, y, z)
    fit[0] = 10000 * fit[0] / 1000
    fit[1] = _rastrigin(x, nx, os_, 1 * nx, mr, 1 * m, 1, 1, y, z)
    fit[1] = 10000 * fit[1] / 1e+3
    fit[2] = _schwefel(x, nx, os_, 2 * nx, mr, 2 * m, 1, 1, y, z)
    fit[2] = 10000 * fit[2] / 4e+3
    fit[3] = _bent_cigar(x, nx, os_, 3 * nx, mr, 3 * m, 1, 1, y, z)
    fit[3] = 10000 * fit[3] / 1e+30
    fit[4] = _ellips(x, nx, os_, 4 * nx, mr, 4 * m, 1, 1, y, z)
    fit[4] = 10000 * fit[4] / 1e+10
    fit[5] = _escaffer6(x, nx, os_, 5 * nx, mr, 5 * m, 1, 1, y, z)
    fit[5] = 10000 * fit[5] / 2e+7
    delta = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    bias = np.array([0.0, 300.0, 500.0, 100.0, 400.0, 200.0])
    return _cf_cal(x, nx, os_, delta, bias, fit, 6)


# ---------------------------------------------------------------------
# Despacho (cec22_test_func) — bias de cada función sumado al final
# ---------------------------------------------------------------------


@njit(cache=True)
def _evaluar(func_num, x, nx, os_, mr, ss, g, gn, y, z):
    if func_num == 1:
        return _zakharov(x, nx, os_, 0, mr, 0, 1, 1, y, z) + 300.0
    if func_num == 2:
        return _rosenbrock(x, nx, os_, 0, mr, 0, 1, 1, y, z) + 400.0
    if func_num == 3:
        return _schaffer_f7(x, nx, os_, 0, mr, 0, 1, 1, y, z) + 600.0
    if func_num == 4:
        # step_rastrigin_func: su bucle de redondeo actúa sobre un `y` que
        # sr_func sobrescribe enseguida (ver docstring) → Rastrigin.
        return _rastrigin(x, nx, os_, 0, mr, 0, 1, 1, y, z) + 800.0
    if func_num == 5:
        return _levy(x, nx, os_, 0, mr, 0, 1, 1, y, z) + 900.0
    if func_num == 6:
        return _hf02(x, nx, os_, mr, ss, g, gn, y, z) + 1800.0
    if func_num == 7:
        return _hf10(x, nx, os_, mr, ss, g, gn, y, z) + 2000.0
    if func_num == 8:
        return _hf06(x, nx, os_, mr, ss, g, gn, y, z) + 2200.0
    if func_num == 9:
        return _cf01(x, nx, os_, mr, y, z) + 2300.0
    if func_num == 10:
        return _cf02(x, nx, os_, mr, y, z) + 2400.0
    if func_num == 11:
        return _cf06(x, nx, os_, mr, y, z) + 2600.0
    return _cf07(x, nx, os_, mr, y, z) + 2700.0


# ---------------------------------------------------------------------
# Carga de datos (como los fscanf del C) y factory
# ---------------------------------------------------------------------

_GP_HIBRIDAS = {6: (0.4, 0.4, 0.2), 7: (0.1, 0.2, 0.2, 0.2, 0.1, 0.2), 8: (0.3, 0.2, 0.2, 0.1, 0.2)}


def _valores(ruta):
    with open(ruta) as fh:
        return [ln.split() for ln in fh if ln.strip()]


def cargar_datos(numero_funcion: int, ndim: int, directorio: str | None = None):
    """(shift, M, shuffle) aplanados, leídos como el C oficial:
    M = los primeros n·n (F1–F8) o todos los valores (F9–F12) del archivo;
    shift = los primeros n valores (F1–F8) o los primeros n de CADA línea
    (F9–F12); shuffle = n enteros (F6–F8, base 1)."""
    d = directorio or directorio_datos()
    lineas_m = _valores(os.path.join(d, f"M_{numero_funcion}_D{ndim}.txt"))
    m = np.array([float(v) for ln in lineas_m for v in ln])
    if numero_funcion < 9:
        m = m[:ndim * ndim]
    lineas_s = _valores(os.path.join(d, f"shift_data_{numero_funcion}.txt"))
    if numero_funcion < 9:
        shift = np.array([float(v) for ln in lineas_s for v in ln][:ndim])
    else:
        shift = np.array([float(v) for ln in lineas_s for v in ln[:ndim]])
    ss = np.zeros(ndim, dtype=np.int64)
    if 6 <= numero_funcion <= 8:
        ss = np.array([int(float(v)) for ln in _valores(
            os.path.join(d, f"shuffle_data_{numero_funcion}_D{ndim}.txt")) for v in ln][:ndim],
            dtype=np.int64)
    return (np.ascontiguousarray(shift), np.ascontiguousarray(m), np.ascontiguousarray(ss))


def _particion_hibrida(numero_funcion: int, ndim: int):
    """G y G_nx de hf02/hf10/hf06: G_nx[i] = ceil(Gp[i]·n) (con el mismo
    redondeo en coma flotante que el C, p. ej. ceil(0.3·10) = 4)."""
    if numero_funcion not in _GP_HIBRIDAS:
        return np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    gp = _GP_HIBRIDAS[numero_funcion]
    gn = [math.ceil(p * ndim) for p in gp[:-1]]
    gn.append(ndim - sum(gn))
    g = [0]
    for k in gn[:-1]:
        g.append(g[-1] + k)
    return np.array(g, dtype=np.int64), np.array(gn, dtype=np.int64)


def construir_evaluador(numero_funcion: int, ndim: int):
    """
    Evaluador compilado `f(x) -> float` de la función `numero_funcion`
    (1..12) en dimensión `ndim`. La primera llamada dispara la compilación
    JIT; `ProblemaCEC2022` hace un warm-up para que no caiga en la corrida.
    Cada evaluador lleva sus propios buffers `y`, `z` (los globales del C).
    """
    shift, m, ss = cargar_datos(numero_funcion, ndim)
    g, gn = _particion_hibrida(numero_funcion, ndim)
    # Empaquetado en pocos arrays: el despacho Python→Numba cobra por
    # argumento, y este evaluador se llama una vez por evaluación.
    datos = np.ascontiguousarray(np.concatenate((shift, m)))
    meta = np.ascontiguousarray(np.concatenate((ss, g, gn)).astype(np.int64))
    trabajo = np.zeros(2 * ndim)   # buffers y | z (los globales del C)
    n_shift, n_g = len(shift), len(g)

    def evaluar(x):
        xv = np.ascontiguousarray(x, dtype=np.float64)
        return _evaluar_empacado(numero_funcion, xv, ndim, datos, n_shift, meta, n_g, trabajo)
    return evaluar


@njit(cache=True)
def _evaluar_empacado(func_num, x, nx, datos, n_shift, meta, n_g, trabajo):
    os_ = datos[:n_shift]
    mr = datos[n_shift:]
    ss = meta[:nx]
    g = meta[nx:nx + n_g]
    gn = meta[nx + n_g:nx + 2 * n_g]
    y = trabajo[:nx]
    z = trabajo[nx:]
    return _evaluar(func_num, x, nx, os_, mr, ss, g, gn, y, z)
