"""
Backend compilado (Numba @njit) para las 12 funciones del benchmark CEC 2022.

Reimplementa, función por función, la lógica de `opfunu.cec_based.cec2022`
y `opfunu.utils.operator` (ver esos módulos para la referencia original) —
NO es una reinterpretación "limpia" de la especificación CEC: se porta
literalmente el mismo cómputo (incluyendo comportamientos particulares
como `_rounder`, usada en F4) porque este backend debe reproducir
exactamente los mismos valores que el backend de referencia (opfunu),
solo que compilados en vez de interpretados. La paridad se valida en
`tests/test_cec2022_numba_paridad.py`.

`construir_evaluador(numero_funcion, funcion_opfunu)` es el único punto de
entrada: recibe la instancia de opfunu YA CONSTRUIDA (de donde salen los
arrays de shift/rotación/shuffle, cargados desde `data_2022/`) y devuelve
un callable `f(x) -> float` compilado, listo para usarse en el bucle
caliente de `ProblemaCEC2022.evaluar()`.
"""

import math

import numpy as np
from numba import njit

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


@njit(cache=True)
def _matvec(m, v, out):
    n = v.shape[0]
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += m[i, j] * v[j]
        out[i] = s


@njit(cache=True)
def _restar(x, shift, out):
    for i in range(x.shape[0]):
        out[i] = x[i] - shift[i]


# ---------------------------------------------------------------------
# Funciones elementales (opfunu/utils/operator.py)
# ---------------------------------------------------------------------


@njit(cache=True)
def _zakharov(z):
    s2 = 0.0
    s1 = 0.0
    for i in range(z.shape[0]):
        s2 += z[i] * z[i]
        s1 += 0.5 * z[i]
    return s2 + s1 ** 2 + s1 ** 4


@njit(cache=True)
def _rosenbrock(z):
    total = 0.0
    for i in range(z.shape[0] - 1):
        t1 = z[i] * z[i] - z[i + 1]
        t2 = z[i] - 1.0
        total += 100.0 * t1 * t1 + t2 * t2
    return total


@njit(cache=True)
def _rotated_expanded_schaffer(z):
    n = z.shape[0]
    total = 0.0
    for i in range(n):
        a = z[i]
        b = z[(i + 1) % n]
        sum_sq = a * a + b * b
        s = math.sin(math.sqrt(sum_sq))
        total += 0.5 + (s * s - 0.5) / (1.0 + 0.001 * sum_sq) ** 2
    return total


@njit(cache=True)
def _rounder(x):
    """Puerto literal de operator.rounder(x, condition=abs(x)) — ver
    opfunu/utils/operator.py. No "corregir": debe reproducir el mismo
    comportamiento (incluyendo casos donde no redondea al entero más
    cercano) que el backend de referencia."""
    temp_2x = 2.0 * x
    inter = float(int(temp_2x))   # trunca hacia cero, como np.modf
    dec = temp_2x - inter
    if temp_2x <= 0.0:
        temp_2x = inter - (1.0 if dec >= 0.5 else 0.0)
    if dec < 0.5:
        temp_2x = inter
    if dec >= 0.5:
        temp_2x = inter + 1.0
    if abs(x) < 0.5:
        return x
    return temp_2x / 2.0


@njit(cache=True)
def _non_continuous_rastrigin(z):
    n = z.shape[0]
    y = np.empty(n)
    for i in range(n):
        y[i] = _rounder(z[i])
    total = 0.0
    for i in range(n):
        yi = y[i]
        ys = y[(i + 1) % n]
        total += yi * yi - 10.0 * math.cos(2.0 * math.pi * yi) + 10.0
        total += ys * ys - 10.0 * math.cos(2.0 * math.pi * ys) + 10.0
    return total


@njit(cache=True)
def _levy(z, shift):
    n = z.shape[0]
    w = np.empty(n)
    for i in range(n):
        w[i] = 1.0 + (z[i] + shift - 1.0) / 4.0
    t1 = math.sin(math.pi * w[0]) ** 2 + (w[n - 1] - 1.0) ** 2 * (
        1.0 + math.sin(2.0 * math.pi * w[n - 1]) ** 2)
    t2 = 0.0
    for i in range(n - 1):
        t2 += (w[i] - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * w[i] + 1.0) ** 2)
    return t1 + t2


@njit(cache=True)
def _bent_cigar(z):
    total = z[0] * z[0]
    for i in range(1, z.shape[0]):
        total += 1e6 * z[i] * z[i]
    return total


@njit(cache=True)
def _hgbat(z, shift):
    t1 = 0.0
    t2 = 0.0
    for i in range(z.shape[0]):
        v = z[i] + shift
        t1 += v
        t2 += v * v
    n = z.shape[0]
    return abs(t2 * t2 - t1 * t1) ** 0.5 + (0.5 * t2 + t1) / n + 0.5


@njit(cache=True)
def _rastrigin(z):
    total = 0.0
    for i in range(z.shape[0]):
        total += z[i] * z[i] - 10.0 * math.cos(2.0 * math.pi * z[i]) + 10.0
    return total


@njit(cache=True)
def _katsuura(z):
    n = z.shape[0]
    result = 1.0
    for idx in range(n):
        temp = 0.0
        p = 2.0
        for _j in range(1, 33):
            temp += abs(p * z[idx] - round(p * z[idx])) / p
            p *= 2.0
        result *= (1.0 + (idx + 1) * temp) ** (10.0 / n ** 1.2)
    return (result - 1.0) * 10.0 / (n * n)


@njit(cache=True)
def _ackley(z):
    n = z.shape[0]
    t1 = 0.0
    t2 = 0.0
    for i in range(n):
        t1 += z[i] * z[i]
        t2 += math.cos(2.0 * math.pi * z[i])
    return (-20.0 * math.exp(-0.2 * math.sqrt(t1 / n)) - math.exp(t2 / n)
            + 20.0 + math.e)


@njit(cache=True)
def _modified_schwefel(z):
    n = z.shape[0]
    total = 0.0
    for i in range(n):
        v = z[i] + 4.209687462275036e2
        if v > 500.0:
            # math.fmod no está tipado por esta versión de Numba; abs(v) y
            # 500.0 son siempre no-negativos, así que "%" coincide con fmod.
            resto = abs(v) % 500.0
            total -= ((500.0 + resto)
                      * math.sin(math.sqrt(500.0 - resto))
                      - ((v - 500.0) / 100.0) ** 2 / n)
        elif v < -500.0:
            resto = abs(v) % 500.0
            total -= ((-500.0 + resto)
                      * math.sin(math.sqrt(500.0 - resto))
                      - ((v + 500.0) / 100.0) ** 2 / n)
        else:
            total -= v * math.sin(math.sqrt(abs(v)))
    return total + 4.189828872724338e2 * n


@njit(cache=True)
def _schaffer_f7(z):
    n = z.shape[0]
    result = 0.0
    for idx in range(n - 1):
        t = z[idx] * z[idx] + z[idx + 1] * z[idx + 1]
        result += math.sqrt(t) * (math.sin(50.0 * t ** 0.2) + 1.0)
    return (result / (n - 1)) ** 2


@njit(cache=True)
def _elliptic(z):
    n = z.shape[0]
    total = 0.0
    for i in range(n):
        total += 10.0 ** (6.0 * i / (n - 1)) * z[i] * z[i]
    return total


@njit(cache=True)
def _discus(z):
    total = 1e6 * z[0] * z[0]
    for i in range(1, z.shape[0]):
        total += z[i] * z[i]
    return total


@njit(cache=True)
def _griewank(z):
    n = z.shape[0]
    t1 = 0.0
    t2 = 1.0
    for i in range(n):
        t1 += z[i] * z[i]
        t2 *= math.cos(z[i] / math.sqrt(i + 1.0))
    return t1 / 4000.0 - t2 + 1.0


@njit(cache=True)
def _calculate_weight(diff, delta):
    n = diff.shape[0]
    temp = 0.0
    for i in range(n):
        temp += diff[i] * diff[i]
    if temp != 0.0:
        return math.sqrt(1.0 / temp) * math.exp(-temp / (2.0 * n * delta * delta))
    return 1e99


@njit(cache=True)
def _grie_rosen_cec(z_in):
    n = z_in.shape[0]
    z = np.empty(n)
    for i in range(n):
        z[i] = z_in[i] + 1.0
    f = 0.0
    for i in range(n - 1):
        tmp1 = (z[i] * z[i] - z[i + 1]) ** 2
        tmp2 = (z[i] - 1.0) ** 2
        temp = 100.0 * tmp1 + tmp2
        f += temp * temp / 4000.0 - math.cos(temp) + 1.0
    tmp1 = (z[n - 1] * z[n - 1] - z[0]) ** 2
    tmp2 = (z[n - 1] - 1.0) ** 2
    temp = 100.0 * tmp1 + tmp2
    f += temp * temp / 4000.0 - math.cos(temp) + 1.0
    return f


@njit(cache=True)
def _happy_cat(z, shift):
    n = z.shape[0]
    t1 = 0.0
    t2 = 0.0
    for i in range(n):
        v = z[i] + shift
        t1 += v
        t2 += v * v
    return abs(t2 - n) ** 0.25 + (0.5 * t2 + t1) / n + 0.5


# ---------------------------------------------------------------------
# F1-F5: shift + rotación simple (opfunu: clase F12022..F52022)
# ---------------------------------------------------------------------


@njit(cache=True)
def _eval_f1(x, shift, matrix, bias):
    n = x.shape[0]
    d = np.empty(n)
    _restar(x, shift, d)
    z = np.empty(n)
    _matvec(matrix, d, z)
    return _zakharov(z) + bias


@njit(cache=True)
def _eval_f2(x, shift, matrix, bias):
    n = x.shape[0]
    d = np.empty(n)
    for i in range(n):
        d[i] = 2.048 * (x[i] - shift[i]) / 100.0
    z = np.empty(n)
    _matvec(matrix, d, z)
    for i in range(n):
        z[i] += 1.0
    return _rosenbrock(z) + bias


@njit(cache=True)
def _eval_f3(x, shift, matrix, bias):
    n = x.shape[0]
    d = np.empty(n)
    for i in range(n):
        d[i] = 0.5 * (x[i] - shift[i]) / 100.0
    z = np.empty(n)
    _matvec(matrix, d, z)
    return _rotated_expanded_schaffer(z) + bias


@njit(cache=True)
def _eval_f4(x, shift, matrix, bias):
    n = x.shape[0]
    d = np.empty(n)
    for i in range(n):
        d[i] = 5.12 * (x[i] - shift[i]) / 100.0
    z = np.empty(n)
    _matvec(matrix, d, z)
    return _non_continuous_rastrigin(z) + bias


@njit(cache=True)
def _eval_f5(x, shift, matrix, bias):
    n = x.shape[0]
    d = np.empty(n)
    for i in range(n):
        d[i] = 5.12 * (x[i] - shift[i]) / 100.0
    z = np.empty(n)
    _matvec(matrix, d, z)
    return _levy(z, 1.0) + bias


# ---------------------------------------------------------------------
# F6-F8: híbridas (shift, shuffle por bloques, rotación, suma de sub-funciones)
# ---------------------------------------------------------------------


@njit(cache=True)
def _eval_f6(x, shift, matrix, perm, n1, n2, bias):
    n = x.shape[0]
    z = np.empty(n)
    _restar(x, shift, z)
    z1 = np.empty(n)
    for i in range(n):
        z1[i] = z[perm[i]]
    mz = np.empty(n)
    _matvec(matrix, z1, mz)
    return (_bent_cigar(mz[0:n1]) + _hgbat(mz[n1:n2], -1.0)
            + _rastrigin(mz[n2:n]) + bias)


@njit(cache=True)
def _eval_f7(x, shift, matrix, perm, n1, n2, n3, n4, n5, bias):
    n = x.shape[0]
    z = np.empty(n)
    _restar(x, shift, z)
    z1 = np.empty(n)
    for i in range(n):
        z1[i] = z[perm[i]]
    mz = np.empty(n)
    _matvec(matrix, z1, mz)
    return (_hgbat(mz[0:n1], -1.0) + _katsuura(mz[n1:n2]) + _ackley(mz[n2:n3])
            + _rastrigin(mz[n3:n4]) + _modified_schwefel(mz[n4:n5])
            + _schaffer_f7(mz[n5:n]) + bias)


@njit(cache=True)
def _eval_f8(x, shift, matrix, perm, n1, n2, n3, n4, bias):
    n = x.shape[0]
    z = np.empty(n)
    _restar(x, shift, z)
    z1 = np.empty(n)
    for i in range(n):
        z1[i] = z[perm[i]]
    mz = np.empty(n)
    _matvec(matrix, z1, mz)
    return (_katsuura(mz[0:n1]) + _happy_cat(mz[n1:n2], -1.0)
            + _grie_rosen_cec(mz[n2:n3]) + _modified_schwefel(mz[n3:n4])
            + _ackley(mz[n4:n]) + bias)


# ---------------------------------------------------------------------
# F9-F12: composición (5-6 sub-funciones ponderadas por calculate_weight)
# ---------------------------------------------------------------------


@njit(cache=True)
def _eval_f9(x, shift, matrix, xichmas, lamdas, biask, f_bias):
    n = x.shape[0]
    d0 = np.empty(n)
    _restar(x, shift[0], d0)

    d0s = np.empty(n)
    for i in range(n):
        d0s[i] = 2.048 * d0[i] / 100.0
    z0 = np.empty(n)
    _matvec(matrix[0:n, :], d0s, z0)
    for i in range(n):
        z0[i] += 1.0
    g0 = lamdas[0] * _rosenbrock(z0) + biask[0]
    w0 = _calculate_weight(d0, xichmas[0])

    z1 = np.empty(n)
    _matvec(matrix[n:2 * n, :], d0, z1)
    g1 = lamdas[1] * _elliptic(z1) + biask[1]
    diff1 = np.empty(n)
    _restar(x, shift[1], diff1)
    w1 = _calculate_weight(diff1, xichmas[1])

    z2 = np.empty(n)
    _matvec(matrix[2 * n:3 * n, :], d0, z2)
    g2 = lamdas[2] * _bent_cigar(z2) + biask[2]
    diff2 = np.empty(n)
    _restar(x, shift[2], diff2)
    w2 = _calculate_weight(diff2, xichmas[2])

    z3 = np.empty(n)
    _matvec(matrix[3 * n:4 * n, :], d0, z3)
    g3 = lamdas[3] * _discus(z3) + biask[3]
    diff3 = np.empty(n)
    _restar(x, shift[3], diff3)
    w3 = _calculate_weight(diff3, xichmas[3])

    z4 = np.empty(n)
    _matvec(matrix[4 * n:5 * n, :], d0, z4)
    g4 = lamdas[4] * _elliptic(z4) + biask[4]
    diff4 = np.empty(n)
    _restar(x, shift[4], diff4)
    w4 = _calculate_weight(diff4, xichmas[4])

    wsum = w0 + w1 + w2 + w3 + w4
    return (w0 * g0 + w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4) / wsum + f_bias


@njit(cache=True)
def _eval_f10(x, shift, matrix, xichmas, lamdas, biask, f_bias):
    n = x.shape[0]
    d0 = np.empty(n)
    _restar(x, shift[0], d0)

    s0 = np.empty(n)
    for i in range(n):
        s0[i] = (1000.0 / 100.0) * d0[i]
    z0 = np.empty(n)
    _matvec(matrix[0:n, :], s0, z0)
    g0 = lamdas[0] * _modified_schwefel(z0) + biask[0]
    w0 = _calculate_weight(d0, xichmas[0])

    s1 = np.empty(n)
    for i in range(n):
        s1[i] = (5.12 / 100.0) * d0[i]
    z1 = np.empty(n)
    _matvec(matrix[n:2 * n, :], s1, z1)
    g1 = lamdas[1] * _rastrigin(z1) + biask[1]
    diff1 = np.empty(n)
    _restar(x, shift[1], diff1)
    w1 = _calculate_weight(diff1, xichmas[1])

    s2 = np.empty(n)
    for i in range(n):
        s2[i] = (5.0 / 100.0) * d0[i]
    z2 = np.empty(n)
    _matvec(matrix[2 * n:3 * n, :], s2, z2)
    g2 = lamdas[2] * _hgbat(z2, -1.0) + biask[2]
    diff2 = np.empty(n)
    _restar(x, shift[2], diff2)
    w2 = _calculate_weight(diff2, xichmas[2])

    wsum = w0 + w1 + w2
    return (w0 * g0 + w1 * g1 + w2 * g2) / wsum + f_bias


@njit(cache=True)
def _eval_f11(x, shift, matrix, xichmas, lamdas, biask, f_bias):
    n = x.shape[0]
    d0 = np.empty(n)
    _restar(x, shift[0], d0)

    s0 = np.empty(n)
    for i in range(n):
        s0[i] = 0.5 * d0[i] / 100.0
    z0 = np.empty(n)
    _matvec(matrix[0:n, :], s0, z0)
    g0 = lamdas[0] * _rotated_expanded_schaffer(z0) + biask[0]
    w0 = _calculate_weight(d0, xichmas[0])

    s1 = np.empty(n)
    for i in range(n):
        s1[i] = 1000.0 * d0[i] / 100.0
    z1 = np.empty(n)
    _matvec(matrix[n:2 * n, :], s1, z1)
    g1 = lamdas[1] * _modified_schwefel(z1) + biask[1]
    diff1 = np.empty(n)
    _restar(x, shift[1], diff1)
    w1 = _calculate_weight(diff1, xichmas[1])

    s2 = np.empty(n)
    for i in range(n):
        s2[i] = 600.0 * d0[i] / 100.0
    z2 = np.empty(n)
    _matvec(matrix[2 * n:3 * n, :], s2, z2)
    g2 = lamdas[2] * _griewank(z2) + biask[2]
    diff2 = np.empty(n)
    _restar(x, shift[2], diff2)
    w2 = _calculate_weight(diff2, xichmas[2])

    s3 = np.empty(n)
    for i in range(n):
        s3[i] = 2.048 * d0[i] / 100.0
    z3 = np.empty(n)
    _matvec(matrix[3 * n:4 * n, :], s3, z3)
    g3 = lamdas[3] * _rosenbrock(z3) + biask[3]
    diff3 = np.empty(n)
    _restar(x, shift[3], diff3)
    w3 = _calculate_weight(diff3, xichmas[3])

    z4 = np.empty(n)
    _matvec(matrix[4 * n:5 * n, :], d0, z4)
    g4 = lamdas[4] * _rastrigin(z4) + biask[4]
    diff4 = np.empty(n)
    _restar(x, shift[4], diff4)
    w4 = _calculate_weight(diff4, xichmas[4])

    wsum = w0 + w1 + w2 + w3 + w4
    return (w0 * g0 + w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4) / wsum + f_bias


@njit(cache=True)
def _eval_f12(x, shift, matrix, xichmas, lamdas, biask, f_bias):
    n = x.shape[0]
    d0 = np.empty(n)
    _restar(x, shift[0], d0)

    s0 = np.empty(n)
    for i in range(n):
        s0[i] = 5.0 * d0[i] / 100.0
    z0 = np.empty(n)
    _matvec(matrix[0:n, :], s0, z0)
    g0 = lamdas[0] * _hgbat(z0, -1.0) + biask[0]
    w0 = _calculate_weight(d0, xichmas[0])

    s1 = np.empty(n)
    for i in range(n):
        s1[i] = 5.12 * d0[i] / 100.0
    z1 = np.empty(n)
    _matvec(matrix[n:2 * n, :], s1, z1)
    g1 = lamdas[1] * _rastrigin(z1) + biask[1]
    diff1 = np.empty(n)
    _restar(x, shift[1], diff1)
    w1 = _calculate_weight(diff1, xichmas[1])

    s2 = np.empty(n)
    for i in range(n):
        s2[i] = 1000.0 * d0[i] / 100.0
    z2 = np.empty(n)
    _matvec(matrix[2 * n:3 * n, :], s2, z2)
    g2 = lamdas[2] * _modified_schwefel(z2) + biask[2]
    diff2 = np.empty(n)
    _restar(x, shift[2], diff2)
    w2 = _calculate_weight(diff2, xichmas[2])

    z3 = np.empty(n)
    _matvec(matrix[3 * n:4 * n, :], d0, z3)
    g3 = lamdas[3] * _bent_cigar(z3) + biask[3]
    diff3 = np.empty(n)
    _restar(x, shift[3], diff3)
    w3 = _calculate_weight(diff3, xichmas[3])

    z4 = np.empty(n)
    _matvec(matrix[4 * n:5 * n, :], d0, z4)
    g4 = lamdas[4] * _elliptic(z4) + biask[4]
    diff4 = np.empty(n)
    _restar(x, shift[4], diff4)
    w4 = _calculate_weight(diff4, xichmas[4])

    z5 = np.empty(n)
    _matvec(matrix[5 * n:6 * n, :], d0, z5)
    g5 = lamdas[5] * _rotated_expanded_schaffer(z5) + biask[5]
    diff5 = np.empty(n)
    _restar(x, shift[5], diff5)
    w5 = _calculate_weight(diff5, xichmas[5])

    wsum = w0 + w1 + w2 + w3 + w4 + w5
    return ((w0 * g0 + w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / wsum
            + f_bias)


# ---------------------------------------------------------------------
# Factory: extrae los datos de la instancia opfunu y arma el closure
# ---------------------------------------------------------------------


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _i64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def construir_evaluador(numero_funcion: int, funcion_opfunu):
    """
    Construye el evaluador compilado para `numero_funcion` (1..12), a
    partir de la instancia YA CONSTRUIDA de opfunu (de donde salen los
    arrays de shift/rotación/shuffle cargados desde data_2022/).

    Devuelve un callable `f(x: np.ndarray) -> float`. La primera llamada
    dispara la compilación JIT (costo único, ~decenas-cientos de ms por
    función/dimensión); `ProblemaCEC2022` hace un warm-up explícito para
    que ese costo no caiga en medio de una corrida real.
    """
    fo = funcion_opfunu
    bias = float(fo.f_bias)

    if numero_funcion in (1, 2, 3, 4, 5):
        shift = _f64(fo.f_shift)
        matrix = _f64(fo.f_matrix)
        kernel = {1: _eval_f1, 2: _eval_f2, 3: _eval_f3,
                  4: _eval_f4, 5: _eval_f5}[numero_funcion]
        return lambda x: kernel(_f64(x), shift, matrix, bias)

    if numero_funcion in (6, 7, 8):
        shift = _f64(fo.f_shift)
        matrix = _f64(fo.f_matrix)
        if numero_funcion == 6:
            perm = _i64(np.concatenate((fo.idx1, fo.idx2, fo.idx3)))
            n1, n2 = int(fo.n1), int(fo.n2)
            return lambda x: _eval_f6(_f64(x), shift, matrix, perm, n1, n2, bias)
        if numero_funcion == 7:
            perm = _i64(np.concatenate(
                (fo.idx1, fo.idx2, fo.idx3, fo.idx4, fo.idx5, fo.idx6)))
            n1, n2, n3, n4, n5 = (int(fo.n1), int(fo.n2), int(fo.n3),
                                  int(fo.n4), int(fo.n5))
            return lambda x: _eval_f7(_f64(x), shift, matrix, perm,
                                       n1, n2, n3, n4, n5, bias)
        perm = _i64(np.concatenate(
            (fo.idx1, fo.idx2, fo.idx3, fo.idx4, fo.idx5)))
        n1, n2, n3, n4 = int(fo.n1), int(fo.n2), int(fo.n3), int(fo.n4)
        return lambda x: _eval_f8(_f64(x), shift, matrix, perm,
                                   n1, n2, n3, n4, bias)

    # F9-F12: composición
    shift = _f64(fo.f_shift)
    matrix = _f64(fo.f_matrix)
    xichmas = _f64(np.asarray(fo.xichmas, dtype=np.float64))
    lamdas = _f64(np.asarray(fo.lamdas, dtype=np.float64))
    biask = _f64(np.asarray(fo.bias, dtype=np.float64))
    kernel = {9: _eval_f9, 10: _eval_f10, 11: _eval_f11, 12: _eval_f12}[numero_funcion]
    return lambda x: kernel(_f64(x), shift, matrix, xichmas, lamdas, biask, bias)
