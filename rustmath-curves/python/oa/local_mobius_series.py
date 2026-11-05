# local_mobius_series.py
"""P5 per-chart residuals: compose the global P, Q through a Moebius gluing as
local power series, so structured constraints are imposed where they are
conditioned. At the order-12 chart b the residual

    P(mu_b(x_b)) - phi_b(x_b) Q(mu_b(x_b))

must have coefficients 0..11 vanishing and coefficient 12 nonzero -- for each
of the two 12-cycle charts. Never impose a 12-fold zero through the global
pinhole coordinate.
"""

import numpy as np


def series_mul(a, b, n):
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)

    out = np.zeros(n, dtype=np.complex128)
    for i in range(min(len(a), n)):
        if abs(a[i]) == 0:
            continue
        for j in range(min(len(b), n - i)):
            out[i + j] += a[i] * b[j]
    return out


def series_inv(a, n):
    """
    Invert a series a with nonzero constant term.
    """
    a = np.asarray(a, dtype=np.complex128)
    out = np.zeros(n, dtype=np.complex128)
    out[0] = 1 / a[0]

    for k in range(1, n):
        s = 0
        for j in range(1, k + 1):
            if j < len(a):
                s += a[j] * out[k - j]
        out[k] = -s / a[0]

    return out


def series_pow(a, k, n):
    out = np.zeros(n, dtype=np.complex128)
    out[0] = 1

    for _ in range(k):
        out = series_mul(out, a, n)

    return out


def mobius_series(alpha, beta, gamma, delta, n):
    """
    Series for X(x)=(alpha+beta*x)/(gamma+delta*x).
    """
    num = np.zeros(n, dtype=np.complex128)
    den = np.zeros(n, dtype=np.complex128)

    num[0] = alpha
    if n > 1:
        num[1] = beta

    den[0] = gamma
    if n > 1:
        den[1] = delta

    return series_mul(num, series_inv(den, n), n)


def poly_compose_in_series(poly_coeffs, Xser, n):
    """
    poly_coeffs ascending: p0 + p1 X + ... + pd X^d.
    """
    out = np.zeros(n, dtype=np.complex128)

    for k, pk in enumerate(poly_coeffs):
        if abs(pk) == 0:
            continue
        out += pk * series_pow(Xser, k, n)

    return out


def local_residual_series(P, Q, phi_local, mobius, n):
    """
    Residual:
        P(mu(x)) - phi_local(x) Q(mu(x)).
    """
    Xser = mobius_series(
        mobius["alpha"], mobius["beta"], mobius["gamma"], mobius["delta"], n
    )

    Pser = poly_compose_in_series(P, Xser, n)
    Qser = poly_compose_in_series(Q, Xser, n)

    return Pser - series_mul(phi_local, Qser, n)
