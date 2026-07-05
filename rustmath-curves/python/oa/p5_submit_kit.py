"""P5/P6 hardening kit (from the 2026-07-06 reply): IRLS weights, two-layer collapse
guard, monicize/validate/emit helpers. Import into the VARPRO driver and Phase V."""
import numpy as np
from math import gcd
from functools import reduce


def irls_weights(Pvals, Qvals, phivals, eps=1e-300):
    scale = np.abs(Pvals) + np.abs(phivals * Qvals)
    return 1.0 / np.maximum(scale, eps)


def sylvester_matrix(p, q):
    p = np.asarray(p, dtype=np.complex128); q = np.asarray(q, dtype=np.complex128)
    m, n = len(p) - 1, len(q) - 1
    S = np.zeros((m + n, m + n), dtype=np.complex128)
    for i in range(m):
        S[i, i:i+n+1] = q
    for i in range(n):
        S[m+i, i:i+m+1] = p
    return S


def common_root_score(p, q):
    s = np.linalg.svd(sylvester_matrix(p, q), compute_uv=False)
    return s[-1] / max(s[0], 1e-300)


def collapse_guard(old_score, new_score, shrink_limit=10.0):
    """Two-layer use: apply to BOTH min_i|Q(y_i)| and common_root_score(P,Q)."""
    return new_score >= old_score / shrink_limit


def clear_denominators_rational_poly(coeffs):
    """coeffs ascending Fractions -> primitive integer coeffs ascending, lead > 0."""
    from math import lcm
    den = 1
    for c in coeffs:
        den = lcm(den, c.denominator)
    ints = [int(c * den) for c in coeffs]
    cont = reduce(gcd, (abs(x) for x in ints if x != 0), 0)
    ints = [x // cont for x in ints]
    if ints[-1] < 0:
        ints = [-x for x in ints]
    return ints


def monicize_degree24(int_coeffs):
    """ascending deg-24: G(y) = ell^23 F(y/ell); monic integral, same field."""
    if len(int_coeffs) != 25:
        raise ValueError("degree-24 polynomial must have 25 coefficients")
    ell = int_coeffs[-1]
    if ell == 0:
        raise ValueError("leading coefficient is zero")
    if ell == 1:
        out = int_coeffs[:]
    else:
        out = [(1 if k == 24 else ak * (ell ** (23 - k))) for k, ak in enumerate(int_coeffs)]
    cont = reduce(gcd, (abs(x) for x in out), 0)
    if cont != 1:
        out = [x // cont for x in out]
    if out[-1] != 1:
        raise AssertionError("monicization failed")
    return out


def validate_submission_coeffs(coeffs):
    if len(coeffs) != 25:
        raise ValueError(f"need exactly 25 coefficients, got {len(coeffs)}")
    if coeffs[0] == 0:
        raise ValueError("a_0 must be nonzero")
    if coeffs[-1] != 1:
        raise ValueError("a_24 must be 1")
    g = reduce(gcd, (abs(int(c)) for c in coeffs), 0)
    if g != 1:
        raise ValueError(f"coefficient gcd must be 1, got {g}")
    return True


def emit_competition_string(coeffs):
    validate_submission_coeffs(coeffs)
    return ",".join(str(int(c)) for c in coeffs)
