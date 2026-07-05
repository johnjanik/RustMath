# sym8_glue.py
"""P4.5 core module: Sym^8 Moebius gluing of Veronese-normalized local frames.

A Moebius map mu(x) = (alpha + beta x)/(gamma + delta x) acts on the degree-8
rational normal curve by its eighth symmetric power:

    Sym^8(mu) nu(x) = (gamma + delta x)^8 nu(mu(x)),   nu(x) = [1, x, ..., x^8].

The transition matrix between two Veronese-normalized frames, S = A_b T C_a^T
(never inverting the badly conditioned C_a), must be proportional to Sym^8 of
the gluing Moebius. Recover mu from the sampled action of S on the curve --
many samples with the full Sym^8 consistency residual, never three noisy
anchor points.
"""

import numpy as np


def veronese(x, d=8):
    return np.array([x**k for k in range(d + 1)], dtype=np.complex128)


def poly_mul(a, b, d):
    out = np.zeros(d + 1, dtype=np.complex128)
    for i, ai in enumerate(a):
        if abs(ai) == 0:
            continue
        for j, bj in enumerate(b):
            if i + j <= d:
                out[i + j] += ai * bj
    return out


def linear_poly(p, q, d):
    out = np.zeros(d + 1, dtype=np.complex128)
    out[0] = p
    if d >= 1:
        out[1] = q
    return out


def poly_pow_linear(p, q, n, d):
    out = np.zeros(d + 1, dtype=np.complex128)
    out[0] = 1.0
    base = linear_poly(p, q, d)

    for _ in range(n):
        out = poly_mul(out, base, d)

    return out


def sym_power_mobius(alpha, beta, gamma, delta, d=8):
    """
    Matrix S such that

        S nu(x) = (gamma + delta*x)^d * nu((alpha + beta*x)/(gamma + delta*x)).

    Row i is coefficients of
        (alpha + beta*x)^i (gamma + delta*x)^(d-i).
    """
    S = np.zeros((d + 1, d + 1), dtype=np.complex128)

    for i in range(d + 1):
        p1 = poly_pow_linear(alpha, beta, i, d)
        p2 = poly_pow_linear(gamma, delta, d - i, d)
        row = poly_mul(p1, p2, d)
        S[i, :] = row

    return S


def recover_x_from_veronese_vector(v, d=8, eps=1e-300):
    """
    Given v ~ lambda * nu(x), recover x by least squares:
        v_{i+1} ~ x v_i.
    Also checks infinity chart.
    """
    v = np.asarray(v, dtype=np.complex128)

    # Finite x chart.
    denom = np.vdot(v[:d], v[:d]).real
    if denom > eps:
        x = np.vdot(v[:d], v[1:]) / denom
        powers = veronese(x, d)
        lam = np.vdot(powers, v) / max(np.vdot(powers, powers), eps)
        res = np.linalg.norm(v - lam * powers) / max(np.linalg.norm(v), eps)
    else:
        x = None
        res = np.inf

    # Infinity chart using y=1/x.
    denom_y = np.vdot(v[1:], v[1:]).real
    if denom_y > eps:
        y = np.vdot(v[1:], v[:d]) / denom_y
        powers_y = np.array([y ** (d - i) for i in range(d + 1)], dtype=np.complex128)
        lam_y = np.vdot(powers_y, v) / max(np.vdot(powers_y, powers_y), eps)
        res_y = np.linalg.norm(v - lam_y * powers_y) / max(np.linalg.norm(v), eps)
    else:
        y = None
        res_y = np.inf

    if res <= res_y:
        return {"chart": "finite", "x": x, "residual": float(res)}

    return {
        "chart": "infinity",
        "x": np.inf if y == 0 else 1 / y,
        "y": y,
        "residual": float(res_y),
    }


def fit_mobius_from_pairs(xs, ys, weights=None):
    """
    Fit y = (alpha + beta*x)/(gamma + delta*x).

    Linear equation:
        alpha + beta*x - y*gamma - y*x*delta = 0.

    Unknown vector:
        [alpha, beta, gamma, delta].
    """
    xs = np.asarray(xs, dtype=np.complex128)
    ys = np.asarray(ys, dtype=np.complex128)

    rows = []
    for x, y in zip(xs, ys):
        rows.append([1.0, x, -y, -y * x])

    A = np.asarray(rows, dtype=np.complex128)

    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=np.float64))
        A = A * w[:, None]

    _, s, vh = np.linalg.svd(A, full_matrices=False)
    v = vh[-1, :].conj()

    alpha, beta, gamma, delta = v

    # Normalize determinant to 1 projectively when possible.
    det = alpha * delta - beta * gamma
    if abs(det) > 0:
        scale = np.sqrt(det)
        alpha, beta, gamma, delta = alpha / scale, beta / scale, gamma / scale, delta / scale

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "singular_values": s,
        "gap": float(s[-2] / s[-1]) if len(s) >= 2 and s[-1] != 0 else None,
    }


def sample_sym8_transition(S, xs, d=8):
    """
    Given a projective Sym^8 transition matrix S, sample its induced map on x.
    """
    ys = []
    residuals = []

    for x in xs:
        v = S @ veronese(x, d)
        rec = recover_x_from_veronese_vector(v, d=d)
        ys.append(rec["x"])
        residuals.append(rec["residual"])

    return np.asarray(ys, dtype=np.complex128), np.asarray(residuals)


def fit_mobius_from_sym8(S, xs=None, d=8):
    """
    Recover the Moebius map represented by a noisy Sym^8 matrix S
    by sampling the induced action on the rational normal curve.
    """
    if xs is None:
        # Avoid too-large values and avoid only real samples.
        xs = []
        for r in [0.0, 0.15, 0.3, 0.55, 0.8]:
            if r == 0:
                xs.append(0.0)
            else:
                for k in range(16):
                    theta = 2 * np.pi * (k + 0.173) / 16
                    xs.append(r * np.exp(1j * theta))
        xs = np.asarray(xs, dtype=np.complex128)

    ys, res = sample_sym8_transition(S, xs, d=d)

    good = np.isfinite(ys) & (res < np.percentile(res, 80))
    weights = 1.0 / np.maximum(res[good], 1e-14)

    mob = fit_mobius_from_pairs(xs[good], ys[good], weights=weights)
    mob["sample_residual_median"] = float(np.median(res))
    mob["sample_residual_max"] = float(np.max(res))
    mob["num_used"] = int(np.sum(good))

    return mob


def sym8_projective_residual(S, mob, d=8):
    """
    Compare noisy S to lambda * Sym^8(mob).
    """
    S0 = sym_power_mobius(
        mob["alpha"], mob["beta"], mob["gamma"], mob["delta"], d=d
    )

    lam = np.vdot(S0.reshape(-1), S.reshape(-1)) / max(
        np.vdot(S0.reshape(-1), S0.reshape(-1)), 1e-300
    )

    res = np.linalg.norm(S - lam * S0) / max(np.linalg.norm(S), 1e-300)

    return {
        "lambda": lam,
        "relative_residual": float(res),
    }
