"""P2: Veronese normalizer -- the global-coordinate extractor.

dim S_4 = 9, so the basepoint-free weight-4 form map z -> [f_0:...:f_8] is the complete series
H^0(P^1, O(8)): the degree-8 rational normal curve [1:X:...:X^8] up to a projective basis change A.
Build A from the ORDER-2 jets so G = A F has G_i = u^i + O(u^9); the same A applied to form values
from any other chart yields G ~ [1:X:...:X^8], and X is read off the Veronese ratios.  This is the
global Hauptmodul, buildable now from the data we already have -- no projective search.

Self-check: on the order-2 samples, recovered X must equal the local coordinate u.
Structural check: the Veronese RESIDUAL measures how close the form map is to the degree-8 rational
normal curve -- small => the dim-9 / complete-series assumption holds.
"""
import numpy as np

def coefficient_matrix(forms_u, degree=8):
    """C[n,j] = coeff of u^n in form f_j, for n,j = 0..degree."""
    C = np.zeros((degree + 1, degree + 1), dtype=np.complex128)
    for j, f in enumerate(forms_u[:degree + 1]):
        for n in range(degree + 1):
            C[n, j] = f[n] if n < len(f) else 0.0
    return C

def build_veronese_normalizer(forms_u, degree=8):
    """A = (C^T)^{-1} so that G = A F has G_i(u) = u^i + O(u^{degree+1})."""
    C = coefficient_matrix(forms_u, degree=degree)
    A = np.linalg.inv(C.T)
    err = np.linalg.norm(A @ C.T - np.eye(degree + 1))
    return A, {"degree": degree, "jet_condition_error": float(err), "cond_C": float(np.linalg.cond(C))}

def apply_normalizer(A, F_values):
    F = np.asarray(F_values, dtype=np.complex128)
    return (A @ F) if F.ndim == 1 else (A @ F.T).T

def recover_x_from_veronese_point(G, eps=1e-300):
    """G ~ lambda [1,x,...,x^d]; recover x, testing both the affine and the infinity chart."""
    G = np.asarray(G, dtype=np.complex128); d = len(G) - 1
    def fit(x, powers):
        lam = np.vdot(powers, G) / max(np.vdot(powers, powers), eps)
        return lam, np.linalg.norm(G - lam * powers) / max(np.linalg.norm(G), eps)
    res_x = np.inf; x = None
    if np.sum(np.abs(G[:d])**2) > eps:
        x = np.sum(np.conj(G[:d]) * G[1:]) / np.sum(np.abs(G[:d])**2)
        lam, res_x = fit(x, np.array([x**i for i in range(d + 1)]))
    res_y = np.inf; y = None
    if np.sum(np.abs(G[1:])**2) > eps:
        y = np.sum(np.conj(G[1:]) * G[:d]) / np.sum(np.abs(G[1:])**2)
        lam, res_y = fit(y, np.array([y**(d - i) for i in range(d + 1)]))
    if res_x <= res_y:
        return {"chart": "x", "x": x, "residual": float(res_x)}
    return {"chart": "infty", "x": (np.inf if y == 0 else 1 / y), "residual": float(res_y)}

def recover_x_values(A, F_values):
    G = apply_normalizer(A, F_values)
    return [recover_x_from_veronese_point(g) for g in G], G

if __name__ == "__main__":
    import sys, os, math
    sys.path.insert(0, os.path.dirname(__file__))
    import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
    import mpmath as mp
    mp.mp.dps = 35
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2950
    vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]; Lu = mapkit.Lu
    dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
    G = jet_dd.dd_gram(C, n_slices=16)
    B = jet_dd.solve_dd_refine(G, vals, 1e-12, jt.tail_weights(N), n_slices=18, iters=8)[0]
    X, ech = jr.hauptmodul_mp(B, dim, mp.mpf('1e-8'))
    forms = [[complex(ech[j][n]) for n in range(Lu)] for j in sorted(ech)]   # 9 forms, val 0..8

    A, info = build_veronese_normalizer(forms, degree=8)
    print(f"N={N}  Veronese normalizer:  jet_condition_error={info['jet_condition_error']:.2e}  cond(C)={info['cond_C']:.2e}")
    ev = lambda coef, u: np.polyval(np.asarray(coef)[::-1], u)
    print("  self-check on order-2 samples  (recovered X should equal local u):")
    print(f"  {'u':>18} {'recovered X':>26} {'|X-u|':>10} {'ver.resid':>10}")
    for r in (0.05, 0.1, 0.2, 0.4, 0.6):
        u = complex(r, 0.3 * r)
        Fu = np.array([ev(f, u) for f in forms])
        rec = recover_x_from_veronese_point(A @ Fu)
        xr = rec["x"]
        print(f"  {u.real:+.3f}{u.imag:+.3f}i   {xr.real:+.6f}{xr.imag:+.6f}i   {abs(xr-u):.2e} {rec['residual']:.2e}")
