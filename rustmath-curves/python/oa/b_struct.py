"""B: structured stratum solve in the REAL gauge (sigma = conj, 12-points at +-i).

Structure: P = A^2 Bs (A, Bs monic real deg 8), Q = R^5 S (R monic real deg 4, S real deg 4
with S[4] = lambda), P - Q = c*(xi^2+1)^12, c real.  Unknowns theta (26 real):
a[0..7], bs[0..7], r[0..3], s[0..4], c.  Square system: 25 coefficient equations + 1
rotation-gauge row (pin a[7]).  Solvers: real LM with data rows (basin), square Newton
(analytic J), then the mp ladder.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import b_config as BC
from math import comb

W25 = np.zeros(25)
for j in range(13):
    W25[2*j] = comb(12, j)          # (xi^2+1)^12


def poly_from(coefs_monic):
    return np.concatenate([coefs_monic, [1.0]])


def build(th):
    a = poly_from(th[0:8])          # deg 8 monic
    bs = poly_from(th[8:16])
    r = poly_from(th[16:20])        # deg 4 monic
    s = th[20:25]                   # deg 4, s[4] = lambda
    c = th[25]
    A2 = np.convolve(a, a)
    P = np.convolve(A2, bs)         # deg 24 monic
    R2 = np.convolve(r, r)
    R4 = np.convolve(R2, R2)
    R5 = np.convolve(R4, r)         # deg 20 monic
    Q = np.convolve(R5, s)          # deg 24, lead lambda
    return a, bs, r, s, c, A2, R4, R5, P, Q


def residual(th, gauge_a7):
    a, bs, r, s, c, A2, R4, R5, P, Q = build(th)
    F = np.empty(26)
    F[:25] = P - Q - c*W25
    F[25] = th[7] - gauge_a7
    return F


def jacobian(th):
    a, bs, r, s, c, A2, R4, R5, P, Q = build(th)
    J = np.zeros((26, 26))
    tw = 2*np.convolve(a, bs)                   # dP/da_j = 2*A*Bs*xi^j (deg 16, len 17)
    for j in range(8):
        J[j:j + 17, j] = tw
    for j in range(8):                          # dP/dbs_j = A^2 xi^j
        J[j:j + 17, 8 + j] = A2
    fiveR4S = 5*np.convolve(R4, s)              # dQ/dr_j = 5 R^4 S xi^j (deg 20, len 21)
    for j in range(4):
        J[j:j + 21, 16 + j] -= fiveR4S
    for j in range(5):                          # dQ/ds_j = R^5 xi^j
        J[j:j + 21, 20 + j] -= R5
    J[:25, 25] = -W25
    Jg = np.zeros(26)
    Jg[7] = 1.0
    return np.vstack([J[:25], Jg])


def newton(th0, gauge_a7, iters=40, tol=1e-13, verbose=True):
    th = th0.copy()
    for it in range(iters):
        F = residual(th, gauge_a7)
        J = jacobian(th)
        nrm = np.linalg.norm(F)
        # equilibrate rows
        rs = 1.0/(1 + np.abs(build(th)[8][:25]) + np.abs(build(th)[9][:25]))
        rs = np.concatenate([rs, [1.0]])
        Je = J*rs[:, None]
        Fe = F*rs
        try:
            dx = np.linalg.solve(Je, Fe)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(Je, Fe, rcond=None)
        t = 1.0
        for _ in range(30):
            thn = th - t*dx
            if np.linalg.norm(residual(thn, gauge_a7)*rs) < nrm:
                th = thn
                break
            t /= 2
        else:
            if verbose:
                print(f"  it {it}: stalled at |F| = {nrm:.3e}")
            break
        if verbose:
            print(f"  it {it}: |F| = {nrm:.3e}  t={t:.3f}")
        if nrm < tol:
            break
    return th
