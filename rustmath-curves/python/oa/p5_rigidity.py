"""Protocol step 7 -- the RIGIDITY TEST.

Build the SQUARE stratum system in the fixed Y-gauge (r1 -> inf baked into Y,
the 2nd 12-point p pinned at y2s inside W12, the origin double-zero pinned by a
single gauge row) and run Newton from the WS=30 seed p5_gn3.

  26 complex unknowns:  a[0..7]  (A double-zero roots, a[0] gauge-pinned)
                        r[0..3]  (R quintuple-pole roots)
                        B[0..7]  (B simple-zero coeffs, B monic deg 8)
                        St[0..4] (Q's degree-4 tail = lambda*S, NOT monic)
                        c        (P-Q = c*(Y-y2s)^12)
  26 residuals:         25 structural  (P - Q - c*W12)_k = 0, k=0..24
                        1  gauge        a[0] - y0s = 0

Riemann-Hurwitz rigidity => the true map is an ISOLATED REGULAR zero.  If the
seed is in its basin, Newton converges QUADRATICALLY and the exact map falls out
(Phase I and Phase III collapse into one).  Divergence => basin still off.
"""
import numpy as np, mpmath as mp, sys, os
from math import comb
sys.path.insert(0, os.path.dirname(__file__))

SW = "/home/john/sweep_2_12_5/"
DPS = int(os.environ.get("DPS", "50"))
NEWT = int(os.environ.get("NEWT", "14"))
mp.mp.dps = DPS

# ---- fixed gauge scalars (fp64 seeds -> mp) --------------------------------
r1  = complex(0.47182600647013, -0.10561240463346)   # 12-point -> infinity
r2  = complex(-0.32224249915043, -0.38542142936930)  # 12-point -> y2s
rc  = complex(-0.056491567434, -0.311094205741)
rc9 = complex(1.152891104470, 0.579237023791)
a_ex = [complex(0.2120091233, -0.5634590964), complex(-0.7190508461, -0.4168884505)]
sc = 8.0
Yf   = lambda X: (1.0/(complex(X)-r1))/sc
y2s  = 1.0/(r2-r1)/sc
y0s  = (-1.0/r1)/sc
ycs, yc9s = Yf(rc), Yf(rc9)
aexY = [complex(y0s), Yf(a_ex[0]), Yf(a_ex[1])]

# ---- seed: WS=30 roots + recovered linear part -----------------------------
roots = np.load(SW + "p5_gn3.npz")['roots']
D = np.load(SW + "p5_samples6.npz")
X, PH, REG = D['X'], D['PHI'], D['region']
WSTRAT = 30.0


def pfr(rr):
    P = np.array([1.0 + 0j])
    for r in rr:
        P = np.convolve(P, np.array([-r, 1.0 + 0j]))
    return P


def irls_weights(Pv, Qv, ph, eps=1e-300):
    return 1.0 / np.maximum(np.abs(Pv) + np.abs(ph * Qv), eps)


def recover_lin(roots):
    """Structural projection: given the A- and R-roots, solve the 25 identity
    equations (P - Q - c*W12)_k = 0 in the 14 linear unknowns (B0..7, St0..4, c)
    by least squares.  This minimises exactly what the Newton residual measures,
    so it is the tightest possible structural seed for these roots."""
    W12 = pfr([complex(y2s)] * 12)              # length 13
    A = pfr(aexY + list(roots[:5]))             # deg 8
    A2 = np.convolve(A, A)                       # deg 16 (len 17)
    R = pfr([ycs, yc9s, roots[5], roots[6]])     # deg 4
    R5 = np.convolve(np.convolve(np.convolve(R, R), np.convolve(R, R)), R)  # deg 20
    M = np.zeros((25, 14), complex)
    b = np.zeros(25, complex)
    for k in range(25):
        for j in range(8):                       # B0..7
            if 0 <= k - j <= 16:
                M[k, j] = A2[k - j]
        b[k] = -(A2[k - 8] if 0 <= k - 8 <= 16 else 0)   # B8 = 1
        for js in range(5):                      # St0..4
            if 0 <= k - js <= 20:
                M[k, 8 + js] = -R5[k - js]
        M[k, 13] = -(W12[k] if k <= 12 else 0)   # c
    v, res, *_ = np.linalg.lstsq(M, b, rcond=None)
    struct = M @ v - b
    print("structural-projection seed: ||resid|| = %.4e" % np.linalg.norm(struct), flush=True)
    return v  # B0..7, St0..4, c


lin = recover_lin(roots)
print("recovered linear seed: |c| = %.3e  lambda(=St4) = %s" %
      (abs(lin[13]), np.round(lin[12], 6)), flush=True)

seed = (list(aexY) + list(roots[:5])            # a[0..7]
        + [ycs, yc9s, roots[5], roots[6]]        # r[0..3]
        + list(lin[:8])                          # B[0..7]
        + list(lin[8:13])                        # St[0..4]
        + [lin[13]])                             # c
assert len(seed) == 26

# ---- mpmath square system --------------------------------------------------
y0m = mp.mpc(y0s)
y2m = mp.mpc(y2s)


def mpconv(a, b):
    out = [mp.mpc(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def poly_from_roots(rs):
    p = [mp.mpc(1)]
    for r in rs:
        p = mpconv(p, [-r, mp.mpc(1)])
    return p


W12m = poly_from_roots([y2m] * 12)  # 13 coeffs, degree 12


def residual(th):
    a = th[0:8]
    r = th[8:12]
    B = th[12:20]
    St = th[20:25]
    c = th[25]
    A = poly_from_roots(a)
    A2 = mpconv(A, A)
    P = mpconv(A2, list(B) + [mp.mpc(1)])          # monic B, deg 24 (25 coeffs)
    R = poly_from_roots(r)
    R2 = mpconv(R, R)
    R5 = mpconv(mpconv(R2, R2), R)                 # deg 20
    Q = mpconv(R5, list(St))                       # deg 24 (25 coeffs)
    res = []
    for k in range(25):
        wk = W12m[k] if k < 13 else mp.mpc(0)
        res.append(P[k] - Q[k] - c * wk)
    res.append(a[0] - y0m)                         # gauge row
    return res


def rnorm(R):
    return mp.sqrt(sum((abs(x)) ** 2 for x in R))


def jacobian(th, h):
    J = mp.matrix(26, 26)
    for k in range(26):
        thp = th[:]; thp[k] = thp[k] + h
        thm = th[:]; thm[k] = thm[k] - h
        Rp = residual(thp); Rm = residual(thm)
        for i in range(26):
            J[i, k] = (Rp[i] - Rm[i]) / (2 * h)
    return J


def lm(th, iters, hexp):
    """Damped Gauss-Newton (complex LM): (J^H J + mu I) dx = -J^H R.
    mu adapts; when small it IS Newton and converges quadratically."""
    th = [mp.mpc(x) for x in th]
    h = mp.mpf(10) ** hexp
    mu = mp.mpf('1e-3')
    hist = []
    R0 = residual(th)
    nrm = rnorm(R0)
    for it in range(iters):
        J = jacobian(th, h)
        Jh = J.H                       # conjugate transpose
        g = Jh * mp.matrix(R0)         # J^H R
        JhJ = Jh * J
        accepted = False
        for _ in range(30):
            A = JhJ + mu * mp.eye(26)
            try:
                dx = mp.lu_solve(A, g)
            except Exception:
                mu *= 8; continue
            thn = [th[i] - dx[i] for i in range(26)]
            Rn = residual(thn)
            nn = rnorm(Rn)
            if nn < nrm:
                th, R0, nrm = thn, Rn, nn
                mu = max(mu / 3, mp.mpf('1e-40'))
                accepted = True
                break
            mu *= 5
        stepn = rnorm(list(dx))
        print("  it %2d  |R| = %s   |dx| = %s   mu = %s" %
              (it, mp.nstr(nrm, 6), mp.nstr(stepn, 4), mp.nstr(mu, 2)), flush=True)
        hist.append(nrm)
        if not accepted:
            print("  (no decrease -- LM stalled)", flush=True)
            break
        if nrm < mp.mpf(10) ** (-(DPS - 8)):
            print("  converged.", flush=True)
            break
    return th, hist


print("\n=== RIGIDITY LM/NEWTON  (dps=%d) ===" % DPS, flush=True)
hexp = -20 if DPS <= 30 else -(DPS // 2)
th, hist = lm(seed, NEWT, hexp)

# ---- rigidity certificate: singular spectrum of the Jacobian ---------------
def analyse_jacobian(thv):
    Jf = jacobian([mp.mpc(x) for x in thv], mp.mpf(10) ** hexp)
    U, S, V = mp.svd(Jf)                       # full SVD (V is conj-transposed)
    svl = [mp.mpf(S[i]) for i in range(26)]
    order = sorted(range(26), key=lambda i: svl[i])
    print("\nsmallest 8 singular values:")
    for i in order[:8]:
        print("   sigma[%2d] = %s" % (i, mp.nstr(svl[i], 6)))
    print("   sigma_max = %s   cond = %s" %
          (mp.nstr(max(svl), 5), mp.nstr(max(svl) / min(svl), 4)))
    # null-vector structure: rows of V for the smallest sigma
    imin = order[0]
    groups = [("A-roots", 0, 8), ("R-roots", 8, 12), ("B", 12, 20), ("St", 20, 25), ("c", 25, 26)]
    print("null-vector (smallest sigma) energy by block:")
    for name, lo, hi in groups:
        e = mp.sqrt(sum(abs(V[imin, j]) ** 2 for j in range(lo, hi)))
        print("   %-8s %s" % (name, mp.nstr(e, 4)))
    return svl


try:
    analyse_jacobian(th)
except Exception as e:
    print("svd failed:", repr(e))

# root-separation audit (merging roots => genuine variety singularity)
def sep_report(thv):
    a = [complex(x) for x in thv[0:8]]
    r = [complex(x) for x in thv[8:12]]
    def mind(pts):
        return min(abs(pts[i] - pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts)))
    print("min |a_i - a_j| (Y) = %.3e   min |r_i - r_j| (Y) = %.3e" % (mind(a), mind(r)))
    print("min |a_i - r_j| (Y) = %.3e" % min(abs(ai - rj) for ai in a for rj in r))
sep_report(th)

# ---- report ----------------------------------------------------------------
final = residual(th)
print("\nfinal |R| = %s" % mp.nstr(rnorm(final), 6))
print("\nA-roots (double zeros)   X = r1 + 1/(y*sc):")
lab_a = ["a0(origin)", "a1(dd)", "a2(dd)", "a3", "a4", "a5", "a6", "a7"]
r1m = mp.mpc(r1)
for l, y in zip(lab_a, th[0:8]):
    Xr = r1m + 1 / (y * sc)
    print("  %-11s X = %s" % (l, mp.nstr(Xr, 12)))
print("R-roots (quintuple poles) X = r1 + 1/(y*sc):")
lab_r = ["r0(c-pole)", "r1(c9-pole)", "r2", "r3"]
for l, y in zip(lab_r, th[8:12]):
    Xr = r1m + 1 / (y * sc)
    print("  %-11s X = %s" % (l, mp.nstr(Xr, 12)))
print("scale  lambda(St4) = %s" % mp.nstr(th[24], 10))
print("c = %s" % mp.nstr(th[25], 10))

# save
out = np.array([complex(x) for x in th], dtype=complex)
np.savez(SW + "p5_rigidity.npz", theta=out,
         final_norm=float(rnorm(final)), hist=[float(h) for h in hist])
print("\nsaved p5_rigidity.npz")
