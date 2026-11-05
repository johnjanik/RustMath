"""Rigidity Newton, done right: ANALYTIC Jacobian + row/column equilibration +
line search, from the structural-projection seed.

The factored square system is full-rank (corank 0) but ill-conditioned ~1e10 by
two mild soft modes of the P=A^2B / Q=R^5 St parameterization, plus a cosmetic
|Y|^k row-scale spread.  Equilibrating both and solving J dx = R directly (NOT
the normal equations, which square the condition number and made LM crawl) gives
a clean quadratically-convergent Newton.  We run a precision ladder and read off
the exact map coefficients.

env: DPS (working precision), ITERS.
"""
import numpy as np, mpmath as mp, sys, os
sys.path.insert(0, os.path.dirname(__file__))
DPS = int(os.environ.get("DPS", "60"))
ITERS = int(os.environ.get("ITERS", "40"))
mp.mp.dps = DPS

SW = "/home/john/sweep_2_12_5/"
r1  = complex(0.47182600647013, -0.10561240463346)
r2  = complex(-0.32224249915043, -0.38542142936930)
rc  = complex(-0.056491567434, -0.311094205741)
rc9 = complex(1.152891104470, 0.579237023791)
a_ex = [complex(0.2120091233, -0.5634590964), complex(-0.7190508461, -0.4168884505)]
sc = 8.0
Yf   = lambda X: (1.0/(complex(X)-r1))/sc
y2s  = 1.0/(r2-r1)/sc
y0s  = (-1.0/r1)/sc
ycs, yc9s = Yf(rc), Yf(rc9)
aexY = [complex(y0s), Yf(a_ex[0]), Yf(a_ex[1])]
roots = np.load(SW + "p5_gn3.npz")['roots']


def pfr(rr):
    P = np.array([1.0 + 0j])
    for r in rr:
        P = np.convolve(P, np.array([-r, 1.0 + 0j]))
    return P


def recover_lin(roots):
    W12 = pfr([complex(y2s)] * 12)
    A = pfr(aexY + list(roots[:5])); A2 = np.convolve(A, A)
    R = pfr([ycs, yc9s, roots[5], roots[6]])
    R5 = np.convolve(np.convolve(np.convolve(R, R), np.convolve(R, R)), R)
    M = np.zeros((25, 14), complex); b = np.zeros(25, complex)
    for k in range(25):
        for j in range(8):
            if 0 <= k - j <= 16:
                M[k, j] = A2[k - j]
        b[k] = -(A2[k - 8] if 0 <= k - 8 <= 16 else 0)
        for js in range(5):
            if 0 <= k - js <= 20:
                M[k, 8 + js] = -R5[k - js]
        M[k, 13] = -(W12[k] if k <= 12 else 0)
    v, *_ = np.linalg.lstsq(M, b, rcond=None)
    return v


lin = recover_lin(roots)
seed = (list(aexY) + list(roots[:5]) + [ycs, yc9s, roots[5], roots[6]]
        + list(lin[:8]) + list(lin[8:13]) + [lin[13]])

# ---- mp polynomial helpers -------------------------------------------------
TINY = mp.mpf(10) ** (-DPS)
y2m = mp.mpc(y2s); y0m = mp.mpc(y0s); r1m = mp.mpc(r1)


def conv(a, b):
    out = [mp.mpc(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def prod_roots(rs):
    p = [mp.mpc(1)]
    for r in rs:
        p = conv(p, [-r, mp.mpc(1)])
    return p


W12 = prod_roots([y2m] * 12)


def unpack(th):
    return (list(th[0:8]), list(th[8:12]), list(th[12:20]), list(th[20:25]), th[25])


def build_PQ(th):
    a, r, B, St, c = unpack(th)
    A = prod_roots(a); A2 = conv(A, A); Bf = B + [mp.mpc(1)]
    P = conv(A2, Bf)
    R = prod_roots(r); R2 = conv(R, R); R4 = conv(R2, R2); R5 = conv(R4, R)
    Q = conv(R5, St)
    return P, Q, A, A2, Bf, R, R4, R5, a, r, B, St, c


def residual(th):
    P, Q, *_ = build_PQ(th)
    res = [mp.mpc(0)] * 26
    for k in range(25):
        wk = W12[k] if k < 13 else mp.mpc(0)
        cval = th[25]
        res[k] = P[k] - Q[k] - cval * wk
    res[25] = th[0] - y0m
    return res, P, Q


def col(poly):
    v = [mp.mpc(0)] * 26
    for k in range(min(len(poly), 25)):
        v[k] = poly[k]
    return v


def build_jac(th):
    P, Q, A, A2, Bf, R, R4, R5, a, r, B, St, c = build_PQ(th)
    J = mp.matrix(26, 26)
    for i in range(8):
        ah = prod_roots([a[j] for j in range(8) if j != i])
        d = conv(conv(ah, A), Bf)
        cv = col([-2 * x for x in d])
        if i == 0:
            cv[25] = mp.mpc(1)
        for k in range(26):
            J[k, i] = cv[k]
    for i in range(4):
        rh = prod_roots([r[j] for j in range(4) if j != i])
        d = conv(conv(rh, R4), St)
        cv = col([5 * x for x in d])
        for k in range(26):
            J[k, 8 + i] = cv[k]
    for j in range(8):
        cv = col([mp.mpc(0)] * j + list(A2))
        for k in range(26):
            J[k, 12 + j] = cv[k]
    for j in range(5):
        cv = col([mp.mpc(0)] * j + [-x for x in R5])
        for k in range(26):
            J[k, 20 + j] = cv[k]
    cv = col([-x for x in W12])
    for k in range(26):
        J[k, 25] = cv[k]
    return J, P, Q


def erow(P, Q):
    return [mp.mpf(1) / (abs(P[k]) + abs(Q[k]) + TINY) for k in range(25)] + [mp.mpf(1)]


def enorm(res, drow):
    return mp.sqrt(sum((abs(drow[k] * res[k])) ** 2 for k in range(26)))


# ---- equilibrated line-search Newton ---------------------------------------
th = [mp.mpc(x) for x in seed]
res, P, Q = residual(th)
drow = erow(P, Q)
er = enorm(res, drow)
print("=== rigidity Newton (dps=%d)  seed rel-|R| = %s ===" % (DPS, mp.nstr(er, 6)), flush=True)

for it in range(ITERS):
    J, P, Q = build_jac(th)
    res, _, _ = residual(th)
    drow = erow(P, Q)
    er = enorm(res, drow)
    # equilibrate: Drow J Dcol u = Drow res ; dx = Dcol u
    Je = mp.matrix(26, 26); be = mp.matrix(26, 1)
    for i in range(26):
        be[i] = drow[i] * res[i]
        for j in range(26):
            Je[i, j] = drow[i] * J[i, j]
    dcol = [mp.mpf(0)] * 26
    for j in range(26):
        nrm = mp.sqrt(sum(abs(Je[i, j]) ** 2 for i in range(26)))
        dcol[j] = mp.mpf(1) / nrm if nrm > 0 else mp.mpf(1)
        for i in range(26):
            Je[i, j] *= dcol[j]
    try:
        u = mp.lu_solve(Je, be)
    except Exception as e:
        print("  lu_solve failed:", e); break
    dx = [dcol[j] * u[j] for j in range(26)]
    # line search on the equilibrated relative residual
    t = mp.mpf(1); ok = False
    for _ in range(40):
        thn = [th[i] - t * dx[i] for i in range(26)]
        resn, Pn, Qn = residual(thn)
        ern = enorm(resn, erow(Pn, Qn))
        if ern < er:
            th, er = thn, ern; ok = True; break
        t /= 2
    print("  it %2d  rel-|R| = %s   t = %s" % (it, mp.nstr(er, 6), mp.nstr(t, 3)), flush=True)
    if not ok:
        print("  (line search stalled)"); break
    if er < mp.mpf(10) ** (-(DPS - 10)):
        print("  converged."); break

# ---- report ----------------------------------------------------------------
res, P, Q = residual(th)
print("\nfinal rel-|R| = %s   abs-|R| = %s" %
      (mp.nstr(enorm(res, erow(P, Q)), 6), mp.nstr(mp.sqrt(sum(abs(x)**2 for x in res)), 6)))
a, r, B, St, c = unpack(th)
print("\nA-roots (double zeros)  X = r1 + 1/(y*sc):")
lab_a = ["a0(origin)", "a1(dd)", "a2(dd)", "a3", "a4", "a5", "a6", "a7"]
for l, y in zip(lab_a, a):
    print("  %-11s X = %s" % (l, mp.nstr(r1m + 1 / (y * sc), 14)))
print("R-roots (quintuple poles):")
for l, y in zip(["r0(c-pole)", "r1(c9-pole)", "r2", "r3"], r):
    print("  %-11s X = %s" % (l, mp.nstr(r1m + 1 / (y * sc), 14)))
print("lambda(St4) = %s" % mp.nstr(St[4], 12))
print("c = %s" % mp.nstr(c, 12))

out = np.array([complex(x) for x in th], dtype=complex)
np.savez(SW + "p5_rigidity3.npz", theta=out, rel_norm=float(enorm(res, erow(P, Q))))
# full-precision theta as strings for the ladder / PSLQ
with open(SW + "p5_rigidity3_theta.txt", "w") as f:
    for x in th:
        f.write(mp.nstr(x.real, DPS) + " " + mp.nstr(x.imag, DPS) + "\n")
print("\nsaved p5_rigidity3.npz + theta.txt")
