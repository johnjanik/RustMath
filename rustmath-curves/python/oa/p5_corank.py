"""Clean corank of the factored square system via the ANALYTIC Jacobian.

Finite differences cannot resolve sigma_24/sigma_25 (they sit at the FD noise
floor).  Here every Jacobian column is an exact polynomial derivative, so the
singular spectrum is trustworthy and the true corank of the rigidity system at
the seed is decided.  This tells us whether the isolated map is a SINGULAR
solution (small corank -> deflate) or the stratum is under-determined.
"""
import numpy as np, mpmath as mp, sys, os
sys.path.insert(0, os.path.dirname(__file__))
mp.mp.dps = int(os.environ.get("DPS", "40"))

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
th = (list(aexY) + list(roots[:5]) + [ycs, yc9s, roots[5], roots[6]]
      + list(lin[:8]) + list(lin[8:13]) + [lin[13]])

# ---- mp polynomial helpers -------------------------------------------------
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


def col(poly):
    """25-vector (pad/truncate to degree 24) + trailing 0 gauge entry -> length 26."""
    v = [mp.mpc(0)] * 26
    for k in range(min(len(poly), 25)):
        v[k] = poly[k]
    return v


a = [mp.mpc(x) for x in th[0:8]]
r = [mp.mpc(x) for x in th[8:12]]
B = [mp.mpc(x) for x in th[12:20]]
St = [mp.mpc(x) for x in th[20:25]]
c = mp.mpc(th[25])
y2m = mp.mpc(y2s); y0m = mp.mpc(y0s)
W12 = prod_roots([y2m] * 12)

A = prod_roots(a); A2 = conv(A, A); Bf = B + [mp.mpc(1)]
P = conv(A2, Bf)
R = prod_roots(r); R2 = conv(R, R); R4 = conv(R2, R2); R5 = conv(R4, R)
Q = conv(R5, St)

# residual
res = [mp.mpc(0)] * 26
for k in range(25):
    wk = W12[k] if k < 13 else mp.mpc(0)
    res[k] = P[k] - Q[k] - c * wk
res[25] = a[0] - y0m
rnorm = mp.sqrt(sum(abs(x) ** 2 for x in res))
print("seed |R| = %s" % mp.nstr(rnorm, 6))

# ---- analytic Jacobian (26 x 26) -------------------------------------------
J = mp.matrix(26, 26)


def ahat(i):
    return prod_roots([a[j] for j in range(8) if j != i])


def rhat(i):
    return prod_roots([r[j] for j in range(4) if j != i])


for i in range(8):                                   # d/da_i : dP = -2 * Ahat_i * A * B
    d = conv(conv(ahat(i), A), Bf)
    cv = col([-2 * x for x in d])
    if i == 0:
        cv[25] = mp.mpc(1)                           # gauge row d(a0-y0)/da0
    for k in range(26):
        J[k, i] = cv[k]
for i in range(4):                                   # d/dr_i : -dQ = +5 * Rhat_i * R4 * St
    d = conv(conv(rhat(i), R4), St)
    cv = col([5 * x for x in d])
    for k in range(26):
        J[k, 8 + i] = cv[k]
for j in range(8):                                   # d/dB_j : A2 * X^j
    cv = col([mp.mpc(0)] * j + list(A2))
    for k in range(26):
        J[k, 12 + j] = cv[k]
for j in range(5):                                   # d/dSt_j : -R5 * X^j
    cv = col([mp.mpc(0)] * j + [-x for x in R5])
    for k in range(26):
        J[k, 20 + j] = cv[k]
cv = col([-x for x in W12])                          # d/dc : -W12
for k in range(26):
    J[k, 25] = cv[k]

# ---- RAW singular spectrum -------------------------------------------------
U, S, V = mp.svd(J)
svl = sorted([mp.mpf(S[i]) for i in range(26)])
print("\nRAW singular spectrum (smallest 12):")
for i in range(12):
    print("   %s" % mp.nstr(svl[i], 6))
print("sigma_max = %s  cond = %s" % (mp.nstr(svl[-1], 6), mp.nstr(svl[-1] / svl[0], 4)))

# ---- ROW-EQUILIBRATED spectrum (the honest conditioning) -------------------
# residual coeff k has natural scale |P_k|+|Q_k|; scale row k by its inverse.
drow = [mp.mpf(1) / (abs(P[k]) + abs(Q[k]) + mp.mpf(10) ** (-mp.mp.dps)) for k in range(25)] + [mp.mpf(1)]
Je = mp.matrix(26, 26)
for i in range(26):
    for j in range(26):
        Je[i, j] = drow[i] * J[i, j]
# column scale by parameter magnitude too (Jacobi equilibration)
dcol = [mp.mpf(1)] * 26
for j in range(26):
    nrm = mp.sqrt(sum(abs(Je[i, j]) ** 2 for i in range(26)))
    dcol[j] = mp.mpf(1) / nrm if nrm > 0 else mp.mpf(1)
    for i in range(26):
        Je[i, j] *= dcol[j]
Ue, Se, Ve = mp.svd(Je)
sve = sorted([mp.mpf(Se[i]) for i in range(26)])
print("\nEQUILIBRATED singular spectrum (smallest 12):")
for i in range(12):
    print("   %s" % mp.nstr(sve[i], 6))
print("sigma_max = %s  cond = %s" % (mp.nstr(sve[-1], 6), mp.nstr(sve[-1] / sve[0], 4)))
tol = sve[-1] * mp.mpf(10) ** (-(mp.mp.dps - 6))
corank = sum(1 for s in sve if s < tol)
print("numeric tol = %s  ->  equilibrated corank = %d" % (mp.nstr(tol, 3), corank))

# null-vector block energy for the smallest few
order = sorted(range(26), key=lambda i: mp.mpf(S[i]))
groups = [("A-roots", 0, 8), ("R-roots", 8, 12), ("B", 12, 20), ("St", 20, 25), ("c", 25, 26)]
for rank_i in range(3):
    im = order[rank_i]
    print("null-vec sigma=%s block energy:" % mp.nstr(mp.mpf(S[im]), 4))
    for name, lo, hi in groups:
        e = mp.sqrt(sum(abs(V[im, j]) ** 2 for j in range(lo, hi)))
        print("   %-8s %s" % (name, mp.nstr(e, 4)))
