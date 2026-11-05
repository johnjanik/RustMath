"""§7: structured degree-24 Belyi-map fit for [2,12,5].  The generic P/Q fit matches phi's series
but overfits form noise -> the fiber over 1 does not collapse to two order-12 points (o12~2.6).
Impose the ramification:  P = A^2 B  (deg A,B = 8),  Q = R^5 S  (deg R,S = 4),  so P/Q automatically
carries the 2^8 / 5^4 structure; seed A,B,R,S by clustering the generic roots, then refine to phi.
Because we now seed from the CORRECT modular forms (real Hauptmodul), this lands on the true
non-degenerate map, unlike the earlier circle-packing factorized solve.

Usage: python3 structured_map.py [N]   (default 2800)"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, physical_selector as ps
from numpy.polynomial.polynomial import polyfromroots
from scipy.optimize import least_squares
import mpmath as mp

SWEEP = "/home/john/sweep_2_12_5"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2800
Lu = mapkit.Lu; phiv = mapkit.phiv; Nfit = 58
cv = lambda z: np.convolve(z[0], z[1]) if isinstance(z, tuple) else z
def pmul(*ps):
    out = ps[0]
    for p in ps[1:]: out = np.convolve(out, p)
    return out

def cluster(roots, mult, count):
    """Greedily pull `count` groups of `mult` nearest roots; return (group centers, leftovers)."""
    R = list(roots); centers = []
    for _ in range(count):
        best = None
        for i in range(len(R)):
            idx = sorted(range(len(R)), key=lambda j: abs(R[j]-R[i]))[:mult]
            sp = max(abs(R[j]-R[i]) for j in idx)
            if best is None or sp < best[0]: best = (sp, idx)
        idx = set(best[1]); centers.append(np.mean([R[j] for j in best[1]]))
        R = [R[j] for j in range(len(R)) if j not in idx]
    return np.array(centers), np.array(R)

def order12_spread(Dcoef):
    """roots of the deg-24 poly D (lowest-first): cluster into 2 groups of 12, rescaled diameter."""
    mp.mp.dps = 30
    cf = [mp.mpc(c.real, c.imag) for c in Dcoef[::-1]]
    while len(cf) > 1 and abs(cf[0]) < 1e-12*max(abs(c) for c in cf): cf = cf[1:]
    rr = [complex(r) for r in mp.polyroots(cf, maxsteps=300, extraprec=250)]
    smed = np.median(np.abs([r for r in rr if abs(r) > 1e-12]))
    R = list(rr); worst = 0.
    for _ in range(2):
        best = None
        for i in range(len(R)):
            idx = sorted(range(len(R)), key=lambda j: abs(R[j]-R[i]))[:12]
            sp = max(abs(R[j]-R[i]) for j in idx)
            if best is None or sp < best[0]: best = (sp, idx)
        worst = max(worst, best[0]/smed); s = set(best[1])
        R = [R[j] for j in range(len(R)) if j not in s]
    return worst

# ---- 1. correct forms -> Hauptmodul -> generic fit ----
dim, Ahi = ps.load_hi(f"{SWEEP}/m_N{N}.bin")
s, Vh = ps.fp64_svd_ascending(Ahi)
Y, _ = ps.sel_band_tail(s, Vh, dim, K=200, power=4)
Xv, ech = mapkit.hauptmodul(Y, dim)
p, q, Xp, sgv = mapkit.fit_generic(Xv, Nfit)
Xp = np.array(Xp)                                  # (25, Lu)
print(f"N={N}: band+tail forms real={mapkit.reality(Xv)[0]:.1e}; generic o12={order12_spread(p-q):.3f}")

# ---- 2. seed A,B,R,S,U,c by clustering the generic roots ----
rP = np.roots(p[::-1]); Ar, Br = cluster(rP, 2, 8)     # 8 double-centers, 8 simple
rQ = np.roots(q[::-1]); Rr, Sr = cluster(rQ, 5, 4)     # 4 quintuple-centers, 4 simple
Dg = np.zeros(25, complex); Dg[:len(p)] += p; Dg[:len(q)] -= q
rD = np.roots(Dg[::-1]); Ur, _ = cluster(rD, 12, 2)    # 2 twelvefold-centers over phi=1
A0 = polyfromroots(Ar).astype(complex)                 # monic deg 8
B0 = polyfromroots(Br).astype(complex) * p[-1]         # match generic P leading coeff
R0 = polyfromroots(Rr).astype(complex)                 # monic deg 4
S0 = polyfromroots(Sr).astype(complex) * q[-1]         # match generic Q leading
U0 = polyfromroots(Ur).astype(complex)                 # monic deg 2
c0 = Dg[-1]

# ---- 3. residual: phi-match (anchor to true non-degenerate map) + P-Q=c U^12 structure ----
def unpack(x):
    z = x[:len(x)//2] + 1j*x[len(x)//2:]
    return (np.append(z[0:8], 1.0), z[8:17], np.append(z[17:21], 1.0), z[21:26],
            np.append(z[26:28], 1.0), z[28])            # A,B,R,S,U,c
def pack(A, B, R, S, U, c):
    z = np.concatenate([A[:8], B, R[:4], S, U[:2], [c]])
    return np.concatenate([z.real, z.imag])
def polys(A, B, R, S):
    P = pmul(A, A, B); Q = pmul(R, R, R, R, R, S)
    Ps = (P[:, None] * Xp[:len(P)]).sum(0); Qs = (Q[:, None] * Xp[:len(Q)]).sum(0)
    return P, Q, Ps, Qs
def resid(x, wphi=1.0):
    A, B, R, S, U, c = unpack(x)
    P, Q, Ps, Qs = polys(A, B, R, S)
    r_phi = (mapkit.sdiv(Ps, Qs, Lu) - phiv)[:Nfit]     # anchor to the modular function
    D = np.zeros(25, complex); D[:len(P)] += P; D[:len(Q)] -= Q
    U12 = pmul(*([U]*12))
    r_str = (D - c*U12) / (np.max(np.abs(D)) + 1e-300)   # over-1 ramification, normalized
    return np.concatenate([wphi*r_phi.real, wphi*r_phi.imag, r_str.real, r_str.imag])

def split_norms(x, wphi):
    A, B, R, S, U, c = unpack(x)
    P, Q, Ps, Qs = polys(A, B, R, S)
    rp = (mapkit.sdiv(Ps, Qs, Lu) - phiv)[:Nfit]
    D = np.zeros(25, complex); D[:len(P)] += P; D[:len(Q)] -= Q
    U12 = pmul(*([U]*12)); rs = (D - c*U12)/(np.max(np.abs(D))+1e-300)
    return np.linalg.norm(rp), np.linalg.norm(rs)

x0 = pack(A0, B0, R0, S0, U0, c0)
A, B, R, S, U, c = unpack(x0)
P0, Q0, _, _ = polys(A, B, R, S)
print(f"seed (structured): o12(P-Q)={order12_spread(P0-Q0):.3f}   "
      f"|r_phi|={split_norms(x0,1)[0]:.2e} |r_str|={split_norms(x0,1)[1]:.2e}")

# structure is EXACT (it is a Belyi map); phi-match is X-noise-limited -> weight structure hard,
# let phi-match only pick the non-degenerate branch.  Scan the anchor weight.
print(f"\n  {'wphi':>7} {'o12(P-Q)':>10} {'|r_phi|':>10} {'|r_str|':>10} {'nfev':>6}", flush=True)
for wphi in (1.0, 1e-2, 1e-4):
    sol = least_squares(lambda x: resid(x, wphi), x0, method='lm', max_nfev=1500, xtol=1e-14, ftol=1e-14)
    A, B, R, S, U, c = unpack(sol.x); P, Q, _, _ = polys(A, B, R, S)
    rp, rs = split_norms(sol.x, wphi)
    print(f"  {wphi:>7.0e} {order12_spread(P-Q):>10.3e} {rp:>10.2e} {rs:>10.2e} {sol.nfev:>6}", flush=True)
