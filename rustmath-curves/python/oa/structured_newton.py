"""Structured on-stratum Newton fit -- the pejorative-manifold map recovery (Zeng, 2301.07880).

The generic degree-24 fit floors o12 ~1.85 regardless of form accuracy: it has 49 free params and
overfits the 12-fold fiber.  Impose the ramification instead, parametrized by the ROOTS (so the
multiplicities are exact by construction):

    phi = kappa * N / D,   N = prod (X-a_i)^2 prod (X-b_j)   [8 double zeros, 8 simple]  (sigma0=2^8 1^8)
                            D = prod (X-r_i)^5 prod (X-s_j)   [4 quintuple poles, 4 simple] (sigma_inf=5^4 1^4)

and Gauss-Newton-fit phi to the dd-accurate Hauptmodul series in mpmath.  If phi is matched to ~1e-12
(only dd makes that possible), phi-1 inherits the 12^2 structure automatically and o12 collapses.

Usage: python3 structured_newton.py [N] [dps] [maxit] [Nfit]
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp

rho = mapkit.rho; Lu = mapkit.Lu; d = mapkit.d
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
mp.mp.dps = int(sys.argv[2]) if len(sys.argv) > 2 else 35
MAXIT = int(sys.argv[3]) if len(sys.argv) > 3 else 12
NFIT = int(sys.argv[4]) if len(sys.argv) > 4 else 46
vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]

def cluster(roots, mult, count):
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

def polyfromroots_mp(roots):
    c = [mp.mpc(1)]
    for rt in roots:
        new = [mp.mpc(0)] * (len(c) + 1)
        for i in range(len(c)): new[i] += -rt * c[i]; new[i+1] += c[i]
        c = new
    return c                                             # lowest-first, monic, len = #roots+1

def poly_to_series(P, Xp):
    return [sum(P[m] * Xp[m][n] for m in range(len(P))) for n in range(Lu)]

def o12_of_coeffs(D):
    rr = [complex(x) for x in mp.polyroots([D[i] for i in range(len(D)-1, -1, -1)],
                                           maxsteps=500, extraprec=500)]
    smed = np.median(np.abs([r for r in rr if abs(r) > 1e-14])); R = list(rr); worst = 0.
    for _ in range(2):
        best = None
        for i in range(len(R)):
            idx = sorted(range(len(R)), key=lambda j: abs(R[j]-R[i]))[:12]
            sp = max(abs(R[j]-R[i]) for j in idx)
            if best is None or sp < best[0]: best = (sp, idx)
        worst = max(worst, best[0]/smed); s = set(best[1]); R = [R[j] for j in range(len(R)) if j not in s]
    return worst

# ---- 1. dd forms -> dd Hauptmodul X, and the generic-fit seed ----
print(f"N={N} dps={mp.mp.dps} maxit={MAXIT} Nfit={NFIT}", flush=True)
t = time.time(); dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=8); w = jt.tail_weights(N)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, w, iters=4)[0]
X, ech = jr.hauptmodul_mp(B, dim, mp.mpf('1e-8'))
print(f"dd Hauptmodul ready [{time.time()-t:.0f}s]", flush=True)
o12_gen, p, q = jr.order12_mp(X)
print(f"generic fit: o12={o12_gen:.4f}", flush=True)

phiv = [mp.mpc(complex(x).real, complex(x).imag) for x in mapkit.phiv]
Xp = [[mp.mpc(0)] * Lu for _ in range(d + 1)]; Xp[0][0] = mp.mpc(1)
for i in range(1, d + 1): Xp[i] = jr.mconv(Xp[i-1], X, Lu)

# seed roots by clustering the generic-fit roots
rP = np.roots([complex(p[i]) for i in range(d, -1, -1)]); ar, br = cluster(rP, 2, 8)
rQ = np.roots([complex(q[i]) for i in range(d, -1, -1)]); rr_, sr = cluster(rQ, 5, 4)
kap0 = complex(p[-1]) / complex(q[-1])
seed = np.concatenate([ar, br, rr_, sr, [kap0]])

def unpack(x):
    z = np.array([mp.mpc(x[i], x[i+25]) for i in range(25)])
    return z[0:8], z[8:16], z[16:20], z[20:24], z[24]     # a, b, r, s, kappa
def pack(z):
    return [mp.re(v) for v in z] + [mp.im(v) for v in z]

def PQ(a, b, r, s, kap):
    Np = polyfromroots_mp(list(a) + list(a) + list(b))    # deg 24
    Dp = polyfromroots_mp(list(r) * 5 + list(s))          # deg 24
    P = [kap * v for v in Np]
    return P, Dp
def resid(x):
    a, b, r, s, kap = unpack(x)
    P, Dp = PQ(a, b, r, s, kap)
    Ns = poly_to_series(P, Xp); Ds = poly_to_series(Dp, Xp)
    phim = jr.msdiv(Ns, Ds, Lu, mp.mpf('1e-30'))
    out = []
    for n in range(NFIT):
        e = phim[n] - phiv[n]; out.append(mp.re(e)); out.append(mp.im(e))
    return out

# ---- 2. Gauss-Newton (mpmath, forward-diff Jacobian, QR least squares) ----
x = pack(np.array([mp.mpc(complex(v).real, complex(v).imag) for v in seed]))
h = mp.mpf('1e-18')
def rnorm(r): return mp.sqrt(sum(e*e for e in r))
r0 = resid(x); nrm = rnorm(r0)
P, Dp = PQ(*unpack(x)); print(f"seed: |r|={float(nrm):.3e} o12={o12_of_coeffs([P[i]-Dp[i] for i in range(25)]):.4e}", flush=True)
mu = mp.mpf('1e-3')                                       # Levenberg-Marquardt damping
for it in range(MAXIT):
    t = time.time()
    r0 = resid(x); m = len(r0); nvar = len(x)
    J = mp.matrix(m, nvar)
    for j in range(nvar):
        xj = list(x); xj[j] += h; rj = resid(xj)
        for i in range(m): J[i, j] = (rj[i] - r0[i]) / h
    JT = J.T; A = JT * J; g = JT * mp.matrix([-e for e in r0])
    accepted = False
    for _try in range(8):
        Ad = A.copy()
        for i in range(nvar): Ad[i, i] += mu * (A[i, i] + mp.mpf('1e-40'))
        try:
            dx = mp.lu_solve(Ad, g)
        except Exception:
            mu *= 8; continue
        xn = [x[i] + dx[i] for i in range(nvar)]; rn = resid(xn); nn = rnorm(rn)
        if float(nn) < float(nrm):
            x = xn; nrm = nn; mu = max(mu / 3, mp.mpf('1e-30')); accepted = True; break
        mu *= 8
    step = mp.sqrt(sum(dx[i]*dx[i] for i in range(nvar)))
    P, Dp = PQ(*unpack(x)); o12 = o12_of_coeffs([P[i]-Dp[i] for i in range(25)])
    print(f"  it{it:2d}: |r|={float(nrm):.3e} o12={o12:.4e} |dx|={float(step):.2e} mu={float(mu):.1e} "
          f"{'ok' if accepted else 'STALL'} [{time.time()-t:.0f}s]", flush=True)
    if not accepted or float(step) < 1e-28: break

a, b, r, s, kap = unpack(x)
P, Dp = PQ(a, b, r, s, kap)
print(f"\nFINAL o12={o12_of_coeffs([P[i]-Dp[i] for i in range(25)]):.4e}")
print("P = kappa*N (deg24) leading coeffs:", [mp.nstr(P[i], 8) for i in range(3)])
