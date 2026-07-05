"""Seeded structured (pejorative-manifold) fit for the [2,12,5] Belyi map.

Enforce the ramification by construction:
  P = alpha * A^2 * B          A monic deg 8 (double zeros), B monic deg 8 (simple zeros)   sigma_0=2^8 1^8
  Q =         R^5 * S          R monic deg 4 (5-fold poles),  S monic deg 4 (simple poles)   sigma_inf=5^4 1^4
  P - Q = c * U^12 * V^12      U,V monic deg 1 (the two order-12 points)                     sigma_1=12^2
  c = alpha - 1  (leading-coeff match, A^2B and R^5S monic, beta=1)

Residuals (fp64, scipy least_squares):
  (1) series match : phiv (x) Q(X(t)) - P(X(t)),  n=0..Nfit-1   (selects THE map, pins scale)
  (2) over-1 ident : (P - Q) - c U^12 V^12,  X-coeffs 0..24     (global multiplicity structure)

Seed: generic degree-24 fit -> divide out the spurious common factor -> reduced zeros/poles.
Stage 0 prints the reduced structure so the a/b and r/s multiplicity split can be set sanely.

Usage: python3 structured_fit.py [N] [Nfit]
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
from numpy.polynomial import polynomial as P_
from scipy.optimize import least_squares

mp.mp.dps = 35
Lu = mapkit.Lu; d = mapkit.d
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2950
Nfit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]; atol = mp.mpf('1e-8')
NA, NB, NR, NS = 8, 8, 4, 4          # root counts

def ppow(poly, n):
    out = np.array([1.0 + 0j])
    for _ in range(n): out = P_.polymul(out, poly)
    return out

# ---------- forms -> X, phiv, Xp powers (fp64) ----------
dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=16)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, jt.tail_weights(N), n_slices=18, iters=8)[0]
Xmp, ech = jr.hauptmodul_mp(B, dim, atol)
Nser = Lu
Xf = np.array([complex(Xmp[n]) for n in range(Nser)])
phiv = np.array([complex(mapkit.phiv[n]) for n in range(Nser)])
Xp = [np.zeros(Nser, complex) for _ in range(d + 1)]; Xp[0][0] = 1.0
for i in range(1, d + 1): Xp[i] = np.convolve(Xp[i - 1], Xf)[:Nser]

def poly_at_Xt(coeffs):                       # P(X(t)) as a t-series, coeffs low->high in X
    out = np.zeros(Nser, complex)
    for k in range(len(coeffs)): out += coeffs[k] * Xp[k]
    return out

# ---------- generic seed ----------
o12, pmp, qmp = jr.order12_mp(Xmp)
Pg = np.array([complex(pmp[i]) for i in range(d + 1)])   # low->high, deg 24
Qg = np.array([complex(qmp[i]) for i in range(d + 1)])
Pg /= Pg[-1]; Qg /= Qg[-1]                                # make monic for root inspection
rP = np.roots(Pg[::-1]); rQ = np.roots(Qg[::-1])
# common roots (spurious factor): each rP near some rQ
common = [z for z in rP if min(abs(z - w) for w in rQ) < 1e-3]
def prune(roots, comm):
    out = list(roots)
    for cz in comm:
        j = int(np.argmin([abs(z - cz) for z in out])); out.pop(j)
    return np.array(out)
zeros = prune(rP, common)          # distinct zeros of phi
poles = prune(rQ, common)          # distinct poles of phi
print(f"N={N} Nfit={Nfit}  generic o12={o12:.3e}", flush=True)
print(f"  common(spurious) factor deg = {len(common)}", flush=True)
print(f"  #distinct zeros = {len(zeros)} (want {NA+NB}=16):", flush=True)
print("    " + " ".join(f"{z:.3f}" for z in sorted(zeros, key=abs)), flush=True)
print(f"  #distinct poles = {len(poles)} (want {NR+NS}=8):", flush=True)
print("    " + " ".join(f"{z:.3f}" for z in sorted(poles, key=abs)), flush=True)

# ---------- pack / unpack ----------
def unpack(x):
    c = x[0::2] + 1j * x[1::2]
    i = 0
    a = c[i:i+NA]; i += NA
    b = c[i:i+NB]; i += NB
    r = c[i:i+NR]; i += NR
    s = c[i:i+NS]; i += NS
    u = c[i]; v = c[i+1]; alpha = c[i+2]
    return a, b, r, s, u, v, alpha
def build_PQ(x):
    a, b, r, s, u, v, alpha = unpack(x)
    A = P_.polyfromroots(a); Bp = P_.polyfromroots(b)
    R = P_.polyfromroots(r); Sp = P_.polyfromroots(s)
    Pp = alpha * P_.polymul(ppow(A, 2), Bp)          # deg 24
    Qp = P_.polymul(ppow(R, 5), Sp)                  # deg 24 (beta=1)
    return Pp, Qp, u, v, alpha

def residuals(x):
    Pp, Qp, u, v, alpha = build_PQ(x)
    Pt = poly_at_Xt(Pp); Qt = poly_at_Xt(Qp)
    res_series = (np.convolve(phiv, Qt)[:Nfit] - Pt[:Nfit])
    c = alpha - 1.0
    UV = P_.polymul(ppow(P_.polyfromroots([u]), 12), ppow(P_.polyfromroots([v]), 12))
    res_over1 = (Pp - Qp) - c * UV[:d + 1]
    r = np.concatenate([res_series, 3.0 * res_over1])
    return np.concatenate([r.real, r.imag])

# ---------- seed ----------
def make_seed(zeros, poles):
    z = list(sorted(zeros, key=abs)); p = list(sorted(poles, key=abs))
    while len(z) < NA + NB: z.append(0.3 + 0.1j)
    while len(p) < NR + NS: p.append(0.3 + 0.1j)
    a = np.array(z[:NA]); b = np.array(z[NA:NA+NB])         # heuristic split
    r = np.array(p[:NR]); s = np.array(p[NR:NR+NS])
    # over-1 points: two roots of (Pg-Qg) not near zeros/poles
    rPQ = np.roots((Pg - Qg)[::-1])
    far = sorted(rPQ, key=lambda w: -min(min(abs(w-zz) for zz in zeros),
                                         min(abs(w-pp) for pp in poles)))
    u, v = far[0], far[1]
    alpha = Pg[-1] / Qg[-1] if abs(Qg[-1]) > 0 else 1.0 + 0j
    c = np.concatenate([a, b, r, s, [u, v, 1.0 + 0j]])
    x = np.empty(2 * len(c)); x[0::2] = c.real; x[1::2] = c.imag
    return x

x0 = make_seed(zeros, poles)
r0 = residuals(x0)
print(f"\n  seed residual norm = {np.linalg.norm(r0):.3e}", flush=True)
t = time.time()
sol = least_squares(residuals, x0, method='trf', max_nfev=4000, xtol=1e-14, ftol=1e-14)
print(f"  LM: cost={sol.cost:.3e} |res|={np.linalg.norm(sol.fun):.3e} nfev={sol.nfev} [{time.time()-t:.0f}s]", flush=True)

# multiplicity quality: spread of A^2 (double zeros should be exact), etc.
a, b, r, s, u, v, alpha = unpack(sol.x)
Pp, Qp, _, _, _ = build_PQ(sol.x)
c = alpha - 1.0
UV = P_.polymul(ppow(P_.polyfromroots([u]), 12), ppow(P_.polyfromroots([v]), 12))
over1 = np.linalg.norm((Pp - Qp) - c * UV[:d+1])
Pt = poly_at_Xt(Pp); Qt = poly_at_Xt(Qp)
ser = np.linalg.norm(np.convolve(phiv, Qt)[:Nfit] - Pt[:Nfit])
print(f"  final: series-resid={ser:.3e}  over1-resid={over1:.3e}", flush=True)
print(f"  |u|={abs(u):.4f} |v|={abs(v):.4f} alpha={alpha:.4f}", flush=True)
