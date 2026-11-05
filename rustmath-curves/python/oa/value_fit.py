"""Approach 1: value-space rational fit (DD_LIFT_NOTE Exp 5).

Evaluate X(t), phi(t) at sample points t_i and fit  P(X_i) = phi_i Q(X_i)  at the VALUES, not
the series.  This dodges two error sources: (a) the deep noisy series coefficients are suppressed
by |t|^n at small radius, and (b) X^k is evaluated as a number, not built by k series convolutions.
Degree oracle: scan d, look for a sharp validation-residual drop at d=24 (primitive, correct degree),
d=12 (quotient coordinate), or none (genuine X/phi accuracy floor -> go to approach 3, deepen forms).

Usage: python3 value_fit.py [N] [dps]
"""
import numpy as np, sys, os, time, math
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
mp.mp.dps = int(sys.argv[2]) if len(sys.argv) > 2 else 35
Lu = mapkit.Lu; d = mapkit.d; vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]

def evalser(a, t):
    acc = mp.mpc(0)
    for c in reversed(a): acc = acc * t + c
    return acc

# ---- dd forms -> X (mpmath series), phi ----
t0 = time.time(); dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=8); w = jt.tail_weights(N)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, w, iters=4)[0]
X, ech = jr.hauptmodul_mp(B, dim, mp.mpf('1e-8'))
phiv = [mp.mpc(complex(x).real, complex(x).imag) for x in mapkit.phiv]
print(f"dd forms + X ready [{time.time()-t0:.0f}s]", flush=True)

# ---- convergence / range diagnostic ----
print("  radius   |X(r)|     |phi(r)|   tail term |X_k r^k| at k=Lu-1", flush=True)
for r in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    tr = mp.mpf(r)
    xr = abs(evalser(X, tr)); pr = abs(evalser(phiv, tr))
    tail = abs(X[Lu-1]) * tr**(Lu-1)
    print(f"  {r:.2f}   {float(xr):.3e}  {float(pr):.3e}   {float(tail):.1e}", flush=True)

# ---- sample points (spread radii; angles offset to avoid symmetry) ----
def sample(radii, per):
    pts = []
    for r in radii:
        for kk in range(per):
            th = 2 * math.pi * (kk + 0.137) / per
            pts.append(mp.mpf(r) * mp.exp(mp.mpc(0, 1) * mp.mpf(th)))
    return pts
fit_pts = sample((0.15, 0.25, 0.35, 0.45), 40)      # 160 fit points
val_pts = sample((0.20, 0.30, 0.40), 25)            # 75 held-out validation points
Xf = np.array([complex(evalser(X, t)) for t in fit_pts]); Ff = np.array([complex(evalser(phiv, t)) for t in fit_pts])
Xv = np.array([complex(evalser(X, t)) for t in val_pts]); Fv = np.array([complex(evalser(phiv, t)) for t in val_pts])

def value_fit(deg):
    Vx = np.vander(Xf, deg + 1, increasing=True)      # [1, X, ..., X^deg]
    A = np.hstack([Vx, -Ff[:, None] * Vx])
    cn = np.linalg.norm(A, axis=0); cn[cn == 0] = 1
    U, s, Vh = np.linalg.svd(A / cn)
    nv = (Vh[-1].conj()) / cn
    p = nv[:deg + 1]; q = nv[deg + 1:]
    # validation: |P(Xv) - Fv Q(Xv)| / |Fv Q(Xv)|
    Vxv = np.vander(Xv, deg + 1, increasing=True)
    P = Vxv @ p; Q = Vxv @ q
    rel = np.linalg.norm(P - Fv * Q) / max(np.linalg.norm(Fv * Q), 1e-300)
    return s, p, q, rel

print("\n  degree oracle (value space):", flush=True)
print(f"  {'d':>3} {'sv_min':>10} {'gap':>9} {'valid_relres':>14}", flush=True)
for deg in (8, 12, 16, 20, 22, 24, 26, 28):
    s, p, q, rel = value_fit(deg)
    print(f"  {deg:>3} {s[-1]:>10.2e} {s[-2]/s[-1]:>9.1e} {rel:>14.2e}", flush=True)

# ---- o12 at d=24 ----
s, p, q, rel = value_fit(24)
D = p - q
mp.mp.dps = 30
rr = [complex(x) for x in mp.polyroots([mp.mpc(D[i]) for i in range(len(D)-1, -1, -1)],
                                       maxsteps=500, extraprec=400)]
smed = np.median(np.abs([r for r in rr if abs(r) > 1e-12])); R = list(rr); worst = 0.
for _ in range(2):
    best = None
    for i in range(len(R)):
        idx = sorted(range(len(R)), key=lambda j: abs(R[j]-R[i]))[:12]
        sp = max(abs(R[j]-R[i]) for j in idx)
        if best is None or sp < best[0]: best = (sp, idx)
    worst = max(worst, best[0]/smed); ss = set(best[1]); R = [R[j] for j in range(len(R)) if j not in ss]
print(f"\n  d=24 value fit: valid_relres={rel:.2e}  o12={worst:.4e}", flush=True)
