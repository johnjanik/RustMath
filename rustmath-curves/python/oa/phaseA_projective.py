"""Option 2, reframed (per the note): the Hauptmodul is a PROJECTION of the form map
z -> [f_0:...:f_8] in P^8, not a fixed ratio.  X = h/(g+ch) was one arbitrary projection that
happened to be near-constant.  Test the note's core claim:

  (A) projective (Fubini-Study) spread of [f_0:...:f_8] over the disk -- do the forms move
      projectively even though the single ratio X barely does?
  (B) coordinate selection: find linear functionals l0,l1 with X=(F l1)/(F l0) maximizing
      GLOBAL variation, not a local echelon ratio.
  (C) degree oracle on the new coordinate from global values.

If (A) spread is large and (B) gives a wide-ranging X -> the forms are fine, the old X was a bad
local projection, and we can proceed WITHOUT the group action.  If spread stays tiny even here,
the u-chart is genuinely too local and we must build the multi-chart (group-action) evaluator.
"""
import numpy as np, sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
mp.mp.dps = 35

# ---- note's routines ----
def normalize_rows(F, eps=1e-300):
    return F / np.maximum(np.linalg.norm(F, axis=1)[:, None], eps)
def fs_dist(v, w):
    num = abs(np.vdot(v, w)); den = max(np.linalg.norm(v) * np.linalg.norm(w), 1e-300)
    return math.acos(min(num / den, 1.0))
def projective_spread(F):
    Fn = normalize_rows(np.asarray(F, complex)); n = Fn.shape[0]
    ds = [fs_dist(Fn[i], Fn[j]) for i in range(n) for j in range(i + 1, n)]
    return dict(n=n, max_fs=float(np.max(ds)), mean_fs=float(np.mean(ds)))
def coord_vals(F, a, b, eps=1e-300):
    den = F @ b
    if np.min(np.abs(den)) < eps: return None
    return (F @ a) / den
def coord_score(X):
    absX = np.abs(X)
    if np.any(~np.isfinite(absX)): return -np.inf
    spread = np.percentile(absX, 95) / max(np.percentile(absX, 5), 1e-300)
    return math.log1p(spread) + math.log1p(np.max(absX) - np.min(absX)) - 0.1 * math.log1p(np.max(absX))
def runit(n, rng):
    z = rng.normal(size=n) + 1j * rng.normal(size=n); return z / max(np.linalg.norm(z), 1e-300)
def select_coord(F, trials=4000, seed=0):
    F = np.asarray(F, complex); rng = np.random.default_rng(seed); nf = F.shape[1]; best = None
    U, s, Vh = np.linalg.svd(F, full_matrices=False)
    cands = [Vh[i].conj() for i in range(min(nf, len(Vh)))] + [runit(nf, rng) for _ in range(trials)]
    for a in cands:
        for _ in range(4):
            b = runit(nf, rng); X = coord_vals(F, a, b)
            if X is None: continue
            sc = coord_score(X)
            if best is None or sc > best['score']:
                best = dict(score=float(sc), a=a, b=b, X=X,
                            min_den=float(np.min(np.abs(F @ b))), maxX=float(np.max(np.abs(X))),
                            spread=float(np.percentile(np.abs(X), 95) / max(np.percentile(np.abs(X), 5), 1e-300)))
    return best
def fit_rat(xs, ys, deg):
    rows = [np.concatenate([[x**k for k in range(deg+1)], [-y * x**k for k in range(deg+1)]]) for x, y in zip(xs, ys)]
    U, s, Vh = np.linalg.svd(np.array(rows), full_matrices=False); v = Vh[-1].conj()
    p = v[:deg+1]; q = v[deg+1:]
    num = sum(p[k]*xs**k for k in range(deg+1)); den = sum(q[k]*xs**k for k in range(deg+1))
    return dict(deg=deg, svmin=float(s[-1]), gap=float(s[-2]/s[-1]), rel=float(np.linalg.norm(num/den-ys)/max(np.linalg.norm(ys),1e-300)))

# ---- forms ----
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2950
vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]; Lu = mapkit.Lu
dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=16)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, jt.tail_weights(N), n_slices=18, iters=8)[0]
Xold, ech = jr.hauptmodul_mp(B, dim, mp.mpf('1e-8'))
forms = [np.array([complex(ech[j][n]) for n in range(Lu)]) for j in sorted(ech)]
phiser = np.array([complex(mapkit.phiv[n]) for n in range(Lu)])
ev = lambda coef, u: np.polyval(coef[::-1], u)

pts = np.array([r * np.exp(1j * 2 * math.pi * (kk + 0.17) / 28)
                for r in (0.1, 0.3, 0.5, 0.7, 0.85, 0.93, 0.97, 0.99) for kk in range(28)])
F = np.array([[ev(f, u) for f in forms] for u in pts])       # (npts, 9)
phivals = np.array([ev(phiser, u) for u in pts])

print(f"N={N}  npts={len(pts)}", flush=True)
print("  OLD X = h/(g+ch) range over these pts: spread(95/5) =",
      f"{np.percentile(np.abs([ev([complex(c) for c in Xold], u) for u in pts]),95)/max(np.percentile(np.abs([ev([complex(c) for c in Xold],u) for u in pts]),5),1e-300):.2f}x", flush=True)
ps = projective_spread(F)
print(f"  (A) projective spread:  max_FS = {ps['max_fs']:.4f} rad   mean_FS = {ps['mean_fs']:.4f} rad   (pi/2 = {math.pi/2:.3f})", flush=True)
best = select_coord(F, trials=4000)
print(f"  (B) best projective coord:  score={best['score']:.3f}  spread(95/5)={best['spread']:.3e}x  maxX={best['maxX']:.3e}  min_den={best['min_den']:.2e}", flush=True)
Xv = best['X']
print(f"  (C) degree oracle on new coordinate (global values, phi at same u_i):", flush=True)
for deg in (8, 12, 16, 20, 24, 28):
    r = fit_rat(Xv, phivals, deg)
    print(f"      d={deg:>2}  svmin={r['svmin']:.2e}  gap={r['gap']:.1e}  rel={r['rel']:.2e}", flush=True)
