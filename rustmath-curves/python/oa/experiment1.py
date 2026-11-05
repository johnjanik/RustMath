"""Experiment 1 (DD_LIFT_NOTE): internal Hauptmodul algebra test.

Does the weight-4 form space close on X, i.e.  f_j = f_0 * p_j(X),  deg p_j <= j ?
This uses NO phi -- it certifies X as a coordinate for the form algebra.  Residual ~ dd form
accuracy (1e-12) => X is fine, blocker is downstream (phi/normalization).  Residual ~6e-4 =>
the Hauptmodul is the wall.  Also runs the shift-recovery cross-check X f_j = f_{j+1}.

Usage: python3 experiment1.py [N] [dps]
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
mp.mp.dps = int(sys.argv[2]) if len(sys.argv) > 2 else 35
Lu = mapkit.Lu
vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]

def lstsq_relres(cols, target, nfit):
    """min || sum_k a_k cols[k] - target ||, first nfit coeffs; return rel residual."""
    ncol = len(cols)
    A = mp.matrix(nfit, ncol)
    for n in range(nfit):
        for k in range(ncol): A[n, k] = cols[k][n]
    b = mp.matrix([target[n] for n in range(nfit)])
    a = mp.qr_solve(A, b)[0]
    pred = A * a
    res = mp.sqrt(sum(abs(pred[n]-b[n])**2 for n in range(nfit)))
    nb = mp.sqrt(sum(abs(b[n])**2 for n in range(nfit)))
    return float(res / nb) if nb != 0 else float('nan')

# ---- dd forms -> mpmath forms_u (echelon) + X ----
t = time.time(); dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=8); w = jt.tail_weights(N)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, w, iters=4)[0]
X, ech = jr.hauptmodul_mp(B, dim, mp.mpf('1e-8'))
print(f"N={N} dps={mp.mp.dps}: dd forms + Hauptmodul ready [{time.time()-t:.0f}s]; "
      f"echelon valuations {sorted(ech)}", flush=True)

# Xp powers
Xp = [[mp.mpc(0)] * Lu for _ in range(9)]; Xp[0][0] = mp.mpc(1)
for i in range(1, 9): Xp[i] = jr.mconv(Xp[i-1], X, Lu)

f = [ech[j] for j in range(9)]                            # forms by valuation 0..8
f0 = f[0]

print("\n  TEST 1  f_j =? f_0 * p_j(X),  deg p_j <= j   (rel residual)", flush=True)
print(f"  {'j':>2} " + "".join(f"{('nfit='+str(nf)):>14}" for nf in (25, 40, 55)), flush=True)
for j in range(1, 9):
    cols = [jr.mconv(f0, Xp[k], Lu) for k in range(j + 1)]
    row = "".join(f"{lstsq_relres(cols, f[j], nf):>14.2e}" for nf in (25, 40, 55))
    print(f"  {j:>2} {row}", flush=True)

# TEST 1b: same closure test but with X' = f_1/f_0 (direct ratio) -- removes the h/(g+ch)
# normalization, so f_1/f_0 = X' EXACTLY.  If forms close as polynomials in X' to dd accuracy,
# the earlier failure was the normalization (fixable); if it still fails, X/forms are inaccurate.
X2 = jr.msdiv(f[1], f[0], Lu, mp.mpf('1e-30'))
Xp2 = [[mp.mpc(0)] * Lu for _ in range(9)]; Xp2[0][0] = mp.mpc(1)
for i in range(1, 9): Xp2[i] = jr.mconv(Xp2[i-1], X2, Lu)
print("\n  TEST 1b  f_j =? f_0 * p_j(X'),  X' = f_1/f_0   (rel residual)", flush=True)
print(f"  {'j':>2} " + "".join(f"{('nfit='+str(nf)):>14}" for nf in (25, 40, 55)), flush=True)
for j in range(1, 9):
    cols = [jr.mconv(f0, Xp2[k], Lu) for k in range(j + 1)]
    row = "".join(f"{lstsq_relres(cols, f[j], nf):>14.2e}" for nf in (25, 40, 55))
    print(f"  {j:>2} {row}", flush=True)

# how accurate is X itself, coarsely: leading coeffs of X and X'
print("\n  X  [1..4]:", [mp.nstr(X[i], 6) for i in range(1, 5)], flush=True)
print("  X' [1..4]:", [mp.nstr(X2[i], 6) for i in range(1, 5)], flush=True)
