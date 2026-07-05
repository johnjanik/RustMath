"""Disambiguate the imX[3] pin: is X[3] a converged (stable) complex value, or a noisy error?
Print X[1..9] to high precision so N=2500 vs N=2950 can be compared coefficient-by-coefficient.
Stable to ~1e-9 across N  => forms are accurate, X[3] genuinely complex, wall is the map fit.
Differs at ~1e-5         => forms limited to 1e-5 by M/construction (N-independent)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
mp.mp.dps = 40
N = int(sys.argv[1]); vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]
dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=16)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, jt.tail_weights(N), n_slices=18, iters=8)[0]
X, ech = jr.hauptmodul_mp(B, dim, mp.mpf('1e-8'))
# also the raw echelon form f0 imag parts (is the FORM itself real?)
f0 = ech[0]
print(f"N={N}", flush=True)
for k in range(1, 10):
    print(f"  X[{k}] = {mp.nstr(X[k], 14)}", flush=True)
print(f"  f0 imag magnitudes k=1..8: " +
      " ".join(f"{float(abs(mp.im(f0[k]))):.1e}" for k in range(1, 9)), flush=True)
