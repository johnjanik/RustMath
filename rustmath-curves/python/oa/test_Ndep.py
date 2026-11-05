"""Approach 3 sizing test: with the now-dd-accurate solve, does form accuracy improve with N?
Tell 1: imag(X[k]) (X should be real -> imag = error).  Tell 2: o12 / x1_err from the map fit.
If these shrink with N -> truncation-limited, larger N is the fix (and this sizes how large).
If flat -> the limit is M or the construction, and larger N is wasted."""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
mp.mp.dps = 40
vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]
Ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1500, 2000, 2500, 2950]

print(f"{'N':>6} {'rho^N':>10} {'reality':>10} {'x1_err':>10} {'o12':>9} "
      f"{'imX[3]':>10} {'imX[5]':>10} {'imX[7]':>10}", flush=True)
for N in Ns:
    dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
    G = jet_dd.dd_gram(C, n_slices=16)
    B = jet_dd.solve_dd_refine(G, vals, 1e-12, jt.tail_weights(N), n_slices=18, iters=8)[0]
    r = mapkit.evaluate(jt.b_to_y(B), dim, "")
    X, ech = jr.hauptmodul_mp(B, dim, mp.mpf('1e-8'))
    im = [abs(mp.im(X[k])) for k in (3, 5, 7)]
    print(f"{N:>6} {mapkit.rho**N:>10.2e} {r['reality']:>10.2e} {r['x1_err']:>10.2e} "
          f"{r['o12']:>9.3e} {float(im[0]):>10.2e} {float(im[1]):>10.2e} {float(im[2]):>10.2e}", flush=True)
