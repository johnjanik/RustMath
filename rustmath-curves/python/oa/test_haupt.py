"""The Hauptmodul construction injects the wall: mapkit builds X from the HIGHEST-valuation
(least accurate) echelon forms.  Test alternative constructions from the LOWEST (most accurate)
forms and measure o12 + N-stability of X[3].  A valid Hauptmodul from accurate forms should give
low o12 (the degree-24 fit absorbs the Mobius change of coordinate between constructions)."""
import numpy as np, sys, os, time, math
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
mp.mp.dps = 35
Lu = mapkit.Lu; rho = mapkit.rho; kk = mapkit.k
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]; atol = mp.mpf('1e-8')
kap2 = mp.mpf(mapkit.KAPPA2)

def echelon(B, dim):
    _, ech = jr.hauptmodul_mp(B, dim, atol)
    return ech

def X_construct(ech, mode):
    cand = [v for v in ech if v + 1 in ech]
    if mode == "ratio":                    # X = ech[1]/ech[0], the two most accurate forms
        Xu = jr.msdiv(ech[1], ech[0], Lu, atol)
    else:
        m = min(cand) if mode == "min" else max(cand)
        g = ech[m]; h = ech[m + 1]; cc = h[m + 2]
        den = [g[i] + cc * h[i] for i in range(Lu)]
        Xu = jr.msdiv(h, den, Lu, atol)
    return [Xu[n] * kap2 ** n for n in range(Lu)]

dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=16)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, jt.tail_weights(N), n_slices=18, iters=8)[0]
ech = echelon(B, dim)
print(f"N={N}  echelon valuations {sorted(ech)}", flush=True)
for mode in ("max", "min", "ratio"):
    try:
        X = X_construct(ech, mode)
        o12, p, q = jr.order12_mp(X)
        print(f"  mode={mode:5s}  o12={o12:.4e}  X[1]={mp.nstr(X[1],8)}  X[3]={mp.nstr(X[3],8)}", flush=True)
    except Exception as e:
        print(f"  mode={mode:5s}  FAILED: {e}", flush=True)
