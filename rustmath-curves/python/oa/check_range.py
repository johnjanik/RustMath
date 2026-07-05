"""Unifying diagnostic: does the Hauptmodul X actually range over P^1, or is it nearly constant?
Evaluate |X(t)| out to the domain boundary |t|->rho, checking the series still converges (tail).
Also test every echelon form-ratio ech[i]/ech[j]: does ANY of them vary over a wide range (a genuine
Hauptmodul), or are all the forms nearly proportional (=> the 9-dim form space is near-degenerate)?"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
mp.mp.dps = 35
Lu = mapkit.Lu; rho = mapkit.rho
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2950
vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]; atol = mp.mpf('1e-8')

def evalser(a, t):
    acc = mp.mpc(0)
    for c in reversed(a): acc = acc * t + c
    return acc

dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=16)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, jt.tail_weights(N), n_slices=18, iters=8)[0]
X, ech = jr.hauptmodul_mp(B, dim, atol)

print(f"N={N}  |X(t)| vs radius (series tail = convergence check):", flush=True)
print(f"  {'|t|':>6} {'|X(t)|':>12} {'|phi(t)|':>12} {'tail':>10}", flush=True)
for r in (0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99):
    tr = mp.mpf(r)
    xr = abs(evalser(X, tr)); pr = abs(evalser([mp.mpc(complex(x)) for x in mapkit.phiv], tr))
    tail = float(abs(X[Lu-1]) * tr**(Lu-1))
    print(f"  {r:>6.2f} {float(xr):>12.4e} {float(pr):>12.4e} {tail:>10.1e}", flush=True)

# form-ratio range test: sample ech[i]/ech[j] over the disc, report dynamic range
print(f"\n  form-ratio dynamic range (max/min |ech[i]/ech[j]| over |t|=0.1..0.95):", flush=True)
ts = [mp.mpf(r) * mp.exp(mp.mpc(0, 1) * mp.mpf(a)) for r in (0.1, 0.5, 0.9, 0.95)
      for a in (0.3, 1.1, 2.0, 2.9, 3.8, 4.7, 5.6)]
keys = sorted(ech)
for i in keys[:4]:
    row = []
    for j in keys[:4]:
        if i == j: row.append("     -   "); continue
        v = [abs(evalser(ech[i], t) / evalser(ech[j], t)) for t in ts if abs(evalser(ech[j], t)) > 1e-40]
        row.append(f"{(max(v)/max(min(v),1e-40)):8.1e}")
    print(f"   ech[{i}]/ech[j]: " + " ".join(row), flush=True)
