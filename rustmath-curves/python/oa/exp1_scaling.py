"""Decisive scaling check: hauptmodul_mp applies rho^-n (the recover_forms convention for a
y-vector), but solve_dd_refine returns b (already physical).  Compare feeding B (current) vs
b_to_y(B) -- does the y-input keep X[1]=kappa^2 AND push the form-algebra accuracy past coeff 55?"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
mp.mp.dps = 35
N = 2500; Lu = mapkit.Lu; vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]

def relres(cols, target, nfit):
    A = mp.matrix(nfit, len(cols))
    for n in range(nfit):
        for k in range(len(cols)): A[n, k] = cols[k][n]
    b = mp.matrix([target[n] for n in range(nfit)])
    a = mp.qr_solve(A, b)[0]; pred = A * a
    res = mp.sqrt(sum(abs(pred[n]-b[n])**2 for n in range(nfit)))
    nb = mp.sqrt(sum(abs(b[n])**2 for n in range(nfit)))
    return float(res/nb) if nb != 0 else float('nan')

t = time.time(); dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=8); w = jt.tail_weights(N)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, w, iters=4)[0]
print(f"dd forms ready [{time.time()-t:.0f}s]  kappa^2={mapkit.KAPPA2:.8g}", flush=True)

for label, inp in [("b-input (current)", B), ("y-input = b_to_y(B)", jt.b_to_y(B))]:
    X, ech = jr.hauptmodul_mp(inp, dim, mp.mpf('1e-8'))
    print(f"\n=== {label} ===", flush=True)
    if X is None:
        print("  no consecutive valuations:", sorted(ech), flush=True); continue
    print(f"  X[1]={mp.nstr(X[1],10)}   X[3]={mp.nstr(X[3],6)}", flush=True)
    f = [ech[j] for j in range(9)]; f0 = f[0]
    X2 = jr.msdiv(f[1], f[0], Lu, mp.mpf('1e-30'))
    Xp2 = [[mp.mpc(0)]*Lu for _ in range(9)]; Xp2[0][0] = mp.mpc(1)
    for i in range(1, 9): Xp2[i] = jr.mconv(Xp2[i-1], X2, Lu)
    print("  Test 1b  f_j = f_0 p_j(X'),  nfit=25/40/55:", flush=True)
    for j in (1, 2, 4, 6, 8):
        cols = [jr.mconv(f0, Xp2[k], Lu) for k in range(j+1)]
        print(f"    j={j}: " + "  ".join(f"{relres(cols, f[j], nf):.2e}" for nf in (25, 40, 55)), flush=True)
