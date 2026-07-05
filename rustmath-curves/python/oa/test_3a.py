"""Approach 3a: does the higher-precision solve (16-slice Gram + dd residual + dd solution accum)
break the 2e-12 refinement stall, and does the form accuracy now extend deep?
Compares the OLD recipe (8-slice Gram, 4 fp64-accum iters) vs the NEW one on Test 1b."""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
mp.mp.dps = 40
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
Lu = mapkit.Lu; vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]

def relres(cols, target, nfit):
    A = mp.matrix(nfit, len(cols))
    for n in range(nfit):
        for k in range(len(cols)): A[n, k] = cols[k][n]
    b = mp.matrix([target[n] for n in range(nfit)])
    a = mp.qr_solve(A, b)[0]; pred = A * a
    res = mp.sqrt(sum(abs(pred[n]-b[n])**2 for n in range(nfit)))
    nb = mp.sqrt(sum(abs(b[n])**2 for n in range(nfit)))
    return float(res/nb) if nb != 0 else float('nan')

def test1b(B, tag):
    X, ech = jr.hauptmodul_mp(B, dim, mp.mpf('1e-8'))
    f = [ech[j] for j in range(9)]; f0 = f[0]
    X2 = jr.msdiv(f[1], f[0], Lu, mp.mpf('1e-35'))
    Xp2 = [[mp.mpc(0)]*Lu for _ in range(9)]; Xp2[0][0] = mp.mpc(1)
    for i in range(1, 9): Xp2[i] = jr.mconv(Xp2[i-1], X2, Lu)
    print(f"\n  [{tag}]  X[3]={mp.nstr(X[3],6)}  (imag part = accuracy tell)", flush=True)
    print(f"  Test 1b  f_j=f_0 p_j(X'),  nfit=25/40/55/70:", flush=True)
    for j in (1, 2, 4, 6, 8):
        cols = [jr.mconv(f0, Xp2[k], Lu) for k in range(j+1)]
        print(f"    j={j}: " + "  ".join(f"{relres(cols, f[j], nf):.2e}" for nf in (25, 40, 55, 70)), flush=True)

t0 = time.time(); dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
print(f"loaded C {dim}x{dim} [{time.time()-t0:.0f}s]; rho^N floor = {mapkit.rho**N:.2e}", flush=True)

# OLD recipe
t = time.time(); Gold = jet_dd.dd_gram(C, n_slices=8)
Bold, rold, _ = jet_dd.solve_dd_refine(Gold, vals, 1e-12, jt.tail_weights(N), n_slices=8, iters=4)
print(f"\nOLD (8-slice Gram, 4 iters): H-resid max={rold.max():.2e} [{time.time()-t:.0f}s]", flush=True)
test1b(Bold, "OLD")

# NEW recipe
t = time.time(); Gnew = jet_dd.dd_gram(C, n_slices=16)
print(f"\n16-slice Gram built [{time.time()-t:.0f}s]", flush=True)
t = time.time(); Bnew, rnew, _ = jet_dd.solve_dd_refine(Gnew, vals, 1e-12, jt.tail_weights(N),
                                                        n_slices=18, iters=10, verbose=True)
print(f"NEW (16-slice Gram, 18-slice resid, 10 dd iters): H-resid max={rnew.max():.2e} [{time.time()-t:.0f}s]", flush=True)
test1b(Bnew, "NEW")
