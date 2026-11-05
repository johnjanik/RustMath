"""Is the o12~1.84 wall a degraded-coefficient artifact or a true generic-fit overfit?
Scan Nfit for the generic degree-24 fit (max Hauptmodul).  49 free params, so Nfit>=49.
If o12 drops near Nfit~50 (minimal degraded coeffs) -> degradation poisons it.
If o12 is flat across Nfit -> genuine overfit of the 12^2 fiber; only the STRUCTURED fit fixes it."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
mp.mp.dps = 35
Lu = mapkit.Lu; d = mapkit.d
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]; atol = mp.mpf('1e-8')

def o12_at(X, Nfit):
    phiv = [mp.mpf(float(x.real)) for x in mapkit.phiv]
    Xp = [[mp.mpc(0)]*Lu for _ in range(d+1)]; Xp[0][0] = mp.mpc(1)
    for i in range(1, d+1): Xp[i] = jr.mconv(Xp[i-1], X, Lu)
    cols = [jr.mconv(phiv, Xp[i], Lu) for i in range(d+1)] + \
           [[-Xp[i][n] for n in range(Lu)] for i in range(d+1)]
    A = mp.matrix(Nfit, 2*(d+1))
    for n in range(Nfit):
        for j in range(2*(d+1)): A[n, j] = cols[j][n]
    U, S, Vt = mp.svd(A)
    svmin = float(S[S.rows-1]); gap = float(S[S.rows-2]/S[S.rows-1])
    nv = [Vt[Vt.rows-1, j] for j in range(2*(d+1))]
    q = nv[:d+1]; p = nv[d+1:]; Dp = [p[i]-q[i] for i in range(d+1)]
    rr = [complex(r) for r in mp.polyroots([Dp[i] for i in range(d,-1,-1)], maxsteps=400, extraprec=400)]
    smed = np.median(np.abs([r for r in rr if abs(r) > 1e-14])); R = list(rr); worst = 0.
    for _ in range(2):
        best = None
        for i in range(len(R)):
            idx = sorted(range(len(R)), key=lambda j: abs(R[j]-R[i]))[:12]
            sp = max(abs(R[j]-R[i]) for j in idx)
            if best is None or sp < best[0]: best = (sp, idx)
        worst = max(worst, best[0]/smed); ss = set(best[1])
        R = [R[j] for j in range(len(R)) if j not in ss]
    return svmin, gap, worst

dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=16)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, jt.tail_weights(N), n_slices=18, iters=8)[0]
X, ech = jr.hauptmodul_mp(B, dim, atol)
print(f"N={N}  scanning Nfit (49 params):", flush=True)
print(f"  {'Nfit':>5} {'sv_min':>10} {'gap':>9} {'o12':>9}", flush=True)
for Nfit in (50, 52, 55, 58, 64, 72, 80):
    sv, gap, o12 = o12_at(X, Nfit)
    print(f"  {Nfit:>5} {sv:>10.2e} {gap:>9.1e} {o12:>9.4e}", flush=True)
