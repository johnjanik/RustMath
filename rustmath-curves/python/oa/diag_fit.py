"""Is phi exactly degree-24 rational in the Hauptmodul X?  Diagnose the generic fit at dd precision:
  (1) singular gap S[-2]/S[-1] and residual S[-1] of the [phi*Xp | -Xp] system,
  (2) actual |P/Q - phi| over the fit window,
  (3) the |P-Q| root distribution (2 clusters of 12, or generic?)."""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
mp.mp.dps = 35
N = 2500; Lu = mapkit.Lu; d = mapkit.d
vals = [0,2,4,6,8,10,12,14,16]

t = time.time(); dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=8); w = jt.tail_weights(N)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, w, iters=4)[0]
X, ech = jr.hauptmodul_mp(B, dim, mp.mpf('1e-8'))
print(f"dd X ready [{time.time()-t:.0f}s]; echelon valuations {sorted(ech)}", flush=True)

phiv = [mp.mpc(complex(x).real, complex(x).imag) for x in mapkit.phiv]
Xp = [[mp.mpc(0)]*Lu for _ in range(d+1)]; Xp[0][0] = mp.mpc(1)
for i in range(1, d+1): Xp[i] = jr.mconv(Xp[i-1], X, Lu)

for Nfit in (46, 52, 58, 64):
    cols = [jr.mconv(phiv, Xp[i], Lu) for i in range(d+1)] + \
           [[-Xp[i][n] for n in range(Lu)] for i in range(d+1)]
    A = mp.matrix(Nfit, 2*(d+1))
    for n in range(Nfit):
        for j in range(2*(d+1)): A[n, j] = cols[j][n]
    U, S, Vt = mp.svd(A)
    svs = [float(S[k]) for k in range(len(S))]
    nv = [Vt[Vt.rows-1, j] for j in range(2*(d+1))]
    q = nv[:d+1]; p = nv[d+1:]
    # P/Q - phi over the window
    Ps = jr.poly_to_series(p, Xp) if hasattr(jr, 'poly_to_series') else [sum(p[m]*Xp[m][n] for m in range(d+1)) for n in range(Lu)]
    Qs = [sum(q[m]*Xp[m][n] for m in range(d+1)) for n in range(Lu)]
    phifit = jr.msdiv(Ps, Qs, Lu, mp.mpf('1e-30'))
    err = max(abs(phifit[n]-phiv[n]) for n in range(Nfit))
    Dp = [p[i]-q[i] for i in range(d+1)]
    rr = sorted([complex(x) for x in mp.polyroots([Dp[i] for i in range(d,-1,-1)], maxsteps=500, extraprec=500)],
                key=lambda z: abs(z))
    print(f"Nfit={Nfit}: sv_gap={svs[-2]/svs[-1]:.2e} sv_min={svs[-1]:.2e} |P/Q-phi|={float(err):.2e} "
          f"o12={jr.order12_mp(X)[0] if Nfit==58 else float('nan'):.3f}", flush=True)
    print("   |P-Q| roots:", " ".join(f"{abs(z):.3f}" for z in rr), flush=True)
