"""SELECTOR_NOTE §4-5: jet-normalized b-space Tikhonov solve -- the escape hatch.

Instead of hunting the smallest singular vectors of M (which finds the tail-freedom overfit),
recover each physical form DIRECTLY as the bounded automorphic coefficient vector with a
prescribed valuation.  In b-space  C b = 0,  C = M diag(rho^n)  (since y_n = rho^n b_n), solve
for each pivot valuation v:

    b_{<v} = 0,  b_v = 1,   minimize  ||C b||^2 + lam ||W_tail b||^2

via the augmented (not normal-equation) least squares.  No SVD, no near-null cluster, so the
ClusterRR accuracy risk cannot arise.  This fp64 prototype validates the SHAPE (reality, X1,
echelon); accuracy still floors at fp64 -- lift to dd/Ozaki least squares once the gates pass.

Usage: python3 jet_tikhonov.py [N]           # default 1400 (a known-good check below the wall)
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, physical_selector as ps
rho = mapkit.rho

def bspace(M):
    """C = M diag(rho^n): column n scaled by rho^n, so C b = M y with y_n = rho^n b_n."""
    N = M.shape[1] - 1
    return M * (rho ** np.arange(N + 1))[None, :]

def tail_weights(N, start_frac=0.65, power=4.0, floor=0.0):
    n = np.arange(N + 1, dtype=float); start = start_frac * N
    t = np.maximum(0.0, (n - start) / max(1.0, N - start))
    return floor + t ** power

def col_echelon_valuations(B, rtol=1e-6):
    """b-space pivot valuations of the columns of B (dim,m): the m leading coefficient indices."""
    Bw = (B / (np.abs(B).max(0) + 1e-300)).astype(complex)
    cols = list(range(Bw.shape[1])); vals = []
    for _ in range(Bw.shape[1]):
        best = None
        for c in cols:
            nz = np.nonzero(np.abs(Bw[:, c]) > rtol)[0]
            if len(nz) and (best is None or nz[0] < best[1]): best = (c, int(nz[0]))
        if best is None: break
        c, v = best; vals.append(v); cols.remove(c)
        piv = Bw[v, c]
        for c2 in cols: Bw[:, c2] -= (Bw[v, c2] / piv) * Bw[:, c]
    return sorted(vals)

def jet_basis(M_or_G, valuations, lam, w=None, start_frac=0.65, power=4.0, floor=0.0, is_gram=False):
    """All forms share one Gram G=C^H C.  For each valuation v the jet-constrained minimizer of
    b^H(G+lam W^2)b with b_{<v}=0,b_v=1 solves H[free,free] x = -H[free,v].  Reuses G across
    every valuation and lam (W^2 is diagonal), so the whole scan costs one Gram + small solves."""
    if is_gram:
        G = M_or_G; N = G.shape[0] - 1
    else:
        C = bspace(M_or_G); N = C.shape[1] - 1; G = C.conj().T @ C
    if w is None: w = tail_weights(N, start_frac, power, floor)
    H = G + np.diag((lam * w * w).astype(complex))
    B = np.zeros((N + 1, len(valuations)), complex); res = []; tn = []
    for k, v in enumerate(valuations):
        free = np.arange(v + 1, N + 1)
        x = np.linalg.solve(H[np.ix_(free, free)], -H[free, v])
        b = np.zeros(N + 1, complex); b[v] = 1.0; b[free] = x
        B[:, k] = b
        res.append(np.sqrt(max(0.0, (b.conj() @ (G @ b)).real))); tn.append(np.linalg.norm(w * b))
    return B, np.array(res), np.array(tn)

def gram(M):
    C = bspace(M); return C.conj().T @ C

def b_to_y(B):
    return B * (rho ** np.arange(B.shape[0]))[:, None]

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1400
    vals = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(9))
    dim, M = ps.load_hi(f"/home/john/sweep_2_12_5/m_N{N}.bin")
    floor = rho ** N
    print(f"N={N}  rho^N={floor:.2e}  M {M.shape}  jet valuations={vals}", flush=True)

    # below-wall reference: raw SVD smallest-9 IS the physical forms there, a known-good target
    t0 = time.time(); s, Vh = ps.fp64_svd_ascending(M)
    rref = mapkit.evaluate(ps.sel_raw(s, Vh, dim), dim, "raw")
    print(f"raw smallest-9: reality={rref['reality']:.2e} |X1-k2|={rref['x1_err']:.2e} "
          f"o12={rref['o12']:.3e}  [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time(); G = gram(M); w = tail_weights(N)
    print(f"Gram C^H C built [{time.time()-t0:.0f}s]; scanning lambda", flush=True)
    lamscan = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else \
              [1e-32, 1e-28, 1e-24, 1e-20, 1e-16]
    for lam in lamscan:
        t0 = time.time()
        B, res, tn = jet_basis(G, vals, lam, w=w, is_gram=True)
        r = mapkit.evaluate(b_to_y(B), dim, "")
        ok = "OK" if (r['reality'] < 1e-3 and r['x1_err'] < 1e-4
                     and r['valuations'][:9] == list(range(9))) else "  "
        print(f"  {ok} lam={lam:.0e}  reality={r['reality']:.2e} |X1-k2|={r['x1_err']:.2e} "
              f"o12={r['o12']:.3e}  res/floor={res.max()/floor:.1e} tailmax={tn.max():.1e} "
              f"[{time.time()-t0:.0f}s]", flush=True)
