"""dd lift of the jet b-space solve (jet_tikhonov.py).

The fp64 jet solve floors o12 ~3 because forming the Gram G = C^H C in fp64 loses the small
entries to the rho^n dynamic range (catastrophic cancellation).  Form G in DOUBLE-DOUBLE via
the Ozaki GEMM and the small entries survive.

  Stage 1  dd Gram -> fp64 solve      : accurate G entries, fp64 storage + solve.
  Stage 2  dd Gram -> dd refinement   : iterative refinement with dd residuals for full accuracy.

Usage: python3 jet_dd.py [N] [valuations-csv] [lam-csv] [stage]
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt
from read_ext import read_ext
from ddcx import (xp, _GPU, cnew, cH, to_c128, from_c128, cscale_ddreal, csub, cget_cols)
from ozaki import ozaki_gemm_complex

rho = mapkit.rho
def asx(a):   return xp.asarray(a)
def tonp(a):  return (a.get() if _GPU else np.asarray(a))

def load_dd_C(path):
    """Load the dd matrix M, return dim and C = M diag(rho^n) as a dd-complex dict on xp."""
    dim, nl, re, im = read_ext(path)
    reh = asx(re[0]); imh = asx(im[0])
    rel = asx(re[1]) if nl > 1 else xp.zeros_like(reh)
    iml = asx(im[1]) if nl > 1 else xp.zeros_like(imh)
    s = (rho ** np.arange(dim)).astype(float)
    sh = asx(s)[None, :]; sl = xp.zeros_like(sh)
    return dim, cscale_ddreal(cnew(reh, rel, imh, iml), sh, sl)

def dd_gram(C, n_slices=16):
    return ozaki_gemm_complex(cH(C), C, n_slices)      # G = C^H C in dd (needs many slices: ~22 orders)

def _two_sum(a, b):                                    # numpy real compensated add
    s = a + b; bb = s - a
    return s, (a - (s - bb)) + (b - bb)
def _cadd_dd_fp(xh, xl, dx):                           # (xh+xl) + dx, complex -> (hi, lo)
    srh, srl = _two_sum(xh.real, dx.real); sih, sil = _two_sum(xh.imag, dx.imag)
    return (srh + 1j*sih), ((srl + xl.real) + 1j*(sil + xl.imag))

def solve_dd_refine(G, vals, lam, w, n_slices=16, iters=8, verbose=False):
    """For each valuation solve H[free,free] x = -H[free,v] to (near) dd accuracy.  H = G + lam W^2.
    fp64 LU factor (once per valuation), residual r = -H[:,v] - H_ff x computed in dd via the Ozaki
    GEMM with MANY slices (G spans ~22 orders), solution x accumulated in dd -- so it converges below
    the fp64 factor's own residual floor.  verbose prints |r| per iteration."""
    import scipy.linalg as sla
    N = G['reh'].shape[0] - 1
    lamw2 = (lam * w * w)
    Hdd = {k: G[k].copy() for k in G}
    idx = xp.arange(N + 1)
    Hdd['reh'][idx, idx] = Hdd['reh'][idx, idx] + asx(lamw2)
    Hhi = tonp(to_c128(Hdd))
    B = np.zeros((N + 1, len(vals)), complex); resid = []; tn = []
    for k, v in enumerate(vals):
        free = np.arange(v + 1, N + 1)
        lu, piv = sla.lu_factor(Hhi[np.ix_(free, free)])
        xh = sla.lu_solve((lu, piv), -Hhi[free, v])    # fp64 seed (hi)
        xl = np.zeros_like(xh)                          # dd accumulation (lo)
        Hff_dd = {kk: Hdd[kk][np.ix_(asx(free), asx(free))] for kk in Hdd}
        Hfv_dd = cget_cols({kk: Hdd[kk][asx(free)] for kk in Hdd}, xp.asarray([v]))
        rn = None
        for it in range(iters):
            xc = cnew(asx(xh.real)[:, None], asx(xl.real)[:, None],
                      asx(xh.imag)[:, None], asx(xl.imag)[:, None])
            Hx = ozaki_gemm_complex(Hff_dd, xc, n_slices)          # H_ff @ x in dd
            r = csub({kk: -Hfv_dd[kk] for kk in Hfv_dd}, Hx)       # -H[:,v] - H_ff x  (dd)
            rnp = tonp(to_c128(r))[:, 0]; rn = np.linalg.norm(rnp)
            dx = sla.lu_solve((lu, piv), rnp)
            xh, xl = _cadd_dd_fp(xh, xl, dx)                       # x += dx in dd
            if verbose: print(f"    v={v:2d} it={it}: |r|={rn:.3e}", flush=True)
        b = np.zeros(N + 1, complex); b[v] = 1.0; b[free] = xh + xl
        B[:, k] = b
        resid.append(rn); tn.append(np.linalg.norm(w * b))
    return B, np.array(resid), np.array(tn)

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    vals = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0,2,4,6,8,10,12,14,16]
    lams = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1e-16, 1e-12, 1e-8]
    stage = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    floor = rho ** N
    print(f"N={N} GPU={_GPU} stage={stage} valuations={vals}", flush=True)
    t = time.time(); dim, C = load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
    print(f"loaded dd C {dim}x{dim} [{time.time()-t:.0f}s]", flush=True)
    t = time.time(); G = dd_gram(C, n_slices=8)
    print(f"dd Gram built [{time.time()-t:.0f}s]", flush=True)
    w = jt.tail_weights(N)
    Ghi = tonp(to_c128(G))                             # accurate entries, fp64 storage
    for lam in lams:
        t = time.time()
        if stage == 1:
            B, res, tn = jt.jet_basis(Ghi, vals, lam, w=w, is_gram=True)
            rr = np.array([np.nan])
        else:
            B, rr, tn = solve_dd_refine(G, vals, lam, w)
        r = mapkit.evaluate(jt.b_to_y(B), dim, "")
        print(f"  lam={lam:.0e} reality={r['reality']:.2e} |X1-k2|={r['x1_err']:.2e} "
              f"o12={r['o12']:.3e} tailmax={tn.max():.1e} resid={rr.max():.1e} [{time.time()-t:.0f}s]",
              flush=True)
