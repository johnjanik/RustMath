"""P4.5.1-2: dd jet-Tikhonov null basis + Veronese ring gates for a vertex chart.

Chart-parameterized version of jet_dd.py (which hardcodes the a-chart rho): load the
streamed EXT dump for ANY chart center, solve the jet-constrained Tikhonov system

    b_{<v} = 0,  b_v = 1,  minimize ||C b||^2 + lam ||W_tail b||^2,   C = M diag(rho^n)

to double-double accuracy (fp64 LU + dd Ozaki residual refinement), keeping BOTH dd
components of the solution, and gate the result in mpmath:

    - ring self-check   x_chart(w) vs w on |w| = R          (jet basis => x = w + O(w^9))
    - Veronese residual on rings R = 0.05, 0.1, 0.2, 0.3
    - null residual / truncation floor rho^N, tail norms

The b-chart valuations are the raw 0..8 (no parity at an order-12 preimage: the s1 cycle
through the base coset has length 12). Gates (P4.5.2): Veronese resid <= 1e-10 on R<=0.2,
<= 1e-8 on R<=0.3, x_b=w_b improving with dd precision.

Usage: python3 chart_dd.py <matrix.bin> <rho-full-decimal> [vals-csv] [lams-csv] [out.npz]
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
import jet_tikhonov as jt
from read_ext import read_ext
from ddcx import (xp, _GPU, cnew, cH, to_c128, cscale_ddreal, csub, cget_cols)
from ozaki import ozaki_gemm_complex

mp.mp.dps = 40


def asx(a):
    return xp.asarray(a)


def tonp(a):
    return (a.get() if _GPU else np.asarray(a))


def dd_powers(rho_str, dim):
    """rho^n for n=0..dim-1 as (hi, lo) f64 pairs, computed in mpmath (40 dps)."""
    r = mp.mpf(rho_str)
    hi = np.empty(dim)
    lo = np.empty(dim)
    p = mp.mpf(1)
    for n in range(dim):
        h = float(p)
        hi[n] = h
        lo[n] = float(p - mp.mpf(h))
        p *= r
    return hi, lo


def load_dd_C(path, rho_str):
    """dim, C = M diag(rho^n) as dd on xp, with dd-accurate scale (unlike jet_dd's fp64 scale)."""
    dim, nl, re, im = read_ext(path)
    reh = asx(re[0]); imh = asx(im[0])
    rel = asx(re[1]) if nl > 1 else xp.zeros_like(reh)
    iml = asx(im[1]) if nl > 1 else xp.zeros_like(imh)
    sh, sl = dd_powers(rho_str, dim)
    return dim, cscale_ddreal(cnew(reh, rel, imh, iml), asx(sh)[None, :], asx(sl)[None, :])


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _cadd_dd_fp(xh, xl, dx):
    srh, srl = _two_sum(xh.real, dx.real)
    sih, sil = _two_sum(xh.imag, dx.imag)
    return (srh + 1j * sih), ((srl + xl.real) + 1j * (sil + xl.imag))


def solve_dd_refine2(G, vals, lam, w, n_slices=16, iters=8, verbose=False):
    """jet_dd.solve_dd_refine, but returns the solution as a dd PAIR (Bh, Bl)."""
    import scipy.linalg as sla
    N = G['reh'].shape[0] - 1
    lamw2 = (lam * w * w)
    Hdd = {k: G[k].copy() for k in G}
    idx = xp.arange(N + 1)
    Hdd['reh'][idx, idx] = Hdd['reh'][idx, idx] + asx(lamw2)
    Hhi = tonp(to_c128(Hdd))
    Bh = np.zeros((N + 1, len(vals)), complex)
    Bl = np.zeros((N + 1, len(vals)), complex)
    resid = []; tn = []
    for k, v in enumerate(vals):
        free = np.arange(v + 1, N + 1)
        lu, piv = sla.lu_factor(Hhi[np.ix_(free, free)])
        xh = sla.lu_solve((lu, piv), -Hhi[free, v])
        xl = np.zeros_like(xh)
        Hff_dd = {kk: Hdd[kk][np.ix_(asx(free), asx(free))] for kk in Hdd}
        Hfv_dd = cget_cols({kk: Hdd[kk][asx(free)] for kk in Hdd}, xp.asarray([v]))
        rn = None
        for it in range(iters):
            xc = cnew(asx(xh.real)[:, None], asx(xl.real)[:, None],
                      asx(xh.imag)[:, None], asx(xl.imag)[:, None])
            Hx = ozaki_gemm_complex(Hff_dd, xc, n_slices)
            r = csub({kk: -Hfv_dd[kk] for kk in Hfv_dd}, Hx)
            rnp = tonp(to_c128(r))[:, 0]
            rn = np.linalg.norm(rnp)
            dx = sla.lu_solve((lu, piv), rnp)
            xh, xl = _cadd_dd_fp(xh, xl, dx)
            if verbose:
                print(f"    v={v:2d} it={it}: |r|={rn:.3e}", flush=True)
        Bh[v, k] = 1.0
        Bh[free, k] = xh
        Bl[free, k] = xl
        resid.append(rn)
        tn.append(np.linalg.norm(w * (Bh[:, k] + Bl[:, k])))
    return Bh, Bl, np.array(resid), np.array(tn)


# ---------- mpmath ring gates ----------

def mp_series_eval(Bh, Bl, w):
    """values of the 9 series sum_n b_n w^n at mpmath complex w -> list of mpc (Horner)."""
    dim, m = Bh.shape
    out = []
    for j in range(m):
        acc = mp.mpc(0)
        for n in range(dim - 1, -1, -1):
            acc = acc * w + (mp.mpc(Bh[n, j]) + mp.mpc(Bl[n, j]))
        out.append(acc)
    return out


def mp_recover_x(G):
    """G ~ lam nu(x): x = <G[:d], G[1:]> / <G[:d], G[:d]>, plus Veronese residual (all mpmath)."""
    d = len(G) - 1
    num = mp.mpc(0); den = mp.mpf(0)
    for i in range(d):
        num += mp.conj(G[i]) * G[i + 1]
        den += abs(G[i]) ** 2
    x = num / den
    nu = [mp.mpc(1)]
    for _ in range(d):
        nu.append(nu[-1] * x)
    ip = mp.mpc(0); nn = mp.mpf(0); gg = mp.mpf(0)
    for i in range(d + 1):
        ip += mp.conj(nu[i]) * G[i]
        nn += abs(nu[i]) ** 2
        gg += abs(G[i]) ** 2
    lam = ip / nn
    rs = mp.mpf(0)
    for i in range(d + 1):
        rs += abs(G[i] - lam * nu[i]) ** 2
    return x, mp.sqrt(rs) / mp.sqrt(gg)


def ring_gates(Bh, Bl, rings=(0.05, 0.1, 0.2, 0.3), nth=8, local_pow=1):
    """per ring: max |x - w^local_pow| and max Veronese residual (jet basis => A = I,
    G = values). local_pow=1 for an ordinary/order-12 chart (x = w + O(w^9)); 2 for the
    order-2 chart, where the even jet basis tracks u = w^2."""
    rows = []
    for R in rings:
        ex, er = mp.mpf(0), mp.mpf(0)
        for kk in range(nth):
            w = mp.mpf(R) * mp.exp(1j * 2 * mp.pi * (kk + 0.3) / nth)
            G = mp_series_eval(Bh, Bl, w)
            x, res = mp_recover_x(G)
            ex = max(ex, abs(x - w ** local_pow))
            er = max(er, res)
        rows.append((R, float(ex), float(er)))
    return rows


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/john/sweep_2_12_5/m_order12_N3000.bin"
    rho_str = sys.argv[2] if len(sys.argv) > 2 else \
        "0.997905455122650969838041889181703530183366396"
    vals = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(9))
    lams = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else \
        [1e-8, 1e-12, 1e-16, 1e-20]
    outnpz = sys.argv[5] if len(sys.argv) > 5 else path.replace(".bin", "_ddbasis.npz")
    local_pow = int(sys.argv[6]) if len(sys.argv) > 6 else 1

    t = time.time()
    dim, C = load_dd_C(path, rho_str)
    N = dim - 1
    floor = float(mp.mpf(rho_str) ** N)
    print(f"{os.path.basename(path)}: dim={dim} GPU={_GPU} rho^N={floor:.2e} vals={vals}", flush=True)
    print(f"loaded dd C [{time.time()-t:.0f}s]", flush=True)

    t = time.time()
    G = ozaki_gemm_complex(cH(C), C, 16)
    print(f"dd Gram built [{time.time()-t:.0f}s]", flush=True)

    w = jt.tail_weights(N)
    best = None
    for lam in lams:
        t = time.time()
        Bh, Bl, res, tn = solve_dd_refine2(G, vals, lam, w)
        gates = ring_gates(Bh, Bl, local_pow=local_pow)
        gs = "  ".join(f"R={R}: |x-w|={ex:.1e} ver={er:.1e}" for R, ex, er in gates)
        print(f"lam={lam:.0e} res/floor={res.max()/floor:.1e} tailmax={tn.max():.1e} "
              f"[{time.time()-t:.0f}s]\n    {gs}", flush=True)
        score = gates[2][2]           # Veronese resid at R=0.2
        if best is None or score < best[0]:
            best = (score, lam, Bh, Bl, res, tn)

    score, lam, Bh, Bl, res, tn = best
    np.savez(outnpz, Bh=Bh, Bl=Bl, vals=np.array(vals), lam=lam,
             rho=rho_str, resid=res, tailnorm=tn)
    print(f"saved {outnpz} (lam={lam:.0e}, ver(R=0.2)={score:.2e})", flush=True)
