"""dd within-span refinement of a chart's null basis (the P4 accuracy lift).

The fp64 SVD gives the 9-dim null span of the y-space matrix M to ~1e-11 absolute
(out-of-span contamination). This refines the SPAN to the matrix's own truncation
floor via block iterative refinement:

    R_k = M Y_k                    (dd, Ozaki GEMM on the dd limbs)
    D_k = V_r S_r^{-1} U_r^H R_k   (fp64 pseudo-inverse over the RANK subspace only --
                                    corrections along the null space don't reduce ||MY||)
    Y_{k+1} = Y_k - D_k            (dd accumulation, hi/lo pairs)

Each pass reduces the out-of-span error by ~eps_fp64/sigma_min(rank); two or three
passes reach the dd/truncation floor. The span error after convergence is
~ ||MY||/gap, i.e. floor-limited -- so the chart matrix must be dumped DEEP enough
(rho^N below the accuracy target) for the refinement to mean anything.

Output: the same npz format as svd_chart.py (jet-normalized b-space dd pairs), plus
the residual history.

Usage: python3 dd_span_refine.py <matrix.bin> <rho-full-decimal> [out.npz] [local_pow] [iters]
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
from read_ext import read_ext
from ddcx import xp, _GPU, cnew, to_c128
from ozaki import ozaki_gemm_complex
from svd_chart import jet_normalize, to_pairs
from chart_dd import ring_gates

mp.mp.dps = 40


def asx(a):
    return xp.asarray(a)


def tonp(a):
    return (a.get() if _GPU else np.asarray(a))


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def refine_span(path, kdim=9, iters=3):
    dim, nl, re, im = read_ext(path)
    Mhi = re[0] + 1j * im[0]
    Mdd = cnew(asx(re[0]), asx(re[1] if nl > 1 else np.zeros_like(re[0])),
               asx(im[0]), asx(im[1] if nl > 1 else np.zeros_like(im[0])))

    t = time.time()
    try:
        import cupy as cp
        U_g, s_g, Vh_g = cp.linalg.svd(cp.asarray(Mhi))
        U, s, Vh = cp.asnumpy(U_g), cp.asnumpy(s_g), cp.asnumpy(Vh_g)
        del U_g, s_g, Vh_g
    except Exception:
        U, s, Vh = np.linalg.svd(Mhi)
    print(f"fp64 SVD [{time.time()-t:.0f}s]  gap={s[-kdim-1]/s[-kdim]:.1e}  "
          f"rank sigma_min={s[-kdim-1]:.3e}", flush=True)

    r = dim - kdim
    Ur, sr, Vr = U[:, :r], s[:r], Vh[:r].conj().T
    Yh = Vh[-kdim:].conj().T.copy()               # (dim, kdim)
    Yl = np.zeros_like(Yh)

    hist = []
    for it in range(iters):
        Ydd = cnew(asx(Yh.real), asx(Yl.real), asx(Yh.imag), asx(Yl.imag))
        Rdd = ozaki_gemm_complex(Mdd, Ydd, 16)
        R = tonp(to_c128(Rdd))                    # (dim, kdim) dd-accurate residual
        rn = np.linalg.norm(R)
        hist.append(rn)
        D = Vr @ ((Ur.conj().T @ R) / sr[:, None])
        # Y -= D in dd
        srh, srl = _two_sum(Yh.real, -D.real)
        sih, sil = _two_sum(Yh.imag, -D.imag)
        Yh = srh + 1j * sih
        Yl = (srl + Yl.real) + 1j * (sil + Yl.imag)
        print(f"  iter {it}: ||M Y|| = {rn:.3e}", flush=True)
    return dim, Yh, Yl, np.array(hist), s


def y_to_b_mp(Yh, Yl, rho_str):
    """b = rho^{-n} y in mpmath, from the dd pair."""
    r = mp.mpf(rho_str)
    dim, kdim = Yh.shape
    B = np.empty((dim, kdim), dtype=object)
    inv_p = mp.mpf(1)
    for n in range(dim):
        for j in range(kdim):
            B[n, j] = (mp.mpc(Yh[n, j]) + mp.mpc(Yl[n, j])) * inv_p
        inv_p /= r
    return B


if __name__ == "__main__":
    path = sys.argv[1]
    rho_str = sys.argv[2]
    outnpz = sys.argv[3] if len(sys.argv) > 3 else path.replace(".bin", "_ddspan.npz")
    local_pow = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    dim, Yh, Yl, hist, s = refine_span(path, iters=iters)
    B = y_to_b_mp(Yh, Yl, rho_str)
    Bn = jet_normalize(B)
    Bh, Bl = to_pairs(Bn)
    gates = ring_gates(Bh, Bl, local_pow=local_pow)
    for R, ex, er in gates:
        print(f"  R={R}: |x-w^{local_pow}|={ex:.2e}  ver={er:.2e}", flush=True)
    np.savez(outnpz, Bh=Bh, Bl=Bl, vals=np.arange(9), lam=0.0, rho=rho_str,
             resid_hist=hist, sv_null=s[-9:], gap=s[-10] / s[-9])
    print(f"saved {outnpz}", flush=True)
