"""P4.5.1' below-wall chart basis: fp64 SVD span + mpmath jet normalization + ring gates.

The jet-Tikhonov solve (chart_dd.py) is the PAST-the-wall tool: below the wall its
per-vector minimizer trades null-span fidelity against the tail penalty and stalls at
~1e-4 ring residuals. The rebased atlas charts all sit far below the wall (floors
8e-9..1e-12), where the smallest-9 SVD subspace IS the physical form space -- the FP64
experiment on the (worse) original b-chart already achieved |x-w| = 4.7e-11 at R=0.05.

Pipeline: fp64 SVD of the hi limb -> smallest-9 right singular vectors Y -> b-space in
mpmath (b_n = rho^{-n} y_n with the full-precision rho) -> jet-normalize INSIDE the span
(B_norm = B inv(B[:9,:]), a well-conditioned 9x9 in mpmath) -> identity jet frame,
F(z) ~ nu(x_chart), same ring gates and npz format as chart_dd.py.

Usage: python3 svd_chart.py <matrix.bin> <rho-full-decimal> [out.npz] [local_pow]
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
from read_ext import read_ext
from chart_dd import ring_gates

mp.mp.dps = 40


def mp_from_pair(h, l):
    return mp.mpc(h) + mp.mpc(l)


def mp_to_pair(z):
    zh = complex(z)
    zl = complex(z - mp.mpc(zh))
    return zh, zl


def svd_basis(path, rho_str, kdim=9):
    dim, nl, re, im = read_ext(path)
    M = re[0] + 1j * im[0]
    t = time.time()
    try:
        import cupy as cp
        s_g, Vh_g = cp.linalg.svd(cp.asarray(M), compute_uv=True)[1:]
        s, Vh = cp.asnumpy(s_g), cp.asnumpy(Vh_g)
        del s_g, Vh_g
    except Exception:
        _, s, Vh = np.linalg.svd(M)
    print(f"fp64 SVD [{time.time()-t:.0f}s]  rank sv[-{kdim+3}:-{kdim}]={s[-kdim-3:-kdim]}", flush=True)
    print(f"null sv[-{kdim}:]={s[-kdim:]}  gap={s[-kdim-1]/s[-kdim]:.1e}", flush=True)
    Y = Vh[-kdim:].conj().T                       # (dim, 9)
    # b_n = rho^{-n} y_n in mpmath
    r = mp.mpf(rho_str)
    B = np.empty((dim, kdim), dtype=object)
    inv_p = mp.mpf(1)
    for n in range(dim):
        for j in range(kdim):
            B[n, j] = mp.mpc(Y[n, j]) * inv_p
        inv_p /= r
    return dim, B, s


def jet_normalize(B, kdim=9):
    """B_norm = B @ inv(B[:9,:]) in mpmath -> top block = I exactly (well-conditioned 9x9)."""
    T = mp.matrix(kdim, kdim)
    for i in range(kdim):
        for j in range(kdim):
            T[i, j] = B[i, j]
    Tinv = T ** -1
    dim = B.shape[0]
    Bn = np.empty_like(B)
    for n in range(dim):
        for j in range(kdim):
            acc = mp.mpc(0)
            for k in range(kdim):
                acc += B[n, k] * Tinv[k, j]
            Bn[n, j] = acc
    return Bn


def to_pairs(B):
    dim, m = B.shape
    Bh = np.empty((dim, m), complex)
    Bl = np.empty((dim, m), complex)
    for n in range(dim):
        for j in range(m):
            Bh[n, j], Bl[n, j] = mp_to_pair(B[n, j])
    return Bh, Bl


if __name__ == "__main__":
    path = sys.argv[1]
    rho_str = sys.argv[2]
    outnpz = sys.argv[3] if len(sys.argv) > 3 else path.replace(".bin", "_svdbasis.npz")
    local_pow = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    dim, B, s = svd_basis(path, rho_str)
    t = time.time()
    Bn = jet_normalize(B)
    Bh, Bl = to_pairs(Bn)
    print(f"jet-normalized in span [{time.time()-t:.0f}s]", flush=True)
    gates = ring_gates(Bh, Bl, local_pow=local_pow)
    for R, ex, er in gates:
        print(f"  R={R}: |x-w^{local_pow}|={ex:.2e}  ver={er:.2e}", flush=True)
    np.savez(outnpz, Bh=Bh, Bl=Bl, vals=np.arange(9), lam=0.0, rho=rho_str,
             sv_null=s[-9:], gap=s[-10] / s[-9])
    print(f"saved {outnpz}", flush=True)
