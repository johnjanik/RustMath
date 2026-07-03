"""Mixed-precision SVD via Ogita-Aishima iterative refinement, complex + cluster-safe.
Implements Algorithms 3 (MxpSVDStep), 2 (ClusterRR), 4 (MxpSVD) from
"Mixed-Precision SVD on GPUs via Ogita-Aishima Iterative Refinement" (Schwarz et al., NVIDIA).

Purpose: resolve the 9-dim near-NULL subspace of the horribly-conditioned KMSV matrix M
(sigma_max ~ rho^-N, null sigma ~ rho^N) to double-double accuracy -- past the FP64 wall,
where the initial FP64 SVD cannot see the null space at all. Test case (12) of the paper
(one tiny sigma buried below the low-precision SVD) is exactly this situation and converges
in ~3 iterations.

All O(n^3) work is the Ozaki dd GEMM (ozaki.py); O(n^2) per-pair work is dd-complex (ddcx.py).
Self-test (__main__) builds a matrix with a known tiny-sigma subspace + huge dynamic range.

STATUS (2026-07-03, paused for power): infra validated (Ozaki GEMM -> 2e-32, dd-complex ops,
dd/td matrix dump). Self-test shows the refinement STEP does NOT yet converge: omega stuck at
~1.4e-6 (should drop quadratically toward 16 n eps_hp smax ~ 1e-22), null-subspace error stays
at the FP64 level. DEBUG PLAN on resume (in order):
  1. Test mxp_svd_step on an EASY matrix first: sigma all O(1) (no tiny, no huge dynamic range),
     random unitary U,V. FP64 SVD is near-exact there, so ONE step must drop omega to ~1e-30.
     This isolates step correctness from the null-space difficulty. If it fails here, the bug is
     purely in Alg-3 assembly (not the conditioning).
  2. omega=nan at iter 0 then finite: hunt the transient nan (likely xp.log2(0) in ozaki_split
     when a row/col of an operand is all-zero, or a 0 corr in sigma). Guard log2 input.
  3. offdiag(T) not decreasing => the branch-A coupling isn't diagonalizing. Hand-check the 2x2:
     verify alpha=t_ij+sigma_j r_ij, beta=t_bar_ji+sigma_i s_ij (NB: Alg 1 uses sigma_i in beta;
     Alg 3 line 7 as printed looked like sigma_j -- RESOLVE which by the 2x2 hand-check), the
     f/g pairing, the tbar=conj(T^T) transpose, and the U(I+F) update SIGN (try I-F if stuck).
  4. Confirm the loop actually mutates U,V (smin identical across iters => step is ~no-op;
     check that F,G are non-negligible and the Ozaki GEMM U@F isn't being zeroed)."""
import numpy as np
try:
    import cupy as xp
    _GPU = True
except Exception:
    import numpy as xp
    _GPU = False
from ozaki import ozaki_gemm_complex, dd_add, dd_sub, two_sum
from ddcx import (cnew, czeros, from_c128, to_c128, cadd, csub, cconj, cscale_ddreal,
                  cdiv_ddreal, cmul, ceye, cH, cget_cols, cset_cols, dd_mul, dd_div)

def _diag(a):
    """diagonal of a dd-complex (n,n) as dd-complex length-n (reh,rel,imh,iml vectors)."""
    n = a['reh'].shape[0]; idx = xp.arange(n)
    return {k: a[k][idx, idx].copy() for k in ('reh', 'rel', 'imh', 'iml')}
def _cT(a):   # transpose (no conjugate)
    return cnew(a['reh'].T.copy(), a['rel'].T.copy(), a['imh'].T.copy(), a['iml'].T.copy())
def _offdiag_frob(a):
    n = a['reh'].shape[0]
    m2 = a['reh']**2 + a['imh']**2
    return float(xp.sqrt(xp.sum(m2) - xp.sum(xp.diagonal(m2))))
def _frob(a):
    return float(xp.sqrt(xp.sum(a['reh']**2 + a['imh']**2)))
def _select(mask, a, b):
    """elementwise dd-complex select: where(mask, a, b)."""
    return cnew(xp.where(mask, a['reh'], b['reh']), xp.where(mask, a['rel'], b['rel']),
                xp.where(mask, a['imh'], b['imh']), xp.where(mask, a['iml'], b['iml']))

def mxp_svd_step(A, U, V, eps_lp, eps_hp, n_slices):
    """One cluster-safe Ogita-Aishima refinement step (Algorithm 3). Square complex A.
    Returns refined U, V, sigma (dd-real hi,lo vectors), and omega."""
    n = U['reh'].shape[0]
    I = ceye(n)
    UH = cH(U); VH = cH(V)
    R = csub(I, ozaki_gemm_complex(UH, U, n_slices))      # I - U^H U
    S = csub(I, ozaki_gemm_complex(VH, V, n_slices))      # I - V^H V
    T = ozaki_gemm_complex(UH, ozaki_gemm_complex(A, V, n_slices), n_slices)  # U^H A V

    # --- diagonals (Alg 3 lines 2-4) ---
    rd = _diag(R); sd = _diag(S); td = _diag(T)
    # sigma_i = |t_ii| / (1 - (Re r_ii + Re s_ii)/2)
    abst_h, abst_l = dd_add(*dd_mul(td['reh'], td['rel'], td['reh'], td['rel']),
                            *dd_mul(td['imh'], td['iml'], td['imh'], td['iml']))
    abst = xp.sqrt(abst_h + abst_l)                       # |t_ii| (fp64 is fine for the magnitude)
    corr = 1.0 - 0.5 * ((rd['reh'] + rd['rel']) + (sd['reh'] + sd['rel']))
    sigma_h = abst / corr; sigma_l = xp.zeros_like(sigma_h)
    smax = float(xp.max(sigma_h))
    gate = (2.0 * sigma_h) > (np.sqrt(eps_lp) * smax)
    delta = xp.where(gate & (sigma_h > 0), (td['imh'] + td['iml']) / (2.0 * sigma_h), 0.0)  # Im(t_ii)/(2 sigma)

    # --- off-diagonal per-pair (Alg 3 lines 5-15), vectorized over the grid ---
    si_h = sigma_h[:, None] + xp.zeros((1, n)); si_l = sigma_l[:, None] + xp.zeros((1, n))
    sj_h = sigma_h[None, :] + xp.zeros((n, 1)); sj_l = sigma_l[None, :] + xp.zeros((n, 1))
    tbar = cconj(_cT(T))                                  # t_bar_ji at grid (i,j)
    half = 0.5
    # Branch A: well separated
    gap_h, gap_l = dd_sub(sj_h, sj_l, si_h, si_l)
    sum_h, sum_l = dd_add(sj_h, sj_l, si_h, si_l)
    alpha = cadd(T, cscale_ddreal(R, sj_h, sj_l))                    # t_ij + sigma_j r_ij
    beta  = cadd(tbar, cscale_ddreal(S, si_h, si_l))                # t_bar_ji + sigma_i s_ij
    denom_h, denom_l = dd_mul(gap_h, gap_l, sum_h, sum_l)           # (sj-si)(sj+si)
    fA = cdiv_ddreal(cadd(cscale_ddreal(alpha, sj_h, sj_l), cscale_ddreal(beta, si_h, si_l)), denom_h, denom_l)
    gA = cdiv_ddreal(cadd(cscale_ddreal(alpha, si_h, si_l), cscale_ddreal(beta, sj_h, sj_l)), denom_h, denom_l)
    # Branch B: near-equal but nonzero -> orthogonality + antisymmetric
    two_sum_h = 2.0 * sum_h; two_sum_l = 2.0 * sum_l
    aB = cdiv_ddreal(csub(T, tbar), two_sum_h, two_sum_l)           # (t_ij - t_bar_ji)/(2(si+sj))
    halfR = cscale_ddreal(R, xp.full((n, n), half), xp.zeros((n, n)))
    halfS = cscale_ddreal(S, xp.full((n, n), half), xp.zeros((n, n)))
    fB = cadd(halfR, aB); gB = csub(halfS, aB)
    # Branch C: orthogonality only
    fC = halfR; gC = halfS
    thr = np.sqrt(eps_lp) * smax
    maskA = xp.abs(gap_h) > thr
    maskB = (~maskA) & ((sum_h) > thr)
    F = _select(maskA, fA, _select(maskB, fB, fC))
    G = _select(maskA, gA, _select(maskB, gB, gC))

    # --- overwrite diagonal (Alg 3 lines 3-4) ---
    idx = xp.arange(n)
    for key, val in (('reh', half * rd['reh']), ('rel', half * rd['rel']),
                     ('imh', delta), ('iml', xp.zeros_like(delta))):
        F[key][idx, idx] = val
    for key, val in (('reh', half * sd['reh']), ('rel', half * sd['rel']),
                     ('imh', -delta), ('iml', xp.zeros_like(delta))):
        G[key][idx, idx] = val

    # --- multiplicative update U(I+F), V(I+G) (Alg 3 line 20) ---
    Up = cadd(U, ozaki_gemm_complex(U, F, n_slices))
    Vp = cadd(V, ozaki_gemm_complex(V, G, n_slices))
    omega = 2.0 * (_offdiag_frob(T) + _frob(A) * max(_frob(R), _frob(S)))
    return Up, Vp, (sigma_h, sigma_l), omega

def cluster_rr(A, U, V, J, n_slices):
    """Algorithm 2: Rayleigh-Ritz refinement of one cluster J (list of column indices)."""
    Jx = xp.asarray(J)
    UJ = cget_cols(U, Jx); VJ = cget_cols(V, Jx)
    AVJ = ozaki_gemm_complex(A, VJ, n_slices)
    C = ozaki_gemm_complex(cH(UJ), AVJ, n_slices)          # |J| x |J|
    Cc = to_c128(C)
    Cc = np.asarray(Cc.get() if _GPU else Cc)
    P, s, Qh = np.linalg.svd(Cc)                           # C = P diag(s) Qh
    Q = Qh.conj().T
    Pdd = from_c128(xp.asarray(P)); Qdd = from_c128(xp.asarray(Q))
    UJn = ozaki_gemm_complex(UJ, Pdd, n_slices)
    VJn = ozaki_gemm_complex(VJ, Qdd, n_slices)
    cset_cols(U, Jx, UJn); cset_cols(V, Jx, VJn)
    return s

def detect_null_cluster(sigma_h, n_expected=None, gap_factor=10.0):
    """Return indices of the smallest-sigma cluster (the null space). Uses the largest
    relative gap in the sorted spectrum; if n_expected given, prefers that split."""
    s = np.asarray(sigma_h.get() if _GPU else sigma_h)
    order = np.argsort(s)
    ss = s[order]
    # find the biggest multiplicative gap in the lower part
    ratios = ss[1:] / np.maximum(ss[:-1], 1e-300)
    if n_expected is not None and 0 < n_expected < len(ss):
        cut = n_expected
    else:
        cut = int(np.argmax(ratios[:len(ss)//2])) + 1
    return list(order[:cut]), ss, cut

def mxp_svd(A_c128, n_slices=6, max_iter=10, n_null=None, verbose=True):
    """Algorithm 4 driver. A_c128: (n,n) complex128 (the hi part of M; dd lo assumed 0 here,
    or pass a dd-complex via mxp_svd_dd for the real dd matrix). Returns dd U,V and sigma."""
    A = from_c128(xp.asarray(A_c128))
    return _mxp_svd_core(A, A_c128, n_slices, max_iter, n_null, verbose)

def mxp_svd_dd(A_dd, A_c128_hi, n_slices=6, max_iter=10, n_null=None, verbose=True):
    """Same, but A given as a dd-complex dict (the extended-precision matrix M)."""
    return _mxp_svd_core(A_dd, A_c128_hi, n_slices, max_iter, n_null, verbose)

def _mxp_svd_core(A, A_c128_hi, n_slices, max_iter, n_null, verbose):
    n = A['reh'].shape[0]
    eps_lp = 2.0**-53; eps_hp = 2.0**-106
    U0, s0, Vh0 = np.linalg.svd(np.asarray(A_c128_hi))     # initial low-precision (fp64) SVD
    U = from_c128(xp.asarray(U0)); V = from_c128(xp.asarray(Vh0.conj().T))
    sigma_h = xp.asarray(s0)
    for it in range(max_iter):
        U, V, (sigma_h, sigma_l), omega = mxp_svd_step(A, U, V, eps_lp, eps_hp, n_slices)
        smax = float(xp.max(sigma_h))
        if verbose:
            print(f"  iter {it}: omega={omega:.3e}  target={16*n*eps_hp*smax:.3e}  smin={float(xp.min(sigma_h)):.3e}")
        if omega <= 16 * n * eps_hp * smax:
            break
    # cluster-refine the null space
    J, ss, cut = detect_null_cluster(sigma_h, n_expected=n_null)
    if verbose:
        print(f"  null cluster: {len(J)} vecs; sigma spectrum tail {ss[:cut+2]}")
    cluster_rr(A, U, V, J, n_slices)
    return U, V, sigma_h, J

# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Self-test: matrix with a KNOWN 2-dim tiny-sigma subspace and a huge dynamic range,
    # mimicking M (sigma_max ~ 1e8, tiny sigma ~ 1e-9). FP64 SVD cannot resolve the tiny
    # subspace; refinement should recover it to ~dd.
    rng = np.random.default_rng(3)
    n = 40
    # random unitaries
    Qr, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))
    Qc, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))
    sig = np.concatenate([np.geomspace(1e8, 1.0, n-2), [3e-9, 1e-9]])  # 2 tiny at the end
    A = (Qr * sig) @ Qc.conj().T
    A = A.astype(np.complex128)
    # true null-ish right vectors = last 2 columns of Qc
    Vtrue = Qc[:, -2:]
    # FP64 baseline
    _, s_fp, Vh_fp = np.linalg.svd(A)
    Vfp = Vh_fp.conj().T[:, -2:]
    def subspace_err(Vapprox, Vtrue):
        Vt = Vtrue.conj().T
        P = Vapprox @ (Vapprox.conj().T)         # projector onto approx
        return np.linalg.norm(P @ Vtrue - Vtrue)  # residual of true in approx span
    print(f"FP64 SVD: smallest two sigma = {s_fp[-2:]}  (true 3e-9, 1e-9)")
    print(f"FP64 null-subspace error = {subspace_err(Vfp, Vtrue):.2e}")
    print("Refining (Ogita-Aishima, dd):")
    U, V, sig_r, J = mxp_svd(A, n_slices=6, max_iter=8, n_null=2, verbose=True)
    Vc = to_c128(V); Vc = np.asarray(Vc.get() if _GPU else Vc)
    Vnull = Vc[:, J]
    print(f"refined null-subspace error = {subspace_err(Vnull, Vtrue):.2e}  (want << FP64)")
