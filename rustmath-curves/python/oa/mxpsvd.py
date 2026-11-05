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

_DBG = {}   # debug knobs (e.g. {'nnull': 2}) for dd block decomposition of T

def _diag(a):
    """diagonal of a dd-complex (n,n) as dd-complex length-n (reh,rel,imh,iml vectors)."""
    n = a['reh'].shape[0]; idx = xp.arange(n)
    return {k: a[k][idx, idx].copy() for k in ('reh', 'rel', 'imh', 'iml')}
def _cT(a):   # transpose (no conjugate)
    return cnew(a['reh'].T.copy(), a['rel'].T.copy(), a['imh'].T.copy(), a['iml'].T.copy())
def _offdiag_frob(a):
    # zero the diagonal FIRST (subtracting it from the total is catastrophic cancellation
    # when the diagonal is O(1) and the off-diagonal is ~1e-15: gives a ~1e-7 floor and,
    # once rounding turns the argument negative, sqrt(neg)=NaN).
    n = a['reh'].shape[0]; idx = xp.arange(n)
    m2 = (a['reh']**2 + a['imh']**2).copy()
    m2[idx, idx] = 0.0
    return float(xp.sqrt(xp.sum(m2)))
def _frob(a):
    return float(xp.sqrt(xp.sum(a['reh']**2 + a['imh']**2)))
def _frob_dd_offdiag(a):
    # dd-accurate off-diagonal Frobenius: |hi+lo|^2 reaches ~1e-30, diagonal zeroed first.
    n = a['reh'].shape[0]; idx = xp.arange(n)
    re = a['reh'] + a['rel']; im = a['imh'] + a['iml']
    m2 = (re*re + im*im).copy(); m2[idx, idx] = 0.0
    return float(xp.sqrt(xp.sum(m2)))
def _frob_dd(a):
    re = a['reh'] + a['rel']; im = a['imh'] + a['iml']
    return float(xp.sqrt(xp.sum(re*re + im*im)))
def _select(mask, a, b):
    """elementwise dd-complex select: where(mask, a, b)."""
    return cnew(xp.where(mask, a['reh'], b['reh']), xp.where(mask, a['rel'], b['rel']),
                xp.where(mask, a['imh'], b['imh']), xp.where(mask, a['iml'], b['iml']))

def mxp_svd_step(A, U, V, eps_lp, eps_hp, n_slices, debug=False):
    """One cluster-safe Ogita-Aishima refinement step (Algorithm 3). Square complex A.
    Returns refined U, V, sigma (dd-real hi,lo vectors), and omega."""
    n = U['reh'].shape[0]
    I = ceye(n)
    UH = cH(U); VH = cH(V)
    R = csub(I, ozaki_gemm_complex(UH, U, n_slices))      # I - U^H U
    S = csub(I, ozaki_gemm_complex(VH, V, n_slices))      # I - V^H V
    T = ozaki_gemm_complex(UH, ozaki_gemm_complex(A, V, n_slices), n_slices)  # U^H A V
    if debug:
        msg = (f"    [dd] ||R||={_frob_dd(R):.2e} ||S||={_frob_dd(S):.2e} "
               f"||offT||={_frob_dd_offdiag(T):.2e}")
        nn = _DBG.get('nnull')
        if nn:
            L = n - nn
            re = T['reh'] + T['rel']; im = T['imh'] + T['iml']
            m = re*re + im*im; md = m.copy(); md[xp.arange(n), xp.arange(n)] = 0.0
            LL = float(xp.sqrt(xp.sum(md[:L, :L]))); LN = float(xp.sqrt(xp.sum(m[:L, L:])))
            NL = float(xp.sqrt(xp.sum(m[L:, :L]))); NN = float(xp.sqrt(xp.sum(md[L:, L:])))
            msg += f"  ddblocks(LL,LN,NL,NN)=({LL:.1e},{LN:.1e},{NL:.1e},{NN:.1e})"
        print(msg)

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
    beta  = cadd(tbar, cscale_ddreal(S, sj_h, sj_l))                # t_bar_ji + sigma_j s_ij
    # NB: beta uses sigma_j (not sigma_i) -- required for f_ij + conj(f_ji) = r_ij (U-orthogonality).
    # sigma_i here gives f+f^H = r - s, which swaps R<->S off-diagonals each step (no convergence).
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

    if debug:
        idxd = xp.arange(n)
        rdd = (R['reh'] + R['rel'])**2 + (R['imh'] + R['iml'])**2
        dR = float(xp.sqrt(xp.sum(rdd[idxd, idxd])))
        oR = float(xp.sqrt(xp.sum(rdd) - xp.sum(rdd[idxd, idxd])))
        print(f"    [dd] ||F||={_frob_dd(F):.2e} ||G||={_frob_dd(G):.2e} "
              f"||diagR||={dR:.2e} ||offR||={oR:.2e}")

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

def mxp_svd_dd(A_dd, A_c128_hi, n_slices=6, max_iter=10, n_null=None, verbose=True, V_seed=None):
    """Same, but A given as a dd-complex dict (the extended-precision matrix M).
    V_seed: optional (n,k) fp64 array of approximate right-null vectors (seed continuation) that
    REPLACE the k smallest-sigma columns of the FP64 initial V — needed past the FP64 wall, where
    the initial SVD's smallest subspace is roundoff noise with ~zero overlap with the true forms."""
    return _mxp_svd_core(A_dd, A_c128_hi, n_slices, max_iter, n_null, verbose, V_seed)

def _mxp_svd_core(A, A_c128_hi, n_slices, max_iter, n_null, verbose, V_seed=None):
    n = A['reh'].shape[0]
    eps_lp = 2.0**-53; eps_hp = 2.0**-106
    U0, s0, Vh0 = xp.linalg.svd(xp.asarray(A_c128_hi))     # initial low-precision SVD (GPU if available)
    U = from_c128(U0); V = from_c128(Vh0.conj().T)
    sigma_h = s0
    if V_seed is not None:
        # inject the continuation seed into the k smallest-sigma columns of V (fp64 Gram-Schmidt:
        # project the seed off the retained columns, orthonormalize, splice back — the retained
        # columns are well above the wall so fp64 is ample; refinement then polishes to dd).
        k = V_seed.shape[1]
        s0n = np.asarray(s0.get() if _GPU else s0)
        Jn = list(np.argsort(s0n)[:k])
        rest = [i for i in range(n) if i not in Jn]
        Vc = to_c128(V); Vc = np.asarray(Vc.get() if _GPU else Vc)
        B = Vc[:, rest]
        S = np.asarray(V_seed, dtype=complex).copy()
        for _ in range(2):
            S = S - B @ (B.conj().T @ S)                  # project out retained subspace (twice)
        Sq, _ = np.linalg.qr(S)                            # orthonormalize the k seed vectors
        Vc[:, Jn] = Sq
        V = from_c128(xp.asarray(Vc))
        if verbose:
            print(f"  seeded {k} null columns (indices {Jn[:3]}...); refining from continuation seed")
    prev = np.inf
    for it in range(max_iter):
        U, V, (sigma_h, sigma_l), omega = mxp_svd_step(A, U, V, eps_lp, eps_hp, n_slices)
        smax = float(xp.max(sigma_h))
        if verbose:
            print(f"  iter {it}: omega={omega:.3e}  target={16*n*eps_hp*smax:.3e}  smin={float(xp.min(sigma_h)):.3e}")
        if omega <= 16 * n * eps_hp * smax:
            break
        if omega > 0.5 * prev:        # stagnation: well-separated part converged, clusters remain
            break
        prev = omega
    # cluster-refine the null space
    J, ss, cut = detect_null_cluster(sigma_h, n_expected=n_null)
    if verbose:
        print(f"  null cluster: {len(J)} vecs; sigma spectrum tail {ss[:cut+2]}")
    cluster_rr(A, U, V, J, n_slices)
    return U, V, sigma_h, J

# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Self-test: matrix with a KNOWN 2-dim near-null subspace and a huge dynamic range
    # (sigma_max ~ 1e8, well-separated large block down to 1e2, tiny sigma ~ 1e-9), mimicking M.
    # The FP64 SVD cannot resolve the tiny subspace; refinement recovers it. Crucially A is
    # fed in DOUBLE-DOUBLE (as the real M will be): feeding A in fp64 caps the null vectors at
    # ~eps*||A||/gap ~ 1e-11 (the fp64-A's own null-vector error), NOT an algorithm limit.
    import mpmath as mp
    mp.mp.dps = 50
    rng = np.random.default_rng(0)
    n = 16
    Qr, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))
    Qc, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))
    sig = np.concatenate([np.geomspace(1e8, 1e2, n-2), [3e-9, 1e-9]])
    Vtrue = Qc[:, -2:]
    # A = Qr diag(sig) Qc^H at 50 digits, split to double-double
    Qrm = mp.matrix([[mp.mpc(Qr[i, j].real, Qr[i, j].imag) for j in range(n)] for i in range(n)])
    Qcm = mp.matrix([[mp.mpc(Qc[i, j].real, Qc[i, j].imag) for j in range(n)] for i in range(n)])
    D = mp.matrix(n, n)
    for k in range(n):
        D[k, k] = mp.mpf(float(sig[k]))
    Am = Qrm * D * Qcm.H
    reh, rel, imh, iml = (np.zeros((n, n)) for _ in range(4)); Ahi = np.zeros((n, n), complex)
    for i in range(n):
        for j in range(n):
            z = Am[i, j]; rh = float(z.real); ih = float(z.imag)
            reh[i, j] = rh; rel[i, j] = float(z.real - mp.mpf(rh))
            imh[i, j] = ih; iml[i, j] = float(z.imag - mp.mpf(ih)); Ahi[i, j] = rh + 1j*ih
    A_dd = cnew(xp.asarray(reh), xp.asarray(rel), xp.asarray(imh), xp.asarray(iml))

    def subspace_err(Vapprox):
        P = Vapprox @ Vapprox.conj().T
        return np.linalg.norm(P @ Vtrue - Vtrue)
    _, s_fp, Vh_fp = np.linalg.svd(Ahi)
    print(f"FP64 SVD: smallest two sigma = {s_fp[-2:]}  (true 3e-9, 1e-9)")
    print(f"FP64 null-subspace error = {subspace_err(Vh_fp.conj().T[:, -2:]):.2e}")
    print("Refining (Ogita-Aishima, dd input):")
    U, V, sig_r, J = mxp_svd_dd(A_dd, Ahi, n_slices=6, max_iter=8, n_null=2, verbose=True)
    Vc = to_c128(V); Vc = np.asarray(Vc.get() if _GPU else Vc)
    err = subspace_err(Vc[:, list(J)])
    print(f"refined null-subspace error = {err:.2e}  ({'PASS' if err < 1e-13 else 'FAIL'}; "
          f"~1e-15 = fp64 measurement floor of to_c128(V))")
