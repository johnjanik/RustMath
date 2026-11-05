"""Ozaki-scheme error-free FP64 GEMM (double-double accurate) on the GPU (cupy).

Splits each operand into low-bit "slices" so slice-products accumulate EXACTLY in FP64,
then sums the partial GEMMs in double-double. This is the high-precision GEMM engine the
Ogita-Aishima refinement (Alg 3/4 of the NVIDIA paper) is dominated by.

Operands and results are double-double: value = hi + lo (two f64 arrays, |lo|~1e-16|hi|).
Validated against mpmath ground truth (see __main__)."""
import numpy as np
try:
    import cupy as xp
    _GPU = True
except Exception:
    import numpy as xp
    _GPU = False

# ---------- error-free transforms (array, cupy or numpy) ----------
_SPLIT = 2.0**27 + 1.0
def two_sum(a, b):
    s = a + b; bb = s - a
    return s, (a - (s - bb)) + (b - bb)
def _split2(a):
    c = _SPLIT * a; hi = c - (c - a); return hi, a - hi
def two_prod(a, b):
    p = a * b
    ah, al = _split2(a); bh, bl = _split2(b)
    return p, ((ah*bh - p) + ah*bl + al*bh) + al*bl

# ---------- double-double scalar ops on arrays ----------
def dd_add(xh, xl, yh, yl):
    s, e = two_sum(xh, yh); e = e + xl + yl
    zh, zl = two_sum(s, e); return zh, zl
def dd_sub(xh, xl, yh, yl):
    return dd_add(xh, xl, -yh, -yl)

# ---------- Ozaki split of a dd real matrix along contraction axis ----------
def _rowmax_along(A_abs, axis):
    return xp.max(A_abs, axis=axis, keepdims=True)

def ozaki_split(hi, lo, k_dim, n_slices, axis):
    """Split dd matrix (hi,lo) into <= n_slices f64 slices whose products (with a
    matching split of the other operand) accumulate exactly over `k_dim`.
    axis = the contraction axis (1 for the left operand A over its columns,
           0 for the right operand B over its rows)."""
    beta = int(np.floor((53 - np.ceil(np.log2(max(k_dim, 2)))) / 2))  # bits per slice
    rh = hi.copy(); rl = lo.copy()                                    # running dd residual
    slices = []
    for _ in range(n_slices):
        mu = _rowmax_along(xp.abs(rh) + xp.abs(rl), axis)             # per-row/col max
        if float(xp.max(mu)) == 0.0:
            break
        mu = xp.where(mu == 0.0, 1.0, mu)
        tau = xp.exp2(xp.ceil(xp.log2(mu)))                           # 2^ceil(log2 mu)
        sigma = tau * (2.0 ** (53 - beta))                           # extraction constant
        s = (rh + sigma) - sigma                                      # top ~beta bits of residual
        slices.append(s)
        rh, rl = dd_sub(rh, rl, s, xp.zeros_like(s))                 # residual -= s (exact-ish, dd)
    return slices

def ozaki_gemm_real(Ahi, Alo, Bhi, Blo, n_slices=6, keep=None):
    """dd C = A @ B for real dd matrices. Returns (Chi, Clo)."""
    k = Ahi.shape[1]
    As = ozaki_split(Ahi, Alo, k, n_slices, axis=1)
    Bs = ozaki_split(Bhi, Blo, k, n_slices, axis=0)
    if keep is None:
        keep = n_slices                                              # keep pairs with p+q < keep
    Ch = xp.zeros((Ahi.shape[0], Bhi.shape[1]), dtype=xp.float64)
    Cl = xp.zeros_like(Ch)
    # accumulate partial products in ascending order of (p+q) so the dd sum stays accurate
    for tot in range(0, len(As) + len(Bs) - 1):
        if tot >= keep:
            break
        for p in range(len(As)):
            q = tot - p
            if q < 0 or q >= len(Bs):
                continue
            P = As[p] @ Bs[q]                                         # EXACT fp64 gemm (by construction)
            Ch, Cl = dd_add(Ch, Cl, P, xp.zeros_like(P))
    return Ch, Cl

def ozaki_gemm_complex(A, B, n_slices=6, keep=None):
    """dd complex C = A @ B. A,B are dd-complex dicts {'reh','rel','imh','iml'}.
    Uses 4 real dd GEMMs (straightforward; 3-mult Karatsuba possible later)."""
    def rg(Xh, Xl, Yh, Yl): return ozaki_gemm_real(Xh, Xl, Yh, Yl, n_slices, keep)
    # (Ar+iAi)(Br+iBi) = (ArBr - AiBi) + i(ArBi + AiBr)
    rr_h, rr_l = rg(A['reh'], A['rel'], B['reh'], B['rel'])
    ii_h, ii_l = rg(A['imh'], A['iml'], B['imh'], B['iml'])
    ri_h, ri_l = rg(A['reh'], A['rel'], B['imh'], B['iml'])
    ir_h, ir_l = rg(A['imh'], A['iml'], B['reh'], B['rel'])
    reh, rel = dd_sub(rr_h, rr_l, ii_h, ii_l)
    imh, iml = dd_add(ri_h, ri_l, ir_h, ir_l)
    return {'reh': reh, 'rel': rel, 'imh': imh, 'iml': iml}

if __name__ == "__main__":
    import mpmath as mp
    mp.mp.dps = 50
    rng = np.random.default_rng(1)
    m, k, n = 60, 500, 40
    # build a dd real matrix: value spanning a big dynamic range (like M)
    Ah = (rng.standard_normal((m, k)) * 2.0**rng.integers(-20, 20, (m, k))).astype(np.float64)
    Bh = (rng.standard_normal((k, n)) * 2.0**rng.integers(-20, 20, (k, n))).astype(np.float64)
    # give them a lo part (simulate dd input)
    Al = (Ah * 1e-17 * rng.standard_normal((m, k))).astype(np.float64)
    Bl = (Bh * 1e-17 * rng.standard_normal((k, n))).astype(np.float64)
    Ahx, Alx = xp.asarray(Ah), xp.asarray(Al)
    Bhx, Blx = xp.asarray(Bh), xp.asarray(Bl)
    for ns in (3, 4, 6):
        Ch, Cl = ozaki_gemm_real(Ahx, Alx, Bhx, Blx, n_slices=ns)
        Ch = np.asarray(Ch.get() if _GPU else Ch); Cl = np.asarray(Cl.get() if _GPU else Cl)
        # ground truth C = (Ah+Al)@(Bh+Bl)
        worst = 0.0
        for i in range(0, m, 7):
            for j in range(0, n, 5):
                ref = mp.mpf(0)
                for t in range(k):
                    ref += (mp.mpf(float(Ah[i,t]))+mp.mpf(float(Al[i,t]))) * \
                           (mp.mpf(float(Bh[t,j]))+mp.mpf(float(Bl[t,j])))
                got = mp.mpf(float(Ch[i,j])) + mp.mpf(float(Cl[i,j]))
                den = abs(ref) if abs(ref) > 0 else mp.mpf(1)
                worst = max(worst, float(abs(got-ref)/den))
        naive = Ah @ Bh
        nworst = 0.0
        for i in range(0, m, 7):
            for j in range(0, n, 5):
                ref = mp.mpf(0)
                for t in range(k):
                    ref += (mp.mpf(float(Ah[i,t]))+mp.mpf(float(Al[i,t]))) * (mp.mpf(float(Bh[t,j]))+mp.mpf(float(Bl[t,j])))
                nworst = max(nworst, float(abs(mp.mpf(float(naive[i,j]))-ref)/(abs(ref) if abs(ref)>0 else mp.mpf(1))))
        print(f"n_slices={ns}: ozaki rel err={worst:.2e}   (naive fp64 gemm={nworst:.2e})")
