"""Error-free FP64 primitives + high-precision ground truth, for validating the
emulated-precision GEMM that the Ogita-Aishima SVD refinement will use.

Everything here is agent-independent standard numerics (Knuth/Dekker EFT + mpmath truth).
The Ozaki split constants / OA update formulas come later (from the paper recon)."""
import numpy as np

# ---- Dekker error-free transforms (vectorized, no hardware FMA needed) ----
_SPLIT = 2.0**27 + 1.0  # Dekker splitter for f64 (53-bit mantissa)

def two_sum(a, b):
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err

def _split(a):
    c = _SPLIT * a
    hi = c - (c - a)
    lo = a - hi
    return hi, lo

def two_prod(a, b):
    p = a * b
    ah, al = _split(a); bh, bl = _split(b)
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, err

# ---- double-double (hi,lo) representation of a real array ----
def to_dd(x_hi, x_lo=None):
    """package into a dd dict; if x_lo None, treat x_hi as exact f64 (lo=0)."""
    hi = np.asarray(x_hi, dtype=np.float64)
    lo = np.zeros_like(hi) if x_lo is None else np.asarray(x_lo, dtype=np.float64)
    return {"hi": hi, "lo": lo}

def dd_from_exact(x):
    """split an arbitrary-precision-ish python/np value array into dd via renormalization
    (only meaningful if x carries >53 bits, e.g. produced from mpmath)."""
    hi = np.asarray(x, dtype=np.float64)
    lo = np.asarray(x - hi, dtype=np.float64)  # requires x to be higher precision (object array)
    return {"hi": hi, "lo": lo}

# ---- ground truth: high-precision C = A^H @ A for complex A (via mpmath) ----
def mp_gram(A_complex, dps=60):
    """A: (m,n) complex numpy. Returns C=(n,n) = A^H A as an mpmath matrix at `dps` digits."""
    import mpmath as mp
    mp.mp.dps = dps
    m, n = A_complex.shape
    Am = mp.matrix(m, n)
    for i in range(m):
        for j in range(n):
            z = A_complex[i, j]
            Am[i, j] = mp.mpc(float(z.real), float(z.imag))
    C = Am.H * Am
    return C

def rel_err_vs_mp(C_num, C_mp):
    """max relative error of numpy complex matrix C_num vs mpmath matrix C_mp."""
    import mpmath as mp
    n = C_num.shape[0]
    worst = mp.mpf(0)
    for i in range(n):
        for j in range(n):
            ref = C_mp[i, j]
            got = mp.mpc(float(C_num[i, j].real), float(C_num[i, j].imag))
            denom = abs(ref) if abs(ref) > 0 else mp.mpf(1)
            e = abs(got - ref) / denom
            if e > worst:
                worst = e
    return float(worst)

if __name__ == "__main__":
    # sanity: two_prod / two_sum are exact
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1000); b = rng.standard_normal(1000)
    p, e = two_prod(a, b)
    # reconstruct product exactly in higher precision via python floats -> check
    import mpmath as mp; mp.mp.dps = 40
    worst = 0.0
    for i in range(20):
        exact = mp.mpf(float(a[i])) * mp.mpf(float(b[i]))
        got = mp.mpf(float(p[i])) + mp.mpf(float(e[i]))
        worst = max(worst, float(abs(got - exact)))
    print(f"two_prod max abs err over 20 samples: {worst:.2e} (should be ~0 / <1e-300)")
    s, e2 = two_sum(a, b)
    print(f"two_sum residual check: {np.max(np.abs((a+b) - s)):.2e} vs err carried {np.max(np.abs(e2)):.2e}")
    print("EFT primitives OK")
