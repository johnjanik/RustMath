"""Physical-subspace selectors for the [2,12,5] null space (per OBSTRUCTION note §3, §5).
Stop taking the 9 smallest singular vectors; instead select a 9-dim subspace with BOUNDED
recovered coefficients b_n = rho^-n y_n.  Success signal = REAL Hauptmodul with X[1]=kappa^2=0.0346.

  raw          : the 9 smallest singular vectors (the failing baseline)
  band_tail    : §3 -- candidate band of K smallest, then the 9 directions minimizing the
                 tail-weighted coefficient penalty  min ||L y||,  L = diag(w_n rho^{N-n})
  anchored     : §5 -- freeze trusted low-N b-coefficients, solve only the tail (regularized)

Usage: python3 physical_selector.py <N2500.bin> [anchor_N1500.bin]"""
import numpy as np, sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import mapkit
from read_ext import read_ext
try:
    import cupy as xp; _GPU = True
except Exception:
    import numpy as xp; _GPU = False

rho = mapkit.rho; nforms = mapkit.nforms

def load_hi(path):
    dim, nl, re, im = read_ext(path)
    return dim, (re[0] + 1j*im[0])

def fp64_svd_ascending(Ahi):
    U, s, Vh = xp.linalg.svd(xp.asarray(Ahi))
    s = np.asarray(s.get() if _GPU else s); Vh = np.asarray(Vh.get() if _GPU else Vh)
    order = np.argsort(s)
    return s[order], Vh[order, :]          # ascending sigma, matching Vh rows

# ---- selectors ----
def sel_raw(s_asc, Vh_asc, dim):
    return Vh_asc[:nforms].conj().T        # dim x 9, the smallest-9 right singular vectors

def tail_weights(N, n0_frac=0.65, power=4.0):
    n = np.arange(N+1, dtype=float); n0 = n0_frac*N
    t = np.maximum(0.0, (n - n0)/max(1.0, N - n0))
    w = 1.0 + t**power
    return w * (rho**(N - n))               # w_n * rho^{N-n} ; L y = w_n rho^N b_n  (tail-weighted |b|)

def sel_band_tail(s_asc, Vh_asc, dim, K=80, n0_frac=0.65, power=4.0):
    N = dim - 1
    Z = Vh_asc[:K].conj().T                 # dim x K candidate band (K smallest sigma)
    L = tail_weights(N, n0_frac, power)
    LZ = L[:, None] * Z
    H = LZ.conj().T @ LZ                     # K x K Hermitian tail-penalty Gram
    mu, W = np.linalg.eigh(H)               # ascending eigenvalues
    Y = Z @ W[:, :nforms]                    # 9 lowest-penalty directions in the band
    Y, _ = np.linalg.qr(Y)
    return Y, mu[:nforms]

def anchor_bspace(anchor_path, N0):
    """Trusted low-N physical b-coefficients B0 (N0+1 x 9) from the below-wall FP64 SVD."""
    dim0, Ahi0 = load_hi(anchor_path)
    s0, Vh0 = fp64_svd_ascending(Ahi0)
    Y0 = Vh0[:nforms].conj().T              # dim0 x 9 (clean below the wall)
    b0 = (rho**(-np.arange(dim0)))[:, None] * Y0   # b_n = rho^-n y_n
    return b0[:N0+1, :]                      # anchor only the trusted low part

def sel_anchored(Ahi, dim, B0_bspace, N0, lam=1e-20, tail_power=2.0):
    """§5 anchored prolongation: freeze b_{<=N0}=B0, solve tail b_{>N0} by regularized LS."""
    N = dim - 1; m = B0_bspace.shape[1]
    Dyb = rho**np.arange(N+1)               # y_n = rho^n b_n
    Y = np.zeros((N+1, m), complex)
    Y[:N0+1, :] = Dyb[:N0+1, None] * B0_bspace
    tail = np.arange(N0+1, N+1); nt = len(tail)
    if nt == 0:
        return np.linalg.qr(Y)[0]
    T = np.zeros((N+1, nt), complex)        # tail b -> y
    T[tail, np.arange(nt)] = Dyb[tail]
    MT = Ahi @ T
    tau = (tail - (N0+1))/max(1, N - N0); w = 1.0 + tau**tail_power
    Amat = MT.conj().T @ MT + lam*np.diag(w**2)
    for j in range(m):
        t = np.linalg.solve(Amat, -(MT.conj().T @ (Ahi @ Y[:, j])))
        Y[tail, j] += Dyb[tail] * t
    return np.linalg.qr(Y)[0]

# ---- experiment ----
def show(res):
    v = res.get('valuations', [])
    okval = "y" if v[:9] == list(range(9)) else "N"
    print(f"  {res['label']:<26} reality(Im/Re)={res['reality']:.2e}  "
          f"|X1-k^2|={res['x1_err']:.2e}  o12={res['o12']:.3e}  ech0-8={okval}")

if __name__ == "__main__":
    path = sys.argv[1]
    anchor = sys.argv[2] if len(sys.argv) > 2 else None
    dim, Ahi = load_hi(path); N = dim - 1
    print(f"matrix N={N} dim={dim}  rho^N={rho**N:.2e}  target: reality<<1, |X1-k^2|<1e-4")
    s_asc, Vh_asc = fp64_svd_ascending(Ahi)

    print("\n[baseline]")
    show(mapkit.evaluate(sel_raw(s_asc, Vh_asc, dim), dim, "raw smallest-9"))

    print("\n[§3 band+tail selector — scan K, power]")
    for K in (40, 80, 120):
        for power in (2.0, 4.0, 8.0):
            Y, pen = sel_band_tail(s_asc, Vh_asc, dim, K=K, power=power)
            show(mapkit.evaluate(Y, dim, f"band K={K} pow={power:g}"))

    if anchor:
        print(f"\n[§5 anchored prolongation from {anchor}]")
        for N0 in (1200, 1500):
            B0 = anchor_bspace(anchor, N0)
            for lam in (1e-24, 1e-18):
                Y = sel_anchored(Ahi, dim, B0, N0, lam=lam)
                show(mapkit.evaluate(Y, dim, f"anchored N0={N0} lam={lam:g}"))
