"""P4 decisive experiment: does the GLOBAL Veronese Hauptmodul X *range* at the order-12
vertex, where the order-2 chart confined it to 9.9x?

Self-contained (FP64 only -- accuracy is not the question here, spread is):
  1. order-2 forms  from m_N*.bin      (center z_a = i,   rho_a = 0.990605)
  2. order-12 forms from m_order12_*.bin (center z_b = mu i, rho_b = 0.997905) via FP64 SVD null space
  3. both are bases of the SAME 9-dim S_4(Gamma'); align by T (F^b = T F^a) at overlap points on
     the imaginary axis between z_a and z_b, where both w-series converge.
  4. global coordinate extractor = A (order-2 Veronese) composed with T^{-1}; apply to order-12
     form values near z_b and read X off the Veronese ratios.
  5. compare the X range at order-12 sample points vs the order-2 confinement.

Success signal: X spans orders of magnitude more range at order-12 than the 9.9x at order-2, with
small Veronese residuals -- i.e. the map becomes visible where the 12^2 ramification lives.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__) or ".")
import numpy as np
from read_ext import read_ext
from veronese_coordinate import build_veronese_normalizer, recover_x_from_veronese_point

K = 4                      # weight
RHO_A = 0.990605           # order-2 domain radius (mapkit)
RHO_B = 0.997905           # order-12 domain radius (from the center=b assembly)
I = 1j
_Lam = math.cos(math.pi / 5) / math.sin(math.pi / 12)
MU = _Lam + math.sqrt(_Lam * _Lam - 1)      # z_b = MU i ; (2,12,5) => MU ~ 6.088
ZB = MU * I

wp_a = lambda z: (z - I) / (z + I)           # order-2 chart
wp_b = lambda z: (z - ZB) / (z + MU * I)     # order-12 chart (z - ZB)/(z - conj ZB), conj ZB = -MU i


def load_M(path):
    dim, nl, re, im = read_ext(path)
    return dim, sum(re) + 1j * sum(im)


def null_bvectors(M, rho, kdim=9):
    """FP64 SVD -> kdim smallest right singular vectors (null space, y-space) -> b = rho^-n y."""
    U, s, Vh = np.linalg.svd(M)
    Y = Vh[-kdim:].conj().T                   # (dim, kdim), columns = null vectors
    scale = rho ** (-np.arange(M.shape[0]))
    return Y * scale[:, None], s


def form_values(B, wfun, zs):
    """actual forms f_j(z) = (1-w)^K sum_n B[n,j] w^n, evaluated at each z in zs -> (kforms, len zs)."""
    dim = B.shape[0]
    F = np.empty((B.shape[1], len(zs)), complex)
    for m, z in enumerate(zs):
        w = wfun(z)
        wn = np.empty(dim, complex); wn[0] = 1.0
        for n in range(1, dim):
            wn[n] = wn[n - 1] * w
        F[:, m] = ((1 - w) ** K) * (B.T @ wn)
    return F


def u_coeff_matrix(B, degree=8):
    """order-2 forms in u = w^2: c_j = (1-w)^4 * B[:,j]; C[i,j] = c_j[2i]  (i,j = 0..degree)."""
    omw = np.array([(-1) ** j * math.comb(K, j) for j in range(K + 1)], complex)
    C = np.zeros((degree + 1, B.shape[1]), complex)
    for j in range(B.shape[1]):
        c = np.convolve(omw, B[:, j])
        C[:, j] = [c[2 * i] for i in range(degree + 1)]   # even coeffs = u-series
    return C


def rng(xs):
    a = np.abs([x for x in xs if np.isfinite(x)])
    return a.max() / max(a.min(), 1e-300), a.min(), a.max()


if __name__ == "__main__":
    N_A = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    ORD12 = sys.argv[2] if len(sys.argv) > 2 else "/home/john/sweep_2_12_5/m_order12_N3000.bin"
    print(f"MU = {MU:.6f}   z_b = {MU:.4f}i   rho_a={RHO_A}  rho_b={RHO_B}")

    # --- recover both form bases ---
    dimA, MA = load_M(f"/home/john/sweep_2_12_5/m_N{N_A}.bin")
    BA, sA = null_bvectors(MA, RHO_A)
    dimB, MB = load_M(ORD12)
    BB, sB = null_bvectors(MB, RHO_B)
    print(f"order-2  N={N_A}: dim={dimA}  sv[-12:-6]={np.array2string(sA[-12:-6], precision=1)}"
          f"  gap sv[-10]/sv[-9]={sA[-10]/sA[-9]:.1e}")
    print(f"order-12      : dim={dimB}  sv[-12:-6]={np.array2string(sB[-12:-6], precision=1)}"
          f"  gap sv[-10]/sv[-9]={sB[-10]/sB[-9]:.1e}")

    # --- Veronese normalizer A from order-2 leading u-jets ---
    C = u_coeff_matrix(BA, degree=8)
    A = np.linalg.inv(C.T)
    print(f"Veronese A: cond(C)={np.linalg.cond(C):.2e}")

    # --- alignment T on the imaginary axis between the two vertices ---
    ts = np.linspace(1.6, 5.6, 24)
    zs = [t * I for t in ts]
    Fa = form_values(BA, wp_a, zs)
    Fb = form_values(BB, wp_b, zs)
    # split fit / holdout
    fit, hold = slice(0, 16), slice(16, 24)
    T = Fb[:, fit] @ np.linalg.pinv(Fa[:, fit])
    res_fit = np.linalg.norm(T @ Fa[:, fit] - Fb[:, fit]) / np.linalg.norm(Fb[:, fit])
    res_hold = np.linalg.norm(T @ Fa[:, hold] - Fb[:, hold]) / np.linalg.norm(Fb[:, hold])
    print(f"alignment T: fit resid={res_fit:.2e}  HOLDOUT resid={res_hold:.2e}  cond(T)={np.linalg.cond(T):.1e}")

    Tinv = np.linalg.inv(T)
    Wb = A @ Tinv                       # global-coordinate extractor for order-12 values

    # --- baseline: order-2 X range near z_a (the confinement) ---
    z2 = [ (1 + r) * I + 0.15 * r for r in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5) ]
    F2 = form_values(BA, wp_a, z2)
    X2 = [recover_x_from_veronese_point(A @ F2[:, m])["x"] for m in range(F2.shape[1])]
    r2, lo2, hi2 = rng(X2)
    print(f"\norder-2  X near z_a:  range={r2:.1f}x   |X| in [{lo2:.2e}, {hi2:.2e}]")

    # --- payoff: order-12 X near z_b (where the map ramifies) ---
    print("\norder-12 X near z_b (global coordinate via A T^-1):")
    print(f"  {'z':>16} {'|w_b|':>7} {'recovered X':>30} {'ver.resid':>10}")
    Xb = []
    for r in (0.03, 0.06, 0.1, 0.15, 0.25, 0.4, 0.6):
        z = ZB * (1 - r) + 0.2 * r * I        # points marching away from z_b
        Fz = form_values(BB, wp_b, [z])[:, 0]
        rec = recover_x_from_veronese_point(Wb @ Fz)
        Xb.append(rec["x"])
        wb = wp_b(z)
        xr = rec["x"]
        print(f"  {z.imag:+.3f}i{'':>6} {abs(wb):7.3f} {xr.real:+.5f}{xr.imag:+.5f}i   {rec['residual']:.2e}")
    rb, lob, hib = rng(Xb)
    print(f"\norder-12 X range={rb:.1f}x   |X| in [{lob:.2e}, {hib:.2e}]")
    print(f"\n=> order-12 / order-2 range ratio = {rb / max(r2,1e-30):.1f}x   "
          f"({'RANGES -- map is visible' if rb > 5 * r2 else 'still confined'})")
