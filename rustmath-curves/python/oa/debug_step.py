"""Isolate mxp_svd_step on an EASY matrix (sigma O(1), well-separated -> only branch A).
Prints orthogonality (||R||,||S||) and diagonality (||offdiag T||) each step so we can see
which one fails to decrease. gitignored (*.py) -- scratch debug harness, not shipped."""
import numpy as np
np.set_printoptions(precision=3, suppress=False, linewidth=140)
from mxpsvd import mxp_svd_step
from ddcx import from_c128, to_c128
try:
    import cupy as xp; _GPU = True
except Exception:
    import numpy as xp; _GPU = False

def diagnostics(Ac, U, V):
    Uc = to_c128(U); Vc = to_c128(V)
    Uc = np.asarray(Uc.get() if _GPU else Uc); Vc = np.asarray(Vc.get() if _GPU else Vc)
    n = Ac.shape[0]
    R = np.eye(n) - Uc.conj().T @ Uc
    S = np.eye(n) - Vc.conj().T @ Vc
    T = Uc.conj().T @ Ac @ Vc
    offT = T.copy(); np.fill_diagonal(offT, 0.0)
    return np.linalg.norm(R), np.linalg.norm(S), np.linalg.norm(offT), T

def offT_blocks(T, nnull):
    L = T.shape[0] - nnull
    off = T.copy(); np.fill_diagonal(off, 0.0)
    LL = np.linalg.norm(off[:L, :L])
    LN = np.linalg.norm(T[:L, L:])              # large-row, null-col
    NL = np.linalg.norm(T[L:, :L])              # null-row, large-col
    NN = np.linalg.norm(off[L:, L:])
    return LL, LN, NL, NN

import os
n = 16
rng = np.random.default_rng(0)
Qr, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))
Qc, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))
if os.environ.get("HARD"):
    lo = float(os.environ.get("SLO", "1e2"))               # smallest large sigma (>> thr)
    sig_true = np.concatenate([np.geomspace(1e8, lo, n-2), [3e-9, 1e-9]])  # huge range + null cluster
else:
    sig_true = np.linspace(3.0, 1.0, n)                 # O(1), well separated
A = (Qr * sig_true) @ Qc.conj().T
A = A.astype(np.complex128)
A_dd = from_c128(xp.asarray(A))

U0, s0, Vh0 = np.linalg.svd(A)
U = from_c128(xp.asarray(U0)); V = from_c128(xp.asarray(Vh0.conj().T))
eps_lp = 2.0**-53; eps_hp = 2.0**-106; n_slices = 6

n_null = 2 if os.environ.get("HARD") else 0
Vtrue = Qc[:, n-n_null:] if n_null else None
def span_err(V):
    Vc = to_c128(V); Vc = np.asarray(Vc.get() if _GPU else Vc)
    Vn = Vc[:, n-n_null:]                 # null block = last n_null columns (smallest sigma)
    P = Vn @ Vn.conj().T
    return np.linalg.norm(P @ Vtrue - Vtrue)

r, s, offt, T = diagnostics(A, U, V)
print(f"init : ||R||={r:.2e} ||S||={s:.2e} ||offT||={offt:.2e}"
      + (f" span={span_err(V):.2e} blocks(LL,LN,NL,NN)={tuple(f'{x:.1e}' for x in offT_blocks(T,n_null))}" if n_null else ""))
print(f"       FP64 sigma err = {np.linalg.norm(np.sort(s0)-np.sort(sig_true)):.2e}")
for it in range(6):
    U, V, (sh, sl), omega = mxp_svd_step(A_dd, U, V, eps_lp, eps_hp, n_slices, debug=False)
    r, s, offt, T = diagnostics(A, U, V)
    sig = np.asarray(sh.get() if _GPU else sh)
    serr = np.linalg.norm(np.sort(sig) - np.sort(sig_true))
    extra = (f" span={span_err(V):.2e} blocks={tuple(f'{x:.1e}' for x in offT_blocks(T,n_null))}") if n_null else ""
    print(f"it {it}: omega={omega:.2e} ||R||={r:.2e} ||S||={s:.2e} ||offT||={offt:.2e} sig_err={serr:.2e}{extra}")
