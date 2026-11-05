"""Decisive diagnostic: measure the dd Rayleigh quotient ||M v||/||v|| for
  (a) the KNOWN form seed (N1400 forms, zero-padded to dim 2500), and
  (b) the FP64 initial SVD's smallest-9 subspace (what the refinement chases).
If (a) ~ 1e-11 (= rho^N/||y||) and (b) ~ 1e-17, then M has spurious near-null directions
BELOW the forms and 'smallest-9' is the wrong selection. If (a) ~ (b), forms really are there.
Usage: python3 diag_sigma.py <N2500_dd.bin> <N1400_dd.bin>"""
import numpy as np, sys, math
sys.path.insert(0, '/tmp/claude-1000/-home-john-inverse-galois-M23/24542307-282e-4596-89f8-915a13a1d65e/scratchpad')
from read_ext import read_ext
from ozaki import ozaki_gemm_complex
from ddcx import cnew, to_c128, from_c128
try:
    import cupy as xp; _GPU = True
except Exception:
    import numpy as xp; _GPU = False

n2500, n1400 = sys.argv[1], sys.argv[2]
nforms = 9; rho = 0.990605

dim, nl, re, im = read_ext(n2500)
A_dd = cnew(xp.asarray(re[0]), xp.asarray(re[1]), xp.asarray(im[0]), xp.asarray(im[1]))
Ahi = re[0] + 1j*im[0]
print(f"N2500 matrix dim={dim}")

def col_sigmas(Vnp, label):
    """||M v_j|| in dd for each column of Vnp (dim x k, fp64), v_j already ~unit."""
    Vdd = from_c128(xp.asarray(Vnp.astype(complex)))
    MV = ozaki_gemm_complex(A_dd, Vdd, 6)
    MVc = to_c128(MV); MVc = np.asarray(MVc.get() if _GPU else MVc)
    s = np.linalg.norm(MVc, axis=0)
    print(f"  {label}: ||M v_j|| = {np.array2string(np.sort(s), precision=3, formatter={'float': lambda x: f'{x:.2e}'})}")
    return s

# (a) the KNOWN form seed: N1400 FP64 forms, zero-padded
sd, snl, sre, sim = read_ext(n1400)
su, ss, svh = np.linalg.svd(sre[0] + 1j*sim[0])
Vs = svh.conj().T[:, np.argsort(ss)[:nforms]]
seed = np.zeros((dim, nforms), complex); seed[:sd, :] = Vs
seed, _ = np.linalg.qr(seed)                 # orthonormalize the padded seed
print(f"\n(a) KNOWN forms (N1400 padded {sd}->{dim}):")
col_sigmas(seed, "form seed")

# (b) FP64 initial SVD smallest-9 of the N2500 matrix (what refinement chases)
u2, s2, vh2 = xp.linalg.svd(xp.asarray(Ahi))
s2n = np.asarray(s2.get() if _GPU else s2); vh2n = np.asarray(vh2.get() if _GPU else vh2)
Vg = vh2n.conj().T[:, np.argsort(s2n)[:nforms]]
print(f"\n(b) FP64 smallest-9 of N2500 (fp64 reports sigma={np.sort(s2n)[:nforms]}):")
col_sigmas(Vg, "fp64 null-9")

# (c) overlap between the two subspaces
ov = np.linalg.norm(seed.conj().T @ Vg)      # Frobenius of cross-gram; ~sqrt(9)=3 if identical spans
print(f"\n(c) subspace overlap ||seed^H Vg||_F = {ov:.3f}  (3.0 = identical 9-dim spans, 0 = orthogonal)")
