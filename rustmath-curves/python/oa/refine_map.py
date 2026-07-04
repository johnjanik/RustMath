"""End-to-end: refined dd null space of M  ->  ramified [2,12,5] Belyi map  -> order-12 check.

Loads the double-double matrix dump, runs the Ogita-Aishima refined SVD to resolve the 9-dim
null space (the weight-4 forms) past the FP64 wall, then recovers the forms and solves the
degree-24 map. The payoff metric: do the roots over phi=1 now CLUSTER onto 2 order-12 points
(FP64 gave a scatter of 3.8x the coordinate scale = a generic, non-ramified map)?

At N=2500 the forms are truncation-limited to rho^N ~ 5.5e-11, so the forms->map stages run in
fp64 (5.5e-11 >> 1e-16); the dd is only needed to RESOLVE the null space of the ill-conditioned M.
Usage: python3 refine_map.py <m_ext_n2500_dd.bin>"""
import numpy as np, sys, math
from scipy.special import gamma
sys.path.insert(0, '/tmp/claude-1000/-home-john-inverse-galois-M23/24542307-282e-4596-89f8-915a13a1d65e/scratchpad')
from phi import phi_in_ukappa
from read_ext import read_ext
from mxpsvd import mxp_svd_dd
from ddcx import to_c128, cnew
try:
    import cupy as xp; _GPU = True
except Exception:
    import numpy as xp; _GPU = False

path = sys.argv[1]
k = 4; rho = 0.990605; nforms = 9; a, b, c = 2, 12, 5; d = 24; Lu = 80
A_ = 0.5*(1+1/a-1/b-1/c); B_ = 0.5*(1+1/a-1/b+1/c); C_ = 1+1/a
Lam = (math.cos(math.pi/a)*math.cos(math.pi/b)+math.cos(math.pi/c))/(math.sin(math.pi/a)*math.sin(math.pi/b))
mu = Lam+math.sqrt(Lam*Lam-1)
kappa = ((mu-1)/(mu+1))*gamma(2-C_)*gamma(C_-A_)*gamma(C_-B_)/(gamma(1-A_)*gamma(1-B_)*gamma(C_))
phi_uk = phi_in_ukappa(a, b, c, 2*Lu+4)
phiv = np.array([float(phi_uk[2*n]) for n in range(Lu)], dtype=complex)

# --- load dd matrix, refine, extract the 9 null vectors ---
dim, nlimbs, re, im = read_ext(path)
print(f"loaded dim={dim} nlimbs={nlimbs}")
A_dd = cnew(xp.asarray(re[0]), xp.asarray(re[1] if nlimbs > 1 else np.zeros_like(re[0])),
            xp.asarray(im[0]), xp.asarray(im[1] if nlimbs > 1 else np.zeros_like(im[0])))
Ahi = (re[0] + 1j*im[0])
import os
MAXIT = int(os.environ.get("MAXIT", "8"))
# seed continuation: FP64-SVD a lower-N matrix (below the wall -> correct forms), zero-pad to dim
V_seed = None
SEED_PATH = os.environ.get("SEED_PATH")
if SEED_PATH:
    sd, snl, sre, sim = read_ext(SEED_PATH)
    su, ss, svh = xp.linalg.svd(xp.asarray(sre[0] + 1j*sim[0]))
    ssn = np.asarray(ss.get() if _GPU else ss); svhn = np.asarray(svh.get() if _GPU else svh)
    Vs = svhn.conj().T[:, np.argsort(ssn)[:nforms]]        # sd x nforms right-null vectors
    V_seed = np.zeros((dim, nforms), complex); V_seed[:sd, :] = Vs
    print(f"seed: N={sd-1} (dim {sd}) forms padded to {dim}; tail-norm ~ rho^{sd} = {rho**sd:.1e}")
print(f"refining (Ogita-Aishima dd)... full SVD refinement of the preconditioned M (max_iter={MAXIT})")
U, V, sigh, J = mxp_svd_dd(A_dd, Ahi, n_slices=6, max_iter=MAXIT, n_null=nforms, verbose=True, V_seed=V_seed)
Vc = to_c128(V); Vc = np.asarray(Vc.get() if _GPU else Vc)
sg = np.asarray(sigh.get() if _GPU else sigh)
Jn = list(np.argsort(sg)[:nforms])
print(f"null sigma (forms) = {np.sort(sg)[:nforms]}")

# --- recover forms: b_n = rho^-n y_n, then (1-w)^k series (fp64 is ample at rho^N~5e-11) ---
scale = rho**(-np.arange(dim))
omw = np.array([(-1.)**j*math.comb(k, j) if j <= k else 0. for j in range(dim)])
# right singular vector, NOT conjugated (matches validated extract_seed: Vh[i].conj() = V[:,i])
forms_u = np.array([np.convolve(omw, Vc[:, jj]*scale)[:2*Lu:2] for jj in Jn])

# --- echelonize by w-valuation (even part, u=w^2) ---
ATOL = 5e-5
def val(v):
    nz = np.nonzero(np.abs(v) > ATOL)[0]; return int(nz[0]) if len(nz) else len(v)
rows = forms_u.copy(); D = len(rows)
for done in range(D):
    vals = [val(rows[i]) if i >= done else 10**9 for i in range(D)]
    pp = int(np.argmin(vals)); pv = vals[pp]
    if pv >= rows.shape[1]: break
    rows[[done, pp]] = rows[[pp, done]]; rows[done] = rows[done]/rows[done][pv]
    for r in range(D):
        if r != done and abs(rows[r][pv]) > ATOL: rows[r] = rows[r]-rows[r][pv]*rows[done]
ech = {val(rows[i]): rows[i] for i in range(D) if val(rows[i]) < rows.shape[1]}
print(f"echelon valuations: {sorted(ech.keys())}")

def sval(sr):
    nz = np.nonzero(np.abs(sr) > ATOL)[0]; return int(nz[0]) if len(nz) else len(sr)
def urecip(sr, L):
    r = np.zeros(L, complex); r[0] = 1/sr[0]
    for n in range(1, L): r[n] = -sum(sr[j]*r[n-j] for j in range(1, min(n, len(sr)-1)+1))/sr[0]
    return r
def sdiv(num, den, L):
    vn, vd = sval(num), sval(den); q = np.convolve(num[vn:], urecip(den[vd:], L))[:L]
    o = np.zeros(L, complex); sh = vn-vd; o[sh:sh+len(q)] = q[:L-sh]; return o
def smul(x, y, L): return np.convolve(x, y)[:L]

m_u = max(v for v in ech if v+1 in ech)
g, h = ech[m_u], ech[m_u+1]; cc = h[m_u+2]
Xu = sdiv(h, g+cc*h, Lu); Xv = Xu*(kappa**2)**np.arange(Lu)
np.save(path + ".Xv.npy", Xv)  # diagnostic: compare Hauptmodul across N
print(f"Xv[m_u..m_u+8] = {np.array2string(Xv[m_u:m_u+8], precision=6, max_line_width=200)}")
Xp = [np.zeros(Lu, complex) for _ in range(d+1)]; Xp[0][0] = 1.
for i in range(1, d+1): Xp[i] = smul(Xp[i-1], Xv, Lu)
Nfit = 58
cols = [smul(phiv, Xp[i], Lu) for i in range(d+1)] + [-Xp[i] for i in range(d+1)]
A0 = np.array([[cols[j][n] for j in range(2*(d+1))] for n in range(Nfit)])
cn = np.linalg.norm(A0, axis=0); cn[cn == 0] = 1
_, sgv, V2 = np.linalg.svd(A0/cn); nv = (V2[-1].conj())/cn
q = nv[:d+1]; p = nv[d+1:]
# fit-error: does phi = P(X)/Q(X) actually hold? (fit range 0..Nfit, then PREDICT beyond)
P = sum(p[i]*Xp[i] for i in range(d+1)); Q = sum(q[i]*Xp[i] for i in range(d+1))
PhiX = sdiv(P, Q, Lu); err = np.abs(PhiX - phiv)
print(f"map solve: m_u={m_u} sv-gap={sgv[-2]/sgv[-1]:.1e}  "
      f"fit_err(0..{Nfit})={err[:Nfit].max():.1e}  PREDICT({Nfit}..{Lu-2})={err[Nfit:Lu-2].max():.1e}")

# --- the payoff: do the roots over phi=1 cluster onto 2 order-12 points? ---
import mpmath as mp
mp.mp.dps = 40
def roots_of(coeff):
    cf = np.array(coeff); deg = len(cf)-1
    while deg > 0 and abs(cf[deg]) < 1e-9*np.abs(cf).max(): deg -= 1
    rr = mp.polyroots([mp.mpc(cf[i].real, cf[i].imag) for i in range(deg, -1, -1)], maxsteps=300, extraprec=250)
    return np.array([complex(r) for r in rr])
def spread(coeff, mult, ngrp, label):
    R = list(roots_of(coeff)); smed = np.median(np.abs([r for r in R if abs(r) > 1e-12]))
    groups = []
    for _ in range(ngrp):
        best = None
        for i in range(len(R)):
            ds = sorted(range(len(R)), key=lambda j: abs(R[j]-R[i]))[:mult]
            sp = max(abs(R[j]-R[i]) for j in ds)
            if best is None or sp < best[0]: best = (sp, ds)
        idx = set(best[1]); groups.append(best[0]/smed); R = [R[j] for j in range(len(R)) if j not in idx]
    print(f"  {label}: {ngrp}x{mult} spreads(rescaled) = {[f'{s:.2e}' for s in groups]}")
    return max(groups)
print("ramification tightness (rescaled spread; FP64 gave ~0.3 doubles, ~3.8 order-12):")
spread(p, 2, 8, "P zeros over 0   (8x2)")
spread(q, 5, 4, "Q poles over inf (4x5)")
o12 = spread(p-q, 12, 2, "P-Q over 1       (2x12)")
print(f"\nORDER-12 max spread = {o12:.2e}  (target << 3.8; ~0.12 expected at N=2500)")
