"""Fast GPU pass over the assembled dd matrices (sweep_assemble.sh output): for each N, refine
the null space, and trace how the Hauptmodul / null-space overfits as N grows.

Per N we report, from the dd-refined spectrum:
  rhoN        = rho^N                         (physical truncation floor / expected form sigma)
  smin        = smallest dd sigma             (< rhoN  =>  tail-freedom OVERFIT)
  n_sub       = # sigma below 0.3*rhoN        (how many overfit directions have appeared)
  reality_lo  = |Im/full| of Hauptmodul from the SMALLEST-9 subspace   (0 good, ~.5 garbage)
  reality_phys= same, but from the 9 sigma CLOSEST to rhoN (the physical forms)
  o12_lo      = order-12 spread from smallest-9;   o12_phys from the physical band
The transition N (reality_lo jumps 0 -> ~.5 while reality_phys stays ~0) localizes the onset,
and whether the physical band still yields the true map above onset.

Usage: python3 sweep_analyze.py /home/john/sweep_2_12_5 [--no-o12]
"""
import numpy as np, sys, math, glob, os, re as _re
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

SWEEP = sys.argv[1] if len(sys.argv) > 1 else '/home/john/sweep_2_12_5'
DO_O12 = '--no-o12' not in sys.argv
k = 4; rho = 0.990605; nforms = 9; a, b, c = 2, 12, 5; d = 24; Lu = 80
A_ = 0.5*(1+1/a-1/b-1/c); B_ = 0.5*(1+1/a-1/b+1/c); C_ = 1+1/a
Lam = (math.cos(math.pi/a)*math.cos(math.pi/b)+math.cos(math.pi/c))/(math.sin(math.pi/a)*math.sin(math.pi/b))
mu = Lam+math.sqrt(Lam*Lam-1)
kappa = ((mu-1)/(mu+1))*gamma(2-C_)*gamma(C_-A_)*gamma(C_-B_)/(gamma(1-A_)*gamma(1-B_)*gamma(C_))
phi_uk = phi_in_ukappa(a, b, c, 2*Lu+4)
phiv = np.array([float(phi_uk[2*n]) for n in range(Lu)], dtype=complex)

# ---- series helpers (same as refine_map) ----
ATOL = 5e-5
def val(v):
    nz = np.nonzero(np.abs(v) > ATOL)[0]; return int(nz[0]) if len(nz) else len(v)
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

def echelonize(forms_u):
    rows = forms_u.copy(); D = len(rows)
    for done in range(D):
        vals = [val(rows[i]) if i >= done else 10**9 for i in range(D)]
        pp = int(np.argmin(vals)); pv = vals[pp]
        if pv >= rows.shape[1]: break
        rows[[done, pp]] = rows[[pp, done]]; rows[done] = rows[done]/rows[done][pv]
        for r in range(D):
            if r != done and abs(rows[r][pv]) > ATOL: rows[r] = rows[r]-rows[r][pv]*rows[done]
    return {val(rows[i]): rows[i] for i in range(D) if val(rows[i]) < rows.shape[1]}

def hauptmodul(Vc, dim, Jsel):
    scale = rho**(-np.arange(dim))
    omw = np.array([(-1.)**j*math.comb(k, j) if j <= k else 0. for j in range(dim)])
    forms_u = np.array([np.convolve(omw, Vc[:, jj]*scale)[:2*Lu:2] for jj in Jsel])
    ech = echelonize(forms_u)
    if not ech: return None, None
    cand = [v for v in ech if v+1 in ech]
    if not cand: return None, ech
    m_u = max(cand); g, h = ech[m_u], ech[m_u+1]; cc = h[m_u+2]
    Xu = sdiv(h, g+cc*h, Lu); Xv = Xu*(kappa**2)**np.arange(Lu)
    return Xv, ech

def reality(Xv):
    """|Im/|.|| of the Hauptmodul measured AT its largest low-order coefficient (avoids the
    noise/noise ratio at the ~0 coefficients). 0 = real/physical, ~1 = complex/garbage."""
    if Xv is None: return float('nan'), complex('nan')
    band = Xv[1:7]; mag = np.abs(band)
    idx = int(np.argmax(mag))
    if mag[idx] == 0: return float('nan'), band[idx]
    return float(abs(band[idx].imag)/mag[idx]), band[idx]

def tail_energy(Vc, dim, cols):
    """fraction of each vector's norm in the high-index (n>dim/2) tail, averaged over cols.
    (Kept for reference; monotone in N -> weak onset discriminant.)"""
    h = dim // 2; te = []
    for j in cols:
        v = Vc[:, j]; nrm = np.linalg.norm(v)
        te.append(np.linalg.norm(v[h:]) / (nrm + 1e-300))
    return float(np.mean(te))

_SCALE = {}
def bgrowth(Vc, dim, cols):
    """max|b_n|/median|b_n| for b_n = rho^-n y_n, averaged over cols.  Physical forms have BOUNDED
    coefficients (b_n ~ O(1)) -> growth ~ O(1); the overfit blows b_n up like rho^-n ~ 1e10 to cancel
    the truncation tail -> growth huge.  The strong physical/overfit discriminant, valid at ANY N."""
    if dim not in _SCALE: _SCALE[dim] = rho**(-np.arange(dim))
    sc = _SCALE[dim]; g = []
    for j in cols:
        b = np.abs(Vc[:, j] * sc); med = np.median(b[b > 0]) + 1e-300
        g.append(float(np.max(b) / med))
    return g

def order12(Xv, ech):
    import mpmath as mp
    mp.mp.dps = 30
    m_u = max(v for v in ech if v+1 in ech)
    g, h = ech[m_u], ech[m_u+1]
    Xp = [np.zeros(Lu, complex) for _ in range(d+1)]; Xp[0][0] = 1.
    for i in range(1, d+1): Xp[i] = smul(Xp[i-1], Xv, Lu)
    Nfit = 58
    cols = [smul(phiv, Xp[i], Lu) for i in range(d+1)] + [-Xp[i] for i in range(d+1)]
    A0 = np.array([[cols[j][n] for j in range(2*(d+1))] for n in range(Nfit)])
    cn = np.linalg.norm(A0, axis=0); cn[cn == 0] = 1
    _, sgv, V2 = np.linalg.svd(A0/cn); nv = (V2[-1].conj())/cn
    q = nv[:d+1]; p = nv[d+1:]
    def roots_of(coeff):
        cf = np.array(coeff); deg = len(cf)-1
        while deg > 0 and abs(cf[deg]) < 1e-9*np.abs(cf).max(): deg -= 1
        rr = mp.polyroots([mp.mpc(cf[i].real, cf[i].imag) for i in range(deg, -1, -1)], maxsteps=250, extraprec=200)
        return [complex(r) for r in rr]
    R = list(roots_of(p-q)); smed = np.median(np.abs([r for r in R if abs(r) > 1e-12]))
    worst = 0.
    for _ in range(2):
        best = None
        for i in range(len(R)):
            ds = sorted(range(len(R)), key=lambda j: abs(R[j]-R[i]))[:12]
            sp = max(abs(R[j]-R[i]) for j in ds)
            if best is None or sp < best[0]: best = (sp, ds)
        worst = max(worst, best[0]/smed); idx = set(best[1]); R = [R[j] for j in range(len(R)) if j not in idx]
    return worst, sgv[-2]/sgv[-1]

def analyze(path):
    dim, nl, re, im = read_ext(path)
    A_dd = cnew(xp.asarray(re[0]), xp.asarray(re[1]), xp.asarray(im[0]), xp.asarray(im[1]))
    Ahi = re[0] + 1j*im[0]
    N = dim - 1; rhoN = rho**N
    U, V, sigh, J = mxp_svd_dd(A_dd, Ahi, n_slices=6, max_iter=int(os.environ.get('MAXIT', '5')),
                               n_null=nforms, verbose=False)
    Vc = to_c128(V); Vc = np.asarray(Vc.get() if _GPU else Vc)
    sg = np.asarray(sigh.get() if _GPU else sigh)
    order = np.argsort(sg); ss = sg[order]
    smin = ss[0]; sig9 = ss[nforms-1]; n_sub = int(np.sum(sg < 0.3*rhoN))
    Jlo = list(order[:nforms])                                   # smallest-9 (chased by SVD/refine)
    # physical band: among the smallest ~2*nforms sigma, the 9 with LOWEST tail energy (stabler
    # selector than b-growth, which conflates rho^-n-amplified numerical noise with real overfit).
    cand = list(order[:2*nforms+2])
    Jphys = sorted(cand, key=lambda j: tail_energy(Vc, dim, [j]))[:nforms]
    out = dict(N=N, rhoN=rhoN, smin=smin, sig9=sig9, n_sub=n_sub, sig=ss[:12].tolist(),
               te_lo=tail_energy(Vc, dim, Jlo), te_phys=tail_energy(Vc, dim, Jphys),
               bgrow_lo=float(np.median(bgrowth(Vc, dim, Jlo))),
               bgrow_phys=float(np.median(bgrowth(Vc, dim, Jphys))))
    for tag, Jsel in (('lo', Jlo), ('phys', Jphys)):
        Xv, ech = hauptmodul(Vc, dim, Jsel)
        r, x1 = reality(Xv)
        out[f'reality_{tag}'] = r; out[f'x1_{tag}'] = x1
        out[f'o12_{tag}'] = float('nan')
        if DO_O12 and Xv is not None and ech and [v for v in ech if v+1 in ech]:
            try:
                o12, _sv = order12(Xv, ech); out[f'o12_{tag}'] = o12
            except Exception:
                pass
    return out

paths = sorted(glob.glob(os.path.join(SWEEP, 'm_N*.bin')),
               key=lambda p: int(_re.search(r'm_N(\d+)\.bin', p).group(1)))
print(f"analyzing {len(paths)} matrices in {SWEEP}  (GPU={_GPU})")
csv = os.path.join(SWEEP, 'metrics.csv')
hdr = "N,rhoN,smin,sig9,n_sub,te_lo,te_phys,bgrow_lo,bgrow_phys,reality_lo,reality_phys,o12_lo,o12_phys," \
      "x1re_lo,x1im_lo,x1re_phys,x1im_phys," + ",".join(f"sig{i}" for i in range(12))
with open(csv, 'w') as f: f.write(hdr + "\n")
print(f"{'N':>5} {'rhoN':>9} {'smin':>9} {'bgro_lo':>8} {'bgro_phy':>8} {'real_lo':>8} {'real_phys':>9} {'o12_lo':>8} {'o12_phys':>8}")
for p in paths:
    try:
        m = analyze(p)
    except Exception as e:
        print(f"  {os.path.basename(p)}: FAILED {e}"); continue
    row = f"{m['N']},{m['rhoN']:.3e},{m['smin']:.3e},{m['sig9']:.3e},{m['n_sub']}," \
          f"{m['te_lo']:.3e},{m['te_phys']:.3e},{m['bgrow_lo']:.3e},{m['bgrow_phys']:.3e}," \
          f"{m['reality_lo']:.3e},{m['reality_phys']:.3e},{m['o12_lo']:.3e},{m['o12_phys']:.3e}," \
          f"{m['x1_lo'].real:.4e},{m['x1_lo'].imag:.4e},{m['x1_phys'].real:.4e},{m['x1_phys'].imag:.4e}," \
          + ",".join(f"{s:.3e}" for s in m['sig'])
    with open(csv, 'a') as f: f.write(row + "\n")
    print(f"{m['N']:>5} {m['rhoN']:>9.2e} {m['smin']:>9.2e} {m['bgrow_lo']:>8.1e} {m['bgrow_phys']:>8.1e} "
          f"{m['reality_lo']:>8.2e} {m['reality_phys']:>9.2e} {m['o12_lo']:>8.2e} {m['o12_phys']:>8.2e}")
print(f"\nwrote {csv}")
