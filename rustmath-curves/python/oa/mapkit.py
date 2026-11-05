"""Reusable [2,12,5] forms -> Hauptmodul -> degree-24 map -> order-12 diagnostics.
Extracted from sweep_analyze so the physical-selector experiments share one code path.
A "basis" here is a (dim, 9) complex array whose columns are candidate right-null vectors y
(y_n = rho^n b_n); we recover b, build the (1-w)^k series, echelonize by valuation, form the
Hauptmodul X, solve phi=P(X)/Q(X), and measure the order-12 ramification collapse."""
import numpy as np, math
from scipy.special import gamma
import sys, os

k = 4; rho = 0.990605; nforms = 9; a, b, c = 2, 12, 5; d = 24; Lu = 80
A_ = 0.5*(1+1/a-1/b-1/c); B_ = 0.5*(1+1/a-1/b+1/c); C_ = 1+1/a
Lam = (math.cos(math.pi/a)*math.cos(math.pi/b)+math.cos(math.pi/c))/(math.sin(math.pi/a)*math.sin(math.pi/b))
mu = Lam+math.sqrt(Lam*Lam-1)
kappa = ((mu-1)/(mu+1))*gamma(2-C_)*gamma(C_-A_)*gamma(C_-B_)/(gamma(1-A_)*gamma(1-B_)*gamma(C_))
KAPPA2 = kappa**2
_PHIV_CACHE = '/home/john/sweep_2_12_5/phiv_2_12_5.npy'   # phi_in_ukappa reversion is ~105s; cache it
if os.path.exists(_PHIV_CACHE):
    phiv = np.load(_PHIV_CACHE)[:Lu].astype(complex)
else:
    sys.path.insert(0, '/tmp/claude-1000/-home-john-inverse-galois-M23/24542307-282e-4596-89f8-915a13a1d65e/scratchpad')
    from phi import phi_in_ukappa
    _phi_uk = phi_in_ukappa(a, b, c, 2*Lu+4)
    phiv = np.array([float(_phi_uk[2*n]) for n in range(Lu)], dtype=complex)
    np.save(_PHIV_CACHE, phiv)

ATOL = 5e-5
def _val(v):
    nz = np.nonzero(np.abs(v) > ATOL)[0]; return int(nz[0]) if len(nz) else len(v)
def _sval(sr):
    nz = np.nonzero(np.abs(sr) > ATOL)[0]; return int(nz[0]) if len(nz) else len(sr)
def _urecip(sr, L):
    r = np.zeros(L, complex); r[0] = 1/sr[0]
    for n in range(1, L): r[n] = -sum(sr[j]*r[n-j] for j in range(1, min(n, len(sr)-1)+1))/sr[0]
    return r
def sdiv(num, den, L):
    vn, vd = _sval(num), _sval(den); q = np.convolve(num[vn:], _urecip(den[vd:], L))[:L]
    o = np.zeros(L, complex); sh = vn-vd; o[sh:sh+len(q)] = q[:L-sh]; return o
def smul(x, y, L): return np.convolve(x, y)[:L]

def recover_forms(Y, dim):
    """Y: (dim, m) columns are y-vectors. Return forms_u: (m, Lu) the (1-w)^k series (even part)."""
    scale = rho**(-np.arange(dim))
    omw = np.array([(-1.)**j*math.comb(k, j) if j <= k else 0. for j in range(dim)])
    return np.array([np.convolve(omw, Y[:, j]*scale)[:2*Lu:2] for j in range(Y.shape[1])])

def echelonize(forms_u):
    rows = forms_u.copy(); D = len(rows)
    for done in range(D):
        vals = [_val(rows[i]) if i >= done else 10**9 for i in range(D)]
        pp = int(np.argmin(vals)); pv = vals[pp]
        if pv >= rows.shape[1]: break
        rows[[done, pp]] = rows[[pp, done]]; rows[done] = rows[done]/rows[done][pv]
        for r in range(D):
            if r != done and abs(rows[r][pv]) > ATOL: rows[r] = rows[r]-rows[r][pv]*rows[done]
    return {_val(rows[i]): rows[i] for i in range(D) if _val(rows[i]) < rows.shape[1]}

def hauptmodul(Y, dim):
    """Return (Xv, ech) or (None, ech). Xv is the Hauptmodul coordinate series."""
    ech = echelonize(recover_forms(Y, dim))
    cand = [v for v in ech if v+1 in ech]
    if not cand: return None, ech
    m_u = max(cand); g, h = ech[m_u], ech[m_u+1]; cc = h[m_u+2]
    Xu = sdiv(h, g+cc*h, Lu); Xv = Xu*(KAPPA2)**np.arange(Lu)
    return Xv, ech

def reality(Xv):
    """(|Im/Re| at the largest low-order coeff, X[1]).  0 = real/physical."""
    if Xv is None: return float('nan'), complex('nan')
    band = Xv[1:7]; mag = np.abs(band); i = int(np.argmax(mag))
    if mag[i] == 0: return float('nan'), band[i]
    return float(abs(band[i].imag)/mag[i]), band[i]

def build_Xp(Xv):
    Xp = [np.zeros(Lu, complex) for _ in range(d+1)]; Xp[0][0] = 1.
    for i in range(1, d+1): Xp[i] = smul(Xp[i-1], Xv, Lu)
    return Xp

def fit_generic(Xv, Nfit=58):
    """Generic degree-24 map fit phi = P(X)/Q(X). Returns (p, q, Xp, sgv) with p,q lowest-first
    coefficient arrays (deg 24). This is the SEED for the structured fit (its P,Q roots are the
    approximate 8-doubles / 4-quintuples / 2-twelvefolds)."""
    Xp = build_Xp(Xv)
    cols = [smul(phiv, Xp[i], Lu) for i in range(d+1)] + [-Xp[i] for i in range(d+1)]
    A0 = np.array([[cols[j][n] for j in range(2*(d+1))] for n in range(Nfit)])
    cn = np.linalg.norm(A0, axis=0); cn[cn == 0] = 1
    _, sgv, V2 = np.linalg.svd(A0/cn); nv = (V2[-1].conj())/cn
    return nv[d+1:], nv[:d+1], Xp, sgv        # p (num, deg24), q (den, deg24), Xp, sgv

def order12(Xv, ech, want_svgap=False):
    """order-12 fiber spread (rescaled). Optionally also the map-solve sv-gap."""
    import mpmath as mp
    mp.mp.dps = 30
    m_u = max(v for v in ech if v+1 in ech)
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
    return (worst, sgv[-2]/sgv[-1]) if want_svgap else worst

def evaluate(Y, dim, label="", do_o12=True):
    """Full diagnostic dict for a candidate basis Y (dim, 9)."""
    Xv, ech = hauptmodul(Y, dim)
    r, x1 = reality(Xv)
    out = {'label': label, 'reality': r, 'x1': x1,
           'x1_err': abs(x1 - KAPPA2) if not (isinstance(x1, complex) and math.isnan(x1.real)) else float('nan'),
           'o12': float('nan'), 'valuations': sorted(ech.keys())}
    if do_o12 and Xv is not None and [v for v in ech if v+1 in ech]:
        try: out['o12'] = order12(Xv, ech)
        except Exception as e: out['o12_err'] = str(e)
    return out
