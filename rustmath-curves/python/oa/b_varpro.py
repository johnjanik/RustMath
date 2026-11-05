"""B VARPRO: outer = the 3 unknown double-pairs (6 reals); inner = LINEAR lstsq.

Pinned: quintuple poles P5 (4, dd: c, c3, sigma(c)=c4, sigma(c3)=c2), 12-points (W fixed),
origin double-pair (0, SIG0).  Given the outer doubles d1..d3 (each contributing the pair
(d, sigma(d))), P = A^2 * Z with A the monic deg-8 double-root poly (fully determined by
outer+pins) and Z the monic deg-8 simple-zero poly -- LINEAR unknowns.  Q = R5 * T with
T = lambda*S deg 4 (5 linear unknowns).  c linear.  Rows: 25 structure + data (all linear).
sigma-stability of the recovered Z, T is NOT imposed -- it emerges at the true point and
serves as a certificate.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import b_config as BC

SW = BC.SW
SIG = np.load(SW + "sigmaB.npy")
R12 = np.load(SW + "pB_r12.npy")
P5 = np.array([complex(-0.24339094187298, -0.19286184625628),
               complex(-0.23789929378150, +0.23252559046980),
               complex(-0.59198701480729, +0.21733468385806),
               complex(-0.64997276203624, -0.09319689864632)])
SIG0 = complex(-0.60511372176989, +0.07804715463807)


def sig(x):
    al, be, ga, de = SIG
    xc = np.conj(np.asarray(x))
    return (al + be*xc)/(ga + de*xc)


def pfr(rr):
    P = np.array([1.0 + 0j])
    for r in rr:
        P = np.convolve(P, np.array([-r, 1.0 + 0j]))
    return P


W24 = pfr([R12[0]]*12 + [R12[1]]*12)
R5 = pfr(list(P5)*5)                      # deg 20 monic

D = np.load(SW + "pB_samples_scan2.npz", allow_pickle=True)
X, PH, REG = D['X'], D['PHI'], D['region']
F4 = np.load(SW + "pB_deg24_fit.npz")
pwS = (X/float(F4['S']))[:, None]**np.arange(25)
WD = 1.0/np.maximum(np.abs(pwS@F4['p']) + np.abs(PH*(pwS@F4['q'])), 1e-300)
WD = WD/np.median(WD)
PW = X[:, None]**np.arange(25)            # raw-X Vandermonde
NS = len(X)
DW = float(os.environ.get("DW", "1.0"))   # data-row weight


def inner(dvec, return_full=False, npass=3):
    """dvec: 3 complex doubles. Linear lstsq for Z (monic deg8: 8), T (5), c, with
    IRLS-style row scales FROZEN from the previous pass's model (rows ~ relative errors)."""
    dbl = [0j, SIG0] + [v for m in dvec for v in (m, sig(m))]
    A = pfr(dbl)                          # deg 8 monic
    A2 = np.convolve(A, A)                # deg 16 monic
    A2v = PW[:, :17]@A2
    R5v = PW[:, :21]@R5
    # initial scales from the generic fit (values) and its coefficient transform
    sD = 1.0/np.maximum(np.abs(A2v)*np.abs(PW[:, :9]@np.ones(9)), 1e-300)  # crude start
    sP = np.ones(25)
    Z = T = c = None
    for it in range(npass):
        M = np.zeros((25 + NS, 14), complex)
        b = np.zeros(25 + NS, complex)
        for k in range(25):
            for j in range(8):
                if 0 <= k - j <= 16:
                    M[k, j] = A2[k - j]
            b[k] = -(A2[k - 8] if 0 <= k - 8 <= 16 else 0)
            for l in range(5):
                if 0 <= k - l <= 20:
                    M[k, 8 + l] = -R5[k - l]
            M[k, 13] = -(W24[k] if k <= 24 else 0)
            M[k] *= sP[k]
            b[k] *= sP[k]
        for j in range(8):
            M[25:, j] = A2v*PW[:, j]*sD*DW
        for l in range(5):
            M[25:, 8 + l] = -PH*R5v*PW[:, l]*sD*DW
        b[25:] = -A2v*PW[:, 8]*sD*DW      # Z monic
        v, *_ = np.linalg.lstsq(M, b, rcond=None)
        rn = np.linalg.norm(M@v - b)
        Z = np.concatenate([v[:8], [1.0 + 0j]])
        T = v[8:13]
        c = v[13]
        P = np.convolve(A2, Z)
        Q = np.convolve(R5, T)
        Pv = PW@P[:25]
        Qv = PW@Q[:25]
        sD = 1.0/np.maximum(np.abs(Pv) + np.abs(PH*Qv), 1e-300)
        sP = 1.0/(1 + np.abs(P[:25]) + np.abs(Q[:25]))
    if not return_full:
        return rn
    return rn, Z, T, c, P, Q


def datamed(P, Q):
    Pv = PW@P[:25]
    Qv = PW@Q[:25]
    rel = np.abs(Pv - PH*Qv)/(np.abs(Pv) + np.abs(PH*Qv) + 1e-300)
    return np.median(rel), np.max(rel)


def outer_lm(d0, iters=120, h=1e-6, verbose=False):
    th = np.array([x for m in d0 for x in (m.real, m.imag)])
    f = inner([th[0] + 1j*th[1], th[2] + 1j*th[3], th[4] + 1j*th[5]])
    lam_ = 1e-3
    for it in range(iters):
        g = np.zeros(6)
        r0 = f
        for k in range(6):
            tp = th.copy(); tp[k] += h
            fp = inner([tp[0] + 1j*tp[1], tp[2] + 1j*tp[3], tp[4] + 1j*tp[5]])
            g[k] = (fp - r0)/h
        ok = False
        for _ in range(25):
            dx = g/(np.linalg.norm(g)**2 + lam_)*r0
            tn = th - dx
            fn = inner([tn[0] + 1j*tn[1], tn[2] + 1j*tn[3], tn[4] + 1j*tn[5]])
            if fn < f:
                th, f = tn, fn
                lam_ = max(lam_/3, 1e-12)
                ok = True
                break
            lam_ *= 5
        if verbose and it % 20 == 0:
            print(f"    it {it}: {f:.6e}", flush=True)
        if not ok:
            break
    return [th[0] + 1j*th[1], th[2] + 1j*th[3], th[4] + 1j*th[5]], f


if __name__ == "__main__":
    from scipy.optimize import minimize
    d0 = [complex(-0.2472, -0.1230), complex(-0.4797, -0.3135), complex(-0.2647, +0.5061)]

    def fun(th):
        return inner([th[0] + 1j*th[1], th[2] + 1j*th[3], th[4] + 1j*th[5]])

    th0 = np.array([x for m in d0 for x in (m.real, m.imag)])
    print(f"seed inner residual: {fun(th0):.6e}", flush=True)
    sol = minimize(fun, th0, method='Nelder-Mead',
                   options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-14})
    print(f"NM: {sol.fun:.6e} after {sol.nit} iters", flush=True)
    dbest = [sol.x[0] + 1j*sol.x[1], sol.x[2] + 1j*sol.x[3], sol.x[4] + 1j*sol.x[5]]
    dbest, f = outer_lm(dbest, iters=200, verbose=True)
    rn, Z, T, c, P, Q = inner(dbest, return_full=True)
    dm, dmx = datamed(P, Q)
    print(f"\nfinal |F| = {rn:.6e}   data med = {dm:.3e} max = {dmx:.3e}")
    print(f"c = {c:+.10e}   lam = T4 = {T[4]:+.10f}")
    for m in dbest:
        print(f"  double: {m:+.12f}   (pair {sig(m):+.12f})")
    # sigma-stability certificate of the recovered Z and T
    zr = np.roots(Z[::-1])
    tr = np.roots(T[::-1])
    for nm, rr in [("Z", zr), ("T", tr)]:
        img = sig(rr)
        mism = max(np.min(np.abs(rr - v)) for v in img)
        print(f"  {nm}-roots sigma-stability: {mism:.2e}")
        for r_ in sorted(rr, key=lambda z: z.real):
            print(f"    {nm}: {r_:+.8f}")
    np.savez(SW + "pB_varpro.npz", d=np.array(dbest), Z=Z, T=T, c=c, P=P, Q=Q)
    print("saved pB_varpro.npz")
