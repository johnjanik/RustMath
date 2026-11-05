"""(3,4,3) structured VARPRO: P = A^3, Q = R^3 T (T = lam*S deg 6), P - Q = c W^4.

Pinned (dd): 4 of 8 A-roots (incl. 0), 4 of 6 R-roots, 4 of 6 W-roots, 1 of 6 S-roots
(the elliptic c2c23 — entered as a soft address row on T's root, not a hard pin, since T
is inner-linear).  Outer: 4 free A + 2 free R + 2 free W = 8 complex = 16 reals (LM with
numerical gradient).  Inner: linear lstsq for T (7) and c (1) over 25 structure rows +
data rows, IRLS-frozen scales, 3 passes.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import c343_config as CC

SW = CC.SW
D = np.load(SW + "pC_samples1.npz", allow_pickle=True)
X, PH, REG = D['X'], D['PHI'], D['region']
PW = X[:, None]**np.arange(25)
NS = len(X)
F24 = np.load(SW + "pC_deg24_fit.npz")
S_ = float(F24['S'])
pwS = (X/S_)[:, None]**np.arange(25)
W0 = 1.0/np.maximum(np.abs(pwS@F24['p']) + np.abs(PH*(pwS@F24['q'])), 1e-300)
W0 = W0/np.median(W0)


def pfr(rr):
    P = np.array([1.0 + 0j])
    for r in rr:
        P = np.convolve(P, np.array([-r, 1.0 + 0j]))
    return P


def inner(theta, return_full=False, npass=3):
    aF = [theta[0] + 1j*theta[1], theta[2] + 1j*theta[3],
          theta[4] + 1j*theta[5], theta[6] + 1j*theta[7]]
    rF = [theta[8] + 1j*theta[9], theta[10] + 1j*theta[11]]
    wF = [theta[12] + 1j*theta[13], theta[14] + 1j*theta[15]]
    A = pfr(CC.A_PINS + aF)                    # deg 8 monic
    P = np.convolve(np.convolve(A, A), A)      # deg 24 monic
    R = pfr(CC.R_PINS + rF)                    # deg 6 monic
    R3 = np.convolve(np.convolve(R, R), R)     # deg 18 monic
    W = pfr(CC.W_PINS + wF)                    # deg 6 monic
    W4 = np.convolve(np.convolve(W, W), np.convolve(W, W))   # deg 24 monic
    R3v = PW[:, :19]@R3
    Pv = PW@P[:25]
    sD = W0.copy()
    sP = 1.0/(1 + np.abs(P[:25]))
    T = c = None
    for _ in range(npass):
        M = np.zeros((25 + NS, 8), complex)
        b = np.zeros(25 + NS, complex)
        for k in range(25):
            for l in range(7):
                if 0 <= k - l <= 18:
                    M[k, l] = R3[k - l]*sP[k]
            M[k, 7] = W4[k]*sP[k]
            b[k] = P[k]*sP[k]
        for l in range(7):
            M[25:, l] = PH*R3v*PW[:, l]*sD
        b[25:] = Pv*sD
        v, *_ = np.linalg.lstsq(M, b, rcond=None)
        T = v[:7]
        c = v[7]
        Q = np.convolve(R3, T)
        Qv = PW@Q[:25]
        rn = np.linalg.norm(M@v - b)
        sD = 1.0/np.maximum(np.abs(Pv) + np.abs(PH*Qv), 1e-300)
        sP = 1.0/(1 + np.abs(P[:25]) + np.abs(Q[:25]))
    if not return_full:
        return rn
    return rn, T, c, P, Q, W


def datamed(P, Q):
    Pv = PW@P[:25]
    Qv = PW@Q[:25]
    rel = np.abs(Pv - PH*Qv)/(np.abs(Pv) + np.abs(PH*Qv) + 1e-300)
    return float(np.median(rel)), float(np.max(rel))


def lm16(th0, iters=300, verbose=True):
    th = th0.copy()
    f = inner(th)
    lam_ = 1e-3
    h = 1e-7
    for it in range(iters):
        g = np.zeros(16)
        for k in range(16):
            tp = th.copy(); tp[k] += h
            g[k] = (inner(tp) - f)/h
        ok = False
        for _ in range(25):
            dx = g/(np.linalg.norm(g)**2 + lam_)*f
            tn = th - dx
            fn = inner(tn)
            if fn < f:
                th, f = tn, fn
                lam_ = max(lam_/3, 1e-14)
                ok = True
                break
            lam_ *= 5
        if verbose and (it % 20 == 0 or not ok):
            print(f"  it {it:3d}: |F| = {f:.6e}", flush=True)
        if not ok:
            break
    return th, f


if __name__ == "__main__":
    from scipy.optimize import minimize
    # census seeds for the free roots
    a_seed = [complex(-0.299161, -0.405134), complex(0.205561, -0.019228),
              complex(0.588612, -0.905159), complex(-0.055285, -0.102498)]
    r_seed = [complex(0.345506, -0.151209), complex(0.467286, -0.373624)]
    w_seed = [complex(0.411156, -0.295225), complex(-0.202326, 0.147932)]
    th0 = np.array(sum(([v.real, v.imag] for v in a_seed + r_seed + w_seed), []))
    print(f"seed inner: {inner(th0):.4e}", flush=True)
    sol = minimize(lambda t: inner(t), th0, method='Nelder-Mead',
                   options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
    print(f"NM: {sol.fun:.6e} after {sol.nit} iters", flush=True)
    th, f = lm16(sol.x)
    rn, T, c, P, Q, W = inner(th, return_full=True)
    dm, dmx = datamed(P, Q)
    print(f"final |F| = {rn:.6e}  data med = {dm:.3e} max = {dmx:.3e}")
    print(f"lam = T[6] = {T[6]:+.10f}   c = {c:+.10e}")
    for nm, i0 in [("A", 0), ("R", 8), ("W", 12)]:
        kmax = {"A": 4, "R": 2, "W": 2}[nm]
        for k in range(kmax):
            print(f"  {nm}-free: {th[i0+2*k] + 1j*th[i0+2*k+1]:+.10f}")
    tr = np.roots(T[::-1])
    print("  T-roots (simple poles; one should hit the c2c23 pin "
          f"{CC.S_PINS[0]:+.8f}):")
    for z in tr:
        print(f"    {z:+.8f}")
    np.savez(SW + "pC_varpro.npz", theta=th, T=T, c=c, P=P, Q=Q, W=W)
    print("saved pC_varpro.npz")
