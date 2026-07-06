"""sigma_B by germ series matching between the mirror charts c and c3.

sigma(point at w in c-chart) = point at eta(w) in c3-chart with eta(w) = t*conj(w)
(t = zeta_5^k kappa_c3/conj(kappa_c), unknown).  Then as power series in s:

    X_c3(t s) = M(Xbar_c(s)),   Xbar_c = X_c with conjugated coefficients,

unknowns t (1 complex) and Moebius M (3 complex).  Taylor coefficients of X_c, X_c3 from
ring FFT at |w| = r0; solve by least squares through order ~12; the full series pins the
branch and the twist uniquely.  Validate: involution, addresses.
"""
import numpy as np, mpmath as mp, sys, os
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(__file__))
from chart_dd import mp_series_eval
import b_config as BC

mp.mp.dps = 40


def x_kmsv(npz, w):
    G = mp_series_eval(npz['Bh'], npz['Bl'], w)
    c = -(mp.mpc(npz['Bh'][8, 7]) + mp.mpc(npz['Bl'][8, 7]))
    return G[8]/(G[7] + c*G[8])


def moeb(m, x):
    al, be, ga, de = [mp.mpc(v) for v in m]
    return (al + be*x)/(ga + de*x)


def taylor(key, r0=0.15, nth=64, nord=20):
    npz = np.load(BC.NPZ[key], allow_pickle=True)
    mu = np.load(BC.SW + f"muB_a_{key}.npy")
    vals = []
    for kk in range(nth):
        w = mp.mpf(r0)*mp.exp(1j*2*mp.pi*kk/nth)
        vals.append(complex(moeb(mu, x_kmsv(npz, w))))
    vals = np.array(vals)
    F = np.fft.fft(vals)/nth
    coefs = F[:nord]/(r0**np.arange(nord))
    tail = np.abs(F[nord:nth//2]).max() if nord < nth//2 else 0.0
    print(f"taylor({key}): |c0..4| = {np.round(np.abs(coefs[:5]), 6)}  fft-tail {tail:.1e}")
    return coefs


def apply_series(coefs, s):
    return sum(c*s**k for k, c in enumerate(coefs))


if __name__ == "__main__":
    NORD = 13
    C1 = taylor('c')      # X_c(s)
    C3 = taylor('c3')     # X_c3(s)
    C1b = np.conj(C1)     # Xbar_c

    svals = 0.12*np.exp(1j*2*np.pi*(np.arange(10) + 0.3)/10)

    def cr(a, b, c, d):
        return ((a - c)*(b - d))/((a - d)*(b - c))

    QUADS = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 0, 4), (1, 5, 2, 8), (3, 7, 9, 6),
             (0, 5, 9, 2), (1, 6, 8, 3)]

    def resid_t(p):
        """Moebius eliminated via cross-ratios: t is the ONLY unknown (collapse-proof).
        sigma antiholomorphic => CR(lhs quad) = conj(CR(xb quad))... note xb already has
        conjugated coefficients, so match CR(X_c3(t s_i)) = CR(xb(s_i)) directly."""
        t = p[0] + 1j*p[1]
        L = np.array([apply_series(C3[:NORD], t*s) for s in svals])
        R = np.array([apply_series(C1b[:NORD], s) for s in svals])
        r = []
        for (i, j, k_, l) in QUADS:
            d = cr(L[i], L[j], L[k_], L[l]) - cr(R[i], R[j], R[k_], R[l])
            r += [d.real, d.imag]
        return np.array(r)

    best = None
    for k in range(5):
        t0 = np.exp(2j*np.pi*k/5)
        sol = least_squares(resid_t, [t0.real, t0.imag], method='lm', max_nfev=4000)
        rn = np.sqrt(2*sol.cost/len(QUADS))
        print(f"start k={k}: rms = {rn:.3e}   t = {sol.x[0]+1j*sol.x[1]:.10f}")
        if best is None or sol.cost < best.cost:
            best = sol
    t = best.x[0] + 1j*best.x[1]
    # recover M from three point-pairs (xb(s) -> X_c3(t s))
    s3 = [svals[0], svals[3], svals[6]]
    zs = [apply_series(C1b[:NORD], s) for s in s3]
    ws = [apply_series(C3[:NORD], t*s) for s in s3]

    def mobius_from_3(zs, ws):
        def to01inf(p):
            z0, z1, z2 = p
            return np.array([[z1 - z2, -z0*(z1 - z2)], [z1 - z0, -z2*(z1 - z0)]], complex)
        T = np.linalg.inv(to01inf(ws)) @ to01inf(zs)
        return (T[0, 1], T[0, 0], T[1, 1], T[1, 0])   # (al, be, ga, de) for (al+be x)/(ga+de x)

    M = mobius_from_3(zs, ws)
    rn = np.sqrt(2*best.cost/len(svals))
    print(f"\nBEST rms = {rn:.3e}")
    print(f"t = {t:.10f}  |t| = {abs(t):.10f}")

    def sig(x):
        al, be, ga, de = M
        xc = np.conj(np.asarray(x))
        return (al + be*xc)/(ga + de*xc)

    probes = np.array([0.1 - 0.2j, -0.3 + 0.1j, 0.4 - 0.1j, -0.1 - 0.35j])
    inv = np.abs(sig(sig(probes)) - probes).max()
    print(f"involution defect: {inv:.2e}")
    Pc = complex(-0.24339094187298, -0.19286184625628)
    Pc3 = complex(-0.23789929378150, 0.23252559046980)
    print(f"sigma(c-pole)  = {sig(Pc):.12f}  vs c3 addr {Pc3:.12f}  d={abs(sig(Pc)-Pc3):.2e}")
    print(f"sigma(c3-pole) = {sig(Pc3):.12f} vs c  addr {Pc:.12f}  d={abs(sig(Pc3)-Pc):.2e}")
    print(f"sigma(0)       = {sig(0j):.12f}")
    if rn < 1e-9:
        np.save(BC.SW + "sigmaB.npy", np.array(M, complex))
        np.save(BC.SW + "sigmaB_t.npy", np.array([t], complex))
        print("saved sigmaB.npy")
