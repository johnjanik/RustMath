"""sigma_B from the 12-chart pair (b <-> b2): the pairing that is combinatorially CERTAIN.

tau swaps the two 12-cycles, so sigma maps the b-chart's center point (12-point r1) to the
b2-chart's (r2).  Locally sigma(point at w in b) = point at eta(w) in b2 with
eta(w) = t*conj(w) (phi = F(w^12)-form at both charts, F real => eta linear-antilinear, t
unknown: 12 branches x the kappa ratio).  Moebius eliminated via cross-ratios: t is the only
unknown.  Then M from 3 point-pairs, involution check, and sigma-images of all measured
addresses (per the atlas: the unmeasured pole/zero addresses for free).

Run AFTER b_pipeline glues both 12-charts.
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


def taylor(key, r0=0.20, nth=64, nord=16):
    npz = np.load(BC.NPZ[key], allow_pickle=True)
    mu = np.load(BC.SW + f"muB_a_{key}.npy")
    vals = np.array([complex(moeb(mu, x_kmsv(npz, mp.mpf(r0)*mp.exp(1j*2*mp.pi*kk/nth))))
                     for kk in range(nth)])
    F = np.fft.fft(vals)/nth
    coefs = F[:nord]/(r0**np.arange(nord))
    print(f"taylor({key}): X(0) = {coefs[0]:.12f}  fft-tail {np.abs(F[nord:nth//2]).max():.1e}",
          flush=True)
    return coefs


def ap(coefs, s):
    return sum(c*s**k for k, c in enumerate(coefs))


def cr(a, b, c, d):
    return ((a - c)*(b - d))/((a - d)*(b - c))


if __name__ == "__main__":
    CB = taylor('b')       # germ at r1
    CB2 = taylor('b2')     # germ at r2
    CBb = np.conj(CB)

    svals = 0.16*np.exp(1j*2*np.pi*(np.arange(10) + 0.3)/10)
    QUADS = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 0, 4), (1, 5, 2, 8), (3, 7, 9, 6),
             (0, 5, 9, 2), (1, 6, 8, 3)]

    def resid_t(p):
        t = p[0] + 1j*p[1]
        L = np.array([ap(CB2[:14], t*s) for s in svals])
        R = np.array([ap(CBb[:14], s) for s in svals])
        out = []
        for (i, j, k_, l) in QUADS:
            d = cr(L[i], L[j], L[k_], L[l]) - cr(R[i], R[j], R[k_], R[l])
            out += [d.real, d.imag]
        return np.array(out)

    best = None
    for k in range(12):
        t0 = np.exp(2j*np.pi*k/12)
        for sc_ in (0.7, 1.0, 1.4):
            sol = least_squares(resid_t, [sc_*t0.real, sc_*t0.imag], method='lm', max_nfev=3000)
            rn = np.sqrt(2*sol.cost/len(QUADS))
            if best is None or sol.cost < best.cost:
                best = (sol.cost, rn, sol.x, k)
    cost, rn, x, k0 = best
    t = x[0] + 1j*x[1]
    print(f"\nBEST: rms = {rn:.3e}  t = {t:.12f}  |t| = {abs(t):.10f}  (seed branch {k0})")

    # M from 3 pairs
    s3 = [svals[0], svals[3], svals[6]]
    zs = [ap(CBb[:14], s) for s in s3]
    ws = [ap(CB2[:14], t*s) for s in s3]

    def mobius_from_3(zs, ws):
        def to01inf(p):
            z0, z1, z2 = p
            return np.array([[z1 - z2, -z0*(z1 - z2)], [z1 - z0, -z2*(z1 - z0)]], complex)
        T = np.linalg.inv(to01inf(ws)) @ to01inf(zs)
        return (T[0, 1], T[0, 0], T[1, 1], T[1, 0])

    M = mobius_from_3(zs, ws)

    def sig(xv):
        al, be, ga, de = M
        xc = np.conj(np.asarray(xv))
        return (al + be*xc)/(ga + de*xc)

    probes = np.array([0.1 - 0.2j, -0.3 + 0.1j, 0.4 - 0.1j, -0.1 - 0.35j, 0.9 + 0.3j])
    inv = np.abs(sig(sig(probes)) - probes).max()
    print(f"involution defect: {inv:.2e}")
    r1 = CB[0]; r2 = CB2[0]
    print(f"r1 = {r1:.12f}   r2 = {r2:.12f}")
    print(f"sigma(r1) = {sig(r1):.12f}  (should be r2: d = {abs(sig(r1)-r2):.2e})")
    print(f"sigma(r2) = {sig(r2):.12f}  (should be r1: d = {abs(sig(r2)-r1):.2e})")
    Pc = complex(-0.24339094187298, -0.19286184625628)
    Pc3 = complex(-0.23789929378150, 0.23252559046980)
    print(f"sigma(c-pole)  = {sig(Pc):.12f}   <- an uncharted 5-pole address (c2 or c4)")
    print(f"sigma(c3-pole) = {sig(Pc3):.12f}  <- the other uncharted 5-pole address")
    print(f"sigma(0)       = {sig(0j):.12f}   <- the (9 15)-double's address")
    if rn < 1e-9 and inv < 1e-8:
        np.save(BC.SW + "sigmaB.npy", np.array(M, complex))
        np.save(BC.SW + "sigmaB_t12.npy", np.array([t], complex))
        print("sigma_B SAVED (sigmaB.npy)")
    else:
        print("NOT saved -- residual/involution above gate")
