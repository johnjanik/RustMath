"""sigma_B from the c <-> c3 mirror pair (B is achiral: this MUST succeed at dd).

Z_C3 = -conj(Z_C)  =>  the H-mirror acts as w_c3 = conj(w_c) between the two chart disks.
B's real structure sigma = (H-mirror + tau relabeling) acts locally as
w_c -> zeta_5^k conj(w) into the c3 chart for one branch k.  Build dd pairs
(X_B(c@w), X_B(c3@zeta^k conj w)) through the two glue Moebius maps, fit an anti-Moebius
per branch, pick the dd-consistent one.  Output: sigma_B = M o conj (saved as sigmaB.npy).
"""
import numpy as np, mpmath as mp, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from chart_dd import mp_series_eval
from sym8_glue import fit_mobius_from_pairs
import b_config as BC

mp.mp.dps = 40


def x_kmsv(npz, w):
    G = mp_series_eval(npz['Bh'], npz['Bl'], w)
    c = -(mp.mpc(npz['Bh'][8, 7]) + mp.mpc(npz['Bl'][8, 7]))
    return G[8]/(G[7] + c*G[8])


def moeb(m, x):
    al, be, ga, de = [mp.mpc(v) for v in m]
    return (al + be*x)/(ga + de*x)


def X_of(key, ws):
    npz = np.load(BC.NPZ[key], allow_pickle=True)
    mu = np.load(BC.SW + f"muB_a_{key}.npy") if key != 'a' else None
    out = []
    for w in ws:
        x = x_kmsv(npz, w)
        out.append(complex(moeb(mu, x)) if mu is not None else complex(x))
    return np.array(out)


def wgrid(radii, nth=6):
    out = []
    for r in radii:
        for kk in range(nth):
            out.append(mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk + 0.11)/nth))
    return out


def sig_apply(m, x):
    al, be, ga, de = m
    xc = np.conj(np.asarray(x))
    return (al + be*xc)/(ga + de*xc)


if __name__ == "__main__":
    grid = wgrid(np.linspace(0.08, 0.26, 4))
    X1 = X_of('c', grid)
    best = None
    for k in range(5):
        zk = mp.exp(1j*2*mp.pi*k/5)
        X2 = X_of('c3', [zk*mp.conj(w) for w in grid])
        mob = fit_mobius_from_pairs(np.conj(X1), X2)
        al, be, ga, de = mob['alpha'], mob['beta'], mob['gamma'], mob['delta']
        pred = (al + be*np.conj(X1))/(ga + de*np.conj(X1))
        res = np.abs(pred - X2)
        print(f"branch k={k}: med={np.median(res):.2e} max={res.max():.2e}", flush=True)
        if best is None or np.median(res) < best[0]:
            best = (np.median(res), k, (al, be, ga, de))
    med, k, m = best
    print(f"\nBEST branch k={k} med={med:.2e}")
    if med < 1e-10:
        # involution defect
        probes = np.array([0.1 - 0.2j, -0.3 + 0.1j, 0.4 - 0.1j])
        inv = np.abs(sig_apply(m, sig_apply(m, probes)) - probes).max()
        print(f"involution defect: {inv:.2e}")
        np.save(BC.SW + "sigmaB.npy", np.array(m, complex))
        print("sigma_B saved. images:")
        Pc = complex(-0.24339094187298, -0.19286184625628)
        print(f"  sigma(0)      = {sig_apply(m, 0j):.12f}")
        print(f"  sigma(c-pole) = {sig_apply(m, Pc):.12f}   (should be the c3 address)")
    else:
        print("NO dd-consistent branch -- investigate (wrong pairing or chart issue)")
