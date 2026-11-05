"""Glue a B-atlas chart to B's a-chart coordinate X_B (:= top-echelon x of mB_a).

All B charts are dumped in ONE group presentation (base 1), so same-z overlap sampling is
exact.  Overlap = segment from z_a = i toward the chart center (t in [tlo, thi]) plus
transverse offsets; Moebius fit from (x_new, X_B) pairs; address = mu(0).

Usage: python3 b_glue.py <chart-key: b|b2|c> [tlo thi rmax_new rmax_a]
"""
import numpy as np, mpmath as mp, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from chart_dd import mp_series_eval
from sym8_glue import fit_mobius_from_pairs
import b_config as BC

mp.mp.dps = 40


def w_of(z, ctr):
    return (z - ctr)/(z - mp.conj(ctr))


def x_kmsv(npz, w):
    G = mp_series_eval(npz['Bh'], npz['Bl'], w)
    c = -(mp.mpc(npz['Bh'][8, 7]) + mp.mpc(npz['Bl'][8, 7]))
    return G[8]/(G[7] + c*G[8])


DEFAULT_RANGE = {           # per-chart overlap windows (both disk constraints satisfied)
    'b':  (0.30, 0.62),
    'b2': (0.61, 0.84),     # z_b2 = i/mu is BELOW i: |w_b2|<=0.5 needs Im z <= 0.493
    'c':  (0.30, 0.62),
    'c3': (0.30, 0.62),
}


def glue(key, tlo=None, thi=None, rmax_new=0.50, rmax_a=0.62):
    if tlo is None or thi is None:
        tlo, thi = DEFAULT_RANGE.get(key, (0.30, 0.62))
    A = np.load(BC.NPZ['a'], allow_pickle=True)
    NEW = np.load(BC.NPZ[key], allow_pickle=True)
    za, zn = BC.CENTER['a'], BC.CENTER[key]
    dirv = zn - za
    zs = []
    for t in np.linspace(tlo, thi, 12):
        for s_ in (0.0, 0.05, -0.05, 0.09):
            z = za + mp.mpf(float(t))*dirv + mp.mpf(repr(s_))*mp.mpc(0, 1)*dirv
            if z.imag <= 0:
                continue
            if abs(w_of(z, zn)) <= rmax_new and abs(w_of(z, za)) <= rmax_a:
                zs.append(z)
    if len(zs) < 12:
        raise RuntimeError(f"only {len(zs)} overlap points")
    XN = np.array([complex(x_kmsv(NEW, w_of(z, zn))) for z in zs])
    XG = np.array([complex(x_kmsv(A, w_of(z, za))) for z in zs])
    mob = fit_mobius_from_pairs(XN, XG)
    al, be, ga, de = mob['alpha'], mob['beta'], mob['gamma'], mob['delta']
    res = np.abs((al + be*XN)/(ga + de*XN) - XG)
    addr = al/ga
    print(f"B-glue {key}: {len(zs)} pairs  max|res|={res.max():.2e} med={np.median(res):.2e}")
    print(f"  address X_B({key}-center) = {addr:.14f}")
    mu = np.array([al, be, ga, de])
    np.save(BC.SW + f"muB_a_{key}.npy", mu)
    print(f"  saved muB_a_{key}.npy")
    return mu, complex(addr), float(res.max())


if __name__ == "__main__":
    key = sys.argv[1]
    args = [float(x) for x in sys.argv[2:6]] if len(sys.argv) > 2 else []
    glue(key, *args)
