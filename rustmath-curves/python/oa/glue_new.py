"""Generic atlas extension: refine + glue a new chart, return its center's global-X address.

Given a new chart (dd-refined basis npz + H-center), pick a glued partner chart, sample the
segment between centers (plus transverse offsets), extract the top-echelon coordinate on both
sides, and Moebius-fit  X_global = mu_new(x_new)  against  X_global = mu_partner(x_partner).
The new center's address is mu_new(0). A-root pins need no phi-germ calibration.

Usage: python3 glue_new.py <new_basis.npz> <zre> <zim> [partner]
   partner in {a, bprime, b2, c, c9}; default a.
"""
import numpy as np, mpmath as mp, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from chart_dd import mp_series_eval
from sym8_glue import fit_mobius_from_pairs

mp.mp.dps = 40

_Lam = mp.cos(mp.pi/5)/mp.sin(mp.pi/12); MU_ = _Lam + mp.sqrt(_Lam*_Lam-1)
PARTNERS = {
    'a':      ("m_glue_a_N1200_ddspan.npz",          mp.mpc(0, 1),                 None),
    'bprime': ("m_order12_cycle1_N6000_ddspan.npz",  mp.mpc(0, 1)*MU_,             "mu_a_bprime.npy"),
    'b2':     ("m_glue_b2_N6900_ddspan.npz",
               mp.mpc(0, 1)*mp.mpf('0.164275700384606020499257420743316339146629866'), "mu_a_b2.npy"),
    'c':      ("m_glue_c_N2400_ddspan.npz",
               mp.mpc('0.793538482569228956003002774509736302229259206',
                      '0.608520070894728592568281209719200629678206803'),          "mu_a_c.npy"),
    'c9':     ("m_glue_c9_N2400_ddspan.npz",
               mp.mpc('-0.793538482569228956003002774509736302229259924',
                      '0.608520070894728592568281209719200629678206803'),          "mu_a_c9.npy"),
}
SW = "/home/john/sweep_2_12_5/"


def w_of(z, ctr):
    return (z - ctr)/(z - mp.conj(ctr))


def x_kmsv(npz, w):
    G = mp_series_eval(npz['Bh'], npz['Bl'], w)
    c = -(mp.mpc(npz['Bh'][8, 7]) + mp.mpc(npz['Bl'][8, 7]))
    return G[8]/(G[7] + c*G[8])


def moeb(m, x):
    al, be, ga, de = [mp.mpc(v) for v in m]
    return (al + be*x)/(ga + de*x)


def glue_chart(new_npz_path, znew, partner='a', rmax_new=0.5, rmax_p=0.62, verbose=True):
    NEW = np.load(new_npz_path, allow_pickle=True)
    pf, zp, mupath = PARTNERS[partner]
    P = np.load(SW + pf, allow_pickle=True)
    mu_p = np.load(SW + mupath) if mupath else None
    dirv = znew - zp
    zs = []
    for t in np.linspace(0.30, 0.62, 12):
        for s_ in (0.0, 0.05, -0.05, 0.09):
            z = zp + mp.mpf(float(t))*dirv + mp.mpf(repr(s_))*mp.mpc(0, 1)*dirv
            if z.imag <= 0:
                continue
            if abs(w_of(z, znew)) <= rmax_new and abs(w_of(z, zp)) <= rmax_p:
                zs.append(z)
    if len(zs) < 12:
        raise RuntimeError(f"only {len(zs)} overlap points (partner {partner})")
    XN = np.array([complex(x_kmsv(NEW, w_of(z, znew))) for z in zs])
    if mu_p is None:
        XG = np.array([complex(x_kmsv(P, w_of(z, zp))) for z in zs])
    else:
        XG = np.array([complex(moeb(mu_p, x_kmsv(P, w_of(z, zp)))) for z in zs])
    mob = fit_mobius_from_pairs(XN, XG)
    al, be, ga, de = mob['alpha'], mob['beta'], mob['gamma'], mob['delta']
    res = np.abs((al + be*XN)/(ga + de*XN) - XG)
    addr = al/ga
    if verbose:
        print(f"  {os.path.basename(new_npz_path)} via {partner}: {len(zs)} pairs, "
              f"max|res|={res.max():.2e}, address X = {addr:.12f}")
    return np.array([al, be, ga, de]), complex(addr), float(res.max())


if __name__ == "__main__":
    path = sys.argv[1]
    znew = mp.mpc(sys.argv[2], sys.argv[3])
    partner = sys.argv[4] if len(sys.argv) > 4 else 'a'
    mu, addr, r = glue_chart(path, znew, partner)
    out = path.replace("_ddspan.npz", "_mu.npy")
    np.save(out, mu)
    print(f"saved {out}")
