"""B-atlas: calibrate the 12-germ scales and build the multi-region sample set.

kappa12 calibration: on the imaginary axis the a-corridor germ (phiv in u=(w_a/KAP)^2,
exact, trust |u|<=3.8) overlaps each 12-chart's disk; solve S12(w_chart/kappa) = phi_corridor
per point (real 1-D root-find near +-0.6566...), demand cross-point consistency.

Samples: region A = corridor (both sides of i), B = ring around z_b, C = ring around i/mu,
each phi from the exact 12-germ, X from the chart + its glue Moebius.
"""
import numpy as np, mpmath as mp, sys, os
sys.path.insert(0, os.path.dirname(__file__))
mp.mp.dps = 45
from chart_dd import mp_series_eval
from phi_vertices import phi_in_u12
import mapkit
import b_config as BC

KAP = mp.mpf(repr(float(mapkit.kappa)))
PHIV = [mp.mpf(repr(float(v.real))) for v in mapkit.phiv]
S12 = [mp.mpf(c.numerator)/mp.mpf(c.denominator) for c in phi_in_u12(2, 12, 5, 240)]


def horner(cs, u):
    acc = mp.mpc(0)
    for n in range(len(cs) - 1, -1, -1):
        acc = acc*u + cs[n]
    return acc


def w_of(z, ctr):
    return (z - ctr)/(z - mp.conj(ctr))


def x_kmsv(npz, w):
    G = mp_series_eval(npz['Bh'], npz['Bl'], w)
    c = -(mp.mpc(npz['Bh'][8, 7]) + mp.mpc(npz['Bl'][8, 7]))
    return G[8]/(G[7] + c*G[8])


def moeb(m, x):
    al, be, ga, de = [mp.mpc(v) for v in m]
    return (al + be*x)/(ga + de*x)


def phi_corridor(z):
    u = (w_of(z, BC.Z_A)/KAP)**2
    if abs(u) > 3.8:
        return None
    return horner(PHIV, u)


def calibrate_k12(key, trange):
    """Solve S12(w_key(it)/kappa) = phi_corridor(it) for real kappa; report spread."""
    zc = BC.CENTER[key]
    kaps = []
    for t in trange:
        z = mp.mpc(0, mp.mpf(repr(float(t))))
        ph = phi_corridor(z)
        if ph is None:
            continue
        wch = w_of(z, zc)

        def f(kap):
            return (horner(S12, wch/kap) - ph).real
        for k0 in (mp.mpf('0.65662764823643'), mp.mpf('-0.65662764823643')):
            try:
                kap = mp.findroot(f, k0)
                chk = horner(S12, wch/kap) - ph
                if abs(chk) < mp.mpf('1e-30'):
                    kaps.append(kap)
                    break
            except Exception:
                continue
    if not kaps:
        raise RuntimeError(f"no calibration points for {key}")
    k = kaps[len(kaps)//2]
    spread = max(abs(x - k) for x in kaps)
    print(f"kappa12({key}) = {mp.nstr(k, 17)}   spread {mp.nstr(spread, 3)}  ({len(kaps)} pts)")
    return k


def build(k_b, k_b2, nang=16, radii=np.linspace(0.10, 0.52, 7)):
    CH = {k: np.load(BC.NPZ[k], allow_pickle=True) for k in ('a', 'b', 'b2')}
    MU = {k: np.load(BC.SW + f"muB_a_{k}.npy") for k in ('b', 'b2')}
    smp = []
    for t in list(np.linspace(1.05, 2.10, 15)) + list(np.linspace(0.476, 0.95, 10)):
        for fr in (0.0, 0.05, -0.05, 0.10):
            z = mp.mpc(mp.mpf(float(t))*mp.mpf(repr(fr)), mp.mpf(float(t)))
            ph = phi_corridor(z)
            if ph is None:
                continue
            smp.append((complex(x_kmsv(CH['a'], w_of(z, BC.Z_A))), complex(ph), 'A'))
    for r in radii:
        for kk in range(nang):
            wq = mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk + 0.21)/nang)
            smp.append((complex(moeb(MU['b'], x_kmsv(CH['b'], wq))),
                        complex(horner(S12, wq/k_b)), 'B'))
            wq2 = mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk + 0.13)/nang)
            smp.append((complex(moeb(MU['b2'], x_kmsv(CH['b2'], wq2))),
                        complex(horner(S12, wq2/k_b2)), 'C'))
    X = np.array([s[0] for s in smp])
    PH = np.array([s[1] for s in smp])
    REG = np.array([s[2] for s in smp])
    np.savez(BC.SW + "pB_samples1.npz", X=X, PHI=PH, region=REG)
    print(f"saved pB_samples1.npz: {len(X)} rows  "
          + " ".join(f"{R}:{(REG == R).sum()}" for R in 'ABC'))
    return X, PH, REG


if __name__ == "__main__":
    k_b = calibrate_k12('b', np.linspace(2.03, 2.14, 12))
    k_b2 = calibrate_k12('b2', np.linspace(0.468, 0.492, 12))
    np.save(BC.SW + "kappaB_12.npy", np.array([complex(k_b), complex(k_b2)]))
    build(k_b, k_b2)
