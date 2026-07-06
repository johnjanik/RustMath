"""(3,4,3) achiral member: kappa calibrations + multi-region samples.

kappa_a: Gamma-formula (mapkit pattern) for the t=0 germ at the order-3 a-vertex,
VALIDATED by the b-overlap consistency. kappa_b (order-4, t=1 germ) calibrated on the
imaginary-axis overlap; kappa_c (order-3 pole germ) on the i -> z_c segment using the
a-germ where it is trusted. Samples: corridor (A) + b-ring (B) + c-ring (C).
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
from math import gamma as fgamma
from chart_dd import mp_series_eval
from phi_vertices import phi_in_u0, phi_in_u12, phi_in_u5
import c343_config as CC

mp.mp.dps = 45
SW = CC.SW
a_, b_, c_ = 3, 4, 3
A_ = 0.5*(1 + 1/a_ - 1/b_ - 1/c_)
B_ = 0.5*(1 + 1/a_ - 1/b_ + 1/c_)
C_ = 1 + 1/a_
import math
Lam = (math.cos(math.pi/a_)*math.cos(math.pi/b_) + math.cos(math.pi/c_)) / \
      (math.sin(math.pi/a_)*math.sin(math.pi/b_))
MUf = Lam + math.sqrt(Lam*Lam - 1)
KAP_A = ((MUf - 1)/(MUf + 1))*fgamma(2 - C_)*fgamma(C_ - A_)*fgamma(C_ - B_) / \
        (fgamma(1 - A_)*fgamma(1 - B_)*fgamma(C_))
print(f"mu = {MUf:.15f}   kappa_a (Gamma formula) = {KAP_A:.15f}")

KAP = mp.mpf(repr(KAP_A))
MU = mp.mpf(repr(MUf))
Z_A = mp.mpc(0, 1)
Z_B = mp.mpc(0, 1)*MU
Z_C = mp.mpc('0.780445410501160217152658739859', '1.13669947519517276543458017678')

U0 = [mp.mpf(x.numerator)/mp.mpf(x.denominator) for x in phi_in_u0(3, 4, 3, 120)]
U4 = [mp.mpf(x.numerator)/mp.mpf(x.denominator) for x in phi_in_u12(3, 4, 3, 160)]
U3 = [mp.mpf(x.numerator)/mp.mpf(x.denominator) for x in phi_in_u5(3, 4, 3, 120)]


def horner(cs, u):
    acc = mp.mpc(0)
    for n in range(len(cs) - 1, -1, -1):
        acc = acc*u + cs[n]
    return acc


def w_of(z, ctr):
    return (z - ctr)/(z - mp.conj(ctr))


def x3(npz, w):
    G = mp_series_eval(npz['Bh'], npz['Bl'], w)
    c = -(mp.mpc(npz['Bh'][2, 1]) + mp.mpc(npz['Bl'][2, 1]))
    return G[2]/(G[1] + c*G[2])


UMAX = 1.15          # measured germ radius ~1.66 (u0/u3), 1.51 (u4); trust to ~0.7x


def phi_a(z, umax=UMAX):
    u = w_of(z, Z_A)/KAP
    if abs(u) > umax:
        return None
    return horner(U0, u)


def calibrate(key, germ, zc, trange, order):
    """kappa for a chart germ by matching phi_a on the overlap. Seeds from germ inversion:
    order-4 (t=1): phi-1 = -v^4(1+...) => v0 = (1-pa)^{1/4} x i^k;
    order-3 pole: 1/phi = u^3(1+...) => u0 = (1/pa)^{1/3} x zeta_3^k."""
    kaps = []
    for t in trange:
        z = Z_A + mp.mpf(repr(float(t)))*(zc - Z_A)
        pa = phi_a(z)
        if pa is None:
            continue
        wch = w_of(z, zc)

        def f(kap):
            v = horner(germ, wch/kap)
            return (v - pa) if key == 'b' else (1/v - pa)
        cands = []
        if key == 'b':
            v0 = (1 - pa)**(mp.mpf(1)/4)
            roots_ = [v0*mp.exp(1j*mp.pi*k/2) for k in range(4)]
        else:
            v0 = (1/pa)**(mp.mpf(1)/3)
            roots_ = [v0*mp.exp(2j*mp.pi*k/3) for k in range(3)]
        for vr in roots_:
            if abs(vr) < 1e-20:
                continue
            cands.append(wch/vr)
        best = None
        for k0 in cands:
            try:
                kap = mp.findroot(f, k0, tol=1e-40, maxsteps=80)
            except Exception:
                continue
            r = abs(f(kap))
            if best is None or r < best[1]:
                best = (kap, r)
        if best and best[1] < mp.mpf('1e-25'):
            kaps.append(best[0])
    if not kaps:
        raise RuntimeError(f"calibration {key} failed")
    # branch coherence: cluster around the median by angle
    k = kaps[len(kaps)//2]
    kaps = [x for x in kaps if abs(x - k) < 0.1*abs(k)]
    k = kaps[len(kaps)//2]
    spread = max(abs(x - k) for x in kaps)
    print(f"kappa_{key} = {mp.nstr(k, 20)}   spread {mp.nstr(spread, 3)}  ({len(kaps)} pts)")
    return k


if __name__ == "__main__":
    CH = {k: np.load(SW + f"mC_{k}_ddspan.npz", allow_pickle=True) for k in ('a', 'b', 'c')}
    MUB = np.load(SW + "muC_a_b.npy")
    MUC = np.load(SW + "muC_a_c.npy")

    def moeb(m, x):
        al, be, ga, de = [mp.mpc(v) for v in m]
        return (al + be*x)/(ga + de*x)

    # kappa_b: axis overlap: |w_b(it)| <= 0.5 and phi_a trusted
    k_b = calibrate('b', U4, Z_B, np.linspace(0.42, 0.60, 10), 4)
    # kappa_c: i -> z_c segment
    k_c = calibrate('c', U3, Z_C, np.linspace(0.42, 0.60, 10), 3)
    np.save(SW + "kappaC.npy", np.array([complex(KAP), complex(k_b), complex(k_c)]))

    smp = []
    for t in np.linspace(0.30, 0.72, 20):
        for fr in (0.0, 0.05, -0.05, 0.10):
            z = Z_A + mp.mpf(repr(float(t)))*(Z_B - Z_A)*(1 + mp.mpf(repr(fr))*mp.mpc(0, 1))
            ph = phi_a(z)
            if ph is None:
                continue
            smp.append((complex(x3(CH['a'], w_of(z, Z_A))), complex(ph), 'A'))
    for r in np.linspace(0.06, 0.30, 7):
        for kk in range(8):
            wq = mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk + 0.21)/8)
            smp.append((complex(moeb(MUB, x3(CH['b'], wq))), complex(horner(U4, wq/k_b)), 'B'))
            wq2 = mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk + 0.13)/8)
            smp.append((complex(moeb(MUC, x3(CH['c'], wq2))),
                        complex(1/horner(U3, wq2/k_c)), 'C'))
    X = np.array([s[0] for s in smp])
    PH = np.array([s[1] for s in smp])
    REG = np.array([s[2] for s in smp])
    np.savez(SW + "pC_samples1.npz", X=X, PHI=PH, region=REG)
    print(f"saved pC_samples1.npz: {len(X)} rows  "
          + " ".join(f"{R}:{(REG == R).sum()}" for R in 'ABC'))
