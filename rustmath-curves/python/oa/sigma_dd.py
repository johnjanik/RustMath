"""dd-accurate real-structure construction from the chart germs.

At each 12-chart the exact germ is phi = S12(w/kappa) with S12 rational and kappa REAL
(measured: -0.6566..., +0.6566..., real to 2e-15).  Hence the curve point at conj(w) has
phi = conj(phi(point at w)) EXACTLY -- so {(X(w), X(conj w))} are (P, sigma P) pairs if a
real structure sigma exists (natural branch).  At the c-chart kappa is complex and the twist
is w -> omega conj(w), omega = kappa_c/conj(kappa_c).  Fit an anti-Moebius through each
chart's pairs (overdetermined: consistency residual = proof the germ pairs lie on ONE
anti-Moebius), then compare sigma_B, sigma_C, sigma_D as maps.  Agreement at dd => the map
is real and sigma is exact; disagreement => not real.
"""
import numpy as np, mpmath as mp, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from chart_dd import mp_series_eval
from sym8_glue import fit_mobius_from_pairs

mp.mp.dps = 40
SW = "/home/john/sweep_2_12_5/"

CH = {k: np.load(SW + f, allow_pickle=True) for k, f in
      [('B', 'm_order12_cycle1_N6000_ddspan.npz'),
       ('C', 'm_glue_b2_N6900_ddspan.npz'),
       ('D', 'm_glue_c_N2400_ddspan.npz'),
       ('E', 'm_glue_c9_N2400_ddspan.npz')]}
MU = {k: np.load(SW + f) for k, f in
      [('B', 'mu_a_bprime.npy'), ('C', 'mu_a_b2.npy'), ('D', 'mu_a_c.npy'), ('E', 'mu_a_c9.npy')]}
k_bp, k_b2 = [mp.mpc(v) for v in np.load(SW + "kappa12_charts.npy")]
k_c = mp.mpc(np.load(SW + "kappa_c.npy")[0])
k_c9 = mp.mpc(np.load(SW + "kappa_c9.npy")[0])
print(f"kappa_bp = {mp.nstr(k_bp, 17)}  (|Im|/|Re| = {abs(k_bp.imag)/abs(k_bp.real):.1e})")
print(f"kappa_b2 = {mp.nstr(k_b2, 17)}  (|Im|/|Re| = {abs(k_b2.imag)/abs(k_b2.real):.1e})")
print(f"kappa_c  = {mp.nstr(k_c, 12)}   kappa_c9 = {mp.nstr(k_c9, 12)}")


def x_kmsv(npz, w):
    G = mp_series_eval(npz['Bh'], npz['Bl'], w)
    c = -(mp.mpc(npz['Bh'][8, 7]) + mp.mpc(npz['Bl'][8, 7]))
    return G[8]/(G[7] + c*G[8])


def moeb(m, x):
    al, be, ga, de = [mp.mpc(v) for v in m]
    return (al + be*x)/(ga + de*x)


def germ_pairs(chart, twist, radii, nth=7):
    """(X(w), X(twist*conj(w))) pairs through the chart + its mu."""
    npz, mu = CH[chart], MU[chart]
    P1, P2 = [], []
    for r in radii:
        for kk in range(nth):
            w = mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk + 0.17)/nth)
            w2 = twist*mp.conj(w)
            P1.append(complex(moeb(mu, x_kmsv(npz, w))))
            P2.append(complex(moeb(mu, x_kmsv(npz, w2))))
    return np.array(P1), np.array(P2)


def fit_sigma(P1, P2):
    """sigma(X) = M(conj X): fit M on conj(P1) -> P2, return (mob, residual array)."""
    mob = fit_mobius_from_pairs(np.conj(P1), P2)
    al, be, ga, de = mob['alpha'], mob['beta'], mob['gamma'], mob['delta']
    pred = (al + be*np.conj(P1))/(ga + de*np.conj(P1))
    return (al, be, ga, de), np.abs(pred - P2)


def sig_apply(m, x):
    al, be, ga, de = m
    xc = np.conj(np.asarray(x))
    return (al + be*xc)/(ga + de*xc)


radii12 = np.linspace(0.12, 0.42, 5)
radii5 = np.linspace(0.10, 0.30, 5)

sigs = {}
for chart, twist, radii in [('B', mp.mpc(1), radii12), ('C', mp.mpc(1), radii12),
                            ('D', k_c/mp.conj(k_c), radii5), ('E', k_c9/mp.conj(k_c9), radii5)]:
    P1, P2 = germ_pairs(chart, twist, radii)
    m, res = fit_sigma(P1, P2)
    sigs[chart] = m
    print(f"\nsigma_{chart}: {len(P1)} pairs, anti-Moebius consistency: "
          f"med={np.median(res):.2e} max={res.max():.2e}")
    print(f"  sigma_{chart}(0)  = {sig_apply(m, 0j):.10f}")
    print(f"  sigma_{chart}(r1) = {sig_apply(m, 0.47182600647013-0.10561240463346j):.10f}")
    print(f"  sigma_{chart}(Xc) = {sig_apply(m, -0.056491567434-0.311094205741j):.10f}")

# pairwise comparison on a test grid
print("\n=== pairwise sigma agreement on grid ===")
rng = np.random.default_rng(3)
grid = (rng.uniform(-0.7, 1.1, 40) + 1j*rng.uniform(-0.6, 0.6, 40))
keys = list(sigs)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        d = np.abs(sig_apply(sigs[keys[i]], grid) - sig_apply(sigs[keys[j]], grid))
        print(f"  sigma_{keys[i]} vs sigma_{keys[j]}: med={np.median(d):.2e} max={d.max():.2e}")
