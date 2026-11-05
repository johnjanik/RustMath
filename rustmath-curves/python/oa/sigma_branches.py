"""Branch-complete sigma scan.

phi is Delta-automorphic => at an order-m vertex its H-germ is a series in u^m only, so the
local anti-maps preserving phi(conj-relation) are w -> zeta_m^k conj(w), k = 0..m-1 (times
the kappa twist).  For each chart and each branch build (X(w), X(branch(w))) pairs and fit
an anti-Moebius; the TRUE sigma (if the field is real) = exactly one branch per chart with
dd-level consistency + cross-chart agreement.  Also the swap families: sigma(r1)=r2 pairs
B->C via v = (kappa_b2/kappa_bp) zeta^k conj(w) = -zeta^k conj(w), and D->E via
v = (kappa_c9/kappa_c) zeta5^k conj(w).
"""
import numpy as np, mpmath as mp, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from chart_dd import mp_series_eval
from sym8_glue import fit_mobius_from_pairs

mp.mp.dps = 40
SW = "/home/john/sweep_2_12_5/"

CH = {k: np.load(SW + f, allow_pickle=True) for k, f in
      [('B', 'm_order12_cycle1_N6000_ddspan.npz'), ('C', 'm_glue_b2_N6900_ddspan.npz'),
       ('D', 'm_glue_c_N2400_ddspan.npz'), ('E', 'm_glue_c9_N2400_ddspan.npz')]}
MU = {k: np.load(SW + f) for k, f in
      [('B', 'mu_a_bprime.npy'), ('C', 'mu_a_b2.npy'), ('D', 'mu_a_c.npy'), ('E', 'mu_a_c9.npy')]}
k_c = mp.mpc(np.load(SW + "kappa_c.npy")[0])
k_c9 = mp.mpc(np.load(SW + "kappa_c9.npy")[0])


def x_kmsv(npz, w):
    G = mp_series_eval(npz['Bh'], npz['Bl'], w)
    c = -(mp.mpc(npz['Bh'][8, 7]) + mp.mpc(npz['Bl'][8, 7]))
    return G[8]/(G[7] + c*G[8])


def moeb(m, x):
    al, be, ga, de = [mp.mpc(v) for v in m]
    return (al + be*x)/(ga + de*x)


def X_of(chart, ws):
    return np.array([complex(moeb(MU[chart], x_kmsv(CH[chart], w))) for w in ws])


def wgrid(radii, nth=5):
    out = []
    for r in radii:
        for kk in range(nth):
            out.append(mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk + 0.13)/nth))
    return out


def fit_sig(P1, P2):
    mob = fit_mobius_from_pairs(np.conj(P1), P2)
    al, be, ga, de = mob['alpha'], mob['beta'], mob['gamma'], mob['delta']
    pred = (al + be*np.conj(P1))/(ga + de*np.conj(P1))
    return (al, be, ga, de), np.abs(pred - P2)


def sig_apply(m, x):
    al, be, ga, de = m
    xc = np.conj(np.asarray(x))
    return (al + be*xc)/(ga + de*xc)


r12 = np.linspace(0.10, 0.30, 4)
r5 = np.linspace(0.08, 0.24, 4)
t0 = time.time()
G12 = wgrid(r12)      # 20 pts
G5 = wgrid(r5)

# precompute base X values
XB = X_of('B', G12); XC = X_of('C', G12)
XD = X_of('D', G5); XE = X_of('E', G5)
print(f"[base grids done {time.time()-t0:.0f}s]", flush=True)

results = []
# fix families
for chart, grid, Xbase, m_ord, twist in [
        ('B', G12, XB, 12, mp.mpc(1)), ('C', G12, XC, 12, mp.mpc(1)),
        ('D', G5, XD, 5, k_c/mp.conj(k_c)), ('E', G5, XE, 5, k_c9/mp.conj(k_c9))]:
    for k in range(m_ord):
        zk = twist*mp.exp(1j*2*mp.pi*k/m_ord)
        X2 = X_of(chart, [zk*mp.conj(w) for w in grid])
        m, res = fit_sig(Xbase, X2)
        results.append((float(np.median(res)), float(res.max()), f"{chart}fix k={k}", m))
        print(f"  {chart} fix k={k:2d}: med={np.median(res):.2e} max={res.max():.2e}", flush=True)

# swap families: sigma maps B-chart pts to C-chart pts (r1 <-> r2), D->E (c <-> c9)
for c1, c2, grid, Xbase, m_ord, twist in [
        ('B', 'C', G12, XB, 12, mp.mpc(-1)),
        ('D', 'E', G5, XD, 5, mp.conj(k_c9)/k_c)]:
    for k in range(m_ord):
        zk = twist*mp.exp(1j*2*mp.pi*k/m_ord)
        X2 = X_of(c2, [zk*mp.conj(w) for w in grid])
        m, res = fit_sig(Xbase, X2)
        results.append((float(np.median(res)), float(res.max()), f"{c1}->{c2} k={k}", m))
        print(f"  {c1}->{c2} k={k:2d}: med={np.median(res):.2e} max={res.max():.2e}", flush=True)

results.sort(key=lambda t: t[0])
print("\n=== best branches ===")
for med, mx, name, m in results[:8]:
    print(f"  {name:12s} med={med:.2e} max={mx:.2e}")

# cross-agreement of the best per family group
print("\n=== cross-chart agreement of survivors (med < 1e-6) ===")
surv = [(name, m) for med, mx, name, m in results if med < 1e-6]
rng = np.random.default_rng(5)
grid_t = rng.uniform(-0.7, 1.1, 30) + 1j*rng.uniform(-0.6, 0.6, 30)
for i in range(len(surv)):
    for j in range(i + 1, len(surv)):
        d = np.abs(sig_apply(surv[i][1], grid_t) - sig_apply(surv[j][1], grid_t))
        print(f"  {surv[i][0]} vs {surv[j][0]}: med={np.median(d):.2e} max={d.max():.2e}")
if surv:
    name, m = surv[0]
    print(f"\n=== sigma from {name} ===")
    pts = {'0': 0j, 'r1': 0.47182600647013-0.10561240463346j,
           'r2': -0.32224249915043-0.38542142936930j,
           'Xc': -0.056491567434-0.311094205741j, 'Xc9': 1.152891104470+0.579237023791j,
           'a1': 0.2120091233-0.5634590964j, 'a2': -0.7190508461-0.4168884505j,
           'c11': -0.4944888699-0.4069986879j,
           's1': 0.4775220321+0.0580930745j, 's2': -0.1303535948-0.4094436433j,
           's3': 0.5758909398-0.1287776533j, 'A6': 0.611-0.085j, 'A7': 0.585-0.120j,
           'A8': 0.544-0.139j, 'R4': 0.592-0.079j}
    for lbl, x in pts.items():
        im = sig_apply(m, x)
        nb = min(pts.items(), key=lambda t: abs(t[1]-im))
        print(f"  sig({lbl:3s}) = {im:.8f} -> nearest {nb[0]:3s} d={abs(nb[1]-im):.4f}")
    np.save(SW + "sigma_mob.npy", np.array(m, complex))
    print("saved sigma_mob.npy")
