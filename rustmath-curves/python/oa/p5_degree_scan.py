"""P5 pass 1: value-matching degree scan for phi = P(X)/Q(X) on the glued atlas.

Samples (X_i, phi_i) from all three chart regions:
  A  : X = x_a (top-echelon KMSV coordinate, dd basis), phi from the t=0 hypergeometric
       series phi = sum phiv[n] u^n, u = (w_a/kappa)^2   (mapkit cache, fp64 kappa).
  B' : X = mu_b'(x_b'), phi from the t=1 reversion phi = 1 - s(u12), u12 = w_b/kappa12.
  B2 : X = mu_b2(x_b2), same germ with its own chart scale kappa_b2.

kappa12 / kappa_b2 are calibrated by matching phi across the overlap (12th-root phase fixed
by the axis mirror symmetry: both centers lie on the imaginary axis, so the scales are real
positive), then VALIDATED at held-out points. Then the degree scan (SVD of the value-matching
matrix, X rescaled to the unit disk): expect a sharp sigma_min cliff at degree 24 and no
[14/14] Pade skeleton.
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
from fractions import Fraction as Fr
from chart_dd import mp_series_eval
from phi_vertices import phi_in_u12

mp.mp.dps = 40

MU_STR = None
_Lam = mp.cos(mp.pi/5) / mp.sin(mp.pi/12)
MU = _Lam + mp.sqrt(_Lam*_Lam - 1)
Q_B2 = mp.mpf('0.164275700384606020499257420743316339146629866')   # z_b2 = i/mu

# --- charts and coordinate ---
A = np.load("/home/john/sweep_2_12_5/m_glue_a_N1200_ddspan.npz", allow_pickle=True)
BP = np.load("/home/john/sweep_2_12_5/m_order12_cycle1_N6000_ddspan.npz", allow_pickle=True)
B2 = np.load("/home/john/sweep_2_12_5/m_glue_b2_N6900_ddspan.npz", allow_pickle=True)
mu_bp = np.load("/home/john/sweep_2_12_5/mu_a_bprime.npy")
mu_b2 = np.load("/home/john/sweep_2_12_5/mu_a_b2.npy")


def x_kmsv(npz, w):
    G = mp_series_eval(npz['Bh'], npz['Bl'], w)
    c = -(mp.mpc(npz['Bh'][8, 7]) + mp.mpc(npz['Bl'][8, 7]))
    return G[8] / (G[7] + c*G[8])


def moeb(m, x):
    al, be, ga, de = [mp.mpc(complex(v)) for v in m]
    return (al + be*x) / (ga + de*x)


def w_of(z, p):
    return (z - p*1j) / (z + p*1j)


# --- phi at t=0 (a-region) ---
import mapkit
KAP = mp.mpf(repr(float(mapkit.kappa.real) if hasattr(mapkit.kappa, 'real') else float(mapkit.kappa)))
PHIV = [mp.mpf(repr(float(v.real))) for v in mapkit.phiv]

def phi_a(z):
    u = (w_of(z, mp.mpf(1)) / KAP) ** 2
    acc = mp.mpc(0)
    for n in range(len(PHIV) - 1, -1, -1):
        acc = acc * u + PHIV[n]
    return acc, abs(u)


# --- phi at t=1 (12-point germ), exact rationals ---
L12 = 240
t0 = time.time()
S12 = phi_in_u12(2, 12, 5, L12)          # phi = 1 - s;  returns [1-s0, -s1, ...] i.e. phi series
S12 = [mp.mpf(c.numerator) / mp.mpf(c.denominator) for c in S12]
print(f"phi_in_u12 rationals L={L12} [{time.time()-t0:.0f}s]", flush=True)

def phi_12(u):
    acc = mp.mpc(0)
    for n in range(L12 - 1, -1, -1):
        acc = acc * u + S12[n]
    return acc

def solve_u12(sval):
    """u with phi_12(u) = 1 - sval ... i.e. s(u)=sval; multiplicative Newton, real branch."""
    u = sval ** (mp.mpf(1)/12)
    for _ in range(60):
        s_cur = 1 - phi_12(u)
        u = u * (sval / s_cur) ** (mp.mpf(1)/12)
    return u


def calibrate(chart_center_q, npz_unused, ts):
    """kappa for a 12-point chart: match phi_a on axis points; check constancy."""
    kaps = []
    for t in ts:
        z = mp.mpc(0, mp.mpf(t))
        pa, umag = phi_a(z)
        s = 1 - pa
        u12 = solve_u12(s.real if abs(s.imag) < 1e-25 else s)
        wq = w_of(z, chart_center_q)
        kaps.append(wq / u12)
    spread = max(abs(k - kaps[0]) for k in kaps)
    return kaps[len(kaps)//2], float(spread), [complex(k) for k in kaps[:3]]


print("calibrating kappa12 (b' chart, center MU*i):", flush=True)
k_bp, spread, ks = calibrate(MU, BP, ['2.35', '2.5', '2.65', '2.8'])
print(f"  kappa12 = {complex(k_bp)}   spread across points = {spread:.2e}", flush=True)
print("calibrating kappa_b2 (center i/MU):", flush=True)
k_b2, spread2, ks2 = calibrate(Q_B2, B2, ['0.37', '0.40', '0.43', '0.46'])
print(f"  kappa_b2 = {complex(k_b2)}   spread = {spread2:.2e}", flush=True)

# --- validation: phi continuity at held-out off-axis points ---
def phi_bchart(z, q, kap):
    return phi_12(w_of(z, q) / kap)

for tag, q, kap, zt in [("b'", MU, k_bp, mp.mpc('0.12', '2.55')),
                        ("b2", Q_B2, k_b2, mp.mpc('0.012', '0.415'))]:
    pa, _ = phi_a(zt)
    pb = phi_bchart(zt, q, kap)
    print(f"  held-out phi match ({tag}, off-axis): |phi_a - phi_12| = {abs(pa-pb):.2e}", flush=True)

# --- assemble samples ---
samples = []
# region A
for t in np.linspace(1.35, 2.6, 14):
    for fr in (0.0, 0.04, -0.04, 0.08):
        z = mp.mpc(mp.mpf(float(t))*mp.mpf(repr(fr)), mp.mpf(float(t)))
        ph, umag = phi_a(z)
        if umag < 0.30:
            X = x_kmsv(A, w_of(z, mp.mpf(1)))
            samples.append((complex(X), complex(ph), 'A'))
# region B'
for r in np.linspace(0.08, 0.34, 7):
    for kk in range(6):
        wq = mp.mpf(float(r)) * mp.exp(1j*2*mp.pi*(kk+0.21)/6)
        z = (MU*1j + wq*MU*1j) / (1 - wq)
        X = moeb(mu_bp, x_kmsv(BP, wq))
        ph = phi_12(wq / k_bp)
        samples.append((complex(X), complex(ph), 'B'))
# region B2
for r in np.linspace(0.08, 0.36, 7):
    for kk in range(6):
        wq = mp.mpf(float(r)) * mp.exp(1j*2*mp.pi*(kk+0.13)/6)
        z = (Q_B2*1j + wq*Q_B2*1j) / (1 - wq)
        X = moeb(mu_b2, x_kmsv(B2, wq))
        ph = phi_12(wq / k_b2)
        samples.append((complex(X), complex(ph), 'C'))

X = np.array([s[0] for s in samples])
PH = np.array([s[1] for s in samples])
print(f"\n{len(samples)} samples: |X| in [{np.abs(X).min():.3f},{np.abs(X).max():.3f}], "
      f"|phi| in [{np.abs(PH).min():.2e},{np.abs(PH).max():.2e}]", flush=True)
np.savez("/home/john/sweep_2_12_5/p5_samples.npz", X=X, PHI=PH,
         region=np.array([s[2] for s in samples]))

# --- degree scan ---
S = 0.55
Xs = X / S
print(f"\ndegree scan (X rescaled by {S}):")
print(f"  {'d':>3} {'sigma_min':>12} {'gap':>10} {'rel resid':>12}")
for d in list(range(8, 31)) + [36]:
    rows = []
    for x, y in zip(Xs, PH):
        powers = np.array([x**k for k in range(d + 1)])
        rows.append(np.concatenate([powers, -y*powers]))
    M = np.vstack(rows)
    _, s, Vh = np.linalg.svd(M, full_matrices=False)
    v = Vh[-1].conj()
    p, q = v[:d+1], v[d+1:]
    num = np.zeros_like(Xs); den = np.zeros_like(Xs)
    for k in range(d, -1, -1):
        num = num*Xs + p[k] if False else num
    num = sum(p[k]*Xs**k for k in range(d+1))
    den = sum(q[k]*Xs**k for k in range(d+1))
    rel = np.linalg.norm(num/den - PH) / np.linalg.norm(PH)
    print(f"  {d:>3} {s[-1]:>12.3e} {s[-2]/s[-1]:>10.1e} {rel:>12.3e}", flush=True)
