"""Discrete real-structure scan.

sigma must map {r1, r2} to itself (fix/swap) -> 4 real conditions; the remaining freedom of
the anti-Moebius sigma = M o conj is exactly ONE complex number w3 = sigma(0), which must be
one of the 8 double zeros.  For each (pairing, w3-candidate): M is determined by 3-point
interpolation conj{r1, r2, 0} -> {images}; score the functional equation
phi(sigma(X)) = conj(phi(X)) on the covered B/C rings + involution defect; then LM-polish w3.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

SW = "/home/john/sweep_2_12_5/"
F4 = np.load(SW + "p5_deg24_fit4.npz")
D = np.load(SW + "p5_samples6.npz")
X, PH, REG = D['X'], D['PHI'], D['region']
s = 0.65
p_, q_ = F4['p'], F4['q']

r1 = 0.47182600647013 - 0.10561240463346j
r2 = -0.32224249915043 - 0.38542142936930j
DOUBLES = {'0': 0j, 'a1': 0.2120091233 - 0.5634590964j, 'a2': -0.7190508461 - 0.4168884505j,
           's1': 0.4775220321 + 0.0580930745j, 's2': -0.1303535948 - 0.4094436433j,
           's3': 0.5758909398 - 0.1287776533j,
           'A6': 0.611 - 0.085j, 'A7': 0.585 - 0.120j, 'A8': 0.544 - 0.139j}
POLES = {'Xc': -0.056491567434 - 0.311094205741j, 'Xc9': 1.152891104470 + 0.579237023791j,
         'c11': -0.4944888699 - 0.4069986879j, 'R4': 0.592 - 0.079j}


def phi4(x):
    xs = np.asarray(x) / s
    pw = xs[..., None] ** np.arange(25)
    return (pw @ p_) / (pw @ q_)


# score samples: B/C rings only, and drop the fit's own outliers
mBC = (REG == 'B') | (REG == 'C')
fit_err = np.abs(phi4(X) - PH) / (1 + np.abs(PH))
good = mBC & (fit_err < 1e-8)
Xg, PHg = X[good], PH[good]
tree = cKDTree(np.c_[X.real, X.imag])
print(f"scoring on {good.sum()} clean B/C samples")


def mobius_from_3(zs, ws):
    """Return 2x2 matrix of the Moebius sending zs[i] -> ws[i]."""
    def to01inf(p):
        z0, z1, z2 = p
        return np.array([[z1 - z2, -z0 * (z1 - z2)], [z1 - z0, -z2 * (z1 - z0)]], complex)
    A = to01inf(zs); B = to01inf(ws)
    return np.linalg.inv(B) @ A


def M_apply(Mm, x):
    a, b = Mm[0]; c, d = Mm[1]
    return (a * np.asarray(x) + b) / (c * np.asarray(x) + d)


def score(pair, w3):
    if pair == 'fix':
        zs = [np.conj(r1), np.conj(r2), 0j]; ws = [r1, r2, w3]
    else:
        zs = [np.conj(r1), np.conj(r2), 0j]; ws = [r2, r1, w3]
    Mm = mobius_from_3(zs, ws)
    sig = lambda x: M_apply(Mm, np.conj(x))
    y = sig(Xg)
    d, _ = tree.query(np.c_[y.real, y.imag])
    ok = d < 0.06
    if ok.sum() < 30:
        return None, np.inf, 0
    e = np.abs(phi4(y[ok]) - np.conj(PHg[ok])) / (1 + np.abs(PHg[ok]))
    # involution defect on probes
    probes = np.array([0.3 - 0.2j, -0.2 - 0.3j, 0.5 - 0.1j, 0.1 + 0.2j])
    inv = np.abs(sig(sig(probes)) - probes).max()
    return Mm, float(np.median(e)) + inv, ok.sum()


rows = []
for pair in ('fix', 'swap'):
    for lbl, w3 in DOUBLES.items():
        Mm, sc_, nok = score(pair, w3)
        rows.append((sc_, pair, lbl, w3, nok))
rows.sort(key=lambda t: t[0])
print("\npair  w3-cand  score(med|e|+invol)  n_covered")
for sc_, pair, lbl, w3, nok in rows:
    print(f"  {pair:4s}  sig(0)={lbl:3s}  {sc_:.3e}  {nok}")

# LM polish of w3 for the top 3
print("\n--- LM polish (w3 free) ---")
for sc_, pair, lbl, w3, nok in rows[:3]:
    if not np.isfinite(sc_):
        continue

    def res(pv):
        _, s_, n_ = score(pair, pv[0] + 1j * pv[1])
        return [s_]

    sol = least_squares(res, [w3.real, w3.imag], method='lm', diff_step=1e-4, max_nfev=300)
    w3p = sol.x[0] + 1j * sol.x[1]
    Mm, sp, nokp = score(pair, w3p)
    print(f"  {pair} sig(0)~{lbl}: w3 -> {w3p:.8f}  score {sc_:.2e} -> {sp:.2e} (n={nokp})")
    if sp < 3e-3:
        sig = lambda x: M_apply(Mm, np.conj(x))
        print("    special-point images:")
        for l2, x2 in {**DOUBLES, **POLES}.items():
            im = sig(x2)
            allpts = {**DOUBLES, **POLES, 'r1': r1, 'r2': r2}
            nb = min(allpts.items(), key=lambda t: abs(t[1] - im))
            print(f"      sig({l2:3s}) = {im:.6f} -> nearest {nb[0]} d={abs(nb[1]-im):.3f}")
