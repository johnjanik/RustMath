"""Round 2: determined-M sigma hypotheses scored on ALL regions.

Each hypothesis = 3 point-mappings conj(z_i) -> w_i determining M.  Score the functional
equation phi(sigma X) = conj(phi X) separately on B/C (12-rings), D (c-ring), E (c9-ring),
A (corridor), using phi4 evaluation everywhere (no coverage gate: D/E have |phi| up to 3e3;
a pole mismatch shows as O(1) relative error).  Also report involution defect and images of
all special points.  Relative error metric: |phi4(sig X) - conj(phi)| / (1 + |phi|).
"""
import numpy as np

SW = "/home/john/sweep_2_12_5/"
F4 = np.load(SW + "p5_deg24_fit4.npz")
D = np.load(SW + "p5_samples6.npz")
X, PH, REG = D['X'], D['PHI'], D['region']
s = 0.65
p_, q_ = F4['p'], F4['q']

r1 = 0.47182600647013 - 0.10561240463346j
r2 = -0.32224249915043 - 0.38542142936930j
Xc = -0.056491567434 - 0.311094205741j
Xc9 = 1.152891104470 + 0.579237023791j
c11 = -0.4944888699 - 0.4069986879j
DOUBLES = {'0': 0j, 'a1': 0.2120091233 - 0.5634590964j, 'a2': -0.7190508461 - 0.4168884505j,
           's1': 0.4775220321 + 0.0580930745j, 's2': -0.1303535948 - 0.4094436433j,
           's3': 0.5758909398 - 0.1287776533j,
           'A6': 0.611 - 0.085j, 'A7': 0.585 - 0.120j, 'A8': 0.544 - 0.139j}
POLES = {'Xc': Xc, 'Xc9': Xc9, 'c11': c11, 'R4': 0.592 - 0.079j}
fit_err = None


def phi4(x):
    xs = np.asarray(x) / s
    pw = xs[..., None] ** np.arange(25)
    return (pw @ p_) / (pw @ q_)


def mobius_from_3(zs, ws):
    def to01inf(p):
        z0, z1, z2 = p
        return np.array([[z1 - z2, -z0 * (z1 - z2)], [z1 - z0, -z2 * (z1 - z0)]], complex)
    return np.linalg.inv(to01inf(ws)) @ to01inf(zs)


def M_apply(Mm, x):
    a, b = Mm[0]; c, d = Mm[1]
    return (a * np.asarray(x) + b) / (c * np.asarray(x) + d)


err0 = np.abs(phi4(X) - PH) / (1 + np.abs(PH))
clean = err0 < 1e-8
print(f"clean samples: {clean.sum()}/{len(X)} "
      + " ".join(f"{R}:{(clean & (REG == R)).sum()}" for R in 'ABCDE'))

HYPS = []
for rmode, rmap in [('Rfix', [(r1, r1), (r2, r2)]), ('Rswap', [(r1, r2), (r2, r1)])]:
    for lbl, w3 in DOUBLES.items():
        HYPS.append((f"{rmode}+0->{lbl}", rmap + [(0j, w3)]))
    for cl, cw in [('c->c9', (Xc, Xc9)), ('c9->c', (Xc9, Xc)), ('c->c11', (Xc, c11)),
                   ('c9->c11', (Xc9, c11)), ('c11->c', (c11, Xc)), ('c11->c9', (c11, Xc9)),
                   ('cfix', (Xc, Xc)), ('c9fix', (Xc9, Xc9)), ('c11fix', (c11, c11))]:
        HYPS.append((f"{rmode}+{cl}", rmap + [cw]))

rows = []
for name, maps in HYPS:
    zs = [np.conj(u) for u, v in maps]
    ws = [v for u, v in maps]
    try:
        Mm = mobius_from_3(zs, ws)
    except Exception:
        continue
    sig = lambda x: M_apply(Mm, np.conj(x))
    probes = np.array([0.3 - 0.2j, -0.2 - 0.3j, 0.5 - 0.1j, 0.1 + 0.2j, -0.6 + 0.1j])
    inv = np.abs(sig(sig(probes)) - probes).max()
    sc = {}
    for R in ('B', 'C', 'D', 'E', 'A'):
        m = clean & (REG == R)
        if m.sum() == 0:
            sc[R] = np.nan; continue
        y = sig(X[m])
        e = np.abs(phi4(y) - np.conj(PH[m])) / (1 + np.abs(PH[m]))
        sc[R] = float(np.median(e))
    tot = np.nanmax([sc[R] for R in 'BCDE'])
    rows.append((tot, inv, name, sc, Mm))

rows.sort(key=lambda t: (t[0] + t[1]))
print("\nname                     invol     medB      medC      medD      medE      medA")
for tot, inv, name, sc, Mm in rows[:14]:
    print(f"{name:24s} {inv:.2e}  " + "  ".join(f"{sc[R]:.2e}" for R in 'BCDEA'))

# full report for the best
tot, inv, name, sc, Mm = rows[0]
sig = lambda x: M_apply(Mm, np.conj(x))
print(f"\n=== BEST: {name} ===")
print(f"M = {np.round(Mm, 8).tolist()}")
allpts = {**DOUBLES, **POLES, 'r1': r1, 'r2': r2}
for lbl, x in allpts.items():
    im = sig(x)
    nb = min(allpts.items(), key=lambda t: abs(t[1] - im))
    print(f"  sig({lbl:3s}) = {im:.8f} -> nearest {nb[0]:3s} d={abs(nb[1]-im):.4f}")
