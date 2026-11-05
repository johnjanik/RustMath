"""Systematic search for the real structure sigma = reflection across a circle/line.

Parameterize: circle family sigma(x) = m + rho^2/conj(x - m)  (params m in C, rho > 0);
line family sigma(x) = e^{i th} conj(x) + t with e^{i th} conj(t) + t = 0.
Constraints from dd-quality addresses only:
  {r1, r2}   12-points        : both fixed (1+1 real eqs) or swapped (2)
  {Xc, Xc9}  quintuple poles  : both fixed or swapped, or EXCLUDED (pair with fuzzy pts)
  {0, a1, a2} dd double zeros : each fixed, paired among themselves, or EXCLUDED
Fit every combination with >= 4 real equations, rank by residual, and for survivors print
predictions: images of c11 (5-pole #4 candidate), softs, crowd.
"""
import numpy as np, itertools, mpmath as mp
from scipy.optimize import least_squares

r1  = complex(0.47182600647013, -0.10561240463346)
r2  = complex(-0.32224249915043, -0.38542142936930)
Xc  = complex(-0.056491567434, -0.311094205741)
Xc9 = complex(1.152891104470, 0.579237023791)
a0  = 0.0 + 0.0j
a1  = complex(0.2120091233, -0.5634590964)
a2  = complex(-0.7190508461, -0.4168884505)
c11 = complex(-0.4944888699, -0.4069986879)
softs = {'s1': complex(0.4775220321, 0.0580930745),
         's2': complex(-0.1303535948, -0.4094436433),
         's3': complex(0.5758909398, -0.1287776533)}
crowd = {'A6': complex(0.611, -0.085), 'A7': complex(0.585, -0.120),
         'A8': complex(0.544, -0.139), 'R4': complex(0.592, -0.079)}


def sig_circle(p, x):
    m = p[0] + 1j*p[1]; rho2 = p[2]**2
    return m + rho2/np.conj(x - m)


def sig_line(p, x):
    th, tr, ti = p
    t = tr + 1j*ti
    # involution requires e^{i th} conj(t) + t = 0; enforce softly via residual row
    return np.exp(1j*th)*np.conj(x) + t


def resid(p, fam, fixed, swaps):
    f = sig_circle if fam == 'C' else sig_line
    rows = []
    for pt in fixed:
        d = f(p, pt) - pt
        rows += [d.real, d.imag]
    for u, v in swaps:
        d = f(p, u) - v
        rows += [d.real, d.imag]
    if fam == 'L':
        th, tr, ti = p
        g = np.exp(1j*th)*np.complex128(tr - 1j*ti) + (tr + 1j*ti)
        rows += [g.real, g.imag]
    return np.array(rows)


# hypothesis space
R_opts = [("Rfix", [r1, r2], []), ("Rswap", [], [(r1, r2)])]
C_opts = [("Cfix", [Xc, Xc9], []), ("Cswap", [], [(Xc, Xc9)]), ("Cout", [], [])]
D_pts = {'0': a0, 'a1': a1, 'a2': a2}
D_opts = []
for combo in [
    ("Dall-fix", ['0', 'a1', 'a2'], []),
    ("D0fix", ['0'], []), ("Da1fix", ['a1'], []), ("Da2fix", ['a2'], []),
    ("D0a1", [], [('0', 'a1')]), ("D0a2", [], [('0', 'a2')]), ("Da1a2", [], [('a1', 'a2')]),
    ("D0fix+a1a2", ['0'], [('a1', 'a2')]),
    ("Da1fix+0?", ['a1'], []),
    ("Dout", [], []),
]:
    name, fx, sw = combo
    D_opts.append((name, [D_pts[k] for k in fx], [(D_pts[u], D_pts[v]) for u, v in sw]))

results = []
for (rn, rf, rs), (cn, cf, cs), (dn, df, ds) in itertools.product(R_opts, C_opts, D_opts):
    fixed = rf + cf + df
    swaps = rs + cs + ds
    neq = 2*len(fixed) + 2*len(swaps)
    if neq < 4:
        continue
    for fam in ('C', 'L'):
        # multistart
        best = None
        for m0 in [(0, 0, 1), (0.3, -0.2, 0.7), (-0.3, 0.3, 1.5), (0.5, 0.5, 2.0), (0.0, -0.5, 0.5)]:
            try:
                sol = least_squares(resid, m0, args=(fam, fixed, swaps), method='lm', max_nfev=2000)
            except Exception:
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        if best is None:
            continue
        rn_ = np.sqrt(2*best.cost)
        results.append((rn_, fam, f"{rn}+{cn}+{dn}", best.x, neq))

results.sort(key=lambda t: t[0])
print("top hypotheses by residual (neq = number of real equations):")
for rn_, fam, name, p, neq in results[:12]:
    print(f"  {rn_:.3e}  fam={fam}  {name}  neq={neq}")

# detailed predictions for the best few DISTINCT hypotheses with residual < 1e-3
print("\n=== survivors (resid < 1e-3) ===")
seen = set()
for rn_, fam, name, p, neq in results:
    if rn_ > 1e-3 or name in seen:
        continue
    seen.add(name)
    f = sig_circle if fam == 'C' else sig_line
    print(f"\n--- {name} fam={fam} resid={rn_:.2e} params={np.round(p,6)} ---")
    print(f"  sigma(0)   = {f(p, a0):.8f}")
    print(f"  sigma(c11) = {f(p, c11):.8f}   <- 4th-5-pole prediction (or c11 partner)")
    for lbl, x in {**softs, **crowd}.items():
        img = f(p, x)
        # nearest known special point
        cands = {'r1': r1, 'r2': r2, 'Xc': Xc, 'Xc9': Xc9, '0': a0, 'a1': a1, 'a2': a2,
                 'c11': c11, **softs, **crowd}
        nb = min(cands.items(), key=lambda t: abs(t[1] - img))
        print(f"  sigma({lbl}) = {img:.6f} -> nearest {nb[0]} (d={abs(nb[1]-img):.3f})")
