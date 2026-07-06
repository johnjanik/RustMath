"""Decisive real-structure test: find Moebius M with  phi(M(conj X)) = conj(phi(X)).

Uses the validated d24 generic fit (p5_deg24_fit4, 1e-12 on covered territory) as phi and
the dense sample set as ground-truth (X_i, phi_i).  A real field of definition (in any
coordinate) is EQUIVALENT to the existence of such an M (branch classes all distinct =>
no {0,1,infty}-twist allowed).  Optimize the 6 real params of M by multistart LM on
   e_i = phi4(M(conj X_i)) - conj(phi_i),
restricted to images that stay in covered territory.  Success -> sigma found (gauge + p4
prediction).  Hard failure at O(1) -> the field is not real (conjugate passport pair).
"""
import numpy as np, sys, os
from scipy.optimize import least_squares

SW = "/home/john/sweep_2_12_5/"
F4 = np.load(SW + "p5_deg24_fit4.npz")
D = np.load(SW + "p5_samples6.npz")
X, PH, REG = D['X'], D['PHI'], D['region']
s = float(F4['scale']) if 'scale' in F4.files else 0.65
p_, q_ = F4['p'], F4['q']


def phi4(x):
    xs = np.asarray(x)/s
    pw = xs[..., None]**np.arange(25)
    return (pw@p_)/(pw@q_)


# sanity: fit reproduces samples
err = np.abs(phi4(X) - PH)/(1 + np.abs(PH))
print(f"d24 fit vs samples: med={np.median(err):.2e} max={err.max():.2e}")

# coverage oracle: a point y is "covered" if it is within dcov of some sample X_i
from scipy.spatial import cKDTree
tree = cKDTree(np.c_[X.real, X.imag])
def covered(y, dcov=0.08):
    d, _ = tree.query(np.c_[np.real(y), np.imag(y)])
    return d < dcov


def M_apply(p, x):
    a = p[0] + 1j*p[1]; b = p[2] + 1j*p[3]; c = p[4] + 1j*p[5]
    return (a + b*x)/(1 + c*x)


def resid(p):
    y = M_apply(p, np.conj(X))
    ok = covered(y)
    e = phi4(y) - np.conj(PH)
    r = np.where(ok, e, 0.15)          # uncovered image = fat penalty (drives M to coverage)
    out = np.empty(2*len(r))
    out[0::2], out[1::2] = np.real(r), np.imag(r)
    return out


rng = np.random.default_rng(7)
starts = []
# survivors from sigma_enum (circle reflections m,rho) -> M(x) = m + rho^2/(x - conj m)
for m, rho in [(-0.415-0.418j, 0.304), (0.0978-0.311j, 0.4267), (0.0472-0.167j, 0.4291)]:
    # M(x) = (m x - m conj(m) + rho^2)/(x - conj(m)) -> normalize const 1 in denom: divide by -conj(m)
    cm = np.conj(m)
    a = (rho**2 - m*cm)/(-cm); b = m/(-cm)*1; c = 1/(-cm)
    starts.append([a.real, a.imag, b.real, b.imag, c.real, c.imag])
# identity-ish and random
starts.append([0, 0, 1, 0, 0, 0])
for _ in range(40):
    starts.append(list(rng.normal(0, 0.7, 6)))

best = None
for st in starts:
    try:
        sol = least_squares(resid, st, method='lm', max_nfev=4000)
    except Exception:
        continue
    if best is None or sol.cost < best.cost:
        best = sol
        y = M_apply(sol.x, np.conj(X)); nok = int(covered(y).sum())
        print(f"  new best: rms={np.sqrt(2*sol.cost/len(X)):.3e} covered={nok}/{len(X)}", flush=True)

p = best.x
y = M_apply(p, np.conj(X))
ok = covered(y)
e = np.abs(phi4(y) - np.conj(PH))[ok]
print(f"\nBEST M: rms(all)={np.sqrt(2*best.cost/len(X)):.3e}")
print(f"covered {ok.sum()}/{len(X)}: med|e|={np.median(e):.3e} max={e.max():.3e}")
a = p[0]+1j*p[1]; b = p[2]+1j*p[3]; c = p[4]+1j*p[5]
print(f"M(x) = ({a:.8f} + {b:.8f} x)/(1 + {c:.8f} x)")

# involution check and special-point images
def M1(x): return M_apply(p, x)
def sig(x): return M1(np.conj(x))
pts = {'0': 0j, 'r1': 0.47182600647013-0.10561240463346j, 'r2': -0.32224249915043-0.38542142936930j,
       'Xc': -0.056491567434-0.311094205741j, 'Xc9': 1.152891104470+0.579237023791j,
       'a1': 0.2120091233-0.5634590964j, 'a2': -0.7190508461-0.4168884505j,
       'c11': -0.4944888699-0.4069986879j,
       's1': 0.4775220321+0.0580930745j, 's2': -0.1303535948-0.4094436433j,
       's3': 0.5758909398-0.1287776533j,
       'A6': 0.611-0.085j, 'A7': 0.585-0.120j, 'A8': 0.544-0.139j, 'R4': 0.592-0.079j}
print("\ninvolution + images:")
for lbl, x in pts.items():
    im = sig(x); im2 = sig(im)
    nb = min(pts.items(), key=lambda t: abs(t[1]-im))
    print(f"  sig({lbl:3s}) = {im:.6f}  (sig^2 err {abs(im2-x):.1e})  nearest {nb[0]} d={abs(nb[1]-im):.3f}")
