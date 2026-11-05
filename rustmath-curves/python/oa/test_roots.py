"""Does the recovered map show the [2,12,5] ramification?  Over 0: sigma_0=2^8 1^8 (P has 8 double
+ 8 simple roots).  Over inf: sigma_inf=5^4 1^4 (Q has 4 five-fold + 4 simple).  Over 1: sigma_1=12^2
(P-Q has two 12-fold roots).  Print sorted roots + nearest-neighbour gaps to see the multiplicity
clustering.  Tight clusters -> X is near the Belyi coordinate, structured fit will snap it."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd, jet_recognize as jr
import mpmath as mp
mp.mp.dps = 40
Lu = mapkit.Lu; d = mapkit.d
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2950
vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]; atol = mp.mpf('1e-8')

dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
G = jet_dd.dd_gram(C, n_slices=16)
B = jet_dd.solve_dd_refine(G, vals, 1e-12, jt.tail_weights(N), n_slices=18, iters=8)[0]
X, ech = jr.hauptmodul_mp(B, dim, atol)
o12, p, q = jr.order12_mp(X)
print(f"N={N}  o12={o12:.4e}", flush=True)

def roots_of(coeffs):  # coeffs low->high
    return [complex(r) for r in mp.polyroots([coeffs[i] for i in range(len(coeffs)-1, -1, -1)],
                                             maxsteps=600, extraprec=600)]
def show(name, coeffs, expect):
    rr = roots_of(coeffs)
    rr = sorted(rr, key=lambda z: (round(abs(z), 6), np.angle(z)))
    gaps = [abs(rr[i+1]-rr[i]) for i in range(len(rr)-1)]
    print(f"\n  {name}  ({len(rr)} roots, expect {expect}):", flush=True)
    print("   |root| : " + " ".join(f"{abs(z):.4f}" for z in rr), flush=True)
    print("   nn-gap : " + " ".join(f"{g:.4f}" for g in gaps), flush=True)

show("P    (over 0,   sigma_0=2^8 1^8)", p, "8 double + 8 simple")
show("Q    (over inf, sigma_inf=5^4 1^4)", q, "4 five-fold + 4 simple")
show("P-Q  (over 1,   sigma_1=12^2)", [p[i]-q[i] for i in range(d+1)], "two 12-fold")
