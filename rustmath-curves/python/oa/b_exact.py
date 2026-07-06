"""Phase IV for B: canonical antipodal gauge + reality certificate + PSLQ probes.

From the 240-digit ladder configuration: (1) exact sigma from three known pairings;
(2) gauge G: origin-double -> 0 (already), its sigma-partner -> infinity, scale so
sigma = -1/conj(x), rotation so r1 > 0 real; (3) transform all roots, rebuild P', Q';
(4) certificate: twisted-reality of coefficients; (5) PSLQ probes of the real invariants.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
import b_config as BC

mp.mp.dps = 240

# ---- load ladder configuration -------------------------------------------
TH = []
with open(BC.SW + "pB_ladder_theta.txt") as f:
    for line in f:
        a, b = line.split()
        TH.append(mp.mpc(a, b))
assert len(TH) == 25
d = TH[0:7]; z = TH[7:15]; q = TH[15:19]; s = TH[19:23]
lam, c = TH[23], TH[24]
R1 = mp.mpc('-0.84076601151413', '0.38995761807378')
R2 = mp.mpc('-0.35478102590543', '-0.02962349269773')
doubles = [mp.mpc(0)] + list(d)

# ---- exact sigma from three pairings (240 digits) --------------------------
# pairings known discretely: 0-double <-> d[0] (=SIG0 continuation), r1 <-> r2,
# c-pole (q[0]) <-> c4-pole (q[2])
def mobius_from_3(zs, ws):
    def to01inf(p):
        z0, z1, z2 = p
        return mp.matrix([[z1 - z2, -z0*(z1 - z2)], [z1 - z0, -z2*(z1 - z0)]])
    A = to01inf(zs); B = to01inf(ws)
    C = B**-1*A
    return (C[0, 1], C[0, 0], C[1, 1], C[1, 0])     # (al, be, ga, de)


SGM = mobius_from_3([mp.conj(mp.mpc(0)), mp.conj(R1), mp.conj(q[0])],
                    [d[0], R2, q[2]])


def sig(x):
    al, be, ga, de = SGM
    xc = mp.conj(x)
    return (al + be*xc)/(ga + de*xc)


# verify sigma at 240 digits on the other pairings
print("sigma checks (should be ~1e-238):")
print("  |sig(R2) - R1| =", mp.nstr(abs(sig(R2) - R1), 3))
print("  |sig(q[1]) - q[3]| =", mp.nstr(abs(sig(q[1]) - q[3]), 3),
      "  (c3 <-> c2 pairing)")
print("  |sig(d[0]) - 0| =", mp.nstr(abs(sig(d[0])), 3))

# ---- gauge: 0 -> 0, d[0] -> inf, sigma -> -1/conj, r1 -> positive real ------
g0 = d[0]


def G1(x):
    return g0*x/(g0 - x)


# sigma in G1 coords: s1(x) = G1(sig(G1inv(x))); G1inv(y) = g0 y/(g0 + y)
def G1inv(y):
    return g0*y/(g0 + y)


def s1(x):
    return G1(sig(G1inv(x)))


# s1(x) = rho/conj(x): rho = s1(1)*conj(1)
rho = s1(mp.mpc(1))
print("\nrho = s1(1) =", mp.nstr(rho, 25), " (want real negative)")
print("  |Im rho| =", mp.nstr(abs(rho.imag), 3))
sc_ = mp.sqrt(-rho.real)


def G2(x):
    return G1(x)/sc_


# now sigma'' = -1/conj(x); rotation: r1'' -> positive real
r1pp = G2(R1)
phase = r1pp/abs(r1pp)


def G(x):
    return G2(x)/phase


# verify final sigma
def sfin(x):
    return G(sig(G1inv(G2(mp.mpc(1))*0 + 0) if False else 0)) if False else None


probes = [mp.mpc('0.3', '-0.2'), mp.mpc('-0.5', '0.4'), mp.mpc('1.2', '0.7')]
defect = max(abs(G(sig(x)) + 1/mp.conj(G(x))) for x in probes)
print("final gauge: |G(sig(x)) + 1/conj(G(x))| =", mp.nstr(defect, 3))
print("G(R1) =", mp.nstr(G(R1), 25))
print("G(R2) =", mp.nstr(G(R2), 25), " (should be -1/G(R1))")

# ---- transform all roots, rebuild P', Q' -----------------------------------
dbl_new = [G(x) for x in doubles[:1] + list(d[1:])]   # 0 stays 0; d[0] -> inf (dropped)
zer_new = [G(x) for x in z]
qui_new = [G(x) for x in q]
sim_new = [G(x) for x in s]
w_new = [G(R1), G(R2)]

# P' has a double at infinity (d[0] -> inf): deg P' = 24 - 2 = 22
# Q' deg 24; phi' = P'/Q' with deg-drop bookkeeping at infinity.
def prod_roots(rs):
    p = [mp.mpc(1)]
    for r in rs:
        p2 = [mp.mpc(0)]*(len(p) + 1)
        for i, pi in enumerate(p):
            p2[i] += -r*pi
            p2[i + 1] += pi
        p = p2
    return p


Ppoly = prod_roots([x for x in dbl_new for _ in range(2)] + zer_new)   # deg 22
Qpoly = prod_roots([x for x in qui_new for _ in range(5)] + sim_new)   # deg 24
Wpoly = prod_roots([w_new[0]]*12 + [w_new[1]]*12)                      # deg 24

# lambda', c' from matching:  phi = P/Q in old coords = kappa * P'(y)/Q'(y) in new;
# determine kappa by evaluating phi at one point both ways.
def phi_old(x):
    P = prod_roots([xx for xx in doubles for _ in range(2)] + list(z))
    Q = prod_roots([xx for xx in q for _ in range(5)] + list(s))
    num = mp.mpc(0); den = mp.mpc(0)
    pw = mp.mpc(1)
    for k in range(25):
        num += P[k]*pw
        den += Q[k]*pw
        pw *= x
    return num/(lam*den)


def evalp(P, x):
    acc = mp.mpc(0)
    for cf in reversed(P):
        acc = acc*x + cf
    return acc


x_test = mp.mpc('0.21', '-0.33')
y_test = G(x_test)
kappa = phi_old(x_test)*evalp(Qpoly, y_test)/evalp(Ppoly, y_test)
# phi'(y) = kappa * P'(y)/Q'(y);  check at another point
x2 = mp.mpc('-0.4', '0.15')
err = phi_old(x2) - kappa*evalp(Ppoly, G(x2))/evalp(Qpoly, G(x2))
print("\nphi' consistency at 2nd point:", mp.nstr(abs(err), 3))

# ---- twisted-reality certificate -------------------------------------------
# sigma-stability of root sets => x^n P'(-1/x) * (-1)^n ... check numerically:
# for a deg-n poly with sigma-stable roots: conj(P'_k) ~ ratio * P'_{n-k} * (-1)^k
def twist_check(name, P, n):
    rats = []
    for k in range(n + 1):
        if abs(P[n - k]) > mp.mpf('1e-100'):
            rats.append(mp.conj(P[k])/(P[n - k]*(-1)**k))
    spread = max(abs(r - rats[0]) for r in rats)
    print(f"  {name}: twist ratio = {mp.nstr(rats[0], 20)}  spread = {mp.nstr(spread, 3)}")
    return rats[0]

print("\ntwisted-reality certificates:")
tP = twist_check("P' (deg 22)", Ppoly, 22)
tQ = twist_check("Q' (deg 24)", Qpoly, 24)
tW = twist_check("W' (deg 24)", Wpoly, 24)

# ---- PSLQ probes ------------------------------------------------------------
print("\nPSLQ probes (mp.pslq [1, v] at 200 digits):")
mp.mp.dps = 200
probes_v = {
    'kappa*conj-invariant |kappa|^2': abs(kappa)**2,
    'lam-invariant |G-ratio|': abs(tQ),
    'w1 = G(R1) (real by gauge)': w_new[0].real,
    'sym12: w1 + w2 (real?)': (w_new[0] + w_new[1]).real,
    'P22 real-part probe': (Ppoly[0]*mp.conj(Ppoly[22])).real,
}
for k, v in probes_v.items():
    try:
        rel = mp.pslq([mp.mpf(1), mp.mpf(v)], maxcoeff=10**30, maxsteps=20000)
    except Exception as e:
        rel = None
    print(f"  {k}: {mp.nstr(mp.mpf(v), 30)}  pslq: {rel}")
np.save(BC.SW + "pB_gauge_data.npy",
        np.array([complex(kappa), complex(w_new[0]), complex(w_new[1])], dtype=complex))
with open(BC.SW + "pB_exact_gauge.txt", "w") as f:
    for nm, poly in [("P", Ppoly), ("Q", Qpoly), ("W", Wpoly)]:
        for k, cf in enumerate(poly):
            f.write(f"{nm}[{k}] {mp.nstr(cf.real, 220)} {mp.nstr(cf.imag, 220)}\n")
    f.write(f"kappa {mp.nstr(kappa.real, 220)} {mp.nstr(kappa.imag, 220)}\n")
print("saved pB_exact_gauge.txt")
