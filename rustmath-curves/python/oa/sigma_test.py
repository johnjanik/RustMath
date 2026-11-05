"""The mirror involution sigma = M o conj on the X_a coordinate.

The dessin's positions show mirror structure (z -> -conj z): the a-center (X=0), both
12-points (imaginary axis) are mirror-FIXED positions; c9 sits at the mirror position of c.
If the map is defined over R (a fortiori over Q), there is an antiholomorphic involution
sigma(x) = M(conj x) of the X-sphere permuting every special-point class.  Determine M from
dd-known data, validate on independent addresses, then PREDICT the unmeasured structure
(4th quintuple pole = sigma(c11), double/simple pairings).
"""
import numpy as np, mpmath as mp
mp.mp.dps = 30

r1  = mp.mpc('0.47182600647013', '-0.10561240463346')
r2  = mp.mpc('-0.32224249915043', '-0.38542142936930')
Xc  = mp.mpc('-0.056491567434', '-0.311094205741')
Xc9 = mp.mpc('1.152891104470', '0.579237023791')
Xc11r = mp.mpc('-0.4944888699', '-0.4069986879')     # ring quality ~1e-3
Xc11g = mp.mpc('-0.4998', '-0.4032')                 # geodesic glue 9e-4
a1 = mp.mpc('0.2120091233', '-0.5634590964')         # dd double
a2 = mp.mpc('-0.7190508461', '-0.4168884505')        # 4e-7 double
s1 = mp.mpc('0.4775220321', '0.0580930745')          # soft doubles
s2 = mp.mpc('-0.1303535948', '-0.4094436433')
s3 = mp.mpc('0.5758909398', '-0.1287776533')
# crowd (fit quality only)
A6 = mp.mpc('0.611', '-0.085'); A7 = mp.mpc('0.585', '-0.120'); A8 = mp.mpc('0.544', '-0.139')
R4 = mp.mpc('0.592', '-0.079')                       # WS=30 blind 4th pole estimate


def fit_M_fix0(p_pairs):
    """M(x) = a x / (1 + b x) with M(0)=0; p_pairs = [(u, v)] meaning M(u) = v.
    Two pairs determine (a, b): a u = v (1 + b u)  =>  linear in (a, b)."""
    (u1, v1), (u2, v2) = p_pairs
    # a u1 - b u1 v1 = v1 ; a u2 - b u2 v2 = v2
    Amat = mp.matrix([[u1, -u1 * v1], [u2, -u2 * v2]])
    rhs = mp.matrix([v1, v2])
    sol = mp.lu_solve(Amat, rhs)
    return sol[0], sol[1]


def M_apply(ab, x):
    a, b = ab
    return a * x / (1 + b * x)


def sigma(ab, x):
    return M_apply(ab, mp.conj(x))


def report(name, ab):
    print(f"--- {name} ---")
    # involution check: sigma(sigma(x)) = x on test points
    for lbl, x in [("Xc", Xc), ("a1", a1), ("r1", r1)]:
        e = abs(sigma(ab, sigma(ab, x)) - x)
        print(f"  invol |s(s({lbl}))-{lbl}| = {mp.nstr(e, 3)}")
    print(f"  sigma(Xc)   = {mp.nstr(sigma(ab, Xc), 12)}   vs Xc9 = {mp.nstr(Xc9, 12)}"
          f"   diff = {mp.nstr(abs(sigma(ab, Xc) - Xc9), 3)}")
    print(f"  sigma(Xc9)  = {mp.nstr(sigma(ab, Xc9), 12)}  vs Xc  = {mp.nstr(Xc, 12)}"
          f"   diff = {mp.nstr(abs(sigma(ab, Xc9) - Xc), 3)}")
    print(f"  PREDICT p4 = sigma(Xc11_ring) = {mp.nstr(sigma(ab, Xc11r), 10)}")
    print(f"  PREDICT p4 = sigma(Xc11_geo)  = {mp.nstr(sigma(ab, Xc11g), 10)}")
    print(f"          (WS=30 blind R4 = {mp.nstr(R4, 6)})")
    # doubles: sigma should permute {0, a1, a2, s1, s2, s3, A6, A7, A8}
    doubles = [("0", mp.mpc(0)), ("a1", a1), ("a2", a2), ("s1", s1), ("s2", s2), ("s3", s3),
               ("A6", A6), ("A7", A7), ("A8", A8)]
    print("  double-zero pairing under sigma:")
    for lbl, x in doubles:
        sx = sigma(ab, x)
        best = min(doubles, key=lambda t: abs(t[1] - sx))
        print(f"    sigma({lbl:3s}) = {mp.nstr(sx, 8)} -> nearest {best[0]:3s} "
              f"(dist {mp.nstr(abs(best[1] - sx), 2)})")


# H1: sigma fixes 0, r1, r2
ab1 = fit_M_fix0([(mp.conj(r1), r1), (mp.conj(r2), r2)])
report("H1: fix 0, fix r1, fix r2", ab1)

# H2: sigma fixes 0, swaps r1 <-> r2
ab2 = fit_M_fix0([(mp.conj(r1), r2), (mp.conj(r2), r1)])
report("H2: fix 0, swap r1 <-> r2", ab2)

# H3: determined instead by (c <-> c9 swap) + fix 0 -- then VALIDATE on r1, r2
ab3 = fit_M_fix0([(mp.conj(Xc), Xc9), (mp.conj(Xc9), Xc)])
print("--- H3: fix 0, swap c <-> c9 (validate on 12-points) ---")
print(f"  sigma(r1) = {mp.nstr(sigma(ab3, r1), 12)} vs r1 = {mp.nstr(r1, 12)} "
      f"diff {mp.nstr(abs(sigma(ab3, r1) - r1), 3)}  vs r2 diff {mp.nstr(abs(sigma(ab3, r1) - r2), 3)}")
print(f"  sigma(r2) = {mp.nstr(sigma(ab3, r2), 12)} vs r2 = {mp.nstr(r2, 12)} "
      f"diff {mp.nstr(abs(sigma(ab3, r2) - r2), 3)}  vs r1 diff {mp.nstr(abs(sigma(ab3, r2) - r1), 3)}")
report("H3 full", ab3)
