"""Seed for the canonical-antipodal ladder: transform ladder-2's configuration through the
(1e-14-noisy) gauge G of b_exact.py and select pair representatives."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
import b_config as BC


def seed():
    dps0 = mp.mp.dps
    mp.mp.dps = max(dps0, 60)
    TH = []
    with open(BC.SW + "pB_ladder_theta.txt") as f:
        for line in f:
            a, b = line.split()
            TH.append(mp.mpc(a, b))
    d = TH[0:7]; z = TH[7:15]; q = TH[15:19]; s = TH[19:23]
    R1 = mp.mpc('-0.84076601151413', '0.38995761807378')
    R2 = mp.mpc('-0.35478102590543', '-0.02962349269773')

    def mobius_from_3(zs, ws):
        def to01inf(p):
            z0, z1, z2 = p
            return mp.matrix([[z1 - z2, -z0*(z1 - z2)], [z1 - z0, -z2*(z1 - z0)]])
        C = to01inf(ws)**-1*to01inf(zs)
        return (C[0, 1], C[0, 0], C[1, 1], C[1, 0])

    SGM = mobius_from_3([mp.conj(mp.mpc(0)), mp.conj(R1), mp.conj(q[0])],
                        [d[0], R2, q[2]])

    def sig(x):
        al, be, ga, de = SGM
        xc = mp.conj(x)
        return (al + be*xc)/(ga + de*xc)

    g0 = d[0]

    def G1(x):
        return g0*x/(g0 - x)

    def G1inv(y):
        return g0*y/(g0 + y)

    rho = G1(sig(G1inv(mp.mpc(1))))
    sc_ = mp.sqrt(-rho.real)

    def G2(x):
        return G1(x)/sc_

    r1pp = G2(R1)
    phase = r1pp/abs(r1pp)

    def G(x):
        return G2(x)/phase

    def rep(m):
        """pick the pair representative inside the unit disk (|m| < 1) for stability"""
        return m if abs(m) <= 1 else -1/mp.conj(m)

    # doubles: skip d[0] (-> infinity); three pairs from d[1..6]: pick one per sigma-pair
    dG = [G(x) for x in d[1:]]
    used = [False]*6
    md = []
    for i in range(6):
        if used[i]:
            continue
        si = -1/mp.conj(dG[i])
        j = min((k for k in range(6) if not used[k] and k != i),
                key=lambda k: abs(dG[k] - si))
        used[i] = used[j] = True
        md.append(rep(dG[i]))
    zG = [G(x) for x in z]
    used = [False]*8
    mz = []
    for i in range(8):
        if used[i]:
            continue
        si = -1/mp.conj(zG[i])
        j = min((k for k in range(8) if not used[k] and k != i),
                key=lambda k: abs(zG[k] - si))
        used[i] = used[j] = True
        mz.append(rep(zG[i]))
    qG = [G(x) for x in q]
    mq = [rep(qG[0]), rep(qG[1])]          # (q0,q2), (q1,q3) are sigma-pairs
    sG = [G(x) for x in s]
    used = [False]*4
    ms = []
    for i in range(4):
        if used[i]:
            continue
        si = -1/mp.conj(sG[i])
        j = min((k for k in range(4) if not used[k] and k != i),
                key=lambda k: abs(sG[k] - si))
        used[i] = used[j] = True
        ms.append(rep(sG[i]))
    r = G(R1).real
    # mu seed: from the leading coefficients -- fit mu so that k=24 row closes:
    # P24' = 0; -mu*prod + mu*W24 -> mu free; take mu from matching k=23 approx: seed 1
    th0 = []
    for m in md + mz + mq + ms:
        th0 += [m.real, m.imag]
    th0 += [r, mp.mpf(1), mp.mpf(0)]
    # refine mu linearly: F(th) rows are linear in mu given the rest: solve 1-var lstsq
    import b_ladder3 as L3
    P, Q, W, mu = L3.build(th0)
    P = P + [mp.mpc(0)]*(25 - len(P))
    num = mp.mpc(0); den = mp.mpf(0)
    for k in range(25):
        qk = Q[k]/mu
        w = 1/(1 + abs(P[k]) + abs(Q[k]))
        a = (qk - W[k])*w
        b = P[k]*w
        num += mp.conj(a)*b
        den += abs(a)**2
    mu_fit = num/den
    th0[-2], th0[-1] = mu_fit.real, mu_fit.imag
    mp.mp.dps = dps0
    return th0
