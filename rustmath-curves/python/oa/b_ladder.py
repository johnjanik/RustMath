"""B: the precision ladder — real-arithmetic mp Newton on the square structured system.

System (all real, xi-gauge, W = (xi^2+1)^12): P = A^2 Bs, Q = R^5 S, F_k = (P - Q - c W)_k
for k = 0..24, plus the rotation-gauge row a7 - a7* = 0.  26 real unknowns.  Analytic
Jacobian, row equilibration, plain Newton with line search; dps doubles per rung.
Quadratic convergence at each rung is the rigidity certificate.

Usage: from b_ladder import ladder;  theta_mp = ladder(theta_fp64, rungs=(60, 120, 240))
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
from math import comb

W25_INT = [comb(12, k//2) if k % 2 == 0 else 0 for k in range(25)]


def conv(a, b):
    out = [mp.mpf(0)]*(len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai*bj
    return out


def build(th):
    a = list(th[0:8]) + [mp.mpf(1)]
    bs = list(th[8:16]) + [mp.mpf(1)]
    r = list(th[16:20]) + [mp.mpf(1)]
    s = list(th[20:25])
    c = th[25]
    A2 = conv(a, a)
    P = conv(A2, bs)
    R2 = conv(r, r); R4 = conv(R2, R2); R5 = conv(R4, r)
    Q = conv(R5, s)
    return a, bs, r, s, c, A2, R4, R5, P, Q


def residual(th, a7g):
    a, bs, r, s, c, A2, R4, R5, P, Q = build(th)
    F = [P[k] - Q[k] - c*W25_INT[k] for k in range(25)]
    F.append(th[7] - a7g)
    return F, P, Q


def jacobian(th):
    a, bs, r, s, c, A2, R4, R5, P, Q = build(th)
    J = mp.matrix(26, 26)
    tw = conv(a, bs)
    for j in range(8):
        for i in range(len(tw)):
            if j + i < 25:
                J[j + i, j] += 2*tw[i]
    for j in range(8):
        for i in range(len(A2)):
            if j + i < 25:
                J[j + i, 8 + j] += A2[i]
    fR4S = conv(R4, s)
    for j in range(4):
        for i in range(len(fR4S)):
            if j + i < 25:
                J[j + i, 16 + j] -= 5*fR4S[i]
    for j in range(5):
        for i in range(len(R5)):
            if j + i < 25:
                J[j + i, 20 + j] -= R5[i]
    for k in range(25):
        J[k, 25] = -mp.mpf(W25_INT[k])
    J[25, 7] = mp.mpf(1)
    return J


def newton_rung(th, a7g, itmax=30):
    for it in range(itmax):
        F, P, Q = residual(th, a7g)
        rs = [1/(1 + abs(P[k]) + abs(Q[k])) for k in range(25)] + [mp.mpf(1)]
        nrm = mp.sqrt(sum((F[i]*rs[i])**2 for i in range(26)))
        J = jacobian(th)
        Je = mp.matrix(26, 26)
        Fe = mp.matrix(26, 1)
        for i in range(26):
            Fe[i] = F[i]*rs[i]
            for j in range(26):
                Je[i, j] = J[i, j]*rs[i]
        dx = mp.lu_solve(Je, Fe)
        t = mp.mpf(1)
        ok = False
        for _ in range(40):
            thn = [th[i] - t*dx[i] for i in range(26)]
            Fn, Pn, Qn = residual(thn, a7g)
            rsn = [1/(1 + abs(Pn[k]) + abs(Qn[k])) for k in range(25)] + [mp.mpf(1)]
            nn = mp.sqrt(sum((Fn[i]*rsn[i])**2 for i in range(26)))
            if nn < nrm:
                th = thn
                ok = True
                break
            t /= 2
        print(f"    it {it:2d}: |F| = {mp.nstr(nrm, 4)}  t = {mp.nstr(t, 3)}", flush=True)
        if not ok:
            break
        if nrm < mp.mpf(10)**(-(mp.mp.dps - 8)):
            break
    return th, nrm


def ladder(th0, a7g=None, rungs=(60, 120, 240)):
    th = [mp.mpf(float(x)) for x in th0]
    if a7g is None:
        a7g = th[7]
    for dps in rungs:
        mp.mp.dps = dps
        th = [+x for x in th]
        a7g = +a7g
        print(f"  === rung dps={dps} ===", flush=True)
        th, nrm = newton_rung(th, a7g)
        print(f"  rung {dps}: final |F| = {mp.nstr(nrm, 4)}", flush=True)
    return th
