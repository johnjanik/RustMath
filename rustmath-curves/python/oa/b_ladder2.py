"""B: the precision ladder on the X-gauge structured system, pins-as-gauge.

Gauge (fixed exactly, DEFINES the coordinate): one double zero at 0, the two 12-points at
their dd addresses r1, r2 (W = (X-r1)^12 (X-r2)^12 fixed).  Unknowns (25 complex):
  d[0..6]  the other 7 double zeros
  z[0..7]  the 8 simple zeros
  q[0..3]  the 4 quintuple poles
  s[0..3]  the 4 simple poles
  lam, c
Equations (25 complex): coefficients k=0..24 of  P - Q - c W = 0,
  P = prod(X-0)^2 (X-d_i)^2 * prod(X-z_j)   [monic deg 24]
  Q = lam * prod(X-q_k)^5 * prod(X-s_l)     [deg 24, lead lam]
Holomorphic square system; analytic Jacobian; mp Newton, dps doubling per rung.
Quadratic convergence at every rung = the rigidity certificate.
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp

R1 = mp.mpc('-0.84076601151413', '0.38995761807378')
R2 = mp.mpc('-0.35478102590543', '-0.02962349269773')


def conv(a, b):
    out = [mp.mpc(0)]*(len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai*bj
    return out


def prod_roots(rs):
    p = [mp.mpc(1)]
    for r in rs:
        p = conv(p, [-r, mp.mpc(1)])
    return p


def build(th):
    d = th[0:7]; z = th[7:15]; q = th[15:19]; s = th[19:23]
    lam, c = th[23], th[24]
    A = prod_roots([mp.mpc(0)] + list(d))          # deg 8
    P = conv(conv(A, A), prod_roots(z))            # deg 24 monic
    Rq = prod_roots(q)
    R5 = conv(conv(conv(Rq, Rq), conv(Rq, Rq)), Rq)
    Q = conv(R5, [x*lam for x in prod_roots(s)])   # deg 24, lead lam
    return A, P, Rq, R5, Q, lam, c


def residual(th, W):
    A, P, Rq, R5, Q, lam, c = build(th)
    return [P[k] - Q[k] - c*W[k] for k in range(25)], P, Q


def jacobian(th, W):
    A, P, Rq, R5, Q, lam, c = build(th)
    d = th[0:7]; z = th[7:15]; q = th[15:19]; s = th[19:23]
    lam, c = th[23], th[24]
    Zp = prod_roots(z)
    Sp = prod_roots(s)
    J = mp.matrix(25, 25)
    droots = [mp.mpc(0)] + list(d)
    for i in range(7):                              # d/dd_i: -2 * Ahat_i * A * Z
        ah = prod_roots([droots[j] for j in range(8) if j != i + 1])
        dd = conv(conv(ah, A), Zp)
        for k in range(min(len(dd), 25)):
            J[k, i] += -2*dd[k]
    for j in range(8):                              # d/dz_j: -A^2 * Zhat_j
        zh = prod_roots([z[m] for m in range(8) if m != j])
        dd = conv(conv(A, A), zh)
        for k in range(min(len(dd), 25)):
            J[k, 7 + j] += -dd[k]
    for i in range(4):                              # d/dq_i: +5 lam * Rhat_i R^4 S
        rh = prod_roots([q[m] for m in range(4) if m != i])
        dd = conv(conv(rh, conv(conv(Rq, Rq), conv(Rq, Rq))), Sp)
        for k in range(min(len(dd), 25)):
            J[k, 15 + i] += 5*lam*dd[k]
    for l in range(4):                              # d/ds_l: +lam R^5 Shat_l
        sh = prod_roots([s[m] for m in range(4) if m != l])
        dd = conv(R5, sh)
        for k in range(min(len(dd), 25)):
            J[k, 19 + l] += lam*dd[k]
    Qs = conv(R5, Sp)                               # d/dlam: -R^5 S
    for k in range(25):
        J[k, 23] = -Qs[k]
        J[k, 24] = -W[k]
    return J


def newton_rung(th, W, itmax=40):
    nrm_prev = None
    for it in range(itmax):
        F, P, Q = residual(th, W)
        rs = [1/(1 + abs(P[k]) + abs(Q[k])) for k in range(25)]
        nrm = mp.sqrt(sum((abs(F[k])*rs[k])**2 for k in range(25)))
        J = jacobian(th, W)
        Je = mp.matrix(25, 25)
        Fe = mp.matrix(25, 1)
        for i in range(25):
            Fe[i] = F[i]*rs[i]
            for j2 in range(25):
                Je[i, j2] = J[i, j2]*rs[i]
        dx = mp.lu_solve(Je, Fe)
        t = mp.mpf(1)
        ok = False
        for _ in range(40):
            thn = [th[i] - t*dx[i] for i in range(25)]
            Fn, Pn, Qn = residual(thn, W)
            rsn = [1/(1 + abs(Pn[k]) + abs(Qn[k])) for k in range(25)]
            nn = mp.sqrt(sum((abs(Fn[k])*rsn[k])**2 for k in range(25)))
            if nn < nrm:
                th = thn
                ok = True
                break
            t /= 2
        print(f"    it {it:2d}: |F| = {mp.nstr(nrm, 4)}  t={mp.nstr(t, 3)}", flush=True)
        if not ok or nrm < mp.mpf(10)**(-(mp.mp.dps - 6)):
            break
    return th, nrm


def ladder(th0, rungs=(60, 120, 240)):
    th = [mp.mpc(x) for x in th0]
    for dps in rungs:
        mp.mp.dps = dps
        th = [+x for x in th]
        W = conv(prod_roots([R1]*12), prod_roots([R2]*12))
        print(f"=== rung dps={dps} ===", flush=True)
        t0 = time.time()
        th, nrm = newton_rung(th, W)
        print(f"  rung done: |F| = {mp.nstr(nrm, 4)}  [{time.time()-t0:.0f}s]", flush=True)
    return th


if __name__ == "__main__":
    import b_config as BC
    V = np.load(BC.SW + "pB_varpro.npz")
    SIG = np.load(BC.SW + "sigmaB.npy")

    def sigf(x):
        al, be, ga, de = SIG
        return (al + be*np.conj(x))/(ga + de*np.conj(x))

    d = list(V['d'])
    SIG0 = complex(-0.60511372176989, +0.07804715463807)
    doubles = [SIG0] + [v for m in d for v in (m, sigf(m))]           # 7 (origin excluded)
    zroots = list(np.roots(V['Z'][::-1]))                              # 8
    P5 = [complex(-0.24339094187298, -0.19286184625628),
          complex(-0.23789929378150, +0.23252559046980),
          complex(-0.59198701480729, +0.21733468385806),
          complex(-0.64997276203624, -0.09319689864632)]
    T = V['T']
    sroots = list(np.roots(T[::-1]))                                   # 4
    lam = complex(T[4])
    c = complex(V['c'])
    th0 = doubles + zroots + P5 + sroots + [lam, c]
    assert len(th0) == 25
    th = ladder(th0, rungs=(60, 120, 240))
    # save full precision
    mp.mp.dps = 240
    with open(BC.SW + "pB_ladder_theta.txt", "w") as f:
        for x in th:
            f.write(mp.nstr(x.real, 220) + " " + mp.nstr(x.imag, 220) + "\n")
    print("saved pB_ladder_theta.txt")
