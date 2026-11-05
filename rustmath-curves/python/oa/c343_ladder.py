"""(3,4,3) precision ladder: square holomorphic Newton on A^3 - lam R^3 S - c W^4 = 0.

Gauge (fixed exactly): one A-root at 0, two W-roots at their measured values.
Unknowns (25 complex): a[7 free A-roots], r[6], s[6], w[4 free W-roots], lam, c.
Equations: coefficients k=0..24. Analytic Jacobian; dps-doubling rungs.
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp

W_G1 = mp.mpc('0.400341547059', '-0.098184224177')     # gauge-pinned W roots
W_G2 = mp.mpc('0.444237652730', '-0.304128490193')


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


def unpack(th):
    return th[0:7], th[7:13], th[13:19], th[19:23], th[23], th[24]


def build(th):
    a, r, s, w, lam, c = unpack(th)
    A = prod_roots([mp.mpc(0)] + list(a))
    P = conv(conv(A, A), A)
    R = prod_roots(r)
    R3 = conv(conv(R, R), R)
    S = prod_roots(s)
    Q = [lam*x for x in conv(R3, S)]
    W = prod_roots([W_G1, W_G2] + list(w))
    W2 = conv(W, W)
    W4 = conv(W2, W2)
    return A, P, R, R3, S, Q, W, W4, lam, c


def residual(th):
    A, P, R, R3, S, Q, W, W4, lam, c = build(th)
    return [P[k] - Q[k] - c*W4[k] for k in range(25)], P, Q


def jacobian(th):
    a, r, s, w, lam, c = unpack(th)
    A, P, R, R3, S, Q, W, W4, lam, c = build(th)
    aroots = [mp.mpc(0)] + list(a)
    J = mp.matrix(25, 25)
    A2 = conv(A, A)
    for i in range(7):                      # d/da_i: -3 Ahat_i A^2
        ah = prod_roots([aroots[j] for j in range(8) if j != i + 1])
        d = conv(ah, A2)
        for k in range(min(len(d), 25)):
            J[k, i] += -3*d[k]
    R2 = conv(R, R)
    for i in range(6):                      # d/dr_i: +3 lam Rhat_i R^2 S
        rh = prod_roots([r[j] for j in range(6) if j != i])
        d = conv(conv(rh, R2), S)
        for k in range(min(len(d), 25)):
            J[k, 7 + i] += 3*lam*d[k]
    for i in range(6):                      # d/ds_i: +lam R^3 Shat_i
        sh = prod_roots([s[j] for j in range(6) if j != i])
        d = conv(R3, sh)
        for k in range(min(len(d), 25)):
            J[k, 13 + i] += lam*d[k]
    W3 = conv(conv(W, W), W)
    wroots = [W_G1, W_G2] + list(w)
    for i in range(4):                      # d/dw_i: +4c What_i W^3
        wh = prod_roots([wroots[j] for j in range(6) if j != i + 2])
        d = conv(wh, W3)
        for k in range(min(len(d), 25)):
            J[k, 19 + i] += 4*c*d[k]
    QS = conv(R3, S)
    for k in range(25):
        J[k, 23] = -QS[k]                   # d/dlam
        J[k, 24] = -W4[k]                   # d/dc
    return J


def newton_rung(th, itmax=40):
    for it in range(itmax):
        F, P, Q = residual(th)
        rs = [1/(1 + abs(P[k]) + abs(Q[k])) for k in range(25)]
        nrm = mp.sqrt(sum((abs(F[k])*rs[k])**2 for k in range(25)))
        J = jacobian(th)
        Je = mp.matrix(25, 25)
        Fe = mp.matrix(25, 1)
        for i in range(25):
            Fe[i] = F[i]*rs[i]
            for j in range(25):
                Je[i, j] = J[i, j]*rs[i]
        try:
            dx = mp.lu_solve(Je, Fe)
        except Exception as e:
            print("    lu fail:", e)
            break
        t = mp.mpf(1)
        ok = False
        for _ in range(40):
            thn = [th[i] - t*dx[i] for i in range(25)]
            Fn, Pn, Qn = residual(thn)
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


def ladder(th0, rungs=(60, 120, 300)):
    th = [mp.mpc(x) for x in th0]
    for dps in rungs:
        mp.mp.dps = dps
        th = [+x for x in th]
        print(f"=== rung dps={dps} ===", flush=True)
        t0 = time.time()
        th, nrm = newton_rung(th)
        print(f"  rung done: |F| = {mp.nstr(nrm, 4)}  [{time.time()-t0:.0f}s]", flush=True)
    return th


if __name__ == "__main__":
    import c343_config as CC
    V = np.load(CC.SW + "pC_varpro2.npz", allow_pickle=True)
    th = V['theta']
    Wfull = list(V['W'])
    Rfull = list(V['R'])
    aF = [complex(th[0], th[1]), complex(th[2], th[3]),
          complex(th[4], th[5]), complex(th[6], th[7])]
    # rebuild the inner solve at the winner to get T (lam S) and c
    def pfr(rr):
        P = np.array([1.0 + 0j])
        for r_ in rr:
            P = np.convolve(P, np.array([-r_, 1.0 + 0j]))
        return P
    D = np.load(CC.SW + "pC_samples1.npz", allow_pickle=True)
    X, PH = D['X'], D['PHI']
    PW = X[:, None]**np.arange(25)
    A = pfr(CC.A_PINS + aF)
    P = np.convolve(np.convolve(A, A), A)
    R = pfr(Rfull)
    R3 = np.convolve(np.convolve(R, R), R)
    W = pfr(Wfull)
    W4 = np.convolve(np.convolve(W, W), np.convolve(W, W))
    F24 = np.load(CC.SW + "pC_deg24_fit.npz")
    pwS = (X/float(F24['S']))[:, None]**np.arange(25)
    W0 = 1.0/np.maximum(np.abs(pwS@F24['p']) + np.abs(PH*(pwS@F24['q'])), 1e-300)
    W0 = W0/np.median(W0)
    M = np.zeros((25 + len(X), 8), complex)
    b = np.zeros(25 + len(X), complex)
    sP = 1.0/(1 + np.abs(P[:25]))
    for k in range(25):
        for l in range(7):
            if 0 <= k - l <= 18:
                M[k, l] = R3[k - l]*sP[k]
        M[k, 7] = W4[k]*sP[k]
        b[k] = P[k]*sP[k]
    R3v = PW[:, :19]@R3
    for l in range(7):
        M[25:, l] = PH*R3v*PW[:, l]*W0
    b[25:] = (PW@P[:25])*W0
    v, *_ = np.linalg.lstsq(M, b, rcond=None)
    T = v[:7]
    c0 = v[7]
    lam0 = T[6]
    sroots = np.roots((T/T[6])[::-1])
    # assemble theta for the ladder: [a7 free A..., r6, s6, w4 free W..., lam, c]
    th0 = (list(CC.A_PINS[1:]) + aF + list(Rfull) + list(sroots)
           + [x for x in Wfull if abs(x - complex(W_G1)) > 1e-9 and abs(x - complex(W_G2)) > 1e-9]
           + [complex(lam0), complex(c0)])
    assert len(th0) == 25, len(th0)
    th = ladder(th0)
    mp.mp.dps = 300
    with open(CC.SW + "pC_ladder_theta.txt", "w") as f:
        for x in th:
            f.write(mp.nstr(x.real, 280) + " " + mp.nstr(x.imag, 280) + "\n")
    print("saved pC_ladder_theta.txt")
