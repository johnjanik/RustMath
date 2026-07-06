"""B ladder in the CANONICAL ANTIPODAL GAUGE: sigma = -1/conj(x) exact by construction.

Gauge: origin double-pair at (0, infinity); every other class = sigma-pairs (m, -1/conj m);
the 12-pair = (r, -1/r), r real > 0.  All sigma-symmetry is EXACT in the parameterization,
so the converged configuration's K0-invariants (e = r - 1/r, coefficient combinations) are
noise-free algebraic numbers -> PSLQ.

Structure:  P = x^2 * prod_{i<3} B(m_d[i])^2 * prod_{j<4} B(m_z[j])      (deg 22, monic)
            Q = mu * prod_{k<2} B(m_q[k])^5 * prod_{l<2} B(m_s[l])       (deg 24, lead mu)
            W = (x^2 - e x - 1)^12,   e = r - 1/r
            F = P - Q + mu*W  ... coefficient bookkeeping:  P - Q = c W with c = -mu
            (k=24: 0 - mu = c;  the identity uses c = -mu exactly.)
where B(m) = (x - m)(x + 1/conj(m)) = x^2 - (m - 1/conj m) x - m/conj(m).

Unknowns (real, 23): Re/Im of m_d[3], m_z[4], m_q[2], m_s[2] (22) + r (1) + Re/Im mu (2)
 -> 25 real. Equations: 25 complex rows BUT twisted-reality makes only ~25 real independent;
we solve least-squares GN on the 50 real components (rank 25) -- at the true point all
vanish, GN is quadratic.
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
import b_config as BC


def conv(a, b):
    out = [mp.mpc(0)]*(len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai*bj
    return out


def Bq(m):
    """(x - m)(x + 1/conj m): [c0, c1, c2] ascending."""
    mc = mp.conj(m)
    return [-m/mc, -(m - 1/mc), mp.mpc(1)]


def build(th):
    md = [mp.mpc(th[0], th[1]), mp.mpc(th[2], th[3]), mp.mpc(th[4], th[5])]
    mz = [mp.mpc(th[6], th[7]), mp.mpc(th[8], th[9]), mp.mpc(th[10], th[11]),
          mp.mpc(th[12], th[13])]
    mq = [mp.mpc(th[14], th[15]), mp.mpc(th[16], th[17])]
    ms = [mp.mpc(th[18], th[19]), mp.mpc(th[20], th[21])]
    r = th[22]
    mu = mp.mpc(th[23], th[24])
    P = [mp.mpc(0), mp.mpc(0), mp.mpc(1)]          # x^2
    for m in md:
        b = Bq(m)
        P = conv(P, conv(b, b))
    for m in mz:
        P = conv(P, Bq(m))                          # deg 22
    Q = [mp.mpc(1)]
    for m in mq:
        b = Bq(m)
        b2 = conv(b, b)
        Q = conv(Q, conv(conv(b2, b2), b))
    for m in ms:
        Q = conv(Q, Bq(m))
    Q = [mu*x for x in Q]                           # deg 24, lead mu
    e = r - 1/r
    w2 = [mp.mpc(-1), mp.mpc(-e), mp.mpc(1)]
    W = [mp.mpc(1)]
    for _ in range(12):
        W = conv(W, w2)                             # deg 24 monic
    return P, Q, W, mu


def residual(th):
    P, Q, W, mu = build(th)
    P = P + [mp.mpc(0)]*(25 - len(P))
    F = [P[k] - Q[k] + mu*W[k] for k in range(25)]
    rs = [1/(1 + abs(P[k]) + abs(Q[k])) for k in range(25)]
    out = []
    for k in range(25):
        v = F[k]*rs[k]
        out.append(v.real)
        out.append(v.imag)
    return out


def gn(th, itmax=40, verbose=True):
    n = len(th)
    hstep = mp.mpf(10)**(-(mp.mp.dps//2))
    for it in range(itmax):
        r0 = residual(th)
        nrm = mp.sqrt(sum(x*x for x in r0))
        J = mp.matrix(len(r0), n)
        for k in range(n):
            tp = list(th)
            tp[k] = tp[k] + hstep
            rp = residual(tp)
            for i in range(len(r0)):
                J[i, k] = (rp[i] - r0[i])/hstep
        JT = J.T
        A = JT*J
        g = JT*mp.matrix(r0)
        try:
            dx = mp.lu_solve(A, g)
        except Exception:
            for i in range(n):
                A[i, i] = A[i, i]*(1 + mp.mpf('1e-12')) + mp.mpf('1e-30')
            dx = mp.lu_solve(A, g)
        t = mp.mpf(1)
        ok = False
        for _ in range(50):
            tn = [th[i] - t*dx[i] for i in range(n)]
            rn_ = residual(tn)
            nn = mp.sqrt(sum(x*x for x in rn_))
            if nn < nrm:
                th = tn
                ok = True
                break
            t /= 2
        if verbose:
            print(f"    it {it:2d}: |F| = {mp.nstr(nrm, 4)}  t={mp.nstr(t, 3)}", flush=True)
        if not ok or nrm < mp.mpf(10)**(-(mp.mp.dps - 8)):
            break
    return th, nrm


def ladder(th0, rungs=(60, 120, 260)):
    th = [mp.mpf(x) if not isinstance(x, mp.mpf) else x for x in th0]
    for dps in rungs:
        mp.mp.dps = dps
        th = [+x for x in th]
        print(f"=== rung dps={dps} ===", flush=True)
        t0 = time.time()
        th, nrm = gn(th)
        print(f"  rung done: |F| = {mp.nstr(nrm, 4)}  [{time.time()-t0:.0f}s]", flush=True)
    return th


if __name__ == "__main__":
    mp.mp.dps = 60
    # seed: transform the ladder-2 configuration through the (noisy) canonical gauge G
    # values computed by b_exact.py at ~1e-14: read the transformed roots fresh here.
    import b_exact_seed as ES     # produced below if absent
    th0 = ES.seed()
    th = ladder(th0)
    mp.mp.dps = 260
    with open(BC.SW + "pB_ladder3_theta.txt", "w") as f:
        for x in th:
            f.write(mp.nstr(mp.mpf(x), 240) + "\n")
    print("saved pB_ladder3_theta.txt")
    # first PSLQ: e = r - 1/r
    r = th[22]
    e = r - 1/r
    print("\nr =", mp.nstr(r, 40))
    print("e = r - 1/r =", mp.nstr(e, 40))
    for name, v in [("e", e), ("e^2", e*e)]:
        rel = mp.pslq([mp.mpf(1), v], maxcoeff=10**24, maxsteps=50000)
        print(f"PSLQ [1, {name}]: {rel}")
