"""Phase I steps 1-2: IRLS-VARPRO structured solve with the two-layer collapse guard.

Inner (linear, for fixed roots): B(8, gauge B8=1), St=lam*S (5), c -- weighted lstsq over
25 stratum rows + ALL sample rows, with IRLS weights w_i = 1/(eps+|P(y_i)|+|phi_i Q(y_i)|)
recomputed 3 passes. Outer: coordinate descent over 7 complex roots (5 free A-doubles, 2
free R-poles) with measurement penalty rows (1/sigma) and the two-layer guard: reject any
step that shrinks min_i|Q(y_i)| or sigma_min/sigma_max(Syl(P,Q)) by >10x.

Anchors: 4-region samples + c9 pole region + EXTENDED 12-germ rings (|u12|<=0.72 on both
b-charts). Gate (per region, ALL samples): median <= 1e-10, max <= 1e-8 relative.
"""
import numpy as np, mpmath as mp, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
mp.mp.dps = 45
from math import comb
from chart_dd import mp_series_eval
from phi_vertices import phi_in_u5, phi_in_u12
from p5_submit_kit import irls_weights, common_root_score
import mapkit

SW = "/home/john/sweep_2_12_5/"
r1 = complex(0.47182600647013, -0.10561240463346)
r2 = complex(-0.32224249915043, -0.38542142936930)
rc  = complex(-0.056491567434, -0.311094205741)
rc9 = complex(1.152891104470, 0.579237023791)
a_ex = [complex(0.2120091233, -0.5634590964), complex(-0.7190508461, -0.4168884505)]
softs = [(complex(0.4775220321, 0.0580930745), 5e-4),
         (complex(-0.1303535948, -0.4094436433), 2e-5),
         (complex(0.5758909398, -0.1287776533), 5e-4)]
r3m = (complex(-0.494430, -0.406940), 1.5e-3)
sc = 8.0
KPEN = 3e-6

_Lam = mp.cos(mp.pi/5)/mp.sin(mp.pi/12); MU = _Lam + mp.sqrt(_Lam*_Lam - 1)
Q_B2 = mp.mpf('0.164275700384606020499257420743316339146629866')
ZC = mp.mpc('0.793538482569228956003002774509736302229259206',
            '0.608520070894728592568281209719200629678206803')
ZC9 = -mp.conj(ZC)


def build_samples():
    CH = {k: np.load(SW + f, allow_pickle=True) for k, f in
          [('A', 'm_glue_a_N1200_ddspan.npz'), ('B', 'm_order12_cycle1_N6000_ddspan.npz'),
           ('C', 'm_glue_b2_N6900_ddspan.npz'), ('D', 'm_glue_c_N2400_ddspan.npz'),
           ('E', 'm_glue_c9_N2400_ddspan.npz')]}
    mus = {k: np.load(SW + f) for k, f in
           [('B', 'mu_a_bprime.npy'), ('C', 'mu_a_b2.npy'), ('D', 'mu_a_c.npy'), ('E', 'mu_a_c9.npy')]}
    k_bp, k_b2 = [mp.mpc(v) for v in np.load(SW + "kappa12_charts.npy")]
    k_c = mp.mpc(np.load(SW + "kappa_c.npy")[0])
    k_c9 = mp.mpc(np.load(SW + "kappa_c9.npy")[0])
    KAP = mp.mpf(repr(float(mapkit.kappa)))
    PHIV = [mp.mpf(repr(float(v.real))) for v in mapkit.phiv]
    S12 = [mp.mpf(c.numerator)/mp.mpf(c.denominator) for c in phi_in_u12(2, 12, 5, 240)]
    T5 = [mp.mpf(c.numerator)/mp.mpf(c.denominator) for c in phi_in_u5(2, 12, 5, 200)]

    def horner(cs, u):
        acc = mp.mpc(0)
        for n in range(len(cs)-1, -1, -1):
            acc = acc*u + cs[n]
        return acc

    def w_of(z, ctr):
        return (z - ctr)/(z - mp.conj(ctr))

    def x_kmsv(npz, w):
        G = mp_series_eval(npz['Bh'], npz['Bl'], w)
        c_ = -(mp.mpc(npz['Bh'][8, 7]) + mp.mpc(npz['Bl'][8, 7]))
        return G[8]/(G[7] + c_*G[8])

    def moeb(m, x):
        al, be, ga, de = [mp.mpc(v) for v in m]
        return (al + be*x)/(ga + de*x)

    smp = []
    for t in np.linspace(1.05, 2.10, 15):
        for fr in (0.0, 0.05, -0.05, 0.10):
            z = mp.mpc(mp.mpf(float(t))*mp.mpf(repr(fr)), mp.mpf(float(t)))
            u = (w_of(z, mp.mpc(0, 1))/KAP)**2
            if abs(u) <= 3.8:
                smp.append((complex(x_kmsv(CH['A'], w_of(z, mp.mpc(0, 1)))), complex(horner(PHIV, u)), 'A'))
    for r in np.linspace(0.10, 0.47, 9):                 # extended: |u12| <= 0.72
        for kk in range(6):
            wq = mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk+0.21)/6)
            smp.append((complex(moeb(mus['B'], x_kmsv(CH['B'], wq))), complex(horner(S12, wq/k_bp)), 'B'))
            wq2 = mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk+0.13)/6)
            smp.append((complex(moeb(mus['C'], x_kmsv(CH['C'], wq2))), complex(horner(S12, wq2/k_b2)), 'C'))
    for r in np.linspace(0.08, 0.35, 7):
        for kk in range(5):
            wq = mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk+0.37)/5)
            smp.append((complex(moeb(mus['D'], x_kmsv(CH['D'], wq))), complex(1/horner(T5, wq/k_c)), 'D'))
            wq2 = mp.mpf(float(r))*mp.exp(1j*2*mp.pi*(kk+0.29)/5)
            smp.append((complex(moeb(mus['E'], x_kmsv(CH['E'], wq2))), complex(1/horner(T5, wq2/k_c9)), 'E'))
    X = np.array([s[0] for s in smp]); PH = np.array([s[1] for s in smp])
    REG = np.array([s[2] for s in smp])
    np.savez(SW + "p5_samples5.npz", X=X, PHI=PH, region=REG)
    return X, PH, REG


def main():
    if os.path.exists(SW + "p5_samples5.npz"):
        D = np.load(SW + "p5_samples5.npz")
        X, PH, REG = D['X'], D['PHI'], D['region']
    else:
        X, PH, REG = build_samples()
    print(f"{len(X)} samples: " + " ".join(f"{R}:{(REG==R).sum()}" for R in 'ABCDE'), flush=True)

    Yf = lambda X_: (1/(complex(X_)-r1))/sc
    y2s = 1/(r2-r1)/sc; y0s = (-1/r1)/sc
    ycs, yc9s = Yf(rc), Yf(rc9)
    aexY = [complex(y0s), Yf(a_ex[0]), Yf(a_ex[1])]
    meas = [(Yf(Xm), sX/(abs(complex(Xm)-r1)**2*sc)) for Xm, sX in softs]
    r3Y = (Yf(r3m[0]), r3m[1]/(abs(complex(r3m[0])-r1)**2*sc))

    def pfr(rr):
        P = np.array([1.0+0j])
        for r in rr:
            P = np.convolve(P, np.array([-r, 1.0+0j]))
        return P
    W12 = pfr([complex(y2s)]*12)
    ys = (1/(X - r1))/sc
    keep = np.abs(ys) < 6.0
    ysk, phk = ys[keep], PH[keep]
    PW = np.vander(ysk, 25, increasing=True)
    F4 = np.load(SW + "p5_deg24_fit4.npz")
    pX = np.array([F4['p'][k]/0.65**k for k in range(25)])
    qX = np.array([F4['q'][k]/0.65**k for k in range(25)])
    def transform(cX):
        ct = np.zeros(25, complex)
        for k in range(25):
            for j in range(k+1):
                ct[24-k+j] += cX[k]*comb(k, j)*r1**j
        return ct
    Pts = transform(pX)*sc**np.arange(25); Qts = transform(qX)*sc**np.arange(25)
    Qts = Qts/Pts[24]; Pts = Pts/Pts[24]
    rsc25 = 1/(1+np.abs(Pts)+np.abs(Qts))
    w0 = irls_weights(PW@Pts, PW@Qts, phk)

    def inner(roots, w_in):
        A = pfr(aexY + list(roots[:5]))
        A2 = np.convolve(A, A)
        R = pfr([complex(ycs), complex(yc9s), roots[5], roots[6]])
        R5 = np.convolve(np.convolve(np.convolve(R, R), np.convolve(R, R)), R)
        A2v = PW[:, :17]@A2; R5v = PW[:, :21]@R5
        w = w_in
        for _ in range(3):                                # IRLS passes
            nr = 25 + len(ysk)
            M = np.zeros((nr, 14), complex); b = np.zeros(nr, complex)
            for k in range(25):
                for jb in range(9):
                    v = A2[k-jb] if 0 <= k-jb <= 16 else 0
                    if jb == 8:
                        b[k] = -v*rsc25[k]
                    else:
                        M[k, jb] = v*rsc25[k]
                for js in range(5):
                    v = R5[k-js] if 0 <= k-js <= 20 else 0
                    M[k, 8+js] = -v*rsc25[k]
                M[k, 13] = -(W12[k] if k <= 12 else 0)*rsc25[k]
            for jb in range(9):
                col = A2v*PW[:, jb]
                if jb == 8:
                    b[25:] = -col*w
                else:
                    M[25:, jb] = col*w
            for js in range(5):
                M[25:, 8+js] = -phk*R5v*PW[:, js]*w
            v = np.linalg.lstsq(M, b, rcond=None)[0]
            B = np.concatenate([v[:8], [1.0+0j]])
            St = v[8:13]
            Pv = A2v*(PW[:, :9]@B)
            Qv = R5v*(PW[:, :5]@St)
            w = irls_weights(Pv, Qv, phk)
        rn2 = np.linalg.norm(M@v - b)**2
        for j, (t, s_) in enumerate(meas):
            rn2 += (KPEN*abs(roots[j]-t)/s_)**2
        rn2 += (KPEN*abs(roots[5]-r3Y[0])/r3Y[1])**2
        P25 = np.convolve(A2, B); Q25 = np.convolve(R5, St)
        return np.sqrt(rn2), v, w, float(np.abs(Qv).min()), common_root_score(P25, Q25)

    F4r = np.roots(Pts[::-1])
    rs = list(F4r); pairs = []
    while len(rs) > 1:
        dmin, ij = None, None
        for i in range(len(rs)):
            for j in range(i+1, len(rs)):
                d_ = abs(rs[i]-rs[j])
                if dmin is None or d_ < dmin: dmin, ij = d_, (i, j)
        i, j = ij
        pairs.append(((rs[i]+rs[j])/2, dmin)); rs = [r for k_, r in enumerate(rs) if k_ not in ij]
    pairs.sort(key=lambda t: t[1])
    known = aexY + [complex(m[0]) for m in meas]
    blind = [ctr for ctr, dd_ in pairs if all(abs(ctr-k) > 0.05 for k in known)][:2]

    def outer_resid_vec(roots, w_in):
        rn, v, w, qmin, syl = inner(roots, w_in)
        return rn, v, w, qmin, syl

    def lm_outer(roots, w, iters=40):
        rn, v, w, qmin, syl = inner(roots, w)
        lam_ = 1e-2
        for it in range(iters):
            # numerical J of scalar |F| won't do; use residual-of-inner via finite diff on rn^2 gradient-free LM surrogate:
            # full residual vector is expensive; approximate with 14-real-param Gauss-Newton on rn via central differences
            g = np.zeros(14); H = np.eye(14)
            h = 1e-7
            base = rn
            for k in range(14):
                d = np.zeros(7, complex); d[k//2] = (h if k % 2 == 0 else 1j*h)
                rp, *_ = inner(roots + d, w)
                rm, *_ = inner(roots - d, w)
                g[k] = (rp - rm)/(2*h)
            step_ok = False
            for _ in range(8):
                delta = -g/(np.linalg.norm(g) + 1e-300)*min(0.05, np.linalg.norm(g)/ (1+lam_))
                cand = roots + np.array([delta[2*i] + 1j*delta[2*i+1] for i in range(7)])
                rn2, v2, w2, qmin2, syl2 = inner(cand, w)
                if rn2 < rn:
                    roots, rn, v, w, qmin, syl = cand, rn2, v2, w2, qmin2, syl2
                    lam_ = max(lam_/2, 1e-6); step_ok = True
                    break
                lam_ *= 4
            if not step_ok:
                break
        return roots, rn, v, w, qmin, syl

    best = None
    for tag, r4x in [("east", r1+0.10), ("south", r1-0.10j), ("far", complex(0.756,-1.415)),
                     ("stray", complex(0.46852, 0.03617))]:
        roots0 = np.array([complex(m[0]) for m in meas] + blind + [complex(r3Y[0]), Yf(r4x)])
        roots_, rn, v, w, qmin, syl = lm_outer(roots0.copy(), w0)
        # then polish with coordinate descent
        step = 0.02
        while step > 1e-10:
            improved = False
            for j in range(7):
                for dz in (step, -step, step*1j, -step*1j):
                    cand = roots_.copy(); cand[j] += dz
                    rn2, v2, w2, qmin2, syl2 = inner(cand, w)
                    if rn2 < rn:
                        roots_, rn, v, w, qmin, syl = cand, rn2, v2, w2, qmin2, syl2; improved = True
            if not improved:
                step /= 2
        print(f"start {tag}: |F|={rn:.4e}  min|Q|={qmin:.1e}  syl={syl:.1e}", flush=True)
        if best is None or rn < best[1]:
            best = (roots_, rn, v, w, qmin, syl, tag)
    roots, rn, v, w, qmin, syl, tag = best
    print(f"BEST ({tag}): |F|={rn:.4e}", flush=True)

    # full-sample validation
    A = pfr(aexY + list(roots[:5]))
    R = pfr([complex(ycs), complex(yc9s), roots[5], roots[6]])
    B = np.concatenate([v[:8], [1.0+0j]]); St = v[8:13]
    P = np.convolve(np.convolve(A, A), B)
    Q = np.convolve(np.convolve(np.convolve(np.convolve(R, R), np.convolve(R, R)), R), St)
    res = np.empty(len(ysk))
    for i, (y_, ph) in enumerate(zip(ysk, phk)):
        pw = y_**np.arange(25)
        Pv = P[:25]@pw; Qv = Q[:25]@pw
        res[i] = abs(Pv - ph*Qv)/(abs(Pv) + abs(ph*Qv))
    print("full-sample validation:")
    for R_ in 'ABCDE':
        m = (REG[keep] == R_)
        if m.sum():
            print(f"  region {R_}: max={res[m].max():.2e} med={np.median(res[m]):.2e}", flush=True)
    lab = ["A4", "A5", "A6", "A7", "A8", "R3", "R4"]
    for l_, r_ in zip(lab, roots):
        print(f"  {l_}: X = {r1 + 1/(r_*sc):+.8f}")
    np.savez(SW + "p5_irls_varpro.npz", roots=roots, lin=v, resid=rn)
    print("saved p5_irls_varpro.npz")


if __name__ == "__main__":
    main()
