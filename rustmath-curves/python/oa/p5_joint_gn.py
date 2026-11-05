"""Joint Gauss-Newton: structure + data + measured addresses, analytic Jacobian.

Unknowns (21 complex): a3..a7 (free double-zero roots), r2 r3 (free 5-pole roots),
B0..B7 (simple-zero block, monic), St0..St4 (= lambda*S, simple poles), c.
Pinned hard (dd): a0 = y0s, a1, a2 (aexY), r0 = ycs, r1 = yc9s.

Rows: 25 structural coeffs of P - Q - c*W12 (equilibrated, weight WS)
      + all clean samples (P(y) - phi Q(y), IRLS-reweighted each iter)
      + soft measured-address rows (softs, c11 ring) at 1/sigma.
Solver: complex LM with Marquardt scaling (JhJ + lam*diag) -- fp64, fast multistart.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))

SW = "/home/john/sweep_2_12_5/"
r1x = complex(0.47182600647013, -0.10561240463346)
r2x = complex(-0.32224249915043, -0.38542142936930)
rc = complex(-0.056491567434, -0.311094205741)
rc9 = complex(1.152891104470, 0.579237023791)
a_ex = [complex(0.2120091233, -0.5634590964), complex(-0.7190508461, -0.4168884505)]
softs = [(complex(0.4775220321, 0.0580930745), 5e-4),
         (complex(-0.1303535948, -0.4094436433), 2e-5),
         (complex(0.5758909398, -0.1287776533), 5e-4)]
r3m = (complex(-0.494430, -0.406940), 1.5e-3)
sc = 8.0
WS = float(os.environ.get("WS", "30.0"))
KW = float(os.environ.get("KW", "1e-4"))     # scale of measurement rows vs IRLS data rows

Yf = lambda X: (1.0/(complex(X) - r1x))/sc
y2s = 1.0/(r2x - r1x)/sc
y0s = (-1.0/r1x)/sc
ycs, yc9s = Yf(rc), Yf(rc9)
aexY = [complex(y0s), Yf(a_ex[0]), Yf(a_ex[1])]
measY = [(Yf(m), s_/(abs(complex(m) - r1x)**2*sc)) for m, s_ in softs]
r3Y = (Yf(r3m[0]), r3m[1]/(abs(complex(r3m[0]) - r1x)**2*sc))

D = np.load(SW + "p5_samples6.npz")
X, PH, REG = D['X'], D['PHI'], D['region']
ys = (1.0/(X - r1x))/sc
keep = np.abs(ys) < 6.0
ysk, phk, regk = ys[keep], PH[keep], REG[keep]
NS = len(ysk)
PWf = np.vander(ysk, 25, increasing=True)          # NS x 25


def pfr(rr):
    P = np.array([1.0 + 0j])
    for r in rr:
        P = np.convolve(P, np.array([-r, 1.0 + 0j]))
    return P


W12 = pfr([complex(y2s)]*12)                        # len 13
W12p = np.zeros(25, complex); W12p[:13] = W12

IDX = dict(a=slice(0, 5), r=slice(5, 7), B=slice(7, 15), St=slice(15, 20), c=20)
NTH = 21


def unpack(th):
    a = aexY + list(th[IDX['a']])
    r = [ycs, yc9s] + list(th[IDX['r']])
    B = np.concatenate([th[IDX['B']], [1.0 + 0j]])
    St = th[IDX['St']]
    return a, r, B, St, th[IDX['c']]


def build(th):
    a, r, B, St, c = unpack(th)
    A = pfr(a); A2 = np.convolve(A, A)
    P = np.convolve(A2, B)
    R = pfr(r); R2 = np.convolve(R, R); R4 = np.convolve(R2, R2); R5 = np.convolve(R4, R)
    Q = np.convolve(R5, St)
    return a, r, B, St, c, A, A2, P, R, R4, R5, Q


RHO_BAR = float(os.environ.get("RHO_BAR", "3e-3"))    # anti-collapse barrier strength


def residual_and_jac(th, w_data, rsc, sbar=None):
    a, r, B, St, c, A, A2, P, R, R4, R5, Q = build(th)
    nbar = 2*NS if sbar is not None else 0
    nrows = 25 + NS + 4 + nbar
    res = np.zeros(nrows, complex)
    J = np.zeros((nrows, NTH), complex)
    # --- structural rows ---
    su = P - Q - c*W12p
    res[:25] = su*rsc*WS
    # dP/da_i (free i -> global root index 3+i): -2*Ahat*A*B
    def pad25(v):
        out = np.zeros(25, complex)
        out[:min(len(v), 25)] = v[:25]
        return out
    dPda = []
    for i in range(5):
        ah = pfr([a[j] for j in range(8) if j != 3 + i])
        d = pad25(-2*np.convolve(np.convolve(ah, A), B))
        dPda.append(d)
        J[:25, i] = d*rsc*WS
    dQdr = []
    for i in range(2):
        rh = pfr([r[j] for j in range(4) if j != 2 + i])
        d = pad25(5*np.convolve(np.convolve(rh, R4), St))  # dQ/dr = -5*Rhat*R4*St; row has -Q
        dQdr.append(d)
        J[:25, 5 + i] = d*rsc*WS
    for j in range(8):
        col = np.zeros(25, complex)
        seg = A2[:25 - j]
        col[j:j + len(seg)] = seg
        J[:25, 7 + j] = col*rsc*WS
    dQdSt = []
    for j in range(5):
        col = np.zeros(25, complex)
        seg = R5[:25 - j]
        col[j:j + len(seg)] = seg
        dQdSt.append(col)
        J[:25, 15 + j] = -col*rsc*WS
    J[:25, 20] = -W12p*rsc*WS
    # --- data rows ---
    Pv = PWf@P[:25]; Qv = PWf@Q[:25]
    res[25:25 + NS] = (Pv - phk*Qv)*w_data
    for i in range(5):
        J[25:25 + NS, i] = (PWf@dPda[i])*w_data
    for i in range(2):
        # res = P - phi*Q; dQ/dr_i = -(d); so d(res)/dr = -phi*dQ = +phi*d
        J[25:25 + NS, 5 + i] = phk*(PWf@dQdr[i])*w_data
    for j in range(8):
        J[25:25 + NS, 7 + j] = (PWf[:, j:j + 17]@A2)*w_data
    for j in range(5):
        J[25:25 + NS, 15 + j] = -phk*(PWf[:, j:j + 21]@R5)*w_data
    # struct rows for r: res has -Q: d(-Q)/dr = +d --> already set above as d*rsc*WS  (correct)
    # --- measured-address rows (softs = the first 3 free doubles, c11 = free r index 5) ---
    for j, (t, s_) in enumerate(measY):
        res[25 + NS + j] = KW*(th[j] - t)/s_
        J[25 + NS + j, j] = KW/s_
    res[25 + NS + 3] = KW*(th[5] - r3Y[0])/r3Y[1]
    J[25 + NS + 3, 5] = KW/r3Y[1]
    # --- anti-collapse barrier: rho*s_P/P(y), rho*s_Q/Q(y) (holomorphic; explodes on
    # fake shared roots, ~rho at the seed) ---
    if sbar is not None:
        sP, sQ = sbar
        Pv2, Qv2 = PWf@P[:25], PWf@Q[:25]
        o = 25 + NS + 4
        res[o:o + NS] = RHO_BAR*sP/Pv2
        res[o + NS:] = RHO_BAR*sQ/Qv2
        gP = -RHO_BAR*sP/Pv2**2
        gQ = -RHO_BAR*sQ/Qv2**2
        for i in range(5):
            J[o:o + NS, i] = gP*(PWf@dPda[i])
        for i in range(2):
            J[o + NS:, 5 + i] = gQ*(-(PWf@dQdr[i]))
        for j in range(8):
            J[o:o + NS, 7 + j] = gP*(PWf[:, j:j + 17]@A2)
        for j in range(5):
            J[o + NS:, 15 + j] = gQ*(PWf[:, j:j + 21]@R5)
    return res, J


def irls_w(th):
    a, r, B, St, c, A, A2, P, R, R4, R5, Q = build(th)
    Pv = PWf@P[:25]; Qv = PWf@Q[:25]
    return 1.0/np.maximum(np.abs(Pv) + np.abs(phk*Qv), 1e-300), \
        1.0/(1 + np.abs(P) + np.abs(Q))


def lm(th0, iters=120, verbose=True, w_from=None):
    """Weights FROZEN at the seed model (or w_from): IRLS self-normalization lets the
    optimizer sacrifice regions (weights die where the model blows up) -- the documented
    collapse trap.  Frozen weights keep every region's rows alive."""
    th = th0.copy()
    if w_from is not None:
        w_data, rsc, sbar = w_from
    else:
        w_data, rsc = irls_w(th)
        a, r, B, St, c, A, A2, P, R, R4, R5, Q = build(th)
        sbar = (np.abs(PWf@P[:25]), np.abs(PWf@Q[:25]))
    res, J = residual_and_jac(th, w_data, rsc, sbar)
    f = np.linalg.norm(res)
    lam = 1e-3
    for it in range(iters):
        JhJ = J.conj().T@J
        g = J.conj().T@res
        Dm = np.abs(np.diag(JhJ)).real
        Dm = np.maximum(Dm, 1e-12*Dm.max())
        ok = False
        for _ in range(25):
            try:
                dx = np.linalg.solve(JhJ + lam*np.diag(Dm), g)
            except np.linalg.LinAlgError:
                lam *= 8; continue
            thn = th - dx
            resn, Jn = residual_and_jac(thn, w_data, rsc, sbar)
            fn = np.linalg.norm(resn)
            if fn < f:
                th, res, J, f = thn, resn, Jn, fn
                lam = max(lam/3, 1e-14)
                ok = True
                break
            lam *= 5
        if verbose and (it % 10 == 0 or not ok):
            print(f"    it {it:3d}  |F| = {f:.6e}  lam={lam:.1e}", flush=True)
        if not ok:
            break
    return th, f


def report(th):
    a, r, B, St, c, A, A2, P, R, R4, R5, Q = build(th)
    Pv = PWf@P[:25]; Qv = PWf@Q[:25]
    rel = np.abs(Pv - phk*Qv)/(np.abs(Pv) + np.abs(phk*Qv))
    for R_ in 'ABCDE':
        m = regk == R_
        if m.sum():
            print(f"    region {R_}: max={rel[m].max():.2e} med={np.median(rel[m]):.2e}")
    su = P - Q - c*W12p
    print(f"    struct |P-Q-cW|/scale: max={np.max(np.abs(su)/(1+np.abs(P)+np.abs(Q))):.2e}")
    lab = ["A4", "A5", "A6", "A7", "A8", "R3", "R4"]
    Xs = [r1x + 1/(v*sc) for v in list(th[:5]) + list(th[5:7])]
    for l_, v in zip(lab, Xs):
        print(f"    {l_}: X = {v:+.10f}")
    return rel


if __name__ == "__main__":
    G3 = np.load(SW + "p5_gn3.npz")['roots']

    def inner_linear(roots, w, rsc):
        """14-col lstsq for (B0..7, St0..4, c) with the SAME rows/weights as the LM."""
        A = pfr(aexY + list(roots[:5])); A2 = np.convolve(A, A)
        R = pfr([ycs, yc9s, roots[5], roots[6]])
        R5 = np.convolve(np.convolve(np.convolve(R, R), np.convolve(R, R)), R)
        A2v = PWf[:, :17]@A2; R5v = PWf[:, :21]@R5
        M = np.zeros((25 + NS, 14), complex); b = np.zeros(25 + NS, complex)
        for k in range(25):
            for jb in range(9):
                v_ = A2[k - jb] if 0 <= k - jb <= 16 else 0
                if jb == 8:
                    b[k] = -v_*rsc[k]*WS
                else:
                    M[k, jb] = v_*rsc[k]*WS
            for js in range(5):
                v_ = R5[k - js] if 0 <= k - js <= 20 else 0
                M[k, 8 + js] = -v_*rsc[k]*WS
            M[k, 13] = -(W12[k] if k <= 12 else 0)*rsc[k]*WS
        for jb in range(9):
            col = A2v*PWf[:, jb]
            if jb == 8:
                b[25:] = -col*w
            else:
                M[25:, jb] = col*w
        for js in range(5):
            M[25:, 8 + js] = -phk*R5v*PWf[:, js]*w
        v, *_ = np.linalg.lstsq(M, b, rcond=None)
        return v, np.linalg.norm(M@v - b)

    # IRLS fixed point: d24-fit weights -> lstsq -> reweight from own model -> ...
    F4 = np.load(SW + "p5_deg24_fit4.npz")
    from math import comb
    pX = np.array([F4['p'][k]/0.65**k for k in range(25)])
    qX = np.array([F4['q'][k]/0.65**k for k in range(25)])

    def transform(cX):
        ct = np.zeros(25, complex)
        for k in range(25):
            for j in range(k + 1):
                ct[24 - k + j] += cX[k]*comb(k, j)*r1x**j
        return ct
    Pts = transform(pX)*sc**np.arange(25); Qts = transform(qX)*sc**np.arange(25)
    Qts = Qts/Pts[24]; Pts = Pts/Pts[24]
    # FROZEN d24-anchored scales for everything (rows, inner solve, barrier)
    w = 1.0/np.maximum(np.abs(PWf@Pts) + np.abs(phk*(PWf@Qts)), 1e-300)
    rsc = 1.0/(1 + np.abs(Pts) + np.abs(Qts))
    sbar = (np.abs(PWf@Pts), np.abs(PWf@Qts))
    roots0 = np.concatenate([G3[:5], G3[5:7]])
    v, rn = inner_linear(roots0, w, rsc)
    th0 = np.concatenate([roots0, v[:8], v[8:13], [v[13]]])
    print(f"seed from p5_gn3 + inner lstsq (frozen d24 weights): resid = {rn:.4e}")
    th, f = lm(th0, w_from=(w, rsc, sbar))
    print(f"  final |F| = {f:.6e}")
    report(th)
    np.savez(SW + "p5_joint.npz", theta=th, resid=f)
    print("saved p5_joint.npz")
