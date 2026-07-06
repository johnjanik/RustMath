"""B structured solve, sigma-constrained, X-gauge (the antipodal-real parameterization).

sigma = M o conj (dd-measured, antipodal).  Every root class is a union of sigma-pairs
(m, sigma(m)); each pair contributes ONE free complex parameter.  Pinned at dd: the four
quintuple poles, both 12-points, the origin double-pair (0, sigma(0)).

P(X) = [(X)(X - sig0)]^2 * prod_{i=1..3} [(X-d_i)(X-sig(d_i))]^2 * prod_{j=1..4} (X-z_j)(X-sig(z_j))
Q(X) = lam * prod_{k=1..4} (X-P5_k)^5 * prod_{l=1..2} (X-s_l)(X-sig(s_l))
F    = P - Q - (1-lam)*W,   W = prod (X-r1)^12 (X-r2)^12 (monic; k=24 row vanishes identically)

Unknowns (10 complex = 20 real): d1..d3, z1..z4, s1..s2, lam.
Residuals: 24 complex structure rows (k=0..23, equilibrated) + 2 equivariance probes
(phi(sigma(x)) - conj(phi(x))) + optional data rows.  Real LM with FD Jacobian (basin),
then mp Newton (ladder) on the same residual.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import b_config as BC

SW = BC.SW
SIG = np.load(SW + "sigmaB.npy")
R12 = np.load(SW + "pB_r12.npy")
P5 = np.array([complex(-0.24339094187298, -0.19286184625628),
               complex(-0.23789929378150, +0.23252559046980),
               complex(-0.59198701480729, +0.21733468385806),
               complex(-0.64997276203624, -0.09319689864632)])
SIG0 = complex(-0.60511372176989, +0.07804715463807)


def sig(x):
    al, be, ga, de = SIG
    xc = np.conj(np.asarray(x))
    return (al + be*xc)/(ga + de*xc)


def pfr(rr):
    P = np.array([1.0 + 0j])
    for r in rr:
        P = np.convolve(P, np.array([-r, 1.0 + 0j]))
    return P


W24 = pfr([R12[0]]*12 + [R12[1]]*12)      # monic deg 24


def unpack(th):
    """th: 20 reals -> (d[3], z[4], s[2], lam) complex."""
    c = th[0::2] + 1j*th[1::2]
    return c[:3], c[3:7], c[7:9], c[9]


def build(th):
    d, z, s, lam = unpack(th)
    dbl = [0j, SIG0] + [v for m in d for v in (m, sig(m))]
    zer = [v for m in z for v in (m, sig(m))]
    pol = [v for m in s for v in (m, sig(m))]
    A = pfr(dbl)                            # deg 8 monic
    P = np.convolve(np.convolve(A, A), pfr(zer))       # deg 24 monic
    R5 = pfr(list(P5)*5)                    # deg 20 monic
    Q = lam*np.convolve(R5, pfr(pol))       # deg 24, lead lam
    return P, Q, lam


PROBES = np.array([0.35 - 0.75j, 0.9 + 0.55j])   # equivariance test points (covered-ish zone)


def phi_of(P, Q, x):
    pw = np.asarray(x)[..., None]**np.arange(25)
    return (pw@P)/(pw@Q)


def residual(th, w_data=None, data=None):
    P, Q, lam = build(th)
    F = P - Q - (1 - lam)*W24
    rsc = 1.0/(1 + np.abs(P[:24]) + np.abs(Q[:24]))
    rows = list((F[:24]*rsc).view(float)) if F[:24].dtype == complex else []
    rows = []
    Fe = F[:24]*rsc
    rows += list(Fe.real) + list(Fe.imag)
    # NOTE: no equivariance probe rows -- sigma-pairing is enforced exactly by the
    # parameterization, and probe rows are hypersensitive when sigma(probe) lands near a
    # model pole (they hijacked the LM).  lambda is determined by the data rows.
    if data is not None:
        X, PH, wd = data
        dv = (phi_of(P, Q, X) - PH)*wd
        rows += list(dv.real) + list(dv.imag)
    return np.array(rows)


def lm(th0, data=None, iters=200, verbose=True):
    th = th0.copy()
    f = np.linalg.norm(residual(th, data=data))
    lam_ = 1e-3
    n = len(th)
    for it in range(iters):
        r0 = residual(th, data=data)
        J = np.empty((len(r0), n))
        h = 1e-7
        for k in range(n):
            tp = th.copy(); tp[k] += h
            J[:, k] = (residual(tp, data=data) - r0)/h
        JtJ = J.T@J
        g = J.T@r0
        D = np.maximum(np.abs(np.diag(JtJ)), 1e-12*np.abs(JtJ).max())
        ok = False
        for _ in range(25):
            try:
                dx = np.linalg.solve(JtJ + lam_*np.diag(D), g)
            except np.linalg.LinAlgError:
                lam_ *= 8; continue
            thn = th - dx
            fn = np.linalg.norm(residual(thn, data=data))
            if fn < f:
                th, f = thn, fn
                lam_ = max(lam_/3, 1e-14)
                ok = True
                break
            lam_ *= 5
        if verbose and (it % 10 == 0 or not ok):
            print(f"  it {it:3d}: |F| = {f:.6e}", flush=True)
        if not ok:
            break
    return th, f


if __name__ == "__main__":
    # seeds from the census sigma-pair analysis
    d0 = [complex(-0.2472, -0.1230), complex(-0.4797, -0.3135), complex(-0.2647, +0.5061)]
    z0 = [complex(-0.786184, -0.150818), complex(-0.771144, -0.036603),
          complex(-0.609228, -0.181670), complex(-0.319166, -0.179817)]
    s0 = [complex(-0.772935, -0.044142), complex(-0.537357, -0.217289)]
    lam0 = 1.0 + 0j
    th0 = np.empty(20)
    seed = d0 + z0 + s0 + [lam0]
    th0[0::2] = [v.real for v in seed]
    th0[1::2] = [v.imag for v in seed]

    D = np.load(SW + "pB_samples_scan2.npz", allow_pickle=True)
    X, PH = D['X'], D['PHI']
    # frozen weights from the generic fit values
    F4 = np.load(SW + "pB_deg24_fit.npz")
    S = float(F4['S'])
    pw = (X/S)[:, None]**np.arange(25)
    Pv = pw@F4['p']; Qv = pw@F4['q']
    wd = 1.0/np.maximum(np.abs(Pv) + np.abs(PH*Qv), 1e-300)
    # rescale sample X into the model's coordinate (model is in raw X units)
    data = (X, PH, wd/np.median(wd)*0.05)

    print("=== structure-only LM ===")
    th1, f1 = lm(th0, data=None)
    print("=== structure + data LM ===")
    th2, f2 = lm(th1, data=data)
    d, z, s, lam = unpack(th2)
    print(f"\nfinal |F| = {f2:.6e}   lam = {lam:.10f}")
    for nm, vals in [("doubles", d), ("simple zeros", z), ("simple poles", s)]:
        for v in vals:
            print(f"  {nm}: {v:+.10f}  (sigma-pair {sig(v):+.10f})")
    np.savez(SW + "pB_struct.npz", theta=th2, resid=f2)
    print("saved pB_struct.npz")
