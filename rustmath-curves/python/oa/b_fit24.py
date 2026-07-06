"""B: generic degree-24 fit (mp, all samples) + root census against the passport.

Fit phi = P(X/S)/Q(X/S) with q0 = 1 gauge at dps 45; report per-region holdout quality;
census: roots of P (expect 2^8 1^8), Q (5^4 1^4), P - Q (12^2) in X-coordinates, matched
against the measured addresses (c-pole, c3-pole, r1, r2 from the atlas; double at X=0).
Saves pB_deg24_fit.npz (p, q, S).
"""
import numpy as np, mpmath as mp, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import b_config as BC

mp.mp.dps = 45
D24 = 24


def fit(path, S):
    D = np.load(path, allow_pickle=True)
    X, PH, REG = D['X'], D['PHI'], D['region']
    Xm = [mp.mpc(x)/mp.mpf(S) for x in X]
    Pm = [mp.mpc(p) for p in PH]
    wm = [1/(1 + abs(p)) for p in Pm]
    n = len(X)
    ncol = 2*(D24 + 1)
    A = mp.matrix(n, ncol - 1)
    b = mp.matrix(n, 1)
    for r in range(n):
        pw = mp.mpc(1)
        cc = 0
        for k in range(D24 + 1):
            A[r, cc] = pw*wm[r]
            cc += 1
            pw *= Xm[r]
        pw = mp.mpc(1)
        for k in range(D24 + 1):
            if k == 0:
                b[r] = Pm[r]*pw*wm[r]
            else:
                A[r, cc] = -Pm[r]*pw*wm[r]
                cc += 1
            pw *= Xm[r]
    t0 = time.time()
    v = mp.qr_solve(A, b)[0]
    print(f"mp lstsq {n}x{ncol-1} [{time.time()-t0:.0f}s]", flush=True)
    p = [v[k] for k in range(D24 + 1)]
    q = [mp.mpc(1)] + [v[D24 + 1 + k] for k in range(D24)]
    # per-region validation
    rels = {}
    for r in range(n):
        pw = mp.mpc(1); num = mp.mpc(0); den = mp.mpc(0)
        for k in range(D24 + 1):
            num += p[k]*pw; den += q[k]*pw; pw *= Xm[r]
        rel = abs(num - Pm[r]*den)/(abs(num) + abs(Pm[r]*den) + mp.mpf('1e-300'))
        rels.setdefault(str(REG[r]), []).append(rel)
    for R in sorted(rels):
        v_ = sorted(rels[R])
        print(f"  region {R}: med={mp.nstr(v_[len(v_)//2], 3)} max={mp.nstr(v_[-1], 3)}")
    return p, q


def census(p, q, S):
    def roots_of(cf):
        deg = len(cf) - 1
        while deg > 0 and abs(cf[deg]) < mp.mpf('1e-30')*max(abs(c) for c in cf):
            deg -= 1
        rr = mp.polyroots([cf[i] for i in range(deg, -1, -1)], maxsteps=300, extraprec=300)
        return [complex(r)*S for r in rr]

    def cluster_report(name, roots, mult):
        roots = sorted(roots, key=lambda z: (round(z.real, 3), round(z.imag, 3)))
        used = [False]*len(roots)
        out = []
        for i in range(len(roots)):
            if used[i]:
                continue
            group = [i]
            for j in range(i + 1, len(roots)):
                if not used[j] and abs(roots[j] - roots[i]) < 0.06:
                    group.append(j)
            if len(group) >= mult:
                for g in group[:mult]:
                    used[g] = True
                ctr = np.mean([roots[g] for g in group[:mult]])
                spread = max(abs(roots[g] - ctr) for g in group[:mult])
                out.append((ctr, spread))
        print(f"  {name}: {len(out)} clusters of mult {mult}:")
        for ctr, spread in out:
            print(f"    {ctr:+.8f}  (ring {spread:.1e})")
        singles = [roots[i] for i in range(len(roots)) if not used[i]]
        if singles:
            print(f"    singles: " + "  ".join(f"{z:+.6f}" for z in singles))
        return out, singles

    print("\n=== census ===")
    pm = [p[k]/mp.mpf(S)**0 for k in range(25)]     # coefficients are in X/S variable
    Pr = roots_of(p)
    Qr = roots_of(q)
    PmQ = [p[k] - q[k] for k in range(25)]
    Wr = roots_of(PmQ)
    print("known addresses: c-pole -0.24339094-0.19286185j, c3-pole -0.23789929+0.23252559j,")
    print("                 double at 0; r1/r2 = the 12-clusters below")
    cluster_report("P zeros (expect 2^8 1^8)", Pr, 2)
    cluster_report("Q poles (expect 5^4 1^4)", Qr, 5)
    cluster_report("P-Q (expect 12^2)", Wr, 12)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else BC.SW + "pB_samples1.npz"
    S = float(sys.argv[2]) if len(sys.argv) > 2 else 2.2
    p, q = fit(path, S)
    np.savez(BC.SW + "pB_deg24_fit.npz",
             p=np.array([complex(x) for x in p]),
             q=np.array([complex(x) for x in q]), S=S)
    print("saved pB_deg24_fit.npz")
    census(p, q, S)
