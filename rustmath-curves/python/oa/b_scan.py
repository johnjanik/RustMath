"""mpmath train/holdout degree scan for B (the honest instrument; fp64 cliffs lie).

For each degree d: solve the value-matching system sum_k p_k X^k - phi sum_k q_k X^k = 0
(rows scaled 1/(1+|phi|), X rescaled into the unit disk) on the TRAIN split at dps 45,
report the relative residual median on the HOLDOUT split.  Expect the cliff at d = 24.

Usage: python3 b_scan.py [samples.npz] [scale] [degrees...]
"""
import numpy as np, mpmath as mp, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import b_config as BC

mp.mp.dps = 45


def mp_lstsq(A, b):
    """Least squares via mp QR on the normal-ish system: use qr_solve (A^T A well-formed
    at dps 45 given equilibrated rows)."""
    return mp.qr_solve(A, b)[0]


def scan(path, S, degrees):
    D = np.load(path, allow_pickle=True)
    X, PH = D['X'], D['PHI']
    n = len(X)
    rng = np.random.default_rng(11)
    idx = rng.permutation(n)
    ntr = int(0.8*n)
    tr, ho = idx[:ntr], idx[ntr:]
    Xm = [mp.mpc(x)/mp.mpf(S) for x in X]
    Pm = [mp.mpc(p) for p in PH]
    wm = [1/(1 + abs(p)) for p in Pm]
    print(f"{n} samples ({ntr} train, {n-ntr} holdout), scale {S}")
    for d in degrees:
        t0 = time.time()
        ncol = 2*(d + 1)
        A = mp.matrix(len(tr), ncol)
        bvec = mp.matrix(len(tr), 1)
        for r, i in enumerate(tr):
            pw = mp.mpc(1)
            for k in range(d + 1):
                A[r, k] = pw*wm[i]
                A[r, d + 1 + k] = -Pm[i]*pw*wm[i]
                pw *= Xm[i]
        # gauge: fix q_0 = 1 -> move column d+1 to rhs
        Ared = mp.matrix(len(tr), ncol - 1)
        for r in range(len(tr)):
            cc = 0
            for k in range(ncol):
                if k == d + 1:
                    bvec[r] = -A[r, k]
                    continue
                Ared[r, cc] = A[r, k]
                cc += 1
        try:
            v = mp_lstsq(Ared, bvec)
        except Exception as e:
            print(f"  d={d}: solve failed: {e}")
            continue
        p = [v[k] for k in range(d + 1)]
        q = [mp.mpc(1)] + [v[d + 1 + k] for k in range(d)]
        rels = []
        for i in ho:
            pw = mp.mpc(1)
            num = mp.mpc(0); den = mp.mpc(0)
            for k in range(d + 1):
                num += p[k]*pw
                den += q[k]*pw
                pw *= Xm[i]
            rels.append(abs(num - Pm[i]*den)/(abs(num) + abs(Pm[i]*den) + mp.mpf('1e-300')))
        rels = sorted(rels)
        med = rels[len(rels)//2]
        print(f"  d={d:2d}: holdout med rel = {mp.nstr(med, 4)}  max = {mp.nstr(rels[-1], 4)}"
              f"  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else BC.SW + "pB_samples1.npz"
    S = float(sys.argv[2]) if len(sys.argv) > 2 else 2.2
    degs = [int(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else [20, 22, 23, 24, 25, 26]
    scan(path, S, degs)
