"""dd jet forms -> high-precision Hauptmodul -> high-precision map fit -> o12 (and, next, LLL).

The dd lift makes the FORMS accurate (~1e-12), but mapkit runs recover/Hauptmodul/fit in fp64, so
that accuracy never reaches the map (o12 stuck ~3; mpmath fit on fp64 X plateaus at 1.87). Here the
whole downstream pipeline runs in mpmath, so the dd form accuracy propagates to the order-12 fiber.

Usage: python3 jet_recognize.py [N] [dps]
"""
import numpy as np, sys, os, time, math
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, jet_tikhonov as jt, jet_dd
import mpmath as mp

rho = mapkit.rho; Lu = mapkit.Lu; d = mapkit.d; k = mapkit.k
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
mp.mp.dps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
vals = [0, 2, 4, 6, 8, 10, 12, 14, 16]

def mconv(a, b, L):
    o = [mp.mpc(0)] * L
    for i in range(min(len(a), L)):
        if a[i] == 0: continue
        for j in range(min(len(b), L - i)): o[i + j] += a[i] * b[j]
    return o
def murecip(sr, L):
    r = [mp.mpc(0)] * L; r[0] = 1 / sr[0]
    for n in range(1, L):
        s = mp.mpc(0)
        for j in range(1, min(n, len(sr) - 1) + 1): s += sr[j] * r[n - j]
        r[n] = -s / sr[0]
    return r
def mval(v, atol):
    for i, x in enumerate(v):
        if abs(x) > atol: return i
    return len(v)
def msdiv(num, den, L, atol):
    vn = mval(num, atol); vd = mval(den, atol)
    q = mconv(num[vn:], murecip(den[vd:], L), L)
    o = [mp.mpc(0)] * L; sh = vn - vd
    for i in range(len(q)):
        if 0 <= sh + i < L: o[sh + i] = q[i]
    return o

def hauptmodul_mp(B, dim, atol):
    """dd forms B (dim,9) -> Hauptmodul X (mpmath series length Lu). Mirrors mapkit in mpmath."""
    kap2 = mp.mpf(mapkit.KAPPA2)
    omw = [mp.mpf((-1) ** j * math.comb(k, j)) for j in range(k + 1)]
    n_needed = 2 * Lu
    scale = [mp.power(mp.mpf(rho), -n) for n in range(n_needed)]
    forms_u = []
    for j in range(B.shape[1]):
        bcol = [mp.mpc(B[n, j].real, B[n, j].imag) * scale[n] for n in range(n_needed)]
        conv = mconv(omw, bcol, n_needed)
        forms_u.append([conv[2 * i] for i in range(Lu)])
    # echelonize by valuation
    rows = [r[:] for r in forms_u]; D = len(rows)
    ech = {}
    for done in range(D):
        vv = [mval(rows[i], atol) if i >= done else 10 ** 9 for i in range(D)]
        pp = int(np.argmin(vv)); pv = vv[pp]
        if pv >= Lu: break
        rows[done], rows[pp] = rows[pp], rows[done]
        piv = rows[done][pv]; rows[done] = [x / piv for x in rows[done]]
        for r in range(D):
            if r != done and abs(rows[r][pv]) > atol:
                f = rows[r][pv]; rows[r] = [rows[r][i] - f * rows[done][i] for i in range(Lu)]
    for i in range(D):
        v = mval(rows[i], atol)
        if v < Lu: ech[v] = rows[i]
    cand = [v for v in ech if v + 1 in ech]
    if not cand: return None, ech
    m = max(cand); g = ech[m]; h = ech[m + 1]; cc = h[m + 2]
    den = [g[i] + cc * h[i] for i in range(Lu)]
    Xu = msdiv(h, den, Lu, atol)
    X = [Xu[n] * kap2 ** n for n in range(Lu)]
    return X, ech

def order12_mp(X):
    phiv = [mp.mpf(float(x.real)) for x in mapkit.phiv]
    Xp = [[mp.mpc(0)] * Lu for _ in range(d + 1)]; Xp[0][0] = mp.mpc(1)
    for i in range(1, d + 1): Xp[i] = mconv(Xp[i - 1], X, Lu)
    Nfit = 58
    cols = [mconv(phiv, Xp[i], Lu) for i in range(d + 1)] + \
           [[-Xp[i][n] for n in range(Lu)] for i in range(d + 1)]
    A = mp.matrix(Nfit, 2 * (d + 1))
    for n in range(Nfit):
        for j in range(2 * (d + 1)): A[n, j] = cols[j][n]
    U, S, Vt = mp.svd(A)
    nv = [Vt[Vt.rows - 1, j] for j in range(2 * (d + 1))]
    q = nv[:d + 1]; p = nv[d + 1:]
    Dp = [p[i] - q[i] for i in range(d + 1)]
    rr = [complex(r) for r in mp.polyroots([Dp[i] for i in range(d, -1, -1)],
                                           maxsteps=400, extraprec=400)]
    smed = np.median(np.abs([r for r in rr if abs(r) > 1e-14])); R = list(rr); worst = 0.
    for _ in range(2):
        best = None
        for i in range(len(R)):
            idx = sorted(range(len(R)), key=lambda j: abs(R[j] - R[i]))[:12]
            sp = max(abs(R[j] - R[i]) for j in idx)
            if best is None or sp < best[0]: best = (sp, idx)
        worst = max(worst, best[0] / smed); s = set(best[1])
        R = [R[j] for j in range(len(R)) if j not in s]
    return worst, p, q

if __name__ == "__main__":
    print(f"N={N} dps={mp.mp.dps}", flush=True)
    t = time.time(); dim, C = jet_dd.load_dd_C(f"/home/john/sweep_2_12_5/m_N{N}.bin")
    G = jet_dd.dd_gram(C, n_slices=8); w = jt.tail_weights(N)
    print(f"dd Gram [{time.time()-t:.0f}s]", flush=True)
    for lam in (1e-12,):
        t = time.time()
        B, rr, tn = jet_dd.solve_dd_refine(G, vals, lam, w, iters=4)
        print(f"dd forms lam={lam:.0e} H-resid={rr.max():.1e} [{time.time()-t:.0f}s]", flush=True)
        t = time.time(); X, ech = hauptmodul_mp(B, dim, mp.mpf('1e-8'))
        if X is None: print("  no consecutive valuations:", sorted(ech)); continue
        o12, p, q = order12_mp(X)
        print(f"  mpmath pipeline: o12={o12:.4e}  X[1]={complex(X[1]):.6g}  [{time.time()-t:.0f}s]",
              flush=True)
