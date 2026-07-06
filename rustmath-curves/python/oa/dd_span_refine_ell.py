"""dd span refinement for an ELLIPTIC chart (order-m cone point: valuations mod m).

The span refinement itself is valuation-agnostic (reuses dd_span_refine.refine_span).
What changes: the jet normalization must invert the VALUATION rows (the top 9x9 block is
singular when vals = {v*, v*+m, ..., v*+8m}), and the ring gate runs at local_pow = m
(the top-echelon ratio x ~ w^m is the curve's local uniformizer at the cone point).
The residue v* is detected from the refined basis's row-magnitude pattern.

Usage: python3 dd_span_refine_ell.py <matrix.bin> <rho-full> <m> [out.npz] [iters]
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
from dd_span_refine import refine_span, y_to_b_mp
from svd_chart import to_pairs
from chart_dd import ring_gates

mp.mp.dps = 40


def detect_vals(B, m, kdim=9, scan=60):
    """Residue class of the valuations: row magnitudes of the b-basis concentrate on
    n = v* mod m."""
    rn = np.array([max(abs(complex(B[n, j])) for j in range(kdim)) for n in range(scan)])
    rn = rn/rn.max()
    scores = [sum(rn[n] for n in range(scan) if n % m == res) for res in range(m)]
    v0 = int(np.argmax(scores))
    others = sorted(scores, reverse=True)
    print(f"valuation residue detect: scores {['%.3f' % s for s in scores]} -> v* = {v0} "
          f"(margin {others[0]/max(others[1], 1e-30):.1f}x)")
    return [v0 + m*j for j in range(kdim)]


def jet_normalize_vals(B, vals, kdim=9):
    T = mp.matrix(kdim, kdim)
    for i, v in enumerate(vals):
        for j in range(kdim):
            T[i, j] = B[v, j]
    Tinv = T ** -1
    dim = B.shape[0]
    Bn = np.empty_like(B)
    for n in range(dim):
        for j in range(kdim):
            acc = mp.mpc(0)
            for k in range(kdim):
                acc += B[n, k]*Tinv[k, j]
            Bn[n, j] = acc
    return Bn


if __name__ == "__main__":
    path = sys.argv[1]
    rho_str = sys.argv[2]
    m = int(sys.argv[3])
    outnpz = sys.argv[4] if len(sys.argv) > 4 else path.replace(".bin", "_ddspan.npz")
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    dim, Yh, Yl, hist, s = refine_span(path, iters=iters)
    B = y_to_b_mp(Yh, Yl, rho_str)
    vals = detect_vals(B, m)
    Bn = jet_normalize_vals(B, vals)
    Bh, Bl = to_pairs(Bn)
    gates = ring_gates(Bh, Bl, local_pow=m)
    for R, ex, er in gates:
        print(f"  R={R}: |x-w^{m}|={ex:.2e}  ver={er:.2e}", flush=True)
    np.savez(outnpz, Bh=Bh, Bl=Bl, vals=np.array(vals), lam=0.0, rho=rho_str,
             resid_hist=hist, sv_null=s[-9:], gap=s[-10]/s[-9])
    print(f"saved {outnpz}", flush=True)
