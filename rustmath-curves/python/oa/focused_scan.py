"""Confirm the §3 band+tail selector recovers REAL physical forms at MULTIPLE N past the wall,
and find the robust (K, power) region.  Success = reality<<1, |X1-k^2|<1e-4, clean echelon 0-8."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mapkit, physical_selector as ps

SWEEP = "/home/john/sweep_2_12_5"
Ns = [int(x) for x in (sys.argv[1:] or [2000, 2300, 2500, 2800])]
for N in Ns:
    dim, Ahi = ps.load_hi(f"{SWEEP}/m_N{N}.bin")
    s, Vh = ps.fp64_svd_ascending(Ahi)
    print(f"\n=== N={N} (rho^N={mapkit.rho**N:.2e}) ===")
    raw = mapkit.evaluate(ps.sel_raw(s, Vh, dim), dim, "raw")
    print(f"  raw smallest-9         reality={raw['reality']:.2e} |X1-k^2|={raw['x1_err']:.2e} o12={raw['o12']:.3e}")
    for K in (100, 120, 160, 200):
        for power in (3.0, 4.0, 5.0):
            Y, _ = ps.sel_band_tail(s, Vh, dim, K=K, power=power)
            r = mapkit.evaluate(Y, dim, "")
            ok = "OK" if (r['reality'] < 1e-3 and r['x1_err'] < 1e-4 and r['valuations'][:9] == list(range(9))) else "  "
            print(f"  {ok} band K={K:3d} pow={power:g}  reality={r['reality']:.2e} "
                  f"|X1-k^2|={r['x1_err']:.2e} o12={r['o12']:.3e}")
