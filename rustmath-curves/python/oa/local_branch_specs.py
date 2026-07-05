"""P0: branch-cycle local specs for the (2,12,5) M24 cover.

At a base elliptic point of order e, a branch cycle of length l has cover stabilizer h = e/l,
local cover coordinate s = u_e^h, and t - t0 ~ s^l.  The order-12 base point is elliptic for the
triangle group but the cover is ramified of order 12 there (l=12 => h=1), so its preimages are
ORDINARY cover points, not elliptic.  This module tabulates (l,h) per branch so the M2a solver
expands correctly at each preimage.
"""
import json, os
from dataclasses import dataclass
from typing import List, Tuple

TRIPLE = "/home/john/inverse_galois/M23/triple_2_12_5.json"

@dataclass(frozen=True)
class LocalBranch:
    vertex_name: str
    base_order: int          # e
    cycle_length: int        # l
    stabilizer_order: int    # h = e/l
    sheet_cycle: Tuple[int, ...]
    ramification_index: int  # l

@dataclass(frozen=True)
class VertexSpec:
    vertex_name: str
    base_order: int
    branch_value: str
    branches: List[LocalBranch]

def cycles_of_perm(p):
    n = len(p); seen = [False]*n; cycles = []
    for i in range(n):
        if seen[i]: continue
        cyc = []; x = i
        while not seen[x]:
            seen[x] = True; cyc.append(x); x = p[x]
        cycles.append(tuple(cyc))
    return cycles

def vertex_spec(vertex_name, base_order, branch_value, perm):
    branches = []
    for cyc in cycles_of_perm(perm):
        l = len(cyc)
        if base_order % l != 0:
            raise ValueError(f"cycle length {l} does not divide base order {base_order}")
        branches.append(LocalBranch(vertex_name, base_order, l, base_order // l, cyc, l))
    return VertexSpec(vertex_name, base_order, branch_value, branches)

def build_specs(sigma0, sigma1, sigmainf):
    return {
        "t0":   vertex_spec("t0",   2,  "0",     sigma0),
        "t1":   vertex_spec("t1",   12, "1",     sigma1),
        "tinf": vertex_spec("tinf", 5,  "infty", sigmainf),
    }

def load_triple(path=TRIPLE):
    d = json.load(open(path))
    return d["sigma0"], d["sigma1"], d["sigmainf"]

def summarize(spec):
    from collections import Counter
    return Counter((b.cycle_length, b.stabilizer_order) for b in spec.branches)

if __name__ == "__main__":
    s0, s1, si = load_triple()
    specs = build_specs(s0, s1, si)
    print("branch-cycle structure (cycle_length l, stabilizer h=e/l) : count")
    expect = {"t0": {(2,1):8,(1,2):8}, "t1": {(12,1):2}, "tinf": {(5,1):4,(1,5):4}}
    ok = True
    for name in ("t0", "t1", "tinf"):
        got = dict(summarize(specs[name]))
        match = (got == expect[name])
        ok = ok and match
        print(f"  {name:5s} e={specs[name].base_order:2d} over phi={specs[name].branch_value:5s}: "
              f"{ {f'(l={l},h={h})': c for (l,h),c in sorted(got.items())} }   "
              f"{'OK' if match else 'MISMATCH expected '+str(expect[name])}")
    print("REGRESSION:", "PASS" if ok else "FAIL")
