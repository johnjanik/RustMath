#!/usr/bin/env python3
"""Build transitive24_cycletypes.jsonl from the IGP24 repo's LMFDB cycle-type distribution.

Source: frobenius/cycle_types.jsonl in https://codeberg.org/johnjanik/inverse_galois
(produced there by frobenius/compute.sage from the LMFDB cycle-type distributions of all
25,000 groups 24Tt). Each source line looks like
    {"t": 1, "order": "24", "ntypes": 8, "dist": {"24": "8", "12,12": "4", ...}}
with cycle types written as descending comma-separated keys.

Output (what rustmath_groups::transitive24::CycleTypeSupport::load expects), one line per group:
    {"t": 1, "types": [[24], [12, 12], ...]}
with each cycle type as an ASCENDING list (the convention of transitive24::cycle_type).

Usage:  python3 build_cycletypes.py /path/to/inverse_galois/frobenius/cycle_types.jsonl [OUT]
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "transitive24_cycletypes.jsonl")
    n = 0
    seen = set()
    with open(src) as f, open(out, "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = int(rec["t"])
            types = sorted(sorted(int(x) for x in key.split(",")) for key in rec["dist"])
            if int(rec.get("ntypes", len(types))) != len(types):
                sys.exit(f"24T{t}: ntypes={rec['ntypes']} but {len(types)} keys")
            if any(sum(ct) != 24 for ct in types):
                sys.exit(f"24T{t}: a cycle type does not sum to 24")
            g.write(json.dumps({"t": t, "types": types}) + "\n")
            seen.add(t)
            n += 1
    if seen != set(range(1, 25001)):
        sys.exit(f"expected t = 1..25000 exactly once, got {n} lines / {len(seen)} distinct t")
    print(f"wrote {n} groups to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
