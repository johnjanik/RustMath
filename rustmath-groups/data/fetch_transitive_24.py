#!/usr/bin/env python3
"""Regenerate rustmath-groups/data/transitive_24.jsonl from the LMFDB public mirror.

All 25,000 degree-24 transitive groups (gps_transitive WHERE n=24). Requires psycopg2.
Usage:  python3 fetch_transitive_24.py [OUT]   (default: transitive_24.jsonl next to this file)

The companion transitive24_cycletypes.jsonl (cycle-type support per group) is derived from
the LMFDB cycle-type distribution; see README.md. Its generator was not committed and still
needs to be recovered from the original workstation.
"""
import json
import os
import sys

import psycopg2

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "transitive_24.jsonl")

conn = psycopg2.connect(host="devmirror.lmfdb.xyz", port=5432, user="lmfdb",
                        password="lmfdb", dbname="lmfdb", connect_timeout=30)
cur = conn.cursor()
cur.execute('''SELECT label,t,"order"::text,gens,parity,prim,solv,ab,cyc,
                      transitivity,num_conj_classes,abstract_label,
                      subfields,quotients,siblings,name,pretty,gapid,nilpotency
               FROM gps_transitive WHERE n=24 ORDER BY t''')
cols = [d[0] for d in cur.description]
n = 0
with open(OUT, "w") as f:
    for row in cur:
        rec = dict(zip(cols, row))
        rec["order"] = str(rec["order"])
        f.write(json.dumps(rec) + "\n")
        n += 1
print(f"wrote {n} rows to {OUT}", file=sys.stderr)
if n != 25000:
    sys.exit(f"expected 25000 rows, got {n}")
