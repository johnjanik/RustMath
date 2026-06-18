# Degree-24 transitive-group data (Phase 4 Galois narrowing)

Two data files back `src/transitive24.rs`. Both are large and **gitignored** —
regenerate them as below.

## `transitive_24.jsonl` (25,000 lines)
All degree-24 transitive groups from the LMFDB public PostgreSQL mirror
(`gps_transitive WHERE n=24`): `label, t, order, gens, parity, prim, solv, …`.
`gens` are permutation generators in 1-indexed disjoint-cycle notation on 24 points.

Regenerate (needs a Python with `psycopg2`):
```python
import psycopg2, json, os
OUT = "transitive_24.jsonl"
conn = psycopg2.connect(host="devmirror.lmfdb.xyz", port=5432, user="lmfdb",
                        password="lmfdb", dbname="lmfdb", connect_timeout=30)
cur = conn.cursor()
cur.execute('''SELECT label,t,"order"::text,gens,parity,prim,solv,ab,cyc,
                      transitivity,num_conj_classes,abstract_label,
                      subfields,quotients,siblings,name,pretty,gapid,nilpotency
               FROM gps_transitive WHERE n=24 ORDER BY t''')
cols = [d[0] for d in cur.description]
with open(OUT, "w") as f:
    for row in cur:
        rec = dict(zip(cols, row)); rec["order"] = str(rec["order"])
        f.write(json.dumps(rec) + "\n")
```
Spot-checks: 25000 rows; 24T1 order 24; 24T24680 (M₂₄) order 244823040, primitive;
24T7817 (PSL(2,23)) order 6072.

## `transitive24_cycletypes.jsonl` (25,000 lines)
Per group: `{"t": …, "types": [[cycle type], …]}` — the **cycle-type support**
(the set of cycle types occurring in the group), i.e. the Frobenius-blind data.
Derived from the LMFDB cycle-type distribution. `CycleTypeSupport::blind_class`
matches a polynomial's observed Frobenius cycle types against these to pin Gal(f)
to its blind class. Native `group_closure` agrees with this support on every
group small enough to enumerate (test `native_closure_matches_precomputed_support`).
