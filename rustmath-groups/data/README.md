# Degree-24 transitive-group data (Phase 4 Galois narrowing)

Two data files back `src/transitive24.rs`. Both are large and **gitignored** —
regenerate them as below.

## `transitive_24.jsonl` (25,000 lines)
All degree-24 transitive groups with permutation generators: `label, t, order, parity, prim,
solv, gens`. `gens` are generators in 1-indexed disjoint-cycle notation on 24 points.

**Regenerate with GAP (recommended, offline):** GAP's TransGrp library uses the same numbering
as LMFDB for degree 24.
```bash
# Ubuntu/Debian: apt install gap-core gap-libs gap-transgrp     FreeBSD: pkg install gap
gap -q build_transitive_24.g      # about a minute
```
Validated 2026-11-05: order, parity and solvability agree with the LMFDB export for all 25,000
groups, and `native_closure_matches_precomputed_support` (GAP generators vs. LMFDB cycle types)
passes. GAP's generators can differ from LMFDB's but generate the same groups.

Alternative, LMFDB's PostgreSQL mirror (needs `psycopg2`, and port 5432 to
`devmirror.lmfdb.xyz`, which cloud sessions cannot reach): `python3 fetch_transitive_24.py`.

Spot-checks: 25000 rows; 24T1 order 24; 24T24680 (M₂₄) order 244823040, primitive;
24T7817 (PSL(2,23)) order 6072.

## `transitive24_cycletypes.jsonl` (25,000 lines)
Per group: `{"t": …, "types": [[cycle type], …]}` — the **cycle-type support**
(the set of cycle types occurring in the group), i.e. the Frobenius-blind data.
Derived from the LMFDB cycle-type distribution. Regenerate from the IGP24 repo
(https://codeberg.org/johnjanik/inverse_galois, see `docs/igp24/SOURCES.md`):
```bash
python3 build_cycletypes.py /path/to/inverse_galois/frobenius/cycle_types.jsonl
``` `CycleTypeSupport::blind_class`
matches a polynomial's observed Frobenius cycle types against these to pin Gal(f)
to its blind class. Native `group_closure` agrees with this support on every
group small enough to enumerate (test `native_closure_matches_precomputed_support`).
