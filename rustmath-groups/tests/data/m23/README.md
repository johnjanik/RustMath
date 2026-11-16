# M23 test data (WP-P)

Reference data for the WP-P tests of `rustmath-groups` (`tests/m23_*.rs`, the example
`examples/m23_class_vectors.rs`). The brief is `docs/workstreams/m23_golay/HANDOFF.md`, Section 4.
The readers are in `tests/m23_common/mod.rs`.

## Vendored files

| File | Source | Licence | sha256 | Bytes |
|---|---|---|---|---|
| `m23_three_point_nielsen.json` | the owner's `johnjanik/M23`, `data/`, commit `2d11038` (unchanged since `d05475b`), copied unchanged | the owner's data, vendored as the HANDOFF allows | `b8189758a483d88beb46668ca67c3927fe3ad1738a84832105f0737c26ff0ba5` | 7,605 |
| `m23_three_point_passports.json` | same | same | `c67ad1b51960a30d0b4362651bcafb7cc6b16b499498c2290f3965afce16711f` | 7,960 |
| `group_facts.json` | same | same | `cd81e9c592fff09b7dd62f7232a58f0b6dba90dd4c3f903b10cfad65157b6e48` | 6,670 |
| `passports_C_23A_23B.json` | same | same | `b163d95b5ba738b56a09bb9addabf1a036ba3590f3117ce70eae9bc879e8a408` | 436,351 |
| `janik_table.json` | derived from the owner's `m23golay/nielsen.py` at `2d11038` (sha256 `86986ec0a0aa2eac5268adbe81daaba9586a030cff6bc745619eebccb4989bec`) by `extract_janik_table.py`; only `JANIK_TABLE` | the owner's transcription of Table 1 of his own manuscript [Janik26] (J. A. Janik, "Real Structure and the Galois Fixed Point in the Nielsen Class (2,23A,23B) of M23", 2026); the owner's data | `3db9d440f978f3ec6c190f66903804bdb6ec80043b0d3d4079847e072fa42dac` | 1,599 |
| `extract_janik_table.py` | new (RustMath) | RustMath | | |

To regenerate `janik_table.json` (the script parses the source with `ast`; it never imports or
runs the owner's code, and it reads `JANIK_TABLE` only, not `LELAY_X` or `HJLPPZ_FILE`):

```bash
python3 -I rustmath-groups/tests/data/m23/extract_janik_table.py <checkout of johnjanik/M23 at 2d11038> rustmath-groups/tests/data/m23/janik_table.json
sha256sum rustmath-groups/tests/data/m23/janik_table.json
```

The owner's `passports_C_23A_23B.json` carries `representatives_g1`: canonical representatives
that his code computed from the published generators. RustMath recomputes them
(`full_c_23a_23b`). They are not copies of HJLPPZ's or Le Lay's permutations.
`group_facts.json`, key `nielsen`, holds indices and dimensions only.

## The character table

`rustmath-groups/data/m23/m23_atlas_table.json` is written by
`rustmath-groups/data/m23/m23_atlas_table.g` (ours) from GAP 4.12.1 with CTblLib 1.3.7,
`CharacterTable("M23")`:

```bash
gap -q rustmath-groups/data/m23/m23_atlas_table.g < /dev/null > rustmath-groups/data/m23/m23_atlas_table.json
sha256sum rustmath-groups/data/m23/m23_atlas_table.json   # f9503e809aef25c369cf2b40c816bf72e23876d8162343cb75166e8d0f5196eb, 3,471 bytes
```

The values are the published ATLAS table of M23 (Conway, Curtis, Norton, Parker, Wilson, 1985,
p. 71), a mathematical fact, written in our own encoding by our own script and checked by
independent computation (exact orthogonality, the class data against RustMath's classes, the
counting identity on all 57 rows). CTblLib is GPL-3.0-or-later. The owner may want to confirm
this footing (follow-up WPP-Q1).

## Read at run time only, never vendored

| File | Licence | Variable | Commit | sha256 |
|---|---|---|---|---|
| `seven_nielsen_class_representatives.m` (HJLPPZ, `shaowuz/m23isgalois`) | none | `$M23ISGALOIS` | `f3866d1` | `2735178d38926b3dbca758540edc0150114823362cd5afa0ca02213221434fbb` |
| `verification/certify_geometric_reflection_obstruction.g` (Le Lay, `research-reflexivity/frontier-math-m23-cover-investigation`) | CC BY-NC-SA 4.0 | `$LELAY_M23` | `3f6ab4b` | `fad8d332cfb3f1748b76fd16cbc6bafe386e5d2cb1bc2b92da73d5424fb0fe61` |

Nothing of Le Lay is vendored: his seven representatives are in his CC BY-NC-SA certificate and
paper, and his CC0 `data/` has no Nielsen representatives. The `env_` tests of
`tests/m23_nielsen.rs` read both files and fail loudly without their variable:

```bash
M23ISGALOIS=<checkout of shaowuz/m23isgalois> \
LELAY_M23=<checkout of research-reflexivity/frontier-math-m23-cover-investigation> \
  cargo test -p rustmath-groups --test m23_nielsen -- --ignored env_
```
