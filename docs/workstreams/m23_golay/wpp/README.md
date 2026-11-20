# WP-P records

The records of WP-P of the M23 Golay workstream: the three-point class vectors of M23. The brief
is `../HANDOFF.md`, Section 4, and Section 8, step 3. Its Status block at the head of Section 4
says what was reproduced and what is open.

| File | What |
|---|---|
| `design/WPP.md` | The design: Revision 1 (the fixes of review 1) and the body. The algorithms and the proofs (class identification, the counting, the generation theorem, the real criteria), the reference values, a Rust prototype with measurements, the API, the test plan, the licence handling |
| `design/WPP.review1.md` | The first design review (not approved: one blocker, the licence handling of Le Lay's representatives; minor items m1 to m10) |
| `design/WPP.review2.md` | The second design review (approved; notes n1 to n5) |
| `impl/WPP.md` | The implementation report: the changes, the tests and their references, the 57 rows, the six `(C, 23A, 23B)` and the dictionaries, the times, the API, the deviations from the design, the follow-ups WPP-1 to WPP-10 and WPP-Q1, WPP-Q2. Its last section, "Minor items of review 1", reports the fixes of the code review's eight items |
| `impl/WPP.review1.md` | The code review (approved; minor items m1 to m8) |
| `impl/WPP.minorscheck1.md` | The check of the fixes of those eight items (approved; two optional nits) |
| `gp/` | The GAP scripts behind the test values and the reviews' GAP checks, with their output |

Line numbers in the copied reports refer to the state at the time of each report. Those of the
design refer to its scratch prototype. Those of `impl/WPP.md` Sections 1 to 11 and of
`impl/WPP.review1.md` refer to the code before the minor fixes, which moved some of them. The last
section of `impl/WPP.md` and `impl/WPP.minorscheck1.md` refer to the final state. The Status
block of `../HANDOFF.md` and the rows in `PROJECT_TRACKER.md` are current. Paths such as
`scratchpad/m23/wpp/...`, `WPP/...`, `WPP.rev1/...` or `impl_logs/...` name the scratch directory
of the session that wrote the reports. It is not kept.

The reports are copied unchanged. They hold no data of HJLPPZ's repository (it has no licence)
and nothing of Le Lay's CC BY-NC-SA paper or code. This was checked by a search of the copies
and of `gp/`: none of the cycles of the seven HJLPPZ representatives or of the permutations in Le
Lay's certificate, other than the published generators, occurs, and none of the 1271 integers of
six or more digits in the files of `m23isgalois` occurs. The published generators `g1`, `g2`,
`g3` do occur: they are printed in HJLPPZ's paper (arXiv:2608.08538v2, 3.1), are in the owner's
`m23golay/m23.py`, and are in RustMath's `M23::published`.

## gp/

GAP 4.12.1 with CTblLib 1.3.7. Run each script from its own folder (`code_review/nielsen_rows.g`
reads its neighbour `reps.g`):

```bash
gap -q <name>.g < /dev/null > <name>.out.txt
```

The `*.out.txt` files were written by the copies in this folder on 2026-11-16. They hold standard
output; `rv_gap.g` also prints a GAP syntax warning on standard error, which is not saved.
`design_review1/rv_gap.out.txt` is byte-identical to the output the first design review saved;
`design_review2/rv2_rows.out.txt` and `code_review/nielsen_rows.out.txt` equal the saved outputs up
to the run times in milliseconds they print. `design/probe.g`, `design/m23_checks.g` and
`design_review2/rv2_aut.g` had no saved output; their values are the ones the design and the second
design review quote (the centralizer orders and class sizes, the seven maximal subgroups, 16 table
automorphisms, `equal true`). Run times: `rv2_rows.g` about 160 s, `nielsen_rows.g` about 60 s, the
others under 10 s.

| Script | Origin | Values, and the tests or checks that use them |
|---|---|---|
| `design/probe.g` | the design | GAP's `CharacterTable("M23")`: class names, element orders, `SizesCentralizers`, `SizesConjugacyClasses`, power maps, the irreducibles. The constants of `tests/m23_classes.rs::centralizers_match_gap` and `GAP_CENTRALIZERS` of `tests/m23_three_point.rs` |
| `design/m23_checks.g` | the design | the table automorphisms (order 16); the maximal subgroups of `MathieuGroup(23)`, all intransitive but 23:11 (the generation theorem, `TrustedResult::AtlasMaximalSubgroups`); `g1 g2 g3 = 1` and the order of `<g1, g2>` (`m23_published_order_and_base`) |
| `design_review1/rv_gap.g` | the first design review | the group of the published generators, its maximal subgroups, the fusion of its classes to the ATLAS table, 3,000 random elements with their ATLAS class, and the 57 structure constants by `ClassMultiplicationCoefficient` (a second route to `atlas_orthogonality_and_57_constants`) |
| `design_review2/rv2_rows.g` | the second design review | ten rows recounted with no Golay trace, no transitivity theorem and no involution formula: survivors by `IsConjugate`, generation by `Size`, real classes by `RepresentativeAction` (a check of the rows of `eighteen_small_rows`) |
| `design_review2/rv2_aut.g` | the second design review | the 16 table automorphisms are exactly the Galois maps `g -> g^k` (exponent 212520). So every structure constant is invariant under them, and a naming of the split pairs that fits all power maps is GAP's up to such a map (the naming conventions checked by `split_invariants_separate`) |
| `code_review/nielsen_rows.g`, with `reps.g` | the code review | twelve rows by a third method: GAP's conjugacy tests against RustMath's class representatives, generation by `Size`, Nielsen classes as `Centralizer(G, b)`-orbits, checked free, real classes by `RepresentativeAction` |

`code_review/reps.g` holds RustMath's 17 class representatives (`ClassIdentifier`, ATLAS order).
It is cut from the `samples.g` that the code review's Rust probe wrote. In
`code_review/nielsen_rows.g` the absolute scratch path of that file is replaced by `reps.g`;
nothing else is changed in any script.

Not copied:

- The design's `m23_atlas_table.g` and its JSON output: the script that writes the embedded table
  is `rustmath-groups/data/m23/m23_atlas_table.g`, and it has the same code as the design's copy
  (only the header comments differ). Its output is the committed
  `rustmath-groups/data/m23/m23_atlas_table.json` (sha256
  `f9503e809aef25c369cf2b40c816bf72e23876d8162343cb75166e8d0f5196eb`).
- The code review's `check_classes.g`: it reads the 1.6 MB `samples.g` (20,000 random elements
  that the review's Rust probe labelled) and needs that probe to make sense. Its result is in
  `impl/WPP.review1.md`, Independent checks, items 1 and 2: none of the 20,000 is misclassified,
  and GAP's own table of the group (Dixon-Schneider) on RustMath's classes has the rows of the
  ATLAS table.
- The Python scripts: the design's `check_table.py` (an exact check of the table and of the 57
  structure constants), the design reviews' `check_n.py`, `check_vs_gap.py`, `colorth.py`,
  `m23rule.py`, `overlap.py`, `dict2.py`, `overlap2.py`, and the extractors `extract_tables.py`
  and `extract_janik_table.py`. The extractor behind the vendored Janik table is
  `rustmath-groups/tests/data/m23/extract_janik_table.py`.
- The Rust prototypes and the logs of the sessions.
- The owner's GAP script `scripts/gap/m23_passports.g` (it wrote his
  `data/m23_three_point_passports.json`, vendored in `rustmath-groups/tests/data/m23/`): it is
  his, in his repository at `2d11038`.
