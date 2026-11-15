# Steiner labelling records

The records of the Steiner labelling exercise of the M23 Golay workstream. The brief is
`../HANDOFF.md`, Section 6 (last paragraph) and Section 8, step 2. Its Status block at the end of
Section 6 says what was reproduced and what is open.

| File | What |
|---|---|
| `design/SL.md` | The design: the algorithm and its proof, the reference values from two oracles, a Rust prototype with measurements, the API, the test plan |
| `design/SL.review1.md` | The design review (approved, minor items m1 to m9) |
| `impl/SL.md` | The implementation report: the changes per crate, the tests and their references, the reproduced values, the times, the deviations from the design, the follow-ups SL-1 to SL-11. Its last section, "Minor items of review 1", reports the fixes of the code review's five items |
| `impl/SL.review1.md` | The code review (approved, five minor items) |
| `impl/SL.minorscheck1.md` | The check of the fixes of those five items (approved, four optional nits) |
| `gp/` | The PARI/GP scripts behind the test values, with their output |

Line numbers in the copied reports refer to the state at review time. Those of the design refer
to its scratch prototype. Those of `impl/SL.md` Sections 1 to 8 and of `impl/SL.review1.md` refer
to the code before the minor fixes, which moved some of them. The last section of `impl/SL.md` and
`impl/SL.minorscheck1.md` refer to the final state. The Status block of `../HANDOFF.md` and the
rows in `PROJECT_TRACKER.md` are current. Paths such as `scratchpad/m23/sl/...`, `design/SL/...`
or `impl/logs/...` name the scratch directory of the session that wrote the reports. It is not
kept.

The reports are copied unchanged, with one exception: `design/SL.review1.md` (Section 4) quoted the
fingerprint line of HJLPPZ's `polynomial_f1.gp`. That repository has no licence, so the value is
redacted in this copy.

## gp/

PARI/GP 2.15.4. Run each script from its own folder, because the scripts read their neighbours by
relative path:

```bash
gp -q -D parisizemax=6000000000 <name>.gp < /dev/null
```

The drivers that use f1 and f2 read them at run time from `$M23ISGALOIS`, a checkout of
`shaowuz/m23isgalois`. In these copies the absolute path of the scratch runs is replaced by
`getenv("M23ISGALOIS")`, as the design review asked (Section 4); without the variable they stop
with "set M23ISGALOIS to a checkout of shaowuz/m23isgalois". No script and no output here holds
HJLPPZ data: every integer of six or more digits in the files of `m23isgalois` was searched for
in this folder, with no match. `design/cc0_poly.gp`, `design_review/cc0_raw.gp` and
`impl/cc0_raw.gp` (the same file) hold Le Lay's `data/specialization_t2_Z.json` of
`research-reflexivity/frontier-math-m23-cover-investigation` (CC0-1.0; Knabble, Inc., dba
Reflexivity); the scripts build its monic associate. `design/signs_data.gp`,
`design_review/signs_f1.gp` and `signs_f2.gp` hold the owner's Frobenius signs
(`data/frobenius_signs.json` of `johnjanik/M23` at `d05475b`).

Outputs. The `*.out.txt` files were written by the copies in this folder on 2026-11-15; every
script ran and reproduced the values that the tests assert. The other outputs are those the
sessions saved: `design/cc0_out.txt`, `cc0_t5.txt`, `cc0_j.txt`, `design_review/cc0_rev_t5.txt`,
`impl/check_sl.txt`, `impl/ff_refs.txt`. The copies reproduce them, except that `ff_refs.txt`
predates the last two lines of `ff_refs.gp` (`ff_refs.out.txt` has them).
`design_review/run_cc0.gp` appends to `cc0_rev_t5.txt`, and `design/run_cc0_67.gp` and
`run_cc0_signs.gp` read `cc0_t5.txt`, so run `design_review/run_cc0.gp` in a copy of its folder.
In `design/run_signs.out.txt` the five "disagreements" of f2 are inert primes below 4000 that the
data file does not list (it lists 40); no listed sign disagrees.

| Script | Origin | Values, and the tests that assert them |
|---|---|---|
| `design/steiner.gp` | the design (library) | heptads from the masks `0xAE3`, `0xC75`; Frobenius-ordered roots by `polrootspadic`; `T5`, `J5`, `J7` |
| `design/run_cc0.gp` (with `cc0_poly.gp`), output `cc0_out.txt`, `cc0_t5.txt`, `cc0_j.txt` | the design | the CC0 polynomial at 29 (`N` = 270 and 250), `J5`, `J7`: `cc0_labelling_at_29`, `cc0_golay_invariants` |
| `design/run_cc0_67.gp` | the design | the CC0 polynomial at 67 (`N` = 216): `cc0_labelling_at_67` |
| `design/run_cc0_signs.gp` | the design | the twelve CC0 Frobenius signs below 900: `cc0_frobenius_signs`, `full_cc0_frobenius_signs` |
| `design/c23.gp`, `c23_p3.gp` | the design | C23 (`polsubcyclo(47, 23)`) at 2 and 3, and `x^23 - x - 1` at 23: `c23_is_unresolved_at_every_precision`, `s23_refutes_the_hypothesis` |
| `design/small_lift.gp` | the design | `x^3 + x + 1` at `5^6`, `x^23 - x - 1` at `23^4`: `small_roots_match_pari` |
| `design/run_f1.gp`, `run_f2.gp`, `run_second.gp`, `run_signs.gp` (with `signs_data.gp`) | the design | f1 and f2 at 29, at the second primes, and the 80 signs: a third check of the `env_` values, which come from the owner's data |
| `design_review/rev.gp` | the design review (library: Newton in `Mod(Mod(., p^N), f)`, direct sums over the 253 heptads) | the second reference |
| `design_review/run_cc0.gp` (with `cc0_raw.gp`), output `cc0_rev_t5.txt` | the design review | the CC0 values at 29 again (`second_reference` of `rustmath-galois/tests/data/m23/cc0_t2_steiner_reference.json`) |
| `design_review/run_f.gp` | the design review | `best`, `B5` and the residues at `N` = 9, 10, 11 and 30 for f1 and f2: `env_f2_minimal_precision`, `env_six_integers_and_signs` |
| `design_review/run_j.gp` | the design review | `B_J5`, `B_J7`, `J5`, `J7` of f1 and f2; C23 at 2, 3, 5; `x^23 - x - 1` at 23: `bounds_and_precision`, `c23_is_unresolved_at_every_precision`, `s23_refutes_the_hypothesis` |
| `design_review/run_signs.gp` (with `signs_f1.gp`, `signs_f2.gp`, `cc0_raw.gp`) | the design review | the 80 signs, the second primes, and the CC0 signs equal to the f1 rows: `env_frobenius_signs`, `env_second_prime`, `full_cc0_frobenius_signs` |
| `design_review/run_wrong.gp` | the design review | `x^23 - x - 1` at 157 with `N` = 4, a false Decided: `false_hypothesis_can_decide` |
| `impl/check_sl.gp` (with `rev.gp`, `cc0_raw.gp`, `cc0_oracle_values.gp`), output `check_sl.txt` | the implementation | the CC0 residues at (29, 250) and (67, 216), C23 at 2, 3, 5 and its residues mod 5 and 25, `x^23 - x - 1` at 23 (`N` = 29, 30) and 157 (`N` = 4, 18) |
| `impl/ff_refs.gp`, output `ff_refs.txt` | the implementation | factorizations mod 2, mod 5 and above `fp_factor::MAX_P`, the image of `t` above `MAX_P`, the inert primes 23 and 157 of `x^23 - x - 1`, `polsym`: `roots_above_the_word_size_bound_match_pari`, `trace_is_the_orbit_sum`, `with_precision_projects_and_continues`, `refusals` of `rustmath-finitefields/tests/galois_ring_frobenius.rs` |
| `impl/small_lift2.gp` | the implementation | `x^5 - x + 1` at `3^7`: `small_roots_match_pari`. It replaces the quintic `x^5 + 2` of `ff_refs.gp`, which is reducible mod 3 (`impl/SL.md`, Section 6, item 13) |

Not copied: the design's `run_duad.gp` and `duad_ref.gp` (a copy of `R253_duad` of the owner's
data, already vendored in `rustmath-galois/tests/data/m23/`, and a check of that data, not of a
test value); the design review's `run_mut.gp` (an experiment: the order `phi^-1` swaps the
signs); the code review's own PARI/GP reproduction (`impl/SL.review1.md`, Section 2), whose drivers
read their helpers by absolute scratch paths and whose `vendored_res.gp` is generated from the
vendored data; and the Python oracle scripts (the one behind the vendored reference file is
`rustmath-galois/tests/data/m23/make_cc0_reference.py`).
