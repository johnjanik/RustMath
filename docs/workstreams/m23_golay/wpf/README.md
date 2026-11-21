# WP-F records

The records of work package WP-F (the Frobenius table of the M23 curve X). The brief is
`../HANDOFF.md`, Section 3; its Status block says what was reproduced and what is open.

| File | What |
|---|---|
| `design/WPF.md` | The design, with the prototype results and the test plan |
| `design/WPF.review1.md` | The design review (decision D1 rejected: Decided only for the nodal certificate) |
| `impl/WPF.md` | The implementation report: changes, tests, regenerated tables, follow-ups F-1 to F-11 |
| `impl/WPF.review1.md` | The code review (approved, six minor items) |
| `impl/WPF_docs_check.md` | The independent check of the tracker and handoff edits (approved, six minor items, fixed; item 6 by a note on the `ch118.IsWeil` override) |
| `impl/WPF_minors.md` | The fixes of the code review's six minor items, with the new tests and their PARI/GP values, and the fix-up after the checks of those fixes |
| `followup_f9/F9.md` | Follow-up F-9 (the owner's decision of 2026-11-15): the reduction certificate of Lemma 9.1 of the owner's paper, the criterion for absolute irreducibility with its proof, the prototype results, the statuses and the test plan; also F-5, F-1, F-8 |
| `followup_f9/F9.review1.md` | The F-9 design review (approved, seven minor items m1 to m7, folded into the implementation: `F9_impl.md` Section 6) |
| `followup_f9/F9_impl.md` | The F-9 implementation report: changes, tests with their references, both regenerated tables with every status, the comparison with the owner's main `2d11038`, times, deviations, handoffs; Section 9 gives the fixes of the code review's minor items |
| `followup_f9/F9.creview1.md` | The F-9 code review (approved, three optional minor items n1 to n3, fixed: `F9_impl.md` Section 9) |
| `followup_f9/F9.minorscheck1.md` | The independent check of those three fixes (approved, no regression; two cosmetic notes c1 and c2, fixed in the docs step that copied these records) |
| `gp/` | The PARI/GP scripts behind the reference values of `rustmath-curves/tests/biprojective_small_curves.rs`, of the overflow test in `rustmath-curves/src/weil.rs` and of the F-9 tests, each with its output (`*.out.txt`, PARI/GP 2.15.4) |

The reports are copied unchanged from the session that wrote them, with one exception: the
column "2-torsion order" of the tables of `impl/WPF.md` Section 3 (and its Section 7, item 5)
gives the values after C-5, as the note below its table description says. Line numbers in
`impl/WPF.md`, the reviews and the docs check refer to commit `a62da12`; the minor fixes moved
some of them.
`impl/WPF_minors.md` lists the new lines of the main items (its last section gives those moved
again by the fix-up), and the Status block of `../HANDOFF.md` is current. Paths of the form
`scratchpad/m23/...` name that session's scratch directory, which is not kept.

The F-9 records name their scratch directory `f9/`. In them, `f9/F9.md`, `f9/F9.review1.md`,
`f9/F9_impl.md`, `f9/F9.creview1.md` and `f9/F9.minorscheck1.md` are the files of
`followup_f9/`, and `f9/gp/...` is this folder's `gp/`. The rest is scratch work and is not
kept: `f9/proto/...` (the prototype), `f9/impl/...` (raw tables, logs and the comparison
script), `f9/review1/...`, `f9/rev1/...` and `f9/minors1/...` (the reviews' and the check's
runs), `f9/owner/...` (copies of the owner's files at `2d11038`), `f9/lemma_9_1.tex` (Lemma 9.1
and the text around it, from `paper/golay_m23.tex` at `2d11038`) and `f9/owner_decisions.md`.
Line numbers in the F-9 records refer to the F-9 change on top of `d91f880`, as reviewed.

The records of requests C-5 and C-7 (`../REQUESTS.md`; C-7 is follow-up F-10) are in
`../c57/`, with their own README.

The PARI/GP scripts and raw outputs of the scratch directories are not copied, except the
small-curve scripts and the F-9 scripts in `gp/`. Some of the others hold the coefficients of
HJLPPZ's `polynomial_F.m`, which has no licence and must not enter this repository; the scripts
in `gp/` hold no HJLPPZ data. Run them from `gp/` (`small_curves4.gp` reads `small_curves2.gp`)
with `gp -q <name>.gp < /dev/null`. Their origin in the scratch directories:

| Script | Origin | Test cases |
|---|---|---|
| `small_curves.gp`, `branch.gp` | `design/WPF/gp/` (the design) | curves A to I2, E1, E2 |
| `tangent.gp` | `design/WPF/review1/` (the design review) | curve T3 (`tacnode_tangent_to_the_fibre`) |
| `small_curves2.gp`, `small_curves4.gp` | `impl/gp/` (the implementation) | curves J, J', J2, K1, K2 |
| `verify.gp` | `impl/minors/` (the minor fixes) | the six curves taken from code review 1, Section 2.2, and E6 |
| `two_torsion_overflow.gp` | `impl/fixup/lcm_case.gp` (the fix-up) | `weil.rs::two_torsion_order_above_u64_panics` |
| `irred.gp` | `f9/gp/` (the F-9 design) | an independent GP version of the criterion (rational fibres, fibres over points of degree d, the smooth-point pass); read by the others |
| `regress_search.gp`, `regress.gp` | `f9/gp/` | the regression curves (i) to (v) of `reducible_reductions_are_never_decided` and (iv) of `smooth_22_curve_certified_by_a_smooth_point` |
| `unit.gp` | `f9/gp/` | Gauss's count of closed points (unit tests of `irreducible.rs`), E6 (`unclassified_e6_points`), `x^2 + 11` |
| `m23_cc0.gp`, `m23_p5.gp`, `m23_factor.gp` | `f9/gp/` | the criterion and PARI's bivariate factorization for G, G7, G13 at every prime 5..131 (`m23_p5.gp`: every rational fibre at 5 and the fibre over `y^2 + 2`; `x_at_5_tacnode_with_conjugate_branches`) |
| `lfield.gp` | `f9/gp/` | `[O_L : Z[s]] = 1` and the ramified primes of L (the doc of `calf_mod`) |
| `l_covers.py` | `f9/gp/` | the criterion on the covers over L: reads `polynomial_calF.m` from `$M23ISGALOIS` at run time, writes nothing, prints verdicts only (`python3 -I l_covers.py $M23ISGALOIS \| gp -q`) |
| `f9_tests.gp` | the F-9 implementation | the orders of the Rust search (closed points, smooth points) for E1, the gcd-2 cases S1 and S2 at p = 3 (`absolute_irreducibility_settled_by_a_residue_degree`) and E6 at 11 |

`m23_cc0.gp`, `m23_p5.gp` and `m23_factor.gp` read `G.gp`, Le Lay's CC0 form, which is not
kept here. Generate it in this folder from the vendored copy with

```
python3 -I -c "import json;c=json.load(open('../../../../../rustmath-curves/tests/data/m23/lelay_optimal_23_4_Z.json'))['coefficients_T_then_W'];print('GZ = '+'+'.join('(%s)*x^%d*y^%d'%(s,i,j) for i,r in enumerate(c) for j,s in enumerate(r) if s!='0')+';')" > G.gp
```

The outputs of the F-9 design scripts were rerun from a copy of this folder by the F-9
implementation and are identical (apart from PARI's stack-size warning).
