# C-5 and C-7 records

The records of requests C-5 and C-7 of the M23 Golay workstream (`../REQUESTS.md`). C-5: the
order of Frobenius on `Jac(X)[2]` from the Frobenius polynomial, by the rank rule of REQUESTS 1.2
and the Weil pairing, with the type at the eigenvalue 1 and the class on the Golay Lagrangian
`W`. C-7: a JSON file per run of the M23 example (follow-up F-10 of WP-F). The brief is
`../HANDOFF.md`, Section 3 (WP-F); its Status block says what was reproduced and what is open.
C-6 (edits of `../HANDOFF.md` only) was done by the docs step of the same change and has no
records here.

| File | What |
|---|---|
| `design/C57.md` | The design, with the proofs of the rank bounds, the Weil-pairing constraints S1 and S2 and the Lagrangian criterion (Section 2), the prototype's tables and the test plan |
| `design/C57.review1.md` | The design review (approved, ten minor items, folded into the implementation: `impl/C57.md` Section 6) |
| `impl/C57.md` | The implementation report: changes, tests with their references, the JSON file of C-7, the tables of X and L with every bound and type (Section 4; 4.3 is REQUESTS 1.2 regenerated), the owner's scripts on the real file, deviations, handoffs; then the fix-up round after code review 1 and the minor items of code review 2, with a table of every `file:line` at the end |
| `impl/C57.review1.md` | Code review 1 (not approved: blocker B1, the memory of the rank bounds at large genus and large q; minor items m1 to m7; fixed in the fix-up round of `impl/C57.md`) |
| `impl/C57.review2.md` | Code review 2 (approved, four minor items n1 to n4, fixed or answered in `impl/C57.md`, section "Minor items (review 2)") |
| `impl/C57.minorscheck.md` | The independent check of those fixes (approved; one new optional item o1, the ` \| ` inside the order column of the text output, open) |

The header of `rustmath-curves/src/weil/two_torsion.rs` cites the proofs as
`docs/workstreams/m23_golay/c57/design/C57.md`, Section 2: that is `design/C57.md` here.

The reports are copied unchanged (byte for byte) from the session that wrote them. They name
their scratch directory `scratchpad/m23/c57/`; its `design/` and `impl/` files listed above are
the files of this folder, at the same paths. The PARI/GP and GAP scripts behind the C-5 tests
are vendored with their outputs in `rustmath-curves/tests/data/two_torsion/` (README with
commands, versions and sha256). The rest is scratch work and is not kept: the owner's files
that the design read (`design/owner/`), the prototype (`design/proto/`), the reruns of its
PARI/GP and GAP scripts (`impl/regen/`), the reviews' runs (`design/review1/`, `impl/review1/`,
`impl/review2/`, `impl/minorscheck/`), the work of the two fix rounds (`impl/fix1/`,
`impl/minors2/`), the logs (`impl/logs/`), the comparison scripts (`impl/gen/`) and their
output (`impl/compare_out.md`), the reports and checks of the docs step (`impl/C57_docs_*.md`,
`impl/docsverify1/`, `impl/docsfix2/`) and the two JSON files of the release runs, `impl/x.json`
(X, `5 131 4`, sha256
`186d1b989985a42ad569f34a1ee78161ebd9d4515ed7cc27492af17d180e1b38`) and `impl/l.json` (L,
`5 127 4 --cover L`, sha256 `88e26126d429b07cf6573994814a01a3923bfe79c1491c1661c2de8d5ad3021f`).
A new run of the example writes the same rows, apart from the times and the provenance
(`../HANDOFF.md`, Section 3, Status, Commands). Line numbers in these records refer to the
uncommitted change on `88f9174`; those of the last table of `impl/C57.md` are the final ones.

The copies hold no data of HJLPPZ's repository (it has no licence): none of the 1271 integers of
six or more digits in the files of `m23isgalois` occurs in them (checked by a search).

The docs step of the same change also replaced the column "2-torsion order" of the tables of
`../wpf/impl/WPF.md` Section 3 by the values after C-5, read from these JSON files (the note
below its table description says so).

A correction made after these records were written: the source of the hypothesis `H_W` is
Theorem 4.1 and Corollary 4.3 of the owner's paper (`paper/golay_m23.tex` of `johnjanik/M23`;
its Section 4.3 applies Theorem 4.1 to the covers over L). `design/C57.md` 2.5 cites it as
"Theorem 4.8 and Corollary 4.4" of that paper, and the `H_W` text of the code first cited
"HJLPPZ Thm. 4.8 and Cor. 4.4". Theorem 4.8 is the image of Galois on `W` for X, and
Corollary 4.4 the lifting to `2^11 : M23`; the Lagrangian is the owner's result (the paper's
introduction: in HJLPPZ the Golay structure enters only to certify the monodromy group). The code
(`tests/common/m23.rs`, `H_W`), the doc of `lagrangian_types` and `../HANDOFF.md` now give the
right source; the copies above are left as written.
