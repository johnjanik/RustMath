# WP-P design review 1

Reviewer: independent adversarial review of `WPP.md` (769 lines, no Revision sections) and its
prototype `WPP/`. Date 2026-11-16. Scratch work in `WPP/review1/` (scripts, GAP output, logs; the
scratch crate's target directory was deleted after the runs). Nothing in the design, the worktree
`/home/user/RustMath-wpp` (clean at `28c4b69`) or the owner's checkout was changed.

## Verdict

**Not approved: one blocker (B1, licence handling of Le Lay's representatives).** The mathematics,
the class identification, the table import, the counting, the generation criterion, the real
counts, the HJLPPZ handling, the tests and the cost claims all hold up under reproduction. The
blocker is cheap to fix. The minor items should be folded into the revision.

## 1. What I reproduced

### 1.1 Group theory (GAP 4.12.1, independent script `review1/rv_gap.g`)

- `Group(g1, g2)` on the published cycles: order 10,200,960, `IsomorphismTypeInfoFiniteSimpleGroup`
  is M(23), `g1 g2 g3 = ()`.
- `MaximalSubgroupClassReps` computed on the published group itself (the design ran it on
  `MathieuGroup(23)`): orders 443520, 40320, 40320, 20160, 7920, 5760, 253; only the last is
  transitive. The generation theorem of 5.2 holds as stated, and ATLAS p. 71 is the M23 entry.
- `AutomorphismsOfTable(CharacterTable("M23"))`: order 16, generators `(16,17)`, `(14,15)`,
  `(10,11)`, `(7,8)(12,13)`. Power map 2 sends 14a to 7a. The column permutations that preserve
  orders, sizes and all power maps form the same `(Z/2)^4`, so the claim of 3.3 (any naming that
  respects the power maps differs from GAP's by a table automorphism) is right.

### 1.2 Class identification against GAP (`review1/py/m23rule.py`, `check_vs_gap.py`)

I re-implemented the rule of Section 3 in Python from the design text (not from the prototype):
cycle type; Golay trace on `G0` for 7, 14, 15, 23 with 7A, 15A = trace 1 and 23A = class of `g2`;
14A by `14A^2 in 7A`; 11A by the heptad-position family of `h11`. GAP labelled its 17 class
representatives and 3,000 `PseudoRandom` elements with ATLAS names through
`TransformingPermutationsCharacterTables(CharacterTable(G), CharacterTable("M23"))`.

Result: the rule agrees with GAP's names on all 3,017 elements up to exactly one table
automorphism (the swap `15a <-> 15b`, which is GAP's arbitrary choice inside that fusion). Every
split class was hit about 200 times. `tr(g2) = 0`, `tr(g3) = 1`, `class(h11) = 11A`,
`class(h11^2) = 11B`, as in 3.3.

Misclassification hunt. Inside M23 I found none, and the argument of 3.2 is sound: each invariant
is a class function, it separates `x` from `x^-1`, and each split type is exactly two classes (the
census checks this). The fast 11 test is a function of the family, so it is exact for the same
reason, and the 35 s family check is redundant. Outside M23 the rule misclassifies, as documented:
an involution of type `2^8 1^7` not in M23 is called 2A, and a 22-cycle (signature `(1,1,1)`) is
called 11B by the fast path. See m9 for the doc mismatch this exposes.

### 1.3 Character table and structure constants

- `gap -q m23_atlas_table.g < /dev/null` regenerates the JSON byte for byte: sha256
  `f9503e809aef25c369cf2b40c816bf72e23876d8162343cb75166e8d0f5196eb`, 3,471 bytes.
- Exact column orthogonality of the JSON (`review1/py/colorth.py`): holds. (Row orthogonality is
  checked by the prototype and by `check_table.py`.)
- Independent structure constants from GAP's `ClassMultiplicationCoefficient(t, i, j, k^-1) /
  |C(k)|` for all 816 multisets of nontrivial classes. Enumerating rational multisets and
  `(R, D, D^-1)` with genus >= 0 and `n > 0` gives exactly the 57 reference vectors, and `n` and
  `genus23` agree with both owner files on every row (`review1/py/check_n.py`).

### 1.4 The prototype, rebuilt and rerun

Scratch crate `review1/proto` (path dependencies on the worktree, its own target dir, offline,
`-j 2`): release build 1 min 16 s, 249 MB (design: 1 min 14 s, 247 MB); debug build 30 s.

| Test | Result | Time here | Design |
|---|---|---|---|
| `proto_release_streaming` | 57/57 rows, identity on every row, 17 sizes and 17 centralizers equal GAP's | 8.4 s, VmHWM 73 MB | 7.9 s, 73 MB |
| `proto_release_nielsen_cohomology` with `M23ISGALOIS` | six `(C, 23A, 23B)` sizes, real data and `dim W = g`, `dim J2 = 2g`, `g + r - 1` on every triple; HJLPPZ rows -> `[5, 4, 0, 2, 1, 3, 6]`; Janik -> `[3, 5, 1, 6, 2, 4, 0]`; Le Lay -> `[0, 1, 2, 4, 6, 5, 3]`; sign = trace on all 887,040 23-elements | 8.1 s in all (C part 0.70 s), VmHWM 82 MB | C part 0.58 s |
| `proto_debug_fast` (debug) | orbits 3,795 and 56,672; 14 centralizers by sifting; table checks; five `(2A, D, D^-1)` rows; `(3A, 5A, 6A)` | 1.86 s | about 1.9 s |
| `proto_debug_18_rows` (debug) | 18 rows | 0.61 s | 0.6 s |
| `proto_debug_dictionaries` (debug) | three dictionaries, seven-class cohomology against `group_facts.json` | 0.02 s | 26 ms |

`run_release2.log` in the design directory records only the failed first run; the passing rerun
was not saved. My rerun above passes, so the claims of Sections 0, 6 and 7 stand.

Cost sample of the `Matrix<PrimeField>` cross-check: one 44 x 66 GF(2) kernel takes 71 ms and one
rank 44 ms in debug, so the debug estimate "1 to 3 s" for the class-5 cross-check is safe.

### 1.5 The tests can fail (mutations of the prototype, debug)

| Mutation | Caught by | Failure |
|---|---|---|
| 11 never separated (`class_of` always 11A) | `proto_debug_18_rows` | identity: 62 against `n = 31` |
| generation test removed | `proto_debug_fast` | `(2A, 14A, 14B)` size 21 against 17 |
| one character degree changed (22 -> 21) | `proto_debug_fast` | identity: `n = 1142/161` against 7 |
| Le Lay read without the braid | `proto_debug_dictionaries` | `index_of` fails |
| 7A/7B (and so 14A/14B) names swapped | nothing, by design | all rows pass, as 3.3 predicts |

The reference values are external: `n` from GAP's table, sizes and real counts from the owner's
JSON, centralizers from GAP's `SizesCentralizers`, dictionaries from `group_facts.json`.

### 1.6 Other checks

- `cargo tree --offline --locked`: neither `rustmath-coding` nor `rustmath-finitefields` reaches
  `rustmath-groups`, so the new dependency and the dev-dependency make no cycle.
  `rustmath-numberfields` does depend on `rustmath-groups` (the reason given for `AtlasNumber`).
- `cyclic_golay::GolaySign::Plus` is `0xAE3`, as the prototype's `+1` system.
- sha256 of the four owner files and of `nielsen.py` match Section 11. The owner's WP-P files are
  unchanged between `d05475b` and `2d11038` (`git diff --stat` empty).
- `.gitignore` does not ignore `rustmath-groups/data/m23/` or `tests/data/m23/`.
- HJLPPZ: the file has no licence (sha256 `2735178d...4fbb` as recorded). The design reads it only
  at run time from `$M23ISGALOIS`, records only its hash, and the `env_` test panics without it.
  `extract_tables.py` takes only `JANIK_TABLE` and `LELAY_X` from `nielsen.py`, not the owner's
  copy `HJLPPZ_FILE` of the HJLPPZ triples. This is correct.
- T-215: `representation.rs:148` and `class_function.rs:211` are as the design describes. The two
  existing `ClassFunction` tests have 2 classes for order 2 and 1 class for order 1, so they pass
  through the "forced" case without changing an assertion.

## 2. Blocker

### B1. Le Lay's seven representatives are copied from his CC BY-NC-SA material

Problem. `vendor/janik_lelay_tables.json` (to be vendored as `tests/data/m23/`) contains
`lelay_x`, the seven involutions `x_1..x_7`. The owner's own comment in `m23golay/nielsen.py`
gives their source: "[LeLay26, Section 8.2.1 and the certificate
certify_geometric_reflection_obstruction.g of the accompanying repository]". I checked
Le Lay's repository at `3f6ab4b`: his CC0 `data/` holds no permutations (a scan of every JSON file
for image lists and cycle lists finds none). The seven involutions are in `verification/*.g`,
`*.m`, `*.py` and `notes/` (CC BY-NC-SA 4.0), for example
`verification/certify_geometric_reflection_obstruction.g`, list `representatives`. The brief says
his paper and code "must not be copied", and HANDOFF Section 1 says they "must not be copied into
GPL code". Taking them through the owner's transcription does not change where they come from.
Section 11 of the design then says "nothing of his is vendored by WP-P". That sentence is false.
(One of the seven, `x_4`, is also the first entry of the second triple of the unlicensed HJLPPZ
file. These are mathematical facts, but the project rule is explicit and the decision is not the
implementer's.)

Required change. Do not vendor `lelay_x`. Make the Le Lay dictionary an `env_` test, like HJLPPZ:
read `$LELAY_M23/verification/certify_geometric_reflection_obstruction.g` at run time (its `y` and
its `representatives` list, GAP cycle syntax), check that `y` is the published `g3`, map the seven
to `[0, 1, 2, 4, 6, 5, 3]` bijectively and check `lelay_id` of `group_facts.json`, and panic with
"set LELAY_M23 to a checkout of research-reflexivity/frontier-math-m23-cover-investigation"
without it. Record the file's sha256 in the README, as for HJLPPZ. This replaces follow-up WPP-5.
Vendor only Janik's table (the owner's own Table 1). Correct the sentence in Section 11 and the
reading table of Section 7. If the owner prefers to vendor the transcription, that needs his
explicit decision (a question like WPP-Q1), recorded before implementation, and the README must
name Le Lay's paper and certificate as the source.

## 3. Minor items

- **m1. Cost claims for the release tests are inconsistent.** Section 0 and Section 12 say "the
  opt-in release test as proposed runs in 7.9 s ... without the 35 s family check". Section 10.2
  lists the family check inside `full_census`, and also lists the 2.9 s sign check and
  `full_c_23a_23b`. The 7.9 s figure is `proto_release_streaming` alone (8.4 s here). State whether
  the family check is in `full_census` (1.2 says it is redundant; I would drop it or put it in its
  own `full_` test), and give a time and VmHWM per `full_` test as they will be.
- **m2. Where `|C(b)|` comes from in `count_class_vector`.** 5.1 says the identity uses RustMath's
  own `|C(b)|`, but the prototype divides by `s_cent`, a hard-coded copy of GAP's
  `SizesCentralizers`, and the API of 8.5 does not say how the function obtains it. Specify it (for
  example `centralizer_order_by_sifting` with the default budget: whenever the vector has a class
  other than 1A, 2A, 3A, `b` lies in one of 4A to 23B, whose `C_Sym(b)` has at most 294,912
  elements, so the sift always fits; otherwise `Err(BudgetExceeded)`), and say that the GAP value
  is used only as a test reference.
- **m3. The rational real count misses `a = c`.** For rational vectors the count runs over
  involutions `c` with `c b c = b^-1` and `a = c u` with `u` an involution. When the `a` class is 2A,
  the pair `(a, b)` with `a = c` (that is `u = 1`) is also inverted by `c` and is not counted. The
  owner's script has the same gap. It changes nothing for the 57 rows (only `(2A, 8A, 8A)` has 2A
  and nonnegative genus, and no involution inverts an element of order 8) or for negative genus (no
  generating pairs). But `count_class_vector` is public and takes any `ClassVector`. Count `c` itself
  when the `a` class is 2A, or state in the code why the case cannot contribute.
- **m4. The rational real criterion is cited but not yet checked.** 5.3 defers the exact statement
  to the implementer ("quotes the exact statement in the module header after checking it there").
  Pin it down in the design: the formula for complex conjugation on a standard generating system
  with three real branch points and the base point on the real arc, and the derivation that a class
  is fixed iff one involution inverts both `a` and `b` (it follows from the formula
  `(g1, g2) -> (g1^-1, g1 g2^-1 g1^-1)` with `k = h g1`). If the source cannot be read before
  implementation, label the rational `real` values as resting on the owner's criterion, and record
  the theorems used in `ClassVectorCount` (as the Steiner labelling records its trusted input).
- **m5. Two "measured" claims were not prototyped.** Column orthogonality (4.3) is checked by
  neither the prototype nor `check_table.py` (it holds; I checked it). `full_c_23a_23b` is said to
  compare `representatives_g1` (10.2), but the prototype compared sizes and `real` entries only.
  Either check them in the implementation or reword.
- **m6. Bit-mask limits of the F_2 layer.** `Span` uses `u128` vectors and `1u128 << idx`
  combination masks, with `idx` up to `3 dim` in the conjugation step. The standard modules stay at
  66 bits, but if `F2Module` or `TripleCohomology::new` accept a general module, a dimension above
  42 overflows the shift (a panic in debug, a wrong answer in release). Make the bound explicit:
  `Err` from every public constructor when `3 dim > 128`.
- **m7. T-215 `Character::inner_product` edge cases.** The design does not say what happens when
  only one side has class sizes or the two size maps differ, and the method returns `Complex`, so
  it cannot return `Err`. Specify it (for example: mismatched sizes panic with a message, or add a
  checked method returning `Result` and keep the old one documented as element-keyed).
- **m8. Fast-path measurement.** The 1.23 s "generic" figure is the naive odometer that rebuilds
  every prefix, not the incremental `StabilizerChain::for_each_element` the design proposes. The
  memory figure (73 MB against about 585 MB for the stored rational classes) already justifies the
  `[u8; 23]` path. Say so, or time the incremental generic version.
- **m9. `class_of` contract.** Section 9 says a split invariant with a third value gives
  `Err(InvariantFailed)`, but the fast 11 path cannot see a third value: it returns 11B for any
  signature-`(1,1,1)` permutation without 11-cycles. State that `InvariantFailed` comes only from
  `heptad_family_11` and `cyclic_steiner_sign`, and that `class_of` on a non-member is unspecified.
  Consider naming it `class_of_member` so the precondition shows at the call site. Similarly
  `NielsenClass23::new` takes a raw slice. Taking `&ClassLists` (elements known to be in the class)
  would remove a way to build a wrong Nielsen class silently.
- **m10. Readers for the example.** `examples/m23_class_vectors.rs --check <dir>` needs the JSON
  readers, which 8.6 places in `tests/m23_common/mod.rs` (test only). Say how the example gets them
  (`#[path]` include, or a small reader module in the library), so the parsing is not written twice.

## 4. Items checked and found sound

- Generators, order, base orbits `[23, 22, 21, 20, 3, 16]`, Steiner system by orbit and
  `is_steiner_system`, `G0` of dimension 11.
- The split-class rule and its proof (3.2), the naming conventions (3.3), the automorphism argument
  for the structure constants, and the braid argument that the owner's names do not matter (I also
  checked that the braid commutes with complex conjugation up to conjugation by `g1^-1`, so the
  real counts are name-independent too).
- Centralizer orders by sifting `C_Sym(x)` (sizes 294,912 to 23 as stated), by orbits for 2A and
  3A, by scan in release.
- The counting identity `#survivors = n |C(b)|` and the roles for rational rows.
- The conjugate-pair real criterion (fixed iff `a = c b c b^-1` for an involution `c`; `c` unique
  for a generating pair) and the source HJLPPZ 3.1 (also stated in the owner's paper, Section 2).
- Ovals, real structure and the cohomology formulas against the owner's `nielsen.py` and
  `cohomology.py` (the prototype's re-indexing of the combination masks is correct).
- One crate per PR (`rustmath-groups`, plus a dev-dependency), no cycle, no `unsafe`, data files
  tracked, table regenerated by a recorded command, GPL question for the table raised as WPP-Q1.
- `CountStatus::Decided` as the only success status with `Err` when the generation criterion does
  not apply: honest, since every number is an exact count.

## 5. Files

- `WPP/review1/rv_gap.g`, `rv_gap.out`: GAP checks, fusion, 3,000 labelled random elements,
  816 structure constants.
- `WPP/review1/py/`: `m23rule.py` (independent rule), `check_vs_gap.py`, `check_n.py`,
  `colorth.py`, `overlap.py`.
- `WPP/review1/proto/`: scratch crate (`Cargo.toml`, the prototype copy, `f2bench.rs`) and the run
  logs `run_streaming.log`, `run_nielsen.log`, `run_proto_debug_*.log`, `build_*.log`.
