# WP-P design: the three-point class vectors of M23

Date: 2026-11-16. Worktree `/home/user/RustMath-wpp`, branch `claude/m23-wpp`, base `main` `28c4b69`.
Brief: `docs/workstreams/m23_golay/HANDOFF.md`, Sections 1, 2, 4, 6 and 8 step 3.
Reference: the owner's repository `johnjanik/M23` at `2d11038` (its WP-P files are unchanged since
`d05475b`), `scripts/passports_all.py` (sha256 `e5d1b615...0c3`) and the `m23golay` modules.

The prototype, its logs and the GAP scripts are in
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/m23/wpp/design/WPP/`
(Section 15). Every number in this document marked "measured" comes from that prototype, run on
this machine in release (or debug where stated), one test thread. Numbers changed by Revision 1
come from the revised prototype in `WPP/rev1/`.

## Revision 1 (2026-11-16)

After review 1 (`WPP.review1.md`: one blocker B1 and ten minor items m1 to m10). The revised
prototype, its logs and the new vendoring script are in `WPP/rev1/` (Section 15). Every number
below marked "measured" was re-run there, release unless stated, one test thread, each `full_`
test in its own process.

| Issue | Change |
|---|---|
| **B1.** Le Lay's seven representatives were vendored through the owner's transcription, but they come from Le Lay's CC BY-NC-SA certificate; Section 11 said nothing of his was vendored. | Nothing of Le Lay is vendored. The Le Lay dictionary is an `env_` test, like HJLPPZ: `env_lelay_dictionary` reads `$LELAY_M23/verification/certify_geometric_reflection_obstruction.g` at run time (its `y` and its `representatives`, GAP cycle syntax), checks that `y` is the published `g3` and that each `x_k` is an involution of M23, maps the seven to `[0, 1, 2, 4, 6, 5, 3]` bijectively, checks `lelay_id` of `group_facts.json`, and panics without the variable. Measured: passes in 22 ms (debug); without `LELAY_M23` it fails with "set LELAY_M23 to a checkout of research-reflexivity/frontier-math-m23-cover-investigation". The README records the file's sha256 (`fad8d332...b0fe61` at `3f6ab4b`). Only Janik's table is vendored: it is the owner's own Table 1 ([Janik26], J. A. Janik), taken by a new `extract_janik_table.py` that reads `JANIK_TABLE` and nothing else. Sections 0, 7, 9, 10, 11, 13, 14, 15 corrected; follow-up WPP-5 is replaced by this test. |
| **m1.** Cost claims for the release tests were inconsistent. | The 11-family check left `full_census` (it is redundant by 3.2) and is its own test `full_family_11`. Time and VmHWM per `full_` test, as the tests will be (measured): `full_census` 8.9 s and 3 MB; `full_57_rows` 3.6 s and 78 MB; `full_c_23a_23b` 1.3 s and 77 MB; `full_family_11` 40.3 s and 3 MB (10.2, 12). |
| **m2.** Where `count_class_vector` gets `|C(b)|`. | From `ClassIdentifier::centralizer_order(g, b_class)`, cached per class: sifting `C_Sym(x)` within the default budget of `10^6` candidates (4A to 23B), else the conjugation orbit under `g1, g2` with a cap of `10^5` (1A, 2A, 3A: 1, 3,795 and 56,672 elements), so `|C(x)| = |G| / |orbit|`. This covers all 17 classes, so no class vector is refused for lack of `|C(b)|`. GAP's `SizesCentralizers` is a test reference only. The prototype now divides by the computed value (measured: 84 ms in release for the eight `b` classes of the 57 rows, 1.54 s in debug for all 17). Re-running found a prototype defect: the size of `C_Sym(x)` for 1A (23!) overflowed `u64` and panicked in debug; the design now requires saturating arithmetic there (3.4). |
| **m3.** The rational real count missed `a = c`. | The count also enumerates `u = 1` (that is `a = c`) when the `a` class is 2A. The design states why it never adds: `<c, b>` is then dihedral of order at most 16, so it is not transitive. Measured on the 21 rational multisets that contain 2A (any genus): `a = c` candidates occur (98 for `(2A,2A,2A)`; 30, 12, 5, 6 for `(2A,2A,X)` with `X` = 3A, 4A, 5A, 6A), none generating (5.3). |
| **m4.** The rational real criterion was cited but not checked. | 5.3 now derives it: base point on the real segment between the branch points `t1 < t2` of `a` and `b`; complex conjugation fixes the two real segments and reverses the small circles, so it sends `gamma_1, gamma_2` to their inverses; the conjugate cover has monodromy `phi o kappa`; an element inverting both `a` and `b` is an involution and is unique. The classical sources (Fried 1977; Krull, Neukirch 1971; Debes, Fried 1994) were not read here, so the derivation is written in `three_point.rs`, `ClassVectorCount` records the results it rests on (`trusted`), and the rational `real` values are labelled as resting on this derivation, which agrees with the owner's criterion. |
| **m5.** Two "measured" claims were not prototyped. | Exact column orthogonality is now checked by the prototype (51 ms, debug). `full_c_23a_23b` now compares `representatives_g1` (parsed as permutations) with the canonical representatives for all six `C`: equal (measured). |
| **m6.** Bit-mask limits of the F_2 layer. | Every public constructor of the F_2 layer returns `Err(F2Error::DimensionTooLarge)` when `3 dim > 128`, and `F2Span::add` returns `Err(F2Error::CapacityExceeded)` after 128 added vectors. The three standard modules (11, 11, 22) are far inside (6.3, 8.6, 9). |
| **m7.** T-215 `Character::inner_product` edge cases. | New `Character::try_inner_product(&self, other) -> Result<Complex, CharacterError>`: sizes on both sides with equal maps over the same keys gives the weighted product; sizes on neither side gives the old element-keyed product (documented as such); sizes on one side only, or different maps, is an `Err`. `inner_product` keeps its signature, calls it and panics with the error text. A panic needs the new sizes, so no existing caller changes (4.4). |
| **m8.** The fast path was measured against the naive odometer. | The incremental generic enumeration (`Vec<usize>`, one buffer per level) is measured: 0.27 s, against 0.15 to 0.19 s for the `u8` recursion and 1.15 to 1.29 s for the naive odometer; with labelling of every element 3.47 s against 3.17 s. Speed does not justify a second enumeration. So `M23::for_each_element` wraps `StabilizerChain::for_each_element` and converts each element to `[u8; 23]`. The `[u8; 23]` type (the existing `transitive23::Perm`) stays for the stored class lists and the inner loops, because of memory: 73 MB for the six rational classes against about 585 MB as `Vec<usize>`. All `full_` figures of this revision use this path (6, 8.2, 12). |
| **m9.** `class_of` contract. | Renamed `class_of_member`. `InvariantFailed` comes only from `heptad_family_11` and `cyclic_steiner_sign`; the result of `class_of_member` on a non-member is unspecified (documented; `class_of_checked` sifts first). `NielsenClass23::new` takes `&ClassLists`, whose lists are built by enumeration or orbit only (8.3, 8.6, 9). |
| **m10.** Readers for the example. | One reader module, `tests/m23_common/mod.rs` with `#![allow(dead_code)]`, used by the tests as `mod m23_common;` and by the example as `#[path = "../tests/m23_common/mod.rs"] mod m23_common;`. Checked in the scratch crate: the example builds, reads the 57 rows, and neither target warns (8.6, 8.7). |

Not changed (confirmed by the review): the class identification and its proof, the table import
and the structure constants, the counting identity, the generation criterion, the ovals and
cohomology formulas, the HJLPPZ handling, and one crate per pull request (`rustmath-groups`, with
`rustmath-coding` as a new dependency). Section 5.3 gains the derivation and the `u = 1` case;
no count changes. All 57 rows, the six `(C, 23A, 23B)` classes and the three dictionaries were
reproduced again by the revised prototype.

## 0. Decisions in one page

1. **Everything goes into `rustmath-groups`**, in one pull request: a public `bsgs` element
   enumeration, a new module `m23` (group, classes, ATLAS table, three-point counting, Nielsen
   classes of `(C, 23A, 23B)`, ovals, parabolic cohomology over `F_2`), and the T-215 fix of the two
   class-function inner products. New dependency: `rustmath-coding` (the two cyclic Golay codes,
   `cyclic_golay::heptads`). Already a dependency: `rustmath-combinatorics` (`is_steiner_system`),
   `rustmath-rationals` (`Rational`), `serde_json`. New dev-dependency: `rustmath-finitefields`
   (`PrimeField`, for one cross-check with `rustmath-matrix`). No cycle: the dependency closure of
   `rustmath-coding` does not contain `rustmath-groups` (checked with `cargo tree`).
2. **All 57 rows are reproduced by the prototype** (measured, release): `n`, `size`, `real` and
   `genus23` of every row of `data/m23_three_point_nielsen.json`, with `#survivors = n |C(b)|`
   asserted on every row and `|C(b)|` computed by RustMath. The opt-in test `full_57_rows` runs in
   3.6 s with 78 MB peak; `full_census` (labels, class sizes, centralizers by scan, sign = trace)
   in 8.9 s with 3 MB (Revision 1, m1).
3. **4.2 step 7 and 4.3 fit the same machinery and are in scope**: the six Nielsen classes
   `(C, 23A, 23B)` (6,127 triples), complex conjugation, real structures, ovals, `dim W = g`,
   `dim Jac[2] = 2g` on every triple and `dim Jac(X)(R)[2] = g + r - 1` on every real cover, all
   matched per index against `data/passports_C_23A_23B.json`, with the representatives
   `representatives_g1`: 0.58 s for the classes and cohomology, 1.3 s for the whole test
   `full_c_23a_23b` in release (measured).
4. **The three dictionaries hold** (measured): the seven HJLPPZ triples map to classes
   `[5, 4, 0, 2, 1, 3, 6]`, Janik's seven rows to `[3, 5, 1, 6, 2, 4, 0]`, Le Lay's seven
   representatives to `[0, 1, 2, 4, 6, 5, 3]`, each bijective and equal to `group_facts.json`.
   HJLPPZ and Le Lay are read at run time from `$M23ISGALOIS` and `$LELAY_M23` (neither is
   vendored); Janik's table, the owner's own, is vendored (Revision 1, B1).
5. **The ATLAS table of M23 is data**, generated by a GAP script (GAP 4.12.1, CTblLib 1.3.7) in an
   exact encoding `x + y b_d`. RustMath computes with it exactly in
   `K = Q(sqrt(-7), sqrt(-11), sqrt(-15), sqrt(-23))` (degree 16, `Rational` coordinates). The
   class order is matched to RustMath's classes by sizes, centralizer orders, all power maps and
   the permutation character; the remaining freedom is the table automorphism group `(Z/2)^4`,
   under which every structure constant is invariant.
6. **The fast path** (Revision 1, m8): one enumeration, the generic incremental
   `StabilizerChain::for_each_element` (measured 0.27 s for all elements, against 0.15 to 0.19 s
   for a `u8` recursion); `M23::for_each_element` converts each element to
   `transitive23::Perm = [u8; 23]` (an existing type). The `[u8; 23]` type is kept for the stored
   class lists and the inner loops because of memory: 73 MB for the six rational classes against
   about 585 MB as `Vec<usize>` (8 bytes per entry).
7. **F_2 linear algebra for 4.3 uses bit masks**, not `Matrix<PrimeField>`. Measured: one kernel
   of a 44 x 66 matrix over GF(2) with `Matrix<PrimeField>` takes 12 ms; the bit-mask version does
   all 6,127 triples times three modules in 0.36 s. `Matrix<PrimeField>` stays as an independent
   cross-check on the seven classes of `(2A, 23A, 23B)` in the fast suite.
8. **T-215** (`representation.rs:148`, `class_function.rs:211`): both inner products are fixed in
   place (weighted by class sizes), and the M23 table supplies a non-abelian test. The other T-215
   items are not used by WP-P and stay open (Section 14).

## 1. Scope

Mandatory (HANDOFF 4.2 steps 1 to 6 and the acceptance): M23 on the published generators; all
10,200,960 elements from the stabilizer chain; the 17 classes with the split pairs identified as in
4.2 step 3; class sizes and centralizer orders; the ATLAS table as data and `n` for all 57 class
vectors; the counting with `#survivors = n |C(b)|`; generation by transitivity; the real points; the
genus of the degree 23 cover; all 57 rows; the three dictionaries.

In scope after the measurements: 4.2 step 7 (ovals) and 4.3 (parabolic cohomology over `F_2`).

Out of scope: 4.4 (twists, small Hurwitz schemes, four branch points). Follow-ups in Section 14.

## 2. Algorithms and references

| Step | Algorithm | Reference |
|---|---|---|
| Group | Deterministic Schreier-Sims (existing `StabilizerChain`, `bsgs.rs:138`) | Sims 1970; Holt, Eick, O'Brien, *Handbook of Computational Group Theory* (2005), 4.4.2; MAGMA ch. 58 |
| Elements | Every `g` is uniquely `u_0 o u_1 o ... o u_5` with `u_i` in the level-`i` transversal (sifting in reverse) | HEO 4.4; owner `m23.py::enumerate_elements` |
| Steiner system | Orbit of `Fix(g1)` under `<g1, g2>`: 253 heptads, checked by `is_steiner_system(23, 4, .)` | Witt 1938; owner `m23.py::steiner` |
| Even Golay code `G0` | Span of `h0 + h` over the heptads `h`, reduced echelon with pivots `i_k` | MacWilliams, Sloane 1977, ch. 20; owner `golay.py::even_subcode` |
| Cycle types | `(Fix g, Fix g^2, Fix g^3)` separates the 12 cycle types | owner `m23.py::SIGNATURES` |
| Split classes | Golay trace (7, 14, 15, 23); heptad-position family (11); cyclic Steiner system (23) | owner `passports_all.py`, `m23.py::sign_of_23_element`; Le Lay, Section 5.4 (two cyclic systems) |
| Character table | ATLAS table from GAP CTblLib as data | Conway, Curtis, Norton, Parker, Wilson, *ATLAS of Finite Groups* (1985), M23 p. 71; GAP 4.12.1, CTblLib 1.3.7; MAGMA ch. 91 |
| Structure constant | `n = |C1||C2||C3| / |G|^2 * sum_chi chi(c1) chi(c2) chi(c3) / chi(1)` | Frobenius; Serre, *Topics in Galois Theory*, Section 7.2; owner `scripts/gap/m23_passports.g` |
| Counting | Fix `b` in `C_b`; survivors are the `a` with `a b` in the right class; `#survivors = n |C(b)|` | owner `passports_all.py` |
| Generation | A subgroup of M23 that contains an element of order not in `{1, 11, 23}` is M23 iff it is transitive on the 23 points | ATLAS p. 71 (maximal subgroups); checked by GAP `MaximalSubgroupClassReps(MathieuGroup(23))` (Section 5.2) |
| Real points | Complex conjugation on Nielsen classes (Section 5.3) | conjugate pairs: HJLPPZ arXiv:2608.08538v2, Section 3.1, and the owner's paper, Section 2; three real branch points: the derivation of 5.3 (Revision 1, m4), classically in Fried, Comm. Algebra 5 (1977), Krull, Neukirch, Math. Ann. 193 (1971), Debes, Fried, Pacific J. Math. 163 (1994) (not quoted here) |
| Genus | Riemann-Hurwitz: `2g - 2 = -46 + sum_i (23 - #cycles(c_i))` | |
| Ovals | Graph on `Fix(c0)` and `Fix(c0 g1)`; components of `X(R)` = cycles | owner's paper, Thm. "real structure" (ii) and Section 7; `nielsen.py::real_components_general` |
| Cohomology | `H^1(P^1, j_* M)` for `M = G0, P0/G0, P0` as parabolic cocycles mod coboundaries over `F_2` | owner's paper, Section 4; `cohomology.py`; MAGMA ch. 68 |

## 3. Class identification and why it is exact

### 3.1 The invariants

All on `g` as an image list in the right-action convention of Section 1 of the HANDOFF
(`x^(gh) = (x^g)^h`).

- **Cycle type**: `sig(g) = (Fix g, Fix g^2, Fix g^3)`, table of `m23.py::SIGNATURES`. A
  signature outside the table is an `Err` (not an element of M23).
- **Golay trace** (7AB, 14AB, 15AB, 23AB): with the reduced echelon basis `v_k` of `G0` and pivots
  `i_k`, `tr(g) = sum_k v_k[g^-1(i_k)]` in `F_2`. For bulk work the identifier uses
  `tr(g^-1) = sum_k v_k[g(i_k)]` (no inversion, 11 lookups) and compares with the value of the
  inverse of the class representative.
- **Heptad-position family** (11AB): label each 11-cycle of `g` by `Z/11` through `g`, record the
  positions of each heptad that meets it in at least 3 points; the multiset of these subsets of
  `Z/11` is the family (owner's `cyclic_design`). It does not depend on the starting points (the
  family is invariant under translation, because `g` preserves the Steiner system).
  Fast evaluation (measured): the 11A family has 33 translation classes of subsets, the 11B family
  33, and 11 classes occur only for 11A. `is_11A(g)` = "some position set of `g` lies in one of
  these 11 translation classes" (a table of 2048 entries). On 200,000 elements of order 11 the
  fast test equals the family test (measured: 0.28 s against 3.76 s).
- **Cyclic Steiner sign** (23AB, second invariant): label the points by `x0^(g^i) -> i` with
  `x0 = 0`; the transported system is shift invariant, hence `S+` or `S-`
  (`rustmath_coding::cyclic_golay::heptads(GolaySign::{Plus, Minus})`). `sign(g2) = -1`,
  `sign(g3) = +1`, as in `group_facts.json` (`signs`).

### 3.2 Why the identification is exact

For each split type `T` with representative `x`:

1. The invariant is a class function on M23. The trace is the trace of an `F_2[M23]`-module. The
   family and the sign are built from the Steiner system and the cycles of `g`, so conjugation by
   `h` in M23 (which preserves the Steiner system) transports them unchanged.
2. It takes different values on `x` and `x^-1` (asserted). So `x` and `x^-1` are not conjugate.
3. The elements of type `T` are exactly the two classes of `x` and `x^-1`: the number of elements
   of type `T` with the value of `x` equals `|G| / |C_G(x)|`, where `|C_G(x)|` is computed
   independently (Section 3.4). The class of `x` lies in that set and has that size, so the two
   sets are equal.

Step 3 needs the full enumeration and runs in the release test (measured: all 17 class sizes from
the labels and all 17 centralizer orders by scanning equal GAP's `SizesConjugacyClasses` and
`SizesCentralizers`). The fast suite checks 1 and 2 on the representatives, and the centralizer
orders by the cheaper methods of 3.4. For 23AB the release test also checks that the sign and the
trace agree on all 887,040 elements of order 23 (measured, 2.9 s): two independent invariants give
the same partition.

The fast 11 test is a function of the family (it asks whether the family contains a set from 11
given translation classes), so it is exact by the same argument: the family is constant on each
class, and the 11B family contains none of those sets. Its agreement with the family test on all
1,854,720 elements of order 11 is therefore redundant. It is kept as a separate opt-in test
`full_family_11` (measured 40.3 s, Revision 1, m1), not as part of `full_census`.

### 3.3 Naming conventions and the match with the ATLAS columns

Names inside a split pair are a convention (HANDOFF Section 1). RustMath fixes them as follows:

| Class | Definition | Golay trace (measured) |
|---|---|---|
| 23A | the class of the published `g2` (HJLPPZ) | `tr(g2) = 0`, `sign(g2) = -1` |
| 11A | the class of `h11`: `0^(g2^i) -> 0^(g2^(2i))`, an element of `N(<g2>) = 23:11` | (trace does not separate) |
| 7A | the class with `tr = 1` | 1 |
| 14A | the class whose squares lie in 7A (ATLAS power map `14a^2 = 7a`) | 1 |
| 15A | the class with `tr = 1` | 1 |

`h11` lies in M23 (sifted) and has order 11 (asserted). RustMath's 17 classes are then numbered
in ATLAS order `1A 2A 3A 4A 5A 6A 7A 7B 8A 11A 11B 14A 14B 15A 15B 23A 23B`, the column order of
GAP's `CharacterTable("M23")`. The match is checked (measured, all pass):

- element orders, class sizes, centralizer orders;
- the power maps `2, 3, 5, 7, 11, 23, -1` of GAP: for every class `c` and every `k`, the class of
  `rep(c)^k` is GAP's `PowerMap(t, k)[c]`;
- the permutation character on 23 points (`PermChars(t, 23)[1]`) against `Fix(rep(c))`.

These data do not fix the names inside a pair. GAP: `AutomorphismsOfTable(t)` has order 16,
generated by `(16,17)`, `(14,15)`, `(10,11)`, `(7,8)(12,13)`. Each generator is a Galois
automorphism of the table (it comes from `g -> g^k` with a suitable `k` prime to 212520), so
`n(C1, C2, C3) = n(C1^k, C2^k, C3^k)` for every triple of classes. Hence every structure constant
is the same under any naming that respects the power maps, and the only real choice is 23A, which
the published `g2` fixes.

The owner's names of 7AB, 11AB, 14AB and 15AB come from a random walk with Python's
`random.Random(23)` and are not portable. They do not matter for the 57 rows: for a row
`(R, D, D^-1)` the braid `(g1, g2, g3) -> (g1, g2 g3 g2^-1, g2)` is a bijection between the
generating Nielsen classes of `(R, D, D^-1)` and `(R, D^-1, D)`, and all 57 rows are reproduced
with RustMath's names (measured).

### 3.4 Centralizer orders

- **Release, exact for all 17**: count the `g` with `g x = x g` in one streamed pass over the
  enumeration (measured 2.3 s for the 17 representatives together; 2.8 s with the generic
  enumeration of Revision 1).
- **Debug, exact**: `|C_G(x)| = #{c in C_Sym(23)(x) : c in M23}`, enumerating
  `C_Sym(x) = prod_l (C_l wr S_{m_l})` and sifting each element. Sizes for M23's types: 4A 294,912,
  5A 90,000, 6A 10,368, 7AB 4,116, 8A 1,024, 11AB 242, 14AB 196, 15AB 225, 23AB 23. A budget
  (default `10^6` candidates) refuses 1A, 2A and 3A with `Err(BudgetExceeded)`. The size
  `prod_l m_l! l^(m_l)` is computed with saturating arithmetic: for 1A it is 23!, which does not
  fit in `u64` (the first prototype overflowed there in debug; Revision 1, m2). For 1A, 2A and 3A
  the class itself is the orbit of `x` under conjugation by `g1, g2` (1, 3,795 and 56,672
  elements), so `|C(x)| = |G| / |orbit|`. Measured debug: 1.41 s for the 14 sifted classes, 0.09 s
  for the two orbits; 1.54 s for all 17 through the combined method below.
- **The method the library uses** (Revision 1, m2): `ClassIdentifier::centralizer_order(g, c)`
  sifts `C_Sym(rep(c))` when it has at most `10^6` elements, and otherwise takes the conjugation
  orbit with a cap of `10^5` elements; the result is cached per class (`OnceLock`). Every class of
  M23 falls under one of the two, so the method never refuses an M23 class (an `Err` there would be
  a defect, not a budget decision). `Err(BudgetExceeded)` stays on the free functions
  `centralizer_order_by_sifting` and `conjugacy_class_by_orbit`, which take the caller's budget.
  `count_class_vector` uses the method; GAP's `SizesCentralizers` is only the test reference.

## 4. The character table

### 4.1 Data

`rustmath-groups/data/m23/m23_atlas_table.g` (new, ours) writes
`rustmath-groups/data/m23/m23_atlas_table.json` (3,471 bytes, sha256
`f9503e809aef25c369cf2b40c816bf72e23876d8162343cb75166e8d0f5196eb`):

```bash
gap -q rustmath-groups/data/m23/m23_atlas_table.g < /dev/null > rustmath-groups/data/m23/m23_atlas_table.json
```

Fields: `source` (`"GAP 4.12.1, CTblLib 1.3.7, CharacterTable(\"M23\")"`), `value_encoding`,
`class_names`, `orders`, `centralizer_orders`, `class_sizes`, `power_maps` (2, 3, 5, 7, 11, 23, -1),
`permutation_character_23`, `irreducibles` (17 rows of 17 values). A value is `[x, y, d]`, meaning
`x + y b_d` with the ATLAS irrationality `b_d = (-1 + sqrt(-d))/2`, `d` in `{7, 11, 15, 23}`, and
`[x, 0, 1]` for a rational value. The script reconstructs each value in GAP and stops with an error
if the encoding fails. It calls `SetPrintFormattingStatus("*stdout*", false)` so that GAP does not
wrap lines.

The library embeds the file with `include_str!` and parses it once (`OnceLock`).

### 4.2 Exact arithmetic

`AtlasNumber`: an element of `K = Q(s7, s11, s15, s23)`, `s_d^2 = -d`, as 16 `Rational`
coordinates on the basis `prod_{i in S} s_i`, `S` a subset of the four. Product:
`s^S s^T = s^(S xor T) prod_{i in S and T} (-d_i)`. Operations: `+`, `-`, `*`, scaling by a
`Rational`, complex conjugation (sign change on odd subsets), `rational_part() -> Option<Rational>`.
It is a field element type for this table, not a numeric wrapper: no lower crate has an exact
quadratic or cyclotomic type, and `rustmath-numberfields` depends on `rustmath-groups`.

### 4.3 Checks (all measured to pass)

- Row orthogonality, exact: `sum_c |c| chi_a(c) conj(chi_b(c)) = |G| delta_ab` (69 ms, debug).
- Column orthogonality, exact: `sum_chi chi(c) conj(chi(c')) = |C(c)| delta_cc'` (51 ms, debug;
  prototyped in Revision 1, m5).
- The class data of 3.3.
- Every structure constant is rational: `rational_part()` is `Some` (else `Err`).
- The 57 structure constants equal the owner's `data/m23_three_point_passports.json` (computed
  there by GAP from the same table) and the counting identity of 5.1 on every row.

### 4.4 T-215

The tracker rows: `representation.rs:148` `Character::inner_product` divides by the number of keys
and never weights by class size (broken); `class_function.rs:211` `ClassFunction::inner_product`
assumes all classes have size 1 (facade). WP-P does not need the f64 layer, but it builds a
character table, so it fixes both in place:

- `Character` gains `class_sizes: Option<HashMap<String, usize>>` and
  `Character::with_class_sizes(degree, values, sizes) -> Result<Self, CharacterError>`, which
  refuses a size map whose keys differ from the value keys or that has a size 0. The edge cases
  (Revision 1, m7) are decided by a new
  `Character::try_inner_product(&self, other) -> Result<Complex, CharacterError>`:

  | Sizes | Result |
  |---|---|
  | on both sides, equal maps (same keys, same sizes) | `Ok(sum |C| chi1(C) conj(chi2(C)) / sum |C|)` |
  | on neither side | `Ok` of the old element-keyed mean over the common keys (each key is one group element; documented as such) |
  | on one side only | `Err(CharacterError::SizesOnOneSide)` |
  | on both sides, different maps | `Err(CharacterError::SizeMismatch { key })` |

  `inner_product(&self, other) -> Complex` keeps its signature: it calls `try_inner_product` and
  panics with the error text on `Err`. Both `Err` cases need the new sizes, so no existing caller
  can reach the panic, and `is_irreducible` and `norm_squared` (a character with itself) never
  panic. `CharacterTable` constructors (`cyclic_group`, `symmetric_group_s3`) attach their class
  sizes, and a new `CharacterTable::inner_product(&self, a, b) -> Result<Complex, CharacterError>`
  uses the table's sizes and refuses a character without a value on some class. The tracker's
  failure case (S3 standard character, 1.667) then gives 1.
- `ClassFunction` gains `class_sizes: Option<HashMap<String, usize>>` (`with_class_sizes`).
  `inner_product` weights by them and returns `None` unless the sizes are known and sum to the group
  order, or the number of classes equals the group order (then every size is 1, forced). The two
  existing tests (`test_inner_product`: 2 classes, order 2; `test_norm`: 1 class, order 1) keep
  their assertions and pass through the forced case.
- New tests: S3 standard character irreducible; S3 trivial and sign orthogonal; the M23 table as an
  f64 `CharacterTable` (`AtlasTable::to_float_table`): the 17 irreducibles orthonormal to `1e-9`;
  a `ClassFunction` with sizes that do not sum to the order gives `None`; `try_inner_product`
  with sizes on one side, and with two different size maps, gives the two `Err` values;
  `inner_product` in the second case panics (`#[should_panic]`).

No existing assertion is weakened. No other T-215 item is touched (Section 14).

## 5. Counting, generation, real points

### 5.1 The class vectors and the counting

`enumerate_class_vectors(&AtlasTable)` reproduces the owner's GAP script
`scripts/gap/m23_passports.g`: rational classes are the classes with rational values on every
irreducible (2A, 3A, 4A, 5A, 6A, 8A; 1A excluded); pairs `(D, D^-1)` from the power map `-1`.
`rational` vectors are the multisets `R1 <= R2 <= R3`; `conjugate_pair` vectors are `(R, D, D^-1)`.
Keep those with genus `>= 0` and `n > 0`. Result: exactly the 57 class vectors of the reference
(checked as a set against both owner files).

Counting (owner's algorithm, step 4):

- `conjugate_pair` `(R, D, D^-1)`: `b = rep(D)`, `a` over the class `R`, survivor iff `a b` lies in
  `D` (then `(a b)^-1` lies in `D^-1`).
- `rational` `(R1, R2, R3)`: `b` in the class with the smallest centralizer; `a` over the smaller of
  the two other classes; survivor iff `a b` has the type of the third class (rational classes are
  closed under inversion, so the order of the roles does not matter). The prototype checked both
  role choices on all 27 rational rows (the owner's `a` in the middle class, and the smaller class):
  identical survivors, sizes and real counts.
- Identity: `#survivors = n |C(b)|`, with `n` from the table and `|C(b)|` from RustMath's own
  computation, `ClassIdentifier::centralizer_order(g, b_class)` (3.4; Revision 1, m2). The roles
  (which class is `b`, which is `a`) are chosen from the class sizes of the table, since any choice
  gives a correct count and the choice only affects speed. A failure is
  `Err(CountIdentityFailed { .. })`, never a silent mismatch. It holds on all 57 rows (measured,
  with the computed `|C(b)|`; the eight `b` classes of the 57 rows are 5A, 6A, 7A, 8A, 11A, 14A,
  15A, 23A, and their centralizers take 84 ms in release).

### 5.2 Generation

Theorem (ATLAS p. 71): the maximal subgroups of M23 are M22, L3(4):2, 2^4:A7, A8, M11,
2^4:(3 x A5):2 and 23:11, and all but 23:11 are intransitive on the 23 points. A transitive proper
subgroup therefore lies in a conjugate of 23:11, whose elements have order 1, 11 or 23. So a
subgroup that contains an element of order outside `{1, 11, 23}` is M23 if and only if it is
transitive. Checked by GAP (`scripts` in the design directory, `m23_checks.g`):
`MaximalSubgroupClassReps(MathieuGroup(23))` gives orders 443520, 40320, 40320, 20160, 7920,
5760, 253, and only the last is transitive.

Every one of the 57 vectors contains a rational class, of order 2 to 8, so the criterion applies.
`size = #(transitive survivors) / |C(b)|`; divisibility is asserted (`Err` otherwise). The
transitivity test is a bit-mask closure of `{0}` under `a` and `b` (at most 23 rounds).

Degenerate: a class vector whose three classes all have order in `{1, 11, 23}` is refused with
`Err(GenerationCriterionNotApplicable)` for `size` and `real` (the survivors and `n` are still
returned by `structure_count`). A bounded search would not decide generation, so there is no
fallback.

### 5.3 Real points

- `conjugate_pair`, branch points `0, s, conj(s)`: complex conjugation acts by
  `(g1, g2, g3) -> (g1^-1, g3^-1, g2^-1)` (HJLPPZ 3.1). A class with `g2 = b` is fixed iff there is
  `c` in M23 with `a = c b c b^-1` (then `c^2` centralizes a generating pair, so `c` is an
  involution). Real classes = `#{c in 2A : c b c b^-1 in R and <a, b> = M23} / |C(b)|`.
- `rational`, three real branch points (Revision 1, m4: the derivation, written out because the
  classical sources were not read here). Let the branch points be `t1 < t2 < t3` in `P^1(R)`
  (after a real Moebius map, which commutes with complex conjugation, all three are finite). Take
  the base point `p` in the open segment `(t1, t2)`. Let `gamma_1` run from `p` along the real
  segment to near `t1`, once around `t1` on a small circle, and back along the segment; `gamma_2`
  the same around `t2` through `(p, t2)`. Both are simple loops, they generate the free group
  `pi_1(P^1 - {t1, t2, t3}, p)`, and `gamma_3` is fixed by the product relation. Complex
  conjugation `kappa` fixes `p` and the two segments pointwise and reflects each small circle,
  which reverses its orientation, so `kappa(gamma_1) = gamma_1^-1` and `kappa(gamma_2) =
  gamma_2^-1` as loops, not only up to homotopy. If the cover has monodromy `phi` on the fibre
  over `p`, its complex conjugate has monodromy `phi o kappa_*` on the conjugate fibre (the base
  point is real). So conjugation sends the Nielsen class of `(a, b) = (phi(gamma_1),
  phi(gamma_2))` to that of `(a^-1, b^-1)`, whatever the left or right action convention (the
  inverses do not depend on it). The Nielsen classes here are up to conjugation in M23, and they
  are also the absolute ones because `N_{S_23}(M23) = M23` (Out(M23) = 1, ATLAS p. 71; Le Lay's
  certificate also checks `Normalizer(S_23, G) = G`). The class is fixed iff some `h` in M23
  inverts both `a` and `b`. Then `h^2` centralizes `<a, b> = M23`, whose centre is trivial, so
  `h^2 = 1`; and `h != 1`, since otherwise `a` and `b` would be involutions, and two involutions
  generate a dihedral group. So `h = c` is an
  involution, and it is unique: two of them, `c` and `c'`, would give `c c'` in the centralizer.
  Write `a = c u` with `u = c a`; then `u^2 = c a c a = a^-1 a = 1`, so `u` is 1 or an involution.
  The fixed pairs `(a, b)` with `b` fixed form a set stable under `C(b)`, which acts freely, so
  real classes = `#{(c, u) : c in 2A, c b c = b^-1, u in {1} u 2A, c u in A, c u b in C,
  <c u, b> = M23} / |C(b)|`, divisibility asserted. This is the owner's criterion
  (`passports_all.py`, step 6), except for `u = 1` (Revision 1, m3): the owner enumerates `u` over
  the involutions only, which misses `a = c` when the `a` class is 2A. That case never adds: then
  `<a, b> = <c, b>` is dihedral of order at most `2 ord(b) <= 16`, which is not transitive on 23
  points. The count enumerates it anyway, so that `real_involutions` counts the complete set, and
  the test asserts that it contributes nothing (measured: `a = c` candidates occur for
  `(2A,2A,2A)`, `(2A,2A,3A)`, `(2A,2A,4A)`, `(2A,2A,5A)`, `(2A,2A,6A)`, with 98, 30, 12, 5 and 6
  candidates, none generating; none of these vectors has genus `>= 0`). The classical statements
  of the action of complex conjugation on branch cycles are in Fried, Comm. Algebra 5 (1977),
  Krull and Neukirch, Math. Ann. 193 (1971), and Debes and Fried, Pacific J. Math. 163 (1994);
  this design does not quote them, so the module header gives the derivation above and
  `ClassVectorCount::trusted` names it (8.5).
- The count of real classes does not depend on which two branch points are adjacent to the base
  point: any two of three points on the circle `P^1(R)` bound an arc that avoids the third, so the
  derivation applies to any choice of the classes of `a` and `b` (the base point on that arc), and
  the number of fixed classes is a property of the Nielsen class. The prototype confirmed this on
  all 27 rational rows by changing the roles (5.1).
- Every class vector with 8A has `real = 0`: no involution inverts an element of order 8
  (owner's remark; reproduced, not asserted as a theorem).

`size - real` is even (conjugation is an involution on the classes): asserted.

### 5.4 Genus

`genus23 = (sum_i (23 - #cycles(rep(c_i))) - 46 + 2) / 2`; the numerator must be even (asserted).
Vectors of negative genus are dropped by the enumeration of 5.1.

## 6. The Nielsen classes `(C, 23A, 23B)`, ovals and cohomology

### 6.1 Nielsen classes

For a rational class `C`: candidates `a` in `C` with `a g2` in 23A (then `(a g2)^-1` in 23B). The
canonical representative is the lexicographically smallest `<g2>`-conjugate
`g2^-k a g2^k` (as a 0-based image list, the order of Python tuples). The classes are numbered by
the sorted canonical representatives, which is the owner's numbering (index `i` in
`passports_C_23A_23B.json` and `group_facts.json`). `C(g2) = <g2>` acts freely: the number of
candidates is 23 times the number of classes (asserted).

`index_of(triple)`: check the product is 1; find `h` in M23 with `h^-1 g2' h = g2` among the 23
maps that send the cycle of `g2'` to that of `g2` (each sifted); canonicalize `h^-1 g1 h`; binary
search. `Err` if the product is not 1, if no `h` lies in M23 (second entry not in 23A), or if the
canonical form is not in the class.

Every triple of `(C, 23A, 23B)` generates M23 (it contains an element of order 23 and one of order
`ord(C)` not in `{1, 11, 23}`; 5.2). The sizes are 7, 134, 660, 1422, 1632, 2272 (measured, equal to
the owner's and to `n`).

### 6.2 Real structure and ovals

Complex conjugation on indices: `index_of(g1^-1, g3^-1, g2^-1)`. For a fixed class the real
structure is the involution `c0` with `c0 g1 c0 = g1^-1`, `c0 g2 c0 = g3^-1`: `c0 = g2^k h` with
`h^-1 g2 h = g3^-1`, `k` in `0..23` (owner's `real_structure`). `None` for a fixed class is an
`Err` (it cannot happen).

Ovals (owner's `real_components_general`): `A = Fix(c0)`, `B = Fix(c0 g1)`; join `x` in `A` to
`x^(g2^-1)` in `B`; on each cycle of `g1`, pattern `(1, 1)` (odd length) joins the `A` point to the
`B` point, `(2, 0)` or `(0, 2)` (even length) joins the two points; every vertex has degree 2
(asserted, `Err` otherwise); components = cycles of the graph.

### 6.3 Parabolic cohomology over `F_2`

Modules (right action `(v.g)[x^g] = v[x]` on 23-bit masks): `G0` (dim 11), `Q = P0/G0` (dim 11),
`P0` (even-weight vectors, dim 22). A module is a subquotient with an ordered basis
(quotient basis first) and coordinates by reduction in an echelon span with combination masks
(owner's `f2.Span`). For a triple: cocycles `(c1, c2)` in `M^2`, `c3 = c1 (g2 g3) + c2 g3`;
coboundaries `(a(1 + g1), a(1 + g2))`; parabolic iff `c_i` lies in `M(1 + g_i)`; orbifold iff
`c_i N_i = 0` with `N_i = 1 + g_i + ... + g_i^(e_i - 1)`. Complex conjugation on parabolic classes:
`z'(gamma1) = c1 g1^-1 c0`, `z'(gamma2) = c3 g3^-1 c0`, expressed on a complement of `B`; it must be
an involution (asserted, `Err` otherwise); `real_part` = dimension of its fixed space.

Checks (measured, all pass): for every one of the 6,127 triples `dim H^1(G0) = dim H^1(Q) = g`,
`dim H^1(P0) = 2g`; for every real cover `dim_J2_real = g + r - 1`, and `components`,
`dim_J2_real`, `dim_W_real`, `real_points_above_0 = |Fix(c0) cap Fix(g1)|` equal the owner's list
entry by entry, index by index, for all six `C`. For `(2A, 23A, 23B)` also `dim_H1_orbifold_G0 = 7`
and the fields of `group_facts.json` `nielsen`.

Vectors of at most 66 bits: `u128` masks. Measured: Nielsen classes plus cohomology for all six
`C` in 0.58 s (release), 26 ms for the seven classes of `(2A, 23A, 23B)` (debug).

Bit-mask limits (Revision 1, m6). A cocycle packs `(c1, c2, c3)` into `3 dim` bits, and the
combination masks of `F2Span` use one bit per added vector, at most `3 dim` of them in the
conjugation step (`B` basis, then the parabolic basis). Both must fit in 128 bits, so every
public constructor (`F2Module::new`, `F2Module::subquotient`, `TripleCohomology::new`) returns
`Err(F2Error::DimensionTooLarge { dim })` when `3 dim > 128` (that is `dim > 42`), and
`F2Span::add` returns `Err(F2Error::CapacityExceeded)` once 128 vectors have been added. No shift
by 128 or more is ever executed, so there is no debug panic and no silent release result. The
standard modules (11, 11, 22) are far inside the bound.

`Matrix<PrimeField>` is too slow for the bulk (measured 12 ms per 44 x 66 kernel, release; at least
four such kernels and about 130 coordinate solves per triple, so over five minutes for 6,127
triples, projected). The bit-mask code is the specialised fast path; it lives with the modules it
serves (`rustmath-groups/src/m23/cohomology.rs`). The fast suite recomputes the three dimensions
of the rational class (index 5) with `Matrix<PrimeField>` ranks over GF(2) (`rank` and `kernel` of
`linear_solve.rs`) as an independent implementation.

## 7. The three dictionaries (acceptance of 4.2)

| Source | Reading | Result (measured) |
|---|---|---|
| HJLPPZ `seven_nielsen_class_representatives.m` (read at run time from `$M23ISGALOIS`) | Magma right action, `g3*g2*g1 = 1`; the file's fixed `g2` is the published `g3`; RustMath triple `(x, z, y)` = (file `g1`, file `g3`, file `g2`) | rows 1..7 -> classes `[5, 4, 0, 2, 1, 3, 6]`; `group_facts` `hjlppz_file_row` agrees |
| Janik's Table 1 ([Janik26], the owner's own manuscript; his transcription `nielsen.py` `JANIK_TABLE`, vendored as `janik_table.json`) | `g1` = published `g1`; every row satisfies `g1 g2 g3 = 1` with `g2` in 23A (the "direct" reading for all seven; checked with the owner's code) | rows 0..6 -> `[3, 5, 1, 6, 2, 4, 0]`; `janik_orbit` agrees |
| Le Lay's seven `x_k` (read at run time from `$LELAY_M23/verification/certify_geometric_reflection_obstruction.g`, assignments `y := ...;;` and `representatives := [ ... ];;`, GAP cycle syntax; Revision 1, B1) | `y` = published `g3` (23B, asserted), each `x_k` an involution in M23 (asserted), `z = (x y)^-1`, braided to `(x, y z y^-1, y)` | classes `[0, 1, 2, 4, 6, 5, 3]` (HANDOFF Section 7), `lelay_id` agrees; his real classes 3, 6, 7 are ours 2, 5, 3; his class 6 is our class 5, the class of the first HJLPPZ triple (both dictionaries give 5) |

Each is bijective onto the seven classes. RustMath implements only the readings that occur. The
owner's `janik_table_indices` also tries a braided and an inverted reading; none is used, so a row
that fails the direct reading is an `Err`, not a silent alternative.

Le Lay's representatives are not in his CC0 `data/` (checked at `3f6ab4b`: no permutations
there). They are in `verification/` and in his paper, both CC BY-NC-SA 4.0, which must not be
copied into RustMath. The owner's `LELAY_X` is a transcription of the same certificate, so taking
it does not change the source. RustMath therefore vendors nothing of Le Lay (Revision 1, B1):
`env_lelay_dictionary` reads his file at run time, like the HJLPPZ test. The reader is a small
GAP-syntax parser in `tests/m23_common/mod.rs`: it finds an assignment `name :=` at the start of a
line, takes the text up to the next `;;`, and reads permutations in cycle notation (a list is
split at commas outside cycles; whitespace and line breaks are ignored; points outside 1..23 or
repeated points are an `Err`). Measured on his file at `3f6ab4b`: `y` is the published `g3`, the
seven `x_k` give `[0, 1, 2, 4, 6, 5, 3]` (22 ms, debug). A scratch check (not part of RustMath)
found his seven equal to the owner's transcription. If the owner prefers to vendor the
transcription instead, that needs his explicit decision, recorded before implementation, and the
README would then name Le Lay's paper and certificate as the source (WPP-Q2, Section 14).

## 8. Public API

Module tree in `rustmath-groups` (each new file cites MAGMA ch. 58 and the papers in its header;
`cohomology.rs` ch. 68; `atlas.rs` ch. 91; `group.rs` also ch. 152 for the Golay code):

```
src/bsgs.rs                 (+ accessors and enumeration)
src/m23/mod.rs              re-exports, module docs, conventions
src/m23/group.rs            M23, published generators, transversals, enumeration, Steiner, G0
src/m23/classes.rs          M23Class, ClassIdentifier, invariants, centralizers
src/m23/atlas.rs            AtlasNumber, AtlasTable (data/m23/m23_atlas_table.json)
src/m23/three_point.rs      ClassVector, enumeration of the 57, counting, generation, real points
src/m23/nielsen.rs          NielsenClass23 for (C, 23A, 23B), conjugation, real structure, ovals
src/m23/cohomology.rs       F2Span, F2Module, TripleCohomology
src/representation.rs, src/class_function.rs   (T-215)
examples/m23_class_vectors.rs
```

### 8.1 `bsgs.rs`

```rust
impl StabilizerChain {
    pub fn num_levels(&self) -> usize;
    pub fn orbit(&self, level: usize) -> &[usize];                      // BFS order
    pub fn transversal_element(&self, level: usize, point: usize) -> Option<&[usize]>; // u(b_i) = point
    pub fn for_each_element(&self, f: impl FnMut(&[usize]));            // each element once
}
```

`for_each_element` composes `u_0 o ... o u_{k-1}` with one buffer per level and recomputes only
the levels below the one that changed. Measured on M23 (Revision 1, m8): 0.27 s in release for all
10,200,960 elements, against 1.15 to 1.29 s for the naive odometer that rebuilds every prefix and
0.15 to 0.19 s for a `[u8; 23]` recursion; the prototype checked that it yields the same elements
in the same order as the recursion. It is the only enumeration in the crate. Tests: S4 (24), A5
(60), M11 on 11 points (7,920): count equals `order()`, all distinct (`HashSet`), each `contains`.
Panics: none; level out of range is `None` / an empty slice.

### 8.2 `m23::group`

```rust
pub type Perm23 = crate::transitive23::Perm;           // [u8; 23], image list
pub fn rmul(p: &Perm23, q: &Perm23) -> Perm23;         // first p, then q (right action)
pub fn inverse(p: &Perm23) -> Perm23;
pub fn from_cycles_1based(c: &[&[u8]]) -> Result<Perm23, M23Error>;

pub struct M23 { .. }
impl M23 {
    /// <g1, g2> of HJLPPZ 3.1; Schreier-Sims order 10,200,960 and the Steiner system are verified.
    pub fn published() -> Result<M23, M23Error>;
    pub fn g1(&self) -> Perm23; pub fn g2(&self) -> Perm23; pub fn g3(&self) -> Perm23;
    pub fn chain(&self) -> &StabilizerChain;
    pub fn base_orbit_lengths(&self) -> Vec<usize>;    // [23, 22, 21, 20, 3, 16]
    pub fn contains(&self, p: &Perm23) -> bool;        // sifting
    pub fn for_each_element(&self, f: impl FnMut(&Perm23));   // chain().for_each_element, converted
    pub fn heptads(&self) -> &[u32];                   // 253 masks, sorted
    pub fn even_golay_basis(&self) -> &[(u32, u8)];    // (v_k, pivot i_k), 11 rows
}
```

`M23::for_each_element` calls `StabilizerChain::for_each_element` and converts each image list to
`[u8; 23]` in one reused buffer before calling `f` (Revision 1, m8: measured 3.47 s with the
labelling of every element, against 3.17 s for a separate `u8` enumeration, so a second
enumeration is not worth its code). The `[u8; 23]` type serves the stored class lists (73 MB for
the six rational classes, about 585 MB as `Vec<usize>`) and the inner loops (`rmul`, the
transitivity masks).

`M23Error`: `NotAPermutation`, `NotInM23`, `Construction(String)` (a verification at
construction failed; a defect, never expected), `BudgetExceeded { needed, budget }`,
`InvariantFailed(String)` (a split invariant took a third value; only `heptad_family_11` and
`cyclic_steiner_sign` return it, Revision 1, m9), `Io(String)`, `Parse(String)`.

### 8.3 `m23::classes`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum M23Class { C1A, C2A, C3A, C4A, C5A, C6A, C7A, C7B, C8A, C11A, C11B, C14A, C14B, C15A, C15B, C23A, C23B }
impl M23Class {
    pub const ALL: [M23Class; 17];                    // ATLAS (GAP column) order
    pub fn index(self) -> usize; pub fn name(self) -> &'static str; pub fn from_name(&str) -> Option<Self>;
    pub fn element_order(self) -> u32; pub fn cycle_type(self) -> &'static [u8];
    pub fn inverse(self) -> M23Class; pub fn is_rational(self) -> bool;
}
pub struct ClassIdentifier { .. }                     // built from &M23; conventions of 3.3
impl ClassIdentifier {
    pub fn new(g: &M23) -> Result<Self, M23Error>;
    /// Precondition: p lies in M23. On a non-member the result is unspecified (it may be Ok with
    /// a wrong class); use class_of_checked. Err(NotInM23) only for a signature outside the 12.
    pub fn class_of_member(&self, p: &Perm23) -> Result<M23Class, M23Error>;
    pub fn class_of_checked(&self, g: &M23, p: &Perm23) -> Result<M23Class, M23Error>; // sifts first
    /// |C_M23(rep(c))|: sifting C_Sym within 10^6, else the orbit within 10^5; cached (3.4).
    pub fn centralizer_order(&self, g: &M23, c: M23Class) -> Result<u64, M23Error>;
    pub fn representative(&self, c: M23Class) -> Perm23;
    pub fn golay_trace(&self, p: &Perm23) -> u8;
    pub fn heptad_family_11(&self, p: &Perm23) -> Result<Vec<(u16, u32)>, M23Error>;
    pub fn cyclic_steiner_sign(&self, p: &Perm23) -> Result<GolaySign, M23Error>;
}
pub fn centralizer_order_by_sifting(g: &M23, x: &Perm23, budget: u64) -> Result<u64, M23Error>;
pub fn conjugacy_class_by_orbit(g: &M23, x: &Perm23, cap: usize) -> Result<Vec<Perm23>, M23Error>;
pub struct Census { pub class_sizes: [u64; 17], pub centralizer_orders: [u64; 17], pub sign_agrees_on_23: u64 }
pub fn census(g: &M23, id: &ClassIdentifier) -> Result<Census, M23Error>;   // full enumeration (release)
pub struct ClassLists { .. }  // elements of chosen classes; built only by the two constructors
impl ClassLists {
    pub fn by_enumeration(g: &M23, id: &ClassIdentifier, classes: &[M23Class]) -> Result<Self, M23Error>;
    pub fn by_orbit(g: &M23, id: &ClassIdentifier, classes: &[M23Class], cap: usize) -> Result<Self, M23Error>;
    pub fn get(&self, c: M23Class) -> Option<&[Perm23]>;
}
```

`class_of_member` of a permutation outside M23 is unspecified: it does not sift, so it can return
a wrong class (for example an involution of type `2^8 1^7` outside M23 is called 2A, and a 22-cycle,
which has the signature `(1, 1, 1)` of the 11-cycles, is called 11B by the fast 11 test; both found
by the review). The name shows the precondition at the call site, the docs state it, and
`class_of_checked` sifts first. A signature outside the 12 types is `Err(NotInM23)`. The fast 11
test cannot see a third value, so `class_of_member` never returns `InvariantFailed`; that error
comes only from `heptad_family_11` and `cyclic_steiner_sign`, which compare with both class values
(Revision 1, m9).

### 8.4 `m23::atlas`

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtlasNumber { /* [Rational; 16] */ }
impl AtlasNumber { pub fn from_atlas(x: i64, y: i64, d: u32) -> Result<Self, String>; pub fn conj(&self) -> Self;
                   pub fn rational_part(&self) -> Option<Rational>; /* Add, Sub, Mul, scale */ }
pub struct AtlasTable { .. }
impl AtlasTable {
    pub fn m23() -> &'static AtlasTable;                       // embedded data, parsed once
    pub fn source(&self) -> &str;
    pub fn value(&self, chi: usize, c: M23Class) -> &AtlasNumber;
    pub fn class_size(&self, c: M23Class) -> u64; pub fn centralizer_order(&self, c: M23Class) -> u64;
    pub fn power_map(&self, k: i64, c: M23Class) -> Option<M23Class>;
    pub fn structure_constant(&self, c: [M23Class; 3]) -> Result<Rational, String>;   // Err if not rational
    pub fn check_orthogonality(&self) -> Result<(), String>;   // rows and columns, exact
    pub fn check_against(&self, g: &M23, id: &ClassIdentifier) -> Result<(), String>;  // 3.3 checks
    pub fn to_float_table(&self) -> crate::representation::CharacterTable;            // T-215 tests
}
```

### 8.5 `m23::three_point`

```rust
pub enum VectorKind { Rational, ConjugatePair }
pub struct ClassVector { pub kind: VectorKind, pub classes: [M23Class; 3] }
pub fn enumerate_class_vectors(t: &AtlasTable, id: &ClassIdentifier) -> Vec<ClassVector>;   // the 57
pub fn genus23(id: &ClassIdentifier, v: &ClassVector) -> Result<i64, M23Error>;
pub enum CountStatus {
    /// Exact count; generation by transitivity (ATLAS maximal subgroups, 5.2).
    Decided,
}
/// The results a count rests on besides exact enumeration (Revision 1, m4).
pub enum TrustedResult {
    /// ATLAS p. 71: every maximal subgroup except 23:11 is intransitive (5.2).
    AtlasMaximalSubgroups,
    /// HJLPPZ 3.1: conjugation on (R, D, D^-1) at 0, s, conj(s) is (g1^-1, g3^-1, g2^-1).
    ConjugationConjugatePair,
    /// Three real branch points: conjugation sends (a, b) to (a^-1, b^-1); derivation in the
    /// header of three_point.rs (5.3), classical sources not quoted.
    ConjugationThreeRealPoints,
}
pub struct ClassVectorCount {
    pub vector: ClassVector, pub genus23: i64, pub structure_constant: Rational,
    pub b_class: M23Class, pub a_class: M23Class,
    /// |C(b)| from ClassIdentifier::centralizer_order (sifting or orbit), not from the table.
    pub centralizer_order: u64,
    pub survivors: u64, pub generating: u64, pub size: u64,
    /// Rational rows: pairs (c, u) with u in {1} u 2A (u = 1 included, Revision 1, m3);
    /// conjugate pairs: involutions c. Only generating pairs are counted.
    pub real_involutions: u64, pub real: u64,
    pub status: CountStatus,
    pub trusted: Vec<TrustedResult>,   // [AtlasMaximalSubgroups, ConjugationConjugatePair or ...ThreeRealPoints]
}
/// |C(b)| comes from id.centralizer_order(g, b_class) (cached); the lists must hold the a class
/// and, for the real count, 2A.
pub fn count_class_vector(g: &M23, id: &ClassIdentifier, t: &AtlasTable, lists: &ClassLists,
                          v: &ClassVector) -> Result<ClassVectorCount, ThreePointError>;
pub enum ThreePointError {
    ClassListMissing(M23Class), CountIdentityFailed { survivors: u64, n: Rational, centralizer: u64 },
    NotDivisible(&'static str), GenerationCriterionNotApplicable, NotRationalVector, Group(M23Error),
}
```

`Decided` is the only success status: every reported number is an exact count and generation is
decided by the theorem. When the criterion does not apply the call is an `Err`, never a size. The
`real` value of a `Rational` vector also rests on `ConjugationThreeRealPoints`, which this design
derives but whose classical source it does not quote; `trusted` says so on every count, and the
example prints it (Revision 1, m4).

### 8.6 `m23::nielsen` and `m23::cohomology`

```rust
pub type Triple = (Perm23, Perm23, Perm23);
pub struct NielsenClass23 { .. }                      // Ni(C, 23A, 23B) with g2 normalized
impl NielsenClass23 {
    /// The candidates come from lists.get(c) (Revision 1, m9: no raw slice, so a list of
    /// elements outside the class cannot be passed); Err(ClassListMissing) without it.
    pub fn new(g: &M23, id: &ClassIdentifier, lists: &ClassLists, c: M23Class) -> Result<Self, NielsenError>;
    pub fn len(&self) -> usize; pub fn triple(&self, i: usize) -> Triple;
    pub fn index_of(&self, t: &Triple) -> Result<usize, NielsenError>;
    pub fn complex_conjugation(&self) -> Result<Vec<usize>, NielsenError>;
    pub fn real_structure(&self, i: usize) -> Result<Option<Perm23>, NielsenError>;
    pub fn real_components(&self, i: usize, c0: &Perm23) -> Result<usize, NielsenError>;
}
pub enum NielsenError { ProductNotOne, SecondEntryNotIn23A, NotInClass, ClassListMissing(M23Class),
                        RealStructureMissing(usize), OvalGraph(String), Group(M23Error) }

pub enum F2Error { DimensionTooLarge { dim: usize }, CapacityExceeded, NotInModule,
                   CoboundaryNotCocycle, NotAnInvolution }
pub struct F2Span { .. }     // u128 vectors, echelon with combination masks
impl F2Span { pub fn add(&mut self, v: u128) -> Result<bool, F2Error>; /* reduce, solve, dim */ }
pub struct F2Module { .. }   // subquotient of F_2^23 with the right permutation action
impl F2Module {
    /// Err(DimensionTooLarge) when 3 dim > 128 (Revision 1, m6).
    pub fn subquotient(sub: &[u32], quo: &[u32]) -> Result<Self, F2Error>;
}
pub fn standard_modules(g: &M23) -> Result<[F2Module; 3], F2Error>;   // G0, Q = P0/G0, P0
pub struct TripleCohomology<'a> { .. }
impl<'a> TripleCohomology<'a> {
    pub fn new(m: &'a F2Module, t: &Triple) -> Result<Self, F2Error>;   // Err if B is not in Z, or 3 dim > 128
    pub fn dim_parabolic(&self) -> usize; pub fn dim_orbifold(&self) -> usize;
    pub fn conjugation_fixed_dim(&self, c0: &Perm23) -> Result<usize, F2Error>;  // Err if not an involution
}
```

Readers (Revision 1, m10): one module, `tests/m23_common/mod.rs`, with `#![allow(dead_code)]`
(each target uses part of it). It holds the readers of the four owner JSON files and of the
vendored `janik_table.json`, `parse_magma_perms` for the HJLPPZ file (`S!(...)` groups), and the
GAP-syntax reader for Le Lay's certificate (Section 7). The tests use it as `mod m23_common;`. The
example includes the same file with `#[path = "../tests/m23_common/mod.rs"] mod m23_common;`, so
the parsing is written once and stays out of the library. Checked in the scratch crate: a test and
an example that share the module both build and read the 57 rows, without warnings.

### 8.7 Example

`cargo run --release -p rustmath-groups --example m23_class_vectors [--check <dir>]` prints the 57
rows (type, classes, genus, `n`, survivors, `|C(b)|`, size, real, status, trusted results, time).
With `--check` it reads `<dir>/m23_three_point_nielsen.json` through `m23_common` (8.6) and adds
`reproduced` or `MISMATCH` per row.

## 9. Degenerate cases

| Case | Behaviour |
|---|---|
| `(2A, 7A, 7B)`: 301 survivors, `n = 43/2`, none generating | `size = 0`, `real = 0`, `Decided`; kept in the table |
| Fractional `n` (17 rows) | the identity is checked as `Rational(survivors) / |C(b)| == n` |
| Vectors with 8A | `real = 0` from the count (no involution inverts an element of order 8) |
| Repeated rational class, e.g. `(4A, 4A, 5A)` | roles as in 5.1; tested |
| All three orders in `{1, 11, 23}` | `Err(GenerationCriterionNotApplicable)` |
| Negative genus, `n = 0` | not enumerated (5.1); `count_class_vector` still works and returns 0 survivors |
| A non-member passed to `class_of_member` | unspecified, may misclassify (documented; Revision 1, m9); `class_of_checked` returns `Err(NotInM23)` |
| Signature outside the 12 types | `Err(NotInM23)` |
| Split invariant with a third value | `Err(InvariantFailed)` from `heptad_family_11` or `cyclic_steiner_sign` only (impossible in M23 by 3.2; checked) |
| Centralizer search above budget (1A, 2A, 3A) | `centralizer_order_by_sifting` gives `Err(BudgetExceeded)` (the size `23!` of `C_Sym` for 1A saturates, no overflow); `ClassIdentifier::centralizer_order` then uses the orbit, so every class gets its order (Revision 1, m2) |
| Rational vector with `a` class 2A, `a = c` (`u = 1`) | enumerated; never generating (`<c, b>` dihedral of order at most 16), asserted (Revision 1, m3) |
| Class list not built | `Err(ClassListMissing)` |
| `#survivors != n |C(b)|`, or generating / real counts not divisible by `|C(b)|`, or `size - real` odd | `Err`, never a reported row |
| Triple with product not 1, second entry not in 23A, or not in the class | `NielsenError` |
| A fixed class without a real structure; an oval graph vertex of degree not 2 | `Err` |
| Cohomology: `B` not in `Z`; conjugation not an involution | `Err` |
| F_2 module or triple with `3 dim > 128`; a span with 128 vectors | `Err(F2Error::DimensionTooLarge)`, `Err(F2Error::CapacityExceeded)`; no shift by 128 or more (Revision 1, m6) |
| `Character` sizes on one side only, or two different size maps | `try_inner_product` gives `Err`; `inner_product` panics with the message (Revision 1, m7) |
| Janik row that fails the direct reading | test failure with the row number (no alternative reading) |
| `$M23ISGALOIS` unset | the `env_` test panics with "set M23ISGALOIS to a checkout of shaowuz/m23isgalois" |
| `$LELAY_M23` unset | `env_lelay_dictionary` panics with "set LELAY_M23 to a checkout of research-reflexivity/frontier-math-m23-cover-investigation" (Revision 1, B1; checked) |
| Le Lay's file without `y :=` or `representatives :=`, malformed cycles, points outside 1..23 or repeated | `Err` from the GAP reader; the test fails with the path (negative cases tested in debug) |
| Empty or malformed data file | `Err(Parse)` in the readers; the test fails with the path |

## 10. Tests

Times are measured with the prototype (same algorithms), unless marked as estimates.

### 10.1 Fast (debug, normal suite)

`rustmath-groups/tests/m23_classes.rs`, `m23_atlas.rs`, `m23_three_point.rs`, `m23_nielsen.rs`:

| Test | Checks | Reference | Debug time |
|---|---|---|---|
| `m23_published_order_and_base` | order 10,200,960; orbit lengths `[23, 22, 21, 20, 3, 16]`; `g1 g2 g3 = 1` | owner `m23.py` `ORDER`, HANDOFF 4.2 step 1; GAP `Size(Group(g1, g2))` | 5 ms (setup) |
| `steiner_and_even_golay_code` | 253 heptads, `is_steiner_system`; `dim G0 = 11`; weights 8, 12, 16 with 506, 1288, 253 words | owner `verify_group_theory.py` | < 0.1 s (est.) |
| `split_invariants_separate` | trace on 7, 14, 15, 23; family on 11; `sign(g2) = -1`, `sign(g3) = +1`; conventions of 3.3 | `group_facts.json` `signs` | in setup |
| `class_data_match_gap` | power maps, permutation character, orders | GAP table (embedded JSON) | 0.07 s |
| `centralizers_match_gap` | `ClassIdentifier::centralizer_order` for all 17: 1A, 2A, 3A by orbits (1, 3,795, 56,672); 4A..23B by sifting | GAP `SizesCentralizers` | 1.54 s (Revision 1) |
| `atlas_orthogonality_and_57_constants` | exact row and column orthogonality; 57 `n` | owner `m23_three_point_passports.json` (GAP) | 0.12 s for the two orthogonalities (Revision 1, m5); 57 `n` < 0.1 s (est.) |
| `class_vector_enumeration_is_the_57` | the set of vectors | both owner files | < 0.1 s (est.) |
| `eighteen_small_rows` | the 18 rows whose `a` class is 2A or 3A: identity with the computed `|C(b)|`, size, real, genus, `size - real` even, `a = c` adds nothing; incl. `(2A,23A,23B)` 7/3, `(2A,14A,14B)` 17/3, `(2A,7A,7B)` 0/0, `(3A,5A,6A)` 70/4 | owner `m23_three_point_nielsen.json` | 1.07 s (Revision 1: 0.46 s more than before, for sifting the `b` centralizers) |
| `nielsen_2a_23a_23b_and_janik` | 7 classes, conjugation `[1,0,2,3,6,5,4]`, dims 4/4/8/7, real data, Janik's map `[3, 5, 1, 6, 2, 4, 0]` | `group_facts.json` `nielsen`; vendored `janik_table.json` | 17 ms |
| `gap_reader_rejects` | the GAP reader of Section 7: points outside 1..23, repeated points, nested or unbalanced cycles, foreign characters, a missing assignment are `Err`; a two-line list and a broken cycle parse | none (behaviour) | < 1 ms |
| `cohomology_cross_check_matrix_gf2` | dims of class 5 with `Matrix<PrimeField>` ranks | same | 1 to 3 s (est., from 71 ms per 44 x 66 kernel and 44 ms per rank in debug, measured by review 1) |
| `degenerate_cases` | the `Err` rows of Section 9 that are cheap | none (behaviour) | < 0.1 s (est.) |
| T-215 tests (`representation.rs`, `class_function.rs` unit tests, `m23_atlas.rs`) | 4.4 | S3 by hand; GAP M23 | < 0.1 s (est.) |

Total fast debug time: about 5 s (measured parts 3.1 s, plus the estimates; each test file
builds its own `M23` and identifier, 5 ms each). Measured with the revised prototype: the parts of
the first five rows together 2.04 s, the 18 rows 1.07 s, the seven classes 17 ms.

### 10.2 Opt-in release (`#[ignore]`, prefix `full_`)

```bash
cargo test --release -p rustmath-groups --test m23_classes -- --ignored full_
cargo test --release -p rustmath-groups --test m23_three_point -- --ignored full_
cargo test --release -p rustmath-groups --test m23_nielsen -- --ignored full_
```

Each `full_` test builds its own `M23` and streams the enumeration itself, so the tests are
independent. Each prints its time and its `VmHWM` (from `/proc/self/status`). `VmHWM` is the peak of
the process, so the per-test figure is exact only when the test runs alone: the acceptance runs
each with `--exact <name>` (Section 13). Times and peaks as the tests will be (Revision 1, m1;
measured with the revised prototype, release, one thread, each test in its own process, generic
enumeration of 8.1):

| Test (file) | Checks | Release time | VmHWM |
|---|---|---|---|
| `full_census` (`m23_classes.rs`) | labels of all 10,200,960 elements (fast 11 test): 17 class sizes, and in the same pass sign = trace on all 887,040 elements of order 23; then 17 centralizers by scan in a second pass | 8.9 s (6.1 s first pass, 2.8 s second) | 3 MB |
| `full_family_11` (`m23_classes.rs`) | fast 11 test = family test on all 1,854,720 elements of order 11 (927,360 in 11A); redundant by 3.2 | 40.3 s | 3 MB |
| `full_57_rows` (`m23_three_point.rs`) | the six rational class lists (0.80 s); `|C(b)|` for the eight `b` classes by `centralizer_order`, equal to GAP's (84 ms); all 57 rows of `m23_three_point_nielsen.json` exactly, with the identity, divisibility and parity on each (2.5 s, slowest row `(8A, 11A, 11B)` 0.45 s); the `a = c` case on the 21 rational multisets containing 2A (0.17 s) | 3.6 s | 78 MB |
| `full_c_23a_23b` (`m23_nielsen.rs`) | the six rational class lists; the six classes against `passports_C_23A_23B.json` entry by entry: sizes, `real_classes`, the `real` entries, and `representatives_g1` (parsed as permutations, equal to the sorted canonical representatives, Revision 1, m5); `dim W = g`, `dim W^dual = g`, `dim Jac[2] = 2g` on all 6,127 triples; `g + r - 1` on every real cover | 1.3 s | 77 MB |

Total for the three commands: about 54 s, 14 s without `full_family_11` (run
`-- --ignored full_ --skip full_family_11` to leave it out). The stored variant of the HANDOFF
(all elements, 234 MB) peaks at 297 MB (measured) and is not needed.

### 10.3 Environment tests

```bash
M23ISGALOIS=<checkout of shaowuz/m23isgalois> \
LELAY_M23=<checkout of research-reflexivity/frontier-math-m23-cover-investigation> \
  cargo test -p rustmath-groups --test m23_nielsen -- --ignored env_
```

`env_hjlppz_dictionary`: reads `seven_nielsen_class_representatives.m` at run time; checks that the
fixed `g2` of the file is the published `g3` and maps the seven rows to `[5, 4, 0, 2, 1, 3, 6]`
(bijective, equal to `hjlppz_file_row`). Debug time measured 33 ms (Revision 1). The earlier
check "Le Lay's class 6 is the first HJLPPZ triple" used the vendored Le Lay data and is dropped:
at the level of classes it follows from the two dictionaries (both give class 5), and a test that
needs both checkouts would add nothing to that.

`env_lelay_dictionary` (Revision 1, B1): reads
`verification/certify_geometric_reflection_obstruction.g` at run time (Section 7); checks that `y`
is the published `g3` and each `x_k` an involution of M23; maps the seven to
`[0, 1, 2, 4, 6, 5, 3]` (bijective, equal to `lelay_id`); checks that his real classes 3, 6, 7 are
the three fixed classes 2, 5, 3. Debug time measured 22 ms.

Both tests are `#[ignore]` (opt-in), so the normal suite does not need the checkouts. Asked for
without its variable, each fails loudly with the message of Section 9 (both checked).

## 11. Data files and licences

| Path | Source | Licence | sha256 |
|---|---|---|---|
| `rustmath-groups/data/m23/m23_atlas_table.json` | our GAP script (next row), GAP 4.12.1, CTblLib 1.3.7 | the ATLAS table of M23 (a mathematical fact) in our encoding; CTblLib is GPL-3.0-or-later, RustMath GPL-2.0-or-later (see note) | `f9503e80...0f5196eb` |
| `rustmath-groups/data/m23/m23_atlas_table.g` | new | RustMath | |
| `rustmath-groups/tests/data/m23/m23_three_point_nielsen.json` | owner, `data/`, commit `2d11038` (unchanged since `d05475b`), unchanged | owner's data, vendored as the brief allows | `b8189758...26ff0ba5` (7,605 bytes) |
| `.../m23_three_point_passports.json` | same | same | `c67ad1b5...c26f711f` (7,960 bytes) |
| `.../group_facts.json` | same | same | `cd81e9c592fff09b7dd62f7232a58f0b6dba90dd4c3f903b10cfad65157b6e48` (6,670 bytes) |
| `.../passports_C_23A_23B.json` | same | same | `b163d95b5ba738b56a09bb9addabf1a036ba3590f3117ce70eae9bc879e8a408` (436,351 bytes) |
| `.../janik_table.json` (Revision 1, B1: replaces `janik_lelay_tables.json`) | derived from the owner's `m23golay/nielsen.py` at `2d11038` (sha256 `86986ec0a0aa2eac5268adbe81daaba9586a030cff6bc745619eebccb4989bec`) by `extract_janik_table.py`; only `JANIK_TABLE` | the owner's transcription of Table 1 of his own manuscript [Janik26] (J. A. Janik, "Real Structure and the Galois Fixed Point in the Nielsen Class (2,23A,23B) of M23", 2026); owner's data | `3db9d440f978f3ec6c190f66903804bdb6ec80043b0d3d4079847e072fa42dac` (1,599 bytes) |
| `.../extract_janik_table.py` | new (reads the source with `ast`, never imports or runs it; reads `JANIK_TABLE` only, not `LELAY_X` or `HJLPPZ_FILE`) | RustMath | |
| `.../README.md` | new: the table above, the commands, the licence notes, and the sha256 of the two files read at run time (below) | | |

Full hashes: `m23_three_point_nielsen.json`
`b8189758a483d88beb46668ca67c3927fe3ad1738a84832105f0737c26ff0ba5`;
`m23_three_point_passports.json`
`c67ad1b51960a30d0b4362651bcafb7cc6b16b499498c2290f3965afce16711f`; `m23_atlas_table.json`
`f9503e809aef25c369cf2b40c816bf72e23876d8162343cb75166e8d0f5196eb`.

Not vendored, read at run time only (Revision 1, B1):

| File | Licence | Variable | Commit | sha256 (recorded in the README only) |
|---|---|---|---|---|
| `seven_nielsen_class_representatives.m` (HJLPPZ) | none | `$M23ISGALOIS` | `f3866d1` | `2735178d38926b3dbca758540edc0150114823362cd5afa0ca02213221434fbb` |
| `verification/certify_geometric_reflection_obstruction.g` (Le Lay) | CC BY-NC-SA 4.0 | `$LELAY_M23` | `3f6ab4b` | `fad8d332cfb3f1748b76fd16cbc6bafe386e5d2cb1bc2b92da73d5424fb0fe61` |

Nothing of Le Lay is vendored: his seven representatives are in his CC BY-NC-SA certificate and
paper (the owner's `LELAY_X` transcribes the certificate), and his CC0 `data/` has no Nielsen
representatives. The first version of this design vendored the owner's transcription and said
here that nothing of his was vendored; that was wrong (review 1, B1). The owner's
`passports_C_23A_23B.json` carries `representatives_g1`; these are canonical representatives that
the owner's code computed from the published generators (RustMath recomputes them, 10.2), not
copies of Le Lay's or HJLPPZ's permutations. `group_facts.json` `nielsen` holds indices and
dimensions only.

Note on the table licence: the values are the published ATLAS table (1985), a mathematical fact,
written in our own encoding by our own script and checked by independent computation (orthogonality,
class data, and the counting identity on all 57 rows). This is the same footing as the owner's
`m23_three_point_passports.json`. The owner may want to confirm; it is listed as WPP-Q1.

`rustmath-groups/data/` already holds the degree-24 atlas scripts; `.gitignore` ignores only
`transitive_24.jsonl` and `transitive24_cycletypes.jsonl`, so `data/m23/` is tracked.

## 12. Cost (measured, this machine, release unless stated)

| Item | Time | Memory |
|---|---|---|
| Chain, Steiner system, `G0`, representatives | 0.84 ms (5 ms debug) | |
| Enumeration, `[u8; 23]`, stored | 0.17 s | 234 MB |
| Enumeration, `[u8; 23]`, streamed with signatures | 0.145 s | |
| Enumeration, naive generic `Vec<usize>` (all prefixes rebuilt per element) | 1.23 s (1.15 to 1.29 s in Revision 1) | |
| Enumeration, incremental generic `Vec<usize>` (8.1; Revision 1, m8) | 0.27 s | |
| Same, `u8` recursion (two runs) | 0.15 to 0.19 s | |
| Labelling of all elements: generic enumeration + conversion, against the `u8` recursion | 3.47 s, against 3.17 s | |
| Streamed labelling of all elements (fast 11 test) | 3.2 s | |
| Same with the family test (HashMap) and the sign check | 42 s | |
| 17 centralizers, one streamed pass | 2.3 s (2.8 s generic) | |
| `|C(b)|` by `centralizer_order` for the eight `b` classes of the 57 rows | 84 ms | |
| 57 rows (fast 11 test, `a` in the smaller class) | 2.36 s (2.5 s with the computed `|C(b)|`) | |
| 57 rows (family test, owner's roles) | 12.7 s | |
| `full_census` (Revision 1, m1) | 8.9 s | 3 MB |
| `full_family_11` | 40.3 s | 3 MB |
| `full_57_rows` | 3.6 s | 78 MB |
| `full_c_23a_23b` | 1.3 s | 77 MB |
| Six `(C, 23A, 23B)`: Nielsen 0.21 s, cohomology 0.36 s | 0.58 s | |
| 17 centralizers in debug (orbits for 1A, 2A, 3A, sifting for the rest) | 1.54 s (debug) | |
| Row and column orthogonality, exact | 69 ms and 51 ms (debug) | |
| Fast debug suite (measured parts) | 3.1 s (debug) | |
| `env_hjlppz_dictionary`, `env_lelay_dictionary` | 33 ms, 22 ms (debug) | |
| `Matrix<PrimeField>` 44 x 66 GF(2) kernel | 12 ms each (71 ms in debug, review 1) | |
| Build of `rustmath-groups` release from clean | 1 min 14 s (`-j 2`; 1 min 16 s in Revision 1) | 247 MB target |

The owner's Python reference takes about six minutes (HANDOFF 4.2).

## 13. Plan

Each step keeps `rustmath-groups` build-green and test-green. All commands run in
`/home/user/RustMath-wpp` with `-j 2`.

0. Baseline: `cargo test -p rustmath-groups` and `cargo clippy -p rustmath-groups --tests` on
   `28c4b69`; record the counts and the existing warnings (52 in the lib).
1. `bsgs.rs`: `num_levels`, `orbit`, `transversal_element`, `for_each_element` (incremental) and
   its tests (S4, A5, M11). Acceptance: `cargo test -p rustmath-groups --lib bsgs`.
2. T-215: `representation.rs`, `class_function.rs` as in 4.4, with the new unit tests; no existing
   assertion changed. Acceptance: `cargo test -p rustmath-groups --lib representation class_function`.
3. `Cargo.toml`: add `rustmath-coding` (and `rustmath-finitefields` as dev-dependency). `m23/group.rs`,
   `m23/classes.rs` with `M23Error` (`class_of_member`, `centralizer_order` with the saturating
   `C_Sym` size); tests `m23_classes.rs` (10.1 rows 1 to 5) with `full_census` and
   `full_family_11`. Acceptance: `cargo test -p rustmath-groups --test m23_classes`;
   `cargo test --release -p rustmath-groups --test m23_classes -- --ignored --exact full_census`
   and the same with `full_family_11` (each alone, so that the printed `VmHWM` is its own).
4. `data/m23/m23_atlas_table.g` and `.json` (regenerate and compare the sha256), `m23/atlas.rs`,
   tests `m23_atlas.rs`, and the M23 test of T-215. Acceptance:
   `gap -q rustmath-groups/data/m23/m23_atlas_table.g < /dev/null | sha256sum` gives `f9503e80...`;
   `cargo test -p rustmath-groups --test m23_atlas`.
5. `m23/three_point.rs` (with the derivation of 5.3 in its header and `TrustedResult`); vendored
   data, `extract_janik_table.py` and README in `tests/data/m23/` (rerun the script and compare
   the sha256 of `janik_table.json`); `tests/m23_common/mod.rs`; `m23_three_point.rs` with the fast
   tests and `full_57_rows`. Acceptance:
   `cargo test -p rustmath-groups --test m23_three_point` and
   `cargo test --release -p rustmath-groups --test m23_three_point -- --ignored --exact full_57_rows`.
6. `m23/nielsen.rs`, `m23/cohomology.rs` (with `F2Error` and the `3 dim <= 128` bound);
   `m23_nielsen.rs` with the fast tests, `full_c_23a_23b`, `env_hjlppz_dictionary` and
   `env_lelay_dictionary`. Acceptance: `cargo test -p rustmath-groups --test m23_nielsen`;
   `cargo test --release -p rustmath-groups --test m23_nielsen -- --ignored --exact full_c_23a_23b`;
   `M23ISGALOIS=/home/user/external/m23isgalois LELAY_M23=/home/user/external/frontier-math-m23-cover-investigation cargo test -p rustmath-groups --test m23_nielsen -- --ignored env_`;
   and once without each variable to see the two loud failures.
7. `examples/m23_class_vectors.rs`. Acceptance:
   `cargo run --release -p rustmath-groups --example m23_class_vectors -- --check rustmath-groups/tests/data/m23`
   prints 57 `reproduced`.
8. Whole crate: `cargo test -p rustmath-groups` (debug, all fast tests, time recorded);
   `cargo clippy -p rustmath-groups --tests --examples` (no warning in the new files); `cargo fmt`
   on the new files only (`cargo fmt -p rustmath-groups --check` reports diffs in many existing
   files today; do not reformat them).
   Record debug and release times and peak memory (`VmHWM` from `/proc/self/status` in the
   `full_` tests). Delete `target/release` afterwards if disk is short.

Done when (HANDOFF Section 8, step 3): all 57 rows reproduced and the three dictionaries hold, plus
the six `(C, 23A, 23B)` classes of 4.3.

## 14. Follow-ups (not in this unit)

- WPP-1: 4.4 item 1, the swap action for the twists of `(R, D, D)` (the same counting machinery;
  out of scope by the brief).
- WPP-2: 4.4 items 2 and 3 (computing the small Hurwitz schemes; four branch points).
- WPP-3: the rest of T-215 (SNF abelianization, Artin and braid word problem, trait-default
  facades, chief series, homomorphism check, `block_systems` at `perm_predicates.rs:182`, abelian
  basis). WP-P uses none of them: transitivity is the generation test, not block systems.
- WPP-4: general centralizers and conjugacy tests over a BSGS (Leon's backtrack, MAGMA ch. 58
  `Centralizer`, `IsConjugate`). WP-P uses the `C_Sym(x)` search with a budget, exact but limited
  to small symmetric centralizers. `conjugacy_classes.rs:161` (O(|G|^2)) stays as it is.
- WPP-5: closed by Revision 1 (B1). The Le Lay dictionary is the `env_lelay_dictionary` test,
  which reads his file at run time; nothing of his is vendored.
- WPP-6: a general character-table algorithm (Dixon-Schneider, MAGMA ch. 91). WP-P stores the ATLAS
  table as data.
- WPP-Q1 (for the owner): the licence footing of the generated table (Section 11).
- WPP-Q2 (for the owner, only if he wants it): vendoring his transcription `LELAY_X` of Le Lay's
  seven representatives instead of reading Le Lay's certificate at run time. The default of this
  design is not to vendor (B1). Vendoring needs his explicit decision, recorded before
  implementation, and a README that names Le Lay's paper and certificate as the source.
- WPP-7: an independent check of the rational real criterion against the classical statements
  (Fried 1977; Krull, Neukirch 1971; Debes, Fried 1994) once a copy can be read. Until then the
  rational `real` values carry `TrustedResult::ConjugationThreeRealPoints` (5.3, 8.5).

## 15. The prototype

Directory: `.../scratchpad/m23/wpp/design/WPP/` (no target directory; it was built as
`rustmath-groups/tests/wpp_proto.rs` in the worktree and that file was deleted).

- `wpp_proto.rs`: the prototype (tests `proto_release_all`, `proto_release_streaming`,
  `proto_release_nielsen_cohomology`, `proto_debug_fast`, `proto_debug_18_rows`,
  `proto_debug_dictionaries`). Logs: `run_release.log` (57 rows, stored variant),
  `run_release2.log` (first part-2 run, which failed only in the prototype's own Magma parser,
  fixed; the rerun passed, with the dictionaries, and its output is quoted in Sections 0, 6 and 7),
  `run_debug.log`, `run_debug2.log`, `run_f2bench.log`. The outputs of `proto_release_streaming`
  and `proto_debug_18_rows` went to the terminal only; their numbers are in Sections 10 and 12.
- `gap/m23_atlas_table.g`, `gap/m23_atlas_table.json`, `gap/check_table.py` (an independent
  Python check of the JSON: orthogonality and the 57 constants against the owner's file, both pass),
  `gap/m23_checks.g` (table automorphisms, maximal subgroups, the published generators).
- `vendor/extract_tables.py`, `vendor/janik_lelay_tables.json`: the first version's vendoring,
  superseded by Revision 1 (it contains Le Lay's representatives and must not be vendored, B1).

The bench `rustmath-finitefields/tests/wpp_f2_bench.rs` (the `Matrix<PrimeField>` timing) was also
deleted; its output is `run_f2bench.log`.

Revision 1 (`WPP/rev1/`):

- `proto/`: a standalone scratch crate (path dependencies on the worktree, its own target
  directory, deleted after the runs, `--offline -j 2`), `Cargo.toml` and `Cargo.lock` from the
  review's crate. `tests/wpp_rev1.rs` is the prototype with the changes of Revision 1:
  `centralizer_order` (sifting with the saturating size, else the orbit) used by every count;
  `u = 1` in the rational real count; exact column orthogonality; the GAP-syntax reader; the
  incremental generic enumeration, which `for_each_element` now wraps. Tests: `full_census`,
  `full_family_11`, `full_57_rows`, `full_c_23a_23b`, `bench_enumeration`,
  `env_hjlppz_dictionary`, `env_lelay_dictionary`, `debug_fast`, `debug_18_rows`,
  `debug_dictionaries`, `debug_gap_reader_rejects`. `parts/` holds the pieces it was assembled
  from. `tests/m23_common/mod.rs`, `tests/uses_common.rs` and `examples/m23_check.rs` are the
  check of m10.
- `vendor/extract_janik_table.py`, `vendor/janik_table.json` (JANIK_TABLE only).
- `logs/`: `build_debug.log`, `build_release.log` (1 min 16 s), `run_debug.log` (the first
  `debug_fast` run panicked on the 23! overflow; the rerun after the fix is appended),
  `run_release.log` (`full_` tests with the `u8` enumeration), `run_release2.log`
  (`full_family_11`, `bench_enumeration`), `run_release_generic.log` (the `full_` tests with the
  generic enumeration: the figures of 10.2), `run_env_debug.log` (the two `env_` tests with and
  without their variables).
