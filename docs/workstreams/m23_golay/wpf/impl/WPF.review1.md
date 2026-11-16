# Review 1 of the WP-F implementation (independent, adversarial)

Date: 2026-11-15. Object: the uncommitted changes in `/home/user/RustMath-wpf` (branch
`claude/m23-wpf`, HEAD `aa485d0`), against the design `scratchpad/m23/design/WPF.md`, its review
`scratchpad/m23/design/WPF.review1.md` (D1 rejected) and the report `scratchpad/m23/impl/WPF.md`.
I read the diff and every new file in full. Nothing in the worktree was edited. Reviewer scratch:
`scratchpad/m23/impl/WPF.rev1/`. I deleted `target/release` and my own build directory afterwards.

## Verdict

**Approve.** There are no blockers. Every test passes, in debug and in release, with and without
`M23ISGALOIS`. The six minor items in Section 3 can be fixed before or after the push.

The counts are right for every input class I could construct. I checked them in three ways that do
not use the code under review: an in-loop PARI/GP count on G, 48 new adversarial curves against
`hyperellcharpoly`, and both reference tables. I found no path that reports an undecided result as
Decided. The licence rules for F and calF hold.

## 1. What I ran

All runs were on the worktree, on 4 shared cores (load about 1 to 4).

| Command | Result | Time |
|---|---|---|
| `cargo test -p rustmath-finitefields --test small_field` | 4 passed | 2.9 s (3.8 s wall) |
| `cargo test -p rustmath-finitefields --lib small_field` | 3 passed | 0.01 s |
| `cargo test -p rustmath-curves --test biprojective_small_curves` | 10 passed | 0.87 s |
| `cargo test -p rustmath-curves --test m23_zeta` (fast set, debug) | 8 passed, 8 ignored | 4.97 s |
| `cargo test -p rustmath-curves --lib -- weil residue` | 9 new unit tests passed | 0.01 s |
| `cargo test -p rustmath-finitefields` (whole crate) | all pass | 16 s wall |
| `cargo test -p rustmath-curves` (whole crate) | lib 308 passed, 9 ignored; every target passes | 1 min 11 s wall |
| `cargo build --release -p rustmath-curves --test m23_zeta --example m23_zeta` | ok, `target/release` 320 MB | 68 s |
| `cargo test --release ... --test m23_zeta -- --ignored full_` | 3 passed | 119.7 s (all three at once) |
| `M23ISGALOIS=/home/user/external/m23isgalois cargo test --release ... -- --ignored env_` | 5 passed | 26.6 s |
| the same `env_anchor` without the variable | fails with "set M23ISGALOIS to a checkout of shaowuz/m23isgalois" | never passes vacuously |
| `target/release/examples/m23_zeta 79 79 4` | Decided nodal, row equal to `zeta_X.json` | 14.9 s (report: 19.5 s under heavier load) |
| `target/release/examples/m23_zeta 83 83 4` | `91, 6885, 574375, 47483041`, Decided nodal, equal to report Table 3.1 | 18.4 s (report: 23.8 s) |
| `cargo clippy -p rustmath-finitefields --tests --no-deps`; `cargo clippy -p rustmath-curves --no-deps --lib --examples --test m23_zeta --test biprojective_small_curves` | no warning in the new files | |
| `rustfmt --check` on the new `.rs` files | clean | |
| `grep unsafe`, em-dash search in the new files | none | |

As the report says, the lib-test clippy target of `rustmath-curves` stops on three pre-existing
`approx_constant` errors in `belyi/kernel_refine.rs:1777, 1801`. No new file is involved. The
measured costs agree with the report and the design: about 2 us wall per quartic fibre on 4 threads
at k = 4, and the debug fast set in seconds.

## 2. Independent checks

### 2.1 The X rows, with no Rust involved (`WPF.rev1/gp/indep.gp`)

I wrote my own PARI/GP count on Le Lay's G, parsed directly from the vendored JSON. It works in the
style of `curve.py`. It sums the distinct roots in `P^1(F_q)` of every fibre over `P^1(F_q)` in V.
In every fibre it finds the singular points with `gcd(f, f_T, F_V)` and applies the Hessian rule.
It also checks the chart at `T = infinity` and the place `V = infinity`.

| p, k | plane | node correction | N | report / reference |
|---|---|---|---|---|
| 17, 2 | 282 | +3 | 285 | `zeta_X.json` 285 |
| 83, 1 and 2 | 90, 6884 | +1, +1 | 91, 6885 | new row: 91, 6885 |
| 89, 1 | 104 | 0 | 104 | new row: 104 |
| 131, 1 and 2 | 144, 17064 | 0, 0 | 144, 17064 | new row: 144, 17064 |
| 107, 1 and 2 | 115, 11481 | 0 | 115, 11481 under the reference rule; one point with H = 0 at (T, V) = (23, 86) | new row: 116, 11482 (exact rule) |

At 107 I classified the point separately in gp (`WPF.rev1/gp/a3_107.gp`). `disc_T G` vanishes to
order exactly 4 at V = 86. The fibre is `(T - 23)^2 q(T)` with `disc q != 0`. The product
`(D / (V - 86)^4)(86) * disc(q)` is a nonzero square mod 107. So the point is an A3 with rational
branches, worth +1 at k = 1 and k = 2. That gives 116 and 11482, the implementation's values.

### 2.2 Adversarial small curves against PARI/GP (`WPF.rev1/gp/adv2.gp`, harness `WPF.rev1/xcheck`)

The harness compiles the worktree's `nodal_biprojective` and `weil` sources unchanged through
`#[path]`. The curves are `X^2 + b(Y) X + c(Y)` at p = 13, built so that `b^2 - 4c = S(Y)^2 H(Y)`.
None of them is in the implementer's tests. They cover the case the report leaves as follow-up F-11.
For each curve I compared N_1..N_6 with `hyperellcharpoly` of the normalization (or with `p^k + 1` for
genus 0). I ran each curve in three charts: the original, `X -> 1/X` and `Y -> 1/Y`.

| family | what it exercises | cases | result |
|---|---|---|---|
| node at a degree-3 place, genus 1 | residue degree 3; branch orbit `[2]` becoming rational at k = 6 | both square classes | counts equal; Decided nodal (3 nodes) |
| A3 at a degree-2 place, genus 1 | odd k over `F_{p^2}`, square class of `-2 a lead(g)` in a degree-2 field | both classes, several H | equal; DeltaSumCertified, orbits `[1,1]` / `[2]` as predicted |
| A5 at a degree-2 place, A3 at a degree-3 place, genus 0 | odd k, residue degree 2 and 3 | both classes | equal |
| A2 at a degree-3 place (genus 1), A4 at a degree-2 place (genus 2) | even k, unibranch | | equal |
| A7, A15 at rational places, genus 0 | precision doubling from y^16 to y^32 | both classes | equal |

Result: 144 of 144 runs equal, with the branch orbits predicted by gp (`WPF.rev1/adv_out.txt`).

Further probes:
- E6 (`Y^3 = X^4`, at p = 7, 11, 13) gives two `Unclassified` points. By default the result is
  Unresolved with no counts. With `unibranch_if_unclassified` the rule is AssumedUnibranch and the
  status is not Decided.
- k = 7 and 8, the `SmallFq<7>` and `SmallFq<8>` paths: on `V^2 = T^3 + 3T + 5` at p = 7, N_1..N_8
  equal the counts predicted by PARI's `ellap = 1`.

### 2.3 Reference tables, licences, data

- `full_x_table_k4` asserts all 18 rows of `zeta_X.json` at k = 4, with P, Weil, mod 23 and the
  table's Q. `env_l_table` asserts the rows at 7, 19, 41 and 43, the refusals at 5 and 11, and both
  rules at (31, 12). `env_anchor_53` asserts 52, 57, 45. `env_f_equals_g_by_mobius` checks the
  exact Integer identity. All of these pass.
- The vendored files have the sha256 values that `tests/data/m23/README.md` gives, and they are
  equal to their sources (Le Lay `data/optimal_23_4_Z.json`; johnjanik/M23 `data/*.json` at `d05475b`).
- I took 20 long coefficients sampled from `polynomial_F.m` and `polynomial_calF.m`. None of them
  occurs anywhere in `rustmath-curves`, `rustmath-finitefields` or `docs/`. F and calF are read only
  through `M23ISGALOIS`. HANDOFF Section 1 allows `h62.json` ("copy the JSON files it tests against").
- `docs/` and `PROJECT_TRACKER.md` are untouched. `Cargo.lock` adds only `serde_json`.

### 2.4 Code read against the design: correct

- **Kernel** (`small_field.rs`): Barrett reduction, accumulator bounds (below 2^48 on every path),
  Frobenius matrix, norm and inverse through the conjugates, `is_square` through the norm, the
  semilinear `x^q`, and `deg gcd(f, x^q - x)`.
- **Fibre loop** (`fibres.rs`): orbit-minimal codes weighted by orbit size, the degree drop counted
  as a root at infinity, the base point at infinity, and a vanishing fibre refused.
- **Singular locus** (`singular.rs`):
  - Every irreducible factor of the formal discriminant is analysed, plus `Y = infinity` when
    `deg D` drops. The discriminant formulas are the standard ones.
  - `deg G <= 3` is complete for quartic fibres and p >= 5. Two distinct multiple roots force
    `deg G = 2`. Degree 3 forces `c (X - tau)^4`.
  - The `X = infinity` test is `c_dx = c_{dx-1} = c_dx' = 0`.
  - Hessian, `QuadExt` norm and inverse, and the resultant norm are right.
- **Classifier** (`local.rs`): the rank-1 reduction `x = u - (b/a) y` kills the cross term and the
  `y^2` term. Newton on the polar curve converges in at most 8 steps. `ord g = k + 1`. The branch
  class is `-2 a lead(g)`. The rank-0 tangent cone is handled with the tangent `y = 0`.
- **Certificate** (`certify.rs`): Decided only for "all nodes, `p_a - g` of them". Any non-nodal
  point gives `Unresolved`. A delta-sum below `p_a - g` is an error. Counts made with
  `AssumedUnibranch` never carry Decided, because a non-nodal point is present. The proof of
  Theorem 4.5 is restated correctly in the header.
- **Weil** (`weil.rs`):
  - Newton's identities with the integrality check, and the functional equation.
  - `R` built from `D_j`.
  - The exact Sturm test on each squarefree factor of `R` and of `S(y)`. The interval `(-1, 4p]` is
    closed at `4p`.
  - Exhaustive `Q Q^rev`.
  - The 2-torsion order: `DeterminedByP` only when `P mod 2` is squarefree, and the driver prints
    "Decided" only on a Decided row.

## 3. Minor items (none blocks approval)

1. **The Unclassified path has no regression test.** No test reaches `Unresolved::Unclassified`,
   `DeltaMismatch { found_is_lower_bound: true }` or `unibranch_if_unclassified`. That is the
   honesty path: no counts by default, never Decided. A wrong edit there would go unnoticed. Add the
   E6 case of 2.2 (`FormModP` with `cols[0][3] = 1`, `cols[4][0] = p - 1`, genus 0): status
   `Unclassified`, `counts` empty. With `unibranch_if_unclassified`: `branch_rule == AssumedUnibranch`
   and `!status.is_decided()`.
2. **The cost of the public counting entry points is not documented**
   (`fibres.rs:153 plane_point_count`, `certify.rs:308 normalization_point_counts`). They accept any
   `p^k < 2^63`. For example, p = 65521 with k = 3 means about 2.8e14 fibres, which is years. The
   per-chunk results are also collected into a `Vec` of `q / 2^14` entries, which grows with q
   (`fibres.rs:127`). State the cost: about `q/k + 1` fibre root counts, measured at about 2 us per
   fibre on 4 threads at k = 4. Sum the chunks with a reduction instead of collecting them. The same
   applies to `singular_locus` when `dy` is large: `FormModP` accepts `dy` up to 4096, which means
   factoring a discriminant of degree up to 24576.
3. **Panics in public functions are not documented.**
   - `count_distinct_roots` and `count_distinct_roots_any_degree` assert degree <= 4096
     (`small_field.rs:734`). The first function's doc does not say so.
   - `FrobeniusPolynomial::two_torsion_frobenius_order` panics when `P mod 2` has an irreducible
     factor of degree above 62 (`weil.rs:459`, possible from genus 32 on). For degree near 62 it
     enumerates divisors by trial up to about 2^31.
   Document both with "Panics if ...", or return `Option`/`Result`.
4. **The driver hides `GenusInconsistent`.** `select_model` (`tests/common/m23.rs:341`) treats
   `Err(CountError::GenusInconsistent)` from one model like a refusal. It can then return Decided
   from another model. But that error contradicts the genus hypothesis on which every certificate
   rests, so the driver should panic or report it, not skip it. The M23 data never triggers it.
   This is test and example code.
5. **Two file headers lack the MAGMA chapter.** `certify.rs` and `residue.rs` cite the papers but
   not the MAGMA chapter digest, as CLAUDE.md asks of each new file. Adding "MAGMA ch. 114" is enough.
6. **Close F-11 with tests.** The A_k part of follow-up F-11 (odd k over a residue field of degree
   > 1) already works (Section 2.2). Moving a few of the `adv2.gp` cases into
   `biprojective_small_curves.rs` would close it with tests: the degree-3 node, A3 at degree 2 in
   both classes, and A15. The report's F-11 would then only need the ordinary m-fold point over
   `F_{p^2}`.

## 4. Notes for the report and the handoff

- The report's claims on scope, statuses, deviations and follow-ups are accurate. Section 3 of the
  report matches what I reproduced: 79, 83, 89, 107 and 131 independently, the rest through the
  reference tables and the tests.
- The opt-in commands work as documented. Without `M23ISGALOIS` the `env_` tests fail loudly.
  Everything else runs on the CC0 model G.
- Reproduce the review with these commands, run in `scratchpad/m23/impl/WPF.rev1/`:
  - `gp -q gp/indep.gp < /dev/null` (Section 2.1)
  - `gp -q gp/a3_107.gp < /dev/null` (Section 2.1, the point at 107)
  - `gp -q gp/adv2.gp < /dev/null > gp/cases.txt` (generates the curves of Section 2.2)
  - `(cd xcheck && cargo build --release --bins && ./target/release/adv ../gp/cases.txt && ./target/release/e6 && ./target/release/k78)` (Section 2.2; about 10 s to build and 2 min to run)
  - afterwards, delete `xcheck/target`.
