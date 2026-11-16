# RustMath handoff: the M23 Golay computations

Date: 2026-11-14. Written for a RustMath session that starts cold. Read `docs/HANDOFF.md` and
`CLAUDE.md` first, as usual; this file is the brief for one workstream.

The reference data, the Python reference implementation and the paper live in the public
repository `johnjanik/M23`, branch `claude/blissful-knuth-qla7ym`, pinned here at commit
`d05475b`. In this file every path that does not start with `rustmath-`, `docs/` or
`PROJECT_TRACKER.md` is relative to a checkout of that repository. Fetch it outside the RustMath
tree:

```bash
git clone --branch claude/blissful-knuth-qla7ym https://github.com/johnjanik/M23 ../M23
```

Nothing from it is vendored into RustMath yet. When a work package starts, copy the JSON files it
tests against into that crate's `tests/data/` with a note of the source commit. The same file is
kept in the M23 repository as `docs/RUSTMATH_HANDOFF.md`; this copy is the one to edit.

Three things are wanted from RustMath. Each has reference values computed independently
(Python/FLINT, cross-checked with PARI and GAP), so each is a test of RustMath as much as a
contribution to the paper `paper/golay_m23.pdf`.

| Work package | Goal | Reference data exists | New mathematics |
|---|---|---|---|
| WP-F | Reproduce and extend the Frobenius table of the genus 4 curve X | yes, 18 primes (19 at the owner's main `2d11038`) | no |
| WP-P | Reproduce and extend the table of three-point class vectors of M23 (Phase 6 of the research note) | yes, all 57 class vectors | the covers themselves are open |
| WP-A | Find an explicit polynomial for the A8 field cut out by the Golay Lagrangian (Question 10.1 of the paper) | acceptance tests only | yes, the field is unknown |

Do them in that order. WP-F and WP-P have exact expected outputs. WP-A does not: it has
predictions that the answer must satisfy, and a designed pipeline that has not been run.

## 1. Background in one page

Read `paper/golay_m23.pdf` Sections 1, 2 and 4 for the full statements. The minimum:

- **The curve.** Huang, Jackson, Lee, Poonen, Pries and Zhang (HJLPPZ, arXiv:2608.08538v2)
  give `F(T,V)` in `Z[T,V]` of bidegree (4,23). Its closure in P^1 x P^1 has exactly 62 nodes;
  the normalization `X` has genus 4. The map `t: X -> P^1` has degree 23, monodromy group M23,
  branch points `0, +sqrt(-23), -sqrt(-23)` with classes `(2A, 23A, 23B)`. The function `v` has
  degree 4.
- **The Golay Lagrangian.** The 23 sheets carry a Steiner system S(4,7,23). The even Golay
  code `G0` (dimension 11) is a sub-local-system of the permutation sheaf mod 2 and
  `W = H^1(P^1, j_* G0)` is a Galois-stable Lagrangian subgroup of `Jac(X)[2]`, of order 2^4
  (paper, Theorem 4.1).
- **What is known about W.** Galois acts on `W = F_2^4` through all of `GL_4(F_2) = A_8`
  (paper, Theorem 4.8). The kernel field `K_W` is an A8 extension of Q unramified outside
  `{2,3,23}`. This is the paper at the owner's main `2d11038`; at `d05475b` it said
  `{2,3,5,23}`. The prime 5 left the set because X has good reduction at 5, by Lemma 9.1 of
  the paper (checked by the owner's classifier and by WP-F: Section 3, Status). Complex
  conjugation acts as a transvection. No polynomial for `K_W` is known.
- **Conventions.** Permutations act on the right: `x^(gh) = (x^g)^h`. Products are read left
  to right, as in MAGMA and GAP. Published cycles are 1-based; the Python library is 0-based.
  `23A`, `23B` are the classes of the published `g2`, `g3`. Class names of other split pairs
  (`7A/7B` and so on) are fixed only up to swapping the pair.

### Where the data is

Everything below is in the M23 repository unless a URL is given.

| File | Contents |
|---|---|
| `m23golay/` | Python reference implementation (permutations, Golay codes, M23, Nielsen classes, mod 2 cohomology, p-adic labelling, point counting) |
| `data/group_facts.json` | class table, cycle types on duads and heptads, Molien series, Nielsen class with real structure |
| `data/zeta_X.json` | point counts and L-polynomials of X for p <= 79 |
| `data/zeta_L.json` | the same for the six conjugate covers at degree-one primes of the sextic field |
| `data/h62.json` | `h62(T)`: `disc_V F = 2^1652 3^626 23^228 131^506 T^8 (T^2+23)^22 h62^2`; its roots are the T-coordinates of the 62 nodes |
| `data/golay_resolvents.json` | J5, J7, T5 and the degree 253 heptad and duad resolvents of the published polynomials f1, f2 |
| `data/J5_family.json` | J5 of the 23 roots of `F(t,V)` as `n(t)/c23(t)` |
| `data/lifts_2_11_M23.json` | fifteen generating triples of `2^11:M23` in S_46 (WP-A, route B) |
| `data/pg32_a8_model.json` | PG(3,2) in the cyclic model with its 56 spreads and the 8 letter families (WP-A, step A5) |
| `data/m23_three_point_passports.json` | 57 rational class vectors with structure constants (GAP character table) |
| `data/m23_three_point_nielsen.json` | exact sizes of the 57 Nielsen classes and numbers of real points |
| `data/passports_C_23A_23B.json` | the six class vectors `(C,23A,23B)` in detail: representatives, ovals |
| `scripts/` | the scripts that produced each file; the README has a command table |

At the owner's main `2d11038` the two zeta files carry his corrected rows (p = 5 is counted)
and, on each row, the singular points found by his classifier (`m23golay/curve.py`). Two files
are new there: `data/rustmath_wpf_tables.json` (RustMath's WP-F rows as he recorded them) and
`data/rustmath_crosscheck.json` (his recount of RustMath's rows at 83, 97 and 107). See Section
3, Status.

External data, not vendored:

| Source | What | Licence note |
|---|---|---|
| `https://github.com/shaowuz/m23isgalois` | `polynomial_F.m` (the equation), `polynomial_calF.m` (model over the sextic field), f1, f2, seven triples | no licence file in the repository; fetch at test time, do not copy into RustMath |
| `https://github.com/research-reflexivity/frontier-math-m23-cover-investigation` | Le Lay's data: canonical models of all seven curves, marked points, the canonical quadric over Q, a second plane model | `data/` is CC0-1.0 and may be vendored with attribution; the paper and code are CC BY-NC-SA 4.0 and must not be copied into GPL code |

`scripts/fetch_external.sh` fetches the first. The parser for `polynomial_F.m` is five lines:
terms have the form `<int>*T^<i>*V^<j>` (see `m23golay/data.py`).

## 2. What RustMath can and cannot be trusted for

From `PROJECT_TRACKER.md` at RustMath commit `6e03987`, with each line checked in the code. Check the
tracker again before relying on any row. Each line names the entry point as `file:line`. The bracket
after it gives the tracker rating: done (tested against reference values), partial, facade, broken or
missing. "No row" means the tracker has no inventory row, so the line rests on the code and its own
tests. Costs are read from the code and were not timed, unless a line names a tracker timing.

### Usable as is

All three packages:

- Integers, rationals, CRT: `rustmath-integers/src/integer.rs:64`, `crt.rs:17`, `rustmath-rationals`
  [done]. `try_is_prime` (`prime.rs:160`) returns a proof or `Err`. `is_prime` (`:173`) panics when
  the ECPP proof is unresolved. The Kronecker symbol for `chi(p)` is `integer.rs:788` [done].
- `GF(p)` on machine words: `rustmath-polynomials/src/fp_factor.rs`, `squarefree_factor` `:379`,
  `distinct_degree_factor` `:435`, `factor` `:809`, `factor_with_multiplicity` `:854` [done]. These
  entries prove `p` with the deterministic `is_prime_u64` (`check_modulus` `:592`), which is cheap.
  They need `p <= 3037000499` (`MAX_P` `:483`). They take `i64` coefficients and read them modulo
  `p`, so the caller reduces a large `Integer` coefficient first. The helpers (`gcd` `:167`, `mul`,
  `div_mod`, `pow_mod` and the rest) check nothing (module doc, `:48`). They need `p` prime
  `<= MAX_P` and coefficients already in `[0, p)`. Characteristic 2 uses the trace map. This is the
  fast path for any computation modulo one prime.
- `GF(p^n)`: `FiniteField::new`, `with_modulus` (`rustmath-finitefields/src/finite_field.rs:145`,
  `:183`) [done]. Both prove `p` once with `try_is_prime`. Element operations do not check it again.
  Conway polynomials exist only for small `p` and `n` (`conway.rs:139`). Otherwise `new` takes the
  first irreducible in base-`p` counter order (`poly_factor.rs:276`). This default may change in a
  future version: `docs/HANDOFF.md:342` notes that a fix of the search changes default moduli. So use
  `with_modulus` when the field must be reproducible. Frobenius `finite_field.rs:537`, `is_square`
  `:699`, `sqrt` `:725` [done].
- Factorization over `GF(q)`: squarefree `rustmath-finitefields/src/poly_factor.rs:651`,
  distinct-degree `:705`, equal-degree `:787`, `factor` `:970`, `roots` `:1022` [done]. A second
  stack, `cantor_zassenhaus.rs:622` [done], is generic over `FiniteFieldElement`, with impls for
  `PrimeField` and `Gfpn` (`ff_poly.rs:213`, `:785`). It proves `p` on every public call
  (`ff_poly.rs:196`). `PrimeField` and `Gfpn` also store `Integer`s, so neither stack is a
  word-size kernel.
  Equal-degree splitting certifies each block. It panics with probability at most 2^-256 and never
  returns an unsplit block.
- Zassenhaus over `Z`: `rustmath-polynomials/src/zassenhaus.rs:385` [done]. Typed errors, and `Err`
  beyond 2^22 recombination subsets (`MAX_RECOMBINE` `:91`). There is no van Hoeij.
- Hensel lifting of a factorization: `rustmath-polynomials/src/zp_hensel.rs:143`, `hensel_lift_all`
  `:192`, quadratic lift mod `p^k` `factorization.rs:671` [done].
- Resultant and discriminant over `Z`: `rustmath-polynomials/src/disc.rs:150`, `:204` [done],
  multimodular with the Hadamard bound. Generic Bareiss resultant `univariate.rs:537` [done].
- Linear algebra over a field: `Matrix<F: Field>`, echelon form
  `rustmath-matrix/src/linear_solve.rs:52`, rank `:158`, solve `:175`, kernel `:374`, inverse `:482`
  [done]. Over `Q` it runs on `Rational`. Over `GF(p)` it runs on `PrimeField`: `solve_gf7` checks
  solve against PARI, and the canonical-form tests run over GF(2), GF(3), GF(5), GF(7) and GF(101)
  (`rustmath-finitefields/tests/matrix_forms_gfp.rs`). Limits: build entries with `PrimeField::new`.
  It does not check that `p` is prime (`prime_field.rs:54`). An entry made only from `Ring::zero()`
  or `Ring::one()` stays an integer (`docs/HANDOFF.md:331`).
- Integer factorization (for `disc(g8)` and similar): `factorization_with_limits`
  (`rustmath-integers/src/factorization.rs:183`) [done]. Factors in its `primes` list are proven. A
  factor whose ECPP proof is unresolved comes back in `probable_primes` (BPSW), as does every factor above
  psi_12 (about 3.2 * 10^23) when the proof is switched off. What it cannot split comes back as a composite, never as a prime.
  There is no NFS. The tracker times the SIQS at about 36 s for 70 digits in a release build.

WP-F:

- Root count of a quartic over `GF(p)`: `fp_factor::distinct_degree_factor` on the radical from
  `squarefree_factor` [done]. It panics on a repeated factor. It returns pairs `(d, g_d)` and leaves
  out every empty block. The number of distinct roots is the degree of `g_1`, from the pair with
  `d = 1`, and 0 when there is no such pair. For example, an irreducible quartic gives `[(4, f)]`.
  One `x^p mod f` per degree step, on `i64` words.
- Root count of a quartic over `GF(p^k)`: `square_free_factorization`, then
  `distinct_degree_factorization` on each squarefree factor (`poly_factor.rs:651`, `:705`), on an
  `FqPoly` over `FiniteField::new(p, k)`. This DDF returns pairs `(g_d, d)` and also leaves out empty
  blocks, so the root count is the sum of the degrees of the `d = 1` blocks over all
  squarefree factors. Limit: cost. Each degree step computes `x^q mod f`, about
  `k log2 p` squarings of a cubic modulo the quartic. Each `GF(p^k)` product is `k^2` products of
  heap-allocated `Integer`s plus a reduction. Correct, but far too slow for the `p^4` base points of
  the `k = 4` count near `p = 100`. `roots` (`:1022`) also runs equal-degree splitting.
- Frobenius orbits and the Hessian square test: `FiniteFieldElement::frobenius`, `is_square`
  (`finite_field.rs:537`, `:699`), `PrimeField::is_square` (`prime_field.rs:241`) [done].
- Sturm sequences over `Q`: `count_real_roots`, `count_real_roots_in`
  (`rustmath-polynomials/src/real_roots.rs:147`, `:158`) [done, four hand-checked tests]. Limit: it
  counts distinct roots in a half-open interval `(a, b]` with rational endpoints. The interval
  `[-2 sqrt(p), 2 sqrt(p)]` of 3.1 step 5 has irrational endpoints and needs an exact reduction.
  The Weil test counts roots with multiplicity, so run it on each squarefree factor of `R` and weight
  by multiplicity.
- `P_p mod 2` and the mod 23 check: `fp_factor::factor_with_multiplicity` [done] and integer
  evaluation.
- Discriminant identity at integer `T`: `disc.rs:204` on `F(t, V)` [done]. Squarefreeness of `h62`
  mod `p`: reduce its coefficients into `[0, p)`. Then the radical from
  `fp_factor::squarefree_factor` has degree 62 exactly when `h62` stays squarefree of degree 62.
  Coprimality with `T(T^2+23)`: the unchecked `fp_factor::gcd` on the reduced inputs.

WP-P:

- Schreier-Sims: `StabilizerChain::new` (`rustmath-groups/src/bsgs.rs:138`), order `:278`,
  membership by sifting `:289` [no row; blocker B-02 records it as resolved in `5ff8749`].
  Deterministic. Tests: `mathieu_m23_order` (`:480`), `m23_membership_respects_point_stabilizer`
  (`:510`, M23 on 23 points), `mathieu_m24_order_and_membership` (`:487`). Limits: products are read
  right to left (`compose(a, b)[i] = a[b[i]]`, `bsgs.rs:43`), the opposite of Section 1. The chain
  keeps its transversals private, so there is no element iterator.
- Orbits, transitivity, primitivity: `rustmath-groups/src/perm_predicates.rs:41`, `:63`, `:80`,
  `:202` [done]. Transitivity is the generation test of 4.2 step 5.
- Cycle types on 23 points: `transitive23::cycle_type` (`transitive23.rs:74`) on `[u8; 23]` [done,
  inside the degree-23 classifier row].
- The whole group by closure: `transitive23::group_closure` (`transitive23.rs:99`) [no row of its
  own]. Limit: closure into an unordered `HashSet`, tested only on the solvable groups of degree 23
  (order at most 506). For M23 it needs several hundred MB (estimate, not measured). It is not the
  ordered stabilizer-chain enumeration of 4.2.
- Binary Golay code `[23,12,7]`: `BinaryGolayCode::new` (`rustmath-coding/src/golay.rs:33`) [done]:
  weight enumerator checked, every error of weight at most 3 corrected. Limits: the coordinates are
  those of the cyclic code of `x^11+x^10+x^6+x^5+x^4+x^2+1`, the reciprocal of the polynomial in
  4.2. They are not the labelling of the published generators. Since the Steiner labelling,
  `rustmath-coding/src/cyclic_golay.rs` has both cyclic codes, of `g+` and of `g-` (the code
  above), with their words (`:103`), weight distributions (`:115`), the 253 heptads of each
  (`:155`, two S(4,7,23), checked exactly) and the 11 shift orbits (`:161`) [done]. There is still
  no accessor for the even subcode `G0` or the automorphism group. The extended code
  `golay.rs:371` [partial] has its 759 octads checked. Its decoder ignores the parity bit.
- Linear algebra over `F_2` for 4.3: `Matrix<PrimeField>` with `p = 2`, as above. The modules have
  dimension at most 22, so cost does not matter.
- Genus 0 Belyi engine, `rustmath-curves/src/belyi/`. These parts take any passport: permutations
  and the Riemann-Hurwitz genus (`monodromy.rs:41`), triangle groups and coset graphs
  (`triangle_group.rs:113`), the ansatz encoder (`encode.rs:178`) [all done], and the KMSV fit of a
  genus 0 map (`genus0_map.rs:116`) on the modular-forms route (`modular_forms.rs:33`) [partial].
  These parts are written for the degree 24 passport `2^8 1^8 / 12^2 / 5^4 1^4` only: the Newton
  solves (`newton.rs:262`, `:785`), the numerical monodromy (`numerical_monodromy.rs:96`, `:152`) and
  the exact verifier `verify_2_12_5_cover` (`verify.rs:59`) [done]. The solves and the monodromy
  take that passport's root layout, `FactorizedRoots` (`factorized_residual.rs:112`). The verifier's
  trait asks for `verify_ramification_2_12_5` (`verify.rs:49`). `candidate_verify.rs:48` checks the
  identity for any passport, but only in f64, and the f64 refinement `refine_candidate`
  (`solve.rs:299`) and the true-root gate `gate_numerical_candidate` (`:278`) take any square system.
  Tracker row 2099 records that the power-series route did not scale for [2,12,5] (rho about 0.9906). Limits: genus 0 only (`genus0_map.rs:1`; the tracker
  lists Belyi maps of genus >= 1 as absent). The path tracking is not certified. So the degree 23
  rows of 4.4 item 2 can use the encoder and the KMSV fit, and need the rest built (What to build).

WP-A:

- Groebner bases over a field: `groebner_basis_field`, `reduced_groebner_basis_field`
  (`rustmath-polynomials/src/groebner.rs:1043`, `:1164`) [done]. Orders Lex, Grlex, Grevlex and
  `Elimination { block }` (`:66`). Elimination ideals `elimination.rs:88` [done]. The number of
  solutions with multiplicity `staircase.rs:58` [done]. The code type-checks for any `Field`:
  `PrimeField`, `FiniteFieldElement` or `GFp<P>` (`rustmath-rings/src/function_field/typed/gfp.rs:32`
  [done]; `P` is not checked prime). But every test is small and over `Q` or `Z`, and none runs over
  a finite field. Before A1, add a small test over `GF(p)` with a Singular or Sage reference. Limits:
  Buchberger with normal selection and the first criterion only (row `groebner.rs:719`, [partial]).
  `groebner_basis_f4` (`:800`) is Buchberger under that name [missing]. There is no FGLM. The default
  budget (`:847`) allows 20,000 pairs, 500 basis elements, degree 40, 5,000 terms and 10 s. It
  returns `Err` when it trips. `GroebnerBudget::generous()` (`:863`) allows degree 500 and 120 s.
  The cost on a zero-dimensional system over `F_p` with hundreds of solutions is unknown. A degree
  240 eliminant already exceeds the default degree.
- Smoothness of the canonical model (A0): the model is a complete intersection of a quadric and a
  cubic in P^3, not a hypersurface. So `hypersurface_singular_locus` and `is_smooth_hypersurface`
  (`rustmath-schemes/src/singularity.rs:96`, `:116`) [done] do not apply. Use `singular_subscheme`
  (`:263`) [partial] on each affine chart `x_i = 1`, and test each result with `is_unit_ideal`
  (`:63`) [done]. Its precondition is an equidimensional `V(I)`, which a complete intersection is.
  Tested over `Q` only.
- Galois rings `GR(p^a, d)`: `GaloisRing::new`, `with_modulus`
  (`rustmath-finitefields/src/galois_ring.rs:77`, `:112`), Teichmueller lift `:240`, `:658`,
  Teichmueller digits `:676`, Frobenius `:717`, Newton unit inverse `:785`, `xgcd` `:908`, orders
  `:1243` [done, row `galois_ring.rs:55`]. `p` is proven at construction. This is the ring
  `(Z/p^N)[x]/(f)`. Since the Steiner labelling, `with_modulus` makes a `D` with `D[d] = 1` mod `p`
  monic mod `p^a` (before, such a `D` was stored as given and the arithmetic was wrong), the
  product reduces lazily (same values), and `defining_polynomial` (`:164`) and `base_ring_value`
  (`:475`) exist. Limits: elements are `Integer` vectors. `GaloisRingElement::frobenius` still
  recomputes all `a` Teichmueller digits by powers of `p^d`, roughly `a (a/d + 1) d log2 p` ring
  products per call; for repeated use take `FrobeniusMatrix` (next item). The ring is not a field,
  so `Matrix` has no solve or inverse over it (see Do not call).
- The Frobenius as a matrix, and roots in a Galois ring:
  `rustmath-finitefields/src/galois_ring_frobenius.rs:114` `FrobeniusMatrix` (Newton lift of
  `w^p` to `phi(w)`; `apply` `:348` is one matrix-vector product; `orbit` `:354`; `trace` `:373`)
  and `:460` `orbit_polynomial` (an `Err` when a coefficient is not in `Z/p^a`) [done, checked
  against the Teichmueller Frobenius and PARI/GP `polsym`]; `:555` `FrobeniusRoots`, the roots
  `phi^i(x)` of a monic `f` that is irreducible mod `p`, in `GR(p^N, deg f)`, with exact
  squarefree and inertness checks [done, PARI/GP `polrootspadic`]. This is what the Steiner
  labelling runs on (Section 6, Status). One polynomial only: the multivariate Newton of A3 is
  still absent (Do not call).
- The same ring with a cheap Frobenius: `UnramifiedExtension`
  (`rustmath-rings/src/padics/unramified.rs:138`) [done]. Frobenius uses the precomputed image of the
  generator (`:243`, `:411`). Norm and trace are cross-certified. `FrobeniusMatrix` (above) gives
  the same cheap Frobenius on `GaloisRing` itself.
- Rational reconstruction: `rational_reconstruct`, `crt_rational_reconstruct`
  (`rustmath-integers/src/crt.rs:130`, `:212`) [done in the tracker's IGP24 table, item C3];
  `recognize_rational` (`rustmath-numberfields/src/recognize.rs:183`) [done, labelled heuristic].
  `rational_reconstruct` returns the unique `n/d` with `|n|, d <= floor(sqrt((M-1)/2))` that matches,
  or `None` when there is none. A returned fraction is not certified: if the true height is larger,
  it can be wrong. There is no stability loop, so the caller adds primes or raises `N` and compares.
  The Steiner labelling needed none: its integers have proven bounds and pass `bounded_integer`
  (`rustmath-galois/src/integrality.rs:110`, `p^N > 2B` or `Err`), and the sign comes from
  `integrality_dichotomy` (`:170`, Proposition 5.1) [done].
- Recognition of algebraic numbers: `recognize_algebraic` (from a p-adic residue), and
  `recognize_complex_algebraic` (`recognize.rs:195`, `:259`) [done, labelled heuristic], and its
  high-precision form (`:354`) [no tracker row]. They find a minimal polynomial by LLL. The caller must certify it by
  exact substitution.
- Maximal order, integral basis, field discriminant: `rustmath-numberfields/src/round2.rs:1086`,
  `:745`, `:732` [done, checked against LMFDB]. Limit: only primes up to 10^5 whose square divides
  `disc(f)` are made maximal (`:682`, bound set at `:733`), and a larger one stays in the answer. A
  result supported on `{2, 3, 23}` therefore proves test 3 of 5.2. A large prime in it must be
  checked.
- Prime decomposition: `rustmath-numberfields/src/ideals.rs:127` [done, LMFDB], with `p` an `i64`.
- polred: `rustmath-numberfields/src/polred.rs:302` [done, LMFDB]. The output always defines the
  same field. Limit: the T2 lattice comes from f64 roots rounded into `i64` (`polred.rs:51-56`,
  `:145-153`), and the LLL is the f64 one of `rustmath-matrix/src/lll.rs:75` [partial].
- LLL: `lll_reduce_rf` with BigFloat Gram-Schmidt (`rustmath-matrix/src/lattice.rs:378`) [done] and
  the exact test `lll_is_reduced_exact` (`:246`) [done]. Limit: both LLL loops stop at an iteration
  cap without saying so (`lll.rs:92`, `lattice.rs:329`). Run the exact test on the output.
- Cycle types mod `p` for 5.2: `frobenius_cycle_types`, `cycle_type_mod_p`
  (`rustmath-galois/src/sieve.rs:103`, `:82`), Dedekind shape
  `rustmath-polynomials/src/padic_factor.rs:72` [done]. `galois_group`
  (`rustmath-galois/src/lib.rs:178`) is exact up to degree 5 [done]. Above that it decides only `A_n`
  and `S_n`, by Jordan's criterion (`sieve.rs:156`, [partial]). For `g8` a Frobenius of type `3 5`
  or `1^3 5` powers to a 5-cycle, so it can decide A8. For `g15` expect `Unresolved`.
- Exact arithmetic in `Q(T)[V]/(F)` for 5.6: `RationalFunction`
  (`rustmath-functionfields/src/ratfunc.rs:30`, a `Field`) with `UnivariatePolynomial` division [no
  row; five unit tests].

### Do not call

Facade, broken or missing at `6e03987`:

- `rustmath-curves` plane-curve layer: geometric genus `genus.rs:30` [broken: always the arithmetic
  genus], `find_singularities` `singularities.rs:104` [facade: returns no points], differentials
  `differentials.rs:95` [facade], Riemann-Roch dimension `riemann_roch.rs:225` [broken] and basis
  `:297` [facade], parametrization `parameterization.rs:38` [facade], the genus 2 stack (gated off,
  does not compile) [broken]. Only the arithmetic genus `genus.rs:20` [done] is real. There is no
  adjoint computation. WP-F and WP-A need their own small exact routines, specified below.
- `rustmath-functionfields/src/genus.rs:71` [broken]: right on its three tests and wrong elsewhere
  (it assumes the constant field is `Q` and skips branch places at poles).
- The function-field layer of `rustmath-rings`: genus and places
  `function_field/function_field_polymod.rs:156`, orders `order_polymod.rs:190` (`count_points`
  `:334` returns `q^n + 1`), Jacobians `jacobian_base.rs:159` (the Frobenius polynomial is a string),
  zeta function `function_field_rational.rs:339` [all facade].
- Point counts, zeta functions and L-polynomials of curves of genus >= 2: absent. The only point
  count is the genus 1 routine `EllipticCurve::compute_a_p` (`rustmath-ellipticcurves/src/curve.rs:328`).
- Character tables (T-215): inner product `rustmath-groups/src/representation.rs:148` [broken],
  class functions `class_function.rs:211` [facade]. Tables exist only for `C_n` and `S_3`
  (`representation.rs:267`, [partial]). There is no ATLAS table of M23 and no Dixon-Schneider.
- Conjugacy classes `conjugacy_classes.rs:161` [partial]: they need the full element list and cost
  `O(|G|^2)`, so they are unusable at `|M23|`. There are no centralizers, normalizers or backtrack
  searches over a BSGS (tracker, absent list), and no public M23 or M24 constructor. The generators
  are in the `bsgs.rs` tests only. `block_systems` `perm_predicates.rs:182` [broken] misses systems.
- `SteinerSystem::new` (`rustmath-combinatorics/src/designs.rs:228`) checks the design property only
  for `t <= 2` and `v <= 10` (`:63`), so it does not verify S(4,7,23) or S(5,8,24) [row `:235`,
  partial]. Use `t_design_index` and `is_steiner_system` (`designs.rs:1058`, `:1154`) instead:
  exact on a block list, refusal above 10^8 `t`-subsets of blocks [done, Steiner labelling].
- Cohomology: `rustmath-homology` gives free ranks over `Z` only (`chain_complex.rs:145`,
  `cochain_complex.rs:143`, [facade]). There is no group cohomology and nothing over `F_2`.
- Multivariate Newton over `GR(p^N, k)`: absent. `newton_lift_bivariate`
  (`rustmath-rings/src/padics/newton_lift.rs:398`) [no row; IGP24 table item C1 done] lifts a square
  system over `Z_p` from a seed in `F_p` only. `PolySystem::jacobian_mod`
  (`rustmath-polynomials/src/poly_system.rs:153`) works modulo an integer. A seed in `GF(p^15)` needs
  new code. `hensel_lift_root` (`rustmath-rings/src/padics/padic_integer.rs:384`) is [partial].
- Resultants over `GF(p)`: no public entry. `resultant_mod_p` (`disc.rs:110`) is private.
  `UnivariatePolynomial::resultant` needs `EuclideanDomain`, which `PrimeField` and
  `FiniteFieldElement` lack (`GFp<P>` has it). Reduce the integer resultant mod `p` or use `gcd`.
  `bivariate::discriminant_in_t` (`bivariate.rs:248`) panics unless the leading coefficient is
  constant in `t`, which is false for `F`.
- Certified enclosures: absent. The f64 `Interval` (`rustmath-reals/src/interval.rs:186`, [broken],
  blocker B-03) rounds to nearest. Complex intervals are f64
  (`rustmath-complex/src/complex_interval.rs:1`, [partial]) with a non-enclosing `sqrt` (`:393`,
  [broken]). `BigRealBall`, `BigComplexBall` (`ball.rs:50`) and `ComplexBall` (`complex_arb.rs:40`)
  do not widen radii for rounding [partial]. `RIF` sin and cos
  (`rustmath-rings/src/real_mpfi.rs:535`) are [broken]. Partial: `RIF` (`real_mpfi.rs:217`) rounds
  +, -, *, / outward on MPFR, but its sqrt, exp and ln may miss by 1 ulp (tracker row). The exact
  rational balls of `real_arb.rs:139` are sound for +, -, * and have no transcendental functions.
  Aberth roots (`rustmath-complex/src/roots.rs:378`, [done]) bound the roots of one polynomial in
  floating point. Nothing certifies the solutions of a system, so the certificate of 5.6 has nothing
  to build on.
- Period matrices: absent. `rustmath-rings/src/function_field/theta_functions.rs:312` sums an f64
  theta series with no truncation bound [partial].
- Number fields: no factorization over a number field, no subfields, no Galois group as a
  permutation group. Stauduhar descent `rustmath-galois/src/stauduhar.rs:58` [missing] returns `Err`.
  The numeric descent `descent.rs:103` [no row] covers degrees 3 to 5. The legacy `NumberField`
  methods give wrong values (`rustmath-numberfields/src/lib.rs`: norm `:126` [broken], trace `:137`,
  signature `:482`, `galois_closure` `:539` [facade]). Use `round2.rs` and `ideals.rs`.
- `QuotientRing` (`rustmath-polynomials/src/quotient.rs:157`, `:168`, `:273`, `:282`) [facade]. Use
  `staircase.rs:58` and `RationalFunction` division instead.

RustMath rules that apply: a bounded search that finds nothing never decides; certifying
routines return `Decided` or `Unresolved`; reference values come from data files, not from
memory; one crate per pull request.

### What to build

The pieces that do not exist yet, per package (Section 6 says where they go):

- WP-F: a word-size `GF(p^k)` kernel with distinct-root counts for the quartic fibres over the
  `V`-line and for the degree 23 fibres over the `T`-line (3.1 step 6); the nodal point counter on
  P^1 x P^1; the L-polynomial from counts (Newton's identities, integrality, functional equation);
  the exact Weil test on `[-2 sqrt(p), 2 sqrt(p)]`; the mod 2 and mod 23 checks; readers for
  `polynomial_F.m`, `polynomial_calF.m` and the zeta JSON files.
- Steiner labelling exercise (end of Section 6, Section 8 step 2): done (the Status block at the end
  of Section 6). Built: Newton lifting of the 23 roots of `f1` and `f2` in `GR(29^N, 23)`; the two
  cyclic S(4,7,23) on those roots; `T5`, `J5`, `J7`; the degree 253 heptad and duad resolvents of
  `data/golay_resolvents.json`. No stability loop was needed: every value is certified by a
  proven bound (Proposition 5.1), so no rational reconstruction either.
- WP-P: element enumeration from a `StabilizerChain`; a public M23 on the published generators;
  class identification with the split classes; `G0`, `P0`, `P0/G0` and S(4,7,23) in the published
  labelling (the two shift-invariant S(4,7,23) in the cyclic coordinates exist: `cyclic_golay.rs`);
  the ATLAS table of M23 as data with a correct class-function inner product (T-215), and
  the class structure constants from it; centralizer orders; Nielsen class counts, real points and
  ovals; the swap action on Nielsen classes for the twists of 4.4 item 1; parabolic cohomology over
  `F_2`. For 4.4 item 2: a passport-generic high-precision Newton solve, monodromy check and exact verifier
  for the genus 0 rows of degree 23; factorization over a quadratic field for the etale algebra of the
  genus 2 rows; branch cycles by certified continuation; Belyi maps of genus >= 1 for
  the genus 2 rows. Left out: 4.4 item 3 (four branch points). Also needed: readers for the WP-P JSON
  files.
- WP-A: adjoints and the canonical model of the nodal curve; point counts on the model in P^3; the
  rulings of the canonical quadric over `F_p` or `F_{p^2}` and the bidegree (3,3) equation
  `G(s, u)` of A1; the tritangency system; zero-dimensional solving over `F_p` that is known to
  finish at 240 solutions; linear solving with unit pivots and multivariate Newton over
  `GR(p^N, 15)`; the square-class test on `X(F_q)`; the stability loop for rational reconstruction;
  the octic of A5 from the spreads of `data/pg32_a8_model.json`; certified complex enclosures for the
  120 planes; local (Puiseux) expansions at the two totally ramified points for the A0 test of the
  vanishing sequence (0,1,2,3); readers for Le Lay's `Fint_coefficients_Z.json` (3.3 item 1) and for
  `f1`, `f2` (Section 6; a test-only reader is `read_f` in
  `rustmath-galois/tests/steiner_labelling.rs:823`). Left out on purpose: a direct check of the A1
  prediction of one theta characteristic over `F_p`. Its odd part follows from the Frobenius orbits of the 120 planes. The even theta
  characteristic would need divisor classes on X, which no package builds.

## 3. WP-F: the Frobenius table

### Status (2026-11-15)

The WP-F core is done (Section 8, step 1). The owner decided the follow-ups on 2026-11-15:
F-9 is done (Lemma 9.1 of his paper is the reduction certificate, and the delta-sum certificate
is Decided, with its own label); F-5 is closed (p = 23, see below); F-8 is done by the owner at
his main `2d11038` (data and paper); F-1 (p = 3) waits for the model of WP-A step A0.
Of the extensions of 3.3 (see below), 3 and 5 are done; 1 is done by the delta-sum certificate
(no recount on a second model); 2 is done for p = 5, and p = 3 moves to WP-A A0 (F-1). The code
is on branch `claude/m23-wpf` (base `origin/main` `0d277f0`).

The records are in `docs/workstreams/m23_golay/wpf/` (its README lists them): the WP-F design,
reviews, implementation report, docs check and minor fixes in `design/` and `impl/`, and the
F-9 design, design review, implementation report, code review and the check of its minor fixes
in `followup_f9/`. The PARI/GP scripts behind the small-curve and F-9 tests, with their output,
are in `wpf/gp/`.

Code:

- `rustmath-finitefields/src/small_field.rs:55` `SmallFq<K>` (word-size `GF(p^K)`, `p < 2^16`,
  `K <= 8`, `p^K < 2^63`) and `:654` `count_distinct_roots`.
- `rustmath-curves/src/nodal_biprojective/fibres.rs:165` `plane_point_count`, `singular.rs:426`
  `singular_locus`, `local.rs:200` the local classifier, `irreducible.rs:481`
  `absolute_irreducibility`, `certify.rs:231` `certify` and `:449` `normalization_point_counts`,
  `form.rs:183` `reduce_primitive`.
- `rustmath-curves/src/weil.rs:112` `FrobeniusPolynomial::from_point_counts`, `:281` `is_weil`
  (exact, by Sturm sequences), `:347` `reciprocal_halves_mod2`, `:398`
  `two_torsion_frobenius_order`.
- Tests: `rustmath-curves/tests/m23_zeta.rs`, `rustmath-curves/tests/biprojective_small_curves.rs`,
  `rustmath-finitefields/tests/small_field.rs`. Data: `rustmath-curves/tests/data/m23/`
  (`zeta_X.json` and `zeta_L.json` from the owner's main `2d11038`, with his corrected rows,
  `h62.json` from `d05475b`, and Le Lay's CC0 form G, with a README). `polynomial_F.m` and
  `polynomial_calF.m` are read only from `$M23ISGALOIS`.

The counts of X run on G, which is F after a Moebius change of `V`. The test
`env_f_equals_g_by_mobius` checks `F o M = c G` exactly. Two more integral models, G7 and G13, are
built from G at run time for p = 7 and p = 13.

Reproduced (against the owner's main `2d11038`):

- `zeta_X.json`: all 19 rows at k = 4, p = 5 included (`full_x_table_k4`, release), and on
  every row the singular points of the owner's classifier (`x_certificates_equal_the_owners`,
  debug, about 6 s, certificates only).
- `zeta_L.json`: the refusals at (5, 3), (11, 2), (23, 5), (23, 15) and the rows at 7, 19, 31, 41
  and 43, with the owner's singular points (`env_l_table`). An L row (p, s0) is the degree-one
  prime (p, s - s0) of L.
- The anchor at 53: 45, 52, 57, all three Decided (`env_anchor_53`).
- The owner's recount of RustMath's rows at 83, 97 and 107 (`data/rustmath_crosscheck.json`;
  `full_x_owner_recount_83_97_107`, release; the four `full_` tests take about 7 min together).
- The derived checks of 3.1 step 6 on every counted row: the exact Weil test,
  `P(chi(p)) = P(chi(p) p) = 0 mod 23` and `P mod 2 = Q Q^rev`
  (`reference_polynomials_pass_the_derived_checks` on the reference rows; `full_x_table_k4`,
  `env_l_table`, `x_at_7_k6_and_11_k5_nodal` and the example `m23_zeta` on the counted rows).
  Both fibrations give the same counts
  (`both_fibrations_and_brute_force`, `full_both_fibrations_k3`).

Statuses. The certificate is Lemma 9.1 of the owner's paper (`paper/golay_m23.tex` at `2d11038`,
`\label{lem:reduction}`): if `F mod p` keeps its bidegree, is absolutely irreducible, and the
delta invariants of its singular points over `F_p-bar` add up to `(a - 1)(b - 1) - g`, the curve
has good reduction at p and the counts of the normalization are those of its reduction. All three
conditions are checked. Absolute irreducibility (`absolute_irreducibility`, new with F-9): no line
of a ruling, then a fibre that is irreducible of full degree over the residue field of its base
point; for X and the covers `gcd(4, 23) = 1` finishes the proof. Two kinds of Decided, with their
own labels: `Decided: nodal (62 nodes)` and `Decided: delta-sum, Lemma 9.1 (61 nodes, A2)`.
Unresolved means an unclassified point, a delta-sum above `p_a - g`, or absolute irreducibility
not shown. A line of a ruling, or a drop of the bidegree, is a refusal of the model. The genus 4
is a trusted input.

- X, Decided, nodal: 7 (G7), 11, 17, 19, 29, 37, 43, 47, 53, 59, 61, 67, 71, 79, 83, 89, 97, 101,
  103, 109, 113, 127, 131.
- X, Decided, delta-sum: 5 (60 nodes and a tacnode with conjugate branches; no `F_5`-rational
  fibre is irreducible, the fibre over the root of `y^2 + 2` is), 13 (G13), 31, 41, 73 (61 nodes
  and one ordinary cusp each), 107 (60 nodes and a tacnode with rational branches).
- So, by Lemma 9.1, X has good reduction at every prime from 5 to 131 except 23: at 5, 13, 31,
  41, 73 and 107 by the delta-sum certificate, at the others by the nodal one. The owner's paper
  at `2d11038` uses the same lemma for its table and now gives `K_W` unramified outside
  `{2, 3, 23}` (Theorem 4.8; Section 1).
- L, Decided, nodal: (7, 5), (19, 3), (41, 8), (43, 15), (47, 13), (53, 21), (53, 36), (53, 41),
  (61, 52), (67, 15), (67, 42), (71, 37), (71, 67), (79, 40), (79, 46), (89, 57), (89, 69),
  (97, 76), (101, 74), (101, 98), (127, 63), (127, 74).
- L, Decided, delta-sum: (31, 12) (57 nodes, an ordinary triple point and a tacnode with
  conjugate branches), (79, 15), (103, 93) (60 nodes and a tacnode with rational branches).
- Refused, because the discriminant vanishes identically mod p: X at 3 and 23 for every model
  (`x_at_3_and_23_refused`); L at (5, 3), (11, 2), (23, 5), (23, 15). The refusal at 23 is final
  (F-5, closed by the owner's decision): by Le Lay's Section 5 X acquires good reduction at 23
  only over a ramified extension, so there is no model over `Z_(23)` with smooth special fibre
  to count on. The refusal at 3 stays until WP-A step A0 gives another model to test at 3
  (F-1); whether X has good reduction at 3 is open (the paper's Question 10.1).
- The owner's corrected rows (F-8, done by the owner at `2d11038`): X at 5, `7, 37, 169, 561`,
  `a = (1, 6, 20, 16)`, and L at (31, 12), `43, 1017, 30265, 923729`, `a = (11, 88, 682, 4438)`.
  The rows they replace (`8, 36, 170, 560`; `44, 1014, 30266, 923726`) are what the
  assumed-unibranch rule gives; the tests keep them as such
  (`x_at_5_tacnode_with_conjugate_branches`, `env_l_table`, `env_l31_k5`).
- Regression: a reducible reduction is never Decided (`reducible_reductions_are_never_decided`:
  two (2,2) curves meeting in 8 nodes, a curve times its conjugate over `F_169`, a line
  component; each was `Decided: nodal` before F-9).

The full tables, with times, are in `wpf/impl/WPF.md` Section 3 (counts) and
`wpf/followup_f9/F9_impl.md` Section 3 (statuses after F-9, with the irreducibility witness and
the owner's singular-point labels of every row). The counts did not change.

Order of Frobenius on `Jac(X)[2]` (3.3 item 5): 15 and Decided at X for p = 11, 31, 79, 97 and at
L for (53, 41), (67, 15). Elsewhere only bounds (`weil.rs:398`, `two_torsion_orders`).

Extensions of 3.3:

1. D flags: explained and certified. Each of 13, 31, 41 and 73 has one ordinary cusp, and the
   delta-sum certificate makes the rows Decided. Not recounted on a second model: Le Lay's
   (8,23) model does not fit the point counter (fibre degree 2 to 4 is needed; its fibre
   degrees are 8 and 23).
2. p = 5: Decided, good reduction (delta-sum certificate). p = 3: refused by every model; it
   waits for the model of WP-A step A0 (F-1).
3. More primes: X to 131, L to 127.
4. Q versus Q^rev: not attempted (WP-A, F-7).
5. Done as above.

Follow-ups (`wpf/impl/WPF.md`, Section 7, and `wpf/followup_f9/F9_impl.md`, Section 7): F-1
p = 3 on the model of WP-A step A0; F-2 Le Lay's model as a cross-check; F-4 a faster quartic
root count above p = 131; F-5 closed (p = 23 refused for good, the owner's decision);
F-6 a CC0 route for the L covers; F-7 Q versus Q^rev; F-8 done by the owner at `2d11038` (data
and paper); F-9 done; F-10 JSON output of the example; F-11 a test of an ordinary m-fold point
over a residue field of degree > 1 (the `A_k` part is closed: degree-3 nodes, A3 at a degree-2
place and A15, in both branch classes, `biprojective_small_curves.rs`).

The six minor items of the code review (`wpf/impl/WPF.review1.md`, Section 3) are fixed
(`wpf/impl/WPF_minors.md`): a test of the Unclassified path (`unclassified_e6_points`), the cost
and the panics in the docs, the chunk sum of `plane_point_count` as a reduction,
`GenusInconsistent` panics in the drivers instead of counting as a refusal, and the MAGMA chapter
in the headers of `certify.rs` and `residue.rs`, and the F-11 tests above. The checks of those fixes added two overflow
guards: a count above `u64::MAX` in `plane_point_count` is `FieldTooLarge`, and an order of
Frobenius on `Jac[2]` above `u64::MAX` (possible from genus 64 on) panics; neither wraps.

Commands:

- `cargo test -p rustmath-curves --test m23_zeta` (fast set, debug, about 10 s);
- `cargo test -p rustmath-curves --test biprojective_small_curves` (debug, about 2 s), and
  `cargo test --release -p rustmath-curves --test biprojective_small_curves -- --ignored`
  (`node_of_degree_3_up_to_k6`, about 2 s);
- `cargo test --release -p rustmath-curves --test m23_zeta -- --ignored full_`;
- `M23ISGALOIS=<checkout of shaowuz/m23isgalois> cargo test --release -p rustmath-curves --test m23_zeta -- --ignored env_`;
- `cargo run --release -p rustmath-curves --example m23_zeta -- <pmin> <pmax> <kmax>`, with
  `--cover L` for the covers over L and `--reference-rule` for the counts of the
  assumed-unibranch rule (never certified).

Three older lines of this file are now out of date. The JSON files are vendored (the opening
paragraph says nothing is). WP-F uses the owner's main `2d11038` for `data/zeta_X.json`,
`data/zeta_L.json` and the paper, not the pinned `d05475b`. Section 2's "Point counts, zeta
functions and L-polynomials of curves of genus >= 2: absent" no longer holds for curves in
`P^1 x P^1` with fibre degree 2 to 4, odd `p < 2^16`, `k <= 8` and `p^k < 2^63`.

### 3.1 Specification

Input: the coefficients `F[i][j]` of `T^i V^j` (0 <= i <= 4, 0 <= j <= 23) and a prime p.
Output: `N_k = #X(F_{p^k})` for k = 1..4, the Frobenius polynomial
`P_p(T) = T^8 + a1 T^7 + a2 T^6 + a3 T^5 + a4 T^4 + p a3 T^3 + p^2 a2 T^2 + p^3 a1 T + p^4`,
and three derived checks.

Algorithm (reference: `m23golay/curve.py`, `count_points` with `fibre="V"`):

1. Fibre the plane model over the `V`-line. For each `v` in `P^1(F_q)` the fibre polynomial
   `F(T, v)` is a quartic in `T`. Count its distinct roots in `P^1(F_q)` (a drop of degree
   counts as a root at infinity).
2. For every repeated root `(t, v)`, decide whether it is a singular point of the model (both
   partial derivatives vanish). If it is smooth, it is a ramification point of the projection
   and counts once. If it is singular, compute the Hessian discriminant
   `F_TV^2 - F_TT F_VV`: nonzero square means a node with two rational branches (count 2),
   nonzero nonsquare means a node with conjugate branches (count 0), zero means the singular
   point is not a node (see 3.3).
3. Use Frobenius orbits of the base point: process only the orbit representative with the
   smallest code and weight by the orbit length. The reference implementation is stateless per
   chunk so that it parallelizes and bounds memory.
4. From `s_k = p^k + 1 - N_k` compute the elementary symmetric functions by Newton's
   identities, check that they are integers, and fill the upper half by the functional equation.
5. Weil check. The reference uses numerical roots. RustMath should do it exactly: write
   `P_p(T) = T^4 R(T + p/T)` with `R` a monic integer quartic and verify with a Sturm sequence
   that `R` has four real roots in `[-2 sqrt(p), 2 sqrt(p)]`.
6. Derived checks, all of which hold in the reference data:
   - `P_p(chi(p)) = P_p(chi(p) p) = 0 mod 23`, with `chi` the quadratic character of
     `Q(sqrt(-23))` (the 23-torsion class of the two totally ramified points);
   - `P_p mod 2 = Q Q^rev` for a quartic `Q` over `F_2` (the Golay Lagrangian);
   - both fibrations (over `V` with quartic fibres, over `T` with degree 23 fibres) give the
     same counts.

### 3.2 Reference values

The two tables below are those of `d05475b`, as the brief first gave them. After WP-F the owner
corrected the row at p = 5 (now `7, 37, 169, 561`, a Weil polynomial) and the L row at
(31, 12) (now `43, 1017, 30265, 923729`), and gave every row the singular points of his
classifier, which explain the D flags (follow-up F-8, main `2d11038`). RustMath tests against
`2d11038` (Status above). At `2d11038` the paper certifies every row by Lemma 9.1, and its
proof of Theorem 4.8 uses the good reduction at 5, 7 and 11.

`T` = model stays nodal mod p; `D` = the plane model has a non-nodal singular point mod p and the
count treats it as unibranch (accepted because it passes the Weil test, not proved).

| p | N1 | N2 | N3 | N4 | (a1, a2, a3, a4) | P_p mod 2 | model |
|---:|---:|---:|---:|---:|---|---|---|
| 5 | | | | | not obtained: not a Weil polynomial | | |
| 7 | 11 | 69 | 383 | 2433 | (3, 14, 46, 138) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 11 | 15 | 133 | 1368 | 14905 | (3, 10, 33, 145) | `(x^4+x+1) (x^4+x^3+1)` | T |
| 13 | 11 | 161 | 2135 | 29089 | (-3, 0, -12, 188) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | D |
| 17 | 21 | 285 | 4923 | 82985 | (3, 2, 0, -130) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 19 | 17 | 405 | 7265 | 130913 | (-3, 26, 66, 74) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 29 | 35 | 789 | 24773 | 708785 | (5, -14, 16, 1060) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 31 | 33 | 993 | 29640 | 926713 | (1, 16, -35, 875) | `(x^4+x+1) (x^4+x^3+1)` | D |
| 37 | 43 | 1433 | 50881 | 1881609 | (5, 44, 254, 3156) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 41 | 42 | 1780 | 69408 | 2827728 | (0, 49, 162, 1692) | `(x+1)^4 (x^2+x+1)^2` | D |
| 43 | 44 | 1996 | 80192 | 3422024 | (0, 73, 228, 3470) | `(x+1)^4 (x^2+x+1)^2` | T |
| 47 | 47 | 2349 | 104321 | 4882281 | (-1, 70, 96, 2934) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | T |
| 53 | 54 | 2804 | 148704 | 7881808 | (0, -3, -58, -2164) | `(x+1)^4 (x^2+x+1)^2` | T |
| 59 | 72 | 3560 | 205056 | 12120780 | (12, 111, 648, 3991) | `(x^4+x^3+x^2+x+1)^2` | T |
| 61 | 64 | 3740 | 226852 | 13855856 | (2, 11, -24, 2476) | `(x+1)^4 (x^2+x+1)^2` | T |
| 67 | 80 | 4460 | 302192 | 20163724 | (12, 57, 584, 8759) | `(x^4+x^3+x^2+x+1)^2` | T |
| 71 | 68 | 5048 | 358256 | 25432076 | (-4, 11, 92, 4679) | `(x^4+x^3+x^2+x+1)^2` | T |
| 73 | 83 | 5541 | 390089 | 28393065 | (9, 146, 1428, 12030) | `(x+1)^2 (x^3+x+1) (x^3+x^2+1)` | D |
| 79 | 77 | 6369 | 491636 | 38964585 | (-3, 68, -663, 7335) | `(x^4+x+1) (x^4+x^3+1)` | T |

Certificate for good reduction at 7 and 11 (the two primes the proof uses):
`scripts/verify_curve_model.py` checks the discriminant identity over Z at 190 integers, that
`h62` stays squarefree of degree 62 and prime to `T(T^2+23)` mod p, and that the model has a node
above each of its 62 roots.

Covers over the sextic field `L = Q[s]/(s^6 - s^5 - 3s^4 - 3s^3 + s^2 + 5s + 4)`, model
`polynomial_calF.m`, reduced at the degree-one prime `(p, s - s0)`:

| p | s0 | N1 | N2 | N3 | N4 | (a1, a2, a3, a4) | Q up to reversal |
|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 3 | | | | | model degenerate at this prime | |
| 7 | 5 | 7 | 45 | 343 | 2545 | (-1, -2, 2, 38) | `x^4+x^3+x^2+1` |
| 11 | 2 | | | | | model degenerate at this prime | |
| 19 | 3 | 15 | 413 | 6849 | 130169 | (-5, 38, -152, 650) | `x^4+x^3+x^2+1` |
| 31 | 12 | 44 | 1014 | 30266 | 923726 | (12, 98, 758, 5021) | `x^4+x^2+1` |
| 41 | 8 | 40 | 1790 | 68866 | 2828954 | (-2, 56, -128, 2402) | `x^4+1` |
| 43 | 15 | 43 | 1829 | 78799 | 3422769 | (-1, -10, -226, 1278) | `x^4+x^3+x^2+1` |

Published anchor: at 53 the three degree-one primes give 45, 57, 52 points over F_53; the values
45 and 52 are in HJLPPZ Proposition 3.14. This is in `tests/test_core.py`.

### 3.3 Extensions wanted

1. **Remove the D flags.** (Status: explained, one ordinary cusp each, and Decided by the
   delta-sum certificate of Lemma 9.1; no second model was counted.) At p = 13, 31, 41, 73
   recount on a second model with different singular points and reconcile. Candidates: Le Lay's
   bidegree (8,23) model (`data/Fint_coefficients_Z.json` in his repository, 84 nodes) or a
   plane projection of the canonical model. Expected: the same `N_k`.
2. **Decide p = 3 and p = 5.** (Status: p = 5 is Decided, good reduction, by the delta-sum
   certificate; p = 3 waits for the model of WP-A step A0, follow-up F-1.) The reference count
   at 5 fails the Weil test, which says only that this plane model is too degenerate there. Find
   a model that is nodal mod p with the right number of nodes, or show that the canonical model
   has singular reduction. This matters for WP-A: X has good reduction at 5, so the A8 field is
   unramified outside `{2,3,23}`; good reduction at 3 as well would give `{2,23}`.
3. **More primes.** p up to about 127. The cost is dominated by k = 4 (`q = p^4` base points,
   one quartic root count each).
4. **Resolve `Q` versus `Q^rev`.** The table gives the characteristic polynomial of Frobenius
   on `W` only up to reversal. It is decided by WP-A (the degree 15 polynomial distinguishes
   points from hyperplanes).
5. **Order of Frobenius on all of `Jac(X)[2]`.** Needed to choose the working prime in WP-A.
   When `P_p mod 2` is squarefree (p = 11, 31, 79) the order is 15.

Suggested home: a new module `rustmath-curves/src/nodal_biprojective.rs` (point counts on a
nodal curve in P^1 x P^1 given by a bihomogeneous form) and `rustmath-curves/src/weil.rs`
(L-polynomial from counts, exact Weil test, mod l factorization). MAGMA reference: Handbook
chapter 114 (`ZetaFunction`, `LPolynomial`, `NumberOfPointsOnCurve`) and chapter 42 for the
function-field versions. The method here is naive counting, which is all genus 4 needs.

Acceptance: exact equality with every row of the two tables, the three derived checks, and the
value 45, 52, 57 at 53.

## 4. WP-P: three-point class vectors of M23

### 4.1 What "Phase 6" asked for and what now exists

The research note's Phase 6: enumerate the class vectors of M23 that can carry a cover defined
over Q, use braid-orbit and real-structure data to shortlist, then compute. For three branch
points the group-theoretic part is done and is the reference for RustMath.

A three-point class vector can carry a cover over Q in two ways (branch cycle lemma):

- `rational`: three rational classes at three rational points. The rational classes of M23 are
  `2A, 3A, 4A, 5A, 6A, 8A`.
- `conjugate_pair`: a rational class `R` at a rational point and a pair `D, D^-1` of irrational
  classes at the two conjugate points of the quadratic field of `D`. The pairs are `7AB`
  (`Q(sqrt(-7))`), `11AB` (`Q(sqrt(-11))`), `14AB` (`Q(sqrt(-7))`), `15AB` (`Q(sqrt(-15))`),
  `23AB` (`Q(sqrt(-23))`). HJLPPZ is `(2A, 23A, 23B)`.

Not yet treated: a repeated rational class placed at two conjugate points of an arbitrary
quadratic field (a twist of the `rational` rows that have a repeated class by the swap of the two
points), and four or more branch points. See 4.4.

For each of the 57 class vectors with nonnegative genus, `data/m23_three_point_nielsen.json`
gives:

- `structure_constant` n: from the character table,
  `n = |C1||C2||C3| / |G|^2 * sum over chi of chi(c1) chi(c2) chi(c3) / chi(1)`;
- `size`: the number of inner classes of generating triples, which is the degree over Q of the
  Hurwitz scheme;
- `real`: the number of classes fixed by complex conjugation, which is the number of real points
  of the Hurwitz scheme;
- `genus23`: the genus of the degree 23 cover.

| type | class vector | genus | n | size | real |
|---|---|---:|---:|---:|---:|
| conjugate_pair | (2A, 7A, 7B) | 0 | 43/2 | 0 | 0 |
| conjugate_pair | (2A, 23A, 23B) | 4 | 7 | 7 | 3 |
| conjugate_pair | (2A, 14A, 14B) | 2 | 43/2 | 17 | 3 |
| conjugate_pair | (2A, 11A, 11B) | 2 | 31 | 22 | 4 |
| conjugate_pair | (2A, 15A, 15B) | 2 | 27 | 24 | 2 |
| conjugate_pair | (3A, 7A, 7B) | 2 | 349 | 100 | 8 |
| conjugate_pair | (3A, 23A, 23B) | 6 | 134 | 134 | 16 |
| conjugate_pair | (3A, 15A, 15B) | 4 | 830/3 | 264 | 20 |
| conjugate_pair | (3A, 14A, 14B) | 4 | 349 | 303 | 15 |
| conjugate_pair | (3A, 11A, 11B) | 4 | 466 | 344 | 18 |
| conjugate_pair | (4A, 23A, 23B) | 7 | 660 | 660 | 42 |
| conjugate_pair | (4A, 7A, 7B) | 3 | 1951 | 995 | 21 |
| conjugate_pair | (5A, 23A, 23B) | 8 | 1422 | 1422 | 32 |
| conjugate_pair | (4A, 15A, 15B) | 5 | 1496 | 1480 | 54 |
| conjugate_pair | (6A, 23A, 23B) | 8 | 1632 | 1632 | 72 |
| conjugate_pair | (4A, 14A, 14B) | 5 | 1951 | 1879 | 55 |
| conjugate_pair | (5A, 7A, 7B) | 4 | 3756 | 1994 | 22 |
| conjugate_pair | (8A, 23A, 23B) | 9 | 2272 | 2272 | 0 |
| conjugate_pair | (4A, 11A, 11B) | 5 | 2772 | 2390 | 58 |
| conjugate_pair | (5A, 15A, 15B) | 6 | 16982/5 | 3381 | 43 |
| conjugate_pair | (6A, 7A, 7B) | 4 | 4371 | 3470 | 40 |
| conjugate_pair | (5A, 14A, 14B) | 6 | 3756 | 3627 | 37 |
| conjugate_pair | (6A, 15A, 15B) | 6 | 11668/3 | 3859 | 103 |
| conjugate_pair | (6A, 14A, 14B) | 6 | 4371 | 4328 | 102 |
| conjugate_pair | (8A, 7A, 7B) | 5 | 5908 | 4848 | 0 |
| conjugate_pair | (5A, 11A, 11B) | 6 | 5658 | 4874 | 54 |
| conjugate_pair | (8A, 15A, 15B) | 7 | 5424 | 5424 | 0 |
| conjugate_pair | (8A, 14A, 14B) | 7 | 5908 | 5888 | 0 |
| conjugate_pair | (6A, 11A, 11B) | 6 | 7056 | 6750 | 96 |
| conjugate_pair | (8A, 11A, 11B) | 7 | 10080 | 9648 | 0 |
| rational | (2A, 8A, 8A) | 0 | 491/8 | 28 | 0 |
| rational | (3A, 5A, 5A) | 0 | 1310/3 | 30 | 0 |
| rational | (3A, 4A, 8A) | 0 | 441/2 | 54 | 0 |
| rational | (3A, 5A, 6A) | 0 | 863/3 | 70 | 4 |
| rational | (4A, 4A, 5A) | 0 | 909 | 104 | 2 |
| rational | (3A, 6A, 6A) | 0 | 1757/4 | 120 | 16 |
| rational | (3A, 5A, 8A) | 1 | 476 | 170 | 0 |
| rational | (4A, 4A, 6A) | 0 | 1713/2 | 192 | 14 |
| rational | (3A, 6A, 8A) | 1 | 634 | 350 | 0 |
| rational | (4A, 5A, 5A) | 1 | 1936 | 396 | 4 |
| rational | (3A, 8A, 8A) | 2 | 889 | 568 | 0 |
| rational | (4A, 4A, 8A) | 1 | 1170 | 568 | 0 |
| rational | (4A, 5A, 6A) | 1 | 1757 | 776 | 22 |
| rational | (5A, 5A, 5A) | 2 | 61058/15 | 972 | 0 |
| rational | (4A, 6A, 6A) | 1 | 8691/4 | 1220 | 56 |
| rational | (4A, 5A, 8A) | 2 | 2712 | 1484 | 0 |
| rational | (5A, 5A, 6A) | 2 | 3836 | 2178 | 24 |
| rational | (4A, 6A, 8A) | 2 | 6615/2 | 2534 | 0 |
| rational | (5A, 5A, 8A) | 3 | 5504 | 3288 | 0 |
| rational | (5A, 6A, 6A) | 2 | 14417/3 | 3416 | 90 |
| rational | (4A, 8A, 8A) | 3 | 10305/2 | 4208 | 0 |
| rational | (6A, 6A, 6A) | 2 | 69737/12 | 4860 | 114 |
| rational | (5A, 6A, 8A) | 3 | 7140 | 5666 | 0 |
| rational | (6A, 6A, 8A) | 3 | 8590 | 7940 | 0 |
| rational | (5A, 8A, 8A) | 4 | 10680 | 8816 | 0 |
| rational | (6A, 8A, 8A) | 4 | 13335 | 12584 | 0 |
| rational | (8A, 8A, 8A) | 5 | 18720 | 17760 | 0 |

Of the 57 class vectors, 1 has no generating triple, 22 have covers but no real point, and 34 have real points.

Consequences already visible:

- Every class vector containing `8A` has no real point, hence no rational point. Reason: the
  eight elements of M23 that invert an element of order 8 all have order 4, and a real structure
  needs an inverting involution.
- `(2A, 7A, 7B)` has no generating triple.
- `(2A, 8A, 8A)`, `(3A, 4A, 8A)`, `(3A, 5A, 5A)` have genus 0 and no real point.
- The smallest live targets are `(2A, 14A, 14B)` (17 covers of genus 2, 3 real, branch points
  `0, +-sqrt(-7)`), `(2A, 11A, 11B)` (22 covers, 4 real, `+-sqrt(-11)`), `(2A, 15A, 15B)`
  (24 covers, 2 real, `+-sqrt(-15)`), then `(3A, 5A, 6A)` (70 covers of genus 0, 4 real) and
  `(4A, 4A, 5A)` (104 covers of genus 0, 2 real).
- Odd size forces a real point. The class vectors of odd size are: `(2A,23A,23B)` (7), `(2A,14A,14B)` (17), `(3A,14A,14B)` (303), `(4A,7A,7B)` (995), `(4A,14A,14B)` (1879), `(5A,15A,15B)` (3381), `(5A,14A,14B)` (3627), `(6A,15A,15B)` (3859).

Whether any of these Hurwitz schemes other than `(2A,23A,23B)` has a rational point is not known
to us. A rational point would be a regular M23 extension of Q(t) different from HJLPPZ's. The
literature should be checked before any claim: Haefner, arXiv:2202.08222 (braid orbits for M23),
Koenig's dissertation (Hurwitz spaces for M23), Malle and Matzat, Inverse Galois Theory.

### 4.2 How the reference was computed (and what RustMath must reproduce)

Reference: `scripts/passports_all.py`, six minutes in Python with numpy.

1. **Elements.** Enumerate all 10,200,960 elements from a stabilizer chain (base of 6 points,
   orbit lengths 23, 22, 21, 20, 3, 16 for the published generators) as rows of a
   `u8` array. RustMath: do the same; 235 MB.
2. **Cycle types.** The triple `(Fix g, Fix g^2, Fix g^3)` separates the 12 cycle types.
3. **Split classes.** For a pair `D, D^-1`:
   - orders 7, 14, 15, 23: the trace of `g` on the even Golay code over `F_2` differs on `D` and
     `D^-1`. With a systematic basis `v_k` (pivot coordinates `i_k`),
     `tr(g) = sum over k of v_k[g^-1(i_k)]`.
   - order 11: the trace does not separate. Label each 11-cycle of `g` by `Z/11` through `g`
     and record the positions of the heptads on it; this family of subsets is the invariant.
     It is preserved by `g -> g^k` exactly when `k` is a square mod 11.
   - order 23 has a second, independent invariant: label the points by `Z/23` through `g`; the
     Steiner system becomes one of the two cyclic systems (generator polynomial
     `z^11+z^9+z^7+z^6+z^5+z+1` or its reciprocal).
4. **Counting.** Fix `b` in the class with the smallest centralizer. Survivors are the `a` in
   `C1` with `a b` in the right class. Then `#survivors = n |C(b)|` exactly; the reference asserts
   this for all 57 rows, which checks the class separation and the character table against each
   other.
5. **Generation.** A subgroup of M23 containing an element of order not in `{1, 11, 23}` is M23
   if and only if it is transitive on the 23 points, because every maximal subgroup other than
   `23:11` is intransitive. So `size = #(transitive survivors) / |C(b)|`.
6. **Real points.**
   - `conjugate_pair`, triple `(g1, g2, g3)` with `g2 = b`: complex conjugation sends it to
     `(g1^-1, g3^-1, g2^-1)`. Real classes correspond to involutions `c` with
     `a := c b c b^-1` in `R` and `<a, b> = M23`; divide the count by `|C(b)|`.
   - `rational`: real classes correspond to pairs `(a, b)` for which some involution inverts
     both. Enumerate the involutions `c` inverting `b` and `a = c u` with `u` an involution.
7. **Ovals** (for `(C, 23A, 23B)` only, `m23golay/nielsen.py`, `real_components_general`):
   `A = Fix(c0)`, `B = Fix(c0 g1)`; join `x` in `A` to `x^(g2^-1)` in `B`; on each `g1`-cycle
   join the points of `A` and `B` (pattern (1,1) for odd length, (2,0) or (0,2) for even length).
   The number of cycles of this graph is the number of components of `X(R)`. Check:
   `dim Jac(X)(R)[2] = g + r - 1`, computed separately from the action of conjugation on
   parabolic cocycles.

RustMath needs for this: element enumeration from a BSGS, class identification as above, and a
correct character table for the cross-check. Either fix the class-function layer (tracker T-215)
and store the ATLAS table of M23 as data, or compute `n` by the counting in step 4 and compare
with `data/m23_three_point_passports.json`. MAGMA reference: chapter 58 (permutation groups,
classes), chapter 91 (characters), chapter 68 for the cohomology in 4.3.

Acceptance: all 57 rows of `data/m23_three_point_nielsen.json`; the seven published triples,
the seven rows of Janik's table and Le Lay's seven representatives each map bijectively onto the
seven classes of `(2A,23A,23B)` (dictionary in `data/group_facts.json`, key `nielsen`).

### 4.3 The Golay data per cover

For every triple of every class vector with a degree 23 cover of genus g, the parabolic
cohomology `H^1(P^1, j_* G0)` has dimension g and sits in `H^1(P^1, j_* P0)` of dimension 2g
(reference: `m23golay/cohomology.py`). Cocycles are pairs `(c1, c2)` in the module, with
`c3 = c1 g2 g3 + c2 g3`; coboundaries are `(a(1+g1), a(1+g2))`; parabolic means
`c_i in M(1+g_i)`. This is linear algebra over `F_2` with 11 and 22 dimensional modules.
RustMath should reproduce `dim W = g` for all seven triples of `(2A,23A,23B)` and all
`(C,23A,23B)`, and the dimensions of the fixed spaces of conjugation in
`data/group_facts.json` and `data/passports_C_23A_23B.json`.

### 4.4 Next steps after the table (the actual Phase 6)

In order of cost:

1. **Twists.** For `rational` rows with a repeated class `(R, D, D)`, compute the action of the
   swap of the last two entries on the Nielsen class. The Hurwitz scheme of
   `(R; D at sqrt(d), D at -sqrt(d))` is the twist by `Q(sqrt(d))`. Report the numbers of orbits
   and of fixed classes, and the real points for `d < 0` (same formula as `conjugate_pair` with
   `D^-1 = D`).
2. **Compute the small Hurwitz schemes.** For `(2A,14A,14B)`, `(2A,11A,11B)`, `(2A,15A,15B)`
   the covers have degree 23 and genus 2 with three branch points. Compute all covers in the
   class numerically, then the etale algebra of the Hurwitz scheme (as HJLPPZ and Le Lay did for
   seven covers: an invariant of the cover, its minimal polynomial over the quadratic field,
   factor degrees). Any degree-one factor is a candidate rational point. The existing Belyi
   engine is genus 0; genus 2 needs the modular-forms route with a basis of weight 2 forms, or
   a different method. The genus 0 rows `(3A,5A,6A)`, `(4A,4A,5A)`, `(3A,6A,6A)`, `(4A,4A,6A)`
   are the natural first run for the genus 0 engine: 70, 104, 120 and 192 Belyi maps of degree 23.
   (Section 2: its encoder and KMSV fit take any passport, but its solve, monodromy and exact
   verifier are written for the degree 24 `[2,12,5]` passport and must be generalized first.)
3. **Four branch points.** Braid orbits on Nielsen classes of length 4, genera of the reduced
   Hurwitz curves. Not started here.

Certification standard before any claim of a new M23 realization: exact Belyi identity, branch
cycles by certified continuation, monodromy group M23, field of moduli Q (the centre of M23 is
trivial, so the cover is then defined over Q).

## 5. WP-A: a polynomial for the A8 field

### 5.1 The structure to exploit (proved in the paper, Proposition 4.5)

Let `Theta` be the set of 256 theta characteristics of X, a torsor under `J[2] = Jac(X)[2]`.
For `theta` in `Theta`, `q_theta(a) = h^0(theta + a) - h^0(theta) mod 2` is a quadratic form on
`J[2]` whose polar form is the Weil pairing. Because `W` is Lagrangian:

- the theta characteristics with `q_theta | W = 0` form one coset `Theta_0` of `W`: 16 even
  theta characteristics, stable under Galois;
- each of the other 15 cosets contains 8 odd and 8 even theta characteristics, and the cosets
  correspond to the nonzero linear forms on `W`, compatibly with Galois.

The canonical model of X is the intersection of a quadric and a cubic in P^3. The quadric is
smooth (its discriminant is 4873 = 11 * 443 up to squares, Le Lay, Proposition 8.22), so the
120 odd theta characteristics are exactly the 120 tritangent planes.

**So the 120 tritangent planes fall into 15 octads, and Galois permutes the octads as A8 = GL_4(2)
permutes the 15 hyperplanes of W.** A symmetric function of each octad is a root of a degree 15
polynomial over Q with Galois group A8 whose splitting field is `K_W`. The octic follows by a
resolvent (5.5).

This was checked numerically only at the level of linear algebra (the coset statement, script
`scripts/verify_group_theory.py`). The tritangent planes themselves have not been computed.

### 5.2 Acceptance tests for the answer

Whatever route is used, the result is a monic `g8` in `Z[x]` of degree 8 (and `g15` of degree 15)
such that:

1. `g8` is irreducible with square discriminant, and its factorization mod p has the cycle type
   in the last column of the table below for every p listed. Cycle types `1 7` and `3 5` at two
   primes, together with the square discriminant, prove that the group is A8 (the only maximal
   subgroup of A8 containing an element of order 15 is `(A5 x 3):2`, of order 360).
2. `g8` has no real root (complex conjugation is a transvection, cycle type `2^4`).
3. The field discriminant of `g8` is divisible only by primes in `{2, 3, 23}` (5 is a prime of
   good reduction of X: Section 3, Status).
4. `g15 mod p` has the factorization type of the matching element of `GL_4(2)` on the 15
   hyperplanes of `F_2^4`.

Frobenius on `W`: characteristic polynomial `Q_p` up to reversal, and cycle type on 8 letters.

| p | Q_p (one of the two) | order of the semisimple part | cycle type on 8 letters |
|---:|---|---:|---|
| 7 | `x^4+x^2+x+1` | 7 | `1 7` |
| 11 | `x^4+x+1` | 15 | `3 5` |
| 13 | `x^4+x^2+x+1` | 7 | `1 7` |
| 17 | `x^4+x^2+x+1` | 7 | `1 7` |
| 19 | `x^4+x^2+x+1` | 7 | `1 7` |
| 29 | `x^4+x^2+x+1` | 7 | `1 7` |
| 31 | `x^4+x+1` | 15 | `3 5` |
| 37 | `x^4+x^2+x+1` | 7 | `1 7` |
| 41 | `x^4+x^3+x+1` | 3 | `1^2 3^2 or 2 6` |
| 43 | `x^4+x^3+x+1` | 3 | `1^2 3^2 or 2 6` |
| 47 | `x^4+x^2+x+1` | 7 | `1 7` |
| 53 | `x^4+x^3+x+1` | 3 | `1^2 3^2 or 2 6` |
| 59 | `x^4+x^3+x^2+x+1` | 5 | `1^3 5` |
| 61 | `x^4+x^3+x+1` | 3 | `1^2 3^2 or 2 6` |
| 67 | `x^4+x^3+x^2+x+1` | 5 | `1^3 5` |
| 71 | `x^4+x^3+x^2+x+1` | 5 | `1^3 5` |
| 73 | `x^4+x^2+x+1` | 7 | `1 7` |
| 79 | `x^4+x+1` | 15 | `3 5` |

How the last column follows from `Q_p`: irreducible quartic of order 15 gives `3 5`;
`(x+1)(x^3+x+1)` or its reverse gives `1 7`; `x^4+x^3+x^2+x+1` gives `1^3 5`;
`(x+1)^2(x^2+x+1)` gives `1^2 3^2` or `2 6`.

A rigorous link between a candidate `g15` and `K_W` needs more than these tests; see 5.6.

### 5.3 Route A (recommended): tritangent planes, exact and p-adic

Everything here uses layers RustMath already has: `GF(p^k)`, Galois rings, Hensel lifting,
linear algebra, rational reconstruction.
Section 2 lists what these layers still lack here (multivariate Newton over `GR`, a stability loop).

**A0. Canonical model of X over Q.** Two sources; use one and test with the other.

- From `F`: the canonical differentials are `A(T,V) dT / F_V` where `A` has bidegree at most
  (2, 21) and vanishes at the 62 nodes. That is 66 unknowns and 62 conditions, so a 4 dimensional
  space. The nodes are the points `(tau, s(tau))` with `h62(tau) = 0`, and `s` is found from the
  gcd of `F(tau, V)` and `F_V(tau, V)`. Do it modulo many primes (each condition expands over
  the residue field), take the reduced row echelon basis and reconstruct by CRT. Then find the
  quadric (the unique relation among the 10 products) and the cubic (the relation among the 20
  cubic monomials modulo multiples of the quadric), again modularly.
- From Le Lay's data (CC0): `data/canonical_quadric_Q.json` and
  `data/hurwitz_canonical_models_candidate.json` (the degree-one component, over
  `Q(sqrt(-23))`), with the explicit descent to Q in his Remark 5.10. Not validated here.

Tests for A0: the quadric is unique and its discriminant is `4873` times a square; the model is
smooth; the number of points over `F_p` (enumerate `P^3(F_p)`) equals `N_1` of the table in 3.2
for several p; the two totally ramified points have canonical vanishing sequence (0,1,2,3).

**A1. Tritangent planes over a finite field.** Choose p with Frobenius of order 15 on `J[2]`
and good behaviour of the quadric: p = 79 first (the plane model is nodal there, so the
reference counts are certified), then p = 31. Do not start with p = 11: the discriminant of the
canonical quadric has odd valuation at 11 and at 443, so the quadric is singular modulo those
primes (the two rulings coincide) and the P^1 x P^1 description fails there.

A plane `H` is tritangent when `H . X = 2(P1 + P2 + P3)`. One formulation: over a field where the
quadric has rulings, X is a curve of bidegree (3,3) on P^1 x P^1 with equation `G(s,u)`; a
nondegenerate plane section is the graph of a Moebius map `u = mu(s)`; `H` is tritangent exactly
when the binary sextic `(gamma s + delta)^3 G(s, mu(s))` is a constant times the square of a
cubic. With the four coefficients of `mu` and the four coefficients of the cubic as unknowns this
is 7 equations, zero dimensional after removing degenerate `mu`, with 240 solutions (each plane
twice). Solve by a Groebner basis over `F_p` (the field engine is the sound one) or by any other
exact method, then factor the eliminant with Cantor-Zassenhaus.

Predictions that test A1 before anything else is built:

- there are exactly 120 tritangent planes over the algebraic closure;
- at p = 31 and p = 79 they form 8 Frobenius orbits of length 15, so the eliminant has 8
  irreducible factors of degree 15;
- there is exactly one theta characteristic defined over `F_p`, and it is even.

(Reason: Frobenius acts on `J[2]` semisimply with characteristic polynomial
`(x^4+x+1)(x^4+x^3+1)`, without eigenvalue 1, so it has one fixed point on `Theta` and 17 orbits
of length 15 on the rest; 120 = 8 * 15 and 135 = 9 * 15.)

**A2. The octads.** Let `phi` be Frobenius, `O_1, ..., O_8` the eight orbits. `J[2]` has exactly
two `phi`-stable subspaces of dimension 4, `U = ker(phi^4 + phi + 1)` and
`U' = ker(phi^4 + phi^3 + 1)`, and `W` is one of them. A coset of `W` that is not `Theta_0`
contains one plane from each orbit, and `phi` permutes the 15 cosets cyclically. So a block
system is determined by seven phases: fix `x` in `O_1`; for each `i >= 2` there is exactly one
`y` in `O_i` with `x - y` in `U`, and exactly one with `x - y` in `U'`.

Test for `x - y` in `U`: `(1 + phi + phi^4)(x - y) = 0`, which says that
`x + phi(x) + phi^4(x)` and `y + phi(y) + phi^4(y)` are linearly equivalent divisor classes of
degree 9, which says that the function
`l_x l_phi(x) l_phi^4(x) / (l_y l_phi(y) l_phi^4(y))` is a square in the function field of X,
where `l_z` is the linear form of the plane `z`. For `U'` use `phi^3` in place of `phi`.

Deciding squareness without divisor arithmetic: a function `g` defined over `F_q` that is a
square over the algebraic closure is `c h^2` with `c` constant, so `g(P)` lies in one square class
for all `P` in `X(F_q)`. For each of the 15 candidates `y` in `O_i`, evaluate at rational points;
two values in different square classes refute the candidate. Exactly one candidate survives for
`U` and one for `U'`, and it is then proved by elimination. Work over `F_q` with `q = p^15`,
where all planes are defined, and use the points of `X(F_p)` (33 at p = 31, 77 at p = 79) or of
a small extension.

**A3. Lift.** Hensel-lift the 120 planes to the Galois ring `GR(p^N, 15)` by Newton iteration on
the tritangency system (the Jacobian is invertible at simple solutions). The block systems are
determined mod p.

**A4. The degree 15 polynomial.** Fix two rational points `P0, P1` of P^3. For an octad `B` put
`tau_B = sum over planes H in B of l_H(P1) / l_H(P0)` and
`g15(z) = product over the 15 octads of (z - tau_B)`. Its coefficients are fixed by Frobenius, so
they lie in `Z/p^N`.

Which of `U`, `U'` is `W`: exactly one of the two block systems is stable under all of Galois
(or both, if `J[2]` splits as a Galois module, which is not known). For the right one the
coefficients are rational numbers; for the wrong one they are not. This is the same dichotomy,
with the same kind of proof, as the Steiner labelling of roots (paper, Proposition 5.1):
reconstruct rationals from `Z/p^N`, increase `N` until stable, and confirm at a second prime
(79 if the first was 31), where the computation is independent. Then reduce `g15` with polred.
The Steiner labelling built two pieces for this step: `orbit_polynomial`
(`rustmath-finitefields/src/galois_ring_frobenius.rs:460`), which forms `prod (z - tau_B)` and
proves its coefficients lie in `Z/p^N`, and `integrality_dichotomy`
(`rustmath-galois/src/integrality.rs:170`), whose candidates are coefficient vectors. Integral
coefficients with a proven bound need nothing more; rational ones need a rationality dichotomy
with a proven height bound (follow-up SL-6 of the Status block in Section 6).

**A5. The octic.** The roots of `g15` are labelled by the nonzero linear forms on `W`, and the
`F_2`-structure is visible in the cyclic model: `lambda_j = lambda_0 phi^-j`, so
`lambda_i + lambda_j = lambda_k` exactly when `x^i + x^j + x^k = 0` in `F_2[x]/(Q^rev)`, where
`Q` is the characteristic polynomial of `phi` on `W`. In `PG(3,2)` there are 35 lines and 56
spreads (sets of 5 disjoint lines). Call two spreads adjacent when they share exactly one line.
This graph is the Johnson graph J(8,3): the spreads are the 3-subsets of a set of 8 letters, the
28 maximal cliques of size 6 are the pairs of letters, and a letter is a set of 7 such cliques
that pairwise share a spread. The union of those 7 cliques is a family of 21 spreads. This is
done and verified in `scripts/pg32_a8_model.py`; the lines, spreads and the 8 families for both
primitive quartics are in `data/pg32_a8_model.json`, with the check that `GL_4(2)` acts on the
8 letters with order 20160, the Singer cycle with type `3 5` and a transvection with type `2^4`.
Set
`theta_m = sum over spreads S in family m of product over lines L in S of (sum of tau over the points of L)`,
`g8(z) = product of (z - theta_m)`. If two `theta_m` collide, change `P0, P1`.

**A6. Certificates.** See 5.6.

### 5.4 Route B: the fifteen covers of degree 46

By the paper (Corollary 4.4), the nonzero elements of `W` are the fifteen lifts of the HJLPPZ
triple to `2^11:M23` with inertia in the classes of `(g_i, 0)`. As permutations of 46 points they
are in `data/lifts_2_11_M23.json` (cycle types `1^14 2^16`, `23^2`, `23^2`, group order
`2^11 * 10200960`, genus 7; verified). Each is the unramified double cover `Z_w -> X`.

Computing these fifteen three-point covers numerically and recognizing a common invariant gives
`g15` directly (the action on points of `P(W)`; route A gives the action on hyperplanes; the two
degree 15 fields are arithmetically equivalent and not isomorphic, and both have Galois closure
`K_W`). The triangle group is `Delta(2,23,23)`, the subgroup has index 46, and the space of
weight 2 forms has dimension 7: 4 from X and 3 Prym differentials with the quadratic character.
This needs the modular-forms route beyond genus 0, so it is a stress test of the Belyi engine and
an independent confirmation, not the first choice.

### 5.5 Route C: periods and theta functions

Compute a period matrix of X to several hundred digits, identify `W` inside `H_1(X, F_2)` from
the monodromy (the cocycle description in 4.3 evaluates on lifted loops), and get the tritangent
planes from gradients of theta functions with odd characteristics. RustMath has no period
matrices and no theta functions (in the handbook digests they appear only in chapter 125, for
hyperelliptic curves), so
this is the most expensive route. It is listed for completeness.
(Section 2: `rustmath-rings` has an f64 theta series with no error bound, and no period matrices.)

### 5.6 What would make the result a theorem

Tests 1 to 4 of 5.2 show that `g8` defines an A8 field with the right local behaviour at 17
primes. That is strong evidence, not a proof that the field is `K_W`. For a proof:

- A0 must be exact (it is: linear algebra over Q, verified by substitution in the function field
  `Q(T)[V]/(F)`);
- the tritangency system must be shown to have exactly the 120 lifted solutions (degree count
  plus simplicity mod p);
- the statement "`g15` vanishes on `tau`" must hold exactly, not to precision `p^N`. Use the
  argument of the paper's Proposition 5.1: `g15(tau_B)` is an algebraic number of bounded degree
  with archimedean absolute values bounded by certified complex approximations of the 120 planes;
  if it is nonzero its norm has bounded p-adic valuation, so vanishing to a large explicit
  precision proves vanishing. This needs certified complex bounds for the solutions, hence real
  interval or ball arithmetic beyond f64. Until that exists, report the result as
  `Unresolved (verified at two primes and 17 Frobenius classes)`.
  (Section 2: a real interval type on MPFR exists; certified complex enclosures do not.)

### 5.7 Risks and unknowns

- The pipeline in 5.3 is designed, not executed. Its internal predictions (120 planes, 8 orbits
  of length 15, exactly one surviving phase per orbit, rational coefficients for exactly one of
  the two block systems) are there so that a wrong step fails loudly.
- The Groebner computation in A1 is the main cost. If it is too slow over `F_p`, first project:
  eliminate to a single coordinate of the dual space and factor a degree 240 polynomial.
- If `J[2]` is a split extension, both block systems are rational. That would be a new fact about
  X (a second rational Lagrangian) and both give the same field `K_W`.
- The squareness test assumes the planes are in general position with respect to the evaluation
  points; skip points where a linear form vanishes.

## 6. Shared infrastructure, by crate

| Need | Used by | Suggested home | MAGMA chapter |
|---|---|---|---|
| Point counts on a nodal curve in P^1 x P^1, L-polynomial, exact Weil test | WP-F | `rustmath-curves` (new modules, independent of the facade layer) | 114, 42 |
| Element enumeration from a BSGS; class identification for M23 including split classes | WP-P | `rustmath-groups` | 58 |
| Class structure constants; a correct character table of M23 as data | WP-P | `rustmath-groups` (fix T-215) | 91 |
| Nielsen classes of length 3, complex conjugation, ovals | WP-P | `rustmath-curves/src/belyi/` or a new `rustmath-groups/src/nielsen.rs` | 58, 131 |
| Binary Golay code, S(4,7,23), modules `G0`, `P0`, `P0/G0`, parabolic cohomology over `F_2` | WP-P, WP-A | `rustmath-coding` and `rustmath-groups` | 152, 147, 68 |
| Adjoint computation: canonical model of a nodal curve in P^1 x P^1 | WP-A | `rustmath-curves` | 114 |
| Zero-dimensional solving over `F_p`, Newton lifting in `GR(p^N, k)` | WP-A | `rustmath-polynomials`, `rustmath-finitefields` | 105, 47, 48 |
| Rational reconstruction with stability check, polred | WP-A | `rustmath-integers`, `rustmath-numberfields` (exist, but not the stability loop; Section 2) | 34 |
| Certified complex enclosures (ball arithmetic) | WP-A certificate | `rustmath-reals` | 25 |

The Steiner labelling of roots (paper, Section 5; `m23golay/arith.py`) is a good first exercise
for the Galois-ring layer, with exact expected output: at p = 29 for `f2`, `T5 = -1059840`,
`J5 = 61916`, `J7 = -83030`; for `f1`, `T5 = 92747040`, `J5 = 524492`, `J7 = -2281968`; the
rejected candidate for `T5` has more than 40 digits. The degree 253 resolvents in
`data/golay_resolvents.json` are the larger version of the same test.

### Steiner labelling exercise: Status (2026-11-15)

Done (Section 8, step 2). The change is on the local branch `claude/m23-steiner` (code commit
`5874caf`, on `claude/m23-wpf` `25c6443`; it was built on `d91f880` and rebased after F-9, with
the code unchanged).

The design, its review, the implementation report, the code review and the check of the minor
fixes are in `docs/workstreams/m23_golay/steiner/` (`design/`, `impl/`). The PARI/GP scripts behind
the test values, with their output, are in `steiner/gp/`.

Code (four crates, as the design planned):

- `rustmath-finitefields/src/galois_ring_frobenius.rs:114` `FrobeniusMatrix` (the Frobenius of
  `GR(p^a, D)` as a matrix, by a Newton lift of `w^p`), `:460` `orbit_polynomial`, `:555`
  `FrobeniusRoots` (the roots `phi^i(x)` of `f` in `GR(p^N, deg f)`). In `galois_ring.rs`: a lazy
  product, `defining_polynomial` (`:164`), `base_ring_value` (`:475`), and a fix in `with_modulus`
  (`:112`): a `D` with `D[d] = 1` mod `p` but not mod `p^a` was stored unnormalized.
- `rustmath-combinatorics/src/designs.rs:1058` `t_design_index` and `:1154` `is_steiner_system`:
  exact on a block list, a refusal above 10^8 `t`-subsets of blocks.
- `rustmath-coding/src/cyclic_golay.rs`: the two cyclic Golay codes (`g+`, `g-`), their weight
  distributions (`:115`), heptads (`:155`) and shift orbits (`:161`).
- `rustmath-galois/src/integrality.rs:110` `bounded_integer`, `:170` `integrality_dichotomy`;
  `rustmath-galois/src/steiner_labelling.rs:748` `steiner_labelling`, `:287` `RootBound`
  (Fujiwara, exact), `:921` `golay_invariants`, `:961` `heptad_resolvent`, `:983` and `:658`
  `duad_resolvent`, `:996` `frobenius_sign`.
- Tests: `rustmath-galois/tests/steiner_labelling.rs` (13 fast tests, 2 `full_`, 8 `env_`),
  `rustmath-finitefields/tests/galois_ring_frobenius.rs`,
  `rustmath-combinatorics/tests/designs_exact.rs`, `rustmath-coding/tests/cyclic_golay.rs`.
- Data: `rustmath-galois/tests/data/m23/` with a README (sources, licences, sha256): the owner's
  `golay_resolvents.json` with its `"f"` fields removed, `frobenius_signs.json`, Le Lay's CC0
  specialization at `T = 2`, and reference values for it. f1 and f2 are read only from
  `$M23ISGALOIS`. So the opening paragraph's "Nothing from it is vendored" is out of date here too.

How it decides. The roots are lifted in `GR(p^N, f)` at an inert prime (29 for f1 and f2). `T5`
is computed for both cyclic systems. With `p^N > 2 B5` (`B5` a proven bound), exactly one
candidate within `B5` is Decided (paper, Proposition 5.1); two are Unresolved; none is
`NotInM23`, a proof that `Gal(f)` lies in no conjugate of M23. Decided rests on the trusted input
`Gal(f) = M23`, recorded in every result (as the genus is in WP-F). `J5`, `J7` and the resolvents
are certified the same way: each value passes its proven bound at `p^N > 2B`, or the call fails.
There is no stability loop. A recomputation at a higher `N` is only a cross-check
(`env_stability_cross_check`).

Reproduced from reference (`data/golay_resolvents.json`, `data/frobenius_signs.json` at
`d05475b`). Every value is Decided, relative to the trusted input.

| Value | f2 | f1 | Test |
|---|---|---|---|
| prime, `N` | 29, 30 | 29, 31 | `env_six_integers_and_signs` |
| sign at 29 | -1 | -1 | same |
| `T5` | -1059840 | 92747040 | same |
| `J5` | 61916 | 524492 | same |
| `J7` | -83030 | -2281968 | same |
| rejected `T5` | 44 digits | 45 digits | same |
| `R253_heptad` | equal (`N` = 360) | equal (`N` = 395) | `env_heptad_resolvents` |
| `R253_duad` | equal (`N` = 267) | equal (`N` = 302) | `env_duad_resolvents` (also by `subset_sum_resolvent`, exact over Q) |
| second prime | 223: `T5`, `J5`, `J7` and both resolvents again | 73: the same | `env_second_prime` |
| Frobenius signs | 40 of 40 | 40 of 40 | `env_frobenius_signs` |
| least precision | `N` = 8 refused, 9 Unresolved, 10 Decided | `N` = 10 Decided | `env_f2_minimal_precision` |

The fast tests need no `$M23ISGALOIS`. They run on the monic associate of Le Lay's certified CC0
polynomial: Decided -1 at 29 (`N` = 270, and the least rigorous `N` = 250), Decided +1 at 67
(`N` = 216), `J5` and `J7` (353 and 491 digits) and two Frobenius signs. Each value has two
references: the owner's `arith.py` with python-flint 0.9.0, and PARI/GP 2.15.4. Small cases: C23
is Unresolved at every precision; `x^23 - x - 1` (Galois group S23) is `NotInM23` at 23, and at 157
with the least `N` = 4 it gives a false Decided, which the default margin rejects. That is why the
hypothesis is recorded in every result.

Commands:

- `cargo test -p rustmath-galois --test steiner_labelling` (fast set, debug, 13 to 16 s of test time
  in the last two runs);
- `cargo test -p rustmath-finitefields --test galois_ring_frobenius`,
  `cargo test -p rustmath-combinatorics --test designs_exact`,
  `cargo test -p rustmath-coding --test cyclic_golay`;
- `cargo test --release -p rustmath-galois --test steiner_labelling -- --ignored full_` (about 1.7 s);
- `M23ISGALOIS=<checkout of shaowuz/m23isgalois> cargo test --release -p rustmath-galois --test steiner_labelling -- --ignored env_`
  (about 29 s on one test thread, after a release build of 1.5 to 2 minutes). Without
  `M23ISGALOIS` each `env_` test fails with "set M23ISGALOIS to a checkout of shaowuz/m23isgalois".

Times (release, one thread): `env_six_integers_and_signs` 0.26 s; heptad resolvents 4.6 s (about
2 s each); duad resolvents 7.2 s (three routes); second prime 6.9 s; the 80 Frobenius signs 5.4 s.

For the later packages: WP-P can use `cyclic_golay::heptads` and `is_steiner_system`; WP-A A4 can
use `orbit_polynomial` and the vector form of `integrality_dichotomy` (5.3, A4).

Follow-ups (`steiner/impl/SL.md`, Handoffs): SL-1 Tschirnhaus (needs a proven coordinate bound);
SL-2 the CC0 resolvents (`N` about 12,500, about 10 minutes, not run); SL-3
`GaloisRingElement::frobenius` through a cached `FrobeniusMatrix`; SL-4 `GaloisCtx` on
`FrobeniusRoots`; SL-5 tighter bounds; SL-6 a rationality dichotomy for WP-A A4; SL-7
`BlockDesign::new` and `SteinerSystem::new` through the exact checker; SL-8 other primes; SL-9 a
refutation screen by Frobenius cycle types (a type outside the 17 of M23 refutes; a clean screen
proves nothing); SL-10 the shared target directory of the worktrees (give each its own, or rebuild
after a switch); SL-11 the clippy errors that `rustmath-combinatorics` had before this change. The
four optional nits of the check of the minor fixes (`steiner/impl/SL.minorscheck1.md`, Section 7)
are open. For the owner: the vendored `golay_resolvents_no_f.json` still determines f1 and f2
through `R253_duad` (README of the data).

## 7. Facts found while checking the sources

These came out of reading Le Lay's paper and Zarhin's two papers against our draft. They matter
for anyone extending the work.

- Le Lay's preprint is now titled "Arithmetic and geometry of an M23 Hurwitz scheme". HJLPPZ v2
  cites it under an earlier title. The bidegree (4,23) equation of HJLPPZ v2 is the minimal
  equation constructed there.
- The lemma that a 23-cycle preserves exactly two Steiner systems, with the same generator
  polynomial, is in Le Lay, Section 5.4. He uses the heptad invariant `J7` to choose between
  them for wild covers in characteristic 23. Our p-adic labelling is the number-field analogue.
- The heptad of unramified points above `t = 0`, its Fano plane, the affine structure on the
  eight ramified points and the real points of that fibre are in Le Lay, Section 8. Our paper
  now attributes them.
- Le Lay's class numbers 1..7 map to ours as `[0, 1, 2, 4, 6, 5, 3]`; his real classes 3, 6, 7
  are our 2, 5, 3 and his distinguished class 6 is our rational class 5. Three independent
  computations of the real classes agree.
- Zarhin's theorems cover `S_n`, `A_n` (2000) and `M11`, `M12`, `L_m(q)` (2002). They need the
  module of even subsets of the roots to be very simple. For M23 in degree 23 that module
  contains the even Golay code, so the method does not apply; the papers do not discuss M23.
- 11 divides the discriminant class 4873 of the canonical quadric, so the canonical quadric of
  the reduction of X at 11 is singular and X mod 11 should have a vanishing even theta-null.
  This follows from the square class and good reduction; it was not checked by computation.
  It is why route A avoids p = 11.

## 8. Order of work and definition of done

1. WP-F core (3.1, 3.2): done when both tables and the three derived checks are reproduced in
   Rust tests that read `data/zeta_X.json` and `data/zeta_L.json`.
2. Steiner labelling exercise (end of Section 6): done when the six integers and one degree 253
   resolvent are reproduced. Done (2026-11-15): the six integers and all four degree 253
   resolvents, at 29 and at the second primes (Status at the end of Section 6).
3. WP-P table (4.2, 4.3): done when all 57 rows are reproduced and the three dictionaries hold.
4. WP-F extensions 1 and 2 (the D flags, p = 3 and 5; Section 3, Status): 1 done by the
   delta-sum certificate (no recount on a second model); 2 done for p = 5, and p = 3 moves to
   WP-A A0 (follow-up F-1).
5. WP-A A0 and A1 with their predictions. Stop and report if a prediction fails.
6. WP-A A2 to A5, then the tests of 5.2 against the table.
7. WP-P 4.4, item 2: the genus 0 class vectors, after the Belyi engine is generalized (Section 2).

Report each result with its status: reproduced from reference, new and verified at two primes,
or new and unverified. Do not report the A8 polynomial as a theorem until 5.6 is done.
