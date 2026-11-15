# Steiner labelling exercise: implementation report

Date: 2026-11-15. Worktree `/home/user/RustMath-steiner`, local branch `claude/m23-steiner`, based on
`claude/m23-wpf` `d91f880`. Design: `design/SL.md` (approved by `design/SL.review1.md`). Nothing was
committed (the brief of this step says not to commit; the design's "four commits" are a later step).
All paths of scratch material below are under
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/m23/sl/impl/`.

## 0. Summary

- **Done criterion reached, in full.** At p = 29, in Rust tests that read f1 and f2 at run time from
  `$M23ISGALOIS`: f2 `T5 = -1059840`, `J5 = 61916`, `J7 = -83030`; f1 `T5 = 92747040`,
  `J5 = 524492`, `J7 = -2281968`; both Decided with sign -1. All four degree 253 resolvents equal
  `golay_resolvents.json`. The second-prime field is reproduced (f1 at 73, f2 at 223: same six
  integers, same four resolvents). All 80 Frobenius signs of `frobenius_signs.json` are reproduced.
- **Fast suite on the CC0 polynomial.** Le Lay's certified specialization: labelling at 29 (N = 270
  and the minimal rigorous N = 250) and at 67, `J5`, `J7`, the identity `T5 = 120 J5 + S`, two
  Frobenius signs; all against two references (arith.py with FLINT, and PARI/GP).
- **Four crates changed**, as the design (D2): `rustmath-finitefields`, `rustmath-combinatorics`,
  `rustmath-coding`, `rustmath-galois`. Every crate's whole test suite passes; clippy shows no
  warning in new code; no `unsafe`.
- **All nine minor notes of review 1 are folded in** (m1 to m9), except the optional part of m1
  (a refutation screen by cycle types), listed as a follow-up.
- **Times.** Release: all `env_` tests 29 s (one thread), `full_` 1.6 s, fast set 2.5 s; all 23
  tests 22 s on two threads. Debug fast set: about 23 s of test time (more than the design's 13 s
  estimate, Section 5).

## 1. What changed, per item

### 1.1 rustmath-finitefields (Plan step 1)

`src/galois_ring.rs`:

- `:112` `GaloisRing::with_modulus`: review m5. `D` with `D[d] = 1 mod p` but `D[d] != 1 mod p^a`
  was stored as given while the reduction assumed `D[d] = 1`. Now `D` is multiplied by
  `D[d]^{-1} mod p^a` (`:135`), which generates the same ideal, so the stored polynomial is monic
  and the generator is a root of it. Chosen over rejecting: the ring the caller asked for is exactly
  this one. Doc updated.
- `:164` `defining_polynomial()` (MAGMA `DefiningPolynomial`), `:171` crate-internal
  `canonical_element()` (the design's `from_canonical`, renamed for clippy `wrong_self_convention`),
  `:475` `GaloisRingElement::base_ring_value()`.
- `:1037` `impl Mul`: now calls `:1063` `mul_reduce` (lazy reduction: convolution over Z, one
  reduction per top coefficient, final `mod m`). Same output (test `lazy_mul_equals_schoolbook`
  against the old body, kept in the test file). D1 of the design.
- `frobenius()` doc: points to `FrobeniusMatrix` for repeated use.

`src/poly_factor.rs:239` `is_squarefree_fp_unchecked` (crate-internal): gcd with `f'` over `F_p` on
`Integer` coefficients, for primes above `fp_factor::MAX_P`.

`src/galois_ring_frobenius.rs` (new, 709 lines). Header: MAGMA ch. 47 (`GaloisImage`, `Trace`),
ch. 48; McDonald XVI, Wan 14, von zur Gathen-Gerhard 9.4 and 15.4; paper Section 5 and arith.py
`frobenius_lift`.

- `:114` `FrobeniusMatrix`: `new` (`:133`, Newton lift of `w^p` with `1/D'(y)`, checks
  `D(y) = 0`, `y = w^p mod p`, `phi^d(w) = w`, each an `Err`), `with_precision` (`:255`,
  projection only), `lift_to` (`:289`, Newton continued into a given higher ring), `ring`,
  `image_of_generator`, `generator_orbit`, `apply` (one matrix-vector product), `orbit`,
  `generator_power_traces` (Newton's identities over `Z/p^a`), `trace`, `orbit_polynomial`.
- `:460` free `orbit_polynomial(values)`: `prod (Z - v)`, `Err(NotInBaseRing { index })` when a
  coefficient is not in `Z/p^a`.
- `:555` `FrobeniusRoots`: `new` (`:573`; refusals in the design's order, exact squarefree and
  irreducibility tests by `fp_factor::factor_with_multiplicity` up to `MAX_P` and by
  `is_squarefree_fp_unchecked` plus Rabin's test above it), `with_precision` (`:618`, projection
  or Newton continuation), `polynomial`, `prime`, `precision`, `modulus`, `ring`, `frobenius`,
  `roots`.
- `:504` `FrobeniusRootsError` and `NotInBaseRing`, with `Display` and `Error`.

`src/lib.rs`: module and re-exports `orbit_polynomial, FrobeniusMatrix, FrobeniusRoots,
FrobeniusRootsError, NotInBaseRing`.

`tests/galois_ring_frobenius.rs` (new, 11 tests): `lazy_mul_equals_schoolbook` (six rings,
including a modulus with full-size coefficients mod `7^60`); `with_modulus_normalizes_a_leading_
coefficient_one_mod_p` (m5: `GR(5^2, 6x^2 + x + 2)` stores `x^2 + 21x + 17`, the generator is a
root); `small_roots_match_pari` (`x^3 + x + 1` at `5^6`, `x^23 - x - 1` at `23^4`, `x^5 - x + 1`
at `3^7`; PARI/GP `polrootspadic` ordered by `t^(p^i)`); `roots_above_the_word_size_bound_match_
pari` (`x^3 - x - 1` at 3037000573, `x^2 + 1` at 3037000507); `newton_image_equals_teichmuller_
frobenius`; `frobenius_is_a_ring_automorphism_of_order_d`; `trace_is_the_orbit_sum` (and PARI
`polsym`); `orbit_polynomial_vanishes_on_the_orbit`; `orbit_polynomial_rejects_an_unstable_family`;
`with_precision_projects_and_continues` (also `lift_to` refusals); `refusals` (every root row of the
design's Section 6, including the large-prime branch).

### 1.2 rustmath-combinatorics (Plan step 2)

`src/designs.rs:967` `DesignCheckError`, `:1011` `DESIGN_CHECK_LIMIT = 10^8`, `:1045`
`t_design_index` (MAGMA `IsDesign`), `:1123` `is_steiner_system` (MAGMA `IsSteiner`). Exact:
enumerates the t-subsets of each block in a hash map, compares the number of distinct subsets met
with `C(v, t)`; refuses above the limit before any enumeration. Header cites ch. 147. `src/lib.rs`
re-exports. `BlockDesign::new` and `SteinerSystem::new` are unchanged.

`tests/designs_exact.rs` (new, 5 tests): Fano plane (t = 0, 1, 2, 3), Fano minus a line, a repeated
line, S(3,4,8) from the Fano plane (t = 1, 2, 3), the cube's faces as a 1-(8,4,3) design (H147E9),
degenerate cases, the four refusals (`TooLarge` for `C(200, 5) = 2535650040`).

### 1.3 rustmath-coding (Plan step 3)

`src/cyclic_golay.rs` (new, 202 lines). Header: ch. 152 `CyclicCode` (H152E5), `QRCode`, ch. 147;
MacWilliams-Sloane ch. 16; Witt 1938; paper Section 2 and golay.py. `:40` `GolaySign` (`ALL`,
`generator_polynomial`, `opposite`, `to_i8`, `from_i8`, `Display`), `:103` `codewords`, `:115`
`weight_distribution`, `:155` `heptads` (cached, `OnceLock`), `:161`
`heptad_orbit_representatives`, `:166` `rotate`, `:176` `negate`, `:183` `scale`, `:192`
`support`, `:197` `mask_of`. `src/lib.rs`: module, `pub use cyclic_golay::GolaySign`.
`Cargo.toml`: dev-dependency `rustmath-combinatorics`.

`tests/cyclic_golay.rs` (new, 7 tests): generator polynomials (`(z+1) g+ g- = z^23 + 1`,
`g- = reverse(g+)`); weight distributions; both systems S(4,7,23) via `is_steiner_system`, with
`lambda_3 = 5`, `lambda_2 = 21`, `lambda_1 = 77`; no common heptad; shift and `i -> 2i`
invariance, `i -> -i` exchanges them, `i -> 5i` does not preserve them; intersections
`{1: 14168, 3: 17710}`; the 11 orbit representatives of golay.py; the 5 heptads through
`{0, 1, 2}`; the 12 rows of `BinaryGolayCode::new()` lie in `G-`.

### 1.4 rustmath-galois (Plan step 4)

`Cargo.toml`: dependencies `rustmath-coding`, `rustmath-combinatorics`; dev-dependency
`serde_json = "1"` (already in the lockfile). `Cargo.lock`: three dependency edges, no version
change.

`src/integrality.rs` (new). Header: ch. 38 `IsInt`, `Bound`; Geissler-Klueners 2000, Stauduhar
1973; paper Proposition 5.1. `:28` `symmetric_residue`, `:43` `IntegralityError::PrecisionTooLow`,
`:85` `bounded_integer`, `:97` `DichotomyGap`, `:108` `Dichotomy<L>`, `:142`
`integrality_dichotomy` (vector-valued candidates, review m9). Unit tests: residues for odd and
even moduli, `m = 2B` refused and `m = 2B + 1` accepted, the three outcomes, vector candidates.

`src/steiner_labelling.rs` (new, 1012 lines). Header: paper Sections 2, 5, 6, 8; arith.py;
Fujiwara 1916; Witt 1938; Stauduhar, Geissler-Klueners; MAGMA ch. 38, 47, 48. The module docs give
the proof, the trusted input, its weaker sufficient form (review 1.3) and the false-hypothesis
example (review m1).

- `:95` `GaloisHypothesis::M23`; `:104` `Certification` (`Symmetric`, or `Labelling { hypothesis,
  sign, prime, precision }`); `:125` `CertifiedInteger` and `:170` `CertifiedResolvent` with
  private fields and getters (review m6); `:159` `GolayInvariants`.
- `:203` `SteinerError` (with `From` for the two error types).
- `:286` `RootBound` (`fujiwara`, `best`, `power_bound`, `binomial_bound`, the five bounds);
  `:375` `precision_for`; `:398` `monic_associate`; `:415` `first_inert_prime` (a bounded search).
- `:451` the run-time check of the two Golay systems (S(4,7,23) by `is_steiner_system`, shift
  invariance, 11 orbits, distinct), once per process.
- `:520` `t5_candidates` (traces over the 11 orbit representatives).
- `:639` free `duad_resolvent(f, p)` (symmetric, no hypothesis).
- `:653` `LabellingOptions` (defaults as the design), `:680` `LabellingStatus` (`Decided { sign }`,
  `Unresolved`, `NotInM23`: review m7), `:696` `SteinerLabelling`, `:719` `steiner_labelling`.
- `SteinerLabelling` (`:791`): `status`, `hypothesis`, `sign`, `t5`, `rejected_t5`, `t5_residue`,
  `polynomial`, `prime`, `precision`, `attempts`, `root_bound`, `t5_bound`, `roots`,
  `golay_invariants` (`:887`), `heptad_resolvent` (`:922`), `duad_resolvent` (`:941`),
  `frobenius_sign` (`:954`, from the certified T5: review m2).
- `:969` `FrobeniusSign`, `:992` `frobenius_sign_given_t5` (the bare-T5 form, documented as a
  trusted input; its doc says why q = 2, 3, 5 with `N <= v_q(120)` never separate: review m9).

`src/lib.rs`: modules `integrality`, `steiner_labelling`, crate docs.

`tests/data/m23/` (new): `golay_resolvents_no_f.json`, `frobenius_signs.json`,
`lelay_specialization_t2_Z.json`, `cc0_t2_steiner_reference.json`, `make_cc0_reference.py`,
`README.md` (sources, licences, sha256, the derivation command, and the note that the stripped
file still determines f1 and f2: review m4).

| File | sha256 |
|---|---|
| `golay_resolvents_no_f.json` | `87123dfe2fe86cb6bf5b40e2a39c2d06a680c6a2df2e8fd8bce1adee4f87a381` (original `dcf1e38c...70d13`; matches the design) |
| `frobenius_signs.json` | `30ec467012f53e2b8e7e6a4d05e321a8adabe747aea1768488b9cda604617a4f` |
| `lelay_specialization_t2_Z.json` | `f775aa4c004541f25d121fac999fca9c62e493100f99982cc8253fbfa4445862` (line 84 of Le Lay's `CHECKSUMS.sha256`) |
| `cc0_t2_steiner_reference.json` | `91532572595a9cd6b3cc87473fca1dc416b32162c91c724bc91a507277f58dee` (reproducible: two reruns gave the same hash) |

The stripped file contains no f1 or f2 coefficient (checked by grep for coefficients of f2). f1
and f2 appear in no RustMath source, comment or data file.

`tests/steiner_labelling.rs` (new, 23 tests: 13 fast, 2 `full_`, 8 `env_`).

## 2. Tests and reference sources

Fast (debug, normal suite):

| Test | Checks | References |
|---|---|---|
| `bounds_and_precision` | `best`, `B5` of `x^23 - x - 1` (256, 136069473) and C23 (1201, 309225934046); `x^2 - 4`; `precision_for` | arith.py; PARI/GP `fbest` (review `rev.gp`), `gp/check_sl.gp` |
| `monic_associate_scales_the_roots` | three small cases | by hand |
| `c23_is_unresolved_at_every_precision` | p = 2 and 3 with two retries: Unresolved, residues `{-98317, -30637}` (swapped at 3); `NotDecided`; `N = 39` refused, `N = 40` accepted; signs given T5 at q = 2, 3, 5 (both match), `t5 = 0` at 2 (none), q = 5, N = 2 (Decided +) | PARI/GP `c23.gp`, `c23_p3.gp`, `gp/check_sl.gp`; arith.py |
| `s23_refutes_the_hypothesis` | `x^23 - x - 1` at 23: N = 29, `NotInM23`, both residues (reductions of the N = 30 values) | PARI/GP, arith.py |
| `false_hypothesis_can_decide` | review m1: at 157, N = 4: Decided(-) with residues 182056093, 128168772; default N = 18: `NotInM23` | review `run_wrong.gp`, arith.py, `gp/check_sl.gp` |
| `duad_resolvent_three_ways` | orbit route = `subset_sum_resolvent(f, 2)` = direct product over the 253 duads mod `23^3` | rustmath-polynomials (exact over Q), direct product |
| `refusals` | not monic, degree 22, `p = 25`, not squarefree, CC0 at 11 (`[2, 7, 14]`), `NoInertPrime { below: 200 }` for a product, `N = 0`, `t5_candidates` on 3 roots | Le Lay's certificate |
| `cc0_certificate_facts` | content 1, degree 23, leading coefficient, factor degrees `[23]` mod 29 and `[2, 7, 14]` mod 11 equal to the certificate; 29 is the first inert prime | Le Lay's file |
| `cc0_labelling_at_29` | `best`, `B5` (364 digits), N = 270, Decided(-), `T5` (356 digits), rejected (395 digits); N = 250 Decided, same T5, rejected (365 digits) = reduction of the one at 270 | arith.py and PARI/GP (`steiner.gp`, `rev.gp`) |
| `cc0_labelling_at_67` | N = 216, Decided(+), same T5, rejected (394 digits) | arith.py, `gp/check_sl.gp` (rev.gp) |
| `cc0_golay_invariants` | `J5` (353 digits), `J7` (491 digits), their bounds, N = 344, the certification; `T5 = 120 J5 + S(g)` | arith.py, PARI/GP; the identity is exact and p-adic free |
| `cc0_frobenius_signs` | Frob at 73 and 109: Decided(-) | arith.py, PARI/GP, the f1 rows of `frobenius_signs.json` |
| `trace_route_equals_direct_sums` | at N = 40, the direct sums over the 253 heptads equal `t5_candidates` | the direct sum |

Opt-in: `full_cc0_frobenius_signs` (the 12 inert primes below 900 and their signs, equal to the
reference file and to the f1 rows), `full_cc0_invariants_at_67`.

`env_` (read f1, f2 from `$M23ISGALOIS`, check the files' own fingerprint lines first):
`env_six_integers_and_signs` (also `best`, `B5`, `B_J5`, `B_J7`: review m8),
`env_f2_minimal_precision` (N = 8 refused, N = 9 Unresolved, N = 10 Decided, one retry gives
attempts `[9, 16]` and Decided; f1 at N = 10), `env_heptad_resolvents`, `env_duad_resolvents` (three
references), `env_second_prime`, `env_frobenius_signs` (80 rows), `env_t5_j5_identity`,
`env_stability_cross_check` (J5, J7 by direct sums and the heptad resolvent by rebuilt orbit
products at N + 7: a cross-check, not a certificate). Without `M23ISGALOIS` they panic with "set
M23ISGALOIS to a checkout of shaowuz/m23isgalois" (checked: `env_six_integers_and_signs` fails in
0.07 s).

New reference values computed in this step (scratch `gp/`, `oracle/`):

- `oracle/make_cc0_reference.py` (vendored as `tests/data/m23/make_cc0_reference.py`): arith.py
  with python-flint 0.9.0, Python 3.13.18, venv `../venv`, 8 s. Its 29 and 270/250 values equal the
  design's `oracle_cc0.json`, the design's PARI/GP `cc0_out.txt` and the review's `cc0_rev_t5.txt`.
- `gp/check_sl.gp` (PARI/GP 2.15.4, the review's `rev.gp`): the CC0 residues at (29, 250) and
  (67, 216) equal arith.py; C23 at 2, 3, 5 and its residues mod 5 and 25; `x^23 - x - 1` at 23
  (N = 29, 30) and 157 (N = 4, 18). Output `gp/check_sl.txt`.
- `gp/ff_refs.gp`, `gp/small_lift2.gp`: the factorizations, the inert primes 23 and 157 of
  `x^23 - x - 1` below 400, `polsym`, the large-prime images, the quintic lift.
- `oracle/f12_bounds.py`: the f1, f2 bounds of review m8 (equal to the review's values).

## 3. Reproduced values

| Value | f2 | f1 | Status |
|---|---|---|---|
| labelling prime, N | 29, 30 | 29, 31 | reproduced |
| sign at 29 | -1 | -1 | reproduced (Decided, relative to Gal = M23) |
| T5 | -1059840 | 92747040 | reproduced |
| J5 | 61916 | 524492 | reproduced (N_J = 8, 9) |
| J7 | -83030 | -2281968 | reproduced |
| rejected T5+ | -29622039743232607524836197957123800868867866 (44 digits) | -883496073094568376895161675988763338141335547 (45 digits) | reproduced (arith.py, PARI/GP) |
| R253_heptad | equal, N = 360 | equal, N = 395 | reproduced |
| R253_duad | equal, N = 267 | equal, N = 302 | reproduced (also by `subset_sum_resolvent`) |
| second prime | 223: Decided(-), N = 19, N_J = 5, N_heptad = 224, N_duad = 167; six integers and four resolvents equal | 73: Decided(-), N = 24, N_J = 7, N_heptad = 310, N_duad = 237; equal | reproduced (`verified_at_second_prime`) |
| Frobenius signs | 40 of 40 | 40 of 40 | reproduced |
| minimal precision | N = 8 refused, N = 9 Unresolved (T5+ = -3209424281433), N = 10 Decided (T5+ = -206309467943599) | N = 10 Decided (T5+ = -114831914638610) | reproduced (reductions of the reference residues) |
| bounds | best 2157, B5 5778453176946, B_J5 7220058114, B_J7 97634149933 | 3473, 62529476039478, 78129291179, 2738950710192 | equal to arith.py |
| `T5 = 120 J5 + S` | holds | holds | new check, exact |

CC0 (monic associate of Le Lay's polynomial): Decided(-) at 29 (N = 270 and 250), Decided(+) at 67
(N = 216), T5, J5, J7 and twelve signs: reproduced from two references. Resolvents of the CC0
polynomial: not computed (follow-up SL-2).

## 4. Public API

As in design Section 5, with the changes of Section 6 below. In short:

```rust
// rustmath-finitefields
GaloisRing::defining_polynomial(&self) -> &[Integer];
GaloisRingElement::base_ring_value(&self) -> Option<Integer>;
FrobeniusMatrix::{new, with_precision, lift_to, ring, image_of_generator, generator_orbit,
                  apply, orbit, generator_power_traces, trace, orbit_polynomial};
orbit_polynomial(&[GaloisRingElement]) -> Result<Vec<Integer>, NotInBaseRing>;
FrobeniusRoots::{new(f, p, N), with_precision, polynomial, prime, precision, modulus, ring,
                 frobenius, roots};
// rustmath-combinatorics
t_design_index(v, t, blocks) -> Result<Option<u64>, DesignCheckError>;
is_steiner_system(v, t, blocks) -> Result<bool, DesignCheckError>;
// rustmath-coding::cyclic_golay
GolaySign, codewords, weight_distribution, heptads, heptad_orbit_representatives, rotate,
negate, scale, support, mask_of, LENGTH;
// rustmath-galois::integrality
symmetric_residue, bounded_integer, integrality_dichotomy, Dichotomy, DichotomyGap,
IntegralityError;
// rustmath-galois::steiner_labelling
steiner_labelling(f, &LabellingOptions) -> Result<SteinerLabelling, SteinerError>;
SteinerLabelling::{status, hypothesis, sign, t5, rejected_t5, t5_residue, ..., golay_invariants,
                   heptad_resolvent, duad_resolvent, frobenius_sign};
LabellingStatus::{Decided { sign }, Unresolved, NotInM23};
t5_candidates, duad_resolvent, frobenius_sign_given_t5, first_inert_prime, monic_associate,
precision_for, RootBound, CertifiedInteger, CertifiedResolvent, Certification,
GaloisHypothesis::M23, FrobeniusSign, DEGREE;
```

## 5. Measured times

Machine: 4 cores shared with the F-9 job, `-j 2`. Wall clock.

Release (`steiner_labelling` test binary, one test thread): build 149 s (from no `target/release`).

| Test | Time |
|---|---|
| `env_six_integers_and_signs` | 0.26 s |
| `env_f2_minimal_precision` | 0.23 s |
| `env_heptad_resolvents` | 4.57 s (f2 2.12 s at N = 360, f1 2.32 s at N = 395) |
| `env_duad_resolvents` | 7.18 s (orbit route 1.21 s and 1.61 s; `subset_sum_resolvent` 0.74 s and 0.82 s) |
| `env_second_prime` | 6.91 s |
| `env_frobenius_signs` | 5.39 s (80 primes) |
| `env_t5_j5_identity` | 0.00 s |
| `env_stability_cross_check` | 4.21 s |
| all `env_` | 28.8 s |
| `full_cc0_frobenius_signs`, `full_cc0_invariants_at_67` | 0.96 s, 0.61 s |
| fast set in release | 2.53 s |

Debug, one test thread, per fast test: `duad_resolvent_three_ways` 9.5 s, `cc0_golay_invariants`
10.3 s (including the shared labelling), `cc0_labelling_at_29` 5.5 s (two labellings),
`cc0_frobenius_signs` 4.0 s, `trace_route_equals_direct_sums` 3.6 s, `cc0_labelling_at_67` 2.8 s,
`false_hypothesis_can_decide` 0.9 s, `c23_...` 0.9 s, `s23_...` 0.3 s, the rest below 0.2 s. The
CC0 labelling at 29 is now shared by three tests (`OnceLock`). Whole fast set with the default
threads: 23.4 s of test time ("finished in" of the crate run; 21.5 s in an earlier run). The design estimated 13 s; the difference is the
duad three-way test (not in the design's list) and the lift from 270 to 344 for J. It stays below
the WP-F fast set's order of magnitude; `duad_resolvent_three_ways` can move to `full_` if wanted.

Whole crates (debug, `cargo test -j 2 -p <crate>`; wall time including the rebuild that the
touch of Section 7 forces; baseline at `d91f880` in brackets, without rebuild):

| Crate | Result | Wall |
|---|---|---|
| rustmath-finitefields | all pass: lib 205, 12 integration files (new `galois_ring_frobenius` 11), doc 8 | 31 s [17 s] |
| rustmath-combinatorics | all pass: lib 1509 (7 ignored), 4 integration files (new `designs_exact` 5), doc 189 | 48 s [34 s] |
| rustmath-coding | all pass: lib 67, new `cyclic_golay` 7, doc 5 | 17 s [9 s] |
| rustmath-galois | all pass: lib 65 (was 62: the 3 `integrality` unit tests), descent 12 (31.5 s), new `steiner_labelling` 13 (10 ignored), others unchanged | 83 s [23 s] |

Dependents of the changed `GaloisRing`: `cargo check -p rustmath-rings -p rustmath-category -p
rustmath-algebras` passes; `rustmath-category` `ring_parents` tests pass (5); `rustmath-rings`
`category_methods::tests::test_galois_ring_parent` and `category_parents::tests::galois_ring_parents`
pass. (The filter `galois_ring` matched no `rustmath-algebras` lib test, so its Galois ring test was
only compiled.) `rustdoc --no-deps` gives no warning in new code.

Final release confirmation after the last edits (the rename and the format pass), all 23 tests with
`--include-ignored`, 2 threads, `M23ISGALOIS=/home/user/external/m23isgalois`: 23 passed, 22.2 s of
test time (123 s with the rebuild). Without `M23ISGALOIS`, `env_six_integers_and_signs` fails with
"set M23ISGALOIS to a checkout of shaowuz/m23isgalois". `target/release` was deleted afterwards
(the F-9 release runs use their own `target-f9rel`, and no process used it). Free disk at the end:
8.2 GB.

## 6. Deviations from the design, with evidence

1. **No commits.** The brief of this step forbids them; the design's four commits (D2) are for the
   later step.
2. **`FrobeniusMatrix::with_precision` only projects; `lift_to(&GaloisRing)` added.** The design
   gave one `with_precision` that also lifts. `D mod p^a` does not determine `D mod p^b` for
   `b > a`, so a lift needs the target ring. The first version built `GR(p^b, D mod p^a)` and the
   test `with_precision_projects_and_continues` failed (the roots lived in a different ring). Now
   `FrobeniusRoots::with_precision` (which keeps `f` over Z) builds the target ring from `f` and
   calls `lift_to`, as designed.
3. **Review m7: a third status.** `LabellingStatus::NotInM23` replaces `Unresolved(NeitherWithinBound)`;
   `LabellingGap` is gone and `SteinerError::NotDecided` carries the status. Its doc says it is a
   proof without hypothesis (review 1.2).
4. **Review m2:** `SteinerLabelling::frobenius_sign(q, N)` uses the certified T5; the bare form is
   `frobenius_sign_given_t5(f, q, t5, N)`.
5. **Review m6:** private fields with getters, and `Certification` recorded in every certified
   value.
6. **Review m9:** `integrality_dichotomy` takes `&[(L, Vec<Integer>)]`; `Dichotomy` carries vectors.
7. **Review m5:** `with_modulus` normalizes `D` (instead of rejecting it).
8. `FrobeniusRootsError::CheckFailed(String)` (design: `&'static str`), to carry the message of the
   failed check.
9. `DesignCheckError::TExceedsPoints { t, v }` added: for `t > v` no `lambda` is determined.
10. `t5_candidates` returns `Result` (`WrongDegree` when there are not 23 roots).
11. Additions: `GolaySign::from_i8`, `Display`, `scale`, `mask_of`; `FrobeniusMatrix::
    generator_orbit`, `generator_power_traces`; `GaloisHypothesis` doc records the weaker
    sufficient hypothesis (review 1.3).
12. `canonical_element` instead of `from_canonical` (clippy).
13. A first quintic reference of mine (`x^5 + 2` at 3, `ff_refs.gp`) was wrong: that polynomial
    is reducible mod 3 (`x = 1` is a root), and `FrobeniusRoots::new` rightly refused it. Replaced
    by `x^5 - x + 1`, irreducible mod 3, with the value from `polrootspadic` (`small_lift2.gp`).
14. Extra tests beyond the design: `false_hypothesis_can_decide` (m1), `duad_resolvent_three_ways`,
    `bounds_and_precision`, `monic_associate_scales_the_roots`,
    `roots_above_the_word_size_bound_match_pari`, `with_precision_projects_and_continues`,
    `with_modulus_normalizes_...`.
15. Fast debug time about 21 s against the design's 13 s (Section 5).

Skipped: the optional part of review m1 (a refutation screen by Frobenius cycle types) is a
follow-up (SL-9).

## 7. Shared target directory: a hazard found

`target` is shared with `/home/user/RustMath-wpf`. Cargo names the artifacts of path dependencies
by their path relative to the workspace root, so both worktrees write the same files. A build in
the other worktree left a `rustmath-finitefields` rlib without the new module, and cargo here took
it as fresh (its dep-info listed the other worktree's unchanged files): the first galois test build
failed with `E0432` (no `FrobeniusRootsError` in `rustmath_finitefields`). Workaround: every cargo
command of this step ran through `cargo_sl.sh`, which touches the `lib.rs` of the four changed
crates first. The reverse also holds: a debug build in the WP-F worktree may now link this
worktree's `rustmath-finitefields` (a superset with the same values), or the other way round. I
did not build any crate that the F-9 work changes (`rustmath-curves`). The F-9 release runs use
their own `--target-dir target-f9rel`, so `target/release` here was used by this step only.

## 8. Left undone, and why

- Tschirnhaus (SL-1): needs a proven coordinate bound first (design 11).
- CC0 resolvents (SL-2): N about 12,500, about 10 minutes even with the lazy product.
- Other primes, WP-A uses (SL-6, SL-8): out of scope.
- Tracker and docs updates: a later step (brief).

## Handoffs

Tracker rows (`PROJECT_TRACKER.md`, later step):

- `rustmath-finitefields/src/galois_ring_frobenius.rs` (new): `FrobeniusMatrix`, `FrobeniusRoots`,
  `orbit_polynomial` [done; PARI/GP and Teichmueller cross-checks].
- `galois_ring.rs`: `Mul` lazy reduction (same values, 2 to 4 times faster); `with_modulus`
  normalization fix (a defect: `D[d] != 1 mod p^a` was stored unnormalized); `defining_polynomial`,
  `base_ring_value`.
- `poly_factor.rs:239` `is_squarefree_fp_unchecked` (crate-internal).
- `rustmath-combinatorics/src/designs.rs:1045` `t_design_index`, `:1123` `is_steiner_system`
  [done, exact]; the `SteinerSystem::new` row stays partial.
- `rustmath-coding/src/cyclic_golay.rs` (new) [done].
- `rustmath-galois/src/integrality.rs`, `steiner_labelling.rs` (new) [done; relative to the
  trusted input Gal(f) = M23].

MAGMA tracker rows: ch. 38 `IsInt` (integers, bounded) done; ch. 47 `GaloisImage`, `Trace` on
Galois rings (via `FrobeniusMatrix`) done for `GR(p^a, D)`; ch. 48 `DefiningPolynomial` done;
ch. 147 `IsDesign`, `IsSteiner` done (exact, block lists); ch. 152 `CyclicCode` only for the two
Golay codes (partial).

Workstream handoff (`docs/workstreams/m23_golay/HANDOFF.md`):

- Section 2, Galois rings: Frobenius is now cheap through `FrobeniusMatrix` (matrix-vector
  product); the `UnramifiedExtension` line can point to `galois_ring_frobenius.rs`. "Do not call":
  `SteinerSystem::new` still verifies only small designs, but `t_design_index` exists.
  "What to build": the Steiner labelling item is done (without a stability loop: all values are
  certified by proven bounds).
- New Status section for the Steiner labelling (like Section 3's), Section 8 step 2 done.
- WP-P: `cyclic_golay::heptads` and `is_steiner_system` are ready; WP-A A4: `orbit_polynomial` and
  `integrality_dichotomy` (vector form) are the pieces; a `rationality_dichotomy` is SL-6.
- Copy the scratch PARI/GP scripts (`design/SL/gp/`, `impl/gp/`) to `docs/workstreams/m23_golay/sl/gp/`
  with the f1/f2 path parameterized (review 1 Section 4).
- Tell the owner (review m4): the vendored `golay_resolvents_no_f.json` still determines f1, f2
  through `R253_duad`.

Follow-ups:

- SL-1 tschirnhaus; SL-2 CC0 resolvents (or a small model of the CC0 field); SL-3
  `GaloisRingElement::frobenius` through a cached `FrobeniusMatrix`; SL-4 port `GaloisCtx` onto
  `FrobeniusRoots`; SL-5 tighter bounds; SL-6 `rationality_dichotomy` for WP-A A4; SL-7
  `BlockDesign::new` through the exact checker; SL-8 other primes.
- SL-9 (new, review m1 optional): a sound refutation screen of the hypothesis by Frobenius cycle
  types (a type outside the 17 of M23 refutes; a clean screen proves nothing).
- SL-10 (new): the shared target directory (Section 7); give each worktree its own target or
  rebuild after switching.
- SL-11 (new): `rustmath-combinatorics` has pre-existing clippy errors (deny-level
  `inherent_to_string_shadow_display` in `binary_words.rs`, `debruijn.rs`), so
  `cargo clippy -p rustmath-combinatorics` fails before this change too; new code was linted with
  `--cap-lints warn` and is clean.

## Appendix: acceptance commands run

All through `impl/cargo_sl.sh` (touches the four `lib.rs`, then `cargo "$@"`), in
`/home/user/RustMath-steiner`:

```bash
cargo test -j 2 -p rustmath-finitefields          # baseline and after: all pass
cargo test -j 2 -p rustmath-combinatorics
cargo test -j 2 -p rustmath-coding
cargo test -j 2 -p rustmath-galois
cargo test -j 2 --release -p rustmath-galois --test steiner_labelling -- --ignored full_
M23ISGALOIS=/home/user/external/m23isgalois \
  cargo test -j 2 --release -p rustmath-galois --test steiner_labelling -- --ignored env_
cargo test -j 2 --release -p rustmath-galois --test steiner_labelling -- --ignored env_six   # fails: no M23ISGALOIS
cargo clippy -j 2 --no-deps -p <crate> --all-targets --message-format=short   # four crates
cargo clippy -j 2 --no-deps -p rustmath-combinatorics --lib --test designs_exact -- --cap-lints warn
rustfmt --edition 2021 <the eight new .rs files>     # existing files: only my hunks adjusted by hand
cargo check -j 2 -p rustmath-rings -p rustmath-category -p rustmath-algebras
cargo doc -j 2 --no-deps -p <crate>
```

Logs: `impl/logs/baseline.txt`, `step23_tests.txt`, `final_debug.txt`, `release_env.txt`,
`release_final.txt`, `dependents.txt`.

rustfmt on changed existing files: `lib.rs` of the four crates, `galois_ring.rs`, `poly_factor.rs`
and `designs.rs` were not rustfmt-clean at `d91f880` (unsorted `mod` lists and older code), so
`rustfmt` was not run on them; the one diff inside my own hunks (`galois_ring.rs`, the
`debug_assert` of `canonical_element`) was applied by hand.

## Minor items of review 1

Date: 2026-11-15. All five items of `impl/SL.review1.md` (Section 8) are fixed. None is rejected.
Still no commit. `git status --short` has the same 19 entries as before. Paths below are relative
to `/home/user/RustMath-steiner`. Logs: `impl/logs/minor1/`. Probe: `impl/minor1probe/probe.rs`
(built with `rustc --edition 2021 -O` against the release rlibs, binary deleted), output
`impl/minor1probe/probe_out.txt`.

### m1. Panic for `p < 2` (fixed)

- `rustmath-galois/src/steiner_labelling.rs:452`: new private `require_prime(p)`. It runs the
  same `try_is_prime` as `FrobeniusRoots::new` and returns
  `Err(SteinerError::Frobenius(FrobeniusRootsError::Prime(..)))` (message `"{p} is not prime"`).
  It costs about 10 us below `2^64`; above that a proof is cached for the process, so the second
  test in `FrobeniusRoots::new` is free. Import `rustmath_core::MathError` at line 78.
- It is called before `precision_for` in `duad_resolvent` (line 660) and in
  `steiner_labelling` (line 763, for a given or a searched prime). A composite `p` is now refused
  before the precision too (before, `p = 25` with an explicit precision `N` and `25^N <= 2 B5`
  gave `PrecisionTooLow`).
- `precision_for` keeps its documented panic for `p < 2`. All its other callers use the proven
  prime of a `FrobeniusRoots`.
- Test: `rustmath-galois/tests/steiner_labelling.rs:514` (in `refusals`):
  `steiner_labelling` with `prime: Some(0)`, `Some(1)`, and `duad_resolvent(x^23 - x - 1, p)`
  with `p = -23, 0, 1, 25`, all `Err(Frobenius(Prime(_)))`. The probe gives the same errors in
  release (`"0 is not prime"`, `"1 is not prime"`, `"-23 is not prime"`, ...); the review's probe
  panicked on these inputs.

### m2. Memory of `t_design_index` (fixed with a compact key; limit kept)

The memory problem was larger than the review measured: a key was a `Vec<usize>` of `t` entries,
so for large `t` it cost `8t` bytes more per subset. One block of 1000 points with `t = 998` is
499,500 subsets, below the limit, and about 4 GB in the first version (estimate: 499,500 keys of
998 entries; not run).

- `rustmath-combinatorics/src/designs.rs:1058` (`t_design_index`): the hash map is replaced by a
  dense `Vec<u32>` of `C(v, t)` counters (line 1097). The index is the colex rank
  `sum_i C(c_i, i + 1)`, a bijection onto `[0, C(v, t))` (combinatorial number system). Each
  block gets a table of `t (|B| - t + 1)` binomials, one per position and point that can occupy
  it. Every entry is a term of the rank of some `t`-subset, so it is below `C(v, t)`.
- New early exit (line 1090): when `S = sum_B C(|B|, t) < C(v, t)`, some `t`-subset lies in no
  block. The answer is then `None` (`Some(0)` for `S = 0`) with no enumeration. So the counter
  array is only built when `C(v, t) <= S <= DESIGN_CHECK_LIMIT`: at most 400 MB, for every `t`,
  and ranks and counts fit in `u32`.
- `DESIGN_CHECK_LIMIT` stays `10^8` (line 1012). Its doc now states the counter bound.
  `t_design_index`'s doc (line 1042) states the cost and the memory with the measurements below.
- Release measurements (probe; peak RSS from `VmHWM`):

| Input | First version (review) | Now |
|---|---|---|
| S(4,7,23), `t = 4`, each sign | | 0.11 ms |
| one block of 120 points, `t = 4` (8,214,570 subsets) | 6.1 s, 1.13 GB | 0.16 s, 35 MB |
| one block of 1000 points, `t = 998` (499,500 subsets) | about 4 GB (estimate, `8t` bytes per key) | 0.45 s, 4.5 MB |
| one block of `10^8` points, `t = 1` (at the limit) | about 14 GB (estimate) | 2.8 s, 2.3 GB peak, of which 0.8 GB is the input and 0.8 GB its sorted copy |
| one block of 30 points, `t = 15`; 1000 points, `t = 3` | | `TooLarge` at once |

- New test `rustmath-combinatorics/tests/designs_exact.rs:104`
  (`colex_ranks_cover_every_subset_once`). One block of all `v` points, given unsorted, has
  index 1 for `(v, t)` in `(1,1), (5,0), (5,2), (9,4), (12,6), (12,11), (12,12), (30,3)`, and
  index 2 when doubled, so a rank collision or gap fails. The complete design of all 4-subsets of
  9 points (helper `k_subsets`, line 85) has `lambda = C(9 - t, 4 - t)` = 21, 6, 1 for
  `t = 2, 3, 4` and 0 for `t = 5` (Beth, Jungnickel, Lenz, *Design Theory*, I.1). Without one
  block it gives `None` through the dense path, not the early exit. All existing assertions are
  unchanged and pass (`designs_exact` 6, `cyclic_golay` 7).

### m3. Costs of the public entry points (documented)

One cost line each in `rustmath-galois/src/steiner_labelling.rs`:

- `duad_resolvent(f, p)` (doc above line 658): `N` is about `253 log_p(2M)`. f2 `N = 267`
  1.3 s, f1 `N = 302` 1.4 to 1.6 s (release). CC0 `N` about 12,400, about 10 minutes (design
  Section 9 estimate, not run).
- `LabellingOptions::search_limit` (line 680): one factorization per prime up to
  `min(search_limit, MAX_P + 1)`; the whole range when there is no inert prime: 1228 primes for
  the default, 0.6 s in release (probe, `(x^11 - x - 1)(x^12 - x - 1)`).
- `extra_digits` (line 684): about `10 / log10 p` more `N` per 10 digits (7 at 29); f1, f2 at
  29 with the default: `N = 31, 30`, about 0.1 s.
- `precision` (line 687): no upper limit; `p^N` is built at once and the lift runs at that `N`.
- `retries` (line 692): each retry is a lift and a `T5` at the new `N`. An always `Unresolved`
  input (C23, `polsubcyclo(47, 23)`) spends every retry.
- `golay_invariants` (line 921): `N_J` about `7 log_p M`. f2 8, f1 9 (milliseconds); CC0 344
  (`cc0_golay_invariants` about 10 s in debug with its labelling).
- `heptad_resolvent` (line 961): `N` about `253 log_p(7M)`. f2 `N = 360` 1.9 s, f1 `N = 395`
  2.1 s (release, this run). CC0 `N` about 12,500, about 10 minutes (estimate, not run).
- `SteinerLabelling::duad_resolvent` (line 983): as the free form; the lift continues from the
  labelling's roots.

### m4. Empty or unequal-length candidates (fixed)

- `rustmath-galois/src/integrality.rs:54, 60`: new `IntegralityError::EmptyCandidate { index }`
  and `CandidateLengths { index, len, expected }`, with `Display`.
- `integrality_dichotomy` (line 170): after the precision check, a candidate with no entry is
  refused (it passed vacuously), and so is a candidate whose length differs from the first
  (line 176). An empty list of candidates still gives `Unresolved { NoneWithinBound }`
  (documented: the caller's premise fails).
- `SteinerError::Integrality` (steiner_labelling.rs line 219) now says that `PrecisionTooLow` is
  the only `IntegralityError` the labelling can cause (it builds two candidates of length 1).
- Test `malformed_candidates_are_refused` (integrality.rs line 314): the review's probe
  `[("empty", []), ("big", [400])]` is now `Err(EmptyCandidate { index: 0 })`; an empty last
  candidate, unequal lengths, and the empty list are covered too.

### m5. Test comment (fixed)

`rustmath-finitefields/tests/galois_ring_frobenius.rs:319` now reads "(Z - w)(Z + w) = Z^2 - w^2:
the constant term -w^2 is not in Z/125.", the review's wording.

### Results

All through `impl/cargo_sl.sh`, in `/home/user/RustMath-steiner`:

| Command | Result | Wall |
|---|---|---|
| `cargo test -j 2 -p rustmath-combinatorics` | all pass: lib 1509 (7 ignored), 4 integration files (`designs_exact` 5 at that time), doc 189 (11 ignored) | 38 s |
| `cargo test -j 2 -p rustmath-combinatorics --test designs_exact` (final) | 6 passed | 9 s |
| `cargo test -j 2 -p rustmath-coding` | all pass: lib 67, `cyclic_golay` 7, doc 5 | 17 s |
| `cargo test -j 2 -p rustmath-finitefields` | all pass: lib 205, 12 integration files (`galois_ring_frobenius` 11), doc 8 | 27 s |
| `cargo test -j 2 -p rustmath-galois` | all pass: lib 66 (was 65, + `malformed_candidates_are_refused`; 2 ignored), descent 12, `steiner_labelling` 13 (10 ignored), others unchanged | 52 s |
| final `cargo test -j 2 -p rustmath-galois --lib --test steiner_labelling` | 66 and 13 passed; fast set 12.8 s of test time | 29 s |
| `cargo test -j 2 --release -p rustmath-galois --test steiner_labelling --no-run` | built | 88 s |
| `M23ISGALOIS=/home/user/external/m23isgalois <binary> --ignored env_ --test-threads 1` | 8 passed, 28.3 s; same six integers, signs, rejected sizes (44, 45 digits), precisions | |
| `<binary> --ignored full_ --test-threads 1` | 2 passed, 1.65 s | |
| `<binary>` (fast set, release) | 13 passed, 1.42 s | |
| `<binary> --ignored env_six` without `M23ISGALOIS` | FAILED, "set M23ISGALOIS to a checkout of shaowuz/m23isgalois" | |
| `cargo clippy -j 2 --no-deps -p rustmath-finitefields -p rustmath-coding -p rustmath-galois --all-targets` | no warning in a changed file | |
| `cargo clippy -j 2 --no-deps -p rustmath-combinatorics --lib --test designs_exact -- --cap-lints warn` | one new warning in my test (`designs_exact.rs:117`, `std::slice::from_ref`), fixed and rechecked; all `designs.rs` warnings are on lines below 958 (pre-existing) | |
| `cargo doc -j 2 --no-deps -p rustmath-galois -p rustmath-combinatorics` | no warning in `steiner_labelling.rs`, `integrality.rs`, `designs.rs` | |
| `rustfmt --edition 2021 --check` on `integrality.rs`, `steiner_labelling.rs` (src and tests), `galois_ring_frobenius.rs` (tests), `designs_exact.rs` | clean (one import reordered, one test loop rewrapped) | |
| `rustfmt` on a copy of `designs.rs` | diffs only at lines 40, 60, 522, 550, 776, 1182 (pre-existing code); my hunk (961 to 1169) is clean | |

The edits after the release build were doc comments only (and the import order from rustfmt).
The final debug runs above cover the final state. No em-dash and no `unsafe` in the changed
lines. `target/release` was deleted afterwards (no other process used it). Free disk at the end:
8.2 GB.
