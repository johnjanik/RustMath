# Review 1 of the Steiner labelling design (SL.md)

Date: 2026-11-15. Reviewer: independent adversarial review. Object: `design/SL.md` (942 lines, no
Revision section yet) and the prototype `design/SL/` (`sl_proto.rs`, `gp/`, `oracle/`).
Scratch material of this review: `design/SL/review1/` (all paths below are relative to
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/m23/sl/`).
Nothing in the worktree `/home/user/RustMath-steiner` was edited (`git status` clean). No release
build was made (see Section 7).

## 0. Verdict

**Approve, with no blockers and nine minor issues (Section 9).** The mathematics is right. I
reproduced every number of the done criterion with my own PARI/GP code, which uses a third method
(Newton in `Mod(Mod(., p^N), f)`), different from both the designer's `polrootspadic` script and
`arith.py`. The Golay facts, the licence hash and the cost claims I sampled hold. The API has no
facade and no stability loop that decides.

The two most useful findings:

- Under a **false** hypothesis the rule can return Decided. Example: `x^23 - x - 1` (Galois group
  S23) at p = 157 and the minimal rigorous precision N = 4 gives Decided(-). Both PARI/GP and
  `arith.py` (with `extra_digits=0`) agree. Under the true hypothesis no wrong decision is possible.
  The design should say this plainly (m1).
- The design's trusted input can be **weakened**. "Gal(f) is contained in a conjugate of M23" is
  enough. Then a Decided labelling even proves Gal(f) = M23 (Section 1.3).

## 1. Mathematics

### 1.1 Frobenius-ordered roots (SL 2.1)

Checked: the Newton step order (y first with the current u, then u at the new y) doubles the
accuracy of both `f(y)` and `1 - u f'(y)`. Hensel uniqueness gives `y = phi(x)`. `phi` is
`Z/p^N`-linear with columns `y^j`. `phi^n = id` is then automatic, so the check is a bug detector,
as the design says. My gp implementation of exactly these steps (`scripts/rev.gp`, `froots`)
reproduces all values below, including the checks `y = t^p mod p` and `phi^23 = id`.

The prototype test `proto_small_lift` (rebuilt here against the worktree's debug rlibs) passes:
`x^3 + x + 1` at `5^6` and `phi(x)` of `x^23 - x - 1` at `23^4` equal the PARI values.

### 1.2 The decision rule (SL 2.3, paper Proposition 5.1)

The proof chain is correct:

1. `Gal(f)`, through the embedding into `Frac(R)`, is a subgroup of `Sym(Z/23)` containing the shift.
2. If it is conjugate to M23 (any transitive degree 23 action of M23 is the natural one, since M22
   is its only class of index 23 subgroups), it is `Aut(S)` for a shift-invariant S(4,7,23), so
   `S` is `S+` or `S-` (Lemma `lem:twocodes`).
3. It preserves exactly one system: systems form one `S_23`-orbit with stabilizer M23 and
   `N(M23) = M23`.
4. For the preserved system `T5^e` is an algebraic integer fixed by `Gal(f)`, so it lies in `Z`
   with `|T5^e| <= 253 (7M)^5 < B5`. With `p^N > 2 B5` its symmetric residue is `T5^e` itself.
5. So the true sign always passes. "Exactly one passes" decides. Using `B5 = floor(beta) + 1`
   instead of `beta` in the test only makes the rule more conservative.

`NeitherWithinBound` is in fact an **unconditional** proof that `Gal(f)` lies in no conjugate of
M23 (it contains the shift, so step 2 applies to any M23-conjugate containing it). See m7.

`retries`: each attempt at `p^N > 2 B5` is a proof on its own, so retrying is not a stability loop.
Agreed (D5).

### 1.3 A weaker hypothesis suffices (observation, part of m1)

The only maximal subgroup of M23 with order divisible by 23 is `23:11` (Le Lay's certificate lists
the ATLAS maximal subgroups). So a subgroup of M23 containing a 23-cycle is `C23`, `23:11` or M23.
The normalizer of the shift in `Aut(S+)` and in `Aut(S-)` is the same group
`{i -> a i + b : a a square mod 23}` of order 253 (2 has order 11 mod 23; `-1` is a non-square and
exchanges the systems). Hence if `Gal(f) <= 23:11` both `T5^+` and `T5^-` are integers within the
bound and the rule always returns `BothWithinBound`.

Consequence: assume only "Gal(f) is contained in a conjugate of M23". Then the rule is still
sound, and **Decided implies Gal(f) = M23**. This is exactly what a specialization of an M23 cover
gives (Le Lay's "subgroup of M23"), so for the CC0 polynomial the element of order 14 is not even
needed for the labelling. The C23 field (`polsubcyclo(47, 23)`) shows the `BothWithinBound` case
(Section 2).

### 1.4 A wrong decision under a false hypothesis (SL 2.3, 6)

`scripts/run_wrong.gp` scans five trinomials of degree 23 at all inert primes below 400, at
`N_rig .. N_rig + 3`. Four hits where exactly one candidate lies within `B5`:

| Polynomial | p | N (N_rig) | residues `T5^+`, `T5^-` | B5 | Rule says |
|---|---|---|---|---|---|
| `x^23 - x - 1` (S23, Osada 1987) | 157 | 4 (4) | 182056093, 128168772 | 136069473 | Decided(-) |
| `x^23 - x + 1` | 157 | 4 (4) | -182056093, -128168772 | 136069473 | Decided(-) |
| `x^23 + 3x + 1` | 29 | 6 (6) | 289769051, -5143997 | 177574130 | Decided(-) |
| `x^23 + 3x + 1` | 41 | 6 (6) | 105069325, -848406566 | 177574130 | Decided(+) |

The owner's `arith.py` agrees on the first row: `steiner_labelling(x^23-x-1, 157, extra_digits=0)`
returns `N = 4`, `sign = -1`, `t5 = {1: 182056093, -1: 128168772}`; with the default
`extra_digits=30` it raises "undecided" (neither within the bound). So the default margin protects
in practice, but the API also accepts `precision: Some(N)` and `extra_digits: 0`.

This is not a defect of the rule: the hypothesis is a trusted input, as the brief says. But SL.md
never says that a false hypothesis can give Decided. Its summary item 6 and the Section 6 row for
`x^23 - x - 1` ("refutes the hypothesis") may lead a reader to think a false input is detected.
`NeitherWithinBound` is only an occasional detection.

### 1.5 Root bound, bounds, precision (SL 2.4, 2.5)

Fujiwara's proof as written is correct (the strict inequality needs `|z| > 2 mu`, and
`sum 2^-k < 1`). The rounding `r_k = ceil(256 |a_{n-k}|^{1/k})` gives `best / 128 >= 2 mu`. My gp
`fbest` gives `best = 2157` (f2), `3473` (f1), the 74-digit CC0 value of SL 2.4, `1201` (C23) and
`256` (`x^23 - x - 1`), all equal to the design.

The bounds `253 (7M)^5`, `5313 M^5`, `253 M^7`, `(1 + 7M)^253`, `(1 + 2M)^253` are valid.
`J5 = sum_H e_5(alpha_H)` is right (a five-subset lies in at most one heptad). Using `e = 0` for the
integer-valued quantities is rigorous (the owner's `relift` uses 30 extra digits; the values are
the same, only the cost differs). Precisions I recomputed: f2 `B5 = 5778453176946`, `N_rig = 9`,
`N30 = 30`, `29^8 <= 2 B5`; f1 `B5 = 62529476039478`, `N_rig = 10`, `N30 = 31`; J precisions 8 (f2)
and 9 (f1); CC0 `B5` 364 digits, `N_rig = 250`, `N30 = 270`, `N_J = 344`.

### 1.6 Orbit compression (SL 2.6)

Correct: `phi(F(alpha_H)) = F(alpha_{H+1})`, orbits have length 23, `Tr(x^j) = P_j` (the power
sums of the roots of f, from Newton's identities, no division). The fixed ring of `phi` on
`GR(p^N, 23)` is `Z/p^N`, so each orbit polynomial has constant coefficients. Small wording point:
the orbit polynomial is the characteristic polynomial of `s_rep` over `Z/p^N` always; distinct
conjugates make it the minimal polynomial.

### 1.7 The T5 and J5 identity (SL 2.7) and a consequence (m5)

The multinomial bookkeeping is right. `scripts/ident.py` (my own code: power sums, monomial
functions by set-partition Moebius inversion) confirms `T5 = 120 J5 + S` for f1 and f2.

Consequence not in the design: `S` is the same for both systems, so
`T5^+ - T5^- = 120 (J5^+ - J5^-)`. The two candidates always agree modulo 8, 3 and 5. For C23 the
difference is 67680 = 120 * 564. So `frobenius_sign` at `q = 2, 3, 5` with `N = 1` can never
separate the signs (it returns `Unresolved`, which is honest). The labelling itself at p = 2, 3, 5
loses a few digits only.

## 2. Reproduced values (independent PARI/GP, `scripts/rev.gp`)

PARI/GP 2.15.4. Roots by Newton in `Mod(Mod(., p^N), f)` (not `polrootspadic`), heptads from bit
masks, direct sums over the 253 heptads.

| Item | Result | Design |
|---|---|---|
| f2 at 29, N = 9 | `T5^+ = -3209424281433` (within), `T5^- = -1059840`: BothWithinBound | same |
| f2 at 29, N = 10 | `T5^+ = -206309467943599` (outside): Decided(-) | same |
| f2 at 29, N = 30 | rejected `-29622039743232607524836197957123800868867866` (44 digits) | same |
| f1 at 29, N = 10 | `T5^+ = -114831914638610`, `T5^- = 92747040`: Decided(-) | same |
| f1 at 29, N = 31 | rejected `-883496073094568376895161675988763338141335547` (45 digits) | same |
| J5, J7 (f2, N = 8; f1, N = 9) | `61916, -83030`; `524492, -2281968` | same |
| f1 at 73 (N = 24, N_J = 7) | Decided(-), `T5 = 92747040`, `J5, J7` as at 29 | same |
| f2 at 223 (N = 19, N_J = 5) | Decided(-), `T5 = -1059840`, `J5, J7` as at 29 | same |
| 80 signs of `frobenius_signs.json` at N = 1 | 0 mismatches; f1 lists all 40 inert primes below 4000; f2 lists the first 40 and misses 3191, 3319, 3527, 3677, 3989 | same |
| CC0 monic associate at 29 | N = 270: `T5^-` 356 digits within, `T5^+` 395 digits outside; N = 250: same decision, same `T5` | same |
| CC0 J5, J7 at N = 344 | 353 and 491 digits, within the bounds | same |
| CC0 values | `T5^-`, rejected `T5^+`, `J5`, `J7` equal to `oracle/oracle_cc0.json` digit for digit | |
| CC0 signs, 12 inert primes below 900 | `[-,+,-,-,+,+,-,-,+,+,-,-]`, equal to the f1 rows | same |
| C23 at 2, 3, 5 | `[-98317, -30637]`, swapped at 3, as at 2 at 5; stable at N, N+10, N+20 | same |
| `x^23 - x - 1` at 23, N = 30 | `[-29593503946453640472557674395096324685942, 26819164821679513709525347975140930751221]`, `B5 = 136069473`, `N_rig = 7` | same |

Further checks:

- The CC0 file equals `G(2, W)` of the WP-F vendored model (Python, exact). Content 1, degree 23,
  leading coefficient as in SL 7.1. Its sha256 is `f775aa4c...45862`, line 84 of Le Lay's
  `CHECKSUMS.sha256`. `data/LICENSE` is CC0-1.0; the top-level LICENSE is CC BY-NC-SA, as the brief
  says.
- `subset_sum_resolvent(f2, 2)` (exact over Q, read in `resolvent.rs:251`) equals `R253_duad`
  (prototype `proto_symmetric_duad`, debug, 3.92 s).
- The improved prototype reproduces `R253_heptad` (N = 360) and `R253_duad` (N = 267) of f2
  exactly against my own reference files taken from `golay_resolvents.json`.
- Mutation: with the order `alpha_i = phi^{-i}(x)` (`scripts/run_mut.gp`) the two candidates swap.
  So the sign assertions catch an ordering bug that `T5` alone would not.

GAP was tried for `|Aut(S+)|` by a set-stabilizer search in `S_23`; it did not finish in 5 minutes
and was dropped. The group facts rest on the paper's Lemma and the ATLAS, as in the design.

## 3. Golay systems and S(4,7,23) (SL 2.2, 7.1)

`scripts/golay_check.py` (independent of `golay.py`): masks `0xae3`, `0xc75`;
`(z+1) g+ g- = z^23 - 1`; `g- = reverse(g+)`; both weight distributions
`1, 253, 506, 1288, 1288, 506, 253, 1`; both S(4,7,23) (8855 distinct 4-subsets, each once); shift
and `i -> 2i` invariant; 77, 21; intersections `{1: 14168, 3: 17710}`; `i -> -i` exchanges them;
they share no heptad. The 11 orbit representatives and the 5 heptads through `{0,1,2}` of both
systems are exactly the lists in SL 7.1. The 12 rows of `BinaryGolayCode::new()` are all divisible
by `g-` (and by `g+` only after reversing the coordinates). Prototype `proto_golay` passes in 12 ms
(debug).

## 4. Licence handling (SL 8)

- The derivation command reproduces the hash: `87123dfe2fe86cb6bf5b40e2a39c2d06a680c6a2df2e8fd8bce1adee4f87a381`,
  133,201 bytes, Python 3.13.18. Original `dcf1e38c...70d13` and `frobenius_signs.json`
  `30ec4670...17a4f` are as stated.
- The stripped file keeps `labelling_prime`, `sign_at_prime`, `T5`, `J5`, `J7`, `R253_heptad`,
  `R253_duad`, `verified_at_second_prime`. No other field holds f.
- **But f is still recoverable from it** (m4). The power sums of the duad sums give the power sums
  of the roots: `Q_k = ((46 - 2^k) P_k + sum_{0<m<k} C(k,m) P_m P_{k-m}) / 2`, and `46 - 2^k != 0`.
  `scripts/recover.py` rebuilds f2 exactly from `R253_duad` of the stripped file. The brief allows
  vendoring the resolvents, so this is compliant. The README should say it, and the owner should
  know.
- `gp/duad_ref.gp` holds `R253_duad` of f1 and f2 (owner data, same remark). `gp/cc0_poly.gp` is
  CC0. The `run_*.gp` drivers read f1 and f2 at run time from an absolute path; parameterize it
  before they go to `docs/`.
- `polynomial_f1.gp` and `polynomial_f2.gp` carry their own fingerprint line
  (`f1_expect = [...]`; the value is redacted in this copy, since it is a line of HJLPPZ's unlicensed file). Reading it at run time copies nothing.
  It checks the parser, not the file's identity; the `env_` value tests do the latter.

## 5. Tests (SL 7)

Good: every value has an external source (owner data, `arith.py`, PARI/GP), the `env_` tests panic
without `$M23ISGALOIS`, and the fast suite runs on the CC0 polynomial with two references. The
mutation in Section 2 shows the sign tests bite. Problems:

- The Section 6 row "Frobenius sign: both match T5 mod q^N: `Unresolved { matching: [+, -] }`,
  C23 (any t5)" is wrong. The two C23 candidates are distinct integers. Both match only when `t5`
  is one of them and `q^N` divides 67680, which holds at `q = 2, 3, 5` with `N = 1` (by Section
  1.7). With another `t5` (for example 0 at q = 2) neither matches. With `t5 = -98317`, `q = 5`,
  `N = 2` exactly one matches and the routine would return Decided. See m3.
- No explicit fast or `env_` assertion of `best` and `B5` for f1 and f2 (2157, 3473,
  5778453176946, 62529476039478; `arith.py` and my gp agree). The brief lists "the bounds". m8.
- Add a test for the false-hypothesis case of Section 1.4 (m1).

## 6. API (SL 5)

No facade: missing labels give `Err(NotDecided)`, values beyond a proven bound give
`BoundExceeded`, a non-constant orbit coefficient is `Err`, a precision at or below `2B` is refused,
and the bounded inert-prime search never decides. No stability loop decides (the `N + 7` recompute
is a cross-check only). Points to tighten:

- `frobenius_sign(f, q, t5, precision)` takes a bare integer. A wrong `t5` that happens to match one
  candidate mod `q^N` gives a wrong Decided (chance about `2 / q^N`). m2.
- `CertifiedInteger` and `CertifiedResolvent` have public fields, so any caller can build a
  "certified" value. They also do not record the hypothesis or the labelling they rest on. m6.
- `integrality_dichotomy` takes one integer per candidate. WP-A A4 has a vector of 15 coefficients
  per candidate (`U`, `U'`). m9.
- D4: `NeitherWithinBound` is a proof, not an absence of a decision (Section 1.2). m7.

`defining_polynomial()` is documented with `D[d] = 1`, which the existing constructor does not
guarantee (m5 below, demonstrated).

## 7. Cost (SL 9), measured samples

The prototype was rebuilt with `rustc 1.97.0 --test` against the worktree's debug rlibs
(`cargo build -j 2 -p rustmath-finitefields --message-format=json` to locate them; no file in the
worktree changed). One thread, the machine shared with another job.

| Sample (debug unless stated) | Measured here | SL.md |
|---|---|---|
| CC0 labelling (N = 270) and J5, J7 (N = 344), existing arithmetic (`proto_fast_cc0`) | 39.8 s | 42.6 s |
| Same, improved (`proto_fast_variant`) | 9.57 s | 9.6 s |
| f2 everything at 29 incl. both resolvents, improved | 48.8 s (resolvents 26.5 s and 17.8 s) | 48.5 s (27 s, 17 s) |
| Lazy versus existing product, f2 modulus, N = 31 / 270 / 400 / 1000 | ratio 2.5 / 2.2 / 2.0 / 2.0 (equal results) | release 2.3 to 4.4 |
| Same, CC0 modulus | ratio 3.1 / 2.2 / 2.1 / 2.1 (equal results) | |
| `subset_sum_resolvent(f2, 2)` | 3.92 s | 4.3 s |
| Golay systems and checks | 12 ms | 10 ms |
| `arith.py` (python-flint 0.9.0): f2 labelling / R heptad | 0.04 s / 9.2 s | 0.06 s / 9.0 s |
| My gp: CC0 T5 at 270 / J at 344 | 0.7 s / 4.1 s | |

Release timings were not re-measured. Creating and then deleting `target/release` on the target
shared with the running WP-F F-9 job could break that job. The debug numbers match the design
within 7 percent, so I have no reason to doubt the release table. The fast suite estimate (about
13 s of tests in debug) is plausible: the labelling and invariants alone take 9.6 s; the second
prime, the minimal-precision run and the signs add a few seconds.

## 8. Existing code claims checked

- `GaloisRing::with_modulus` (`galois_ring.rs:105`): proves p (`try_is_prime`), checks
  `D mod p` monic and irreducible (`is_irreducible_fp_unchecked`, a correct Rabin test).
- `Mul` (`:984`): schoolbook with a reduction after every addition, as described.
- **Existing defect (m5).** `with_modulus` accepts `D` with `D[d] = 1 mod p` but `D[d] != 1 mod p^a`,
  while `reduce` (`:269`) assumes `D[d] = 1`. Demonstrated (`review1/dcheck.rs` against the debug
  rlib): `GR(5^2, 6x^2 + x + 2)` is accepted and `D(generator) = 15 + 20x != 0`. The generator is a
  root of `x^2 + x + 2`, not of `D`. FrobeniusRoots is safe (it refuses non-monic f), but
  `FrobeniusMatrix::new(ring)` takes any ring, and the new `defining_polynomial()` doc would be
  false for such a ring.
- Dependencies: `rustmath-galois` does not yet depend on `rustmath-coding`; adding it creates no
  cycle. `rustmath-combinatorics` is already in its graph through `rustmath-groups`.
  `rustmath-rings` (home of `UnramifiedExtension`) would be a much heavier dependency, so putting
  `FrobeniusMatrix` on `GaloisRing` is the right call.
- `BlockDesign::new` verifies only for `t <= 2`, `v <= 10`, and `machine_word_counts.rs:678`
  builds a `v = 2^33` design: a separate checker is needed, as designed.

## 9. Issues

All minor. None blocks the implementation.

- **m1 (docs, test): a false hypothesis can give Decided.** State in SL 2.3, the module docs and the
  `steiner_labelling` docs that Decided is relative to the trusted input and that a false input can
  be Decided at a small margin. Give the example `x^23 - x - 1` at p = 157, N = 4 (Decided(-); PARI/GP
  and `arith.py`). Add a fast test that records it (`precision: Some(4)` gives Decided(-); the
  default margin gives `NeitherWithinBound` there, per `arith.py`). Record the weaker sufficient
  hypothesis of Section 1.3 ("Gal(f) is contained in a conjugate of M23"; then Decided proves
  Gal(f) = M23). Optional: a sound refutation screen by Frobenius cycle types at a few primes
  (a type outside the 17 of M23 refutes; a clean screen proves nothing).
- **m2 (API): `frobenius_sign` with an unverified T5.** Take the Decided `SteinerLabelling` (or a
  `CertifiedInteger` from it) instead of a bare `&Integer`, or keep the bare form but name it
  `frobenius_sign_given_t5` and record the given `t5` as a trusted input in the result.
- **m3 (test spec): the C23 "both match" row is wrong.** Replace "C23 (any t5)" by: C23 with
  `t5 = -98317` at q = 2, 3 or 5 with N = 1 gives `matching = [+, -]` (the candidates differ by
  67680 = 120 * 564); and show that `t5 = 0` at q = 2 gives `matching = []`.
- **m4 (licence transparency): f is recoverable from the stripped file.** The README of
  `rustmath-galois/tests/data/m23/` should say that `R253_duad` (and so the file) determines f1 and
  f2 by Newton's identities, and that vendoring it rests on the brief's permission for the
  resolvents. Mention it in the report to the user so the owner can decide.
- **m5 (existing defect in the file being edited): `GaloisRing::with_modulus` with `D[d] != 1`.**
  In step 1, reject `D` unless `D[d] = 1 mod p^a` (or normalize `D` by the inverse of `D[d]` mod
  `p^a`), with a test on `GR(5^2, 6x^2 + x + 2)`. Then `defining_polynomial()` can promise
  `D[d] = 1`.
- **m6 (API): certificates can be forged.** Make the fields of `CertifiedInteger` and
  `CertifiedResolvent` private with getters, built only by the certifying code. Record in them the
  hypothesis and the labelling (prime, precision, sign) they depend on; the duad resolvent needs no
  hypothesis and can say so.
- **m7 (D4): `NeitherWithinBound` is a refutation.** It proves, without any hypothesis, that
  `Gal(f)` lies in no conjugate of M23. Prefer a third status (`HypothesisRefuted` or
  `NotInM23`) over an `Unresolved` variant. If the variant stays, its doc must say it is a proof.
- **m8 (tests): assert the f1, f2 bounds.** In `env_six_integers_and_signs` assert `best` (2157,
  3473), `B5` (5778453176946, 62529476039478) and the J bounds (7220058114, 97634149933;
  78129291179, 2738950710192), source `arith.py` and PARI/GP.
- **m9 (reuse for WP-A A4): vector-valued candidates.** Let the dichotomy take
  `&[(L, Vec<Integer>)]` (a candidate passes when every entry passes), so that SL-6
  `rationality_dichotomy` is a drop-in for the 15 coefficients of `g15`. Also note in SL 2.8 that
  `frobenius_sign` cannot separate at q = 2, 3, 5 with `N <= v_q(120)` (Section 1.7); using `J5`
  instead of `T5` there avoids the factor 120 and has an 800 times smaller bound.

## 10. Answers to the decisions (SL 12)

- D1 (lazy `Mul` in place): agree. Equal results confirmed here on both moduli up to N = 1000.
  Fold m5 into the same step.
- D2 (four crates, four commits): agree. Each piece has an owner crate and a later user.
- D3 (orbit compression): agree. The identities are exact and the direct sums stay as tests.
- D4: see m7.
- D5 (`extra_digits = 30`, `retries = 0`): agree, with the m1 caveat in the docs.

## 11. Files of this review

`design/SL/review1/`:

- `scripts/rev.gp` (my PARI/GP reference), drivers `run_f.gp`, `run_j.gp`, `run_cc0.gp`,
  `run_signs.gp`, `run_wrong.gp`, `run_mut.gp`; `cc0_raw.gp`, `signs_f1.gp`, `signs_f2.gp`
  (generated from CC0 and owner data); `cc0_rev_t5.txt` (CC0 `T5^-`, rejected `T5^+`, `J5`, `J7`).
- `scripts/golay_check.py`, `ident.py`, `recover.py`, `oracle_review.py` (run with
  `venv/bin/python -I`, python-flint 0.9.0).
- `sl_proto_review.rs` (the prototype with its reference directory moved to `review1/ref/`),
  `ref/` (resolvents from `golay_resolvents.json`), `cc0_monic.txt`, `f2_fast_debug.log`,
  `dcheck.rs` (the `with_modulus` defect), `golay_resolvents_no_f.json` (re-derived, hash checked).
  The two test binaries were deleted after the runs.

Commands: `gp -q -f <driver> < /dev/null` in `review1/scripts/`; the prototype as
`rustc --edition 2021 --test sl_proto_review.rs -L dependency=<worktree>/target/debug/deps --extern ...`
then `./sl_proto_debug --ignored --nocapture --test-threads 1 <name>` with the `SL_*` variables of
the design's appendix.
