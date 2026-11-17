# Steiner labelling exercise: design

Date: 2026-11-15. Worktree `/home/user/RustMath-steiner`, local branch `claude/m23-steiner`, based on
`claude/m23-wpf` `d91f880` (WP-F, PR #581). Brief: `docs/workstreams/m23_golay/HANDOFF.md`, Sections 1,
2 (Galois ring entries and "Do not call"), 3 (Status, as the model for records), 5.3 A4, 6 (last
paragraph), 7 and 8 step 2.

Scratch material (all paths under
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/m23/sl/`):

- `design/SL/sl_proto.rs`: the Rust prototype. It ran as a temporary integration test of
  `rustmath-finitefields` in the worktree and was deleted from the worktree afterwards.
- `design/SL/gp/`: the independent PARI/GP reference (`steiner.gp` and the `run_*.gp` drivers).
- `design/SL/oracle/`: the scripts that run the owner's `arith.py` as an oracle (python-flint 0.9.0
  in the venv `venv/`), and their JSON output.

Nothing was committed. The worktree is clean. `target/release` and the prototype test binaries were
deleted.

## 0. Summary

1. **Done criterion reached in the prototype.** On the worktree crates (`GaloisRing` of
   `rustmath-finitefields`, reading f1 and f2 at run time from
   `/home/user/external/m23isgalois/polynomial_f*.gp`), at p = 29:
   - f2: `T5 = -1059840`, `J5 = 61916`, `J7 = -83030`, sign -1;
   - f1: `T5 = 92747040`, `J5 = 524492`, `J7 = -2281968`, sign -1;
   - the rejected candidate at the reference precision: f2 (N = 30)
     `-29622039743232607524836197957123800868867866` (44 digits), f1 (N = 31)
     `-883496073094568376895161675988763338141335547` (45 digits), equal to the oracle and to PARI/GP;
   - all four degree 253 resolvents equal to `data/golay_resolvents.json`;
   - the second-prime check: f1 at 73 and f2 at 223 give sign -1 and the same six integers and the
     same four resolvents (`verified_at_second_prime` reproduced).
2. **The CC0 polynomial works for the fast tests.** Le Lay's certified specialization
   `data/specialization_t2_Z.json` (equal to `G(2, W)` of the CC0 model already vendored by WP-F) is
   not monic. Its monic associate `g(x) = a^22 f(x/a)` has coefficients of up to 1605 digits. Its
   labelling at 29 is Decided (sign -1) at the minimal rigorous precision N = 250. `T5` (356 digits),
   `J5` (353 digits), `J7` (491 digits), the rejected candidate at N = 270 (395 digits) and twelve
   Frobenius signs agree with two independent references (arith.py with FLINT, and PARI/GP with
   `polrootspadic`).
3. **A new cross-check.** The CC0 polynomial has the same inert primes and the same Frobenius signs as
   f1 at every prime tested (twelve). This is expected: f1 is the polredbest form of the same
   specialization (HJLPPZ Example 1.2), so the two polynomials define the same field. A fast test can
   therefore check the CC0 signs against the owner's vendored f1 sign data without `$M23ISGALOIS`.
4. **Three algorithmic choices beyond the reference.** They change the cost, not the result.
   - Frobenius is a linear map. Its matrix (columns `phi(x)^j`) costs 22 ring products once. Each
     further root is then one matrix-vector product. The reference composes by Horner (22 ring
     products per root).
   - Orbit compression. The 253 heptads, the 253 duads and the 5313 five-subsets fall into 11, 11 and
     231 orbits of the shift. A sum over a shift-invariant family is a sum of traces over orbit
     representatives. A product over the family is a product of 11 orbit polynomials of degree 23.
     This needs 11 times fewer ring products than the reference.
   - Lazy reduction in `GaloisRingElement::mul`: accumulate the convolution, reduce once. Same
     output, 2.3 to 4.4 times faster (measured).
5. **Cost (release, 2 threads).** Everything for f1 and f2 (labelling, J5, J7, both resolvents, both
   primes): about 15 s with the improved arithmetic, about 50 s with the existing arithmetic (25 s
   measured at 29, the second primes estimated alike). The
   FLINT oracle needs 33 s for the first prime alone. Debug fast suite on the CC0 polynomial: about
   13 s improved, about 45 s with the existing arithmetic. Debug is too slow for the f1/f2 resolvents
   (about 50 s for f2 alone), so the `env_` tests run in release.
6. **Degenerate cases with reference values.** f2 at the minimal rigorous precision N = 9 is
   Unresolved: both candidates fall within the bound. The degree 23 subfield of `Q(zeta_47)`
   (Galois group C23, inside both `Aut(S+)` and `Aut(S-)`) is Unresolved at every precision, with the
   two integer candidates -98317 and -30637. `x^23 - x - 1` (Galois group S23) gives neither candidate
   within the bound, which refutes the hypothesis.
7. **Where the pieces go.** `rustmath-finitefields`: the Frobenius matrix, Frobenius-ordered roots,
   orbit polynomials, lazy multiplication. `rustmath-combinatorics`: an exact t-design test (MAGMA
   `IsDesign`, `IsSteiner`). `rustmath-coding`: the two cyclic Golay codes and their heptads.
   `rustmath-galois`: bounded integrality and the dichotomy (MAGMA `IsInt`), and the labelling with
   its invariants. Four commits on one branch (decision D2).

## 1. Sources and conventions

Mathematics and reference code:

- The paper `paper/golay_m23.tex` of the owner's M23 repository at `d05475b`: Section 2 (the two
  cyclic codes, Lemma `lem:twocodes`, Definition `def:sign`), Section 5 (Proposition `prop:labelling`
  with its proof, Theorem `thm:invariants` with its proof, Lemma `lem:sums`, the Remark on Frobenius
  signs), Section 6 (the duad resolvent as a symmetric resolvent), Section 8 (`prop:J5`, the
  definitions of `J5` and `J7`), Section 9 ("p-adic labelling").
- `m23golay/arith.py` (`root_bound`, `frobenius_lift`, `steiner_labelling`, `golay_invariants`,
  `heptad_resolvent`, `duad_resolvent`, `frobenius_sign`, `tschirnhaus`) and `m23golay/golay.py`
  (`G_PLUS = 0xAE3`, `G_MINUS = 0xC75`, `heptads`, `verify_basic_facts`). The owner's code may be
  ported as an algorithm. The port cites the paper.
- `data/golay_resolvents.json`, `data/frobenius_signs.json`, `scripts/verify_resolvents.py`,
  `tests/test_core.py`.

Literature for the headers:

- H. Fujiwara, Ueber die obere Schranke des absoluten Betrages der Wurzeln einer algebraischen
  Gleichung, Tohoku Math. J. 10 (1916) 167-171 (the root bound).
- E. Witt, Die 5-fach transitiven Gruppen von Mathieu, and Ueber Steinersche Systeme, Abh. Math. Sem.
  Hamburg 12 (1938) (uniqueness of S(4,7,23), automorphism group M23).
- F. J. MacWilliams, N. J. A. Sloane, The Theory of Error-Correcting Codes, North-Holland 1977,
  Ch. 16 (quadratic residue codes, the binary Golay code as a cyclic code).
- R. P. Stauduhar, The determination of Galois groups, Math. Comp. 27 (1973) 981-996, and K. Geissler,
  J. Klueners, Galois group computation for rational polynomials, J. Symbolic Comput. 30 (2000)
  653-674 (p-adic roots ordered by Frobenius, bounded integrality of invariants).
- B. R. McDonald, Finite Rings with Identity, Dekker 1974, Ch. XVI, and Z.-X. Wan, Lectures on Finite
  Fields and Galois Rings, World Scientific 2003, Ch. 14 (Frobenius of a Galois ring, its fixed ring,
  the trace).
- J. von zur Gathen, J. Gerhard, Modern Computer Algebra, 3rd ed., Cambridge 2013, Sections 9.4 and
  15.4 (Newton iteration and Hensel lifting).
- H. Osada, The Galois groups of the polynomials X^n + aX^l + b, J. Number Theory 25 (1987) 230-238
  (`x^n - x - 1` has Galois group S_n; used by one degenerate test).
- HJLPPZ, arXiv:2608.08538v2 (f1, f2: Examples 1.2 and 3.17). Le Lay, data of
  `frontier-math-m23-cover-investigation` (CC0), `data/specialization_t2_Z.json`.

MAGMA chapter digests (`rustmath-lava/docs/`): ch. 38 (`GaloisRoot`, `IsInt(x, B, S)`, `Bound`,
`Stauduhar`), ch. 47 (unramified extensions, `GaloisImage`, `Trace`, `TeichmuellerLift`), ch. 48
(`GaloisRing`, `Eltseq`), ch. 147 (`IsDesign`, `IsSteiner`), ch. 152 (`CyclicCode`, worked example
H152E5: the Golay code as a cyclic code; `QRCode`), ch. 23 (univariate polynomials).

Conventions (those of `arith.py`): roots are indexed by `Z/23`, `alpha_i = phi^i(x)`, 0-based.
Sign `+` is the system of `G+` (generator `g+ = z^11+z^9+z^7+z^6+z^5+z+1`, mask `0xAE3`, bit i the
coefficient of `z^i`), sign `-` that of `G-` (`g- = z^11+z^10+z^6+z^5+z^4+z^2+1`, `0xC75`). Coefficient
vectors are little-endian. "Symmetric residue" of `a` mod `m` is the representative in `(-m/2, m/2]`.

## 2. Mathematics

### 2.1 Frobenius-ordered roots (arith.py `frobenius_lift`)

Input: monic `f` in `Z[x]` of degree `n >= 2`, a prime `p` with `f mod p` irreducible, a precision
`N >= 1`. Let `R = (Z/p^N)[x]/(f) = GR(p^N, f)`. It is the reduction mod `p^N` of the ring of integers
of the unramified extension of `Q_p` of degree `n`, and `f` splits in it.

1. `y_1 = x^p mod (p, f)`, computed in `GR(p, f) = GF(p^n)`. `u_1 = 1/f'(y_1)` in `GF(p^n)`. It
   exists because `f mod p` is irreducible, hence separable over the perfect field `F_p`.
2. Newton, doubling the precision `k -> min(2k, N)`, all arithmetic mod `p^k` and `f`:
   `y <- y - f(y) u` with the current `u`, then `u <- u (2 - f'(y) u)`. If `f(y) = 0` and
   `u f'(y) = 1` modulo `p^{k/2}`, both hold modulo `p^k` after the step (the error terms are products
   of two quantities divisible by `p^{k/2}`).
3. Checks at `p^N`: `f(y) = 0` in `R` and `y = x^p` mod `p`. By Hensel's lemma the root of `f` in
   `R` congruent to `x^p` mod `p` is unique, so `y = phi(x)` mod `p^N`, where `phi` is the Frobenius
   automorphism of `R` (the unique automorphism that lifts `a -> a^p` on the residue field).
4. `phi(sum a_j x^j) = sum a_j y^j` because `phi` fixes `Z/p^N`. So `phi` is `Z/p^N`-linear with the
   matrix `Phi` whose column `j` is the coefficient vector of `y^j` (22 ring products for
   `n = 23`).
5. Roots `alpha_0 = x`, `alpha_{i+1} = Phi alpha_i` for `i < n - 1`. Check `Phi alpha_{n-1} = x`
   (`phi^n = id`). The `alpha_i` are the reductions mod `p^N` of the `n` distinct p-adic roots of `f`.

Refusals, before any lifting: `f` not monic; `n < 2`; `N = 0`; `p` not proven prime; `f mod p` not
squarefree; `f mod p` squarefree but reducible (the error carries the factor degrees, from
`fp_factor::factor_with_multiplicity` when `p <= MAX_P`). A failed check in 3 or 5 is an `Err`
(it would mean an arithmetic bug), never a panic and never a returned value.

The prototype matches PARI/GP on two small cases. `x^3 + x + 1` at `p = 5`, `N = 6` gives the roots
`[0,1,0], [15371,571,15244], [254,15053,381]`. `x^23 - x - 1` at `p = 23`, `N = 4` gives
`phi(x) = [1, 49750, 206034, 90114, 100901, ..., 12719]` (`gp/small_lift.gp`; PARI computes the roots
with `polrootspadic` over `Q_p[t]/(f(t))` and orders them by `t^(p^i)` mod `p`, an independent
method). The second case has `p` equal to the degree: nothing in the method divides by an integer.
The Newton image also equals the existing Teichmueller-digit Frobenius
`GaloisRingElement::frobenius()` of the generator for f2 at N = 4, 10 and 30.

### 2.2 The two cyclic Golay systems (golay.py; paper Section 2)

Over `F_2`, `z^23 - 1 = (z - 1) g+(z) g-(z)`, and `g-` is the reciprocal of `g+`. The cyclic code
generated by `g+` (resp. `g-`) has the 4096 words `m(z) g(z)`, `deg m < 12`. Its 253 words of weight 7
("heptads") form a Steiner system `S+` (resp. `S-`) `S(4,7,23)` on `Z/23`. Both are invariant under
the shift `i -> i + 1` and under `i -> 2i` (2 is a square mod 23). The map `i -> -i` exchanges them.

Lemma (paper, `lem:twocodes`; Le Lay 5.4): a 23-cycle preserves exactly two Steiner systems
`S(4,7,23)`. For the shift they are `S+` and `S-`. The proof counts pairs (system, 23-cycle) using
Witt's uniqueness and `N_{S23}(M23) = M23`.

Facts the tests check (from `golay.py` `verify_basic_facts`): weight distribution
`1, 253, 506, 1288, 1288, 506, 253, 1` in weights `0, 7, 8, 11, 12, 15, 16, 23`; every 4-subset in
exactly one heptad; a point in 77 heptads, a pair in 21, a triple in 5; two heptads meet in 1 or 3
points (14168 and 17710 pairs); 11 shift orbits of 23 heptads each (no heptad is fixed by a non-trivial
shift because 23 is prime).

The existing `BinaryGolayCode::new()` of `rustmath-coding` generates exactly `G-` in these coordinates
(each of its 12 generator rows, read as `sum r_i z^i`, is divisible by `g-`; checked in the
prototype phase). The HANDOFF's remark that its coordinates are "the reciprocal of the polynomial in
4.2" is consistent with this.

### 2.3 The labelling criterion and why the decision is a proof (paper, Proposition 5.1)

Let `f` be monic of degree 23 with `Gal(f) = M23` in its natural action. Let `p` be inert and the
`alpha_i` as in 2.1. Put `s_H = sum_{i in H} alpha_i` and
`T5^e = sum_{H in S^e} s_H^5` in `Z_p` for `e = +, -`.

Proof structure (paper):

1. An embedding of the splitting field into `Frac(R)` identifies `Gal(f)` with a subgroup of
   `Sym(Z/23)` isomorphic to M23 and containing the shift (the image of `Frob_p`).
2. By the Lemma it is `Aut(S+)` or `Aut(S-)`. It is not both: M23 preserves exactly one Steiner
   system. Call the preserved one `S^e`.
3. `T5^e` and `T5^-e` are fixed by `phi`, hence lie in `Z_p`. `T5^e` is fixed by `Gal(f)` and is an
   algebraic integer, so it lies in `Z`, and `|T5^e| <= 253 (7M)^5 < B5` (2.5).

Decision rule. Fix `N` with `p^N > 2 B5`. Let `r^+`, `r^-` be the symmetric residues of `T5^+`,
`T5^-` mod `p^N`.

- Since `|T5^e| < B5 < p^N / 2`, `r^e = T5^e`, so `|r^e| <= B5`. The true sign always passes.
- If exactly one sign passes `|r| <= B5`, it is `e`. The labelling is **Decided**: `S^e` is the Steiner
  system of `f`, `epsilon(Frob_p) = e`, and `T5(f) = r^e`.
- If both pass: **Unresolved** (`BothWithinBound`). A larger `N` may separate them. Each attempt is a
  proof on its own, so trying a few larger `N` is allowed; if none separates, the answer stays
  Unresolved.
- If neither passes: **Unresolved** (`NeitherWithinBound`). This contradicts step 3, so it proves
  that `Gal(f)` is contained in neither `Aut(S+)` nor `Aut(S-)`. Since `Gal(f)` contains the shift, it
  is in no conjugate of M23: the hypothesis `Gal(f) = M23` is false (or the input is not what the
  caller thinks).

The hypothesis `Gal(f) = M23` is a trusted input, as the genus is in WP-F. It is recorded in the
result and never checked. For f1 and f2 it is HJLPPZ's theorem (their `verify.m`). For the CC0
polynomial it is Le Lay's certificate: the polynomial specializes an M23 cover, so its group is a
subgroup of M23, and it contains elements of orders 23 (irreducible mod 29) and 14 (factor degrees
2, 7, 14 mod 11); the only maximal subgroup of M23 with order divisible by 23 is 23:11, of odd order.
The tests re-check the two modular facts.

Why `extra_digits`: at the least `N` with `p^N > 2 B5`, the wrong residue is a random element of
`(-p^N/2, p^N/2]`, so it falls within the bound with probability about `2 B5 / p^N`, which can be
close to 1. The reference uses `p^N > 2 B5 10^30`. f2 at 29 shows the effect: the least rigorous
precision is N = 9, and there `r^+ = -3209424281433` lies within `B5 = 5778453176946`
(Unresolved); N = 10 decides. The margin changes nothing in the proof. The default
`extra_digits = 30` reproduces the reference precisions (N = 30, 31, 270) and so the reported rejected
candidates.

### 2.4 The rigorous root bound (arith.py `root_bound`)

For monic `f = x^n + a_{n-1} x^{n-1} + ... + a_0` put `mu = max_{1<=k<=n} |a_{n-k}|^{1/k}`.
Fujiwara: every complex root `z` has `|z| <= 2 mu`. Proof: if `|z| > 2 mu` then
`|a_{n-k} z^{n-k}| <= mu^k |z|^{n-k} < 2^{-k} |z|^n`, so
`|f(z)| >= |z|^n - sum_{k=1}^n |a_{n-k}| |z|^{n-k} > |z|^n (1 - sum 2^{-k}) > 0`. (Fujiwara's sharper
form uses `|a_0/2|^{1/n}`; the reference and this design use `|a_0|^{1/n}`, which is weaker and still
valid.)

Exact rounding: `best = max_k r_k` with `r_k` the least integer such that `r_k^k >= 256^k |a_{n-k}|`,
from `Integer::nth_root` (floor) plus one unless exact. Then `best / 256 >= mu`, and
`M = 2 best / 256 = best / 128` is a rigorous bound, rational with denominator dividing 128. All later
bounds are computed exactly from `best`.

Values: f2 `best = 2157` (`M = 16.8515625`), f1 `best = 3473` (`M = 27.1328125`), CC0
`best = 37006872220001887110556356679375229330854157360435885765997306615192150016`
(`M = 2.89e71`), C23 `best = 1201`, `x^23 - x - 1` `best = 256` (`M = 2`). All equal to `arith.py`.

### 2.5 Bounds and precision

Each bound `B` is `floor(beta) + 1` for the real bound `beta` (exact integer arithmetic on `best`),
as in `arith.py`. A quantity is returned as an integer only from a precision with `p^N > 2B`.

| Quantity | Definition | Integer because | Bound `beta` | Proof |
|---|---|---|---|---|
| `T5^e` | `sum_{H in S^e} s_H^5` | `Gal`-invariant algebraic integer (2.3) | `253 (7M)^5` | `|s_H| <= 7M` |
| `J5` | sum over the 5313 five-subsets in a heptad of the product of their roots | M23-invariant | `5313 M^5` | 5313 terms of size `<= M^5` |
| `J7` | `sum_H prod_{i in H} alpha_i` | M23-invariant | `253 M^7` | 253 terms |
| heptad resolvent `R(Z) = prod_H (Z - s_H)` | coefficient of `Z^{253-k}` is `(-1)^k e_k(s)` | M23 permutes the `s_H` | `(1 + 7M)^253` | `|e_k(s)| <= C(253,k)(7M)^k <= sum_k C(253,k)(7M)^k` |
| duad resolvent `D(Z) = prod_{i<j}(Z - alpha_i - alpha_j)` | symmetric in the roots | S23-invariant (no labelling) | `(1 + 2M)^253` | as above with `|alpha_i + alpha_j| <= 2M` |

`J5` is a sum over heptads: a five-subset of a heptad lies in no other heptad (it contains a
four-subset, which lies in one heptad), so `J5 = sum_H e_5(alpha_H)`, with `253 * C(7,5) = 5313`
terms.

Precision rule: `N(B, e) = least N with p^N > 2 B 10^e`. The labelling uses `e = extra_digits`
(default 30). The integer-valued quantities use `e = 0`, which is already rigorous. A caller-given `N`
with `p^N <= 2B` is refused (`PrecisionTooLow`), never computed.

### 2.6 Orbit compression (traces and orbit polynomials)

`phi(alpha_i) = alpha_{i+1}`, so `phi(s_H) = s_{H+1}`: `phi` acts on any family of root subsets by
the shift. The heptads (both systems), the duads and the 5313 five-subsets each split into shift
orbits of length 23 (23 is prime and no proper non-empty subset of `Z/23` is shift-invariant):
11, 11 and 231 orbits. Duad orbit representatives are `{0, d}` for `d = 1..11`.

- **Sums.** For a function `F` of a subset and a shift-invariant family `S`,
  `sum_{H in S} F(alpha_H) = sum_{reps} Tr(F(alpha_rep))`, where
  `Tr(a) = sum_{i<23} phi^i(a)` is the trace of `R` over `Z/p^N` (McDonald XVI, Wan 14). With
  `a = sum a_j x^j`, `Tr(a) = sum_j a_j P_j`, where `P_j = sum_i alpha_i^j` are the power sums of the
  roots of `f`. They are integers computed exactly by Newton's identities over `Z`
  (`P_0 = 23`, `P_k = -k a_{n-k} - sum_{j<k} a_{n-j} P_{k-j}`), then reduced mod `p^N`. No division
  is involved, so any `p` works. `T5`: 11 representatives, 3 ring products each. `J7`: 6 products
  each. `J5`: `e_5` of 7 elements by the usual recurrence, 35 products each.
- **Products.** `prod_{H in S}(Z - s_H) = prod_{reps} O_rep(Z)` with
  `O_rep(Z) = prod_{i<23}(Z - s_{rep+i})`. `phi` permutes the factors of each `O_rep`, so its
  coefficients lie in the fixed ring of `phi`, which is `Z/p^N`. The values `s_{rep+i}` are sums of
  shifted roots (no Frobenius needed). Each `O_rep` costs `1 + 2 + ... + 23 = 276` ring products, and
  each of its 24 coefficients is checked to be a constant (a bug detector: a non-constant
  coefficient is an `Err`). The 11 orbit polynomials are multiplied in `(Z/p^N)[Z]`
  (`rustmath_polynomials::zx::mul`, then reduction). Total about 3036 ring products, against about
  32000 for the reference's direct product of 253 linear factors over `R`.

The direct sums (253 heptads, the sum checked to be a constant) are kept as a test cross-check. In
the prototype they agree with the trace route at every precision.

The orbit polynomial `O_rep` is the characteristic polynomial of `s_rep` over `Z/p^N` when the 23
conjugates are distinct. WP-A A4 needs exactly this: `g15` is the orbit polynomial of `tau_B` over a
Frobenius orbit of 15 octads in `GR(p^N, 15)`.

### 2.7 A symmetric identity linking T5 and J5 (new cross-check)

Expanding `s_H^5` and summing over heptads with `lambda_1 = 77`, `lambda_2 = 21`, `lambda_3 = 5`,
`lambda_4 = 1` (the 4-design property) gives, for either system,

`T5 = 120 J5 + 77 m_5 + 105 m_41 + 210 m_32 + 100 m_311 + 150 m_221 + 60 m_2111`,

where `m_lambda` are the monomial symmetric functions of all 23 roots (integers from the coefficients
of `f`). The coefficients are multinomial coefficients times `lambda_t` (for example `m_41`:
`5!/(4!1!) = 5` times `lambda_2 = 21`). Checked exactly for f1, f2 and the CC0 polynomial
(`oracle/t5_j5_identity.py`, computing `m_lambda` from power sums by set partitions). It is independent
of all p-adic code, so it tests `T5` against `J5`. It also shows that `J5` alone would decide the
labelling with a bound 800 times smaller (`5313 M^5` against `253 (7M)^5`). The design keeps `T5`, as
in the paper.

### 2.8 Frobenius signs at further primes (paper, Remark; arith.py `frobenius_sign`)

Given `T5(f)` (an integer from a Decided labelling) and another inert prime `q`, compute the two
candidates `T5^+` and `T5^-` mod `q^N` (default `N = 1`, a computation in `GF(q^23)`). The true sign
`e(Frob_q)` gives `T5^e = T5(f)` exactly, so its residue matches. If exactly one matches, the sign is
**Decided** (given `Gal(f) = M23` and the value of `T5`). If both match: **Unresolved** (raise `N`).
If neither: the inputs are inconsistent (`T5` wrong or hypothesis false). At all 80 primes of
`data/frobenius_signs.json`, PARI/GP separates the signs at `N = 1` and agrees with the file.

### 2.9 Second-prime check (`verified_at_second_prime`)

The whole computation at a second inert prime (73 for f1, 223 for f2) gives the same integers. The
Frobenius labelling there is unrelated to the one at 29, so this is a strong test (paper, after
Theorem 5.2). Reproduced in the prototype: both labellings Decided with sign -1, the same six
integers and the same four resolvents.

## 3. Existing code: what is used and what is not

| Code | Assessment (measured in release unless stated) | Use |
|---|---|---|
| `GaloisRing::with_modulus` (`galois_ring.rs:105`) | Accepts `f` itself as the modulus at any precision `p^N`. Checks `p` prime (`try_is_prime`), `D` monic mod `p`, `D` irreducible mod `p` (`is_irreducible_fp_unchecked`), then builds the residue field. 18 ms per call for degree 23 (the irreducibility proof is repeated at every call). The reduction assumes `D[d] = 1` exactly; `f` monic over `Z` satisfies it. | Yes: one ring at precision 1 (this is the inertness proof) and one at the target precision. The Newton steps work on coefficient vectors and do not build rings. |
| `GaloisRingElement` `Mul` (`:984`) | Schoolbook with a reduction mod `p^a` after every addition and clones. 0.33 ms at N = 31, 0.94 ms at N = 270, 1.5 ms at N = 400 (modulus f2). With the CC0 modulus (full-size coefficients) 2.2 ms at N = 270. | Yes, after the lazy-reduction rewrite (5.1). |
| `GaloisRingElement::unit_inverse` (`:732`) | Residue inverse then Newton. Correct. | Only through the residue field at precision 1. |
| `GaloisRingElement::frobenius` (`:664`) | Teichmueller digits: 0.08 s at N = 4, 0.31 s at N = 10, 5.5 s at N = 30, per call. Equal to the Newton image. | Not used for the roots. Used in tests as an independent Frobenius. |
| `UnramifiedExtension` (`rustmath-rings/src/padics/unramified.rs:146`) | Already Hensel-lifts the Frobenius image (`compute_frobenius_image`, `:307`) and has `conjugates()`. Its arithmetic is the same schoolbook. It lives in `rustmath-rings`, which `rustmath-galois` does not depend on. | No. The design moves the same idea onto `GaloisRing`, as the HANDOFF asks. |
| `GaloisCtx` (`rustmath-galois/src/galois_ctx.rs`) | Frobenius-labelled roots for Stauduhar, with private ring arithmetic and one Newton lift per root. No tracker row. | No. A later unit may port it onto the new module (follow-up). |
| `fp_factor::factor_with_multiplicity` (`rustmath-polynomials/src/fp_factor.rs:854`) | Exact, `p <= MAX_P`. | Diagnostics of a refused prime; `first_inert_prime`. |
| `Integer::nth_root` (`rustmath-integers/src/integer.rs:258`) | Floor root. 0.3 ms for all CC0 roots. | The root bound. |
| `resolvent::subset_sum_resolvent(f, 2)` (`rustmath-polynomials/src/resolvent.rs:251`) | The duad resolvent from power sums, no p-adics. f2 0.73 s, f1 0.87 s (debug f2 4.3 s). Equal to the reference. | Third reference for the duad resolvent in the `env_` tests. |
| `resolvent::pair_sum_resolvent` (`:138`) | Bivariate resultant over `Q`: killed after 120 s on f2. | No. |
| `BinaryGolayCode` (`rustmath-coding/src/golay.rs:33`) | Systematic generator matrix of `G-`. No heptads, no `G+`. | A test links it to the new `G-`. |
| `SteinerSystem::new` / `BlockDesign::new` (`rustmath-combinatorics/src/designs.rs:38`, `:226`) | Verifies the design only for `t <= 2`, `v <= 10`. A test (`machine_word_counts.rs:678`) builds an unverified design with `v = 2^33` and must keep passing, so `new` cannot start verifying everything. | New exact checker beside it (5.2). |
| Python oracle `arith.py` | f2: labelling 0.06 s, J 0.26 s, R_heptad 9.0 s, R_duad 6.2 s; f1: 0.04, 0.27, 10.3, 7.2 s; CC0: labelling 0.44 s, J 3.1 s. | Reference only. |

## 4. Where each piece lives

| Piece | Crate and file | Why there | MAGMA chapter |
|---|---|---|---|
| Lazy multiplication, `defining_polynomial`, `base_ring_value`, crate-internal `from_canonical` and `mul_reduce` | `rustmath-finitefields/src/galois_ring.rs` | The crate that owns `GaloisRing`; same output, measured 2.3-4.4 times faster | 48 |
| `FrobeniusMatrix`, `FrobeniusRoots`, `orbit_polynomial`, trace | `rustmath-finitefields/src/galois_ring_frobenius.rs` (new) | Operations of the Galois ring; WP-A A3/A4 need them on `GR(p^N, 15)` | 47 (`GaloisImage`, `Trace`), 48 |
| Exact t-design test | `rustmath-combinatorics/src/designs.rs` (new functions) | Design theory; WP-P needs it on the published labelling | 147 (`IsDesign`, `IsSteiner`) |
| Cyclic Golay codes `G+`, `G-`, heptads, orbit representatives | `rustmath-coding/src/cyclic_golay.rs` (new) | Codes; WP-P needs `S(4,7,23)` and the even subcode | 152 (`CyclicCode`, H152E5) |
| Bounded integrality and the integrality dichotomy | `rustmath-galois/src/integrality.rs` (new) | Galois theory; reused by WP-A A4 | 38 (`IsInt`, `Bound`) |
| Root bound, labelling, J5, J7, resolvents, Frobenius signs | `rustmath-galois/src/steiner_labelling.rs` (new) | It decides the Galois action on the roots | 38 (`GaloisRoot`, `Stauduhar`), 47, 48 |

Dependency changes: `rustmath-galois` gains `rustmath-coding` and `rustmath-combinatorics` (both
already in its graph through `rustmath-groups` or light), and the dev-dependency `serde_json = "1"`
(already in the lockfile). `rustmath-coding` gains the dev-dependency `rustmath-combinatorics`. No new
external crate. No `unsafe`.

## 5. Public API

### 5.1 `rustmath-finitefields`

`galois_ring.rs` (changes):

```rust
impl GaloisRing {
    /// The defining polynomial D, little-endian, coefficients in [0, p^a), D[d] = 1.
    pub fn defining_polynomial(&self) -> &[Integer];
    /// Element from coefficients already in [0, p^a) (debug_assert), no reduction.
    pub(crate) fn from_canonical(&self, coeffs: Vec<Integer>) -> GaloisRingElement;
}
impl GaloisRingElement {
    /// Some(a_0) in [0, p^a) when a_1 = ... = a_{d-1} = 0 (the element lies in Z/p^a), else None.
    pub fn base_ring_value(&self) -> Option<Integer>;
}
/// Product in (Z/m)[x]/(D) of canonical vectors: convolution without intermediate reduction,
/// then from the top degree: c = r[k] mod m, r[k-d+i] -= c D[i]; finally every entry mod m.
pub(crate) fn mul_reduce(a: &[Integer], b: &[Integer], d_mod_m: &[Integer], m: &Integer) -> Vec<Integer>;
```

`impl Mul for GaloisRingElement` keeps its unbound-sentinel handling and calls `mul_reduce`. The old
schoolbook body moves into the test file as the reference of an equality test.

`galois_ring_frobenius.rs` (new). Header: MAGMA ch. 47 `GaloisImage`, `Trace`; ch. 48; McDonald XVI,
Wan 14, von zur Gathen-Gerhard 9.4 and 15.4; paper Section 5 and arith.py `frobenius_lift`.

```rust
/// Frobenius phi of R = GR(p^a, D) through phi(w), w the generator: the matrix of phi on the power
/// basis (column j = phi(w)^j) and the traces Tr(w^j).
pub struct FrobeniusMatrix { /* ring, image, columns, traces */ }
impl FrobeniusMatrix {
    /// Newton lift of w^p (mod p) to the root of D in R (2.1 steps 1-4), checked:
    /// D(y) = 0, y = w^p mod p, phi^d(w) = w. Err(MathError) on a failed check.
    pub fn new(ring: &GaloisRing) -> Result<Self>;
    pub fn ring(&self) -> &GaloisRing;
    pub fn image_of_generator(&self) -> &GaloisRingElement;          // phi(w)
    pub fn apply(&self, a: &GaloisRingElement) -> GaloisRingElement; // phi(a), one matrix-vector product
    pub fn orbit(&self, a: &GaloisRingElement) -> Vec<GaloisRingElement>; // a, phi(a), ..., phi^{d-1}(a)
    pub fn trace(&self, a: &GaloisRingElement) -> Integer;          // in [0, p^a)
    /// prod_{i<d} (Z - phi^i(a)), ascending, coefficients in [0, p^a).
    pub fn orbit_polynomial(&self, a: &GaloisRingElement) -> Result<Vec<Integer>>;
}

/// prod (Z - v) over a family stable under phi (the caller's claim); ascending, monic. Err(NotInBaseRing
/// { index }) when a coefficient is not in Z/p^a, which disproves the claim.
pub fn orbit_polynomial(values: &[GaloisRingElement]) -> std::result::Result<Vec<Integer>, NotInBaseRing>;

/// The roots alpha_i = phi^i(x) of a monic f, irreducible mod p, in GR(p^N, f) (2.1).
pub struct FrobeniusRoots { /* f over Z, matrix, roots */ }
impl FrobeniusRoots {
    pub fn new(f: &[Integer], p: &Integer, precision: u32) -> std::result::Result<Self, FrobeniusRootsError>;
    /// The same roots at another precision: projection when lower, Newton continued from the current
    /// image when higher (no new x^p, one new irreducibility proof for the new ring).
    pub fn with_precision(&self, precision: u32) -> std::result::Result<Self, FrobeniusRootsError>;
    pub fn polynomial(&self) -> &[Integer];
    pub fn prime(&self) -> &Integer;
    pub fn precision(&self) -> u32;
    pub fn modulus(&self) -> &Integer;               // p^N
    pub fn ring(&self) -> &GaloisRing;
    pub fn frobenius(&self) -> &FrobeniusMatrix;
    pub fn roots(&self) -> &[GaloisRingElement];     // length deg f, alpha_0 = generator
}

pub enum FrobeniusRootsError {
    NotMonic,
    DegreeTooSmall(usize),                   // deg f < 2
    ZeroPrecision,
    Prime(MathError),                        // not prime, or the ECPP proof is unresolved
    NotSquarefreeModP,
    NotInertModP { factor_degrees: Option<Vec<usize>> }, // degrees known when p <= MAX_P
    CheckFailed(&'static str),               // f(y) != 0, y != x^p mod p, or phi^n != id (a bug)
}
```

`FrobeniusRootsError` and `NotInBaseRing` implement `Display` and `std::error::Error`.

### 5.2 `rustmath-combinatorics` (`designs.rs`)

```rust
/// MAGMA IsDesign(D, t) (ch. 147): Ok(Some(lambda)) when every t-subset of {0..v-1} lies in exactly
/// lambda blocks, Ok(None) when the counts differ. Exact: counts every t-subset of every block.
pub fn t_design_index(v: usize, t: usize, blocks: &[Vec<usize>]) -> Result<Option<u64>, DesignCheckError>;
/// MAGMA IsSteiner(D, t): t_design_index == Some(1) and the blocks have one size k >= t.
pub fn is_steiner_system(v: usize, t: usize, blocks: &[Vec<usize>]) -> Result<bool, DesignCheckError>;
pub enum DesignCheckError { PointOutOfRange { block: usize }, RepeatedPoint { block: usize },
                            TooLarge { subsets: u128, limit: u64 } }
```

Cost `b C(k, t)` hash insertions plus `C(v, t)` for the coverage test; refused above `10^8` (an
`Err`, not a guess). Blocks of size `< t` contain no t-subset. `BlockDesign::new` and
`SteinerSystem::new` are unchanged (their documented limit stays; a later unit may route them through
the checker). For S(4,7,23): 8855 insertions.

### 5.3 `rustmath-coding` (`cyclic_golay.rs`, new)

Header: MAGMA ch. 152 `CyclicCode` (H152E5), `QRCode`; MacWilliams-Sloane Ch. 16; paper Section 2;
golay.py.

```rust
pub enum GolaySign { Plus, Minus }
impl GolaySign {
    pub const ALL: [GolaySign; 2];
    pub fn generator_polynomial(self) -> u32;   // 0xAE3 / 0xC75, bit i = coefficient of z^i
    pub fn opposite(self) -> GolaySign;
    pub fn to_i8(self) -> i8;                   // +1 / -1, as golay.py
}
pub const LENGTH: usize = 23;
/// The 4096 words m(z) g(z), deg m < 12, as 23-bit masks (bit i = coordinate i).
pub fn codewords(sign: GolaySign) -> Vec<u32>;
pub fn weight_distribution(sign: GolaySign) -> [u64; 24];
/// The 253 weight-7 words (heptads), sorted, computed once (OnceLock).
pub fn heptads(sign: GolaySign) -> &'static [u32];
/// The least mask of each shift orbit of heptads, sorted (11 masks).
pub fn heptad_orbit_representatives(sign: GolaySign) -> &'static [u32];
pub fn rotate(mask: u32, k: u32) -> u32;      // i -> i + k
pub fn negate(mask: u32) -> u32;              // i -> -i
pub fn support(mask: u32) -> Vec<usize>;
```

### 5.4 `rustmath-galois`

`integrality.rs` (new). Header: MAGMA ch. 38 `IsInt(x, B, S)`, `Bound`; Geissler-Klueners 2000.

```rust
/// The representative of a mod m in (-m/2, m/2].
pub fn symmetric_residue(a: &Integer, m: &Integer) -> Integer;
pub enum IntegralityError { PrecisionTooLow { modulus: Integer, bound: Integer } } // m <= 2B
/// IsInt: Ok(Some(n)) when the symmetric residue n has |n| <= B; then n is the only integer of
/// absolute value <= B in the class (m > 2B). Ok(None) otherwise.
pub fn bounded_integer(residue: &Integer, modulus: &Integer, bound: &Integer)
    -> Result<Option<Integer>, IntegralityError>;
/// Exactly one candidate integral within the bound decides (2.3).
pub enum Dichotomy<L> {
    Decided { label: L, value: Integer, rejected: Vec<(L, Integer)> },
    Unresolved { gap: DichotomyGap, residues: Vec<(L, Integer)> },
}
pub enum DichotomyGap { SeveralWithinBound, NoneWithinBound }
pub fn integrality_dichotomy<L: Clone>(candidates: &[(L, Integer)], modulus: &Integer, bound: &Integer)
    -> Result<Dichotomy<L>, IntegralityError>;
```

WP-A A4 decides by rationality, not integrality. The enum is generic over the label. A follow-up adds
`rationality_dichotomy` (rational reconstruction under a proven height bound, `p^N > 2 H^2`).

`steiner_labelling.rs` (new). Header: paper Section 5 (Proposition 5.1, Theorem 5.2, Remark on
signs), Sections 6 and 8; arith.py; MAGMA ch. 38, 47, 48; Fujiwara 1916; Witt 1938.

```rust
pub use rustmath_coding::cyclic_golay::GolaySign;
pub const DEGREE: usize = 23;

/// The trusted input behind every Decided status (recorded, never checked).
pub enum GaloisHypothesis { M23 }

/// Fujiwara's bound M = best / 128 (2.4).
pub struct RootBound { /* best */ }
impl RootBound {
    pub fn fujiwara(f: &[Integer]) -> Result<RootBound, SteinerError>;   // f monic, deg >= 1
    pub fn best(&self) -> &Integer;
    /// floor(count (scale M)^e) + 1 and floor((1 + scale M)^e) + 1, exact.
    pub fn power_bound(&self, count: u64, scale: u64, e: u32) -> Integer;
    pub fn binomial_bound(&self, scale: u64, e: u32) -> Integer;
    pub fn t5_bound(&self) -> Integer;                 // power_bound(253, 7, 5)
    pub fn j5_bound(&self) -> Integer;                 // power_bound(5313, 1, 5)
    pub fn j7_bound(&self) -> Integer;                 // power_bound(253, 1, 7)
    pub fn heptad_resolvent_bound(&self) -> Integer;   // binomial_bound(7, 253)
    pub fn duad_resolvent_bound(&self) -> Integer;     // binomial_bound(2, 253)
}
/// Least N with p^N > 2 B 10^extra_digits.
pub fn precision_for(p: &Integer, bound: &Integer, extra_digits: u32) -> u32;

pub struct LabellingOptions {
    pub prime: Option<u64>,       // None: first_inert_prime(f, 3, search_limit)
    pub search_limit: u64,        // default 10_000
    pub extra_digits: u32,        // default 30 (the reference margin)
    pub precision: Option<u32>,   // override; refused when p^N <= 2 B5
    pub retries: u32,             // default 0; each retry adds 10 digits after BothWithinBound
}

pub enum LabellingGap { BothWithinBound, NeitherWithinBound }
pub enum LabellingStatus { Decided { sign: GolaySign }, Unresolved(LabellingGap) }

pub struct SteinerLabelling { /* fields below, plus the FrobeniusRoots */ }
impl SteinerLabelling {
    pub fn status(&self) -> &LabellingStatus;
    pub fn hypothesis(&self) -> GaloisHypothesis;
    pub fn sign(&self) -> Option<GolaySign>;
    pub fn t5(&self) -> Option<&Integer>;              // the integer T5(f) when Decided
    pub fn rejected_t5(&self) -> Option<&Integer>;     // residue of the other sign when Decided
    pub fn t5_residue(&self, sign: GolaySign) -> &Integer; // symmetric residues, always available
    pub fn prime(&self) -> &Integer;
    pub fn precision(&self) -> u32;
    pub fn attempts(&self) -> &[u32];                  // precisions tried (retries)
    pub fn root_bound(&self) -> &RootBound;
    pub fn t5_bound(&self) -> &Integer;
    pub fn roots(&self) -> &FrobeniusRoots;
    pub fn golay_invariants(&self) -> Result<GolayInvariants, SteinerError>;   // Err(NotDecided)
    pub fn heptad_resolvent(&self) -> Result<CertifiedResolvent, SteinerError>; // Err(NotDecided)
    pub fn duad_resolvent(&self) -> Result<CertifiedResolvent, SteinerError>;   // works when Unresolved
}

pub fn steiner_labelling(f: &[Integer], opts: &LabellingOptions) -> Result<SteinerLabelling, SteinerError>;
/// Symmetric residues of T5^+ and T5^- for Frobenius-ordered roots (2.6), any precision.
pub fn t5_candidates(roots: &FrobeniusRoots) -> [(GolaySign, Integer); 2];
pub fn duad_resolvent(f: &[Integer], p: &Integer) -> Result<CertifiedResolvent, SteinerError>;
pub enum FrobeniusSign { Decided(GolaySign), Unresolved { matching: Vec<GolaySign> } }
pub fn frobenius_sign(f: &[Integer], q: &Integer, t5: &Integer, precision: u32)
    -> Result<FrobeniusSign, SteinerError>;
/// Bounded search: None says nothing about primes at or above `below`.
pub fn first_inert_prime(f: &[Integer], from: u64, below: u64) -> Option<u64>;
/// g(x) = a^(n-1) f(x/a) with a = lc(f) (roots multiplied by a); returns (g, a).
pub fn monic_associate(f: &[Integer]) -> Result<(Vec<Integer>, Integer), SteinerError>;

pub struct CertifiedInteger { pub value: Integer, pub bound: Integer, pub prime: Integer, pub precision: u32 }
pub struct GolayInvariants { pub j5: CertifiedInteger, pub j7: CertifiedInteger }
pub struct CertifiedResolvent {
    pub coefficients: Vec<Integer>,   // ascending, length 254, leading 1
    pub bound: Integer, pub prime: Integer, pub precision: u32,
}

pub enum SteinerError {
    NotMonic,
    WrongDegree(usize),
    ZeroLeadingCoefficient,                  // monic_associate of a zero polynomial
    NoInertPrime { below: u64 },
    Frobenius(FrobeniusRootsError),
    Integrality(IntegralityError),           // includes PrecisionTooLow
    NotDecided(LabellingGap),
    BoundExceeded { quantity: &'static str, index: Option<usize> }, // a value beyond its proven bound
    NotInBaseRing { quantity: &'static str },                         // orbit coefficient not constant (a bug)
    GolaySystems(String),                    // the run-time check of the two systems failed (a bug)
}
```

Behaviour:

- `steiner_labelling` checks `deg f = 23` and monic, computes the root bound, `B5`, `N`, the
  Frobenius roots, `t5_candidates`, and calls `integrality_dichotomy`. Before the first labelling it
  checks once (cached in a `OnceLock`) that `heptads(+)` and `heptads(-)` are `S(4,7,23)`
  (`is_steiner_system`), shift-invariant and distinct. These are the facts the proof uses (2.3); the
  check costs 1 ms in release and 10 ms in debug.
- `golay_invariants` lifts to `N_J = max(N(B_J5, 0), N(B_J7, 0))` with `with_precision` (projection
  when `N_J` is lower, as for f1 and f2) and evaluates by traces on the 11 representatives of the
  decided system. Each value goes through `bounded_integer`; a value beyond its bound is
  `BoundExceeded`, never returned.
- `heptad_resolvent` and `duad_resolvent` lift to `N(B, 0)` and use orbit polynomials (2.6). Every
  coefficient goes through `bounded_integer`. `duad_resolvent` does not need the labelling: its
  coefficients are symmetric.
- `frobenius_sign` builds `FrobeniusRoots` at `q^N` and compares `t5_candidates` mod `q^N` with `t5`.
- A stability loop is not part of the API. The tests recompute some values at a larger `N` and
  compare (a cross-check only).

## 6. Degenerate cases

| Input | Behaviour | Test |
|---|---|---|
| `f` not monic | `NotMonic` (`FrobeniusRootsError::NotMonic` below); `monic_associate` is the documented way | CC0 raw coefficients |
| `deg f != 23` (labelling) | `WrongDegree` | degree 22 input |
| `deg f < 2` (roots) | `DegreeTooSmall` | `x + 1` |
| `N = 0` | `ZeroPrecision` | unit |
| `p` composite or unproven | `Frobenius(Prime(..))` | `p = 25` |
| `f mod p` not squarefree | `NotSquarefreeModP` | `x^2 + 1` at 2; any `f` at a prime dividing `disc f`, e.g. f-type input at 2 |
| `f mod p` reducible, squarefree | `NotInertModP { factor_degrees }` | `x^3 - x - 1` at 5: degrees `[1, 2]`; CC0 at 11: `[2, 7, 14]` |
| no inert prime below the limit | `NoInertPrime` (no claim beyond it) | a product of two polynomials (never inert) |
| user `N` with `p^N <= 2 B5` | `Integrality(PrecisionTooLow)` | f2 at `N = 8` (env) |
| both candidates within the bound | `Unresolved(BothWithinBound)`; retries add digits | f2 at `N = 9` (env); C23 at every `N` (fast) |
| neither within the bound | `Unresolved(NeitherWithinBound)`: refutes `Gal(f) = M23` | `x^23 - x - 1` at 23 (fast) |
| invariants on an Unresolved labelling | `NotDecided(gap)` | C23 |
| invariant beyond its proven bound | `BoundExceeded` (cannot happen under the hypothesis) | unit test of `bounded_integer` |
| orbit polynomial coefficient not constant | `NotInBaseRing` (a bug detector) | unit test with a non-stable family |
| Frobenius sign: both match `T5` mod `q^N` | `Unresolved { matching: [+, -] }` | C23 (any `t5`) |
| Frobenius sign: neither matches | `Unresolved { matching: [] }` | a wrong `t5` (`T5 + 1`) |
| `p = 2`, `p <= 23` | supported (no division in the method) | C23 at 2 and 3; `x^23 - x - 1` at 23 |
| even modulus in `symmetric_residue` | representative in `(-m/2, m/2]` | unit (`m = 2^k`) |
| huge input (CC0 resolvents, `N` about 12,500) | computed, but costs about 10 minutes (estimate); not in any test | follow-up |

## 7. Tests

### 7.1 Fast tests (normal suite, debug)

`rustmath-finitefields/tests/galois_ring_frobenius.rs`:

- `lazy_mul_equals_schoolbook`: random elements of `GR(p^a, D)` for several `p`, `a`, `d`, including a
  modulus with huge coefficients; the old algorithm is the reference.
- `small_roots_match_pari`: `x^3 + x + 1` at `5^6` (three roots) and `phi(x)` of `x^23 - x - 1` at
  `23^4` (values of `gp/small_lift.gp`, PARI/GP 2.15.4, in a comment).
- `newton_image_equals_teichmuller_frobenius`: `FrobeniusMatrix::new(GR)` image equals
  `generator().frobenius()` for `GaloisRing::new(5, 4, 3)`, `GaloisRing::new(2, 8, 4)` and
  `x^23 - x - 1` at `23^3`; `apply(a) == a.frobenius()` for random `a`.
- `frobenius_is_a_ring_automorphism_of_order_d`, `trace_is_the_orbit_sum`,
  `orbit_polynomial_vanishes_at_a`, `orbit_polynomial_rejects_an_unstable_family`.
- `refusals`: the rows of Section 6 that concern roots.

`rustmath-combinatorics` (unit tests in `designs.rs` or `tests/designs_exact.rs`): the Fano plane
2-(7,3,1) gives `Some(1)`; the Fano plane minus a line gives `None`; the 3-(8,4,1) design (the 7 Fano
lines with an eighth point added, and their 7 complements) gives `Some(1)` for `t = 3`; the 6 faces of
the cube on its 8 vertices give `Some(3)` for `t = 1` (a 1-(8,4,3) design, as in H147E9); out-of-range and repeated points give `Err`; `TooLarge` above the
limit.

`rustmath-coding/tests/cyclic_golay.rs` (golay.py `verify_basic_facts` and the paper, Section 2):
weight distribution; `(z+1) g+ g- = z^23 - 1` and `g- = reverse(g+)` over `F_2`; both systems
`S(4,7,23)` (via `is_steiner_system`); shift and `i -> 2i` invariance; `negate` exchanges them; they
differ; 77, 21, 5; intersections `{1: 14168, 3: 17710}`; 11 orbit representatives equal to golay.py:
`+`: `0xae3, 0x34a9, 0x483d, 0xb115, 0xcd81, 0x14345, 0x18427, 0x21c13, 0x22259, 0x32921, 0x51231`;
`-`: `0xc75, 0x254b, 0x5e09, 0x81b3, 0xa88d, 0x14585, 0x1c843, 0x21253, 0x26911, 0x320e1, 0x46245`;
the 5 heptads through `{0,1,2}`: `+`: `{0,1,2,3,9,12,21}, {0,1,2,4,6,18,19}, {0,1,2,5,10,15,16},
{0,1,2,7,13,14,17}, {0,1,2,8,11,20,22}`; `-`: `{0,1,2,3,5,14,17}, {0,1,2,4,13,16,22},
{0,1,2,6,7,19,21}, {0,1,2,8,11,12,18}, {0,1,2,9,10,15,20}`; the rows of `BinaryGolayCode::new()` lie
in `G-`.

`rustmath-galois` unit tests (`integrality.rs`): symmetric residues (odd and even moduli),
`bounded_integer` at `m = 2B` (refused) and `m = 2B + 1`, the three dichotomy outcomes.

`rustmath-galois/tests/steiner_labelling.rs` (fast part, no environment variable):

- `cc0_certificate_facts`: the vendored `lelay_specialization_t2_Z.json` has content 1, degree 23,
  the leading coefficient
  `239024099105221865598815800284190403153851130391180859371046699194977683`; `f mod 29` is
  irreducible; `f mod 11` has factor degrees `[2, 7, 14]` (`fp_factor`, matching the file's
  `modular_certificates`).
- `cc0_labelling_at_29`: `monic_associate`, the root bound (`best` above), `B5` (364 digits),
  `N = 270`; Decided, sign `-`, `T5`, and the rejected residue (395 digits) equal to
  `cc0_t2_steiner_reference.json`; the same labelling at the minimal rigorous `N = 250` is Decided with
  the same `T5`, and its rejected residue is the reduction of the one at 270.
- `cc0_labelling_at_67`: `N = 216`, Decided with sign `+`, the same `T5` as at 29, rejected residue of
  394 digits (second prime; arith.py `oracle_cc0_67.py` and PARI/GP `run_cc0_67.gp` agree).
- `cc0_golay_invariants`: `J5`, `J7` (N = 344) equal to the reference file; `T5 = 120 J5 + S(g)`
  (2.7).
- `cc0_frobenius_signs`: `frobenius_sign` at 73 and 109 from the known `T5` gives `-`, `-`, as both
  references and as the f1 rows of the vendored `frobenius_signs.json`.
- `c23_is_unresolved_at_every_precision`: `polsubcyclo(47, 23)` (coefficients in the test, from
  PARI/GP) at `p = 2` and `p = 3`: `Unresolved(BothWithinBound)`, residues `{-98317, -30637}` (PARI/GP
  `c23.gp`, `c23_p3.gp` and arith.py agree at 2, 3 and 5; the signs swap between 2 and 3), `golay_invariants` gives
  `NotDecided`.
- `s23_refutes_the_hypothesis`: `x^23 - x - 1` at 23: `B5 = 136069473`,
  `Unresolved(NeitherWithinBound)`, residues equal to the reductions of the PARI/GP and arith.py values
  at `N = 30`: `T5^+ = -29593503946453640472557674395096324685942`,
  `T5^- = 26819164821679513709525347975140930751221`.
- `trace_route_equals_direct_sums`: on the CC0 roots at a small precision (`N = 40`), the direct sum
  over 253 heptads (a constant) equals `t5_candidates`.
- `refusals`: the labelling rows of Section 6 that need no external data.

Measured debug times of the prototype equivalents (improved arithmetic, one thread): CC0 labelling at
29 2.4 s, J5 and J7 7.2 s, two Frobenius signs about 0.8 s, labelling at 67 about 2 s (estimate), C23 and
`x^23 - x - 1` under 1 s, Golay and design checks 10 ms. Total about 13 s plus compile time. With
the existing `Mul` and Horner composition the same set takes about 45 s (measured 42.6 s for the CC0
labelling and invariants alone).

### 7.2 Opt-in tests needing `$M23ISGALOIS` (`#[ignore]`, release)

They read `polynomial_f1.gp` and `polynomial_f2.gp` from the directory named by `M23ISGALOIS`. The
reader parses `<name> += (<int>)*x^<i>;` and checks the file's own fingerprint line
`<name>_expect = [...]` (degree, number of non-zero terms, total and maximal digit counts, content,
`f(3) mod 2^61 - 1`) before use. Without the variable each test panics with
"set M23ISGALOIS to a checkout of shaowuz/m23isgalois" (the WP-F helper); it never passes vacuously.
f1 and f2 never appear in RustMath sources, comments or data.

- `env_six_integers_and_signs`: at 29, both Decided with sign `-`; `T5`, `J5`, `J7` equal to the
  vendored data; rejected residues equal to the 44- and 45-digit values above (oracle and PARI/GP);
  `N = 30` and `N = 31`.
- `env_f2_minimal_precision`: `N = 8` refused (`PrecisionTooLow`), `N = 9`
  `Unresolved(BothWithinBound)` with `T5^+ = -3209424281433`, `N = 10` Decided with
  `T5^+ = -206309467943599`; f1 at its minimal `N = 10` Decided with `T5^+ = -114831914638610` (these
  are the symmetric reductions of the reference residues at 30 and 31).
- `env_heptad_resolvents`: f1, f2 equal to `R253_heptad` of the vendored data (N = 395, 360).
- `env_duad_resolvents`: f1, f2 equal to `R253_duad` and to `subset_sum_resolvent(f, 2)` (three
  references: the owner's data, the p-adic route, the power-sum route).
- `env_second_prime`: f1 at 73 and f2 at 223: Decided `-`, same six integers, same four resolvents
  (`verified_at_second_prime`).
- `env_frobenius_signs`: all 40 primes of each polynomial in `frobenius_signs.json` (`N = 1`).
- `env_t5_j5_identity`: 2.7 for f1 and f2.
- `env_stability_cross_check`: `J5`, `J7` and the heptad resolvent of f2 recomputed at `N + 7`
  agree (a cross-check, not a certificate).

Measured release times (prototype, improved arithmetic, 2 threads): f2 all of 29 in 3.4 s, f1 in
4.3 s; f1 at 73 3.7 s, f2 at 223 3.3 s; `subset_sum_resolvent` 0.7 s and 0.9 s; signs (estimate)
about 4 s. Total about 20 s. With the existing arithmetic about 60 s.

### 7.3 Opt-in release tests without external data (`#[ignore]`, prefix `full_`)

- `full_cc0_frobenius_signs`: the twelve inert primes below 900 (29 to 887), equal to both references
  and to the f1 rows (4.8 s in debug, under 1 s in release).
- `full_cc0_invariants_at_67`: `J5`, `J7` at 67 equal to those at 29 (both references).

### 7.4 Reference values and their sources

| Value | Source 1 | Source 2 |
|---|---|---|
| six integers, signs at 29, resolvents (f1, f2) | `golay_resolvents.json` (owner, FLINT) | arith.py rerun here (python-flint 0.9.0) and PARI/GP `run_f1.gp`, `run_f2.gp` for `T5`, `J5`, `J7` |
| rejected residues (f1, f2) | arith.py | PARI/GP |
| duad resolvents | `golay_resolvents.json` | PARI/GP resultant identity `run_duad.gp`; `subset_sum_resolvent` |
| second primes | `golay_resolvents.json` field | PARI/GP `run_second.gp` (`T5`, `J5`, `J7` at 73 and 223) |
| 80 Frobenius signs | `frobenius_signs.json` | PARI/GP `run_signs.gp` (all agree; f2 has 5 more inert primes below 4000 that the file does not list) |
| CC0 `T5`, rejected residue, `J5`, `J7`, signs | arith.py `oracle_cc0.py`, `oracle_cc0_signs.py` | PARI/GP `run_cc0.gp`, `run_cc0_signs.gp` |
| C23 and `x^23 - x - 1` residues | arith.py | PARI/GP `c23.gp`, `c23_p3.gp` |
| CC0 at the second prime 67 | arith.py `oracle_cc0_67.py` | PARI/GP `run_cc0_67.gp` |
| small lifts | PARI/GP `small_lift.gp` | `GaloisRingElement::frobenius` (Teichmueller) |
| heptad constants | golay.py | the counts of the paper, Section 2 |

## 8. Data files and licence handling

New directory `rustmath-galois/tests/data/m23/` with a `README.md` modelled on
`rustmath-curves/tests/data/m23/README.md`:

| File | Source | Licence | sha256 |
|---|---|---|---|
| `golay_resolvents_no_f.json` | derived from `johnjanik/M23` `data/golay_resolvents.json` at `d05475b` (original sha256 `dcf1e38c4f13fc0b1c5503a1e2a074c6cdce2f74ea4dd292b0dc0855ee870d13`) by removing the `"f"` field of each entry | the owner's computed data, vendored as the brief allows; f1 and f2 are removed because they come from the unlicensed HJLPPZ repository | `87123dfe2fe86cb6bf5b40e2a39c2d06a680c6a2df2e8fd8bce1adee4f87a381` (133,201 bytes) |
| `frobenius_signs.json` | `johnjanik/M23` `data/frobenius_signs.json` at `d05475b`, unchanged | the owner's data | `30ec467012f53e2b8e7e6a4d05e321a8adabe747aea1768488b9cda604617a4f` |
| `lelay_specialization_t2_Z.json` | Le Lay, `data/specialization_t2_Z.json` at `3f6ab4b`, unchanged (it equals `G(2, W)` of the WP-F vendored model) | CC0-1.0, attributed to Knabble, Inc. (dba Reflexivity) | `f775aa4c004541f25d121fac999fca9c62e493100f99982cc8253fbfa4445862` (as line 84 of his `CHECKSUMS.sha256`) |
| `cc0_t2_steiner_reference.json` | new: values for the monic associate of the previous file (`T5`, rejected residue at N = 270, `J5`, `J7`, signs at the twelve primes), with the commands and versions that produced them | computed here from CC0 data | recorded at creation |

The derivation command, recorded in the README (it reproduces the hash with Python 3.13):

```bash
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); [v.pop("f") for v in d.values()]; json.dump(d, open(sys.argv[2],"w"), indent=0)' \
  data/golay_resolvents.json golay_resolvents_no_f.json
```

The README also states: f1 and f2 (and HJLPPZ's F) are read only from `$M23ISGALOIS`; the paper and
code of Le Lay (CC BY-NC-SA) are not used; the owner's code is ported as algorithms with a citation of
the paper. The PARI/GP scripts of `design/SL/gp/` go to `docs/workstreams/m23_golay/sl/gp/` in the
docs step (like `wpf/gp/`). They read f1 and f2 at run time; `duad_ref.gp` (owner data) and
`cc0_poly.gp` (CC0) may be kept or regenerated.

## 9. Cost (measured)

Machine: 4 cores shared with another job, `-j 2`, one test thread. Times are wall clock.

Precision needed (p = 29 unless stated). `N_rig` is the least `N` with `p^N > 2B`; "stable" is the
least `N` with `p^N > 2|value|` (for information only).

| Polynomial | Quantity | B digits | N_rig | N used | Stable N | Value digits |
|---|---|---|---|---|---|---|
| f2 | T5 | 13 | 9 (Unresolved) | 30 | 5 | 7 |
| f2 | J5, J7 | 10, 11 | 7, 8 | 8 | 4, 4 | 5, 5 |
| f2 | R heptad, R duad | 526, 390 | 360, 267 | same | 145, 110 | 212, 161 |
| f1 | T5 | 14 | 10 (Decided) | 31 | 6 | 8 |
| f1 | J5, J7 | 11, 13 | 8, 9 | 9 | 5, 5 | 6, 7 |
| f1 | R heptad, R duad | 578, 441 | 395, 302 | same | 178, 132 | 260, 192 |
| f1 at 73 | T5 / J / R heptad / R duad | | | 24 / 7 / 310 / 237 | | |
| f2 at 223 | T5 / J / R heptad / R duad | | | 19 / 5 / 224 / 167 | | |
| CC0 | T5 | 364 | 250 (Decided) | 270 | 244 | 356 |
| CC0 | J5, J7 | 362, 503 | 248, 344 | 344 | 242, 336 | 353, 491 |
| CC0 | R heptad, R duad | 18,294, 18,156 | 12,510, 12,416 | not run | | |

Ring product in `GR(29^N, f)` (release):

| N | existing `Mul`, f2 modulus | lazy, f2 | existing, CC0 modulus | lazy, CC0 |
|---|---|---|---|---|
| 31 | 0.35 ms | 0.08 ms | 0.38 ms | 0.09 ms |
| 270 | 0.98 ms | 0.31 ms | 2.15 ms | 0.59 ms |
| 400 | 1.69 ms | 0.51 ms | 2.80 ms | 1.22 ms |
| 1000 | 4.84 ms | 1.76 ms | 13.5 ms | 3.62 ms |

Whole computations (release):

| Step | Existing arithmetic | Improved (lazy, matrix, orbits) |
|---|---|---|
| f2: lift at 30 / labelling | 0.40 s (0.55 s direct T5, 0.02 s trace) | 0.06 s / 0.01 s |
| f2: R heptad (N = 360) | 1.45 s lift + 5.66 s | 0.17 s + 1.87 s |
| f2: R duad (N = 267) | 1.09 s + 3.45 s | 0.13 s + 1.08 s |
| f1: R heptad (N = 395) | 1.41 s + 5.76 s | 0.24 s + 2.40 s |
| f1: R duad (N = 302) | 1.18 s + 3.93 s | 0.15 s + 1.30 s |
| f1 everything at 29 / at 73 | about 13 s | 4.3 s / 3.7 s |
| f2 everything at 29 / at 223 | about 12 s | 3.4 s / 3.3 s |
| CC0 labelling (N = 270), J (N = 344) | 1.3 s + 0.1 s, 1.7 s + 1.0 s | not timed in release |

The "existing" column already uses orbit compression for the resolvents and traces for the sums. With
the reference's direct product of 253 linear factors over `R` the resolvents would take about 10 times
longer.

Debug (prototype equivalents): CC0 labelling and invariants 42.6 s existing, 9.6 s improved; f2
everything at 29 48.5 s improved (resolvents 27 s and 17 s), hence the `env_` tests are release-only.

## 10. Plan with acceptance commands

All commands in `/home/user/RustMath-steiner`, with `-j 2`. Never set `CARGO_TARGET_DIR`.

Step 0. Baseline at `d91f880`: `cargo test -j 2 -p rustmath-finitefields`,
`cargo test -j 2 -p rustmath-combinatorics`, `cargo test -j 2 -p rustmath-coding`,
`cargo test -j 2 -p rustmath-galois`. Record pass counts and times. (A debug test build of
`rustmath-galois` took 13 s here; its dependencies are built.)

Step 1 (`rustmath-finitefields`, commit 1). `mul_reduce` and the new `Mul` body; `defining_polynomial`,
`from_canonical`, `base_ring_value`; `galois_ring_frobenius.rs` with `FrobeniusMatrix`,
`orbit_polynomial`, `FrobeniusRoots`, `FrobeniusRootsError`; re-exports in `lib.rs`;
`tests/galois_ring_frobenius.rs`. Acceptance: `cargo test -j 2 -p rustmath-finitefields` (whole crate
green, the existing Galois ring tests unchanged), `cargo clippy -j 2 -p rustmath-finitefields`.

Step 2 (`rustmath-combinatorics`, commit 2). `t_design_index`, `is_steiner_system`,
`DesignCheckError` and tests. Acceptance: `cargo test -j 2 -p rustmath-combinatorics`,
`cargo clippy -j 2 -p rustmath-combinatorics`.

Step 3 (`rustmath-coding`, commit 3). `cyclic_golay.rs`, export in `lib.rs`, dev-dependency on
`rustmath-combinatorics`, `tests/cyclic_golay.rs`. Acceptance: `cargo test -j 2 -p rustmath-coding`,
`cargo clippy -j 2 -p rustmath-coding`.

Step 4 (`rustmath-galois`, commit 4). Dependencies, `integrality.rs`, `steiner_labelling.rs`, the data
directory with its README (hashes checked by `sha256sum`), `tests/steiner_labelling.rs`. Acceptance:

```bash
cargo test -j 2 -p rustmath-galois                                    # fast set, debug, about 13 s of tests
cargo test -j 2 --release -p rustmath-galois --test steiner_labelling -- --ignored full_
M23ISGALOIS=/home/user/external/m23isgalois \
  cargo test -j 2 --release -p rustmath-galois --test steiner_labelling -- --ignored env_
cargo test -j 2 --release -p rustmath-galois --test steiner_labelling -- --ignored env_six   # must FAIL: no M23ISGALOIS
cargo clippy -j 2 -p rustmath-galois
cargo fmt --check -p rustmath-finitefields -p rustmath-combinatorics -p rustmath-coding -p rustmath-galois
```

Paste the outputs and the measured times into the implementation report. Delete `target/release` when
done; if free space falls under 5 GB, delete stale test executables in `target/debug/deps`.

Acceptance of the unit: the six integers, the signs, the rejected residues (more than 40 digits) and all
four resolvents of f1 and f2 reproduced in the `env_` tests; the second-prime field reproduced; the 80
signs; the CC0 labelling, invariants and signs in the fast suite with two references each; every row
of Section 6 behaves as stated.

The tracker rows and the HANDOFF status are a later step (not this unit). Rows to add then:
`galois_ring_frobenius.rs` (new), the `Mul` row note, `designs.rs` checker, `cyclic_golay.rs`,
`integrality.rs`, `steiner_labelling.rs`; and in HANDOFF Section 2 the `UnramifiedExtension` line
can point to the new module.

## 11. Follow-ups (not in this unit)

- **SL-1. `tschirnhaus`** (`data/tschirnhaus_f2.json`): `P(alpha_i) = D_i(alpha)` for
  `D = grad J7`, coefficients with denominator `disc(f2)` (125 digits) and numerators up to 132 digits.
  Cheap with the new pieces (one element `d0`, `p^N > 2 disc B`), but it needs a proven bound for the
  power-basis coordinates (the reference uses a generous one and verifies afterwards) and the
  vendored file contains `f`. Out of scope; decide the bound first.
- **SL-2. CC0 resolvents.** `N` about 12,500 at 29; estimated 10 minutes even with lazy products.
  Better: a small model of the CC0 field (PARI `polredbest` with the transformation, certified in
  Rust by checking `g'(P(x)) = 0 mod g`). Then the CC0 resolvents become fast tests.
- **SL-3. `GaloisRingElement::frobenius` through the matrix.** Replace the Teichmueller-digit
  Frobenius (5.5 s at N = 30) by a cached `FrobeniusMatrix` in the ring data. Same values (checked by
  the new test), large speed-up for every caller.
- **SL-4. Port `GaloisCtx` onto `FrobeniusRoots`** (one arithmetic, fewer private kernels).
- **SL-5. Tighter rigorous bounds.** Individual root moduli would roughly halve the resolvent
  precision (578 bound digits against 260 actual for f1), but need certified enclosures (absent,
  HANDOFF Section 2). A Maclaurin bound from `sum |s_H|^2 = 56 sum |alpha_i|^2 + 21 |sum alpha_i|^2`
  saves about 120 of the 578 digits without enclosures. Not needed at current costs.
- **SL-6. `rationality_dichotomy`** for WP-A A4 (2.6, 5.4).
- **SL-7. `BlockDesign::new` through the exact checker** within a size limit, with the
  `machine_word_counts.rs` overflow test adjusted by its owner.
- **SL-8. Other primes** (the full sign tables, other labelling primes): runs of the shipped API;
  no new code.

## 12. Decisions for the reviewer

- **D1 (lazy multiplication in place).** Rewrite `GaloisRingElement::mul` rather than add a second
  kernel. Same results, 2.3-4.4 times faster, all users benefit. Alternative: keep `Mul` and put
  `mul_reduce` only behind the new module (then the fast suite costs about 45 s in debug, or the CC0
  invariants move to `full_`).
- **D2 (four crates, four commits, one branch).** Each piece goes to its owner crate, as WP-F's D4 did
  for two crates. Alternative: put the S(4,7,23) check inside `cyclic_golay.rs` and skip
  `rustmath-combinatorics` (three crates), at the cost of a design checker that WP-P cannot reuse on
  other labellings.
- **D3 (orbit compression).** Use traces and orbit polynomials instead of the reference's direct sums
  and products. Exact identities, 11 times fewer products; the direct sums remain as test
  cross-checks.
- **D4 (statuses).** `NeitherWithinBound` is an `Unresolved` status whose documentation says it
  refutes the hypothesis. Alternative: a third status `HypothesisRefuted`.
- **D5 (margin).** Default `extra_digits = 30` as the reference; `retries = 0`. A Decided status at any
  `N` with `p^N > 2B` is a proof either way.

## Appendix: prototype, oracles and versions

- Rust prototype `design/SL/sl_proto.rs` (temporary integration test of `rustmath-finitefields`, run
  with `cargo test -j 2 [--release] -p rustmath-finitefields --test sl_proto -- --ignored --nocapture
  --test-threads 1 <name>` and the variables `SL_POLY`, `SL_P`, `SL_SIGN`, `SL_NAME`, `SL_RES`,
  `SL_DIRECT`, `SL_MODE`). Functions: `proto_golay`, `proto_mul_cost`, `proto_lazy_mul`,
  `proto_labelling`, `proto_invariants`, `proto_resolvents` (existing arithmetic),
  `proto_fast_variant` (lazy products, Frobenius matrix, orbits), `proto_fast_cc0`,
  `proto_small_lift`, `proto_symmetric_duad`, `proto_teichmuller_frobenius_cost`, `proto_signs_cost`.
- Oracle: Python 3.13.18, `python3 -m venv venv; venv/bin/pip install python-flint==0.9.0 mpmath`;
  scripts `oracle/oracle_f12.py`, `oracle_cc0.py`, `oracle_cc0_signs.py`, `bounds.py`,
  `t5_j5_identity.py`, `export_ref.py` (they import the owner's `m23golay` from
  `/home/user/johnjanik/m23` at `d05475b` and read f1, f2 from `m23isgalois` at `f3866d1`).
  `oracle_f12.py` reproduces `golay_resolvents.json` completely (33 s).
- PARI/GP 2.15.4: `gp/steiner.gp` (heptads from the generator polynomials, roots by `polrootspadic`
  over `Q_p[t]/(f)` ordered by `t^(p^i)` mod `p`, exact arithmetic in `(Z/p^N)[t]/(f)`), drivers
  `run_f1.gp`, `run_f2.gp`, `run_cc0.gp`, `run_signs.gp`, `run_second.gp`, `run_duad.gp`,
  `run_cc0_signs.gp`, `c23.gp`, `small_lift.gp`. Run with `gp -q -f <file> < /dev/null`.
- GAP was not needed: the group facts used (Lemma 2.1, the maximal subgroups of M23) are cited from
  the paper and the ATLAS through Le Lay's certificate.
