# Design F1: rustmath-finitefields (with the container hook in rustmath-core and rustmath-polynomials)

Unit F1 of the HANDOFF follow-ups (`docs/HANDOFF.md`, "Open follow-ups", lines at `9c24389`, same code as
HEAD `c79a582`). Implemented after I3, I1, I2 and P1. Line numbers below are HEAD line numbers.

## Revision 1 (after review 1)

Experiments of this revision are in `$X/rev1/` (`$X` is defined below): `rev1_ref.gp`, `rev1_ref2.gp` and
their `.log` files (PARI/GP 2.15.4); `rev1_cont.rs` and `rev1_cont.log` (HEAD behaviour of
`MultivariatePolynomial`, `LaurentPolynomial` and `Gfpn` equality, plus a prototype of the binding rule of
§7.2 on a local sparse container); `rc.sh` (same build helper as `$X/rc.sh`).

| Review issue | What changed |
|---|---|
| B1 (item 2): `t^2+t+1` is reducible over F_7 | Confirmed: gp `factormod(t^2+t+1, 7)` = `(t+3)(t+5)`. The normalization row of §2.1 and the test `gfpn_normalizes_the_modulus` (§2.5) now use p = 5 with `[6,1,1]` and `[1,1,1]` (gp `polisirreducible(Mod(1,5)*(t^2+t+1))` = 1; `[6,1,1]` mod 5 = `[1,1,1]`). At HEAD the two `one()`s are unequal (`rev1_cont.log`). The other uses of `t^2+t+1` are over 2^80+13, which is 2 mod 3, so the polynomial is irreducible there (gp: 1). |
| B2 (item 4): the capped-hook test depends on test order through the cache | Confirmed. The P43 hook test moves to its own file `tests/unit_group_orders_capped.rs`, so it runs in its own process. That file holds one test. Phase 1 asserts the capped answers before any exact call. Phase 2 runs the exact inherent orders and then asserts that the same hooks are now decided, which is the documented cache effect. The hook docs and the core doc of `MultiplicativeOrder::Undecided` say that a capped `Undecided` can become decided later in the same process, and that both answers are honest. The cache is now keyed by the integer that was factored, so the four types share the factorization of P43 - 1 (§4.3, §4.4, §4.6). |
| B3 (item 7): `MultivariatePolynomial` and `LaurentPolynomial` were left out | Agreed. Both are designed and tested now (§7.2, §7.5). Rule: an `add_term` that brings a bound coefficient into an unbound map binds the map once; an unbound coefficient that enters a bound map is bound on entry; terms that become zero are dropped; a manual coercing `PartialEq` replaces the derived one. The prototype gives the reviewer's value: over GR(8,2), `(x_0+1)^8 - x_0^8 + w` has the 4 terms `(1+w) + 4x^2 + 6x^4 + 4x^6` in either order of addition, and equals the bound construction (HEAD: 8 terms, degree 7). `LaurentPolynomial` implements the `Ring` hooks too. A `LaurentPolynomial` zero is an empty map with no parent, so the univariate rules now pick a bound model by a scan instead of reading `coeffs[0]`. `Matrix`, `Vector`, `SparseMatrix`, the power and Laurent series types, and R1's p-adic sentinels are an explicit hand-off (§7.6: each type, its interim behaviour, the hook call that fixes it). rustmath-polynomials gets a test sentinel type, so plan step 2 is tested inside its own crate. |
| m1: `should_panic` strings miss "Invalid argument: " | `Gfpn::new` panics with the error's `Display` (`panic!("{e}")`). The `try_new` messages keep their "Gfpn: " context, so the panic text is "Invalid argument: Gfpn: the defining polynomial is reducible over GF(3)", and the tests expect exactly that. I checked every other expected panic text in the design against how it is formed: `String` errors appear verbatim, `MathError`s through their `Display` (§2.3, §2.5, §3.4). |
| m2: no ±1 shortcut on the Galois-ring path | The shortcuts now live in the shared `group_order::unit_order` (1 has order 1, -1 ≠ 1 has order 2, no factoring). Every order of the crate uses it, including the residue step of `GaloisRingElement`. New Galois-ring rows in the capped test: -1 gives `Finite(2)`; `1 + P43` and `2·P43 - 1` give `FiniteExceedsU64` (orders P43 and 2·P43, gp `znorder`) without any factoring (§4.3, §4.4, §4.6). |
| m3: plan step 3.8 turns the pinned test red | The item 7 step (now 3.9, after a new first step 3.1 for `is_irreducible_fp`) replaces `polynomials_from_static_constructors_keep_unbound_coefficients` in the same step, together with the impls that turn it red (§9.1). |
| m4: `is_irreducible_fp` answers over a non-field | Fixed, not only listed. It now checks the characteristic and panics with "poly_factor::is_irreducible_fp: Invalid argument: characteristic 15 is not prime". The HEAD body becomes `pub(crate) fn is_irreducible_fp_unchecked`, used by `Gfpn::try_new`, `FiniteField::with_modulus`, `GaloisRing::with_modulus` and the generators, which have proven p prime already (§3.2, §3.3, §3.4). |
| m5: "capped" is not a time cap, and `PrimeField` proves m for nothing | (a) `Effort::Capped` runs I2's driver with `proof: false`. It then proves each remaining probable prime itself, but only up to 256 bits (I1 `prove_primality`; I1 measured 0.36 s mean and 0.52 s max in debug at 256 bits). A larger probable prime gives `Undecided`. So one hook call costs at most trial division to 10^4, 8191 rho steps per composite, a sieve on at most 30 digits, and one ECPP proof of at most 256 bits per prime factor. (b) `unit_order_mod` first tests a^(m-1) = 1 (mod m), one exponentiation. When that holds, h = m - 1 is a multiple of the order, so m is neither factored nor proven prime. Otherwise it uses λ(m). This serves `IntegerMod` and `PrimeField`, both the hooks and the inherent methods. New rows: 2 mod 341 has order 10 (a Fermat liar, h = 340); 3 mod 341 has order 30 (the λ path); 2 mod 561 has order 40 (gp `znorder`). (c) The core doc of `Undecided` now covers "not decided within the type's cost cap" (§0.1, §4.3, §4.4, §4.6, §4.7). |
| m6: the item 6 fallback needs I1's code | Replaced by a fallback on HEAD's API (§6). |
| m7: R1's p-adic sentinels | `PadicInteger`, `PadicRational` and `PadicExtensionElement` are named in the §7.6 hand-off, with the hooks they must implement. |
| m8: docs | `# Panics` notes for `is_primitive` and `factored_multiplicative_order`. `factored_multiplicative_order` of 1 is `Some(vec![])`; this is stated and tested (MAGMA `FactoredOrder(1)` is the empty sequence; gp `factor(1)` = `matrix(0,2)`). The core docs of `is_unbound_constant` and `bind_to_parent_of` describe containers: "unbound" means no coefficient is bound (the image of an element of Z[x]), and a zero container counts as unbound (§4.4, §4.6, §7.2). |

I disagree with none of the issues.

Experiments: `$X` = `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/F1-finitefields/`.
`$X/rc.sh prog.rs [opt] args` builds a standalone program against the HEAD rlibs copied to `$X/lib` (exact
paths in `$X/rlibs.txt`, from `cargo build -p rustmath-finitefields --message-format=json`). All timings
are debug builds. Logs sit next to their programs (`t*.log`, `p*.log`); gp scripts are `*.gp` and their
output `*_ref.log` (PARI/GP 2.15.4, run as `gp -q < file.gp`).

| File | What it is |
|---|---|
| `t1_ddf.rs`, `t1_ddf.log`, `t1_ref.gp` | HEAD: both DDFs on non-squarefree input and 0; lava squarefree/DDF with composite characteristic |
| `t2_gfpn.rs`, `t2_gfpn.log` | HEAD: `Gfpn::new` with reducible, non-monic, empty, degree-0, unreduced moduli and composite p; costs of `new`, `mul`, `pow`, Rabin, memo lookup |
| `t3_order.rs`, `t3_order*.log`, `t3_ref.gp` | HEAD: cost of the order loops and of trial-division factoring of q - 1 |
| `t4_rand.rs`, `t4_rand.log` | HEAD: coefficient ranges of both `random_irreducible`; cost of the default `from_int` on `Gfpn` |
| `t5_sqf.rs`, `t5_sqf.log` | HEAD: differential check of both squarefree factorizations (0 bad in 999 + 766 random cases) |
| `t8_badp.rs`, `t8_badp.log` | HEAD: the irreducible-polynomial generators with p = 15, 4, 1, 0 |
| `t9_ffnew.rs` | HEAD: `FiniteField::new(2^80+13, 3)` does not return (finding, section 8) |
| `p1_sqrt.rs`, `p1_sqrt.log`, `t6_ref.gp` | Prototype of IsSquare/Sqrt in GF(p^n); exhaustive check on 21 fields; matches PARI on every vector |
| `p2_order.rs`, `p2_order.log`, `p2_ref.gp`, `t10_ref.gp` | Prototype of order-from-factorization (Cohen 1.4.3, Carmichael lambda, cyclotomic split); matches the HEAD loop on 48517 units and PARI `znorder`/`fforder` |
| `p3_randirr.rs`, `p3_randirr.log`, `p3_ref.gp` | Prototype of the uniform `random_irreducible` (both stacks); pinned outputs, PARI `polisirreducible` = 1 |
| `t7_ref.gp`, `t11_ref.gp` | References for the capped-hook test prime and the container tests |
| `rev1/rev1_ref.gp`, `rev1/rev1_ref2.gp` (+ `.log`) | Revision 1 references: irreducibility of `t^2+t+1` mod 5 and 7, P43 orders in Z/P43 and Z/P43^2, the Fermat-path rows (341, 561), `factor(1)`, the sparse-container values |
| `rev1/rev1_cont.rs`, `rev1/rev1_cont.log` | HEAD: `MultivariatePolynomial` and `LaurentPolynomial` over GR(8,2) and GF(3) keep unbound terms; HEAD `Gfpn` equality with an unreduced modulus. Prototype of the sparse binding rule and the coercing `==` |

## 0. Summary

| # | Item | At HEAD | Fix |
|---|---|---|---|
| 1 | DDFs `poly_factor.rs:613`, `cantor_zassenhaus.rs:448` do not check squarefree; lava `squarefree_factorization` (`:363`) checks no characteristic | Confirmed (`t1_ddf.log`): `(x+1)^2` mod 5 gives "degree 1" twice; `(x+1)^2(x^2+2)(x^2+3)` mod 5 gives the label 2 on `x^5+x^4+x+1 = (x+1)(x^2+2)(x^2+3)`; lava DDF and squarefree accept modulus 15 (`x^2+1` "irreducible of degree 2") | Eager checks with panics naming the violation (same wording as P1's `fp_factor` fix); private unchecked bodies for the internal callers, so no check runs twice |
| 2 | `Gfpn` with a reducible defining polynomial is not detected (`ff_poly.rs:590`) | Confirmed, plus more (`t2_gfpn.log`): composite p accepted; a non-monic modulus gives wrong products silently (x*x = 4 in F5[x]/(2x^2+1), PARI: 2); empty or constant modulus panic with an arithmetic overflow or an index error | `Gfpn::try_new` validates (degree >= 1, p proven prime, monic mod p, Rabin) and normalizes the modulus; `Gfpn::new` panics with the reason; a memo of validated `(p, modulus)` keeps the per-call cost at a hash lookup. `Gfpn::new` is the only constructor path (all other `Gfpn` values are derived from an existing one) |
| 3 | `random_irreducible` non-uniform for p > 2^63 (`irreducible.rs:168`, `poly_factor.rs:256`, `:263`); `Gfpn` keeps the O(min(n, p)) default `from_int` (`ff_poly.rs:61`) | Confirmed (`t4_rand.log`): canonical never draws a coefficient >= 2^63 (0 of 360); lava draws only 0 and 1 for p = 2^80+13; `from_int(10^6)` on `Gfpn` takes 1.4 s. Also found (`t8_badp.log`): `poly_factor::find_irreducible(15, 2)` returns `x^2+1`, `random_irreducible(1, 2, 7)` returns `x^2`, `generate_gfpn_modulus(15, 2)` returns `x^2+1` | Every coefficient drawn with the crate's exact sampler `uniform_below(p, ..)`; the generators and the public `is_irreducible_fp` check that p is prime; `from_int` is O(1) on `Gfpn` and O(log p) additions in the trait default |
| 4 | O(m) order loops (`integer_mod.rs:220`, `prime_field.rs:104`) and trial division of q - 1 (`finite_field.rs:611`) | Confirmed (`t3_order*.log`): order of 2 mod 10^9+7 takes 523 s; GF(p16) takes 21.8 s; GF(p30) is out of reach (1.4·10^15 trial divisions) | New `group_order` module: Cohen GTM 138 Alg. 1.4.3 on a factored multiple of the order (m - 1 when a^(m-1) = 1 mod m, else Carmichael λ(m) for Z/mZ; q - 1 split by cyclotomic values for GF(p^n)), factored with I2's `factorization_with_limits`; ±1 need no factoring. Inherent methods are complete; the `Ring` hooks have a cost cap (sieve to 30 digits, proofs to 256 bits) and answer `Undecided` beyond it. Adds MAGMA `FactoredOrder` |
| 5 | MAGMA `IsSquare`/`Sqrt` with the root on `PrimeField` and GF(p^n) | Confirmed: `PrimeField` has only the bool `is_quadratic_residue`; `FiniteFieldElement` has nothing | `PrimeField::sqrt`/`is_square`; `FiniteFieldElement::sqrt`/`is_square` (Euler's criterion decides, Tonelli-Shanks in GF(q)^*, inverse Frobenius in characteristic 2), canonical least root; `ExtensionField` forwards |
| 6 | `square_root_mod_prime` runs `is_prime` twice (`integer_mod.rs:651`) | Confirmed by I1 (I1 design §1.4) | Done by I1 §2.7 (`Integer::try_sqrt_mod_prime`, one test). F1 verifies it and uses the same primitive in `PrimeField::sqrt`; a fallback on HEAD's API in section 6 if I1 did not land it |
| 7 | Galois-ring containers built from `R::one()`/`var()` keep unbound coefficients (degree 7 instead of 6) | Confirmed: pinned by `tests/galois_ring_sentinel.rs:196`; `MultivariatePolynomial` and `LaurentPolynomial` keep 8 terms for the same value (`$X/rev1/rev1_cont.log`) | New core hooks `Ring::HAS_UNBOUND_CONSTANTS`, `is_unbound_constant`, `bind_to_parent_of` (defaults: none); the four sentinel types and `ExtensionField` implement them. `UnivariatePolynomial::new` binds every unbound coefficient into the parent of a bound sibling before trimming; `MultivariatePolynomial` and `LaurentPolynomial` bind in `add_term` and drop the terms that become zero; all three get a coercing `PartialEq`, and `UnivariatePolynomial` and `LaurentPolynomial` implement the hooks (nested containers bind). Degree 6, `q == qb`. Same fix for `IntegerMod`, `PrimeField`, GF(p^n) |

Nothing in the unit is declared out of scope. Two limits are stated, not hidden:
- Item 7 fixes the three containers of rustmath-polynomials, the crate F1 already edits. The generic
  containers of rustmath-matrix (`Matrix`, `Vector`, `SparseMatrix`), rustmath-powerseries and
  rustmath-rings (power and Laurent series) still bind entry by entry, as at HEAD. They are an explicit
  hand-off (§7.6 lists each type, its interim behaviour and the hook call that fixes it). rustmath-matrix
  and rustmath-rings belong to units designed now against HEAD (M1a, M1b, R1), so editing them here would
  collide with those designs. rustmath-powerseries belongs to no unit; it is a separate crate (one crate per
  PR), and no code uses its series over a sentinel type. The matrix code already builds its constants from a bound entry
  (`krylov.rs:36-41`) and documents what happens to matrices of only sentinel entries (`krylov.rs:43-47`,
  `companion.rs:36-41`: an `Err` and a fallback, never a wrong value).
- Item 4: a `Ring` hook answers `Undecided` when the group exponent cannot be factored and proven within
  its cost cap (a proof of nothing, as the trait requires). The inherent methods are exact and may be slow
  for a group order that is hard to factor; their `# Cost` says so.

Findings outside the items (section 8): `FiniteField::new(p, n)` without a Conway polynomial can take O(p)
Rabin tests (`FiniteField::new(2^80+13, 3)` does not return); canonical `roots(0)` and `factor(0)` return
`[]`. They go to HANDOFF.

### 0.1 APIs this unit consumes from earlier units

| Unit | API (as designed) | Used for | If it landed under another name |
|---|---|---|---|
| I1 | `rustmath_integers::prime::try_is_prime(&Integer) -> Result<bool>` (a proof; `Err` when ECPP is unresolved) | characteristic checks (items 1, 2, 3) | at HEAD `Integer::is_prime` (a proof below psi_12) with `Ok(..)` around it |
| I1 | `prime::is_probable_prime(&Integer) -> bool` (BPSW) | `PrimeField::sqrt` precheck (item 5) | HEAD `is_prime` |
| I1 | `Integer::try_sqrt_mod_prime(&self, p) -> Result<Option<Integer>>` (no primality test; a returned root is squared back; `None` only on Jacobi -1) | items 5, 6 | HEAD `Integer::sqrt_mod_prime` (tests p again; section 6 gives the HEAD fallback) |
| I1 | `prime::prove_primality(&Integer) -> Primality { Prime(cert), NotPrime(w), Unresolved(String) }` | item 4, `Effort::Capped` proves its probable primes of at most 256 bits | none at HEAD (no ECPP): every probable prime >= psi_12 then gives `Err` (an `Undecided` hook) |
| I2 | `rustmath_integers::factorization::{factorization_with_limits, FactorizationLimits, LimitedFactorization}` (`primes`, `probable_primes`, `composites`, `is_complete()`; limits `trial_division_limit`, `pollard_rho_limit`, `mpqs_limit`, `proof`) | item 4 | HEAD `prime::factor_pollard_rho` for `Effort::Complete` and `factorint::factor_trial_division(n, Some(10_000))` for `Effort::Capped`, accepted only when its last entry passes `is_prime` (that entry may be a composite cofactor) (the prototype `p2_order.rs` runs on HEAD's `factor_pollard_rho`) |
| P1 | `Ring::static_is_commutative` in `traits.rs` | item 7 places its hooks after it | none needed |

Every consumer sits behind one private function (`group_order::factor_proven`,
`ff_poly::check_prime_characteristic` (moved there from `cantor_zassenhaus.rs:141`), `poly_factor::check_char`,
`PrimeField::sqrt`, `square_root_mod_prime`), so a renamed API is a one-line change.

Error and panic texts: a `String` error appears verbatim in a panic; a `MathError` appears through its
`Display`, which prefixes the variant (`InvalidArgument(s)` displays as "Invalid argument: {s}",
`rustmath-core/src/error.rs:20`). Every expected panic text below is written out in full on that basis.

---

## 1. Item 1: distinct-degree and squarefree factorizations check their preconditions

MAGMA ch. 23 §23.8.1 (digest `rustmath-lava/docs/Chapter_23_Univariate_Polynomial_Rings.md:298-299`):
`DistinctDegreeFactorization(f)` is defined for "a squarefree f ∈ F[x] (F a finite field)";
`SquarefreeFactorization(f)` for "f over Z or any field".

### 1.1 Defect at HEAD

`cantor_zassenhaus.rs:448` and `poly_factor.rs:613` loop on `gcd(x^(q^d) - x, f)` with no check;
`cantor_zassenhaus.rs:52-68` says "`squarefree_factorization` and `distinct_degree_factorization` check
nothing". After step d only one copy of a repeated degree-d factor is removed; the second one is caught at
a multiple of d or lands in the tail block. `$X/t1_ddf.log` (lava = `cantor_zassenhaus`, canon = `poly_factor`):

| Input | Both DDFs at HEAD | Reference (gp, `$X/t1_ref.gp`) |
|---|---|---|
| `(x+1)^2` mod 5 = `[1,2,1]` | `[(x+1, 1), (x+1, 1)]` | not squarefree |
| `(x^2+1)^2` mod 3 = `[1,0,2,0,1]` | `[(x^2+1, 2), (x^2+1, 2)]` | not squarefree |
| `(x+1)^2(x^2+2)(x^2+3)` mod 5 = `[1,2,1,0,1,2,1]` | `[(x+1, 1), (x^5+x^4+x+1, 2)]` | `factormod(...)` has `x+1` twice; `factormod(x^5+x^4+x+1, 5)` = `(x+1)(x^2+2)(x^2+3)`: the "degree 2" block holds a linear factor |
| `x^2` mod 2 (f' = 0) | `[(x, 1), (x, 1)]` | not squarefree |
| `0` mod 5 | `[]` | 0 is not squarefree |

Lava with a composite characteristic (`PrimeField::new` does not check primality):

| Call | HEAD | Reference |
|---|---|---|
| `cz::squarefree_factorization(x^2+1 mod 15)` | `[(x^2+1, 1)]` | no field; meaningless |
| `cz::distinct_degree_factorization(x^2+1 mod 15)` | `[(x^2+1, 2)]` ("irreducible of degree 2") | mod 3 irreducible, mod 5 splits (gp `factormod(x^2+1,3)`, `factormod(x^2+1,5)`) |
| `cz::squarefree_factorization((x+1)^2 mod 4)` | panic `nonzero divisor: ... gcd(2, 4) = 2` from `ff_poly.rs:469` | a precise refusal |
| `cz::distinct_degree_factorization((x+1)^2 mod 4)` | `[(x+3, 1), (x+3, 1)]` | |

Both squarefree factorizations are correct over a field: `$X/t5_sqf.rs` builds `a·b^2·c^p` at random over
GF(2), GF(3), GF(5), GF(7), GF(4), GF(9), GF(8) (canonical, 999 cases) and GF(2..7) (lava, 766 cases) and
checks that every part is squarefree, the parts are pairwise coprime and their product is the monic `f`:
0 failures. So routing `factor` through a checked DDF cannot start panicking on valid input.

### 1.2 Fix

Shared private helper in each module (`cantor_zassenhaus.rs` generic over `F: ff_poly::FiniteFieldElement`,
`poly_factor.rs` on `FqPoly`):

```rust
/// Ok(()) iff the monic, nonconstant f is squarefree: f' != 0 and gcd(f, f') = 1. Over GF(q) a
/// nonconstant f with f' = 0 is g(x)^p, so this test is exact. Err("f is not squarefree").
fn squarefree_check(fm: &Poly) -> Result<(), &'static str>;
```

`block_contract` (`cantor_zassenhaus.rs:180-183`) and `fq_edf_contract` (`poly_factor.rs:719-721`) and
`factor_squarefree_prime_char` (`cantor_zassenhaus.rs:539-542`) already contain this code inline; they call
the helper instead (same message, so the existing should-panic tests keep their expectations).

**`cantor_zassenhaus::distinct_degree_factorization`** (signature unchanged):

```rust
/// MAGMA `DistinctDegreeFactorization(f)` (ch. 23 §23.8.1). Returns (g_d, d), d increasing, g_d the
/// monic product of the irreducible factors of f of degree exactly d. A nonzero constant gives vec![].
///
/// # Panics
/// In this order: the characteristic is not a prime (`check_prime_characteristic`); f = 0
/// ("f = 0 is not squarefree"); f is not squarefree ("f is not squarefree"). Each message starts with
/// "cantor_zassenhaus::distinct_degree_factorization: ".
pub fn distinct_degree_factorization<F: FiniteFieldElement>(f: &FFPoly<F>) -> Vec<(FFPoly<F>, usize)>
```

The HEAD loop (`:451-480`) moves unchanged into `fn ddf_squarefree<F>(fm: &FFPoly<F>) -> Vec<..>` (monic,
squarefree, prime characteristic: the caller's duty). `factor_squarefree_prime_char` (`:546`) calls
`ddf_squarefree` (it has just checked), so no gcd runs twice.

**`cantor_zassenhaus::squarefree_factorization`** (signature unchanged): checks the characteristic first
(panic `"cantor_zassenhaus::squarefree_factorization: characteristic 15 is not a prime"`), then the HEAD
body moves into `fn squarefree_prime_char<F>(f) -> Vec<..>`. `factor` (`:601`) calls `squarefree_prime_char`
(it checked the characteristic at `:590`), so the primality test runs once per `factor` call. A zero or
constant `f` still gives `[]` (unchanged; the same convention as
`rustmath_polynomials::factorization::square_free_factorization`, pinned by
`tests/ch23_quotrem_gcd.rs:1441`); the doc says so. The `p.to_usize().unwrap_or(0)` at `:372` stays: for
p > usize::MAX a nonconstant f with f' = 0 has degree >= p, which no `Vec` holds, so `pth_root` is never
reached; the doc comment states this.

**`check_prime_characteristic`** (`:141-148`) uses I1's `try_is_prime`: `Ok(true)` passes, `Ok(false)` is
`Err("characteristic {c} is not a prime")`, `Err(e)` is `Err("the primality of the characteristic {c} is
not decided ({e})")`. Its doc and the module docs (`:52-68`) drop the psi_12 caveat (I1 makes the test a
proof) and the sentences "A `Gfpn` whose defining polynomial is reducible ... is not checked" and
"`squarefree_factorization` and `distinct_degree_factorization` check nothing" (items 1 and 2 remove both).

**`poly_factor::distinct_degree_factorization`** (signature unchanged): f = 0 panics
`"poly_factor::distinct_degree_factorization: f = 0 is not squarefree"`; a constant gives `[]`; a
non-squarefree f panics `"poly_factor::distinct_degree_factorization: f is not squarefree"`; then
`fn ddf_squarefree(fm: &FqPoly)` (the HEAD body `:614-638`). No characteristic check: a `FiniteField` is
a field by construction (`FiniteField::new`/`with_modulus` check p and the modulus).

**`poly_factor::factor`** (`:850`): calls `squarefree_check` on each part of `square_free_factorization`,
panicking `"poly_factor::factor: a part returned by square_free_factorization fails the squarefree check
({why}); this is a bug"` (the lava `factor` does the same at `:602-607`), then `ddf_squarefree`. One gcd per
part, negligible next to the Frobenius powers of the DDF.

Correctness: with f squarefree over GF(q), every irreducible factor occurs once, so after step d all
factors of degree <= d are removed and the remaining factor of degree < 2(d+1) is irreducible ([vzGG]
§14.2, Rabin 1980). The check is exact (f' = 0 iff f = g^p over a perfect field).

### 1.3 Callers (grep `distinct_degree_factorization\|squarefree_factorization\|square_free_factorization` over the workspace)

| Caller | Input | Change |
|---|---|---|
| `cantor_zassenhaus.rs:546` (`factor_squarefree_prime_char`) | checked squarefree, prime characteristic | calls `ddf_squarefree`; no extra cost |
| `cantor_zassenhaus.rs:601` (`factor`) | characteristic checked | calls `squarefree_prime_char`; no extra cost |
| `poly_factor.rs:853` (`factor`) | parts of `square_free_factorization` | one squarefree check per part (0 bad in 999 random cases) |
| `poly_factor.rs:580`, `:603` (recursion in `square_free_factorization`) | unchanged | none |
| `poly_factor.rs:1021` (test `distinct_degree_split`), `cantor_zassenhaus.rs:734` (test) | squarefree / prime | none |
| `rustmath-polynomials` `factorization::square_free_factorization` | a different function | none |

No caller outside the crate. No existing behaviour changes on valid input.

### 1.4 Tests

| Name (module) | Input | Expected | HEAD |
|---|---|---|---|
| `ddf_rejects_non_squarefree` (`cantor_zassenhaus` tests) | the four non-squarefree rows of 1.1 over `PrimeField` | each panics with "cantor_zassenhaus::distinct_degree_factorization: f is not squarefree" (one `#[should_panic]` per row, or a `catch_unwind` loop over a `panic_message` helper) | fails: returns the lists above |
| `ddf_rejects_zero` (cz) | `[0]` mod 5 | "... f = 0 is not squarefree" | fails (`[]`) |
| `ddf_rejects_composite_characteristic` (cz) | `x^2+1` mod 15 | "... characteristic 15 is not a prime" | fails (`[(x^2+1, 2)]`) |
| `squarefree_factorization_rejects_composite_characteristic` (cz) | `x^2+1` mod 15; `(x+1)^2` mod 4 | "cantor_zassenhaus::squarefree_factorization: characteristic 15 is not a prime" / "... 4 ..." | fails (returns; panics with another message) |
| `squarefree_factorization_rejects_the_sentinel` (cz) | a polynomial whose sample is `PrimeField::zero()` (modulus 0) | "... characteristic 0 is not a prime" | fails |
| `ddf_squarefree_matches_pari` (cz and `poly_factor`) | `x^5+x^4+x+1` mod 5; `[3]` mod 5 | `[(x+1, 1), (x^4+1, 2)]` (gp `factormod(x^5+x^4+x+1,5)`: `x+1`, `x^2+2`, `x^2+3`, and `(x^2+2)(x^2+3) = x^4+1` mod 5); `[]` | passes |
| `ddf_rejects_non_squarefree` (`poly_factor` tests) | rows of 1.1 over `FiniteField::new(p, 1)`, plus `(x + w)^2` over GF(4) | "poly_factor::distinct_degree_factorization: f is not squarefree" | fails |
| `ddf_rejects_zero` (`poly_factor`) | zero over GF(5) | "... f = 0 is not squarefree" | fails |

Existing tests that must stay green: `factor_squarefree_rejects_*`, `edf_rejects_*`, `distinct_degree_split`,
`test_squarefree_factorization`, `factor_*` (both modules).

### 1.5 Cost

One gcd(f, f') per public DDF call; none on the internal paths. The primality test of the characteristic
runs once per public call (I1: about 10 µs below 2^64; above psi_12 one ECPP proof, then cached by I1).

---

## 2. Item 2: every `Gfpn` is a field

MAGMA ch. 21 §21.2 (digest `Chapter_21_Finite_Fields.md`): `ext<F | f>` and `FiniteField(p, n)` require an
irreducible f; the canonical `FiniteField::with_modulus` (`finite_field.rs:184-212`) already checks prime p,
monic and Rabin.

### 2.1 Defect at HEAD (`$X/t2_gfpn.log`)

`Gfpn::new(coeffs, p, modulus)` (`ff_poly.rs:590-619`) checks nothing:

| Call | HEAD | Reference |
|---|---|---|
| "GF(9)" = F3[t]/(t^2+2) = F3[t]/((t+1)(t+2)): `invert(t+1)` | `Err("element not invertible (modulus not irreducible?)")` | gp `polisirreducible(Mod(1,3)*(x^2+2))` = 0 |
| same ring: `order()` | `9` | not a field |
| same ring: `cz::factor(y^2+1)` | panic `nonzero divisor: ... element not invertible` deep in the gcd | in GF(9) (t^2+1) it splits: gp `factorff(y^2+1, 3, t^2+1)` gives two linear factors; the genuine field gives `[(1,1),(1,1)]` at HEAD too |
| `p = 15`, modulus `t^2+1`: `order()` | `225` | no field |
| modulus `2t^2+1` over F5: `t * t` | `[4, 0]` (silently wrong: `mul` assumes monic) | gp `lift(lift(Mod(Mod(1,5)*x, Mod(1,5)*(2*x^2+1))^2))` = 2 |
| modulus `[]` | panic "attempt to subtract with overflow" | a precise refusal |
| modulus `[1]` | panic "index out of bounds" in `is_one` | a precise refusal |
| modulus `[6,1,1]` vs `[1,1,1]`, p = 5: `new([1], 5, [6,1,1]) == new([1], 5, [1,1,1])` | `false` (`$X/rev1/rev1_cont.log`) | the same field: `[6,1,1]` mod 5 = `[1,1,1]` and gp `polisirreducible(Mod(1,5)*(t^2+t+1))` = 1 (`$X/rev1/rev1_ref.log`) |

(The first version of this row used p = 7, where `t^2+t+1 = (t+3)(t+5)` is reducible, gp `factormod`; the
designed `try_new` rejects both moduli there. The `t2_gfpn.log` line "unreduced modulus [8,1,1] p=7" was run
on that reducible modulus and only shows that HEAD compares unreduced vectors.)

### 2.2 Constructor paths (grep `Gfpn {` and `Gfpn::new` over the workspace)

- `Gfpn::new` (`ff_poly.rs:590`) is the only public constructor.
- Every other `Gfpn { .. }` literal (`:599`, `:614`, `:639`, `:670`, `:680`, `:698`, `:710`, `:752`, `:806`,
  `:830`) copies `p` and `modulus` from an existing `Gfpn`, so validity is inherited.
- `Gfpn` has no `Default`, no `serde` derive (the crate's `serde` feature does not touch it) and no
  `From` impl.
- Callers of `Gfpn::new`: `rustmath-galois/src/galois_ctx.rs:213`, `:585` (modulus = a mod-p irreducible
  factor from `fp_factor`, padded to length d + 1, which it already has); `relative_invariant.rs:300`,
  `:379`, `:382` (modulus from `generate_gfpn_modulus`, Rabin-certified); tests in `ff_poly.rs`,
  `irreducible.rs`, `cantor_zassenhaus.rs`.

### 2.3 Fix

```rust
impl Gfpn {
    /// GF(p^n) = F_p[t]/(f) with f = `modulus` (low degree first). Validates, in this order:
    /// - `modulus.len() >= 2`, else InvalidArgument("Gfpn: the defining polynomial must have degree >= 1");
    /// - p is prime (I1 `try_is_prime`): Ok(false) gives InvalidArgument("Gfpn: the characteristic {p}
    ///   is not prime"); Err(e) is returned as is (unresolved primality);
    /// - the modulus reduced into [0, p) is monic, else InvalidArgument("Gfpn: the defining polynomial
    ///   is not monic modulo {p}");
    /// - for degree >= 2, `poly_factor::is_irreducible_fp_unchecked` (Rabin; p is proven prime by
    ///   now), else InvalidArgument("Gfpn: the defining polynomial is reducible over GF({p})").
    /// The stored modulus is the reduced one, so moduli that agree mod p define the same field.
    /// `coeffs` are reduced mod p and mod f as before.
    pub fn try_new(coeffs: Vec<Integer>, p: Integer, modulus: Vec<Integer>) -> Result<Self>;

    /// [`Gfpn::try_new`], panicking with the error's `Display` (signature unchanged):
    /// `Self::try_new(..).unwrap_or_else(|e| panic!("{e}"))`. Example of a panic text:
    /// "Invalid argument: Gfpn: the defining polynomial is reducible over GF(3)".
    pub fn new(coeffs: Vec<Integer>, p: Integer, modulus: Vec<Integer>) -> Self;
}
```

Memo (validation is a Rabin test, 4.3 ms for degree 6 over p = 10007, while `new` costs 2.1 µs):

```rust
/// (p, reduced modulus) pairs that passed `try_new`'s checks. Grows with the number of distinct
/// fields, like the canonical `FF_CACHE`; never evicted.
static GFPN_VALID: LazyLock<RwLock<HashMap<Integer, HashSet<Vec<Integer>>>>> = ...;
```

`try_new`: length check; reduce the modulus (skip the allocation when every entry is already in
[0, p)); read-lock and look up `p` then the slice (`HashSet<Vec<Integer>>::contains(&[Integer])`, no key
clone); on a miss run the checks without holding a lock, then write-lock and insert. A poisoned lock is
recovered with `into_inner()` (no code panics while holding it). Measured lookup with a key clone 1.5 µs
(`t2_gfpn.log`); the borrowed lookup is cheaper.

Consequences inside `ff_poly.rs`: `Gfpn::invert`'s "modulus not irreducible?" branch (`:787-791`) becomes
unreachable; keep it as an `Err` with the text "internal: gcd with the defining polynomial is not 1",
never a guess. The `FiniteFieldElement for Gfpn` docs and the `cantor_zassenhaus` module docs say a `Gfpn`
is a field by construction.

### 2.4 Callers and behaviour changes

- `rustmath-galois`: moduli are irreducible, so values are unchanged; cost is one Rabin test per distinct
  `(p, modulus)` per process (`build_ctx` picks one prime per polynomial; a context has at most deg f
  factor moduli).
- Tests that build a `Gfpn` over a reducible modulus: none found (`grep -n "Gfpn::new" ...`); if one
  appears in the acceptance run it was relying on undefined behaviour and must move to a field.
- Visible changes: invalid input now panics in `new` with the reason (or `Err` from `try_new`); an
  unreduced modulus is normalized (`modulus()` returns the reduced vector).

### 2.5 Tests (in `ff_poly.rs` tests)

| Name | Input | Expected | HEAD |
|---|---|---|---|
| `gfpn_rejects_reducible_modulus` | `try_new([1,1], 3, [2,0,1])` (gp `polisirreducible(Mod(1,3)*(x^2+2))` = 0); `new` in a `#[should_panic(expected = "Invalid argument: Gfpn: the defining polynomial is reducible over GF(3)")]` twin | `Err(InvalidArgument("Gfpn: the defining polynomial is reducible over GF(3)"))` | accepted |
| `gfpn_rejects_composite_characteristic` | `try_new([1,1], 15, [1,0,1])` | `Err(InvalidArgument("Gfpn: the characteristic 15 is not prime"))` | accepted, `order() = 225` |
| `gfpn_rejects_non_monic_modulus` | `try_new([0,1], 5, [1,0,2])` | `Err(InvalidArgument("Gfpn: the defining polynomial is not monic modulo 5"))` | accepted, `t*t = 4` (gp: 2) |
| `gfpn_rejects_degenerate_modulus` | modulus `[]` and `[1]` | `Err(InvalidArgument("Gfpn: the defining polynomial must have degree >= 1"))` | overflow / index panic |
| `gfpn_normalizes_the_modulus` | `[6,1,1]` and `[1,1,1]` at p = 5 (gp `polisirreducible(Mod(1,5)*(t^2+t+1))` = 1) | both `Ok`; `modulus() == [1,1,1]` for both; `new([1], 5, [6,1,1]) == new([1], 5, [1,1,1])` | `false` (`$X/rev1/rev1_cont.log`) |
| existing `test_gfpn_field_ops`, `test_gfpn_gf9`, `prime_field_random_element_is_fast_and_full_range` (GF((2^80+13)^2) with `t^2+t+1`, gp `polisirreducible` = 1), `cantor_zassenhaus` Gfpn tests | valid fields | unchanged | pass |

### 2.6 Cost

First construction of a field: one primality test (I1) and one Rabin test (`m` Frobenius powers of degree
`m` polynomials). Later constructions: reduction check plus a hash lookup, about the cost of the existing
coefficient reduction.

---

## 3. Item 3: uniform irreducible-polynomial generators, checked characteristic, O(1) `from_int`

MAGMA ch. 21 §21.5 (digest `:265-266`): `RandomIrreduciblePolynomial(F, n)` is a "random irreducible
polynomial", by "random polynomial with irreducibility testing"; `IrreduciblePolynomial(F, n)`.

### 3.1 Defects at HEAD

Draws (`$X/t4_rand.log`, 60 seeds per row, n = 1, 2, 3):

| p | canonical `poly_factor::random_irreducible` (`:256`, `:263`: `redp(Integer::from(next as i64).abs(), p)`) | lava `irreducible::random_irreducible` (`:160-172`) |
|---|---|---|
| 2^64 - 59 | 0 of 360 coefficients >= 2^63 (expected about half) | 144 of 300 >= 2^63; draws go through `as i64` and modulo bias |
| 2^80 + 13 | 0 of 360 >= 2^63 (expected almost all) | every coefficient in {0, 1} (`p.to_usize().unwrap_or(2)`) |
| 10^9 + 7 | in range | in range |

Each candidate is Rabin-tested, so outputs are irreducible; the defect is the distribution (MAGMA: "a
random irreducible polynomial"; a uniform monic candidate conditioned on irreducibility is uniform over the
monic irreducibles).

Non-prime p (`$X/t8_badp.log`), found while checking the same functions:

| Call | HEAD |
|---|---|
| `poly_factor::find_irreducible(15, 2)` | `Ok(x^2 + 1)`: a fabricated "irreducible over F_15" |
| `poly_factor::random_irreducible(1, 2, 7)`, `find_irreducible(1, 2)` | `Ok(x^2)` |
| `poly_factor::random_irreducible(15, 2, 7)`, `(4, 2, 7)`, `find_irreducible(4, 2)` | panic "not invertible mod p" |
| `poly_factor::{random,find}_irreducible(0, ..)` | panic "attempt to divide by zero" |
| `irreducible::generate_gfpn_modulus(15, 2)` | `Some(x^2 + 1)` (fed to `Gfpn::new`) |
| `irreducible::random_irreducible(15, 2, 7)`, `(4, ..)`, `(1, ..)`, `(0, ..)` | panics from `PrimeField` or a division by zero |

`from_int` (`ff_poly.rs:61-81`): the trait default adds `one` to itself `n mod p` times; `Gfpn` does not
override it. `Gfpn::from_int(10^6)` over p = 2^61 - 1 takes 1.42 s (linear: 10^5 takes 0.14 s);
`FFPoly::derivative` (`:481-491`) calls it for every coefficient index.

### 3.2 Fix

Sampler: the crate-private exact rejection sampler `ff_poly::uniform_below(m, next)` (`:146`), which
`random_element` and `poly_factor::random_fq_elt` already use.

**Canonical `poly_factor::random_irreducible(p, n, seed) -> Result<Vec<Integer>>`**:
1. `n == 0`: `Err(InvalidArgument("degree must be >= 1"))` (unchanged).
2. `check_char(p)`: I1 `try_is_prime(p)`; `Ok(false)` gives `Err(InvalidArgument("characteristic {p} is not
   prime"))`, `Err(e)` is returned.
3. `fn random_irreducible_unchecked(p, n, seed)`: `n == 1` seeds `Rng(seed)` and returns `[c, 1]` with
   `c = uniform_below(p, next)`; `n >= 2` seeds `Rng(seed ^ 0xA5A5_5A5A_0F0F_F0F0)` (both seedings kept),
   draws `c_0, ..., c_{n-1}` in that order with `uniform_below`, appends 1, Rabin (`is_irreducible_fp_unchecked`),
   at most `512 n` candidates, then `Err(NotSupported(..))` (unchanged; it decides nothing).

**Canonical `poly_factor::find_irreducible(p, n)`**: same `check_char` first; body into
`fn find_irreducible_unchecked(p, n)`. `FiniteField::new` (`finite_field.rs:177`) and
`irreducible_polynomial` (`:912`) call the unchecked variant (their p is already proven prime), so no
primality test runs twice.

**Canonical `poly_factor::is_irreducible_fp(f, p) -> bool`** (public, re-exported at `lib.rs:57`; review 1,
m4): at HEAD it answers `true` for `x^2+1` mod 15 (`$X/review1/r3_irr.rs`), a statement about a ring that is
not a field. New body:

```rust
/// Rabin irreducibility test for a monic f over F_p.
/// # Panics
/// When p is not proven prime, with "poly_factor::is_irreducible_fp: {e}" where e is the `MathError` of
/// `check_char` (e.g. "poly_factor::is_irreducible_fp: Invalid argument: characteristic 15 is not prime").
pub fn is_irreducible_fp(f: &[Integer], p: &Integer) -> bool {
    if let Err(e) = check_char(p) { panic!("poly_factor::is_irreducible_fp: {e}") }
    is_irreducible_fp_unchecked(f, p)
}
/// The HEAD body (`:171-210`); precondition: p is prime (the caller proved it).
pub(crate) fn is_irreducible_fp_unchecked(f: &[Integer], p: &Integer) -> bool;
```

A `bool` predicate has no honest value over a non-field, so it panics (as the lava `is_irreducible` below);
the signature stays. Callers that have proven p call the unchecked body, so no primality test runs twice:
`find_irreducible_unchecked` (`:222`), `random_irreducible_unchecked` (`:266`),
`FiniteField::with_modulus` (`finite_field.rs:202`, p checked at `:185`), `GaloisRing::with_modulus`
(`galois_ring.rs:120`, p checked at `:106`), `Gfpn::try_new` (item 2). The in-crate tests
(`poly_factor.rs:956-970`, `:1086`, `:1201`, `conway.rs:360`) keep the checked public function (small p,
one cheap test per call).

`check_char(p: &Integer) -> Result<()>` (crate-private): I1 `try_is_prime(p)`; `Ok(true)` gives `Ok(())`,
`Ok(false)` gives `Err(InvalidArgument(format!("characteristic {p} is not prime")))`, `Err(e)` is returned
as is (an unresolved proof, `NotSupported`).

**Lava `irreducible::random_irreducible(p, n, seed) -> Option<FFPoly<PrimeField>>`**: `None` for `n == 0`
or when p is not proven prime (`try_is_prime` false or `Err`; the doc says both). Then `SplitMix64::new(seed)`,
coefficients `c_0..c_{n-1}` drawn with `uniform_below(p, ..)` in order, leading 1, Rabin through the private
`fn rabin_irreducible(f)` (the HEAD body of `is_irreducible`), at most `64 n` candidates, then
`find_irreducible_unchecked(p, n)` (unchanged fallback; reached with probability about e^-64 per call).
`poly_from_index` stays for `find_irreducible`.

**Lava `find_irreducible(p, n)` and `generate_gfpn_modulus(p, n)`**: `None` when p is not proven prime;
`generate_gfpn_modulus` calls `find_irreducible` once (one test).

**Lava `is_irreducible(f)`**: checks the characteristic of `f.sample()` with
`cantor_zassenhaus`'s `check_prime_characteristic` logic (move it to `ff_poly.rs` as `pub(crate) fn
check_prime_characteristic`, used by both modules) and panics `"irreducible::is_irreducible: characteristic
15 is not a prime"`; then `rabin_irreducible`. A bool predicate over a non-field has no honest answer, and
`factor` already panics there.

**`from_int`**:
- Trait default (`ff_poly.rs:61`): `r = n mod p` in [0, p) (as now), then double-and-add over the bits of
  r (`acc = acc + acc`, plus `one` on a 1 bit): O(log p) additions. Same value.
- `impl FiniteFieldElement for Gfpn`: `fn from_int(&self, n) -> Self` returns the element with coefficient
  vector `[n mod p, 0, ..., 0]` directly (same field as `self`, no validation). O(1) big-integer operations.

Correctness: `uniform_below(p, ..)` is exact for every p (its own doc and tests `uniform_below_scripted`);
independent coordinates give a uniform monic candidate; rejection on irreducibility keeps uniformity over
the monic irreducibles. The double-and-add computes r·1 in O(log r) steps.

### 3.3 Callers and behaviour changes (grep `random_irreducible\|find_irreducible\|generate_gfpn_modulus\|from_int`)

| Caller | Change |
|---|---|
| `poly_factor.rs:1083-1092` (`random_irreducible_various_p_n`) | checks irreducibility and determinism only: still passes; the polynomials themselves change for most seeds (no test pins them) |
| `irreducible.rs:243` (`test_random_irreducible_deterministic`) | same |
| `finite_field.rs:177`, `poly_factor.rs:912` | call `find_irreducible_unchecked`; same values |
| `is_irreducible_fp` callers (grep over the workspace: only `rustmath-finitefields`; `rustmath-numberfields/src/local_field.rs:37` and `rustmath-rings/src/valuation/maclane.rs:1579` define private functions of the same name) | `finite_field.rs:202`, `galois_ring.rs:120`, `poly_factor.rs:222`, `:266` and `Gfpn::try_new` call `is_irreducible_fp_unchecked`; tests keep the public function; same values for prime p |
| `rustmath-galois/src/relative_invariant.rs:472` (`generate_gfpn_modulus(&p, m)?`) | p is prime there; unchanged. A composite p now gives `None` (propagated by `?`) instead of a reducible modulus |
| `FFPoly::derivative` over `Gfpn`; `random_element` default (unused: both impls override it) | faster, same values |
| No caller outside the crate for `random_irreducible`, `find_irreducible`, `is_irreducible` (lava) | |

### 3.4 Tests

| Name | Input | Expected | HEAD |
|---|---|---|---|
| `random_irreducible_draws_the_whole_field` (`poly_factor` tests) | `random_irreducible(2^80+13, 1, 7)`; `(2^80+13, 2, 1)`; `(2^80+13, 3, 42)`; `(2^64-59, 2, 1)` | `[482205081176368571616727, 1]`; `[566343447911828653673491, 847000722540561595848212, 1]`; `[211961487743374730434531, 828113852171267793509212, 1001365246395949401026822, 1]`; `[8730120229108003598, 5076488698419838370, 1]` (`$X/p3_randirr.log`; gp `$X/p3_ref.gp`: `polisirreducible` = 1 for each) | fails (all coefficients < 2^63) |
| same test, range part | seeds 0..64, n = 2: p = 2^64-59, count of coefficients >= 2^63; p = 2^80+13, count >= 2^64 | in [32, 96] (prototype 65 of 128); 128 of 128 | 0; 0 |
| `random_irreducible_draws_the_whole_field` (`irreducible` tests) | lava `(2^80+13, 2, 1)`, `(2^80+13, 3, 42)`, `(2^64-59, 2, 1)` | `[1116388955813207555071169, 949416489080899775911262, 1]`; `[445155756734992886557959, 520708005305936630825557, 826732766301133076639116, 1]`; `[10451216379200822465, 13757245211066428519, 1]` (gp: irreducible); range counts as above (prototype 67 of 128; 128 of 128) | fails (only 0 and 1 for 2^80+13) |
| `generators_reject_non_prime_p` (`poly_factor`) | `find_irreducible(15, 2)`, `random_irreducible(1, 2, 7)`, `(0, 2, 7)`, `(4, 2, 7)` | `Err(InvalidArgument(..not prime..))` each | `Ok(x^2+1)`, `Ok(x^2)`, panics |
| `generators_reject_non_prime_p` (`irreducible`) | `find_irreducible(15, 2)`, `random_irreducible(15, 2, 7)`, `generate_gfpn_modulus(15, 2)`, `(0, 2)` | `None` each; `is_irreducible` over `PrimeField` mod 15 panics "irreducible::is_irreducible: characteristic 15 is not a prime" (a `String` error, verbatim) | `Some(x^2+1)`, panics |
| `is_irreducible_fp_rejects_non_prime_p` (`poly_factor` tests) | `is_irreducible_fp(&[1, 0, 1], 15)`; the same with p = 1 and p = 0 | `#[should_panic(expected = "poly_factor::is_irreducible_fp: Invalid argument: characteristic 15 is not prime")]`; for 1 and 0 a `catch_unwind` loop checks "characteristic 1 is not prime", "characteristic 0 is not prime" | `true` for 15 (`$X/review1/r3_irr.rs`) |
| `is_irreducible_fp_unchanged_for_primes` (`poly_factor` tests) | `x^2+x+1` mod 2 and mod 5, `x^2+1` mod 5, `x^4+x+1` mod 2 | `true`, `true`, `false`, `true` (gp `polisirreducible(Mod(1,p)*f)`) | passes |
| `gfpn_from_int_is_constant_time` (`ff_poly`) | GF((2^61-1)^2) = F_p[t]/(t^2 - 3) (gp: `kronecker(3, 2^61-1)` = -1, `polisirreducible(Mod(1,p)*(t^2-3))` = 1); `from_int(10^18)`, `from_int(-5)`, `from_int(2^100)` | `[10^18, 0]`, `[2305843009213693946, 0]`, `[549755813888, 0]` (gp `10^18 % p`, `(-5) % p`, `2^100 % p`) | does not finish (10^18 additions) |
| `default_from_int_is_logarithmic` (`ff_poly`) | a test-only newtype over `PrimeField` that implements the trait by delegation but keeps the default `from_int`; p = 2^61-1, n = 10^18 and -5 | equals `PrimeField::new(n, p)` | does not finish |

### 3.5 Cost

Draws: `uniform_below` rejects a word with probability < m/2^64 below 2^64 and a round with probability < 1/2
above; same order of cost as HEAD. Prime check: one per public call.

---

## 4. Item 4: multiplicative orders from the factored group exponent

MAGMA: ch. 19 §19.7 `Order(a)` ("the multiplicative order k >= 1 of a if a is in (Z/mZ)^*, and zero if a
is not a unit", digest `Chapter_19_Integer_Residue_Class_Rings.md:220`), `IsPrimitive(n)` (`:218`);
ch. 21 §21.4.7 `Order(a)`, `FactoredOrder(a)` ("Factorisation of #F - 1, then order-from-factorisation",
digest `:252-253`), §21.4.4 `IsPrimitive(a)` (`:213`). (The task note says "ch. 22"; the ch. 22 digest is
Nearfields and lists only a nearfield `Order(x)`; the finite-field rows are ch. 21.)

### 4.1 Defects at HEAD (`$X/t3_order.log`, `$X/t3_order_slow.log`; references `$X/t3_ref.log`)

| Call | HEAD | gp |
|---|---|---|
| `IntegerMod(2, 1000003).multiplicative_order()` (`integer_mod.rs:220-254`, a loop to m) | 1.08 s | `znorder(Mod(2,1000003))` = 1000002 |
| `PrimeField(2, 1000003)`, inherent (`prime_field.rs:104-135`) and `Ring` hook (`:474-485`) | 1.05 s, 1.04 s | same |
| `IntegerMod(2, 10^9+7)` | 523 s | 500000003 |
| `FiniteField(2000000000123).from_int(3)` (`finite_field.rs:604-621`, `prime::factor` = trial division) | 0.22 s | 1000000000061 |
| `FiniteField(p16 = 20000000000002643)`, element 4 (p16 - 1 = 2·q, q prime) | 21.8 s (10^8 trial divisions) | 10000000000001321 |
| GF(p30), p30 = 2000000000000000000000000000543 = 2q + 1 | about 1.4·10^15 trial divisions: out of reach | `znorder(Mod(2,p30))` = 1000000000000000000000000000271 |

The trial division in `finite_field.rs` is `rustmath_integers::prime::factor` (`prime.rs:415-452`).

### 4.2 Algorithms

- **Order from a factored exponent** (H. Cohen, *A Course in Computational Algebraic Number Theory*,
  GTM 138, §1.4.1, Alg. 1.4.3 "Order of an Element"; Alg. 1.4.4, the primitive-root test, is the same
  loop): given x with x^h = 1 and h = ∏ p_i^{v_i}: e ← h; for each i: e ← e / p_i^{v_i}, g ← x^e,
  while g ≠ 1: g ← g^{p_i}, e ← e·p_i. Output e, and its factorization (the final exponent of each p_i).
  Invariant: x^e = 1 after each i, and after step i the p_i-adic valuation of e equals that of ord(x)
  (the least k with x^(e' p_i^k) = 1, e' the p_i-free part, is v_{p_i}(ord x) because e' is a multiple of
  the p_i-free part of ord x); later steps do not change it. Correct for any multiple h of the order.
- **Z/mZ**: the exponent of (Z/mZ)^* is the Carmichael function λ(m) = lcm over p^e ‖ m of λ(p^e), with
  λ(p^e) = p^(e-1)(p - 1) for odd p, λ(2) = 1, λ(4) = 2, λ(2^e) = 2^(e-2) for e >= 3 (R. D. Carmichael,
  *Note on a new number theory function*, Bull. AMS 16 (1910) 232-238; Ireland-Rosen GTM 84, Ch. 4 §1).
  Its factorization comes from those of m and of each p - 1 (lcm = maximum of exponents).
- **GF(p^n)**: q - 1 = ∏_{d | n} Φ_d(p), with Φ_d(p) = ∏_{e | d} (p^e - 1)^{μ(d/e)} computed by exact integer
  division; each Φ_d(p) is factored separately and the factorizations are merged by adding exponents (the
  Φ_d(p) can share small primes, e.g. 2 | p - 1 and 2 | p + 1). This is the algebraic factorization of the
  Cunningham tables (J. Brillhart, D. H. Lehmer, J. L. Selfridge, B. Tuckerman, S. S. Wagstaff Jr.,
  *Factorizations of b^n ± 1*, Contemporary Mathematics 22, AMS, 3rd ed. 2002). Example: 3^40 - 1 splits
  into pieces below 3^16, while the whole number has 64 bits.
- **A multiple of the order in Z/mZ without factoring m** (review 1, m5): Alg. 1.4.3 needs only some
  known multiple h of ord(a). If a^(m-1) = 1 (mod m), checked with one exponentiation, then ord(a) divides
  m - 1, so h = m - 1 works, and m is neither factored nor proven prime. This always holds for a prime m,
  and for a composite m exactly when a is a Fermat liar (2 mod 341: 2^340 = 1, ord 10). Otherwise
  h = λ(m), which needs the factorization of m (3 mod 341: 3^340 = 56, ord 30 = λ(341)). gp values in
  `$X/rev1/rev1_ref.log`.
- **±1 need no factoring** (review 1, m2): 1 has order 1, and -1 ≠ 1 has order 2, in every group. The
  shared `unit_order` (4.3) applies both before it asks for h, for every type of the crate, including the
  residue step of the Galois-ring order.
- **Factoring**: I2's `factorization_with_limits` (section 0.1) with the two efforts of 4.3. A
  factorization is used only when it is complete and every base is a proven prime (`primes` are proven by
  trial division, exactness below psi_12 or an ECPP certificate). Probable primes or unsplit composites
  mean "not available": Alg. 1.4.3 with a composite "prime" r would treat r as one prime and can return a
  multiple of the true order.

### 4.3 New module `rustmath-finitefields/src/group_order.rs` (crate-private)

File header: MAGMA ch. 19 §19.7 `Order`, ch. 21 §21.4.4 `IsPrimitive`, §21.4.7 `Order`, `FactoredOrder`;
ch. 48 (Galois rings use the residue field); [Coh93] Alg. 1.4.3; [Car10]; [BLSTW02]; I2's driver
(ch. 18 §18.10.1 `Factorization` with limits).

```rust
/// How hard the group exponent is factored and proven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Effort {
    /// The inherent methods. I2 `FactorizationLimits::default()` (MAGMA defaults: trial division to
    /// 10^4, 8191 rho steps, the sieve without a digit limit, `proof: true`). Complete unless an ECPP
    /// proof is unresolved (I1 saw none below 1536 bits). No time bound.
    Complete,
    /// The `Ring` hooks. I2 limits { trial_division_limit: CAPPED_TRIAL_DIVISION,
    /// pollard_rho_limit: CAPPED_RHO_STEPS, mpqs_limit: Some(CAPPED_SIEVE_DIGITS), proof: false };
    /// then each probable prime q of the result with q.bits() <= CAPPED_PROOF_BITS is proven with
    /// I1 `prove_primality`, and a larger one is not attempted. Cost bound of one factorization: the
    /// trial division, 8191 rho steps per composite, the sieve on composites of at most 30 digits,
    /// and one ECPP proof of at most 256 bits per prime factor (I1: 0.36 s mean, 0.52 s max, debug).
    Capped,
}

pub(crate) const CAPPED_TRIAL_DIVISION: u64 = 10_000;
pub(crate) const CAPPED_RHO_STEPS: u64 = 8191;
pub(crate) const CAPPED_SIEVE_DIGITS: u32 = 30;
pub(crate) const CAPPED_PROOF_BITS: u64 = 256;

/// Prime factorization, ascending, every base a proven prime. The empty vector is 1.
pub(crate) type Factored = Vec<(Integer, u32)>;

/// n >= 1 factored with every base a proven prime, or why not. The Err text names the unsplit
/// composites, the probable primes left unproven (Complete: ECPP unresolved; Capped: above 256 bits,
/// or unresolved), and a probable prime that ECPP showed composite (Capped only; that would be a BPSW
/// pseudoprime, and none is known). Proven results go into the cache, keyed by n.
pub(crate) fn factor_proven(n: &Integer, effort: Effort) -> std::result::Result<Arc<Factored>, String>;

/// The same computation without the cache (never looked up, never stored). `factor_proven` is a
/// cache lookup around it. The module's unit tests that assert a Capped failure call this one, so a
/// complete factorization cached by another test in the same binary cannot change their outcome.
fn factor_proven_uncached(n: &Integer, effort: Effort) -> std::result::Result<Factored, String>;

/// λ(m) factored (m >= 2), combined from factor_proven(m) and factor_proven(p - 1) for each p | m.
pub(crate) fn carmichael_lambda(m: &Integer, effort: Effort) -> std::result::Result<Factored, String>;

/// p^n - 1 factored through the cyclotomic values Φ_d(p), d | n, each through factor_proven.
pub(crate) fn field_group_order(p: &Integer, n: usize, effort: Effort) -> std::result::Result<Factored, String>;

/// Cohen Alg. 1.4.3. Precondition x^h = 1, checked (one exponentiation; a failure panics with
/// "group_order: x^h != 1", which the callers' exponents rule out). Returns the order, factored.
pub(crate) fn order_from_exponent<T>(x: &T, h: &[(Integer, u32)],
    pow: impl Fn(&T, &Integer) -> T, is_one: impl Fn(&T) -> bool) -> Factored;

/// The order of a unit x of a finite group, factored; every order in the crate goes through it.
/// Shortcuts without factoring: is_one(x) gives vec![] (order 1); else `minus_one` (the caller
/// says x = -1 and -1 != 1) gives vec![(2, 1)]. Otherwise exponent() supplies a factored multiple of
/// ord(x), and Alg. 1.4.3 runs. Err only from exponent().
pub(crate) fn unit_order<T>(x: &T, minus_one: bool,
    exponent: impl FnOnce() -> std::result::Result<Factored, String>,
    pow: impl Fn(&T, &Integer) -> T, is_one: impl Fn(&T) -> bool)
    -> std::result::Result<Factored, String>;

/// The order of the unit a mod m (m >= 2, gcd(a, m) = 1), factored, on integers with
/// `Integer::mod_pow`: unit_order with minus_one = (a = m - 1 and m > 2), and exponent
/// factor_proven(m - 1) when a^(m-1) = 1 (mod m), else carmichael_lambda(m).
pub(crate) fn unit_order_mod(a: &Integer, m: &Integer, effort: Effort) -> std::result::Result<Factored, String>;

pub(crate) fn product(f: &[(Integer, u32)]) -> Integer;
```

`factor_proven` with `Capped`: call I2, return `Err` if `composites` is nonempty; for each `(q, e)` of
`probable_primes`: `q.bits() > 256` gives `Err`; else `prove_primality(q)`: `Prime` moves `(q, e)` to the
primes, `NotPrime` or `Unresolved` give `Err`. The bases of I2's lists are pairwise coprime, so the merge is
a sort. With `Complete`: `Ok` iff `is_complete()`.

Cache (review 1, B2): `static CACHE: LazyLock<Mutex<HashMap<Integer, Arc<Factored>>>>`, keyed by the
integer n that `factor_proven` factored (m - 1, m, p - 1, Φ_d(p)). Only complete, proven factorizations are
stored, whichever effort found them, so all types share them (the four types of the capped test all factor
P43 - 1). Consequence: a hook that answers `Undecided` because its capped factorization failed can answer a
decided value later in the same process, once an exact method has factored the same integer. Both answers
are honest, since `Undecided` claims nothing; the hook docs and the core doc of `Undecided` say so. Tests
that assert a capped `Undecided` therefore live in their own test binary (4.6). At 4096 entries the map is
cleared (recomputing is always correct). No lock is held while factoring; a poisoned lock is recovered with
`into_inner()` (no code panics while holding it).

### 4.4 Public behaviour (signatures unchanged unless marked new)

| Method | New behaviour |
|---|---|
| `IntegerMod::multiplicative_order(&self) -> Option<Integer>` | sentinel rules unchanged; non-unit `None`; else `unit_order_mod(value, m, Complete)`. `# Panics`: only when a prime factor of the exponent used (m - 1 on the Fermat path; m and each p - 1 on the λ path) has no primality proof (I1 ECPP unresolved; the message names the number). `# Cost`: one exponentiation, then factoring m - 1 (Fermat path) or m and each p - 1 (λ path); no time bound |
| `IntegerMod` hook `multiplicative_order_status` (`:594`) | sentinel unchanged; non-unit `NotAUnit`; else `unit_order_mod(.., Capped)` (1 and -1 need no factoring): `Ok` maps through `order_from_integer`, `Err` gives `Undecided`. Docs: the cost bound of `Effort::Capped` (sieve on at most 30 digits, ECPP on at most 256 bits per factor), and "an `Undecided` can become decided later in the same process, once an exact call has factored the same exponent (the factorizations are cached); both answers are honest" |
| `PrimeField::multiplicative_order` (`:104`) | same as `IntegerMod`. For a prime modulus the Fermat test always holds, so m itself is neither factored nor proven prime (review 1, m5). The modulus may be composite (`PrimeField::new` does not check); m - 1 after a passed Fermat test, or λ(m), is a multiple of the order for every m |
| `PrimeField::is_generator` (`:141`) | `multiplicative_order() == Some(m - 1)` (false for every composite m, since ord(a) <= λ(m) < m - 1). `# Panics` as `multiplicative_order` |
| `PrimeField` hook (`:474`) | as `IntegerMod`'s (same docs) |
| `FiniteFieldElement::multiplicative_order` (`:604`) | through the new crate-private `fn order_factored(&self, effort: Effort) -> std::result::Result<Option<Factored>, String>` (`Ok(None)` for 0): `unit_order` with `minus_one` = (self + 1 = 0 and p ≠ 2), exponent `field_group_order(p, n, effort)`, pow `pow_int`. The inherent uses `Complete`. `# Panics`: an unresolved ECPP proof of a prime factor of q - 1 ("FiniteFieldElement::multiplicative_order: {why}"). `use rustmath_integers::prime::factor` (`:48`) goes |
| new `FiniteFieldElement::factored_multiplicative_order(&self) -> Option<Vec<(Integer, u32)>>` | MAGMA `FactoredOrder` (§21.4.7) through `order_factored(Complete)`: `None` for 0; `Some(vec![])` for 1 (MAGMA: the empty sequence; gp `factor(1)` = `matrix(0,2)`); else the factored order, ascending. `# Panics` as `multiplicative_order` (review 1, m8) |
| `FiniteFieldElement::is_primitive` (`:625`) | x ≠ 0 and x^((q-1)/r) ≠ 1 for every prime r dividing q - 1, from `field_group_order(p, n, Complete)`; no order is computed. In GF(2), 1 is primitive (q - 1 = 1 has no prime divisor), as at HEAD. `# Panics`: an unresolved ECPP proof of a prime factor of q - 1 ("FiniteFieldElement::is_primitive: {why}") (review 1, m8) |
| `FiniteFieldElement` hook (`:909`) | `order_factored(Capped)`: `Ok(None)` gives `NotAUnit`, `Ok(Some(f))` gives `order_from_integer(&product(&f))`, `Err` gives `Undecided`. Same docs as the `IntegerMod` hook |
| `GaloisRingElement::multiplicative_order` (`galois_ring.rs:1115`) and hook (`:1074`) | `order_by_residue_lifting(effort)` takes an `Effort`. Its residue step (`:1153`) calls `self.residue().order_factored(effort)`, so the ±1 shortcuts of `unit_order` apply to the residue (review 1, m2): -1 has residue order 2, and (-1)^2 = 1 gives order 2; an element with residue 1 has residue order 1 and a p-power order; neither factors anything. A residue-order `Err` gives a new variant `UnitOrder::Undecided(String)`: the hook (`Capped`) maps it to `MultiplicativeOrder::Undecided`, and the inherent (`Complete`) panics with the reason (an unresolved ECPP proof; its `# Panics` says so) |
| `ExtensionField` hook (`extension_field.rs:242`) | delegates (unchanged code) |

Core doc of `MultiplicativeOrder::Undecided` (`rustmath-core/src/traits.rs:208`, review 1, m5), replacing
"This element type cannot decide.":

```rust
    /// Not decided. Either the element type cannot decide, or deciding needs work beyond the hook's
    /// cost cap (the finite-ring types of rustmath-finitefields: a factorization of the group
    /// exponent with proven prime factors; their exact inherent `multiplicative_order()` decides).
    /// A capped `Undecided` can become a decided answer later in the same process, when cached
    /// work makes it cheap; both answers are honest.
    Undecided,
```

Honesty: hooks return a decided value only from a complete, proven factorization, or from the ±1 and
p-power steps that need none; `Undecided` otherwise. The inherent methods never guess; they panic only on an
unresolved primality proof (as I1's `is_prime`).

### 4.5 Callers (grep `multiplicative_order\|is_generator\|is_primitive\|multiplicative_order_status`)

In-crate only: `galois_ring.rs:1153` (residue order), `embedding.rs:261` (test), `extension_field.rs:243`,
tests `ring_predicates.rs` (31 sites), `galois_ring_sentinel.rs` (5), `finite_field.rs` and `prime_field.rs`
and `integer_mod.rs` unit tests. Other crates' `is_primitive`/`is_generator`/`multiplicative_order` hits
(quadratic forms, Dirichlet characters, Lie algebras, `rustmath-category` on i32/Integer) are different
methods. Values are unchanged everywhere (the prototype matches the HEAD loop on every unit mod m < 400,
48517 units, and the HEAD field order on all 690 nonzero elements of GF(2^6), GF(3^4), GF(5^3), GF(7^2),
GF(2^8), GF(11^2)). The prototype used λ(m) only; the Fermat path of revision 1 changes only which multiple
h of the order Alg. 1.4.3 starts from, and the algorithm returns ord(a) for every multiple (4.2), so the
values stay the same; `hooks_agree_with_inherent_small` checks this against brute force on every unit mod
m < 200, Fermat liars included. The `Undecided` assertions of `ring_predicates.rs` are on unbound sentinels
(`:164`, `:170`, `:202`, `:204`, `:260`, `:265`, answered by `sentinel_order` before any factoring) or on a
test wrapper that keeps the trait defaults (`:550-554`, `GrDefaults`), so the cache cannot change them. The comment in `ring_predicates.rs:236` ("so the
trial-division factoriser is instant") is updated.

### 4.6 Tests

Two new test files (separate binaries, so separate processes and separate caches; review 1, B2).

**`tests/unit_group_orders.rs`** (header citing ch. 19 §19.7, ch. 21 §21.4.4, §21.4.7, Cohen Alg. 1.4.3).
Nothing in this file asserts an `Undecided`, so the shared cache cannot change an expected value.

| Name | Input | Expected (gp) | HEAD |
|---|---|---|---|
| `integer_mod_orders_match_pari` | (2, 1000003); (2, 10^9+7); (5, 10^9+7); (3, 2^66); (2, 3^40·(10^9+7)); (5, 2^20·3^5·1000003); (2, 15); (7, 15); (2, 341) (Fermat path: 2^340 = 1 mod 341); (3, 341) (λ path: 3^340 = 56); (2, 561) (Carmichael number, Fermat path); (5, 561) | 1000002; 500000003; 1000000006; 2^64; 4052555177334307185113857602; 3538951077888; 4; 4; 10; 30; 40; 80 (`znorder(Mod(a,m))`, `$X/t3_ref.log`, `$X/t10_ref.log`, `$X/rev1/rev1_ref.log`, `$X/rev1/rev1_ref2.log`) | passes only after 523+ s (10^9+7 rows) |
| `prime_field_orders_match_pari` | `PrimeField` (3, 2000000000123); (4, 20000000000002643); (2, 15); (2, 341); (3, 341) | 1000000000061; 10000000000001321; 4; 10; 30 | slow (O(m)) |
| `finite_field_orders_match_pari` | GF(p16) element 4; GF(p30) element 2; default GF(5^30) (assert the modulus `t^30+t^3+t+3` first) w, w+1; GF(3^40) (`t^40+t+2`) w; GF(2^127) (`t^127+t+1`) w; GF(101^12) (`t^12+t+11`) w | 10000000000001321; 1000000000000000000000000000271; 133046082087925502232, 14782898009769500248; 12157665459056928800; 2^127 - 1; 1126825030131969720661200 (`fforder(ffgen(Mod(1,p)*pol,'t))`, `$X/p2_ref.log`) | GF(p16) 21.8 s; GF(p30) out of reach |
| `factored_order_matches_pari` | GF(5^30) w+1; GF(125) (Conway `t^3+3t+3`) w+1; one of GF(125); zero | `[(2,3),(11,1),(31,1),(61,1),(71,1),(181,1),(521,1),(1741,1),(7621,1)]`; `[(31,1)]`; `Some(vec![])` (MAGMA `FactoredOrder(1)` = []; gp `factor(1)` = `matrix(0,2)`); `None` (`factor(fforder(..))`, `$X/t10_ref.log`, `$X/rev1/rev1_ref.log`) | new API |
| `is_primitive_and_is_generator` | GF(125) w, w+1; `PrimeField(2, P43)`, `PrimeField(3, P43)` (P43 below) | true, false (gp `fforder` 124 and 31); true, false (`znorder(Mod(2,P43)) == P43-1`, `znorder(Mod(3,P43)) == (P43-1)/2`, `$X/rev1/rev1_ref.log`). The P43 half runs the sieve once on the 41-digit cofactor of P43 - 1 (I2 measured about 0.4 s at 39 digits in debug); if it takes more than 10 s in debug, mark it `#[ignore]` with the gp values in a comment | slow / out of reach for P43 |
| `hooks_agree_with_inherent_small` | every unit of Z/mZ, m < 200 (`IntegerMod` and `PrimeField`; includes Fermat liars such as 3 mod 91), every nonzero element of GF(3^4) and GR(27, 2) | hook == `order_from_integer(inherent)`; inherent == brute force (first k with x^k = 1, computed in the test) | passes (slowly) |

**`tests/unit_group_orders_capped.rs`** holds exactly one test, on purpose. Its header says why: the
factorization cache is process-wide, and a second test that factors P43 - 1 exactly could run first and turn
phase 1's `Undecided` into a decided answer. P43 = 2580000000000000001462000000000000000177763 (gp
`$X/t7_ref.gp`, `$X/rev1/rev1_ref.log`: `isprime` = 1; P43 - 1 = 2·43·C41 with C41 =
100000000000000000039·300000000000000000053, a 41-digit composite: 8191 rho steps do not split it and it is
above the 30-digit sieve limit).

| Name | Input | Expected (gp) | HEAD |
|---|---|---|---|
| `capped_hooks_then_exact_orders_p43`, phase 1 (no exact call on P43 - 1 yet) | hooks of 2, -1, 1, 0 in `PrimeField::new(.., P43)`, `IntegerMod::new(.., P43)`, `FiniteField::new(P43, 1)`, `GaloisRing::new(P43, 2, 1)` (= Z/P43^2); in the Galois ring also `1 + P43`, `2·P43 - 1` and P43 | 2: `Undecided` (all four); -1: `Finite(2)`; 1: `Finite(1)`; 0 and P43: `NotAUnit`; `1 + P43`: `FiniteExceedsU64`, inherent P43 (`znorder(Mod(1+P43,P43^2))`); `2·P43 - 1`: `FiniteExceedsU64`, inherent 2·P43 (`znorder(Mod(2*P43-1,P43^2))`, `$X/rev1/rev1_ref.log`). The last two need no factoring (residue 1 and -1) | O(P43) loop; trial division of P43 - 1 |
| same test, phase 2 | inherent `PrimeField(2, P43).multiplicative_order()` and `PrimeField(3, P43)`; then the hooks of 2 again in the four types, and their inherent orders | P43 - 1 and (P43 - 1)/2; hooks now `FiniteExceedsU64` (the exact call cached the factorization of P43 - 1); inherent orders of 2: P43 - 1 in `PrimeField`, `IntegerMod`, GF(P43), and P43·(P43 - 1) in Z/P43^2 (gp `znorder(Mod(2,P43^2)) == P43*(P43-1)` = 1, `$X/rev1/rev1_ref2.log`) | out of reach |

Existing tests that stay: `ring_predicates.rs::gf1013_order_beyond_1000`, `order_beyond_u64` (p = 12·2^64+1,
`znorder(Mod(2,p))` = 110680464442257309696), `gr_*_orders_*`, `galois_ring_sentinel.rs::unbound_ring_hooks`,
`finite_field.rs::multiplicative_order_and_primitive`, `prime_field.rs::test_multiplicative_order`,
`test_generator`, `integer_mod.rs::test_multiplicative_order`.

### 4.7 Cost (prototype `$X/p2_order.rs` on HEAD's `factor_pollard_rho`, debug)

| Case | HEAD | New |
|---|---|---|
| 2 mod 1000003 | 1.08 s | 3 ms |
| 2 mod 10^9+7 | 523 s | 4 ms |
| GF(p16), 4 | 21.8 s | 11 ms |
| GF(p30), 2 | out of reach | 39 ms |
| 2 mod p16·p12 | out of reach (an O(10^28) loop) | 70 s with rho on the 13-digit factor; I2's sieve on this 29-digit modulus replaces that rho run |
| GF(3^40), GF(2^64), GF(2^127), GF(5^30), GF(101^12): w and w+1 | 0.29 s, 0.54 s, n/a, n/a, n/a | 0.3 to 1.9 s (dominated by `pow_int` in GF(p^n)) |

The Fermat path adds one exponentiation mod m to every order in Z/mZ and removes the factorization (for a
prime m, the primality proof) of m. The prototype rows above for prime m factored m - 1 already, so their
cost does not change.

Cost bound of one capped hook call (review 1, m5), from the limits of `Effort::Capped` and the measurements
of I1 and I2 (debug): one exponentiation (the Fermat test, Z/mZ only), trial division to 10^4, 8191 Brent
rho steps per composite piece, the sieve only on composites of at most 30 digits (I2: about 0.4 s at 39
digits, so less below 30), and at most one ECPP proof of at most 256 bits per probable prime factor (I1:
0.36 s mean, 0.52 s max at 256 bits). A hook on a group order with a 300-bit prime factor, or with a
40-digit hard composite part, answers `Undecided` within that bound. The `Complete` effort of the inherent
methods has no bound (the sieve has no digit limit; ECPP runs on every probable prime).

Memory: the factorizations; the cache holds at most 4096 of them.

---

## 5. Item 5: MAGMA `IsSquare` / `Sqrt` with the root on `PrimeField` and GF(p^n)

MAGMA ch. 21 §21.4.4 `IsSquare(a)`: "Return true and a square root b (with b^2 = a) if a is a square in F,
otherwise false" (digest `:217`); §21.4.7 `SquareRoot(a)`/`Sqrt(a)`: "an element y with y^2 = a. Errors if
a is not a square", "Tonelli-Shanks or Cipolla's algorithm" (`:254`). Ch. 19 §19.5.6 is already served by
`IntegerMod::sqrt`.

### 5.1 Defect at HEAD

`PrimeField` has `is_quadratic_residue(&self) -> bool` (`prime_field.rs:186`) and no root; GF(p^n)
(`FiniteFieldElement`) has neither (`grep -n "fn sqrt\|fn is_square" src/` finds only `integer_mod.rs`).
Tracker rows `ch021.IsSquare` and `ch021.SquareRoot` are "partial" for this reason ("no sqrt in GF(p^n),
n>1"; "a bool and a root come from separate calls").

### 5.2 Algorithm (finite_field.rs; prototype `$X/p1_sqrt.rs`)

For a bound element a of GF(q), q = p^n:
1. a = 0: root 0.
2. p = 2: the Frobenius is a bijection; the root is a^(q/2) (its inverse, a^(2^(n-1))).
3. Odd q: Euler's criterion decides: a is a square iff a^((q-1)/2) = 1 (GF(q)^* is cyclic of even order).
   Non-square: `None`.
4. Root: q ≡ 3 (mod 4): a^((q+1)/4). Otherwise Tonelli-Shanks in the cyclic group GF(q)^* of order
   q - 1 = 2^e t, t odd (Cohen GTM 138, Alg. 1.5.1, stated for Z/pZ; only cyclicity and the non-square
   are used). The non-square z: n >= 2: z = w + c for c = 0, 1, ..., p - 1 (w the generator), then every
   element in index order; n = 1: z = c for c = 2, 3, ..., then every element. The scan ends because
   GF(q)^* has (q - 1)/2 non-squares; it only finds a helper, never decides anything. Measured: the first
   candidate w is a non-square in every Conway field tested and in GF(41^2), GF(97^2) (default non-Conway
   moduli); GF((2^80+13)^2) with `t^2+t+1` needs 3 candidates.
5. Canonical root: of r and -r, the one whose coefficient vector read as the integer Σ c_i p^i is smaller.
   For n = 1 this is the least root in [0, p), the same choice as `IntegerMod::sqrt` and
   `square_root_mod_prime_power(a, p, 1)`.
6. Always-on check r^2 = a before returning (one multiplication).

### 5.3 API

```rust
impl FiniteFieldElement {
    /// MAGMA `IsSquare(a)` (ch. 21 §21.4.4). Euler's criterion; every element when p = 2.
    /// Panics for an unbound element (needs the parent field, see the type docs).
    pub fn is_square(&self) -> bool;
    /// MAGMA `Sqrt(a)` / `SquareRoot(a)` (ch. 21 §21.4.7) and the root of `IsSquare`: `Some(r)` with
    /// r^2 = self, the canonical root of step 5; `None` iff self is not a square (a proof).
    /// Panics for an unbound element.
    pub fn sqrt(&self) -> Option<FiniteFieldElement>;
}

impl PrimeField {
    /// MAGMA `IsSquare(a)`: the same contract and panics as `is_quadratic_residue` (an alias).
    pub fn is_square(&self) -> bool;
    /// MAGMA `Sqrt(a)` and the root of `IsSquare(a)`: the least root in [0, p), or `None` (a proof).
    /// Modulus-0 sentinel: the non-negative root of a perfect square in Z, as a sentinel, else None.
    /// A zero element gives Some(0) for every modulus (as `is_quadratic_residue`).
    /// Otherwise: panics "PrimeField::sqrt: the modulus {m} is not prime" when I1's
    /// `is_probable_prime(m)` is false, then `value.try_sqrt_mod_prime(m)`: Ok(Some(r)) gives
    /// min(r, m - r); Ok(None) gives None; Err(e) panics "PrimeField::sqrt: the modulus {m} is not
    /// prime ({e})".
    pub fn sqrt(&self) -> Option<PrimeField>;
}

impl ExtensionField {
    pub fn is_square(&self) -> bool;          // forwards
    pub fn sqrt(&self) -> Option<Self>;       // forwards, wraps with from_elem
}
```

Correctness of `PrimeField::sqrt` beyond the prime check: `try_sqrt_mod_prime` squares its root back, and a
`None` comes only from a Jacobi symbol -1, which proves non-squareness modulo any odd m; so even a BPSW
pseudoprime modulus (none known) cannot produce a wrong answer, only a panic. `is_square` and `sqrt` agree
(`is_quadratic_residue_prime` decides by the same two certificates).

### 5.4 Callers

New methods; no existing caller. `PrimeField::is_quadratic_residue` keeps its name and behaviour.
The REPL (`rustmath-jupyter`) uses `IntegerMod`/`square_root_mod_prime`; unchanged here.

### 5.5 Tests (new `tests/field_sqrt.rs`, header citing ch. 21 §21.4.4, §21.4.7, Cohen Alg. 1.5.1)

| Name | Input | Expected (gp `$X/t6_ref.gp`, `lr(a,p)`, `show(a,p)` with `canon`) | HEAD |
|---|---|---|---|
| `prime_field_sqrt_matches_pari` | (a, p): (2, 10^9+7), (3, 10^9+7), (7, 998244353), (17, 998244353), (5, 998244353), (2, 2^61-1), (3, 2^61-1), (11, 2^80+13), (2, 2^80+13), (1, 2), (0, 7), (6, 7) | 59713600, 82062379, 116190042, 473844410, None, 2147483648, None, 87776246428256997082236, None, 1, 0, None; `is_square` agrees | new API |
| `prime_field_sqrt_sentinel_and_composite` | sentinel 4, sentinel 2; `PrimeField(4, 15).sqrt()` | `Some(sentinel 2)`, `None`; panic "PrimeField::sqrt: the modulus 15 is not prime" | new API |
| `gf9_sqrt_matches_pari` | GF(9) (Conway `t^2+2t+2`), all 9 elements | squares and roots: 0 → 0, 1 → 1, 2 → 1+w, 1+w → w, 2+2w → 2+w; w, 2+w, 2w, 1+2w non-squares | new API |
| `gf125_and_gf256_sqrt_match_pari` | GF(125) (`t^3+3t+3`): w, 1+w, 3+2w^2, w^2, 3, 2; GF(256) (`t^8+t^4+t^3+t^2+1`): w, 1+w | non-sq; 2+3w+w^2; 2+w+w^2; w; non-sq; non-sq; sqrt(w) = 1+w^2+w^7, sqrt(w+1) = w^2+w^7 | new API |
| `large_field_sqrt_matches_pari` | GF((2^61-1)^2) = F_p[t]/(t^2-3): w, 1+w, 2+w, 5+3w, 3+7w, 3, 9876543210+12345678901234567w; GF((2^80+13)^2) = F_p[t]/(t^2+t+1): w, 1+w, 2+w, 5+3w, 7, 2 | the roots in `$X/t6_ref.log` (e.g. sqrt(w) = 691116165103479690 + 369169261380160233w; sqrt(3) = w; sqrt(2+w) = 1073741824 + 1073741824w; over 2^80+13: sqrt(w) = 1 + w, sqrt(1+w) = 453758120640033424617241w) | new API |
| `finite_field_sqrt_exhaustive_small_fields` | GF(p^n) for (3,1), (5,1), (7,1), (13,1), (17,1), (2,1..4), (3,2..5), (5,2), (5,3), (7,2), (11,2), (13,2), (17,2), (41,2), (97,2) | `is_square` = membership in {x^2}; `sqrt(a)^2 = a`; the count of squares is (q+1)/2 for odd q and q for even q (`$X/p1_sqrt.log`) | new API |
| `finite_field_sqrt_unbound_panics` | `<FiniteFieldElement as Ring>::one().sqrt()` | panic containing "needs a parent field" | new API |
| `extension_field_sqrt_forwards` | `ExtensionField` over GF(9) with the Conway modulus `t^2+2t+2`, element 2 | root 1 + w | new API |

### 5.6 Cost

One exponentiation by (q - 1)/2 for the decision; Tonelli-Shanks O(e^2 + log q) multiplications
(e = v_2(q - 1): 23 for 998244353); the non-square scan costs one Euler test per candidate. GF((2^80+13)^2):
39 roots plus 39 decisions in 0.74 s (`$X/p1_sqrt.log`).

---

## 6. Item 6: `square_root_mod_prime` runs `is_prime` twice

Taken by I1 (design `I1-integers-primality.md` §1.4, §2.7): `square_root_mod_prime` keeps `assert_prime`
(one test) and calls `Integer::try_sqrt_mod_prime`, which tests nothing. F1's work:
- At the start of implementation, check that `integer_mod.rs` `square_root_mod_prime` has exactly one
  primality test on its path (`assert_prime`, then `try_sqrt_mod_prime`), and that I1's test
  `square_root_mod_prime_refuses_strong_pseudoprime` (in `tests/prime_residues.rs`) exists and passes.
- If I1 did not land it (review 1, m6: I1's snippet needs I1's `try_sqrt_mod_prime`, so it cannot apply
  alone), use this fallback on HEAD's API. It drops `assert_prime` and lets `Integer::sqrt_mod_prime` run
  the one primality test (`integer.rs:883`, before any other work); that function reduces a, returns a for
  p = 2 and decides by the Jacobi symbol inside `sqrt_mod_prime_checked` (`:1154-1170`), so the comment
  about (3/2) = -1 at `integer_mod.rs:653-654` still holds:

  ```rust
  pub fn square_root_mod_prime(a: &Integer, p: &Integer) -> Option<Integer> {
      // The one primality test runs inside Integer::sqrt_mod_prime (integer.rs:883).
      match a.sqrt_mod_prime(p) {
          Ok(r) => Some(r),
          // `sqrt_mod_prime` maps the Jacobi symbol -1 (a proof) to this exact message (:890-892).
          Err(MathError::InvalidArgument(m)) if m == "Not a quadratic residue" => None,
          // Its primality refusal (:883-887); same panic text as the HEAD `assert_prime`.
          Err(MathError::InvalidArgument(m)) if m == "Modulus must be prime" => {
              panic!("square_root_mod_prime: p = {p} is not prime")
          }
          // p passed the test, so any other error proves p composite (as at HEAD :662-664).
          Err(e) => panic!("square_root_mod_prime: p = {p} is not prime ({e})"),
      }
  }
  ```

  The two message strings are constants of `Integer::sqrt_mod_prime`; a comment there names this caller.
  Tests that pin the fallback: the existing should-panic test "square_root_mod_prime: p = 15 is not prime"
  (also for p = 0 and 1: `is_prime` refuses them before any division), the doc example (4 mod 7, 3 mod 7),
  and a new `square_root_mod_prime_mod_two` ((3, 2) gives `Some(1)`, gp `sqrt(Mod(3,2))` = `Mod(1,2)`).
- `PrimeField::sqrt` (item 5) uses the same primitive and runs one cheap BPSW test, never two proofs.

---

## 7. Item 7: containers bind unbound constants

Commit `029f7ab` gave `GaloisRingElement` the sentinel design of `FiniteFieldElement` and the modulus-0
`IntegerMod`/`PrimeField`: the static `Ring::zero()`/`one()` are integer constants c awaiting the canonical
map Z → R (MAGMA ch. 48 §48.2.3: `R ! a` is "a mod p^a"), bound on contact with a bound element.

### 7.1 Defect at HEAD

`UnivariatePolynomial` combines coefficients position by position, so a coefficient that only ever meets
unbound constants stays an integer in Z. Over GR(8, 2), `q = (x+1)^8 - x^8 + w` from `var()` and
`R::one()` keeps `[1 + w, 8, 28, 56, 70, 56, 28, 8]` (degree 7); the same polynomial from bound elements is
`(1 + w) + 4x^2 + 6x^4 + 4x^6` (degree 6). Pinned by
`tests/galois_ring_sentinel.rs:191-245` (`polynomials_from_static_constructors_keep_unbound_coefficients`)
and documented at `galois_ring.rs:370-381`. The same holds for the other sentinel types:

| Built from `var()`, `R::one()` | HEAD | gp (`$X/t10_ref.gp`, `$X/t11_ref.gp`) |
|---|---|---|
| GF(3): `(x+1)^3 - x^3 + 2` | `[0 (bound), 3, 3]`, degree 2 | `lift(Mod(1,3)*((x+1)^3 - x^3 + 2))` = 0 |
| Z/8: `(x+1)^8 - x^8 + 1` | degree 7 | `4*x^6 + 6*x^4 + 4*x^2 + 2` |
| GF(4): `(x+1)^2 - x^2 + w` | `[1 + w, 2]`, degree 1 | `w^0*((x+1)^2 - x^2) + w` = `t + 1` (degree 0) |

The sparse containers of the same crate have the same defect (`$X/rev1/rev1_cont.log`, review 1, B3), and
their derived `PartialEq` compares keys, so it cannot see that an unbound term is 0 in the ring:

| Built from `variable(0)`/`var()`/`monomial(R::one(), -1)` and `R::one()` | HEAD | gp (`$X/rev1/rev1_ref.log`) |
|---|---|---|
| `MultivariatePolynomial`, GR(8, 2): `(x_0+1)^8 - x_0^8 + w` (either order of the last addition) | 8 terms, degree 7: constant `1 + w` bound, the terms 8, 28, 56, 70, 56, 28, 8 unbound | `lift(Mod(1,8)*((x+1)^8-x^8))` = `4*x^6 + 6*x^4 + 4*x^2 + 1`, so `(1+w) + 4x^2 + 6x^4 + 4x^6`, 4 terms |
| `MultivariatePolynomial`, GF(3): `(x+y)^3 - x^3 - y^3 + 2xy` | 3 terms: `2xy` bound, `3x^2y` and `3xy^2` unbound | `lift(Mod(1,3)*((x+y)^3 - x^3 - y^3 + 2*x*y))` = `2*y*x` |
| `LaurentPolynomial`, GR(8, 2): `(q+1)^8 - q^8 + w`; `(q^-1+1)^8 - q^-8 + w` | 8 terms each (max exponent 7; min exponent -7) | 4 terms: `(1+w) + 4q^2 + 6q^4 + 4q^6`; `(1+w) + 4q^-2 + 6q^-4 + 4q^-6` |
| `{1: u1, x_0: u8} == constant(r.one())` (multivariate); `{0: u1, 1: u8} == constant(r.one())` (Laurent) | `false`, `false` | equal in GR(8, 2)[x] (8 = 0) |

### 7.2 Design

Binding is canonical: Z → R is the unique ring homomorphism, so replacing an unbound c by its image in the
ring of a bound coefficient does not change the element of R[x] the polynomial stands for (the existing test
already shows `q.scalar_mul(&r.one()) == qb`).

**rustmath-core `Ring` (traits.rs, after `static_characteristic` and P1's `static_is_commutative`)**:

```rust
    // --- Unbound integer constants (RustMath; MAGMA has none) -----------------------------------
    // Parent-carrying element types of rustmath-finitefields (IntegerMod, PrimeField with modulus 0,
    // FiniteFieldElement, GaloisRingElement, ExtensionField) return from the static zero()/one() an
    // integer constant c awaiting the canonical map Z -> R. Containers bind such constants to the
    // parent of a bound sibling with these hooks.

    /// Whether values of this type can be unbound constants. Default false; containers skip all
    /// binding work for such types (a compile-time constant). A container over R sets it to
    /// R::HAS_UNBOUND_CONSTANTS.
    const HAS_UNBOUND_CONSTANTS: bool = false;

    /// Whether self is an unbound constant: a value without a parent that stands for the image of
    /// an integer under the canonical map Z -> R (the static zero()/one() of a parent-carrying type,
    /// and everything computed from such values alone). For a container over such a type (a
    /// polynomial in R[x] or R[x, x^-1]) it means that no coefficient is bound: the value is the
    /// image of an element of Z[x] (or Z[x, x^-1]); a zero container with no stored coefficient
    /// counts as unbound, since it is 0 in every ring. Default false.
    fn is_unbound_constant(&self) -> bool { false }

    /// The image of self under the canonical map into the parent of `model`. Contract: for a type
    /// with HAS_UNBOUND_CONSTANTS, this is Some(..) exactly when self.is_unbound_constant() and
    /// !model.is_unbound_constant(); None otherwise, and always None for types without unbound
    /// constants. A container maps coefficient by coefficient (to the parent of one of model's bound
    /// coefficients) and drops the coefficients that become zero.
    fn bind_to_parent_of(&self, model: &Self) -> Option<Self> { let _ = model; None }
```

**rustmath-finitefields impls** (each `const HAS_UNBOUND_CONSTANTS: bool = true;`):

| Type | `is_unbound_constant` | `bind_to_parent_of(model)` (only when self unbound and model bound) |
|---|---|---|
| `IntegerMod` | `modulus == 0` | `Some(IntegerMod::make(value, model.modulus))` (`integer_mod.rs:107`, reduces) |
| `PrimeField` | `modulus == 0` | same with `PrimeField::make` (`prime_field.rs:69`) |
| `FiniteFieldElement` | `field.is_none()` | `Some(self.bind_into(model.field))` (`finite_field.rs:469`) |
| `GaloisRingElement` | `ring.is_none()` | `Some(ring.from_int(c))` |
| `ExtensionField` | `!elem.is_bound()` | `elem.bind_to_parent_of(&model.elem).map(from_elem)` |

Each satisfies the contract: a bound value of these types always has a parent to bind into.

**rustmath-polynomials `UnivariatePolynomial` (univariate.rs)**. A private helper
`fn bound_model(coeffs: &[R]) -> Option<&R>` returns the first coefficient with `!is_unbound_constant()`
(`None` at once when `!R::HAS_UNBOUND_CONSTANTS`). It scans instead of reading `coeffs[0]`, because a
coefficient type may have a zero that carries no parent: a zero `LaurentPolynomial` is an empty map, and it
stays unbound after binding (review 1, B3).
1. `new` (`:52`): before trimming, if `R::HAS_UNBOUND_CONSTANTS` and `bound_model` finds a model, clone it
   and replace every unbound c by `c.bind_to_parent_of(&model)` (`Some` by the contract; on `None` keep c).
   Invariant after `new` (U): if some coefficient is bound, every coefficient has been through the binding;
   for scalar coefficient types (and for `UnivariatePolynomial` or `LaurentPolynomial` over them) no
   coefficient is unbound then, except zero containers, which are 0 in every ring. Then trim (`is_zero` now
   sees the bound value: `unbound(8)` becomes 0 in GR(8, 2)). `new` is the only constructor (`:62` is the
   only struct literal, and no method takes `&mut self` on `coeffs`), so every operation keeps U.
2. `coeff(i)` past the degree (`:99-104`): the lazily created zero is `R::zero()` bound to the model when
   there is one, so `q.coeff(9).is_bound()` for a bound polynomial.
3. `PartialEq` (`:42-46`): equal lengths compare coefficient by coefficient, as now. The coefficient `==`
   already coerces, so this gives the same answer as binding first: if binding would shorten the unbound
   side, its top coefficient binds to 0 while the bound side's top coefficient is nonzero, and that pair
   already compares unequal. Different lengths, when `R::HAS_UNBOUND_CONSTANTS`: if one
   side has a bound model and the other has none, rebuild the other side through `new` with every
   coefficient bound to that model, and compare. Otherwise (both bound, or both unbound) different lengths
   mean different degrees: unequal. This makes `P::new([unbound(1), unbound(8)]) == P::constant(r.one())`
   over GR(8, 2), while `P::new([unbound(1), unbound(8)]) != P::constant(unbound(1))` (in Z[x],
   1 + 8x ≠ 1).
4. `impl Ring for UnivariatePolynomial<R>` (`:857`): `HAS_UNBOUND_CONSTANTS = R::HAS_UNBOUND_CONSTANTS`;
   `is_unbound_constant` = `R::HAS_UNBOUND_CONSTANTS && coeffs.iter().all(|c| c.is_unbound_constant())`;
   `bind_to_parent_of(model)`: when self is unbound and `bound_model(&model.coeffs)` is `Some(m)`, `new` of
   the coefficients bound to m; else `None`. So R[x][y] binds too.

**rustmath-polynomials `MultivariatePolynomial` (multivariate.rs) and `LaurentPolynomial` (laurent.rs)**
(review 1, B3). Both store a `BTreeMap` of nonzero coefficients (`multivariate.rs:291`, `laurent.rs:18`).
The field is private; it is written only by `zero`/`new` (empty), `constant`, `variable`/`var`/`monomial`
(one term), and `add_term` (`multivariate.rs:324`, `laurent.rs:52`). Every other operation (`Add`, `Sub`,
`Mul`, `Neg`, `scalar_mul`, `monomial_mul`, `relabel`, `substitute`, `divide`, `divide_multiple`,
`partial_derivative`) builds its result through `add_term`. So the rule goes into `add_term`, and the
invariant holds for every value:

Invariant (M): every stored coefficient is bound, or none is.

Private helpers in each file:
- `fn bound_model(&self) -> Option<&R>`: `None` when `!R::HAS_UNBOUND_CONSTANTS`; else the first stored
  coefficient if it is bound (by M the first one decides; stored coefficients are nonzero, so never an
  empty container).
- `fn bind_all(&mut self, model: &R)`: every stored c becomes `c.bind_to_parent_of(model)` (`Some` by the
  contract, since by M all stored are unbound when this is called); entries whose new value `is_zero()` are
  removed (rebuild from `std::mem::take(&mut self.terms)`).

`add_term(key, c)`, when `R::HAS_UNBOUND_CONSTANTS` and the map is nonempty, before the HEAD body:
- c unbound and the map bound: `c = c.bind_to_parent_of(model)`; return if it is zero;
- c bound and the map unbound: `self.bind_all(&c)` (O(n), at most once per value: a bound map stays bound,
  because bound + bound and unbound 0 + bound are bound).

Then the HEAD body: `or_insert_with(R::zero)` gives the unbound 0, and 0 + c has the state of c, so M holds
after the insertion; the entry is removed when it becomes zero. Proof of M by cases: an empty map takes one
term; a bound map only receives bound values; an unbound map receiving a bound value is bound entirely
first; an unbound map receiving an unbound value stays unbound (unbound + unbound is unbound).

`coefficient(&m)` (`multivariate.rs:341`) and `coeff(power)` (`laurent.rs:66`): a missing key gives
`R::zero()` bound to `bound_model()` when there is one.

`PartialEq`: the derives (`multivariate.rs:287`, `laurent.rs:14`) lose `PartialEq`, and a manual impl
replaces it: when `!R::HAS_UNBOUND_CONSTANTS`, `self.terms == other.terms` (the derived behaviour). Otherwise,
if exactly one side has a bound model and the other side is nonempty, clone the other side, `bind_all` it
with that model, and compare the maps; else compare the maps. `LaurentPolynomial` keeps `#[derive(Eq)]`
(it generates `impl<R: Ring + Eq> Eq`, as now). Neither type derives `Hash`, so no `Hash` impl must agree.
Stated consequence: a sparse zero carries no parent, so an all-unbound map that would become zero in some
ring is not equal to the empty map (it is a nonzero element of Z[x]), unlike the univariate `[b0]`, which
carries its ring. This is the representation, not a new defect, and the type docs say so.

`impl Ring for LaurentPolynomial<R>` (`laurent.rs:195`, next to P1's hooks): `HAS_UNBOUND_CONSTANTS =
R::HAS_UNBOUND_CONSTANTS`; `is_unbound_constant` = `R::HAS_UNBOUND_CONSTANTS && self.bound_model().is_none()`
(the empty map counts as unbound); `bind_to_parent_of(model)`: when self is unbound and
`model.bound_model()` is `Some(m)`, a clone with `bind_all(m)`; else `None`. `MultivariatePolynomial`
implements no `Ring` (grep `Ring for MultivariatePolynomial` finds nothing), so it needs no hooks.

Prototype (`$X/rev1/rev1_cont.rs`): the rule on a local `BTreeMap` container over HEAD's sentinel types,
with `bind_to_parent_of` emulated as `c + (m - m)`. Over GR(8, 2), `(x_0+1)^8 - x_0^8 + w` from
`variable(0)` and `R::one()` gives the 4 terms `(1+w) + 4x^2 + 6x^4 + 4x^6` (HEAD: 8 terms, degree 7);
adding w first gives the same map; it equals the bound construction; the all-unbound part keeps its 8 terms;
multiplying it by `constant(r.one())` gives the 4 terms; `{1: u1, x: u8} == constant(r.one())` holds both
ways (HEAD: false) and `{1: u1, x: u8} != constant(u1)`. The Laurent value `(q^-1+1)^8 - q^-8 + w` gives
`(1+w) + 4q^-2 + 6q^-4 + 4q^-6`. Over GF(3), `(x+y)^3 - x^3 - y^3 + 2xy` gives the single term `2xy` (HEAD:
3 terms, two of them the unbound integer 3).

Cost: for coefficient types without unbound constants every branch is compiled out (a `const` test).
For the sentinel types: one `Option` check per coefficient in univariate `new`, one per `add_term` call in
the sparse types, a clone of one coefficient as model, and one O(n) `bind_all` per value at most.

### 7.3 Docs

`galois_ring.rs:370-381` ("containers" bullet) is rewritten: `UnivariatePolynomial` (on construction),
`MultivariatePolynomial` and `LaurentPolynomial` (in `add_term`) bind every unbound coefficient to the ring
of a bound coefficient (`Ring::bind_to_parent_of`), so `(x+1)^8 - x^8 + w` has degree 6; a container with
only unbound coefficients stays over Z (degree 7) until it meets a bound element; the generic containers of
rustmath-matrix, rustmath-powerseries and rustmath-rings still bind entry by entry (HANDOFF, §7.6). The
same sentence goes into the type docs of `FiniteFieldElement` (`finite_field.rs:396-425`), `PrimeField`
(`prime_field.rs:12-35`) and `IntegerMod` (`integer_mod.rs:47-58`). The three container types get a short
"Unbound constants" paragraph (the rule, invariant U or M, and the sparse-zero consequence). `rustmath-core`
`traits.rs` documents the hooks as in 7.2.

### 7.4 Callers and behaviour changes

`UnivariatePolynomial::new`, and `add_term` of the sparse types, are called everywhere, but behaviour changes
only for coefficient types with `HAS_UNBOUND_CONSTANTS = true` (the five types above and containers over
them) and only for values that mix bound and unbound coefficients; there, degrees, `is_zero`, term counts,
`leading_coefficient` and `==` now follow the bound ring, which is what the HEAD docs called the caveat.
Users of univariate polynomials over these types (grep): `rustmath-matrix` (`companion.rs`,
`polynomial_matrix.rs`, `krylov.rs`, `charpoly_exact.rs`; matrices of only sentinels stay in Z),
`rustmath-curves` (`cantor.rs`, `divisor.rs`, `jacobian.rs`, `hyperelliptic.rs`, generic R),
`rustmath-coding` (`reed_solomon_generic.rs`), `rustmath-crypto` (`poly_bridge.rs`, Integer only),
`rustmath-combinatorics`, `rustmath-quantumgroups`, `rustmath-rings` (`function_field/typed`),
`rustmath-jupyter`, finitefields tests (`matrix_forms_gfp.rs`: its sentinel-only checks compare
equal-length polynomials, unaffected). `MultivariatePolynomial` and `LaurentPolynomial`: no file in the
workspace names one of them together with a sentinel type (grep over every file that mentions
`MultivariatePolynomial`, `MultiPoly` or `LaurentPolynomial` for `PrimeField`, `IntegerMod`,
`FiniteFieldElement`, `GaloisRingElement`, `ExtensionField`, `Gfpn`: none); generic code over them (the
Groebner engine over `F: Field`) changes only for mixed values. `rustmath-topology`'s `LaurentPolynomial` is
a separate, non-generic type (`HashMap<i32, i64>`). The acceptance run covers all of these.

### 7.5 Tests

In rustmath-polynomials (plan step 2), new `tests/unbound_constants.rs` with a test-only sentinel type
`Zs { v: i64, m: i64 }` (m = 0 is unbound; arithmetic binds an unbound operand into the other's modulus and
reduces; two bound operands with different moduli panic; `==` coerces like the real types; `zero()`/`one()`
are unbound; the three hooks as in 7.2). All values mod 8 come from gp `lift(Mod(1,8)*(..))`
(`$X/rev1/rev1_ref.log`). `Zs` needs the new hooks, so these tests cannot compile at HEAD; the HEAD column
gives what the HEAD containers do with an equivalent sentinel (the same arithmetic as HEAD's `IntegerMod`
modulus-0 sentinel, `$X/review1/r7_cont.rs`, `$X/rev1/rev1_cont.log`):

| Name | Input | Expected | HEAD |
|---|---|---|---|
| `univariate_binds_mixed_coefficients` | `(x+1)^8 - x^8 + constant(Zs(1, 8))` from `var()` and `constant(Zs::one())` | coefficients `[2, 0, 4, 0, 6, 0, 4]`, all with m = 8, degree 6 (gp `4*x^6 + 6*x^4 + 4*x^2 + 2`); `coeff(9)` has m = 8 | degree 7 |
| `univariate_all_unbound_stays_in_z` | `(x+1)^8 - x^8` | `[1, 8, 28, 56, 70, 56, 28, 8]`, all unbound (gp `Vecrev((x+1)^8 - x^8)`) | passes |
| `univariate_eq_coerces_across_lengths` | `P::new([u1, u8])` vs `P::constant(Zs(1, 8))`, both orders; vs `P::constant(u1)` | equal, equal; not equal | not equal |
| `nested_univariate_binds` | `UnivariatePolynomial<UnivariatePolynomial<Zs>>`: `A·y + B·y^2` with `A = (x+1)^4` (all unbound) and `B = constant(Zs(3, 8))` | every inner coefficient of A and of the zero coefficient at y^0 has m = 8; A = `[1, 4, 6, 4, 1]` (gp `lift(Mod(1,8)*((x+1)^4*y + 3*y^2))`) | A unbound |
| `univariate_over_laurent_binds` | `UnivariatePolynomial<LaurentPolynomial<Zs>>` with coefficients `[0, var(), constant(Zs(3, 8))]` (the 0 is an empty map) | the outer value is bound (`is_unbound_constant` false); the `var()` coefficient has m = 8; the empty zero stays empty; `==` with the same value built from bound coefficients holds | `var()` coefficient unbound |
| `multivariate_binds_on_add_term` | `(x_0+1)^8 - x_0^8 + constant(Zs(1, 8))`, and with the constant added first | 4 terms: 1 → 2, x_0^2 → 4, x_0^4 → 6, x_0^6 → 4, all m = 8; `coefficient(x_0^3)` is 0 with m = 8; both orders equal | 8 terms |
| `multivariate_eq_coerces` | `{1: u1, x_0: u8}` vs `constant(Zs(1, 8))`, both orders; vs `constant(u1)` | equal, equal; not equal | not equal |
| `laurent_binds_on_add_term` | `(q^-1 + 1)^8 - q^-8 + constant(Zs(1, 8))` from `monomial(Zs::one(), -1)` | terms q^0 → 2, q^-2 → 4, q^-4 → 6, q^-6 → 4, all m = 8 (gp: `4*y^6 + 6*y^4 + 4*y^2 + 2` with y = q^-1) | 8 terms |
| `laurent_ring_hooks` | `LaurentPolynomial::<Zs>::one()`, `::zero()`, `var()`; bound `constant(Zs(3, 8))` | `HAS_UNBOUND_CONSTANTS` true for `<Zs>`, false for `<Integer>`; `one()`, `zero()`, `var()` are unbound; `var().bind_to_parent_of(&bound)` has its q coefficient with m = 8; `bound.bind_to_parent_of(&var())` is `None` | new API |
| `default_hooks_are_inert_for_containers` | `UnivariatePolynomial<Integer>`, `MultivariatePolynomial<Integer>`, `LaurentPolynomial<Integer>` | `HAS_UNBOUND_CONSTANTS` false; `is_unbound_constant` false; equality as at HEAD (e.g. `x + 1 != x`) | new API |

In rustmath-finitefields (plan step 3.9), with the real types:

| Name | Input | Expected | HEAD |
|---|---|---|---|
| `polynomials_from_static_constructors_bind_on_construction` (replaces the test at `galois_ring_sentinel.rs:196` in the same step) | q as in 7.1 over GR(8, 2) | `q.degree() == Some(6)`; `q == qb`; every coefficient bound; `q.coeff(9)` bound zero; the all-unbound `(x+1)^8 - x^8` keeps `[1, 8, 28, 56, 70, 56, 28, 8]` unbound (gp `Vecrev((x+1)^8 - x^8)`); `P::new([unbound(1), unbound(8)]) == P::constant(r.one())`; `q.scalar_mul(&r.one()) == qb` still | fails (degree 7) |
| `sparse_polynomials_from_static_constructors_bind` (new, `galois_ring_sentinel.rs`; the reviewer's test) | over GR(8, 2): `(x_0+1)^8 - x_0^8 + constant(w)` from `MultivariatePolynomial::variable(0)` and `constant(R::one())`, both orders of the last addition; `(q^-1+1)^8 - q^-8 + constant(w)` as a `LaurentPolynomial` from `monomial(R::one(), -1)` | multivariate: `num_terms() == 4`, `degree() == Some(6)`, constant term `1 + w` (eltseq `[1, 1]`), x_0^2 → 4, x_0^4 → 6, x_0^6 → 4, all bound, x_0^1, x_0^3, x_0^5, x_0^7 absent; equal to the same value built from `r.one()`; Laurent: q^0 → 1 + w, q^-2 → 4, q^-4 → 6, q^-6 → 4 (gp `lift(Mod(1,8)*((x+1)^8-x^8))` = `4*x^6 + 6*x^4 + 4*x^2 + 1`; `$X/rev1/rev1_cont.log` prototype) | 8 terms each, degree 7 (`$X/rev1/rev1_cont.log`) |
| `polynomials_over_galois_ring` (`:164`) | unchanged | passes | passes |
| `containers_bind_unbound_constants` (new, `tests/sentinel_eq.rs`) | univariate: GF(3) `(x+1)^3 - x^3 + constant(gf3(2))`; Z/8 `(x+1)^8 - x^8 + constant(1 mod 8)`; GF(4) `(x+1)^2 - x^2 + constant(w)`. Multivariate over `PrimeField` mod 3: `(x+y)^3 - x^3 - y^3 + constant(2)·x·y` | zero polynomial; `[2, 0, 4, 0, 6, 0, 4]` (degree 6); `constant(1 + w)` (gp rows of 7.1); the single term `2xy` (gp `lift(Mod(1,3)*((x+y)^3 - x^3 - y^3 + 2*x*y))` = `2*y*x`) | degree 2; 7; 1; 3 terms |
| `nested_polynomials_bind` (same file) | `UnivariatePolynomial<UnivariatePolynomial<PrimeField>>` with an unbound inner coefficient and a bound one | all inner coefficients bound | fails |
| `default_hooks_are_inert` (`rustmath-core` traits tests) | `i32`, `Integer` | `HAS_UNBOUND_CONSTANTS == false`, `5.bind_to_parent_of(&3) == None` | new API |

### 7.6 Other containers: explicit hand-off

The generic containers outside rustmath-polynomials keep entry-by-entry binding, as at HEAD. Most live in
crates of units designed now against HEAD (rustmath-matrix: M1a, M1b; rustmath-rings: R1), so editing them
here would collide with those designs. rustmath-powerseries belongs to no unit of this run; it is a separate
crate (one crate per PR), and no code builds its series over a sentinel type (grep over every file naming a
series type for the sentinel types: only `rustmath-powerseries/src/algebraic.rs`, which uses `PrimeField`
values and no series of them), so its interim behaviour reaches no caller. Matrices over the sentinel types
are common (for instance `rustmath-finitefields/tests/matrix_forms_gfp.rs`), but the matrix code builds its
own constants from a bound entry (`krylov.rs:36-41`), so a mixed matrix arises only when user code puts
`R::one()` into a bound matrix; the sentinel-only case is documented (`krylov.rs:43-47`,
`companion.rs:36-41`: an `Err` and a fallback, never a wrong value). The hook call that fixes each one is the rule of `UnivariatePolynomial::new`: in every
constructor and in-place setter, when `R::HAS_UNBOUND_CONSTANTS`, take the first entry e with
`!e.is_unbound_constant()` as the model and replace every unbound entry c by `c.bind_to_parent_of(&e)`
(dense types); for sparse types, the `add_term` rule of 7.2.

| Type | Where | Interim behaviour | Fix |
|---|---|---|---|
| `Matrix<R>` | `rustmath-matrix/src/matrix.rs:9` (dense `Vec<R>`; ten constructors plus `set`) | an entry that met only unbound constants stays an integer in Z; `is_zero`, rank and determinant of a mixed matrix can see those integers; equality is entrywise and coerces | the dense rule in each constructor and in `set` |
| `Vector<R>` | `rustmath-matrix/src/vector.rs:8` | same | the dense rule in `new` and in setters |
| `SparseMatrix<F: Field>` | `rustmath-matrix/src/sparse.rs:20` (stores nonzero values) | an unbound value that is 0 in the field is stored as nonzero | the sparse `add_term` rule at every insertion |
| `PowerSeries<R>`, `LaurentSeries<R>` | `rustmath-powerseries/src/series.rs:17`, `laurent.rs:29` | valuation and trimming see integers in Z | the dense rule in the constructors |
| `PowerSeriesPari<R>`, `PowerSeriesPoly<R>`, `LaurentSeries<R>`, `LaurentSeriesElement<R>` | `rustmath-rings/src/power_series_pari.rs:43`, `power_series_poly.rs:40` (public `coefficients` field: bind in the operations that read it), `laurent_series_ring_element.rs:46`, `laurent_series_ring.rs:127` | same | the dense rule in the constructors and operations |
| `PadicInteger`, `PadicRational`, `PadicExtensionElement` (R1 §A.3 gives them unbound sentinels; review 1, m7) | `rustmath-rings/src/padics/padic_integer.rs`, `padic_rational.rs`, `extension.rs` | until they implement the hooks, a container over them binds entry by entry (no `UnivariatePolynomial<Padic..>` exists yet, R1 §A.2, so nothing breaks now) | implement `HAS_UNBOUND_CONSTANTS = true`, `is_unbound_constant` (= `!is_bound()`) and `bind_to_parent_of` (MAGMA `L ! c` at the model's precision, R1 §A.3) |

HANDOFF gets these rows as one follow-up: "Generic containers outside rustmath-polynomials (table in the F1
design §7.6): bind unbound constants with `Ring::bind_to_parent_of` (F1 hook); R1's p-adic sentinels
implement the three hooks." The docs of 7.3 say exactly which containers bind.

---

## 8. Findings outside the items (for HANDOFF, not fixed here)

1. `FiniteField::new(p, n)` without a Conway polynomial calls `poly_factor::find_irreducible`, a
   lexicographic scan with c_0 varying fastest. When no x^n + c_0 is irreducible (for instance n = 3 and
   p ≡ 2 mod 3, where cubing is a bijection of F_p), the scan tests about p candidates:
   `FiniteField::new(2^80+13, 3)` does not return within 120 s (`$X/t9_ffnew.rs`). A fix changes the
   default moduli of non-Conway fields (MAGMA uses "random search with irreducibility test", digest
   §21.5), so it needs its own decision.
2. Canonical `poly_factor::roots(0)` and `factor(0)` return `[]` (every element is a root of 0).
3. Other trial-division sites in the crate: `IntegerMod::sqrt` (`:348`), `all_sqrts` (`:396`), `num_sqrts`,
   `is_nilpotent` (`:272`), `Zmod::factored_modulus`, `ring_hooks::is_nilpotent_mod`. They belong to the
   REPL `sqrt_mod` follow-up (J1); with this unit each is a one-line switch to
   `group_order::factor_proven(.., Effort::Complete)`.

---

## 9. Plan

### 9.1 Implementation order (each step builds and tests green before the next)

1. **rustmath-core** `src/traits.rs`: the three hooks of 7.2 after `static_is_commutative` (P1), with the
   container wording of their docs; the new doc of `MultiplicativeOrder::Undecided` (4.4); a unit test
   `default_hooks_are_inert`. `cargo test -p rustmath-core`.
2. **rustmath-polynomials**: `src/univariate.rs` (`bound_model`, `new`, `coeff`, `PartialEq`, the hooks in
   `impl Ring`); `src/multivariate.rs` (`bound_model`, `bind_all`, `add_term`, `coefficient`, manual
   `PartialEq` instead of the derive); `src/laurent.rs` (the same, plus the hooks in `impl Ring` next to
   P1's); the "Unbound constants" paragraphs of the three type docs; new `tests/unbound_constants.rs`
   with the `Zs` test type (7.5, first table). The existing tests do not change (Integer, Rational, i32,
   i64 coefficients have no unbound constants, so every new branch is compiled out for them).
   `cargo test -p rustmath-polynomials`.
3. **rustmath-finitefields**, in this order:
   1. `src/poly_factor.rs`, part A: `check_char`; `is_irreducible_fp` checked and
      `is_irreducible_fp_unchecked` (3.2); `finite_field.rs:202` and `galois_ring.rs:120` call the
      unchecked body; tests `is_irreducible_fp_rejects_non_prime_p`, `is_irreducible_fp_unchanged_for_primes`.
      (First, because `Gfpn::try_new` in the next step uses the unchecked body.)
   2. `src/ff_poly.rs`: `pub(crate) fn check_prime_characteristic` (moved from `cantor_zassenhaus.rs`,
      using `try_is_prime`); `from_int` default (double-and-add) and `Gfpn::from_int`; `Gfpn::try_new`,
      `Gfpn::new`, the `GFPN_VALID` memo, the `invert` message; docs; tests of 2.5 and the `from_int`
      tests of 3.4.
   3. `src/cantor_zassenhaus.rs`: `squarefree_check`, `ddf_squarefree`, `squarefree_prime_char`, checked
      public DDF and squarefree factorization, internal callers switched; module docs `:52-68`, `:135-140`,
      `:500`, `:560`, `:585`; tests of 1.4.
   4. `src/irreducible.rs`: `rabin_irreducible`, checked `is_irreducible`, uniform `random_irreducible`,
      checked `find_irreducible` (with `find_irreducible_unchecked`) and `generate_gfpn_modulus`; tests.
   5. `src/poly_factor.rs`, part B: uniform `random_irreducible` (+ unchecked), checked `find_irreducible`
      (+ unchecked), checked DDF (+ `ddf_squarefree`), `factor`'s part check; tests.
      `src/finite_field.rs:177` calls `find_irreducible_unchecked`.
   6. `src/group_order.rs` (new) and `mod group_order;` in `src/lib.rs` (crate-private), with unit tests
      of `order_from_exponent`, `unit_order` (the ±1 shortcuts), `unit_order_mod` (Fermat and λ paths:
      2 and 3 mod 341, 2 mod 561), `carmichael_lambda` (λ(341) = 30, λ(2^10) = 256, λ(561) = 80, gp
      `lcm` of the λ(p^e)), and `factor_proven_uncached` with both efforts on P43 - 1 (`Capped`: `Err`
      naming the 41-digit composite; `Complete`: `[(2,1),(43,1),(100000000000000000039,1),
      (300000000000000000053,1)]`, gp `factor(P43-1)`). The Capped assertion uses the uncached function,
      so it does not depend on test order (4.3).
   7. `src/integer_mod.rs`, `src/prime_field.rs`, `src/finite_field.rs`, `src/galois_ring.rs`: orders,
      hooks and their docs, `is_generator`, `is_primitive`, `factored_multiplicative_order`,
      `order_factored`, `UnitOrder::Undecided`; remove `use rustmath_integers::prime::factor` from
      `finite_field.rs`. In the same step the new `tests/unit_group_orders.rs` and
      `tests/unit_group_orders_capped.rs` (4.6), and the comment fix in `tests/ring_predicates.rs:236`.
   8. `src/prime_field.rs`, `src/finite_field.rs`, `src/extension_field.rs`: `is_square`, `sqrt`; new
      `tests/field_sqrt.rs`.
   9. Item 7: the core hooks in the five types; docs of 7.3. In the same step (review 1, m3): replace
      `galois_ring_sentinel.rs::polynomials_from_static_constructors_keep_unbound_coefficients` by
      `polynomials_from_static_constructors_bind_on_construction`, add
      `sparse_polynomials_from_static_constructors_bind`, and add the container tests of
      `tests/sentinel_eq.rs` (7.5, second table). The pinned test turns red exactly when these impls land,
      so it must change in this step.
   10. Item 6: the check of section 6, or its HEAD fallback with `square_root_mod_prime_mod_two`.
4. **Docs**: `PROJECT_TRACKER.md` rows 805 (cantor_zassenhaus: drop "Not checked: a Gfpn ...; squarefree
   ... and DDF check nothing"), 807 (DDF, now checked), 809, 812 (generators: uniform, checked p;
   `is_irreducible_fp` checked), 814 (orders: group_order, Alg. 1.4.3, Fermat multiple, capped hooks with
   their cost bound), 817 (PrimeField: is_square/sqrt with the root), 822 (from_int O(1) on Gfpn), 823; a
   new row for `Gfpn::try_new`; the rows of `UnivariatePolynomial`, `MultivariatePolynomial`,
   `LaurentPolynomial` (binding of unbound constants); `docs/magma/status_overrides.csv` rows
   ch021.IsSquare and ch021.SquareRoot (partial → done, with test names), ch021.FactoredOrder (missing →
   done), ch021.Order, ch021.IsPrimitive, ch019.Order (notes and paths), ch021.RandomIrreduciblePolynomial
   (still partial: F_p only, now uniform), ch023.DistinctDegreeFactorization and
   ch023.SquarefreeFactorization (notes); run `python3 tools/magma_tracker/build.py`. `docs/HANDOFF.md`:
   remove the seven items, add the hand-off of 7.6 and the lines of section 8.

### 9.2 Acceptance commands

```bash
cargo test -p rustmath-core --no-fail-fast
cargo test -p rustmath-polynomials --no-fail-fast
cargo test -p rustmath-finitefields --no-fail-fast
cargo test -p rustmath-finitefields --test unit_group_orders_capped -- --test-threads=1   # B2: deterministic
# downstream of finitefields (Cargo.toml grep) and of the polynomial changes
cargo test -p rustmath-galois --no-fail-fast
cargo test -p rustmath-matrix --no-fail-fast
cargo test -p rustmath-rings --no-fail-fast
cargo test -p rustmath-coding -p rustmath-crypto -p rustmath-curves -p rustmath-ellipticcurves \
           -p rustmath-algebras -p rustmath-modules -p rustmath-powerseries -p rustmath-schemes --no-fail-fast
cargo test -p rustmath-jupyter --no-fail-fast        # long; detach with nohup and log to the scratchpad
cargo check --workspace --tests
cargo clippy -p rustmath-finitefields -p rustmath-polynomials -p rustmath-core
```

The core change rebuilds the whole workspace (as P1's does).

### 9.3 Risks

- **I1/I2 API drift**: every use goes through one private function (section 0.1); the HEAD fallbacks are
  listed there. The prototype `p2_order.rs` already runs on HEAD's `factor_pollard_rho`.
- **The capped hook's limits** (30-digit sieve, 8191 rho steps, ECPP to 256 bits) are a cost choice. Too
  small makes more hooks `Undecided` (never wrong); too large makes generic code slow on hard group orders.
  The cache makes repeated hooks in one field cheap.
- **Process-wide cache and tests**: a capped `Undecided` can become decided after another call factored
  the same integer. Only `tests/unit_group_orders_capped.rs` asserts a capped `Undecided` on a large
  group order, and it holds one test, so its phases run in a fixed order in a process of their own. A
  future test that asserts a capped `Undecided` must go into a file of its own for the same reason; the
  header of that file says so. The `Undecided` assertions of `ring_predicates.rs` are on unbound sentinels
  (`sentinel_order`), which the cache does not touch.
- **Container binding** is a semantic change for mixed values over the sentinel types. A downstream test
  that relied on an unbound coefficient staying in Z inside a mixed polynomial would change; the acceptance
  list runs every crate that builds such polynomials. Values with only unbound coefficients are unchanged.
- **`PartialEq` on polynomials** now coerces across lengths (univariate) or key sets (sparse), bound
  against unbound. The transitivity caveat of the element `==` carries over (an all-unbound polynomial can
  equal bound polynomials of different rings). A sparse zero carries no parent (7.2).
- **Global state**: two memos (`GFPN_VALID`, the order cache), both behind locks never held across a
  panic or a long computation; the order cache is bounded.
- **New panics**: lava DDF/squarefree/`is_irreducible` and canonical `is_irreducible_fp` with a composite
  characteristic, `Gfpn::new` with an invalid modulus, `PrimeField::sqrt` with a composite modulus. Each
  replaces a wrong value or an unrelated panic deep inside the arithmetic.
- **Test time**: `is_primitive_and_is_generator` and phase 2 of `capped_hooks_then_exact_orders_p43` each
  run the sieve once on the 41-digit cofactor of P43 - 1 (each file is its own process, so the cache does
  not share it; I2 measured about 0.4 s at 39 digits in debug). If either exceeds 10 s in debug, mark it
  `#[ignore]` with the gp values in a comment.
