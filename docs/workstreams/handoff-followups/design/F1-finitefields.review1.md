# Review 1 of design F1 (rustmath-finitefields)

Reviewer scratch: `$R` = `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/F1-finitefields/review1/`
(`rc.sh` builds against the designer's HEAD rlibs; `r1_ddf.rs`, `r3_irr.rs`, `r4_order.rs`, `r7_cont.rs`, `ref1.gp`).

Verdict: **not approved**. Three blockers (one wrong test input, one order-dependent test, one
under-justified scope limit in item 7). The algorithms, citations and most reference values are right.

## What I reproduced at HEAD c79a582

| Item | Claim | My check | Result |
|---|---|---|---|
| 1 | Both DDFs accept non-squarefree input | `r1_ddf.rs`: `(x+1)^2(x^2+2)(x^2+3)` mod 5 gives `[(x+1,1), (x^5+x^4+x+1,2)]` in both stacks; `(x+w)^2` over GF(4) gives `x+w` twice | confirmed |
| 1 | Lava DDF/squarefree accept modulus 15; DDF(0) = [] | `x^2+1 mod 15`: DDF `[(x^2+1, 2)]`, sqf `[(x^2+1, 1)]`; DDF of 0 is empty | confirmed |
| 2 | `Gfpn::new` checks nothing | non-monic `2t^2+1` mod 5: `t*t = [4,0]` (gp: 2); `t^2+2` mod 3: `invert(t+1)` Err, `order() = 9`; p = 15 `order() = 225`; `[]` overflow panic; `[1]` index panic | confirmed |
| 2 | Only constructor path is `Gfpn::new` | grep `Gfpn {` / `Gfpn::new` over the workspace: literals only inside `ff_poly.rs`, all copy `p`/`modulus` from an existing value; callers galois_ctx.rs:213, :585, relative_invariant.rs:300, :379, :382 and in-crate tests; no serde/Default/From | confirmed |
| 3 | `from_int` linear on `Gfpn` | `from_int(10^5)` over 2^61-1: 0.15-0.23 s | confirmed |
| 3 | Non-prime p in generators | `is_irreducible_fp(x^2+1, 15)` = true, lava `is_irreducible(x^2+1 mod 15)` = true (`r3_irr.rs`); designer's `t8_badp.log` agrees | confirmed (and see m4) |
| 4 | O(m) loops, trial division | `r4_order.rs`: `IntegerMod(2, 1000003)` 1.13 s, `PrimeField` hook 1.19 s, GF(p12) order(3) 0.22 s | confirmed (matches 1.08/1.04/0.22 s) |
| 5 | No root API on `PrimeField`/GF(p^n) | grep | confirmed |
| 6 | `square_root_mod_prime` tests primality twice | integer_mod.rs:651 `assert_prime` then `sqrt_mod_prime`, whose `is_quadratic_residue_prime`/`sqrt_mod_prime` run `is_prime` again; I1 design §2.7 takes it | confirmed; hand-off consistent with I1 |
| 7 | Containers keep unbound coefficients | `r7_cont.rs`: Z/8 `(x+1)^8-x^8+1` degree 7; GF(3) `[0b, 3u, 3u]` degree 2; GF(4) degree 1; `P::new([u1,u8]) == P::constant(1 mod 8)` false | confirmed |

Reference values: I ran my own gp script (`$R/ref1.gp`, PARI/GP 2.15.4) for a sample: `znorder` rows of
`integer_mod_orders_match_pari` (all 7), `factor(P43-1)`, `fforder` for GF(5^30) w, w+1 and GF(101^12) w,
`factor(fforder(w+1))` in GF(125), the square roots over GF((2^61-1)^2), GF((2^80+13)^2), GF(256) and mod
2^80+13 and 10^9+7 (canonical choice re-derived from `+-r`), the `from_int` residues mod 2^61-1, the Z/8
container value, and `prod_{d|40} Phi_d(3) = 3^40-1`. All agree with the design tables. Citations checked
in the digests: ch. 23 §23.8.1 (DDF needs squarefree f), ch. 19 `Order` (0 for non-units), ch. 21 §21.4.4
`IsSquare` returns the root, §21.4.7 `Order`/`FactoredOrder`/`Sqrt`, §21.5 `RandomIrreduciblePolynomial`,
ch. 48 §48.2.3 `R ! a`. The note that "ch. 22" is Nearfields is right. Cohen Alg. 1.4.3, Carmichael lambda,
the cyclotomic split of q - 1 and Tonelli-Shanks in a cyclic group are correctly stated; I re-derived the
invariant of Alg. 1.4.3 (any multiple h of the order works) and the char-2 and q = 3 mod 4 root formulas.
I1/I2 APIs named in §0.1 exist in those designs with the stated semantics.

## Blockers

### B1. `gfpn_normalizes_the_modulus` uses a reducible modulus (item 2)

§2.1 row "modulus `[8,1,1]` vs `[1,1,1]`, p = 7: unequal fields / the same field" and test
`gfpn_normalizes_the_modulus` (§2.5) expect `Ok` and equal `one()`s. But `t^2+t+1` is reducible over F_7:
gp `polisirreducible(Mod(1,7)*(t^2+t+1))` = 0, `factormod(t^2+t+1,7)` = `(t+3)(t+5)`. Under the designed
`try_new` (Rabin check) both calls return `Err(.. reducible ..)`, so the test cannot pass, and the evidence
row ("the same field") is false (the designer's own `t2_gfpn.log` line used this reducible modulus).
**Change:** use an irreducible modulus, e.g. p = 5 with `[6,1,1]` vs `[1,1,1]` (gp
`polisirreducible(Mod(1,5)*(t^2+t+1))` = 1), or p = 2 with `[3,1,1]`; fix the §2.1 row.

### B2. The capped-hook test is order-dependent through the global order cache (item 4)

§4.3: the cache stores complete factorizations "from either effort, so a hook benefits from an earlier
inherent call". §4.6 puts `capped_hook_inherent_order` (inherent, `Complete`, 2 and 3 mod P43) and
`is_primitive_and_is_generator` (`PrimeField(2, P43)`, `Complete`) in the same binary
(`tests/unit_group_orders.rs`) as `hooks_are_capped_and_honest`, which asserts `Undecided` for the
`PrimeField`/`IntegerMod` hooks of 2 mod P43. Once either Complete test has run, `Lambda(P43)` is cached
complete and those hooks return `FiniteExceedsU64` (P43 - 1 > 2^64). libtest runs tests in name order:
with `--test-threads=1`, `capped_hook_inherent_order` runs before `hooks_are_capped_and_honest`, so the hook
test fails deterministically; with parallel threads it is a race. **Change:** give the hook test a modulus
that no `Complete` call in that binary factors (or move it to its own test file, a separate process), keep
one `Undecided` case per hook type that is cache-proof, and say in the hook docs that a capped answer may
become decided after an exact computation in the same process (both answers are honest).

### B3. Item 7: the scope limit for `MultivariatePolynomial` (and `LaurentPolynomial`) is not justified

The item is about "containers built from `R::one()` or `UnivariatePolynomial::var()`", and the type docs it
points to (`galois_ring.rs:370-381`) speak of "containers (polynomials, matrices, ...)". The design fixes
`UnivariatePolynomial` and leaves `MultivariatePolynomial` as a "stated limit" because binding "needs either
a per-operation normalizing pass or a flag ... separate changes in other units' crates". That reason does not
hold for the two containers in `rustmath-polynomials`: F1 already edits that crate (univariate.rs), P1 lands
before F1, and both types are small: `MultivariatePolynomial` has three struct literals (`multivariate.rs:300`,
`:313`, `:320`) and every other insertion goes through `add_term` (`:324`) or the four operator impls
(`:760-814`); `LaurentPolynomial` (`laurent.rs:15`) is a `BTreeMap<i32, R>` with two literals. A normalize step
(bind unbound coefficients to a bound sibling, drop the ones that become zero) at the end of
`constant`/`monomial`/`add_term`-based operations plus a coercing `PartialEq` (derived `PartialEq` on the map
compares keys first, so it has the same cross-length problem as the univariate case) is feasible here.
**Change:** design and test the binding for `MultivariatePolynomial` and `LaurentPolynomial`. A test in the
style of the univariate one: over GR(8, 2), `(x_0 + 1)^8 - x_0^8 + w` built from `variable(0)` and `R::one()`
keeps the terms 8, 28, 56, 70, 56, 28, 8 unbound at HEAD; after binding it must equal
`(1 + w) + 4x^2 + 6x^4 + 4x^6` (gp `lift(Mod(1,8)*((x+1)^8-x^8))` = `4*x^6 + 6*x^4 + 4*x^2 + 1`), with the
x^1, x^3, x^5, x^7 terms gone from the map. `Matrix`, `Vector`, `LaurentSeries` live in other crates whose units come later;
keeping them on entry-by-entry binding is acceptable only as an explicit hand-off that lists each type, its
interim behaviour, and the hook call that fixes it.

## Minor issues

- **m1 (item 2 tests).** `Gfpn::new` panics with "Gfpn::new: {error}", and `MathError::InvalidArgument`
  displays as "Invalid argument: {0}" (`rustmath-core/src/error.rs:20`), so the message is
  "Gfpn::new: Invalid argument: Gfpn: the defining polynomial is reducible over GF(3)". The
  `#[should_panic(expected = "Gfpn::new: Gfpn: the defining polynomial ...")]` substring does not match.
  Use the inner string in the panic or fix the expected substring.
- **m2 (item 4, Galois rings).** The test `hooks_are_capped_and_honest` includes `GaloisRing::new(P43, 2, 1)`
  and expects `-1` to give `Finite(2)`, but the GR row of §4.4 only says the residue order uses
  `field_group_order(.., effort)`. Without a `+-1` shortcut in the shared residue-order helper (or in the GR
  hook) the order of `-1` needs the capped factorization of P43 - 1 and comes back `Undecided`. Specify the
  shortcut in the helper.
- **m3 (plan).** Step 3.8 (item 7 impls) turns `galois_ring_sentinel.rs::polynomials_from_static_constructors_keep_unbound_coefficients`
  red until step 3.10 replaces it, contrary to "each step builds and tests green". Replace the test in 3.8.
- **m4 (item 3 neighbour).** `poly_factor::is_irreducible_fp` is public and re-exported at the crate root
  (`lib.rs:57`); it answers `true` for `x^2+1` mod 15 (`r3_irr.rs`). The design checks the lava
  `is_irreducible` but leaves this predicate answering over a non-field. Add the same characteristic check
  (with an unchecked crate-private body for `try_new`, `with_modulus` and the generators), or at least list
  it in §8.
- **m5 (item 4 cost).** "Capped" still runs ECPP (`proof: true`), so it is not a time cap for large prime
  factors. For `PrimeField` the hook proves the modulus prime only to form lambda(m), although Alg. 1.4.3
  needs only a known multiple of the order: try `h = m - 1` first when `x^(m-1) = 1` (checked), and use
  lambda(m) only otherwise. Also update the core doc of `MultiplicativeOrder::Undecided` ("This element type
  cannot decide") to cover "not decided within the type's cost cap".
- **m6 (item 6 fallback).** "If I1 did not land it, apply I1 §2.7's snippet" is not applicable on its own: the
  snippet needs `Integer::try_sqrt_mod_prime`, which is I1's. Give a fallback that works on HEAD's API (drop
  `assert_prime` and map `sqrt_mod_prime`'s "Modulus must be prime" error to the same panic), or drop the
  fallback.
- **m7 (cross-unit).** R1 (implemented after F1) introduces unbound sentinels for `PadicInteger`,
  `PadicRational`, `PadicExtensionElement` (R1 §A.3). The HANDOFF line of §7.6 should name them as future
  implementers of `HAS_UNBOUND_CONSTANTS`/`bind_to_parent_of` (no `UnivariatePolynomial<Padic..>` exists yet,
  R1 §A.2, so nothing breaks now).
- **m8 (docs).** `FiniteFieldElement::is_primitive` and `factored_multiplicative_order` use `Complete` and so
  panic on an unresolved primality proof; their rows in §4.4 should say so. `factored_multiplicative_order(1)`
  is `Some(vec![])` (MAGMA `FactoredOrder(1)` is the empty sequence); state and test it. The core docs of
  `is_unbound_constant` say "integer constant", but for `UnivariatePolynomial` it means "a polynomial with
  only unbound coefficients" (the image of Z[x]); say so.

## Checked and fine

- Item 1 fix: eager checks, private unchecked bodies, internal callers use them (no gcd or primality test
  runs twice); the constant/zero conventions are stated; the `p_usize = 0` argument for p > usize::MAX is
  right (a nonconstant f with f' = 0, or a factor of multiplicity >= p, needs degree >= p).
- Item 2: validation order, normalization, the memo (borrowed `HashSet<Vec<Integer>>` lookup, no lock held
  during Rabin, poison recovery), the now-unreachable `invert` branch kept as an `Err`. All in-tree
  `Gfpn` moduli (cz tests, irreducible tests, galois) are irreducible.
- Item 3: `uniform_below` is exact; the prototype reimplements the crate's sampler and SplitMix64 exactly
  (compared with `ff_poly.rs:146` and `poly_factor.rs:534`), so the pinned outputs are reproducible; gp
  confirms each pinned polynomial irreducible; bounded search returns `Err(NotSupported)` (decides nothing).
- Item 4: Alg. 1.4.3 with any multiple h; lambda(m) for every m (so `PrimeField` with a composite modulus
  is right, and `is_generator` is false for composite m since lambda(m) < m - 1); only proven-complete
  factorizations are used; hooks map failure to `Undecided`. Existing ring_predicates tests stay decided
  (their group orders factor trivially).
- Item 5: Euler's criterion decides; the non-square scan is only a helper; the canonical root matches
  `IntegerMod::sqrt` for n = 1; values match PARI.
- Item 7 univariate design: the "all bound or none" invariant holds through `new` (only struct literal at
  `univariate.rs:62`, no `&mut self` methods), `Add`/`Sub`/`Mul` mixed results bind, the cross-length
  `PartialEq` fix is needed and sufficient, no `Hash` impl exists to break.
- Plan: acceptance list covers every reverse dependency of rustmath-finitefields that has tests
  (algebras, coding, crypto, curves, ellipticcurves, galois, jupyter, modules, powerseries, rings, schemes)
  plus `cargo check --workspace --tests`.
