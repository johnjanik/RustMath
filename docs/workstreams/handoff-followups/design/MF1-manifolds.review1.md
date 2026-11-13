# MF1-manifolds: review 1 (independent, adversarial)

Design reviewed: `fu2/design/MF1-manifolds.md` (730 lines, no Revision sections). HEAD `c79a582`, tree clean.
Scratch: `$R = .../fu2/design/MF1-manifolds/review1/` (programs built with the designer's `../build.sh`
against the HEAD debug rlibs; binaries deleted afterwards).

Verdict: **not approved**. Two blockers, both in the new zero test (`expr_zero.rs`, design 1.6), which
every new operation runs through `normalize`. The rest of the design is sound and well evidenced.

## What I reproduced and found to hold

* HEAD defects (`$R/repro_head`, built from the designer's `repro_head.rs`): d(xy dx + x^2 dy) gives
  `[(y + x), (2 * x), 0, 0]`; dx∧dx gives `(1 * 1)` nodes; trace of diag(5, 7) gives `[1]`; the tensor
  product of (0,1) by (1,0) gives `[3, 6, 4, 8]` (true `[3, 4, 6, 8]`); panics: `j % 0` at
  `diff_form.rs:151`, `pow` overflow at `tensor_field.rs:158` and `:105`, `p + 1` at `diff_form.rs:132`,
  `p + q` at `:182`; `rank as u32` truncation gives 2 for rank 2^32 + 1. All as stated.
* HEAD cost (`measure_head`, debug): d(dx_0) n = 1000: 7.85 s, 253 MB; d(d(dx_0)) n = 100: 1.64 s, 444 MB;
  dx_0∧dx_0 n = 1000: 0.29 s, 253 MB. Prototype (`measure_new`): 0.15 s / 0.11 s / 0.20 s, all 66 MB.
  Symbolic wedge, 999000 slots: 10.98 s, 800 MB. d∘d of x_{i+1}x_{i+2} dx_i, n = 60: all zero. Claims hold
  for these inputs.
* Formulas re-derived. d: (dω)_I = Σ_k (-1)^k ∂_{i_k} ω_{I\i_k} follows from d(f dx^J) = df∧dx^J and
  dx^{i_k}∧dx^{I\i_k} = (-1)^k dx^I. Wedge: grouping the p! q! permutations of Alt(ω⊗η) by the set A gives
  Σ_A sgn(A, I\A) ω_A η_{I\A} (determinant convention); the inversion count `inv += p - x` is that sign.
  Heap's algorithm flips parity at each swap. Interior product inserts X into the first slot (Lee).
  Cartan's formula and the p = 0, p = n, p > n cases are right. Contraction and the mixed-rank product
  layout are right.
* References: `gp -q refs.gp` prints every value quoted in 1.8 and 3.3. I re-derived dA, dB, A∧B, a∧b
  (1-form by 2-form) and the ω^k signs by hand, and fetched the Sage `diff_form` doc page: dA, A∧B,
  a∧b = (-x^2 + (y^3 - x - 1)*z + 2*z^2 - x) dx∧dy∧dz, da = (2*x + 2*y) dx∧dy and
  a∧b = (-x^3 - (x - 1)*y^2) dx∧dy are quoted correctly. The prototype checks (`t_ref`, `t_ref2`,
  `t_zero`, `t_contact`) all print `Zero` against them. Volume-form signs (-1)^(k(k-1)/2) match the
  `standard_form` layout (q_1..q_n, p_1..p_n, `symplectic.rs:194-200`).
* Callers: `git grep` over the workspace agrees with 3.4. Only rustmath-jupyter depends on the crate. No
  `match` on `ManifoldError` exists. `to_flat`, `total_rank`, `num_components` have no callers outside
  the crate. The REPL claims of 4.1 to 4.3 hold: `test_form_component_limits` lines 12359-12367 are the
  only REPL lines that reach the new chart check; `exterior_derivative(fs, CS)` errs on the lookup first;
  the `Cm` lines use the p >= n and p + q > n shortcuts. J1a and J1b already consume 4.4.
* Baseline: `cargo test -p rustmath-manifolds --no-fail-fast` at HEAD: 258 passed, 11 ignored, 31 doc
  tests, as the plan states.
* `try_reserve_exact` of 2^56 `Expr` (and of 2^40) fails here (overcommit mode 0, 15 GB), so the
  allocation test is sound on this machine. `size_of::<Expr>() == 64`.
* MAGMA citations exist: Chapter_82 §82.3 (exterior algebras, prose only), Chapter_42 §42.15
  (`Differential(a)`), Chapter_24 §24.4.2 and §24.4.7, Chapter_41, Chapter_111.

## Blockers

### B1. The zero test decides wrongly on the symbolic crate's reserved constants and on infinities

The soundness argument of 1.6 treats every `Symbol` as an independent indeterminate and every other
leaf as a finite value. rustmath-symbolic does not satisfy either assumption: its parser maps `I` and `i`
to `Symbol("I")`, `pi` to `Symbol("pi")`, `e` to `Symbol("e")`, and `oo` / `inf` / `infinity` to
`Symbol("oo")` (`rustmath-symbolic/src/parser.rs:216-237`), and `Expr::Real` can hold ±inf and NaN.
Unit S1, implemented just before MF1, makes `Symbol("I")` the imaginary unit and `pi`, `e` named
constants (`S1-symbolic.md:231-237`).

Reproduced with the prototype (`$R/r_consts.rs`, `$R/r_inf.rs`):

| Input | Prototype / design answer | Truth |
|---|---|---|
| `parse("I*I + 1")`, also after `simplify` | NonZero | 0 |
| `parse("i^2 + 1")` | NonZero | 0 |
| d((I^2 + 1) x dy) on R^2, slot (0,1) | NonZero, so `is_closed` would answer Ok(false) | the form is closed |
| `oo - oo` | Zero (`normalize` would store a literal 0) | undefined |
| `0 * oo` | Zero | undefined |
| `inf*x - inf*x` (`Expr::Real(INFINITY)`) | Zero | NaN |

So `NonZero` and `Zero` are not proofs as the design claims, and `normalize` can replace an undefined
component by a fabricated 0. Joint polynomial relations in `pi` and `e` are an open problem, so a
NonZero answer over both is not a proof either.

Required change (small): in the conversion, (a) a `Symbol` named `I`, `pi` or `e` is an opaque atom (Zero
answers stay sound, NonZero becomes Unresolved; `I` may instead be reduced with I^2 = -1); (b) any
occurrence of `Symbol("oo")` or of a non-finite `Real` anywhere in the tree gives Unresolved at once (an
opaque atom is not enough: `t - t` with t = oo still gives Zero); (c) state the precondition as "every
opaque atom is finite on a dense open set" (so `log(0)` and `1/(sin^2 + cos^2 - 1)` are outside the
contract, like the identically vanishing divisor already is); (d) keep the reserved names in one place
and cite `parser.rs` and S1's kernel classes so the two zero tests agree. Add the six rows above as tests
of `test_zero_test_cases` (expected: Unresolved for each).

### B2. The zero-test caps bound memory, not time: one d of a two-slot form takes 27 s

The caps of 1.6 are per operation (10^6 term products per multiplication, 10^4 terms per result,
|k| <= 64). A `Pow` node performs up to 64 multiplications, each up to the cap, and nothing bounds the
total or the coefficient size. Measured on the prototype (debug, same caps, `$R/r_cost.rs`,
`$R/r_dcost.rs`):

| Input | Time | Result |
|---|---|---|
| `zero_test(s^64 - (s*1)^64)`, s = 1 + x + ... + x^49 | 53.1 s | Zero |
| `zero_test` of the same with 10 terms | 1.78 s | Zero |
| d of the 1-form (0, s^64) on R^2, s with 50 terms | 26.8 s (0.00 s with the zero test switched off) | NonZero, component kept |
| same, s with 20 terms | 3.95 s | |

The input is a tree of about 100 nodes. This contradicts the cost statements of 1.9 and 6.3 ("about
15 µs per nonzero non-constant independent component", "about 4x"), which hold only for the measured
low-degree inputs. F2 is a cost item, and the fix adds a new slow path to exactly the functions it fixes.

Required change: give each `zero_test` call one cumulative work budget (for example the sum of
term-product counts over the whole conversion, plus a bound on coefficient bit length), exhausted means
`Unresolved` (sound, since normalize then keeps the simplified expression). Optional cheap shortcuts:
`a - a` with structurally equal operands is Zero before conversion. Add a test that the 50-term case
returns within a fixed bound (Unresolved), and restate the cost claim with the worst case.

## Minor issues

1. **Neg and invariant I1.** I1 is stated structurally (odd slots hold `neg(v_I)`), but Neg writes the
   old value v into the odd slots, which is `neg(nv)` only in value. Either state I1 up to value (all
   readers only need value consistency, and the antisymmetry check already falls back to the zero test)
   or have Neg write `neg(nv)`.
2. **Compile order.** `tensor_field.rs:361` (`idx.to_flat(3)` in `test_multi_index`) also needs
   `.unwrap()`; the list in 3.4 omits it. `lie_derivative.rs:225` (`num_components()`) must change in plan
   step 4, not step 6, or the crate does not build between the steps.
3. **REPL replacement block (4.3).** On R^1 the block only exercises the p >= n and p + q > n shortcuts
   (d of a 1-form on R^1 is the component-less 2-form), so "d runs on R^1 here" overstates it: after MF1 no
   REPL test computes a nontrivial d until J1a fixes S9 and S14. Say so in the comment; the library tests
   carry the real coverage.
4. **J1b recommendation 3.** Mapping `component_count`'s Err to "more than usize::MAX" is wrong for counts
   that fit in usize but exceed `isize::MAX` bytes (n^k between 2^57 and 2^64 for 64-byte slots). Use "too
   many to store" or report the count.
5. **`test_component_count_checked`.** The (2, 57) Err and (2, 56) Ok rows hold only for
   `size_of::<Expr>() == 64`. Derive the boundary from `size_of::<Expr>()` in the test, or assert the size.
6. **Facades left next to edited code.** `maps.rs:349-350` (pullback of p-forms, p > 1, returns its
   input) now also pays the n^p antisymmetry check. Turning it into an honest `Err(NotImplemented)` is a
   two-line change in this unit; otherwise make sure the HANDOFF entry of section 5 lands. The symplectic
   `is_non_degenerate` / `matrix_at` facade is already a tracked 🛑 row; mention it in section 5 since
   `symplectic.rs` is edited.
7. **Dense storage limit.** With n^p slots, `volume_form` on R^12 needs ω^4 = 12^8 slots (27.5 GB) and fails
   with `ComponentAllocationFailed` on this machine; Sage stores C(12, 6) = 924. Deferring antisymmetric
   storage is defensible for these items, but record the limit in section 5 and HANDOFF.
8. **Two zero tests.** S1 adds a three-valued zero test in rustmath-symbolic (`SeriesConst`). Note the
   overlap, and either place `expr_zero` in rustmath-symbolic later or keep the reserved-symbol semantics
   identical (see B1 (d)).
9. **`ContactForm::standard`.** It never calls `is_contact`, so "standard(1) would fail" (3.4) is
   inaccurate; the coordinate fix is still needed for `test_contact_volume`.
10. **Real fast path.** `Real(x)` with finite x != 0 gives NonZero. A float residue of inexact arithmetic
    (0.1 + 0.2 - 0.3 folded by simplify) is then "nonzero". Document that a `Real` is taken at its exact
    binary value.
11. **"Fails at HEAD".** Most new tests use new APIs (`zero_test`, `vanishes`, `from_independent_components`)
    and cannot compile on HEAD. Say "fails or does not compile on HEAD"; the prototype programs are the
    real HEAD evidence.

## Files

`$R/r_consts.rs` (B1, I), `$R/r_inf.rs` (B1, infinities), `$R/r_cost.rs` and `$R/r_dcost.rs` (B2),
`$R/r_alloc.rs` (allocation failure and `size_of::<Expr>()`), copies of the designer's `proto.rs` and
`t_*.rs` used unchanged.
