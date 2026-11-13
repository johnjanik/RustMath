# M1b-matrix-forms: multi-RHS IsConsistent, PrimaryRationalForm, JordanForm, IsSimilar

Unit M1b of the HANDOFF follow-ups (`docs/HANDOFF.md:333` at `9c24389`, "Not tracked as rows":
"the multi-RHS `IsConsistent`, `PrimaryRationalForm`, `JordanForm`, `IsSimilar`"). Crate: rustmath-matrix,
plus one trait in rustmath-polynomials and one trait impl in rustmath-finitefields (section 0.4 says why).
Runs after M1a (bounded rational form); section 0.3 is the contract between the two units.

Scratch: `$S = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/M1b-matrix-forms`.

| Path | Content |
|---|---|
| `$S/exp/forms.rs` | Prototype of every algorithm here, written against the public HEAD API (rlibs pinned in `$S/exp/pin`) |
| `$S/exp/t0.rs` .. `t10.rs`, `*.log` | Drivers and their output (HEAD behaviour, examples, timings) |
| `$S/gp/*.gp`, `*.log` | PARI/GP 2.15 reference scripts and output |
| `$S/gp/pif.g`, `pif.log` | GAP 4 primary invariant factors (Smith form of `xI - A` over K[x], then `Factors`) |
| `$S/dl/text586.txt` | MAGMA V2.x handbook page "Canonical Forms" (mirror `https://www.math.ru.nl/magma/text586.html`): intrinsic texts and example H51E8. It matches the digest's description of H26E9 (5×5 over GF(5), primary invariant factors, rational form, Smith form check), and a search-engine snippet of the current page shows the same first row `0, 2, 4, 2, 0` and a diagonal JordanForm `4, 4, 1, 1, 1` |
| `$S/dl/text587.txt`, `text489.txt`, `text512.txt` | MAGMA V2.x pages for `IsConsistent` / `Solution` (vector and sequence forms) |
| `$S/dl/sage_matrix2.txt` | Sage reference manual, `matrix2` (`jordan_form`, `is_similar` examples) |
| `$S/dl/a.txt` | R. Preusser, arXiv:2006.06502, text (Definition 5: generalised Jordan block) |

## 0. Summary

| Item | At HEAD | Fix |
|---|---|---|
| A. Multi-RHS `IsConsistent(A, W)` (W an r×n matrix), `IsConsistent(A, Q)` (Q a sequence), `Solution` of both | Missing: `linear_solve.rs:157/198/218` take one vector; tracker `ch026.IsConsistent.2` and `ch026.Solution.2` "missing". k systems cost k eliminations (n = k = 30 over Q: 11.6 s against 3.2 s for one elimination, debug) | One Gauss-Jordan elimination of `[A | B]` for all right-hand sides (Rouché-Capelli per column). New `Matrix::{solve_matrix, solve_left_matrix, solve_left_matrix_with_kernel, solve_left_sequence, solve_left_sequence_with_kernel}`; `solve` delegates to the shared helper (same output) |
| B. `PrimaryInvariantFactors`, `PrimaryRationalForm` | Missing (no function in the workspace; tracker "missing") | Factor the minimal polynomial (new trait `FactorizationField`: Q by Zassenhaus, GF(p) by Berlekamp), split each invariant factor by CRT, and build the cyclic bases of the primary components from the rational-form transform. Certified (`M·P = P·G`, P invertible) |
| C. `JordanForm` (split and non-split) | Missing: `eigenvalues.rs:253-381` is a numeric sketch inside a `/* */` block; the exported `JordanForm` struct (`eigenvalues.rs:387`, `lib.rs:49`) has no constructor | Same construction with the basis `x^t p^j q` of each primary component: the usual Jordan form when the minimal polynomial splits, MAGMA's generalised Jordan form otherwise. The dead sketch and the old struct go |
| D. `IsSimilar` with transform | Missing (no `is_similar` in rustmath-matrix) | Equal invariant factors decide (exactly); the transform is `T = P_B·P_A⁻¹` from the two rational-form transforms, certified by `T·P_A = P_B` |

Nothing is declared out of scope. Stated limits (documented, not hidden): fields only for item A, as for
the existing single-vector forms (MAGMA also allows Euclidean rings); `FactorizationField` is implemented for
Q and GF(p) (`PrimeField`); the generalised Jordan block layout for an irreducible factor of degree >= 2 could
not be checked against MAGMA output (section 0.2).

### 0.1 MAGMA specification (digest and the V2.x handbook text)

Digest `rustmath-lava/docs/Chapter_26_Matrices.md` §26.8 (`:222-225`) and §26.12.2 (`:299-312`), and
`Chapter_83_Matrix_Algebras.md` §83.11.2 (`:321-326`):

* `IsConsistent(A, W)`: A m×n, W a vector of length n or an r×n matrix. True iff `V·A = W` is consistent;
  then also a particular solution V and the nullspace N of A. `IsConsistent(A, Q)`: Q a sequence of vectors;
  true iff every `V[i]·A = Q[i]` is consistent; then the sequence V and N. `Solution(A, W)`, `Solution(A, Q)`:
  the same values, an error when inconsistent.
* `PrimaryRationalForm(A)`: (F, T, pairs) with `T·A·T⁻¹ = F`, F block diagonal of companion matrices of powers
  of irreducible polynomials, pairs = `PrimaryInvariantFactors(A)`.
* `JordanForm(A)`: (J, T, pairs), `T·A·T⁻¹ = J`, J the generalised Jordan form ("reduces to the usual Jordan
  form when the minimal polynomial splits over K").
* `IsSimilar(A, B)`: true iff similar, and then T with `T·A·T⁻¹ = B`.
* Algorithm in MAGMA: Steel [Ste97], A. Steel, *A new algorithm for the computation of canonical forms of
  matrices over fields*, J. Symbolic Comput. 24 (1997) 409-432. Not implemented here (the paper is not
  reachable from this container); this unit derives the primary forms from the rational form instead
  (section 2.2), which needs no new elimination code.

MAGMA H26E9 (V2.x number H51E8, `$S/dl/text586.txt`), verbatim output over GF(5):

```
a = [0 2 4 2 0; 2 2 2 3 3; 3 4 4 1 3; 0 0 0 0 1; 0 0 0 1 0]
PrimaryInvariantFactors(a) = [ <x + 1, 1>, <x + 1, 1>, <x + 4, 1>, <x + 4, 1>, <x + 4, 1> ]
r, t, f := RationalForm(a);  r = [1 0 0 0 0; 0 0 1 0 0; 0 1 0 0 0; 0 0 0 0 1; 0 0 0 1 0]
t = [1 3 0 2 1; 2 1 2 2 0; 3 4 3 4 1; 1 0 0 0 0; 0 2 4 2 0];  f = [ x + 4, x^2 + 4, x^2 + 4 ]
```

`$S/gp/h51e8.log`: `t*a*t^-1 == r` is 1 in PARI (MAGMA's row convention confirmed), `matdet(t) = 3`.

### 0.2 Conventions (decided here)

* **Transforms follow MAGMA**: every new transform satisfies `T·A·T⁻¹ = F` (row convention). The existing
  `RationalCanonicalForm::change_of_basis` keeps its column convention (`T⁻¹·A·T = C`); the new field is
  called `transform`, not `change_of_basis`, so the two cannot be confused. Sage's `jordan_form(transformation
  = True)` returns P with `P⁻¹·A·P = J`; our `T⁻¹` is such a P. Sage's `is_similar(B, transformation=True)`
  returns S with `A = S⁻¹·B·S`, the same as MAGMA's T.
* **Blocks follow MAGMA's `CompanionMatrix`** (ch. 23: 1s on the superdiagonal, last row `-c_0, ..., -c_{k-1}`),
  the transpose of the column-style `companion_matrix` of this crate.
  * Primary rational form: the block of `p^e` is `CompanionMatrix(p^e)`.
  * Generalised Jordan block of `p^e`, `p = x^d + c_{d-1}x^{d-1} + ... + c_0` irreducible: size `ed`, all
    superdiagonal entries 1, and for `j = 0..e-1` row `jd + d - 1` (0-based) holds `-c_0, ..., -c_{d-1}` in
    columns `jd .. jd + d - 1`. Equivalently: e copies of `CompanionMatrix(p)` on the diagonal, consecutive
    copies joined by a 1 in the bottom-left corner of the block above the diagonal. For d = 1 this is the usual
    Jordan block `J_e(λ)` (λ on the diagonal, 1s on the superdiagonal), the Sage and textbook layout. For
    `(x^2 + 1)^2` over Q it is `[0,1,0,0; -1,0,1,0; 0,0,0,1; 0,0,-1,0]`.
  * Source of the d >= 2 layout: it is the transpose of the column-convention generalised Jordan block
    `J(P^q)` of R. Preusser, arXiv:2006.06502, Definition 5 and Theorem 6 (`$S/dl/a.txt:134-170`: column-style
    companion `[P]` on the diagonal and ξ, "a 1 at position (1, n)", below it), who cites P. B. Bhattacharya,
    S. K. Jain, S. R. Nagpaul, *Basic Abstract Algebra*, 2nd ed., Part V, Ch. 21, §5. MAGMA's printed layout
    for d >= 2 could not be checked: the current handbook needs a login, the V2.x mirror and H26E9 only show
    linear factors, and no MAGMA binary is available. The docs say so.
* **Block order** (MAGMA's order is only visible in H26E9, which this order reproduces exactly):
  1. the irreducible factors `p_1, p_2, ...` of the minimal polynomial in the order of
     `FactorizationField::factor_monic_polynomial`, stable-sorted by degree. Over GF(p) this is PARI
     `factormod` order (degree, then coefficients from the top down as integers in [0, p)), which gives
     `x + 1` before `x + 4` in H26E9. Over Q it is PARI `factor` order (degree, then the primitive integer
     coefficients from the top down: `x - 3, x + 3, 2x - 1`, `$S/gp/order.log`). MAGMA's order over Q is not
     documented; the docs say so.
  2. for each `p_k`, the blocks `p_k^e` by increasing e (each divides the next, the order of `RationalForm`),
     each repeated as often as it occurs.
  `primary_invariant_factors` lists the pairs `(p_k, e)` in exactly this block order.

### 0.3 Coordination with M1a (bounded rational form, implemented just before)

M1b consumes only these documented contracts of `companion.rs`/`krylov.rs`/`polynomial_matrix.rs`:

| Item | Contract used |
|---|---|
| `invariant_factors(&M) -> Result<Vec<UnivariatePolynomial<F>>>` | monic, `f_1 \| f_2 \| ... \| f_r`, product = charpoly, certified |
| `rational_canonical_form(&M) -> Result<RationalCanonicalForm<F>>` | `invariant_factors` as above; `change_of_basis = Some(T)` with `T⁻¹·M·T = canonical_matrix`; blocks column-style (`companion_matrix`) in the order of `invariant_factors`. This implies (from `M·T = T·C` alone) that the columns of block i are `g_i, M·g_i, ..., M^{k_i-1}·g_i` |
| `krylov::field_constants` (`krylov.rs:74`, pub(crate)) | `(z, one, neg_one)` bound to the field of the entries |
| `krylov::linearly_independent` (`krylov.rs:234`, pub(crate)) | exact independence test, pivots inverted with `Field::inverse` |
| `polynomial_matrix::div_rem_field` (`polynomial_matrix.rs:524`, pub(crate)) | division over a field |

If M1a keeps these (its prototype `M1a-matrix-rcf-bound/exp/frob.rs` still uses `field_constants`, `mat_vec`,
`div_rem_field`), M1b needs no change. If M1a renames a crate-private helper, the implementer follows the
rename. If M1a changes the block layout of `canonical_matrix` (it must not: the field doc pins it), the one
function that reads `change_of_basis` (`primary_columns`, section 2.3) must follow, and the certificate turns
any mismatch into `Err`, never a wrong value. Tests of this unit never pin a transform, only its defining
property, so a different (valid) RCF transform from M1a changes no expected value.

### 0.4 Where the factorization capability lives

The primary forms need "a field allowing factorization of univariate polynomials" (MAGMA's precondition).
`rustmath-rings` has such a trait (`FactorableConstantField`, `function_field/typed/factor.rs:20`) but rings
depends on matrix, so matrix cannot use it. Placing a new trait in matrix would force rustmath-finitefields
(which has matrix only as a dev-dependency, `rustmath-finitefields/Cargo.toml:16-17`) to take matrix as a
normal dependency to implement it for `PrimeField`. The trait therefore goes to **rustmath-polynomials**, which
defines `UnivariatePolynomial`, already depends on rustmath-rationals, has the Q factorizer, and is a normal
dependency of both matrix and finitefields. No new crate edges. HANDOFF gets a line: `FactorableConstantField`
(rings) can become a thin alias of `FactorizationField` later.

---

## 1. Item A: multi-RHS `IsConsistent` / `Solution`

### 1.1 Defect at HEAD

`rustmath-matrix/src/linear_solve.rs`: `solve(&self, b: &[F])` (`:157`, `A·x = b`), `solve_left(&self, b: &[F])`
(`:198`, `x·A = b`), `solve_left_with_kernel(&self, b: &[F])` (`:218`). No matrix W and no sequence Q.
Tracker rows `ch026.IsConsistent.2` and `ch026.Solution.2` are "missing" (`docs/magma/status_overrides.csv:3151`,
`:3153`); `ch026.IsConsistent` and `ch026.Solution` say "one vector w, no sequence form" (`:3150`, `:3152`).
The workaround (one `solve_left` per row) repeats the elimination of A for every row
(`$S/exp/t7.log`, Q, entries in [-9, 9], debug):

| n, k | one elimination of `[Aᵀ | Wᵀ]` | k × `solve_left` | same V |
|---|---|---|---|
| 20, 20 | 0.46 s | 1.10 s | yes |
| 30, 30 | 3.2 s | 11.6 s | yes |

### 1.2 Fix

Algorithm: Gauss-Jordan elimination of the augmented matrix `[A | B]` (B with k columns) and the
Rouché-Capelli criterion column by column (K. Hoffman, R. Kunze, *Linear Algebra*, 2nd ed., §1.4, cited in the
module header already). The left block of `RREF [A | B]` is `RREF A` (pivot choice in a column depends only on
that column), so all k systems are consistent iff no pivot lies in a B column, i.e. iff the last pivot is
`< n`. Then the particular solution of column j has `X[c][j] = R[r][n + j]` at pivot row r with pivot column c,
and every free variable 0, exactly as `solve` does for one column. The row forms are the transposes:
`V·A = W` iff `Aᵀ·Vᵀ = Wᵀ`. The nullspace N of A (MAGMA's second value) is `self.transpose().kernel()` (the
existing `solve_left_with_kernel` pattern; computed only when consistent).

API (in `impl<F: Field> Matrix<F>`, `linear_solve.rs`, after `solve_left_with_kernel`):

```rust
/// Shared body of `solve` and `solve_matrix`: RREF of [A | B], None iff some column of B is inconsistent.
fn solve_columns(&self, b: &Matrix<F>) -> Result<Option<Matrix<F>>>;

/// A·X = B for an m×k matrix B (column convention; Sage `solve_right` with a matrix).
/// Ok(Some(X)), X n×k, iff every column is consistent; X has every free variable 0 (column j of X is what
/// `solve` returns for column j of B). Ok(None) iff some column is inconsistent.
/// Err(InvalidArgument) if b.rows() != self.rows().
pub fn solve_matrix(&self, b: &Matrix<F>) -> Result<Option<Matrix<F>>>;

/// MAGMA `IsConsistent(A, W)` with an r×n matrix W (Handbook ch. 26 §26.8): V·A = W.
/// Ok(Some(V)), V r×m, iff every row of W is consistent (row i of V is what `solve_left` returns for row i);
/// Ok(None) otherwise. Err(InvalidArgument) if w.cols() != self.cols().
pub fn solve_left_matrix(&self, w: &Matrix<F>) -> Result<Option<Matrix<F>>>;

/// MAGMA `Solution(A, W)` and the extra values of `IsConsistent(A, W)`: Ok(Some((V, N))), N a basis of the
/// left nullspace {u : u·A = 0}, so every solution is V + (rows in span N). Ok(None) iff inconsistent (MAGMA
/// raises an error; this returns the documented None, as `solve_left_with_kernel` does).
pub fn solve_left_matrix_with_kernel(&self, w: &Matrix<F>) -> Result<Option<(Matrix<F>, Vec<Vec<F>>)>>;

/// MAGMA `IsConsistent(A, Q)` for a sequence Q of vectors of length n: Ok(Some(V)) with V[i]·A = Q[i] for
/// every i, iff all are consistent; Ok(None) otherwise. Q empty: Ok(Some(vec![])).
/// Err(InvalidArgument) naming the index if some Q[i].len() != self.cols().
pub fn solve_left_sequence(&self, q: &[Vec<F>]) -> Result<Option<Vec<Vec<F>>>>;

/// MAGMA `Solution(A, Q)`: Ok(Some((V, N))) as above, Ok(None) iff inconsistent.
pub fn solve_left_sequence_with_kernel(&self, q: &[Vec<F>]) -> Result<Option<(Vec<Vec<F>>, Vec<Vec<F>>)>>;
```

Implementation notes:
* `solve_columns`: build the m×(n+k) augmented data row by row, `reduced_row_echelon_form()`, return None iff
  `rref.pivots.last().is_some_and(|&c| c >= n)`; `X = vec![F::zero(); n*k]` and `X[c*k + j] = R[r][n+j]`.
  `F::zero()` for the free variables keeps `solve` bit-identical to HEAD (`linear_solve.rs:181-186`).
* `solve(b)` becomes: the HEAD length check and message ("Vector length must match number of rows"), then
  build the m×1 matrix of b, `solve_columns`, return column 0. Same augmented matrix,
  same pivots, same values as HEAD, so no behaviour change (the existing tests `solve_*` and
  `matrix_forms_gfp.rs::solve_gf7` guard it).
* `solve_left_matrix(w)`: shape check, then `self.transpose().solve_matrix(&w.transpose())?.map(|x| x.transpose())`.
* `solve_left_sequence(q)`: check each length (message names i), build the `q.len()`×n matrix, call
  `solve_left_matrix`, split rows. Every shape works (0 rows, 0 columns; `$S/exp/t5.log`).
* Docs: module header (`linear_solve.rs:1-12`) lists the four MAGMA forms and their methods; `solve_left`'s
  doc points to the matrix and sequence forms. Exactness note as for `solve`: decided exactly over an exact
  field, no claim for floating-point `Field` impls. Fields only (MAGMA also allows Euclidean rings; not this
  item).

Correctness: as for `solve` (one column), applied to all columns of one elimination: row operations act on
whole rows, so `RREF [A | B]` restricted to `[A | b_j]` is a row-echelon form of `[A | b_j]` whose pivots in A
are those of `RREF A`; `b_j` is consistent iff column `n + j` is not a pivot column; if no B column is a pivot
column, the pivot rows give `x_c + Σ_free R[r][f]·x_f = R[r][n+j]`, so the free-zero vector solves column j.

### 1.3 Callers and behaviour changes

`Matrix::solve` callers (grep `\.solve(` restricted to `Matrix`): `rustmath-matrix/src/vector_space.rs:144`,
`:237`, `:362`, `companion.rs:1086` (test), `rustmath-rings/src/dedekind/nfctx.rs:243`,
`rustmath-jupyter/src/repl.rs:1435`, `rustmath-finitefields/tests/matrix_forms_gfp.rs`. Output identical, so
none changes. `solve_left`, `solve_left_with_kernel`: unchanged. New names collide with nothing (grep
`solve_matrix\|solve_left_matrix\|solve_left_sequence`: no hit).

### 1.4 Tests (`linear_solve.rs` tests over Q; GF(3) in `matrix_forms_gfp.rs`). All fail on HEAD (no API).

| Test | Input | Expected | Reference |
|---|---|---|---|
| `solve_left_matrix_two_rhs` | A = [1,2;2,4;0,1], W = [3,7;1,3] | V = [3,0,1;1,0,1]; V·A = W | gp `matinverseimage(A~,W~)~` = `[3,0,1;1,0,1]` (`$S/gp/check1.log`) |
| `solve_left_matrix_with_kernel_two_rhs` | same | N = [[-2,1,0]]; `(V[i] + N[0])·A = W[i]` | gp `matker(A~)` = `[-2,1,0]~` |
| `solve_left_matrix_one_bad_row` | A = [1,2;2,4], W = [3,6;1,3] | None; `solve_left([3,6]) = Some([3,0])`, `solve_left([1,3]) = None` | gp `matinverseimage(A~,W~)` = `[;]`, rows `[3,0]~` and `[]~` (`$S/gp/more.log`) |
| `solve_left_matrix_unique` | A = [2,1,1;1,3,2;1,0,0] (det -1), W = [4,5,3;1,1,1] | V = [-1,2,4;1,0,-1] | gp `W*A^-1` (`more.log`) |
| `solve_matrix_column_convention` | A = [1,2,0;2,4,1], B = [3,1;7,3] | X = [3,1;0,0;1,1] | gp `matinverseimage(A,B)` (`more.log`) |
| `solve_left_sequence_matches_rows` | A, rows of W from the first test; then [[3,6],[1,3]] for [1,2;2,4]; a length-1 vector | Some(rows of V); None; Err(InvalidArgument) | as above |
| `solve_left_matrix_shapes` | W 2 cols for a 3-col A; W 0×2; A 0×2 with W = [0,0] and [0,1]; A 2×0 with W 3×0 | Err; Some(0×3); Some(1×0), None; Some(3×2 zero) and N has 2 vectors | `$S/exp/t5.log` |
| `solve_left_matrix_gf3_all_solutions` (finitefields; MAGMA H26E8 is "enumerating all solutions to V·X = W over GF(3)", whose data is not in the digest) | A = [1,0,1;0,1,1;1,1,2;2,1,0], W = [1,1,2;2,0,2] mod 3 | V = [1,1,0,0;2,0,0,0], N = [[2,2,1,0],[1,2,0,1]]; brute force over all 81 v: each row has exactly 9 solutions and they are `V[i] + c1·N1 + c2·N2`; W2 = [1,1,2;1,0,0] gives None | gp `$S/gp/gf3_multi.gp`: V, `matker(A~)`, counts [9, 9], `[;]` |

The existing `solve_*` tests stay unchanged and green.

### 1.5 Cost

One elimination of an m×(n+k) matrix: O(m·n·(n+k)) field operations, against O(k·m·n²) for k separate
solves. Table in 1.1. Over Q the entries grow as in `solve` (no Dixon lifting; tracker "missing features").

---

## 2. Item B: `PrimaryInvariantFactors`, `PrimaryRationalForm`

### 2.1 Defect at HEAD

No function anywhere (grep `primary_rational\|primary_invariant\|PrimaryRationalForm`: only docs and the
tracker). Tracker `ch026.PrimaryRationalForm`, `ch026.PrimaryInvariantFactors` "missing [review 2026-11-06;
checked]" (`status_overrides.csv:3188`, `:3191`); `ch083.*` rows inferred missing (`intrinsics.csv:7207`, `:7210`).

### 2.2 Algorithm

Theory: the elementary divisors (primary invariant factors) of A are the prime powers `p^e` with
`e = v_p(f_i) > 0` over all invariant factors `f_i` and irreducible p (D. S. Dummit, R. M. Foote, *Abstract
Algebra*, 3rd ed., §12.1, Theorem 6 and the passage from invariant factors to elementary divisors via the
Chinese Remainder Theorem; §12.2, rational canonical form; N. Jacobson, *Basic Algebra I*, 2nd ed., §3.9-3.10).
Construction:

1. Column convention on `M = Aᵀ` (row-convention forms of A are column-convention forms of Aᵀ; section 2.4).
   `rcf = rational_canonical_form(&M)`: invariant factors `f_1 | ... | f_r` and `P_R` with
   `P_R⁻¹·M·P_R = ⊕ C(f_i)`. Block i of `P_R` has columns `M^s·g_i`, `s < k_i = deg f_i`, and `ann(g_i) = f_i`.
   In that basis a vector `h(M)·g_i` with `deg h < k_i` has coordinates = the coefficients of h, so
   `h(M)·g_i = Σ_s h_s·P_R[:, off_i + s]` with no evaluation at M.
2. Factor the minimal polynomial `f_r` once: `p_1, ..., p_s` (every `f_i` divides `f_r`, so its prime factors
   are among these). For each i and k: `e_ik = v_{p_k}(f_i)` by repeated exact division, then check that
   `f_i / Π_k p_k^{e_ik}` is the constant 1 (else `Err(InvalidOperation)`: the oracle broke its contract).
3. For each block `(k, e = e_ik > 0, i)` in the order of 0.2: `q = f_i / p_k^e` (exact). The vectors
   `(x^t·q)(M)·g_i`, `t < e·deg p_k`, form a basis of the cyclic subspace `Z(q(M)·g_i)`, whose annihilator is
   `f_i / gcd(f_i, q) = p_k^e`; `deg(x^t q) <= deg f_i - 1`, so no reduction modulo `f_i` is needed. By CRT,
   `F[x]/(f_i) = ⊕_k q_ik·F[x]/(f_i)`, so these subspaces over k form a direct sum equal to `Z(g_i)`, and over
   all i they give a basis of `K^n`. In that basis M acts on block `(k, e, i)` as the column-style companion
   matrix of `p_k^e`.
4. `P = [all these columns]`, `G = ⊕ companion_col(p_k^e)`. Certificate: `M·P == P·G` and
   `krylov::linearly_independent(columns of P)`. Then `P⁻¹·M·P = G` is proven, independently of steps 1-3.
5. MAGMA convention (section 2.4): `F = Gᵀ` (row-style blocks `CompanionMatrix(p^e)`), `T = Pᵀ`, and
   `T·A·T⁻¹ = (P⁻¹·Aᵀ·P)ᵀ = Gᵀ = F`.

Canonicity: G is determined by the multiset of `p^e` and the order of 0.2, so (given the factorization order)
`primary_rational_form(A).canonical_matrix` depends only on the similarity class of A.

### 2.3 API and data structures

**rustmath-polynomials** (new file `src/factorization_field.rs`; `pub mod factorization_field;` and
`pub use factorization_field::FactorizationField;` in `src/lib.rs`):

```rust
//! Header: MAGMA ch. 23 §23.8.1 `Factorization` (digest Chapter_23_Univariate_Polynomial_Rings.md:294) and the
//! precondition of ch. 26 §26.12.2 ("a field allowing factorization of univariate polynomials").
//! Q: Zassenhaus (H. Zassenhaus, J. Number Theory 1 (1969) 291-311) through `factor_over_integers`.
//! GF(p): implemented in rustmath-finitefields for `PrimeField` (Berlekamp).

/// A field over which univariate polynomials factor into monic irreducibles.
pub trait FactorizationField: Field {
    /// The factorization of a monic `f` of positive degree.
    ///
    /// Ok([(g_1, e_1), ..., (g_k, e_k)]): every g_i monic, irreducible, of positive degree, pairwise distinct,
    /// e_i >= 1, and f = Π g_i^e_i exactly (each impl multiplies back and compares). Order: by degree first,
    /// deterministic; each impl states the rest.
    /// Err(InvalidArgument) if f is zero, constant or not monic. Errors of the underlying factorizer pass
    /// through (never a partial list). Err(InvalidOperation) if the multiply-back check fails.
    fn factor_monic_polynomial(f: &UnivariatePolynomial<Self>)
        -> Result<Vec<(UnivariatePolynomial<Self>, usize)>>;
}

impl FactorizationField for Rational { ... }
```

`Rational` impl: D = lcm of the denominators; `factor_over_integers(D·f)` (`factorization.rs:424`, a stable
API across P1, which only rewrites its error text); drop degree-0 entries (the content); sort by (degree,
coefficient vector from the top down compared as `Integer`s), which is PARI `factor` order; divide each
factor by its leading coefficient; multiply back and compare with f. Zassenhaus' recombination cap
(`MathError::NotSupported`) passes through.

**rustmath-finitefields** (new private module `src/prime_field_factor.rs`; `mod prime_field_factor;` in
`src/lib.rs`):

```rust
//! Header: MAGMA ch. 23 §23.8.1 over GF(p); E. R. Berlekamp, Math. Comp. 24 (1970) 713-735, through
//! `rustmath_polynomials::berlekamp::factor` (certified factor count dim ker(Q - I), PARI factormod order).
impl FactorizationField for PrimeField { ... }
```

p = the modulus of the first coefficient whose modulus is not 0; none (every coefficient an unbound
`Ring::zero()`/`one()` constant): `Err(InvalidArgument("... the prime p is unknown: every coefficient is an
unbound integer constant"))`. Coefficients are passed as `value()` integers (berlekamp reads them modulo p);
`berlekamp::factor(&coeffs, &p)` checks that p is prime (`Err(InvalidArgument)` otherwise) and returns
`(u, list)` in PARI `factormod` order; each factor becomes `PrimeField::new(c, p)`; multiply back.

**rustmath-matrix** (new file `src/primary_forms.rs`, `pub mod primary_forms;` in `lib.rs`):

```rust
//! Header: MAGMA ch. 26 §26.12.2 (`PrimaryRationalForm`, `JordanForm`, `PrimaryInvariantFactors`,
//! `IsSimilar`) and ch. 83 §83.11.2, digests Chapter_26_Matrices.md:299-312, Chapter_83_Matrix_Algebras.md:321-326.
//! Algorithm: from the rational form and the factored minimal polynomial (Dummit-Foote §12.1-12.3;
//! Jacobson BA I §3.9-3.10); generalised Jordan block: Preusser arXiv:2006.06502 Def. 5 (transposed);
//! MAGMA uses [Ste97] (not implemented). Conventions of section 0.2 stated in full.

pub use rustmath_polynomials::FactorizationField;

#[derive(Debug, Clone)]
pub struct PrimaryRationalForm<F: Field> {
    /// MAGMA's third value: (p, e) per block, p monic irreducible, block = CompanionMatrix(p^e).
    pub primary_invariant_factors: Vec<(UnivariatePolynomial<F>, usize)>,
    /// MAGMA's F.
    pub canonical_matrix: Matrix<F>,
    /// MAGMA's T: T·A·T⁻¹ = canonical_matrix.
    pub transform: Matrix<F>,
}

#[derive(Debug, Clone)]
pub struct JordanForm<F: Field> { /* same three fields; canonical_matrix = generalised Jordan form */ }

/// MAGMA `PrimaryInvariantFactors(A)`: from `invariant_factors(A)` and one factorization, no transform.
pub fn primary_invariant_factors<F: FactorizationField>(a: &Matrix<F>)
    -> Result<Vec<(UnivariatePolynomial<F>, usize)>>;
pub fn primary_rational_form<F: FactorizationField>(a: &Matrix<F>) -> Result<PrimaryRationalForm<F>>;
pub fn jordan_form<F: FactorizationField>(a: &Matrix<F>) -> Result<JordanForm<F>>;
pub fn is_similar<F: Field>(a: &Matrix<F>, b: &Matrix<F>) -> Result<Option<Matrix<F>>>;
```

Private pieces (names are suggestions):

```rust
/// Square check, binding of the entries, field probe (section 5). n == 0 is answered by the callers first.
fn prepared<F: Field>(a: &Matrix<F>, what: &str) -> Result<(Matrix<F>, F, F)>; // (bound A, z, one)

struct PrimaryBlock<F: Field> { p: UnivariatePolynomial<F>, e: usize, i: usize, q: UnivariatePolynomial<F> }

/// Steps 2-3 of 2.2: blocks in the order of 0.2 (stable sort by degree of the oracle's list, then (k, e, i)).
fn primary_blocks<F: FactorizationField>(fs: &[UnivariatePolynomial<F>], one: &F) -> Result<Vec<PrimaryBlock<F>>>;

enum Shape { Companion, Jordan }

/// Column convention on M (bound, square, n >= 1): (pairs, G, P) with M·P = P·G certified.
fn primary_core<F: FactorizationField>(m: &Matrix<F>, shape: Shape)
    -> Result<(Vec<(UnivariatePolynomial<F>, usize)>, Matrix<F>, Matrix<F>)>;

/// The columns h(M)·g_i = Σ_s h_s·P_R[:, off_i + s] for the polynomials h of one block (deg h < k_i checked).
fn primary_columns<F: Field>(p_r: &Matrix<F>, off: usize, k: usize, hs: &[UnivariatePolynomial<F>], z: &F)
    -> Result<Vec<Vec<F>>>;
```

Details the implementer must keep:
* All constants in G (and so in F, J) are the bound `z`, `one` of `field_constants`, never `F::zero()`, so a
  `PrimeField` result has every entry bound. Build G by hand; do not call `companion_matrix` (it uses the
  `R::zero()`/`R::one()` sentinels).
* Validate the oracle output in `primary_blocks`: each factor monic (`leading_coefficient().is_one()`), degree
  >= 1, multiplicity >= 1, pairwise distinct; plus the per-`f_i` division check of step 2.
* Quotients and powers: `div_rem_field` with a zero-remainder check; `p^e` by repeated multiplication starting
  from the constant `one`.
* `n == 0`: `primary_invariant_factors` returns `[]`; the forms return empty pairs and 0×0 matrices.

Errors (all functions): `Err(InvalidArgument)` for a non-square A, and for entries that are not in a field
(section 5); errors of `rational_canonical_form`, `invariant_factors` and the factorization pass through;
`Err(InvalidOperation("internal: ..."))` for a broken oracle contract or a failed certificate. Never a value
that was not certified.

### 2.4 Why `Aᵀ`

MAGMA's `T·A·T⁻¹ = F` means: the rows `t_i` of T satisfy `t_i·A = Σ_j F_ij t_j`, i.e. A acting on row vectors
from the right. That is the column convention for `Aᵀ`: `P⁻¹·Aᵀ·P = G` gives `Pᵀ·A·(Pᵀ)⁻¹ = Gᵀ`. So the core
runs once on `M = Aᵀ` and the wrapper transposes G and P. `Aᵀ` is similar to A, so the pairs are the same.

### 2.5 Tests (all new, all fail on HEAD: no API)

`primary_forms.rs` tests (Q; helper `assert_magma_transform(a, f, t)`: `t·a == f·t` and `t.rank() == n`).
Transforms are never pinned (they depend on the RCF transform, which M1a may change). The matrix G below is
`U·C·U⁻¹` with `C = [0,0,0,-1,0; 1,0,0,0,0; 0,1,0,-2,0; 0,0,1,0,0; 0,0,0,0,1]` (column-style companion of
`(x²+1)² = x⁴+2x²+1`, then [1]) and the unimodular `U = [1,2,0,1,0; 0,1,1,0,0; 0,0,1,3,1; 0,0,0,1,2; 0,0,0,0,1]`
(`$S/exp/t1.rs`).

| Test | Input | Expected | Reference |
|---|---|---|---|
| `primary_invariant_factors_q` | G = [2,-4,5,-18,31; 1,-1,1,-6,11; 0,1,2,-8,15; 0,0,1,-3,7; 0,0,0,0,1] (= U·(C((x²+1)²) ⊕ [1])·U⁻¹); Sage `m` = [5,4,2,1; 0,1,-1,-1; -1,-1,3,0; 1,1,-1,2]; Sage `a` = [1,0,0,0; 0,1,0,0; 1,-1,1,0; 1,-1,1,2] | [(x-1,1),(x²+1,2)]; [(x-4,2),(x-2,1),(x-1,1)]; [(x-2,1),(x-1,1),(x-1,2)] | gp `pif(A)` (factor of each `matfrobenius(A,1)` entry, `$S/gp/check1.gp`); GAP `$S/gp/pif.log` |
| `primary_rational_form_q` | G; Sage `m` | F = [1,0,0,0,0; 0,0,1,0,0; 0,0,0,1,0; 0,0,0,0,1; 0,-1,0,-2,0]; F = [0,1,0,0; -16,8,0,0; 0,0,2,0; 0,0,0,1]; transform property | gp `T*A*T^-1 == F` for the prototype's T (`check1.log`), `factor(charpoly(F))` |
| `primary_forms_agree` | the three matrices | `primary_invariant_factors(A) == jordan_form(A).primary_invariant_factors == primary_rational_form(A).primary_invariant_factors` | same |
| `primary_forms_shapes` | 2×3 matrix; 0×0 | Err(InvalidArgument); `[]`, 0×0 forms | |

`rustmath-polynomials` (`factorization_field.rs` tests):

| Test | Input | Expected | Reference |
|---|---|---|---|
| `rational_factorization_pari_order` | monic `(x-3)(x+3)(x-1/2)(x²+1)²` | [(x-3,1),(x+3,1),(x-1/2,1),(x²+1,2)] | gp `factor((x-3)*(x+3)*(2*x-1)*(x^2+1)^2)` = `[x-3,1; x+3,1; 2*x-1,1; x^2+1,2]` (`$S/gp/fac.log`) |
| `rational_factorization_charpoly` | x⁴ - 11x³ + 42x² - 64x + 32 | [(x-4,2),(x-2,1),(x-1,1)] | gp `factor(...)` (`fac.log`) |
| `rational_factorization_rejects` | 0; 1; 2x + 1 | Err(InvalidArgument) each | |

`rustmath-finitefields` (`prime_field_factor.rs` unit tests):

| Test | Input | Expected | Reference |
|---|---|---|---|
| `prime_field_factorization_matches_factormod` | x⁵+x⁴+x+1 mod 5; (x+1)²(x²+2) mod 5 | [(x+1,1),(x²+2,1),(x²+3,1)]; [(x+1,2),(x²+2,1)] | gp `factormod(...)` (`fac.log`) |
| `prime_field_factorization_rejects` | x + 1 from `PrimeField::one()` only; x² + 1 mod 15 | Err(InvalidArgument) (p unknown); Err (berlekamp: 15 not prime) | |

### 2.6 Cost

`RCF(Aᵀ)` (whatever M1a makes it) + one factorization of the minimal polynomial (Zassenhaus over Q,
Berlekamp over GF(p)) + building P (`O(n·Σ k_i²) <= O(n³)` field operations) + certificate (two products and
one elimination, `O(n³)`). Measured with the prototype on HEAD (debug libraries, `$S/exp/t2.log`, `t4.log`):

| Input | RCF(Aᵀ) | Jordan form (total) |
|---|---|---|
| Q, dense n = 8 / 12 / 16, entries in [-5, 5] | 4 / 38 / 212 ms | 9 / 59 / 356 ms |
| Q, conjugated Jordan structure n = 16 (6 blocks) | 78 ms | 114 ms |
| GF(101), GF(10⁹+7), GF(~10²⁴), dense n = 20 | | 0.15 s, 0.19 s, 0.97 s |

---

## 3. Item C: `JordanForm`

### 3.1 Defect at HEAD

`rustmath-matrix/src/eigenvalues.rs:22-383` is one `/* ... */` block ("Commented out: Requires from_f64");
inside it `jordan_form(&self, max_iterations, tolerance)` (`:253-381`) is a QR-iteration sketch that would only
try `k <= 3` generalised eigenvectors and set superdiagonal 1s by adjacency, so it could not be right for
non-diagonalizable input even if compiled. The struct `JordanForm { j, p }` (`:385-392`) is exported
(`lib.rs:49`) and has no constructor (tracker `PROJECT_TRACKER.md:940`, `:979`; `ch026.JordanForm` "missing",
`status_overrides.csv:3189`; `ch083.JordanForm` "candidate" pointing at the dead code, `intrinsics.csv:7208`).

### 3.2 Algorithm

As 2.2 with another basis of each primary component: for the block `(p, e, i)`, `d = deg p`,
`q = f_i / p^e`, the polynomials `h_{jd+t} = x^t·p^j·q` for `j < e`, `t < d` (in this order). They form a basis
of `q·F[x]/(f_i)` because `{x^t p^j : j < e, t < d}` is a basis of the polynomials of degree `< ed`
(p-adic expansion) and `h ↦ h·q` maps `F[x]/(p^e)` isomorphically onto `q·F[x]/(f_i)`. Also
`deg h_{jd+t} <= deg f_i - 1`. The action of x (column convention):
`x·x^t p^j q = x^{t+1} p^j q` for `t < d-1`, and `x·x^{d-1} p^j q = p^{j+1} q - Σ_s c_s x^s p^j q` (with
`p^e q = f_i ≡ 0`). So column-convention G has every subdiagonal entry 1 inside the block and, for each j,
column `jd + d - 1` holds `-c_0..-c_{d-1}` in rows `jd .. jd+d-1`. This is Preusser's `J(P^q)` (section 0.2).
Its transpose is the layout of 0.2. Same certificate as 2.2, then `J = Gᵀ`, `T = Pᵀ`.

When every irreducible factor is linear (`p = x - λ`, the minimal polynomial splits), J is the usual Jordan
form with 1s on the superdiagonal, MAGMA's "usual Jordan form" case; otherwise it is MAGMA's generalised form.
No extension field is used (Sage instead fails with "Some eigenvalue does not exist in Rational Field",
`$S/dl/sage_matrix2.txt:12455-12530`).

### 3.3 API

`jordan_form` (2.3). The old `eigenvalues::JordanForm { j, p }` is deleted and the new struct is exported from
`lib.rs` under the same name. The commented-out `jordan_form` sketch (`eigenvalues.rs:253-381`) is deleted;
the rest of the comment block (power iteration, QR eigenvalues, eigenvectors) is not this unit's and stays.
The module doc of `eigenvalues.rs` gets one line: the exact Jordan form is `primary_forms::jordan_form`.

### 3.4 Callers

`JordanForm` (grep `JordanForm\|jordan_form` over the workspace): only `lib.rs:49` and the dead code. No user
of the fields `j`, `p`. The public type changes its fields; nothing else changes.

### 3.5 Tests (all new, fail on HEAD)

| Test | Input | Expected | Reference |
|---|---|---|---|
| `jordan_form_sage_examples` | Sage `a`; Sage `m` (2.5) | J = [2,0,0,0; 0,1,0,0; 0,0,1,1; 0,0,0,1]; J = [4,1,0,0; 0,4,0,0; 0,0,2,0; 0,0,0,1]; transform property | Sage `jordan_form` docs (`sage_matrix2.txt:12506-12520`, `:12605-12620`): blocks {[2], J₂(1), [1]} and {[2], [1], J₂(4)} (Sage orders blocks differently; the test compares our exact J and the block multiset); gp ranks `matrank((A-l)^k)` = [2,1,1] at l = 1 (`check1.log`) |
| `jordan_form_generalized_block_q` | G | J = [1,0,0,0,0; 0,0,1,0,0; 0,-1,0,1,0; 0,0,0,0,1; 0,0,0,-1,0], pairs [(x-1,1),(x²+1,2)] | gp `factor(minpoly([0,1,0,0;-1,0,1,0;0,0,0,1;0,0,-1,0]))` = `(x^2+1)^2`, `T*G*T^-1 == J` (`check1.log`) |
| `jordan_form_sage_similarity_matrices` | Sage A3 = [19,-7,-29; -16,11,30; 15,-7,-25], B3 = [-38,-63,42; 14,25,-14; -14,-21,18] | J(A3) = [4,1,0; 0,4,0; 0,0,-3], pairs [(x-4,2),(x+3,1)]; J(B3) = diag(4,4,-3), pairs [(x-4,1),(x-4,1),(x+3,1)] | gp `matfrobenius(A3,1)` = `[x^3-5*x^2-8*x+48]`, `matfrobenius(B3,1)` = `[x^2-x-12, x-4]` (`$S/gp/sage_sim.log`) |
| doc test on `jordan_form` | Sage `a` | the J above | |

GF(p) tests are in section 6.

---

## 4. Item D: `IsSimilar`

### 4.1 Defect at HEAD

No `is_similar` in the workspace (grep `fn is_similar`: none). Tracker `ch026.IsSimilar` "missing"
(`status_overrides.csv:3193`), `ch083.IsSimilar` inferred missing.

### 4.2 Algorithm and correctness

Two square matrices over a field are similar iff they have the same invariant factors (Dummit-Foote §12.2,
the similarity theorem for rational canonical forms). The invariant factors are computed exactly and
certified (`companion.rs:466` docs), so a negative answer is decided, not a search result.

```
is_similar(A, B):
  shape: both square, same n, else Err(InvalidArgument); n == 0: Ok(Some(0×0))
  (z, one) = field_constants over the entries of A and B; bind both (x ↦ x + z); field probe (section 5)
  if invariant_factors(A) != invariant_factors(B): return Ok(None)            // decided
  ra = rational_canonical_form(A); rb = rational_canonical_form(B)
  if ra.canonical_matrix != rb.canonical_matrix: Err(InvalidOperation("internal"))
  P_A, P_B = the change_of_basis matrices (T⁻¹·X·T = C)
  T = P_A.solve_left_matrix(P_B)  (T·P_A = P_B; None: Err(InvalidOperation("internal")))
  certificate: T·P_A == P_B, else Err(InvalidOperation)
  Ok(Some(T))
```

Proof of the certificate: the RCF contract gives invertible `P_A`, `P_B` with `P_A⁻¹·A·P_A = C = P_B⁻¹·B·P_B`
(certified inside `rational_canonical_form`). `T·P_A = P_B` gives `T = P_B·P_A⁻¹`, invertible, and
`T·A·T⁻¹ = P_B·P_A⁻¹·A·P_A·P_B⁻¹ = P_B·C·P_B⁻¹ = B`. One product suffices; the second check `T·A == B·T` would
add about 35% to the cost on Q (`$S/exp/t3.log`). Tests check `T·A == B·T` directly.

`F: Field` only: no factorization is needed (MAGMA asks for factorization because [Ste97] goes through the
primary form). For a rounding `Field` impl (`Real`, `Complex`) neither answer is a proof (same statement as
`invariant_factors`). Entries of A and B must lie in one field (`PrimeField` arithmetic panics on two different
moduli, as everywhere).

### 4.3 Tests (all new, fail on HEAD)

| Test | Input | Expected | Reference |
|---|---|---|---|
| `is_similar_sage_examples` | Sage ex. 1: A = [-5,2,-11; -6,7,-42; 0,1,-6], B = [1,12,3; -1,-6,-1; 0,6,1]; Sage ex. 2: A2 = [2,3,-3,-6; 0,1,-2,-8; -3,-3,4,3; -1,-2,2,6], B2 = [1,1,2,4; -1,2,-3,-7; -2,3,-4,-7; 0,-1,0,0]; Sage ex. 3: A3, B3 (3.5) | Some(T), `T·A == B·T`, T invertible, and Sage's S = [1,0,0; -2/3,1/6,-5/6; 2/3,0,-1/3] also has `S·A == B·S`; None; None | Sage `is_similar` docs (`sage_matrix2.txt:11750-11960`); gp `matfrobenius(A) == matfrobenius(B)` = 1, `T*A*T^-1 == B` = 1, charpolys of A2, B2 differ, A3/B3 same charpoly and different `matfrobenius` (`sage_sim.log`) |
| `is_similar_same_char_and_min_poly` | J₂(0)⊕J₂(0) and J₂(0)⊕0⊕0 | None | gp `matfrobenius` = `[x^2, x^2]` vs `[x^2, x, x]`, charpoly x⁴ and minpoly x² for both (`check1.log`) |
| `is_similar_to_own_forms` | G and C (2.5) | Some(T) with `T·G == X·T` for X = C, `jordan_form(G).canonical_matrix`, `primary_rational_form(G).canonical_matrix`, Gᵀ | gp `T*G*T^-1 == C` for the prototype's T (`check1.log`); a matrix is similar to its transpose |
| `is_similar_shapes` | non-square; 2×2 vs 3×3; 0×0 vs 0×0 | Err(InvalidArgument) twice; Some(0×0) | |

### 4.4 Cost

`2 × invariant_factors + 2 × RCF + one solve + one product`. Over Q the solve and the product dominate for
dense input because `P_A⁻¹` has large entries (`$S/exp/t3.log`, n = 16 dense, `B = J(A)`: invariant factors
0.22 s, RCFs 0.36 s, solve 2.1 s, product 3.0 s; T has entries of 264 digits). With the prototype's two
certificates the totals were 0.11 s / 1.6 s / 8.9 s at n = 8 / 12 / 16 (`t2.log`); one certificate saves
about a third. GF(p), n = 20: 0.2-0.7 s (`t4.log`). Documented as "no bound is claimed for the size of T over
Q" in the doc comment.

---

## 5. Shared input handling: binding and the field probe

`PrimeField` and the GF(p^n) types use unbound integer constants for `Ring::zero()`/`one()` (computed in Z).
Two cases matter (`$S/exp/t0.log`, `t4.log`):
* **Mixed entries** (some bound, some unbound, e.g. `PrimeField::one() + PrimeField::one()` next to bound
  entries mod 2): a raw unbound 2 is nonzero in Z but 0 in GF(2). Every new function first maps each entry
  `x ↦ x + z` (`z` from `field_constants`, bound as soon as one entry is bound), which binds it. Cost O(n²).
  Example: `[[2u, 1], [0, 2u]]` with bound 1, 0 mod 2 gives J = [0,1; 0,0] and pairs [(x, 2)] (gp
  `matfrobenius(Mod([2,1;0,2],2),1)` = `[x^2]`, `more.log`).
* **Only unbound entries**: the field is unknown. Probe: `two = one + one`; the arithmetic is a field's only
  if `two.is_zero() || two.inverse().is_ok()` (in a field 2 is 0 or a unit; in Z semantics 2 is neither:
  `inverse` gives `Err("No inverse exists: gcd(2, 0) = 2")`, `t0.log`). If the probe fails:
  `Err(InvalidArgument("... 2 is neither 0 nor invertible in the arithmetic of the entries, so they do not lie
  in a field (for PrimeField: every entry is an unbound Ring::zero()/one() constant, or the modulus is even
  and composite)"))`. This is needed: in Z semantics `PrimeField`'s `/` panics (`prime_field.rs:392`,
  `t0.log`), the factorization cannot know p, and a negative `is_similar` would be wrong for some p
  (`[2u]` vs `[0u]` differ over Z but are equal in GF(2)).
* `n == 0` is answered before the probe (an empty `PrimeField` matrix has no bound entry).

The probe and the binding are private helpers in `primary_forms.rs`; the multi-RHS solvers do not use them
(unchanged behaviour of `solve`).

---

## 6. GF(p) tests (`rustmath-finitefields/tests/matrix_forms_gfp.rs`; header gains §26.12.2
`PrimaryRationalForm`, `JordanForm`, `PrimaryInvariantFactors`, `IsSimilar` and the multi-RHS §26.8 forms)

| Test | Input | Expected | Reference | HEAD |
|---|---|---|---|---|
| `h26e9_magma_handbook_example` (port of MAGMA H26E9 = V2.x H51E8) | a of 0.1 over GF(5); MAGMA's r and t | `primary_invariant_factors(a)` = [(x+1,1),(x+1,1),(x+4,1),(x+4,1),(x+4,1)] (MAGMA's printed list, same order); `jordan_form(a)` and `primary_rational_form(a)` canonical matrix = diag(4,4,1,1,1); `invariant_factors(a)` = [x+4, x²+4, x²+4] (MAGMA's f); `t·a == r·t` in our arithmetic; `is_similar(a, r)` = Some(T) with `T·a == r·T` | MAGMA output (`$S/dl/text586.txt`); gp `h51e8.log` (`t*a*t^-1 == r` = 1, `factor` of `matfrobenius(a,1)`); GAP `pif.log` | fails (API); the `invariant_factors` and `t·a == r·t` parts pass |
| `jordan_form_gf3_mixed_blocks` | B = J₂(1)⊕J₂(1)⊕[2]⊕[0,1;2,0] over GF(3) (7×7) | pairs [(x+1,1),(x+2,2),(x+2,2),(x²+1,1)]; J = [2,0,0,0,0,0,0; 0,1,1,0,0,0,0; 0,0,1,0,0,0,0; 0,0,0,1,1,0,0; 0,0,0,0,1,0,0; 0,0,0,0,0,0,1; 0,0,0,0,0,2,0] | gp `matfrobenius(Mod(B,3),1)` = `[x^5+2*x^4+2*x+1, x^2+x+1]`, factors `x+1, (x+2)^2, x^2+1` and `(x+2)^2` (`more.log`); GAP `pif.log` | fails |
| `generalized_jordan_gf3` | A = [1,2,2,0; 1,1,1,1; 0,1,2,2; 0,0,1,2] mod 3 (= U·C((x²+1)²)·U⁻¹) | F = [0,1,0,0; 0,0,1,0; 0,0,0,1; 2,0,1,0]; J = [0,1,0,0; 2,0,1,0; 0,0,0,1; 0,0,2,0]; pairs [(x²+1,2)] | gp `gf3_gen.log`: `matfrobenius` = `[x^4+2*x^2+1]`, `T*A*T^-1 == F`, `minpoly(J)` = `(x^2+1)^2` | fails |
| `jordan_form_large_prime` | p = 2^64 + 13 (gp `nextprime(2^64)`, below the proven range of `is_prime`), A = [3,1,-1; 0,3,2; 0,0,5] | pairs [(x-5,1),(x-3,2)]; J = [5,0,0; 0,3,1; 0,0,3] | gp `bigp.log`: `factormod((x-3)^2*(x-5), p)` order `x+(p-5)`, `x+(p-3)` | fails |
| `primary_forms_reject_unbound_entries` | identity from `PrimeField::one()`/`zero()` | Err(InvalidArgument) from all four functions | | fails |
| `primary_forms_bind_mixed_entries` | `[[2u, 1], [0, 2u]]`, 1 and 0 bound mod 2 | J = [0,1; 0,0] (every entry bound), pairs [(x, 2)] | gp `matfrobenius(Mod([2,1;0,2],2),1)` = `[x^2]` | fails |
| `is_similar_gf5_negative` | I₂ and J₂(1) over GF(5) | None | gp `[x+4, x+4]` vs `[x^2+3*x+1]` (`more.log`) | fails |
| `solve_left_matrix_gf3_all_solutions` | section 1.4 | section 1.4 | `gf3_multi.log` | fails |

---

## 7. Plan

### 7.1 Implementation order (each step builds and tests green before the next)

1. **rustmath-polynomials**: `src/factorization_field.rs` (trait, `Rational` impl, three tests, header);
   `src/lib.rs`: `pub mod factorization_field;` and `pub use factorization_field::FactorizationField;`.
   `cargo test -p rustmath-polynomials --lib factorization_field`.
2. **rustmath-finitefields**: `src/prime_field_factor.rs` (impl for `PrimeField`, two tests, header);
   `src/lib.rs`: `mod prime_field_factor;`. `cargo test -p rustmath-finitefields --lib prime_field_factor`.
3. **rustmath-matrix `linear_solve.rs`**: `solve_columns`, `solve` delegates, the five new methods, module and
   method docs, tests of 1.4. `cargo test -p rustmath-matrix --lib linear_solve`.
4. **rustmath-matrix `primary_forms.rs`** (new): helpers of 2.3 and 5, the four public functions, the two
   structs, tests of 2.5, 3.5, 4.3, doc test. `eigenvalues.rs`: delete the commented-out `jordan_form`
   (`:253-381`) and the struct `JordanForm` (`:385-392`), add the module-doc pointer. `lib.rs`:
   `pub mod primary_forms;`; replace `pub use eigenvalues::{EigenDecomposition, Eigenvector, JordanForm};` by
   `pub use eigenvalues::{EigenDecomposition, Eigenvector};` and add
   `pub use primary_forms::{is_similar, jordan_form, primary_invariant_factors, primary_rational_form,
   FactorizationField, JordanForm, PrimaryRationalForm};`. Once the struct is gone, `use crate::Matrix;` in
   `eigenvalues.rs` is unused outside the comment block: remove it (or the build warns).
   Doc requirements: each public item states the convention of 0.2 (row convention, MAGMA block layouts,
   block order, the d >= 2 caveat), that MAGMA's "multiplicity" in the pairs is the exponent e of the block
   `p^e`, the errors of 2.3, and the cost lines of 2.6 / 4.4.
5. **rustmath-finitefields `tests/matrix_forms_gfp.rs`**: the tests of section 6 and `solve_left_matrix_gf3_all_solutions`;
   header update.
6. **Trackers and docs**:
   * `PROJECT_TRACKER.md:940`: the Jordan form is no longer in the dead block (now `primary_forms.rs`); the
     row keeps the QR/power-iteration status. `:959`: add the matrix and sequence forms and their tests.
     `:979` ("Jordan canonical form (working)"): remove. New 🟢 row "Primary rational form, generalised
     Jordan form, similarity with transform (from the certified rational form and the factored minimal
     polynomial)" with `rustmath-matrix/src/primary_forms.rs:<line>`, the test names and the stated limits
     (d >= 2 layout unverified against MAGMA; Q and GF(p) only).
   * `docs/magma/status_overrides.csv`: `ch026.IsConsistent`, `ch026.Solution` stay `partial` (fields only),
     notes name the matrix form; `ch026.IsConsistent.2`, `ch026.Solution.2` become `partial` (fields only);
     `ch026.PrimaryRationalForm`, `ch026.JordanForm`, `ch026.PrimaryInvariantFactors` become `partial`
     ("Q and GF(p) through FactorizationField; MAGMA allows any field with factorization; generalised Jordan
     layout for degree >= 2 unverified"); `ch026.IsSimilar` becomes `done` (any exact field); add the four
     `ch083.*` rows with the same statuses. Then `python3 tools/magma_tracker/build.py`.
   * `docs/HANDOFF.md:333`: drop "the multi-RHS `IsConsistent`, `PrimaryRationalForm`, `JordanForm`,
     `IsSimilar`"; add the follow-ups of 7.4.

### 7.2 Acceptance commands

```bash
cargo test -p rustmath-polynomials --no-fail-fast
cargo test -p rustmath-finitefields --no-fail-fast
cargo test -p rustmath-matrix --no-fail-fast
cargo test -p rustmath-rings --lib dedekind --no-fail-fast   # Matrix::solve caller (nfctx.rs:243)
cargo check --workspace --tests                              # JordanForm field change, new re-exports
cargo clippy -p rustmath-matrix -p rustmath-polynomials -p rustmath-finitefields
```

### 7.3 Risks

* **M1a changes the RCF contract.** Mitigation in 0.3: tests pin no transform; the certificate catches any
  layout change. If M1a's RCF becomes faster or slower, the costs of 2.6 and 4.4 move with it.
* **Unbound constants.** Section 5 handles them; F1's planned `Ring::bind_to_parent_of` hook is not needed
  (the `x + z` binding works with HEAD's arithmetic) but may replace it later.
* **Factorization limits.** Zassenhaus has a recombination cap (`MathError::NotSupported`), Berlekamp a
  randomized give-up (`NotSupported`) and a primality check; all pass through as `Err`. For p above the
  proven range of `is_prime` (about 3.19·10²³ at HEAD) berlekamp treats p as a probable prime (its docs);
  I1 may turn that into a proof or an `Err`. The large-prime test uses 2^64 + 13, inside the proven range.
* **Size of T in `is_similar` over Q** (4.4). Documented; Dixon or modular methods are tracker
  "missing features".
* **MAGMA layout for d >= 2 and block order over Q** cannot be checked (0.2). Documented in the module docs
  and the tracker notes.

### 7.4 Follow-ups for HANDOFF (not part of this unit)

* `FactorizationField` impls for `FiniteFieldElement` (GF(p^n), through `poly_factor::factor`) and for the
  rings `GFp<P>`; `FactorableConstantField` (rings) as an alias of `FactorizationField`.
* A MAGMA-convention `RationalForm` (`T·A·T⁻¹ = Cᵀ`) is now a few lines (`rational_canonical_form(&Aᵀ)`,
  transposed); it belongs to the rational-form owner (M1a's rows).
* Multi-RHS `IsConsistent` over Euclidean rings (Z), MAGMA §26.8.

---

## Appendix A. Core in pseudo-Rust (the prototype `$S/exp/forms.rs`, `primary_core`, ran on every example)

```rust
fn primary_core<F: FactorizationField>(m: &Matrix<F>, shape: Shape) -> Result<(Pairs<F>, Matrix<F>, Matrix<F>)> {
    let n = m.rows();                                   // m bound, square, n >= 1, field arithmetic
    let (z, one, _) = field_constants(m.data().iter());
    let rcf = rational_canonical_form(m)?;
    let p_r = rcf.change_of_basis.as_ref().ok_or_else(|| internal("RCF without transform"))?;
    let fs = &rcf.invariant_factors;                    // f_1 | ... | f_r
    let offs: Vec<usize> = prefix sums of deg f_i;
    let blocks = primary_blocks(fs, &one)?;             // (p, e, i, q = f_i / p^e), order of 0.2
    let mut cols = Vec::with_capacity(n);
    let mut g = vec![z.clone(); n * n];
    let mut pos = 0;
    let mut pairs = Vec::new();
    for b in &blocks {
        let d = deg b.p; let size = b.e * d;
        let hs: Vec<Poly<F>> = match shape {
            Shape::Companion => (0..size).map(|t| x^t * q).collect(),
            Shape::Jordan => for j in 0..e, for t in 0..d: x^t * p^j * q,
        };
        cols.extend(primary_columns(p_r, offs[b.i], deg f_i, &hs, &z)?);   // Σ_s h_s · P_R[:, off + s]
        for c in 0..size - 1 { g[(pos + c + 1) * n + pos + c] = one.clone(); }   // subdiagonal 1s
        match shape {
            Shape::Companion => for s in 0..size { g[(pos + s) * n + pos + size - 1] = z - coeff_s(p^e); },
            Shape::Jordan => for j in 0..e, s in 0..d { g[(pos + j*d + s) * n + pos + j*d + d - 1] = z - coeff_s(p); },
        }
        pos += size;
        pairs.push((b.p.clone(), b.e));
    }
    if pos != n { return Err(internal("primary blocks do not fill n")); }
    let p = matrix with columns cols; let g = Matrix::from_vec(n, n, g)?;
    if m.mul(&p)? != p.mul(&g)? || !linearly_independent(&cols)? { return Err(internal("certificate failed")); }
    Ok((pairs, g, p))
}
// MAGMA wrappers: (pairs, g, p) = primary_core(&a_bound.transpose(), shape); F = g.transpose(); T = p.transpose().
```

## Appendix B. Reference commands (all in `$S/gp/`, outputs in the `.log` next to each)

PARI/GP 2.15 (`gp -q file.gp < /dev/null`):

```gp
\\ primary invariant factors from the invariant factors (check1.gp)
pif(A) = { my(L = matfrobenius(A, 1), out = List()); for(i = 1, #L, my(f = factor(L[i])); for(j = 1, #f~, listput(out, [f[j,1], f[j,2]]))); Vec(out) };
\\ H26E9 (h51e8.gp)
a = Mod([0,2,4,2,0; 2,2,2,3,3; 3,4,4,1,3; 0,0,0,0,1; 0,0,0,1,0], 5);
r = Mod([1,0,0,0,0; 0,0,1,0,0; 0,1,0,0,0; 0,0,0,0,1; 0,0,0,1,0], 5);
t = Mod([1,3,0,2,1; 2,1,2,2,0; 3,4,3,4,1; 1,0,0,0,0; 0,2,4,2,0], 5);
t*a*t^-1 == r        \\ 1
matfrobenius(a, 1)   \\ [x^2 + Mod(4,5), x^2 + Mod(4,5), x + Mod(4,5)]
\\ Jordan structure (check1.gp)
vector(3, k, matrank((A - l*matid(#A))^k))
\\ transform checks (check1.gp, gf3_gen.gp)
T*A*T^-1 == F
\\ multi-RHS (check1.gp, more.gp, gf3_multi.gp)
matinverseimage(A~, W~)~ ; matker(A~) ; matinverseimage(A, B)
\\ similarity (sage_sim.gp)
matfrobenius(A) == matfrobenius(B) ; matfrobenius(A, 1) ; factor(charpoly(A)) ; minpoly(A)
\\ factorization order (order.gp, fac.gp)
factor((x-3)*(x+3)*(2*x-1)*(x^2+1)^2) ; lift(factormod(x^5+x^4+x+1, 5))
\\ large prime (bigp.gp)
p = nextprime(2^64); lift(factormod((x-3)^2*(x-5), p))
```

GAP 4 (`gap -q pif.g < /dev/null`): primary invariant factors from the Smith form of `xI - A` over `K[x]`:

```gap
PIF := function(F, A)
  local R, x, n, M, ed, out, f, fs, p;
  R := PolynomialRing(F, 1); x := IndeterminatesOfPolynomialRing(R)[1];
  n := Length(A); M := x * IdentityMat(n, R) - A * One(R);
  ed := ElementaryDivisorsMat(R, M); out := [];
  for f in ed do
    if DegreeOfLaurentPolynomial(f) > 0 then
      for p in Collected(Factors(R, f)) do Add(out, [p[1], p[2]]); od;
    fi;
  od;
  return [ed, out];
end;;
```

Results (`pif.log`): H26E9 gives `x-1` three times and `x+1` twice over GF(5) (`x - 1 = x + 4`); the GF(3)
matrix B gives `(x-1)^2` twice, `x+1`, `x^2+1`; G gives `x-1`, `(x^2+1)^2`; Sage `m` gives `(x-4)^2`, `x-2`,
`x-1`. GAP lists them per Smith diagonal entry, so only the multisets are compared with ours.
