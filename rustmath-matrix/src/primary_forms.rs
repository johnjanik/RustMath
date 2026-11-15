//! Primary rational form, (generalised) Jordan form, primary invariant factors and similarity
//! with a transform, over a field.
//!
//! Specification: MAGMA Handbook ch. 26 §26.12.2 `PrimaryRationalForm`, `JordanForm`,
//! `PrimaryInvariantFactors`, `IsSimilar` (digest
//! `rustmath-lava/docs/Chapter_26_Matrices.md:299-312`) and ch. 83 §83.11.2 (digest
//! `rustmath-lava/docs/Chapter_83_Matrix_Algebras.md:321-326`).
//!
//! # Algorithm
//!
//! MAGMA uses A. Steel, *A new algorithm for the computation of canonical forms of matrices
//! over fields*, J. Symbolic Comput. 24 (1997) 409-432 [Ste97]. That algorithm is not
//! implemented here. The primary forms are derived instead from the certified rational form
//! ([`crate::rational_canonical_form`]) and one factorization of the minimal polynomial
//! (D. S. Dummit, R. M. Foote, *Abstract Algebra*, 3rd ed., §12.1 Theorem 6 and the passage from
//! invariant factors to elementary divisors by the Chinese Remainder Theorem, §12.2-12.3;
//! N. Jacobson, *Basic Algebra I*, 2nd ed., §3.9-3.10):
//!
//! 1. Work in the column convention on `M = Aᵀ`. The rational form gives the invariant factors
//!    `f_1 | ... | f_r` and `P_R` with `P_R⁻¹·M·P_R = ⊕ C(f_i)`. The columns of block i of
//!    `P_R` are `g_i, M·g_i, ..., M^{k_i-1}·g_i` (`k_i = deg f_i`, `ann(g_i) = f_i`), so
//!    `h(M)·g_i = Σ_s h_s·P_R[:, off_i + s]` for `deg h < k_i`, with no evaluation at M.
//! 2. Factor the minimal polynomial `f_r` once ([`FactorizationField`]) into `p_1, ..., p_s`.
//!    The exponents `e_ik = v_{p_k}(f_i)` come from repeated exact division, and the quotient
//!    `f_i / Π_k p_k^{e_ik}` is checked to be 1.
//! 3. For each block `(p, e, i)` with `e = v_p(f_i) > 0` let `q = f_i / p^e` and `d = deg p`.
//!    The cyclic subspace of `q(M)·g_i` has annihilator `p^e`, and by the CRT these subspaces
//!    form a direct sum equal to `K^n`. Its basis is `(x^t·q)(M)·g_i`, `t < e·d` (primary
//!    rational form, M acts as the companion matrix of `p^e`), or `(x^t·p^j·q)(M)·g_i`,
//!    `j < e`, `t < d` (Jordan form, the p-adic expansion basis). Every such polynomial has
//!    degree `< k_i`, so step 1 applies.
//! 4. Certificate: `M·P == P·G` and the columns of P are linearly independent (exact
//!    elimination). Then `P⁻¹·M·P = G` is proven, independently of steps 1-3. A failed check is
//!    `Err(InvalidOperation)`, never a value.
//! 5. MAGMA's row convention: `F = Gᵀ`, `T = Pᵀ`, and `T·A·T⁻¹ = (P⁻¹·Aᵀ·P)ᵀ = Gᵀ = F`.
//!
//! [`is_similar`] needs no factorization: two square matrices over a field are similar iff they
//! have the same invariant factors (Dummit-Foote §12.2), which are computed exactly and
//! certified, so `None` is decided, not the result of a search.
//!
//! # Conventions
//!
//! * **Transforms follow MAGMA**: every transform `T` here satisfies `T·A·T⁻¹ = F` (row
//!   convention). This differs from [`crate::RationalCanonicalForm::change_of_basis`]
//!   (`T⁻¹·A·T = C`, column convention), which is why the field here is called `transform`.
//!   Sage's `jordan_form(transformation=True)` returns P with `P⁻¹·A·P = J`; `T⁻¹` is such a P.
//!   Sage's `is_similar(B, transformation=True)` returns S with `A = S⁻¹·B·S`, the same as
//!   MAGMA's `T`.
//! * **Blocks follow MAGMA's `CompanionMatrix`** (Handbook ch. 23: 1s on the superdiagonal, last
//!   row `-c_0, ..., -c_{k-1}`), the transpose of [`crate::companion_matrix`].
//!   * Primary rational form: the block of `p^e` is `CompanionMatrix(p^e)`.
//!   * Generalised Jordan block of `p^e`, `p = x^d + c_{d-1}x^{d-1} + ... + c_0` irreducible:
//!     size `e·d`, every superdiagonal entry 1, and for `j = 0..e-1` row `j·d + d - 1` (0-based)
//!     holds `-c_0, ..., -c_{d-1}` in columns `j·d .. j·d + d - 1`. Equivalently: e copies of
//!     `CompanionMatrix(p)` on the diagonal, consecutive copies joined by a 1 in the bottom-left
//!     corner of the block above the diagonal. For `d = 1` this is the usual Jordan block
//!     `J_e(λ)` (λ on the diagonal, 1s on the superdiagonal). For `(x² + 1)²` over Q it is
//!     `[0,1,0,0; -1,0,1,0; 0,0,0,1; 0,0,-1,0]`. The layout for `d >= 2` is the transpose of the
//!     column-convention block `J(P^q)` of R. Preusser, arXiv:2006.06502, Definition 5 and
//!     Theorem 6 (who cites Bhattacharya, Jain, Nagpaul, *Basic Abstract Algebra*, 2nd ed.,
//!     Ch. 21 §5). MAGMA's printed layout for `d >= 2` could not be checked (no MAGMA was
//!     available, and the handbook example H26E9 has only linear factors).
//! * **Block order**: first by the irreducible factors `p_1, p_2, ...` of the minimal
//!   polynomial in the order of [`FactorizationField::factor_monic_polynomial`] (stable-sorted
//!   by degree), then, for each `p_k`, the blocks `p_k^e` by increasing e, each repeated as
//!   often as it occurs. Over GF(p) the factor order is PARI `factormod` order (degree, then the
//!   coefficients from the top down as integers in `[0, p)`), which reproduces MAGMA's H26E9
//!   output exactly. Over Q it is PARI `factor` order (degree, then the primitive integer
//!   coefficients from the top down: `x - 3, x + 3, 2x - 1`); MAGMA's order over Q is not
//!   documented.
//! * **Pairs**: `primary_invariant_factors` lists `(p, e)` in exactly the block order. MAGMA
//!   calls `e` the "multiplicity"; it is the exponent of the block `p^e`, not the number of
//!   blocks.
//!
//! # Entries and fields
//!
//! `PrimeField` and the GF(p^n) types return unbound integer constants from `Ring::zero()` and
//! `Ring::one()`. Every function here first maps each entry `x ↦ x + z` (`z` the zero of the
//! field of the entries, bound as soon as one entry is bound), which binds mixed entries. If
//! every entry is unbound the field is unknown; the probe "2 is 0 or invertible" (true in every
//! field, false in the Z semantics of the unbound constants) then rejects the input with
//! `Err(InvalidArgument)`. For a `PrimeField` with an odd composite modulus the probe passes and
//! the results are unspecified (out of contract: the probe only detects even composite moduli
//! and sentinel-only input); the factorization of the forms rejects such a modulus.
//!
//! # Cost
//!
//! `RCF(Aᵀ)` + one factorization of the minimal polynomial (Zassenhaus over Q, Berlekamp over
//! GF(p)) + building P (`O(n·Σ k_i²) <= O(n³)` field operations) + the certificate (two products
//! and one elimination, `O(n³)`). Measured times are in the docs of [`jordan_form`] and
//! [`is_similar`].

use crate::companion::{invariant_factors, rational_canonical_form};
use crate::krylov::{field_constants, linearly_independent};
use crate::polynomial_matrix::div_rem_field;
use crate::Matrix;
use rustmath_core::{Field, MathError, Result};
use rustmath_polynomials::UnivariatePolynomial;

pub use rustmath_polynomials::FactorizationField;

type Poly<F> = UnivariatePolynomial<F>;

/// The pairs `(p, e)`: p monic irreducible, block `p^e`.
type Pairs<F> = Vec<(Poly<F>, usize)>;

/// MAGMA `PrimaryRationalForm(A)` (Handbook ch. 26 §26.12.2): the three return values.
#[derive(Debug, Clone)]
pub struct PrimaryRationalForm<F: Field> {
    /// MAGMA's third value, `PrimaryInvariantFactors(A)`: `(p, e)` per block, in block order,
    /// `p` monic irreducible, block = `CompanionMatrix(p^e)`.
    pub primary_invariant_factors: Vec<(UnivariatePolynomial<F>, usize)>,
    /// MAGMA's `F`: block diagonal of the `CompanionMatrix(p^e)` (module docs, Conventions).
    pub canonical_matrix: Matrix<F>,
    /// MAGMA's `T`: invertible with `T·A·T⁻¹ = canonical_matrix` (row convention).
    pub transform: Matrix<F>,
}

/// MAGMA `JordanForm(A)` (Handbook ch. 26 §26.12.2): the three return values.
#[derive(Debug, Clone)]
pub struct JordanForm<F: Field> {
    /// MAGMA's third value, `PrimaryInvariantFactors(A)`: `(p, e)` per block, in block order.
    pub primary_invariant_factors: Vec<(UnivariatePolynomial<F>, usize)>,
    /// MAGMA's `F`: the generalised Jordan form, block diagonal of the generalised Jordan
    /// blocks of the `p^e` (module docs, Conventions). The usual Jordan form when the minimal
    /// polynomial splits.
    pub canonical_matrix: Matrix<F>,
    /// MAGMA's `T`: invertible with `T·A·T⁻¹ = canonical_matrix` (row convention).
    pub transform: Matrix<F>,
}

/// The basis of each primary component: companion (`x^t·q`) or Jordan (`x^t·p^j·q`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Companion,
    Jordan,
}

/// One primary block: `p^e` from the invariant factor `f_i`, `q = f_i / p^e`.
struct PrimaryBlock<F: Field> {
    p: Poly<F>,
    e: usize,
    i: usize,
    q: Poly<F>,
}

fn internal(msg: &str) -> MathError {
    MathError::InvalidOperation(format!("internal: {msg}"))
}

/// The constants `(z, one)` of the field of `entries` (bound as soon as one entry is bound),
/// after the field probe of the module docs.
fn field_of<'a, F: Field + 'a>(entries: impl Iterator<Item = &'a F>, what: &str) -> Result<(F, F)> {
    let (z, one, _) = field_constants(entries);
    let two = one.clone() + one.clone();
    if two.is_zero() || two.inverse().is_ok() {
        Ok((z, one))
    } else {
        Err(MathError::InvalidArgument(format!(
            "{what}: 2 is neither 0 nor invertible in the arithmetic of the entries, so they do \
             not lie in a field (for PrimeField: every entry is an unbound Ring::zero()/one() \
             constant, or the modulus is even and composite)"
        )))
    }
}

/// `x ↦ x + z` on every entry, which binds unbound entries to the field of `z`.
fn bind<F: Field>(a: &Matrix<F>, z: &F) -> Result<Matrix<F>> {
    Matrix::from_vec(
        a.rows(),
        a.cols(),
        a.data().iter().map(|x| x.clone() + z.clone()).collect(),
    )
}

/// `c ↦ c + z` on every coefficient (review M2: the oracle sees bound coefficients however the
/// polynomial was built).
fn bind_poly<F: Field>(f: &Poly<F>, z: &F) -> Poly<F> {
    Poly::new(
        f.coefficients()
            .iter()
            .map(|c| c.clone() + z.clone())
            .collect(),
    )
}

/// Square check, then the field probe and the binding. `n == 0` is answered by the callers
/// before this.
fn prepared<F: Field>(a: &Matrix<F>, what: &str) -> Result<(Matrix<F>, F, F)> {
    if !a.is_square() {
        return Err(MathError::InvalidArgument(format!(
            "{what}: the matrix must be square, it is {}x{}",
            a.rows(),
            a.cols()
        )));
    }
    let (z, one) = field_of(a.data().iter(), what)?;
    Ok((bind(a, &z)?, z, one))
}

/// `p^e` by repeated multiplication, starting from the constant `one`.
fn poly_pow<F: Field>(p: &Poly<F>, e: usize, one: &F) -> Poly<F> {
    let mut acc = Poly::new(vec![one.clone()]);
    for _ in 0..e {
        acc = acc * p.clone();
    }
    acc
}

/// `a / b` when `b` divides `a` exactly, `None` otherwise.
fn exact_div<F: Field>(a: &Poly<F>, b: &Poly<F>) -> Result<Option<Poly<F>>> {
    let (q, r) = div_rem_field(a, b)?;
    Ok(r.is_zero().then_some(q))
}

/// Steps 2-3 of the module docs: the primary blocks of the invariant factors `fs`, in the
/// block order of the module docs. Validates the factorization oracle's output.
fn primary_blocks<F: FactorizationField>(
    fs: &[Poly<F>],
    z: &F,
    one: &F,
) -> Result<Vec<PrimaryBlock<F>>> {
    let fs: Vec<Poly<F>> = fs.iter().map(|f| bind_poly(f, z)).collect();
    let Some(last) = fs.last() else {
        return Ok(Vec::new());
    };
    let mut irr = F::factor_monic_polynomial(last)?;
    // Validate the oracle: monic, degree >= 1, multiplicity >= 1, pairwise distinct.
    for (idx, (g, e)) in irr.iter().enumerate() {
        let monic = g.leading_coefficient().is_some_and(|c| c.is_one());
        if !monic || g.degree().is_none_or(|d| d == 0) || *e == 0 {
            return Err(internal(
                "the factorization returned a factor that is not monic of positive degree, or \
                 a multiplicity 0",
            ));
        }
        if irr[..idx].iter().any(|(h, _)| h == g) {
            return Err(internal("the factorization returned a factor twice"));
        }
    }
    // Stable sort by degree (the oracle's contract already orders by degree first).
    irr.sort_by_key(|(g, _)| g.degree());

    // Exponents e_ik = v_{p_k}(f_i); the rest must be 1.
    let mut exps: Vec<Vec<usize>> = Vec::with_capacity(fs.len());
    for f in &fs {
        let mut rest = f.clone();
        let mut row = Vec::with_capacity(irr.len());
        for (p, _) in &irr {
            let mut e = 0;
            while let Some(q) = exact_div(&rest, p)? {
                // A nonconstant monic p lowers the degree, so this terminates.
                if q.degree() >= rest.degree() {
                    return Err(internal("an exact division did not lower the degree"));
                }
                rest = q;
                e += 1;
            }
            row.push(e);
        }
        if rest.degree() != Some(0) || !rest.coeff(0).is_one() {
            return Err(internal(
                "an invariant factor is not a product of the irreducible factors of the minimal \
                 polynomial",
            ));
        }
        exps.push(row);
    }
    // The exponents of the minimal polynomial are the oracle's multiplicities.
    let last_row = exps.last().ok_or_else(|| internal("no invariant factor"))?;
    if last_row.iter().zip(&irr).any(|(e, (_, m))| e != m) {
        return Err(internal(
            "the multiplicities of the factorization disagree with the exact valuations",
        ));
    }

    // Blocks (k, e, i), sorted: by factor, then by exponent, then by invariant factor.
    let mut keys: Vec<(usize, usize, usize)> = Vec::new();
    for (i, row) in exps.iter().enumerate() {
        for (k, &e) in row.iter().enumerate() {
            if e > 0 {
                keys.push((k, e, i));
            }
        }
    }
    keys.sort_unstable();
    let mut blocks = Vec::with_capacity(keys.len());
    for (k, e, i) in keys {
        let p = irr[k].0.clone();
        let pe = poly_pow(&p, e, one);
        let q = exact_div(&fs[i], &pe)?
            .ok_or_else(|| internal("p^e does not divide its invariant factor"))?;
        blocks.push(PrimaryBlock { p, e, i, q });
    }
    Ok(blocks)
}

/// The columns `h(M)·g_i = Σ_s h_s·P_R[:, off + s]` for the polynomials `h` of one block
/// (`deg h < k` is checked, `k = deg f_i`).
fn primary_columns<F: Field>(
    p_r: &Matrix<F>,
    off: usize,
    k: usize,
    hs: &[Poly<F>],
    z: &F,
) -> Result<Vec<Vec<F>>> {
    let n = p_r.rows();
    let w = p_r.cols();
    let d = p_r.data();
    let mut cols = Vec::with_capacity(hs.len());
    for h in hs {
        if h.degree().is_some_and(|dh| dh >= k) {
            return Err(internal(
                "a basis polynomial has degree >= its invariant factor",
            ));
        }
        let mut col = vec![z.clone(); n];
        for (s, c) in h.coefficients().iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            for (row, cr) in col.iter_mut().enumerate() {
                *cr = cr.clone() + c.clone() * d[row * w + off + s].clone();
            }
        }
        cols.push(col);
    }
    Ok(cols)
}

/// The column-convention core on `M` (bound, square, n >= 1, field arithmetic): `(pairs, G, P)`
/// with `M·P = P·G` and P invertible, both certified (module docs, steps 1-4).
fn primary_core<F: FactorizationField>(
    m: &Matrix<F>,
    shape: Shape,
) -> Result<(Pairs<F>, Matrix<F>, Matrix<F>)> {
    let n = m.rows();
    let (z, one, _) = field_constants(m.data().iter());
    let rcf = rational_canonical_form(m)?;
    let p_r = rcf
        .change_of_basis
        .as_ref()
        .ok_or_else(|| internal("the rational form has no change of basis"))?;
    let fs = &rcf.invariant_factors;
    let mut offs = Vec::with_capacity(fs.len());
    let mut degs = Vec::with_capacity(fs.len());
    let mut o = 0;
    for f in fs {
        let d = f
            .degree()
            .ok_or_else(|| internal("a zero invariant factor"))?;
        offs.push(o);
        degs.push(d);
        o += d;
    }
    if o != n || p_r.rows() != n || p_r.cols() != n {
        return Err(internal("the rational form does not have size n"));
    }
    let blocks = primary_blocks(fs, &z, &one)?;

    let mut cols: Vec<Vec<F>> = Vec::with_capacity(n);
    let mut g = vec![z.clone(); n * n];
    let mut pairs = Vec::with_capacity(blocks.len());
    let mut pos = 0;
    for b in &blocks {
        let d =
            b.p.degree()
                .ok_or_else(|| internal("a zero irreducible factor"))?;
        let size = b.e * d;
        if pos + size > n {
            return Err(internal("the primary blocks exceed n"));
        }
        // The basis polynomials of the block, and its entries of G (column convention).
        let mut hs: Vec<Poly<F>> = Vec::with_capacity(size);
        match shape {
            Shape::Companion => {
                let mut h = b.q.clone();
                for _ in 0..size {
                    hs.push(h.clone());
                    h = h.shift(1);
                }
                let pe = poly_pow(&b.p, b.e, &one);
                for s in 0..size {
                    g[(pos + s) * n + pos + size - 1] = z.clone() - pe.coeff(s).clone();
                }
            }
            Shape::Jordan => {
                let mut pj = b.q.clone();
                for j in 0..b.e {
                    let mut h = pj.clone();
                    for _ in 0..d {
                        hs.push(h.clone());
                        h = h.shift(1);
                    }
                    pj = pj * b.p.clone();
                    for s in 0..d {
                        g[(pos + j * d + s) * n + pos + j * d + d - 1] =
                            z.clone() - b.p.coeff(s).clone();
                    }
                }
            }
        }
        // Subdiagonal 1s inside the block.
        for c in 0..size.saturating_sub(1) {
            g[(pos + c + 1) * n + pos + c] = one.clone();
        }
        cols.extend(primary_columns(p_r, offs[b.i], degs[b.i], &hs, &z)?);
        pos += size;
        pairs.push((b.p.clone(), b.e));
    }
    if pos != n || cols.len() != n {
        return Err(internal("the primary blocks do not fill n"));
    }
    let mut pdata = Vec::with_capacity(n * n);
    for r in 0..n {
        for col in &cols {
            pdata.push(col[r].clone());
        }
    }
    let p = Matrix::from_vec(n, n, pdata)?;
    let g = Matrix::from_vec(n, n, g)?;
    // Certificate: M·P = P·G and P invertible prove P⁻¹·M·P = G.
    if m.mul(&p)? != p.mul(&g)? {
        return Err(internal("certificate M·P = P·G failed"));
    }
    if !linearly_independent(&cols)? {
        return Err(internal("certificate failed: the transform is singular"));
    }
    Ok((pairs, g, p))
}

/// The MAGMA-convention wrapper of [`primary_core`]: `(pairs, F, T)` with `T·A·T⁻¹ = F`.
fn magma_form<F: FactorizationField>(
    a: &Matrix<F>,
    shape: Shape,
    what: &str,
) -> Result<(Pairs<F>, Matrix<F>, Matrix<F>)> {
    if a.is_square() && a.rows() == 0 {
        let e = Matrix::from_vec(0, 0, Vec::new())?;
        return Ok((Vec::new(), e.clone(), e));
    }
    let (a, _, _) = prepared(a, what)?;
    let (pairs, g, p) = primary_core(&a.transpose(), shape)?;
    Ok((pairs, g.transpose(), p.transpose()))
}

/// MAGMA `PrimaryInvariantFactors(A)` (Handbook ch. 26 §26.12.2, ch. 83 §83.11.2): the pairs
/// `(p, e)`, p monic irreducible, one per block `p^e` of the primary rational form, in the block
/// order of the module docs. MAGMA's "multiplicity" is the exponent e. The same list as the
/// `primary_invariant_factors` of [`primary_rational_form`] and [`jordan_form`], computed from
/// [`crate::invariant_factors`] and one factorization of the minimal polynomial, without a
/// transform.
///
/// * `Ok(vec![])` for the 0×0 matrix.
/// * `Err(InvalidArgument)` if `A` is not square, or if its entries do not lie in a field (the
///   probe of the module docs: for `PrimeField`, every entry an unbound constant).
/// * Errors of [`crate::invariant_factors`] and of the factorization pass through (Zassenhaus'
///   recombination cap over Q, Berlekamp's give-up and primality check over GF(p)).
/// * `Err(InvalidOperation("internal: ..."))` if the factorization breaks its contract.
pub fn primary_invariant_factors<F: FactorizationField>(
    a: &Matrix<F>,
) -> Result<Vec<(UnivariatePolynomial<F>, usize)>> {
    let what = "primary_invariant_factors";
    if a.is_square() && a.rows() == 0 {
        return Ok(Vec::new());
    }
    let (a, z, one) = prepared(a, what)?;
    let fs = invariant_factors(&a)?;
    Ok(primary_blocks(&fs, &z, &one)?
        .into_iter()
        .map(|b| (b.p, b.e))
        .collect())
}

/// MAGMA `PrimaryRationalForm(A)` (Handbook ch. 26 §26.12.2, ch. 83 §83.11.2): `F`, `T` with
/// `T·A·T⁻¹ = F` (row convention), and the pairs of [`primary_invariant_factors`].
///
/// `F` is block diagonal with the blocks `CompanionMatrix(p^e)` (MAGMA's layout: 1s on the
/// superdiagonal, `-c_0, ..., -c_{k-1}` in the last row) in the block order of the module docs,
/// so it depends only on the similarity class of `A` (given the factor order). Every entry of
/// `F` and `T` is bound to the field of the entries of `A`.
///
/// * `Ok` with empty pairs and 0×0 matrices for the 0×0 matrix.
/// * `Err(InvalidArgument)` if `A` is not square or its entries do not lie in a field (module
///   docs).
/// * Errors of [`crate::rational_canonical_form`] and of the factorization pass through.
/// * `Err(InvalidOperation("internal: ..."))` if the factorization breaks its contract or the
///   certificate (`M·P = P·G`, P invertible; module docs) fails. No uncertified value is
///   returned.
///
/// Cost: as [`jordan_form`].
pub fn primary_rational_form<F: FactorizationField>(
    a: &Matrix<F>,
) -> Result<PrimaryRationalForm<F>> {
    let (pairs, f, t) = magma_form(a, Shape::Companion, "primary_rational_form")?;
    Ok(PrimaryRationalForm {
        primary_invariant_factors: pairs,
        canonical_matrix: f,
        transform: t,
    })
}

/// MAGMA `JordanForm(A)` (Handbook ch. 26 §26.12.2, ch. 83 §83.11.2): the generalised Jordan
/// form `F`, `T` with `T·A·T⁻¹ = F` (row convention), and the pairs of
/// [`primary_invariant_factors`].
///
/// `F` is block diagonal with the generalised Jordan blocks of the `p^e` (module docs,
/// Conventions) in the block order of the module docs. When the minimal polynomial splits
/// (every `p = x - λ`) this is the usual Jordan form, `λ` on the diagonal and 1s on the
/// superdiagonal; otherwise it is MAGMA's generalised form, and no extension field is used.
/// MAGMA's printed layout for `deg p >= 2` is unverified (module docs). Every entry of `F` and
/// `T` is bound to the field of the entries of `A`. Errors as for [`primary_rational_form`].
///
/// ```
/// use rustmath_matrix::{jordan_form, Matrix};
/// use rustmath_rationals::Rational;
///
/// // Sage's `jordan_form` example: invariant factors x - 1 and (x - 1)^2 (x - 2).
/// let q = |v: &[i64]| v.iter().map(|&x| Rational::from_integer(x)).collect::<Vec<_>>();
/// let a = Matrix::from_vec(4, 4, q(&[1, 0, 0, 0, 0, 1, 0, 0, 1, -1, 1, 0, 1, -1, 1, 2])).unwrap();
/// let jf = jordan_form(&a).unwrap();
/// let j = Matrix::from_vec(4, 4, q(&[2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1])).unwrap();
/// assert_eq!(jf.canonical_matrix, j);
/// let t = &jf.transform;
/// assert_eq!(t.mul(&a).unwrap(), j.mul(t).unwrap());
/// ```
///
/// Cost: `RCF(Aᵀ)`, one factorization of the minimal polynomial, `O(n³)` field operations for
/// P and the certificate (module docs). Measured in a debug build:
///
/// | Input | `RCF(Aᵀ)` | `jordan_form` (total) |
/// |---|---|---|
/// | Q, dense n = 8 / 12 / 16, entries in [-5, 5] | 2 / 12 / 49 ms | 4 / 19 / 72 ms |
/// | Q, `U·J·U⁻¹`, n = 16, six Jordan blocks, U unimodular | 59 ms | 80 ms |
/// | GF(101), GF(10⁹ + 7), GF(10²⁴ + 7), dense n = 20 | | 0.07 s, 0.09 s, 0.73 s |
///
/// [`primary_rational_form`] costs the same.
pub fn jordan_form<F: FactorizationField>(a: &Matrix<F>) -> Result<JordanForm<F>> {
    let (pairs, f, t) = magma_form(a, Shape::Jordan, "jordan_form")?;
    Ok(JordanForm {
        primary_invariant_factors: pairs,
        canonical_matrix: f,
        transform: t,
    })
}

/// MAGMA `IsSimilar(A, B)` (Handbook ch. 26 §26.12.2, ch. 83 §83.11.2): `Ok(Some(T))` with T
/// invertible and `T·A·T⁻¹ = B` iff A and B are similar, `Ok(None)` iff they are not.
///
/// Decision: A and B are similar iff they have the same invariant factors (Dummit-Foote §12.2),
/// which [`crate::invariant_factors`] computes exactly and certifies, so `None` is decided.
/// Transform: with the rational-form changes of basis `P_A`, `P_B` (`P⁻¹·X·P = C`, the same C
/// for both), `T = P_B·P_A⁻¹`, obtained from `T·P_A = P_B` by one multi-RHS solve
/// ([`Matrix::solve_left_matrix`]) and certified by that product. Then
/// `T·A·T⁻¹ = P_B·C·P_B⁻¹ = B` follows from the certified rational forms.
///
/// No factorization is needed, so any field works (MAGMA asks for factorization because [Ste97]
/// goes through the primary form). Over an exact field (Q, GF(p), GF(p^n), number fields) both
/// answers are proofs. For a rounding `Field` impl (`Real`, `Complex`) neither answer is a
/// proof. The entries of A and B must lie in one field (`PrimeField` arithmetic panics on two
/// different moduli); for a `PrimeField` with a composite modulus the result is unspecified (the
/// probe only detects even composite moduli and sentinel-only input).
///
/// * `Ok(Some(0×0))` for two 0×0 matrices.
/// * `Err(InvalidArgument)` if A or B is not square, if their sizes differ, or if their entries
///   do not lie in a field (module docs).
/// * Errors of [`crate::invariant_factors`] and [`crate::rational_canonical_form`] pass through;
///   `Err(InvalidOperation("internal: ..."))` if the certificate fails.
///
/// Cost: two invariant-factor computations, two rational forms, one solve and one product. Over
/// Q the solve and the product dominate for dense input, because `P_A⁻¹` has large entries; no
/// bound is claimed for the size of T over Q. Measured in a debug build, `B` the Jordan form of
/// `A`: over Q, dense n = 8 / 12 / 16 with entries in [-5, 5], 16 / 137 / 610 ms (the entries of
/// T reach 29 / 73 / 135 digits), and 0.42 s for the n = 16 input with six Jordan blocks of
/// [`jordan_form`]; over GF(101), GF(10⁹ + 7), GF(10²⁴ + 7), dense n = 20, 0.10 / 0.10 / 0.34 s.
pub fn is_similar<F: Field>(a: &Matrix<F>, b: &Matrix<F>) -> Result<Option<Matrix<F>>> {
    let what = "is_similar";
    if !a.is_square() || !b.is_square() {
        return Err(MathError::InvalidArgument(format!(
            "{what}: both matrices must be square, they are {}x{} and {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        )));
    }
    if a.rows() != b.rows() {
        return Err(MathError::InvalidArgument(format!(
            "{what}: the matrices have different sizes {} and {}",
            a.rows(),
            b.rows()
        )));
    }
    if a.rows() == 0 {
        return Ok(Some(Matrix::from_vec(0, 0, Vec::new())?));
    }
    let (z, _) = field_of(a.data().iter().chain(b.data().iter()), what)?;
    let (a, b) = (bind(a, &z)?, bind(b, &z)?);
    if invariant_factors(&a)? != invariant_factors(&b)? {
        return Ok(None); // decided: different invariant factors
    }
    let ra = rational_canonical_form(&a)?;
    let rb = rational_canonical_form(&b)?;
    if ra.canonical_matrix != rb.canonical_matrix {
        return Err(internal(
            "equal invariant factors but different rational forms",
        ));
    }
    let pa = ra
        .change_of_basis
        .ok_or_else(|| internal("the rational form of A has no change of basis"))?;
    let pb = rb
        .change_of_basis
        .ok_or_else(|| internal("the rational form of B has no change of basis"))?;
    let t = pa
        .solve_left_matrix(&pb)?
        .ok_or_else(|| internal("T·P_A = P_B has no solution although P_A is invertible"))?;
    // One certificate suffices (see the docs). A second check T·A == B·T would cost two more
    // products with the large entries of T over Q (about 60% more time, 3.4 s on 5.6 s, in the
    // design prototype's measurement at n = 16, dense).
    if t.mul(&pa)? != pb {
        return Err(internal("certificate T·P_A = P_B failed"));
    }
    Ok(Some(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn q(v: i64) -> Rational {
        Rational::from_integer(v)
    }

    fn qm(rows: usize, cols: usize, v: &[i64]) -> Matrix<Rational> {
        Matrix::from_vec(rows, cols, v.iter().map(|&x| q(x)).collect()).unwrap()
    }

    /// The polynomial with coefficients `c` (low to high) over Q.
    fn qp(c: &[i64]) -> Poly<Rational> {
        Poly::new(c.iter().map(|&x| q(x)).collect())
    }

    /// MAGMA's defining property: `t·a == f·t` and t invertible.
    fn assert_magma_transform(a: &Matrix<Rational>, f: &Matrix<Rational>, t: &Matrix<Rational>) {
        assert_eq!(t.mul(a).unwrap(), f.mul(t).unwrap());
        assert_eq!(t.rank().unwrap(), a.rows());
    }

    /// G = U·(C((x²+1)²) ⊕ [1])·U⁻¹ (design §2.5; U unimodular).
    fn g_matrix() -> Matrix<Rational> {
        qm(
            5,
            5,
            &[
                2, -4, 5, -18, 31, 1, -1, 1, -6, 11, 0, 1, 2, -8, 15, 0, 0, 1, -3, 7, 0, 0, 0, 0, 1,
            ],
        )
    }

    /// The column-style companion form C((x²+1)²) ⊕ [1], similar to G.
    fn c_matrix() -> Matrix<Rational> {
        qm(
            5,
            5,
            &[
                0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
            ],
        )
    }

    /// Sage `m` (jordan_form docs): charpoly (x-4)²(x-2)(x-1), one Jordan block J₂(4).
    fn sage_m() -> Matrix<Rational> {
        qm(4, 4, &[5, 4, 2, 1, 0, 1, -1, -1, -1, -1, 3, 0, 1, 1, -1, 2])
    }

    /// Sage `a` (jordan_form docs): invariant factors x - 1, (x - 1)²(x - 2).
    fn sage_a() -> Matrix<Rational> {
        qm(4, 4, &[1, 0, 0, 0, 0, 1, 0, 0, 1, -1, 1, 0, 1, -1, 1, 2])
    }

    #[test]
    fn g_is_u_c_u_inverse() {
        // gp: U = [1,2,0,1,0; 0,1,1,0,0; 0,0,1,3,1; 0,0,0,1,2; 0,0,0,0,1]; U*C*U^-1 == G
        let u = qm(
            5,
            5,
            &[
                1, 2, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1,
            ],
        );
        assert_eq!(u.mul(&c_matrix()).unwrap(), g_matrix().mul(&u).unwrap());
    }

    #[test]
    fn primary_invariant_factors_q() {
        // gp: pif(A) = the factors of each matfrobenius(A, 1) entry (design check1.gp):
        //   G: [[x - 1, 1], [x^2 + 1, 2]]; sage_m: [[x - 4, 2], [x - 2, 1], [x - 1, 1]];
        //   sage_a: [[x - 2, 1], [x - 1, 2], [x - 1, 1]] (our order: x - 1 blocks by increasing e).
        // GAP (design pif.g, Smith form of xI - A over Q[x]) gives the same multisets.
        assert_eq!(
            primary_invariant_factors(&g_matrix()).unwrap(),
            vec![(qp(&[-1, 1]), 1), (qp(&[1, 0, 1]), 2)]
        );
        assert_eq!(
            primary_invariant_factors(&sage_m()).unwrap(),
            vec![(qp(&[-4, 1]), 2), (qp(&[-2, 1]), 1), (qp(&[-1, 1]), 1)]
        );
        assert_eq!(
            primary_invariant_factors(&sage_a()).unwrap(),
            vec![(qp(&[-2, 1]), 1), (qp(&[-1, 1]), 1), (qp(&[-1, 1]), 2)]
        );
    }

    #[test]
    fn primary_rational_form_q() {
        // gp: T*A*T^-1 == F = 1 for the prototype's T (design check1.log), with
        //   G: F = [1,0,0,0,0; 0,0,1,0,0; 0,0,0,1,0; 0,0,0,0,1; 0,-1,0,-2,0]
        //   sage_m: F = [0,1,0,0; -16,8,0,0; 0,0,2,0; 0,0,0,1]
        //   factor(charpoly(F)) = (x - 1)(x^2 + 1)^2 and (x - 4)^2 (x - 2)(x - 1)
        let g = g_matrix();
        let prf = primary_rational_form(&g).unwrap();
        let f = qm(
            5,
            5,
            &[
                1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, -1, 0, -2, 0,
            ],
        );
        assert_eq!(prf.canonical_matrix, f);
        assert_magma_transform(&g, &f, &prf.transform);
        let m = sage_m();
        let prf = primary_rational_form(&m).unwrap();
        let f = qm(4, 4, &[0, 1, 0, 0, -16, 8, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
        assert_eq!(prf.canonical_matrix, f);
        assert_magma_transform(&m, &f, &prf.transform);
    }

    #[test]
    fn primary_forms_agree() {
        for a in [g_matrix(), sage_m(), sage_a()] {
            let pif = primary_invariant_factors(&a).unwrap();
            assert_eq!(jordan_form(&a).unwrap().primary_invariant_factors, pif);
            assert_eq!(
                primary_rational_form(&a).unwrap().primary_invariant_factors,
                pif
            );
        }
    }

    #[test]
    fn primary_forms_shapes() {
        let r = qm(2, 3, &[1, 2, 3, 4, 5, 6]);
        assert!(matches!(
            primary_invariant_factors(&r),
            Err(MathError::InvalidArgument(_))
        ));
        assert!(matches!(
            primary_rational_form(&r),
            Err(MathError::InvalidArgument(_))
        ));
        assert!(matches!(
            jordan_form(&r),
            Err(MathError::InvalidArgument(_))
        ));
        let e = Matrix::<Rational>::from_vec(0, 0, vec![]).unwrap();
        assert!(primary_invariant_factors(&e).unwrap().is_empty());
        let prf = primary_rational_form(&e).unwrap();
        assert!(prf.primary_invariant_factors.is_empty());
        assert_eq!((prf.canonical_matrix.rows(), prf.transform.rows()), (0, 0));
        let jf = jordan_form(&e).unwrap();
        assert!(jf.primary_invariant_factors.is_empty());
        assert_eq!((jf.canonical_matrix.rows(), jf.transform.rows()), (0, 0));
        // 1×1: [7] is its own form.
        let s = qm(1, 1, &[7]);
        let jf = jordan_form(&s).unwrap();
        assert_eq!(jf.canonical_matrix, s);
        assert_eq!(jf.primary_invariant_factors, vec![(qp(&[-7, 1]), 1)]);
    }

    #[test]
    fn jordan_form_sage_examples() {
        // Sage jordan_form docs (matrix2.pyx): `a` has blocks {[2], J₂(1), [1]}, `m` has
        // {[2], [1], J₂(4)}; Sage orders the blocks differently, ours follow the factor order.
        // gp: vector(3, k, matrank((A - 1)^k)) = [2, 1, 1] for `a` at λ = 1 (one J₂(1), one
        // J₁(1)); design check1.log: T*A*T^-1 == J = 1 for the prototype's T.
        let a = sage_a();
        let jf = jordan_form(&a).unwrap();
        let j = qm(4, 4, &[2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]);
        assert_eq!(jf.canonical_matrix, j);
        assert_magma_transform(&a, &j, &jf.transform);
        let m = sage_m();
        let jf = jordan_form(&m).unwrap();
        let j = qm(4, 4, &[4, 1, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
        assert_eq!(jf.canonical_matrix, j);
        assert_magma_transform(&m, &j, &jf.transform);
        // Sage's T⁻¹ convention: P = T⁻¹ has P⁻¹·A·P = J.
        let p = jf.transform.inverse().unwrap().unwrap();
        assert_eq!(m.mul(&p).unwrap(), p.mul(&j).unwrap());
    }

    #[test]
    fn jordan_form_generalized_block_q() {
        // gp: factor(minpoly([0,1,0,0;-1,0,1,0;0,0,0,1;0,0,-1,0])) = (x^2 + 1)^2;
        //     T*G*T^-1 == J = 1 for the prototype's T (design check1.log).
        let g = g_matrix();
        let jf = jordan_form(&g).unwrap();
        let j = qm(
            5,
            5,
            &[
                1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0,
            ],
        );
        assert_eq!(jf.canonical_matrix, j);
        assert_eq!(
            jf.primary_invariant_factors,
            vec![(qp(&[-1, 1]), 1), (qp(&[1, 0, 1]), 2)]
        );
        assert_magma_transform(&g, &j, &jf.transform);
    }

    #[test]
    fn jordan_form_sage_similarity_matrices() {
        // gp (design sage_sim.log): matfrobenius(A3, 1) = [x^3 - 5*x^2 - 8*x + 48] (cyclic,
        // charpoly (x - 4)^2 (x + 3)); matfrobenius(B3, 1) = [x^2 - x - 12, x - 4].
        let a3 = qm(3, 3, &[19, -7, -29, -16, 11, 30, 15, -7, -25]);
        let jf = jordan_form(&a3).unwrap();
        let j = qm(3, 3, &[4, 1, 0, 0, 4, 0, 0, 0, -3]);
        assert_eq!(jf.canonical_matrix, j);
        assert_eq!(
            jf.primary_invariant_factors,
            vec![(qp(&[-4, 1]), 2), (qp(&[3, 1]), 1)]
        );
        assert_magma_transform(&a3, &j, &jf.transform);
        let b3 = qm(3, 3, &[-38, -63, 42, 14, 25, -14, -14, -21, 18]);
        let jf = jordan_form(&b3).unwrap();
        let j = qm(3, 3, &[4, 0, 0, 0, 4, 0, 0, 0, -3]);
        assert_eq!(jf.canonical_matrix, j);
        assert_eq!(
            jf.primary_invariant_factors,
            vec![(qp(&[-4, 1]), 1), (qp(&[-4, 1]), 1), (qp(&[3, 1]), 1)]
        );
        assert_magma_transform(&b3, &j, &jf.transform);
    }

    /// `t·a == b·t` and t invertible.
    fn assert_similar_by(a: &Matrix<Rational>, b: &Matrix<Rational>, t: &Matrix<Rational>) {
        assert_eq!(t.mul(a).unwrap(), b.mul(t).unwrap());
        assert_eq!(t.rank().unwrap(), a.rows());
    }

    #[test]
    fn is_similar_sage_examples() {
        // Sage is_similar docs (matrix2.pyx), ex. 1: similar, with S = [1,0,0; -2/3,1/6,-5/6;
        // 2/3,0,-1/3] and A = S⁻¹·B·S. gp: matfrobenius(A) == matfrobenius(B) = 1,
        // matfrobenius(A, 1) = [x^3 + 4*x^2 + 7*x + 6], T*A*T^-1 == B = 1 for S.
        let a = qm(3, 3, &[-5, 2, -11, -6, 7, -42, 0, 1, -6]);
        let b = qm(3, 3, &[1, 12, 3, -1, -6, -1, 0, 6, 1]);
        let t = is_similar(&a, &b).unwrap().expect("similar");
        assert_similar_by(&a, &b, &t);
        let s = Matrix::from_vec(
            3,
            3,
            vec![
                q(1),
                q(0),
                q(0),
                Rational::new(-2, 3).unwrap(),
                Rational::new(1, 6).unwrap(),
                Rational::new(-5, 6).unwrap(),
                Rational::new(2, 3).unwrap(),
                q(0),
                Rational::new(-1, 3).unwrap(),
            ],
        )
        .unwrap();
        assert_similar_by(&a, &b, &s);
        // Ex. 2: different characteristic polynomials (gp: charpoly(A2) =
        // x^4 - 13*x^3 + 13*x^2 + 16*x + 1, charpoly(B2) = x^4 + x^3 - 3*x^2 - 9*x + 1).
        let a2 = qm(
            4,
            4,
            &[2, 3, -3, -6, 0, 1, -2, -8, -3, -3, 4, 3, -1, -2, 2, 6],
        );
        let b2 = qm(
            4,
            4,
            &[1, 1, 2, 4, -1, 2, -3, -7, -2, 3, -4, -7, 0, -1, 0, 0],
        );
        assert_eq!(is_similar(&a2, &b2).unwrap(), None);
        // Ex. 3: same characteristic polynomial (x - 4)^2 (x + 3), different invariant factors
        // (gp: [x^3 - 5*x^2 - 8*x + 48] against [x^2 - x - 12, x - 4]).
        let a3 = qm(3, 3, &[19, -7, -29, -16, 11, 30, 15, -7, -25]);
        let b3 = qm(3, 3, &[-38, -63, 42, 14, 25, -14, -14, -21, 18]);
        assert_eq!(is_similar(&a3, &b3).unwrap(), None);
    }

    #[test]
    fn is_similar_same_char_and_min_poly() {
        // gp: matfrobenius = [x^2, x^2] against [x^2, x, x]; charpoly x^4 and minpoly x^2 for
        // both (design check1.log).
        let j1 = qm(4, 4, &[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
        let j2 = qm(4, 4, &[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(is_similar(&j1, &j2).unwrap(), None);
        assert_eq!(is_similar(&j2, &j1).unwrap(), None);
    }

    #[test]
    fn is_similar_to_own_forms() {
        // gp: T*G*T^-1 == C = 1 for the prototype's T (design check1.log); a matrix is similar
        // to its transpose.
        let g = g_matrix();
        let targets = [
            c_matrix(),
            jordan_form(&g).unwrap().canonical_matrix,
            primary_rational_form(&g).unwrap().canonical_matrix,
            g.transpose(),
        ];
        for x in targets {
            let t = is_similar(&g, &x).unwrap().expect("similar");
            assert_similar_by(&g, &x, &t);
        }
    }

    #[test]
    fn is_similar_shapes() {
        let r = qm(2, 3, &[1, 2, 3, 4, 5, 6]);
        assert!(matches!(
            is_similar(&r, &r),
            Err(MathError::InvalidArgument(_))
        ));
        let a = qm(2, 2, &[1, 0, 0, 1]);
        let b = qm(3, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1]);
        assert!(matches!(
            is_similar(&a, &b),
            Err(MathError::InvalidArgument(_))
        ));
        let e = Matrix::<Rational>::from_vec(0, 0, vec![]).unwrap();
        let t = is_similar(&e, &e).unwrap().expect("similar");
        assert_eq!((t.rows(), t.cols()), (0, 0));
        // 1×1: [2] ~ [2], [2] !~ [3].
        assert!(is_similar(&qm(1, 1, &[2]), &qm(1, 1, &[2]))
            .unwrap()
            .is_some());
        assert_eq!(is_similar(&qm(1, 1, &[2]), &qm(1, 1, &[3])).unwrap(), None);
    }
}
