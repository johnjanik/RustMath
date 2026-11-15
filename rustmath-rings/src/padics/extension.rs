//! p-adic field extensions
//!
//! This module implements extensions of p-adic fields, including:
//! - Unramified extensions (extensions by roots of unity / finite field extensions)
//! - Eisenstein extensions (totally ramified extensions)
//! - General extensions (compositions of unramified and ramified)
//!
//! # Mathematical Background
//!
//! ## Ramification Theory
//!
//! For an extension K/qp of degree n, we have the fundamental relation:
//! ```text
//! n = e·f
//! ```
//! where:
//! - e = ramification index (index of ramification of p in K)
//! - f = residue field degree [k_K : F_p]
//!
//! ### Unramified Extensions
//! An extension is **unramified** if e = 1 and f = n. These extensions:
//! - Correspond to finite field extensions F_{p^f} / F_p
//! - Are obtained by adjoining roots of unity
//! - Have unique unramified extension of each degree
//! - Are Galois with cyclic Galois group of order f
//!
//! ### Eisenstein Extensions
//! An extension is **totally ramified** (Eisenstein) if e = n and f = 1. These:
//! - Are defined by Eisenstein polynomials
//! - An Eisenstein polynomial f(x) = a_n x^n + ... + a_1 x + a_0 satisfies:
//!   * v_p(a_i) ≥ 1 for i < n
//!   * v_p(a_n) = 0
//!   * v_p(a_0) = 1
//! - The root π of such a polynomial is a uniformizer of the extension
//!
//! ### General Extensions
//! Any extension can be written as a tower K = K_unr · K_ram where:
//! - K_unr/qp is the maximal unramified subextension
//! - K_ram/K_unr is totally ramified
//!
//! # Examples
//!
//! ```rust,ignore
//! use rustmath_rings::padics::extension::{PadicExtension, ExtensionType};
//! use rustmath_integers::Integer;
//! use rustmath_polynomials::UnivariatePolynomial;
//!
//! // Create Q_5(√5), an Eisenstein extension
//! let p = Integer::from(5);
//! // Polynomial x^2 - 5 (Eisenstein at 5)
//! let poly = UnivariatePolynomial::new(vec![
//!     Integer::from(-5),  // constant
//!     Integer::from(0),   // x coefficient
//!     Integer::from(1),   // x^2 coefficient
//! ]);
//! let ext = PadicExtension::eisenstein(p, poly, 10).unwrap();
//! ```

use crate::category_methods::{padic_base_prime_power, PrimePowerStatus};
use rustmath_core::{CommutativeRing, MathError, MultiplicativeOrder, Result, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use super::om_factorization::om_factorization;
use super::poly_arith::is_irreducible_mod_p;
use super::unramified::{unramified_modulus, UnramifiedExtension};
use super::{PadicInteger, PadicRational};
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::{Arc, OnceLock};

/// Structural sameness of extensions (prime, degree, defining polynomial,
/// precision, type — the display-only generator name is ignored).
///
/// `Arc::ptr_eq` alone is wrong here: e.g. [`PadicExtension::generator`]
/// clones the extension into a fresh `Arc`, and elements built from it must
/// still interoperate. This fixed the two formerly-ignored tests below.
fn same_extension(a: &Arc<PadicExtension>, b: &Arc<PadicExtension>) -> bool {
    Arc::ptr_eq(a, b)
        || (a.prime == b.prime
            && a.degree == b.degree
            && a.defining_polynomial == b.defining_polynomial
            && a.precision == b.precision
            && a.extension_type == b.extension_type)
}

/// Multiply two coefficient vectors and reduce modulo the defining polynomial
/// `def` (little-endian, length n+1, leading coefficient nonzero), exactly
/// over `Q`.
fn rational_polymul_mod(a: &[Rational], b: &[Rational], def: &[Rational]) -> Vec<Rational> {
    let n = def.len() - 1;
    let zero = Rational::from(0);
    if a.is_empty() || b.is_empty() {
        return vec![zero; n];
    }
    let mut res = vec![zero.clone(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            if bj.is_zero() {
                continue;
            }
            res[i + j] = res[i + j].clone() + ai.clone() * bj.clone();
        }
    }
    for k in (n..res.len()).rev() {
        if res[k].is_zero() {
            continue;
        }
        let factor = std::mem::replace(&mut res[k], zero.clone()) / def[n].clone();
        for i in 0..n {
            if def[i].is_zero() {
                continue;
            }
            res[k - n + i] = res[k - n + i].clone() - factor.clone() * def[i].clone();
        }
    }
    res.truncate(n);
    res.resize(n, zero);
    res
}

/// Exact determinant over `Q` by Gaussian elimination (product of pivots,
/// sign-tracked row swaps).
fn rational_det(mut m: Vec<Vec<Rational>>) -> Rational {
    let n = m.len();
    let mut det = Rational::from(1);
    for k in 0..n {
        let Some(piv) = (k..n).find(|&i| !m[i][k].is_zero()) else {
            return Rational::from(0);
        };
        if piv != k {
            m.swap(k, piv);
            det = -det;
        }
        let pivot = m[k][k].clone();
        det = det * pivot.clone();
        for i in k + 1..n {
            if m[i][k].is_zero() {
                continue;
            }
            let factor = m[i][k].clone() / pivot.clone();
            for j in k..n {
                m[i][j] = m[i][j].clone() - factor.clone() * m[k][j].clone();
            }
        }
    }
    det
}

/// Solve `m y = b` exactly over `Q` for a nonsingular square `m` (Gauss-Jordan elimination
/// with row swaps, the elimination of [`rational_det`] extended to a right-hand side);
/// `None` when `m` is singular.
fn rational_solve(mut m: Vec<Vec<Rational>>, mut b: Vec<Rational>) -> Option<Vec<Rational>> {
    let n = m.len();
    for k in 0..n {
        let piv = (k..n).find(|&i| !m[i][k].is_zero())?;
        m.swap(k, piv);
        b.swap(k, piv);
        let pivot = m[k][k].clone();
        for i in 0..n {
            if i == k || m[i][k].is_zero() {
                continue;
            }
            let factor = m[i][k].clone() / pivot.clone();
            let row_k = m[k][k..].to_vec();
            for (mij, mkj) in m[i][k..].iter_mut().zip(row_k) {
                *mij = mij.clone() - factor.clone() * mkj;
            }
            b[i] = b[i].clone() - factor * b[k].clone();
        }
    }
    Some((0..n).map(|i| b[i].clone() / m[i][i].clone()).collect())
}

/// `v_p(r)` of a nonzero rational, `p` prime (from the reduced fraction).
fn rational_valuation(r: &Rational, p: &Integer) -> i64 {
    i64::from(r.numerator().valuation(p)) - i64::from(r.denominator().valuation(p))
}

/// `ceil(a / b)` for `b > 0`.
fn ceil_div(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

/// How `PadicExtension::new` proved that `K = Q_p[x]/(f)` is a field, which also fixes the
/// valuation of K in the power basis `1, g, ..., g^(d-1)` (used by
/// [`PadicExtensionElement::inverse`]). Computed from f and a proven prime p only; the
/// [`ExtensionType`] label is not trusted.
///
/// Each shape proves f irreducible over Q_p: a linear f is; an Eisenstein f is (Eisenstein's
/// criterion for f/lc over Z_p); for an irreducible reduction, a factorisation over Q_p
/// descends by Gauss's lemma to Z_p[x] with unit leading coefficients, whose reduction
/// would factor f̄ (Neukirch, *Algebraic Number Theory*, II.(5.9)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldCertificate {
    /// `deg f = 1`: K = Q_p and v_K = v_p.
    Linear,
    /// f is Eisenstein at p: g is a uniformizer, e = d = deg f, `v_K(p) = d`, and
    /// `v_K(Σ c_i g^i) = min_i (d·v_p(c_i) + i)` (the terms have distinct valuations mod d;
    /// Serre, *Local Fields*, Ch. I §6).
    Eisenstein,
    /// `lc(f)` is a p-adic unit and `f mod p`, made monic, is irreducible over F_p: K/Q_p is
    /// unramified, O_K = Z_p[g], `v_K(p) = 1` and `v_K(Σ c_i g^i) = min_i v_p(c_i)` (the
    /// reduction of a primitive element is nonzero because `1, ḡ, ..., ḡ^(d-1)` is a basis of
    /// F_p[x]/(f̄); Serre, *Local Fields*, Ch. III §5).
    IrreducibleReduction,
}

/// The [`FieldCertificate`] of `f` at a proven prime `p`, checked in the order Linear,
/// Eisenstein, IrreducibleReduction.
fn certify_field(f: &UnivariatePolynomial<Integer>, p: &Integer) -> Option<FieldCertificate> {
    let coeffs = f.coefficients();
    let d = coeffs.len().checked_sub(1)?;
    if d == 1 {
        return Some(FieldCertificate::Linear);
    }
    if PadicExtension::is_eisenstein(f, p) {
        return Some(FieldCertificate::Eisenstein);
    }
    let lc = coeffs[d].modulo(p);
    if lc.is_zero() {
        return None;
    }
    // f mod p, made monic by lc^-1 mod p.
    let lc_inv = lc.mod_pow(&(p.clone() - Integer::from(2)), p).ok()?;
    let monic: Vec<Integer> = coeffs
        .iter()
        .map(|c| (c.clone() * lc_inv.clone()).modulo(p))
        .collect();
    match is_irreducible_mod_p(&monic, p) {
        Ok(true) => Some(FieldCertificate::IrreducibleReduction),
        _ => None,
    }
}

/// Type of p-adic extension
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtensionType {
    /// Unramified extension (e=1, f=degree)
    /// Obtained by finite field extension
    Unramified,

    /// Eisenstein extension (e=degree, f=1)
    /// Totally ramified extension defined by Eisenstein polynomial
    Eisenstein,

    /// General extension (both e>1 and f>1)
    /// Composition of unramified and ramified parts
    General { ramification_index: usize, residue_degree: usize },
}

/// p-adic field extension
///
/// Represents an extension K/qp where K is obtained by adjoining a root
/// of a polynomial to the base field qp.
#[derive(Clone, Debug)]
pub struct PadicExtension {
    /// Base prime p
    prime: Integer,

    /// Extension degree [K:qp]
    degree: usize,

    /// Ramification index e
    ramification_index: usize,

    /// Residue field degree f
    residue_degree: usize,

    /// Type of extension
    extension_type: ExtensionType,

    /// Defining polynomial (minimal polynomial of generator)
    /// Must be irreducible over qp
    defining_polynomial: UnivariatePolynomial<Integer>,

    /// Precision for p-adic computations
    precision: usize,

    /// Generator name (for display purposes)
    generator_name: String,

    /// Whether p is a prime power q^a (q proven prime), computed once in `new`
    /// (`padic_base_prime_power`).
    prime_power: PrimePowerStatus,

    /// How K was proven a field (`None`: not certified); needs p itself proven prime.
    field_certificate: Option<FieldCertificate>,

    /// f is squarefree over Q, so K is a finite product of fields (reduced).
    reduced: bool,

    /// `field_status()`, computed on first use. Behind an `Arc`, so that clones (for example
    /// the one in each element made by [`Self::generator`]) share one cache: the value
    /// depends only on p and f, which no method changes.
    field_status: Arc<OnceLock<Option<bool>>>,
}

impl PadicExtension {
    /// Create a new p-adic extension
    ///
    /// # Arguments
    ///
    /// * `prime` - Prime p
    /// * `defining_polynomial` - Irreducible polynomial defining the extension
    /// * `precision` - p-adic precision
    /// * `extension_type` - Type of extension (unramified, Eisenstein, or general)
    ///
    /// # Returns
    ///
    /// New p-adic extension or error if polynomial is not valid
    pub fn new(
        prime: Integer,
        defining_polynomial: UnivariatePolynomial<Integer>,
        precision: usize,
        extension_type: ExtensionType,
    ) -> Result<Self> {
        if prime <= Integer::one() {
            return Err(MathError::InvalidArgument("Prime must be > 1".to_string()));
        }

        if precision == 0 {
            return Err(MathError::InvalidArgument("Precision must be > 0".to_string()));
        }

        // A nonzero constant f gives the zero ring Q_p[x]/(c) = 0; the parent and hook
        // answers below (characteristic 0, nonzero) assume deg f >= 1.
        let degree = defining_polynomial
            .degree()
            .filter(|&d| d >= 1)
            .ok_or_else(|| {
                MathError::InvalidArgument("Polynomial must have positive degree".to_string())
            })?;

        // Compute ramification index and residue degree based on extension type
        let (ramification_index, residue_degree) = match &extension_type {
            ExtensionType::Unramified => (1, degree),
            ExtensionType::Eisenstein => (degree, 1),
            ExtensionType::General { ramification_index, residue_degree } => {
                if ramification_index * residue_degree != degree {
                    return Err(MathError::InvalidArgument(
                        "e·f must equal extension degree".to_string(),
                    ));
                }
                (*ramification_index, *residue_degree)
            }
        };

        let prime_power = padic_base_prime_power(&prime);
        let field_certificate = match prime_power {
            PrimePowerStatus::PrimePower { a: 1, .. } => {
                certify_field(&defining_polynomial, &prime)
            }
            _ => None,
        };
        let reduced = defining_polynomial.is_square_free();

        Ok(PadicExtension {
            prime,
            degree,
            ramification_index,
            residue_degree,
            extension_type,
            defining_polynomial,
            precision,
            generator_name: "π".to_string(),
            prime_power,
            field_certificate,
            reduced,
            field_status: Arc::new(OnceLock::new()),
        })
    }

    /// How `new` proved that K is a field (see [`FieldCertificate`]): `Some(_)` iff p is
    /// proven prime and f is linear, Eisenstein at p, or has a unit leading coefficient and
    /// an irreducible reduction mod p.
    ///
    /// For a proper prime power p = q^a (a > 1) K = Q_q\[x\]/(f) is decided by
    /// [`Self::field_status`] (OM at q), but no certificate is given: the coefficient bounds
    /// of [`PadicExtensionElement::inverse`] would build coefficients such as 1/q, and
    /// `PadicRational::from_rational` needs a denominator coprime to p after the powers of
    /// p are removed (1/2 in Q_4 returns `Err`).
    pub fn field_certificate(&self) -> Option<FieldCertificate> {
        self.field_certificate
    }

    /// `field_certificate().is_some()`.
    pub fn is_certified_field(&self) -> bool {
        self.field_certificate.is_some()
    }

    /// Whether f is squarefree over Q (`UnivariatePolynomial::is_square_free`). Then K is
    /// the product of `Q_p[x]/(g)` over the distinct irreducible factors g of f over Q_p, a
    /// reduced ring, for every p > 1.
    pub fn is_reduced(&self) -> bool {
        self.reduced
    }

    /// Whether K = Q_p[x]/(f) is a field. MAGMA requires f irreducible for `ext< L | f >`
    /// (Ch. 47 §47.4); here it is decided, in this order:
    /// 1. a [`FieldCertificate`]: `Some(true)`;
    /// 2. f not squarefree: `Some(false)` (f = g^2 h gives the nonzero nilpotent (gh)(α));
    /// 3. p proven not a prime power: `Some(false)` (Q_p ≅ ∏_{q | p} Q_q, so K is a product
    ///    of at least two nonzero rings);
    /// 4. deg f = 1: K = Q_p, a field iff p is a prime power;
    /// 5. p = q^a with q proven prime and q < 2^31: the OM factorisation
    ///    ([`om_factorization`], MacLane) over Q_q = Q_p of the monic integral transform
    ///    F(y) = lc^(d-1) f(y / lc) (K ≅ Q_p[y]/(F) by y = lc·g; F is squarefree iff f is):
    ///    `Some(one factor)`, and `None` when it returns `Err`;
    /// 6. otherwise `None`.
    ///
    /// Computed on first use and cached. The cache is shared by every clone of this
    /// extension (it depends only on p and f), so elements made by [`Self::generator`] or
    /// from a clone do not rerun the OM factorisation of step 5.
    pub fn field_status(&self) -> Option<bool> {
        *self.field_status.get_or_init(|| self.compute_field_status())
    }

    fn compute_field_status(&self) -> Option<bool> {
        if self.field_certificate.is_some() {
            return Some(true);
        }
        if !self.reduced {
            return Some(false);
        }
        let q = match &self.prime_power {
            PrimePowerStatus::NotPrimePower => return Some(false),
            PrimePowerStatus::Undecided => return None,
            PrimePowerStatus::PrimePower { q, .. } => q,
        };
        if self.degree == 1 {
            return Some(true);
        }
        let q = i64::try_from(q).ok().filter(|&q| q < (1i64 << 31))?;
        let coeffs = self.defining_polynomial.coefficients();
        let d = self.degree;
        let lc = coeffs[d].clone();
        // F(y) = lc^(d-1) f(y / lc) = Σ a_i lc^(d-1-i) y^i (monic, integral).
        let transformed: Vec<Rational> = coeffs
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let c = if i == d {
                    Integer::one()
                } else {
                    a.clone() * lc.pow((d - 1 - i) as u32)
                };
                Rational::from_integer(c)
            })
            .collect();
        om_factorization(&UnivariatePolynomial::new(transformed), q, 1)
            .ok()
            .map(|fac| fac.is_irreducible())
    }

    /// Create an unramified extension
    ///
    /// # Arguments
    ///
    /// * `prime` - Prime p
    /// * `degree` - Extension degree (= residue field degree)
    /// * `precision` - p-adic precision
    ///
    /// # Returns
    ///
    /// Unramified extension of degree `degree`
    ///
    /// # Note
    ///
    /// The defining polynomial is the certified irreducible-mod-p lift from
    /// [`unramified_modulus`]: the Conway polynomial when the verified table
    /// of `rustmath-finitefields` covers `(p, degree)`, otherwise the first
    /// monic irreducible found by enumeration — in both cases re-certified
    /// irreducible mod p at construction. (The previous placeholder
    /// `x^degree - (1+p)` was mathematically wrong: e.g. `x^2 - 6` factors as
    /// `(x-1)(x+1)` mod 5, so the "extension" was not a field.)
    pub fn unramified(prime: Integer, degree: usize, precision: usize) -> Result<Self> {
        let poly = unramified_modulus(&prime, degree)?;
        Self::new(prime, poly, precision, ExtensionType::Unramified)
    }

    /// Create an Eisenstein extension
    ///
    /// # Arguments
    ///
    /// * `prime` - Prime p
    /// * `eisenstein_polynomial` - Eisenstein polynomial
    /// * `precision` - p-adic precision
    ///
    /// # Returns
    ///
    /// Eisenstein extension or error if polynomial is not Eisenstein
    pub fn eisenstein(
        prime: Integer,
        eisenstein_polynomial: UnivariatePolynomial<Integer>,
        precision: usize,
    ) -> Result<Self> {
        // Verify that the polynomial is Eisenstein
        if !Self::is_eisenstein(&eisenstein_polynomial, &prime) {
            return Err(MathError::InvalidArgument(
                "Polynomial is not Eisenstein".to_string(),
            ));
        }

        Self::new(prime, eisenstein_polynomial, precision, ExtensionType::Eisenstein)
    }

    /// Check if a polynomial is Eisenstein at prime p
    ///
    /// A polynomial f(x) = a_n x^n + ... + a_1 x + a_0 is Eisenstein at p if:
    /// - a_n is not divisible by p (leading coefficient is a unit)
    /// - a_i is divisible by p for i < n
    /// - a_0 is divisible by p but not by p^2
    fn is_eisenstein(poly: &UnivariatePolynomial<Integer>, prime: &Integer) -> bool {
        let coeffs = poly.coefficients();
        if coeffs.is_empty() {
            return false;
        }

        let n = coeffs.len() - 1;

        // Check leading coefficient is not divisible by p
        if coeffs[n].clone() % prime.clone() == Integer::zero() {
            return false;
        }

        // Check intermediate coefficients are divisible by p
        for i in 1..n {
            if coeffs[i].clone() % prime.clone() != Integer::zero() {
                return false;
            }
        }

        // Check constant term: divisible by p but not p^2
        let c0 = &coeffs[0];
        let valuation = c0.valuation(prime);

        valuation == 1
    }

    /// Get the prime p
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Get the extension degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the ramification index
    pub fn ramification_index(&self) -> usize {
        self.ramification_index
    }

    /// Get the residue field degree
    pub fn residue_degree(&self) -> usize {
        self.residue_degree
    }

    /// Get the extension type
    pub fn extension_type(&self) -> &ExtensionType {
        &self.extension_type
    }

    /// Get the defining polynomial
    pub fn defining_polynomial(&self) -> &UnivariatePolynomial<Integer> {
        &self.defining_polynomial
    }

    /// Get the precision
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Check if extension is unramified
    pub fn is_unramified(&self) -> bool {
        self.ramification_index == 1
    }

    /// Check if extension is totally ramified
    pub fn is_totally_ramified(&self) -> bool {
        self.residue_degree == 1
    }

    /// Check if extension is Galois
    ///
    /// All unramified extensions are Galois (cyclic).
    /// Eisenstein extensions may or may not be Galois.
    pub fn is_galois(&self) -> bool {
        match self.extension_type {
            ExtensionType::Unramified => true,
            ExtensionType::Eisenstein => {
                // For Eisenstein extensions, we'd need to check if the
                // polynomial splits completely. This is non-trivial.
                // For now, return false conservatively.
                false
            }
            ExtensionType::General { .. } => false,
        }
    }

    /// Create a generator element
    ///
    /// Returns an element representing the root of the defining polynomial
    /// (for a degree-1 extension it returns 0, not the root -a_0/a_1; a known defect).
    pub fn generator(&self) -> PadicExtensionElement {
        // Generator is represented as x in the quotient ring qp[x]/(f(x))
        // where f is the defining polynomial
        let mut coeffs = vec![PadicRational::from_padic_integer(
            PadicInteger::zero(self.prime.clone(), self.precision).unwrap()
        ); self.degree];

        // Set coefficient of x^1 to 1
        if self.degree >= 2 {
            coeffs[1] = PadicRational::from_padic_integer(
                PadicInteger::one(self.prime.clone(), self.precision).unwrap()
            );
        }

        PadicExtensionElement::new(Arc::new(self.clone()), coeffs)
    }

    /// Set the generator name for display
    pub fn set_generator_name(&mut self, name: String) {
        self.generator_name = name;
    }

    /// Get the generator name
    pub fn generator_name(&self) -> &str {
        &self.generator_name
    }
}

impl fmt::Display for PadicExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.extension_type {
            ExtensionType::Unramified => {
                write!(f, "Unramified extension of Q_{} of degree {}", self.prime, self.degree)
            }
            ExtensionType::Eisenstein => {
                write!(f, "Eisenstein extension of Q_{} by {}", self.prime, self.defining_polynomial)
            }
            ExtensionType::General { ramification_index, residue_degree } => {
                write!(
                    f,
                    "Extension of Q_{} with e={}, f={}",
                    self.prime, ramification_index, residue_degree
                )
            }
        }
    }
}

/// Element of a p-adic field extension
///
/// Represented as a polynomial in the generator modulo the defining polynomial.
/// For an extension K = qp[π]/(f(π)), elements are represented as
/// a_0 + a_1·π + ... + a_{n-1}·π^{n-1} where a_i ∈ qp.
///
/// # Model, equality and unbound constants
///
/// The model is that of [`PadicRational`] and [`PadicInteger`] (whose docs list the
/// departures from MAGMA Ch. 47): `==` compares the coefficients by value to their common
/// precision, `is_zero`/`is_one` are "to the known precision", and every ring predicate is
/// a proof for every element of K that the approximation represents, under the convention
/// that `is_zero` and `is_one` are exact: an element indistinguishable from 0 (from 1) is
/// treated as 0 (as 1). The second half matters for `±1`: `unit_status` says `Some(true)`,
/// `unit_inverse` returns the element itself, and the order is `Finite(1)` or `Finite(2)`,
/// in any K, although a lift such as `1 + 5^10` of `1 + O(5^10)` has infinite order.
/// `is_zero` and `is_one` never both hold.
///
/// The static `Ring::zero()`/`one()` return an **unbound** element: the unbound
/// `PadicRational` constant c times 1, with no extension (MAGMA `L ! c`, Ch. 47 §47.7.1). It
/// binds on contact: `x + c` adds c to the constant coefficient (c = 0 leaves x unchanged),
/// `x * c` scales every coefficient by c, `x * 0` and `0 * x` are the unbound 0, and
/// `x == c` iff the constant coefficient equals c and the others are zero.
/// [`PadicExtensionElement::new`] and [`PadicExtensionElement::from_base_field`] bind
/// unbound coefficients at the extension's precision. [`PadicExtensionElement::extension`]
/// panics on an unbound element, [`PadicExtensionElement::coefficients`] is the
/// one-element slice `[c]`, and `norm`/`trace` return `Err(InvalidArgument)` (they depend
/// on the degree).
#[derive(Clone, Debug)]
pub struct PadicExtensionElement {
    /// The extension field this element belongs to; `None` for an unbound constant.
    extension: Option<Arc<PadicExtension>>,

    /// Coefficients a_0, a_1, ..., a_{n-1} where element = Σ a_i π^i
    /// Length must be equal to extension degree; `[c]` (unbound) for an unbound constant.
    coefficients: Vec<PadicRational>,
}

impl PadicExtensionElement {
    /// Create a new extension element
    ///
    /// # Arguments
    ///
    /// * `extension` - The extension field
    /// * `coefficients` - Coefficients in terms of powers of the generator; an unbound
    ///   coefficient (from `Ring::zero()`/`one()`) is bound at the extension's precision
    pub fn new(extension: Arc<PadicExtension>, coefficients: Vec<PadicRational>) -> Self {
        assert_eq!(
            coefficients.len(),
            extension.degree(),
            "Number of coefficients must match extension degree"
        );
        let coefficients = coefficients
            .into_iter()
            .map(|c| bind_coefficient(c, &extension))
            .collect();

        PadicExtensionElement {
            extension: Some(extension),
            coefficients,
        }
    }

    /// The unbound integer constant `c`.
    pub(crate) fn unbound(c: Integer) -> Self {
        PadicExtensionElement {
            extension: None,
            coefficients: vec![PadicRational::unbound(c)],
        }
    }

    /// Whether this element belongs to an extension (see the type docs).
    pub fn is_bound(&self) -> bool {
        self.extension.is_some()
    }

    /// The extension, with the documented panic for an unbound constant.
    fn ext(&self, op: &str) -> &Arc<PadicExtension> {
        match &self.extension {
            Some(e) => e,
            None => panic!(
                "PadicExtensionElement::{op}: the unbound constant {} (from Ring::zero()/one()) \
                 has no extension; combine it with a bound element, or use \
                 PadicExtensionElement::zero(ext)/one(ext)",
                self.constant()
            ),
        }
    }

    /// The integer constant of an unbound element.
    fn constant(&self) -> &Integer {
        debug_assert!(!self.is_bound());
        self.coefficients[0].unit().value()
    }

    /// `self` (unbound) bound into `ext`: the constant at the extension's precision.
    fn bind_into(&self, ext: &Arc<PadicExtension>) -> PadicExtensionElement {
        PadicExtensionElement::from_base_field(
            ext.clone(),
            PadicRational::bind_integer(self.constant(), ext.prime(), ext.precision()),
        )
    }

    /// Create zero element
    pub fn zero(extension: Arc<PadicExtension>) -> Self {
        let coeffs = vec![
            PadicRational::from_padic_integer(
                PadicInteger::zero(extension.prime().clone(), extension.precision()).unwrap()
            );
            extension.degree()
        ];
        Self::new(extension, coeffs)
    }

    /// Create one element
    pub fn one(extension: Arc<PadicExtension>) -> Self {
        let mut coeffs = vec![
            PadicRational::from_padic_integer(
                PadicInteger::zero(extension.prime().clone(), extension.precision()).unwrap()
            );
            extension.degree()
        ];
        coeffs[0] = PadicRational::from_padic_integer(
            PadicInteger::one(extension.prime().clone(), extension.precision()).unwrap()
        );
        Self::new(extension, coeffs)
    }

    /// Create element from a base field element (qp); an unbound `value` is bound at the
    /// extension's precision
    pub fn from_base_field(extension: Arc<PadicExtension>, value: PadicRational) -> Self {
        let mut coeffs = vec![
            PadicRational::from_padic_integer(
                PadicInteger::zero(extension.prime().clone(), extension.precision()).unwrap()
            );
            extension.degree()
        ];
        coeffs[0] = value;
        Self::new(extension, coeffs)
    }

    /// Get the extension field
    ///
    /// # Panics
    ///
    /// On an unbound constant (see the type docs).
    pub fn extension(&self) -> &Arc<PadicExtension> {
        self.ext("extension")
    }

    /// Get the coefficients (the one-element slice `[c]` for an unbound constant)
    pub fn coefficients(&self) -> &[PadicRational] {
        &self.coefficients
    }

    /// The n×n multiplication-by-α matrix over `Q` in the basis
    /// `1, π, ..., π^{n-1}` (column j = coefficients of `α · π^j`), computed
    /// exactly from the canonical rational representatives of the
    /// coefficients, with exact reduction modulo the defining polynomial.
    fn multiplication_matrix_rational(&self) -> Vec<Vec<Rational>> {
        let ext = self.ext("multiplication_matrix_rational");
        let n = ext.degree();
        let def: Vec<Rational> = ext
            .defining_polynomial()
            .coefficients()
            .iter()
            .map(|c| Rational::new(c.clone(), Integer::one()).expect("denominator 1"))
            .collect();
        let a: Vec<Rational> = self
            .coefficients
            .iter()
            .map(PadicRational::representative)
            .collect();
        let mut cols = Vec::with_capacity(n);
        for j in 0..n {
            let mut basis = vec![Rational::from(0); n];
            basis[j] = Rational::from(1);
            cols.push(rational_polymul_mod(&a, &basis, &def));
        }
        (0..n)
            .map(|i| (0..n).map(|j| cols[j][i].clone()).collect())
            .collect()
    }

    fn unbound_degree_error(&self, op: &str) -> MathError {
        MathError::InvalidArgument(format!(
            "PadicExtensionElement::{op}: the unbound constant {} has no extension, so no \
             degree; bind it first",
            self.coefficients[0].unit().value()
        ))
    }

    /// Compute the norm N_{K/qp}(α)
    ///
    /// The norm is the product of all Galois conjugates, or equivalently the
    /// determinant of the multiplication-by-α map — computed here as the
    /// exact rational determinant of the multiplication matrix (any degree,
    /// any extension type), then read back into `qp` at the extension's
    /// precision. This replaced the old degree>2 `unimplemented!` facade;
    /// values are pinned by sympy-verified tests (e.g. `N(3+2√5) = -11` over
    /// `Q_5(√5)`, `N(π) = 2` over `Q_2(2^{1/3})`, `N(g) = -3` for the degree-3
    /// unramified extension of `Q_5`). An unbound constant gives `Err(InvalidArgument)`.
    pub fn norm(&self) -> Result<PadicRational> {
        let Some(ext) = &self.extension else {
            return Err(self.unbound_degree_error("norm"));
        };
        let det = rational_det(self.multiplication_matrix_rational());
        if det.is_zero() {
            return Ok(PadicRational::from_padic_integer(
                PadicInteger::zero(ext.prime().clone(), ext.precision())?,
            ));
        }
        PadicRational::from_rational(det, ext.prime().clone(), ext.precision())
    }

    /// Compute the trace Tr_{K/qp}(α)
    ///
    /// The trace is the sum of all Galois conjugates, or equivalently the
    /// trace of the multiplication-by-α map (exact rational diagonal sum of
    /// the multiplication matrix; any degree, any extension type). An unbound constant
    /// gives `Err(InvalidArgument)`.
    pub fn trace(&self) -> Result<PadicRational> {
        let Some(ext) = &self.extension else {
            return Err(self.unbound_degree_error("trace"));
        };
        let m = self.multiplication_matrix_rational();
        let mut t = Rational::from(0);
        for (i, row) in m.iter().enumerate() {
            t = t + row[i].clone();
        }
        if t.is_zero() {
            return Ok(PadicRational::from_padic_integer(
                PadicInteger::zero(ext.prime().clone(), ext.precision())?,
            ));
        }
        PadicRational::from_rational(t, ext.prime().clone(), ext.precision())
    }

    /// The valuation data of a bound element under a [`FieldCertificate`]: `(e, s)` with
    /// `w_i(t) = e·t + s_i` (`Eisenstein`: e = d, s_i = i; otherwise e = 1, s_i = 0), the
    /// exact representatives `r_i`, `V = min_{r_i != 0} w_i(v_p(r_i))` (= v_K of the
    /// representative; `None` when every r_i is 0) and `E = min_i w_i(A_i)` with `A_i` the
    /// absolute precision of coefficient i (every element the approximation represents
    /// differs from the representative by an element of valuation >= E).
    fn certified_valuation(
        &self,
        cert: FieldCertificate,
    ) -> (i64, Vec<i64>, Vec<Rational>, Option<i64>, i64) {
        let ext = self.ext("certified_valuation");
        let d = ext.degree();
        let p = ext.prime();
        let (e, s): (i64, Vec<i64>) = match cert {
            FieldCertificate::Eisenstein => (d as i64, (0..d as i64).collect()),
            FieldCertificate::Linear | FieldCertificate::IrreducibleReduction => (1, vec![0; d]),
        };
        let reps: Vec<Rational> = self
            .coefficients
            .iter()
            .map(PadicRational::representative)
            .collect();
        let v = reps
            .iter()
            .zip(&s)
            .filter(|(r, _)| !r.is_zero())
            .map(|(r, si)| e * rational_valuation(r, p) + si)
            .min();
        let error = self
            .coefficients
            .iter()
            .zip(&s)
            .map(|(c, si)| e * c.absolute_precision() + si)
            .min()
            .expect("degree >= 1");
        (e, s, reps, v, error)
    }

    /// MAGMA `1/x` in K = Q_p[x]/(f) (Ch. 47 §47.8.1). `Ok(y)`: y is the inverse of every
    /// element of K that x approximates; coefficient i of y is known to absolute precision
    /// `P_i = ceil((B - s_i)/e)`, `B = E - 2V`, in the notation of the valuation data
    /// (`v_K(x) = V` exactly, and `v_K(x^-1 - y_rep) >= B`).
    ///
    /// Algorithm: solve `M_x y = e_1` exactly over Q (`M_x` the multiplication matrix of
    /// the representative; f is irreducible over Q_p, hence over Q, so `M_x` is
    /// nonsingular). For every lift `x = x_rep + ε`, `v_K(ε) >= E > V`, so `v_K(x) = V` and
    /// `x^-1 - y_rep = -ε/(x·x_rep)` has valuation `>= E - 2V`. For the three certified
    /// shapes the valuation formula is an equality, which bounds each coefficient. The
    /// precisions agree with PARI's own p-adic polmod inverses on the tests.
    ///
    /// Errors:
    /// * unbound c: `±1` give themselves, `0` gives `Err(NotInvertible)`, any other c
    ///   `Err(InvalidArgument)` (1/c is not an integer constant; bind it first);
    /// * x weakly zero: `Err(NotInvertible)` (the convention);
    /// * no [`FieldCertificate`] (p not proven prime, K not certified a field, or certified
    ///   by OM only, which gives no power-basis valuation formula): `Err(NotImplemented)`;
    /// * x not known well enough (`V >= E`): `Err(NumericalError)`, undecided.
    pub fn inverse(&self) -> Result<PadicExtensionElement> {
        let Some(ext) = &self.extension else {
            let c = self.constant();
            if c.is_one() || (-c.clone()).is_one() {
                return Ok(self.clone());
            }
            if c.is_zero() {
                return Err(MathError::NotInvertible);
            }
            return Err(MathError::InvalidArgument(format!(
                "PadicExtensionElement::inverse: 1/{c} is not an integer constant; bind it first"
            )));
        };
        if self.is_zero() {
            return Err(MathError::NotInvertible);
        }
        let Some(cert) = ext.field_certificate() else {
            return Err(MathError::NotImplemented(
                "PadicExtensionElement::inverse: the extension has no field certificate (p is \
                 not proven prime, K may not be a field, or K is a field only by the OM \
                 factorisation, which gives no coefficient bounds without an integral basis)"
                    .to_string(),
            ));
        };
        let (e, s, reps, v, error) = self.certified_valuation(cert);
        let v = v.expect("an element that is not weakly zero has a nonzero representative");
        if v >= error {
            return Err(MathError::NumericalError(format!(
                "PadicExtensionElement::inverse: undecided, the element is not known well \
                 enough (v_K of the representative {v} >= error bound {error})"
            )));
        }
        let d = ext.degree();
        let m = self.multiplication_matrix_rational();
        let mut e1 = vec![Rational::from(0); d];
        e1[0] = Rational::from(1);
        let y = rational_solve(m, e1).ok_or_else(|| {
            MathError::InvalidOperation(
                "PadicExtensionElement::inverse: singular multiplication matrix for a nonzero \
                 element of a certified field"
                    .to_string(),
            )
        })?;
        debug_assert_eq!(reps.len(), d);
        let b = error - 2 * v;
        let p = ext.prime();
        let cap = ext.precision();
        let coeffs = y
            .iter()
            .zip(&s)
            .map(|(yi, si)| {
                PadicRational::from_rational_to_absolute_precision(
                    yi,
                    p,
                    ceil_div(b - si, e),
                    cap,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PadicExtensionElement::new(ext.clone(), coeffs))
    }
}

/// An unbound coefficient bound at the extension's precision; a bound one unchanged.
fn bind_coefficient(c: PadicRational, ext: &PadicExtension) -> PadicRational {
    if c.is_bound() {
        c
    } else {
        PadicRational::bind_integer(c.unit().value(), ext.prime(), ext.precision())
    }
}

impl fmt::Display for PadicExtensionElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(ext) = &self.extension else {
            return write!(f, "{}", self.constant());
        };
        let mut first = true;
        let gen_name = ext.generator_name();

        for (i, coeff) in self.coefficients.iter().enumerate() {
            if coeff.unit().is_zero() {
                continue;
            }

            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if i == 0 {
                write!(f, "{}", coeff)?;
            } else if i == 1 {
                write!(f, "{}·{}", coeff, gen_name)?;
            } else {
                write!(f, "{}·{}^{}", coeff, gen_name, i)?;
            }
        }

        if first {
            write!(f, "0")?;
        }

        Ok(())
    }
}

/// Coefficient-wise value equality ([`PadicRational`]'s); an unbound c equals x iff x's
/// constant coefficient equals c and the others are zero.
impl PartialEq for PadicExtensionElement {
    fn eq(&self, other: &Self) -> bool {
        match (&self.extension, &other.extension) {
            (None, None) => self.coefficients[0] == other.coefficients[0],
            (Some(_), None) => {
                self.coefficients[0] == other.coefficients[0]
                    && self.coefficients[1..].iter().all(|c| c.is_zero())
            }
            (None, Some(_)) => other == self,
            (Some(a), Some(b)) => {
                same_extension(a, b)
                    && self
                        .coefficients
                        .iter()
                        .zip(other.coefficients.iter())
                        .all(|(a, b)| a == b)
            }
        }
    }
}

impl Add for PadicExtensionElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match (self.is_bound(), other.is_bound()) {
            (false, false) => PadicExtensionElement {
                extension: None,
                coefficients: vec![
                    self.coefficients[0].clone() + other.coefficients[0].clone(),
                ],
            },
            (true, false) | (false, true) => {
                let (mut x, c) = if self.is_bound() { (self, other) } else { (other, self) };
                // c = 0 leaves x (and its precision) unchanged (PadicRational's rule).
                let c0 = std::mem::replace(&mut x.coefficients[0], PadicRational::unbound(Integer::zero()));
                x.coefficients[0] = c0 + c.coefficients[0].clone();
                x
            }
            (true, true) => {
                let ext = self.ext("add").clone();
                assert!(
                    same_extension(&ext, other.ext("add")),
                    "Elements must be from the same extension"
                );

                let coeffs: Vec<PadicRational> = self
                    .coefficients
                    .into_iter()
                    .zip(other.coefficients)
                    .map(|(a, b)| a + b)
                    .collect();

                PadicExtensionElement::new(ext, coeffs)
            }
        }
    }
}

impl Sub for PadicExtensionElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if !(self.is_bound() && other.is_bound()) {
            return self + (-other);
        }
        let ext = self.ext("sub").clone();
        assert!(
            same_extension(&ext, other.ext("sub")),
            "Elements must be from the same extension"
        );

        let coeffs: Vec<PadicRational> = self
            .coefficients
            .into_iter()
            .zip(other.coefficients)
            .map(|(a, b)| a - b)
            .collect();

        PadicExtensionElement::new(ext, coeffs)
    }
}

impl Mul for PadicExtensionElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let exact_zero = |x: &Self| !x.is_bound() && x.constant().is_zero();
        if exact_zero(&self) || exact_zero(&other) {
            return PadicExtensionElement::unbound(Integer::zero());
        }
        match (self.is_bound(), other.is_bound()) {
            (false, false) => {
                return PadicExtensionElement {
                    extension: None,
                    coefficients: vec![
                        self.coefficients[0].clone() * other.coefficients[0].clone(),
                    ],
                }
            }
            (true, false) | (false, true) => {
                // Q_p-linear scaling by the constant (PadicRational binds it).
                let (x, c) = if self.is_bound() { (self, other) } else { (other, self) };
                let c = c.coefficients[0].clone();
                let ext = x.ext("mul").clone();
                let coeffs = x.coefficients.into_iter().map(|a| a * c.clone()).collect();
                return PadicExtensionElement::new(ext, coeffs);
            }
            (true, true) => {}
        }
        let ext = self.ext("mul").clone();
        assert!(
            same_extension(&ext, other.ext("mul")),
            "Elements must be from the same extension"
        );

        let n = ext.degree();
        let mut result_coeffs = vec![
            PadicRational::from_padic_integer(
                PadicInteger::zero(ext.prime().clone(), ext.precision()).unwrap()
            );
            2 * n - 1
        ];

        // Polynomial multiplication
        for (i, a) in self.coefficients.iter().enumerate() {
            for (j, b) in other.coefficients.iter().enumerate() {
                result_coeffs[i + j] = result_coeffs[i + j].clone() + (a.clone() * b.clone());
            }
        }

        // Reduce modulo defining polynomial
        // For a polynomial f(x) of degree n with integer coefficients,
        // we need to reduce terms x^k where k >= n using f(x) = 0
        // This is a simplified reduction - full implementation would use polynomial division

        let def_poly = ext.defining_polynomial();
        let def_coeffs = def_poly.coefficients();

        // Reduce high degree terms
        for k in (n..result_coeffs.len()).rev() {
            if !result_coeffs[k].unit().is_zero() {
                let factor = result_coeffs[k].clone();
                result_coeffs[k] = PadicRational::from_padic_integer(
                    PadicInteger::zero(ext.prime().clone(), ext.precision()).unwrap()
                );

                // x^k ≡ -(a_0 + a_1·x + ... + a_{n-1}·x^{n-1})/a_n · x^{k-n} mod f
                // where f = a_n·x^n + ... + a_0
                let leading = PadicRational::from_padic_integer(
                    PadicInteger::from_integer(
                        def_coeffs[n].clone(),
                        ext.prime().clone(),
                        ext.precision(),
                    ).unwrap()
                );

                for i in 0..n {
                    let coeff = PadicRational::from_padic_integer(
                        PadicInteger::from_integer(
                            def_coeffs[i].clone(),
                            ext.prime().clone(),
                            ext.precision(),
                        ).unwrap()
                    );

                    if k - n + i < result_coeffs.len() {
                        result_coeffs[k - n + i] = result_coeffs[k - n + i].clone()
                            - (factor.clone() * coeff / leading.clone());
                    }
                }
            }
        }

        // Truncate to extension degree
        result_coeffs.truncate(n);
        while result_coeffs.len() < n {
            result_coeffs.push(PadicRational::from_padic_integer(
                PadicInteger::zero(ext.prime().clone(), ext.precision()).unwrap()
            ));
        }

        PadicExtensionElement::new(ext, result_coeffs)
    }
}

impl Neg for PadicExtensionElement {
    type Output = Self;

    fn neg(self) -> Self {
        let coefficients = self.coefficients.into_iter().map(|c| -c).collect();
        PadicExtensionElement {
            extension: self.extension,
            coefficients,
        }
    }
}

impl Ring for PadicExtensionElement {
    /// The unbound constant 0 (see the type docs).
    fn zero() -> Self {
        PadicExtensionElement::unbound(Integer::zero())
    }

    /// The unbound constant 1 (see the type docs).
    fn one() -> Self {
        PadicExtensionElement::unbound(Integer::one())
    }

    fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c.unit().is_zero())
    }

    /// The constant coefficient is one by value (`PadicRational::is_one`, false when it is
    /// weakly zero) and the others are zero: `(g·g)·(1/5)` in Q_5(√5), whose constant
    /// coefficient is stored as (5, -1), is one.
    fn is_one(&self) -> bool {
        self.coefficients[0].is_one() && self.coefficients[1..].iter().all(|c| c.is_zero())
    }

    /// Square-and-multiply; `x.pow(0)` of a bound `x` is `one(ext)`; an unbound c gives the
    /// unbound c^n.
    fn pow(&self, n: u32) -> Self {
        let Some(ext) = &self.extension else {
            return PadicExtensionElement {
                extension: None,
                coefficients: vec![self.coefficients[0].pow(n)],
            };
        };
        if n == 0 {
            return PadicExtensionElement::one(ext.clone());
        }
        let mut result = self.clone();
        for bit in (0..(31 - n.leading_zeros())).rev() {
            result = result.clone() * result;
            if (n >> bit) & 1 == 1 {
                result = result * self.clone();
            }
        }
        result
    }

    /// Weakly zero: `Some(false)`; `±1`: `Some(true)`; K proven a field
    /// ([`PadicExtension::field_status`]): `Some(true)` (a nonzero element of a field; every
    /// lift of an element that is not weakly zero is nonzero); otherwise `None`. Unbound c:
    /// `Some(c != 0)` (a nonzero integer is a unit of Q_p ⊂ K).
    fn unit_status(&self) -> Option<bool> {
        let Some(ext) = &self.extension else {
            return self.coefficients[0].unit_status();
        };
        if self.is_zero() {
            return Some(false);
        }
        if self.is_one() || (-self.clone()).is_one() {
            return Some(true);
        }
        (ext.field_status() == Some(true)).then_some(true)
    }

    /// Weakly zero: `Some(true)` (the convention); f squarefree over Q
    /// ([`PadicExtension::is_reduced`]): `Some(false)` (K is a product of fields); otherwise
    /// `None` (K has nonzero nilpotents, and this element may or may not be one). Unbound c:
    /// `Some(c == 0)`.
    fn nilpotent_status(&self) -> Option<bool> {
        let Some(ext) = &self.extension else {
            return Some(self.coefficients[0].is_zero());
        };
        if self.is_zero() {
            Some(true)
        } else if ext.is_reduced() {
            Some(false)
        } else {
            None
        }
    }

    /// [`PadicExtensionElement::inverse`] (`NotInvertible` only for a weakly zero element or
    /// the unbound 0; every other `Err` is undecided). When that is undecided for `±1`, the
    /// element itself, as in the core default: under the convention that `is_one` is exact
    /// (see the type docs), `±1` are units with inverse `±1` in any K, which is also why
    /// `unit_status` answers `Some(true)` for them.
    fn unit_inverse(&self) -> Result<Self> {
        match self.inverse() {
            Err(e) if !matches!(e, MathError::NotInvertible) => {
                if self.is_one() || (-self.clone()).is_one() {
                    Ok(self.clone())
                } else {
                    Err(e)
                }
            }
            r => r,
        }
    }

    /// Weakly zero: `NotAUnit`; `1`: `Finite(1)`; `-1`: `Finite(2)`; with a
    /// [`FieldCertificate`], `V < E` and `V != 0` (valuation data of
    /// [`PadicExtensionElement::inverse`]): `Infinite` (v_K(x) = V, so v_K(x^m) = mV != 0);
    /// otherwise `Undecided` (units of valuation 0 need more than a valuation: the roots of
    /// unity of K). Unbound c: `PadicRational`'s rule (`|c| >= 2` is `Infinite`).
    fn multiplicative_order_status(&self) -> MultiplicativeOrder {
        let Some(ext) = &self.extension else {
            return self.coefficients[0].multiplicative_order_status();
        };
        if self.is_zero() {
            return MultiplicativeOrder::NotAUnit;
        }
        if self.is_one() {
            return MultiplicativeOrder::Finite(1);
        }
        if (-self.clone()).is_one() {
            return MultiplicativeOrder::Finite(2);
        }
        if let Some(cert) = ext.field_certificate() {
            let (_, _, _, v, error) = self.certified_valuation(cert);
            if let Some(v) = v {
                if v < error && v != 0 {
                    return MultiplicativeOrder::Infinite;
                }
            }
        }
        MultiplicativeOrder::Undecided
    }

    /// `Some(0)`: Q ⊂ K.
    fn static_characteristic() -> Option<u64> {
        Some(0)
    }

    fn static_is_commutative() -> Option<bool> {
        Some(true)
    }

    const HAS_UNBOUND_CONSTANTS: bool = true;

    fn is_unbound_constant(&self) -> bool {
        !self.is_bound()
    }

    /// MAGMA `L ! c` in the model's extension, at the extension's precision.
    fn bind_to_parent_of(&self, model: &Self) -> Option<Self> {
        match (&self.extension, &model.extension) {
            (None, Some(ext)) => Some(self.bind_into(ext)),
            _ => None,
        }
    }
}

impl CommutativeRing for PadicExtensionElement {}

/// Embedding of p-adic extensions
///
/// Provides functionality for embedding p-adic extensions into larger extensions
/// or into the algebraic closure of qp.
#[derive(Clone, Debug)]
pub struct PadicEmbedding {
    /// Source extension
    source: Arc<PadicExtension>,

    /// Target extension (or None if embedding into algebraic closure)
    target: Option<Arc<PadicExtension>>,

    /// Embedding map (coefficients of source generator in terms of target)
    /// For source K and target L, this represents the image of the generator π_K
    /// as an element of L
    generator_image: Vec<PadicRational>,
}

impl PadicEmbedding {
    /// Create an embedding from one extension to another
    ///
    /// # Arguments
    ///
    /// * `source` - Source extension
    /// * `target` - Target extension
    /// * `generator_image` - Image of the source generator in the target
    ///
    /// # Returns
    ///
    /// Embedding or error if incompatible
    pub fn new(
        source: Arc<PadicExtension>,
        target: Arc<PadicExtension>,
        generator_image: Vec<PadicRational>,
    ) -> Result<Self> {
        // Verify that source and target have the same prime
        if source.prime() != target.prime() {
            return Err(MathError::InvalidArgument(
                "Source and target must have the same prime".to_string(),
            ));
        }

        // Verify that target degree is divisible by source degree
        if target.degree() % source.degree() != 0 {
            return Err(MathError::InvalidArgument(
                "Target degree must be divisible by source degree".to_string(),
            ));
        }

        // Verify generator_image has correct length
        if generator_image.len() != target.degree() {
            return Err(MathError::InvalidArgument(
                "Generator image must have length equal to target degree".to_string(),
            ));
        }

        Ok(PadicEmbedding {
            source,
            target: Some(target),
            generator_image,
        })
    }

    /// Create an embedding into the algebraic closure
    ///
    /// This is a formal embedding that doesn't require an explicit target.
    /// The algebraic closure Q̄_p is represented implicitly.
    pub fn into_algebraic_closure(source: Arc<PadicExtension>) -> Self {
        // For embedding into algebraic closure, we use the generator itself
        let n = source.degree();
        let mut coeffs = vec![
            PadicRational::from_padic_integer(
                PadicInteger::zero(source.prime().clone(), source.precision()).unwrap()
            );
            n
        ];

        if n >= 2 {
            coeffs[1] = PadicRational::from_padic_integer(
                PadicInteger::one(source.prime().clone(), source.precision()).unwrap()
            );
        }

        PadicEmbedding {
            source,
            target: None,
            generator_image: coeffs,
        }
    }

    /// Apply the embedding to an element
    ///
    /// Maps an element from the source extension to the target extension
    pub fn apply(&self, element: &PadicExtensionElement) -> Result<PadicExtensionElement> {
        // An unbound constant (from `Ring::zero()`/`one()`) is first bound into the source.
        let bound;
        let element = if element.is_bound() {
            element
        } else {
            bound = element.bind_into(&self.source);
            &bound
        };
        // Verify element is from source extension (structurally: fresh Arcs
        // of the same extension interoperate)
        if !same_extension(element.ext("apply"), &self.source) {
            return Err(MathError::InvalidArgument(
                "Element must be from source extension".to_string(),
            ));
        }

        if let Some(target) = &self.target {
            // Map each power of the generator
            let mut result = PadicExtensionElement::zero(target.clone());

            for (i, coeff) in element.coefficients().iter().enumerate() {
                // Compute generator^i in target
                let mut gen_power = PadicExtensionElement::one(target.clone());
                let gen_image = PadicExtensionElement::new(
                    target.clone(),
                    self.generator_image.clone(),
                );

                for _ in 0..i {
                    gen_power = gen_power * gen_image.clone();
                }

                // Multiply by coefficient and add to result
                let term = PadicExtensionElement::from_base_field(target.clone(), coeff.clone());
                result = result + (term * gen_power);
            }

            Ok(result)
        } else {
            // Embedding into algebraic closure - return the element itself
            // (in practice, this would need a proper algebraic closure type)
            Ok(element.clone())
        }
    }

    /// Compose two embeddings
    ///
    /// If we have K → L and L → M, compute K → M
    pub fn compose(&self, other: &PadicEmbedding) -> Result<Self> {
        if let (Some(self_target), other_source) = (&self.target, &other.source) {
            if !same_extension(self_target, other_source) {
                return Err(MathError::InvalidArgument(
                    "Cannot compose: target of first != source of second".to_string(),
                ));
            }

            // Compute image of source generator under composition
            let intermediate = PadicExtensionElement::new(
                self_target.clone(),
                self.generator_image.clone(),
            );

            let final_image = other.apply(&intermediate)?;

            Ok(PadicEmbedding {
                source: self.source.clone(),
                target: other.target.clone(),
                generator_image: final_image.coefficients().to_vec(),
            })
        } else {
            Err(MathError::InvalidArgument(
                "Cannot compose embeddings involving algebraic closure".to_string(),
            ))
        }
    }

    /// Check if this embedding is Galois (automorphism)
    pub fn is_galois_automorphism(&self) -> bool {
        if let Some(target) = &self.target {
            // Check if source and target are the same and extension is Galois
            same_extension(&self.source, target) && self.source.is_galois()
        } else {
            false
        }
    }

    /// Get the source extension
    pub fn source(&self) -> &Arc<PadicExtension> {
        &self.source
    }

    /// Get the target extension
    pub fn target(&self) -> Option<&Arc<PadicExtension>> {
        self.target.as_ref()
    }
}

impl fmt::Display for PadicEmbedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(target) = &self.target {
            write!(f, "Embedding from {} to {}", self.source, target)
        } else {
            write!(f, "Embedding from {} to Q̄_{}", self.source, self.source.prime())
        }
    }
}

/// Galois group of an unramified extension
///
/// For an unramified extension of degree f, the Galois group is cyclic of order f,
/// generated by the Frobenius automorphism x ↦ x^p.
#[derive(Clone, Debug)]
pub struct GaloisGroup {
    /// The extension
    extension: Arc<PadicExtension>,

    /// Galois automorphisms
    automorphisms: Vec<PadicEmbedding>,
}

impl GaloisGroup {
    /// Create the Galois group for an unramified extension
    ///
    /// # Arguments
    ///
    /// * `extension` - Must be an unramified extension
    ///
    /// # Returns
    ///
    /// Galois group or error if extension is not Galois
    /// For an unramified extension of degree f the group is cyclic of order
    /// f, generated by the **real** Frobenius: the automorphism sending the
    /// generator to the Hensel-lifted root of the defining polynomial
    /// congruent to `x^p` mod p (computed and certified by
    /// [`UnramifiedExtension`]). This replaced the old facade whose
    /// non-identity automorphisms were placeholder identities (wrong already
    /// at degree 2) and which panicked `unimplemented!` at degree > 2.
    ///
    /// Requires the defining polynomial to be monic and irreducible mod p
    /// (true for extensions built by [`PadicExtension::unramified`]); returns
    /// an honest error otherwise.
    pub fn new(extension: Arc<PadicExtension>) -> Result<Self> {
        if !extension.is_galois() {
            return Err(MathError::InvalidArgument(
                "Extension must be Galois".to_string(),
            ));
        }

        // is_galois() is only true for unramified extensions; build the real
        // Frobenius from the certified unramified machinery.
        let f = extension.residue_degree();
        let unram = UnramifiedExtension::with_modulus(
            extension.prime().clone(),
            extension.defining_polynomial(),
            extension.precision() as u32,
        )?;

        let to_padic = |coeffs: &[Integer]| -> Result<Vec<PadicRational>> {
            coeffs
                .iter()
                .map(|c| {
                    Ok(PadicRational::from_padic_integer(PadicInteger::from_integer(
                        c.clone(),
                        extension.prime().clone(),
                        extension.precision(),
                    )?))
                })
                .collect()
        };

        let mut automorphisms = Vec::with_capacity(f);

        // Identity automorphism: generator maps to itself
        let identity_image = {
            let mut coeffs = vec![
                PadicRational::from_padic_integer(
                    PadicInteger::zero(extension.prime().clone(), extension.precision()).unwrap()
                );
                extension.degree()
            ];
            if extension.degree() >= 2 {
                coeffs[1] = PadicRational::from_padic_integer(
                    PadicInteger::one(extension.prime().clone(), extension.precision()).unwrap()
                );
            }
            coeffs
        };
        automorphisms.push(PadicEmbedding::new(
            extension.clone(),
            extension.clone(),
            identity_image,
        )?);

        // sigma^k for k = 1..f-1: generator |-> frobenius^k(generator)
        let mut cur = unram.generator().frobenius();
        for _ in 1..f {
            automorphisms.push(PadicEmbedding::new(
                extension.clone(),
                extension.clone(),
                to_padic(cur.coefficients())?,
            )?);
            cur = cur.frobenius();
        }

        Ok(GaloisGroup {
            extension,
            automorphisms,
        })
    }

    /// Get the order of the Galois group
    pub fn order(&self) -> usize {
        self.automorphisms.len()
    }

    /// Get all automorphisms
    pub fn automorphisms(&self) -> &[PadicEmbedding] {
        &self.automorphisms
    }

    /// Get the Frobenius automorphism (generator of the group for unramified extensions)
    pub fn frobenius(&self) -> Option<&PadicEmbedding> {
        if self.extension.is_unramified() && self.automorphisms.len() >= 2 {
            Some(&self.automorphisms[1])
        } else {
            None
        }
    }
}

impl fmt::Display for GaloisGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Galois group of {} (order {})",
            self.extension,
            self.order()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_eisenstein_polynomial() {
        let p = Integer::from(5);

        // x^2 - 5 is Eisenstein at 5
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-5),  // constant: -5 (v_5(-5) = 1)
            Integer::from(0),   // x coefficient
            Integer::from(1),   // x^2 coefficient (not divisible by 5)
        ]);

        assert!(PadicExtension::is_eisenstein(&poly, &p));

        // x^2 - 10 is not Eisenstein (v_5(10) = 1 but we want exactly 1)
        let poly2 = UnivariatePolynomial::new(vec![
            Integer::from(-10),
            Integer::from(0),
            Integer::from(1),
        ]);

        // Actually this should pass since v_5(10) = 1
        assert!(PadicExtension::is_eisenstein(&poly2, &p));

        // x^2 - 25 is not Eisenstein (v_5(25) = 2 > 1)
        let poly3 = UnivariatePolynomial::new(vec![
            Integer::from(-25),
            Integer::from(0),
            Integer::from(1),
        ]);

        assert!(!PadicExtension::is_eisenstein(&poly3, &p));
    }

    #[test]
    fn test_eisenstein_extension_creation() {
        let p = Integer::from(5);
        let precision = 10;

        // Create Q_5(√5) via x^2 - 5
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-5),
            Integer::from(0),
            Integer::from(1),
        ]);

        let ext = PadicExtension::eisenstein(p, poly, precision).unwrap();

        assert_eq!(ext.degree(), 2);
        assert_eq!(ext.ramification_index(), 2);
        assert_eq!(ext.residue_degree(), 1);
        assert!(ext.is_totally_ramified());
        assert!(!ext.is_unramified());
    }

    #[test]
    fn test_unramified_extension_creation() {
        let p = Integer::from(5);
        let precision = 10;

        let ext = PadicExtension::unramified(p, 2, precision).unwrap();

        assert_eq!(ext.degree(), 2);
        assert_eq!(ext.ramification_index(), 1);
        assert_eq!(ext.residue_degree(), 2);
        assert!(ext.is_unramified());
        assert!(!ext.is_totally_ramified());
        assert!(ext.is_galois()); // Unramified extensions are always Galois
    }

    #[test]
    fn test_extension_element_arithmetic() {
        let p = Integer::from(5);
        let precision = 10;

        // Create Q_5(√5)
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-5),
            Integer::from(0),
            Integer::from(1),
        ]);

        let ext = Arc::new(PadicExtension::eisenstein(p.clone(), poly, precision).unwrap());

        // Create element 2 + 3√5
        let a = PadicExtensionElement::new(
            ext.clone(),
            vec![
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(2), p.clone(), precision).unwrap()
                ),
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(3), p.clone(), precision).unwrap()
                ),
            ],
        );

        // Create element 1 + √5
        let b = PadicExtensionElement::new(
            ext.clone(),
            vec![
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(1), p.clone(), precision).unwrap()
                ),
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(1), p.clone(), precision).unwrap()
                ),
            ],
        );

        // Test addition: (2 + 3√5) + (1 + √5) = 3 + 4√5
        let sum = a.clone() + b.clone();
        assert_eq!(sum.coefficients[0].unit().value(), &Integer::from(3));
        assert_eq!(sum.coefficients[1].unit().value(), &Integer::from(4));
    }

    #[test]
    fn test_extension_element_multiplication() {
        let p = Integer::from(5);
        let precision = 10;

        // Create Q_5(√5) with defining polynomial x^2 - 5
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-5),
            Integer::from(0),
            Integer::from(1),
        ]);

        let ext = Arc::new(PadicExtension::eisenstein(p.clone(), poly, precision).unwrap());

        // Create √5 (generator)
        let sqrt5 = ext.generator();

        // √5 * √5 should equal 5
        let product = sqrt5.clone() * sqrt5.clone();

        // Check that product ≈ 5 (might have some precision issues)
        // The result should be 5 + 0·√5
        assert_eq!(product.coefficients[0].unit().value(), &Integer::from(5));
        assert!(product.coefficients[1].unit().is_zero());
    }

    #[test]
    fn test_norm_and_trace() {
        let p = Integer::from(5);
        let precision = 10;

        // Create Q_5(√5)
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-5),
            Integer::from(0),
            Integer::from(1),
        ]);

        let ext = Arc::new(PadicExtension::eisenstein(p.clone(), poly, precision).unwrap());

        // For element a + b√5 in Q_5(√5):
        // Norm(a + b√5) = a² - 5b²
        // Trace(a + b√5) = 2a

        let elem = PadicExtensionElement::new(
            ext.clone(),
            vec![
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(3), p.clone(), precision).unwrap()
                ),
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(2), p.clone(), precision).unwrap()
                ),
            ],
        );

        // sympy-verified: N(3 + 2√5) = 9 - 20 = -11, Tr(3 + 2√5) = 6
        let norm = elem.norm().unwrap();
        let trace = elem.trace().unwrap();
        let expect_norm = PadicRational::from_rational(
            Rational::from(-11),
            p.clone(),
            precision,
        )
        .unwrap();
        let expect_trace = PadicRational::from_rational(
            Rational::from(6),
            p.clone(),
            precision,
        )
        .unwrap();
        assert_eq!(norm, expect_norm);
        assert_eq!(trace, expect_trace);
    }

    /// Degree-3 Eisenstein norm/trace through the old API — this used to be
    /// an `unimplemented!` facade. sympy-verified: over Q_2 with x^3 - 2,
    /// N(pi) = 2, Tr(pi) = 0, N(4 + 2pi + pi^2) = 36, Tr(4 + 2pi + pi^2) = 12.
    #[test]
    fn test_norm_and_trace_degree3_eisenstein() {
        let p = Integer::from(2);
        let precision = 16;
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-2),
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
        ]);
        let ext = Arc::new(PadicExtension::eisenstein(p.clone(), poly, precision).unwrap());

        let pi = ext.generator();
        assert_eq!(
            pi.norm().unwrap(),
            PadicRational::from_rational(Rational::from(2), p.clone(), precision).unwrap()
        );
        assert!(pi.trace().unwrap().unit().is_zero());

        let elem = PadicExtensionElement::new(
            ext.clone(),
            vec![
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(4), p.clone(), precision).unwrap(),
                ),
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(2), p.clone(), precision).unwrap(),
                ),
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(1), p.clone(), precision).unwrap(),
                ),
            ],
        );
        assert_eq!(
            elem.norm().unwrap(),
            PadicRational::from_rational(Rational::from(36), p.clone(), precision).unwrap()
        );
        assert_eq!(
            elem.trace().unwrap(),
            PadicRational::from_rational(Rational::from(12), p.clone(), precision).unwrap()
        );
    }

    /// Degree-3 unramified norm/trace through the old API (Conway(5,3) =
    /// x^3 + 3x + 3; sympy-verified N(g) = -3, Tr(g) = 0, N(2+3g+g^2) = 11).
    #[test]
    fn test_norm_and_trace_degree3_unramified() {
        let p = Integer::from(5);
        let precision = 12;
        let ext = Arc::new(PadicExtension::unramified(p.clone(), 3, precision).unwrap());
        // the defining polynomial must be the certified Conway lift
        assert_eq!(
            ext.defining_polynomial().coefficients(),
            &[
                Integer::from(3),
                Integer::from(3),
                Integer::from(0),
                Integer::from(1)
            ]
        );

        let g = ext.generator();
        assert_eq!(
            g.norm().unwrap(),
            PadicRational::from_rational(Rational::from(-3), p.clone(), precision).unwrap()
        );
        assert!(g.trace().unwrap().unit().is_zero());

        let elem = PadicExtensionElement::new(
            ext.clone(),
            vec![
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(2), p.clone(), precision).unwrap(),
                ),
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(3), p.clone(), precision).unwrap(),
                ),
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(Integer::from(1), p.clone(), precision).unwrap(),
                ),
            ],
        );
        assert_eq!(
            elem.norm().unwrap(),
            PadicRational::from_rational(Rational::from(11), p.clone(), precision).unwrap()
        );
        assert!(elem.trace().unwrap().unit().is_zero());
    }

    #[test]
    fn test_embedding_into_algebraic_closure() {
        let p = Integer::from(5);
        let precision = 10;

        // Create Q_5(√5)
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-5),
            Integer::from(0),
            Integer::from(1),
        ]);

        let ext = Arc::new(PadicExtension::eisenstein(p.clone(), poly, precision).unwrap());

        // Create embedding into algebraic closure
        let embedding = PadicEmbedding::into_algebraic_closure(ext.clone());

        // Create an element
        let elem = ext.generator();

        // Apply embedding
        let embedded = embedding.apply(&elem).unwrap();

        // Embedded element should be the same (identity embedding)
        assert_eq!(embedded.coefficients(), elem.coefficients());
    }

    #[test]
    fn test_galois_group_unramified() {
        let p = Integer::from(5);
        let precision = 10;

        // Create unramified extension of degree 2
        let ext = Arc::new(PadicExtension::unramified(p, 2, precision).unwrap());

        // Create Galois group
        let galois_group = GaloisGroup::new(ext.clone()).unwrap();

        // Galois group should have order 2 (cyclic group)
        assert_eq!(galois_group.order(), 2);

        // Should have Frobenius automorphism
        assert!(galois_group.frobenius().is_some());

        // The Frobenius must be REAL: it moves the generator (the old facade
        // used the identity here), and applying it twice is the identity.
        let g = ext.generator();
        let frob = galois_group.frobenius().unwrap();
        let sg = frob.apply(&g).unwrap();
        assert_ne!(sg, g, "Frobenius must move the generator");
        let ssg = frob.apply(&sg).unwrap();
        assert_eq!(ssg, g, "sigma^2 = id for f = 2");
    }

    /// Degree-3 Galois group — this used to be an `unimplemented!` facade.
    /// The three automorphisms are id, sigma, sigma^2 with sigma real
    /// Frobenius; the product of the conjugate orbit of g is N(g) = -3
    /// (sympy-verified for Conway(5,3) = x^3 + 3x + 3).
    #[test]
    fn test_galois_group_degree3_real_frobenius() {
        let p = Integer::from(5);
        let precision = 12;
        let ext = Arc::new(PadicExtension::unramified(p.clone(), 3, precision).unwrap());
        let galois_group = GaloisGroup::new(ext.clone()).unwrap();
        assert_eq!(galois_group.order(), 3);

        let g = ext.generator();
        let frob = galois_group.frobenius().unwrap();
        let sg = frob.apply(&g).unwrap();
        let ssg = frob.apply(&sg).unwrap();
        let sssg = frob.apply(&ssg).unwrap();
        assert_ne!(sg, g);
        assert_ne!(ssg, g);
        assert_ne!(ssg, sg);
        assert_eq!(sssg, g, "sigma^3 = id for f = 3");

        // N(g) = g * sigma(g) * sigma^2(g) must equal norm() = -3 and land
        // in the base field (non-constant coefficients zero)
        let orbit_prod = g.clone() * sg * ssg;
        assert!(orbit_prod.coefficients()[1].unit().is_zero());
        assert!(orbit_prod.coefficients()[2].unit().is_zero());
        let expect =
            PadicRational::from_rational(Rational::from(-3), p.clone(), precision).unwrap();
        assert_eq!(orbit_prod.coefficients()[0].unit(), expect.unit());
        assert_eq!(g.norm().unwrap(), expect);

        // the third automorphism is sigma^2: its generator image agrees with
        // applying sigma twice
        let sigma2 = &galois_group.automorphisms()[2];
        let via_embedding = sigma2.apply(&g).unwrap();
        let ssg2 = frob.apply(&frob.apply(&g).unwrap()).unwrap();
        assert_eq!(via_embedding, ssg2);
    }

    #[test]
    fn test_galois_group_eisenstein_fails() {
        let p = Integer::from(5);
        let precision = 10;

        // Create Eisenstein extension
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-5),
            Integer::from(0),
            Integer::from(1),
        ]);

        let ext = Arc::new(PadicExtension::eisenstein(p, poly, precision).unwrap());

        // Galois group creation should fail (Eisenstein ext may not be Galois)
        let result = GaloisGroup::new(ext);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_composition() {
        let p = Integer::from(5);
        let precision = 10;

        // Create two degree-2 unramified extensions
        let ext1 = Arc::new(PadicExtension::unramified(p.clone(), 2, precision).unwrap());
        let ext2 = Arc::new(PadicExtension::unramified(p, 2, precision).unwrap());

        // Create identity embeddings
        let mut coeffs1 = vec![
            PadicRational::from_padic_integer(
                PadicInteger::zero(ext1.prime().clone(), precision).unwrap()
            );
            ext1.degree()
        ];
        coeffs1[1] = PadicRational::from_padic_integer(
            PadicInteger::one(ext1.prime().clone(), precision).unwrap()
        );

        let emb1 = PadicEmbedding::new(ext1.clone(), ext1.clone(), coeffs1.clone()).unwrap();

        let mut coeffs2 = vec![
            PadicRational::from_padic_integer(
                PadicInteger::zero(ext2.prime().clone(), precision).unwrap()
            );
            ext2.degree()
        ];
        coeffs2[1] = PadicRational::from_padic_integer(
            PadicInteger::one(ext2.prime().clone(), precision).unwrap()
        );

        let emb2 = PadicEmbedding::new(ext2.clone(), ext2.clone(), coeffs2).unwrap();

        // Composing two identity embeddings should fail (different extensions)
        // unless they're the same extension
        // This test just ensures composition doesn't panic
        let _ = emb1.compose(&emb1);
    }

    #[test]
    fn test_ramification_index_and_residue_degree() {
        let p = Integer::from(7);
        let precision = 10;

        // Test Eisenstein extension
        let eisenstein_poly = UnivariatePolynomial::new(vec![
            Integer::from(-7),
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
        ]);

        let eisenstein_ext = PadicExtension::eisenstein(p.clone(), eisenstein_poly, precision).unwrap();

        // For Eisenstein extension of degree 3: e=3, f=1
        assert_eq!(eisenstein_ext.ramification_index(), 3);
        assert_eq!(eisenstein_ext.residue_degree(), 1);
        assert_eq!(eisenstein_ext.degree(), 3);
        assert!(eisenstein_ext.is_totally_ramified());

        // Test unramified extension
        let unram_ext = PadicExtension::unramified(p, 3, precision).unwrap();

        // For unramified extension of degree 3: e=1, f=3
        assert_eq!(unram_ext.ramification_index(), 1);
        assert_eq!(unram_ext.residue_degree(), 3);
        assert_eq!(unram_ext.degree(), 3);
        assert!(unram_ext.is_unramified());
    }

    #[test]
    fn test_extension_element_zero_and_one() {
        let p = Integer::from(5);
        let precision = 10;

        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-5),
            Integer::from(0),
            Integer::from(1),
        ]);

        let ext = Arc::new(PadicExtension::eisenstein(p, poly, precision).unwrap());

        let zero = PadicExtensionElement::zero(ext.clone());
        let one = PadicExtensionElement::one(ext.clone());

        assert!(zero.is_zero());
        assert!(!zero.is_one());

        assert!(!one.is_zero());
        assert!(one.is_one());

        // Test additive identity
        let elem = ext.generator();
        let sum = elem.clone() + zero.clone();
        assert_eq!(sum.coefficients(), elem.coefficients());

        // Test multiplicative identity
        let product = elem.clone() * one.clone();
        assert_eq!(product.coefficients(), elem.coefficients());
    }

    #[test]
    fn test_extension_display() {
        let p = Integer::from(5);
        let precision = 10;

        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-5),
            Integer::from(0),
            Integer::from(1),
        ]);

        let ext = PadicExtension::eisenstein(p, poly, precision).unwrap();

        let display = format!("{}", ext);
        assert!(display.contains("Eisenstein"));
        assert!(display.contains("Q_5"));
    }

    #[test]
    fn test_different_primes() {
        // Test extensions over different primes
        for prime_val in [2, 3, 5, 7, 11] {
            let p = Integer::from(prime_val);
            let precision = 8;

            // Create Eisenstein extension x^2 - p
            let poly = UnivariatePolynomial::new(vec![
                -p.clone(),
                Integer::zero(),
                Integer::one(),
            ]);

            let ext = PadicExtension::eisenstein(p.clone(), poly, precision).unwrap();

            assert_eq!(ext.degree(), 2);
            assert_eq!(ext.prime(), &p);
            assert!(ext.is_totally_ramified());
        }
    }

    #[test]
    fn test_generator_name() {
        let p = Integer::from(5);
        let precision = 10;

        let poly = UnivariatePolynomial::new(vec![
            Integer::from(-5),
            Integer::from(0),
            Integer::from(1),
        ]);

        let mut ext = PadicExtension::eisenstein(p, poly, precision).unwrap();

        assert_eq!(ext.generator_name(), "π");

        ext.set_generator_name("α".to_string());
        assert_eq!(ext.generator_name(), "α");
    }

    #[test]
    fn test_general_extension_type() {
        let p = Integer::from(5);
        let precision = 10;

        // Create a general extension with e=2, f=2, degree=4
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
        ]);

        let ext_type = ExtensionType::General {
            ramification_index: 2,
            residue_degree: 2,
        };

        let ext = PadicExtension::new(p, poly, precision, ext_type).unwrap();

        assert_eq!(ext.degree(), 4);
        assert_eq!(ext.ramification_index(), 2);
        assert_eq!(ext.residue_degree(), 2);
        assert!(!ext.is_unramified());
        assert!(!ext.is_totally_ramified());
    }

    #[test]
    fn test_invalid_general_extension() {
        let p = Integer::from(5);
        let precision = 10;

        // Try to create extension where e*f != degree
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
        ]); // degree 3

        let ext_type = ExtensionType::General {
            ramification_index: 2,
            residue_degree: 2, // 2*2 = 4 != 3
        };

        let result = PadicExtension::new(p, poly, precision, ext_type);
        assert!(result.is_err());
    }

    // A nonzero constant f would give the zero ring Q_5[x]/(3) = 0 (3 is a unit), for which
    // the parent rows (characteristic 0, nonzero) are false; `new` refuses it, as it refuses
    // the zero polynomial.
    #[test]
    fn test_constant_polynomial_rejected() {
        for cs in [vec![Integer::from(3)], vec![Integer::from(5)], vec![]] {
            let result = PadicExtension::new(
                Integer::from(5),
                UnivariatePolynomial::new(cs),
                10,
                ExtensionType::Unramified,
            );
            assert!(matches!(result, Err(MathError::InvalidArgument(_))));
        }
    }

    // ---- Unbound constants, value equality, field status, hooks, inverse (R1) ----

    fn q(n: i64, d: i64) -> Rational {
        Rational::new(Integer::from(n), Integer::from(d)).unwrap()
    }
    fn qp(n: i64, d: i64, p: i64, prec: usize) -> PadicRational {
        PadicRational::from_rational(q(n, d), Integer::from(p), prec).unwrap()
    }
    fn poly(cs: &[i64]) -> UnivariatePolynomial<Integer> {
        UnivariatePolynomial::new(cs.iter().map(|&c| Integer::from(c)).collect())
    }
    /// Q_5(√5) = eisenstein(5, x^2 - 5, 10).
    fn sqrt5() -> Arc<PadicExtension> {
        Arc::new(PadicExtension::eisenstein(Integer::from(5), poly(&[-5, 0, 1]), 10).unwrap())
    }
    /// `PadicExtension::new(p, f, prec, Unramified)` (the label is not trusted).
    fn ext_new(p: i64, cs: &[i64], prec: usize) -> Arc<PadicExtension> {
        Arc::new(
            PadicExtension::new(Integer::from(p), poly(cs), prec, ExtensionType::Unramified)
                .unwrap(),
        )
    }
    fn rone() -> PadicExtensionElement {
        <PadicExtensionElement as Ring>::one()
    }
    fn rzero() -> PadicExtensionElement {
        <PadicExtensionElement as Ring>::zero()
    }

    // At c79a582 `Ring::zero()`/`one()` panicked ("Cannot create PadicExtensionElement::zero()
    // without extension parameter").
    #[test]
    fn ring_constants_bind() {
        let ext = sqrt5();
        let g = ext.generator();
        let s = rzero() + g.clone();
        assert_eq!(s, g);
        assert_eq!(s.coefficients(), g.coefficients());
        for (a, b) in s.coefficients().iter().zip(g.coefficients()) {
            assert_eq!((a.valuation(), a.precision()), (b.valuation(), b.precision()));
        }
        assert_eq!(g.clone() * rone(), g);
        assert_eq!(rone() + g.clone(), PadicExtensionElement::one(ext.clone()) + g.clone());
        assert_eq!(rzero(), PadicExtensionElement::zero(ext.clone()));
        assert_eq!(PadicExtensionElement::zero(ext.clone()), rzero());
        let z = g.clone() * rzero();
        assert!(!z.is_bound() && z.is_zero());
        assert!(rone().is_one() && !rone().is_zero());
        assert_eq!(format!("{}", rone() + rone()), "2");
        // pow(0) is the bound 1 (at c79a582 it panicked in the static one()).
        let p0 = g.pow(0);
        assert!(p0.is_bound() && p0.is_one());
        // gp: Mod(x, x^2-5)^3 -> Mod(5*x, x^2 - 5).
        assert_eq!(
            g.pow(3),
            PadicExtensionElement::from_base_field(ext.clone(), qp(5, 1, 5, 10)) * g.clone()
        );
    }

    // `new` and `from_base_field` bind unbound coefficients at the extension's precision.
    #[test]
    fn new_binds_unbound_coefficients() {
        let ext = sqrt5();
        let one = <PadicRational as Ring>::one();
        let zero = <PadicRational as Ring>::zero();
        let x = PadicExtensionElement::new(ext.clone(), vec![one.clone(), zero]);
        for c in x.coefficients() {
            assert!(c.is_bound());
            assert_eq!(c.prime(), &Integer::from(5));
            assert_eq!(c.precision(), 10);
        }
        assert_eq!(x, PadicExtensionElement::one(ext.clone()));
        assert!(x.is_one());
        assert_eq!(
            PadicExtensionElement::from_base_field(ext.clone(), one),
            PadicExtensionElement::one(ext)
        );
    }

    // gp: x^2 - 5 defines √5, so √5·√5 = 5. `Mul` stores the product as (5, 0) (see
    // test_extension_element_multiplication), from_rational(5) is (1, 1); at c79a582 the
    // structural equality said they differ.
    #[test]
    fn product_of_generators_is_five() {
        let ext = sqrt5();
        let g = ext.generator();
        assert_eq!(
            g.clone() * g,
            PadicExtensionElement::from_base_field(ext, qp(5, 1, 5, 10))
        );
    }

    // (g·g)·(1/5) has constant coefficient (5, -1): one by value (HEAD: not one).
    #[test]
    fn one_after_dividing_by_five_is_one() {
        let ext = sqrt5();
        let g = ext.generator();
        let x = (g.clone() * g) * PadicExtensionElement::from_base_field(ext, qp(1, 5, 5, 10));
        assert_eq!(x.coefficients()[0].valuation(), -1);
        assert!(x.is_one());
        assert!(!x.is_zero());
    }

    #[test]
    #[should_panic(expected = "PadicExtensionElement::extension")]
    fn unbound_extension_element_refuses() {
        let _ = rone().extension();
    }

    #[test]
    fn unbound_extension_element_has_no_degree() {
        assert!(matches!(rone().norm(), Err(MathError::InvalidArgument(_))));
        assert!(matches!(rone().trace(), Err(MathError::InvalidArgument(_))));
        assert_eq!(rone().coefficients().len(), 1);
    }

    #[test]
    fn embedding_binds_unbound() {
        let ext = sqrt5();
        let emb = PadicEmbedding::into_algebraic_closure(ext.clone());
        let y = emb.apply(&rone()).unwrap();
        assert!(y.is_bound() && y.is_one());
        assert_eq!(y, PadicExtensionElement::one(ext));
    }

    // The three containers' hooks (F1): an unbound constant binds into a model's extension.
    #[test]
    fn unbound_constant_hooks() {
        const { assert!(<PadicExtensionElement as Ring>::HAS_UNBOUND_CONSTANTS) };
        let ext = sqrt5();
        let g = ext.generator();
        assert!(rone().is_unbound_constant() && !g.is_unbound_constant());
        let b = rone().bind_to_parent_of(&g).unwrap();
        assert!(b.is_bound() && b.is_one());
        assert_eq!(g.bind_to_parent_of(&g), None);
        assert_eq!(rone().bind_to_parent_of(&rone()), None);
    }

    // The modulus of unramified(5, 2) is the Conway polynomial x^2 + 4x + 2 (read back
    // here). The label of `new` is ignored: x^2 - 5 labelled Unramified is Eisenstein.
    #[test]
    fn field_certificates() {
        use FieldCertificate::*;
        assert_eq!(sqrt5().field_certificate(), Some(Eisenstein));
        let unr = PadicExtension::unramified(Integer::from(5), 2, 10).unwrap();
        assert_eq!(
            unr.defining_polynomial().coefficients(),
            &[Integer::from(2), Integer::from(4), Integer::from(1)]
        );
        assert_eq!(unr.field_certificate(), Some(IrreducibleReduction));
        assert_eq!(ext_new(5, &[-1, 5], 10).field_certificate(), Some(Linear));
        assert_eq!(ext_new(5, &[-5, 0, 1], 10).field_certificate(), Some(Eisenstein));
        for (p, cs) in [(5, &[-1, 0, 1][..]), (5, &[-125, 0, 1][..]), (6, &[-5, 0, 1][..])] {
            let e = ext_new(p, cs, 10);
            assert_eq!(e.field_certificate(), None, "p = {p}, f = {cs:?}");
            assert!(!e.is_certified_field());
        }
    }

    // gp: #factorpadic(x^2-125,5,10)[,1] -> 1; #factorpadic(x^2-1,5,10)[,1] -> 2;
    // #factorpadic(x^2-2,7,10)[,1] -> 2; #factorpadic(x^2-5,2,10)[,1] -> 1 (so Q_4(√5) =
    // Q_2(√5) is a field); issquarefree((x-1)^2) -> 0; 6 = 2·3.
    #[test]
    fn field_status_decides() {
        assert_eq!(sqrt5().field_status(), Some(true));
        assert_eq!(
            PadicExtension::unramified(Integer::from(5), 2, 10).unwrap().field_status(),
            Some(true)
        );
        assert_eq!(ext_new(5, &[-1, 5], 10).field_status(), Some(true));
        assert_eq!(ext_new(5, &[-125, 0, 1], 10).field_status(), Some(true));
        assert_eq!(ext_new(5, &[-1, 0, 1], 10).field_status(), Some(false));
        assert_eq!(ext_new(7, &[-2, 0, 1], 10).field_status(), Some(false));
        assert_eq!(ext_new(5, &[1, -2, 1], 10).field_status(), Some(false));
        assert_eq!(ext_new(6, &[-5, 0, 1], 10).field_status(), Some(false));
        assert_eq!(ext_new(4, &[-5, 0, 1], 10).field_status(), Some(true));
        for (p, cs, reduced) in [
            (5, &[1, -2, 1][..], false),
            (5, &[-1, 0, 1][..], true),
            (5, &[-125, 0, 1][..], true),
            (5, &[-1, 5][..], true),
        ] {
            assert_eq!(ext_new(p, cs, 10).is_reduced(), reduced, "f = {cs:?}");
        }
    }

    // The `field_status` cache is shared by clones (fix-up round 1): `generator()` clones
    // the extension, and the OM run made for the generator fills the cache of `ext` too.
    // gp: #factorpadic(x^2-125,5,10)[,1] -> 1.
    #[test]
    fn field_status_cache_is_shared_by_clones() {
        let ext = ext_new(5, &[-125, 0, 1], 10);
        assert!(ext.field_certificate().is_none());
        assert!(ext.field_status.get().is_none());
        let g = ext.generator();
        assert_eq!(g.unit_status(), Some(true));
        assert_eq!(ext.field_status.get(), Some(&Some(true)));
        let copy = (*ext).clone();
        assert!(Arc::ptr_eq(&copy.field_status, &ext.field_status));
    }

    // gp: factorpadic(x^2-1,5,5) has two factors, so 1 + g is a zero divisor
    // ((1+g)(1-g) = 1 - g^2 = 0); issquarefree(x^2-1) -> 1; (g-1)^2 = f(g) = 0 for
    // f = (x-1)^2.
    #[test]
    fn hooks_in_extensions() {
        use MultiplicativeOrder::*;
        let ext = sqrt5();
        let g = ext.generator();
        assert_eq!(g.unit_status(), Some(true));
        assert_eq!(g.nilpotent_status(), Some(false));
        assert_eq!(g.multiplicative_order_status(), Infinite);
        assert_eq!(
            g.unit_inverse().unwrap(),
            g.clone() * PadicExtensionElement::from_base_field(ext.clone(), qp(1, 5, 5, 10))
        );
        let one_g = PadicExtensionElement::one(ext.clone()) + g.clone();
        assert_eq!(one_g.multiplicative_order_status(), Undecided);
        assert_eq!(PadicExtensionElement::zero(ext.clone()).nilpotent_status(), Some(true));
        assert_eq!(PadicExtensionElement::zero(ext.clone()).multiplicative_order_status(), NotAUnit);
        assert_eq!((-PadicExtensionElement::one(ext.clone())).multiplicative_order_status(), Finite(2));

        let split = ext_new(5, &[-1, 0, 1], 10);
        let x = PadicExtensionElement::one(split.clone()) + split.generator();
        assert_eq!(x.unit_status(), None);
        assert_eq!(x.nilpotent_status(), Some(false));
        assert!(matches!(x.unit_inverse(), Err(MathError::NotImplemented(_))));

        let square = ext_new(5, &[1, -2, 1], 10);
        let y = square.generator() - PadicExtensionElement::one(square.clone());
        assert_eq!(y.nilpotent_status(), None);

        // ±1 are self-inverse in any K (the convention that `is_one` is exact), also where
        // `inverse` is undecided (no certificate); in a certified K the computed inverse.
        for k in [split.clone(), square.clone(), ext_new(6, &[-5, 0, 1], 10), ext.clone()] {
            let one = PadicExtensionElement::one(k.clone());
            for u in [one.clone(), -one] {
                assert_eq!(u.unit_status(), Some(true));
                assert_eq!(u.unit_inverse().unwrap(), u);
            }
        }

        let om = ext_new(5, &[-125, 0, 1], 10);
        assert_eq!(om.generator().unit_status(), Some(true));
        assert!(matches!(om.generator().unit_inverse(), Err(MathError::NotImplemented(_))));

        let two = rone() + rone();
        assert_eq!(two.unit_status(), Some(true));
        assert_eq!(two.nilpotent_status(), Some(false));
        assert_eq!(two.multiplicative_order_status(), Infinite);
        assert_eq!(rzero().nilpotent_status(), Some(true));
        assert_eq!(<PadicExtensionElement as Ring>::static_characteristic(), Some(0));
    }

    fn assert_coeff(c: &PadicRational, value: Rational, absolute: i64) {
        assert!(c.is_bound());
        assert_eq!(c.absolute_precision(), absolute, "absolute precision of {c}");
        let p = c.prime().clone();
        let expected = if value.is_zero() {
            PadicRational::from_rational_to_absolute_precision(&value, &p, absolute, 10).unwrap()
        } else {
            PadicRational::from_rational(value, p, 40).unwrap()
        };
        assert_eq!(c, &expected);
    }

    // References (gp): lift(Mod(1,x^2-5)/(1+x)) -> 1/4*x - 1/4; lift(Mod(1,x^2-5)/x) ->
    // 1/5*x; lift(Mod(1,x^2-5)/(5+5*x)) -> 1/20*x - 1/20; lift(Mod(1,x^2+4*x+2)/x) ->
    // -1/2*x - 2; lift(Mod(1,x^2+4*x+2)/(5*x)) -> -1/10*x - 2/5; lift(Mod(1,x^3-2)/(1+x+x^2))
    // -> x - 1. PARI's own p-adic polmod inverses have the same precisions:
    // Mod((1+O(5^10)) + (1+O(5^10))*x, x^2-5)^-1 has both coefficients + O(5^10);
    // Mod(O(5^10) + (1+O(5^10))*x, x^2-5)^-1 -> (5^-1 + O(5^9))*x + O(5^9);
    // Mod((5+O(5^11)) + (5+O(5^11))*x, x^2-5)^-1 has both + O(5^9);
    // Mod(O(5^10) + (1+O(5^10))*x, x^2+4*x+2)^-1 has both + O(5^10);
    // Mod((1+O(2^8)) + (1+O(2^8))*x + (1+O(2^8))*x^2, x^3-2)^-1 has + O(2^8).
    // At c79a582 there was no inverse.
    #[test]
    fn inverse_in_certified_fields() {
        let ext = sqrt5();
        let g = ext.generator();
        let one = PadicExtensionElement::one(ext.clone());

        let x = one.clone() + g.clone();
        let y = x.inverse().unwrap();
        assert_coeff(&y.coefficients()[0], q(-1, 4), 10);
        assert_coeff(&y.coefficients()[1], q(1, 4), 10);
        assert_eq!(x * y, one);

        let y = g.inverse().unwrap();
        assert_coeff(&y.coefficients()[0], q(0, 1), 9);
        assert!(y.coefficients()[0].is_zero());
        assert_coeff(&y.coefficients()[1], q(1, 5), 9);
        assert_eq!(g.clone() * y, one);

        let x = PadicExtensionElement::new(ext.clone(), vec![qp(5, 1, 5, 10), qp(5, 1, 5, 10)]);
        let y = x.inverse().unwrap();
        assert_coeff(&y.coefficients()[0], q(-1, 20), 9);
        assert_coeff(&y.coefficients()[1], q(1, 20), 9);
        assert_eq!(x * y, one);

        let unr = Arc::new(PadicExtension::unramified(Integer::from(5), 2, 10).unwrap());
        let ug = unr.generator();
        let y = ug.inverse().unwrap();
        assert_coeff(&y.coefficients()[0], q(-2, 1), 10);
        assert_coeff(&y.coefficients()[1], q(-1, 2), 10);
        assert_eq!(ug.clone() * y, PadicExtensionElement::one(unr.clone()));
        let x = PadicExtensionElement::from_base_field(unr.clone(), qp(5, 1, 5, 10)) * ug;
        let y = x.inverse().unwrap();
        assert_coeff(&y.coefficients()[0], q(-2, 5), 8);
        assert_coeff(&y.coefficients()[1], q(-1, 10), 8);

        let e3 = Arc::new(
            PadicExtension::eisenstein(Integer::from(2), poly(&[-2, 0, 0, 1]), 8).unwrap(),
        );
        let pi = e3.generator();
        let one3 = PadicExtensionElement::one(e3.clone());
        let x = one3.clone() + pi.clone() + pi.clone() * pi;
        let y = x.inverse().unwrap();
        assert_eq!(y.coefficients()[0], qp(-1, 1, 2, 8));
        assert_eq!(y.coefficients()[1], qp(1, 1, 2, 8));
        assert!(y.coefficients()[2].is_zero());
        for c in y.coefficients() {
            assert_eq!(c.absolute_precision(), 8);
        }
        assert_eq!(x * y, one3);

        let lin = ext_new(5, &[-1, 5], 10);
        let x = PadicExtensionElement::from_base_field(lin.clone(), qp(25, 1, 5, 10));
        assert_eq!(
            x.inverse().unwrap(),
            PadicExtensionElement::from_base_field(lin, qp(1, 25, 5, 10))
        );
    }

    // Coefficient 0 known only to O(5^0): V = 1 >= E = 0, undecided (PARI also refuses:
    // "impossible inverse in divpp: O(5^0)"). With coefficient 1 known to O(5^0) instead:
    // V = 0, E = 1, B = 1, so x^-1 = (1 + O(5)) + O(5^0)·g.
    #[test]
    fn inverse_errors_and_low_precision() {
        let ext = sqrt5();
        let p = Integer::from(5);
        let weak = PadicRational::from_rational_to_absolute_precision(&q(0, 1), &p, 0, 10).unwrap();
        assert_eq!(weak.absolute_precision(), 0);
        let x = PadicExtensionElement::new(ext.clone(), vec![weak.clone(), qp(1, 1, 5, 10)]);
        assert!(matches!(x.inverse(), Err(MathError::NumericalError(_))));
        let x = PadicExtensionElement::new(ext.clone(), vec![qp(1, 1, 5, 10), weak]);
        let y = x.inverse().unwrap();
        assert_coeff(&y.coefficients()[0], q(1, 1), 1);
        assert_eq!(y.coefficients()[1].absolute_precision(), 0);
        assert!(y.coefficients()[1].is_zero());

        assert_eq!(PadicExtensionElement::zero(ext.clone()).inverse(), Err(MathError::NotInvertible));
        let u = rone().inverse().unwrap();
        assert!(!u.is_bound() && u.is_one());
        assert_eq!(rzero().inverse(), Err(MathError::NotInvertible));
        assert!(matches!((rone() + rone()).inverse(), Err(MathError::InvalidArgument(_))));
        for cs in [&[-1, 0, 1][..], &[-125, 0, 1][..]] {
            let e = ext_new(5, cs, 10);
            assert!(matches!(e.generator().inverse(), Err(MathError::NotImplemented(_))));
        }
    }
}
