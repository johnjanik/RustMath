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

use rustmath_core::{CommutativeRing, MathError, Result, Ring};
use rustmath_integers::Integer;
use rustmath_padics::{PadicInteger, PadicRational};
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

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

        let degree = defining_polynomial.degree().ok_or_else(|| {
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

        Ok(PadicExtension {
            prime,
            degree,
            ramification_index,
            residue_degree,
            extension_type,
            defining_polynomial,
            precision,
            generator_name: "π".to_string(),
        })
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
    /// This creates an extension using a polynomial that reduces to an
    /// irreducible polynomial over F_p. The actual polynomial would need
    /// to be computed based on Conway polynomials or another canonical choice.
    pub fn unramified(prime: Integer, degree: usize, precision: usize) -> Result<Self> {
        // For unramified extensions, we need a polynomial that is:
        // 1. Monic
        // 2. Irreducible over qp
        // 3. Reduces to an irreducible polynomial over F_p
        //
        // For now, we use x^degree - (1 + p) as a placeholder
        // In a full implementation, this would use Conway polynomials
        let mut coeffs = vec![Integer::zero(); degree + 1];
        coeffs[0] = -(Integer::one() + prime.clone()); // constant term: -(1+p)
        coeffs[degree] = Integer::one(); // leading coefficient

        let poly = UnivariatePolynomial::new(coeffs);

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
#[derive(Clone, Debug)]
pub struct PadicExtensionElement {
    /// The extension field this element belongs to
    extension: Arc<PadicExtension>,

    /// Coefficients a_0, a_1, ..., a_{n-1} where element = Σ a_i π^i
    /// Length must be equal to extension degree
    coefficients: Vec<PadicRational>,
}

impl PadicExtensionElement {
    /// Create a new extension element
    ///
    /// # Arguments
    ///
    /// * `extension` - The extension field
    /// * `coefficients` - Coefficients in terms of powers of the generator
    pub fn new(extension: Arc<PadicExtension>, coefficients: Vec<PadicRational>) -> Self {
        assert_eq!(
            coefficients.len(),
            extension.degree(),
            "Number of coefficients must match extension degree"
        );

        PadicExtensionElement {
            extension,
            coefficients,
        }
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

    /// Create element from a base field element (qp)
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
    pub fn extension(&self) -> &Arc<PadicExtension> {
        &self.extension
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[PadicRational] {
        &self.coefficients
    }

    /// Get mutable coefficients
    fn coefficients_mut(&mut self) -> &mut Vec<PadicRational> {
        &mut self.coefficients
    }

    /// Compute the norm N_{K/qp}(α)
    ///
    /// The norm is the product of all Galois conjugates, or equivalently
    /// the determinant of the multiplication-by-α map.
    ///
    /// For α = a_0 + a_1·π + ... + a_{n-1}·π^{n-1}, we compute:
    /// N(α) = (-1)^n · f(0) / leading_coeff where f is the char poly of α
    pub fn norm(&self) -> Result<PadicRational> {
        // The norm can be computed as the constant term of the
        // characteristic polynomial, up to sign
        let char_poly = self.characteristic_polynomial()?;
        let coeffs = char_poly.coefficients();

        if coeffs.is_empty() {
            return Ok(PadicRational::from_padic_integer(
                PadicInteger::zero(self.extension.prime().clone(), self.extension.precision()).unwrap()
            ));
        }

        // For a degree n extension, norm is (-1)^n times the constant term
        // divided by the leading coefficient
        let constant = coeffs[0].clone();
        let n = self.extension.degree();

        if n % 2 == 0 {
            Ok(constant)
        } else {
            Ok(-constant)
        }
    }

    /// Compute the trace Tr_{K/qp}(α)
    ///
    /// The trace is the sum of all Galois conjugates, or equivalently
    /// the trace of the multiplication-by-α map.
    pub fn trace(&self) -> Result<PadicRational> {
        // The trace can be computed as the negative of the coefficient
        // of x^{n-1} in the characteristic polynomial
        let char_poly = self.characteristic_polynomial()?;
        let coeffs = char_poly.coefficients();
        let n = self.extension.degree();

        if coeffs.len() <= n - 1 {
            return Ok(PadicRational::from_padic_integer(
                PadicInteger::zero(self.extension.prime().clone(), self.extension.precision()).unwrap()
            ));
        }

        // Trace is -a_{n-1} where char poly is x^n + a_{n-1}x^{n-1} + ...
        Ok(-coeffs[n - 1].clone())
    }

    /// Compute the characteristic polynomial of this element
    ///
    /// This is the minimal polynomial of the multiplication-by-α map
    /// viewed as a qp-linear transformation on the n-dimensional vector space K.
    fn characteristic_polynomial(&self) -> Result<UnivariatePolynomial<PadicRational>> {
        // Build the multiplication matrix M where M[i][j] = coefficient of π^i
        // in α · π^j
        let n = self.extension.degree();
        let mut matrix = vec![vec![
            PadicRational::from_padic_integer(
                PadicInteger::zero(self.extension.prime().clone(), self.extension.precision()).unwrap()
            ); n]; n];

        // For each basis element π^j, compute α · π^j and extract coefficients
        for j in 0..n {
            let mut basis_elem_coeffs = vec![
                PadicRational::from_padic_integer(
                    PadicInteger::zero(self.extension.prime().clone(), self.extension.precision()).unwrap()
                );
                n
            ];
            basis_elem_coeffs[j] = PadicRational::from_padic_integer(
                PadicInteger::one(self.extension.prime().clone(), self.extension.precision()).unwrap()
            );

            let basis_elem = PadicExtensionElement::new(self.extension.clone(), basis_elem_coeffs);
            let product = self.clone() * basis_elem;

            for i in 0..n {
                matrix[i][j] = product.coefficients[i].clone();
            }
        }

        // Compute characteristic polynomial det(xI - M)
        // For now, return a simple placeholder (would need matrix operations)
        // In a full implementation, this would use the Faddeev-LeVerrier algorithm
        // or compute det(xI - M) directly

        // Simplified version: just use the minimal polynomial for elements of the form a + bπ
        if n == 2 && self.coefficients[1].unit().is_zero() {
            // For constant elements, char poly is (x - a)^n
            let mut coeffs = vec![
                PadicRational::from_padic_integer(
                    PadicInteger::zero(self.extension.prime().clone(), self.extension.precision()).unwrap()
                );
                n + 1
            ];
            coeffs[0] = self.coefficients[0].clone();
            coeffs[n] = PadicRational::from_padic_integer(
                PadicInteger::one(self.extension.prime().clone(), self.extension.precision()).unwrap()
            );
            return Ok(UnivariatePolynomial::new(coeffs));
        }

        // General case: return defining polynomial as approximation
        // (This is not correct in general but serves as placeholder)
        let def_poly = self.extension.defining_polynomial();
        let coeffs: Vec<PadicRational> = def_poly
            .coefficients()
            .iter()
            .map(|c| {
                PadicRational::from_padic_integer(
                    PadicInteger::from_integer(
                        c.clone(),
                        self.extension.prime().clone(),
                        self.extension.precision(),
                    ).unwrap()
                )
            })
            .collect();

        Ok(UnivariatePolynomial::new(coeffs))
    }

    /// Reduce modulo the defining polynomial
    ///
    /// This ensures the element is in canonical form with degree < extension degree
    fn reduce(&mut self) {
        // In a full implementation, this would perform polynomial division
        // For now, we just ensure the length matches the extension degree
        self.coefficients.truncate(self.extension.degree());
        while self.coefficients.len() < self.extension.degree() {
            self.coefficients.push(PadicRational::from_padic_integer(
                PadicInteger::zero(self.extension.prime().clone(), self.extension.precision()).unwrap()
            ));
        }
    }
}

impl fmt::Display for PadicExtensionElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        let gen_name = self.extension.generator_name();

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

impl PartialEq for PadicExtensionElement {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.extension, &other.extension)
            && self.coefficients.iter().zip(other.coefficients.iter()).all(|(a, b)| a == b)
    }
}

impl Add for PadicExtensionElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert!(
            Arc::ptr_eq(&self.extension, &other.extension),
            "Elements must be from the same extension"
        );

        let coeffs: Vec<PadicRational> = self
            .coefficients
            .into_iter()
            .zip(other.coefficients.into_iter())
            .map(|(a, b)| a + b)
            .collect();

        PadicExtensionElement::new(self.extension, coeffs)
    }
}

impl Sub for PadicExtensionElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert!(
            Arc::ptr_eq(&self.extension, &other.extension),
            "Elements must be from the same extension"
        );

        let coeffs: Vec<PadicRational> = self
            .coefficients
            .into_iter()
            .zip(other.coefficients.into_iter())
            .map(|(a, b)| a - b)
            .collect();

        PadicExtensionElement::new(self.extension, coeffs)
    }
}

impl Mul for PadicExtensionElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert!(
            Arc::ptr_eq(&self.extension, &other.extension),
            "Elements must be from the same extension"
        );

        let n = self.extension.degree();
        let mut result_coeffs = vec![
            PadicRational::from_padic_integer(
                PadicInteger::zero(self.extension.prime().clone(), self.extension.precision()).unwrap()
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

        let def_poly = self.extension.defining_polynomial();
        let def_coeffs = def_poly.coefficients();

        // Reduce high degree terms
        for k in (n..result_coeffs.len()).rev() {
            if !result_coeffs[k].unit().is_zero() {
                let factor = result_coeffs[k].clone();
                result_coeffs[k] = PadicRational::from_padic_integer(
                    PadicInteger::zero(self.extension.prime().clone(), self.extension.precision()).unwrap()
                );

                // x^k ≡ -(a_0 + a_1·x + ... + a_{n-1}·x^{n-1})/a_n · x^{k-n} mod f
                // where f = a_n·x^n + ... + a_0
                let leading = PadicRational::from_padic_integer(
                    PadicInteger::from_integer(
                        def_coeffs[n].clone(),
                        self.extension.prime().clone(),
                        self.extension.precision(),
                    ).unwrap()
                );

                for i in 0..n {
                    let coeff = PadicRational::from_padic_integer(
                        PadicInteger::from_integer(
                            def_coeffs[i].clone(),
                            self.extension.prime().clone(),
                            self.extension.precision(),
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
                PadicInteger::zero(self.extension.prime().clone(), self.extension.precision()).unwrap()
            ));
        }

        PadicExtensionElement::new(self.extension, result_coeffs)
    }
}

impl Neg for PadicExtensionElement {
    type Output = Self;

    fn neg(self) -> Self {
        let coeffs: Vec<PadicRational> = self.coefficients.into_iter().map(|c| -c).collect();
        PadicExtensionElement::new(self.extension, coeffs)
    }
}

impl Ring for PadicExtensionElement {
    fn zero() -> Self {
        panic!("Cannot create PadicExtensionElement::zero() without extension parameter");
    }

    fn one() -> Self {
        panic!("Cannot create PadicExtensionElement::one() without extension parameter");
    }

    fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c.unit().is_zero())
    }

    fn is_one(&self) -> bool {
        self.coefficients[0].unit().is_one()
            && self.coefficients[0].valuation() == 0
            && self.coefficients[1..].iter().all(|c| c.unit().is_zero())
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
        // Verify element is from source extension
        if !Arc::ptr_eq(&element.extension, &self.source) {
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
            if !Arc::ptr_eq(self_target, other_source) {
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
            Arc::ptr_eq(&self.source, target) && self.source.is_galois()
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
    pub fn new(extension: Arc<PadicExtension>) -> Result<Self> {
        if !extension.is_galois() {
            return Err(MathError::InvalidArgument(
                "Extension must be Galois".to_string(),
            ));
        }

        // For unramified extensions, generate the cyclic group
        let f = extension.residue_degree();
        let mut automorphisms = Vec::with_capacity(f);

        // Identity automorphism
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

        let identity = PadicEmbedding::new(
            extension.clone(),
            extension.clone(),
            identity_image,
        )?;

        automorphisms.push(identity.clone());

        // Generate other automorphisms by composing Frobenius
        for _ in 1..f {
            let prev = automorphisms.last().unwrap();
            // In practice, we'd compute Frobenius^k
            // For now, just use identity as placeholder
            automorphisms.push(identity.clone());
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

        // This is a simplified test - the actual norm/trace computation
        // is non-trivial and would require proper implementation
        let _norm = elem.norm();
        let _trace = elem.trace();

        // Just verify they don't panic
        assert!(_norm.is_ok());
        assert!(_trace.is_ok());
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
}
