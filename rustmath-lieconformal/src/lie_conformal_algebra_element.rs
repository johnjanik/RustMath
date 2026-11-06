//! Lie Conformal Algebra Elements
//!
//! Provides the element structure for Lie conformal algebras.
//!
//! Elements are represented as linear combinations of basis elements with
//! coefficients in R[∂] (polynomials in the derivation operator).
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.lie_conformal_algebra_element

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::Hash;
use crate::lie_conformal_algebra::GeneratorIndex;

/// Element of a Lie conformal algebra
///
/// Represented as a linear combination of basis elements, where each
/// coefficient is a polynomial in ∂.
///
/// # Type Parameters
///
/// * `R` - The base ring
/// * `B` - The basis index type
///
/// # Mathematical Representation
///
/// An element is written as:
/// Σᵢ pᵢ(∂) ⊗ bᵢ
///
/// where pᵢ(∂) ∈ R[∂] and bᵢ are basis elements.
#[derive(Clone, Debug)]
pub struct LieConformalAlgebraElement<R: Ring, B: Clone + Eq + Hash> {
    /// Terms: map from basis element to polynomial in ∂
    /// The polynomial is represented as a Vec<R> where index i holds the coefficient of ∂^i
    terms: HashMap<B, Vec<R>>,
}

impl<R: Ring + Clone, B: Clone + Eq + Hash> LieConformalAlgebraElement<R, B> {
    /// Create a new element
    pub fn new(terms: HashMap<B, Vec<R>>) -> Self {
        LieConformalAlgebraElement { terms }
    }

    /// Create the zero element
    pub fn zero() -> Self {
        LieConformalAlgebraElement {
            terms: HashMap::new(),
        }
    }

    /// Create an element from a single basis element
    pub fn from_basis(basis: B) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(basis, vec![R::from(1)]);
        LieConformalAlgebraElement { terms }
    }

    /// Create an element with a polynomial coefficient
    pub fn with_polynomial(basis: B, poly: Vec<R>) -> Self {
        let mut terms = HashMap::new();
        terms.insert(basis, poly);
        LieConformalAlgebraElement { terms }
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<B, Vec<R>> {
        &self.terms
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.terms.is_empty() || self.terms.values().all(|poly| poly.iter().all(|c| c.is_zero()))
    }

    /// Get the degree (maximum ∂ power)
    pub fn degree(&self) -> usize {
        self.terms
            .values()
            .map(|poly| poly.len().saturating_sub(1))
            .max()
            .unwrap_or(0)
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + PartialEq,
    {
        let mut result = self.terms.clone();

        for (basis, poly) in &other.terms {
            let new_poly = if let Some(existing) = result.get(basis) {
                add_polynomials(existing, poly)
            } else {
                poly.clone()
            };

            if !is_zero_polynomial(&new_poly) {
                result.insert(basis.clone(), new_poly);
            } else {
                result.remove(basis);
            }
        }

        LieConformalAlgebraElement { terms: result }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: std::ops::Mul<Output = R> + PartialEq,
    {
        if scalar.is_zero() {
            return Self::zero();
        }

        let terms = self
            .terms
            .iter()
            .map(|(basis, poly)| {
                let new_poly: Vec<R> = poly.iter().map(|c| c.clone() * scalar.clone()).collect();
                (basis.clone(), new_poly)
            })
            .collect();

        LieConformalAlgebraElement { terms }
    }

    /// Apply the derivation operator ∂
    ///
    /// The element `Σ_b p_b(∂) ⊗ b` stands for `Σ_b p_b(∂)·b` in the free
    /// R[∂]-module, so `∂` acts by multiplying each coefficient polynomial by
    /// `∂` (shifting its coefficients up by one). It is not the formal
    /// derivative of `p_b`.
    ///
    /// This type does not know which generators the parent algebra declares
    /// `∂`-torsion (e.g. a central `C` with `∂C = 0`); reducing such terms is
    /// the parent's job.
    pub fn apply_derivation(&self) -> Self
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + From<i64>,
    {
        self.shift_derivation(1)
    }

    /// Multiply every coefficient polynomial by `∂^n`.
    fn shift_derivation(&self, n: usize) -> Self {
        let terms = self
            .terms
            .iter()
            .filter(|(_, poly)| !is_zero_polynomial(poly))
            .map(|(basis, poly)| {
                let mut new_poly = vec![R::zero(); n];
                new_poly.extend(poly.iter().cloned());
                (basis.clone(), new_poly)
            })
            .collect();

        LieConformalAlgebraElement { terms }
    }

    /// Negate the element
    pub fn negate(&self) -> Self
    where
        R: std::ops::Neg<Output = R>,
    {
        let terms = self
            .terms
            .iter()
            .map(|(basis, poly)| {
                let new_poly: Vec<R> = poly.iter().map(|c| -c.clone()).collect();
                (basis.clone(), new_poly)
            })
            .collect();

        LieConformalAlgebraElement { terms }
    }

    /// Check if this element is a monomial (single basis term)
    pub fn is_monomial(&self) -> bool
    where
        R: PartialEq,
    {
        // Count non-zero terms
        let non_zero_count = self.terms
            .iter()
            .filter(|(_, poly)| !is_zero_polynomial(poly))
            .count();
        non_zero_count == 1
    }

    /// Get monomial coefficients as a HashMap
    ///
    /// Returns a map from basis elements to their polynomial coefficients.
    /// This matches SageMath's _monomial_coefficients attribute.
    pub fn monomial_coefficients(&self) -> HashMap<B, Vec<R>>
    where
        R: PartialEq,
    {
        self.terms
            .iter()
            .filter(|(_, poly)| !is_zero_polynomial(poly))
            .map(|(b, p)| (b.clone(), p.clone()))
            .collect()
    }
}

/// Element of a Lie conformal algebra with generators
///
/// This element type is used for algebras with a designated set of generators.
/// It extends the base element functionality with derivative operations in the
/// "divided powers" notation where T^(j) = T^j / j!.
///
/// Corresponds to sage.algebras.lie_conformal_algebras.lie_conformal_algebra_element.LCAWithGeneratorsElement
#[derive(Clone, Debug)]
pub struct LCAWithGeneratorsElement<R: Ring, B: Clone + Eq + Hash> {
    /// The underlying element
    element: LieConformalAlgebraElement<R, B>,
}

impl<R: Ring + Clone, B: Clone + Eq + Hash> LCAWithGeneratorsElement<R, B> {
    /// Create a new element with generators
    pub fn new(element: LieConformalAlgebraElement<R, B>) -> Self {
        LCAWithGeneratorsElement { element }
    }

    /// Create from terms directly
    pub fn from_terms(terms: HashMap<B, Vec<R>>) -> Self {
        LCAWithGeneratorsElement {
            element: LieConformalAlgebraElement::new(terms),
        }
    }

    /// Create the zero element
    pub fn zero() -> Self {
        LCAWithGeneratorsElement {
            element: LieConformalAlgebraElement::zero(),
        }
    }

    /// Create from a single basis element
    pub fn from_basis(basis: B) -> Self
    where
        R: From<i64>,
    {
        LCAWithGeneratorsElement {
            element: LieConformalAlgebraElement::from_basis(basis),
        }
    }

    /// Get the underlying element
    pub fn element(&self) -> &LieConformalAlgebraElement<R, B> {
        &self.element
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<B, Vec<R>> {
        self.element.terms()
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.element.is_zero()
    }

    /// Check if this element is a monomial
    pub fn is_monomial(&self) -> bool
    where
        R: PartialEq,
    {
        self.element.is_monomial()
    }

    /// Get monomial coefficients
    pub fn monomial_coefficients(&self) -> HashMap<B, Vec<R>>
    where
        R: PartialEq,
    {
        self.element.monomial_coefficients()
    }

    /// Apply the n-th power of the derivation `T = ∂`
    ///
    /// Returns `∂^n · self`, matching SageMath's `LCAWithGeneratorsElement.T(n)`
    /// (e.g. `L.T(3) = 6*T^(3)L` there, since Sage prints in the divided-power
    /// basis `T^(j) = T^j / j!`). Coefficients here are stored in ordinary
    /// powers of `∂`, so this shifts each coefficient polynomial up by `n`.
    ///
    /// # Arguments
    ///
    /// * `n` - The power of the derivative to apply
    pub fn T(&self, n: usize) -> Self
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + From<i64> + PartialEq,
    {
        if n == 0 {
            return self.clone();
        }

        if self.is_zero() {
            return Self::zero();
        }

        LCAWithGeneratorsElement {
            element: self.element.shift_derivation(n),
        }
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + PartialEq,
    {
        LCAWithGeneratorsElement {
            element: self.element.add(&other.element),
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: std::ops::Mul<Output = R> + PartialEq,
    {
        LCAWithGeneratorsElement {
            element: self.element.scalar_mul(scalar),
        }
    }

    /// Negate the element
    pub fn negate(&self) -> Self
    where
        R: std::ops::Neg<Output = R>,
    {
        LCAWithGeneratorsElement {
            element: self.element.negate(),
        }
    }
}

impl<R: Ring + Clone + Display, B: Clone + Eq + Hash + Display> Display for LCAWithGeneratorsElement<R, B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.element)
    }
}

/// Element of a Lie conformal algebra with structure coefficients
///
/// This specialized element type is used for algebras defined by explicit
/// structure coefficients. It provides the lambda bracket operation.
///
/// Corresponds to sage.algebras.lie_conformal_algebras.lie_conformal_algebra_element.LCAStructureCoefficientsElement
#[derive(Clone, Debug)]
pub struct LCAStructureCoefficientsElement<R: Ring, B: Clone + Eq + Hash> {
    /// The underlying element with generators
    base: LCAWithGeneratorsElement<R, B>,
}

impl<R: Ring + Clone, B: Clone + Eq + Hash> LCAStructureCoefficientsElement<R, B> {
    /// Create a new element with structure coefficients
    pub fn new(base: LCAWithGeneratorsElement<R, B>) -> Self {
        LCAStructureCoefficientsElement { base }
    }

    /// Create from terms directly
    pub fn from_terms(terms: HashMap<B, Vec<R>>) -> Self {
        LCAStructureCoefficientsElement {
            base: LCAWithGeneratorsElement::from_terms(terms),
        }
    }

    /// Create the zero element
    pub fn zero() -> Self {
        LCAStructureCoefficientsElement {
            base: LCAWithGeneratorsElement::zero(),
        }
    }

    /// Create from a single basis element
    pub fn from_basis(basis: B) -> Self
    where
        R: From<i64>,
    {
        LCAStructureCoefficientsElement {
            base: LCAWithGeneratorsElement::from_basis(basis),
        }
    }

    /// Get the underlying base element
    pub fn base(&self) -> &LCAWithGeneratorsElement<R, B> {
        &self.base
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<B, Vec<R>> {
        self.base.terms()
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.base.is_zero()
    }

    /// Check if this element is a monomial
    pub fn is_monomial(&self) -> bool
    where
        R: PartialEq,
    {
        self.base.is_monomial()
    }

    /// Get monomial coefficients
    pub fn monomial_coefficients(&self) -> HashMap<B, Vec<R>>
    where
        R: PartialEq,
    {
        self.base.monomial_coefficients()
    }

    /// Apply `∂^n` (see [`LCAWithGeneratorsElement::T`])
    pub fn T(&self, n: usize) -> Self
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + From<i64> + PartialEq,
    {
        LCAStructureCoefficientsElement {
            base: self.base.T(n),
        }
    }

    /// Lambda bracket with another element
    ///
    /// Computes the λ-bracket [self_λ right], returning a dictionary mapping
    /// nonnegative integer powers of λ to resulting elements.
    ///
    /// # Arguments
    ///
    /// * `right` - The right operand
    /// * `structure_coeffs` - The structure coefficients from the parent algebra
    ///   mapping (i, j, k) to the coefficient of λ^k in [g_i, g_j]
    ///
    /// # Returns
    ///
    /// A HashMap mapping λ-powers to elements, representing:
    /// [self_λ right] = Σ_k λ^k ⊗ element_k
    ///
    /// # Mathematical Details
    ///
    /// For basis elements g_i and g_j with structure coefficients s_{ij}^k(λ),
    /// we compute [g_i_λ g_j] using the stored coefficients and extend linearly.
    ///
    /// # Panics
    ///
    /// Always: not implemented. The `(B, B, usize) -> R` table cannot hold
    /// brackets with `∂` in their values (e.g. `[L_λ L] = (∂ + 2λ)L + ...`),
    /// and the sesquilinear and skew-symmetric extension is missing. It used
    /// to return an empty map, which claims `[self_λ right] = 0`.
    pub fn lambda_bracket(
        &self,
        _right: &Self,
        _structure_coeffs: &HashMap<(B, B, usize), R>,
    ) -> HashMap<usize, Self>
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + From<i64> + PartialEq,
    {
        unimplemented!(
            "LCAStructureCoefficientsElement::lambda_bracket: λ-bracket from structure \
             coefficients is not implemented"
        )
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + PartialEq,
    {
        LCAStructureCoefficientsElement {
            base: self.base.add(&other.base),
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: std::ops::Mul<Output = R> + PartialEq,
    {
        LCAStructureCoefficientsElement {
            base: self.base.scalar_mul(scalar),
        }
    }

    /// Negate the element
    pub fn negate(&self) -> Self
    where
        R: std::ops::Neg<Output = R>,
    {
        LCAStructureCoefficientsElement {
            base: self.base.negate(),
        }
    }
}

impl<R: Ring + Clone + Display, B: Clone + Eq + Hash + Display> Display for LCAStructureCoefficientsElement<R, B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Format using T^(j) notation for divided powers
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (basis, poly) in self.terms() {
            if is_zero_polynomial(poly) {
                continue;
            }

            if !first {
                write!(f, " + ")?;
            }
            first = false;

            // Format with T notation
            if poly.len() == 1 {
                if poly[0].is_one() {
                    write!(f, "{}", basis)?;
                } else {
                    write!(f, "{}*{}", poly[0], basis)?;
                }
            } else {
                let mut term_first = true;
                for (j, coeff) in poly.iter().enumerate() {
                    if coeff.is_zero() {
                        continue;
                    }

                    if !term_first {
                        write!(f, " + ")?;
                    }
                    term_first = false;

                    if j == 0 {
                        write!(f, "{}*{}", coeff, basis)?;
                    } else if coeff.is_one() {
                        write!(f, "T^({})·{}", j, basis)?;
                    } else {
                        write!(f, "{}*T^({})·{}", coeff, j, basis)?;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Element wrapper for simpler cases
///
/// Wraps an element from an underlying algebra (e.g., for quotients or subalgebras).
#[derive(Clone, Debug)]
pub struct LCAElementWrapper<E> {
    /// The underlying element
    element: E,
}

impl<E> LCAElementWrapper<E> {
    /// Create a new wrapper
    pub fn new(element: E) -> Self {
        LCAElementWrapper { element }
    }

    /// Get the underlying element
    pub fn element(&self) -> &E {
        &self.element
    }

    /// Unwrap to get the element
    pub fn into_element(self) -> E {
        self.element
    }
}

// Helper functions for polynomial operations

/// Add two polynomials (represented as coefficient vectors)
fn add_polynomials<R: Ring + std::ops::Add<Output = R>>(p: &[R], q: &[R]) -> Vec<R> {
    let max_len = p.len().max(q.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let p_coeff = if i < p.len() { p[i].clone() } else { R::zero() };
        let q_coeff = if i < q.len() { q[i].clone() } else { R::zero() };
        result.push(p_coeff + q_coeff);
    }

    // Remove trailing zeros
    while result.len() > 1 && result.last().map_or(false, |c| c.is_zero()) {
        result.pop();
    }
    if result.is_empty() {
        result.push(R::zero());
    }

    result
}

/// Check if a polynomial is zero
fn is_zero_polynomial<R: Ring + PartialEq>(poly: &[R]) -> bool {
    poly.is_empty() || poly.iter().all(|c| c.is_zero())
}

/// Differentiate a polynomial (formal derivative with respect to ∂)
///
/// Not the action of ∂ on the module (that is multiplication by ∂, see
/// [`LieConformalAlgebraElement::apply_derivation`]).
#[cfg(test)]
fn differentiate_polynomial<R: Ring + std::ops::Mul<Output = R> + From<i64>>(poly: &[R]) -> Vec<R> {
    if poly.len() <= 1 {
        return vec![R::zero()];
    }

    let mut result = Vec::with_capacity(poly.len() - 1);
    for i in 1..poly.len() {
        result.push(poly[i].clone() * R::from(i as i64));
    }

    if result.is_empty() {
        result.push(R::zero());
    }

    result
}

impl<R: Ring + Clone + Display, B: Clone + Eq + Hash + Display> Display
    for LieConformalAlgebraElement<R, B>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut first = true;
        for (basis, poly) in &self.terms {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }

            // Format polynomial
            if poly.len() == 1 && poly[0].is_one() {
                write!(f, "{}", basis)?;
            } else {
                write!(f, "(")?;
                for (i, coeff) in poly.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    if i == 0 {
                        write!(f, "{}", coeff)?;
                    } else if i == 1 {
                        write!(f, "{}*∂", coeff)?;
                    } else {
                        write!(f, "{}*∂^{}", coeff, i)?;
                    }
                }
                write!(f, ") ⊗ {}", basis)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_creation() {
        let elem: LieConformalAlgebraElement<i64, usize> =
            LieConformalAlgebraElement::from_basis(0);
        assert!(!elem.is_zero());
        assert_eq!(elem.terms().len(), 1);
    }

    #[test]
    fn test_element_addition() {
        let elem1: LieConformalAlgebraElement<i64, usize> =
            LieConformalAlgebraElement::from_basis(0);
        let elem2: LieConformalAlgebraElement<i64, usize> =
            LieConformalAlgebraElement::from_basis(1);

        let sum = elem1.add(&elem2);
        assert_eq!(sum.terms().len(), 2);
    }

    #[test]
    fn test_polynomial_operations() {
        let p = vec![1i64, 2, 3]; // 1 + 2∂ + 3∂²
        let q = vec![4i64, 5];    // 4 + 5∂

        let sum = add_polynomials(&p, &q);
        assert_eq!(sum, vec![5, 7, 3]); // 5 + 7∂ + 3∂²

        let deriv = differentiate_polynomial(&p);
        assert_eq!(deriv, vec![2, 6]); // 2 + 6∂
    }

    #[test]
    fn test_zero_polynomial() {
        assert!(is_zero_polynomial(&[0i64]));
        assert!(is_zero_polynomial(&[0i64, 0, 0]));
        assert!(!is_zero_polynomial(&[1i64]));
        assert!(!is_zero_polynomial(&[0i64, 1]));
    }

    #[test]
    fn test_is_monomial() {
        let elem: LieConformalAlgebraElement<i64, usize> =
            LieConformalAlgebraElement::from_basis(0);
        assert!(elem.is_monomial());

        let elem2: LieConformalAlgebraElement<i64, usize> =
            LieConformalAlgebraElement::from_basis(1);
        let sum = elem.add(&elem2);
        assert!(!sum.is_monomial()); // Two terms, not a monomial
    }

    #[test]
    fn test_lca_with_generators_creation() {
        let elem: LCAWithGeneratorsElement<i64, usize> =
            LCAWithGeneratorsElement::from_basis(0);
        assert!(!elem.is_zero());
        assert!(elem.is_monomial());
    }

    #[test]
    fn test_lca_with_generators_is_monomial() {
        let elem1: LCAWithGeneratorsElement<i64, usize> =
            LCAWithGeneratorsElement::from_basis(0);
        let elem2: LCAWithGeneratorsElement<i64, usize> =
            LCAWithGeneratorsElement::from_basis(1);

        assert!(elem1.is_monomial());
        assert!(elem2.is_monomial());

        let sum = elem1.add(&elem2);
        assert!(!sum.is_monomial());
    }

    #[test]
    fn test_lca_structure_coefficients_creation() {
        let elem: LCAStructureCoefficientsElement<i64, usize> =
            LCAStructureCoefficientsElement::from_basis(0);
        assert!(!elem.is_zero());
        assert!(elem.is_monomial());
    }

    #[test]
    fn test_lca_structure_coefficients_operations() {
        let elem1: LCAStructureCoefficientsElement<i64, usize> =
            LCAStructureCoefficientsElement::from_basis(0);
        let elem2: LCAStructureCoefficientsElement<i64, usize> =
            LCAStructureCoefficientsElement::from_basis(1);

        let sum = elem1.add(&elem2);
        assert_eq!(sum.terms().len(), 2);

        let neg = elem1.negate();
        assert!(!neg.is_zero());
    }

    #[test]
    fn test_lca_structure_coefficients_monomial_coefficients() {
        let mut terms = HashMap::new();
        terms.insert(0_usize, vec![2i64, 3]); // 2 + 3∂
        let elem = LCAStructureCoefficientsElement::from_terms(terms);

        let coeffs = elem.monomial_coefficients();
        assert_eq!(coeffs.len(), 1);
        assert!(coeffs.contains_key(&0));
    }

    #[test]
    fn test_lca_with_generators_zero() {
        let zero: LCAWithGeneratorsElement<i64, usize> = LCAWithGeneratorsElement::zero();
        assert!(zero.is_zero());
        assert!(!zero.is_monomial()); // Zero has no monomials
    }

    #[test]
    fn test_apply_derivation_multiplies_by_partial() {
        // In the free R[∂]-module, ∂·(p(∂) b) = (∂ p(∂)) b.
        let b: LieConformalAlgebraElement<i64, usize> = LieConformalAlgebraElement::from_basis(0);
        let db = b.apply_derivation();
        assert_eq!(db.terms().get(&0), Some(&vec![0, 1])); // ∂b
        // ∂(2 + 3∂) b = (2∂ + 3∂²) b
        let e: LieConformalAlgebraElement<i64, usize> =
            LieConformalAlgebraElement::with_polynomial(0, vec![2, 3]);
        assert_eq!(e.apply_derivation().terms().get(&0), Some(&vec![0, 2, 3]));
    }

    #[test]
    fn test_t_is_nth_derivative() {
        // Sage: Vir.inject_variables(); L.T(3) == 6*T^(3)L, i.e. ∂^3 L.
        let l: LCAWithGeneratorsElement<i64, usize> = LCAWithGeneratorsElement::from_basis(0);
        assert_eq!(l.T(3).terms().get(&0), Some(&vec![0, 0, 0, 1]));
        assert_eq!(l.T(0).terms().get(&0), Some(&vec![1]));
        // T(m) T(n) = T(m + n)
        assert_eq!(l.T(1).T(2).terms(), l.T(3).terms());
        let s: LCAStructureCoefficientsElement<i64, usize> =
            LCAStructureCoefficientsElement::from_basis(0);
        assert_eq!(s.T(2).terms().get(&0), Some(&vec![0, 0, 1]));
    }

    #[test]
    #[should_panic(expected = "not implemented")]
    fn test_structure_coefficients_element_bracket_is_unimplemented() {
        let a: LCAStructureCoefficientsElement<i64, usize> =
            LCAStructureCoefficientsElement::from_basis(0);
        let _ = a.lambda_bracket(&a, &HashMap::new());
    }

    #[test]
    fn test_lca_structure_coefficients_zero() {
        let zero: LCAStructureCoefficientsElement<i64, usize> =
            LCAStructureCoefficientsElement::zero();
        assert!(zero.is_zero());
        assert!(!zero.is_monomial());
    }
}
