//! Iwahori-Hecke Algebras
//!
//! The Iwahori-Hecke algebra is a deformation of the group algebra of a Coxeter group
//! (typically a Weyl group). These algebras play a fundamental role in representation
//! theory, quantum groups, knot theory, and statistical mechanics.
//!
//! # Mathematical Background
//!
//! The Iwahori-Hecke algebra H_{q₁,q₂}(W,S) is defined by:
//! - Generators T_s for s ∈ S (simple reflections)
//! - Quadratic relation: (T_s - q₁)(T_s - q₂) = 0
//! - Braid relations: T_s T_t T_s ... = T_t T_s T_t ... (m_{st} factors)
//!
//! When q₁ = 1 and q₂ = 0, this reduces to the group algebra of the Coxeter group.
//!
//! # Multiple Bases
//!
//! The algebra supports several bases:
//! - **T-basis**: Standard basis {T_w | w ∈ W}
//! - **C-basis**: Kazhdan-Lusztig basis (canonical basis)
//! - **Cp-basis**: Dual canonical basis
//! - **A-basis** and **B-basis**: Additional bases when 2 is a unit
//!
//! Corresponds to sage.algebras.iwahori_hecke_algebra
//!
//! # References
//!
//! - Kazhdan, D. and Lusztig, G. "Representations of Coxeter groups and Hecke algebras" (1979)
//! - Humphreys, J. "Reflection Groups and Coxeter Groups" (1990)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Basis type for the Iwahori-Hecke algebra
///
/// # Variants
///
/// - `T`: Standard T-basis (generators satisfy quadratic relation)
/// - `C`: Kazhdan-Lusztig C-basis (canonical basis)
/// - `Cp`: Kazhdan-Lusztig Cp-basis (dual canonical basis)
/// - `A`: A-basis (defined when 2 is a unit)
/// - `B`: B-basis (defined when 2 is a unit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeckeBasisType {
    /// Standard T-basis
    T,
    /// Kazhdan-Lusztig C-basis
    C,
    /// Kazhdan-Lusztig Cp-basis (dual)
    Cp,
    /// A-basis
    A,
    /// B-basis
    B,
}

impl Display for HeckeBasisType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            HeckeBasisType::T => write!(f, "T"),
            HeckeBasisType::C => write!(f, "C"),
            HeckeBasisType::Cp => write!(f, "C'"),
            HeckeBasisType::A => write!(f, "A"),
            HeckeBasisType::B => write!(f, "B"),
        }
    }
}

/// Iwahori-Hecke Algebra
///
/// A deformation of the group algebra of a Coxeter group.
///
/// # Type Parameters
///
/// * `R` - The base ring (must contain the parameters q₁, q₂)
///
/// # Mathematical Structure
///
/// The algebra is defined over a Coxeter group W with simple reflections S.
/// Each generator T_s satisfies:
///
/// (T_s - q₁)(T_s - q₂) = 0
///
/// and the braid relations from the Coxeter group.
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::iwahori_hecke_algebra::IwahoriHeckeAlgebra;
/// # use rustmath_integers::Integer;
/// // Create the Hecke algebra for the symmetric group S_3
/// let h3: IwahoriHeckeAlgebra<Integer> = IwahoriHeckeAlgebra::new(3);
/// assert_eq!(h3.rank(), 2); // 2 simple reflections
/// ```
#[derive(Debug, Clone)]
pub struct IwahoriHeckeAlgebra<R: Ring> {
    /// Rank of the Coxeter group (number of simple reflections)
    rank: usize,
    /// Cartan type (e.g., "A", "B", "D", "E")
    cartan_type: String,
    /// First parameter q₁
    q1: PhantomData<R>,
    /// Second parameter q₂
    q2: PhantomData<R>,
    /// Current basis
    basis: HeckeBasisType,
}

impl<R: Ring> IwahoriHeckeAlgebra<R> {
    /// Create a new Iwahori-Hecke algebra for type A_n (symmetric group)
    ///
    /// # Arguments
    ///
    /// * `n` - Creates the algebra for S_{n+1} (type A_n)
    pub fn new(n: usize) -> Self {
        IwahoriHeckeAlgebra {
            rank: n,
            cartan_type: format!("A{}", n),
            q1: PhantomData,
            q2: PhantomData,
            basis: HeckeBasisType::T,
        }
    }

    /// Create an Iwahori-Hecke algebra for a specific Cartan type
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The Cartan type ("A", "B", "C", "D", "E", "F", "G")
    /// * `rank` - The rank of the Coxeter group
    pub fn with_cartan_type(cartan_type: String, rank: usize) -> Self {
        IwahoriHeckeAlgebra {
            rank,
            cartan_type,
            q1: PhantomData,
            q2: PhantomData,
            basis: HeckeBasisType::T,
        }
    }

    /// Get the rank (number of simple reflections)
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &str {
        &self.cartan_type
    }

    /// Get the current basis type
    pub fn basis_type(&self) -> HeckeBasisType {
        self.basis
    }

    /// Change to a different basis
    pub fn with_basis(&self, basis: HeckeBasisType) -> Self {
        let mut algebra = self.clone();
        algebra.basis = basis;
        algebra
    }

    /// Get the zero element
    pub fn zero(&self) -> IwahoriHeckeElement<R>
    where
        R: From<i64>,
    {
        IwahoriHeckeElement::zero()
    }

    /// Get the identity element (T_e for the identity element e)
    pub fn one(&self) -> IwahoriHeckeElement<R>
    where
        R: From<i64>,
    {
        IwahoriHeckeElement::one()
    }

    /// Get the T-basis generator T_i (i-th simple reflection)
    ///
    /// # Arguments
    ///
    /// * `i` - Index of the simple reflection (0 ≤ i < rank)
    pub fn generator(&self, i: usize) -> Option<IwahoriHeckeElement<R>>
    where
        R: From<i64>,
    {
        if i >= self.rank {
            return None;
        }
        Some(IwahoriHeckeElement::generator(i))
    }

    /// Get all T-basis generators
    pub fn generators(&self) -> Vec<IwahoriHeckeElement<R>>
    where
        R: From<i64>,
    {
        (0..self.rank)
            .map(|i| IwahoriHeckeElement::generator(i))
            .collect()
    }

    /// Check if the algebra is finite-dimensional
    ///
    /// The algebra is finite-dimensional when the underlying Coxeter group is finite.
    pub fn is_finite_dimensional(&self) -> bool {
        // Type A_n is always finite
        // Types B_n, D_n, E_6, E_7, E_8, F_4, G_2, H_3, H_4, I_2(m) are finite
        matches!(self.cartan_type.chars().next(), Some('A') | Some('B') | Some('D') | Some('E') | Some('F') | Some('G') | Some('H'))
    }

    /// Dimension of the algebra (if finite-dimensional)
    pub fn dimension(&self) -> Option<usize> {
        if !self.is_finite_dimensional() {
            return None;
        }

        // Dimension is |W| where W is the Coxeter group
        match self.cartan_type.chars().next() {
            Some('A') => {
                // Type A_n: dimension is (n+1)!
                Some(factorial(self.rank + 1))
            }
            Some('B') | Some('C') => {
                // Type B_n/C_n: dimension is 2^n * n!
                Some((1 << self.rank) * factorial(self.rank))
            }
            Some('D') if self.rank >= 4 => {
                // Type D_n: dimension is 2^(n-1) * n!
                Some((1 << (self.rank - 1)) * factorial(self.rank))
            }
            Some('E') if self.rank == 6 => Some(51840),
            Some('E') if self.rank == 7 => Some(2903040),
            Some('E') if self.rank == 8 => Some(696729600),
            Some('F') if self.rank == 4 => Some(1152),
            Some('G') if self.rank == 2 => Some(12),
            _ => None,
        }
    }
}

impl<R: Ring> Display for IwahoriHeckeAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Iwahori-Hecke algebra of type {} with rank {} in {}-basis",
            self.cartan_type, self.rank, self.basis
        )
    }
}

/// Element of an Iwahori-Hecke algebra
///
/// Represented as a linear combination of basis elements indexed by
/// Coxeter group elements (words in simple reflections).
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::iwahori_hecke_algebra::IwahoriHeckeElement;
/// # use rustmath_integers::Integer;
/// let one: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::one();
/// let gen: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::generator(0);
/// ```
#[derive(Clone)]
pub struct IwahoriHeckeElement<R: Ring> {
    /// Basis elements: map from word (as vec of indices) to coefficient
    /// The word [0, 1, 0] represents s₀s₁s₀
    terms: HashMap<Vec<usize>, R>,
}

impl<R: Ring + Clone> IwahoriHeckeElement<R> {
    /// Create a new element
    pub fn new(terms: HashMap<Vec<usize>, R>) -> Self {
        IwahoriHeckeElement { terms }
    }

    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        IwahoriHeckeElement {
            terms: HashMap::new(),
        }
    }

    /// Create the identity element
    pub fn one() -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(vec![], R::from(1)); // Empty word = identity
        IwahoriHeckeElement { terms }
    }

    /// Create a generator T_i
    pub fn generator(i: usize) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(vec![i], R::from(1));
        IwahoriHeckeElement { terms }
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<Vec<usize>, R> {
        &self.terms
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.terms.is_empty() || self.terms.values().all(|c| c.is_zero())
    }

    /// Check if this is the identity
    pub fn is_one(&self) -> bool
    where
        R: From<i64> + PartialEq,
    {
        if self.terms.len() != 1 {
            return false;
        }

        if let Some((word, coeff)) = self.terms.iter().next() {
            word.is_empty() && *coeff == R::from(1)
        } else {
            false
        }
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + PartialEq,
    {
        let mut result = self.terms.clone();

        for (word, coeff) in &other.terms {
            let new_coeff = if let Some(existing) = result.get(word) {
                existing.clone() + coeff.clone()
            } else {
                coeff.clone()
            };

            if new_coeff.is_zero() {
                result.remove(word);
            } else {
                result.insert(word.clone(), new_coeff);
            }
        }

        IwahoriHeckeElement { terms: result }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: From<i64> + std::ops::Mul<Output = R> + PartialEq,
    {
        if scalar.is_zero() {
            return Self::zero();
        }

        let terms = self
            .terms
            .iter()
            .map(|(word, coeff)| (word.clone(), coeff.clone() * scalar.clone()))
            .collect();

        IwahoriHeckeElement { terms }
    }

    /// Multiply two elements (without applying braid/quadratic relations)
    ///
    /// This is formal multiplication of words. In a full implementation,
    /// this would apply the braid relations and quadratic relation.
    pub fn multiply(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq,
    {
        let mut result_terms: HashMap<Vec<usize>, R> = HashMap::new();

        for (word1, coeff1) in &self.terms {
            for (word2, coeff2) in &other.terms {
                // Concatenate words (formal multiplication)
                let mut new_word = word1.clone();
                new_word.extend_from_slice(word2);
                let new_coeff = coeff1.clone() * coeff2.clone();

                let final_coeff = if let Some(existing) = result_terms.get(&new_word) {
                    existing.clone() + new_coeff
                } else {
                    new_coeff
                };

                if final_coeff.is_zero() {
                    result_terms.remove(&new_word);
                } else {
                    result_terms.insert(new_word, final_coeff);
                }
            }
        }

        IwahoriHeckeElement { terms: result_terms }
    }

    /// Get the degree (maximum word length)
    pub fn degree(&self) -> usize {
        self.terms.keys().map(|w| w.len()).max().unwrap_or(0)
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for IwahoriHeckeElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }

        for (word, coeff) in &self.terms {
            match other.terms.get(word) {
                Some(other_coeff) if coeff == other_coeff => continue,
                _ => return false,
            }
        }

        true
    }
}

impl<R: Ring + Clone + PartialEq> Eq for IwahoriHeckeElement<R> {}

impl<R: Ring + Clone + Display> Display for IwahoriHeckeElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut sorted: Vec<_> = self.terms.iter().collect();
        sorted.sort_by_key(|(word, _)| *word);

        let mut first = true;
        for (word, coeff) in sorted {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }

            if word.is_empty() {
                write!(f, "{}", coeff)?;
            } else {
                write!(f, "{}*T_{:?}", coeff, word)?;
            }
        }

        Ok(())
    }
}

/// Compare two Coxeter group elements by Bruhat order
///
/// Returns -1 if w1 < w2, 0 if w1 = w2, 1 if w1 > w2 in Bruhat order.
/// For a simple implementation, we compare by length first.
pub fn index_cmp(w1: &[usize], w2: &[usize]) -> i32 {
    match w1.len().cmp(&w2.len()) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Equal => {
            // Lexicographic comparison
            match w1.cmp(w2) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Equal => 0,
            }
        }
    }
}

/// Normalize a Laurent polynomial to a polynomial
///
/// Extracts a polynomial from its representation in the field of fractions.
/// This is used when working with the generic parameters q₁, q₂.
pub fn normalized_laurent_polynomial<R: Ring>(poly: R) -> R {
    // In a full implementation, this would:
    // 1. Check if the Laurent polynomial is actually a polynomial
    // 2. Return the polynomial part
    // For now, just return the input
    poly
}

// Helper function for factorial
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_hecke_creation() {
        let h3: IwahoriHeckeAlgebra<Integer> = IwahoriHeckeAlgebra::new(3);
        assert_eq!(h3.rank(), 3);
        assert_eq!(h3.cartan_type(), "A3");
        assert_eq!(h3.basis_type(), HeckeBasisType::T);
    }

    #[test]
    fn test_hecke_finite_dimensional() {
        let h_a3: IwahoriHeckeAlgebra<Integer> = IwahoriHeckeAlgebra::new(3);
        assert!(h_a3.is_finite_dimensional());

        let h_b3: IwahoriHeckeAlgebra<Integer> = IwahoriHeckeAlgebra::with_cartan_type("B".to_string(), 3);
        assert!(h_b3.is_finite_dimensional());
    }

    #[test]
    fn test_hecke_dimension() {
        let h2: IwahoriHeckeAlgebra<Integer> = IwahoriHeckeAlgebra::new(2);
        assert_eq!(h2.dimension(), Some(6)); // S_3 has 6 elements

        let h3: IwahoriHeckeAlgebra<Integer> = IwahoriHeckeAlgebra::new(3);
        assert_eq!(h3.dimension(), Some(24)); // S_4 has 24 elements
    }

    #[test]
    fn test_basis_change() {
        let h3: IwahoriHeckeAlgebra<Integer> = IwahoriHeckeAlgebra::new(3);
        let h3_c = h3.with_basis(HeckeBasisType::C);

        assert_eq!(h3.basis_type(), HeckeBasisType::T);
        assert_eq!(h3_c.basis_type(), HeckeBasisType::C);
    }

    #[test]
    fn test_generators() {
        let h3: IwahoriHeckeAlgebra<Integer> = IwahoriHeckeAlgebra::new(3);
        let gens = h3.generators();
        assert_eq!(gens.len(), 3); // T_0, T_1, T_2
    }

    #[test]
    fn test_element_creation() {
        let one: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::one();
        assert!(one.is_one());
        assert!(!one.is_zero());

        let zero: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::zero();
        assert!(zero.is_zero());
        assert!(!zero.is_one());
    }

    #[test]
    fn test_element_generator() {
        let t0: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::generator(0);
        assert!(!t0.is_zero());
        assert!(!t0.is_one());
        assert_eq!(t0.degree(), 1);
    }

    #[test]
    fn test_element_addition() {
        let t0: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::generator(0);
        let t1: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::generator(1);

        let sum = t0.add(&t1);
        assert_eq!(sum.terms().len(), 2);
    }

    #[test]
    fn test_element_scalar_mul() {
        let t0: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::generator(0);
        let scaled = t0.scalar_mul(&Integer::from(3));

        assert_eq!(scaled.terms().len(), 1);
    }

    #[test]
    fn test_element_multiplication() {
        let t0: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::generator(0);
        let t1: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::generator(1);

        let product = t0.multiply(&t1);
        assert_eq!(product.degree(), 2); // T_0 * T_1 has length 2
    }

    #[test]
    fn test_identity_multiplication() {
        let one: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::one();
        let t0: IwahoriHeckeElement<Integer> = IwahoriHeckeElement::generator(0);

        let product1 = one.multiply(&t0);
        let product2 = t0.multiply(&one);

        assert_eq!(product1, t0);
        assert_eq!(product2, t0);
    }

    #[test]
    fn test_index_cmp() {
        assert_eq!(index_cmp(&[0], &[1]), 0); // Same length, compare lexicographically
        assert_eq!(index_cmp(&[0], &[0, 1]), -1); // First is shorter
        assert_eq!(index_cmp(&[0, 1], &[0]), 1); // First is longer
    }

    #[test]
    fn test_basis_type_display() {
        assert_eq!(format!("{}", HeckeBasisType::T), "T");
        assert_eq!(format!("{}", HeckeBasisType::C), "C");
        assert_eq!(format!("{}", HeckeBasisType::Cp), "C'");
    }
}
