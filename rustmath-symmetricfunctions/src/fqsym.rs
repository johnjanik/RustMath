//! Free quasi-symmetric functions (FQSym) with F-basis and shuffle product
//!
//! FQSym is the Hopf algebra of free quasi-symmetric functions. It is indexed by
//! integer compositions (ordered partitions) and has the shuffle product as its
//! multiplication operation.
//!
//! The F-basis is the fundamental basis for FQSym, where F_I is indexed by a
//! composition I.

use rustmath_combinatorics::Composition;
use rustmath_core::Ring;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// A free quasi-symmetric function represented in the F-basis
///
/// FQSym elements are linear combinations of F-basis elements, each indexed
/// by a composition. Unlike symmetric functions (indexed by partitions),
/// the order of parts in the composition matters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FQSym {
    /// Coefficients: maps composition to coefficient in F-basis
    pub coeffs: HashMap<Composition, Rational>,
}

impl FQSym {
    /// Create a new zero element in FQSym
    pub fn zero() -> Self {
        FQSym {
            coeffs: HashMap::new(),
        }
    }

    /// Create a basis element F_I for composition I
    ///
    /// # Example
    /// ```
    /// use rustmath_symmetricfunctions::fqsym::FQSym;
    /// use rustmath_combinatorics::Composition;
    ///
    /// let comp = Composition::new(vec![2, 1, 1]).unwrap();
    /// let f = FQSym::f_basis(comp);
    /// ```
    pub fn f_basis(composition: Composition) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(composition, Rational::one());
        FQSym { coeffs }
    }

    /// Create the unit element (empty composition)
    pub fn one() -> Self {
        let empty_comp = Composition::new(vec![]).unwrap();
        FQSym::f_basis(empty_comp)
    }

    /// Add a term with the given coefficient
    pub fn add_term(&mut self, composition: Composition, coeff: Rational) {
        if !coeff.is_zero() {
            let entry = self.coeffs.entry(composition.clone()).or_insert(Rational::zero());
            *entry = entry.clone() + coeff;

            // Remove if the result is zero
            if entry.is_zero() {
                self.coeffs.remove(&composition);
            }
        }
    }

    /// Get the coefficient of a composition
    pub fn coeff(&self, composition: &Composition) -> Rational {
        self.coeffs.get(composition).cloned().unwrap_or(Rational::zero())
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.values().all(|c| c.is_zero())
    }

    /// Get the degree (largest composition sum)
    pub fn degree(&self) -> usize {
        self.coeffs.keys().map(|c| c.sum()).max().unwrap_or(0)
    }

    /// Get all compositions with non-zero coefficients
    pub fn support(&self) -> Vec<Composition> {
        self.coeffs
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(comp, _)| comp.clone())
            .collect()
    }

    /// Multiply by a scalar
    pub fn scale(&self, scalar: &Rational) -> Self {
        if scalar.is_zero() {
            return FQSym::zero();
        }

        let mut result = self.clone();
        for coeff in result.coeffs.values_mut() {
            *coeff = coeff.clone() * scalar.clone();
        }
        result
    }

    /// Add two FQSym elements
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (composition, coeff) in &other.coeffs {
            result.add_term(composition.clone(), coeff.clone());
        }
        result
    }

    /// Subtract two FQSym elements
    pub fn subtract(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (composition, coeff) in &other.coeffs {
            result.add_term(composition.clone(), -coeff.clone());
        }
        result
    }

    /// Negate an FQSym element
    pub fn negate(&self) -> Self {
        self.scale(&Rational::from(-1))
    }

    /// Shuffle product of two FQSym elements
    ///
    /// The shuffle product is the multiplication in FQSym. It is defined by:
    /// F_I * F_J = sum over all shuffles σ of F_{σ(I,J)}
    ///
    /// where a shuffle is a way to interleave the parts of I and J while
    /// preserving their relative order.
    pub fn shuffle_product(&self, other: &Self) -> Self {
        let mut result = FQSym::zero();

        for (comp1, coeff1) in &self.coeffs {
            for (comp2, coeff2) in &other.coeffs {
                // Compute all shuffles of comp1 and comp2
                let shuffles = compute_shuffles(comp1.parts(), comp2.parts());

                let product_coeff = coeff1.clone() * coeff2.clone();

                for shuffle in shuffles {
                    if let Some(shuffle_comp) = Composition::new(shuffle) {
                        result.add_term(shuffle_comp, product_coeff.clone());
                    }
                }
            }
        }

        result
    }

    /// Compute the nth power using the shuffle product
    pub fn shuffle_power(&self, n: usize) -> Self {
        if n == 0 {
            return FQSym::one();
        }
        if n == 1 {
            return self.clone();
        }

        let mut result = self.clone();
        for _ in 1..n {
            result = result.shuffle_product(self);
        }
        result
    }
}

/// Compute all shuffles of two sequences
///
/// A shuffle of sequences a and b is a way to interleave them while
/// preserving the relative order within each sequence.
///
/// # Example
/// shuffles([1,2], [3]) = [[1,2,3], [1,3,2], [3,1,2]]
fn compute_shuffles(a: &[usize], b: &[usize]) -> Vec<Vec<usize>> {
    if a.is_empty() {
        return vec![b.to_vec()];
    }
    if b.is_empty() {
        return vec![a.to_vec()];
    }

    let mut result = Vec::new();

    // Take first element from a
    let rest_a = compute_shuffles(&a[1..], b);
    for mut shuffle in rest_a {
        shuffle.insert(0, a[0]);
        result.push(shuffle);
    }

    // Take first element from b
    let rest_b = compute_shuffles(a, &b[1..]);
    for mut shuffle in rest_b {
        shuffle.insert(0, b[0]);
        result.push(shuffle);
    }

    result
}

/// Concatenation product (overlapping shuffle)
///
/// This is another product on FQSym where F_I * F_J = F_{I||J}
/// where I||J is the concatenation of compositions
impl FQSym {
    pub fn concatenation_product(&self, other: &Self) -> Self {
        let mut result = FQSym::zero();

        for (comp1, coeff1) in &self.coeffs {
            for (comp2, coeff2) in &other.coeffs {
                let mut concat_parts = comp1.parts().to_vec();
                concat_parts.extend_from_slice(comp2.parts());

                if let Some(concat_comp) = Composition::new(concat_parts) {
                    let product_coeff = coeff1.clone() * coeff2.clone();
                    result.add_term(concat_comp, product_coeff);
                }
            }
        }

        result
    }
}

/// Standard FQSym operations
impl std::ops::Add for FQSym {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        FQSym::add(&self, &other)
    }
}

impl std::ops::Sub for FQSym {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        FQSym::subtract(&self, &other)
    }
}

impl std::ops::Neg for FQSym {
    type Output = Self;

    fn neg(self) -> Self {
        FQSym::negate(&self)
    }
}

impl std::ops::Mul for FQSym {
    type Output = Self;

    /// Multiply using shuffle product (default multiplication)
    fn mul(self, other: Self) -> Self {
        FQSym::shuffle_product(&self, &other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fqsym_zero() {
        let f = FQSym::zero();
        assert!(f.is_zero());
        assert_eq!(f.degree(), 0);
    }

    #[test]
    fn test_fqsym_one() {
        let f = FQSym::one();
        assert!(!f.is_zero());
        assert_eq!(f.degree(), 0);

        // Verify it has coefficient 1 on empty composition
        let empty = Composition::new(vec![]).unwrap();
        assert_eq!(f.coeff(&empty), Rational::one());
    }

    #[test]
    fn test_fqsym_basis_element() {
        let comp = Composition::new(vec![2, 1]).unwrap();
        let f = FQSym::f_basis(comp.clone());

        assert!(!f.is_zero());
        assert_eq!(f.degree(), 3);
        assert_eq!(f.coeff(&comp), Rational::one());

        let other_comp = Composition::new(vec![1, 2]).unwrap();
        assert_eq!(f.coeff(&other_comp), Rational::zero());
    }

    #[test]
    fn test_fqsym_addition() {
        let comp1 = Composition::new(vec![2, 1]).unwrap();
        let comp2 = Composition::new(vec![1, 2]).unwrap();

        let f1 = FQSym::f_basis(comp1.clone());
        let f2 = FQSym::f_basis(comp2.clone());

        let sum = f1.add(&f2);
        assert_eq!(sum.coeff(&comp1), Rational::one());
        assert_eq!(sum.coeff(&comp2), Rational::one());
        assert_eq!(sum.degree(), 3);
    }

    #[test]
    fn test_fqsym_scale() {
        let comp = Composition::new(vec![2, 1]).unwrap();
        let f = FQSym::f_basis(comp.clone());

        let scaled = f.scale(&Rational::from(3));
        assert_eq!(scaled.coeff(&comp), Rational::from(3));

        let zero_scaled = f.scale(&Rational::zero());
        assert!(zero_scaled.is_zero());
    }

    #[test]
    fn test_compute_shuffles_simple() {
        let shuffles = compute_shuffles(&[1], &[2]);
        assert_eq!(shuffles.len(), 2);
        assert!(shuffles.contains(&vec![1, 2]));
        assert!(shuffles.contains(&vec![2, 1]));
    }

    #[test]
    fn test_compute_shuffles_empty() {
        let shuffles = compute_shuffles(&[], &[1, 2]);
        assert_eq!(shuffles.len(), 1);
        assert_eq!(shuffles[0], vec![1, 2]);

        let shuffles2 = compute_shuffles(&[1, 2], &[]);
        assert_eq!(shuffles2.len(), 1);
        assert_eq!(shuffles2[0], vec![1, 2]);
    }

    #[test]
    fn test_compute_shuffles_two_two() {
        let shuffles = compute_shuffles(&[1, 2], &[3, 4]);
        // Should have C(4,2) = 6 shuffles
        assert_eq!(shuffles.len(), 6);

        // All should have length 4
        for shuffle in &shuffles {
            assert_eq!(shuffle.len(), 4);
        }

        // Check some specific shuffles
        assert!(shuffles.contains(&vec![1, 2, 3, 4]));
        assert!(shuffles.contains(&vec![3, 4, 1, 2]));
        assert!(shuffles.contains(&vec![1, 3, 2, 4]));
    }

    #[test]
    fn test_shuffle_product_with_one() {
        let comp = Composition::new(vec![2, 1]).unwrap();
        let f = FQSym::f_basis(comp.clone());
        let one = FQSym::one();

        // F * 1 = F
        let product = f.shuffle_product(&one);
        assert_eq!(product.coeff(&comp), Rational::one());
        assert_eq!(product.support().len(), 1);

        // 1 * F = F
        let product2 = one.shuffle_product(&f);
        assert_eq!(product2.coeff(&comp), Rational::one());
        assert_eq!(product2.support().len(), 1);
    }

    #[test]
    fn test_shuffle_product_simple() {
        // F_[1] * F_[1] should give 2 * F_[1,1]
        let comp1 = Composition::new(vec![1]).unwrap();
        let f1 = FQSym::f_basis(comp1);

        let product = f1.shuffle_product(&f1);

        let result_comp = Composition::new(vec![1, 1]).unwrap();
        assert_eq!(product.coeff(&result_comp), Rational::from(2));
    }

    #[test]
    fn test_shuffle_product_different() {
        // F_[1] * F_[2]
        let comp1 = Composition::new(vec![1]).unwrap();
        let comp2 = Composition::new(vec![2]).unwrap();

        let f1 = FQSym::f_basis(comp1);
        let f2 = FQSym::f_basis(comp2);

        let product = f1.shuffle_product(&f2);

        // Should give F_[1,2] + F_[2,1]
        let result1 = Composition::new(vec![1, 2]).unwrap();
        let result2 = Composition::new(vec![2, 1]).unwrap();

        assert_eq!(product.coeff(&result1), Rational::one());
        assert_eq!(product.coeff(&result2), Rational::one());
        assert_eq!(product.support().len(), 2);
    }

    #[test]
    fn test_concatenation_product() {
        let comp1 = Composition::new(vec![1, 2]).unwrap();
        let comp2 = Composition::new(vec![3]).unwrap();

        let f1 = FQSym::f_basis(comp1);
        let f2 = FQSym::f_basis(comp2);

        let product = f1.concatenation_product(&f2);

        let result = Composition::new(vec![1, 2, 3]).unwrap();
        assert_eq!(product.coeff(&result), Rational::one());
        assert_eq!(product.support().len(), 1);
    }

    #[test]
    fn test_shuffle_power() {
        let comp = Composition::new(vec![1]).unwrap();
        let f = FQSym::f_basis(comp);

        // F_[1]^0 = 1
        let power0 = f.shuffle_power(0);
        assert_eq!(power0.coeff(&Composition::new(vec![]).unwrap()), Rational::one());

        // F_[1]^1 = F_[1]
        let power1 = f.shuffle_power(1);
        assert_eq!(power1.coeff(&Composition::new(vec![1]).unwrap()), Rational::one());

        // F_[1]^2 = 2*F_[1,1]
        let power2 = f.shuffle_power(2);
        assert_eq!(power2.coeff(&Composition::new(vec![1, 1]).unwrap()), Rational::from(2));
    }

    #[test]
    fn test_operator_overloads() {
        let comp1 = Composition::new(vec![1]).unwrap();
        let comp2 = Composition::new(vec![2]).unwrap();

        let f1 = FQSym::f_basis(comp1.clone());
        let f2 = FQSym::f_basis(comp2.clone());

        // Test addition
        let sum = f1.clone() + f2.clone();
        assert_eq!(sum.coeff(&comp1), Rational::one());
        assert_eq!(sum.coeff(&comp2), Rational::one());

        // Test subtraction
        let diff = f1.clone() - f2.clone();
        assert_eq!(diff.coeff(&comp1), Rational::one());
        assert_eq!(diff.coeff(&comp2), -Rational::one());

        // Test negation
        let neg = -f1.clone();
        assert_eq!(neg.coeff(&comp1), -Rational::one());

        // Test multiplication (shuffle product)
        let prod = f1.clone() * f2.clone();
        let result1 = Composition::new(vec![1, 2]).unwrap();
        let result2 = Composition::new(vec![2, 1]).unwrap();
        assert_eq!(prod.coeff(&result1), Rational::one());
        assert_eq!(prod.coeff(&result2), Rational::one());
    }

    #[test]
    fn test_fqsym_associativity() {
        // Test that shuffle product is associative: (a*b)*c = a*(b*c)
        let a = FQSym::f_basis(Composition::new(vec![1]).unwrap());
        let b = FQSym::f_basis(Composition::new(vec![1]).unwrap());
        let c = FQSym::f_basis(Composition::new(vec![1]).unwrap());

        let left = a.shuffle_product(&b).shuffle_product(&c);
        let right = a.shuffle_product(&b.shuffle_product(&c));

        // Both should equal the same linear combination
        assert_eq!(left, right);
    }

    #[test]
    fn test_fqsym_commutativity() {
        // Shuffle product is commutative
        let comp1 = Composition::new(vec![1, 2]).unwrap();
        let comp2 = Composition::new(vec![2]).unwrap();

        let f1 = FQSym::f_basis(comp1);
        let f2 = FQSym::f_basis(comp2);

        let prod1 = f1.shuffle_product(&f2);
        let prod2 = f2.shuffle_product(&f1);

        assert_eq!(prod1, prod2);
    }

    #[test]
    fn test_fqsym_distributivity() {
        // Test that shuffle product distributes over addition
        let a = FQSym::f_basis(Composition::new(vec![1]).unwrap());
        let b = FQSym::f_basis(Composition::new(vec![2]).unwrap());
        let c = FQSym::f_basis(Composition::new(vec![1, 1]).unwrap());

        let left = a.shuffle_product(&(b.clone() + c.clone()));
        let right = a.shuffle_product(&b) + a.shuffle_product(&c);

        assert_eq!(left, right);
    }
}
