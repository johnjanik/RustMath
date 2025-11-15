//! Hall Algebras
//!
//! The Hall algebra is an associative algebra structure on integer partitions.
//! It is the free R-module with basis I_λ indexed by partitions λ, with
//! multiplication defined via Hall polynomials:
//!   I_μ · I_λ = Σ_ν P^ν_{μ,λ}(q) I_ν
//!
//! The Hall algebra is isomorphic to symmetric functions and carries a
//! natural Hopf algebra structure, making it a connected graded Hopf algebra.
//!
//! Corresponds to sage.algebras.hall_algebra
//!
//! References:
//! - Macdonald, I.G. "Symmetric Functions and Hall Polynomials" (1995)
//! - Stanley, R.P. "Enumerative Combinatorics, Volume 2" (1999)

use rustmath_core::Ring;
use rustmath_modules::CombinatorialFreeModuleElement;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::cmp::Ordering;

/// An integer partition
///
/// Represented as a non-increasing sequence of positive integers
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Partition {
    /// Parts of the partition in non-increasing order
    parts: Vec<usize>,
}

impl Partition {
    /// Create a new partition
    ///
    /// Parts will be sorted in non-increasing order
    pub fn new(mut parts: Vec<usize>) -> Self {
        // Remove zeros and sort in decreasing order
        parts.retain(|&p| p > 0);
        parts.sort_by(|a, b| b.cmp(a));
        Partition { parts }
    }

    /// Create the empty partition
    pub fn empty() -> Self {
        Partition { parts: Vec::new() }
    }

    /// Get the parts
    pub fn parts(&self) -> &[usize] {
        &self.parts
    }

    /// Size of the partition (sum of parts)
    pub fn size(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Length (number of parts)
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Check if this is the empty partition
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Get the conjugate (transpose) partition
    pub fn conjugate(&self) -> Partition {
        if self.is_empty() {
            return Partition::empty();
        }

        let max_part = *self.parts.iter().max().unwrap_or(&0);
        let mut conj_parts = Vec::new();

        for i in 1..=max_part {
            let count = self.parts.iter().filter(|&&p| p >= i).count();
            if count > 0 {
                conj_parts.push(count);
            }
        }

        Partition::new(conj_parts)
    }
}

impl PartialOrd for Partition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Partition {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lexicographic order on parts
        self.parts.cmp(&other.parts)
    }
}

impl Display for Partition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            write!(f, "[]")
        } else {
            write!(f, "{:?}", self.parts)
        }
    }
}

/// Comparison function for transpose ordering
pub fn transpose_cmp(a: &Partition, b: &Partition) -> Ordering {
    let a_conj = a.conjugate();
    let b_conj = b.conjugate();
    a_conj.cmp(&b_conj)
}

/// Hall Algebra
///
/// The Hall algebra with partition basis I_λ
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (typically contains a parameter q)
pub struct HallAlgebra<R: Ring> {
    /// Coefficient ring marker
    coefficient_ring: std::marker::PhantomData<R>,
    /// Parameter q (stored as metadata)
    q_value: Option<String>,
}

impl<R: Ring + Clone> HallAlgebra<R> {
    /// Create a new Hall algebra
    pub fn new() -> Self {
        HallAlgebra {
            coefficient_ring: std::marker::PhantomData,
            q_value: None,
        }
    }

    /// Create with a specific q parameter name
    pub fn with_q(q_name: String) -> Self {
        HallAlgebra {
            coefficient_ring: std::marker::PhantomData,
            q_value: Some(q_name),
        }
    }

    /// Get basis element I_λ
    pub fn basis_element(&self, partition: Partition) -> HallAlgebraElement<R>
    where
        R: From<i64>,
    {
        HallAlgebraElement::basis_element(partition)
    }

    /// Product on basis: I_μ · I_λ
    ///
    /// Simplified implementation - full version would use Hall polynomials
    pub fn product_on_basis(
        &self,
        mu: &Partition,
        lambda: &Partition,
    ) -> HashMap<Partition, R>
    where
        R: From<i64> + std::ops::Add<Output = R>,
    {
        // Simplified: In the full implementation, this would compute
        // Σ_ν P^ν_{μ,λ}(q) I_ν using Hall polynomials

        // For now, just return the concatenation
        let mut combined = mu.parts().to_vec();
        combined.extend_from_slice(lambda.parts());
        let result_partition = Partition::new(combined);

        let mut map = HashMap::new();
        map.insert(result_partition, R::from(1));
        map
    }

    /// Coproduct on basis
    ///
    /// Defines the Hopf algebra coproduct structure
    pub fn coproduct_on_basis(&self, _lambda: &Partition) -> Vec<(Partition, Partition, R)>
    where
        R: From<i64>,
    {
        // Simplified implementation
        // Full version would implement the proper Hopf coproduct
        Vec::new()
    }

    /// Antipode on basis
    ///
    /// The antipode of the Hopf algebra structure
    pub fn antipode_on_basis(&self, _lambda: &Partition) -> HallAlgebraElement<R>
    where
        R: From<i64>,
    {
        // Simplified implementation
        HallAlgebraElement::zero()
    }

    /// Scalar product
    pub fn scalar(&self, _a: &HallAlgebraElement<R>, _b: &HallAlgebraElement<R>) -> R
    where
        R: From<i64>,
    {
        // Simplified implementation
        R::from(0)
    }
}

impl<R: Ring + Clone> Default for HallAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Hall Algebra with Monomial Basis
///
/// An alternative basis as monomials in I_{(1^r)} elements
pub struct HallAlgebraMonomials<R: Ring> {
    /// Coefficient ring marker
    coefficient_ring: std::marker::PhantomData<R>,
}

impl<R: Ring + Clone> HallAlgebraMonomials<R> {
    /// Create a new monomial Hall algebra
    pub fn new() -> Self {
        HallAlgebraMonomials {
            coefficient_ring: std::marker::PhantomData,
        }
    }

    /// Product on monomial basis
    ///
    /// Simply concatenates and sorts partition parts
    pub fn product_on_basis(
        &self,
        a: &Partition,
        b: &Partition,
    ) -> Partition {
        let mut combined = a.parts().to_vec();
        combined.extend_from_slice(b.parts());
        Partition::new(combined)
    }

    /// Convert monomial basis to natural (partition) basis
    pub fn to_natural_on_basis(&self, _a: &Partition) -> HashMap<Partition, R>
    where
        R: From<i64>,
    {
        // Simplified implementation
        // Full version would perform basis conversion
        HashMap::new()
    }
}

impl<R: Ring + Clone> Default for HallAlgebraMonomials<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Element of a Hall algebra
///
/// Linear combination of partition basis elements
pub struct HallAlgebraElement<R: Ring> {
    /// Coefficients for each partition
    coefficients: HashMap<Partition, R>,
}

impl<R: Ring + Clone> HallAlgebraElement<R> {
    /// Create a new element
    pub fn new(coefficients: HashMap<Partition, R>) -> Self {
        HallAlgebraElement { coefficients }
    }

    /// Create the zero element
    pub fn zero() -> Self {
        HallAlgebraElement {
            coefficients: HashMap::new(),
        }
    }

    /// Create a basis element
    pub fn basis_element(partition: Partition) -> Self
    where
        R: From<i64>,
    {
        let mut coefficients = HashMap::new();
        coefficients.insert(partition, R::from(1));
        HallAlgebraElement { coefficients }
    }

    /// Get coefficient of a partition
    pub fn coefficient(&self, partition: &Partition) -> Option<&R> {
        self.coefficients.get(partition)
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.coefficients.values().all(|c| c.is_zero())
    }

    /// Get all partitions with non-zero coefficients
    pub fn support(&self) -> Vec<&Partition> {
        self.coefficients.keys().collect()
    }

    /// Degree of the element (maximum partition size)
    pub fn degree(&self) -> Option<usize> {
        self.coefficients.keys().map(|p| p.size()).max()
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for HallAlgebraElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.coefficients.len() != other.coefficients.len() {
            return false;
        }

        for (partition, coeff) in &self.coefficients {
            if let Some(other_coeff) = other.coefficients.get(partition) {
                if coeff != other_coeff {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

impl<R: Ring + Clone + PartialEq> Eq for HallAlgebraElement<R> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_creation() {
        let p = Partition::new(vec![3, 2, 1]);
        assert_eq!(p.parts(), &[3, 2, 1]);
        assert_eq!(p.size(), 6);
        assert_eq!(p.length(), 3);

        let empty = Partition::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.size(), 0);
    }

    #[test]
    fn test_partition_sorting() {
        // Parts should be sorted in non-increasing order
        let p = Partition::new(vec![1, 3, 2]);
        assert_eq!(p.parts(), &[3, 2, 1]);
    }

    #[test]
    fn test_partition_conjugate() {
        let p = Partition::new(vec![3, 2, 1]);
        let conj = p.conjugate();
        // [3,2,1] conjugate is [3,2,1]
        assert_eq!(conj.parts(), &[3, 2, 1]);

        let p2 = Partition::new(vec![4, 2]);
        let conj2 = p2.conjugate();
        // [4,2] conjugate is [2,2,1,1]
        assert_eq!(conj2.parts(), &[2, 2, 1, 1]);
    }

    #[test]
    fn test_partition_ordering() {
        let p1 = Partition::new(vec![3, 2, 1]);
        let p2 = Partition::new(vec![3, 2]);
        let p3 = Partition::new(vec![4]);

        assert!(p1 > p2);
        assert!(p3 > p1);
    }

    #[test]
    fn test_hall_algebra_creation() {
        let algebra: HallAlgebra<i64> = HallAlgebra::new();
        let p = Partition::new(vec![2, 1]);
        let elem = algebra.basis_element(p);
        assert!(!elem.is_zero());
    }

    #[test]
    fn test_monomial_algebra() {
        let mono_algebra: HallAlgebraMonomials<i64> = HallAlgebraMonomials::new();
        let p1 = Partition::new(vec![2]);
        let p2 = Partition::new(vec![1]);
        let product = mono_algebra.product_on_basis(&p1, &p2);
        assert_eq!(product.parts(), &[2, 1]);
    }

    #[test]
    fn test_element_creation() {
        let elem: HallAlgebraElement<i64> = HallAlgebraElement::zero();
        assert!(elem.is_zero());

        let p = Partition::new(vec![3, 1]);
        let basis_elem: HallAlgebraElement<i64> = HallAlgebraElement::basis_element(p);
        assert!(!basis_elem.is_zero());
        assert_eq!(basis_elem.degree(), Some(4));
    }

    #[test]
    fn test_element_support() {
        let p1 = Partition::new(vec![2, 1]);
        let p2 = Partition::new(vec![3]);
        let mut coeffs = HashMap::new();
        coeffs.insert(p1.clone(), 2i64);
        coeffs.insert(p2.clone(), 3i64);

        let elem: HallAlgebraElement<i64> = HallAlgebraElement::new(coeffs);
        let support = elem.support();
        assert_eq!(support.len(), 2);
    }
}
