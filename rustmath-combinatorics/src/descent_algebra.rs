//! Solomon's Descent Algebra
//!
//! This module implements Solomon's descent algebra, a fundamental structure in algebraic
//! combinatorics. The descent algebra Σ_n is a subalgebra of the group algebra Q[S_n]
//! indexed by subsets of {1, 2, ..., n-1}.
//!
//! # Mathematical Background
//!
//! For a permutation π ∈ S_n, the descent set Des(π) is the set of positions i where π(i) > π(i+1).
//! For each subset S ⊆ {1, 2, ..., n-1}, the descent algebra element D_S is defined as:
//!
//! D_S = Σ_{π: Des(π) = S} π
//!
//! These elements form a basis for the descent algebra. Solomon's product formula describes
//! the multiplication of these basis elements.
//!
//! # Examples
//!
//! ## Creating descent sets
//!
//! ```
//! use rustmath_combinatorics::descent_algebra::{DescentSet, DescentAlgebraElement};
//! use rustmath_combinatorics::Permutation;
//!
//! // Create a descent set manually for S_4
//! let ds = DescentSet::new(4, vec![1, 2]).unwrap();
//! assert_eq!(ds.cardinality(), 2);
//!
//! // Create from a permutation
//! let perm = Permutation::from_vec(vec![2, 0, 1, 3]).unwrap();
//! let ds_from_perm = DescentSet::from_permutation(&perm);
//! ```
//!
//! ## Working with compositions
//!
//! ```
//! use rustmath_combinatorics::descent_algebra::DescentSet;
//!
//! // A composition [2, 1, 1] corresponds to descents at positions 1 and 2
//! let ds = DescentSet::from_composition(&[2, 1, 1]).unwrap();
//! let comp = ds.to_composition();
//! assert_eq!(comp, vec![2, 1, 1]);
//! ```
//!
//! ## Multiplying descent algebra elements
//!
//! ```
//! use rustmath_combinatorics::descent_algebra::{DescentSet, DescentAlgebraElement};
//!
//! // Create basis elements
//! let empty = DescentSet::empty(3);
//! let d_empty = DescentAlgebraElement::basis(empty.clone());
//!
//! // Multiply: D_∅ * D_∅
//! let product = d_empty.multiply(&d_empty).unwrap();
//!
//! // The identity element squared is itself
//! assert!(!product.is_zero());
//! ```
//!
//! ## Linear combinations
//!
//! ```
//! use rustmath_combinatorics::descent_algebra::{DescentSet, DescentAlgebraElement};
//! use rustmath_rationals::Rational;
//!
//! let ds1 = DescentSet::new(3, vec![0]).unwrap();
//! let ds2 = DescentSet::new(3, vec![1]).unwrap();
//!
//! let elem1 = DescentAlgebraElement::basis(ds1);
//! let elem2 = DescentAlgebraElement::basis(ds2);
//!
//! // Create 2*D_{0} + 3*D_{1}
//! let scaled1 = elem1.scalar_mul(Rational::from(2));
//! let scaled2 = elem2.scalar_mul(Rational::from(3));
//! let sum = scaled1.add(&scaled2).unwrap();
//! ```
//!
//! # References
//!
//! - Solomon, Louis (1976). "A Mackey formula in the group ring of a Coxeter group"
//! - Reutenauer, Christophe (1995). "Free Lie Algebras"
//! - Garsia, A. M., & Reutenauer, C. (1989). "A decomposition of Solomon's descent algebra"

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::{BTreeMap, BTreeSet};
use crate::permutations::Permutation;

/// A descent set, represented as a sorted set of positions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DescentSet {
    /// The set of descent positions (0-indexed)
    positions: BTreeSet<usize>,
    /// The size of the symmetric group (n for S_n)
    n: usize,
}

impl DescentSet {
    /// Create a new descent set for S_n
    pub fn new(n: usize, positions: Vec<usize>) -> Option<Self> {
        // Validate that all positions are in valid range [0, n-2]
        if n == 0 {
            return None;
        }

        for &pos in &positions {
            if pos >= n - 1 {
                return None;
            }
        }

        let positions = positions.into_iter().collect();
        Some(DescentSet { positions, n })
    }

    /// Create an empty descent set (identity permutation)
    pub fn empty(n: usize) -> Self {
        DescentSet {
            positions: BTreeSet::new(),
            n,
        }
    }

    /// Create the full descent set (maximal descents)
    pub fn full(n: usize) -> Self {
        DescentSet {
            positions: (0..n.saturating_sub(1)).collect(),
            n,
        }
    }

    /// Get the positions in this descent set
    pub fn positions(&self) -> &BTreeSet<usize> {
        &self.positions
    }

    /// Get the size of the symmetric group
    pub fn size(&self) -> usize {
        self.n
    }

    /// Check if a position is in the descent set
    pub fn contains(&self, pos: usize) -> bool {
        self.positions.contains(&pos)
    }

    /// Get the number of descents
    pub fn cardinality(&self) -> usize {
        self.positions.len()
    }

    /// Compute the descent set from a permutation
    pub fn from_permutation(perm: &Permutation) -> Self {
        let n = perm.size();
        let positions = perm.descents().into_iter().collect();
        DescentSet { positions, n }
    }

    /// Get the composition corresponding to this descent set
    ///
    /// A descent set determines a composition of n by the block sizes
    /// between consecutive descents
    pub fn to_composition(&self) -> Vec<usize> {
        if self.n == 0 {
            return vec![];
        }

        let mut composition = Vec::new();
        let mut last = 0;

        for &pos in &self.positions {
            composition.push(pos + 1 - last);
            last = pos + 1;
        }

        // Add the final block
        composition.push(self.n - last);

        composition
    }

    /// Create a descent set from a composition
    ///
    /// The composition [c_1, c_2, ..., c_k] gives descent positions
    /// at c_1, c_1 + c_2, ..., c_1 + ... + c_{k-1}
    pub fn from_composition(composition: &[usize]) -> Option<Self> {
        let n: usize = composition.iter().sum();
        if n == 0 {
            return None;
        }

        let mut positions = BTreeSet::new();
        let mut cumsum = 0;

        for (i, &block_size) in composition.iter().enumerate() {
            if block_size == 0 {
                return None; // Invalid composition
            }
            cumsum += block_size;

            // Add descent position after each block except the last
            if i < composition.len() - 1 {
                positions.insert(cumsum - 1);
            }
        }

        Some(DescentSet { positions, n })
    }
}

/// An element of Solomon's descent algebra
///
/// Represented as a linear combination of descent algebra basis elements D_S
#[derive(Debug, Clone, PartialEq)]
pub struct DescentAlgebraElement {
    /// Coefficients for each descent set basis element
    coefficients: BTreeMap<DescentSet, Rational>,
    /// The size of the symmetric group
    n: usize,
}

impl DescentAlgebraElement {
    /// Create a new descent algebra element
    pub fn new(n: usize) -> Self {
        DescentAlgebraElement {
            coefficients: BTreeMap::new(),
            n,
        }
    }

    /// Create a basis element D_S
    pub fn basis(descent_set: DescentSet) -> Self {
        let n = descent_set.n;
        let mut coefficients = BTreeMap::new();
        coefficients.insert(descent_set, Rational::one());

        DescentAlgebraElement { coefficients, n }
    }

    /// Get the size of the symmetric group
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the coefficient of a particular descent set
    pub fn coefficient(&self, descent_set: &DescentSet) -> Rational {
        self.coefficients.get(descent_set).cloned().unwrap_or(Rational::zero())
    }

    /// Get all non-zero coefficients
    pub fn coefficients(&self) -> &BTreeMap<DescentSet, Rational> {
        &self.coefficients
    }

    /// Add a term with given coefficient
    pub fn add_term(&mut self, descent_set: DescentSet, coefficient: Rational) {
        if descent_set.n != self.n {
            return; // Incompatible sizes
        }

        let entry = self.coefficients.entry(descent_set.clone()).or_insert(Rational::zero());
        *entry = entry.clone() + coefficient.clone();

        let is_zero = entry.clone() == Rational::zero();

        // Remove zero coefficients
        if is_zero {
            self.coefficients.remove(&descent_set);
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: Rational) -> Self {
        let mut result = DescentAlgebraElement::new(self.n);

        for (descent_set, coeff) in &self.coefficients {
            result.coefficients.insert(
                descent_set.clone(),
                coeff.clone() * scalar.clone()
            );
        }

        result
    }

    /// Add two descent algebra elements
    pub fn add(&self, other: &DescentAlgebraElement) -> Option<Self> {
        if self.n != other.n {
            return None;
        }

        let mut result = self.clone();

        for (descent_set, coeff) in &other.coefficients {
            result.add_term(descent_set.clone(), coeff.clone());
        }

        Some(result)
    }

    /// Multiply two descent algebra elements using Solomon's product formula
    ///
    /// The product D_S * D_T is computed by multiplying all pairs of permutations
    /// with descent sets S and T, and collecting the resulting descent sets.
    pub fn multiply(&self, other: &DescentAlgebraElement) -> Option<Self> {
        if self.n != other.n {
            return None;
        }

        let mut result = DescentAlgebraElement::new(self.n);

        for (s_set, s_coeff) in &self.coefficients {
            for (t_set, t_coeff) in &other.coefficients {
                let product_coeff = s_coeff.clone() * t_coeff.clone();
                let product_term = solomon_product_direct(s_set, t_set);

                // Add the resulting descent algebra element scaled by the coefficient
                for (descent_set, count) in product_term.coefficients {
                    result.add_term(
                        descent_set,
                        product_coeff.clone() * count
                    );
                }
            }
        }

        Some(result)
    }

    /// Check if this element is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Compute the support (non-zero basis elements)
    pub fn support(&self) -> Vec<DescentSet> {
        self.coefficients.keys().cloned().collect()
    }
}

/// Compute Solomon's product formula for two descent sets by direct enumeration
///
/// D_S * D_T = Σ_{σ: Des(σ)=S, τ: Des(τ)=T} D_{Des(στ)}
///
/// This implementation generates all permutations with descent sets S and T,
/// multiplies them, and collects the resulting descent sets.
fn solomon_product_direct(s: &DescentSet, t: &DescentSet) -> DescentAlgebraElement {
    let n = s.n;

    // Generate all permutations with the given descent sets
    let perms_s = generate_permutations_with_descent_set(s);
    let perms_t = generate_permutations_with_descent_set(t);

    let mut result = DescentAlgebraElement::new(n);

    // Multiply all pairs and collect descent sets
    for perm_s in &perms_s {
        for perm_t in &perms_t {
            // Multiply permutations
            if let Some(product) = perm_s.compose(perm_t) {
                let product_descent_set = DescentSet::from_permutation(&product);
                result.add_term(product_descent_set, Rational::one());
            }
        }
    }

    result
}

/// Generate all permutations with a given descent set
fn generate_permutations_with_descent_set(descent_set: &DescentSet) -> Vec<Permutation> {
    let n = descent_set.n;
    let all_perms = crate::permutations::all_permutations(n);

    all_perms
        .into_iter()
        .filter(|perm| {
            let ds = DescentSet::from_permutation(perm);
            ds == *descent_set
        })
        .collect()
}


/// Count the number of permutations with a given descent set
///
/// This uses direct enumeration for small n. For large n, more sophisticated
/// methods using Eulerian numbers and inclusion-exclusion can be implemented.
pub fn count_permutations_with_descent_set(descent_set: &DescentSet) -> Integer {
    let n = descent_set.n;

    if n == 0 {
        return Integer::zero();
    }
    if n == 1 {
        return Integer::one();
    }

    // Use direct counting method (suitable for small n)
    count_by_enumeration(descent_set)
}

/// Count permutations with a specific descent set by enumeration (for small n)
fn count_by_enumeration(descent_set: &DescentSet) -> Integer {
    let n = descent_set.n;

    // Generate all permutations and count those with the specified descent set
    let all_perms = crate::all_permutations(n);
    let mut count = 0;

    for perm in all_perms {
        let perm_descent_set = DescentSet::from_permutation(&perm);
        if perm_descent_set == *descent_set {
            count += 1;
        }
    }

    Integer::from(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descent_set_creation() {
        let ds = DescentSet::new(4, vec![0, 2]).unwrap();
        assert_eq!(ds.size(), 4);
        assert_eq!(ds.cardinality(), 2);
        assert!(ds.contains(0));
        assert!(ds.contains(2));
        assert!(!ds.contains(1));
    }

    #[test]
    fn test_descent_set_from_permutation() {
        // Permutation [2, 0, 1, 3] has descent at position 0 (2 > 0)
        let perm = Permutation::from_vec(vec![2, 0, 1, 3]).unwrap();
        let ds = DescentSet::from_permutation(&perm);

        assert_eq!(ds.positions(), &vec![0].into_iter().collect());
    }

    #[test]
    fn test_descent_set_composition_conversion() {
        // Composition [2, 1, 1] for n=4 gives descents at positions 1, 2
        let ds = DescentSet::from_composition(&[2, 1, 1]).unwrap();
        assert_eq!(ds.size(), 4);
        assert_eq!(ds.positions(), &vec![1, 2].into_iter().collect());

        // Convert back to composition
        let comp = ds.to_composition();
        assert_eq!(comp, vec![2, 1, 1]);
    }

    #[test]
    fn test_descent_set_empty_and_full() {
        let empty = DescentSet::empty(4);
        assert_eq!(empty.cardinality(), 0);
        assert_eq!(empty.to_composition(), vec![4]);

        let full = DescentSet::full(4);
        assert_eq!(full.cardinality(), 3);
        assert_eq!(full.to_composition(), vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_basis_element() {
        let ds = DescentSet::new(3, vec![1]).unwrap();
        let elem = DescentAlgebraElement::basis(ds.clone());

        assert_eq!(elem.coefficient(&ds), Rational::one());
        assert_eq!(elem.size(), 3);
    }

    #[test]
    fn test_algebra_addition() {
        let ds1 = DescentSet::new(3, vec![0]).unwrap();
        let ds2 = DescentSet::new(3, vec![1]).unwrap();

        let elem1 = DescentAlgebraElement::basis(ds1.clone());
        let elem2 = DescentAlgebraElement::basis(ds2.clone());

        let sum = elem1.add(&elem2).unwrap();

        assert_eq!(sum.coefficient(&ds1), Rational::one());
        assert_eq!(sum.coefficient(&ds2), Rational::one());
    }

    #[test]
    fn test_scalar_multiplication() {
        let ds = DescentSet::new(3, vec![1]).unwrap();
        let elem = DescentAlgebraElement::basis(ds.clone());

        let scaled = elem.scalar_mul(Rational::from(2));
        assert_eq!(scaled.coefficient(&ds), Rational::from(2));
    }

    #[test]
    fn test_solomon_product_simple() {
        // Test D_∅ * D_∅ = D_∅ for S_2
        let empty = DescentSet::empty(2);
        let d_empty = DescentAlgebraElement::basis(empty.clone());

        let product = d_empty.multiply(&d_empty).unwrap();

        // Should give back D_∅
        assert_eq!(product.coefficient(&empty), Rational::one());
    }

    #[test]
    fn test_solomon_product_s3() {
        // Test in S_3: D_{1} * D_∅
        let ds1 = DescentSet::new(3, vec![1]).unwrap(); // Composition [2, 1]
        let empty = DescentSet::empty(3); // Composition [3]

        let d1 = DescentAlgebraElement::basis(ds1.clone());
        let d_empty = DescentAlgebraElement::basis(empty.clone());

        let product = d1.multiply(&d_empty).unwrap();

        // The product should have non-zero coefficients
        assert!(!product.is_zero());
    }

    #[test]
    fn test_composition_roundtrip() {
        // Test various compositions round-trip correctly
        let compositions = vec![
            vec![4],
            vec![1, 3],
            vec![2, 2],
            vec![3, 1],
            vec![1, 1, 2],
            vec![1, 2, 1],
            vec![2, 1, 1],
            vec![1, 1, 1, 1],
        ];

        for comp in compositions {
            let ds = DescentSet::from_composition(&comp).unwrap();
            let comp2 = ds.to_composition();
            assert_eq!(comp, comp2, "Composition {:?} didn't round-trip", comp);
        }
    }

    #[test]
    fn test_count_permutations_descent_set() {
        // For S_3, the empty descent set corresponds to [0,1,2]
        let empty = DescentSet::empty(3);
        let count = count_permutations_with_descent_set(&empty);
        assert_eq!(count, Integer::one());

        // Descent set {1} corresponds to permutations [0,2,1] and [1,2,0]
        let ds1 = DescentSet::new(3, vec![1]).unwrap();
        let count1 = count_permutations_with_descent_set(&ds1);
        assert_eq!(count1, Integer::from(2));
    }

    #[test]
    fn test_algebra_zero_element() {
        let elem = DescentAlgebraElement::new(3);
        assert!(elem.is_zero());

        let ds = DescentSet::empty(3);
        let basis = DescentAlgebraElement::basis(ds);
        assert!(!basis.is_zero());
    }
}
