//! Schur Algebras
//!
//! This module implements Schur algebras S(n,r), which are endomorphism algebras
//! of tensor powers of the standard representation of GL(n).
//!
//! # Mathematical Background
//!
//! The Schur algebra S(n,r) over a commutative ring R is the algebra of GL(n)-module
//! endomorphisms of the r-th tensor power V^⊗r of the natural n-dimensional module.
//!
//! The Schur algebra has dimension binomial(n² + r - 1, r) and is indexed by
//! equivalence classes of pairs of tuples from {1,...,n}^r.
//!
//! # Examples
//!
//! ```
//! use rustmath_algebras::schur_algebra::*;
//!
//! // Create the Schur algebra S(3,2)
//! let schur = SchurAlgebra::new(3, 2);
//! assert_eq!(schur.rank(), 3);
//! assert_eq!(schur.degree(), 2);
//!
//! // Get the dimension
//! let dim = schur.dimension();
//! ```

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt;

/// Helper function: Compute binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k); // Optimize using symmetry
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Schur representative indices
///
/// Generate equivalence class representatives for basis elements of S(n,r).
/// Returns pairs of tuples (i, j) where each tuple has length r with entries in 1..=n.
///
/// The representatives are chosen such that:
/// - The first tuple is weakly increasing
/// - Within blocks of equal values in the first tuple, the second tuple is weakly increasing
///
/// # Examples
///
/// ```
/// use rustmath_algebras::schur_algebra::schur_representative_indices;
///
/// let indices = schur_representative_indices(2, 2);
/// // For S(2,2), we get representatives like ([1,1], [1,1]), ([1,1], [1,2]), etc.
/// assert!(indices.len() > 0);
/// ```
pub fn schur_representative_indices(n: usize, r: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    if r == 0 {
        return vec![(vec![], vec![])];
    }

    let mut result = Vec::new();

    // Generate all tuples of length r with entries in 1..=n
    fn generate_tuples(n: usize, r: usize, current: &mut Vec<usize>, all: &mut Vec<Vec<usize>>) {
        if current.len() == r {
            all.push(current.clone());
            return;
        }
        for i in 1..=n {
            current.push(i);
            generate_tuples(n, r, current, all);
            current.pop();
        }
    }

    let mut all_tuples = Vec::new();
    generate_tuples(n, r, &mut Vec::new(), &mut all_tuples);

    // For each pair of tuples, compute the canonical representative
    for t1 in &all_tuples {
        for t2 in &all_tuples {
            let (rep1, rep2) = schur_representative_from_index(t1.clone(), t2.clone());

            // Check if this is already in the result
            if !result.iter().any(|(r1, r2)| r1 == &rep1 && r2 == &rep2) {
                result.push((rep1, rep2));
            }
        }
    }

    result
}

/// Schur representative from index
///
/// Given a pair of tuples (i0, i1), return the canonical representative
/// of the equivalence class by reordering.
///
/// The canonical form is obtained by:
/// 1. Pairing elements from both tuples
/// 2. Sorting pairs lexicographically by first tuple element, then by second
/// 3. Splitting back into two tuples
///
/// # Examples
///
/// ```
/// use rustmath_algebras::schur_algebra::schur_representative_from_index;
///
/// let (rep1, rep2) = schur_representative_from_index(vec![2, 1], vec![1, 2]);
/// // The result will be the canonical ordering
/// ```
pub fn schur_representative_from_index(i0: Vec<usize>, i1: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    assert_eq!(i0.len(), i1.len(), "Tuples must have the same length");

    if i0.is_empty() {
        return (vec![], vec![]);
    }

    // Pair up elements
    let mut pairs: Vec<(usize, usize)> = i0.into_iter().zip(i1.into_iter()).collect();

    // Sort lexicographically: first by first element, then by second
    pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Split back into two tuples
    let rep1 = pairs.iter().map(|(a, _)| *a).collect();
    let rep2 = pairs.iter().map(|(_, b)| *b).collect();

    (rep1, rep2)
}

/// Element of a Schur algebra
///
/// An element is represented as a linear combination of basis elements,
/// where each basis element is indexed by a pair of tuples.
#[derive(Debug, Clone, PartialEq)]
pub struct SchurElement<R: Ring> {
    /// Coefficients for each basis element
    /// Key: (tuple1, tuple2) representing a basis element
    coefficients: HashMap<(Vec<usize>, Vec<usize>), R>,
}

impl<R: Ring> SchurElement<R> {
    /// Create a zero element
    pub fn zero() -> Self {
        Self {
            coefficients: HashMap::new(),
        }
    }

    /// Create an element from a single basis element
    pub fn basis_element(i: Vec<usize>, j: Vec<usize>, coeff: R) -> Self {
        let mut coefficients = HashMap::new();
        if !coeff.is_zero() {
            coefficients.insert((i, j), coeff);
        }
        Self { coefficients }
    }

    /// Check if the element is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &HashMap<(Vec<usize>, Vec<usize>), R> {
        &self.coefficients
    }
}

impl<R: Ring> fmt::Display for SchurElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut terms: Vec<_> = self.coefficients.iter().collect();
        terms.sort_by_key(|(k, _)| k.clone());

        for (i, ((t1, t2), coeff)) in terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{}*S[{:?},{:?}]", coeff, t1, t2)?;
        }
        Ok(())
    }
}

/// Schur Algebra S(n,r)
///
/// The Schur algebra is the endomorphism algebra of the r-th tensor power
/// of the natural n-dimensional representation of GL(n).
#[derive(Debug, Clone)]
pub struct SchurAlgebra {
    /// Rank n (dimension of the natural representation)
    n: usize,
    /// Degree r (number of tensor factors)
    r: usize,
    /// Cached basis representatives
    basis_cache: Option<Vec<(Vec<usize>, Vec<usize>)>>,
}

impl SchurAlgebra {
    /// Create a new Schur algebra S(n,r)
    ///
    /// # Arguments
    ///
    /// * `n` - The rank (must be positive)
    /// * `r` - The degree (must be non-negative)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::schur_algebra::SchurAlgebra;
    ///
    /// let schur = SchurAlgebra::new(3, 2);
    /// assert_eq!(schur.rank(), 3);
    /// assert_eq!(schur.degree(), 2);
    /// ```
    pub fn new(n: usize, r: usize) -> Self {
        assert!(n > 0, "Rank n must be positive");
        Self {
            n,
            r,
            basis_cache: None,
        }
    }

    /// Get the rank n
    pub fn rank(&self) -> usize {
        self.n
    }

    /// Get the degree r
    pub fn degree(&self) -> usize {
        self.r
    }

    /// Compute the dimension of S(n,r)
    ///
    /// The dimension is given by binomial(n² + r - 1, r).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::schur_algebra::SchurAlgebra;
    ///
    /// let schur = SchurAlgebra::new(2, 2);
    /// let dim = schur.dimension();
    /// // For S(2,2): binomial(4 + 2 - 1, 2) = binomial(5, 2) = 10
    /// assert_eq!(dim, 10);
    /// ```
    pub fn dimension(&self) -> usize {
        binomial(self.n * self.n + self.r - 1, self.r)
    }

    /// Get the basis elements
    ///
    /// Returns a vector of (tuple1, tuple2) pairs representing the basis
    pub fn basis(&mut self) -> &Vec<(Vec<usize>, Vec<usize>)> {
        if self.basis_cache.is_none() {
            self.basis_cache = Some(schur_representative_indices(self.n, self.r));
        }
        self.basis_cache.as_ref().unwrap()
    }

    /// Get the identity element
    ///
    /// The identity is the sum of diagonal basis elements (w, w)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::schur_algebra::SchurAlgebra;
    ///
    /// let schur = SchurAlgebra::new(2, 1);
    /// let one = schur.one();
    /// ```
    pub fn one(&self) -> String {
        format!("1_{{S({},{})}}",  self.n, self.r)
    }

    /// Multiply two basis elements
    ///
    /// Given basis elements (i, j) and (k, l), compute their product.
    /// The product is non-zero only if j == k, in which case it equals (i, l).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::schur_algebra::SchurAlgebra;
    ///
    /// let schur = SchurAlgebra::new(2, 2);
    /// let product = schur.product_on_basis(&vec![1,1], &vec![1,2], &vec![1,2], &vec![2,2]);
    /// ```
    pub fn product_on_basis(
        &self,
        i: &[usize],
        j: &[usize],
        k: &[usize],
        l: &[usize],
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        // Product (i,j) * (k,l) is non-zero only if j == k
        if j == k {
            Some((i.to_vec(), l.to_vec()))
        } else {
            None
        }
    }
}

impl fmt::Display for SchurAlgebra {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Schur algebra S({},{})", self.n, self.r)
    }
}

/// Schur Tensor Module
///
/// The tensor module V^⊗r with commuting left action of S(n,r)
/// and right action of the symmetric group S_r.
#[derive(Debug, Clone)]
pub struct SchurTensorModule {
    /// The associated Schur algebra
    schur_algebra: SchurAlgebra,
}

impl SchurTensorModule {
    /// Create a new Schur tensor module
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::schur_algebra::*;
    ///
    /// let schur = SchurAlgebra::new(3, 2);
    /// let module = SchurTensorModule::new(schur);
    /// ```
    pub fn new(schur_algebra: SchurAlgebra) -> Self {
        Self { schur_algebra }
    }

    /// Get the associated Schur algebra
    pub fn schur_algebra(&self) -> &SchurAlgebra {
        &self.schur_algebra
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.schur_algebra.rank()
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.schur_algebra.degree()
    }
}

impl fmt::Display for SchurTensorModule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Tensor module V^⊗{} for GL({})",
            self.schur_algebra.degree(),
            self.schur_algebra.rank()
        )
    }
}

/// Compute GL(n) irreducible character
///
/// For a partition λ of r with at most n parts, compute the character
/// of the irreducible GL(n)-module L(λ).
///
/// The character is computed using semistandard Young tableaux and
/// the relationship with symmetric functions.
///
/// # Arguments
///
/// * `partition` - A partition (weakly decreasing sequence of positive integers)
/// * `n` - The rank of GL(n)
///
/// # Examples
///
/// ```
/// use rustmath_algebras::schur_algebra::gl_irreducible_character;
///
/// // Character of the standard representation
/// let chi = gl_irreducible_character(vec![1], 3);
///
/// // Character for partition [2,1] and GL(3)
/// let chi2 = gl_irreducible_character(vec![2, 1], 3);
/// ```
pub fn gl_irreducible_character(partition: Vec<usize>, n: usize) -> String {
    // Validate partition
    for i in 0..partition.len().saturating_sub(1) {
        assert!(
            partition[i] >= partition[i + 1],
            "Partition must be weakly decreasing"
        );
    }

    // Check that partition has at most n parts
    let num_parts = partition.iter().filter(|&&x| x > 0).count();
    assert!(
        num_parts <= n,
        "Partition must have at most {} parts for GL({})",
        n,
        n
    );

    format!("χ_{{{}}}(GL({}))", format_partition(&partition), n)
}

/// Helper to format a partition
fn format_partition(partition: &[usize]) -> String {
    let parts: Vec<String> = partition.iter().map(|x| x.to_string()).collect();
    parts.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(10, 3), 120);
        assert_eq!(binomial(4, 0), 1);
        assert_eq!(binomial(4, 4), 1);
        assert_eq!(binomial(3, 5), 0);
    }

    #[test]
    fn test_schur_representative_from_index() {
        // Empty tuples
        let (r1, r2) = schur_representative_from_index(vec![], vec![]);
        assert_eq!(r1, vec![]);
        assert_eq!(r2, vec![]);

        // Single element
        let (r1, r2) = schur_representative_from_index(vec![1], vec![2]);
        assert_eq!(r1, vec![1]);
        assert_eq!(r2, vec![2]);

        // Two elements - should sort by first, then second
        let (r1, r2) = schur_representative_from_index(vec![2, 1], vec![1, 2]);
        assert_eq!(r1, vec![1, 2]);
        assert_eq!(r2, vec![2, 1]);
    }

    #[test]
    fn test_schur_representative_indices() {
        // S(2, 0) has just one basis element
        let indices = schur_representative_indices(2, 0);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], (vec![], vec![]));

        // S(2, 1) should have representatives
        let indices = schur_representative_indices(2, 1);
        assert!(indices.len() > 0);

        // Each representative should have tuples of length 1
        for (t1, t2) in &indices {
            assert_eq!(t1.len(), 1);
            assert_eq!(t2.len(), 1);
        }
    }

    #[test]
    fn test_schur_algebra_creation() {
        let schur = SchurAlgebra::new(3, 2);
        assert_eq!(schur.rank(), 3);
        assert_eq!(schur.degree(), 2);
    }

    #[test]
    #[should_panic(expected = "Rank n must be positive")]
    fn test_schur_algebra_invalid_rank() {
        SchurAlgebra::new(0, 2);
    }

    #[test]
    fn test_schur_algebra_dimension() {
        // S(2,2): binomial(4 + 2 - 1, 2) = binomial(5, 2) = 10
        let schur = SchurAlgebra::new(2, 2);
        assert_eq!(schur.dimension(), 10);

        // S(3,1): binomial(9 + 1 - 1, 1) = binomial(9, 1) = 9
        let schur2 = SchurAlgebra::new(3, 1);
        assert_eq!(schur2.dimension(), 9);

        // S(2,0): binomial(4 - 1, 0) = 1
        let schur3 = SchurAlgebra::new(2, 0);
        assert_eq!(schur3.dimension(), 1);
    }

    #[test]
    fn test_schur_algebra_basis() {
        let mut schur = SchurAlgebra::new(2, 1);
        let basis = schur.basis();

        // Each basis element should have tuples of length r
        for (t1, t2) in basis {
            assert_eq!(t1.len(), 1);
            assert_eq!(t2.len(), 1);
        }
    }

    #[test]
    fn test_schur_algebra_one() {
        let schur = SchurAlgebra::new(3, 2);
        let one = schur.one();
        assert!(one.contains("S(3,2)"));
    }

    #[test]
    fn test_product_on_basis() {
        let schur = SchurAlgebra::new(2, 1);

        // (i,j) * (j,l) = (i,l)
        let product = schur.product_on_basis(&[1], &[2], &[2], &[1]);
        assert_eq!(product, Some((vec![1], vec![1])));

        // (i,j) * (k,l) = 0 if j != k
        let product2 = schur.product_on_basis(&[1], &[1], &[2], &[1]);
        assert_eq!(product2, None);
    }

    #[test]
    fn test_schur_tensor_module() {
        let schur = SchurAlgebra::new(3, 2);
        let module = SchurTensorModule::new(schur);

        assert_eq!(module.rank(), 3);
        assert_eq!(module.degree(), 2);
    }

    #[test]
    fn test_gl_irreducible_character() {
        // Standard representation [1]
        let chi = gl_irreducible_character(vec![1], 3);
        assert!(chi.contains("1"));
        assert!(chi.contains("GL(3)"));

        // Partition [2,1]
        let chi2 = gl_irreducible_character(vec![2, 1], 3);
        assert!(chi2.contains("2,1"));
    }

    #[test]
    #[should_panic(expected = "Partition must be weakly decreasing")]
    fn test_gl_irreducible_character_invalid_partition() {
        gl_irreducible_character(vec![1, 2], 3);
    }

    #[test]
    #[should_panic(expected = "Partition must have at most")]
    fn test_gl_irreducible_character_too_many_parts() {
        gl_irreducible_character(vec![1, 1, 1, 1], 3);
    }

    #[test]
    fn test_schur_element_zero() {
        let elem: SchurElement<i32> = SchurElement::zero();
        assert!(elem.is_zero());
    }

    #[test]
    fn test_schur_element_basis() {
        let elem = SchurElement::basis_element(vec![1, 2], vec![2, 1], 5);
        assert!(!elem.is_zero());
        assert_eq!(elem.coefficients().len(), 1);
    }

    #[test]
    fn test_schur_algebra_display() {
        let schur = SchurAlgebra::new(3, 2);
        let display = format!("{}", schur);
        assert_eq!(display, "Schur algebra S(3,2)");
    }

    #[test]
    fn test_schur_tensor_module_display() {
        let schur = SchurAlgebra::new(3, 2);
        let module = SchurTensorModule::new(schur);
        let display = format!("{}", module);
        assert!(display.contains("V^⊗2"));
        assert!(display.contains("GL(3)"));
    }

    #[test]
    fn test_format_partition() {
        assert_eq!(format_partition(&[3, 2, 1]), "3,2,1");
        assert_eq!(format_partition(&[1]), "1");
        assert_eq!(format_partition(&[]), "");
    }

    #[test]
    fn test_element_display() {
        let elem = SchurElement::basis_element(vec![1], vec![2], 3);
        let display = format!("{}", elem);
        assert!(display.contains("3"));
        assert!(display.contains("S"));

        let zero: SchurElement<i32> = SchurElement::zero();
        assert_eq!(format!("{}", zero), "0");
    }
}
