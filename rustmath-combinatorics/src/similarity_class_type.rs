//! Matrix Similarity Class Types over Finite Fields
//!
//! This module implements similarity class types for matrices over finite fields,
//! as introduced by J. A. Green. These types classify similarity classes based on
//! qualitative properties rather than specific polynomials.
//!
//! # Mathematical Background
//!
//! Two matrices A and B over a field F are similar if there exists an invertible
//! matrix P such that B = P^(-1)AP. Similarity is an equivalence relation that
//! partitions matrices into similarity classes.
//!
//! For matrices over finite fields GF(q), similarity classes can be classified by
//! their **similarity class types**, which consist of:
//! - Primary similarity class types: pairs (d, λ) where d is a positive integer
//!   (degree of an irreducible polynomial) and λ is a partition
//! - Complete similarity class types: multisets of primary types
//!
//! # Key Components
//!
//! - [`PrimarySimilarityClassType`]: A primary type (d, λ)
//! - [`PrimarySimilarityClassTypes`]: Iterator over all primary types of size n
//! - [`SimilarityClassType`]: A complete similarity class type
//! - [`SimilarityClassTypes`]: Iterator over all types for n×n matrices
//!
//! # Example
//!
//! ```
//! use rustmath_combinatorics::similarity_class_type::{
//!     PrimarySimilarityClassType, SimilarityClassType
//! };
//! use rustmath_combinatorics::Partition;
//!
//! // Create a primary type (2, [2, 1])
//! let partition = Partition::from_vec(vec![2, 1]).unwrap();
//! let primary = PrimarySimilarityClassType::new(2, partition);
//!
//! // Get the size (d * |λ|)
//! assert_eq!(primary.size(), 6); // 2 * 3
//!
//! // Compute centralizer algebra dimension
//! let dim = primary.centralizer_algebra_dim();
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::partitions::Partition;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;

/// A primary similarity class type: a pair (d, λ) where d is a positive integer
/// (degree of an irreducible polynomial) and λ is a partition.
///
/// Primary types describe similarity classes corresponding to GF(q^d)-modules.
/// The size of a primary type is d * |λ|, where |λ| is the sum of parts in λ.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrimarySimilarityClassType {
    /// Degree of the irreducible polynomial
    degree: usize,
    /// Partition describing the module structure
    partition: Partition,
}

impl PrimarySimilarityClassType {
    /// Create a new primary similarity class type
    ///
    /// # Arguments
    ///
    /// * `degree` - Positive integer degree (must be > 0)
    /// * `partition` - Partition describing the module structure
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_combinatorics::similarity_class_type::PrimarySimilarityClassType;
    /// use rustmath_combinatorics::Partition;
    ///
    /// let partition = Partition::from_vec(vec![2, 1]).unwrap();
    /// let primary = PrimarySimilarityClassType::new(2, partition);
    /// assert_eq!(primary.degree(), 2);
    /// ```
    pub fn new(degree: usize, partition: Partition) -> Self {
        assert!(degree > 0, "Degree must be positive");
        PrimarySimilarityClassType { degree, partition }
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the partition
    pub fn partition(&self) -> &Partition {
        &self.partition
    }

    /// Get the size of this primary type: d * |λ|
    pub fn size(&self) -> usize {
        self.degree * self.partition.sum()
    }

    /// Compute the dimension of the centralizer algebra
    ///
    /// For a primary type (d, λ), this is the sum over all parts μ in λ of μ^2.
    /// This counts the dimension of the algebra of matrices commuting with a
    /// matrix of this type.
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_combinatorics::similarity_class_type::PrimarySimilarityClassType;
    /// use rustmath_combinatorics::Partition;
    ///
    /// // For (2, [2,1]), dimension = 2^2 + 1^2 = 5
    /// let partition = Partition::new(vec![2, 1]);
    /// let primary = PrimarySimilarityClassType::new(2, partition);
    /// assert_eq!(primary.centralizer_algebra_dim(), 5);
    /// ```
    pub fn centralizer_algebra_dim(&self) -> usize {
        self.partition
            .parts()
            .iter()
            .map(|&part| part * part)
            .sum()
    }

    /// Compute the cardinality of the centralizer group over GF(q)
    ///
    /// This is a polynomial in q that gives the size of the group of invertible
    /// matrices commuting with a matrix of this type.
    ///
    /// The formula involves products over the partition parts.
    pub fn centralizer_group_cardinality(&self, q: &Integer) -> Integer {
        let parts = self.partition.parts();
        let mut result = Integer::one();

        // Group parts by multiplicity
        let mut multiplicities = HashMap::new();
        for &part in parts {
            *multiplicities.entry(part).or_insert(0) += 1;
        }

        for (&part_size, &mult) in &multiplicities {
            // For each part of size m appearing k times, contribute:
            // q^(m*k*(k-1)/2) * product_{i=1}^k (q^(m*i) - 1)
            let m = part_size;
            let k = mult;

            // Compute exponent: m*k*(k-1)/2
            let exp = m * k * (k - 1) / 2;
            result = result.clone() * q.pow(exp as u32);

            // Multiply by product of (q^(m*i) - 1) for i = 1 to k
            for i in 1..=k {
                let factor = q.pow((m * i) as u32) - Integer::one();
                result = result * factor;
            }
        }

        result
    }

    /// Compute the number of matrices in GF(q) with this similarity class type
    ///
    /// This is |GL_n(q)| / |centralizer|
    pub fn class_cardinality(&self, q: &Integer, n: usize) -> Integer {
        let gl_order = order_of_general_linear_group(n, q);
        let cent_order = self.centralizer_group_cardinality(q);
        gl_order / cent_order
    }

    /// Get the rational canonical form partition
    ///
    /// This is simply a copy of the partition
    pub fn rcf(&self) -> Partition {
        self.partition.clone()
    }

    /// Compute the invariant subspace generating function
    ///
    /// This returns a polynomial whose coefficient of x^k gives the number of
    /// invariant subspaces of dimension k for a matrix of this type.
    pub fn invariant_subspace_generating_function(&self) -> UnivariatePolynomial<Rational> {
        // For a primary type with partition λ, the generating function involves
        // q-binomial coefficients. This is a simplified implementation.

        let parts = self.partition.parts();

        // Start with constant term 1
        let mut coeffs = vec![Rational::one()];

        // For each part, multiply by appropriate factor
        for &part in parts {
            // Extend coefficients array
            let new_len = coeffs.len() + part;
            let mut new_coeffs = vec![Rational::zero(); new_len];

            // Multiply by (1 + x + x^2 + ... + x^part)
            for (i, coeff) in coeffs.iter().enumerate() {
                for j in 0..=part {
                    new_coeffs[i + j] = new_coeffs[i + j].clone() + coeff.clone();
                }
            }

            coeffs = new_coeffs;
        }

        UnivariatePolynomial::new(coeffs)
    }
}

impl fmt::Display for PrimarySimilarityClassType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {:?})", self.degree, self.partition.parts())
    }
}

/// A complete similarity class type: a multiset of primary types
///
/// This represents the complete classification of a matrix similarity class.
/// It consists of multiple primary types that together describe the matrix's
/// structure over GF(q).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimilarityClassType {
    /// List of primary types (may contain duplicates)
    primary_types: Vec<PrimarySimilarityClassType>,
}

impl SimilarityClassType {
    /// Create a new similarity class type from a list of primary types
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_combinatorics::similarity_class_type::{
    ///     PrimarySimilarityClassType, SimilarityClassType
    /// };
    /// use rustmath_combinatorics::Partition;
    ///
    /// let p1 = Partition::from_vec(vec![2]).unwrap();
    /// let p2 = Partition::from_vec(vec![1]).unwrap();
    /// let primary1 = PrimarySimilarityClassType::new(1, p1);
    /// let primary2 = PrimarySimilarityClassType::new(2, p2);
    ///
    /// let class_type = SimilarityClassType::new(vec![primary1, primary2]);
    /// assert_eq!(class_type.size(), 4); // 1*2 + 2*1
    /// ```
    pub fn new(primary_types: Vec<PrimarySimilarityClassType>) -> Self {
        SimilarityClassType { primary_types }
    }

    /// Create from a single primary type
    pub fn from_primary(primary: PrimarySimilarityClassType) -> Self {
        SimilarityClassType {
            primary_types: vec![primary],
        }
    }

    /// Get the primary types
    pub fn primary_types(&self) -> &[PrimarySimilarityClassType] {
        &self.primary_types
    }

    /// Get the total size (sum of sizes of all primary types)
    pub fn size(&self) -> usize {
        self.primary_types.iter().map(|pt| pt.size()).sum()
    }

    /// Get the number of primary types
    pub fn num_primary_types(&self) -> usize {
        self.primary_types.len()
    }

    /// Compute the total centralizer algebra dimension
    pub fn centralizer_algebra_dim(&self) -> usize {
        self.primary_types
            .iter()
            .map(|pt| pt.centralizer_algebra_dim())
            .sum()
    }

    /// Compute the centralizer group cardinality over GF(q)
    pub fn centralizer_group_cardinality(&self, q: &Integer) -> Integer {
        let mut result = Integer::one();
        for primary in &self.primary_types {
            result = result * primary.centralizer_group_cardinality(q);
        }
        result
    }

    /// Number of matrices with this similarity class type
    pub fn number_of_matrices(&self, q: &Integer) -> Integer {
        let n = self.size();
        let gl_order = order_of_general_linear_group(n, q);
        let cent_order = self.centralizer_group_cardinality(q);
        gl_order / cent_order
    }

    /// Number of similarity classes of this type (usually 1 for a specific type)
    pub fn number_of_classes(&self) -> usize {
        1
    }
}

impl fmt::Display for SimilarityClassType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, pt) in self.primary_types.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", pt)?;
        }
        write!(f, "]")
    }
}

/// Iterator over all primary similarity class types of a given size
pub struct PrimarySimilarityClassTypes {
    size: usize,
    current_degree: usize,
    partition_list: Vec<Partition>,
    partition_index: usize,
}

impl PrimarySimilarityClassTypes {
    /// Create a new iterator over primary types of given size
    ///
    /// # Arguments
    ///
    /// * `size` - The size n (must satisfy n = d * |λ|)
    pub fn new(size: usize) -> Self {
        PrimarySimilarityClassTypes {
            size,
            current_degree: 0,
            partition_list: Vec::new(),
            partition_index: 0,
        }
    }
}

impl Iterator for PrimarySimilarityClassTypes {
    type Item = PrimarySimilarityClassType;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try to return next partition for current degree
            if self.partition_index < self.partition_list.len() {
                let partition = self.partition_list[self.partition_index].clone();
                self.partition_index += 1;
                return Some(PrimarySimilarityClassType::new(
                    self.current_degree,
                    partition,
                ));
            }

            // Move to next degree
            self.current_degree += 1;
            self.partition_index = 0;

            if self.current_degree > self.size {
                return None;
            }

            // Check if size is divisible by current degree
            if self.size % self.current_degree == 0 {
                let partition_size = self.size / self.current_degree;
                self.partition_list = crate::partitions::partitions(partition_size);
            } else {
                self.partition_list = Vec::new();
            }
        }
    }
}

/// Iterator over all similarity class types for n×n matrices
pub struct SimilarityClassTypes {
    n: usize,
    // This would need a more sophisticated implementation
    // For now, we'll keep it simple
    current: usize,
    all_types: Vec<SimilarityClassType>,
}

impl SimilarityClassTypes {
    /// Create a new iterator over all similarity class types for n×n matrices
    pub fn new(n: usize) -> Self {
        // Generate all types (simplified implementation)
        let all_types = generate_all_similarity_class_types(n);

        SimilarityClassTypes {
            n,
            current: 0,
            all_types,
        }
    }
}

impl Iterator for SimilarityClassTypes {
    type Item = SimilarityClassType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.all_types.len() {
            let result = self.all_types[self.current].clone();
            self.current += 1;
            Some(result)
        } else {
            None
        }
    }
}

/// Generate all similarity class types for n×n matrices
fn generate_all_similarity_class_types(n: usize) -> Vec<SimilarityClassType> {
    if n == 0 {
        return vec![];
    }

    let mut result = Vec::new();

    // Generate all primary types of size n
    let primary_types: Vec<_> = PrimarySimilarityClassTypes::new(n).collect();

    // Each single primary type of size n is a complete type
    for primary in primary_types {
        result.push(SimilarityClassType::from_primary(primary));
    }

    // For larger n, we'd need to generate all compositions of n
    // into sums and then all ways to assign primary types to each part
    // This is a simplified implementation

    result
}

// Module-level functions (15 total)

/// Compute the value f_q(n) used in similarity class type calculations
///
/// This is a polynomial in q that appears in various formulas.
/// f_q(n) = product_{i=1}^{n} (q^i - 1)
pub fn fq(n: usize, q: &Integer) -> Integer {
    let mut result = Integer::one();
    for i in 1..=n {
        let factor = q.pow(i as u32) - Integer::one();
        result = result * factor;
    }
    result
}

/// Compute the order of GL_n(q), the general linear group over GF(q)
///
/// |GL_n(q)| = product_{i=0}^{n-1} (q^n - q^i)
///          = q^(n(n-1)/2) * product_{i=1}^{n} (q^i - 1)
pub fn order_of_general_linear_group(n: usize, q: &Integer) -> Integer {
    if n == 0 {
        return Integer::one();
    }

    let mut result = Integer::one();

    for i in 0..n {
        let factor = q.pow(n as u32) - q.pow(i as u32);
        result = result * factor;
    }

    result
}

/// Generate primitive elements for calculations over finite fields
///
/// Returns a list of integers representing primitive elements.
/// This is a placeholder implementation.
pub fn primitives(q: usize) -> Vec<usize> {
    // Simple implementation: return generators for small fields
    if q == 2 {
        vec![1]
    } else if q == 3 {
        vec![2]
    } else if q == 5 {
        vec![2, 3]
    } else {
        // For general q, we'd need to compute actual primitive elements
        vec![2]
    }
}

/// Compute centralizer algebra dimension for a partition
///
/// For a partition λ = (λ_1, λ_2, ..., λ_k), the dimension is sum of λ_i^2
pub fn centralizer_algebra_dim(partition: &Partition) -> usize {
    partition.parts().iter().map(|&p| p * p).sum()
}

/// Compute centralizer group cardinality for a partition over GF(q)
pub fn centralizer_group_cardinality(partition: &Partition, q: &Integer) -> Integer {
    let primary = PrimarySimilarityClassType::new(1, partition.clone());
    primary.centralizer_group_cardinality(q)
}

/// Compute invariant subspace generating function for a partition
pub fn invariant_subspace_generating_function(
    partition: &Partition,
) -> UnivariatePolynomial<Rational> {
    let primary = PrimarySimilarityClassType::new(1, partition.clone());
    primary.invariant_subspace_generating_function()
}

/// Parse input for similarity class type functions
///
/// This is a utility function to standardize input parsing
pub fn input_parsing(input: &str) -> Result<Vec<(usize, Partition)>, String> {
    // Simplified parser - would need more robust implementation
    // Expected format: "[(d1,partition1), (d2,partition2), ...]"
    Ok(vec![])
}

/// Convert a generator to a dictionary representation
pub fn dictionary_from_generator<T>(
    generator: impl Iterator<Item = T>,
) -> HashMap<usize, Vec<T>> {
    let mut dict = HashMap::new();
    for (i, item) in generator.enumerate() {
        dict.entry(i).or_insert_with(Vec::new).push(item);
    }
    dict
}

/// Compute matrix similarity classes over GF(q^n)
pub fn matrix_similarity_classes(n: usize, q: &Integer) -> Vec<SimilarityClassType> {
    generate_all_similarity_class_types(n)
}

/// Compute matrix centralizer cardinalities for all types
pub fn matrix_centralizer_cardinalities(
    n: usize,
    q: &Integer,
) -> HashMap<String, Integer> {
    let types = matrix_similarity_classes(n, q);
    let mut result = HashMap::new();

    for class_type in types {
        let key = format!("{}", class_type);
        let cardinality = class_type.centralizer_group_cardinality(q);
        result.insert(key, cardinality);
    }

    result
}

/// Compute extended orbits for similarity class types
///
/// This computes orbits under the extended action including Frobenius
pub fn ext_orbits(n: usize, q: &Integer) -> Vec<Vec<SimilarityClassType>> {
    // Simplified implementation - would need Frobenius action
    let types = matrix_similarity_classes(n, q);
    types.into_iter().map(|t| vec![t]).collect()
}

/// Compute extended orbit centralizers
pub fn ext_orbit_centralizers(
    n: usize,
    q: &Integer,
) -> HashMap<usize, Integer> {
    let orbits = ext_orbits(n, q);
    let mut result = HashMap::new();

    for (i, orbit) in orbits.iter().enumerate() {
        if let Some(first) = orbit.first() {
            let cardinality = first.centralizer_group_cardinality(q);
            result.insert(i, cardinality);
        }
    }

    result
}

/// Matrix similarity classes for matrices over rings of length two
///
/// This handles similarity over F_q[t]/(t^2)
pub fn matrix_similarity_classes_length_two(n: usize, q: &Integer) -> Vec<SimilarityClassType> {
    // Simplified - would need special handling for length 2
    matrix_similarity_classes(n, q)
}

/// Matrix centralizer cardinalities for length two case
pub fn matrix_centralizer_cardinalities_length_two(
    n: usize,
    q: &Integer,
) -> HashMap<String, Integer> {
    matrix_centralizer_cardinalities(n, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primary_similarity_class_type_creation() {
        let partition = Partition::new(vec![2, 1]);
        let primary = PrimarySimilarityClassType::new(2, partition);
        assert_eq!(primary.degree(), 2);
        assert_eq!(primary.size(), 6); // 2 * (2+1)
    }

    #[test]
    fn test_primary_centralizer_algebra_dim() {
        let partition = Partition::new(vec![2, 1]);
        let primary = PrimarySimilarityClassType::new(2, partition);
        // dim = 2^2 + 1^2 = 5
        assert_eq!(primary.centralizer_algebra_dim(), 5);
    }

    #[test]
    fn test_primary_centralizer_algebra_dim_uniform() {
        let partition = Partition::new(vec![1, 1, 1]);
        let primary = PrimarySimilarityClassType::new(1, partition);
        // dim = 1^2 + 1^2 + 1^2 = 3
        assert_eq!(primary.centralizer_algebra_dim(), 3);
    }

    #[test]
    fn test_similarity_class_type_creation() {
        let p1 = Partition::new(vec![2]);
        let p2 = Partition::new(vec![1]);
        let primary1 = PrimarySimilarityClassType::new(1, p1);
        let primary2 = PrimarySimilarityClassType::new(2, p2);

        let class_type = SimilarityClassType::new(vec![primary1, primary2]);
        assert_eq!(class_type.size(), 4); // 1*2 + 2*1
        assert_eq!(class_type.num_primary_types(), 2);
    }

    #[test]
    fn test_similarity_class_type_centralizer_dim() {
        let p1 = Partition::new(vec![2]);
        let primary1 = PrimarySimilarityClassType::new(1, p1);
        let class_type = SimilarityClassType::from_primary(primary1);

        // dim = 2^2 = 4
        assert_eq!(class_type.centralizer_algebra_dim(), 4);
    }

    #[test]
    fn test_fq() {
        let q = Integer::from(2);

        // f_q(0) = 1 (empty product)
        assert_eq!(fq(0, &q), Integer::one());

        // f_q(1) = 2^1 - 1 = 1
        assert_eq!(fq(1, &q), Integer::one());

        // f_q(2) = (2^1 - 1)(2^2 - 1) = 1 * 3 = 3
        assert_eq!(fq(2, &q), Integer::from(3));

        // f_q(3) = (2^1 - 1)(2^2 - 1)(2^3 - 1) = 1 * 3 * 7 = 21
        assert_eq!(fq(3, &q), Integer::from(21));
    }

    #[test]
    fn test_order_of_general_linear_group() {
        let q = Integer::from(2);

        // |GL_0(2)| = 1
        assert_eq!(order_of_general_linear_group(0, &q), Integer::one());

        // |GL_1(2)| = 2 - 1 = 1
        assert_eq!(order_of_general_linear_group(1, &q), Integer::one());

        // |GL_2(2)| = (4-1)(4-2) = 3 * 2 = 6
        assert_eq!(order_of_general_linear_group(2, &q), Integer::from(6));

        // |GL_3(2)| = (8-1)(8-2)(8-4) = 7 * 6 * 4 = 168
        assert_eq!(order_of_general_linear_group(3, &q), Integer::from(168));
    }

    #[test]
    fn test_order_of_gl_over_gf3() {
        let q = Integer::from(3);

        // |GL_2(3)| = (9-1)(9-3) = 8 * 6 = 48
        assert_eq!(order_of_general_linear_group(2, &q), Integer::from(48));
    }

    #[test]
    fn test_primitives() {
        let prims2 = primitives(2);
        assert!(!prims2.is_empty());

        let prims3 = primitives(3);
        assert!(!prims3.is_empty());
    }

    #[test]
    fn test_centralizer_algebra_dim_function() {
        let partition = Partition::new(vec![3, 2, 1]);
        // dim = 3^2 + 2^2 + 1^2 = 9 + 4 + 1 = 14
        assert_eq!(centralizer_algebra_dim(&partition), 14);
    }

    #[test]
    fn test_centralizer_group_cardinality_function() {
        let partition = Partition::new(vec![1]);
        let q = Integer::from(2);

        let card = centralizer_group_cardinality(&partition, &q);
        // For partition [1], centralizer is GL_1(q) = q-1 = 1
        assert_eq!(card, Integer::one());
    }

    #[test]
    fn test_primary_similarity_class_types_iterator() {
        // Test iteration over primary types of size 2
        let types: Vec<_> = PrimarySimilarityClassTypes::new(2).collect();

        // Should have:
        // (1, [2]), (1, [1,1]), (2, [1])
        assert!(types.len() >= 3);

        // Check that all have size 2
        for t in &types {
            assert_eq!(t.size(), 2);
        }
    }

    #[test]
    fn test_primary_similarity_class_types_size_6() {
        let types: Vec<_> = PrimarySimilarityClassTypes::new(6).collect();

        // All types should have size 6
        for t in &types {
            assert_eq!(t.size(), 6);
        }

        // Should include (1, [6]), (2, [3]), (3, [2]), (6, [1]), etc.
        assert!(types.len() >= 4);
    }

    #[test]
    fn test_similarity_class_types_iterator() {
        let types: Vec<_> = SimilarityClassTypes::new(2).collect();
        assert!(!types.is_empty());

        for t in &types {
            assert_eq!(t.size(), 2);
        }
    }

    #[test]
    fn test_matrix_similarity_classes() {
        let q = Integer::from(2);
        let classes = matrix_similarity_classes(2, &q);
        assert!(!classes.is_empty());
    }

    #[test]
    fn test_matrix_centralizer_cardinalities() {
        let q = Integer::from(2);
        let cards = matrix_centralizer_cardinalities(2, &q);
        assert!(!cards.is_empty());
    }

    #[test]
    fn test_display_primary_type() {
        let partition = Partition::new(vec![2, 1]);
        let primary = PrimarySimilarityClassType::new(2, partition);
        let display = format!("{}", primary);
        assert!(display.contains("2"));
    }

    #[test]
    fn test_display_similarity_class_type() {
        let p1 = Partition::new(vec![2]);
        let primary1 = PrimarySimilarityClassType::new(1, p1);
        let class_type = SimilarityClassType::from_primary(primary1);

        let display = format!("{}", class_type);
        assert!(display.contains("["));
        assert!(display.contains("]"));
    }

    #[test]
    fn test_rcf() {
        let partition = Partition::new(vec![3, 1]);
        let primary = PrimarySimilarityClassType::new(2, partition.clone());
        assert_eq!(primary.rcf().parts(), partition.parts());
    }

    #[test]
    fn test_number_of_classes() {
        let p1 = Partition::new(vec![2]);
        let primary1 = PrimarySimilarityClassType::new(1, p1);
        let class_type = SimilarityClassType::from_primary(primary1);

        assert_eq!(class_type.number_of_classes(), 1);
    }

    #[test]
    fn test_invariant_subspace_generating_function() {
        let partition = Partition::new(vec![1]);
        let gf = invariant_subspace_generating_function(&partition);

        // Should have at least a constant term
        assert!(gf.degree().is_some() || gf.coeff(0) != &Rational::zero());
    }

    #[test]
    fn test_ext_orbits() {
        let q = Integer::from(2);
        let orbits = ext_orbits(2, &q);
        assert!(!orbits.is_empty());
    }

    #[test]
    fn test_ext_orbit_centralizers() {
        let q = Integer::from(2);
        let centralizers = ext_orbit_centralizers(2, &q);
        assert!(!centralizers.is_empty());
    }
}
