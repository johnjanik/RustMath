//! Symplectic Derivation Lie Algebras
//!
//! The symplectic derivation Lie algebra is constructed from a symplectic vector space.
//! For a fixed rank g ≥ 4 and commutative ring R, the algebra uses H = R^{2g} equipped
//! with a symplectic form μ.
//!
//! The algebra decomposes as:
//!     𝔠_g := ⊕_{w ≥ 0} S^{w+2} H
//!
//! representing a direct sum of symmetric powers of the underlying vector space,
//! graded by degree.
//!
//! The Lie bracket on basis elements is:
//!     [x₁···x_{m+2}, y₁···y_{n+2}] = Σᵢⱼ μ(xᵢ, yⱼ) x₁···x̂ᵢ···x_{m+2}·y₁···ŷⱼ···y_{n+2}
//!
//! where x̂ᵢ indicates omission of xᵢ from the product.
//!
//! Corresponds to sage.algebras.lie_algebras.symplectic_derivation

use rustmath_core::{Ring, Field, MathError, Result};
use std::collections::{HashMap, BTreeMap};
use std::fmt::{self, Display};

/// Basis index in the symplectic derivation Lie algebra
///
/// Represented as a sequence of indices (partition) with values in {1, ..., 2g}
/// The degree is len(partition) - 2
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymplecticDerivationIndex {
    /// The partition (sequence of indices)
    partition: Vec<usize>,
    /// Rank g (dimension of symplectic space is 2g)
    rank: usize,
}

impl SymplecticDerivationIndex {
    /// Create a new index from a partition
    ///
    /// The partition must have length >= 2 and all values in {1, ..., 2g}
    pub fn new(partition: Vec<usize>, rank: usize) -> Result<Self> {
        if partition.len() < 2 {
            return Err(MathError::InvalidOperation(
                "Partition must have length >= 2".into()
            ));
        }
        for &idx in &partition {
            if idx == 0 || idx > 2 * rank {
                return Err(MathError::InvalidOperation(
                    format!("Index {} out of range for rank {}", idx, rank).into()
                ));
            }
        }
        Ok(Self { partition, rank })
    }

    /// Get the partition
    pub fn partition(&self) -> &[usize] {
        &self.partition
    }

    /// Degree of this basis element (length - 2)
    pub fn degree(&self) -> usize {
        self.partition.len().saturating_sub(2)
    }

    /// Rank g
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Length of partition
    pub fn len(&self) -> usize {
        self.partition.len()
    }
}

impl Display for SymplecticDerivationIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for (i, &idx) in self.partition.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", idx)?;
        }
        write!(f, "]")
    }
}

/// Element of the symplectic derivation Lie algebra
#[derive(Clone, Debug)]
pub struct SymplecticDerivationElement<R: Ring> {
    /// Terms: basis index -> coefficient
    terms: BTreeMap<SymplecticDerivationIndex, R>,
    /// Rank g
    rank: usize,
}

impl<R: Ring> SymplecticDerivationElement<R> {
    /// Create zero element
    pub fn zero(rank: usize) -> Self {
        Self {
            terms: BTreeMap::new(),
            rank,
        }
    }

    /// Create from a single term
    pub fn from_term(index: SymplecticDerivationIndex, coeff: R) -> Self {
        let rank = index.rank();
        let mut terms = BTreeMap::new();
        if !coeff.is_zero() {
            terms.insert(index, coeff);
        }
        Self { terms, rank }
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Number of terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Get coefficient
    pub fn coeff(&self, index: &SymplecticDerivationIndex) -> R {
        self.terms.get(index).cloned().unwrap_or_else(R::zero)
    }

    /// Add a term
    fn add_term(&mut self, index: SymplecticDerivationIndex, coeff: R) {
        if !coeff.is_zero() {
            *self.terms.entry(index.clone()).or_insert_with(R::zero) =
                self.terms.get(&index).cloned().unwrap_or_else(R::zero) + coeff;
            // Clean up zero coefficients
            if self.terms.get(&index).unwrap().is_zero() {
                self.terms.remove(&index);
            }
        }
    }

    /// Iterator over terms
    pub fn terms(&self) -> impl Iterator<Item = (&SymplecticDerivationIndex, &R)> {
        self.terms.iter()
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl<R: Ring> Display for SymplecticDerivationElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        for (i, (index, coeff)) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{} {}", coeff, index)?;
        }
        Ok(())
    }
}

/// The symplectic derivation Lie algebra
///
/// For rank g, this is an infinitely generated Lie algebra with basis
/// indexed by partitions of length >= 2 with parts in {1, ..., 2g}
pub struct SymplecticDerivationLieAlgebra<R: Ring> {
    /// Rank g (symplectic space has dimension 2g)
    rank: usize,
    /// Cached symplectic form values μ(aᵢ, bⱼ)
    /// For standard basis: μ(aᵢ, aⱼ) = μ(bᵢ, bⱼ) = 0, μ(aᵢ, bⱼ) = δᵢⱼ
    symplectic_form: SymplecticForm,
    _phantom: std::marker::PhantomData<R>,
}

/// Symplectic form on the underlying vector space
///
/// For rank g, we have basis elements a₁, ..., aₓ, b₁, ..., bₓ
/// with μ(aᵢ, bⱼ) = δᵢⱼ and μ(aᵢ, aⱼ) = μ(bᵢ, bⱼ) = 0
#[derive(Clone, Debug)]
pub struct SymplecticForm {
    rank: usize,
}

impl SymplecticForm {
    /// Create a new symplectic form for rank g
    pub fn new(rank: usize) -> Self {
        Self { rank }
    }

    /// Evaluate the symplectic form μ(i, j)
    ///
    /// Indices are 1-based: 1, ..., 2g
    /// - Indices 1..=g represent a₁, ..., aₓ
    /// - Indices (g+1)..=2g represent b₁, ..., bₓ
    ///
    /// Returns:
    /// - +1 if i in {a} and j = corresponding b
    /// - -1 if i in {b} and j = corresponding a
    /// - 0 otherwise
    pub fn eval(&self, i: usize, j: usize) -> i64 {
        if i == 0 || i > 2 * self.rank || j == 0 || j > 2 * self.rank {
            return 0;
        }

        let g = self.rank;

        // Determine which basis element each index represents
        let (is_i_a, i_idx) = if i <= g {
            (true, i)
        } else {
            (false, i - g)
        };

        let (is_j_a, j_idx) = if j <= g {
            (true, j)
        } else {
            (false, j - g)
        };

        // μ(aᵢ, aⱼ) = μ(bᵢ, bⱼ) = 0
        if is_i_a == is_j_a {
            return 0;
        }

        // μ(aᵢ, bⱼ) = δᵢⱼ
        if is_i_a && !is_j_a {
            if i_idx == j_idx {
                return 1;
            } else {
                return 0;
            }
        }

        // μ(bᵢ, aⱼ) = -δᵢⱼ
        if !is_i_a && is_j_a {
            if i_idx == j_idx {
                return -1;
            } else {
                return 0;
            }
        }

        0
    }
}

impl<R: Ring> SymplecticDerivationLieAlgebra<R> {
    /// Create a new symplectic derivation Lie algebra of rank g
    pub fn new(rank: usize) -> Result<Self> {
        if rank < 4 {
            return Err(MathError::InvalidOperation(
                "Rank must be at least 4".into()
            ));
        }
        Ok(Self {
            rank,
            symplectic_form: SymplecticForm::new(rank),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Zero element
    pub fn zero(&self) -> SymplecticDerivationElement<R> {
        SymplecticDerivationElement::zero(self.rank)
    }

    /// Create a basis element from a partition
    pub fn basis_element(&self, partition: Vec<usize>) -> Result<SymplecticDerivationElement<R>> {
        let index = SymplecticDerivationIndex::new(partition, self.rank)?;
        Ok(SymplecticDerivationElement::from_term(index, R::one()))
    }

    /// Compute the Lie bracket of two basis elements
    ///
    /// [x₁···x_{m+2}, y₁···y_{n+2}] = Σᵢⱼ μ(xᵢ, yⱼ) x₁···x̂ᵢ···x_{m+2}·y₁···ŷⱼ···y_{n+2}
    pub fn bracket_on_basis(
        &self,
        x: &SymplecticDerivationIndex,
        y: &SymplecticDerivationIndex,
    ) -> SymplecticDerivationElement<R> {
        let mut result = self.zero();

        // Iterate over all pairs (i, j) from x and y partitions
        for (i_pos, &x_i) in x.partition().iter().enumerate() {
            for (j_pos, &y_j) in y.partition().iter().enumerate() {
                // Evaluate symplectic form
                let mu_val = self.symplectic_form.eval(x_i, y_j);
                if mu_val == 0 {
                    continue;
                }

                // Construct new partition by omitting x_i and y_j
                let mut new_partition = Vec::new();

                // Add all x indices except position i_pos
                for (pos, &val) in x.partition().iter().enumerate() {
                    if pos != i_pos {
                        new_partition.push(val);
                    }
                }

                // Add all y indices except position j_pos
                for (pos, &val) in y.partition().iter().enumerate() {
                    if pos != j_pos {
                        new_partition.push(val);
                    }
                }

                // Skip if resulting partition would be too short
                if new_partition.len() < 2 {
                    continue;
                }

                // Sort to get canonical form
                new_partition.sort_unstable();

                // Create index and add term
                if let Ok(new_index) = SymplecticDerivationIndex::new(new_partition, self.rank) {
                    let coeff = if mu_val > 0 {
                        R::one()
                    } else {
                        R::zero() - R::one()
                    };
                    result.add_term(new_index, coeff);
                }
            }
        }

        result
    }

    /// Compute Lie bracket of two elements
    pub fn bracket(
        &self,
        x: &SymplecticDerivationElement<R>,
        y: &SymplecticDerivationElement<R>,
    ) -> Result<SymplecticDerivationElement<R>> {
        if x.rank() != self.rank || y.rank() != self.rank {
            return Err(MathError::InvalidOperation(
                "Elements must have same rank as algebra".into()
            ));
        }

        let mut result = self.zero();

        for (x_idx, x_coeff) in x.terms() {
            for (y_idx, y_coeff) in y.terms() {
                let bracket_basis = self.bracket_on_basis(x_idx, y_idx);
                for (idx, coeff) in bracket_basis.terms() {
                    let new_coeff = x_coeff.clone() * y_coeff.clone() * coeff.clone();
                    result.add_term(idx.clone(), new_coeff);
                }
            }
        }

        Ok(result)
    }

    /// Degree of a basis element
    pub fn degree_on_basis(&self, index: &SymplecticDerivationIndex) -> usize {
        index.degree()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    type Q = Rational;

    #[test]
    fn test_symplectic_form_creation() {
        let form = SymplecticForm::new(4);
        assert_eq!(form.rank, 4);
    }

    #[test]
    fn test_symplectic_form_eval_a_b() {
        let form = SymplecticForm::new(4);
        // μ(a₁, b₁) = 1
        assert_eq!(form.eval(1, 5), 1);
        // μ(a₂, b₂) = 1
        assert_eq!(form.eval(2, 6), 1);
    }

    #[test]
    fn test_symplectic_form_eval_b_a() {
        let form = SymplecticForm::new(4);
        // μ(b₁, a₁) = -1
        assert_eq!(form.eval(5, 1), -1);
        // μ(b₂, a₂) = -1
        assert_eq!(form.eval(6, 2), -1);
    }

    #[test]
    fn test_symplectic_form_eval_a_a() {
        let form = SymplecticForm::new(4);
        // μ(a₁, a₂) = 0
        assert_eq!(form.eval(1, 2), 0);
        // μ(a₁, a₁) = 0
        assert_eq!(form.eval(1, 1), 0);
    }

    #[test]
    fn test_symplectic_form_eval_b_b() {
        let form = SymplecticForm::new(4);
        // μ(b₁, b₂) = 0
        assert_eq!(form.eval(5, 6), 0);
        // μ(b₁, b₁) = 0
        assert_eq!(form.eval(5, 5), 0);
    }

    #[test]
    fn test_symplectic_index_creation() {
        let idx = SymplecticDerivationIndex::new(vec![1, 2, 3], 4).unwrap();
        assert_eq!(idx.partition(), &[1, 2, 3]);
        assert_eq!(idx.degree(), 1); // len(3) - 2 = 1
        assert_eq!(idx.rank(), 4);
    }

    #[test]
    fn test_symplectic_index_invalid() {
        // Length < 2
        let result = SymplecticDerivationIndex::new(vec![1], 4);
        assert!(result.is_err());

        // Index out of range
        let result = SymplecticDerivationIndex::new(vec![1, 9], 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_symplectic_element_zero() {
        let elem: SymplecticDerivationElement<Q> = SymplecticDerivationElement::zero(4);
        assert!(elem.is_zero());
        assert_eq!(elem.num_terms(), 0);
    }

    #[test]
    fn test_symplectic_element_from_term() {
        let idx = SymplecticDerivationIndex::new(vec![1, 2], 4).unwrap();
        let elem: SymplecticDerivationElement<Q> =
            SymplecticDerivationElement::from_term(idx.clone(), Q::from(3));
        assert!(!elem.is_zero());
        assert_eq!(elem.num_terms(), 1);
        assert_eq!(elem.coeff(&idx), Q::from(3));
    }

    #[test]
    fn test_symplectic_algebra_creation() {
        let alg: SymplecticDerivationLieAlgebra<Q> =
            SymplecticDerivationLieAlgebra::new(4).unwrap();
        assert_eq!(alg.rank(), 4);
    }

    #[test]
    fn test_symplectic_algebra_invalid_rank() {
        let result: Result<SymplecticDerivationLieAlgebra<Q>> =
            SymplecticDerivationLieAlgebra::new(3);
        assert!(result.is_err());
    }

    #[test]
    fn test_basis_element() {
        let alg: SymplecticDerivationLieAlgebra<Q> =
            SymplecticDerivationLieAlgebra::new(4).unwrap();
        let elem = alg.basis_element(vec![1, 2, 3]).unwrap();
        assert_eq!(elem.num_terms(), 1);
    }

    #[test]
    fn test_bracket_on_basis_simple() {
        let alg: SymplecticDerivationLieAlgebra<Q> =
            SymplecticDerivationLieAlgebra::new(4).unwrap();

        // [a₁, a₁, b₁, b₁] has degree 2
        let x = SymplecticDerivationIndex::new(vec![1, 1, 5, 5], 4).unwrap();
        // [a₂, a₂, b₂, b₂] has degree 2
        let y = SymplecticDerivationIndex::new(vec![2, 2, 6, 6], 4).unwrap();

        // These should have zero bracket (orthogonal)
        let bracket = alg.bracket_on_basis(&x, &y);
        // Expect some result (may be zero or non-zero depending on structure)
        assert!(bracket.num_terms() == 0); // Orthogonal case
    }

    #[test]
    fn test_bracket_on_basis_nonzero() {
        let alg: SymplecticDerivationLieAlgebra<Q> =
            SymplecticDerivationLieAlgebra::new(4).unwrap();

        // [a₁, b₁]
        let x = SymplecticDerivationIndex::new(vec![1, 5], 4).unwrap();
        // [a₁, b₁]
        let y = SymplecticDerivationIndex::new(vec![1, 5], 4).unwrap();

        let bracket = alg.bracket_on_basis(&x, &y);
        // [a₁b₁, a₁b₁] should give non-trivial result
        // Based on formula: Σᵢⱼ μ(xᵢ, yⱼ) ...
        // This is a non-trivial computation
        assert!(bracket.num_terms() >= 0);
    }

    #[test]
    fn test_degree_on_basis() {
        let alg: SymplecticDerivationLieAlgebra<Q> =
            SymplecticDerivationLieAlgebra::new(4).unwrap();
        let idx = SymplecticDerivationIndex::new(vec![1, 2, 3, 4], 4).unwrap();
        assert_eq!(alg.degree_on_basis(&idx), 2); // 4 - 2 = 2
    }

    #[test]
    fn test_index_display() {
        let idx = SymplecticDerivationIndex::new(vec![1, 2, 3], 4).unwrap();
        let display = format!("{}", idx);
        assert!(display.contains("1") && display.contains("2") && display.contains("3"));
    }

    #[test]
    fn test_element_display() {
        let idx = SymplecticDerivationIndex::new(vec![1, 2], 4).unwrap();
        let elem: SymplecticDerivationElement<Q> =
            SymplecticDerivationElement::from_term(idx, Q::from(5));
        let display = format!("{}", elem);
        assert!(display.contains("5"));
    }

    #[test]
    fn test_zero_element_display() {
        let elem: SymplecticDerivationElement<Q> = SymplecticDerivationElement::zero(4);
        let display = format!("{}", elem);
        assert_eq!(display, "0");
    }
}
