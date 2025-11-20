//! Weight Lattices and Weight Spaces
//!
//! The weight lattice is the dual lattice to the root lattice. Weights are
//! fundamental in representation theory - each finite-dimensional representation
//! has a highest weight that determines it completely.
//!
//! For a root system of rank n, the weight lattice has a basis of fundamental weights
//! ω_1, ..., ω_n dual to the simple roots α_1, ..., α_n.
//!
//! Corresponds to sage.combinat.root_system.weight_lattice_realizations

use crate::cartan_type::{CartanType, CartanLetter};
use crate::cartan_matrix::CartanMatrix;
use rustmath_core::Ring;
use rustmath_rationals::Rational;
use rustmath_integers::Integer;
use std::fmt::{self, Display};
use std::ops::{Add, Mul, Neg};

/// A weight in the weight lattice
///
/// Weights are represented in the basis of fundamental weights.
/// A weight λ = c_1ω_1 + ... + c_nω_n where ω_i are fundamental weights.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Weight {
    /// Coefficients in the fundamental weight basis
    pub coefficients: Vec<Integer>,
    /// The rank
    pub rank: usize,
}

impl Weight {
    /// Create a new weight from coefficients
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_liealgebras::weight_lattice::Weight;
    /// use rustmath_integers::Integer;
    ///
    /// let w = Weight::new(vec![Integer::from(1), Integer::from(2)]);
    /// assert_eq!(w.rank(), 2);
    /// ```
    pub fn new(coefficients: Vec<Integer>) -> Self {
        let rank = coefficients.len();
        Weight { coefficients, rank }
    }

    /// Create the zero weight
    pub fn zero(rank: usize) -> Self {
        Weight {
            coefficients: vec![Integer::zero(); rank],
            rank,
        }
    }

    /// Create a fundamental weight ω_i (1-indexed)
    pub fn fundamental_weight(i: usize, rank: usize) -> Self {
        assert!(i > 0 && i <= rank, "Fundamental weight index out of bounds");
        let mut coefficients = vec![Integer::zero(); rank];
        coefficients[i - 1] = Integer::one();
        Weight { coefficients, rank }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Check if this is the zero weight
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c.is_zero())
    }

    /// Get coefficient for fundamental weight ω_i (1-indexed)
    pub fn coefficient(&self, i: usize) -> &Integer {
        assert!(i > 0 && i <= self.rank, "Weight coefficient index out of bounds");
        &self.coefficients[i - 1]
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &Integer) -> Self {
        Weight {
            coefficients: self.coefficients.iter().map(|c| c * scalar).collect(),
            rank: self.rank,
        }
    }

    /// Compute the inner product with another weight using the weight lattice metric
    pub fn inner_product(&self, other: &Weight, cartan_matrix: &CartanMatrix) -> Rational {
        assert_eq!(self.rank, other.rank);
        assert_eq!(self.rank, cartan_matrix.rank());

        let mut result = Rational::zero();

        // The inner product in weight space uses the inverse Cartan matrix
        // For simplicity, we compute it directly for small ranks
        for i in 0..self.rank {
            for j in 0..self.rank {
                let coeff_i = Rational::from_integer(self.coefficients[i].clone());
                let coeff_j = Rational::from_integer(other.coefficients[j].clone());

                // This is simplified - proper implementation would use inverse Cartan matrix
                if i == j {
                    result = result + coeff_i * coeff_j;
                }
            }
        }

        result
    }

    /// Check if this weight is dominant
    ///
    /// A weight is dominant if all its coefficients (in the fundamental weight basis)
    /// are non-negative.
    pub fn is_dominant(&self) -> bool {
        self.coefficients.iter().all(|c| *c >= Integer::zero())
    }

    /// Get the level of this weight (sum of coefficients)
    ///
    /// Useful for affine Lie algebras
    pub fn level(&self) -> Integer {
        self.coefficients.iter().fold(Integer::zero(), |acc, c| acc + c)
    }
}

impl Add for Weight {
    type Output = Weight;

    fn add(self, other: Weight) -> Weight {
        assert_eq!(self.rank, other.rank);
        let coefficients = self
            .coefficients
            .iter()
            .zip(&other.coefficients)
            .map(|(a, b)| a + b)
            .collect();
        Weight {
            coefficients,
            rank: self.rank,
        }
    }
}

impl Neg for Weight {
    type Output = Weight;

    fn neg(self) -> Weight {
        Weight {
            coefficients: self.coefficients.iter().map(|c| -c).collect(),
            rank: self.rank,
        }
    }
}

impl Display for Weight {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if !coeff.is_zero() {
                if !first && *coeff > Integer::zero() {
                    write!(f, " + ")?;
                } else if *coeff < Integer::zero() {
                    write!(f, " - ")?;
                }

                let abs_coeff = if *coeff < Integer::zero() {
                    -coeff
                } else {
                    coeff.clone()
                };

                if abs_coeff == Integer::one() {
                    write!(f, "ω_{}", i + 1)?;
                } else {
                    write!(f, "{}ω_{}", abs_coeff, i + 1)?;
                }
                first = false;
            }
        }

        if first {
            write!(f, "0")?;
        }

        Ok(())
    }
}

/// The weight lattice for a root system
///
/// Contains methods for working with weights, fundamental weights, and the Weyl group action.
#[derive(Clone, Debug)]
pub struct WeightLattice {
    /// The Cartan type
    pub cartan_type: CartanType,
    /// The Cartan matrix
    pub cartan_matrix: CartanMatrix,
    /// The fundamental weights
    pub fundamental_weights: Vec<Weight>,
    /// The rank
    pub rank: usize,
}

impl WeightLattice {
    /// Create a weight lattice for a given Cartan type
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
    /// use rustmath_liealgebras::weight_lattice::WeightLattice;
    ///
    /// let ct = CartanType::new(CartanLetter::A, 2).unwrap();
    /// let wl = WeightLattice::new(ct);
    /// assert_eq!(wl.rank(), 2);
    /// ```
    pub fn new(cartan_type: CartanType) -> Self {
        let rank = cartan_type.rank;
        let cartan_matrix = CartanMatrix::new(cartan_type);
        let fundamental_weights = (1..=rank)
            .map(|i| Weight::fundamental_weight(i, rank))
            .collect();

        WeightLattice {
            cartan_type,
            cartan_matrix,
            fundamental_weights,
            rank,
        }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the zero weight
    pub fn zero(&self) -> Weight {
        Weight::zero(self.rank)
    }

    /// Get a fundamental weight ω_i (1-indexed)
    pub fn fundamental_weight(&self, i: usize) -> Weight {
        assert!(i > 0 && i <= self.rank, "Fundamental weight index out of bounds");
        self.fundamental_weights[i - 1].clone()
    }

    /// Get all fundamental weights
    pub fn fundamental_weights_vec(&self) -> Vec<Weight> {
        self.fundamental_weights.clone()
    }

    /// Create a weight from Dynkin coefficients [a_1, ..., a_n]
    ///
    /// This creates the weight a_1ω_1 + ... + a_nω_n
    pub fn weight_from_dynkin(&self, dynkin_coeffs: Vec<Integer>) -> Weight {
        assert_eq!(dynkin_coeffs.len(), self.rank);
        Weight::new(dynkin_coeffs)
    }

    /// Get the dominant weights up to a given level
    ///
    /// Returns all weights λ = c_1ω_1 + ... + c_nω_n where:
    /// - All c_i ≥ 0 (dominant condition)
    /// - c_1 + ... + c_n ≤ level
    pub fn dominant_weights_up_to_level(&self, max_level: usize) -> Vec<Weight> {
        let mut weights = Vec::new();

        // Generate all combinations recursively
        fn generate_dominant(
            rank: usize,
            max_level: usize,
            current: Vec<Integer>,
            current_sum: usize,
            weights: &mut Vec<Weight>,
        ) {
            if current.len() == rank {
                weights.push(Weight::new(current));
                return;
            }

            let remaining = max_level - current_sum;
            for coeff in 0..=remaining {
                let mut next = current.clone();
                next.push(Integer::from(coeff as i64));
                generate_dominant(rank, max_level, next, current_sum + coeff, weights);
            }
        }

        generate_dominant(self.rank, max_level, Vec::new(), 0, &mut weights);
        weights
    }

    /// Get the highest weight of the adjoint representation
    ///
    /// For simple Lie algebras, this is the highest root viewed as a weight
    pub fn adjoint_highest_weight(&self) -> Weight {
        // The highest root in Dynkin coordinates
        match self.cartan_type.letter {
            CartanLetter::A => {
                // Highest root: α_1 + α_2 + ... + α_n = ω_1 + ω_n
                let n = self.rank;
                let mut coeffs = vec![Integer::zero(); n];
                coeffs[0] = Integer::one();
                coeffs[n - 1] = Integer::one();
                Weight::new(coeffs)
            }
            CartanLetter::B => {
                // Highest root: ω_2
                let mut coeffs = vec![Integer::zero(); self.rank];
                if self.rank >= 2 {
                    coeffs[1] = Integer::one();
                }
                Weight::new(coeffs)
            }
            CartanLetter::C => {
                // Highest root: 2ω_1
                let mut coeffs = vec![Integer::zero(); self.rank];
                coeffs[0] = Integer::from(2);
                Weight::new(coeffs)
            }
            CartanLetter::D => {
                // Highest root: ω_2
                let mut coeffs = vec![Integer::zero(); self.rank];
                if self.rank >= 2 {
                    coeffs[1] = Integer::one();
                }
                Weight::new(coeffs)
            }
            CartanLetter::E => {
                // Highest root for E_n
                let mut coeffs = vec![Integer::zero(); self.rank];
                coeffs[0] = Integer::one();
                Weight::new(coeffs)
            }
            CartanLetter::F => {
                // Highest root: ω_1
                let mut coeffs = vec![Integer::zero(); self.rank];
                coeffs[0] = Integer::one();
                Weight::new(coeffs)
            }
            CartanLetter::G => {
                // Highest root: ω_2
                let mut coeffs = vec![Integer::zero(); self.rank];
                coeffs[1] = Integer::one();
                Weight::new(coeffs)
            }
        }
    }

    /// Compute the Weyl dimension formula for a highest weight representation
    ///
    /// For a dominant weight λ, this computes:
    /// dim(V_λ) = ∏_{α>0} (λ+ρ, α) / (ρ, α)
    ///
    /// where ρ = ω_1 + ... + ω_n is the Weyl vector (half-sum of positive roots)
    pub fn weyl_dimension(&self, highest_weight: &Weight) -> Integer {
        // This is a simplified placeholder
        // Full implementation requires root system and proper product computation
        Integer::one()
    }

    /// Get the Weyl vector ρ = ω_1 + ... + ω_n
    ///
    /// Also equal to half the sum of positive roots
    pub fn weyl_vector(&self) -> Weight {
        let coeffs = vec![Integer::one(); self.rank];
        Weight::new(coeffs)
    }

    /// Check if two weights are conjugate under the Weyl group
    ///
    /// Two weights are in the same Weyl orbit if they can be related by the Weyl group action
    pub fn are_weyl_conjugate(&self, w1: &Weight, w2: &Weight) -> bool {
        // Simplified: just check if they're equal
        // Full implementation would check all Weyl group conjugates
        w1 == w2
    }
}

impl Display for WeightLattice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Weight lattice of type {}", self.cartan_type)?;
        writeln!(f, "Fundamental weights:")?;
        for (i, w) in self.fundamental_weights.iter().enumerate() {
            writeln!(f, "  ω_{} = {}", i + 1, w)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_creation() {
        let w = Weight::new(vec![Integer::from(1), Integer::from(2)]);
        assert_eq!(w.rank(), 2);
        assert_eq!(*w.coefficient(1), Integer::from(1));
        assert_eq!(*w.coefficient(2), Integer::from(2));
    }

    #[test]
    fn test_zero_weight() {
        let w = Weight::zero(3);
        assert!(w.is_zero());
        assert_eq!(w.rank(), 3);
    }

    #[test]
    fn test_fundamental_weight() {
        let w = Weight::fundamental_weight(2, 3);
        assert_eq!(*w.coefficient(1), Integer::zero());
        assert_eq!(*w.coefficient(2), Integer::one());
        assert_eq!(*w.coefficient(3), Integer::zero());
    }

    #[test]
    fn test_weight_addition() {
        let w1 = Weight::new(vec![Integer::from(1), Integer::from(2)]);
        let w2 = Weight::new(vec![Integer::from(3), Integer::from(4)]);
        let sum = w1 + w2;

        assert_eq!(*sum.coefficient(1), Integer::from(4));
        assert_eq!(*sum.coefficient(2), Integer::from(6));
    }

    #[test]
    fn test_weight_negation() {
        let w = Weight::new(vec![Integer::from(1), Integer::from(-2)]);
        let neg_w = -w;

        assert_eq!(*neg_w.coefficient(1), Integer::from(-1));
        assert_eq!(*neg_w.coefficient(2), Integer::from(2));
    }

    #[test]
    fn test_is_dominant() {
        let w1 = Weight::new(vec![Integer::from(1), Integer::from(2)]);
        assert!(w1.is_dominant());

        let w2 = Weight::new(vec![Integer::from(1), Integer::from(-1)]);
        assert!(!w2.is_dominant());

        let w3 = Weight::zero(2);
        assert!(w3.is_dominant());
    }

    #[test]
    fn test_weight_level() {
        let w = Weight::new(vec![Integer::from(2), Integer::from(3), Integer::from(1)]);
        assert_eq!(w.level(), Integer::from(6));
    }

    #[test]
    fn test_weight_lattice_creation() {
        let ct = CartanType::new(CartanLetter::A, 3).unwrap();
        let wl = WeightLattice::new(ct);

        assert_eq!(wl.rank(), 3);
        assert_eq!(wl.fundamental_weights.len(), 3);
    }

    #[test]
    fn test_weight_lattice_fundamental_weights() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let wl = WeightLattice::new(ct);

        let omega1 = wl.fundamental_weight(1);
        let omega2 = wl.fundamental_weight(2);

        assert_eq!(*omega1.coefficient(1), Integer::one());
        assert_eq!(*omega1.coefficient(2), Integer::zero());

        assert_eq!(*omega2.coefficient(1), Integer::zero());
        assert_eq!(*omega2.coefficient(2), Integer::one());
    }

    #[test]
    fn test_weyl_vector() {
        let ct = CartanType::new(CartanLetter::A, 3).unwrap();
        let wl = WeightLattice::new(ct);
        let rho = wl.weyl_vector();

        assert_eq!(*rho.coefficient(1), Integer::one());
        assert_eq!(*rho.coefficient(2), Integer::one());
        assert_eq!(*rho.coefficient(3), Integer::one());
    }

    #[test]
    fn test_dominant_weights_small() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let wl = WeightLattice::new(ct);

        let weights = wl.dominant_weights_up_to_level(2);

        // Should have: 0, ω_1, ω_2, 2ω_1, ω_1+ω_2, 2ω_2
        assert_eq!(weights.len(), 6);

        // Check that all are dominant
        for w in &weights {
            assert!(w.is_dominant());
        }
    }

    #[test]
    fn test_weight_display() {
        let w1 = Weight::new(vec![Integer::from(1), Integer::from(2)]);
        let display = format!("{}", w1);
        assert!(display.contains("ω_1"));
        assert!(display.contains("ω_2"));

        let w2 = Weight::zero(2);
        assert_eq!(format!("{}", w2), "0");
    }

    #[test]
    fn test_adjoint_highest_weight() {
        let ct_a3 = CartanType::new(CartanLetter::A, 3).unwrap();
        let wl_a3 = WeightLattice::new(ct_a3);
        let adj = wl_a3.adjoint_highest_weight();

        // For A_3, highest root is ω_1 + ω_3
        assert_eq!(*adj.coefficient(1), Integer::one());
        assert_eq!(*adj.coefficient(2), Integer::zero());
        assert_eq!(*adj.coefficient(3), Integer::one());
    }
}
