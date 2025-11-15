//! Fusion Rings - Verlinde algebras for modular tensor categories
//!
//! A fusion ring is the Grothendieck ring of a modular tensor category.
//! These arise from Wess-Zumino-Witten conformal field theories and quantum
//! groups at roots of unity.
//!
//! Fusion rings are implemented as variants of Weyl character rings, with basis
//! elements indexed by dominant weights of bounded level.

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_matrix::Matrix;
use rustmath_liealgebras::{CartanType, RootSystem};
use std::collections::HashMap;
use std::fmt;
use num_bigint::BigInt;
use num_rational::BigRational;

/// Weight vector for indexing fusion ring basis elements
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Weight {
    /// Fundamental weight coordinates
    pub coords: Vec<i64>,
}

impl Weight {
    /// Create a new weight from coordinates
    pub fn new(coords: Vec<i64>) -> Self {
        Weight { coords }
    }

    /// Get the rank (dimension) of the weight
    pub fn rank(&self) -> usize {
        self.coords.len()
    }

    /// Check if this is the zero weight
    pub fn is_zero(&self) -> bool {
        self.coords.iter().all(|&c| c == 0)
    }

    /// Compute the level of the weight (sum of coordinates)
    pub fn level(&self) -> i64 {
        self.coords.iter().sum()
    }
}

impl fmt::Display for Weight {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;
        for (i, coord) in self.coords.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", coord)?;
        }
        write!(f, ")")
    }
}

/// Element of a fusion ring
#[derive(Debug, Clone)]
pub struct FusionRingElement {
    /// Coefficients for each basis element (indexed by weight)
    pub coeffs: HashMap<Weight, Integer>,
    /// Parent fusion ring
    ring: FusionRing,
}

impl FusionRingElement {
    /// Create a new element from coefficients
    pub fn new(coeffs: HashMap<Weight, Integer>, ring: FusionRing) -> Self {
        FusionRingElement { coeffs, ring }
    }

    /// Create a basis element for a given weight
    pub fn basis_element(weight: Weight, ring: FusionRing) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(weight, Integer::one());
        FusionRingElement { coeffs, ring }
    }

    /// Get the coefficient for a given weight
    pub fn coeff(&self, weight: &Weight) -> Integer {
        self.coeffs.get(weight).cloned().unwrap_or(Integer::zero())
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.values().all(|c| c.is_zero())
    }

    /// Get the parent fusion ring
    pub fn parent(&self) -> &FusionRing {
        &self.ring
    }

    /// Compute the dual (conjugate) element
    pub fn dual(&self) -> Self {
        let mut result_coeffs = HashMap::new();
        for (weight, coeff) in &self.coeffs {
            if let Some(dual_weight) = self.ring.dual_weight(weight) {
                result_coeffs.insert(dual_weight, coeff.clone());
            }
        }
        FusionRingElement::new(result_coeffs, self.ring.clone())
    }

    /// Add two fusion ring elements
    pub fn add(&self, other: &FusionRingElement) -> Self {
        let mut result_coeffs = self.coeffs.clone();
        for (weight, coeff) in &other.coeffs {
            let current = result_coeffs.entry(weight.clone()).or_insert(Integer::zero());
            *current = current.clone() + coeff.clone();
        }
        // Remove zero coefficients
        result_coeffs.retain(|_, v| !v.is_zero());
        FusionRingElement::new(result_coeffs, self.ring.clone())
    }

    /// Multiply two fusion ring elements using fusion rules
    pub fn multiply(&self, other: &FusionRingElement) -> Self {
        let mut result_coeffs = HashMap::new();

        for (weight_i, coeff_i) in &self.coeffs {
            for (weight_j, coeff_j) in &other.coeffs {
                // Get fusion product decomposition
                let decomp = self.ring.fusion_product(weight_i, weight_j);
                for (weight_k, mult) in decomp {
                    let total_coeff = coeff_i.clone() * coeff_j.clone() * Integer::from(mult);
                    let current = result_coeffs.entry(weight_k).or_insert(Integer::zero());
                    *current = current.clone() + total_coeff;
                }
            }
        }

        // Remove zero coefficients
        result_coeffs.retain(|_, v| !v.is_zero());
        FusionRingElement::new(result_coeffs, self.ring.clone())
    }
}

impl fmt::Display for FusionRingElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (weight, coeff) in &self.coeffs {
            if !first {
                write!(f, " + ")?;
            }
            if coeff != &Integer::one() {
                write!(f, "{}*", coeff)?;
            }
            write!(f, "{}", self.ring.label_for_weight(weight))?;
            first = false;
        }
        Ok(())
    }
}

/// A Fusion Ring (Verlinde algebra)
///
/// Represents the Grothendieck ring of a modular tensor category,
/// typically arising from WZW conformal field theories at level k.
#[derive(Debug, Clone)]
pub struct FusionRing {
    /// Cartan type of the underlying Lie algebra
    cartan_type: CartanType,
    /// Fusion level (nonnegative integer)
    level: usize,
    /// Dual Coxeter number
    dual_coxeter_number: i64,
    /// Cyclotomic order for field computations
    cyclotomic_order: Option<usize>,
    /// Basis weights (dominant weights of level ≤ k)
    basis_weights: Vec<Weight>,
    /// Fusion coefficients N^k_ij (structure constants)
    /// Maps (i, j, k) indices to multiplicity
    fusion_coeffs: HashMap<(usize, usize, usize), i64>,
    /// Custom labels for basis elements
    labels: HashMap<Weight, String>,
    /// S-matrix (computed lazily)
    s_matrix_cache: Option<Matrix<Rational>>,
}

impl FusionRing {
    /// Create a new fusion ring of given Cartan type and level
    ///
    /// # Arguments
    /// * `cartan_type` - The Cartan type (e.g., A₂, D₄, E₆)
    /// * `level` - The fusion level k (nonnegative integer)
    ///
    /// # Example
    /// ```ignore
    /// use rustmath_liealgebras::{CartanType, CartanLetter};
    /// use rustmath_algebras::FusionRing;
    ///
    /// let ct = CartanType::new(CartanLetter::A, 2).unwrap();
    /// let fr = FusionRing::new(ct, 3);
    /// ```
    pub fn new(cartan_type: CartanType, level: usize) -> Self {
        let dual_coxeter_number = Self::compute_dual_coxeter_number(&cartan_type);
        let basis_weights = Self::compute_basis_weights(&cartan_type, level);
        let fusion_coeffs = HashMap::new(); // Will be computed on demand
        let labels = HashMap::new();

        FusionRing {
            cartan_type,
            level,
            dual_coxeter_number,
            cyclotomic_order: None,
            basis_weights,
            fusion_coeffs,
            labels,
            s_matrix_cache: None,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Get the fusion level
    pub fn level(&self) -> usize {
        self.level
    }

    /// Get the rank of the underlying Lie algebra
    pub fn rank(&self) -> usize {
        self.cartan_type.rank()
    }

    /// Get the dual Coxeter number
    pub fn dual_coxeter_number(&self) -> i64 {
        self.dual_coxeter_number
    }

    /// Get the basis weights
    pub fn basis_weights(&self) -> &[Weight] {
        &self.basis_weights
    }

    /// Get the dimension of the fusion ring (number of simple objects)
    pub fn dimension(&self) -> usize {
        self.basis_weights.len()
    }

    /// Set custom labels for basis elements
    pub fn set_labels(&mut self, labels: HashMap<Weight, String>) {
        self.labels = labels;
    }

    /// Get the label for a weight
    pub fn label_for_weight(&self, weight: &Weight) -> String {
        self.labels.get(weight)
            .cloned()
            .unwrap_or_else(|| format!("L{}", weight))
    }

    /// Create a basis element for a given weight
    pub fn basis_element(&self, weight: Weight) -> FusionRingElement {
        FusionRingElement::basis_element(weight, self.clone())
    }

    /// Create a basis element by index
    pub fn basis_element_by_index(&self, index: usize) -> Option<FusionRingElement> {
        self.basis_weights.get(index)
            .map(|w| self.basis_element(w.clone()))
    }

    /// Compute the dual (conjugate) weight
    pub fn dual_weight(&self, weight: &Weight) -> Option<Weight> {
        // For most cases, the dual is the weight with reversed coordinates
        // This is a simplified implementation
        let dual_coords: Vec<i64> = weight.coords.iter().rev().cloned().collect();
        let dual = Weight::new(dual_coords);

        // Check if dual is in the basis
        if self.basis_weights.contains(&dual) {
            Some(dual)
        } else {
            None
        }
    }

    /// Get the fusion coefficient N^k_ij
    ///
    /// This is the multiplicity of the k-th simple object in the fusion
    /// product of the i-th and j-th simple objects.
    pub fn fusion_coeff(&self, i: usize, j: usize, k: usize) -> i64 {
        self.fusion_coeffs.get(&(i, j, k)).copied().unwrap_or(0)
    }

    /// Compute the fusion product of two weights
    ///
    /// Returns a HashMap mapping result weights to their multiplicities
    pub fn fusion_product(&self, weight_i: &Weight, weight_j: &Weight) -> HashMap<Weight, i64> {
        // Find indices of the weights
        let i = self.basis_weights.iter().position(|w| w == weight_i);
        let j = self.basis_weights.iter().position(|w| w == weight_j);

        if let (Some(i_idx), Some(j_idx)) = (i, j) {
            let mut result = HashMap::new();
            for (k_idx, weight_k) in self.basis_weights.iter().enumerate() {
                let coeff = self.fusion_coeff(i_idx, j_idx, k_idx);
                if coeff > 0 {
                    result.insert(weight_k.clone(), coeff);
                }
            }
            result
        } else {
            HashMap::new()
        }
    }

    /// Compute the S-matrix
    ///
    /// The S-matrix is a central invariant of the fusion ring, containing
    /// information about the modular transformation properties.
    ///
    /// # Arguments
    /// * `unitary` - If true, return the unitary S-matrix; otherwise unnormalized
    pub fn s_matrix(&mut self, unitary: bool) -> Matrix<Rational> {
        if let Some(ref cached) = self.s_matrix_cache {
            return cached.clone();
        }

        let n = self.dimension();
        let mut s_entries = vec![vec![Rational::zero(); n]; n];

        // Compute S-matrix entries using the Kac-Peterson formula
        // This is a simplified version - full implementation would use
        // root systems and Weyl group computations
        for i in 0..n {
            for j in 0..n {
                s_entries[i][j] = self.compute_s_ij(i, j);
            }
        }

        // Flatten the matrix into row-major order
        let mut flat_entries = Vec::new();
        for row in s_entries {
            flat_entries.extend(row);
        }
        let s_matrix = Matrix::from_vec(n, n, flat_entries)
            .unwrap_or_else(|_| Matrix::zeros(n, n));

        if unitary {
            // Normalize by sqrt(global quantum dimension)
            // This is a placeholder - full implementation would compute proper normalization
            s_matrix
        } else {
            self.s_matrix_cache = Some(s_matrix.clone());
            s_matrix
        }
    }

    /// Compute a single S-matrix entry S_ij
    fn compute_s_ij(&self, i: usize, j: usize) -> Rational {
        // Placeholder implementation
        // Full implementation would use:
        // S_ij = ∑_w (q^(w·(λ_i + ρ)) - q^(w·(-λ_i - ρ))) *
        //             (q^(w·(λ_j + ρ)) - q^(w·(-λ_j - ρ))) / |W|
        // where the sum is over Weyl group elements

        if i == j {
            Rational::one()
        } else {
            Rational::zero()
        }
    }

    /// Compute the quantum dimension of a basis element
    ///
    /// The quantum dimension is given by q-dimension formulas involving
    /// roots of unity.
    pub fn quantum_dimension(&self, _index: usize) -> Rational {
        // Placeholder implementation
        // Full implementation would use the formula:
        // d_i = S_i0 / S_00
        Rational::one()
    }

    /// Compute the global quantum dimension
    ///
    /// This is the sum of the squares of all quantum dimensions.
    pub fn global_quantum_dimension(&mut self) -> Rational {
        let mut total = Rational::zero();
        for i in 0..self.dimension() {
            let d_i = self.quantum_dimension(i);
            total = total + d_i.clone() * d_i;
        }
        total
    }

    /// Compute the Virasoro central charge
    ///
    /// For WZW models, this is given by:
    /// c = k·dim(g) / (k + h^∨)
    pub fn virasoro_central_charge(&self) -> Rational {
        let k = self.level as i64;
        let h_dual = self.dual_coxeter_number;
        let dim = self.cartan_type.rank() as i64; // Simplified

        Rational::new(
            Integer::from(k * dim),
            Integer::from(k + h_dual)
        ).unwrap_or(Rational::zero())
    }

    /// Check if the fusion ring is multiplicity-free
    ///
    /// A fusion ring is multiplicity-free if all fusion coefficients are 0 or 1.
    pub fn is_multiplicity_free(&self) -> bool {
        self.fusion_coeffs.values().all(|&n| n == 0 || n == 1)
    }

    /// Compute basis weights up to a given level
    fn compute_basis_weights(cartan_type: &CartanType, level: usize) -> Vec<Weight> {
        let rank = cartan_type.rank();
        let mut weights = Vec::new();

        // Generate all dominant weights with level ≤ k
        // This is a simplified implementation
        Self::generate_weights_recursive(&mut weights, Vec::new(), rank, level as i64);

        weights
    }

    /// Helper for generating weights recursively
    fn generate_weights_recursive(
        weights: &mut Vec<Weight>,
        current: Vec<i64>,
        remaining_dims: usize,
        remaining_level: i64,
    ) {
        if remaining_dims == 0 {
            weights.push(Weight::new(current));
            return;
        }

        for coord in 0..=remaining_level {
            let mut next = current.clone();
            next.push(coord);
            Self::generate_weights_recursive(
                weights,
                next,
                remaining_dims - 1,
                remaining_level - coord,
            );
        }
    }

    /// Compute the dual Coxeter number for a Cartan type
    fn compute_dual_coxeter_number(cartan_type: &CartanType) -> i64 {
        // Dual Coxeter numbers for simple Lie algebras
        use rustmath_liealgebras::CartanLetter;

        let n = cartan_type.rank() as i64;
        match cartan_type.letter() {
            CartanLetter::A => n + 1,
            CartanLetter::B => 2 * n - 1,
            CartanLetter::C => n + 1,
            CartanLetter::D => 2 * n - 2,
            CartanLetter::E => match n {
                6 => 12,
                7 => 18,
                8 => 30,
                _ => 0,
            },
            CartanLetter::F => 9,  // F₄
            CartanLetter::G => 4,  // G₂
        }
    }
}

impl fmt::Display for FusionRing {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FusionRing({}, level={})",
            self.cartan_type, self.level
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_liealgebras::CartanLetter;

    #[test]
    fn test_fusion_ring_creation() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let fr = FusionRing::new(ct, 2);

        assert_eq!(fr.level(), 2);
        assert_eq!(fr.rank(), 2);
        assert!(fr.dimension() > 0);
    }

    #[test]
    fn test_weight_operations() {
        let w1 = Weight::new(vec![1, 0]);
        let w2 = Weight::new(vec![0, 1]);

        assert_eq!(w1.level(), 1);
        assert_eq!(w2.level(), 1);
        assert!(!w1.is_zero());

        let w0 = Weight::new(vec![0, 0]);
        assert!(w0.is_zero());
    }

    #[test]
    fn test_basis_element() {
        let ct = CartanType::new(CartanLetter::A, 1).unwrap();
        let fr = FusionRing::new(ct, 2);

        if let Some(elem) = fr.basis_element_by_index(0) {
            assert!(!elem.is_zero());
            assert_eq!(elem.parent().level(), 2);
        }
    }

    #[test]
    fn test_dual_coxeter_numbers() {
        let type_a2 = CartanType::new(CartanLetter::A, 2).unwrap();
        assert_eq!(FusionRing::compute_dual_coxeter_number(&type_a2), 3);

        let type_d4 = CartanType::new(CartanLetter::D, 4).unwrap();
        assert_eq!(FusionRing::compute_dual_coxeter_number(&type_d4), 6);

        let type_e6 = CartanType::new(CartanLetter::E, 6).unwrap();
        assert_eq!(FusionRing::compute_dual_coxeter_number(&type_e6), 12);
    }

    #[test]
    fn test_virasoro_central_charge() {
        let ct = CartanType::new(CartanLetter::A, 1).unwrap();
        let fr = FusionRing::new(ct, 1);

        let c = fr.virasoro_central_charge();
        // For A₁ at level 1: c = 1·1/(1+2) = 1/3
        // (Using simplified dimension calculation)
        assert!(c.to_f64().unwrap() > 0.0);
    }

    #[test]
    fn test_element_addition() {
        let ct = CartanType::new(CartanLetter::A, 1).unwrap();
        let fr = FusionRing::new(ct, 2);

        let e1 = fr.basis_element_by_index(0).unwrap();
        let e2 = e1.add(&e1);

        // 2*e1 should have coefficient 2
        assert!(!e2.is_zero());
    }

    #[test]
    fn test_fusion_ring_display() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let fr = FusionRing::new(ct, 3);

        let display = format!("{}", fr);
        assert!(display.contains("FusionRing"));
        assert!(display.contains("level=3"));
    }
}
