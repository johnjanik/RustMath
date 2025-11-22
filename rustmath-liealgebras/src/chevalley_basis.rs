//! Chevalley Basis for Simple Lie Algebras
//!
//! This module implements simple Lie algebras using the Chevalley basis, which is
//! a canonical presentation defined by the root system. The Chevalley basis provides
//! a uniform construction that works for all simple Lie algebra types without requiring
//! explicit matrix representations.
//!
//! The Chevalley basis consists of:
//! - h_i: Elements of the Cartan subalgebra (i = 1, ..., rank)
//! - e_α: Root vectors for each root α
//!
//! The bracket relations are determined by the root system structure:
//! - [h_i, e_α] = ⟨α, α_i^∨⟩ e_α (where α_i^∨ is the i-th simple coroot)
//! - [e_α, e_{-α}] = h_α (the coroot corresponding to α)
//! - [e_α, e_β] = N_{α,β} e_{α+β} (when α + β is a root)
//!
//! Corresponds to sage.algebras.lie_algebras.classical_lie_algebra.LieAlgebraChevalleyBasis
//!
//! # References
//!
//! - Humphreys, J. "Introduction to Lie Algebras and Representation Theory" (1972)
//! - Carter, R. "Simple Groups of Lie Type" (1972)
//! - Kac, V. "Infinite Dimensional Lie Algebras" (1990)

use crate::cartan_type::CartanType;
use crate::root_system::{Root, RootSystem};
use rustmath_core::{Ring, NumericConversion};
use rustmath_rationals::Rational;
use rustmath_integers::Integer;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Element in the Chevalley basis
///
/// Represents a basis element of the Lie algebra:
/// - Cartan(i): The i-th Cartan generator h_i
/// - RootVector(root): The root vector e_α for root α
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChevalleyBasisElement {
    /// Cartan subalgebra element h_i (indexed from 0)
    Cartan(usize),
    /// Root vector e_α for root α
    RootVector(Root),
}

impl Display for ChevalleyBasisElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ChevalleyBasisElement::Cartan(i) => write!(f, "h_{}", i + 1),
            ChevalleyBasisElement::RootVector(root) => {
                write!(f, "e_[")?;
                for (i, coord) in root.coordinates.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", coord)?;
                }
                write!(f, "]")
            }
        }
    }
}

/// Lie Algebra in Chevalley Basis
///
/// A simple finite-dimensional Lie algebra constructed using the Chevalley basis,
/// which is determined entirely by the root system data.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (typically ℚ or ℤ)
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::chevalley_basis::LieAlgebraChevalleyBasis;
/// use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
/// use rustmath_rationals::Rational;
///
/// // Create sl(3) = type A_2 in Chevalley basis
/// let ct = CartanType::new(CartanLetter::A, 2).unwrap();
/// let lie_alg = LieAlgebraChevalleyBasis::<Rational>::new(ct);
/// assert_eq!(lie_alg.rank(), 2);
/// assert_eq!(lie_alg.dimension(), 8); // 2 + 6 = rank + |roots|
/// ```
#[derive(Clone, Debug)]
pub struct LieAlgebraChevalleyBasis<R: Ring> {
    /// Cartan type of the Lie algebra
    cartan_type: CartanType,
    /// Root system
    root_system: RootSystem,
    /// Structure coefficients: (basis_i, basis_j) -> coefficients of [basis_i, basis_j]
    structure_coeffs: HashMap<(ChevalleyBasisElement, ChevalleyBasisElement), Vec<(R, ChevalleyBasisElement)>>,
    /// Coefficient ring marker
    _phantom: PhantomData<R>,
}

impl<R: Ring + Clone + From<i64>> LieAlgebraChevalleyBasis<R> {
    /// Create a new Lie algebra in Chevalley basis
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The Cartan type specifying which simple Lie algebra
    pub fn new(cartan_type: CartanType) -> Self {
        let root_system = RootSystem::new(cartan_type.clone());
        let structure_coeffs = Self::compute_structure_coefficients(&root_system);

        Self {
            cartan_type,
            root_system,
            structure_coeffs,
            _phantom: PhantomData,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Get the rank (dimension of Cartan subalgebra)
    pub fn rank(&self) -> usize {
        self.cartan_type.rank
    }

    /// Get the dimension of the Lie algebra
    ///
    /// dimension = rank + number of roots
    pub fn dimension(&self) -> usize {
        self.rank() + 2 * self.root_system.positive_roots.len()
    }

    /// Compute the Cartan matrix entry A_{ij} = ⟨α_i, α_j^∨⟩
    ///
    /// This is the action of the j-th coroot on the i-th simple root
    fn cartan_matrix_entry(&self, i: usize, j: usize) -> Rational {
        let alpha_i = &self.root_system.simple_roots[i];
        let alpha_j = &self.root_system.simple_roots[j];

        // ⟨α_i, α_j^∨⟩ = 2⟨α_i, α_j⟩ / ⟨α_j, α_j⟩
        let numerator = Rational::from_integer(2) * alpha_i.inner_product(alpha_j);
        let denominator = alpha_j.inner_product(alpha_j);

        numerator / denominator
    }

    /// Compute structure coefficients for the Chevalley basis
    ///
    /// This constructs the bracket relations:
    /// - [h_i, h_j] = 0 (Cartan subalgebra is abelian)
    /// - [h_i, e_α] = A(α_i, α) e_α
    /// - [e_α, e_{-α}] = h_α
    /// - [e_α, e_β] = N_{α,β} e_{α+β} (when α + β is a root)
    ///
    /// Note: For simplicity, we use integer coefficients for structure constants
    fn compute_structure_coefficients(
        root_system: &RootSystem,
    ) -> HashMap<(ChevalleyBasisElement, ChevalleyBasisElement), Vec<(R, ChevalleyBasisElement)>> {
        let mut coeffs = HashMap::new();
        let rank = root_system.simple_roots.len();

        // [h_i, h_j] = 0 (Cartan is abelian)
        for i in 0..rank {
            for j in 0..rank {
                coeffs.insert(
                    (ChevalleyBasisElement::Cartan(i), ChevalleyBasisElement::Cartan(j)),
                    vec![],
                );
            }
        }

        // [h_i, e_α] = ⟨α, α_i^∨⟩ e_α
        // For simplicity, we use the integer part of the coefficient
        for i in 0..rank {
            let alpha_i = &root_system.simple_roots[i];

            // Positive roots
            for alpha in &root_system.positive_roots {
                let coeff_val = Self::root_action_coefficient(alpha, alpha_i);
                // Convert Rational to Integer (floor)
                let int_coeff = Self::rational_to_int(&coeff_val);

                if int_coeff != Integer::zero() {
                    let coeff = Self::integer_to_ring(&int_coeff);
                    coeffs.insert(
                        (ChevalleyBasisElement::Cartan(i), ChevalleyBasisElement::RootVector(alpha.clone())),
                        vec![(coeff, ChevalleyBasisElement::RootVector(alpha.clone()))],
                    );
                }

                // Also for negative roots
                let neg_alpha = alpha.negate();
                let neg_int_coeff = -int_coeff.clone();
                if neg_int_coeff != Integer::zero() {
                    let neg_coeff = Self::integer_to_ring(&neg_int_coeff);
                    coeffs.insert(
                        (ChevalleyBasisElement::Cartan(i), ChevalleyBasisElement::RootVector(neg_alpha.clone())),
                        vec![(neg_coeff, ChevalleyBasisElement::RootVector(neg_alpha))],
                    );
                }
            }
        }

        // [e_α, e_{-α}] = h_α (simplified: sum of Cartan generators)
        for alpha in &root_system.positive_roots {
            let neg_alpha = alpha.negate();

            // Express h_α as linear combination of simple coroots
            let mut h_alpha_coeffs = vec![];
            for i in 0..rank {
                let coeff_val = Self::root_action_coefficient(alpha, &root_system.simple_roots[i]);
                let int_coeff = Self::rational_to_int(&coeff_val);
                if int_coeff != Integer::zero() {
                    let coeff = Self::integer_to_ring(&int_coeff);
                    h_alpha_coeffs.push((coeff, ChevalleyBasisElement::Cartan(i)));
                }
            }

            coeffs.insert(
                (ChevalleyBasisElement::RootVector(alpha.clone()), ChevalleyBasisElement::RootVector(neg_alpha.clone())),
                h_alpha_coeffs.clone(),
            );

            // Anti-symmetry: [e_{-α}, e_α] = -h_α
            let neg_h_alpha_coeffs: Vec<_> = h_alpha_coeffs
                .into_iter()
                .map(|(c, b)| {
                    let zero = R::from(0);
                    (zero - c, b)
                })
                .collect();
            coeffs.insert(
                (ChevalleyBasisElement::RootVector(neg_alpha), ChevalleyBasisElement::RootVector(alpha.clone())),
                neg_h_alpha_coeffs,
            );
        }

        // [e_α, e_β] = N_{α,β} e_{α+β} when α + β is a root
        // For simplicity, we set N_{α,β} = 1 (proper computation requires root string theory)
        for alpha in &root_system.positive_roots {
            for beta in &root_system.positive_roots {
                if alpha != beta {
                    // Check if alpha + beta is a root (simplified check)
                    let sum_root = Self::add_roots(alpha, beta);
                    if Self::is_root(&sum_root, &root_system.positive_roots) {
                        coeffs.insert(
                            (ChevalleyBasisElement::RootVector(alpha.clone()), ChevalleyBasisElement::RootVector(beta.clone())),
                            vec![(R::from(1), ChevalleyBasisElement::RootVector(sum_root))],
                        );
                    }
                }
            }
        }

        coeffs
    }

    /// Convert Rational to Integer (numerator / denominator, taking floor)
    fn rational_to_int(r: &Rational) -> Integer {
        // For exact division, just return numerator if denominator is 1
        // Otherwise, do integer division
        let num = r.numerator().clone();
        let den = r.denominator().clone();
        if den == Integer::one() {
            num
        } else {
            // Integer division (floor)
            num / den
        }
    }

    /// Convert Integer to Ring element
    fn integer_to_ring(i: &Integer) -> R {
        // Try to convert to i64 first, otherwise use 0 as fallback
        let val = i.to_i64();
        R::from(val)
    }

    /// Compute ⟨β, α^∨⟩ = 2⟨β, α⟩ / ⟨α, α⟩
    fn root_action_coefficient(beta: &Root, alpha: &Root) -> Rational {
        let numerator = Rational::from_integer(2) * beta.inner_product(alpha);
        let denominator = alpha.inner_product(alpha);
        numerator / denominator
    }

    /// Add two roots
    fn add_roots(alpha: &Root, beta: &Root) -> Root {
        let coords: Vec<_> = alpha
            .coordinates
            .iter()
            .zip(&beta.coordinates)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        Root::new(coords)
    }

    /// Check if a root is in the set of positive roots
    fn is_root(root: &Root, roots: &[Root]) -> bool {
        roots.iter().any(|r| r == root)
    }

    /// Compute the bracket [x, y] where x and y are basis elements
    ///
    /// Returns a linear combination of basis elements
    pub fn bracket(
        &self,
        x: &ChevalleyBasisElement,
        y: &ChevalleyBasisElement,
    ) -> Vec<(R, ChevalleyBasisElement)> {
        self.structure_coeffs
            .get(&(x.clone(), y.clone()))
            .cloned()
            .unwrap_or_else(Vec::new)
    }

    /// Get all basis elements
    pub fn basis(&self) -> Vec<ChevalleyBasisElement> {
        let mut basis = vec![];

        // Cartan subalgebra
        for i in 0..self.rank() {
            basis.push(ChevalleyBasisElement::Cartan(i));
        }

        // Positive root vectors
        for root in &self.root_system.positive_roots {
            basis.push(ChevalleyBasisElement::RootVector(root.clone()));
        }

        // Negative root vectors
        for root in &self.root_system.positive_roots {
            basis.push(ChevalleyBasisElement::RootVector(root.negate()));
        }

        basis
    }
}

impl<R: Ring + Clone + From<i64>> Display for LieAlgebraChevalleyBasis<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lie algebra of type {} in Chevalley basis", self.cartan_type)
    }
}

/// Simply-laced variant of Chevalley basis Lie algebra
///
/// For simply-laced types (A, D, E), this provides an alternative construction
/// with an asymmetry function parameter. Simply-laced means all roots have the
/// same length, which simplifies many calculations.
///
/// Corresponds to sage.algebras.lie_algebras.classical_lie_algebra.LieAlgebraChevalleyBasis_simply_laced
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::chevalley_basis::LieAlgebraChevalleyBasisSimplyLaced;
/// use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
/// use rustmath_rationals::Rational;
///
/// // Create sl(4) = type A_3 in simply-laced Chevalley basis
/// let ct = CartanType::new(CartanLetter::A, 3).unwrap();
/// let lie_alg = LieAlgebraChevalleyBasisSimplyLaced::<Rational>::new(ct);
/// assert_eq!(lie_alg.rank(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct LieAlgebraChevalleyBasisSimplyLaced<R: Ring> {
    /// The underlying Chevalley basis Lie algebra
    base: LieAlgebraChevalleyBasis<R>,
    /// Asymmetry function (epsilon parameter)
    /// Maps pairs of roots to {-1, 1}
    epsilon: HashMap<(Root, Root), i64>,
}

impl<R: Ring + Clone + From<i64>> LieAlgebraChevalleyBasisSimplyLaced<R> {
    /// Create a new simply-laced Lie algebra in Chevalley basis
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - Must be a simply-laced type (A, D, or E)
    ///
    /// # Panics
    ///
    /// Panics if the Cartan type is not simply-laced
    pub fn new(cartan_type: CartanType) -> Self {
        use crate::cartan_type::CartanLetter;

        // Verify simply-laced
        match cartan_type.letter {
            CartanLetter::A | CartanLetter::D | CartanLetter::E => {},
            _ => panic!("Type {:?} is not simply-laced", cartan_type.letter),
        }

        let base = LieAlgebraChevalleyBasis::new(cartan_type);
        let epsilon = Self::construct_epsilon(&base.root_system);

        Self { base, epsilon }
    }

    /// Construct the asymmetry function epsilon
    ///
    /// This can be chosen arbitrarily subject to antisymmetry:
    /// epsilon(α, β) = -epsilon(β, α)
    fn construct_epsilon(root_system: &RootSystem) -> HashMap<(Root, Root), i64> {
        let mut epsilon = HashMap::new();

        // Simple construction: epsilon(α, β) = 1 if α < β lexicographically
        for alpha in &root_system.positive_roots {
            for beta in &root_system.positive_roots {
                if alpha != beta {
                    let val = if Self::lexicographic_less(alpha, beta) { 1 } else { -1 };
                    epsilon.insert((alpha.clone(), beta.clone()), val);
                    epsilon.insert((beta.clone(), alpha.clone()), -val);
                }
            }
        }

        epsilon
    }

    /// Lexicographic comparison of roots
    fn lexicographic_less(alpha: &Root, beta: &Root) -> bool {
        for (a, b) in alpha.coordinates.iter().zip(&beta.coordinates) {
            if a < b {
                return true;
            } else if a > b {
                return false;
            }
        }
        false
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.base.rank()
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.base.dimension()
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &CartanType {
        self.base.cartan_type()
    }

    /// Compute bracket with asymmetry function
    ///
    /// For simply-laced types: [e_α, e_β] = epsilon(α, β) e_{α+β}
    pub fn bracket(
        &self,
        x: &ChevalleyBasisElement,
        y: &ChevalleyBasisElement,
    ) -> Vec<(R, ChevalleyBasisElement)> {
        // Use asymmetry function for root vector brackets
        if let (ChevalleyBasisElement::RootVector(alpha), ChevalleyBasisElement::RootVector(beta)) = (x, y) {
            if let Some(&eps) = self.epsilon.get(&(alpha.clone(), beta.clone())) {
                let sum_root = LieAlgebraChevalleyBasis::<R>::add_roots(alpha, beta);
                if LieAlgebraChevalleyBasis::<R>::is_root(&sum_root, &self.base.root_system.positive_roots) {
                    return vec![(R::from(eps), ChevalleyBasisElement::RootVector(sum_root))];
                }
            }
        }

        // Otherwise use base bracket
        self.base.bracket(x, y)
    }
}

impl<R: Ring + Clone + From<i64>> Display for LieAlgebraChevalleyBasisSimplyLaced<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Simply-laced Lie algebra of type {} in Chevalley basis", self.base.cartan_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartan_type::{CartanLetter, CartanType};

    #[test]
    fn test_chevalley_basis_type_a2() {
        // sl(3) has type A_2
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let lie_alg = LieAlgebraChevalleyBasis::<Rational>::new(ct);

        assert_eq!(lie_alg.rank(), 2);
        assert_eq!(lie_alg.dimension(), 8); // 2 (Cartan) + 6 (roots)

        let basis = lie_alg.basis();
        assert_eq!(basis.len(), 8);
    }

    #[test]
    fn test_chevalley_basis_type_b2() {
        // so(5) has type B_2
        let ct = CartanType::new(CartanLetter::B, 2).unwrap();
        let lie_alg = LieAlgebraChevalleyBasis::<Rational>::new(ct);

        assert_eq!(lie_alg.rank(), 2);
        assert_eq!(lie_alg.dimension(), 10); // 2 (Cartan) + 8 (roots)
    }

    #[test]
    fn test_chevalley_basis_type_g2() {
        let ct = CartanType::new(CartanLetter::G, 2).unwrap();
        let lie_alg = LieAlgebraChevalleyBasis::<Rational>::new(ct);

        assert_eq!(lie_alg.rank(), 2);
        assert_eq!(lie_alg.dimension(), 14); // 2 (Cartan) + 12 (roots)
    }

    #[test]
    fn test_cartan_is_abelian() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let lie_alg = LieAlgebraChevalleyBasis::<Rational>::new(ct);

        let h0 = ChevalleyBasisElement::Cartan(0);
        let h1 = ChevalleyBasisElement::Cartan(1);

        let bracket = lie_alg.bracket(&h0, &h1);
        assert_eq!(bracket.len(), 0); // [h_0, h_1] = 0
    }

    #[test]
    fn test_simply_laced_type_a() {
        let ct = CartanType::new(CartanLetter::A, 3).unwrap();
        let lie_alg = LieAlgebraChevalleyBasisSimplyLaced::<Rational>::new(ct);

        assert_eq!(lie_alg.rank(), 3);
        assert_eq!(lie_alg.dimension(), 15); // 3 + 12
    }

    #[test]
    fn test_simply_laced_type_d() {
        let ct = CartanType::new(CartanLetter::D, 4).unwrap();
        let lie_alg = LieAlgebraChevalleyBasisSimplyLaced::<Rational>::new(ct);

        assert_eq!(lie_alg.rank(), 4);
        assert_eq!(lie_alg.dimension(), 28); // 4 + 24
    }

    #[test]
    #[should_panic(expected = "not simply-laced")]
    fn test_simply_laced_rejects_non_simply_laced() {
        let ct = CartanType::new(CartanLetter::B, 2).unwrap();
        let _lie_alg = LieAlgebraChevalleyBasisSimplyLaced::<Rational>::new(ct);
    }

    #[test]
    fn test_basis_element_display() {
        let h0 = ChevalleyBasisElement::Cartan(0);
        assert_eq!(format!("{}", h0), "h_1");

        let root = Root::new(vec![Rational::one(), Rational::zero()]);
        let e_root = ChevalleyBasisElement::RootVector(root);
        assert!(format!("{}", e_root).starts_with("e_["));
    }

    #[test]
    fn test_lie_algebra_display() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let lie_alg = LieAlgebraChevalleyBasis::<Rational>::new(ct);

        let display = format!("{}", lie_alg);
        assert!(display.contains("A_2"));
        assert!(display.contains("Chevalley basis"));
    }
}
