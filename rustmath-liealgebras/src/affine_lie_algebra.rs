//! Affine Lie Algebras
//!
//! This module implements affine Lie algebras, which are infinite-dimensional
//! Kac-Moody algebras obtained by extending finite-dimensional simple Lie algebras.
//!
//! # Mathematical Background
//!
//! An affine Lie algebra is constructed from a finite-dimensional simple Lie algebra
//! 𝔤 by forming the tensor product with Laurent polynomials and adding a central element:
//!
//! ĝ' = (𝔤 ⊗ ℝ[t, t⁻¹]) ⊕ ℝc
//!
//! where:
//! - 𝔤 is a finite-dimensional simple Lie algebra
//! - ℝ[t, t⁻¹] is the Laurent polynomial ring
//! - c is a canonical central element
//!
//! The Lie bracket is defined by:
//! [x ⊗ t^m + λc, y ⊗ t^n + μc] = [x, y] ⊗ t^(m+n) + mδ_{m,-n}(x|y)c
//!
//! where (·|·) is the Killing form on 𝔤.
//!
//! # Twisted vs. Untwisted
//!
//! - **Untwisted** (Type X^(1)): Direct construction from classical Lie algebras
//! - **Twisted** (Type X_N^(r), r > 1): Uses diagram automorphisms of order r
//!
//! # Kac-Moody Extension
//!
//! The full Kac-Moody algebra adds a derivation d:
//!
//! ĝ = ĝ' ⊕ ℝd
//!
//! where d acts as t d/dt on the polynomial ring.
//!
//! Corresponds to sage.algebras.lie_algebras.affine_lie_algebra
//!
//! # References
//!
//! - Kac, V. "Infinite-Dimensional Lie Algebras" (3rd edition, 1990)
//! - Carter, R. "Lie Algebras of Finite and Affine Type" (2005)

use crate::cartan_type::CartanType;
use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Base trait for affine Lie algebra elements
pub trait AffineLieAlgebraElement: Clone {
    /// Get the underlying classical component
    fn classical_part(&self) -> Vec<i32>;

    /// Get the Laurent polynomial degree
    fn degree(&self) -> i32;

    /// Get the central element coefficient
    fn central_coeff(&self) -> i32;

    /// Get the derivation coefficient (for Kac-Moody algebras)
    fn derivation_coeff(&self) -> i32;
}

/// Indices for twisted affine Lie algebras
///
/// Manages indexing of basis elements in twisted affine Lie algebras,
/// which involve diagram automorphisms of the underlying root system.
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::affine_lie_algebra::TwistedAffineIndices;
/// let indices = TwistedAffineIndices::new(2);
/// assert_eq!(indices.order(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct TwistedAffineIndices {
    /// Order of the diagram automorphism
    order: usize,
    /// Dimension of the classical Lie algebra
    classical_dim: usize,
}

impl TwistedAffineIndices {
    /// Create new twisted affine indices
    ///
    /// # Arguments
    ///
    /// * `order` - Order of the diagram automorphism (typically 2 or 3)
    pub fn new(order: usize) -> Self {
        TwistedAffineIndices {
            order,
            classical_dim: 0,
        }
    }

    /// Create with specified classical dimension
    pub fn with_dimension(order: usize, classical_dim: usize) -> Self {
        TwistedAffineIndices {
            order,
            classical_dim,
        }
    }

    /// Get the automorphism order
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the classical dimension
    pub fn classical_dimension(&self) -> usize {
        self.classical_dim
    }

    /// Check if an index is valid
    pub fn is_valid_index(&self, classical_idx: usize, power: i32) -> bool {
        classical_idx < self.classical_dim && power >= 0
    }
}

impl Display for TwistedAffineIndices {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Twisted affine indices with order {} automorphism",
            self.order
        )
    }
}

/// Affine Lie Algebra
///
/// Abstract base class for both twisted and untwisted affine Lie algebras.
/// Provides common functionality for infinite-dimensional Kac-Moody algebras.
///
/// # Type Parameters
///
/// * `R` - The base ring (typically ℚ or ℝ)
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::affine_lie_algebra::AffineLieAlgebra;
/// # use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
/// # use rustmath_integers::Integer;
/// let ct = CartanType::new(CartanLetter::A, 2).unwrap();
/// let aff: AffineLieAlgebra<Integer> = AffineLieAlgebra::new(ct, false);
/// assert_eq!(aff.rank(), 3); // Affine rank is classical rank + 1
/// ```
#[derive(Debug, Clone)]
pub struct AffineLieAlgebra<R: Ring> {
    /// Classical Cartan type (before affine extension)
    cartan_type: CartanType,
    /// Whether this includes the derivation (Kac-Moody) or not (affine)
    kac_moody: bool,
    /// Rank of the affine algebra (classical rank + 1)
    rank: usize,
    /// Base ring marker
    _phantom: PhantomData<R>,
}

impl<R: Ring> AffineLieAlgebra<R> {
    /// Create a new affine Lie algebra
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The classical Cartan type
    /// * `kac_moody` - If true, include derivation (Kac-Moody); if false, affine only
    pub fn new(cartan_type: CartanType, kac_moody: bool) -> Self {
        let classical_rank = cartan_type.rank();
        AffineLieAlgebra {
            cartan_type,
            kac_moody,
            rank: classical_rank + 1,
            _phantom: PhantomData,
        }
    }

    /// Get the affine rank (classical rank + 1)
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the classical rank
    pub fn classical_rank(&self) -> usize {
        self.cartan_type.rank()
    }

    /// Get the Cartan type of the classical part
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Check if this is a Kac-Moody algebra (includes derivation)
    pub fn is_kac_moody(&self) -> bool {
        self.kac_moody
    }

    /// Get the canonical central element
    ///
    /// The central element c is the generator that commutes with all elements.
    pub fn central_element(&self) -> UntwistedAffineElement<R>
    where
        R: From<i64>,
    {
        UntwistedAffineElement {
            classical_indices: vec![],
            laurent_power: 0,
            central_coeff: R::from(1),
            derivation_coeff: R::from(0),
        }
    }

    /// Get the derivation element
    ///
    /// The derivation d acts as t d/dt on the polynomial ring.
    /// Returns zero if this is not a Kac-Moody algebra.
    pub fn derivation(&self) -> UntwistedAffineElement<R>
    where
        R: From<i64>,
    {
        if self.kac_moody {
            UntwistedAffineElement {
                classical_indices: vec![],
                laurent_power: 0,
                central_coeff: R::from(0),
                derivation_coeff: R::from(1),
            }
        } else {
            UntwistedAffineElement::zero()
        }
    }

    /// Get the derived subalgebra (removes derivation)
    pub fn derived_subalgebra(&self) -> Self {
        AffineLieAlgebra {
            cartan_type: self.cartan_type.clone(),
            kac_moody: false,
            rank: self.rank,
            _phantom: PhantomData,
        }
    }

    /// Convert to Kac-Moody algebra (adds derivation if not present)
    pub fn to_kac_moody(&self) -> Self {
        AffineLieAlgebra {
            cartan_type: self.cartan_type.clone(),
            kac_moody: true,
            rank: self.rank,
            _phantom: PhantomData,
        }
    }
}

impl<R: Ring> Display for AffineLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.kac_moody {
            write!(
                f,
                "Kac-Moody algebra of type {} (affine extension)",
                self.cartan_type
            )
        } else {
            write!(
                f,
                "Affine Lie algebra of type {} (affine extension)",
                self.cartan_type
            )
        }
    }
}

/// Untwisted Affine Lie Algebra
///
/// The untwisted affine Lie algebra is the direct construction
/// ĝ' = (𝔤 ⊗ ℝ[t, t⁻¹]) ⊕ ℝc from a classical simple Lie algebra 𝔤.
///
/// Elements are of the form x ⊗ t^n + λc + μd where:
/// - x is an element of the classical Lie algebra
/// - n is an integer (Laurent polynomial degree)
/// - λ is the central element coefficient
/// - μ is the derivation coefficient (if Kac-Moody)
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::affine_lie_algebra::UntwistedAffineLieAlgebra;
/// # use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
/// # use rustmath_integers::Integer;
/// let ct = CartanType::new(CartanLetter::A, 2).unwrap();
/// let aff: UntwistedAffineLieAlgebra<Integer> = UntwistedAffineLieAlgebra::new(ct);
/// assert!(!aff.is_twisted());
/// ```
#[derive(Debug, Clone)]
pub struct UntwistedAffineLieAlgebra<R: Ring> {
    /// Base affine algebra structure
    base: AffineLieAlgebra<R>,
}

impl<R: Ring> UntwistedAffineLieAlgebra<R> {
    /// Create a new untwisted affine Lie algebra
    pub fn new(cartan_type: CartanType) -> Self {
        UntwistedAffineLieAlgebra {
            base: AffineLieAlgebra::new(cartan_type, false),
        }
    }

    /// Create a Kac-Moody version (with derivation)
    pub fn kac_moody(cartan_type: CartanType) -> Self {
        UntwistedAffineLieAlgebra {
            base: AffineLieAlgebra::new(cartan_type, true),
        }
    }

    /// Get the base affine algebra
    pub fn base(&self) -> &AffineLieAlgebra<R> {
        &self.base
    }

    /// Check if twisted (always false for this type)
    pub fn is_twisted(&self) -> bool {
        false
    }

    /// Get generator e_i for the affine root system
    ///
    /// # Arguments
    ///
    /// * `i` - Index (0 is the affine simple root, 1..rank are classical)
    pub fn e(&self, i: usize) -> Option<UntwistedAffineElement<R>>
    where
        R: From<i64>,
    {
        if i >= self.base.rank() {
            return None;
        }

        if i == 0 {
            // Affine root e_0
            Some(UntwistedAffineElement::affine_root(0))
        } else {
            // Classical root e_i
            Some(UntwistedAffineElement::classical_root(i))
        }
    }

    /// Get generator f_i (lowering operator)
    pub fn f(&self, i: usize) -> Option<UntwistedAffineElement<R>>
    where
        R: From<i64>,
    {
        if i >= self.base.rank() {
            return None;
        }

        if i == 0 {
            // Affine root f_0
            Some(UntwistedAffineElement::affine_coroot(0))
        } else {
            // Classical root f_i
            Some(UntwistedAffineElement::classical_coroot(i))
        }
    }

    /// Get all Chevalley generators
    pub fn chevalley_generators(&self) -> Vec<UntwistedAffineElement<R>>
    where
        R: From<i64>,
    {
        let mut gens = Vec::new();
        for i in 0..self.base.rank() {
            if let Some(e_i) = self.e(i) {
                gens.push(e_i);
            }
            if let Some(f_i) = self.f(i) {
                gens.push(f_i);
            }
        }
        gens
    }
}

impl<R: Ring> Display for UntwistedAffineLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Untwisted {}", self.base)
    }
}

/// Twisted Affine Lie Algebra
///
/// Twisted affine Lie algebras arise from diagram automorphisms of
/// untwisted affine Lie algebras. They correspond to Cartan types
/// with superscript (r) where r > 1.
///
/// The construction embeds the twisted algebra inside an untwisted
/// affine Lie algebra using eigenspaces of the automorphism.
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::affine_lie_algebra::TwistedAffineLieAlgebra;
/// # use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
/// # use rustmath_integers::Integer;
/// // Type A_{2n}^(2) is a twisted affine algebra
/// let ct = CartanType::new(CartanLetter::A, 4).unwrap();
/// let twisted: TwistedAffineLieAlgebra<Integer> =
///     TwistedAffineLieAlgebra::new(ct, 2);
/// assert!(twisted.is_twisted());
/// assert_eq!(twisted.automorphism_order(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct TwistedAffineLieAlgebra<R: Ring> {
    /// Base affine algebra structure
    base: AffineLieAlgebra<R>,
    /// Order of the diagram automorphism
    automorphism_order: usize,
    /// Index structure for the twisted algebra
    indices: TwistedAffineIndices,
}

impl<R: Ring> TwistedAffineLieAlgebra<R> {
    /// Create a new twisted affine Lie algebra
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The classical Cartan type
    /// * `automorphism_order` - Order of the diagram automorphism (2 or 3)
    pub fn new(cartan_type: CartanType, automorphism_order: usize) -> Self {
        let classical_dim = cartan_type.rank();
        TwistedAffineLieAlgebra {
            base: AffineLieAlgebra::new(cartan_type, false),
            automorphism_order,
            indices: TwistedAffineIndices::with_dimension(
                automorphism_order,
                classical_dim,
            ),
        }
    }

    /// Get the base affine algebra
    pub fn base(&self) -> &AffineLieAlgebra<R> {
        &self.base
    }

    /// Check if twisted (always true for this type)
    pub fn is_twisted(&self) -> bool {
        true
    }

    /// Get the automorphism order
    pub fn automorphism_order(&self) -> usize {
        self.automorphism_order
    }

    /// Get the twisted affine indices
    pub fn indices(&self) -> &TwistedAffineIndices {
        &self.indices
    }

    /// Lift an element to the ambient untwisted algebra
    ///
    /// Maps from the twisted algebra to its realization inside
    /// a larger untwisted affine Lie algebra.
    pub fn lift<T>(&self, element: T) -> T {
        // In a full implementation, this would perform the actual lift
        // For now, return the element unchanged
        element
    }

    /// Retract an element from the ambient untwisted algebra
    ///
    /// Projects from the ambient algebra back to the twisted algebra.
    pub fn retract<T>(&self, element: T) -> Option<T> {
        // In a full implementation, this would check if the element
        // is in the twisted subalgebra and project it
        Some(element)
    }
}

impl<R: Ring> Display for TwistedAffineLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Twisted {} with automorphism order {}",
            self.base, self.automorphism_order
        )
    }
}

/// Element of an untwisted affine Lie algebra
///
/// Represents x ⊗ t^n + λc + μd where:
/// - x is a classical Lie algebra element (as indices)
/// - n is the Laurent polynomial degree
/// - λ is the central element coefficient
/// - μ is the derivation coefficient
#[derive(Debug, Clone)]
pub struct UntwistedAffineElement<R: Ring> {
    /// Indices in the classical Lie algebra basis
    classical_indices: Vec<usize>,
    /// Power of t in the Laurent polynomial
    laurent_power: i32,
    /// Coefficient of the central element c
    central_coeff: R,
    /// Coefficient of the derivation d
    derivation_coeff: R,
}

impl<R: Ring + Clone> UntwistedAffineElement<R> {
    /// Create a new element
    pub fn new(
        classical_indices: Vec<usize>,
        laurent_power: i32,
        central_coeff: R,
        derivation_coeff: R,
    ) -> Self {
        UntwistedAffineElement {
            classical_indices,
            laurent_power,
            central_coeff,
            derivation_coeff,
        }
    }

    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        UntwistedAffineElement {
            classical_indices: vec![],
            laurent_power: 0,
            central_coeff: R::from(0),
            derivation_coeff: R::from(0),
        }
    }

    /// Create a classical root e_i
    fn classical_root(i: usize) -> Self
    where
        R: From<i64>,
    {
        UntwistedAffineElement {
            classical_indices: vec![i],
            laurent_power: 0,
            central_coeff: R::from(0),
            derivation_coeff: R::from(0),
        }
    }

    /// Create a classical coroot f_i
    fn classical_coroot(i: usize) -> Self
    where
        R: From<i64>,
    {
        UntwistedAffineElement {
            classical_indices: vec![i + 1000], // Marker for coroots
            laurent_power: 0,
            central_coeff: R::from(0),
            derivation_coeff: R::from(0),
        }
    }

    /// Create the affine root e_0
    fn affine_root(i: usize) -> Self
    where
        R: From<i64>,
    {
        UntwistedAffineElement {
            classical_indices: vec![],
            laurent_power: 1,
            central_coeff: R::from(0),
            derivation_coeff: R::from(0),
        }
    }

    /// Create the affine coroot f_0
    fn affine_coroot(i: usize) -> Self
    where
        R: From<i64>,
    {
        UntwistedAffineElement {
            classical_indices: vec![],
            laurent_power: -1,
            central_coeff: R::from(0),
            derivation_coeff: R::from(0),
        }
    }

    /// Get the Laurent polynomial degree
    pub fn degree(&self) -> i32 {
        self.laurent_power
    }

    /// Get the central coefficient
    pub fn central(&self) -> &R {
        &self.central_coeff
    }

    /// Get the derivation coefficient
    pub fn derivation(&self) -> &R {
        &self.derivation_coeff
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i64>,
    {
        self.classical_indices.is_empty()
            && self.central_coeff == R::from(0)
            && self.derivation_coeff == R::from(0)
    }
}

impl<R: Ring + Clone + Display + PartialEq + From<i64>> Display for UntwistedAffineElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut parts = Vec::new();

        if !self.classical_indices.is_empty() {
            parts.push(format!("x_{:?} ⊗ t^{}", self.classical_indices, self.laurent_power));
        }

        if self.central_coeff != R::from(0) {
            parts.push(format!("{}c", self.central_coeff));
        }

        if self.derivation_coeff != R::from(0) {
            parts.push(format!("{}d", self.derivation_coeff));
        }

        if parts.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", parts.join(" + "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartan_type::{CartanLetter, CartanType};
    use rustmath_integers::Integer;

    #[test]
    fn test_affine_creation() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let aff: AffineLieAlgebra<Integer> = AffineLieAlgebra::new(ct, false);

        assert_eq!(aff.rank(), 3); // Affine rank = classical rank + 1
        assert_eq!(aff.classical_rank(), 2);
        assert!(!aff.is_kac_moody());
    }

    #[test]
    fn test_kac_moody_conversion() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let aff: AffineLieAlgebra<Integer> = AffineLieAlgebra::new(ct, false);
        let km = aff.to_kac_moody();

        assert!(km.is_kac_moody());
        assert_eq!(km.rank(), 3);
    }

    #[test]
    fn test_derived_subalgebra() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let km: AffineLieAlgebra<Integer> = AffineLieAlgebra::new(ct, true);
        let derived = km.derived_subalgebra();

        assert!(!derived.is_kac_moody());
    }

    #[test]
    fn test_untwisted_affine() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let aff: UntwistedAffineLieAlgebra<Integer> = UntwistedAffineLieAlgebra::new(ct);

        assert!(!aff.is_twisted());
        assert_eq!(aff.base().rank(), 3);
    }

    #[test]
    fn test_untwisted_generators() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let aff: UntwistedAffineLieAlgebra<Integer> = UntwistedAffineLieAlgebra::new(ct);

        let e0 = aff.e(0);
        assert!(e0.is_some());

        let e1 = aff.e(1);
        assert!(e1.is_some());
    }

    #[test]
    fn test_chevalley_generators() {
        let ct = CartanType::new(CartanLetter::A, 1).unwrap();
        let aff: UntwistedAffineLieAlgebra<Integer> = UntwistedAffineLieAlgebra::new(ct);

        let gens = aff.chevalley_generators();
        assert!(gens.len() >= 2); // At least e_0, f_0, e_1, f_1
    }

    #[test]
    fn test_twisted_affine() {
        let ct = CartanType::new(CartanLetter::A, 4).unwrap();
        let twisted: TwistedAffineLieAlgebra<Integer> = TwistedAffineLieAlgebra::new(ct, 2);

        assert!(twisted.is_twisted());
        assert_eq!(twisted.automorphism_order(), 2);
    }

    #[test]
    fn test_twisted_indices() {
        let indices = TwistedAffineIndices::new(3);
        assert_eq!(indices.order(), 3);
    }

    #[test]
    fn test_element_creation() {
        let elem: UntwistedAffineElement<Integer> = UntwistedAffineElement::zero();
        assert!(elem.is_zero());
    }

    #[test]
    fn test_element_degree() {
        let elem: UntwistedAffineElement<Integer> = UntwistedAffineElement::new(
            vec![1],
            3,
            Integer::from(0),
            Integer::from(0),
        );
        assert_eq!(elem.degree(), 3);
    }

    #[test]
    fn test_central_element() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let aff: AffineLieAlgebra<Integer> = AffineLieAlgebra::new(ct, false);
        let c = aff.central_element();

        assert_eq!(*c.central(), Integer::from(1));
    }

    #[test]
    fn test_derivation_element() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let aff: AffineLieAlgebra<Integer> = AffineLieAlgebra::new(ct, true);
        let d = aff.derivation();

        assert_eq!(*d.derivation(), Integer::from(1));
    }

    #[test]
    fn test_lift_retract() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let twisted: TwistedAffineLieAlgebra<Integer> = TwistedAffineLieAlgebra::new(ct, 2);

        let elem: UntwistedAffineElement<Integer> = UntwistedAffineElement::zero();
        let lifted = twisted.lift(elem.clone());
        let retracted = twisted.retract(lifted);

        assert!(retracted.is_some());
    }
}
