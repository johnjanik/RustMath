//! Demonstrating Parent trait usage with algebras
//!
//! This module shows how to integrate the Parent trait with our algebra
//! implementations to provide proper parent/element relationships.

use rustmath_core::{Ring, Parent, ParentWithBasis, ParentWithGenerators};
use crate::down_up_algebra::{DownUpAlgebra, Element as DownUpElement, DownUpBasisIndex};

/// DownUpAlgebra with Parent implementation
///
/// This wraps the DownUpAlgebra to provide Parent trait functionality
impl<R: Ring> Parent for DownUpAlgebra<R> {
    type Element = DownUpElement<R>;

    fn contains(&self, _element: &Self::Element) -> bool {
        // All DownUpElements belong to this algebra
        // In a more sophisticated implementation, we could check if the element
        // was actually created from this specific algebra instance
        true
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(self.zero())
    }

    fn one(&self) -> Option<Self::Element> {
        Some(self.one())
    }

    /// `Some(1)` over the zero ring; otherwise `None` (no claim, see
    /// [`Parent::cardinality`]; the algebra is then infinite, see `is_finite`).
    fn cardinality(&self) -> Option<usize> {
        if self.base_ring_is_zero() {
            Some(1)
        } else {
            None
        }
    }

    /// MAGMA `IsFinite` (Ch. 17 §17.4.3). The monomials u^i (du)^j d^k form a basis
    /// (Benkart and Roby, "Down-up algebras", J. Algebra 209 (1998); the crate's normal
    /// form `DownUpBasisIndex`), so over a nonzero base ring the u^i, i >= 0, are distinct
    /// and the algebra is infinite. Over the zero ring it is the zero ring.
    ///
    /// Whether the base ring is the zero ring is read off the stored parameter `alpha`
    /// (see `base_ring_is_zero`), never from the static `R::zero()`/`R::one()`: those panic
    /// for some parent-carrying element types (the p-adic elements), and for others
    /// (`GaloisRingElement`, `FiniteFieldElement`) they are unbound integer constants that
    /// know nothing about the ring. `DownUpAlgebra::new` does not call those constructors,
    /// so such an algebra can be built.
    fn is_finite(&self) -> Option<bool> {
        Some(self.base_ring_is_zero())
    }

    fn name(&self) -> String {
        format!("Down-Up Algebra DU({}, {}, {})",
                self.alpha(), self.beta(), self.gamma())
    }
}

impl<R: Ring> DownUpAlgebra<R> {
    /// Whether the base ring is the zero ring, decided from the stored parameter `alpha`:
    /// in the zero ring every element is both 0 and 1, and in a nonzero ring no element is
    /// (`x = 0` and `x = 1` give `0 = 1`). Uses no static constructor of `R`.
    fn base_ring_is_zero(&self) -> bool {
        let alpha = self.alpha();
        alpha.is_zero() && alpha.is_one()
    }
}

impl<R: Ring> ParentWithGenerators for DownUpAlgebra<R> {
    fn generators(&self) -> Vec<Self::Element> {
        let (d, u) = self.algebra_generators();
        vec![d, u]
    }

    fn num_generators(&self) -> usize {
        2
    }
}

/// Example: A simple algebra parent with finite basis
#[derive(Debug, Clone)]
pub struct SimpleAlgebra<R: Ring> {
    dimension: usize,
    base_ring: std::marker::PhantomData<R>,
}

impl<R: Ring> SimpleAlgebra<R> {
    pub fn new(dimension: usize) -> Self {
        SimpleAlgebra {
            dimension,
            base_ring: std::marker::PhantomData,
        }
    }
}

/// Element type for SimpleAlgebra
#[derive(Debug, Clone, PartialEq)]
pub struct SimpleElement<R: Ring> {
    pub coefficients: Vec<R>,
}

impl<R: Ring> Parent for SimpleAlgebra<R> {
    type Element = SimpleElement<R>;

    fn contains(&self, element: &Self::Element) -> bool {
        element.coefficients.len() == self.dimension
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(SimpleElement {
            coefficients: vec![R::zero(); self.dimension],
        })
    }

    fn one(&self) -> Option<Self::Element> {
        let mut coeffs = vec![R::zero(); self.dimension];
        if self.dimension > 0 {
            coeffs[0] = R::one();
        }
        Some(SimpleElement { coefficients: coeffs })
    }

    /// `Some(1)` for dimension 0; otherwise `None`: the size `#R^dim` depends on the base
    /// ring, which this parent does not carry.
    fn cardinality(&self) -> Option<usize> {
        if self.dimension == 0 {
            Some(1)
        } else {
            None
        }
    }

    /// MAGMA `IsFinite` (Ch. 17 §17.4.3). `R^dim` with `dim >= 1` is finite iff `R` is.
    /// `Some(false)` when the element type fixes characteristic 0 (a nonzero ring of
    /// characteristic 0 is infinite); `None` when only the runtime parent of `R` knows
    /// (for `R = IntegerMod` or `FiniteFieldElement` the algebra is finite, of unknown size).
    fn is_finite(&self) -> Option<bool> {
        if self.dimension == 0 {
            Some(true)
        } else if R::static_characteristic() == Some(0) {
            Some(false)
        } else {
            None
        }
    }

    fn name(&self) -> String {
        format!("Simple Algebra of dimension {}", self.dimension)
    }
}

impl<R: Ring> ParentWithBasis for SimpleAlgebra<R> {
    type BasisIndex = usize;

    fn dimension(&self) -> Option<usize> {
        Some(self.dimension)
    }

    fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element> {
        if *index < self.dimension {
            let mut coeffs = vec![R::zero(); self.dimension];
            coeffs[*index] = R::one();
            Some(SimpleElement { coefficients: coeffs })
        } else {
            None
        }
    }

    fn basis_indices(&self) -> Vec<Self::BasisIndex> {
        (0..self.dimension).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_down_up_algebra_as_parent() {
        let algebra = DownUpAlgebra::new(
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
        );

        // Test Parent trait methods
        assert_eq!(algebra.name(), "Down-Up Algebra DU(1, 0, 0)");
        assert_eq!(algebra.is_finite(), Some(false));
        assert_eq!(algebra.cardinality(), None);

        // Test zero and one (Parent trait returns Option)
        let zero = Parent::zero(&algebra).unwrap();
        assert!(zero.is_zero());

        let one = Parent::one(&algebra).unwrap();
        assert!(!one.is_zero());

        // Test generators
        assert_eq!(algebra.num_generators(), 2);
        let gens = algebra.generators();
        assert_eq!(gens.len(), 2);
    }

    #[test]
    fn test_simple_algebra_as_parent() {
        let algebra: SimpleAlgebra<Integer> = SimpleAlgebra::new(3);

        // Test Parent trait
        assert_eq!(algebra.name(), "Simple Algebra of dimension 3");
        assert_eq!(algebra.is_finite(), Some(false));

        // Test zero element
        let zero = algebra.zero().unwrap();
        assert!(algebra.contains(&zero));
        assert_eq!(zero.coefficients, vec![Integer::zero(); 3]);

        // Test one element
        let one = algebra.one().unwrap();
        assert!(algebra.contains(&one));
        assert_eq!(one.coefficients[0], Integer::one());

        // Test basis
        assert_eq!(algebra.dimension(), Some(3));
        let b0 = algebra.basis_element(&0).unwrap();
        assert_eq!(b0.coefficients[0], Integer::one());
        assert_eq!(b0.coefficients[1], Integer::zero());

        let b1 = algebra.basis_element(&1).unwrap();
        assert_eq!(b1.coefficients[1], Integer::one());

        // Invalid index
        assert!(algebra.basis_element(&10).is_none());

        // Test basis indices
        let indices = algebra.basis_indices();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_element_membership() {
        let algebra: SimpleAlgebra<Integer> = SimpleAlgebra::new(3);

        // Valid element
        let elem = SimpleElement {
            coefficients: vec![Integer::from(1), Integer::from(2), Integer::from(3)],
        };
        assert!(algebra.contains(&elem));

        // Invalid element (wrong dimension)
        let bad_elem = SimpleElement {
            coefficients: vec![Integer::from(1), Integer::from(2)],
        };
        assert!(!algebra.contains(&bad_elem));
    }

    // MAGMA Ch. 17 §17.4.3 IsFinite: decided only where the element type proves it.
    #[test]
    fn test_simple_algebra_finiteness() {
        use rustmath_polynomials::UnivariatePolynomial;
        use rustmath_rationals::Rational;

        // R^0 = {0}.
        let zero_dim: SimpleAlgebra<Integer> = SimpleAlgebra::new(0);
        assert_eq!(zero_dim.is_finite(), Some(true));
        assert_eq!(zero_dim.cardinality(), Some(1));

        // Q^2 is infinite (GAP: IsFinite(Rationals) -> false).
        let q2: SimpleAlgebra<Rational> = SimpleAlgebra::new(2);
        assert_eq!(q2.is_finite(), Some(false));
        assert_eq!(q2.cardinality(), None);

        // The polynomial element type states no characteristic: undecided, not "infinite".
        let poly2: SimpleAlgebra<UnivariatePolynomial<Integer>> = SimpleAlgebra::new(2);
        assert_eq!(poly2.is_finite(), None);
    }

    #[test]
    fn test_down_up_algebra_over_q_is_infinite() {
        use rustmath_rationals::Rational;
        let algebra = DownUpAlgebra::new(
            Rational::from_integer(1),
            Rational::from_integer(1),
            Rational::from_integer(0),
        );
        assert_eq!(algebra.is_finite(), Some(false));
        assert_eq!(algebra.cardinality(), None);
    }

    // GR(2^3, 2) is a nonzero ring (MAGMA Ch. 48: #GR(8, 2) = 8^2 = 64), so DU(a, b, c)
    // over it has the basis u^i (du)^j d^k (Benkart and Roby, J. Algebra 209 (1998)) and is
    // infinite. (`GaloisRingElement::zero()`/`one()` no longer panic, they return unbound
    // constants, so the guard that `cardinality`/`is_finite` never call the static
    // constructors is `test_down_up_algebra_never_calls_static_constructors` below.)
    #[test]
    fn test_down_up_algebra_over_galois_ring_does_not_panic() {
        use rustmath_finitefields::{GaloisRing, GaloisRingElement};
        let r = GaloisRing::new(Integer::from(2), 3, 2).unwrap();
        let algebra = DownUpAlgebra::<GaloisRingElement>::new(
            r.from_int(Integer::from(1)),
            r.from_int(Integer::from(3)),
            r.generator(),
        );
        assert_eq!(algebra.cardinality(), None);
        assert_eq!(algebra.is_finite(), Some(false));
        // A zero parameter does not make the base ring the zero ring.
        let algebra0 = DownUpAlgebra::<GaloisRingElement>::new(r.zero(), r.zero(), r.zero());
        assert_eq!(algebra0.cardinality(), None);
        assert_eq!(algebra0.is_finite(), Some(false));
    }

    /// Z/nZ elements that carry their modulus and whose static constructors panic, like the
    /// p-adic element types: the guard that `DownUpAlgebra`'s `cardinality`/`is_finite` never
    /// call `R::zero()`/`R::one()`. With n = 1 it is the zero ring (0 = 1).
    #[derive(Clone, Debug, PartialEq)]
    struct PanickyZn {
        v: u64,
        n: u64,
    }

    impl PanickyZn {
        fn new(v: u64, n: u64) -> Self {
            PanickyZn { v: v % n, n }
        }
        fn modulus(&self, other: &Self) -> u64 {
            assert_eq!(self.n, other.n, "operands live in different rings");
            self.n
        }
    }

    impl std::fmt::Display for PanickyZn {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{} (mod {})", self.v, self.n)
        }
    }
    impl std::ops::Add for PanickyZn {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            let n = self.modulus(&o);
            PanickyZn::new(self.v + o.v, n)
        }
    }
    impl std::ops::Sub for PanickyZn {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            let n = self.modulus(&o);
            PanickyZn::new(self.v + (n - o.v), n)
        }
    }
    impl std::ops::Mul for PanickyZn {
        type Output = Self;
        fn mul(self, o: Self) -> Self {
            let n = self.modulus(&o);
            PanickyZn::new(self.v * o.v, n)
        }
    }
    impl std::ops::Neg for PanickyZn {
        type Output = Self;
        fn neg(self) -> Self {
            PanickyZn::new(self.n - self.v, self.n)
        }
    }
    impl Ring for PanickyZn {
        fn zero() -> Self {
            panic!("PanickyZn::zero() needs a parent ring")
        }
        fn one() -> Self {
            panic!("PanickyZn::one() needs a parent ring")
        }
        fn is_zero(&self) -> bool {
            self.v == 0
        }
        fn is_one(&self) -> bool {
            self.v == 1 % self.n
        }
    }

    // Review follow-up (8cc9f75): `cardinality`/`is_finite` must not call the static
    // `R::zero()`/`R::one()`. Over Z/8Z (a nonzero ring) the algebra is infinite (basis
    // u^i (du)^j d^k, Benkart and Roby 1998); over Z/1Z (the zero ring) it has one element.
    #[test]
    fn test_down_up_algebra_never_calls_static_constructors() {
        let z8 = |v| PanickyZn::new(v, 8);
        let algebra = DownUpAlgebra::<PanickyZn>::new(z8(1), z8(3), z8(5));
        assert_eq!(algebra.cardinality(), None);
        assert_eq!(algebra.is_finite(), Some(false));
        let algebra0 = DownUpAlgebra::<PanickyZn>::new(z8(0), z8(0), z8(0));
        assert_eq!(algebra0.cardinality(), None);
        assert_eq!(algebra0.is_finite(), Some(false));
        let z1 = |v| PanickyZn::new(v, 1);
        let zero_ring = DownUpAlgebra::<PanickyZn>::new(z1(0), z1(0), z1(0));
        assert_eq!(zero_ring.cardinality(), Some(1));
        assert_eq!(zero_ring.is_finite(), Some(true));
    }

    /// The zero ring {0 = 1}, for the zero-ring branch of `DownUpAlgebra`'s finiteness.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ZeroRing;

    impl std::fmt::Display for ZeroRing {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "0")
        }
    }
    impl std::ops::Add for ZeroRing {
        type Output = Self;
        fn add(self, _: Self) -> Self {
            ZeroRing
        }
    }
    impl std::ops::Sub for ZeroRing {
        type Output = Self;
        fn sub(self, _: Self) -> Self {
            ZeroRing
        }
    }
    impl std::ops::Mul for ZeroRing {
        type Output = Self;
        fn mul(self, _: Self) -> Self {
            ZeroRing
        }
    }
    impl std::ops::Neg for ZeroRing {
        type Output = Self;
        fn neg(self) -> Self {
            ZeroRing
        }
    }
    impl Ring for ZeroRing {
        fn zero() -> Self {
            ZeroRing
        }
        fn one() -> Self {
            ZeroRing
        }
        fn is_zero(&self) -> bool {
            true
        }
        fn is_one(&self) -> bool {
            true
        }
    }

    // Over the zero ring the down-up algebra is the zero ring: one element.
    #[test]
    fn test_down_up_algebra_over_zero_ring_is_finite() {
        let algebra = DownUpAlgebra::<ZeroRing>::new(ZeroRing, ZeroRing, ZeroRing);
        assert_eq!(algebra.is_finite(), Some(true));
        assert_eq!(algebra.cardinality(), Some(1));
    }
}
