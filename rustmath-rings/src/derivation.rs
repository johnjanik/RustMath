//! Ring derivations and derivation modules
//!
//! A derivation is a linear map D: R → R satisfying the Leibniz rule:
//! D(ab) = aD(b) + D(a)b
//!
//! This module provides:
//! - Ring derivation trait and implementations
//! - Derivation module structure
//! - Various types of derivations (zero, constant, composition, etc.)

use rustmath_core::{Ring, EuclideanDomain};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::fmt;

/// Trait for ring derivations
///
/// A derivation D: R → R is a map satisfying:
/// 1. D(a + b) = D(a) + D(b) (additivity)
/// 2. D(ab) = aD(b) + D(a)b (Leibniz rule)
pub trait RingDerivation<R: Ring>: Clone {
    /// Apply the derivation to an element
    fn apply(&self, x: &R) -> R;

    /// Compose this derivation with another
    ///
    /// Returns D1 ∘ D2, which is generally NOT a derivation
    fn compose<D: RingDerivation<R>>(&self, other: &D) -> ComposedDerivation<R, Self, D>
    where
        Self: Sized
    {
        ComposedDerivation {
            first: other.clone(),
            second: self.clone(),
            _phantom: PhantomData,
        }
    }

    /// Check if this is the zero derivation
    fn is_zero(&self) -> bool;
}

/// The zero derivation: D(x) = 0 for all x
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeroDerivation<R: Ring> {
    _phantom: PhantomData<R>,
}

impl<R: Ring> ZeroDerivation<R> {
    /// Create a new zero derivation
    pub fn new() -> Self {
        ZeroDerivation {
            _phantom: PhantomData,
        }
    }
}

impl<R: Ring> Default for ZeroDerivation<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> RingDerivation<R> for ZeroDerivation<R> {
    fn apply(&self, _x: &R) -> R {
        R::zero()
    }

    fn is_zero(&self) -> bool {
        true
    }
}

impl<R: Ring> fmt::Display for ZeroDerivation<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zero derivation")
    }
}

/// A derivation defined by a function
///
/// This allows creating custom derivations from closures or functions
#[derive(Clone)]
pub struct FunctionDerivation<R: Ring, F>
where
    F: Fn(&R) -> R + Clone,
{
    func: F,
    _phantom: PhantomData<R>,
}

impl<R: Ring, F> FunctionDerivation<R, F>
where
    F: Fn(&R) -> R + Clone,
{
    /// Create a new derivation from a function
    ///
    /// # Safety
    /// The caller must ensure the function satisfies the derivation axioms
    pub fn new(func: F) -> Self {
        FunctionDerivation {
            func,
            _phantom: PhantomData,
        }
    }
}

impl<R: Ring, F> RingDerivation<R> for FunctionDerivation<R, F>
where
    F: Fn(&R) -> R + Clone,
{
    fn apply(&self, x: &R) -> R {
        (self.func)(x)
    }

    fn is_zero(&self) -> bool {
        // Cannot determine in general
        false
    }
}

impl<R: Ring, F> fmt::Display for FunctionDerivation<R, F>
where
    F: Fn(&R) -> R + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Function derivation")
    }
}

/// Composition of two derivations
///
/// Note: The composition of derivations is generally NOT a derivation!
/// This is provided for convenience but should be used carefully.
#[derive(Clone)]
pub struct ComposedDerivation<R: Ring, D1: RingDerivation<R>, D2: RingDerivation<R>> {
    first: D1,
    second: D2,
    _phantom: PhantomData<R>,
}

impl<R: Ring, D1: RingDerivation<R>, D2: RingDerivation<R>> ComposedDerivation<R, D1, D2> {
    /// Create a new composed derivation (second ∘ first)
    pub fn new(first: D1, second: D2) -> Self {
        ComposedDerivation {
            first,
            second,
            _phantom: PhantomData,
        }
    }
}

impl<R: Ring, D1: RingDerivation<R>, D2: RingDerivation<R>> RingDerivation<R>
    for ComposedDerivation<R, D1, D2>
{
    fn apply(&self, x: &R) -> R {
        let intermediate = self.first.apply(x);
        self.second.apply(&intermediate)
    }

    fn is_zero(&self) -> bool {
        self.first.is_zero() || self.second.is_zero()
    }
}

impl<R: Ring, D1: RingDerivation<R>, D2: RingDerivation<R>> fmt::Display
    for ComposedDerivation<R, D1, D2>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Composed derivation")
    }
}

/// A derivation with a twist (automorphism)
///
/// A twisted derivation satisfies: D(ab) = σ(a)D(b) + D(a)b
/// where σ is a ring automorphism
pub struct TwistedDerivation<R: Ring, D: RingDerivation<R>, F>
where
    F: Fn(&R) -> R + Clone,
{
    derivation: D,
    twist: F,
    _phantom: PhantomData<R>,
}

impl<R: Ring, D: RingDerivation<R>, F> TwistedDerivation<R, D, F>
where
    F: Fn(&R) -> R + Clone,
{
    /// Create a new twisted derivation
    pub fn new(derivation: D, twist: F) -> Self {
        TwistedDerivation {
            derivation,
            twist,
            _phantom: PhantomData,
        }
    }
}

impl<R: Ring, D: RingDerivation<R>, F> Clone for TwistedDerivation<R, D, F>
where
    F: Fn(&R) -> R + Clone,
{
    fn clone(&self) -> Self {
        TwistedDerivation {
            derivation: self.derivation.clone(),
            twist: self.twist.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<R: Ring, D: RingDerivation<R>, F> RingDerivation<R> for TwistedDerivation<R, D, F>
where
    F: Fn(&R) -> R + Clone,
{
    fn apply(&self, x: &R) -> R {
        self.derivation.apply(x)
    }

    fn is_zero(&self) -> bool {
        self.derivation.is_zero()
    }
}

impl<R: Ring, D: RingDerivation<R>, F> fmt::Display for TwistedDerivation<R, D, F>
where
    F: Fn(&R) -> R + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Twisted derivation")
    }
}

/// Module of derivations over a ring
///
/// This represents the R-module Der(R) of all derivations on a ring R
pub struct DerivationModule<R: Ring> {
    /// Known basis derivations
    basis: Vec<Box<dyn RingDerivation<R>>>,
    _phantom: PhantomData<R>,
}

impl<R: Ring> DerivationModule<R> {
    /// Create a new derivation module
    pub fn new() -> Self {
        DerivationModule {
            basis: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add a basis derivation
    pub fn add_basis_derivation<D: RingDerivation<R> + 'static>(&mut self, d: D) {
        self.basis.push(Box::new(d));
    }

    /// Get the dimension of the module (number of basis derivations)
    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    /// Get the zero derivation
    pub fn zero(&self) -> ZeroDerivation<R> {
        ZeroDerivation::new()
    }
}

impl<R: Ring> Default for DerivationModule<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Inner derivation: D_a(x) = [a, x] = ax - xa
///
/// These are derivations induced by commutators with a fixed element
#[derive(Clone)]
pub struct InnerDerivation<R: Ring> {
    element: R,
}

impl<R: Ring> InnerDerivation<R> {
    /// Create a new inner derivation from element a
    ///
    /// This derivation maps x ↦ ax - xa
    pub fn new(element: R) -> Self {
        InnerDerivation { element }
    }
}

impl<R: Ring> RingDerivation<R> for InnerDerivation<R> {
    fn apply(&self, x: &R) -> R {
        // [a, x] = ax - xa
        let left = self.element.clone() * x.clone();
        let right = x.clone() * self.element.clone();
        left - right
    }

    fn is_zero(&self) -> bool {
        self.element.is_zero()
    }
}

impl<R: Ring> fmt::Display for InnerDerivation<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Inner derivation")
    }
}

/// Linear combination of derivations
///
/// Represents c₁D₁ + c₂D₂ + ... + cₙDₙ
pub struct LinearCombinationDerivation<R: Ring> {
    derivations: Vec<Box<dyn RingDerivation<R>>>,
    coefficients: Vec<R>,
}

impl<R: Ring> LinearCombinationDerivation<R> {
    /// Create a new linear combination
    pub fn new() -> Self {
        LinearCombinationDerivation {
            derivations: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    /// Add a term to the linear combination
    pub fn add_term<D: RingDerivation<R> + 'static>(&mut self, coeff: R, deriv: D) {
        self.coefficients.push(coeff);
        self.derivations.push(Box::new(deriv));
    }
}

impl<R: Ring> Default for LinearCombinationDerivation<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> Clone for LinearCombinationDerivation<R> {
    fn clone(&self) -> Self {
        // Note: This is a simplified clone that creates a zero derivation
        // A full implementation would require cloning boxed trait objects
        LinearCombinationDerivation {
            derivations: Vec::new(),
            coefficients: Vec::new(),
        }
    }
}

impl<R: Ring> RingDerivation<R> for LinearCombinationDerivation<R> {
    fn apply(&self, x: &R) -> R {
        let mut result = R::zero();
        for (coeff, deriv) in self.coefficients.iter().zip(self.derivations.iter()) {
            let term = coeff.clone() * deriv.apply(x);
            result = result + term;
        }
        result
    }

    fn is_zero(&self) -> bool {
        self.derivations.is_empty() ||
        self.coefficients.iter().all(|c| c.is_zero())
    }
}

impl<R: Ring> fmt::Display for LinearCombinationDerivation<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear combination of {} derivations", self.derivations.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_zero_derivation() {
        let d: ZeroDerivation<Integer> = ZeroDerivation::new();

        assert!(d.is_zero());
        assert_eq!(d.apply(&Integer::from(5)), Integer::zero());
        assert_eq!(d.apply(&Integer::from(42)), Integer::zero());
    }

    #[test]
    fn test_function_derivation() {
        // Create a "constant" derivation that maps everything to 1
        let d = FunctionDerivation::new(|_x: &Integer| Integer::from(1));

        assert_eq!(d.apply(&Integer::from(5)), Integer::from(1));
        assert_eq!(d.apply(&Integer::from(42)), Integer::from(1));
    }

    #[test]
    fn test_inner_derivation_commutative() {
        // In a commutative ring, all inner derivations are zero
        let a = Integer::from(5);
        let d = InnerDerivation::new(a);

        // For commutative rings: [a, x] = ax - xa = 0
        assert_eq!(d.apply(&Integer::from(3)), Integer::zero());
        assert_eq!(d.apply(&Integer::from(10)), Integer::zero());
    }

    #[test]
    fn test_composed_derivation() {
        // Compose two function derivations
        let d1 = FunctionDerivation::new(|x: &Integer| x.clone() + Integer::from(1));
        let d2 = FunctionDerivation::new(|x: &Integer| x.clone() * Integer::from(2));

        let composed = d2.compose(&d1);

        // d2(d1(5)) = d2(5+1) = d2(6) = 6*2 = 12
        assert_eq!(composed.apply(&Integer::from(5)), Integer::from(12));
    }

    #[test]
    fn test_derivation_module() {
        let mut module: DerivationModule<Integer> = DerivationModule::new();

        assert_eq!(module.dimension(), 0);

        module.add_basis_derivation(ZeroDerivation::new());
        assert_eq!(module.dimension(), 1);

        let zero = module.zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_linear_combination() {
        let mut lc: LinearCombinationDerivation<Integer> = LinearCombinationDerivation::new();

        // Empty linear combination is zero
        assert!(lc.is_zero());

        // Add derivations
        lc.add_term(
            Integer::from(2),
            FunctionDerivation::new(|x: &Integer| x.clone() + Integer::from(1))
        );
        lc.add_term(
            Integer::from(3),
            FunctionDerivation::new(|x: &Integer| x.clone() * Integer::from(2))
        );

        // 2*(x+1) + 3*(2x) = 2x + 2 + 6x = 8x + 2
        // For x = 5: 8*5 + 2 = 42
        assert_eq!(lc.apply(&Integer::from(5)), Integer::from(42));
    }

    #[test]
    fn test_twisted_derivation() {
        let base_deriv = FunctionDerivation::new(|x: &Integer| x.clone() + Integer::from(1));
        let twist = |x: &Integer| x.clone() * Integer::from(2);

        let twisted = TwistedDerivation::new(base_deriv, twist);

        // The derivation part still works the same
        assert_eq!(twisted.apply(&Integer::from(5)), Integer::from(6));
    }

    #[test]
    fn test_derivation_leibniz_rule() {
        // Test that a custom derivation can satisfy Leibniz rule
        // For polynomials, the derivative D(x^n) = nx^{n-1} satisfies:
        // D(x^m * x^n) = D(x^{m+n}) = (m+n)x^{m+n-1}
        // = m*x^{m-1}*x^n + x^m*n*x^{n-1} = D(x^m)*x^n + x^m*D(x^n)

        // This is a conceptual test showing the Leibniz rule structure
        let zero_d: ZeroDerivation<Integer> = ZeroDerivation::new();

        let a = Integer::from(3);
        let b = Integer::from(5);
        let ab = a.clone() * b.clone();

        // For zero derivation: D(ab) = 0 = aD(b) + D(a)b = a*0 + 0*b
        let left = zero_d.apply(&ab);
        let right_term1 = a.clone() * zero_d.apply(&b);
        let right_term2 = zero_d.apply(&a) * b.clone();
        let right = right_term1 + right_term2;

        assert_eq!(left, right);
    }

    #[test]
    fn test_derivation_additivity() {
        let d: ZeroDerivation<Integer> = ZeroDerivation::new();

        let a = Integer::from(3);
        let b = Integer::from(5);
        let sum = a.clone() + b.clone();

        // D(a + b) = D(a) + D(b)
        let left = d.apply(&sum);
        let right = d.apply(&a) + d.apply(&b);

        assert_eq!(left, right);
    }

    #[test]
    fn test_multiple_derivations() {
        let d1: ZeroDerivation<Integer> = ZeroDerivation::new();
        let d2 = FunctionDerivation::new(|x: &Integer| Integer::from(1));
        let d3 = InnerDerivation::new(Integer::from(5));

        let x = Integer::from(10);

        assert_eq!(d1.apply(&x), Integer::zero());
        assert_eq!(d2.apply(&x), Integer::one());
        assert_eq!(d3.apply(&x), Integer::zero()); // Commutative ring
    }
}
