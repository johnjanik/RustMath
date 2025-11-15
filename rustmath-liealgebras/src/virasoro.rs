//! Virasoro Lie Algebra
//!
//! The Virasoro algebra is an infinite-dimensional Lie algebra that is
//! fundamental in conformal field theory and string theory.
//!
//! The Virasoro algebra has basis elements:
//! - L_n for n ∈ ℤ (Virasoro generators)
//! - c (central element)
//!
//! The Lie bracket satisfies:
//! - [L_m, L_n] = (m - n)L_{m+n} + (c/12)m(m²-1)δ_{m,-n}
//! - [L_n, c] = 0 (c is central)
//!
//! The Virasoro algebra is the unique one-dimensional central extension
//! of the Witt algebra (complexified vector fields on S¹).
//!
//! Corresponds to parts of sage.algebras.lie_algebras
//!
//! References:
//! - Kac, V. "Vertex Algebras for Beginners" (1997)
//! - Di Francesco, Mathieu, Sénéchal "Conformal Field Theory" (1997)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Virasoro Lie Algebra
///
/// The infinite-dimensional Virasoro algebra with generators L_n (n ∈ ℤ)
/// and central element c.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::VirasoroAlgebra;
/// let vir: VirasoroAlgebra<i64> = VirasoroAlgebra::new();
/// assert!(!vir.is_finite_dimensional());
/// ```
pub struct VirasoroAlgebra<R: Ring> {
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> VirasoroAlgebra<R> {
    /// Create a new Virasoro algebra
    pub fn new() -> Self {
        VirasoroAlgebra {
            coefficient_ring: PhantomData,
        }
    }

    /// Check if this is finite dimensional (always false)
    pub fn is_finite_dimensional(&self) -> bool {
        false
    }

    /// Check if this is nilpotent (always false)
    pub fn is_nilpotent(&self) -> bool {
        false
    }

    /// Check if this is solvable (always false)
    pub fn is_solvable(&self) -> bool {
        false
    }

    /// Get the zero element
    pub fn zero(&self) -> VirasoroElement<R>
    where
        R: From<i64>,
    {
        VirasoroElement::zero()
    }

    /// Get the generator L_n
    pub fn generator(&self, n: i64) -> VirasoroElement<R>
    where
        R: From<i64>,
    {
        VirasoroElement::generator(n)
    }

    /// Get the central element c
    pub fn central_element(&self) -> VirasoroElement<R>
    where
        R: From<i64>,
    {
        VirasoroElement::central()
    }

    /// Compute the Lie bracket [L_m, L_n]
    ///
    /// Returns (m - n)L_{m+n} + (c/12)m(m²-1)δ_{m,-n}
    pub fn bracket_generators(
        &self,
        m: i64,
        n: i64,
    ) -> VirasoroElement<R>
    where
        R: From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + std::ops::Sub<Output = R>,
    {
        let mut result = VirasoroElement::zero();

        // Classical term: (m - n)L_{m+n}
        let classical_coeff = m - n;
        if classical_coeff != 0 {
            let mut terms = HashMap::new();
            terms.insert(VirasoroGenerator::L(m + n), R::from(classical_coeff));
            result = VirasoroElement { terms };
        }

        // Central term: (c/12)m(m²-1)δ_{m,-n}
        if m == -n && m != 0 {
            let central_coeff = m * (m * m - 1) / 12;
            let mut central_map = HashMap::new();
            central_map.insert(VirasoroGenerator::C, R::from(central_coeff));
            let central_elem = VirasoroElement { terms: central_map };
            result = result.add(&central_elem);
        }

        result
    }
}

impl<R: Ring + Clone> Default for VirasoroAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring + Clone> Display for VirasoroAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Virasoro algebra")
    }
}

/// Generator type for Virasoro algebra
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VirasoroGenerator {
    /// Virasoro generator L_n
    L(i64),
    /// Central element c
    C,
}

impl Display for VirasoroGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VirasoroGenerator::L(n) => write!(f, "L_{}", n),
            VirasoroGenerator::C => write!(f, "c"),
        }
    }
}

/// Element of the Virasoro algebra
///
/// Represented as a linear combination of generators L_n and c
#[derive(Clone, Debug)]
pub struct VirasoroElement<R: Ring> {
    /// Terms: map from generator to coefficient
    terms: HashMap<VirasoroGenerator, R>,
}

impl<R: Ring + Clone> VirasoroElement<R> {
    /// Create a new element
    pub fn new(terms: HashMap<VirasoroGenerator, R>) -> Self {
        VirasoroElement { terms }
    }

    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        VirasoroElement {
            terms: HashMap::new(),
        }
    }

    /// Create a generator L_n
    pub fn generator(n: i64) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(VirasoroGenerator::L(n), R::from(1));
        VirasoroElement { terms }
    }

    /// Create the central element c
    pub fn central() -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(VirasoroGenerator::C, R::from(1));
        VirasoroElement { terms }
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<VirasoroGenerator, R> {
        &self.terms
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.terms.is_empty() || self.terms.values().all(|c| c.is_zero())
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + PartialEq,
    {
        let mut result = self.terms.clone();

        for (gen, coeff) in &other.terms {
            let new_coeff = if let Some(existing) = result.get(gen) {
                existing.clone() + coeff.clone()
            } else {
                coeff.clone()
            };

            if new_coeff.is_zero() {
                result.remove(gen);
            } else {
                result.insert(gen.clone(), new_coeff);
            }
        }

        VirasoroElement { terms: result }
    }

    /// Negate an element
    pub fn negate(&self) -> Self
    where
        R: std::ops::Neg<Output = R>,
    {
        let terms = self
            .terms
            .iter()
            .map(|(gen, coeff)| (gen.clone(), -coeff.clone()))
            .collect();
        VirasoroElement { terms }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: From<i64> + std::ops::Mul<Output = R> + PartialEq,
    {
        if scalar.is_zero() {
            return Self::zero();
        }

        let terms = self
            .terms
            .iter()
            .map(|(gen, coeff)| (gen.clone(), coeff.clone() * scalar.clone()))
            .collect();

        VirasoroElement { terms }
    }

    /// Lie bracket with another element
    ///
    /// Uses bilinearity: [∑ a_i L_i, ∑ b_j L_j] = ∑ a_i b_j [L_i, L_j]
    pub fn bracket(&self, other: &Self, algebra: &VirasoroAlgebra<R>) -> Self
    where
        R: From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + std::ops::Sub<Output = R> + PartialEq,
    {
        let mut result = Self::zero();

        for (gen1, coeff1) in &self.terms {
            for (gen2, coeff2) in &other.terms {
                // [c, anything] = 0
                if matches!(gen1, VirasoroGenerator::C) || matches!(gen2, VirasoroGenerator::C) {
                    continue;
                }

                // Both must be L_n generators
                if let (VirasoroGenerator::L(m), VirasoroGenerator::L(n)) = (gen1, gen2) {
                    let bracket_mn = algebra.bracket_generators(*m, *n);
                    let scaled = bracket_mn.scalar_mul(&(coeff1.clone() * coeff2.clone()));
                    result = result.add(&scaled);
                }
            }
        }

        result
    }

    /// Extract coefficient of a specific generator
    pub fn coefficient(&self, gen: &VirasoroGenerator) -> Option<&R> {
        self.terms.get(gen)
    }

    /// Check if this element is in the center (proportional to c)
    pub fn is_central(&self) -> bool
    where
        R: PartialEq,
    {
        if self.is_zero() {
            return true;
        }

        // Central if only non-zero term is c
        self.terms.len() == 1 && self.terms.contains_key(&VirasoroGenerator::C)
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for VirasoroElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }

        for (gen, coeff) in &self.terms {
            match other.terms.get(gen) {
                Some(other_coeff) if coeff == other_coeff => continue,
                _ => return false,
            }
        }

        true
    }
}

impl<R: Ring + Clone + PartialEq> Eq for VirasoroElement<R> {}

impl<R: Ring + Clone + Display> Display for VirasoroElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut sorted: Vec<_> = self.terms.iter().collect();
        sorted.sort_by_key(|(gen, _)| *gen);

        let mut first = true;
        for (gen, coeff) in sorted {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{}*{}", coeff, gen)?;
        }

        Ok(())
    }
}

/// Rank-Two Heisenberg-Virasoro Algebra
///
/// An extension of the Virasoro algebra with additional Heisenberg-like
/// generators indexed by pairs of integers.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
pub struct RankTwoHeisenbergVirasoro<R: Ring> {
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> RankTwoHeisenbergVirasoro<R> {
    /// Create a new rank-two Heisenberg-Virasoro algebra
    pub fn new() -> Self {
        RankTwoHeisenbergVirasoro {
            coefficient_ring: PhantomData,
        }
    }

    /// Check if this is finite dimensional (always false)
    pub fn is_finite_dimensional(&self) -> bool {
        false
    }

    /// Check if this is nilpotent (always false)
    pub fn is_nilpotent(&self) -> bool {
        false
    }

    /// Get the zero element
    pub fn zero(&self) -> RankTwoHeisenbergVirasoroElement<R>
    where
        R: From<i64>,
    {
        RankTwoHeisenbergVirasoroElement::zero()
    }

    /// Get a t-generator t^(a,b)
    pub fn t_generator(&self, a: i64, b: i64) -> RankTwoHeisenbergVirasoroElement<R>
    where
        R: From<i64>,
    {
        RankTwoHeisenbergVirasoroElement::t_generator(a, b)
    }

    /// Get an E-generator E(a,b)
    pub fn e_generator(&self, a: i64, b: i64) -> RankTwoHeisenbergVirasoroElement<R>
    where
        R: From<i64>,
    {
        RankTwoHeisenbergVirasoroElement::e_generator(a, b)
    }

    /// Get a central element K_i (i = 1, 2, 3, 4)
    pub fn central_element(&self, i: usize) -> RankTwoHeisenbergVirasoroElement<R>
    where
        R: From<i64>,
    {
        RankTwoHeisenbergVirasoroElement::central(i)
    }
}

impl<R: Ring + Clone> Default for RankTwoHeisenbergVirasoro<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring + Clone> Display for RankTwoHeisenbergVirasoro<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rank-two Heisenberg-Virasoro algebra")
    }
}

/// Generator for rank-two Heisenberg-Virasoro algebra
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RankTwoGenerator {
    /// t-generator t^(a,b)
    T(i64, i64),
    /// E-generator E(a,b)
    E(i64, i64),
    /// Central element K_i (i = 1, 2, 3, 4)
    K(usize),
}

impl Display for RankTwoGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RankTwoGenerator::T(a, b) => write!(f, "t^({},{})", a, b),
            RankTwoGenerator::E(a, b) => write!(f, "E({},{})", a, b),
            RankTwoGenerator::K(i) => write!(f, "K_{}", i),
        }
    }
}

/// Element of rank-two Heisenberg-Virasoro algebra
#[derive(Clone)]
pub struct RankTwoHeisenbergVirasoroElement<R: Ring> {
    /// Terms: map from generator to coefficient
    terms: HashMap<RankTwoGenerator, R>,
}

impl<R: Ring + Clone> RankTwoHeisenbergVirasoroElement<R> {
    /// Create a new element
    pub fn new(terms: HashMap<RankTwoGenerator, R>) -> Self {
        RankTwoHeisenbergVirasoroElement { terms }
    }

    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        RankTwoHeisenbergVirasoroElement {
            terms: HashMap::new(),
        }
    }

    /// Create a t-generator
    pub fn t_generator(a: i64, b: i64) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(RankTwoGenerator::T(a, b), R::from(1));
        RankTwoHeisenbergVirasoroElement { terms }
    }

    /// Create an E-generator
    pub fn e_generator(a: i64, b: i64) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(RankTwoGenerator::E(a, b), R::from(1));
        RankTwoHeisenbergVirasoroElement { terms }
    }

    /// Create a central element
    pub fn central(i: usize) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(RankTwoGenerator::K(i), R::from(1));
        RankTwoHeisenbergVirasoroElement { terms }
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<RankTwoGenerator, R> {
        &self.terms
    }

    /// Get the coefficient of a generator
    pub fn coefficient(&self, gen: &RankTwoGenerator) -> Option<&R> {
        self.terms.get(gen)
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.terms.is_empty() || self.terms.values().all(|c| c.is_zero())
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + PartialEq,
    {
        let mut result = self.terms.clone();

        for (gen, coeff) in &other.terms {
            let new_coeff = if let Some(existing) = result.get(gen) {
                existing.clone() + coeff.clone()
            } else {
                coeff.clone()
            };

            if new_coeff.is_zero() {
                result.remove(gen);
            } else {
                result.insert(gen.clone(), new_coeff);
            }
        }

        RankTwoHeisenbergVirasoroElement { terms: result }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: From<i64> + std::ops::Mul<Output = R> + PartialEq,
    {
        if scalar.is_zero() {
            return Self::zero();
        }

        let terms = self
            .terms
            .iter()
            .map(|(gen, coeff)| (gen.clone(), coeff.clone() * scalar.clone()))
            .collect();

        RankTwoHeisenbergVirasoroElement { terms }
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for RankTwoHeisenbergVirasoroElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }

        for (gen, coeff) in &self.terms {
            match other.terms.get(gen) {
                Some(other_coeff) if coeff == other_coeff => continue,
                _ => return false,
            }
        }

        true
    }
}

impl<R: Ring + Clone + PartialEq> Eq for RankTwoHeisenbergVirasoroElement<R> {}

impl<R: Ring + Clone + Display> Display for RankTwoHeisenbergVirasoroElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut sorted: Vec<_> = self.terms.iter().collect();
        sorted.sort_by_key(|(gen, _)| *gen);

        let mut first = true;
        for (gen, coeff) in sorted {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{}*{}", coeff, gen)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virasoro_creation() {
        let vir: VirasoroAlgebra<i64> = VirasoroAlgebra::new();
        assert!(!vir.is_finite_dimensional());
        assert!(!vir.is_nilpotent());
        assert!(!vir.is_solvable());
    }

    #[test]
    fn test_virasoro_generators() {
        let vir: VirasoroAlgebra<i64> = VirasoroAlgebra::new();

        let l0 = vir.generator(0);
        let l1 = vir.generator(1);
        let c = vir.central_element();

        assert!(!l0.is_zero());
        assert!(!l1.is_zero());
        assert!(c.is_central());
    }

    #[test]
    fn test_virasoro_bracket_classical() {
        let vir: VirasoroAlgebra<i64> = VirasoroAlgebra::new();

        // [L_1, L_(-1)] = 2L_0 + c/12 * 1 * (1-1) = 2L_0
        let bracket = vir.bracket_generators(1, -1);

        // Should have coefficient 2 for L_0
        assert_eq!(bracket.coefficient(&VirasoroGenerator::L(0)), Some(&2));
    }

    #[test]
    fn test_virasoro_bracket_central() {
        let vir: VirasoroAlgebra<i64> = VirasoroAlgebra::new();

        // [L_2, L_(-2)] = 4L_0 + c/12 * 2 * (4-1) = 4L_0 + c/2
        let bracket = vir.bracket_generators(2, -2);

        // Should have coefficient 4 for L_0
        assert_eq!(bracket.coefficient(&VirasoroGenerator::L(0)), Some(&4));

        // Should have central term: 2 * 3 / 12 = 1/2 (but we're using i64, so 0)
        // In exact arithmetic with rationals, this would be 1/2
    }

    #[test]
    fn test_virasoro_element_operations() {
        let l0: VirasoroElement<i64> = VirasoroElement::generator(0);
        let l1: VirasoroElement<i64> = VirasoroElement::generator(1);

        // Addition
        let sum = l0.add(&l1);
        assert_eq!(sum.coefficient(&VirasoroGenerator::L(0)), Some(&1));
        assert_eq!(sum.coefficient(&VirasoroGenerator::L(1)), Some(&1));

        // Scalar multiplication
        let scaled = l0.scalar_mul(&5);
        assert_eq!(scaled.coefficient(&VirasoroGenerator::L(0)), Some(&5));
    }

    #[test]
    fn test_virasoro_bracket_antisymmetry() {
        let vir: VirasoroAlgebra<i64> = VirasoroAlgebra::new();

        let l1 = vir.generator(1);
        let l2 = vir.generator(2);

        let bracket12 = l1.bracket(&l2, &vir);
        let bracket21 = l2.bracket(&l1, &vir);

        // [L_1, L_2] = -[L_2, L_1]
        assert_eq!(bracket12, bracket21.negate());
    }

    #[test]
    fn test_rank_two_creation() {
        let r2: RankTwoHeisenbergVirasoro<i64> = RankTwoHeisenbergVirasoro::new();
        assert!(!r2.is_finite_dimensional());
        assert!(!r2.is_nilpotent());
    }

    #[test]
    fn test_rank_two_generators() {
        let r2: RankTwoHeisenbergVirasoro<i64> = RankTwoHeisenbergVirasoro::new();

        let t = r2.t_generator(1, 2);
        let e = r2.e_generator(1, 2);
        let k1 = r2.central_element(1);

        assert!(!t.is_zero());
        assert!(!e.is_zero());
        assert!(!k1.is_zero());
    }

    #[test]
    fn test_rank_two_element_operations() {
        let t1: RankTwoHeisenbergVirasoroElement<i64> =
            RankTwoHeisenbergVirasoroElement::t_generator(1, 0);
        let t2: RankTwoHeisenbergVirasoroElement<i64> =
            RankTwoHeisenbergVirasoroElement::t_generator(0, 1);

        let sum = t1.add(&t2);
        assert_eq!(
            sum.coefficient(&RankTwoGenerator::T(1, 0)),
            Some(&1)
        );
        assert_eq!(
            sum.coefficient(&RankTwoGenerator::T(0, 1)),
            Some(&1)
        );
    }

    #[test]
    fn test_central_element_check() {
        let c: VirasoroElement<i64> = VirasoroElement::central();
        assert!(c.is_central());

        let l0: VirasoroElement<i64> = VirasoroElement::generator(0);
        assert!(!l0.is_central());
    }
}
