//! Onsager Lie Algebra
//!
//! The Onsager algebra is an infinite-dimensional Lie algebra introduced by
//! Lars Onsager in 1944 in his solution of the two-dimensional Ising model.
//!
//! The Onsager algebra has two families of generators:
//! - A_n for n ∈ ℤ (primary generators)
//! - G_n for n ∈ ℤ (derived generators, appearing in brackets)
//!
//! The Lie bracket satisfies:
//! - [A_m, A_n] = G_{m-n} (for m < n, sign convention)
//! - [G_n, G_m] = 0 (G generators are central to each other)
//! - [A_m, G_n] = 2(A_{m+n} - A_{m-n})
//!
//! The Onsager algebra is isomorphic to the Chevalley involution-invariant
//! subalgebra of the affine Kac-Moody algebra of type A₁⁽¹⁾.
//!
//! Corresponds to sage.algebras.lie_algebras.onsager
//!
//! References:
//! - Onsager, L. "Crystal statistics. I. A two-dimensional model with an
//!   order-disorder transition" (1944)
//! - Davies, B. "Onsager's algebra and superintegrability" (1990)
//! - Uglov, D. & Ivanov, I. "sl(N) Onsager's algebra" (1996)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

/// Type of Onsager generator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OnsagerGeneratorType {
    /// Primary generator A_n
    A,
    /// Derived generator G_n (appears in brackets)
    G,
}

/// Onsager Lie Algebra Generator
///
/// Represents a single basis element A_n or G_n
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OnsagerGenerator {
    /// Type of generator (A or G)
    pub gen_type: OnsagerGeneratorType,
    /// Index n
    pub index: i64,
}

impl OnsagerGenerator {
    /// Create an A_n generator
    pub fn a(n: i64) -> Self {
        OnsagerGenerator {
            gen_type: OnsagerGeneratorType::A,
            index: n,
        }
    }

    /// Create a G_n generator
    pub fn g(n: i64) -> Self {
        OnsagerGenerator {
            gen_type: OnsagerGeneratorType::G,
            index: n,
        }
    }

    /// Check if this is an A generator
    pub fn is_a(&self) -> bool {
        self.gen_type == OnsagerGeneratorType::A
    }

    /// Check if this is a G generator
    pub fn is_g(&self) -> bool {
        self.gen_type == OnsagerGeneratorType::G
    }
}

impl Display for OnsagerGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.gen_type {
            OnsagerGeneratorType::A => write!(f, "A_{}", self.index),
            OnsagerGeneratorType::G => write!(f, "G_{}", self.index),
        }
    }
}

/// Element of the Onsager Lie Algebra
///
/// Represented as a formal linear combination of generators
#[derive(Debug, Clone)]
pub struct OnsagerElement<R: Ring> {
    /// Coefficients for each generator
    coeffs: HashMap<OnsagerGenerator, R>,
}

impl<R: Ring + Clone> OnsagerElement<R> {
    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        OnsagerElement {
            coeffs: HashMap::new(),
        }
    }

    /// Create generator A_n with coefficient 1
    pub fn a(n: i64) -> Self
    where
        R: From<i64>,
    {
        let mut coeffs = HashMap::new();
        coeffs.insert(OnsagerGenerator::a(n), R::from(1));
        OnsagerElement { coeffs }
    }

    /// Create generator G_n with coefficient 1
    pub fn g(n: i64) -> Self
    where
        R: From<i64>,
    {
        let mut coeffs = HashMap::new();
        coeffs.insert(OnsagerGenerator::g(n), R::from(1));
        OnsagerElement { coeffs }
    }

    /// Create an element from a single generator with given coefficient
    pub fn from_generator(gen: OnsagerGenerator, coeff: R) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(gen, coeff);
        OnsagerElement { coeffs }
    }

    /// Get the coefficient of a generator
    pub fn coeff(&self, gen: &OnsagerGenerator) -> Option<&R> {
        self.coeffs.get(gen)
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool
    where
        R: From<i64> + PartialEq,
    {
        let zero = R::from(0);
        self.coeffs.is_empty() || self.coeffs.values().all(|c| c == &zero)
    }

    /// Add a term with the given coefficient
    pub fn add_term(&mut self, gen: OnsagerGenerator, coeff: R)
    where
        R: From<i64> + Add<Output = R> + PartialEq,
    {
        let entry = self.coeffs.entry(gen).or_insert(R::from(0));
        *entry = entry.clone() + coeff.clone();

        // Remove zero coefficients
        if *entry == R::from(0) {
            self.coeffs.remove(&gen);
        }
    }

    /// Scale this element by a scalar
    pub fn scale(&self, scalar: &R) -> Self
    where
        R: Mul<Output = R>,
    {
        let mut result = HashMap::new();
        for (gen, coeff) in &self.coeffs {
            result.insert(*gen, coeff.clone() * scalar.clone());
        }
        OnsagerElement { coeffs: result }
    }

    /// Negate this element
    pub fn negate(&self) -> Self
    where
        R: Neg<Output = R>,
    {
        let mut result = HashMap::new();
        for (gen, coeff) in &self.coeffs {
            result.insert(*gen, -coeff.clone());
        }
        OnsagerElement { coeffs: result }
    }

    /// Get all generators with non-zero coefficients
    pub fn support(&self) -> Vec<OnsagerGenerator> {
        self.coeffs.keys().copied().collect()
    }

    /// Get the number of terms
    pub fn num_terms(&self) -> usize {
        self.coeffs.len()
    }
}

impl<R: Ring + Clone> Add for OnsagerElement<R>
where
    R: From<i64> + Add<Output = R> + PartialEq,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        for (gen, coeff) in other.coeffs {
            result.add_term(gen, coeff);
        }
        result
    }
}

impl<R: Ring + Clone> Sub for OnsagerElement<R>
where
    R: From<i64> + Add<Output = R> + Neg<Output = R> + PartialEq,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + other.negate()
    }
}

impl<R: Ring + Clone + Display> Display for OnsagerElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.coeffs.is_empty() {
            return write!(f, "0");
        }

        let mut first = true;
        for (gen, coeff) in &self.coeffs {
            if !first {
                write!(f, " + ")?;
            }
            write!(f, "{} * {}", coeff, gen)?;
            first = false;
        }
        Ok(())
    }
}

/// Onsager Lie Algebra
///
/// The infinite-dimensional Onsager algebra with generators A_n and G_n (n ∈ ℤ).
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::{OnsagerAlgebra, OnsagerElement};
/// let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
/// let a0 = ons.generator_a(0);
/// let a1 = ons.generator_a(1);
/// let bracket = ons.bracket(&a0, &a1);
/// // [A_0, A_1] = G_{-1}
/// ```
pub struct OnsagerAlgebra<R: Ring> {
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> OnsagerAlgebra<R> {
    /// Create a new Onsager algebra
    pub fn new() -> Self {
        OnsagerAlgebra {
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
    pub fn zero(&self) -> OnsagerElement<R>
    where
        R: From<i64>,
    {
        OnsagerElement::zero()
    }

    /// Get the generator A_n
    pub fn generator_a(&self, n: i64) -> OnsagerElement<R>
    where
        R: From<i64>,
    {
        OnsagerElement::a(n)
    }

    /// Get the generator G_n
    pub fn generator_g(&self, n: i64) -> OnsagerElement<R>
    where
        R: From<i64>,
    {
        OnsagerElement::g(n)
    }

    /// Compute the Lie bracket of two basis generators
    ///
    /// # Bracket relations:
    /// - [A_m, A_n] = -G_{n-m} for m < n (with sign convention)
    /// - [G_n, G_m] = 0 (all G generators commute)
    /// - [A_m, G_n] = -(2A_{m-n} - 2A_{m+n})
    pub fn bracket_on_basis(
        &self,
        x: &OnsagerGenerator,
        y: &OnsagerGenerator,
    ) -> OnsagerElement<R>
    where
        R: From<i64> + Add<Output = R> + Neg<Output = R> + PartialEq,
    {
        use OnsagerGeneratorType::*;

        match (x.gen_type, y.gen_type) {
            // [A_m, A_n]
            (A, A) => {
                let m = x.index;
                let n = y.index;

                if m == n {
                    // [A_m, A_m] = 0
                    OnsagerElement::zero()
                } else if m < n {
                    // [A_m, A_n] = -G_{n-m}
                    OnsagerElement::from_generator(
                        OnsagerGenerator::g(n - m),
                        R::from(-1),
                    )
                } else {
                    // [A_m, A_n] = -[A_n, A_m] = G_{m-n}
                    OnsagerElement::from_generator(
                        OnsagerGenerator::g(m - n),
                        R::from(1),
                    )
                }
            }

            // [G_n, G_m] = 0
            (G, G) => OnsagerElement::zero(),

            // [A_m, G_n] = -(2A_{m-n} - 2A_{m+n})
            (A, G) => {
                let m = x.index;
                let n = y.index;

                let mut result = OnsagerElement::zero();
                result.add_term(OnsagerGenerator::a(m - n), R::from(-2));
                result.add_term(OnsagerGenerator::a(m + n), R::from(2));
                result
            }

            // [G_n, A_m] = -[A_m, G_n]
            (G, A) => {
                let bracket = self.bracket_on_basis(
                    &OnsagerGenerator::a(y.index),
                    &OnsagerGenerator::g(x.index),
                );
                bracket.negate()
            }
        }
    }

    /// Compute the Lie bracket of two elements
    pub fn bracket(&self, x: &OnsagerElement<R>, y: &OnsagerElement<R>) -> OnsagerElement<R>
    where
        R: From<i64> + Add<Output = R> + Mul<Output = R> + Neg<Output = R> + PartialEq,
    {
        let mut result = OnsagerElement::zero();

        for (gen_x, coeff_x) in &x.coeffs {
            for (gen_y, coeff_y) in &y.coeffs {
                let basis_bracket = self.bracket_on_basis(gen_x, gen_y);
                let scaled = basis_bracket.scale(&(coeff_x.clone() * coeff_y.clone()));
                result = result + scaled;
            }
        }

        result
    }

    /// Compute the triple bracket [A_0, [A_0, [A_0, A_1]]]
    ///
    /// This should equal -4[A_0, A_1] = 4 G_1 by the Onsager relations
    pub fn check_onsager_relation(&self) -> bool
    where
        R: From<i64> + Add<Output = R> + Mul<Output = R> + Neg<Output = R> + PartialEq,
    {
        let a0 = self.generator_a(0);
        let a1 = self.generator_a(1);

        // Compute [A_0, A_1]
        let b1 = self.bracket(&a0, &a1);

        // Compute [A_0, [A_0, A_1]]
        let b2 = self.bracket(&a0, &b1);

        // Compute [A_0, [A_0, [A_0, A_1]]]
        let b3 = self.bracket(&a0, &b2);

        // Expected: -4[A_0, A_1]
        let expected = b1.scale(&R::from(-4));

        // Compare
        let diff = b3 - expected;
        diff.is_zero()
    }
}

impl<R: Ring + Clone> Default for OnsagerAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Alternating Central Extension (ACE) of the Onsager Algebra
///
/// The ACE has generators:
/// - 𝒜_k for k ∈ ℤ (type A generators)
/// - ℬ_k for k ∈ ℤ (type B generators)
///
/// With bracket relations:
/// - [𝒜_k, 𝒜_m] = ℬ_{k-m} - ℬ_{m-k}
/// - [𝒜_k, ℬ_m] = 𝒜_{k+m} - 𝒜_{k-m}
/// - [ℬ_k, ℬ_m] = 0
///
/// This is an infinite-dimensional Lie algebra that projects onto
/// the classical Onsager algebra.
pub struct OnsagerAlgebraACE<R: Ring> {
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

/// ACE Generator Type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ACEGeneratorType {
    /// Type A generator (script A)
    ScriptA,
    /// Type B generator (script B)
    ScriptB,
}

/// ACE Generator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ACEGenerator {
    /// Type of generator
    pub gen_type: ACEGeneratorType,
    /// Index
    pub index: i64,
}

impl ACEGenerator {
    /// Create a script-A generator
    pub fn script_a(k: i64) -> Self {
        ACEGenerator {
            gen_type: ACEGeneratorType::ScriptA,
            index: k,
        }
    }

    /// Create a script-B generator
    pub fn script_b(k: i64) -> Self {
        ACEGenerator {
            gen_type: ACEGeneratorType::ScriptB,
            index: k,
        }
    }
}

impl Display for ACEGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.gen_type {
            ACEGeneratorType::ScriptA => write!(f, "𝒜_{}", self.index),
            ACEGeneratorType::ScriptB => write!(f, "ℬ_{}", self.index),
        }
    }
}

/// Element of the ACE
#[derive(Debug, Clone)]
pub struct ACEElement<R: Ring> {
    /// Coefficients for each generator
    coeffs: HashMap<ACEGenerator, R>,
}

impl<R: Ring + Clone> ACEElement<R> {
    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        ACEElement {
            coeffs: HashMap::new(),
        }
    }

    /// Create script-A generator
    pub fn script_a(k: i64) -> Self
    where
        R: From<i64>,
    {
        let mut coeffs = HashMap::new();
        coeffs.insert(ACEGenerator::script_a(k), R::from(1));
        ACEElement { coeffs }
    }

    /// Create script-B generator
    pub fn script_b(k: i64) -> Self
    where
        R: From<i64>,
    {
        let mut coeffs = HashMap::new();
        coeffs.insert(ACEGenerator::script_b(k), R::from(1));
        ACEElement { coeffs }
    }

    /// Add a term
    pub fn add_term(&mut self, gen: ACEGenerator, coeff: R)
    where
        R: From<i64> + Add<Output = R> + PartialEq,
    {
        let entry = self.coeffs.entry(gen).or_insert(R::from(0));
        *entry = entry.clone() + coeff.clone();

        if *entry == R::from(0) {
            self.coeffs.remove(&gen);
        }
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: From<i64> + PartialEq,
    {
        let zero = R::from(0);
        self.coeffs.is_empty() || self.coeffs.values().all(|c| c == &zero)
    }

    /// Scale by scalar
    pub fn scale(&self, scalar: &R) -> Self
    where
        R: Mul<Output = R>,
    {
        let mut result = HashMap::new();
        for (gen, coeff) in &self.coeffs {
            result.insert(*gen, coeff.clone() * scalar.clone());
        }
        ACEElement { coeffs: result }
    }

    /// Negate
    pub fn negate(&self) -> Self
    where
        R: Neg<Output = R>,
    {
        let mut result = HashMap::new();
        for (gen, coeff) in &self.coeffs {
            result.insert(*gen, -coeff.clone());
        }
        ACEElement { coeffs: result }
    }
}

impl<R: Ring + Clone> Add for ACEElement<R>
where
    R: From<i64> + Add<Output = R> + PartialEq,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        for (gen, coeff) in other.coeffs {
            result.add_term(gen, coeff);
        }
        result
    }
}

impl<R: Ring + Clone> Sub for ACEElement<R>
where
    R: From<i64> + Add<Output = R> + Neg<Output = R> + PartialEq,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + other.negate()
    }
}

impl<R: Ring + Clone> OnsagerAlgebraACE<R> {
    /// Create a new ACE
    pub fn new() -> Self {
        OnsagerAlgebraACE {
            coefficient_ring: PhantomData,
        }
    }

    /// Check if finite dimensional (always false)
    pub fn is_finite_dimensional(&self) -> bool {
        false
    }

    /// Get the zero element
    pub fn zero(&self) -> ACEElement<R>
    where
        R: From<i64>,
    {
        ACEElement::zero()
    }

    /// Get script-A generator
    pub fn generator_script_a(&self, k: i64) -> ACEElement<R>
    where
        R: From<i64>,
    {
        ACEElement::script_a(k)
    }

    /// Get script-B generator
    pub fn generator_script_b(&self, k: i64) -> ACEElement<R>
    where
        R: From<i64>,
    {
        ACEElement::script_b(k)
    }

    /// Compute bracket on basis generators
    ///
    /// Relations:
    /// - [𝒜_k, 𝒜_m] = ℬ_{k-m} - ℬ_{m-k}
    /// - [𝒜_k, ℬ_m] = 𝒜_{k+m} - 𝒜_{k-m}
    /// - [ℬ_k, ℬ_m] = 0
    pub fn bracket_on_basis(&self, x: &ACEGenerator, y: &ACEGenerator) -> ACEElement<R>
    where
        R: From<i64> + Add<Output = R> + Neg<Output = R> + PartialEq,
    {
        use ACEGeneratorType::*;

        match (x.gen_type, y.gen_type) {
            // [𝒜_k, 𝒜_m] = ℬ_{k-m} - ℬ_{m-k}
            (ScriptA, ScriptA) => {
                let k = x.index;
                let m = y.index;

                if k == m {
                    ACEElement::zero()
                } else {
                    let mut result = ACEElement::zero();
                    result.add_term(ACEGenerator::script_b(k - m), R::from(1));
                    result.add_term(ACEGenerator::script_b(m - k), R::from(-1));
                    result
                }
            }

            // [ℬ_k, ℬ_m] = 0
            (ScriptB, ScriptB) => ACEElement::zero(),

            // [𝒜_k, ℬ_m] = 𝒜_{k+m} - 𝒜_{k-m}
            (ScriptA, ScriptB) => {
                let k = x.index;
                let m = y.index;

                let mut result = ACEElement::zero();
                result.add_term(ACEGenerator::script_a(k + m), R::from(1));
                result.add_term(ACEGenerator::script_a(k - m), R::from(-1));
                result
            }

            // [ℬ_m, 𝒜_k] = -[𝒜_k, ℬ_m]
            (ScriptB, ScriptA) => {
                let bracket = self.bracket_on_basis(
                    &ACEGenerator::script_a(y.index),
                    &ACEGenerator::script_b(x.index),
                );
                bracket.negate()
            }
        }
    }

    /// Compute the Lie bracket of two ACE elements
    pub fn bracket(&self, x: &ACEElement<R>, y: &ACEElement<R>) -> ACEElement<R>
    where
        R: From<i64> + Add<Output = R> + Mul<Output = R> + Neg<Output = R> + PartialEq,
    {
        let mut result = ACEElement::zero();

        for (gen_x, coeff_x) in &x.coeffs {
            for (gen_y, coeff_y) in &y.coeffs {
                let basis_bracket = self.bracket_on_basis(gen_x, gen_y);
                let scaled = basis_bracket.scale(&(coeff_x.clone() * coeff_y.clone()));
                result = result + scaled;
            }
        }

        result
    }

    /// Project an ACE element to the classical Onsager algebra
    ///
    /// The projection map ρ: ACE → Onsager is defined by:
    /// - ρ(𝒜_k) = A_k
    /// - ρ(ℬ_k) = 0 (the B generators form the center)
    pub fn projection(&self, x: &ACEElement<R>) -> OnsagerElement<R>
    where
        R: From<i64>,
    {
        let mut result = OnsagerElement::zero();

        for (gen, coeff) in &x.coeffs {
            if gen.gen_type == ACEGeneratorType::ScriptA {
                result.add_term(OnsagerGenerator::a(gen.index), coeff.clone());
            }
            // ScriptB generators map to zero (they're in the kernel/center)
        }

        result
    }
}

impl<R: Ring + Clone> Default for OnsagerAlgebraACE<R> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onsager_creation() {
        let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
        assert!(!ons.is_finite_dimensional());
        assert!(!ons.is_nilpotent());
        assert!(!ons.is_solvable());
    }

    #[test]
    fn test_onsager_generators() {
        let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
        let a0 = ons.generator_a(0);
        let a1 = ons.generator_a(1);
        let g1 = ons.generator_g(1);

        assert_eq!(a0.num_terms(), 1);
        assert_eq!(a1.num_terms(), 1);
        assert_eq!(g1.num_terms(), 1);
    }

    #[test]
    fn test_onsager_bracket_aa() {
        let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
        let a0 = ons.generator_a(0);
        let a1 = ons.generator_a(1);

        // [A_0, A_1] = -G_1
        let bracket = ons.bracket(&a0, &a1);
        assert_eq!(bracket.num_terms(), 1);

        let gen = OnsagerGenerator::g(1);
        assert_eq!(bracket.coeff(&gen), Some(&-1));
    }

    #[test]
    fn test_onsager_bracket_gg() {
        let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
        let g1 = ons.generator_g(1);
        let g2 = ons.generator_g(2);

        // [G_1, G_2] = 0
        let bracket = ons.bracket(&g1, &g2);
        assert!(bracket.is_zero());
    }

    #[test]
    fn test_onsager_bracket_ag() {
        let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
        let a0 = ons.generator_a(0);
        let g1 = ons.generator_g(1);

        // [A_0, G_1] = -(2A_{-1} - 2A_1)
        let bracket = ons.bracket(&a0, &g1);
        assert_eq!(bracket.num_terms(), 2);

        assert_eq!(bracket.coeff(&OnsagerGenerator::a(-1)), Some(&-2));
        assert_eq!(bracket.coeff(&OnsagerGenerator::a(1)), Some(&2));
    }

    #[test]
    fn test_onsager_antisymmetry() {
        let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
        let a0 = ons.generator_a(0);
        let a1 = ons.generator_a(1);

        let b1 = ons.bracket(&a0, &a1);
        let b2 = ons.bracket(&a1, &a0);

        // [A_0, A_1] = -[A_1, A_0]
        let sum = b1 + b2;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_onsager_relation() {
        let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
        // Check that [A_0, [A_0, [A_0, A_1]]] = -4[A_0, A_1]
        assert!(ons.check_onsager_relation());
    }

    #[test]
    fn test_ace_creation() {
        let ace: OnsagerAlgebraACE<i64> = OnsagerAlgebraACE::new();
        assert!(!ace.is_finite_dimensional());
    }

    #[test]
    fn test_ace_generators() {
        let ace: OnsagerAlgebraACE<i64> = OnsagerAlgebraACE::new();
        let a0 = ace.generator_script_a(0);
        let b1 = ace.generator_script_b(1);

        assert_eq!(a0.coeffs.len(), 1);
        assert_eq!(b1.coeffs.len(), 1);
    }

    #[test]
    fn test_ace_bracket_aa() {
        let ace: OnsagerAlgebraACE<i64> = OnsagerAlgebraACE::new();
        let a0 = ace.generator_script_a(0);
        let a1 = ace.generator_script_a(1);

        // [𝒜_0, 𝒜_1] = ℬ_{-1} - ℬ_1
        let bracket = ace.bracket(&a0, &a1);
        assert_eq!(bracket.coeffs.len(), 2);

        assert_eq!(
            bracket.coeffs.get(&ACEGenerator::script_b(-1)),
            Some(&1)
        );
        assert_eq!(
            bracket.coeffs.get(&ACEGenerator::script_b(1)),
            Some(&-1)
        );
    }

    #[test]
    fn test_ace_bracket_bb() {
        let ace: OnsagerAlgebraACE<i64> = OnsagerAlgebraACE::new();
        let b0 = ace.generator_script_b(0);
        let b1 = ace.generator_script_b(1);

        // [ℬ_0, ℬ_1] = 0
        let bracket = ace.bracket(&b0, &b1);
        assert!(bracket.is_zero());
    }

    #[test]
    fn test_ace_bracket_ab() {
        let ace: OnsagerAlgebraACE<i64> = OnsagerAlgebraACE::new();
        let a0 = ace.generator_script_a(0);
        let b1 = ace.generator_script_b(1);

        // [𝒜_0, ℬ_1] = 𝒜_1 - 𝒜_{-1}
        let bracket = ace.bracket(&a0, &b1);
        assert_eq!(bracket.coeffs.len(), 2);

        assert_eq!(
            bracket.coeffs.get(&ACEGenerator::script_a(1)),
            Some(&1)
        );
        assert_eq!(
            bracket.coeffs.get(&ACEGenerator::script_a(-1)),
            Some(&-1)
        );
    }

    #[test]
    fn test_ace_projection() {
        let ace: OnsagerAlgebraACE<i64> = OnsagerAlgebraACE::new();

        // Create element 𝒜_0 + 2𝒜_1 + 3ℬ_0
        let mut x = ace.generator_script_a(0);
        x.add_term(ACEGenerator::script_a(1), 2);
        x.add_term(ACEGenerator::script_b(0), 3);

        // Project to Onsager algebra
        let proj = ace.projection(&x);

        // Should get A_0 + 2A_1 (ℬ_0 maps to 0)
        assert_eq!(proj.num_terms(), 2);
        assert_eq!(proj.coeff(&OnsagerGenerator::a(0)), Some(&1));
        assert_eq!(proj.coeff(&OnsagerGenerator::a(1)), Some(&2));
    }

    #[test]
    fn test_element_addition() {
        let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
        let a0 = ons.generator_a(0);
        let a1 = ons.generator_a(1);

        let sum = a0.clone() + a1.clone();
        assert_eq!(sum.num_terms(), 2);

        let sum_self = a0.clone() + a0.clone();
        assert_eq!(sum_self.num_terms(), 1);
        assert_eq!(sum_self.coeff(&OnsagerGenerator::a(0)), Some(&2));
    }

    #[test]
    fn test_element_scaling() {
        let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
        let a0 = ons.generator_a(0);

        let scaled = a0.scale(&3);
        assert_eq!(scaled.coeff(&OnsagerGenerator::a(0)), Some(&3));
    }

    #[test]
    fn test_jacobi_identity() {
        let ons: OnsagerAlgebra<i64> = OnsagerAlgebra::new();
        let a0 = ons.generator_a(0);
        let a1 = ons.generator_a(1);
        let g1 = ons.generator_g(1);

        // [[a0, a1], g1] + [[a1, g1], a0] + [[g1, a0], a1] = 0
        let b1 = ons.bracket(&a0, &a1);
        let b2 = ons.bracket(&b1, &g1);

        let b3 = ons.bracket(&a1, &g1);
        let b4 = ons.bracket(&b3, &a0);

        let b5 = ons.bracket(&g1, &a0);
        let b6 = ons.bracket(&b5, &a1);

        let total = (b2 + b4) + b6;
        assert!(total.is_zero());
    }
}
