//! Axioms for category theory and algebraic structures
//!
//! This module defines traits representing fundamental mathematical axioms
//! that algebraic structures must satisfy. These axioms can be used to:
//! - Verify structure properties at compile time
//! - Document mathematical requirements
//! - Enable trait-based generic algorithms
//!
//! # Mathematical Background
//!
//! Axioms describe the fundamental properties that operations must satisfy:
//! - **Associativity**: (a ∘ b) ∘ c = a ∘ (b ∘ c)
//! - **Commutativity**: a ∘ b = b ∘ a
//! - **Identity**: ∃e: e ∘ a = a ∘ e = a
//! - **Inverse**: ∀a ∃a⁻¹: a ∘ a⁻¹ = a⁻¹ ∘ a = e
//! - **Distributivity**: a ∘ (b + c) = (a ∘ b) + (a ∘ c)

use std::fmt;

/// Base trait for all axioms
///
/// An axiom represents a property that must hold for an algebraic structure
pub trait Axiom: fmt::Debug {
    /// Get the name of this axiom
    fn name(&self) -> &str;

    /// Get a mathematical description of this axiom
    fn description(&self) -> String {
        format!("Axiom: {}", self.name())
    }

    /// Get the symbolic representation (e.g., "a ∘ b = b ∘ a")
    fn symbolic(&self) -> &str {
        ""
    }
}

/// Associativity axiom: (a ∘ b) ∘ c = a ∘ (b ∘ c)
///
/// An operation ∘ is associative if the grouping of operations doesn't matter.
///
/// # Examples
/// - Addition: (1 + 2) + 3 = 1 + (2 + 3) = 6
/// - Multiplication: (2 × 3) × 4 = 2 × (3 × 4) = 24
/// - Function composition: (f ∘ g) ∘ h = f ∘ (g ∘ h)
///
/// # Usage
/// ```
/// use rustmath_category::axioms::{Axiom, Associativity};
///
/// let assoc = Associativity;
/// assert_eq!(assoc.name(), "associativity");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Associativity;

impl Axiom for Associativity {
    fn name(&self) -> &str {
        "associativity"
    }

    fn description(&self) -> String {
        "The operation is associative: (a ∘ b) ∘ c = a ∘ (b ∘ c)".to_string()
    }

    fn symbolic(&self) -> &str {
        "(a ∘ b) ∘ c = a ∘ (b ∘ c)"
    }
}

/// Commutativity axiom: a ∘ b = b ∘ a
///
/// An operation ∘ is commutative if the order of operands doesn't matter.
///
/// # Examples
/// - Addition: 2 + 3 = 3 + 2 = 5
/// - Multiplication: 4 × 5 = 5 × 4 = 20
/// - Set union: A ∪ B = B ∪ A
///
/// # Non-examples
/// - Subtraction: 5 - 3 ≠ 3 - 5
/// - Matrix multiplication: AB ≠ BA (in general)
/// - Function composition: f ∘ g ≠ g ∘ f (in general)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Commutativity;

impl Axiom for Commutativity {
    fn name(&self) -> &str {
        "commutativity"
    }

    fn description(&self) -> String {
        "The operation is commutative: a ∘ b = b ∘ a".to_string()
    }

    fn symbolic(&self) -> &str {
        "a ∘ b = b ∘ a"
    }
}

/// Identity axiom: ∃e such that e ∘ a = a ∘ e = a for all a
///
/// An identity element is a special element that leaves other elements unchanged.
///
/// # Examples
/// - Addition: 0 is the identity (a + 0 = 0 + a = a)
/// - Multiplication: 1 is the identity (a × 1 = 1 × a = a)
/// - Function composition: id is the identity (f ∘ id = id ∘ f = f)
/// - Matrix multiplication: I (identity matrix)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Identity;

impl Axiom for Identity {
    fn name(&self) -> &str {
        "identity"
    }

    fn description(&self) -> String {
        "There exists an identity element e: e ∘ a = a ∘ e = a".to_string()
    }

    fn symbolic(&self) -> &str {
        "∃e: e ∘ a = a ∘ e = a"
    }
}

/// Unity axiom: multiplicative identity exists
///
/// Similar to Identity but specifically for multiplicative operations.
/// The unity element is typically denoted as 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unity;

impl Axiom for Unity {
    fn name(&self) -> &str {
        "unity"
    }

    fn description(&self) -> String {
        "There exists a multiplicative identity 1: 1 × a = a × 1 = a".to_string()
    }

    fn symbolic(&self) -> &str {
        "∃1: 1 × a = a × 1 = a"
    }
}

/// Inverse axiom: ∀a ∃a⁻¹ such that a ∘ a⁻¹ = a⁻¹ ∘ a = e
///
/// For every element, there exists an inverse that combines with it to give the identity.
///
/// # Examples
/// - Additive inverse: For any number a, -a is its inverse (a + (-a) = 0)
/// - Multiplicative inverse: For any nonzero number a, 1/a is its inverse (a × 1/a = 1)
/// - Group inverse: Every element has an inverse in a group
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Inverse;

impl Axiom for Inverse {
    fn name(&self) -> &str {
        "inverse"
    }

    fn description(&self) -> String {
        "Every element has an inverse: ∀a ∃a⁻¹: a ∘ a⁻¹ = a⁻¹ ∘ a = e".to_string()
    }

    fn symbolic(&self) -> &str {
        "∀a ∃a⁻¹: a ∘ a⁻¹ = a⁻¹ ∘ a = e"
    }
}

/// Distributivity axiom: a ∘ (b + c) = (a ∘ b) + (a ∘ c)
///
/// One operation distributes over another.
///
/// # Examples
/// - Multiplication over addition: 2 × (3 + 4) = (2 × 3) + (2 × 4) = 14
/// - Set intersection over union: A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Distributivity;

impl Axiom for Distributivity {
    fn name(&self) -> &str {
        "distributivity"
    }

    fn description(&self) -> String {
        "One operation distributes over another: a ∘ (b + c) = (a ∘ b) + (a ∘ c)".to_string()
    }

    fn symbolic(&self) -> &str {
        "a ∘ (b + c) = (a ∘ b) + (a ∘ c)"
    }
}

/// Closure axiom: a ∘ b is in the set for all a, b in the set
///
/// The operation always produces an element within the same set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Closure;

impl Axiom for Closure {
    fn name(&self) -> &str {
        "closure"
    }

    fn description(&self) -> String {
        "The operation is closed: ∀a,b ∈ S: a ∘ b ∈ S".to_string()
    }

    fn symbolic(&self) -> &str {
        "∀a,b ∈ S: a ∘ b ∈ S"
    }
}

/// Idempotence axiom: a ∘ a = a
///
/// An operation is idempotent if applying it multiple times has the same effect as applying it once.
///
/// # Examples
/// - Set union: A ∪ A = A
/// - Set intersection: A ∩ A = A
/// - Boolean operations: true OR true = true
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Idempotence;

impl Axiom for Idempotence {
    fn name(&self) -> &str {
        "idempotence"
    }

    fn description(&self) -> String {
        "The operation is idempotent: a ∘ a = a".to_string()
    }

    fn symbolic(&self) -> &str {
        "a ∘ a = a"
    }
}

/// Absorption axiom: a ∘ (a + b) = a
///
/// Common in lattice theory.
///
/// # Examples
/// - Boolean algebra: a AND (a OR b) = a
/// - Set operations: A ∩ (A ∪ B) = A
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Absorption;

impl Axiom for Absorption {
    fn name(&self) -> &str {
        "absorption"
    }

    fn description(&self) -> String {
        "The absorption law holds: a ∘ (a + b) = a".to_string()
    }

    fn symbolic(&self) -> &str {
        "a ∘ (a + b) = a"
    }
}

/// No zero divisors: if a ∘ b = 0, then a = 0 or b = 0
///
/// This is a key property of integral domains.
///
/// # Examples
/// - Integers: If a × b = 0, then a = 0 or b = 0
/// - Real numbers: No zero divisors
///
/// # Non-examples
/// - Z/6Z: 2 × 3 = 0 (mod 6), but 2 ≠ 0 and 3 ≠ 0
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NoZeroDivisors;

impl Axiom for NoZeroDivisors {
    fn name(&self) -> &str {
        "no_zero_divisors"
    }

    fn description(&self) -> String {
        "No zero divisors: if a × b = 0, then a = 0 or b = 0".to_string()
    }

    fn symbolic(&self) -> &str {
        "a × b = 0 ⟹ a = 0 ∨ b = 0"
    }
}

/// A collection of axioms that defines an algebraic structure
///
/// This allows grouping multiple axioms together to define structures like
/// groups, rings, fields, etc.
#[derive(Debug, Clone)]
pub struct AxiomSet {
    axioms: Vec<Box<dyn Axiom>>,
}

impl AxiomSet {
    /// Create a new empty axiom set
    pub fn new() -> Self {
        AxiomSet { axioms: vec![] }
    }

    /// Add an axiom to the set
    pub fn add_axiom<A: Axiom + 'static>(&mut self, axiom: A) {
        self.axioms.push(Box::new(axiom));
    }

    /// Get all axioms in the set
    pub fn axioms(&self) -> &[Box<dyn Axiom>] {
        &self.axioms
    }

    /// Get axiom names
    pub fn axiom_names(&self) -> Vec<&str> {
        self.axioms.iter().map(|a| a.name()).collect()
    }

    /// Create an axiom set for a group
    ///
    /// Groups satisfy: closure, associativity, identity, inverse
    pub fn group() -> Self {
        let mut set = Self::new();
        set.add_axiom(Closure);
        set.add_axiom(Associativity);
        set.add_axiom(Identity);
        set.add_axiom(Inverse);
        set
    }

    /// Create an axiom set for an abelian group
    ///
    /// Abelian groups satisfy: closure, associativity, commutativity, identity, inverse
    pub fn abelian_group() -> Self {
        let mut set = Self::group();
        set.add_axiom(Commutativity);
        set
    }

    /// Create an axiom set for a ring
    ///
    /// Rings have additive group structure, multiplicative associativity, and distributivity
    pub fn ring() -> Self {
        let mut set = Self::new();
        // Additive group axioms
        set.add_axiom(Closure);
        set.add_axiom(Associativity);
        set.add_axiom(Identity);
        set.add_axiom(Inverse);
        set.add_axiom(Commutativity);
        // Multiplicative and ring axioms
        set.add_axiom(Distributivity);
        set
    }

    /// Create an axiom set for an integral domain
    ///
    /// Integral domains are commutative rings with unity and no zero divisors
    pub fn integral_domain() -> Self {
        let mut set = Self::ring();
        set.add_axiom(Unity);
        set.add_axiom(NoZeroDivisors);
        set
    }

    /// Create an axiom set for a field
    ///
    /// Fields are commutative rings where every nonzero element has a multiplicative inverse
    pub fn field() -> Self {
        let mut set = Self::integral_domain();
        // Multiplicative inverse exists for all nonzero elements
        // (This is implicit in the Field trait, but we note it here)
        set
    }

    /// Check if an axiom is in the set
    pub fn has_axiom(&self, name: &str) -> bool {
        self.axioms.iter().any(|a| a.name() == name)
    }
}

impl Default for AxiomSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AxiomSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Axiom Set ({} axioms):", self.axioms.len())?;
        for axiom in &self.axioms {
            writeln!(f, "  - {}: {}", axiom.name(), axiom.symbolic())?;
        }
        Ok(())
    }
}

/// Trait for types that satisfy a specific axiom
///
/// This allows compile-time verification of axiom satisfaction
pub trait SatisfiesAxiom<A: Axiom> {
    /// Verify that this type satisfies the axiom (runtime check for testing)
    fn verify_axiom(&self) -> bool {
        true // Default: assume satisfaction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_associativity_axiom() {
        let axiom = Associativity;
        assert_eq!(axiom.name(), "associativity");
        assert!(axiom.description().contains("associative"));
        assert_eq!(axiom.symbolic(), "(a ∘ b) ∘ c = a ∘ (b ∘ c)");
    }

    #[test]
    fn test_commutativity_axiom() {
        let axiom = Commutativity;
        assert_eq!(axiom.name(), "commutativity");
        assert_eq!(axiom.symbolic(), "a ∘ b = b ∘ a");
    }

    #[test]
    fn test_identity_axiom() {
        let axiom = Identity;
        assert_eq!(axiom.name(), "identity");
        assert!(axiom.symbolic().contains("∃e"));
    }

    #[test]
    fn test_unity_axiom() {
        let axiom = Unity;
        assert_eq!(axiom.name(), "unity");
        assert!(axiom.symbolic().contains("∃1"));
    }

    #[test]
    fn test_inverse_axiom() {
        let axiom = Inverse;
        assert_eq!(axiom.name(), "inverse");
        assert!(axiom.symbolic().contains("a⁻¹"));
    }

    #[test]
    fn test_distributivity_axiom() {
        let axiom = Distributivity;
        assert_eq!(axiom.name(), "distributivity");
        assert!(axiom.symbolic().contains("(b + c)"));
    }

    #[test]
    fn test_closure_axiom() {
        let axiom = Closure;
        assert_eq!(axiom.name(), "closure");
        assert!(axiom.symbolic().contains("∈ S"));
    }

    #[test]
    fn test_idempotence_axiom() {
        let axiom = Idempotence;
        assert_eq!(axiom.name(), "idempotence");
        assert_eq!(axiom.symbolic(), "a ∘ a = a");
    }

    #[test]
    fn test_absorption_axiom() {
        let axiom = Absorption;
        assert_eq!(axiom.name(), "absorption");
    }

    #[test]
    fn test_no_zero_divisors_axiom() {
        let axiom = NoZeroDivisors;
        assert_eq!(axiom.name(), "no_zero_divisors");
        assert!(axiom.symbolic().contains("⟹"));
    }

    #[test]
    fn test_axiom_set_creation() {
        let mut set = AxiomSet::new();
        set.add_axiom(Associativity);
        set.add_axiom(Commutativity);

        assert_eq!(set.axioms().len(), 2);
        assert!(set.has_axiom("associativity"));
        assert!(set.has_axiom("commutativity"));
        assert!(!set.has_axiom("identity"));
    }

    #[test]
    fn test_group_axiom_set() {
        let group = AxiomSet::group();
        let names = group.axiom_names();

        assert!(names.contains(&"closure"));
        assert!(names.contains(&"associativity"));
        assert!(names.contains(&"identity"));
        assert!(names.contains(&"inverse"));
        assert!(!names.contains(&"commutativity"));
    }

    #[test]
    fn test_abelian_group_axiom_set() {
        let abelian = AxiomSet::abelian_group();
        let names = abelian.axiom_names();

        assert!(names.contains(&"closure"));
        assert!(names.contains(&"associativity"));
        assert!(names.contains(&"commutativity"));
        assert!(names.contains(&"identity"));
        assert!(names.contains(&"inverse"));
    }

    #[test]
    fn test_ring_axiom_set() {
        let ring = AxiomSet::ring();
        let names = ring.axiom_names();

        assert!(names.contains(&"associativity"));
        assert!(names.contains(&"commutativity"));
        assert!(names.contains(&"distributivity"));
    }

    #[test]
    fn test_integral_domain_axiom_set() {
        let domain = AxiomSet::integral_domain();
        let names = domain.axiom_names();

        assert!(names.contains(&"unity"));
        assert!(names.contains(&"no_zero_divisors"));
    }

    #[test]
    fn test_field_axiom_set() {
        let field = AxiomSet::field();
        let names = field.axiom_names();

        // Fields inherit all integral domain axioms
        assert!(names.contains(&"unity"));
        assert!(names.contains(&"no_zero_divisors"));
        assert!(names.contains(&"distributivity"));
    }

    #[test]
    fn test_axiom_set_display() {
        let set = AxiomSet::group();
        let display = format!("{}", set);

        assert!(display.contains("Axiom Set"));
        assert!(display.contains("closure"));
        assert!(display.contains("associativity"));
    }

    #[test]
    fn test_axiom_equality() {
        let a1 = Associativity;
        let a2 = Associativity;
        assert_eq!(a1, a2);

        let c1 = Commutativity;
        let c2 = Commutativity;
        assert_eq!(c1, c2);
    }
}
