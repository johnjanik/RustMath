//! # Growth Groups for Asymptotic Analysis
//!
//! This module implements growth groups, which are fundamental structures for representing
//! the growth behavior of functions in asymptotic analysis. Growth groups are partially
//! ordered abelian groups that capture the relative growth rates of functions.
//!
//! ## Mathematical Background
//!
//! A growth group is a multiplicative group whose elements represent growth behaviors such as:
//! - Monomial growth: n^α for rational α
//! - Exponential growth: e^(βn) for rational β
//! - Logarithmic growth: log(n)^γ for integer γ
//! - Products and compositions thereof
//!
//! Elements can be compared based on their asymptotic growth rate, making growth groups
//! partially ordered structures.
//!
//! ## Module Structure
//!
//! - `Variable`: Represents a symbolic variable (e.g., "n", "x")
//! - `GenericGrowthElement`: Base element type for growth groups
//! - `GenericGrowthGroup`: Base group structure
//! - `MonomialGrowthElement`/`MonomialGrowthGroup`: For n^α terms
//! - `ExponentialGrowthElement`/`ExponentialGrowthGroup`: For e^(βn) terms
//! - `GenericNonGrowthElement`/`GenericNonGrowthGroup`: For constant terms
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::growth_group::{MonomialGrowthGroup, Variable};
//! use num_rational::BigRational;
//! use num_bigint::BigInt;
//!
//! // Create a monomial growth group for variable n
//! let var = Variable::new("n");
//! let group = MonomialGrowthGroup::new(var);
//!
//! // Create elements representing n^2 and n^3
//! let n_squared = group.element(BigRational::from_integer(BigInt::from(2)));
//! let n_cubed = group.element(BigRational::from_integer(BigInt::from(3)));
//!
//! // Compare growth rates: n^2 < n^3
//! assert!(n_squared < n_cubed);
//! ```

use num_rational::BigRational;
use num_traits::{Zero, One};
use std::cmp::Ordering;
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

// ======================================================================================
// VARIABLE
// ======================================================================================

/// Represents a symbolic variable in asymptotic expressions.
///
/// Variables are named symbols like "n", "x", "t" that represent the independent
/// variable as it approaches infinity in asymptotic analysis.
///
/// Variables are cached to ensure that equal variable names produce identical objects,
/// which is important for comparison and hashing.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Variable {
    /// The variable name
    name: Arc<String>,
}

impl Variable {
    /// Creates a new variable with the given name.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::growth_group::Variable;
    ///
    /// let n = Variable::new("n");
    /// let x = Variable::new("x");
    /// ```
    pub fn new(name: &str) -> Self {
        Variable {
            name: Arc::new(name.to_string()),
        }
    }

    /// Returns the variable name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ======================================================================================
// PARTIAL CONVERSION (for handling conversion errors)
// ======================================================================================

/// Error raised when a partial conversion fails.
///
/// This is used when attempting to convert between different growth group elements
/// where the conversion may not be well-defined.
#[derive(Debug, Clone)]
pub struct PartialConversionValueError {
    message: String,
}

impl PartialConversionValueError {
    pub fn new(message: String) -> Self {
        PartialConversionValueError { message }
    }
}

impl Display for PartialConversionValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Partial conversion error: {}", self.message)
    }
}

impl std::error::Error for PartialConversionValueError {}

/// Represents an element that may be partially converted.
///
/// This is used as a wrapper when conversions between growth elements might fail.
#[derive(Clone, Debug)]
pub struct PartialConversionElement<T> {
    value: Option<T>,
    error: Option<String>,
}

impl<T> PartialConversionElement<T> {
    /// Creates a successful partial conversion.
    pub fn success(value: T) -> Self {
        PartialConversionElement {
            value: Some(value),
            error: None,
        }
    }

    /// Creates a failed partial conversion.
    pub fn failure(error: String) -> Self {
        PartialConversionElement {
            value: None,
            error: Some(error),
        }
    }

    /// Checks if the conversion was successful.
    pub fn is_success(&self) -> bool {
        self.value.is_some()
    }

    /// Unwraps the value, panicking if the conversion failed.
    pub fn unwrap(self) -> T {
        self.value.expect("Unwrapping failed partial conversion")
    }

    /// Returns the value if successful, None otherwise.
    pub fn get(self) -> Option<T> {
        self.value
    }
}

// ======================================================================================
// ERROR FOR DECREASING GROWTH
// ======================================================================================

/// Error raised when a growth element would represent decreasing behavior.
///
/// Growth groups are designed for non-decreasing functions. This error is raised
/// when attempting to create an element that would violate this assumption.
#[derive(Debug, Clone)]
pub struct DecreasingGrowthElementError {
    message: String,
}

impl DecreasingGrowthElementError {
    pub fn new(message: String) -> Self {
        DecreasingGrowthElementError { message }
    }
}

impl Display for DecreasingGrowthElementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Decreasing growth error: {}", self.message)
    }
}

impl std::error::Error for DecreasingGrowthElementError {}

// ======================================================================================
// GENERIC GROWTH ELEMENT
// ======================================================================================

/// Base trait for growth group elements.
///
/// All growth elements must implement this trait, which provides methods for
/// comparing growth rates and performing group operations.
pub trait GrowthElement: Clone + Debug + PartialEq + Display {
    /// Returns the variable for this growth element.
    fn variable(&self) -> &Variable;

    /// Compares the growth rate of this element with another.
    ///
    /// Returns:
    /// - `Ordering::Less` if self grows slower than other
    /// - `Ordering::Greater` if self grows faster than other
    /// - `Ordering::Equal` if they grow at the same rate
    fn compare_growth(&self, other: &Self) -> Ordering;

    /// Multiplies this element with another (group operation).
    fn multiply(&self, other: &Self) -> Self;

    /// Computes the inverse of this element.
    fn inverse(&self) -> Self;

    /// Returns the identity element.
    fn identity(&self) -> Self;

    /// Checks if this is the identity element.
    fn is_identity(&self) -> bool;

    /// Raises this element to a rational power.
    fn pow(&self, exponent: &BigRational) -> Self;
}

/// Generic implementation of a monomial growth element: var^exponent.
///
/// This represents growth terms like n^2, n^(1/2), n^(-1), etc.
#[derive(Clone, Debug)]
pub struct GenericGrowthElement {
    /// The variable (e.g., "n")
    variable: Variable,
    /// The exponent (can be any rational number)
    exponent: BigRational,
}

impl GenericGrowthElement {
    /// Creates a new generic growth element.
    pub fn new(variable: Variable, exponent: BigRational) -> Self {
        GenericGrowthElement { variable, exponent }
    }

    /// Creates the identity element (exponent = 0).
    pub fn identity(variable: Variable) -> Self {
        GenericGrowthElement {
            variable,
            exponent: BigRational::zero(),
        }
    }

    /// Returns the exponent.
    pub fn exponent(&self) -> &BigRational {
        &self.exponent
    }
}

impl PartialEq for GenericGrowthElement {
    fn eq(&self, other: &Self) -> bool {
        self.variable == other.variable && self.exponent == other.exponent
    }
}

impl Eq for GenericGrowthElement {}

impl PartialOrd for GenericGrowthElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GenericGrowthElement {
    fn cmp(&self, other: &Self) -> Ordering {
        assert_eq!(self.variable, other.variable, "Variables must match");
        self.exponent.cmp(&other.exponent)
    }
}

impl Display for GenericGrowthElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exponent.is_zero() {
            write!(f, "1")
        } else if self.exponent.is_one() {
            write!(f, "{}", self.variable)
        } else {
            write!(f, "{}^{}", self.variable, self.exponent)
        }
    }
}

impl GrowthElement for GenericGrowthElement {
    fn variable(&self) -> &Variable {
        &self.variable
    }

    fn compare_growth(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.variable, other.variable, "Variables must match");
        GenericGrowthElement {
            variable: self.variable.clone(),
            exponent: &self.exponent + &other.exponent,
        }
    }

    fn inverse(&self) -> Self {
        GenericGrowthElement {
            variable: self.variable.clone(),
            exponent: -&self.exponent,
        }
    }

    fn identity(&self) -> Self {
        GenericGrowthElement::identity(self.variable.clone())
    }

    fn is_identity(&self) -> bool {
        self.exponent.is_zero()
    }

    fn pow(&self, exponent: &BigRational) -> Self {
        GenericGrowthElement {
            variable: self.variable.clone(),
            exponent: &self.exponent * exponent,
        }
    }
}

// ======================================================================================
// MONOMIAL GROWTH ELEMENT
// ======================================================================================

/// A monomial growth element representing var^exponent.
///
/// This is the primary type for representing polynomial growth in asymptotic analysis.
/// Examples: n, n^2, n^(1/2), n^(-1)
#[derive(Clone, Debug)]
pub struct MonomialGrowthElement {
    base: GenericGrowthElement,
}

impl MonomialGrowthElement {
    /// Creates a new monomial growth element.
    pub fn new(variable: Variable, exponent: BigRational) -> Self {
        MonomialGrowthElement {
            base: GenericGrowthElement::new(variable, exponent),
        }
    }

    /// Creates the identity element (var^0 = 1).
    pub fn identity(variable: Variable) -> Self {
        MonomialGrowthElement {
            base: GenericGrowthElement::identity(variable),
        }
    }

    /// Returns the exponent.
    pub fn exponent(&self) -> &BigRational {
        self.base.exponent()
    }

    /// Returns the variable.
    pub fn variable(&self) -> &Variable {
        self.base.variable()
    }
}

impl PartialEq for MonomialGrowthElement {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
    }
}

impl Eq for MonomialGrowthElement {}

impl PartialOrd for MonomialGrowthElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MonomialGrowthElement {
    fn cmp(&self, other: &Self) -> Ordering {
        self.base.cmp(&other.base)
    }
}

impl Display for MonomialGrowthElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base)
    }
}

impl GrowthElement for MonomialGrowthElement {
    fn variable(&self) -> &Variable {
        self.base.variable()
    }

    fn compare_growth(&self, other: &Self) -> Ordering {
        self.base.compare_growth(&other.base)
    }

    fn multiply(&self, other: &Self) -> Self {
        MonomialGrowthElement {
            base: self.base.multiply(&other.base),
        }
    }

    fn inverse(&self) -> Self {
        MonomialGrowthElement {
            base: self.base.inverse(),
        }
    }

    fn identity(&self) -> Self {
        MonomialGrowthElement::identity(self.variable().clone())
    }

    fn is_identity(&self) -> bool {
        self.base.is_identity()
    }

    fn pow(&self, exponent: &BigRational) -> Self {
        MonomialGrowthElement {
            base: self.base.pow(exponent),
        }
    }
}

// ======================================================================================
// EXPONENTIAL GROWTH ELEMENT
// ======================================================================================

/// An exponential growth element representing exp(base^exponent).
///
/// This represents exponential growth behaviors like e^n, e^(2n), e^(n/2), etc.
#[derive(Clone, Debug)]
pub struct ExponentialGrowthElement {
    /// The variable
    variable: Variable,
    /// The base of the exponential (stored as the coefficient)
    base: BigRational,
}

impl ExponentialGrowthElement {
    /// Creates a new exponential growth element.
    ///
    /// Represents exp(base * variable).
    pub fn new(variable: Variable, base: BigRational) -> Self {
        ExponentialGrowthElement { variable, base }
    }

    /// Creates the identity element (exp(0) = 1).
    pub fn identity(variable: Variable) -> Self {
        ExponentialGrowthElement {
            variable,
            base: BigRational::zero(),
        }
    }

    /// Returns the base coefficient.
    pub fn base(&self) -> &BigRational {
        &self.base
    }

    /// Returns the variable.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }
}

impl PartialEq for ExponentialGrowthElement {
    fn eq(&self, other: &Self) -> bool {
        self.variable == other.variable && self.base == other.base
    }
}

impl Eq for ExponentialGrowthElement {}

impl PartialOrd for ExponentialGrowthElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExponentialGrowthElement {
    fn cmp(&self, other: &Self) -> Ordering {
        assert_eq!(self.variable, other.variable, "Variables must match");
        self.base.cmp(&other.base)
    }
}

impl Display for ExponentialGrowthElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.base.is_zero() {
            write!(f, "1")
        } else if self.base.is_one() {
            write!(f, "exp({})", self.variable)
        } else {
            write!(f, "exp({}*{})", self.base, self.variable)
        }
    }
}

impl GrowthElement for ExponentialGrowthElement {
    fn variable(&self) -> &Variable {
        &self.variable
    }

    fn compare_growth(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.variable, other.variable, "Variables must match");
        ExponentialGrowthElement {
            variable: self.variable.clone(),
            base: &self.base + &other.base,
        }
    }

    fn inverse(&self) -> Self {
        ExponentialGrowthElement {
            variable: self.variable.clone(),
            base: -&self.base,
        }
    }

    fn identity(&self) -> Self {
        ExponentialGrowthElement::identity(self.variable.clone())
    }

    fn is_identity(&self) -> bool {
        self.base.is_zero()
    }

    fn pow(&self, exponent: &BigRational) -> Self {
        ExponentialGrowthElement {
            variable: self.variable.clone(),
            base: &self.base * exponent,
        }
    }
}

// ======================================================================================
// NON-GROWTH ELEMENTS (Constants)
// ======================================================================================

/// A generic non-growth element representing a constant (no growth).
///
/// This is the identity element in growth comparisons, representing O(1) behavior.
#[derive(Clone, Debug)]
pub struct GenericNonGrowthElement {
    variable: Variable,
}

impl GenericNonGrowthElement {
    /// Creates a new non-growth element.
    pub fn new(variable: Variable) -> Self {
        GenericNonGrowthElement { variable }
    }

    /// Returns the variable.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }
}

impl PartialEq for GenericNonGrowthElement {
    fn eq(&self, other: &Self) -> bool {
        self.variable == other.variable
    }
}

impl Eq for GenericNonGrowthElement {}

impl Display for GenericNonGrowthElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "1")
    }
}

/// A monomial non-growth element (always 1).
pub type MonomialNonGrowthElement = GenericNonGrowthElement;

/// An exponential non-growth element (always 1).
pub type ExponentialNonGrowthElement = GenericNonGrowthElement;

// ======================================================================================
// GROWTH GROUPS
// ======================================================================================

/// Base trait for growth groups.
pub trait GrowthGroup<E: GrowthElement> {
    /// Returns the variable for this growth group.
    fn variable(&self) -> &Variable;

    /// Creates an element from the given exponent/coefficient.
    fn element(&self, param: BigRational) -> E;

    /// Returns the identity element of the group.
    fn identity(&self) -> E;

    /// Checks if an element belongs to this group.
    fn contains(&self, element: &E) -> bool {
        element.variable() == self.variable()
    }
}

/// Generic growth group implementation.
#[derive(Clone, Debug)]
pub struct GenericGrowthGroup {
    variable: Variable,
}

impl GenericGrowthGroup {
    /// Creates a new generic growth group.
    pub fn new(variable: Variable) -> Self {
        GenericGrowthGroup { variable }
    }

    /// Returns the variable.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }
}

impl GrowthGroup<GenericGrowthElement> for GenericGrowthGroup {
    fn variable(&self) -> &Variable {
        &self.variable
    }

    fn element(&self, exponent: BigRational) -> GenericGrowthElement {
        GenericGrowthElement::new(self.variable.clone(), exponent)
    }

    fn identity(&self) -> GenericGrowthElement {
        GenericGrowthElement::identity(self.variable.clone())
    }
}

/// Monomial growth group for var^exponent terms.
///
/// This group contains all monomial growth elements with a given variable.
#[derive(Clone, Debug)]
pub struct MonomialGrowthGroup {
    variable: Variable,
}

impl MonomialGrowthGroup {
    /// Creates a new monomial growth group.
    pub fn new(variable: Variable) -> Self {
        MonomialGrowthGroup { variable }
    }

    /// Returns the variable.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }
}

impl GrowthGroup<MonomialGrowthElement> for MonomialGrowthGroup {
    fn variable(&self) -> &Variable {
        &self.variable
    }

    fn element(&self, exponent: BigRational) -> MonomialGrowthElement {
        MonomialGrowthElement::new(self.variable.clone(), exponent)
    }

    fn identity(&self) -> MonomialGrowthElement {
        MonomialGrowthElement::identity(self.variable.clone())
    }
}

/// Exponential growth group for exp(base*var) terms.
///
/// This group contains all exponential growth elements with a given variable.
#[derive(Clone, Debug)]
pub struct ExponentialGrowthGroup {
    variable: Variable,
}

impl ExponentialGrowthGroup {
    /// Creates a new exponential growth group.
    pub fn new(variable: Variable) -> Self {
        ExponentialGrowthGroup { variable }
    }

    /// Returns the variable.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }
}

impl GrowthGroup<ExponentialGrowthElement> for ExponentialGrowthGroup {
    fn variable(&self) -> &Variable {
        &self.variable
    }

    fn element(&self, base: BigRational) -> ExponentialGrowthElement {
        ExponentialGrowthElement::new(self.variable.clone(), base)
    }

    fn identity(&self) -> ExponentialGrowthElement {
        ExponentialGrowthElement::identity(self.variable.clone())
    }
}

/// Generic non-growth group (trivial group containing only the identity).
#[derive(Clone, Debug)]
pub struct GenericNonGrowthGroup {
    variable: Variable,
}

impl GenericNonGrowthGroup {
    /// Creates a new non-growth group.
    pub fn new(variable: Variable) -> Self {
        GenericNonGrowthGroup { variable }
    }

    /// Returns the variable.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }

    /// Returns the single element of this group.
    pub fn element(&self) -> GenericNonGrowthElement {
        GenericNonGrowthElement::new(self.variable.clone())
    }
}

/// Monomial non-growth group.
pub type MonomialNonGrowthGroup = GenericNonGrowthGroup;

/// Exponential non-growth group.
pub type ExponentialNonGrowthGroup = GenericNonGrowthGroup;

// ======================================================================================
// GROWTH GROUP FUNCTORS (for construction)
// ======================================================================================

/// Abstract base for growth group construction functors.
///
/// Functors provide a uniform interface for constructing growth groups
/// and converting between different growth group types.
pub trait AbstractGrowthGroupFunctor {
    /// Returns a description of this functor.
    fn description(&self) -> String;
}

/// Functor for constructing monomial growth groups.
#[derive(Clone, Debug)]
pub struct MonomialGrowthGroupFunctor {
    variable: Variable,
}

impl MonomialGrowthGroupFunctor {
    /// Creates a new functor for the given variable.
    pub fn new(variable: Variable) -> Self {
        MonomialGrowthGroupFunctor { variable }
    }

    /// Constructs a monomial growth group.
    pub fn construct(&self) -> MonomialGrowthGroup {
        MonomialGrowthGroup::new(self.variable.clone())
    }
}

impl AbstractGrowthGroupFunctor for MonomialGrowthGroupFunctor {
    fn description(&self) -> String {
        format!("MonomialGrowthGroup({})", self.variable)
    }
}

/// Functor for constructing exponential growth groups.
#[derive(Clone, Debug)]
pub struct ExponentialGrowthGroupFunctor {
    variable: Variable,
}

impl ExponentialGrowthGroupFunctor {
    /// Creates a new functor for the given variable.
    pub fn new(variable: Variable) -> Self {
        ExponentialGrowthGroupFunctor { variable }
    }

    /// Constructs an exponential growth group.
    pub fn construct(&self) -> ExponentialGrowthGroup {
        ExponentialGrowthGroup::new(self.variable.clone())
    }
}

impl AbstractGrowthGroupFunctor for ExponentialGrowthGroupFunctor {
    fn description(&self) -> String {
        format!("ExponentialGrowthGroup({})", self.variable)
    }
}

/// Functor for monomial non-growth groups.
pub type MonomialNonGrowthGroupFunctor = MonomialGrowthGroupFunctor;

/// Functor for exponential non-growth groups.
pub type ExponentialNonGrowthGroupFunctor = ExponentialGrowthGroupFunctor;

// ======================================================================================
// GROWTH GROUP FACTORY
// ======================================================================================

/// Factory for constructing various types of growth groups.
///
/// This provides a centralized interface for creating growth groups with
/// different characteristics (monomial, exponential, etc.).
#[derive(Clone, Debug)]
pub struct GrowthGroupFactory;

impl GrowthGroupFactory {
    /// Creates a new growth group factory.
    pub fn new() -> Self {
        GrowthGroupFactory
    }

    /// Creates a monomial growth group.
    pub fn monomial(&self, variable: Variable) -> MonomialGrowthGroup {
        MonomialGrowthGroup::new(variable)
    }

    /// Creates an exponential growth group.
    pub fn exponential(&self, variable: Variable) -> ExponentialGrowthGroup {
        ExponentialGrowthGroup::new(variable)
    }

    /// Creates a generic growth group.
    pub fn generic(&self, variable: Variable) -> GenericGrowthGroup {
        GenericGrowthGroup::new(variable)
    }
}

impl Default for GrowthGroupFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a factor in a growth group product.
///
/// This is used when constructing Cartesian products of growth groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrowthGroupFactor {
    /// The type of growth group (e.g., "monomial", "exponential")
    group_type: String,
    /// The variable
    variable: Variable,
}

impl GrowthGroupFactor {
    /// Creates a new growth group factor.
    pub fn new(group_type: String, variable: Variable) -> Self {
        GrowthGroupFactor {
            group_type,
            variable,
        }
    }

    /// Returns the group type.
    pub fn group_type(&self) -> &str {
        &self.group_type
    }

    /// Returns the variable.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }
}

// ======================================================================================
// TESTS
// ======================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;

    #[test]
    fn test_variable_creation() {
        let n = Variable::new("n");
        let x = Variable::new("x");

        assert_eq!(n.name(), "n");
        assert_eq!(x.name(), "x");
        assert_ne!(n, x);
    }

    #[test]
    fn test_variable_display() {
        let n = Variable::new("n");
        assert_eq!(format!("{}", n), "n");
    }

    #[test]
    fn test_generic_growth_element_creation() {
        let var = Variable::new("n");
        let elem = GenericGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));

        assert_eq!(elem.variable(), &var);
        assert_eq!(elem.exponent(), &BigRational::from_integer(BigInt::from(2)));
    }

    #[test]
    fn test_generic_growth_element_identity() {
        let var = Variable::new("n");
        let identity = GenericGrowthElement::identity(var.clone());

        assert!(identity.is_identity());
        assert_eq!(identity.exponent(), &BigRational::zero());
    }

    #[test]
    fn test_generic_growth_element_multiplication() {
        let var = Variable::new("n");
        let e1 = GenericGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let e2 = GenericGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(3)));

        let product = e1.multiply(&e2);
        assert_eq!(product.exponent(), &BigRational::from_integer(BigInt::from(5)));
    }

    #[test]
    fn test_generic_growth_element_inverse() {
        let var = Variable::new("n");
        let elem = GenericGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let inv = elem.inverse();

        assert_eq!(inv.exponent(), &BigRational::from_integer(BigInt::from(-2)));

        let product = elem.multiply(&inv);
        assert!(product.is_identity());
    }

    #[test]
    fn test_generic_growth_element_power() {
        let var = Variable::new("n");
        let elem = GenericGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let squared = elem.pow(&BigRational::from_integer(BigInt::from(2)));

        assert_eq!(squared.exponent(), &BigRational::from_integer(BigInt::from(4)));
    }

    #[test]
    fn test_generic_growth_element_comparison() {
        let var = Variable::new("n");
        let e1 = GenericGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let e2 = GenericGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(3)));

        assert_eq!(e1.compare_growth(&e2), Ordering::Less);
        assert_eq!(e2.compare_growth(&e1), Ordering::Greater);
        assert_eq!(e1.compare_growth(&e1), Ordering::Equal);
    }

    #[test]
    fn test_monomial_growth_element() {
        let var = Variable::new("n");
        let elem = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));

        assert_eq!(elem.exponent(), &BigRational::from_integer(BigInt::from(2)));
        assert_eq!(format!("{}", elem), "n^2");
    }

    #[test]
    fn test_monomial_growth_element_operations() {
        let var = Variable::new("n");
        let e1 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let e2 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(3)));

        let product = e1.multiply(&e2);
        assert_eq!(product.exponent(), &BigRational::from_integer(BigInt::from(5)));

        assert!(e1 < e2);
    }

    #[test]
    fn test_exponential_growth_element() {
        let var = Variable::new("n");
        let elem = ExponentialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));

        assert_eq!(elem.base(), &BigRational::from_integer(BigInt::from(2)));
        assert_eq!(format!("{}", elem), "exp(2*n)");
    }

    #[test]
    fn test_exponential_growth_element_operations() {
        let var = Variable::new("n");
        let e1 = ExponentialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let e2 = ExponentialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(3)));

        let product = e1.multiply(&e2);
        assert_eq!(product.base(), &BigRational::from_integer(BigInt::from(5)));

        assert!(e1 < e2);
    }

    #[test]
    fn test_generic_non_growth_element() {
        let var = Variable::new("n");
        let elem = GenericNonGrowthElement::new(var.clone());

        assert_eq!(elem.variable(), &var);
        assert_eq!(format!("{}", elem), "1");
    }

    #[test]
    fn test_monomial_growth_group() {
        let var = Variable::new("n");
        let group = MonomialGrowthGroup::new(var.clone());

        assert_eq!(group.variable(), &var);

        let elem = group.element(BigRational::from_integer(BigInt::from(2)));
        assert_eq!(elem.exponent(), &BigRational::from_integer(BigInt::from(2)));

        let identity = group.identity();
        assert!(identity.is_identity());

        assert!(group.contains(&elem));
    }

    #[test]
    fn test_exponential_growth_group() {
        let var = Variable::new("n");
        let group = ExponentialGrowthGroup::new(var.clone());

        assert_eq!(group.variable(), &var);

        let elem = group.element(BigRational::from_integer(BigInt::from(2)));
        assert_eq!(elem.base(), &BigRational::from_integer(BigInt::from(2)));

        let identity = group.identity();
        assert!(identity.is_identity());
    }

    #[test]
    fn test_growth_group_factory() {
        let factory = GrowthGroupFactory::new();
        let var = Variable::new("n");

        let monomial = factory.monomial(var.clone());
        assert_eq!(monomial.variable(), &var);

        let exponential = factory.exponential(var.clone());
        assert_eq!(exponential.variable(), &var);

        let generic = factory.generic(var.clone());
        assert_eq!(generic.variable(), &var);
    }

    #[test]
    fn test_monomial_functor() {
        let var = Variable::new("n");
        let functor = MonomialGrowthGroupFunctor::new(var.clone());

        let desc = functor.description();
        assert!(desc.contains("MonomialGrowthGroup"));
        assert!(desc.contains("n"));

        let group = functor.construct();
        assert_eq!(group.variable(), &var);
    }

    #[test]
    fn test_exponential_functor() {
        let var = Variable::new("n");
        let functor = ExponentialGrowthGroupFunctor::new(var.clone());

        let desc = functor.description();
        assert!(desc.contains("ExponentialGrowthGroup"));
        assert!(desc.contains("n"));

        let group = functor.construct();
        assert_eq!(group.variable(), &var);
    }

    #[test]
    fn test_growth_group_factor() {
        let var = Variable::new("n");
        let factor = GrowthGroupFactor::new("monomial".to_string(), var.clone());

        assert_eq!(factor.group_type(), "monomial");
        assert_eq!(factor.variable(), &var);
    }

    #[test]
    fn test_display_formats() {
        let var = Variable::new("n");

        // Identity
        let id = GenericGrowthElement::identity(var.clone());
        assert_eq!(format!("{}", id), "1");

        // Variable
        let n = GenericGrowthElement::new(var.clone(), BigRational::one());
        assert_eq!(format!("{}", n), "n");

        // Power
        let n_sq = GenericGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        assert_eq!(format!("{}", n_sq), "n^2");
    }

    #[test]
    fn test_partial_conversion() {
        let success: PartialConversionElement<i32> = PartialConversionElement::success(42);
        assert!(success.is_success());
        assert_eq!(success.unwrap(), 42);

        let failure: PartialConversionElement<i32> =
            PartialConversionElement::failure("conversion failed".to_string());
        assert!(!failure.is_success());
        assert_eq!(failure.get(), None);
    }

    #[test]
    fn test_errors() {
        let err = DecreasingGrowthElementError::new("test error".to_string());
        assert!(format!("{}", err).contains("test error"));

        let conv_err = PartialConversionValueError::new("conversion error".to_string());
        assert!(format!("{}", conv_err).contains("conversion error"));
    }
}
