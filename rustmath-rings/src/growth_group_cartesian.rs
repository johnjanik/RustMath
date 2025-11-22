//! # Cartesian Products of Growth Groups
//!
//! This module implements Cartesian products of growth groups, which are used to represent
//! multivariate asymptotic behavior. A Cartesian product allows combining different types of
//! growth (e.g., monomial and exponential) or the same type of growth in multiple variables.
//!
//! ## Mathematical Background
//!
//! Given growth groups G₁, G₂, ..., Gₙ, their Cartesian product G₁ × G₂ × ... × Gₙ consists
//! of tuples (g₁, g₂, ..., gₙ) where each gᵢ ∈ Gᵢ. The group operation is component-wise:
//! (g₁, ..., gₙ) · (h₁, ..., hₙ) = (g₁·h₁, ..., gₙ·hₙ)
//!
//! The ordering is lexicographic: compare elements left-to-right by growth rate.
//!
//! ## Use Cases
//!
//! - Multivariate asymptotic analysis: analyze f(m,n) as both m,n → ∞
//! - Mixed growth types: combine polynomial and exponential growth
//! - Hierarchical growth comparisons: prioritize certain variables
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::growth_group_cartesian::{GenericProduct, CartesianProductFactory};
//! use rustmath_rings::growth_group::{MonomialGrowthGroup, Variable};
//!
//! // Create a product of two monomial growth groups: (m^α, n^β)
//! let m = Variable::new("m");
//! let n = Variable::new("n");
//!
//! // Elements represent products like m^2 * n^3
//! ```

use crate::growth_group::{
    GrowthElement, GrowthGroup, Variable, GrowthGroupFactor,
};
use std::cmp::Ordering;
use std::fmt::{self, Debug, Display};

// ======================================================================================
// CARTESIAN PRODUCT ELEMENT
// ======================================================================================

/// An element in a Cartesian product of growth groups.
///
/// Represents a tuple (g₁, g₂, ..., gₙ) where each gᵢ is an element of a growth group.
#[derive(Clone, Debug)]
pub struct Element<T: GrowthElement> {
    /// The components of the product
    components: Vec<T>,
}

impl<T: GrowthElement> Element<T> {
    /// Creates a new Cartesian product element.
    pub fn new(components: Vec<T>) -> Self {
        Element { components }
    }

    /// Returns the components.
    pub fn components(&self) -> &[T] {
        &self.components
    }

    /// Returns the number of components.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Returns true if this has no components.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Returns a specific component.
    pub fn component(&self, index: usize) -> Option<&T> {
        self.components.get(index)
    }

    /// Multiplies two elements component-wise.
    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(
            self.len(),
            other.len(),
            "Product elements must have same number of components"
        );

        let components = self
            .components
            .iter()
            .zip(&other.components)
            .map(|(a, b)| a.multiply(b))
            .collect();

        Element { components }
    }

    /// Computes the inverse element (component-wise).
    pub fn inverse(&self) -> Self {
        let components = self.components.iter().map(|c| c.inverse()).collect();
        Element { components }
    }

    /// Checks if this is the identity element.
    pub fn is_identity(&self) -> bool {
        self.components.iter().all(|c| c.is_identity())
    }
}

impl<T: GrowthElement> PartialEq for Element<T> {
    fn eq(&self, other: &Self) -> bool {
        self.components == other.components
    }
}

impl<T: GrowthElement> Eq for Element<T> {}

impl<T: GrowthElement> PartialOrd for Element<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: GrowthElement> Ord for Element<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lexicographic ordering: compare components left to right
        for (a, b) in self.components.iter().zip(&other.components) {
            match a.compare_growth(b) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

impl<T: GrowthElement> Display for Element<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "1");
        }

        let non_identity: Vec<String> = self
            .components
            .iter()
            .filter(|c| !c.is_identity())
            .map(|c| format!("{}", c))
            .collect();

        if non_identity.is_empty() {
            write!(f, "1")
        } else {
            write!(f, "{}", non_identity.join(" * "))
        }
    }
}

// ======================================================================================
// GENERIC PRODUCT
// ======================================================================================

/// A Cartesian product of growth groups.
///
/// This represents the product G₁ × G₂ × ... × Gₙ of multiple growth groups.
/// Elements are ordered lexicographically.
#[derive(Clone, Debug)]
pub struct GenericProduct {
    /// The growth group factors
    factors: Vec<GrowthGroupFactor>,
    /// Description of the product
    description: String,
}

impl GenericProduct {
    /// Creates a new Cartesian product of growth groups.
    pub fn new(factors: Vec<GrowthGroupFactor>) -> Self {
        let desc_parts: Vec<String> = factors
            .iter()
            .map(|f| format!("{}({})", f.group_type(), f.variable()))
            .collect();

        GenericProduct {
            factors,
            description: desc_parts.join(" × "),
        }
    }

    /// Returns the factors.
    pub fn factors(&self) -> &[GrowthGroupFactor] {
        &self.factors
    }

    /// Returns the number of factors.
    pub fn num_factors(&self) -> usize {
        self.factors.len()
    }

    /// Returns a description of this product.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Checks if a variable appears in any factor.
    pub fn contains_variable(&self, var: &Variable) -> bool {
        self.factors.iter().any(|f| f.variable() == var)
    }
}

impl Display for GenericProduct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

// ======================================================================================
// UNIVARIATE PRODUCT
// ======================================================================================

/// A Cartesian product where all factors use the same variable.
///
/// This is a special case that arises when combining different growth types
/// (e.g., monomial and exponential) for a single variable.
#[derive(Clone, Debug)]
pub struct UnivariateProduct {
    /// The common variable
    variable: Variable,
    /// The underlying generic product
    base: GenericProduct,
}

impl UnivariateProduct {
    /// Creates a new univariate product.
    pub fn new(variable: Variable, factors: Vec<GrowthGroupFactor>) -> Self {
        // Verify all factors use the same variable
        for factor in &factors {
            assert_eq!(
                factor.variable(),
                &variable,
                "All factors must use the same variable"
            );
        }

        UnivariateProduct {
            variable: variable.clone(),
            base: GenericProduct::new(factors),
        }
    }

    /// Returns the variable.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }

    /// Returns the underlying generic product.
    pub fn base(&self) -> &GenericProduct {
        &self.base
    }

    /// Returns the number of factors.
    pub fn num_factors(&self) -> usize {
        self.base.num_factors()
    }
}

impl Display for UnivariateProduct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Univariate({}, {})", self.variable, self.base)
    }
}

// ======================================================================================
// MULTIVARIATE PRODUCT
// ======================================================================================

/// A Cartesian product where factors use different variables.
///
/// This is the general case for multivariate asymptotic analysis.
#[derive(Clone, Debug)]
pub struct MultivariateProduct {
    /// The variables (in order)
    variables: Vec<Variable>,
    /// The underlying generic product
    base: GenericProduct,
}

impl MultivariateProduct {
    /// Creates a new multivariate product.
    pub fn new(factors: Vec<GrowthGroupFactor>) -> Self {
        // Extract unique variables in order
        let mut variables = Vec::new();
        for factor in &factors {
            let var = factor.variable().clone();
            if !variables.contains(&var) {
                variables.push(var);
            }
        }

        MultivariateProduct {
            variables,
            base: GenericProduct::new(factors),
        }
    }

    /// Returns the variables.
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// Returns the number of variables.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Returns the underlying generic product.
    pub fn base(&self) -> &GenericProduct {
        &self.base
    }

    /// Checks if a variable appears in this product.
    pub fn contains_variable(&self, var: &Variable) -> bool {
        self.variables.contains(var)
    }
}

impl Display for MultivariateProduct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Multivariate([{}], {})",
            self.variables
                .iter()
                .map(|v| format!("{}", v))
                .collect::<Vec<_>>()
                .join(", "),
            self.base
        )
    }
}

// ======================================================================================
// CARTESIAN PRODUCT FACTORY
// ======================================================================================

/// Factory for constructing Cartesian products of growth groups.
///
/// This provides a uniform interface for creating different types of products
/// (generic, univariate, multivariate) with automatic type detection.
#[derive(Clone, Debug)]
pub struct CartesianProductFactory;

impl CartesianProductFactory {
    /// Creates a new factory.
    pub fn new() -> Self {
        CartesianProductFactory
    }

    /// Creates a generic product from factors.
    pub fn generic(&self, factors: Vec<GrowthGroupFactor>) -> GenericProduct {
        GenericProduct::new(factors)
    }

    /// Creates a univariate product.
    ///
    /// All factors must use the same variable.
    pub fn univariate(&self, variable: Variable, factors: Vec<GrowthGroupFactor>) -> UnivariateProduct {
        UnivariateProduct::new(variable, factors)
    }

    /// Creates a multivariate product.
    pub fn multivariate(&self, factors: Vec<GrowthGroupFactor>) -> MultivariateProduct {
        MultivariateProduct::new(factors)
    }

    /// Automatically determines the appropriate product type and creates it.
    pub fn auto(&self, factors: Vec<GrowthGroupFactor>) -> GenericProduct {
        // Could enhance this to return an enum of different product types
        GenericProduct::new(factors)
    }
}

impl Default for CartesianProductFactory {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================================
// HELPER FUNCTIONS
// ======================================================================================

/// Creates a monomial-exponential product for a single variable.
///
/// This is a common case combining polynomial and exponential growth.
///
/// # Arguments
/// * `variable` - The variable
///
/// # Returns
/// A univariate product with monomial and exponential factors
pub fn monomial_exponential_product(variable: Variable) -> UnivariateProduct {
    let factors = vec![
        GrowthGroupFactor::new("monomial".to_string(), variable.clone()),
        GrowthGroupFactor::new("exponential".to_string(), variable.clone()),
    ];
    UnivariateProduct::new(variable, factors)
}

/// Creates a product of monomial groups for multiple variables.
///
/// # Arguments
/// * `variables` - The variables
///
/// # Returns
/// A multivariate product with monomial factors for each variable
pub fn multivariate_monomial_product(variables: Vec<Variable>) -> MultivariateProduct {
    let factors = variables
        .into_iter()
        .map(|var| GrowthGroupFactor::new("monomial".to_string(), var))
        .collect();
    MultivariateProduct::new(factors)
}

// ======================================================================================
// TESTS
// ======================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::growth_group::{MonomialGrowthElement, MonomialGrowthGroup};

    #[test]
    fn test_element_creation() {
        let var = Variable::new("n");
        let e1 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let e2 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(3)));

        let elem = Element::new(vec![e1, e2]);
        assert_eq!(elem.len(), 2);
        assert!(!elem.is_empty());
    }

    #[test]
    fn test_element_multiplication() {
        let var = Variable::new("n");
        let e1 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let e2 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(3)));
        let elem1 = Element::new(vec![e1.clone(), e2.clone()]);

        let e3 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(1)));
        let e4 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(1)));
        let elem2 = Element::new(vec![e3, e4]);

        let product = elem1.multiply(&elem2);
        assert_eq!(product.len(), 2);
    }

    #[test]
    fn test_element_inverse() {
        let var = Variable::new("n");
        let e1 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let e2 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(3)));
        let elem = Element::new(vec![e1, e2]);

        let inv = elem.inverse();
        let product = elem.multiply(&inv);
        assert!(product.is_identity());
    }

    #[test]
    fn test_element_comparison() {
        let var = Variable::new("n");
        let e1 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let e2 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(3)));
        let elem1 = Element::new(vec![e1.clone()]);
        let elem2 = Element::new(vec![e2.clone()]);

        assert!(elem1 < elem2);
        assert!(elem2 > elem1);
        assert_eq!(elem1, elem1);
    }

    #[test]
    fn test_generic_product() {
        let var = Variable::new("n");
        let factors = vec![
            GrowthGroupFactor::new("monomial".to_string(), var.clone()),
            GrowthGroupFactor::new("exponential".to_string(), var.clone()),
        ];

        let product = GenericProduct::new(factors);
        assert_eq!(product.num_factors(), 2);
        assert!(product.contains_variable(&var));
    }

    #[test]
    fn test_univariate_product() {
        let var = Variable::new("n");
        let factors = vec![
            GrowthGroupFactor::new("monomial".to_string(), var.clone()),
            GrowthGroupFactor::new("exponential".to_string(), var.clone()),
        ];

        let product = UnivariateProduct::new(var.clone(), factors);
        assert_eq!(product.variable(), &var);
        assert_eq!(product.num_factors(), 2);
    }

    #[test]
    fn test_multivariate_product() {
        let m = Variable::new("m");
        let n = Variable::new("n");
        let factors = vec![
            GrowthGroupFactor::new("monomial".to_string(), m.clone()),
            GrowthGroupFactor::new("monomial".to_string(), n.clone()),
        ];

        let product = MultivariateProduct::new(factors);
        assert_eq!(product.num_variables(), 2);
        assert!(product.contains_variable(&m));
        assert!(product.contains_variable(&n));
    }

    #[test]
    fn test_cartesian_product_factory() {
        let factory = CartesianProductFactory::new();
        let var = Variable::new("n");

        let factors = vec![
            GrowthGroupFactor::new("monomial".to_string(), var.clone()),
        ];

        let product = factory.generic(factors);
        assert_eq!(product.num_factors(), 1);
    }

    #[test]
    fn test_monomial_exponential_product() {
        let var = Variable::new("n");
        let product = monomial_exponential_product(var.clone());

        assert_eq!(product.variable(), &var);
        assert_eq!(product.num_factors(), 2);
    }

    #[test]
    fn test_multivariate_monomial_product() {
        let m = Variable::new("m");
        let n = Variable::new("n");
        let vars = vec![m.clone(), n.clone()];

        let product = multivariate_monomial_product(vars);
        assert_eq!(product.num_variables(), 2);
    }

    #[test]
    fn test_element_display() {
        let var = Variable::new("n");
        let e1 = MonomialGrowthElement::new(var.clone(), BigRational::from_integer(BigInt::from(2)));
        let elem = Element::new(vec![e1]);

        let display = format!("{}", elem);
        assert!(display.contains("n"));
    }

    #[test]
    fn test_product_display() {
        let var = Variable::new("n");
        let factors = vec![
            GrowthGroupFactor::new("monomial".to_string(), var.clone()),
        ];

        let product = GenericProduct::new(factors);
        let display = format!("{}", product);
        assert!(display.contains("monomial"));
    }
}
