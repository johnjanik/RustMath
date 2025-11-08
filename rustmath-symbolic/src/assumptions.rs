//! Symbolic assumptions system
//!
//! This module provides a system for declaring and querying assumptions about
//! symbolic variables (e.g., "x is positive", "y is real", etc.).

use crate::symbol::Symbol;
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

/// Properties that can be assumed about a symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Property {
    /// The symbol is positive (> 0)
    Positive,
    /// The symbol is negative (< 0)
    Negative,
    /// The symbol is non-negative (>= 0)
    NonNegative,
    /// The symbol is non-positive (<= 0)
    NonPositive,
    /// The symbol is zero
    Zero,
    /// The symbol is real-valued
    Real,
    /// The symbol is complex-valued
    Complex,
    /// The symbol is an integer
    Integer,
    /// The symbol is rational
    Rational,
    /// The symbol is even (implies integer)
    Even,
    /// The symbol is odd (implies integer)
    Odd,
    /// The symbol is a prime number (implies positive integer)
    Prime,
}

/// Global assumptions database
///
/// Maps symbol IDs to sets of properties
static ASSUMPTIONS: RwLock<Option<HashMap<usize, HashSet<Property>>>> = RwLock::new(None);

/// Initialize the assumptions database if not already initialized
fn ensure_initialized() {
    let mut db = ASSUMPTIONS.write().unwrap();
    if db.is_none() {
        *db = Some(HashMap::new());
    }
}

/// Add an assumption about a symbol
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::{Expr, assume, Property};
///
/// let x = Expr::symbol("x");
/// assume(&x.as_symbol().unwrap(), Property::Positive);
/// assume(&x.as_symbol().unwrap(), Property::Real);
/// ```
pub fn assume(symbol: &Symbol, property: Property) {
    ensure_initialized();
    let mut db = ASSUMPTIONS.write().unwrap();

    db.as_mut()
        .unwrap()
        .entry(symbol.id())
        .or_insert_with(HashSet::new)
        .insert(property);

    // Add implied properties
    let symbol_id = symbol.id();
    let implied = get_implied_properties(property);

    for implied_prop in implied {
        db.as_mut()
            .unwrap()
            .get_mut(&symbol_id)
            .unwrap()
            .insert(implied_prop);
    }
}

/// Get properties that are implied by a given property
fn get_implied_properties(property: Property) -> Vec<Property> {
    match property {
        Property::Positive => vec![Property::Real, Property::NonNegative],
        Property::Negative => vec![Property::Real, Property::NonPositive],
        Property::Zero => vec![Property::Real, Property::NonNegative, Property::NonPositive],
        Property::Integer => vec![Property::Real, Property::Rational],
        Property::Rational => vec![Property::Real],
        Property::Even => vec![Property::Integer, Property::Real, Property::Rational],
        Property::Odd => vec![Property::Integer, Property::Real, Property::Rational],
        Property::Prime => vec![Property::Positive, Property::Integer, Property::Real, Property::Rational, Property::NonNegative],
        Property::Real => vec![Property::Complex],
        _ => vec![],
    }
}

/// Check if a symbol has a specific property
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::{Expr, assume, has_property, Property};
///
/// let x = Expr::symbol("x");
/// let x_sym = x.as_symbol().unwrap();
///
/// assume(&x_sym, Property::Positive);
/// assert!(has_property(&x_sym, Property::Positive));
/// assert!(has_property(&x_sym, Property::Real)); // Implied by Positive
/// ```
pub fn has_property(symbol: &Symbol, property: Property) -> bool {
    ensure_initialized();
    let db = ASSUMPTIONS.read().unwrap();

    db.as_ref()
        .and_then(|map| map.get(&symbol.id()))
        .map(|props| props.contains(&property))
        .unwrap_or(false)
}

/// Remove all assumptions about a symbol
pub fn forget(symbol: &Symbol) {
    ensure_initialized();
    let mut db = ASSUMPTIONS.write().unwrap();

    if let Some(map) = db.as_mut() {
        map.remove(&symbol.id());
    }
}

/// Remove all assumptions
pub fn forget_all() {
    ensure_initialized();
    let mut db = ASSUMPTIONS.write().unwrap();

    if let Some(map) = db.as_mut() {
        map.clear();
    }
}

/// Get all assumptions about a symbol
pub fn get_assumptions(symbol: &Symbol) -> Vec<Property> {
    ensure_initialized();
    let db = ASSUMPTIONS.read().unwrap();

    db.as_ref()
        .and_then(|map| map.get(&symbol.id()))
        .map(|props| props.iter().copied().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Symbol;

    #[test]
    fn test_assume_positive() {
        let x = Symbol::new("x");

        assume(&x, Property::Positive);

        assert!(has_property(&x, Property::Positive));
        assert!(has_property(&x, Property::Real)); // Implied
        assert!(has_property(&x, Property::NonNegative)); // Implied
        assert!(!has_property(&x, Property::Negative));

        forget(&x);
    }

    #[test]
    fn test_assume_integer() {
        let n = Symbol::new("n");

        assume(&n, Property::Integer);

        assert!(has_property(&n, Property::Integer));
        assert!(has_property(&n, Property::Real)); // Implied
        assert!(has_property(&n, Property::Rational)); // Implied
        assert!(!has_property(&n, Property::Positive));

        forget(&n);
    }

    #[test]
    fn test_assume_prime() {
        let p = Symbol::new("p");

        assume(&p, Property::Prime);

        assert!(has_property(&p, Property::Prime));
        assert!(has_property(&p, Property::Positive)); // Implied
        assert!(has_property(&p, Property::Integer)); // Implied
        assert!(has_property(&p, Property::Real)); // Implied

        forget(&p);
    }

    #[test]
    fn test_multiple_assumptions() {
        let x = Symbol::new("x");

        assume(&x, Property::Positive);
        assume(&x, Property::Integer);

        assert!(has_property(&x, Property::Positive));
        assert!(has_property(&x, Property::Integer));
        assert!(has_property(&x, Property::Real)); // Implied by both

        forget(&x);
    }

    #[test]
    fn test_forget() {
        let x = Symbol::new("x");

        assume(&x, Property::Positive);
        assert!(has_property(&x, Property::Positive));

        forget(&x);
        assert!(!has_property(&x, Property::Positive));
    }

    #[test]
    fn test_get_assumptions() {
        let x = Symbol::new("x");

        assume(&x, Property::Positive);

        let assumptions = get_assumptions(&x);
        assert!(assumptions.contains(&Property::Positive));
        assert!(assumptions.contains(&Property::Real));
        assert!(assumptions.contains(&Property::NonNegative));

        forget(&x);
    }
}
