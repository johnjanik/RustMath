//! Multivariate polynomials (polynomials in multiple variables)

use rustmath_core::Ring;
use std::collections::BTreeMap;
use std::fmt;

/// A monomial in multiple variables
///
/// Represents x₀^e₀ × x₁^e₁ × ... × xₙ^eₙ
/// Stored as a map from variable index to exponent
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Monomial {
    /// Map from variable index to exponent
    /// Only non-zero exponents are stored
    exponents: BTreeMap<usize, u32>,
}

impl Monomial {
    /// Create a new monomial
    pub fn new() -> Self {
        Monomial {
            exponents: BTreeMap::new(),
        }
    }

    /// Create a monomial from exponents
    pub fn from_exponents(exponents: BTreeMap<usize, u32>) -> Self {
        let mut filtered = BTreeMap::new();
        for (var, exp) in exponents {
            if exp > 0 {
                filtered.insert(var, exp);
            }
        }
        Monomial {
            exponents: filtered,
        }
    }

    /// Create a monomial representing a single variable to a power
    pub fn variable(var: usize, power: u32) -> Self {
        let mut exponents = BTreeMap::new();
        if power > 0 {
            exponents.insert(var, power);
        }
        Monomial { exponents }
    }

    /// Get the exponent for a variable
    pub fn exponent(&self, var: usize) -> u32 {
        *self.exponents.get(&var).unwrap_or(&0)
    }

    /// Get the total degree (sum of all exponents)
    pub fn degree(&self) -> u32 {
        self.exponents.values().sum()
    }

    /// Multiply two monomials
    pub fn mul(&self, other: &Monomial) -> Monomial {
        let mut exponents = self.exponents.clone();
        for (var, exp) in &other.exponents {
            *exponents.entry(*var).or_insert(0) += exp;
        }
        Monomial { exponents }
    }

    /// Check if this is the constant monomial (1)
    pub fn is_one(&self) -> bool {
        self.exponents.is_empty()
    }

    /// Get all variables that appear in this monomial
    pub fn variables(&self) -> Vec<usize> {
        self.exponents.keys().copied().collect()
    }
}

impl Default for Monomial {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_one() {
            write!(f, "1")?;
            return Ok(());
        }

        let mut first = true;
        for (var, exp) in &self.exponents {
            if !first {
                write!(f, "*")?;
            }
            first = false;

            write!(f, "x{}", var)?;
            if *exp > 1 {
                write!(f, "^{}", exp)?;
            }
        }
        Ok(())
    }
}

/// A multivariate polynomial over a ring R
///
/// Represented as a map from monomials to coefficients
#[derive(Clone, Debug, PartialEq)]
pub struct MultivariatePolynomial<R: Ring> {
    /// Map from monomial to coefficient
    /// Only non-zero coefficients are stored
    terms: BTreeMap<Monomial, R>,
}

impl<R: Ring> MultivariatePolynomial<R> {
    /// Create a new zero polynomial
    pub fn zero() -> Self {
        MultivariatePolynomial {
            terms: BTreeMap::new(),
        }
    }

    /// Create a constant polynomial
    pub fn constant(c: R) -> Self {
        if c.is_zero() {
            return Self::zero();
        }

        let mut terms = BTreeMap::new();
        terms.insert(Monomial::new(), c);
        MultivariatePolynomial { terms }
    }

    /// Create a polynomial representing a single variable
    pub fn variable(var: usize) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(Monomial::variable(var, 1), R::one());
        MultivariatePolynomial { terms }
    }

    /// Add a term to the polynomial
    pub fn add_term(&mut self, monomial: Monomial, coeff: R) {
        if coeff.is_zero() {
            return;
        }

        // Clone monomial for potential removal later
        let monomial_key = monomial.clone();
        let entry = self.terms.entry(monomial).or_insert_with(|| R::zero());
        *entry = entry.clone() + coeff;

        // Remove if it became zero
        if entry.is_zero() {
            self.terms.remove(&monomial_key);
        }
    }

    /// Get the coefficient of a monomial
    pub fn coefficient(&self, monomial: &Monomial) -> R {
        self.terms.get(monomial).cloned().unwrap_or_else(|| R::zero())
    }

    /// Get the total degree of the polynomial
    pub fn degree(&self) -> Option<u32> {
        self.terms.keys().map(|m| m.degree()).max()
    }

    /// Get the degree in a specific variable
    pub fn degree_in(&self, var: usize) -> Option<u32> {
        self.terms.keys().map(|m| m.exponent(var)).max()
    }

    /// Check if the polynomial is zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Check if the polynomial is a constant
    pub fn is_constant(&self) -> bool {
        self.terms.len() <= 1 && self.terms.keys().all(|m| m.is_one())
    }

    /// Get all variables that appear in the polynomial
    pub fn variables(&self) -> Vec<usize> {
        let mut vars = std::collections::HashSet::new();
        for monomial in self.terms.keys() {
            for var in monomial.variables() {
                vars.insert(var);
            }
        }
        let mut result: Vec<_> = vars.into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Number of terms in the polynomial
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }
}

impl<R: Ring> std::ops::Add for MultivariatePolynomial<R> {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        for (monomial, coeff) in other.terms {
            self.add_term(monomial, coeff);
        }
        self
    }
}

impl<R: Ring> std::ops::Sub for MultivariatePolynomial<R> {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        for (monomial, coeff) in other.terms {
            self.add_term(monomial, -coeff);
        }
        self
    }
}

impl<R: Ring> std::ops::Mul for MultivariatePolynomial<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        let mut result = Self::zero();

        for (m1, c1) in &self.terms {
            for (m2, c2) in &other.terms {
                let new_monomial = m1.mul(m2);
                let new_coeff = c1.clone() * c2.clone();
                result.add_term(new_monomial, new_coeff);
            }
        }

        result
    }
}

impl<R: Ring> std::ops::Neg for MultivariatePolynomial<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut result = Self::zero();
        for (monomial, coeff) in self.terms {
            result.add_term(monomial, -coeff);
        }
        result
    }
}

impl<R: Ring> fmt::Display for MultivariatePolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (monomial, coeff) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if monomial.is_one() {
                write!(f, "{}", coeff)?;
            } else if coeff.is_one() {
                write!(f, "{}", monomial)?;
            } else {
                write!(f, "{}*{}", coeff, monomial)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monomial_creation() {
        let m = Monomial::variable(0, 2);
        assert_eq!(m.exponent(0), 2);
        assert_eq!(m.degree(), 2);
    }

    #[test]
    fn test_monomial_multiplication() {
        let m1 = Monomial::variable(0, 2); // x₀²
        let m2 = Monomial::variable(1, 3); // x₁³

        let m3 = m1.mul(&m2); // x₀²x₁³
        assert_eq!(m3.exponent(0), 2);
        assert_eq!(m3.exponent(1), 3);
        assert_eq!(m3.degree(), 5);
    }

    #[test]
    fn test_polynomial_creation() {
        let p: MultivariatePolynomial<i32> = MultivariatePolynomial::constant(5);
        assert!(p.is_constant());
        assert_eq!(p.coefficient(&Monomial::new()), 5);
    }

    #[test]
    fn test_polynomial_variable() {
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        assert!(!x.is_constant());
        assert!(!y.is_constant());
        assert_eq!(x.degree(), Some(1));
        assert_eq!(y.degree(), Some(1));
    }

    #[test]
    fn test_polynomial_addition() {
        // x + y
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let sum = x + y;
        assert_eq!(sum.num_terms(), 2);
        assert_eq!(sum.degree(), Some(1));
    }

    #[test]
    fn test_polynomial_multiplication() {
        // (x + 1) * (y + 2) = xy + 2x + y + 2
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);
        let one = MultivariatePolynomial::constant(1);
        let two = MultivariatePolynomial::constant(2);

        let p1 = x.clone() + one;
        let p2 = y + two;
        let product = p1 * p2;

        assert_eq!(product.num_terms(), 4);
        assert_eq!(product.degree(), Some(2)); // xy has degree 2
    }

    #[test]
    fn test_polynomial_display() {
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let p = x + y;
        let display = format!("{}", p);
        assert!(display.contains("x0") || display.contains("x1"));
    }
}
