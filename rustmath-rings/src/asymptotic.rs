//! # Asymptotic Rings
//!
//! This module implements asymptotic expansion rings for analyzing the asymptotic behavior
//! of sequences and functions. This is particularly useful in analytic combinatorics,
//! complexity analysis, and asymptotic analysis.
//!
//! ## Mathematical Background
//!
//! An asymptotic expansion represents a function's behavior as its argument approaches
//! infinity (or some other limit). For example:
//! - f(n) ~ n² as n → ∞
//! - f(x) ~ e^x / √(2πx) as x → ∞
//!
//! Asymptotic rings allow algebraic manipulation of such expressions while preserving
//! their asymptotic properties.
//!
//! ## Structure
//!
//! An asymptotic expansion consists of summands of the form:
//! - coefficient * growth_term
//!
//! where growth terms are products of powers, logarithms, and exponentials:
//! - n^α * log(n)^β * e^(γn)
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::asymptotic::{AsymptoticRing, AsymptoticExpansion};
//!
//! // Create ring for asymptotic analysis with respect to variable n
//! let ring = AsymptoticRing::new("n".to_string());
//!
//! // Create expression: n^2 + 2*n*log(n)
//! // (simplified example - actual API would be more complex)
//! ```

use rustmath_core::{Ring, CommutativeRing};
use rustmath_rationals::Rational;
use num_rational::BigRational;
use num_bigint::BigInt;
use num_traits::{Zero, One};
use std::fmt::{self, Debug, Display};
use std::cmp::Ordering;

/// A growth term in an asymptotic expansion.
///
/// Represents terms like n^α, log(n)^β, e^(γn), or products thereof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrowthTerm {
    /// The variable (e.g., "n", "x")
    variable: String,
    /// Power exponent (for n^α)
    power: BigRational,
    /// Logarithm exponent (for log(n)^β)
    log_power: i64,
    /// Exponential coefficient (for e^(γn))
    exp_coeff: BigRational,
}

impl GrowthTerm {
    /// Creates a new growth term.
    pub fn new(variable: String, power: BigRational, log_power: i64, exp_coeff: BigRational) -> Self {
        GrowthTerm {
            variable,
            power,
            log_power,
            exp_coeff,
        }
    }

    /// Creates the identity growth term (1).
    pub fn one(variable: String) -> Self {
        GrowthTerm {
            variable,
            power: BigRational::zero(),
            log_power: 0,
            exp_coeff: BigRational::zero(),
        }
    }

    /// Creates a power term n^α.
    pub fn power(variable: String, power: BigRational) -> Self {
        GrowthTerm {
            variable,
            power,
            log_power: 0,
            exp_coeff: BigRational::zero(),
        }
    }

    /// Creates a logarithm term log(n)^β.
    pub fn log(variable: String, log_power: i64) -> Self {
        GrowthTerm {
            variable,
            power: BigRational::zero(),
            log_power,
            exp_coeff: BigRational::zero(),
        }
    }

    /// Creates an exponential term e^(γn).
    pub fn exp(variable: String, exp_coeff: BigRational) -> Self {
        GrowthTerm {
            variable,
            power: BigRational::zero(),
            log_power: 0,
            exp_coeff,
        }
    }

    /// Multiplies two growth terms.
    pub fn multiply(&self, other: &GrowthTerm) -> GrowthTerm {
        assert_eq!(self.variable, other.variable, "Variables must match");

        GrowthTerm {
            variable: self.variable.clone(),
            power: &self.power + &other.power,
            log_power: self.log_power + other.log_power,
            exp_coeff: &self.exp_coeff + &other.exp_coeff,
        }
    }

    /// Compares growth rates of two terms.
    ///
    /// Returns:
    /// - `Ordering::Less` if self grows slower than other
    /// - `Ordering::Greater` if self grows faster than other
    /// - `Ordering::Equal` if they grow at the same rate
    pub fn compare_growth(&self, other: &GrowthTerm) -> Ordering {
        // First compare exponential growth
        match self.exp_coeff.cmp(&other.exp_coeff) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Then compare polynomial growth
        match self.power.cmp(&other.power) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Finally compare logarithmic growth
        self.log_power.cmp(&other.log_power)
    }
}

impl Display for GrowthTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();

        // Exponential part
        if !self.exp_coeff.is_zero() {
            parts.push(format!("exp({}*{})", self.exp_coeff, self.variable));
        }

        // Power part
        if !self.power.is_zero() {
            if self.power == BigRational::one() {
                parts.push(self.variable.clone());
            } else {
                parts.push(format!("{}^{}", self.variable, self.power));
            }
        }

        // Logarithm part
        if self.log_power != 0 {
            if self.log_power == 1 {
                parts.push(format!("log({})", self.variable));
            } else {
                parts.push(format!("log({})^{}", self.variable, self.log_power));
            }
        }

        if parts.is_empty() {
            write!(f, "1")
        } else {
            write!(f, "{}", parts.join(" * "))
        }
    }
}

/// A summand in an asymptotic expansion.
///
/// Represents coefficient * growth_term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsymptoticSummand {
    /// The coefficient (from the coefficient ring)
    coefficient: BigRational,
    /// The growth term
    growth: GrowthTerm,
}

impl AsymptoticSummand {
    /// Creates a new summand.
    pub fn new(coefficient: BigRational, growth: GrowthTerm) -> Self {
        AsymptoticSummand { coefficient, growth }
    }

    /// Creates a constant summand.
    pub fn constant(variable: String, value: BigRational) -> Self {
        AsymptoticSummand {
            coefficient: value,
            growth: GrowthTerm::one(variable),
        }
    }

    /// Returns true if the coefficient is zero.
    pub fn is_zero(&self) -> bool {
        self.coefficient.is_zero()
    }

    /// Multiplies this summand by a scalar.
    pub fn scale(&self, scalar: &BigRational) -> AsymptoticSummand {
        AsymptoticSummand {
            coefficient: &self.coefficient * scalar,
            growth: self.growth.clone(),
        }
    }

    /// Multiplies two summands.
    pub fn multiply(&self, other: &AsymptoticSummand) -> AsymptoticSummand {
        AsymptoticSummand {
            coefficient: &self.coefficient * &other.coefficient,
            growth: self.growth.multiply(&other.growth),
        }
    }
}

impl Display for AsymptoticSummand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coefficient.is_one() {
            write!(f, "{}", self.growth)
        } else {
            write!(f, "{} * {}", self.coefficient, self.growth)
        }
    }
}

/// An asymptotic expansion.
///
/// Represents a sum of summands: Σ coefficient_i * growth_i.
/// Summands are kept in order from fastest-growing to slowest-growing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsymptoticExpansion {
    /// The variable for the expansion
    variable: String,
    /// The summands, sorted by growth rate (descending)
    summands: Vec<AsymptoticSummand>,
    /// Default precision (number of summands to keep)
    precision: usize,
}

impl AsymptoticExpansion {
    /// Creates a new asymptotic expansion.
    pub fn new(variable: String, summands: Vec<AsymptoticSummand>, precision: usize) -> Self {
        let mut expansion = AsymptoticExpansion {
            variable,
            summands,
            precision,
        };
        expansion.normalize();
        expansion
    }

    /// Creates the zero expansion.
    pub fn zero(variable: String, precision: usize) -> Self {
        AsymptoticExpansion {
            variable,
            summands: vec![],
            precision,
        }
    }

    /// Creates the one expansion.
    pub fn one(variable: String, precision: usize) -> Self {
        AsymptoticExpansion {
            variable: variable.clone(),
            summands: vec![AsymptoticSummand::constant(variable, BigRational::one())],
            precision,
        }
    }

    /// Creates a variable expansion (n).
    pub fn variable(variable: String, precision: usize) -> Self {
        AsymptoticExpansion {
            variable: variable.clone(),
            summands: vec![AsymptoticSummand::new(
                BigRational::one(),
                GrowthTerm::power(variable, BigRational::one()),
            )],
            precision,
        }
    }

    /// Normalizes the expansion by sorting summands and combining like terms.
    fn normalize(&mut self) {
        // Sort by growth rate (descending)
        self.summands.sort_by(|a, b| {
            b.growth.compare_growth(&a.growth)
        });

        // Combine like terms
        let mut combined = Vec::new();
        for summand in &self.summands {
            if let Some(last) = combined.last_mut() {
                let last_summand: &mut AsymptoticSummand = last;
                if last_summand.growth == summand.growth {
                    last_summand.coefficient = &last_summand.coefficient + &summand.coefficient;
                    continue;
                }
            }
            combined.push(summand.clone());
        }

        // Remove zero summands
        combined.retain(|s| !s.is_zero());

        // Truncate to precision
        combined.truncate(self.precision);

        self.summands = combined;
    }

    /// Returns true if this is the zero expansion.
    pub fn is_zero(&self) -> bool {
        self.summands.is_empty()
    }

    /// Returns true if this is the one expansion.
    pub fn is_one(&self) -> bool {
        self.summands.len() == 1
            && self.summands[0].coefficient.is_one()
            && self.summands[0].growth.power.is_zero()
            && self.summands[0].growth.log_power == 0
            && self.summands[0].growth.exp_coeff.is_zero()
    }

    /// Returns the number of summands.
    pub fn len(&self) -> usize {
        self.summands.len()
    }

    /// Returns the summands.
    pub fn summands(&self) -> &[AsymptoticSummand] {
        &self.summands
    }

    /// Returns the leading term (fastest-growing term).
    pub fn leading_term(&self) -> Option<&AsymptoticSummand> {
        self.summands.first()
    }
}

impl Display for AsymptoticExpansion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else {
            for (i, summand) in self.summands.iter().enumerate() {
                if i > 0 {
                    write!(f, " + ")?;
                }
                write!(f, "{}", summand)?;
            }
            Ok(())
        }
    }
}

impl std::ops::Add for &AsymptoticExpansion {
    type Output = AsymptoticExpansion;

    fn add(self, other: &AsymptoticExpansion) -> AsymptoticExpansion {
        assert_eq!(self.variable, other.variable, "Variables must match");

        let mut summands = Vec::new();
        summands.extend(self.summands.clone());
        summands.extend(other.summands.clone());

        AsymptoticExpansion::new(
            self.variable.clone(),
            summands,
            self.precision.max(other.precision),
        )
    }
}

impl std::ops::Mul for &AsymptoticExpansion {
    type Output = AsymptoticExpansion;

    fn mul(self, other: &AsymptoticExpansion) -> AsymptoticExpansion {
        assert_eq!(self.variable, other.variable, "Variables must match");

        let mut summands = Vec::new();
        for s1 in &self.summands {
            for s2 in &other.summands {
                summands.push(s1.multiply(s2));
            }
        }

        AsymptoticExpansion::new(
            self.variable.clone(),
            summands,
            self.precision.max(other.precision),
        )
    }
}

impl std::ops::Neg for &AsymptoticExpansion {
    type Output = AsymptoticExpansion;

    fn neg(self) -> AsymptoticExpansion {
        let summands: Vec<AsymptoticSummand> = self
            .summands
            .iter()
            .map(|s| AsymptoticSummand {
                coefficient: -&s.coefficient,
                growth: s.growth.clone(),
            })
            .collect();

        AsymptoticExpansion {
            variable: self.variable.clone(),
            summands,
            precision: self.precision,
        }
    }
}

/// The asymptotic ring.
///
/// This is a ring of asymptotic expansions with respect to a given variable.
#[derive(Clone, Debug)]
pub struct AsymptoticRing {
    /// The variable (e.g., "n", "x")
    variable: String,
    /// Default precision for expansions
    default_precision: usize,
}

impl AsymptoticRing {
    /// Creates a new asymptotic ring.
    pub fn new(variable: String) -> Self {
        AsymptoticRing {
            variable,
            default_precision: 10,
        }
    }

    /// Creates a new asymptotic ring with specified precision.
    pub fn with_precision(variable: String, precision: usize) -> Self {
        AsymptoticRing {
            variable,
            default_precision: precision,
        }
    }

    /// Returns the variable.
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Returns the default precision.
    pub fn precision(&self) -> usize {
        self.default_precision
    }

    /// Creates the zero expansion.
    pub fn zero(&self) -> AsymptoticExpansion {
        AsymptoticExpansion::zero(self.variable.clone(), self.default_precision)
    }

    /// Creates the one expansion.
    pub fn one(&self) -> AsymptoticExpansion {
        AsymptoticExpansion::one(self.variable.clone(), self.default_precision)
    }

    /// Creates a variable expansion.
    pub fn var(&self) -> AsymptoticExpansion {
        AsymptoticExpansion::variable(self.variable.clone(), self.default_precision)
    }

    /// Creates a constant expansion.
    pub fn constant(&self, value: BigRational) -> AsymptoticExpansion {
        AsymptoticExpansion::new(
            self.variable.clone(),
            vec![AsymptoticSummand::constant(self.variable.clone(), value)],
            self.default_precision,
        )
    }

    /// Creates a power expansion n^α.
    pub fn power(&self, exponent: BigRational) -> AsymptoticExpansion {
        AsymptoticExpansion::new(
            self.variable.clone(),
            vec![AsymptoticSummand::new(
                BigRational::one(),
                GrowthTerm::power(self.variable.clone(), exponent),
            )],
            self.default_precision,
        )
    }

    /// Creates a logarithm expansion log(n)^β.
    pub fn log(&self, exponent: i64) -> AsymptoticExpansion {
        AsymptoticExpansion::new(
            self.variable.clone(),
            vec![AsymptoticSummand::new(
                BigRational::one(),
                GrowthTerm::log(self.variable.clone(), exponent),
            )],
            self.default_precision,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Ratio;

    #[test]
    fn test_growth_term_creation() {
        let term = GrowthTerm::power("n".to_string(), BigRational::from_integer(BigInt::from(2)));
        assert_eq!(term.power, BigRational::from_integer(BigInt::from(2)));
        assert_eq!(term.log_power, 0);
    }

    #[test]
    fn test_growth_term_multiplication() {
        let t1 = GrowthTerm::power("n".to_string(), BigRational::from_integer(BigInt::from(2)));
        let t2 = GrowthTerm::power("n".to_string(), BigRational::from_integer(BigInt::from(3)));
        let product = t1.multiply(&t2);

        assert_eq!(product.power, BigRational::from_integer(BigInt::from(5)));
    }

    #[test]
    fn test_growth_term_comparison() {
        let t1 = GrowthTerm::power("n".to_string(), BigRational::from_integer(BigInt::from(2)));
        let t2 = GrowthTerm::power("n".to_string(), BigRational::from_integer(BigInt::from(3)));

        assert_eq!(t1.compare_growth(&t2), Ordering::Less);
        assert_eq!(t2.compare_growth(&t1), Ordering::Greater);
        assert_eq!(t1.compare_growth(&t1), Ordering::Equal);
    }

    #[test]
    fn test_asymptotic_ring_creation() {
        let ring = AsymptoticRing::new("n".to_string());
        assert_eq!(ring.variable(), "n");
        assert_eq!(ring.precision(), 10);
    }

    #[test]
    fn test_asymptotic_expansion_zero_one() {
        let ring = AsymptoticRing::new("n".to_string());

        let zero = ring.zero();
        assert!(zero.is_zero());

        let one = ring.one();
        assert!(one.is_one());
    }

    #[test]
    fn test_asymptotic_expansion_addition() {
        let ring = AsymptoticRing::new("n".to_string());

        let zero = ring.zero();
        let one = ring.one();

        let sum = &zero + &one;
        assert!(sum.is_one());

        let sum = &one + &one;
        assert_eq!(sum.summands().len(), 1);
        assert_eq!(sum.summands()[0].coefficient, BigRational::from_integer(BigInt::from(2)));
    }

    #[test]
    fn test_asymptotic_expansion_multiplication() {
        let ring = AsymptoticRing::new("n".to_string());

        let one = ring.one();
        let var = ring.var();

        // 1 * n = n
        let prod = &one * &var;
        assert_eq!(prod.summands().len(), 1);
        assert_eq!(prod.summands()[0].growth.power, BigRational::from_integer(BigInt::from(1)));

        // n * n = n^2
        let prod = &var * &var;
        assert_eq!(prod.summands().len(), 1);
        assert_eq!(prod.summands()[0].growth.power, BigRational::from_integer(BigInt::from(2)));
    }

    #[test]
    fn test_asymptotic_expansion_negation() {
        let ring = AsymptoticRing::new("n".to_string());

        let one = ring.one();
        let neg_one = -&one;

        assert_eq!(neg_one.summands().len(), 1);
        assert_eq!(neg_one.summands()[0].coefficient, BigRational::from_integer(BigInt::from(-1)));
    }

    #[test]
    fn test_asymptotic_expansion_normalization() {
        let ring = AsymptoticRing::new("n".to_string());

        // Create n + n (should normalize to 2n)
        let var = ring.var();
        let sum = &var + &var;

        assert_eq!(sum.summands().len(), 1);
        assert_eq!(sum.summands()[0].coefficient, BigRational::from_integer(BigInt::from(2)));
    }

    #[test]
    fn test_asymptotic_power_log() {
        let ring = AsymptoticRing::new("n".to_string());

        let n_squared = ring.power(BigRational::from_integer(BigInt::from(2)));
        assert_eq!(n_squared.summands()[0].growth.power, BigRational::from_integer(BigInt::from(2)));

        let log_n = ring.log(1);
        assert_eq!(log_n.summands()[0].growth.log_power, 1);
    }

    #[test]
    fn test_display() {
        let ring = AsymptoticRing::new("n".to_string());

        let zero = ring.zero();
        assert_eq!(format!("{}", zero), "0");

        let one = ring.one();
        assert_eq!(format!("{}", one), "1");

        let var = ring.var();
        let display_str = format!("{}", var);
        assert!(display_str.contains("n"));
    }
}
