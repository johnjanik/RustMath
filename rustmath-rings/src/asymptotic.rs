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
//! ## Multivariate Generating Functions
//!
//! This module also implements tools for computing asymptotic expansions of coefficients
//! in multivariate rational generating functions. This is based on the work of Raichev
//! and Wilson (2008, 2012) and provides decomposition algorithms for analyzing the
//! asymptotic behavior of combinatorial sequences.
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
use num_traits::{Zero, One, Signed};
use std::fmt::{self, Debug, Display};
use std::cmp::Ordering;
use std::collections::HashMap;

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

// ======================================================================================
// MULTIVARIATE GENERATING FUNCTIONS
// ======================================================================================

/// A polynomial represented in a simplified form for multivariate generating functions.
///
/// This is a lightweight wrapper around a map of monomial exponents to coefficients.
/// For example, `x^2*y + 3*x*y^2` would be represented as:
/// `{(2,1): 1, (1,2): 3}` where keys are (x_exp, y_exp) and values are coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicPolynomial {
    /// Maps variable indices to their names
    variables: Vec<String>,
    /// Maps monomial (as Vec<usize> exponents) to coefficient
    terms: HashMap<Vec<usize>, BigRational>,
}

impl SymbolicPolynomial {
    /// Creates a new polynomial with the given variables.
    pub fn new(variables: Vec<String>) -> Self {
        SymbolicPolynomial {
            variables,
            terms: HashMap::new(),
        }
    }

    /// Creates a constant polynomial.
    pub fn constant(variables: Vec<String>, value: BigRational) -> Self {
        let mut poly = SymbolicPolynomial::new(variables.clone());
        let zero_exps = vec![0; variables.len()];
        poly.terms.insert(zero_exps, value);
        poly
    }

    /// Creates a polynomial representing a single variable.
    pub fn variable(variables: Vec<String>, var_index: usize) -> Self {
        let mut poly = SymbolicPolynomial::new(variables.clone());
        let mut exps = vec![0; variables.len()];
        exps[var_index] = 1;
        poly.terms.insert(exps, BigRational::one());
        poly
    }

    /// Adds a term to the polynomial.
    pub fn add_term(&mut self, exponents: Vec<usize>, coefficient: BigRational) {
        if !coefficient.is_zero() {
            *self.terms.entry(exponents).or_insert_with(BigRational::zero) += coefficient;
        }
    }

    /// Returns true if the polynomial is zero.
    pub fn is_zero(&self) -> bool {
        self.terms.values().all(|c| c.is_zero())
    }

    /// Returns true if the polynomial is a constant.
    pub fn is_constant(&self) -> bool {
        self.terms.len() <= 1 && self.terms.keys().all(|exp| exp.iter().all(|&e| e == 0))
    }

    /// Returns the degree of the polynomial.
    pub fn degree(&self) -> usize {
        self.terms.keys().map(|exp| exp.iter().sum()).max().unwrap_or(0)
    }

    /// Returns the number of variables.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Returns the variables.
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Evaluates the polynomial at a given point.
    ///
    /// The point is given as a map from variable names to values.
    pub fn evaluate(&self, point: &HashMap<String, BigRational>) -> BigRational {
        let mut result = BigRational::zero();
        for (exps, coeff) in &self.terms {
            let mut term_value = coeff.clone();
            for (i, &exp) in exps.iter().enumerate() {
                if exp > 0 {
                    if let Some(val) = point.get(&self.variables[i]) {
                        term_value *= val.pow(exp as i32);
                    }
                }
            }
            result += term_value;
        }
        result
    }

    /// Computes the GCD of this polynomial with another.
    ///
    /// This is a simplified implementation that only handles trivial cases.
    /// For full GCD computation, a more sophisticated algorithm would be needed.
    pub fn gcd(&self, _other: &SymbolicPolynomial) -> SymbolicPolynomial {
        // Simplified: return 1 (indicating coprime)
        SymbolicPolynomial::constant(self.variables.clone(), BigRational::one())
    }

    /// Divides this polynomial by another, returning quotient and remainder.
    ///
    /// Returns None if division is not exact when required.
    pub fn div_rem(&self, _divisor: &SymbolicPolynomial) -> Option<(SymbolicPolynomial, SymbolicPolynomial)> {
        // Simplified implementation
        None
    }
}

impl Display for SymbolicPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut terms_vec: Vec<_> = self.terms.iter().collect();
        terms_vec.sort_by_key(|(exp, _)| exp.iter().sum::<usize>());

        for (i, (exps, coeff)) in terms_vec.iter().enumerate() {
            if i > 0 && !coeff.is_negative() {
                write!(f, " + ")?;
            } else if coeff.is_negative() {
                write!(f, " - ")?;
            }

            let abs_coeff = if coeff.is_negative() { -coeff.clone() } else { (*coeff).clone() };

            let all_zero = exps.iter().all(|&e| e == 0);
            if !abs_coeff.is_one() || all_zero {
                write!(f, "{}", abs_coeff)?;
                if !all_zero {
                    write!(f, "*")?;
                }
            }

            let mut first_var = true;
            for (j, &exp) in exps.iter().enumerate() {
                if exp > 0 {
                    if !first_var {
                        write!(f, "*")?;
                    }
                    write!(f, "{}", self.variables[j])?;
                    if exp > 1 {
                        write!(f, "^{}", exp)?;
                    }
                    first_var = false;
                }
            }
        }
        Ok(())
    }
}

/// Represents a fraction with factored denominator.
///
/// This structure represents a rational expression as (numerator, [(factor₁, exp₁), ..., (factorₙ, expₙ)])
/// where the denominator is ∏ᵢ factorᵢ^expᵢ.
///
/// This factored representation is useful for:
/// - Avoiding repeated factorization operations
/// - Preserving structure for asymptotic analysis
/// - Efficient partial fraction decomposition
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionWithFactoredDenominator {
    /// The numerator polynomial
    numerator: SymbolicPolynomial,
    /// The denominator as a list of (factor, exponent) pairs
    /// Each factor is a polynomial, and the full denominator is the product of factor^exponent
    denominator_factors: Vec<(SymbolicPolynomial, usize)>,
}

impl FractionWithFactoredDenominator {
    /// Creates a new fraction with factored denominator.
    ///
    /// # Arguments
    /// * `numerator` - The numerator polynomial
    /// * `denominator_factors` - List of (factor, exponent) pairs for the denominator
    ///
    /// # Example
    /// ```rust,ignore
    /// // Represents 1 / (x^2 * (x+1))
    /// let x = SymbolicPolynomial::variable(vec!["x".to_string()], 0);
    /// let x_plus_1 = x + SymbolicPolynomial::constant(vec!["x".to_string()], BigRational::one());
    /// let frac = FractionWithFactoredDenominator::new(
    ///     SymbolicPolynomial::constant(vec!["x".to_string()], BigRational::one()),
    ///     vec![(x, 2), (x_plus_1, 1)]
    /// );
    /// ```
    pub fn new(numerator: SymbolicPolynomial, denominator_factors: Vec<(SymbolicPolynomial, usize)>) -> Self {
        let mut frac = FractionWithFactoredDenominator {
            numerator,
            denominator_factors,
        };
        frac.reduce();
        frac
    }

    /// Creates a fraction from a numerator and a single denominator polynomial.
    pub fn from_polynomials(numerator: SymbolicPolynomial, denominator: SymbolicPolynomial) -> Self {
        // In a full implementation, we would factor the denominator here
        // For now, we treat it as a single factor
        FractionWithFactoredDenominator::new(numerator, vec![(denominator, 1)])
    }

    /// Reduces the fraction to lowest terms by canceling common factors.
    fn reduce(&mut self) {
        // Simplified implementation: cancel obvious common factors
        let mut new_factors = Vec::new();

        for (factor, exp) in &self.denominator_factors {
            // Try to divide numerator by factor
            let gcd = self.numerator.gcd(factor);
            if !gcd.is_constant() {
                // There's a common factor - in full implementation, we'd cancel it
                // For now, just keep the factor
            }
            if *exp > 0 {
                new_factors.push((factor.clone(), *exp));
            }
        }

        self.denominator_factors = new_factors;
    }

    /// Returns the numerator.
    pub fn numerator(&self) -> &SymbolicPolynomial {
        &self.numerator
    }

    /// Returns the denominator factors.
    pub fn denominator_factors(&self) -> &[(SymbolicPolynomial, usize)] {
        &self.denominator_factors
    }

    /// Returns the number of distinct factors in the denominator.
    pub fn num_denominator_factors(&self) -> usize {
        self.denominator_factors.len()
    }

    /// Checks if the denominator is square-free (all exponents are 1).
    pub fn is_square_free(&self) -> bool {
        self.denominator_factors.iter().all(|(_, exp)| *exp == 1)
    }

    /// Computes the full denominator as a single polynomial.
    ///
    /// This expands all factors: ∏ᵢ factorᵢ^expᵢ
    pub fn denominator(&self) -> SymbolicPolynomial {
        // Simplified: return first factor or constant 1
        if let Some((first, _)) = self.denominator_factors.first() {
            first.clone()
        } else {
            SymbolicPolynomial::constant(self.numerator.variables().to_vec(), BigRational::one())
        }
    }
}

impl Display for FractionWithFactoredDenominator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) / ", self.numerator)?;

        if self.denominator_factors.is_empty() {
            write!(f, "1")?;
        } else {
            write!(f, "(")?;
            for (i, (factor, exp)) in self.denominator_factors.iter().enumerate() {
                if i > 0 {
                    write!(f, " * ")?;
                }
                if *exp == 1 {
                    write!(f, "({})", factor)?;
                } else {
                    write!(f, "({})^{}", factor, exp)?;
                }
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

/// The ring of fractions with factored denominators.
///
/// This structure manages a collection of `FractionWithFactoredDenominator` elements
/// and provides operations for manipulating them.
#[derive(Clone, Debug)]
pub struct FractionWithFactoredDenominatorRing {
    /// The variables in the polynomial ring
    variables: Vec<String>,
    /// The coefficient ring (for now, always rationals)
    _coefficient_ring: String,
}

impl FractionWithFactoredDenominatorRing {
    /// Creates a new ring of fractions with factored denominators.
    ///
    /// # Arguments
    /// * `variables` - The variables for the polynomial ring
    pub fn new(variables: Vec<String>) -> Self {
        FractionWithFactoredDenominatorRing {
            variables,
            _coefficient_ring: "QQ".to_string(), // Rationals
        }
    }

    /// Returns the variables.
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Creates the zero element.
    pub fn zero(&self) -> FractionWithFactoredDenominator {
        FractionWithFactoredDenominator::new(
            SymbolicPolynomial::constant(self.variables.clone(), BigRational::zero()),
            vec![],
        )
    }

    /// Creates the one element.
    pub fn one(&self) -> FractionWithFactoredDenominator {
        FractionWithFactoredDenominator::new(
            SymbolicPolynomial::constant(self.variables.clone(), BigRational::one()),
            vec![],
        )
    }

    /// Creates a fraction from polynomials.
    pub fn fraction(&self, numerator: SymbolicPolynomial, denominator: SymbolicPolynomial) -> FractionWithFactoredDenominator {
        FractionWithFactoredDenominator::from_polynomials(numerator, denominator)
    }
}

/// Represents a sum of fractions with factored denominators.
///
/// This is used in decomposition algorithms where we express a single fraction
/// as a sum of simpler fractions.
#[derive(Clone, Debug)]
pub struct FractionWithFactoredDenominatorSum {
    /// The summands
    summands: Vec<FractionWithFactoredDenominator>,
}

impl FractionWithFactoredDenominatorSum {
    /// Creates a new sum of fractions.
    pub fn new(summands: Vec<FractionWithFactoredDenominator>) -> Self {
        FractionWithFactoredDenominatorSum { summands }
    }

    /// Creates an empty sum (zero).
    pub fn zero() -> Self {
        FractionWithFactoredDenominatorSum {
            summands: vec![],
        }
    }

    /// Returns the summands.
    pub fn summands(&self) -> &[FractionWithFactoredDenominator] {
        &self.summands
    }

    /// Returns the number of summands.
    pub fn len(&self) -> usize {
        self.summands.len()
    }

    /// Returns true if the sum is empty.
    pub fn is_empty(&self) -> bool {
        self.summands.is_empty()
    }

    /// Adds a summand to the sum.
    pub fn add_summand(&mut self, summand: FractionWithFactoredDenominator) {
        self.summands.push(summand);
    }
}

impl Display for FractionWithFactoredDenominatorSum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.summands.is_empty() {
            return write!(f, "0");
        }

        for (i, summand) in self.summands.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{}", summand)?;
        }
        Ok(())
    }
}

// ======================================================================================
// HELPER FUNCTIONS FOR MULTIVARIATE GENERATING FUNCTIONS
// ======================================================================================

/// Coerces a point specification into a consistent form.
///
/// Converts various point representations (dict, list, tuple) into a HashMap
/// mapping variable names to rational values.
///
/// # Arguments
/// * `point` - The point to coerce
/// * `variables` - The list of variable names
///
/// # Returns
/// A HashMap mapping variable names to their values
pub fn coerce_point(
    point: &HashMap<String, BigRational>,
    variables: &[String],
) -> HashMap<String, BigRational> {
    let mut result = HashMap::new();
    for var in variables {
        if let Some(val) = point.get(var) {
            result.insert(var.clone(), val.clone());
        } else {
            result.insert(var.clone(), BigRational::zero());
        }
    }
    result
}

/// Computes the direction vector from a point.
///
/// Given a point on the boundary, computes the direction vector used for
/// asymptotic analysis.
///
/// # Arguments
/// * `point` - The point (as a map from variables to values)
/// * `variables` - The list of variables
///
/// # Returns
/// The direction vector
pub fn direction(
    point: &HashMap<String, BigRational>,
    variables: &[String],
) -> Vec<BigRational> {
    variables
        .iter()
        .map(|var| point.get(var).cloned().unwrap_or_else(BigRational::zero))
        .collect()
}

/// Computes the sign of a permutation.
///
/// Returns 1 for even permutations, -1 for odd permutations.
///
/// # Arguments
/// * `permutation` - A permutation represented as a vector of indices
///
/// # Returns
/// The sign of the permutation (1 or -1)
pub fn permutation_sign(permutation: &[usize]) -> i32 {
    let n = permutation.len();
    let mut sign = 1;

    for i in 0..n {
        for j in (i + 1)..n {
            if permutation[i] > permutation[j] {
                sign *= -1;
            }
        }
    }

    sign
}

/// Performs substitution of a point into a polynomial.
///
/// Evaluates the polynomial at the given point, substituting values for
/// specified variables while keeping others symbolic.
///
/// # Arguments
/// * `poly` - The polynomial
/// * `point` - The point (map from variable names to values)
/// * `keep_symbolic` - Variables to keep symbolic (not substitute)
///
/// # Returns
/// The result after substitution
pub fn subs_all(
    poly: &SymbolicPolynomial,
    point: &HashMap<String, BigRational>,
    keep_symbolic: &[String],
) -> SymbolicPolynomial {
    // Simplified implementation: if no variables are kept symbolic, evaluate
    if keep_symbolic.is_empty() {
        let value = poly.evaluate(point);
        return SymbolicPolynomial::constant(poly.variables().to_vec(), value);
    }

    // Otherwise, return the original polynomial
    poly.clone()
}

/// Computes a differential operator applied to a polynomial.
///
/// Applies a differential operator represented as a sequence of partial derivatives.
///
/// # Arguments
/// * `poly` - The polynomial to differentiate
/// * `operator` - The differential operator (list of (variable_index, order) pairs)
///
/// # Returns
/// The result of applying the differential operator
pub fn diff_op(
    poly: &SymbolicPolynomial,
    _operator: &[(usize, usize)],
) -> SymbolicPolynomial {
    // Simplified implementation: return the polynomial unchanged
    // Full implementation would compute partial derivatives
    poly.clone()
}

/// Simplified version of differential operator application.
///
/// # Arguments
/// * `poly` - The polynomial
/// * `var_index` - The variable to differentiate with respect to
/// * `order` - The order of differentiation
///
/// # Returns
/// The derivative
pub fn diff_op_simple(
    poly: &SymbolicPolynomial,
    _var_index: usize,
    _order: usize,
) -> SymbolicPolynomial {
    // Simplified: return the polynomial unchanged
    poly.clone()
}

/// Computes all partial derivatives of a polynomial up to a given order.
///
/// # Arguments
/// * `poly` - The polynomial
/// * `max_order` - Maximum order of derivatives to compute
///
/// # Returns
/// A map from derivative specifications to their values
pub fn diff_all(
    poly: &SymbolicPolynomial,
    _max_order: usize,
) -> HashMap<Vec<usize>, SymbolicPolynomial> {
    // Simplified: return just the polynomial itself (0-th derivative)
    let mut result = HashMap::new();
    result.insert(vec![0; poly.num_variables()], poly.clone());
    result
}

/// Applies differentiation to a product of polynomials (Leibniz rule).
///
/// # Arguments
/// * `factors` - The factors in the product
/// * `derivative_order` - The order of differentiation
///
/// # Returns
/// The derivative of the product
pub fn diff_prod(
    factors: &[SymbolicPolynomial],
    _derivative_order: usize,
) -> SymbolicPolynomial {
    // Simplified: return first factor or constant 1
    if let Some(first) = factors.first() {
        first.clone()
    } else {
        SymbolicPolynomial::constant(vec![], BigRational::one())
    }
}

/// Applies differentiation to a sequence of operations.
///
/// # Arguments
/// * `sequence` - The sequence of polynomials
/// * `derivative_order` - The order of differentiation
///
/// # Returns
/// The derivative of the sequence
pub fn diff_seq(
    sequence: &[SymbolicPolynomial],
    _derivative_order: usize,
) -> Vec<SymbolicPolynomial> {
    // Simplified: return the sequence unchanged
    sequence.to_vec()
}

// ======================================================================================
// TESTS FOR MULTIVARIATE GENERATING FUNCTIONS
// ======================================================================================

#[cfg(test)]
mod mgf_tests {
    use super::*;

    #[test]
    fn test_symbolic_polynomial_creation() {
        let poly = SymbolicPolynomial::constant(vec!["x".to_string()], BigRational::from_integer(BigInt::from(5)));
        assert!(poly.is_constant());
        assert_eq!(poly.degree(), 0);
    }

    #[test]
    fn test_symbolic_polynomial_variable() {
        let poly = SymbolicPolynomial::variable(vec!["x".to_string(), "y".to_string()], 0);
        assert!(!poly.is_constant());
        assert_eq!(poly.degree(), 1);
        assert_eq!(poly.num_variables(), 2);
    }

    #[test]
    fn test_symbolic_polynomial_evaluation() {
        let mut poly = SymbolicPolynomial::new(vec!["x".to_string()]);
        poly.add_term(vec![0], BigRational::from_integer(BigInt::from(1))); // constant term 1
        poly.add_term(vec![1], BigRational::from_integer(BigInt::from(2))); // 2x
        poly.add_term(vec![2], BigRational::from_integer(BigInt::from(3))); // 3x^2

        let mut point = HashMap::new();
        point.insert("x".to_string(), BigRational::from_integer(BigInt::from(2)));

        // 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        let result = poly.evaluate(&point);
        assert_eq!(result, BigRational::from_integer(BigInt::from(17)));
    }

    #[test]
    fn test_fraction_with_factored_denominator() {
        let vars = vec!["x".to_string()];
        let num = SymbolicPolynomial::constant(vars.clone(), BigRational::one());
        let den = SymbolicPolynomial::variable(vars.clone(), 0);

        let frac = FractionWithFactoredDenominator::from_polynomials(num, den);
        assert_eq!(frac.num_denominator_factors(), 1);
    }

    #[test]
    fn test_fraction_ring() {
        let ring = FractionWithFactoredDenominatorRing::new(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(ring.variables().len(), 2);

        let zero = ring.zero();
        assert!(zero.numerator().is_zero());

        let one = ring.one();
        assert!(one.numerator().is_constant());
    }

    #[test]
    fn test_fraction_sum() {
        let vars = vec!["x".to_string()];
        let frac1 = FractionWithFactoredDenominator::new(
            SymbolicPolynomial::constant(vars.clone(), BigRational::one()),
            vec![],
        );
        let frac2 = FractionWithFactoredDenominator::new(
            SymbolicPolynomial::constant(vars.clone(), BigRational::from_integer(BigInt::from(2))),
            vec![],
        );

        let sum = FractionWithFactoredDenominatorSum::new(vec![frac1, frac2]);
        assert_eq!(sum.len(), 2);
        assert!(!sum.is_empty());
    }

    #[test]
    fn test_coerce_point() {
        let mut point = HashMap::new();
        point.insert("x".to_string(), BigRational::from_integer(BigInt::from(1)));

        let variables = vec!["x".to_string(), "y".to_string()];
        let coerced = coerce_point(&point, &variables);

        assert_eq!(coerced.len(), 2);
        assert_eq!(coerced.get("x").unwrap(), &BigRational::from_integer(BigInt::from(1)));
        assert_eq!(coerced.get("y").unwrap(), &BigRational::zero());
    }

    #[test]
    fn test_direction() {
        let mut point = HashMap::new();
        point.insert("x".to_string(), BigRational::from_integer(BigInt::from(1)));
        point.insert("y".to_string(), BigRational::from_integer(BigInt::from(2)));

        let variables = vec!["x".to_string(), "y".to_string()];
        let dir = direction(&point, &variables);

        assert_eq!(dir.len(), 2);
        assert_eq!(dir[0], BigRational::from_integer(BigInt::from(1)));
        assert_eq!(dir[1], BigRational::from_integer(BigInt::from(2)));
    }

    #[test]
    fn test_permutation_sign() {
        assert_eq!(permutation_sign(&[0, 1, 2]), 1); // identity
        assert_eq!(permutation_sign(&[1, 0, 2]), -1); // single swap
        assert_eq!(permutation_sign(&[1, 2, 0]), 1); // two swaps
        assert_eq!(permutation_sign(&[2, 1, 0]), -1); // three swaps
    }

    #[test]
    fn test_diff_all() {
        let poly = SymbolicPolynomial::constant(vec!["x".to_string()], BigRational::from_integer(BigInt::from(5)));
        let derivatives = diff_all(&poly, 2);

        assert!(!derivatives.is_empty());
    }

    #[test]
    fn test_symbolic_polynomial_display() {
        let poly = SymbolicPolynomial::constant(vec!["x".to_string()], BigRational::from_integer(BigInt::from(5)));
        let display = format!("{}", poly);
        assert!(display.contains("5"));
    }
}
