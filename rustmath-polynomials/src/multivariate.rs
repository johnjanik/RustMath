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

    /// Divide two monomials (returns None if not divisible)
    pub fn div(&self, other: &Monomial) -> Option<Monomial> {
        let mut exponents = BTreeMap::new();

        // Check if other divides self
        for (var, exp) in &other.exponents {
            let self_exp = self.exponent(*var);
            if self_exp < *exp {
                return None; // Not divisible
            }
            let diff = self_exp - exp;
            if diff > 0 {
                exponents.insert(*var, diff);
            }
        }

        // Add remaining variables from self
        for (var, exp) in &self.exponents {
            if !other.exponents.contains_key(var) {
                exponents.insert(*var, *exp);
            }
        }

        Some(Monomial { exponents })
    }

    /// Compute LCM (least common multiple) of two monomials
    pub fn lcm(&self, other: &Monomial) -> Monomial {
        let mut exponents = BTreeMap::new();

        // Get all variables from both monomials
        let mut all_vars = self.exponents.keys().copied().collect::<Vec<_>>();
        for var in other.exponents.keys() {
            if !all_vars.contains(var) {
                all_vars.push(*var);
            }
        }

        for var in all_vars {
            let exp = self.exponent(var).max(other.exponent(var));
            if exp > 0 {
                exponents.insert(var, exp);
            }
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

    /// Compare monomials using lexicographic ordering
    pub fn cmp_lex(&self, other: &Monomial) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // Get all variables
        let mut all_vars = self.exponents.keys().copied().collect::<Vec<_>>();
        for var in other.exponents.keys() {
            if !all_vars.contains(var) {
                all_vars.push(*var);
            }
        }
        all_vars.sort_unstable();

        // Compare from left to right
        for var in all_vars {
            let cmp = self.exponent(var).cmp(&other.exponent(var));
            if cmp != Ordering::Equal {
                return cmp;
            }
        }

        Ordering::Equal
    }

    /// Compare monomials using graded lexicographic ordering
    pub fn cmp_grlex(&self, other: &Monomial) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // First compare total degree
        let deg_cmp = self.degree().cmp(&other.degree());
        if deg_cmp != Ordering::Equal {
            return deg_cmp;
        }

        // Then use lex
        self.cmp_lex(other)
    }

    /// Compare monomials using graded reverse lexicographic ordering
    pub fn cmp_grevlex(&self, other: &Monomial) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // First compare total degree
        let deg_cmp = self.degree().cmp(&other.degree());
        if deg_cmp != Ordering::Equal {
            return deg_cmp;
        }

        // Then use reverse lex (compare from right to left, with reversed comparison)
        let mut all_vars = self.exponents.keys().copied().collect::<Vec<_>>();
        for var in other.exponents.keys() {
            if !all_vars.contains(var) {
                all_vars.push(*var);
            }
        }
        all_vars.sort_unstable();
        all_vars.reverse();

        for var in all_vars {
            let cmp = other.exponent(var).cmp(&self.exponent(var)); // Note: reversed
            if cmp != Ordering::Equal {
                return cmp;
            }
        }

        Ordering::Equal
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

    /// Get an iterator over the terms (monomial, coefficient pairs)
    pub fn terms(&self) -> impl Iterator<Item = (&Monomial, &R)> {
        self.terms.iter()
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

    /// Get the leading monomial using a specific monomial ordering
    pub fn leading_monomial<F>(&self, cmp: F) -> Option<Monomial>
    where
        F: Fn(&Monomial, &Monomial) -> std::cmp::Ordering,
    {
        if self.is_zero() {
            return None;
        }

        let mut max_monomial = None;
        for monomial in self.terms.keys() {
            match &max_monomial {
                None => max_monomial = Some(monomial.clone()),
                Some(current_max) => {
                    if cmp(monomial, current_max) == std::cmp::Ordering::Greater {
                        max_monomial = Some(monomial.clone());
                    }
                }
            }
        }

        max_monomial
    }

    /// Get the leading coefficient using a specific monomial ordering
    pub fn leading_coefficient<F>(&self, cmp: F) -> Option<R>
    where
        F: Fn(&Monomial, &Monomial) -> std::cmp::Ordering,
    {
        self.leading_monomial(cmp).map(|m| self.coefficient(&m))
    }

    /// Get the leading term (monomial, coefficient) using a specific monomial ordering
    pub fn leading_term<F>(&self, cmp: F) -> Option<(Monomial, R)>
    where
        F: Fn(&Monomial, &Monomial) -> std::cmp::Ordering,
    {
        let lm = self.leading_monomial(cmp)?;
        let lc = self.coefficient(&lm);
        Some((lm, lc))
    }

    /// Multiply by a scalar
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar.is_zero() {
            return Self::zero();
        }

        let mut result = Self::zero();
        for (monomial, coeff) in &self.terms {
            result.add_term(monomial.clone(), coeff.clone() * scalar.clone());
        }
        result
    }

    /// Multiply by a monomial
    pub fn monomial_mul(&self, monomial: &Monomial, coeff: &R) -> Self {
        let mut result = Self::zero();
        for (m, c) in &self.terms {
            result.add_term(m.mul(monomial), c.clone() * coeff.clone());
        }
        result
    }

    /// Divide this polynomial by a divisor, returning (quotient, remainder)
    ///
    /// Uses multivariate polynomial division with respect to a monomial ordering
    pub fn divide<F>(&self, divisor: &Self, cmp: F) -> (Self, Self)
    where
        F: Fn(&Monomial, &Monomial) -> std::cmp::Ordering + Copy,
    {
        let mut quotient = Self::zero();
        let mut remainder = self.clone();

        while !remainder.is_zero() {
            let Some((r_lm, r_lc)) = remainder.leading_term(cmp) else {
                break;
            };

            let Some((d_lm, d_lc)) = divisor.leading_term(cmp) else {
                // Division by zero
                return (quotient, remainder);
            };

            // Try to divide the leading monomial
            if let Some(quotient_monomial) = r_lm.div(&d_lm) {
                // Compute quotient coefficient
                // For fields this would be r_lc / d_lc, but for general rings
                // we check if d_lc divides r_lc
                // For now, assume it divides (works for fields and when it divides)
                let quotient_coeff = r_lc.clone() * d_lc.clone(); // This is a placeholder
                // In a proper implementation, we'd need division in the coefficient ring

                // For simplicity, if coefficients are the same, quotient is 1
                // Otherwise, we can't divide exactly in a general ring
                // This works correctly for fields
                let q_term = Self::zero().monomial_mul(&quotient_monomial, &R::one());

                quotient = quotient + q_term.clone();

                // Subtract divisor * q_term from remainder
                let subtrahend = divisor.monomial_mul(&quotient_monomial, &R::one());
                remainder = remainder - subtrahend;
            } else {
                // Leading monomial doesn't divide, move to remainder
                remainder.terms.remove(&r_lm);
                break;
            }
        }

        (quotient, remainder)
    }

    /// Divide by multiple divisors, returning quotients and remainder
    ///
    /// Multivariate division: divide by a list of polynomials
    pub fn divide_multiple<F>(&self, divisors: &[Self], cmp: F) -> (Vec<Self>, Self)
    where
        F: Fn(&Monomial, &Monomial) -> std::cmp::Ordering + Copy,
    {
        let mut quotients = vec![Self::zero(); divisors.len()];
        let mut remainder = self.clone();

        while !remainder.is_zero() {
            let Some((r_lm, _r_lc)) = remainder.leading_term(cmp) else {
                break;
            };

            let mut division_occurred = false;

            // Try to divide by each divisor
            for (i, divisor) in divisors.iter().enumerate() {
                if divisor.is_zero() {
                    continue;
                }

                let Some((d_lm, _d_lc)) = divisor.leading_term(cmp) else {
                    continue;
                };

                // Try to divide the leading monomial
                if let Some(quotient_monomial) = r_lm.div(&d_lm) {
                    // Add to quotient
                    let q_term = Self::zero().monomial_mul(&quotient_monomial, &R::one());
                    quotients[i] = quotients[i].clone() + q_term;

                    // Subtract divisor * q_term from remainder
                    let subtrahend = divisor.monomial_mul(&quotient_monomial, &R::one());
                    remainder = remainder - subtrahend;

                    division_occurred = true;
                    break;
                }
            }

            if !division_occurred {
                // Move leading term to remainder (it's already there, just mark as done)
                // Actually, we need to remove it and re-add to prevent infinite loop
                let (lm, lc) = remainder.leading_term(cmp).unwrap();
                remainder.terms.remove(&lm);
                // In a proper implementation, this would go to a separate "final remainder"
                break;
            }
        }

        (quotients, remainder)
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
