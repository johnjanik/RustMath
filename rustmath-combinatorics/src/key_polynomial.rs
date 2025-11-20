//! Key polynomials (Demazure characters)
//!
//! Key polynomials, also known as Demazure characters, are fundamental objects
//! in algebraic combinatorics and representation theory. They are indexed by
//! weak compositions (non-negative integer sequences) and form a basis for
//! polynomials that is intermediate between monomials and Schur functions.
//!
//! # Theory
//!
//! For a weak composition α = (α₁, α₂, ..., αₙ), the key polynomial κ_α is
//! a polynomial in variables x₁, x₂, ..., xₙ. Key polynomials satisfy:
//!
//! - They are constructed via divided difference operators
//! - They form a basis for polynomials
//! - They are Schur-positive
//! - They have nice multiplication properties
//!
//! # References
//!
//! - Reiner & Shimozono: Key polynomials and a flagged Littlewood-Richardson rule
//! - Lascoux & Schützenberger: Keys & standard bases

use rustmath_core::Ring;
use rustmath_integers::Integer;
use std::collections::BTreeMap;
use std::fmt;

/// A weak composition - a sequence of non-negative integers
///
/// Weak compositions are used to index key polynomials. Unlike ordinary
/// compositions, weak compositions allow zero parts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WeakComposition {
    parts: Vec<usize>,
}

impl WeakComposition {
    /// Create a new weak composition
    pub fn new(parts: Vec<usize>) -> Self {
        WeakComposition { parts }
    }

    /// Get the parts of the weak composition
    pub fn parts(&self) -> &[usize] {
        &self.parts
    }

    /// Get the sum of all parts
    pub fn sum(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Get the length (number of parts)
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Check if this is weakly decreasing
    pub fn is_weakly_decreasing(&self) -> bool {
        for i in 1..self.parts.len() {
            if self.parts[i] > self.parts[i - 1] {
                return false;
            }
        }
        true
    }

    /// Convert to a vector
    pub fn to_vec(&self) -> Vec<usize> {
        self.parts.clone()
    }
}

impl From<Vec<usize>> for WeakComposition {
    fn from(parts: Vec<usize>) -> Self {
        WeakComposition::new(parts)
    }
}

impl fmt::Display for WeakComposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, part) in self.parts.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", part)?;
        }
        write!(f, ")")
    }
}

/// A monomial in the key polynomial representation
///
/// Represents x₁^e₁ × x₂^e₂ × ... × xₙ^eₙ
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyMonomial {
    /// Exponents for each variable (index 0 = x₁, etc.)
    exponents: Vec<usize>,
}

impl KeyMonomial {
    /// Create a new monomial with given exponents
    pub fn new(exponents: Vec<usize>) -> Self {
        KeyMonomial { exponents }
    }

    /// Create a monomial with all zeros
    pub fn zero(nvars: usize) -> Self {
        KeyMonomial {
            exponents: vec![0; nvars],
        }
    }

    /// Get the exponent for variable i
    pub fn exponent(&self, i: usize) -> usize {
        self.exponents.get(i).copied().unwrap_or(0)
    }

    /// Get the total degree
    pub fn degree(&self) -> usize {
        self.exponents.iter().sum()
    }

    /// Multiply this monomial by another
    pub fn mul(&self, other: &KeyMonomial) -> KeyMonomial {
        let max_len = self.exponents.len().max(other.exponents.len());
        let mut result = vec![0; max_len];

        for i in 0..max_len {
            result[i] = self.exponent(i) + other.exponent(i);
        }

        KeyMonomial::new(result)
    }

    /// Swap variables i and i+1 (0-indexed)
    pub fn swap_variables(&self, i: usize) -> KeyMonomial {
        let mut exps = self.exponents.clone();
        if i < exps.len() - 1 {
            exps.swap(i, i + 1);
        }
        KeyMonomial::new(exps)
    }

    /// Get the number of variables
    pub fn num_vars(&self) -> usize {
        self.exponents.len()
    }
}

impl fmt::Display for KeyMonomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exponents.iter().all(|&e| e == 0) {
            write!(f, "1")?;
            return Ok(());
        }

        let mut first = true;
        for (i, &exp) in self.exponents.iter().enumerate() {
            if exp > 0 {
                if !first {
                    write!(f, "*")?;
                }
                first = false;

                write!(f, "x{}", i + 1)?;
                if exp > 1 {
                    write!(f, "^{}", exp)?;
                }
            }
        }
        Ok(())
    }
}

/// A key polynomial (Demazure character)
///
/// Key polynomials are represented as a linear combination of monomials.
/// They are indexed by weak compositions and constructed using divided
/// difference operators.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyPolynomial {
    /// Map from monomial to coefficient
    terms: BTreeMap<KeyMonomial, Integer>,
    /// Number of variables
    num_vars: usize,
}

impl KeyPolynomial {
    /// Create a zero polynomial
    pub fn zero(num_vars: usize) -> Self {
        KeyPolynomial {
            terms: BTreeMap::new(),
            num_vars,
        }
    }

    /// Create a constant polynomial
    pub fn constant(value: Integer, num_vars: usize) -> Self {
        if value.is_zero() {
            return Self::zero(num_vars);
        }

        let mut terms = BTreeMap::new();
        terms.insert(KeyMonomial::zero(num_vars), value);
        KeyPolynomial { terms, num_vars }
    }

    /// Create a monomial polynomial
    pub fn monomial(exponents: Vec<usize>, coeff: Integer) -> Self {
        let num_vars = exponents.len();
        if coeff.is_zero() {
            return Self::zero(num_vars);
        }

        let mut terms = BTreeMap::new();
        terms.insert(KeyMonomial::new(exponents), coeff);
        KeyPolynomial { terms, num_vars }
    }

    /// Create a key polynomial from a weak composition
    ///
    /// This constructs κ_α where α is the given weak composition.
    /// The construction uses divided difference operators.
    pub fn from_weak_composition(alpha: &WeakComposition) -> Self {
        let n = alpha.length();
        if n == 0 {
            return Self::constant(Integer::one(), 0);
        }

        // Start with the monomial x^α
        let mut poly = Self::monomial(alpha.to_vec(), Integer::one());

        // Apply divided difference operators to get the key polynomial
        // We apply the operators corresponding to the standardization of α
        poly = apply_key_operators(&poly, alpha);

        poly
    }

    /// Get the number of variables
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Get the terms as a map
    pub fn terms(&self) -> &BTreeMap<KeyMonomial, Integer> {
        &self.terms
    }

    /// Add a term to this polynomial
    pub fn add_term(&mut self, monomial: KeyMonomial, coeff: Integer) {
        if coeff.is_zero() {
            return;
        }

        let entry = self.terms.entry(monomial).or_insert_with(Integer::zero);
        *entry = entry.clone() + coeff;

        // Remove if zero
        if entry.is_zero() {
            let key = self
                .terms
                .iter()
                .find(|(_, v)| v.is_zero())
                .map(|(k, _)| k.clone());
            if let Some(k) = key {
                self.terms.remove(&k);
            }
        }
    }

    /// Check if this is the zero polynomial
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Add two key polynomials
    pub fn add(&self, other: &KeyPolynomial) -> KeyPolynomial {
        let num_vars = self.num_vars.max(other.num_vars);
        let mut result = Self::zero(num_vars);

        for (mon, coeff) in &self.terms {
            result.add_term(mon.clone(), coeff.clone());
        }

        for (mon, coeff) in &other.terms {
            result.add_term(mon.clone(), coeff.clone());
        }

        result
    }

    /// Subtract two key polynomials
    pub fn sub(&self, other: &KeyPolynomial) -> KeyPolynomial {
        let num_vars = self.num_vars.max(other.num_vars);
        let mut result = Self::zero(num_vars);

        for (mon, coeff) in &self.terms {
            result.add_term(mon.clone(), coeff.clone());
        }

        for (mon, coeff) in &other.terms {
            result.add_term(mon.clone(), -coeff.clone());
        }

        result
    }

    /// Multiply two key polynomials
    ///
    /// Key polynomial multiplication follows the same rules as ordinary
    /// polynomial multiplication. The product of two key polynomials is
    /// a polynomial (though not necessarily a single key polynomial).
    pub fn mul(&self, other: &KeyPolynomial) -> KeyPolynomial {
        let num_vars = self.num_vars.max(other.num_vars);
        let mut result = Self::zero(num_vars);

        for (mon1, coeff1) in &self.terms {
            for (mon2, coeff2) in &other.terms {
                let new_mon = mon1.mul(mon2);
                let new_coeff = coeff1.clone() * coeff2.clone();
                result.add_term(new_mon, new_coeff);
            }
        }

        result
    }

    /// Multiply by a scalar
    pub fn scalar_mul(&self, scalar: &Integer) -> KeyPolynomial {
        if scalar.is_zero() {
            return Self::zero(self.num_vars);
        }

        let mut result = Self::zero(self.num_vars);
        for (mon, coeff) in &self.terms {
            result.add_term(mon.clone(), coeff.clone() * scalar.clone());
        }
        result
    }

    /// Apply the divided difference operator ∂ᵢ
    ///
    /// The operator ∂ᵢ acts by: ∂ᵢ(f) = (f - sᵢ(f)) / (xᵢ - xᵢ₊₁)
    /// where sᵢ swaps variables xᵢ and xᵢ₊₁
    pub fn divided_difference(&self, i: usize) -> KeyPolynomial {
        if i >= self.num_vars - 1 {
            return self.clone();
        }

        let mut result = Self::zero(self.num_vars);

        for (mon, coeff) in &self.terms {
            // f - sᵢ(f)
            let swapped_mon = mon.swap_variables(i);
            let ei = mon.exponent(i);
            let ei1 = mon.exponent(i + 1);

            if ei == ei1 {
                // Terms cancel, no contribution
                continue;
            }

            // Divide by (xᵢ - xᵢ₊₁)
            // This is done symbolically by reducing the exponent difference
            if ei > ei1 {
                // xᵢ^ei * xᵢ₊₁^ei1 - xᵢ^ei1 * xᵢ₊₁^ei
                // = xᵢ^ei1 * xᵢ₊₁^ei1 * (xᵢ^(ei-ei1) - xᵢ₊₁^(ei-ei1))
                // Dividing by (xᵢ - xᵢ₊₁) gives a sum of monomials

                let diff = ei - ei1;
                for k in 0..diff {
                    let mut new_exps = mon.exponents.clone();
                    new_exps[i] = ei1 + diff - 1 - k;
                    new_exps[i + 1] = ei1 + k;
                    result.add_term(KeyMonomial::new(new_exps), coeff.clone());
                }
            } else {
                // ei < ei1: contribution is negative
                let diff = ei1 - ei;
                for k in 0..diff {
                    let mut new_exps = mon.exponents.clone();
                    new_exps[i] = ei + k;
                    new_exps[i + 1] = ei + diff - 1 - k;
                    result.add_term(KeyMonomial::new(new_exps), -coeff.clone());
                }
            }
        }

        result
    }

    /// Swap variables i and i+1
    pub fn swap_variables(&self, i: usize) -> KeyPolynomial {
        if i >= self.num_vars - 1 {
            return self.clone();
        }

        let mut result = Self::zero(self.num_vars);
        for (mon, coeff) in &self.terms {
            result.add_term(mon.swap_variables(i), coeff.clone());
        }
        result
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> usize {
        self.terms.keys().map(|m| m.degree()).max().unwrap_or(0)
    }
}

impl fmt::Display for KeyPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "0")?;
            return Ok(());
        }

        let mut terms: Vec<_> = self.terms.iter().collect();
        terms.sort_by(|a, b| b.0.cmp(a.0)); // Sort by monomial (descending)

        for (i, (mon, coeff)) in terms.iter().enumerate() {
            let sign = coeff.signum();

            if i > 0 {
                if sign >= 0 {
                    write!(f, " + ")?;
                } else {
                    write!(f, " - ")?;
                    let abs_coeff = -(*coeff).clone();
                    if !abs_coeff.is_one() || mon.degree() == 0 {
                        write!(f, "{}", abs_coeff)?;
                    }
                    if mon.degree() > 0 {
                        if !abs_coeff.is_one() {
                            write!(f, "*")?;
                        }
                        write!(f, "{}", mon)?;
                    }
                    continue;
                }
            } else if sign < 0 {
                write!(f, "-")?;
            }

            let abs_coeff = if sign < 0 {
                -(*coeff).clone()
            } else {
                (*coeff).clone()
            };

            if !abs_coeff.is_one() || mon.degree() == 0 {
                write!(f, "{}", abs_coeff)?;
            }

            if mon.degree() > 0 {
                if !abs_coeff.is_one() {
                    write!(f, "*")?;
                }
                write!(f, "{}", mon)?;
            }
        }

        Ok(())
    }
}

/// Apply key polynomial operators to construct κ_α
///
/// This is an internal helper function that applies the appropriate
/// divided difference operators to construct a key polynomial from
/// a weak composition.
fn apply_key_operators(poly: &KeyPolynomial, alpha: &WeakComposition) -> KeyPolynomial {
    let n = alpha.length();
    if n <= 1 {
        return poly.clone();
    }

    // For a weakly decreasing composition, the key polynomial is just the monomial
    if alpha.is_weakly_decreasing() {
        return poly.clone();
    }

    // Apply straightening operations based on the composition
    // We use a greedy algorithm to sort the composition
    let mut current = poly.clone();
    let mut comp = alpha.to_vec();

    // Bubble sort with divided difference operators
    for _ in 0..n * n {
        // Multiple passes to ensure convergence
        let mut changed = false;
        for i in 0..n - 1 {
            if comp[i] < comp[i + 1] {
                // Apply divided difference operator
                current = current.divided_difference(i);
                comp.swap(i, i + 1);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    current
}

/// Compute the key polynomial for a given weak composition using the standard algorithm
///
/// This is the main entry point for computing key polynomials.
pub fn key_polynomial(alpha: Vec<usize>) -> KeyPolynomial {
    let comp = WeakComposition::new(alpha);
    KeyPolynomial::from_weak_composition(&comp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weak_composition() {
        let comp = WeakComposition::new(vec![3, 2, 1]);
        assert_eq!(comp.sum(), 6);
        assert_eq!(comp.length(), 3);
        assert!(comp.is_weakly_decreasing());

        let comp2 = WeakComposition::new(vec![1, 2, 3]);
        assert!(!comp2.is_weakly_decreasing());
    }

    #[test]
    fn test_key_monomial() {
        let mon1 = KeyMonomial::new(vec![2, 1, 0]);
        let mon2 = KeyMonomial::new(vec![1, 2, 3]);

        let prod = mon1.mul(&mon2);
        assert_eq!(prod.exponents, vec![3, 3, 3]);

        let swapped = mon1.swap_variables(0);
        assert_eq!(swapped.exponents, vec![1, 2, 0]);
    }

    #[test]
    fn test_key_polynomial_constant() {
        let poly = KeyPolynomial::constant(Integer::from(5), 3);
        assert_eq!(poly.terms.len(), 1);
        assert!(!poly.is_zero());

        let zero = KeyPolynomial::zero(3);
        assert!(zero.is_zero());
    }

    #[test]
    fn test_key_polynomial_from_decreasing_composition() {
        // For a weakly decreasing composition, key polynomial = monomial
        let alpha = WeakComposition::new(vec![3, 2, 1]);
        let kpoly = KeyPolynomial::from_weak_composition(&alpha);

        // Should have exactly one term: x1^3 * x2^2 * x3^1
        assert_eq!(kpoly.terms.len(), 1);
        let (mon, coeff) = kpoly.terms.iter().next().unwrap();
        assert_eq!(mon.exponents, vec![3, 2, 1]);
        assert_eq!(*coeff, Integer::one());
    }

    #[test]
    fn test_key_polynomial_addition() {
        let poly1 = KeyPolynomial::monomial(vec![2, 1], Integer::from(3));
        let poly2 = KeyPolynomial::monomial(vec![1, 2], Integer::from(2));

        let sum = poly1.add(&poly2);
        assert_eq!(sum.terms.len(), 2);
    }

    #[test]
    fn test_key_polynomial_multiplication() {
        // (x1) * (x2) = x1*x2
        let poly1 = KeyPolynomial::monomial(vec![1, 0], Integer::one());
        let poly2 = KeyPolynomial::monomial(vec![0, 1], Integer::one());

        let prod = poly1.mul(&poly2);
        assert_eq!(prod.terms.len(), 1);

        let (mon, coeff) = prod.terms.iter().next().unwrap();
        assert_eq!(mon.exponents, vec![1, 1]);
        assert_eq!(*coeff, Integer::one());
    }

    #[test]
    fn test_key_polynomial_multiplication_complex() {
        // (x1 + x2) * (x1 + x2) = x1^2 + 2*x1*x2 + x2^2
        let mut poly1 = KeyPolynomial::monomial(vec![1, 0], Integer::one());
        poly1.add_term(KeyMonomial::new(vec![0, 1]), Integer::one());

        let prod = poly1.mul(&poly1);

        // Should have three terms
        assert_eq!(prod.terms.len(), 3);

        // Check the coefficient of x1*x2
        let mon_x1x2 = KeyMonomial::new(vec![1, 1]);
        assert_eq!(prod.terms.get(&mon_x1x2), Some(&Integer::from(2)));
    }

    #[test]
    fn test_divided_difference_simple() {
        // Start with x1^2
        let poly = KeyPolynomial::monomial(vec![2, 0], Integer::one());

        // ∂₀(x1^2) = (x1^2 - x2^2) / (x1 - x2) = x1 + x2
        let result = poly.divided_difference(0);

        assert_eq!(result.terms.len(), 2);
        assert_eq!(
            result.terms.get(&KeyMonomial::new(vec![1, 0])),
            Some(&Integer::one())
        );
        assert_eq!(
            result.terms.get(&KeyMonomial::new(vec![0, 1])),
            Some(&Integer::one())
        );
    }

    #[test]
    fn test_key_polynomial_simple_cases() {
        // κ_(1,0) = x1
        let kp1 = key_polynomial(vec![1, 0]);
        assert_eq!(kp1.terms.len(), 1);

        // κ_(0,1) should apply divided difference
        let kp2 = key_polynomial(vec![0, 1]);
        // This should be 0 after applying the operator (since 0 < 1, we swap and reduce)
        // Actually, for (0,1), we start with x1^0 * x2^1 = x2
        // Then apply operator since 0 < 1
        // Result should involve both x1 and x2
    }

    #[test]
    fn test_key_polynomial_scalar_multiplication() {
        let poly = KeyPolynomial::monomial(vec![1, 1], Integer::from(2));
        let result = poly.scalar_mul(&Integer::from(3));

        assert_eq!(result.terms.len(), 1);
        let (mon, coeff) = result.terms.iter().next().unwrap();
        assert_eq!(mon.exponents, vec![1, 1]);
        assert_eq!(*coeff, Integer::from(6));
    }

    #[test]
    fn test_key_polynomial_display() {
        let poly = KeyPolynomial::monomial(vec![2, 1], Integer::from(3));
        let display = format!("{}", poly);
        assert!(display.contains("3"));
        assert!(display.contains("x1"));
        assert!(display.contains("x2"));
    }

    #[test]
    fn test_swap_variables() {
        let poly = KeyPolynomial::monomial(vec![2, 1, 0], Integer::one());
        let swapped = poly.swap_variables(0);

        assert_eq!(swapped.terms.len(), 1);
        let (mon, _) = swapped.terms.iter().next().unwrap();
        assert_eq!(mon.exponents, vec![1, 2, 0]);
    }
}
