//! Laurent polynomials - polynomials with both positive and negative powers
//!
//! A Laurent polynomial is a polynomial that can have negative powers of the variable.
//! It has the form: a_{-n}*x^{-n} + ... + a_{-1}*x^{-1} + a_0 + a_1*x + ... + a_m*x^m
//!
//! Ring hooks (MAGMA ch. 23 §23.4.4 `IsUnit`, `IsNilpotent`; ch. 17 §17.4.2-17.4.3
//! `Characteristic`, `IsCommutative`), for `f = Σ_{i=a}^{b} c_i x^i` with `c_a, c_b != 0`:
//! a monomial `c x^k` is a unit (nilpotent) iff `c` is, since `x^k` is a central unit and
//! `x ↦ 1` is a ring map to `R`. A non-monomial is not nilpotent when `c_a` or `c_b` is
//! not, and is not a unit when both are units (in every ring: a product with `g != 0` keeps
//! two extreme terms). Over a commutative `R`: `f` is nilpotent iff every coefficient is;
//! `f` is not a unit when `c_a c_b` is not nilpotent (a prime `P` with `c_a c_b ∉ P` maps `f`
//! to a non-monomial over the domain `R/P`, whose Laurent units are monomials); and `f` is a
//! unit iff, when `c_a` (or `c_b`) is a unit, the other coefficients are nilpotent
//! (Atiyah-Macdonald, *Introduction to Commutative Algebra*, Ch. 1, Exercise 2, on
//! `x^(-a) f` or `x^b f(1/x)`). The inverse is `x^(-a) p^(-1)` or `x^(-b) q^(-1)(1/x)` with the
//! polynomial inverse of [`crate::UnivariatePolynomial`]. A Laurent unit need not have a unit
//! extreme coefficient (`3 + 4x` over Z/6 has the inverse `3 + 4x^(-1)`): such units stay
//! undecided, since deciding them needs the idempotents of `R`.

use crate::univariate::{UnivariatePolynomial, UNIT_INVERSE_WORK};
use rustmath_core::{MathError, MultiplicativeOrder, Result, Ring};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A Laurent polynomial over a ring R
///
/// Represented as a sparse map from exponents (which can be negative) to coefficients
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LaurentPolynomial<R: Ring> {
    /// Map from exponent to coefficient
    /// Only non-zero coefficients are stored
    terms: BTreeMap<i32, R>,
}

impl<R: Ring> LaurentPolynomial<R> {
    /// Create a new Laurent polynomial
    pub fn new() -> Self {
        LaurentPolynomial {
            terms: BTreeMap::new(),
        }
    }

    /// Create a Laurent polynomial from a monomial c*x^n
    pub fn monomial(coeff: R, power: i32) -> Self {
        if coeff.is_zero() {
            return LaurentPolynomial::new();
        }

        let mut terms = BTreeMap::new();
        terms.insert(power, coeff);
        LaurentPolynomial { terms }
    }

    /// Create a constant Laurent polynomial
    pub fn constant(c: R) -> Self {
        Self::monomial(c, 0)
    }

    /// Create the variable x
    pub fn var() -> Self {
        Self::monomial(R::one(), 1)
    }

    /// Add a term to this polynomial. No `R::zero()` is created (it panics for
    /// parent-carrying coefficient types): a new exponent stores `coeff` itself.
    pub fn add_term(&mut self, coeff: R, power: i32) {
        if coeff.is_zero() {
            return;
        }

        match self.terms.get_mut(&power) {
            Some(entry) => {
                *entry = entry.clone() + coeff;
                // Remove if becomes zero
                if entry.is_zero() {
                    self.terms.remove(&power);
                }
            }
            None => {
                self.terms.insert(power, coeff);
            }
        }
    }

    /// Get the coefficient of x^n
    pub fn coeff(&self, power: i32) -> R {
        self.terms.get(&power).cloned().unwrap_or_else(R::zero)
    }

    /// Check if this is the zero polynomial
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get the minimal exponent (lowest power)
    pub fn min_exponent(&self) -> Option<i32> {
        self.terms.keys().next().copied()
    }

    /// Get the maximal exponent (highest power)
    pub fn max_exponent(&self) -> Option<i32> {
        self.terms.keys().next_back().copied()
    }

    /// Get all terms as a vector of (power, coefficient) pairs
    pub fn terms(&self) -> Vec<(i32, R)> {
        self.terms
            .iter()
            .map(|(p, c)| (*p, c.clone()))
            .collect()
    }
}

impl<R: Ring> Default for LaurentPolynomial<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> fmt::Display for LaurentPolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        // Iterate in descending order of powers
        for (&power, coeff) in self.terms.iter().rev() {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if power == 0 {
                write!(f, "{}", coeff)?;
            } else if power == 1 {
                if coeff.is_one() {
                    write!(f, "q")?;
                } else {
                    write!(f, "{}*q", coeff)?;
                }
            } else if power < 0 {
                if coeff.is_one() {
                    write!(f, "q^({})", power)?;
                } else {
                    write!(f, "{}*q^({})", coeff, power)?;
                }
            } else {
                // power > 1
                if coeff.is_one() {
                    write!(f, "q^{}", power)?;
                } else {
                    write!(f, "{}*q^{}", coeff, power)?;
                }
            }
        }
        Ok(())
    }
}

impl<R: Ring> Add for LaurentPolynomial<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        for (power, coeff) in other.terms {
            result.add_term(coeff, power);
        }
        result
    }
}

impl<R: Ring> Sub for LaurentPolynomial<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<R: Ring> Neg for LaurentPolynomial<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut result = LaurentPolynomial::new();
        for (power, coeff) in self.terms {
            result.add_term(-coeff, power);
        }
        result
    }
}

impl<R: Ring> Mul for LaurentPolynomial<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return LaurentPolynomial::new();
        }

        let mut result = LaurentPolynomial::new();

        for (p1, c1) in &self.terms {
            for (p2, c2) in &other.terms {
                let new_power = p1 + p2;
                let new_coeff = c1.clone() * c2.clone();
                result.add_term(new_coeff, new_power);
            }
        }

        result
    }
}

/// A term `(exponent, coefficient)` borrowed from the sparse map.
type Term<'a, R> = (i32, &'a R);

/// The extreme terms `(a, c_a)` and `(b, c_b)` of a nonzero Laurent polynomial.
fn extremes<R: Ring>(terms: &BTreeMap<i32, R>) -> Option<(Term<'_, R>, Term<'_, R>)> {
    let (&a, ca) = terms.iter().next()?;
    let (&b, cb) = terms.iter().next_back()?;
    Some(((a, ca), (b, cb)))
}

/// The dense coefficients `[c_(from), c_(from ± 1), ...]` of the sparse terms, walking up
/// (`up`) from `from` to `to` or down. Gaps are filled with `c - c` for a stored `c`, so no
/// `R::zero()` is created.
fn dense<R: Ring>(terms: &BTreeMap<i32, R>, from: i32, to: i32, up: bool) -> Vec<R> {
    let (_, first) = terms.iter().next().expect("nonzero");
    let zero = first.clone() - first.clone();
    let len = (to as i64 - from as i64).unsigned_abs() as usize + 1;
    (0..len)
        .map(|j| {
            let e = if up { from as i64 + j as i64 } else { from as i64 - j as i64 };
            terms
                .get(&(e as i32))
                .cloned()
                .unwrap_or_else(|| zero.clone())
        })
        .collect()
}

/// The sparse product of two nonzero Laurent polynomials, without `R::zero()`; `None` when
/// an exponent sum overflows `i32`.
fn sparse_product<R: Ring>(
    x: &BTreeMap<i32, R>,
    y: &BTreeMap<i32, R>,
) -> Option<BTreeMap<i32, R>> {
    let mut out: BTreeMap<i32, R> = BTreeMap::new();
    for (&e1, c1) in x {
        for (&e2, c2) in y {
            let e = e1.checked_add(e2)?;
            let t = c1.clone() * c2.clone();
            match out.get_mut(&e) {
                Some(v) => *v = v.clone() + t,
                None => {
                    out.insert(e, t);
                }
            }
        }
    }
    out.retain(|_, v| !v.is_zero());
    Some(out)
}

impl<R: Ring> Ring for LaurentPolynomial<R> {
    fn zero() -> Self {
        LaurentPolynomial::new()
    }

    fn one() -> Self {
        LaurentPolynomial::constant(R::one())
    }

    fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    fn is_one(&self) -> bool {
        self.terms.len() == 1 && {
            let (power, coeff) = self.terms.iter().next().unwrap();
            *power == 0 && coeff.is_one()
        }
    }

    /// See the module documentation for the rules.
    fn unit_status(&self) -> Option<bool> {
        let ((_, ca), (_, cb)) = match extremes(&self.terms) {
            None => return Some(false), // 0 (in a nonzero ring)
            Some(e) => e,
        };
        if self.terms.len() == 1 {
            return ca.unit_status();
        }
        if ca.unit_status() == Some(true) && cb.unit_status() == Some(true) {
            return Some(false);
        }
        if R::static_is_commutative() == Some(true) {
            if (ca.clone() * cb.clone()).nilpotent_status() == Some(false) {
                return Some(false);
            }
            for (extreme, other_side) in [(ca, true), (cb, false)] {
                if extreme.unit_status() != Some(true) {
                    continue;
                }
                // The coefficients other than this extreme one.
                let rest: Vec<&R> = if other_side {
                    self.terms.values().skip(1).collect()
                } else {
                    self.terms.values().take(self.terms.len() - 1).collect()
                };
                if rest.iter().any(|c| c.nilpotent_status() == Some(false)) {
                    return Some(false);
                }
                if rest.iter().all(|c| c.nilpotent_status() == Some(true)) {
                    return Some(true);
                }
            }
        }
        None
    }

    /// See the module documentation for the rules.
    fn nilpotent_status(&self) -> Option<bool> {
        let ((_, ca), (_, cb)) = match extremes(&self.terms) {
            None => return Some(true),
            Some(e) => e,
        };
        if self.terms.len() == 1 {
            return ca.nilpotent_status();
        }
        if ca.nilpotent_status() == Some(false) || cb.nilpotent_status() == Some(false) {
            return Some(false);
        }
        if R::static_is_commutative() == Some(true) {
            if self.terms.values().any(|c| c.nilpotent_status() == Some(false)) {
                return Some(false);
            }
            if self.terms.values().all(|c| c.nilpotent_status() == Some(true)) {
                return Some(true);
            }
        }
        None
    }

    /// The inverse of a unit. A monomial `c x^k`: `c^(-1) x^(-k)` (`Err(NotImplemented)` for
    /// `k = i32::MIN`, whose negative is not an `i32`: the monomial is still a unit). A
    /// non-monomial: `Err(NotInvertible)` on a proven non-unit; otherwise `x^(-a) p^(-1)`
    /// when `c_a` is invertible (`p = x^(-a) f`), else `x^(-b) q^(-1)(1/x)` when `c_b` is
    /// (`q(y) = y^b f(1/y)`), with the checked two-sided polynomial inverse of
    /// [`UnivariatePolynomial::unit_inverse`]. Otherwise `Err(NotImplemented)`. Exponent
    /// overflow gives `Err(NotImplemented)`.
    fn unit_inverse(&self) -> Result<Self> {
        let ((a, ca), (b, cb)) = match extremes(&self.terms) {
            None => return Err(MathError::NotInvertible),
            Some(e) => e,
        };
        if self.terms.len() == 1 {
            let inv = ca.unit_inverse()?;
            let k = a.checked_neg().ok_or_else(|| {
                MathError::NotImplemented(format!(
                    "LaurentPolynomial::unit_inverse: the exponent {a} has no negative in i32"
                ))
            })?;
            return Ok(Self::monomial(inv, k));
        }
        if self.unit_status() == Some(false) {
            return Err(MathError::NotInvertible);
        }
        let overflow = || {
            MathError::NotImplemented(
                "LaurentPolynomial::unit_inverse: an exponent of the inverse overflows i32"
                    .to_string(),
            )
        };
        let span = (b as i64 - a as i64) as u64;
        if span.saturating_mul(span + 1) / 2 > UNIT_INVERSE_WORK {
            return Err(MathError::NotImplemented(format!(
                "LaurentPolynomial::unit_inverse: the exponent span {span} is beyond the work \
                 bound of {UNIT_INVERSE_WORK} coefficient products"
            )));
        }
        // Low side: f = x^a p(x), f^(-1) = x^(-a) p^(-1)(x).
        if ca.unit_inverse().is_ok() {
            let p = UnivariatePolynomial::new(dense(&self.terms, a, b, true));
            match p.unit_inverse() {
                Ok(u) => {
                    let shift = a.checked_neg().ok_or_else(overflow)?;
                    let mut out = Self::new();
                    for (j, c) in u.coefficients().iter().enumerate() {
                        if c.is_zero() {
                            continue;
                        }
                        let e = i32::try_from(j)
                            .ok()
                            .and_then(|j| shift.checked_add(j))
                            .ok_or_else(overflow)?;
                        out.terms.insert(e, c.clone());
                    }
                    return Ok(out);
                }
                // A Laurent inverse of p would be its right inverse in R[x], which the
                // polynomial checks proved is not a left inverse (or p is a proven non-unit,
                // which the rules above already report for f).
                Err(MathError::NotInvertible) => return Err(MathError::NotInvertible),
                Err(_) => {}
            }
        }
        // High side: f = x^b q(1/x), f^(-1) = x^(-b) w(1/x) with w = q^(-1).
        if cb.unit_inverse().is_ok() {
            let q = UnivariatePolynomial::new(dense(&self.terms, b, a, false));
            match q.unit_inverse() {
                Ok(w) => {
                    let shift = b.checked_neg().ok_or_else(overflow)?;
                    let mut out = Self::new();
                    for (j, c) in w.coefficients().iter().enumerate() {
                        if c.is_zero() {
                            continue;
                        }
                        let e = i32::try_from(j)
                            .ok()
                            .and_then(|j| shift.checked_sub(j))
                            .ok_or_else(overflow)?;
                        out.terms.insert(e, c.clone());
                    }
                    return Ok(out);
                }
                Err(MathError::NotInvertible) => return Err(MathError::NotInvertible),
                Err(_) => {}
            }
        }
        Err(MathError::NotImplemented(
            "LaurentPolynomial::unit_inverse: no extreme coefficient gives a computed inverse \
             (a Laurent unit need not have a unit extreme coefficient)"
                .to_string(),
        ))
    }

    /// MAGMA `Order(f)`. A monomial `c x^k`: as `c` for `k = 0`; for `k != 0`, `Infinite`
    /// when `c` is a proven unit (`(c x^k)^m = c^m x^(km) != 1`), `NotAUnit` when `c` is a
    /// proven non-unit. A non-monomial: `NotAUnit` on a proven non-unit; for a proven unit
    /// the first `k` with `f^k = 1` within 65536 coefficient products is `Finite(k)`;
    /// otherwise `Undecided`.
    fn multiplicative_order_status(&self) -> MultiplicativeOrder {
        let ((a, ca), _) = match extremes(&self.terms) {
            None => return MultiplicativeOrder::NotAUnit,
            Some(e) => e,
        };
        if self.terms.len() == 1 {
            if a == 0 {
                return ca.multiplicative_order_status();
            }
            return match ca.unit_status() {
                Some(true) => MultiplicativeOrder::Infinite,
                Some(false) => MultiplicativeOrder::NotAUnit,
                None => MultiplicativeOrder::Undecided,
            };
        }
        match self.unit_status() {
            Some(false) => MultiplicativeOrder::NotAUnit,
            Some(true) => {
                let mut power = self.terms.clone();
                let mut work: u64 = 0;
                let mut k: u64 = 1;
                loop {
                    if power.len() == 1 && power.get(&0).is_some_and(|c| c.is_one()) {
                        return MultiplicativeOrder::Finite(k);
                    }
                    work = work.saturating_add((power.len() * self.terms.len()) as u64);
                    if work > UNIT_INVERSE_WORK {
                        return MultiplicativeOrder::Undecided;
                    }
                    power = match sparse_product(&power, &self.terms) {
                        Some(next) => next,
                        None => return MultiplicativeOrder::Undecided,
                    };
                    k += 1;
                }
            }
            None => MultiplicativeOrder::Undecided,
        }
    }

    fn static_characteristic() -> Option<u64> {
        R::static_characteristic()
    }

    fn static_is_commutative() -> Option<bool> {
        R::static_is_commutative()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_monomial() {
        let p = LaurentPolynomial::<Integer>::monomial(Integer::from(3), 2);
        assert_eq!(p.coeff(2), Integer::from(3));
        assert_eq!(p.coeff(0), Integer::zero());
    }

    #[test]
    fn test_negative_power() {
        let p = LaurentPolynomial::<Integer>::monomial(Integer::from(1), -2);
        assert_eq!(p.coeff(-2), Integer::from(1));
        assert_eq!(p.min_exponent(), Some(-2));
        assert_eq!(p.max_exponent(), Some(-2));
    }

    #[test]
    fn test_addition() {
        let p1 = LaurentPolynomial::<Integer>::monomial(Integer::from(1), 2);
        let p2 = LaurentPolynomial::<Integer>::monomial(Integer::from(1), -1);

        let sum = p1 + p2;
        assert_eq!(sum.coeff(2), Integer::from(1));
        assert_eq!(sum.coeff(-1), Integer::from(1));
    }

    #[test]
    fn test_multiplication() {
        // (q^2) * (q^{-1}) = q
        let p1 = LaurentPolynomial::<Integer>::monomial(Integer::from(1), 2);
        let p2 = LaurentPolynomial::<Integer>::monomial(Integer::from(1), -1);

        let prod = p1 * p2;
        assert_eq!(prod.coeff(1), Integer::from(1));
        assert_eq!(prod.coeff(0), Integer::zero());
    }

    #[test]
    fn test_zero() {
        let p = LaurentPolynomial::<Integer>::zero();
        assert!(p.is_zero());
    }

    #[test]
    fn test_one() {
        let p = LaurentPolynomial::<Integer>::one();
        assert!(p.is_one());
        assert_eq!(p.coeff(0), Integer::one());
    }
}
