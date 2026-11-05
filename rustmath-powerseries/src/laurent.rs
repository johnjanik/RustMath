//! Laurent series `R((x))` — integral exponents, possibly negative.
//!
//! MAGMA source: Handbook Chapter 49 "Power, Laurent and Puiseux Series",
//! category `RngSerLaur`/`RngSerLaurElt`.  Covers §49.1.1 (integral valuation),
//! §49.4.5 (AbsolutePrecision / RelativePrecision / ChangePrecision), §49.4.6
//! (Valuation / Coefficient / LeadingCoefficient / Degree), §49.3.1
//! (ShiftValuation via multiplication by a power of the uniformizer) and, when
//! the coefficient ring is a field, the fact that `R((x))` is the **field of
//! fractions** of `R[[x]]` (a field — every non-zero element is invertible).
//!
//! Precision model: like the existing `PowerSeries`, elements are *truncated*
//! (always finite precision).  An element stores its lowest known exponent
//! `val` and a dense coefficient window `coeffs` (so the coefficient of
//! `x^{val+i}` is `coeffs[i]`); everything below `val` is a genuine zero and the
//! absolute precision — the first *unknown* exponent — is `val + coeffs.len()`.
//! This is the `Precision::Fixed`-style behaviour; the internal
//! [`crate::precision::Precision`] enum records `Free`/`Fixed` at the ring level.

use crate::precision::{Precision, DEFAULT_PRECISION};
use crate::series::PowerSeries;
use rustmath_core::valuation::DiscreteValuation;
use rustmath_core::{CommutativeRing, Field, IntegralDomain, MathError, Result, Ring};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A truncated Laurent series `Σ_{i≥0} coeffs[i] x^{val+i} + O(x^{val+len})`.
#[derive(Clone, Debug)]
pub struct LaurentSeries<R: Ring> {
    /// Exponent of `coeffs[0]` (may be negative).
    val: i64,
    /// Dense coefficients starting at exponent `val`.
    coeffs: Vec<R>,
}

impl<R: Ring> LaurentSeries<R> {
    /// Build from a low exponent `val` and a dense coefficient window.
    pub fn new(val: i64, coeffs: Vec<R>) -> Self {
        LaurentSeries { val, coeffs }
    }

    /// Build from exact coefficient data under a [`Precision`] regime
    /// (MAGMA Chapter 49.1.5).  The input window is treated as **exact**
    /// (Laurent-polynomial) data:
    ///
    /// * `Precision::Fixed(n)` — *capped* semantics: the window carries
    ///   exactly `n` terms (relative precision `n`); coefficients beyond are
    ///   discarded, shorter inputs are padded with genuine zeros.
    /// * `Precision::Free(d)` — *exact-polynomial* semantics: every given
    ///   coefficient is kept, shorter inputs are padded with genuine zeros up
    ///   to the ring default `d` terms.
    pub fn with_precision(val: i64, mut coeffs: Vec<R>, prec: Precision) -> Self {
        match prec {
            Precision::Fixed(n) => {
                coeffs.truncate(n);
                coeffs.resize_with(n, R::zero);
            }
            Precision::Free(d) => {
                if coeffs.len() < d {
                    coeffs.resize_with(d, R::zero);
                }
            }
        }
        LaurentSeries { val, coeffs }
    }

    /// The zero series to relative precision `prec` (`0 + O(x^prec)`).
    pub fn zero(prec: usize) -> Self {
        LaurentSeries {
            val: 0,
            coeffs: vec![R::zero(); prec],
        }
    }

    /// The constant `1` to relative precision `prec`.
    pub fn one(prec: usize) -> Self {
        let mut coeffs = vec![R::zero(); prec.max(1)];
        coeffs[0] = R::one();
        LaurentSeries { val: 0, coeffs }
    }

    /// The uniformizer `x` (valuation 1) to the given relative precision.
    pub fn gen(prec: usize) -> Self {
        let mut coeffs = vec![R::zero(); prec.max(1)];
        coeffs[0] = R::one();
        LaurentSeries { val: 1, coeffs }
    }

    /// The monomial `coeff · x^exp` to relative precision `prec`.
    pub fn monomial(exp: i64, coeff: R, prec: usize) -> Self {
        let mut coeffs = vec![R::zero(); prec.max(1)];
        coeffs[0] = coeff;
        LaurentSeries { val: exp, coeffs }
    }

    /// Embed a power series as a Laurent series (valuation `≥ 0`).
    pub fn from_power_series(f: &PowerSeries<R>) -> Self {
        let coeffs = (0..f.precision()).map(|i| f.coeff(i).clone()).collect();
        LaurentSeries { val: 0, coeffs }
    }

    /// The absolute precision: the first exponent whose coefficient is unknown
    /// (`f ∈ O(x^p)`), i.e. `val + relative_precision`.
    pub fn absolute_precision(&self) -> i64 {
        self.val + self.coeffs.len() as i64
    }

    /// The relative precision: number of stored coefficients.
    pub fn relative_precision(&self) -> usize {
        self.coeffs.len()
    }

    /// The lowest stored exponent (the exponent of `coeffs[0]`).  Unlike
    /// [`Self::valuation`] this does not skip leading zero coefficients; it is
    /// the left edge of the stored coefficient window.
    pub fn lowest_exponent(&self) -> i64 {
        self.val
    }

    /// The valuation: the smallest exponent whose coefficient is not known to be
    /// zero.  For an all-zero (to precision) series this is the absolute
    /// precision (a lower bound); [`DiscreteValuation::INFINITY`] is returned by
    /// the [`LaurentValuation`] adaptor for that case.
    pub fn valuation(&self) -> i64 {
        for (i, c) in self.coeffs.iter().enumerate() {
            if !c.is_zero() {
                return self.val + i as i64;
            }
        }
        self.absolute_precision()
    }

    /// The degree: the exponent of the last known non-zero term, or `None` if
    /// none is known.
    pub fn degree(&self) -> Option<i64> {
        for i in (0..self.coeffs.len()).rev() {
            if !self.coeffs[i].is_zero() {
                return Some(self.val + i as i64);
            }
        }
        None
    }

    /// The coefficient of `x^exp` (genuine zero below `val`, unknown-as-zero at
    /// or beyond the absolute precision).
    pub fn coefficient(&self, exp: i64) -> R {
        if exp < self.val {
            return R::zero();
        }
        let idx = (exp - self.val) as usize;
        self.coeffs.get(idx).cloned().unwrap_or_else(R::zero)
    }

    /// The leading (first non-zero) coefficient, or `None` if none is known.
    pub fn leading_coefficient(&self) -> Option<R> {
        self.coeffs.iter().find(|c| !c.is_zero()).cloned()
    }

    /// `ShiftValuation`: multiply by `x^n` (shifts every exponent by `n`).
    pub fn shift_valuation(&self, n: i64) -> Self {
        LaurentSeries {
            val: self.val + n,
            coeffs: self.coeffs.clone(),
        }
    }

    /// `ChangePrecision(f, r)`: return `f` with absolute precision `r`.
    pub fn change_precision(&self, new_abs_prec: i64) -> Self {
        let new_len = (new_abs_prec - self.val).max(0) as usize;
        let mut coeffs = self.coeffs.clone();
        coeffs.resize_with(new_len, R::zero);
        LaurentSeries {
            val: self.val,
            coeffs,
        }
    }

    /// Whether every known coefficient is zero.
    pub fn is_weakly_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }

    fn combine_bounds(&self, other: &Self) -> (i64, i64) {
        let lo = self.val.min(other.val);
        let hi = self.absolute_precision().min(other.absolute_precision());
        (lo, hi)
    }
}

impl<R: Ring> PartialEq for LaurentSeries<R> {
    fn eq(&self, other: &Self) -> bool {
        let (lo, hi) = self.combine_bounds(other);
        (lo..hi).all(|e| self.coefficient(e) == other.coefficient(e))
    }
}

impl<R: Ring> Add for LaurentSeries<R> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let (lo, hi) = self.combine_bounds(&other);
        if hi <= lo {
            return LaurentSeries {
                val: lo,
                coeffs: vec![],
            };
        }
        let coeffs = (lo..hi)
            .map(|e| self.coefficient(e) + other.coefficient(e))
            .collect();
        LaurentSeries { val: lo, coeffs }
    }
}

impl<R: Ring> Neg for LaurentSeries<R> {
    type Output = Self;
    fn neg(self) -> Self {
        LaurentSeries {
            val: self.val,
            coeffs: self.coeffs.into_iter().map(|c| -c).collect(),
        }
    }
}

impl<R: Ring> Sub for LaurentSeries<R> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<R: Ring> Mul for LaurentSeries<R> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        // Absolute precision of a product: min(p_a + v_b, p_b + v_a).
        let abs = (self.absolute_precision() + other.val)
            .min(other.absolute_precision() + self.val);
        let base_val = self.val + other.val;
        let new_len = (abs - base_val).max(0) as usize;
        if new_len == 0 || self.coeffs.is_empty() || other.coeffs.is_empty() {
            return LaurentSeries {
                val: abs,
                coeffs: vec![],
            };
        }
        let mut coeffs = vec![R::zero(); new_len];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                let k = i + j;
                if k < new_len {
                    coeffs[k] = coeffs[k].clone() + a.clone() * b.clone();
                }
            }
        }
        LaurentSeries {
            val: base_val,
            coeffs,
        }
    }
}

impl<R: Ring> Ring for LaurentSeries<R> {
    fn zero() -> Self {
        LaurentSeries::zero(DEFAULT_PRECISION)
    }
    fn one() -> Self {
        LaurentSeries::one(DEFAULT_PRECISION)
    }
    fn is_zero(&self) -> bool {
        self.is_weakly_zero()
    }
    fn is_one(&self) -> bool {
        self.coefficient(0).is_one()
            && (self.val..self.absolute_precision()).all(|e| e == 0 || self.coefficient(e).is_zero())
    }
}

impl<R: CommutativeRing> CommutativeRing for LaurentSeries<R> {}
impl<R: IntegralDomain> IntegralDomain for LaurentSeries<R> {}

impl<R: Field> LaurentSeries<R> {
    /// Multiplicative inverse.  `R((x))` is a field when `R` is: factor out the
    /// valuation `x^v`, invert the unit part `1 + …` as a power series (Newton),
    /// and shift back to valuation `-v`.
    pub fn try_inverse(&self) -> Result<Self> {
        let v = match self.coeffs.iter().position(|c| !c.is_zero()) {
            Some(i) => i,
            None => return Err(MathError::DivisionByZero),
        };
        let unit_coeffs: Vec<R> = self.coeffs[v..].to_vec();
        let prec = unit_coeffs.len();
        let unit = PowerSeries::new(unit_coeffs, prec);
        let inv_unit = unit.inverse()?;
        let inv_coeffs: Vec<R> = (0..inv_unit.precision())
            .map(|i| inv_unit.coeff(i).clone())
            .collect();
        Ok(LaurentSeries {
            val: -(self.val + v as i64),
            coeffs: inv_coeffs,
        })
    }
}

impl<R: Field> Div for LaurentSeries<R> {
    type Output = Self;
    // Division in a field of fractions is multiplication by the inverse.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: Self) -> Self {
        let inv = other
            .try_inverse()
            .expect("division by a zero Laurent series");
        self * inv
    }
}

impl<R: Field> Field for LaurentSeries<R> {
    fn inverse(&self) -> Result<Self> {
        self.try_inverse()
    }
}

/// A [`DiscreteValuation`] adaptor for `R((x))`: `v(f)` is the exponent of the
/// first non-zero term, with `v(0) = +∞` and uniformizer `x`.
pub struct LaurentValuation<R: Ring> {
    prec: usize,
    _marker: PhantomData<R>,
}

impl<R: Ring> LaurentValuation<R> {
    /// A valuation whose uniformizer carries relative precision `prec`.
    pub fn new(prec: usize) -> Self {
        LaurentValuation {
            prec,
            _marker: PhantomData,
        }
    }
}

impl<R: Ring> Default for LaurentValuation<R> {
    fn default() -> Self {
        LaurentValuation::new(DEFAULT_PRECISION)
    }
}

impl<R: Ring> DiscreteValuation<LaurentSeries<R>> for LaurentValuation<R> {
    fn valuation(&self, x: &LaurentSeries<R>) -> i64 {
        if x.is_weakly_zero() {
            Self::INFINITY
        } else {
            x.valuation()
        }
    }

    fn uniformizer(&self) -> LaurentSeries<R> {
        LaurentSeries::gen(self.prec)
    }
}

impl<R: Ring> fmt::Display for LaurentSeries<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let e = self.val + i as i64;
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            if e == 0 {
                write!(f, "{c}")?;
            } else if e == 1 {
                if c.is_one() {
                    write!(f, "x")?;
                } else {
                    write!(f, "{c}*x")?;
                }
            } else if c.is_one() {
                write!(f, "x^{e}")?;
            } else {
                write!(f, "{c}*x^{e}")?;
            }
        }
        if first {
            write!(f, "0")?;
        }
        write!(f, " + O(x^{})", self.absolute_precision())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn q(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    #[test]
    fn valuation_and_precision() {
        // 3x^{-2} + x^{-1} + O(x^3)
        let f = LaurentSeries::new(-2, vec![q(3), q(1), q(0), q(0), q(0)]);
        assert_eq!(f.valuation(), -2);
        assert_eq!(f.absolute_precision(), 3);
        assert_eq!(f.relative_precision(), 5);
        assert_eq!(f.coefficient(-2), q(3));
        assert_eq!(f.coefficient(-1), q(1));
        assert_eq!(f.coefficient(0), q(0));
        assert_eq!(f.coefficient(-5), q(0));
        assert_eq!(f.degree(), Some(-1));
        assert_eq!(f.leading_coefficient(), Some(q(3)));
    }

    #[test]
    fn shift_and_change_precision() {
        let f = LaurentSeries::new(-1, vec![q(1), q(2), q(3)]);
        let g = f.shift_valuation(2); // multiply by x^2
        assert_eq!(g.valuation(), 1);
        assert_eq!(g.coefficient(1), q(1));
        let h = f.change_precision(1);
        assert_eq!(h.absolute_precision(), 1);
        assert_eq!(h.coefficient(-1), q(1));
        assert_eq!(h.coefficient(0), q(2));
    }

    #[test]
    fn with_precision_capped_vs_exact() {
        use crate::precision::Precision;
        let w = vec![q(1), q(2), q(3)];
        // Fixed(2): window capped at 2 terms => absolute precision val + 2.
        let capped = LaurentSeries::with_precision(-1, w.clone(), Precision::Fixed(2));
        assert_eq!(capped.relative_precision(), 2);
        assert_eq!(capped.absolute_precision(), 1);
        assert_eq!(capped.coefficient(-1), q(1));
        assert_eq!(capped.coefficient(0), q(2));
        // Free(5): exact window kept, padded with genuine zeros to 5 terms.
        let free = LaurentSeries::with_precision(-1, w, Precision::Free(5));
        assert_eq!(free.relative_precision(), 5);
        assert_eq!(free.coefficient(1), q(3));
        assert_eq!(free.coefficient(3), q(0));
    }

    #[test]
    fn arithmetic_propagates_min_precision() {
        // Sum: absolute precision is the min of the operands'.
        let a = LaurentSeries::new(-1, vec![q(1), q(1), q(1)]); // O(x^2)
        let b = LaurentSeries::new(0, vec![q(1), q(1), q(1), q(1), q(1)]); // O(x^5)
        assert_eq!((a.clone() + b.clone()).absolute_precision(), 2);
        assert_eq!((b.clone() - a.clone()).absolute_precision(), 2);
        // Product: min(p_a + v_b, p_b + v_a) with v the window base.
        let p = a.clone() * b.clone();
        assert_eq!(p.absolute_precision(), (2 + 0).min(5 + (-1)));
    }

    #[test]
    fn arithmetic() {
        // (x^{-1} + 1) + (2 + x) = x^{-1} + 3 + x
        let a = LaurentSeries::new(-1, vec![q(1), q(1), q(0)]);
        let b = LaurentSeries::new(0, vec![q(2), q(1), q(0)]);
        let s = a.clone() + b.clone();
        assert_eq!(s.coefficient(-1), q(1));
        assert_eq!(s.coefficient(0), q(3));
        assert_eq!(s.coefficient(1), q(1));

        // (x^{-1})(x^2) = x
        let p = LaurentSeries::new(-1, vec![q(1), q(0), q(0)])
            * LaurentSeries::new(2, vec![q(1), q(0), q(0)]);
        assert_eq!(p.valuation(), 1);
        assert_eq!(p.coefficient(1), q(1));
    }

    #[test]
    fn inverse_is_field() {
        // 1/(1 - x) = 1 + x + x^2 + ...  represented as a Laurent series
        let one_minus_x = LaurentSeries::new(0, vec![q(1), q(-1), q(0), q(0), q(0), q(0)]);
        let inv = one_minus_x.try_inverse().unwrap();
        for i in 0..5 {
            assert_eq!(inv.coefficient(i), q(1), "coeff {i}");
        }
        // 1/x has valuation -1
        let x = LaurentSeries::<Rational>::gen(6);
        let inv_x = x.try_inverse().unwrap();
        assert_eq!(inv_x.valuation(), -1);
        assert_eq!(inv_x.coefficient(-1), q(1));
        // x * (1/x) = 1
        let prod = LaurentSeries::<Rational>::gen(6) * inv_x;
        assert!(prod.is_one());
    }

    #[test]
    fn discrete_valuation_adaptor() {
        let v = LaurentValuation::<Rational>::default();
        let f = LaurentSeries::new(-2, vec![q(5), q(0), q(1)]);
        assert_eq!(v.valuation(&f), -2);
        let z = LaurentSeries::<Rational>::zero(4);
        assert_eq!(
            v.valuation(&z),
            <LaurentValuation<Rational> as DiscreteValuation<LaurentSeries<Rational>>>::INFINITY
        );
        assert_eq!(v.uniformizer().valuation(), 1);
        assert!(v.is_unit(&LaurentSeries::new(0, vec![q(7), q(1)])));
    }

    #[test]
    fn ring_tower() {
        fn needs_ring<T: Ring>(a: T, b: T) -> T {
            a * b
        }
        let a = LaurentSeries::new(-1, vec![q(1), q(1), q(0), q(0)]);
        let b = LaurentSeries::new(1, vec![q(1), q(0), q(0), q(0)]);
        let p = needs_ring(a, b); // (x^{-1}+1)(x) = 1 + x
        assert_eq!(p.coefficient(0), q(1));
        assert_eq!(p.coefficient(1), q(1));
    }
}
