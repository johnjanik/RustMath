//! Lazy power series (Chapter 50).
//!
//! MAGMA source: Handbook Chapter 50 "Lazy Power Series Rings" — series whose
//! coefficients are only computed on demand and then memoised, allowing exact
//! (unbounded-precision) representatives.  Covers §50.4.1 (creation from a
//! coefficient sequence or a map), §50.4.2 (arithmetic), §50.4.3 (finding
//! coefficients / `Valuation`), §50.4.4/§50.4.5 (`IsWeaklyZero`/`IsWeaklyEqual`).
//!
//! Design: a [`LazyPowerSeries`] is a concrete generic (never a `dyn` series
//! object) wrapping `Rc<LazyInner<R>>`, where `LazyInner` holds a
//! `RefCell<Vec<Option<R>>>` memoisation cache and a coefficient *recipe*
//! `Rc<dyn Fn(&LazyPowerSeries<R>, usize) -> R>`.  Only the recipe closure is a
//! trait object; the algebraic type stays a concrete generic so it can implement
//! the `rustmath-core` `Ring` tower.  The recipe receives the series itself so
//! recursively-defined series (e.g. a lazy multiplicative inverse) can read their
//! own lower-index coefficients — this is well founded whenever coefficient `n`
//! depends only on coefficients `< n`.  Cloning shares the cache (an `Rc` bump),
//! so memoised work is never repeated.  Zero `unsafe`.
//!
//! Equality/zero-ness of an unbounded series is only semi-decidable, so the
//! `PartialEq`/`Ring::is_zero` implementations are *weak*: they compare a bounded
//! number of leading coefficients (`WEAK_TERMS`).  Use [`LazyPowerSeries::is_weakly_zero`]
//! /[`LazyPowerSeries::is_weakly_equal`] with an explicit bound for control.

use crate::precision::DEFAULT_PRECISION;
use crate::series::PowerSeries;
use rustmath_core::{CommutativeRing, Field, IntegralDomain, MathError, NumericConversion, Result, Ring};
use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

/// Number of leading coefficients compared by the *weak* `PartialEq` /
/// `is_zero` used to satisfy the `Ring` contract.
pub const WEAK_TERMS: usize = DEFAULT_PRECISION;

type Recipe<R> = Rc<dyn Fn(&LazyPowerSeries<R>, usize) -> R>;

struct LazyInner<R: Ring> {
    cache: RefCell<Vec<Option<R>>>,
    recipe: Recipe<R>,
}

/// A lazily evaluated, memoised formal power series `Σ a_n x^n`.
#[derive(Clone)]
pub struct LazyPowerSeries<R: Ring> {
    inner: Rc<LazyInner<R>>,
}

impl<R: Ring + 'static> LazyPowerSeries<R> {
    fn from_recipe<F>(recipe: F) -> Self
    where
        F: Fn(&LazyPowerSeries<R>, usize) -> R + 'static,
    {
        LazyPowerSeries {
            inner: Rc::new(LazyInner {
                cache: RefCell::new(Vec::new()),
                recipe: Rc::new(recipe),
            }),
        }
    }

    /// Create a lazy series from an independent coefficient map `n ↦ a_n`
    /// (Chapter 50.4.1.1, "creation from maps").
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(usize) -> R + 'static,
    {
        Self::from_recipe(move |_self, n| f(n))
    }

    /// Create a recursively-defined lazy series: `recipe(self, n)` may read
    /// `self.coeff(k)` for `k < n`.
    pub fn from_recursive<F>(recipe: F) -> Self
    where
        F: Fn(&LazyPowerSeries<R>, usize) -> R + 'static,
    {
        Self::from_recipe(recipe)
    }

    /// A finite lazy series with the given coefficients (`R ! S`, Chapter
    /// 50.4.1).
    pub fn from_coeffs(coeffs: Vec<R>) -> Self {
        Self::from_fn(move |n| coeffs.get(n).cloned().unwrap_or_else(R::zero))
    }

    /// The constant series `c`.
    pub fn constant(c: R) -> Self {
        Self::from_fn(move |n| if n == 0 { c.clone() } else { R::zero() })
    }

    /// The series `x` (the generator).
    pub fn gen() -> Self {
        Self::from_fn(|n| if n == 1 { R::one() } else { R::zero() })
    }

    /// The `n`-th coefficient, computing and memoising on demand.
    pub fn coeff(&self, n: usize) -> R {
        if let Some(Some(c)) = self.inner.cache.borrow().get(n) {
            return c.clone();
        }
        // Do not hold the cache borrow across the (possibly recursive) recipe.
        let c = (self.inner.recipe)(self, n);
        let mut cache = self.inner.cache.borrow_mut();
        if cache.len() <= n {
            cache.resize(n + 1, None);
        }
        cache[n] = Some(c.clone());
        c
    }

    /// The coefficients `[a_0, .., a_{n-1}]`.
    pub fn coefficients(&self, n: usize) -> Vec<R> {
        (0..n).map(|i| self.coeff(i)).collect()
    }

    /// Truncate to a fixed-precision [`PowerSeries`] with `prec` terms.
    pub fn approximate(&self, prec: usize) -> PowerSeries<R> {
        PowerSeries::new(self.coefficients(prec), prec)
    }

    /// Whether the first `bound` coefficients are all zero (`IsWeaklyZero`).
    pub fn is_weakly_zero(&self, bound: usize) -> bool {
        (0..bound).all(|i| self.coeff(i).is_zero())
    }

    /// Whether two series agree on the first `bound` coefficients
    /// (`IsWeaklyEqual`).
    pub fn is_weakly_equal(&self, other: &Self, bound: usize) -> bool {
        (0..bound).all(|i| self.coeff(i) == other.coeff(i))
    }

    /// The valuation (index of the first non-zero coefficient), searched up to
    /// `bound`; `None` if all of the first `bound` coefficients are zero.
    pub fn valuation(&self, bound: usize) -> Option<usize> {
        (0..bound).find(|&i| !self.coeff(i).is_zero())
    }

    /// Lazy negation.
    pub fn neg_ref(&self) -> Self {
        let a = self.clone();
        Self::from_fn(move |n| -a.coeff(n))
    }

    /// Lazy addition.
    pub fn add_ref(&self, other: &Self) -> Self {
        let a = self.clone();
        let b = other.clone();
        Self::from_fn(move |n| a.coeff(n) + b.coeff(n))
    }

    /// Lazy subtraction.
    pub fn sub_ref(&self, other: &Self) -> Self {
        let a = self.clone();
        let b = other.clone();
        Self::from_fn(move |n| a.coeff(n) - b.coeff(n))
    }

    /// Lazy Cauchy product.
    pub fn mul_ref(&self, other: &Self) -> Self {
        let a = self.clone();
        let b = other.clone();
        Self::from_fn(move |n| {
            let mut acc = R::zero();
            for k in 0..=n {
                acc = acc + a.coeff(k) * b.coeff(n - k);
            }
            acc
        })
    }

    /// Lazy scalar multiplication by a coefficient.
    pub fn scalar_mul(&self, c: &R) -> Self {
        let a = self.clone();
        let c = c.clone();
        Self::from_fn(move |n| c.clone() * a.coeff(n))
    }

    /// Lazy formal derivative.
    pub fn derivative(&self) -> Self
    where
        R: NumericConversion,
    {
        let a = self.clone();
        Self::from_fn(move |n| R::from_i64((n + 1) as i64) * a.coeff(n + 1))
    }
}

impl<R: Field + NumericConversion + 'static> LazyPowerSeries<R> {
    /// Lazy formal integral with zero constant term (`Σ a_n x^n ↦ Σ a_n/(n+1) x^{n+1}`).
    pub fn integral(&self) -> Self {
        let a = self.clone();
        Self::from_recursive(move |_s, n| {
            if n == 0 {
                R::zero()
            } else {
                a.coeff(n - 1) * R::from_i64(n as i64).inverse().expect("char 0")
            }
        })
    }
}

impl<R: Field + 'static> LazyPowerSeries<R> {
    /// Lazy multiplicative inverse (requires a unit constant term).  Uses the
    /// self-referential recurrence `g_0 = a_0^{-1}`,
    /// `g_n = -a_0^{-1} Σ_{k=1}^n a_k g_{n-k}`.
    pub fn try_inverse(&self) -> Result<Self> {
        let a0 = self.coeff(0);
        if a0.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let a0_inv = a0.inverse()?;
        let f = self.clone();
        Ok(Self::from_recursive(move |g, n| {
            if n == 0 {
                return a0_inv.clone();
            }
            let mut acc = R::zero();
            for k in 1..=n {
                acc = acc + f.coeff(k) * g.coeff(n - k);
            }
            -(a0_inv.clone() * acc)
        }))
    }
}

impl<R: Ring + 'static> Add for LazyPowerSeries<R> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.add_ref(&other)
    }
}

impl<R: Ring + 'static> Sub for LazyPowerSeries<R> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.sub_ref(&other)
    }
}

impl<R: Ring + 'static> Mul for LazyPowerSeries<R> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.mul_ref(&other)
    }
}

impl<R: Ring + 'static> Neg for LazyPowerSeries<R> {
    type Output = Self;
    fn neg(self) -> Self {
        self.neg_ref()
    }
}

impl<R: Ring + 'static> PartialEq for LazyPowerSeries<R> {
    /// Weak equality: compares the first [`WEAK_TERMS`] coefficients.
    fn eq(&self, other: &Self) -> bool {
        self.is_weakly_equal(other, WEAK_TERMS)
    }
}

impl<R: Ring + 'static> Ring for LazyPowerSeries<R> {
    fn zero() -> Self {
        Self::from_fn(|_| R::zero())
    }
    fn one() -> Self {
        Self::constant(R::one())
    }
    /// Weak zero test: checks the first [`WEAK_TERMS`] coefficients.
    fn is_zero(&self) -> bool {
        self.is_weakly_zero(WEAK_TERMS)
    }
    fn is_one(&self) -> bool {
        self.coeff(0).is_one() && (1..WEAK_TERMS).all(|i| self.coeff(i).is_zero())
    }
}

impl<R: CommutativeRing + 'static> CommutativeRing for LazyPowerSeries<R> {}
impl<R: IntegralDomain + 'static> IntegralDomain for LazyPowerSeries<R> {}

impl<R: Ring + 'static> fmt::Debug for LazyPowerSeries<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LazyPowerSeries({:?} + ...)", self.coefficients(4))
    }
}

impl<R: Ring + 'static> fmt::Display for LazyPowerSeries<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print up to WEAK_TERMS leading terms.
        write!(f, "{}", self.approximate(WEAK_TERMS))
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
    fn geometric_series_from_fn() {
        // 1/(1-x) = Σ x^n via a coefficient map.
        let g = LazyPowerSeries::<Rational>::from_fn(|_| q(1));
        assert_eq!(g.coeff(0), q(1));
        assert_eq!(g.coeff(50), q(1));
        // memoised: cache holds index 50 now
        assert!(g.coeff(50).is_one());
    }

    #[test]
    fn lazy_inverse_matches_geometric() {
        // (1 - x)^{-1} = 1 + x + x^2 + ...
        let one_minus_x = LazyPowerSeries::from_coeffs(vec![q(1), q(-1)]);
        let inv = one_minus_x.try_inverse().unwrap();
        for i in 0..20 {
            assert_eq!(inv.coeff(i), q(1), "coeff {i}");
        }
        // f * f^{-1} = 1 (weakly)
        let prod = one_minus_x.mul_ref(&inv);
        assert_eq!(prod.coeff(0), q(1));
        for i in 1..20 {
            assert_eq!(prod.coeff(i), q(0), "coeff {i}");
        }
    }

    #[test]
    fn fibonacci_generating_function() {
        // F(x) = x/(1 - x - x^2) has Fibonacci coefficients 0,1,1,2,3,5,8,...
        let denom = LazyPowerSeries::from_coeffs(vec![q(1), q(-1), q(-1)]);
        let num = LazyPowerSeries::from_coeffs(vec![q(0), q(1)]);
        let fib = num.mul_ref(&denom.try_inverse().unwrap());
        let expected = [0i64, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(fib.coeff(i), q(e), "fib {i}");
        }
    }

    #[test]
    fn ring_and_weak_equality() {
        let a = LazyPowerSeries::<Rational>::from_coeffs(vec![q(1), q(2), q(3)]);
        let b = <LazyPowerSeries<Rational> as Ring>::one();
        let prod = a.clone() * b;
        assert!(a.is_weakly_equal(&prod, WEAK_TERMS));
        let zero = <LazyPowerSeries<Rational> as Ring>::zero();
        assert!(zero.is_zero());
        assert!(!a.is_zero());
    }

    #[test]
    fn derivative_and_integral_roundtrip() {
        // d/dx of Σ x^n = Σ (n+1) x^n ; integral brings it back (no constant).
        let g = LazyPowerSeries::<Rational>::from_fn(|_| q(1));
        let d = g.derivative();
        assert_eq!(d.coeff(0), q(1));
        assert_eq!(d.coeff(1), q(2));
        assert_eq!(d.coeff(2), q(3));
        let back = d.integral();
        // integral of d has same coeffs as g for n>=1, constant term 0
        assert_eq!(back.coeff(0), q(0));
        for i in 1..10 {
            assert_eq!(back.coeff(i), q(1), "coeff {i}");
        }
    }

    #[test]
    fn approximate_bridges_to_power_series() {
        let g = LazyPowerSeries::<Rational>::from_fn(|n| q(n as i64));
        let ps = g.approximate(5);
        assert_eq!(ps.precision(), 5);
        assert_eq!(ps.coeff(3), &q(3));
    }
}
