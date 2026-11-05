//! Algebraic real numbers (elements of AA)
//!
//! An algebraic real is a real number that is a root of a non-zero
//! polynomial with rational coefficients.

use crate::algebraic_number::AlgebraicNumber;
use crate::descriptor::{
    eval_interval, interval_sign, sign_of_rational, AlgebraicDescriptor, ANRoot, ANUnaryExpr,
    UnaryOp,
};
use rustmath_core::{Field, Ring, MathError, Result};
use rustmath_rationals::Rational;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_integers::Integer;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::cmp::Ordering;
use std::fmt;

/// An algebraic real number (element of AA)
///
/// Represents a real number that is a root of a polynomial with rational coefficients.
/// This type is more restrictive than AlgebraicNumber - it only represents real values.
#[derive(Debug, Clone)]
pub struct AlgebraicReal {
    /// Internal representation using AlgebraicNumber
    inner: AlgebraicNumber,
}

impl AlgebraicReal {
    /// Create an algebraic real from a rational
    pub fn from_rational(r: Rational) -> Self {
        Self {
            inner: AlgebraicNumber::from_rational(r),
        }
    }

    /// Create an algebraic real from an integer
    pub fn from_i64(n: i64) -> Self {
        Self::from_rational(Rational::from_i64(n))
    }

    /// Create the square root of a positive integer
    ///
    /// # Arguments
    /// * `n` - A positive integer
    ///
    /// # Returns
    /// * sqrt(n) as an algebraic real number
    ///
    /// # Examples
    /// ```
    /// use rustmath_algebraic::AlgebraicReal;
    ///
    /// let sqrt2 = AlgebraicReal::sqrt(2);
    /// let sqrt3 = AlgebraicReal::sqrt(3);
    /// ```
    pub fn sqrt(n: i64) -> Self {
        if n == 0 {
            return Self::from_i64(0);
        }

        if n < 0 {
            panic!("Cannot take square root of negative number in AlgebraicReal");
        }

        // Check if n is a perfect square
        let sqrt_n = (n as f64).sqrt();
        if sqrt_n.fract() == 0.0 {
            return Self::from_i64(sqrt_n as i64);
        }

        // Create polynomial x^2 - n
        let poly = UnivariatePolynomial::new(vec![Integer::from(-n), Integer::zero(), Integer::one()]);

        // Create isolating interval using Newton's method approximation
        let approx = sqrt_n;
        let epsilon = 0.1;
        let lower = Rational::from_f64(approx - epsilon).unwrap();
        let upper = Rational::from_f64(approx + epsilon).unwrap();

        let root = ANRoot::new(poly, Some((lower, upper)), None);

        Self {
            inner: AlgebraicNumber::new(AlgebraicDescriptor::Root(root)),
        }
    }

    /// Create the nth root of a number
    pub fn nth_root(n: i64, degree: u32) -> Self {
        if degree == 0 {
            panic!("Cannot take 0th root");
        }

        if degree == 1 {
            return Self::from_i64(n);
        }

        if degree == 2 {
            return Self::sqrt(n);
        }

        if n < 0 && degree % 2 == 0 {
            panic!("Cannot take even root of negative number in AlgebraicReal");
        }

        // For odd degree, we can handle negative n
        let abs_n = n.abs();
        let sign = if n < 0 { -1 } else { 1 };

        // Check if abs_n is a perfect power
        let root_approx = (abs_n as f64).powf(1.0 / degree as f64);
        let root_int = root_approx.round() as i64;
        if root_int.pow(degree) == abs_n {
            return Self::from_i64(sign * root_int);
        }

        // Create polynomial x^degree - n
        let mut coeffs = vec![Integer::from(-n)];
        for _ in 1..degree {
            coeffs.push(Integer::zero());
        }
        coeffs.push(Integer::one());

        let poly = UnivariatePolynomial::new(coeffs);

        // Create isolating interval
        let approx = (n as f64).powf(1.0 / degree as f64);
        let epsilon = 0.1;
        let lower = Rational::from_f64(approx - epsilon).unwrap();
        let upper = Rational::from_f64(approx + epsilon).unwrap();

        let root = ANRoot::new(poly, Some((lower, upper)), None);

        Self {
            inner: AlgebraicNumber::new(AlgebraicDescriptor::Root(root)),
        }
    }

    /// Check if this is a rational number
    pub fn is_rational(&self) -> bool {
        self.inner.is_rational()
    }

    /// Try to convert to a rational number
    pub fn to_rational(&self) -> Option<Rational> {
        self.inner.to_rational()
    }

    /// Get a floating-point approximation
    pub fn to_f64(&self, precision: usize) -> f64 {
        if let Some(r) = self.to_rational() {
            r.to_f64().unwrap_or(0.0)
        } else {
            // TODO: Implement proper evaluation
            0.0
        }
    }

    /// Simplify this algebraic real
    pub fn simplify(&self) -> Self {
        Self {
            inner: self.inner.simplify(),
        }
    }

    /// Create the zero element (0)
    pub fn zero() -> Self {
        Self::from_i64(0)
    }

    /// Create the one element (1)
    pub fn one() -> Self {
        Self::from_i64(1)
    }

    /// Check if this is exactly zero
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Check if this is exactly one
    pub fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    /// Compute the multiplicative inverse
    pub fn inverse(&self) -> Result<Self> {
        Ok(Self {
            inner: self.inner.inverse()?,
        })
    }

    /// Compute the absolute value
    pub fn abs(&self) -> Self {
        Self {
            inner: AlgebraicNumber::new(AlgebraicDescriptor::UnaryExpr(ANUnaryExpr::new(
                UnaryOp::Abs,
                self.inner.descriptor().clone(),
            ))),
        }
    }

    /// Convert to AlgebraicNumber
    pub fn to_algebraic_number(&self) -> AlgebraicNumber {
        self.inner.clone()
    }

    /// Initial number of bisection steps applied to each `Root` leaf when
    /// refining isolating intervals for sign determination.
    const SIGN_INITIAL_BISECTIONS: usize = 8;

    /// Number of refinement rounds `sign()` attempts, each widening the
    /// bisection depth 4x, before falling back to a best-effort decision.
    /// With these numbers the deepest round refines to `8 * 4^3 = 512` bits
    /// of precision per `Root` leaf - comfortably enough to separate any two
    /// distinct algebraic numbers built from realistic inputs (the crude
    /// starting interval already separates most cases; genuinely equal
    /// composite expressions never separate no matter how deep we go, so
    /// there is little value pushing this cap higher). Bisection cost grows
    /// super-linearly with the bit depth (bignum arithmetic on growing
    /// rationals), so keep this bounded to keep `sign()` fast.
    ///
    /// Measured on the worst-case exhaustion path (the `test_golden_ratio`
    /// case below, whose difference is exactly zero and so never separates,
    /// forcing every round to run to completion) via
    /// `eval_interval(desc, bisections)` directly: 8 bisections -> 1.4ms, 32
    /// -> 6.7ms, 128 -> 185ms, 512 (the current cap) -> 3.15s, 2048 (one
    /// more round) -> 70.7s. Raising `SIGN_REFINEMENT_ROUNDS` by even one is
    /// therefore *not* cheap - it would turn every `sign()`/`cmp()`/`eq()`
    /// call on a genuinely-zero or pathologically-close composite expression
    /// from ~3s into more than a minute - so the bound is intentionally left
    /// as-is rather than raised further.
    const SIGN_REFINEMENT_ROUNDS: usize = 4;

    /// Get the sign of this number (-1, 0, or 1)
    ///
    /// For values that are (or simplify to) an exact rational, this is exact
    /// and immediate. For genuinely irrational values, this refines the
    /// isolating interval(s) of the underlying expression tree via exact
    /// rational bisection (see `crate::descriptor::eval_interval`) until the
    /// sign is unambiguous.
    ///
    /// # Best-effort-zero caveat (NOT a certified decision procedure)
    ///
    /// Two *distinct* algebraic reals are always bounded away from zero, so
    /// in practice refinement separates them from zero after a handful of
    /// bisection steps. If refinement is exhausted (`SIGN_REFINEMENT_ROUNDS`
    /// rounds, i.e. `SIGN_INITIAL_BISECTIONS * 4^(SIGN_REFINEMENT_ROUNDS - 1)`
    /// = 512 bits of precision per `Root` leaf with the current constants)
    /// without the bound separating from zero, `sign()` gives up and reports
    /// `0` ("exactly zero") as a best-effort guess - **this is not a
    /// formally certified proof**, since this crate does not yet compute
    /// resultant-based minimal polynomials for composite expressions to
    /// decide equality symbolically. It is possible, in principle, to
    /// construct two distinct algebraic reals whose difference stays within
    /// 512 bits of zero for the entire isolating interval throughout all
    /// refinement rounds, in which case `sign()` (and therefore `cmp`/`eq`,
    /// which are defined in terms of it) will incorrectly report them as
    /// equal. In practice this only affects composite expressions that are
    /// provably equal via *exact* algebraic identities this crate cannot yet
    /// see through numerically alone (e.g. golden-ratio style identities), or
    /// pathologically close-but-distinct constructions; it never causes two
    /// clearly-different irrationals (like `sqrt(2)` and `sqrt(3)`) to compare
    /// equal, since those separate from zero almost immediately. Callers that
    /// need a *certified* zero/equality test (e.g. for soundness-critical
    /// dedup or set membership) should not rely on this crate's `Ord`/`Eq`/
    /// `sign()` alone.
    pub fn sign(&self) -> i32 {
        // Fast, exact path: this also covers algebraic expressions that
        // `simplify` is able to fold down to an exact rational (e.g.
        // sqrt(2) * sqrt(2) -> 2, via the quadratic-surd reduction rule in
        // `AlgebraicDescriptor::simplify`).
        if let Some(r) = self.to_rational() {
            return sign_of_rational(&r);
        }

        // Slow path for genuinely irrational values: refine the isolating
        // interval(s) of every `Root` leaf in the expression tree via
        // bisection, using exact `Rational` arithmetic, until the resulting
        // bound on the whole expression no longer straddles zero.
        let simplified = self.inner.simplify();
        let desc = simplified.descriptor();

        let mut bisections = Self::SIGN_INITIAL_BISECTIONS;
        for _ in 0..Self::SIGN_REFINEMENT_ROUNDS {
            if let Some((lo, hi)) = eval_interval(desc, bisections) {
                if let Some(s) = interval_sign(&lo, &hi) {
                    return s;
                }
            }
            bisections = bisections.saturating_mul(4);
        }

        // Refinement cap reached without separating from zero: best-effort
        // "exactly zero" decision (see doc comment above). Deliberately not
        // reached by clearly-unequal irrationals - only by values that are
        // (numerically, to thousands of bits) indistinguishable from zero.
        0
    }
}

impl PartialEq for AlgebraicReal {
    /// Exact equality, defined as `(self - other).sign() == 0`.
    ///
    /// # Best-effort-zero caveat
    ///
    /// This inherits the best-effort-zero fallback of [`AlgebraicReal::sign`]:
    /// for two *distinct* values whose difference cannot be separated from
    /// zero within `SIGN_REFINEMENT_ROUNDS` rounds of exact rational interval
    /// bisection (bottoming out at `SIGN_INITIAL_BISECTIONS *
    /// 4^(SIGN_REFINEMENT_ROUNDS - 1)` bits of precision per `Root` leaf -
    /// currently 512 bits), `eq` reports `true` even though the values are
    /// not actually equal. This is a soundness gap, not just an imprecision:
    /// it is *not* a certified-equality test. It only bites composite
    /// expressions whose difference is numerically indistinguishable from
    /// zero to hundreds of bits without actually being the zero polynomial
    /// relation this crate can detect symbolically; two clearly-different
    /// algebraic reals (e.g. `sqrt(2)` vs `sqrt(3)`) separate from zero
    /// almost immediately and are never affected. See `AlgebraicReal::sign`
    /// for the full rationale and why the bound is not raised further.
    fn eq(&self, other: &Self) -> bool {
        // Defined in terms of `cmp` so that `Eq`/`Ord`/`PartialOrd` stay
        // consistent (`a == b` iff `a.cmp(&b) == Ordering::Equal`), and so
        // equality benefits from the same sound interval refinement as
        // ordering instead of the old (always-`false`-for-irrationals)
        // structural comparison.
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for AlgebraicReal {}

impl PartialOrd for AlgebraicReal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AlgebraicReal {
    /// Exact ordering, defined as the sign of `self - other`.
    ///
    /// # Best-effort-zero caveat
    ///
    /// Like [`AlgebraicReal::sign`], this is a **best-effort** decision
    /// procedure, not a certified one. `Ordering::Equal` is returned either
    /// when `self` and `other` are provably equal (both simplify to the same
    /// exact rational, or the interval refinement of `self - other` collapses
    /// to a point), *or* when refinement of `self - other` is exhausted
    /// (after `AlgebraicReal::SIGN_REFINEMENT_ROUNDS` rounds, ~512 bits of
    /// precision) without separating from zero - in the latter case `self`
    /// and `other` may in fact be distinct but too close (or the current
    /// numeric refinement too weak, since this crate does not yet compute
    /// resultant-based minimal polynomials for composite expressions) to
    /// prove it. Callers relying on `Ord`/`Eq` for correctness-critical
    /// dedup, sorting invariants, or set membership on non-trivial composite
    /// expressions should be aware `Equal` is not a certified proof of
    /// equality.
    ///
    /// Implemented as `sign(self - other)` via exact interval refinement
    /// (see [`AlgebraicReal::sign`]). Note this recomputes `self - other`'s
    /// isolating intervals from scratch rather than reusing any
    /// previously-refined intervals for `self`/`other` individually, which
    /// is what lets the two sides' shared `Root` leaves (e.g. comparing an
    /// expression against itself, or two expressions built from the same
    /// radical) refine in lock step and actually separate/cancel correctly.
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.clone() - other.clone()).sign() {
            0 => Ordering::Equal,
            s if s > 0 => Ordering::Greater,
            _ => Ordering::Less,
        }
    }
}

impl fmt::Display for AlgebraicReal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

// Arithmetic operations

impl Add for AlgebraicReal {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            inner: self.inner + other.inner,
        }
    }
}

impl Sub for AlgebraicReal {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            inner: self.inner - other.inner,
        }
    }
}

impl Mul for AlgebraicReal {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            inner: self.inner * other.inner,
        }
    }
}

impl Div for AlgebraicReal {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            inner: self.inner / other.inner,
        }
    }
}

impl Neg for AlgebraicReal {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            inner: -self.inner,
        }
    }
}

// Ring trait implementation

impl Ring for AlgebraicReal {
    fn zero() -> Self {
        AlgebraicReal::zero()
    }

    fn one() -> Self {
        AlgebraicReal::one()
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn is_one(&self) -> bool {
        self.is_one()
    }
}

impl rustmath_core::CommutativeRing for AlgebraicReal {}
impl rustmath_core::IntegralDomain for AlgebraicReal {}

impl Field for AlgebraicReal {
    fn inverse(&self) -> Result<Self> {
        self.inverse()
    }

    fn divide(&self, other: &Self) -> Result<Self> {
        if other.is_zero() {
            Err(MathError::DivisionByZero)
        } else {
            Ok(self.clone() / other.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_algebraic_reals() {
        let a = AlgebraicReal::from_rational(Rational::new(3, 4).unwrap());
        let b = AlgebraicReal::from_rational(Rational::new(1, 2).unwrap());

        assert!(a > b);
        assert!(b < a);
        assert_eq!(a.clone(), a.clone());
    }

    #[test]
    fn test_sqrt_perfect_square() {
        let sqrt4 = AlgebraicReal::sqrt(4);
        assert!(sqrt4.is_rational());
        assert_eq!(sqrt4.to_rational().unwrap(), Rational::new(2, 1).unwrap());
    }

    #[test]
    fn test_sqrt_irrational() {
        let sqrt2 = AlgebraicReal::sqrt(2);
        let sqrt2_squared = sqrt2.clone() * sqrt2.clone();

        assert!(sqrt2_squared.is_rational());
        assert_eq!(
            sqrt2_squared.to_rational().unwrap(),
            Rational::new(2, 1).unwrap()
        );
    }

    #[test]
    fn test_golden_ratio() {
        // φ = (1 + sqrt(5)) / 2
        let one = AlgebraicReal::from_i64(1);
        let two = AlgebraicReal::from_i64(2);
        let sqrt5 = AlgebraicReal::sqrt(5);

        let phi = (one + sqrt5) / two;

        // φ² = φ + 1
        let phi_squared = phi.clone() * phi.clone();
        let phi_plus_one = phi.clone() + AlgebraicReal::from_i64(1);

        // Both should simplify to the same value
        // (exact comparison will work once we implement it properly)
        assert_eq!(phi_squared, phi_plus_one);
    }

    #[test]
    fn test_nth_root() {
        let cube_root_8 = AlgebraicReal::nth_root(8, 3);
        assert!(cube_root_8.is_rational());
        assert_eq!(cube_root_8.to_rational().unwrap(), Rational::new(2, 1).unwrap());
    }

    #[test]
    fn test_ordering() {
        let a = AlgebraicReal::from_i64(3);
        let b = AlgebraicReal::from_i64(5);
        let c = AlgebraicReal::from_i64(3);

        assert!(a < b);
        assert!(b > a);
        assert!(a == c);
    }

    // --- Irrational sign / ordering (interval refinement) -----------------
    //
    // These exercise the interval-refinement based `sign()`/`cmp()` for
    // genuinely irrational algebraic reals, where the old implementation
    // silently returned 0 / Ordering::Equal for everything.

    #[test]
    fn test_sqrt2_sign_is_positive() {
        let sqrt2 = AlgebraicReal::sqrt(2);
        assert_eq!(sqrt2.sign(), 1);
    }

    #[test]
    fn test_neg_sqrt2_sign_is_negative() {
        let neg_sqrt2 = -AlgebraicReal::sqrt(2);
        assert_eq!(neg_sqrt2.sign(), -1);
    }

    #[test]
    fn test_sqrt2_greater_than_one() {
        let sqrt2 = AlgebraicReal::sqrt(2);
        let one = AlgebraicReal::from_i64(1);
        assert!(sqrt2 > one);
        assert!(one < sqrt2);
        assert_eq!(sqrt2.cmp(&one), Ordering::Greater);
    }

    #[test]
    fn test_sqrt2_less_than_sqrt3() {
        let sqrt2 = AlgebraicReal::sqrt(2);
        let sqrt3 = AlgebraicReal::sqrt(3);
        assert!(sqrt2 < sqrt3);
        assert!(sqrt3 > sqrt2);
        assert_eq!(sqrt2.cmp(&sqrt3), Ordering::Less);
    }

    #[test]
    fn test_sqrt2_not_equal_sqrt3() {
        let sqrt2 = AlgebraicReal::sqrt(2);
        let sqrt3 = AlgebraicReal::sqrt(3);
        assert!(sqrt2 != sqrt3);
        assert_ne!(sqrt2.cmp(&sqrt3), Ordering::Equal);
        assert_ne!((sqrt2 - sqrt3).sign(), 0);
    }

    #[test]
    fn test_sqrt2_times_sqrt2_equals_two_sign_checks() {
        let sqrt2 = AlgebraicReal::sqrt(2);
        let product = sqrt2.clone() * sqrt2.clone();
        let two = AlgebraicReal::from_i64(2);

        // sign(sqrt2*sqrt2 - 2) == 0
        assert_eq!((product.clone() - two.clone()).sign(), 0);
        assert_eq!(product.cmp(&two), Ordering::Equal);
        assert!(product == two);
        assert!(!(product.clone() < two.clone()));
        assert!(!(product > two));
    }
}
