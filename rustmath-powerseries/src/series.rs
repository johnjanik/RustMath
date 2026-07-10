//! Power series implementation
//!
//! Analytic operations (`exp`, `log`, `sqrt`, `sin`/`cos`, `integral`,
//! `laplace`) live in [`crate::transcendental`]; they replaced older
//! commented-out drafts that used to sit in this file.  Composition and
//! reversion (Lagrange/compositional inversion) are implemented here.

use crate::precision::Precision;
use rustmath_core::{Field, MathError, Result, Ring};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Truncated power series over a ring R
///
/// Represents Σ(n=0 to precision-1) aₙxⁿ where coefficients are stored in increasing degree order
#[derive(Clone, Debug)]
pub struct PowerSeries<R: Ring> {
    coeffs: Vec<R>,
    precision: usize,
}

impl<R: Ring> PowerSeries<R> {
    /// Create a new power series with given coefficients and precision
    ///
    /// If coefficients are fewer than precision, they are padded with zeros.
    /// If coefficients exceed precision, they are truncated.
    pub fn new(mut coeffs: Vec<R>, precision: usize) -> Self {
        // Truncate or pad as needed
        if coeffs.len() < precision {
            coeffs.resize_with(precision, || R::zero());
        } else if coeffs.len() > precision {
            coeffs.truncate(precision);
        }

        PowerSeries { coeffs, precision }
    }

    /// Create the zero series
    pub fn zero(precision: usize) -> Self {
        PowerSeries {
            coeffs: vec![R::zero(); precision],
            precision,
        }
    }

    /// Create the constant series (just a₀)
    pub fn constant(value: R, precision: usize) -> Self {
        let mut coeffs = vec![R::zero(); precision];
        if precision > 0 {
            coeffs[0] = value;
        }
        PowerSeries { coeffs, precision }
    }

    /// Create the series representing x (identity)
    pub fn var(precision: usize) -> Self {
        let mut coeffs = vec![R::zero(); precision];
        if precision > 1 {
            coeffs[1] = R::one();
        }
        PowerSeries { coeffs, precision }
    }

    /// Create a series from exact coefficient data under a [`Precision`]
    /// regime (MAGMA Chapter 49.1.5 free vs fixed precision).
    ///
    /// The input `coeffs` are treated as **exact** (polynomial) data:
    ///
    /// * `Precision::Fixed(n)` — *capped* semantics: the element lives in a
    ///   fixed-precision ring, so it carries exactly `n` terms; coefficients
    ///   beyond `x^{n-1}` are discarded, shorter inputs are padded with
    ///   genuine zeros.
    /// * `Precision::Free(d)` — *exact-polynomial* semantics: every given
    ///   coefficient is kept (never truncated); shorter inputs are padded
    ///   with genuine zeros up to the ring default `d` terms.
    pub fn with_precision(coeffs: Vec<R>, prec: Precision) -> Self {
        let terms = match prec {
            Precision::Fixed(n) => n,
            Precision::Free(d) => coeffs.len().max(d),
        };
        PowerSeries::new(coeffs, terms)
    }

    /// The generator `x` under a [`Precision`] regime (its term count is the
    /// regime's default).
    pub fn var_with(prec: Precision) -> Self {
        Self::var(prec.default_terms())
    }

    /// Get the coefficient of xⁿ
    ///
    /// # Panics
    ///
    /// Panics if `n >= self.precision()`: the coefficient of `xⁿ` is *unknown*
    /// (the series is only stored modulo `x^precision`), and returning any
    /// value would be fabricated.  (An earlier version silently returned the
    /// constant term here.)
    pub fn coeff(&self, n: usize) -> &R {
        self.coeffs.get(n).unwrap_or_else(|| {
            panic!(
                "PowerSeries::coeff: coefficient {} requested but the series is only known to O(x^{})",
                n, self.precision
            )
        })
    }

    /// Get the precision (number of terms tracked)
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Shift the series: multiply by xⁿ
    pub fn shift(&self, n: usize) -> Self {
        if n == 0 {
            return self.clone();
        }

        let mut new_coeffs = vec![R::zero(); self.precision];

        for i in 0..self.precision {
            if i + n < self.precision {
                new_coeffs[i + n] = self.coeffs[i].clone();
            }
        }

        PowerSeries {
            coeffs: new_coeffs,
            precision: self.precision,
        }
    }

    /// Truncate series to lower precision
    pub fn truncate(&self, new_precision: usize) -> Self {
        let mut coeffs = self.coeffs.clone();
        if new_precision < coeffs.len() {
            coeffs.truncate(new_precision);
        } else {
            coeffs.resize_with(new_precision, || R::zero());
        }

        PowerSeries {
            coeffs,
            precision: new_precision,
        }
    }

    /// Compose two series: compute f(g(x)), requiring `g(0) = 0` so that the
    /// formal composition converges coefficient-wise.
    ///
    /// The result's precision is the minimum of the two operands' precisions
    /// (each term is a product, and products propagate minimum precision).
    ///
    /// Errors with [`MathError::InvalidArgument`] if `g(0) ≠ 0`.
    pub fn try_compose(&self, g: &Self) -> Result<Self> {
        if self.coeffs.is_empty() {
            // Precision 0: nothing is known about the result either.
            return Ok(self.clone());
        }
        // Check that g(0) = 0
        if !g.coeffs.is_empty() && !g.coeffs[0].is_zero() {
            return Err(MathError::InvalidArgument(
                "compose: inner series must have zero constant term".to_string(),
            ));
        }

        let mut result = Self::constant(self.coeffs[0].clone(), self.precision);
        let mut g_power = g.clone();

        for i in 1..self.precision {
            let term = Self::constant(self.coeffs[i].clone(), self.precision) * g_power.clone();
            result = result + term;

            if i + 1 < self.precision {
                g_power = g_power * g.clone();
            }
        }

        Ok(result)
    }

    /// Compose two series: compute f(g(x)).  Requires `g(0) = 0`.
    ///
    /// # Panics
    ///
    /// Panics if `g(0) ≠ 0` (the formal composition does not exist); use
    /// [`Self::try_compose`] for a fallible version.  (An earlier version
    /// silently returned the zero series in that case.)
    pub fn compose(&self, g: &Self) -> Self
    where
        R: Clone,
    {
        self.try_compose(g)
            .expect("compose: inner series must have zero constant term")
    }

    /// Reversion (compositional inverse, MAGMA Chapter 49.4.9 `Reversion`):
    /// given `f = a₁x + a₂x² + …` with `a₀ = 0` and `a₁` invertible, find `g`
    /// with `f(g(x)) = x` (equivalently `g(f(x)) = x`) to this precision.
    ///
    /// Uses Newton iteration `g ← g − (f∘g − x)/(f′∘g)`, doubling the number
    /// of correct terms each step (this is the fast form of Lagrange
    /// inversion; the classical Lagrange formula `n·b_n = [x^{n-1}](x/f)^n`
    /// gives the same coefficients).
    ///
    /// Errors if `f(0) ≠ 0` or if the coefficient of `x` is not invertible.
    pub fn reversion(&self) -> Result<Self>
    where
        R: Field,
    {
        let p = self.precision;
        if p == 0 {
            return Ok(self.clone());
        }
        if !self.coeffs[0].is_zero() {
            return Err(MathError::InvalidArgument(
                "reversion: series must have zero constant term".to_string(),
            ));
        }
        if p == 1 {
            // Only O(x) data: the reverse is 0 + O(x).
            return Ok(Self::zero(1));
        }
        if self.coeffs[1].is_zero() {
            return Err(MathError::InvalidArgument(
                "reversion: coefficient of x must be invertible".to_string(),
            ));
        }
        let a1_inv = self.coeffs[1].inverse()?;

        // g₀ = a₁⁻¹ x is correct to O(x²); each Newton step doubles that.
        let mut g = PowerSeries::new(vec![R::zero(), a1_inv], p);
        let fp = self.derivative();
        let id = Self::var(p);
        let mut correct = 2usize;
        while correct < p {
            let fg = self.try_compose(&g)?;
            // f′(g) has constant term a₁ ≠ 0, hence is invertible.
            let denom_inv = fp.try_compose(&g)?.inverse()?;
            g = g - (fg - id.clone()) * denom_inv;
            correct *= 2;
        }
        Ok(g)
    }

    /// Compute the derivative
    pub fn derivative(&self) -> Self {
        let mut coeffs = Vec::with_capacity(self.precision);

        for n in 1..self.precision {
            // Coefficient of x^(n-1) in derivative is n * a_n
            let mut coeff = R::zero();
            for _ in 0..n {
                coeff = coeff + self.coeffs[n].clone();
            }
            coeffs.push(coeff);
        }

        // Pad with zero if needed
        if coeffs.len() < self.precision {
            coeffs.resize_with(self.precision, || R::zero());
        }

        PowerSeries {
            coeffs,
            precision: self.precision,
        }
    }

    // NOTE: `integral`, `exp`, `log` (and `sqrt`, `sin`/`cos`, `laplace`) are
    // implemented in `crate::transcendental` via the exact coefficient
    // recurrences over any `Field + NumericConversion`, with honest errors on
    // domain violations (`exp` needs f(0)=0, `log` needs f(0)=1, `integral`
    // needs each n+1 invertible).  Older draft implementations that used to
    // sit here commented-out were superseded by that module.

    /// Compute the multiplicative inverse 1/f where f(0) ≠ 0
    pub fn inverse(&self) -> Result<Self>
    where
        R: Field,
    {
        if self.coeffs.is_empty() || self.coeffs[0].is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let a0_inv = self.coeffs[0].inverse()?;

        // Use Newton's method: g_{n+1} = g_n * (2 - f * g_n).  The initial
        // guess is correct to 1 term and every iteration doubles the number
        // of correct terms, so iterate until 2^k ≥ precision.  (An earlier
        // version hard-coded 5 iterations and was wrong beyond 32 terms.)
        let mut result = Self::constant(a0_inv, self.precision);
        let two = Self::constant(R::one() + R::one(), self.precision);

        let mut correct = 1usize;
        while correct < self.precision {
            let fg = self.clone() * result.clone();
            let correction = two.clone() - fg;
            result = result * correction;
            correct *= 2;
        }

        Ok(result)
    }
}

impl<R: Ring> fmt::Display for PowerSeries<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;

        for (i, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }

            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if i == 0 {
                write!(f, "{}", coeff)?;
            } else if i == 1 {
                if coeff.is_one() {
                    write!(f, "x")?;
                } else {
                    write!(f, "{}*x", coeff)?;
                }
            } else if coeff.is_one() {
                write!(f, "x^{}", i)?;
            } else {
                write!(f, "{}*x^{}", coeff, i)?;
            }
        }

        if first {
            write!(f, "0")?;
        }

        write!(f, " + O(x^{})", self.precision)
    }
}

// Typesetting implementation for power series where coefficients implement MathDisplay
impl<R> rustmath_typesetting::MathDisplay for PowerSeries<R>
where
    R: Ring + rustmath_typesetting::MathDisplay + std::fmt::Display,
{
    fn math_format(&self, options: &rustmath_typesetting::FormatOptions) -> String {
        use rustmath_typesetting::OutputFormat;

        let var_name = "x";
        let mut result = String::new();
        let mut first = true;

        // Format polynomial terms
        for (i, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }

            let coeff_str = coeff.to_string();

            if !first {
                result.push_str(" + ");
            }
            first = false;

            // Format the term (similar to polynomial)
            if i == 0 {
                result.push_str(&coeff_str);
            } else if i == 1 {
                if coeff.is_one() {
                    result.push_str(var_name);
                } else {
                    result.push_str(&coeff_str);
                    if !options.implicit_multiply {
                        result.push('*');
                    }
                    result.push_str(var_name);
                }
            } else {
                if !coeff.is_one() {
                    result.push_str(&coeff_str);
                    if !options.implicit_multiply {
                        result.push('*');
                    }
                }
                match options.format {
                    OutputFormat::LaTeX => {
                        result.push_str(&format!("{}^{{{}}}", var_name, i));
                    }
                    OutputFormat::Unicode => {
                        result.push_str(var_name);
                        result.push_str(&rustmath_typesetting::utils::to_superscript(&i.to_string()));
                    }
                    _ => {
                        result.push_str(&format!("{}^{}", var_name, i));
                    }
                }
            }
        }

        if first {
            result.push('0');
        }

        // Add big-O notation for truncation
        result.push_str(" + ");
        match options.format {
            OutputFormat::LaTeX => {
                result.push_str(&format!(r"O({}^{{{}}})", var_name, self.precision));
            }
            OutputFormat::Unicode => {
                result.push_str("O(");
                result.push_str(var_name);
                result.push_str(&rustmath_typesetting::utils::to_superscript(&self.precision.to_string()));
                result.push(')');
            }
            _ => {
                result.push_str(&format!("O({}^{})", var_name, self.precision));
            }
        }

        result
    }

    fn precedence(&self) -> i32 {
        rustmath_typesetting::utils::precedence::ADD
    }
}

impl<R: Ring> Add for PowerSeries<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let precision = self.precision.min(other.precision);
        let mut coeffs = Vec::with_capacity(precision);

        for i in 0..precision {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            let b = other.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            coeffs.push(a + b);
        }

        PowerSeries { coeffs, precision }
    }
}

impl<R: Ring> Sub for PowerSeries<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let precision = self.precision.min(other.precision);
        let mut coeffs = Vec::with_capacity(precision);

        for i in 0..precision {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            let b = other.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            coeffs.push(a - b);
        }

        PowerSeries { coeffs, precision }
    }
}

impl<R: Ring> Mul for PowerSeries<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let precision = self.precision.min(other.precision);
        let mut coeffs = vec![R::zero(); precision];

        for i in 0..precision {
            for j in 0..=i {
                if j < self.coeffs.len() && (i - j) < other.coeffs.len() {
                    coeffs[i] = coeffs[i].clone()
                        + self.coeffs[j].clone() * other.coeffs[i - j].clone();
                }
            }
        }

        PowerSeries { coeffs, precision }
    }
}

impl<R: Ring> Neg for PowerSeries<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let coeffs = self.coeffs.into_iter().map(|c| -c).collect();
        PowerSeries {
            coeffs,
            precision: self.precision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_basic_ops() {
        let s1 = PowerSeries::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            5,
        );
        let s2 = PowerSeries::new(
            vec![Integer::from(1), Integer::from(1), Integer::from(1)],
            5,
        );

        let sum = s1.clone() + s2.clone();
        assert_eq!(sum.coeff(0), &Integer::from(2));
        assert_eq!(sum.coeff(1), &Integer::from(3));
        assert_eq!(sum.coeff(2), &Integer::from(4));
    }

    #[test]
    fn test_multiplication() {
        // (1 + x) * (1 + x) = 1 + 2x + x^2
        let s1 = PowerSeries::new(vec![Integer::from(1), Integer::from(1)], 5);

        let prod = s1.clone() * s1;
        assert_eq!(prod.coeff(0), &Integer::from(1));
        assert_eq!(prod.coeff(1), &Integer::from(2));
        assert_eq!(prod.coeff(2), &Integer::from(1));
    }

    #[test]
    fn test_shift() {
        let s = PowerSeries::new(vec![Integer::from(1), Integer::from(2)], 5);
        let shifted = s.shift(2);

        assert_eq!(shifted.coeff(0), &Integer::from(0));
        assert_eq!(shifted.coeff(1), &Integer::from(0));
        assert_eq!(shifted.coeff(2), &Integer::from(1));
        assert_eq!(shifted.coeff(3), &Integer::from(2));
    }

    #[test]
    fn test_derivative() {
        // Derivative of 1 + 2x + 3x^2 is 2 + 6x
        let s = PowerSeries::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            5,
        );
        let deriv = s.derivative();

        assert_eq!(deriv.coeff(0), &Integer::from(2));
        assert_eq!(deriv.coeff(1), &Integer::from(6));
    }

    #[test]
    #[should_panic(expected = "only known to O(x^3)")]
    fn coeff_beyond_precision_panics() {
        // The coefficient of x^5 of a series known mod x^3 is unknown;
        // asking for it must not fabricate a value.
        let s = PowerSeries::new(vec![Integer::from(7), Integer::from(1)], 3);
        let _ = s.coeff(5);
    }

    #[test]
    fn with_precision_capped_vs_exact() {
        use crate::precision::Precision;
        let coeffs = vec![Integer::from(1), Integer::from(2), Integer::from(3)];
        // Fixed(2) caps: exactly 2 terms, the x^2 coefficient is discarded.
        let capped = PowerSeries::with_precision(coeffs.clone(), Precision::Fixed(2));
        assert_eq!(capped.precision(), 2);
        assert_eq!(capped.coeff(1), &Integer::from(2));
        // Free(5) keeps everything and pads to the default term count.
        let free = PowerSeries::with_precision(coeffs.clone(), Precision::Free(5));
        assert_eq!(free.precision(), 5);
        assert_eq!(free.coeff(2), &Integer::from(3));
        assert_eq!(free.coeff(4), &Integer::from(0));
        // Free never truncates: a longer exact input keeps all its terms.
        let long = PowerSeries::with_precision(coeffs, Precision::Free(2));
        assert_eq!(long.precision(), 3);
        assert_eq!(long.coeff(2), &Integer::from(3));
        // The generator under a regime uses the regime's default term count.
        assert_eq!(
            PowerSeries::<Integer>::var_with(Precision::Fixed(4)).precision(),
            4
        );
    }

    #[test]
    fn arithmetic_propagates_min_precision() {
        let a = PowerSeries::new(vec![Integer::from(1), Integer::from(2)], 3);
        let b = PowerSeries::new(vec![Integer::from(1)], 7);
        assert_eq!((a.clone() + b.clone()).precision(), 3);
        assert_eq!((a.clone() - b.clone()).precision(), 3);
        assert_eq!((a.clone() * b.clone()).precision(), 3);
        // compose propagates min precision too
        let g = PowerSeries::new(vec![Integer::from(0), Integer::from(1)], 5);
        assert_eq!(a.compose(&g).precision(), 3);
        assert_eq!(g.compose(&a.shift(1)).precision(), 3);
    }

    mod field_ops {
        use super::PowerSeries;
        use rustmath_rationals::Rational;

        fn q(n: i64) -> Rational {
            Rational::from_i64(n)
        }

        #[test]
        fn compose_rejects_nonzero_constant_term() {
            let f = PowerSeries::new(vec![q(1), q(1)], 4);
            let g = PowerSeries::new(vec![q(1), q(1)], 4); // g(0) = 1 ≠ 0
            assert!(f.try_compose(&g).is_err());
        }

        #[test]
        fn inverse_correct_beyond_32_terms() {
            // 1/(1-x) = Σ x^n; the old hard-coded 5 Newton iterations were
            // only correct to 32 terms.
            let p = 40;
            let one_minus_x = PowerSeries::new(vec![q(1), q(-1)], p);
            let inv = one_minus_x.inverse().unwrap();
            for i in 0..p {
                assert_eq!(inv.coeff(i), &q(1), "coeff {i}");
            }
        }

        #[test]
        fn reversion_of_x_plus_x2_is_signed_catalan() {
            // Reversion of f = x + x^2 is g = (-1 + sqrt(1+4x))/2
            //   = Σ_{n≥1} (-1)^{n-1} C_{n-1} x^n  (Catalan numbers C_k).
            // Coefficients verified independently with sympy:
            //   [0, 1, -1, 2, -5, 14, -42, 132, -429, 1430]
            let p = 10;
            let f = PowerSeries::new(vec![q(0), q(1), q(1)], p);
            let g = f.reversion().unwrap();
            let expected = [0i64, 1, -1, 2, -5, 14, -42, 132, -429, 1430];
            for (i, &e) in expected.iter().enumerate() {
                assert_eq!(g.coeff(i), &q(e), "coeff {i}");
            }
            // Both round-trips are the identity to the working precision.
            let fg = f.compose(&g);
            let gf = g.compose(&f);
            let id = PowerSeries::<Rational>::var(p);
            for i in 0..p {
                assert_eq!(fg.coeff(i), id.coeff(i), "f(g) coeff {i}");
                assert_eq!(gf.coeff(i), id.coeff(i), "g(f) coeff {i}");
            }
        }

        #[test]
        fn reversion_general_linear_coefficient() {
            // f = 2x + x^3: reversion needs a_1 = 2 inverted, not just a_1 = 1.
            let p = 9;
            let f = PowerSeries::new(vec![q(0), q(2), q(0), q(1)], p);
            let g = f.reversion().unwrap();
            let id = PowerSeries::<Rational>::var(p);
            let fg = f.compose(&g);
            for i in 0..p {
                assert_eq!(fg.coeff(i), id.coeff(i), "f(g) coeff {i}");
            }
        }

        #[test]
        fn reversion_domain_errors() {
            // f(0) ≠ 0 has no compositional inverse.
            assert!(PowerSeries::new(vec![q(1), q(1)], 5).reversion().is_err());
            // f'(0) = 0 is not invertible.
            assert!(PowerSeries::new(vec![q(0), q(0), q(1)], 5)
                .reversion()
                .is_err());
        }
    }
}
