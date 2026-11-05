//! Transcendental and analytic operations on power series.
//!
//! MAGMA source: Handbook Chapter 49 §49.4.7 (Integral, Laplace), §49.4.8
//! (Square Root), §49.5 (Transcendental Functions: Exp/Log, Sin/Cos).
//!
//! Chapter 49.5 requires the coefficient ring to be a **field**, and (for the
//! genuinely transcendental content) a field of **characteristic zero** so that
//! the division by `n` / `n!` in the defining recurrences is legal.  Rust's core
//! tower has no characteristic-zero marker, so we gate on `Field +
//! NumericConversion` (to embed the integers `n`) and make every routine return
//! a `Result`: if an integer denominator is not invertible in `R` — e.g. `R` is
//! a finite field of characteristic `p` and `p | n` — the operation fails with
//! `MathError::DivisionByZero` instead of returning a wrong answer.  This keeps
//! the "status honesty" discipline: no silently incorrect output.
//!
//! All routines use the coefficient recurrences derived from the defining ODEs
//! (`(exp f)' = f' exp f`, `(log f)' = f'/f`, `sqrt(f)^2 = f`, …) which are exact
//! and avoid forming high powers of `f`.

use crate::series::PowerSeries;
use rustmath_core::{Field, MathError, NumericConversion, Result};

/// `1/n` in `R`, or `Err` if `n` is not invertible in `R` (characteristic `> 0`
/// dividing `n`, or `n = 0`).
fn inv_int<R: Field + NumericConversion>(n: i64) -> Result<R> {
    R::from_i64(n).inverse()
}

impl<R: Field + NumericConversion> PowerSeries<R> {
    /// The coefficients `[a_0, .., a_{p-1}]` as an owned vector.
    fn coeffs_owned(&self) -> Vec<R> {
        (0..self.precision()).map(|i| self.coeff(i).clone()).collect()
    }

    /// Antiderivative `F` with `F(0) = 0` (Chapter 49.4.7 `Integral`): if
    /// `self = Σ a_n x^n` then `F = Σ a_n/(n+1) x^{n+1}`.
    ///
    /// Errors if some `n+1` is not invertible in `R`.
    pub fn integral(&self) -> Result<Self> {
        let a = self.coeffs_owned();
        let p = self.precision();
        let mut out = vec![R::zero(); p];
        for n in 0..p {
            if n + 1 < p {
                out[n + 1] = a[n].clone() * inv_int::<R>((n + 1) as i64)?;
            }
        }
        Ok(PowerSeries::new(out, p))
    }

    /// Laplace transform (Chapter 49.4.7): `Σ a_i x^i ↦ Σ i! a_i x^i`.
    /// Requires non-negative integral valuation (always true for a power series).
    pub fn laplace(&self) -> Self {
        let a = self.coeffs_owned();
        let p = self.precision();
        let mut fact = R::one();
        let mut out = vec![R::zero(); p];
        for (i, ai) in a.iter().enumerate() {
            if i > 0 {
                fact = fact * R::from_i64(i as i64);
            }
            out[i] = fact.clone() * ai.clone();
        }
        PowerSeries::new(out, p)
    }

    /// Exponential `exp(self)` for a series with **zero constant term**
    /// (Chapter 49.5.1).  Uses `g' = f' g`, `g_0 = 1`,
    /// `g_n = (1/n) Σ_{k=1}^n k a_k g_{n-k}`.
    ///
    /// Errors if the constant term is non-zero (formal `exp` needs a real/complex
    /// coefficient domain to evaluate `exp(a_0)`), or if some `n` is not
    /// invertible in `R`.
    pub fn exp(&self) -> Result<Self> {
        let a = self.coeffs_owned();
        let p = self.precision();
        if p == 0 {
            return Ok(self.clone());
        }
        if !a[0].is_zero() {
            return Err(MathError::InvalidArgument(
                "exp: series must have zero constant term".to_string(),
            ));
        }
        let mut g = vec![R::zero(); p];
        g[0] = R::one();
        for n in 1..p {
            let mut acc = R::zero();
            for k in 1..=n {
                // k * a_k * g_{n-k}
                acc = acc + R::from_i64(k as i64) * a[k].clone() * g[n - k].clone();
            }
            g[n] = acc * inv_int::<R>(n as i64)?;
        }
        Ok(PowerSeries::new(g, p))
    }

    /// Natural logarithm `log(self)` for a series with **constant term one**
    /// (Chapter 49.5.1, char. 0).  Uses `g' = f'/f`, `g_0 = 0`,
    /// so `n g_n = n a_n - Σ_{k=1}^{n-1} k g_k a_{n-k}`.
    ///
    /// Errors if the constant term is not `1` (a general non-zero constant needs
    /// `log(a_0)` from a real/complex domain), or if some `n` is not invertible.
    pub fn log(&self) -> Result<Self> {
        let a = self.coeffs_owned();
        let p = self.precision();
        if p == 0 {
            return Ok(self.clone());
        }
        if !a[0].is_one() {
            return Err(MathError::InvalidArgument(
                "log: series must have constant term 1".to_string(),
            ));
        }
        let mut g = vec![R::zero(); p];
        // g_0 = log(1) = 0
        for n in 1..p {
            // n a_n - Σ_{k=1}^{n-1} k g_k a_{n-k}
            let mut acc = R::from_i64(n as i64) * a[n].clone();
            for k in 1..n {
                acc = acc - R::from_i64(k as i64) * g[k].clone() * a[n - k].clone();
            }
            g[n] = acc * inv_int::<R>(n as i64)?;
        }
        Ok(PowerSeries::new(g, p))
    }

    /// Square root of a series with **constant term one** (Chapter 49.4.8).
    /// Uses `g^2 = f`, `g_0 = 1`,
    /// `g_n = (1/2)(a_n - Σ_{k=1}^{n-1} g_k g_{n-k})`.
    ///
    /// A general leading square root would need `Sqrt` on the coefficient field;
    /// we restrict to constant term `1`, the case that arises from
    /// `sqrt(1 + higher-order)`.  Errors otherwise (or if `2` is not invertible).
    pub fn sqrt(&self) -> Result<Self> {
        let a = self.coeffs_owned();
        let p = self.precision();
        if p == 0 {
            return Ok(self.clone());
        }
        if !a[0].is_one() {
            return Err(MathError::InvalidArgument(
                "sqrt: series must have constant term 1".to_string(),
            ));
        }
        let half = inv_int::<R>(2)?;
        let mut g = vec![R::zero(); p];
        g[0] = R::one();
        for n in 1..p {
            let mut acc = a[n].clone();
            for k in 1..n {
                acc = acc - g[k].clone() * g[n - k].clone();
            }
            g[n] = acc * half.clone();
        }
        Ok(PowerSeries::new(g, p))
    }

    /// `(sin(self), cos(self))` for a series with **zero constant term**
    /// (Chapter 49.5.2).  Computed together via `s' = f' c`, `c' = -f' s`,
    /// `s_0 = 0`, `c_0 = 1`.
    ///
    /// Errors if the constant term is non-zero or some `n` is not invertible.
    pub fn sin_cos(&self) -> Result<(Self, Self)> {
        let a = self.coeffs_owned();
        let p = self.precision();
        if p == 0 {
            return Ok((self.clone(), self.clone()));
        }
        if !a[0].is_zero() {
            return Err(MathError::InvalidArgument(
                "sin/cos: series must have zero constant term".to_string(),
            ));
        }
        let mut s = vec![R::zero(); p];
        let mut c = vec![R::zero(); p];
        c[0] = R::one();
        for n in 1..p {
            // n s_n =  Σ_{k=1}^n k a_k c_{n-k}
            // n c_n = -Σ_{k=1}^n k a_k s_{n-k}
            let mut sn = R::zero();
            let mut cn = R::zero();
            for k in 1..=n {
                let kak = R::from_i64(k as i64) * a[k].clone();
                sn = sn + kak.clone() * c[n - k].clone();
                cn = cn - kak * s[n - k].clone();
            }
            let inv_n = inv_int::<R>(n as i64)?;
            s[n] = sn * inv_n.clone();
            c[n] = cn * inv_n;
        }
        Ok((PowerSeries::new(s, p), PowerSeries::new(c, p)))
    }

    /// `sin(self)` (see [`Self::sin_cos`]).
    pub fn sin(&self) -> Result<Self> {
        Ok(self.sin_cos()?.0)
    }

    /// `cos(self)` (see [`Self::sin_cos`]).
    pub fn cos(&self) -> Result<Self> {
        Ok(self.sin_cos()?.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn q(n: i64) -> Rational {
        Rational::from_i64(n)
    }
    fn qq(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }
    fn x(prec: usize) -> PowerSeries<Rational> {
        PowerSeries::var(prec)
    }

    #[test]
    fn exp_of_x() {
        // exp(x) = 1 + x + x^2/2 + x^3/6 + x^4/24 + ...
        let e = x(6).exp().unwrap();
        assert_eq!(e.coeff(0), &q(1));
        assert_eq!(e.coeff(1), &q(1));
        assert_eq!(e.coeff(2), &qq(1, 2));
        assert_eq!(e.coeff(3), &qq(1, 6));
        assert_eq!(e.coeff(4), &qq(1, 24));
        assert_eq!(e.coeff(5), &qq(1, 120));
    }

    #[test]
    fn log_exp_roundtrip() {
        // log(exp(x)) = x
        let e = x(8).exp().unwrap();
        let l = e.log().unwrap();
        assert_eq!(l.coeff(0), &q(0));
        assert_eq!(l.coeff(1), &q(1));
        for i in 2..8 {
            assert_eq!(l.coeff(i), &q(0), "coeff {i}");
        }
    }

    #[test]
    fn exp_log_roundtrip_one_plus_x() {
        // exp(log(1+x)) = 1 + x to the working precision.
        let p = 10;
        let one_plus_x = PowerSeries::new(vec![q(1), q(1)], p);
        let back = one_plus_x.log().unwrap().exp().unwrap();
        assert_eq!(back.coeff(0), &q(1));
        assert_eq!(back.coeff(1), &q(1));
        for i in 2..p {
            assert_eq!(back.coeff(i), &q(0), "coeff {i}");
        }
    }

    #[test]
    fn exp_satisfies_its_ode() {
        // d/dx exp(f) = f' · exp(f) for f = x + 2x² − x³/3 (zero constant term).
        // exp(f) itself verified independently with sympy:
        //   [1, 1, 5/2, 11/6, 65/24, 181/120, 1261/720, 731/1008]
        let p = 8;
        let f = PowerSeries::new(vec![q(0), q(1), q(2), qq(-1, 3)], p);
        let ef = f.exp().unwrap();
        let expected = [
            qq(1, 1),
            qq(1, 1),
            qq(5, 2),
            qq(11, 6),
            qq(65, 24),
            qq(181, 120),
            qq(1261, 720),
            qq(731, 1008),
        ];
        for (i, e) in expected.iter().enumerate() {
            assert_eq!(ef.coeff(i), e, "exp(f) coeff {i}");
        }
        let lhs = ef.derivative();
        let rhs = f.derivative() * ef.clone();
        // The derivative of a series known mod x^p is known mod x^{p-1}, so
        // the identity holds for coefficients 0..p-2 (the working precision
        // of the derivative).
        for i in 0..p - 1 {
            assert_eq!(lhs.coeff(i), rhs.coeff(i), "ODE coeff {i}");
        }
    }

    #[test]
    fn exp_and_log_reject_bad_constant_terms() {
        // exp needs f(0) = 0; log needs f(0) = 1. Anything else is an error,
        // never a fabricated value.
        let nonzero_const = PowerSeries::new(vec![q(2), q(1)], 5);
        assert!(nonzero_const.exp().is_err());
        assert!(nonzero_const.log().is_err());
        let zero_const = PowerSeries::new(vec![q(0), q(1)], 5);
        assert!(zero_const.log().is_err()); // log needs constant term exactly 1
    }

    #[test]
    fn integral_divides_exactly() {
        // ∫(1 + x + x²) = x + x²/2 + x³/3 with exact rational division.
        let f = PowerSeries::new(vec![q(1), q(1), q(1)], 5);
        let int = f.integral().unwrap();
        assert_eq!(int.coeff(0), &q(0));
        assert_eq!(int.coeff(1), &q(1));
        assert_eq!(int.coeff(2), &qq(1, 2));
        assert_eq!(int.coeff(3), &qq(1, 3));
    }

    #[test]
    fn log_of_one_plus_x() {
        // log(1+x) = x - x^2/2 + x^3/3 - x^4/4 + ...
        let one_plus_x = PowerSeries::new(vec![q(1), q(1)], 6);
        let l = one_plus_x.log().unwrap();
        assert_eq!(l.coeff(1), &q(1));
        assert_eq!(l.coeff(2), &qq(-1, 2));
        assert_eq!(l.coeff(3), &qq(1, 3));
        assert_eq!(l.coeff(4), &qq(-1, 4));
    }

    #[test]
    fn sqrt_squares_back() {
        // sqrt(1+x)^2 = 1 + x
        let one_plus_x = PowerSeries::new(vec![q(1), q(1)], 8);
        let s = one_plus_x.sqrt().unwrap();
        // sqrt(1+x) = 1 + x/2 - x^2/8 + x^3/16 - ...
        assert_eq!(s.coeff(0), &q(1));
        assert_eq!(s.coeff(1), &qq(1, 2));
        assert_eq!(s.coeff(2), &qq(-1, 8));
        assert_eq!(s.coeff(3), &qq(1, 16));
        let back = s.clone() * s;
        assert_eq!(back.coeff(0), &q(1));
        assert_eq!(back.coeff(1), &q(1));
        for i in 2..8 {
            assert_eq!(back.coeff(i), &q(0), "coeff {i}");
        }
    }

    #[test]
    fn sin_cos_pythagoras() {
        // sin^2 + cos^2 = 1
        let (s, c) = x(9).sin_cos().unwrap();
        // sin(x) = x - x^3/6 + x^5/120
        assert_eq!(s.coeff(1), &q(1));
        assert_eq!(s.coeff(3), &qq(-1, 6));
        assert_eq!(s.coeff(5), &qq(1, 120));
        // cos(x) = 1 - x^2/2 + x^4/24
        assert_eq!(c.coeff(0), &q(1));
        assert_eq!(c.coeff(2), &qq(-1, 2));
        assert_eq!(c.coeff(4), &qq(1, 24));
        let id = s.clone() * s + c.clone() * c;
        assert_eq!(id.coeff(0), &q(1));
        for i in 1..9 {
            assert_eq!(id.coeff(i), &q(0), "coeff {i}");
        }
    }

    #[test]
    fn integral_of_derivative() {
        // Integral(Derivative(f)) recovers f up to constant term.
        let f = PowerSeries::new(vec![q(0), q(1), q(2), q(3)], 6);
        let g = f.derivative().integral().unwrap();
        for i in 1..6 {
            assert_eq!(g.coeff(i), f.coeff(i), "coeff {i}");
        }
    }

    #[test]
    fn laplace_scales_by_factorial() {
        // Laplace(exp(x)) = Σ i! (1/i!) x^i = Σ x^i = 1/(1-x)
        let e = x(6).exp().unwrap();
        let lp = e.laplace();
        for i in 0..6 {
            assert_eq!(lp.coeff(i), &q(1), "coeff {i}");
        }
    }
}
