//! Power series implementation

use rustmath_core::{Field, MathError, NumericConversion, Result, Ring};
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

    /// Get the coefficient of xⁿ
    pub fn coeff(&self, n: usize) -> &R {
        self.coeffs.get(n).unwrap_or(&self.coeffs[0])
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

    /// Compose two series: compute f(g(x))
    ///
    /// Requires g(0) = 0 for convergence
    pub fn compose(&self, g: &Self) -> Self
    where
        R: Clone,
    {
        // Check that g(0) = 0
        if !g.coeffs[0].is_zero() {
            // Return zero series if composition doesn't converge
            return Self::zero(self.precision);
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

        result
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

    /* // Commented out: Requires Field trait for division
    /// Compute the integral (with zero constant term)
    pub fn integral(&self) -> Self
    where
        R: NumericConversion,
    {
        let mut coeffs = vec![R::zero()]; // Constant of integration = 0

        for (n, coeff) in self.coeffs.iter().enumerate() {
            if n + 1 < self.precision {
                // Coefficient of x^(n+1) in integral is a_n / (n+1)
                let divisor = R::from_i64((n + 1) as i64);
                let integrated_coeff = coeff.clone() / divisor;
                coeffs.push(integrated_coeff);
            }
        }

        PowerSeries::new(coeffs, self.precision)
    }
    */

    /* // Commented out: Requires Field trait for division
    /// Compute the exponential exp(f) where f(0) = 0
    pub fn exp(&self) -> Self
    where
        R: NumericConversion,
    {
        if !self.coeffs[0].is_zero() {
            // exp(f) requires f(0) = 0 for power series computation
            return Self::constant(R::one(), self.precision);
        }

        let mut result = Self::constant(R::one(), self.precision);
        let mut f_power = self.clone();
        let mut factorial = R::one();

        for n in 1..self.precision {
            // Add f^n / n!
            let n_r = R::from_i64(n as i64);
            factorial = factorial * n_r;

            let mut term_coeffs = f_power.coeffs.clone();
            for coeff in &mut term_coeffs {
                *coeff = coeff.clone() / factorial.clone();
            }

            let term = PowerSeries::new(term_coeffs, self.precision);
            result = result + term;

            if n + 1 < self.precision {
                f_power = f_power.clone() * self.clone();
            }
        }

        result
    }
    */

    /* // Commented out: Requires Field trait for division
    /// Compute the logarithm log(1+f) where f(0) = 0
    pub fn log(&self) -> Self
    where
        R: NumericConversion,
    {
        if !self.coeffs[0].is_zero() {
            // log(1+f) requires f(0) = 0
            return Self::zero(self.precision);
        }

        let mut result = Self::zero(self.precision);
        let mut f_power = self.clone();

        for n in 1..self.precision {
            // Add (-1)^(n+1) * f^n / n
            let n_r = R::from_i64(n as i64);
            let sign = if n % 2 == 1 {
                R::one()
            } else {
                R::zero() - R::one()
            };

            let mut term_coeffs = f_power.coeffs.clone();
            for coeff in &mut term_coeffs {
                *coeff = sign.clone() * coeff.clone() / n_r.clone();
            }

            let term = PowerSeries::new(term_coeffs, self.precision);
            result = result + term;

            if n + 1 < self.precision {
                f_power = f_power.clone() * self.clone();
            }
        }

        result
    }
    */

    /// Compute the multiplicative inverse 1/f where f(0) ≠ 0
    pub fn inverse(&self) -> Result<Self>
    where
        R: Field,
    {
        if self.coeffs[0].is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let a0_inv = self.coeffs[0].inverse()?;

        // Use Newton's method: g_{n+1} = g_n * (2 - f * g_n)
        let mut result = Self::constant(a0_inv, self.precision);
        let two = Self::constant(R::one() + R::one(), self.precision);

        // Iterate to refine
        for _ in 0..5.min(self.precision) {
            let fg = self.clone() * result.clone();
            let correction = two.clone() - fg;
            result = result * correction;
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
}
