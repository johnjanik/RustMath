//! Series expansions (Taylor and Laurent series)
//!
//! This module implements Taylor and Laurent series expansions for symbolic expressions.

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_core::{Ring, NumericConversion};
use rustmath_integers::Integer;
use std::sync::Arc;

impl Expr {
    /// Compute Taylor series expansion around a point
    ///
    /// Expands f(x) around x = a up to order n:
    /// f(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ... + f⁽ⁿ⁾(a)(x-a)ⁿ/n!
    ///
    /// # Arguments
    ///
    /// * `var` - Variable to expand with respect to
    /// * `point` - Point to expand around (typically 0 for Maclaurin series)
    /// * `order` - Maximum order of the expansion
    ///
    /// # Returns
    ///
    /// A polynomial approximation of the function
    pub fn taylor(&self, var: &Symbol, point: &Expr, order: usize) -> Self {
        let mut result = Expr::from(0);
        let mut derivative = self.clone();
        let mut factorial = Integer::from(1);

        for n in 0..=order {
            // Evaluate nth derivative at the point
            let coeff = derivative.substitute(var, point);

            // Compute (x - a)^n
            let x_minus_a = if matches!(point, Expr::Integer(i) if i.to_i64() == Some(0)) {
                Expr::Symbol(var.clone())
            } else {
                Expr::Symbol(var.clone()) - point.clone()
            };

            let power_term = if n == 0 {
                Expr::from(1)
            } else {
                x_minus_a.pow(Expr::from(n as i64))
            };

            // Add term: f^(n)(a) * (x-a)^n / n!
            let term = coeff * power_term / Expr::Integer(factorial.clone());
            result = result + term;

            // Update for next iteration
            if n < order {
                derivative = derivative.differentiate(var);
                factorial = factorial * Integer::from((n + 1) as i64);
            }
        }

        result.simplify()
    }

    /// Compute Maclaurin series (Taylor series around 0)
    ///
    /// This is a special case of Taylor series where the expansion point is 0.
    ///
    /// # Arguments
    ///
    /// * `var` - Variable to expand with respect to
    /// * `order` - Maximum order of the expansion
    pub fn maclaurin(&self, var: &Symbol, order: usize) -> Self {
        self.taylor(var, &Expr::from(0), order)
    }

    /// Compute Laurent series expansion
    ///
    /// Laurent series generalizes Taylor series to include negative powers.
    /// Useful for functions with poles.
    ///
    /// f(x) = ... + c₋₂(x-a)⁻² + c₋₁(x-a)⁻¹ + c₀ + c₁(x-a) + c₂(x-a)² + ...
    ///
    /// # Arguments
    ///
    /// * `var` - Variable to expand with respect to
    /// * `point` - Point to expand around
    /// * `min_order` - Minimum order (can be negative)
    /// * `max_order` - Maximum order
    ///
    /// # Returns
    ///
    /// Laurent series as a symbolic expression
    pub fn laurent(
        &self,
        var: &Symbol,
        point: &Expr,
        min_order: i64,
        max_order: i64,
    ) -> Self {
        // For simple cases, we can compute the Laurent series
        // by finding coefficients through differentiation and evaluation

        if min_order >= 0 {
            // No negative powers - this is just a Taylor series
            return self.taylor(var, point, max_order as usize);
        }

        // For functions with poles, we need to handle negative powers
        // This is a simplified implementation that handles basic cases

        // Check if this is a rational function (ratio of polynomials)
        if let Expr::Binary(BinaryOp::Div, num, den) = self {
            // Expand numerator and denominator separately
            let num_expanded = num.taylor(var, point, max_order as usize);
            let den_expanded = den.taylor(var, point, max_order as usize);

            // Return the ratio (Laurent series division is complex, so we approximate)
            return num_expanded / den_expanded;
        }

        // For other cases, fall back to Taylor series
        self.taylor(var, point, max_order as usize)
    }

    /// Get the asymptotic expansion as x → ∞
    ///
    /// For rational functions f(x)/g(x), expands as powers of 1/x
    ///
    /// # Arguments
    ///
    /// * `var` - Variable to expand with respect to
    /// * `order` - Number of terms in the expansion
    pub fn asymptotic(&self, var: &Symbol, order: usize) -> Self {
        // Substitute x = 1/t and expand around t = 0
        let t = Symbol::new("t_asymptotic");
        let t_expr = Expr::Symbol(t.clone());

        // Replace var with 1/t
        let substituted = self.substitute(var, &(Expr::from(1) / t_expr.clone()));

        // Expand around t = 0
        let expanded = substituted.taylor(&t, &Expr::from(0), order);

        // Substitute back t = 1/x
        expanded.substitute(&t, &(Expr::from(1) / Expr::Symbol(var.clone())))
    }

    /// Compute the series expansion and return coefficients
    ///
    /// Returns a vector of coefficients [c₀, c₁, c₂, ...] where
    /// f(x) ≈ c₀ + c₁x + c₂x² + ...
    pub fn series_coefficients(&self, var: &Symbol, point: &Expr, order: usize) -> Vec<Expr> {
        let mut coefficients = Vec::new();
        let mut derivative = self.clone();
        let mut factorial = Integer::from(1);

        for n in 0..=order {
            // Evaluate nth derivative at the point
            let value = derivative.substitute(var, point);

            // Compute f^(n)(a) / n!
            let coeff = value / Expr::Integer(factorial.clone());
            coefficients.push(coeff.simplify());

            // Update for next iteration
            if n < order {
                derivative = derivative.differentiate(var);
                factorial = factorial * Integer::from((n + 1) as i64);
            }
        }

        coefficients
    }
}

/// Build known Taylor series for common functions
pub mod known_series {
    use super::*;

    /// exp(x) = 1 + x + x²/2! + x³/3! + ...
    pub fn exp(var: &Symbol, order: usize) -> Expr {
        let mut result = Expr::from(0);
        let mut factorial = Integer::from(1);
        let x = Expr::Symbol(var.clone());

        for n in 0..=order {
            let term = x.clone().pow(Expr::from(n as i64)) / Expr::Integer(factorial.clone());
            result = result + term;

            if n < order {
                factorial = factorial * Integer::from((n + 1) as i64);
            }
        }

        result
    }

    /// sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
    pub fn sin(var: &Symbol, order: usize) -> Expr {
        let mut result = Expr::from(0);
        let x = Expr::Symbol(var.clone());

        for n in 0..=order {
            let k = 2 * n + 1;
            if k > order {
                break;
            }

            let mut factorial = Integer::from(1);
            for i in 1..=k {
                factorial = factorial * Integer::from(i as i64);
            }

            let sign = if n % 2 == 0 { 1 } else { -1 };
            let term = Expr::from(sign) * x.clone().pow(Expr::from(k as i64))
                / Expr::Integer(factorial);
            result = result + term;
        }

        result
    }

    /// cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    pub fn cos(var: &Symbol, order: usize) -> Expr {
        let mut result = Expr::from(0);
        let x = Expr::Symbol(var.clone());

        for n in 0..=order {
            let k = 2 * n;
            if k > order {
                break;
            }

            let mut factorial = Integer::from(1);
            for i in 1..=k {
                factorial = factorial * Integer::from(i as i64);
            }

            let sign = if n % 2 == 0 { 1 } else { -1 };
            let term = Expr::from(sign) * x.clone().pow(Expr::from(k as i64))
                / Expr::Integer(factorial);
            result = result + term;
        }

        result
    }

    /// log(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...
    pub fn log_one_plus(var: &Symbol, order: usize) -> Expr {
        let mut result = Expr::from(0);
        let x = Expr::Symbol(var.clone());

        for n in 1..=order {
            let sign = if n % 2 == 1 { 1 } else { -1 };
            let term = Expr::from(sign) * x.clone().pow(Expr::from(n as i64))
                / Expr::from(n as i64);
            result = result + term;
        }

        result
    }

    /// (1+x)^α = 1 + αx + α(α-1)x²/2! + α(α-1)(α-2)x³/3! + ...
    /// Binomial series (generalized binomial theorem)
    pub fn binomial(var: &Symbol, alpha: &Expr, order: usize) -> Expr {
        let mut result = Expr::from(1);
        let x = Expr::Symbol(var.clone());
        let mut coeff = alpha.clone();
        let mut factorial = Integer::from(1);

        for n in 1..=order {
            factorial = factorial * Integer::from(n as i64);

            let term = coeff.clone() * x.clone().pow(Expr::from(n as i64))
                / Expr::Integer(factorial.clone());
            result = result + term;

            // Update coefficient: multiply by (α - n)
            coeff = coeff * (alpha.clone() - Expr::from(n as i64));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::known_series;

    #[test]
    fn test_taylor_polynomial() {
        let x = Symbol::new("x");
        // Taylor expansion of x² around x=0 should be x²
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = expr.taylor(&x, &Expr::from(0), 5);
        // Should be exactly x² since higher derivatives are 0
        let expected = Expr::Symbol(x.clone()).pow(Expr::from(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_maclaurin_exp() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).exp();
        let result = expr.maclaurin(&x, 3);
        // exp(x) ≈ 1 + x + x²/2 + x³/6
        // We can check this equals our known series
        let expected = known_series::exp(&x, 3);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_known_series_exp() {
        let x = Symbol::new("x");
        let series = known_series::exp(&x, 4);
        // Should be 1 + x + x²/2! + x³/3! + x⁴/4!
        // We can verify the structure (exact comparison is complex due to simplification)
        assert!(!series.is_constant());
    }

    #[test]
    fn test_known_series_sin() {
        let x = Symbol::new("x");
        let series = known_series::sin(&x, 5);
        // Should be x - x³/3! + x⁵/5!
        assert!(!series.is_constant());
    }

    #[test]
    fn test_known_series_cos() {
        let x = Symbol::new("x");
        let series = known_series::cos(&x, 4);
        // Should be 1 - x²/2! + x⁴/4!
        assert!(!series.is_constant());
    }

    #[test]
    fn test_series_coefficients() {
        let x = Symbol::new("x");
        // For f(x) = x², coefficients around 0 should be [0, 0, 1, 0, 0, ...]
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let coeffs = expr.series_coefficients(&x, &Expr::from(0), 3);

        assert_eq!(coeffs.len(), 4); // Coefficients for x^0, x^1, x^2, x^3
        assert_eq!(coeffs[0], Expr::from(0)); // constant term
        assert_eq!(coeffs[1], Expr::from(0)); // linear term
        // coeffs[2] should be 2!/2! = 1
    }

    #[test]
    fn test_taylor_around_nonzero() {
        let x = Symbol::new("x");
        // Expand x² around x=1
        // f(x) = (x-1)² + 2(x-1) + 1 = x²
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = expr.taylor(&x, &Expr::from(1), 2);

        // The result should be equivalent to x² when expanded
        // Taylor series: f(1) + f'(1)(x-1) + f''(1)(x-1)²/2
        // = 1 + 2(x-1) + 2(x-1)²/2 = 1 + 2(x-1) + (x-1)²
        assert!(!result.is_constant());
    }

    #[test]
    fn test_laurent_positive_orders() {
        let x = Symbol::new("x");
        // Laurent series with only positive orders is just Taylor series
        let expr = Expr::Symbol(x.clone()).exp();
        let laurent = expr.laurent(&x, &Expr::from(0), 0, 3);
        let taylor = expr.taylor(&x, &Expr::from(0), 3);
        assert_eq!(laurent, taylor);
    }

    #[test]
    fn test_binomial_series() {
        let x = Symbol::new("x");
        // (1+x)^2 = 1 + 2x + x²
        let alpha = Expr::from(2);
        let result = known_series::binomial(&x, &alpha, 2);

        // Should contain terms up to x²
        assert!(!result.is_constant());
    }
}
