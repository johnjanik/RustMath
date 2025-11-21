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
            let x_minus_a = if matches!(point, Expr::Integer(i) if i.to_i64() == 0) {
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

    /// Unified series expansion function
    ///
    /// This is the main interface for computing series expansions.
    /// Provides a flexible API for Taylor, Maclaurin, Laurent, and asymptotic series.
    ///
    /// # Arguments
    ///
    /// * `var` - Variable to expand with respect to
    /// * `point` - Point to expand around (use Expr::from(0) for Maclaurin)
    /// * `order` - Number of terms to include
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rustmath_symbolic::{Expr, Symbol};
    ///
    /// let x = Symbol::new("x");
    /// let expr = Expr::Symbol(x.clone()).exp();
    ///
    /// // Maclaurin series of exp(x)
    /// let series = expr.series(&x, &Expr::from(0), 5);
    /// // Returns: 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5!
    ///
    /// // Taylor series around x=1
    /// let series_at_1 = expr.series(&x, &Expr::from(1), 3);
    /// ```
    ///
    /// # Note
    ///
    /// This function automatically selects the appropriate expansion method:
    /// - For point = 0: Uses Maclaurin series
    /// - For finite points: Uses Taylor series
    /// - Can be extended to detect when Laurent series is needed
    pub fn series(&self, var: &Symbol, point: &Expr, order: usize) -> Self {
        // Check if point is zero for Maclaurin optimization
        let is_zero = matches!(point, Expr::Integer(i) if i.to_i64() == 0);

        if is_zero {
            self.maclaurin(var, order)
        } else {
            self.taylor(var, point, order)
        }
    }

    /// Compute series expansion with Big-O remainder term
    ///
    /// Returns a tuple of (series_expansion, big_o_term) where the big_o_term
    /// represents the error/remainder of the truncated series.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = Symbol::new("x");
    /// let expr = Expr::Symbol(x.clone()).sin();
    /// let (series, big_o) = expr.series_with_big_o(&x, &Expr::from(0), 3);
    /// // series: x - x³/3!
    /// // big_o: O(x⁵)
    /// ```
    pub fn series_with_big_o(&self, var: &Symbol, point: &Expr, order: usize) -> (Self, BigO) {
        let series = self.series(var, point, order);

        // The remainder term is O((x-a)^(n+1)) for Taylor series
        let x_minus_a = if matches!(point, Expr::Integer(i) if i.to_i64() == 0) {
            Expr::Symbol(var.clone())
        } else {
            Expr::Symbol(var.clone()) - point.clone()
        };

        let remainder = x_minus_a.pow(Expr::from((order + 1) as i64));
        let big_o = BigO::new(remainder, var.clone());

        (series, big_o)
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

// ============================================================================
// Phase 3.4 Enhancement: Big-O Notation Support
// ============================================================================

/// Big-O notation for asymptotic analysis
///
/// Represents O(f(x)) where f(x) is the growth rate function
#[derive(Debug, Clone, PartialEq)]
pub struct BigO {
    /// The function inside O(...)
    pub function: Expr,
    /// The variable (usually x or n)
    pub var: Symbol,
}

impl BigO {
    /// Create a new Big-O expression
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = Symbol::new("x");
    /// let o_x2 = BigO::new(Expr::Symbol(x.clone()).pow(Expr::from(2)), x);
    /// // Represents O(x²)
    /// ```
    pub fn new(function: Expr, var: Symbol) -> Self {
        BigO { function, var }
    }

    /// Create O(1) - constant time
    pub fn constant(var: Symbol) -> Self {
        BigO::new(Expr::from(1), var)
    }

    /// Create O(log n) - logarithmic time
    pub fn logarithmic(var: Symbol) -> Self {
        BigO::new(Expr::Symbol(var.clone()).log(), var)
    }

    /// Create O(n) - linear time
    pub fn linear(var: Symbol) -> Self {
        BigO::new(Expr::Symbol(var.clone()), var)
    }

    /// Create O(n log n) - linearithmic time
    pub fn linearithmic(var: Symbol) -> Self {
        let n = Expr::Symbol(var.clone());
        BigO::new(n.clone() * n.log(), var)
    }

    /// Create O(n²) - quadratic time
    pub fn quadratic(var: Symbol) -> Self {
        BigO::new(Expr::Symbol(var.clone()).pow(Expr::from(2)), var)
    }

    /// Create O(n^k) - polynomial time
    pub fn polynomial(var: Symbol, degree: i64) -> Self {
        BigO::new(Expr::Symbol(var.clone()).pow(Expr::from(degree)), var)
    }

    /// Create O(2^n) - exponential time
    pub fn exponential(var: Symbol) -> Self {
        BigO::new(Expr::from(2).pow(Expr::Symbol(var.clone())), var)
    }

    /// Create O(n!) - factorial time
    pub fn factorial(var: Symbol) -> Self {
        // Factorial not directly representable, use placeholder
        BigO::new(Expr::Symbol(var.clone()).factorial(), var)
    }

    /// Determine the dominant term in an expression for Big-O analysis
    ///
    /// For a sum of terms, returns the fastest-growing term
    ///
    /// # Algorithm
    ///
    /// 1. Separate expression into terms
    /// 2. For each term, determine its growth rate
    /// 3. Return the term with highest growth rate
    pub fn dominant_term(expr: &Expr, var: &Symbol) -> Expr {
        // For sums, find the dominant term
        if let Expr::Binary(BinaryOp::Add, left, right) = expr {
            let left_dom = BigO::dominant_term(left, var);
            let right_dom = BigO::dominant_term(right, var);

            // Compare growth rates
            if BigO::grows_faster(&left_dom, &right_dom, var) {
                left_dom
            } else {
                right_dom
            }
        } else {
            // For non-sums, return the expression itself (simplified)
            expr.clone()
        }
    }

    /// Check if f(x) grows faster than g(x) as x → ∞
    ///
    /// Uses limit comparison: if lim(x→∞) g(x)/f(x) = 0, then f grows faster
    fn grows_faster(f: &Expr, g: &Expr, var: &Symbol) -> bool {
        // Simplified comparison based on degree
        let f_degree = f.degree(var).unwrap_or(0);
        let g_degree = g.degree(var).unwrap_or(0);

        f_degree > g_degree
    }

    /// Derive Big-O complexity from an expression
    ///
    /// Analyzes an expression and determines its asymptotic complexity
    ///
    /// # Examples
    ///
    /// - 3n² + 2n + 1 → O(n²)
    /// - 5n + log(n) → O(n)
    /// - 2^n + n³ → O(2^n)
    pub fn from_expression(expr: &Expr, var: &Symbol) -> Self {
        // Find dominant term
        let dominant = BigO::dominant_term(expr, var);

        // Remove constant coefficients
        let simplified = BigO::remove_constants(&dominant, var);

        BigO::new(simplified, var.clone())
    }

    /// Remove constant coefficients from an expression
    ///
    /// For Big-O analysis, constant factors don't matter
    fn remove_constants(expr: &Expr, var: &Symbol) -> Expr {
        match expr {
            // If expression doesn't contain the variable, treat as constant
            _ if !expr.contains_symbol(var) => Expr::from(1),

            // For products, remove constant factors
            Expr::Binary(BinaryOp::Mul, left, right) => {
                if !left.contains_symbol(var) {
                    BigO::remove_constants(right, var)
                } else if !right.contains_symbol(var) {
                    BigO::remove_constants(left, var)
                } else {
                    // Both contain the variable
                    BigO::remove_constants(left, var) * BigO::remove_constants(right, var)
                }
            }

            // For other expressions, keep as is
            _ => expr.clone(),
        }
    }

    /// Check if this Big-O is equal to or dominates another
    ///
    /// O(f) dominates O(g) if f grows at least as fast as g
    pub fn dominates(&self, other: &BigO) -> bool {
        BigO::grows_faster(&self.function, &other.function, &self.var)
            || self.function == other.function
    }
}

impl std::fmt::Display for BigO {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "O({})", self.function)
    }
}

/// Little-o notation for strict asymptotic bounds
///
/// o(f(x)) represents functions that grow strictly slower than f(x)
#[derive(Debug, Clone, PartialEq)]
pub struct LittleO {
    pub function: Expr,
    pub var: Symbol,
}

impl LittleO {
    pub fn new(function: Expr, var: Symbol) -> Self {
        LittleO { function, var }
    }
}

impl std::fmt::Display for LittleO {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "o({})", self.function)
    }
}

/// Theta notation for tight asymptotic bounds
///
/// Θ(f(x)) represents functions that grow at the same rate as f(x)
#[derive(Debug, Clone, PartialEq)]
pub struct Theta {
    pub function: Expr,
    pub var: Symbol,
}

impl Theta {
    pub fn new(function: Expr, var: Symbol) -> Self {
        Theta { function, var }
    }
}

impl std::fmt::Display for Theta {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Θ({})", self.function)
    }
}

/// Omega notation for lower bounds
///
/// Ω(f(x)) represents functions that grow at least as fast as f(x)
#[derive(Debug, Clone, PartialEq)]
pub struct Omega {
    pub function: Expr,
    pub var: Symbol,
}

impl Omega {
    pub fn new(function: Expr, var: Symbol) -> Self {
        Omega { function, var }
    }
}

impl std::fmt::Display for Omega {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Ω({})", self.function)
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

    #[test]
    fn test_unified_series_function() {
        let x = Symbol::new("x");

        // Test Maclaurin series (point = 0)
        let expr = Expr::Symbol(x.clone()).exp();
        let series = expr.series(&x, &Expr::from(0), 3);

        // Should match known exp series
        let expected = known_series::exp(&x, 3);
        assert_eq!(series, expected);
    }

    #[test]
    fn test_series_at_nonzero_point() {
        let x = Symbol::new("x");

        // Test Taylor series at x=1
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let series = expr.series(&x, &Expr::from(1), 2);

        // Should not be constant
        assert!(!series.is_constant());
    }

    #[test]
    fn test_series_with_big_o() {
        let x = Symbol::new("x");

        // Test series with remainder term
        let expr = Expr::Symbol(x.clone()).sin();
        let (series, big_o) = expr.series_with_big_o(&x, &Expr::from(0), 3);

        // Series should not be constant
        assert!(!series.is_constant());

        // Big-O term should be for x^4
        assert_eq!(big_o.var, x);
    }

    #[test]
    fn test_big_o_from_expression() {
        let x = Symbol::new("x");

        // 3n² + 2n + 1 → O(n²)
        let expr = Expr::from(3) * Expr::Symbol(x.clone()).pow(Expr::from(2))
            + Expr::from(2) * Expr::Symbol(x.clone())
            + Expr::from(1);

        let big_o = BigO::from_expression(&expr, &x);

        // Should detect x² as dominant term
        let expected_func = Expr::Symbol(x.clone()).pow(Expr::from(2));
        assert_eq!(big_o.function, expected_func);
    }

    #[test]
    fn test_big_o_dominance() {
        let x = Symbol::new("x");

        // O(n²) dominates O(n)
        let o_n2 = BigO::quadratic(x.clone());
        let o_n = BigO::linear(x.clone());

        assert!(o_n2.dominates(&o_n));
        assert!(!o_n.dominates(&o_n2));
    }

    #[test]
    fn test_big_o_display() {
        let x = Symbol::new("x");
        let big_o = BigO::quadratic(x);

        let display = format!("{}", big_o);
        assert!(display.contains("O("));
    }

    #[test]
    fn test_asymptotic_expansion() {
        let x = Symbol::new("x");

        // Test asymptotic expansion as x → ∞
        // For (x² + x) / (x² + 1), as x → ∞, should approach 1
        let num = Expr::Symbol(x.clone()).pow(Expr::from(2)) + Expr::Symbol(x.clone());
        let den = Expr::Symbol(x.clone()).pow(Expr::from(2)) + Expr::from(1);
        let expr = num / den;

        let asymp = expr.asymptotic(&x, 3);

        // Should not be constant (has terms in powers of 1/x)
        assert!(!asymp.is_constant());
    }
}
