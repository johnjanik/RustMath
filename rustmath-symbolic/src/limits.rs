//! Symbolic limits
//!
//! This module implements limit computation for symbolic expressions.
//! Includes both two-sided limits and one-sided limits (left/right).

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_core::{Ring, NumericConversion};
use std::sync::Arc;

/// Direction for one-sided limits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Limit from the left (x → a⁻)
    Left,
    /// Limit from the right (x → a⁺)
    Right,
    /// Two-sided limit
    Both,
}

/// Result of limit computation
#[derive(Debug, Clone, PartialEq)]
pub enum LimitResult {
    /// Finite limit value
    Finite(Expr),
    /// Positive infinity
    Infinity,
    /// Negative infinity
    NegInfinity,
    /// Limit does not exist
    DoesNotExist,
    /// Unable to determine
    Unknown,
}

impl Expr {
    /// Compute the limit of an expression as var approaches point
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // lim(x→0) (sin(x)/x) = 1
    /// let x = Symbol::new("x");
    /// let expr = Expr::Symbol(x.clone()).sin() / Expr::Symbol(x.clone());
    /// let limit = expr.limit(&x, &Expr::from(0), Direction::Both);
    /// ```
    pub fn limit(&self, var: &Symbol, point: &Expr, dir: Direction) -> LimitResult {
        // First try direct substitution
        let substituted = self.substitute(var, point);

        // If we get a finite value (no division by zero, etc.), return it
        if substituted.is_finite_value() {
            return LimitResult::Finite(substituted);
        }

        // Check for common indeterminate forms and apply limit laws
        self.limit_advanced(var, point, dir)
    }

    /// Advanced limit computation for indeterminate forms
    fn limit_advanced(&self, var: &Symbol, point: &Expr, dir: Direction) -> LimitResult {
        match self {
            Expr::Integer(_) | Expr::Rational(_) => {
                LimitResult::Finite(self.clone())
            }

            Expr::Symbol(s) => {
                if s == var {
                    LimitResult::Finite(point.clone())
                } else {
                    LimitResult::Finite(self.clone())
                }
            }

            Expr::Binary(op, left, right) => match op {
                BinaryOp::Add => {
                    let left_limit = left.limit(var, point, dir);
                    let right_limit = right.limit(var, point, dir);
                    Self::combine_limits_add(left_limit, right_limit)
                }

                BinaryOp::Sub => {
                    let left_limit = left.limit(var, point, dir);
                    let right_limit = right.limit(var, point, dir);
                    Self::combine_limits_sub(left_limit, right_limit)
                }

                BinaryOp::Mul => {
                    let left_limit = left.limit(var, point, dir);
                    let right_limit = right.limit(var, point, dir);
                    Self::combine_limits_mul(left_limit, right_limit)
                }

                BinaryOp::Div => {
                    let num_limit = left.limit(var, point, dir);
                    let den_limit = right.limit(var, point, dir);

                    // Check for 0/0 or ∞/∞ indeterminate forms
                    match (&num_limit, &den_limit) {
                        (LimitResult::Finite(n), LimitResult::Finite(d)) if d.is_zero() => {
                            // Apply L'Hôpital's rule if numerator is also 0
                            if n.is_zero() {
                                self.lhopital(var, point, dir)
                            } else if n.is_positive_value() {
                                match dir {
                                    Direction::Left => {
                                        if d.is_positive_value() {
                                            LimitResult::Infinity
                                        } else {
                                            LimitResult::NegInfinity
                                        }
                                    }
                                    Direction::Right => {
                                        if d.is_positive_value() {
                                            LimitResult::Infinity
                                        } else {
                                            LimitResult::NegInfinity
                                        }
                                    }
                                    Direction::Both => LimitResult::DoesNotExist,
                                }
                            } else {
                                LimitResult::Unknown
                            }
                        }
                        _ => Self::combine_limits_div(num_limit, den_limit),
                    }
                }

                BinaryOp::Pow => {
                    let base_limit = left.limit(var, point, dir);
                    let exp_limit = right.limit(var, point, dir);
                    Self::combine_limits_pow(base_limit, exp_limit)
                }
            },

            Expr::Unary(op, inner) => {
                let inner_limit = inner.limit(var, point, dir);
                Self::apply_function_limit(*op, inner_limit)
            }

            Expr::Function(_, _) => LimitResult::Unknown,
        }
    }

    /// Apply L'Hôpital's rule for 0/0 indeterminate form
    fn lhopital(&self, var: &Symbol, point: &Expr, dir: Direction) -> LimitResult {
        if let Expr::Binary(BinaryOp::Div, num, den) = self {
            // Take derivatives
            let num_prime = num.differentiate(var);
            let den_prime = den.differentiate(var);

            // Compute limit of ratio of derivatives
            let new_expr = num_prime / den_prime;
            new_expr.limit(var, point, dir)
        } else {
            LimitResult::Unknown
        }
    }

    /// Combine limits for addition
    fn combine_limits_add(left: LimitResult, right: LimitResult) -> LimitResult {
        match (left, right) {
            (LimitResult::Finite(l), LimitResult::Finite(r)) => {
                LimitResult::Finite(l + r)
            }
            (LimitResult::Infinity, LimitResult::Infinity) => LimitResult::Infinity,
            (LimitResult::NegInfinity, LimitResult::NegInfinity) => LimitResult::NegInfinity,
            (LimitResult::Infinity, LimitResult::NegInfinity) => LimitResult::DoesNotExist,
            (LimitResult::NegInfinity, LimitResult::Infinity) => LimitResult::DoesNotExist,
            (LimitResult::Infinity, LimitResult::Finite(_)) => LimitResult::Infinity,
            (LimitResult::Finite(_), LimitResult::Infinity) => LimitResult::Infinity,
            (LimitResult::NegInfinity, LimitResult::Finite(_)) => LimitResult::NegInfinity,
            (LimitResult::Finite(_), LimitResult::NegInfinity) => LimitResult::NegInfinity,
            _ => LimitResult::Unknown,
        }
    }

    /// Combine limits for subtraction
    fn combine_limits_sub(left: LimitResult, right: LimitResult) -> LimitResult {
        match (left, right) {
            (LimitResult::Finite(l), LimitResult::Finite(r)) => {
                LimitResult::Finite(l - r)
            }
            (LimitResult::Infinity, LimitResult::Finite(_)) => LimitResult::Infinity,
            (LimitResult::Finite(_), LimitResult::NegInfinity) => LimitResult::Infinity,
            (LimitResult::NegInfinity, LimitResult::Finite(_)) => LimitResult::NegInfinity,
            (LimitResult::Finite(_), LimitResult::Infinity) => LimitResult::NegInfinity,
            (LimitResult::Infinity, LimitResult::Infinity) => LimitResult::DoesNotExist,
            (LimitResult::NegInfinity, LimitResult::NegInfinity) => LimitResult::DoesNotExist,
            _ => LimitResult::Unknown,
        }
    }

    /// Combine limits for multiplication
    fn combine_limits_mul(left: LimitResult, right: LimitResult) -> LimitResult {
        match (left, right) {
            (LimitResult::Finite(l), LimitResult::Finite(r)) => {
                LimitResult::Finite(l * r)
            }
            (LimitResult::Infinity, LimitResult::Finite(f)) if f.is_positive_value() => {
                LimitResult::Infinity
            }
            (LimitResult::Infinity, LimitResult::Finite(f)) if f.is_negative_value() => {
                LimitResult::NegInfinity
            }
            (LimitResult::Finite(f), LimitResult::Infinity) if f.is_positive_value() => {
                LimitResult::Infinity
            }
            (LimitResult::Finite(f), LimitResult::Infinity) if f.is_negative_value() => {
                LimitResult::NegInfinity
            }
            (LimitResult::Infinity, LimitResult::Infinity) => LimitResult::Infinity,
            (LimitResult::NegInfinity, LimitResult::NegInfinity) => LimitResult::Infinity,
            (LimitResult::Infinity, LimitResult::NegInfinity) => LimitResult::NegInfinity,
            (LimitResult::NegInfinity, LimitResult::Infinity) => LimitResult::NegInfinity,
            _ => LimitResult::Unknown,
        }
    }

    /// Combine limits for division
    fn combine_limits_div(num: LimitResult, den: LimitResult) -> LimitResult {
        match (num, den) {
            (LimitResult::Finite(n), LimitResult::Finite(d)) if !d.is_zero() => {
                LimitResult::Finite(n / d)
            }
            (LimitResult::Finite(_), LimitResult::Infinity) => LimitResult::Finite(Expr::from(0)),
            (LimitResult::Finite(_), LimitResult::NegInfinity) => {
                LimitResult::Finite(Expr::from(0))
            }
            (LimitResult::Infinity, LimitResult::Finite(f)) if f.is_positive_value() => {
                LimitResult::Infinity
            }
            (LimitResult::Infinity, LimitResult::Finite(f)) if f.is_negative_value() => {
                LimitResult::NegInfinity
            }
            _ => LimitResult::Unknown,
        }
    }

    /// Combine limits for exponentiation
    fn combine_limits_pow(base: LimitResult, exp: LimitResult) -> LimitResult {
        match (base, exp) {
            (LimitResult::Finite(b), LimitResult::Finite(e)) => {
                LimitResult::Finite(b.pow(e))
            }
            (LimitResult::Infinity, LimitResult::Finite(e)) if e.is_positive_value() => {
                LimitResult::Infinity
            }
            (LimitResult::Infinity, LimitResult::Finite(e)) if e.is_negative_value() => {
                LimitResult::Finite(Expr::from(0))
            }
            _ => LimitResult::Unknown,
        }
    }

    /// Apply function to limit
    fn apply_function_limit(op: UnaryOp, inner: LimitResult) -> LimitResult {
        match inner {
            LimitResult::Finite(val) => {
                let result = match op {
                    UnaryOp::Neg => -val,
                    UnaryOp::Sin => val.sin(),
                    UnaryOp::Cos => val.cos(),
                    UnaryOp::Tan => val.tan(),
                    UnaryOp::Exp => val.exp(),
                    UnaryOp::Log => val.log(),
                    UnaryOp::Sqrt => val.sqrt(),
                    UnaryOp::Abs => val.abs(),
                    UnaryOp::Sinh => val.sinh(),
                    UnaryOp::Cosh => val.cosh(),
                    UnaryOp::Tanh => val.tanh(),
                    UnaryOp::Arcsin => val.arcsin(),
                    UnaryOp::Arccos => val.arccos(),
                    UnaryOp::Arctan => val.arctan(),
                    _ => return LimitResult::Unknown,
                };
                LimitResult::Finite(result)
            }
            LimitResult::Infinity => match op {
                UnaryOp::Exp => LimitResult::Infinity,
                UnaryOp::Log => LimitResult::Infinity,
                UnaryOp::Sin | UnaryOp::Cos => LimitResult::DoesNotExist,
                _ => LimitResult::Unknown,
            },
            _ => LimitResult::Unknown,
        }
    }

    /// Check if expression is a finite numerical value
    fn is_finite_value(&self) -> bool {
        match self {
            Expr::Integer(_) | Expr::Rational(_) => true,
            _ => false,
        }
    }

    /// Check if expression is zero
    fn is_zero(&self) -> bool {
        matches!(self, Expr::Integer(i) if i.to_i64() == Some(0))
    }

    /// Check if expression is a positive numerical value
    fn is_positive_value(&self) -> bool {
        match self {
            Expr::Integer(i) => i.to_i64().map(|v| v > 0).unwrap_or(false),
            Expr::Rational(r) => r.to_f64().map(|v| v > 0.0).unwrap_or(false),
            _ => false,
        }
    }

    /// Check if expression is a negative numerical value
    fn is_negative_value(&self) -> bool {
        match self {
            Expr::Integer(i) => i.to_i64().map(|v| v < 0).unwrap_or(false),
            Expr::Rational(r) => r.to_f64().map(|v| v < 0.0).unwrap_or(false),
            _ => false,
        }
    }

    // ========================================================================
    // Phase 3.3 Enhancements: Improved Limit Computation
    // ========================================================================

    /// Apply L'Hôpital's rule multiple times
    ///
    /// Continues applying L'Hôpital's rule until either:
    /// - A determinate limit is found
    /// - Maximum iterations reached
    /// - An indeterminate form other than 0/0 or ∞/∞ is encountered
    ///
    /// # Arguments
    ///
    /// * `var` - The variable approaching the limit
    /// * `point` - The point being approached
    /// * `dir` - Direction of approach
    /// * `max_iterations` - Maximum number of L'Hôpital applications (default 5)
    pub fn lhopital_repeated(
        &self,
        var: &Symbol,
        point: &Expr,
        dir: Direction,
        max_iterations: usize,
    ) -> LimitResult {
        let mut expr = self.clone();
        let mut iterations = 0;

        while iterations < max_iterations {
            // Try direct substitution
            let substituted = expr.substitute(var, point);
            if substituted.is_finite_value() {
                return LimitResult::Finite(substituted);
            }

            // Check if we have a 0/0 or ∞/∞ indeterminate form
            if let Expr::Binary(BinaryOp::Div, num, den) = &expr {
                let num_limit = num.limit(var, point, dir);
                let den_limit = den.limit(var, point, dir);

                // Check for 0/0
                let is_zero_over_zero = matches!(
                    (&num_limit, &den_limit),
                    (LimitResult::Finite(n), LimitResult::Finite(d))
                    if n.is_zero() && d.is_zero()
                );

                // Check for ∞/∞
                let is_inf_over_inf = matches!(
                    (&num_limit, &den_limit),
                    (LimitResult::Infinity, LimitResult::Infinity)
                        | (LimitResult::NegInfinity, LimitResult::NegInfinity)
                        | (LimitResult::Infinity, LimitResult::NegInfinity)
                        | (LimitResult::NegInfinity, LimitResult::Infinity)
                );

                if is_zero_over_zero || is_inf_over_inf {
                    // Apply L'Hôpital: differentiate numerator and denominator
                    let num_prime = num.differentiate(var);
                    let den_prime = den.differentiate(var);

                    expr = num_prime / den_prime;
                    iterations += 1;
                    continue;
                }
            }

            // If not 0/0 or ∞/∞, compute the limit normally
            return expr.limit(var, point, dir);
        }

        // Reached max iterations without resolving
        LimitResult::Unknown
    }

    /// Compute limit using series expansion
    ///
    /// For limits at a point, expand the function as a Taylor series
    /// around that point and determine the leading behavior.
    ///
    /// # Algorithm
    ///
    /// 1. Expand f(x) as Taylor series around the point
    /// 2. Identify the lowest-order non-zero term
    /// 3. The limit is the value of that term
    ///
    /// # Arguments
    ///
    /// * `var` - The variable approaching the limit
    /// * `point` - The point being approached
    /// * `order` - Order of Taylor expansion (default 5)
    pub fn limit_via_series(&self, var: &Symbol, point: &Expr, order: usize) -> LimitResult {
        // Compute Taylor series expansion
        let series = self.taylor(var, point, order);

        // Try to evaluate the series at the point
        // The constant term gives the limit
        let result = series.substitute(var, point);

        if result.is_finite_value() {
            LimitResult::Finite(result)
        } else {
            // Series expansion didn't help, return unknown
            LimitResult::Unknown
        }
    }

    /// Compute limit at infinity using asymptotic expansion
    ///
    /// For limits as x → ∞, substitutes x = 1/t and computes lim(t→0+)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // lim(x→∞) (2x + 1)/(x + 3) = 2
    /// // Substitute x = 1/t: lim(t→0+) (2/t + 1)/(1/t + 3) = lim(t→0+) (2 + t)/(1 + 3t) = 2
    /// ```
    pub fn limit_at_infinity(&self, var: &Symbol, dir: Direction) -> LimitResult {
        // Create a new variable t
        let t = Symbol::new("t_limit_inf");
        let t_expr = Expr::Symbol(t.clone());

        // Substitute x = 1/t
        let substituted = self.substitute(var, &(Expr::from(1) / t_expr.clone()));

        // Compute limit as t → 0+
        let limit_dir = match dir {
            Direction::Right => Direction::Right, // x→+∞ means t→0+
            Direction::Left => Direction::Left,   // x→-∞ means t→0-
            Direction::Both => Direction::Right,  // Default to +∞
        };

        substituted.limit(&t, &Expr::from(0), limit_dir)
    }

    /// Enhanced limit computation with all available techniques
    ///
    /// Tries multiple strategies in order:
    /// 1. Direct substitution
    /// 2. Algebraic simplification
    /// 3. L'Hôpital's rule (repeated if necessary)
    /// 4. Series expansion
    /// 5. For infinity limits: asymptotic analysis
    pub fn limit_advanced(&self, var: &Symbol, point: &Expr, dir: Direction) -> LimitResult {
        // Strategy 1: Direct substitution
        let substituted = self.substitute(var, point);
        if substituted.is_finite_value() {
            return LimitResult::Finite(substituted);
        }

        // Strategy 2: Simplify first, then try direct substitution
        let simplified = self.simplify();
        let sub_simplified = simplified.substitute(var, point);
        if sub_simplified.is_finite_value() {
            return LimitResult::Finite(sub_simplified);
        }

        // Strategy 3: Try repeated L'Hôpital for indeterminate forms
        if matches!(self, Expr::Binary(BinaryOp::Div, _, _)) {
            let lhopital_result = self.lhopital_repeated(var, point, dir, 5);
            if !matches!(lhopital_result, LimitResult::Unknown) {
                return lhopital_result;
            }
        }

        // Strategy 4: Try series expansion (for finite points)
        if point.is_finite_value() {
            let series_result = self.limit_via_series(var, point, 5);
            if !matches!(series_result, LimitResult::Unknown) {
                return series_result;
            }
        }

        // Strategy 5: For infinity limits, use asymptotic expansion
        // (This would check if point represents infinity, but we don't have
        // an infinity type yet, so we skip this for now)

        // All strategies failed
        LimitResult::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limit_constant() {
        let x = Symbol::new("x");
        let expr = Expr::from(5);
        let result = expr.limit(&x, &Expr::from(0), Direction::Both);
        assert_eq!(result, LimitResult::Finite(Expr::from(5)));
    }

    #[test]
    fn test_limit_variable() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone());
        let result = expr.limit(&x, &Expr::from(2), Direction::Both);
        assert_eq!(result, LimitResult::Finite(Expr::from(2)));
    }

    #[test]
    fn test_limit_polynomial() {
        let x = Symbol::new("x");
        // lim(x→2) (x² + 1) = 5
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2)) + Expr::from(1);
        let result = expr.limit(&x, &Expr::from(2), Direction::Both);
        assert_eq!(result, LimitResult::Finite(Expr::from(5)));
    }

    #[test]
    fn test_limit_lhopital_simple() {
        let x = Symbol::new("x");
        // lim(x→0) x/x = 1 (using L'Hôpital's rule)
        let expr = Expr::Symbol(x.clone()) / Expr::Symbol(x.clone());
        let result = expr.limit(&x, &Expr::from(0), Direction::Both);
        // After L'Hôpital: lim(x→0) 1/1 = 1
        assert_eq!(result, LimitResult::Finite(Expr::from(1)));
    }

    #[test]
    fn test_limit_at_infinity_reciprocal() {
        // This test is conceptual - we'd need to handle infinity as a point
        // For now, we can test that division by large numbers approaches 0
        let x = Symbol::new("x");
        let expr = Expr::from(1) / Expr::Symbol(x.clone());
        // As x → ∞, 1/x → 0
        // We can test with a large number
        let result = expr.substitute(&x, &Expr::from(1000000));
        // Result should be very small
        assert!(matches!(result, Expr::Rational(_)));
    }

    #[test]
    fn test_limit_sum() {
        let x = Symbol::new("x");
        // lim(x→1) (x + 2) = 3
        let expr = Expr::Symbol(x.clone()) + Expr::from(2);
        let result = expr.limit(&x, &Expr::from(1), Direction::Both);
        assert_eq!(result, LimitResult::Finite(Expr::from(3)));
    }

    #[test]
    fn test_limit_product() {
        let x = Symbol::new("x");
        // lim(x→2) (3x) = 6
        let expr = Expr::from(3) * Expr::Symbol(x.clone());
        let result = expr.limit(&x, &Expr::from(2), Direction::Both);
        assert_eq!(result, LimitResult::Finite(Expr::from(6)));
    }
}
