//! Polynomial operations on symbolic expressions

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_integers::Integer;
use std::sync::Arc;

impl Expr {
    /// Check if the expression is a polynomial in the given variable
    ///
    /// A polynomial in x is an expression that can be written as a sum of terms
    /// of the form c*x^n where c is a constant (free of x) and n is a non-negative integer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    /// use rustmath_symbolic::Symbol;
    ///
    /// let x = Expr::symbol("x");
    /// let y = Expr::symbol("y");
    ///
    /// // x^2 + 2*x + 1 is a polynomial in x
    /// let p = x.clone().pow(Expr::from(2)) + Expr::from(2) * x.clone() + Expr::from(1);
    /// assert!(p.is_polynomial(&Symbol::new("x")));
    ///
    /// // sin(x) is not a polynomial in x
    /// let s = x.clone().sin();
    /// assert!(!s.is_polynomial(&Symbol::new("x")));
    /// ```
    pub fn is_polynomial(&self, var: &Symbol) -> bool {
        self.is_polynomial_recursive(var, true)
    }

    fn is_polynomial_recursive(&self, var: &Symbol, allow_var: bool) -> bool {
        match self {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => true,
            Expr::Symbol(s) => {
                if s == var {
                    allow_var
                } else {
                    true // Other symbols are treated as constants
                }
            }
            Expr::Binary(op, left, right) => match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                    left.is_polynomial_recursive(var, allow_var)
                        && right.is_polynomial_recursive(var, allow_var)
                }
                BinaryOp::Mod => {
                    // Modulo is not a polynomial operation
                    false
                }
                BinaryOp::Div => {
                    // Division is only allowed if divisor is constant in var
                    left.is_polynomial_recursive(var, allow_var)
                        && right.is_polynomial_recursive(var, false)
                }
                BinaryOp::Pow => {
                    // For polynomials, exponent must be a non-negative integer constant
                    if let Expr::Integer(exp) = right.as_ref() {
                        if exp >= &Integer::zero() {
                            return left.is_polynomial_recursive(var, allow_var);
                        }
                    }
                    // Also check if both base and exponent are constant in var
                    left.is_polynomial_recursive(var, false)
                        && right.is_polynomial_recursive(var, false)
                }
            },
            Expr::Unary(op, inner) => match op {
                UnaryOp::Neg => inner.is_polynomial_recursive(var, allow_var),
                // Transcendental functions are only polynomial if applied to constants
                _ => inner.is_polynomial_recursive(var, false),
            },
            // Functions are polynomial only if all arguments are constant in var
            Expr::Function(_, args) => {
                args.iter().all(|arg| arg.is_polynomial_recursive(var, false))
            }
        }
    }

    /// Get the degree of the polynomial in the given variable
    ///
    /// Returns None if the expression is not a polynomial in the variable,
    /// or Some(degree) where degree is the highest power of the variable.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    /// use rustmath_symbolic::Symbol;
    ///
    /// let x = Expr::symbol("x");
    ///
    /// // x^3 + 2*x + 1 has degree 3
    /// let p = x.clone().pow(Expr::from(3)) + Expr::from(2) * x.clone() + Expr::from(1);
    /// assert_eq!(p.degree(&Symbol::new("x")), Some(3));
    ///
    /// // Constant has degree 0
    /// assert_eq!(Expr::from(5).degree(&Symbol::new("x")), Some(0));
    /// ```
    pub fn degree(&self, var: &Symbol) -> Option<i64> {
        if !self.is_polynomial(var) {
            return None;
        }

        self.degree_recursive(var)
    }

    fn degree_recursive(&self, var: &Symbol) -> Option<i64> {
        match self {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => Some(0),
            Expr::Symbol(s) => {
                if s == var {
                    Some(1)
                } else {
                    Some(0) // Other symbols are constants
                }
            }
            Expr::Binary(op, left, right) => match op {
                BinaryOp::Add | BinaryOp::Sub => {
                    let left_deg = left.degree_recursive(var)?;
                    let right_deg = right.degree_recursive(var)?;
                    Some(left_deg.max(right_deg))
                }
                BinaryOp::Mul => {
                    let left_deg = left.degree_recursive(var)?;
                    let right_deg = right.degree_recursive(var)?;
                    Some(left_deg + right_deg)
                }
                BinaryOp::Mod => {
                    // Modulo is not a polynomial operation
                    None
                }
                BinaryOp::Div => {
                    // For polynomials, divisor must be constant
                    let left_deg = left.degree_recursive(var)?;
                    if right.is_polynomial_recursive(var, false) {
                        Some(left_deg)
                    } else {
                        None
                    }
                }
                BinaryOp::Pow => {
                    if let Expr::Integer(exp) = right.as_ref() {
                        let exp_i64 = exp.to_i64();
                        if exp_i64 >= 0 {
                            let base_deg = left.degree_recursive(var)?;
                            return Some(base_deg * exp_i64);
                        }
                    }
                    // Constant power
                    Some(0)
                }
            },
            Expr::Unary(op, inner) => match op {
                UnaryOp::Neg => inner.degree_recursive(var),
                _ => Some(0), // Transcendental functions applied to constants
            },
            // Functions are treated as constants (degree 0)
            Expr::Function(_, _) => Some(0),
        }
    }

    /// Get the coefficient of x^n in the polynomial
    ///
    /// Returns the coefficient as an expression (which should be constant in the variable).
    /// Returns None if the expression is not a polynomial or if extraction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    /// use rustmath_symbolic::Symbol;
    ///
    /// let x = Expr::symbol("x");
    ///
    /// // 3*x^2 + 2*x + 1
    /// let p = Expr::from(3) * x.clone().pow(Expr::from(2))
    ///       + Expr::from(2) * x.clone()
    ///       + Expr::from(1);
    ///
    /// // Coefficient of x^2 is 3
    /// let coeff = p.coefficient(&Symbol::new("x"), 2);
    /// assert!(coeff.is_some());
    /// ```
    pub fn coefficient(&self, var: &Symbol, n: i64) -> Option<Expr> {
        if !self.is_polynomial(var) {
            return None;
        }

        self.coefficient_recursive(var, n)
    }

    fn coefficient_recursive(&self, var: &Symbol, n: i64) -> Option<Expr> {
        if n < 0 {
            return Some(Expr::from(0));
        }

        match self {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => {
                if n == 0 {
                    Some(self.clone())
                } else {
                    Some(Expr::from(0))
                }
            }
            Expr::Symbol(s) => {
                if s == var {
                    if n == 1 {
                        Some(Expr::from(1))
                    } else {
                        Some(Expr::from(0))
                    }
                } else {
                    // Other symbols are constants
                    if n == 0 {
                        Some(self.clone())
                    } else {
                        Some(Expr::from(0))
                    }
                }
            }
            Expr::Binary(op, left, right) => match op {
                BinaryOp::Add => {
                    let left_coeff = left.coefficient_recursive(var, n)?;
                    let right_coeff = right.coefficient_recursive(var, n)?;
                    Some(left_coeff + right_coeff)
                }
                BinaryOp::Sub => {
                    let left_coeff = left.coefficient_recursive(var, n)?;
                    let right_coeff = right.coefficient_recursive(var, n)?;
                    Some(left_coeff - right_coeff)
                }
                BinaryOp::Mul => {
                    // (c1 * x^n1) * (c2 * x^n2) contributes to x^(n1+n2)
                    // We need to sum over all ways to partition n = d1 + d2
                    let mut result = Expr::from(0);
                    for d in 0..=n {
                        let left_coeff = left.coefficient_recursive(var, d)?;
                        let right_coeff = right.coefficient_recursive(var, n - d)?;
                        result = result + (left_coeff * right_coeff);
                    }
                    Some(result)
                }
                BinaryOp::Mod => {
                    // Modulo is not a polynomial operation
                    None
                }
                BinaryOp::Pow => {
                    if let Expr::Integer(exp) = right.as_ref() {
                        let exp_i64 = exp.to_i64();
                        if exp_i64 >= 0 {
                            // Simple case: if base doesn't contain var (is constant)
                            if left.is_polynomial_recursive(var, false) {
                                if n == 0 {
                                    return Some(self.clone());
                                } else {
                                    return Some(Expr::from(0));
                                }
                            }
                            // Simple case: if base is exactly the variable (like x^k)
                            if let Expr::Symbol(s) = left.as_ref() {
                                if s == var {
                                    if n == exp_i64 {
                                        return Some(Expr::from(1));
                                    } else {
                                        return Some(Expr::from(0));
                                    }
                                }
                            }
                            // TODO: Full binomial expansion for (a + b)^k
                        }
                    }
                    // Constant case
                    if n == 0 {
                        Some(self.clone())
                    } else {
                        Some(Expr::from(0))
                    }
                }
                BinaryOp::Div => {
                    // Divisor must be constant
                    let left_coeff = left.coefficient_recursive(var, n)?;
                    if right.is_polynomial_recursive(var, false) {
                        Some(Expr::Binary(
                            BinaryOp::Div,
                            Arc::new(left_coeff),
                            Arc::new(right.as_ref().clone()),
                        ))
                    } else {
                        None
                    }
                }
            },
            Expr::Unary(op, inner) => match op {
                UnaryOp::Neg => {
                    let inner_coeff = inner.coefficient_recursive(var, n)?;
                    Some(-inner_coeff)
                }
                _ => {
                    // Transcendental functions
                    if n == 0 {
                        Some(self.clone())
                    } else {
                        Some(Expr::from(0))
                    }
                }
            },
            // Functions are treated as constants
            Expr::Function(_, _) => {
                if n == 0 {
                    Some(self.clone())
                } else {
                    Some(Expr::from(0))
                }
            }
        }
    }

    /// Check if the expression is a rational expression
    ///
    /// A rational expression is a ratio of two polynomials.
    pub fn is_rational_expression(&self, var: &Symbol) -> bool {
        match self {
            Expr::Binary(BinaryOp::Div, num, den) => {
                num.is_polynomial(var) && den.is_polynomial(var)
            }
            _ => self.is_polynomial(var),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_polynomial() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let var_x = Symbol::new("x");

        // x^2 + 2*x + 1 is a polynomial in x
        let p1 = x.clone().pow(Expr::from(2)) + Expr::from(2) * x.clone() + Expr::from(1);
        assert!(p1.is_polynomial(&var_x));

        // x*y + y^2 is a polynomial in x (y is treated as constant)
        let p2 = x.clone() * y.clone() + y.clone().pow(Expr::from(2));
        assert!(p2.is_polynomial(&var_x));

        // sin(x) is not a polynomial in x
        let s = x.clone().sin();
        assert!(!s.is_polynomial(&var_x));

        // 1/x is not a polynomial in x
        let inv = Expr::from(1) / x.clone();
        assert!(!inv.is_polynomial(&var_x));

        // Constant is a polynomial
        assert!(Expr::from(5).is_polynomial(&var_x));
    }

    #[test]
    fn test_degree() {
        let x = Expr::symbol("x");
        let var_x = Symbol::new("x");

        // x^3 + 2*x + 1 has degree 3
        let p1 = x.clone().pow(Expr::from(3)) + Expr::from(2) * x.clone() + Expr::from(1);
        assert_eq!(p1.degree(&var_x), Some(3));

        // x has degree 1
        assert_eq!(x.degree(&var_x), Some(1));

        // Constant has degree 0
        assert_eq!(Expr::from(5).degree(&var_x), Some(0));

        // x^2 * x = x^3 has degree 3
        let p2 = x.clone().pow(Expr::from(2)) * x.clone();
        assert_eq!(p2.degree(&var_x), Some(3));

        // sin(x) is not a polynomial
        let s = x.clone().sin();
        assert_eq!(s.degree(&var_x), None);
    }

    #[test]
    fn test_coefficient() {
        let x = Expr::symbol("x");
        let var_x = Symbol::new("x");

        // 3*x^2 + 2*x + 1
        let p = Expr::from(3) * x.clone().pow(Expr::from(2))
            + Expr::from(2) * x.clone()
            + Expr::from(1);

        // Coefficient of x^0 (constant term) is 1
        let c0 = p.coefficient(&var_x, 0);
        assert!(c0.is_some());
        assert_eq!(c0.unwrap().eval_rational(), Some(rustmath_rationals::Rational::new(1, 1).unwrap()));

        // Coefficient of x^1 is 2
        let c1 = p.coefficient(&var_x, 1);
        assert!(c1.is_some());

        // Coefficient of x^2 is 3
        let c2 = p.coefficient(&var_x, 2);
        assert!(c2.is_some());

        // Coefficient of x^3 is 0
        let c3 = p.coefficient(&var_x, 3);
        assert!(c3.is_some());
        assert_eq!(c3.unwrap().eval_rational(), Some(rustmath_rationals::Rational::new(0, 1).unwrap()));
    }

    #[test]
    fn test_is_rational_expression() {
        let x = Expr::symbol("x");
        let var_x = Symbol::new("x");

        // (x^2 + 1) / (x + 1) is a rational expression
        let r = (x.clone().pow(Expr::from(2)) + Expr::from(1)) / (x.clone() + Expr::from(1));
        assert!(r.is_rational_expression(&var_x));

        // x^2 is a rational expression (polynomial)
        let p = x.clone().pow(Expr::from(2));
        assert!(p.is_rational_expression(&var_x));

        // sin(x) / x is not a rational expression
        let s = x.clone().sin() / x.clone();
        assert!(!s.is_rational_expression(&var_x));
    }
}
