//! Symbolic expression factorization
//!
//! This module provides factorization capabilities for symbolic expressions,
//! integrating with the polynomial factorization from rustmath-polynomials.

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::{factor_over_integers, UnivariatePolynomial};
use std::sync::Arc;

impl Expr {
    /// Factor a symbolic expression
    ///
    /// This attempts to factor polynomial expressions into irreducible factors.
    /// For univariate polynomials over integers, it uses advanced factorization algorithms.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    /// use rustmath_symbolic::Symbol;
    ///
    /// let x = Expr::symbol("x");
    ///
    /// // x^2 - 1 = (x - 1)(x + 1)
    /// let expr = x.clone().pow(Expr::from(2)) - Expr::from(1);
    /// let factored = expr.factor();
    /// ```
    pub fn factor(&self) -> Expr {
        factor_expression(self)
    }
}

/// Main factorization function
fn factor_expression(expr: &Expr) -> Expr {
    match expr {
        // Constants and symbols are already factored
        Expr::Integer(_) | Expr::Rational(_) | Expr::Symbol(_) => expr.clone(),

        // For sums, try to extract common factors
        Expr::Binary(BinaryOp::Add, _, _) | Expr::Binary(BinaryOp::Sub, _, _) => {
            factor_sum(expr)
        }

        // For products, factor each term
        Expr::Binary(BinaryOp::Mul, left, right) => {
            let left_factored = factor_expression(left);
            let right_factored = factor_expression(right);
            Expr::Binary(BinaryOp::Mul, Arc::new(left_factored), Arc::new(right_factored))
        }

        // For powers, factor the base
        Expr::Binary(BinaryOp::Pow, base, exp) => {
            let base_factored = factor_expression(base);
            Expr::Binary(BinaryOp::Pow, Arc::new(base_factored), exp.clone())
        }

        // Other expressions
        _ => expr.clone(),
    }
}

/// Try to factor a sum expression
fn factor_sum(expr: &Expr) -> Expr {
    // Try to identify if this is a polynomial in one variable
    let vars = collect_variables(expr);

    if vars.is_empty() {
        // Just a constant
        return expr.clone();
    }

    if vars.len() == 1 {
        // Univariate polynomial - use polynomial factorization
        let var = vars.into_iter().next().unwrap();
        if let Some(factored) = factor_univariate_polynomial(expr, &var) {
            return factored;
        }
    }

    // For multivariate or non-polynomial expressions, try to extract common factors
    extract_common_factors(expr)
}

/// Factor a univariate polynomial expression
fn factor_univariate_polynomial(expr: &Expr, var: &Symbol) -> Option<Expr> {
    // Convert expression to polynomial
    let poly = expr_to_univariate_poly(expr, var)?;

    // Factor the polynomial
    let factored = factor_over_integers(&poly).ok()?;

    // Convert factored form back to expression
    Some(factored_poly_to_expr(&factored, var))
}

/// Convert a symbolic expression to a univariate polynomial
fn expr_to_univariate_poly(expr: &Expr, var: &Symbol) -> Option<UnivariatePolynomial<Integer>> {
    if !expr.is_polynomial(var) {
        return None;
    }

    let degree = expr.degree(var)?;
    let mut coeffs = Vec::new();

    // Extract coefficients from degree 0 to degree
    for i in 0..=degree {
        let coeff_expr = expr.coefficient(var, i)?;
        let coeff_int = eval_to_integer(&coeff_expr)?;
        coeffs.push(coeff_int);
    }

    Some(UnivariatePolynomial::new(coeffs))
}

/// Evaluate an expression to an integer (if it's a constant integer expression)
fn eval_to_integer(expr: &Expr) -> Option<Integer> {
    match expr {
        Expr::Integer(n) => Some(n.clone()),
        Expr::Binary(BinaryOp::Add, left, right) => {
            let l = eval_to_integer(left)?;
            let r = eval_to_integer(right)?;
            Some(l + r)
        }
        Expr::Binary(BinaryOp::Sub, left, right) => {
            let l = eval_to_integer(left)?;
            let r = eval_to_integer(right)?;
            Some(l - r)
        }
        Expr::Binary(BinaryOp::Mul, left, right) => {
            let l = eval_to_integer(left)?;
            let r = eval_to_integer(right)?;
            Some(l * r)
        }
        Expr::Unary(UnaryOp::Neg, inner) => {
            let n = eval_to_integer(inner)?;
            Some(-n)
        }
        _ => None,
    }
}

/// Convert factored polynomial to expression
fn factored_poly_to_expr(
    factored: &Vec<(UnivariatePolynomial<Integer>, u32)>,
    var: &Symbol,
) -> Expr {
    let mut result = Expr::from(1);

    for (poly, multiplicity) in factored {
        let poly_expr = poly_to_expr(poly, var);

        let factor = if *multiplicity == 1 {
            poly_expr
        } else {
            Expr::Binary(
                BinaryOp::Pow,
                Arc::new(poly_expr),
                Arc::new(Expr::Integer(Integer::from(*multiplicity as i64))),
            )
        };

        result = Expr::Binary(BinaryOp::Mul, Arc::new(result), Arc::new(factor));
    }

    result.simplify()
}

/// Convert polynomial to expression
fn poly_to_expr(poly: &UnivariatePolynomial<Integer>, var: &Symbol) -> Expr {
    let degree = poly.degree();
    if degree.is_none() {
        return Expr::from(0);
    }

    let degree = degree.unwrap();
    let coeffs = poly.coefficients();

    let mut terms = Vec::new();

    for (i, coeff) in coeffs.iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }

        let term = if i == 0 {
            Expr::Integer(coeff.clone())
        } else if i == 1 {
            if coeff.is_one() {
                Expr::Symbol(var.clone())
            } else {
                Expr::Binary(
                    BinaryOp::Mul,
                    Arc::new(Expr::Integer(coeff.clone())),
                    Arc::new(Expr::Symbol(var.clone())),
                )
            }
        } else {
            let power = Expr::Binary(
                BinaryOp::Pow,
                Arc::new(Expr::Symbol(var.clone())),
                Arc::new(Expr::Integer(Integer::from(i as i64))),
            );
            if coeff.is_one() {
                power
            } else {
                Expr::Binary(
                    BinaryOp::Mul,
                    Arc::new(Expr::Integer(coeff.clone())),
                    Arc::new(power),
                )
            }
        };

        terms.push(term);
    }

    if terms.is_empty() {
        return Expr::from(0);
    }

    if terms.len() == 1 {
        return terms.into_iter().next().unwrap();
    }

    // Build sum
    let mut result = terms[0].clone();
    for term in terms.into_iter().skip(1) {
        result = Expr::Binary(BinaryOp::Add, Arc::new(result), Arc::new(term));
    }

    result
}

/// Collect all variables in an expression
fn collect_variables(expr: &Expr) -> Vec<Symbol> {
    let mut vars = Vec::new();
    collect_variables_rec(expr, &mut vars);
    vars.sort_by(|a, b| a.name().cmp(b.name()));
    vars.dedup();
    vars
}

fn collect_variables_rec(expr: &Expr, vars: &mut Vec<Symbol>) {
    match expr {
        Expr::Symbol(s) => vars.push(s.clone()),
        Expr::Binary(_, left, right) => {
            collect_variables_rec(left, vars);
            collect_variables_rec(right, vars);
        }
        Expr::Unary(_, inner) => {
            collect_variables_rec(inner, vars);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_variables_rec(arg, vars);
            }
        }
        _ => {}
    }
}

/// Extract common factors from a sum
fn extract_common_factors(expr: &Expr) -> Expr {
    let terms = collect_sum_terms(expr);
    if terms.len() <= 1 {
        return expr.clone();
    }

    // Find GCD of integer coefficients
    let mut gcd_coeff = Integer::zero();
    for term in &terms {
        let coeff = extract_integer_coefficient(term);
        if gcd_coeff.is_zero() {
            gcd_coeff = coeff.abs();
        } else {
            gcd_coeff = gcd_coeff.gcd(&coeff.abs());
        }
    }

    // If GCD is greater than 1, factor it out
    if gcd_coeff > Integer::one() {
        let factored_terms: Vec<Expr> = terms
            .into_iter()
            .map(|term| divide_by_integer(&term, &gcd_coeff))
            .collect();

        let sum = build_sum(factored_terms);
        return Expr::Binary(
            BinaryOp::Mul,
            Arc::new(Expr::Integer(gcd_coeff)),
            Arc::new(sum),
        );
    }

    expr.clone()
}

/// Collect all terms in a sum expression
fn collect_sum_terms(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::Binary(BinaryOp::Add, left, right) => {
            let mut terms = collect_sum_terms(left);
            terms.extend(collect_sum_terms(right));
            terms
        }
        Expr::Binary(BinaryOp::Sub, left, right) => {
            let mut terms = collect_sum_terms(left);
            let right_neg = Expr::Unary(UnaryOp::Neg, right.clone());
            terms.extend(collect_sum_terms(&right_neg));
            terms
        }
        _ => vec![expr.clone()],
    }
}

/// Extract the integer coefficient from a term
fn extract_integer_coefficient(term: &Expr) -> Integer {
    match term {
        Expr::Integer(n) => n.clone(),
        Expr::Binary(BinaryOp::Mul, left, _) => {
            if let Expr::Integer(n) = left.as_ref() {
                n.clone()
            } else {
                Integer::one()
            }
        }
        Expr::Unary(UnaryOp::Neg, inner) => -extract_integer_coefficient(inner),
        _ => Integer::one(),
    }
}

/// Divide a term by an integer
fn divide_by_integer(term: &Expr, divisor: &Integer) -> Expr {
    if divisor.is_one() {
        return term.clone();
    }

    match term {
        Expr::Integer(n) => Expr::Integer(n.clone() / divisor.clone()),
        Expr::Binary(BinaryOp::Mul, left, right) => {
            if let Expr::Integer(n) = left.as_ref() {
                let new_coeff = n.clone() / divisor.clone();
                if new_coeff.is_one() {
                    (**right).clone()
                } else {
                    Expr::Binary(
                        BinaryOp::Mul,
                        Arc::new(Expr::Integer(new_coeff)),
                        right.clone(),
                    )
                }
            } else {
                term.clone()
            }
        }
        Expr::Unary(UnaryOp::Neg, inner) => {
            let divided = divide_by_integer(inner, divisor);
            Expr::Unary(UnaryOp::Neg, Arc::new(divided))
        }
        _ => term.clone(),
    }
}

/// Build a sum from a list of terms
fn build_sum(terms: Vec<Expr>) -> Expr {
    if terms.is_empty() {
        return Expr::from(0);
    }

    if terms.len() == 1 {
        return terms.into_iter().next().unwrap();
    }

    let mut result = terms[0].clone();
    for term in terms.into_iter().skip(1) {
        result = Expr::Binary(BinaryOp::Add, Arc::new(result), Arc::new(term));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_difference_of_squares() {
        let x = Expr::symbol("x");
        let var_x = Symbol::new("x");

        // x^2 - 1 should factor to (x - 1)(x + 1)
        let expr = x.clone().pow(Expr::from(2)) - Expr::from(1);
        let factored = expr.factor();

        // The result should be a product
        assert!(matches!(factored, Expr::Binary(BinaryOp::Mul, _, _)));
    }

    #[test]
    fn test_factor_quadratic() {
        let x = Expr::symbol("x");

        // x^2 + 5x + 6 = (x + 2)(x + 3)
        let expr = x.clone().pow(Expr::from(2))
            + Expr::from(5) * x.clone()
            + Expr::from(6);
        let factored = expr.factor();

        // The result should be a product
        assert!(matches!(factored, Expr::Binary(BinaryOp::Mul, _, _)));
    }

    #[test]
    fn test_factor_extract_common() {
        let x = Expr::symbol("x");

        // 2x + 4 = 2(x + 2)
        let expr = Expr::from(2) * x.clone() + Expr::from(4);
        let factored = expr.factor();

        // The result should extract the factor 2
        assert!(matches!(factored, Expr::Binary(BinaryOp::Mul, _, _)));
    }

    #[test]
    fn test_factor_constant() {
        let expr = Expr::from(42);
        let factored = expr.factor();
        assert_eq!(factored, Expr::from(42));
    }
}
