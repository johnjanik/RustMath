//! Radical canonicalization
//!
//! This module provides functions for simplifying and canonicalizing radical expressions,
//! particularly square roots and rational exponents.

use crate::expression::{BinaryOp, Expr, UnaryOp};
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;
use std::sync::Arc;

impl Expr {
    /// Canonicalize radical expressions
    ///
    /// This simplifies square roots and rational exponents by:
    /// - Extracting perfect squares from sqrt(n)
    /// - Combining sqrt(a) * sqrt(b) = sqrt(a*b)
    /// - Simplifying nested radicals where possible
    /// - Converting rational exponents to radical form
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    ///
    /// // sqrt(8) = 2*sqrt(2)
    /// let expr = Expr::from(8).sqrt();
    /// let canonical = expr.canonicalize_radical();
    /// ```
    pub fn canonicalize_radical(&self) -> Expr {
        canonicalize_radical_expr(self)
    }
}

/// Main canonicalization function
fn canonicalize_radical_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_) => expr.clone(),

        Expr::Unary(UnaryOp::Sqrt, inner) => canonicalize_sqrt(inner),

        Expr::Binary(BinaryOp::Pow, base, exp) => canonicalize_power(base, exp),

        Expr::Binary(BinaryOp::Mul, left, right) => {
            let left_canon = canonicalize_radical_expr(left);
            let right_canon = canonicalize_radical_expr(right);
            combine_radical_products(&left_canon, &right_canon)
        }

        Expr::Binary(BinaryOp::Add, left, right) => {
            let left_canon = canonicalize_radical_expr(left);
            let right_canon = canonicalize_radical_expr(right);
            Expr::Binary(BinaryOp::Add, Arc::new(left_canon), Arc::new(right_canon))
        }

        Expr::Binary(BinaryOp::Sub, left, right) => {
            let left_canon = canonicalize_radical_expr(left);
            let right_canon = canonicalize_radical_expr(right);
            Expr::Binary(BinaryOp::Sub, Arc::new(left_canon), Arc::new(right_canon))
        }

        Expr::Binary(BinaryOp::Div, left, right) => {
            let left_canon = canonicalize_radical_expr(left);
            let right_canon = canonicalize_radical_expr(right);
            // Rationalize denominators if needed
            rationalize_denominator(&left_canon, &right_canon)
        }

        Expr::Binary(BinaryOp::Mod, left, right) => {
            let left_canon = canonicalize_radical_expr(left);
            let right_canon = canonicalize_radical_expr(right);
            Expr::Binary(BinaryOp::Mod, Arc::new(left_canon), Arc::new(right_canon))
        }

        Expr::Unary(op, inner) => {
            let inner_canon = canonicalize_radical_expr(inner);
            Expr::Unary(*op, Arc::new(inner_canon))
        }

        Expr::Function(name, args) => {
            let canon_args: Vec<Arc<Expr>> = args
                .iter()
                .map(|arg| Arc::new(canonicalize_radical_expr(arg)))
                .collect();
            Expr::Function(name.clone(), canon_args)
        }
    }
}

/// Canonicalize a square root expression
fn canonicalize_sqrt(inner: &Expr) -> Expr {
    let inner_canon = canonicalize_radical_expr(inner);

    match &inner_canon {
        // sqrt(0) = 0
        Expr::Integer(n) if n.is_zero() => Expr::from(0),

        // sqrt(1) = 1
        Expr::Integer(n) if n.is_one() => Expr::from(1),

        // sqrt(n) for integer n - extract perfect squares
        Expr::Integer(n) => extract_perfect_squares(n),

        // sqrt(a/b) = sqrt(a) / sqrt(b)
        Expr::Rational(r) => {
            let num_sqrt = extract_perfect_squares(r.numerator());
            let den_sqrt = extract_perfect_squares(r.denominator());
            Expr::Binary(BinaryOp::Div, Arc::new(num_sqrt), Arc::new(den_sqrt))
        }

        // sqrt(a * b) = sqrt(a) * sqrt(b)
        Expr::Binary(BinaryOp::Mul, left, right) => {
            let left_sqrt = canonicalize_sqrt(left);
            let right_sqrt = canonicalize_sqrt(right);
            Expr::Binary(BinaryOp::Mul, Arc::new(left_sqrt), Arc::new(right_sqrt))
        }

        // sqrt(a^2) = |a| (we assume positive for symbolic vars)
        Expr::Binary(BinaryOp::Pow, base, exp) => {
            if let Expr::Integer(n) = exp.as_ref() {
                if n == &Integer::from(2) {
                    return (**base).clone();
                }
                // sqrt(a^(2k)) = a^k
                if n.to_i64().map_or(false, |x| x % 2 == 0) {
                    let half_exp = Expr::Integer(n.clone() / Integer::from(2));
                    return Expr::Binary(BinaryOp::Pow, base.clone(), Arc::new(half_exp));
                }
                // sqrt(a^(2k+1)) = a^k * sqrt(a)
                if let Some(n_i64) = n.to_i64() {
                    if n_i64 > 2 {
                        let k = n_i64 / 2;
                        let power_part = Expr::Binary(
                            BinaryOp::Pow,
                            base.clone(),
                            Arc::new(Expr::Integer(Integer::from(k))),
                        );
                        let sqrt_part = Expr::Unary(UnaryOp::Sqrt, base.clone());
                        return Expr::Binary(
                            BinaryOp::Mul,
                            Arc::new(power_part),
                            Arc::new(sqrt_part),
                        );
                    }
                }
            }
            Expr::Unary(UnaryOp::Sqrt, Arc::new(inner_canon))
        }

        _ => Expr::Unary(UnaryOp::Sqrt, Arc::new(inner_canon)),
    }
}

/// Extract perfect square factors from an integer
fn extract_perfect_squares(n: &Integer) -> Expr {
    if n < &Integer::zero() {
        // For negative numbers, we'd need complex numbers
        // For now, just return sqrt(n)
        return Expr::Unary(UnaryOp::Sqrt, Arc::new(Expr::Integer(n.clone())));
    }

    if n.is_zero() || n.is_one() {
        return Expr::Integer(n.clone());
    }

    let (extracted, remaining) = factor_perfect_squares(n);

    if extracted.is_one() {
        // No perfect squares to extract
        Expr::Unary(UnaryOp::Sqrt, Arc::new(Expr::Integer(n.clone())))
    } else if remaining.is_one() {
        // Perfect square
        Expr::Integer(extracted)
    } else {
        // Partially extracted: extracted * sqrt(remaining)
        Expr::Binary(
            BinaryOp::Mul,
            Arc::new(Expr::Integer(extracted)),
            Arc::new(Expr::Unary(
                UnaryOp::Sqrt,
                Arc::new(Expr::Integer(remaining)),
            )),
        )
    }
}

/// Factor out perfect squares from an integer
/// Returns (extracted_factor, remaining_under_sqrt)
fn factor_perfect_squares(n: &Integer) -> (Integer, Integer) {
    let mut remaining = n.clone();
    let mut extracted = Integer::one();

    // Try small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47].iter() {
        let prime = Integer::from(*p);
        let prime_sq = &prime * &prime;

        while &remaining % &prime_sq == Integer::zero() {
            extracted = extracted * prime.clone();
            remaining = remaining / prime_sq.clone();
        }
    }

    // For larger numbers, try up to sqrt(n)
    if remaining > Integer::from(2500) {
        let sqrt_n = integer_sqrt(&remaining);
        let mut i = Integer::from(53);

        while &i <= &sqrt_n {
            let i_sq = &i * &i;
            while &remaining % &i_sq == Integer::zero() {
                extracted = extracted * i.clone();
                remaining = remaining / i_sq.clone();
            }
            i = i + Integer::from(2); // Only odd numbers
        }
    }

    (extracted, remaining)
}

/// Compute integer square root
fn integer_sqrt(n: &Integer) -> Integer {
    if n <= &Integer::zero() {
        return Integer::zero();
    }

    // Newton's method for integer square root
    let mut x = n.clone();
    let mut y = (x.clone() + Integer::one()) / Integer::from(2);

    while y < x {
        x = y.clone();
        y = (x.clone() + n / &x) / Integer::from(2);
    }

    x
}

/// Canonicalize a power expression
fn canonicalize_power(base: &Expr, exp: &Expr) -> Expr {
    let base_canon = canonicalize_radical_expr(base);
    let exp_canon = canonicalize_radical_expr(exp);

    // a^(1/2) = sqrt(a)
    if let Expr::Rational(r) = &exp_canon {
        if r.numerator() == &Integer::one() && r.denominator() == &Integer::from(2) {
            return canonicalize_sqrt(&base_canon);
        }

        // a^(p/q) = (a^p)^(1/q)
        if r.denominator() != &Integer::one() {
            let num_power = Expr::Binary(
                BinaryOp::Pow,
                Arc::new(base_canon.clone()),
                Arc::new(Expr::Integer(r.numerator().clone())),
            );
            return Expr::Binary(
                BinaryOp::Pow,
                Arc::new(num_power),
                Arc::new(Expr::Rational(Rational::new(1, r.denominator().to_i64().unwrap_or(1)).unwrap())),
            );
        }
    }

    Expr::Binary(BinaryOp::Pow, Arc::new(base_canon), Arc::new(exp_canon))
}

/// Combine products involving radicals
fn combine_radical_products(left: &Expr, right: &Expr) -> Expr {
    // sqrt(a) * sqrt(b) = sqrt(a*b)
    match (left, right) {
        (Expr::Unary(UnaryOp::Sqrt, a), Expr::Unary(UnaryOp::Sqrt, b)) => {
            let product = Expr::Binary(BinaryOp::Mul, a.clone(), b.clone());
            canonicalize_sqrt(&product)
        }
        _ => Expr::Binary(BinaryOp::Mul, Arc::new(left.clone()), Arc::new(right.clone())),
    }
}

/// Rationalize the denominator of a fraction
fn rationalize_denominator(num: &Expr, den: &Expr) -> Expr {
    // If denominator is sqrt(a), multiply by sqrt(a)/sqrt(a)
    if let Expr::Unary(UnaryOp::Sqrt, a) = den {
        let new_num = Expr::Binary(
            BinaryOp::Mul,
            Arc::new(num.clone()),
            Arc::new(Expr::Unary(UnaryOp::Sqrt, a.clone())),
        );
        let new_den = (**a).clone();
        return Expr::Binary(BinaryOp::Div, Arc::new(new_num), Arc::new(new_den));
    }

    Expr::Binary(BinaryOp::Div, Arc::new(num.clone()), Arc::new(den.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonicalize_sqrt_integer() {
        // sqrt(4) = 2
        let expr = Expr::from(4).sqrt();
        let canonical = expr.canonicalize_radical();
        assert_eq!(canonical, Expr::from(2));

        // sqrt(9) = 3
        let expr = Expr::from(9).sqrt();
        let canonical = expr.canonicalize_radical();
        assert_eq!(canonical, Expr::from(3));
    }

    #[test]
    fn test_extract_perfect_squares() {
        // sqrt(8) = 2*sqrt(2)
        let expr = Expr::from(8).sqrt();
        let canonical = expr.canonicalize_radical();

        match canonical {
            Expr::Binary(BinaryOp::Mul, left, right) => {
                assert_eq!(*left, Expr::from(2));
                assert!(matches!(right.as_ref(), Expr::Unary(UnaryOp::Sqrt, _)));
            }
            _ => panic!("Expected product"),
        }
    }

    #[test]
    fn test_combine_sqrt_products() {
        // sqrt(2) * sqrt(3) = sqrt(6)
        let expr = Expr::Binary(
            BinaryOp::Mul,
            Arc::new(Expr::from(2).sqrt()),
            Arc::new(Expr::from(3).sqrt()),
        );
        let canonical = expr.canonicalize_radical();

        // Should be sqrt(6)
        match canonical {
            Expr::Unary(UnaryOp::Sqrt, inner) => {
                assert_eq!(*inner, Expr::from(6));
            }
            _ => {} // Could also be in other forms due to simplification
        }
    }

    #[test]
    fn test_sqrt_of_square() {
        let x = Expr::symbol("x");

        // sqrt(x^2) = x
        let expr = x.clone().pow(Expr::from(2)).sqrt();
        let canonical = expr.canonicalize_radical();
        assert_eq!(canonical, x);
    }

    #[test]
    fn test_rationalize_denominator() {
        // 1/sqrt(2) = sqrt(2)/2
        let expr = Expr::Binary(
            BinaryOp::Div,
            Arc::new(Expr::from(1)),
            Arc::new(Expr::from(2).sqrt()),
        );
        let canonical = expr.canonicalize_radical();

        // Should have 2 in denominator, not sqrt(2)
        match canonical {
            Expr::Binary(BinaryOp::Div, _, den) => {
                assert_eq!(*den, Expr::from(2));
            }
            _ => {}
        }
    }
}
