//! Functional calculus operations
//!
//! This module provides high-level functional operations including
//! simplification, expansion, and other expression manipulations.

use rustmath_symbolic::{BinaryOp, Expr, UnaryOp};

/// Expand an expression by distributing products over sums
///
/// # Arguments
///
/// * `expr` - The expression to expand
///
/// # Returns
///
/// The expanded expression
///
/// # Examples
///
/// ```
/// use rustmath_calculus::functional::expand;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// let y = Expr::symbol("y");
/// // (x + 1) * (y + 2) expands to xy + 2x + y + 2
/// let expr = (x.clone() + Expr::from(1)) * (y.clone() + Expr::from(2));
/// let expanded = expand(&expr);
/// ```
pub fn expand(expr: &Expr) -> Expr {
    match expr {
        // Expand binary operations
        Expr::Binary(op, left, right) => {
            let left_exp = expand(left);
            let right_exp = expand(right);

            match op {
                // Distribute multiplication over addition: (a+b)*c = a*c + b*c
                BinaryOp::Mul => expand_product(&left_exp, &right_exp),

                // Expand power: (a+b)^n
                BinaryOp::Pow => {
                    if let Expr::Integer(n) = &right_exp {
                        use rustmath_integers::Integer;
                        let n_str = format!("{}", n);
                        if let Ok(n_val) = n_str.parse::<i64>() {
                            if n_val >= 0 && n_val <= 10 {
                                // Expand small integer powers
                                expand_power(&left_exp, n_val as usize)
                            } else {
                                Expr::Binary(*op, std::sync::Arc::new(left_exp), std::sync::Arc::new(right_exp))
                            }
                        } else {
                            Expr::Binary(*op, std::sync::Arc::new(left_exp), std::sync::Arc::new(right_exp))
                        }
                    } else {
                        Expr::Binary(*op, std::sync::Arc::new(left_exp), std::sync::Arc::new(right_exp))
                    }
                }

                // Recursively expand other operations
                _ => Expr::Binary(*op, std::sync::Arc::new(left_exp), std::sync::Arc::new(right_exp)),
            }
        }

        // Expand unary operations
        Expr::Unary(op, inner) => {
            let inner_exp = expand(inner);
            Expr::Unary(*op, std::sync::Arc::new(inner_exp))
        }

        // Expand function arguments
        Expr::Function(name, args) => {
            let args_exp: Vec<std::sync::Arc<Expr>> = args.iter()
                .map(|arg| std::sync::Arc::new(expand(arg)))
                .collect();
            Expr::Function(name.clone(), args_exp)
        }

        // Atoms don't need expansion
        _ => expr.clone(),
    }
}

/// Expand a product, distributing over addition
fn expand_product(left: &Expr, right: &Expr) -> Expr {
    match (left, right) {
        // (a + b) * c = a*c + b*c
        (Expr::Binary(BinaryOp::Add, a, b), c) => {
            expand_product(a, c) + expand_product(b, c)
        }
        // (a - b) * c = a*c - b*c
        (Expr::Binary(BinaryOp::Sub, a, b), c) => {
            expand_product(a, c) - expand_product(b, c)
        }
        // c * (a + b) = c*a + c*b
        (c, Expr::Binary(BinaryOp::Add, a, b)) => {
            expand_product(c, a) + expand_product(c, b)
        }
        // c * (a - b) = c*a - c*b
        (c, Expr::Binary(BinaryOp::Sub, a, b)) => {
            expand_product(c, a) - expand_product(c, b)
        }
        // Otherwise, just multiply
        _ => left.clone() * right.clone(),
    }
}

/// Expand a power expression
fn expand_power(base: &Expr, exp: usize) -> Expr {
    if exp == 0 {
        return Expr::from(1);
    }
    if exp == 1 {
        return base.clone();
    }

    // For (a + b)^n, use binomial expansion for small n
    match base {
        Expr::Binary(BinaryOp::Add, a, b) => {
            // Binomial expansion
            let mut result = Expr::from(0);
            for k in 0..=exp {
                let coeff = binomial_coefficient(exp, k);
                let a_pow = expand_power(a, exp - k);
                let b_pow = expand_power(b, k);
                let term = Expr::from(coeff) * a_pow * b_pow;
                result = result + term;
            }
            expand(&result)
        }
        Expr::Binary(BinaryOp::Sub, a, b) => {
            // Convert a - b to a + (-b) and expand
            let neg_b = -(**b).clone();
            let as_add = Expr::Binary(BinaryOp::Add, a.clone(), std::sync::Arc::new(neg_b));
            expand_power(&as_add, exp)
        }
        _ => base.clone().pow(Expr::from(exp as i64)),
    }
}

/// Compute binomial coefficient C(n, k)
fn binomial_coefficient(n: usize, k: usize) -> i64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k); // Use symmetry
    let mut result = 1i64;
    for i in 0..k {
        result = result * (n - i) as i64 / (i + 1) as i64;
    }
    result
}

/// Simplify an expression using basic algebraic rules
///
/// # Arguments
///
/// * `expr` - The expression to simplify
///
/// # Returns
///
/// The simplified expression
///
/// # Examples
///
/// ```
/// use rustmath_calculus::functional::simplify;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// // x + 0 simplifies to x
/// let expr = x.clone() + Expr::from(0);
/// let simplified = simplify(&expr);
/// ```
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Binary(op, left, right) => {
            let left_simp = simplify(left);
            let right_simp = simplify(right);

            simplify_binary(*op, &left_simp, &right_simp)
        }

        Expr::Unary(op, inner) => {
            let inner_simp = simplify(inner);
            match op {
                UnaryOp::Neg => simplify_neg(&inner_simp),
                _ => Expr::Unary(*op, std::sync::Arc::new(inner_simp)),
            }
        }

        Expr::Function(name, args) => {
            let args_simp: Vec<std::sync::Arc<Expr>> = args.iter()
                .map(|arg| std::sync::Arc::new(simplify(arg)))
                .collect();
            Expr::Function(name.clone(), args_simp)
        }

        _ => expr.clone(),
    }
}

/// Helper to simplify binary operations
fn simplify_binary(op: BinaryOp, left: &Expr, right: &Expr) -> Expr {
    match op {
        BinaryOp::Add => simplify_add(left, right),
        BinaryOp::Sub => simplify_sub(left, right),
        BinaryOp::Mul => simplify_mul(left, right),
        BinaryOp::Div => simplify_div(left, right),
        BinaryOp::Pow => simplify_pow(left, right),
    }
}

/// Simplify addition
fn simplify_add(left: &Expr, right: &Expr) -> Expr {
    use rustmath_integers::Integer;
    match (left, right) {
        // 0 + x = x
        (Expr::Integer(n), x) if n == &Integer::from(0) => x.clone(),
        // x + 0 = x
        (x, Expr::Integer(n)) if n == &Integer::from(0) => x.clone(),
        // Combine integer constants
        (Expr::Integer(a), Expr::Integer(b)) => Expr::Integer(a.clone() + b.clone()),
        // Otherwise keep as-is
        _ => left.clone() + right.clone(),
    }
}

/// Simplify subtraction
fn simplify_sub(left: &Expr, right: &Expr) -> Expr {
    use rustmath_integers::Integer;
    match (left, right) {
        // x - 0 = x
        (x, Expr::Integer(n)) if n == &Integer::from(0) => x.clone(),
        // x - x = 0 (requires equality checking)
        _ if left == right => Expr::from(0),
        // Combine integer constants
        (Expr::Integer(a), Expr::Integer(b)) => Expr::Integer(a.clone() - b.clone()),
        _ => left.clone() - right.clone(),
    }
}

/// Simplify multiplication
fn simplify_mul(left: &Expr, right: &Expr) -> Expr {
    use rustmath_integers::Integer;
    match (left, right) {
        // 0 * x = 0
        (Expr::Integer(n), _) if n == &Integer::from(0) => Expr::from(0),
        // x * 0 = 0
        (_, Expr::Integer(n)) if n == &Integer::from(0) => Expr::from(0),
        // 1 * x = x
        (Expr::Integer(n), x) if n == &Integer::from(1) => x.clone(),
        // x * 1 = x
        (x, Expr::Integer(n)) if n == &Integer::from(1) => x.clone(),
        // Combine integer constants
        (Expr::Integer(a), Expr::Integer(b)) => Expr::Integer(a.clone() * b.clone()),
        _ => left.clone() * right.clone(),
    }
}

/// Simplify division
fn simplify_div(left: &Expr, right: &Expr) -> Expr {
    use rustmath_integers::Integer;
    match (left, right) {
        // 0 / x = 0 (x != 0)
        (Expr::Integer(n), _) if n == &Integer::from(0) => Expr::from(0),
        // x / 1 = x
        (x, Expr::Integer(n)) if n == &Integer::from(1) => x.clone(),
        // x / x = 1
        _ if left == right => Expr::from(1),
        _ => left.clone() / right.clone(),
    }
}

/// Simplify power
fn simplify_pow(base: &Expr, exp: &Expr) -> Expr {
    use rustmath_integers::Integer;
    match (base, exp) {
        // x^0 = 1
        (_, Expr::Integer(n)) if n == &Integer::from(0) => Expr::from(1),
        // x^1 = x
        (x, Expr::Integer(n)) if n == &Integer::from(1) => x.clone(),
        // 0^x = 0 (x > 0)
        (Expr::Integer(n), _) if n == &Integer::from(0) => Expr::from(0),
        // 1^x = 1
        (Expr::Integer(n), _) if n == &Integer::from(1) => Expr::from(1),
        _ => base.clone().pow(exp.clone()),
    }
}

/// Simplify negation
fn simplify_neg(expr: &Expr) -> Expr {
    match expr {
        // -(-x) = x
        Expr::Unary(UnaryOp::Neg, inner) => (**inner).clone(),
        // -(constant)
        Expr::Integer(n) => Expr::Integer(-n.clone()),
        _ => -expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_product() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");

        // (x + 1) * y should expand to x*y + y
        let expr = (x.clone() + Expr::from(1)) * y.clone();
        let _expanded = expand(&expr);
        // Result structure will be: (x*y) + (1*y)
    }

    #[test]
    fn test_expand_power_small() {
        let x = Expr::symbol("x");

        // (x + 1)^2 = x^2 + 2x + 1
        let expr = (x.clone() + Expr::from(1)).pow(Expr::from(2));
        let _expanded = expand(&expr);
        // Result should be expanded form
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(4, 0), 1);
        assert_eq!(binomial_coefficient(4, 1), 4);
        assert_eq!(binomial_coefficient(4, 2), 6);
        assert_eq!(binomial_coefficient(4, 3), 4);
        assert_eq!(binomial_coefficient(4, 4), 1);
    }

    #[test]
    fn test_simplify_addition() {
        let x = Expr::symbol("x");

        // x + 0 = x
        let expr = x.clone() + Expr::from(0);
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 0 + x = x
        let expr = Expr::from(0) + x.clone();
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 3 + 4 = 7
        let expr = Expr::from(3) + Expr::from(4);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::from(7));
    }

    #[test]
    fn test_simplify_multiplication() {
        let x = Expr::symbol("x");

        // x * 1 = x
        let expr = x.clone() * Expr::from(1);
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // x * 0 = 0
        let expr = x.clone() * Expr::from(0);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::from(0));

        // 3 * 4 = 12
        let expr = Expr::from(3) * Expr::from(4);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::from(12));
    }

    #[test]
    fn test_simplify_power() {
        let x = Expr::symbol("x");

        // x^1 = x
        let expr = x.clone().pow(Expr::from(1));
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // x^0 = 1
        let expr = x.clone().pow(Expr::from(0));
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::from(1));
    }

    #[test]
    fn test_simplify_negation() {
        let x = Expr::symbol("x");

        // -(-x) = x
        let expr = -(-x.clone());
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // -5 = -5
        let expr = -Expr::from(5);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::from(-5));
    }

    #[test]
    fn test_simplify_division() {
        let x = Expr::symbol("x");

        // x / 1 = x
        let expr = x.clone() / Expr::from(1);
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // x / x = 1
        let expr = x.clone() / x.clone();
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::from(1));
    }
}
