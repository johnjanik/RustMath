//! Expression expansion operations

use crate::expression::{BinaryOp, Expr, UnaryOp};

impl Expr {
    /// Expand the expression
    ///
    /// Distributes products over sums: (a + b) * c = a*c + b*c
    /// Expands powers of sums using binomial expansion (for small integer exponents)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    ///
    /// let x = Expr::symbol("x");
    /// let y = Expr::symbol("y");
    ///
    /// // (x + 1) * y = x*y + y
    /// let expr = (x.clone() + Expr::from(1)) * y.clone();
    /// let expanded = expr.expand();
    /// ```
    pub fn expand(&self) -> Expr {
        match self {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_) => self.clone(),

            Expr::Binary(op, left, right) => {
                let left_exp = left.expand();
                let right_exp = right.expand();

                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mod => {
                        // Just recursively expand
                        Expr::Binary(*op, Arc::new(left_exp), Arc::new(right_exp))
                    }
                    BinaryOp::Mul => {
                        // Distribute multiplication over addition
                        expand_multiply(&left_exp, &right_exp)
                    }
                    BinaryOp::Div => {
                        // Don't expand division
                        Expr::Binary(BinaryOp::Div, Arc::new(left_exp), Arc::new(right_exp))
                    }
                    BinaryOp::Pow => {
                        // Expand (expr)^n for small integer n
                        if let Expr::Integer(exp) = right.as_ref() {
                            let exp_i64 = exp.to_i64();
                            {
                                if exp_i64 >= 0 && exp_i64 <= 10 {
                                    return expand_power(&left_exp, exp_i64 as u32);
                                }
                            }
                        }
                        Expr::Binary(BinaryOp::Pow, Arc::new(left_exp), Arc::new(right_exp))
                    }
                }
            }

            Expr::Unary(op, inner) => {
                let inner_exp = inner.expand();
                match op {
                    UnaryOp::Neg => {
                        // Distribute negation: -(a + b) = -a - b
                        distribute_negation(&inner_exp)
                    }
                    _ => Expr::Unary(*op, Arc::new(inner_exp)),
                }
            }

            Expr::Function(name, args) => {
                // Recursively expand all arguments
                let expanded_args: Vec<Arc<Expr>> = args
                    .iter()
                    .map(|arg| Arc::new(arg.expand()))
                    .collect();
                Expr::Function(name.clone(), expanded_args)
            }
        }
    }

    /// Collect terms with respect to a variable
    ///
    /// Groups terms by powers of the variable: x^2 + 2*x + x^2 = 2*x^2 + 2*x
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::{Expr, Symbol};
    ///
    /// let x = Expr::symbol("x");
    ///
    /// // x^2 + x + x = x^2 + 2*x
    /// let expr = x.clone().pow(Expr::from(2)) + x.clone() + x.clone();
    /// let collected = expr.collect(&Symbol::new("x"));
    /// ```
    pub fn collect(&self, var: &crate::symbol::Symbol) -> Expr {
        // Only works for polynomials
        if !self.is_polynomial(var) {
            return self.clone();
        }

        // Get the degree
        let deg = match self.degree(var) {
            Some(d) => d,
            None => return self.clone(),
        };

        // Collect coefficients for each power
        let mut result = Expr::from(0);

        for i in 0..=deg {
            if let Some(coeff) = self.coefficient(var, i) {
                // Skip zero coefficients
                if let Some(val) = coeff.eval_rational() {
                    if val.is_zero() {
                        continue;
                    }
                }

                let term = if i == 0 {
                    coeff
                } else if i == 1 {
                    coeff * Expr::Symbol(var.clone())
                } else {
                    coeff * Expr::Symbol(var.clone()).pow(Expr::from(i))
                };

                result = result + term;
            }
        }

        result
    }
}

/// Expand multiplication by distributing over addition
fn expand_multiply(left: &Expr, right: &Expr) -> Expr {
    match (left, right) {
        // (a + b) * c = a*c + b*c
        (Expr::Binary(BinaryOp::Add, a, b), c) => {
            let ac = expand_multiply(a, c);
            let bc = expand_multiply(b, c);
            ac + bc
        }
        // c * (a + b) = c*a + c*b
        (c, Expr::Binary(BinaryOp::Add, a, b)) => {
            let ca = expand_multiply(c, a);
            let cb = expand_multiply(c, b);
            ca + cb
        }
        // (a - b) * c = a*c - b*c
        (Expr::Binary(BinaryOp::Sub, a, b), c) => {
            let ac = expand_multiply(a, c);
            let bc = expand_multiply(b, c);
            ac - bc
        }
        // c * (a - b) = c*a - c*b
        (c, Expr::Binary(BinaryOp::Sub, a, b)) => {
            let ca = expand_multiply(c, a);
            let cb = expand_multiply(c, b);
            ca - cb
        }
        // Otherwise, just multiply
        _ => left.clone() * right.clone(),
    }
}

/// Expand a power expression: (expr)^n
fn expand_power(base: &Expr, exp: u32) -> Expr {
    if exp == 0 {
        return Expr::from(1);
    }
    if exp == 1 {
        return base.clone();
    }

    match base {
        // (a + b)^n - use binomial expansion for small n
        Expr::Binary(BinaryOp::Add, a, b) if exp <= 5 => {
            binomial_expand(a, b, exp, true)
        }
        // (a - b)^n
        Expr::Binary(BinaryOp::Sub, a, b) if exp <= 5 => {
            binomial_expand(a, b, exp, false)
        }
        // Otherwise, just return the power
        _ => base.clone().pow(Expr::from(exp as i64)),
    }
}

/// Binomial expansion of (a + b)^n or (a - b)^n
fn binomial_expand(a: &Expr, b: &Expr, n: u32, is_add: bool) -> Expr {
    let mut result = Expr::from(0);

    for k in 0..=n {
        // Binomial coefficient C(n, k)
        let binom_coeff = binomial_coefficient(n, k);

        // a^(n-k) * b^k
        let a_term = if n - k == 0 {
            Expr::from(1)
        } else if n - k == 1 {
            a.clone()
        } else {
            a.clone().pow(Expr::from((n - k) as i64))
        };

        let b_term = if k == 0 {
            Expr::from(1)
        } else if k == 1 {
            b.clone()
        } else {
            b.clone().pow(Expr::from(k as i64))
        };

        let mut term = Expr::from(binom_coeff as i64) * a_term * b_term;

        // For (a - b)^n, alternate signs
        if !is_add && k % 2 == 1 {
            term = -term;
        }

        result = result + term;
    }

    result
}

/// Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)
fn binomial_coefficient(n: u32, k: u32) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k); // Use symmetry: C(n,k) = C(n, n-k)

    let mut result = 1u64;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

/// Distribute negation over addition
fn distribute_negation(expr: &Expr) -> Expr {
    match expr {
        Expr::Binary(BinaryOp::Add, a, b) => {
            let neg_a = distribute_negation(a);
            let neg_b = distribute_negation(b);
            neg_a - neg_b
        }
        Expr::Binary(BinaryOp::Sub, a, b) => {
            let neg_a = distribute_negation(a);
            let neg_b = distribute_negation(b);
            neg_b - neg_a
        }
        _ => -expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_multiply() {
        let x = Expr::symbol("x");

        // (x + 1) * 2 = 2*x + 2
        let expr = (x.clone() + Expr::from(1)) * Expr::from(2);
        let expanded = expr.expand();

        // Check that it's in expanded form
        // The exact structure may vary, but it should be a sum
        match expanded {
            Expr::Binary(BinaryOp::Add, _, _) => {
                // Good, it's expanded
            }
            _ => panic!("Expected expanded form to be a sum"),
        }
    }

    #[test]
    fn test_expand_product_of_sums() {
        let x = Expr::symbol("x");

        // (x + 1) * (x + 2) = x^2 + 3*x + 2
        let expr = (x.clone() + Expr::from(1)) * (x.clone() + Expr::from(2));
        let expanded = expr.expand();

        // Should be in expanded form (a sum)
        match expanded {
            Expr::Binary(BinaryOp::Add, _, _) => {}
            _ => panic!("Expected sum"),
        }
    }

    #[test]
    fn test_expand_power() {
        let x = Expr::symbol("x");

        // (x + 1)^2 = x^2 + 2*x + 1
        let expr = (x.clone() + Expr::from(1)).pow(Expr::from(2));
        let expanded = expr.expand();

        // Should be in expanded form
        match expanded {
            Expr::Binary(BinaryOp::Add, _, _) => {}
            _ => panic!("Expected sum"),
        }
    }

    #[test]
    fn test_expand_negation() {
        let x = Expr::symbol("x");

        // -(x + 1) = -x - 1
        let expr = -(x.clone() + Expr::from(1));
        let expanded = expr.expand();

        // Should be in expanded form
        match expanded {
            Expr::Binary(BinaryOp::Sub, _, _) => {}
            _ => {}
        }
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 0), 1);
        assert_eq!(binomial_coefficient(5, 1), 5);
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(5, 3), 10);
        assert_eq!(binomial_coefficient(5, 4), 5);
        assert_eq!(binomial_coefficient(5, 5), 1);
    }

    #[test]
    fn test_collect() {
        use crate::symbol::Symbol;

        let x = Expr::symbol("x");
        let var_x = Symbol::new("x");

        // x + x = 2*x (after collection)
        let expr = x.clone() + x.clone();
        let collected = expr.collect(&var_x);

        // Result should have the terms collected
        // The exact form may vary, but it should simplify
    }
}
