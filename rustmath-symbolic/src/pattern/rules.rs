//! Pattern-based rewrite rules for simplification

use super::{Pattern, RewriteRule, Substitution};
use crate::expression::{BinaryOp, Expr, UnaryOp};
use std::sync::Arc;

/// Trigonometric simplification rule
#[derive(Clone)]
pub struct TrigRule {
    pub rule: RewriteRule,
}

/// Exponential/logarithm simplification rule
#[derive(Clone)]
pub struct ExpLogRule {
    pub rule: RewriteRule,
}

/// Database of rewrite rules
pub struct RuleDatabase {
    trig_rules: Vec<TrigRule>,
    exp_log_rules: Vec<ExpLogRule>,
}

impl RuleDatabase {
    /// Create a new rule database with all built-in rules
    pub fn new() -> Self {
        RuleDatabase {
            trig_rules: get_trig_rules(),
            exp_log_rules: get_exp_log_rules(),
        }
    }

    /// Apply all rules to an expression until no more changes occur
    pub fn apply_all(&self, expr: &Expr) -> Expr {
        let mut current = expr.clone();
        let mut changed = true;

        // Apply rules iteratively until fixed point
        while changed {
            let new_expr = self.apply_once(&current);
            changed = new_expr != current;
            current = new_expr;
        }

        current
    }

    /// Apply rules once (single pass)
    fn apply_once(&self, expr: &Expr) -> Expr {
        // Try trig rules first
        for trig_rule in &self.trig_rules {
            if let Some(result) = trig_rule.rule.apply(expr) {
                return result;
            }
        }

        // Then try exp/log rules
        for exp_log_rule in &self.exp_log_rules {
            if let Some(result) = exp_log_rule.rule.apply(expr) {
                return result;
            }
        }

        // Recursively apply to subexpressions
        match expr {
            Expr::Binary(op, left, right) => {
                let new_left = self.apply_once(left);
                let new_right = self.apply_once(right);
                Expr::Binary(*op, Arc::new(new_left), Arc::new(new_right))
            }
            Expr::Unary(op, inner) => {
                let new_inner = self.apply_once(inner);
                Expr::Unary(*op, Arc::new(new_inner))
            }
            Expr::Function(name, args) => {
                let new_args: Vec<Arc<Expr>> = args
                    .iter()
                    .map(|arg| Arc::new(self.apply_once(arg)))
                    .collect();
                Expr::Function(name.clone(), new_args)
            }
            _ => expr.clone(),
        }
    }
}

impl Default for RuleDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Get all trigonometric simplification rules
pub fn get_trig_rules() -> Vec<TrigRule> {
    let mut rules = Vec::new();

    // ===== Pythagorean Identities =====

    // sin(x)^2 + cos(x)^2 = 1
    let pattern = Pattern::add(
        Pattern::pow(Pattern::sin(Pattern::named("x")), Pattern::Integer(2)),
        Pattern::pow(Pattern::cos(Pattern::named("x")), Pattern::Integer(2)),
    );
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |_subst| Expr::from(1),
            "sin²(x) + cos²(x) = 1",
        ),
    });

    // cos(x)^2 + sin(x)^2 = 1 (reversed order)
    let pattern = Pattern::add(
        Pattern::pow(Pattern::cos(Pattern::named("x")), Pattern::Integer(2)),
        Pattern::pow(Pattern::sin(Pattern::named("x")), Pattern::Integer(2)),
    );
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |_subst| Expr::from(1),
            "cos²(x) + sin²(x) = 1",
        ),
    });

    // 1 + tan(x)^2 = sec(x)^2
    // Note: We don't have sec yet, so we express as 1/cos(x)^2
    let pattern = Pattern::add(
        Pattern::Integer(1),
        Pattern::pow(Pattern::unary(UnaryOp::Tan, Pattern::named("x")), Pattern::Integer(2)),
    );
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                Expr::from(1) / x.cos().pow(Expr::from(2))
            },
            "1 + tan²(x) = sec²(x) = 1/cos²(x)",
        ),
    });

    // 1 + cot(x)^2 = csc(x)^2 = 1/sin(x)^2
    // We'll add this when we have cot function

    // ===== Double Angle Formulas =====

    // sin(2*x) = 2*sin(x)*cos(x)
    let pattern = Pattern::sin(Pattern::mul(Pattern::Integer(2), Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                Expr::from(2) * x.clone().sin() * x.cos()
            },
            "sin(2x) = 2sin(x)cos(x)",
        ),
    });

    // cos(2*x) = cos(x)^2 - sin(x)^2
    let pattern = Pattern::cos(Pattern::mul(Pattern::Integer(2), Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                x.clone().cos().pow(Expr::from(2)) - x.sin().pow(Expr::from(2))
            },
            "cos(2x) = cos²(x) - sin²(x)",
        ),
    });

    // ===== Even/Odd Function Properties =====

    // sin(-x) = -sin(x)
    let pattern = Pattern::sin(Pattern::unary(UnaryOp::Neg, Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                -x.sin()
            },
            "sin(-x) = -sin(x)",
        ),
    });

    // cos(-x) = cos(x)
    let pattern = Pattern::cos(Pattern::unary(UnaryOp::Neg, Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                x.cos()
            },
            "cos(-x) = cos(x)",
        ),
    });

    // tan(-x) = -tan(x)
    let pattern = Pattern::unary(UnaryOp::Tan, Pattern::unary(UnaryOp::Neg, Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                -x.tan()
            },
            "tan(-x) = -tan(x)",
        ),
    });

    // ===== Special Values =====

    // sin(0) = 0
    let pattern = Pattern::sin(Pattern::Integer(0));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |_subst| Expr::from(0),
            "sin(0) = 0",
        ),
    });

    // cos(0) = 1
    let pattern = Pattern::cos(Pattern::Integer(0));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |_subst| Expr::from(1),
            "cos(0) = 1",
        ),
    });

    // tan(0) = 0
    let pattern = Pattern::unary(UnaryOp::Tan, Pattern::Integer(0));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |_subst| Expr::from(0),
            "tan(0) = 0",
        ),
    });

    // ===== Hyperbolic Identities =====

    // cosh(x)^2 - sinh(x)^2 = 1
    let pattern = Pattern::binary(
        BinaryOp::Sub,
        Pattern::pow(Pattern::unary(UnaryOp::Cosh, Pattern::named("x")), Pattern::Integer(2)),
        Pattern::pow(Pattern::unary(UnaryOp::Sinh, Pattern::named("x")), Pattern::Integer(2)),
    );
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |_subst| Expr::from(1),
            "cosh²(x) - sinh²(x) = 1",
        ),
    });

    // sinh(-x) = -sinh(x)
    let pattern = Pattern::unary(UnaryOp::Sinh, Pattern::unary(UnaryOp::Neg, Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                -x.sinh()
            },
            "sinh(-x) = -sinh(x)",
        ),
    });

    // cosh(-x) = cosh(x)
    let pattern = Pattern::unary(UnaryOp::Cosh, Pattern::unary(UnaryOp::Neg, Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                x.cosh()
            },
            "cosh(-x) = cosh(x)",
        ),
    });

    // ===== Inverse Function Compositions =====

    // sin(arcsin(x)) = x
    let pattern = Pattern::sin(Pattern::unary(UnaryOp::Arcsin, Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| subst.get("x").unwrap().clone(),
            "sin(arcsin(x)) = x",
        ),
    });

    // arcsin(sin(x)) = x (for x in [-π/2, π/2])
    let pattern = Pattern::unary(UnaryOp::Arcsin, Pattern::sin(Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| subst.get("x").unwrap().clone(),
            "arcsin(sin(x)) = x",
        ),
    });

    // cos(arccos(x)) = x
    let pattern = Pattern::cos(Pattern::unary(UnaryOp::Arccos, Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| subst.get("x").unwrap().clone(),
            "cos(arccos(x)) = x",
        ),
    });

    // arccos(cos(x)) = x (for x in [0, π])
    let pattern = Pattern::unary(UnaryOp::Arccos, Pattern::cos(Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| subst.get("x").unwrap().clone(),
            "arccos(cos(x)) = x",
        ),
    });

    // tan(arctan(x)) = x
    let pattern = Pattern::unary(UnaryOp::Tan, Pattern::unary(UnaryOp::Arctan, Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| subst.get("x").unwrap().clone(),
            "tan(arctan(x)) = x",
        ),
    });

    // arctan(tan(x)) = x (for x in (-π/2, π/2))
    let pattern = Pattern::unary(UnaryOp::Arctan, Pattern::unary(UnaryOp::Tan, Pattern::named("x")));
    rules.push(TrigRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| subst.get("x").unwrap().clone(),
            "arctan(tan(x)) = x",
        ),
    });

    rules
}

/// Get all exponential and logarithm simplification rules
pub fn get_exp_log_rules() -> Vec<ExpLogRule> {
    let mut rules = Vec::new();

    // ===== Exponential Rules =====

    // exp(0) = 1
    let pattern = Pattern::exp(Pattern::Integer(0));
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |_subst| Expr::from(1),
            "exp(0) = 1",
        ),
    });

    // exp(1) = e (we'll use a symbolic constant)
    // For now, leave as exp(1)

    // exp(x) * exp(y) = exp(x + y)
    let pattern = Pattern::mul(
        Pattern::exp(Pattern::named("x")),
        Pattern::exp(Pattern::named("y")),
    );
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                let y = subst.get("y").unwrap().clone();
                (x + y).exp()
            },
            "exp(x) * exp(y) = exp(x + y)",
        ),
    });

    // exp(x) / exp(y) = exp(x - y)
    let pattern = Pattern::binary(
        BinaryOp::Div,
        Pattern::exp(Pattern::named("x")),
        Pattern::exp(Pattern::named("y")),
    );
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                let y = subst.get("y").unwrap().clone();
                (x - y).exp()
            },
            "exp(x) / exp(y) = exp(x - y)",
        ),
    });

    // exp(x)^n = exp(n*x)
    let pattern = Pattern::pow(
        Pattern::exp(Pattern::named("x")),
        Pattern::named("n"),
    );
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                let n = subst.get("n").unwrap().clone();
                (n * x).exp()
            },
            "exp(x)^n = exp(n*x)",
        ),
    });

    // ===== Logarithm Rules =====

    // log(1) = 0
    let pattern = Pattern::log(Pattern::Integer(1));
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |_subst| Expr::from(0),
            "log(1) = 0",
        ),
    });

    // log(x * y) = log(x) + log(y)
    let pattern = Pattern::log(Pattern::mul(Pattern::named("x"), Pattern::named("y")));
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                let y = subst.get("y").unwrap().clone();
                x.log() + y.log()
            },
            "log(x * y) = log(x) + log(y)",
        ),
    });

    // log(x / y) = log(x) - log(y)
    let pattern = Pattern::log(Pattern::binary(
        BinaryOp::Div,
        Pattern::named("x"),
        Pattern::named("y"),
    ));
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                let y = subst.get("y").unwrap().clone();
                x.log() - y.log()
            },
            "log(x / y) = log(x) - log(y)",
        ),
    });

    // log(x^n) = n * log(x)
    let pattern = Pattern::log(Pattern::pow(Pattern::named("x"), Pattern::named("n")));
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                let n = subst.get("n").unwrap().clone();
                n * x.log()
            },
            "log(x^n) = n * log(x)",
        ),
    });

    // ===== Exponential-Logarithm Compositions =====

    // exp(log(x)) = x
    let pattern = Pattern::exp(Pattern::log(Pattern::named("x")));
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| subst.get("x").unwrap().clone(),
            "exp(log(x)) = x",
        ),
    });

    // log(exp(x)) = x
    let pattern = Pattern::log(Pattern::exp(Pattern::named("x")));
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| subst.get("x").unwrap().clone(),
            "log(exp(x)) = x",
        ),
    });

    // ===== Square Root Rules =====

    // sqrt(x^2) = |x| (we'll simplify to abs(x))
    // Note: with assumptions (x > 0), this becomes x
    let pattern = Pattern::sqrt(Pattern::pow(Pattern::named("x"), Pattern::Integer(2)));
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                x.abs()
            },
            "sqrt(x^2) = |x|",
        ),
    });

    // sqrt(x) * sqrt(y) = sqrt(x * y)
    let pattern = Pattern::mul(
        Pattern::sqrt(Pattern::named("x")),
        Pattern::sqrt(Pattern::named("y")),
    );
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                let y = subst.get("y").unwrap().clone();
                (x * y).sqrt()
            },
            "sqrt(x) * sqrt(y) = sqrt(x * y)",
        ),
    });

    // sqrt(x / y) = sqrt(x) / sqrt(y)
    let pattern = Pattern::sqrt(Pattern::binary(
        BinaryOp::Div,
        Pattern::named("x"),
        Pattern::named("y"),
    ));
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| {
                let x = subst.get("x").unwrap().clone();
                let y = subst.get("y").unwrap().clone();
                x.sqrt() / y.sqrt()
            },
            "sqrt(x / y) = sqrt(x) / sqrt(y)",
        ),
    });

    // sqrt(x)^2 = x (for x >= 0)
    let pattern = Pattern::pow(Pattern::sqrt(Pattern::named("x")), Pattern::Integer(2));
    rules.push(ExpLogRule {
        rule: RewriteRule::with_description(
            pattern,
            |subst| subst.get("x").unwrap().clone(),
            "sqrt(x)^2 = x",
        ),
    });

    rules
}

/// Apply a set of rules to an expression
pub fn apply_rules(expr: &Expr, rules: &[RewriteRule]) -> Expr {
    for rule in rules {
        if let Some(result) = rule.apply(expr) {
            return result;
        }
    }
    expr.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Expr;

    #[test]
    fn test_pythagorean_identity() {
        let rules = get_trig_rules();
        let db = RuleDatabase::new();

        // sin(x)^2 + cos(x)^2 should simplify to 1
        let x = Expr::symbol("x");
        let expr = x.clone().sin().pow(Expr::from(2)) + x.clone().cos().pow(Expr::from(2));

        let simplified = db.apply_all(&expr);
        assert_eq!(simplified, Expr::from(1));
    }

    #[test]
    fn test_exp_log_composition() {
        let db = RuleDatabase::new();

        // exp(log(x)) should simplify to x
        let x = Expr::symbol("x");
        let expr = x.clone().log().exp();

        let simplified = db.apply_all(&expr);
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_log_product_rule() {
        let db = RuleDatabase::new();

        // log(x * y) should expand to log(x) + log(y)
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let expr = (x.clone() * y.clone()).log();

        let simplified = db.apply_all(&expr);
        let expected = x.log() + y.log();
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_sin_arcsin() {
        let db = RuleDatabase::new();

        // sin(arcsin(x)) should simplify to x
        let x = Expr::symbol("x");
        let expr = x.clone().arcsin().sin();

        let simplified = db.apply_all(&expr);
        assert_eq!(simplified, x);
    }
}
