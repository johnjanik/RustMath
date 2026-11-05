//! Symbolic integration
//!
//! Integration runs in three tiers (see [`Expr::integrate`] for the exact
//! contract):
//!
//! 1. **Decided — rational functions** ([`crate::risch`]): any integrand
//!    that normalizes to a univariate rational function over ℚ is integrated
//!    completely and exactly (Hermite reduction + Rothstein–Trager), with
//!    the defining identities re-verified by exact algebra before returning.
//! 2. **Decided — exact gated patterns** ([`crate::integrate_exact`]):
//!    `c·u′/u ↦ c·log u`, `p(x)·exp(ax+b)`, and `p(x)·sin/cos(ax+b)`
//!    (`p ∈ ℚ[x]`, `a, b ∈ ℚ`), each certified by differentiating the
//!    candidate back and deciding `F′ − f ≡ 0` with exact algebra.
//! 3. **Heuristic — the legacy table** (this module): pattern matching for
//!    common integrals (trig/hyperbolic tables, sqrt forms, integration by
//!    parts, shallow partial fractions). These results are *not* verified;
//!    the tier is kept for coverage beyond the decided classes.
//!
//! `None` means honest refusal: no tier could produce (tiers 1–2: certify)
//! an antiderivative. Non-elementary integrands such as `exp(x)/x`,
//! `1/log(x)`, `exp(x²)`, `sin(x)/x` are refused, never approximated.

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;

impl Expr {
    /// Integrate the expression with respect to a symbol.
    ///
    /// # What is decided, what is heuristic, what is refused
    ///
    /// - **Decided (complete, exact, self-verifying):** univariate rational
    ///   functions over ℚ — every such integrand succeeds, routed through
    ///   [`crate::risch::integrate_rational_risch`]. When some logarithm
    ///   coefficients are irrational algebraic numbers the result contains
    ///   the symbolic `Function("root_sum", [f(τ), τ·log v(x,τ)])` object
    ///   (see the `crate::risch` module docs); [`Expr::differentiate`]
    ///   applies the exact rule `d/dx root_sum(f, g) = root_sum(f, ∂g/∂x)`
    ///   to it, and such results are additionally certified by the Risch
    ///   module's internal exact gates. The tier runs under the default
    ///   [`crate::risch::RischBudget`]: on inputs whose exact answer is too
    ///   expensive (e.g. very high multiplicities, or log terms needing a
    ///   high-degree algebraic number field) it steps aside with a labeled
    ///   `BudgetExceeded` and the remaining tiers are tried exactly as if
    ///   this tier had declined — the ultimate `None` is then the honest
    ///   "not integrated". Callers who want the exact answer at a higher
    ///   price can invoke
    ///   [`crate::risch::integrate_rational_risch_with_budget`] directly.
    /// - **Decided (exact, differentiate-back-gated):** `c·u′/u ↦ c·log u`
    ///   with `c ∈ ℚ`; `p(x)·exp(ax+b)`; `p(x)·sin(ax+b)`; `p(x)·cos(ax+b)`
    ///   with `p ∈ ℚ[x]`, `a, b ∈ ℚ`, `a ≠ 0` — see
    ///   [`crate::integrate_exact::integrate_exact_patterns`].
    /// - **Heuristic (unverified):** the legacy table below — linearity,
    ///   power rule (including non-integer exponents), elementary function
    ///   table (`sin`, `cos`, `exp`, `log`, `tan`, `sinh`, `cosh`),
    ///   `1/sqrt(a²±x²)` forms, LIATE integration by parts, and shallow
    ///   partial-fraction patterns. Results in this tier are plausible
    ///   forms, not certified ones.
    /// - **Refused (`None`):** everything else, including non-elementary
    ///   integrands (`exp(x)/x`, `1/log(x)`, `exp(x²)`, `sin(x)/x`, ...).
    ///   `None` is correct behavior, never a bug workaround.
    pub fn integrate(&self, var: &Symbol) -> Option<Self> {
        // Tier 1: the rational front door. Complete for its class within the
        // default budget, so a rational integrand only falls through to the
        // heuristics on an explicit resource refusal (BudgetExceeded) — in
        // which case the remaining tiers run exactly as if tier 1 declined,
        // and a final None is the honest "not integrated" of the Option
        // contract (never a hang, never a fabricated answer).
        match crate::risch::integrate_rational_risch(self, var) {
            crate::risch::RischResult::Elementary(f)
            | crate::risch::RischResult::WithRootSum(f) => return Some(f),
            crate::risch::RischResult::NotRational
            | crate::risch::RischResult::BudgetExceeded { .. } => {}
        }
        // Tier 2: exact, differentiate-back-gated patterns.
        if let Some(f) = crate::integrate_exact::integrate_exact_patterns(self, var) {
            return Some(f);
        }
        // Tier 3: the legacy heuristic table.
        self.integrate_table(var)
    }

    /// The legacy heuristic integration table (tier 3 of [`Expr::integrate`]).
    /// Pattern-matched plausible forms, kept unchanged for coverage beyond
    /// the decided classes; results are not verified.
    fn integrate_table(&self, var: &Symbol) -> Option<Self> {
        match self {
            // ∫ c dx = c*x for constants (simplified so e.g. ∫1 dx = x, not 1*x)
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => {
                Some((self.clone() * Expr::Symbol(var.clone())).simplify())
            }

            // ∫ x dx = x²/2
            Expr::Symbol(s) => {
                if s == var {
                    // ∫ x dx = x²/2
                    let x = Expr::Symbol(var.clone());
                    Some(x.pow(Expr::from(2)) / Expr::from(2))
                } else {
                    // ∫ y dx = y*x for y ≠ x
                    Some(self.clone() * Expr::Symbol(var.clone()))
                }
            }

            // Binary operations
            Expr::Binary(op, left, right) => match op {
                // Linearity: ∫ (f + g) dx = ∫f dx + ∫g dx
                BinaryOp::Add => {
                    let left_integral = left.integrate(var)?;
                    let right_integral = right.integrate(var)?;
                    Some(left_integral + right_integral)
                }

                // Linearity: ∫ (f - g) dx = ∫f dx - ∫g dx
                BinaryOp::Sub => {
                    let left_integral = left.integrate(var)?;
                    let right_integral = right.integrate(var)?;
                    Some(left_integral - right_integral)
                }

                // Modulo: not integrable in standard sense
                BinaryOp::Mod => None,

                // Constant multiple: ∫ c*f dx = c*∫f dx
                BinaryOp::Mul => {
                    if left.is_constant() && !left.contains_symbol(var) {
                        let integral = right.integrate(var)?;
                        Some((**left).clone() * integral)
                    } else if right.is_constant() && !right.contains_symbol(var) {
                        let integral = left.integrate(var)?;
                        Some(integral * (**right).clone())
                    } else {
                        // Try integration by parts for products
                        advanced::integrate_by_parts_auto(self, var)
                    }
                }

                // Division: handle special cases
                BinaryOp::Div => {
                    // ∫ 1/x dx = log(|x|)
                    if left.is_one() && matches!(**right, Expr::Symbol(ref s) if s == var) {
                        Some(Expr::Symbol(var.clone()).log())
                    }
                    // ∫ f/c dx = (1/c)*∫f dx for constant c
                    else if right.is_constant() && !right.contains_symbol(var) {
                        let integral = left.integrate(var)?;
                        Some(integral / (**right).clone())
                    }
                    // Try trigonometric substitution for expressions with sqrt
                    else if matches!(**right, Expr::Unary(UnaryOp::Sqrt, _)) {
                        advanced::integrate_with_sqrt(self, var)
                            .or_else(|| advanced::try_trig_substitution(self, var))
                    }
                    // Try partial fractions for rational functions
                    else {
                        advanced::integrate_rational(left, right, var)
                    }
                }

                // Power rule: ∫ x^n dx = x^(n+1)/(n+1)
                BinaryOp::Pow => {
                    if matches!(**left, Expr::Symbol(ref s) if s == var) && right.is_constant() {
                        // Check if exponent is -1
                        if right.is_minus_one() {
                            // ∫ x^(-1) dx = log(|x|)
                            Some(Expr::Symbol(var.clone()).log())
                        } else {
                            // ∫ x^n dx = x^(n+1)/(n+1)
                            let n = (**right).clone();
                            let x = Expr::Symbol(var.clone());
                            let exponent = (n.clone() + Expr::from(1)).simplify();
                            Some(x.pow(exponent.clone()) / exponent)
                        }
                    } else {
                        None
                    }
                }
            },

            // Unary operations
            Expr::Unary(op, inner) => {
                // Simple cases where inner is just the variable
                if matches!(**inner, Expr::Symbol(ref s) if s == var) {
                    match op {
                        // ∫ sin(x) dx = -cos(x)
                        UnaryOp::Sin => {
                            Some(-Expr::Symbol(var.clone()).cos())
                        }

                        // ∫ cos(x) dx = sin(x)
                        UnaryOp::Cos => {
                            Some(Expr::Symbol(var.clone()).sin())
                        }

                        // ∫ exp(x) dx = exp(x)
                        UnaryOp::Exp => {
                            Some(Expr::Symbol(var.clone()).exp())
                        }

                        // ∫ 1/x dx is handled in division
                        // ∫ log(x) dx = x*log(x) - x
                        UnaryOp::Log => {
                            let x = Expr::Symbol(var.clone());
                            Some(x.clone() * x.clone().log() - x)
                        }

                        // ∫ sinh(x) dx = cosh(x)
                        UnaryOp::Sinh => {
                            Some(Expr::Symbol(var.clone()).cosh())
                        }

                        // ∫ cosh(x) dx = sinh(x)
                        UnaryOp::Cosh => {
                            Some(Expr::Symbol(var.clone()).sinh())
                        }

                        // ∫ tan(x) dx = -log(|cos(x)|)
                        UnaryOp::Tan => {
                            Some(-Expr::Symbol(var.clone()).cos().log())
                        }

                        _ => None,
                    }
                } else {
                    // For composite functions, we'd need substitution
                    // This is more complex and not implemented yet
                    None
                }
            }

            Expr::Function(_, _) => None,
        }
    }

    /// Definite integral from a to b
    ///
    /// Uses the fundamental theorem of calculus: ∫[a,b] f(x) dx = F(b) - F(a)
    /// where F is the antiderivative of f
    pub fn integrate_definite(&self, var: &Symbol, a: &Expr, b: &Expr) -> Option<Self> {
        // Find the antiderivative
        let antiderivative = self.integrate(var)?;

        // Evaluate at b and a
        let fb = antiderivative.substitute(var, b);
        let fa = antiderivative.substitute(var, a);

        Some((fb - fa).simplify())
    }

    /// Check if expression equals 1
    fn is_one(&self) -> bool {
        matches!(self, Expr::Integer(i) if i.to_i64() == 1)
    }

    /// Check if expression equals -1
    fn is_minus_one(&self) -> bool {
        matches!(self, Expr::Integer(i) if i.to_i64() == -1)
    }

    /// Integration by parts: ∫ u dv = uv - ∫ v du
    ///
    /// # Arguments
    ///
    /// * `u` - The u function
    /// * `dv` - The dv function (derivative already taken)
    /// * `var` - Variable of integration
    ///
    /// # Returns
    ///
    /// The result of integration by parts if successful
    pub fn integrate_by_parts(u: &Expr, dv: &Expr, var: &Symbol) -> Option<Expr> {
        // Compute du
        let du = u.differentiate(var);

        // Integrate dv to get v
        let v = dv.integrate(var)?;

        // Compute ∫ v du
        let v_du = (v.clone() * du).integrate(var)?;

        // Return uv - ∫ v du
        Some(u.clone() * v - v_du)
    }

    /// Try common integration by parts patterns
    ///
    /// Automatically selects u and dv for common patterns
    pub fn try_integration_by_parts(&self, var: &Symbol) -> Option<Expr> {
        match self {
            // ∫ x * sin(x) dx
            Expr::Binary(BinaryOp::Mul, left, right) => {
                // Try left as u, right as dv
                if let Some(result) = Self::integrate_by_parts(left, right, var) {
                    return Some(result);
                }

                // Try right as u, left as dv
                if let Some(result) = Self::integrate_by_parts(right, left, var) {
                    return Some(result);
                }

                None
            }
            _ => None,
        }
    }

    /// Integration by substitution: if f(x) = g(u(x)) * u'(x), then ∫f dx = ∫g(u) du
    ///
    /// # Arguments
    ///
    /// * `substitution` - The substitution u = substitution(x)
    /// * `var` - Original variable
    /// * `new_var` - New variable name
    ///
    /// # Returns
    ///
    /// The integral in terms of the new variable
    pub fn integrate_with_substitution(
        &self,
        substitution: &Expr,
        var: &Symbol,
        new_var: &Symbol,
    ) -> Option<Expr> {
        // Compute du/dx
        let _du_dx = substitution.differentiate(var);

        // Try to write integrand as g(u) * du/dx
        // This is a simplified version - full implementation would need pattern matching

        // For now, handle simple case: ∫ f(g(x)) * g'(x) dx
        // Substitute u for g(x) everywhere
        let mut transformed = self.clone();

        // Replace substitution with new_var
        // This is a placeholder - full implementation would be more sophisticated
        transformed = transformed.substitute(var, &Expr::Symbol(new_var.clone()));

        // Integrate with respect to new variable
        let integrated = transformed.integrate(new_var)?;

        // Substitute back
        Some(integrated.substitute(new_var, substitution))
    }

    /// Double integral: ∫∫ f(x,y) dy dx over a rectangular region
    ///
    /// Computes ∫[x_min, x_max] ∫[y_min, y_max] f(x,y) dy dx
    ///
    /// Uses Fubini's theorem: integrate with respect to y first, then x
    pub fn integrate_double(
        &self,
        x_var: &Symbol,
        y_var: &Symbol,
        x_min: &Expr,
        x_max: &Expr,
        y_min: &Expr,
        y_max: &Expr,
    ) -> Option<Expr> {
        // First integrate with respect to y
        let inner_integral = self.integrate(y_var)?;

        // Evaluate the inner integral at the y bounds
        let inner_at_y_max = inner_integral.substitute(y_var, y_max);
        let inner_at_y_min = inner_integral.substitute(y_var, y_min);
        let inner_result = (inner_at_y_max - inner_at_y_min).simplify();

        // Now integrate with respect to x
        let outer_integral = inner_result.integrate(x_var)?;

        // Evaluate at the x bounds
        let result_at_x_max = outer_integral.substitute(x_var, x_max);
        let result_at_x_min = outer_integral.substitute(x_var, x_min);

        Some((result_at_x_max - result_at_x_min).simplify())
    }

    /// Triple integral: ∫∫∫ f(x,y,z) dz dy dx over a rectangular box
    ///
    /// Computes ∫[x_min, x_max] ∫[y_min, y_max] ∫[z_min, z_max] f(x,y,z) dz dy dx
    pub fn integrate_triple(
        &self,
        x_var: &Symbol,
        y_var: &Symbol,
        z_var: &Symbol,
        x_min: &Expr,
        x_max: &Expr,
        y_min: &Expr,
        y_max: &Expr,
        z_min: &Expr,
        z_max: &Expr,
    ) -> Option<Expr> {
        // Integrate with respect to z
        let integral_z = self.integrate(z_var)?;
        let result_z =
            (integral_z.substitute(z_var, z_max) - integral_z.substitute(z_var, z_min)).simplify();

        // Integrate with respect to y
        let integral_y = result_z.integrate(y_var)?;
        let result_y =
            (integral_y.substitute(y_var, y_max) - integral_y.substitute(y_var, y_min)).simplify();

        // Integrate with respect to x
        let integral_x = result_y.integrate(x_var)?;
        let result_x =
            (integral_x.substitute(x_var, x_max) - integral_x.substitute(x_var, x_min)).simplify();

        Some(result_x)
    }

    /// Change of variables for double integrals
    ///
    /// When transforming from (u,v) to (x,y), the Jacobian determinant is needed:
    /// ∫∫ f(x,y) dx dy = ∫∫ f(x(u,v), y(u,v)) |J| du dv
    ///
    /// where J = ∂(x,y)/∂(u,v) is the Jacobian determinant
    pub fn jacobian_2d(
        x_of_uv: &Expr,
        y_of_uv: &Expr,
        u_var: &Symbol,
        v_var: &Symbol,
    ) -> Expr {
        // Compute partial derivatives
        let dx_du = x_of_uv.differentiate(u_var);
        let dx_dv = x_of_uv.differentiate(v_var);
        let dy_du = y_of_uv.differentiate(u_var);
        let dy_dv = y_of_uv.differentiate(v_var);

        // Jacobian determinant: (∂x/∂u)(∂y/∂v) - (∂x/∂v)(∂y/∂u)
        (dx_du * dy_dv - dx_dv * dy_du).simplify()
    }
}

/// Advanced integration techniques
pub mod advanced {
    use super::*;

    // ========================================================================
    // Phase 3.1 Enhancements: Advanced Integration Techniques
    // ========================================================================

    /// Trigonometric substitution type
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TrigSubstitution {
        /// x = a sin(θ) for sqrt(a² - x²)
        Sine,
        /// x = a tan(θ) for sqrt(a² + x²)
        Tangent,
        /// x = a sec(θ) for sqrt(x² - a²)
        Secant,
    }

    /// Detect which trigonometric substitution to use
    ///
    /// Analyzes expressions involving sqrt to determine the appropriate substitution:
    /// - sqrt(a² - x²) → x = a sin(θ)
    /// - sqrt(a² + x²) → x = a tan(θ)
    /// - sqrt(x² - a²) → x = a sec(θ)
    pub fn detect_trig_substitution(expr: &Expr, var: &Symbol) -> Option<(TrigSubstitution, Expr)> {
        // Look for sqrt patterns anywhere in the expression
        detect_trig_substitution_recursive(expr, var)
    }

    /// Recursive helper to find sqrt patterns
    fn detect_trig_substitution_recursive(expr: &Expr, var: &Symbol) -> Option<(TrigSubstitution, Expr)> {
        match expr {
            Expr::Unary(UnaryOp::Sqrt, inner) => {
                match &**inner {
                    // sqrt(a² - x²) or sqrt(x² - a²)
                    Expr::Binary(BinaryOp::Sub, left, right) => {
                        // Check if right is x² -> sqrt(a² - x²)
                        if is_square_of_var(right, var) {
                            // left should be a constant a²
                            return Some((TrigSubstitution::Sine, (**left).clone()));
                        }
                        // Check if left is x² -> sqrt(x² - a²)
                        else if is_square_of_var(left, var) {
                            return Some((TrigSubstitution::Secant, (**right).clone()));
                        }
                    }
                    // sqrt(a² + x²)
                    Expr::Binary(BinaryOp::Add, left, right) => {
                        // Check if one is x²
                        if is_square_of_var(right, var) {
                            return Some((TrigSubstitution::Tangent, (**left).clone()));
                        } else if is_square_of_var(left, var) {
                            return Some((TrigSubstitution::Tangent, (**right).clone()));
                        }
                    }
                    _ => {}
                }
            }
            Expr::Binary(_, left, right) => {
                // Search in left subtree
                if let Some(result) = detect_trig_substitution_recursive(left, var) {
                    return Some(result);
                }
                // Search in right subtree
                if let Some(result) = detect_trig_substitution_recursive(right, var) {
                    return Some(result);
                }
            }
            Expr::Unary(_, inner) => {
                return detect_trig_substitution_recursive(inner, var);
            }
            _ => {}
        }

        None
    }

    /// Check if an expression is x²
    fn is_square_of_var(expr: &Expr, var: &Symbol) -> bool {
        matches!(expr,
            Expr::Binary(BinaryOp::Pow, x, two)
            if matches!(**x, Expr::Symbol(ref s) if s == var)
                && matches!(**two, Expr::Integer(ref i) if i.to_i64() == 2)
        )
    }

    /// Try trigonometric substitution automatically
    ///
    /// Detects the appropriate substitution and applies it
    pub fn try_trig_substitution(expr: &Expr, var: &Symbol) -> Option<Expr> {
        // Detect which substitution to use
        let (sub_type, a_squared) = detect_trig_substitution(expr, var)?;

        // Apply the substitution
        apply_trig_substitution(expr, var, sub_type, &a_squared)
    }

    /// Perform trigonometric substitution
    ///
    /// Transforms the integral using the appropriate trig substitution
    ///
    /// # Common Patterns
    ///
    /// For sqrt(a² - x²):
    /// - ∫ 1/sqrt(a² - x²) dx = arcsin(x/a)
    /// - ∫ sqrt(a² - x²) dx = (x/2)sqrt(a² - x²) + (a²/2)arcsin(x/a)
    /// - ∫ x²/sqrt(a² - x²) dx = -(x/2)sqrt(a² - x²) + (a²/2)arcsin(x/a)
    ///
    /// For sqrt(a² + x²):
    /// - ∫ 1/sqrt(a² + x²) dx = arcsinh(x/a) = log(x + sqrt(x² + a²))
    /// - ∫ sqrt(a² + x²) dx = (x/2)sqrt(a² + x²) + (a²/2)arcsinh(x/a)
    ///
    /// For sqrt(x² - a²):
    /// - ∫ 1/sqrt(x² - a²) dx = arccosh(x/a) = log(x + sqrt(x² - a²))
    /// - ∫ sqrt(x² - a²) dx = (x/2)sqrt(x² - a²) - (a²/2)arccosh(x/a)
    pub fn apply_trig_substitution(
        expr: &Expr,
        var: &Symbol,
        sub_type: TrigSubstitution,
        a_squared: &Expr,
    ) -> Option<Expr> {
        let x = Expr::Symbol(var.clone());
        let a = a_squared.clone().sqrt();

        match sub_type {
            TrigSubstitution::Sine => {
                // Pattern: expressions with sqrt(a² - x²)

                // ∫ 1/sqrt(a² - x²) dx = arcsin(x/a)
                if let Expr::Binary(BinaryOp::Div, num, den) = expr {
                    if num.is_one() {
                        if let Expr::Unary(UnaryOp::Sqrt, inner) = &**den {
                            if let Expr::Binary(BinaryOp::Sub, left, right) = &**inner {
                                if **left == *a_squared && is_square_of_var(right, var) {
                                    return Some((x.clone() / a).arcsin());
                                }
                            }
                        }
                    }
                }

                // ∫ sqrt(a² - x²) dx = (x/2)sqrt(a² - x²) + (a²/2)arcsin(x/a)
                if let Expr::Unary(UnaryOp::Sqrt, inner) = expr {
                    if let Expr::Binary(BinaryOp::Sub, left, right) = &**inner {
                        if **left == *a_squared && is_square_of_var(right, var) {
                            let sqrt_term = expr.clone();
                            let arcsin_term = (x.clone() / a.clone()).arcsin();
                            return Some(
                                (x.clone() / Expr::from(2)) * sqrt_term +
                                (a_squared.clone() / Expr::from(2)) * arcsin_term
                            );
                        }
                    }
                }
            }

            TrigSubstitution::Tangent => {
                // Pattern: expressions with sqrt(a² + x²)

                // ∫ 1/sqrt(a² + x²) dx = arcsinh(x/a)
                if let Expr::Binary(BinaryOp::Div, num, den) = expr {
                    if num.is_one() {
                        if let Expr::Unary(UnaryOp::Sqrt, inner) = &**den {
                            if let Expr::Binary(BinaryOp::Add, left, right) = &**inner {
                                let (const_part, _var_part) = if is_square_of_var(right, var) {
                                    (left, right)
                                } else if is_square_of_var(left, var) {
                                    (right, left)
                                } else {
                                    return None;
                                };

                                if **const_part == *a_squared {
                                    return Some((x.clone() / a).arcsinh());
                                }
                            }
                        }
                    }
                }

                // ∫ sqrt(a² + x²) dx = (x/2)sqrt(a² + x²) + (a²/2)arcsinh(x/a)
                if let Expr::Unary(UnaryOp::Sqrt, inner) = expr {
                    if let Expr::Binary(BinaryOp::Add, left, right) = &**inner {
                        let const_part = if is_square_of_var(right, var) {
                            left
                        } else if is_square_of_var(left, var) {
                            right
                        } else {
                            return None;
                        };

                        if **const_part == *a_squared {
                            let sqrt_term = expr.clone();
                            let arcsinh_term = (x.clone() / a.clone()).arcsinh();
                            return Some(
                                (x.clone() / Expr::from(2)) * sqrt_term +
                                (a_squared.clone() / Expr::from(2)) * arcsinh_term
                            );
                        }
                    }
                }
            }

            TrigSubstitution::Secant => {
                // Pattern: expressions with sqrt(x² - a²)

                // ∫ 1/sqrt(x² - a²) dx = arccosh(x/a)
                if let Expr::Binary(BinaryOp::Div, num, den) = expr {
                    if num.is_one() {
                        if let Expr::Unary(UnaryOp::Sqrt, inner) = &**den {
                            if let Expr::Binary(BinaryOp::Sub, left, right) = &**inner {
                                if is_square_of_var(left, var) && **right == *a_squared {
                                    return Some((x.clone() / a).arccosh());
                                }
                            }
                        }
                    }
                }

                // ∫ sqrt(x² - a²) dx = (x/2)sqrt(x² - a²) - (a²/2)arccosh(x/a)
                if let Expr::Unary(UnaryOp::Sqrt, inner) = expr {
                    if let Expr::Binary(BinaryOp::Sub, left, right) = &**inner {
                        if is_square_of_var(left, var) && **right == *a_squared {
                            let sqrt_term = expr.clone();
                            let arccosh_term = (x.clone() / a.clone()).arccosh();
                            return Some(
                                (x.clone() / Expr::from(2)) * sqrt_term -
                                (a_squared.clone() / Expr::from(2)) * arccosh_term
                            );
                        }
                    }
                }
            }
        }

        None
    }

    /// Polynomial degree helper
    pub fn polynomial_degree(expr: &Expr, var: &Symbol) -> Option<i64> {
        match expr {
            Expr::Integer(_) | Expr::Rational(_) => Some(0),
            Expr::Symbol(s) if s == var => Some(1),
            Expr::Symbol(_) => Some(0), // Other variables are constants
            Expr::Binary(BinaryOp::Pow, base, exp) => {
                if matches!(**base, Expr::Symbol(ref s) if s == var) {
                    if let Expr::Integer(n) = &**exp {
                        Some(n.to_i64())
                    } else {
                        None
                    }
                } else {
                    Some(0)
                }
            }
            Expr::Binary(BinaryOp::Add, left, right) | Expr::Binary(BinaryOp::Sub, left, right) => {
                let deg_left = polynomial_degree(left, var)?;
                let deg_right = polynomial_degree(right, var)?;
                Some(deg_left.max(deg_right))
            }
            Expr::Binary(BinaryOp::Mul, left, right) => {
                let deg_left = polynomial_degree(left, var)?;
                let deg_right = polynomial_degree(right, var)?;
                Some(deg_left + deg_right)
            }
            _ => None,
        }
    }

    /// Partial fraction decomposition for rational functions
    ///
    /// Decomposes P(x)/Q(x) into simpler fractions that can be integrated separately
    ///
    /// # Algorithm
    ///
    /// 1. If deg(P) >= deg(Q), perform polynomial long division
    /// 2. Factor the denominator Q(x) = (x - r₁)^m₁ × ... × (x - rₖ)^mₖ
    /// 3. Write as sum: A₁/(x-r₁) + A₂/(x-r₁)² + ... + B₁/(x-r₂) + ...
    /// 4. Solve for coefficients A₁, A₂, ..., B₁, ...
    ///
    /// # Implementation Status
    ///
    /// This is an enhanced implementation that handles:
    /// - Simple linear factors in denominator
    /// - Polynomial long division for improper fractions
    /// - Basic factorization patterns
    pub fn partial_fractions(numerator: &Expr, denominator: &Expr, var: &Symbol) -> Option<Vec<Expr>> {
        // Check if we need polynomial long division
        let deg_num = polynomial_degree(numerator, var)?;
        let deg_den = polynomial_degree(denominator, var)?;

        // If deg(numerator) >= deg(denominator), we need long division first
        // For now, we handle the case where they're already proper fractions

        // Handle common patterns:

        // Pattern 1: A/(x + b) - already in simplest form
        if numerator.is_constant() {
            // Check for (ax + b) in denominator
            if let Expr::Binary(BinaryOp::Add, left, right) = denominator {
                if matches!(**left, Expr::Symbol(ref s) if s == var) && right.is_constant() {
                    return Some(vec![numerator.clone() / denominator.clone()]);
                }
                if let Expr::Binary(BinaryOp::Mul, coeff, x) = &**left {
                    if coeff.is_constant() && matches!(**x, Expr::Symbol(ref s) if s == var) && right.is_constant() {
                        return Some(vec![numerator.clone() / denominator.clone()]);
                    }
                }
            }
            // Check for (x - b)
            if let Expr::Binary(BinaryOp::Sub, left, right) = denominator {
                if matches!(**left, Expr::Symbol(ref s) if s == var) && right.is_constant() {
                    return Some(vec![numerator.clone() / denominator.clone()]);
                }
            }
        }

        // Pattern 2: A/(x² + bx + c) - try to factor
        // For quadratic denominators, we could factor if possible
        // For example: 1/(x² - 1) = 1/2 * (1/(x-1) - 1/(x+1))
        if numerator.is_constant() && deg_den == 2 {
            // Try to factor x² - a²
            if let Expr::Binary(BinaryOp::Sub, x_sq, a_sq) = denominator {
                if is_square_of_var(x_sq, var) && a_sq.is_constant() {
                    // Factor as (x - a)(x + a)
                    let a = (**a_sq).clone().sqrt();
                    let x = Expr::Symbol(var.clone());

                    // 1/(x² - a²) = 1/(2a) * (1/(x-a) - 1/(x+a))
                    let two_a = Expr::from(2) * a.clone();
                    let term1 = numerator.clone() / (x.clone() - a.clone()) / two_a.clone();
                    let term2 = numerator.clone() / (x.clone() + a.clone()) / two_a;
                    return Some(vec![term1, -term2]);
                }
            }

            // Try to factor x² + a²
            // This would give complex roots, which we handle differently
            // For integration purposes: ∫ 1/(x² + a²) dx = (1/a) arctan(x/a)
        }

        // Pattern 3: (Ax + B)/(x² + c) - can be split
        if deg_num == 1 && deg_den == 2 {
            // Split into A*x/(x²+c) + B/(x²+c)
            // The first integrates to (A/2)log(x²+c)
            // The second uses arctan

            // This is handled better directly in integrate_rational
        }

        // For more complex cases, would need full factorization
        // Return None to fall back to other integration methods
        None
    }

    /// Integrate rational function using partial fractions
    pub fn integrate_rational_with_partial_fractions(
        numerator: &Expr,
        denominator: &Expr,
        var: &Symbol,
    ) -> Option<Expr> {
        // Get partial fraction decomposition
        let fractions = partial_fractions(numerator, denominator, var)?;

        // Integrate each fraction
        let mut result = Expr::from(0);
        for fraction in fractions {
            let integral = fraction.integrate(var)?;
            result = result + integral;
        }

        Some(result)
    }

    /// Enhanced pattern matching for integration by parts
    ///
    /// Automatically chooses u and dv based on the LIATE rule:
    /// - L: Logarithmic functions
    /// - I: Inverse trigonometric functions
    /// - A: Algebraic functions (polynomials)
    /// - T: Trigonometric functions
    /// - E: Exponential functions
    ///
    /// Choose u in order of priority L > I > A > T > E
    pub fn integrate_by_parts_auto(expr: &Expr, var: &Symbol) -> Option<Expr> {
        if let Expr::Binary(BinaryOp::Mul, left, right) = expr {
            // Determine priority of left and right
            let left_priority = function_priority(left);
            let right_priority = function_priority(right);

            // Higher priority should be u, lower should be dv
            let (u, dv) = if left_priority > right_priority {
                (left, right)
            } else {
                (right, left)
            };

            // Try integration by parts
            return Expr::integrate_by_parts(u, dv, var);
        }

        None
    }

    /// Determine priority for LIATE rule
    fn function_priority(expr: &Expr) -> u8 {
        match expr {
            Expr::Unary(UnaryOp::Log, _) => 5,  // Logarithmic
            Expr::Unary(UnaryOp::Arcsin, _) | Expr::Unary(UnaryOp::Arccos, _)
            | Expr::Unary(UnaryOp::Arctan, _) => 4,  // Inverse trig
            Expr::Symbol(_) | Expr::Binary(BinaryOp::Pow, _, _) => 3,  // Algebraic
            Expr::Unary(UnaryOp::Sin, _) | Expr::Unary(UnaryOp::Cos, _)
            | Expr::Unary(UnaryOp::Tan, _) => 2,  // Trigonometric
            Expr::Unary(UnaryOp::Exp, _) => 1,  // Exponential
            _ => 0,
        }
    }

    /// Recognize and integrate rational functions
    ///
    /// For f(x)/g(x) where f and g are polynomials
    pub fn integrate_rational(numerator: &Expr, denominator: &Expr, var: &Symbol) -> Option<Expr> {
        // Check if it's a simple case: constant/x = c*log(x)
        if numerator.is_constant() && matches!(denominator, Expr::Symbol(s) if s == var) {
            return Some(numerator.clone() * Expr::Symbol(var.clone()).log());
        }

        // Check for constant/(x² + a²) = (1/a)*arctan(x/a)
        if numerator.is_constant() {
            if let Expr::Binary(BinaryOp::Add, x_sq, a_sq) = denominator {
                if is_square_of_var(x_sq, var) && a_sq.is_constant() {
                    let a = (**a_sq).clone().sqrt();
                    let x = Expr::Symbol(var.clone());
                    return Some(numerator.clone() * (x / a.clone()).arctan() / a);
                }
            }
        }

        // Check for constant/(x² - a²) - use partial fractions
        if numerator.is_constant() {
            if let Expr::Binary(BinaryOp::Sub, x_sq, a_sq) = denominator {
                if is_square_of_var(x_sq, var) && a_sq.is_constant() {
                    // Use partial fractions: 1/(x² - a²) = 1/(2a) * (1/(x-a) - 1/(x+a))
                    let a = (**a_sq).clone().sqrt();
                    let x = Expr::Symbol(var.clone());
                    let two_a = Expr::from(2) * a.clone();

                    // Integrate: 1/(2a) * (log|x-a| - log|x+a|) = 1/(2a) * log|(x-a)/(x+a)|
                    let term1 = (x.clone() - a.clone()).log();
                    let term2 = (x.clone() + a.clone()).log();
                    return Some(numerator.clone() * (term1 - term2) / two_a);
                }
            }
        }

        // Check for x/(x² + a²) = (1/2)*log(x² + a²)
        if matches!(numerator, Expr::Symbol(s) if s == var) {
            if let Expr::Binary(BinaryOp::Add, x_sq, a_sq) = denominator {
                if is_square_of_var(x_sq, var) && a_sq.is_constant() {
                    return Some(denominator.clone().log() / Expr::from(2));
                }
            }
        }

        // Check for (Ax + B)/(x² + C) - split into two integrals
        let deg_num = polynomial_degree(numerator, var)?;
        let deg_den = polynomial_degree(denominator, var)?;

        if deg_num == 1 && deg_den == 2 {
            // Try to split numerator into A*x + B
            if let Expr::Binary(BinaryOp::Add, ax_term, b_term) = numerator {
                // Check patterns for ax + b
                let (a_coeff, b_coeff) = if matches!(**ax_term, Expr::Symbol(ref s) if s == var) && b_term.is_constant() {
                    (Expr::from(1), (**b_term).clone())
                } else if b_term.is_constant() {
                    if let Expr::Binary(BinaryOp::Mul, a, x) = &**ax_term {
                        if a.is_constant() && matches!(**x, Expr::Symbol(ref s) if s == var) {
                            ((**a).clone(), (**b_term).clone())
                        } else {
                            return integrate_rational_with_partial_fractions(numerator, denominator, var);
                        }
                    } else {
                        return integrate_rational_with_partial_fractions(numerator, denominator, var);
                    }
                } else {
                    return integrate_rational_with_partial_fractions(numerator, denominator, var);
                };

                // Integrate A*x/(x²+c) + B/(x²+c) separately
                let x = Expr::Symbol(var.clone());
                let int1 = integrate_rational(&(a_coeff * x.clone()), denominator, var)?;
                let int2 = integrate_rational(&b_coeff, denominator, var)?;
                return Some(int1 + int2);
            }
        }

        // Try partial fractions for more complex cases
        integrate_rational_with_partial_fractions(numerator, denominator, var)
    }

    /// Integrate expressions involving sqrt
    pub fn integrate_with_sqrt(expr: &Expr, var: &Symbol) -> Option<Expr> {
        // Common patterns:
        // ∫ 1/sqrt(1-x²) dx = arcsin(x)
        // ∫ 1/sqrt(1+x²) dx = arcsinh(x) = log(x + sqrt(1+x²))
        // ∫ 1/sqrt(x²-1) dx = arccosh(x) = log(x + sqrt(x²-1))

        if let Expr::Binary(BinaryOp::Div, num, den) = expr {
            if num.is_one() {
                if let Expr::Unary(UnaryOp::Sqrt, inner) = &**den {
                    // Check for 1/sqrt(1-x²)
                    if let Expr::Binary(BinaryOp::Sub, one, x_sq) = &**inner {
                        if one.is_one() && matches!(**x_sq, Expr::Binary(BinaryOp::Pow, ref x, ref two)
                            if matches!(**x, Expr::Symbol(ref s) if s == var) && matches!(**two, Expr::Integer(_))) {
                            return Some(Expr::Symbol(var.clone()).arcsin());
                        }
                    }
                }
            }
        }

        None
    }

    /// Table of common integrals
    pub struct IntegralTable;

    impl IntegralTable {
        /// Look up integral in table of known integrals
        pub fn lookup(expr: &Expr, var: &Symbol) -> Option<Expr> {
            // ∫ 1/(1+x²) dx = arctan(x)
            if let Expr::Binary(BinaryOp::Div, num, den) = expr {
                if num.is_one() {
                    // Check for 1/(1+x²)
                    if let Expr::Binary(BinaryOp::Add, one, x_sq) = &**den {
                        if one.is_one() {
                            if let Expr::Binary(BinaryOp::Pow, x, two) = &**x_sq {
                                if matches!(**x, Expr::Symbol(ref s) if s == var)
                                    && matches!(**two, Expr::Integer(ref i) if i.to_i64() == 2) {
                                    return Some(Expr::Symbol(var.clone()).arctan());
                                }
                            }
                        }
                    }
                }
            }

            // ∫ sec²(x) dx = tan(x)
            // ∫ csc²(x) dx = -cot(x)
            // etc.

            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_polynomials::UnivariatePolynomial;
    use rustmath_rationals::Rational;

    /// The differentiation gate for rational integrands: F' − f must
    /// normalize to exactly zero through the exact rational normalizer.
    /// Strictly stronger than asserting any particular literal form of F
    /// (two antiderivatives may legitimately differ in form or by a
    /// constant), and exact — no floating point, no simplify().
    fn assert_exact_antiderivative(input: &Expr, result: &Expr, var: &Symbol) {
        let diff = result.differentiate(var) - input.clone();
        let (n, d) = crate::risch::as_rational_function(&diff, var)
            .expect("gate: F' - f must normalize as a rational function");
        assert!(
            n.is_zero() && d == UnivariatePolynomial::<Rational>::one(),
            "gate failed: F' - f = {} / {}, expected 0",
            n,
            d
        );
    }

    #[test]
    fn test_integrate_constant() {
        let x = Symbol::new("x");
        let expr = Expr::from(5);
        let result = expr.integrate(&x).unwrap();
        // ∫ 5 dx = 5x
        assert_eq!(result, Expr::from(5) * Expr::Symbol(x));
    }

    #[test]
    fn test_integrate_variable() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone());
        let result = expr.integrate(&x).unwrap();
        // ∫ x dx = x²/2. The Risch tier emits (1/2)·x² rather than the old
        // table's x²/2 — the same antiderivative in a different literal
        // form, so assert through the differentiation gate instead.
        assert_exact_antiderivative(&expr, &result, &x);
    }

    #[test]
    fn test_integrate_power() {
        let x = Symbol::new("x");
        // ∫ x² dx = x³/3. Risch emits (1/3)·x³ (same antiderivative,
        // different literal form) — assert via the differentiation gate.
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = expr.integrate(&x).unwrap();
        assert_exact_antiderivative(&expr, &result, &x);
    }

    #[test]
    fn test_integrate_sum() {
        let x = Symbol::new("x");
        // ∫ (x + 1) dx = x²/2 + x. Risch emits x + (1/2)·x² (same
        // antiderivative, different term order and coefficient form) —
        // assert via the differentiation gate.
        let expr = Expr::Symbol(x.clone()) + Expr::from(1);
        let result = expr.integrate(&x).unwrap();
        assert_exact_antiderivative(&expr, &result, &x);
    }

    #[test]
    fn test_integrate_sin() {
        let x = Symbol::new("x");
        // ∫ sin(x) dx = -cos(x)
        let expr = Expr::Symbol(x.clone()).sin();
        let result = expr.integrate(&x).unwrap();
        let expected = -Expr::Symbol(x.clone()).cos();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_cos() {
        let x = Symbol::new("x");
        // ∫ cos(x) dx = sin(x)
        let expr = Expr::Symbol(x.clone()).cos();
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).sin();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_exp() {
        let x = Symbol::new("x");
        // ∫ exp(x) dx = exp(x)
        let expr = Expr::Symbol(x.clone()).exp();
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).exp();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_inverse() {
        let x = Symbol::new("x");
        // ∫ 1/x dx = log(x)
        let expr = Expr::from(1) / Expr::Symbol(x.clone());
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).log();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_constant_multiple() {
        let x = Symbol::new("x");
        // ∫ 3x dx = 3x²/2. Risch emits (3/2)·x² (same antiderivative,
        // different literal form) — assert via the differentiation gate.
        let expr = Expr::from(3) * Expr::Symbol(x.clone());
        let result = expr.integrate(&x).unwrap();
        assert_exact_antiderivative(&expr, &result, &x);
    }

    // ---- routing: the rational front door must be the decision procedure ----

    #[test]
    fn test_rational_front_door_is_risch() {
        let x = Symbol::new("x");
        let xe = Expr::Symbol(x.clone());
        // ∫ 1/(x²+1): decided exactly; gate the returned antiderivative.
        let expr = Expr::from(1) / (xe.clone().pow(Expr::from(2)) + Expr::from(1));
        let result = expr.integrate(&x).unwrap();
        assert_exact_antiderivative(&expr, &result, &x);
        // ∫ (x⁵+3x−2)/(x³−x): far beyond the legacy table, decided exactly.
        let num = xe.clone().pow(Expr::from(5)) + Expr::from(3) * xe.clone() - Expr::from(2);
        let den = xe.clone().pow(Expr::from(3)) - xe.clone();
        let expr = num / den;
        let result = expr.integrate(&x).unwrap();
        assert_exact_antiderivative(&expr, &result, &x);
    }

    #[test]
    fn test_rational_front_door_root_sum() {
        let x = Symbol::new("x");
        let xe = Expr::Symbol(x.clone());
        // ∫ 1/(x²−2): the residues ±√2/4 are irrational, so the exact
        // answer carries a root_sum object, certified by the Risch module's
        // internal trace gate (differentiate() knows the pass-through rule
        // d/dx root_sum(f, g) = root_sum(f, ∂g/∂x), but as_rational_function
        // cannot normalize a root_sum, so the rational gate does not apply).
        let expr = Expr::from(1) / (xe.clone().pow(Expr::from(2)) - Expr::from(2));
        let result = expr.integrate(&x).unwrap();
        fn contains_root_sum(e: &Expr) -> bool {
            match e {
                Expr::Function(name, args) => {
                    name == "root_sum" || args.iter().any(|a| contains_root_sum(a))
                }
                Expr::Unary(_, inner) => contains_root_sum(inner),
                Expr::Binary(_, l, r) => contains_root_sum(l) || contains_root_sum(r),
                _ => false,
            }
        }
        assert!(contains_root_sum(&result));
    }

    // ---- routing: a tier-1 budget trip falls through, never hangs ----

    // The four audited blocker inputs: pre-budget, Expr::integrate was
    // effectively non-terminating on each (measured kills at 150 s, 240 s /
    // 550 s, 63 s, 300 s in debug). The Risch tier now refuses them with a
    // labeled BudgetExceeded, the front door falls through to tiers 2–3
    // exactly as if tier 1 declined, and the honest outcome under the
    // pre-existing Option contract is None — reached in milliseconds
    // (time-asserted generously for slow CI).
    #[test]
    fn test_budget_trip_falls_through_to_honest_none() {
        let x = Symbol::new("x");
        let xe = Expr::Symbol(x.clone());
        let blockers: Vec<Expr> = vec![
            Expr::from(1) / (xe.clone().pow(Expr::from(15)) + xe.clone() + Expr::from(1)),
            Expr::from(1) / (xe.clone().pow(Expr::from(40)) + xe.clone() + Expr::from(1)),
            Expr::from(1)
                / ((xe.clone() - Expr::from(1)).pow(Expr::from(30))
                    * (xe.clone() + Expr::from(2))),
            (xe.clone().pow(Expr::from(3)) + Expr::from(1))
                / ((xe.clone().pow(Expr::from(2)) + Expr::from(1)).pow(Expr::from(10))
                    * (xe.clone() - Expr::from(2)).pow(Expr::from(20))),
        ];
        for e in blockers {
            let t0 = std::time::Instant::now();
            // Tier 1 refuses on budget (verified stage-by-stage in the risch
            // tests); tiers 2–3 have no pattern for these, so the honest
            // front-door answer is None — not a hang, not a wrong form.
            assert_eq!(e.integrate(&x), None, "expected honest None for {}", e);
            println!("blocker {} -> None in {:?}", e, t0.elapsed());
            assert!(
                t0.elapsed() < std::time::Duration::from_secs(30),
                "front door must return promptly for {}, took {:?}",
                e,
                t0.elapsed()
            );
        }
    }

    // ---- routing: the exact gated patterns run before the legacy table ----

    #[test]
    fn test_exact_patterns_reachable_through_integrate() {
        let x = Symbol::new("x");
        let xe = Expr::Symbol(x.clone());
        // ∫ x·sin(x): decided by the p(x)·sin(ax+b) recurrence (the old
        // table's by-parts heuristic could not complete it).
        let expr = xe.clone() * xe.clone().sin();
        assert!(expr.integrate(&x).is_some());
        // ∫ cos(x)/sin(x): the u'/u pattern.
        let expr = xe.clone().cos() / xe.clone().sin();
        assert_eq!(expr.integrate(&x).unwrap(), xe.clone().sin().log());
        // ∫ exp(2x): the old table only knew exp(x).
        let expr = (Expr::from(2) * xe.clone()).exp();
        assert!(expr.integrate(&x).is_some());
    }

    // ---- the honest frontier: None is correct behavior ----

    #[test]
    fn test_refuses_nonelementary_integrands() {
        let x = Symbol::new("x");
        let xe = Expr::Symbol(x.clone());
        let refusals: Vec<Expr> = vec![
            xe.clone().exp() / xe.clone(),        // Ei(x)
            Expr::from(1) / xe.clone().log(),     // li(x)
            xe.clone().pow(Expr::from(2)).exp(),  // erfi
            xe.clone().sin() / xe.clone(),        // Si(x)
        ];
        for e in refusals {
            assert_eq!(e.integrate(&x), None, "must refuse {}", e);
        }
    }

    #[test]
    fn test_definite_integral() {
        let x = Symbol::new("x");
        // ∫[0,1] x dx = 1/2
        let expr = Expr::Symbol(x.clone());
        let result = expr.integrate_definite(&x, &Expr::from(0), &Expr::from(1)).unwrap();
        // Should be (1²/2 - 0²/2) = 1/2
        // The result will be a symbolic expression, we can simplify it
        let simplified = result.simplify();
        // We expect 1/2
        assert!(matches!(simplified, Expr::Rational(_)));
    }

    // ========================================================================
    // Tests for Advanced Integration Techniques
    // ========================================================================

    #[test]
    fn test_integration_by_parts_x_sin() {
        use super::advanced;

        let x = Symbol::new("x");
        // ∫ x*sin(x) dx = -x*cos(x) + sin(x)
        let expr = Expr::Symbol(x.clone()) * Expr::Symbol(x.clone()).sin();

        // Integration by parts should be able to handle this, but may not always succeed
        // due to the recursive nature of the integral ∫ v du
        let result = advanced::integrate_by_parts_auto(&expr, &x);

        // For now, just test that the function doesn't panic
        // In future, we could add more sophisticated integration by parts that handles recursion
        let _ = result;
    }

    #[test]
    fn test_integration_by_parts_x_exp() {
        use super::advanced;

        let x = Symbol::new("x");
        // ∫ x*exp(x) dx = (x-1)*exp(x)
        let expr = Expr::Symbol(x.clone()) * Expr::Symbol(x.clone()).exp();

        // Integration by parts should be able to handle this, but may not always succeed
        let result = advanced::integrate_by_parts_auto(&expr, &x);

        // For now, just test that the function doesn't panic
        let _ = result;
    }

    #[test]
    fn test_integration_by_parts_log() {
        let x = Symbol::new("x");
        // ∫ log(x) dx = x*log(x) - x (already implemented in basic integration)
        let expr = Expr::Symbol(x.clone()).log();
        let result = expr.integrate(&x);
        assert!(result.is_some());
    }

    #[test]
    fn test_trig_substitution_arcsin() {
        let x = Symbol::new("x");
        // ∫ 1/sqrt(1 - x²) dx = arcsin(x)
        let one = Expr::from(1);
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let sqrt_term = (one.clone() - x_sq).sqrt();
        let expr = one / sqrt_term;
        let result = expr.integrate(&x);
        assert!(result.is_some());
    }

    #[test]
    fn test_trig_substitution_arcsinh() {
        let x = Symbol::new("x");
        // ∫ 1/sqrt(1 + x²) dx = arcsinh(x)
        let one = Expr::from(1);
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let sqrt_term = (one.clone() + x_sq).sqrt();
        let expr = one / sqrt_term;
        let result = expr.integrate(&x);
        assert!(result.is_some());
    }

    #[test]
    fn test_trig_substitution_arccosh() {
        let x = Symbol::new("x");
        // ∫ 1/sqrt(x² - 1) dx = arccosh(x)
        let one = Expr::from(1);
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let sqrt_term = (x_sq - one.clone()).sqrt();
        let expr = one / sqrt_term;
        let result = expr.integrate(&x);
        assert!(result.is_some());
    }

    #[test]
    fn test_trig_substitution_sqrt_integral() {
        use super::advanced;

        let x = Symbol::new("x");
        // ∫ sqrt(1 - x²) dx = (x/2)sqrt(1-x²) + (1/2)arcsin(x)
        let one = Expr::from(1);
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let expr = (one - x_sq).sqrt();
        let result = advanced::try_trig_substitution(&expr, &x);
        assert!(result.is_some());
    }

    #[test]
    fn test_partial_fractions_difference_of_squares() {
        let x = Symbol::new("x");
        // ∫ 1/(x² - 1) dx = (1/2)log|(x-1)/(x+1)|
        let one = Expr::from(1);
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let expr = one.clone() / (x_sq - one);
        let result = expr.integrate(&x);
        assert!(result.is_some());
    }

    #[test]
    fn test_partial_fractions_sum_of_squares() {
        let x = Symbol::new("x");
        // ∫ 1/(x² + 1) dx = arctan(x)
        let one = Expr::from(1);
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let expr = one.clone() / (x_sq + one);
        let result = expr.integrate(&x);
        assert!(result.is_some());

        // The result should be arctan(x)
        if let Some(integral) = result {
            // Check if result contains arctan
            let result_str = format!("{:?}", integral);
            assert!(result_str.contains("Arctan") || result_str.contains("arctan"));
        }
    }

    #[test]
    fn test_rational_x_over_x_squared_plus_a() {
        let x = Symbol::new("x");
        // ∫ x/(x² + 4) dx = (1/2)log(x² + 4)
        let four = Expr::from(4);
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let expr = Expr::Symbol(x.clone()) / (x_sq + four);
        let result = expr.integrate(&x);
        assert!(result.is_some());
    }

    #[test]
    fn test_rational_linear_over_quadratic() {
        use super::advanced;

        let x = Symbol::new("x");
        // ∫ (2x + 3)/(x² + 1) dx
        // This should split into: ∫ 2x/(x²+1) dx + ∫ 3/(x²+1) dx
        // = log(x²+1) + 3*arctan(x)
        let two = Expr::from(2);
        let three = Expr::from(3);
        let one = Expr::from(1);
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let numerator = two * Expr::Symbol(x.clone()) + three;
        let denominator = x_sq + one;

        let result = advanced::integrate_rational(&numerator, &denominator, &x);

        // This is a complex case that may not always succeed with our current implementation
        // Just test that it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_polynomial_degree() {
        use super::advanced::polynomial_degree;

        let x = Symbol::new("x");

        // Constant has degree 0
        assert_eq!(polynomial_degree(&Expr::from(5), &x), Some(0));

        // x has degree 1
        assert_eq!(polynomial_degree(&Expr::Symbol(x.clone()), &x), Some(1));

        // x² has degree 2
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));
        assert_eq!(polynomial_degree(&x_sq, &x), Some(2));

        // x² + x has degree 2
        let poly = x_sq.clone() + Expr::Symbol(x.clone());
        assert_eq!(polynomial_degree(&poly, &x), Some(2));

        // 3x³ has degree 3
        let x_cubed = Expr::Symbol(x.clone()).pow(Expr::from(3));
        let poly = Expr::from(3) * x_cubed;
        assert_eq!(polynomial_degree(&poly, &x), Some(3));
    }

    #[test]
    fn test_trig_substitution_detection() {
        use super::advanced::{detect_trig_substitution, TrigSubstitution};

        let x = Symbol::new("x");

        // sqrt(4 - x²) should detect Sine substitution
        let four = Expr::from(4);
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let expr = (four.clone() - x_sq.clone()).sqrt();
        let result = detect_trig_substitution(&expr, &x);
        assert!(result.is_some());
        if let Some((sub_type, _)) = result {
            assert_eq!(sub_type, TrigSubstitution::Sine);
        }

        // sqrt(4 + x²) should detect Tangent substitution
        let expr = (four.clone() + x_sq.clone()).sqrt();
        let result = detect_trig_substitution(&expr, &x);
        assert!(result.is_some());
        if let Some((sub_type, _)) = result {
            assert_eq!(sub_type, TrigSubstitution::Tangent);
        }

        // sqrt(x² - 4) should detect Secant substitution
        let expr = (x_sq.clone() - four).sqrt();
        let result = detect_trig_substitution(&expr, &x);
        assert!(result.is_some());
        if let Some((sub_type, _)) = result {
            assert_eq!(sub_type, TrigSubstitution::Secant);
        }
    }

    #[test]
    fn test_integration_completeness() {
        let x = Symbol::new("x");

        // Test various expressions to ensure they either integrate successfully
        // or return None gracefully (no panics)

        // Basic polynomials
        let _ = Expr::from(1).integrate(&x);
        let _ = Expr::Symbol(x.clone()).integrate(&x);
        let _ = (Expr::Symbol(x.clone()).pow(Expr::from(2))).integrate(&x);

        // Trigonometric
        let _ = Expr::Symbol(x.clone()).sin().integrate(&x);
        let _ = Expr::Symbol(x.clone()).cos().integrate(&x);

        // Exponential and logarithmic
        let _ = Expr::Symbol(x.clone()).exp().integrate(&x);
        let _ = Expr::Symbol(x.clone()).log().integrate(&x);

        // Rational functions
        let one = Expr::from(1);
        let x_sq = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let _ = (one.clone() / (x_sq.clone() + one.clone())).integrate(&x);
        let _ = (one.clone() / (x_sq.clone() - one.clone())).integrate(&x);

        // Square roots
        let _ = (one.clone() - x_sq.clone()).sqrt().integrate(&x);
        let _ = (one + x_sq).sqrt().integrate(&x);
    }
}
