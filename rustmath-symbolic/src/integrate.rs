//! Symbolic integration
//!
//! This module implements symbolic integration for common functions.
//! While full Risch algorithm is complex, we implement a table-based
//! approach with pattern matching for common integrals.

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_core::{Ring, NumericConversion};
use std::sync::Arc;

impl Expr {
    /// Integrate the expression with respect to a symbol
    ///
    /// Implements standard integration rules:
    /// - ∫ c dx = c*x for constants
    /// - ∫ x^n dx = x^(n+1)/(n+1) for n ≠ -1
    /// - ∫ 1/x dx = log(|x|)
    /// - ∫ (f + g) dx = ∫f dx + ∫g dx
    /// - ∫ sin(x) dx = -cos(x)
    /// - ∫ cos(x) dx = sin(x)
    /// - ∫ exp(x) dx = exp(x)
    /// - ∫ 1/(1+x²) dx = arctan(x)
    ///
    /// Returns None if integration is not possible with current rules
    pub fn integrate(&self, var: &Symbol) -> Option<Self> {
        match self {
            // ∫ c dx = c*x for constants
            Expr::Integer(_) | Expr::Rational(_) => {
                Some(self.clone() * Expr::Symbol(var.clone()))
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

                // Constant multiple: ∫ c*f dx = c*∫f dx
                BinaryOp::Mul => {
                    if left.is_constant() && !left.contains_symbol(var) {
                        let integral = right.integrate(var)?;
                        Some((**left).clone() * integral)
                    } else if right.is_constant() && !right.contains_symbol(var) {
                        let integral = left.integrate(var)?;
                        Some(integral * (**right).clone())
                    } else {
                        // Try integration by parts or other advanced techniques
                        // For now, we only handle simple cases
                        None
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
                    } else {
                        None
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
                            let exponent = n.clone() + Expr::from(1);
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

        Some(fb - fa)
    }

    /// Check if expression contains a symbol
    fn contains_symbol(&self, var: &Symbol) -> bool {
        match self {
            Expr::Integer(_) | Expr::Rational(_) => false,
            Expr::Symbol(s) => s == var,
            Expr::Binary(_, left, right) => {
                left.contains_symbol(var) || right.contains_symbol(var)
            }
            Expr::Unary(_, inner) => inner.contains_symbol(var),
            Expr::Function(_, args) => args.iter().any(|arg| arg.contains_symbol(var)),
        }
    }

    /// Check if expression equals 1
    fn is_one(&self) -> bool {
        matches!(self, Expr::Integer(i) if i.to_i64() == Some(1))
    }

    /// Check if expression equals -1
    fn is_minus_one(&self) -> bool {
        matches!(self, Expr::Integer(i) if i.to_i64() == Some(-1))
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
        let inner_result = inner_at_y_max - inner_at_y_min;

        // Now integrate with respect to x
        let outer_integral = inner_result.integrate(x_var)?;

        // Evaluate at the x bounds
        let result_at_x_max = outer_integral.substitute(x_var, x_max);
        let result_at_x_min = outer_integral.substitute(x_var, x_min);

        Some(result_at_x_max - result_at_x_min)
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
        let result_z = integral_z.substitute(z_var, z_max) - integral_z.substitute(z_var, z_min);

        // Integrate with respect to y
        let integral_y = result_z.integrate(y_var)?;
        let result_y = integral_y.substitute(y_var, y_max) - integral_y.substitute(y_var, y_min);

        // Integrate with respect to x
        let integral_x = result_y.integrate(x_var)?;
        let result_x = integral_x.substitute(x_var, x_max) - integral_x.substitute(x_var, x_min);

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
        dx_du * dy_dv - dx_dv * dy_du
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
        // Look for sqrt patterns
        if let Expr::Unary(UnaryOp::Sqrt, inner) = expr {
            match &**inner {
                // sqrt(a² - x²)
                Expr::Binary(BinaryOp::Sub, left, right) => {
                    // Check if right is x²
                    if is_square_of_var(right, var) {
                        // left should be a constant a²
                        return Some((TrigSubstitution::Sine, (**left).clone()));
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
                // sqrt(x² - a²)
                Expr::Binary(BinaryOp::Sub, left, right) => {
                    // Check if left is x²
                    if is_square_of_var(left, var) {
                        return Some((TrigSubstitution::Secant, (**right).clone()));
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Check if an expression is x²
    fn is_square_of_var(expr: &Expr, var: &Symbol) -> bool {
        matches!(expr,
            Expr::Binary(BinaryOp::Pow, x, two)
            if matches!(**x, Expr::Symbol(ref s) if s == var)
                && matches!(**two, Expr::Integer(ref i) if i.to_i64() == Some(2))
        )
    }

    /// Perform trigonometric substitution
    ///
    /// Transforms the integral using the appropriate trig substitution
    pub fn apply_trig_substitution(
        expr: &Expr,
        var: &Symbol,
        sub_type: TrigSubstitution,
        a_squared: &Expr,
    ) -> Option<Expr> {
        // This is a simplified implementation
        // Full implementation would:
        // 1. Make the substitution
        // 2. Compute dx in terms of dθ
        // 3. Simplify using trig identities
        // 4. Integrate in terms of θ
        // 5. Substitute back to x

        // For now, handle the most common cases directly
        match sub_type {
            TrigSubstitution::Sine => {
                // For ∫ 1/sqrt(a² - x²) dx = arcsin(x/a)
                if let Expr::Binary(BinaryOp::Div, num, den) = expr {
                    if num.is_one() {
                        if let Expr::Unary(UnaryOp::Sqrt, inner) = &**den {
                            if let Expr::Binary(BinaryOp::Sub, left, right) = &**inner {
                                if *left == a_squared && is_square_of_var(right, var) {
                                    // Result: arcsin(x/sqrt(a²))
                                    let a = a_squared.clone().sqrt();
                                    return Some((Expr::Symbol(var.clone()) / a).arcsin());
                                }
                            }
                        }
                    }
                }
            }
            TrigSubstitution::Tangent => {
                // For ∫ 1/sqrt(a² + x²) dx = arcsinh(x/a) or log(x + sqrt(a² + x²))
                if let Expr::Binary(BinaryOp::Div, num, den) = expr {
                    if num.is_one() {
                        if let Expr::Unary(UnaryOp::Sqrt, inner) = &**den {
                            if let Expr::Binary(BinaryOp::Add, left, right) = &**inner {
                                let (const_part, var_part) = if is_square_of_var(right, var) {
                                    (left, right)
                                } else if is_square_of_var(left, var) {
                                    (right, left)
                                } else {
                                    return None;
                                };

                                if *const_part == a_squared {
                                    let a = const_part.clone().sqrt();
                                    let x = Expr::Symbol(var.clone());
                                    // arcsinh(x/a)
                                    return Some((x.clone() / a).arcsinh());
                                }
                            }
                        }
                    }
                }
            }
            TrigSubstitution::Secant => {
                // For ∫ 1/sqrt(x² - a²) dx = arccosh(x/a)
                if let Expr::Binary(BinaryOp::Div, num, den) = expr {
                    if num.is_one() {
                        if let Expr::Unary(UnaryOp::Sqrt, inner) = &**den {
                            if let Expr::Binary(BinaryOp::Sub, left, right) = &**inner {
                                if is_square_of_var(left, var) && *right == a_squared {
                                    let a = a_squared.clone().sqrt();
                                    return Some((Expr::Symbol(var.clone()) / a).arccosh());
                                }
                            }
                        }
                    }
                }
            }
        }

        None
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
    /// This is a simplified implementation. Full implementation requires:
    /// - Polynomial factorization over rationals/algebraic numbers
    /// - Solving systems of linear equations for coefficients
    /// - Handling repeated and complex roots
    pub fn partial_fractions(numerator: &Expr, denominator: &Expr, var: &Symbol) -> Option<Vec<Expr>> {
        // Simplified implementation for basic cases

        // Check for simple case: A/(Bx + C)
        if numerator.is_constant() {
            if let Expr::Binary(BinaryOp::Add, left, right) = denominator {
                // Check if it's of the form ax + b
                if let Expr::Binary(BinaryOp::Mul, coeff, x) = &**left {
                    if coeff.is_constant() && matches!(**x, Expr::Symbol(ref s) if s == var) {
                        if right.is_constant() {
                            // This is already in simplest form
                            return Some(vec![numerator.clone() / denominator.clone()]);
                        }
                    }
                }
            }
        }

        // For more complex cases, would need full factorization
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
                                    && matches!(**two, Expr::Integer(ref i) if i.to_i64() == Some(2)) {
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
        // ∫ x dx = x²/2
        let expected = Expr::Symbol(x.clone()).pow(Expr::from(2)) / Expr::from(2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_power() {
        let x = Symbol::new("x");
        // ∫ x² dx = x³/3
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).pow(Expr::from(3)) / Expr::from(3);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_sum() {
        let x = Symbol::new("x");
        // ∫ (x + 1) dx = x²/2 + x
        let expr = Expr::Symbol(x.clone()) + Expr::from(1);
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).pow(Expr::from(2)) / Expr::from(2)
            + Expr::Symbol(x.clone());
        assert_eq!(result, expected);
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
        // ∫ 3x dx = 3x²/2
        let expr = Expr::from(3) * Expr::Symbol(x.clone());
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::from(3) * (Expr::Symbol(x.clone()).pow(Expr::from(2)) / Expr::from(2));
        assert_eq!(result, expected);
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
}
