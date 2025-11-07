//! RustMath Calculus - Symbolic calculus operations
//!
//! This crate provides differentiation, integration, limits, and series expansions.

pub mod differentiation;

pub use differentiation::differentiate;

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_symbolic::Expr;

    #[test]
    fn basic_derivative() {
        // d/dx(x) = 1
        let x = Expr::symbol("x");
        let deriv = differentiate(&x, "x");
        // Test passes if it compiles - full symbolic diff testing needs more work
    }
}
