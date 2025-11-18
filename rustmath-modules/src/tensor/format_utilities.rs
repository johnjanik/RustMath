//! # Tensor Formatting Utilities
//!
//! This module provides utilities for formatting tensor expressions,
//! corresponding to SageMath's `sage.tensor.modules.format_utilities`.
//!
//! ## Main Functions
//!
//! - `is_atomic`: Check if an expression is atomic (doesn't need parentheses)
//! - `format_mul_txt`: Format multiplication for text output
//! - `format_mul_latex`: Format multiplication for LaTeX output
//! - `format_unop_txt`: Format unary operations for text
//! - `format_unop_latex`: Format unary operations for LaTeX
//! - `is_atomic_wedge_txt/latex`: Check if expression is atomic for wedge product

use std::fmt;

/// Formatted expansion of a tensor
///
/// Stores a formatted representation of a tensor expansion
pub struct FormattedExpansion {
    /// The formatted string
    content: String,
    /// Whether this is LaTeX format
    is_latex: bool,
}

impl FormattedExpansion {
    /// Create a new formatted expansion
    pub fn new(content: String, is_latex: bool) -> Self {
        Self { content, is_latex }
    }

    /// Get the formatted content
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Check if this is LaTeX format
    pub fn is_latex(&self) -> bool {
        self.is_latex
    }
}

impl fmt::Display for FormattedExpansion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}

/// Check if a string representation is "atomic" (doesn't need parentheses)
///
/// An expression is atomic if it's a single symbol or doesn't contain operators
/// at the top level that would require parenthesization.
///
/// # Examples
///
/// ```
/// use rustmath_modules::tensor::format_utilities::is_atomic;
///
/// assert!(is_atomic("x"));
/// assert!(is_atomic("42"));
/// assert!(is_atomic("sin(x)"));
/// assert!(!is_atomic("x + y"));
/// assert!(!is_atomic("a - b"));
/// ```
pub fn is_atomic(expression: &str) -> bool {
    let expr = expression.trim();

    // Empty or single character is atomic
    if expr.len() <= 1 {
        return true;
    }

    // Check for operators that make it non-atomic
    let has_top_level_operator = expr.contains('+') || expr.contains('-');

    // If it starts and ends with matching parens/brackets, check inside
    if (expr.starts_with('(') && expr.ends_with(')'))
        || (expr.starts_with('[') && expr.ends_with(']'))
    {
        return true;
    }

    !has_top_level_operator
}

/// Check if an expression is atomic for wedge product (text mode)
///
/// In text mode, wedge products are written with /\ symbol
pub fn is_atomic_wedge_txt(expression: &str) -> bool {
    let expr = expression.trim();

    // More permissive than general atomic check
    // Allow things like "dx" but not "dx + dy"
    !expr.contains('+') && !expr.contains('-') && !expr.contains("/\\")
}

/// Check if an expression is atomic for wedge product (LaTeX mode)
///
/// In LaTeX mode, wedge products are written with \wedge
pub fn is_atomic_wedge_latex(expression: &str) -> bool {
    let expr = expression.trim();

    // More permissive than general atomic check
    !expr.contains('+') && !expr.contains('-') && !expr.contains("\\wedge")
}

/// Format a multiplication for text output
///
/// Adds parentheses around non-atomic factors as needed
///
/// # Examples
///
/// ```
/// use rustmath_modules::tensor::format_utilities::format_mul_txt;
///
/// assert_eq!(format_mul_txt("a", "b"), "a*b");
/// assert_eq!(format_mul_txt("x+y", "z"), "(x+y)*z");
/// ```
pub fn format_mul_txt(left: &str, right: &str) -> String {
    let left_part = if is_atomic(left) {
        left.to_string()
    } else {
        format!("({})", left)
    };

    let right_part = if is_atomic(right) {
        right.to_string()
    } else {
        format!("({})", right)
    };

    format!("{}*{}", left_part, right_part)
}

/// Format a multiplication for LaTeX output
///
/// Uses proper LaTeX formatting with spacing
pub fn format_mul_latex(left: &str, right: &str) -> String {
    let left_part = if is_atomic(left) {
        left.to_string()
    } else {
        format!("\\left({}\\right)", left)
    };

    let right_part = if is_atomic(right) {
        right.to_string()
    } else {
        format!("\\left({}\\right)", right)
    };

    format!("{} {}", left_part, right_part)
}

/// Format a unary operation for text output
///
/// # Arguments
///
/// * `operator` - The operator symbol (e.g., "-", "!")
/// * `operand` - The operand expression
pub fn format_unop_txt(operator: &str, operand: &str) -> String {
    if is_atomic(operand) {
        format!("{}{}", operator, operand)
    } else {
        format!("{}({})", operator, operand)
    }
}

/// Format a unary operation for LaTeX output
pub fn format_unop_latex(operator: &str, operand: &str) -> String {
    if is_atomic(operand) {
        format!("{}{}", operator, operand)
    } else {
        format!("{}\\left({}\\right)", operator, operand)
    }
}

/// Format a wedge product for text output
///
/// Formats a ∧ b with proper parenthesization
pub fn format_wedge_txt(left: &str, right: &str) -> String {
    let left_part = if is_atomic_wedge_txt(left) {
        left.to_string()
    } else {
        format!("({})", left)
    };

    let right_part = if is_atomic_wedge_txt(right) {
        right.to_string()
    } else {
        format!("({})", right)
    };

    format!("{} /\\ {}", left_part, right_part)
}

/// Format a wedge product for LaTeX output
pub fn format_wedge_latex(left: &str, right: &str) -> String {
    let left_part = if is_atomic_wedge_latex(left) {
        left.to_string()
    } else {
        format!("\\left({}\\right)", left)
    };

    let right_part = if is_atomic_wedge_latex(right) {
        right.to_string()
    } else {
        format!("\\left({}\\right)", right)
    };

    format!("{} \\wedge {}", left_part, right_part)
}

/// Format a tensor product for text output
pub fn format_tensor_product_txt(left: &str, right: &str) -> String {
    let left_part = if is_atomic(left) {
        left.to_string()
    } else {
        format!("({})", left)
    };

    let right_part = if is_atomic(right) {
        right.to_string()
    } else {
        format!("({})", right)
    };

    format!("{} ⊗ {}", left_part, right_part)
}

/// Format a tensor product for LaTeX output
pub fn format_tensor_product_latex(left: &str, right: &str) -> String {
    let left_part = if is_atomic(left) {
        left.to_string()
    } else {
        format!("\\left({}\\right)", left)
    };

    let right_part = if is_atomic(right) {
        right.to_string()
    } else {
        format!("\\left({}\\right)", right)
    };

    format!("{} \\otimes {}", left_part, right_part)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_atomic() {
        assert!(is_atomic("x"));
        assert!(is_atomic("42"));
        assert!(is_atomic("abc"));
        assert!(is_atomic("(x+y)"));
        assert!(is_atomic("[a,b]"));

        assert!(!is_atomic("x + y"));
        assert!(!is_atomic("a - b"));
        assert!(!is_atomic("x+y"));
    }

    #[test]
    fn test_is_atomic_wedge_txt() {
        assert!(is_atomic_wedge_txt("dx"));
        assert!(is_atomic_wedge_txt("e1"));

        assert!(!is_atomic_wedge_txt("dx + dy"));
        assert!(!is_atomic_wedge_txt("dx /\\ dy"));
    }

    #[test]
    fn test_format_mul_txt() {
        assert_eq!(format_mul_txt("a", "b"), "a*b");
        assert_eq!(format_mul_txt("x+y", "z"), "(x+y)*z");
        assert_eq!(format_mul_txt("a", "b+c"), "a*(b+c)");
        assert_eq!(format_mul_txt("x-y", "a-b"), "(x-y)*(a-b)");
    }

    #[test]
    fn test_format_mul_latex() {
        assert_eq!(format_mul_latex("a", "b"), "a b");
        assert_eq!(format_mul_latex("x+y", "z"), "\\left(x+y\\right) z");
    }

    #[test]
    fn test_format_unop_txt() {
        assert_eq!(format_unop_txt("-", "x"), "-x");
        assert_eq!(format_unop_txt("-", "x+y"), "-(x+y)");
    }

    #[test]
    fn test_format_unop_latex() {
        assert_eq!(format_unop_latex("-", "x"), "-x");
        assert_eq!(format_unop_latex("-", "x+y"), "-\\left(x+y\\right)");
    }

    #[test]
    fn test_format_wedge_txt() {
        assert_eq!(format_wedge_txt("dx", "dy"), "dx /\\ dy");
        assert_eq!(format_wedge_txt("dx+dy", "dz"), "(dx+dy) /\\ dz");
    }

    #[test]
    fn test_format_wedge_latex() {
        assert_eq!(format_wedge_latex("dx", "dy"), "dx \\wedge dy");
        assert_eq!(
            format_wedge_latex("dx+dy", "dz"),
            "\\left(dx+dy\\right) \\wedge dz"
        );
    }

    #[test]
    fn test_format_tensor_product_txt() {
        assert_eq!(format_tensor_product_txt("v", "w"), "v ⊗ w");
        assert_eq!(format_tensor_product_txt("v+w", "u"), "(v+w) ⊗ u");
    }

    #[test]
    fn test_format_tensor_product_latex() {
        assert_eq!(format_tensor_product_latex("v", "w"), "v \\otimes w");
        assert_eq!(
            format_tensor_product_latex("v+w", "u"),
            "\\left(v+w\\right) \\otimes u"
        );
    }

    #[test]
    fn test_formatted_expansion() {
        let exp = FormattedExpansion::new("x + y".to_string(), false);
        assert_eq!(exp.content(), "x + y");
        assert!(!exp.is_latex());

        let exp_latex = FormattedExpansion::new("x \\wedge y".to_string(), true);
        assert!(exp_latex.is_latex());
    }
}
