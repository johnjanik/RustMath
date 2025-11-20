//! LaTeX mathematical typesetting

use crate::{BracketStyle, DisplayMode, FormatOptions, OutputFormat};
use crate::utils;

/// Format a fraction in LaTeX
pub fn fraction(numerator: &str, denominator: &str, options: &FormatOptions) -> String {
    match options.mode {
        DisplayMode::Display => format!(r"\frac{{{}}}{{{}}}", numerator, denominator),
        DisplayMode::Inline => format!(r"{}/{}", numerator, denominator),
    }
}

/// Format a power/exponent in LaTeX
pub fn power(base: &str, exponent: &str) -> String {
    format!("{}^{{{}}}", base, exponent)
}

/// Format a subscript in LaTeX
pub fn subscript(base: &str, sub: &str) -> String {
    format!("{}_{{{}}}", base, sub)
}

/// Format a square root in LaTeX
pub fn sqrt(content: &str) -> String {
    format!(r"\sqrt{{{}}}", content)
}

/// Format an nth root in LaTeX
pub fn root(content: &str, n: &str) -> String {
    format!(r"\sqrt[{}]{{{}}}", n, content)
}

/// Format absolute value in LaTeX
pub fn abs(content: &str) -> String {
    format!(r"\left|{}\right|", content)
}

/// Format a sum with limits in LaTeX
pub fn sum(lower: &str, upper: &str, body: &str) -> String {
    format!(r"\sum_{{{}}}^{{{}}} {}", lower, upper, body)
}

/// Format a product with limits in LaTeX
pub fn product(lower: &str, upper: &str, body: &str) -> String {
    format!(r"\prod_{{{}}}^{{{}}} {}", lower, upper, body)
}

/// Format an integral in LaTeX
pub fn integral(lower: Option<&str>, upper: Option<&str>, integrand: &str, var: &str) -> String {
    match (lower, upper) {
        (Some(l), Some(u)) => format!(r"\int_{{{}}}^{{{}}} {} \, d{}", l, u, integrand, var),
        _ => format!(r"\int {} \, d{}", integrand, var),
    }
}

/// Format a limit in LaTeX
pub fn limit(var: &str, value: &str, expr: &str) -> String {
    format!(r"\lim_{{{} \to {}}} {}", var, value, expr)
}

/// Format a function call in LaTeX
pub fn function(name: &str, arg: &str) -> String {
    // Special functions get their own LaTeX commands
    match name {
        "sin" | "cos" | "tan" | "cot" | "sec" | "csc" |
        "sinh" | "cosh" | "tanh" | "coth" |
        "arcsin" | "arccos" | "arctan" |
        "log" | "ln" | "exp" |
        "det" | "tr" | "dim" | "ker" => {
            format!(r"\{}\left({}\right)", name, arg)
        }
        _ => format!(r"{}\left({}\right)", name, arg),
    }
}

/// Format a matrix in LaTeX
pub fn matrix(
    rows: &[Vec<String>],
    bracket_style: BracketStyle,
    mode: DisplayMode,
) -> String {
    if rows.is_empty() {
        return match bracket_style {
            BracketStyle::Square => r"\left[\right]".to_string(),
            BracketStyle::Round => r"\left(\right)".to_string(),
            _ => String::new(),
        };
    }

    let env = match (bracket_style, mode) {
        (BracketStyle::Square, DisplayMode::Display) => "bmatrix",
        (BracketStyle::Round, DisplayMode::Display) => "pmatrix",
        (BracketStyle::Vertical, DisplayMode::Display) => "vmatrix",
        (BracketStyle::DoubleVertical, DisplayMode::Display) => "Vmatrix",
        _ => "matrix",
    };

    let mut result = format!(r"\begin{{{}}}", env);
    result.push('\n');

    for (i, row) in rows.iter().enumerate() {
        result.push_str(&row.join(" & "));
        if i < rows.len() - 1 {
            result.push_str(r" \\");
            result.push('\n');
        }
    }

    result.push('\n');
    result.push_str(&format!(r"\end{{{}}}", env));

    // For inline mode, wrap in delimiters
    if mode == DisplayMode::Inline {
        let open = utils::opening_bracket(bracket_style, OutputFormat::LaTeX);
        let close = utils::closing_bracket(bracket_style, OutputFormat::LaTeX);
        format!(
            r"{}\begin{{matrix}}{}\end{{matrix}}{}",
            open,
            rows.iter()
                .map(|row| row.join(" & "))
                .collect::<Vec<_>>()
                .join(r" \\ "),
            close
        )
    } else {
        result
    }
}

/// Format a vector in LaTeX
pub fn vector(elements: &[String], bracket_style: BracketStyle) -> String {
    let rows: Vec<Vec<String>> = elements.iter().map(|e| vec![e.clone()]).collect();
    matrix(&rows, bracket_style, DisplayMode::Inline)
}

/// Format a piecewise function in LaTeX
pub fn piecewise(cases: &[(String, String)]) -> String {
    let mut result = r"\begin{cases}".to_string();
    result.push('\n');

    for (i, (expr, cond)) in cases.iter().enumerate() {
        result.push_str(&format!("{} & \\text{{if }} {}", expr, cond));
        if i < cases.len() - 1 {
            result.push_str(r" \\");
            result.push('\n');
        }
    }

    result.push('\n');
    result.push_str(r"\end{cases}");
    result
}

/// Escape special LaTeX characters
pub fn escape(s: &str) -> String {
    s.replace('\\', r"\textbackslash ")
        .replace('{', r"\{")
        .replace('}', r"\}")
        .replace('$', r"\$")
        .replace('%', r"\%")
        .replace('&', r"\&")
        .replace('#', r"\#")
        .replace('_', r"\_")
        .replace('^', r"\^{}")
        .replace('~', r"\~{}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fraction() {
        let opts = FormatOptions::latex().with_mode(DisplayMode::Display);
        assert_eq!(fraction("1", "2", &opts), r"\frac{1}{2}");

        let opts_inline = FormatOptions::latex().with_mode(DisplayMode::Inline);
        assert_eq!(fraction("1", "2", &opts_inline), "1/2");
    }

    #[test]
    fn test_power() {
        assert_eq!(power("x", "2"), "x^{2}");
        assert_eq!(power("10", "n"), "10^{n}");
    }

    #[test]
    fn test_subscript() {
        assert_eq!(subscript("x", "0"), "x_{0}");
        assert_eq!(subscript("a", "i"), "a_{i}");
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(sqrt("2"), r"\sqrt{2}");
        assert_eq!(sqrt("x^2 + 1"), r"\sqrt{x^2 + 1}");
    }

    #[test]
    fn test_root() {
        assert_eq!(root("8", "3"), r"\sqrt[3]{8}");
    }

    #[test]
    fn test_abs() {
        assert_eq!(abs("x"), r"\left|x\right|");
    }

    #[test]
    fn test_sum() {
        assert_eq!(sum("i=1", "n", "i^2"), r"\sum_{i=1}^{n} i^2");
    }

    #[test]
    fn test_product() {
        assert_eq!(product("i=1", "n", "i"), r"\prod_{i=1}^{n} i");
    }

    #[test]
    fn test_integral() {
        assert_eq!(
            integral(Some("0"), Some("1"), "x^2", "x"),
            r"\int_{0}^{1} x^2 \, dx"
        );
        assert_eq!(integral(None, None, "f(x)", "x"), r"\int f(x) \, dx");
    }

    #[test]
    fn test_limit() {
        assert_eq!(limit("x", "0", "\\frac{\\sin x}{x}"), r"\lim_{x \to 0} \frac{\sin x}{x}");
    }

    #[test]
    fn test_function() {
        assert_eq!(function("sin", "x"), r"\sin\left(x\right)");
        assert_eq!(function("f", "x"), r"f\left(x\right)");
    }

    #[test]
    fn test_vector() {
        let elements = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let result = vector(&elements, BracketStyle::Square);
        assert!(result.contains("1") && result.contains("2") && result.contains("3"));
    }

    #[test]
    fn test_escape() {
        assert_eq!(escape("a_b"), r"a\_b");
        assert_eq!(escape("x^2"), r"x\^{}2");
        assert_eq!(escape("50%"), r"50\%");
    }

    #[test]
    fn test_matrix_empty() {
        let result = matrix(&[], BracketStyle::Square, DisplayMode::Display);
        assert_eq!(result, r"\left[\right]");
    }

    #[test]
    fn test_matrix_simple() {
        let rows = vec![
            vec!["1".to_string(), "2".to_string()],
            vec!["3".to_string(), "4".to_string()],
        ];
        let result = matrix(&rows, BracketStyle::Square, DisplayMode::Display);
        assert!(result.contains(r"\begin{bmatrix}"));
        assert!(result.contains("1 & 2"));
        assert!(result.contains(r"\\"));
        assert!(result.contains("3 & 4"));
    }
}
