//! Unicode mathematical symbol rendering

use crate::{BracketStyle, FormatOptions};
use crate::utils::{to_subscript, to_superscript};

/// Format a fraction using Unicode
pub fn fraction(numerator: &str, denominator: &str) -> String {
    // For simple single-digit fractions, use Unicode fraction characters
    match (numerator, denominator) {
        ("1", "2") => "½".to_string(),
        ("1", "3") => "⅓".to_string(),
        ("2", "3") => "⅔".to_string(),
        ("1", "4") => "¼".to_string(),
        ("3", "4") => "¾".to_string(),
        ("1", "5") => "⅕".to_string(),
        ("2", "5") => "⅖".to_string(),
        ("3", "5") => "⅗".to_string(),
        ("4", "5") => "⅘".to_string(),
        ("1", "6") => "⅙".to_string(),
        ("5", "6") => "⅚".to_string(),
        ("1", "8") => "⅛".to_string(),
        ("3", "8") => "⅜".to_string(),
        ("5", "8") => "⅝".to_string(),
        ("7", "8") => "⅞".to_string(),
        _ => format!("{}/{}", numerator, denominator),
    }
}

/// Format a power/exponent using Unicode superscripts
pub fn power(base: &str, exponent: &str) -> String {
    // For simple numeric exponents, use superscripts
    if exponent.chars().all(|c| c.is_ascii_digit() || c == '-' || c == '+') {
        format!("{}{}", base, to_superscript(exponent))
    } else {
        format!("{}^({})", base, exponent)
    }
}

/// Format a subscript using Unicode subscripts
pub fn subscript(base: &str, sub: &str) -> String {
    // For simple numeric subscripts, use subscript characters
    if sub.chars().all(|c| c.is_ascii_digit() || c == '+' || c == '-') {
        format!("{}{}", base, to_subscript(sub))
    } else {
        format!("{}_({})", base, sub)
    }
}

/// Format a square root using Unicode
pub fn sqrt(content: &str) -> String {
    format!("√({})", content)
}

/// Format an nth root using Unicode
pub fn root(content: &str, n: &str) -> String {
    format!("{}√({})", to_superscript(n), content)
}

/// Format absolute value using Unicode
pub fn abs(content: &str) -> String {
    format!("|{}|", content)
}

/// Format a sum using Unicode
pub fn sum(lower: &str, upper: &str, body: &str) -> String {
    format!("∑({} to {}) {}", lower, upper, body)
}

/// Format a product using Unicode
pub fn product(lower: &str, upper: &str, body: &str) -> String {
    format!("∏({} to {}) {}", lower, upper, body)
}

/// Format an integral using Unicode
pub fn integral(lower: Option<&str>, upper: Option<&str>, integrand: &str, var: &str) -> String {
    match (lower, upper) {
        (Some(l), Some(u)) => format!("∫[{} to {}] {} d{}", l, u, integrand, var),
        _ => format!("∫ {} d{}", integrand, var),
    }
}

/// Format a limit using Unicode
pub fn limit(var: &str, value: &str, expr: &str) -> String {
    format!("lim({} → {}) {}", var, value, expr)
}

/// Format infinity symbol
pub fn infinity() -> &'static str {
    "∞"
}

/// Multiplication dot
pub fn multiply_dot() -> &'static str {
    "·"
}

/// Division symbol
pub fn divide() -> &'static str {
    "÷"
}

/// Plus-minus symbol
pub fn plus_minus() -> &'static str {
    "±"
}

/// Minus-plus symbol
pub fn minus_plus() -> &'static str {
    "∓"
}

/// Not equal symbol
pub fn not_equal() -> &'static str {
    "≠"
}

/// Less than or equal
pub fn less_equal() -> &'static str {
    "≤"
}

/// Greater than or equal
pub fn greater_equal() -> &'static str {
    "≥"
}

/// Much less than
pub fn much_less() -> &'static str {
    "≪"
}

/// Much greater than
pub fn much_greater() -> &'static str {
    "≫"
}

/// Approximately equal
pub fn approx() -> &'static str {
    "≈"
}

/// Identical/equivalent to
pub fn equiv() -> &'static str {
    "≡"
}

/// Element of
pub fn element_of() -> &'static str {
    "∈"
}

/// Not element of
pub fn not_element_of() -> &'static str {
    "∉"
}

/// Subset
pub fn subset() -> &'static str {
    "⊂"
}

/// Superset
pub fn superset() -> &'static str {
    "⊃"
}

/// Subset or equal
pub fn subset_eq() -> &'static str {
    "⊆"
}

/// Superset or equal
pub fn superset_eq() -> &'static str {
    "⊇"
}

/// Union
pub fn union() -> &'static str {
    "∪"
}

/// Intersection
pub fn intersection() -> &'static str {
    "∩"
}

/// Empty set
pub fn empty_set() -> &'static str {
    "∅"
}

/// For all (universal quantifier)
pub fn forall() -> &'static str {
    "∀"
}

/// Exists (existential quantifier)
pub fn exists() -> &'static str {
    "∃"
}

/// Logical and
pub fn logical_and() -> &'static str {
    "∧"
}

/// Logical or
pub fn logical_or() -> &'static str {
    "∨"
}

/// Logical not
pub fn logical_not() -> &'static str {
    "¬"
}

/// Implies
pub fn implies() -> &'static str {
    "⇒"
}

/// If and only if
pub fn iff() -> &'static str {
    "⇔"
}

/// Partial derivative
pub fn partial() -> &'static str {
    "∂"
}

/// Nabla (del operator)
pub fn nabla() -> &'static str {
    "∇"
}

/// Get opening bracket for matrix/vector
pub fn opening_bracket(style: BracketStyle) -> &'static str {
    match style {
        BracketStyle::Square => "[",
        BracketStyle::Round => "(",
        BracketStyle::Curly => "{",
        BracketStyle::Vertical => "|",
        BracketStyle::DoubleVertical => "‖",
    }
}

/// Get closing bracket for matrix/vector
pub fn closing_bracket(style: BracketStyle) -> &'static str {
    match style {
        BracketStyle::Square => "]",
        BracketStyle::Round => ")",
        BracketStyle::Curly => "}",
        BracketStyle::Vertical => "|",
        BracketStyle::DoubleVertical => "‖",
    }
}

/// Format a matrix using Unicode box-drawing characters
pub fn matrix(rows: &[Vec<String>], bracket_style: BracketStyle, _options: &FormatOptions) -> String {
    if rows.is_empty() {
        return format!(
            "{}{}",
            opening_bracket(bracket_style),
            closing_bracket(bracket_style)
        );
    }

    // Calculate column widths
    let num_cols = rows[0].len();
    let mut col_widths = vec![0; num_cols];

    for row in rows {
        for (j, elem) in row.iter().enumerate() {
            col_widths[j] = col_widths[j].max(elem.len());
        }
    }

    let open = opening_bracket(bracket_style);
    let close = closing_bracket(bracket_style);

    let mut result = String::new();

    for (i, row) in rows.iter().enumerate() {
        if i == 0 {
            result.push_str(open);
        } else {
            result.push(' ');
        }

        for (j, elem) in row.iter().enumerate() {
            let padding = col_widths[j] - elem.len();
            result.push_str(&" ".repeat(padding));
            result.push_str(elem);

            if j < row.len() - 1 {
                result.push_str("  ");
            }
        }

        if i == rows.len() - 1 {
            result.push_str(close);
        } else {
            result.push('\n');
        }
    }

    result
}

/// Format a vector using Unicode
pub fn vector(elements: &[String], bracket_style: BracketStyle, options: &FormatOptions) -> String {
    let rows: Vec<Vec<String>> = elements.iter().map(|e| vec![e.clone()]).collect();
    matrix(&rows, bracket_style, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fraction() {
        assert_eq!(fraction("1", "2"), "½");
        assert_eq!(fraction("1", "4"), "¼");
        assert_eq!(fraction("3", "4"), "¾");
        assert_eq!(fraction("2", "7"), "2/7");
    }

    #[test]
    fn test_power() {
        assert_eq!(power("x", "2"), "x²");
        assert_eq!(power("10", "3"), "10³");
        assert_eq!(power("x", "n"), "x^(n)");
    }

    #[test]
    fn test_subscript() {
        assert_eq!(subscript("x", "0"), "x₀");
        assert_eq!(subscript("a", "123"), "a₁₂₃");
        assert_eq!(subscript("x", "i"), "x_(i)");
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(sqrt("2"), "√(2)");
    }

    #[test]
    fn test_root() {
        assert_eq!(root("8", "3"), "³√(8)");
    }

    #[test]
    fn test_abs() {
        assert_eq!(abs("x"), "|x|");
    }

    #[test]
    fn test_sum() {
        assert_eq!(sum("i=1", "n", "i"), "∑(i=1 to n) i");
    }

    #[test]
    fn test_product() {
        assert_eq!(product("i=1", "n", "i"), "∏(i=1 to n) i");
    }

    #[test]
    fn test_integral() {
        assert_eq!(integral(Some("0"), Some("1"), "x", "x"), "∫[0 to 1] x dx");
        assert_eq!(integral(None, None, "f(x)", "x"), "∫ f(x) dx");
    }

    #[test]
    fn test_symbols() {
        assert_eq!(infinity(), "∞");
        assert_eq!(multiply_dot(), "·");
        assert_eq!(divide(), "÷");
        assert_eq!(plus_minus(), "±");
        assert_eq!(not_equal(), "≠");
        assert_eq!(less_equal(), "≤");
        assert_eq!(element_of(), "∈");
    }

    #[test]
    fn test_matrix() {
        let rows = vec![
            vec!["1".to_string(), "2".to_string()],
            vec!["3".to_string(), "4".to_string()],
        ];
        let opts = FormatOptions::unicode();
        let result = matrix(&rows, BracketStyle::Square, &opts);
        assert!(result.contains("1"));
        assert!(result.contains("2"));
        assert!(result.contains("3"));
        assert!(result.contains("4"));
        assert!(result.contains('['));
        assert!(result.contains(']'));
    }

    #[test]
    fn test_empty_matrix() {
        let opts = FormatOptions::unicode();
        let result = matrix(&[], BracketStyle::Round, &opts);
        assert_eq!(result, "()");
    }
}
