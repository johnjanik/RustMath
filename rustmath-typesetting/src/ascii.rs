//! ASCII art rendering for mathematical objects

use crate::{BracketStyle, DisplayMode, FormatOptions};

/// Format a fraction as ASCII art
pub fn fraction(numerator: &str, denominator: &str, mode: DisplayMode) -> String {
    match mode {
        DisplayMode::Display => {
            let width = numerator.len().max(denominator.len());
            let num_padding = (width - numerator.len()) / 2;
            let denom_padding = (width - denominator.len()) / 2;

            format!(
                "{}{}\n{}\n{}{}",
                " ".repeat(num_padding),
                numerator,
                "-".repeat(width),
                " ".repeat(denom_padding),
                denominator
            )
        }
        DisplayMode::Inline => format!("{}/{}", numerator, denominator),
    }
}

/// Format a power/exponent as ASCII
pub fn power(base: &str, exponent: &str) -> String {
    format!("{}^{}", base, exponent)
}

/// Format a subscript as ASCII
pub fn subscript(base: &str, sub: &str) -> String {
    format!("{}_{}", base, sub)
}

/// Format a square root as ASCII
pub fn sqrt(content: &str) -> String {
    format!("sqrt({})", content)
}

/// Format an nth root as ASCII
pub fn root(content: &str, n: &str) -> String {
    format!("root({}, {})", n, content)
}

/// Format absolute value as ASCII
pub fn abs(content: &str) -> String {
    format!("|{}|", content)
}

/// Format a sum as ASCII
pub fn sum(lower: &str, upper: &str, body: &str) -> String {
    format!("sum({} to {}, {})", lower, upper, body)
}

/// Format a product as ASCII
pub fn product(lower: &str, upper: &str, body: &str) -> String {
    format!("product({} to {}, {})", lower, upper, body)
}

/// Format an integral as ASCII
pub fn integral(lower: Option<&str>, upper: Option<&str>, integrand: &str, var: &str) -> String {
    match (lower, upper) {
        (Some(l), Some(u)) => format!("integral({} to {}, {} d{})", l, u, integrand, var),
        _ => format!("integral({} d{})", integrand, var),
    }
}

/// Format a limit as ASCII
pub fn limit(var: &str, value: &str, expr: &str) -> String {
    format!("lim({} -> {}, {})", var, value, expr)
}

/// Get opening bracket for ASCII
pub fn opening_bracket(style: BracketStyle) -> &'static str {
    match style {
        BracketStyle::Square => "[",
        BracketStyle::Round => "(",
        BracketStyle::Curly => "{",
        BracketStyle::Vertical => "|",
        BracketStyle::DoubleVertical => "||",
    }
}

/// Get closing bracket for ASCII
pub fn closing_bracket(style: BracketStyle) -> &'static str {
    match style {
        BracketStyle::Square => "]",
        BracketStyle::Round => ")",
        BracketStyle::Curly => "}",
        BracketStyle::Vertical => "|",
        BracketStyle::DoubleVertical => "||",
    }
}

/// Format a matrix as ASCII art
pub fn matrix(rows: &[Vec<String>], bracket_style: BracketStyle, mode: DisplayMode) -> String {
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

    match mode {
        DisplayMode::Display => {
            let mut result = String::new();
            let open = opening_bracket(bracket_style);
            let close = closing_bracket(bracket_style);

            for (i, row) in rows.iter().enumerate() {
                // Opening bracket (only on first row)
                if i == 0 {
                    result.push_str(open);
                } else {
                    result.push_str(&" ".repeat(open.len()));
                }

                result.push(' ');

                // Matrix elements with padding
                for (j, elem) in row.iter().enumerate() {
                    let padding = col_widths[j] - elem.len();
                    result.push_str(&" ".repeat(padding));
                    result.push_str(elem);

                    if j < row.len() - 1 {
                        result.push_str("  ");
                    }
                }

                result.push(' ');

                // Closing bracket (only on first row)
                if i == 0 {
                    result.push_str(close);
                } else {
                    result.push_str(&" ".repeat(close.len()));
                }

                if i < rows.len() - 1 {
                    result.push('\n');
                }
            }

            result
        }
        DisplayMode::Inline => {
            let open = opening_bracket(bracket_style);
            let close = closing_bracket(bracket_style);
            let mut result = String::from(open);

            for (i, row) in rows.iter().enumerate() {
                result.push('[');
                for (j, elem) in row.iter().enumerate() {
                    result.push_str(elem);
                    if j < row.len() - 1 {
                        result.push_str(", ");
                    }
                }
                result.push(']');

                if i < rows.len() - 1 {
                    result.push_str(", ");
                }
            }

            result.push_str(close);
            result
        }
    }
}

/// Format a vector as ASCII art
pub fn vector(elements: &[String], bracket_style: BracketStyle, mode: DisplayMode) -> String {
    match mode {
        DisplayMode::Display => {
            let rows: Vec<Vec<String>> = elements.iter().map(|e| vec![e.clone()]).collect();
            matrix(&rows, bracket_style, mode)
        }
        DisplayMode::Inline => {
            let open = opening_bracket(bracket_style);
            let close = closing_bracket(bracket_style);
            format!("{}{}{}", open, elements.join(", "), close)
        }
    }
}

/// Create ASCII art box around text
pub fn box_text(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let width = lines.iter().map(|l| l.len()).max().unwrap_or(0);

    let mut result = String::new();

    // Top border
    result.push_str(&format!("+{}+\n", "-".repeat(width + 2)));

    // Content
    for line in lines {
        let padding = width - line.len();
        result.push_str(&format!("| {}{} |\n", line, " ".repeat(padding)));
    }

    // Bottom border
    result.push_str(&format!("+{}+", "-".repeat(width + 2)));

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fraction_display() {
        let result = fraction("123", "456", DisplayMode::Display);
        assert!(result.contains("123"));
        assert!(result.contains("456"));
        assert!(result.contains("---"));
        assert!(result.contains('\n'));
    }

    #[test]
    fn test_fraction_inline() {
        assert_eq!(fraction("1", "2", DisplayMode::Inline), "1/2");
    }

    #[test]
    fn test_power() {
        assert_eq!(power("x", "2"), "x^2");
    }

    #[test]
    fn test_subscript() {
        assert_eq!(subscript("x", "0"), "x_0");
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(sqrt("2"), "sqrt(2)");
    }

    #[test]
    fn test_root() {
        assert_eq!(root("8", "3"), "root(3, 8)");
    }

    #[test]
    fn test_abs() {
        assert_eq!(abs("x"), "|x|");
    }

    #[test]
    fn test_sum() {
        assert_eq!(sum("i=1", "n", "i"), "sum(i=1 to n, i)");
    }

    #[test]
    fn test_matrix_display() {
        let rows = vec![
            vec!["1".to_string(), "2".to_string()],
            vec!["3".to_string(), "4".to_string()],
        ];
        let result = matrix(&rows, BracketStyle::Square, DisplayMode::Display);
        assert!(result.contains("1"));
        assert!(result.contains("2"));
        assert!(result.contains("3"));
        assert!(result.contains("4"));
        assert!(result.contains('['));
        assert!(result.contains(']'));
    }

    #[test]
    fn test_matrix_inline() {
        let rows = vec![
            vec!["1".to_string(), "2".to_string()],
            vec!["3".to_string(), "4".to_string()],
        ];
        let result = matrix(&rows, BracketStyle::Round, DisplayMode::Inline);
        assert!(result.contains("1"));
        assert!(result.contains("2"));
        assert!(result.contains("3"));
        assert!(result.contains("4"));
        assert!(result.starts_with('('));
        assert!(result.ends_with(')'));
    }

    #[test]
    fn test_vector_inline() {
        let elements = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let result = vector(&elements, BracketStyle::Square, DisplayMode::Inline);
        assert_eq!(result, "[1, 2, 3]");
    }

    #[test]
    fn test_box_text() {
        let result = box_text("Hello");
        assert!(result.contains("Hello"));
        assert!(result.contains('+'));
        assert!(result.contains('-'));
        assert!(result.contains('|'));
    }
}
