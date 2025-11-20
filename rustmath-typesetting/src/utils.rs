//! Utility functions for mathematical typesetting

use crate::{BracketStyle, OutputFormat};

/// Operator precedence levels (higher = tighter binding)
pub mod precedence {
    /// Atomic expressions (numbers, variables) - never need parens
    pub const ATOMIC: i32 = 0;
    /// Function calls and subscripts
    pub const FUNCTION: i32 = 1;
    /// Exponentiation
    pub const POWER: i32 = 2;
    /// Unary minus and plus
    pub const UNARY: i32 = 3;
    /// Implicit multiplication
    pub const IMPLICIT_MULTIPLY: i32 = 4;
    /// Multiplication and division
    pub const MULTIPLY: i32 = 5;
    /// Addition and subtraction
    pub const ADD: i32 = 6;
    /// Relations (=, <, >, etc.)
    pub const RELATION: i32 = 7;
    /// Logical operations
    pub const LOGICAL: i32 = 8;
}

/// Get the opening bracket for a given style and format
pub fn opening_bracket(style: BracketStyle, format: OutputFormat) -> &'static str {
    match (style, format) {
        (BracketStyle::Square, OutputFormat::LaTeX) => r"\left[",
        (BracketStyle::Square, _) => "[",
        (BracketStyle::Round, OutputFormat::LaTeX) => r"\left(",
        (BracketStyle::Round, _) => "(",
        (BracketStyle::Curly, OutputFormat::LaTeX) => r"\left\{",
        (BracketStyle::Curly, _) => "{",
        (BracketStyle::Vertical, OutputFormat::LaTeX) => r"\left|",
        (BracketStyle::Vertical, _) => "|",
        (BracketStyle::DoubleVertical, OutputFormat::LaTeX) => r"\left\|",
        (BracketStyle::DoubleVertical, OutputFormat::Unicode) => "‖",
        (BracketStyle::DoubleVertical, _) => "||",
    }
}

/// Get the closing bracket for a given style and format
pub fn closing_bracket(style: BracketStyle, format: OutputFormat) -> &'static str {
    match (style, format) {
        (BracketStyle::Square, OutputFormat::LaTeX) => r"\right]",
        (BracketStyle::Square, _) => "]",
        (BracketStyle::Round, OutputFormat::LaTeX) => r"\right)",
        (BracketStyle::Round, _) => ")",
        (BracketStyle::Curly, OutputFormat::LaTeX) => r"\right\}",
        (BracketStyle::Curly, _) => "}",
        (BracketStyle::Vertical, OutputFormat::LaTeX) => r"\right|",
        (BracketStyle::Vertical, _) => "|",
        (BracketStyle::DoubleVertical, OutputFormat::LaTeX) => r"\right\|",
        (BracketStyle::DoubleVertical, OutputFormat::Unicode) => "‖",
        (BracketStyle::DoubleVertical, _) => "||",
    }
}

/// Convert a variable name to use Greek letters if appropriate
pub fn greek_substitute(name: &str, format: OutputFormat, use_greek: bool) -> String {
    if !use_greek {
        return name.to_string();
    }

    match format {
        OutputFormat::LaTeX => match name.to_lowercase().as_str() {
            "alpha" => r"\alpha".to_string(),
            "beta" => r"\beta".to_string(),
            "gamma" => r"\gamma".to_string(),
            "delta" => r"\delta".to_string(),
            "epsilon" => r"\epsilon".to_string(),
            "zeta" => r"\zeta".to_string(),
            "eta" => r"\eta".to_string(),
            "theta" => r"\theta".to_string(),
            "iota" => r"\iota".to_string(),
            "kappa" => r"\kappa".to_string(),
            "lambda" => r"\lambda".to_string(),
            "mu" => r"\mu".to_string(),
            "nu" => r"\nu".to_string(),
            "xi" => r"\xi".to_string(),
            "pi" => r"\pi".to_string(),
            "rho" => r"\rho".to_string(),
            "sigma" => r"\sigma".to_string(),
            "tau" => r"\tau".to_string(),
            "upsilon" => r"\upsilon".to_string(),
            "phi" => r"\phi".to_string(),
            "chi" => r"\chi".to_string(),
            "psi" => r"\psi".to_string(),
            "omega" => r"\omega".to_string(),
            _ => name.to_string(),
        },
        OutputFormat::Unicode => match name.to_lowercase().as_str() {
            "alpha" => "α".to_string(),
            "beta" => "β".to_string(),
            "gamma" => "γ".to_string(),
            "delta" => "δ".to_string(),
            "epsilon" => "ε".to_string(),
            "zeta" => "ζ".to_string(),
            "eta" => "η".to_string(),
            "theta" => "θ".to_string(),
            "iota" => "ι".to_string(),
            "kappa" => "κ".to_string(),
            "lambda" => "λ".to_string(),
            "mu" => "μ".to_string(),
            "nu" => "ν".to_string(),
            "xi" => "ξ".to_string(),
            "pi" => "π".to_string(),
            "rho" => "ρ".to_string(),
            "sigma" => "σ".to_string(),
            "tau" => "τ".to_string(),
            "upsilon" => "υ".to_string(),
            "phi" => "φ".to_string(),
            "chi" => "χ".to_string(),
            "psi" => "ψ".to_string(),
            "omega" => "ω".to_string(),
            _ => name.to_string(),
        },
        _ => name.to_string(),
    }
}

/// Convert a digit to its Unicode subscript form
///
/// Note: For more comprehensive subscript support including letters,
/// use `unicode_art::unicode_subscript` instead.
pub fn subscript_digit(c: char) -> char {
    match c {
        '0' => '₀',
        '1' => '₁',
        '2' => '₂',
        '3' => '₃',
        '4' => '₄',
        '5' => '₅',
        '6' => '₆',
        '7' => '₇',
        '8' => '₈',
        '9' => '₉',
        _ => c,
    }
}

/// Convert a digit to its Unicode superscript form
///
/// Note: For more comprehensive superscript support including letters,
/// use `unicode_art::unicode_superscript` instead.
pub fn superscript_digit(c: char) -> char {
    match c {
        '0' => '⁰',
        '1' => '¹',
        '2' => '²',
        '3' => '³',
        '4' => '⁴',
        '5' => '⁵',
        '6' => '⁶',
        '7' => '⁷',
        '8' => '⁸',
        '9' => '⁹',
        '+' => '⁺',
        '-' => '⁻',
        '=' => '⁼',
        '(' => '⁽',
        ')' => '⁾',
        _ => c,
    }
}

/// Convert a string to Unicode subscript
///
/// This is a simple version that only handles digits and basic operators.
/// For full letter support, use `unicode_art::unicode_subscript`.
pub fn to_subscript(s: &str) -> String {
    crate::unicode_art::unicode_subscript(s)
}

/// Convert a string to Unicode superscript
///
/// This is a simple version that delegates to the comprehensive implementation.
/// For full letter support, use `unicode_art::unicode_superscript`.
pub fn to_superscript(s: &str) -> String {
    crate::unicode_art::unicode_superscript(s)
}

/// Wrap content in parentheses if needed based on precedence
pub fn maybe_parens(content: &str, parent_prec: i32, self_prec: i32) -> String {
    if self_prec > parent_prec {
        format!("({})", content)
    } else {
        content.to_string()
    }
}

/// Format a multiplication operator based on options
pub fn multiply_symbol(format: OutputFormat, explicit: bool) -> &'static str {
    if !explicit {
        return "";
    }

    match format {
        OutputFormat::LaTeX => r"\cdot ",
        OutputFormat::Unicode => "·",
        OutputFormat::Html => "&middot;",
        _ => "*",
    }
}

/// Format a division operator based on format
pub fn divide_symbol(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Unicode => "÷",
        OutputFormat::Html => "&divide;",
        _ => "/",
    }
}

/// Truncate a string with ellipsis if it exceeds max length
pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Calculate the maximum width of strings in a collection
pub fn max_width<'a, I>(items: I) -> usize
where
    I: IntoIterator<Item = &'a str>,
{
    items.into_iter().map(|s| s.len()).max().unwrap_or(0)
}

/// Pad a string to a given width (right-aligned for numbers)
pub fn pad_right(s: &str, width: usize) -> String {
    let padding = width.saturating_sub(s.len());
    format!("{}{}", " ".repeat(padding), s)
}

/// Pad a string to a given width (left-aligned for text)
pub fn pad_left(s: &str, width: usize) -> String {
    let padding = width.saturating_sub(s.len());
    format!("{}{}", s, " ".repeat(padding))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precedence_ordering() {
        assert!(precedence::ATOMIC < precedence::POWER);
        assert!(precedence::POWER < precedence::MULTIPLY);
        assert!(precedence::MULTIPLY < precedence::ADD);
    }

    #[test]
    fn test_brackets() {
        assert_eq!(
            opening_bracket(BracketStyle::Square, OutputFormat::LaTeX),
            r"\left["
        );
        assert_eq!(
            closing_bracket(BracketStyle::Square, OutputFormat::LaTeX),
            r"\right]"
        );
        assert_eq!(opening_bracket(BracketStyle::Round, OutputFormat::Unicode), "(");
    }

    #[test]
    fn test_greek_substitute() {
        assert_eq!(
            greek_substitute("theta", OutputFormat::Unicode, true),
            "θ"
        );
        assert_eq!(
            greek_substitute("pi", OutputFormat::LaTeX, true),
            r"\pi"
        );
        assert_eq!(
            greek_substitute("theta", OutputFormat::Unicode, false),
            "theta"
        );
        assert_eq!(greek_substitute("x", OutputFormat::Unicode, true), "x");
    }

    #[test]
    fn test_subscript() {
        assert_eq!(to_subscript("123"), "₁₂₃");
        assert_eq!(to_subscript("0"), "₀");
    }

    #[test]
    fn test_superscript() {
        assert_eq!(to_superscript("123"), "¹²³");
        assert_eq!(to_superscript("(n)"), "⁽ⁿ⁾");
    }

    #[test]
    fn test_maybe_parens() {
        assert_eq!(maybe_parens("x+y", precedence::MULTIPLY, precedence::ADD), "(x+y)");
        assert_eq!(maybe_parens("x*y", precedence::ADD, precedence::MULTIPLY), "x*y");
    }

    #[test]
    fn test_symbols() {
        assert_eq!(multiply_symbol(OutputFormat::Unicode, true), "·");
        assert_eq!(multiply_symbol(OutputFormat::LaTeX, true), r"\cdot ");
        assert_eq!(divide_symbol(OutputFormat::Unicode), "÷");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_max_width() {
        let items = vec!["a", "abc", "ab"];
        assert_eq!(max_width(items.iter().copied()), 3);
    }

    #[test]
    fn test_padding() {
        assert_eq!(pad_right("5", 3), "  5");
        assert_eq!(pad_left("x", 3), "x  ");
    }
}
