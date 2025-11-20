//! # RustMath Typesetting System
//!
//! This crate provides a comprehensive typesetting system for mathematical objects,
//! supporting multiple output formats:
//! - LaTeX: Professional mathematical typesetting
//! - ASCII Art: Terminal-friendly rendering
//! - Unicode: Pretty-printed mathematical symbols
//! - HTML/MathML: Web-based mathematical display
//!
//! ## Design Philosophy
//!
//! The typesetting system is built around the `MathDisplay` trait, which allows
//! mathematical objects to render themselves in different formats with customizable
//! options. The system is designed to:
//! - Support operator precedence and parenthesization
//! - Handle alignment and layout (especially for matrices)
//! - Provide both inline and display modes
//! - Support truncation for large structures
//! - Handle generic types that work over any ring/field

pub mod ascii;
pub mod html;
pub mod latex;
pub mod unicode;
pub mod utils;

use std::fmt;

/// Supported output formats for mathematical objects
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    /// LaTeX mathematical typesetting
    LaTeX,
    /// ASCII art rendering for terminals
    Ascii,
    /// Unicode mathematical symbols
    Unicode,
    /// HTML with MathML
    Html,
    /// Plain text (minimal formatting)
    Plain,
}

/// Display mode for mathematical expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    /// Inline mode: compact, single-line when possible
    Inline,
    /// Display mode: centered, multi-line allowed
    Display,
}

/// Configuration options for mathematical typesetting
#[derive(Debug, Clone)]
pub struct FormatOptions {
    /// Output format to use
    pub format: OutputFormat,
    /// Display mode (inline or display)
    pub mode: DisplayMode,
    /// Maximum width for output (None = unlimited)
    pub max_width: Option<usize>,
    /// Maximum number of elements to show before truncation
    pub max_elements: Option<usize>,
    /// Show explicit multiplication symbols (vs implicit)
    pub explicit_multiply: bool,
    /// Operator precedence level (for parenthesization)
    pub precedence: i32,
    /// Use implicit multiplication (e.g., "2x" instead of "2*x")
    pub implicit_multiply: bool,
    /// Use Greek letter substitutions (theta -> θ)
    pub use_greek_letters: bool,
    /// Matrix bracket style
    pub matrix_brackets: BracketStyle,
    /// Number of decimal places for approximate numbers
    pub decimal_places: Option<usize>,
}

/// Bracket styles for matrices and vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BracketStyle {
    /// Square brackets [...]
    Square,
    /// Round parentheses (...)
    Round,
    /// Curly braces {...}
    Curly,
    /// Vertical bars |...|
    Vertical,
    /// Double vertical bars ||...||
    DoubleVertical,
}

impl Default for FormatOptions {
    fn default() -> Self {
        Self {
            format: OutputFormat::Unicode,
            mode: DisplayMode::Inline,
            max_width: Some(120),
            max_elements: Some(100),
            explicit_multiply: false,
            precedence: 0,
            implicit_multiply: true,
            use_greek_letters: true,
            matrix_brackets: BracketStyle::Square,
            decimal_places: Some(6),
        }
    }
}

impl FormatOptions {
    /// Create options for LaTeX output
    pub fn latex() -> Self {
        Self {
            format: OutputFormat::LaTeX,
            ..Default::default()
        }
    }

    /// Create options for ASCII output
    pub fn ascii() -> Self {
        Self {
            format: OutputFormat::Ascii,
            use_greek_letters: false,
            ..Default::default()
        }
    }

    /// Create options for Unicode output
    pub fn unicode() -> Self {
        Self {
            format: OutputFormat::Unicode,
            use_greek_letters: true,
            ..Default::default()
        }
    }

    /// Create options for HTML output
    pub fn html() -> Self {
        Self {
            format: OutputFormat::Html,
            ..Default::default()
        }
    }

    /// Set display mode
    pub fn with_mode(mut self, mode: DisplayMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set precedence level
    pub fn with_precedence(mut self, precedence: i32) -> Self {
        self.precedence = precedence;
        self
    }

    /// Set matrix bracket style
    pub fn with_brackets(mut self, style: BracketStyle) -> Self {
        self.matrix_brackets = style;
        self
    }
}

/// Core trait for mathematical typesetting
///
/// Types implementing this trait can render themselves in multiple formats
/// with customizable options.
pub trait MathDisplay {
    /// Render the mathematical object with the given format options
    fn math_format(&self, options: &FormatOptions) -> String;

    /// Render as LaTeX (convenience method)
    fn to_latex(&self) -> String {
        self.math_format(&FormatOptions::latex())
    }

    /// Render as ASCII art (convenience method)
    fn to_ascii(&self) -> String {
        self.math_format(&FormatOptions::ascii())
    }

    /// Render as Unicode (convenience method)
    fn to_unicode(&self) -> String {
        self.math_format(&FormatOptions::unicode())
    }

    /// Render as HTML/MathML (convenience method)
    fn to_html(&self) -> String {
        self.math_format(&FormatOptions::html())
    }

    /// Get operator precedence for this object (used for parenthesization)
    fn precedence(&self) -> i32 {
        0 // Default: atomic (highest precedence, never needs parens)
    }

    /// Check if this object needs parentheses at the given precedence level
    fn needs_parens(&self, parent_precedence: i32) -> bool {
        self.precedence() > parent_precedence
    }
}

/// Context for rendering mathematical expressions
///
/// Tracks state needed during rendering, such as current precedence level
/// and whether we're in a specific context (numerator, exponent, etc.)
#[derive(Debug, Clone)]
pub struct RenderContext {
    /// Current format options
    pub options: FormatOptions,
    /// Current precedence level
    pub precedence: i32,
    /// Whether we're in a subscript position
    pub in_subscript: bool,
    /// Whether we're in a superscript position
    pub in_superscript: bool,
    /// Whether we're in a fraction numerator
    pub in_numerator: bool,
    /// Whether we're in a fraction denominator
    pub in_denominator: bool,
}

impl RenderContext {
    /// Create a new render context with the given options
    pub fn new(options: FormatOptions) -> Self {
        Self {
            precedence: options.precedence,
            options,
            in_subscript: false,
            in_superscript: false,
            in_numerator: false,
            in_denominator: false,
        }
    }

    /// Create a child context with increased precedence
    pub fn with_precedence(&self, precedence: i32) -> Self {
        let mut ctx = self.clone();
        ctx.precedence = precedence;
        ctx
    }
}

/// Result type for typesetting operations
pub type TypesetResult = Result<String, TypesetError>;

/// Errors that can occur during typesetting
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypesetError {
    /// Output would exceed maximum width
    TooWide,
    /// Structure is too large to render
    TooLarge,
    /// Unsupported operation for this format
    UnsupportedFormat(String),
    /// Generic error with message
    Other(String),
}

impl fmt::Display for TypesetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypesetError::TooWide => write!(f, "Output exceeds maximum width"),
            TypesetError::TooLarge => write!(f, "Structure too large to render"),
            TypesetError::UnsupportedFormat(msg) => write!(f, "Unsupported format: {}", msg),
            TypesetError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for TypesetError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_options_defaults() {
        let opts = FormatOptions::default();
        assert_eq!(opts.format, OutputFormat::Unicode);
        assert_eq!(opts.mode, DisplayMode::Inline);
        assert!(opts.use_greek_letters);
        assert!(opts.implicit_multiply);
    }

    #[test]
    fn test_format_options_builders() {
        let latex = FormatOptions::latex();
        assert_eq!(latex.format, OutputFormat::LaTeX);

        let ascii = FormatOptions::ascii();
        assert_eq!(ascii.format, OutputFormat::Ascii);
        assert!(!ascii.use_greek_letters);

        let unicode = FormatOptions::unicode();
        assert_eq!(unicode.format, OutputFormat::Unicode);
        assert!(unicode.use_greek_letters);
    }

    #[test]
    fn test_format_options_chaining() {
        let opts = FormatOptions::latex()
            .with_mode(DisplayMode::Display)
            .with_precedence(5)
            .with_brackets(BracketStyle::Round);

        assert_eq!(opts.format, OutputFormat::LaTeX);
        assert_eq!(opts.mode, DisplayMode::Display);
        assert_eq!(opts.precedence, 5);
        assert_eq!(opts.matrix_brackets, BracketStyle::Round);
    }

    #[test]
    fn test_render_context_creation() {
        let opts = FormatOptions::latex();
        let ctx = RenderContext::new(opts.clone());

        assert_eq!(ctx.options.format, OutputFormat::LaTeX);
        assert_eq!(ctx.precedence, 0);
        assert!(!ctx.in_subscript);
        assert!(!ctx.in_superscript);
    }

    #[test]
    fn test_render_context_precedence() {
        let opts = FormatOptions::default();
        let ctx = RenderContext::new(opts);
        let child_ctx = ctx.with_precedence(10);

        assert_eq!(child_ctx.precedence, 10);
        assert_eq!(ctx.precedence, 0); // Original unchanged
    }
}
