//! # Unicode Art Module
//!
//! This module provides comprehensive Unicode art rendering for mathematical objects,
//! including subscript and superscript conversions with extensive character support.
//!
//! ## Unicode Ranges Used
//!
//! - **Superscripts** (U+2070 - U+207F): Superscript digits and mathematical operators
//!   - Digits: ⁰¹²³⁴⁵⁶⁷⁸⁹
//!   - Signs: ⁺⁻⁼
//!   - Parentheses: ⁽⁾
//!   - Latin letters (scattered): ⁱⁿ and others in U+1D2C - U+1D9C
//!
//! - **Subscripts** (U+2080 - U+209F): Subscript digits and operators
//!   - Digits: ₀₁₂₃₄₅₆₇₈₉
//!   - Signs: ₊₋₌
//!   - Parentheses: ₍₎
//!   - Latin letters: ₐₑₒₓₔ (U+2090 - U+209C)
//!
//! - **Mathematical Alphanumeric Symbols** (U+1D00 - U+1D7FF):
//!   - Modifier letters used as superscripts
//!   - Various mathematical alphabets
//!
//! ## Limitations
//!
//! Not all ASCII characters have Unicode subscript/superscript equivalents.
//! Characters without equivalents are returned unchanged or wrapped in
//! parentheses for clarity.

use std::collections::HashMap;
use once_cell::sync::Lazy;

// ============================================================================
// SUPERSCRIPT CHARACTER MAPPINGS
// ============================================================================

/// Map ASCII characters to their Unicode superscript equivalents
///
/// Unicode ranges:
/// - U+2070-207F: Superscripts and Subscripts block
/// - U+1D2C-1D9C: Phonetic Extensions and Modifier Letters
static SUPERSCRIPT_MAP: Lazy<HashMap<char, char>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Digits (U+2070, U+00B9, U+00B2, U+00B3, U+2074-2079)
    map.insert('0', '⁰');  // U+2070
    map.insert('1', '¹');  // U+00B9
    map.insert('2', '²');  // U+00B2
    map.insert('3', '³');  // U+00B3
    map.insert('4', '⁴');  // U+2074
    map.insert('5', '⁵');  // U+2075
    map.insert('6', '⁶');  // U+2076
    map.insert('7', '⁷');  // U+2077
    map.insert('8', '⁸');  // U+2078
    map.insert('9', '⁹');  // U+2079

    // Mathematical operators (U+207A-207E)
    map.insert('+', '⁺');  // U+207A
    map.insert('-', '⁻');  // U+207B (also minus)
    map.insert('=', '⁼');  // U+207C
    map.insert('(', '⁽');  // U+207D
    map.insert(')', '⁾');  // U+207E

    // Lowercase letters (scattered across multiple Unicode blocks)
    map.insert('a', 'ᵃ');  // U+1D43
    map.insert('b', 'ᵇ');  // U+1D47
    map.insert('c', 'ᶜ');  // U+1D9C
    map.insert('d', 'ᵈ');  // U+1D48
    map.insert('e', 'ᵉ');  // U+1D49
    map.insert('f', 'ᶠ');  // U+1DA0
    map.insert('g', 'ᵍ');  // U+1D4D
    map.insert('h', 'ʰ');  // U+02B0
    map.insert('i', 'ⁱ');  // U+2071
    map.insert('j', 'ʲ');  // U+02B2
    map.insert('k', 'ᵏ');  // U+1D4F
    map.insert('l', 'ˡ');  // U+02E1
    map.insert('m', 'ᵐ');  // U+1D50
    map.insert('n', 'ⁿ');  // U+207F
    map.insert('o', 'ᵒ');  // U+1D52
    map.insert('p', 'ᵖ');  // U+1D56
    map.insert('r', 'ʳ');  // U+02B3
    map.insert('s', 'ˢ');  // U+02E2
    map.insert('t', 'ᵗ');  // U+1D57
    map.insert('u', 'ᵘ');  // U+1D58
    map.insert('v', 'ᵛ');  // U+1D5B
    map.insert('w', 'ʷ');  // U+02B7
    map.insert('x', 'ˣ');  // U+02E3
    map.insert('y', 'ʸ');  // U+02B8
    map.insert('z', 'ᶻ');  // U+1DBB

    // Uppercase letters (limited availability)
    map.insert('A', 'ᴬ');  // U+1D2C
    map.insert('B', 'ᴮ');  // U+1D2E
    map.insert('D', 'ᴰ');  // U+1D30
    map.insert('E', 'ᴱ');  // U+1D31
    map.insert('G', 'ᴳ');  // U+1D33
    map.insert('H', 'ᴴ');  // U+1D34
    map.insert('I', 'ᴵ');  // U+1D35
    map.insert('J', 'ᴶ');  // U+1D36
    map.insert('K', 'ᴷ');  // U+1D37
    map.insert('L', 'ᴸ');  // U+1D38
    map.insert('M', 'ᴹ');  // U+1D39
    map.insert('N', 'ᴺ');  // U+1D3A
    map.insert('O', 'ᴼ');  // U+1D3C
    map.insert('P', 'ᴾ');  // U+1D3E
    map.insert('R', 'ᴿ');  // U+1D3F
    map.insert('T', 'ᵀ');  // U+1D40
    map.insert('U', 'ᵁ');  // U+1D41
    map.insert('V', 'ⱽ');  // U+2C7D
    map.insert('W', 'ᵂ');  // U+1D42

    // Greek letters (limited)
    map.insert('α', 'ᵅ');  // U+1D45
    map.insert('β', 'ᵝ');  // U+1D5D
    map.insert('γ', 'ᵞ');  // U+1D5E
    map.insert('δ', 'ᵟ');  // U+1D5F
    map.insert('θ', 'ᶿ');  // U+1DBF
    map.insert('φ', 'ᵠ');  // U+1D60
    map.insert('χ', 'ᵡ');  // U+1D61

    map
});

// ============================================================================
// SUBSCRIPT CHARACTER MAPPINGS
// ============================================================================

/// Map ASCII characters to their Unicode subscript equivalents
///
/// Unicode ranges:
/// - U+2080-209C: Subscripts block
static SUBSCRIPT_MAP: Lazy<HashMap<char, char>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Digits (U+2080-2089)
    map.insert('0', '₀');  // U+2080
    map.insert('1', '₁');  // U+2081
    map.insert('2', '₂');  // U+2082
    map.insert('3', '₃');  // U+2083
    map.insert('4', '₄');  // U+2084
    map.insert('5', '₅');  // U+2085
    map.insert('6', '₆');  // U+2086
    map.insert('7', '₇');  // U+2087
    map.insert('8', '₈');  // U+2088
    map.insert('9', '₉');  // U+2089

    // Mathematical operators (U+208A-208E)
    map.insert('+', '₊');  // U+208A
    map.insert('-', '₋');  // U+208B
    map.insert('=', '₌');  // U+208C
    map.insert('(', '₍');  // U+208D
    map.insert(')', '₎');  // U+208E

    // Lowercase letters (U+2090-209C) - very limited
    map.insert('a', 'ₐ');  // U+2090
    map.insert('e', 'ₑ');  // U+2091
    map.insert('o', 'ₒ');  // U+2092
    map.insert('x', 'ₓ');  // U+2093
    map.insert('ə', 'ₔ');  // U+2094 (schwa)
    map.insert('h', 'ₕ');  // U+2095
    map.insert('k', 'ₖ');  // U+2096
    map.insert('l', 'ₗ');  // U+2097
    map.insert('m', 'ₘ');  // U+2098
    map.insert('n', 'ₙ');  // U+2099
    map.insert('p', 'ₚ');  // U+209A
    map.insert('s', 'ₛ');  // U+209B
    map.insert('t', 'ₜ');  // U+209C

    // Greek letters (very limited)
    map.insert('β', 'ᵦ');  // U+1D66
    map.insert('γ', 'ᵧ');  // U+1D67
    map.insert('ρ', 'ᵨ');  // U+1D68
    map.insert('φ', 'ᵩ');  // U+1D69
    map.insert('χ', 'ᵪ');  // U+1D6A

    map
});

// ============================================================================
// SPECIAL MATHEMATICAL SYMBOLS
// ============================================================================

/// Additional mathematical symbols used in Unicode art
pub mod symbols {
    /// Horizontal box drawing
    pub const HORIZONTAL: char = '─';
    pub const VERTICAL: char = '│';
    pub const TOP_LEFT: char = '┌';
    pub const TOP_RIGHT: char = '┐';
    pub const BOTTOM_LEFT: char = '└';
    pub const BOTTOM_RIGHT: char = '┘';

    /// Mathematical operators
    pub const INFINITY: char = '∞';
    pub const MULTIPLY: char = '×';
    pub const DOT_MULTIPLY: char = '·';
    pub const DIVIDE: char = '÷';
    pub const MINUS: char = '−';
    pub const PLUS_MINUS: char = '±';
    pub const MINUS_PLUS: char = '∓';

    /// Comparison operators
    pub const LESS_EQUAL: char = '≤';
    pub const GREATER_EQUAL: char = '≥';
    pub const NOT_EQUAL: char = '≠';
    pub const APPROX: char = '≈';
    pub const EQUIV: char = '≡';
    pub const PROPORTIONAL: char = '∝';

    /// Set theory
    pub const ELEMENT_OF: char = '∈';
    pub const NOT_ELEMENT_OF: char = '∉';
    pub const SUBSET: char = '⊂';
    pub const SUPERSET: char = '⊃';
    pub const SUBSET_EQ: char = '⊆';
    pub const SUPERSET_EQ: char = '⊇';
    pub const UNION: char = '∪';
    pub const INTERSECTION: char = '∩';
    pub const EMPTY_SET: char = '∅';

    /// Calculus
    pub const INTEGRAL: char = '∫';
    pub const DOUBLE_INTEGRAL: char = '∬';
    pub const TRIPLE_INTEGRAL: char = '∭';
    pub const CONTOUR_INTEGRAL: char = '∮';
    pub const PARTIAL: char = '∂';
    pub const NABLA: char = '∇';

    /// Logic
    pub const FORALL: char = '∀';
    pub const EXISTS: char = '∃';
    pub const NOT_EXISTS: char = '∄';
    pub const AND: char = '∧';
    pub const OR: char = '∨';
    pub const NOT: char = '¬';
    pub const IMPLIES: char = '⇒';
    pub const IFF: char = '⇔';

    /// Greek alphabet (commonly used)
    pub const ALPHA: char = 'α';
    pub const BETA: char = 'β';
    pub const GAMMA: char = 'γ';
    pub const DELTA: char = 'δ';
    pub const EPSILON: char = 'ε';
    pub const THETA: char = 'θ';
    pub const LAMBDA: char = 'λ';
    pub const MU: char = 'μ';
    pub const PI: char = 'π';
    pub const SIGMA: char = 'σ';
    pub const PHI: char = 'φ';
    pub const OMEGA: char = 'ω';
}

// ============================================================================
// PUBLIC API FUNCTIONS
// ============================================================================

/// Convert text to Unicode superscript
///
/// This function converts ASCII characters to their Unicode superscript equivalents
/// where available. Characters without superscript equivalents are returned unchanged.
///
/// # Examples
///
/// ```
/// use rustmath_typesetting::unicode_art::unicode_superscript;
///
/// assert_eq!(unicode_superscript("123"), "¹²³");
/// assert_eq!(unicode_superscript("n+1"), "ⁿ⁺¹");
/// assert_eq!(unicode_superscript("2x"), "²ˣ");
/// ```
///
/// # Unicode Ranges
///
/// - Digits: U+2070, U+00B9-00B3, U+2074-2079
/// - Letters: U+1D2C-1D42 (uppercase), U+02B0-02E3 (lowercase modifiers)
/// - Operators: U+207A-207E
pub fn unicode_superscript(text: &str) -> String {
    text.chars()
        .map(|c| *SUPERSCRIPT_MAP.get(&c).unwrap_or(&c))
        .collect()
}

/// Convert text to Unicode subscript
///
/// This function converts ASCII characters to their Unicode subscript equivalents
/// where available. Note that subscript support is more limited than superscript.
/// Characters without subscript equivalents are returned unchanged.
///
/// # Examples
///
/// ```
/// use rustmath_typesetting::unicode_art::unicode_subscript;
///
/// assert_eq!(unicode_subscript("123"), "₁₂₃");
/// assert_eq!(unicode_subscript("n+1"), "ₙ₊₁");
/// assert_eq!(unicode_subscript("a0"), "ₐ₀");
/// ```
///
/// # Unicode Ranges
///
/// - Digits: U+2080-2089
/// - Letters: U+2090-209C (very limited set)
/// - Operators: U+208A-208E
pub fn unicode_subscript(text: &str) -> String {
    text.chars()
        .map(|c| *SUBSCRIPT_MAP.get(&c).unwrap_or(&c))
        .collect()
}

/// Trait for types that can render themselves as Unicode art
///
/// This trait allows mathematical objects to provide their own Unicode art
/// representation. It's similar to `Display` but specifically for Unicode
/// mathematical typesetting.
///
/// # Examples
///
/// ```
/// use rustmath_typesetting::unicode_art::UnicodeArt;
///
/// struct Fraction { num: i32, den: i32 }
///
/// impl UnicodeArt for Fraction {
///     fn unicode_art(&self) -> String {
///         if self.den == 1 {
///             self.num.to_string()
///         } else {
///             format!("{}/{}", self.num, self.den)
///         }
///     }
/// }
///
/// let f = Fraction { num: 3, den: 4 };
/// assert_eq!(f.unicode_art(), "3/4");
/// ```
pub trait UnicodeArt {
    /// Render this object as Unicode art
    fn unicode_art(&self) -> String;
}

/// Render any object implementing `UnicodeArt` as Unicode art
///
/// This is a convenience function that calls the `unicode_art` method
/// on objects implementing the `UnicodeArt` trait.
///
/// # Examples
///
/// ```
/// use rustmath_typesetting::unicode_art::{UnicodeArt, unicode_art};
///
/// struct Complex { re: f64, im: f64 }
///
/// impl UnicodeArt for Complex {
///     fn unicode_art(&self) -> String {
///         if self.im >= 0.0 {
///             format!("{} + {}i", self.re, self.im)
///         } else {
///             format!("{} - {}i", self.re, -self.im)
///         }
///     }
/// }
///
/// let z = Complex { re: 3.0, im: 4.0 };
/// assert_eq!(unicode_art(&z), "3 + 4i");
/// ```
pub fn unicode_art<T: UnicodeArt>(obj: &T) -> String {
    obj.unicode_art()
}

// ============================================================================
// BUILT-IN IMPLEMENTATIONS
// ============================================================================

impl UnicodeArt for i32 {
    fn unicode_art(&self) -> String {
        self.to_string()
    }
}

impl UnicodeArt for i64 {
    fn unicode_art(&self) -> String {
        self.to_string()
    }
}

impl UnicodeArt for f64 {
    fn unicode_art(&self) -> String {
        if self.is_infinite() {
            if self.is_sign_positive() {
                symbols::INFINITY.to_string()
            } else {
                format!("-{}", symbols::INFINITY)
            }
        } else if self.is_nan() {
            "NaN".to_string()
        } else {
            self.to_string()
        }
    }
}

impl UnicodeArt for String {
    fn unicode_art(&self) -> String {
        self.clone()
    }
}

impl UnicodeArt for &str {
    fn unicode_art(&self) -> String {
        self.to_string()
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Check if a character has a superscript equivalent
pub fn has_superscript(c: char) -> bool {
    SUPERSCRIPT_MAP.contains_key(&c)
}

/// Check if a character has a subscript equivalent
pub fn has_subscript(c: char) -> bool {
    SUBSCRIPT_MAP.contains_key(&c)
}

/// Get all characters that have superscript equivalents
pub fn superscript_chars() -> Vec<char> {
    let mut chars: Vec<char> = SUPERSCRIPT_MAP.keys().copied().collect();
    chars.sort();
    chars
}

/// Get all characters that have subscript equivalents
pub fn subscript_chars() -> Vec<char> {
    let mut chars: Vec<char> = SUBSCRIPT_MAP.keys().copied().collect();
    chars.sort();
    chars
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test superscript digits
    #[test]
    fn test_superscript_digits() {
        assert_eq!(unicode_superscript("0"), "⁰");
        assert_eq!(unicode_superscript("1"), "¹");
        assert_eq!(unicode_superscript("2"), "²");
        assert_eq!(unicode_superscript("3"), "³");
        assert_eq!(unicode_superscript("4"), "⁴");
        assert_eq!(unicode_superscript("5"), "⁵");
        assert_eq!(unicode_superscript("6"), "⁶");
        assert_eq!(unicode_superscript("7"), "⁷");
        assert_eq!(unicode_superscript("8"), "⁸");
        assert_eq!(unicode_superscript("9"), "⁹");
        assert_eq!(unicode_superscript("0123456789"), "⁰¹²³⁴⁵⁶⁷⁸⁹");
    }

    #[test]
    fn test_superscript_operators() {
        assert_eq!(unicode_superscript("+"), "⁺");
        assert_eq!(unicode_superscript("-"), "⁻");
        assert_eq!(unicode_superscript("="), "⁼");
        assert_eq!(unicode_superscript("("), "⁽");
        assert_eq!(unicode_superscript(")"), "⁾");
        assert_eq!(unicode_superscript("(x+1)"), "⁽ˣ⁺¹⁾");
    }

    #[test]
    fn test_superscript_lowercase_letters() {
        assert_eq!(unicode_superscript("a"), "ᵃ");
        assert_eq!(unicode_superscript("b"), "ᵇ");
        assert_eq!(unicode_superscript("c"), "ᶜ");
        assert_eq!(unicode_superscript("n"), "ⁿ");
        assert_eq!(unicode_superscript("x"), "ˣ");
        assert_eq!(unicode_superscript("abc"), "ᵃᵇᶜ");
    }

    #[test]
    fn test_superscript_uppercase_letters() {
        assert_eq!(unicode_superscript("A"), "ᴬ");
        assert_eq!(unicode_superscript("B"), "ᴮ");
        assert_eq!(unicode_superscript("N"), "ᴺ");
        assert_eq!(unicode_superscript("T"), "ᵀ");
    }

    #[test]
    fn test_superscript_expressions() {
        assert_eq!(unicode_superscript("n+1"), "ⁿ⁺¹");
        assert_eq!(unicode_superscript("2n"), "²ⁿ");
        assert_eq!(unicode_superscript("x2"), "ˣ²");
        assert_eq!(unicode_superscript("10"), "¹⁰");
    }

    #[test]
    fn test_subscript_digits() {
        assert_eq!(unicode_subscript("0"), "₀");
        assert_eq!(unicode_subscript("1"), "₁");
        assert_eq!(unicode_subscript("2"), "₂");
        assert_eq!(unicode_subscript("3"), "₃");
        assert_eq!(unicode_subscript("4"), "₄");
        assert_eq!(unicode_subscript("5"), "₅");
        assert_eq!(unicode_subscript("6"), "₆");
        assert_eq!(unicode_subscript("7"), "₇");
        assert_eq!(unicode_subscript("8"), "₈");
        assert_eq!(unicode_subscript("9"), "₉");
        assert_eq!(unicode_subscript("0123456789"), "₀₁₂₃₄₅₆₇₈₉");
    }

    #[test]
    fn test_subscript_operators() {
        assert_eq!(unicode_subscript("+"), "₊");
        assert_eq!(unicode_subscript("-"), "₋");
        assert_eq!(unicode_subscript("="), "₌");
        assert_eq!(unicode_subscript("("), "₍");
        assert_eq!(unicode_subscript(")"), "₎");
    }

    #[test]
    fn test_subscript_letters() {
        assert_eq!(unicode_subscript("a"), "ₐ");
        assert_eq!(unicode_subscript("e"), "ₑ");
        assert_eq!(unicode_subscript("o"), "ₒ");
        assert_eq!(unicode_subscript("x"), "ₓ");
        assert_eq!(unicode_subscript("n"), "ₙ");
        assert_eq!(unicode_subscript("aeo"), "ₐₑₒ");
    }

    #[test]
    fn test_subscript_expressions() {
        assert_eq!(unicode_subscript("n+1"), "ₙ₊₁");
        assert_eq!(unicode_subscript("a0"), "ₐ₀");
        assert_eq!(unicode_subscript("x1"), "ₓ₁");
    }

    #[test]
    fn test_mixed_characters() {
        // Characters without mappings should pass through unchanged
        let result = unicode_superscript("abc123XYZ");
        assert!(result.contains('ᵃ'));
        assert!(result.contains('¹'));
        assert!(result.contains('²'));
        assert!(result.contains('³'));
    }

    #[test]
    fn test_unicode_art_trait_i32() {
        let x: i32 = 42;
        assert_eq!(x.unicode_art(), "42");
        assert_eq!(unicode_art(&x), "42");
    }

    #[test]
    fn test_unicode_art_trait_i64() {
        let x: i64 = -123;
        assert_eq!(x.unicode_art(), "-123");
    }

    #[test]
    fn test_unicode_art_trait_f64() {
        let x = 3.14;
        assert_eq!(x.unicode_art(), "3.14");

        let inf = f64::INFINITY;
        assert_eq!(inf.unicode_art(), "∞");

        let neg_inf = f64::NEG_INFINITY;
        assert_eq!(neg_inf.unicode_art(), "-∞");

        let nan = f64::NAN;
        assert_eq!(nan.unicode_art(), "NaN");
    }

    #[test]
    fn test_unicode_art_trait_string() {
        let s = "Hello".to_string();
        assert_eq!(s.unicode_art(), "Hello");

        let s = "x²";
        assert_eq!(s.unicode_art(), "x²");
    }

    #[test]
    fn test_has_superscript() {
        assert!(has_superscript('0'));
        assert!(has_superscript('a'));
        assert!(has_superscript('n'));
        assert!(has_superscript('+'));
        assert!(!has_superscript('!'));
        assert!(!has_superscript('&'));
    }

    #[test]
    fn test_has_subscript() {
        assert!(has_subscript('0'));
        assert!(has_subscript('a'));
        assert!(has_subscript('x'));
        assert!(!has_subscript('b'));  // 'b' doesn't have subscript
        assert!(!has_subscript('!'));
    }

    #[test]
    fn test_superscript_chars() {
        let chars = superscript_chars();
        assert!(chars.contains(&'0'));
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'n'));
        assert!(chars.contains(&'+'));
    }

    #[test]
    fn test_subscript_chars() {
        let chars = subscript_chars();
        assert!(chars.contains(&'0'));
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'x'));
        assert!(chars.contains(&'+'));
    }

    #[test]
    fn test_symbols() {
        assert_eq!(symbols::INFINITY, '∞');
        assert_eq!(symbols::MULTIPLY, '×');
        assert_eq!(symbols::ELEMENT_OF, '∈');
        assert_eq!(symbols::INTEGRAL, '∫');
        assert_eq!(symbols::FORALL, '∀');
        assert_eq!(symbols::ALPHA, 'α');
        assert_eq!(symbols::PI, 'π');
    }

    #[test]
    fn test_comprehensive_superscript_mapping() {
        // Test all digits
        for c in "0123456789".chars() {
            let result = unicode_superscript(&c.to_string());
            assert_ne!(result, c.to_string(), "Digit {} should have superscript", c);
            assert_eq!(result.chars().count(), 1, "Should produce single character");
        }

        // Test common letters
        for c in "abcdefghijklmnoprstuvwxyz".chars() {
            let result = unicode_superscript(&c.to_string());
            // All these letters should have superscripts
            assert_ne!(result, c.to_string(), "Letter {} should have superscript", c);
        }
    }

    #[test]
    fn test_comprehensive_subscript_mapping() {
        // Test all digits
        for c in "0123456789".chars() {
            let result = unicode_subscript(&c.to_string());
            assert_ne!(result, c.to_string(), "Digit {} should have subscript", c);
            assert_eq!(result.chars().count(), 1, "Should produce single character");
        }

        // Test available letters
        for c in "aeoxhklmnpst".chars() {
            let result = unicode_subscript(&c.to_string());
            assert_ne!(result, c.to_string(), "Letter {} should have subscript", c);
        }
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(unicode_superscript(""), "");
        assert_eq!(unicode_subscript(""), "");
    }

    #[test]
    fn test_greek_superscripts() {
        assert_eq!(unicode_superscript("α"), "ᵅ");
        assert_eq!(unicode_superscript("β"), "ᵝ");
        assert_eq!(unicode_superscript("γ"), "ᵞ");
        assert_eq!(unicode_superscript("δ"), "ᵟ");
        assert_eq!(unicode_superscript("φ"), "ᵠ");
    }

    #[test]
    fn test_greek_subscripts() {
        assert_eq!(unicode_subscript("β"), "ᵦ");
        assert_eq!(unicode_subscript("γ"), "ᵧ");
        assert_eq!(unicode_subscript("ρ"), "ᵨ");
        assert_eq!(unicode_subscript("φ"), "ᵩ");
        assert_eq!(unicode_subscript("χ"), "ᵪ");
    }

    #[test]
    fn test_mathematical_expressions() {
        // x^(n+1)
        let expr = format!("x{}", unicode_superscript("n+1"));
        assert_eq!(expr, "xⁿ⁺¹");

        // a_0
        let expr = format!("a{}", unicode_subscript("0"));
        assert_eq!(expr, "a₀");

        // x^2 + y^2
        let expr = format!("x{} + y{}", unicode_superscript("2"), unicode_superscript("2"));
        assert_eq!(expr, "x² + y²");
    }
}
