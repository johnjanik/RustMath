//! Color parsing from various string and numeric formats

use crate::color::Color;
use crate::named_colors::get_named_color;
use crate::{ColorError, Result};

/// Parse a color from a string
///
/// Supports multiple formats:
/// - Named colors: "red", "blue", "orange", etc.
/// - Hex colors: "#FF0000", "#F00", "#FF0000FF" (with alpha)
/// - RGB tuples: Will need to use parse_color_tuple for tuples
///
/// # Examples
/// ```
/// use rustmath_colors::parse_color;
///
/// let red = parse_color("red").unwrap();
/// let blue = parse_color("#0000FF").unwrap();
/// let green = parse_color("#0F0").unwrap();
/// ```
pub fn parse_color(s: &str) -> Result<Color> {
    let s = s.trim();

    // Try hex format first
    if s.starts_with('#') {
        return parse_hex_color(s);
    }

    // Try named color
    if let Some(color) = get_named_color(s) {
        return Ok(color);
    }

    Err(ColorError::InvalidFormat(format!(
        "Could not parse color from: {}",
        s
    )))
}

/// Parse a hex color string
///
/// Supports formats:
/// - #RGB (e.g., #F00 for red)
/// - #RRGGBB (e.g., #FF0000 for red)
/// - #RRGGBBAA (e.g., #FF0000FF for opaque red)
fn parse_hex_color(s: &str) -> Result<Color> {
    let s = s.trim();

    if !s.starts_with('#') {
        return Err(ColorError::InvalidHex(format!(
            "Hex color must start with #: {}",
            s
        )));
    }

    let hex = &s[1..];
    let len = hex.len();

    match len {
        3 => {
            // #RGB format
            let r = parse_hex_digit(hex.chars().nth(0).unwrap())?;
            let g = parse_hex_digit(hex.chars().nth(1).unwrap())?;
            let b = parse_hex_digit(hex.chars().nth(2).unwrap())?;
            Ok(Color::rgb(r, g, b))
        }
        6 => {
            // #RRGGBB format
            let r = parse_hex_byte(&hex[0..2])?;
            let g = parse_hex_byte(&hex[2..4])?;
            let b = parse_hex_byte(&hex[4..6])?;
            Ok(Color::rgb(r, g, b))
        }
        8 => {
            // #RRGGBBAA format
            let r = parse_hex_byte(&hex[0..2])?;
            let g = parse_hex_byte(&hex[2..4])?;
            let b = parse_hex_byte(&hex[4..6])?;
            let a = parse_hex_byte(&hex[6..8])?;
            Ok(Color::rgba(r, g, b, a))
        }
        _ => Err(ColorError::InvalidHex(format!(
            "Hex color must be 3, 6, or 8 characters (got {}): {}",
            len, s
        ))),
    }
}

/// Parse a single hex digit (0-F) to a float (0.0-1.0)
fn parse_hex_digit(c: char) -> Result<f64> {
    match c.to_ascii_uppercase() {
        '0' => Ok(0.0 / 15.0),
        '1' => Ok(1.0 / 15.0),
        '2' => Ok(2.0 / 15.0),
        '3' => Ok(3.0 / 15.0),
        '4' => Ok(4.0 / 15.0),
        '5' => Ok(5.0 / 15.0),
        '6' => Ok(6.0 / 15.0),
        '7' => Ok(7.0 / 15.0),
        '8' => Ok(8.0 / 15.0),
        '9' => Ok(9.0 / 15.0),
        'A' => Ok(10.0 / 15.0),
        'B' => Ok(11.0 / 15.0),
        'C' => Ok(12.0 / 15.0),
        'D' => Ok(13.0 / 15.0),
        'E' => Ok(14.0 / 15.0),
        'F' => Ok(15.0 / 15.0),
        _ => Err(ColorError::InvalidHex(format!("Invalid hex digit: {}", c))),
    }
}

/// Parse a two-character hex byte (00-FF) to a float (0.0-1.0)
fn parse_hex_byte(s: &str) -> Result<f64> {
    if s.len() != 2 {
        return Err(ColorError::InvalidHex(format!(
            "Hex byte must be 2 characters: {}",
            s
        )));
    }

    let byte = u8::from_str_radix(s, 16).map_err(|_| {
        ColorError::InvalidHex(format!("Could not parse hex byte: {}", s))
    })?;

    Ok(byte as f64 / 255.0)
}

/// Parse a color from RGB tuple (r, g, b) where values can be 0-255 or 0.0-1.0
pub fn parse_color_from_rgb(r: f64, g: f64, b: f64) -> Result<Color> {
    parse_color_from_rgba(r, g, b, 1.0)
}

/// Parse a color from RGBA tuple (r, g, b, a) where values can be 0-255 or 0.0-1.0
pub fn parse_color_from_rgba(r: f64, g: f64, b: f64, a: f64) -> Result<Color> {
    // First check that all values are non-negative
    if r < 0.0 || g < 0.0 || b < 0.0 || a < 0.0 {
        return Err(ColorError::InvalidValue(
            "Color values cannot be negative".to_string(),
        ));
    }

    // Check for ambiguous values in the (1.0, 2.0) range
    // These are unclear: too large for normalized [0,1], too small for byte [0,255]
    let has_ambiguous = [r, g, b, a].iter().any(|&v| v > 1.0 && v < 2.0);
    if has_ambiguous {
        return Err(ColorError::InvalidValue(format!(
            "Ambiguous color values: values in range (1.0, 2.0) are unclear. Use [0.0, 1.0] or [0, 255]. Got r={}, g={}, b={}, a={}",
            r, g, b, a
        )));
    }

    // Determine if values are in 0-255 range or 0.0-1.0 range
    // If ANY value is >= 2.0, we treat as byte range
    let is_byte_range = r >= 2.0 || g >= 2.0 || b >= 2.0 || a >= 2.0;

    if is_byte_range {
        // In byte range mode, ALL values must be in [0, 255]
        if r > 255.0 || g > 255.0 || b > 255.0 || a > 255.0 {
            return Err(ColorError::InvalidValue(format!(
                "Color values must be in range [0, 255], got r={}, g={}, b={}, a={}",
                r, g, b, a
            )));
        }
        Ok(Color::rgba(r / 255.0, g / 255.0, b / 255.0, a / 255.0))
    } else {
        // In normalized range mode, ALL values must be in [0.0, 1.0]
        if r > 1.0 || g > 1.0 || b > 1.0 || a > 1.0 {
            return Err(ColorError::InvalidValue(format!(
                "Color values must be in range [0.0, 1.0], got r={}, g={}, b={}, a={}",
                r, g, b, a
            )));
        }
        Ok(Color::rgba(r, g, b, a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_named_color() {
        let red = parse_color("red").unwrap();
        assert_eq!(red, Color::red_color());

        let blue = parse_color("blue").unwrap();
        assert_eq!(blue, Color::blue_color());

        let white = parse_color("  white  ").unwrap();
        assert_eq!(white, Color::white());
    }

    #[test]
    fn test_parse_hex_3_digit() {
        let red = parse_color("#F00").unwrap();
        assert!((red.red() - 1.0).abs() < 1e-10);
        assert!((red.green() - 0.0).abs() < 1e-10);
        assert!((red.blue() - 0.0).abs() < 1e-10);

        let cyan = parse_color("#0FF").unwrap();
        assert!((cyan.red() - 0.0).abs() < 1e-10);
        assert!((cyan.green() - 1.0).abs() < 1e-10);
        assert!((cyan.blue() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_hex_6_digit() {
        let red = parse_color("#FF0000").unwrap();
        assert_eq!(red, Color::red_color());

        let green = parse_color("#00FF00").unwrap();
        assert_eq!(green, Color::green_color());

        let blue = parse_color("#0000FF").unwrap();
        assert_eq!(blue, Color::blue_color());
    }

    #[test]
    fn test_parse_hex_8_digit() {
        let semi_red = parse_color("#FF000080").unwrap();
        assert!((semi_red.red() - 1.0).abs() < 1e-10);
        assert!((semi_red.green() - 0.0).abs() < 1e-10);
        assert!((semi_red.blue() - 0.0).abs() < 1e-10);
        assert!((semi_red.alpha() - 128.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_hex_lowercase() {
        let red = parse_color("#ff0000").unwrap();
        assert_eq!(red, Color::red_color());
    }

    #[test]
    fn test_parse_invalid_hex() {
        assert!(parse_color("#GG0000").is_err());
        assert!(parse_color("#F").is_err());
        assert!(parse_color("#FF").is_err());
        assert!(parse_color("#FFFF").is_err());
        assert!(parse_color("#FFFFF").is_err());
        assert!(parse_color("FF0000").is_err()); // Missing #
    }

    #[test]
    fn test_parse_unknown() {
        assert!(parse_color("notacolor").is_err());
    }

    #[test]
    fn test_parse_rgb_float() {
        let red = parse_color_from_rgb(1.0, 0.0, 0.0).unwrap();
        assert_eq!(red, Color::red_color());

        let gray = parse_color_from_rgb(0.5, 0.5, 0.5).unwrap();
        assert_eq!(gray, Color::gray_color());
    }

    #[test]
    fn test_parse_rgb_byte() {
        let red = parse_color_from_rgb(255.0, 0.0, 0.0).unwrap();
        assert!((red.red() - 1.0).abs() < 1e-10);

        let gray = parse_color_from_rgb(128.0, 128.0, 128.0).unwrap();
        assert!((gray.red() - 128.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_rgba_with_alpha() {
        let semi_red = parse_color_from_rgba(1.0, 0.0, 0.0, 0.5).unwrap();
        assert_eq!(semi_red.alpha(), 0.5);

        let semi_blue = parse_color_from_rgba(0.0, 0.0, 255.0, 128.0).unwrap();
        assert!((semi_blue.alpha() - 128.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_rgb_invalid() {
        assert!(parse_color_from_rgb(1.5, 0.0, 0.0).is_err());
        assert!(parse_color_from_rgb(-0.1, 0.0, 0.0).is_err());
        assert!(parse_color_from_rgb(256.0, 0.0, 0.0).is_err());
    }
}
