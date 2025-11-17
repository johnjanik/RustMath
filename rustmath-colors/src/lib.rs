//! Color system for RustMath plotting
//!
//! This crate provides a comprehensive color handling system including:
//! - RGB, HSL, HSV color spaces with conversions
//! - Color parsing from various formats (hex, named colors, tuples)
//! - Colormaps and gradients for scientific visualization
//! - Rainbow and hue-based color generation
//!
//! Based on SageMath's sage.plot.colors module.

mod color;
mod colormap;
mod conversions;
mod named_colors;
mod parser;

pub use color::{Color, ColorSpace};
pub use colormap::{Colormap, gradient, rainbow, hue};
pub use conversions::{rgb_to_hsl, hsl_to_rgb, rgb_to_hsv, hsv_to_rgb};
pub use parser::parse_color;

/// Errors that can occur during color operations
#[derive(Debug, thiserror::Error)]
pub enum ColorError {
    #[error("Invalid color value: {0}")]
    InvalidValue(String),

    #[error("Invalid hex color format: {0}")]
    InvalidHex(String),

    #[error("Unknown color name: {0}")]
    UnknownColorName(String),

    #[error("Invalid color format: {0}")]
    InvalidFormat(String),
}

pub type Result<T> = std::result::Result<T, ColorError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_color_creation() {
        let red = Color::rgb(1.0, 0.0, 0.0);
        assert_eq!(red.red(), 1.0);
        assert_eq!(red.green(), 0.0);
        assert_eq!(red.blue(), 0.0);
        assert_eq!(red.alpha(), 1.0);
    }

    #[test]
    fn test_color_with_alpha() {
        let semi_transparent = Color::rgba(0.5, 0.5, 0.5, 0.5);
        assert_eq!(semi_transparent.alpha(), 0.5);
    }

    #[test]
    fn test_rgb_to_hsl_conversion() {
        let (h, s, l) = rgb_to_hsl(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1e-10);
        assert!((s - 1.0).abs() < 1e-10);
        assert!((l - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hsl_to_rgb_conversion() {
        let (r, g, b) = hsl_to_rgb(0.0, 1.0, 0.5);
        assert!((r - 1.0).abs() < 1e-10);
        assert!((g - 0.0).abs() < 1e-10);
        assert!((b - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_rgb_hsl() {
        let (r1, g1, b1) = (0.3, 0.7, 0.9);
        let (h, s, l) = rgb_to_hsl(r1, g1, b1);
        let (r2, g2, b2) = hsl_to_rgb(h, s, l);
        assert!((r1 - r2).abs() < 1e-10);
        assert!((g1 - g2).abs() < 1e-10);
        assert!((b1 - b2).abs() < 1e-10);
    }
}
