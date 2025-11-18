//! Core color representation


/// Represents a color space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// Red, Green, Blue color space
    RGB,
    /// Hue, Saturation, Lightness color space
    HSL,
    /// Hue, Saturation, Value color space
    HSV,
}

/// Represents a color with RGBA components
///
/// All components are stored as f64 values in the range [0.0, 1.0].
/// Based on SageMath's Color class from sage.plot.colors
#[derive(Debug, Clone, PartialEq)]
pub struct Color {
    r: f64,
    g: f64,
    b: f64,
    alpha: f64,
}

impl Color {
    /// Create a new color from RGB components
    ///
    /// # Arguments
    /// * `r` - Red component (0.0 to 1.0)
    /// * `g` - Green component (0.0 to 1.0)
    /// * `b` - Blue component (0.0 to 1.0)
    ///
    /// # Panics
    /// Panics if any component is outside [0.0, 1.0]
    pub fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self::rgba(r, g, b, 1.0)
    }

    /// Create a new color from RGBA components
    ///
    /// # Arguments
    /// * `r` - Red component (0.0 to 1.0)
    /// * `g` - Green component (0.0 to 1.0)
    /// * `b` - Blue component (0.0 to 1.0)
    /// * `alpha` - Alpha/opacity component (0.0 to 1.0)
    ///
    /// # Panics
    /// Panics if any component is outside [0.0, 1.0]
    pub fn rgba(r: f64, g: f64, b: f64, alpha: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&r),
            "Red component must be in [0.0, 1.0], got {}",
            r
        );
        assert!(
            (0.0..=1.0).contains(&g),
            "Green component must be in [0.0, 1.0], got {}",
            g
        );
        assert!(
            (0.0..=1.0).contains(&b),
            "Blue component must be in [0.0, 1.0], got {}",
            b
        );
        assert!(
            (0.0..=1.0).contains(&alpha),
            "Alpha component must be in [0.0, 1.0], got {}",
            alpha
        );

        Self { r, g, b, alpha }
    }

    /// Create a color from HSL components
    ///
    /// # Arguments
    /// * `h` - Hue (0.0 to 1.0, wraps around)
    /// * `s` - Saturation (0.0 to 1.0)
    /// * `l` - Lightness (0.0 to 1.0)
    pub fn hsl(h: f64, s: f64, l: f64) -> Self {
        Self::hsla(h, s, l, 1.0)
    }

    /// Create a color from HSLA components
    ///
    /// # Arguments
    /// * `h` - Hue (0.0 to 1.0, wraps around)
    /// * `s` - Saturation (0.0 to 1.0)
    /// * `l` - Lightness (0.0 to 1.0)
    /// * `alpha` - Alpha/opacity (0.0 to 1.0)
    pub fn hsla(h: f64, s: f64, l: f64, alpha: f64) -> Self {
        let (r, g, b) = crate::conversions::hsl_to_rgb(h, s, l);
        Self::rgba(r, g, b, alpha)
    }

    /// Create a color from HSV components
    ///
    /// # Arguments
    /// * `h` - Hue (0.0 to 1.0, wraps around)
    /// * `s` - Saturation (0.0 to 1.0)
    /// * `v` - Value/brightness (0.0 to 1.0)
    pub fn hsv(h: f64, s: f64, v: f64) -> Self {
        Self::hsva(h, s, v, 1.0)
    }

    /// Create a color from HSVA components
    ///
    /// # Arguments
    /// * `h` - Hue (0.0 to 1.0, wraps around)
    /// * `s` - Saturation (0.0 to 1.0)
    /// * `v` - Value/brightness (0.0 to 1.0)
    /// * `alpha` - Alpha/opacity (0.0 to 1.0)
    pub fn hsva(h: f64, s: f64, v: f64, alpha: f64) -> Self {
        let (r, g, b) = crate::conversions::hsv_to_rgb(h, s, v);
        Self::rgba(r, g, b, alpha)
    }

    /// Get the red component
    pub fn red(&self) -> f64 {
        self.r
    }

    /// Get the green component
    pub fn green(&self) -> f64 {
        self.g
    }

    /// Get the blue component
    pub fn blue(&self) -> f64 {
        self.b
    }

    /// Get the alpha component
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get RGB components as a tuple
    pub fn rgb_tuple(&self) -> (f64, f64, f64) {
        (self.r, self.g, self.b)
    }

    /// Get RGBA components as a tuple
    pub fn rgba_tuple(&self) -> (f64, f64, f64, f64) {
        (self.r, self.g, self.b, self.alpha)
    }

    /// Convert to HSL color space
    ///
    /// Returns (hue, saturation, lightness) where all values are in [0.0, 1.0]
    pub fn to_hsl(&self) -> (f64, f64, f64) {
        crate::conversions::rgb_to_hsl(self.r, self.g, self.b)
    }

    /// Convert to HSV color space
    ///
    /// Returns (hue, saturation, value) where all values are in [0.0, 1.0]
    pub fn to_hsv(&self) -> (f64, f64, f64) {
        crate::conversions::rgb_to_hsv(self.r, self.g, self.b)
    }

    /// Convert to hex string (e.g., "#FF0000" for red)
    pub fn to_hex(&self) -> String {
        format!(
            "#{:02X}{:02X}{:02X}",
            (self.r * 255.0).round() as u8,
            (self.g * 255.0).round() as u8,
            (self.b * 255.0).round() as u8
        )
    }

    /// Convert to hex string with alpha (e.g., "#FF000080" for semi-transparent red)
    pub fn to_hex_with_alpha(&self) -> String {
        format!(
            "#{:02X}{:02X}{:02X}{:02X}",
            (self.r * 255.0).round() as u8,
            (self.g * 255.0).round() as u8,
            (self.b * 255.0).round() as u8,
            (self.alpha * 255.0).round() as u8
        )
    }

    /// Set the alpha channel
    pub fn with_alpha(&self, alpha: f64) -> Self {
        Self::rgba(self.r, self.g, self.b, alpha)
    }

    /// Lighten the color by a factor
    ///
    /// # Arguments
    /// * `factor` - Amount to lighten (0.0 = no change, 1.0 = white)
    pub fn lighten(&self, factor: f64) -> Self {
        let (h, s, l) = self.to_hsl();
        let new_l = l + (1.0 - l) * factor.clamp(0.0, 1.0);
        Self::hsla(h, s, new_l, self.alpha)
    }

    /// Darken the color by a factor
    ///
    /// # Arguments
    /// * `factor` - Amount to darken (0.0 = no change, 1.0 = black)
    pub fn darken(&self, factor: f64) -> Self {
        let (h, s, l) = self.to_hsl();
        let new_l = l * (1.0 - factor.clamp(0.0, 1.0));
        Self::hsla(h, s, new_l, self.alpha)
    }

    /// Blend two colors together
    ///
    /// # Arguments
    /// * `other` - The other color to blend with
    /// * `t` - Blend factor (0.0 = self, 1.0 = other)
    pub fn blend(&self, other: &Color, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        let r = self.r * (1.0 - t) + other.r * t;
        let g = self.g * (1.0 - t) + other.g * t;
        let b = self.b * (1.0 - t) + other.b * t;
        let alpha = self.alpha * (1.0 - t) + other.alpha * t;
        Self::rgba(r, g, b, alpha)
    }

    /// Common color: Black
    pub fn black() -> Self {
        Self::rgb(0.0, 0.0, 0.0)
    }

    /// Common color: White
    pub fn white() -> Self {
        Self::rgb(1.0, 1.0, 1.0)
    }

    /// Common color: Red
    pub fn red_color() -> Self {
        Self::rgb(1.0, 0.0, 0.0)
    }

    /// Common color: Green
    pub fn green_color() -> Self {
        Self::rgb(0.0, 1.0, 0.0)
    }

    /// Common color: Blue
    pub fn blue_color() -> Self {
        Self::rgb(0.0, 0.0, 1.0)
    }

    /// Common color: Yellow
    pub fn yellow_color() -> Self {
        Self::rgb(1.0, 1.0, 0.0)
    }

    /// Common color: Cyan
    pub fn cyan_color() -> Self {
        Self::rgb(0.0, 1.0, 1.0)
    }

    /// Common color: Magenta
    pub fn magenta_color() -> Self {
        Self::rgb(1.0, 0.0, 1.0)
    }

    /// Common color: Gray (50%)
    pub fn gray_color() -> Self {
        Self::rgb(0.5, 0.5, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_creation() {
        let c = Color::rgb(0.5, 0.6, 0.7);
        assert_eq!(c.red(), 0.5);
        assert_eq!(c.green(), 0.6);
        assert_eq!(c.blue(), 0.7);
        assert_eq!(c.alpha(), 1.0);
    }

    #[test]
    fn test_hsl_creation() {
        let c = Color::hsl(0.0, 1.0, 0.5);
        // Should be red
        assert!((c.red() - 1.0).abs() < 1e-10);
        assert!((c.green() - 0.0).abs() < 1e-10);
        assert!((c.blue() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hex_conversion() {
        let c = Color::rgb(1.0, 0.0, 0.0);
        assert_eq!(c.to_hex(), "#FF0000");

        let c2 = Color::rgb(0.0, 0.5, 1.0);
        assert_eq!(c2.to_hex(), "#0080FF");
    }

    #[test]
    fn test_blend() {
        let c1 = Color::black();
        let c2 = Color::white();
        let blended = c1.blend(&c2, 0.5);
        assert!((blended.red() - 0.5).abs() < 1e-10);
        assert!((blended.green() - 0.5).abs() < 1e-10);
        assert!((blended.blue() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_lighten_darken() {
        let c = Color::rgb(0.5, 0.5, 0.5);
        let lighter = c.lighten(0.5);
        let darker = c.darken(0.5);

        // Lighter should have higher lightness
        let (_, _, l1) = c.to_hsl();
        let (_, _, l2) = lighter.to_hsl();
        let (_, _, l3) = darker.to_hsl();

        assert!(l2 > l1);
        assert!(l3 < l1);
    }

    #[test]
    #[should_panic]
    fn test_invalid_rgb() {
        Color::rgb(1.5, 0.0, 0.0); // Should panic
    }
}
