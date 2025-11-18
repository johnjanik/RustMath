//! Colormap and gradient system for scientific visualization
//!
//! Based on SageMath's colormap functionality from sage.plot.colors

use crate::color::Color;
use crate::conversions::mod_one;

/// Represents a colormap (gradient) for mapping values to colors
#[derive(Debug, Clone)]
pub struct Colormap {
    name: String,
    colors: Vec<Color>,
}

impl Colormap {
    /// Create a new colormap from a list of colors
    pub fn new(name: impl Into<String>, colors: Vec<Color>) -> Self {
        assert!(!colors.is_empty(), "Colormap must have at least one color");
        Self {
            name: name.into(),
            colors,
        }
    }

    /// Get the name of the colormap
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get a color from the colormap at position t (0.0 to 1.0)
    ///
    /// Interpolates between colors in the colormap
    pub fn get_color(&self, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);

        if self.colors.len() == 1 {
            return self.colors[0].clone();
        }

        let scaled = t * (self.colors.len() - 1) as f64;
        let index = scaled.floor() as usize;
        let fraction = scaled - index as f64;

        if index >= self.colors.len() - 1 {
            return self.colors[self.colors.len() - 1].clone();
        }

        self.colors[index].blend(&self.colors[index + 1], fraction)
    }

    /// Get a sample of n colors evenly distributed across the colormap
    pub fn sample(&self, n: usize) -> Vec<Color> {
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![self.get_color(0.5)];
        }

        (0..n)
            .map(|i| self.get_color(i as f64 / (n - 1) as f64))
            .collect()
    }

    /// Create a reversed version of the colormap
    pub fn reversed(&self) -> Self {
        let mut colors = self.colors.clone();
        colors.reverse();
        Self {
            name: format!("{}_r", self.name),
            colors,
        }
    }

    /// Predefined colormap: Rainbow (red -> orange -> yellow -> green -> blue -> violet)
    pub fn rainbow() -> Self {
        Self::new(
            "rainbow",
            (0..256)
                .map(|i| {
                    let t = i as f64 / 255.0;
                    hue(t)
                })
                .collect(),
        )
    }

    /// Predefined colormap: Viridis (perceptually uniform)
    pub fn viridis() -> Self {
        Self::new(
            "viridis",
            vec![
                Color::rgb(0.267004, 0.004874, 0.329415),
                Color::rgb(0.282623, 0.140926, 0.457517),
                Color::rgb(0.253935, 0.265254, 0.529983),
                Color::rgb(0.206756, 0.371758, 0.553117),
                Color::rgb(0.163625, 0.471133, 0.558148),
                Color::rgb(0.127568, 0.566949, 0.550556),
                Color::rgb(0.134692, 0.658636, 0.517649),
                Color::rgb(0.266941, 0.748751, 0.440573),
                Color::rgb(0.477504, 0.821444, 0.318195),
                Color::rgb(0.741388, 0.873449, 0.149561),
                Color::rgb(0.993248, 0.906157, 0.143936),
            ],
        )
    }

    /// Predefined colormap: Plasma
    pub fn plasma() -> Self {
        Self::new(
            "plasma",
            vec![
                Color::rgb(0.050383, 0.029803, 0.527975),
                Color::rgb(0.282623, 0.140926, 0.457517),
                Color::rgb(0.477504, 0.821444, 0.318195),
                Color::rgb(0.991541, 0.927856, 0.073130),
            ],
        )
    }

    /// Predefined colormap: Hot (black -> red -> yellow -> white)
    pub fn hot() -> Self {
        Self::new(
            "hot",
            vec![
                Color::rgb(0.0, 0.0, 0.0),
                Color::rgb(1.0, 0.0, 0.0),
                Color::rgb(1.0, 1.0, 0.0),
                Color::rgb(1.0, 1.0, 1.0),
            ],
        )
    }

    /// Predefined colormap: Cool (cyan -> magenta)
    pub fn cool() -> Self {
        Self::new(
            "cool",
            vec![Color::rgb(0.0, 1.0, 1.0), Color::rgb(1.0, 0.0, 1.0)],
        )
    }

    /// Predefined colormap: Gray (black -> white)
    pub fn gray() -> Self {
        Self::new(
            "gray",
            vec![Color::rgb(0.0, 0.0, 0.0), Color::rgb(1.0, 1.0, 1.0)],
        )
    }

    /// Predefined colormap: Jet (blue -> cyan -> yellow -> red)
    pub fn jet() -> Self {
        Self::new(
            "jet",
            vec![
                Color::rgb(0.0, 0.0, 0.5),
                Color::rgb(0.0, 0.0, 1.0),
                Color::rgb(0.0, 1.0, 1.0),
                Color::rgb(1.0, 1.0, 0.0),
                Color::rgb(1.0, 0.0, 0.0),
                Color::rgb(0.5, 0.0, 0.0),
            ],
        )
    }
}

/// Create a color from a hue value
///
/// # Arguments
/// * `h` - Hue value (0.0 to 1.0, wraps around)
///
/// # Returns
/// A fully saturated color at the given hue
///
/// Based on sage.plot.colors.hue
pub fn hue(h: f64) -> Color {
    Color::hsl(mod_one(h), 1.0, 0.5)
}

/// Generate a rainbow of n colors
///
/// # Arguments
/// * `n` - Number of colors to generate
///
/// # Returns
/// Vector of n evenly-spaced colors around the color wheel
///
/// Based on sage.plot.colors.rainbow
pub fn rainbow(n: usize) -> Vec<Color> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![hue(0.0)];
    }

    (0..n).map(|i| hue(i as f64 / n as f64)).collect()
}

/// Create a gradient between two colors
///
/// # Arguments
/// * `start` - Starting color
/// * `end` - Ending color
/// * `n` - Number of colors in the gradient
///
/// # Returns
/// Vector of n colors interpolated between start and end
pub fn gradient(start: &Color, end: &Color, n: usize) -> Vec<Color> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start.clone()];
    }

    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            start.blend(end, t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hue() {
        let red = hue(0.0);
        assert!((red.red() - 1.0).abs() < 1e-10);
        assert!((red.green() - 0.0).abs() < 1e-10);
        assert!((red.blue() - 0.0).abs() < 1e-10);

        let green = hue(1.0 / 3.0);
        assert!((green.red() - 0.0).abs() < 1e-10);
        assert!((green.green() - 1.0).abs() < 1e-10);
        assert!((green.blue() - 0.0).abs() < 1e-10);

        let blue = hue(2.0 / 3.0);
        assert!((blue.red() - 0.0).abs() < 1e-10);
        assert!((blue.green() - 0.0).abs() < 1e-10);
        assert!((blue.blue() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rainbow() {
        let colors = rainbow(6);
        assert_eq!(colors.len(), 6);

        // First should be red
        assert!((colors[0].red() - 1.0).abs() < 1e-10);

        // Should cycle through the color wheel
        for i in 0..6 {
            let expected_hue = i as f64 / 6.0;
            let (h, _, _) = colors[i].to_hsl();
            assert!((h - expected_hue).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gradient() {
        let black = Color::black();
        let white = Color::white();
        let grad = gradient(&black, &white, 5);

        assert_eq!(grad.len(), 5);
        assert_eq!(grad[0], black);
        assert_eq!(grad[4], white);

        // Middle should be gray
        let middle = &grad[2];
        assert!((middle.red() - 0.5).abs() < 1e-10);
        assert!((middle.green() - 0.5).abs() < 1e-10);
        assert!((middle.blue() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_colormap_creation() {
        let colors = vec![Color::black(), Color::white()];
        let cmap = Colormap::new("test", colors);
        assert_eq!(cmap.name(), "test");
    }

    #[test]
    fn test_colormap_get_color() {
        let cmap = Colormap::new("bw", vec![Color::black(), Color::white()]);

        let c0 = cmap.get_color(0.0);
        assert_eq!(c0, Color::black());

        let c1 = cmap.get_color(1.0);
        assert_eq!(c1, Color::white());

        let c_mid = cmap.get_color(0.5);
        assert!((c_mid.red() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_colormap_sample() {
        let cmap = Colormap::new("bw", vec![Color::black(), Color::white()]);
        let samples = cmap.sample(5);

        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0], Color::black());
        assert_eq!(samples[4], Color::white());
    }

    #[test]
    fn test_colormap_reversed() {
        let cmap = Colormap::new("bw", vec![Color::black(), Color::white()]);
        let reversed = cmap.reversed();

        assert_eq!(reversed.name(), "bw_r");
        assert_eq!(reversed.get_color(0.0), Color::white());
        assert_eq!(reversed.get_color(1.0), Color::black());
    }

    #[test]
    fn test_predefined_colormaps() {
        let rainbow = Colormap::rainbow();
        assert_eq!(rainbow.name(), "rainbow");
        assert_eq!(rainbow.colors.len(), 256);

        let viridis = Colormap::viridis();
        assert_eq!(viridis.name(), "viridis");
        assert!(viridis.colors.len() > 0);

        let hot = Colormap::hot();
        assert_eq!(hot.name(), "hot");

        let gray = Colormap::gray();
        assert_eq!(gray.name(), "gray");
    }
}
