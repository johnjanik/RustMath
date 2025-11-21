//! Text primitive for plotting
//!
//! Based on SageMath's sage.plot.text module

use rustmath_plot_core::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
    TextOptions,
};

/// Text label to display on a plot
///
/// Based on SageMath's Text class from sage.plot.text
pub struct Text {
    /// The text to display
    text: String,

    /// Position of the text
    position: Point2D,

    /// Text-specific options
    text_options: TextOptions,

    /// Plot options (for compatibility with GraphicPrimitive)
    plot_options: PlotOptions,
}

impl Text {
    /// Create a new Text primitive
    ///
    /// # Arguments
    /// * `text` - The text to display
    /// * `position` - The position where the text will be displayed
    /// * `text_options` - Text-specific styling options
    ///
    /// # Examples
    /// ```ignore
    /// let text = Text::new("Hello", (1.0, 2.0), TextOptions::default(), PlotOptions::default());
    /// ```
    pub fn new(
        text: impl Into<String>,
        position: impl Into<Point2D>,
        text_options: TextOptions,
        plot_options: PlotOptions,
    ) -> Self {
        Self {
            text: text.into(),
            position: position.into(),
            text_options,
            plot_options,
        }
    }

    /// Create text with default options
    pub fn simple(text: impl Into<String>, position: impl Into<Point2D>) -> Self {
        Self {
            text: text.into(),
            position: position.into(),
            text_options: TextOptions::default(),
            plot_options: PlotOptions::default(),
        }
    }

    /// Get the text content
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get the position
    pub fn position(&self) -> Point2D {
        self.position
    }

    /// Get the text options
    pub fn text_options(&self) -> &TextOptions {
        &self.text_options
    }

    /// Get a mutable reference to the text options
    pub fn text_options_mut(&mut self) -> &mut TextOptions {
        &mut self.text_options
    }

    /// Set the font size
    pub fn set_font_size(&mut self, size: f64) {
        self.text_options.font_size = size;
    }

    /// Set the rotation angle (in degrees)
    pub fn set_rotation(&mut self, angle: f64) {
        self.text_options.rotation = angle;
    }
}

impl Renderable for Text {
    fn bounding_box(&self) -> BoundingBox {
        // For text, we'll create a small bounding box around the position
        // A proper implementation would calculate the actual text bounds
        // based on font size and string length
        let margin = self.text_options.font_size / 10.0;
        BoundingBox::new(
            self.position.x - margin,
            self.position.x + margin,
            self.position.y - margin,
            self.position.y + margin,
        )
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        backend.draw_text(self.position, &self.text, &self.text_options)
    }
}

impl GraphicPrimitive for Text {
    fn options(&self) -> &PlotOptions {
        &self.plot_options
    }

    fn options_mut(&mut self) -> &mut PlotOptions {
        &mut self.plot_options
    }

    fn set_options(&mut self, options: PlotOptions) {
        self.plot_options = options;
    }
}

/// Factory function to create a Text primitive
///
/// # Arguments
/// * `text` - The text to display
/// * `position` - The position where the text will be displayed
/// * `text_options` - Optional text styling options
///
/// # Examples
/// ```ignore
/// use rustmath_plot::primitives::text;
///
/// let t1 = text("Hello", (1.0, 2.0), None);
/// let t2 = text("World", (2.0, 3.0), Some(TextOptions::default()));
/// ```
pub fn text(
    text: impl Into<String>,
    position: impl Into<Point2D>,
    text_options: Option<TextOptions>,
) -> Box<Text> {
    Box::new(Text::new(
        text,
        position,
        text_options.unwrap_or_default(),
        PlotOptions::default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_creation() {
        let t = Text::new("Hello", (1.0, 2.0), TextOptions::default(), PlotOptions::default());
        assert_eq!(t.text(), "Hello");
        assert_eq!(t.position(), Point2D::new(1.0, 2.0));
    }

    #[test]
    fn test_text_simple() {
        let t = Text::simple("World", (3.0, 4.0));
        assert_eq!(t.text(), "World");
        assert_eq!(t.position(), Point2D::new(3.0, 4.0));
    }

    #[test]
    fn test_text_font_size() {
        let mut t = Text::simple("Test", (0.0, 0.0));
        t.set_font_size(20.0);
        assert_eq!(t.text_options().font_size, 20.0);
    }

    #[test]
    fn test_text_rotation() {
        let mut t = Text::simple("Rotated", (0.0, 0.0));
        t.set_rotation(45.0);
        assert_eq!(t.text_options().rotation, 45.0);
    }

    #[test]
    fn test_text_bounding_box() {
        let t = Text::simple("Test", (0.0, 0.0));
        let bbox = t.bounding_box();
        // Bounding box should contain the position
        assert!(bbox.contains(&Point2D::new(0.0, 0.0)));
    }

    #[test]
    fn test_text_factory() {
        let t = text("Factory", (1.0, 1.0), None);
        assert_eq!(t.text(), "Factory");
    }
}
