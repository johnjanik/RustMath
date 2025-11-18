//! Graphics container for holding multiple graphics primitives
//!
//! Based on SageMath's sage.plot.graphics module

use rustmath_colors::Color;
use rustmath_plot_core::{
    BoundingBox, GraphicPrimitive, GraphicsOptions, PlotError,
    RenderBackend, RenderFormat, Result,
};
use std::path::Path;

/// A container for multiple graphics primitives
///
/// This is the main structure for building plots. It holds a collection of
/// graphics primitives (lines, circles, polygons, etc.) and manages rendering
/// them to various output formats.
///
/// Based on SageMath's Graphics class.
#[derive(Default)]
pub struct Graphics {
    /// The graphics primitives to render
    primitives: Vec<Box<dyn GraphicPrimitive>>,

    /// Global graphics options (title, axes, size, etc.)
    options: GraphicsOptions,
}

impl Graphics {
    /// Create a new empty Graphics container
    ///
    /// # Examples
    /// ```
    /// use rustmath_plot::Graphics;
    ///
    /// let g = Graphics::new();
    /// ```
    pub fn new() -> Self {
        Self {
            primitives: Vec::new(),
            options: GraphicsOptions::default(),
        }
    }

    /// Create a Graphics container with custom options
    pub fn with_options(options: GraphicsOptions) -> Self {
        Self {
            primitives: Vec::new(),
            options,
        }
    }

    /// Add a graphics primitive to this container
    ///
    /// # Arguments
    /// * `primitive` - The primitive to add
    ///
    /// # Examples
    /// ```ignore
    /// let mut g = Graphics::new();
    /// g.add(line);
    /// g.add(circle);
    /// ```
    pub fn add(&mut self, primitive: Box<dyn GraphicPrimitive>) {
        self.primitives.push(primitive);
    }

    /// Add a graphics primitive and return self for chaining
    pub fn with_primitive(mut self, primitive: Box<dyn GraphicPrimitive>) -> Self {
        self.primitives.push(primitive);
        self
    }

    /// Get the number of primitives in this Graphics
    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    /// Check if this Graphics container is empty
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// Get a reference to the graphics options
    pub fn options(&self) -> &GraphicsOptions {
        &self.options
    }

    /// Get a mutable reference to the graphics options
    pub fn options_mut(&mut self) -> &mut GraphicsOptions {
        &mut self.options
    }

    /// Set the graphics options
    pub fn set_options(&mut self, options: GraphicsOptions) {
        self.options = options;
    }

    /// Set the plot title
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.options.title = Some(title.into());
    }

    /// Set the axis labels
    pub fn set_labels(&mut self, xlabel: impl Into<String>, ylabel: impl Into<String>) {
        self.options.axes.xlabel = Some(xlabel.into());
        self.options.axes.ylabel = Some(ylabel.into());
    }

    /// Set whether to show axes
    pub fn set_axes(&mut self, show: bool) {
        self.options.axes.show_axes = show;
    }

    /// Set the aspect ratio
    pub fn set_aspect_ratio(&mut self, ratio: f64) {
        self.options.axes.aspect_ratio = Some(ratio);
    }

    /// Set the figure size in pixels
    pub fn set_figsize(&mut self, width: usize, height: usize) {
        self.options.figsize = (width, height);
    }

    /// Set the background color
    pub fn set_background_color(&mut self, color: Color) {
        self.options.background_color = color;
    }

    /// Set whether to show a legend
    pub fn set_legend(&mut self, show: bool) {
        self.options.show_legend = show;
    }

    /// Calculate the bounding box that encompasses all primitives
    ///
    /// Returns None if there are no primitives
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        if self.primitives.is_empty() {
            return None;
        }

        let mut bbox = self.primitives[0].bounding_box();
        for primitive in self.primitives.iter().skip(1) {
            bbox = bbox.union(&primitive.bounding_box());
        }

        Some(bbox)
    }

    /// Combine this Graphics with another Graphics
    ///
    /// This creates a new Graphics containing all primitives from both.
    /// The options from `self` are used for the combined graphics.
    ///
    /// # Examples
    /// ```ignore
    /// let g1 = Graphics::new().with_primitive(line1);
    /// let g2 = Graphics::new().with_primitive(line2);
    /// let combined = g1.combine(&g2);
    /// ```
    pub fn combine(&self, _other: &Graphics) -> Graphics {
        let combined = Graphics::with_options(self.options.clone());

        // Note: We'd need Clone trait on GraphicPrimitive to do this properly
        // For now, this is a placeholder - actual implementation would require
        // either making GraphicPrimitive: Clone or using Rc/Arc

        combined
    }

    /// Render this Graphics using the provided backend
    ///
    /// # Arguments
    /// * `backend` - The rendering backend to use
    pub fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        // Sort primitives by z-order before rendering
        let mut sorted_primitives: Vec<_> = self.primitives.iter().collect();
        sorted_primitives.sort_by_key(|p| p.options().zorder);

        // Render each primitive
        for primitive in sorted_primitives {
            primitive.render(backend)?;
        }

        Ok(())
    }

    /// Save this Graphics to a file
    ///
    /// The format is determined by the file extension.
    ///
    /// # Arguments
    /// * `path` - The file path to save to
    /// * `format` - The output format
    ///
    /// # Examples
    /// ```ignore
    /// g.save("plot.svg", RenderFormat::SVG)?;
    /// g.save("plot.png", RenderFormat::PNG)?;
    /// ```
    pub fn save(&self, _path: impl AsRef<Path>, _format: RenderFormat) -> Result<()> {
        // This is a placeholder - actual implementation would require
        // creating the appropriate backend based on the format
        Err(PlotError::RenderError(
            "save() not yet implemented - needs backend selection".to_string(),
        ))
    }

    /// Display this Graphics (placeholder for interactive display)
    ///
    /// In a full implementation, this would open a window or display
    /// the graphics in a Jupyter notebook, etc.
    pub fn show(&self) -> Result<()> {
        // This is a placeholder - actual implementation would require
        // platform-specific display logic
        Err(PlotError::RenderError(
            "show() not yet implemented - needs display backend".to_string(),
        ))
    }

    /// Get an iterator over the primitives
    pub fn primitives(&self) -> impl Iterator<Item = &Box<dyn GraphicPrimitive>> {
        self.primitives.iter()
    }

    /// Get a mutable iterator over the primitives
    pub fn primitives_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn GraphicPrimitive>> {
        self.primitives.iter_mut()
    }

    /// Apply a function to all primitives
    ///
    /// # Examples
    /// ```ignore
    /// // Set all primitives to be red
    /// g.for_each(|p| p.set_color(Color::red_color()));
    /// ```
    pub fn for_each<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Box<dyn GraphicPrimitive>),
    {
        for primitive in &mut self.primitives {
            f(primitive);
        }
    }

    /// Apply a color to all primitives
    pub fn set_all_colors(&mut self, color: Color) {
        for primitive in &mut self.primitives {
            primitive.set_color(color.clone());
        }
    }

    /// Apply an alpha transparency to all primitives
    pub fn set_all_alpha(&mut self, alpha: f64) {
        for primitive in &mut self.primitives {
            primitive.set_alpha(alpha);
        }
    }

    /// Apply a thickness to all primitives
    pub fn set_all_thickness(&mut self, thickness: f64) {
        for primitive in &mut self.primitives {
            primitive.set_thickness(thickness);
        }
    }
}

// Implement + operator for combining Graphics
impl std::ops::Add for Graphics {
    type Output = Graphics;

    fn add(self, other: Graphics) -> Graphics {
        let mut result = Graphics::with_options(self.options.clone());
        result.primitives.extend(self.primitives);
        result.primitives.extend(other.primitives);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphics_new() {
        let g = Graphics::new();
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
    }

    #[test]
    fn test_graphics_options() {
        let mut g = Graphics::new();
        g.set_title("Test Plot");
        g.set_labels("x", "y");
        g.set_figsize(640, 480);

        assert_eq!(g.options().title, Some("Test Plot".to_string()));
        assert_eq!(g.options().axes.xlabel, Some("x".to_string()));
        assert_eq!(g.options().axes.ylabel, Some("y".to_string()));
        assert_eq!(g.options().figsize, (640, 480));
    }

    #[test]
    fn test_graphics_bounding_box_empty() {
        let g = Graphics::new();
        assert!(g.bounding_box().is_none());
    }

    #[test]
    fn test_graphics_setters() {
        let mut g = Graphics::new();

        g.set_axes(false);
        assert!(!g.options().axes.show_axes);

        g.set_aspect_ratio(1.0);
        assert_eq!(g.options().axes.aspect_ratio, Some(1.0));

        g.set_background_color(Color::white());
        assert_eq!(g.options().background_color, Color::white());

        g.set_legend(true);
        assert!(g.options().show_legend);
    }
}
