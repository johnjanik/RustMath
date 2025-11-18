//! Disk primitive for plotting
//!
//! Based on SageMath's sage.plot.disk module

use rustmath_colors::Color;
use rustmath_plot_core::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
};

/// A filled circle (disk)
///
/// Based on SageMath's Disk class from sage.plot.disk
pub struct Disk {
    /// Center point of the disk
    center: Point2D,

    /// Radius of the disk
    radius: f64,

    /// Plot options (fill color, edge color, etc.)
    options: PlotOptions,
}

impl Disk {
    /// Create a new Disk primitive
    ///
    /// # Arguments
    /// * `center` - The center point of the disk
    /// * `radius` - The radius of the disk
    /// * `options` - Plotting options (should have fill enabled)
    ///
    /// # Examples
    /// ```ignore
    /// let mut opts = PlotOptions::default();
    /// opts.fill = true;
    /// opts.fill_color = Some(Color::blue_color());
    /// let disk = Disk::new((0.0, 0.0), 1.0, opts);
    /// ```
    pub fn new(center: impl Into<Point2D>, radius: f64, mut options: PlotOptions) -> Self {
        assert!(radius >= 0.0, "radius must be non-negative");

        // Ensure fill is enabled for a disk
        options.fill = true;
        if options.fill_color.is_none() {
            options.fill_color = Some(options.color.clone());
        }

        Self {
            center: center.into(),
            radius,
            options,
        }
    }

    /// Get the center point
    pub fn center(&self) -> Point2D {
        self.center
    }

    /// Get the radius
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Get the diameter
    pub fn diameter(&self) -> f64 {
        self.radius * 2.0
    }

    /// Get the area
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    /// Check if a point is inside the disk
    pub fn contains(&self, point: &Point2D) -> bool {
        self.center.distance_to(point) <= self.radius
    }
}

impl Renderable for Disk {
    fn bounding_box(&self) -> BoundingBox {
        BoundingBox::new(
            self.center.x - self.radius,
            self.center.x + self.radius,
            self.center.y - self.radius,
            self.center.y + self.radius,
        )
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        backend.draw_circle(self.center, self.radius, &self.options)
    }
}

impl GraphicPrimitive for Disk {
    fn options(&self) -> &PlotOptions {
        &self.options
    }

    fn options_mut(&mut self) -> &mut PlotOptions {
        &mut self.options
    }

    fn set_options(&mut self, mut options: PlotOptions) {
        // Ensure fill is always enabled for a disk
        options.fill = true;
        if options.fill_color.is_none() {
            options.fill_color = Some(options.color.clone());
        }
        self.options = options;
    }
}

/// Factory function to create a Disk primitive
///
/// # Arguments
/// * `center` - The center point of the disk
/// * `radius` - The radius of the disk
/// * `options` - Optional plot options (fill will be enabled automatically)
///
/// # Examples
/// ```ignore
/// use rustmath_plot::primitives::disk;
///
/// let d1 = disk((0.0, 0.0), 1.0, None);
/// let d2 = disk((1.0, 1.0), 2.0, Some(PlotOptions::default().with_fill(Color::red_color())));
/// ```
pub fn disk(center: impl Into<Point2D>, radius: f64, options: Option<PlotOptions>) -> Box<Disk> {
    Box::new(Disk::new(center, radius, options.unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disk_creation() {
        let d = Disk::new((0.0, 0.0), 1.0, PlotOptions::default());
        assert_eq!(d.center(), Point2D::new(0.0, 0.0));
        assert_eq!(d.radius(), 1.0);
        assert_eq!(d.diameter(), 2.0);
        assert!(d.options().fill); // Should be enabled
    }

    #[test]
    fn test_disk_area() {
        let d = Disk::new((0.0, 0.0), 1.0, PlotOptions::default());
        let area = d.area();
        assert!((area - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_disk_bounding_box() {
        let d = Disk::new((1.0, 2.0), 3.0, PlotOptions::default());
        let bbox = d.bounding_box();
        assert_eq!(bbox.xmin, -2.0);
        assert_eq!(bbox.xmax, 4.0);
        assert_eq!(bbox.ymin, -1.0);
        assert_eq!(bbox.ymax, 5.0);
    }

    #[test]
    fn test_disk_contains() {
        let d = Disk::new((0.0, 0.0), 1.0, PlotOptions::default());
        assert!(d.contains(&Point2D::new(0.0, 0.0))); // Center
        assert!(d.contains(&Point2D::new(0.5, 0.5))); // Inside
        assert!(d.contains(&Point2D::new(1.0, 0.0))); // On edge
        assert!(!d.contains(&Point2D::new(2.0, 0.0))); // Outside
    }

    #[test]
    fn test_disk_fill_color() {
        let mut opts = PlotOptions::default();
        opts.fill_color = Some(Color::red_color());
        let d = Disk::new((0.0, 0.0), 1.0, opts);
        assert!(d.options().fill);
        assert!(d.options().fill_color.is_some());
    }

    #[test]
    #[should_panic(expected = "radius must be non-negative")]
    fn test_disk_negative_radius() {
        Disk::new((0.0, 0.0), -1.0, PlotOptions::default());
    }

    #[test]
    fn test_disk_factory() {
        let d = disk((0.0, 0.0), 1.0, None);
        assert_eq!(d.radius(), 1.0);
        assert!(d.options().fill);
    }
}
