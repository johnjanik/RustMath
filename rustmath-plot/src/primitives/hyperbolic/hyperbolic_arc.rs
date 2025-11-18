//! Hyperbolic arc primitive
//!
//! Draws geodesic arcs in hyperbolic geometry.

use crate::{
    BoundingBox, GraphicPrimitive, PlotOptions, Point2D, Renderable, RenderBackend, Result,
};
use super::{HyperbolicModel, utils};

/// A geodesic arc in hyperbolic geometry
///
/// Based on SageMath's HyperbolicArc from sage.plot.hyperbolic_arc
#[derive(Debug, Clone)]
pub struct HyperbolicArc {
    /// Start point
    start: Point2D,
    /// End point
    end: Point2D,
    /// Hyperbolic model to use
    model: HyperbolicModel,
    /// Rendering options
    options: PlotOptions,
}

impl HyperbolicArc {
    /// Create a new hyperbolic arc
    ///
    /// # Arguments
    ///
    /// * `start` - Starting point
    /// * `end` - Ending point
    /// * `model` - Hyperbolic model to use
    /// * `options` - Rendering options
    pub fn new(start: Point2D, end: Point2D, model: HyperbolicModel, options: PlotOptions) -> Self {
        Self {
            start,
            end,
            model,
            options,
        }
    }

    /// Get the start point
    pub fn start(&self) -> Point2D {
        self.start
    }

    /// Get the end point
    pub fn end(&self) -> Point2D {
        self.end
    }

    /// Get the hyperbolic model
    pub fn model(&self) -> HyperbolicModel {
        self.model
    }

    /// Sample points along the hyperbolic geodesic
    pub fn sample(&self, num_points: usize) -> Vec<Point2D> {
        match self.model {
            HyperbolicModel::PoincareDisk => self.sample_poincare(num_points),
            HyperbolicModel::UpperHalfPlane => self.sample_uhp(num_points),
            HyperbolicModel::KleinDisk => self.sample_klein(num_points),
        }
    }

    /// Sample in Poincaré disk model
    fn sample_poincare(&self, num_points: usize) -> Vec<Point2D> {
        // Find the geodesic (circular arc orthogonal to unit circle)
        if let Some((center, radius)) = utils::poincare_geodesic_arc(self.start, self.end) {
            // Sample along the circular arc
            let cx = center.x;
            let cy = center.y;
            let x1 = self.start.x;
            let y1 = self.start.y;
            let x2 = self.end.x;
            let y2 = self.end.y;

            let angle1 = (y1 - cy).atan2(x1 - cx);
            let angle2 = (y2 - cy).atan2(x2 - cx);

            // Determine which way to go around the arc
            let mut angle_diff = angle2 - angle1;
            if angle_diff > std::f64::consts::PI {
                angle_diff -= 2.0 * std::f64::consts::PI;
            } else if angle_diff < -std::f64::consts::PI {
                angle_diff += 2.0 * std::f64::consts::PI;
            }

            (0..num_points)
                .map(|i| {
                    let t = i as f64 / (num_points - 1) as f64;
                    let angle = angle1 + t * angle_diff;
                    let x = cx + radius * angle.cos();
                    let y = cy + radius * angle.sin();
                    Point2D::new(x, y)
                })
                .collect()
        } else {
            // Geodesic is a straight line through the origin
            (0..num_points)
                .map(|i| {
                    let t = i as f64 / (num_points - 1) as f64;
                    let x = self.start.x + t * (self.end.x - self.start.x);
                    let y = self.start.y + t * (self.end.y - self.start.y);
                    Point2D::new(x, y)
                })
                .collect()
        }
    }

    /// Sample in upper half-plane model
    fn sample_uhp(&self, num_points: usize) -> Vec<Point2D> {
        // In upper half-plane, geodesics are semicircles orthogonal to real axis
        // or vertical lines
        let x1 = self.start.x;
        let y1 = self.start.y;
        let x2 = self.end.x;
        let y2 = self.end.y;

        // Check if vertical line
        if (x1 - x2).abs() < 1e-10 {
            // Vertical geodesic
            (0..num_points)
                .map(|i| {
                    let t = i as f64 / (num_points - 1) as f64;
                    let y = y1 + t * (y2 - y1);
                    Point2D::new(x1, y)
                })
                .collect()
        } else {
            // Semicircular geodesic
            let cx = (x1 + x2) / 2.0;
            let k = (y1 * y1 + (x2 - x1) * (x2 - x1) / 4.0 - y2 * y2) / (2.0 * (x1 - x2));
            let center_x = cx + k;
            let radius = ((x1 - center_x) * (x1 - center_x) + y1 * y1).sqrt();

            let angle1 = y1.atan2(x1 - center_x);
            let angle2 = y2.atan2(x2 - center_x);

            (0..num_points)
                .map(|i| {
                    let t = i as f64 / (num_points - 1) as f64;
                    let angle = angle1 + t * (angle2 - angle1);
                    let x = center_x + radius * angle.cos();
                    let y = radius * angle.sin();
                    Point2D::new(x, y)
                })
                .collect()
        }
    }

    /// Sample in Klein disk model
    fn sample_klein(&self, num_points: usize) -> Vec<Point2D> {
        // In Klein model, geodesics are straight lines
        (0..num_points)
            .map(|i| {
                let t = i as f64 / (num_points - 1) as f64;
                let x = self.start.x + t * (self.end.x - self.start.x);
                let y = self.start.y + t * (self.end.y - self.start.y);
                Point2D::new(x, y)
            })
            .collect()
    }
}

impl Renderable for HyperbolicArc {
    fn bounding_box(&self) -> BoundingBox {
        let samples = self.sample(20);
        if samples.is_empty() {
            return BoundingBox::new(0.0, 0.0, 0.0, 0.0);
        }

        let mut xmin = samples[0].x;
        let mut xmax = samples[0].x;
        let mut ymin = samples[0].y;
        let mut ymax = samples[0].y;

        for sample in &samples {
            xmin = xmin.min(sample.x);
            xmax = xmax.max(sample.x);
            ymin = ymin.min(sample.y);
            ymax = ymax.max(sample.y);
        }

        BoundingBox::new(xmin, xmax, ymin, ymax)
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        let samples = self.sample(50);
        backend.draw_line(&samples, &self.options)?;
        Ok(())
    }
}

impl GraphicPrimitive for HyperbolicArc {
    fn options(&self) -> &PlotOptions {
        &self.options
    }

    fn options_mut(&mut self) -> &mut PlotOptions {
        &mut self.options
    }

    fn set_options(&mut self, options: PlotOptions) {
        self.options = options;
    }
}

/// Create a hyperbolic arc
///
/// # Arguments
///
/// * `start` - Starting point
/// * `end` - Ending point
/// * `model` - Optional hyperbolic model (defaults to Poincaré disk)
/// * `options` - Optional rendering options
///
/// # Examples
///
/// ```ignore
/// use rustmath_plot::primitives::hyperbolic::hyperbolic_arc;
/// use rustmath_plot::primitives::hyperbolic::HyperbolicModel;
///
/// let arc = hyperbolic_arc(
///     (0.0, 0.0),
///     (0.5, 0.5),
///     Some(HyperbolicModel::PoincareDisk),
///     None,
/// );
/// ```
pub fn hyperbolic_arc(
    start: Point2D,
    end: Point2D,
    model: Option<HyperbolicModel>,
    options: Option<PlotOptions>,
) -> HyperbolicArc {
    HyperbolicArc::new(
        start,
        end,
        model.unwrap_or(HyperbolicModel::PoincareDisk),
        options.unwrap_or_default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_arc_creation() {
        let arc = hyperbolic_arc(Point2D::new(0.0, 0.0), Point2D::new(0.5, 0.5), None, None);
        assert_eq!(arc.start(), Point2D::new(0.0, 0.0));
        assert_eq!(arc.end(), Point2D::new(0.5, 0.5));
        assert_eq!(arc.model(), HyperbolicModel::PoincareDisk);
    }

    #[test]
    fn test_hyperbolic_arc_sample_poincare() {
        let arc = hyperbolic_arc(
            Point2D::new(0.0, 0.0),
            Point2D::new(0.5, 0.0),
            Some(HyperbolicModel::PoincareDisk),
            None,
        );
        let samples = arc.sample(10);
        assert_eq!(samples.len(), 10);
        // First point should be start
        assert!((samples[0].x - 0.0).abs() < 1e-10);
        // Last point should be end
        assert!((samples[9].x - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_arc_sample_klein() {
        let arc = hyperbolic_arc(
            Point2D::new(0.0, 0.0),
            Point2D::new(0.5, 0.5),
            Some(HyperbolicModel::KleinDisk),
            None,
        );
        let samples = arc.sample(10);
        assert_eq!(samples.len(), 10);
        // In Klein model, geodesics are straight lines
        assert!((samples[5].x - 0.25).abs() < 1e-10);
        assert!((samples[5].y - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_arc_bounding_box() {
        let arc = hyperbolic_arc(Point2D::new(0.0, 0.0), Point2D::new(0.5, 0.5), None, None);
        let bbox = arc.bounding_box();
        assert!(bbox.xmin <= 0.0);
        assert!(bbox.xmax >= 0.5);
        assert!(bbox.ymin <= 0.0);
        assert!(bbox.ymax >= 0.5);
    }
}
