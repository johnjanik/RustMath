//! Bezier path primitives for smooth curves
//!
//! Provides support for cubic and quadratic Bezier curves,
//! similar to SageMath's bezier_path functionality.
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_plot::primitives::bezier_path;
//!
//! // Create a cubic Bezier curve
//! let path = bezier_path(
//!     vec![
//!         (0.0, 0.0),
//!         (1.0, 2.0),
//!         (2.0, 2.0),
//!         (3.0, 0.0),
//!     ],
//!     None,
//! );
//! ```

use crate::{
    BoundingBox, Color, GraphicPrimitive, LineStyle, PlotOptions, Point2D, Renderable,
    RenderBackend, Result,
};

/// A Bezier path consisting of one or more Bezier curve segments
///
/// Based on SageMath's BezierPath class from sage.plot.bezier_path
#[derive(Debug, Clone)]
pub struct BezierPath {
    /// Control points for the Bezier curves
    /// For cubic Bezier: groups of 4 points (start, control1, control2, end)
    /// For quadratic Bezier: groups of 3 points (start, control, end)
    points: Vec<Point2D>,
    /// Options for rendering the path
    options: PlotOptions,
    /// Type of Bezier curve (cubic or quadratic)
    curve_type: BezierType,
}

/// Type of Bezier curve
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BezierType {
    /// Cubic Bezier curves (4 control points per segment)
    Cubic,
    /// Quadratic Bezier curves (3 control points per segment)
    Quadratic,
}

impl BezierPath {
    /// Create a new cubic Bezier path from control points
    ///
    /// # Arguments
    ///
    /// * `points` - Control points (must be a multiple of 4 for cubic, or 3 for quadratic)
    /// * `options` - Rendering options
    ///
    /// # Panics
    ///
    /// Panics if the number of points is not appropriate for the curve type
    pub fn new(points: Vec<Point2D>, options: PlotOptions, curve_type: BezierType) -> Self {
        let points_per_segment = match curve_type {
            BezierType::Cubic => 4,
            BezierType::Quadratic => 3,
        };

        assert!(
            points.len() >= points_per_segment,
            "Need at least {} points for a {:?} Bezier curve",
            points_per_segment,
            curve_type
        );

        Self {
            points,
            options,
            curve_type,
        }
    }

    /// Create a cubic Bezier path
    pub fn cubic(points: Vec<Point2D>, options: PlotOptions) -> Self {
        Self::new(points, options, BezierType::Cubic)
    }

    /// Create a quadratic Bezier path
    pub fn quadratic(points: Vec<Point2D>, options: PlotOptions) -> Self {
        Self::new(points, options, BezierType::Quadratic)
    }

    /// Get the control points
    pub fn points(&self) -> &[Point2D] {
        &self.points
    }

    /// Get the curve type
    pub fn curve_type(&self) -> BezierType {
        self.curve_type
    }

    /// Evaluate a cubic Bezier curve at parameter t ∈ [0, 1]
    ///
    /// Uses the formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    fn eval_cubic(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D, t: f64) -> Point2D {
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        let x = mt3 * p0.x + 3.0 * mt2 * t * p1.x + 3.0 * mt * t2 * p2.x + t3 * p3.x;
        let y = mt3 * p0.y + 3.0 * mt2 * t * p1.y + 3.0 * mt * t2 * p2.y + t3 * p3.y;

        Point2D::new(x, y)
    }

    /// Evaluate a quadratic Bezier curve at parameter t ∈ [0, 1]
    ///
    /// Uses the formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
    fn eval_quadratic(p0: Point2D, p1: Point2D, p2: Point2D, t: f64) -> Point2D {
        let t2 = t * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;

        let x = mt2 * p0.x + 2.0 * mt * t * p1.x + t2 * p2.x;
        let y = mt2 * p0.y + 2.0 * mt * t * p1.y + t2 * p2.y;

        Point2D::new(x, y)
    }

    /// Sample points along the Bezier path for rendering
    ///
    /// # Arguments
    ///
    /// * `num_points` - Number of points to sample per segment
    pub fn sample(&self, num_points: usize) -> Vec<Point2D> {
        let mut result = Vec::new();

        match self.curve_type {
            BezierType::Cubic => {
                for chunk in self.points.chunks(4) {
                    if chunk.len() == 4 {
                        for i in 0..num_points {
                            let t = i as f64 / (num_points - 1) as f64;
                            let point = Self::eval_cubic(chunk[0], chunk[1], chunk[2], chunk[3], t);
                            result.push(point);
                        }
                    }
                }
            }
            BezierType::Quadratic => {
                for chunk in self.points.chunks(3) {
                    if chunk.len() == 3 {
                        for i in 0..num_points {
                            let t = i as f64 / (num_points - 1) as f64;
                            let point = Self::eval_quadratic(chunk[0], chunk[1], chunk[2], t);
                            result.push(point);
                        }
                    }
                }
            }
        }

        result
    }

    /// Get the number of curve segments in the path
    pub fn num_segments(&self) -> usize {
        match self.curve_type {
            BezierType::Cubic => (self.points.len() - 1) / 3,
            BezierType::Quadratic => (self.points.len() - 1) / 2,
        }
    }

    /// Split a segment into two segments at parameter t
    ///
    /// Uses De Casteljau's algorithm
    pub fn split_cubic_segment(
        p0: Point2D,
        p1: Point2D,
        p2: Point2D,
        p3: Point2D,
        t: f64,
    ) -> ([Point2D; 4], [Point2D; 4]) {
        // First level of interpolation
        let q0 = Self::lerp(p0, p1, t);
        let q1 = Self::lerp(p1, p2, t);
        let q2 = Self::lerp(p2, p3, t);

        // Second level
        let r0 = Self::lerp(q0, q1, t);
        let r1 = Self::lerp(q1, q2, t);

        // Third level (point on curve)
        let s = Self::lerp(r0, r1, t);

        // First segment: p0, q0, r0, s
        // Second segment: s, r1, q2, p3
        ([p0, q0, r0, s], [s, r1, q2, p3])
    }

    /// Linear interpolation between two points
    fn lerp(p0: Point2D, p1: Point2D, t: f64) -> Point2D {
        Point2D::new(p0.x + t * (p1.x - p0.x), p0.y + t * (p1.y - p0.y))
    }

    /// Calculate the arc length of the Bezier curve using adaptive subdivision
    pub fn arc_length(&self, tolerance: f64) -> f64 {
        let mut total_length = 0.0;

        match self.curve_type {
            BezierType::Cubic => {
                for chunk in self.points.chunks(4) {
                    if chunk.len() == 4 {
                        total_length +=
                            Self::arc_length_cubic(chunk[0], chunk[1], chunk[2], chunk[3], tolerance);
                    }
                }
            }
            BezierType::Quadratic => {
                for chunk in self.points.chunks(3) {
                    if chunk.len() == 3 {
                        total_length += Self::arc_length_quadratic(chunk[0], chunk[1], chunk[2], tolerance);
                    }
                }
            }
        }

        total_length
    }

    /// Calculate arc length of a cubic Bezier segment
    fn arc_length_cubic(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D, tolerance: f64) -> f64 {
        // Approximate by sampling
        let samples = 100;
        let mut length = 0.0;
        let mut prev = p0;

        for i in 1..=samples {
            let t = i as f64 / samples as f64;
            let curr = Self::eval_cubic(p0, p1, p2, p3, t);
            let dx = curr.x - prev.x;
            let dy = curr.y - prev.y;
            length += (dx * dx + dy * dy).sqrt();
            prev = curr;
        }

        length
    }

    /// Calculate arc length of a quadratic Bezier segment
    fn arc_length_quadratic(p0: Point2D, p1: Point2D, p2: Point2D, tolerance: f64) -> f64 {
        let samples = 100;
        let mut length = 0.0;
        let mut prev = p0;

        for i in 1..=samples {
            let t = i as f64 / samples as f64;
            let curr = Self::eval_quadratic(p0, p1, p2, t);
            let dx = curr.x - prev.x;
            let dy = curr.y - prev.y;
            length += (dx * dx + dy * dy).sqrt();
            prev = curr;
        }

        length
    }
}

impl Renderable for BezierPath {
    fn bounding_box(&self) -> BoundingBox {
        if self.points.is_empty() {
            return BoundingBox::new(0.0, 0.0, 0.0, 0.0);
        }

        let mut xmin = self.points[0].x;
        let mut xmax = self.points[0].x;
        let mut ymin = self.points[0].y;
        let mut ymax = self.points[0].y;

        for point in &self.points {
            xmin = xmin.min(point.x);
            xmax = xmax.max(point.x);
            ymin = ymin.min(point.y);
            ymax = ymax.max(point.y);
        }

        BoundingBox::new(xmin, xmax, ymin, ymax)
    }

    fn render(&self, backend: &mut dyn RenderBackend) -> Result<()> {
        // Sample the Bezier curve and render as a polyline
        let samples = self.sample(50); // 50 points per segment

        if samples.is_empty() {
            return Ok(());
        }

        backend.draw_line(&samples, &self.options)?;
        Ok(())
    }
}

impl GraphicPrimitive for BezierPath {
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

/// Create a cubic Bezier path
///
/// # Arguments
///
/// * `points` - Control points (should be groups of 4 for cubic Bezier)
/// * `options` - Optional rendering options
///
/// # Examples
///
/// ```ignore
/// use rustmath_plot::primitives::bezier_path;
///
/// let path = bezier_path(
///     vec![(0.0, 0.0), (1.0, 2.0), (2.0, 2.0), (3.0, 0.0)],
///     None,
/// );
/// ```
pub fn bezier_path(points: Vec<Point2D>, options: Option<PlotOptions>) -> BezierPath {
    BezierPath::cubic(points, options.unwrap_or_default())
}

/// Create a quadratic Bezier path
///
/// # Arguments
///
/// * `points` - Control points (should be groups of 3 for quadratic Bezier)
/// * `options` - Optional rendering options
///
/// # Examples
///
/// ```ignore
/// use rustmath_plot::primitives::bezier_quadratic;
///
/// let path = bezier_quadratic(
///     vec![(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)],
///     None,
/// );
/// ```
pub fn bezier_quadratic(points: Vec<Point2D>, options: Option<PlotOptions>) -> BezierPath {
    BezierPath::quadratic(points, options.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bezier_path_creation() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(2.0, 1.0),
            Point2D::new(3.0, 0.0),
        ];
        let path = bezier_path(points.clone(), None);
        assert_eq!(path.points(), &points);
        assert_eq!(path.curve_type(), BezierType::Cubic);
    }

    #[test]
    fn test_quadratic_bezier_creation() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 2.0),
            Point2D::new(2.0, 0.0),
        ];
        let path = bezier_quadratic(points.clone(), None);
        assert_eq!(path.points(), &points);
        assert_eq!(path.curve_type(), BezierType::Quadratic);
    }

    #[test]
    fn test_cubic_bezier_eval() {
        let p0 = Point2D::new(0.0, 0.0);
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(2.0, 2.0);
        let p3 = Point2D::new(3.0, 0.0);

        // At t=0, should be p0
        let start = BezierPath::eval_cubic(p0, p1, p2, p3, 0.0);
        assert!((start.x - p0.x).abs() < 1e-10);
        assert!((start.y - p0.y).abs() < 1e-10);

        // At t=1, should be p3
        let end = BezierPath::eval_cubic(p0, p1, p2, p3, 1.0);
        assert!((end.x - p3.x).abs() < 1e-10);
        assert!((end.y - p3.y).abs() < 1e-10);

        // At t=0.5, should be somewhere in the middle
        let mid = BezierPath::eval_cubic(p0, p1, p2, p3, 0.5);
        assert!(mid.x > 0.0 && mid.x < 3.0);
        assert!(mid.y > 0.0);
    }

    #[test]
    fn test_quadratic_bezier_eval() {
        let p0 = Point2D::new(0.0, 0.0);
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(2.0, 0.0);

        // At t=0, should be p0
        let start = BezierPath::eval_quadratic(p0, p1, p2, 0.0);
        assert!((start.x - p0.x).abs() < 1e-10);
        assert!((start.y - p0.y).abs() < 1e-10);

        // At t=1, should be p2
        let end = BezierPath::eval_quadratic(p0, p1, p2, 1.0);
        assert!((end.x - p2.x).abs() < 1e-10);
        assert!((end.y - p2.y).abs() < 1e-10);
    }

    #[test]
    fn test_bezier_sample() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(2.0, 1.0),
            Point2D::new(3.0, 0.0),
        ];
        let path = bezier_path(points, None);
        let samples = path.sample(10);

        assert_eq!(samples.len(), 10);
        // First point should be close to start
        assert!((samples[0].x - 0.0).abs() < 1e-10);
        // Last point should be close to end
        assert!((samples[9].x - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bezier_num_segments() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(2.0, 1.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(4.0, -1.0),
            Point2D::new(5.0, -1.0),
            Point2D::new(6.0, 0.0),
        ];
        let path = bezier_path(points, None);
        assert_eq!(path.num_segments(), 2);
    }

    #[test]
    fn test_bezier_split_segment() {
        let p0 = Point2D::new(0.0, 0.0);
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(2.0, 2.0);
        let p3 = Point2D::new(3.0, 0.0);

        let (first, second) = BezierPath::split_cubic_segment(p0, p1, p2, p3, 0.5);

        // First segment should start at p0
        assert_eq!(first[0].x, p0.x);
        assert_eq!(first[0].y, p0.y);
        // Second segment should end at p3
        assert_eq!(second[3].x, p3.x);
        assert_eq!(second[3].y, p3.y);
        // The segments should connect
        assert_eq!(first[3].x, second[0].x);
        assert_eq!(first[3].y, second[0].y);
    }

    #[test]
    fn test_bezier_lerp() {
        let p0 = Point2D::new(0.0, 0.0);
        let p1 = Point2D::new(2.0, 4.0);

        let mid = BezierPath::lerp(p0, p1, 0.5);
        assert!((mid.x - 1.0).abs() < 1e-10);
        assert!((mid.y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bezier_arc_length() {
        // Straight line from (0,0) to (3,0)
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(3.0, 0.0),
        ];
        let path = bezier_path(points, None);
        let length = path.arc_length(0.01);

        // Should be approximately 3.0
        assert!((length - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_bezier_bounding_box() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 2.0),
            Point2D::new(2.0, 2.0),
            Point2D::new(3.0, 0.0),
        ];
        let path = bezier_path(points, None);
        let bbox = path.bounding_box();

        assert_eq!(bbox.xmin, 0.0);
        assert_eq!(bbox.xmax, 3.0);
        assert_eq!(bbox.ymin, 0.0);
        assert_eq!(bbox.ymax, 2.0);
    }
}
