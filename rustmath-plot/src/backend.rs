//! Rendering backend utilities and helpers
//!
//! This module provides utilities for implementing rendering backends.
//! The main RenderBackend trait is defined in rustmath-plot-core.

use rustmath_plot_core::{BoundingBox, Point2D, Transform2D};

/// Helper for calculating viewport transformations
///
/// This converts from data coordinates to screen/pixel coordinates.
pub struct ViewportTransform {
    /// Transformation matrix
    transform: Transform2D,
}

impl ViewportTransform {
    /// Create a new viewport transform
    ///
    /// # Arguments
    /// * `data_bbox` - The bounding box in data coordinates
    /// * `viewport_width` - Width of the viewport in pixels
    /// * `viewport_height` - Height of the viewport in pixels
    /// * `preserve_aspect` - Whether to preserve aspect ratio
    pub fn new(
        data_bbox: &BoundingBox,
        viewport_width: f64,
        viewport_height: f64,
        preserve_aspect: bool,
    ) -> Self {
        let data_width = data_bbox.width();
        let data_height = data_bbox.height();

        // Calculate scale factors
        let mut scale_x = viewport_width / data_width;
        let mut scale_y = viewport_height / data_height;

        if preserve_aspect {
            // Use the smaller scale to ensure everything fits
            let scale = scale_x.min(scale_y);
            scale_x = scale;
            scale_y = scale;
        }

        // Note: In screen coordinates, y typically increases downward,
        // so we flip the y-axis
        scale_y = -scale_y;

        // Create transformation:
        // 1. Translate to move data origin to (0, 0)
        // 2. Scale to viewport size
        // 3. Translate to center in viewport if needed

        // Start with scaling
        let mut transform = Transform2D::scale(scale_x, scale_y);

        // Translate from data coordinates
        // We want to map data_bbox.xmin to 0 and data_bbox.ymax to 0 (top-left)
        let translate_x = -data_bbox.xmin * scale_x;
        let translate_y = -data_bbox.ymax * scale_y;

        transform = Transform2D::translate(translate_x, translate_y).compose(&transform);

        Self { transform }
    }

    /// Transform a point from data coordinates to viewport coordinates
    pub fn transform_point(&self, point: &Point2D) -> Point2D {
        self.transform.transform_point(*point)
    }

    /// Transform an array of points
    pub fn transform_points(&self, points: &[Point2D]) -> Vec<Point2D> {
        points.iter().map(|p| self.transform_point(p)).collect()
    }

    /// Get the underlying transformation matrix
    pub fn matrix(&self) -> &Transform2D {
        &self.transform
    }
}

/// Helper for color interpolation in gradients
pub struct ColorInterpolator;

impl ColorInterpolator {
    /// Linear interpolation between two values
    pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t
    }

    /// Clamp a value between 0 and 1
    pub fn clamp01(value: f64) -> f64 {
        value.clamp(0.0, 1.0)
    }
}

/// Helper for adaptive sampling of functions
///
/// This is useful for plotting functions with varying detail levels.
pub struct AdaptiveSampler {
    /// Maximum recursion depth
    max_depth: usize,

    /// Tolerance for linearity
    tolerance: f64,
}

impl AdaptiveSampler {
    /// Create a new adaptive sampler
    ///
    /// # Arguments
    /// * `max_depth` - Maximum recursion depth
    /// * `tolerance` - Tolerance for linearity (smaller = more detail)
    pub fn new(max_depth: usize, tolerance: f64) -> Self {
        Self {
            max_depth,
            tolerance,
        }
    }

    /// Check if three points are approximately collinear
    fn is_linear(&self, p1: &Point2D, p2: &Point2D, p3: &Point2D) -> bool {
        // Calculate the distance from p2 to the line p1-p3
        let dx = p3.x - p1.x;
        let dy = p3.y - p1.y;
        let len = (dx * dx + dy * dy).sqrt();

        if len == 0.0 {
            return true;
        }

        // Distance from p2 to line p1-p3
        let dist = ((p2.x - p1.x) * dy - (p2.y - p1.y) * dx).abs() / len;

        dist < self.tolerance
    }

    /// Sample a function adaptively between two x values
    ///
    /// # Arguments
    /// * `f` - The function to sample
    /// * `x1` - Start x value
    /// * `x2` - End x value
    /// * `depth` - Current recursion depth
    ///
    /// # Returns
    /// A vector of sampled points
    pub fn sample<F>(&self, f: &F, x1: f64, x2: f64, depth: usize) -> Vec<Point2D>
    where
        F: Fn(f64) -> f64,
    {
        let p1 = Point2D::new(x1, f(x1));
        let p3 = Point2D::new(x2, f(x2));

        if depth >= self.max_depth {
            return vec![p1, p3];
        }

        let x_mid = (x1 + x2) / 2.0;
        let p2 = Point2D::new(x_mid, f(x_mid));

        if self.is_linear(&p1, &p2, &p3) {
            vec![p1, p3]
        } else {
            // Recursively sample left and right halves
            let mut left = self.sample(f, x1, x_mid, depth + 1);
            let right = self.sample(f, x_mid, x2, depth + 1);

            // Remove duplicate point at midpoint
            left.pop();
            left.extend(right);
            left
        }
    }

    /// Sample a function over a range with initial uniform sampling
    ///
    /// # Arguments
    /// * `f` - The function to sample
    /// * `x_min` - Minimum x value
    /// * `x_max` - Maximum x value
    /// * `initial_points` - Number of initial uniform samples
    pub fn sample_range<F>(
        &self,
        f: &F,
        x_min: f64,
        x_max: f64,
        initial_points: usize,
    ) -> Vec<Point2D>
    where
        F: Fn(f64) -> f64,
    {
        if initial_points < 2 {
            return self.sample(f, x_min, x_max, 0);
        }

        let mut result = Vec::new();
        let dx = (x_max - x_min) / (initial_points - 1) as f64;

        for i in 0..initial_points - 1 {
            let x1 = x_min + i as f64 * dx;
            let x2 = x_min + (i + 1) as f64 * dx;

            let mut segment = self.sample(f, x1, x2, 0);
            if i > 0 {
                // Remove duplicate point
                segment.remove(0);
            }
            result.extend(segment);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewport_transform() {
        let bbox = BoundingBox::new(0.0, 10.0, 0.0, 10.0);
        let transform = ViewportTransform::new(&bbox, 100.0, 100.0, true);

        let p = Point2D::new(5.0, 5.0);
        let transformed = transform.transform_point(&p);

        // Should map to middle of viewport
        // Note: y-axis is flipped
        assert_eq!(transformed.x, 50.0);
        assert_eq!(transformed.y, 50.0);
    }

    #[test]
    fn test_color_interpolator() {
        let result = ColorInterpolator::lerp(0.0, 10.0, 0.5);
        assert_eq!(result, 5.0);

        let result = ColorInterpolator::lerp(0.0, 10.0, 0.0);
        assert_eq!(result, 0.0);

        let result = ColorInterpolator::lerp(0.0, 10.0, 1.0);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_clamp01() {
        assert_eq!(ColorInterpolator::clamp01(-0.5), 0.0);
        assert_eq!(ColorInterpolator::clamp01(0.5), 0.5);
        assert_eq!(ColorInterpolator::clamp01(1.5), 1.0);
    }

    #[test]
    fn test_adaptive_sampler_linear() {
        let sampler = AdaptiveSampler::new(10, 0.01);

        // Linear function should need few points
        let f = |x: f64| 2.0 * x + 1.0;
        let points = sampler.sample(&f, 0.0, 10.0, 0);

        // Should only need start and end points for a linear function
        assert_eq!(points.len(), 2);
        assert_eq!(points[0], Point2D::new(0.0, 1.0));
        assert_eq!(points[1], Point2D::new(10.0, 21.0));
    }

    #[test]
    fn test_adaptive_sampler_nonlinear() {
        let sampler = AdaptiveSampler::new(10, 0.01);

        // Non-linear function should need more points
        let f = |x: f64| x * x;
        let points = sampler.sample(&f, 0.0, 10.0, 0);

        // Should need more than 2 points for a quadratic
        assert!(points.len() > 2);
    }

    #[test]
    fn test_adaptive_sampler_range() {
        let sampler = AdaptiveSampler::new(10, 0.1);

        let f = |x: f64| x.sin();
        let points = sampler.sample_range(&f, 0.0, 2.0 * std::f64::consts::PI, 10);

        // Should have sampled points
        assert!(!points.is_empty());
        assert_eq!(points[0].x, 0.0);
    }
}
