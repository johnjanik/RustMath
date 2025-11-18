//! Hyperbolic geometry primitives
//!
//! Provides support for drawing shapes in hyperbolic geometry,
//! using the Poincaré disk model or upper half-plane model.
//!
//! Based on SageMath's sage.plot.hyperbolic_* modules.

mod hyperbolic_arc;
mod hyperbolic_polygon;
mod hyperbolic_regular_polygon;

pub use hyperbolic_arc::{hyperbolic_arc, HyperbolicArc};
pub use hyperbolic_polygon::{hyperbolic_polygon, HyperbolicPolygon};
pub use hyperbolic_regular_polygon::{hyperbolic_regular_polygon, HyperbolicRegularPolygon};

/// Hyperbolic geometry model to use for rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperbolicModel {
    /// Poincaré disk model (unit disk)
    PoincareDisk,
    /// Upper half-plane model
    UpperHalfPlane,
    /// Klein disk model
    KleinDisk,
}

/// Utility functions for hyperbolic geometry
pub mod utils {
    use crate::Point2D;

    /// Convert from upper half-plane to Poincaré disk coordinates
    pub fn uhp_to_poincare(z: (f64, f64)) -> Point2D {
        let (x, y) = z;
        let denom = (x * x + (y + 1.0) * (y + 1.0));
        let u = 2.0 * x / denom;
        let v = (x * x + y * y - 1.0) / denom;
        Point2D::new(u, v)
    }

    /// Convert from Poincaré disk to upper half-plane coordinates
    pub fn poincare_to_uhp(p: Point2D) -> (f64, f64) {
        let u = p.x;
        let v = p.y;
        let denom = u * u + (1.0 - v) * (1.0 - v);
        let x = 2.0 * u / denom;
        let y = (1.0 - u * u - v * v) / denom;
        (x, y)
    }

    /// Calculate hyperbolic distance in the Poincaré disk model
    pub fn poincare_distance(p1: Point2D, p2: Point2D) -> f64 {
        let x1 = p1.x;
        let y1 = p1.y;
        let x2 = p2.x;
        let y2 = p2.y;

        let dx = x2 - x1;
        let dy = y2 - y1;
        let dist_sq = dx * dx + dy * dy;

        let r1_sq = x1 * x1 + y1 * y1;
        let r2_sq = x2 * x2 + y2 * y2;

        // Hyperbolic distance formula
        let arg = 1.0 + 2.0 * dist_sq / ((1.0 - r1_sq) * (1.0 - r2_sq));
        arg.acosh()
    }

    /// Calculate hyperbolic distance in upper half-plane model
    pub fn uhp_distance(z1: (f64, f64), z2: (f64, f64)) -> f64 {
        let (x1, y1) = z1;
        let (x2, y2) = z2;

        let dx = x2 - x1;
        let dy = y2 - y1;
        let dist_sq = dx * dx + dy * dy;

        let arg = 1.0 + dist_sq / (2.0 * y1 * y2);
        arg.acosh()
    }

    /// Check if a point is inside the Poincaré disk
    pub fn is_in_poincare_disk(p: Point2D) -> bool {
        let x = p.x;
        let y = p.y;
        x * x + y * y < 1.0
    }

    /// Check if a point is in the upper half-plane
    pub fn is_in_uhp(z: (f64, f64)) -> bool {
        z.1 > 0.0
    }

    /// Find the hyperbolic geodesic (shortest path) between two points in Poincaré disk
    /// Returns the center and radius of the circular arc representing the geodesic
    pub fn poincare_geodesic_arc(p1: Point2D, p2: Point2D) -> Option<(Point2D, f64)> {
        let x1 = p1.x;
        let y1 = p1.y;
        let x2 = p2.x;
        let y2 = p2.y;

        // Check if points are approximately on a diameter (straight line through origin)
        let det = x1 * y2 - x2 * y1;
        if det.abs() < 1e-10 {
            // Geodesic is a straight line through origin
            return None;
        }

        // Calculate the center of the circle
        let k1 = x1 * x1 + y1 * y1;
        let k2 = x2 * x2 + y2 * y2;

        let cx = ((k1 - 1.0) * y2 - (k2 - 1.0) * y1) / (2.0 * det);
        let cy = ((k2 - 1.0) * x1 - (k1 - 1.0) * x2) / (2.0 * det);

        let radius = ((cx - x1) * (cx - x1) + (cy - y1) * (cy - y1)).sqrt();

        Some((Point2D::new(cx, cy), radius))
    }
}
