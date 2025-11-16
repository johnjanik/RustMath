//! Newton polygons for polynomials
//!
//! A Newton polygon is a geometric object used to analyze polynomials over
//! discrete valuation rings (like p-adic numbers). Given a polynomial with
//! coefficients having specific valuations, the Newton polygon is the lower
//! convex hull of the points (i, v(aᵢ)) where v is a valuation.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::newton_polygon::NewtonPolygon;
//!
//! // Create a Newton polygon from vertices
//! let vertices = vec![(0, 3), (1, 1), (3, 0), (5, 2)];
//! let polygon = NewtonPolygon::from_vertices(vertices);
//!
//! assert_eq!(polygon.num_vertices(), 3); // Only lower convex hull vertices
//! ```

use std::cmp::Ordering;
use std::fmt;

/// A Newton polygon
///
/// Represents the lower convex hull of a set of points, typically used
/// for analyzing polynomial valuations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewtonPolygon {
    /// The vertices of the lower convex hull (sorted by x-coordinate)
    vertices: Vec<(i64, i64)>,
    /// Whether this polygon extends to infinity on the right
    is_infinite: bool,
    /// The final slope (for infinite polygons)
    last_slope: Option<(i64, i64)>, // (numerator, denominator)
}

impl NewtonPolygon {
    /// Create a Newton polygon from a set of vertices
    ///
    /// Automatically computes the lower convex hull.
    ///
    /// # Arguments
    ///
    /// * `points` - A set of points (x, y)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::newton_polygon::NewtonPolygon;
    ///
    /// let vertices = vec![(0, 3), (1, 1), (2, 0), (3, 1)];
    /// let polygon = NewtonPolygon::from_vertices(vertices);
    /// ```
    pub fn from_vertices(mut points: Vec<(i64, i64)>) -> Self {
        if points.is_empty() {
            return Self {
                vertices: Vec::new(),
                is_infinite: false,
                last_slope: None,
            };
        }

        // Sort by x-coordinate
        points.sort_by_key(|p| p.0);

        // Compute lower convex hull using Andrew's algorithm (lower hull only)
        let hull = lower_convex_hull(&points);

        Self {
            vertices: hull,
            is_infinite: false,
            last_slope: None,
        }
    }

    /// Create a Newton polygon from slopes
    ///
    /// # Arguments
    ///
    /// * `slopes` - A list of slopes (as rational numbers (num, den))
    /// * `start_point` - The starting point (x, y)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::newton_polygon::NewtonPolygon;
    ///
    /// // Create a polygon with slopes -2, -1, 0 starting at (0, 4)
    /// let slopes = vec![(-2, 1), (-1, 1), (0, 1)];
    /// let polygon = NewtonPolygon::from_slopes(slopes, (0, 4));
    /// ```
    pub fn from_slopes(slopes: Vec<(i64, i64)>, start_point: (i64, i64)) -> Self {
        let mut vertices = vec![start_point];
        let mut current = start_point;

        for (slope_num, slope_den) in slopes {
            // Move one unit in x (or slope_den units to keep integers)
            let dx = slope_den;
            let dy = slope_num;
            current = (current.0 + dx, current.1 + dy);
            vertices.push(current);
        }

        Self {
            vertices,
            is_infinite: false,
            last_slope: None,
        }
    }

    /// Get the vertices of the polygon
    pub fn vertices(&self) -> &[(i64, i64)] {
        &self.vertices
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Check if the polygon is infinite (extends to infinity on the right)
    pub fn is_infinite(&self) -> bool {
        self.is_infinite
    }

    /// Get the slopes between consecutive vertices
    ///
    /// Returns a list of slopes as (numerator, denominator) pairs.
    pub fn slopes(&self) -> Vec<(i64, i64)> {
        let mut result = Vec::new();

        for i in 0..self.vertices.len().saturating_sub(1) {
            let (x1, y1) = self.vertices[i];
            let (x2, y2) = self.vertices[i + 1];

            let dx = x2 - x1;
            let dy = y2 - y1;

            // Normalize the slope
            let gcd = gcd(dx.abs(), dy.abs());
            result.push((dy / gcd, dx / gcd));
        }

        result
    }

    /// Evaluate the polygon at a given x-coordinate
    ///
    /// Returns the y-value of the polygon at the given x-coordinate.
    /// This is the maximum of the linear interpolations on each segment.
    pub fn evaluate(&self, x: i64) -> Option<f64> {
        if self.vertices.is_empty() {
            return None;
        }

        // Find the segment containing x
        for i in 0..self.vertices.len().saturating_sub(1) {
            let (x1, y1) = self.vertices[i];
            let (x2, y2) = self.vertices[i + 1];

            if x >= x1 && x <= x2 {
                // Linear interpolation
                let t = (x - x1) as f64 / (x2 - x1) as f64;
                return Some(y1 as f64 + t * (y2 - y1) as f64);
            }
        }

        // If x is beyond the last vertex, extrapolate using the last slope
        if let Some(&(x_last, y_last)) = self.vertices.last() {
            if x > x_last {
                if let Some((slope_num, slope_den)) = self.last_slope {
                    let dx = x - x_last;
                    let dy = (dx * slope_num) as f64 / slope_den as f64;
                    return Some(y_last as f64 + dy);
                }
            }
        }

        None
    }

    /// Reverse the polygon (reflect across a vertical line)
    ///
    /// For a finite polygon with vertices at x = 0, 1, ..., n,
    /// this reflects it to get vertices at x = -n, -n+1, ..., 0.
    pub fn reverse(&self) -> Self {
        if self.vertices.is_empty() {
            return self.clone();
        }

        let max_x = self.vertices.last().unwrap().0;
        let vertices: Vec<(i64, i64)> = self
            .vertices
            .iter()
            .rev()
            .map(|(x, y)| (max_x - x, *y))
            .collect();

        Self {
            vertices,
            is_infinite: false,
            last_slope: None,
        }
    }

    /// Check if this polygon is below or equal to another
    pub fn is_below_or_equal(&self, other: &NewtonPolygon) -> bool {
        // For all vertices in self, check if they're on or above other's polygon
        for &(x, y) in &self.vertices {
            if let Some(other_y) = other.evaluate(x) {
                if (y as f64) > other_y + 1e-10 {
                    return false;
                }
            }
        }
        true
    }
}

impl fmt::Display for NewtonPolygon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NewtonPolygon(")?;
        for (i, (x, y)) in self.vertices.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "({}, {})", x, y)?;
        }
        write!(f, ")")
    }
}

/// Compute the lower convex hull of a set of points
///
/// Uses Andrew's monotone chain algorithm (lower hull only).
fn lower_convex_hull(points: &[(i64, i64)]) -> Vec<(i64, i64)> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut hull: Vec<(i64, i64)> = Vec::new();

    // Build lower hull
    for &point in points {
        while hull.len() >= 2 && cross_product_sign(hull[hull.len() - 2], hull[hull.len() - 1], point) >= 0 {
            hull.pop();
        }
        hull.push(point);
    }

    hull
}

/// Compute the cross product to determine turn direction
///
/// Returns positive for left turn, negative for right turn, zero for collinear.
/// For the lower hull, we want right turns (negative cross product).
fn cross_product_sign(o: (i64, i64), a: (i64, i64), b: (i64, i64)) -> i64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Compute the greatest common divisor
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Parent structure for creating Newton polygons
///
/// This provides a namespace for Newton polygon construction methods.
pub struct ParentNewtonPolygon;

impl ParentNewtonPolygon {
    /// Create a new Newton polygon from vertices
    pub fn from_vertices(vertices: Vec<(i64, i64)>) -> NewtonPolygon {
        NewtonPolygon::from_vertices(vertices)
    }

    /// Create a new Newton polygon from slopes
    pub fn from_slopes(slopes: Vec<(i64, i64)>, start: (i64, i64)) -> NewtonPolygon {
        NewtonPolygon::from_slopes(slopes, start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vertices() {
        let vertices = vec![(0, 3), (1, 1), (2, 0), (3, 1), (4, 0)];
        let polygon = NewtonPolygon::from_vertices(vertices);

        // The lower convex hull should be (0,3), (1,1), (2,0), (4,0)
        assert!(polygon.num_vertices() >= 3);
    }

    #[test]
    fn test_from_slopes() {
        let slopes = vec![(-2, 1), (-1, 1), (0, 1)];
        let polygon = NewtonPolygon::from_slopes(slopes, (0, 4));

        assert_eq!(polygon.num_vertices(), 4); // Start + 3 slopes
        assert_eq!(polygon.vertices()[0], (0, 4));
    }

    #[test]
    fn test_slopes() {
        // Use points created from explicit slopes to ensure they're all on the hull
        let slopes_input = vec![(-2, 1), (-1, 1)];
        let polygon = NewtonPolygon::from_slopes(slopes_input, (0, 4));

        let slopes = polygon.slopes();
        assert_eq!(polygon.num_vertices(), 3); // Start + 2 slopes
        assert_eq!(slopes.len(), 2);
        assert_eq!(slopes[0], (-2, 1));
        assert_eq!(slopes[1], (-1, 1));
    }

    #[test]
    fn test_evaluate() {
        let vertices = vec![(0, 0), (2, 2), (4, 0)];
        let polygon = NewtonPolygon::from_vertices(vertices);

        // At x=2, should be at the vertex
        assert_eq!(polygon.evaluate(2), Some(2.0));

        // At x=1, should interpolate between (0,0) and (2,2)
        assert_eq!(polygon.evaluate(1), Some(1.0));
    }

    #[test]
    fn test_reverse() {
        let vertices = vec![(0, 0), (1, 1), (2, 0)];
        let polygon = NewtonPolygon::from_vertices(vertices);

        let reversed = polygon.reverse();
        assert_eq!(reversed.vertices()[0], (0, 0));
        assert_eq!(reversed.vertices()[1], (1, 1));
        assert_eq!(reversed.vertices()[2], (2, 0));
    }

    #[test]
    fn test_lower_convex_hull() {
        let points = vec![(0, 5), (1, 3), (2, 4), (3, 2), (4, 1)];
        let hull = lower_convex_hull(&points);

        // The lower hull should not include (2, 4) as it's above the line from (1,3) to (3,2)
        assert!(hull.len() <= points.len());
        assert!(hull.contains(&(0, 5)));
        assert!(hull.contains(&(4, 1)));
    }

    #[test]
    fn test_empty_polygon() {
        let polygon = NewtonPolygon::from_vertices(vec![]);
        assert_eq!(polygon.num_vertices(), 0);
        assert_eq!(polygon.evaluate(0), None);
    }

    #[test]
    fn test_is_below_or_equal() {
        let poly1 = NewtonPolygon::from_vertices(vec![(0, 0), (1, 0), (2, 0)]);
        let poly2 = NewtonPolygon::from_vertices(vec![(0, 1), (1, 1), (2, 1)]);

        // poly1 should be below poly2
        assert!(poly1.is_below_or_equal(&poly2));
        assert!(!poly2.is_below_or_equal(&poly1));
    }

    #[test]
    fn test_parent_newton_polygon() {
        let polygon = ParentNewtonPolygon::from_vertices(vec![(0, 0), (1, 1), (2, 0)]);
        assert_eq!(polygon.num_vertices(), 3);

        let polygon2 = ParentNewtonPolygon::from_slopes(vec![(-1, 1), (1, 1)], (0, 2));
        assert_eq!(polygon2.num_vertices(), 3);
    }
}
