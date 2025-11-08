//! Polygon structures and algorithms

use crate::point::Point2D;
use std::f64::consts::PI;

/// A polygon in 2D space defined by an ordered list of vertices
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    vertices: Vec<Point2D>,
}

impl Polygon {
    /// Create a new polygon from a list of vertices
    pub fn new(vertices: Vec<Point2D>) -> Result<Self, String> {
        if vertices.len() < 3 {
            return Err("Polygon must have at least 3 vertices".to_string());
        }
        Ok(Polygon { vertices })
    }

    /// Get the vertices of the polygon
    pub fn vertices(&self) -> &[Point2D] {
        &self.vertices
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Calculate the perimeter of the polygon
    pub fn perimeter(&self) -> f64 {
        let n = self.vertices.len();
        let mut perim = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            perim += self.vertices[i].distance(&self.vertices[j]);
        }

        perim
    }

    /// Calculate the area of the polygon using the shoelace formula
    pub fn area(&self) -> f64 {
        let n = self.vertices.len();
        let mut area = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x * self.vertices[j].y;
            area -= self.vertices[j].x * self.vertices[i].y;
        }

        (area / 2.0).abs()
    }

    /// Check if a point is inside the polygon using ray casting algorithm
    pub fn contains_point(&self, p: &Point2D) -> bool {
        let n = self.vertices.len();
        let mut inside = false;

        let mut j = n - 1;
        for i in 0..n {
            let vi = &self.vertices[i];
            let vj = &self.vertices[j];

            // Check if ray from point crosses edge
            if ((vi.y > p.y) != (vj.y > p.y)) &&
               (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x) {
                inside = !inside;
            }

            j = i;
        }

        inside
    }

    /// Check if the polygon is convex
    pub fn is_convex(&self) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }

        let mut sign = None;

        for i in 0..n {
            let p1 = &self.vertices[i];
            let p2 = &self.vertices[(i + 1) % n];
            let p3 = &self.vertices[(i + 2) % n];

            let cross = Point2D::orientation(p1, p2, p3);

            if cross.abs() > 1e-10 {
                let current_sign = cross > 0.0;

                if let Some(expected_sign) = sign {
                    if current_sign != expected_sign {
                        return false;
                    }
                } else {
                    sign = Some(current_sign);
                }
            }
        }

        true
    }

    /// Calculate the centroid (center of mass) of the polygon
    pub fn centroid(&self) -> Point2D {
        let n = self.vertices.len();
        let area = self.area();

        if area < 1e-10 {
            // Degenerate polygon, return average of vertices
            let sum_x: f64 = self.vertices.iter().map(|p| p.x).sum();
            let sum_y: f64 = self.vertices.iter().map(|p| p.y).sum();
            return Point2D::new(sum_x / n as f64, sum_y / n as f64);
        }

        let mut cx = 0.0;
        let mut cy = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            let vi = &self.vertices[i];
            let vj = &self.vertices[j];

            let cross = vi.x * vj.y - vj.x * vi.y;
            cx += (vi.x + vj.x) * cross;
            cy += (vi.y + vj.y) * cross;
        }

        let factor = 1.0 / (6.0 * area);
        Point2D::new(cx * factor, cy * factor)
    }
}

/// Compute the convex hull of a set of 2D points using Graham's scan algorithm
///
/// Returns the vertices of the convex hull in counter-clockwise order
pub fn convex_hull(points: &[Point2D]) -> Vec<Point2D> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find the point with the lowest y-coordinate (and leftmost if tie)
    let mut start_idx = 0;
    for i in 1..points.len() {
        if points[i].y < points[start_idx].y ||
           (points[i].y == points[start_idx].y && points[i].x < points[start_idx].x) {
            start_idx = i;
        }
    }

    let start_point = points[start_idx];

    // Sort points by polar angle with respect to start point
    let mut sorted_points: Vec<(Point2D, f64)> = points.iter()
        .enumerate()
        .filter(|(i, _)| *i != start_idx)
        .map(|(_, &p)| {
            let angle = start_point.angle_to(&p);
            (p, angle)
        })
        .collect();

    sorted_points.sort_by(|a, b| {
        a.1.partial_cmp(&b.1).unwrap()
            .then_with(|| {
                // If angles are equal, sort by distance
                let dist_a = start_point.distance_squared(&a.0);
                let dist_b = start_point.distance_squared(&b.0);
                dist_a.partial_cmp(&dist_b).unwrap()
            })
    });

    // Build the convex hull
    let mut hull = vec![start_point];

    for (point, _) in sorted_points {
        // Remove points that make a clockwise turn
        while hull.len() >= 2 {
            let p1 = hull[hull.len() - 2];
            let p2 = hull[hull.len() - 1];

            let orientation = Point2D::orientation(&p1, &p2, &point);

            if orientation <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }

        hull.push(point);
    }

    hull
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_area() {
        // Unit square
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        let poly = Polygon::new(vertices).unwrap();
        assert!((poly.area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_perimeter() {
        // Unit square
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        let poly = Polygon::new(vertices).unwrap();
        assert!((poly.perimeter() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_in_polygon() {
        // Unit square
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(2.0, 2.0),
            Point2D::new(0.0, 2.0),
        ];
        let poly = Polygon::new(vertices).unwrap();

        // Point inside
        assert!(poly.contains_point(&Point2D::new(1.0, 1.0)));

        // Point outside
        assert!(!poly.contains_point(&Point2D::new(3.0, 3.0)));

        // Point on boundary (may be inside or outside depending on implementation)
        let boundary_point = Point2D::new(0.0, 1.0);
        // This implementation may vary for boundary points
    }

    #[test]
    fn test_polygon_is_convex() {
        // Convex polygon (square)
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        let poly = Polygon::new(vertices).unwrap();
        assert!(poly.is_convex());

        // Non-convex polygon (L-shape)
        let vertices2 = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(2.0, 1.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(1.0, 2.0),
            Point2D::new(0.0, 2.0),
        ];
        let poly2 = Polygon::new(vertices2).unwrap();
        assert!(!poly2.is_convex());
    }

    #[test]
    fn test_convex_hull_square() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(0.5, 0.5), // Interior point
        ];

        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4); // Should be the 4 corners
    }

    #[test]
    fn test_convex_hull_triangle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];

        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn test_polygon_centroid() {
        // Unit square centered at origin
        let vertices = vec![
            Point2D::new(-1.0, -1.0),
            Point2D::new(1.0, -1.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(-1.0, 1.0),
        ];
        let poly = Polygon::new(vertices).unwrap();
        let centroid = poly.centroid();

        assert!((centroid.x - 0.0).abs() < 1e-10);
        assert!((centroid.y - 0.0).abs() < 1e-10);
    }
}
