//! 2D triangulation algorithms

use crate::point::Point2D;
use std::collections::HashSet;

/// A triangle in 2D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    pub a: Point2D,
    pub b: Point2D,
    pub c: Point2D,
}

impl Triangle {
    /// Create a new triangle
    pub fn new(a: Point2D, b: Point2D, c: Point2D) -> Self {
        Triangle { a, b, c }
    }

    /// Calculate the circumcenter of the triangle
    pub fn circumcenter(&self) -> Point2D {
        let ax = self.a.x;
        let ay = self.a.y;
        let bx = self.b.x;
        let by = self.b.y;
        let cx = self.c.x;
        let cy = self.c.y;

        let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

        if d.abs() < 1e-10 {
            // Degenerate triangle, return centroid
            return Point2D::new(
                (ax + bx + cx) / 3.0,
                (ay + by + cy) / 3.0,
            );
        }

        let ux = ((ax * ax + ay * ay) * (by - cy) +
                   (bx * bx + by * by) * (cy - ay) +
                   (cx * cx + cy * cy) * (ay - by)) / d;

        let uy = ((ax * ax + ay * ay) * (cx - bx) +
                   (bx * bx + by * by) * (ax - cx) +
                   (cx * cx + cy * cy) * (bx - ax)) / d;

        Point2D::new(ux, uy)
    }

    /// Calculate the circumradius of the triangle
    pub fn circumradius(&self) -> f64 {
        let center = self.circumcenter();
        center.distance(&self.a)
    }

    /// Check if a point is inside the circumcircle of this triangle
    pub fn in_circumcircle(&self, p: &Point2D) -> bool {
        let center = self.circumcenter();
        let radius = self.circumradius();
        center.distance(p) < radius + 1e-10
    }

    /// Check if this triangle contains a given edge
    pub fn has_edge(&self, p1: &Point2D, p2: &Point2D) -> bool {
        (self.a == *p1 && self.b == *p2) ||
        (self.a == *p2 && self.b == *p1) ||
        (self.b == *p1 && self.c == *p2) ||
        (self.b == *p2 && self.c == *p1) ||
        (self.c == *p1 && self.a == *p2) ||
        (self.c == *p2 && self.a == *p1)
    }

    /// Get the three edges of this triangle
    pub fn edges(&self) -> [(Point2D, Point2D); 3] {
        [(self.a, self.b), (self.b, self.c), (self.c, self.a)]
    }
}

/// Compute Delaunay triangulation using Bowyer-Watson algorithm
///
/// Returns a list of triangles that form the Delaunay triangulation
pub fn delaunay_triangulation(points: &[Point2D]) -> Vec<Triangle> {
    if points.len() < 3 {
        return vec![];
    }

    // Find bounding box
    let mut min_x = points[0].x;
    let mut max_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_y = points[0].y;

    for p in points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    // Create super-triangle that contains all points
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta_max = dx.max(dy);
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    let p1 = Point2D::new(mid_x - 20.0 * delta_max, mid_y - delta_max);
    let p2 = Point2D::new(mid_x, mid_y + 20.0 * delta_max);
    let p3 = Point2D::new(mid_x + 20.0 * delta_max, mid_y - delta_max);

    let mut triangles = vec![Triangle::new(p1, p2, p3)];

    // Add points one at a time
    for point in points {
        let mut bad_triangles = Vec::new();

        // Find all triangles whose circumcircle contains the point
        for (i, tri) in triangles.iter().enumerate() {
            if tri.in_circumcircle(point) {
                bad_triangles.push(i);
            }
        }

        // Find the boundary of the polygonal hole
        let mut polygon = Vec::new();

        for &idx in &bad_triangles {
            let tri = &triangles[idx];
            for edge in &tri.edges() {
                // Check if edge is shared by another bad triangle
                let mut is_shared = false;
                for &other_idx in &bad_triangles {
                    if other_idx != idx {
                        if triangles[other_idx].has_edge(&edge.0, &edge.1) {
                            is_shared = true;
                            break;
                        }
                    }
                }

                if !is_shared {
                    polygon.push(*edge);
                }
            }
        }

        // Remove bad triangles (in reverse order to maintain indices)
        for &idx in bad_triangles.iter().rev() {
            triangles.remove(idx);
        }

        // Create new triangles from the point to each edge of the polygon
        for edge in polygon {
            triangles.push(Triangle::new(*point, edge.0, edge.1));
        }
    }

    // Remove triangles that contain vertices from the super-triangle
    triangles.retain(|tri| {
        tri.a != p1 && tri.a != p2 && tri.a != p3 &&
        tri.b != p1 && tri.b != p2 && tri.b != p3 &&
        tri.c != p1 && tri.c != p2 && tri.c != p3
    });

    triangles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_circumcenter() {
        // Right triangle at origin
        let tri = Triangle::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        );

        let center = tri.circumcenter();

        // For a right triangle, circumcenter is at the midpoint of the hypotenuse
        assert!((center.x - 0.5).abs() < 1e-10);
        assert!((center.y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_delaunay_triangulation_square() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];

        let triangles = delaunay_triangulation(&points);

        // A square should be triangulated into 2 triangles
        assert_eq!(triangles.len(), 2);
    }

    #[test]
    fn test_delaunay_triangulation_triangle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];

        let triangles = delaunay_triangulation(&points);

        // Three points should form one triangle
        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn test_triangle_in_circumcircle() {
        let tri = Triangle::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        );

        // Point at the center should be inside
        let center = Point2D::new(0.3, 0.3);
        assert!(tri.in_circumcircle(&center));

        // Point far away should be outside
        let far = Point2D::new(10.0, 10.0);
        assert!(!tri.in_circumcircle(&far));
    }
}
