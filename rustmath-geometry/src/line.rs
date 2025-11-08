//! Line and line segment structures

use crate::point::Point2D;

/// A line segment in 2D space defined by two endpoints
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSegment2D {
    pub start: Point2D,
    pub end: Point2D,
}

impl LineSegment2D {
    /// Create a new line segment
    pub fn new(start: Point2D, end: Point2D) -> Self {
        LineSegment2D { start, end }
    }

    /// Calculate the length of the line segment
    pub fn length(&self) -> f64 {
        self.start.distance(&self.end)
    }

    /// Calculate the midpoint
    pub fn midpoint(&self) -> Point2D {
        Point2D::new(
            (self.start.x + self.end.x) / 2.0,
            (self.start.y + self.end.y) / 2.0,
        )
    }

    /// Check if a point lies on this line segment
    pub fn contains_point(&self, p: &Point2D) -> bool {
        // Check if point is collinear
        if !Point2D::collinear(&self.start, &self.end, p) {
            return false;
        }

        // Check if point is within the bounding box
        let min_x = self.start.x.min(self.end.x);
        let max_x = self.start.x.max(self.end.x);
        let min_y = self.start.y.min(self.end.y);
        let max_y = self.start.y.max(self.end.y);

        p.x >= min_x - 1e-10 && p.x <= max_x + 1e-10 &&
        p.y >= min_y - 1e-10 && p.y <= max_y + 1e-10
    }

    /// Check if two line segments intersect
    ///
    /// Returns Some(point) if they intersect at a single point,
    /// None if they don't intersect or are collinear/overlapping
    pub fn intersects(&self, other: &LineSegment2D) -> Option<Point2D> {
        let p1 = self.start;
        let p2 = self.end;
        let p3 = other.start;
        let p4 = other.end;

        let d1 = Point2D::orientation(&p3, &p4, &p1);
        let d2 = Point2D::orientation(&p3, &p4, &p2);
        let d3 = Point2D::orientation(&p1, &p2, &p3);
        let d4 = Point2D::orientation(&p1, &p2, &p4);

        // General case: segments intersect if they straddle each other
        if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
           ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
            // Calculate intersection point
            let x1 = p1.x;
            let y1 = p1.y;
            let x2 = p2.x;
            let y2 = p2.y;
            let x3 = p3.x;
            let y3 = p3.y;
            let x4 = p4.x;
            let y4 = p4.y;

            let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
            if denom.abs() < 1e-10 {
                return None;
            }

            let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;

            let x = x1 + t * (x2 - x1);
            let y = y1 + t * (y2 - y1);

            return Some(Point2D::new(x, y));
        }

        // Special cases: collinear segments
        if d1.abs() < 1e-10 && self.contains_point(&p3) {
            return Some(p3);
        }
        if d2.abs() < 1e-10 && self.contains_point(&p4) {
            return Some(p4);
        }
        if d3.abs() < 1e-10 && other.contains_point(&p1) {
            return Some(p1);
        }
        if d4.abs() < 1e-10 && other.contains_point(&p2) {
            return Some(p2);
        }

        None
    }

    /// Calculate the distance from a point to this line segment
    pub fn distance_to_point(&self, p: &Point2D) -> f64 {
        let l2 = self.start.distance_squared(&self.end);

        if l2 < 1e-10 {
            // Segment is a point
            return self.start.distance(p);
        }

        // Project point onto line
        let t = ((p.x - self.start.x) * (self.end.x - self.start.x) +
                 (p.y - self.start.y) * (self.end.y - self.start.y)) / l2;

        if t < 0.0 {
            // Beyond start point
            self.start.distance(p)
        } else if t > 1.0 {
            // Beyond end point
            self.end.distance(p)
        } else {
            // Projection is on the segment
            let projection = Point2D::new(
                self.start.x + t * (self.end.x - self.start.x),
                self.start.y + t * (self.end.y - self.start.y),
            );
            p.distance(&projection)
        }
    }
}

/// An infinite line in 2D space defined by ax + by + c = 0
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line2D {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Line2D {
    /// Create a line from two points
    pub fn from_points(p1: &Point2D, p2: &Point2D) -> Self {
        let a = p2.y - p1.y;
        let b = p1.x - p2.x;
        let c = -(a * p1.x + b * p1.y);
        Line2D { a, b, c }
    }

    /// Create a line from slope-intercept form: y = mx + b
    pub fn from_slope_intercept(m: f64, b: f64) -> Self {
        // y = mx + b  =>  mx - y + b = 0
        Line2D { a: m, b: -1.0, c: b }
    }

    /// Check if a point lies on this line
    pub fn contains_point(&self, p: &Point2D) -> bool {
        (self.a * p.x + self.b * p.y + self.c).abs() < 1e-10
    }

    /// Calculate the distance from a point to this line
    pub fn distance_to_point(&self, p: &Point2D) -> f64 {
        let numerator = (self.a * p.x + self.b * p.y + self.c).abs();
        let denominator = (self.a * self.a + self.b * self.b).sqrt();
        numerator / denominator
    }

    /// Check if two lines are parallel
    pub fn is_parallel(&self, other: &Line2D) -> bool {
        // Lines are parallel if their normal vectors are parallel
        (self.a * other.b - self.b * other.a).abs() < 1e-10
    }

    /// Find the intersection point of two lines
    pub fn intersect(&self, other: &Line2D) -> Option<Point2D> {
        let denom = self.a * other.b - self.b * other.a;

        if denom.abs() < 1e-10 {
            // Lines are parallel
            return None;
        }

        let x = (self.b * other.c - self.c * other.b) / denom;
        let y = (self.c * other.a - self.a * other.c) / denom;

        Some(Point2D::new(x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_segment_length() {
        let seg = LineSegment2D::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 4.0),
        );
        assert!((seg.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_line_segment_intersection() {
        let seg1 = LineSegment2D::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 2.0),
        );
        let seg2 = LineSegment2D::new(
            Point2D::new(0.0, 2.0),
            Point2D::new(2.0, 0.0),
        );

        let intersection = seg1.intersects(&seg2);
        assert!(intersection.is_some());

        let point = intersection.unwrap();
        assert!((point.x - 1.0).abs() < 1e-10);
        assert!((point.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_line_segment_no_intersection() {
        let seg1 = LineSegment2D::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
        );
        let seg2 = LineSegment2D::new(
            Point2D::new(2.0, 2.0),
            Point2D::new(3.0, 3.0),
        );

        assert!(seg1.intersects(&seg2).is_none());
    }

    #[test]
    fn test_line2d_from_points() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(1.0, 1.0);
        let line = Line2D::from_points(&p1, &p2);

        // Line should pass through both points
        assert!(line.contains_point(&p1));
        assert!(line.contains_point(&p2));

        // And through the midpoint
        let mid = Point2D::new(0.5, 0.5);
        assert!(line.contains_point(&mid));
    }

    #[test]
    fn test_line_intersection() {
        // x-axis
        let line1 = Line2D { a: 0.0, b: 1.0, c: 0.0 }; // y = 0
        // y-axis
        let line2 = Line2D { a: 1.0, b: 0.0, c: 0.0 }; // x = 0

        let intersection = line1.intersect(&line2);
        assert!(intersection.is_some());

        let point = intersection.unwrap();
        assert!((point.x - 0.0).abs() < 1e-10);
        assert!((point.y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_point() {
        let seg = LineSegment2D::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
        );

        let p = Point2D::new(1.0, 1.0);
        let dist = seg.distance_to_point(&p);
        assert!((dist - 1.0).abs() < 1e-10);
    }
}
