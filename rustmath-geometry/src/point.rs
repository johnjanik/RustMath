//! 2D and 3D point structures

use std::ops::{Add, Sub};

/// A point in 2D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    /// Create a new 2D point
    pub fn new(x: f64, y: f64) -> Self {
        Point2D { x, y }
    }

    /// Calculate the distance to another point
    pub fn distance(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate the squared distance (avoids sqrt for comparisons)
    pub fn distance_squared(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Calculate the cross product with another point (treating as vectors from origin)
    /// Returns the z-component of the 3D cross product
    pub fn cross(&self, other: &Point2D) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Calculate the dot product with another point (treating as vectors from origin)
    pub fn dot(&self, other: &Point2D) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Calculate the orientation of three points (p, q, r)
    /// Returns:
    /// - Positive if counterclockwise
    /// - Negative if clockwise
    /// - Zero if collinear
    pub fn orientation(p: &Point2D, q: &Point2D, r: &Point2D) -> f64 {
        (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)
    }

    /// Check if three points are collinear
    pub fn collinear(p: &Point2D, q: &Point2D, r: &Point2D) -> bool {
        Self::orientation(p, q, r).abs() < 1e-10
    }

    /// Calculate the angle to another point from this point (in radians)
    pub fn angle_to(&self, other: &Point2D) -> f64 {
        (other.y - self.y).atan2(other.x - self.x)
    }
}

impl Add for Point2D {
    type Output = Point2D;

    fn add(self, other: Point2D) -> Point2D {
        Point2D::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub for Point2D {
    type Output = Point2D;

    fn sub(self, other: Point2D) -> Point2D {
        Point2D::new(self.x - other.x, self.y - other.y)
    }
}

/// A point in 3D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    /// Create a new 3D point
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point3D { x, y, z }
    }

    /// Calculate the distance to another point
    pub fn distance(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate the dot product
    pub fn dot(&self, other: &Point3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Calculate the cross product
    pub fn cross(&self, other: &Point3D) -> Point3D {
        Point3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Calculate the magnitude (length) of the vector from origin
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize the vector (returns unit vector in same direction)
    pub fn normalize(&self) -> Point3D {
        let mag = self.magnitude();
        if mag < 1e-10 {
            return *self;
        }
        Point3D::new(self.x / mag, self.y / mag, self.z / mag)
    }
}

impl Add for Point3D {
    type Output = Point3D;

    fn add(self, other: Point3D) -> Point3D {
        Point3D::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Point3D {
    type Output = Point3D;

    fn sub(self, other: Point3D) -> Point3D {
        Point3D::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point2d_orientation() {
        let p = Point2D::new(0.0, 0.0);
        let q = Point2D::new(1.0, 0.0);
        let r = Point2D::new(1.0, 1.0);

        // Counterclockwise
        assert!(Point2D::orientation(&p, &q, &r) > 0.0);

        // Clockwise
        let r2 = Point2D::new(1.0, -1.0);
        assert!(Point2D::orientation(&p, &q, &r2) < 0.0);

        // Collinear
        let r3 = Point2D::new(2.0, 0.0);
        assert!(Point2D::collinear(&p, &q, &r3));
    }

    #[test]
    fn test_point3d_cross() {
        let p1 = Point3D::new(1.0, 0.0, 0.0);
        let p2 = Point3D::new(0.0, 1.0, 0.0);
        let cross = p1.cross(&p2);

        assert!((cross.x - 0.0).abs() < 1e-10);
        assert!((cross.y - 0.0).abs() < 1e-10);
        assert!((cross.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_normalize() {
        let p = Point3D::new(3.0, 4.0, 0.0);
        let normalized = p.normalize();

        assert!((normalized.x - 0.6).abs() < 1e-10);
        assert!((normalized.y - 0.8).abs() < 1e-10);
        assert!((normalized.magnitude() - 1.0).abs() < 1e-10);
    }
}
