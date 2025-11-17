//! Common types for plotting

/// A 2D point
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    /// Create a new 2D point
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Create a point at the origin
    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Calculate the distance to another point
    pub fn distance_to(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Convert to a tuple
    pub fn as_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

impl From<(f64, f64)> for Point2D {
    fn from((x, y): (f64, f64)) -> Self {
        Self::new(x, y)
    }
}

/// A 2D vector
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2D {
    pub x: f64,
    pub y: f64,
}

impl Vector2D {
    /// Create a new 2D vector
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Zero vector
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Calculate the magnitude (length) of the vector
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Normalize the vector to unit length
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag == 0.0 {
            Self::zero()
        } else {
            Self {
                x: self.x / mag,
                y: self.y / mag,
            }
        }
    }

    /// Dot product with another vector
    pub fn dot(&self, other: &Vector2D) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

impl From<(f64, f64)> for Vector2D {
    fn from((x, y): (f64, f64)) -> Self {
        Self::new(x, y)
    }
}

/// A 3D point
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    /// Create a new 3D point
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Create a point at the origin
    pub fn origin() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Calculate the distance to another point
    pub fn distance_to(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Convert to a tuple
    pub fn as_tuple(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
}

impl From<(f64, f64, f64)> for Point3D {
    fn from((x, y, z): (f64, f64, f64)) -> Self {
        Self::new(x, y, z)
    }
}

/// A 3D vector
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3D {
    /// Create a new 3D vector
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Zero vector
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Calculate the magnitude (length) of the vector
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize the vector to unit length
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag == 0.0 {
            Self::zero()
        } else {
            Self {
                x: self.x / mag,
                y: self.y / mag,
                z: self.z / mag,
            }
        }
    }

    /// Dot product with another vector
    pub fn dot(&self, other: &Vector3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product with another vector
    pub fn cross(&self, other: &Vector3D) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl From<(f64, f64, f64)> for Vector3D {
    fn from((x, y, z): (f64, f64, f64)) -> Self {
        Self::new(x, y, z)
    }
}

/// Output format for rendered graphics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderFormat {
    /// Scalable Vector Graphics
    SVG,
    /// Portable Network Graphics (raster)
    PNG,
    /// JPEG (raster, lossy)
    JPEG,
    /// Portable Document Format
    PDF,
    /// Encapsulated PostScript
    EPS,
}

impl RenderFormat {
    /// Get the file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            RenderFormat::SVG => "svg",
            RenderFormat::PNG => "png",
            RenderFormat::JPEG => "jpg",
            RenderFormat::PDF => "pdf",
            RenderFormat::EPS => "eps",
        }
    }

    /// Check if this is a vector format
    pub fn is_vector(&self) -> bool {
        matches!(self, RenderFormat::SVG | RenderFormat::PDF | RenderFormat::EPS)
    }

    /// Check if this is a raster format
    pub fn is_raster(&self) -> bool {
        !self.is_vector()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert_eq!(p1.distance_to(&p2), 5.0);
    }

    #[test]
    fn test_vector2d_normalize() {
        let v = Vector2D::new(3.0, 4.0);
        let normalized = v.normalize();
        assert!((normalized.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector2d_dot() {
        let v1 = Vector2D::new(1.0, 0.0);
        let v2 = Vector2D::new(0.0, 1.0);
        assert_eq!(v1.dot(&v2), 0.0); // Perpendicular vectors
    }

    #[test]
    fn test_point3d() {
        let p1 = Point3D::origin();
        let p2 = Point3D::new(1.0, 0.0, 0.0);
        assert_eq!(p1.distance_to(&p2), 1.0);
    }

    #[test]
    fn test_vector3d_cross() {
        let v1 = Vector3D::new(1.0, 0.0, 0.0);
        let v2 = Vector3D::new(0.0, 1.0, 0.0);
        let cross = v1.cross(&v2);
        assert_eq!(cross, Vector3D::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_render_format() {
        assert_eq!(RenderFormat::SVG.extension(), "svg");
        assert!(RenderFormat::SVG.is_vector());
        assert!(!RenderFormat::PNG.is_vector());
        assert!(RenderFormat::PNG.is_raster());
    }
}
