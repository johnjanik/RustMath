//! Bounding box for graphics primitives

use crate::{PlotError, Point2D, Point3D, Result};

/// A 2D axis-aligned bounding box
///
/// Represents a rectangular region defined by minimum and maximum x and y coordinates.
/// Based on SageMath's bounding box handling in sage.plot.graphics
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
}

impl BoundingBox {
    /// Create a new bounding box
    ///
    /// # Arguments
    /// * `xmin` - Minimum x coordinate
    /// * `xmax` - Maximum x coordinate
    /// * `ymin` - Minimum y coordinate
    /// * `ymax` - Maximum y coordinate
    ///
    /// # Panics
    /// Panics if xmin > xmax or ymin > ymax
    pub fn new(xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Self {
        assert!(
            xmin <= xmax,
            "xmin must be <= xmax, got xmin={}, xmax={}",
            xmin,
            xmax
        );
        assert!(
            ymin <= ymax,
            "ymin must be <= ymax, got ymin={}, ymax={}",
            ymin,
            ymax
        );

        Self {
            xmin,
            xmax,
            ymin,
            ymax,
        }
    }

    /// Create a bounding box from a list of points
    pub fn from_points(points: &[Point2D]) -> Result<Self> {
        if points.is_empty() {
            return Err(PlotError::InvalidBoundingBox(
                "Cannot create bounding box from empty point list".to_string(),
            ));
        }

        let mut xmin = points[0].x;
        let mut xmax = points[0].x;
        let mut ymin = points[0].y;
        let mut ymax = points[0].y;

        for point in points.iter().skip(1) {
            xmin = xmin.min(point.x);
            xmax = xmax.max(point.x);
            ymin = ymin.min(point.y);
            ymax = ymax.max(point.y);
        }

        Ok(Self {
            xmin,
            xmax,
            ymin,
            ymax,
        })
    }

    /// Create an empty bounding box at the origin
    pub fn empty() -> Self {
        Self {
            xmin: 0.0,
            xmax: 0.0,
            ymin: 0.0,
            ymax: 0.0,
        }
    }

    /// Create an infinite bounding box
    pub fn infinite() -> Self {
        Self {
            xmin: f64::NEG_INFINITY,
            xmax: f64::INFINITY,
            ymin: f64::NEG_INFINITY,
            ymax: f64::INFINITY,
        }
    }

    /// Get the width of the bounding box
    pub fn width(&self) -> f64 {
        self.xmax - self.xmin
    }

    /// Get the height of the bounding box
    pub fn height(&self) -> f64 {
        self.ymax - self.ymin
    }

    /// Get the center point of the bounding box
    pub fn center(&self) -> Point2D {
        Point2D::new(
            (self.xmin + self.xmax) / 2.0,
            (self.ymin + self.ymax) / 2.0,
        )
    }

    /// Get the area of the bounding box
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Check if a point is inside the bounding box
    pub fn contains(&self, point: &Point2D) -> bool {
        point.x >= self.xmin
            && point.x <= self.xmax
            && point.y >= self.ymin
            && point.y <= self.ymax
    }

    /// Expand the bounding box to include another bounding box
    pub fn union(&self, other: &BoundingBox) -> Self {
        Self {
            xmin: self.xmin.min(other.xmin),
            xmax: self.xmax.max(other.xmax),
            ymin: self.ymin.min(other.ymin),
            ymax: self.ymax.max(other.ymax),
        }
    }

    /// Expand the bounding box to include a point
    pub fn expand_to_include(&self, point: &Point2D) -> Self {
        Self {
            xmin: self.xmin.min(point.x),
            xmax: self.xmax.max(point.x),
            ymin: self.ymin.min(point.y),
            ymax: self.ymax.max(point.y),
        }
    }

    /// Expand the bounding box by a margin
    ///
    /// # Arguments
    /// * `margin` - Margin to add on all sides (can be negative to shrink)
    pub fn expand(&self, margin: f64) -> Self {
        Self {
            xmin: self.xmin - margin,
            xmax: self.xmax + margin,
            ymin: self.ymin - margin,
            ymax: self.ymax + margin,
        }
    }

    /// Check if the bounding box intersects with another
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        !(self.xmax < other.xmin
            || self.xmin > other.xmax
            || self.ymax < other.ymin
            || self.ymin > other.ymax)
    }

    /// Get the intersection of two bounding boxes
    ///
    /// Returns None if they don't intersect
    pub fn intersection(&self, other: &BoundingBox) -> Option<Self> {
        if !self.intersects(other) {
            return None;
        }

        Some(Self {
            xmin: self.xmin.max(other.xmin),
            xmax: self.xmax.min(other.xmax),
            ymin: self.ymin.max(other.ymin),
            ymax: self.ymax.min(other.ymax),
        })
    }

    /// Check if the bounding box is valid (not containing NaN or infinite values)
    pub fn is_valid(&self) -> bool {
        self.xmin.is_finite()
            && self.xmax.is_finite()
            && self.ymin.is_finite()
            && self.ymax.is_finite()
            && self.xmin <= self.xmax
            && self.ymin <= self.ymax
    }

    /// Get the aspect ratio (width / height)
    pub fn aspect_ratio(&self) -> f64 {
        let h = self.height();
        if h == 0.0 {
            f64::INFINITY
        } else {
            self.width() / h
        }
    }
}

/// A 3D axis-aligned bounding box
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox3D {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
    pub zmin: f64,
    pub zmax: f64,
}

impl BoundingBox3D {
    /// Create a new 3D bounding box
    pub fn new(xmin: f64, xmax: f64, ymin: f64, ymax: f64, zmin: f64, zmax: f64) -> Self {
        assert!(xmin <= xmax);
        assert!(ymin <= ymax);
        assert!(zmin <= zmax);

        Self {
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        }
    }

    /// Create a bounding box from a list of 3D points
    pub fn from_points(points: &[Point3D]) -> Result<Self> {
        if points.is_empty() {
            return Err(PlotError::InvalidBoundingBox(
                "Cannot create bounding box from empty point list".to_string(),
            ));
        }

        let mut xmin = points[0].x;
        let mut xmax = points[0].x;
        let mut ymin = points[0].y;
        let mut ymax = points[0].y;
        let mut zmin = points[0].z;
        let mut zmax = points[0].z;

        for point in points.iter().skip(1) {
            xmin = xmin.min(point.x);
            xmax = xmax.max(point.x);
            ymin = ymin.min(point.y);
            ymax = ymax.max(point.y);
            zmin = zmin.min(point.z);
            zmax = zmax.max(point.z);
        }

        Ok(Self {
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        })
    }

    /// Get the center point of the bounding box
    pub fn center(&self) -> Point3D {
        Point3D::new(
            (self.xmin + self.xmax) / 2.0,
            (self.ymin + self.ymax) / 2.0,
            (self.zmin + self.zmax) / 2.0,
        )
    }

    /// Get the volume of the bounding box
    pub fn volume(&self) -> f64 {
        (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)
    }

    /// Check if a point is inside the bounding box
    pub fn contains(&self, point: &Point3D) -> bool {
        point.x >= self.xmin
            && point.x <= self.xmax
            && point.y >= self.ymin
            && point.y <= self.ymax
            && point.z >= self.zmin
            && point.z <= self.zmax
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_creation() {
        let bbox = BoundingBox::new(0.0, 10.0, 0.0, 5.0);
        assert_eq!(bbox.width(), 10.0);
        assert_eq!(bbox.height(), 5.0);
        assert_eq!(bbox.area(), 50.0);
    }

    #[test]
    fn test_bbox_from_points() {
        let points = vec![
            Point2D::new(1.0, 2.0),
            Point2D::new(5.0, 8.0),
            Point2D::new(3.0, 4.0),
        ];
        let bbox = BoundingBox::from_points(&points).unwrap();
        assert_eq!(bbox.xmin, 1.0);
        assert_eq!(bbox.xmax, 5.0);
        assert_eq!(bbox.ymin, 2.0);
        assert_eq!(bbox.ymax, 8.0);
    }

    #[test]
    fn test_bbox_center() {
        let bbox = BoundingBox::new(0.0, 10.0, 0.0, 10.0);
        let center = bbox.center();
        assert_eq!(center, Point2D::new(5.0, 5.0));
    }

    #[test]
    fn test_bbox_contains() {
        let bbox = BoundingBox::new(0.0, 10.0, 0.0, 10.0);
        assert!(bbox.contains(&Point2D::new(5.0, 5.0)));
        assert!(!bbox.contains(&Point2D::new(15.0, 5.0)));
    }

    #[test]
    fn test_bbox_union() {
        let bbox1 = BoundingBox::new(0.0, 5.0, 0.0, 5.0);
        let bbox2 = BoundingBox::new(3.0, 10.0, 3.0, 10.0);
        let union = bbox1.union(&bbox2);
        assert_eq!(union.xmin, 0.0);
        assert_eq!(union.xmax, 10.0);
        assert_eq!(union.ymin, 0.0);
        assert_eq!(union.ymax, 10.0);
    }

    #[test]
    fn test_bbox_expand() {
        let bbox = BoundingBox::new(5.0, 10.0, 5.0, 10.0);
        let expanded = bbox.expand(1.0);
        assert_eq!(expanded.xmin, 4.0);
        assert_eq!(expanded.xmax, 11.0);
    }

    #[test]
    fn test_bbox_intersects() {
        let bbox1 = BoundingBox::new(0.0, 5.0, 0.0, 5.0);
        let bbox2 = BoundingBox::new(3.0, 8.0, 3.0, 8.0);
        let bbox3 = BoundingBox::new(10.0, 15.0, 10.0, 15.0);

        assert!(bbox1.intersects(&bbox2));
        assert!(!bbox1.intersects(&bbox3));
    }

    #[test]
    fn test_bbox_intersection() {
        let bbox1 = BoundingBox::new(0.0, 5.0, 0.0, 5.0);
        let bbox2 = BoundingBox::new(3.0, 8.0, 3.0, 8.0);
        let intersection = bbox1.intersection(&bbox2).unwrap();
        assert_eq!(intersection.xmin, 3.0);
        assert_eq!(intersection.xmax, 5.0);
        assert_eq!(intersection.ymin, 3.0);
        assert_eq!(intersection.ymax, 5.0);
    }

    #[test]
    fn test_bbox3d() {
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(10.0, 10.0, 10.0),
        ];
        let bbox = BoundingBox3D::from_points(&points).unwrap();
        assert_eq!(bbox.volume(), 1000.0);
        assert_eq!(bbox.center(), Point3D::new(5.0, 5.0, 5.0));
    }

    #[test]
    #[should_panic]
    fn test_bbox_invalid() {
        BoundingBox::new(10.0, 0.0, 0.0, 10.0); // xmin > xmax
    }
}
