//! Basic 3D shape primitives
//!
//! This module provides basic 3D geometric shapes that can be rendered
//! as triangle meshes. All shapes implement the Graphics3dPrimitive trait.

use crate::base::{Graphics3dPrimitive, IndexFaceSet, BoundingBox3D, Point3D, Vector3D};
use crate::Result;
use rustmath_colors::Color;
use std::f64::consts::PI;

/// A sphere centered at a point with a given radius
#[derive(Debug, Clone)]
pub struct Sphere {
    /// Center of the sphere
    pub center: Point3D,
    /// Radius of the sphere
    pub radius: f64,
    /// Number of subdivisions in theta (latitude)
    pub u_resolution: usize,
    /// Number of subdivisions in phi (longitude)
    pub v_resolution: usize,
    /// Color of the sphere
    pub color: Option<Color>,
}

impl Sphere {
    /// Create a new sphere
    pub fn new(center: Point3D, radius: f64) -> Self {
        Self {
            center,
            radius,
            u_resolution: 32,
            v_resolution: 32,
            color: None,
        }
    }

    /// Set the resolution (number of subdivisions)
    pub fn with_resolution(mut self, u_res: usize, v_res: usize) -> Self {
        self.u_resolution = u_res;
        self.v_resolution = v_res;
        self
    }

    /// Set the color
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
}

impl Graphics3dPrimitive for Sphere {
    fn bounding_box(&self) -> BoundingBox3D {
        BoundingBox3D {
            xmin: self.center.x - self.radius,
            xmax: self.center.x + self.radius,
            ymin: self.center.y - self.radius,
            ymax: self.center.y + self.radius,
            zmin: self.center.z - self.radius,
            zmax: self.center.z + self.radius,
        }
    }

    fn clone_box(&self) -> std::boxed::Box<dyn Graphics3dPrimitive> {
        std::boxed::Box::new(self.clone())
    }

    fn to_mesh(&self) -> Result<IndexFaceSet> {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        // Generate vertices using spherical coordinates
        // theta (latitude): 0 to PI
        // phi (longitude): 0 to 2*PI
        for i in 0..=self.u_resolution {
            let theta = (i as f64 * PI) / self.u_resolution as f64;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for j in 0..=self.v_resolution {
                let phi = (j as f64 * 2.0 * PI) / self.v_resolution as f64;
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                let x = self.center.x + self.radius * sin_theta * cos_phi;
                let y = self.center.y + self.radius * sin_theta * sin_phi;
                let z = self.center.z + self.radius * cos_theta;

                vertices.push(Point3D { x, y, z });
            }
        }

        // Generate faces (two triangles per quad)
        for i in 0..self.u_resolution {
            for j in 0..self.v_resolution {
                let idx0 = i * (self.v_resolution + 1) + j;
                let idx1 = idx0 + 1;
                let idx2 = (i + 1) * (self.v_resolution + 1) + j;
                let idx3 = idx2 + 1;

                // First triangle
                faces.push([idx0, idx2, idx1]);
                // Second triangle
                faces.push([idx1, idx2, idx3]);
            }
        }

        let mut mesh = IndexFaceSet::new(vertices, faces);
        mesh.compute_normals();

        // Set uniform color if specified
        if let Some(color) = &self.color {
            mesh.vertex_colors = Some(vec![color.clone(); mesh.vertices.len()]);
        }

        Ok(mesh)
    }
}

/// An axis-aligned box (rectangular prism)
#[derive(Debug, Clone)]
pub struct BoxShape {
    /// Minimum corner of the box
    pub min_corner: Point3D,
    /// Maximum corner of the box
    pub max_corner: Point3D,
    /// Color of the box
    pub color: Option<Color>,
}

impl BoxShape {
    /// Create a new box from two corners
    pub fn new(min_corner: Point3D, max_corner: Point3D) -> Self {
        Self {
            min_corner,
            max_corner,
            color: None,
        }
    }

    /// Create a box centered at origin with given size
    pub fn centered(width: f64, height: f64, depth: f64) -> Self {
        let half_w = width / 2.0;
        let half_h = height / 2.0;
        let half_d = depth / 2.0;
        Self {
            min_corner: Point3D {
                x: -half_w,
                y: -half_h,
                z: -half_d,
            },
            max_corner: Point3D {
                x: half_w,
                y: half_h,
                z: half_d,
            },
            color: None,
        }
    }

    /// Set the color
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
}

impl Graphics3dPrimitive for BoxShape {
    fn bounding_box(&self) -> BoundingBox3D {
        BoundingBox3D {
            xmin: self.min_corner.x,
            xmax: self.max_corner.x,
            ymin: self.min_corner.y,
            ymax: self.max_corner.y,
            zmin: self.min_corner.z,
            zmax: self.max_corner.z,
        }
    }

    fn clone_box(&self) -> std::boxed::Box<dyn Graphics3dPrimitive> {
        std::boxed::Box::new(self.clone())
    }

    fn to_mesh(&self) -> Result<IndexFaceSet> {
        let min = &self.min_corner;
        let max = &self.max_corner;

        // 8 vertices of the box
        let vertices = vec![
            Point3D { x: min.x, y: min.y, z: min.z }, // 0
            Point3D { x: max.x, y: min.y, z: min.z }, // 1
            Point3D { x: max.x, y: max.y, z: min.z }, // 2
            Point3D { x: min.x, y: max.y, z: min.z }, // 3
            Point3D { x: min.x, y: min.y, z: max.z }, // 4
            Point3D { x: max.x, y: min.y, z: max.z }, // 5
            Point3D { x: max.x, y: max.y, z: max.z }, // 6
            Point3D { x: min.x, y: max.y, z: max.z }, // 7
        ];

        // 12 triangles (2 per face, 6 faces)
        let faces = vec![
            // Bottom face (z = min.z)
            [0, 1, 2], [0, 2, 3],
            // Top face (z = max.z)
            [4, 7, 6], [4, 6, 5],
            // Front face (y = min.y)
            [0, 4, 5], [0, 5, 1],
            // Back face (y = max.y)
            [3, 2, 6], [3, 6, 7],
            // Left face (x = min.x)
            [0, 3, 7], [0, 7, 4],
            // Right face (x = max.x)
            [1, 5, 6], [1, 6, 2],
        ];

        let mut mesh = IndexFaceSet::new(vertices, faces);
        mesh.compute_normals();

        // Set uniform color if specified
        if let Some(color) = &self.color {
            mesh.vertex_colors = Some(vec![color.clone(); mesh.vertices.len()]);
        }

        Ok(mesh)
    }
}

/// A cylinder with a circular base
#[derive(Debug, Clone)]
pub struct Cylinder {
    /// Base center point
    pub base: Point3D,
    /// Height (along z-axis from base)
    pub height: f64,
    /// Radius of the circular base
    pub radius: f64,
    /// Number of subdivisions around the circle
    pub resolution: usize,
    /// Whether to cap the top and bottom
    pub capped: bool,
    /// Color of the cylinder
    pub color: Option<Color>,
}

impl Cylinder {
    /// Create a new cylinder
    pub fn new(base: Point3D, height: f64, radius: f64) -> Self {
        Self {
            base,
            height,
            radius,
            resolution: 32,
            capped: true,
            color: None,
        }
    }

    /// Set the resolution (number of subdivisions)
    pub fn with_resolution(mut self, resolution: usize) -> Self {
        self.resolution = resolution;
        self
    }

    /// Set whether to cap the ends
    pub fn with_caps(mut self, capped: bool) -> Self {
        self.capped = capped;
        self
    }

    /// Set the color
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
}

impl Graphics3dPrimitive for Cylinder {
    fn bounding_box(&self) -> BoundingBox3D {
        BoundingBox3D {
            xmin: self.base.x - self.radius,
            xmax: self.base.x + self.radius,
            ymin: self.base.y - self.radius,
            ymax: self.base.y + self.radius,
            zmin: self.base.z,
            zmax: self.base.z + self.height,
        }
    }

    fn clone_box(&self) -> std::boxed::Box<dyn Graphics3dPrimitive> {
        std::boxed::Box::new(self.clone())
    }

    fn to_mesh(&self) -> Result<IndexFaceSet> {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        // Generate vertices for the side surface
        for i in 0..=1 {
            let z = self.base.z + i as f64 * self.height;
            for j in 0..=self.resolution {
                let angle = (j as f64 * 2.0 * PI) / self.resolution as f64;
                let x = self.base.x + self.radius * angle.cos();
                let y = self.base.y + self.radius * angle.sin();
                vertices.push(Point3D { x, y, z });
            }
        }

        // Generate side faces
        for j in 0..self.resolution {
            let idx0 = j;
            let idx1 = j + 1;
            let idx2 = j + self.resolution + 1;
            let idx3 = j + self.resolution + 2;

            faces.push([idx0, idx2, idx1]);
            faces.push([idx1, idx2, idx3]);
        }

        // Add caps if requested
        if self.capped {
            // Bottom cap center
            let bottom_center_idx = vertices.len();
            vertices.push(self.base);

            // Top cap center
            let top_center_idx = vertices.len();
            vertices.push(Point3D {
                x: self.base.x,
                y: self.base.y,
                z: self.base.z + self.height,
            });

            // Bottom cap triangles
            for j in 0..self.resolution {
                faces.push([bottom_center_idx, j + 1, j]);
            }

            // Top cap triangles
            let top_offset = self.resolution + 1;
            for j in 0..self.resolution {
                faces.push([top_center_idx, top_offset + j, top_offset + j + 1]);
            }
        }

        let mut mesh = IndexFaceSet::new(vertices, faces);
        mesh.compute_normals();

        // Set uniform color if specified
        if let Some(color) = &self.color {
            mesh.vertex_colors = Some(vec![color.clone(); mesh.vertices.len()]);
        }

        Ok(mesh)
    }
}

/// A cone with a circular base
#[derive(Debug, Clone)]
pub struct Cone {
    /// Base center point
    pub base: Point3D,
    /// Height (along z-axis from base)
    pub height: f64,
    /// Radius of the circular base
    pub radius: f64,
    /// Number of subdivisions around the circle
    pub resolution: usize,
    /// Whether to cap the bottom
    pub capped: bool,
    /// Color of the cone
    pub color: Option<Color>,
}

impl Cone {
    /// Create a new cone
    pub fn new(base: Point3D, height: f64, radius: f64) -> Self {
        Self {
            base,
            height,
            radius,
            resolution: 32,
            capped: true,
            color: None,
        }
    }

    /// Set the resolution (number of subdivisions)
    pub fn with_resolution(mut self, resolution: usize) -> Self {
        self.resolution = resolution;
        self
    }

    /// Set whether to cap the bottom
    pub fn with_cap(mut self, capped: bool) -> Self {
        self.capped = capped;
        self
    }

    /// Set the color
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
}

impl Graphics3dPrimitive for Cone {
    fn bounding_box(&self) -> BoundingBox3D {
        BoundingBox3D {
            xmin: self.base.x - self.radius,
            xmax: self.base.x + self.radius,
            ymin: self.base.y - self.radius,
            ymax: self.base.y + self.radius,
            zmin: self.base.z,
            zmax: self.base.z + self.height,
        }
    }

    fn clone_box(&self) -> std::boxed::Box<dyn Graphics3dPrimitive> {
        std::boxed::Box::new(self.clone())
    }

    fn to_mesh(&self) -> Result<IndexFaceSet> {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        // Apex of the cone
        let apex_idx = vertices.len();
        vertices.push(Point3D {
            x: self.base.x,
            y: self.base.y,
            z: self.base.z + self.height,
        });

        // Base circle vertices
        for j in 0..=self.resolution {
            let angle = (j as f64 * 2.0 * PI) / self.resolution as f64;
            let x = self.base.x + self.radius * angle.cos();
            let y = self.base.y + self.radius * angle.sin();
            vertices.push(Point3D { x, y, z: self.base.z });
        }

        // Side faces (triangles from apex to base)
        for j in 0..self.resolution {
            let idx0 = j + 1;
            let idx1 = j + 2;
            faces.push([apex_idx, idx0, idx1]);
        }

        // Add bottom cap if requested
        if self.capped {
            let center_idx = vertices.len();
            vertices.push(self.base);

            for j in 0..self.resolution {
                faces.push([center_idx, j + 2, j + 1]);
            }
        }

        let mut mesh = IndexFaceSet::new(vertices, faces);
        mesh.compute_normals();

        // Set uniform color if specified
        if let Some(color) = &self.color {
            mesh.vertex_colors = Some(vec![color.clone(); mesh.vertices.len()]);
        }

        Ok(mesh)
    }
}

/// A torus (donut shape)
#[derive(Debug, Clone)]
pub struct Torus {
    /// Center of the torus
    pub center: Point3D,
    /// Major radius (from center to tube center)
    pub major_radius: f64,
    /// Minor radius (tube radius)
    pub minor_radius: f64,
    /// Number of subdivisions around the major circle
    pub u_resolution: usize,
    /// Number of subdivisions around the minor circle (tube)
    pub v_resolution: usize,
    /// Color of the torus
    pub color: Option<Color>,
}

impl Torus {
    /// Create a new torus
    pub fn new(center: Point3D, major_radius: f64, minor_radius: f64) -> Self {
        Self {
            center,
            major_radius,
            minor_radius,
            u_resolution: 32,
            v_resolution: 16,
            color: None,
        }
    }

    /// Set the resolution (number of subdivisions)
    pub fn with_resolution(mut self, u_res: usize, v_res: usize) -> Self {
        self.u_resolution = u_res;
        self.v_resolution = v_res;
        self
    }

    /// Set the color
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
}

impl Graphics3dPrimitive for Torus {
    fn bounding_box(&self) -> BoundingBox3D {
        let outer_radius = self.major_radius + self.minor_radius;
        BoundingBox3D {
            xmin: self.center.x - outer_radius,
            xmax: self.center.x + outer_radius,
            ymin: self.center.y - outer_radius,
            ymax: self.center.y + outer_radius,
            zmin: self.center.z - self.minor_radius,
            zmax: self.center.z + self.minor_radius,
        }
    }

    fn clone_box(&self) -> std::boxed::Box<dyn Graphics3dPrimitive> {
        std::boxed::Box::new(self.clone())
    }

    fn to_mesh(&self) -> Result<IndexFaceSet> {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        // Generate vertices using torus parametric equations
        // u: angle around the major circle (0 to 2*PI)
        // v: angle around the minor circle (0 to 2*PI)
        for i in 0..=self.u_resolution {
            let u = (i as f64 * 2.0 * PI) / self.u_resolution as f64;
            let cos_u = u.cos();
            let sin_u = u.sin();

            for j in 0..=self.v_resolution {
                let v = (j as f64 * 2.0 * PI) / self.v_resolution as f64;
                let cos_v = v.cos();
                let sin_v = v.sin();

                let x = self.center.x + (self.major_radius + self.minor_radius * cos_v) * cos_u;
                let y = self.center.y + (self.major_radius + self.minor_radius * cos_v) * sin_u;
                let z = self.center.z + self.minor_radius * sin_v;

                vertices.push(Point3D { x, y, z });
            }
        }

        // Generate faces
        for i in 0..self.u_resolution {
            for j in 0..self.v_resolution {
                let idx0 = i * (self.v_resolution + 1) + j;
                let idx1 = idx0 + 1;
                let idx2 = (i + 1) * (self.v_resolution + 1) + j;
                let idx3 = idx2 + 1;

                faces.push([idx0, idx2, idx1]);
                faces.push([idx1, idx2, idx3]);
            }
        }

        let mut mesh = IndexFaceSet::new(vertices, faces);
        mesh.compute_normals();

        // Set uniform color if specified
        if let Some(color) = &self.color {
            mesh.vertex_colors = Some(vec![color.clone(); mesh.vertices.len()]);
        }

        Ok(mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_creation() {
        let sphere = Sphere::new(Point3D { x: 0.0, y: 0.0, z: 0.0 }, 1.0);
        let bbox = sphere.bounding_box();
        assert_eq!(bbox.xmin, -1.0);
        assert_eq!(bbox.xmax, 1.0);
        assert_eq!(bbox.ymin, -1.0);
        assert_eq!(bbox.ymax, 1.0);
        assert_eq!(bbox.zmin, -1.0);
        assert_eq!(bbox.zmax, 1.0);
    }

    #[test]
    fn test_sphere_mesh() {
        let sphere = Sphere::new(Point3D { x: 0.0, y: 0.0, z: 0.0 }, 1.0)
            .with_resolution(8, 8);
        let mesh = sphere.to_mesh().unwrap();
        assert_eq!(mesh.vertices.len(), 81); // (8+1) * (8+1)
        assert_eq!(mesh.faces.len(), 128); // 8 * 8 * 2
    }

    #[test]
    fn test_box_creation() {
        let b = BoxShape::centered(2.0, 2.0, 2.0);
        let bbox = b.bounding_box();
        assert_eq!(bbox.xmin, -1.0);
        assert_eq!(bbox.xmax, 1.0);
    }

    #[test]
    fn test_box_mesh() {
        let b = BoxShape::centered(2.0, 2.0, 2.0);
        let mesh = b.to_mesh().unwrap();
        assert_eq!(mesh.vertices.len(), 8);
        assert_eq!(mesh.faces.len(), 12); // 2 triangles per face, 6 faces
    }

    #[test]
    fn test_cylinder_creation() {
        let cyl = Cylinder::new(Point3D { x: 0.0, y: 0.0, z: 0.0 }, 2.0, 1.0);
        let bbox = cyl.bounding_box();
        assert_eq!(bbox.zmin, 0.0);
        assert_eq!(bbox.zmax, 2.0);
    }

    #[test]
    fn test_cylinder_mesh() {
        let cyl = Cylinder::new(Point3D { x: 0.0, y: 0.0, z: 0.0 }, 2.0, 1.0)
            .with_resolution(8);
        let mesh = cyl.to_mesh().unwrap();
        // 2 rings of 9 vertices + 2 cap centers = 20 vertices
        assert_eq!(mesh.vertices.len(), 20);
    }

    #[test]
    fn test_cone_creation() {
        let cone = Cone::new(Point3D { x: 0.0, y: 0.0, z: 0.0 }, 2.0, 1.0);
        let bbox = cone.bounding_box();
        assert_eq!(bbox.zmin, 0.0);
        assert_eq!(bbox.zmax, 2.0);
    }

    #[test]
    fn test_cone_mesh() {
        let cone = Cone::new(Point3D { x: 0.0, y: 0.0, z: 0.0 }, 2.0, 1.0)
            .with_resolution(8);
        let mesh = cone.to_mesh().unwrap();
        // 1 apex + 9 base vertices + 1 cap center = 11 vertices
        assert_eq!(mesh.vertices.len(), 11);
    }

    #[test]
    fn test_torus_creation() {
        let torus = Torus::new(Point3D { x: 0.0, y: 0.0, z: 0.0 }, 2.0, 0.5);
        let bbox = torus.bounding_box();
        assert_eq!(bbox.xmin, -2.5);
        assert_eq!(bbox.xmax, 2.5);
        assert_eq!(bbox.zmin, -0.5);
        assert_eq!(bbox.zmax, 0.5);
    }

    #[test]
    fn test_torus_mesh() {
        let torus = Torus::new(Point3D { x: 0.0, y: 0.0, z: 0.0 }, 2.0, 0.5)
            .with_resolution(8, 4);
        let mesh = torus.to_mesh().unwrap();
        assert_eq!(mesh.vertices.len(), 45); // (8+1) * (4+1)
        assert_eq!(mesh.faces.len(), 64); // 8 * 4 * 2
    }
}
