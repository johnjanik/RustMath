//! Core 3D plotting types and traits

use rustmath_colors::Color;
use std::fmt;

/// A 3D point
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn distance_to(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
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
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn unit_x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    pub fn unit_y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    pub fn unit_z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag == 0.0 {
            *self
        } else {
            Self::new(self.x / mag, self.y / mag, self.z / mag)
        }
    }

    pub fn dot(&self, other: &Vector3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vector3D) -> Vector3D {
        Vector3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn scale(&self, scalar: f64) -> Vector3D {
        Vector3D::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl std::ops::Add for Vector3D {
    type Output = Vector3D;

    fn add(self, other: Vector3D) -> Vector3D {
        Vector3D::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::Sub for Vector3D {
    type Output = Vector3D;

    fn sub(self, other: Vector3D) -> Vector3D {
        Vector3D::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl std::ops::Neg for Vector3D {
    type Output = Vector3D;

    fn neg(self) -> Vector3D {
        Vector3D::new(-self.x, -self.y, -self.z)
    }
}

/// 3D bounding box
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
    pub fn new(xmin: f64, xmax: f64, ymin: f64, ymax: f64, zmin: f64, zmax: f64) -> Self {
        Self { xmin, xmax, ymin, ymax, zmin, zmax }
    }

    pub fn empty() -> Self {
        Self {
            xmin: f64::INFINITY,
            xmax: f64::NEG_INFINITY,
            ymin: f64::INFINITY,
            ymax: f64::NEG_INFINITY,
            zmin: f64::INFINITY,
            zmax: f64::NEG_INFINITY,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.xmin > self.xmax || self.ymin > self.ymax || self.zmin > self.zmax
    }

    pub fn expand_to_include(&mut self, point: &Point3D) {
        self.xmin = self.xmin.min(point.x);
        self.xmax = self.xmax.max(point.x);
        self.ymin = self.ymin.min(point.y);
        self.ymax = self.ymax.max(point.y);
        self.zmin = self.zmin.min(point.z);
        self.zmax = self.zmax.max(point.z);
    }

    pub fn merge(&mut self, other: &BoundingBox3D) {
        if !other.is_empty() {
            self.xmin = self.xmin.min(other.xmin);
            self.xmax = self.xmax.max(other.xmax);
            self.ymin = self.ymin.min(other.ymin);
            self.ymax = self.ymax.max(other.ymax);
            self.zmin = self.zmin.min(other.zmin);
            self.zmax = self.zmax.max(other.zmax);
        }
    }

    pub fn center(&self) -> Point3D {
        Point3D::new(
            (self.xmin + self.xmax) / 2.0,
            (self.ymin + self.ymax) / 2.0,
            (self.zmin + self.zmax) / 2.0,
        )
    }

    pub fn diagonal(&self) -> f64 {
        let dx = self.xmax - self.xmin;
        let dy = self.ymax - self.ymin;
        let dz = self.zmax - self.zmin;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Triangle mesh representation using indexed faces
///
/// This is the fundamental mesh representation used by all 3D primitives.
/// Vertices are stored once and referenced by index in the face list.
#[derive(Debug, Clone)]
pub struct IndexFaceSet {
    /// Vertex positions
    pub vertices: Vec<Point3D>,

    /// Triangle faces (indices into vertices array)
    pub faces: Vec<[usize; 3]>,

    /// Vertex normals (optional, same length as vertices if present)
    pub normals: Option<Vec<Vector3D>>,

    /// Vertex colors (optional, same length as vertices if present)
    pub vertex_colors: Option<Vec<Color>>,

    /// Face colors (optional, same length as faces if present)
    pub face_colors: Option<Vec<Color>>,
}

impl IndexFaceSet {
    /// Create a new empty mesh
    pub fn new(vertices: Vec<Point3D>, faces: Vec<[usize; 3]>) -> Self {
        Self {
            vertices,
            faces,
            normals: None,
            vertex_colors: None,
            face_colors: None,
        }
    }

    /// Create an empty mesh
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
            normals: None,
            vertex_colors: None,
            face_colors: None,
        }
    }

    /// Create a mesh with pre-allocated capacity
    pub fn with_capacity(num_vertices: usize, num_faces: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(num_vertices),
            faces: Vec::with_capacity(num_faces),
            normals: None,
            vertex_colors: None,
            face_colors: None,
        }
    }

    /// Add a vertex and return its index
    pub fn add_vertex(&mut self, vertex: Point3D) -> usize {
        let index = self.vertices.len();
        self.vertices.push(vertex);
        index
    }

    /// Add a triangular face
    pub fn add_face(&mut self, v0: usize, v1: usize, v2: usize) {
        self.faces.push([v0, v1, v2]);
    }

    /// Compute face normals (averaged to vertices)
    pub fn compute_normals(&mut self) {
        if self.vertices.is_empty() {
            return;
        }

        let mut normals = vec![Vector3D::zero(); self.vertices.len()];
        let mut counts = vec![0; self.vertices.len()];

        // Accumulate face normals
        for face in &self.faces {
            let v0 = &self.vertices[face[0]];
            let v1 = &self.vertices[face[1]];
            let v2 = &self.vertices[face[2]];

            // Compute face normal via cross product
            let edge1 = Vector3D::new(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            let edge2 = Vector3D::new(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
            let normal = edge1.cross(&edge2);

            // Add to vertex normals
            for &idx in face.iter() {
                normals[idx] = normals[idx] + normal;
                counts[idx] += 1;
            }
        }

        // Average and normalize
        for i in 0..normals.len() {
            if counts[i] > 0 {
                normals[i] = normals[i].scale(1.0 / counts[i] as f64).normalize();
            }
        }

        self.normals = Some(normals);
    }

    /// Compute the bounding box of the mesh
    pub fn bounding_box(&self) -> BoundingBox3D {
        let mut bbox = BoundingBox3D::empty();
        for vertex in &self.vertices {
            bbox.expand_to_include(vertex);
        }
        bbox
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of faces
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }
}

impl Default for IndexFaceSet {
    fn default() -> Self {
        Self::empty()
    }
}

// Implement Graphics3dPrimitive for IndexFaceSet
impl Graphics3dPrimitive for IndexFaceSet {
    fn bounding_box(&self) -> BoundingBox3D {
        self.bounding_box()
    }

    fn to_mesh(&self) -> crate::Result<IndexFaceSet> {
        Ok(self.clone())
    }

    fn clone_box(&self) -> Box<dyn Graphics3dPrimitive> {
        Box::new(self.clone())
    }
}

/// Trait for 3D graphics primitives
///
/// All 3D objects must implement this trait to be renderable.
pub trait Graphics3dPrimitive: fmt::Debug {
    /// Get the bounding box of this primitive
    fn bounding_box(&self) -> BoundingBox3D;

    /// Convert this primitive to a triangle mesh
    fn to_mesh(&self) -> crate::Result<IndexFaceSet>;

    /// Clone the primitive into a boxed trait object
    fn clone_box(&self) -> Box<dyn Graphics3dPrimitive>;
}

// Implement Clone for Box<dyn Graphics3dPrimitive>
impl Clone for Box<dyn Graphics3dPrimitive> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Options for 3D graphics rendering
#[derive(Debug, Clone)]
pub struct Graphics3dOptions {
    /// Background color
    pub background_color: Color,

    /// Default color for objects (if not specified)
    pub default_color: Option<Color>,

    /// Aspect ratio (width/height), or None for automatic
    pub aspect_ratio: Option<f64>,

    /// Enable lighting
    pub lighting: bool,

    /// Enable wireframe mode
    pub wireframe: bool,

    /// Line width for wireframe
    pub line_width: f64,

    /// Enable smooth shading (vs flat)
    pub smooth: bool,

    /// Enable transparency
    pub transparent: bool,

    /// Opacity (0.0 = transparent, 1.0 = opaque)
    pub opacity: f64,
}

impl Graphics3dOptions {
    pub fn new() -> Self {
        Self {
            background_color: Color::white(),
            default_color: None,
            aspect_ratio: None,
            lighting: true,
            wireframe: false,
            line_width: 1.0,
            smooth: true,
            transparent: false,
            opacity: 1.0,
        }
    }
}

impl Default for Graphics3dOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// A collection of 3D graphics primitives
///
/// This is the main container for 3D plots, similar to Graphics for 2D.
#[derive(Debug, Clone)]
pub struct Graphics3d {
    /// The 3D objects in this graphics
    pub objects: Vec<Box<dyn Graphics3dPrimitive>>,

    /// Rendering options
    pub options: Graphics3dOptions,

    /// Camera (optional, default will be computed from bounding box)
    pub camera: Option<crate::camera::Camera>,

    /// Lights (optional, default lighting will be used if empty)
    pub lights: Vec<crate::camera::Light>,
}

impl Graphics3d {
    /// Create a new empty 3D graphics
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            options: Graphics3dOptions::new(),
            camera: None,
            lights: Vec::new(),
        }
    }

    /// Add a 3D object to this graphics
    pub fn add(&mut self, object: Box<dyn Graphics3dPrimitive>) {
        self.objects.push(object);
    }

    /// Add a mesh to this graphics (convenience method)
    pub fn add_mesh(&mut self, mesh: IndexFaceSet) {
        self.objects.push(Box::new(mesh));
    }

    /// Set the camera
    pub fn set_camera(&mut self, camera: crate::camera::Camera) {
        self.camera = Some(camera);
    }

    /// Add a light source
    pub fn add_light(&mut self, light: crate::camera::Light) {
        self.lights.push(light);
    }

    /// Set rendering options
    pub fn set_options(&mut self, options: Graphics3dOptions) {
        self.options = options;
    }

    /// Compute the overall bounding box
    pub fn bounding_box(&self) -> BoundingBox3D {
        let mut bbox = BoundingBox3D::empty();
        for obj in &self.objects {
            bbox.merge(&obj.bounding_box());
        }
        bbox
    }

    /// Get the number of objects
    pub fn num_objects(&self) -> usize {
        self.objects.len()
    }

    /// Combine two graphics into one
    pub fn combine(&mut self, other: &Graphics3d) {
        for obj in &other.objects {
            self.objects.push(obj.clone());
        }
    }
}

impl Default for Graphics3d {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point3d() {
        let p1 = Point3D::new(1.0, 2.0, 3.0);
        let p2 = Point3D::new(4.0, 6.0, 8.0);

        assert_eq!(p1.x, 1.0);
        assert_eq!(p1.y, 2.0);
        assert_eq!(p1.z, 3.0);

        let dist = p1.distance_to(&p2);
        let expected = ((3.0_f64).powi(2) + (4.0_f64).powi(2) + (5.0_f64).powi(2)).sqrt();
        assert!((dist - expected).abs() < 1e-10);
    }

    #[test]
    fn test_vector3d() {
        let v1 = Vector3D::new(1.0, 0.0, 0.0);
        let v2 = Vector3D::new(0.0, 1.0, 0.0);

        assert_eq!(v1.magnitude(), 1.0);
        assert_eq!(v1.dot(&v2), 0.0);

        let cross = v1.cross(&v2);
        assert_eq!(cross, Vector3D::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_vector3d_operations() {
        let v1 = Vector3D::new(1.0, 2.0, 3.0);
        let v2 = Vector3D::new(4.0, 5.0, 6.0);

        let sum = v1 + v2;
        assert_eq!(sum, Vector3D::new(5.0, 7.0, 9.0));

        let diff = v2 - v1;
        assert_eq!(diff, Vector3D::new(3.0, 3.0, 3.0));

        let neg = -v1;
        assert_eq!(neg, Vector3D::new(-1.0, -2.0, -3.0));

        let scaled = v1.scale(2.0);
        assert_eq!(scaled, Vector3D::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vector3d_normalize() {
        let v = Vector3D::new(3.0, 4.0, 0.0);
        let normalized = v.normalize();
        assert!((normalized.magnitude() - 1.0).abs() < 1e-10);
        assert_eq!(normalized, Vector3D::new(0.6, 0.8, 0.0));
    }

    #[test]
    fn test_bounding_box3d() {
        let mut bbox = BoundingBox3D::empty();
        assert!(bbox.is_empty());

        bbox.expand_to_include(&Point3D::new(1.0, 2.0, 3.0));
        bbox.expand_to_include(&Point3D::new(4.0, 5.0, 6.0));

        assert_eq!(bbox.xmin, 1.0);
        assert_eq!(bbox.xmax, 4.0);
        assert_eq!(bbox.ymin, 2.0);
        assert_eq!(bbox.ymax, 5.0);
        assert_eq!(bbox.zmin, 3.0);
        assert_eq!(bbox.zmax, 6.0);

        let center = bbox.center();
        assert_eq!(center, Point3D::new(2.5, 3.5, 4.5));
    }

    #[test]
    fn test_index_face_set() {
        let mut mesh = IndexFaceSet::new();

        let v0 = mesh.add_vertex(Point3D::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3D::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3D::new(0.0, 1.0, 0.0));

        mesh.add_face(v0, v1, v2);

        assert_eq!(mesh.num_vertices(), 3);
        assert_eq!(mesh.num_faces(), 1);

        let bbox = mesh.bounding_box();
        assert_eq!(bbox.xmin, 0.0);
        assert_eq!(bbox.xmax, 1.0);
        assert_eq!(bbox.ymin, 0.0);
        assert_eq!(bbox.ymax, 1.0);
        assert_eq!(bbox.zmin, 0.0);
        assert_eq!(bbox.zmax, 0.0);
    }

    #[test]
    fn test_compute_normals() {
        let mut mesh = IndexFaceSet::new();

        // Create a simple triangle in the XY plane
        let v0 = mesh.add_vertex(Point3D::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3D::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3D::new(0.0, 1.0, 0.0));

        mesh.add_face(v0, v1, v2);
        mesh.compute_normals();

        assert!(mesh.normals.is_some());
        let normals = mesh.normals.unwrap();
        assert_eq!(normals.len(), 3);

        // All normals should point in +Z direction
        for normal in normals {
            assert!((normal.x).abs() < 1e-10);
            assert!((normal.y).abs() < 1e-10);
            assert!((normal.z - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_graphics3d() {
        let mut g = Graphics3d::new();
        assert_eq!(g.num_objects(), 0);

        let bbox = g.bounding_box();
        assert!(bbox.is_empty());
    }
}
