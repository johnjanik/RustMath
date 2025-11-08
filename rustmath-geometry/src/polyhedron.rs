//! 3D Polyhedron structures

use crate::point::Point3D;
use std::collections::HashSet;

/// A face of a polyhedron, represented as indices into the vertex list
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Face {
    pub vertices: Vec<usize>,
}

impl Face {
    /// Create a new face
    pub fn new(vertices: Vec<usize>) -> Self {
        Face { vertices }
    }

    /// Get the number of vertices in this face
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }
}

/// A polyhedron in 3D space
#[derive(Debug, Clone)]
pub struct Polyhedron {
    vertices: Vec<Point3D>,
    faces: Vec<Face>,
}

impl Polyhedron {
    /// Create a new polyhedron from vertices and faces
    pub fn new(vertices: Vec<Point3D>, faces: Vec<Face>) -> Result<Self, String> {
        if vertices.is_empty() {
            return Err("Polyhedron must have at least one vertex".to_string());
        }
        if faces.is_empty() {
            return Err("Polyhedron must have at least one face".to_string());
        }

        // Validate face indices
        for face in &faces {
            for &idx in &face.vertices {
                if idx >= vertices.len() {
                    return Err(format!("Face contains invalid vertex index: {}", idx));
                }
            }
        }

        Ok(Polyhedron { vertices, faces })
    }

    /// Get the vertices of the polyhedron
    pub fn vertices(&self) -> &[Point3D] {
        &self.vertices
    }

    /// Get the faces of the polyhedron
    pub fn faces(&self) -> &[Face] {
        &self.faces
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of faces
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Get the number of edges using Euler's formula: V - E + F = 2
    pub fn num_edges(&self) -> usize {
        // Count edges by iterating through faces
        let mut edge_set = HashSet::new();

        for face in &self.faces {
            let n = face.vertices.len();
            for i in 0..n {
                let v1 = face.vertices[i];
                let v2 = face.vertices[(i + 1) % n];
                let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                edge_set.insert(edge);
            }
        }

        edge_set.len()
    }

    /// Calculate the surface area of the polyhedron
    /// Assumes all faces are planar and computes their areas
    pub fn surface_area(&self) -> f64 {
        let mut total_area = 0.0;

        for face in &self.faces {
            if face.vertices.len() < 3 {
                continue;
            }

            // Use triangulation for each face
            let v0 = self.vertices[face.vertices[0]];

            for i in 1..face.vertices.len() - 1 {
                let v1 = self.vertices[face.vertices[i]];
                let v2 = self.vertices[face.vertices[i + 1]];

                // Calculate triangle area using cross product
                let edge1 = v1 - v0;
                let edge2 = v2 - v0;
                let cross = edge1.cross(&edge2);
                total_area += cross.magnitude() / 2.0;
            }
        }

        total_area
    }

    /// Calculate the volume of the polyhedron
    /// Uses the divergence theorem (assuming polyhedron is convex and origin is inside)
    pub fn volume(&self) -> f64 {
        let mut volume = 0.0;

        for face in &self.faces {
            if face.vertices.len() < 3 {
                continue;
            }

            // Triangulate the face from vertex 0
            let v0 = self.vertices[face.vertices[0]];

            for i in 1..face.vertices.len() - 1 {
                let v1 = self.vertices[face.vertices[i]];
                let v2 = self.vertices[face.vertices[i + 1]];

                // Calculate signed volume of tetrahedron formed by origin and triangle
                volume += v0.dot(&v1.cross(&v2));
            }
        }

        (volume / 6.0).abs()
    }

    /// Check if this is a lattice polytope (all vertices have integer coordinates)
    pub fn is_lattice_polytope(&self) -> bool {
        for vertex in &self.vertices {
            if (vertex.x.fract().abs() > 1e-10) ||
               (vertex.y.fract().abs() > 1e-10) ||
               (vertex.z.fract().abs() > 1e-10) {
                return false;
            }
        }
        true
    }

    /// Create a cube with side length s centered at the origin
    pub fn cube(s: f64) -> Self {
        let half = s / 2.0;
        let vertices = vec![
            Point3D::new(-half, -half, -half), // 0
            Point3D::new(half, -half, -half),  // 1
            Point3D::new(half, half, -half),   // 2
            Point3D::new(-half, half, -half),  // 3
            Point3D::new(-half, -half, half),  // 4
            Point3D::new(half, -half, half),   // 5
            Point3D::new(half, half, half),    // 6
            Point3D::new(-half, half, half),   // 7
        ];

        let faces = vec![
            Face::new(vec![0, 1, 2, 3]), // Bottom
            Face::new(vec![4, 7, 6, 5]), // Top
            Face::new(vec![0, 4, 5, 1]), // Front
            Face::new(vec![2, 6, 7, 3]), // Back
            Face::new(vec![0, 3, 7, 4]), // Left
            Face::new(vec![1, 5, 6, 2]), // Right
        ];

        Polyhedron::new(vertices, faces).unwrap()
    }

    /// Create a regular tetrahedron with edge length a
    pub fn tetrahedron(a: f64) -> Self {
        let h = a * (2.0_f64 / 3.0).sqrt();
        let r = a / (2.0_f64).sqrt();

        let vertices = vec![
            Point3D::new(0.0, 0.0, h / 2.0),
            Point3D::new(r, 0.0, -h / 4.0),
            Point3D::new(-r / 2.0, r * 3.0_f64.sqrt() / 2.0, -h / 4.0),
            Point3D::new(-r / 2.0, -r * 3.0_f64.sqrt() / 2.0, -h / 4.0),
        ];

        let faces = vec![
            Face::new(vec![0, 1, 2]),
            Face::new(vec![0, 2, 3]),
            Face::new(vec![0, 3, 1]),
            Face::new(vec![1, 3, 2]),
        ];

        Polyhedron::new(vertices, faces).unwrap()
    }

    /// Create a regular octahedron with edge length a
    pub fn octahedron(a: f64) -> Self {
        let r = a / 2.0_f64.sqrt();

        let vertices = vec![
            Point3D::new(r, 0.0, 0.0),  // +X
            Point3D::new(-r, 0.0, 0.0), // -X
            Point3D::new(0.0, r, 0.0),  // +Y
            Point3D::new(0.0, -r, 0.0), // -Y
            Point3D::new(0.0, 0.0, r),  // +Z
            Point3D::new(0.0, 0.0, -r), // -Z
        ];

        let faces = vec![
            Face::new(vec![0, 2, 4]),
            Face::new(vec![0, 4, 3]),
            Face::new(vec![0, 3, 5]),
            Face::new(vec![0, 5, 2]),
            Face::new(vec![1, 4, 2]),
            Face::new(vec![1, 3, 4]),
            Face::new(vec![1, 5, 3]),
            Face::new(vec![1, 2, 5]),
        ];

        Polyhedron::new(vertices, faces).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_properties() {
        let cube = Polyhedron::cube(2.0);

        assert_eq!(cube.num_vertices(), 8);
        assert_eq!(cube.num_faces(), 6);
        assert_eq!(cube.num_edges(), 12);

        // Check Euler's formula: V - E + F = 2
        let euler = cube.num_vertices() as i32 - cube.num_edges() as i32 + cube.num_faces() as i32;
        assert_eq!(euler, 2);

        // Volume of cube with side 2 is 8
        let volume = cube.volume();
        assert!((volume - 8.0).abs() < 1e-6);

        // Surface area of cube with side 2 is 24
        let area = cube.surface_area();
        assert!((area - 24.0).abs() < 1e-6);
    }

    #[test]
    fn test_tetrahedron_properties() {
        let tetra = Polyhedron::tetrahedron(1.0);

        assert_eq!(tetra.num_vertices(), 4);
        assert_eq!(tetra.num_faces(), 4);
        assert_eq!(tetra.num_edges(), 6);

        // Check Euler's formula
        let euler = tetra.num_vertices() as i32 - tetra.num_edges() as i32 + tetra.num_faces() as i32;
        assert_eq!(euler, 2);
    }

    #[test]
    fn test_octahedron_properties() {
        let octa = Polyhedron::octahedron(1.0);

        assert_eq!(octa.num_vertices(), 6);
        assert_eq!(octa.num_faces(), 8);
        assert_eq!(octa.num_edges(), 12);

        // Check Euler's formula
        let euler = octa.num_vertices() as i32 - octa.num_edges() as i32 + octa.num_faces() as i32;
        assert_eq!(euler, 2);
    }

    #[test]
    fn test_lattice_polytope() {
        let cube = Polyhedron::cube(2.0);
        assert!(cube.is_lattice_polytope());

        let vertices = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.5, 0.0, 0.0), // Non-integer
            Point3D::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![Face::new(vec![0, 1, 2])];
        let poly = Polyhedron::new(vertices, faces).unwrap();

        assert!(!poly.is_lattice_polytope());
    }
}
