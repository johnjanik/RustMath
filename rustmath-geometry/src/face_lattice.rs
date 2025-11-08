//! Face lattice structure for polyhedra

use crate::polyhedron::Polyhedron;
use std::collections::{HashMap, HashSet};

/// A face in the face lattice (can be a vertex, edge, or proper face)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LatticeFace {
    /// Indices of vertices in this face
    pub vertices: Vec<usize>,
    /// Dimension: 0 for vertex, 1 for edge, 2 for face, etc.
    pub dimension: usize,
}

impl LatticeFace {
    /// Create a new lattice face
    pub fn new(mut vertices: Vec<usize>, dimension: usize) -> Self {
        vertices.sort();
        LatticeFace { vertices, dimension }
    }

    /// Check if this face is contained in another face
    pub fn is_subface_of(&self, other: &LatticeFace) -> bool {
        if self.dimension >= other.dimension {
            return false;
        }

        self.vertices.iter().all(|v| other.vertices.contains(v))
    }
}

/// Face lattice of a polyhedron
///
/// A partially ordered set of all faces (vertices, edges, faces) of a polytope
#[derive(Debug, Clone)]
pub struct FaceLattice {
    /// All faces organized by dimension
    faces_by_dimension: HashMap<usize, Vec<LatticeFace>>,
    /// Inclusion relations: face_idx -> list of face indices it contains
    subfaces: HashMap<usize, Vec<usize>>,
    /// Flattened list of all faces
    all_faces: Vec<LatticeFace>,
}

impl FaceLattice {
    /// Compute the face lattice of a polyhedron
    pub fn from_polyhedron(poly: &Polyhedron) -> Self {
        let mut faces_by_dimension: HashMap<usize, Vec<LatticeFace>> = HashMap::new();
        let mut all_faces = Vec::new();

        // Dimension 0: Vertices
        let mut vertices = Vec::new();
        for i in 0..poly.num_vertices() {
            let face = LatticeFace::new(vec![i], 0);
            vertices.push(face.clone());
            all_faces.push(face);
        }
        faces_by_dimension.insert(0, vertices);

        // Dimension 1: Edges
        let mut edges = Vec::new();
        let mut edge_set = HashSet::new();

        for face in poly.faces() {
            let n = face.vertices.len();
            for i in 0..n {
                let v1 = face.vertices[i];
                let v2 = face.vertices[(i + 1) % n];
                let edge_key = if v1 < v2 { (v1, v2) } else { (v2, v1) };

                if !edge_set.contains(&edge_key) {
                    edge_set.insert(edge_key);
                    let edge_face = LatticeFace::new(vec![edge_key.0, edge_key.1], 1);
                    edges.push(edge_face.clone());
                    all_faces.push(edge_face);
                }
            }
        }
        faces_by_dimension.insert(1, edges);

        // Dimension 2: Proper faces
        let mut proper_faces = Vec::new();
        for face in poly.faces() {
            let face_vertices = face.vertices.clone();
            let lattice_face = LatticeFace::new(face_vertices, 2);
            proper_faces.push(lattice_face.clone());
            all_faces.push(lattice_face);
        }
        faces_by_dimension.insert(2, proper_faces);

        // Compute subface relations
        let mut subfaces = HashMap::new();
        for (idx, face) in all_faces.iter().enumerate() {
            let mut sub_list = Vec::new();

            for (sub_idx, sub_face) in all_faces.iter().enumerate() {
                if sub_face.is_subface_of(face) {
                    sub_list.push(sub_idx);
                }
            }

            if !sub_list.is_empty() {
                subfaces.insert(idx, sub_list);
            }
        }

        FaceLattice {
            faces_by_dimension,
            subfaces,
            all_faces,
        }
    }

    /// Get all faces of a specific dimension
    pub fn faces_of_dimension(&self, dim: usize) -> Option<&Vec<LatticeFace>> {
        self.faces_by_dimension.get(&dim)
    }

    /// Get the total number of faces in the lattice
    pub fn num_faces(&self) -> usize {
        self.all_faces.len()
    }

    /// Get the number of faces of a specific dimension
    pub fn num_faces_of_dimension(&self, dim: usize) -> usize {
        self.faces_by_dimension.get(&dim).map_or(0, |v| v.len())
    }

    /// Get all faces
    pub fn all_faces(&self) -> &[LatticeFace] {
        &self.all_faces
    }

    /// Get the subfaces of a specific face
    pub fn subfaces_of(&self, face_idx: usize) -> Option<&Vec<usize>> {
        self.subfaces.get(&face_idx)
    }

    /// Check if the face lattice satisfies basic properties
    pub fn is_valid(&self) -> bool {
        // Every face should have proper subface relations
        for (idx, face) in self.all_faces.iter().enumerate() {
            if let Some(subs) = self.subfaces.get(&idx) {
                for &sub_idx in subs {
                    if !self.all_faces[sub_idx].is_subface_of(face) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Get the f-vector of the polytope
    ///
    /// The f-vector is (f_0, f_1, f_2, ...) where f_i is the number of i-dimensional faces
    pub fn f_vector(&self) -> Vec<usize> {
        let max_dim = *self.faces_by_dimension.keys().max().unwrap_or(&0);
        let mut f_vec = Vec::new();

        for d in 0..=max_dim {
            f_vec.push(self.num_faces_of_dimension(d));
        }

        f_vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyhedron::Polyhedron;

    #[test]
    fn test_face_lattice_cube() {
        let cube = Polyhedron::cube(2.0);
        let lattice = FaceLattice::from_polyhedron(&cube);

        // Cube has 8 vertices, 12 edges, 6 faces
        assert_eq!(lattice.num_faces_of_dimension(0), 8);
        assert_eq!(lattice.num_faces_of_dimension(1), 12);
        assert_eq!(lattice.num_faces_of_dimension(2), 6);

        // F-vector should be [8, 12, 6]
        let f_vec = lattice.f_vector();
        assert_eq!(f_vec, vec![8, 12, 6]);
    }

    #[test]
    fn test_face_lattice_tetrahedron() {
        let tetra = Polyhedron::tetrahedron(1.0);
        let lattice = FaceLattice::from_polyhedron(&tetra);

        // Tetrahedron has 4 vertices, 6 edges, 4 faces
        assert_eq!(lattice.num_faces_of_dimension(0), 4);
        assert_eq!(lattice.num_faces_of_dimension(1), 6);
        assert_eq!(lattice.num_faces_of_dimension(2), 4);

        // F-vector should be [4, 6, 4]
        let f_vec = lattice.f_vector();
        assert_eq!(f_vec, vec![4, 6, 4]);
    }

    #[test]
    fn test_face_lattice_is_valid() {
        let cube = Polyhedron::cube(2.0);
        let lattice = FaceLattice::from_polyhedron(&cube);

        assert!(lattice.is_valid());
    }

    #[test]
    fn test_subface_relation() {
        let face1 = LatticeFace::new(vec![0, 1], 1); // Edge
        let face2 = LatticeFace::new(vec![0, 1, 2, 3], 2); // Face

        assert!(face1.is_subface_of(&face2));
        assert!(!face2.is_subface_of(&face1));
    }
}
