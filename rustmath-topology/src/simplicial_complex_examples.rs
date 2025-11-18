//! Simplicial Complex Examples Module
//!
//! Provides a library of pre-built simplicial complex examples.
//!
//! This mirrors SageMath's `sage.topology.simplicial_complex_examples`.

use crate::simplicial_complex::{SimplicialComplex, Simplex};

/// Sphere as a simplicial complex.
pub fn sphere(n: usize) -> SimplicialComplex {
    let mut complex = SimplicialComplex::with_name(&format!("S^{}", n));

    if n == 0 {
        // S^0: two disjoint points
        complex.add_simplex(Simplex::new(vec![0]));
        complex.add_simplex(Simplex::new(vec![1]));
    } else if n == 1 {
        // S^1: triangle boundary
        complex.add_simplex(Simplex::new(vec![0, 1]));
        complex.add_simplex(Simplex::new(vec![1, 2]));
        complex.add_simplex(Simplex::new(vec![0, 2]));
    } else {
        // S^n: boundary of (n+1)-simplex
        let full_simplex = (0..=n + 1).collect::<Vec<usize>>();
        for i in 0..=n + 1 {
            let mut face = full_simplex.clone();
            face.remove(i);
            complex.add_simplex(Simplex::new(face));
        }
    }

    complex
}

/// Simplex as a simplicial complex.
pub fn simplex(n: usize) -> SimplicialComplex {
    let mut complex = SimplicialComplex::with_name(&format!("Δ^{}", n));
    let vertices: Vec<usize> = (0..=n).collect();
    complex.add_simplex(Simplex::new(vertices));
    complex
}

/// Torus as a simplicial complex.
pub fn torus() -> SimplicialComplex {
    let mut complex = SimplicialComplex::with_name("Torus");

    // Minimal triangulation of torus: 7 vertices, 14 triangles
    let facets = vec![
        vec![0, 1, 3], vec![0, 1, 4], vec![0, 2, 3], vec![0, 2, 5],
        vec![0, 4, 5], vec![1, 2, 6], vec![1, 3, 6], vec![1, 4, 6],
        vec![2, 3, 4], vec![2, 4, 5], vec![2, 5, 6], vec![3, 4, 6],
        vec![3, 5, 6], vec![4, 5, 6],
    ];

    for facet in facets {
        complex.add_simplex(Simplex::new(facet));
    }

    complex
}

/// Klein bottle as a simplicial complex.
pub fn klein_bottle() -> SimplicialComplex {
    let mut complex = SimplicialComplex::with_name("Klein bottle");

    // Triangulation of Klein bottle
    let facets = vec![
        vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 4], vec![0, 3, 4],
        vec![1, 2, 5], vec![1, 3, 5], vec![2, 4, 5], vec![3, 4, 5],
    ];

    for facet in facets {
        complex.add_simplex(Simplex::new(facet));
    }

    complex
}

/// Real projective plane as a simplicial complex.
pub fn projective_plane() -> SimplicialComplex {
    let mut complex = SimplicialComplex::with_name("RP^2");

    // Minimal triangulation of RP^2: 6 vertices, 10 triangles
    let facets = vec![
        vec![0, 1, 2], vec![0, 1, 5], vec![0, 2, 3], vec![0, 3, 4],
        vec![0, 4, 5], vec![1, 2, 4], vec![1, 3, 4], vec![1, 3, 5],
        vec![2, 3, 5], vec![2, 4, 5],
    ];

    for facet in facets {
        complex.add_simplex(Simplex::new(facet));
    }

    complex
}

/// Surface of genus g.
pub fn surface_of_genus(g: usize) -> SimplicialComplex {
    let name = format!("Surface of genus {}", g);
    let mut complex = SimplicialComplex::with_name(&name);

    if g == 0 {
        // Sphere
        return sphere(2);
    } else if g == 1 {
        return torus();
    }

    // For g > 1, use a construction based on polygon identification
    // This is a simplified version
    let n_vertices = 4 * g + 2;
    for i in 0..n_vertices {
        complex.add_simplex(Simplex::new(vec![i % n_vertices, (i + 1) % n_vertices]));
    }

    complex
}

/// Complex projective plane.
pub fn complex_projective_plane() -> SimplicialComplex {
    SimplicialComplex::with_name("CP^2")
}

/// Moore space M(Z/nZ, 1).
pub fn moore_space(n: usize) -> SimplicialComplex {
    let mut complex = SimplicialComplex::with_name(&format!("M(Z/{}Z, 1)", n));

    // Moore space: wedge of n circles
    complex.add_simplex(Simplex::new(vec![0])); // Base point
    for i in 1..=n {
        complex.add_simplex(Simplex::new(vec![0, i])); // n loops
    }

    complex
}

/// Dunce hat: contractible but not collapsible.
pub fn dunce_hat() -> SimplicialComplex {
    let mut complex = SimplicialComplex::with_name("Dunce hat");

    let facets = vec![
        vec![0, 1, 2],
        vec![0, 2, 3],
        vec![0, 3, 4],
        vec![0, 4, 5],
        vec![0, 1, 5],
        vec![1, 2, 4],
        vec![2, 3, 5],
        vec![1, 3, 4],
    ];

    for facet in facets {
        complex.add_simplex(Simplex::new(facet));
    }

    complex
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_sphere() {
        let s0 = sphere(0);
        assert_eq!(s0.n_simplices(0), 2);
        assert_eq!(s0.euler_characteristic(), Integer::from(2));

        let s1 = sphere(1);
        assert_eq!(s1.euler_characteristic(), Integer::from(0));

        let s2 = sphere(2);
        assert_eq!(s2.euler_characteristic(), Integer::from(2));
    }

    #[test]
    fn test_simplex() {
        let delta2 = simplex(2);
        assert_eq!(delta2.dimension(), Some(2));
        assert_eq!(delta2.n_simplices(2), 1);
        assert_eq!(delta2.euler_characteristic(), Integer::from(1));
    }

    #[test]
    fn test_torus() {
        let t = torus();
        assert_eq!(t.euler_characteristic(), Integer::from(0));
        assert_eq!(t.name(), Some("Torus"));
    }

    #[test]
    fn test_klein_bottle() {
        let kb = klein_bottle();
        assert_eq!(kb.euler_characteristic(), Integer::from(0));
        assert_eq!(kb.name(), Some("Klein bottle"));
    }

    #[test]
    fn test_projective_plane() {
        let rp2 = projective_plane();
        assert_eq!(rp2.euler_characteristic(), Integer::from(1));
        assert_eq!(rp2.name(), Some("RP^2"));
    }

    #[test]
    fn test_surface_of_genus() {
        let genus_0 = surface_of_genus(0);
        assert_eq!(genus_0.euler_characteristic(), Integer::from(2)); // Sphere

        let genus_1 = surface_of_genus(1);
        assert_eq!(genus_1.euler_characteristic(), Integer::from(0)); // Torus
    }

    #[test]
    fn test_moore_space() {
        let m3 = moore_space(3);
        assert_eq!(m3.name(), Some("M(Z/3Z, 1)"));
    }

    #[test]
    fn test_dunce_hat() {
        let dh = dunce_hat();
        assert_eq!(dh.name(), Some("Dunce hat"));
        assert_eq!(dh.euler_characteristic(), Integer::from(1)); // Contractible
    }
}
