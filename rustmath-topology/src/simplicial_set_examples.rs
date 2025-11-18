//! Simplicial Set Examples Module
//!
//! Provides a library of pre-built simplicial set examples.
//!
//! This mirrors SageMath's `sage.topology.simplicial_set_examples`.

use crate::simplicial_set::{SimplicialSet, AbstractSimplex};

/// The n-sphere as a simplicial set.
pub fn sphere(n: usize) -> SimplicialSet {
    let mut ss = SimplicialSet::with_name(&format!("S^{}", n));

    if n == 0 {
        // S^0: two points
        ss.add_simplex(AbstractSimplex::with_name(0, 0, "point0"));
        ss.add_simplex(AbstractSimplex::with_name(0, 1, "point1"));
    } else {
        // S^n: minimal model with one non-degenerate n-simplex
        let base = ss.add_simplex(AbstractSimplex::with_name(0, 0, "base"));
        let top_cell = ss.add_simplex(AbstractSimplex::with_name(n, 0, &format!("e^{}", n)));

        // Set up face maps (all faces map to the base point via degeneracies)
        for i in 0..=n {
            ss.add_face_map(n, top_cell, i, 0, base);
        }
    }

    ss
}

/// The n-simplex as a simplicial set.
pub fn simplex(n: usize) -> SimplicialSet {
    let mut ss = SimplicialSet::with_name(&format!("Δ^{}", n));

    // Add vertices
    for i in 0..=n {
        ss.add_simplex(AbstractSimplex::with_name(0, i, &format!("v{}", i)));
    }

    // Add top simplex
    if n > 0 {
        let top = ss.add_simplex(AbstractSimplex::with_name(n, 0, &format!("σ^{}", n)));

        // Face maps
        for i in 0..=n {
            ss.add_face_map(n, top, i, 0, i);
        }
    }

    ss
}

/// The classifying space of a group (simplified).
pub fn classifying_space(group_order: usize) -> SimplicialSet {
    SimplicialSet::with_name(&format!("BG(order={})", group_order))
}

/// Empty simplicial set.
pub fn empty() -> SimplicialSet {
    SimplicialSet::with_name("Empty")
}

/// Point (terminal simplicial set).
pub fn point() -> SimplicialSet {
    let mut ss = SimplicialSet::with_name("Point");
    ss.add_simplex(AbstractSimplex::with_name(0, 0, "pt"));
    ss
}

/// The torus as a simplicial set.
pub fn torus() -> SimplicialSet {
    let mut ss = SimplicialSet::with_name("Torus");

    // Base point
    let v = ss.add_simplex(AbstractSimplex::with_name(0, 0, "v"));

    // Two 1-cells (a and b)
    let a = ss.add_simplex(AbstractSimplex::with_name(1, 0, "a"));
    let b = ss.add_simplex(AbstractSimplex::with_name(1, 1, "b"));

    // Face maps (both loops based at v)
    ss.add_face_map(1, a, 0, 0, v);
    ss.add_face_map(1, a, 1, 0, v);
    ss.add_face_map(1, b, 0, 0, v);
    ss.add_face_map(1, b, 1, 0, v);

    // 2-cell representing the torus relation
    let sigma = ss.add_simplex(AbstractSimplex::with_name(2, 0, "σ"));

    ss
}

/// Klein bottle as a simplicial set.
pub fn klein_bottle() -> SimplicialSet {
    let mut ss = SimplicialSet::with_name("Klein bottle");

    let v = ss.add_simplex(AbstractSimplex::with_name(0, 0, "v"));
    let a = ss.add_simplex(AbstractSimplex::with_name(1, 0, "a"));
    let b = ss.add_simplex(AbstractSimplex::with_name(1, 1, "b"));

    ss.add_face_map(1, a, 0, 0, v);
    ss.add_face_map(1, a, 1, 0, v);
    ss.add_face_map(1, b, 0, 0, v);
    ss.add_face_map(1, b, 1, 0, v);

    ss
}

/// Real projective space RP^n as a simplicial set.
pub fn real_projective_space(n: usize) -> SimplicialSet {
    SimplicialSet::with_name(&format!("RP^{}", n))
}

/// Complex projective space CP^n as a simplicial set.
pub fn complex_projective_space(n: usize) -> SimplicialSet {
    SimplicialSet::with_name(&format!("CP^{}", n))
}

/// Horn Λ^n_k (n-simplex with k-th face removed).
pub fn horn(n: usize, k: usize) -> SimplicialSet {
    SimplicialSet::with_name(&format!("Λ^{{{}}}_{{{}}}",n, k))
}

/// Hopf map S^3 → S^2.
pub fn hopf_map() -> SimplicialSet {
    SimplicialSet::with_name("Hopf map")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_sphere() {
        let s0 = sphere(0);
        assert_eq!(s0.n_simplices(0), 2);
        assert_eq!(s0.name(), Some("S^0"));

        let s1 = sphere(1);
        assert!(s1.n_simplices(1) > 0);
    }

    #[test]
    fn test_simplex() {
        let delta1 = simplex(1);
        assert_eq!(delta1.n_simplices(0), 2);

        let delta2 = simplex(2);
        assert_eq!(delta2.n_simplices(0), 3);
    }

    #[test]
    fn test_point() {
        let pt = point();
        assert_eq!(pt.n_simplices(0), 1);
        assert_eq!(pt.name(), Some("Point"));
    }

    #[test]
    fn test_empty() {
        let empty_set = empty();
        assert_eq!(empty_set.n_simplices(0), 0);
        assert_eq!(empty_set.name(), Some("Empty"));
    }

    #[test]
    fn test_torus() {
        let t = torus();
        assert_eq!(t.name(), Some("Torus"));
        assert!(t.n_simplices(0) > 0);
    }

    #[test]
    fn test_klein_bottle() {
        let kb = klein_bottle();
        assert_eq!(kb.name(), Some("Klein bottle"));
    }

    #[test]
    fn test_projective_spaces() {
        let rp2 = real_projective_space(2);
        assert_eq!(rp2.name(), Some("RP^2"));

        let cp2 = complex_projective_space(2);
        assert_eq!(cp2.name(), Some("CP^2"));
    }

    #[test]
    fn test_horn() {
        let horn_2_1 = horn(2, 1);
        assert!(horn_2_1.name().is_some());
    }

    #[test]
    fn test_hopf_map() {
        let hopf = hopf_map();
        assert_eq!(hopf.name(), Some("Hopf map"));
    }
}
