//! Circle packing on the flag triangulation — the conformal step of the P2
//! construction (Koebe–Andreev–Thurston / KMSV).
//!
//! A circle packing assigns a radius `r_v > 0` to each triangulation vertex so
//! that in every triangle the three vertex-circles are mutually externally
//! tangent. The radii that make every vertex's *angle sum* hit its target encode
//! the conformal structure; laying the circles out and projecting to the round
//! sphere gives the approximate positions of the dessin's `0/1/∞` preimages on
//! `ℙ¹(ℂ)` — the near-cover that local Newton then refines and the true-root gate
//! screens.
//!
//! This module provides the **combinatorial + Euclidean-geometry core**:
//! * [`PackingComplex`] — the triangle/incidence structure, read straight off the
//!   [`FlagTriangulation`] flag corners. Because flags are abstract triangles, the
//!   two sides of a leaf edge stay distinct (no bigon collapse), so degenerate
//!   dessin vertices are handled correctly.
//! * [`euclidean_flag_angle`] — the tangent-circle angle law.
//! * [`relax_euclidean`] — Thurston's angle-sum relaxation (per-vertex monotone
//!   solve, Gauss–Seidel sweeps) for a Euclidean packing with fixed boundary radii.
//!
//! The spherical layout + stereographic projection (removing one vertex to a disk,
//! packing, projecting, Möbius-normalising) is the next step and is not here yet.

use crate::belyi::flags::FlagTriangulation;
use std::f64::consts::PI;

/// A triangulated complex for circle packing: a vertex count and a triangle list,
/// with per-vertex incidence (the two other corners of each incident triangle).
#[derive(Debug, Clone)]
pub struct PackingComplex {
    pub n_vertices: usize,
    /// One triple per triangle (a flag, for a flag triangulation).
    pub triangles: Vec<[usize; 3]>,
    /// `incident[v]` = for each triangle containing `v`, the two *other* vertices.
    incident: Vec<Vec<(usize, usize)>>,
}

impl PackingComplex {
    /// Build from an explicit vertex count and triangle list.
    pub fn new(n_vertices: usize, triangles: Vec<[usize; 3]>) -> Self {
        let mut incident = vec![Vec::new(); n_vertices];
        for t in &triangles {
            let [a, b, c] = *t;
            incident[a].push((b, c));
            incident[b].push((a, c));
            incident[c].push((a, b));
        }
        Self {
            n_vertices,
            triangles,
            incident,
        }
    }

    /// Build the packing complex from a flag triangulation: each flag is a triangle
    /// `[vertex, midpoint, face]`. Leaf-edge sides remain distinct triangles.
    pub fn from_flags(tri: &FlagTriangulation) -> Self {
        let triangles: Vec<[usize; 3]> = tri
            .corners
            .iter()
            .map(|c| [c.vertex, c.midpoint, c.face])
            .collect();
        Self::new(tri.n_vertices(), triangles)
    }

    /// Number of triangles incident to `v`.
    pub fn degree(&self, v: usize) -> usize {
        self.incident[v].len()
    }

    /// Angle sum at `v`: sum over incident triangles of the tangent-circle angle at
    /// `v` (using the current radii).
    pub fn euclidean_angle_sum(&self, radii: &[f64], v: usize) -> f64 {
        self.incident[v]
            .iter()
            .map(|&(a, b)| euclidean_flag_angle(radii[v], radii[a], radii[b]))
            .sum()
    }
}

/// The interior angle at the centre circle of a triangle of three mutually
/// externally tangent circles with radii `(r_c, r_a, r_b)` — the corner at the
/// `r_c` circle. Side lengths are the tangent distances `r_c+r_a`, `r_c+r_b`,
/// `r_a+r_b`; the angle follows from the law of cosines. Monotonically decreasing
/// in `r_c` (from `π` as `r_c→0` to `0` as `r_c→∞`).
pub fn euclidean_flag_angle(r_c: f64, r_a: f64, r_b: f64) -> f64 {
    let s_a = r_c + r_a; // side from centre to circle a
    let s_b = r_c + r_b; // side from centre to circle b
    let s_o = r_a + r_b; // opposite side
    let cos = (s_a * s_a + s_b * s_b - s_o * s_o) / (2.0 * s_a * s_b);
    cos.clamp(-1.0, 1.0).acos()
}

/// Configuration for the Euclidean relaxation.
#[derive(Debug, Clone)]
pub struct PackingConfig {
    pub max_iters: usize,
    /// Stop when the max `|angle_sum − target|` over interior vertices is below this.
    pub tol: f64,
}

impl Default for PackingConfig {
    fn default() -> Self {
        Self {
            max_iters: 5000,
            tol: 1e-10,
        }
    }
}

/// Outcome of a relaxation run.
#[derive(Debug, Clone)]
pub struct PackingResult {
    pub iterations: usize,
    pub max_angle_error: f64,
    pub converged: bool,
}

/// For interior vertex `v`, find the radius making its angle sum equal `target[v]`.
/// Angle sum is strictly decreasing in `r_v`, so a monotone bisection is exact.
fn solve_radius_for_target(
    complex: &PackingComplex,
    radii: &mut [f64],
    v: usize,
    target: f64,
) {
    let mut lo = 1e-14_f64;
    let mut hi = 1e14_f64;
    // angle_sum(lo) is near degree·π (max); angle_sum(hi) ~ 0. Bisect for target.
    for _ in 0..80 {
        let mid = (lo * hi).sqrt(); // geometric mean: radii span many decades
        radii[v] = mid;
        let a = complex.euclidean_angle_sum(radii, v);
        if a > target {
            // angle too large ⇒ radius too small ⇒ raise lo
            lo = mid;
        } else {
            hi = mid;
        }
    }
    radii[v] = (lo * hi).sqrt();
}

/// Thurston's angle-sum relaxation for a Euclidean packing. `interior[v]` marks the
/// vertices whose radii are solved (to `targets[v]`); the rest are held fixed
/// (boundary conditions). Radii are updated in place (Gauss–Seidel).
pub fn relax_euclidean(
    complex: &PackingComplex,
    radii: &mut [f64],
    interior: &[bool],
    targets: &[f64],
    cfg: &PackingConfig,
) -> PackingResult {
    let mut iterations = 0;
    let mut max_err = f64::INFINITY;
    while iterations < cfg.max_iters {
        for v in 0..complex.n_vertices {
            if interior[v] {
                solve_radius_for_target(complex, radii, v, targets[v]);
            }
        }
        // Convergence: max angle-sum error over interior vertices.
        max_err = (0..complex.n_vertices)
            .filter(|&v| interior[v])
            .map(|v| (complex.euclidean_angle_sum(radii, v) - targets[v]).abs())
            .fold(0.0_f64, f64::max);
        iterations += 1;
        if max_err < cfg.tol {
            break;
        }
    }
    PackingResult {
        iterations,
        max_angle_error: max_err,
        converged: max_err < cfg.tol,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belyi::flags::flag_triangulation;
    use crate::belyi::monodromy::Permutation;

    #[test]
    fn equal_circles_make_sixty_degrees() {
        // Three equal mutually tangent circles ⇒ equilateral ⇒ π/3 at each corner.
        assert!((euclidean_flag_angle(1.0, 1.0, 1.0) - PI / 3.0).abs() < 1e-12);
    }

    #[test]
    fn angle_is_monotone_decreasing_in_center_radius() {
        let small = euclidean_flag_angle(0.5, 1.0, 1.0);
        let mid = euclidean_flag_angle(1.0, 1.0, 1.0);
        let big = euclidean_flag_angle(2.0, 1.0, 1.0);
        assert!(small > mid && mid > big, "{small} {mid} {big}");
        // Limits: r_c→0 ⇒ π, r_c→∞ ⇒ 0.
        assert!(euclidean_flag_angle(1e-9, 1.0, 1.0) > PI - 1e-3);
        assert!(euclidean_flag_angle(1e9, 1.0, 1.0) < 1e-3);
    }

    #[test]
    fn hexagonal_flower_center_radius_is_one() {
        // Center vertex 0 with a fan of 6 triangles to boundary 1..=6 (cyclic).
        // Boundary radii fixed at 1; solving the center to angle sum 2π must give
        // radius 1 (6 · π/3 = 2π by symmetry).
        let tris = vec![
            [0, 1, 2], [0, 2, 3], [0, 3, 4],
            [0, 4, 5], [0, 5, 6], [0, 6, 1],
        ];
        let complex = PackingComplex::new(7, tris);
        let mut radii = vec![1.0; 7];
        radii[0] = 5.0; // deliberately wrong start
        let mut interior = vec![false; 7];
        interior[0] = true;
        let mut targets = vec![0.0; 7];
        targets[0] = 2.0 * PI;
        let res = relax_euclidean(&complex, &mut radii, &interior, &targets, &PackingConfig::default());
        assert!(res.converged, "err {}", res.max_angle_error);
        assert!((radii[0] - 1.0).abs() < 1e-9, "center radius {}", radii[0]);
    }

    #[test]
    fn heptagon_flower_center_grows() {
        // Seven unit boundary circles: exactly 6 equal pennies fit around one, so a
        // 7th forces a LARGER center (angle 2π/7 < π/3, and angle decreases in r_c).
        let tris: Vec<[usize; 3]> = (0..7).map(|i| [0, 1 + i, 1 + (i + 1) % 7]).collect();
        let complex = PackingComplex::new(8, tris);
        let mut radii = vec![1.0; 8];
        let mut interior = vec![false; 8];
        interior[0] = true;
        let mut targets = vec![0.0; 8];
        targets[0] = 2.0 * PI;
        let res = relax_euclidean(&complex, &mut radii, &interior, &targets, &PackingConfig::default());
        assert!(res.converged);
        assert!(radii[0] > 1.0, "center radius {}", radii[0]);
        // Cross-check the angle sum directly.
        assert!((complex.euclidean_angle_sum(&radii, 0) - 2.0 * PI).abs() < 1e-9);
    }

    #[test]
    fn packing_complex_from_star_dessin() {
        // The star S3 flag triangulation ⇒ 8 vertices, 12 flag triangles.
        let s0 = Permutation::from_cycles(3, &[vec![0, 1, 2]]).unwrap();
        let s1 = Permutation::identity(3);
        let tri = flag_triangulation(&s0, &s1).unwrap();
        let complex = PackingComplex::from_flags(&tri);
        assert_eq!(complex.n_vertices, tri.n_vertices()); // 1+3+1+3 = 8
        assert_eq!(complex.triangles.len(), 12); // 4·deg
        // Every triangle incidence is accounted for: Σ degree = 3 · #triangles.
        let total: usize = (0..complex.n_vertices).map(|v| complex.degree(v)).sum();
        assert_eq!(total, 3 * complex.triangles.len());
    }
}
