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
use std::collections::HashMap;
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

// ---------------------------------------------------------------------------
// Layout: place circle centres in the plane from the solved radii.
// ---------------------------------------------------------------------------

/// Sorted vertex pair, the key for an undirected triangulation edge.
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Place the third circle centre of a triangle given the two placed centres `a`,
/// `b`, the target tangency distances `da = r_a+r_q`, `db = r_b+r_q`, and a side
/// (`+1`/`-1`) selecting which half-plane of line `ab`.
fn place_third(a: [f64; 2], b: [f64; 2], da: f64, db: f64, side: f64) -> [f64; 2] {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let d = (dx * dx + dy * dy).sqrt();
    let (ux, uy) = (dx / d, dy / d);
    let t = (da * da - db * db + d * d) / (2.0 * d);
    let h = (da * da - t * t).max(0.0).sqrt();
    // n = u rotated +90°.
    [a[0] + t * ux - side * h * uy, a[1] + t * uy + side * h * ux]
}

fn cross(o: [f64; 2], p: [f64; 2], q: [f64; 2]) -> f64 {
    (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
}

/// Lay out circle centres in the plane by propagating tangency across shared
/// edges, starting from `seed_tri`. Each new triangle places its third centre on
/// the half-plane opposite the already-laid neighbour across the shared edge, so
/// the layout is orientation-consistent (non-overlapping for a valid packing).
/// Vertices not reachable from the seed's component are left `None`.
pub fn layout_euclidean(
    complex: &PackingComplex,
    radii: &[f64],
    seed_tri: usize,
) -> Vec<Option<[f64; 2]>> {
    let mut pos: Vec<Option<[f64; 2]>> = vec![None; complex.n_vertices];
    let ntri = complex.triangles.len();
    let mut placed = vec![false; ntri];

    // edge -> triangles sharing it.
    let mut edge_map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (ti, t) in complex.triangles.iter().enumerate() {
        let [a, b, c] = *t;
        for &(u, v) in &[(a, b), (b, c), (a, c)] {
            edge_map.entry(edge_key(u, v)).or_default().push(ti);
        }
    }

    let third = |t: [usize; 3], a: usize, b: usize| -> usize {
        *t.iter().find(|&&x| x != a && x != b).unwrap()
    };

    // Seed placement.
    let [s0, s1, s2] = complex.triangles[seed_tri];
    pos[s0] = Some([0.0, 0.0]);
    pos[s1] = Some([radii[s0] + radii[s1], 0.0]);
    pos[s2] = Some(place_third(
        pos[s0].unwrap(),
        pos[s1].unwrap(),
        radii[s0] + radii[s2],
        radii[s1] + radii[s2],
        1.0,
    ));
    placed[seed_tri] = true;

    // Stack of (triangle, shared placed edge).
    let mut stack: Vec<(usize, (usize, usize))> = Vec::new();
    let push_edges = |stack: &mut Vec<_>, placed: &[bool], t: [usize; 3], em: &HashMap<_, Vec<usize>>| {
        let [a, b, c] = t;
        for &(u, v) in &[(a, b), (b, c), (a, c)] {
            let k = edge_key(u, v);
            for &nt in em.get(&k).into_iter().flatten() {
                if !placed[nt] {
                    stack.push((nt, k));
                }
            }
        }
    };
    push_edges(&mut stack, &placed, [s0, s1, s2], &edge_map);

    while let Some((t, (a, b))) = stack.pop() {
        if placed[t] || pos[a].is_none() || pos[b].is_none() {
            continue;
        }
        let q = third(complex.triangles[t], a, b);
        // Reference: the third vertex of an already-placed triangle across (a,b).
        let side = edge_map[&(a, b)]
            .iter()
            .find(|&&nt| placed[nt] && nt != t)
            .and_then(|&nt| {
                let p = third(complex.triangles[nt], a, b);
                pos[p].map(|pp| -cross(pos[a].unwrap(), pos[b].unwrap(), pp).signum())
            })
            .unwrap_or(1.0);
        let newpos = place_third(
            pos[a].unwrap(),
            pos[b].unwrap(),
            radii[a] + radii[q],
            radii[b] + radii[q],
            side,
        );
        if pos[q].is_none() {
            pos[q] = Some(newpos);
        }
        placed[t] = true;
        push_edges(&mut stack, &placed, complex.triangles[t], &edge_map);
    }
    pos
}

// ---------------------------------------------------------------------------
// Stereographic projection: plane (ℂ = ℙ¹ minus ∞) <-> round unit sphere.
// ---------------------------------------------------------------------------

/// Project a plane point to the unit sphere (from the north pole). Origin ->
/// south pole `(0,0,-1)`, `|z|=1` -> equator, `|z|→∞` -> north pole `(0,0,1)`.
pub fn stereographic_to_sphere(z: [f64; 2]) -> [f64; 3] {
    let (x, y) = (z[0], z[1]);
    let d = 1.0 + x * x + y * y;
    [2.0 * x / d, 2.0 * y / d, (x * x + y * y - 1.0) / d]
}

/// Inverse projection: unit sphere point (with `z < 1`) back to the plane.
pub fn stereographic_to_plane(p: [f64; 3]) -> [f64; 2] {
    let s = 1.0 - p[2];
    [p[0] / s, p[1] / s]
}

/// The north pole — the image of `∞`.
pub const NORTH_POLE: [f64; 3] = [0.0, 0.0, 1.0];

/// A spherical layout of the dessin's triangulation vertices on `ℙ¹(ℂ)`.
#[derive(Debug, Clone)]
pub struct SphereLayout {
    /// Vertex chosen to carry `∞` (placed at the north pole).
    pub infinity_vertex: usize,
    /// Position of each triangulation vertex on the unit sphere (`None` if it was
    /// only incident to the removed `∞` star, i.e. unreachable in the disk).
    pub positions: Vec<Option<[f64; 3]>>,
    /// The relaxation outcome for the punctured disk.
    pub packing: PackingResult,
}

/// Build an approximate spherical circle packing of a genus-0 flag triangulation:
/// puncture at the highest-degree vertex, Euclidean-pack the resulting disk
/// (interior angle sums -> 2π, the puncture's link held as fixed-radius boundary),
/// lay the centres out in the plane, and stereographically project to the sphere
/// with the puncture at the north pole.
///
/// Note: the boundary uses a fixed radius rather than the rigorous horocyclic
/// max-packing condition, so this is a first approximation of the canonical
/// packing — enough to seed local Newton, to be refined next.
pub fn sphere_layout(tri: &FlagTriangulation, cfg: &PackingConfig) -> SphereLayout {
    let full = PackingComplex::from_flags(tri);
    let n = full.n_vertices;
    let total = full.triangles.len();
    // Puncture at the max-degree vertex whose removal still leaves a non-empty disk
    // (a vertex incident to *every* triangle — e.g. the single face of a star —
    // cannot serve as the ∞ puncture).
    let v_inf = (0..n)
        .filter(|&v| full.degree(v) < total)
        .max_by_key(|&v| full.degree(v))
        .unwrap_or(0);

    // Disk = triangles not incident to v_inf.
    let sub: Vec<[usize; 3]> = full
        .triangles
        .iter()
        .filter(|t| !t.contains(&v_inf))
        .copied()
        .collect();
    let disk = PackingComplex::new(n, sub);

    // Boundary = neighbours of v_inf; interior = the rest (excluding v_inf and any
    // vertex that survived only in the removed star).
    let mut boundary = vec![false; n];
    for &(a, b) in &full.incident[v_inf] {
        boundary[a] = true;
        boundary[b] = true;
    }
    let interior: Vec<bool> = (0..n)
        .map(|v| v != v_inf && !boundary[v] && disk.degree(v) > 0)
        .collect();

    let mut radii = vec![1.0_f64; n];
    let targets = vec![2.0 * PI; n];
    let packing = relax_euclidean(&disk, &mut radii, &interior, &targets, cfg);

    // Layout from the first disk triangle, then project.
    let seed = 0;
    let planar = layout_euclidean(&disk, &radii, seed);
    let mut positions: Vec<Option<[f64; 3]>> = planar
        .iter()
        .map(|p| p.map(stereographic_to_sphere))
        .collect();
    positions[v_inf] = Some(NORTH_POLE);

    SphereLayout {
        infinity_vertex: v_inf,
        positions,
        packing,
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
    fn hexagon_layout_is_a_regular_hexagon() {
        // Unit hexagonal flower: the 6 boundary centres must land on a regular
        // hexagon of circumradius r_center + r_boundary = 1 + 1 = 2.
        let tris = vec![
            [0, 1, 2], [0, 2, 3], [0, 3, 4],
            [0, 4, 5], [0, 5, 6], [0, 6, 1],
        ];
        let complex = PackingComplex::new(7, tris);
        let radii = vec![1.0; 7]; // exact packing (center already solves to 1)
        let pos = layout_euclidean(&complex, &radii, 0);
        assert_eq!(pos[0].unwrap(), [0.0, 0.0], "center at origin");
        let mut angles: Vec<f64> = Vec::new();
        for v in 1..=6 {
            let [x, y] = pos[v].expect("placed");
            assert!(((x * x + y * y).sqrt() - 2.0).abs() < 1e-9, "circumradius");
            angles.push(y.atan2(x).rem_euclid(2.0 * PI));
        }
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 0..6 {
            let expected = k as f64 * PI / 3.0;
            assert!((angles[k] - expected).abs() < 1e-9, "vertex angle {k}: {}", angles[k]);
        }
    }

    #[test]
    fn layout_tangencies_hold() {
        // Every laid triangle edge must realise external tangency |cᵢ−cⱼ| = rᵢ+rⱼ.
        let tris: Vec<[usize; 3]> = (0..6).map(|i| [0, 1 + i, 1 + (i + 1) % 6]).collect();
        let complex = PackingComplex::new(7, tris);
        let radii = vec![1.0; 7];
        let pos = layout_euclidean(&complex, &radii, 0);
        for t in &complex.triangles {
            let [a, b, c] = *t;
            for &(u, v) in &[(a, b), (b, c), (a, c)] {
                let (pu, pv) = (pos[u].unwrap(), pos[v].unwrap());
                let d = ((pu[0] - pv[0]).powi(2) + (pu[1] - pv[1]).powi(2)).sqrt();
                assert!((d - (radii[u] + radii[v])).abs() < 1e-9, "tangency {u}-{v}");
            }
        }
    }

    #[test]
    fn stereographic_known_points_and_roundtrip() {
        // Origin -> south pole, |z|=1 -> equator, big -> near north pole.
        let s = stereographic_to_sphere([0.0, 0.0]);
        assert!((s[2] + 1.0).abs() < 1e-12, "origin -> south pole");
        let e = stereographic_to_sphere([1.0, 0.0]);
        assert!(e[2].abs() < 1e-12, "unit circle -> equator");
        let far = stereographic_to_sphere([1e6, 0.0]);
        assert!(far[2] > 1.0 - 1e-6, "far -> near north pole");
        // On the sphere.
        for z in [[0.3, -0.7], [2.0, 1.5], [-1.2, 0.4]] {
            let p = stereographic_to_sphere(z);
            assert!((p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - 1.0).abs() < 1e-12);
            let back = stereographic_to_plane(p);
            assert!((back[0] - z[0]).abs() < 1e-9 && (back[1] - z[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn sphere_layout_of_hexagon_cycle_is_on_the_sphere() {
        // A non-degenerate leaf-free genus-0 dessin: the bipartite 6-cycle
        // (3 black + 3 white, all valence 2; two valence-3 faces). σ0=(5 0)(1 2)(3 4),
        // σ1=(0 1)(2 3)(4 5) ⇒ σ∞ = two 3-cycles. No leaves and no monogons, so the
        // complement of one vertex star is a connected disk covering every other
        // vertex. End to end: everything lands on the unit sphere, ∞ at the pole.
        let s0 = Permutation::from_cycles(6, &[vec![5, 0], vec![1, 2], vec![3, 4]]).unwrap();
        let s1 = Permutation::from_cycles(6, &[vec![0, 1], vec![2, 3], vec![4, 5]]).unwrap();
        let tri = flag_triangulation(&s0, &s1).unwrap();
        assert_eq!(tri.euler_characteristic(), 2, "genus 0");
        assert_eq!((tri.n_black, tri.n_white, tri.n_face), (3, 3, 2));

        let layout = sphere_layout(&tri, &PackingConfig::default());
        assert!(layout.packing.converged, "disk relaxation converged");
        assert_eq!(layout.positions[layout.infinity_vertex].unwrap(), NORTH_POLE);
        let mut placed = 0;
        for p in layout.positions.iter().flatten() {
            let norm = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-9, "on unit sphere");
            placed += 1;
        }
        assert_eq!(placed, tri.n_vertices(), "all vertices placed");
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
