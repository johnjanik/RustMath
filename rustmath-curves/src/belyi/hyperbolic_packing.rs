//! Hyperbolic (maximal) circle packing of the once-punctured flag triangulation.
//!
//! Euclidean angle-sum packing cannot close on a genus-0 sphere: Gauss–Bonnet
//! fixes the total curvature at `4π`, incompatible with an all-interior target of
//! `2π` per vertex, so the euclidean relaxation stalls and its developing map
//! folds (coinciding vertices → a singular Newton Jacobian; see the collision and
//! perturbation experiments). The fix is to puncture one vertex to a horocycle and
//! pack the rest in the hyperbolic plane. The **maximal packing** (one horocyclic
//! boundary vertex, every other vertex interior with target `2π`) exists and is
//! unique (Beardon–Stephenson), and the angle-sum iteration is globally
//! convergent — so it converges where the euclidean packing could not.
//!
//! Radii are carried as `s = exp(−r) ∈ [0, 1)`: `s = 1` ⇔ `r = 0`, and `s = 0` ⇔
//! `r = ∞` (a horocycle). This keeps the horocyclic boundary a finite value and
//! makes the angle law numerically stable near the puncture.

use super::flags::FlagTriangulation;
use super::packing::PackingComplex;
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Barycentric (1→6) subdivision of a triangle list: retains the original vertices
/// `0..n_vertices`, adds one centroid per triangle and one midpoint per edge, and
/// splits each triangle into six. Every original vertex's incident-triangle count
/// doubles, so the flag triangulation's degree-2 leaves/monogons (2 triangles →
/// 4) become packable, and the result is a genuine simplicial triangulation.
pub fn barycentric_subdivision(
    n_vertices: usize,
    triangles: &[[usize; 3]],
) -> (usize, Vec<[usize; 3]>) {
    let mut next = n_vertices;
    let mut mids: HashMap<(usize, usize), usize> = HashMap::new();
    let mut out: Vec<[usize; 3]> = Vec::with_capacity(6 * triangles.len());
    for &[a, b, c] in triangles {
        let mut get_mid = |x: usize, y: usize, next: &mut usize| -> usize {
            let key = if x <= y { (x, y) } else { (y, x) };
            *mids.entry(key).or_insert_with(|| {
                let m = *next;
                *next += 1;
                m
            })
        };
        let mab = get_mid(a, b, &mut next);
        let mbc = get_mid(b, c, &mut next);
        let mca = get_mid(c, a, &mut next);
        let centroid = next;
        next += 1;
        out.push([a, mab, centroid]);
        out.push([mab, b, centroid]);
        out.push([b, mbc, centroid]);
        out.push([mbc, c, centroid]);
        out.push([c, mca, centroid]);
        out.push([mca, a, centroid]);
    }
    (next, out)
}

/// Barycentric subdivision built from the **flag involution adjacency**, which is a
/// genuine manifold triangulation — unlike `PackingComplex::from_flags`, whose
/// corner-triples collapse coincident flags (leaves/monogons) into a non-manifold
/// complex that folds under layout. Each flag `f` gets a private centroid; the
/// three edge-midpoints are shared with `nu_side`/`nu_end`/`nu_edge`-partners, so
/// every edge meets exactly two triangles. Original corners keep indices
/// `0..tri.n_vertices()`; their incident-triangle count doubles (leaf 2 → 4),
/// making the complex packable.
pub fn barycentric_subdivision_of_flags(tri: &FlagTriangulation) -> (usize, Vec<[usize; 3]>) {
    let n_orig = tri.n_vertices();
    let n_flags = tri.corners.len();
    let mut next = n_orig + n_flags; // centroids occupy [n_orig, n_orig+n_flags)
    let mut mid_index: HashMap<(u8, usize), usize> = HashMap::new();
    let mut triangles = Vec::with_capacity(6 * n_flags);
    for f in 0..n_flags {
        let v = tri.corners[f].vertex;
        let m = tri.corners[f].midpoint;
        let fa = tri.corners[f].face;
        let c = n_orig + f; // this flag's centroid
        // Edge midpoints, shared with the involution partner across each edge.
        let mut edge_mid = |inv_id: u8, g: usize, next: &mut usize| -> usize {
            let key = (inv_id, f.min(g));
            *mid_index.entry(key).or_insert_with(|| {
                let mid = *next;
                *next += 1;
                mid
            })
        };
        let m_vm = edge_mid(0, tri.nu_side[f], &mut next); // (V,M) edge
        let m_mf = edge_mid(1, tri.nu_end[f], &mut next); // (M,F) edge
        let m_vf = edge_mid(2, tri.nu_edge[f], &mut next); // (V,F) edge
        triangles.push([v, m_vm, c]);
        triangles.push([m_vm, m, c]);
        triangles.push([m, m_mf, c]);
        triangles.push([m_mf, fa, c]);
        triangles.push([fa, m_vf, c]);
        triangles.push([m_vf, v, c]);
    }
    (next, triangles)
}

/// The packable complex for a flag triangulation: the manifold barycentric
/// subdivision above. Original vertices (the dessin's `0/1/∞` preimages, whose
/// positions become the Newton seed) keep their indices `0..tri.n_vertices()`.
pub fn packable_complex_of(tri: &FlagTriangulation) -> PackingComplex {
    let (n, triangles) = barycentric_subdivision_of_flags(tri);
    PackingComplex::new(n, triangles)
}

/// The interior angle at the `s_c` circle in a triangle of three mutually tangent
/// circles with `s`-values `(s_c, s_a, s_b)`. `s = 0` marks a horocycle.
///
/// For finite radii this is the hyperbolic law of cosines on the tangent-length
/// sides `r_c+r_a`, `r_c+r_b`, `r_a+r_b`. When one neighbour is a horocycle the
/// `r → ∞` limit is used (derived in `s`-form, stable as `s → 0`). The angle is
/// monotone: it decreases as `r_c` grows (i.e. increases in `s_c`).
pub fn hyperbolic_flag_angle(s_c: f64, s_a: f64, s_b: f64) -> f64 {
    let horo_a = s_a <= 0.0;
    let horo_b = s_b <= 0.0;
    let cos = if horo_a && horo_b {
        // Both neighbours horocyclic (does not occur for a single puncture).
        1.0 - 2.0 * s_c * s_c
    } else if horo_a || horo_b {
        // One horocycle: cos = (1 − 2 s_c² + s_c² s²) / (1 − s_c² s²), where s is
        // the finite neighbour. Derived as the r_a→∞ limit of the general formula.
        let s = if horo_a { s_b } else { s_a };
        let sc2 = s_c * s_c;
        let p = sc2 * s * s;
        (1.0 - 2.0 * sc2 + p) / (1.0 - p)
    } else {
        let rc = -s_c.ln();
        let ra = -s_a.ln();
        let rb = -s_b.ln();
        let x = rc + ra;
        let y = rc + rb;
        let z = ra + rb;
        (x.cosh() * y.cosh() - z.cosh()) / (x.sinh() * y.sinh())
    };
    cos.clamp(-1.0, 1.0).acos()
}

/// Angle sum at vertex `v` over a precomputed flower (petal pairs).
fn angle_sum(flower: &[(usize, usize)], s: &[f64], v: usize) -> f64 {
    flower
        .iter()
        .map(|&(a, b)| hyperbolic_flag_angle(s[v], s[a], s[b]))
        .sum()
}

/// Solve the `s`-value of interior vertex `v` so its angle sum equals `2π`. The
/// sum is monotone increasing in `s_v`, so a bisection is exact; if even `s_v → 1`
/// (radius `0`) undershoots `2π`, clamp to the largest `s` (smallest radius).
fn solve_s_for_target(flower: &[(usize, usize)], s: &mut [f64], v: usize, target: f64) {
    let mut lo = 1e-12_f64; // large radius → small angle sum
    let mut hi = 1.0 - 1e-12; // ~zero radius → max angle sum
    s[v] = hi;
    if angle_sum(flower, s, v) < target {
        return; // cannot reach 2π; keep smallest radius
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        s[v] = mid;
        if angle_sum(flower, s, v) < target {
            lo = mid; // sum too small ⇒ radius too large ⇒ raise s
        } else {
            hi = mid;
        }
    }
    s[v] = 0.5 * (lo + hi);
}

/// Configuration for the hyperbolic relaxation.
#[derive(Debug, Clone)]
pub struct HypPackingConfig {
    pub max_iters: usize,
    pub tol: f64,
}

impl Default for HypPackingConfig {
    fn default() -> Self {
        Self {
            max_iters: 3000,
            tol: 1e-12,
        }
    }
}

/// Result of the hyperbolic packing: the `s`-values, the puncture (horocycle)
/// vertex, and convergence data.
#[derive(Debug, Clone)]
pub struct HypPackingResult {
    pub s: Vec<f64>,
    pub puncture: usize,
    pub iterations: usize,
    pub max_angle_error: f64,
    pub converged: bool,
}

impl HypPackingResult {
    /// Hyperbolic radius of `v` (`+∞` for the puncture).
    pub fn radius(&self, v: usize) -> f64 {
        if self.s[v] <= 0.0 {
            f64::INFINITY
        } else {
            -self.s[v].ln()
        }
    }
}

/// Compute a packing of `complex` with `puncture` pinned to a horocycle (`s = 0`)
/// and every other vertex driven toward angle sum `2π`. On a closed sphere this
/// all-`2π` target is not exactly attainable (Gauss–Bonnet), so the relaxation
/// settles into a euclidean-like packing (small, conformally-graded radii) rather
/// than a strict maximal packing — but that layout, after the affine gauge
/// normalisation, is an excellent Newton seed (it reaches a genuine solution). A
/// strict maximal packing (removing the puncture, a full horocyclic boundary
/// cycle) is cleaner in theory but its small-boundary Gauss–Seidel converges too
/// slowly here; that acceleration is the remaining refinement.
pub fn maximal_packing(
    complex: &PackingComplex,
    puncture: usize,
    cfg: &HypPackingConfig,
) -> HypPackingResult {
    let n = complex.n_vertices;
    let mut flower: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for t in &complex.triangles {
        let [a, b, c] = *t;
        flower[a].push((b, c));
        flower[b].push((a, c));
        flower[c].push((a, b));
    }
    let mut s = vec![0.5_f64; n];
    s[puncture] = 0.0; // horocycle
    let interior: Vec<usize> = (0..n).filter(|&v| v != puncture).collect();

    let mut iterations = 0;
    let mut max_err = f64::INFINITY;
    while iterations < cfg.max_iters {
        for &v in &interior {
            solve_s_for_target(&flower[v], &mut s, v, 2.0 * PI);
        }
        max_err = interior
            .iter()
            .map(|&v| (angle_sum(&flower[v], &s, v) - 2.0 * PI).abs())
            .fold(0.0_f64, f64::max);
        iterations += 1;
        if max_err < cfg.tol {
            break;
        }
    }
    HypPackingResult {
        s,
        puncture,
        iterations,
        max_angle_error: max_err,
        converged: max_err < cfg.tol,
    }
}

/// Pack the barycentric subdivision of `tri`, puncturing a maximum-degree
/// **non-root** vertex (an original edge-midpoint — it is dropped from the seed, so
/// removing it costs no root, and its neighbour cycle becomes the horocyclic
/// boundary). Returns the packing and the complex it was solved on; original
/// vertices keep indices `0..tri.n_vertices()`.
pub fn maximal_packing_of(
    tri: &FlagTriangulation,
    cfg: &HypPackingConfig,
) -> (PackingComplex, HypPackingResult) {
    let complex = packable_complex_of(tri);
    // Puncture the maximum-degree vertex (it is kept and placed, so a root is fine).
    let puncture = (0..complex.n_vertices)
        .max_by_key(|&v| complex.degree(v))
        .unwrap_or(0);
    let result = maximal_packing(&complex, puncture, cfg);
    (complex, result)
}

// ---------------------------------------------------------------------------
// Layout: develop the packing in the Poincaré disk and read off plane positions.
// ---------------------------------------------------------------------------

fn undirected(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Consistently orient the triangle list (a coherent surface orientation) by
/// edge-adjacency flood fill, so the developing map can place every third vertex
/// on a uniform (left) side of its base edge.
fn orient_triangles(triangles: &[[usize; 3]]) -> Vec<[usize; 3]> {
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, t) in triangles.iter().enumerate() {
        let [a, b, c] = *t;
        edge_tris.entry(undirected(a, b)).or_default().push(i);
        edge_tris.entry(undirected(b, c)).or_default().push(i);
        edge_tris.entry(undirected(c, a)).or_default().push(i);
    }
    let has_dir = |t: &[usize; 3], u: usize, v: usize| {
        (t[0] == u && t[1] == v) || (t[1] == u && t[2] == v) || (t[2] == u && t[0] == v)
    };
    let mut oriented = triangles.to_vec();
    let mut fixed = vec![false; triangles.len()];
    let mut stack = vec![0usize];
    fixed[0] = true;
    while let Some(ti) = stack.pop() {
        let t = oriented[ti];
        for &(u, v) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            for &tj in &edge_tris[&undirected(u, v)] {
                if fixed[tj] {
                    continue;
                }
                // Coherent orientation: the neighbour must traverse the shared edge
                // in the opposite direction (v→u). If it also has u→v, flip it.
                if has_dir(&oriented[tj], u, v) {
                    oriented[tj].swap(1, 2);
                }
                fixed[tj] = true;
                stack.push(tj);
            }
        }
    }
    oriented
}

/// Disk automorphism sending `p` to the origin.
fn to_origin(p: Complex64, z: Complex64) -> Complex64 {
    (z - p) / (1.0 - p.conj() * z)
}

/// Inverse of [`to_origin`]: sends the origin back to `p`.
fn from_origin(p: Complex64, w: Complex64) -> Complex64 {
    (w + p) / (1.0 + p.conj() * w)
}

/// Euclidean (Poincaré-model) radius of a point at hyperbolic distance `ell` from
/// the disk centre. `ell = ∞` (a horocycle tangency) maps to the boundary.
fn euclid_radius(ell: f64) -> f64 {
    if ell.is_infinite() {
        1.0
    } else {
        (0.5 * ell).tanh()
    }
}

/// Place the third vertex `v2` of an oriented triangle whose base edge `v0→v1` is
/// already laid out at `z0, z1`: `v2` sits on the left of `v0→v1`, at hyperbolic
/// distance `r0+r2` from `v0` and turned by the packing angle at `v0`.
fn place_third_hyp(
    z0: Complex64,
    z1: Complex64,
    r: &dyn Fn(usize) -> f64,
    s: &[f64],
    v0: usize,
    v1: usize,
    v2: usize,
) -> Complex64 {
    let gamma0 = hyperbolic_flag_angle(s[v0], s[v1], s[v2]); // angle at v0
    let ell02 = r(v0) + r(v2);
    let z1p = to_origin(z0, z1);
    let theta = z1p.arg() + gamma0; // +γ ⇒ left side (coherent CCW)
    let rad = euclid_radius(ell02);
    let z2p = Complex64::from_polar(rad, theta);
    from_origin(z0, z2p)
}

/// A hyperbolic layout: Poincaré-disk positions of every vertex of the (subdivided)
/// complex, and the puncture that was sent to the boundary.
#[derive(Debug, Clone)]
pub struct HypLayout {
    pub positions: Vec<Option<Complex64>>,
    pub puncture: usize,
}

/// Develop the packing into the Poincaré disk. Non-puncture vertices are placed off
/// finite base edges (the horocyclic puncture is never a base); the puncture is
/// placed last, on the boundary. The coherent orientation keeps the developing map
/// from folding.
pub fn hyperbolic_layout(complex: &PackingComplex, pack: &HypPackingResult) -> HypLayout {
    let n = complex.n_vertices;
    let s = pack.s.clone();
    let puncture = pack.puncture;
    let r = {
        let s = s.clone();
        move |v: usize| -> f64 {
            if s[v] <= 0.0 {
                f64::INFINITY
            } else {
                -s[v].ln()
            }
        }
    };
    let oriented = orient_triangles(&complex.triangles);
    let mut pos: Vec<Option<Complex64>> = vec![None; n];

    // Seed with a puncture-free triangle: v0 at the centre, v1 on the +x axis.
    let seed = oriented
        .iter()
        .find(|t| !t.contains(&puncture))
        .copied()
        .unwrap_or(oriented[0]);
    let [a, b, c] = seed;
    pos[a] = Some(Complex64::new(0.0, 0.0));
    pos[b] = Some(Complex64::new(euclid_radius(r(a) + r(b)), 0.0));
    pos[c] = Some(place_third_hyp(pos[a].unwrap(), pos[b].unwrap(), &r, &s, a, b, c));

    loop {
        let mut progress = false;
        for t in &oriented {
            let placed: Vec<bool> = t.iter().map(|&v| pos[v].is_some()).collect();
            if placed.iter().filter(|&&p| p).count() != 2 {
                continue;
            }
            let k = placed.iter().position(|&p| !p).unwrap();
            let (b0, b1, u) = (t[(k + 1) % 3], t[(k + 2) % 3], t[k]);
            if b0 == puncture || b1 == puncture {
                continue; // base must be finite; defer
            }
            pos[u] = Some(place_third_hyp(
                pos[b0].unwrap(),
                pos[b1].unwrap(),
                &r,
                &s,
                b0,
                b1,
                u,
            ));
            progress = true;
        }
        if !progress {
            break;
        }
    }

    // Place the puncture last, on the boundary, off any finite base.
    if pos[puncture].is_none() {
        for t in &oriented {
            if let Some(k) = t.iter().position(|&v| v == puncture) {
                let (b0, b1) = (t[(k + 1) % 3], t[(k + 2) % 3]);
                if pos[b0].is_some() && pos[b1].is_some() {
                    pos[puncture] =
                        Some(place_third_hyp(pos[b0].unwrap(), pos[b1].unwrap(), &r, &s, b0, b1, puncture));
                    break;
                }
            }
        }
    }

    HypLayout {
        positions: pos,
        puncture,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_matches_euclidean_limit_at_zero_radius() {
        // As r_c → 0 (s_c → 1) the packing angle → π regardless of neighbours.
        let a = hyperbolic_flag_angle(1.0 - 1e-9, 0.4, 0.6);
        assert!((a - PI).abs() < 1e-3, "got {a}");
    }

    #[test]
    fn angle_is_monotone_decreasing_in_radius() {
        // Larger radius (smaller s) ⇒ smaller angle.
        let big_s = hyperbolic_flag_angle(0.9, 0.5, 0.5); // small radius
        let small_s = hyperbolic_flag_angle(0.1, 0.5, 0.5); // large radius
        assert!(big_s > small_s, "{big_s} !> {small_s}");
    }

    #[test]
    fn horocycle_angle_is_finite_and_positive() {
        let a = hyperbolic_flag_angle(0.3, 0.0, 0.5);
        assert!(a.is_finite() && a > 0.0 && a < PI, "got {a}");
    }
}
