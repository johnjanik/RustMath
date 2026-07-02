//! Flag-native circle packing — lay the packing out by walking the GEM
//! involutions, not the vertex-collapsed complex.
//!
//! The audit finding: collapsing the 4E flags to `[vertex, mid, face]` triples is
//! globally non-simplicial for a dessin with valence-1 features (leaves, monogons),
//! so a layout that propagates across *shared vertex pairs* gets stuck (it only
//! reached 5 of 14 vertices on the hexagon, and cannot traverse the M24 dessin).
//!
//! The fix is to walk the **flag adjacency** given by the three involutions
//! (`nu_end`, `nu_side`, `nu_edge`): each takes a flag to the flag across one of
//! its three edges. That graph is connected (the dessin is transitive), so the
//! walk places every orbit vertex — including leaves and monogon-face centers —
//! even though their vertex triples collapse.
//!
//! Angle-sum relaxation is unchanged: the per-orbit angle sum over incident flags
//! is already correct (the collapse never broke it, only the layout). Degree-2 leaf
//! orbits (σ0's fixed points) cannot reach `2π` and are held as *carried points*
//! (fixed radius) — their positions still come from the walk, good enough to seed
//! Newton.

use crate::belyi::flags::FlagTriangulation;
use crate::belyi::packing::{
    relax_euclidean, stereographic_to_sphere, PackingComplex, PackingConfig, PackingResult,
    NORTH_POLE,
};
use std::f64::consts::PI;

/// A flag-native spherical layout of a dessin's triangulation vertices.
#[derive(Debug, Clone)]
pub struct FlagLayout {
    pub infinity_vertex: usize,
    pub positions_plane: Vec<Option<[f64; 2]>>,
    pub positions_sphere: Vec<Option<[f64; 3]>>,
    pub radii: Vec<f64>,
    pub packing: PackingResult,
    /// Degree-2 orbit vertices held as carried points (not solved to 2π).
    pub carried_leaves: Vec<usize>,
}

fn place_third(a: [f64; 2], b: [f64; 2], da: f64, db: f64, side: f64) -> [f64; 2] {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let d = (dx * dx + dy * dy).sqrt();
    let (ux, uy) = (dx / d, dy / d);
    let t = (da * da - db * db + d * d) / (2.0 * d);
    let h = (da * da - t * t).max(0.0).sqrt();
    [a[0] + t * ux - side * h * uy, a[1] + t * uy + side * h * ux]
}

fn cross(o: [f64; 2], p: [f64; 2], q: [f64; 2]) -> f64 {
    (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
}

/// The three flag corners as orbit ids `(vertex, mid, face)`.
fn corners(tri: &FlagTriangulation, f: usize) -> (usize, usize, usize) {
    let c = &tri.corners[f];
    (c.vertex, c.midpoint, c.face)
}

/// For an involution edge, return `(shared_a, shared_b, third_of_g, third_of_f)` as
/// orbit ids, where `f -> g` across that edge. `nu_end` shares (mid, face) and
/// changes the vertex; `nu_side` shares (vertex, mid) and changes the face;
/// `nu_edge` shares (vertex, face) and changes the mid.
fn edge_roles(tri: &FlagTriangulation, f: usize, g: usize, which: usize) -> (usize, usize, usize, usize) {
    let (vf, mf, ff) = corners(tri, f);
    let (vg, mg, fg) = corners(tri, g);
    match which {
        0 => (mf, ff, vg, vf), // nu_end
        1 => (vf, mf, fg, ff), // nu_side
        _ => (vf, ff, mg, mf), // nu_edge
    }
}

/// Lay out orbit-vertex positions in the plane by walking the flag adjacency over
/// the `active` flags, starting from `seed`. Positions of the three corners come
/// from the tangent-circle geometry; each orbit vertex keeps its first placement.
pub fn flag_walk_layout(
    tri: &FlagTriangulation,
    radii: &[f64],
    active: &[bool],
    seed: usize,
) -> Vec<Option<[f64; 2]>> {
    let n_orbits = tri.n_vertices();
    let mut pos: Vec<Option<[f64; 2]>> = vec![None; n_orbits];
    let n_flags = tri.n_flags();

    // Seed placement: vertex at origin, mid on +x, face by tangency.
    let (v, m, fa) = corners(tri, seed);
    pos[v] = Some([0.0, 0.0]);
    pos[m] = Some([radii[v] + radii[m], 0.0]);
    pos[fa] = Some(place_third(
        pos[v].unwrap(),
        pos[m].unwrap(),
        radii[v] + radii[fa],
        radii[m] + radii[fa],
        1.0,
    ));

    // Fixpoint sweep: across every active involution edge f→g, if the two shared
    // corners are placed and g's opposite corner is not, place it on the side away
    // from f's opposite corner. This catches corners (e.g. leaf vertices) that are
    // only ever the "third" across one particular involution.
    let neighbors = [&tri.nu_end, &tri.nu_side, &tri.nu_edge];
    loop {
        let mut progress = false;
        for f in 0..n_flags {
            if !active[f] {
                continue;
            }
            for (which, nu) in neighbors.iter().enumerate() {
                let g = nu[f];
                if !active[g] {
                    continue;
                }
                let (a, b, q, third_f) = edge_roles(tri, f, g, which);
                if pos[q].is_some() || pos[a].is_none() || pos[b].is_none() || pos[third_f].is_none()
                {
                    continue;
                }
                let side = -cross(pos[a].unwrap(), pos[b].unwrap(), pos[third_f].unwrap()).signum();
                pos[q] = Some(place_third(
                    pos[a].unwrap(),
                    pos[b].unwrap(),
                    radii[a] + radii[q],
                    radii[b] + radii[q],
                    if side == 0.0 { 1.0 } else { side },
                ));
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }
    pos
}

/// Flags not incident to `v_inf` — the punctured disk.
fn active_flags_of(tri: &FlagTriangulation, v_inf: usize) -> Vec<bool> {
    (0..tri.n_flags())
        .map(|f| {
            let (v, m, fa) = corners(tri, f);
            v != v_inf && m != v_inf && fa != v_inf
        })
        .collect()
}

/// Is the active-flag subgraph connected under the three involutions? Puncturing a
/// high-valence vertex (e.g. a 12-valent white vertex) can split the disk's flag
/// graph into several components, which would strand most of the layout.
fn active_flags_connected(tri: &FlagTriangulation, active: &[bool]) -> bool {
    let n_active = active.iter().filter(|&&a| a).count();
    if n_active == 0 {
        return false;
    }
    let neighbors = [&tri.nu_end, &tri.nu_side, &tri.nu_edge];
    let seed = (0..tri.n_flags()).find(|&f| active[f]).unwrap();
    let mut seen = vec![false; tri.n_flags()];
    seen[seed] = true;
    let mut count = 1;
    let mut stack = vec![seed];
    while let Some(f) = stack.pop() {
        for nu in neighbors {
            let g = nu[f];
            if active[g] && !seen[g] {
                seen[g] = true;
                count += 1;
                stack.push(g);
            }
        }
    }
    count == n_active
}

/// Build a flag-native spherical packing of a genus-0 dessin: puncture the highest
/// degree orbit *whose removal leaves the disk flag-graph connected* (the natural ∞
/// is a face pole; a 12-valent white vertex would disconnect the disk), relax the
/// disk (interior → 2π; degree-2 leaves carried), flag-walk the layout, and project
/// to the sphere with `∞` at the pole.
pub fn flag_pack(tri: &FlagTriangulation, cfg: &PackingConfig) -> FlagLayout {
    let full = PackingComplex::from_flags(tri);
    let n = full.n_vertices;
    let total = full.triangles.len();

    let v_inf = (0..n)
        .filter(|&v| full.degree(v) < total && active_flags_connected(tri, &active_flags_of(tri, v)))
        .max_by_key(|&v| full.degree(v))
        .unwrap_or(0);

    let active = active_flags_of(tri, v_inf);

    // Disk complex from active flags, for the relaxation angle sums.
    let disk_tris: Vec<[usize; 3]> = (0..tri.n_flags())
        .filter(|&f| active[f])
        .map(|f| {
            let (v, m, fa) = corners(tri, f);
            [v, m, fa]
        })
        .collect();
    let disk = PackingComplex::new(n, disk_tris);

    let carried_leaves: Vec<usize> = (0..n).filter(|&u| full.degree(u) < 3).collect();
    let mut is_carried = vec![false; n];
    for &u in &carried_leaves {
        is_carried[u] = true;
    }
    // Boundary = orbits adjacent to v_inf (they lost triangles to the puncture).
    let mut boundary = vec![false; n];
    for f in 0..tri.n_flags() {
        let (v, m, fa) = corners(tri, f);
        if v == v_inf || m == v_inf || fa == v_inf {
            for u in [v, m, fa] {
                if u != v_inf {
                    boundary[u] = true;
                }
            }
        }
    }

    let interior: Vec<bool> = (0..n)
        .map(|u| u != v_inf && !boundary[u] && !is_carried[u] && disk.degree(u) >= 3)
        .collect();

    let mut radii = vec![1.0_f64; n];
    let targets = vec![2.0 * PI; n];
    let packing = relax_euclidean(&disk, &mut radii, &interior, &targets, cfg);

    // The puncture defines the disk for the RELAXATION (radii). The LAYOUT walks
    // ALL flags so nothing is stranded — puncturing a face would orphan the leaves
    // and monogon-vertices private to it (they have no active flag). The developing
    // map has a small defect at v_inf (its angle sum was not solved to 2π), so we
    // override v_inf to the pole afterwards; that neighbourhood is refined later.
    let seed = (0..tri.n_flags()).find(|&f| active[f]).unwrap_or(0);
    let all_flags = vec![true; tri.n_flags()];
    let positions_plane = flag_walk_layout(tri, &radii, &all_flags, seed);
    let mut positions_sphere: Vec<Option<[f64; 3]>> = positions_plane
        .iter()
        .map(|p| p.map(stereographic_to_sphere))
        .collect();
    positions_sphere[v_inf] = Some(NORTH_POLE);

    FlagLayout {
        infinity_vertex: v_inf,
        positions_plane,
        positions_sphere,
        radii,
        packing,
        carried_leaves,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belyi::flags::flag_triangulation;
    use crate::belyi::monodromy::Permutation;

    // The genuine M24 [2,12,5] triple.
    const SIGMA0: [usize; 24] = [
        0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6,
    ];
    const SIGMA1: [usize; 24] = [
        14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6,
    ];

    fn m24_tri() -> FlagTriangulation {
        let s0 = Permutation::new(SIGMA0.to_vec()).unwrap();
        let s1 = Permutation::new(SIGMA1.to_vec()).unwrap();
        flag_triangulation(&s0, &s1).unwrap()
    }

    #[test]
    fn flag_walk_places_every_orbit_on_hexagon_cycle() {
        // Leaf-free dessin: flag-walk should place all vertices (the old vertex-edge
        // layout stalled at 5 of 14).
        let s0 = Permutation::from_cycles(6, &[vec![5, 0], vec![1, 2], vec![3, 4]]).unwrap();
        let s1 = Permutation::from_cycles(6, &[vec![0, 1], vec![2, 3], vec![4, 5]]).unwrap();
        let tri = flag_triangulation(&s0, &s1).unwrap();
        let layout = flag_pack(&tri, &PackingConfig::default());
        let placed = layout.positions_sphere.iter().filter(|p| p.is_some()).count();
        assert_eq!(placed, tri.n_vertices(), "all orbits placed");
    }

    #[test]
    fn flag_walk_traverses_the_full_m24_dessin() {
        // The real target: the collapse-based layout could not traverse this at all.
        // Flag-walk must place every one of the 50 orbit vertices on the unit sphere.
        let tri = m24_tri();
        assert_eq!(tri.n_vertices(), 50);
        let layout = flag_pack(&tri, &PackingConfig::default());

        assert_eq!(layout.carried_leaves.len(), 8, "8 leaf blacks are carried");
        assert_eq!(
            layout.positions_sphere[layout.infinity_vertex].unwrap(),
            NORTH_POLE
        );
        let mut placed = 0;
        for p in layout.positions_sphere.iter().flatten() {
            let norm = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-9, "on unit sphere: {p:?}");
            assert!(p[0].is_finite() && p[1].is_finite() && p[2].is_finite());
            placed += 1;
        }
        assert_eq!(placed, 50, "all 50 orbit vertices placed (vs 5 before)");
    }
}


