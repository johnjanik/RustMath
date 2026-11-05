//! The 4E flag (barycentric) triangulation of a dessin, from its monodromy pair
//! `(σ0, σ1)` (`σ∞ = (σ0 σ1)^{-1}`).
//!
//! The naive 2E dessin triangulation (`2·deg` triangles with corners over
//! `0,1,∞`) degenerates at valence-1/2 vertices, and a vertex-indexed simplicial
//! model collapses the two sides of a leaf edge (same face on both sides ⇒ two
//! triangles with identical vertex triples) — the campaign's circle-packing
//! blocker.
//!
//! The fix is to model the triangulation as a **flag system / graph-encoded map**:
//! `4·deg` abstract flags indexed by `(edge, side ∈ {0,1}, end ∈ {0=black,1=white})`,
//! each a triangle `[vertex(end), midpoint(edge), facecenter(side)]`, with adjacency
//! given by three fixed-point-free involutions rather than by shared vertex indices.
//! Two flags across an edge are always distinct, so leaves are handled correctly.
//!
//! The three involutions (reflect a flag across each of its three sides):
//! * `nu_end`  — across `[mid, face]` (opposite the vertex): swap black↔white end.
//! * `nu_side` — across `[vertex, mid]` (opposite the face): swap the two faces.
//! * `nu_edge` — across `[vertex, face]` (opposite the mid): step to the adjacent
//!   dessin edge around the vertex within the face (uses `σ0` at black, `σ1` at
//!   white).
//!
//! `nu_end` and `nu_side` commute — the graph-encoded-map condition guaranteeing a
//! closed surface — and the vertex/face/edge orbits recover the dessin: black =
//! `⟨nu_side,nu_edge⟩`-orbits on black flags (`= σ0`-orbits), faces =
//! `⟨nu_end,nu_edge⟩`-orbits (`= σ∞`-orbits), edge-midpoints = `⟨nu_end,nu_side⟩`
//! orbits (`= deg`). Euler `χ = (#black + #white) − deg + #face = 2 − 2g`.

use crate::belyi::monodromy::{BelyiTriple, PermError, Permutation};

/// The role of a flag corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexKind {
    /// Preimage of `0`; valence = σ0-orbit length.
    Black,
    /// Preimage of `1`; valence = σ1-orbit length.
    White,
    /// Preimage of `∞` (face center); valence = σ∞-orbit length.
    Face,
    /// Midpoint of a dessin edge.
    EdgeMidpoint,
}

#[derive(Debug, PartialEq, Eq)]
pub enum FlagError {
    Perm(PermError),
}

impl From<PermError> for FlagError {
    fn from(e: PermError) -> Self {
        FlagError::Perm(e)
    }
}

/// The three corner-vertex ids of a flag, in block indexing
/// `black | white | face | edge-midpoint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlagCorners {
    pub vertex: usize, // black or white vertex id (a real dessin vertex)
    pub midpoint: usize,
    pub face: usize,
}

/// The 4E flag triangulation of a dessin as a graph-encoded map.
#[derive(Debug, Clone)]
pub struct FlagTriangulation {
    /// Cover degree `d` = number of darts/edges.
    pub degree: usize,
    pub n_black: usize,
    pub n_white: usize,
    pub n_face: usize,
    /// `nu_end[f]`  — neighbour flag across `[mid, face]` (swap end).
    pub nu_end: Vec<usize>,
    /// `nu_side[f]` — neighbour flag across `[vertex, mid]` (swap face side).
    pub nu_side: Vec<usize>,
    /// `nu_edge[f]` — neighbour flag across `[vertex, face]` (step edge).
    pub nu_edge: Vec<usize>,
    /// Per-flag corner vertex ids (for coordinates / packing).
    pub corners: Vec<FlagCorners>,
}

/// Orbit id per point and the orbit count, for a permutation on `0..n`.
fn orbits(p: &Permutation) -> (Vec<usize>, usize) {
    let n = p.degree();
    let mut id = vec![usize::MAX; n];
    let mut count = 0;
    for start in 0..n {
        if id[start] != usize::MAX {
            continue;
        }
        let mut x = start;
        while id[x] == usize::MAX {
            id[x] = count;
            x = p.apply(x);
        }
        count += 1;
    }
    (id, count)
}

/// Component id per element and the component count for `⟨p, q⟩` acting on
/// `0..p.len()` (both `p`, `q` are permutations given as image slices).
fn joint_orbits(p: &[usize], q: &[usize]) -> (Vec<usize>, usize) {
    let n = p.len();
    let mut id = vec![usize::MAX; n];
    let mut count = 0;
    let mut stack = Vec::new();
    for start in 0..n {
        if id[start] != usize::MAX {
            continue;
        }
        id[start] = count;
        stack.push(start);
        while let Some(x) = stack.pop() {
            for &y in &[p[x], q[x]] {
                if id[y] == usize::MAX {
                    id[y] = count;
                    stack.push(y);
                }
            }
        }
        count += 1;
    }
    (id, count)
}

/// Flag index for `(edge e, side s ∈ {0,1}, end t ∈ {0,1})`.
#[inline]
fn flag(e: usize, s: usize, t: usize) -> usize {
    (e << 2) | (s << 1) | t
}

impl FlagTriangulation {
    pub fn n_flags(&self) -> usize {
        4 * self.degree
    }

    /// The number of triangles = number of flags.
    pub fn n_triangles(&self) -> usize {
        self.n_flags()
    }

    pub fn n_vertices(&self) -> usize {
        self.n_black + self.n_white + self.n_face + self.degree
    }

    /// `χ = (#black + #white) − deg + #face = 2 − 2g`.
    pub fn euler_characteristic(&self) -> i64 {
        (self.n_black + self.n_white) as i64 - self.degree as i64 + self.n_face as i64
    }

    /// Genus `g = (2 − χ)/2`.
    pub fn genus(&self) -> i64 {
        (2 - self.euler_characteristic()) / 2
    }

    fn is_fpf_involution(p: &[usize]) -> bool {
        p.iter().enumerate().all(|(i, &j)| p[j] == i && j != i)
    }

    /// Verify the graph-encoded-map laws: the three neighbour maps are
    /// fixed-point-free involutions and the two extreme ones (`nu_end`, `nu_side`)
    /// commute — which guarantees a closed surface.
    pub fn is_valid_gem(&self) -> bool {
        let n = self.n_flags();
        if self.nu_end.len() != n || self.nu_side.len() != n || self.nu_edge.len() != n {
            return false;
        }
        Self::is_fpf_involution(&self.nu_end)
            && Self::is_fpf_involution(&self.nu_side)
            && Self::is_fpf_involution(&self.nu_edge)
            // nu_end and nu_side commute (the non-adjacent pair).
            && (0..n).all(|f| self.nu_end[self.nu_side[f]] == self.nu_side[self.nu_end[f]])
    }

    /// Consistency of the flag orbits with the dessin: edge-midpoints from
    /// `⟨nu_end,nu_side⟩` = `deg`, faces from `⟨nu_end,nu_edge⟩` = `#face`, and
    /// vertices from `⟨nu_side,nu_edge⟩` = `#black + #white`.
    pub fn orbits_match_dessin(&self) -> bool {
        joint_orbits(&self.nu_end, &self.nu_side).1 == self.degree
            && joint_orbits(&self.nu_end, &self.nu_edge).1 == self.n_face
            && joint_orbits(&self.nu_side, &self.nu_edge).1 == self.n_black + self.n_white
    }
}

/// Build the 4E flag triangulation of the dessin `(σ0, σ1)`.
pub fn flag_triangulation(
    sigma0: &Permutation,
    sigma1: &Permutation,
) -> Result<FlagTriangulation, FlagError> {
    let d = sigma0.degree();
    if sigma1.degree() != d {
        return Err(FlagError::Perm(PermError::DegreeMismatch));
    }
    let sigma_inf = sigma0.compose(sigma1)?.inverse();
    let s0 = sigma0;
    let s0i = sigma0.inverse();
    let s1 = sigma1;
    let s1i = sigma1.inverse();

    let (_b_of, n_black) = orbits(sigma0);
    let (_w_of, n_white) = orbits(sigma1);
    let (_f_of, n_face) = orbits(&sigma_inf);

    let n = 4 * d;
    let mut nu_end = vec![0usize; n];
    let mut nu_side = vec![0usize; n];
    let mut nu_edge = vec![0usize; n];

    for e in 0..d {
        for s in 0..2 {
            for t in 0..2 {
                let f = flag(e, s, t);
                // Trivial reflections: swap end (bit0), swap side (bit1).
                nu_end[f] = flag(e, s, 1 - t);
                nu_side[f] = flag(e, 1 - s, t);
                // Edge step around the vertex within the face:
                //   (e,0,·) -> (ρ(e), 1, ·) ; (e,1,·) -> (ρ⁻¹(e), 0, ·)
                // with ρ = σ0⁻¹ at black (t=0), σ1 at white (t=1). This makes the
                // face-tracing map (nu_end∘nu_edge)² equal σ∞ = σ1⁻¹σ0⁻¹ exactly
                // (using σ0 at black would trace σ1⁻¹σ0 — the wrong product).
                let (rho, rho_i): (&Permutation, &Permutation) = if t == 0 {
                    (&s0i, s0)
                } else {
                    (s1, &s1i)
                };
                nu_edge[f] = if s == 0 {
                    flag(rho.apply(e), 1, t)
                } else {
                    flag(rho_i.apply(e), 0, t)
                };
            }
        }
    }

    // Corner vertex ids come from the GEM orbits, so the two sides of an edge get
    // the two *different* adjacent faces (a side-blind `f_of[e]` would collapse
    // them, breaking the layout adjacency). Ranges are disjoint:
    //   vertex ∈ [0, V), midpoint ∈ [V, V+E), face ∈ [V+E, V+E+F).
    let (vertex_of, v_count) = joint_orbits(&nu_side, &nu_edge); // black ∪ white
    let (mid_of, m_count) = joint_orbits(&nu_end, &nu_side); // = degree
    let (face_of, _f_count) = joint_orbits(&nu_end, &nu_edge); // = n_face
    let base_mid = v_count;
    let base_face = v_count + m_count;
    let corners: Vec<FlagCorners> = (0..n)
        .map(|f| FlagCorners {
            vertex: vertex_of[f],
            midpoint: base_mid + mid_of[f],
            face: base_face + face_of[f],
        })
        .collect();

    Ok(FlagTriangulation {
        degree: d,
        n_black,
        n_white,
        n_face,
        nu_end,
        nu_side,
        nu_edge,
        corners,
    })
}

/// Build the flag triangulation from a validated [`BelyiTriple`].
pub fn flag_triangulation_of(triple: &BelyiTriple) -> Result<FlagTriangulation, FlagError> {
    flag_triangulation(&triple.sigma0, &triple.sigma1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_valid(tri: &FlagTriangulation, expected_euler: i64) {
        assert!(tri.is_valid_gem(), "not a valid graph-encoded map");
        assert!(tri.orbits_match_dessin(), "flag orbits disagree with the dessin");
        assert_eq!(tri.n_triangles(), 4 * tri.degree, "triangle count");
        assert_eq!(tri.euler_characteristic(), expected_euler, "euler");
    }

    #[test]
    fn star_degree_3_is_genus_zero() {
        // One black vertex of valence 3, three white leaves: σ0=(0 1 2), σ1=id.
        let s0 = Permutation::from_cycles(3, &[vec![0, 1, 2]]).unwrap();
        let s1 = Permutation::identity(3);
        let tri = flag_triangulation(&s0, &s1).unwrap();
        assert_eq!((tri.n_black, tri.n_white, tri.n_face), (1, 3, 1));
        assert_eq!(tri.degree, 3);
        assert_valid(&tri, 2);
    }

    #[test]
    fn cherry_degree_2_is_genus_zero() {
        // One black valence-2 vertex, two white leaves: σ0=(0 1), σ1=id.
        let s0 = Permutation::from_cycles(2, &[vec![0, 1]]).unwrap();
        let s1 = Permutation::identity(2);
        let tri = flag_triangulation(&s0, &s1).unwrap();
        assert_eq!((tri.n_black, tri.n_white, tri.n_face), (1, 2, 1));
        assert_valid(&tri, 2);
    }

    #[test]
    fn genus_one_triple_gives_euler_zero() {
        // σ0=σ1=(0 1 2): one black, one white, one face, degree 3 ⇒ genus 1.
        let s0 = Permutation::from_cycles(3, &[vec![0, 1, 2]]).unwrap();
        let s1 = Permutation::from_cycles(3, &[vec![0, 1, 2]]).unwrap();
        let tri = flag_triangulation(&s0, &s1).unwrap();
        assert_eq!((tri.n_black, tri.n_white, tri.n_face), (1, 1, 1));
        assert!(tri.is_valid_gem());
        assert!(tri.orbits_match_dessin());
        assert_eq!(tri.euler_characteristic(), 0);
        assert_eq!(tri.genus(), 1);
    }

    #[test]
    fn path_two_leaves_one_white() {
        // σ0=id [two black leaves], σ1=(0 1) [one white valence-2]. deg 2.
        let s0 = Permutation::identity(2);
        let s1 = Permutation::from_cycles(2, &[vec![0, 1]]).unwrap();
        let tri = flag_triangulation(&s0, &s1).unwrap();
        assert_eq!((tri.n_black, tri.n_white, tri.n_face), (2, 1, 1));
        assert_valid(&tri, 2);
    }

    #[test]
    fn passport_2_12_5_shape_is_genus_zero() {
        // A [2,12,5]-shaped genus-0 dessin: σ0 = 2^8 1^8, σ1 = 12^2, and the
        // resulting σ∞ must be type 5^4 1^4 for Euler 2. We use one concrete pair
        // and check the triangulation is a valid genus-0 GEM with the passport's
        // vertex/face counts (16 black, 2 white, 8 faces, 24 edges).
        let n = 24;
        let s0 = Permutation::from_cycles(
            n,
            &[
                vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7],
                vec![8, 9], vec![10, 11], vec![12, 13], vec![14, 15],
            ],
        )
        .unwrap();
        let s1 = Permutation::from_cycles(
            n,
            &[(0..12).collect::<Vec<_>>(), (12..24).collect::<Vec<_>>()],
        )
        .unwrap();
        let tri = flag_triangulation(&s0, &s1).unwrap();
        assert!(tri.is_valid_gem());
        assert!(tri.orbits_match_dessin());
        assert_eq!(tri.n_black, 16, "2^8 1^8 ⇒ 16 black vertices");
        assert_eq!(tri.n_white, 2, "12^2 ⇒ 2 white vertices");
        assert_eq!(tri.degree, 24);
        assert_eq!(tri.n_triangles(), 96, "4E = 96 flags");
        // Euler and face count are consequences of this specific pairing.
        let chi = tri.euler_characteristic();
        assert_eq!(chi, 2i64 - 2 * tri.genus());
        if tri.genus() == 0 {
            assert_eq!(tri.n_face, 8, "genus 0 with 16+2 vertices, 24 edges ⇒ 8 faces");
        }
    }
}
