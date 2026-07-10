//! Integral Simplicial Homology
//!
//! Computes the integral homology groups H_k(X; Z) of a
//! [`SimplicialComplex`], including torsion, via integer boundary matrices
//! and Smith normal form (SNF) over the Euclidean domain Z:
//!
//! - the boundary map d_k: C_k -> C_{k-1} sends an ordered simplex
//!   [v_0, ..., v_k] (vertices sorted ascending) to
//!   sum_i (-1)^i [v_0, ..., v-hat_i, ..., v_k];
//! - rank H_k = (kernel rank of d_k) - (image rank of d_{k+1})
//!   = n_k - rank(d_k) - rank(d_{k+1});
//! - the torsion coefficients of H_k are the invariant factors of d_{k+1}
//!   (the SNF diagonal entries) that are neither zero nor units.
//!
//! This mirrors SageMath's `SimplicialComplex.homology()` with
//! `base_ring=ZZ` (non-reduced homology).
//!
//! # Examples
//!
//! ```
//! use rustmath_topology::simplicial_complex_examples::projective_plane;
//! use rustmath_integers::Integer;
//!
//! // RP^2: H_0 = Z, H_1 = Z/2 (torsion!), H_2 = 0.
//! let rp2 = projective_plane();
//! let h = rp2.homology_all().unwrap();
//! assert_eq!(h[0].rank(), 1);
//! assert!(h[0].torsion().is_empty());
//! assert_eq!(h[1].rank(), 0);
//! assert_eq!(h[1].torsion(), &[Integer::from(2)]);
//! assert!(h[2].is_trivial());
//! ```

use crate::simplicial_complex::{Simplex, SimplicialComplex};
use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use std::collections::HashMap;
use std::fmt;

/// A finitely generated abelian group `Z^rank + Z/d_1 + ... + Z/d_t`,
/// with `d_1 | d_2 | ... | d_t` and every `d_i > 1` (invariant factor form).
///
/// This is the value type of [`SimplicialComplex::homology`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HomologyGroup {
    /// Rank of the free part (the Betti number).
    rank: usize,
    /// Torsion coefficients `d_1 | d_2 | ... | d_t`, each `> 1`.
    torsion: Vec<Integer>,
}

impl HomologyGroup {
    /// The trivial group `0`.
    pub fn trivial() -> Self {
        Self {
            rank: 0,
            torsion: Vec::new(),
        }
    }

    /// Construct from a free rank and torsion coefficients.
    ///
    /// The coefficients must be `> 1` and form a divisibility chain
    /// `d_1 | d_2 | ...` (as produced by Smith normal form); otherwise an
    /// `InvalidArgument` error is returned.
    pub fn new(rank: usize, torsion: Vec<Integer>) -> Result<Self> {
        let one = Integer::from(1);
        for d in &torsion {
            if *d <= one {
                return Err(MathError::InvalidArgument(format!(
                    "torsion coefficient {} is not > 1",
                    d
                )));
            }
        }
        for w in torsion.windows(2) {
            if !(w[1].clone() % w[0].clone()).is_zero() {
                return Err(MathError::InvalidArgument(format!(
                    "torsion coefficients not a divisibility chain: {} does not divide {}",
                    w[0], w[1]
                )));
            }
        }
        Ok(Self { rank, torsion })
    }

    /// Rank of the free part (the Betti number).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Torsion coefficients in invariant factor form (`d_1 | d_2 | ...`,
    /// each `> 1`). Empty when the group is free.
    pub fn torsion(&self) -> &[Integer] {
        &self.torsion
    }

    /// Is this the trivial group?
    pub fn is_trivial(&self) -> bool {
        self.rank == 0 && self.torsion.is_empty()
    }

    /// Order of the group: `Some(1)` for the trivial group, the product of
    /// the torsion coefficients when finite, `None` when infinite (rank > 0).
    pub fn order(&self) -> Option<Integer> {
        if self.rank > 0 {
            return None;
        }
        let mut ord = Integer::from(1);
        for d in &self.torsion {
            ord = ord * d.clone();
        }
        Some(ord)
    }
}

impl fmt::Display for HomologyGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_trivial() {
            return write!(f, "0");
        }
        let mut parts = Vec::new();
        match self.rank {
            0 => {}
            1 => parts.push("Z".to_string()),
            r => parts.push(format!("Z^{}", r)),
        }
        for d in &self.torsion {
            parts.push(format!("Z/{}", d));
        }
        write!(f, "{}", parts.join(" + "))
    }
}

/// Rank and torsion of an integer matrix, via Smith normal form:
/// rank = number of nonzero invariant factors, torsion = the invariant
/// factors that are neither zero nor units (returned positive, in
/// divisibility order).
fn snf_rank_and_torsion(m: &Matrix<Integer>) -> Result<(usize, Vec<Integer>)> {
    if m.rows() == 0 || m.cols() == 0 {
        return Ok((0, Vec::new()));
    }
    // elementary_divisors returns the nonzero SNF diagonal, d_1 | d_2 | ...
    // (up to sign over Z, so take absolute values).
    let divisors: Vec<Integer> = m
        .elementary_divisors()?
        .into_iter()
        .map(|d| d.abs())
        .collect();
    let rank = divisors.len();
    let one = Integer::from(1);
    let torsion = divisors.into_iter().filter(|d| *d != one).collect();
    Ok((rank, torsion))
}

impl SimplicialComplex {
    /// The `dim`-simplices in a deterministic order (sorted by vertex list).
    fn sorted_simplices(&self, dim: usize) -> Vec<Simplex> {
        let mut sims = self.simplices(dim);
        sims.sort_by(|a, b| a.vertices().cmp(b.vertices()));
        sims
    }

    /// The `k`-th integer boundary matrix `d_k: C_k -> C_{k-1}`.
    ///
    /// Rows are indexed by the (k-1)-simplices and columns by the
    /// k-simplices, both sorted by vertex list; the entry for
    /// (face, simplex) is `(-1)^i` where the face omits the `i`-th
    /// (0-based, ascending) vertex of the simplex.
    ///
    /// For `k = 0` this is the `0 x n_0` zero map (non-reduced homology:
    /// no augmentation by the empty simplex), and for `k > dim + 1` it is
    /// a `0 x 0` matrix.
    pub fn boundary_matrix(&self, k: usize) -> Result<Matrix<Integer>> {
        let cols = self.sorted_simplices(k);
        let rows = if k == 0 {
            Vec::new()
        } else {
            self.sorted_simplices(k - 1)
        };

        let mut m = Matrix::zeros(rows.len(), cols.len());
        if k == 0 {
            return Ok(m);
        }

        let row_index: HashMap<&Simplex, usize> =
            rows.iter().enumerate().map(|(i, s)| (s, i)).collect();

        for (j, simplex) in cols.iter().enumerate() {
            let verts = simplex.vertices();
            for i in 0..verts.len() {
                let mut face_verts = verts.to_vec();
                face_verts.remove(i);
                let face = Simplex::new(face_verts);
                let r = *row_index.get(&face).ok_or_else(|| {
                    MathError::InvalidArgument(format!(
                        "complex is not closed under faces: {} of {} is missing",
                        face, simplex
                    ))
                })?;
                let sign = if i % 2 == 0 {
                    Integer::from(1)
                } else {
                    Integer::from(-1)
                };
                m.set(r, j, sign)?;
            }
        }
        Ok(m)
    }

    /// The integral homology group `H_k(X; Z)` (non-reduced), with torsion.
    ///
    /// Computed from Smith normal forms of the boundary matrices:
    /// `rank H_k = n_k - rank(d_k) - rank(d_{k+1})`, and the torsion
    /// coefficients are the non-unit invariant factors of `d_{k+1}`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_topology::simplicial_complex_examples::{sphere, torus};
    ///
    /// let s2 = sphere(2);
    /// assert_eq!(s2.homology(0).unwrap().to_string(), "Z");
    /// assert_eq!(s2.homology(1).unwrap().to_string(), "0");
    /// assert_eq!(s2.homology(2).unwrap().to_string(), "Z");
    ///
    /// let t2 = torus();
    /// assert_eq!(t2.homology(1).unwrap().to_string(), "Z^2");
    /// ```
    pub fn homology(&self, k: usize) -> Result<HomologyGroup> {
        let n_k = self.n_simplices(k);
        if n_k == 0 {
            return Ok(HomologyGroup::trivial());
        }
        let (rank_dk, _) = snf_rank_and_torsion(&self.boundary_matrix(k)?)?;
        let (rank_dk1, torsion) = snf_rank_and_torsion(&self.boundary_matrix(k + 1)?)?;

        // im d_{k+1} is contained in ker d_k for a chain complex, so
        // rank(d_k) + rank(d_{k+1}) <= n_k; anything else means the
        // complex (or the boundary matrices) are inconsistent.
        if rank_dk + rank_dk1 > n_k {
            return Err(MathError::InvalidArgument(format!(
                "not a chain complex in degree {}: rank(d_{}) = {} + rank(d_{}) = {} exceeds n_{} = {}",
                k,
                k,
                rank_dk,
                k + 1,
                rank_dk1,
                k,
                n_k
            )));
        }

        HomologyGroup::new(n_k - rank_dk - rank_dk1, torsion)
    }

    /// All integral homology groups `[H_0, H_1, ..., H_dim]`.
    ///
    /// Returns the empty vector for the empty complex. Each boundary
    /// matrix SNF is computed once and shared between the two degrees
    /// that use it.
    pub fn homology_all(&self) -> Result<Vec<HomologyGroup>> {
        let dim = match self.dimension() {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };
        // rank/torsion of d_k for k = 0..=dim+1 (d_0 and d_{dim+1} are zero maps).
        let mut rt = Vec::with_capacity(dim + 2);
        for k in 0..=dim + 1 {
            rt.push(snf_rank_and_torsion(&self.boundary_matrix(k)?)?);
        }
        let mut groups = Vec::with_capacity(dim + 1);
        for k in 0..=dim {
            let n_k = self.n_simplices(k);
            let rank_dk = rt[k].0;
            let (rank_dk1, torsion) = (rt[k + 1].0, rt[k + 1].1.clone());
            if rank_dk + rank_dk1 > n_k {
                return Err(MathError::InvalidArgument(format!(
                    "not a chain complex in degree {}: rank(d_{}) = {} + rank(d_{}) = {} exceeds n_{} = {}",
                    k,
                    k,
                    rank_dk,
                    k + 1,
                    rank_dk1,
                    k,
                    n_k
                )));
            }
            groups.push(HomologyGroup::new(n_k - rank_dk - rank_dk1, torsion)?);
        }
        Ok(groups)
    }

    /// The `k`-th Betti number: `rank H_k(X; Z)`.
    pub fn betti_number(&self, k: usize) -> Result<usize> {
        Ok(self.homology(k)?.rank())
    }

    /// All Betti numbers `[b_0, ..., b_dim]`.
    ///
    /// Their alternating sum equals the Euler characteristic.
    pub fn betti_numbers(&self) -> Result<Vec<usize>> {
        Ok(self.homology_all()?.into_iter().map(|h| h.rank()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplicial_complex_examples::{
        dunce_hat, klein_bottle, projective_plane, simplex, sphere, torus,
    };

    /// Alternating sum of Betti numbers must equal the Euler characteristic.
    fn assert_betti_euler_consistent(complex: &SimplicialComplex) {
        let betti = complex.betti_numbers().unwrap();
        let mut alt = Integer::from(0);
        for (k, b) in betti.iter().enumerate() {
            let term = Integer::from(*b as i64);
            if k % 2 == 0 {
                alt = alt + term;
            } else {
                alt = alt - term;
            }
        }
        assert_eq!(
            alt,
            complex.euler_characteristic(),
            "alternating Betti sum != Euler characteristic"
        );
    }

    fn homology_strings(complex: &SimplicialComplex) -> Vec<String> {
        complex
            .homology_all()
            .unwrap()
            .iter()
            .map(|h| h.to_string())
            .collect()
    }

    #[test]
    fn test_homology_group_display() {
        assert_eq!(HomologyGroup::trivial().to_string(), "0");
        assert_eq!(HomologyGroup::new(1, vec![]).unwrap().to_string(), "Z");
        assert_eq!(HomologyGroup::new(2, vec![]).unwrap().to_string(), "Z^2");
        assert_eq!(
            HomologyGroup::new(1, vec![Integer::from(2)])
                .unwrap()
                .to_string(),
            "Z + Z/2"
        );
        assert_eq!(
            HomologyGroup::new(0, vec![Integer::from(2), Integer::from(6)])
                .unwrap()
                .to_string(),
            "Z/2 + Z/6"
        );
    }

    #[test]
    fn test_homology_group_new_rejects_bad_torsion() {
        // Unit coefficient.
        assert!(HomologyGroup::new(0, vec![Integer::from(1)]).is_err());
        // Not a divisibility chain: 2 does not divide 3.
        assert!(HomologyGroup::new(0, vec![Integer::from(2), Integer::from(3)]).is_err());
    }

    #[test]
    fn test_homology_group_order() {
        assert_eq!(HomologyGroup::trivial().order(), Some(Integer::from(1)));
        assert_eq!(HomologyGroup::new(1, vec![]).unwrap().order(), None);
        assert_eq!(
            HomologyGroup::new(0, vec![Integer::from(2), Integer::from(4)])
                .unwrap()
                .order(),
            Some(Integer::from(8))
        );
    }

    #[test]
    fn test_boundary_matrix_shapes_and_d_squared_zero() {
        // On the torus: d_1 is 7 x 21, d_2 is 21 x 14, and d_1 * d_2 = 0.
        let t = torus();
        let d1 = t.boundary_matrix(1).unwrap();
        let d2 = t.boundary_matrix(2).unwrap();
        assert_eq!((d1.rows(), d1.cols()), (7, 21));
        assert_eq!((d2.rows(), d2.cols()), (21, 14));
        let composite = (d1 * d2).unwrap();
        assert!(composite.data().iter().all(|x| x.is_zero()));

        // d_0 is the zero map out of C_0.
        let d0 = t.boundary_matrix(0).unwrap();
        assert_eq!((d0.rows(), d0.cols()), (0, 7));
        // Beyond the dimension the matrices are empty.
        let d3 = t.boundary_matrix(3).unwrap();
        assert_eq!((d3.rows(), d3.cols()), (14, 0));
    }

    #[test]
    fn test_homology_sphere_2() {
        // S^2: H_0 = Z, H_1 = 0, H_2 = Z (verified with SymPy SNF on the
        // boundary-of-a-3-simplex triangulation).
        let s2 = sphere(2);
        assert_eq!(homology_strings(&s2), vec!["Z", "0", "Z"]);
        assert_betti_euler_consistent(&s2);
    }

    #[test]
    fn test_homology_sphere_1_and_0() {
        // S^1: H_0 = Z, H_1 = Z.
        let s1 = sphere(1);
        assert_eq!(homology_strings(&s1), vec!["Z", "Z"]);
        assert_betti_euler_consistent(&s1);

        // S^0 (two points): H_0 = Z^2.
        let s0 = sphere(0);
        assert_eq!(homology_strings(&s0), vec!["Z^2"]);
        assert_betti_euler_consistent(&s0);
    }

    #[test]
    fn test_homology_torus() {
        // Torus: H_0 = Z, H_1 = Z^2, H_2 = Z (verified with SymPy SNF on
        // the 7-vertex Csaszar triangulation used by torus()).
        let t = torus();
        assert_eq!(homology_strings(&t), vec!["Z", "Z^2", "Z"]);
        assert_betti_euler_consistent(&t);
    }

    #[test]
    fn test_homology_klein_bottle_torsion() {
        // Klein bottle: H_0 = Z, H_1 = Z + Z/2, H_2 = 0. The Z/2 torsion
        // in H_1 is the invariant factor 2 of d_2 (verified with SymPy SNF
        // on the 8-vertex triangulation used by klein_bottle()).
        let kb = klein_bottle();
        let h = kb.homology_all().unwrap();
        assert_eq!(h[0].rank(), 1);
        assert!(h[0].torsion().is_empty());
        assert_eq!(h[1].rank(), 1);
        assert_eq!(h[1].torsion(), &[Integer::from(2)]);
        assert!(h[2].is_trivial());
        assert_eq!(homology_strings(&kb), vec!["Z", "Z + Z/2", "0"]);
        assert_betti_euler_consistent(&kb);
    }

    #[test]
    fn test_homology_projective_plane_torsion() {
        // RP^2: H_0 = Z, H_1 = Z/2, H_2 = 0 (verified with SymPy SNF on the
        // minimal 6-vertex triangulation used by projective_plane()).
        let rp2 = projective_plane();
        let h = rp2.homology_all().unwrap();
        assert_eq!(h[0].rank(), 1);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[1].torsion(), &[Integer::from(2)]);
        assert!(h[2].is_trivial());
        assert_eq!(homology_strings(&rp2), vec!["Z", "Z/2", "0"]);
        assert_betti_euler_consistent(&rp2);
    }

    #[test]
    fn test_homology_dunce_hat_contractible() {
        // Dunce hat is contractible: H_0 = Z, H_1 = 0, H_2 = 0 (verified
        // with SymPy SNF on the 8-vertex Hachimori triangulation).
        let dh = dunce_hat();
        assert_eq!(homology_strings(&dh), vec!["Z", "0", "0"]);
        assert_betti_euler_consistent(&dh);
    }

    #[test]
    fn test_homology_full_simplex_contractible() {
        // A solid simplex is contractible: only H_0 = Z survives.
        let d3 = simplex(3);
        assert_eq!(homology_strings(&d3), vec!["Z", "0", "0", "0"]);
        assert_betti_euler_consistent(&d3);
    }

    #[test]
    fn test_homology_disconnected_counts_components() {
        // Two disjoint triangles (boundaries only): H_0 = Z^2, H_1 = Z^2.
        let mut complex = SimplicialComplex::new();
        for tri in [[0, 1, 2], [3, 4, 5]] {
            for i in 0..3 {
                let mut edge = tri.to_vec();
                edge.remove(i);
                complex.add_simplex(Simplex::new(edge));
            }
        }
        assert_eq!(homology_strings(&complex), vec!["Z^2", "Z^2"]);
        assert_betti_euler_consistent(&complex);
    }

    #[test]
    fn test_homology_empty_complex() {
        let complex = SimplicialComplex::new();
        assert!(complex.homology_all().unwrap().is_empty());
        assert!(complex.betti_numbers().unwrap().is_empty());
        // Any single degree of the empty complex is trivial.
        assert!(complex.homology(0).unwrap().is_trivial());
        assert!(complex.homology(5).unwrap().is_trivial());
    }

    #[test]
    fn test_homology_above_dimension_is_trivial() {
        let s2 = sphere(2);
        assert!(s2.homology(3).unwrap().is_trivial());
        assert!(s2.homology(10).unwrap().is_trivial());
    }

    #[test]
    fn test_betti_numbers_match_single_degree_homology() {
        let kb = klein_bottle();
        let betti = kb.betti_numbers().unwrap();
        assert_eq!(betti, vec![1, 1, 0]);
        for (k, b) in betti.iter().enumerate() {
            assert_eq!(kb.betti_number(k).unwrap(), *b);
        }
    }
}
