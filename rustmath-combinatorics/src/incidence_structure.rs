//! Incidence structures and t-designs (MAGMA Chapter 147).
//!
//! An *incidence structure* `D = (P, B)` has a point set `P = {0, ..., v-1}` and a
//! collection of *blocks* `B` (subsets of `P`; repeated blocks allowed). This generalises
//! the fixed-block-size `BlockDesign` in `designs.rs` to variable block sizes and adds the
//! related-structure operations and t-design testing of Chapter 147.
//!
//! Implemented intrinsics (subset of §147.2–§147.8):
//!
//! | MAGMA intrinsic                | method here                          |
//! |--------------------------------|--------------------------------------|
//! | `IncidenceStructure<v \| X>`   | [`IncidenceStructure::new`], [`from_incidence_matrix`] |
//! | `IncidenceMatrix(D)`           | [`IncidenceStructure::incidence_matrix`] |
//! | `NumberOfPoints`/`NumberOfBlocks` | [`num_points`], [`num_blocks`]    |
//! | `PointDegrees`/`BlockSizes`    | [`point_degrees`], [`block_sizes`]   |
//! | `Covalence(D, S)`              | [`covalence`]                        |
//! | `Complement`/`Dual`           | [`complement`], [`dual`]             |
//! | `Contraction`/`Residual`      | [`contraction_point`], [`contraction_block`], [`residual_point`], [`residual_block`] |
//! | `Restriction`/`Simplify`      | [`restriction`], [`simplify`]        |
//! | `IsSimple`/`IsUniform`         | [`is_simple`], [`is_uniform`]        |
//! | `IsDesign`/`IsBalanced`/`IsSteiner` | [`is_design`], [`is_balanced`], [`is_steiner`] |
//! | `Parameters`/`ReplicationNumber` | [`parameters`], [`replication_number`] |
//! | `IntersectionNumber`          | [`intersection_number`]              |
//!
//! t-balance testing uses the brute-force ("NoOrbits") algorithm.
//!
//! Reference: MAGMA Handbook, Chapter 147.
//!
//! [`from_incidence_matrix`]: IncidenceStructure::from_incidence_matrix
//! [`num_points`]: IncidenceStructure::num_points
//! [`num_blocks`]: IncidenceStructure::num_blocks
//! [`point_degrees`]: IncidenceStructure::point_degrees
//! [`block_sizes`]: IncidenceStructure::block_sizes
//! [`covalence`]: IncidenceStructure::covalence
//! [`complement`]: IncidenceStructure::complement
//! [`dual`]: IncidenceStructure::dual
//! [`contraction_point`]: IncidenceStructure::contraction_point
//! [`contraction_block`]: IncidenceStructure::contraction_block
//! [`residual_point`]: IncidenceStructure::residual_point
//! [`residual_block`]: IncidenceStructure::residual_block
//! [`restriction`]: IncidenceStructure::restriction
//! [`simplify`]: IncidenceStructure::simplify
//! [`is_simple`]: IncidenceStructure::is_simple
//! [`is_uniform`]: IncidenceStructure::is_uniform
//! [`is_design`]: IncidenceStructure::is_design
//! [`is_balanced`]: IncidenceStructure::is_balanced
//! [`is_steiner`]: IncidenceStructure::is_steiner
//! [`parameters`]: IncidenceStructure::parameters
//! [`replication_number`]: IncidenceStructure::replication_number
//! [`intersection_number`]: IncidenceStructure::intersection_number

use rustmath_integers::Integer;
use rustmath_matrix::Matrix;

/// The parameters `t-(v, b, r, k, λ)` of a t-design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DesignParameters {
    pub t: usize,
    pub v: usize,
    pub b: usize,
    pub r: usize,
    pub k: usize,
    pub lambda: usize,
}

/// An incidence structure on points `{0, ..., v-1}` with a list of blocks (point subsets).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncidenceStructure {
    v: usize,
    blocks: Vec<Vec<usize>>,
}

/// All `t`-subsets of `{0, ..., n-1}` in lexicographic order.
fn combinations(n: usize, t: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if t > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..t).collect();
    loop {
        out.push(idx.clone());
        if t == 0 {
            break;
        }
        // advance
        let mut i = t;
        while i > 0 {
            i -= 1;
            if idx[i] != i + n - t {
                idx[i] += 1;
                for j in i + 1..t {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return out;
            }
        }
    }
    out
}

impl IncidenceStructure {
    /// Construct an incidence structure with `v` points and the given blocks. Each block is
    /// canonicalised (sorted, duplicate points removed); repeated blocks are retained.
    /// Returns `None` if any block references a point `≥ v`.
    pub fn new(v: usize, blocks: Vec<Vec<usize>>) -> Option<Self> {
        let mut cleaned = Vec::with_capacity(blocks.len());
        for b in blocks {
            let mut b = b;
            b.sort_unstable();
            b.dedup();
            if b.last().map_or(false, |&p| p >= v) {
                return None;
            }
            cleaned.push(b);
        }
        Some(IncidenceStructure { v, blocks: cleaned })
    }

    /// Build from a `v × b` incidence matrix: block `j` is `{ i : M[i][j] ≠ 0 }`.
    pub fn from_incidence_matrix(m: &Matrix<Integer>) -> Self {
        let v = m.rows();
        let b = m.cols();
        let mut blocks = Vec::with_capacity(b);
        for j in 0..b {
            let mut block = Vec::new();
            for i in 0..v {
                if !m.get(i, j).unwrap().is_zero() {
                    block.push(i);
                }
            }
            blocks.push(block);
        }
        IncidenceStructure { v, blocks }
    }

    /// The Fano plane, the `2-(7,3,1)` design (points `0..7`).
    pub fn fano_plane() -> Self {
        IncidenceStructure {
            v: 7,
            blocks: vec![
                vec![0, 1, 2],
                vec![0, 3, 4],
                vec![0, 5, 6],
                vec![1, 3, 5],
                vec![1, 4, 6],
                vec![2, 3, 6],
                vec![2, 4, 5],
            ],
        }
    }

    /// `NumberOfPoints(D)`.
    pub fn num_points(&self) -> usize {
        self.v
    }

    /// `NumberOfBlocks(D)`.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// The blocks of `D`.
    pub fn blocks(&self) -> &[Vec<usize>] {
        &self.blocks
    }

    /// `IncidenceMatrix(D)` — the `v × b` `(0,1)`-matrix over the integers.
    pub fn incidence_matrix(&self) -> Matrix<Integer> {
        let b = self.blocks.len();
        let mut data = vec![Integer::zero(); self.v * b];
        for (j, block) in self.blocks.iter().enumerate() {
            for &p in block {
                data[p * b + j] = Integer::one();
            }
        }
        Matrix::from_vec(self.v, b, data).unwrap()
    }

    /// `PointDegrees(D)` — number of blocks containing each point.
    pub fn point_degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.v];
        for block in &self.blocks {
            for &p in block {
                deg[p] += 1;
            }
        }
        deg
    }

    /// `BlockSizes(D)` — size of each block.
    pub fn block_sizes(&self) -> Vec<usize> {
        self.blocks.iter().map(|b| b.len()).collect()
    }

    /// `Covalence(D, S)` — the number of blocks containing the point subset `S`.
    pub fn covalence(&self, s: &[usize]) -> usize {
        let sset: std::collections::BTreeSet<usize> = s.iter().copied().collect();
        self.blocks
            .iter()
            .filter(|b| {
                let bset: std::collections::BTreeSet<usize> = b.iter().copied().collect();
                sset.is_subset(&bset)
            })
            .count()
    }

    /// `IsSimple(D)` — true iff there are no repeated blocks.
    pub fn is_simple(&self) -> bool {
        let mut sorted = self.blocks.clone();
        sorted.sort();
        for i in 1..sorted.len() {
            if sorted[i] == sorted[i - 1] {
                return false;
            }
        }
        true
    }

    /// `IsUniform(D)` — if every block has the same size `k`, return `Some(k)`.
    pub fn is_uniform(&self) -> Option<usize> {
        let mut it = self.blocks.iter().map(|b| b.len());
        let first = it.next()?;
        if it.all(|k| k == first) {
            Some(first)
        } else {
            None
        }
    }

    /// `IsBalanced(D, t)` — if every `t`-subset of points lies in the same number `λ` of
    /// blocks, return `Some(λ)`. `t = 0` returns `Some(number of blocks)`.
    pub fn is_balanced(&self, t: usize) -> Option<usize> {
        if t == 0 {
            return Some(self.blocks.len());
        }
        if t > self.v {
            return None;
        }
        let subsets = combinations(self.v, t);
        let mut lambda: Option<usize> = None;
        for s in subsets {
            let c = self.covalence(&s);
            match lambda {
                None => lambda = Some(c),
                Some(l) if l != c => return None,
                _ => {}
            }
        }
        lambda
    }

    /// `IsDesign(D, t)` — true iff `D` is a `t-(v,k,λ)` design (simple, uniform, and
    /// `t`-balanced with `λ > 0`); returns `Some((k, λ))` if so.
    pub fn is_design(&self, t: usize) -> Option<(usize, usize)> {
        if t == 0 || !self.is_simple() {
            return None;
        }
        let k = self.is_uniform()?;
        let lambda = self.is_balanced(t)?;
        if lambda == 0 {
            return None;
        }
        Some((k, lambda))
    }

    /// `IsSteiner(D, t)` — true iff `D` is a Steiner `t`-design (`λ = 1`).
    pub fn is_steiner(&self, t: usize) -> bool {
        matches!(self.is_design(t), Some((_, 1)))
    }

    /// `ReplicationNumber(D)` — the number `r` of blocks through a point, if constant.
    pub fn replication_number(&self) -> Option<usize> {
        let deg = self.point_degrees();
        let first = *deg.first()?;
        if deg.iter().all(|&d| d == first) {
            Some(first)
        } else {
            None
        }
    }

    /// `Parameters(D)` — the `t-(v,b,r,k,λ)` parameters, if `D` is a `t`-design.
    pub fn parameters(&self, t: usize) -> Option<DesignParameters> {
        let (k, lambda) = self.is_design(t)?;
        let r = self.replication_number()?;
        Some(DesignParameters {
            t,
            v: self.v,
            b: self.blocks.len(),
            r,
            k,
            lambda,
        })
    }

    /// `IntersectionNumber(D, i, j)` — for a t-design, the number of blocks containing a
    /// fixed `i`-subset and disjoint from a fixed (disjoint) `j`-subset (`i + j ≤ v`).
    /// Well-defined for designs; computed on a representative pair of subsets.
    pub fn intersection_number(&self, i: usize, j: usize) -> Option<usize> {
        if i + j > self.v {
            return None;
        }
        let iset: Vec<usize> = (0..i).collect();
        let jset: Vec<usize> = (i..i + j).collect();
        let count = self
            .blocks
            .iter()
            .filter(|b| {
                iset.iter().all(|p| b.contains(p)) && jset.iter().all(|p| !b.contains(p))
            })
            .count();
        Some(count)
    }

    /// `Complement(D)` — each block is replaced by its complement in the point set.
    pub fn complement(&self) -> Self {
        let blocks = self
            .blocks
            .iter()
            .map(|b| {
                let bset: std::collections::BTreeSet<usize> = b.iter().copied().collect();
                (0..self.v).filter(|p| !bset.contains(p)).collect()
            })
            .collect();
        IncidenceStructure { v: self.v, blocks }
    }

    /// `Dual(D)` — interchange the roles of points and blocks.
    pub fn dual(&self) -> Self {
        let new_v = self.blocks.len();
        let mut new_blocks = vec![Vec::new(); self.v];
        for (j, block) in self.blocks.iter().enumerate() {
            for &p in block {
                new_blocks[p].push(j);
            }
        }
        IncidenceStructure {
            v: new_v,
            blocks: new_blocks,
        }
    }

    /// `Simplify(D)` — remove repeated blocks.
    pub fn simplify(&self) -> Self {
        let mut seen = std::collections::BTreeSet::new();
        let mut blocks = Vec::new();
        for b in &self.blocks {
            if seen.insert(b.clone()) {
                blocks.push(b.clone());
            }
        }
        IncidenceStructure { v: self.v, blocks }
    }

    /// Remap points to `0..v'` by dropping the points not in `keep` (sorted), applying `f`.
    fn relabel(&self, keep: &[usize], blocks: Vec<Vec<usize>>) -> Self {
        let mut map = vec![usize::MAX; self.v];
        for (new, &old) in keep.iter().enumerate() {
            map[old] = new;
        }
        let blocks = blocks
            .into_iter()
            .map(|b| b.into_iter().map(|p| map[p]).collect())
            .collect();
        IncidenceStructure {
            v: keep.len(),
            blocks,
        }
    }

    /// `Contraction(D, p)` — delete point `p`, keep the blocks incident with `p` (minus `p`).
    pub fn contraction_point(&self, p: usize) -> Self {
        let keep: Vec<usize> = (0..self.v).filter(|&x| x != p).collect();
        let blocks: Vec<Vec<usize>> = self
            .blocks
            .iter()
            .filter(|b| b.contains(&p))
            .map(|b| b.iter().copied().filter(|&x| x != p).collect())
            .collect();
        self.relabel(&keep, blocks)
    }

    /// `Residual(D, p)` — delete point `p`, keep the blocks *not* containing `p`.
    pub fn residual_point(&self, p: usize) -> Self {
        let keep: Vec<usize> = (0..self.v).filter(|&x| x != p).collect();
        let blocks: Vec<Vec<usize>> = self
            .blocks
            .iter()
            .filter(|b| !b.contains(&p))
            .cloned()
            .collect();
        self.relabel(&keep, blocks)
    }

    /// `Contraction(D, b)` — point set is block `b_idx`; blocks are the non-empty
    /// intersections of `b_idx` with the other blocks.
    pub fn contraction_block(&self, b_idx: usize) -> Self {
        let base: std::collections::BTreeSet<usize> = self.blocks[b_idx].iter().copied().collect();
        let keep: Vec<usize> = self.blocks[b_idx].clone();
        let mut blocks = Vec::new();
        for (j, c) in self.blocks.iter().enumerate() {
            if j == b_idx {
                continue;
            }
            let inter: Vec<usize> = c.iter().copied().filter(|p| base.contains(p)).collect();
            if !inter.is_empty() {
                blocks.push(inter);
            }
        }
        self.relabel(&keep, blocks)
    }

    /// `Residual(D, b)` — point set is `P − b_idx`; blocks are the non-empty intersections
    /// of `P − b_idx` with the other blocks.
    pub fn residual_block(&self, b_idx: usize) -> Self {
        let base: std::collections::BTreeSet<usize> = self.blocks[b_idx].iter().copied().collect();
        let keep: Vec<usize> = (0..self.v).filter(|p| !base.contains(p)).collect();
        let mut blocks = Vec::new();
        for (j, c) in self.blocks.iter().enumerate() {
            if j == b_idx {
                continue;
            }
            let inter: Vec<usize> = c.iter().copied().filter(|p| !base.contains(p)).collect();
            if !inter.is_empty() {
                blocks.push(inter);
            }
        }
        self.relabel(&keep, blocks)
    }

    /// `Restriction(D, S)` — restrict to the point subset `S`; blocks are the non-empty
    /// intersections of the blocks with `S`.
    pub fn restriction(&self, s: &[usize]) -> Self {
        let mut keep = s.to_vec();
        keep.sort_unstable();
        keep.dedup();
        let sset: std::collections::BTreeSet<usize> = keep.iter().copied().collect();
        let blocks: Vec<Vec<usize>> = self
            .blocks
            .iter()
            .map(|b| b.iter().copied().filter(|p| sset.contains(p)).collect::<Vec<_>>())
            .filter(|b: &Vec<usize>| !b.is_empty())
            .collect();
        self.relabel(&keep, blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combinations() {
        assert_eq!(combinations(3, 0), vec![Vec::<usize>::new()]);
        assert_eq!(combinations(4, 2).len(), 6);
        assert_eq!(combinations(4, 2)[0], vec![0, 1]);
        assert_eq!(combinations(4, 2).last().unwrap(), &vec![2, 3]);
        assert_eq!(combinations(3, 4), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn test_fano_is_design() {
        let d = IncidenceStructure::fano_plane();
        assert_eq!(d.num_points(), 7);
        assert_eq!(d.num_blocks(), 7);
        // 2-(7,3,1) design.
        assert_eq!(d.is_design(2), Some((3, 1)));
        assert!(d.is_steiner(2));
        assert_eq!(d.is_uniform(), Some(3));
        assert!(d.is_simple());
        assert_eq!(d.replication_number(), Some(3));
        let p = d.parameters(2).unwrap();
        assert_eq!((p.t, p.v, p.b, p.r, p.k, p.lambda), (2, 7, 7, 3, 3, 1));
    }

    #[test]
    fn test_incidence_matrix_roundtrip() {
        let d = IncidenceStructure::fano_plane();
        let m = d.incidence_matrix();
        assert_eq!(m.rows(), 7);
        assert_eq!(m.cols(), 7);
        // Each column (block) has exactly 3 ones; each row (point) has exactly 3 ones.
        let d2 = IncidenceStructure::from_incidence_matrix(&m);
        assert_eq!(d2.block_sizes(), vec![3; 7]);
        assert_eq!(d2.point_degrees(), vec![3; 7]);
        // The reconstructed structure is the same 2-(7,3,1) design.
        assert_eq!(d2.is_design(2), Some((3, 1)));
    }

    #[test]
    fn test_covalence_and_balance() {
        let d = IncidenceStructure::fano_plane();
        // Any pair of points is in exactly 1 block.
        assert_eq!(d.covalence(&[0, 1]), 1);
        assert_eq!(d.covalence(&[2, 5]), 1);
        assert_eq!(d.is_balanced(2), Some(1));
        // Any single point: in 3 blocks (also 1-balanced).
        assert_eq!(d.is_balanced(1), Some(3));
        assert_eq!(d.is_balanced(0), Some(7));
    }

    #[test]
    fn test_dual_of_fano_is_fano_like() {
        let d = IncidenceStructure::fano_plane();
        let dual = d.dual();
        // Fano plane is self-dual: dual is again a 2-(7,3,1) design.
        assert_eq!(dual.num_points(), 7);
        assert_eq!(dual.is_design(2), Some((3, 1)));
    }

    #[test]
    fn test_complement() {
        let d = IncidenceStructure::fano_plane();
        let c = d.complement();
        // Complement of a 2-(7,3,1) has block size 4.
        assert_eq!(c.is_uniform(), Some(4));
        assert_eq!(c.num_blocks(), 7);
    }

    #[test]
    fn test_contraction_and_residual() {
        let d = IncidenceStructure::fano_plane();
        // Contraction at point 0: blocks through 0 minus 0 -> 3 blocks of size 2 on 6 points.
        let contr = d.contraction_point(0);
        assert_eq!(contr.num_points(), 6);
        assert_eq!(contr.num_blocks(), 3);
        assert_eq!(contr.is_uniform(), Some(2));
        // Residual at point 0: blocks not through 0 -> 4 blocks on 6 points.
        let res = d.residual_point(0);
        assert_eq!(res.num_points(), 6);
        assert_eq!(res.num_blocks(), 4);
        assert_eq!(res.is_uniform(), Some(3));
    }

    #[test]
    fn test_simplify_and_is_simple() {
        // Structure with a repeated block.
        let d = IncidenceStructure::new(4, vec![vec![0, 1], vec![0, 1], vec![2, 3]]).unwrap();
        assert!(!d.is_simple());
        let s = d.simplify();
        assert!(s.is_simple());
        assert_eq!(s.num_blocks(), 2);
    }

    #[test]
    fn test_intersection_number() {
        let d = IncidenceStructure::fano_plane();
        // λ_1^0 = replication number = 3.
        assert_eq!(d.intersection_number(1, 0), Some(3));
        // λ_2^0 = 1 (blocks through a fixed pair).
        assert_eq!(d.intersection_number(2, 0), Some(1));
    }

    #[test]
    fn test_new_rejects_bad_point() {
        assert!(IncidenceStructure::new(3, vec![vec![0, 3]]).is_none());
    }
}
