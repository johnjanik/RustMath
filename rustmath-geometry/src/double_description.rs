//! Exact double description over Q: conversion between the H-representation
//! (intersection of half-spaces) and V-representation (convex hull of vertices
//! plus conic hull of rays) of a polyhedron, with all arithmetic performed in
//! exact rational numbers. No floating point is used anywhere.
//!
//! # Algorithm
//!
//! Both conversions reduce to one primitive: enumerating the extreme rays of a
//! polyhedral cone `K = {y : R y >= 0}` given its constraint rows, via the
//! double description method of Motzkin–Raiffa–Thompson–Thrall (as presented in
//! Fukuda–Prodon, "Double description method revisited", 1996):
//!
//! 1. The lineality space `L = ker R` is computed exactly; the cone is
//!    `K = (K ∩ W) ⊕ L` for the coordinate subspace `W` spanned by the pivot
//!    columns of `R` (a complement of `L`), so the enumeration runs on the
//!    *pointed* cone `K ∩ W` obtained by restricting the rows to the pivot
//!    columns. `L` is reported as `lines`.
//! 2. The pointed part starts from `r = rank(R)` linearly independent rows,
//!    whose simplicial cone has the columns of the inverse matrix as extreme
//!    rays, and inserts the remaining rows one at a time. Inserting row `w`
//!    keeps the rays with `w·y >= 0` and adds, for each *adjacent* pair of
//!    rays `(p, q)` with `w·p > 0 > w·q`, the combination
//!    `(w·p) q - (w·q) p` (a positive combination tight on `w`).
//!
//! # Adjacency criterion
//!
//! The **algebraic (rank) adjacency test** is used: extreme rays `u`, `v` of
//! the pointed cone `{y in Q^r : A y >= 0}` are adjacent if and only if the
//! rows of `A` tight at *both* `u` and `v` have rank exactly `r - 2`
//! (Fukuda–Prodon, Proposition 7). Correctness: `x = u + v` lies in the cone
//! and its tight set is exactly the common tight set `Z(u) ∩ Z(v)`, so the
//! minimal face containing both rays is `F = {y in K : A_{Z(u)∩Z(v)} y = 0}`
//! of dimension `r - rank(A_{Z(u)∩Z(v)})`; `u` and `v` are adjacent iff that
//! face is 2-dimensional, and a 2-face of a pointed cone contains exactly two
//! extreme rays. The test is exact over Q and requires no genericity or
//! full-dimensionality assumption — only pointedness, which step 1 guarantees.
//!
//! # H -> V (`h_to_v`)
//!
//! The polyhedron `P = {x : A x <= b, C x = d}` in Q^n is homogenized to the
//! cone over `P` at height 1: `K = {(x0, x) : x0 >= 0, b x0 - A x >= 0,
//! d x0 - C x = 0}` in Q^(n+1) (equations become inequality pairs). Extreme
//! rays of `K` with height `x0 > 0` dehomogenize (divide by `x0`) to the
//! vertices of `P`; extreme rays at height 0 are the extreme rays of the
//! recession cone; the lineality of `K` (always at height 0 because of the
//! `x0 >= 0` row) gives the `lines`. `P` is empty iff no extreme ray has
//! positive height, in which case the honest answer — the empty polyhedron —
//! is returned with *no* vertices, rays, or lines.
//!
//! # V -> H (`v_to_h`): duality
//!
//! Given generators (vertices `v_i`, rays `r_j`, lines `l_k`), the homogenized
//! cone is `K = cone{(1, v_i), (0, r_j)} + span{(0, l_k)}`. The standard
//! duality trick: run the *same* cone enumeration on the dual cone
//! `K* = {c : c·g >= 0 for all generators g, c·l = 0 for all lines}`, whose
//! H-representation rows are exactly the generators (transposing the roles of
//! input rows and output rays). By Minkowski–Weyl, `K** = K`, so the extreme
//! rays of `K*` are the irredundant inequality normals of `K` (one per facet)
//! and a basis of the lineality of `K*` — which is `span(K)^⊥` — yields the
//! affine-hull *equations*. A ray `c = (c0, a)` of `K*` dehomogenizes to the
//! inequality `(-a)·x <= c0`, a line of `K*` to the equation `a·x = -c0`.
//!
//! # Output design for lower-dimensional polyhedra
//!
//! The H-representation is always the unambiguous pair
//! `{equations, inequalities}`: the equations cut out the affine hull, and the
//! inequalities cut out the polyhedron *within* the affine hull (one per
//! facet). Canonicalization makes this exactly unique — see below.
//!
//! # Canonicalization
//!
//! `HPolyhedron::canonicalize`:
//! * equations: augmented rows `[a | b]` are put in exact RREF (this detects a
//!   contradictory equation system), scaled to primitive coprime-integer form
//!   (the leading coefficient, an RREF pivot, stays positive), ordered by
//!   pivot column;
//! * inequalities: each augmented row is reduced modulo the equation RREF
//!   (the pivot variables are eliminated), which fixes the representative of
//!   the inequality modulo the affine hull; rows whose normal vanishes are
//!   dropped when the constant is `>= 0` (trivial) and mark the polyhedron
//!   empty when the constant is `< 0`; the survivors are scaled by a positive
//!   rational to primitive coprime-integer form (the direction of `<=` forbids
//!   sign flips), deduplicated, and sorted lexicographically.
//! * The canonical *empty* H-form is the single inequality `0·x <= -1`.
//!
//! `VPolyhedron::canonicalize`: vertices are deduplicated and sorted; rays are
//! scaled to primitive integer direction (positive scaling only),
//! deduplicated, sorted; lines are replaced by the RREF basis of their span in
//! primitive integer form, sorted. A V-representation with no vertices denotes
//! the empty polyhedron (`conv(∅) = ∅`), so its rays and lines are dropped.
//!
//! Canonical forms are unique for **minimal** representations, and both
//! conversions produce minimal representations, so round trips satisfy
//! `v_to_h(h_to_v(h)) == h.canonicalize()` and
//! `h_to_v(v_to_h(v)) == v.canonicalize()` *exactly* (for `v` listing exactly
//! the extreme points, and feasible minimal `h`).
//!
//! # Budget
//!
//! Double description can blow up combinatorially on adversarial inputs, so
//! every conversion takes a [`DdBudget`] bounding the number of intermediate
//! rays and returns an honest `Err` when the bound is exceeded.

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// Resource budget for a double description run.
///
/// `max_rays` bounds the number of rays held at any moment (including
/// intermediate steps, whose ray count can exceed the final answer's).
/// Exceeding the budget aborts the conversion with an `Err` — never a
/// truncated or fabricated result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DdBudget {
    /// Maximum number of rays allowed at any point during the computation.
    pub max_rays: usize,
}

impl DdBudget {
    /// Budget capped at `max_rays` simultaneous rays.
    pub fn new(max_rays: usize) -> Self {
        DdBudget { max_rays }
    }
}

impl Default for DdBudget {
    fn default() -> Self {
        DdBudget { max_rays: 100_000 }
    }
}

/// H-representation: `{x : a·x <= b for (a, b) in inequalities,
/// a·x = b for (a, b) in equations}`, exact rational coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HPolyhedron {
    /// Ambient dimension `n` (length of every coefficient vector).
    pub ambient_dim: usize,
    /// Rows `(a, b)` meaning `a·x <= b`.
    pub inequalities: Vec<(Vec<Rational>, Rational)>,
    /// Rows `(a, b)` meaning `a·x = b` (affine hull of a lower-dimensional
    /// polyhedron).
    pub equations: Vec<(Vec<Rational>, Rational)>,
}

/// V-representation: `conv(vertices) + cone(rays) + span(lines)`.
///
/// Conventions: a representation with **no vertices is the empty polyhedron**
/// regardless of rays or lines (`conv(∅) = ∅`). When `lines` is nonempty the
/// polyhedron has no true vertices; the entries of `vertices` are then
/// representative points modulo the lineality space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VPolyhedron {
    /// Ambient dimension `n` (length of every point/direction vector).
    pub ambient_dim: usize,
    /// Extreme points (representative points modulo `lines` if any).
    pub vertices: Vec<Vec<Rational>>,
    /// Extreme recession directions, primitive integer form when canonical.
    pub rays: Vec<Vec<Rational>>,
    /// Basis of the lineality space (two-way infinite directions).
    pub lines: Vec<Vec<Rational>>,
}

// ---------------------------------------------------------------------------
// Exact linear algebra helpers over Rational
// ---------------------------------------------------------------------------

fn sign(x: &Rational) -> i8 {
    x.numerator().signum()
}

fn dot(a: &[Rational], b: &[Rational]) -> Rational {
    debug_assert_eq!(a.len(), b.len());
    let mut s = Rational::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        s = s + x * y;
    }
    s
}

/// Scale a rational vector by a positive rational so that its entries become
/// coprime integers. Returns `None` for the zero vector. The positive scaling
/// preserves signs, hence the direction of inequalities.
fn primitive(v: &[Rational]) -> Option<Vec<Rational>> {
    let mut l = Integer::one();
    for x in v {
        l = l.lcm(x.denominator());
    }
    let ints: Vec<Integer> = v
        .iter()
        .map(|x| x.numerator().clone() * (&l / x.denominator()))
        .collect();
    let mut g = Integer::zero();
    for i in &ints {
        // Integer::gcd (via EuclideanDomain) is sign-carrying: gcd(0, -1) = -1.
        // A positive divisor is required so the scaling never flips signs.
        g = g.gcd(i).abs();
    }
    if g.is_zero() {
        return None;
    }
    Some(
        ints.into_iter()
            .map(|i| Rational::from_integer(i / g.clone()))
            .collect(),
    )
}

/// Exact in-place reduced row echelon form. Returns the pivot columns in
/// increasing order.
fn rref(mat: &mut [Vec<Rational>]) -> Vec<usize> {
    let rows = mat.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = mat[0].len();
    let mut pivots = Vec::new();
    let mut r = 0;
    for c in 0..cols {
        if r >= rows {
            break;
        }
        let Some(pr) = (r..rows).find(|&i| !mat[i][c].is_zero()) else {
            continue;
        };
        mat.swap(r, pr);
        let inv = mat[r][c]
            .reciprocal()
            .expect("pivot entry is nonzero by construction");
        for x in mat[r][c..].iter_mut() {
            let t = &*x * &inv;
            *x = t;
        }
        let pivot_row = mat[r].clone();
        for (i, row) in mat.iter_mut().enumerate() {
            if i != r && !row[c].is_zero() {
                let f = row[c].clone();
                for (x, p) in row[c..].iter_mut().zip(&pivot_row[c..]) {
                    let t = &f * p;
                    *x = &*x - &t;
                }
            }
        }
        pivots.push(c);
        r += 1;
    }
    pivots
}

fn rank(rows: &[Vec<Rational>]) -> usize {
    let mut m = rows.to_vec();
    rref(&mut m).len()
}

/// Basis of `{x in Q^n : row·x = 0 for every row}`.
fn kernel_basis(rows: &[Vec<Rational>], n: usize) -> Vec<Vec<Rational>> {
    let mut m = rows.to_vec();
    let pivots = rref(&mut m);
    let mut basis = Vec::new();
    for free in 0..n {
        if pivots.contains(&free) {
            continue;
        }
        let mut v = vec![Rational::zero(); n];
        v[free] = Rational::one();
        for (i, &p) in pivots.iter().enumerate() {
            v[p] = -m[i][free].clone();
        }
        basis.push(v);
    }
    basis
}

/// Exact inverse of a square matrix, or `None` if singular.
fn invert(m: &[Vec<Rational>]) -> Option<Vec<Vec<Rational>>> {
    let r = m.len();
    let mut aug: Vec<Vec<Rational>> = m
        .iter()
        .enumerate()
        .map(|(i, row)| {
            debug_assert_eq!(row.len(), r);
            let mut v = row.clone();
            for j in 0..r {
                v.push(if i == j {
                    Rational::one()
                } else {
                    Rational::zero()
                });
            }
            v
        })
        .collect();
    let pivots = rref(&mut aug);
    if pivots.len() != r || pivots.iter().enumerate().any(|(i, &p)| p != i) {
        return None;
    }
    Some(aug.into_iter().map(|row| row[r..].to_vec()).collect())
}

/// Indices of a maximal linearly independent subset of `rows`, chosen
/// greedily in input order (incremental exact Gaussian elimination).
fn independent_row_indices(rows: &[Vec<Rational>]) -> Vec<usize> {
    let mut ech: Vec<Vec<Rational>> = Vec::new(); // echelon rows, sorted by leading column
    let mut idx = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let mut v = row.clone();
        for e in &ech {
            let lead = e
                .iter()
                .position(|x| !x.is_zero())
                .expect("echelon rows are nonzero");
            if !v[lead].is_zero() {
                let f = &v[lead] / &e[lead];
                for j in 0..v.len() {
                    let t = &f * &e[j];
                    v[j] = &v[j] - &t;
                }
            }
        }
        if v.iter().any(|x| !x.is_zero()) {
            idx.push(i);
            ech.push(v);
            ech.sort_by_key(|e| e.iter().position(|x| !x.is_zero()).unwrap_or(usize::MAX));
        }
    }
    idx
}

// ---------------------------------------------------------------------------
// Core: extreme rays of a cone {y : R y >= 0}
// ---------------------------------------------------------------------------

struct ConeGenerators {
    /// Extreme rays (modulo `lines`), each in primitive integer form.
    rays: Vec<Vec<Rational>>,
    /// Basis of the lineality space `ker R`.
    lines: Vec<Vec<Rational>>,
}

/// Algebraic adjacency test (Fukuda–Prodon Prop. 7): extreme rays `u`, `v`
/// of the pointed cone cut out by the `processed` rows of `rrows` (in Q^r)
/// are adjacent iff the rows tight at both have rank exactly `r - 2`.
fn adjacent(
    u: &[Rational],
    v: &[Rational],
    rrows: &[Vec<Rational>],
    processed: &[usize],
    r: usize,
) -> bool {
    let mut common: Vec<Vec<Rational>> = Vec::new();
    for &k in processed {
        let row = &rrows[k];
        if dot(row, u).is_zero() && dot(row, v).is_zero() {
            common.push(row.clone());
        }
    }
    if common.len() + 2 < r {
        return false; // rank <= |common| < r - 2: cheap necessary condition
    }
    rank(&common) + 2 == r
}

/// Extreme rays and lineality basis of `K = {y in Q^n : row·y >= 0}` by the
/// double description method. See the module docs for the algorithm and the
/// adjacency criterion.
fn cone_dd(
    rows_in: &[Vec<Rational>],
    n: usize,
    budget: &DdBudget,
) -> Result<ConeGenerators, String> {
    // Canonical, deduplicated, nonzero constraint rows.
    let mut rows: Vec<Vec<Rational>> = Vec::new();
    for r in rows_in {
        debug_assert_eq!(r.len(), n);
        if let Some(p) = primitive(r) {
            if !rows.contains(&p) {
                rows.push(p);
            }
        }
    }

    // Lineality space; if it is everything, there are no extreme rays.
    let lines = kernel_basis(&rows, n);
    let r = n - lines.len();
    if r == 0 {
        return Ok(ConeGenerators {
            rays: Vec::new(),
            lines,
        });
    }

    // Pivot columns J of the row matrix span a coordinate complement W of the
    // lineality space, and K = (K ∩ W) ⊕ L with K ∩ W pointed; restricting
    // the rows to J is faithful (a row vanishing on W and on L is zero).
    let piv_cols = {
        let mut m = rows.clone();
        rref(&mut m)
    };
    debug_assert_eq!(piv_cols.len(), r);
    let rrows: Vec<Vec<Rational>> = rows
        .iter()
        .map(|row| piv_cols.iter().map(|&c| row[c].clone()).collect())
        .collect();

    // Initial simplicial cone from r independent rows: its extreme rays are
    // the columns of the inverse matrix. Starting from a full-rank subset
    // keeps every intermediate cone pointed, as the DD invariant requires.
    let init = independent_row_indices(&rrows);
    debug_assert_eq!(init.len(), r);
    let m0: Vec<Vec<Rational>> = init.iter().map(|&i| rrows[i].clone()).collect();
    let minv = invert(&m0).expect("independent rows form an invertible matrix");
    let mut rays: Vec<Vec<Rational>> = (0..r)
        .map(|j| {
            let col: Vec<Rational> = (0..r).map(|i| minv[i][j].clone()).collect();
            primitive(&col).expect("a column of an invertible matrix is nonzero")
        })
        .collect();
    if rays.len() > budget.max_rays {
        return Err(format!(
            "double description ray budget exceeded: {} rays needed at initialization, budget is {}",
            rays.len(),
            budget.max_rays
        ));
    }

    // Insert the remaining rows one at a time.
    let mut processed: Vec<usize> = init.clone();
    for k in 0..rrows.len() {
        if init.contains(&k) {
            continue;
        }
        let row = &rrows[k];
        let vals: Vec<Rational> = rays.iter().map(|y| dot(row, y)).collect();
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        let mut new_rays: Vec<Vec<Rational>> = Vec::new();
        for (i, v) in vals.iter().enumerate() {
            match sign(v) {
                1 => {
                    pos.push(i);
                    new_rays.push(rays[i].clone());
                }
                0 => new_rays.push(rays[i].clone()),
                _ => neg.push(i),
            }
        }
        for &p in &pos {
            for &q in &neg {
                if adjacent(&rays[p], &rays[q], &rrows, &processed, r) {
                    // Positive combination of the adjacent pair, tight on `row`:
                    let w: Vec<Rational> = (0..r)
                        .map(|j| {
                            let t1 = &vals[p] * &rays[q][j];
                            let t2 = &vals[q] * &rays[p][j];
                            &t1 - &t2
                        })
                        .collect();
                    let w = primitive(&w)
                        .expect("a positive combination of two distinct extreme rays is nonzero");
                    if !new_rays.contains(&w) {
                        new_rays.push(w);
                    }
                    if new_rays.len() > budget.max_rays {
                        return Err(format!(
                            "double description ray budget exceeded: more than {} intermediate rays",
                            budget.max_rays
                        ));
                    }
                }
            }
        }
        rays = new_rays;
        processed.push(k);
    }

    // Lift the rays from W back to Q^n (zero on the non-pivot coordinates).
    let lifted = rays
        .into_iter()
        .map(|z| {
            let mut y = vec![Rational::zero(); n];
            for (t, &c) in piv_cols.iter().enumerate() {
                y[c] = z[t].clone();
            }
            y
        })
        .collect();
    Ok(ConeGenerators {
        rays: lifted,
        lines,
    })
}

// ---------------------------------------------------------------------------
// HPolyhedron / VPolyhedron
// ---------------------------------------------------------------------------

impl HPolyhedron {
    /// Build an H-representation, validating coefficient vector lengths.
    pub fn new(
        ambient_dim: usize,
        inequalities: Vec<(Vec<Rational>, Rational)>,
        equations: Vec<(Vec<Rational>, Rational)>,
    ) -> Result<Self, String> {
        for (a, _) in inequalities.iter().chain(equations.iter()) {
            if a.len() != ambient_dim {
                return Err(format!(
                    "coefficient vector has length {}, expected ambient dimension {}",
                    a.len(),
                    ambient_dim
                ));
            }
        }
        Ok(HPolyhedron {
            ambient_dim,
            inequalities,
            equations,
        })
    }

    /// The canonical H-representation of the empty polyhedron: `0·x <= -1`.
    pub fn canonical_empty(ambient_dim: usize) -> Self {
        HPolyhedron {
            ambient_dim,
            inequalities: vec![(vec![Rational::zero(); ambient_dim], -Rational::one())],
            equations: Vec::new(),
        }
    }

    /// Whether this is exactly the canonical empty form.
    pub fn is_canonical_empty(&self) -> bool {
        *self == Self::canonical_empty(self.ambient_dim)
    }

    /// Exact membership test `x ∈ P`.
    ///
    /// PANICS on a wrong-length point: a `debug_assert` here would compile
    /// out in release and let `zip` silently truncate the dot products,
    /// answering membership for a *different* point.
    pub fn contains(&self, x: &[Rational]) -> bool {
        assert_eq!(
            x.len(),
            self.ambient_dim,
            "HPolyhedron::contains: point has dimension {} but the polyhedron lives in R^{}",
            x.len(),
            self.ambient_dim
        );
        self.inequalities
            .iter()
            .all(|(a, b)| sign(&(&dot(a, x) - b)) <= 0)
            && self.equations.iter().all(|(a, b)| dot(a, x) == *b)
    }

    /// Canonical form: see the module documentation. Detects contradictory
    /// equations and constant-infeasible inequalities (both yield the
    /// canonical empty form) but performs no general feasibility check.
    pub fn canonicalize(&self) -> HPolyhedron {
        let d = self.ambient_dim;

        // Equations: exact RREF of the augmented rows [a | b].
        let mut eq: Vec<Vec<Rational>> = self
            .equations
            .iter()
            .map(|(a, b)| {
                let mut row = a.clone();
                row.push(b.clone());
                row
            })
            .collect();
        let pivots = rref(&mut eq);
        if pivots.last() == Some(&d) {
            // A pivot in the constant column is the row 0·x = 1: contradictory.
            return HPolyhedron::canonical_empty(d);
        }
        let eq_rows: Vec<Vec<Rational>> = eq.into_iter().take(pivots.len()).collect();

        // Inequalities: eliminate the equation pivot variables (RREF pivot
        // coefficients are 1), then scale to primitive integer form.
        let mut ineq_rows: Vec<Vec<Rational>> = Vec::new();
        for (a, b) in &self.inequalities {
            let mut row = a.clone();
            row.push(b.clone());
            for (erow, &p) in eq_rows.iter().zip(&pivots) {
                if !row[p].is_zero() {
                    let f = row[p].clone();
                    for j in 0..=d {
                        let t = &f * &erow[j];
                        row[j] = &row[j] - &t;
                    }
                }
            }
            if row[..d].iter().all(|x| x.is_zero()) {
                if sign(&row[d]) < 0 {
                    // 0·x <= negative constant: infeasible.
                    return HPolyhedron::canonical_empty(d);
                }
                continue; // trivially true: drop
            }
            let p = primitive(&row).expect("row has a nonzero normal");
            if !ineq_rows.contains(&p) {
                ineq_rows.push(p);
            }
        }
        ineq_rows.sort();

        let split = |mut row: Vec<Rational>| {
            let b = row.pop().expect("augmented row is nonempty");
            (row, b)
        };
        let equations = eq_rows
            .into_iter()
            .map(|row| split(primitive(&row).expect("RREF equation row is nonzero")))
            .collect();
        let inequalities = ineq_rows.into_iter().map(split).collect();
        HPolyhedron {
            ambient_dim: d,
            inequalities,
            equations,
        }
    }
}

impl VPolyhedron {
    /// Build a V-representation, validating vector lengths.
    pub fn new(
        ambient_dim: usize,
        vertices: Vec<Vec<Rational>>,
        rays: Vec<Vec<Rational>>,
        lines: Vec<Vec<Rational>>,
    ) -> Result<Self, String> {
        for v in vertices.iter().chain(rays.iter()).chain(lines.iter()) {
            if v.len() != ambient_dim {
                return Err(format!(
                    "vector has length {}, expected ambient dimension {}",
                    v.len(),
                    ambient_dim
                ));
            }
        }
        Ok(VPolyhedron {
            ambient_dim,
            vertices,
            rays,
            lines,
        })
    }

    /// A polytope given by its (candidate) vertices.
    pub fn from_vertices(ambient_dim: usize, vertices: Vec<Vec<Rational>>) -> Result<Self, String> {
        Self::new(ambient_dim, vertices, Vec::new(), Vec::new())
    }

    /// The empty polyhedron.
    pub fn empty(ambient_dim: usize) -> Self {
        VPolyhedron {
            ambient_dim,
            vertices: Vec::new(),
            rays: Vec::new(),
            lines: Vec::new(),
        }
    }

    /// Whether this represents the empty polyhedron (no vertices).
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Canonical form: see the module documentation. Does *not* remove
    /// non-extreme points — canonical forms are unique only for minimal
    /// representations (which the conversions produce).
    pub fn canonicalize(&self) -> VPolyhedron {
        let d = self.ambient_dim;
        if self.vertices.is_empty() {
            return VPolyhedron::empty(d);
        }
        let mut vertices = self.vertices.clone();
        vertices.sort();
        vertices.dedup();
        let mut rays: Vec<Vec<Rational>> = self.rays.iter().filter_map(|r| primitive(r)).collect();
        rays.sort();
        rays.dedup();
        let mut lmat = self.lines.clone();
        let _ = rref(&mut lmat);
        let mut lines: Vec<Vec<Rational>> =
            lmat.iter().filter_map(|row| primitive(row)).collect();
        lines.sort();
        VPolyhedron {
            ambient_dim: d,
            vertices,
            rays,
            lines,
        }
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// Convert an H-representation to the minimal canonical V-representation.
///
/// Returns the empty `VPolyhedron` when the constraints are infeasible.
/// Errors only on malformed input or when `budget` is exceeded.
pub fn h_to_v(h: &HPolyhedron, budget: &DdBudget) -> Result<VPolyhedron, String> {
    let d = h.ambient_dim;
    let n = d + 1;
    let mut rows: Vec<Vec<Rational>> = Vec::new();
    // Homogenization variable: x0 >= 0.
    let mut e0 = vec![Rational::zero(); n];
    e0[0] = Rational::one();
    rows.push(e0);
    for (a, b) in &h.inequalities {
        if a.len() != d {
            return Err(format!(
                "inequality coefficient vector has length {}, expected {}",
                a.len(),
                d
            ));
        }
        // a·x <= b  homogenizes to  b·x0 - a·x >= 0.
        let mut row = Vec::with_capacity(n);
        row.push(b.clone());
        for ai in a {
            row.push(-ai.clone());
        }
        rows.push(row);
    }
    for (c, b) in &h.equations {
        if c.len() != d {
            return Err(format!(
                "equation coefficient vector has length {}, expected {}",
                c.len(),
                d
            ));
        }
        // c·x = b  homogenizes to the pair  ±(b·x0 - c·x) >= 0.
        let mut row = Vec::with_capacity(n);
        row.push(b.clone());
        for ci in c {
            row.push(-ci.clone());
        }
        rows.push(row.iter().map(|x| -x.clone()).collect());
        rows.push(row);
    }

    let gens = cone_dd(&rows, n, budget)?;
    let mut vertices: Vec<Vec<Rational>> = Vec::new();
    let mut rays: Vec<Vec<Rational>> = Vec::new();
    for y in gens.rays {
        match sign(&y[0]) {
            1 => vertices.push((1..n).map(|j| &y[j] / &y[0]).collect()),
            0 => rays.push(y[1..].to_vec()),
            _ => unreachable!("x0 >= 0 is a constraint row of the homogenized cone"),
        }
    }
    if vertices.is_empty() {
        // No generator at positive height: the polyhedron is empty. The
        // height-0 rays/lines only describe the recession of an infeasible
        // system, so reporting them would be dishonest.
        return Ok(VPolyhedron::empty(d));
    }
    let lines: Vec<Vec<Rational>> = gens
        .lines
        .into_iter()
        .map(|l| {
            debug_assert!(l[0].is_zero(), "lineality is at height 0 (x0 >= 0 row)");
            l[1..].to_vec()
        })
        .collect();
    Ok(VPolyhedron {
        ambient_dim: d,
        vertices,
        rays,
        lines,
    }
    .canonicalize())
}

/// Convert a V-representation to the minimal canonical H-representation
/// `{equations, inequalities}` via double description on the dual cone.
///
/// Errors only on malformed input or when `budget` is exceeded.
pub fn v_to_h(v: &VPolyhedron, budget: &DdBudget) -> Result<HPolyhedron, String> {
    let d = v.ambient_dim;
    if v.vertices.is_empty() {
        // conv(∅) = ∅ by convention, regardless of rays/lines.
        return Ok(HPolyhedron::canonical_empty(d));
    }
    let n = d + 1;
    let mut rows: Vec<Vec<Rational>> = Vec::new();
    for (kind, vecs, height) in [
        ("vertex", &v.vertices, Rational::one()),
        ("ray", &v.rays, Rational::zero()),
    ] {
        for p in vecs {
            if p.len() != d {
                return Err(format!(
                    "{} vector has length {}, expected {}",
                    kind,
                    p.len(),
                    d
                ));
            }
            let mut row = Vec::with_capacity(n);
            row.push(height.clone());
            row.extend(p.iter().cloned());
            rows.push(row);
        }
    }
    for l in &v.lines {
        if l.len() != d {
            return Err(format!(
                "line vector has length {}, expected {}",
                l.len(),
                d
            ));
        }
        let mut row = Vec::with_capacity(n);
        row.push(Rational::zero());
        row.extend(l.iter().cloned());
        rows.push(row.iter().map(|x| -x.clone()).collect());
        rows.push(row);
    }

    // Dual cone K* = {c : G c >= 0}: its extreme rays are the irredundant
    // inequalities of the homogenized cone K, its lineality the equations.
    let gens = cone_dd(&rows, n, budget)?;
    let inequalities = gens
        .rays
        .into_iter()
        .map(|c| {
            // c0·x0 + a·x >= 0 on K, so at height 1: (-a)·x <= c0.
            let a: Vec<Rational> = c[1..].iter().map(|x| -x.clone()).collect();
            (a, c[0].clone())
        })
        .collect();
    let equations = gens
        .lines
        .into_iter()
        .map(|c| {
            // c0·x0 + a·x = 0 on K, so at height 1: a·x = -c0.
            (c[1..].to_vec(), -c[0].clone())
        })
        .collect();
    Ok(HPolyhedron {
        ambient_dim: d,
        inequalities,
        equations,
    }
    .canonicalize())
}

// ---------------------------------------------------------------------------
// Tests. Every expected vertex and facet below was derived independently in
// exact python fractions (brute force over d-subsets of points -> hyperplane
// -> all points weakly on one side; vertices confirmed by the tight-facet
// rank criterion) before being asserted here.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_polytope::hull_vertices;

    fn q(x: i64) -> Rational {
        Rational::from_i64(x)
    }

    fn qv(v: &[i64]) -> Vec<Rational> {
        v.iter().map(|&x| q(x)).collect()
    }

    fn hp(d: usize, ineqs: &[(&[i64], i64)], eqs: &[(&[i64], i64)]) -> HPolyhedron {
        HPolyhedron::new(
            d,
            ineqs.iter().map(|(a, b)| (qv(a), q(*b))).collect(),
            eqs.iter().map(|(a, b)| (qv(a), q(*b))).collect(),
        )
        .unwrap()
    }

    fn vp(d: usize, pts: &[&[i64]]) -> VPolyhedron {
        VPolyhedron::from_vertices(d, pts.iter().map(|p| qv(p)).collect()).unwrap()
    }

    fn rows(data: &[(&[i64], i64)]) -> Vec<(Vec<Rational>, Rational)> {
        data.iter().map(|(a, b)| (qv(a), q(*b))).collect()
    }

    fn pts(data: &[&[i64]]) -> Vec<Vec<Rational>> {
        data.iter().map(|p| qv(p)).collect()
    }

    /// Convert exact rational vertices (which must be integral) to Integer
    /// points for cross-validation against `hull_vertices`.
    fn to_integer_points(vs: &[Vec<Rational>]) -> Vec<Vec<Integer>> {
        vs.iter()
            .map(|v| {
                v.iter()
                    .map(|x| {
                        assert!(x.is_integer(), "expected integral vertex");
                        x.numerator().clone()
                    })
                    .collect()
            })
            .collect()
    }

    fn assert_cross_validated_against_hull_vertices(
        input_points: &[Vec<Integer>],
        dd_vertices: &[Vec<Rational>],
    ) {
        let mut expected = hull_vertices(input_points);
        expected.sort();
        let mut got = to_integer_points(dd_vertices);
        got.sort();
        assert_eq!(got, expected, "double description vs hull_vertices disagree");
    }

    // -- python-derived data ------------------------------------------------

    const CUBE_FACETS: &[(&[i64], i64)] = &[
        (&[-1, 0, 0], 1),
        (&[0, -1, 0], 1),
        (&[0, 0, -1], 1),
        (&[0, 0, 1], 1),
        (&[0, 1, 0], 1),
        (&[1, 0, 0], 1),
    ];
    const CUBE_VERTICES: &[&[i64]] = &[
        &[-1, -1, -1],
        &[-1, -1, 1],
        &[-1, 1, -1],
        &[-1, 1, 1],
        &[1, -1, -1],
        &[1, -1, 1],
        &[1, 1, -1],
        &[1, 1, 1],
    ];

    const SIMPLEX4_FACETS: &[(&[i64], i64)] = &[
        (&[-1, 0, 0, 0], 0),
        (&[0, -1, 0, 0], 0),
        (&[0, 0, -1, 0], 0),
        (&[0, 0, 0, -1], 0),
        (&[1, 1, 1, 1], 1),
    ];
    const SIMPLEX4_VERTICES: &[&[i64]] = &[
        &[0, 0, 0, 0],
        &[0, 0, 0, 1],
        &[0, 0, 1, 0],
        &[0, 1, 0, 0],
        &[1, 0, 0, 0],
    ];

    const OCTAHEDRON_FACETS: &[(&[i64], i64)] = &[
        (&[-1, -1, -1], 1),
        (&[-1, -1, 1], 1),
        (&[-1, 1, -1], 1),
        (&[-1, 1, 1], 1),
        (&[1, -1, -1], 1),
        (&[1, -1, 1], 1),
        (&[1, 1, -1], 1),
        (&[1, 1, 1], 1),
    ];
    const OCTAHEDRON_VERTICES: &[&[i64]] = &[
        &[-1, 0, 0],
        &[0, -1, 0],
        &[0, 0, -1],
        &[0, 0, 1],
        &[0, 1, 0],
        &[1, 0, 0],
    ];

    const CELL24_FACETS: &[(&[i64], i64)] = &[
        (&[-1, -1, -1, -1], 2),
        (&[-1, -1, -1, 1], 2),
        (&[-1, -1, 1, -1], 2),
        (&[-1, -1, 1, 1], 2),
        (&[-1, 0, 0, 0], 1),
        (&[-1, 1, -1, -1], 2),
        (&[-1, 1, -1, 1], 2),
        (&[-1, 1, 1, -1], 2),
        (&[-1, 1, 1, 1], 2),
        (&[0, -1, 0, 0], 1),
        (&[0, 0, -1, 0], 1),
        (&[0, 0, 0, -1], 1),
        (&[0, 0, 0, 1], 1),
        (&[0, 0, 1, 0], 1),
        (&[0, 1, 0, 0], 1),
        (&[1, -1, -1, -1], 2),
        (&[1, -1, -1, 1], 2),
        (&[1, -1, 1, -1], 2),
        (&[1, -1, 1, 1], 2),
        (&[1, 0, 0, 0], 1),
        (&[1, 1, -1, -1], 2),
        (&[1, 1, -1, 1], 2),
        (&[1, 1, 1, -1], 2),
        (&[1, 1, 1, 1], 2),
    ];
    const CELL24_VERTICES: &[&[i64]] = &[
        &[-1, -1, 0, 0],
        &[-1, 0, -1, 0],
        &[-1, 0, 0, -1],
        &[-1, 0, 0, 1],
        &[-1, 0, 1, 0],
        &[-1, 1, 0, 0],
        &[0, -1, -1, 0],
        &[0, -1, 0, -1],
        &[0, -1, 0, 1],
        &[0, -1, 1, 0],
        &[0, 0, -1, -1],
        &[0, 0, -1, 1],
        &[0, 0, 1, -1],
        &[0, 0, 1, 1],
        &[0, 1, -1, 0],
        &[0, 1, 0, -1],
        &[0, 1, 0, 1],
        &[0, 1, 1, 0],
        &[1, -1, 0, 0],
        &[1, 0, -1, 0],
        &[1, 0, 0, -1],
        &[1, 0, 0, 1],
        &[1, 0, 1, 0],
        &[1, 1, 0, 0],
    ];

    const RAND9_FACETS: &[(&[i64], i64)] = &[
        (&[-20, 25, -29], 125),
        (&[-10, -9, -4], 41),
        (&[-8, 5, 9], 45),
        (&[-4, -10, -9], 41),
        (&[-4, 5, 5], 25),
        (&[-1, -1, -1], 4),
        (&[-1, 3, 3], 15),
        (&[3, -1, 3], 15),
        (&[3, 3, -1], 15),
        (&[5, -4, 5], 25),
        (&[5, 5, -4], 25),
        (&[7, -12, 15], 75),
        (&[9, -8, 5], 45),
        (&[15, 7, -12], 75),
    ];
    const RAND9_VERTICES: &[&[i64]] = &[
        &[-5, 1, 0],
        &[-3, -3, 4],
        &[0, -5, 1],
        &[0, 0, 5],
        &[0, 5, 0],
        &[1, 0, -5],
        &[3, 3, 3],
        &[4, -3, -3],
        &[5, 0, 0],
    ];

    /// Full round trip on a full-dimensional polytope: both directions,
    /// exact canonical equality, plus exact vertex/facet content checks
    /// against independently derived data.
    fn full_dim_gate(
        d: usize,
        facets: &[(&[i64], i64)],
        vertices: &[&[i64]],
        budget: &DdBudget,
    ) -> (HPolyhedron, VPolyhedron) {
        let h = hp(d, facets, &[]);
        let hc = h.canonicalize();
        // Input facet lists are already canonical (primitive, sorted):
        assert_eq!(hc.inequalities, rows(facets));
        assert!(hc.equations.is_empty());

        let v = h_to_v(&h, budget).unwrap();
        assert_eq!(v.vertices, pts(vertices), "h_to_v vertices");
        assert!(v.rays.is_empty(), "polytope has no rays");
        assert!(v.lines.is_empty(), "polytope has no lines");

        // Round trip 1: v_to_h(h_to_v(H)) == canonicalize(H), exactly.
        let h2 = v_to_h(&v, budget).unwrap();
        assert_eq!(h2, hc, "v_to_h(h_to_v(H)) != canonical H");
        assert_eq!(h2.inequalities, rows(facets), "facet content");

        // Round trip 2: h_to_v(v_to_h(V)) == canonicalize(V), exactly,
        // starting from the raw vertex list.
        let vraw = vp(d, vertices);
        let h3 = v_to_h(&vraw, budget).unwrap();
        assert_eq!(h3, hc);
        let v2 = h_to_v(&h3, budget).unwrap();
        assert_eq!(v2, vraw.canonicalize());
        assert_eq!(v2, v);
        (hc, v)
    }

    #[test]
    fn test_cube_round_trip() {
        let b = DdBudget::default();
        let (_, v) = full_dim_gate(3, CUBE_FACETS, CUBE_VERTICES, &b);
        assert_eq!(v.vertices.len(), 8);
        assert_cross_validated_against_hull_vertices(
            &to_integer_points(&pts(CUBE_VERTICES)),
            &v.vertices,
        );
    }

    #[test]
    fn test_simplex4_round_trip() {
        let b = DdBudget::default();
        let (h, v) = full_dim_gate(4, SIMPLEX4_FACETS, SIMPLEX4_VERTICES, &b);
        assert_eq!(h.inequalities.len(), 5);
        assert_eq!(v.vertices.len(), 5);
        assert_cross_validated_against_hull_vertices(
            &to_integer_points(&pts(SIMPLEX4_VERTICES)),
            &v.vertices,
        );
    }

    #[test]
    fn test_octahedron_round_trip() {
        let b = DdBudget::default();
        let (h, v) = full_dim_gate(3, OCTAHEDRON_FACETS, OCTAHEDRON_VERTICES, &b);
        assert_eq!(h.inequalities.len(), 8);
        assert_eq!(v.vertices.len(), 6);
        assert_cross_validated_against_hull_vertices(
            &to_integer_points(&pts(OCTAHEDRON_VERTICES)),
            &v.vertices,
        );
    }

    #[test]
    fn test_24_cell_round_trip() {
        // The self-dual gem: 24 vertices, 24 facets, in R^4.
        let b = DdBudget::default();
        let (h, v) = full_dim_gate(4, CELL24_FACETS, CELL24_VERTICES, &b);
        assert_eq!(h.inequalities.len(), 24);
        assert_eq!(v.vertices.len(), 24);
    }

    /// hull_vertices is exponential in ambient_dim + 1 and needs ~80 s on the
    /// 24-cell in debug mode, so this cross-validation runs only on demand
    /// (`cargo test -- --ignored`). It passes.
    #[test]
    #[ignore = "expensive (~80s debug): hull_vertices on 24 points in R^4"]
    fn test_24_cell_cross_validated_against_hull_vertices() {
        let b = DdBudget::default();
        let h = hp(4, CELL24_FACETS, &[]);
        let v = h_to_v(&h, &b).unwrap();
        assert_cross_validated_against_hull_vertices(
            &to_integer_points(&pts(CELL24_VERTICES)),
            &v.vertices,
        );
    }

    #[test]
    fn test_rand9_round_trip() {
        // 9 integer points in convex position; the 14 facets were derived by
        // brute force in python (every 3-subset -> plane -> one-side check).
        let b = DdBudget::default();
        let (h, v) = full_dim_gate(3, RAND9_FACETS, RAND9_VERTICES, &b);
        assert_eq!(h.inequalities.len(), 14);
        assert_eq!(v.vertices.len(), 9);
        assert_cross_validated_against_hull_vertices(
            &to_integer_points(&pts(RAND9_VERTICES)),
            &v.vertices,
        );
    }

    #[test]
    fn test_degenerate_inputs_are_filtered() {
        // rand9 plus junk: a duplicate vertex, the origin (verified strictly
        // interior in python), and (5/2, 5/2, 0) (verified in python to lie on
        // the relative interior of the edge (5,0,0)-(0,5,0), i.e. on the
        // boundary but NOT a vertex). None may survive as a vertex.
        let b = DdBudget::default();
        let mut points = pts(RAND9_VERTICES);
        points.push(qv(&[5, 0, 0])); // duplicate
        points.push(qv(&[0, 0, 0])); // interior point
        points.push(vec![
            Rational::new(5, 2).unwrap(),
            Rational::new(5, 2).unwrap(),
            q(0),
        ]); // relative interior of an edge
        let v = VPolyhedron::from_vertices(3, points).unwrap();
        let h = v_to_h(&v, &b).unwrap();
        assert_eq!(h.inequalities, rows(RAND9_FACETS));
        assert!(h.equations.is_empty());
        let v2 = h_to_v(&h, &b).unwrap();
        assert_eq!(v2.vertices, pts(RAND9_VERTICES), "junk points must vanish");

        // Cross-validate against hull_vertices on the integral junk points.
        let mut int_points = to_integer_points(&pts(RAND9_VERTICES));
        int_points.push(to_integer_points(&[qv(&[5, 0, 0])])[0].clone());
        int_points.push(to_integer_points(&[qv(&[0, 0, 0])])[0].clone());
        assert_cross_validated_against_hull_vertices(&int_points, &v2.vertices);
    }

    #[test]
    fn test_collinear_points_reduce_to_segment() {
        // (0,0), (1,1), (2,2), (5,5) are collinear; the hull is the segment
        // from (0,0) to (5,5).
        let b = DdBudget::default();
        let v = vp(2, &[&[0, 0], &[1, 1], &[2, 2], &[5, 5]]);
        let h = v_to_h(&v, &b).unwrap();
        // Affine hull: x - y = 0 (canonical: pivot column 0, primitive).
        assert_eq!(h.equations, rows(&[(&[1, -1], 0)]));
        let v2 = h_to_v(&h, &b).unwrap();
        assert_eq!(v2.vertices, pts(&[&[0, 0], &[5, 5]]));
        assert!(v2.rays.is_empty() && v2.lines.is_empty());
        // Round trip.
        assert_eq!(v_to_h(&v2, &b).unwrap(), h);
        // Two independent implementations agree on the vertex set:
        assert_cross_validated_against_hull_vertices(
            &to_integer_points(&pts(&[&[0, 0], &[1, 1], &[2, 2], &[5, 5]])),
            &v2.vertices,
        );
    }

    #[test]
    fn test_lower_dim_triangle_in_r3() {
        // conv(e1, e2, e3): affine hull x1 + x2 + x3 = 1 plus three facet
        // inequalities. Canonical form derived independently in python:
        // equation (1,1,1 | 1); inequalities (after eliminating the pivot
        // variable x1): (0,-1,0 | 0), (0,0,-1 | 0), (0,1,1 | 1).
        let b = DdBudget::default();
        let h_in = hp(
            3,
            &[(&[-1, 0, 0], 0), (&[0, -1, 0], 0), (&[0, 0, -1], 0)],
            &[(&[1, 1, 1], 1)],
        );
        let expected_eqs = rows(&[(&[1, 1, 1], 1)]);
        let expected_ineqs = rows(&[(&[0, -1, 0], 0), (&[0, 0, -1], 0), (&[0, 1, 1], 1)]);
        let hc = h_in.canonicalize();
        assert_eq!(hc.equations, expected_eqs);
        assert_eq!(hc.inequalities, expected_ineqs);

        let v = h_to_v(&h_in, &b).unwrap();
        assert_eq!(v.vertices, pts(&[&[0, 0, 1], &[0, 1, 0], &[1, 0, 0]]));
        assert!(v.rays.is_empty() && v.lines.is_empty());

        // Round trips in both directions.
        let h2 = v_to_h(&v, &b).unwrap();
        assert_eq!(h2, hc);
        let vraw = vp(3, &[&[1, 0, 0], &[0, 1, 0], &[0, 0, 1]]);
        let h3 = v_to_h(&vraw, &b).unwrap();
        assert_eq!(h3, hc);
        assert_eq!(h_to_v(&h3, &b).unwrap(), v);

        // Semantic checks: every inequality is valid on all vertices and
        // tight on exactly two of the three (an edge = facet of a 2-polytope).
        for (a, bb) in &hc.inequalities {
            let tight = v
                .vertices
                .iter()
                .filter(|x| dot(a, x) == *bb)
                .count();
            assert!(v.vertices.iter().all(|x| sign(&(&dot(a, x) - bb)) <= 0));
            assert_eq!(tight, 2);
        }
        assert_cross_validated_against_hull_vertices(
            &to_integer_points(&pts(&[&[1, 0, 0], &[0, 1, 0], &[0, 0, 1]])),
            &v.vertices,
        );
    }

    #[test]
    fn test_lower_dim_segment_in_r3() {
        // conv((1,2,3), (4,5,6)). Canonical form derived independently:
        // equations x1 - x3 = -2 and x2 - x3 = -1 (in RREF pivot-column
        // order); inequalities (endpoints, reduced modulo the equations):
        // (0,0,-1 | -3), (0,0,1 | 6).
        let b = DdBudget::default();
        let v_in = vp(3, &[&[1, 2, 3], &[4, 5, 6]]);
        let h = v_to_h(&v_in, &b).unwrap();
        assert_eq!(h.equations, rows(&[(&[1, 0, -1], -2), (&[0, 1, -1], -1)]));
        assert_eq!(h.inequalities, rows(&[(&[0, 0, -1], -3), (&[0, 0, 1], 6)]));
        let v2 = h_to_v(&h, &b).unwrap();
        assert_eq!(v2.vertices, pts(&[&[1, 2, 3], &[4, 5, 6]]));
        assert!(v2.rays.is_empty() && v2.lines.is_empty());
        assert_eq!(v_to_h(&v2, &b).unwrap(), h);
        // Each inequality is tight at exactly one endpoint (the two facets of
        // a 1-polytope), valid at both.
        for (a, bb) in &h.inequalities {
            assert!(v2.vertices.iter().all(|x| sign(&(&dot(a, x) - bb)) <= 0));
            assert_eq!(v2.vertices.iter().filter(|x| dot(a, x) == *bb).count(), 1);
        }
    }

    #[test]
    fn test_single_point() {
        // conv((3,1,4)): affine hull is the point itself; no inequalities
        // survive canonicalization (all reduce to constants).
        let b = DdBudget::default();
        let v_in = vp(3, &[&[3, 1, 4]]);
        let h = v_to_h(&v_in, &b).unwrap();
        assert_eq!(
            h.equations,
            rows(&[(&[1, 0, 0], 3), (&[0, 1, 0], 1), (&[0, 0, 1], 4)])
        );
        assert!(h.inequalities.is_empty());
        let v2 = h_to_v(&h, &b).unwrap();
        assert_eq!(v2.vertices, pts(&[&[3, 1, 4]]));
        assert!(v2.rays.is_empty() && v2.lines.is_empty());
        assert_eq!(v_to_h(&v2, &b).unwrap(), h);
    }

    #[test]
    fn test_empty_from_contradictory_halfspaces() {
        let b = DdBudget::default();
        // x <= -1 and x >= 1: empty, recession cone trivial.
        let h = hp(2, &[(&[1, 0], -1), (&[-1, 0], -1)], &[]);
        let v = h_to_v(&h, &b).unwrap();
        assert!(v.is_empty());
        assert!(v.vertices.is_empty() && v.rays.is_empty() && v.lines.is_empty());
        // The canonical empty round-trips exactly.
        let h2 = v_to_h(&v, &b).unwrap();
        assert_eq!(h2, HPolyhedron::canonical_empty(2));
        assert!(h2.is_canonical_empty());
        assert_eq!(h_to_v(&h2, &b).unwrap(), VPolyhedron::empty(2));
        assert_eq!(v_to_h(&VPolyhedron::empty(2), &b).unwrap(), h2);

        // x <= -1 and x >= 2: empty, but the homogenized cone has a nonzero
        // lineality (the free y direction); it must still come back empty and
        // must not leak recession directions of an infeasible system.
        let h3 = hp(2, &[(&[1, 0], -1), (&[-1, 0], -2)], &[]);
        let v3 = h_to_v(&h3, &b).unwrap();
        assert!(v3.is_empty() && v3.rays.is_empty() && v3.lines.is_empty());

        // Contradictory equations detected in canonicalization.
        let h4 = hp(2, &[], &[(&[1, 0], 1), (&[1, 0], 2)]);
        assert!(h4.canonicalize().is_canonical_empty());
    }

    #[test]
    fn test_unbounded_orthant_has_rays() {
        // {x >= 0, y >= 0}: vertex at the origin, extreme rays e1 and e2.
        let b = DdBudget::default();
        let h = hp(2, &[(&[-1, 0], 0), (&[0, -1], 0)], &[]);
        let v = h_to_v(&h, &b).unwrap();
        assert_eq!(v.vertices, pts(&[&[0, 0]]));
        assert_eq!(v.rays, pts(&[&[0, 1], &[1, 0]]));
        assert!(v.lines.is_empty());
        // Round trip: the trivial facet-at-infinity (x0 >= 0 appears as a
        // facet of the homogenized cone of an unbounded polyhedron) must be
        // dropped by canonicalization.
        let h2 = v_to_h(&v, &b).unwrap();
        assert_eq!(h2, h.canonicalize());
        assert_eq!(h2.inequalities, rows(&[(&[-1, 0], 0), (&[0, -1], 0)]));
        assert_eq!(h_to_v(&h2, &b).unwrap(), v);
    }

    #[test]
    fn test_halfplane_has_line() {
        // {y >= 0} in R^2: no true vertex; representative point (0,0), ray
        // +e2, lineality e1.
        let b = DdBudget::default();
        let h = hp(2, &[(&[0, -1], 0)], &[]);
        let v = h_to_v(&h, &b).unwrap();
        assert_eq!(v.vertices, pts(&[&[0, 0]]));
        assert_eq!(v.rays, pts(&[&[0, 1]]));
        assert_eq!(v.lines, pts(&[&[1, 0]]));
        let h2 = v_to_h(&v, &b).unwrap();
        assert_eq!(h2, h.canonicalize());
        assert_eq!(h_to_v(&h2, &b).unwrap(), v);
    }

    #[test]
    fn test_whole_space() {
        // No constraints at all: R^2 = point + two lines, and back.
        let b = DdBudget::default();
        let h = HPolyhedron::new(2, Vec::new(), Vec::new()).unwrap();
        let v = h_to_v(&h, &b).unwrap();
        assert_eq!(v.vertices, pts(&[&[0, 0]]));
        assert!(v.rays.is_empty());
        assert_eq!(v.lines, pts(&[&[0, 1], &[1, 0]]));
        let h2 = v_to_h(&v, &b).unwrap();
        assert!(h2.inequalities.is_empty() && h2.equations.is_empty());
    }

    #[test]
    fn test_budget_trips_honestly() {
        let b_tiny = DdBudget::new(2);
        // The homogenized cube cone lives in Q^4 and needs 4 initial rays:
        // trips at initialization.
        let cube = hp(3, CUBE_FACETS, &[]);
        let err = h_to_v(&cube, &b_tiny).unwrap_err();
        assert!(err.contains("budget"), "unexpected message: {err}");

        // The 24-cell cone has 24 extreme rays (init needs only 5): a budget
        // of 6 must trip during insertion, not at initialization.
        let cell24 = hp(4, CELL24_FACETS, &[]);
        let err = h_to_v(&cell24, &DdBudget::new(6)).unwrap_err();
        assert!(err.contains("budget"), "unexpected message: {err}");

        // Same conversions succeed under the default budget.
        assert!(h_to_v(&cube, &DdBudget::default()).is_ok());
        assert!(h_to_v(&cell24, &DdBudget::default()).is_ok());

        // v_to_h honors the budget too (dual cone in Q^4, 4 initial rays).
        let octa = vp(3, OCTAHEDRON_VERTICES);
        let err = v_to_h(&octa, &DdBudget::new(3)).unwrap_err();
        assert!(err.contains("budget"), "unexpected message: {err}");
    }

    #[test]
    fn test_canonicalize_idempotent_and_contains() {
        let b = DdBudget::default();
        let h = hp(3, RAND9_FACETS, &[]);
        let hc = h.canonicalize();
        assert_eq!(hc.canonicalize(), hc);
        let v = h_to_v(&h, &b).unwrap();
        assert_eq!(v.canonicalize(), v);
        // contains() agrees with the derived data: all vertices in, the
        // far-away point out, the interior origin in.
        for p in &v.vertices {
            assert!(hc.contains(p));
        }
        assert!(hc.contains(&qv(&[0, 0, 0])));
        assert!(!hc.contains(&qv(&[100, 0, 0])));
    }
}
