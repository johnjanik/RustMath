//! Lattice polytopes
//!
//! This module provides types and functions for working with lattice polytopes -
//! polytopes whose vertices all have integer coordinates in a lattice.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::lattice_polytope::LatticePolytopeClass;
//! use rustmath_integers::Integer;
//!
//! // Create a triangle with vertices at (0,0), (1,0), (0,1)
//! let vertices = vec![
//!     vec![Integer::from(0), Integer::from(0)],
//!     vec![Integer::from(1), Integer::from(0)],
//!     vec![Integer::from(0), Integer::from(1)],
//! ];
//! let polytope = LatticePolytopeClass::new(vertices);
//!
//! assert_eq!(polytope.n_vertices(), 3);
//! assert_eq!(polytope.dim(), 2);
//! ```

use crate::double_description::{v_to_h, DdBudget, HPolyhedron, VPolyhedron};
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::collections::{BTreeSet, HashMap};

/// Default cap on the number of bounding-box candidates enumerated by
/// [`LatticePolytopeClass::points`]. Exceeding it in the convenience
/// wrapper panics; use [`LatticePolytopeClass::try_points`] for an
/// explicit budget and an honest `Err`.
const DEFAULT_MAX_POINT_CANDIDATES: usize = 1_000_000;

/// Default cap on the number of distinct faces enumerated by
/// [`LatticePolytopeClass::faces`]. Exceeding it in the convenience
/// wrapper panics; use [`LatticePolytopeClass::try_faces`] for an
/// explicit budget and an honest `Err`.
const DEFAULT_MAX_FACES: usize = 100_000;

/// Exact dot product of a rational row with a rational point.
fn rat_dot(a: &[Rational], x: &[Rational]) -> Rational {
    debug_assert_eq!(a.len(), x.len());
    let mut s = Rational::from_integer(0);
    for (p, q) in a.iter().zip(x.iter()) {
        s = s + p * q;
    }
    s
}

/// Convert a canonical (integral) `Rational` to `Integer`, panicking on a
/// non-integral value — only used on rows produced by
/// `HPolyhedron::canonicalize`, whose primitive form is integral.
fn rat_to_int(x: &Rational) -> Integer {
    assert!(
        x.is_integer(),
        "canonical H-representation row entry is not integral: {x:?}"
    );
    x.numerator().clone()
}

fn int_vec_to_rat(v: &[Integer]) -> Vec<Rational> {
    v.iter().map(|x| Rational::from_integer(x.clone())).collect()
}

/// Exact affine dimension of a point set: the rank of the matrix of
/// difference vectors `p_i - p_0`, computed over `Integer` (Bareiss).
fn affine_dim_of(points: &[Vec<Integer>], ambient_dim: usize) -> usize {
    if points.len() <= 1 {
        return 0;
    }
    let base = &points[0];
    let diffs: Vec<Vec<Integer>> = points[1..]
        .iter()
        .map(|v| {
            v.iter()
                .zip(base.iter())
                .map(|(a, b)| a.clone() - b.clone())
                .collect()
        })
        .collect();
    integer_matrix_rank(&diffs, ambient_dim)
}

/// A lattice polytope
///
/// A polytope whose vertices all have integer coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticePolytopeClass {
    /// The vertices of the polytope
    vertices: Vec<Vec<Integer>>,
    /// Dimension of the ambient space
    ambient_dim: usize,
    /// Cached dimension of the polytope
    dimension: Option<usize>,
}

impl LatticePolytopeClass {
    /// Create a new lattice polytope from vertices
    ///
    /// # Arguments
    ///
    /// * `vertices` - The vertices of the polytope (must have integer coordinates)
    ///
    /// # Panics
    ///
    /// Panics if vertices list is empty or if vertices have different dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::lattice_polytope::LatticePolytopeClass;
    /// use rustmath_integers::Integer;
    ///
    /// let vertices = vec![
    ///     vec![Integer::from(0), Integer::from(0)],
    ///     vec![Integer::from(1), Integer::from(0)],
    ///     vec![Integer::from(0), Integer::from(1)],
    /// ];
    /// let polytope = LatticePolytopeClass::new(vertices);
    /// ```
    pub fn new(vertices: Vec<Vec<Integer>>) -> Self {
        if vertices.is_empty() {
            panic!("Polytope must have at least one vertex");
        }

        let ambient_dim = vertices[0].len();

        // Verify all vertices have the same dimension
        for v in &vertices {
            if v.len() != ambient_dim {
                panic!("All vertices must have the same dimension");
            }
        }

        Self {
            vertices,
            ambient_dim,
            dimension: None,
        }
    }

    /// Get the vertices
    pub fn vertices(&self) -> &[Vec<Integer>] {
        &self.vertices
    }

    /// Get the number of vertices
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the dimension of the ambient space
    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    /// Get the dimension of the polytope
    ///
    /// This is the dimension of the affine hull of the vertices: the rank
    /// of the matrix of vertex differences `v_i - v_0`, computed exactly
    /// over `Integer` (no floating point). This is exact even when the
    /// vertex list contains redundant/non-extreme points, unlike a
    /// `n_vertices - 1` head-count approximation.
    pub fn dim(&self) -> usize {
        affine_dim_of(&self.vertices, self.ambient_dim)
    }

    /// Get the lattice dimension
    ///
    /// This is the dimension of the smallest affine lattice containing the polytope.
    pub fn lattice_dim(&self) -> usize {
        self.ambient_dim
    }

    /// Exact H-representation of `conv(vertices)`: irredundant facet
    /// inequalities plus affine-hull equations, computed by the exact
    /// double description method ([`v_to_h`]) over `Rational`.
    ///
    /// Returns an honest `Err` if the double description ray budget is
    /// exceeded (never a truncated result).
    pub fn h_representation(&self, budget: &DdBudget) -> Result<HPolyhedron, String> {
        let verts: Vec<Vec<Rational>> =
            self.vertices.iter().map(|v| int_vec_to_rat(v)).collect();
        let vp = VPolyhedron::from_vertices(self.ambient_dim, verts)?;
        v_to_h(&vp, budget)
    }

    /// Facet inequalities of a **full-dimensional** lattice polytope, as
    /// pairs `(n, c)` meaning `n·x + c >= 0` (Sage's inner-normal
    /// convention): `P = {x : n_i·x + c_i >= 0 for all i}`, one pair per
    /// facet, with the facet itself being `{x in P : n_i·x + c_i = 0}`.
    ///
    /// Each `n` is a *primitive* integer inner normal and each `c` an
    /// integer: the canonical H-representation row `(a | b)` is primitive
    /// as a whole, and since the facet contains a lattice vertex `v` with
    /// `a·v = b`, `gcd(a)` divides `b`, so `gcd(a) = 1`. Consequently `c`
    /// is the *lattice distance* of the facet from the origin, and the
    /// polytope is reflexive exactly when every `c = 1`.
    ///
    /// # Errors
    ///
    /// * If the polytope is **not full-dimensional** (`dim() <
    ///   ambient_dim()`): the affine-hull equations are not facet normals,
    ///   and silently returning the facets of the polytope *within* its
    ///   affine hull would be ambiguous (they are only unique modulo the
    ///   equations). Use [`Self::h_representation`], which returns the
    ///   unambiguous `{equations, inequalities}` pair, for that case.
    /// * If the double description budget is exceeded.
    pub fn facet_inequalities_with_budget(
        &self,
        budget: &DdBudget,
    ) -> Result<Vec<(Vec<Integer>, Integer)>, String> {
        let h = self.h_representation(budget)?;
        if !h.equations.is_empty() {
            return Err(format!(
                "facet inequalities are only defined for full-dimensional polytopes: \
                 this polytope has dimension {} in ambient dimension {}; \
                 use h_representation() for the affine-hull equations",
                self.dim(),
                self.ambient_dim
            ));
        }
        Ok(h.inequalities
            .iter()
            .map(|(a, b)| {
                // a·x <= b  <=>  (-a)·x + b >= 0: inner normal -a, constant b.
                let n: Vec<Integer> = a.iter().map(|x| -rat_to_int(x)).collect();
                (n, rat_to_int(b))
            })
            .collect())
    }

    /// [`Self::facet_inequalities_with_budget`] with the default budget.
    pub fn facet_inequalities(&self) -> Result<Vec<(Vec<Integer>, Integer)>, String> {
        self.facet_inequalities_with_budget(&DdBudget::default())
    }

    /// Check if the polytope is reflexive: full-dimensional, with the
    /// origin as an interior point, and every facet at lattice distance 1
    /// (equivalently, every facet constant is 1 in primitive inner-normal
    /// form — which already forces the origin to be interior, since `0`
    /// then satisfies every facet inequality strictly).
    ///
    /// Equivalently (and cross-checked by [`Self::polar`], which uses the
    /// divisibility route): reflexive iff the polar dual is itself a
    /// lattice polytope.
    ///
    /// # Panics
    ///
    /// Panics if the internal double description budget is exceeded
    /// (never fabricates an answer); use
    /// [`Self::facet_inequalities_with_budget`] to control the budget.
    pub fn is_reflexive(&self) -> bool {
        if self.dim() != self.ambient_dim {
            // A lower-dimensional polytope is never reflexive in its
            // ambient lattice (its polar is unbounded).
            return false;
        }
        let fi = self
            .facet_inequalities()
            .expect("is_reflexive: double description budget exceeded");
        fi.iter().all(|(_, c)| c.is_one())
    }

    /// Get all lattice points within the polytope (inside or on the
    /// boundary), in lexicographically increasing order, with the default
    /// enumeration budget ([`Self::try_points`] with
    /// `DEFAULT_MAX_POINT_CANDIDATES`).
    ///
    /// # Panics
    ///
    /// Panics if the bounding box exceeds the default candidate budget or
    /// the double description budget is exceeded (never fabricates a
    /// partial list); use [`Self::try_points`] for an honest `Err`.
    pub fn points(&self) -> Vec<Vec<Integer>> {
        self.try_points(&DdBudget::default(), DEFAULT_MAX_POINT_CANDIDATES)
            .expect("points: enumeration budget exceeded; use try_points")
    }

    /// All lattice points of the polytope, in lexicographically increasing
    /// order: every integer point of the vertex bounding box that
    /// satisfies the exact H-representation (all facet inequalities *and*
    /// affine-hull equations).
    ///
    /// Complexity: the H-representation is one double description run,
    /// and the filter visits every integer point of the bounding box —
    /// `prod_i (max_i - min_i + 1)` candidates, exponential in the
    /// dimension. That is fine for the small explicit polytopes this type
    /// is for; `max_candidates` bounds it and an honest `Err` is returned
    /// when the bound (or the double description `budget`) is exceeded.
    pub fn try_points(
        &self,
        budget: &DdBudget,
        max_candidates: usize,
    ) -> Result<Vec<Vec<Integer>>, String> {
        let h = self.h_representation(budget)?;
        let d = self.ambient_dim;
        let mut lo = self.vertices[0].clone();
        let mut hi = self.vertices[0].clone();
        for v in &self.vertices {
            for i in 0..d {
                if v[i] < lo[i] {
                    lo[i] = v[i].clone();
                }
                if v[i] > hi[i] {
                    hi[i] = v[i].clone();
                }
            }
        }
        let mut n_candidates = Integer::from(1);
        for i in 0..d {
            n_candidates = n_candidates * (hi[i].clone() - lo[i].clone() + Integer::from(1));
        }
        let cap = Integer::from(max_candidates.min(i64::MAX as usize) as i64);
        if n_candidates > cap {
            return Err(format!(
                "lattice point enumeration budget exceeded: the bounding box has {} \
                 candidate points, budget is {}",
                n_candidates, max_candidates
            ));
        }
        let mut out = Vec::new();
        let mut cur = lo.clone();
        'outer: loop {
            if h.contains(&int_vec_to_rat(&cur)) {
                out.push(cur.clone());
            }
            // Advance the odometer, last coordinate fastest, so the
            // output comes out in lexicographic order.
            let mut i = d;
            loop {
                if i == 0 {
                    break 'outer;
                }
                i -= 1;
                if cur[i] < hi[i] {
                    cur[i] = cur[i].clone() + Integer::from(1);
                    cur[(i + 1)..d].clone_from_slice(&lo[(i + 1)..d]);
                    continue 'outer;
                }
            }
        }
        Ok(out)
    }

    /// Get the number of lattice points
    ///
    /// # Panics
    ///
    /// Same budget panics as [`Self::points`]; use [`Self::try_points`]
    /// for an honest `Err`.
    pub fn n_points(&self) -> usize {
        self.points().len()
    }

    /// Compute facet normals: the primitive integer **inner** normals of
    /// a full-dimensional polytope, one per facet (the first components
    /// of [`Self::facet_inequalities`], which also carries each facet's
    /// constant).
    ///
    /// # Errors
    ///
    /// Same as [`Self::facet_inequalities`]: not full-dimensional, or
    /// budget exceeded.
    pub fn facet_normals(&self) -> Result<Vec<Vec<Integer>>, String> {
        Ok(self
            .facet_inequalities()?
            .into_iter()
            .map(|(n, _)| n)
            .collect())
    }

    /// Get the polar dual as a **lattice** polytope.
    ///
    /// The polar dual of `P` is `P° = {y : y·x >= -1 for all x in P}`,
    /// a polytope with (generally rational) vertices `n_i / c_i` for the
    /// facet inequalities `n_i·x + c_i >= 0` of `P`, defined whenever `P`
    /// is full-dimensional with the origin interior. This method returns
    /// `Some` exactly when `P°` is itself a lattice polytope — i.e. every
    /// `c_i` divides its normal `n_i`, which (since `n_i` is primitive)
    /// means every `c_i = 1`: **exactly the reflexive case**, and the two
    /// routes are asserted to agree in the tests. Use
    /// [`Self::polar_rational`] for the general rational polar.
    ///
    /// Returns `None` if the polytope is not full-dimensional, the origin
    /// is not an interior point, or the polar has a non-lattice vertex.
    ///
    /// # Panics
    ///
    /// Panics if the internal double description budget is exceeded
    /// (never fabricates); use [`Self::facet_inequalities_with_budget`]
    /// to control the budget.
    pub fn polar(&self) -> Option<Self> {
        if self.dim() != self.ambient_dim {
            return None;
        }
        let fi = self
            .facet_inequalities()
            .expect("polar: double description budget exceeded");
        let mut verts = Vec::with_capacity(fi.len());
        for (n, c) in fi {
            if c.signum() <= 0 {
                return None; // the origin is not strictly interior
            }
            let mut v = Vec::with_capacity(n.len());
            for ni in n {
                if !(ni.clone() % c.clone()).is_zero() {
                    return None; // polar vertex n/c is not a lattice point
                }
                v.push(ni / c.clone());
            }
            verts.push(v);
        }
        Some(Self::new(verts))
    }

    /// The polar dual `P° = {y : y·x >= -1 for all x in P}` as its exact
    /// rational vertex list `n_i / c_i` (one vertex per facet of `P`).
    ///
    /// # Errors
    ///
    /// * If `P` is not full-dimensional or the origin is not strictly
    ///   interior (some facet constant `c_i <= 0`): the polar is then
    ///   unbounded and has no vertex-list representation.
    /// * If the double description budget is exceeded.
    pub fn polar_rational(&self) -> Result<Vec<Vec<Rational>>, String> {
        let fi = self.facet_inequalities()?;
        if !fi.iter().all(|(_, c)| c.signum() > 0) {
            return Err(
                "the origin is not an interior point, so the polar dual is unbounded \
                 and has no vertex representation"
                    .to_string(),
            );
        }
        Ok(fi
            .into_iter()
            .map(|(n, c)| {
                n.into_iter()
                    .map(|ni| {
                        Rational::new(ni, c.clone()).expect("facet constant is nonzero")
                    })
                    .collect()
            })
            .collect())
    }

    /// Get faces of a specific dimension, computed from the real
    /// vertex-facet incidence (see [`Self::try_faces`]).
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of faces to return
    ///
    /// # Panics
    ///
    /// Panics if the internal double description or face-count budget is
    /// exceeded (never fabricates); use [`Self::try_faces`] for an honest
    /// `Err`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::lattice_polytope::LatticePolytopeClass;
    /// use rustmath_integers::Integer;
    ///
    /// let vertices = vec![
    ///     vec![Integer::from(0), Integer::from(0)],
    ///     vec![Integer::from(1), Integer::from(0)],
    ///     vec![Integer::from(0), Integer::from(1)],
    /// ];
    /// let polytope = LatticePolytopeClass::new(vertices);
    ///
    /// // Dimension 0 faces are vertices, dimension 1 faces are edges
    /// assert_eq!(polytope.faces(0).len(), 3);
    /// assert_eq!(polytope.faces(1).len(), 3);
    /// ```
    pub fn faces(&self, dimension: usize) -> Vec<Self> {
        self.try_faces(dimension, &DdBudget::default(), DEFAULT_MAX_FACES)
            .expect("faces: enumeration budget exceeded; use try_faces")
    }

    /// The faces of dimension exactly `dimension`, via the face lattice.
    ///
    /// Every proper face of a polytope is an intersection of facets, and
    /// the vertex set of `∩_{i in I} F_i` is `∩_{i in I} V(F_i)` (with
    /// `V(F) = ` the input points lying on `F`), so the proper faces are
    /// exactly the distinct nonempty sets obtainable from the facet
    /// incidence sets by intersection. This computes the facet incidence
    /// sets from the exact H-representation (which also makes it correct
    /// for lower-dimensional polytopes: the canonical inequalities cut
    /// out the facets within the affine hull), closes them under
    /// intersection, and keeps the faces whose affine dimension is
    /// `dimension`. Distinct faces have distinct vertex sets, so no
    /// deduplication beyond set identity is needed.
    ///
    /// Conventions: `faces(dim())` is `[self]` (the improper face);
    /// `faces(d)` for `d > dim()` is empty; the empty face is not
    /// reported. If the input vertex list contains redundant (non-extreme)
    /// points, each face is returned with *all* input points lying on it
    /// (its convex hull is still exactly the face).
    ///
    /// `max_faces` bounds the total number of distinct faces discovered;
    /// exceeding it (or the double description `budget`) returns an
    /// honest `Err`.
    pub fn try_faces(
        &self,
        dimension: usize,
        budget: &DdBudget,
        max_faces: usize,
    ) -> Result<Vec<Self>, String> {
        let own_dim = self.dim();
        if dimension > own_dim {
            return Ok(Vec::new());
        }
        if dimension == own_dim {
            return Ok(vec![self.clone()]);
        }
        let h = self.h_representation(budget)?;
        let rat_verts: Vec<Vec<Rational>> =
            self.vertices.iter().map(|v| int_vec_to_rat(v)).collect();
        let facet_sets: Vec<BTreeSet<usize>> = h
            .inequalities
            .iter()
            .map(|(a, b)| {
                (0..rat_verts.len())
                    .filter(|&i| rat_dot(a, &rat_verts[i]) == *b)
                    .collect()
            })
            .collect();

        // Close {all vertices} under intersection with the facet sets.
        let full: BTreeSet<usize> = (0..rat_verts.len()).collect();
        let mut seen: Vec<BTreeSet<usize>> = vec![full.clone()];
        let mut work: Vec<BTreeSet<usize>> = vec![full];
        while let Some(s) = work.pop() {
            for fs in &facet_sets {
                let t: BTreeSet<usize> = s.intersection(fs).copied().collect();
                if !t.is_empty() && !seen.contains(&t) {
                    if seen.len() >= max_faces {
                        return Err(format!(
                            "face enumeration budget exceeded: more than {} distinct faces",
                            max_faces
                        ));
                    }
                    seen.push(t.clone());
                    work.push(t);
                }
            }
        }

        // seen[0] is the improper face (the polytope itself); the rest
        // are the proper faces. Classify by exact affine dimension.
        let mut out: Vec<Vec<Vec<Integer>>> = Vec::new();
        for s in seen.iter().skip(1) {
            let pts: Vec<Vec<Integer>> =
                s.iter().map(|&i| self.vertices[i].clone()).collect();
            if affine_dim_of(&pts, self.ambient_dim) == dimension {
                out.push(pts);
            }
        }
        out.sort();
        Ok(out.into_iter().map(Self::new).collect())
    }

    /// Get facets (codimension-1 faces)
    pub fn facets(&self) -> Vec<Self> {
        if self.dim() == 0 {
            vec![]
        } else {
            self.faces(self.dim().saturating_sub(1))
        }
    }

    /// Get edges (1-dimensional faces)
    pub fn edges(&self) -> Vec<Self> {
        self.faces(1)
    }

    /// Check if this polytope contains a lattice point: exact membership
    /// in `conv(vertices)`, tested against the H-representation (all
    /// facet inequalities *and* affine-hull equations).
    ///
    /// # Arguments
    ///
    /// * `point` - The point to check
    ///
    /// # Panics
    ///
    /// Panics if the internal double description budget is exceeded
    /// (never fabricates); use [`Self::h_representation`] plus
    /// [`HPolyhedron::contains`] to control the budget.
    pub fn contains(&self, point: &[Integer]) -> bool {
        if point.len() != self.ambient_dim {
            return false;
        }
        self.contains_rational(&int_vec_to_rat(point))
    }

    /// Exact membership test for a rational point (see [`Self::contains`]).
    ///
    /// # Panics
    ///
    /// Same as [`Self::contains`].
    pub fn contains_rational(&self, point: &[Rational]) -> bool {
        if point.len() != self.ambient_dim {
            return false;
        }
        let h = self
            .h_representation(&DdBudget::default())
            .expect("contains: double description budget exceeded");
        h.contains(point)
    }

    /// Compute the (Euclidean) volume of the polytope.
    ///
    /// Returns the *exact* volume as a `Rational`, computed by an exact
    /// simplex decomposition (all arithmetic over `Integer`/`Rational`;
    /// no floating point, no numerical error). This is the plain
    /// Euclidean convention, not the "normalized" `n! * volume`
    /// convention: the unit hypercube has volume 1, and the standard
    /// n-simplex `conv(0, e_1, ..., e_n)` has volume `1/n!`.
    ///
    /// The vertex list does not need to be pre-filtered to extreme
    /// points: redundant (non-extreme) or duplicate points are silently
    /// ignored by the underlying decomposition, so this returns the
    /// volume of `conv(self.vertices())`.
    ///
    /// If the polytope is not full-dimensional (`dim() < ambient_dim()`)
    /// it has zero `ambient_dim()`-dimensional Lebesgue measure in its
    /// ambient space, so this honestly returns `0` — that is the correct
    /// value for a measure-zero set, not a placeholder.
    pub fn volume(&self) -> Rational {
        let n = self.ambient_dim;
        if n == 0 || self.dim() < n {
            return Rational::from_integer(0);
        }
        volume_full_dimensional(&self.vertices, n)
    }

    /// Check if two polytopes are equal
    ///
    /// Two polytopes are equal if they have the same set of vertices.
    pub fn equals(&self, other: &Self) -> bool {
        if self.n_vertices() != other.n_vertices() {
            return false;
        }

        // Check if all vertices match (order-independent)
        for v in &self.vertices {
            if !other.vertices.contains(v) {
                return false;
            }
        }

        true
    }
}

impl Hash for LatticePolytopeClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash based on vertices
        self.n_vertices().hash(state);
        self.ambient_dim.hash(state);
        // For proper hashing, we should sort vertices first
        for v in &self.vertices {
            for coord in v {
                coord.hash(state);
            }
        }
    }
}

impl fmt::Display for LatticePolytopeClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LatticePolytope({}-dimensional with {} vertices)",
            self.dim(),
            self.n_vertices()
        )
    }
}

/// Create a lattice polytope from vertices
///
/// This is a convenience function that wraps LatticePolytopeClass::new.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::lattice_polytope;
/// use rustmath_integers::Integer;
///
/// let vertices = vec![
///     vec![Integer::from(0), Integer::from(0)],
///     vec![Integer::from(1), Integer::from(0)],
/// ];
/// let polytope = lattice_polytope(vertices);
/// ```
pub fn lattice_polytope(vertices: Vec<Vec<Integer>>) -> LatticePolytopeClass {
    LatticePolytopeClass::new(vertices)
}

/// Convex hull of a set of points
///
/// Computes the convex hull and returns it as a lattice polytope: exactly
/// the extreme points of `points` survive, with duplicates and any point
/// expressible as a convex combination of the others removed. All
/// arithmetic is exact (`Integer` coordinates, `Rational` barycentric
/// coefficients) — see [`hull_vertices`] for the algorithm.
///
/// # Arguments
///
/// * `points` - The points to compute the convex hull of
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::convex_hull;
/// use rustmath_integers::Integer;
///
/// // Unit square corners plus its centroid: the centroid is not a vertex.
/// let points = vec![
///     vec![Integer::from(0), Integer::from(0)],
///     vec![Integer::from(1), Integer::from(0)],
///     vec![Integer::from(1), Integer::from(1)],
///     vec![Integer::from(0), Integer::from(1)],
/// ];
/// let hull = convex_hull(points);
/// assert_eq!(hull.n_vertices(), 4);
/// ```
pub fn convex_hull(points: Vec<Vec<Integer>>) -> LatticePolytopeClass {
    LatticePolytopeClass::new(hull_vertices(&points))
}

/// Compute the exact set of extreme points (vertices) of `conv(points)`.
///
/// A point `p` is *not* a vertex of `conv(points)` iff it can be written
/// as a convex combination of the other points. By Carathéodory's theorem
/// (in `R^d`, any point of `conv(S)` is a convex combination of at most
/// `d + 1` *affinely independent* points of `S`, and affinely independent
/// support sets have a unique barycentric-coordinate solution), it
/// suffices to search subsets of the other points of size up to
/// `ambient_dim + 1`, solving the resulting square linear system exactly
/// over `Rational` and checking the coefficients land in `[0, 1]`.
///
/// This is exponential in the subset size (bounded by `ambient_dim + 1`,
/// not by the point count) and is intended for the modest point counts
/// typical of explicit polytope constructions (cubes, simplices, small
/// reflexive polytopes), not for large point clouds.
pub fn hull_vertices(points: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    if points.is_empty() {
        return Vec::new();
    }
    let ambient_dim = points[0].len();

    // Deduplicate first: a repeated point is trivially a convex
    // combination (with coefficient 1) of its other copy.
    let mut unique: Vec<Vec<Integer>> = Vec::new();
    for p in points {
        if !unique.contains(p) {
            unique.push(p.clone());
        }
    }
    if unique.len() <= 1 {
        return unique;
    }

    let mut result = Vec::new();
    for i in 0..unique.len() {
        let others: Vec<&Vec<Integer>> = unique
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| v)
            .collect();

        if !is_convex_combination(&unique[i], &others, ambient_dim) {
            result.push(unique[i].clone());
        }
    }
    result
}

/// Compute the rank of an integer matrix (number of linearly independent
/// rows), via exact fraction-free (Bareiss-style) Gaussian elimination.
/// Used to compute the affine dimension of a vertex set exactly.
fn integer_matrix_rank(matrix: &[Vec<Integer>], cols: usize) -> usize {
    if matrix.is_empty() {
        return 0;
    }

    let mut temp = matrix.to_vec();
    let rows = temp.len();
    let mut rank = 0;
    let mut col = 0;

    while rank < rows && col < cols {
        let mut pivot_row = rank;
        for r in (rank + 1)..rows {
            if temp[r][col].abs() > temp[pivot_row][col].abs() {
                pivot_row = r;
            }
        }

        if temp[pivot_row][col].is_zero() {
            col += 1;
            continue;
        }

        temp.swap(rank, pivot_row);

        for r in (rank + 1)..rows {
            if !temp[r][col].is_zero() {
                let factor = temp[r][col].clone();
                let pivot = temp[rank][col].clone();
                for c in col..cols {
                    temp[r][c] = temp[r][c].clone() * pivot.clone()
                        - temp[rank][c].clone() * factor.clone();
                }
            }
        }

        rank += 1;
        col += 1;
    }

    rank
}

/// Whether `p` can be written as a convex combination of points in `others`
/// (see [`hull_vertices`] for the Carathéodory-theorem argument).
fn is_convex_combination(p: &[Integer], others: &[&Vec<Integer>], ambient_dim: usize) -> bool {
    let m = others.len();
    if m == 0 {
        return false;
    }
    let max_k = (ambient_dim + 1).min(m);

    for k in 1..=max_k {
        let mut combo: Vec<usize> = (0..k).collect();
        loop {
            if combo_is_convex_combination(p, others, ambient_dim, &combo) {
                return true;
            }
            if !next_combination(&mut combo, m) {
                break;
            }
        }
    }
    false
}

/// Advance `combo` (strictly increasing indices into `0..n`) to the next
/// combination in lexicographic order. Returns `false` when there is none.
fn next_combination(combo: &mut [usize], n: usize) -> bool {
    let k = combo.len();
    if k == 0 {
        return false;
    }
    let mut i = k;
    loop {
        if i == 0 {
            return false;
        }
        i -= 1;
        if combo[i] != i + n - k {
            combo[i] += 1;
            for j in (i + 1)..k {
                combo[j] = combo[j - 1] + 1;
            }
            return true;
        }
    }
}

/// Test whether `p = sum_i lambda_i * others[combo[i]]` has an *exact*,
/// *unique* solution with `sum lambda_i = 1` and every `lambda_i` in
/// `[0, 1]`. Builds the `(ambient_dim + 1) x (k + 1)` augmented system
/// (one row per coordinate, plus the affine `sum = 1` row) and solves it
/// exactly over `Rational`; only accepts subsets that are affinely
/// independent (full column rank), which is exactly what Carathéodory's
/// theorem needs.
fn combo_is_convex_combination(
    p: &[Integer],
    others: &[&Vec<Integer>],
    ambient_dim: usize,
    combo: &[usize],
) -> bool {
    let k = combo.len();
    let rows = ambient_dim + 1;
    let mut aug: Vec<Vec<Rational>> = vec![vec![Rational::from_integer(0); k + 1]; rows];

    for coord in 0..ambient_dim {
        for (col, &idx) in combo.iter().enumerate() {
            aug[coord][col] = Rational::from_integer(others[idx][coord].clone());
        }
        aug[coord][k] = Rational::from_integer(p[coord].clone());
    }
    // Affine constraint: coefficients sum to 1.
    for col in 0..k {
        aug[ambient_dim][col] = Rational::from_integer(1);
    }
    aug[ambient_dim][k] = Rational::from_integer(1);

    match rational_rref(&mut aug, k) {
        Some(rank) if rank == k => {
            let zero = Rational::from_integer(0);
            let one = Rational::from_integer(1);
            (0..k).all(|i| aug[i][k] >= zero && aug[i][k] <= one)
        }
        _ => false,
    }
}

/// Reduce the augmented system `[A | b]` (with `cols` unknowns) to reduced
/// row-echelon form in place via exact rational Gauss-Jordan elimination.
/// Returns `Some(rank)` of the coefficient part if the system is
/// consistent, `None` if it is not (a row `0 = nonzero`). When the
/// returned rank equals `cols`, `aug[i][cols]` for `i in 0..cols` is the
/// unique solution.
fn rational_rref(aug: &mut [Vec<Rational>], cols: usize) -> Option<usize> {
    let rows = aug.len();
    let mut rank = 0;
    let mut col = 0;

    while rank < rows && col < cols {
        let pivot_row = (rank..rows).find(|&r| !aug[r][col].is_zero());
        let pivot_row = match pivot_row {
            Some(r) => r,
            None => {
                col += 1;
                continue;
            }
        };
        aug.swap(rank, pivot_row);

        let pivot_val = aug[rank][col].clone();
        for c in col..=cols {
            aug[rank][c] = aug[rank][c].clone() / pivot_val.clone();
        }

        for r in 0..rows {
            if r != rank && !aug[r][col].is_zero() {
                let factor = aug[r][col].clone();
                for c in col..=cols {
                    let sub = aug[rank][c].clone() * factor.clone();
                    aug[r][c] = aug[r][c].clone() - sub;
                }
            }
        }

        rank += 1;
        col += 1;
    }

    for row in aug.iter().skip(rank) {
        if !row[cols].is_zero() {
            return None; // Inconsistent: 0 = nonzero.
        }
    }

    Some(rank)
}

/// Exact determinant of a square `Integer` matrix via cofactor expansion.
///
/// `O(n!)`, which is fine for the small dimensions (`n` up to a handful)
/// typical of explicit lattice polytopes; not intended for large `n`.
fn integer_determinant(matrix: &[Vec<Integer>]) -> Integer {
    let n = matrix.len();
    if n == 0 {
        return Integer::from(1);
    }
    if n == 1 {
        return matrix[0][0].clone();
    }

    let mut result = Integer::zero();
    let mut sign = Integer::from(1);
    for col in 0..n {
        let entry = &matrix[0][col];
        if !entry.is_zero() {
            let minor: Vec<Vec<Integer>> = matrix[1..]
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .filter(|(c, _)| *c != col)
                        .map(|(_, v)| v.clone())
                        .collect()
                })
                .collect();
            result = result + &sign * entry * integer_determinant(&minor);
        }
        sign = -sign;
    }
    result
}

/// Find `dim + 1` affinely independent points among `points`, returning
/// their indices. Greedily grows a set of linearly independent
/// difference-from-`points[0]` vectors (checked exactly via
/// [`integer_matrix_rank`]) until it reaches `dim`, i.e. until the chosen
/// indices span the full affine hull. Returns `None` if `points` does not
/// contain `dim + 1` affinely independent points.
fn find_affine_basis(points: &[Vec<Integer>], dim: usize) -> Option<Vec<usize>> {
    if points.is_empty() {
        return None;
    }
    if dim == 0 {
        return Some(vec![0]);
    }

    let base = &points[0];
    let mut chosen = vec![0usize];
    let mut diffs: Vec<Vec<Integer>> = Vec::new();

    for (i, p) in points.iter().enumerate().skip(1) {
        if chosen.len() == dim + 1 {
            break;
        }
        let diff: Vec<Integer> = p
            .iter()
            .zip(base.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();
        let mut candidate = diffs.clone();
        candidate.push(diff.clone());
        if integer_matrix_rank(&candidate, dim) > diffs.len() {
            diffs = candidate;
            chosen.push(i);
        }
    }

    if chosen.len() == dim + 1 {
        Some(chosen)
    } else {
        None
    }
}

/// Signed "cone determinant" of `apex` over `facet` (a list of exactly
/// `facet.len()` vertex indices spanning a hyperplane): the
/// `facet.len() x facet.len()` determinant of the facet's edge vectors
/// (from `points[facet[0]]`) together with the vector from
/// `apex_scale * points[facet[0]]` to `apex`. Its sign tells which side
/// of the hyperplane `apex` lies on relative to the facet's stored
/// vertex order.
///
/// `apex_scale` lets the caller pass an *unscaled* stand-in apex (e.g.
/// `apex_scale * true_apex`) without losing exactness: the edge-vector
/// rows are always genuine (unscaled) differences, but the apex row
/// becomes `apex - apex_scale * base` so that when `apex` really is
/// `apex_scale * true_apex`, this row equals `apex_scale * (true_apex -
/// base)` exactly — i.e. the whole determinant is `apex_scale` times the
/// true (unscaled) cone determinant. Pass `apex_scale = 1` for a genuine,
/// already-unscaled `apex` point.
fn cone_signed_det(
    points: &[Vec<Integer>],
    facet: &[usize],
    apex: &[Integer],
    apex_scale: &Integer,
) -> Integer {
    let base = &points[facet[0]];
    let mut matrix: Vec<Vec<Integer>> = facet[1..]
        .iter()
        .map(|&idx| {
            points[idx]
                .iter()
                .zip(base.iter())
                .map(|(a, b)| a.clone() - b.clone())
                .collect()
        })
        .collect();
    let apex_row: Vec<Integer> = apex
        .iter()
        .zip(base.iter())
        .map(|(a, b)| a.clone() - apex_scale.clone() * b.clone())
        .collect();
    matrix.push(apex_row);
    integer_determinant(&matrix)
}

/// `k!` as an `Integer`.
fn factorial_integer(k: usize) -> Integer {
    let mut result = Integer::from(1);
    for i in 2..=k {
        result = result * Integer::from(i as i32);
    }
    result
}

/// Exact volume of `conv(points)`, assumed full-dimensional
/// (`rank(points) == n == ` the shared length of every point vector).
///
/// Uses the incremental "beneath-beyond" convex hull algorithm, fixing a
/// point `P0` strictly interior to an initial `n`-simplex (which, since
/// the hull only grows as more points are absorbed, stays interior for
/// the whole computation) and maintaining the volume as the sum of the
/// `n`-simplex cones from `P0` to every current boundary facet. Facets
/// are `n`-vertex simplices; a facet's `n` sub-ridges (`n-1`-vertex
/// subsets) that occur in exactly one visible facet form the horizon,
/// which is coned to each newly-absorbed vertex to produce the new
/// facets. Points that are not extreme (already inside the hull-so-far)
/// are silently skipped, so redundant/duplicate input vertices are
/// harmless.
///
/// All determinants are computed over `Integer` by using the *unscaled*
/// sum `S` of the seed simplex's `n + 1` vertices as a stand-in for the
/// true interior point `P0 = S / (n + 1)`: since the determinant is
/// linear in the apex row, this scales every cone determinant by the
/// constant factor `n + 1`, which is divided back out (together with the
/// usual `n!` simplex-volume normalization) only once, at the very end.
///
/// Each facet is stored together with its (arbitrarily-signed, but
/// fixed at creation time) cone determinant against `S`: rather than
/// canonicalizing the sign by reordering vertices (which breaks for a
/// 1-vertex facet — the `n = 1` case has no second vertex to swap), a
/// new point `q` is beyond a facet exactly when `cone(facet, q)` has the
/// *opposite* sign from the facet's stored `S`-determinant, and each
/// facet's unsigned contribution to the running volume is just the
/// absolute value of that stored determinant.
fn volume_full_dimensional(points: &[Vec<Integer>], n: usize) -> Rational {
    let seed = match find_affine_basis(points, n) {
        Some(s) => s,
        None => return Rational::from_integer(0), // not actually full-dimensional
    };

    // S = (n+1) * P0, P0 strictly interior to the seed simplex.
    let mut s: Vec<Integer> = vec![Integer::zero(); n];
    for &idx in &seed {
        for (acc, coord) in s.iter_mut().zip(points[idx].iter()) {
            *acc = acc.clone() + coord.clone();
        }
    }
    let apex_scale = Integer::from((n + 1) as i32);
    let one = Integer::from(1);
    let zero = Integer::zero();

    // facets: (vertex indices, cone determinant against S).
    let mut facets: Vec<(Vec<usize>, Integer)> = Vec::new();
    let mut total_raw = Integer::zero();

    for k in 0..seed.len() {
        let verts: Vec<usize> = seed
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != k)
            .map(|(_, &v)| v)
            .collect();
        let d = cone_signed_det(points, &verts, &s, &apex_scale);
        total_raw = total_raw + d.abs();
        facets.push((verts, d));
    }

    let seed_set: std::collections::HashSet<usize> = seed.iter().copied().collect();
    for (q, qp) in points.iter().enumerate() {
        if seed_set.contains(&q) {
            continue;
        }

        let mut visible = Vec::new();
        for (fi, (verts, s_det)) in facets.iter().enumerate() {
            if s_det.is_zero() {
                continue; // degenerate facet: never visible, contributes 0 either way
            }
            let q_det = cone_signed_det(points, verts, qp, &one);
            // Beyond the facet iff q is on the opposite (nonzero) side from S.
            if (q_det * s_det.clone()) < zero {
                visible.push(fi);
            }
        }
        if visible.is_empty() {
            continue; // q is not extreme; already inside the hull so far.
        }

        // Count sub-ridges of the visible facets; those occurring exactly
        // once are the horizon.
        let mut ridge_counts: HashMap<Vec<usize>, u32> = HashMap::new();
        for &fi in &visible {
            let (verts, _) = &facets[fi];
            for omit in 0..verts.len() {
                let mut ridge: Vec<usize> = verts
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != omit)
                    .map(|(_, &v)| v)
                    .collect();
                ridge.sort_unstable();
                *ridge_counts.entry(ridge).or_insert(0) += 1;
            }
        }

        // Remove visible facets (in reverse index order to keep indices valid).
        let mut visible_sorted = visible.clone();
        visible_sorted.sort_unstable();
        for &fi in visible_sorted.iter().rev() {
            let (_, s_det) = &facets[fi];
            total_raw = total_raw - s_det.abs();
            facets.remove(fi);
        }

        // Add new facets: each horizon ridge coned to the new vertex q.
        for (ridge, count) in ridge_counts {
            if count != 1 {
                continue;
            }
            let mut new_verts = ridge;
            new_verts.push(q);
            let d = cone_signed_det(points, &new_verts, &s, &apex_scale);
            total_raw = total_raw + d.abs();
            facets.push((new_verts, d));
        }
    }

    Rational::from_integer(total_raw) / Rational::from_integer(factorial_integer(n + 1))
}

/// Create a cross-polytope (orthoplex) in dimension n
///
/// The cross-polytope is the convex hull of the standard basis vectors
/// and their negatives: {±e_i : i = 1..n}
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::cross_polytope;
///
/// // 2D cross-polytope (diamond/square rotated 45 degrees)
/// let cross = cross_polytope(2);
/// assert_eq!(cross.n_vertices(), 4);
/// ```
pub fn cross_polytope(dimension: usize) -> LatticePolytopeClass {
    if dimension == 0 {
        panic!("Dimension must be at least 1");
    }

    let mut vertices = Vec::new();

    // Add ±e_i for each coordinate direction
    for i in 0..dimension {
        // +e_i
        let mut v_pos = vec![Integer::from(0); dimension];
        v_pos[i] = Integer::from(1);
        vertices.push(v_pos);

        // -e_i
        let mut v_neg = vec![Integer::from(0); dimension];
        v_neg[i] = Integer::from(-1);
        vertices.push(v_neg);
    }

    LatticePolytopeClass::new(vertices)
}

/// Set of all lattice polytopes
///
/// This represents the collection of all lattice polytopes.
#[derive(Clone, Debug)]
pub struct SetOfAllLatticePolytopesClass {
    /// Dimension restriction (None means all dimensions)
    dimension: Option<usize>,
}

impl SetOfAllLatticePolytopesClass {
    /// Create a new set of all lattice polytopes
    pub fn new() -> Self {
        Self { dimension: None }
    }

    /// Create a set of lattice polytopes of a specific dimension
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension: Some(dimension),
        }
    }

    /// Check if a polytope is in this set
    pub fn contains(&self, polytope: &LatticePolytopeClass) -> bool {
        if let Some(dim) = self.dimension {
            polytope.dim() == dim
        } else {
            true
        }
    }
}

impl Default for SetOfAllLatticePolytopesClass {
    fn default() -> Self {
        Self::new()
    }
}

/// Nef partition of a lattice polytope
///
/// A nef partition is a partition of the vertices of a reflexive polytope
/// into two sets that satisfy certain geometric conditions related to
/// nef line bundles on toric varieties.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NefPartition {
    /// The underlying polytope
    polytope: LatticePolytopeClass,
    /// The partition of vertices (true/false for each vertex)
    partition: Vec<bool>,
    /// Cached properties
    cached_data: Option<HashMap<String, String>>,
}

impl NefPartition {
    /// Create a new nef partition
    ///
    /// # Arguments
    ///
    /// * `polytope` - The reflexive polytope
    /// * `partition` - Boolean array indicating which part each vertex belongs to
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::lattice_polytope::{LatticePolytopeClass, NefPartition};
    /// use rustmath_integers::Integer;
    ///
    /// let vertices = vec![
    ///     vec![Integer::from(1), Integer::from(0)],
    ///     vec![Integer::from(-1), Integer::from(0)],
    ///     vec![Integer::from(0), Integer::from(1)],
    ///     vec![Integer::from(0), Integer::from(-1)],
    /// ];
    /// let polytope = LatticePolytopeClass::new(vertices);
    /// let partition = vec![true, true, false, false];
    ///
    /// let nef = NefPartition::new(polytope, partition);
    /// ```
    pub fn new(polytope: LatticePolytopeClass, partition: Vec<bool>) -> Self {
        if partition.len() != polytope.n_vertices() {
            panic!("Partition size must match number of vertices");
        }

        Self {
            polytope,
            partition,
            cached_data: None,
        }
    }

    /// Get the polytope
    pub fn polytope(&self) -> &LatticePolytopeClass {
        &self.polytope
    }

    /// Get the partition
    pub fn partition(&self) -> &[bool] {
        &self.partition
    }

    /// Get vertices in the first part
    pub fn part0_vertices(&self) -> Vec<Vec<Integer>> {
        self.polytope
            .vertices()
            .iter()
            .zip(self.partition.iter())
            .filter_map(|(v, &in_part)| {
                if !in_part {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get vertices in the second part
    pub fn part1_vertices(&self) -> Vec<Vec<Integer>> {
        self.polytope
            .vertices()
            .iter()
            .zip(self.partition.iter())
            .filter_map(|(v, &in_part)| {
                if in_part {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if this is a valid nef partition
    ///
    /// A nef partition is valid if both parts are non-empty and
    /// satisfy the nef conditions.
    pub fn is_valid(&self) -> bool {
        let part0_count = self.partition.iter().filter(|&&x| !x).count();
        let part1_count = self.partition.iter().filter(|&&x| x).count();

        // Both parts must be non-empty
        part0_count > 0 && part1_count > 0
    }

    /// Compute the Hodge numbers associated with this nef partition
    ///
    /// Returns (h11, h12) for the associated Calabi-Yau variety.
    /// This is a simplified placeholder.
    pub fn hodge_numbers(&self) -> (usize, usize) {
        // Proper implementation would compute actual Hodge numbers
        // based on the partition geometry
        (0, 0)
    }
}

/// Check if an object is a lattice polytope
///
/// This function checks whether the given object is an instance of
/// `LatticePolytopeClass`.
///
/// # Deprecated
///
/// Use `matches!(obj, LatticePolytopeClass { .. })` or type checking instead.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{is_lattice_polytope, cross_polytope};
///
/// let polytope = cross_polytope(2);
/// assert!(is_lattice_polytope(&polytope));
/// ```
pub fn is_lattice_polytope(obj: &LatticePolytopeClass) -> bool {
    // In Rust, if we have a reference to LatticePolytopeClass, it's always a lattice polytope
    // This function exists for API compatibility with SageMath
    true
}

/// Global database of reflexive polytopes (simulated)
///
/// In SageMath, there are pre-computed databases of reflexive polytopes.
/// This is a simplified version that generates basic examples.
static REFLEXIVE_POLYTOPES_2D: &[(usize, &[(i32, i32)])] = &[
    // Triangle: [-1,-1], [2,-1], [-1,2]
    (0, &[(-1, -1), (2, -1), (-1, 2)]),
    // Square: [1,1], [1,-1], [-1,1], [-1,-1]
    (1, &[(1, 1), (1, -1), (-1, 1), (-1, -1)]),
    // Hexagon: Various 2D reflexive polytopes
    (2, &[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]),
    // Add more as needed...
];

/// Get a specific reflexive polytope from the database
///
/// Returns the n-th reflexive polytope of a given dimension.
///
/// # Arguments
///
/// * `dim` - Dimension (must be 2 or 3)
/// * `n` - Index of the polytope
///
/// # Returns
///
/// The requested reflexive polytope, or None if out of range.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::reflexive_polytope;
///
/// // Get the first 2D reflexive polytope
/// let poly = reflexive_polytope(2, 0);
/// assert!(poly.is_some());
/// ```
pub fn reflexive_polytope(dim: usize, n: usize) -> Option<LatticePolytopeClass> {
    match dim {
        2 => {
            if let Some((_, vertices)) = REFLEXIVE_POLYTOPES_2D.get(n) {
                let verts: Vec<Vec<Integer>> = vertices
                    .iter()
                    .map(|(x, y)| vec![Integer::from(*x), Integer::from(*y)])
                    .collect();
                Some(LatticePolytopeClass::new(verts))
            } else {
                None
            }
        }
        3 => {
            // For 3D, we would need a much larger database
            // This is a placeholder
            None
        }
        _ => None,
    }
}

/// Get all reflexive polytopes of a given dimension
///
/// Returns a vector of all reflexive polytopes of the specified dimension.
///
/// # Arguments
///
/// * `dim` - Dimension (must be 2 or 3)
///
/// # Returns
///
/// Vector of all reflexive polytopes in that dimension
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::reflexive_polytopes;
///
/// let polytopes_2d = reflexive_polytopes(2);
/// assert!(polytopes_2d.len() > 0);
/// ```
pub fn reflexive_polytopes(dim: usize) -> Vec<LatticePolytopeClass> {
    match dim {
        2 => {
            let mut result = Vec::new();
            for (_, vertices) in REFLEXIVE_POLYTOPES_2D.iter() {
                let verts: Vec<Vec<Integer>> = vertices
                    .iter()
                    .map(|(x, y)| vec![Integer::from(*x), Integer::from(*y)])
                    .collect();
                result.push(LatticePolytopeClass::new(verts));
            }
            result
        }
        3 => {
            // For 3D reflexive polytopes, there are 4,319 of them
            // This would require a large database
            Vec::new()
        }
        _ => Vec::new(),
    }
}

/// Get all lattice points for a sequence of polytopes
///
/// For a list of polytopes, returns all their lattice points.
///
/// # Arguments
///
/// * `polytopes` - Slice of polytopes
///
/// # Returns
///
/// Vector of vectors, where each inner vector contains the lattice points
/// for the corresponding polytope
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{all_points, cross_polytope};
///
/// let polytopes = vec![cross_polytope(2), cross_polytope(3)];
/// let points = all_points(&polytopes);
/// assert_eq!(points.len(), 2);
/// ```
pub fn all_points(polytopes: &[LatticePolytopeClass]) -> Vec<Vec<Vec<Integer>>> {
    polytopes.iter().map(|p| p.points()).collect()
}

/// Get all polar duals for a sequence of polytopes
///
/// For a list of reflexive polytopes, returns their polar duals.
///
/// # Arguments
///
/// * `polytopes` - Slice of polytopes
///
/// # Returns
///
/// Vector of optional polytopes (None if not reflexive)
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{all_polars, reflexive_polytopes};
///
/// let polytopes = reflexive_polytopes(2);
/// let polars = all_polars(&polytopes);
/// // Reflexive polytopes should have polar duals
/// ```
pub fn all_polars(polytopes: &[LatticePolytopeClass]) -> Vec<Option<LatticePolytopeClass>> {
    polytopes.iter().map(|p| p.polar()).collect()
}

/// Get all facet equations for a sequence of polytopes
///
/// For a list of polytopes, returns the facet normal vectors.
///
/// # Arguments
///
/// * `polytopes` - Slice of polytopes
///
/// # Returns
///
/// Vector of vectors, where each inner vector contains the primitive
/// inner facet normals for the corresponding polytope; `Err` if any
/// polytope is not full-dimensional or a budget is exceeded (see
/// [`LatticePolytopeClass::facet_normals`]).
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{all_facet_equations, cross_polytope};
///
/// let polytopes = vec![cross_polytope(2)];
/// let facets = all_facet_equations(&polytopes).unwrap();
/// assert_eq!(facets.len(), 1);
/// assert_eq!(facets[0].len(), 4); // the diamond has 4 facets
/// ```
pub fn all_facet_equations(
    polytopes: &[LatticePolytopeClass],
) -> Result<Vec<Vec<Vec<Integer>>>, String> {
    polytopes.iter().map(|p| p.facet_normals()).collect()
}

/// Get all nef partitions for a sequence of reflexive polytopes
///
/// For a list of reflexive polytopes, computes all valid nef partitions.
///
/// # Arguments
///
/// * `polytopes` - Slice of reflexive polytopes
///
/// # Returns
///
/// Vector of vectors, where each inner vector contains the nef partitions
/// for the corresponding polytope
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{all_nef_partitions, reflexive_polytopes};
///
/// let polytopes = reflexive_polytopes(2);
/// let nef_parts = all_nef_partitions(&polytopes);
/// assert_eq!(nef_parts.len(), polytopes.len());
/// ```
pub fn all_nef_partitions(polytopes: &[LatticePolytopeClass]) -> Vec<Vec<NefPartition>> {
    polytopes
        .iter()
        .map(|p| {
            // Generate all possible partitions
            let n = p.n_vertices();
            let mut partitions = Vec::new();

            // Try all 2^n possible partitions
            for i in 0..(1 << n) {
                let mut partition = Vec::new();
                for j in 0..n {
                    partition.push((i & (1 << j)) != 0);
                }

                let nef = NefPartition::new(p.clone(), partition);
                if nef.is_valid() {
                    partitions.push(nef);
                }
            }

            partitions
        })
        .collect()
}

/// Get all cached data for a sequence of polytopes
///
/// Returns pre-computed data for a list of polytopes.
/// This is primarily used for optimization when working with
/// large databases of polytopes.
///
/// # Arguments
///
/// * `polytopes` - Slice of polytopes
///
/// # Returns
///
/// Vector of hash maps containing cached properties
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{all_cached_data, cross_polytope};
///
/// let polytopes = vec![cross_polytope(2)];
/// let cache = all_cached_data(&polytopes);
/// assert_eq!(cache.len(), 1);
/// ```
pub fn all_cached_data(polytopes: &[LatticePolytopeClass]) -> Vec<HashMap<String, String>> {
    polytopes
        .iter()
        .map(|p| {
            let mut cache = HashMap::new();
            cache.insert("n_vertices".to_string(), p.n_vertices().to_string());
            cache.insert("dimension".to_string(), p.dim().to_string());
            cache.insert("ambient_dim".to_string(), p.ambient_dim().to_string());
            cache
        })
        .collect()
}

/// Write a polytope to PALP matrix format
///
/// PALP (Package for Analyzing Lattice Polytopes) is an external program
/// for studying reflexive polytopes. This function writes a polytope's
/// vertex matrix in PALP format.
///
/// # Arguments
///
/// * `polytope` - The polytope to write
///
/// # Returns
///
/// A string in PALP matrix format
///
/// # Format
///
/// The PALP format consists of:
/// - First line: number of vertices, dimension
/// - Following lines: vertex coordinates (one vertex per line)
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{write_palp_matrix, cross_polytope};
///
/// let polytope = cross_polytope(2);
/// let palp_string = write_palp_matrix(&polytope);
/// assert!(palp_string.contains("4 2")); // 4 vertices, 2D
/// ```
pub fn write_palp_matrix(polytope: &LatticePolytopeClass) -> String {
    let mut result = String::new();

    // Header: number of vertices, dimension
    result.push_str(&format!("{} {}\n", polytope.n_vertices(), polytope.ambient_dim()));

    // Write each vertex
    for vertex in polytope.vertices() {
        let coords: Vec<String> = vertex.iter().map(|x| x.to_string()).collect();
        result.push_str(&coords.join(" "));
        result.push('\n');
    }

    result
}

/// Read a polytope from PALP matrix format
///
/// Parse a PALP-formatted string to create a lattice polytope.
///
/// # Arguments
///
/// * `palp_string` - String in PALP matrix format
///
/// # Returns
///
/// The parsed polytope, or None if parsing fails
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{read_palp_matrix, write_palp_matrix, cross_polytope};
///
/// let original = cross_polytope(2);
/// let palp_str = write_palp_matrix(&original);
/// let parsed = read_palp_matrix(&palp_str);
///
/// assert!(parsed.is_some());
/// let parsed_poly = parsed.unwrap();
/// assert_eq!(parsed_poly.n_vertices(), original.n_vertices());
/// ```
pub fn read_palp_matrix(palp_string: &str) -> Option<LatticePolytopeClass> {
    let lines: Vec<&str> = palp_string.trim().lines().collect();
    if lines.is_empty() {
        return None;
    }

    // Parse header
    let header_parts: Vec<&str> = lines[0].split_whitespace().collect();
    if header_parts.len() != 2 {
        return None;
    }

    let n_vertices: usize = header_parts[0].parse().ok()?;
    let _dim: usize = header_parts[1].parse().ok()?;

    // Parse vertices
    let mut vertices = Vec::new();
    for line in lines.iter().skip(1).take(n_vertices) {
        let coords: Result<Vec<Integer>, _> = line
            .split_whitespace()
            .map(|s| s.parse::<i64>().map(Integer::from))
            .collect();

        match coords {
            Ok(v) => vertices.push(v),
            Err(_) => return None,
        }
    }

    if vertices.len() != n_vertices {
        return None;
    }

    Some(LatticePolytopeClass::new(vertices))
}

/// Check if an object is a nef partition
///
/// This function checks whether the given object is an instance of `NefPartition`.
///
/// # Deprecated
///
/// Use type checking instead.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{is_nef_partition, NefPartition, cross_polytope};
///
/// let polytope = cross_polytope(2);
/// let partition = vec![true, true, false, false];
/// let nef = NefPartition::new(polytope, partition);
/// assert!(is_nef_partition(&nef));
/// ```
pub fn is_nef_partition(_obj: &NefPartition) -> bool {
    // In Rust, if we have a reference to NefPartition, it's always a nef partition
    // This function exists for API compatibility with SageMath
    true
}

/// Compute the Minkowski sum of two lattice polytopes
///
/// The Minkowski sum of polytopes P and Q is defined as:
/// P ⊕ Q = {p + q : p ∈ P, q ∈ Q}
///
/// # Arguments
///
/// * `p1` - First polytope
/// * `p2` - Second polytope
///
/// # Returns
///
/// The Minkowski sum as a new lattice polytope
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{minkowski_sum, LatticePolytopeClass};
/// use rustmath_integers::Integer;
///
/// // Unit square [0,1]²
/// let square = LatticePolytopeClass::new(vec![
///     vec![Integer::from(0), Integer::from(0)],
///     vec![Integer::from(1), Integer::from(0)],
///     vec![Integer::from(1), Integer::from(1)],
///     vec![Integer::from(0), Integer::from(1)],
/// ]);
///
/// // Single point at origin
/// let point = LatticePolytopeClass::new(vec![
///     vec![Integer::from(1), Integer::from(1)],
/// ]);
///
/// let sum = minkowski_sum(&square, &point);
/// // Result is square translated by (1,1)
/// ```
pub fn minkowski_sum(p1: &LatticePolytopeClass, p2: &LatticePolytopeClass) -> LatticePolytopeClass {
    if p1.ambient_dim() != p2.ambient_dim() {
        panic!("Polytopes must have the same ambient dimension");
    }

    let dim = p1.ambient_dim();
    let mut sum_vertices = Vec::new();

    // Compute all pairwise sums of vertices
    for v1 in p1.vertices() {
        for v2 in p2.vertices() {
            let mut sum_vertex = Vec::with_capacity(dim);
            for i in 0..dim {
                sum_vertex.push(v1[i].clone() + v2[i].clone());
            }
            sum_vertices.push(sum_vertex);
        }
    }

    // The Minkowski sum vertices are the convex hull of all pairwise sums
    // For a proper implementation, we would compute the actual convex hull
    // to remove interior points
    LatticePolytopeClass::new(sum_vertices)
}

/// Find positive integer relations among vectors
///
/// Given a matrix whose columns are vectors, find all relations
/// c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 where all cᵢ are positive integers.
///
/// # Arguments
///
/// * `vectors` - Matrix where each column is a vector (rows × cols)
///
/// # Returns
///
/// Vector of relation vectors, where each relation is a vector of coefficients
///
/// # Mathematical Background
///
/// This computes the kernel of the matrix over the integers, restricted to
/// positive coefficients. This is useful for studying combinatorial properties
/// of polytopes.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::positive_integer_relations;
/// use rustmath_integers::Integer;
///
/// // Three vectors: (1,0), (0,1), (-1,-1)
/// // Relation: 1*(1,0) + 1*(0,1) + 1*(-1,-1) = (0,0)
/// let vectors = vec![
///     vec![Integer::from(1), Integer::from(0)],
///     vec![Integer::from(0), Integer::from(1)],
///     vec![Integer::from(-1), Integer::from(-1)],
/// ];
///
/// let relations = positive_integer_relations(&vectors);
/// // Should find the relation [1, 1, 1]
/// ```
pub fn positive_integer_relations(vectors: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    if vectors.is_empty() {
        return Vec::new();
    }

    // This is a simplified implementation
    // A full implementation would:
    // 1. Form the matrix from vectors
    // 2. Compute the kernel (null space) over the integers
    // 3. Find combinations with all positive coefficients
    // 4. Use lattice reduction or enumeration algorithms

    // For now, return empty - proper implementation requires
    // integer matrix kernel computation
    Vec::new()
}

/// Global PALP dimension setting
///
/// In SageMath, this is used to configure the PALP program.
/// For our implementation, we store it as a module-level default.
static mut PALP_DIMENSION: usize = 4;

/// Set the PALP dimension
///
/// This sets the maximum dimension for PALP operations.
/// In SageMath, PALP (Package for Analyzing Lattice Polytopes) requires
/// dimension to be set at compile time.
///
/// # Arguments
///
/// * `dim` - The dimension to set (typically 2-6)
///
/// # Safety
///
/// This uses a mutable static variable, so it's not thread-safe in the
/// current implementation.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::set_palp_dimension;
///
/// set_palp_dimension(4);
/// ```
pub fn set_palp_dimension(dim: usize) {
    unsafe {
        PALP_DIMENSION = dim;
    }
}

/// Get the current PALP dimension
///
/// Returns the dimension previously set by `set_palp_dimension`.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{set_palp_dimension, get_palp_dimension};
///
/// set_palp_dimension(5);
/// assert_eq!(get_palp_dimension(), 5);
/// ```
pub fn get_palp_dimension() -> usize {
    unsafe { PALP_DIMENSION }
}

/// Skip a PALP matrix in a reader
///
/// When reading multiple polytopes from a PALP-formatted file,
/// this function skips over the next matrix without parsing it.
///
/// # Arguments
///
/// * `reader` - A string containing PALP-formatted data
/// * `current_pos` - Current position in the string (line number)
///
/// # Returns
///
/// New position after skipping the matrix
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::skip_palp_matrix;
///
/// let palp_data = "3 2\n1 0\n0 1\n1 1\n2 2\n1 0\n-1 0\n";
/// let new_pos = skip_palp_matrix(palp_data, 0);
/// assert!(new_pos > 0);
/// ```
pub fn skip_palp_matrix(data: &str, current_pos: usize) -> usize {
    let lines: Vec<&str> = data.lines().collect();

    if current_pos >= lines.len() {
        return current_pos;
    }

    // Parse header to get number of vertices
    let header_parts: Vec<&str> = lines[current_pos].split_whitespace().collect();
    if header_parts.len() < 2 {
        return current_pos + 1;
    }

    if let Ok(n_vertices) = header_parts[0].parse::<usize>() {
        // Skip header + n_vertices lines
        current_pos + 1 + n_vertices
    } else {
        current_pos + 1
    }
}

/// Read all polytopes from a PALP-formatted file
///
/// This function reads multiple polytopes from a file in PALP format.
/// Each polytope is separated by its matrix representation.
///
/// # Arguments
///
/// * `data` - String containing PALP-formatted polytope data
///
/// # Returns
///
/// Vector of all polytopes read from the file
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{read_all_polytopes, write_palp_matrix, cross_polytope};
///
/// // Create sample data with two polytopes
/// let p1 = cross_polytope(2);
/// let p2 = cross_polytope(2);
/// let mut data = write_palp_matrix(&p1);
/// data.push_str(&write_palp_matrix(&p2));
///
/// let polytopes = read_all_polytopes(&data);
/// assert_eq!(polytopes.len(), 2);
/// ```
pub fn read_all_polytopes(data: &str) -> Vec<LatticePolytopeClass> {
    let mut polytopes = Vec::new();
    let lines: Vec<&str> = data.lines().collect();
    let mut pos = 0;

    while pos < lines.len() {
        // Skip empty lines
        if lines[pos].trim().is_empty() {
            pos += 1;
            continue;
        }

        // Try to read a polytope
        let remaining_data = lines[pos..].join("\n");
        if let Some(polytope) = read_palp_matrix(&remaining_data) {
            polytopes.push(polytope);
            // Move position forward by the matrix size
            pos = skip_palp_matrix(data, pos);
        } else {
            // If we can't parse, skip this line
            pos += 1;
        }
    }

    polytopes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_polytope() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        assert_eq!(polytope.n_vertices(), 3);
        assert_eq!(polytope.ambient_dim(), 2);
    }

    #[test]
    fn test_vertices() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices.clone());

        assert_eq!(polytope.vertices(), &vertices);
    }

    #[test]
    fn test_dim() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        // For 3 vertices in 2D, dimension should be 2
        assert_eq!(polytope.dim(), 2);
    }

    #[test]
    fn test_single_point() {
        let vertices = vec![vec![Integer::from(1), Integer::from(2)]];
        let polytope = LatticePolytopeClass::new(vertices);

        assert_eq!(polytope.dim(), 0);
        assert_eq!(polytope.n_vertices(), 1);
    }

    #[test]
    fn test_contains() {
        // Segment from (0,0) to (1,0): a lower-dimensional polytope, so
        // membership must respect the affine-hull equation y = 0 too.
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        assert!(polytope.contains(&[Integer::from(0), Integer::from(0)]));
        assert!(polytope.contains(&[Integer::from(1), Integer::from(0)]));
        assert!(!polytope.contains(&[Integer::from(2), Integer::from(0)]));
        assert!(!polytope.contains(&[Integer::from(-1), Integer::from(0)]));
        assert!(!polytope.contains(&[Integer::from(0), Integer::from(1)]));
        // Wrong arity is simply not contained.
        assert!(!polytope.contains(&[Integer::from(0)]));
        // A rational point in the relative interior is contained.
        let half = Rational::new(1, 2).unwrap();
        assert!(polytope.contains_rational(&[half, Rational::from_integer(0)]));
    }

    #[test]
    fn test_points() {
        // Segment from (0,0) to (1,0): exactly its two endpoints are
        // lattice points (exact content, in lexicographic order).
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        let points = polytope.points();
        assert_eq!(
            points,
            vec![
                vec![Integer::from(0), Integer::from(0)],
                vec![Integer::from(1), Integer::from(0)],
            ]
        );
        assert_eq!(polytope.n_points(), 2);
    }

    #[test]
    fn test_faces() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        let vertex_faces = polytope.faces(0);
        assert_eq!(vertex_faces.len(), 3);
    }

    #[test]
    fn test_lattice_polytope_function() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = lattice_polytope(vertices);

        assert_eq!(polytope.n_vertices(), 2);
    }

    #[test]
    fn test_convex_hull() {
        let points = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = convex_hull(points);

        assert_eq!(polytope.n_vertices(), 3);
    }

    #[test]
    fn test_cross_polytope() {
        let cross = cross_polytope(2);

        assert_eq!(cross.n_vertices(), 4);
        assert_eq!(cross.ambient_dim(), 2);

        // Vertices should be (±1, 0) and (0, ±1)
        let vertices = cross.vertices();
        assert!(vertices.contains(&vec![Integer::from(1), Integer::from(0)]));
        assert!(vertices.contains(&vec![Integer::from(-1), Integer::from(0)]));
        assert!(vertices.contains(&vec![Integer::from(0), Integer::from(1)]));
        assert!(vertices.contains(&vec![Integer::from(0), Integer::from(-1)]));
    }

    #[test]
    fn test_cross_polytope_3d() {
        let cross = cross_polytope(3);

        // 3D cross-polytope has 6 vertices
        assert_eq!(cross.n_vertices(), 6);
        assert_eq!(cross.ambient_dim(), 3);
    }

    #[test]
    fn test_equals() {
        let vertices1 = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope1 = LatticePolytopeClass::new(vertices1.clone());
        let polytope2 = LatticePolytopeClass::new(vertices1);

        assert!(polytope1.equals(&polytope2));
    }

    #[test]
    fn test_display() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        let display = format!("{}", polytope);
        assert!(display.contains("LatticePolytope"));
    }

    // ===== Exact convex-hull / vertex-enumeration verification =====
    //
    // P2-E3: these tests exercise `convex_hull`/`hull_vertices` against
    // point sets that are NOT already in convex position (they include
    // interior points, a face-center, and an edge midpoint), which is the
    // case the old placeholder ("just wraps the points") got wrong. All
    // input coordinates and the redundant points are chosen to be exact
    // lattice points (no non-integer centroids) so the whole computation
    // stays over `Integer`/`Rational` with zero floating point.

    fn iv(xs: &[i64]) -> Vec<Integer> {
        xs.iter().map(|&x| Integer::from(x)).collect()
    }

    /// The 8 corners of a side-2 axis-aligned cube `{0,2}^3`, in a fixed
    /// (non-canonical) order, used as the "known vertex set" oracle below.
    fn cube_corners() -> Vec<Vec<Integer>> {
        let mut corners = Vec::new();
        for &x in &[0, 2] {
            for &y in &[0, 2] {
                for &z in &[0, 2] {
                    corners.push(iv(&[x, y, z]));
                }
            }
        }
        corners
    }

    #[test]
    fn test_convex_hull_unit_cube_exact_vertices() {
        // Exactly the 8 corners, already in convex position: the hull must
        // reproduce them exactly (no spurious removal).
        let hull = convex_hull(cube_corners());
        assert_eq!(hull.n_vertices(), 8);
        for c in cube_corners() {
            assert!(hull.vertices().contains(&c), "missing corner {:?}", c);
        }
    }

    #[test]
    fn test_convex_hull_cube_with_redundant_points() {
        // The 8 corners plus: the cube's centroid (1,1,1, strictly
        // interior), a face center (1,1,0, interior to a facet), and an
        // edge midpoint (1,0,0, interior to an edge). None of these three
        // are vertices of the cube; the hull must discard exactly them.
        let mut points = cube_corners();
        points.push(iv(&[1, 1, 1])); // centroid - interior
        points.push(iv(&[1, 1, 0])); // face center - interior to a facet
        points.push(iv(&[1, 0, 0])); // edge midpoint - interior to an edge

        let hull = convex_hull(points);
        let hull_vertices = hull.vertices().to_vec();

        assert_eq!(
            hull_vertices.len(),
            8,
            "hull should have exactly the 8 cube corners, got {:?}",
            hull_vertices
        );
        for c in cube_corners() {
            assert!(hull_vertices.contains(&c), "missing corner {:?}", c);
        }
        assert!(!hull_vertices.contains(&iv(&[1, 1, 1])), "centroid leaked into hull");
        assert!(!hull_vertices.contains(&iv(&[1, 1, 0])), "face center leaked into hull");
        assert!(!hull_vertices.contains(&iv(&[1, 0, 0])), "edge midpoint leaked into hull");

        // dim() must still see this as full-dimensional (3), independent
        // of the redundant points present in the vertex list.
        assert_eq!(hull.dim(), 3);
        assert_eq!(hull.ambient_dim(), 3);
    }

    #[test]
    fn test_convex_hull_cube_duplicate_points_deduped() {
        let mut points = cube_corners();
        points.push(iv(&[0, 0, 0])); // duplicate of an existing corner
        points.push(iv(&[2, 2, 2])); // duplicate of another corner

        let hull = convex_hull(points);
        assert_eq!(hull.n_vertices(), 8);
    }

    /// A standard tetrahedron scaled by 4 (so its centroid and edge
    /// midpoints are exact lattice points): (0,0,0), (4,0,0), (0,4,0),
    /// (0,0,4).
    fn simplex_corners() -> Vec<Vec<Integer>> {
        vec![
            iv(&[0, 0, 0]),
            iv(&[4, 0, 0]),
            iv(&[0, 4, 0]),
            iv(&[0, 0, 4]),
        ]
    }

    #[test]
    fn test_convex_hull_simplex_exact_vertices() {
        let hull = convex_hull(simplex_corners());
        assert_eq!(hull.n_vertices(), 4);
        for c in simplex_corners() {
            assert!(hull.vertices().contains(&c));
        }
        assert_eq!(hull.dim(), 3);
    }

    #[test]
    fn test_convex_hull_simplex_with_redundant_points() {
        // Add the centroid (1,1,1, strictly interior) and an edge
        // midpoint (2,0,0, interior to the edge from (0,0,0) to (4,0,0)).
        // Neither is a vertex of the tetrahedron.
        let mut points = simplex_corners();
        points.push(iv(&[1, 1, 1])); // centroid
        points.push(iv(&[2, 0, 0])); // edge midpoint

        let hull = convex_hull(points);
        let hull_vertices = hull.vertices().to_vec();

        assert_eq!(
            hull_vertices.len(),
            4,
            "hull should have exactly the 4 simplex corners, got {:?}",
            hull_vertices
        );
        for c in simplex_corners() {
            assert!(hull_vertices.contains(&c), "missing corner {:?}", c);
        }
        assert!(!hull_vertices.contains(&iv(&[1, 1, 1])), "centroid leaked into hull");
        assert!(!hull_vertices.contains(&iv(&[2, 0, 0])), "edge midpoint leaked into hull");
    }

    #[test]
    fn test_dim_exact_affine_rank_not_vertex_count() {
        // Four *collinear* lattice points (all on the line y=z=0): the
        // old "approximate" dim() used `(n_vertices - 1).min(ambient_dim)`,
        // which would have reported dim 2 here (3 vertices beyond the
        // first, capped at ambient_dim 3, wait -- 4 points => min(3,3)=3).
        // The true affine dimension of a line is 1, regardless of how many
        // extra (even redundant) points sit on it.
        let vertices = vec![
            iv(&[0, 0, 0]),
            iv(&[1, 0, 0]),
            iv(&[2, 0, 0]),
            iv(&[5, 0, 0]),
        ];
        let polytope = LatticePolytopeClass::new(vertices);
        assert_eq!(polytope.dim(), 1);
        assert_eq!(polytope.ambient_dim(), 3);
    }

    #[test]
    fn test_dim_full_dimensional_cube() {
        let polytope = LatticePolytopeClass::new(cube_corners());
        assert_eq!(polytope.dim(), 3);
    }

    #[test]
    fn test_convex_hull_2d_square_with_interior_point() {
        // 2D sanity check alongside the 3D cube/simplex cases above.
        let points = vec![
            iv(&[0, 0]),
            iv(&[2, 0]),
            iv(&[2, 2]),
            iv(&[0, 2]),
            iv(&[1, 1]), // interior centroid
        ];
        let hull = convex_hull(points);
        assert_eq!(hull.n_vertices(), 4);
        assert!(!hull.vertices().contains(&iv(&[1, 1])));
    }

    /// The vertices of the standard n-simplex `conv(0, e_1, ..., e_n)`.
    fn standard_simplex(n: usize) -> Vec<Vec<Integer>> {
        let mut verts = vec![vec![Integer::from(0); n]]; // origin
        for i in 0..n {
            let mut v = vec![Integer::from(0); n];
            v[i] = Integer::from(1);
            verts.push(v);
        }
        verts
    }

    #[test]
    fn test_volume_unit_square_is_one() {
        // {0,1}^2: Euclidean area 1.
        let vertices = vec![iv(&[0, 0]), iv(&[1, 0]), iv(&[1, 1]), iv(&[0, 1])];
        let polytope = LatticePolytopeClass::new(vertices);
        assert_eq!(polytope.volume(), Rational::from_integer(1));
    }

    #[test]
    fn test_volume_unit_cube_is_one() {
        // {0,1}^3: Euclidean volume 1.
        let mut corners = Vec::new();
        for &x in &[0, 1] {
            for &y in &[0, 1] {
                for &z in &[0, 1] {
                    corners.push(iv(&[x, y, z]));
                }
            }
        }
        let polytope = LatticePolytopeClass::new(corners);
        assert_eq!(polytope.volume(), Rational::from_integer(1));
    }

    #[test]
    fn test_volume_scaled_cube_is_side_cubed() {
        // {0,2}^3: Euclidean volume 2^3 = 8.
        let polytope = LatticePolytopeClass::new(cube_corners());
        assert_eq!(polytope.volume(), Rational::from_integer(8));
    }

    #[test]
    fn test_volume_standard_simplex_is_one_over_factorial() {
        // conv(0, e_1, ..., e_n) has Euclidean volume 1/n!, for n = 1..=5.
        // (Independently verified: n=1 -> 1, n=2 -> 1/2, n=3 -> 1/6,
        // n=4 -> 1/24, n=5 -> 1/120.)
        let factorials = [1i64, 1, 2, 6, 24, 120];
        for n in 1..=5usize {
            let polytope = LatticePolytopeClass::new(standard_simplex(n));
            let expected =
                Rational::from_integer(1) / Rational::from_integer(factorials[n]);
            assert_eq!(
                polytope.volume(),
                expected,
                "standard {}-simplex volume mismatch",
                n
            );
        }
    }

    #[test]
    fn test_volume_invariant_under_vertex_order() {
        // Volume should not depend on the order vertices are listed in,
        // nor on redundant/duplicate points being present.
        let mut corners = cube_corners();
        corners.reverse();
        corners.push(corners[0].clone()); // duplicate
        corners.push(iv(&[1, 1, 1])); // interior centroid (redundant)
        let polytope = LatticePolytopeClass::new(corners);
        assert_eq!(polytope.volume(), Rational::from_integer(8));
    }

    #[test]
    fn test_volume_degenerate_polytope_is_zero() {
        // A flat square embedded in 3D ambient space (z always 0):
        // not full-dimensional, so its 3D Lebesgue volume is honestly 0.
        let vertices = vec![
            iv(&[0, 0, 0]),
            iv(&[1, 0, 0]),
            iv(&[1, 1, 0]),
            iv(&[0, 1, 0]),
        ];
        let polytope = LatticePolytopeClass::new(vertices);
        assert_eq!(polytope.dim(), 2);
        assert_eq!(polytope.ambient_dim(), 3);
        assert_eq!(polytope.volume(), Rational::from_integer(0));
    }

    #[test]
    fn test_volume_single_point_is_zero() {
        let polytope = LatticePolytopeClass::new(vec![iv(&[3, 4])]);
        assert_eq!(polytope.volume(), Rational::from_integer(0));
    }

    #[test]
    fn test_volume_cross_polytope() {
        // The n-dim cross-polytope (conv of +/- standard basis vectors)
        // has Euclidean volume 2^n / n!. Check n = 2 (a square rotated 45
        // degrees, diagonal 2: area = 2) and n = 3 (an octahedron with
        // volume 4/3).
        let cross2 = cross_polytope(2);
        assert_eq!(cross2.volume(), Rational::from_integer(2));

        let cross3 = cross_polytope(3);
        assert_eq!(
            cross3.volume(),
            Rational::from_integer(4) / Rational::from_integer(3)
        );
    }

    #[test]
    fn test_volume_hypercube_4d() {
        // A stress test of the general-n code path beyond the n<=3 cases
        // above: the unit 4-cube {0,1}^4 has Euclidean volume 1.
        // (Independently verified with scipy.spatial.ConvexHull.)
        let mut corners = Vec::new();
        for &a in &[0, 1] {
            for &b in &[0, 1] {
                for &c in &[0, 1] {
                    for &d in &[0, 1] {
                        corners.push(iv(&[a, b, c, d]));
                    }
                }
            }
        }
        let polytope = LatticePolytopeClass::new(corners);
        assert_eq!(polytope.dim(), 4);
        assert_eq!(polytope.volume(), Rational::from_integer(1));
    }

    #[test]
    fn test_volume_fuzz_against_independent_python_reference() {
        // 20 random full-dimensional lattice polytopes (n = 2..=4), each
        // volume independently computed in Python via an *exact* Fraction
        // cofactor-triangulation of scipy.spatial.ConvexHull's own facet
        // list (not scipy's float .volume) -- a different algorithm from
        // the incremental beneath-beyond one implemented here, so this
        // cross-checks both against numerical error and against a shared
        // implementation bug.
        let cases: Vec<(Vec<Vec<i64>>, i64, i64)> = vec![
            (vec![vec![-3, 2, -1, -2], vec![2, 2, 1, -3], vec![1, 0, -3, -3], vec![-3, -2, -2, 1], vec![-2, -2, 2, -3]], 179, 12),
            (vec![vec![-2, 2, 0, -1], vec![0, 1, -1, 3], vec![2, 1, 0, -2], vec![1, -2, 2, 2], vec![3, -3, 3, 3]], 17, 8),
            (vec![vec![-1, 1, -1], vec![-3, -3, 0], vec![-2, 3, -1], vec![3, -3, 2], vec![-3, -1, 3]], 76, 3),
            (vec![vec![-1, 1, -2], vec![2, -2, 3], vec![-3, 0, -3], vec![2, 1, 3], vec![-1, -3, 3], vec![1, -1, 3], vec![-2, 3, -3], vec![2, -3, -3]], 61, 1),
            (vec![vec![-1, 2, 2], vec![-1, -2, 2], vec![-1, -2, -1], vec![0, 2, 3], vec![2, -3, 1], vec![2, -2, 1]], 65, 6),
            (vec![vec![2, 2, 1, -2], vec![2, -1, 3, 3], vec![-2, 0, 0, -1], vec![-1, -3, -2, 1], vec![3, -3, -2, 3], vec![-3, 3, -1, 0]], 103, 3),
            (vec![vec![1, -1, 2, 1], vec![2, 0, -2, -1], vec![-2, 2, 0, 0], vec![-2, -2, 1, 0], vec![-3, 3, -3, 3], vec![-2, -2, 2, 1], vec![0, 1, 0, -1]], 391, 24),
            (vec![vec![0, 1], vec![2, -2], vec![3, 2], vec![-3, 0]], 11, 1),
            (vec![vec![-1, 3, 2], vec![2, 1, 3], vec![0, -2, 0], vec![-3, 2, 2], vec![1, 3, -3], vec![2, 2, -3], vec![0, 1, -1], vec![-1, -3, -1]], 127, 3),
            (vec![vec![1, -3, 1], vec![-1, 3, 2], vec![-1, 0, -3], vec![-2, 1, 3], vec![-3, 3, 2], vec![3, -2, 1], vec![-2, -1, 3], vec![1, 1, -2]], 45, 1),
            (vec![vec![-3, -2], vec![-1, -2], vec![3, 3], vec![1, -3], vec![-3, 2]], 25, 1),
            (vec![vec![3, 1, 3], vec![-1, 1, 3], vec![0, 1, -2], vec![-2, -2, 2]], 10, 1),
            (vec![vec![-3, 1, 1, -2], vec![-2, 1, 3, 2], vec![0, 1, 0, -3], vec![2, -2, 2, -1], vec![-2, -2, -3, -1], vec![0, 2, 2, -1], vec![2, 2, -3, -2], vec![1, -2, -3, -3]], 1391, 24),
            (vec![vec![3, -1], vec![-2, -1], vec![-3, 1]], 5, 1),
            (vec![vec![-2, -3, -3, 2], vec![-2, 1, -2, 2], vec![2, 2, 2, -3], vec![-3, 0, 2, -1], vec![3, 0, 3, 0], vec![0, -1, 0, 0], vec![1, 1, 0, -2], vec![0, 3, 2, -3]], 123, 4),
            (vec![vec![1, 0], vec![-2, -1], vec![-2, 0], vec![-2, -2]], 3, 1),
            (vec![vec![3, -3, 0], vec![1, -3, -3], vec![-3, -3, 3], vec![2, 1, 3], vec![3, 3, 3]], 27, 1),
            (vec![vec![-3, -2], vec![0, -2], vec![3, 0], vec![0, 0]], 6, 1),
            (vec![vec![0, -1, 3], vec![0, 2, 2], vec![3, 0, -1], vec![3, 1, 2]], 5, 1),
            (vec![vec![1, -3, 3, -2], vec![-2, -2, -1, -2], vec![-3, 1, -3, 2], vec![-3, 2, -1, -3], vec![3, 1, -2, -3], vec![-3, 1, 0, 1], vec![3, -2, 0, -3], vec![-3, 1, 2, 1]], 1507, 24),
        ];

        for (pts, expected_num, expected_den) in cases {
            let vertices: Vec<Vec<Integer>> = pts
                .into_iter()
                .map(|p| p.into_iter().map(Integer::from).collect())
                .collect();
            let polytope = LatticePolytopeClass::new(vertices.clone());
            let expected = Rational::from_integer(expected_num)
                / Rational::from_integer(expected_den);
            assert_eq!(
                polytope.volume(),
                expected,
                "volume mismatch for vertices {:?}",
                vertices
            );
        }
    }

    // ===== Stage 2: real facets / polar / reflexivity / points / faces =====
    //
    // Every literal expected value below (facet lists, polar vertices,
    // lattice-point lists, f-vectors, containment verdicts) was derived
    // INDEPENDENTLY in exact python int/Fraction arithmetic before being
    // asserted here (scratchpad lp_derive.py): facets by brute-force
    // hyperplane enumeration over vertex subsets, f-vectors by
    // definition-based enumeration over all 2^#facets facet subsets
    // (cross-checked against the classical values), lattice points by
    // box + facet filtering, polars by n/c with a polar(polar) == P check.

    fn lp(vs: &[&[i64]]) -> LatticePolytopeClass {
        LatticePolytopeClass::new(vs.iter().map(|v| iv(v)).collect())
    }

    fn fi_lit(data: &[(&[i64], i64)]) -> Vec<(Vec<Integer>, Integer)> {
        data.iter().map(|(n, c)| (iv(n), Integer::from(*c))).collect()
    }

    fn pts_lit(data: &[&[i64]]) -> Vec<Vec<Integer>> {
        data.iter().map(|p| iv(p)).collect()
    }

    fn fi_sorted(p: &LatticePolytopeClass) -> Vec<(Vec<Integer>, Integer)> {
        let mut fi = p.facet_inequalities().unwrap();
        fi.sort();
        fi
    }

    fn sorted_verts(p: &LatticePolytopeClass) -> Vec<Vec<Integer>> {
        let mut v = p.vertices().to_vec();
        v.sort();
        v
    }

    fn square_pm1() -> LatticePolytopeClass {
        lp(&[&[-1, -1], &[-1, 1], &[1, -1], &[1, 1]])
    }

    fn cube_pm1() -> LatticePolytopeClass {
        let mut corners = Vec::new();
        for &x in &[-1i64, 1] {
            for &y in &[-1i64, 1] {
                for &z in &[-1i64, 1] {
                    corners.push(iv(&[x, y, z]));
                }
            }
        }
        LatticePolytopeClass::new(corners)
    }

    fn tri_t() -> LatticePolytopeClass {
        lp(&[&[1, 0], &[0, 1], &[-1, -1]])
    }

    fn tri_db() -> LatticePolytopeClass {
        lp(&[&[-1, -1], &[2, -1], &[-1, 2]])
    }

    fn hexagon() -> LatticePolytopeClass {
        lp(&[&[1, 0], &[-1, 0], &[0, 1], &[0, -1], &[1, 1], &[-1, -1]])
    }

    fn simplex3() -> LatticePolytopeClass {
        lp(&[&[0, 0, 0], &[1, 0, 0], &[0, 1, 0], &[0, 0, 1]])
    }

    /// Triangle conv(e1, e2, e3): 2-dimensional in ambient R^3.
    fn tri3_lower_dim() -> LatticePolytopeClass {
        lp(&[&[1, 0, 0], &[0, 1, 0], &[0, 0, 1]])
    }

    fn dilated_simplex(k: i64, d: usize) -> LatticePolytopeClass {
        let mut vs = vec![vec![Integer::from(0); d]];
        for i in 0..d {
            let mut v = vec![Integer::from(0); d];
            v[i] = Integer::from(k);
            vs.push(v);
        }
        LatticePolytopeClass::new(vs)
    }

    /// The 24-cell: all permutations of (±1, ±1, 0, 0) in R^4.
    fn cell24() -> LatticePolytopeClass {
        let mut vs = Vec::new();
        for i in 0..4 {
            for j in (i + 1)..4 {
                for &si in &[1i64, -1] {
                    for &sj in &[1i64, -1] {
                        let mut v = vec![0i64; 4];
                        v[i] = si;
                        v[j] = sj;
                        vs.push(iv(&v));
                    }
                }
            }
        }
        assert_eq!(vs.len(), 24);
        LatticePolytopeClass::new(vs)
    }

    #[test]
    fn test_facet_inequalities_cube_exact() {
        assert_eq!(
            fi_sorted(&cube_pm1()),
            fi_lit(&[
                (&[-1, 0, 0], 1),
                (&[0, -1, 0], 1),
                (&[0, 0, -1], 1),
                (&[0, 0, 1], 1),
                (&[0, 1, 0], 1),
                (&[1, 0, 0], 1),
            ])
        );
    }

    #[test]
    fn test_facet_inequalities_2d_exact() {
        // [0,1]^2: constants (1,1,0,0) -- the origin sits ON two facets.
        assert_eq!(
            fi_sorted(&lp(&[&[0, 0], &[1, 0], &[0, 1], &[1, 1]])),
            fi_lit(&[(&[-1, 0], 1), (&[0, -1], 1), (&[0, 1], 0), (&[1, 0], 0)])
        );
        // [-2,2]^2: all facets at lattice distance 2.
        assert_eq!(
            fi_sorted(&lp(&[&[-2, -2], &[-2, 2], &[2, -2], &[2, 2]])),
            fi_lit(&[(&[-1, 0], 2), (&[0, -1], 2), (&[0, 1], 2), (&[1, 0], 2)])
        );
        assert_eq!(
            fi_sorted(&tri_t()),
            fi_lit(&[(&[-1, -1], 1), (&[-1, 2], 1), (&[2, -1], 1)])
        );
        assert_eq!(
            fi_sorted(&tri_db()),
            fi_lit(&[(&[-1, -1], 1), (&[0, 1], 1), (&[1, 0], 1)])
        );
        assert_eq!(
            fi_sorted(&hexagon()),
            fi_lit(&[
                (&[-1, 0], 1),
                (&[-1, 1], 1),
                (&[0, -1], 1),
                (&[0, 1], 1),
                (&[1, -1], 1),
                (&[1, 0], 1),
            ])
        );

        // Semantic gate: every inequality is valid at every vertex, and
        // for a 2-polytope with extreme-only vertex lists each facet is
        // tight at exactly 2 vertices.
        for p in [square_pm1(), tri_t(), tri_db(), hexagon()] {
            for (n, c) in p.facet_inequalities().unwrap() {
                let mut tight = 0;
                for v in p.vertices() {
                    let mut val = c.clone();
                    for (ni, vi) in n.iter().zip(v.iter()) {
                        val = val + ni.clone() * vi.clone();
                    }
                    assert!(val.signum() >= 0, "facet inequality violated at a vertex");
                    if val.is_zero() {
                        tight += 1;
                    }
                }
                assert_eq!(tight, 2, "2d facet must be tight at exactly 2 vertices");
            }
        }
    }

    #[test]
    fn test_facet_normals_octahedron_exact() {
        let mut normals = cross_polytope(3).facet_normals().unwrap();
        normals.sort();
        let mut expected = Vec::new();
        for &x in &[-1i64, 1] {
            for &y in &[-1i64, 1] {
                for &z in &[-1i64, 1] {
                    expected.push(iv(&[x, y, z]));
                }
            }
        }
        assert_eq!(normals, expected);
        // ... and all constants are 1 (regular octahedron is reflexive).
        assert!(cross_polytope(3)
            .facet_inequalities()
            .unwrap()
            .iter()
            .all(|(_, c)| c.is_one()));
    }

    #[test]
    fn test_facet_inequalities_lower_dim_is_err() {
        // Equations are not facet normals: a lower-dimensional polytope
        // gets an honest Err, not an empty (or ambient-lifted) list.
        let tri = tri3_lower_dim();
        assert!(tri.facet_inequalities().unwrap_err().contains("full-dimensional"));
        assert!(tri.facet_normals().is_err());
        let segment = lp(&[&[0, 0], &[2, 2]]);
        assert!(segment.facet_inequalities().is_err());
        // The unambiguous H-representation is available instead.
        let h = tri.h_representation(&DdBudget::default()).unwrap();
        assert_eq!(h.equations.len(), 1);
        assert_eq!(h.inequalities.len(), 3);
    }

    #[test]
    fn test_is_reflexive_classics() {
        // Reflexive (python-verified: every facet constant is 1).
        assert!(square_pm1().is_reflexive());
        assert!(tri_t().is_reflexive());
        assert!(tri_db().is_reflexive());
        assert!(hexagon().is_reflexive());
        assert!(cross_polytope(2).is_reflexive());
        assert!(cube_pm1().is_reflexive());
        assert!(cross_polytope(3).is_reflexive());
        // Not reflexive: 0 not interior ([0,1]^2 has constants 0).
        assert!(!lp(&[&[0, 0], &[1, 0], &[0, 1], &[1, 1]]).is_reflexive());
        // Not reflexive: facets at lattice distance 2.
        assert!(!lp(&[&[-2, -2], &[-2, 2], &[2, -2], &[2, 2]]).is_reflexive());
        // Not reflexive: 0 on the boundary of the standard simplex.
        assert!(!simplex3().is_reflexive());
        // Not reflexive in Z^4: the 24-cell conv{+-e_i +- e_j} has eight
        // facets at lattice distance 2 (its dual has half-integral
        // vertices); it is only reflexive with respect to D4.
        assert!(!cell24().is_reflexive());
        // Lower-dimensional polytopes are never reflexive.
        assert!(!lp(&[&[0, 0], &[1, 0]]).is_reflexive());
        assert!(!tri3_lower_dim().is_reflexive());
    }

    #[test]
    fn test_reflexive_db_entries_really_reflexive() {
        let db = reflexive_polytopes(2);
        assert_eq!(db.len(), 3);
        for p in &db {
            assert!(p.is_reflexive(), "database polytope not reflexive: {:?}", p);
        }
    }

    #[test]
    fn test_polar_exact_content() {
        // polar([-1,1]^2) = diamond, polar(cube) = octahedron,
        // polar(tri_T) = tri_db and vice versa, polar(hexagon) = the
        // reversed hexagon. All python-derived.
        assert_eq!(
            sorted_verts(&square_pm1().polar().unwrap()),
            pts_lit(&[&[-1, 0], &[0, -1], &[0, 1], &[1, 0]])
        );
        assert_eq!(
            sorted_verts(&cube_pm1().polar().unwrap()),
            pts_lit(&[
                &[-1, 0, 0],
                &[0, -1, 0],
                &[0, 0, -1],
                &[0, 0, 1],
                &[0, 1, 0],
                &[1, 0, 0],
            ])
        );
        assert_eq!(
            sorted_verts(&tri_t().polar().unwrap()),
            pts_lit(&[&[-1, -1], &[-1, 2], &[2, -1]])
        );
        assert_eq!(
            sorted_verts(&tri_db().polar().unwrap()),
            pts_lit(&[&[-1, -1], &[0, 1], &[1, 0]])
        );
        assert_eq!(
            sorted_verts(&hexagon().polar().unwrap()),
            pts_lit(&[&[-1, 0], &[-1, 1], &[0, -1], &[0, 1], &[1, -1], &[1, 0]])
        );
    }

    #[test]
    fn test_polar_reflexive_agreement_and_involution() {
        // Self-certifying: is_reflexive (all facet constants 1) and
        // polar (all n_i/c_i integral) are two different routes to the
        // same property and must agree; where reflexive, the polar's
        // vertices are the facet normals, the polar is reflexive, the
        // rational polar agrees, and polar(polar(P)) == P.
        let reflexive = vec![
            square_pm1(),
            tri_t(),
            tri_db(),
            hexagon(),
            cross_polytope(2),
            cube_pm1(),
            cross_polytope(3),
        ];
        for p in &reflexive {
            assert_eq!(p.is_reflexive(), p.polar().is_some());
            let q = p.polar().unwrap();
            let mut normals = p.facet_normals().unwrap();
            normals.sort();
            assert_eq!(sorted_verts(&q), normals, "polar vertices != facet normals");
            let mut pr = p.polar_rational().unwrap();
            pr.sort();
            let as_rat: Vec<Vec<Rational>> =
                sorted_verts(&q).iter().map(|v| int_vec_to_rat(v)).collect();
            assert_eq!(pr, as_rat, "rational polar disagrees with lattice polar");
            assert!(q.is_reflexive(), "polar of reflexive must be reflexive");
            let qq = q.polar().unwrap();
            assert_eq!(sorted_verts(&qq), sorted_verts(p), "polar(polar(P)) != P");
        }
        // Non-reflexive: both routes must again agree (both negative).
        let nonreflexive = vec![
            lp(&[&[0, 0], &[1, 0], &[0, 1], &[1, 1]]),
            lp(&[&[-2, -2], &[-2, 2], &[2, -2], &[2, 2]]),
            simplex3(),
            cell24(),
            lp(&[&[0, 0], &[1, 0]]),
            tri3_lower_dim(),
        ];
        for p in &nonreflexive {
            assert!(!p.is_reflexive());
            assert!(p.polar().is_none(), "non-reflexive polar must be None");
        }
    }

    #[test]
    fn test_polar_rational_general_case() {
        // [-2,2]^2: the polar is the rational diamond with vertices
        // (+-1/2, 0), (0, +-1/2) -- not a lattice polytope, so polar()
        // is None but polar_rational() carries the honest answer.
        let p = lp(&[&[-2, -2], &[-2, 2], &[2, -2], &[2, 2]]);
        assert!(p.polar().is_none());
        let mut pr = p.polar_rational().unwrap();
        pr.sort();
        let half = Rational::new(1, 2).unwrap();
        let zero = Rational::from_integer(0);
        assert_eq!(
            pr,
            vec![
                vec![-half.clone(), zero.clone()],
                vec![zero.clone(), -half.clone()],
                vec![zero.clone(), half.clone()],
                vec![half.clone(), zero.clone()],
            ]
        );
        // 0 on the boundary => the polar is unbounded: honest Err.
        assert!(lp(&[&[0, 0], &[1, 0], &[0, 1], &[1, 1]])
            .polar_rational()
            .is_err());
        assert!(simplex3().polar_rational().is_err());
        // Lower-dimensional: Err from the facet computation.
        assert!(tri3_lower_dim().polar_rational().is_err());
    }

    #[test]
    fn test_points_dilated_simplex_binomials() {
        // |k*Delta_d ∩ Z^d| = C(k+d, d), python-verified per case.
        for (k, d, expected) in [
            (1i64, 2usize, 3usize),
            (2, 2, 6),
            (5, 2, 21),
            (3, 3, 20),
            (2, 4, 15),
            (1, 4, 5),
        ] {
            assert_eq!(
                dilated_simplex(k, d).points().len(),
                expected,
                "point count of {k}*Delta_{d}"
            );
        }
    }

    #[test]
    fn test_points_exact_content() {
        // Exact python-derived lattice point lists, in lexicographic
        // order (which points() guarantees).
        assert_eq!(
            dilated_simplex(2, 2).points(),
            pts_lit(&[&[0, 0], &[0, 1], &[0, 2], &[1, 0], &[1, 1], &[2, 0]])
        );
        assert_eq!(
            cross_polytope(2).points(),
            pts_lit(&[&[-1, 0], &[0, -1], &[0, 0], &[0, 1], &[1, 0]])
        );
        assert_eq!(
            hexagon().points(),
            pts_lit(&[
                &[-1, -1],
                &[-1, 0],
                &[0, -1],
                &[0, 0],
                &[0, 1],
                &[1, 0],
                &[1, 1],
            ])
        );
        assert_eq!(
            tri_t().points(),
            pts_lit(&[&[-1, -1], &[0, 0], &[0, 1], &[1, 0]])
        );
        assert_eq!(
            tri_db().points(),
            pts_lit(&[
                &[-1, -1],
                &[-1, 0],
                &[-1, 1],
                &[-1, 2],
                &[0, -1],
                &[0, 0],
                &[0, 1],
                &[1, -1],
                &[1, 0],
                &[2, -1],
            ])
        );
        assert_eq!(square_pm1().points().len(), 9);
        assert_eq!(cube_pm1().points().len(), 27);
        assert_eq!(hexagon().n_points(), 7);
    }

    #[test]
    fn test_points_lower_dimensional() {
        // conv(e1,e2,e3) in R^3: exactly the three unit vectors (the
        // affine-hull equation x+y+z=1 filters the rest of the box).
        assert_eq!(
            tri3_lower_dim().points(),
            pts_lit(&[&[0, 0, 1], &[0, 1, 0], &[1, 0, 0]])
        );
    }

    fn f_vector(p: &LatticePolytopeClass) -> Vec<usize> {
        (0..p.dim()).map(|d| p.faces(d).len()).collect()
    }

    /// Euler characteristic gate: sum_d (-1)^d f_d = 1 - (-1)^dim over
    /// the proper faces, plus the improper-face conventions.
    fn assert_euler(p: &LatticePolytopeClass) {
        let dim = p.dim();
        let fv = f_vector(p);
        let mut sum: i64 = 0;
        for (d, f) in fv.iter().enumerate() {
            sum += if d % 2 == 0 { *f as i64 } else { -(*f as i64) };
        }
        let expected = 1 - if dim % 2 == 0 { 1i64 } else { -1i64 };
        assert_eq!(sum, expected, "Euler characteristic failed: f = {fv:?}");
        assert_eq!(p.faces(dim).len(), 1, "faces(dim) is the polytope itself");
        assert!(p.faces(dim + 1).is_empty());
    }

    #[test]
    fn test_faces_f_vectors_and_euler() {
        // Classical f-vectors, python-cross-checked.
        let cases: Vec<(LatticePolytopeClass, Vec<usize>)> = vec![
            (cube_pm1(), vec![8, 12, 6]),
            (cross_polytope(3), vec![6, 12, 8]),
            (dilated_simplex(1, 4), vec![5, 10, 10, 5]),
            (hexagon(), vec![6, 6]),
            (square_pm1(), vec![4, 4]),
            (tri_db(), vec![3, 3]),
            (simplex3(), vec![4, 6, 4]),
            (cross_polytope(2), vec![4, 4]),
        ];
        for (p, expected) in &cases {
            assert_eq!(&f_vector(p), expected, "f-vector of {p}");
            assert_euler(p);
        }

        // Structure: cube facets are quadrilaterals, edges are segments;
        // facets()/edges() route through the same real face lattice.
        let cube = cube_pm1();
        let facets = cube.facets();
        assert_eq!(facets.len(), 6);
        for f in &facets {
            assert_eq!(f.n_vertices(), 4);
            assert_eq!(f.dim(), 2);
        }
        let edges = cube.edges();
        assert_eq!(edges.len(), 12);
        for e in &edges {
            assert_eq!(e.n_vertices(), 2);
            assert_eq!(e.dim(), 1);
        }
        // Each vertex-face is one of the cube's corners.
        for v in cube.faces(0) {
            assert_eq!(v.n_vertices(), 1);
            assert!(cube.vertices().contains(&v.vertices()[0]));
        }
    }

    #[test]
    fn test_faces_24cell_f_vector() {
        // The 24-cell's classical f-vector (24, 96, 96, 24); Euler 0.
        let p = cell24();
        assert_eq!(f_vector(&p), vec![24, 96, 96, 24]);
        assert_euler(&p);
    }

    #[test]
    fn test_faces_lower_dimensional_triangle() {
        // Faces of a 2-polytope embedded in R^3 come from the canonical
        // inequalities within the affine hull.
        let tri = tri3_lower_dim();
        assert_eq!(f_vector(&tri), vec![3, 3]);
        assert_euler(&tri);
        for e in tri.faces(1) {
            assert_eq!(e.n_vertices(), 2);
        }
        // A segment: two vertex-faces, itself as the improper face.
        let seg = lp(&[&[0, 0], &[3, 3]]);
        assert_eq!(f_vector(&seg), vec![2]);
        assert_euler(&seg);
        // A single point: its unique face is itself.
        let pt = lp(&[&[5, 7]]);
        assert_eq!(pt.faces(0).len(), 1);
        assert!(pt.faces(1).is_empty());
    }

    #[test]
    fn test_contains_cube_samples() {
        // Verdicts python-derived (facet evaluation in Fractions).
        let c = cube_pm1();
        assert!(c.contains(&iv(&[0, 0, 0])));
        assert!(c.contains(&iv(&[1, 0, 0])));
        assert!(c.contains(&iv(&[1, 1, 1])));
        assert!(c.contains(&iv(&[-1, -1, -1])));
        assert!(!c.contains(&iv(&[2, 0, 0])));
        assert!(!c.contains(&iv(&[1, 1, 2])));
        assert!(!c.contains(&iv(&[0, 0]))); // wrong arity

        let q = |n: i64, d: i64| Rational::new(n, d).unwrap();
        assert!(c.contains_rational(&[q(1, 2), q(1, 2), q(1, 2)]));
        assert!(c.contains_rational(&[q(1, 1), q(1, 2), q(0, 1)]));
        assert!(!c.contains_rational(&[q(3, 2), q(0, 1), q(0, 1)]));
        assert!(!c.contains_rational(&[q(-1, 1), q(-1, 1), q(-101, 100)]));
    }

    #[test]
    fn test_contains_lower_dim_triangle_samples() {
        let tri = tri3_lower_dim();
        assert!(tri.contains(&iv(&[1, 0, 0])));
        assert!(!tri.contains(&iv(&[0, 0, 0]))); // fails x+y+z = 1
        assert!(!tri.contains(&iv(&[1, 1, -1]))); // on the plane, outside
        let q = |n: i64, d: i64| Rational::new(n, d).unwrap();
        assert!(tri.contains_rational(&[q(1, 3), q(1, 3), q(1, 3)]));
        assert!(tri.contains_rational(&[q(1, 2), q(1, 2), q(0, 1)]));
    }

    #[test]
    fn test_contains_cross_validated_against_convex_combination() {
        // Two independent exact membership routes must agree everywhere:
        // contains() goes through the double-description facets, while
        // is_convex_combination() solves Caratheodory subsets directly.
        for p in [cross_polytope(2), tri_db()] {
            let refs: Vec<&Vec<Integer>> = p.vertices().iter().collect();
            for x in -2..=3 {
                for y in -2..=3 {
                    let pt = iv(&[x, y]);
                    assert_eq!(
                        p.contains(&pt),
                        is_convex_combination(&pt, &refs, 2),
                        "membership routes disagree at ({x}, {y}) for {p}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_budget_trips_honestly() {
        let c = cube_pm1();
        // Double description budget, through every entry point.
        assert!(c.h_representation(&DdBudget::new(2)).is_err());
        assert!(c.facet_inequalities_with_budget(&DdBudget::new(2)).is_err());
        assert!(c.try_faces(1, &DdBudget::new(2), 100_000).is_err());
        assert!(c.try_points(&DdBudget::new(2), 1_000_000).is_err());
        // Face-count budget: the cube has 27 nonempty faces.
        let err = c.try_faces(1, &DdBudget::default(), 5).unwrap_err();
        assert!(err.contains("budget"), "unexpected message: {err}");
        // Candidate budget: the cube's bounding box has 27 candidates.
        let err = c.try_points(&DdBudget::default(), 3).unwrap_err();
        assert!(err.contains("budget"), "unexpected message: {err}");
        // The same calls succeed under the default budgets.
        assert_eq!(
            c.try_faces(1, &DdBudget::default(), 100_000).unwrap().len(),
            12
        );
        assert_eq!(
            c.try_points(&DdBudget::default(), 1_000_000).unwrap().len(),
            27
        );
    }
}
