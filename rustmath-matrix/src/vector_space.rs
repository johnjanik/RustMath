//! Vector space structures and operations
//!
//! MAGMA Handbook ch. 28 (digest `rustmath-lava/docs/Chapter_28_Vector_Spaces.md`):
//! §28.3.2 `quo<V | U>` / `V / U` ([`VectorSpace::quotient`], dimension `dim V - dim U`),
//! §28.6 `ReduceVector(W, v)` ([`VectorSpace::reduce_vector`], the canonical representative of
//! the coset `v + W`), §28.7 `Coordinates(V, v)` and `VectorSpaceWithBasis`
//! ([`VectorSpace::coordinates`], [`VectorSpace::from_basis`]).
//!
//! Algorithms: exact linear algebra through [`Matrix::solve`] (Rouché–Capelli) and echelon
//! reduction, K. Hoffman and R. Kunze, *Linear Algebra*, 2nd ed., §1.4 (row-reduced echelon
//! matrices), §2.4 (coordinates), §2.6 (computations concerning subspaces). `reduce_vector` is
//! reduction by the reduced echelon basis of `W` (GAP's `SiftedVector` on a semi-echelon basis).
//!
//! Every "no" (not in the span, not a subspace, dependent basis) is a decided `Err` over an
//! exact field; internal-consistency failures are `Err(InvalidOperation)`, never values.

use crate::Matrix;
use rustmath_core::{Field, MathError, Result};

/// A vector space over a field F
///
/// Represents a finite-dimensional subspace of `F^n` (`n` = [`Self::ambient_dimension`]) given
/// by a linearly independent basis.
pub struct VectorSpace<F: Field> {
    /// The field over which this vector space is defined
    _field: std::marker::PhantomData<F>,
    /// Dimension of the vector space (always equal to `basis.len()`)
    dimension: usize,
    /// Length of the vectors: the space lives in `F^ambient_dimension`
    ambient_dimension: usize,
    /// Basis vectors (linearly independent, each of length `ambient_dimension`)
    basis: Vec<Vec<F>>,
}

impl<F: Field> VectorSpace<F> {
    /// Create a vector space with given dimension and standard basis
    ///
    /// The standard basis for F^n is {e_1, e_2, ..., e_n} where e_i has 1 in position i.
    pub fn new(dimension: usize) -> Self {
        let mut basis = Vec::with_capacity(dimension);

        for i in 0..dimension {
            let mut v = vec![F::zero(); dimension];
            v[i] = F::one();
            basis.push(v);
        }

        VectorSpace {
            _field: std::marker::PhantomData,
            dimension,
            ambient_dimension: dimension,
            basis,
        }
    }

    /// The zero subspace of `F^ambient_dimension`.
    pub fn zero_subspace(ambient_dimension: usize) -> Self {
        VectorSpace {
            _field: std::marker::PhantomData,
            dimension: 0,
            ambient_dimension,
            basis: Vec::new(),
        }
    }

    /// Create a vector space from a given basis (MAGMA `VectorSpaceWithBasis`, ch. 28 §28.7)
    ///
    /// * `Err(InvalidArgument)` if the basis is empty (the ambient dimension is then unknown;
    ///   use [`Self::zero_subspace`]).
    /// * `Err(InvalidArgument)` if the vectors do not all have the same length.
    /// * `Err(InvalidArgument)` if the vectors are linearly dependent.
    pub fn from_basis(basis: Vec<Vec<F>>) -> Result<Self> {
        if basis.is_empty() {
            return Err(MathError::InvalidArgument(
                "Basis cannot be empty".to_string(),
            ));
        }

        let dimension = basis.len();
        let vec_dim = basis[0].len();

        // Check all basis vectors have same dimension
        for v in &basis {
            if v.len() != vec_dim {
                return Err(MathError::InvalidArgument(
                    "All basis vectors must have same dimension".to_string(),
                ));
            }
        }

        let rows: Vec<F> = basis.iter().flat_map(|v| v.iter().cloned()).collect();
        if Matrix::from_vec(dimension, vec_dim, rows)?.rank()? < dimension {
            return Err(MathError::InvalidArgument(
                "basis vectors are linearly dependent".to_string(),
            ));
        }

        Ok(VectorSpace {
            _field: std::marker::PhantomData,
            dimension,
            ambient_dimension: vec_dim,
            basis,
        })
    }

    /// Get the dimension of this vector space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The length `n` of the vectors: this space is a subspace of `F^n`
    /// (MAGMA `OverDimension`).
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    /// Get a reference to the basis vectors
    pub fn basis(&self) -> &[Vec<F>] {
        &self.basis
    }

    /// The `ambient_dimension × cols.len()` matrix whose columns are `cols`.
    fn column_matrix(&self, cols: &[&Vec<F>]) -> Result<Matrix<F>> {
        let n = self.ambient_dimension;
        let mut data = Vec::with_capacity(n * cols.len());
        for i in 0..n {
            for c in cols {
                data.push(c[i].clone());
            }
        }
        Matrix::from_vec(n, cols.len(), data)
    }

    /// `Ok(Some(c))` with `v = Σ c_i b_i`, `Ok(None)` iff `v` is not in the span.
    fn coordinates_opt(&self, v: &[F]) -> Result<Option<Vec<F>>> {
        if v.len() != self.ambient_dimension {
            return Err(MathError::InvalidArgument(
                "Vector dimension doesn't match vector space ambient dimension".to_string(),
            ));
        }
        let cols: Vec<&Vec<F>> = self.basis.iter().collect();
        // Unique when it exists, because the basis is independent. With no basis vectors the
        // matrix is n×0 and `solve` returns Some([]) iff v = 0.
        self.column_matrix(&cols)?.solve(v)
    }

    /// Express a vector in terms of the basis (MAGMA `Coordinates(V, v)`, ch. 28 §28.7)
    ///
    /// Given a vector v, find the unique coefficients c_i such that v = Σ c_i * b_i
    /// where b_i are the basis vectors.
    ///
    /// * `Err(InvalidArgument)` if `v.len() != ambient_dimension`.
    /// * `Err(InvalidArgument("Vector is not in the span of the basis"))` if `v` is not in
    ///   this space (a decided answer).
    pub fn coordinates(&self, v: &[F]) -> Result<Vec<F>> {
        self.coordinates_opt(v)?.ok_or_else(|| {
            MathError::InvalidArgument("Vector is not in the span of the basis".to_string())
        })
    }

    /// Compute the direct sum with another vector space
    ///
    /// The direct sum V ⊕ W has dimension dim(V) + dim(W) and
    /// basis {v_1, ..., v_n, w_1, ..., w_m} where v_i ∈ V and w_j ∈ W,
    /// in `F^(ambient(V) + ambient(W))`.
    pub fn direct_sum(&self, other: &Self) -> Self {
        let new_dim = self.dimension + other.dimension;
        let ambient_dim = self.ambient_dimension;
        let other_ambient_dim = other.ambient_dimension;

        let mut new_basis = Vec::with_capacity(new_dim);

        // Add vectors from first space (padded with zeros)
        for v in &self.basis {
            let mut padded = v.clone();
            padded.extend(vec![F::zero(); other_ambient_dim]);
            new_basis.push(padded);
        }

        // Add vectors from second space (padded with zeros at start)
        for w in &other.basis {
            let mut padded = vec![F::zero(); ambient_dim];
            padded.extend(w.clone());
            new_basis.push(padded);
        }

        VectorSpace {
            _field: std::marker::PhantomData,
            dimension: new_dim,
            ambient_dimension: ambient_dim + other_ambient_dim,
            basis: new_basis,
        }
    }

    /// Check if a vector is in this vector space
    ///
    /// True iff `v` has length `ambient_dimension` and lies in the span of the basis (exact
    /// over an exact field). A wrong length gives `false`.
    pub fn contains(&self, v: &[F]) -> bool {
        matches!(self.coordinates_opt(v), Ok(Some(_)))
    }

    /// Compute the quotient space V/W (MAGMA `V / U`, `quo<V | U>`, ch. 28 §28.3.2)
    ///
    /// Given this vector space V and a subspace W, returns the quotient space V/W, of
    /// dimension dim(V) - dim(W). It is represented by a complement of W in V: the basis
    /// vectors of V, taken in order, that are not in the span of W and the earlier chosen ones
    /// (an analogue of MAGMA `Complement(V, U)`, ch. 28; the chosen complement may differ from
    /// MAGMA's). The basis of V/W is `{c + W}` for `c` in that complement.
    ///
    /// * `Err(InvalidArgument)` if W lives in a different ambient space.
    /// * `Err(InvalidArgument("subspace is not contained in this space"))` unless W ⊆ V.
    pub fn quotient(&self, subspace: &VectorSpace<F>) -> Result<QuotientSpace<F>> {
        if self.ambient_dimension != subspace.ambient_dimension {
            return Err(MathError::InvalidArgument(
                "Subspace must live in the same ambient space".to_string(),
            ));
        }

        for w in &subspace.basis {
            if self.coordinates_opt(w)?.is_none() {
                return Err(MathError::InvalidArgument(
                    "subspace is not contained in this space".to_string(),
                ));
            }
        }

        // Extend the basis of W by basis vectors of V, keeping each one not already in the
        // span of W and the vectors kept so far.
        let mut complement_basis: Vec<Vec<F>> = Vec::new();
        for v in &self.basis {
            let span: Vec<&Vec<F>> = subspace
                .basis
                .iter()
                .chain(complement_basis.iter())
                .collect();
            if self.column_matrix(&span)?.solve(v)?.is_none() {
                complement_basis.push(v.clone());
            }
        }

        if Some(complement_basis.len()) != self.dimension.checked_sub(subspace.dimension) {
            return Err(MathError::InvalidOperation(
                "internal: complement dimension is not dim V - dim W".to_string(),
            ));
        }

        let quotient_dimension = complement_basis.len();

        Ok(QuotientSpace {
            _field: std::marker::PhantomData,
            original_space: self.clone(),
            subspace: subspace.clone(),
            dimension: quotient_dimension,
            complement_basis,
        })
    }

    /// MAGMA `ReduceVector(W, v)` (ch. 28 §28.6), with `self = W`: the canonical
    /// representative of the coset `v + W`.
    ///
    /// The result is the unique vector of `v + W` that is zero at every pivot column of the
    /// reduced row echelon form of W's basis, so it does not depend on the chosen basis of W.
    /// `v` need not lie in any particular space containing W.
    /// `Err(InvalidArgument)` if `v.len() != ambient_dimension`.
    pub fn reduce_vector(&self, v: &[F]) -> Result<Vec<F>> {
        let n = self.ambient_dimension;
        if v.len() != n {
            return Err(MathError::InvalidArgument(
                "Vector dimension doesn't match vector space ambient dimension".to_string(),
            ));
        }
        let mut out = v.to_vec();
        if self.dimension == 0 {
            return Ok(out);
        }
        let rows: Vec<F> = self.basis.iter().flat_map(|b| b.iter().cloned()).collect();
        let rref = Matrix::from_vec(self.dimension, n, rows)?.reduced_row_echelon_form()?;
        // Row r has a 1 in pivot column c and 0 in every other pivot column, so subtracting
        // out[c]·row_r clears column c without disturbing the columns already cleared.
        for (r, &c) in rref.pivots.iter().enumerate() {
            let f = out[c].clone();
            if !f.is_zero() {
                for (j, o) in out.iter_mut().enumerate() {
                    *o = o.clone() - f.clone() * rref.matrix.data()[r * n + j].clone();
                }
            }
        }
        Ok(out)
    }
}

impl<F: Field> Clone for VectorSpace<F> {
    fn clone(&self) -> Self {
        VectorSpace {
            _field: std::marker::PhantomData,
            dimension: self.dimension,
            ambient_dimension: self.ambient_dimension,
            basis: self.basis.clone(),
        }
    }
}

/// A quotient space V/W
///
/// Represents the quotient of a vector space V by a subspace W.
/// Elements are equivalence classes [v] = v + W. The basis of V/W is `{c + W}` for `c` in
/// [`Self::complement_basis`]. MAGMA's `V / U` chooses its own basis, so coordinates computed
/// here can differ from MAGMA's by a change of basis of V/W; the kernel and the dimension of
/// the natural map agree.
pub struct QuotientSpace<F: Field> {
    _field: std::marker::PhantomData<F>,
    /// The original vector space V
    original_space: VectorSpace<F>,
    /// The subspace W
    subspace: VectorSpace<F>,
    /// Dimension of the quotient space (dim(V) - dim(W))
    dimension: usize,
    /// A basis for a complement of W in V
    complement_basis: Vec<Vec<F>>,
}

impl<F: Field> QuotientSpace<F> {
    /// Get the dimension of the quotient space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The basis vectors of the complement of W in V used as the basis of V/W
    /// (an analogue of MAGMA `Complement(V, U)`; the chosen complement may differ from
    /// MAGMA's).
    pub fn complement_basis(&self) -> &[Vec<F>] {
        &self.complement_basis
    }

    /// Project a vector from V to V/W (the natural homomorphism V → V/W ≅ F^r)
    ///
    /// Returns the coordinates of the class `v + W` in the basis `{c + W}`: writing
    /// `v = w + Σ y_j c_j` with `w ∈ W` (unique, because W's basis and the complement together
    /// form a basis of V), the result is `y`.
    ///
    /// * `Err(InvalidArgument)` if `v` has the wrong length.
    /// * `Err(InvalidArgument("Vector is not in the original vector space"))` if `v ∉ V`.
    pub fn project(&self, v: &[F]) -> Result<Vec<F>> {
        let ambient_dim = self.original_space.ambient_dimension;

        if v.len() != ambient_dim {
            return Err(MathError::InvalidArgument(
                "Vector dimension doesn't match".to_string(),
            ));
        }

        let cols: Vec<&Vec<F>> = self
            .subspace
            .basis
            .iter()
            .chain(self.complement_basis.iter())
            .collect();
        let y = self
            .original_space
            .column_matrix(&cols)?
            .solve(v)?
            .ok_or_else(|| {
                MathError::InvalidArgument("Vector is not in the original vector space".to_string())
            })?;
        Ok(y[self.subspace.dimension..].to_vec())
    }

    /// Get a representative vector for the equivalence class with given coordinates
    ///
    /// Returns `Σ coords[j]·c_j` (the zero vector of `F^n` when the quotient is 0).
    /// `Err(InvalidArgument)` if `coords.len() != dimension`.
    pub fn representative(&self, coords: &[F]) -> Result<Vec<F>> {
        if coords.len() != self.dimension {
            return Err(MathError::InvalidArgument(
                "Coordinate vector has wrong dimension".to_string(),
            ));
        }

        let ambient_dim = self.original_space.ambient_dimension;
        let mut result = vec![F::zero(); ambient_dim];

        for j in 0..self.dimension {
            for i in 0..ambient_dim {
                result[i] =
                    result[i].clone() + coords[j].clone() * self.complement_basis[j][i].clone();
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_vector_space_creation() {
        let v: VectorSpace<Rational> = VectorSpace::new(3);
        assert_eq!(v.dimension(), 3);
        assert_eq!(v.basis().len(), 3);
    }

    #[test]
    fn test_standard_basis() {
        let v: VectorSpace<Rational> = VectorSpace::new(2);
        let basis = v.basis();

        // e_1 = [1, 0]
        assert_eq!(basis[0][0], Rational::from_integer(1));
        assert_eq!(basis[0][1], Rational::from_integer(0));

        // e_2 = [0, 1]
        assert_eq!(basis[1][0], Rational::from_integer(0));
        assert_eq!(basis[1][1], Rational::from_integer(1));
    }

    #[test]
    fn test_coordinates() {
        let v: VectorSpace<Rational> = VectorSpace::new(2);

        // Vector [3, 4] should have coordinates [3, 4] in standard basis
        let vec = vec![Rational::from_integer(3), Rational::from_integer(4)];
        let coords = v.coordinates(&vec).unwrap();

        assert_eq!(coords[0], Rational::from_integer(3));
        assert_eq!(coords[1], Rational::from_integer(4));
    }

    #[test]
    fn test_direct_sum() {
        let v1: VectorSpace<Rational> = VectorSpace::new(2);
        let v2: VectorSpace<Rational> = VectorSpace::new(3);

        let sum = v1.direct_sum(&v2);

        assert_eq!(sum.dimension(), 5);
        assert_eq!(sum.basis()[0].len(), 5);
    }

    #[test]
    fn test_contains() {
        let v: VectorSpace<Rational> = VectorSpace::new(2);

        // Any 2D vector should be in R^2
        let vec = vec![Rational::from_integer(5), Rational::from_integer(7)];
        assert!(v.contains(&vec));
    }

    // Reference values: PARI/GP 2.15.4 (`matinverseimage(B, v)` solves B*X = v, `[]~` if
    // none) and GAP 4 (`Coefficients`, `SiftedVector(SemiEchelonBasis(W), v)`).

    fn qv(v: &[i64]) -> Vec<Rational> {
        v.iter().map(|&x| Rational::from_integer(x)).collect()
    }

    fn space(basis: &[&[i64]]) -> VectorSpace<Rational> {
        VectorSpace::from_basis(basis.iter().map(|b| qv(b)).collect()).unwrap()
    }

    #[test]
    fn coordinates_proper_subspace() {
        // The ch028.Coordinates defect: coordinates in a proper subspace used to be Err.
        // gp: matinverseimage(Mat([1;0]),[5,0]~) = [5]~, matinverseimage(Mat([1;0]),[0,1]~) = []~
        let w = space(&[&[1, 0]]);
        assert_eq!(w.ambient_dimension(), 2);
        assert_eq!(w.coordinates(&qv(&[5, 0])).unwrap(), qv(&[5]));
        assert!(w.coordinates(&qv(&[0, 1])).is_err());
        assert!(w.contains(&qv(&[5, 0])));
        assert!(!w.contains(&qv(&[0, 1])));
        assert!(!w.contains(&qv(&[5])));
    }

    #[test]
    fn coordinates_skew_basis() {
        // gp: matinverseimage([1,0;1,1;0,1],[2,5,3]~) = [2, 3]~, ...[1,0,0]~) = []~
        // GAP: Coefficients(Basis(W,[[1,1,0],[0,1,1]]),[2,5,3]) = [ 2, 3 ]
        let w = space(&[&[1, 1, 0], &[0, 1, 1]]);
        assert_eq!(w.coordinates(&qv(&[2, 5, 3])).unwrap(), qv(&[2, 3]));
        assert!(w.coordinates(&qv(&[1, 0, 0])).is_err());
    }

    #[test]
    fn from_basis_rejects_dependent() {
        // gp: matrank([1,2;2,4]) = 1
        assert!(VectorSpace::from_basis(vec![qv(&[1, 2]), qv(&[2, 4])]).is_err());
        assert!(VectorSpace::<Rational>::from_basis(vec![]).is_err());
        assert!(VectorSpace::from_basis(vec![qv(&[1, 2]), qv(&[1])]).is_err());
    }

    #[test]
    fn quotient_q3_by_e1e2() {
        // The ch028.quo / ch028.V.3 defect: this used to have dimension 3.
        // gp: 3 - matrank([1,0;0,1;0,0]) = 1; GAP: Dimension(V) - Dimension(W) = 1
        let v: VectorSpace<Rational> = VectorSpace::new(3);
        let q = v.quotient(&space(&[&[1, 0, 0], &[0, 1, 0]])).unwrap();
        assert_eq!(q.dimension(), 1);
        assert_eq!(q.complement_basis(), &[qv(&[0, 0, 1])][..]);
        assert_eq!(q.project(&qv(&[4, 5, 6])).unwrap(), qv(&[6]));
        assert_eq!(q.representative(&qv(&[6])).unwrap(), qv(&[0, 0, 6]));
    }

    #[test]
    fn quotient_q2_by_e1() {
        // The defect: e1 (which lies in W) used to be kept as the complement.
        // gp: matsolve(matid(2),[0,1]~)[2] = 1, matsolve(matid(2),[7,3]~)[2] = 3
        let v: VectorSpace<Rational> = VectorSpace::new(2);
        let q = v.quotient(&space(&[&[1, 0]])).unwrap();
        assert_eq!(q.dimension(), 1);
        assert_eq!(q.complement_basis(), &[qv(&[0, 1])][..]);
        assert_eq!(q.project(&qv(&[0, 1])).unwrap(), qv(&[1]));
        assert_eq!(q.project(&qv(&[7, 3])).unwrap(), qv(&[3]));
    }

    #[test]
    fn quotient_by_diagonal_line() {
        // gp: matdet([1,1,0;1,0,1;1,0,0]) = 1, matsolve([1,1,0;1,0,1;1,0,0],[1,2,3]~) =
        //     [3, -2, -1]~, so project = [-2, -1]
        let v: VectorSpace<Rational> = VectorSpace::new(3);
        let q = v.quotient(&space(&[&[1, 1, 1]])).unwrap();
        assert_eq!(q.dimension(), 2);
        assert_eq!(q.complement_basis(), &[qv(&[1, 0, 0]), qv(&[0, 1, 0])][..]);
        assert_eq!(q.project(&qv(&[1, 2, 3])).unwrap(), qv(&[-2, -1]));
        // (1,2,3) - rep(project) = (1,2,3) - (-2,-1,0) = 3·(1,1,1) ∈ W
        let rep = q.representative(&qv(&[-2, -1])).unwrap();
        assert_eq!(rep, qv(&[-2, -1, 0]));
    }

    #[test]
    fn quotient_of_proper_subspace() {
        // gp: matinverseimage([1,1;1,0;0,0],[0,1,0]~) = [1, -1]~, ...[0,0,1]~) = []~
        let v = space(&[&[1, 0, 0], &[0, 1, 0]]);
        let q = v.quotient(&space(&[&[1, 1, 0]])).unwrap();
        assert_eq!(q.dimension(), 1);
        assert_eq!(q.complement_basis(), &[qv(&[1, 0, 0])][..]);
        assert_eq!(q.project(&qv(&[0, 1, 0])).unwrap(), qv(&[-1]));
        assert!(q.project(&qv(&[0, 0, 1])).is_err());
    }

    #[test]
    fn quotient_requires_subspace() {
        // Definitional: <e2> is not contained in <e1>.
        let v = space(&[&[1, 0]]);
        assert!(v.quotient(&space(&[&[0, 1]])).is_err());
        assert!(v.quotient(&space(&[&[1, 0, 0]])).is_err());
    }

    #[test]
    fn reduce_vector_matches_gap() {
        // GAP: SiftedVector(SemiEchelonBasis(W), v):
        //   W = <(1,1,1)>, v = (1,2,3): [ 0, 1, 2 ]
        //   W = <(1,1,0),(0,1,1)>, v = (1,0,0): [ 0, 0, 1 ]; v = (4,5,6): [ 0, 0, 5 ]
        //   W = <e1,e2>, v = (4,5,6): [ 0, 0, 6 ]
        assert_eq!(
            space(&[&[1, 1, 1]]).reduce_vector(&qv(&[1, 2, 3])).unwrap(),
            qv(&[0, 1, 2])
        );
        let w2 = space(&[&[1, 1, 0], &[0, 1, 1]]);
        assert_eq!(w2.reduce_vector(&qv(&[1, 0, 0])).unwrap(), qv(&[0, 0, 1]));
        assert_eq!(w2.reduce_vector(&qv(&[4, 5, 6])).unwrap(), qv(&[0, 0, 5]));
        let w3 = space(&[&[1, 0, 0], &[0, 1, 0]]);
        assert_eq!(w3.reduce_vector(&qv(&[4, 5, 6])).unwrap(), qv(&[0, 0, 6]));
        // Canonical: another basis of the same W gives the same representative.
        let w2b = space(&[&[1, 2, 1], &[1, 0, -1]]);
        assert_eq!(w2b.reduce_vector(&qv(&[4, 5, 6])).unwrap(), qv(&[0, 0, 5]));
        assert!(w2.reduce_vector(&qv(&[1, 0])).is_err());
    }

    #[test]
    fn zero_dimensional_edges() {
        // Definitional: no panics on 0-dimensional spaces and quotients.
        let z: VectorSpace<Rational> = VectorSpace::new(0);
        assert_eq!(z.coordinates(&[]).unwrap(), Vec::<Rational>::new());
        let v: VectorSpace<Rational> = VectorSpace::new(2);
        let q = v.quotient(&VectorSpace::new(2)).unwrap();
        assert_eq!(q.dimension(), 0);
        assert_eq!(q.project(&qv(&[3, 4])).unwrap(), Vec::<Rational>::new());
        assert_eq!(q.representative(&[]).unwrap(), qv(&[0, 0]));
        // V / 0 = V, and the zero subspace contains only 0.
        let z2 = VectorSpace::<Rational>::zero_subspace(2);
        assert!(z2.contains(&qv(&[0, 0])));
        assert!(!z2.contains(&qv(&[1, 0])));
        assert_eq!(z2.reduce_vector(&qv(&[1, 2])).unwrap(), qv(&[1, 2]));
        let q0 = v.quotient(&z2).unwrap();
        assert_eq!(q0.dimension(), 2);
        assert_eq!(q0.project(&qv(&[3, 4])).unwrap(), qv(&[3, 4]));
        assert_eq!(v.direct_sum(&z2).ambient_dimension(), 4);
    }
}
