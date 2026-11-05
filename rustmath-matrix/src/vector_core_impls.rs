//! Core-trait adoption for vectors and linear transformations (MAGMA ch. 28).
//!
//! MAGMA source: Handbook chapter 28 (Vector Spaces), §28.1–28.4 (`VectorSpace`,
//! `Dimension`, `Basis`), §28.5 (`sub<>`, `Morphism`) and §28.8 (`Hom(V,W)`,
//! `Image`, `Kernel`, `Rank`, `Domain`, `Codomain`, composition).
//!
//! This module is **purely additive**: it does not touch the existing
//! [`crate::Vector`]/[`crate::VectorSpace`] logic. Instead it adds
//!
//! * a **fixed-dimension** vector [`FixedVector<R, N>`] that implements the
//!   `rustmath-core` [`Module<R>`](rustmath_core::Module) trait and, over a
//!   field, [`VectorSpace<F>`](rustmath_core::VectorSpace); and
//! * a first-class linear-transformation type [`LinearMap<F>`] built on
//!   `rustmath-core`'s object-safe [`Morphism`](rustmath_core::morphism::Morphism)
//!   layer, exposing `Kernel`, `Image`, `Rank`, `Domain`/`Codomain` and
//!   composition.
//!
//! ## Why a *fixed-dimension* vector?
//!
//! The core `Module<R>` trait requires total operators (`Add<Output = Self>`,
//! `Neg`, …) and a *static* `zero()`/`dimension()`. The crate's historical
//! [`crate::Vector`] uses **fallible** operators (`Add<Output = Result<Self>>`)
//! and carries its length only at runtime, so it cannot satisfy those bounds
//! without a breaking rewrite. Encoding the dimension in a const generic makes
//! `zero()`/`dimension()` well-defined and every operator total and
//! length-correct, which is exactly the contract `Module`/`VectorSpace` expect.
//! `From`/`Into` bridges connect it to the runtime [`crate::Vector`].

use crate::Matrix;
use crate::Vector;
use rustmath_core::morphism::Morphism;
use rustmath_core::{Field, MathError, Module, Result, Ring, VectorSpace as CoreVectorSpace};
use std::ops::{Add, Neg, Sub};

/// A vector of statically known dimension `N` over a ring `R`.
///
/// Unlike [`crate::Vector`], all arithmetic is total (dimensions always match by
/// construction), which lets it implement the core [`Module`]/[`CoreVectorSpace`]
/// traits with their static `zero()`/`dimension()` contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedVector<R: Ring, const N: usize> {
    data: Vec<R>, // invariant: data.len() == N
}

impl<R: Ring, const N: usize> FixedVector<R, N> {
    /// Build from a length-`N` array.
    pub fn new(data: [R; N]) -> Self {
        FixedVector {
            data: data.into_iter().collect(),
        }
    }

    /// Build from a slice; errors unless it has length exactly `N`.
    pub fn from_slice(data: &[R]) -> Result<Self> {
        if data.len() != N {
            return Err(MathError::InvalidArgument(format!(
                "FixedVector<_, {N}> needs exactly {N} entries, got {}",
                data.len()
            )));
        }
        Ok(FixedVector {
            data: data.to_vec(),
        })
    }

    /// The (static) dimension `N`.
    pub fn len(&self) -> usize {
        N
    }

    /// Whether this is the zero-dimensional vector.
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// Entry access.
    pub fn get(&self, i: usize) -> Option<&R> {
        self.data.get(i)
    }

    /// The underlying entries.
    pub fn as_slice(&self) -> &[R] {
        &self.data
    }

    /// The `i`-th standard basis vector `e_i`.
    pub fn basis_vector(i: usize) -> Self {
        let mut data = vec![R::zero(); N];
        if i < N {
            data[i] = R::one();
        }
        FixedVector { data }
    }

    /// Exact dot product `⟨self, other⟩`.
    pub fn dot(&self, other: &Self) -> R {
        let mut acc = R::zero();
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            acc = acc + a.clone() * b.clone();
        }
        acc
    }
}

impl<R: Ring, const N: usize> Add for FixedVector<R, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let data = self
            .data
            .into_iter()
            .zip(rhs.data)
            .map(|(a, b)| a + b)
            .collect();
        FixedVector { data }
    }
}

impl<R: Ring, const N: usize> Sub for FixedVector<R, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let data = self
            .data
            .into_iter()
            .zip(rhs.data)
            .map(|(a, b)| a - b)
            .collect();
        FixedVector { data }
    }
}

impl<R: Ring, const N: usize> Neg for FixedVector<R, N> {
    type Output = Self;
    fn neg(self) -> Self {
        FixedVector {
            data: self.data.into_iter().map(|a| -a).collect(),
        }
    }
}

impl<R: Ring, const N: usize> Module<R> for FixedVector<R, N> {
    fn scalar_mul(&self, scalar: &R) -> Self {
        FixedVector {
            data: self.data.iter().map(|a| a.clone() * scalar.clone()).collect(),
        }
    }

    fn zero() -> Self {
        FixedVector {
            data: vec![R::zero(); N],
        }
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|a| a.is_zero())
    }
}

impl<F: Field, const N: usize> CoreVectorSpace<F> for FixedVector<F, N> {
    fn dimension() -> Option<usize> {
        Some(N)
    }
}

impl<R: Ring, const N: usize> From<FixedVector<R, N>> for Vector<R> {
    fn from(v: FixedVector<R, N>) -> Self {
        Vector::new(v.data)
    }
}

impl<R: Ring, const N: usize> TryFrom<Vector<R>> for FixedVector<R, N> {
    type Error = MathError;
    fn try_from(v: Vector<R>) -> Result<Self> {
        FixedVector::from_slice(v.data())
    }
}

/// A linear transformation `V → W` between finite-dimensional `F`-vector spaces,
/// represented (as in MAGMA's `Hom(V, W)`) by its `codim × dim` matrix acting on
/// column vectors: `apply(v) = A · v`.
///
/// This is a first-class homomorphism type (ch. 28.8): it plugs into the
/// object-safe [`Morphism`] layer so maps can be boxed and composed, and it
/// exposes `Kernel`, `Image`, `Rank`, `Domain` and `Codomain`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearMap<F: Field> {
    /// Matrix of the map, `codomain_dim × domain_dim`, acting on column vectors.
    matrix: Matrix<F>,
}

impl<F: Field> LinearMap<F> {
    /// Build a linear map from its matrix (rows = codomain dim, cols = domain dim).
    pub fn from_matrix(matrix: Matrix<F>) -> Self {
        LinearMap { matrix }
    }

    /// The identity map on `F^n`.
    pub fn identity(n: usize) -> Self {
        LinearMap {
            matrix: Matrix::identity(n),
        }
    }

    /// The dimension of the domain `V` (number of matrix columns).
    pub fn domain_dim(&self) -> usize {
        self.matrix.cols()
    }

    /// The dimension of the codomain `W` (number of matrix rows).
    pub fn codomain_dim(&self) -> usize {
        self.matrix.rows()
    }

    /// The defining matrix of the map.
    pub fn matrix(&self) -> &Matrix<F> {
        &self.matrix
    }

    /// Apply the map to a runtime vector, checking dimensions.
    pub fn apply_checked(&self, v: &Vector<F>) -> Result<Vector<F>> {
        if v.dim() != self.domain_dim() {
            return Err(MathError::InvalidArgument(format!(
                "linear map expects a domain vector of length {}, got {}",
                self.domain_dim(),
                v.dim()
            )));
        }
        Ok(Vector::new(self.matvec(v.data())))
    }

    fn matvec(&self, x: &[F]) -> Vec<F> {
        let m = self.matrix.rows();
        let n = self.matrix.cols();
        let mut y = vec![F::zero(); m];
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = F::zero();
            for j in 0..n {
                acc = acc + self.matrix[(i, j)].clone() * x[j].clone();
            }
            *yi = acc;
        }
        y
    }

    /// The rank of the map (dimension of its image).
    pub fn rank(&self) -> Result<usize> {
        self.matrix.rank()
    }

    /// A basis of the kernel `{v : A·v = 0}` (ch. 28.8 `Kernel`).
    pub fn kernel(&self) -> Result<Vec<Vec<F>>> {
        self.matrix.kernel()
    }

    /// A basis of the image (ch. 28.8 `Image`), as spanning column vectors.
    pub fn image(&self) -> Result<Vec<Vec<F>>> {
        self.matrix.image()
    }

    /// Compose `self` after `first`: `(self ∘ first)(v) = self(first(v))`.
    ///
    /// Well-typed iff `first.codomain_dim() == self.domain_dim()`.
    pub fn compose(&self, first: &LinearMap<F>) -> Result<LinearMap<F>> {
        if first.codomain_dim() != self.domain_dim() {
            return Err(MathError::InvalidArgument(format!(
                "cannot compose: inner codomain {} != outer domain {}",
                first.codomain_dim(),
                self.domain_dim()
            )));
        }
        let prod = self.matrix.mul(&first.matrix)?;
        Ok(LinearMap { matrix: prod })
    }
}

impl<F: Field> Morphism for LinearMap<F> {
    type Domain = Vector<F>;
    type Codomain = Vector<F>;

    fn apply(&self, x: &Vector<F>) -> Vector<F> {
        // Total per the `Morphism` contract; domain vectors have the right length
        // by construction. Mismatches are a programming error.
        assert_eq!(
            x.dim(),
            self.domain_dim(),
            "LinearMap::apply: domain vector length {} != map domain dim {}",
            x.dim(),
            self.domain_dim()
        );
        Vector::new(self.matvec(x.data()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::morphism::{boxed, BoxedMorphism};
    use rustmath_rationals::Rational;

    fn q(n: i64) -> Rational {
        Rational::from_integer(n)
    }

    #[test]
    fn fixedvector_is_a_module() {
        let u = FixedVector::<Rational, 3>::new([q(1), q(2), q(3)]);
        let v = FixedVector::<Rational, 3>::new([q(4), q(5), q(6)]);

        // Module axioms exercised: add, sub, neg, scalar_mul, zero.
        let s = u.clone() + v.clone();
        assert_eq!(s, FixedVector::new([q(5), q(7), q(9)]));

        let d = v.clone() - u.clone();
        assert_eq!(d, FixedVector::new([q(3), q(3), q(3)]));

        let scaled = u.scalar_mul(&q(2));
        assert_eq!(scaled, FixedVector::new([q(2), q(4), q(6)]));

        let z = <FixedVector<Rational, 3> as Module<Rational>>::zero();
        assert!(z.is_zero());
        assert_eq!(u.clone() + z, u.clone());
        assert_eq!(u.clone() + (-u.clone()), <FixedVector<Rational, 3> as Module<Rational>>::zero());
    }

    #[test]
    fn fixedvector_is_a_vector_space() {
        assert_eq!(
            <FixedVector<Rational, 4> as CoreVectorSpace<Rational>>::dimension(),
            Some(4)
        );
        // dot product and standard basis
        let e0 = FixedVector::<Rational, 2>::basis_vector(0);
        let e1 = FixedVector::<Rational, 2>::basis_vector(1);
        assert_eq!(e0.dot(&e1), q(0));
        assert_eq!(e0.dot(&e0), q(1));
    }

    #[test]
    fn fixedvector_roundtrips_with_runtime_vector() {
        let fv = FixedVector::<Rational, 2>::new([q(7), q(8)]);
        let rt: Vector<Rational> = fv.clone().into();
        let back: FixedVector<Rational, 2> = rt.try_into().unwrap();
        assert_eq!(fv, back);
    }

    #[test]
    fn linear_map_apply_kernel_image_rank() {
        // Projection onto the first coordinate in F^2: A = [[1,0],[0,0]].
        let a = Matrix::from_vec(2, 2, vec![q(1), q(0), q(0), q(0)]).unwrap();
        let f = LinearMap::from_matrix(a);

        let out = f.apply(&Vector::new(vec![q(5), q(9)]));
        assert_eq!(out.data(), &[q(5), q(0)]);

        assert_eq!(f.rank().unwrap(), 1);
        // Kernel is spanned by (0,1); image by (1,0).
        assert_eq!(f.kernel().unwrap().len(), 1);
        assert_eq!(f.image().unwrap().len(), 1);
    }

    #[test]
    fn linear_map_composition_matches_matrix_product() {
        // a: F^2 -> F^2 swap; b: F^2 -> F^2 scale-by-2. b∘a then apply.
        let swap = Matrix::from_vec(2, 2, vec![q(0), q(1), q(1), q(0)]).unwrap();
        let scale = Matrix::from_vec(2, 2, vec![q(2), q(0), q(0), q(2)]).unwrap();
        let a = LinearMap::from_matrix(swap);
        let b = LinearMap::from_matrix(scale);

        let comp = b.compose(&a).unwrap();
        let v = Vector::new(vec![q(3), q(5)]);
        // (b∘a)(3,5) = b(5,3) = (10,6)
        assert_eq!(comp.apply(&v).data(), &[q(10), q(6)]);

        // Object-safe boxing round-trip (uses the erased Morphism layer).
        let erased: BoxedMorphism<Vector<Rational>, Vector<Rational>> = boxed(comp);
        assert_eq!(erased.apply(&v).data(), &[q(10), q(6)]);
    }

    #[test]
    fn linear_map_dimension_mismatch_is_reported() {
        let a = Matrix::from_vec(2, 3, vec![q(1), q(0), q(0), q(0), q(1), q(0)]).unwrap();
        let f = LinearMap::from_matrix(a);
        assert_eq!(f.domain_dim(), 3);
        assert_eq!(f.codomain_dim(), 2);
        assert!(f.apply_checked(&Vector::new(vec![q(1), q(2)])).is_err());
    }
}
