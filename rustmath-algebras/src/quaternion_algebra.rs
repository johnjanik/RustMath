//! Quaternion Algebra
//!
//! Implements quaternion algebras over a base ring.
//!
//! A quaternion algebra over a ring R with parameters (a,b) is a 4-dimensional
//! R-algebra with basis {1, i, j, k} satisfying:
//! - i² = a
//! - j² = b
//! - ij = k = -ji
//!
//! This gives k² = (ij)(ij) = i(ji)j = -i²j² = -ab
//!
//! Corresponds to sage.algebras.quatalg.quaternion_algebra

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use std::fmt::{self, Display};

/// A quaternion algebra over a base ring
///
/// Defined by two parameters a and b from the base ring,
/// representing the relations i² = a and j² = b
pub struct QuaternionAlgebra<R: Ring> {
    /// The base ring
    base_ring: std::marker::PhantomData<R>,
    /// Parameter a (i² = a)
    a: R,
    /// Parameter b (j² = b)
    b: R,
}

impl<R: Ring> QuaternionAlgebra<R> {
    /// Create a new quaternion algebra with parameters (a, b)
    ///
    /// # Arguments
    /// * `a` - The parameter for i² = a
    /// * `b` - The parameter for j² = b
    ///
    /// # Examples
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_algebras::QuaternionAlgebra;
    ///
    /// let a = Integer::from(-1);
    /// let b = Integer::from(-1);
    /// let quat = QuaternionAlgebra::new(a, b);
    /// ```
    pub fn new(a: R, b: R) -> Self
    where
        R: Clone,
    {
        QuaternionAlgebra {
            base_ring: std::marker::PhantomData,
            a,
            b,
        }
    }

    /// Get the parameter a
    pub fn a(&self) -> &R {
        &self.a
    }

    /// Get the parameter b
    pub fn b(&self) -> &R {
        &self.b
    }

    /// Create the zero quaternion
    pub fn zero(&self) -> Quaternion<R>
    where
        R: Clone,
    {
        Quaternion::zero()
    }

    /// Create the identity quaternion (1)
    pub fn one(&self) -> Quaternion<R>
    where
        R: Clone,
    {
        Quaternion::one()
    }

    /// Create the basis element i
    pub fn i(&self) -> Quaternion<R>
    where
        R: Clone,
    {
        Quaternion::i()
    }

    /// Create the basis element j
    pub fn j(&self) -> Quaternion<R>
    where
        R: Clone,
    {
        Quaternion::j()
    }

    /// Create the basis element k
    pub fn k(&self) -> Quaternion<R>
    where
        R: Clone,
    {
        Quaternion::k()
    }

    /// Create a quaternion from components
    ///
    /// Creates the quaternion a + bi + cj + dk
    pub fn from_components(&self, a: R, b: R, c: R, d: R) -> Quaternion<R>
    where
        R: Clone,
    {
        Quaternion::new(a, b, c, d)
    }

    /// Multiply two quaternions using the algebra relations
    ///
    /// Uses i² = a, j² = b, ij = -ji = k
    pub fn multiply(&self, q1: &Quaternion<R>, q2: &Quaternion<R>) -> Quaternion<R>
    where
        R: Clone,
    {
        // (a1 + b1*i + c1*j + d1*k) * (a2 + b2*i + c2*j + d2*k)
        let a1 = &q1.real;
        let b1 = &q1.i_coeff;
        let c1 = &q1.j_coeff;
        let d1 = &q1.k_coeff;

        let a2 = &q2.real;
        let b2 = &q2.i_coeff;
        let c2 = &q2.j_coeff;
        let d2 = &q2.k_coeff;

        // Real part: a1*a2 + b1*b2*i² + c1*c2*j² + d1*d2*k²
        //          = a1*a2 + b1*b2*a + c1*c2*b + d1*d2*(-ab)
        let real = a1.clone() * a2.clone()
            + b1.clone() * b2.clone() * self.a.clone()
            + c1.clone() * c2.clone() * self.b.clone()
            - d1.clone() * d2.clone() * self.a.clone() * self.b.clone();

        // i coefficient: a1*b2 + b1*a2 + c1*d2 - d1*c2
        // (from 1*i, i*1, j*k=-i*j², k*j=i*j²)
        let i_coeff = a1.clone() * b2.clone()
            + b1.clone() * a2.clone()
            + c1.clone() * d2.clone() * self.b.clone()
            - d1.clone() * c2.clone() * self.b.clone();

        // j coefficient: a1*c2 + c1*a2 + d1*b2 - b1*d2
        let j_coeff = a1.clone() * c2.clone()
            + c1.clone() * a2.clone()
            - b1.clone() * d2.clone() * self.a.clone()
            + d1.clone() * b2.clone() * self.a.clone();

        // k coefficient: a1*d2 + d1*a2 + b1*c2 - c1*b2
        let k_coeff = a1.clone() * d2.clone()
            + d1.clone() * a2.clone()
            + b1.clone() * c2.clone()
            - c1.clone() * b2.clone();

        Quaternion::new(real, i_coeff, j_coeff, k_coeff)
    }

    /// Check if this is the standard Hamilton quaternions (a=-1, b=-1)
    pub fn is_hamilton(&self) -> bool
    where
        R: PartialEq,
    {
        self.a == R::one().negate() && self.b == R::one().negate()
    }
}

impl<R: Ring> Display for QuaternionAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Quaternion Algebra ({:?}, {:?})", self.a, self.b)
    }
}

/// An element of a quaternion algebra
///
/// Represented as a + bi + cj + dk where a, b, c, d are in the base ring
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quaternion<R: Ring> {
    /// Real (scalar) part
    real: R,
    /// Coefficient of i
    i_coeff: R,
    /// Coefficient of j
    j_coeff: R,
    /// Coefficient of k
    k_coeff: R,
}

impl<R: Ring> Quaternion<R> {
    /// Create a new quaternion from components
    ///
    /// Creates a + bi + cj + dk
    pub fn new(real: R, i_coeff: R, j_coeff: R, k_coeff: R) -> Self {
        Quaternion {
            real,
            i_coeff,
            j_coeff,
            k_coeff,
        }
    }

    /// Create the zero quaternion
    pub fn zero() -> Self
    where
        R: Clone,
    {
        Quaternion {
            real: R::zero(),
            i_coeff: R::zero(),
            j_coeff: R::zero(),
            k_coeff: R::zero(),
        }
    }

    /// Create the identity quaternion (1 + 0i + 0j + 0k)
    pub fn one() -> Self
    where
        R: Clone,
    {
        Quaternion {
            real: R::one(),
            i_coeff: R::zero(),
            j_coeff: R::zero(),
            k_coeff: R::zero(),
        }
    }

    /// Create the basis element i (0 + 1i + 0j + 0k)
    pub fn i() -> Self
    where
        R: Clone,
    {
        Quaternion {
            real: R::zero(),
            i_coeff: R::one(),
            j_coeff: R::zero(),
            k_coeff: R::zero(),
        }
    }

    /// Create the basis element j (0 + 0i + 1j + 0k)
    pub fn j() -> Self
    where
        R: Clone,
    {
        Quaternion {
            real: R::zero(),
            i_coeff: R::zero(),
            j_coeff: R::one(),
            k_coeff: R::zero(),
        }
    }

    /// Create the basis element k (0 + 0i + 0j + 1k)
    pub fn k() -> Self
    where
        R: Clone,
    {
        Quaternion {
            real: R::zero(),
            i_coeff: R::zero(),
            j_coeff: R::zero(),
            k_coeff: R::one(),
        }
    }

    /// Get the real (scalar) part
    pub fn real(&self) -> &R {
        &self.real
    }

    /// Get the i coefficient
    pub fn i_coeff(&self) -> &R {
        &self.i_coeff
    }

    /// Get the j coefficient
    pub fn j_coeff(&self) -> &R {
        &self.j_coeff
    }

    /// Get the k coefficient
    pub fn k_coeff(&self) -> &R {
        &self.k_coeff
    }

    /// Get components as a tuple (real, i, j, k)
    pub fn components(&self) -> (&R, &R, &R, &R) {
        (&self.real, &self.i_coeff, &self.j_coeff, &self.k_coeff)
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: Clone,
    {
        self.real.is_zero()
            && self.i_coeff.is_zero()
            && self.j_coeff.is_zero()
            && self.k_coeff.is_zero()
    }

    /// Check if this is one
    pub fn is_one(&self) -> bool
    where
        R: Clone,
    {
        self.real.is_one()
            && self.i_coeff.is_zero()
            && self.j_coeff.is_zero()
            && self.k_coeff.is_zero()
    }

    /// Check if this is a scalar (pure real, no i,j,k components)
    pub fn is_scalar(&self) -> bool
    where
        R: Clone,
    {
        self.i_coeff.is_zero() && self.j_coeff.is_zero() && self.k_coeff.is_zero()
    }

    /// Add two quaternions
    pub fn add(&self, other: &Self) -> Self
    where
        R: Clone,
    {
        Quaternion {
            real: self.real.clone() + other.real.clone(),
            i_coeff: self.i_coeff.clone() + other.i_coeff.clone(),
            j_coeff: self.j_coeff.clone() + other.j_coeff.clone(),
            k_coeff: self.k_coeff.clone() + other.k_coeff.clone(),
        }
    }

    /// Negate this quaternion
    pub fn negate(&self) -> Self
    where
        R: Clone,
    {
        Quaternion {
            real: self.real.negate(),
            i_coeff: self.i_coeff.negate(),
            j_coeff: self.j_coeff.negate(),
            k_coeff: self.k_coeff.negate(),
        }
    }

    /// Subtract another quaternion from this one
    pub fn subtract(&self, other: &Self) -> Self
    where
        R: Clone,
    {
        self.add(&other.negate())
    }

    /// Multiply by a scalar from the base ring
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: Clone,
    {
        Quaternion {
            real: self.real.clone() * scalar.clone(),
            i_coeff: self.i_coeff.clone() * scalar.clone(),
            j_coeff: self.j_coeff.clone() * scalar.clone(),
            k_coeff: self.k_coeff.clone() * scalar.clone(),
        }
    }

    /// Compute the conjugate
    ///
    /// The conjugate of a + bi + cj + dk is a - bi - cj - dk
    pub fn conjugate(&self) -> Self
    where
        R: Clone,
    {
        Quaternion {
            real: self.real.clone(),
            i_coeff: self.i_coeff.negate(),
            j_coeff: self.j_coeff.negate(),
            k_coeff: self.k_coeff.negate(),
        }
    }

    /// Compute the reduced norm
    ///
    /// For a quaternion q = a + bi + cj + dk in algebra (A, B):
    /// norm(q) = a² - Ab² - Bc² + ABd²
    ///
    /// Note: This requires the algebra parameters A and B
    pub fn norm(&self, alg_a: &R, alg_b: &R) -> R
    where
        R: Clone,
    {
        let a_sq = self.real.clone() * self.real.clone();
        let b_sq = self.i_coeff.clone() * self.i_coeff.clone();
        let c_sq = self.j_coeff.clone() * self.j_coeff.clone();
        let d_sq = self.k_coeff.clone() * self.k_coeff.clone();

        a_sq - alg_a.clone() * b_sq - alg_b.clone() * c_sq
            + (alg_a.clone() * alg_b.clone()) * d_sq
    }

    /// Compute the reduced trace
    ///
    /// trace(a + bi + cj + dk) = 2a
    pub fn trace(&self) -> R
    where
        R: Clone,
    {
        self.real.clone() + self.real.clone()
    }
}

impl<R: Ring> Display for Quaternion<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut parts = Vec::new();

        if !self.real.is_zero() || (self.i_coeff.is_zero() && self.j_coeff.is_zero() && self.k_coeff.is_zero()) {
            parts.push(format!("{:?}", self.real));
        }

        if !self.i_coeff.is_zero() {
            if self.i_coeff.is_one() {
                parts.push("i".to_string());
            } else {
                parts.push(format!("{:?}*i", self.i_coeff));
            }
        }

        if !self.j_coeff.is_zero() {
            if self.j_coeff.is_one() {
                parts.push("j".to_string());
            } else {
                parts.push(format!("{:?}*j", self.j_coeff));
            }
        }

        if !self.k_coeff.is_zero() {
            if self.k_coeff.is_one() {
                parts.push("k".to_string());
            } else {
                parts.push(format!("{:?}*k", self.k_coeff));
            }
        }

        if parts.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", parts.join(" + "))
        }
    }
}

/// A quaternion order
///
/// An order in a quaternion algebra is a subring that is also a full lattice.
/// It consists of quaternions with integer-like coefficients that form a ring.
///
/// Corresponds to sage.algebras.quatalg.quaternion_algebra.QuaternionOrder
#[derive(Debug, Clone)]
pub struct QuaternionOrder<R: Ring> {
    /// The ambient quaternion algebra
    algebra: QuaternionAlgebra<R>,
    /// Basis for the order (4 quaternion elements)
    basis: [Quaternion<R>; 4],
}

impl<R: Ring> QuaternionOrder<R> {
    /// Create a new quaternion order with the given basis
    ///
    /// # Arguments
    /// * `algebra` - The ambient quaternion algebra
    /// * `basis` - Four quaternions forming a Z-basis for the order
    ///
    /// # Examples
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_algebras::{QuaternionAlgebra, Quaternion, QuaternionOrder};
    ///
    /// let alg = QuaternionAlgebra::new(Integer::from(-1), Integer::from(-1));
    /// let basis = [
    ///     Quaternion::one(),
    ///     Quaternion::i(),
    ///     Quaternion::j(),
    ///     Quaternion::k(),
    /// ];
    /// let order = QuaternionOrder::new(alg, basis);
    /// ```
    pub fn new(algebra: QuaternionAlgebra<R>, basis: [Quaternion<R>; 4]) -> Self {
        QuaternionOrder { algebra, basis }
    }

    /// Get the ambient quaternion algebra
    pub fn algebra(&self) -> &QuaternionAlgebra<R> {
        &self.algebra
    }

    /// Get the basis of this order
    pub fn basis(&self) -> &[Quaternion<R>; 4] {
        &self.basis
    }

    /// Check if a quaternion is in this order
    ///
    /// A quaternion is in the order if it can be expressed as an integer
    /// linear combination of the basis elements.
    pub fn contains(&self, _q: &Quaternion<R>) -> bool
    where
        R: Clone + PartialEq,
    {
        // TODO: Implement membership testing via linear algebra
        // Would need to solve q = c₁b₁ + c₂b₂ + c₃b₃ + c₄b₄ for integer cᵢ
        false
    }

    /// Compute the discriminant of this order
    ///
    /// The discriminant is det(Tr(bᵢ*conj(bⱼ))) where bᵢ are basis elements.
    /// For the standard basis {1, i, j, k}, this gives specific values
    /// depending on the algebra parameters.
    pub fn discriminant(&self) -> R
    where
        R: Clone,
    {
        // Compute the Gram matrix G where G[i,j] = trace(basis[i] * conj(basis[j]))
        // The discriminant is det(G)

        // For now, return a placeholder
        // Full implementation requires matrix determinant over the base ring
        R::one()
    }

    /// Get the unit element (1) in this order
    pub fn one(&self) -> Quaternion<R>
    where
        R: Clone,
    {
        self.algebra.one()
    }

    /// Check if this is a maximal order
    ///
    /// A maximal order is one that is not properly contained in any other order.
    pub fn is_maximal(&self) -> bool
    where
        R: Clone + PartialEq,
    {
        // Placeholder: proper implementation requires comparing discriminants
        // with known maximal order discriminants
        false
    }
}

impl<R: Ring> Display for QuaternionOrder<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Order in {:?} with basis [", self.algebra)?;
        for (i, b) in self.basis.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", b)?;
        }
        write!(f, "]")
    }
}

/// A fractional ideal in a quaternion algebra
///
/// A fractional ideal is a lattice in the quaternion algebra that is
/// closed under left (or right) multiplication by an order.
///
/// Corresponds to sage.algebras.quatalg.quaternion_algebra.QuaternionFractionalIdeal
#[derive(Debug, Clone)]
pub struct QuaternionFractionalIdeal<R: Ring> {
    /// The ambient quaternion algebra
    algebra: QuaternionAlgebra<R>,
    /// Basis for the ideal (as quaternions)
    basis: Vec<Quaternion<R>>,
    /// The left order (elements that left-multiply the ideal into itself)
    left_order: Option<QuaternionOrder<R>>,
    /// The right order (elements that right-multiply the ideal into itself)
    right_order: Option<QuaternionOrder<R>>,
}

impl<R: Ring> QuaternionFractionalIdeal<R> {
    /// Create a new fractional ideal
    ///
    /// # Arguments
    /// * `algebra` - The ambient quaternion algebra
    /// * `basis` - Generators for the ideal
    /// * `left_order` - Optional left order
    /// * `right_order` - Optional right order
    pub fn new(
        algebra: QuaternionAlgebra<R>,
        basis: Vec<Quaternion<R>>,
        left_order: Option<QuaternionOrder<R>>,
        right_order: Option<QuaternionOrder<R>>,
    ) -> Self {
        QuaternionFractionalIdeal {
            algebra,
            basis,
            left_order,
            right_order,
        }
    }

    /// Get the basis of this ideal
    pub fn basis(&self) -> &[Quaternion<R>] {
        &self.basis
    }

    /// Get the left order of this ideal
    pub fn left_order(&self) -> Option<&QuaternionOrder<R>> {
        self.left_order.as_ref()
    }

    /// Get the right order of this ideal
    pub fn right_order(&self) -> Option<&QuaternionOrder<R>> {
        self.right_order.as_ref()
    }

    /// Get the ambient algebra
    pub fn algebra(&self) -> &QuaternionAlgebra<R> {
        &self.algebra
    }

    /// Scale the ideal by a scalar
    ///
    /// Returns a new ideal consisting of all elements multiplied by the scalar
    pub fn scale(&self, scalar: &R) -> Self
    where
        R: Clone,
    {
        let scaled_basis: Vec<_> = self
            .basis
            .iter()
            .map(|q| q.scalar_mul(scalar))
            .collect();

        QuaternionFractionalIdeal {
            algebra: self.algebra.clone(),
            basis: scaled_basis,
            left_order: self.left_order.clone(),
            right_order: self.right_order.clone(),
        }
    }

    /// Find an element of minimal norm in the ideal
    ///
    /// This is useful for various algorithms in quaternion algebra theory.
    pub fn minimal_element(&self) -> Option<Quaternion<R>>
    where
        R: Clone + PartialOrd,
    {
        if self.basis.is_empty() {
            return None;
        }

        // Simple implementation: just return the first basis element
        // A full implementation would search over linear combinations
        Some(self.basis[0].clone())
    }
}

impl<R: Ring> Display for QuaternionFractionalIdeal<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Fractional ideal in {:?} with {} generators", self.algebra, self.basis.len())
    }
}

/// A fractional ideal in a rational quaternion algebra
///
/// Specialized version for quaternion algebras over Q (rationals).
/// Provides optimized algorithms specific to the rational case.
///
/// Corresponds to sage.algebras.quatalg.quaternion_algebra.QuaternionFractionalIdeal_rational
#[derive(Debug, Clone)]
pub struct QuaternionFractionalIdealRational {
    /// The underlying fractional ideal
    inner: QuaternionFractionalIdeal<Integer>,
    /// Cached norm value if computed
    cached_norm: Option<Integer>,
}

impl QuaternionFractionalIdealRational {
    /// Create a new rational quaternion fractional ideal
    pub fn new(
        algebra: QuaternionAlgebra<Integer>,
        basis: Vec<Quaternion<Integer>>,
        left_order: Option<QuaternionOrder<Integer>>,
        right_order: Option<QuaternionOrder<Integer>>,
    ) -> Self {
        QuaternionFractionalIdealRational {
            inner: QuaternionFractionalIdeal::new(algebra, basis, left_order, right_order),
            cached_norm: None,
        }
    }

    /// Get the underlying fractional ideal
    pub fn inner(&self) -> &QuaternionFractionalIdeal<Integer> {
        &self.inner
    }

    /// Compute the norm of this ideal
    ///
    /// The norm is related to the determinant of the Gram matrix
    pub fn norm(&mut self) -> Integer {
        if let Some(ref n) = self.cached_norm {
            return n.clone();
        }

        // Placeholder: compute actual norm
        let norm = Integer::one();
        self.cached_norm = Some(norm.clone());
        norm
    }

    /// Get the basis
    pub fn basis(&self) -> &[Quaternion<Integer>] {
        self.inner.basis()
    }

    /// Get the left order
    pub fn left_order(&self) -> Option<&QuaternionOrder<Integer>> {
        self.inner.left_order()
    }

    /// Get the right order
    pub fn right_order(&self) -> Option<&QuaternionOrder<Integer>> {
        self.inner.right_order()
    }
}

impl Display for QuaternionFractionalIdealRational {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rational fractional ideal with {} generators", self.inner.basis().len())
    }
}

/// Helper function: Compute a basis for a quaternion lattice
///
/// Given a set of quaternion generators, compute a reduced basis using
/// Hermite normal form or similar lattice reduction techniques.
///
/// Corresponds to sage.algebras.quatalg.quaternion_algebra.basis_for_quaternion_lattice
pub fn basis_for_quaternion_lattice<R: Ring>(
    generators: &[Quaternion<R>],
) -> Vec<Quaternion<R>>
where
    R: Clone,
{
    // In a full implementation, this would:
    // 1. Express each quaternion as a 4-vector over the base ring
    // 2. Form a matrix with these vectors as rows
    // 3. Compute HNF (Hermite Normal Form) to get reduced basis
    // 4. Convert back to quaternions

    // For now, return the generators as-is
    generators.to_vec()
}

/// Helper function: Compute intersection of row modules over ZZ
///
/// Given two sets of row vectors over the integers, compute their intersection.
///
/// Corresponds to sage.algebras.quatalg.quaternion_algebra.intersection_of_row_modules_over_ZZ
pub fn intersection_of_row_modules_over_zz(
    module1: &Matrix<Integer>,
    module2: &Matrix<Integer>,
) -> Matrix<Integer> {
    // Placeholder: proper implementation requires:
    // 1. Stack the two matrices vertically
    // 2. Compute kernel of the combined matrix
    // 3. Extract the intersection basis

    // For now, return an empty matrix with the right dimensions
    if module1.cols() > 0 {
        Matrix::zero(0, module1.cols())
    } else {
        Matrix::zero(0, 0)
    }
}

/// Check if an object is a quaternion algebra
///
/// This is a simple type check function.
///
/// Corresponds to sage.algebras.quatalg.quaternion_algebra.is_QuaternionAlgebra
/// Note: In SageMath this is deprecated in favor of isinstance checks
pub fn is_quaternion_algebra<R: Ring>(_obj: &QuaternionAlgebra<R>) -> bool {
    // In Rust, if the type matches, it's a quaternion algebra
    true
}

/// Normalize basis at prime p
///
/// Normalize a quaternion lattice basis at a specific prime p,
/// used in maximal order computations.
///
/// Corresponds to sage.algebras.quatalg.quaternion_algebra.normalize_basis_at_p
pub fn normalize_basis_at_p<R: Ring>(
    basis: &[Quaternion<R>],
    _p: &Integer,
) -> Vec<Quaternion<R>>
where
    R: Clone,
{
    // Placeholder: full implementation involves p-adic valuation analysis
    basis.to_vec()
}

/// Solve auxiliary equation in maximal order computation
///
/// Solves specific equations that arise when computing maximal orders
/// at the prime p=2.
///
/// Corresponds to sage.algebras.quatalg.quaternion_algebra.maxord_solve_aux_eq
pub fn maxord_solve_aux_eq(
    _a: &Integer,
    _b: &Integer,
) -> Option<(Integer, Integer)> {
    // Placeholder: This solves Diophantine equations of the form
    // needed in Voight's maximal order algorithm
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_quaternion_creation() {
        let q: Quaternion<Integer> = Quaternion::zero();
        assert!(q.is_zero());

        let q: Quaternion<Integer> = Quaternion::one();
        assert!(q.is_one());

        let q: Quaternion<Integer> = Quaternion::i();
        assert_eq!(q.i_coeff(), &Integer::one());
        assert_eq!(q.real(), &Integer::zero());

        let q: Quaternion<Integer> = Quaternion::j();
        assert_eq!(q.j_coeff(), &Integer::one());

        let q: Quaternion<Integer> = Quaternion::k();
        assert_eq!(q.k_coeff(), &Integer::one());
    }

    #[test]
    fn test_quaternion_addition() {
        let q1 = Quaternion::new(
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
        );
        let q2 = Quaternion::new(
            Integer::from(5),
            Integer::from(6),
            Integer::from(7),
            Integer::from(8),
        );

        let sum = q1.add(&q2);
        assert_eq!(sum.real(), &Integer::from(6));
        assert_eq!(sum.i_coeff(), &Integer::from(8));
        assert_eq!(sum.j_coeff(), &Integer::from(10));
        assert_eq!(sum.k_coeff(), &Integer::from(12));
    }

    #[test]
    fn test_quaternion_negation() {
        let q = Quaternion::new(
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
        );
        let neg = q.negate();

        assert_eq!(neg.real(), &Integer::from(-1));
        assert_eq!(neg.i_coeff(), &Integer::from(-2));
        assert_eq!(neg.j_coeff(), &Integer::from(-3));
        assert_eq!(neg.k_coeff(), &Integer::from(-4));
    }

    #[test]
    fn test_quaternion_subtraction() {
        let q1 = Quaternion::new(
            Integer::from(5),
            Integer::from(6),
            Integer::from(7),
            Integer::from(8),
        );
        let q2 = Quaternion::new(
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
        );

        let diff = q1.subtract(&q2);
        assert_eq!(diff.real(), &Integer::from(4));
        assert_eq!(diff.i_coeff(), &Integer::from(4));
        assert_eq!(diff.j_coeff(), &Integer::from(4));
        assert_eq!(diff.k_coeff(), &Integer::from(4));
    }

    #[test]
    fn test_quaternion_scalar_multiplication() {
        let q = Quaternion::new(
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
        );
        let scalar = Integer::from(3);
        let product = q.scalar_mul(&scalar);

        assert_eq!(product.real(), &Integer::from(3));
        assert_eq!(product.i_coeff(), &Integer::from(6));
        assert_eq!(product.j_coeff(), &Integer::from(9));
        assert_eq!(product.k_coeff(), &Integer::from(12));
    }

    #[test]
    fn test_quaternion_conjugate() {
        let q = Quaternion::new(
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
        );
        let conj = q.conjugate();

        assert_eq!(conj.real(), &Integer::from(1));
        assert_eq!(conj.i_coeff(), &Integer::from(-2));
        assert_eq!(conj.j_coeff(), &Integer::from(-3));
        assert_eq!(conj.k_coeff(), &Integer::from(-4));
    }

    #[test]
    fn test_quaternion_algebra_creation() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let quat = QuaternionAlgebra::new(a, b);

        assert!(quat.is_hamilton());
    }

    #[test]
    fn test_quaternion_algebra_generators() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let quat = QuaternionAlgebra::new(a, b);

        let i: Quaternion<Integer> = quat.i();
        assert!(i.i_coeff().is_one());
        assert!(i.real().is_zero());

        let one: Quaternion<Integer> = quat.one();
        assert!(one.is_one());
    }

    #[test]
    fn test_quaternion_algebra_multiplication() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a.clone(), b.clone());

        // Test i * i = a = -1
        let i = alg.i();
        let i_squared = alg.multiply(&i, &i);
        assert_eq!(i_squared.real(), &Integer::from(-1));
        assert!(i_squared.is_scalar());

        // Test j * j = b = -1
        let j = alg.j();
        let j_squared = alg.multiply(&j, &j);
        assert_eq!(j_squared.real(), &Integer::from(-1));
        assert!(j_squared.is_scalar());

        // Test i * j = k
        let k = alg.multiply(&i, &j);
        assert_eq!(k.k_coeff(), &Integer::one());
        assert!(k.real().is_zero());
        assert!(k.i_coeff().is_zero());
        assert!(k.j_coeff().is_zero());

        // Test j * i = -k
        let neg_k = alg.multiply(&j, &i);
        assert_eq!(neg_k.k_coeff(), &Integer::from(-1));
    }

    #[test]
    fn test_quaternion_norm() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);

        // For q = 1 + 2i + 3j + 4k in (-1, -1)
        // norm = 1² - (-1)*2² - (-1)*3² + (-1)(-1)*4²
        //      = 1 + 4 + 9 + 16 = 30
        let q = Quaternion::new(
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
        );
        let n = q.norm(&a, &b);
        assert_eq!(n, Integer::from(30));
    }

    #[test]
    fn test_quaternion_trace() {
        let q = Quaternion::new(
            Integer::from(5),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
        );
        let tr = q.trace();
        assert_eq!(tr, Integer::from(10)); // 2*5 = 10
    }

    #[test]
    fn test_quaternion_is_scalar() {
        let q1 = Quaternion::new(
            Integer::from(5),
            Integer::zero(),
            Integer::zero(),
            Integer::zero(),
        );
        assert!(q1.is_scalar());

        let q2 = Quaternion::new(
            Integer::from(5),
            Integer::from(1),
            Integer::zero(),
            Integer::zero(),
        );
        assert!(!q2.is_scalar());
    }

    #[test]
    fn test_from_components() {
        let a = Integer::from(2);
        let b = Integer::from(3);
        let alg = QuaternionAlgebra::new(a, b);

        let q = alg.from_components(
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
        );

        assert_eq!(q.real(), &Integer::from(1));
        assert_eq!(q.i_coeff(), &Integer::from(2));
        assert_eq!(q.j_coeff(), &Integer::from(3));
        assert_eq!(q.k_coeff(), &Integer::from(4));
    }

    // Tests for QuaternionOrder
    #[test]
    fn test_quaternion_order_creation() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let basis = [
            Quaternion::one(),
            Quaternion::i(),
            Quaternion::j(),
            Quaternion::k(),
        ];

        let order = QuaternionOrder::new(alg.clone(), basis);
        assert_eq!(order.basis().len(), 4);
        assert!(order.one().is_one());
    }

    #[test]
    fn test_quaternion_order_basis_access() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let basis = [
            Quaternion::one(),
            Quaternion::i(),
            Quaternion::j(),
            Quaternion::k(),
        ];

        let order = QuaternionOrder::new(alg, basis.clone());
        let retrieved_basis = order.basis();

        assert_eq!(retrieved_basis[0], basis[0]);
        assert_eq!(retrieved_basis[1], basis[1]);
        assert_eq!(retrieved_basis[2], basis[2]);
        assert_eq!(retrieved_basis[3], basis[3]);
    }

    #[test]
    fn test_quaternion_order_discriminant() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let basis = [
            Quaternion::one(),
            Quaternion::i(),
            Quaternion::j(),
            Quaternion::k(),
        ];

        let order = QuaternionOrder::new(alg, basis);
        // Just verify it computes without panic (placeholder implementation)
        let _disc = order.discriminant();
    }

    // Tests for QuaternionFractionalIdeal
    #[test]
    fn test_fractional_ideal_creation() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let generators = vec![Quaternion::one(), Quaternion::i()];
        let ideal = QuaternionFractionalIdeal::new(alg, generators, None, None);

        assert_eq!(ideal.basis().len(), 2);
        assert!(ideal.left_order().is_none());
        assert!(ideal.right_order().is_none());
    }

    #[test]
    fn test_fractional_ideal_with_orders() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a.clone(), b.clone());

        let order_basis = [
            Quaternion::one(),
            Quaternion::i(),
            Quaternion::j(),
            Quaternion::k(),
        ];
        let order = QuaternionOrder::new(alg.clone(), order_basis);

        let generators = vec![Quaternion::one(), Quaternion::i()];
        let ideal = QuaternionFractionalIdeal::new(
            alg,
            generators,
            Some(order.clone()),
            Some(order.clone()),
        );

        assert!(ideal.left_order().is_some());
        assert!(ideal.right_order().is_some());
    }

    #[test]
    fn test_fractional_ideal_scaling() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let generators = vec![
            Quaternion::new(
                Integer::from(1),
                Integer::from(2),
                Integer::from(3),
                Integer::from(4),
            ),
        ];
        let ideal = QuaternionFractionalIdeal::new(alg, generators, None, None);

        let scalar = Integer::from(2);
        let scaled = ideal.scale(&scalar);

        assert_eq!(scaled.basis()[0].real(), &Integer::from(2));
        assert_eq!(scaled.basis()[0].i_coeff(), &Integer::from(4));
        assert_eq!(scaled.basis()[0].j_coeff(), &Integer::from(6));
        assert_eq!(scaled.basis()[0].k_coeff(), &Integer::from(8));
    }

    #[test]
    fn test_fractional_ideal_minimal_element() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let generators = vec![
            Quaternion::new(
                Integer::from(1),
                Integer::from(2),
                Integer::from(3),
                Integer::from(4),
            ),
        ];
        let ideal = QuaternionFractionalIdeal::new(alg, generators.clone(), None, None);

        let min_elem = ideal.minimal_element();
        assert!(min_elem.is_some());
        assert_eq!(min_elem.unwrap(), generators[0]);
    }

    #[test]
    fn test_fractional_ideal_empty_basis() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let ideal: QuaternionFractionalIdeal<Integer> = QuaternionFractionalIdeal::new(alg, vec![], None, None);

        let min_elem = ideal.minimal_element();
        assert!(min_elem.is_none());
    }

    // Tests for QuaternionFractionalIdealRational
    #[test]
    fn test_rational_fractional_ideal_creation() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let generators = vec![Quaternion::one(), Quaternion::i()];
        let ideal = QuaternionFractionalIdealRational::new(alg, generators, None, None);

        assert_eq!(ideal.basis().len(), 2);
    }

    #[test]
    fn test_rational_fractional_ideal_norm() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let generators = vec![Quaternion::one()];
        let mut ideal = QuaternionFractionalIdealRational::new(alg, generators, None, None);

        let norm1 = ideal.norm();
        let norm2 = ideal.norm(); // Should use cache
        assert_eq!(norm1, norm2);
    }

    #[test]
    fn test_rational_fractional_ideal_orders() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a.clone(), b.clone());

        let order_basis = [
            Quaternion::one(),
            Quaternion::i(),
            Quaternion::j(),
            Quaternion::k(),
        ];
        let order = QuaternionOrder::new(alg.clone(), order_basis);

        let generators = vec![Quaternion::one()];
        let ideal = QuaternionFractionalIdealRational::new(
            alg,
            generators,
            Some(order.clone()),
            Some(order.clone()),
        );

        assert!(ideal.left_order().is_some());
        assert!(ideal.right_order().is_some());
    }

    // Tests for helper functions
    #[test]
    fn test_basis_for_quaternion_lattice() {
        let generators = vec![
            Quaternion::new(
                Integer::from(1),
                Integer::from(0),
                Integer::from(0),
                Integer::from(0),
            ),
            Quaternion::new(
                Integer::from(0),
                Integer::from(1),
                Integer::from(0),
                Integer::from(0),
            ),
        ];

        let basis = basis_for_quaternion_lattice(&generators);
        assert_eq!(basis.len(), 2);
    }

    #[test]
    fn test_intersection_of_row_modules_over_zz() {
        let m1 = Matrix::zero(2, 3);
        let m2 = Matrix::zero(2, 3);

        let intersection = intersection_of_row_modules_over_zz(&m1, &m2);
        assert_eq!(intersection.cols(), 3);
    }

    #[test]
    fn test_is_quaternion_algebra() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        assert!(is_quaternion_algebra(&alg));
    }

    #[test]
    fn test_normalize_basis_at_p() {
        let basis = vec![
            Quaternion::one(),
            Quaternion::i(),
        ];

        let p = Integer::from(2);
        let normalized = normalize_basis_at_p(&basis, &p);
        assert_eq!(normalized.len(), 2);
    }

    #[test]
    fn test_maxord_solve_aux_eq() {
        let a = Integer::from(2);
        let b = Integer::from(3);

        let result = maxord_solve_aux_eq(&a, &b);
        // Placeholder returns None, so we just check it doesn't panic
        assert!(result.is_none());
    }

    #[test]
    fn test_quaternion_order_display() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let basis = [
            Quaternion::one(),
            Quaternion::i(),
            Quaternion::j(),
            Quaternion::k(),
        ];

        let order = QuaternionOrder::new(alg, basis);
        let display_str = format!("{}", order);
        assert!(display_str.contains("Order"));
    }

    #[test]
    fn test_fractional_ideal_display() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let generators = vec![Quaternion::one()];
        let ideal = QuaternionFractionalIdeal::new(alg, generators, None, None);

        let display_str = format!("{}", ideal);
        assert!(display_str.contains("Fractional ideal"));
    }

    #[test]
    fn test_rational_fractional_ideal_display() {
        let a = Integer::from(-1);
        let b = Integer::from(-1);
        let alg = QuaternionAlgebra::new(a, b);

        let generators = vec![Quaternion::one()];
        let ideal = QuaternionFractionalIdealRational::new(alg, generators, None, None);

        let display_str = format!("{}", ideal);
        assert!(display_str.contains("Rational fractional ideal"));
    }
}
