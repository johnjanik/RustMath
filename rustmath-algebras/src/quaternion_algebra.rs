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
}
