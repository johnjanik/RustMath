//! Non-associative algebras and Lie algebras.
//!
//! MAGMA source: Handbook chapters 100–104 (Lie algebras, quantum groups).
//!
//! The Lie bracket is **not** the ring multiplication (`[x,y]` is antisymmetric
//! and non-associative, and there is no unit), so a Lie algebra cannot be
//! modelled as a [`Ring`](crate::Ring). Instead both a general non-associative
//! product and the Lie bracket live on top of the additive/scalar structure
//! provided by [`Module`]. Associative (universal) enveloping algebras keep
//! using the `Ring`/`Algebra` tower.
//!
//! Purely additive.

use crate::{Field, Module};

/// A (possibly) non-associative algebra over a field `F`.
///
/// The underlying additive group and scalar action come from [`Module<F>`]; the
/// [`mul`](NonAssociativeAlgebra::mul) method supplies a bilinear product that is
/// **not** required to be associative or unital.
pub trait NonAssociativeAlgebra<F: Field>: Module<F> {
    /// The (bilinear, not necessarily associative) product `x · y`.
    fn mul(&self, other: &Self) -> Self;
}

/// A Lie algebra over a field `F`.
///
/// The [`bracket`](LieAlgebra::bracket) is bilinear and must satisfy
/// antisymmetry (`[x,x] = 0`) and the Jacobi identity. It is deliberately a
/// separate method from any [`NonAssociativeAlgebra::mul`] so that a type may
/// carry both an associative-style product and a Lie bracket if desired.
pub trait LieAlgebra<F: Field>: Module<F> {
    /// The Lie bracket `[x, y]`.
    fn bracket(&self, other: &Self) -> Self;

    /// Whether the bracket vanishes on `self` and `other` (they commute).
    fn commutes_with(&self, other: &Self) -> bool {
        self.bracket(other).is_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CommutativeRing, MathError, Result, Ring};
    use std::fmt;
    use std::ops::{Add, Div, Mul, Neg, Sub};

    /// A tiny `f64`-backed field, sufficient to exercise the trait bounds
    /// without depending on `rustmath-rationals`/`rustmath-reals`.
    #[derive(Clone, Copy, Debug)]
    struct Dbl(f64);

    impl fmt::Display for Dbl {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl PartialEq for Dbl {
        fn eq(&self, o: &Self) -> bool {
            (self.0 - o.0).abs() < 1e-9
        }
    }
    impl Add for Dbl {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            Dbl(self.0 + o.0)
        }
    }
    impl Sub for Dbl {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            Dbl(self.0 - o.0)
        }
    }
    impl Mul for Dbl {
        type Output = Self;
        fn mul(self, o: Self) -> Self {
            Dbl(self.0 * o.0)
        }
    }
    impl Div for Dbl {
        type Output = Self;
        fn div(self, o: Self) -> Self {
            Dbl(self.0 / o.0)
        }
    }
    impl Neg for Dbl {
        type Output = Self;
        fn neg(self) -> Self {
            Dbl(-self.0)
        }
    }
    impl Ring for Dbl {
        fn zero() -> Self {
            Dbl(0.0)
        }
        fn one() -> Self {
            Dbl(1.0)
        }
        fn is_zero(&self) -> bool {
            self.0.abs() < 1e-9
        }
        fn is_one(&self) -> bool {
            (self.0 - 1.0).abs() < 1e-9
        }
    }
    impl CommutativeRing for Dbl {}
    impl Field for Dbl {
        fn inverse(&self) -> Result<Self> {
            if self.is_zero() {
                Err(MathError::DivisionByZero)
            } else {
                Ok(Dbl(1.0 / self.0))
            }
        }
    }

    /// `R^3` as a Lie algebra under the cross product (this is `so(3)`).
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Vec3([f64; 3]);

    impl Add for Vec3 {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            Vec3([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
        }
    }
    impl Sub for Vec3 {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            Vec3([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]])
        }
    }
    impl Neg for Vec3 {
        type Output = Self;
        fn neg(self) -> Self {
            Vec3([-self.0[0], -self.0[1], -self.0[2]])
        }
    }
    impl Module<Dbl> for Vec3 {
        fn scalar_mul(&self, s: &Dbl) -> Self {
            Vec3([self.0[0] * s.0, self.0[1] * s.0, self.0[2] * s.0])
        }
        fn zero() -> Self {
            Vec3([0.0; 3])
        }
        fn is_zero(&self) -> bool {
            self.0.iter().all(|c| c.abs() < 1e-9)
        }
    }
    fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
        let (u, v) = (a.0, b.0);
        Vec3([
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ])
    }
    impl LieAlgebra<Dbl> for Vec3 {
        fn bracket(&self, other: &Self) -> Self {
            cross(self, other)
        }
    }
    impl NonAssociativeAlgebra<Dbl> for Vec3 {
        fn mul(&self, other: &Self) -> Self {
            cross(self, other)
        }
    }

    #[test]
    fn test_bracket_antisymmetry() {
        let x = Vec3([1.0, 2.0, 3.0]);
        let y = Vec3([4.0, 5.0, 6.0]);
        assert_eq!(x.bracket(&y), -(y.bracket(&x)));
        assert!(x.bracket(&x).is_zero());
    }

    #[test]
    fn test_jacobi_identity() {
        let x = Vec3([1.0, 0.0, 0.0]);
        let y = Vec3([0.0, 1.0, 0.0]);
        let z = Vec3([0.0, 0.0, 1.0]);
        // [x,[y,z]] + [y,[z,x]] + [z,[x,y]] == 0
        let j = x.bracket(&y.bracket(&z))
            + y.bracket(&z.bracket(&x))
            + z.bracket(&x.bracket(&y));
        assert!(j.is_zero());
    }

    #[test]
    fn test_nonassoc_mul_matches_bracket() {
        let x = Vec3([1.0, 2.0, 3.0]);
        let y = Vec3([0.0, 1.0, 0.0]);
        assert_eq!(NonAssociativeAlgebra::mul(&x, &y), x.bracket(&y));
        assert!(x.commutes_with(&x.scalar_mul(&Dbl(2.0)))); // x ∥ 2x
    }
}
