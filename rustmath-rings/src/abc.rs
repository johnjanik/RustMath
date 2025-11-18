//! # Abstract Base Classes for Rings
//!
//! This module provides trait definitions and type markers for categorizing rings
//! according to their mathematical properties. It mirrors SageMath's `sage.rings.abc`
//! module, providing a hierarchy of ring categories.
//!
//! ## Ring Hierarchy
//!
//! The hierarchy of ring structures (from most general to most specific):
//! - `Ring`: General rings with addition and multiplication
//! - `CommutativeRing`: Rings where multiplication is commutative
//! - `IntegralDomain`: Commutative rings with no zero divisors
//! - `Field`: Integral domains where every nonzero element has a multiplicative inverse
//!
//! ## Specialized Ring Categories
//!
//! This module also defines specialized categories for specific types of rings:
//! - Algebraic fields (algebraic closures, algebraic real fields)
//! - Numeric fields (complex, real, with various precision levels)
//! - Number fields (quadratic, cyclotomic)
//! - p-adic rings and fields
//! - Integer quotient rings (Z/nZ)

use rustmath_core::{Ring, CommutativeRing, Field, IntegralDomain};
use std::fmt::Debug;

/// Trait for integer rings (Z).
///
/// This represents the ring of integers with standard addition and multiplication.
/// Integer rings are integral domains but not fields.
pub trait IntegerRing: IntegralDomain + Clone + Debug {}

/// Trait for rational fields (Q).
///
/// This represents the field of rational numbers. Rational fields are fields
/// with characteristic 0.
pub trait RationalField: Field + Clone + Debug {}

/// Trait for algebraic field common properties.
///
/// Algebraic fields are fields where every element is algebraic over some base field
/// (typically the rationals). This includes algebraic closures and algebraic real fields.
pub trait AlgebraicFieldCommon: Field + Clone + Debug {
    /// Returns true if this element is algebraic (always true for algebraic fields).
    fn is_algebraic(&self) -> bool {
        true
    }
}

/// Trait for algebraic closures of the rationals (Q̄).
///
/// The algebraic closure of Q contains all algebraic numbers (roots of polynomials
/// with rational coefficients).
pub trait AlgebraicField: AlgebraicFieldCommon {
    /// The minimal polynomial of this element over Q.
    fn minimal_polynomial(&self) -> Vec<Self> where Self: Sized;

    /// The degree of this element (degree of its minimal polynomial).
    fn degree(&self) -> usize;
}

/// Trait for algebraic real fields (real algebraic numbers).
///
/// This is the real closure of Q, containing all real roots of polynomials
/// with rational coefficients.
pub trait AlgebraicRealField: AlgebraicFieldCommon + PartialOrd {
    /// Returns true if this element is positive.
    fn is_positive(&self) -> bool;

    /// Returns true if this element is negative.
    fn is_negative(&self) -> bool;
}

/// Trait for real number fields.
///
/// This represents fields of real numbers with various precision levels.
pub trait RealFieldTrait: Field + PartialOrd + Clone + Debug {
    /// Returns the precision in bits (if applicable).
    fn precision(&self) -> Option<usize> {
        None
    }

    /// Returns true if this element is positive.
    fn is_positive(&self) -> bool;

    /// Returns true if this element is negative.
    fn is_negative(&self) -> bool;
}

/// Trait for real ball arithmetic (interval arithmetic for reals).
///
/// Real balls represent real numbers with guaranteed error bounds,
/// useful for rigorous numerical computation.
pub trait RealBallField: Field + Clone + Debug {
    /// Returns the midpoint of the ball.
    fn midpoint(&self) -> Self;

    /// Returns the radius of the ball (error bound).
    fn radius(&self) -> Self;
}

/// Trait for real interval fields.
///
/// Similar to real balls but using interval representation.
pub trait RealIntervalField: Field + Clone + Debug {
    /// Returns the lower bound of the interval.
    fn lower(&self) -> Self;

    /// Returns the upper bound of the interval.
    fn upper(&self) -> Self;
}

/// Trait for real double precision fields (IEEE 754 double precision).
pub trait RealDoubleField: RealFieldTrait {
    /// Convert to f64.
    fn to_f64(&self) -> f64;

    /// Create from f64.
    fn from_f64(value: f64) -> Self;
}

/// Trait for complex number fields.
///
/// Complex fields are algebraically closed fields of characteristic 0.
pub trait ComplexFieldTrait: Field + Clone + Debug {
    /// The real part type.
    type RealPart: RealFieldTrait;

    /// Returns the real part.
    fn real(&self) -> Self::RealPart;

    /// Returns the imaginary part.
    fn imag(&self) -> Self::RealPart;

    /// Returns the complex conjugate.
    fn conjugate(&self) -> Self;

    /// Returns the absolute value (modulus).
    fn abs(&self) -> Self::RealPart;

    /// Returns the argument (phase angle).
    fn arg(&self) -> Self::RealPart;
}

/// Trait for complex ball fields (interval arithmetic for complex numbers).
pub trait ComplexBallField: Field + Clone + Debug {
    /// The real ball type.
    type RealBall: RealBallField;

    /// Returns the real part as a ball.
    fn real_ball(&self) -> Self::RealBall;

    /// Returns the imaginary part as a ball.
    fn imag_ball(&self) -> Self::RealBall;
}

/// Trait for complex interval fields.
pub trait ComplexIntervalField: Field + Clone + Debug {
    /// The real interval type.
    type RealInterval: RealIntervalField;

    /// Returns the real part as an interval.
    fn real_interval(&self) -> Self::RealInterval;

    /// Returns the imaginary part as an interval.
    fn imag_interval(&self) -> Self::RealInterval;
}

/// Trait for complex double precision fields (IEEE 754 complex double).
pub trait ComplexDoubleField: ComplexFieldTrait
where
    Self::RealPart: RealDoubleField,
{
    /// Convert to (f64, f64) representing (real, imaginary).
    fn to_f64_tuple(&self) -> (f64, f64);

    /// Create from (f64, f64) representing (real, imaginary).
    fn from_f64_tuple(real: f64, imag: f64) -> Self;
}

/// Trait for finite fields (Galois fields).
///
/// Finite fields have a prime power order p^n and are denoted GF(p^n) or F_q.
pub trait FiniteFieldTrait: Field + Clone + Debug {
    /// Returns the characteristic (the prime p).
    fn characteristic(&self) -> usize;

    /// Returns the degree (the exponent n in p^n).
    fn degree(&self) -> usize;

    /// Returns the order (cardinality) of the field (p^n).
    fn order(&self) -> usize {
        self.characteristic().pow(self.degree() as u32)
    }

    /// Returns true if this is a prime field (degree 1).
    fn is_prime_field(&self) -> bool {
        self.degree() == 1
    }
}

/// Trait for integer quotient rings (Z/nZ).
///
/// These are rings of integers modulo n, also written as Z_n.
pub trait IntegerModRing: CommutativeRing + Clone + Debug {
    /// Returns the modulus n.
    fn modulus(&self) -> usize;

    /// Returns true if this is a field (n is prime).
    fn is_field(&self) -> bool;
}

/// Trait for number fields.
///
/// Number fields are finite extensions of the rationals Q.
pub trait NumberFieldTrait: Field + Clone + Debug {
    /// Returns the degree of the number field over Q.
    fn degree(&self) -> usize;

    /// Returns the discriminant of the number field.
    fn discriminant(&self) -> i64;
}

/// Trait for quadratic number fields Q(√d).
///
/// Quadratic fields are 2-dimensional extensions of Q generated by √d.
pub trait NumberFieldQuadratic: NumberFieldTrait {
    /// Returns the discriminant d (the value under the square root).
    fn quadratic_discriminant(&self) -> i64;

    /// Returns true if this is a real quadratic field (d > 0).
    fn is_real(&self) -> bool {
        self.quadratic_discriminant() > 0
    }

    /// Returns true if this is an imaginary quadratic field (d < 0).
    fn is_imaginary(&self) -> bool {
        self.quadratic_discriminant() < 0
    }
}

/// Trait for cyclotomic number fields Q(ζ_n).
///
/// Cyclotomic fields are generated by nth roots of unity.
pub trait NumberFieldCyclotomic: NumberFieldTrait {
    /// Returns the order n of the root of unity.
    fn cyclotomic_order(&self) -> usize;

    /// Returns Euler's totient function φ(n) (the degree of the field).
    fn euler_phi(&self) -> usize;
}

/// Trait for p-adic rings Z_p.
///
/// p-adic rings are completions of Z at the prime p.
pub trait PAdicRingTrait: CommutativeRing + Clone + Debug {
    /// Returns the prime p.
    fn prime(&self) -> usize;

    /// Returns the precision (number of significant p-adic digits).
    fn precision(&self) -> usize;
}

/// Trait for p-adic fields Q_p.
///
/// p-adic fields are completions of Q at the prime p.
pub trait PAdicFieldTrait: Field + Clone + Debug {
    /// Returns the prime p.
    fn prime(&self) -> usize;

    /// Returns the precision (number of significant p-adic digits).
    fn precision(&self) -> usize;

    /// Returns the p-adic valuation (the exponent of p in the factorization).
    fn valuation(&self) -> i64;
}

/// Trait for symbolic rings.
///
/// Symbolic rings contain symbolic expressions with variables.
pub trait SymbolicRing: CommutativeRing + Clone + Debug {
    /// Type representing a variable.
    type Variable: Clone + Debug;

    /// Creates a new symbolic variable.
    fn var(&self, name: &str) -> Self::Variable;

    /// Substitutes a value for a variable.
    fn substitute(&self, var: &Self::Variable, value: &Self) -> Self;
}

/// Trait for callable symbolic expression rings.
///
/// These are symbolic expressions that can be called as functions.
pub trait CallableSymbolicExpressionRing: SymbolicRing {
    /// Calls the expression with given arguments.
    fn call(&self, args: &[Self]) -> Self;
}

/// Trait for universal cyclotomic fields.
///
/// This is the field generated by all roots of unity.
pub trait UniversalCyclotomicField: Field + Clone + Debug {
    /// Returns the conductor (smallest n such that the element is in Q(ζ_n)).
    fn conductor(&self) -> usize;
}

/// Trait for orders in number fields.
///
/// An order is a subring of a number field that is a finitely generated Z-module.
pub trait Order: IntegralDomain + Clone + Debug {
    /// The associated number field.
    type NumberField: NumberFieldTrait;

    /// Returns the discriminant of the order.
    fn discriminant(&self) -> i64;

    /// Returns the ring of integers (maximal order).
    fn maximal_order(&self) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test implementations to verify trait definitions compile

    #[derive(Clone, Debug, PartialEq)]
    struct TestRing;

    impl Ring for TestRing {
        fn add(&self, _other: &Self) -> Self { TestRing }
        fn mul(&self, _other: &Self) -> Self { TestRing }
        fn neg(&self) -> Self { TestRing }
        fn zero() -> Self { TestRing }
        fn one() -> Self { TestRing }
        fn is_zero(&self) -> bool { false }
    }

    impl CommutativeRing for TestRing {}
    impl IntegralDomain for TestRing {}
    impl IntegerRing for TestRing {}

    #[test]
    fn test_integer_ring_trait() {
        let r = TestRing;
        let _ = r.clone();
    }

    #[test]
    fn test_finite_field_trait_order_calculation() {
        // Test that order calculation works correctly for finite fields
        struct TestFiniteField {
            p: usize,
            n: usize,
        }

        impl TestFiniteField {
            fn new(p: usize, n: usize) -> Self {
                TestFiniteField { p, n }
            }
        }

        impl Clone for TestFiniteField {
            fn clone(&self) -> Self {
                TestFiniteField { p: self.p, n: self.n }
            }
        }

        impl Debug for TestFiniteField {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "GF({}^{})", self.p, self.n)
            }
        }

        impl Ring for TestFiniteField {
            fn add(&self, _other: &Self) -> Self { self.clone() }
            fn mul(&self, _other: &Self) -> Self { self.clone() }
            fn neg(&self) -> Self { self.clone() }
            fn zero() -> Self { TestFiniteField { p: 2, n: 1 } }
            fn one() -> Self { TestFiniteField { p: 2, n: 1 } }
            fn is_zero(&self) -> bool { false }
        }

        impl CommutativeRing for TestFiniteField {}
        impl IntegralDomain for TestFiniteField {}
        impl Field for TestFiniteField {
            fn inv(&self) -> Self { self.clone() }
            fn div(&self, _other: &Self) -> Self { self.clone() }
        }

        impl FiniteFieldTrait for TestFiniteField {
            fn characteristic(&self) -> usize { self.p }
            fn degree(&self) -> usize { self.n }
        }

        let gf4 = TestFiniteField::new(2, 2);
        assert_eq!(gf4.order(), 4);
        assert!(!gf4.is_prime_field());

        let gf5 = TestFiniteField::new(5, 1);
        assert_eq!(gf5.order(), 5);
        assert!(gf5.is_prime_field());
    }
}
