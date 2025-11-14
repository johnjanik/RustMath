//! Cusps of modular curves
//!
//! A cusp is a rational number p/q (including infinity) that represents
//! a point on the boundary of the upper half-plane.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{Zero, One, Signed};
use std::fmt;

/// A cusp of a modular curve, represented as p/q in lowest terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cusp {
    /// Rational cusp p/q
    Rational(BigInt, BigInt),
    /// The cusp at infinity
    Infinity,
}

impl Cusp {
    /// Create a new cusp from numerator and denominator
    pub fn new(p: BigInt, q: BigInt) -> Self {
        if q.is_zero() {
            Cusp::Infinity
        } else {
            // Reduce to lowest terms
            let g = p.gcd(&q);
            let mut p_reduced = p / &g;
            let mut q_reduced = q / &g;

            // Ensure denominator is positive
            if q_reduced.is_negative() {
                p_reduced = -p_reduced;
                q_reduced = -q_reduced;
            }

            if q_reduced.is_one() && p_reduced.is_zero() {
                Cusp::Infinity
            } else {
                Cusp::Rational(p_reduced, q_reduced)
            }
        }
    }

    /// Create cusp from i64 values
    pub fn from_i64(p: i64, q: i64) -> Self {
        Cusp::new(BigInt::from(p), BigInt::from(q))
    }

    /// Create the cusp at 0
    pub fn zero() -> Self {
        Cusp::Rational(BigInt::zero(), BigInt::one())
    }

    /// Create the cusp at infinity
    pub fn infinity() -> Self {
        Cusp::Infinity
    }

    /// Convert to a rational number (None for infinity)
    pub fn to_rational(&self) -> Option<BigRational> {
        match self {
            Cusp::Rational(p, q) => Some(BigRational::new(p.clone(), q.clone())),
            Cusp::Infinity => None,
        }
    }

    /// Get numerator (None for infinity)
    pub fn numerator(&self) -> Option<&BigInt> {
        match self {
            Cusp::Rational(p, _) => Some(p),
            Cusp::Infinity => None,
        }
    }

    /// Get denominator (None for infinity)
    pub fn denominator(&self) -> Option<&BigInt> {
        match self {
            Cusp::Rational(_, q) => Some(q),
            Cusp::Infinity => None,
        }
    }

    /// Check if this is the cusp at infinity
    pub fn is_infinity(&self) -> bool {
        matches!(self, Cusp::Infinity)
    }

    /// Apply a matrix transformation to the cusp
    /// If [[a,b],[c,d]] acts on p/q, result is (ap+bq)/(cp+dq)
    pub fn apply_matrix(&self, a: &BigInt, b: &BigInt, c: &BigInt, d: &BigInt) -> Self {
        match self {
            Cusp::Rational(p, q) => {
                let new_p = a * p + b * q;
                let new_q = c * p + d * q;
                Cusp::new(new_p, new_q)
            }
            Cusp::Infinity => {
                // Infinity maps to a/c
                Cusp::new(a.clone(), c.clone())
            }
        }
    }

    /// Check if two cusps are equivalent under the action of SL(2, Z)
    pub fn is_equivalent_sl2z(&self, other: &Cusp) -> bool {
        // Two cusps are equivalent under SL(2,Z) if and only if they differ by an integer
        match (self, other) {
            (Cusp::Infinity, Cusp::Infinity) => true,
            (Cusp::Rational(p1, q1), Cusp::Rational(p2, q2)) => {
                if q1 == q2 {
                    // Same denominator, check if numerators differ by a multiple of denominator
                    ((p1 - p2) % q1).is_zero()
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Width of cusp with respect to Gamma0(N)
    pub fn width_gamma0(&self, level: u64) -> u64 {
        match self {
            Cusp::Infinity => level / Integer::gcd(&level, &1),
            Cusp::Rational(_, q) => {
                let q_val = q.to_string().parse::<u64>().unwrap_or(1);
                level / Integer::gcd(&level, &q_val)
            }
        }
    }
}

impl fmt::Display for Cusp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cusp::Infinity => write!(f, "∞"),
            Cusp::Rational(p, q) => {
                if q.is_one() {
                    write!(f, "{}", p)
                } else {
                    write!(f, "{}/{}", p, q)
                }
            }
        }
    }
}

impl From<BigRational> for Cusp {
    fn from(r: BigRational) -> Self {
        Cusp::new(r.numer().clone(), r.denom().clone())
    }
}

impl From<i64> for Cusp {
    fn from(n: i64) -> Self {
        Cusp::from_i64(n, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cusp_creation() {
        let c1 = Cusp::from_i64(1, 2);
        assert_eq!(c1.numerator(), Some(&BigInt::from(1)));
        assert_eq!(c1.denominator(), Some(&BigInt::from(2)));

        let c2 = Cusp::from_i64(2, 4); // Should reduce to 1/2
        assert_eq!(c2.numerator(), Some(&BigInt::from(1)));
        assert_eq!(c2.denominator(), Some(&BigInt::from(2)));

        let inf = Cusp::infinity();
        assert!(inf.is_infinity());
        assert_eq!(inf.numerator(), None);
    }

    #[test]
    fn test_cusp_zero() {
        let c = Cusp::zero();
        assert_eq!(c.numerator(), Some(&BigInt::zero()));
        assert_eq!(c.denominator(), Some(&BigInt::one()));
    }

    #[test]
    fn test_cusp_matrix_action() {
        // Apply [[1,1],[0,1]] (translation by 1) to 0
        let c = Cusp::zero();
        let result = c.apply_matrix(
            &BigInt::one(),
            &BigInt::one(),
            &BigInt::zero(),
            &BigInt::one(),
        );
        assert_eq!(result.numerator(), Some(&BigInt::one()));
        assert_eq!(result.denominator(), Some(&BigInt::one()));

        // Apply to infinity
        let inf = Cusp::infinity();
        let result_inf = inf.apply_matrix(
            &BigInt::from(2),
            &BigInt::from(3),
            &BigInt::from(4),
            &BigInt::from(5),
        );
        assert_eq!(result_inf.numerator(), Some(&BigInt::from(1))); // 2/4 = 1/2
        assert_eq!(result_inf.denominator(), Some(&BigInt::from(2)));
    }

    #[test]
    fn test_cusp_equivalence() {
        let c1 = Cusp::from_i64(1, 3);
        let c2 = Cusp::from_i64(4, 3); // Differs by 1
        assert!(c1.is_equivalent_sl2z(&c2));

        let c3 = Cusp::from_i64(1, 2);
        assert!(!c1.is_equivalent_sl2z(&c3));
    }

    #[test]
    fn test_cusp_display() {
        assert_eq!(format!("{}", Cusp::from_i64(1, 2)), "1/2");
        assert_eq!(format!("{}", Cusp::from_i64(3, 1)), "3");
        assert_eq!(format!("{}", Cusp::infinity()), "∞");
    }
}
