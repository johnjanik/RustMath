//! Cusps of modular curves
//!
//! A cusp is a rational number p/q (including infinity) that represents
//! a point on the boundary of the upper half-plane.

use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;

/// A cusp of a modular curve, represented as p/q in lowest terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cusp {
    /// Rational cusp p/q
    Rational(Integer, Integer),
    /// The cusp at infinity
    Infinity,
}

impl Cusp {
    /// Create a new cusp from numerator and denominator
    pub fn new(p: Integer, q: Integer) -> Self {
        if q.is_zero() {
            Cusp::Infinity
        } else {
            // Reduce to lowest terms
            let g = p.gcd(&q);
            let mut p_reduced = &p / &g;
            let mut q_reduced = &q / &g;

            // Ensure denominator is positive
            if q_reduced.signum() < 0 {
                p_reduced = -p_reduced;
                q_reduced = -q_reduced;
            }

            Cusp::Rational(p_reduced, q_reduced)
        }
    }

    /// Create cusp from i64 values
    pub fn from_i64(p: i64, q: i64) -> Self {
        Cusp::new(Integer::from(p), Integer::from(q))
    }

    /// Create the cusp at 0
    pub fn zero() -> Self {
        Cusp::Rational(Integer::zero(), Integer::one())
    }

    /// Create the cusp at infinity
    pub fn infinity() -> Self {
        Cusp::Infinity
    }

    /// Convert to a rational number (None for infinity)
    pub fn to_rational(&self) -> Option<Rational> {
        match self {
            Cusp::Rational(p, q) => Some(
                Rational::new(p.clone(), q.clone())
                    .expect("Rational cusp has nonzero denominator"),
            ),
            Cusp::Infinity => None,
        }
    }

    /// Get numerator (None for infinity)
    pub fn numerator(&self) -> Option<&Integer> {
        match self {
            Cusp::Rational(p, _) => Some(p),
            Cusp::Infinity => None,
        }
    }

    /// Get denominator (None for infinity)
    pub fn denominator(&self) -> Option<&Integer> {
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
    pub fn apply_matrix(&self, a: &Integer, b: &Integer, c: &Integer, d: &Integer) -> Self {
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
                    (&(p1 - p2) % q1).is_zero()
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Width of cusp with respect to Gamma0(N)
    pub fn width_gamma0(&self, level: u64) -> u64 {
        fn gcd_u64(a: u64, b: u64) -> u64 {
            if b == 0 { a } else { gcd_u64(b, a % b) }
        }
        match self {
            Cusp::Infinity => level / gcd_u64(level, 1),
            Cusp::Rational(_, q) => {
                let q_val = q.to_string().parse::<u64>().unwrap_or(1);
                level / gcd_u64(level, q_val)
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

impl From<Rational> for Cusp {
    fn from(r: Rational) -> Self {
        Cusp::new(r.numerator().clone(), r.denominator().clone())
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
        assert_eq!(c1.numerator(), Some(&Integer::from(1)));
        assert_eq!(c1.denominator(), Some(&Integer::from(2)));

        let c2 = Cusp::from_i64(2, 4); // Should reduce to 1/2
        assert_eq!(c2.numerator(), Some(&Integer::from(1)));
        assert_eq!(c2.denominator(), Some(&Integer::from(2)));

        let inf = Cusp::infinity();
        assert!(inf.is_infinity());
        assert_eq!(inf.numerator(), None);
    }

    #[test]
    fn test_cusp_zero() {
        let c = Cusp::zero();
        assert_eq!(c.numerator(), Some(&Integer::zero()));
        assert_eq!(c.denominator(), Some(&Integer::one()));
    }

    #[test]
    fn test_cusp_zero_over_q_is_not_infinity() {
        // 0/q is the cusp 0, not the cusp at infinity, for every nonzero q.
        // (Previously `Cusp::new` special-cased a reduced form of
        // `Rational(0, 1)` and mapped it to `Infinity`, which conflated the
        // rational cusp 0 with the point at infinity.)
        for q in [1i64, 2, -2, 3, -3, 100] {
            let c = Cusp::from_i64(0, q);
            assert!(!c.is_infinity(), "0/{q} must not be Infinity");
            assert_eq!(c, Cusp::zero(), "0/{q} must reduce to the cusp 0");
            assert_eq!(c.numerator(), Some(&Integer::zero()));
            assert_eq!(c.denominator(), Some(&Integer::one()));
        }
    }

    #[test]
    fn test_cusp_zero_infinity_and_half_are_pairwise_distinct() {
        let zero = Cusp::zero();
        let infinity = Cusp::infinity();
        let half = Cusp::from_i64(1, 2);

        assert_ne!(zero, infinity);
        assert_ne!(zero, half);
        assert_ne!(infinity, half);

        assert!(!zero.is_infinity());
        assert!(infinity.is_infinity());
        assert!(!half.is_infinity());

        assert!(!zero.is_equivalent_sl2z(&infinity));
        assert!(!zero.is_equivalent_sl2z(&half));
        assert!(!infinity.is_equivalent_sl2z(&half));
    }

    #[test]
    fn test_cusp_only_zero_denominator_is_infinity() {
        // The *only* route to `Cusp::Infinity` is a literal zero
        // denominator; a zero numerator with nonzero denominator must not
        // take that path.
        assert_eq!(Cusp::new(Integer::from(5), Integer::zero()), Cusp::Infinity);
        assert_eq!(Cusp::new(Integer::zero(), Integer::zero()), Cusp::Infinity);
        assert_ne!(
            Cusp::new(Integer::zero(), Integer::from(7)),
            Cusp::Infinity
        );
    }

    #[test]
    fn test_cusp_matrix_action() {
        // Apply [[1,1],[0,1]] (translation by 1) to 0
        let c = Cusp::zero();
        let result = c.apply_matrix(
            &Integer::one(),
            &Integer::one(),
            &Integer::zero(),
            &Integer::one(),
        );
        assert_eq!(result.numerator(), Some(&Integer::one()));
        assert_eq!(result.denominator(), Some(&Integer::one()));

        // Apply to infinity
        let inf = Cusp::infinity();
        let result_inf = inf.apply_matrix(
            &Integer::from(2),
            &Integer::from(3),
            &Integer::from(4),
            &Integer::from(5),
        );
        assert_eq!(result_inf.numerator(), Some(&Integer::from(1))); // 2/4 = 1/2
        assert_eq!(result_inf.denominator(), Some(&Integer::from(2)));
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
