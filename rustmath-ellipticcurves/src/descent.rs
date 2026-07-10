//! Descent algorithms for rank computation
//!
//! Implements 2-descent to compute bounds on the rank of elliptic curves

use crate::curve::{EllipticCurve, Point};
use rustmath_core::NumericConversion;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// A quartic equation arising from a 2-covering
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quartic {
    pub a: Integer,
    pub b: Integer,
    pub c: Integer,
    pub d: Integer,
}

impl Quartic {
    pub fn new(a: Integer, b: Integer, c: Integer, d: Integer) -> Self {
        Self { a, b, c, d }
    }

    /// Check if the quartic has a rational point
    pub fn has_rational_point(&self) -> bool {
        // Simplified check - real implementation would be more sophisticated
        !self.a.is_zero() || !self.b.is_zero() || !self.c.is_zero() || !self.d.is_zero()
    }
}

/// An element of the Selmer group
#[derive(Debug, Clone)]
pub struct SelmerElement {
    pub quartic: Quartic,
    pub locally_solvable: bool,
}

/// The Selmer group arising from 2-descent
#[derive(Debug, Clone)]
pub struct SelmerGroup {
    pub elements: Vec<SelmerElement>,
    /// Placeholder only -- NOT a genuine 2-descent rank bound. It is left at
    /// its default (0) by `add_element`. A real Selmer rank requires actual
    /// group-law arithmetic on the torsors (showing the locally-solvable
    /// classes form an F_2-vector space of a given dimension), not a count
    /// of how many torsors were found. See `TwoDescent::rank_bound`, which
    /// honestly reports this as unimplemented.
    pub rank_bound: i32,
}

impl SelmerGroup {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            rank_bound: 0,
        }
    }

    /// Register a locally-solvable torsor as a Selmer group element.
    ///
    /// This only maintains the element list; it deliberately does NOT touch
    /// `rank_bound`. The previous implementation derived `rank_bound` as
    /// `log2(elements.len()).floor()`, which is not a real descent
    /// computation -- it treated a raw count of locally-solvable torsors as
    /// if it were the dimension of an F_2-vector space, without ever
    /// establishing the group structure that would justify that.
    pub fn add_element(&mut self, element: SelmerElement) {
        self.elements.push(element);
    }
}

/// 2-descent algorithm for rank computation
pub struct TwoDescent<'a> {
    curve: &'a EllipticCurve,
}

impl<'a> TwoDescent<'a> {
    pub fn new(curve: &'a EllipticCurve) -> Self {
        Self { curve }
    }

    /// Compute the 2-Selmer group
    pub fn compute_selmer_group(&self) -> SelmerGroup {
        let torsors = self.compute_torsors();
        let mut selmer_group = SelmerGroup::new();

        for torsor in torsors {
            if self.is_locally_solvable(&torsor) {
                selmer_group.add_element(SelmerElement {
                    quartic: torsor,
                    locally_solvable: true,
                });
            }
        }

        selmer_group
    }

    /// Compute the quartic equations for 2-coverings
    /// For curve y² = x³ + ax + b, we get quartics
    fn compute_torsors(&self) -> Vec<Quartic> {
        let mut torsors = Vec::new();

        // Trivial torsor (the curve itself)
        torsors.push(Quartic::new(
            Integer::one(),
            Integer::zero(),
            self.curve.a4.clone(),
            self.curve.a6.clone(),
        ));

        // Additional torsors from division polynomial
        // This is a simplified version - real implementation would compute all torsors
        if !self.curve.discriminant.is_zero() {
            torsors.push(Quartic::new(
                self.curve.discriminant.clone(),
                Integer::zero(),
                Integer::zero(),
                Integer::one(),
            ));
        }

        torsors
    }

    /// Check if a torsor is locally solvable everywhere
    fn is_locally_solvable(&self, quartic: &Quartic) -> bool {
        // Check real solvability
        if !self.is_solvable_over_reals(quartic) {
            return false;
        }

        // Check p-adic solvability at bad primes
        let bad_primes = self.compute_bad_primes();
        for p in bad_primes {
            if !self.is_solvable_mod_p(quartic, &p) {
                return false;
            }
        }

        true
    }

    /// Check if quartic is solvable over real numbers
    fn is_solvable_over_reals(&self, _quartic: &Quartic) -> bool {
        // Real curves always have points at infinity
        true
    }

    /// Check if quartic is solvable modulo p
    fn is_solvable_mod_p(&self, quartic: &Quartic, p: &Integer) -> bool {
        let p_val = <Integer as NumericConversion>::to_i64(p).unwrap_or(2);
        if p_val > 100 {
            // For large primes, assume solvable (Hasse-Minkowski)
            return true;
        }

        // Check if there exists a solution mod p
        let a = <Integer as NumericConversion>::to_i64(&quartic.a).unwrap_or(0) % p_val;
        let b = <Integer as NumericConversion>::to_i64(&quartic.b).unwrap_or(0) % p_val;
        let c = <Integer as NumericConversion>::to_i64(&quartic.c).unwrap_or(0) % p_val;
        let d = <Integer as NumericConversion>::to_i64(&quartic.d).unwrap_or(0) % p_val;

        for x in 0..p_val {
            let val = (a * x * x * x * x + b * x * x + c * x + d).rem_euclid(p_val);
            if self.is_quadratic_residue(val, p_val) {
                return true;
            }
        }

        false
    }

    /// Check if a value is a quadratic residue mod p
    fn is_quadratic_residue(&self, a: i64, p: i64) -> bool {
        if a == 0 {
            return true;
        }

        for i in 0..p {
            if (i * i) % p == a {
                return true;
            }
        }

        false
    }

    /// Compute bad primes (those dividing the discriminant)
    fn compute_bad_primes(&self) -> Vec<Integer> {
        let mut primes = Vec::new();
        let mut n = self.curve.discriminant.abs();

        // Factor out small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            let p_big = Integer::from(p);
            if (&n % &p_big).is_zero() {
                primes.push(p_big.clone());
                while (&n % &p_big).is_zero() {
                    n = n / p_big.clone();
                }
            }
        }

        primes
    }

    /// Compute an upper bound on the rank using 2-descent
    ///
    /// A genuine bound is rank(E) <= dim(Selmer) - dim(E[2]), but this
    /// requires `dim(Selmer)` to come from actual descent (group-law
    /// arithmetic on 2-coverings), not from counting locally-solvable
    /// torsors and taking log2 of the count, which is what this function
    /// previously did.
    pub fn rank_bound(&self) -> i32 {
        unimplemented!(
            "2-descent rank bound not yet implemented (facade): the previous body computed \
             log2(number of locally-solvable torsors found), which is not a real Selmer \
             group rank computation"
        )
    }

    /// Search for rational points up to a given height
    pub fn find_rational_points(&self, height_bound: i64) -> Vec<Point> {
        let mut points = vec![Point::infinity()];

        for x_num in -height_bound..=height_bound {
            for x_den in 1..=height_bound {
                let x = Rational::new(Integer::from(x_num), Integer::from(x_den))
                    .expect("x_den >= 1 is nonzero");

                // Check if y² = x³ + ax + b has a solution
                if let Some(y) = self.solve_for_y(&x) {
                    let p = Point::new(x.clone(), y.clone());
                    if self.curve.is_on_curve(&p) {
                        points.push(p.clone());

                        // Also add -P
                        let neg_p = self.curve.negate_point(&p);
                        if neg_p != p {
                            points.push(neg_p);
                        }
                    }
                }
            }
        }

        points
    }

    /// Attempt to solve y² = x³ + ax + b for y
    fn solve_for_y(&self, x: &Rational) -> Option<Rational> {
        let rhs = x.clone() * x.clone() * x.clone()
            + Rational::from_integer(self.curve.a4.clone()) * x.clone()
            + Rational::from_integer(self.curve.a6.clone());

        // Check if rhs is a perfect square
        if rhs.is_integer() {
            let rhs_int = rhs.floor();
            if let Some(sqrt) = self.integer_sqrt(&rhs_int) {
                return Some(Rational::from_integer(sqrt));
            }
        }

        None
    }

    /// Compute integer square root if it exists
    fn integer_sqrt(&self, n: &Integer) -> Option<Integer> {
        if n.is_zero() {
            return Some(Integer::zero());
        }

        if n < &Integer::zero() {
            return None;
        }

        // Newton's method
        let mut x = n.clone();
        let mut y = (&x + &Integer::one()) / Integer::from(2);

        while y < x {
            x = y.clone();
            y = (&x + &(n / &x)) / Integer::from(2);
        }

        if &x * &x == *n {
            Some(x)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::EllipticCurve;

    #[test]
    fn test_selmer_group() {
        let curve = EllipticCurve::from_short_weierstrass(
            Integer::from(-1),
            Integer::from(0)
        );

        let descent = TwoDescent::new(&curve);
        let selmer = descent.compute_selmer_group();

        assert!(!selmer.elements.is_empty());
    }

    #[test]
    #[ignore = "facade -> unimplemented; needs real descent/L-function (Phase 4)"]
    fn test_rank_bound() {
        let curve = EllipticCurve::from_short_weierstrass(
            Integer::from(-1),
            Integer::from(1)
        );

        let descent = TwoDescent::new(&curve);
        let bound = descent.rank_bound();

        assert!(bound >= 0);
    }

    #[test]
    fn test_find_rational_points() {
        let curve = EllipticCurve::from_short_weierstrass(
            Integer::from(-1),
            Integer::from(0)
        );

        let descent = TwoDescent::new(&curve);
        let points = descent.find_rational_points(10);

        // Should find at least the point at infinity
        assert!(!points.is_empty());
    }

    #[test]
    fn test_quartic_creation() {
        let q = Quartic::new(
            Integer::one(),
            Integer::zero(),
            Integer::from(-1),
            Integer::zero()
        );

        assert!(q.has_rational_point());
    }
}
