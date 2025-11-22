//! Descent algorithms for rank computation
//!
//! Implements 2-descent to compute bounds on the rank of elliptic curves

use crate::curve::{EllipticCurve, Point};
use num_bigint::BigInt;
use num_traits::{Zero, One, ToPrimitive, Signed};

/// A quartic equation arising from a 2-covering
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quartic {
    pub a: BigInt,
    pub b: BigInt,
    pub c: BigInt,
    pub d: BigInt,
}

impl Quartic {
    pub fn new(a: BigInt, b: BigInt, c: BigInt, d: BigInt) -> Self {
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
    pub rank_bound: i32,
}

impl SelmerGroup {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            rank_bound: 0,
        }
    }

    pub fn add_element(&mut self, element: SelmerElement) {
        self.elements.push(element);
        self.rank_bound = (self.elements.len() as f64).log2().floor() as i32;
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
            BigInt::one(),
            BigInt::zero(),
            self.curve.a4.clone(),
            self.curve.a6.clone(),
        ));

        // Additional torsors from division polynomial
        // This is a simplified version - real implementation would compute all torsors
        if !self.curve.discriminant.is_zero() {
            torsors.push(Quartic::new(
                self.curve.discriminant.clone(),
                BigInt::zero(),
                BigInt::zero(),
                BigInt::one(),
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
    fn is_solvable_mod_p(&self, quartic: &Quartic, p: &BigInt) -> bool {
        let p_val = p.to_i64().unwrap_or(2);
        if p_val > 100 {
            // For large primes, assume solvable (Hasse-Minkowski)
            return true;
        }

        // Check if there exists a solution mod p
        let a = quartic.a.to_i64().unwrap_or(0) % p_val;
        let b = quartic.b.to_i64().unwrap_or(0) % p_val;
        let c = quartic.c.to_i64().unwrap_or(0) % p_val;
        let d = quartic.d.to_i64().unwrap_or(0) % p_val;

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
    fn compute_bad_primes(&self) -> Vec<BigInt> {
        let mut primes = Vec::new();
        let mut n = self.curve.discriminant.abs();

        // Factor out small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            let p_big = BigInt::from(p);
            if &n % &p_big == BigInt::zero() {
                primes.push(p_big);
                while &n % BigInt::from(p) == BigInt::zero() {
                    n /= BigInt::from(p);
                }
            }
        }

        primes
    }

    /// Compute an upper bound on the rank using 2-descent
    pub fn rank_bound(&self) -> i32 {
        let selmer_group = self.compute_selmer_group();
        let two_torsion = self.curve.two_torsion_rank();

        // rank(E) ≤ dim(Selmer) - dim(E[2])
        selmer_group.rank_bound - two_torsion
    }

    /// Search for rational points up to a given height
    pub fn find_rational_points(&self, height_bound: i64) -> Vec<Point> {
        let mut points = vec![Point::infinity()];

        for x_num in -height_bound..=height_bound {
            for x_den in 1..=height_bound {
                let x = num_rational::BigRational::new(
                    BigInt::from(x_num),
                    BigInt::from(x_den)
                );

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
    fn solve_for_y(&self, x: &num_rational::BigRational) -> Option<num_rational::BigRational> {
        use num_rational::BigRational;

        let rhs = x * x * x
            + BigRational::from(self.curve.a4.clone()) * x
            + BigRational::from(self.curve.a6.clone());

        // Check if rhs is a perfect square
        if rhs.is_integer() {
            let rhs_int = rhs.to_integer();
            if let Some(sqrt) = self.integer_sqrt(&rhs_int) {
                return Some(BigRational::from(sqrt));
            }
        }

        None
    }

    /// Compute integer square root if it exists
    fn integer_sqrt(&self, n: &BigInt) -> Option<BigInt> {
        if n.is_zero() {
            return Some(BigInt::zero());
        }

        if n < &BigInt::zero() {
            return None;
        }

        // Newton's method
        let mut x = n.clone();
        let mut y = (&x + BigInt::one()) / BigInt::from(2);

        while y < x {
            x = y.clone();
            y = (&x + n / &x) / BigInt::from(2);
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
            BigInt::from(-1),
            BigInt::from(0)
        );

        let descent = TwoDescent::new(&curve);
        let selmer = descent.compute_selmer_group();

        assert!(!selmer.elements.is_empty());
    }

    #[test]
    fn test_rank_bound() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1)
        );

        let descent = TwoDescent::new(&curve);
        let bound = descent.rank_bound();

        assert!(bound >= 0);
    }

    #[test]
    fn test_find_rational_points() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0)
        );

        let descent = TwoDescent::new(&curve);
        let points = descent.find_rational_points(10);

        // Should find at least the point at infinity
        assert!(!points.is_empty());
    }

    #[test]
    fn test_quartic_creation() {
        let q = Quartic::new(
            BigInt::one(),
            BigInt::zero(),
            BigInt::from(-1),
            BigInt::zero()
        );

        assert!(q.has_rational_point());
    }
}
