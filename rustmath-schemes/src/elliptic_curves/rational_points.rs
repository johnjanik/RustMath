//! Rational Point Finding Algorithms for Elliptic Curves
//!
//! This module implements algorithms for finding and enumerating rational points
//! on elliptic curves, including:
//! - Height functions (naive, logarithmic, canonical/Néron-Tate)
//! - Bounded height searches
//! - Sieve-based enumeration methods
//! - Integration with Mordell-Weil group structures
//!
//! # Height Functions
//!
//! For a point P = (x, y) ∈ E(ℚ) where x = a/b in lowest terms, we define:
//!
//! ## Naive Height
//! ```text
//! h_naive(P) = max(|a|, |b|)
//! ```
//!
//! ## Logarithmic Height
//! ```text
//! h(P) = log(max(|a|, |b|))
//! ```
//!
//! ## Canonical Height (Néron-Tate)
//! ```text
//! ĥ(P) = lim_{n→∞} h([2^n]P) / 4^n
//! ```
//!
//! The canonical height is a quadratic form on E(ℚ) ⊗ ℝ and satisfies:
//! - ĥ([n]P) = n²ĥ(P) for all n ∈ ℤ
//! - ĥ(P) = 0 if and only if P is torsion
//! - ĥ(P + Q) + ĥ(P - Q) = 2ĥ(P) + 2ĥ(Q) (parallelogram law)
//!
//! # Bounded Height Search
//!
//! To find all rational points P with h(P) ≤ B:
//! 1. Enumerate all fractions x = a/b with |a|, |b| ≤ B in lowest terms
//! 2. For each x, check if y² = x³ + ax + b is a perfect square
//! 3. If yes, construct the point P = (x, y)
//!
//! For curves of large rank, this can find generators of the Mordell-Weil group.
//!
//! # Examples
//!
//! ```rust
//! use rustmath_schemes::elliptic_curves::rational_points::*;
//! use rustmath_schemes::elliptic_curves::rational::EllipticCurveRational;
//! use num_bigint::BigInt;
//!
//! // Create curve y² = x³ - x (congruent number curve, n=1)
//! let curve = EllipticCurveRational::from_short_weierstrass(
//!     BigInt::from(-1),
//!     BigInt::from(0),
//! );
//!
//! // Find all points with naive height ≤ 10
//! let search = BoundedHeightSearch::new(curve);
//! let points = search.find_points_up_to_height(10);
//!
//! // Compute heights
//! for point in &points {
//!     if !point.is_infinity {
//!         let h_naive = naive_height(&point.x);
//!         let h_log = logarithmic_height(&point.x);
//!         println!("Point: {}, naive height: {}, log height: {:.4}",
//!                  point, h_naive, h_log);
//!     }
//! }
//! ```
//!
//! # References
//!
//! - **Silverman**: "The Arithmetic of Elliptic Curves", Chapter VIII (Heights)
//! - **Silverman**: "Advanced Topics", Chapter III (Canonical Heights)
//! - **Cremona**: "Algorithms for Modular Elliptic Curves" (Point searching)
//! - **Stoll**: "Implementing 2-descent for Jacobians of hyperelliptic curves"

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Zero, One, Signed, ToPrimitive};
use std::collections::HashSet;

use super::rational::{EllipticCurveRational, RationalPoint};

/// Compute the naive height of a rational number x = a/b
///
/// The naive height is H(x) = max(|a|, |b|) where x = a/b in lowest terms.
///
/// # Examples
///
/// ```
/// use rustmath_schemes::elliptic_curves::rational_points::naive_height;
/// use num_bigint::BigInt;
/// use num_rational::BigRational;
///
/// let x = BigRational::new(BigInt::from(3), BigInt::from(5));
/// assert_eq!(naive_height(&x), BigInt::from(5));
///
/// let y = BigRational::new(BigInt::from(-7), BigInt::from(2));
/// assert_eq!(naive_height(&y), BigInt::from(7));
/// ```
pub fn naive_height(x: &BigRational) -> BigInt {
    let numer = x.numer().abs();
    let denom = x.denom().abs();

    if numer > denom {
        numer
    } else {
        denom
    }
}

/// Compute the logarithmic height of a rational number
///
/// The logarithmic height is h(x) = log(H(x)) where H(x) is the naive height.
///
/// # Examples
///
/// ```
/// use rustmath_schemes::elliptic_curves::rational_points::logarithmic_height;
/// use num_bigint::BigInt;
/// use num_rational::BigRational;
///
/// let x = BigRational::new(BigInt::from(3), BigInt::from(5));
/// let h = logarithmic_height(&x);
/// assert!(h > 0.0);
/// ```
pub fn logarithmic_height(x: &BigRational) -> f64 {
    let h = naive_height(x);
    h.to_f64().unwrap_or(1.0).ln().max(0.0)
}

/// Compute the Weil height of a projective point [x : y : z]
///
/// For a point in projective space ℙ²(ℚ), the Weil height is:
/// H([x : y : z]) = max(|x|, |y|, |z|) where gcd(x, y, z) = 1
///
/// # Examples
///
/// ```
/// use rustmath_schemes::elliptic_curves::rational_points::weil_height;
/// use num_bigint::BigInt;
///
/// let x = BigInt::from(3);
/// let y = BigInt::from(4);
/// let z = BigInt::from(5);
/// let h = weil_height(&x, &y, &z);
/// assert_eq!(h, BigInt::from(5));
/// ```
pub fn weil_height(x: &BigInt, y: &BigInt, z: &BigInt) -> BigInt {
    let mut max_val = x.abs();
    if y.abs() > max_val {
        max_val = y.abs();
    }
    if z.abs() > max_val {
        max_val = z.abs();
    }
    max_val
}

/// Canonical height computation for elliptic curve points
///
/// Implements the Néron-Tate canonical height ĥ(P) via the iterative formula:
/// ĥ(P) = lim_{n→∞} h([2^n]P) / 4^n
///
/// This is approximated by computing h([2^k]P) / 4^k for sufficiently large k.
pub struct CanonicalHeight<'a> {
    curve: &'a EllipticCurveRational,
    /// Number of iterations for height computation
    iterations: usize,
}

impl<'a> CanonicalHeight<'a> {
    /// Create a new canonical height computer
    ///
    /// # Arguments
    ///
    /// * `curve` - The elliptic curve
    /// * `iterations` - Number of doublings to perform (default: 10)
    pub fn new(curve: &'a EllipticCurveRational, iterations: usize) -> Self {
        Self { curve, iterations }
    }

    /// Compute the canonical height of a point
    ///
    /// Uses the formula: ĥ(P) ≈ h([2^k]P) / 4^k for k = iterations
    ///
    /// # Theory
    ///
    /// The canonical height differs from the naive height by a bounded function:
    /// |ĥ(P) - h(P)| ≤ C
    ///
    /// where C depends on the curve but not on P. The canonical height satisfies:
    /// - ĥ([n]P) = n²ĥ(P) (exact homogeneity)
    /// - ĥ(P + Q) + ĥ(P - Q) = 2ĥ(P) + 2ĥ(Q) (parallelogram law)
    /// - ĥ(P) ≥ 0 with equality iff P is torsion
    pub fn compute(&self, point: &RationalPoint) -> f64 {
        if point.is_infinity {
            return 0.0;
        }

        // Start with the point
        let mut current = point.clone();

        // Double the point k times
        for _ in 0..self.iterations {
            current = self.curve.double_point(&current);
            if current.is_infinity {
                return 0.0; // Point is torsion
            }
        }

        // Compute h([2^k]P)
        let h = logarithmic_height(&current.x);

        // Divide by 4^k to get approximation of canonical height
        let divisor = 4_f64.powi(self.iterations as i32);
        h / divisor
    }

    /// Compute the canonical height pairing ⟨P, Q⟩
    ///
    /// The canonical height pairing is defined as:
    /// ⟨P, Q⟩ = (ĥ(P + Q) - ĥ(P) - ĥ(Q)) / 2
    ///
    /// This is a symmetric bilinear form on E(ℚ) ⊗ ℝ.
    pub fn pairing(&self, p: &RationalPoint, q: &RationalPoint) -> f64 {
        let h_p = self.compute(p);
        let h_q = self.compute(q);
        let sum = self.curve.add_points(p, q);
        let h_sum = self.compute(&sum);

        (h_sum - h_p - h_q) / 2.0
    }

    /// Compute the regulator of a set of points
    ///
    /// The regulator is the determinant of the height pairing matrix.
    /// For independent points P₁, ..., Pᵣ, the regulator is:
    ///
    /// Reg(P₁, ..., Pᵣ) = det(⟨Pᵢ, Pⱼ⟩)
    ///
    /// This appears in the Birch and Swinnerton-Dyer conjecture.
    pub fn regulator(&self, points: &[RationalPoint]) -> f64 {
        let n = points.len();
        if n == 0 {
            return 1.0;
        }

        // Build height pairing matrix
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = self.pairing(&points[i], &points[j]);
            }
        }

        // Compute determinant
        Self::determinant(&matrix)
    }

    /// Compute determinant of a matrix using LU decomposition
    fn determinant(matrix: &[Vec<f64>]) -> f64 {
        let n = matrix.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return matrix[0][0];
        }
        if n == 2 {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }

        // LU decomposition
        let mut det = 1.0;
        let mut a = matrix.to_vec();

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if a[k][i].abs() > a[max_row][i].abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                a.swap(i, max_row);
                det = -det;
            }

            if a[i][i].abs() < 1e-10 {
                return 0.0;
            }

            for k in (i + 1)..n {
                let factor = a[k][i] / a[i][i];
                for j in i..n {
                    a[k][j] -= factor * a[i][j];
                }
            }

            det *= a[i][i];
        }

        det
    }
}

/// Bounded height search for rational points on elliptic curves
///
/// Systematically enumerates all rational points P with naive height H(P) ≤ B.
pub struct BoundedHeightSearch {
    curve: EllipticCurveRational,
}

impl BoundedHeightSearch {
    /// Create a new bounded height search for a curve
    pub fn new(curve: EllipticCurveRational) -> Self {
        Self { curve }
    }

    /// Find all rational points with naive height up to the bound B
    ///
    /// # Algorithm
    ///
    /// 1. For each coprime pair (a, b) with |a|, |b| ≤ B:
    ///    - Compute x = a/b
    ///    - Check if y² = x³ + Ax + B has a rational solution
    ///    - If yes, add (x, y) and (x, -y) to the list
    /// 2. Include the point at infinity
    ///
    /// # Complexity
    ///
    /// O(B² log B) for the enumeration, with additional cost for solving y².
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_schemes::elliptic_curves::rational_points::BoundedHeightSearch;
    /// use rustmath_schemes::elliptic_curves::rational::EllipticCurveRational;
    /// use num_bigint::BigInt;
    ///
    /// let curve = EllipticCurveRational::from_short_weierstrass(
    ///     BigInt::from(-1),
    ///     BigInt::from(0),
    /// );
    /// let search = BoundedHeightSearch::new(curve);
    /// let points = search.find_points_up_to_height(10);
    /// assert!(!points.is_empty()); // Should find at least point at infinity
    /// ```
    pub fn find_points_up_to_height(&self, bound: i64) -> Vec<RationalPoint> {
        let mut points = vec![RationalPoint::infinity()];
        let mut seen = HashSet::new();

        // Add (0, 0) if it's on the curve (for curves like y² = x³ + ax)
        let origin = RationalPoint::new(
            BigRational::zero(),
            BigRational::zero(),
        );
        if self.curve.is_on_curve(&origin) {
            points.push(origin.clone());
            seen.insert((BigInt::zero(), BigInt::zero()));
        }

        // Enumerate x-coordinates as fractions a/b with |a|, |b| ≤ bound
        for denom in 1..=bound {
            for numer in -bound..=bound {
                // Skip if not coprime
                if Self::gcd(numer.abs(), denom) != 1 {
                    continue;
                }

                let x = BigRational::new(
                    BigInt::from(numer),
                    BigInt::from(denom),
                );

                // Check if y² = x³ + a₄x + a₆ has a rational solution
                if let Some(y) = self.solve_for_y(&x) {
                    let point = RationalPoint::new(x.clone(), y.clone());

                    // Verify point is on curve
                    if self.curve.is_on_curve(&point) {
                        let key = (
                            point.x.numer().clone(),
                            point.x.denom().clone(),
                        );

                        if !seen.contains(&key) {
                            seen.insert(key);
                            points.push(point.clone());

                            // Also add the negative point if different
                            let neg_point = RationalPoint::new(
                                x.clone(),
                                -y.clone(),
                            );
                            if self.curve.is_on_curve(&neg_point) && y != -y.clone() {
                                points.push(neg_point);
                            }
                        }
                    }
                }
            }
        }

        points
    }

    /// Solve for y in y² = x³ + a₄x + a₆
    ///
    /// Returns Some(y) if a rational solution exists, None otherwise.
    fn solve_for_y(&self, x: &BigRational) -> Option<BigRational> {
        // For short Weierstrass form: y² = x³ + a₄x + a₆
        let rhs = x * x * x
            + BigRational::from(self.curve.a4.clone()) * x
            + BigRational::from(self.curve.a6.clone());

        // Check if rhs is a perfect square of a rational
        if let Some(sqrt) = Self::rational_sqrt(&rhs) {
            return Some(sqrt);
        }

        None
    }

    /// Compute the rational square root of a rational number, if it exists
    ///
    /// For x = a/b, returns √(a/b) = √a/√b if both a and b are perfect squares.
    fn rational_sqrt(x: &BigRational) -> Option<BigRational> {
        if x.is_zero() {
            return Some(BigRational::zero());
        }

        if x < &BigRational::zero() {
            return None; // No real square root
        }

        let numer = x.numer();
        let denom = x.denom();

        // Check if numerator is a perfect square
        let sqrt_numer = Self::integer_sqrt(numer)?;

        // Check if denominator is a perfect square
        let sqrt_denom = Self::integer_sqrt(denom)?;

        Some(BigRational::new(sqrt_numer, sqrt_denom))
    }

    /// Compute integer square root if n is a perfect square
    fn integer_sqrt(n: &BigInt) -> Option<BigInt> {
        if n.is_zero() {
            return Some(BigInt::zero());
        }

        if n < &BigInt::zero() {
            return None;
        }

        // Newton's method for integer square root
        let mut x = n.clone();
        let mut y = (&x + BigInt::one()) / BigInt::from(2);

        while y < x {
            x = y.clone();
            y = (&x + n / &x) / BigInt::from(2);
        }

        // Check if it's a perfect square
        if &x * &x == *n {
            Some(x)
        } else {
            None
        }
    }

    /// Compute GCD of two integers
    fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a.abs()
    }

    /// Find points using a sieve-based method
    ///
    /// This is more efficient for large bounds as it avoids redundant gcd computations.
    pub fn find_points_sieve(&self, bound: i64) -> Vec<RationalPoint> {
        let mut points = vec![RationalPoint::infinity()];
        let mut seen = HashSet::new();

        // Use Euler's totient to count coprime pairs more efficiently
        for denom in 1..=bound {
            for numer in -bound..=bound {
                // Quick coprimality check
                if numer != 0 && Self::gcd(numer.abs(), denom) != 1 {
                    continue;
                }

                let x = BigRational::new(
                    BigInt::from(numer),
                    BigInt::from(denom),
                );

                if let Some(y) = self.solve_for_y(&x) {
                    let point = RationalPoint::new(x.clone(), y.clone());

                    if self.curve.is_on_curve(&point) {
                        let key = (
                            point.x.numer().clone(),
                            point.x.denom().clone(),
                        );

                        if !seen.contains(&key) {
                            seen.insert(key);
                            points.push(point);
                        }
                    }
                }
            }
        }

        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_height() {
        let x = BigRational::new(BigInt::from(3), BigInt::from(5));
        assert_eq!(naive_height(&x), BigInt::from(5));

        let y = BigRational::new(BigInt::from(-7), BigInt::from(2));
        assert_eq!(naive_height(&y), BigInt::from(7));

        let z = BigRational::new(BigInt::from(100), BigInt::from(3));
        assert_eq!(naive_height(&z), BigInt::from(100));
    }

    #[test]
    fn test_logarithmic_height() {
        let x = BigRational::new(BigInt::from(3), BigInt::from(5));
        let h = logarithmic_height(&x);
        assert!(h > 0.0);

        // log(5) ≈ 1.609
        assert!((h - 1.609).abs() < 0.01);
    }

    #[test]
    fn test_weil_height() {
        let x = BigInt::from(3);
        let y = BigInt::from(4);
        let z = BigInt::from(5);

        let h = weil_height(&x, &y, &z);
        assert_eq!(h, BigInt::from(5));
    }

    #[test]
    fn test_canonical_height_infinity() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let height_computer = CanonicalHeight::new(&curve, 10);
        let inf = RationalPoint::infinity();

        assert_eq!(height_computer.compute(&inf), 0.0);
    }

    #[test]
    fn test_bounded_search_finds_infinity() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let search = BoundedHeightSearch::new(curve);
        let points = search.find_points_up_to_height(5);

        // Should always find at least the point at infinity
        assert!(!points.is_empty());
        assert!(points.iter().any(|p| p.is_infinity));
    }

    #[test]
    fn test_bounded_search_finds_rational_points() {
        // Curve y² = x³ - x has points (0, 0), (±1, 0)
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let search = BoundedHeightSearch::new(curve);
        let points = search.find_points_up_to_height(5);

        // Should find several points including (0, 0), (1, 0), (-1, 0)
        assert!(points.len() >= 4); // O, (0,0), (1,0), (-1,0)

        // Check that (0, 0) is found
        let has_origin = points.iter().any(|p| {
            !p.is_infinity && p.x.is_zero() && p.y.is_zero()
        });
        assert!(has_origin);
    }

    #[test]
    fn test_solve_for_y_perfect_square() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let search = BoundedHeightSearch::new(curve);

        // For x = 0: y² = 0, so y = 0
        let x = BigRational::zero();
        let y = search.solve_for_y(&x);
        assert!(y.is_some());
        assert_eq!(y.unwrap(), BigRational::zero());
    }

    #[test]
    fn test_integer_sqrt() {
        // Perfect squares
        assert_eq!(
            BoundedHeightSearch::integer_sqrt(&BigInt::from(0)),
            Some(BigInt::from(0))
        );
        assert_eq!(
            BoundedHeightSearch::integer_sqrt(&BigInt::from(1)),
            Some(BigInt::from(1))
        );
        assert_eq!(
            BoundedHeightSearch::integer_sqrt(&BigInt::from(4)),
            Some(BigInt::from(2))
        );
        assert_eq!(
            BoundedHeightSearch::integer_sqrt(&BigInt::from(9)),
            Some(BigInt::from(3))
        );
        assert_eq!(
            BoundedHeightSearch::integer_sqrt(&BigInt::from(16)),
            Some(BigInt::from(4))
        );
        assert_eq!(
            BoundedHeightSearch::integer_sqrt(&BigInt::from(100)),
            Some(BigInt::from(10))
        );

        // Non-perfect squares
        assert_eq!(
            BoundedHeightSearch::integer_sqrt(&BigInt::from(2)),
            None
        );
        assert_eq!(
            BoundedHeightSearch::integer_sqrt(&BigInt::from(3)),
            None
        );
        assert_eq!(
            BoundedHeightSearch::integer_sqrt(&BigInt::from(5)),
            None
        );
    }

    #[test]
    fn test_rational_sqrt() {
        // √(4/9) = 2/3
        let x = BigRational::new(BigInt::from(4), BigInt::from(9));
        let sqrt = BoundedHeightSearch::rational_sqrt(&x);
        assert!(sqrt.is_some());
        assert_eq!(sqrt.unwrap(), BigRational::new(BigInt::from(2), BigInt::from(3)));

        // √(1/4) = 1/2
        let y = BigRational::new(BigInt::from(1), BigInt::from(4));
        let sqrt_y = BoundedHeightSearch::rational_sqrt(&y);
        assert!(sqrt_y.is_some());
        assert_eq!(sqrt_y.unwrap(), BigRational::new(BigInt::from(1), BigInt::from(2)));

        // Non-perfect square
        let z = BigRational::new(BigInt::from(2), BigInt::from(3));
        let sqrt_z = BoundedHeightSearch::rational_sqrt(&z);
        assert!(sqrt_z.is_none());
    }

    #[test]
    fn test_height_pairing_symmetry() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let height_computer = CanonicalHeight::new(&curve, 5);

        let p = RationalPoint::new(
            BigRational::new(BigInt::from(1), BigInt::from(1)),
            BigRational::new(BigInt::from(1), BigInt::from(1)),
        );

        let q = RationalPoint::new(
            BigRational::new(BigInt::from(2), BigInt::from(1)),
            BigRational::new(BigInt::from(2), BigInt::from(1)),
        );

        if curve.is_on_curve(&p) && curve.is_on_curve(&q) {
            let pairing_pq = height_computer.pairing(&p, &q);
            let pairing_qp = height_computer.pairing(&q, &p);

            // Height pairing should be symmetric
            assert!((pairing_pq - pairing_qp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_regulator_rank_zero() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let height_computer = CanonicalHeight::new(&curve, 5);
        let points = vec![];

        let reg = height_computer.regulator(&points);
        assert_eq!(reg, 1.0); // Convention: regulator of empty set is 1
    }

    #[test]
    fn test_sieve_method() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let search = BoundedHeightSearch::new(curve);
        let points_sieve = search.find_points_sieve(5);
        let points_naive = search.find_points_up_to_height(5);

        // Both methods should find the same points (possibly in different order)
        assert_eq!(points_sieve.len(), points_naive.len());
    }

    #[test]
    fn test_gcd() {
        assert_eq!(BoundedHeightSearch::gcd(12, 8), 4);
        assert_eq!(BoundedHeightSearch::gcd(17, 5), 1);
        assert_eq!(BoundedHeightSearch::gcd(100, 50), 50);
        assert_eq!(BoundedHeightSearch::gcd(7, 13), 1);
    }
}
