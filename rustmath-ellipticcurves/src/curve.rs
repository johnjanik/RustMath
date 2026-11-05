//! Core elliptic curve implementation
//!
//! Provides elliptic curves in generalized Weierstrass form:
//! y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
//!
//! Also provides short Weierstrass form: y² = x³ + ax + b
//!
//! # Canonicalization note (B3)
//!
//! This is the over-Q implementation, now backed by
//! `rustmath_integers::Integer` / `rustmath_rationals::Rational` (Phase-2
//! num-*-to-core normalization). It still overlaps with
//! `rustmath_schemes::elliptic_curves::rational::EllipticCurveRational`
//! (also over Q); merging the two over-Q implementations remains DEFERRED
//! until that one is normalized as well. The generic-field curve lives at
//! `rustmath_ellipticcurves::generic::EllipticCurve` and is unaffected.

use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;

/// An elliptic curve in generalized Weierstrass form
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EllipticCurve {
    pub a1: Integer,
    pub a2: Integer,
    pub a3: Integer,
    pub a4: Integer,
    pub a6: Integer,
    pub discriminant: Integer,
    pub conductor: Option<Integer>,
}

/// A point on an elliptic curve
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Point {
    pub x: Rational,
    pub y: Rational,
    pub infinity: bool,
}

impl EllipticCurve {
    /// Create a new elliptic curve from Weierstrass coefficients
    /// y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
    pub fn new(a1: Integer, a2: Integer, a3: Integer, a4: Integer, a6: Integer) -> Self {
        // Compute b-invariants
        let b2 = &a1 * &a1 + Integer::from(4) * a2.clone();
        let b4 = Integer::from(2) * a4.clone() + &a1 * &a3;
        let b6 = &a3 * &a3 + Integer::from(4) * a6.clone();
        let b8 = a1.clone() * a1.clone() * a6.clone() + Integer::from(4) * a2.clone() * a6.clone()
            - a1.clone() * a3.clone() * a4.clone()
            + a2.clone() * a3.clone() * a3.clone()
            - a4.clone() * a4.clone();

        // Compute discriminant: Δ = -b₂²b₈ - 8b₄³ - 27b₆² + 9b₂b₄b₆
        let discriminant = -(b2.clone() * b2.clone() * b8)
            - Integer::from(8) * b4.clone() * b4.clone() * b4.clone()
            - Integer::from(27) * b6.clone() * b6.clone()
            + Integer::from(9) * b2 * b4 * b6;

        Self {
            a1,
            a2,
            a3,
            a4,
            a6,
            discriminant,
            conductor: None,
        }
    }

    /// Create an elliptic curve in short Weierstrass form: y² = x³ + ax + b
    pub fn from_short_weierstrass(a: Integer, b: Integer) -> Self {
        Self::new(Integer::zero(), Integer::zero(), Integer::zero(), a, b)
    }

    /// Check if the curve is singular (discriminant is zero)
    pub fn is_singular(&self) -> bool {
        self.discriminant.is_zero()
    }

    /// Get the j-invariant of the curve
    pub fn j_invariant(&self) -> Option<Rational> {
        if self.is_singular() {
            return None;
        }

        let b2 = &self.a1 * &self.a1 + Integer::from(4) * self.a2.clone();
        let b4 = Integer::from(2) * self.a4.clone() + &self.a1 * &self.a3;
        let _b6 = &self.a3 * &self.a3 + Integer::from(4) * self.a6.clone();

        let c4 = &b2 * &b2 - Integer::from(24) * b4;
        let numerator = c4.pow(3);

        Some(
            Rational::new(numerator, self.discriminant.clone())
                .expect("non-singular curve has nonzero discriminant"),
        )
    }

    /// Add two points on the curve (general Weierstrass form).
    ///
    /// For the chord through P ≠ ±Q with slope λ and intercept ν
    /// (y = λx + ν), the sum is
    /// x₃ = λ² + a₁λ − a₂ − x₁ − x₂,  y₃ = −(λ + a₁)x₃ − ν − a₃.
    /// Both inputs are assumed to lie on the curve.
    pub fn add_points(&self, p: &Point, q: &Point) -> Point {
        if p.infinity {
            return q.clone();
        }
        if q.infinity {
            return p.clone();
        }

        if p.x == q.x {
            // Same x-coordinate: either negatives of each other or equal
            // (two points on the curve with equal x are equal or negatives).
            let neg_y = self.negate_y(&p.x, &p.y);
            if q.y == neg_y {
                return Point::infinity();
            }
            return self.double_point(p);
        }

        // λ = (y₂ - y₁) / (x₂ - x₁), ν = y₁ - λx₁
        let lambda = (&q.y - &p.y) / (&q.x - &p.x);
        let nu = p.y.clone() - lambda.clone() * p.x.clone();

        let a1 = Rational::from_integer(self.a1.clone());
        let a2 = Rational::from_integer(self.a2.clone());
        let a3 = Rational::from_integer(self.a3.clone());

        // x₃ = λ² + a₁λ − a₂ − x₁ − x₂
        let x3 = &lambda * &lambda + a1.clone() * lambda.clone() - a2 - p.x.clone() - q.x.clone();

        // y₃ = −(λ + a₁)x₃ − ν − a₃
        let y3 = -((lambda + a1) * x3.clone()) - nu - a3;

        Point {
            x: x3,
            y: y3,
            infinity: false,
        }
    }

    /// Double a point on the curve (general Weierstrass form).
    ///
    /// The tangent slope is λ = (3x² + 2a₂x + a₄ − a₁y) / (2y + a₁x + a₃);
    /// a vanishing denominator means P is 2-torsion and [2]P = O.
    pub fn double_point(&self, p: &Point) -> Point {
        if p.infinity {
            return p.clone();
        }

        let a1 = Rational::from_integer(self.a1.clone());
        let a2 = Rational::from_integer(self.a2.clone());
        let a3 = Rational::from_integer(self.a3.clone());
        let a4 = Rational::from_integer(self.a4.clone());
        let two = Rational::from_i64(2);
        let three = Rational::from_i64(3);

        let numerator =
            three * p.x.clone() * p.x.clone() + two.clone() * a2.clone() * p.x.clone() + a4
                - a1.clone() * p.y.clone();
        let denominator = two.clone() * p.y.clone() + a1.clone() * p.x.clone() + a3.clone();

        if denominator.is_zero() {
            return Point::infinity();
        }

        let lambda = numerator / denominator;
        let nu = p.y.clone() - lambda.clone() * p.x.clone();

        // x₃ = λ² + a₁λ − a₂ − 2x
        let x3 = &lambda * &lambda + a1.clone() * lambda.clone() - a2 - two * p.x.clone();

        // y₃ = −(λ + a₁)x₃ − ν − a₃
        let y3 = -((lambda + a1) * x3.clone()) - nu - a3;

        Point {
            x: x3,
            y: y3,
            infinity: false,
        }
    }

    /// Scalar multiplication: compute [n]P
    pub fn scalar_mul(&self, n: &Integer, p: &Point) -> Point {
        if n.is_zero() || p.infinity {
            return Point::infinity();
        }

        if n.signum() < 0 {
            let neg_p = self.negate_point(p);
            return self.scalar_mul(&-n, &neg_p);
        }

        // Binary method (double-and-add)
        let mut result = Point::infinity();
        let mut base = p.clone();
        let mut k = n.clone();
        let two = Integer::from(2);

        while !k.is_zero() {
            if (&k % &two).is_one() {
                result = self.add_points(&result, &base);
            }
            base = self.double_point(&base);
            k = k / two.clone();
        }

        result
    }

    /// Negate a point on the curve
    pub fn negate_point(&self, p: &Point) -> Point {
        if p.infinity {
            return p.clone();
        }

        let neg_y = self.negate_y(&p.x, &p.y);
        Point {
            x: p.x.clone(),
            y: neg_y,
            infinity: false,
        }
    }

    /// Compute the negation of y-coordinate for a given x
    /// For short Weierstrass, this is simply -y
    /// For general form: -(y + a₁x + a₃)
    fn negate_y(&self, x: &Rational, y: &Rational) -> Rational {
        let a1_term = Rational::from_integer(self.a1.clone()) * x.clone();
        let a3_term = Rational::from_integer(self.a3.clone());
        -(y.clone() + a1_term + a3_term)
    }

    /// Check if a point is on the curve (general Weierstrass form):
    /// y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆.
    pub fn is_on_curve(&self, p: &Point) -> bool {
        if p.infinity {
            return true;
        }

        let a1 = Rational::from_integer(self.a1.clone());
        let a2 = Rational::from_integer(self.a2.clone());
        let a3 = Rational::from_integer(self.a3.clone());
        let a4 = Rational::from_integer(self.a4.clone());
        let a6 = Rational::from_integer(self.a6.clone());

        let lhs = &p.y * &p.y + a1 * p.x.clone() * p.y.clone() + a3 * p.y.clone();
        let rhs = p.x.clone() * p.x.clone() * p.x.clone()
            + a2 * p.x.clone() * p.x.clone()
            + a4 * p.x.clone()
            + a6;

        lhs == rhs
    }

    /// The 2-rank of E(Q)[2]: r ∈ {0, 1, 2} with E(Q)[2] ≅ (Z/2)^r.
    ///
    /// The x-coordinates of 2-torsion points are the roots of
    /// 4x³ + b₂x² + 2b₄x + b₆. Under X = 36x + 3b₂ these correspond
    /// exactly to the rational (hence integral) roots of the monic cubic
    /// X³ − 27c₄X − 54c₆.
    pub fn two_torsion_rank(&self) -> i32 {
        let (c4, c6) = self.c_invariants();
        let a = Integer::from(-27) * c4;
        let b = Integer::from(-54) * c6;
        let roots = crate::torsion::integer_cubic_roots(&a, &b);
        match roots.len() {
            0 => 0,
            1 => 1,
            3 => 2,
            n => unreachable!(
                "cubic with {} rational roots (nonsingular curve has distinct roots)",
                n
            ),
        }
    }

    /// Check if a prime is a bad prime (divides the discriminant)
    pub fn is_bad_prime(&self, p: &Integer) -> bool {
        (&self.discriminant % p).is_zero()
    }

    /// The trace of Frobenius a_p at ANY prime p, computed exactly on the
    /// general Weierstrass model with `Integer` arithmetic throughout.
    ///
    /// * **Good reduction** (including primes dividing the discriminant of
    ///   the *given* model at which the curve nevertheless has good
    ///   reduction — Tate's algorithm supplies the p-minimal model first):
    ///   a_p = p + 1 − #E(F_p).
    /// * **Bad reduction**: a_p = p − #E_ns(F_p) where E_ns is the smooth
    ///   locus of the reduced minimal model. The three cases (Silverman AEC
    ///   App. C §16 / Silverman ATAEC IV.9):
    ///   - additive: E_ns(F_p) ≅ (F_p, +), so #E_ns = p and **a_p = 0**;
    ///   - split multiplicative: E_ns(F_p) ≅ F_p^*, #E_ns = p − 1, **a_p = +1**;
    ///   - non-split multiplicative: E_ns(F_p) is the norm-one torus of
    ///     F_{p²}/F_p, #E_ns = p + 1, **a_p = −1**.
    ///
    ///   Split vs non-split is decided exactly by Tate's algorithm (step 3):
    ///   after translating the node to the origin the tangent-cone quadratic
    ///   is T² + a₁T − a₂; the reduction is split iff its two roots (the two
    ///   tangent directions at the node) lie in F_p — for odd p iff its
    ///   discriminant b₂ is a nonzero square mod p, and for p = 2 (where a₁
    ///   is odd here) iff a₂ ≡ 0 mod 2. See [`crate::tate`], whose
    ///   split/non-split branch is PARI `ellap`-validated in its tests.
    ///
    /// # Point counting (good p)
    ///
    /// For odd p, complete the square: (2y + a₁x + a₃)² = 4x³ + b₂x² +
    /// 2b₄x + b₆ =: g(x), a bijection on y-fibres since 2 is invertible; the
    /// number of points over x is 1 + χ(g(x)) with χ the quadratic character
    /// of F_p (χ(0) = 0), so #E(F_p) = p + 1 + Σ_x χ(g(x)). χ is evaluated
    /// by Euler's criterion g^((p−1)/2) mod p, exactly. For p = 2 the four
    /// affine candidates are enumerated directly.
    ///
    /// Cost: O(p) modular exponentiations (naive counting, not Schoof) —
    /// fine for the small primes used in L-series work here.
    ///
    /// # Panics
    ///
    /// Panics if p is not prime, if the curve is singular, or if the Hasse
    /// bound a_p² ≤ 4p fails at a good prime (an internal bug detector,
    /// never an answer).
    pub fn compute_a_p(&self, p: &Integer) -> i64 {
        assert!(
            rustmath_integers::prime::is_prime(p),
            "compute_a_p: p = {} is not prime",
            p
        );
        assert!(!self.is_singular(), "compute_a_p: curve is singular");
        if self.is_bad_prime(p) {
            let ld = self.local_data(p);
            return match ld.reduction {
                crate::tate::ReductionType::Good => {
                    // The given model is non-minimal at p; count on the
                    // p-minimal model, whose reduction is honest.
                    assert!(
                        !ld.minimal_model.is_bad_prime(p),
                        "p-minimal model still bad at a good prime: bug"
                    );
                    ld.minimal_model.trace_of_frobenius_good(p)
                }
                crate::tate::ReductionType::SplitMultiplicative => 1,
                crate::tate::ReductionType::NonsplitMultiplicative => -1,
                crate::tate::ReductionType::Additive => 0,
            };
        }
        self.trace_of_frobenius_good(p)
    }

    /// a_p = p + 1 − #E(F_p) for a prime of good reduction of THIS model.
    fn trace_of_frobenius_good(&self, p: &Integer) -> i64 {
        let p_i = <Integer as NumericConversion>::to_i64(p)
            .expect("compute_a_p: p too large for naive point counting");
        let count = self.count_points_mod_p(p);
        let a_p = p_i + 1 - count;
        // Hasse bound |a_p| <= 2 sqrt(p): a bug detector on the count.
        assert!(
            (a_p as i128) * (a_p as i128) <= 4 * (p_i as i128),
            "Hasse bound violated at p = {}: a_p = {} (bug in point counting)",
            p,
            a_p
        );
        a_p
    }

    /// #E(F_p) for a prime of good reduction of this model, exactly:
    /// p + 1 + Σ_x χ(4x³ + b₂x² + 2b₄x + b₆) for odd p (see
    /// [`Self::compute_a_p`] for the derivation); direct enumeration for
    /// p = 2. O(p) modular exponentiations.
    fn count_points_mod_p(&self, p: &Integer) -> i64 {
        let two = Integer::from(2);
        if *p == two {
            // Enumerate the 4 affine candidates over F_2.
            let mut count = 1i64; // point at infinity
            let red = |v: &Integer| -> i64 {
                if (v % &two).is_zero() {
                    0
                } else {
                    1
                }
            };
            let (a1, a2, a3, a4, a6) = (
                red(&self.a1),
                red(&self.a2),
                red(&self.a3),
                red(&self.a4),
                red(&self.a6),
            );
            for x in 0..2i64 {
                for y in 0..2i64 {
                    let lhs = y * y + a1 * x * y + a3 * y;
                    let rhs = x * x * x + a2 * x * x + a4 * x + a6;
                    if (lhs - rhs).rem_euclid(2) == 0 {
                        count += 1;
                    }
                }
            }
            return count;
        }

        // Odd p: quadratic-character sum over g(x) = 4x^3 + b2 x^2 + 2 b4 x + b6.
        let b2 = (&self.a1 * &self.a1 + Integer::from(4) * self.a2.clone()).modulo(p);
        let b4 = (Integer::from(2) * self.a4.clone() + &self.a1 * &self.a3).modulo(p);
        let b6 = (&self.a3 * &self.a3 + Integer::from(4) * self.a6.clone()).modulo(p);
        let four = Integer::from(4);
        let two_b4 = (two.clone() * b4).modulo(p);
        let euler_exp = (p.clone() - Integer::one()) / two;
        let p_minus_1 = p.clone() - Integer::one();

        let p_i = <Integer as NumericConversion>::to_i64(p)
            .expect("count_points_mod_p: p too large for naive point counting");
        let mut chi_sum = 0i64;
        let mut x = Integer::zero();
        for _ in 0..p_i {
            // Horner, reduced mod p at each step.
            let g = ((((&four * &x).modulo(p) + b2.clone()) * x.clone()).modulo(p)
                + two_b4.clone())
            .modulo(p)
                * x.clone();
            let g = (g.modulo(p) + b6.clone()).modulo(p);
            if !g.is_zero() {
                let r = g
                    .mod_pow(&euler_exp, p)
                    .expect("mod_pow with prime modulus");
                if r.is_one() {
                    chi_sum += 1;
                } else {
                    assert!(
                        r == p_minus_1,
                        "Euler criterion returned a value other than ±1: bug"
                    );
                    chi_sum -= 1;
                }
            }
            x = x + Integer::one();
        }
        p_i + 1 + chi_sum
    }
}

impl Point {
    /// Create a new affine point
    pub fn new(x: Rational, y: Rational) -> Self {
        Self {
            x,
            y,
            infinity: false,
        }
    }

    /// Create the point at infinity
    pub fn infinity() -> Self {
        Self {
            x: Rational::zero(),
            y: Rational::zero(),
            infinity: true,
        }
    }

    /// Create a point from integer coordinates
    pub fn from_integers(x: i64, y: i64) -> Self {
        Self {
            x: Rational::from_i64(x),
            y: Rational::from_i64(y),
            infinity: false,
        }
    }
}

impl fmt::Display for EllipticCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.a1.is_zero() && self.a2.is_zero() && self.a3.is_zero() {
            write!(f, "y² = x³ + {}x + {}", self.a4, self.a6)
        } else {
            write!(
                f,
                "y² + {}xy + {}y = x³ + {}x² + {}x + {}",
                self.a1, self.a3, self.a2, self.a4, self.a6
            )
        }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.infinity {
            write!(f, "O (point at infinity)")
        } else {
            write!(f, "({}, {})", self.x, self.y)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curve_creation() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(1));
        assert!(!curve.is_singular());
    }

    #[test]
    fn test_point_on_curve() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        // Point (0, 0) should be on y² = x³ - x
        let p = Point::new(Rational::zero(), Rational::zero());
        assert!(curve.is_on_curve(&p));

        // Point (1, 0) should be on y² = x³ - x
        let q = Point::new(Rational::one(), Rational::zero());
        assert!(curve.is_on_curve(&q));
    }

    #[test]
    fn test_point_addition() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        let p = Point::new(Rational::zero(), Rational::zero());
        let q = Point::infinity();

        let r = curve.add_points(&p, &q);
        assert_eq!(r, p);
    }

    #[test]
    fn test_point_doubling() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(2), Integer::from(3));

        // Point (-1, 0) is on y² = x³ + 2x + 3
        let p = Point::new(Rational::from_i64(-1), Rational::from_i64(0));

        assert!(curve.is_on_curve(&p));
        let doubled = curve.double_point(&p);
        // Doubling a point where y=0 gives infinity
        assert!(doubled.infinity);
    }

    #[test]
    fn test_scalar_multiplication() {
        // Use curve y² = x³ - x for simplicity
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        // Point (0, 0) is on the curve
        let p = Point::new(Rational::from_i64(0), Rational::from_i64(0));

        assert!(curve.is_on_curve(&p));
        let result = curve.scalar_mul(&Integer::from(2), &p);
        // [2]P for a point of order 2 is infinity
        assert!(result.infinity || curve.is_on_curve(&result));
    }

    #[test]
    fn test_j_invariant() {
        // For y² = x³ + x (curve with CM by Gaussian integers)
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(1), Integer::from(0));

        let j = curve.j_invariant();
        assert!(j.is_some());
        // j-invariant should be 1728 for this curve
    }

    /// a_p gates for general Weierstrass models, INCLUDING bad primes.
    /// Every expected value below was derived independently in Python by
    /// direct point counting on the reduced curve (smooth points only at
    /// bad primes: a_p = p − #E_ns(F_p)) before this test was written; the
    /// same tables are cross-checked a third way (modular eigensystems via
    /// Eichler–Shimura) in tests/modular_crosscheck.rs.
    #[test]
    fn test_a_p_tables_point_counted() {
        let primes: [i64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        // (label, [a1,a2,a3,a4,a6], [a_p for the primes above])
        let table: [(&str, [i64; 5], [i64; 12]); 7] = [
            (
                "11a1",
                [0, -1, 1, -10, -20],
                [-2, -1, 1, -2, 1, 4, -2, 0, -1, 0, 7, 3],
            ),
            (
                "14a1",
                [1, 0, 1, 4, -6],
                [-1, -2, 0, 1, 0, -4, 6, 2, 0, -6, -4, 2],
            ),
            (
                "15a1",
                [1, 1, 1, -10, -10],
                [-1, -1, 1, 0, -4, -2, 2, 4, 0, -2, 0, -10],
            ),
            (
                "37a1",
                [0, 0, 1, -1, 0],
                [-2, -3, -2, -1, -5, -2, 0, 0, 2, 6, -4, -1],
            ),
            (
                "37b1",
                [0, 1, 1, -23, -50],
                [0, 1, 0, -1, 3, -4, 6, 2, 6, -6, -4, 1],
            ),
            (
                "49a1",
                [1, -1, 0, -2, -1],
                [1, 0, 0, 0, 4, 0, 0, 0, 8, 2, 0, -6],
            ),
            (
                "389a1",
                [0, 1, 1, -2, 0],
                [-2, -2, -3, -5, -4, -3, -6, 5, -4, -6, 4, -8],
            ),
        ];
        for (label, a, expected) in &table {
            let e = EllipticCurve::new(
                Integer::from(a[0]),
                Integer::from(a[1]),
                Integer::from(a[2]),
                Integer::from(a[3]),
                Integer::from(a[4]),
            );
            for (p, want) in primes.iter().zip(expected.iter()) {
                assert_eq!(
                    e.compute_a_p(&Integer::from(*p)),
                    *want,
                    "a_{} of {}",
                    p,
                    label
                );
            }
        }
    }

    /// a_p at bad primes follows the reduction type: 11a1 is split
    /// multiplicative at 11 (a_11 = +1), 37a1 non-split at 37 (a_37 = −1),
    /// 49a1 additive at 7 (a_7 = 0). All PARI `ellap`-consistent and
    /// re-derived by smooth-point counting in Python.
    #[test]
    fn test_a_p_bad_prime_conventions() {
        let e11 = EllipticCurve::new(
            Integer::from(0),
            Integer::from(-1),
            Integer::from(1),
            Integer::from(-10),
            Integer::from(-20),
        );
        assert_eq!(e11.compute_a_p(&Integer::from(11)), 1);
        let e37 = EllipticCurve::new(
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
            Integer::from(-1),
            Integer::from(0),
        );
        assert_eq!(e37.compute_a_p(&Integer::from(37)), -1);
        let e49 = EllipticCurve::new(
            Integer::from(1),
            Integer::from(-1),
            Integer::from(0),
            Integer::from(-2),
            Integer::from(-1),
        );
        assert_eq!(e49.compute_a_p(&Integer::from(7)), 0);
    }

    /// A model non-minimal at p with good reduction after minimalization
    /// must yield the good-reduction a_p, not a bad-prime value: 11a1
    /// scaled by u = 2 ([0,-4,8,-160,-1280]) has a_2 = a_2(11a1) = −2.
    #[test]
    fn test_a_p_nonminimal_model_good_reduction() {
        let e = EllipticCurve::new(
            Integer::from(0),
            Integer::from(-4),
            Integer::from(8),
            Integer::from(-160),
            Integer::from(-1280),
        );
        assert_eq!(e.compute_a_p(&Integer::from(2)), -2);
        // and the untouched primes agree with 11a1 as well
        assert_eq!(e.compute_a_p(&Integer::from(3)), -1);
        assert_eq!(e.compute_a_p(&Integer::from(11)), 1);
    }

    #[test]
    #[should_panic(expected = "not prime")]
    fn test_a_p_rejects_composite() {
        let e = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(1));
        let _ = e.compute_a_p(&Integer::from(6));
    }
}
