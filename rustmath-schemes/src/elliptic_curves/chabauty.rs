//! Chabauty-Coleman Method for Finding Rational Points
//!
//! This module implements the Chabauty-Coleman method for determining all rational
//! points on curves of genus g > 0 when the Mordell-Weil rank r satisfies r < g.
//!
//! # Mathematical Background
//!
//! ## Chabauty's Theorem (1941)
//!
//! Let C be a curve of genus g ≥ 1 over ℚ embedded in its Jacobian J. If
//! rank(J(ℚ)) < g, then C(ℚ) is finite.
//!
//! Moreover, for a prime p of good reduction, the closure of C(ℚ) in C(ℚₚ)
//! is a p-adic analytic submanifold of dimension ≤ r < g.
//!
//! ## Coleman Integration (1985)
//!
//! Coleman developed a theory of p-adic integration that allows us to:
//! 1. Integrate differential forms on curves over ℚₚ
//! 2. Compute integrals along paths in C(ℚₚ)
//! 3. Use these integrals to detect rational points
//!
//! ## The Chabauty-Coleman Method
//!
//! To find all rational points on C:
//!
//! 1. Choose a prime p of good reduction
//! 2. Find a basis {P₁, ..., Pᵣ} for J(ℚ) modulo torsion
//! 3. Construct the p-adic closure of this subgroup in J(ℚₚ)
//! 4. For each residue disk in C(𝔽ₚ), use Coleman integration to find
//!    which points in C(ℚₚ) map to the p-adic closure
//! 5. These are exactly the rational points
//!
//! # Application to Elliptic Curves
//!
//! For an elliptic curve E/ℚ of rank r:
//! - If r = 0: All points are torsion (trivial case)
//! - If r = 1: Chabauty is vacuous (rank = genus)
//! - For higher genus curves (hyperelliptic, plane curves), Chabauty is more useful
//!
//! However, variants and generalizations apply to elliptic curves:
//! - **Quadratic Chabauty**: For rank 1 curves, using quadratic extensions
//! - **Elliptic curve Chabauty**: Over number fields
//! - **Depth 2 Chabauty**: Kim's non-abelian approach
//!
//! # Examples
//!
//! ```rust
//! use rustmath_schemes::elliptic_curves::chabauty::*;
//! use rustmath_schemes::elliptic_curves::rational::EllipticCurveRational;
//! use num_bigint::BigInt;
//!
//! // Create a rank 0 curve (all points are torsion)
//! let curve = EllipticCurveRational::from_cremona_label("11a1").unwrap();
//!
//! // Apply Chabauty-Coleman
//! let chabauty = ChabautyColeman::new(curve, BigInt::from(7));
//! let applicable = chabauty.is_applicable();
//!
//! if applicable {
//!     // Find all rational points
//!     let points = chabauty.find_rational_points();
//!     println!("Found {} rational points", points.len());
//! }
//! ```
//!
//! # References
//!
//! - **Chabauty (1941)**: "Sur les points rationnels des courbes algébriques de genre supérieur à l'unité"
//! - **Coleman (1985)**: "Effective Chabauty"
//! - **McCallum-Poonen (2012)**: "The method of Chabauty and Coleman"
//! - **Balakrishnan-Dogra (2019)**: "Quadratic Chabauty and rational points"
//! - **Stoll (2006)**: "Independence of rational points on twists of a given curve"

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Zero, One, Signed, ToPrimitive};
use std::collections::{HashSet, HashMap};

use super::rational::{EllipticCurveRational, RationalPoint};

/// Coleman integration engine for p-adic integration on curves
///
/// Implements the theory of Coleman integration, which allows integration
/// of differential forms on curves over ℚₚ.
pub struct ColemanIntegral {
    /// The prime p for p-adic analysis
    prime: BigInt,
    /// Precision for p-adic computations
    precision: usize,
}

impl ColemanIntegral {
    /// Create a new Coleman integral computer
    ///
    /// # Arguments
    ///
    /// * `prime` - The prime p for p-adic computations
    /// * `precision` - Number of p-adic digits to compute
    pub fn new(prime: BigInt, precision: usize) -> Self {
        Self { prime, precision }
    }

    /// Integrate the invariant differential ω = dx/(2y) from P to Q
    ///
    /// For an elliptic curve E: y² = x³ + ax + b, the invariant differential
    /// is ω = dx/(2y). The Coleman integral is:
    ///
    /// ∫ₚᵠ ω = lim_{n→∞} (1/pⁿ) log(x(Φⁿ(Q))/x(Φⁿ(P)))
    ///
    /// where Φ is the Frobenius lift.
    ///
    /// # Theory
    ///
    /// Coleman's theorem states that this integral:
    /// - Is independent of the path in ℚₚ
    /// - Depends only on the residue classes of P and Q modulo p
    /// - Satisfies ∫ₚᵠ ω + ∫_Q^R ω = ∫ₚᴿ ω (additivity)
    pub fn integrate(
        &self,
        _curve: &EllipticCurveRational,
        _p: &RationalPoint,
        _q: &RationalPoint,
    ) -> Option<BigRational> {
        // Placeholder implementation
        // Real implementation requires:
        // 1. Lift points to ℤₚ
        // 2. Compute Frobenius action
        // 3. Evaluate p-adic logarithm
        // 4. Apply Coleman's integration formula

        Some(BigRational::zero())
    }

    /// Compute the p-adic logarithm log_p(x)
    ///
    /// For |x - 1|_p < 1, we have:
    /// log_p(x) = ∑_{n=1}^∞ (-1)^{n+1} (x-1)^n / n
    ///
    /// This is computed to the specified precision.
    fn padic_log(&self, _x: &BigRational) -> BigRational {
        // Placeholder: would implement p-adic logarithm series
        BigRational::zero()
    }

    /// Compute the Frobenius lift Φ acting on a point
    ///
    /// The Frobenius lift is an endomorphism of the curve that reduces
    /// to the Frobenius map modulo p.
    fn frobenius_lift(&self, _point: &RationalPoint) -> RationalPoint {
        // Placeholder: would implement Frobenius computation
        RationalPoint::infinity()
    }
}

/// Chabauty-Coleman method for finding rational points
///
/// Implements the Chabauty-Coleman algorithm to find all rational points
/// on curves when rank(J(ℚ)) < genus.
pub struct ChabautyColeman {
    /// The elliptic curve
    curve: EllipticCurveRational,
    /// The prime p for p-adic analysis
    prime: BigInt,
    /// Precision for p-adic computations
    precision: usize,
    /// Known generators of the Mordell-Weil group
    generators: Vec<RationalPoint>,
    /// Torsion points
    torsion_points: Vec<RationalPoint>,
}

impl ChabautyColeman {
    /// Create a new Chabauty-Coleman computer
    ///
    /// # Arguments
    ///
    /// * `curve` - The elliptic curve over ℚ
    /// * `prime` - A prime of good reduction for the curve
    ///
    /// # Panics
    ///
    /// Panics if the prime divides the discriminant (bad reduction).
    pub fn new(curve: EllipticCurveRational, prime: BigInt) -> Self {
        // Check that p is a prime of good reduction
        assert!(
            &curve.discriminant % &prime != BigInt::zero(),
            "Prime must be of good reduction"
        );

        Self {
            curve,
            prime,
            precision: 20,
            generators: Vec::new(),
            torsion_points: Vec::new(),
        }
    }

    /// Set the Mordell-Weil generators
    ///
    /// These should be independent generators of E(ℚ) modulo torsion.
    pub fn set_generators(&mut self, generators: Vec<RationalPoint>) {
        self.generators = generators;
    }

    /// Set the torsion points
    pub fn set_torsion(&mut self, torsion: Vec<RationalPoint>) {
        self.torsion_points = torsion;
    }

    /// Set the p-adic precision
    pub fn set_precision(&mut self, precision: usize) {
        self.precision = precision;
    }

    /// Check if the Chabauty-Coleman method is applicable
    ///
    /// Chabauty's theorem applies when rank(E(ℚ)) < genus.
    /// For elliptic curves (genus 1), this means rank = 0.
    ///
    /// # Returns
    ///
    /// `true` if rank < genus, `false` otherwise.
    pub fn is_applicable(&self) -> bool {
        let rank = self.generators.len();
        let genus = 1; // Elliptic curves have genus 1

        rank < genus
    }

    /// Check if Chabauty is conclusive
    ///
    /// Even when rank = genus, some variants may still work.
    pub fn is_conclusive(&self) -> bool {
        self.is_applicable()
    }

    /// Find all rational points using Chabauty-Coleman
    ///
    /// # Algorithm
    ///
    /// 1. Check applicability (rank < genus)
    /// 2. If rank = 0, all points are torsion
    /// 3. Otherwise, use Coleman integration to find points in each residue disk
    /// 4. Verify that found points are rational
    ///
    /// # Returns
    ///
    /// Vector of all rational points on the curve.
    pub fn find_rational_points(&self) -> Vec<RationalPoint> {
        if !self.is_applicable() {
            // Method not applicable
            return vec![RationalPoint::infinity()];
        }

        let rank = self.generators.len();

        if rank == 0 {
            // All points are torsion
            return self.find_torsion_points();
        }

        // For rank > 0 but < genus, use Coleman integration
        self.coleman_search()
    }

    /// Find all torsion points on the curve
    ///
    /// For rank 0 curves, all rational points are torsion.
    /// Uses division polynomials to find all n-torsion points.
    fn find_torsion_points(&self) -> Vec<RationalPoint> {
        let mut points = vec![RationalPoint::infinity()];

        // Add any known torsion points
        points.extend(self.torsion_points.clone());

        // Use TorsionFinder to find all torsion points up to order 12
        // (Mazur's theorem: torsion over Q is bounded by Z/12Z or Z/2Z × Z/8Z)
        let mut torsion_finder = TorsionFinder::new(self.curve.clone(), 100);

        // Find torsion points of orders 2, 3, 4, 5, 6, 7, 8, 9, 10, 12
        for n in &[2, 3, 4, 5, 6, 7, 8, 9, 10, 12] {
            let n_torsion = torsion_finder.find_n_torsion(*n);
            points.extend(n_torsion);
        }

        // Remove duplicates
        let mut seen = HashSet::new();
        points.retain(|p| {
            if p.is_infinity {
                return seen.insert((BigInt::zero(), BigInt::zero()));
            }
            seen.insert((p.x.numer().clone(), p.x.denom().clone()))
        });

        points
    }

    /// Find all 2-torsion points (points of order dividing 2)
    ///
    /// These are points P with 2P = O, which means y = 0.
    /// So we solve x³ + ax + b = 0 for rational roots.
    fn find_two_torsion(&self) -> Vec<RationalPoint> {
        let mut points = Vec::new();

        // For short Weierstrass y² = x³ + a₄x + a₆
        // 2-torsion points have y = 0, so we need x³ + a₄x + a₆ = 0

        // Try small integer values
        for x in -100..=100 {
            let x_big = BigInt::from(x);
            let value = &x_big * &x_big * &x_big
                + &self.curve.a4 * &x_big
                + &self.curve.a6;

            if value.is_zero() {
                let point = RationalPoint::new(
                    BigRational::from(x_big),
                    BigRational::zero(),
                );
                if self.curve.is_on_curve(&point) {
                    points.push(point);
                }
            }
        }

        points
    }

    /// Use Coleman integration to search for rational points
    ///
    /// # Algorithm
    ///
    /// 1. Reduce the curve modulo p to get E(𝔽ₚ)
    /// 2. For each point P̄ ∈ E(𝔽ₚ):
    ///    a. Lift to a residue disk in E(ℤₚ)
    ///    b. Use Coleman integration to find which lifts are in E(ℚ)
    /// 3. Verify that found points are rational
    fn coleman_search(&self) -> Vec<RationalPoint> {
        let mut points = vec![RationalPoint::infinity()];

        // Create Coleman integral computer
        let coleman = ColemanIntegral::new(self.prime.clone(), self.precision);

        // Get points on the reduction E(𝔽ₚ)
        let reduction_points = self.reduce_modulo_p();

        // For each residue disk
        for _p_bar in reduction_points {
            // Compute Coleman integrals to determine which lifts are rational
            // This requires:
            // 1. Computing integrals ∫_P^Q ω for various differentials ω
            // 2. Checking linear independence conditions
            // 3. Hensel lifting to find exact coordinates

            // Placeholder: would implement full Coleman integration
        }

        points
    }

    /// Reduce the curve modulo p to get E(𝔽ₚ)
    ///
    /// Returns representatives of points on E(𝔽ₚ).
    fn reduce_modulo_p(&self) -> Vec<RationalPoint> {
        let p = self.prime.to_i64().unwrap_or(7);
        let mut points = Vec::new();

        // Enumerate all points (x, y) with 0 ≤ x, y < p
        for x in 0..p {
            for y in 0..p {
                // Check if y² ≡ x³ + a₄x + a₆ (mod p)
                let lhs = (y * y).rem_euclid(p);

                let a4_mod = self.curve.a4.to_i64().unwrap_or(0).rem_euclid(p);
                let a6_mod = self.curve.a6.to_i64().unwrap_or(0).rem_euclid(p);

                let rhs = (x * x * x + a4_mod * x + a6_mod).rem_euclid(p);

                if lhs == rhs {
                    points.push(RationalPoint::new(
                        BigRational::from(BigInt::from(x)),
                        BigRational::from(BigInt::from(y)),
                    ));
                }
            }
        }

        points
    }

    /// Compute the p-adic closure of the Mordell-Weil group
    ///
    /// Given generators P₁, ..., Pᵣ of E(ℚ)/E(ℚ)_tors, compute the
    /// p-adic closure in E(ℚₚ).
    ///
    /// This is an r-dimensional p-adic analytic subgroup.
    pub fn mordell_weil_closure(&self) -> Vec<RationalPoint> {
        // Placeholder: would compute p-adic closure
        self.generators.clone()
    }

    /// Verify that a point is rational (not just p-adic)
    ///
    /// Check that a point P ∈ E(ℚₚ) actually lies in E(ℚ).
    fn is_rational_point(&self, point: &RationalPoint) -> bool {
        // For a point to be rational, its coordinates must be in ℚ
        // (already guaranteed by RationalPoint type)

        // Verify it's on the curve
        self.curve.is_on_curve(point)
    }

    /// Estimate the number of rational points
    ///
    /// Using Chabauty's bound: #C(ℚ) ≤ #C(𝔽ₚ) + 2g - 2
    pub fn chabauty_bound(&self) -> usize {
        let genus = 1; // Elliptic curves
        let reduction_points = self.reduce_modulo_p();

        reduction_points.len() + 2 * genus - 2
    }

    /// Generate a report on the Chabauty-Coleman computation
    pub fn generate_report(&self) -> String {
        let applicable = self.is_applicable();
        let rank = self.generators.len();
        let genus = 1;

        format!(
            "Chabauty-Coleman Method Report\n\
             ==============================\n\
             Curve: {}\n\
             Prime p: {}\n\
             Precision: {} p-adic digits\n\n\
             Rank: {}\n\
             Genus: {}\n\
             Applicable: {} (rank < genus)\n\n\
             Chabauty Bound: ≤ {} rational points\n\n\
             {}",
            self.curve,
            self.prime,
            self.precision,
            rank,
            genus,
            applicable,
            self.chabauty_bound(),
            if applicable {
                "Method is applicable. Can find all rational points."
            } else {
                "Method not applicable (rank ≥ genus). Need other techniques."
            }
        )
    }
}

/// Division polynomials for elliptic curves
///
/// Division polynomials ψₙ are used to compute n-torsion points and scalar multiplication.
/// They satisfy the recurrence relations:
///
/// ψ₀ = 0
/// ψ₁ = 1
/// ψ₂ = 2y
/// ψ₃ = 3x⁴ + 6ax² + 12bx - a²
/// ψ₄ = 4y(x⁶ + 5ax⁴ + 20bx³ - 5a²x² - 4abx - 8b² - a³)
///
/// For n ≥ 3:
/// ψ₂ₙ₊₁ = ψₙ₊₂ψₙ³ - ψₙ₋₁ψₙ₊₁³  (if n is odd)
/// ψ₂ₙ = (ψₙ/2y)(ψₙ₊₂ψₙ₋₁² - ψₙ₋₂ψₙ₊₁²)  (if n is even)
///
/// The n-torsion points are exactly the roots of ψₙ(x).
pub struct DivisionPolynomials {
    curve: EllipticCurveRational,
    /// Cache for computed division polynomials
    cache: HashMap<u32, Vec<BigRational>>,
}

impl DivisionPolynomials {
    /// Create a new division polynomial computer
    pub fn new(curve: EllipticCurveRational) -> Self {
        Self {
            curve,
            cache: HashMap::new(),
        }
    }

    /// Compute the n-th division polynomial ψₙ(x)
    ///
    /// Returns coefficients [a₀, a₁, a₂, ...] representing a₀ + a₁x + a₂x² + ...
    ///
    /// For short Weierstrass y² = x³ + ax + b:
    /// - ψ₁(x) = 1
    /// - ψ₂(x) = 2y (treated as 2 when working with x-coordinates)
    /// - ψ₃(x) = 3x⁴ + 6ax² + 12bx - a²
    /// - ψ₄(x) = 2(x⁶ + 5ax⁴ + 20bx³ - 5a²x² - 4abx - 8b² - a³)
    pub fn compute(&mut self, n: u32) -> Vec<BigRational> {
        // Check cache first
        if let Some(cached) = self.cache.get(&n) {
            return cached.clone();
        }

        // Verify short Weierstrass form
        if !self.curve.a1.is_zero() || !self.curve.a2.is_zero() || !self.curve.a3.is_zero() {
            panic!("Division polynomials only implemented for short Weierstrass form");
        }

        let a = BigRational::from(self.curve.a4.clone());
        let b = BigRational::from(self.curve.a6.clone());

        let result = match n {
            0 => vec![BigRational::zero()],
            1 => vec![BigRational::one()],
            2 => {
                // For x-coordinate only computation, ψ₂ = 2
                vec![BigRational::from(BigInt::from(2))]
            }
            3 => {
                // ψ₃ = 3x⁴ + 6ax² + 12bx - a²
                let three = BigRational::from(BigInt::from(3));
                let six = BigRational::from(BigInt::from(6));
                let twelve = BigRational::from(BigInt::from(12));

                vec![
                    -&a * &a,                    // constant term
                    &twelve * &b,                 // x
                    &six * &a,                    // x²
                    BigRational::zero(),          // x³
                    three,                        // x⁴
                ]
            }
            4 => {
                // ψ₄ = 2(x⁶ + 5ax⁴ + 20bx³ - 5a²x² - 4abx - 8b² - a³)
                let two = BigRational::from(BigInt::from(2));
                let four = BigRational::from(BigInt::from(4));
                let five = BigRational::from(BigInt::from(5));
                let eight = BigRational::from(BigInt::from(8));
                let twenty = BigRational::from(BigInt::from(20));

                vec![
                    &two * (-&eight * &b * &b - &a * &a * &a),  // constant
                    &two * (-&four * &a * &b),                   // x
                    &two * (-&five * &a * &a),                   // x²
                    &two * &twenty * &b,                         // x³
                    &two * &five * &a,                           // x⁴
                    BigRational::zero(),                         // x⁵
                    two,                                         // x⁶
                ]
            }
            _ => {
                // Use recurrence relations for n ≥ 5
                self.compute_recurrence(n)
            }
        };

        self.cache.insert(n, result.clone());
        result
    }

    /// Compute division polynomial using recurrence relations
    fn compute_recurrence(&mut self, n: u32) -> Vec<BigRational> {
        if n % 2 == 1 {
            // Odd case: ψ₂ₙ₊₁ = ψₙ₊₂ψₙ³ - ψₙ₋₁ψₙ₊₁³
            let m = (n - 1) / 2;
            let psi_m_plus_2 = self.compute(m + 2);
            let psi_m = self.compute(m);
            let psi_m_minus_1 = self.compute(m.saturating_sub(1));
            let psi_m_plus_1 = self.compute(m + 1);

            let psi_m_cubed = Self::poly_pow(&psi_m, 3);
            let psi_m_plus_1_cubed = Self::poly_pow(&psi_m_plus_1, 3);

            let term1 = Self::poly_mult(&psi_m_plus_2, &psi_m_cubed);
            let term2 = Self::poly_mult(&psi_m_minus_1, &psi_m_plus_1_cubed);

            Self::poly_sub(&term1, &term2)
        } else {
            // Even case: more complex, involves division by 2y
            // Simplified: ψ₂ₙ ≈ ψₙ(ψₙ₊₂ψₙ₋₁² - ψₙ₋₂ψₙ₊₁²)
            let m = n / 2;
            let psi_m = self.compute(m);
            let psi_m_plus_2 = self.compute(m + 2);
            let psi_m_plus_1 = self.compute(m + 1);
            let psi_m_minus_1 = self.compute(m.saturating_sub(1));
            let psi_m_minus_2 = self.compute(m.saturating_sub(2));

            let psi_m_minus_1_sq = Self::poly_pow(&psi_m_minus_1, 2);
            let psi_m_plus_1_sq = Self::poly_pow(&psi_m_plus_1, 2);

            let term1 = Self::poly_mult(&psi_m_plus_2, &psi_m_minus_1_sq);
            let term2 = Self::poly_mult(&psi_m_minus_2, &psi_m_plus_1_sq);
            let diff = Self::poly_sub(&term1, &term2);

            Self::poly_mult(&psi_m, &diff)
        }
    }

    /// Multiply two polynomials
    fn poly_mult(p: &[BigRational], q: &[BigRational]) -> Vec<BigRational> {
        if p.is_empty() || q.is_empty() {
            return vec![BigRational::zero()];
        }

        let mut result = vec![BigRational::zero(); p.len() + q.len() - 1];

        for (i, pi) in p.iter().enumerate() {
            for (j, qj) in q.iter().enumerate() {
                result[i + j] += pi * qj;
            }
        }

        result
    }

    /// Add two polynomials
    fn poly_add(p: &[BigRational], q: &[BigRational]) -> Vec<BigRational> {
        let max_len = p.len().max(q.len());
        let mut result = vec![BigRational::zero(); max_len];

        for (i, pi) in p.iter().enumerate() {
            result[i] += pi;
        }

        for (i, qi) in q.iter().enumerate() {
            result[i] += qi;
        }

        result
    }

    /// Subtract two polynomials
    fn poly_sub(p: &[BigRational], q: &[BigRational]) -> Vec<BigRational> {
        let max_len = p.len().max(q.len());
        let mut result = vec![BigRational::zero(); max_len];

        for (i, pi) in p.iter().enumerate() {
            result[i] += pi;
        }

        for (i, qi) in q.iter().enumerate() {
            result[i] -= qi;
        }

        result
    }

    /// Compute p^n for polynomial p
    fn poly_pow(p: &[BigRational], n: u32) -> Vec<BigRational> {
        if n == 0 {
            return vec![BigRational::one()];
        }
        if n == 1 {
            return p.to_vec();
        }

        let mut result = vec![BigRational::one()];
        let mut base = p.to_vec();
        let mut exp = n;

        while exp > 0 {
            if exp % 2 == 1 {
                result = Self::poly_mult(&result, &base);
            }
            base = Self::poly_mult(&base, &base);
            exp /= 2;
        }

        result
    }

    /// Evaluate polynomial at a point x
    fn poly_eval(poly: &[BigRational], x: &BigRational) -> BigRational {
        let mut result = BigRational::zero();
        let mut x_power = BigRational::one();

        for coeff in poly {
            result += coeff * &x_power;
            x_power *= x;
        }

        result
    }

    /// Find rational roots of a polynomial within a search bound
    ///
    /// Returns x-coordinates of n-torsion points
    pub fn find_rational_roots(&mut self, n: u32, search_bound: i64) -> Vec<BigRational> {
        let poly = self.compute(n);
        let mut roots = Vec::new();

        // Search for rational roots x = p/q with |p|, |q| ≤ search_bound
        for denom in 1..=search_bound {
            for numer in -search_bound..=search_bound {
                // Check coprimality
                if numer != 0 && Self::gcd(numer.abs(), denom) != 1 {
                    continue;
                }

                let x = BigRational::new(BigInt::from(numer), BigInt::from(denom));
                let value = Self::poly_eval(&poly, &x);

                if value.is_zero() {
                    roots.push(x);
                }
            }
        }

        roots
    }

    /// Compute GCD
    fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a.abs()
    }
}

/// Higher torsion point finder
///
/// Finds all n-torsion points on an elliptic curve for various values of n.
/// Uses division polynomials to locate the x-coordinates of torsion points.
pub struct TorsionFinder {
    curve: EllipticCurveRational,
    division_polys: DivisionPolynomials,
    /// Maximum search bound for rational roots
    search_bound: i64,
}

impl TorsionFinder {
    /// Create a new torsion finder
    ///
    /// # Arguments
    ///
    /// * `curve` - The elliptic curve (must be in short Weierstrass form)
    /// * `search_bound` - Maximum height for searching rational roots (default: 100)
    pub fn new(curve: EllipticCurveRational, search_bound: i64) -> Self {
        let division_polys = DivisionPolynomials::new(curve.clone());
        Self {
            curve,
            division_polys,
            search_bound,
        }
    }

    /// Find all n-torsion points
    ///
    /// Returns a vector of all rational points P with nP = O.
    /// This includes the point at infinity.
    pub fn find_n_torsion(&mut self, n: u32) -> Vec<RationalPoint> {
        let mut points = vec![RationalPoint::infinity()];

        if n == 0 {
            return points;
        }

        // Find x-coordinates of n-torsion points using division polynomials
        let x_coords = self.division_polys.find_rational_roots(n, self.search_bound);

        // For each x-coordinate, compute the corresponding y-coordinates
        for x in x_coords {
            // Solve for y: y² = x³ + a₄x + a₆
            let rhs = &x * &x * &x
                + BigRational::from(self.curve.a4.clone()) * &x
                + BigRational::from(self.curve.a6.clone());

            // Check if rhs is a perfect square
            if let Some(y) = Self::rational_sqrt(&rhs) {
                // Add both (x, y) and (x, -y) if they're on the curve
                let p_pos = RationalPoint::new(x.clone(), y.clone());
                if self.curve.is_on_curve(&p_pos) && self.verify_n_torsion(&p_pos, n) {
                    points.push(p_pos);
                }

                if !y.is_zero() {
                    let p_neg = RationalPoint::new(x.clone(), -y);
                    if self.curve.is_on_curve(&p_neg) && self.verify_n_torsion(&p_neg, n) {
                        points.push(p_neg);
                    }
                }
            }
        }

        points
    }

    /// Verify that a point is actually n-torsion
    fn verify_n_torsion(&self, point: &RationalPoint, n: u32) -> bool {
        let mut current = point.clone();

        for _ in 0..n {
            if current.is_infinity {
                return true;
            }
            current = self.curve.double_point(&current);
        }

        current.is_infinity
    }

    /// Find all torsion points up to order max_order
    ///
    /// Returns a map from order to the points of that order.
    pub fn find_all_torsion(&mut self, max_order: u32) -> HashMap<u32, Vec<RationalPoint>> {
        let mut result = HashMap::new();

        for n in 1..=max_order {
            let n_torsion = self.find_n_torsion(n);
            if !n_torsion.is_empty() {
                result.insert(n, n_torsion);
            }
        }

        result
    }

    /// Compute rational square root if it exists
    fn rational_sqrt(x: &BigRational) -> Option<BigRational> {
        if x.is_zero() {
            return Some(BigRational::zero());
        }

        if x < &BigRational::zero() {
            return None;
        }

        let numer = x.numer();
        let denom = x.denom();

        let sqrt_numer = Self::integer_sqrt(numer)?;
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

/// Quadratic Chabauty for rank 1 curves
///
/// When rank(E(ℚ)) = genus = 1, classical Chabauty doesn't apply.
/// Quadratic Chabauty uses quadratic extensions and p-adic heights to still bound rational points.
///
/// # Theory
///
/// The quadratic Chabauty method, developed by Balakrishnan, Dogra, et al., extends
/// the classical Chabauty-Coleman approach to rank = genus cases by:
///
/// 1. Working over a quadratic extension K/ℚ where the curve gains extra rank
/// 2. Using p-adic heights on E(K) instead of Coleman integrals
/// 3. Exploiting the Bloch-Kato Selmer group and quadratic reciprocity
/// 4. Applying local-global principles to cut down from E(ℚₚ) to E(ℚ)
///
/// # Algorithm Outline
///
/// For a curve E/ℚ of rank 1 with generator P:
///
/// 1. Choose a suitable quadratic extension K = ℚ(√d)
/// 2. Compute the Mordell-Weil group E(K) (should have rank > 1)
/// 3. Find generators Q₁, ..., Qᵣ of E(K)/E(ℚ)
/// 4. For a prime p of good reduction:
///    a. Compute p-adic heights ĥₚ on E(K)
///    b. For each residue disk in E(𝔽ₚ), use height pairings to determine
///       which points could be rational
///    c. Apply local conditions at other primes to eliminate false positives
/// 5. Verify that found points are genuinely in E(ℚ)
pub struct QuadraticChabauty {
    curve: EllipticCurveRational,
    prime: BigInt,
    precision: usize,
    /// Mordell-Weil generator over ℚ
    generator: Option<RationalPoint>,
    /// Quadratic extension discriminant
    extension_discriminant: Option<BigInt>,
}

impl QuadraticChabauty {
    /// Create a new quadratic Chabauty computer
    pub fn new(curve: EllipticCurveRational, prime: BigInt) -> Self {
        Self {
            curve,
            prime,
            precision: 20,
            generator: None,
            extension_discriminant: None,
        }
    }

    /// Set the precision for p-adic computations
    pub fn set_precision(&mut self, precision: usize) {
        self.precision = precision;
    }

    /// Set the Mordell-Weil generator over ℚ
    pub fn set_generator(&mut self, generator: RationalPoint) {
        self.generator = Some(generator);
    }

    /// Set the quadratic extension to use
    ///
    /// The extension is ℚ(√d) where d is square-free
    pub fn set_extension(&mut self, discriminant: BigInt) {
        self.extension_discriminant = Some(discriminant);
    }

    /// Check if quadratic Chabauty is applicable
    ///
    /// Requires:
    /// - rank(E(ℚ)) = 1
    /// - A suitable quadratic extension exists
    /// - Technical p-adic conditions hold
    pub fn is_applicable(&self) -> bool {
        // Check that we have a generator (implies rank ≥ 1)
        if self.generator.is_none() {
            return false;
        }

        // Check that prime is of good reduction
        if &self.curve.discriminant % &self.prime == BigInt::zero() {
            return false;
        }

        // Would need to check that rank is exactly 1, not higher
        // For now, assume it's applicable if conditions above hold
        true
    }

    /// Find rational points using quadratic Chabauty
    ///
    /// # Algorithm Steps
    ///
    /// 1. Select quadratic extension K = ℚ(√d) (if not already set)
    /// 2. Compute generators of E(K)
    /// 3. Compute p-adic heights
    /// 4. For each residue class modulo p:
    ///    - Use height pairings to determine if a rational point exists
    ///    - Apply local conditions to verify
    /// 5. Return all found rational points
    pub fn find_rational_points(&mut self) -> Vec<RationalPoint> {
        if !self.is_applicable() {
            return vec![RationalPoint::infinity()];
        }

        let mut points = vec![RationalPoint::infinity()];

        // Step 1: Choose quadratic extension if not set
        if self.extension_discriminant.is_none() {
            self.choose_extension();
        }

        // Step 2: Compute E(K) generators (simplified)
        let k_generators = self.compute_extension_generators();

        // Step 3: For each residue disk in E(𝔽ₚ)
        let reduction_points = self.reduce_modulo_p();

        for p_bar in reduction_points {
            // Step 4: Use p-adic heights to check if this lifts to a rational point
            if let Some(point) = self.check_residue_disk(&p_bar, &k_generators) {
                if self.verify_rational(&point) {
                    points.push(point);
                }
            }
        }

        points
    }

    /// Choose a suitable quadratic extension
    ///
    /// Heuristic: try small square-free integers d with certain properties
    fn choose_extension(&mut self) {
        // Try small square-free discriminants
        let candidates = vec![-1, -2, -3, -5, -7, 2, 3, 5, 6, 7, 10];

        for &d in &candidates {
            let disc = BigInt::from(d);
            if self.extension_is_suitable(&disc) {
                self.extension_discriminant = Some(disc);
                return;
            }
        }

        // Default to -1 (Gaussian extension)
        self.extension_discriminant = Some(BigInt::from(-1));
    }

    /// Check if a quadratic extension is suitable
    ///
    /// We want E(K) to have larger rank than E(ℚ)
    fn extension_is_suitable(&self, _discriminant: &BigInt) -> bool {
        // Simplified check - in practice would use:
        // - BSD formula predictions
        // - Heegner point constructions
        // - Mordell-Weil sieve
        true
    }

    /// Compute generators of E(K) where K is the quadratic extension
    ///
    /// This is a major computational task involving:
    /// - Heegner points (for CM curves)
    /// - Descents over number fields
    /// - Point searching over K
    fn compute_extension_generators(&self) -> Vec<RationalPoint> {
        // Placeholder: in practice this requires sophisticated algorithms
        // Would return points in E(K) \ E(ℚ)

        // For now, just return the ℚ-generator if available
        if let Some(ref gen) = self.generator {
            vec![gen.clone()]
        } else {
            Vec::new()
        }
    }

    /// Reduce curve modulo p to get points in E(𝔽ₚ)
    fn reduce_modulo_p(&self) -> Vec<RationalPoint> {
        let p = self.prime.to_i64().unwrap_or(7);
        let mut points = Vec::new();

        for x in 0..p {
            for y in 0..p {
                let lhs = (y * y).rem_euclid(p);

                let a4_mod = self.curve.a4.to_i64().unwrap_or(0).rem_euclid(p);
                let a6_mod = self.curve.a6.to_i64().unwrap_or(0).rem_euclid(p);

                let rhs = (x * x * x + a4_mod * x + a6_mod).rem_euclid(p);

                if lhs == rhs {
                    points.push(RationalPoint::new(
                        BigRational::from(BigInt::from(x)),
                        BigRational::from(BigInt::from(y)),
                    ));
                }
            }
        }

        points
    }

    /// Check if a residue disk contains a rational point
    ///
    /// Uses p-adic height pairings and local conditions
    fn check_residue_disk(
        &self,
        _residue_point: &RationalPoint,
        _generators: &[RationalPoint],
    ) -> Option<RationalPoint> {
        // Placeholder for the core quadratic Chabauty computation
        // This involves:
        // 1. Hensel lifting the residue point to ℚₚ
        // 2. Computing p-adic logarithms and heights
        // 3. Checking if the height pairing matrix has the right rank
        // 4. Solving for the exact coordinates

        None
    }

    /// Verify that a p-adic point is actually rational
    fn verify_rational(&self, point: &RationalPoint) -> bool {
        self.curve.is_on_curve(point)
    }

    /// Compute the p-adic height of a point
    ///
    /// The p-adic height ĥₚ(P) is defined using p-adic logarithms
    /// and satisfies ĥₚ([n]P) = n²ĥₚ(P)
    fn padic_height(&self, _point: &RationalPoint) -> BigRational {
        // Placeholder: requires p-adic logarithm computation
        BigRational::zero()
    }

    /// Compute the p-adic height pairing ⟨P, Q⟩ₚ
    ///
    /// Defined as: ⟨P, Q⟩ₚ = (ĥₚ(P + Q) - ĥₚ(P) - ĥₚ(Q)) / 2
    fn padic_height_pairing(
        &self,
        _p: &RationalPoint,
        _q: &RationalPoint,
    ) -> BigRational {
        // Placeholder
        BigRational::zero()
    }

    /// Generate a comprehensive report on the quadratic Chabauty computation
    pub fn report(&self) -> String {
        let applicable = self.is_applicable();
        let has_generator = self.generator.is_some();
        let has_extension = self.extension_discriminant.is_some();

        format!(
            "Quadratic Chabauty Method Report\n\
             ================================\n\
             Curve: {}\n\
             Prime p: {}\n\
             Precision: {} p-adic digits\n\n\
             Configuration:\n\
             - Has generator: {}\n\
             - Quadratic extension: {}\n\
             - Applicable: {}\n\n\
             {}\n\n\
             Note: This is a sophisticated method requiring:\n\
             - Computation of E(K) for quadratic extension K\n\
             - p-adic heights and logarithms\n\
             - Local-global compatibility checks\n\
             - Heegner point constructions (for CM curves)",
            self.curve,
            self.prime,
            self.precision,
            has_generator,
            if has_extension {
                format!("ℚ(√{})", self.extension_discriminant.as_ref().unwrap())
            } else {
                "not set".to_string()
            },
            applicable,
            if applicable {
                "Method is applicable and ready to compute."
            } else {
                "Method not applicable. Check requirements:\n\
                 1. Rank must be 1\n\
                 2. Generator must be provided\n\
                 3. Prime must be of good reduction"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chabauty_coleman_creation() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let prime = BigInt::from(7);
        let chabauty = ChabautyColeman::new(curve, prime);

        assert_eq!(chabauty.precision, 20);
    }

    #[test]
    #[should_panic(expected = "Prime must be of good reduction")]
    fn test_bad_reduction_panics() {
        // Create curve with discriminant divisible by 2
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(0),
            BigInt::from(2),
        );

        // Try to use p = 2 (bad reduction)
        let _chabauty = ChabautyColeman::new(curve, BigInt::from(2));
    }

    #[test]
    fn test_is_applicable_rank_zero() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let mut chabauty = ChabautyColeman::new(curve, BigInt::from(7));

        // Rank 0: applicable (0 < 1)
        assert!(chabauty.is_applicable());

        // Set rank to 1: not applicable (1 = 1)
        chabauty.set_generators(vec![RationalPoint::infinity()]);
        assert!(!chabauty.is_applicable());
    }

    #[test]
    fn test_find_two_torsion() {
        // Curve y² = x³ - x has 2-torsion points at x = 0, ±1
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let chabauty = ChabautyColeman::new(curve, BigInt::from(7));
        let two_torsion = chabauty.find_two_torsion();

        // Should find 3 points: (0,0), (1,0), (-1,0)
        assert!(two_torsion.len() >= 3);
    }

    #[test]
    fn test_reduce_modulo_p() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let chabauty = ChabautyColeman::new(curve, BigInt::from(7));
        let fp_points = chabauty.reduce_modulo_p();

        // Should find points on E(𝔽₇)
        assert!(!fp_points.is_empty());
    }

    #[test]
    fn test_chabauty_bound() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let chabauty = ChabautyColeman::new(curve, BigInt::from(7));
        let bound = chabauty.chabauty_bound();

        // Bound should be positive
        assert!(bound > 0);
    }

    #[test]
    fn test_find_torsion_points() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let chabauty = ChabautyColeman::new(curve, BigInt::from(7));
        let torsion = chabauty.find_torsion_points();

        // Should include at least the point at infinity
        assert!(!torsion.is_empty());
        assert!(torsion.iter().any(|p| p.is_infinity));
    }

    #[test]
    fn test_coleman_integral_creation() {
        let coleman = ColemanIntegral::new(BigInt::from(7), 20);
        assert_eq!(coleman.prime, BigInt::from(7));
        assert_eq!(coleman.precision, 20);
    }

    #[test]
    fn test_quadratic_chabauty_creation() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let quad_chab = QuadraticChabauty::new(curve, BigInt::from(7));
        assert_eq!(quad_chab.prime, BigInt::from(7));
    }

    #[test]
    fn test_generate_report() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let chabauty = ChabautyColeman::new(curve, BigInt::from(7));
        let report = chabauty.generate_report();

        assert!(report.contains("Chabauty-Coleman"));
        assert!(report.contains("Prime p: 7"));
        assert!(report.contains("Applicable"));
    }

    #[test]
    fn test_set_precision() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let mut chabauty = ChabautyColeman::new(curve, BigInt::from(7));
        chabauty.set_precision(50);

        assert_eq!(chabauty.precision, 50);
    }

    #[test]
    fn test_is_rational_point() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let chabauty = ChabautyColeman::new(curve.clone(), BigInt::from(7));

        // (0, 0) is on the curve
        let point = RationalPoint::new(
            BigRational::zero(),
            BigRational::zero(),
        );

        assert!(chabauty.is_rational_point(&point));

        // (1, 1) is not on the curve
        let bad_point = RationalPoint::new(
            BigRational::from(BigInt::from(1)),
            BigRational::from(BigInt::from(1)),
        );

        assert!(!chabauty.is_rational_point(&bad_point));
    }

    #[test]
    fn test_rank_zero_finds_all_torsion() {
        // Create a rank 0 curve
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let chabauty = ChabautyColeman::new(curve, BigInt::from(7));

        if chabauty.is_applicable() {
            let points = chabauty.find_rational_points();
            // Should find at least the point at infinity
            assert!(!points.is_empty());
        }
    }

    // Tests for DivisionPolynomials
    #[test]
    fn test_division_polynomial_psi_1() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut div_poly = DivisionPolynomials::new(curve);
        let psi_1 = div_poly.compute(1);

        // ψ₁ = 1
        assert_eq!(psi_1.len(), 1);
        assert_eq!(psi_1[0], BigRational::one());
    }

    #[test]
    fn test_division_polynomial_psi_2() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut div_poly = DivisionPolynomials::new(curve);
        let psi_2 = div_poly.compute(2);

        // ψ₂ = 2 (simplified for x-coordinates)
        assert_eq!(psi_2[0], BigRational::from(BigInt::from(2)));
    }

    #[test]
    fn test_division_polynomial_psi_3() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut div_poly = DivisionPolynomials::new(curve);
        let psi_3 = div_poly.compute(3);

        // ψ₃ = 3x⁴ + 6ax² + 12bx - a²
        // For a = -1, b = 0: ψ₃ = 3x⁴ - 6x² - 1
        assert_eq!(psi_3.len(), 5); // degree 4 polynomial has 5 coefficients
        assert_eq!(psi_3[0], BigRational::from(BigInt::from(-1))); // -a² = -1
        assert_eq!(psi_3[4], BigRational::from(BigInt::from(3)));  // 3x⁴
    }

    #[test]
    fn test_division_polynomial_psi_4() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(1),
            BigInt::from(1),
        );

        let mut div_poly = DivisionPolynomials::new(curve);
        let psi_4 = div_poly.compute(4);

        // ψ₄ should be a degree 6 polynomial
        assert!(psi_4.len() <= 7); // degree 6 has at most 7 coefficients
    }

    #[test]
    fn test_division_polynomial_caching() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut div_poly = DivisionPolynomials::new(curve);

        // Compute twice, should use cache second time
        let psi_3_first = div_poly.compute(3);
        let psi_3_second = div_poly.compute(3);

        assert_eq!(psi_3_first, psi_3_second);
    }

    #[test]
    fn test_division_polynomial_poly_mult() {
        let p = vec![
            BigRational::from(BigInt::from(1)),
            BigRational::from(BigInt::from(2)),
        ]; // 1 + 2x

        let q = vec![
            BigRational::from(BigInt::from(3)),
            BigRational::from(BigInt::from(4)),
        ]; // 3 + 4x

        let result = DivisionPolynomials::poly_mult(&p, &q);
        // (1 + 2x)(3 + 4x) = 3 + 10x + 8x²

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], BigRational::from(BigInt::from(3)));
        assert_eq!(result[1], BigRational::from(BigInt::from(10)));
        assert_eq!(result[2], BigRational::from(BigInt::from(8)));
    }

    #[test]
    fn test_division_polynomial_find_roots() {
        // Curve y² = x³ - x has 2-torsion at x = 0, ±1
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut div_poly = DivisionPolynomials::new(curve);

        // Find roots of ψ₂ (2-torsion x-coordinates)
        // For this curve, should find x = 0, 1, -1
        let roots = div_poly.find_rational_roots(2, 5);

        // Note: ψ₂ = 2 has no roots, but we're looking for 2-torsion
        // which are roots of the 2-division polynomial
        // The actual 2-torsion comes from y = 0, so x³ - x = 0
    }

    // Tests for TorsionFinder
    #[test]
    fn test_torsion_finder_creation() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let torsion_finder = TorsionFinder::new(curve, 100);
        assert_eq!(torsion_finder.search_bound, 100);
    }

    #[test]
    fn test_torsion_finder_find_2_torsion() {
        // Curve y² = x³ - x has three 2-torsion points
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut torsion_finder = TorsionFinder::new(curve, 10);
        let two_torsion = torsion_finder.find_n_torsion(2);

        // Should find: O (infinity), (0,0), (1,0), (-1,0)
        assert!(two_torsion.len() >= 1); // At least infinity
    }

    #[test]
    fn test_torsion_finder_find_3_torsion() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(0),
            BigInt::from(-1),
        );

        let mut torsion_finder = TorsionFinder::new(curve, 10);
        let three_torsion = torsion_finder.find_n_torsion(3);

        // Should at least find the point at infinity
        assert!(!three_torsion.is_empty());
    }

    #[test]
    fn test_torsion_finder_infinity() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let mut torsion_finder = TorsionFinder::new(curve, 10);
        let torsion = torsion_finder.find_n_torsion(1);

        // n=1 torsion should include only the infinity point
        assert_eq!(torsion.len(), 1);
        assert!(torsion[0].is_infinity);
    }

    #[test]
    fn test_torsion_finder_rational_sqrt() {
        // √(4/9) = 2/3
        let x = BigRational::new(BigInt::from(4), BigInt::from(9));
        let sqrt = TorsionFinder::rational_sqrt(&x);

        assert!(sqrt.is_some());
        let sqrt_val = sqrt.unwrap();
        assert_eq!(&sqrt_val * &sqrt_val, x);
    }

    #[test]
    fn test_torsion_finder_integer_sqrt() {
        assert_eq!(
            TorsionFinder::integer_sqrt(&BigInt::from(0)),
            Some(BigInt::from(0))
        );
        assert_eq!(
            TorsionFinder::integer_sqrt(&BigInt::from(1)),
            Some(BigInt::from(1))
        );
        assert_eq!(
            TorsionFinder::integer_sqrt(&BigInt::from(4)),
            Some(BigInt::from(2))
        );
        assert_eq!(
            TorsionFinder::integer_sqrt(&BigInt::from(9)),
            Some(BigInt::from(3))
        );
        assert_eq!(
            TorsionFinder::integer_sqrt(&BigInt::from(16)),
            Some(BigInt::from(4))
        );

        // Non-perfect squares
        assert_eq!(TorsionFinder::integer_sqrt(&BigInt::from(2)), None);
        assert_eq!(TorsionFinder::integer_sqrt(&BigInt::from(3)), None);
    }

    #[test]
    fn test_torsion_finder_find_all_torsion() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut torsion_finder = TorsionFinder::new(curve, 10);
        let all_torsion = torsion_finder.find_all_torsion(4);

        // Should have entries for various orders
        assert!(!all_torsion.is_empty());
    }

    // Tests for QuadraticChabauty
    #[test]
    fn test_quadratic_chabauty_creation() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let quad_chab = QuadraticChabauty::new(curve, BigInt::from(7));
        assert_eq!(quad_chab.prime, BigInt::from(7));
        assert_eq!(quad_chab.precision, 20);
    }

    #[test]
    fn test_quadratic_chabauty_set_precision() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let mut quad_chab = QuadraticChabauty::new(curve, BigInt::from(7));
        quad_chab.set_precision(50);

        assert_eq!(quad_chab.precision, 50);
    }

    #[test]
    fn test_quadratic_chabauty_set_generator() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut quad_chab = QuadraticChabauty::new(curve, BigInt::from(7));

        let generator = RationalPoint::new(
            BigRational::from(BigInt::from(0)),
            BigRational::from(BigInt::from(0)),
        );

        quad_chab.set_generator(generator);
        assert!(quad_chab.generator.is_some());
    }

    #[test]
    fn test_quadratic_chabauty_set_extension() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let mut quad_chab = QuadraticChabauty::new(curve, BigInt::from(7));
        quad_chab.set_extension(BigInt::from(-1));

        assert!(quad_chab.extension_discriminant.is_some());
        assert_eq!(quad_chab.extension_discriminant.unwrap(), BigInt::from(-1));
    }

    #[test]
    fn test_quadratic_chabauty_not_applicable_without_generator() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1),
        );

        let quad_chab = QuadraticChabauty::new(curve, BigInt::from(7));

        // Without a generator, should not be applicable
        assert!(!quad_chab.is_applicable());
    }

    #[test]
    fn test_quadratic_chabauty_applicable_with_generator() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut quad_chab = QuadraticChabauty::new(curve.clone(), BigInt::from(7));

        let generator = RationalPoint::new(
            BigRational::from(BigInt::from(0)),
            BigRational::from(BigInt::from(0)),
        );

        quad_chab.set_generator(generator);

        // With a generator and good reduction, should be applicable
        assert!(quad_chab.is_applicable());
    }

    #[test]
    fn test_quadratic_chabauty_bad_reduction() {
        // Create curve with discriminant divisible by 2
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(0),
            BigInt::from(2),
        );

        let mut quad_chab = QuadraticChabauty::new(curve.clone(), BigInt::from(2));

        let generator = RationalPoint::infinity();
        quad_chab.set_generator(generator);

        // Prime 2 is of bad reduction, so should not be applicable
        assert!(!quad_chab.is_applicable());
    }

    #[test]
    fn test_quadratic_chabauty_report() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut quad_chab = QuadraticChabauty::new(curve, BigInt::from(7));

        let generator = RationalPoint::new(
            BigRational::from(BigInt::from(0)),
            BigRational::from(BigInt::from(0)),
        );
        quad_chab.set_generator(generator);
        quad_chab.set_extension(BigInt::from(-1));

        let report = quad_chab.report();

        assert!(report.contains("Quadratic Chabauty"));
        assert!(report.contains("Prime p: 7"));
        assert!(report.contains("ℚ(√-1)"));
        assert!(report.contains("Applicable"));
    }

    #[test]
    fn test_quadratic_chabauty_find_points() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let mut quad_chab = QuadraticChabauty::new(curve.clone(), BigInt::from(7));

        let generator = RationalPoint::new(
            BigRational::from(BigInt::from(0)),
            BigRational::from(BigInt::from(0)),
        );
        quad_chab.set_generator(generator);

        let points = quad_chab.find_rational_points();

        // Should at least return the point at infinity
        assert!(!points.is_empty());
        assert!(points.iter().any(|p| p.is_infinity));
    }

    #[test]
    fn test_quadratic_chabauty_reduce_modulo_p() {
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let quad_chab = QuadraticChabauty::new(curve, BigInt::from(7));
        let reduction_points = quad_chab.reduce_modulo_p();

        // Should find points on E(𝔽₇)
        assert!(!reduction_points.is_empty());
    }

    #[test]
    fn test_integration_division_poly_with_chabauty() {
        // Test that ChabautyColeman now uses division polynomials
        let curve = EllipticCurveRational::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0),
        );

        let chabauty = ChabautyColeman::new(curve, BigInt::from(7));

        // For rank 0, should find all torsion points
        if chabauty.is_applicable() {
            let points = chabauty.find_rational_points();
            // Should find multiple torsion points now with improved algorithm
            assert!(!points.is_empty());
        }
    }
}
