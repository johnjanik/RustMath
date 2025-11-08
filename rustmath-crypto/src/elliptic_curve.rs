//! Elliptic curve arithmetic over rationals
//!
//! This module implements elliptic curves in Weierstrass form: y^2 = x^3 + ax + b

use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;

/// A point on an elliptic curve
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EllipticCurvePoint<T> {
    /// The point at infinity (identity element)
    Infinity,
    /// An affine point (x, y)
    Affine { x: T, y: T },
}

impl<T: fmt::Display> fmt::Display for EllipticCurvePoint<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EllipticCurvePoint::Infinity => write!(f, "O (point at infinity)"),
            EllipticCurvePoint::Affine { x, y } => write!(f, "({}, {})", x, y),
        }
    }
}

/// An elliptic curve in Weierstrass form: y^2 = x^3 + ax + b
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EllipticCurve<T> {
    /// Coefficient a
    pub a: T,
    /// Coefficient b
    pub b: T,
}

impl EllipticCurve<Rational> {
    /// Create a new elliptic curve over the rationals
    ///
    /// Returns an error if the curve is singular (discriminant = 0)
    pub fn new(a: Rational, b: Rational) -> Result<Self, String> {
        // Check discriminant: Δ = -16(4a³ + 27b²)
        let a_cubed = a.clone() * a.clone() * a.clone();
        let b_squared = b.clone() * b.clone();

        let discriminant = (Rational::new(4, 1).unwrap() * a_cubed + Rational::new(27, 1).unwrap() * b_squared)
                         * Rational::new(-16, 1).unwrap();

        if discriminant == Rational::new(0, 1).unwrap() {
            return Err("Curve is singular (discriminant is zero)".to_string());
        }

        Ok(EllipticCurve { a, b })
    }

    /// Check if a point is on the curve
    pub fn contains_point(&self, point: &EllipticCurvePoint<Rational>) -> bool {
        match point {
            EllipticCurvePoint::Infinity => true,
            EllipticCurvePoint::Affine { x, y } => {
                // Check if y^2 = x^3 + ax + b
                let lhs = y.clone() * y.clone();
                let rhs = x.clone() * x.clone() * x.clone()
                        + self.a.clone() * x.clone()
                        + self.b.clone();
                lhs == rhs
            }
        }
    }

    /// Add two points on the elliptic curve
    pub fn add(&self, p: &EllipticCurvePoint<Rational>, q: &EllipticCurvePoint<Rational>)
        -> Result<EllipticCurvePoint<Rational>, String> {

        // Check that points are on the curve
        if !self.contains_point(p) {
            return Err("Point p is not on the curve".to_string());
        }
        if !self.contains_point(q) {
            return Err("Point q is not on the curve".to_string());
        }

        match (p, q) {
            // O + Q = Q
            (EllipticCurvePoint::Infinity, _) => Ok(q.clone()),
            // P + O = P
            (_, EllipticCurvePoint::Infinity) => Ok(p.clone()),

            (EllipticCurvePoint::Affine { x: x1, y: y1 },
             EllipticCurvePoint::Affine { x: x2, y: y2 }) => {

                // Check if points are inverses: P + (-P) = O
                if x1 == x2 && y1 != y2 {
                    return Ok(EllipticCurvePoint::Infinity);
                }

                // Calculate slope
                let slope = if x1 == x2 && y1 == y2 {
                    // Point doubling: slope = (3x₁² + a) / (2y₁)
                    if y1 == &Rational::new(0, 1).unwrap() {
                        return Ok(EllipticCurvePoint::Infinity);
                    }
                    let numerator = Rational::new(3, 1).unwrap() * x1.clone() * x1.clone() + self.a.clone();
                    let denominator = Rational::new(2, 1).unwrap() * y1.clone();
                    numerator / denominator
                } else {
                    // Point addition: slope = (y₂ - y₁) / (x₂ - x₁)
                    let numerator = y2.clone() - y1.clone();
                    let denominator = x2.clone() - x1.clone();

                    if denominator == Rational::new(0, 1).unwrap() {
                        return Ok(EllipticCurvePoint::Infinity);
                    }
                    numerator / denominator
                };

                // Calculate x₃ = slope² - x₁ - x₂
                let x3 = slope.clone() * slope.clone() - x1.clone() - x2.clone();

                // Calculate y₃ = slope(x₁ - x₃) - y₁
                let y3 = slope * (x1.clone() - x3.clone()) - y1.clone();

                Ok(EllipticCurvePoint::Affine { x: x3, y: y3 })
            }
        }
    }

    /// Double a point on the curve (more efficient than add(p, p))
    pub fn double(&self, p: &EllipticCurvePoint<Rational>)
        -> Result<EllipticCurvePoint<Rational>, String> {
        self.add(p, p)
    }

    /// Negate a point on the curve
    pub fn negate(&self, p: &EllipticCurvePoint<Rational>) -> EllipticCurvePoint<Rational> {
        match p {
            EllipticCurvePoint::Infinity => EllipticCurvePoint::Infinity,
            EllipticCurvePoint::Affine { x, y } => {
                EllipticCurvePoint::Affine {
                    x: x.clone(),
                    y: -y.clone()
                }
            }
        }
    }

    /// Scalar multiplication: compute n * P using double-and-add
    pub fn scalar_mul(&self, n: i64, p: &EllipticCurvePoint<Rational>)
        -> Result<EllipticCurvePoint<Rational>, String> {

        if n == 0 {
            return Ok(EllipticCurvePoint::Infinity);
        }

        if n < 0 {
            let neg_p = self.negate(p);
            return self.scalar_mul(-n, &neg_p);
        }

        // Double-and-add algorithm
        let mut result = EllipticCurvePoint::Infinity;
        let mut temp = p.clone();
        let mut k = n;

        while k > 0 {
            if k & 1 == 1 {
                result = self.add(&result, &temp)?;
            }
            temp = self.double(&temp)?;
            k >>= 1;
        }

        Ok(result)
    }

    /// Get the discriminant of the curve
    pub fn discriminant(&self) -> Rational {
        let a_cubed = self.a.clone() * self.a.clone() * self.a.clone();
        let b_squared = self.b.clone() * self.b.clone();

        (Rational::new(4, 1).unwrap() * a_cubed + Rational::new(27, 1).unwrap() * b_squared) * Rational::new(-16, 1).unwrap()
    }

    /// Get the j-invariant of the curve
    ///
    /// The j-invariant is given by: j = c₄³ / Δ
    /// where c₄ = -48a and Δ is the discriminant
    pub fn j_invariant(&self) -> Rational {
        let discriminant = self.discriminant();

        if discriminant == Rational::new(0, 1).unwrap() {
            panic!("Cannot compute j-invariant for singular curve");
        }

        // c₄ = -48a for Weierstrass form y² = x³ + ax + b
        let c4 = Rational::new(-48, 1).unwrap() * self.a.clone();
        let c4_cubed = c4.clone() * c4.clone() * c4.clone();

        // j = c₄³ / Δ
        c4_cubed / discriminant
    }

    /// Compute the order of a point (smallest n > 0 such that nP = O)
    ///
    /// Returns None if the point has infinite order or if the order exceeds max_order
    pub fn point_order(&self, p: &EllipticCurvePoint<Rational>, max_order: i64)
        -> Result<Option<i64>, String> {

        if !self.contains_point(p) {
            return Err("Point is not on the curve".to_string());
        }

        if p == &EllipticCurvePoint::Infinity {
            return Ok(Some(1)); // Order of identity is 1
        }

        let mut current = p.clone();
        for n in 1..=max_order {
            if current == EllipticCurvePoint::Infinity {
                return Ok(Some(n));
            }
            current = self.add(&current, p)?;
        }

        Ok(None) // Order exceeds max_order or point has infinite order
    }

    /// Check if a point is a torsion point (has finite order)
    ///
    /// Checks up to max_order. Returns true if the point has finite order <= max_order.
    pub fn is_torsion(&self, p: &EllipticCurvePoint<Rational>, max_order: i64)
        -> Result<bool, String> {
        let order = self.point_order(p, max_order)?;
        Ok(order.is_some())
    }

    /// Find all torsion points of a specific order
    ///
    /// This is a naive implementation that only works for small orders.
    /// For n-torsion, we need points P such that nP = O but kP ≠ O for 0 < k < n.
    ///
    /// Note: This returns only rational torsion points found by checking multiples.
    pub fn torsion_points_of_order(&self, order: i64, max_search: i64)
        -> Result<Vec<EllipticCurvePoint<Rational>>, String> {

        if order <= 0 {
            return Err("Order must be positive".to_string());
        }

        let mut torsion_points = Vec::new();

        // Always include the identity for order 1
        if order == 1 {
            torsion_points.push(EllipticCurvePoint::Infinity);
            return Ok(torsion_points);
        }

        // Search for rational points by trying x values
        // For each rational x, solve y^2 = x^3 + ax + b
        for num in -max_search..=max_search {
            for denom in 1..=max_search {
                let x = Rational::new(num, denom).unwrap();

                // Compute x^3 + ax + b
                let rhs = x.clone() * x.clone() * x.clone()
                        + self.a.clone() * x.clone()
                        + self.b.clone();

                // Check if rhs is a perfect square (approximate check for rationals)
                // For rational torsion points, we need y to be rational
                // This is a simplified check - try y values
                for y_num in -max_search..=max_search {
                    for y_denom in 1..=max_search {
                        let y = Rational::new(y_num, y_denom).unwrap();

                        if y.clone() * y.clone() == rhs {
                            let point = EllipticCurvePoint::Affine {
                                x: x.clone(),
                                y: y.clone(),
                            };

                            if let Ok(Some(pt_order)) = self.point_order(&point, order) {
                                if pt_order == order && !torsion_points.contains(&point) {
                                    torsion_points.push(point);
                                }
                            }

                            // Also check the negation
                            let neg_point = EllipticCurvePoint::Affine {
                                x: x.clone(),
                                y: -y.clone(),
                            };

                            if let Ok(Some(pt_order)) = self.point_order(&neg_point, order) {
                                if pt_order == order && !torsion_points.contains(&neg_point) {
                                    torsion_points.push(neg_point);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(torsion_points)
    }

    /// Get all 2-torsion points (points of order dividing 2)
    ///
    /// These are points (x, 0) where x^3 + ax + b = 0, plus the identity.
    /// Over the rationals, there can be 0, 1, or 3 such rational points.
    pub fn two_torsion_points(&self) -> Vec<EllipticCurvePoint<Rational>> {
        let mut points = vec![EllipticCurvePoint::Infinity];

        // Find roots of x^3 + ax + b = 0
        // This is a cubic equation, which is complex to solve in general
        // For now, try small integer values
        for x_num in -100..=100 {
            let x = Rational::new(x_num, 1).unwrap();
            let value = x.clone() * x.clone() * x.clone()
                      + self.a.clone() * x.clone()
                      + self.b.clone();

            if value == Rational::new(0, 1).unwrap() {
                points.push(EllipticCurvePoint::Affine {
                    x,
                    y: Rational::new(0, 1).unwrap(),
                });
            }
        }

        points
    }

    /// Search for rational points on the curve with bounded height
    ///
    /// Returns points (x, y) where |numerator| <= max_height and |denominator| <= max_height
    pub fn find_rational_points(&self, max_height: i64) -> Vec<EllipticCurvePoint<Rational>> {
        let mut points = vec![EllipticCurvePoint::Infinity];

        // Try rational x values
        for x_num in -max_height..=max_height {
            for x_denom in 1..=max_height {
                let x = Rational::new(x_num, x_denom).unwrap();

                // Compute y² = x³ + ax + b
                let y_squared = x.clone() * x.clone() * x.clone()
                              + self.a.clone() * x.clone()
                              + self.b.clone();

                // Check if y_squared is a perfect square
                // Try to find rational y
                for y_num in -max_height..=max_height {
                    for y_denom in 1..=max_height {
                        let y = Rational::new(y_num, y_denom).unwrap();

                        if y.clone() * y.clone() == y_squared {
                            let point = EllipticCurvePoint::Affine {
                                x: x.clone(),
                                y: y.clone(),
                            };

                            if !points.contains(&point) {
                                points.push(point);
                            }

                            // Also add the negation
                            let neg_point = EllipticCurvePoint::Affine {
                                x: x.clone(),
                                y: -y.clone(),
                            };

                            if !points.contains(&neg_point) && y != Rational::new(0, 1).unwrap() {
                                points.push(neg_point);
                            }
                        }
                    }
                }
            }
        }

        points
    }

    /// Check if two elliptic curves are isomorphic
    ///
    /// Two elliptic curves over the same field are isomorphic if and only if
    /// they have the same j-invariant
    pub fn is_isomorphic_to(&self, other: &EllipticCurve<Rational>) -> bool {
        self.j_invariant() == other.j_invariant()
    }

    /// Get the quadratic twist of this curve
    ///
    /// The quadratic twist by d is: dy² = x³ + ax + b
    /// which is isomorphic to y² = x³ + ad²x + bd³
    pub fn quadratic_twist(&self, d: Rational) -> Result<EllipticCurve<Rational>, String> {
        let d_squared = d.clone() * d.clone();
        let d_cubed = d_squared.clone() * d.clone();

        let new_a = self.a.clone() * d_squared;
        let new_b = self.b.clone() * d_cubed;

        EllipticCurve::new(new_a, new_b)
    }

    /// Count the number of affine rational points (excluding infinity) found with bounded search
    ///
    /// This is a naive implementation - proper point counting requires more advanced techniques
    pub fn count_rational_points(&self, max_height: i64) -> usize {
        let points = self.find_rational_points(max_height);
        // Subtract 1 to exclude the point at infinity
        points.len() - 1
    }

    /// Check if the curve has complex multiplication (CM)
    ///
    /// A curve has CM if j-invariant is 0 or 1728 (over Q)
    /// j = 0: y² = x³ + b (CM by cube roots of unity)
    /// j = 1728: y² = x³ + ax (CM by i)
    pub fn has_complex_multiplication(&self) -> bool {
        let j = self.j_invariant();
        j == Rational::new(0, 1).unwrap() || j == Rational::new(1728, 1).unwrap()
    }

    /// Get the conductor (simplified version for curves in minimal Weierstrass form)
    ///
    /// This is a placeholder - proper conductor computation requires reduction theory
    pub fn conductor_info(&self) -> String {
        format!(
            "Conductor computation requires reduction at each prime and is not yet implemented.\n\
             For curve y² = x³ + {}x + {}, discriminant = {}",
            self.a, self.b, self.discriminant()
        )
    }

    /// Count the number of points on the curve modulo prime p
    ///
    /// This counts solutions to y² ≡ x³ + ax + b (mod p) plus the point at infinity.
    /// For prime p, this is used to compute a_p for the L-function.
    ///
    /// # Arguments
    /// * `p` - A prime number
    ///
    /// # Returns
    /// The number of F_p-rational points on the curve (including infinity)
    pub fn count_points_mod_p(&self, p: i64) -> Result<i64, String> {
        if p <= 2 {
            return Err("Prime must be greater than 2".to_string());
        }

        let mut count = 1; // Start with 1 for the point at infinity

        // Convert a and b to integers mod p
        let a_num = integer_to_i64(self.a.numerator())?;
        let a_den = integer_to_i64(self.a.denominator())?;
        let b_num = integer_to_i64(self.b.numerator())?;
        let b_den = integer_to_i64(self.b.denominator())?;

        // Compute a mod p (need to handle rational a = a_num / a_den)
        // a ≡ a_num * a_den^(-1) (mod p)
        // First reduce modulo p, then compute inverse
        let a_num_mod = ((a_num % p) + p) % p;
        let b_num_mod = ((b_num % p) + p) % p;
        let a_den_mod = ((a_den % p) + p) % p;
        let b_den_mod = ((b_den % p) + p) % p;

        let a_mod = (a_num_mod * mod_inverse(a_den_mod, p)?) % p;
        let b_mod = (b_num_mod * mod_inverse(b_den_mod, p)?) % p;

        // For each x in F_p, check if x³ + ax + b is a quadratic residue
        for x in 0..p {
            let rhs = (x * x % p * x % p + a_mod * x % p + b_mod) % p;
            let rhs = ((rhs % p) + p) % p; // Ensure positive

            if rhs == 0 {
                count += 1; // Point (x, 0)
            } else {
                // Check if rhs is a quadratic residue using Legendre symbol
                if legendre_symbol(rhs, p) == 1 {
                    count += 2; // Two points (x, ±y)
                }
            }
        }

        Ok(count)
    }

    /// Compute the a_p coefficient for the L-function
    ///
    /// For prime p of good reduction, a_p = p + 1 - #E(F_p)
    /// This is the trace of Frobenius.
    pub fn a_p(&self, p: i64) -> Result<i64, String> {
        let point_count = self.count_points_mod_p(p)?;
        Ok(p + 1 - point_count)
    }

    /// Find integral points on the curve (points with integer coordinates)
    ///
    /// Searches for points (x, y) where both x and y are integers,
    /// with |x| ≤ max_x and |y| ≤ max_y.
    ///
    /// Note: For many elliptic curves, there are only finitely many integral points.
    pub fn find_integral_points(&self, max_x: i64, max_y: i64) -> Vec<EllipticCurvePoint<Rational>> {
        let mut points = vec![];

        for x_int in -max_x..=max_x {
            let x = Rational::new(x_int, 1).unwrap();

            // Compute y² = x³ + ax + b
            let y_squared = x.clone() * x.clone() * x.clone()
                          + self.a.clone() * x.clone()
                          + self.b.clone();

            // Check if y² is the square of an integer
            for y_int in -max_y..=max_y {
                let y = Rational::new(y_int, 1).unwrap();

                if y.clone() * y.clone() == y_squared {
                    let point = EllipticCurvePoint::Affine {
                        x: x.clone(),
                        y: y.clone(),
                    };

                    if !points.contains(&point) {
                        points.push(point);
                    }
                }
            }
        }

        points
    }

    /// Compute the naive height of a point
    ///
    /// For a point P = (x, y) with x = n/d in lowest terms,
    /// the naive height is h(P) = log(max(|n|, |d|))
    ///
    /// This is used as an approximation to the canonical height.
    pub fn naive_height(&self, p: &EllipticCurvePoint<Rational>) -> f64 {
        match p {
            EllipticCurvePoint::Infinity => 0.0,
            EllipticCurvePoint::Affine { x, .. } => {
                let num = integer_to_i64(x.numerator()).unwrap_or(1).abs();
                let den = integer_to_i64(x.denominator()).unwrap_or(1).abs();
                let max_val = num.max(den);
                (max_val as f64).ln()
            }
        }
    }

    /// Estimate whether the curve has positive rank
    ///
    /// Uses heuristics based on the number of small rational points found.
    /// This is NOT a rigorous rank computation, just a heuristic indicator.
    pub fn has_positive_rank_heuristic(&self, search_height: i64) -> bool {
        let points = self.find_rational_points(search_height);

        // Count non-torsion-looking points
        // A rough heuristic: if we find more than 10 points, likely positive rank
        // This is very crude and not mathematically rigorous!
        let non_torsion_count = points.iter()
            .filter(|p| {
                if let Ok(order_opt) = self.point_order(p, 20) {
                    order_opt.is_none() // No finite order found
                } else {
                    false
                }
            })
            .count();

        non_torsion_count > 0
    }

    /// Compute L-series information (partial implementation)
    ///
    /// Returns a_p for primes p up to max_prime.
    /// This is a step toward L-function computation.
    pub fn l_series_coefficients(&self, max_prime: i64) -> Vec<(i64, i64)> {
        let mut coefficients = Vec::new();

        for p in 2..=max_prime {
            if is_prime_simple(p) {
                if let Ok(a_p) = self.a_p(p) {
                    coefficients.push((p, a_p));
                }
            }
        }

        coefficients
    }
}

/// Helper function to convert Integer to i64
fn integer_to_i64(n: &Integer) -> Result<i64, String> {
    // Try to convert to string and parse
    let s = format!("{}", n);
    s.parse::<i64>()
        .map_err(|_| format!("Integer {} too large to convert to i64", n))
}

/// Helper function to compute modular inverse
fn mod_inverse(a: i64, m: i64) -> Result<i64, String> {
    let (g, x, _) = extended_gcd(a, m);
    if g != 1 {
        return Err(format!("{} has no inverse modulo {}", a, m));
    }
    Ok(((x % m) + m) % m)
}

/// Extended GCD
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a, 1, 0)
    } else {
        let (g, x, y) = extended_gcd(b, a % b);
        (g, y, x - (a / b) * y)
    }
}

/// Compute Legendre symbol (a/p)
fn legendre_symbol(a: i64, p: i64) -> i64 {
    if a % p == 0 {
        return 0;
    }

    // Use Euler's criterion: (a/p) ≡ a^((p-1)/2) (mod p)
    let exp = (p - 1) / 2;
    let result = mod_pow(a, exp, p);

    if result == 1 {
        1
    } else if result == p - 1 {
        -1
    } else {
        0
    }
}

/// Modular exponentiation
fn mod_pow(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
    if modulus == 1 {
        return 0;
    }

    let mut result = 1;
    base = ((base % modulus) + modulus) % modulus;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }

    result
}

/// Simple primality test
fn is_prime_simple(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }

    true
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elliptic_curve_creation() {
        // y^2 = x^3 - x (discriminant = 64)
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        );
        assert!(curve.is_ok());

        // Singular curve: y^2 = x^3 (discriminant = 0)
        let singular = EllipticCurve::new(
            Rational::new(0, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        );
        assert!(singular.is_err());
    }

    #[test]
    fn test_point_on_curve() {
        // y^2 = x^3 - x
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // (0, 0) is on the curve
        let p1 = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };
        assert!(curve.contains_point(&p1));

        // (1, 0) is on the curve: 0 = 1 - 1 + 0 = 0
        let p2 = EllipticCurvePoint::Affine {
            x: Rational::new(1, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };
        assert!(curve.contains_point(&p2));

        // (1, 1) is not on the curve: 1 ≠ 1 - 1 + 0 = 0
        let p3 = EllipticCurvePoint::Affine {
            x: Rational::new(1, 1).unwrap(),
            y: Rational::new(1, 1).unwrap(),
        };
        assert!(!curve.contains_point(&p3));
    }

    #[test]
    fn test_point_addition_identity() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let p = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };

        // P + O = P
        let result = curve.add(&p, &EllipticCurvePoint::Infinity).unwrap();
        assert_eq!(result, p);

        // O + P = P
        let result = curve.add(&EllipticCurvePoint::Infinity, &p).unwrap();
        assert_eq!(result, p);
    }

    #[test]
    fn test_point_addition_inverse() {
        let curve = EllipticCurve::new(
            Rational::new(2, 1).unwrap(),
            Rational::new(3, 1).unwrap()
        ).unwrap();

        // Point and its inverse
        // For y^2 = x^3 + 2x + 3, point (-1, 0) is on the curve
        let p = EllipticCurvePoint::Affine {
            x: Rational::new(-1, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };

        let neg_p = curve.negate(&p);

        // P + (-P) = O (in this case p = -p since y = 0)
        let result = curve.add(&p, &neg_p).unwrap();
        assert_eq!(result, EllipticCurvePoint::Infinity);
    }

    #[test]
    fn test_scalar_multiplication() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let p = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };

        // 0 * P = O
        let result = curve.scalar_mul(0, &p).unwrap();
        assert_eq!(result, EllipticCurvePoint::Infinity);

        // 1 * P = P
        let result = curve.scalar_mul(1, &p).unwrap();
        assert_eq!(result, p);

        // 2 * P = P + P
        let double = curve.double(&p).unwrap();
        let scalar_2 = curve.scalar_mul(2, &p).unwrap();
        assert_eq!(double, scalar_2);
    }

    #[test]
    fn test_discriminant() {
        // y^2 = x^3 - x has discriminant 64
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let disc = curve.discriminant();
        assert_eq!(disc, Rational::new(64, 1).unwrap());
    }

    #[test]
    fn test_point_order_identity() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let order = curve.point_order(&EllipticCurvePoint::Infinity, 100).unwrap();
        assert_eq!(order, Some(1));
    }

    #[test]
    fn test_point_order_two_torsion() {
        // y^2 = x^3 - x has 2-torsion points at (0,0), (1,0), (-1,0)
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let p = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };

        let order = curve.point_order(&p, 10).unwrap();
        assert_eq!(order, Some(2));
    }

    #[test]
    fn test_is_torsion() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // (0, 0) is a 2-torsion point
        let p = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };

        assert!(curve.is_torsion(&p, 10).unwrap());
    }

    #[test]
    fn test_two_torsion_points() {
        // y^2 = x^3 - x has three 2-torsion points: (0,0), (1,0), (-1,0)
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let torsion = curve.two_torsion_points();

        // Should find 4 points total: O and the three points with y=0
        assert_eq!(torsion.len(), 4);
        assert!(torsion.contains(&EllipticCurvePoint::Infinity));
    }

    #[test]
    fn test_torsion_points_of_order_one() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let order_1 = curve.torsion_points_of_order(1, 5).unwrap();
        assert_eq!(order_1.len(), 1);
        assert_eq!(order_1[0], EllipticCurvePoint::Infinity);
    }

    #[test]
    fn test_j_invariant() {
        // y^2 = x^3 + x has j-invariant 1728 (curve with CM by i)
        let curve = EllipticCurve::new(
            Rational::new(1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let j = curve.j_invariant();
        assert_eq!(j, Rational::new(1728, 1).unwrap());
    }

    #[test]
    fn test_j_invariant_other() {
        // y^2 = x^3 + 1 has j-invariant 0 (curve with CM by ω)
        let curve = EllipticCurve::new(
            Rational::new(0, 1).unwrap(),
            Rational::new(1, 1).unwrap()
        ).unwrap();

        let j = curve.j_invariant();
        assert_eq!(j, Rational::new(0, 1).unwrap());
    }

    #[test]
    fn test_find_rational_points() {
        // y^2 = x^3 - x
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // Search for points with small height
        let points = curve.find_rational_points(2);

        // Should find at least the torsion points and infinity
        assert!(points.len() >= 4); // O, (0,0), (1,0), (-1,0)
        assert!(points.contains(&EllipticCurvePoint::Infinity));
    }

    #[test]
    fn test_is_isomorphic() {
        // Two curves with the same j-invariant are isomorphic
        let curve1 = EllipticCurve::new(
            Rational::new(1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let curve2 = EllipticCurve::new(
            Rational::new(4, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // Both have j = 1728, so they're isomorphic
        assert!(curve1.is_isomorphic_to(&curve2));
    }

    #[test]
    fn test_not_isomorphic() {
        let curve1 = EllipticCurve::new(
            Rational::new(1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let curve2 = EllipticCurve::new(
            Rational::new(0, 1).unwrap(),
            Rational::new(1, 1).unwrap()
        ).unwrap();

        // Different j-invariants (1728 vs 0)
        assert!(!curve1.is_isomorphic_to(&curve2));
    }

    #[test]
    fn test_quadratic_twist() {
        let curve = EllipticCurve::new(
            Rational::new(1, 1).unwrap(),
            Rational::new(2, 1).unwrap()
        ).unwrap();

        let twist = curve.quadratic_twist(Rational::new(2, 1).unwrap());
        assert!(twist.is_ok());

        let twisted_curve = twist.unwrap();
        // Twisted curve should have: a' = a*d², b' = b*d³
        // a' = 1*4 = 4, b' = 2*8 = 16
        assert_eq!(twisted_curve.a, Rational::new(4, 1).unwrap());
        assert_eq!(twisted_curve.b, Rational::new(16, 1).unwrap());
    }

    #[test]
    fn test_has_complex_multiplication() {
        // y^2 = x^3 + x has j = 1728 (CM by i)
        let curve1 = EllipticCurve::new(
            Rational::new(1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();
        assert!(curve1.has_complex_multiplication());

        // y^2 = x^3 + 1 has j = 0 (CM by ω)
        let curve2 = EllipticCurve::new(
            Rational::new(0, 1).unwrap(),
            Rational::new(1, 1).unwrap()
        ).unwrap();
        assert!(curve2.has_complex_multiplication());

        // y^2 = x^3 - x + 1 has generic j-invariant
        let curve3 = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(1, 1).unwrap()
        ).unwrap();
        assert!(!curve3.has_complex_multiplication());
    }

    #[test]
    fn test_count_rational_points() {
        // y^2 = x^3 - x
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let count = curve.count_rational_points(2);
        // Should find the three 2-torsion points at minimum
        assert!(count >= 3);
    }

    #[test]
    fn test_conductor_info() {
        let curve = EllipticCurve::new(
            Rational::new(1, 1).unwrap(),
            Rational::new(1, 1).unwrap()
        ).unwrap();

        let info = curve.conductor_info();
        assert!(info.contains("Conductor"));
        assert!(info.contains("discriminant"));
    }

    #[test]
    fn test_count_points_mod_p() {
        // y^2 = x^3 - x over F_5
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // Count points modulo 5
        let count = curve.count_points_mod_p(5).unwrap();

        // For y^2 = x^3 - x mod 5, we expect a specific count
        // This should include the point at infinity
        assert!(count > 0);
        assert!(count <= 11); // At most p + 1 + 2*sqrt(p) by Hasse's theorem
    }

    #[test]
    fn test_a_p_coefficient() {
        // y^2 = x^3 + x over F_p
        let curve = EllipticCurve::new(
            Rational::new(1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // Compute a_5
        let a_5 = curve.a_p(5).unwrap();

        // |a_p| ≤ 2*sqrt(p) by Hasse's theorem
        // For p=5, |a_p| ≤ 2*sqrt(5) ≈ 4.47
        assert!(a_5.abs() <= 5);
    }

    #[test]
    fn test_a_p_multiple_primes() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // Test several small primes
        for p in [3, 5, 7, 11, 13].iter() {
            let a_p = curve.a_p(*p).unwrap();
            // Hasse's theorem: |a_p| ≤ 2*sqrt(p)
            let bound = 2.0 * (*p as f64).sqrt();
            assert!((a_p as f64).abs() <= bound + 1.0); // +1 for rounding
        }
    }

    #[test]
    fn test_find_integral_points() {
        // y^2 = x^3 - x has integral points like (0,0), (1,0), (-1,0)
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let int_points = curve.find_integral_points(5, 10);

        // Should find the three 2-torsion points at least
        assert!(int_points.len() >= 3);

        // Check that all points have integer coordinates
        for point in &int_points {
            if let EllipticCurvePoint::Affine { x, y } = point {
                assert_eq!(x.denominator(), &Integer::from(1));
                assert_eq!(y.denominator(), &Integer::from(1));
            }
        }
    }

    #[test]
    fn test_naive_height_infinity() {
        let curve = EllipticCurve::new(
            Rational::new(1, 1).unwrap(),
            Rational::new(1, 1).unwrap()
        ).unwrap();

        let height = curve.naive_height(&EllipticCurvePoint::Infinity);
        assert_eq!(height, 0.0);
    }

    #[test]
    fn test_naive_height() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // Point with x = 1/1
        let p1 = EllipticCurvePoint::Affine {
            x: Rational::new(1, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };
        let h1 = curve.naive_height(&p1);
        assert_eq!(h1, (1.0_f64).ln()); // ln(max(1, 1)) = ln(1) = 0

        // Point with larger coordinates
        let p2 = EllipticCurvePoint::Affine {
            x: Rational::new(5, 2).unwrap(),
            y: Rational::new(1, 1).unwrap(),
        };
        let h2 = curve.naive_height(&p2);
        assert!(h2 > 0.0);
        assert_eq!(h2, (5.0_f64).ln()); // max(5, 2) = 5
    }

    #[test]
    fn test_l_series_coefficients() {
        let curve = EllipticCurve::new(
            Rational::new(1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let coeffs = curve.l_series_coefficients(20);

        // Should have coefficients for primes 2, 3, 5, 7, 11, 13, 17, 19
        assert!(coeffs.len() >= 6); // At least primes up to 13

        // Check that all returned values are for primes
        for (p, _) in &coeffs {
            assert!(is_prime_simple(*p));
        }

        // Check Hasse's theorem for each coefficient
        for (p, a_p) in &coeffs {
            let bound = 2.0 * (*p as f64).sqrt();
            assert!((*a_p as f64).abs() <= bound + 1.0);
        }
    }

    #[test]
    fn test_has_positive_rank_heuristic() {
        // y^2 = x^3 - x (rank 0, only torsion)
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // This curve has rank 0, so heuristic might say false
        // But it's just a heuristic, so we just check it doesn't panic
        let _ = curve.has_positive_rank_heuristic(3);
    }

    #[test]
    fn test_integral_points_specific() {
        // y^2 = x^3 + 1 has integral point (0, ±1), (2, ±3)
        let curve = EllipticCurve::new(
            Rational::new(0, 1).unwrap(),
            Rational::new(1, 1).unwrap()
        ).unwrap();

        let int_points = curve.find_integral_points(10, 20);

        // Check that (0, 1) and (0, -1) are found
        let p1 = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(1, 1).unwrap(),
        };
        let p2 = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(-1, 1).unwrap(),
        };

        assert!(int_points.contains(&p1));
        assert!(int_points.contains(&p2));
    }

    #[test]
    fn test_point_count_hasse_bound() {
        // Test Hasse's theorem: |#E(F_p) - (p+1)| ≤ 2*sqrt(p)
        let curve = EllipticCurve::new(
            Rational::new(2, 1).unwrap(),
            Rational::new(3, 1).unwrap()
        ).unwrap();

        for p in [7, 11, 13, 17, 19].iter() {
            let count = curve.count_points_mod_p(*p).unwrap();
            let diff = (count - (*p + 1)).abs();
            let hasse_bound = 2.0 * (*p as f64).sqrt();

            assert!((diff as f64) <= hasse_bound + 1.0);
        }
    }
}
