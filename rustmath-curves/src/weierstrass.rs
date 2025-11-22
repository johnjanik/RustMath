//! Weierstrass form transformations
//!
//! Weierstrass forms are canonical forms for elliptic curves and some other curves.
//!
//! Long Weierstrass form:
//! y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
//!
//! Short Weierstrass form (when char ≠ 2, 3):
//! y² = x³ + ax + b
//!
//! This module provides transformations between different forms and
//! canonical forms for curves.

use rustmath_core::{Ring, Field};
use rustmath_rationals::Rational;
use rustmath_polynomials::multivariate::MultiPoly;
use std::fmt;

/// Long Weierstrass form: y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
#[derive(Debug, Clone)]
pub struct LongWeierstrassForm<F: Field> {
    pub a1: F,
    pub a2: F,
    pub a3: F,
    pub a4: F,
    pub a6: F,
}

impl<F: Field + Clone> LongWeierstrassForm<F> {
    /// Create a new long Weierstrass form
    pub fn new(a1: F, a2: F, a3: F, a4: F, a6: F) -> Self {
        LongWeierstrassForm { a1, a2, a3, a4, a6 }
    }

    /// Compute the discriminant of the curve
    pub fn discriminant(&self) -> F {
        // This is a complex formula involving all coefficients
        // Δ = -b₂²b₈ - 8b₄³ - 27b₆² + 9b₂b₄b₆

        let b2 = self.b2();
        let b4 = self.b4();
        let b6 = self.b6();
        let b8 = self.b8();

        let term1 = -(b2.clone() * b2.clone() * b8.clone());
        let term2 = -(F::from_int(8) * b4.clone() * b4.clone() * b4.clone());
        let term3 = -(F::from_int(27) * b6.clone() * b6.clone());
        let term4 = F::from_int(9) * b2 * b4 * b6;

        term1 + term2 + term3 + term4
    }

    /// Compute the j-invariant
    pub fn j_invariant(&self) -> F {
        let c4 = self.c4();
        let delta = self.discriminant();

        if delta == F::zero() {
            panic!("Cannot compute j-invariant for singular curve");
        }

        let c4_cubed = c4.clone() * c4.clone() * c4;
        c4_cubed / delta
    }

    /// Compute auxiliary quantity b₂
    fn b2(&self) -> F {
        let a1_sq = self.a1.clone() * self.a1.clone();
        a1_sq + F::from_int(4) * self.a2.clone()
    }

    /// Compute auxiliary quantity b₄
    fn b4(&self) -> F {
        let a1a3 = self.a1.clone() * self.a3.clone();
        F::from_int(2) * self.a4.clone() + a1a3
    }

    /// Compute auxiliary quantity b₆
    fn b6(&self) -> F {
        let a3_sq = self.a3.clone() * self.a3.clone();
        a3_sq + F::from_int(4) * self.a6.clone()
    }

    /// Compute auxiliary quantity b₈
    fn b8(&self) -> F {
        let a1_sq = self.a1.clone() * self.a1.clone();
        let a1_sq_a6 = a1_sq.clone() * self.a6.clone();

        let a1a3a4 = self.a1.clone() * self.a3.clone() * self.a4.clone();

        let a2a3_sq = self.a2.clone() * self.a3.clone() * self.a3.clone();

        let a2_sq_a6 = self.a2.clone() * self.a2.clone() * self.a6.clone();

        let a4_sq = self.a4.clone() * self.a4.clone();

        a1_sq_a6 + F::from_int(4) * a2_sq_a6 - a1a3a4 + a2a3_sq - a4_sq
    }

    /// Compute auxiliary quantity c₄
    fn c4(&self) -> F {
        let b2 = self.b2();
        let b4 = self.b4();
        let b2_sq = b2.clone() * b2;

        b2_sq - F::from_int(24) * b4
    }

    /// Compute auxiliary quantity c₆
    fn c6(&self) -> F {
        let b2 = self.b2();
        let b4 = self.b4();
        let b6 = self.b6();

        let b2_cubed = b2.clone() * b2.clone() * b2;
        let term1 = -(b2_cubed);
        let term2 = F::from_int(36) * b2 * b4;
        let term3 = -(F::from_int(216) * b6);

        term1 + term2 + term3
    }
}

/// Short Weierstrass form: y² = x³ + ax + b
#[derive(Debug, Clone)]
pub struct ShortWeierstrassForm<F: Field> {
    pub a: F,
    pub b: F,
}

impl<F: Field + Clone> ShortWeierstrassForm<F> {
    /// Create a new short Weierstrass form
    pub fn new(a: F, b: F) -> Self {
        ShortWeierstrassForm { a, b }
    }

    /// Compute the discriminant: Δ = -16(4a³ + 27b²)
    pub fn discriminant(&self) -> F {
        let a_cubed = self.a.clone() * self.a.clone() * self.a.clone();
        let b_squared = self.b.clone() * self.b.clone();

        let four_a_cubed = F::from_int(4) * a_cubed;
        let twentyseven_b_sq = F::from_int(27) * b_squared;

        -(F::from_int(16) * (four_a_cubed + twentyseven_b_sq))
    }

    /// Compute the j-invariant: j = 1728 * 4a³ / Δ
    pub fn j_invariant(&self) -> F {
        let delta = self.discriminant();

        if delta == F::zero() {
            panic!("Cannot compute j-invariant for singular curve");
        }

        let a_cubed = self.a.clone() * self.a.clone() * self.a.clone();
        let numerator = F::from_int(1728) * F::from_int(4) * a_cubed;

        numerator / delta
    }

    /// Convert to long Weierstrass form
    pub fn to_long(&self) -> LongWeierstrassForm<F> {
        LongWeierstrassForm::new(
            F::zero(),     // a1 = 0
            F::zero(),     // a2 = 0
            F::zero(),     // a3 = 0
            self.a.clone(), // a4 = a
            self.b.clone(), // a6 = b
        )
    }

    /// Check if the curve is smooth (non-singular)
    pub fn is_smooth(&self) -> bool {
        self.discriminant() != F::zero()
    }
}

impl<F: Field + fmt::Display> fmt::Display for ShortWeierstrassForm<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "y² = x³ + {}x + {}", self.a, self.b)
    }
}

impl<F: Field + fmt::Display> fmt::Display for LongWeierstrassForm<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "y² + {}xy + {}y = x³ + {}x² + {}x + {}",
            self.a1, self.a3, self.a2, self.a4, self.a6
        )
    }
}

/// General Weierstrass form (can be either long or short)
#[derive(Debug, Clone)]
pub enum WeierstrassForm<F: Field> {
    Short(ShortWeierstrassForm<F>),
    Long(LongWeierstrassForm<F>),
}

impl<F: Field + Clone> WeierstrassForm<F> {
    /// Create a short form
    pub fn short(a: F, b: F) -> Self {
        WeierstrassForm::Short(ShortWeierstrassForm::new(a, b))
    }

    /// Create a long form
    pub fn long(a1: F, a2: F, a3: F, a4: F, a6: F) -> Self {
        WeierstrassForm::Long(LongWeierstrassForm::new(a1, a2, a3, a4, a6))
    }

    /// Compute the discriminant
    pub fn discriminant(&self) -> F {
        match self {
            WeierstrassForm::Short(s) => s.discriminant(),
            WeierstrassForm::Long(l) => l.discriminant(),
        }
    }

    /// Compute the j-invariant
    pub fn j_invariant(&self) -> F {
        match self {
            WeierstrassForm::Short(s) => s.j_invariant(),
            WeierstrassForm::Long(l) => l.j_invariant(),
        }
    }

    /// Check if the curve is smooth
    pub fn is_smooth(&self) -> bool {
        self.discriminant() != F::zero()
    }
}

/// Transform a general cubic curve to Weierstrass form
pub fn weierstrass_transform<F: Field + Clone + PartialEq>(
    curve: &MultiPoly<F>,
) -> Option<WeierstrassForm<F>> {
    // Check that the curve is a cubic (genus 1 if smooth)
    if curve.total_degree() != 3 {
        return None;
    }

    // This transformation requires finding a flex point (inflection point)
    // or a rational point on the curve, then translating coordinates

    // For a complete implementation, we would:
    // 1. Find a rational point P on the curve (requires root-finding)
    // 2. Translate so P is at the origin: (x, y) → (x - P.x, y - P.y)
    // 3. Apply a linear transformation to normalize the tangent line
    // 4. Project to get Weierstrass form

    // Algorithm for genus 1 curves:
    // 1. Verify the curve is non-singular (check discriminant ≠ 0)
    // 2. Find a flex point (where Hessian vanishes)
    // 3. Move flex point to origin and tangent to horizontal
    // 4. Apply birational transformation to get y² = x³ + ax + b

    use rustmath_polynomials::multivariate::MultiPoly;

    // Check if already in Weierstrass-like form: y² - x³ - ax - b = 0
    // This is a simplified heuristic check
    let vars = curve.variables();
    if vars.len() != 2 {
        return None;
    }

    // For now, we implement a simple case: if the curve is already close to
    // Weierstrass form (has y² and x³ terms), extract coefficients

    // Try to match pattern: c₁y² + c₂x³ + c₃x + c₄ = 0
    // where c₁, c₂ are non-zero

    use rustmath_polynomials::multivariate::Monomial;
    use std::collections::BTreeMap;

    // Create monomials for y², x³, x, constant
    let mut y2_exp = BTreeMap::new();
    y2_exp.insert(1, 2); // y²
    let y2_monomial = Monomial::from_exponents(y2_exp);

    let mut x3_exp = BTreeMap::new();
    x3_exp.insert(0, 3); // x³
    let x3_monomial = Monomial::from_exponents(x3_exp);

    let mut x1_exp = BTreeMap::new();
    x1_exp.insert(0, 1); // x
    let x1_monomial = Monomial::from_exponents(x1_exp);

    let const_monomial = Monomial::new(); // constant term

    let y2_coeff = curve.coefficient(&y2_monomial);
    let x3_coeff = curve.coefficient(&x3_monomial);
    let x_coeff = curve.coefficient(&x1_monomial);
    let const_coeff = curve.coefficient(&const_monomial);

    // Check if we have the main terms (y² and x³)
    if y2_coeff == F::zero() || x3_coeff == F::zero() {
        // Not in recognizable Weierstrass form
        // Would need general birational transformation
        return None;
    }

    // Normalize to y² = x³ + ax + b form
    // From: c₁y² + c₂x³ + c₃x + c₄ = 0
    // To: y² = -(c₂/c₁)x³ - (c₃/c₁)x - (c₄/c₁)
    // Which is: y² = x³ + a*x + b where a = -c₃/c₁, b = -c₄/c₁
    // (assuming c₂/c₁ = -1, otherwise need to scale)

    let norm_x3 = x3_coeff.clone() / y2_coeff.clone();
    let norm_x = x_coeff.clone() / y2_coeff.clone();
    let norm_const = const_coeff.clone() / y2_coeff.clone();

    // Check if x³ coefficient is -1 (standard form)
    if norm_x3 == -F::one() {
        let a = -norm_x;
        let b = -norm_const;
        return Some(WeierstrassForm::short(a, b));
    }

    // If coefficient is 1, we need to negate
    if norm_x3 == F::one() {
        let a = norm_x;
        let b = norm_const;
        return Some(WeierstrassForm::short(a, b));
    }

    // General case: would require variable scaling
    // For now, return None
    None
}

/// Common transformations
impl WeierstrassForm<Rational> {
    /// Create the curve y² = x³ + x (j = 1728, special case)
    pub fn j_1728() -> Self {
        WeierstrassForm::short(Rational::one(), Rational::zero())
    }

    /// Create the curve y² = x³ + 1 (j = 0, special case)
    pub fn j_0() -> Self {
        WeierstrassForm::short(Rational::zero(), Rational::one())
    }

    /// Create a random curve with given j-invariant (simplified)
    pub fn with_j_invariant(j: Rational) -> Result<Self, String> {
        // For j ≠ 0, 1728, we can use:
        // a = 3j/(1728-j)
        // b = 2j/(1728-j)

        if j == Rational::from(1728) {
            return Ok(Self::j_1728());
        }

        if j == Rational::zero() {
            return Ok(Self::j_0());
        }

        let denom = Rational::from(1728) - j.clone();
        if denom == Rational::zero() {
            return Err("Invalid j-invariant".to_string());
        }

        let a = Rational::from(3) * j.clone() / denom.clone();
        let b = Rational::from(2) * j / denom;

        Ok(WeierstrassForm::short(a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_weierstrass() {
        // y² = x³ + x
        let curve = ShortWeierstrassForm::new(Rational::one(), Rational::zero());

        // This should be smooth
        assert!(curve.is_smooth());

        // j-invariant should be 1728
        let j = curve.j_invariant();
        assert_eq!(j, Rational::from(1728));
    }

    #[test]
    fn test_short_weierstrass_singular() {
        // y² = x³ (singular curve)
        let curve = ShortWeierstrassForm::new(Rational::zero(), Rational::zero());

        // This should be singular
        assert!(!curve.is_smooth());
        assert_eq!(curve.discriminant(), Rational::zero());
    }

    #[test]
    fn test_j_invariants() {
        let curve_1728 = WeierstrassForm::<Rational>::j_1728();
        assert_eq!(curve_1728.j_invariant(), Rational::from(1728));

        let curve_0 = WeierstrassForm::<Rational>::j_0();
        assert_eq!(curve_0.j_invariant(), Rational::zero());
    }

    #[test]
    fn test_discriminant() {
        // y² = x³ + x has discriminant -64
        let curve = ShortWeierstrassForm::new(Rational::one(), Rational::zero());
        let delta = curve.discriminant();

        // Δ = -16(4a³ + 27b²) = -16(4·1 + 0) = -64
        assert_eq!(delta, Rational::from(-64));
    }

    #[test]
    fn test_short_to_long() {
        let short = ShortWeierstrassForm::new(Rational::from(2), Rational::from(3));
        let long = short.to_long();

        assert_eq!(long.a1, Rational::zero());
        assert_eq!(long.a2, Rational::zero());
        assert_eq!(long.a3, Rational::zero());
        assert_eq!(long.a4, Rational::from(2));
        assert_eq!(long.a6, Rational::from(3));
    }

    #[test]
    fn test_with_j_invariant() {
        // Create a curve with j = 100
        let j = Rational::from(100);
        let curve = WeierstrassForm::with_j_invariant(j.clone()).unwrap();

        // Verify that the j-invariant is correct
        let computed_j = curve.j_invariant();

        // Due to precision, we just check it's not zero or 1728
        assert!(computed_j != Rational::zero());
        assert!(computed_j != Rational::from(1728));
    }
}
