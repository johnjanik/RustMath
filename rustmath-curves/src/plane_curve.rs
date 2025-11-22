//! Plane curves: affine and projective curves in the plane
//!
//! A plane curve is defined by a polynomial equation F(x,y) = 0 (affine)
//! or F(x,y,z) = 0 (projective homogeneous).

use rustmath_core::Ring;
use rustmath_polynomials::multivariate::MultiPoly;
use rustmath_rationals::Rational;
use rustmath_symbolic::expression::Expr;
use std::fmt;

/// A plane curve defined by a polynomial equation
#[derive(Debug, Clone)]
pub struct PlaneCurve<R: Ring> {
    /// The defining polynomial F(x,y) = 0 or F(x,y,z) = 0
    pub polynomial: MultiPoly<R>,
    /// Whether this is a projective curve (true) or affine (false)
    pub is_projective: bool,
    /// Variable names
    pub variables: Vec<String>,
}

impl<R: Ring> PlaneCurve<R> {
    /// Create a new plane curve from a polynomial
    pub fn new(polynomial: MultiPoly<R>, is_projective: bool) -> Self {
        let num_vars = if is_projective { 3 } else { 2 };
        let variables = if is_projective {
            vec!["x".to_string(), "y".to_string(), "z".to_string()]
        } else {
            vec!["x".to_string(), "y".to_string()]
        };

        assert_eq!(
            polynomial.variables().len(),
            num_vars,
            "Polynomial must have {} variables for {} curve",
            num_vars,
            if is_projective { "projective" } else { "affine" }
        );

        PlaneCurve {
            polynomial,
            is_projective,
            variables,
        }
    }

    /// Get the degree of the curve
    pub fn degree(&self) -> usize {
        self.polynomial.total_degree()
    }

    /// Check if a point lies on the curve
    pub fn contains_point(&self, point: &[R]) -> bool
    where
        R: PartialEq + Clone,
    {
        if self.is_projective {
            assert_eq!(point.len(), 3, "Projective point must have 3 coordinates");
        } else {
            assert_eq!(point.len(), 2, "Affine point must have 2 coordinates");
        }

        let value = self.polynomial.evaluate(point);
        value == R::zero()
    }

    /// Check if the curve is smooth (no singularities)
    pub fn is_smooth(&self) -> bool {
        self.singular_points().is_empty()
    }

    /// Find singular points (basic implementation)
    pub fn singular_points(&self) -> Vec<Vec<R>>
    where
        R: Clone + PartialEq,
    {
        // A point is singular if F = ∂F/∂x = ∂F/∂y = 0
        // (and ∂F/∂z = 0 for projective curves)

        // This is a simplified implementation
        // A full implementation would use Gröbner bases to solve the system
        vec![]
    }

    // TODO: Implement to_projective and to_affine when homogenize/dehomogenize are available
    // /// Convert to projective form (if affine)
    // pub fn to_projective(&self) -> PlaneCurve<R>
    // where
    //     R: Clone,
    // {
    //     if self.is_projective {
    //         return self.clone();
    //     }
    //
    //     // Homogenize the polynomial by introducing z
    //     let homogenized = self.polynomial.homogenize(2); // 2 is the z variable index
    //
    //     PlaneCurve::new(homogenized, true)
    // }
    //
    // /// Convert to affine form (if projective, by setting z=1)
    // pub fn to_affine(&self) -> PlaneCurve<R>
    // where
    //     R: Clone + PartialEq,
    // {
    //     if !self.is_projective {
    //         return self.clone();
    //     }
    //
    //     // Dehomogenize by setting z = 1
    //     let dehomogenized = self.polynomial.dehomogenize(2); // 2 is the z variable index
    //
    //     PlaneCurve::new(dehomogenized, false)
    // }
}

impl<R: Ring + fmt::Display> fmt::Display for PlaneCurve<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} curve: {} = 0",
            if self.is_projective { "Projective" } else { "Affine" },
            self.polynomial
        )
    }
}

/// Convenience type for affine curves
pub type AffineCurve<R> = PlaneCurve<R>;

/// Convenience type for projective curves
pub type ProjectiveCurve<R> = PlaneCurve<R>;

/// Common named curves
impl PlaneCurve<Rational> {
    /// Create a circle: x^2 + y^2 - r^2 = 0
    pub fn circle(radius: Rational) -> Self {
        use rustmath_polynomials::multivariate::Monomial;
        use std::collections::BTreeMap;

        // Create polynomial x^2 + y^2 - r^2
        let mut poly = MultiPoly::zero();

        // x^2 term
        let mut x2_exp = BTreeMap::new();
        x2_exp.insert(0, 2);
        poly.add_term(Monomial::from_exponents(x2_exp), Rational::one());

        // y^2 term
        let mut y2_exp = BTreeMap::new();
        y2_exp.insert(1, 2);
        poly.add_term(Monomial::from_exponents(y2_exp), Rational::one());

        // constant term -r^2
        poly.add_term(Monomial::new(), -radius.clone() * radius);

        PlaneCurve::new(poly, false)
    }

    /// Create a line: ax + by + c = 0
    pub fn line(a: Rational, b: Rational, c: Rational) -> Self {
        use rustmath_polynomials::multivariate::Monomial;
        use std::collections::BTreeMap;

        let mut poly = MultiPoly::zero();

        // ax term
        let mut x_exp = BTreeMap::new();
        x_exp.insert(0, 1);
        poly.add_term(Monomial::from_exponents(x_exp), a);

        // by term
        let mut y_exp = BTreeMap::new();
        y_exp.insert(1, 1);
        poly.add_term(Monomial::from_exponents(y_exp), b);

        // constant term c
        poly.add_term(Monomial::new(), c);

        PlaneCurve::new(poly, false)
    }

    /// Create a conic section: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    pub fn conic(a: Rational, b: Rational, c: Rational, d: Rational, e: Rational, f: Rational) -> Self {
        use rustmath_polynomials::multivariate::Monomial;
        use std::collections::BTreeMap;

        let mut poly = MultiPoly::zero();

        // Ax^2
        let mut x2_exp = BTreeMap::new();
        x2_exp.insert(0, 2);
        poly.add_term(Monomial::from_exponents(x2_exp), a);

        // Bxy
        let mut xy_exp = BTreeMap::new();
        xy_exp.insert(0, 1);
        xy_exp.insert(1, 1);
        poly.add_term(Monomial::from_exponents(xy_exp), b);

        // Cy^2
        let mut y2_exp = BTreeMap::new();
        y2_exp.insert(1, 2);
        poly.add_term(Monomial::from_exponents(y2_exp), c);

        // Dx
        let mut x_exp = BTreeMap::new();
        x_exp.insert(0, 1);
        poly.add_term(Monomial::from_exponents(x_exp), d);

        // Ey
        let mut y_exp = BTreeMap::new();
        y_exp.insert(1, 1);
        poly.add_term(Monomial::from_exponents(y_exp), e);

        // F
        poly.add_term(Monomial::new(), f);

        PlaneCurve::new(poly, false)
    }

    /// Create an elliptic curve in short Weierstrass form: y^2 = x^3 + ax + b
    /// Represented as: y^2 - x^3 - ax - b = 0
    pub fn elliptic_short_weierstrass(a: Rational, b: Rational) -> Self {
        use rustmath_polynomials::multivariate::Monomial;
        use std::collections::BTreeMap;

        let mut poly = MultiPoly::zero();

        // y^2
        let mut y2_exp = BTreeMap::new();
        y2_exp.insert(1, 2);
        poly.add_term(Monomial::from_exponents(y2_exp), Rational::one());

        // -x^3
        let mut x3_exp = BTreeMap::new();
        x3_exp.insert(0, 3);
        poly.add_term(Monomial::from_exponents(x3_exp), -Rational::one());

        // -ax
        let mut x_exp = BTreeMap::new();
        x_exp.insert(0, 1);
        poly.add_term(Monomial::from_exponents(x_exp), -a);

        // -b
        poly.add_term(Monomial::new(), -b);

        PlaneCurve::new(poly, false)
    }

    /// Create a cubic curve: y^2 = f(x) where f is a cubic
    pub fn cubic(f_coeffs: &[Rational]) -> Self {
        use rustmath_polynomials::multivariate::Monomial;
        use std::collections::BTreeMap;

        assert!(f_coeffs.len() <= 4, "Cubic polynomial has at most 4 coefficients");

        let mut poly = MultiPoly::zero();

        // y^2 term
        let mut y2_exp = BTreeMap::new();
        y2_exp.insert(1, 2);
        poly.add_term(Monomial::from_exponents(y2_exp), Rational::one());

        // -f(x) terms
        for (i, coeff) in f_coeffs.iter().enumerate() {
            let mut x_exp = BTreeMap::new();
            if i > 0 {
                x_exp.insert(0, i as u32);
            }
            poly.add_term(Monomial::from_exponents(x_exp), -coeff.clone());
        }

        PlaneCurve::new(poly, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_circle() {
        let r = Rational::from(5);
        let circle = PlaneCurve::circle(r);

        assert_eq!(circle.degree(), 2);
        assert!(!circle.is_projective);

        // Point (3, 4) is on the circle of radius 5
        let point = vec![Rational::from(3), Rational::from(4)];
        assert!(circle.contains_point(&point));

        // Point (1, 1) is not on the circle
        let point2 = vec![Rational::from(1), Rational::from(1)];
        assert!(!circle.contains_point(&point2));
    }

    #[test]
    fn test_line() {
        // Line: x + y - 1 = 0
        let line = PlaneCurve::line(
            Rational::one(),
            Rational::one(),
            -Rational::one()
        );

        assert_eq!(line.degree(), 1);

        // Point (0, 1) is on the line
        assert!(line.contains_point(&vec![Rational::zero(), Rational::one()]));

        // Point (1, 0) is on the line
        assert!(line.contains_point(&vec![Rational::one(), Rational::zero()]));
    }

    #[test]
    fn test_elliptic_curve() {
        // y^2 = x^3 + x (this is a smooth elliptic curve)
        let curve = PlaneCurve::elliptic_short_weierstrass(
            Rational::one(),
            Rational::zero()
        );

        assert_eq!(curve.degree(), 3);

        // Point (0, 0) is on the curve
        assert!(curve.contains_point(&vec![Rational::zero(), Rational::zero()]));
    }

    // TODO: Re-enable when to_projective and to_affine are implemented
    // #[test]
    // fn test_projective_conversion() {
    //     let line = PlaneCurve::line(
    //         Rational::one(),
    //         Rational::one(),
    //         -Rational::one()
    //     );
    //
    //     let proj = line.to_projective();
    //     assert!(proj.is_projective);
    //
    //     let affine = proj.to_affine();
    //     assert!(!affine.is_projective);
    // }

    #[test]
    fn test_conic() {
        // Unit circle: x^2 + y^2 - 1 = 0
        let conic = PlaneCurve::conic(
            Rational::one(),  // A
            Rational::zero(), // B
            Rational::one(),  // C
            Rational::zero(), // D
            Rational::zero(), // E
            -Rational::one()  // F
        );

        assert_eq!(conic.degree(), 2);
        assert!(conic.contains_point(&vec![Rational::one(), Rational::zero()]));
        assert!(conic.contains_point(&vec![Rational::zero(), Rational::one()]));
    }
}
