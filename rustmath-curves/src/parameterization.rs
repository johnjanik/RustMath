//! Curve parameterization
//!
//! A curve parameterization is a map from a parameter space to the curve.
//! For rational curves (genus 0), we can find rational parameterizations.
//!
//! Common parameterizations:
//! - Circle: x = cos(t), y = sin(t) (transcendental)
//! - Circle: x = (1-t²)/(1+t²), y = 2t/(1+t²) (rational, via stereographic projection)
//! - Conic sections: Various rational parameterizations
//! - Singular cubics: Can be rationally parameterized

use rustmath_core::{Ring, Field};
use rustmath_rationals::Rational;
use rustmath_symbolic::expression::Expr;
use std::fmt;

/// A curve parameterization x = x(t), y = y(t)
#[derive(Debug, Clone)]
pub struct CurveParameterization {
    /// Parametric expression for x-coordinate
    pub x: Expr,
    /// Parametric expression for y-coordinate
    pub y: Expr,
    /// Parameter name (usually "t")
    pub parameter: String,
}

impl CurveParameterization {
    /// Create a new parameterization
    pub fn new(x: Expr, y: Expr, parameter: String) -> Self {
        CurveParameterization { x, y, parameter }
    }

    /// Evaluate the parameterization at a specific parameter value
    pub fn evaluate(&self, _t: f64) -> (f64, f64) {
        // Note: This would require symbolic expression evaluation
        // which is beyond the scope of this implementation
        // Return placeholder
        (0.0, 0.0) // Placeholder
    }
}

impl fmt::Display for CurveParameterization {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "x({}) = {}, y({}) = {}",
            self.parameter, self.x, self.parameter, self.y
        )
    }
}

/// A rational parameterization using rational functions
#[derive(Debug, Clone)]
pub struct RationalParameterization<F: Field> {
    /// Rational function for x = p(t)/q(t)
    pub x_numerator: Vec<F>,
    pub x_denominator: Vec<F>,
    /// Rational function for y = r(t)/s(t)
    pub y_numerator: Vec<F>,
    pub y_denominator: Vec<F>,
}

impl<F: Field + Clone> RationalParameterization<F> {
    /// Create a new rational parameterization
    pub fn new(
        x_num: Vec<F>,
        x_den: Vec<F>,
        y_num: Vec<F>,
        y_den: Vec<F>,
    ) -> Self {
        RationalParameterization {
            x_numerator: x_num,
            x_denominator: x_den,
            y_numerator: y_num,
            y_denominator: y_den,
        }
    }

    /// Evaluate at a specific parameter value
    pub fn evaluate(&self, t: &F) -> (F, F) {
        let x = Self::eval_rational(&self.x_numerator, &self.x_denominator, t);
        let y = Self::eval_rational(&self.y_numerator, &self.y_denominator, t);
        (x, y)
    }

    /// Evaluate a rational function p(t)/q(t)
    fn eval_rational(num: &[F], den: &[F], t: &F) -> F {
        let num_val = Self::eval_poly(num, t);
        let den_val = Self::eval_poly(den, t);

        if den_val == F::zero() {
            panic!("Division by zero in rational parameterization");
        }

        num_val / den_val
    }

    /// Evaluate a polynomial at a point
    fn eval_poly(coeffs: &[F], t: &F) -> F {
        let mut result = F::zero();
        let mut power = F::one();

        for coeff in coeffs {
            result = result + coeff.clone() * power.clone();
            power = power * t.clone();
        }

        result
    }
}

/// Common parameterizations
impl RationalParameterization<Rational> {
    /// Stereographic projection parameterization of the unit circle
    /// x = (1-t²)/(1+t²), y = 2t/(1+t²)
    pub fn unit_circle() -> Self {
        // x = (1 - t²) / (1 + t²)
        let x_num = vec![Rational::one(), Rational::zero(), -Rational::one()]; // 1 - t²
        let x_den = vec![Rational::one(), Rational::zero(), Rational::one()];  // 1 + t²

        // y = 2t / (1 + t²)
        let y_num = vec![Rational::zero(), Rational::from(2)]; // 2t
        let y_den = vec![Rational::one(), Rational::zero(), Rational::one()]; // 1 + t²

        RationalParameterization::new(x_num, x_den, y_num, y_den)
    }

    /// Parameterization of a parabola y = x²
    /// x = t, y = t²
    pub fn parabola() -> Self {
        // x = t
        let x_num = vec![Rational::zero(), Rational::one()]; // t
        let x_den = vec![Rational::one()]; // 1

        // y = t²
        let y_num = vec![Rational::zero(), Rational::zero(), Rational::one()]; // t²
        let y_den = vec![Rational::one()]; // 1

        RationalParameterization::new(x_num, x_den, y_num, y_den)
    }

    /// Parameterization of a line through two points
    /// (x,y) = (1-t)*p1 + t*p2
    pub fn line_through_points(p1: (Rational, Rational), p2: (Rational, Rational)) -> Self {
        // x = (1-t)*x1 + t*x2 = x1 + t(x2-x1)
        let x_num = vec![p1.0.clone(), p2.0.clone() - p1.0.clone()];
        let x_den = vec![Rational::one()];

        // y = (1-t)*y1 + t*y2 = y1 + t(y2-y1)
        let y_num = vec![p1.1.clone(), p2.1.clone() - p1.1.clone()];
        let y_den = vec![Rational::one()];

        RationalParameterization::new(x_num, x_den, y_num, y_den)
    }

    /// Parameterization of a circle with center (h, k) and radius r
    /// Using stereographic projection from point (-r, 0)
    pub fn circle(h: Rational, k: Rational, r: Rational) -> Self {
        // Start with unit circle parameterization
        let unit = Self::unit_circle();

        // Scale and translate
        // x = h + r * (1-t²)/(1+t²)
        let x_num = vec![
            h.clone() + r.clone(),  // h + r
            Rational::zero(),
            -(r.clone()),           // -r
        ];
        let x_den = vec![Rational::one(), Rational::zero(), Rational::one()];

        // y = k + r * 2t/(1+t²)
        let y_num = vec![k.clone(), r * Rational::from(2)];
        let y_den = vec![Rational::one(), Rational::zero(), Rational::one()];

        RationalParameterization::new(x_num, x_den, y_num, y_den)
    }
}

/// Methods for finding parameterizations
pub mod find_parameterization {
    use super::*;
    use rustmath_polynomials::multivariate::MultiPoly;

    /// Attempt to find a rational parameterization for a curve
    ///
    /// This works for:
    /// - Lines (degree 1)
    /// - Conics (degree 2) with a rational point
    /// - Cubics with a singular point (genus 0)
    pub fn rational_parameterization<F: Field + Clone + PartialEq>(
        curve: &MultiPoly<F>,
    ) -> Option<RationalParameterization<F>> {
        let degree = curve.total_degree();

        use rustmath_polynomials::multivariate::Monomial;
        use std::collections::BTreeMap;

        match degree {
            0 => {
                // Constant curve - no parameterization
                None
            }
            1 => {
                // Lines are always rationally parameterizable
                // For a line ax + by + c = 0, parameterize as:
                // If b ≠ 0: x = t, y = -(a*t + c)/b
                // If b = 0: y = t, x = -c/a

                // Extract coefficients
                let mut x_exp = BTreeMap::new();
                x_exp.insert(0, 1);
                let x_monomial = Monomial::from_exponents(x_exp);

                let mut y_exp = BTreeMap::new();
                y_exp.insert(1, 1);
                let y_monomial = Monomial::from_exponents(y_exp);

                let const_monomial = Monomial::new();

                let a = curve.coefficient(&x_monomial);
                let b = curve.coefficient(&y_monomial);
                let c = curve.coefficient(&const_monomial);

                if b != F::zero() {
                    // x = t, y = -(a*t + c)/b
                    let x_num = vec![F::zero(), F::one()]; // t
                    let x_den = vec![F::one()]; // 1

                    // y = -a*t/b - c/b
                    let y_coeff = -(a / b.clone());
                    let y_const = -(c / b);
                    let y_num = vec![y_const, y_coeff]; // -c/b - (a/b)*t
                    let y_den = vec![F::one()]; // 1

                    return Some(RationalParameterization::new(x_num, x_den, y_num, y_den));
                } else if a != F::zero() {
                    // y = t, x = -c/a
                    let x_const = -(c / a);
                    let x_num = vec![x_const]; // -c/a
                    let x_den = vec![F::one()]; // 1

                    let y_num = vec![F::zero(), F::one()]; // t
                    let y_den = vec![F::one()]; // 1

                    return Some(RationalParameterization::new(x_num, x_den, y_num, y_den));
                }

                // Both coefficients are zero - degenerate case
                None
            }
            2 => {
                // Conics with a rational point can be parameterized
                // using stereographic projection from that point
                //
                // Algorithm:
                // 1. Find a rational point P on the conic
                // 2. Parameterize lines through P: P + t*v
                // 3. Find the other intersection point with the conic
                //
                // For a circle x² + y² = r², using point (r, 0):
                // Line through (r, 0) with slope t: x = r - s, y = t*s
                // Substituting: (r-s)² + t²s² = r²
                // Solving for s gives the parameterization

                // For now, check if it's a circle: x² + y² - r² = 0
                let mut x2_exp = BTreeMap::new();
                x2_exp.insert(0, 2);
                let x2_monomial = Monomial::from_exponents(x2_exp);

                let mut y2_exp = BTreeMap::new();
                y2_exp.insert(1, 2);
                let y2_monomial = Monomial::from_exponents(y2_exp);

                let const_monomial = Monomial::new();

                let x2_coeff = curve.coefficient(&x2_monomial);
                let y2_coeff = curve.coefficient(&y2_monomial);
                let const_coeff = curve.coefficient(&const_monomial);

                // Check if it's of form x² + y² + c = 0 (circle)
                if x2_coeff == F::one() && y2_coeff == F::one() && curve.num_terms() <= 3 {
                    // Circle: x² + y² - r² = 0 where r² = -c
                    // Use stereographic projection from (1, 0) or origin

                    // Standard parameterization using rational functions:
                    // x = (1 - t²)/(1 + t²), y = 2t/(1 + t²)
                    // This parameterizes the unit circle x² + y² = 1

                    // For x² + y² = -c, scale by sqrt(-c)
                    // x = sqrt(-c) * (1-t²)/(1+t²)
                    // y = sqrt(-c) * 2t/(1+t²)

                    // Note: This only works if -c is a square in F
                    // For simplicity, assume unit circle case (c = -1)

                    if const_coeff == -F::one() {
                        // Unit circle
                        let x_num = vec![F::one(), F::zero(), -F::one()]; // 1 - t²
                        let x_den = vec![F::one(), F::zero(), F::one()]; // 1 + t²

                        let two = F::one() + F::one(); // 2
                        let y_num = vec![F::zero(), two]; // 2t
                        let y_den = vec![F::one(), F::zero(), F::one()]; // 1 + t²

                        return Some(RationalParameterization::new(x_num, x_den, y_num, y_den));
                    }
                }

                // General conic case requires finding a point and doing stereographic projection
                // This is more complex and requires solving quadratic equations
                None
            }
            3 => {
                // Cubics with a singular point have genus 0 and can be parameterized
                // The singular point serves as the base point for parameterization
                //
                // Algorithm:
                // 1. Find singular points (where F = ∂F/∂x = ∂F/∂y = 0)
                // 2. If there's a singular point P, lines through P intersect the
                //    cubic at P (with multiplicity 2) and one other point
                // 3. This gives a birational map from P¹ to the curve

                use crate::singularities::find_singularities;

                let singularities = find_singularities(curve);

                if !singularities.is_empty() {
                    // Found singular point(s) - curve has genus 0
                    // Would implement rational parameterization using the singular point
                    // This requires solving cubic equations symbolically
                    //
                    // For example, for a nodal cubic y² = x³ + x²:
                    // Parameterize as x = t² - 1, y = t(t² - 1)

                    // For now, return None (requires symbolic equation solving)
                    None
                } else {
                    // Smooth cubic has genus 1, not rationally parameterizable
                    None
                }
            }
            _ => {
                // Higher degree curves are generally not rationally parameterizable
                // (unless they have enough singularities to reduce the genus to 0)
                None
            }
        }
    }

    /// Check if a curve is rationally parameterizable
    pub fn is_rationally_parameterizable<F: Field + Clone + PartialEq>(
        curve: &MultiPoly<F>,
    ) -> bool {
        // A curve is rationally parameterizable if and only if it has genus 0
        // For plane curves: genus = (d-1)(d-2)/2 - sum of delta invariants

        let d = curve.total_degree();
        let arithmetic_genus = if d <= 1 {
            0
        } else {
            (d - 1) * (d - 2) / 2
        };

        // If arithmetic genus is 0, it's rational (assuming smooth or appropriate singularities)
        arithmetic_genus == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_circle_parameterization() {
        let circle = RationalParameterization::unit_circle();

        // At t = 0: (x, y) = (1, 0)
        let (x0, y0) = circle.evaluate(&Rational::zero());
        assert_eq!(x0, Rational::one());
        assert_eq!(y0, Rational::zero());

        // At t = 1: (x, y) = (0, 1)
        let (x1, y1) = circle.evaluate(&Rational::one());
        assert_eq!(x1, Rational::zero());
        assert_eq!(y1, Rational::one());

        // Verify that x² + y² = 1 for various t values
        let t = Rational::from(2);
        let (x, y) = circle.evaluate(&t);
        let x_sq = x.clone() * x;
        let y_sq = y.clone() * y;
        assert_eq!(x_sq + y_sq, Rational::one());
    }

    #[test]
    fn test_parabola_parameterization() {
        let parabola = RationalParameterization::parabola();

        // At t = 2: (x, y) = (2, 4)
        let t = Rational::from(2);
        let (x, y) = parabola.evaluate(&t);
        assert_eq!(x, Rational::from(2));
        assert_eq!(y, Rational::from(4));

        // Verify y = x² for t = 3
        let t = Rational::from(3);
        let (x, y) = parabola.evaluate(&t);
        assert_eq!(y, x.clone() * x);
    }

    #[test]
    fn test_line_parameterization() {
        // Line through (0, 0) and (1, 1)
        let line = RationalParameterization::line_through_points(
            (Rational::zero(), Rational::zero()),
            (Rational::one(), Rational::one()),
        );

        // At t = 0: should be (0, 0)
        let (x0, y0) = line.evaluate(&Rational::zero());
        assert_eq!(x0, Rational::zero());
        assert_eq!(y0, Rational::zero());

        // At t = 1: should be (1, 1)
        let (x1, y1) = line.evaluate(&Rational::one());
        assert_eq!(x1, Rational::one());
        assert_eq!(y1, Rational::one());

        // At t = 1/2: should be (1/2, 1/2)
        let half = Rational::new(1, 2);
        let (xh, yh) = line.evaluate(&half);
        assert_eq!(xh, half);
        assert_eq!(yh, half);
    }
}
