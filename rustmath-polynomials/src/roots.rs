//! Polynomial root finding algorithms

use crate::univariate::UnivariatePolynomial;
use rustmath_core::{NumericConversion, Result, MathError};
use rustmath_rationals::Rational;

/// Find rational roots of a polynomial using the rational root theorem
///
/// For a polynomial with integer coefficients, any rational root p/q
/// must have p dividing the constant term and q dividing the leading coefficient.
pub fn rational_roots(poly: &UnivariatePolynomial<rustmath_integers::Integer>) -> Vec<Rational> {
    use rustmath_integers::Integer;

    if poly.is_zero() {
        return vec![];
    }

    let Some(degree) = poly.degree() else {
        return vec![];
    };

    if degree == 0 {
        return vec![];
    }

    let constant = poly.coeff(0);
    let leading = poly.coeff(degree);

    if constant.is_zero() {
        // 0 is a root
        let mut roots = vec![Rational::from_integer(Integer::zero())];
        // Factor out x and find roots of the remaining polynomial
        let mut new_coeffs = Vec::new();
        for i in 1..=degree {
            new_coeffs.push(poly.coeff(i).clone());
        }
        let reduced = UnivariatePolynomial::new(new_coeffs);
        roots.extend(rational_roots(&reduced));
        return roots;
    }

    // Get divisors of constant term and leading coefficient
    let const_divisors = constant.divisors().unwrap_or_default();
    let lead_divisors = leading.divisors().unwrap_or_default();

    let mut roots = Vec::new();

    // Try all possible p/q combinations
    for p in &const_divisors {
        for q in &lead_divisors {
            // Try both positive and negative
            for &sign in &[1, -1] {
                let p_signed = Integer::from(sign) * p.clone();
                let candidate = Rational::new(p_signed, q.clone()).unwrap();

                // Evaluate polynomial at candidate
                if eval_rational(poly, &candidate).numerator().is_zero() {
                    if !roots.contains(&candidate) {
                        roots.push(candidate.clone());
                    }
                }
            }
        }
    }

    roots
}

/// Evaluate a polynomial with integer coefficients at a rational point
fn eval_rational(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
    x: &Rational,
) -> Rational {
    use rustmath_integers::Integer;

    let coeffs = poly.coefficients();
    let mut result = Rational::from_integer(Integer::zero());
    let mut power = Rational::from_integer(Integer::one());

    for coeff in coeffs {
        let term = Rational::from_integer(coeff.clone()) * power.clone();
        result = result + term;
        power = power * x.clone();
    }

    result
}

/// Represents roots of a quadratic equation
#[derive(Debug, Clone, PartialEq)]
pub enum QuadraticRoots {
    /// Two distinct real rational roots
    TwoRational(Rational, Rational),
    /// One repeated rational root
    OneRational(Rational),
    /// Two real roots involving a square root: (p ± sqrt(d)) / q
    TwoIrrational {
        p: rustmath_integers::Integer,
        d: rustmath_integers::Integer, // discriminant
        q: rustmath_integers::Integer,
    },
    /// Two complex conjugate roots (discriminant < 0)
    Complex {
        real: Rational,
        discriminant: rustmath_integers::Integer, // negative
    },
}

/// Solve a quadratic equation ax² + bx + c = 0
///
/// Uses the quadratic formula: x = (-b ± sqrt(b²-4ac)) / (2a)
pub fn solve_quadratic(
    a: rustmath_integers::Integer,
    b: rustmath_integers::Integer,
    c: rustmath_integers::Integer,
) -> Result<QuadraticRoots> {
    use rustmath_integers::Integer;

    if a.is_zero() {
        return Err(MathError::InvalidArgument(
            "Leading coefficient cannot be zero".to_string(),
        ));
    }

    // Compute discriminant: b² - 4ac
    let b_squared = b.clone() * b.clone();
    let four_ac = Integer::from(4) * a.clone() * c.clone();
    let discriminant = b_squared - four_ac;

    if discriminant.is_zero() {
        // One repeated root: -b / (2a)
        let numerator = -b;
        let denominator = Integer::from(2) * a;
        let root = Rational::new(numerator, denominator)?;
        Ok(QuadraticRoots::OneRational(root))
    } else if discriminant.signum() > 0 {
        // Check if discriminant is a perfect square
        let sqrt_d = discriminant.sqrt()?;
        if sqrt_d.clone() * sqrt_d.clone() == discriminant {
            // Rational roots
            let root1_num = -b.clone() + sqrt_d.clone();
            let root2_num = -b.clone() - sqrt_d;
            let denom = Integer::from(2) * a;

            let root1 = Rational::new(root1_num, denom.clone())?;
            let root2 = Rational::new(root2_num, denom)?;

            Ok(QuadraticRoots::TwoRational(root1, root2))
        } else {
            // Irrational roots: (-b ± sqrt(d)) / (2a)
            Ok(QuadraticRoots::TwoIrrational {
                p: -b,
                d: discriminant,
                q: Integer::from(2) * a,
            })
        }
    } else {
        // Complex roots
        let real_part = Rational::new(-b, Integer::from(2) * a)?;
        Ok(QuadraticRoots::Complex {
            real: real_part,
            discriminant,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_solve_quadratic_rational() {
        // x² - 5x + 6 = (x-2)(x-3) = 0
        let roots = solve_quadratic(Integer::from(1), Integer::from(-5), Integer::from(6)).unwrap();
        match roots {
            QuadraticRoots::TwoRational(r1, r2) => {
                assert!(
                    (r1 == Rational::new(2, 1).unwrap() && r2 == Rational::new(3, 1).unwrap())
                        || (r1 == Rational::new(3, 1).unwrap()
                            && r2 == Rational::new(2, 1).unwrap())
                );
            }
            _ => panic!("Expected two rational roots"),
        }
    }

    #[test]
    fn test_solve_quadratic_repeated() {
        // x² - 4x + 4 = (x-2)² = 0
        let roots = solve_quadratic(Integer::from(1), Integer::from(-4), Integer::from(4)).unwrap();
        match roots {
            QuadraticRoots::OneRational(r) => {
                assert_eq!(r, Rational::new(2, 1).unwrap());
            }
            _ => panic!("Expected one repeated root"),
        }
    }

    #[test]
    fn test_solve_quadratic_irrational() {
        // x² - 2 = 0, roots = ±√2
        let roots = solve_quadratic(Integer::from(1), Integer::from(0), Integer::from(-2)).unwrap();
        match roots {
            QuadraticRoots::TwoIrrational { p, d, q } => {
                assert_eq!(p, Integer::from(0));
                assert_eq!(d, Integer::from(8)); // discriminant = 0² - 4(1)(-2) = 8
                assert_eq!(q, Integer::from(2));
            }
            _ => panic!("Expected irrational roots"),
        }
    }

    #[test]
    fn test_solve_quadratic_complex() {
        // x² + 1 = 0, roots = ±i
        let roots = solve_quadratic(Integer::from(1), Integer::from(0), Integer::from(1)).unwrap();
        match roots {
            QuadraticRoots::Complex { real, discriminant } => {
                assert_eq!(real, Rational::new(0, 1).unwrap());
                assert_eq!(discriminant, Integer::from(-4));
            }
            _ => panic!("Expected complex roots"),
        }
    }

    #[test]
    fn test_rational_roots_simple() {
        // x² - 5x + 6 = (x-2)(x-3)
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(6),
            Integer::from(-5),
            Integer::from(1),
        ]);

        let mut roots = rational_roots(&poly);
        roots.sort_by(|a, b| {
            let diff = a.clone() - b.clone();
            if diff.numerator().signum() < 0 {
                std::cmp::Ordering::Less
            } else if diff.numerator().is_zero() {
                std::cmp::Ordering::Equal
            } else {
                std::cmp::Ordering::Greater
            }
        });

        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0], Rational::new(2, 1).unwrap());
        assert_eq!(roots[1], Rational::new(3, 1).unwrap());
    }

    #[test]
    fn test_rational_roots_with_zero() {
        // x² - 4x = x(x-4)
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(0),
            Integer::from(-4),
            Integer::from(1),
        ]);

        let roots = rational_roots(&poly);
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&Rational::new(0, 1).unwrap()));
        assert!(roots.contains(&Rational::new(4, 1).unwrap()));
    }
}
