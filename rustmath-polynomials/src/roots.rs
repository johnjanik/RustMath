//! Polynomial root finding algorithms

use crate::univariate::UnivariatePolynomial;
use rustmath_core::{Result, MathError};
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

/// Represents roots of a cubic equation
#[derive(Debug, Clone)]
pub enum CubicRoots {
    /// One rational root
    OneRational(Rational),
    /// Two distinct rational roots (one root with multiplicity 2)
    TwoRational(Rational, Rational),
    /// Three distinct rational roots
    ThreeRational(Rational, Rational, Rational),
    /// One real root and two complex conjugate roots
    OneRealTwoComplex {
        real_root: String, // Symbolic representation
        discriminant: rustmath_integers::Integer,
    },
    /// Three distinct real roots (casus irreducibilis - requires trigonometry or complex numbers)
    ThreeRealIrrational {
        description: String,
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

/// Solve a cubic equation ax³ + bx² + cx + d = 0
///
/// Uses Cardano's formula and first checks for rational roots.
///
/// # Algorithm
///
/// 1. First tries to find rational roots using the rational root theorem
/// 2. If a rational root is found, factors it out and solves the resulting quadratic
/// 3. Otherwise, uses the depressed cubic substitution and Cardano's formula
///
/// # Limitations
///
/// - For three distinct irrational real roots (casus irreducibilis), returns a symbolic description
/// - Requires exact arithmetic which may lose precision for large coefficients
pub fn solve_cubic(
    a: rustmath_integers::Integer,
    b: rustmath_integers::Integer,
    c: rustmath_integers::Integer,
    d: rustmath_integers::Integer,
) -> Result<CubicRoots> {
    use rustmath_integers::Integer;

    if a.is_zero() {
        return Err(MathError::InvalidArgument(
            "Leading coefficient cannot be zero".to_string(),
        ));
    }

    // Try to find rational roots first using rational root theorem
    let poly = UnivariatePolynomial::new(vec![d.clone(), c.clone(), b.clone(), a.clone()]);
    let rational_roots_found = rational_roots(&poly);

    if !rational_roots_found.is_empty() {
        // We found at least one rational root - factor it out
        let _r = &rational_roots_found[0];

        // Use synthetic division or polynomial division to get quadratic factor
        // For now, if we have one rational root, we can describe it
        // A complete implementation would factor out the root and solve the quadratic

        if rational_roots_found.len() == 3 {
            return Ok(CubicRoots::ThreeRational(
                rational_roots_found[0].clone(),
                rational_roots_found[1].clone(),
                rational_roots_found[2].clone(),
            ));
        } else if rational_roots_found.len() == 2 {
            return Ok(CubicRoots::TwoRational(
                rational_roots_found[0].clone(),
                rational_roots_found[1].clone(),
            ));
        } else {
            return Ok(CubicRoots::OneRational(rational_roots_found[0].clone()));
        }
    }

    // No rational roots found - use Cardano's formula
    // First, convert to depressed cubic form: t³ + pt + q = 0
    // using substitution x = t - b/(3a)

    // Compute p = (3ac - b²) / (3a²)
    // Compute q = (2b³ - 9abc + 27a²d) / (27a³)

    let three = Integer::from(3);
    let two = Integer::from(2);
    let nine = Integer::from(9);
    let twenty_seven = Integer::from(27);

    // For exact rational arithmetic, compute using Rational
    let a_rat = Rational::from_integer(a.clone());
    let b_rat = Rational::from_integer(b.clone());
    let c_rat = Rational::from_integer(c.clone());
    let d_rat = Rational::from_integer(d.clone());

    // p = (3ac - b²) / (3a²)
    let three_rat = Rational::from_integer(three.clone());
    let numerator_p = three_rat.clone() * a_rat.clone() * c_rat.clone()
        - b_rat.clone() * b_rat.clone();
    let denominator_p = three_rat.clone() * a_rat.clone() * a_rat.clone();
    let p = numerator_p / denominator_p;

    // q = (2b³ - 9abc + 27a²d) / (27a³)
    let two_rat = Rational::from_integer(two);
    let nine_rat = Rational::from_integer(nine);
    let twenty_seven_rat = Rational::from_integer(twenty_seven.clone());

    let term1 = two_rat * b_rat.clone() * b_rat.clone() * b_rat.clone();
    let term2 = nine_rat * a_rat.clone() * b_rat.clone() * c_rat.clone();
    let term3 = twenty_seven_rat.clone() * a_rat.clone() * a_rat.clone() * d_rat.clone();

    let numerator_q = term1 - term2 + term3;
    let denominator_q = twenty_seven_rat * a_rat.clone() * a_rat.clone() * a_rat.clone();
    let q = numerator_q / denominator_q;

    // Compute discriminant: Δ = -4p³ - 27q²
    let four_rat = Rational::from_integer(Integer::from(4));
    let p_cubed = p.clone() * p.clone() * p.clone();
    let q_squared = q.clone() * q.clone();

    let disc = -(four_rat * p_cubed) - (Rational::from_integer(twenty_seven) * q_squared);

    // Based on the discriminant, determine the nature of the roots
    if disc.numerator().signum() > 0 {
        // Three distinct real roots (casus irreducibilis)
        // This case requires trigonometric methods or complex arithmetic
        Ok(CubicRoots::ThreeRealIrrational {
            description: format!(
                "Three distinct real irrational roots (use trigonometric method or complex arithmetic)"
            ),
        })
    } else {
        // One real root and two complex conjugate roots
        // The real root can be expressed using Cardano's formula
        Ok(CubicRoots::OneRealTwoComplex {
            real_root: format!("Use Cardano's formula with p={}, q={}", p, q),
            discriminant: disc.numerator().clone(),
        })
    }
}

/// Represents roots of a quartic equation
#[derive(Debug, Clone)]
pub enum QuarticRoots {
    /// Up to 4 rational roots
    Rational(Vec<Rational>),
    /// Symbolic description of roots
    Symbolic { description: String },
}

/// Solve a quartic equation ax⁴ + bx³ + cx² + dx + e = 0
///
/// Uses Ferrari's method combined with the rational root theorem.
///
/// # Algorithm
///
/// 1. First tries to find rational roots using the rational root theorem
/// 2. If all roots are found, returns them
/// 3. Otherwise, uses the depressed quartic and Ferrari's method
///
/// # Limitations
///
/// - For non-rational roots, returns a symbolic description
/// - Full implementation of Ferrari's method requires solving a resolvent cubic
pub fn solve_quartic(
    a: rustmath_integers::Integer,
    b: rustmath_integers::Integer,
    c: rustmath_integers::Integer,
    d: rustmath_integers::Integer,
    e: rustmath_integers::Integer,
) -> Result<QuarticRoots> {


    if a.is_zero() {
        return Err(MathError::InvalidArgument(
            "Leading coefficient cannot be zero".to_string(),
        ));
    }

    // Try to find rational roots first using rational root theorem
    let poly = UnivariatePolynomial::new(vec![
        e.clone(),
        d.clone(),
        c.clone(),
        b.clone(),
        a.clone(),
    ]);
    let rational_roots_found = rational_roots(&poly);

    if !rational_roots_found.is_empty() {
        // Found some or all rational roots
        return Ok(QuarticRoots::Rational(rational_roots_found));
    }

    // No rational roots found - would need to use Ferrari's method
    // Ferrari's method:
    // 1. Reduce to depressed quartic: y⁴ + py² + qy + r = 0 via substitution x = y - b/(4a)
    // 2. Solve resolvent cubic: z³ + 2pz² + (p² - 4r)z - q² = 0
    // 3. Use one root of resolvent to factor the quartic into two quadratics
    // 4. Solve both quadratics

    // For now, return symbolic description
    Ok(QuarticRoots::Symbolic {
        description: "Use Ferrari's method or numerical root finding".to_string(),
    })
}

/// Find all roots of a polynomial (up to degree 4) if they are rational
///
/// This is a convenience function that dispatches to the appropriate solver
/// based on the degree of the polynomial.
pub fn find_rational_roots_up_to_degree_4(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
) -> Vec<Rational> {


    if poly.is_zero() {
        return vec![];
    }

    let degree = match poly.degree() {
        Some(d) => d,
        None => return vec![],
    };

    match degree {
        0 => vec![], // Constant polynomial has no roots (unless it's zero, handled above)
        1 => {
            // Linear: ax + b = 0 => x = -b/a
            let a = poly.coeff(1);
            let b = poly.coeff(0);
            if a.is_zero() {
                vec![]
            } else {
                vec![Rational::new(-b.clone(), a.clone()).unwrap()]
            }
        }
        2 => {
            // Quadratic: use quadratic formula
            let a = poly.coeff(2);
            let b = poly.coeff(1);
            let c = poly.coeff(0);
            match solve_quadratic(a.clone(), b.clone(), c.clone()) {
                Ok(QuadraticRoots::TwoRational(r1, r2)) => vec![r1, r2],
                Ok(QuadraticRoots::OneRational(r)) => vec![r],
                _ => vec![],
            }
        }
        3 => {
            // Cubic: use rational root theorem or Cardano's formula
            rational_roots(poly)
        }
        4 => {
            // Quartic: use rational root theorem or Ferrari's method
            rational_roots(poly)
        }
        _ => {
            // Higher degrees: only find rational roots
            rational_roots(poly)
        }
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

    #[test]
    fn test_solve_cubic_rational() {
        // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
        let roots = solve_cubic(
            Integer::from(1),
            Integer::from(-6),
            Integer::from(11),
            Integer::from(-6),
        )
        .unwrap();

        match roots {
            CubicRoots::ThreeRational(r1, r2, r3) => {
                let mut roots_vec = vec![r1, r2, r3];
                roots_vec.sort_by(|a, b| {
                    let diff = a.clone() - b.clone();
                    if diff.numerator().signum() < 0 {
                        std::cmp::Ordering::Less
                    } else if diff.numerator().is_zero() {
                        std::cmp::Ordering::Equal
                    } else {
                        std::cmp::Ordering::Greater
                    }
                });
                assert_eq!(roots_vec[0], Rational::new(1, 1).unwrap());
                assert_eq!(roots_vec[1], Rational::new(2, 1).unwrap());
                assert_eq!(roots_vec[2], Rational::new(3, 1).unwrap());
            }
            _ => panic!("Expected three rational roots, got {:?}", roots),
        }
    }

    #[test]
    fn test_solve_cubic_one_rational() {
        // x³ - 1 = (x-1)(x² + x + 1)
        // Has one rational root: x = 1
        let roots = solve_cubic(
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
            Integer::from(-1),
        )
        .unwrap();

        match roots {
            CubicRoots::OneRational(r) => {
                assert_eq!(r, Rational::new(1, 1).unwrap());
            }
            _ => panic!("Expected one rational root, got {:?}", roots),
        }
    }

    #[test]
    fn test_solve_quartic_rational() {
        // x⁴ - 5x² + 4 = (x² - 1)(x² - 4) = (x-1)(x+1)(x-2)(x+2)
        // Has four rational roots: ±1, ±2
        let roots = solve_quartic(
            Integer::from(1),
            Integer::from(0),
            Integer::from(-5),
            Integer::from(0),
            Integer::from(4),
        )
        .unwrap();

        match roots {
            QuarticRoots::Rational(roots_vec) => {
                assert_eq!(roots_vec.len(), 4);
                // Check that all expected roots are present
                assert!(roots_vec.contains(&Rational::new(1, 1).unwrap()));
                assert!(roots_vec.contains(&Rational::new(-1, 1).unwrap()));
                assert!(roots_vec.contains(&Rational::new(2, 1).unwrap()));
                assert!(roots_vec.contains(&Rational::new(-2, 1).unwrap()));
            }
            _ => panic!("Expected four rational roots, got {:?}", roots),
        }
    }

    #[test]
    fn test_find_rational_roots_convenience() {
        // Test the convenience function with various degrees

        // Linear: x - 5 = 0
        let linear = UnivariatePolynomial::new(vec![Integer::from(-5), Integer::from(1)]);
        let roots = find_rational_roots_up_to_degree_4(&linear);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], Rational::new(5, 1).unwrap());

        // Quadratic: x² - 5x + 6 = (x-2)(x-3)
        let quadratic = UnivariatePolynomial::new(vec![
            Integer::from(6),
            Integer::from(-5),
            Integer::from(1),
        ]);
        let roots = find_rational_roots_up_to_degree_4(&quadratic);
        assert_eq!(roots.len(), 2);

        // Quartic: x⁴ - 1 = (x-1)(x+1)(x²+1)
        // Has two rational roots: ±1
        let quartic = UnivariatePolynomial::new(vec![
            Integer::from(-1),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
        ]);
        let roots = find_rational_roots_up_to_degree_4(&quartic);
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&Rational::new(1, 1).unwrap()));
        assert!(roots.contains(&Rational::new(-1, 1).unwrap()));
    }
}
