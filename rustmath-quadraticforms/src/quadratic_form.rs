//! Quadratic Forms
//!
//! Implementation of quadratic forms, reduction algorithms, and related theory.

use rustmath_integers::Integer;
use rustmath_core::Ring;
use std::fmt;

/// Represents a binary quadratic form ax² + bxy + cy²
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadraticForm {
    pub a: Integer,
    pub b: Integer,
    pub c: Integer,
    discriminant: Integer,
}

impl QuadraticForm {
    /// Create a new quadratic form ax² + bxy + cy²
    pub fn new(a: Integer, b: Integer, c: Integer) -> Self {
        let discriminant = b.clone() * b.clone() - Integer::from(4) * a.clone() * c.clone();
        Self { a, b, c, discriminant }
    }

    /// Create from a slice of coefficients
    /// - [a] creates unary form a*x²
    /// - [a, c] creates diagonal binary form a*x² + c*y²
    /// - [a, b, c] creates general binary form a*x² + b*xy + c*y²
    pub fn from_coefficients(coefficients: &[Integer]) -> Self {
        match coefficients.len() {
            1 => Self::unary(coefficients[0].clone()),
            2 => Self::diagonal_binary(coefficients[0].clone(), coefficients[1].clone()),
            3 => Self::binary(coefficients[0].clone(), coefficients[1].clone(), coefficients[2].clone()),
            _ => panic!("Only unary and binary forms supported"),
        }
    }

    /// Create a unary quadratic form a*x²
    pub fn unary(a: Integer) -> Self {
        Self::new(a, Integer::zero(), Integer::zero())
    }

    /// Create a diagonal binary quadratic form a*x² + c*y²
    pub fn diagonal_binary(a: Integer, c: Integer) -> Self {
        Self::new(a, Integer::zero(), c)
    }

    /// Create a general binary quadratic form a*x² + b*xy + c*y²
    pub fn binary(a: Integer, b: Integer, c: Integer) -> Self {
        Self::new(a, b, c)
    }

    /// Evaluate the form at the point (x, y)
    pub fn evaluate(&self, x: &Integer, y: &Integer) -> Integer {
        self.a.clone() * x.clone() * x.clone() +
        self.b.clone() * x.clone() * y.clone() +
        self.c.clone() * y.clone() * y.clone()
    }

    /// Get the discriminant Δ = b² - 4ac
    pub fn discriminant(&self) -> &Integer {
        &self.discriminant
    }

    /// Check if the form is positive definite (a > 0 and Δ < 0)
    pub fn is_positive_definite(&self) -> bool {
        self.a > Integer::zero() && self.discriminant < Integer::zero()
    }

    /// Check if the form is negative definite (a < 0 and Δ < 0)
    pub fn is_negative_definite(&self) -> bool {
        self.a < Integer::zero() && self.discriminant < Integer::zero()
    }

    /// Check if the form is indefinite (Δ > 0)
    pub fn is_indefinite(&self) -> bool {
        self.discriminant > Integer::zero()
    }

    /// Check if the form is degenerate (Δ = 0)
    pub fn is_degenerate(&self) -> bool {
        self.discriminant.is_zero()
    }

    /// Check if the form is primitive (gcd(a, b, c) = 1)
    pub fn is_primitive(&self) -> bool {
        let gcd1 = self.a.gcd(&self.b);
        let gcd_total = gcd1.gcd(&self.c);
        gcd_total.is_one()
    }

    /// Compute the reduced form for positive definite binary quadratic forms
    ///
    /// A form [a, b, c] is reduced if:
    /// - |b| ≤ a ≤ c
    /// - If |b| = a or a = c, then b ≥ 0
    pub fn reduced_form(&self) -> Self {
        if !self.is_positive_definite() {
            return self.clone();
        }

        let mut current = self.clone();
        let mut iterations = 0;
        let max_iterations = 1000; // Safety limit

        loop {
            if iterations >= max_iterations {
                // Return current form if we've iterated too long
                return current;
            }
            iterations += 1;

            let a = current.a.clone();
            let b = current.b.clone();
            let c = current.c.clone();

            // Check if already reduced
            let b_abs = b.abs();
            if &b_abs <= &a && &a <= &c {
                // Check boundary conditions
                if &b_abs == &a || &a == &c {
                    if b >= Integer::zero() {
                        return current;
                    }
                } else {
                    return current;
                }
            }

            // Apply reduction step
            if &b_abs > &a {
                // Reduce b using: (x,y) → (x - ky, y) where k = round(b/(2a))
                let two_a = Integer::from(2) * a.clone();
                let k = if b >= Integer::zero() {
                    (b.clone() + a.clone()) / two_a.clone()  // Round b/(2a)
                } else {
                    (b.clone() - a.clone()) / two_a
                };
                let new_b = b.clone() - Integer::from(2) * k.clone() * a.clone();
                let new_c = a.clone() * k.clone() * k.clone() - b.clone() * k + c.clone();
                current = Self::new(a, new_b, new_c);
            } else if &a > &c {
                // Swap a and c: (x,y) → (y,x)
                current = Self::new(c, -b.clone(), a);
            } else if &b < &Integer::zero() && &b_abs <= &a {
                // Make b non-negative
                current = Self::new(a, -b, c);
            } else {
                // Shouldn't reach here if logic is correct
                break;
            }
        }

        current
    }

    /// Estimate the class number for this discriminant
    /// Uses a simplified approximation of Dirichlet's class number formula
    pub fn class_number_estimate(&self) -> Integer {
        let d = (-self.discriminant()).clone();

        if d <= Integer::zero() {
            return Integer::one();
        }

        // Rough approximation: h(d) ≈ sqrt(|d|) / (π * log(|d|))
        // For demonstration, we use a simpler estimate
        let sqrt_d = Self::integer_sqrt(&d);
        let estimate = sqrt_d / Integer::from(3);
        if estimate > Integer::one() {
            estimate
        } else {
            Integer::one()
        }
    }

    /// Compute integer square root using Newton's method
    fn integer_sqrt(n: &Integer) -> Integer {
        if n < &Integer::zero() {
            panic!("Cannot compute square root of negative number");
        }

        if n.is_zero() {
            return Integer::zero();
        }

        let mut x = n.clone();
        let mut y = (x.clone() + Integer::one()) / Integer::from(2);

        while &y < &x {
            x = y.clone();
            y = (x.clone() + n.clone() / x.clone()) / Integer::from(2);
        }

        x
    }

    /// Check if two forms are equivalent (same orbit under SL₂(Z))
    /// This is a simplified check - full equivalence testing is complex
    pub fn is_equivalent_to(&self, other: &QuadraticForm) -> bool {
        // Same discriminant is necessary
        if self.discriminant() != other.discriminant() {
            return false;
        }

        // For positive definite forms, compare reduced forms
        if self.is_positive_definite() && other.is_positive_definite() {
            let self_reduced = self.reduced_form();
            let other_reduced = other.reduced_form();
            return self_reduced == other_reduced;
        }

        // For indefinite forms, the theory is more complex
        // We just check if they're literally equal for now
        self == other
    }
}

impl fmt::Display for QuadraticForm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.a, self.b, self.c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_form() {
        let form = QuadraticForm::new(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );

        assert_eq!(form.a, Integer::from(1));
        assert_eq!(form.b, Integer::from(0));
        assert_eq!(form.c, Integer::from(1));
        assert_eq!(*form.discriminant(), Integer::from(-4));
    }

    #[test]
    fn test_evaluate() {
        let form = QuadraticForm::new(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );

        // 1*2² + 0*2*3 + 1*3² = 4 + 9 = 13
        assert_eq!(
            form.evaluate(&Integer::from(2), &Integer::from(3)),
            Integer::from(13)
        );
    }

    #[test]
    fn test_positive_definite() {
        let form = QuadraticForm::new(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        assert!(form.is_positive_definite());

        let form2 = QuadraticForm::new(
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
        );
        assert!(form2.is_positive_definite());
    }

    #[test]
    fn test_indefinite() {
        let form = QuadraticForm::new(
            Integer::from(1),
            Integer::from(0),
            Integer::from(-1),
        );
        assert!(form.is_indefinite());
    }

    #[test]
    fn test_primitive() {
        let form = QuadraticForm::new(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        assert!(form.is_primitive());

        let form2 = QuadraticForm::new(
            Integer::from(2),
            Integer::from(4),
            Integer::from(6),
        );
        assert!(!form2.is_primitive());
    }

    #[test]
    fn test_reduced_form() {
        // x² + y² is already reduced
        let form = QuadraticForm::new(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let reduced = form.reduced_form();
        assert_eq!(reduced, form);

        // x² + xy + y² is already reduced
        let form2 = QuadraticForm::new(
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
        );
        let reduced2 = form2.reduced_form();
        assert_eq!(reduced2, form2);
    }

    #[test]
    fn test_integer_sqrt() {
        assert_eq!(QuadraticForm::integer_sqrt(&Integer::from(0)), Integer::from(0));
        assert_eq!(QuadraticForm::integer_sqrt(&Integer::from(1)), Integer::from(1));
        assert_eq!(QuadraticForm::integer_sqrt(&Integer::from(4)), Integer::from(2));
        assert_eq!(QuadraticForm::integer_sqrt(&Integer::from(9)), Integer::from(3));
        assert_eq!(QuadraticForm::integer_sqrt(&Integer::from(15)), Integer::from(3)); // Floor of sqrt(15)
        assert_eq!(QuadraticForm::integer_sqrt(&Integer::from(16)), Integer::from(4));
    }
}
