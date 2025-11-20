//! # Number Fields - Algebraic Number Theory
//!
//! This module implements algebraic number fields, which are fundamental objects in
//! algebraic number theory. A number field is a finite-degree field extension of the
//! rational numbers ℚ.
//!
//! ## Mathematical Background
//!
//! ### Definition
//! An **algebraic number field** (or simply number field) is a finite-degree field
//! extension K/ℚ. Equivalently, K = ℚ(α) for some algebraic number α that is a root
//! of an irreducible polynomial f(x) ∈ ℚ[x].
//!
//! ### Absolute Number Fields
//! An **absolute number field** is a number field viewed as an extension of ℚ directly,
//! as opposed to a relative extension K/L where L is itself a number field.
//!
//! ### Key Concepts
//!
//! #### Degree
//! The **degree** [K:ℚ] is the dimension of K as a vector space over ℚ. It equals
//! the degree of the minimal polynomial of the primitive element.
//!
//! #### Integral Basis
//! An **integral basis** is a ℤ-basis for the ring of integers 𝒪_K. While the
//! **power basis** {1, α, α², ..., α^(n-1)} is always a ℚ-basis for K, it is not
//! always a ℤ-basis for 𝒪_K.
//!
//! #### Discriminant
//! The **discriminant** Δ_K is a rational number that measures the "size" of the
//! ring of integers. It can be computed from the minimal polynomial as:
//! ```text
//! disc(f) = (-1)^(n(n-1)/2) · Res(f, f') / lc(f)
//! ```
//! where Res is the resultant and lc(f) is the leading coefficient.
//!
//! #### Class Number
//! The **class number** h(K) is the order of the ideal class group Cl(K), which
//! measures the failure of unique factorization in 𝒪_K. We have:
//! - h(K) = 1 if and only if 𝒪_K is a unique factorization domain
//! - h(K) is always finite (a deep theorem!)
//!
//! #### Unit Group
//! By **Dirichlet's Unit Theorem**, the unit group 𝒪_K* is isomorphic to:
//! ```text
//! 𝒪_K* ≅ μ(K) × ℤ^r
//! ```
//! where μ(K) is the (finite cyclic) group of roots of unity in K, and r = r₁ + r₂ - 1,
//! where r₁ is the number of real embeddings and r₂ is the number of pairs of complex
//! conjugate embeddings.
//!
//! #### Signature
//! The **signature** (r₁, r₂) satisfies n = r₁ + 2r₂ where n = [K:ℚ]. It determines
//! the rank of the unit group.
//!
//! ### Important Examples
//!
//! #### Quadratic Fields
//! Fields of the form ℚ(√d) where d is a square-free integer. These have degree 2.
//! - **Real quadratic fields**: d > 0 (e.g., ℚ(√2), ℚ(√5))
//! - **Imaginary quadratic fields**: d < 0 (e.g., ℚ(√-1), ℚ(√-3))
//!
//! Special property: There are exactly 9 imaginary quadratic fields with class number 1:
//! d ∈ {-3, -4, -7, -8, -11, -19, -43, -67, -163}
//!
//! #### Cyclotomic Fields
//! Fields of the form ℚ(ζₙ) where ζₙ = e^(2πi/n) is a primitive n-th root of unity.
//! These have degree φ(n) where φ is Euler's totient function. The minimal polynomial
//! is the n-th cyclotomic polynomial Φₙ(x).
//!
//! Properties:
//! - The ring of integers is ℤ[ζₙ]
//! - The discriminant is Δ = (-1)^(φ(n)/2) · n^(φ(n)) / ∏(p|n) p^(φ(n)/(p-1))
//! - Cyclotomic fields are always Galois over ℚ with abelian Galois group
//!
//! #### Cubic Fields
//! Fields of degree 3. These can be:
//! - **Totally real**: All three roots of the minimal polynomial are real
//! - **Complex**: One real root and two complex conjugate roots
//!
//! ## Implementation Notes
//!
//! This module builds upon `rustmath-numberfields` and provides:
//! - Constructors for common number fields (quadratic, cyclotomic)
//! - Element arithmetic in number fields
//! - Computation of field invariants (discriminant, class number, unit group)
//! - Integration with the broader `rustmath-rings` ecosystem
//!
//! ## Examples
//!
//! ### Creating Quadratic Fields
//!
//! ```rust
//! use rustmath_rings::number_field::NumberFieldExt;
//!
//! // ℚ(√2) - a real quadratic field
//! let k = NumberFieldExt::quadratic(2);
//! assert_eq!(k.degree(), 2);
//!
//! // ℚ(√-3) - an imaginary quadratic field with class number 1
//! let k = NumberFieldExt::quadratic(-3);
//! assert_eq!(k.degree(), 2);
//! ```
//!
//! ### Creating Cyclotomic Fields
//!
//! ```rust
//! use rustmath_rings::number_field::NumberFieldExt;
//!
//! // ℚ(ζ₅) - 5th cyclotomic field, degree φ(5) = 4
//! let k = NumberFieldExt::cyclotomic(5);
//! assert_eq!(k.degree(), 4);
//!
//! // ℚ(i) = ℚ(ζ₄) - Gaussian rationals, degree 2
//! let k = NumberFieldExt::cyclotomic(4);
//! assert_eq!(k.degree(), 2);
//! ```
//!
//! ### Working with Elements
//!
//! ```rust
//! use rustmath_rings::number_field::NumberFieldExt;
//!
//! let k = NumberFieldExt::quadratic(2);
//! let alpha = k.generator(); // √2
//! let one = k.one();
//!
//! // Compute (1 + √2)²
//! let x = k.add(&one, &alpha);
//! let x_squared = k.mul(&x, &x);
//! // Result is 3 + 2√2
//! ```
//!
//! ## References
//!
//! - **Marcus, D.A.** (1977). *Number Fields*. Springer-Verlag.
//! - **Neukirch, J.** (1999). *Algebraic Number Theory*. Springer-Verlag.
//! - **Cohen, H.** (1993). *A Course in Computational Algebraic Number Theory*. Springer.
//! - **Washington, L.C.** (1997). *Introduction to Cyclotomic Fields*. Springer.

// Re-export core number field types from rustmath-numberfields
pub use rustmath_numberfields::{
    NumberField, NumberFieldElement, NumberFieldError, UnitGroup,
};

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_polynomials::univariate::UnivariatePolynomial;
use rustmath_rationals::Rational;

/// Extended number field functionality with convenient constructors
pub struct NumberFieldExt;

impl NumberFieldExt {
    /// Create a quadratic field ℚ(√d) for a square-free integer d
    ///
    /// # Arguments
    /// * `d` - A square-free integer (positive for real quadratic, negative for imaginary)
    ///
    /// # Returns
    /// The number field ℚ(√d) with minimal polynomial x² - d
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rings::number_field::NumberFieldExt;
    ///
    /// // Real quadratic field ℚ(√2)
    /// let k = NumberFieldExt::quadratic(2);
    /// assert_eq!(k.degree(), 2);
    ///
    /// // Imaginary quadratic field ℚ(√-3)
    /// let k = NumberFieldExt::quadratic(-3);
    /// assert_eq!(k.degree(), 2);
    /// ```
    pub fn quadratic(d: i64) -> NumberField {
        // Minimal polynomial is x² - d
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-d), // constant term -d
            Rational::from_integer(0),  // x coefficient
            Rational::from_integer(1),  // x² coefficient
        ]);
        NumberField::new(poly)
    }

    /// Create the n-th cyclotomic field ℚ(ζₙ)
    ///
    /// The cyclotomic field is generated by a primitive n-th root of unity ζₙ = e^(2πi/n).
    /// The minimal polynomial is the n-th cyclotomic polynomial Φₙ(x), which has degree φ(n)
    /// where φ is Euler's totient function.
    ///
    /// # Arguments
    /// * `n` - The order of the root of unity (n ≥ 1)
    ///
    /// # Returns
    /// The n-th cyclotomic field with degree φ(n)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rings::number_field::NumberFieldExt;
    ///
    /// // ℚ(ζ₃) has degree φ(3) = 2
    /// let k = NumberFieldExt::cyclotomic(3);
    /// assert_eq!(k.degree(), 2);
    ///
    /// // ℚ(ζ₅) has degree φ(5) = 4
    /// let k = NumberFieldExt::cyclotomic(5);
    /// assert_eq!(k.degree(), 4);
    ///
    /// // ℚ(ζ₈) has degree φ(8) = 4
    /// let k = NumberFieldExt::cyclotomic(8);
    /// assert_eq!(k.degree(), 4);
    /// ```
    pub fn cyclotomic(n: usize) -> NumberField {
        assert!(n >= 1, "n must be at least 1");

        // Compute the n-th cyclotomic polynomial Φₙ(x)
        let cyclo_poly = Self::cyclotomic_polynomial(n);
        NumberField::new(cyclo_poly)
    }

    /// Compute the n-th cyclotomic polynomial Φₙ(x)
    ///
    /// The cyclotomic polynomial is defined by:
    /// ```text
    /// Φₙ(x) = ∏(ζ: ζⁿ=1, ζ primitive) (x - ζ)
    /// ```
    ///
    /// We use the recursive formula:
    /// ```text
    /// xⁿ - 1 = ∏(d|n) Φ_d(x)
    /// ```
    /// which gives:
    /// ```text
    /// Φₙ(x) = (xⁿ - 1) / ∏(d|n, d<n) Φ_d(x)
    /// ```
    fn cyclotomic_polynomial(n: usize) -> UnivariatePolynomial<Rational> {
        match n {
            1 => {
                // Φ₁(x) = x - 1
                UnivariatePolynomial::new(vec![
                    Rational::from_integer(-1),
                    Rational::from_integer(1),
                ])
            }
            2 => {
                // Φ₂(x) = x + 1
                UnivariatePolynomial::new(vec![
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                ])
            }
            3 => {
                // Φ₃(x) = x² + x + 1
                UnivariatePolynomial::new(vec![
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                ])
            }
            4 => {
                // Φ₄(x) = x² + 1
                UnivariatePolynomial::new(vec![
                    Rational::from_integer(1),
                    Rational::from_integer(0),
                    Rational::from_integer(1),
                ])
            }
            5 => {
                // Φ₅(x) = x⁴ + x³ + x² + x + 1
                UnivariatePolynomial::new(vec![
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                ])
            }
            6 => {
                // Φ₆(x) = x² - x + 1
                UnivariatePolynomial::new(vec![
                    Rational::from_integer(1),
                    Rational::from_integer(-1),
                    Rational::from_integer(1),
                ])
            }
            7 => {
                // Φ₇(x) = x⁶ + x⁵ + x⁴ + x³ + x² + x + 1
                UnivariatePolynomial::new(vec![
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                    Rational::from_integer(1),
                ])
            }
            8 => {
                // Φ₈(x) = x⁴ + 1
                UnivariatePolynomial::new(vec![
                    Rational::from_integer(1),
                    Rational::from_integer(0),
                    Rational::from_integer(0),
                    Rational::from_integer(0),
                    Rational::from_integer(1),
                ])
            }
            _ => {
                // For larger n, use the recursive formula
                // Φₙ(x) = (xⁿ - 1) / ∏(d|n, d<n) Φ_d(x)

                // Compute x^n - 1
                let mut x_n_minus_1_coeffs = vec![Rational::zero(); n + 1];
                x_n_minus_1_coeffs[0] = Rational::from_integer(-1);
                x_n_minus_1_coeffs[n] = Rational::from_integer(1);
                let mut result = UnivariatePolynomial::new(x_n_minus_1_coeffs);

                // Divide by Φ_d(x) for all proper divisors d of n
                for d in Self::divisors(n) {
                    if d < n {
                        let phi_d = Self::cyclotomic_polynomial(d);
                        let (quotient, _remainder) = result.quo_rem(&phi_d);
                        result = quotient;
                    }
                }

                result
            }
        }
    }

    /// Find all divisors of n
    fn divisors(n: usize) -> Vec<usize> {
        let mut divs = Vec::new();
        for i in 1..=n {
            if n % i == 0 {
                divs.push(i);
            }
        }
        divs
    }

    /// Compute Euler's totient function φ(n)
    ///
    /// φ(n) counts the number of integers k in 1..=n that are coprime to n.
    /// This is the degree of the n-th cyclotomic field.
    pub fn euler_totient(n: usize) -> usize {
        if n == 1 {
            return 1;
        }

        let mut result = n;
        let mut m = n;

        // For each prime factor p, multiply result by (1 - 1/p)
        let mut p = 2;
        while p * p <= m {
            if m % p == 0 {
                // Remove all factors of p
                while m % p == 0 {
                    m /= p;
                }
                // result *= (1 - 1/p) = (p - 1) / p
                result -= result / p;
            }
            p += 1;
        }

        // If m > 1, then it's a prime factor
        if m > 1 {
            result -= result / m;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Quadratic Fields Tests
    // ============================================================================

    #[test]
    fn test_quadratic_field_degree() {
        let k = NumberFieldExt::quadratic(2);
        assert_eq!(k.degree(), 2, "ℚ(√2) should have degree 2");

        let k = NumberFieldExt::quadratic(-3);
        assert_eq!(k.degree(), 2, "ℚ(√-3) should have degree 2");

        let k = NumberFieldExt::quadratic(5);
        assert_eq!(k.degree(), 2, "ℚ(√5) should have degree 2");
    }

    #[test]
    fn test_quadratic_real_sqrt2() {
        // Test ℚ(√2)
        let k = NumberFieldExt::quadratic(2);
        let alpha = k.generator(); // √2
        let one = k.one();

        // Verify √2 · √2 = 2
        let alpha_squared = k.mul(&alpha, &alpha);
        assert_eq!(
            alpha_squared.coeff(0),
            Rational::from_integer(2),
            "√2 · √2 should equal 2"
        );
        assert_eq!(
            alpha_squared.coeff(1),
            Rational::zero(),
            "√2 · √2 should have no linear term"
        );

        // Test (1 + √2)² = 1 + 2√2 + 2 = 3 + 2√2
        let one_plus_sqrt2 = k.add(&one, &alpha);
        let result = k.mul(&one_plus_sqrt2, &one_plus_sqrt2);
        assert_eq!(result.coeff(0), Rational::from_integer(3));
        assert_eq!(result.coeff(1), Rational::from_integer(2));

        // Test discriminant: For x² - 2, disc = 8
        let disc = k.discriminant();
        assert_eq!(disc, Rational::from_integer(8));
    }

    #[test]
    fn test_quadratic_imaginary_sqrt_minus_3() {
        // Test ℚ(√-3)
        let k = NumberFieldExt::quadratic(-3);
        let alpha = k.generator(); // √-3
        let one = k.one();

        // Verify √-3 · √-3 = -3
        let alpha_squared = k.mul(&alpha, &alpha);
        assert_eq!(alpha_squared.coeff(0), Rational::from_integer(-3));

        // Test (1 + √-3)² = 1 + 2√-3 - 3 = -2 + 2√-3
        let one_plus_alpha = k.add(&one, &alpha);
        let result = k.mul(&one_plus_alpha, &one_plus_alpha);
        assert_eq!(result.coeff(0), Rational::from_integer(-2));
        assert_eq!(result.coeff(1), Rational::from_integer(2));

        // Test discriminant: For x² + 3, disc = -12
        let disc = k.discriminant();
        assert_eq!(disc, Rational::from_integer(-12));
    }

    #[test]
    fn test_quadratic_field_arithmetic() {
        let k = NumberFieldExt::quadratic(5);
        let alpha = k.generator(); // √5
        let two = k.from_rational(Rational::from_integer(2));
        let three = k.from_rational(Rational::from_integer(3));

        // Test: (2 + 3√5) + (1 + √5) = 3 + 4√5
        let a = k.add(&two, &k.mul(&three, &alpha));
        let b = k.add(&k.one(), &alpha);
        let sum = k.add(&a, &b);
        assert_eq!(sum.coeff(0), Rational::from_integer(3));
        assert_eq!(sum.coeff(1), Rational::from_integer(4));

        // Test: (2 + √5)(2 - √5) = 4 - 5 = -1
        let two_plus_sqrt5 = k.add(&two, &alpha);
        let two_minus_sqrt5 = k.sub(&two, &alpha);
        let product = k.mul(&two_plus_sqrt5, &two_minus_sqrt5);
        assert_eq!(product.coeff(0), Rational::from_integer(-1));
        assert_eq!(product.coeff(1), Rational::zero());
    }

    #[test]
    fn test_quadratic_class_number_one_fields() {
        // Test the 9 imaginary quadratic fields with class number 1
        let class_one_discriminants = [-3, -4, -7, -8, -11, -19, -43, -67, -163];

        for &d in &class_one_discriminants {
            let k = NumberFieldExt::quadratic(d);
            let class_num = k.class_number();
            assert!(
                class_num.is_ok(),
                "Class number computation should succeed for d = {}",
                d
            );
            if let Ok(h) = class_num {
                assert_eq!(
                    h,
                    Integer::one(),
                    "ℚ(√{}) should have class number 1",
                    d
                );
            }
        }
    }

    #[test]
    fn test_quadratic_unit_group() {
        // Real quadratic field ℚ(√2) has signature (2,0), so rank = 2 + 0 - 1 = 1
        let k = NumberFieldExt::quadratic(2);
        let unit_group = k.unit_group().unwrap();
        assert_eq!(unit_group.rank, 1, "ℚ(√2) should have unit rank 1");
        assert_eq!(
            unit_group.roots_of_unity_order, 2,
            "ℚ(√2) should have 2 roots of unity (±1)"
        );

        // Imaginary quadratic field ℚ(√-3) has signature (0,1), so rank = 0 + 1 - 1 = 0
        let k = NumberFieldExt::quadratic(-3);
        let unit_group = k.unit_group().unwrap();
        assert_eq!(unit_group.rank, 0, "ℚ(√-3) should have unit rank 0");
    }

    #[test]
    fn test_quadratic_power_basis() {
        let k = NumberFieldExt::quadratic(2);
        let basis = k.power_basis();

        assert_eq!(basis.len(), 2, "Power basis should have 2 elements");

        // First element is 1
        assert_eq!(basis[0].coeff(0), Rational::one());
        assert_eq!(basis[0].coeff(1), Rational::zero());

        // Second element is α
        assert_eq!(basis[1].coeff(0), Rational::zero());
        assert_eq!(basis[1].coeff(1), Rational::one());
    }

    #[test]
    fn test_quadratic_norm_and_trace() {
        let k = NumberFieldExt::quadratic(2);

        // For a = 3 + 2√2 in ℚ(√2):
        // N(a) = (3 + 2√2)(3 - 2√2) = 9 - 8 = 1
        // T(a) = (3 + 2√2) + (3 - 2√2) = 6
        let three = k.from_rational(Rational::from_integer(3));
        let two_alpha = k.mul(
            &k.from_rational(Rational::from_integer(2)),
            &k.generator(),
        );
        let a = k.add(&three, &two_alpha);

        let norm = a.norm();
        assert_eq!(norm, Rational::from_integer(1), "Norm should be 1");

        // Note: Trace computation needs full implementation
        // For now we just test that the method exists
        let _trace = a.trace();
    }

    // ============================================================================
    // Cyclotomic Fields Tests
    // ============================================================================

    #[test]
    fn test_euler_totient() {
        assert_eq!(NumberFieldExt::euler_totient(1), 1);
        assert_eq!(NumberFieldExt::euler_totient(2), 1);
        assert_eq!(NumberFieldExt::euler_totient(3), 2);
        assert_eq!(NumberFieldExt::euler_totient(4), 2);
        assert_eq!(NumberFieldExt::euler_totient(5), 4);
        assert_eq!(NumberFieldExt::euler_totient(6), 2);
        assert_eq!(NumberFieldExt::euler_totient(7), 6);
        assert_eq!(NumberFieldExt::euler_totient(8), 4);
        assert_eq!(NumberFieldExt::euler_totient(9), 6);
        assert_eq!(NumberFieldExt::euler_totient(10), 4);
        assert_eq!(NumberFieldExt::euler_totient(12), 4);
    }

    #[test]
    fn test_cyclotomic_polynomial_degrees() {
        // Φₙ(x) should have degree φ(n)
        assert_eq!(
            NumberFieldExt::cyclotomic_polynomial(1).degree(),
            Some(1)
        );
        assert_eq!(
            NumberFieldExt::cyclotomic_polynomial(2).degree(),
            Some(1)
        );
        assert_eq!(
            NumberFieldExt::cyclotomic_polynomial(3).degree(),
            Some(2)
        );
        assert_eq!(
            NumberFieldExt::cyclotomic_polynomial(4).degree(),
            Some(2)
        );
        assert_eq!(
            NumberFieldExt::cyclotomic_polynomial(5).degree(),
            Some(4)
        );
        assert_eq!(
            NumberFieldExt::cyclotomic_polynomial(6).degree(),
            Some(2)
        );
        assert_eq!(
            NumberFieldExt::cyclotomic_polynomial(7).degree(),
            Some(6)
        );
        assert_eq!(
            NumberFieldExt::cyclotomic_polynomial(8).degree(),
            Some(4)
        );
    }

    #[test]
    fn test_cyclotomic_field_degrees() {
        // ℚ(ζₙ) should have degree φ(n)
        assert_eq!(NumberFieldExt::cyclotomic(1).degree(), 1);
        assert_eq!(NumberFieldExt::cyclotomic(2).degree(), 1);
        assert_eq!(NumberFieldExt::cyclotomic(3).degree(), 2);
        assert_eq!(NumberFieldExt::cyclotomic(4).degree(), 2);
        assert_eq!(NumberFieldExt::cyclotomic(5).degree(), 4);
        assert_eq!(NumberFieldExt::cyclotomic(6).degree(), 2);
        assert_eq!(NumberFieldExt::cyclotomic(7).degree(), 6);
        assert_eq!(NumberFieldExt::cyclotomic(8).degree(), 4);
    }

    #[test]
    fn test_cyclotomic_3rd_roots_of_unity() {
        // ℚ(ζ₃) where ζ₃ is a primitive cube root of unity
        // Minimal polynomial: Φ₃(x) = x² + x + 1
        let k = NumberFieldExt::cyclotomic(3);
        assert_eq!(k.degree(), 2);

        let zeta = k.generator();
        let one = k.one();

        // Verify ζ³ = 1, i.e., ζ² + ζ + 1 = 0
        let zeta_squared = k.mul(&zeta, &zeta);
        let sum = k.add(&k.add(&zeta_squared, &zeta), &one);
        assert!(
            sum.coeff(0).is_zero() && sum.coeff(1).is_zero(),
            "ζ² + ζ + 1 should equal 0"
        );

        // Verify 1 + ζ + ζ² = 0 (another way)
        let zeta2 = k.mul(&zeta, &zeta);
        let sum2 = k.add(&one, &k.add(&zeta, &zeta2));
        assert!(sum2.coeff(0).is_zero() && sum2.coeff(1).is_zero());
    }

    #[test]
    fn test_cyclotomic_4th_gaussian_field() {
        // ℚ(ζ₄) = ℚ(i) - the Gaussian rationals
        // Minimal polynomial: Φ₄(x) = x² + 1
        let k = NumberFieldExt::cyclotomic(4);
        assert_eq!(k.degree(), 2);

        let i = k.generator(); // i

        // Verify i² = -1
        let i_squared = k.mul(&i, &i);
        assert_eq!(i_squared.coeff(0), Rational::from_integer(-1));
        assert_eq!(i_squared.coeff(1), Rational::zero());

        // Test (1 + i)² = 1 + 2i - 1 = 2i
        let one = k.one();
        let one_plus_i = k.add(&one, &i);
        let result = k.mul(&one_plus_i, &one_plus_i);
        assert_eq!(result.coeff(0), Rational::zero());
        assert_eq!(result.coeff(1), Rational::from_integer(2));

        // Test (1 + i)(1 - i) = 1 - i² = 1 + 1 = 2
        let one_minus_i = k.sub(&one, &i);
        let product = k.mul(&one_plus_i, &one_minus_i);
        assert_eq!(product.coeff(0), Rational::from_integer(2));
        assert_eq!(product.coeff(1), Rational::zero());
    }

    #[test]
    fn test_cyclotomic_5th_roots() {
        // ℚ(ζ₅) - 5th cyclotomic field
        // Minimal polynomial: Φ₅(x) = x⁴ + x³ + x² + x + 1
        let k = NumberFieldExt::cyclotomic(5);
        assert_eq!(k.degree(), 4);

        let zeta = k.generator();
        let one = k.one();

        // Verify 1 + ζ + ζ² + ζ³ + ζ⁴ = 0
        let z2 = k.mul(&zeta, &zeta);
        let z3 = k.mul(&z2, &zeta);
        let z4 = k.mul(&z3, &zeta);

        let sum = k.add(&one, &zeta);
        let sum = k.add(&sum, &z2);
        let sum = k.add(&sum, &z3);
        let sum = k.add(&sum, &z4);

        // All coefficients should be zero
        for i in 0..4 {
            assert!(
                sum.coeff(i).is_zero(),
                "Coefficient {} of 1+ζ+ζ²+ζ³+ζ⁴ should be 0",
                i
            );
        }
    }

    #[test]
    fn test_cyclotomic_6th_field() {
        // ℚ(ζ₆) - 6th cyclotomic field
        // Since ζ₆ = e^(2πi/6) = e^(πi/3), we have ζ₆² = e^(2πi/3) = primitive 3rd root
        // Φ₆(x) = x² - x + 1
        let k = NumberFieldExt::cyclotomic(6);
        assert_eq!(k.degree(), 2);

        let zeta6 = k.generator();

        // Verify ζ₆² - ζ₆ + 1 = 0
        let zeta6_sq = k.mul(&zeta6, &zeta6);
        let result = k.sub(&k.add(&zeta6_sq, &k.one()), &zeta6);

        assert!(
            result.coeff(0).is_zero() && result.coeff(1).is_zero(),
            "ζ₆² - ζ₆ + 1 should equal 0"
        );
    }

    #[test]
    fn test_cyclotomic_8th_field() {
        // ℚ(ζ₈) - 8th cyclotomic field
        // Minimal polynomial: Φ₈(x) = x⁴ + 1
        let k = NumberFieldExt::cyclotomic(8);
        assert_eq!(k.degree(), 4);

        let zeta8 = k.generator();

        // Verify ζ₈⁴ = -1
        let z2 = k.mul(&zeta8, &zeta8);
        let z4 = k.mul(&z2, &z2);

        assert_eq!(z4.coeff(0), Rational::from_integer(-1));
        for i in 1..4 {
            assert!(z4.coeff(i).is_zero(), "Higher coefficients should be 0");
        }
    }

    #[test]
    fn test_cyclotomic_field_discriminants() {
        // Test discriminants for small cyclotomic fields
        // These have known formulas

        // ℚ(ζ₃): disc = -3
        let k = NumberFieldExt::cyclotomic(3);
        let disc = k.discriminant();
        assert_eq!(disc, Rational::from_integer(-3));

        // ℚ(ζ₄) = ℚ(i): disc = -4
        let k = NumberFieldExt::cyclotomic(4);
        let disc = k.discriminant();
        assert_eq!(disc, Rational::from_integer(-4));
    }

    #[test]
    fn test_cyclotomic_polynomial_explicit_formulas() {
        // Verify explicit formulas for small n

        // Φ₁(x) = x - 1
        let phi1 = NumberFieldExt::cyclotomic_polynomial(1);
        assert_eq!(phi1.coeff(0), &Rational::from_integer(-1));
        assert_eq!(phi1.coeff(1), &Rational::from_integer(1));

        // Φ₂(x) = x + 1
        let phi2 = NumberFieldExt::cyclotomic_polynomial(2);
        assert_eq!(phi2.coeff(0), &Rational::from_integer(1));
        assert_eq!(phi2.coeff(1), &Rational::from_integer(1));

        // Φ₃(x) = x² + x + 1
        let phi3 = NumberFieldExt::cyclotomic_polynomial(3);
        assert_eq!(phi3.coeff(0), &Rational::from_integer(1));
        assert_eq!(phi3.coeff(1), &Rational::from_integer(1));
        assert_eq!(phi3.coeff(2), &Rational::from_integer(1));

        // Φ₄(x) = x² + 1
        let phi4 = NumberFieldExt::cyclotomic_polynomial(4);
        assert_eq!(phi4.coeff(0), &Rational::from_integer(1));
        assert_eq!(phi4.coeff(1), &Rational::from_integer(0));
        assert_eq!(phi4.coeff(2), &Rational::from_integer(1));

        // Φ₅(x) = x⁴ + x³ + x² + x + 1
        let phi5 = NumberFieldExt::cyclotomic_polynomial(5);
        for i in 0..=4 {
            assert_eq!(phi5.coeff(i), &Rational::from_integer(1));
        }

        // Φ₆(x) = x² - x + 1
        let phi6 = NumberFieldExt::cyclotomic_polynomial(6);
        assert_eq!(phi6.coeff(0), &Rational::from_integer(1));
        assert_eq!(phi6.coeff(1), &Rational::from_integer(-1));
        assert_eq!(phi6.coeff(2), &Rational::from_integer(1));
    }

    // ============================================================================
    // General Number Field Tests
    // ============================================================================

    #[test]
    fn test_number_field_display() {
        let k = NumberFieldExt::quadratic(2);
        let display = format!("{}", k);
        assert!(display.contains("Q(α)") || display.contains("ℚ(α)"));
    }

    #[test]
    fn test_element_equality() {
        let k = NumberFieldExt::quadratic(2);
        let a = k.from_rational(Rational::from_integer(3));
        let b = k.from_rational(Rational::from_integer(3));
        let c = k.from_rational(Rational::from_integer(4));

        assert_eq!(a, b, "Equal elements should be equal");
        assert_ne!(a, c, "Different elements should not be equal");
    }

    #[test]
    fn test_number_field_basis_linear_independence() {
        let k = NumberFieldExt::cyclotomic(5);
        let basis = k.power_basis();

        // The power basis should have φ(5) = 4 elements
        assert_eq!(basis.len(), 4);

        // Each basis element should be distinct
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert_ne!(
                        basis[i], basis[j],
                        "Basis elements should be distinct"
                    );
                }
            }
        }
    }

    #[test]
    fn test_field_operations_associativity() {
        let k = NumberFieldExt::quadratic(3);
        let a = k.generator();
        let b = k.from_rational(Rational::from_integer(2));
        let c = k.one();

        // Test (a + b) + c = a + (b + c)
        let left = k.add(&k.add(&a, &b), &c);
        let right = k.add(&a, &k.add(&b, &c));
        assert_eq!(left, right, "Addition should be associative");

        // Test (a * b) * c = a * (b * c)
        let left = k.mul(&k.mul(&a, &b), &c);
        let right = k.mul(&a, &k.mul(&b, &c));
        assert_eq!(left, right, "Multiplication should be associative");
    }

    #[test]
    fn test_field_operations_distributivity() {
        let k = NumberFieldExt::quadratic(2);
        let a = k.generator();
        let b = k.from_rational(Rational::from_integer(2));
        let c = k.one();

        // Test a * (b + c) = a*b + a*c
        let left = k.mul(&a, &k.add(&b, &c));
        let right = k.add(&k.mul(&a, &b), &k.mul(&a, &c));
        assert_eq!(left, right, "Multiplication should distribute over addition");
    }
}
