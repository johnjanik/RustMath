//! Univariate polynomials (polynomials in one variable)

use crate::polynomial::Polynomial;
use rustmath_core::{CommutativeRing, EuclideanDomain, IntegralDomain, MathError, Result, Ring};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Univariate polynomial over a ring R
#[derive(Clone, PartialEq, Eq)]
pub struct UnivariatePolynomial<R: Ring> {
    /// Coefficients in increasing degree order: [a0, a1, a2, ...] represents a0 + a1*x + a2*x^2 + ...
    coeffs: Vec<R>,
}

impl<R: Ring> UnivariatePolynomial<R> {
    /// Create a new polynomial from coefficients
    pub fn new(mut coeffs: Vec<R>) -> Self {
        // Remove leading zeros
        while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
            coeffs.pop();
        }

        if coeffs.is_empty() {
            coeffs.push(R::zero());
        }

        UnivariatePolynomial { coeffs }
    }

    /// Create a polynomial from coefficients (alias for new)
    pub fn from_coefficients(coeffs: Vec<R>) -> Self {
        Self::new(coeffs)
    }

    /// Create a constant polynomial
    pub fn constant(c: R) -> Self {
        UnivariatePolynomial::new(vec![c])
    }

    /// Create the polynomial x (the variable)
    pub fn var() -> Self {
        UnivariatePolynomial::new(vec![R::zero(), R::one()])
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[R] {
        &self.coeffs
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.len() == 1 && self.coeffs[0].is_zero() {
            None // Zero polynomial has undefined degree
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Get coefficient at given degree
    pub fn coeff(&self, degree: usize) -> &R {
        self.coeffs.get(degree).unwrap_or(&self.coeffs[0])
    }

    /// Get the leading coefficient (highest degree non-zero coefficient)
    pub fn leading_coefficient(&self) -> Option<&R> {
        self.leading_coeff()
    }

    /// Get the leading coefficient (internal method)
    fn leading_coeff(&self) -> Option<&R> {
        if self.is_zero() {
            None
        } else {
            self.coeffs.last()
        }
    }

    /// Create the zero polynomial
    pub fn zero() -> Self {
        UnivariatePolynomial::new(vec![R::zero()])
    }

    /// Create the one polynomial
    pub fn one() -> Self {
        UnivariatePolynomial::new(vec![R::one()])
    }

    /// Create a polynomial from a usize constant
    pub fn from_usize(n: usize) -> Self
    where
        R: rustmath_core::NumericConversion,
    {
        UnivariatePolynomial::constant(R::from_u64(n as u64))
    }

    /// Evaluate the polynomial at a point (alias for eval)
    pub fn evaluate(&self, point: &R) -> R {
        self.eval(point)
    }

    /// Negate the polynomial (returns the additive inverse)
    pub fn negate(self) -> Self {
        -self
    }

    /// Check if this is the zero polynomial
    pub fn is_zero(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_zero()
    }

    /// Multiply by a scalar
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        if scalar.is_zero() {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        let coeffs = self.coeffs.iter().map(|c| c.clone() * scalar.clone()).collect();
        UnivariatePolynomial::new(coeffs)
    }

    /// Shift the polynomial by multiplying by x^n
    pub fn shift(&self, n: usize) -> Self {
        if n == 0 || self.is_zero() {
            return self.clone();
        }

        let mut coeffs = vec![R::zero(); n];
        coeffs.extend_from_slice(&self.coeffs);
        UnivariatePolynomial::new(coeffs)
    }

    /// Polynomial composition: compute p(q(x))
    ///
    /// Substitutes polynomial q into this polynomial p
    pub fn compose(&self, q: &Self) -> Self {
        if self.is_zero() {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        // Use Horner's method: p(q) = a_0 + q*(a_1 + q*(a_2 + ...))
        let mut result = UnivariatePolynomial::new(vec![self.coeffs.last().unwrap().clone()]);

        for coeff in self.coeffs.iter().rev().skip(1) {
            // result = result * q + coeff
            result = (result.clone() * q.clone())
                + UnivariatePolynomial::new(vec![coeff.clone()]);
        }

        result
    }

    /// Scale the variable: compute p(c*x) for constant c
    pub fn scale_variable(&self, c: &R) -> Self {
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        let mut power_of_c = R::one();

        for coeff in &self.coeffs {
            coeffs.push(coeff.clone() * power_of_c.clone());
            power_of_c = power_of_c * c.clone();
        }

        UnivariatePolynomial::new(coeffs)
    }

    /// Translate the polynomial: compute p(x + a)
    pub fn translate(&self, a: &R) -> Self {
        // p(x + a) = p composed with (x + a)
        let x_plus_a = UnivariatePolynomial::new(vec![a.clone(), R::one()]);
        self.compose(&x_plus_a)
    }

    /// Compute the derivative
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        let coeffs = self.coeffs[1..]
            .iter()
            .enumerate()
            .map(|(i, c)| {
                // Multiply coefficient by (i+1) using repeated addition
                let mut result = R::zero();
                for _ in 0..=(i as u32) {
                    result = result + c.clone();
                }
                result
            })
            .collect();

        UnivariatePolynomial::new(coeffs)
    }

    /// Compute the indefinite integral (antiderivative)
    ///
    /// Returns ∫p(x)dx with constant of integration = 0
    /// For coefficient ring where division by integers is available
    pub fn integrate(&self) -> Self
    where
        R: rustmath_core::NumericConversion + rustmath_core::Field,
    {
        if self.is_zero() {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        // Add a zero constant term, then divide each coefficient by its new index
        let mut coeffs = vec![R::zero()]; // Constant of integration

        for (i, c) in self.coeffs.iter().enumerate() {
            // Coefficient for x^(i+1) is c/(i+1)
            let divisor = R::from_i64((i + 1) as i64);
            // For integer coefficients, this is an approximation
            // In a proper implementation, we'd need rational coefficients
            let new_coeff = c.clone() * divisor.inverse().unwrap();
            coeffs.push(new_coeff);
        }

        UnivariatePolynomial::new(coeffs)
    }

    /// Compute definite integral from a to b
    ///
    /// ∫[a,b] p(x)dx = P(b) - P(a) where P is the antiderivative
    pub fn definite_integral(&self, a: &R, b: &R) -> R
    where
        R: rustmath_core::NumericConversion + rustmath_core::Field,
    {
        let antiderivative = self.integrate();
        antiderivative.eval(b) - antiderivative.eval(a)
    }

    /// Compute polynomial GCD (for polynomials over a field or Euclidean domain)
    ///
    /// # Limitations
    ///
    /// This implementation uses the Euclidean algorithm which requires exact division.
    /// For polynomials with integer coefficients, this may fail or produce incorrect
    /// results when the leading coefficient of the divisor doesn't divide the leading
    /// coefficient of the dividend.
    ///
    /// **TODO**: Implement pseudo-division or subresultant-based GCD algorithm for
    /// polynomials over integers (Z[x]). Until then, GCD over integer polynomials
    /// works reliably only when coefficients divide cleanly.
    pub fn gcd(&self, other: &Self) -> Self
    where
        R: EuclideanDomain,
    {
        let mut a = self.clone();
        let mut b = other.clone();

        while !b.is_zero() {
            let (_, r) = a.div_rem(&b).unwrap();

            // `quo_rem` gives up (returning the dividend untouched) as soon as the
            // divisor's leading coefficient does not divide exactly in R — which is
            // unavoidable, because R[x] over a non-field is not a Euclidean domain.
            // The plain Euclidean loop then never reduces the degree and cycles forever
            // (e.g. gcd(-x³+4x²-5x+2, -3x²+8x-5) over Z spun until the harness killed
            // it). Fall back to the pseudo-remainder, which cancels the leading term by
            // construction and therefore strictly lowers the degree, so the loop
            // terminates. The result is then the gcd up to a factor in R (the usual
            // pseudo-remainder-sequence caveat); every input on which the ordinary
            // division already made progress takes the original path unchanged.
            let no_progress = match (r.degree(), b.degree()) {
                (Some(dr), Some(db)) => dr >= db,
                _ => false,
            };
            let r = if no_progress { a.pseudo_rem(&b) } else { r };

            a = b;
            b = r;
        }

        // Note: Making polynomial monic (leading coefficient = 1) requires
        // division by the leading coefficient, which needs the Field trait.
        // This is left as a future enhancement.

        a
    }

    /// The pseudo-remainder of `self` by `divisor`.
    ///
    /// Repeatedly scales the running remainder by the divisor's leading coefficient
    /// before subtracting, so the leading term always cancels exactly and the degree
    /// strictly decreases — no exact divisibility in `R` is needed. This is what makes a
    /// gcd over a non-field coefficient ring terminate.
    fn pseudo_rem(&self, divisor: &Self) -> Self
    where
        R: EuclideanDomain,
    {
        let Some(divisor_degree) = divisor.degree() else {
            panic!("Pseudo-division by zero polynomial");
        };
        let divisor_leading = divisor.coeffs.last().unwrap().clone();

        let mut remainder = self.clone();
        while let Some(rem_degree) = remainder.degree() {
            if rem_degree < divisor_degree {
                break;
            }

            let rem_leading = remainder.coeffs.last().unwrap().clone();
            let mut mono_coeffs = vec![R::zero(); rem_degree - divisor_degree];
            mono_coeffs.push(rem_leading);
            let mono = UnivariatePolynomial::new(mono_coeffs);

            // lc(divisor)·remainder − lead(remainder)·x^(dr−db)·divisor
            remainder = remainder.scalar_mul(&divisor_leading) - mono * divisor.clone();
        }

        remainder
    }

    /// Compute polynomial LCM (least common multiple)
    ///
    /// For polynomials f and g: lcm(f, g) = (f * g) / gcd(f, g)
    ///
    /// # Limitations
    ///
    /// Same limitations as GCD - works best over fields or when coefficients divide cleanly.
    pub fn lcm(&self, other: &Self) -> Self
    where
        R: EuclideanDomain,
    {
        if self.is_zero() || other.is_zero() {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        let g = self.gcd(other);
        let product = self.clone() * other.clone();

        // lcm = (self * other) / gcd
        match product.div_rem(&g) {
            Ok((quotient, _)) => quotient,
            Err(_) => product, // Fallback if division fails
        }
    }

    /// Compute the discriminant of a polynomial
    ///
    /// Degrees 2 and 3 use the classical closed forms; degree >= 4 uses the
    /// resultant formula
    /// `disc(f) = (-1)^(n(n-1)/2) * Res(f, f') / lc(f)`,
    /// where `Res` is computed by the fraction-free Bareiss engine of
    /// [`Self::resultant`]. The division by the leading coefficient is exact
    /// over any integral domain (the resultant of `f` and `f'` is `lc(f)`
    /// times the discriminant, a polynomial identity in the coefficients);
    /// if the coefficient ring's `div_rem` reports a nonzero remainder —
    /// which is mathematically impossible over an integral domain and would
    /// indicate a broken `EuclideanDomain` impl — `None` is returned rather
    /// than a wrong value.
    ///
    /// Returns None for the zero polynomial.
    ///
    /// For `R = Integer` a multi-modular CRT engine also exists
    /// ([`crate::disc::discriminant`] on coefficient slices); it is the
    /// preferred entry point in hot integer-only paths. Delegating to it from
    /// here per-type is not possible without specialization, so this generic
    /// path uses Bareiss, which is exact and fast for moderate degrees.
    pub fn discriminant(&self) -> Option<R>
    where
        R: rustmath_core::NumericConversion + EuclideanDomain,
    {
        match self.degree()? {
            0 | 1 => Some(R::one()),
            2 => {
                // ax² + bx + c
                // Discriminant = b² - 4ac
                let a = self.coeff(2);
                let b = self.coeff(1);
                let c = self.coeff(0);

                let b_squared = b.clone() * b.clone();
                let four = R::from_i64(4);
                let four_ac = four * a.clone() * c.clone();
                Some(b_squared - four_ac)
            }
            3 => {
                // ax³ + bx² + cx + d
                // Discriminant = 18abcd - 4b³d + b²c² - 4ac³ - 27a²d²
                let a = self.coeff(3);
                let b = self.coeff(2);
                let c = self.coeff(1);
                let d = self.coeff(0);

                let term1 = R::from_i64(18) * a.clone() * b.clone() * c.clone() * d.clone();
                let term2 = R::from_i64(4) * b.clone() * b.clone() * b.clone() * d.clone();
                let term3 = b.clone() * b.clone() * c.clone() * c.clone();
                let term4 = R::from_i64(4) * a.clone() * c.clone() * c.clone() * c.clone();
                let term5 = R::from_i64(27) * a.clone() * a.clone() * d.clone() * d.clone();

                Some(term1 - term2 + term3 - term4 - term5)
            }
            n => {
                // disc(f) = (-1)^(n(n-1)/2) * Res(f, f') / lc(f)
                let fp = self.derivative();
                let res = self.resultant(&fp);
                let lc = self.leading_coeff()?.clone();
                let (q, r) = res.div_rem(&lc).ok()?;
                if !r.is_zero() {
                    // Impossible over an integral domain; refuse to guess.
                    return None;
                }
                if (n * (n - 1) / 2) % 2 == 1 {
                    Some(-q)
                } else {
                    Some(q)
                }
            }
        }
    }

    /// Check if this polynomial is monic (leading coefficient is 1)
    pub fn is_monic(&self) -> bool {
        if let Some(lc) = self.leading_coeff() {
            lc.is_one()
        } else {
            false
        }
    }

    /// Make the polynomial monic (leading coefficient = 1)
    ///
    /// Divides all coefficients by the leading coefficient
    pub fn make_monic(self) -> Self
    where
        R: rustmath_core::Field,
    {
        if self.is_zero() {
            return self;
        }

        if self.is_monic() {
            return self;
        }

        let lc = self.leading_coeff().unwrap().clone();
        let lc_inv = lc.inverse().unwrap();

        let coeffs = self.coeffs.into_iter().map(|c| c * lc_inv.clone()).collect();
        UnivariatePolynomial::new(coeffs)
    }

    /// Get the content of the polynomial (GCD of all coefficients)
    ///
    /// Only works for coefficients in a Euclidean domain
    pub fn content(&self) -> R
    where
        R: EuclideanDomain,
    {
        if self.coeffs.is_empty() {
            return R::zero();
        }

        let mut gcd = self.coeffs[0].clone();
        for coeff in &self.coeffs[1..] {
            gcd = EuclideanDomain::gcd(&gcd, coeff);
            if gcd.is_one() {
                break;
            }
        }
        gcd
    }

    /// Check if polynomial is square-free (has no repeated factors)
    ///
    /// A polynomial is square-free if it has no repeated factors,
    /// which is equivalent to gcd(f, f') = 1 where f' is the derivative
    pub fn is_square_free(&self) -> bool
    where
        R: EuclideanDomain,
    {
        if self.is_zero() {
            return false;
        }

        let derivative = self.derivative();

        // If derivative is zero, polynomial is not square-free
        // (unless it's a constant, but we handle that above)
        if derivative.is_zero() {
            return false;
        }

        // Square-free ⟺ gcd(f, f') is a nonzero constant. `gcd` here is not
        // normalized to monic (see its note), so a coprime pair yields a nonzero
        // constant ≠ 1; test degree == 0 rather than `is_one()` to avoid a
        // false negative for every f whose gcd unit is not exactly 1.
        let g = self.gcd(&derivative);
        g.degree() == Some(0)
    }

    /// Construct the Sylvester matrix of two polynomials
    ///
    /// For polynomials f of degree m and g of degree n, the Sylvester matrix
    /// is an (m+n) × (m+n) matrix whose determinant is the resultant of f and g.
    ///
    /// Returns the matrix as a vector of rows (Vec<Vec<R>>)
    pub fn sylvester_matrix(&self, other: &Self) -> Vec<Vec<R>> {
        let m = self.degree().unwrap_or(0);
        let n = other.degree().unwrap_or(0);
        let size = m + n;

        let mut matrix = vec![vec![R::zero(); size]; size];

        // First n rows: shifted coefficients of self
        for i in 0..n {
            for j in 0..=m {
                matrix[i][i + j] = self.coeffs.get(m - j).cloned().unwrap_or_else(|| R::zero());
            }
        }

        // Last m rows: shifted coefficients of other
        for i in 0..m {
            for j in 0..=n {
                matrix[n + i][i + j] = other.coeffs.get(n - j).cloned().unwrap_or_else(|| R::zero());
            }
        }

        matrix
    }

    /// Compute the resultant of two polynomials
    ///
    /// The resultant is the determinant of the Sylvester matrix.
    /// It is zero if and only if the polynomials have a common root (over an algebraically closed field).
    /// Two nonzero constants have resultant 1 (the empty Sylvester determinant).
    ///
    /// # Algorithm and coefficient-ring requirements
    ///
    /// The Sylvester determinant is computed by fraction-free Bareiss
    /// elimination, which is O(n^3) ring operations with intermediate entries
    /// bounded by minors of the input (no coefficient explosion). This
    /// replaced a cofactor expansion that was O(n!) and unusable beyond
    /// degree ~10.
    ///
    /// Bareiss requires `R` to be an **integral domain with exact division**:
    /// every division it performs is by a previous pivot and is exact by the
    /// Sylvester-identity invariant (each intermediate entry is itself a minor
    /// of the original matrix). The `EuclideanDomain` bound (which extends
    /// `IntegralDomain` in this workspace) supplies the `div_rem` used for
    /// those exact divisions; exactness is asserted, so a non-domain `R`
    /// smuggled in through a lawless impl panics rather than returning a
    /// wrong value.
    pub fn resultant(&self, other: &Self) -> R
    where
        R: EuclideanDomain,
    {
        if self.is_zero() || other.is_zero() {
            return R::zero();
        }

        let matrix = self.sylvester_matrix(other);
        Self::bareiss_determinant(matrix)
    }

    /// Determinant by fraction-free Bareiss elimination (see [`Self::resultant`]
    /// for the exact-division argument). The empty matrix has determinant 1.
    fn bareiss_determinant(mut m: Vec<Vec<R>>) -> R
    where
        R: EuclideanDomain,
    {
        let n = m.len();
        if n == 0 {
            return R::one();
        }

        let mut sign_flips = 0usize;
        // prev = pivot of the previous step; 1 initially. Never zero at use:
        // it is m[k-1][k-1] after a successful (nonzero) pivot selection.
        let mut prev = R::one();

        for k in 0..n - 1 {
            // Pivot: find a row at or below k with a nonzero entry in column k.
            if m[k][k].is_zero() {
                match (k + 1..n).find(|&i| !m[i][k].is_zero()) {
                    Some(swap) => {
                        m.swap(k, swap);
                        sign_flips += 1;
                    }
                    // Whole column zero => singular matrix => determinant 0.
                    None => return R::zero(),
                }
            }

            for i in k + 1..n {
                for j in k + 1..n {
                    let num = m[i][j].clone() * m[k][k].clone()
                        - m[i][k].clone() * m[k][j].clone();
                    let (q, r) = num
                        .div_rem(&prev)
                        .expect("Bareiss: division by a nonzero previous pivot");
                    assert!(
                        r.is_zero(),
                        "Bareiss: pivot division was not exact; coefficient ring is not an integral domain"
                    );
                    m[i][j] = q;
                }
                m[i][k] = R::zero();
            }
            prev = m[k][k].clone();
        }

        let det = m[n - 1][n - 1].clone();
        if sign_flips % 2 == 1 {
            -det
        } else {
            det
        }
    }

    /// Compute both quotient and remainder of polynomial division
    ///
    /// Returns (quotient, remainder) such that self = quotient * other + remainder
    ///
    /// # Examples
    /// ```
    /// use rustmath_polynomials::UnivariatePolynomial;
    /// use rustmath_integers::Integer;
    ///
    /// // p(x) = x^2 + 3x + 2, q(x) = x + 1
    /// let p = UnivariatePolynomial::new(vec![
    ///     Integer::from(2),
    ///     Integer::from(3),
    ///     Integer::from(1)
    /// ]);
    /// let q = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(1)]);
    ///
    /// let (quo, rem) = p.quo_rem(&q);
    /// // quo = x + 2, rem = 0
    /// ```
    pub fn quo_rem(&self, divisor: &Self) -> (Self, Self)
    where
        R: EuclideanDomain,
    {
        if divisor.is_zero() {
            panic!("Division by zero polynomial");
        }

        let mut remainder = self.clone();
        let mut quotient = UnivariatePolynomial::new(vec![R::zero()]);

        let divisor_degree = divisor.degree().unwrap();
        let divisor_leading = divisor.coeffs.last().unwrap();

        while let Some(rem_degree) = remainder.degree() {
            if rem_degree < divisor_degree {
                break;
            }

            let rem_leading = remainder.coeffs.last().unwrap();
            let coeff_degree = rem_degree - divisor_degree;

            // Compute quotient coefficient
            let (q, r) = rem_leading.clone().div_rem(divisor_leading).unwrap();
            if !r.is_zero() {
                // Not exact division in the coefficient ring
                break;
            }

            // Create monomial q * x^(coeff_degree)
            let mut mono_coeffs = vec![R::zero(); coeff_degree];
            mono_coeffs.push(q.clone());
            let mono = UnivariatePolynomial::new(mono_coeffs);

            quotient = quotient.clone() + mono.clone();
            remainder = remainder.clone() - mono * divisor.clone();
        }

        (quotient, remainder)
    }

    /// Compute squarefree decomposition (Musser's algorithm)
    ///
    /// Returns a vector of (factor, multiplicity) pairs where each factor is squarefree
    /// and the original polynomial is the product of factor^multiplicity
    ///
    /// # Coefficient ring
    ///
    /// This requires a **field**. The algorithm divides by `gcd(f, f')` and by successive
    /// gcds, and those divisions have to be exact; over a ring such as `Z` they are not
    /// (`quo_rem` simply gives up when the leading coefficient does not divide), and the
    /// recursion then never makes progress. This used to be bounded on `EuclideanDomain`
    /// and, when instantiated at `Z`, looped forever — the doctest below was the example
    /// that hung.
    ///
    /// For `Z[x]`, use [`crate::zx::squarefree_decomposition`], which does the
    /// primitive-part bookkeeping that the integer case actually needs.
    ///
    /// # Examples
    /// ```
    /// use rustmath_polynomials::UnivariatePolynomial;
    /// use rustmath_rationals::Rational;
    /// use rustmath_integers::Integer;
    ///
    /// let c = |n: i64| Rational::from_integer(Integer::from(n));
    ///
    /// // p(x) = -(x-1)^2 * (x-2) = -x^3 + 4x^2 - 5x + 2
    /// let p = UnivariatePolynomial::new(vec![c(2), c(-5), c(4), c(-1)]);
    ///
    /// let decomp = p.squarefree_decomposition();
    /// // (x-1) appears with multiplicity 2, (x-2) with multiplicity 1.
    /// assert_eq!(decomp.len(), 2);
    /// ```
    pub fn squarefree_decomposition(&self) -> Vec<(Self, usize)>
    where
        R: EuclideanDomain + rustmath_core::Field,
    {
        if self.is_zero() {
            return vec![];
        }

        let mut result = Vec::new();
        let f = self.clone();
        let mut i = 1;

        // Compute f' (derivative)
        let fp = self.derivative();

        if fp.is_zero() {
            // Characteristic p case or constant polynomial
            return vec![(f, 1)];
        }

        // Compute gcd(f, f')
        let mut g = f.gcd(&fp);

        // Compute squarefree part
        let (mut s, _) = f.quo_rem(&g);

        // Yun's recursion stops once the squarefree part is constant. Testing only
        // `!s.is_one()` looped forever whenever `s` degenerated to a constant that is not
        // literally 1 — over Z the gcd is only defined up to a unit, so `s` can settle on
        // -1 (or on the content) and never move again.
        while !s.is_one() && s.degree().unwrap_or(0) > 0 {
            let h = s.gcd(&g);
            let (factor, _) = s.quo_rem(&h);

            if !factor.is_one() {
                result.push((factor, i));
            }

            s = h.clone();
            let (new_g, _) = g.quo_rem(&h);
            g = new_g;
            i += 1;
        }

        if !g.is_one() {
            result.push((g, i));
        }

        result
    }

    /// Check if polynomial is constant 1
    fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_one()
    }
}

impl<R: Ring> Polynomial for UnivariatePolynomial<R> {
    type Coeff = R;
    type Var = ();

    fn from_coeffs(coeffs: Vec<R>) -> Self {
        UnivariatePolynomial::new(coeffs)
    }

    fn degree(&self) -> Option<usize> {
        if self.coeffs.len() == 1 && self.coeffs[0].is_zero() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    fn eval(&self, point: &R) -> R {
        // Horner's method for evaluation
        if self.coeffs.is_empty() {
            return R::zero();
        }

        let mut result = self.coeffs.last().unwrap().clone();
        for coeff in self.coeffs.iter().rev().skip(1) {
            result = result * point.clone() + coeff.clone();
        }

        result
    }

    fn coeff(&self, degree: usize) -> &R {
        self.coeffs.get(degree).unwrap_or(&self.coeffs[0])
    }
}

impl<R: Ring> fmt::Display for UnivariatePolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (i, coeff) in self.coeffs.iter().enumerate().rev() {
            if coeff.is_zero() {
                continue;
            }

            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if i == 0 {
                write!(f, "{}", coeff)?;
            } else if i == 1 {
                if coeff.is_one() {
                    write!(f, "x")?;
                } else {
                    write!(f, "{}*x", coeff)?;
                }
            } else if coeff.is_one() {
                write!(f, "x^{}", i)?;
            } else {
                write!(f, "{}*x^{}", coeff, i)?;
            }
        }

        Ok(())
    }
}

impl<R: Ring> fmt::Debug for UnivariatePolynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Poly({:?})", self.coeffs)
    }
}

// Typesetting implementation for polynomials where coefficients implement MathDisplay
impl<R> rustmath_typesetting::MathDisplay for UnivariatePolynomial<R>
where
    R: Ring + rustmath_typesetting::MathDisplay + std::fmt::Display,
{
    fn math_format(&self, options: &rustmath_typesetting::FormatOptions) -> String {
        use rustmath_typesetting::OutputFormat;

        if self.is_zero() {
            return "0".to_string();
        }

        let var_name = "x";
        let mut result = String::new();
        let mut first = true;

        // Iterate through coefficients in descending order (highest degree first)
        for (i, coeff) in self.coeffs.iter().enumerate().rev() {
            if coeff.is_zero() {
                continue;
            }

            let coeff_str = coeff.to_string();

            // Add sign (+ or -)
            if !first {
                result.push_str(" + ");
            }
            first = false;

            // Format the term based on degree
            if i == 0 {
                // Constant term
                result.push_str(&coeff_str);
            } else if i == 1 {
                // Linear term
                if coeff.is_one() {
                    result.push_str(var_name);
                } else {
                    result.push_str(&coeff_str);
                    match options.format {
                        OutputFormat::LaTeX => {
                            if options.implicit_multiply && options.explicit_multiply {
                                result.push_str(r" \cdot ");
                            }
                        }
                        OutputFormat::Unicode if options.explicit_multiply => {
                            result.push('·');
                        }
                        _ => {
                            if options.explicit_multiply {
                                result.push('*');
                            }
                        }
                    }
                    result.push_str(var_name);
                }
            } else {
                // Higher degree terms
                if !coeff.is_one() {
                    result.push_str(&coeff_str);
                    if options.explicit_multiply {
                        match options.format {
                            OutputFormat::LaTeX => result.push_str(r" \cdot "),
                            OutputFormat::Unicode => result.push('·'),
                            _ => result.push('*'),
                        }
                    }
                }

                // Variable with power
                match options.format {
                    OutputFormat::LaTeX => {
                        result.push_str(&format!("{}^{{{}}}", var_name, i));
                    }
                    OutputFormat::Unicode => {
                        result.push_str(var_name);
                        result.push_str(&rustmath_typesetting::utils::to_superscript(&i.to_string()));
                    }
                    OutputFormat::Html => {
                        result.push_str(&rustmath_typesetting::html::power(
                            &rustmath_typesetting::html::identifier(var_name),
                            &rustmath_typesetting::html::number(&i.to_string()),
                        ));
                    }
                    _ => {
                        result.push_str(&format!("{}^{}", var_name, i));
                    }
                }
            }
        }

        result
    }

    fn precedence(&self) -> i32 {
        // Polynomials with multiple terms need parentheses in multiplication
        if self.coeffs.iter().filter(|c| !c.is_zero()).count() > 1 {
            rustmath_typesetting::utils::precedence::ADD
        } else {
            rustmath_typesetting::utils::precedence::ATOMIC
        }
    }
}

// Ring implementation for polynomials over rings
impl<R: Ring> Ring for UnivariatePolynomial<R> {
    fn zero() -> Self {
        UnivariatePolynomial::new(vec![R::zero()])
    }

    fn one() -> Self {
        UnivariatePolynomial::new(vec![R::one()])
    }

    fn is_zero(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_zero()
    }

    fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_one()
    }
}

// CommutativeRing implementation for polynomials over commutative rings
impl<R: CommutativeRing> CommutativeRing for UnivariatePolynomial<R> {}

// IntegralDomain implementation for polynomials over integral domains
impl<R: IntegralDomain> IntegralDomain for UnivariatePolynomial<R> {}

impl<R: Ring> Add for UnivariatePolynomial<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            let b = other.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            coeffs.push(a + b);
        }

        UnivariatePolynomial::new(coeffs)
    }
}

impl<R: Ring> Sub for UnivariatePolynomial<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            let b = other.coeffs.get(i).cloned().unwrap_or_else(|| R::zero());
            coeffs.push(a - b);
        }

        UnivariatePolynomial::new(coeffs)
    }
}

impl<R: Ring> Mul for UnivariatePolynomial<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return UnivariatePolynomial::new(vec![R::zero()]);
        }

        let result_len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut coeffs = vec![R::zero(); result_len];

        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in other.coeffs.iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].clone() + a.clone() * b.clone();
            }
        }

        UnivariatePolynomial::new(coeffs)
    }
}

impl<R: Ring> Neg for UnivariatePolynomial<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let coeffs = self.coeffs.into_iter().map(|c| -c).collect();
        UnivariatePolynomial::new(coeffs)
    }
}

// Division for polynomials over fields
impl<R: Ring + EuclideanDomain> Div for UnivariatePolynomial<R> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self.div_rem(&other).unwrap().0
    }
}

impl<R: Ring + EuclideanDomain> Rem for UnivariatePolynomial<R> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        self.div_rem(&other).unwrap().1
    }
}

impl<R: Ring> UnivariatePolynomial<R> {
    /// Division with remainder (for polynomials over fields/Euclidean domains)
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self)>
    where
        R: EuclideanDomain,
    {
        if divisor.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        if self.is_zero() {
            return Ok((
                UnivariatePolynomial::new(vec![R::zero()]),
                UnivariatePolynomial::new(vec![R::zero()]),
            ));
        }

        let dividend_deg = self.degree().unwrap();
        let divisor_deg = divisor.degree().unwrap();

        if dividend_deg < divisor_deg {
            return Ok((UnivariatePolynomial::new(vec![R::zero()]), self.clone()));
        }

        let mut quotient = vec![R::zero(); dividend_deg - divisor_deg + 1];
        let mut remainder = self.clone();

        let divisor_lc = divisor.leading_coeff().unwrap().clone();

        while !remainder.is_zero() {
            let remainder_deg = match remainder.degree() {
                Some(d) => d,
                None => break,
            };

            if remainder_deg < divisor_deg {
                break;
            }

            let remainder_lc = remainder.leading_coeff().unwrap().clone();
            let (coeff_quot, _) = remainder_lc.div_rem(&divisor_lc)?;

            let shift = remainder_deg - divisor_deg;
            quotient[shift] = coeff_quot.clone();

            let mut subtrahend_coeffs = vec![R::zero(); shift];
            subtrahend_coeffs.extend(divisor.coeffs.iter().map(|c| c.clone() * coeff_quot.clone()));

            let subtrahend = UnivariatePolynomial::new(subtrahend_coeffs);
            remainder = remainder - subtrahend;
        }

        Ok((
            UnivariatePolynomial::new(quotient),
            remainder,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_creation() {
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
        ]);

        assert_eq!(p.degree(), Some(2));
        assert_eq!(p.coeff(0), &Integer::from(1));
        assert_eq!(p.coeff(1), &Integer::from(2));
        assert_eq!(p.coeff(2), &Integer::from(3));
    }

    #[test]
    fn test_eval() {
        // p(x) = 1 + 2x + 3x^2
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
        ]);

        // p(2) = 1 + 4 + 12 = 17
        assert_eq!(p.eval(&Integer::from(2)), Integer::from(17));
    }

    #[test]
    fn test_addition() {
        let p1 = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(2)]);
        let p2 = UnivariatePolynomial::new(vec![Integer::from(3), Integer::from(4)]);

        let sum = p1 + p2;
        assert_eq!(sum.coefficients(), &[Integer::from(4), Integer::from(6)]);
    }

    #[test]
    fn test_multiplication() {
        // (1 + x) * (1 + x) = 1 + 2x + x^2
        let p = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(1)]);
        let prod = p.clone() * p;

        assert_eq!(
            prod.coefficients(),
            &[Integer::from(1), Integer::from(2), Integer::from(1)]
        );
    }

    #[test]
    fn test_derivative() {
        // p(x) = 1 + 2x + 3x^2
        // p'(x) = 2 + 6x
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
        ]);

        let deriv = p.derivative();
        assert_eq!(deriv.coefficients(), &[Integer::from(2), Integer::from(6)]);
    }

    #[test]
    fn test_discriminant() {
        // Quadratic: x^2 - 5x + 6 = (x-2)(x-3)
        // Discriminant = b^2 - 4ac = 25 - 24 = 1
        let p = UnivariatePolynomial::new(vec![
            Integer::from(6),
            Integer::from(-5),
            Integer::from(1),
        ]);
        assert_eq!(p.discriminant(), Some(Integer::from(1)));

        // Quadratic: x^2 + 1 (no real roots)
        // Discriminant = 0 - 4 = -4
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        ]);
        assert_eq!(p.discriminant(), Some(Integer::from(-4)));

        // Linear polynomial: discriminant = 1
        let p = UnivariatePolynomial::new(vec![
            Integer::from(2),
            Integer::from(3),
        ]);
        assert_eq!(p.discriminant(), Some(Integer::from(1)));

        // Cubic (old closed-form path, must be unchanged): gp poldisc
        // x^3 - 2 -> -108; x^3 - x -> 4.
        let p = UnivariatePolynomial::new(vec![
            Integer::from(-2),
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
        ]);
        assert_eq!(p.discriminant(), Some(Integer::from(-108)));
        let p = UnivariatePolynomial::new(vec![
            Integer::from(0),
            Integer::from(-1),
            Integer::from(0),
            Integer::from(1),
        ]);
        assert_eq!(p.discriminant(), Some(Integer::from(4)));
    }

    /// Parse a (possibly signed) decimal string into an `Integer`, so tests can
    /// pin gp-derived expected values that exceed `i64`.
    fn int(s: &str) -> Integer {
        let (neg, digits) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        let ten = Integer::from(10);
        let mut acc = Integer::zero();
        for ch in digits.chars() {
            assert!(ch.is_ascii_digit());
            acc = acc * ten.clone() + Integer::from((ch as u8 - b'0') as i64);
        }
        if neg {
            -acc
        } else {
            acc
        }
    }

    fn poly(coeffs: &[i64]) -> UnivariatePolynomial<Integer> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| Integer::from(c)).collect())
    }

    #[test]
    fn test_discriminant_deg4_to_8_matches_pari() {
        // All expected values derived with PARI/GP poldisc.
        // deg 4
        assert_eq!(poly(&[1, 1, 0, 0, 1]).discriminant(), Some(int("229"))); // x^4+x+1
        assert_eq!(
            poly(&[5, -1, 0, 3, 2]).discriminant(), // 2x^4+3x^3-x+5 (non-monic)
            Some(int("258385"))
        );
        assert_eq!(
            poly(&[9, 0, -10, 0, 1]).discriminant(), // x^4-10x^2+9
            Some(int("589824"))
        );
        // deg 5
        assert_eq!(poly(&[-1, -1, 0, 0, 0, 1]).discriminant(), Some(int("2869"))); // x^5-x-1
        assert_eq!(
            poly(&[-4, 1, -7, 0, 2, 3]).discriminant(), // 3x^5+2x^4-7x^2+x-4
            Some(int("298866725"))
        );
        // deg 6
        assert_eq!(
            poly(&[1, 0, 0, 1, 0, 0, 1]).discriminant(), // x^6+x^3+1
            Some(int("-19683"))
        );
        assert_eq!(
            poly(&[987654321, 0, 123456789, 0, 0, -3, 1]).discriminant(),
            Some(int(
                "-3580946598583719546632278617502018645264577739978967981613067"
            ))
        );
        // deg 7
        assert_eq!(
            poly(&[3, -7, 0, 0, 0, 0, 0, 1]).discriminant(), // x^7-7x+3
            Some(int("37822859361"))
        );
        // deg 8, big coefficients
        assert_eq!(
            poly(&[11111111111111, -98765432109876, 0, 0, 123456789012345, 0, 0, 0, 1])
                .discriminant(),
            Some(int(
                "-282642315708058613219874509111064468379472805900988881933849002243064648402426721161068285050627655984412535164755330665535204869133425952458497292480904691712"
            ))
        );
        assert_eq!(
            poly(&[1, -2, 0, 3, 0, 0, -4, 0, 5]).discriminant(), // 5x^8-4x^6+3x^3-2x+1
            Some(int("49510436665"))
        );
    }

    #[test]
    fn test_discriminant_zero_for_repeated_root() {
        // (x-1)^2 (x+2) (x^2+3), degree 5 — repeated root => disc = 0 (gp poldisc).
        let f = poly(&[-1, 1]);
        let g = f.clone() * f * poly(&[2, 1]) * poly(&[3, 0, 1]);
        assert_eq!(g.degree(), Some(5));
        assert_eq!(g.discriminant(), Some(Integer::zero()));
    }

    #[test]
    fn test_resultant_battery_matches_pari() {
        // gp polresultant gates.
        assert_eq!(
            poly(&[-2, 0, 0, 1]).resultant(&poly(&[1, 1, 1])),
            int("1")
        ); // Res(x^3-2, x^2+x+1)
        assert_eq!(
            poly(&[-3, 1, 0, 0, 2]).resultant(&poly(&[7, -5, 3])),
            int("11015")
        ); // Res(2x^4+x-3, 3x^2-5x+7)
        assert_eq!(
            poly(&[-1, 0, 1, 0, 0, 1]).resultant(&poly(&[-1, -1, 0, 1])),
            int("-5")
        ); // Res(x^5+x^2-1, x^3-x-1)
        // shared factor (x-3) => 0
        let f = poly(&[-3, 1]) * poly(&[2, 0, 1]);
        let g = poly(&[-3, 1]) * poly(&[1, 1]);
        assert_eq!(f.resultant(&g), Integer::zero());
        // two nonzero constants: empty Sylvester matrix, resultant 1
        assert_eq!(poly(&[2]).resultant(&poly(&[3])), Integer::one());
        // constant vs poly: Res(c, g) = c^deg(g)
        assert_eq!(poly(&[2]).resultant(&poly(&[1, 0, 0, 1])), Integer::from(8));
    }

    #[test]
    fn test_resultant_self_certifying_identities() {
        // res(f,g) = (-1)^(deg f * deg g) res(g,f); res(f, g*h) = res(f,g) res(f,h).
        let f = poly(&[3, -1, 4, 1]); // deg 3
        let g = poly(&[-5, 9, 2]); // deg 2
        let h = poly(&[6, -5, 3, 5, 1]); // deg 4
        let sign = |a: &UnivariatePolynomial<Integer>, b: &UnivariatePolynomial<Integer>| {
            if (a.degree().unwrap() * b.degree().unwrap()) % 2 == 1 {
                -Integer::one()
            } else {
                Integer::one()
            }
        };
        assert_eq!(f.resultant(&g), sign(&f, &g) * g.resultant(&f));
        assert_eq!(f.resultant(&h), sign(&f, &h) * h.resultant(&f));
        assert_eq!(g.resultant(&h), sign(&g, &h) * h.resultant(&g));
        assert_eq!(
            f.resultant(&(g.clone() * h.clone())),
            f.resultant(&g) * f.resultant(&h)
        );
        assert_eq!(
            h.resultant(&(f.clone() * g.clone())),
            h.resultant(&f) * h.resultant(&g)
        );
    }

    #[test]
    fn test_resultant_consistent_with_discriminant() {
        // disc(f) * lc(f) = (-1)^(n(n-1)/2) * res(f, f').
        // h = 2x^4+3x^3-x+5: gp says res(h, h') = 516770, disc = 258385, lc = 2, n = 4.
        let h = poly(&[5, -1, 0, 3, 2]);
        let res = h.resultant(&h.derivative());
        assert_eq!(res, int("516770"));
        assert_eq!(
            h.discriminant().unwrap() * Integer::from(2),
            res // (-1)^(4*3/2) = +1
        );
    }

    #[test]
    fn test_resultant_deg15_x_deg12_20_digit_coeffs() {
        // The old cofactor engine was O(n!) — a 27x27 Sylvester determinant was
        // unreachable. Bareiss is O(n^3). Coefficients and expected value pinned
        // from PARI/GP:
        //   f = sum(i=0,15, (12345678901234567890 + i*1111111111111111111*(-1)^i) x^i)
        //   g = sum(i=0,12, (98765432109876543210 - i*2222222222222222222*(-1)^i) x^i)
        let base_f = int("12345678901234567890");
        let step_f = int("1111111111111111111");
        let f = UnivariatePolynomial::new(
            (0..=15i64)
                .map(|i| {
                    let delta = Integer::from(i) * step_f.clone();
                    if i % 2 == 0 {
                        base_f.clone() + delta
                    } else {
                        base_f.clone() - delta
                    }
                })
                .collect(),
        );
        let base_g = int("98765432109876543210");
        let step_g = int("2222222222222222222");
        let g = UnivariatePolynomial::new(
            (0..=12i64)
                .map(|i| {
                    let delta = Integer::from(i) * step_g.clone();
                    if i % 2 == 0 {
                        base_g.clone() - delta
                    } else {
                        base_g.clone() + delta
                    }
                })
                .collect(),
        );
        assert_eq!(f.degree(), Some(15));
        assert_eq!(g.degree(), Some(12));

        let start = std::time::Instant::now();
        let res = f.resultant(&g);
        let elapsed = start.elapsed();

        let expected = int(
            "349520787347914443056336940748715934940543594517360759162590508700077253233713533103725935023184792921984050679795684957300963039250162670301230187978113776850370014680374077115674687399683442342110830500385433756692941073454493326417631914715778670112408892491978757359559642240028526274375587289629580615175017571034642024643610299441311431311131843521814413533314648160044752685512744797908669014071336175556090111426512208380504891794684259795911653154697219538565317496181212123829753828266965056138844838445860835051268881285120",
        );
        assert_eq!(res, expected);
        // deg 15 * deg 12 is even => res(g,f) has the same sign.
        assert_eq!(g.resultant(&f), expected);
        // The old engine would not finish in the lifetime of the universe;
        // Bareiss must be fast. Keep the bound loose for slow debug builds but
        // catastrophic-failure-proof.
        assert!(
            elapsed.as_secs() < 30,
            "resultant took {elapsed:?}; the O(n!) engine is back?"
        );
        println!("deg15 x deg12 resultant took {elapsed:?}");
    }

    #[test]
    fn test_is_monic() {
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(1), // Leading coefficient
        ]);
        assert!(p.is_monic());

        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3), // Leading coefficient
        ]);
        assert!(!p.is_monic());
    }

    #[test]
    fn test_content() {
        // 6 + 9x + 12x^2, content = gcd(6, 9, 12) = 3
        let p = UnivariatePolynomial::new(vec![
            Integer::from(6),
            Integer::from(9),
            Integer::from(12),
        ]);
        assert_eq!(p.content(), Integer::from(3));

        // 2 + 4x, content = 2
        let p = UnivariatePolynomial::new(vec![
            Integer::from(2),
            Integer::from(4),
        ]);
        assert_eq!(p.content(), Integer::from(2));
    }

    #[test]
    fn test_compose() {
        // p(x) = x + 1, q(x) = 2x
        // p(q(x)) = 2x + 1
        let p = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(1)]);
        let q = UnivariatePolynomial::new(vec![Integer::from(0), Integer::from(2)]);

        let composed = p.compose(&q);
        assert_eq!(composed.coefficients(), &[Integer::from(1), Integer::from(2)]);

        // p(x) = x^2, q(x) = x + 1
        // p(q(x)) = (x+1)^2 = x^2 + 2x + 1
        let p = UnivariatePolynomial::new(vec![Integer::from(0), Integer::from(0), Integer::from(1)]);
        let q = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(1)]);

        let composed = p.compose(&q);
        assert_eq!(
            composed.coefficients(),
            &[Integer::from(1), Integer::from(2), Integer::from(1)]
        );
    }

    #[test]
    fn test_scale_variable() {
        // p(x) = x^2 + 2x + 3
        // p(2x) = 4x^2 + 4x + 3
        let p = UnivariatePolynomial::new(vec![
            Integer::from(3),
            Integer::from(2),
            Integer::from(1),
        ]);

        let scaled = p.scale_variable(&Integer::from(2));
        assert_eq!(
            scaled.coefficients(),
            &[Integer::from(3), Integer::from(4), Integer::from(4)]
        );
    }

    #[test]
    fn test_translate() {
        // p(x) = x^2, p(x + 1) = x^2 + 2x + 1
        let p = UnivariatePolynomial::new(vec![Integer::from(0), Integer::from(0), Integer::from(1)]);
        let translated = p.translate(&Integer::from(1));

        assert_eq!(
            translated.coefficients(),
            &[Integer::from(1), Integer::from(2), Integer::from(1)]
        );
    }
}
