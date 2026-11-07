//! Univariate polynomials (polynomials in one variable)
//!
//! Specification: MAGMA Handbook ch. 23, *Univariate Polynomial Rings* (digest
//! `rustmath-lava/docs/Chapter_23_Univariate_Polynomial_Rings.md`): §23.4.5 `Coefficient`;
//! §23.4.10 `Quotrem`, `div`, `mod`, `IsDivisibleBy`, `ExactQuotient`, `PseudoRemainder`;
//! §23.5 `GCD`, `LCM`, `Normalize`, `Content`, `PrimitivePart`, `ContentAndPrimitivePart`.
//!
//! Algorithms (division, pseudo-division and gcd):
//! - D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed. (1997), §4.6.1:
//!   Algorithm D (division), Algorithm R (pseudo-division), Algorithm C (subresultant gcd).
//! - G. E. Collins, *Subresultants and reduced polynomial remainder sequences*,
//!   J. ACM 14 (1967) 128-142.
//! - W. S. Brown and J. F. Traub, *On Euclid's algorithm and the theory of subresultants*,
//!   J. ACM 18 (1971) 505-514.
//! - H. Cohen, *A Course in Computational Algebraic Number Theory*, GTM 138 (1993),
//!   Alg. 3.1.2 (pseudo-division) and Alg. 3.3.1 (sub-resultant GCD).

use crate::polynomial::Polynomial;
use rustmath_core::{CommutativeRing, EuclideanDomain, IntegralDomain, MathError, Result, Ring};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::sync::OnceLock;

/// Univariate polynomial over a ring R
#[derive(Clone)]
pub struct UnivariatePolynomial<R: Ring> {
    /// Coefficients in increasing degree order: [a0, a1, a2, ...] represents a0 + a1*x + a2*x^2 + ...
    /// Trimmed (no zero leading coefficient) and never empty: the zero polynomial is `[0]`.
    coeffs: Vec<R>,
    /// The zero of `R`, created on first use and returned by [`Self::coeff`] past the degree
    /// (MAGMA `Coefficient(p, i)` is 0 there, Handbook ch. 23 §23.4.5). Created lazily
    /// because `R::zero()` panics for parent-carrying coefficient types. Not part of the
    /// value: equality ignores it.
    zero: OnceLock<R>,
}

impl<R: Ring> PartialEq for UnivariatePolynomial<R> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<R: Ring + Eq> Eq for UnivariatePolynomial<R> {}

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

        UnivariatePolynomial {
            coeffs,
            zero: OnceLock::new(),
        }
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

    /// The coefficient of `x^degree` (MAGMA `Coefficient(p, i)`, Handbook ch. 23 §23.4.5):
    /// zero when `degree` exceeds the degree of `self`.
    pub fn coeff(&self, degree: usize) -> &R {
        match self.coeffs.get(degree) {
            Some(c) => c,
            None => self.zero.get_or_init(R::zero),
        }
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

        // Square-free ⟺ gcd(f, f') is a nonzero constant. `gcd` is normalized, but over a
        // non-field it keeps the content (gcd(4x + 4, 4) = 4 over Z), so test
        // `degree == 0` rather than `is_one()`.
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

    /// Panicking form of [`Self::div_rem`] (MAGMA `Quotrem`, Handbook ch. 23 §23.4.10).
    ///
    /// Returns `(q, r)` with `self = q * divisor + r` and `deg r` minimal; see
    /// [`Self::div_rem`] for the convention over a non-field. Panics with
    /// "Division by zero polynomial" for a zero divisor.
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
    /// assert_eq!(quo, UnivariatePolynomial::new(vec![Integer::from(2), Integer::from(1)]));
    /// assert!(rem.is_zero());
    /// ```
    pub fn quo_rem(&self, divisor: &Self) -> (Self, Self)
    where
        R: EuclideanDomain,
    {
        match self.div_rem(divisor) {
            Ok(qr) => qr,
            Err(MathError::DivisionByZero) => panic!("Division by zero polynomial"),
            Err(e) => panic!("quo_rem: {e}"),
        }
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
        UnivariatePolynomial::coeff(self, degree)
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

/// MAGMA `f div g` (Handbook ch. 23 §23.4.10): the quotient of [`UnivariatePolynomial::div_rem`].
///
/// This is division with a minimal-degree remainder, not a claim that `R[x]` is Euclidean
/// (`Z[x]` is not; no `EuclideanDomain` impl exists for polynomials). Panics where
/// `div_rem` returns `Err` (a zero divisor, or a lawless coefficient ring).
impl<R: Ring + EuclideanDomain> Div for UnivariatePolynomial<R> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self.div_rem(&other)
            .unwrap_or_else(|e| panic!("polynomial div/mod: {e}"))
            .0
    }
}

/// MAGMA `f mod g` (Handbook ch. 23 §23.4.10): the remainder of
/// [`UnivariatePolynomial::div_rem`]. Panics where `div_rem` returns `Err`.
impl<R: Ring + EuclideanDomain> Rem for UnivariatePolynomial<R> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        self.div_rem(&other)
            .unwrap_or_else(|e| panic!("polynomial div/mod: {e}"))
            .1
    }
}

/// `a / b` in `R` for a division that theory guarantees exact (content removal, the
/// subresultant recurrence, a unit inverse). Panics, rather than returning a wrong value,
/// if it is not exact: then `R` is not an integral domain with a lawful `div_rem` (for
/// example GF(p) with p composite). Same policy as the Bareiss determinant above.
fn exact_div_elem<R: EuclideanDomain>(a: &R, b: &R) -> R {
    let (q, r) = a
        .div_rem(b)
        .unwrap_or_else(|e| panic!("UnivariatePolynomial: exact coefficient division failed: {e}"));
    assert!(
        r.is_zero() && q.clone() * b.clone() == *a,
        "UnivariatePolynomial: a coefficient division that theory guarantees exact was not \
         exact; the coefficient ring is not an integral domain with exact division"
    );
    q
}

/// The inverse of a unit `u` of `R`, by exact division (see [`exact_div_elem`]).
fn inverse_of_unit<R: EuclideanDomain>(u: &R) -> R {
    exact_div_elem(&R::one(), u)
}

/// The canonical associate of a coefficient (MAGMA `Normalize` on `R`): `|c|` over Z, `1` for
/// a nonzero element of a field, `0` for `0`, and `c` itself when `R` names no canonical
/// associate ([`EuclideanDomain::normalization_unit`] is `None`).
fn normalize_elem<R: EuclideanDomain>(c: &R) -> R {
    match c.normalization_unit() {
        Some(u) if !u.is_one() => exact_div_elem(c, &u),
        _ => c.clone(),
    }
}

/// Remove trailing (high-degree) zeros from a coefficient vector.
fn trim_top<R: Ring>(v: &mut Vec<R>) {
    while v.last().is_some_and(|c| c.is_zero()) {
        v.pop();
    }
}

// Division, pseudo-division, gcd, lcm and content (MAGMA Handbook ch. 23 §23.4.10, §23.5).
impl<R: Ring> UnivariatePolynomial<R> {
    /// Quotient and remainder (MAGMA `Quotrem`, Handbook ch. 23 §23.4.10): `(q, r)` with
    /// `self = q * divisor + r` exactly and `deg r` minimal over all `q` in `R[x]`.
    ///
    /// * `divisor = 0` gives `Err(DivisionByZero)`; `self = 0` gives `(0, 0)`;
    ///   `deg self < deg divisor` gives `(0, self)`.
    /// * `lc(divisor)` a unit (every nonzero divisor over a field, `lc = ±1` over Z):
    ///   classical division (Knuth, TAOCP vol. 2 §4.6.1, Algorithm D), `deg r < deg divisor`,
    ///   and `(q, r)` is unique.
    /// * `lc(divisor)` not a unit (for example `2x` in `Z[x]`): leading terms are cancelled
    ///   while `lc(divisor)` divides the remainder's leading coefficient exactly in `R`, and
    ///   the division stops at the first one it does not divide. Then `deg r` is minimal
    ///   (the coefficient of `x^(deg r)` in `self - q'·divisor` is `lc(r) - t·lc(divisor) != 0`
    ///   for every `q'`), and `r = 0` exactly when `divisor` divides `self` in `R[x]`. Here
    ///   `deg r >= deg divisor` is possible (`x^2 = 0·(2x) + x^2` over Z), so this is not a
    ///   Euclidean step; use [`Self::gcd`] or [`Self::pseudo_quo_rem`] for that. The
    ///   remainder of minimal degree is not unique over a non-field; FLINT's
    ///   `fmpz_poly_divrem`, which keeps reducing each coefficient modulo `lc(divisor)`, can
    ///   return a different remainder of the same degree.
    /// * Never loops: every step lowers `deg r`, so there are at most
    ///   `deg self - deg divisor + 1` steps. If a step does not cancel the leading term
    ///   (a coefficient ring whose `div_rem` reports an exact quotient that is not one), the
    ///   result is `Err(InvalidOperation)`, not a wrong value.
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self)>
    where
        R: EuclideanDomain,
    {
        let (q, r) = self.division(divisor, true)?;
        Ok((q.unwrap_or_else(Self::zero), r))
    }

    /// The common core of [`Self::div_rem`] and the gcd. The quotient is returned only when
    /// `want_quotient` is set (the gcd needs only the remainder).
    fn division(&self, divisor: &Self, want_quotient: bool) -> Result<(Option<Self>, Self)>
    where
        R: EuclideanDomain,
    {
        let Some(dg) = divisor.degree() else {
            return Err(MathError::DivisionByZero);
        };
        let Some(df) = self.degree() else {
            return Ok((want_quotient.then(Self::zero), Self::zero()));
        };
        if df < dg {
            return Ok((want_quotient.then(Self::zero), self.clone()));
        }
        let lg = &divisor.coeffs[dg];
        let mut q: Vec<R> = if want_quotient {
            vec![R::zero(); df - dg + 1]
        } else {
            Vec::new()
        };
        // Working remainder, kept trimmed: empty for 0, otherwise its last entry is nonzero.
        let mut r: Vec<R> = self.coeffs.clone();
        while let Some(lr) = r.last() {
            let dr = r.len() - 1;
            if dr < dg {
                break;
            }
            let (c, rho) = lr.div_rem(lg)?;
            if !rho.is_zero() {
                // lc(divisor) does not divide lc(r) in R: r has minimal degree.
                break;
            }
            let s = dr - dg;
            // r <- r - c·x^s·divisor (c on the left, as in the product q·divisor).
            for (j, gj) in divisor.coeffs.iter().enumerate() {
                r[s + j] = r[s + j].clone() - c.clone() * gj.clone();
            }
            if !r[dr].is_zero() {
                return Err(MathError::InvalidOperation(
                    "div_rem: the coefficient ring's div_rem reported an exact quotient of the \
                     leading coefficients, but subtracting it did not cancel the leading term"
                        .into(),
                ));
            }
            if want_quotient {
                // Each shift s is visited once: deg r strictly drops.
                q[s] = c;
            }
            trim_top(&mut r);
        }
        Ok((want_quotient.then(|| Self::new(q)), Self::new(r)))
    }

    /// MAGMA `ExactQuotient(f, g)` (Handbook ch. 23 §23.4.10): `Ok(q)` with `self = q * g`.
    ///
    /// `Err(DivisionByZero)` for `g = 0`; `Err(NotDivisible)` when `g` does not divide
    /// `self` in `R[x]` (for example `x^2` by `2x` over Z, whose quotient `x/2` is not
    /// integral).
    pub fn exact_quotient(&self, g: &Self) -> Result<Self>
    where
        R: EuclideanDomain,
    {
        let (q, r) = self.div_rem(g)?;
        if r.is_zero() {
            Ok(q)
        } else {
            Err(MathError::NotDivisible(
                "ExactQuotient: the divisor does not divide the dividend in R[x]".into(),
            ))
        }
    }

    /// MAGMA `IsDivisibleBy(f, g)` (Handbook ch. 23 §23.4.10): `Ok(Some(q))` with
    /// `self = q * g` when `g` divides `self` in `R[x]`, `Ok(None)` when it does not.
    ///
    /// `Err(DivisionByZero)` for `g = 0`, as for [`Self::exact_quotient`] (for `self = 0`
    /// every `q` satisfies `0 = q * 0`, so there is no quotient to return).
    pub fn is_divisible_by(&self, g: &Self) -> Result<Option<Self>>
    where
        R: EuclideanDomain,
    {
        let (q, r) = self.div_rem(g)?;
        Ok(r.is_zero().then_some(q))
    }

    /// Pseudo-division (MAGMA `PseudoRemainder`, Handbook ch. 23 §23.4.10; Knuth, TAOCP
    /// vol. 2 §4.6.1, Algorithm R; Cohen, GTM 138, Alg. 3.1.2): `(q, r)` with
    /// `lc(g)^d · self = q * g + r` and `deg r < deg g`, where
    /// `d = max(0, deg self - deg g + 1)`. Unique over an integral domain.
    ///
    /// The exponent `d` is the full one even when a step cancels more than one degree (so
    /// `pseudo_quo_rem(3x^4 + x + 7, 2x^2 + 1) = (12x^2 - 6, 8x + 62)`, the PARI value of
    /// `[(4*f)\g, (4*f)%g]`). `Err(DivisionByZero)` for `g = 0`; `Err(InvalidOperation)` if
    /// a step does not cancel the leading term (non-commutative or lawless `R`), so it never
    /// loops.
    pub fn pseudo_quo_rem(&self, g: &Self) -> Result<(Self, Self)>
    where
        R: IntegralDomain,
    {
        let (q, r) = self.pseudo_division(g, true)?;
        Ok((q.unwrap_or_else(Self::zero), r))
    }

    /// MAGMA `PseudoRemainder(f, g)`: the `r` of [`Self::pseudo_quo_rem`].
    pub fn pseudo_remainder(&self, g: &Self) -> Result<Self>
    where
        R: IntegralDomain,
    {
        Ok(self.pseudo_division(g, false)?.1)
    }

    /// The common core of [`Self::pseudo_quo_rem`] and the subresultant gcd.
    fn pseudo_division(&self, g: &Self, want_quotient: bool) -> Result<(Option<Self>, Self)>
    where
        R: IntegralDomain,
    {
        let Some(dg) = g.degree() else {
            return Err(MathError::DivisionByZero);
        };
        let Some(df) = self.degree() else {
            return Ok((want_quotient.then(Self::zero), Self::zero()));
        };
        if df < dg {
            // d = 0
            return Ok((want_quotient.then(Self::zero), self.clone()));
        }
        let d = df - dg + 1;
        let c = &g.coeffs[dg];
        let mut q: Vec<R> = if want_quotient {
            vec![R::zero(); d]
        } else {
            Vec::new()
        };
        let mut r: Vec<R> = self.coeffs.clone();
        let mut steps = 0usize;
        // Invariant: c^steps · self = q·g + r.
        while let Some(lr) = r.last().cloned() {
            let dr = r.len() - 1;
            if dr < dg {
                break;
            }
            let s = dr - dg;
            if want_quotient {
                // q <- c·q + lr·x^s
                for qi in q.iter_mut() {
                    *qi = c.clone() * qi.clone();
                }
                q[s] = q[s].clone() + lr.clone();
            }
            // r <- c·r - lr·x^s·g
            for ri in r.iter_mut() {
                *ri = c.clone() * ri.clone();
            }
            for (j, gj) in g.coeffs.iter().enumerate() {
                r[s + j] = r[s + j].clone() - lr.clone() * gj.clone();
            }
            if !r[dr].is_zero() {
                return Err(MathError::InvalidOperation(
                    "pseudo_quo_rem: c·lc(r) - lc(r)·c did not cancel the leading term; the \
                     coefficient ring is not commutative or its arithmetic is lawless"
                        .into(),
                ));
            }
            trim_top(&mut r);
            steps += 1;
        }
        // Scale up to the full exponent d.
        let missing = d - steps;
        if missing > 0 {
            let ce = c.pow(missing as u32);
            for ri in r.iter_mut() {
                *ri = ce.clone() * ri.clone();
            }
            if want_quotient {
                for qi in q.iter_mut() {
                    *qi = ce.clone() * qi.clone();
                }
            }
        }
        Ok((want_quotient.then(|| Self::new(q)), Self::new(r)))
    }

    /// MAGMA `Normalize(f)` (Handbook ch. 23 §23.5.1): the canonical associate `u⁻¹·f`,
    /// `u` the [`EuclideanDomain::normalization_unit`] of the leading coefficient. Over a
    /// field it is monic; over Z (also `i32`, `i64`) its leading coefficient is positive;
    /// `0` stays `0`. If `R` names no canonical associate of the leading coefficient
    /// (`normalization_unit` is `None`; no coefficient ring in this workspace does that),
    /// `self` is returned unchanged.
    pub fn normalize(&self) -> Self
    where
        R: EuclideanDomain,
    {
        let Some(lc) = self.leading_coeff() else {
            return self.clone();
        };
        match lc.normalization_unit() {
            Some(u) if !u.is_one() => self.scalar_mul(&inverse_of_unit(&u)),
            _ => self.clone(),
        }
    }

    /// The gcd of the coefficients through [`EuclideanDomain::gcd`], up to a unit; `0` for
    /// the zero polynomial.
    fn raw_content(&self) -> R
    where
        R: EuclideanDomain,
    {
        let mut g = self.coeffs[0].clone();
        for c in &self.coeffs[1..] {
            if g.is_one() || (-g.clone()).is_one() {
                break;
            }
            g = EuclideanDomain::gcd(&g, c);
        }
        g
    }

    /// Every coefficient divided exactly by the nonzero `c` (see [`exact_div_elem`]).
    fn exact_scalar_div(&self, c: &R) -> Self
    where
        R: EuclideanDomain,
    {
        if c.is_one() {
            return self.clone();
        }
        Self::new(self.coeffs.iter().map(|a| exact_div_elem(a, c)).collect())
    }

    /// MAGMA `Content(p)` (Handbook ch. 23 §23.5.2): the gcd of the coefficients in `R`,
    /// normalized. Positive over Z (`content(-6 - 9x) = 3`, PARI `content(-6-9*x)`); `0` for
    /// the zero polynomial; `1` for a nonzero polynomial over a field (every nonzero element
    /// of a field is a unit, so the ring-theoretic gcd is 1). PARI's `content` over Q is a
    /// different notion (the Z-module content, `content(2/3*x+4/5) = 2/15`) and is not
    /// provided here.
    pub fn content(&self) -> R
    where
        R: EuclideanDomain,
    {
        normalize_elem(&self.raw_content())
    }

    /// MAGMA `PrimitivePart(p)` (Handbook ch. 23 §23.5.2): `p` divided by its content (so
    /// over Z the sign is kept: `primitive_part(-6 - 9x) = -2 - 3x`); `0` for `0`; `p` itself
    /// over a field.
    pub fn primitive_part(&self) -> Self
    where
        R: EuclideanDomain,
    {
        self.content_and_primitive_part().1
    }

    /// MAGMA `ContentAndPrimitivePart(p)` (Handbook ch. 23 §23.5.2): `(content, p / content)`;
    /// `(0, 0)` for `0`.
    pub fn content_and_primitive_part(&self) -> (R, Self)
    where
        R: EuclideanDomain,
    {
        if self.is_zero() {
            return (R::zero(), Self::zero());
        }
        let c = self.content();
        let pp = self.exact_scalar_div(&c);
        (c, pp)
    }

    /// MAGMA `GCD(f, g)` (Handbook ch. 23 §23.5.1): the normalized greatest common divisor
    /// in `R[x]` (see [`Self::normalize`]): monic over a field; over Z (`i32`, `i64`) it
    /// carries the content `gcd(content f, content g)` and has positive leading coefficient
    /// (`gcd(2x^2 - 2, 4x + 4) = 2x + 2`, PARI `gcd(2*x^2-2, 4*x+4)`). With one input zero
    /// it is the other one, normalized; `gcd(0, 0) = 0`.
    ///
    /// Algorithm: Euclid's algorithm while the division makes progress, which over a field is
    /// the whole computation (it is a valid gcd step over any ring: `a = q·b + r` gives the
    /// same common divisors for `(a, b)` and `(b, r)`). At the first division that stops at a
    /// leading coefficient `lc(b)` does not divide (only over a non-field), the rest is the
    /// sub-resultant algorithm on the primitive parts with the content split off by Gauss'
    /// lemma (Collins 1967; Brown and Traub 1971; Knuth §4.6.1 Algorithm C; Cohen GTM 138
    /// Alg. 3.3.1), the generic algorithm MAGMA uses over "any other ring". MAGMA uses
    /// GCDHEU and modular algorithms over Z and Q; the result is the same, the speed is not.
    ///
    /// Panics only if a division that theory guarantees exact is not exact, which means `R`
    /// is not an integral domain with a lawful `div_rem` (for example GF(p), p composite).
    pub fn gcd(&self, other: &Self) -> Self
    where
        R: EuclideanDomain,
    {
        if self.is_zero() {
            return other.normalize();
        }
        if other.is_zero() {
            return self.normalize();
        }
        let (mut a, mut b) = if self.degree() >= other.degree() {
            (self.clone(), other.clone())
        } else {
            (other.clone(), self.clone())
        };
        loop {
            let (_, r) = a
                .division(&b, false)
                .unwrap_or_else(|e| panic!("UnivariatePolynomial::gcd: {e}"));
            if r.is_zero() {
                return b.normalize();
            }
            if r.degree() < b.degree() {
                a = std::mem::replace(&mut b, r);
                continue;
            }
            // `division` stopped at a leading coefficient lc(b) does not divide. Since
            // r = a - q·b, gcd(a, b) = gcd(b, r); both are nonzero.
            return Self::subresultant_gcd(b, r);
        }
    }

    /// The sub-resultant gcd of two nonzero polynomials (Cohen GTM 138 Alg. 3.3.1), with
    /// content, normalized.
    fn subresultant_gcd(f: Self, g: Self) -> Self
    where
        R: EuclideanDomain,
    {
        let cf = f.raw_content();
        let cg = g.raw_content();
        let d = EuclideanDomain::gcd(&cf, &cg);
        let mut a = f.exact_scalar_div(&cf);
        let mut b = g.exact_scalar_div(&cg);
        if a.degree() < b.degree() {
            std::mem::swap(&mut a, &mut b);
        }
        let (mut lead, mut h) = (R::one(), R::one());
        let last = loop {
            let delta = a.degree().unwrap() - b.degree().unwrap();
            // Full exponent delta + 1 (the exact divisions below depend on it).
            let (_, r) = a
                .pseudo_division(&b, false)
                .unwrap_or_else(|e| panic!("UnivariatePolynomial::gcd: {e}"));
            match r.degree() {
                None => break b,              // gcd of the primitive parts is pp(b)
                Some(0) => break Self::one(), // the primitive parts are coprime
                Some(_) => {}
            }
            let den = lead.clone() * h.pow(delta as u32);
            a = b;
            // Exact by the subresultant theorem.
            b = r.exact_scalar_div(&den);
            lead = a.leading_coeff().unwrap().clone();
            h = match delta {
                0 => h,
                1 => lead.clone(),
                _ => exact_div_elem(&lead.pow(delta as u32), &h.pow(delta as u32 - 1)),
            };
        };
        let pp_last = last.exact_scalar_div(&last.raw_content());
        pp_last.scalar_mul(&d).normalize()
    }

    /// MAGMA `LCM(f, g)` (Handbook ch. 23 §23.5.1): `Normalize((f div GCD(f, g)) * g)`, and
    /// `0` if either input is `0`. Monic over a field (`lcm(2x, x) = x` over Q); positive
    /// leading coefficient over Z (`lcm(x^2, 2x) = 2x^2`, PARI `lcm(x^2, 2*x)`).
    pub fn lcm(&self, other: &Self) -> Self
    where
        R: EuclideanDomain,
    {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let g = self.gcd(other);
        // Exact: g divides self in R[x], and then `div_rem` returns remainder 0.
        let q = self
            .exact_quotient(&g)
            .unwrap_or_else(|e| panic!("UnivariatePolynomial::lcm: {e}"));
        (q * other.clone()).normalize()
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

    // MAGMA `Coefficient(p, i)` is 0 for i > deg p (Handbook ch. 23 §23.4.5).
    // PARI/GP: `polcoef(x+5,3)` -> 0, `polcoef(x^2+1,5)` -> 0, `polcoef(x+5,0)` -> 5.
    #[test]
    fn test_coeff_past_degree_is_zero() {
        let p = UnivariatePolynomial::new(vec![Integer::from(5), Integer::from(1)]);
        assert_eq!(p.coeff(0), &Integer::from(5));
        assert_eq!(p.coeff(1), &Integer::from(1));
        assert_eq!(p.coeff(2), &Integer::from(0));
        assert_eq!(p.coeff(3), &Integer::from(0));
        assert_eq!(
            <UnivariatePolynomial<Integer> as Polynomial>::coeff(&p, 3),
            &Integer::from(0)
        );
        assert_eq!(
            <UnivariatePolynomial<Integer> as Polynomial>::coeff(&p, 0),
            &Integer::from(5)
        );

        let zero = UnivariatePolynomial::<Integer>::zero();
        assert_eq!(zero.coeff(0), &Integer::from(0));
        assert_eq!(zero.coeff(7), &Integer::from(0));

        let q =
            UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(0), Integer::from(1)]);
        assert_eq!(q.coeff(5), &Integer::from(0));
        assert_eq!(q.coeff(2), &Integer::from(1));
    }

    // The lazily created zero is not part of the value.
    #[test]
    fn test_equality_ignores_the_zero_sentinel() {
        let p = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(2)]);
        let fresh = p.clone();
        assert!(p.coeff(9).is_zero()); // initializes the sentinel of p only
        assert_eq!(p, fresh);
        assert_eq!(p.clone(), p);
        assert_eq!(
            UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(2)]),
            UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(2), Integer::from(0)])
        );
        assert_ne!(
            p,
            UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(3)])
        );
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
