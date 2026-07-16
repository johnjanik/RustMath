//! Continued fractions representation and operations

use crate::Rational;
use rug::Float;
use rustmath_core::EuclideanDomain;
use rustmath_integers::Integer;

/// Recognize the best rational approximation of a real number `x` whose
/// denominator does not exceed `max_denom`, using the continued-fraction
/// expansion computed directly from the real value.
///
/// This is a *from-real* recognizer, distinct from [`Rational::from_f64`]
/// (an f64-only linear search over denominators up to a fixed 10^6, with an
/// f64-epsilon early exit — neither arbitrary-precision nor caller-bounded)
/// and from the modulus-driven rational reconstruction used in number-field
/// recognition.
///
/// # Algorithm
///
/// The partial quotients `a_k = floor(x_k)` are generated in `rug::Float`
/// arithmetic (`x_{k+1} = 1 / (x_k - a_k)`), and the convergents `p_k/q_k` are
/// built with the standard recurrence. Generation stops as soon as a convergent
/// would have `q_k > max_denom` (or the remainder underflows the working
/// precision, or a precision-derived iteration bound is hit — so an irrational
/// input never loops forever). The returned value is the genuinely closest
/// rational with denominator `<= max_denom`: it is either the last convergent
/// `p_{m-1}/q_{m-1}` or the largest feasible *semiconvergent*
/// `(p_{m-2} + t·p_{m-1}) / (q_{m-2} + t·q_{m-1})`, whichever lies nearer `x`
/// (the classical "half rule", with an exact-tie fallback that compares the two
/// candidate distances directly).
///
/// `max_denom` is treated as `max(max_denom, 1)` (a denominator must be
/// positive). Panics if `x` is not finite (NaN/inf have no rational value).
///
/// # Examples
///
/// ```
/// use rug::Float;
/// use rustmath_integers::Integer;
/// use rustmath_rationals::continued_fraction::from_real;
///
/// let pi = Float::with_val(256, rug::float::Constant::Pi);
/// let approx = from_real(&pi, &Integer::from(200));
/// assert_eq!(approx.numerator(), &Integer::from(355));
/// assert_eq!(approx.denominator(), &Integer::from(113));
/// ```
pub fn from_real(x: &Float, max_denom: &Integer) -> Rational {
    assert!(
        x.is_finite(),
        "from_real: x must be finite (NaN/inf have no rational value)"
    );

    let prec = x.prec();
    let one = Integer::one();
    // A denominator must be positive; clamp a non-positive bound to 1
    // (best rational with denominator <= 1 is the nearest integer).
    let max_eff = if *max_denom < one {
        one.clone()
    } else {
        max_denom.clone()
    };

    // Convergent recurrence state: P_{n-1} = (h1, k1), P_{n-2} = (h2, k2),
    // seeded with P_{-1} = 1/0 and P_{-2} = 0/1.
    let mut h2 = Integer::zero();
    let mut k2 = Integer::one();
    let mut h1 = Integer::one();
    let mut k1 = Integer::zero();

    let mut r = Float::with_val(prec, x);
    let one_f = Float::with_val(prec, 1);
    // A value known to `prec` bits has at most ~prec meaningful partial
    // quotients; this hard cap guarantees termination on an irrational input.
    let max_iters = prec as usize + 64;

    // The partial quotient a_m that first overflows max_denom, if any.
    let mut overflow_a: Option<Integer> = None;

    for _ in 0..max_iters {
        let floor_f = r.clone().floor();
        let a = float_floor_to_integer(&floor_f);

        let h = a.clone() * h1.clone() + h2.clone();
        let k = a.clone() * k1.clone() + k2.clone();
        if k > max_eff {
            overflow_a = Some(a);
            break;
        }

        // Accept the convergent P_n = (h, k).
        h2 = h1;
        k2 = k1;
        h1 = h;
        k1 = k;

        let frac = Float::with_val(prec, &r - &floor_f);
        if frac.is_zero() {
            // x is captured exactly (to the working precision); P_n is exact.
            return Rational::new(h1, k1).unwrap();
        }
        r = Float::with_val(prec, &one_f / &frac);
    }

    let a_m = match overflow_a {
        // Ran out of precision without ever exceeding max_denom: the best
        // approximation visible at this precision is the last convergent.
        None => return Rational::new(h1, k1).unwrap(),
        Some(a) => a,
    };

    // Best-approximation refinement. The last accepted convergent is
    // P_{m-1} = (h1, k1); the one before is P_{m-2} = (h2, k2). The closest
    // rational with denominator <= max_denom on the far side of x is the
    // semiconvergent with the largest feasible multiplier
    //   t = floor((max_denom - q_{m-2}) / q_{m-1}),   0 <= t < a_m,
    // and the overall best is the closer of that semiconvergent and P_{m-1}.
    let t = (max_eff.clone() - k2.clone()) / k1.clone();
    let hs = h2.clone() + t.clone() * h1.clone();
    let ks = k2.clone() + t.clone() * k1.clone();

    let two_t = t.clone() + t.clone();
    let choose_semi = if two_t > a_m {
        true
    } else if two_t < a_m {
        false
    } else {
        // Exact half-way tie: decide by the genuine distances to x.
        distance_to(x, &hs, &ks, prec) < distance_to(x, &h1, &k1, prec)
    };

    if choose_semi {
        Rational::new(hs, ks).unwrap()
    } else {
        Rational::new(h1, k1).unwrap()
    }
}

/// Convenience wrapper of [`from_real`] for `f64` inputs.
///
/// The double is loaded losslessly into a 53-bit `rug::Float` (its exact IEEE
/// value) before the best-approximation search runs. Panics on NaN/inf.
pub fn from_f64(x: f64, max_denom: &Integer) -> Rational {
    from_real(&Float::with_val(53, x), max_denom)
}

/// Convert an already-integral `rug::Float` (e.g. the output of `.floor()`) to
/// an [`Integer`] exactly, via its decimal digits (no `f64` truncation).
fn float_floor_to_integer(f: &Float) -> Integer {
    let z = f
        .to_integer()
        .expect("from_real: floor of a finite value is an integer");
    rug_integer_to_integer(&z)
}

/// Exact `rug::Integer` -> [`Integer`] via the decimal representation.
fn rug_integer_to_integer(z: &rug::Integer) -> Integer {
    let big = num_bigint::BigInt::parse_bytes(z.to_string().as_bytes(), 10)
        .expect("from_real: rug integer parses as decimal");
    Integer::from(big)
}

/// Exact [`Integer`] -> `rug::Integer` via the decimal representation.
fn integer_to_rug(n: &Integer) -> rug::Integer {
    rug::Integer::from_str_radix(&n.to_string(), 10)
        .expect("from_real: integer parses as decimal")
}

/// `|x - num/den|` evaluated at `prec` bits.
fn distance_to(x: &Float, num: &Integer, den: &Integer, prec: u32) -> Float {
    let nf = Float::with_val(prec, integer_to_rug(num));
    let df = Float::with_val(prec, integer_to_rug(den));
    let approx = Float::with_val(prec, &nf / &df);
    Float::with_val(prec, x - &approx).abs()
}

/// A continued fraction representation
///
/// Represents a number as: a₀ + 1/(a₁ + 1/(a₂ + 1/(a₃ + ...)))
///
/// Stored as the sequence [a₀, a₁, a₂, a₃, ...]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuedFraction {
    /// The coefficients of the continued fraction
    coefficients: Vec<Integer>,
}

impl ContinuedFraction {
    /// Create a new continued fraction from coefficients
    pub fn new(coefficients: Vec<Integer>) -> Self {
        ContinuedFraction { coefficients }
    }

    /// Create a continued fraction from a rational number
    ///
    /// Uses the Euclidean algorithm to compute the continued fraction expansion
    pub fn from_rational(r: &Rational) -> Self {
        let mut coefficients = Vec::new();
        let mut num = r.numerator().clone();
        let mut den = r.denominator().clone();

        // Euclidean algorithm
        while !den.is_zero() {
            let (q, rem) = num.div_rem(&den).unwrap();
            coefficients.push(q);
            num = den;
            den = rem;
        }

        ContinuedFraction { coefficients }
    }

    /// Convert the continued fraction back to a rational number
    pub fn to_rational(&self) -> Rational {
        if self.coefficients.is_empty() {
            return Rational::new(0, 1).unwrap();
        }

        // Work backwards through the continued fraction
        let mut numerator = Integer::one();
        let mut denominator = Integer::zero();

        for coeff in self.coefficients.iter().rev() {
            // Add the coefficient
            let new_num = coeff.clone() * numerator.clone() + denominator;
            denominator = numerator;
            numerator = new_num;
        }

        Rational::new(numerator, denominator).unwrap()
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[Integer] {
        &self.coefficients
    }

    /// Compute the nth convergent of the continued fraction
    ///
    /// The nth convergent is the rational approximation using the first n+1 coefficients
    pub fn convergent(&self, n: usize) -> Rational {
        if n >= self.coefficients.len() {
            return self.to_rational();
        }

        let coeffs: Vec<Integer> = self.coefficients[..=n].to_vec();
        let cf = ContinuedFraction::new(coeffs);
        cf.to_rational()
    }

    /// Get all convergents up to the full continued fraction
    pub fn all_convergents(&self) -> Vec<Rational> {
        (0..self.coefficients.len())
            .map(|i| self.convergent(i))
            .collect()
    }

    /// Check if this is a finite continued fraction
    pub fn is_finite(&self) -> bool {
        // All continued fractions from rationals are finite
        true
    }

    /// Get the length of the continued fraction
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }
}

/// A periodic continued fraction representation
///
/// Represents a number as: [a₀, a₁, ..., aₙ; repeating₀, repeating₁, ...]
/// where the "repeating" part repeats indefinitely.
///
/// This is used for quadratic irrationals like √2 = [1; 2, 2, 2, ...] = [1; (2)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeriodicContinuedFraction {
    /// The initial (non-repeating) coefficients
    initial: Vec<Integer>,
    /// The repeating coefficients
    repeating: Vec<Integer>,
}

impl PeriodicContinuedFraction {
    /// Create a new periodic continued fraction
    ///
    /// # Arguments
    /// * `initial` - The initial non-repeating coefficients
    /// * `repeating` - The repeating part
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::continued_fraction::PeriodicContinuedFraction;
    /// use rustmath_integers::Integer;
    ///
    /// // √2 = [1; (2)]
    /// let sqrt2 = PeriodicContinuedFraction::new(
    ///     vec![Integer::from(1)],
    ///     vec![Integer::from(2)]
    /// );
    /// ```
    pub fn new(initial: Vec<Integer>, repeating: Vec<Integer>) -> Self {
        if repeating.is_empty() {
            panic!("Repeating part cannot be empty for periodic continued fraction");
        }
        PeriodicContinuedFraction { initial, repeating }
    }

    /// Create a periodic continued fraction for √n
    ///
    /// Computes the periodic continued fraction representation of √n
    /// using the standard algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::continued_fraction::PeriodicContinuedFraction;
    /// use rustmath_integers::Integer;
    ///
    /// // √2 = [1; (2)]
    /// let sqrt2 = PeriodicContinuedFraction::from_sqrt(&Integer::from(2)).unwrap();
    /// assert_eq!(sqrt2.initial(), &[Integer::from(1)]);
    /// assert_eq!(sqrt2.repeating(), &[Integer::from(2)]);
    /// ```
    pub fn from_sqrt(n: &Integer) -> Option<Self> {
        // Check if n is a perfect square
        let sqrt_n = n.sqrt().ok()?;
        if &(sqrt_n.clone() * sqrt_n.clone()) == n {
            // Perfect square - not a quadratic irrational
            return None;
        }

        // Canonical (m, d, a) recurrence for √n with a₀ = floor(√n):
        //   m₀ = 0, d₀ = 1, a₀ = floor(√n)
        //   m_{k+1} = d_k·a_k − m_k
        //   d_{k+1} = (n − m_{k+1}²) / d_k   (always an exact division, d_{k+1} > 0)
        //   a_{k+1} = floor((a₀ + m_{k+1}) / d_{k+1})
        //
        // For √n the expansion is [a₀; (a₁, ..., a_p)] — the non-repeating part
        // is exactly [a₀] — and by the classical theorem the period ends at the
        // first k ≥ 1 with a_k = 2·a₀ (equivalently d_k = 1), so the recurrence
        // is guaranteed to terminate for every non-square n > 0.
        let a0 = sqrt_n;
        let two_a0 = a0.clone() + a0.clone();

        let mut m = Integer::zero();
        let mut d = Integer::one();
        let mut a = a0.clone();

        let mut repeating = Vec::new();
        loop {
            m = &d * &a - m;
            d = (n.clone() - &m * &m) / d;

            if d.is_zero() {
                // Unreachable for non-square n > 0; guard against division by zero.
                return None;
            }

            a = (&a0 + &m) / d.clone();
            repeating.push(a.clone());

            if a == two_a0 {
                break;
            }
        }

        Some(PeriodicContinuedFraction {
            initial: vec![a0],
            repeating,
        })
    }

    /// Get the initial (non-repeating) coefficients
    pub fn initial(&self) -> &[Integer] {
        &self.initial
    }

    /// Get the repeating coefficients
    pub fn repeating(&self) -> &[Integer] {
        &self.repeating
    }

    /// Get the nth coefficient
    ///
    /// This includes both the initial and repeating parts
    pub fn get_coefficient(&self, n: usize) -> Integer {
        if n < self.initial.len() {
            self.initial[n].clone()
        } else {
            let idx = (n - self.initial.len()) % self.repeating.len();
            self.repeating[idx].clone()
        }
    }

    /// Compute the nth convergent
    ///
    /// Returns a rational approximation using the first n+1 terms
    pub fn convergent(&self, n: usize) -> Rational {
        if n == 0 {
            return Rational::new(self.get_coefficient(0), Integer::one()).unwrap();
        }

        // Build continued fraction with first n+1 terms
        let coeffs: Vec<Integer> = (0..=n).map(|i| self.get_coefficient(i)).collect();
        let cf = ContinuedFraction::new(coeffs);
        cf.to_rational()
    }

    /// Get convergents up to the nth term
    pub fn convergents(&self, n: usize) -> Vec<Rational> {
        (0..=n).map(|i| self.convergent(i)).collect()
    }

    /// Get the period length
    pub fn period_length(&self) -> usize {
        self.repeating.len()
    }
}

impl std::fmt::Display for PeriodicContinuedFraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;

        // Write initial part
        for (i, coeff) in self.initial.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
        }

        // Write repeating part
        write!(f, "; (")?;
        for (i, coeff) in self.repeating.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
        }
        write!(f, ")]")
    }
}

impl std::fmt::Display for ContinuedFraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if i > 0 {
                write!(f, "; ")?;
            }
            write!(f, "{}", coeff)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continued_fraction_simple() {
        // 3/2 = [1; 2]
        let r = Rational::new(3, 2).unwrap();
        let cf = ContinuedFraction::from_rational(&r);

        assert_eq!(cf.coefficients(), &[Integer::from(1), Integer::from(2)]);
        assert_eq!(cf.to_rational(), r);
    }

    #[test]
    fn test_continued_fraction_integer() {
        // 5 = [5]
        let r = Rational::new(5, 1).unwrap();
        let cf = ContinuedFraction::from_rational(&r);

        assert_eq!(cf.coefficients(), &[Integer::from(5)]);
        assert_eq!(cf.to_rational(), r);
    }

    #[test]
    fn test_continued_fraction_complex() {
        // 649/200 = [3; 4, 12, 4]
        let r = Rational::new(649, 200).unwrap();
        let cf = ContinuedFraction::from_rational(&r);

        assert_eq!(
            cf.coefficients(),
            &[
                Integer::from(3),
                Integer::from(4),
                Integer::from(12),
                Integer::from(4)
            ]
        );
        assert_eq!(cf.to_rational(), r);
    }

    #[test]
    fn test_convergents() {
        // 649/200 = [3; 4, 12, 4]
        let r = Rational::new(649, 200).unwrap();
        let cf = ContinuedFraction::from_rational(&r);

        let conv0 = cf.convergent(0);
        assert_eq!(conv0, Rational::new(3, 1).unwrap());

        let conv1 = cf.convergent(1);
        assert_eq!(conv1, Rational::new(13, 4).unwrap());

        let conv2 = cf.convergent(2);
        assert_eq!(conv2, Rational::new(159, 49).unwrap());

        let conv3 = cf.convergent(3);
        assert_eq!(conv3, Rational::new(649, 200).unwrap());
    }

    #[test]
    fn test_all_convergents() {
        let r = Rational::new(22, 7).unwrap();
        let cf = ContinuedFraction::from_rational(&r);
        let convergents = cf.all_convergents();

        // 22/7 = [3; 7]
        assert_eq!(convergents.len(), 2);
        assert_eq!(convergents[0], Rational::new(3, 1).unwrap());
        assert_eq!(convergents[1], Rational::new(22, 7).unwrap());
    }

    #[test]
    fn test_display() {
        let r = Rational::new(355, 113).unwrap();
        let cf = ContinuedFraction::from_rational(&r);
        let display = format!("{}", cf);

        // Should show the coefficients
        assert!(display.starts_with('['));
        assert!(display.ends_with(']'));
    }

    #[test]
    fn test_periodic_continued_fraction_sqrt2() {
        // √2 = [1; (2)]
        let sqrt2 = PeriodicContinuedFraction::from_sqrt(&Integer::from(2)).unwrap();

        assert_eq!(sqrt2.initial(), &[Integer::from(1)]);
        assert_eq!(sqrt2.repeating(), &[Integer::from(2)]);
        assert_eq!(sqrt2.period_length(), 1);

        // Check some coefficients
        assert_eq!(sqrt2.get_coefficient(0), Integer::from(1));
        assert_eq!(sqrt2.get_coefficient(1), Integer::from(2));
        assert_eq!(sqrt2.get_coefficient(2), Integer::from(2));
        assert_eq!(sqrt2.get_coefficient(3), Integer::from(2));
    }

    #[test]
    fn test_periodic_continued_fraction_sqrt3() {
        // √3 = [1; (1, 2)]
        let sqrt3 = PeriodicContinuedFraction::from_sqrt(&Integer::from(3)).unwrap();

        assert_eq!(sqrt3.initial(), &[Integer::from(1)]);
        assert_eq!(
            sqrt3.repeating(),
            &[Integer::from(1), Integer::from(2)]
        );
        assert_eq!(sqrt3.period_length(), 2);

        // Check coefficient pattern: 1, 1, 2, 1, 2, 1, 2, ...
        assert_eq!(sqrt3.get_coefficient(0), Integer::from(1));
        assert_eq!(sqrt3.get_coefficient(1), Integer::from(1));
        assert_eq!(sqrt3.get_coefficient(2), Integer::from(2));
        assert_eq!(sqrt3.get_coefficient(3), Integer::from(1));
        assert_eq!(sqrt3.get_coefficient(4), Integer::from(2));
    }

    #[test]
    fn test_periodic_continued_fraction_sqrt5() {
        // √5 = [2; (4)]
        let sqrt5 = PeriodicContinuedFraction::from_sqrt(&Integer::from(5)).unwrap();

        assert_eq!(sqrt5.initial(), &[Integer::from(2)]);
        assert_eq!(sqrt5.repeating(), &[Integer::from(4)]);
        assert_eq!(sqrt5.period_length(), 1);
    }

    #[test]
    fn test_periodic_continued_fraction_sqrt7() {
        // √7 = [2; (1, 1, 1, 4)]
        let sqrt7 = PeriodicContinuedFraction::from_sqrt(&Integer::from(7)).unwrap();

        assert_eq!(sqrt7.initial(), &[Integer::from(2)]);
        assert_eq!(
            sqrt7.repeating(),
            &[
                Integer::from(1),
                Integer::from(1),
                Integer::from(1),
                Integer::from(4)
            ]
        );
        assert_eq!(sqrt7.period_length(), 4);
    }

    #[test]
    fn test_periodic_continued_fraction_longer_periods() {
        // √19 = [4; (2, 1, 3, 1, 2, 8)]
        let sqrt19 = PeriodicContinuedFraction::from_sqrt(&Integer::from(19)).unwrap();
        assert_eq!(sqrt19.initial(), &[Integer::from(4)]);
        assert_eq!(
            sqrt19.repeating(),
            &[2, 1, 3, 1, 2, 8].map(Integer::from)
        );

        // √46 = [6; (1, 3, 1, 1, 2, 6, 2, 1, 1, 3, 1, 12)]
        let sqrt46 = PeriodicContinuedFraction::from_sqrt(&Integer::from(46)).unwrap();
        assert_eq!(sqrt46.initial(), &[Integer::from(6)]);
        assert_eq!(
            sqrt46.repeating(),
            &[1, 3, 1, 1, 2, 6, 2, 1, 1, 3, 1, 12].map(Integer::from)
        );

        // Invariant: for every non-square n, the period of √n ends with 2·a₀.
        for n in 2..200 {
            if let Some(cf) = PeriodicContinuedFraction::from_sqrt(&Integer::from(n)) {
                let a0 = cf.initial()[0].clone();
                assert_eq!(cf.initial().len(), 1, "√{} initial part must be [a0]", n);
                assert_eq!(
                    cf.repeating().last().unwrap(),
                    &(a0.clone() + a0),
                    "√{} period must end with 2·a0",
                    n
                );
            }
        }
    }

    #[test]
    fn test_periodic_continued_fraction_convergents() {
        // √2 = [1; (2)]
        let sqrt2 = PeriodicContinuedFraction::from_sqrt(&Integer::from(2)).unwrap();

        // First few convergents of √2:
        // p₀/q₀ = 1/1
        // p₁/q₁ = 3/2
        // p₂/q₂ = 7/5
        // p₃/q₃ = 17/12

        let conv = sqrt2.convergents(3);
        assert_eq!(conv[0], Rational::new(1, 1).unwrap());
        assert_eq!(conv[1], Rational::new(3, 2).unwrap());
        assert_eq!(conv[2], Rational::new(7, 5).unwrap());
        assert_eq!(conv[3], Rational::new(17, 12).unwrap());

        // Verify these are good approximations of √2 ≈ 1.41421356...
        // 1/1 = 1.0
        // 3/2 = 1.5
        // 7/5 = 1.4
        // 17/12 ≈ 1.41666...
    }

    #[test]
    fn test_periodic_continued_fraction_perfect_square() {
        // Perfect squares should return None
        let result = PeriodicContinuedFraction::from_sqrt(&Integer::from(4));
        assert!(result.is_none());

        let result = PeriodicContinuedFraction::from_sqrt(&Integer::from(9));
        assert!(result.is_none());

        let result = PeriodicContinuedFraction::from_sqrt(&Integer::from(16));
        assert!(result.is_none());
    }

    // ----- from_real: float -> best-rational via continued fractions -----
    //
    // Every expected value below was derived independently with Python's
    // `fractions.Fraction(mpmath.mpf(v)).limit_denominator(N)` (the canonical
    // best-rational-approximation-with-bounded-denominator oracle), NOT read
    // out of the code under test.

    fn q(num: i64, den: i64) -> Rational {
        Rational::new(num, den).unwrap()
    }

    /// A 512-bit rug::Float of pi.
    fn pi512() -> Float {
        Float::with_val(512, rug::float::Constant::Pi)
    }

    /// Render the exact rational num/den to a 512-bit rug::Float.
    fn rational_512(num: i64, den: i64) -> Float {
        let n = Float::with_val(512, num);
        let d = Float::with_val(512, den);
        Float::with_val(512, &n / &d)
    }

    #[test]
    fn test_from_real_pi_denom_200_is_355_113() {
        // Best rational approx to pi with denom <= 200 is the convergent 355/113.
        assert_eq!(from_real(&pi512(), &Integer::from(200)), q(355, 113));
    }

    #[test]
    fn test_from_real_exact_rational_roundtrip() {
        // 123456/98765 rendered to 512 bits, recovered exactly with denom <= 100000.
        let x = rational_512(123456, 98765);
        assert_eq!(from_real(&x, &Integer::from(100000)), q(123456, 98765));
    }

    #[test]
    fn test_from_real_bound_too_small_returns_best_coarse() {
        // With denom <= 100 the best approximant to pi is the SEMICONVERGENT
        // 311/99 (err ~1.8e-4), strictly better than the convergent 22/7
        // (err ~1.3e-3). This proves the semiconvergent refinement is active
        // and that a too-small bound returns the best coarse value, not 355/113.
        let best = from_real(&pi512(), &Integer::from(100));
        assert_eq!(best, q(311, 99));
        assert_ne!(best, q(22, 7));
        assert_ne!(best, q(355, 113));
    }

    #[test]
    fn test_from_real_more_oracle_values() {
        // sqrt(2), denom <= 100  -> 140/99
        let sqrt2 = Float::with_val(512, 2).sqrt();
        assert_eq!(from_real(&sqrt2, &Integer::from(100)), q(140, 99));

        // e, denom <= 50 -> 106/39 ; denom <= 100 -> 193/71
        let e = Float::with_val(512, 1).exp();
        assert_eq!(from_real(&e, &Integer::from(50)), q(106, 39));
        assert_eq!(from_real(&e, &Integer::from(100)), q(193, 71));

        // pi, large bound reaches the deep convergent 245850922/78256779
        assert_eq!(
            from_real(&pi512(), &Integer::from(99999999)),
            Rational::new(Integer::from(245850922i64), Integer::from(78256779i64)).unwrap()
        );
    }

    #[test]
    fn test_from_real_negative() {
        // -pi, denom <= 113 -> -355/113
        let neg_pi = Float::with_val(512, -1) * pi512();
        assert_eq!(from_real(&neg_pi, &Integer::from(113)), q(-355, 113));
    }

    #[test]
    fn test_from_real_integer_and_zero() {
        // Exact integers and zero come back exactly with denominator 1.
        assert_eq!(from_real(&Float::with_val(64, 5), &Integer::from(1000)), q(5, 1));
        assert_eq!(from_real(&Float::with_val(64, 0), &Integer::from(1000)), q(0, 1));
        assert_eq!(from_real(&Float::with_val(64, -7), &Integer::from(2)), q(-7, 1));
    }

    #[test]
    fn test_from_real_denom_bound_clamped() {
        // denom <= 1 must yield the nearest integer (pi -> 3/1); a zero bound
        // is clamped to 1 rather than dividing by zero.
        assert_eq!(from_real(&pi512(), &Integer::from(1)), q(3, 1));
        assert_eq!(from_real(&pi512(), &Integer::from(0)), q(3, 1));
    }

    #[test]
    fn test_from_f64_wrapper() {
        // The exact IEEE value of 0.1 best-approximates to 1/10 within denom <= 1000.
        assert_eq!(from_f64(0.1, &Integer::from(1000)), q(1, 10));
        // 1/3 rendered as f64, denom <= 2 -> 1/2.
        assert_eq!(from_f64(1.0 / 3.0, &Integer::from(2)), q(1, 2));
    }

    #[test]
    fn test_from_real_best_beats_naive_grid() {
        // Independently re-derive the answer by brute force: among ALL den
        // 1..=N, the closest p/q. This must match from_real for pi, N = 250.
        let x = pi512();
        let n = 250u64;
        let mut best: Option<Rational> = None;
        let mut best_err = Float::with_val(512, 10);
        for den in 1..=n {
            let df = Float::with_val(512, den);
            let num = Float::with_val(512, &x * &df).round();
            let approx = Float::with_val(512, &num / &df);
            let err = Float::with_val(512, &x - &approx).abs();
            if err < best_err {
                best_err = err;
                let num_i = float_floor_to_integer(&num);
                best = Some(Rational::new(num_i, Integer::from(den as i64)).unwrap());
            }
        }
        assert_eq!(from_real(&x, &Integer::from(n as i64)), best.unwrap());
    }

    #[test]
    fn test_periodic_continued_fraction_display() {
        // √2 = [1; (2)]
        let sqrt2 = PeriodicContinuedFraction::from_sqrt(&Integer::from(2)).unwrap();
        let display = format!("{}", sqrt2);
        assert_eq!(display, "[1; (2)]");

        // √3 = [1; (1, 2)]
        let sqrt3 = PeriodicContinuedFraction::from_sqrt(&Integer::from(3)).unwrap();
        let display = format!("{}", sqrt3);
        assert_eq!(display, "[1; (1, 2)]");
    }
}
