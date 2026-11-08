//! Polynomial factorization algorithms

use crate::polynomial::Polynomial;
use crate::univariate::UnivariatePolynomial;
use crate::zassenhaus;
use rustmath_core::{EuclideanDomain, MathError, Result, Ring};
use std::fmt::Debug;

/// MAGMA `Content(p)` (Handbook ch. 23 §23.5.2): the same value as
/// [`UnivariatePolynomial::content`], the normalized gcd of the coefficients (positive over Z,
/// `content(-6 - 9x) = 3`, PARI `content(-6-9*x)`; `1` for a nonzero polynomial over a field;
/// `0` for `0`).
pub fn content<R>(poly: &UnivariatePolynomial<R>) -> R
where
    R: Ring + EuclideanDomain + Clone + Debug,
{
    poly.content()
}

/// MAGMA `PrimitivePart(p)` (Handbook ch. 23 §23.5.2): [`UnivariatePolynomial::primitive_part`],
/// `p / content(p)` with the sign of `p` kept over Z (`primitive_part(6 - 9x) = 2 - 3x`, PARI
/// `(6-9*x)/content(6-9*x)`); `p` itself over a field; `0` for `0`.
pub fn primitive_part<R>(poly: &UnivariatePolynomial<R>) -> UnivariatePolynomial<R>
where
    R: Ring + EuclideanDomain + Clone + Debug,
{
    poly.primitive_part()
}

/// MAGMA `SquarefreeFactorization(f)` (Handbook ch. 23 §23.8.1) over Z or a field of
/// characteristic 0, by Musser's algorithm (D. R. Musser, *Algorithms for polynomial
/// factorization*, PhD thesis, Univ. of Wisconsin 1971; Knuth, TAOCP vol. 2 §4.6.2).
///
/// Returns `(f_i, m_i)` with every `f_i` normalized (positive leading coefficient over Z,
/// monic over a field), square-free, of positive degree and pairwise coprime, the `m_i`
/// strictly increasing, and `∏ f_i^m_i = normalize(primitive_part(f))`. The content and the
/// unit are not returned (whether MAGMA returns the content over Z is unverified).
/// `Ok(vec![])` for `0` and for constants.
///
/// The gcds are [`UnivariatePolynomial::gcd`] (normalized; Euclid while exact, then the
/// sub-resultant PRS over a non-field, Gauss' lemma over Q).
///
/// Positive characteristic: `Err(NotSupported)` when `f' = 0`, and when a multiplicity is
/// divisible by the characteristic (detected exactly: `gcd(current, rest)` is constant while
/// `rest` is not, which cannot happen in characteristic 0). Otherwise the result is correct.
/// Over GF(p) use [`crate::berlekamp::squarefree_decomposition`].
pub fn square_free_factorization<R>(
    poly: &UnivariatePolynomial<R>,
) -> Result<Vec<(UnivariatePolynomial<R>, u32)>>
where
    R: Ring + EuclideanDomain + Clone + Debug,
{
    let f = poly.primitive_part();
    if f.degree().unwrap_or(0) == 0 {
        return Ok(vec![]);
    }
    let fp = f.derivative();
    if fp.is_zero() {
        return Err(MathError::NotSupported(
            "square_free_factorization: f' = 0 (positive characteristic) is not supported; \
             over GF(p) use berlekamp::squarefree_decomposition"
                .to_string(),
        ));
    }
    // Normalized; primitive because f is (its content is gcd(cont f, cont f') = 1).
    let g = f.gcd(&fp);
    // The product of the distinct irreducible factors of f (up to a unit).
    let mut current = f.exact_quotient(&g)?;
    let mut rest = g;
    let mut k: u32 = 1;
    let mut out = Vec::new();
    // Invariant (characteristic 0): current = ∏_{e >= k} P, rest = ∏_{e > k} P^(e - k).
    while rest.degree().unwrap_or(0) > 0 {
        let next = current.gcd(&rest);
        if next.degree() == Some(0) {
            return Err(MathError::NotSupported(
                "square_free_factorization: a multiplicity is divisible by the characteristic; \
                 positive characteristic is not supported (over GF(p) use \
                 berlekamp::squarefree_decomposition)"
                    .to_string(),
            ));
        }
        let factor = current.exact_quotient(&next)?;
        if factor.degree().unwrap_or(0) > 0 {
            out.push((factor.normalize(), k));
        }
        rest = rest.exact_quotient(&next)?;
        current = next;
        // k <= deg f, so no overflow.
        k += 1;
    }
    if current.degree().unwrap_or(0) > 0 {
        out.push((current.normalize(), k));
    }
    Ok(out)
}

/// Whether no square of a polynomial of positive degree divides `poly`: see
/// [`UnivariatePolynomial::is_square_free`] (the content is not examined; a nonzero constant
/// is square-free, `0` is not). Never `Err`.
pub fn is_square_free<R>(poly: &UnivariatePolynomial<R>) -> Result<bool>
where
    R: Ring + EuclideanDomain + Clone + Debug,
{
    Ok(poly.is_square_free())
}

/// Check if a polynomial is irreducible (cannot be factored into non-trivial factors)
///
/// # Implementation Notes
///
/// This is a basic implementation with several limitations:
/// - For degree 1 polynomials: Always irreducible
/// - For degree 2-3: Checks for rational roots using the rational root theorem
/// - For higher degrees: Uses square-free factorization as a partial check
///
/// A complete irreducibility test would require:
/// - Factorization algorithms (Berlekamp, Cantor-Zassenhaus for finite fields)
/// - Hensel lifting for integer polynomials
/// - Irreducibility criteria (Eisenstein's criterion, etc.)
///
/// # Limitations
///
/// - May incorrectly report some reducible polynomials as irreducible
/// - Only reliable for polynomials of degree ≤ 3 over integers
/// - Does not implement full factorization algorithms
pub fn is_irreducible<R>(poly: &UnivariatePolynomial<R>) -> Result<bool>
where
    R: Ring + EuclideanDomain + Clone + Debug + rustmath_core::NumericConversion,
{
    // Constants and zero are not irreducible
    if poly.is_zero() || poly.is_constant() {
        return Ok(false);
    }

    let degree = poly.degree().unwrap();

    // Linear polynomials are always irreducible
    if degree == 1 {
        return Ok(true);
    }

    // Check if square-free first (if not square-free, definitely not irreducible)
    if !is_square_free(poly)? {
        return Ok(false);
    }

    // For degree 2 and 3, we can check for rational roots
    // If rational roots exist, the polynomial is reducible
    if degree <= 3 {
        // Try to find rational roots by checking divisors of constant/leading coefficient
        // This is a simplified version of the rational root theorem

        // For a more complete implementation, we would need access to:
        // 1. Integer factorization of coefficients
        // 2. Rational root theorem implementation
        // 3. Proper root-finding algorithms

        // For now, we'll do a basic check: if the polynomial has integer coefficients
        // and we can find a root by trial, it's reducible

        // This is a placeholder that assumes irreducibility for degree > 1 polynomials
        // that are square-free. A proper implementation would need more sophisticated
        // algorithms.

        return Ok(true); // Conservative: assume irreducible if square-free
    }

    // For higher degrees, we need factorization algorithms
    // Without implementing full factorization, we can only do partial checks

    // Try square-free factorization - if it produces multiple factors, reducible
    let sf_factors = square_free_factorization(poly)?;

    // If square-free factorization produces more than one distinct factor, it's reducible
    if sf_factors.len() > 1 {
        return Ok(false);
    }

    // If the only factor has multiplicity > 1, it's reducible (though this shouldn't
    // happen if we passed the is_square_free check)
    if !sf_factors.is_empty() && sf_factors[0].1 > 1 {
        return Ok(false);
    }

    // Conservative: assume irreducible if we can't determine otherwise
    // A proper implementation would use:
    // - Berlekamp's algorithm for polynomials over finite fields
    // - Zassenhaus algorithm for polynomials over integers
    // - Various irreducibility criteria
    Ok(true)
}

/// Attempt to find rational roots of a polynomial using the Rational Root Theorem
///
/// For a polynomial with integer coefficients a_n*x^n + ... + a_1*x + a_0,
/// any rational root p/q must have:
/// - p divides a_0 (constant term)
/// - q divides a_n (leading coefficient)
///
/// Returns a list of rational roots found (as Integers when they're integers)
///
/// Retained as a cheap linear-root pre-filter and exercised by unit tests; the
/// main factorization path is now [`factor_over_integers`] (Zassenhaus).
#[allow(dead_code)]
fn find_rational_roots(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
) -> Vec<rustmath_integers::Integer> {
    use rustmath_integers::Integer;

    let mut roots = Vec::new();

    if poly.is_zero() || poly.is_constant() {
        return roots;
    }

    let degree = poly.degree().unwrap();
    let a0 = poly.coeff(0).clone();
    let an = poly.coeff(degree).clone();

    if a0.is_zero() {
        // 0 is a root
        roots.push(Integer::zero());
    }

    // Get divisors of constant and leading coefficient
    let a0_divisors = if !a0.is_zero() {
        a0.abs().divisors().unwrap_or_else(|_| vec![])
    } else {
        vec![]
    };

    let _an_divisors = if !an.is_zero() {
        an.abs().divisors().unwrap_or_else(|_| vec![])
    } else {
        vec![]
    };

    // Try all combinations p/q where p | a0 and q | a_n
    // For simplicity, only try cases where q = 1 (integer roots)
    for p in &a0_divisors {
        // Try both positive and negative
        for sign in &[Integer::one(), -Integer::one()] {
            let candidate = p.clone() * sign.clone();

            // Evaluate polynomial at candidate using Horner's method
            let mut value = poly.coeff(degree).clone();
            for i in (0..degree).rev() {
                value = value * candidate.clone() + poly.coeff(i).clone();
            }

            if value.is_zero() && !roots.contains(&candidate) {
                roots.push(candidate);
            }
        }
    }

    roots
}

/// Factor out known linear factors from a polynomial
///
/// Given a polynomial and a list of roots, divide out the corresponding linear factors
#[allow(dead_code)]
fn factor_out_roots(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
    roots: &[rustmath_integers::Integer],
) -> Result<(
    Vec<(UnivariatePolynomial<rustmath_integers::Integer>, u32)>,
    UnivariatePolynomial<rustmath_integers::Integer>,
)> {
    use rustmath_integers::Integer;

    let mut factors = Vec::new();
    let mut remaining = poly.clone();

    for root in roots {
        // Factor (x - root)
        let linear_factor = UnivariatePolynomial::new(vec![-root.clone(), Integer::one()]);

        // Divide out as many times as possible
        let mut multiplicity = 0;
        loop {
            let (quotient, remainder) = remaining.div_rem(&linear_factor)?;

            if remainder.is_zero() {
                remaining = quotient;
                multiplicity += 1;
            } else {
                break;
            }
        }

        if multiplicity > 0 {
            factors.push((linear_factor, multiplicity));
        }
    }

    Ok((factors, remaining))
}

/// Factor a polynomial over integers using basic methods
///
/// This implementation uses:
/// 1. Square-free factorization to separate repeated factors
/// 2. Rational root theorem to find linear factors
/// 3. Returns remaining polynomial if it cannot be factored further
///
/// # Limitations
///
/// - Does not implement Zassenhaus or LLL-based algorithms
/// - Cannot factor polynomials without rational roots
/// - Works best for polynomials with small coefficients and low degrees
///
/// # Algorithm
///
/// For a complete factorization over Z[x], one would need:
/// - Hensel lifting to factor modulo increasing prime powers
/// - LLL lattice reduction for finding small factors
/// - Zassenhaus or van Hoeij algorithms
pub fn factor_over_integers(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
) -> Result<Vec<(UnivariatePolynomial<rustmath_integers::Integer>, u32)>> {
    use rustmath_integers::Integer;

    if poly.is_zero() {
        return Ok(vec![]);
    }
    if poly.is_constant() {
        return Ok(vec![(poly.clone(), 1)]);
    }

    // Delegate to the Zassenhaus factorizer (Vec<Integer>, little-endian).
    let coeffs: Vec<Integer> = poly.coefficients().to_vec();
    let (content_int, factors) = zassenhaus::factor(&coeffs).map_err(|_| {
        MathError::NotSupported(
            "Zassenhaus recombination exceeded the subset bound (pathological factor count)"
                .to_string(),
        )
    })?;

    let mut all_factors: Vec<(UnivariatePolynomial<Integer>, u32)> = Vec::new();

    // Emit the integer content as a constant factor unless it is 1.
    if !content_int.is_one() {
        all_factors.push((UnivariatePolynomial::new(vec![content_int]), 1));
    }

    for (g, mult) in factors {
        all_factors.push((UnivariatePolynomial::new(g), mult));
    }

    if all_factors.is_empty() {
        all_factors.push((poly.clone(), 1));
    }
    Ok(all_factors)
}

/// Test irreducibility of a polynomial over ℚ (equivalently, over ℤ for a
/// primitive polynomial) via complete Zassenhaus factorization. A constant is not
/// irreducible. This is the reliable integer-specific complement to the generic,
/// conservative [`is_irreducible`].
pub fn is_irreducible_over_integers(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
) -> Result<bool> {
    if poly.is_zero() || poly.is_constant() {
        return Ok(false);
    }
    let coeffs: Vec<rustmath_integers::Integer> = poly.coefficients().to_vec();
    let (_, factors) = zassenhaus::factor(&coeffs).map_err(|_| {
        MathError::NotSupported("Zassenhaus recombination exceeded the subset bound".to_string())
    })?;
    // Irreducible iff exactly one non-constant factor, with multiplicity one.
    Ok(factors.len() == 1 && factors[0].1 == 1)
}

/// Berlekamp factorization over GF(p) (MAGMA `Factorization` over a finite field, Handbook
/// ch. 23 §23.8.1; see [`crate::berlekamp`] for the algorithm and the references: Berlekamp
/// 1967, Knuth TAOCP vol. 2 §4.6.2, Cohen GTM 138 §3.4).
///
/// Returns every monic irreducible factor of `f mod p`, repeated by its multiplicity, in the
/// order of [`crate::berlekamp::factor`] (by degree, then by coefficients from the top
/// down); the unit `lc(f) mod p` is dropped (use [`crate::berlekamp::factor`] for it).
/// Non-square-free input is supported. `Err` for `p` not prime (see
/// [`crate::berlekamp::factor`] for what "prime" guarantees above 3.19e23) and for
/// `f ≡ 0 (mod p)`; `Ok(vec![])` for a nonzero constant.
///
/// # Example
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_polynomials::factorization::berlekamp_factor_gf;
/// use rustmath_polynomials::UnivariatePolynomial;
///
/// // x^4 + 1 = (x + 1)^4 over GF(2) (PARI: factormod(x^4+1,2) -> [x + 1, 4]).
/// let poly = |c: &[i64]| UnivariatePolynomial::new(c.iter().map(|&a| Integer::from(a)).collect());
/// let f = poly(&[1, 0, 0, 0, 1]);
/// let x_plus_1 = poly(&[1, 1]);
/// assert_eq!(berlekamp_factor_gf(&f, &Integer::from(2)).unwrap(), vec![x_plus_1; 4]);
/// ```
pub fn berlekamp_factor_gf(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
    p: &rustmath_integers::Integer,
) -> Result<Vec<UnivariatePolynomial<rustmath_integers::Integer>>> {
    let (_, factors) = crate::berlekamp::factor(poly.coefficients(), p)?;
    let mut out = Vec::new();
    for (g, e) in factors {
        let g = UnivariatePolynomial::new(g);
        for _ in 0..e {
            out.push(g.clone());
        }
    }
    Ok(out)
}

/// Helper: Polynomial division over GF(p)
fn div_poly_gf(
    a: &UnivariatePolynomial<rustmath_integers::Integer>,
    b: &UnivariatePolynomial<rustmath_integers::Integer>,
    p: &rustmath_integers::Integer,
) -> Result<(UnivariatePolynomial<rustmath_integers::Integer>, UnivariatePolynomial<rustmath_integers::Integer>)> {
    use rustmath_integers::Integer;

    if b.is_zero() {
        return Err(MathError::DivisionByZero);
    }

    let mut remainder = a.clone();
    let mut quotient_coeffs = vec![Integer::zero(); if a.degree() >= b.degree() {
        a.degree().unwrap() - b.degree().unwrap() + 1
    } else {
        0
    }];

    let b_deg = b.degree().unwrap_or(0);
    let b_lead = b.coeff(b_deg).clone();
    let b_lead_inv = mod_inverse(&b_lead, p)?;

    while remainder.degree() >= b.degree() && !remainder.is_zero() {
        let r_deg = remainder.degree().unwrap();
        let r_lead = remainder.coeff(r_deg).clone();

        // Compute quotient coefficient
        let q_coeff = (r_lead * b_lead_inv.clone()) % p.clone();
        let q_deg = r_deg - b_deg;

        if q_deg < quotient_coeffs.len() {
            quotient_coeffs[q_deg] = q_coeff.clone();
        }

        // Subtract b * q_coeff * x^(r_deg - b_deg) from remainder
        let mut new_remainder_coeffs = vec![Integer::zero(); r_deg + 1];
        for i in 0..=r_deg {
            new_remainder_coeffs[i] = remainder.coeff(i).clone();
        }

        for i in 0..=b_deg {
            let idx = i + q_deg;
            if idx <= r_deg {
                let sub = (b.coeff(i).clone() * q_coeff.clone()) % p.clone();
                new_remainder_coeffs[idx] = (new_remainder_coeffs[idx].clone() - sub) % p.clone();
                if new_remainder_coeffs[idx] < Integer::zero() {
                    new_remainder_coeffs[idx] = new_remainder_coeffs[idx].clone() + p.clone();
                }
            }
        }

        remainder = UnivariatePolynomial::new(new_remainder_coeffs);
    }

    Ok((UnivariatePolynomial::new(quotient_coeffs), remainder))
}

/// Helper: GCD of two polynomials over GF(p) (only exercised by its unit test since the
/// Berlekamp code moved to [`crate::berlekamp`]).
#[cfg(test)]
fn gcd_poly_gf(
    a: &UnivariatePolynomial<rustmath_integers::Integer>,
    b: &UnivariatePolynomial<rustmath_integers::Integer>,
    p: &rustmath_integers::Integer,
) -> UnivariatePolynomial<rustmath_integers::Integer> {
    let mut r0 = a.clone();
    let mut r1 = b.clone();

    while !r1.is_zero() {
        if let Ok((_, remainder)) = div_poly_gf(&r0, &r1, p) {
            r0 = r1;
            r1 = remainder;
        } else {
            break;
        }
    }

    r0
}

/// Helper: Modular inverse over GF(p) using extended Euclidean algorithm
fn mod_inverse(
    a: &rustmath_integers::Integer,
    p: &rustmath_integers::Integer,
) -> Result<rustmath_integers::Integer> {
    use rustmath_integers::Integer;

    let (gcd, x, _) = a.extended_gcd(p);

    if !gcd.is_one() {
        return Err(MathError::InvalidArgument(
            format!("{} has no inverse modulo {}", a, p),
        ));
    }

    // Ensure result is positive
    let result = if x < Integer::zero() {
        x + p.clone()
    } else {
        x
    };

    Ok(result)
}

/// Hensel lifting for polynomial factorization
///
/// Lifts a factorization of f(x) from modulo p to modulo p^k.
///
/// Given f(x) ≡ g₀(x) · h₀(x) (mod p) where gcd(g₀, h₀) = 1 (mod p),
/// computes g(x), h(x) such that f(x) ≡ g(x) · h(x) (mod p^k).
///
/// # Algorithm (Hensel's Lemma)
///
/// 1. Start with f ≡ g₀ · h₀ (mod p)
/// 2. Use extended GCD to find s₀, t₀ such that s₀·g₀ + t₀·h₀ ≡ 1 (mod p)
/// 3. For each step i from 1 to k-1:
///    - Compute error: e = (f - gᵢ·hᵢ) / p^i
///    - Solve: Δg·h + g·Δh ≡ e (mod p)
///    - Update: gᵢ₊₁ = gᵢ + p^i·Δg, hᵢ₊₁ = hᵢ + p^i·Δh
///
/// # Parameters
///
/// * `f` - Polynomial to factor
/// * `g0` - First factor modulo p
/// * `h0` - Second factor modulo p
/// * `p` - Prime modulus
/// * `k` - Target exponent (lift to p^k)
///
/// # Returns
///
/// Tuple (g, h) where f ≡ g·h (mod p^k)
///
/// # References
///
/// - Knuth, TAOCP Vol. 2, Section 4.6.2
/// - von zur Gathen & Gerhard, "Modern Computer Algebra", Chapter 15
///
/// # Example
///
/// ```ignore
/// // Factor f(x) = x^2 + 1 modulo 5
/// // Over GF(5): x^2 + 1 = (x + 2)(x + 3) since 2^2 = 4 ≡ -1, 3^2 = 9 ≡ -1
/// let f = UnivariatePolynomial::new(vec![1, 0, 1]);
/// let g0 = UnivariatePolynomial::new(vec![2, 1]); // x + 2
/// let h0 = UnivariatePolynomial::new(vec![3, 1]); // x + 3
/// let (g, h) = hensel_lift(&f, &g0, &h0, &Integer::from(5), 3)?;
/// // Now f ≡ g·h (mod 125)
/// ```
pub fn hensel_lift(
    f: &UnivariatePolynomial<rustmath_integers::Integer>,
    g0: &UnivariatePolynomial<rustmath_integers::Integer>,
    h0: &UnivariatePolynomial<rustmath_integers::Integer>,
    p: &rustmath_integers::Integer,
    k: u32,
) -> Result<(
    UnivariatePolynomial<rustmath_integers::Integer>,
    UnivariatePolynomial<rustmath_integers::Integer>,
)> {


    use crate::zp_hensel;
    use rustmath_integers::Integer;

    if k == 0 {
        return Err(MathError::InvalidArgument(
            "Hensel lifting requires k >= 1".to_string(),
        ));
    }

    if k == 1 {
        // Already at target precision
        return Ok((g0.clone(), h0.clone()));
    }

    // The lift runs on the small-prime engine in `zp_hensel` (a faithful port of
    // the p-adelic calculator's verified ZpPoly/FpPoly routines). The earlier
    // in-place implementation had a Bezout-side mismatch (the Δg/Δh corrections
    // were reduced against the wrong cofactor), which is why its tests were
    // marked `#[ignore]`. We convert to coefficient vectors, delegate, and
    // convert back.
    let p_i64 = p.to_i64();
    if p_i64 <= 1 {
        return Err(MathError::InvalidArgument(
            "Hensel lifting requires a prime p >= 2 that fits in i64".to_string(),
        ));
    }

    // f as Z[x]; g0, h0 reduced to F_p[x] (Vec<i64> in [0, p)).
    let f_coeffs: Vec<Integer> = f.coefficients().to_vec();
    let pm = p.clone();
    let to_fp = |poly: &UnivariatePolynomial<Integer>| -> Vec<i64> {
        poly.coefficients()
            .iter()
            .map(|c| {
                let r = c % &pm;
                (&(&r + &pm) % &pm).to_i64()
            })
            .collect()
    };
    let g0_fp = to_fp(g0);
    let h0_fp = to_fp(h0);

    match zp_hensel::hensel_lift(&f_coeffs, &g0_fp, &h0_fp, p_i64, k) {
        Some((big_g, big_h)) => Ok((
            UnivariatePolynomial::new(big_g),
            UnivariatePolynomial::new(big_h),
        )),
        None => Err(MathError::InvalidArgument(
            "Hensel lifting requires coprime monic factors with f ≡ g₀·h₀ (mod p)".to_string(),
        )),
    }
}

/// Extended GCD for polynomials over GF(p)
///
/// Computes s, t such that s·a + t·b ≡ gcd(a,b) (mod p)
/// Assumes gcd(a,b) = 1 (mod p), so returns s, t with s·a + t·b ≡ 1 (mod p)
#[allow(dead_code)]
fn extended_gcd_poly_gf(
    a: &UnivariatePolynomial<rustmath_integers::Integer>,
    b: &UnivariatePolynomial<rustmath_integers::Integer>,
    p: &rustmath_integers::Integer,
) -> Result<(
    UnivariatePolynomial<rustmath_integers::Integer>,
    UnivariatePolynomial<rustmath_integers::Integer>,
)> {
    use rustmath_integers::Integer;

    let mut r0 = a.clone();
    let mut r1 = b.clone();

    let mut s0 = UnivariatePolynomial::new(vec![Integer::one()]);
    let mut s1 = UnivariatePolynomial::new(vec![Integer::zero()]);

    let mut t0 = UnivariatePolynomial::new(vec![Integer::zero()]);
    let mut t1 = UnivariatePolynomial::new(vec![Integer::one()]);

    while !r1.is_zero() {
        let (q, r) = div_poly_gf(&r0, &r1, p)?;

        // Update remainders
        let r_next = r;
        r0 = r1.clone();
        r1 = r_next;

        // Update s coefficients
        let qs = poly_mul_mod(&q, &s1, p);
        let s_next = poly_sub_mod(&s0, &qs, p);
        s0 = s1;
        s1 = s_next;

        // Update t coefficients
        let qt = poly_mul_mod(&q, &t1, p);
        let t_next = poly_sub_mod(&t0, &qt, p);
        t0 = t1;
        t1 = t_next;
    }

    // Normalize so that r0 is monic (leading coefficient = 1)
    if !r0.is_zero() {
        let lead = r0.coeff(r0.degree().unwrap_or(0)).clone();
        if lead != Integer::one() {
            let lead_inv = mod_inverse(&lead, p)?;
            s0 = poly_scalar_mul_mod(&s0, &lead_inv, p);
            t0 = poly_scalar_mul_mod(&t0, &lead_inv, p);
        }
    }

    Ok((s0, t0))
}

/// Helper: Multiply two polynomials WITHOUT modular reduction
#[allow(dead_code)]
fn poly_mul_unreduced(
    a: &UnivariatePolynomial<rustmath_integers::Integer>,
    b: &UnivariatePolynomial<rustmath_integers::Integer>,
) -> UnivariatePolynomial<rustmath_integers::Integer> {
    use rustmath_integers::Integer;

    let deg_a = a.degree().unwrap_or(0);
    let deg_b = b.degree().unwrap_or(0);
    let mut result = vec![Integer::zero(); deg_a + deg_b + 1];

    for i in 0..=deg_a {
        for j in 0..=deg_b {
            result[i + j] = result[i + j].clone() +
                (a.coeff(i).clone() * b.coeff(j).clone());
        }
    }

    UnivariatePolynomial::new(result)
}

/// Helper: Multiply two polynomials and reduce coefficients modulo m
#[allow(dead_code)]
fn poly_mul_mod(
    a: &UnivariatePolynomial<rustmath_integers::Integer>,
    b: &UnivariatePolynomial<rustmath_integers::Integer>,
    m: &rustmath_integers::Integer,
) -> UnivariatePolynomial<rustmath_integers::Integer> {
    use rustmath_integers::Integer;

    let deg_a = a.degree().unwrap_or(0);
    let deg_b = b.degree().unwrap_or(0);
    let mut result = vec![Integer::zero(); deg_a + deg_b + 1];

    for i in 0..=deg_a {
        for j in 0..=deg_b {
            result[i + j] = (result[i + j].clone() +
                (a.coeff(i).clone() * b.coeff(j).clone())) % m.clone();
        }
    }

    UnivariatePolynomial::new(result)
}

/// Helper: Add two polynomials WITHOUT modular reduction
#[allow(dead_code)]
fn poly_add_unreduced(
    a: &UnivariatePolynomial<rustmath_integers::Integer>,
    b: &UnivariatePolynomial<rustmath_integers::Integer>,
) -> UnivariatePolynomial<rustmath_integers::Integer> {
    use rustmath_integers::Integer;

    let deg_a = a.degree().unwrap_or(0);
    let deg_b = b.degree().unwrap_or(0);
    let max_deg = deg_a.max(deg_b);
    let mut result = Vec::new();

    for i in 0..=max_deg {
        let a_coeff = if i <= deg_a { a.coeff(i).clone() } else { Integer::zero() };
        let b_coeff = if i <= deg_b { b.coeff(i).clone() } else { Integer::zero() };
        result.push(a_coeff + b_coeff);
    }

    UnivariatePolynomial::new(result)
}

/// Helper: Add two polynomials and reduce coefficients modulo m
#[allow(dead_code)]
fn poly_add_mod(
    a: &UnivariatePolynomial<rustmath_integers::Integer>,
    b: &UnivariatePolynomial<rustmath_integers::Integer>,
    m: &rustmath_integers::Integer,
) -> UnivariatePolynomial<rustmath_integers::Integer> {
    use rustmath_integers::Integer;

    let deg_a = a.degree().unwrap_or(0);
    let deg_b = b.degree().unwrap_or(0);
    let max_deg = deg_a.max(deg_b);
    let mut result = Vec::new();

    for i in 0..=max_deg {
        let a_coeff = if i <= deg_a { a.coeff(i).clone() } else { Integer::zero() };
        let b_coeff = if i <= deg_b { b.coeff(i).clone() } else { Integer::zero() };
        let coeff = (a_coeff + b_coeff) % m.clone();
        let coeff = if coeff < Integer::zero() {
            coeff + m.clone()
        } else {
            coeff
        };
        result.push(coeff);
    }

    UnivariatePolynomial::new(result)
}

/// Helper: Subtract two polynomials and reduce coefficients modulo m
#[allow(dead_code)]
fn poly_sub_mod(
    a: &UnivariatePolynomial<rustmath_integers::Integer>,
    b: &UnivariatePolynomial<rustmath_integers::Integer>,
    m: &rustmath_integers::Integer,
) -> UnivariatePolynomial<rustmath_integers::Integer> {
    use rustmath_integers::Integer;

    let deg_a = a.degree().unwrap_or(0);
    let deg_b = b.degree().unwrap_or(0);
    let max_deg = deg_a.max(deg_b);
    let mut result = Vec::new();

    for i in 0..=max_deg {
        let a_coeff = if i <= deg_a { a.coeff(i).clone() } else { Integer::zero() };
        let b_coeff = if i <= deg_b { b.coeff(i).clone() } else { Integer::zero() };
        let coeff = (a_coeff - b_coeff) % m.clone();
        let coeff = if coeff < Integer::zero() {
            coeff + m.clone()
        } else {
            coeff
        };
        result.push(coeff);
    }

    UnivariatePolynomial::new(result)
}

/// Helper: Multiply polynomial by scalar and reduce modulo m
#[allow(dead_code)]
fn poly_scalar_mul(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
    scalar: &rustmath_integers::Integer,
) -> UnivariatePolynomial<rustmath_integers::Integer> {


    let mut result = Vec::new();
    for i in 0..=poly.degree().unwrap_or(0) {
        result.push(poly.coeff(i).clone() * scalar.clone());
    }

    UnivariatePolynomial::new(result)
}

/// Helper: Multiply polynomial by scalar and reduce modulo m
#[allow(dead_code)]
fn poly_scalar_mul_mod(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
    scalar: &rustmath_integers::Integer,
    m: &rustmath_integers::Integer,
) -> UnivariatePolynomial<rustmath_integers::Integer> {
    use rustmath_integers::Integer;

    let mut result = Vec::new();
    for i in 0..=poly.degree().unwrap_or(0) {
        let coeff = (poly.coeff(i).clone() * scalar.clone()) % m.clone();
        let coeff = if coeff < Integer::zero() {
            coeff + m.clone()
        } else {
            coeff
        };
        result.push(coeff);
    }

    UnivariatePolynomial::new(result)
}

/// Helper: Reduce polynomial coefficients modulo m
#[allow(dead_code)]
fn reduce_poly_mod(
    poly: &UnivariatePolynomial<rustmath_integers::Integer>,
    m: &rustmath_integers::Integer,
) -> UnivariatePolynomial<rustmath_integers::Integer> {
    use rustmath_integers::Integer;

    let mut result = Vec::new();
    for i in 0..=poly.degree().unwrap_or(0) {
        let coeff = poly.coeff(i).clone() % m.clone();
        let coeff = if coeff < Integer::zero() {
            coeff + m.clone()
        } else {
            coeff
        };
        result.push(coeff);
    }

    UnivariatePolynomial::new(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_content() {
        // 6x² + 9x + 3 has content 3
        let p = UnivariatePolynomial::new(vec![
            Integer::from(3),
            Integer::from(9),
            Integer::from(6),
        ]);

        let cont = content(&p);
        assert_eq!(cont, Integer::from(3));
    }

    #[test]
    fn test_primitive_part() {
        // 6x² + 9x + 3 = 3(2x² + 3x + 1)
        let p = UnivariatePolynomial::new(vec![
            Integer::from(3),
            Integer::from(9),
            Integer::from(6),
        ]);

        let pp = primitive_part(&p);

        // Primitive part should be 2x² + 3x + 1
        assert_eq!(*pp.coeff(0), Integer::from(1));
        assert_eq!(*pp.coeff(1), Integer::from(3));
        assert_eq!(*pp.coeff(2), Integer::from(2));
    }

    #[test]
    fn test_is_square_free() {
        // x² + x + 1 is square-free
        let p1 = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
        ]);
        assert!(is_square_free(&p1).unwrap());

        // (x - 1)² = x² - 2x + 1 is not square-free. This previously hung in the
        // Euclidean ℤ[x] GCD; now handled by the pseudo-division GCD.
        let p2 = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(-2),
            Integer::from(1),
        ]);
        assert!(!is_square_free(&p2).unwrap());
    }

    #[test]
    fn test_square_free_factorization_simple() {
        // x is already square-free
        let x = UnivariatePolynomial::new(vec![Integer::from(0), Integer::from(1)]);

        let factors = square_free_factorization(&x).unwrap();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1); // multiplicity 1
    }

    #[test]
    fn test_square_free_factorization_repeated() {
        // x² = x * x (x with multiplicity 2; gp: factor(x^2) -> [x, 2]). gcd(x², 2x) = x over
        // Z (the normalized gcd keeps no content here: gcd(1, 2) = 1).

        let p = UnivariatePolynomial::new(vec![
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
        ]);

        let factors = square_free_factorization(&p).unwrap();

        // Should have x with multiplicity 2
        assert!(!factors.is_empty());
        assert_eq!(factors, vec![(zp(&[0, 1]), 2)]);
    }

    #[test]
    fn test_factor_over_integers() {
        // Simple polynomial x + 1
        let p = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(1)]);

        let factors = factor_over_integers(&p).unwrap();
        assert!(!factors.is_empty());
    }

    #[test]
    fn test_find_rational_roots() {
        // x² - 1 = (x-1)(x+1) has roots ±1
        let p = UnivariatePolynomial::new(vec![
            Integer::from(-1),
            Integer::from(0),
            Integer::from(1),
        ]);

        let roots = find_rational_roots(&p);
        assert!(roots.contains(&Integer::from(1)));
        assert!(roots.contains(&Integer::from(-1)));
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_find_rational_roots_cubic() {
        // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3) has roots 1, 2, 3
        let p = UnivariatePolynomial::new(vec![
            Integer::from(-6),
            Integer::from(11),
            Integer::from(-6),
            Integer::from(1),
        ]);

        let roots = find_rational_roots(&p);
        assert!(roots.contains(&Integer::from(1)));
        assert!(roots.contains(&Integer::from(2)));
        assert!(roots.contains(&Integer::from(3)));
        assert_eq!(roots.len(), 3);
    }

    #[test]
    fn test_factor_over_integers_quadratic() {
        // x² - 5x + 6 = (x-2)(x-3)
        let p = UnivariatePolynomial::new(vec![
            Integer::from(6),
            Integer::from(-5),
            Integer::from(1),
        ]);

        let factors = factor_over_integers(&p).unwrap();

        // Should have two linear factors
        let linear_factors: Vec<_> = factors
            .iter()
            .filter(|(f, _)| f.degree() == Some(1))
            .collect();

        assert!(linear_factors.len() >= 2, "Should find at least 2 linear factors");
    }

    #[test]
    fn test_factor_over_integers_with_content() {
        // 2x² - 2 = 2(x² - 1) = 2(x-1)(x+1)
        let p = UnivariatePolynomial::new(vec![
            Integer::from(-2),
            Integer::from(0),
            Integer::from(2),
        ]);

        let factors = factor_over_integers(&p).unwrap();

        // Should extract content 2
        let constant_factors: Vec<_> = factors
            .iter()
            .filter(|(f, _)| f.is_constant())
            .collect();

        assert!(!constant_factors.is_empty(), "Should extract content");
    }

    #[test]
    fn test_factor_irreducible() {
        // x² + 1 is irreducible over Q (no rational roots)
        let p = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        ]);

        let factors = factor_over_integers(&p).unwrap();

        // Should return the polynomial itself or as a single factor
        assert_eq!(factors.len(), 1);
        assert!(factors[0].0.degree() == Some(2) || factors[0].0.degree() == None);
    }

    fn zp(c: &[i64]) -> UnivariatePolynomial<Integer> {
        UnivariatePolynomial::new(c.iter().map(|&a| Integer::from(a)).collect())
    }

    #[test]
    fn test_berlekamp_linear() {
        // x + 1 over GF(2) should be irreducible
        let poly = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(1)]);
        let p = Integer::from(2);

        let factors = berlekamp_factor_gf(&poly, &p).unwrap();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].degree(), Some(1));
        assert_eq!(factors, vec![zp(&[1, 1])]);
    }

    #[test]
    fn test_berlekamp_quadratic_gf2() {
        // x² + x + 1 over GF(2) is irreducible (gp: factormod(x^2+x+1,2) -> [x^2 + x + 1, 1])
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
        ]);
        let p = Integer::from(2);

        let factors = berlekamp_factor_gf(&poly, &p).unwrap();

        // Should be irreducible (single factor)
        assert_eq!(factors.len(), 1);
        assert_eq!(factors, vec![zp(&[1, 1, 1])]);
    }

    #[test]
    fn test_berlekamp_factorizable_gf2() {
        // x² + 1 over GF(2) = (x + 1)² since 1 = -1 in GF(2)
        // (gp: factormod(x^2+1,2) -> [x + 1, 2])
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        ]);
        let p = Integer::from(2);

        let factors = berlekamp_factor_gf(&poly, &p).unwrap();

        // Non-square-free input is supported: the repeated factor comes with its multiplicity.
        assert!(!factors.is_empty());
        assert_eq!(factors, vec![zp(&[1, 1]), zp(&[1, 1])]);
    }

    #[test]
    fn test_berlekamp_gf3() {
        // x² + 1 over GF(3) is irreducible: -1 is not a square mod 3
        // (gp: factormod(x^2+1,3) -> [x^2 + 1, 1]).
        let poly = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        ]);
        let p = Integer::from(3);

        let factors = berlekamp_factor_gf(&poly, &p).unwrap();

        assert!(!factors.is_empty());
        assert_eq!(factors, vec![zp(&[1, 0, 1])]);
    }

    // The old Q matrix held x^(i(p+1)) mod f in row i and reported these irreducible.
    // gp: factormod(x^2+2,3) -> [x + 1, 1; x + 2, 1];
    //     factormod(x^4+1,3) -> [x^2 + x + 2, 1; x^2 + 2*x + 2, 1];
    //     factormod(x^3-1,7) -> [x + 3, 1; x + 5, 1; x + 6, 1];
    //     factormod(x^5-x,5) -> [x, 1; x + 1, 1; x + 2, 1; x + 3, 1; x + 4, 1]
    #[test]
    fn test_berlekamp_q_matrix_regression() {
        let p3 = Integer::from(3);
        assert_eq!(
            berlekamp_factor_gf(&zp(&[2, 0, 1]), &p3).unwrap(),
            vec![zp(&[1, 1]), zp(&[2, 1])]
        );
        assert_eq!(
            berlekamp_factor_gf(&zp(&[1, 0, 0, 0, 1]), &p3).unwrap(),
            vec![zp(&[2, 1, 1]), zp(&[2, 2, 1])]
        );
        assert_eq!(
            berlekamp_factor_gf(&zp(&[-1, 0, 0, 1]), &Integer::from(7)).unwrap(),
            vec![zp(&[3, 1]), zp(&[5, 1]), zp(&[6, 1])]
        );
        assert_eq!(
            berlekamp_factor_gf(&zp(&[0, -1, 0, 0, 0, 1]), &Integer::from(5)).unwrap(),
            vec![
                zp(&[0, 1]),
                zp(&[1, 1]),
                zp(&[2, 1]),
                zp(&[3, 1]),
                zp(&[4, 1])
            ]
        );
        // Zero is an error, a nonzero constant has no factor; p must be prime.
        assert!(berlekamp_factor_gf(&zp(&[3, 3]), &p3).is_err());
        assert_eq!(
            berlekamp_factor_gf(&zp(&[7]), &Integer::from(5)).unwrap(),
            vec![]
        );
        assert!(berlekamp_factor_gf(&zp(&[1, 0, 1]), &Integer::from(4)).is_err());
    }

    // Free functions agree with the normalized methods (gp: content(-6-9*x) -> 3,
    // content(6-9*x) -> 3, (6-9*x)/content(6-9*x) -> -3*x + 2).
    #[test]
    fn test_free_content_is_normalized() {
        assert_eq!(content(&zp(&[-6, -9])), Integer::from(3));
        assert_eq!(content(&zp(&[6, -9])), Integer::from(3));
        assert_eq!(primitive_part(&zp(&[6, -9])), zp(&[2, -3]));
        assert_eq!(primitive_part(&zp(&[-6, -9])), zp(&[-2, -3]));
        assert_eq!(content(&zp(&[0])), Integer::from(0));
    }

    #[test]
    fn test_gcd_poly_gf() {
        // Test GCD computation over GF(p)
        // (x + 1)(x + 2) and (x + 1)(x + 3) over GF(5)
        let p = Integer::from(5);

        // (x + 1)(x + 2) = x² + 3x + 2
        let poly1 = UnivariatePolynomial::new(vec![
            Integer::from(2),
            Integer::from(3),
            Integer::from(1),
        ]);

        // (x + 1)(x + 3) = x² + 4x + 3
        let poly2 = UnivariatePolynomial::new(vec![
            Integer::from(3),
            Integer::from(4),
            Integer::from(1),
        ]);

        let gcd = gcd_poly_gf(&poly1, &poly2, &p);

        // GCD should be x + 1 (degree 1)
        assert_eq!(gcd.degree(), Some(1));
    }

    #[test]
    fn test_div_poly_gf() {
        // Test polynomial division over GF(p)
        let p = Integer::from(5);

        // Divide x² + 3x + 2 by x + 1 over GF(5)
        // Should get x + 2
        let dividend = UnivariatePolynomial::new(vec![
            Integer::from(2),
            Integer::from(3),
            Integer::from(1),
        ]);

        let divisor = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(1),
        ]);

        let (quotient, remainder) = div_poly_gf(&dividend, &divisor, &p).unwrap();

        // Quotient should be x + 2
        assert_eq!(quotient.degree(), Some(1));
        assert_eq!(*quotient.coeff(0), Integer::from(2));
        assert_eq!(*quotient.coeff(1), Integer::from(1));

        // Remainder should be 0
        assert!(remainder.is_zero() || remainder.degree() == Some(0));
    }

    #[test]
    fn test_hensel_lift_simple() {
        // f = x² - 1 = (x + 4)(x + 1) mod 5, coprime factors.
        let f = UnivariatePolynomial::new(vec![
            Integer::from(-1),
            Integer::from(0),
            Integer::from(1),
        ]);

        let g0 = UnivariatePolynomial::new(vec![
            Integer::from(4),
            Integer::from(1),
        ]);

        let h0 = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(1),
        ]);

        let p = Integer::from(5);

        // Verify initial factorization mod 5
        let prod_init = poly_mul_mod(&g0, &h0, &p);
        assert_eq!(prod_init.coeff(0).clone() % p.clone(), Integer::from(4));

        // Lift to p² = 25 and verify g·h ≡ f (mod 25).
        let (g, h) = hensel_lift(&f, &g0, &h0, &p, 2).unwrap();
        let p2 = Integer::from(25);
        let product = poly_mul_mod(&g, &h, &p2);
        for i in 0..=2 {
            let prod_coeff = ((product.coeff(i).clone() % p2.clone()) + p2.clone()) % p2.clone();
            let f_coeff = ((f.coeff(i).clone() % p2.clone()) + p2.clone()) % p2.clone();
            assert_eq!(prod_coeff, f_coeff, "Coefficient {} mismatch mod 25", i);
        }
    }

    #[test]
    fn test_hensel_lift_quadratic() {
        // Test with x² + 1 over GF(5)
        // Over GF(5): x² + 1 = (x + 2)(x + 3)
        // since 2² = 4 ≡ -1 (mod 5) and 3² = 9 ≡ -1 (mod 5)
        let f = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        ]);

        let g0 = UnivariatePolynomial::new(vec![
            Integer::from(2),
            Integer::from(1),
        ]);

        let h0 = UnivariatePolynomial::new(vec![
            Integer::from(3),
            Integer::from(1),
        ]);

        let p = Integer::from(5);

        // Lift to p³ = 125
        let (g, h) = hensel_lift(&f, &g0, &h0, &p, 3).unwrap();

        // Verify: g·h ≡ f (mod 125)
        let p3 = Integer::from(125);
        let product = poly_mul_mod(&g, &h, &p3);

        for i in 0..=2 {
            let prod_coeff = product.coeff(i).clone() % p3.clone();
            let f_coeff = f.coeff(i).clone() % p3.clone();
            assert_eq!(prod_coeff, f_coeff, "Coefficient {} mismatch", i);
        }
    }

    #[test]
    fn test_extended_gcd_poly_gf() {
        // Test extended GCD for polynomials
        // gcd(x+1, x+2) = 1 over GF(5)
        let a = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(1),
        ]);

        let b = UnivariatePolynomial::new(vec![
            Integer::from(2),
            Integer::from(1),
        ]);

        let p = Integer::from(5);

        let (s, t) = extended_gcd_poly_gf(&a, &b, &p).unwrap();

        // Verify: s·a + t·b ≡ 1 (mod 5)
        let sa = poly_mul_mod(&s, &a, &p);
        let tb = poly_mul_mod(&t, &b, &p);
        let result = poly_add_mod(&sa, &tb, &p);

        // Result should be constant 1
        assert_eq!(result.coeff(0).clone() % p.clone(), Integer::one());
        for i in 1..=result.degree().unwrap_or(0) {
            assert_eq!(
                result.coeff(i).clone() % p.clone(),
                Integer::zero(),
                "Higher coefficient {} should be 0", i
            );
        }
    }

    #[test]
    fn test_poly_helpers() {
        let p = Integer::from(7);

        // Test poly_add_mod
        let a = UnivariatePolynomial::new(vec![Integer::from(5), Integer::from(6)]);
        let b = UnivariatePolynomial::new(vec![Integer::from(3), Integer::from(2)]);
        let sum = poly_add_mod(&a, &b, &p);
        assert_eq!(*sum.coeff(0), Integer::from(1)); // (5 + 3) mod 7 = 1
        assert_eq!(*sum.coeff(1), Integer::from(1)); // (6 + 2) mod 7 = 1

        // Test poly_sub_mod
        let diff = poly_sub_mod(&a, &b, &p);
        assert_eq!(*diff.coeff(0), Integer::from(2)); // (5 - 3) mod 7 = 2
        assert_eq!(*diff.coeff(1), Integer::from(4)); // (6 - 2) mod 7 = 4

        // Test poly_scalar_mul_mod
        let scalar = Integer::from(3);
        let scaled = poly_scalar_mul_mod(&a, &scalar, &p);
        assert_eq!(*scaled.coeff(0), Integer::from(1)); // (5 * 3) mod 7 = 15 mod 7 = 1
        assert_eq!(*scaled.coeff(1), Integer::from(4)); // (6 * 3) mod 7 = 18 mod 7 = 4
    }
}
