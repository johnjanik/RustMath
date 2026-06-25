//! Hilbert class field of an imaginary quadratic field, by CM
//! (`docs/algorithm_notes/abext_notes.md` §3B).
//!
//! For `K = ℚ(√d)` with `d < 0` a fundamental discriminant, the Hilbert class
//! field `H_K` is generated over `K` by a **singular modulus** `j(τ)`. The
//! `h = h(d)` reduced primitive positive-definite binary quadratic forms
//! `[a, b, c]` of discriminant `d` give the CM points `τ = (−b + √d)/(2a)`, and
//!
//! ```text
//!     H_d(X) = ∏_{[a,b,c]} (X − j(τ_{[a,b,c]}))  ∈  ℤ[X]
//! ```
//!
//! is the **Hilbert class polynomial**, irreducible over `ℚ` of degree `h`.
//! Then `ℚ(j)` is a degree-`h` field, `H_K = K(j) = ℚ(√d, j)` is its
//! compositum with `K` (degree `2h` over `ℚ`), and `H_d` is irreducible over
//! `K` (linear disjointness: `[ℚ(j):ℚ] = [K(j):K] = h`).
//!
//! This module assembles the **absolute** Hilbert class field by reusing the
//! multiplication-matrix `rnfequation` of [`crate::rnfeq`]: with `g = T² − d`
//! the minimal polynomial of `√d` and `H_d` viewed as a monic relative
//! polynomial over `K` (its integer coefficients lie in `K`), Algorithm 1A
//! returns the degree-`2h` absolute defining polynomial of `H_K`.
//!
//! Step 1 (this file) takes `H_d` as input and is fully exact (no floats); the
//! companion routine that *computes* `H_d` from high-precision `j(τ)` is the
//! `rug`/MPC piece added on top.

use crate::rnfeq::{absolute_defining_polynomial, AbsoluteField};
use rustmath_complex::mpc::ComplexMPFR;
use rustmath_integers::Integer;
use rustmath_polynomials::disc::discriminant as poly_discriminant_fn;
use rustmath_polynomials::univariate::UnivariatePolynomial;
use rustmath_rationals::Rational;
use rustmath_reals::mpfr::RealMPFR;

/// Reduced primitive positive-definite binary quadratic forms `[a, b, c]` of
/// discriminant `d = b² − 4ac < 0`.
///
/// Reduction conditions: `|b| ≤ a ≤ c`, with `b ≥ 0` whenever `|b| = a` or
/// `a = c`. Primitivity: `gcd(a, b, c) = 1`. The count is the class number
/// `h(d)` (for fundamental `d`, the class number of `ℚ(√d)`).
pub fn reduced_forms(d: i64) -> Vec<(i64, i64, i64)> {
    assert!(d < 0, "discriminant must be negative");
    let mut forms = Vec::new();
    // a ≤ sqrt(|d|/3).
    let a_max = ((-d) as f64 / 3.0).sqrt().floor() as i64;
    for a in 1..=a_max.max(1) {
        for b in -a..=a {
            // b ≡ d (mod 2)
            if (b - d).rem_euclid(2) != 0 {
                continue;
            }
            let num = b * b - d; // = b² − d = 4ac > 0
            let denom = 4 * a;
            if num % denom != 0 {
                continue;
            }
            let c = num / denom;
            if c < a {
                continue; // need a ≤ c
            }
            // boundary sign condition: b ≥ 0 if |b| = a or a = c
            if (b.abs() == a || a == c) && b < 0 {
                continue;
            }
            if gcd3(a.abs(), b.abs(), c.abs()) != 1 {
                continue; // primitive only
            }
            forms.push((a, b, c));
        }
    }
    forms
}

fn gcd3(a: i64, b: i64, c: i64) -> i64 {
    gcd2(gcd2(a, b), c)
}
fn gcd2(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.abs()
}

/// Class number of `ℚ(√d)` (`d < 0` fundamental) = number of reduced forms.
pub fn class_number_imag_quadratic(d: i64) -> usize {
    reduced_forms(d).len()
}

/// The assembled absolute Hilbert class field of `K = ℚ(√d)`.
///
/// Note: the absolute model returned by Algorithm 1A is **correct but not
/// reduced** — a compositum primitive element `√d + s·j` is far from a power
/// basis of the maximal order, so the polynomial discriminant carries a large
/// index factor. (The in-repo `polredabs` does not minimise models this large;
/// good reduction is a separate downstream concern.) The field discriminant is
/// nonetheless pinned exactly by the invariant `poly_discriminant = d^h · k²`
/// for an integer index `k` — see [`HilbertClassField::field_discriminant`].
#[derive(Clone, Debug)]
pub struct HilbertClassField {
    /// Fundamental discriminant `d < 0`.
    pub d: i64,
    /// Class number `h = [H_K : K]`.
    pub class_number: usize,
    /// Absolute defining polynomial of `H_K` over `ℚ` (monic, degree `2h`),
    /// via Algorithm 1A.
    pub absolute: AbsoluteField,
    /// Discriminant of `absolute.poly` (polynomial, not field, discriminant).
    pub poly_discriminant: Integer,
}

impl HilbertClassField {
    /// The field discriminant of `H_K`, which for the Hilbert class field equals
    /// `d^h` (relative extension unramified ⟹ `disc H_K = d_K^{[H_K:K]}`).
    /// Returns `Some(d^h)` together with the squared index `k` iff the
    /// polynomial discriminant factors as `d^h · k²` (the consistency check that
    /// the assembled field is the Hilbert class field); `None` otherwise.
    pub fn field_discriminant(&self) -> Option<(Integer, Integer)> {
        let dh = Integer::from(self.d).pow(self.class_number as u32);
        if self.poly_discriminant.clone() % dh.clone() != Integer::from(0) {
            return None;
        }
        let k2 = self.poly_discriminant.clone() / dh.clone();
        if k2 <= Integer::from(0) || !k2.is_perfect_square() {
            return None;
        }
        Some((dh, k2.sqrt().ok()?))
    }
}

/// Convert a monic `UnivariatePolynomial<Rational>` with integer coefficients to
/// `Vec<Integer>` (low→high). Errors if any coefficient is not integral.
fn integer_coeffs(p: &UnivariatePolynomial<Rational>) -> Result<Vec<Integer>, String> {
    p.coefficients()
        .iter()
        .map(|q| {
            if q.denominator() == &Integer::from(1) {
                Ok(q.numerator().clone())
            } else {
                Err(format!("non-integral coefficient {q:?} in absolute HCF poly"))
            }
        })
        .collect()
}

/// Assemble the absolute Hilbert class field of `K = ℚ(√d)` (`d < 0`
/// fundamental) from a given Hilbert class polynomial `hcp = H_d`.
///
/// * `d` — fundamental discriminant `< 0`.
/// * `hcp` — coefficients of `H_d` low→high, **monic** of degree `h`
///   (`hcp.len() == h + 1`, `hcp[h] == 1`).
///
/// Builds `g = T² − d` (minimal polynomial of `√d`) and feeds `H_d` as the
/// relative polynomial over `K` to Algorithm 1A, then `polredabs`-reduces and
/// records the field discriminant.
pub fn hilbert_class_field_from_hcp(
    d: i64,
    hcp: &[Integer],
) -> Result<HilbertClassField, String> {
    if d >= 0 {
        return Err("discriminant must be negative".into());
    }
    if hcp.len() < 2 {
        return Err("Hilbert class polynomial must have degree ≥ 1".into());
    }
    let h = hcp.len() - 1;
    if hcp[h] != Integer::from(1) {
        return Err("Hilbert class polynomial must be monic".into());
    }

    // g = T² − d  (d < 0 ⟹ constant term −d = |d| > 0): coeffs [−d, 0, 1].
    let g = UnivariatePolynomial::new(vec![
        Rational::from_integer(-d),
        Rational::from_integer(0),
        Rational::from_integer(1),
    ]);

    // Relative polynomial H_d over K: monic, drop the leading 1; each integer
    // coefficient is a degree-0 element of K, i.e. a length-1 θ-polynomial.
    let h_coeffs: Vec<Vec<Rational>> = hcp[..h]
        .iter()
        .map(|c| vec![Rational::from_integer(c.clone())])
        .collect();

    let absolute = absolute_defining_polynomial(&g, &h_coeffs)?;
    let abs_int = integer_coeffs(&absolute.poly)?;
    let poly_discriminant = poly_discriminant_fn(&abs_int);

    Ok(HilbertClassField {
        d,
        class_number: h,
        absolute,
        poly_discriminant,
    })
}

/// Sum of cubes of the divisors of `n` (`σ₃(n)`), for the `E₄` q-series.
fn sigma3(n: u64) -> Integer {
    let mut s = Integer::from(0);
    let mut k = 1u64;
    while k * k <= n {
        if n % k == 0 {
            let d1 = k;
            let d2 = n / k;
            s = s + Integer::from(d1).pow(3);
            if d2 != d1 {
                s = s + Integer::from(d2).pow(3);
            }
        }
        k += 1;
    }
    s
}

/// `j(τ) = E₄(τ)³ / Δ(τ)` from the q-series, `q = e^{2πiτ}`, truncated at
/// `nterms` (chosen so `|q|^{nterms} < 2^{-prec}`):
///   `Δ = q · ∏_{n≥1}(1−qⁿ)²⁴`,  `E₄ = 1 + 240·Σ_{n≥1} σ₃(n) qⁿ`.
fn j_invariant(q: &ComplexMPFR, one: &ComplexMPFR, prec: u32, nterms: usize) -> ComplexMPFR {
    let mut qn = q.clone(); // qⁿ, starting at q¹
    let mut delta_prod = one.clone();
    let mut e4 = one.clone();
    for n in 1..=nterms {
        let factor = (one - &qn).powi(24); // (1−qⁿ)²⁴
        delta_prod = &delta_prod * &factor;
        let coeff = ComplexMPFR::with_val_integers(
            prec,
            &(Integer::from(240) * sigma3(n as u64)),
            &Integer::from(0),
        );
        e4 = &e4 + &(&coeff * &qn);
        qn = &qn * q;
    }
    let delta = q * &delta_prod;
    let e4_cubed = e4.powi(3);
    &e4_cubed / &delta
}

/// Multiply the polynomial `poly` (coeffs low→high) by `(X − j)`.
fn mul_linear(poly: &[ComplexMPFR], j: &ComplexMPFR, prec: u32) -> Vec<ComplexMPFR> {
    let zero = ComplexMPFR::with_val(prec, (0.0, 0.0));
    let n = poly.len();
    let mut out = vec![zero; n + 1];
    for i in 0..n {
        out[i + 1] = &out[i + 1] + &poly[i]; // X · p_i
        out[i] = &out[i] - &(j * &poly[i]); // − j · p_i
    }
    out
}

/// Compute the Hilbert class polynomial `H_d ∈ ℤ[X]` for a fundamental
/// discriminant `d < 0`, by the CM/singular-moduli route: evaluate `j(τ)` at
/// each reduced-form CM point to high precision and round the symmetric
/// functions to integers. Returns the coefficients low→high (monic, degree
/// `h(d)`).
pub fn hilbert_class_polynomial(d: i64) -> Vec<Integer> {
    let forms = reduced_forms(d);
    let absd = -d;

    // Precision (bits): the largest |coeff| of H_d is ~ exp(π√|d| · Σ 1/a).
    let sum_inv_a: f64 = forms.iter().map(|&(a, _, _)| 1.0 / (a as f64)).sum();
    let need =
        std::f64::consts::PI * (absd as f64).sqrt() * sum_inv_a / std::f64::consts::LN_2;
    let prec: u32 = need.ceil() as u32 + 80; // generous rounding margin

    // q-series length: need |q|^N < 2^-prec with |q| = exp(−π√|d|/a_max).
    let a_max = forms.iter().map(|&(a, _, _)| a).max().unwrap_or(1);
    let nterms = (prec as f64 * std::f64::consts::LN_2 * a_max as f64
        / (std::f64::consts::PI * (absd as f64).sqrt()))
    .ceil() as usize
        + 5;

    let pi = RealMPFR::pi(prec);
    let two_pi = &pi * &RealMPFR::with_val(prec, 2.0);
    let sqrt_absd = RealMPFR::with_val_integer(prec, &Integer::from(absd)).sqrt();
    let zero_r = RealMPFR::with_val(prec, 0.0);
    let one_c = ComplexMPFR::with_val(prec, (1.0, 0.0));

    let mut poly: Vec<ComplexMPFR> = vec![one_c.clone()];
    for &(a, b, _c) in &forms {
        // τ = −b/(2a) + i·√|d|/(2a)
        let tau_re = RealMPFR::with_val_rational(prec, &Rational::new(-b, 2 * a).unwrap());
        let tau_im = &sqrt_absd / &RealMPFR::with_val(prec, (2 * a) as f64);
        let tau = ComplexMPFR::with_val_reals(tau_re, tau_im);

        // q = exp(2πiτ)
        let two_pi_i = ComplexMPFR::with_val_reals(zero_r.clone(), two_pi.clone());
        let q = (&two_pi_i * &tau).exp();

        let j = j_invariant(&q, &one_c, prec, nterms);
        poly = mul_linear(&poly, &j, prec);
    }

    poly.iter().map(|c| c.real_part().round()).collect()
}

/// Full imaginary-quadratic Hilbert class field of `K = ℚ(√d)` from scratch:
/// compute `H_d` by CM, then assemble the absolute field via Algorithm 1A.
pub fn hilbert_class_field(d: i64) -> Result<HilbertClassField, String> {
    let hcp = hilbert_class_polynomial(d);
    hilbert_class_field_from_hcp(d, &hcp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn reduced_forms_disc_m23() {
        // h(−23) = 3: forms [1,1,6], [2,1,3], [2,−1,3].
        let mut f = reduced_forms(-23);
        f.sort();
        assert_eq!(f, vec![(1, 1, 6), (2, -1, 3), (2, 1, 3)]);
        assert_eq!(class_number_imag_quadratic(-23), 3);
    }

    #[test]
    fn reduced_forms_disc_m4_m3() {
        // h(−4) = 1 (form [1,0,1]); h(−3) = 1 (form [1,1,1]).
        assert_eq!(reduced_forms(-4), vec![(1, 0, 1)]);
        assert_eq!(reduced_forms(-3), vec![(1, 1, 1)]);
    }

    #[test]
    fn reduced_forms_disc_m47() {
        // h(−47) = 5.
        assert_eq!(class_number_imag_quadratic(-47), 5);
    }

    #[test]
    fn hilbert_class_field_m23() {
        // H_{−23}(X) = X³ + 3491750 X² − 5151296875 X + 12771880859375
        // (PARI polclass(-23)).  HCF = ℚ(√−23, j), degree 6, disc = (−23)³.
        let hcp = ints(&[12771880859375, -5151296875, 3491750, 1]);
        let hk = hilbert_class_field_from_hcp(-23, &hcp).expect("HCF assembly");

        assert_eq!(hk.class_number, 3);
        assert_eq!(hk.absolute.poly.degree(), Some(6), "absolute degree 2h = 6");

        // Model-independent identification: the field discriminant of the
        // Hilbert class field is d^h = (−23)³ = −12167, so the polynomial
        // discriminant must factor as (−12167)·(index²).
        let (dfield, index) = hk
            .field_discriminant()
            .expect("poly disc must be d^h times a perfect square");
        assert_eq!(dfield, Integer::from(-12167));
        assert!(index > Integer::from(1), "the raw compositum model is non-maximal");
    }

    #[test]
    fn class_poly_h1_cases() {
        // h(−3)=h(−4)=1. j(ω)=0, j(i)=1728.  H_d = X − j.
        assert_eq!(hilbert_class_polynomial(-3), ints(&[0, 1]));
        assert_eq!(hilbert_class_polynomial(-4), ints(&[-1728, 1]));
        // d=−163 (h=1): j = −640320³ = −262537412640768000 (precision stress).
        assert_eq!(
            hilbert_class_polynomial(-163),
            ints(&[262537412640768000, 1])
        );
    }

    #[test]
    fn class_poly_m23_matches_polclass() {
        // CM computation of H_{−23} must match PARI polclass(-23).
        assert_eq!(
            hilbert_class_polynomial(-23),
            ints(&[12771880859375, -5151296875, 3491750, 1])
        );
    }

    #[test]
    fn hilbert_class_field_from_scratch_disc_invariant() {
        // Full pipeline j(τ) → H_d → absolute HCF; the field discriminant must
        // be d^h for each (the unramified Hilbert class field). Validates the
        // CM computation arithmetically without hardcoding each H_d.
        for &d in &[-23i64, -31, -47] {
            let hk = hilbert_class_field(d).expect("HCF from scratch");
            let (dfield, _idx) = hk
                .field_discriminant()
                .expect("poldisc = d^h · square");
            assert_eq!(dfield, Integer::from(d).pow(hk.class_number as u32));
        }
    }
}
