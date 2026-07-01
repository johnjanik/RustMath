//! Interop between LFSR connection polynomials and `rustmath-polynomials`.
//!
//! Port support for MAGMA Handbook **Chapter 158 — Pseudo-random Bit Sequences**,
//! §158.2. The LFSR / Berlekamp–Massey machinery in [`crate::lfsr`] stores a
//! connection polynomial `C(D)` as a plain `Vec<F>` of field elements. This module
//! bridges that representation to `rustmath_polynomials::UnivariatePolynomial`
//! (consumed **read-only**) so callers can reuse the polynomial toolbox (degree,
//! irreducibility, factorisation, …) on connection / characteristic polynomials
//! over `GF(p)`.

use rustmath_finitefields::PrimeField;
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;

/// Convert a `GF(p)` connection polynomial `[c_0, c_1, …]` into a
/// `UnivariatePolynomial<Integer>` whose coefficients are the canonical
/// representatives in `[0, p)`.
pub fn connection_polynomial_to_univariate(c: &[PrimeField]) -> UnivariatePolynomial<Integer> {
    let coeffs: Vec<Integer> = c.iter().map(|e| e.value().clone()).collect();
    UnivariatePolynomial::new(coeffs)
}

/// Convert a `UnivariatePolynomial<Integer>` into a `GF(p)` connection polynomial
/// `[c_0, c_1, …]`, reducing each coefficient modulo the prime `p`.
///
/// Returns an error if any `PrimeField` element cannot be built (e.g. `p ≤ 1`).
pub fn connection_polynomial_from_univariate(
    poly: &UnivariatePolynomial<Integer>,
    p: &Integer,
) -> rustmath_core::Result<Vec<PrimeField>> {
    poly.coefficients()
        .iter()
        .map(|a| PrimeField::new(a.clone(), p.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gf7(v: i64) -> PrimeField {
        PrimeField::new(Integer::from(v), Integer::from(7)).unwrap()
    }

    #[test]
    fn round_trip_gf7() {
        // C(D) = 1 + 3D + 5D^2 over GF(7).
        let c = vec![gf7(1), gf7(3), gf7(5)];
        let poly = connection_polynomial_to_univariate(&c);
        assert_eq!(poly.degree(), Some(2));
        assert_eq!(poly.coefficients()[1], Integer::from(3));

        let back = connection_polynomial_from_univariate(&poly, &Integer::from(7)).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn reduces_coefficients_mod_p() {
        // Coefficients get reduced mod 7: 10 -> 3, 8 -> 1.
        let poly = UnivariatePolynomial::new(vec![Integer::from(8), Integer::from(10)]);
        let c = connection_polynomial_from_univariate(&poly, &Integer::from(7)).unwrap();
        assert_eq!(c[0], gf7(1));
        assert_eq!(c[1], gf7(3));
    }
}
