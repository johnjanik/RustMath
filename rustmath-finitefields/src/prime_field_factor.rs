//! [`FactorizationField`] for GF(p) ([`PrimeField`]).
//!
//! Specification: MAGMA Handbook ch. 23 §23.8.1 `Factorization` over a finite field (digest
//! `rustmath-lava/docs/Chapter_23_Univariate_Polynomial_Rings.md:294`), as needed by the
//! canonical forms of ch. 26 §26.12.2 (`PrimaryRationalForm`, `JordanForm`).
//!
//! Algorithm: E. R. Berlekamp, *Factoring polynomials over large finite fields*, Math. Comp. 24
//! (1970) 713-735, through [`rustmath_polynomials::berlekamp::factor`]: the number of
//! irreducible factors of each square-free part is `dim ker(Q - I)`, computed exactly, so the
//! factorization is certified for a prime p. The prime is checked by that function (a proof for
//! every size). Factor order: degree, then the coefficients from the top down as integers in
//! `[0, p)`, which is the order of PARI/GP `factormod`.

use crate::prime_field::PrimeField;
use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::berlekamp;
use rustmath_polynomials::factorization_field::{check_monic, check_product};
use rustmath_polynomials::{FactorizationField, UnivariatePolynomial};

/// GF(p) by Berlekamp's method (module docs).
///
/// * p is the modulus of the bound coefficients. If every coefficient is an unbound
///   `Ring::zero()`/`one()` constant, p is unknown and the result is `Err(InvalidArgument)`;
///   bound coefficients with two different moduli are `Err(InvalidArgument)` too.
/// * A modulus that is not prime is `Err(InvalidArgument)` from the primality check of
///   [`berlekamp::factor`]; its other errors (`NotSupported` for a probable prime whose proof is
///   unresolved, or for the randomized split that did not finish) pass through.
/// * Every coefficient of every factor is bound (`PrimeField::new(c, p)`).
impl FactorizationField for PrimeField {
    fn factor_monic_polynomial(
        f: &UnivariatePolynomial<PrimeField>,
    ) -> Result<Vec<(UnivariatePolynomial<PrimeField>, usize)>> {
        let who = "FactorizationField for PrimeField";
        check_monic(f, who)?;
        let mut p: Option<&Integer> = None;
        for c in f.coefficients() {
            let m = c.modulus();
            if m.is_zero() {
                continue;
            }
            match p {
                None => p = Some(m),
                Some(q) if q != m => {
                    return Err(MathError::InvalidArgument(format!(
                        "{who}: coefficients of two different fields (moduli {q} and {m})"
                    )))
                }
                Some(_) => {}
            }
        }
        let p = p.cloned().ok_or_else(|| {
            MathError::InvalidArgument(format!(
                "{who}: the prime p is unknown: every coefficient is an unbound integer constant"
            ))
        })?;
        // berlekamp reads the integer values modulo p (an unbound value is an integer).
        let zc: Vec<Integer> = f.coefficients().iter().map(|c| c.value().clone()).collect();
        let (u, list) = berlekamp::factor(&zc, &p)?;
        if !u.is_one() {
            return Err(MathError::InvalidOperation(format!(
                "internal: {who}: the unit of a monic polynomial is {u}, not 1"
            )));
        }
        let mut out = Vec::with_capacity(list.len());
        for (g, e) in list {
            let coeffs = g
                .into_iter()
                .map(|c| PrimeField::new(c, p.clone()))
                .collect::<Result<Vec<_>>>()?;
            let e = usize::try_from(e).map_err(|_| {
                MathError::InvalidOperation(format!("internal: {who}: multiplicity overflow"))
            })?;
            out.push((UnivariatePolynomial::new(coeffs), e));
        }
        check_product(f, &out, PrimeField::new(Integer::one(), p)?, who)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::Ring;

    fn gf(v: i64, p: i64) -> PrimeField {
        PrimeField::new(Integer::from(v), Integer::from(p)).unwrap()
    }

    fn gp(c: &[i64], p: i64) -> UnivariatePolynomial<PrimeField> {
        UnivariatePolynomial::new(c.iter().map(|&v| gf(v, p)).collect())
    }

    // gp: lift(factormod(x^5+x^4+x+1, 5)) = [x + 1, 1; x^2 + 2, 1; x^2 + 3, 1]
    // gp: lift(factormod((x+1)^2*(x^2+2), 5)) = [x + 1, 2; x^2 + 2, 1]
    #[test]
    fn prime_field_factorization_matches_factormod() {
        let f = gp(&[1, 1, 0, 0, 1, 1], 5);
        let got = PrimeField::factor_monic_polynomial(&f).unwrap();
        assert_eq!(
            got,
            vec![
                (gp(&[1, 1], 5), 1),
                (gp(&[2, 0, 1], 5), 1),
                (gp(&[3, 0, 1], 5), 1)
            ]
        );
        // (x+1)^2 (x^2+2) = x^4 + 2x^3 + 3x^2 + 4x + 2
        let f = gp(&[2, 4, 3, 2, 1], 5);
        let got = PrimeField::factor_monic_polynomial(&f).unwrap();
        assert_eq!(got, vec![(gp(&[1, 1], 5), 2), (gp(&[2, 0, 1], 5), 1)]);
        for (g, _) in &got {
            assert!(g.coefficients().iter().all(|c| !c.is_unbound_constant()));
        }
    }

    // A monic polynomial with unbound coefficients next to a bound one: x^2 + 1 mod 5 with the
    // constant term from PrimeField::one(). gp: lift(factormod(x^2+1, 5)) = [x + 2, 1; x + 3, 1]
    #[test]
    fn prime_field_factorization_mixed_coefficients() {
        let f = UnivariatePolynomial::new(vec![PrimeField::one(), gf(0, 5), gf(1, 5)]);
        let got = PrimeField::factor_monic_polynomial(&f).unwrap();
        assert_eq!(got, vec![(gp(&[2, 1], 5), 1), (gp(&[3, 1], 5), 1)]);
    }

    #[test]
    fn prime_field_factorization_rejects() {
        // x + 1 from the unbound constants only: p is unknown.
        let f = UnivariatePolynomial::new(vec![PrimeField::one(), PrimeField::one()]);
        assert!(matches!(
            PrimeField::factor_monic_polynomial(&f),
            Err(MathError::InvalidArgument(_))
        ));
        // x^2 + 1 mod 15: berlekamp rejects the composite modulus.
        let f = gp(&[1, 0, 1], 15);
        assert!(matches!(
            PrimeField::factor_monic_polynomial(&f),
            Err(MathError::InvalidArgument(_))
        ));
        // zero, constant and non-monic inputs.
        for f in [gp(&[0], 5), gp(&[3], 5), gp(&[1, 2], 5)] {
            assert!(matches!(
                PrimeField::factor_monic_polynomial(&f),
                Err(MathError::InvalidArgument(_))
            ));
        }
    }
}
