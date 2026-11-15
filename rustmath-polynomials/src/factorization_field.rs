//! Fields over which univariate polynomials factor (the trait [`FactorizationField`]).
//!
//! Specification: MAGMA Handbook ch. 23, *Univariate Polynomial Rings* (digest
//! `rustmath-lava/docs/Chapter_23_Univariate_Polynomial_Rings.md:294`), §23.8.1
//! `Factorization`, and the precondition of ch. 26 §26.12.2 (`PrimaryRationalForm`,
//! `JordanForm`, `PrimaryInvariantFactors`): "a field allowing factorization of univariate
//! polynomials". The matrix canonical forms of `rustmath-matrix` (`primary_forms`) consume this
//! trait.
//!
//! Implementations:
//! * Q ([`Rational`], here): clear the denominators, then the Zassenhaus factorizer over Z
//!   (H. Zassenhaus, *On Hensel factorization I*, J. Number Theory 1 (1969) 291-311) through
//!   [`crate::factorization::factor_over_integers`].
//! * GF(p) (`PrimeField`): in rustmath-finitefields (module `prime_field_factor`), by
//!   Berlekamp's method through [`crate::berlekamp::factor`].
//!
//! The trait lives in this crate because it defines [`UnivariatePolynomial`] and has the Q
//! factorizer, and both rustmath-matrix and rustmath-finitefields depend on it, so neither
//! needs a new crate edge.

use crate::factorization::factor_over_integers;
use crate::univariate::UnivariatePolynomial;
use rustmath_core::{Field, MathError, Result, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// A field over which univariate polynomials factor into monic irreducibles.
pub trait FactorizationField: Field {
    /// The factorization of a monic `f` of positive degree.
    ///
    /// `Ok([(g_1, e_1), ..., (g_k, e_k)])`: every `g_i` monic, irreducible, of positive degree,
    /// pairwise distinct, `e_i >= 1`, and `f = Π g_i^e_i` exactly (each impl multiplies back and
    /// compares). Order: by degree first, deterministic; each impl states the rest.
    ///
    /// * `Err(InvalidArgument)` if `f` is zero, constant or not monic.
    /// * Errors of the underlying factorizer pass through (never a partial list).
    /// * `Err(InvalidOperation)` if the multiply-back check fails.
    #[allow(clippy::type_complexity)]
    fn factor_monic_polynomial(
        f: &UnivariatePolynomial<Self>,
    ) -> Result<Vec<(UnivariatePolynomial<Self>, usize)>>;
}

/// The input check shared by every impl of [`FactorizationField`]: `Ok(())` iff `f` is nonzero,
/// of positive degree and monic; otherwise `Err(InvalidArgument)` naming `who`.
pub fn check_monic<F: Field>(f: &UnivariatePolynomial<F>, who: &str) -> Result<()> {
    match f.degree() {
        None => Err(MathError::InvalidArgument(format!(
            "{who}: the zero polynomial has no factorization"
        ))),
        Some(0) => Err(MathError::InvalidArgument(format!(
            "{who}: a constant polynomial has no factorization into monic irreducibles"
        ))),
        Some(d) => {
            if f.coeff(d).is_one() {
                Ok(())
            } else {
                Err(MathError::InvalidArgument(format!(
                    "{who}: the polynomial must be monic"
                )))
            }
        }
    }
}

/// The multiply-back check shared by every impl of [`FactorizationField`]: `Ok(())` iff
/// `Π g^e == f` (the product starts from the constant `one`, which should be bound to the
/// field of `f` for parent-carrying types); otherwise `Err(InvalidOperation)` naming `who`.
pub fn check_product<F: Field>(
    f: &UnivariatePolynomial<F>,
    factors: &[(UnivariatePolynomial<F>, usize)],
    one: F,
    who: &str,
) -> Result<()> {
    let mut acc = UnivariatePolynomial::new(vec![one]);
    for (g, e) in factors {
        for _ in 0..*e {
            acc = acc * g.clone();
        }
    }
    if acc == *f {
        Ok(())
    } else {
        Err(MathError::InvalidOperation(format!(
            "internal: {who}: the product of the factors is not the input"
        )))
    }
}

/// Q: Zassenhaus over Z after clearing denominators.
///
/// Order: PARI/GP `factor` order. The factors are sorted by degree, then by the coefficients of
/// the primitive integer factor (positive leading coefficient, as returned by
/// [`factor_over_integers`]) from the top down, compared as integers, and only then divided by
/// their leading coefficient. So `(x-3)(x+3)(x-1/2)` gives `x-3, x+3, x-1/2`, because `2x-1`
/// comes after `x+3` (PARI: `factor((x-3)*(x+3)*(2*x-1))` is `[x-3,1; x+3,1; 2*x-1,1]`).
/// MAGMA's order over Q is not documented. Zassenhaus' recombination cap
/// (`Err(NotSupported)`) and its other errors pass through.
impl FactorizationField for Rational {
    fn factor_monic_polynomial(
        f: &UnivariatePolynomial<Rational>,
    ) -> Result<Vec<(UnivariatePolynomial<Rational>, usize)>> {
        let who = "FactorizationField for Rational";
        check_monic(f, who)?;
        // D = lcm of the denominators; D·f is primitive in Z[x] (its content divides the
        // leading coefficient D and is prime to every prime of D).
        let mut den = Integer::one();
        for c in f.coefficients() {
            den = den.lcm(c.denominator());
        }
        let zc: Vec<Integer> = f
            .coefficients()
            .iter()
            .map(|c| c.numerator().clone() * (den.clone() / c.denominator().clone()))
            .collect();
        let factors = factor_over_integers(&UnivariatePolynomial::new(zc))?;
        // Drop the content (degree 0), then sort the primitive integer factors (PARI order).
        let mut prim: Vec<(UnivariatePolynomial<Integer>, u32)> = factors
            .into_iter()
            .filter(|(g, _)| g.degree().is_some_and(|d| d > 0))
            .collect();
        prim.sort_by(|a, b| {
            let (ca, cb) = (a.0.coefficients(), b.0.coefficients());
            ca.len()
                .cmp(&cb.len())
                .then_with(|| ca.iter().rev().cmp(cb.iter().rev()))
        });
        let mut out = Vec::with_capacity(prim.len());
        for (g, e) in prim {
            let lc = Rational::from_integer(g.leading_coefficient().cloned().ok_or_else(|| {
                MathError::InvalidOperation(format!("internal: {who}: a zero factor"))
            })?);
            let monic: Vec<Rational> = g
                .coefficients()
                .iter()
                .map(|c| Rational::from_integer(c.clone()) / lc.clone())
                .collect();
            let e = usize::try_from(e).map_err(|_| {
                MathError::InvalidOperation(format!("internal: {who}: multiplicity overflow"))
            })?;
            if e == 0 {
                return Err(MathError::InvalidOperation(format!(
                    "internal: {who}: multiplicity 0"
                )));
            }
            out.push((UnivariatePolynomial::new(monic), e));
        }
        check_product(f, &out, Rational::one(), who)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(v: i64) -> Rational {
        Rational::from_integer(v)
    }

    fn qr(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }

    fn poly(cs: &[Rational]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(cs.to_vec())
    }

    fn prod(fs: &[UnivariatePolynomial<Rational>]) -> UnivariatePolynomial<Rational> {
        fs.iter().fold(poly(&[q(1)]), |acc, g| acc * g.clone())
    }

    // gp: factor((x-3)*(x+3)*(2*x-1)*(x^2+1)^2)
    //   = [x - 3, 1; x + 3, 1; 2*x - 1, 1; x^2 + 1, 2]
    // The monic input is (x-3)(x+3)(x-1/2)(x^2+1)^2; 2x-1 sorts after x+3 (PARI order on the
    // primitive integer factors), so x-1/2 is third.
    #[test]
    fn rational_factorization_pari_order() {
        let xm3 = poly(&[q(-3), q(1)]);
        let xp3 = poly(&[q(3), q(1)]);
        let xmh = poly(&[qr(-1, 2), q(1)]);
        let x2p1 = poly(&[q(1), q(0), q(1)]);
        let f = prod(&[
            xm3.clone(),
            xp3.clone(),
            xmh.clone(),
            x2p1.clone(),
            x2p1.clone(),
        ]);
        let got = Rational::factor_monic_polynomial(&f).unwrap();
        assert_eq!(got, vec![(xm3, 1), (xp3, 1), (xmh, 1), (x2p1, 2)]);
    }

    // gp: factor(x^4 - 11*x^3 + 42*x^2 - 64*x + 32) = [x - 4, 2; x - 2, 1; x - 1, 1]
    // (the characteristic polynomial of Sage's matrix `m` in the jordan_form docs).
    #[test]
    fn rational_factorization_charpoly() {
        let f = poly(&[q(32), q(-64), q(42), q(-11), q(1)]);
        let got = Rational::factor_monic_polynomial(&f).unwrap();
        assert_eq!(
            got,
            vec![
                (poly(&[q(-4), q(1)]), 2),
                (poly(&[q(-2), q(1)]), 1),
                (poly(&[q(-1), q(1)]), 1),
            ]
        );
    }

    // gp: factor(x^2 + 1/2) = Mat([2*x^2 + 1, 1]): irreducible, so the monic factor is
    // x^2 + 1/2 itself. The denominators are cleared before Zassenhaus.
    #[test]
    fn rational_factorization_irreducible_with_denominator() {
        let f = poly(&[qr(1, 2), q(0), q(1)]);
        let got = Rational::factor_monic_polynomial(&f).unwrap();
        assert_eq!(got, vec![(f, 1)]);
    }

    #[test]
    fn rational_factorization_rejects() {
        for f in [poly(&[q(0)]), poly(&[q(1)]), poly(&[q(1), q(2)])] {
            assert!(matches!(
                Rational::factor_monic_polynomial(&f),
                Err(MathError::InvalidArgument(_))
            ));
        }
    }
}
