//! Factorization capability for the constant field K of K(x).
//!
//! Decomposing div(f) into places requires factoring K[x] polynomials into
//! monic irreducibles. That is not possible for an arbitrary `Field`, so the
//! capability is expressed as a trait, implemented for `Rational` (via
//! Zassenhaus over Z after clearing denominators) and for `GFp<P>` (via
//! Cantor-Zassenhaus in `rustmath_polynomials::fp_factor`).
//!
//! Every implementation is self-certifying: the factorization is multiplied
//! back together and compared with the input; a mismatch is an `Err`, never a
//! silently wrong answer.

use super::gfp::GFp;
use rustmath_core::{EuclideanDomain, Field, MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;

/// A field over which univariate polynomials can be fully factored.
pub trait FactorableConstantField: Field + EuclideanDomain {
    /// Factor a nonzero polynomial into `(unit, [(monic irreducible, multiplicity)])`
    /// with `poly = unit * prod f_i^{e_i}` exactly.
    ///
    /// - A nonzero constant yields `(constant, [])`.
    /// - `Err(DivisionByZero)` on the zero polynomial.
    /// - `Err` if the certified reconstruction check fails (never a wrong list).
    fn factor_poly(
        poly: &UnivariatePolynomial<Self>,
    ) -> Result<(Self, Vec<(UnivariatePolynomial<Self>, usize)>)>;
}

/// Multiply a factorization back together: `unit * prod f_i^{e_i}`.
fn reconstruct<K: Field + EuclideanDomain>(
    unit: &K,
    factors: &[(UnivariatePolynomial<K>, usize)],
) -> UnivariatePolynomial<K> {
    let mut acc = UnivariatePolynomial::constant(unit.clone());
    for (f, e) in factors {
        for _ in 0..*e {
            acc = acc * f.clone();
        }
    }
    acc
}

/// Certify `poly == unit * prod f_i^{e_i}` and that every factor is monic
/// and non-constant; return the factorization or an honest `Err`.
fn certify<K: Field + EuclideanDomain>(
    poly: &UnivariatePolynomial<K>,
    unit: K,
    factors: Vec<(UnivariatePolynomial<K>, usize)>,
) -> Result<(K, Vec<(UnivariatePolynomial<K>, usize)>)> {
    for (f, e) in &factors {
        if !f.is_monic() || f.degree().unwrap_or(0) == 0 || *e == 0 {
            return Err(MathError::NumericalError(
                "factor_poly produced a non-monic, constant, or zero-multiplicity factor"
                    .to_string(),
            ));
        }
    }
    if &reconstruct(&unit, &factors) != poly {
        return Err(MathError::NumericalError(
            "factor_poly reconstruction check failed".to_string(),
        ));
    }
    Ok((unit, factors))
}

/// Merge a monic factor into an accumulator, summing multiplicities.
fn merge<K: Field + EuclideanDomain>(
    acc: &mut Vec<(UnivariatePolynomial<K>, usize)>,
    f: UnivariatePolynomial<K>,
    e: usize,
) {
    for (g, m) in acc.iter_mut() {
        if *g == f {
            *m += e;
            return;
        }
    }
    acc.push((f, e));
}

impl FactorableConstantField for Rational {
    fn factor_poly(
        poly: &UnivariatePolynomial<Rational>,
    ) -> Result<(Rational, Vec<(UnivariatePolynomial<Rational>, usize)>)> {
        if poly.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let lc = poly
            .leading_coefficient()
            .expect("nonzero polynomial")
            .clone();
        if poly.degree().unwrap_or(0) == 0 {
            return Ok((lc, vec![]));
        }

        // Clear denominators: multiply by the lcm of coefficient denominators.
        let mut den_lcm = Integer::one();
        for c in poly.coefficients() {
            den_lcm = den_lcm.lcm(c.denominator());
        }
        let int_coeffs: Vec<Integer> = poly
            .coefficients()
            .iter()
            .map(|c| {
                let (q, r) = (c.numerator().clone() * den_lcm.clone())
                    .div_rem(c.denominator())
                    .expect("nonzero denominator");
                debug_assert!(r.is_zero());
                q
            })
            .collect();
        let zpoly = UnivariatePolynomial::new(int_coeffs);

        // Zassenhaus factorization over Z.
        let zfactors = rustmath_polynomials::factor_over_integers(&zpoly)?;

        // Convert non-constant factors to monic polynomials over Q, merging
        // multiplicities (associate integer factors become equal monic ones).
        let mut factors: Vec<(UnivariatePolynomial<Rational>, usize)> = Vec::new();
        for (zf, e) in zfactors {
            if zf.degree().unwrap_or(0) == 0 {
                continue; // integer content: absorbed into the unit
            }
            let qf = UnivariatePolynomial::new(
                zf.coefficients()
                    .iter()
                    .map(|c| Rational::from_integer(c.clone()))
                    .collect(),
            )
            .make_monic();
            merge(&mut factors, qf, e as usize);
        }

        // poly = lc * prod (monic irreducibles): certified below.
        certify(poly, lc, factors)
    }
}

impl<const P: u64> FactorableConstantField for GFp<P> {
    fn factor_poly(
        poly: &UnivariatePolynomial<GFp<P>>,
    ) -> Result<(GFp<P>, Vec<(UnivariatePolynomial<GFp<P>>, usize)>)> {
        if poly.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        if !GFp::<P>::modulus_is_prime() {
            return Err(MathError::InvalidArgument(format!(
                "GFp modulus {} is not prime; GF(p)[x] factorization undefined",
                P
            )));
        }
        // fp_factor works in i64 with products of residues: need (P-1)^2 < i64::MAX.
        if P > 3_037_000_499 {
            return Err(MathError::NotSupported(format!(
                "GFp modulus {} too large for the i64-based factorizer",
                P
            )));
        }
        let lc = poly
            .leading_coefficient()
            .expect("nonzero polynomial")
            .clone();
        if poly.degree().unwrap_or(0) == 0 {
            return Ok((lc, vec![]));
        }

        let coeffs_i64: Vec<i64> = poly.coefficients().iter().map(|c| c.value() as i64).collect();
        let raw = rustmath_polynomials::fp_factor::factor(&coeffs_i64, P as i64);

        // fp_factor::factor returns DISTINCT monic irreducibles without
        // multiplicities; recover each multiplicity by repeated exact division.
        let mut factors: Vec<(UnivariatePolynomial<GFp<P>>, usize)> = Vec::new();
        for fac in raw {
            let f = UnivariatePolynomial::new(
                fac.iter().map(|&c| GFp::<P>::new(c)).collect::<Vec<_>>(),
            );
            let mut e = 0usize;
            let mut work = poly.clone();
            loop {
                let (q, r) = work.div_rem(&f)?;
                if !r.is_zero() {
                    break;
                }
                e += 1;
                work = q;
                if work.degree().is_none() || work.degree() == Some(0) {
                    break;
                }
            }
            if e == 0 {
                return Err(MathError::NumericalError(
                    "fp_factor returned a factor that does not divide the input".to_string(),
                ));
            }
            factors.push((f, e));
        }

        certify(poly, lc, factors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::Ring;

    fn qpoly(coeffs: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| Rational::from_i64(c)).collect())
    }

    fn fpoly<const P: u64>(coeffs: &[i64]) -> UnivariatePolynomial<GFp<P>> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| GFp::<P>::new(c)).collect())
    }

    #[test]
    fn test_factor_q_x2_minus_1() {
        // sympy: x^2-1 = (x-1)(x+1) over Q.
        let (unit, factors) = Rational::factor_poly(&qpoly(&[-1, 0, 1])).unwrap();
        assert!(unit.is_one());
        assert_eq!(factors.len(), 2);
        assert!(factors.iter().all(|(_, e)| *e == 1));
        assert!(factors.iter().any(|(f, _)| *f == qpoly(&[-1, 1])));
        assert!(factors.iter().any(|(f, _)| *f == qpoly(&[1, 1])));
    }

    #[test]
    fn test_factor_q_with_multiplicity_and_unit() {
        // 2(x-1)^2 (x^2+2): sympy-verified irreducible quadratic x^2+2 over Q.
        let p = qpoly(&[2]) * qpoly(&[-1, 1]) * qpoly(&[-1, 1]) * qpoly(&[2, 0, 1]);
        let (unit, factors) = Rational::factor_poly(&p).unwrap();
        assert_eq!(unit, Rational::from_i64(2));
        assert_eq!(factors.len(), 2);
        assert!(factors
            .iter()
            .any(|(f, e)| *f == qpoly(&[-1, 1]) && *e == 2));
        assert!(factors
            .iter()
            .any(|(f, e)| *f == qpoly(&[2, 0, 1]) && *e == 1));
    }

    #[test]
    fn test_factor_q_rational_coefficients() {
        // (1/2)x + 1/2 = (1/2)(x+1): unit 1/2, factor x+1.
        let p = UnivariatePolynomial::new(vec![
            Rational::new(1, 2).unwrap(),
            Rational::new(1, 2).unwrap(),
        ]);
        let (unit, factors) = Rational::factor_poly(&p).unwrap();
        assert_eq!(unit, Rational::new(1, 2).unwrap());
        assert_eq!(factors, vec![(qpoly(&[1, 1]), 1)]);
    }

    #[test]
    fn test_factor_gf5_x2_plus_1() {
        // sympy: x^2+1 = (x+2)(x+3) over GF(5).
        let (unit, factors) = GFp::<5>::factor_poly(&fpoly::<5>(&[1, 0, 1])).unwrap();
        assert!(unit.is_one());
        assert_eq!(factors.len(), 2);
        assert!(factors.iter().any(|(f, _)| *f == fpoly::<5>(&[2, 1])));
        assert!(factors.iter().any(|(f, _)| *f == fpoly::<5>(&[3, 1])));
    }

    #[test]
    fn test_factor_gf5_irreducible_quintic() {
        // sympy: x^5 - x + 1 is irreducible over GF(5) (Artin-Schreier).
        let (unit, factors) = GFp::<5>::factor_poly(&fpoly::<5>(&[1, -1, 0, 0, 0, 1])).unwrap();
        assert!(unit.is_one());
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1);
        assert_eq!(factors[0].0.degree(), Some(5));
    }

    #[test]
    fn test_factor_gf5_multiplicity() {
        // 3(x+1)^3 over GF(5): unit 3, factor (x+1) with e=3.
        let p = fpoly::<5>(&[3]) * fpoly::<5>(&[1, 1]) * fpoly::<5>(&[1, 1]) * fpoly::<5>(&[1, 1]);
        let (unit, factors) = GFp::<5>::factor_poly(&p).unwrap();
        assert_eq!(unit, GFp::<5>::new(3));
        assert_eq!(factors, vec![(fpoly::<5>(&[1, 1]), 3)]);
    }

    #[test]
    fn test_factor_zero_is_err() {
        assert!(Rational::factor_poly(&UnivariatePolynomial::zero()).is_err());
        assert!(GFp::<5>::factor_poly(&UnivariatePolynomial::zero()).is_err());
    }

    #[test]
    fn test_factor_constant() {
        let (unit, factors) = Rational::factor_poly(&qpoly(&[7])).unwrap();
        assert_eq!(unit, Rational::from_i64(7));
        assert!(factors.is_empty());
    }

    #[test]
    fn test_factor_composite_modulus_is_err() {
        assert!(GFp::<6>::factor_poly(&fpoly::<6>(&[1, 1])).is_err());
    }
}
