//! Certified arbitrary-precision eigenvalue approximations: the exact
//! (division-free Berkowitz) characteristic polynomial handed to the
//! certified Aberth–Ehrlich root finder in `rustmath-complex`.
//!
//! This closes the seam documented in [`crate::charpoly_exact`]: that module
//! deliberately computes only the *exact* rational eigenvalues
//! ([`Matrix::rational_eigenvalues`]) and exposes the exact characteristic
//! polynomial for a root isolator to consume. [`Matrix::eigenvalues_approx`]
//! is that consumer.
//!
//! # What is returned
//!
//! [`Matrix::eigenvalues_approx`] returns the full [`PolynomialRoots`] output
//! of [`aberth_roots`] applied to `charpoly(A) = det(xI − A)` at the
//! requested precision (bits):
//!
//! * `roots` — all `n` approximate eigenvalues as
//!   [`BigComplex`](rustmath_complex::bigcomplex::BigComplex) values, each carrying a
//!   **certified forward error bound**
//!   ([`RootEstimate::error_bound`](rustmath_complex::RootEstimate)): a true
//!   root of the characteristic polynomial lies within that distance. For
//!   exact coefficient rings (`Integer`, `Rational`) the characteristic
//!   polynomial is computed *exactly* by
//!   [`charpoly_berkowitz`](crate::charpoly_exact::charpoly_berkowitz), so — up to
//!   the one documented rounding of the coefficients to working precision
//!   inside `aberth_roots` — those certificates apply to the true spectrum of
//!   the matrix.
//! * `clusters` — approximations agreeing to roughly half the requested
//!   precision, grouped. Per the honest Aberth semantics, **cluster size is a
//!   heuristic multiplicity, not a certified one** (except the exact origin
//!   cluster, whose multiplicity is exact from vanishing low coefficients —
//!   i.e. the exact algebraic multiplicity of the eigenvalue `0`). At an
//!   `m`-fold eigenvalue the `m` approximations agree only to `≈ wp/m` bits
//!   and their individual `error_bound`s honestly report that larger
//!   uncertainty.
//! * `converged` — whether every root hit a stopping criterion before the
//!   sweep cap; on `false` the estimates and their certified bounds are still
//!   returned and still valid.
//!
//! # Honesty caveats
//!
//! * Eigenvalues are **approximations with certificates**, never exact values.
//!   Callers wanting exact rational eigenvalues should keep using
//!   [`Matrix::rational_eigenvalues`]; the two views are consistent (a
//!   rational eigenvalue shows up here as an approximation within its
//!   certified bound).
//! * For floating coefficient rings (`BigFloat`, `BigComplex`) the Berkowitz
//!   recursion itself rounds at the entries' precision, so the certificates
//!   are relative to the *computed* characteristic polynomial, not to the
//!   input matrix. Use exact entries when the certificate must reach the
//!   matrix itself.

use crate::Matrix;
use rustmath_complex::{aberth_roots, PolynomialRoots, RootCoefficient};
use rustmath_core::{CommutativeRing, Result};

impl<R: CommutativeRing + RootCoefficient> Matrix<R> {
    /// Certified approximations of **all** `n` complex eigenvalues of a
    /// square matrix, to `precision` bits.
    ///
    /// Pipeline: exact division-free characteristic polynomial
    /// ([`charpoly_berkowitz`](crate::charpoly_exact::charpoly_berkowitz)) →
    /// certified Aberth–Ehrlich root finding ([`aberth_roots`]). See the
    /// [module docs](self) for exactly what is and is not certified,
    /// including the honest multiple-eigenvalue (cluster) semantics.
    ///
    /// Errors on a non-square matrix. A `0×0` matrix has characteristic
    /// polynomial `1` and returns an empty root list.
    pub fn eigenvalues_approx(&self, precision: u64) -> Result<PolynomialRoots> {
        let cp = self.charpoly_exact()?;
        // charpoly is monic of degree n, coefficients in ascending order —
        // exactly the input contract of aberth_roots, which therefore
        // returns all n eigenvalue approximations.
        aberth_roots(cp.coefficients(), precision)
    }
}

#[cfg(test)]
mod tests {
    use crate::companion::companion_matrix;
    use crate::Matrix;
    use rustmath_complex::bigcomplex::BigComplex;
    use rustmath_complex::PolynomialRoots;
    use rustmath_core::analytic::{ComplexField, RealField};
    use rustmath_core::Ring;
    use rustmath_integers::Integer;
    use rustmath_polynomials::UnivariatePolynomial;
    use rustmath_rationals::Rational;
    use rustmath_reals::bigfloat::BigFloat;

    fn q(n: i64) -> Rational {
        Rational::from_integer(n)
    }
    fn z(n: i64) -> Integer {
        Integer::from(n)
    }
    fn bfi(n: i64, prec: u64) -> BigFloat {
        BigFloat::from_integer(&Integer::from(n), prec)
    }
    fn bci(re: i64, im: i64, prec: u64) -> BigComplex {
        BigComplex::new(bfi(re, prec), bfi(im, prec))
    }

    /// Exactly 10^-50 as a 300-bit float (2^-200 ≈ 6e-61, so 1e-50 is a
    /// meaningful target at 200 bits).
    fn tol_1e50() -> BigFloat {
        let ten50 = Integer::from(10).pow(50);
        BigFloat::from_rational(&Rational::new(Integer::from(1), ten50).unwrap(), 300)
    }

    fn min_dist(result: &PolynomialRoots, target: &BigComplex) -> BigFloat {
        result
            .roots
            .iter()
            .map(|r| (r.value.clone() - target.clone()).abs())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    #[test]
    fn companion_of_x2p1_xm2_gives_i_minus_i_2() {
        // (x^2+1)(x-2) = x^3 - 2x^2 + x - 2, ascending coeffs [-2, 1, -2, 1]
        // (sympy-verified); mpmath polyroots at 300 bits: {2, i, -i}.
        let p = UnivariatePolynomial::new(vec![q(-2), q(1), q(-2), q(1)]);
        let a = companion_matrix(&p).unwrap();
        // The companion matrix's exact charpoly is the input polynomial.
        assert_eq!(a.charpoly_exact().unwrap().coefficients(), p.coefficients());

        let r = a.eigenvalues_approx(200).unwrap();
        assert_eq!(r.roots.len(), 3, "3x3 matrix must yield 3 eigenvalues");
        assert!(r.converged);
        assert_eq!(r.precision, 200);
        let tol = tol_1e50();
        for expected in [bci(2, 0, 300), bci(0, 1, 300), bci(0, -1, 300)] {
            let d = min_dist(&r, &expected);
            assert!(
                d < tol,
                "eigenvalue {expected} missed by {} (need < 1e-50 at 200 bits)",
                d.to_decimal_string(5)
            );
        }
        // Simple, well-separated spectrum: every estimate certified and
        // isolated, three singleton clusters.
        for est in &r.roots {
            let e = est.error_bound.as_ref().expect("simple eigenvalues carry a bound");
            assert!(e < &tol, "certified bound too large: {}", e.to_decimal_string(5));
            assert!(est.isolated);
        }
        assert_eq!(r.clusters.len(), 3);
        assert!(r.clusters.iter().all(|c| c.multiplicity == 1));
    }

    #[test]
    fn symmetric_integer_matrix_irrational_spectrum() {
        // [[2,1],[1,3]]: charpoly x^2 - 5x + 5, eigenvalues (5 ± √5)/2
        // (sympy + mpmath verified: 3.61803398874989484820458683436563811772…
        // and 1.38196601125010515179541316563436188227…).
        let a = Matrix::from_vec(2, 2, vec![z(2), z(1), z(1), z(3)]).unwrap();
        assert_eq!(
            a.charpoly_exact().unwrap().coefficients(),
            &[z(5), z(-5), z(1)]
        );

        let r = a.eigenvalues_approx(200).unwrap();
        assert_eq!(r.roots.len(), 2);
        assert!(r.converged);
        let p = 300;
        let sqrt5 = bfi(5, p).sqrt();
        let two = bfi(2, p);
        let lam_plus = BigComplex::new((bfi(5, p) + sqrt5.clone()) / two.clone(), bfi(0, p));
        let lam_minus = BigComplex::new((bfi(5, p) - sqrt5) / two, bfi(0, p));
        let tol = tol_1e50();
        for expected in [lam_plus, lam_minus] {
            let d = min_dist(&r, &expected);
            assert!(
                d < tol,
                "eigenvalue {expected} missed by {}",
                d.to_decimal_string(5)
            );
        }
        for est in &r.roots {
            assert!(est.error_bound.as_ref().unwrap() < &tol);
            assert!(est.isolated);
        }
    }

    #[test]
    fn rational_spectrum_exact_path_intact_and_consistent() {
        // diag(2/3, 5): the exact path must still return exact rationals,
        // and the certified approximations must agree with them.
        let a = Matrix::from_vec(
            2,
            2,
            vec![Rational::new(2, 3).unwrap(), q(0), q(0), q(5)],
        )
        .unwrap();

        // Exact path: exact values, not approximations.
        let mut evals = a.rational_eigenvalues().unwrap();
        evals.sort_by(|x, y| x.0.to_f64().unwrap().partial_cmp(&y.0.to_f64().unwrap()).unwrap());
        assert_eq!(evals, vec![(Rational::new(2, 3).unwrap(), 1), (q(5), 1)]);

        // Approximate path agrees within the certified tolerance.
        let r = a.eigenvalues_approx(200).unwrap();
        assert!(r.converged);
        let tol = tol_1e50();
        let two_thirds = BigComplex::from_rational(&Rational::new(2, 3).unwrap(), 300);
        assert!(min_dist(&r, &two_thirds) < tol);
        assert!(min_dist(&r, &bci(5, 0, 300)) < tol);
    }

    #[test]
    fn repeated_eigenvalue_reported_as_honest_cluster() {
        // Shear [[1,1],[0,1]]: charpoly (x-1)^2, double eigenvalue 1. Per the
        // documented Aberth semantics the two approximations stagnate in a
        // tiny cluster around 1 (agreement ≈ half the working precision) and
        // the cluster reports heuristic multiplicity 2.
        let a = Matrix::from_vec(2, 2, vec![z(1), z(1), z(0), z(1)]).unwrap();
        let r = a.eigenvalues_approx(200).unwrap();
        assert_eq!(r.roots.len(), 2);
        assert_eq!(r.clusters.len(), 1, "double eigenvalue must form one cluster");
        assert_eq!(r.clusters[0].multiplicity, 2);
        // Members and center agree with 1 to well beyond half precision.
        let tol30 = BigFloat::from_rational(
            &Rational::new(Integer::from(1), Integer::from(10).pow(30)).unwrap(),
            300,
        );
        assert!((r.clusters[0].center.clone() - bci(1, 0, 300)).abs() < tol30);
        for est in &r.roots {
            assert!((est.value.clone() - bci(1, 0, 300)).abs() < tol30);
        }
    }

    #[test]
    fn zero_eigenvalue_multiplicity_is_exact() {
        // Nilpotent [[0,1],[0,0]]: charpoly x^2; the origin multiplicity is
        // exact (from vanishing low coefficients), not heuristic.
        let a = Matrix::from_vec(2, 2, vec![z(0), z(1), z(0), z(0)]).unwrap();
        let r = a.eigenvalues_approx(200).unwrap();
        assert_eq!(r.zero_multiplicity, 2);
        assert_eq!(r.roots.len(), 2);
        assert!(r.roots.iter().all(|e| e.value.is_zero()));
    }

    #[test]
    fn non_square_is_an_error() {
        let a = Matrix::from_vec(1, 2, vec![z(1), z(2)]).unwrap();
        assert!(a.eigenvalues_approx(64).is_err());
    }
}
