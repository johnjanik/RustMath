//! Multivariate polynomial *systems* over the integers — the shared solve object.
//!
//! Ported from `dessin_engine/src/mpoly.rs` (`MPolySystem`, `MPoly`) in
//! `/home/john/inverse_galois/M23/dessin_engine`. The private `Rat`/`BigInt`
//! wrappers of the reference implementation are replaced by RustMath's foundation
//! types (`rustmath_integers::Integer`, `rustmath_rationals::Rational`) and the
//! systems are built on this crate's existing
//! [`MultivariatePolynomial`](crate::multivariate::MultivariatePolynomial)
//! representation rather than dessin_engine's private `MPoly`.
//!
//! A Belyi system (once cleared to integer coefficients) is stored here as a list
//! of equations `f_1 = … = f_eqs = 0` in `nvars` variables. The downstream
//! numerical / Belyi layers (Wave 2/3) consume this type and need three things:
//!
//! * [`PolySystem::evaluate_mod`] and [`PolySystem::jacobian_mod`] — value and
//!   Jacobian reduced modulo `m` (for the Newton lift over `m = p^k`);
//! * [`PolySystem::evaluate`] — exact rational evaluation (to certify a recognized
//!   solution by back-substitution);
//! * [`PolySystem::is_exact_solution`] — the exact zero check.
//!
//! This is the stable interface contract for Waves 2 and 3.

use crate::multivariate::{Monomial, MultivariatePolynomial};
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::BTreeMap;

/// Build an integer-coefficient multivariate polynomial from a list of
/// `(dense exponent vector, coefficient)` terms.
///
/// Each exponent vector must have length `nvars`; `exps[j]` is the power of
/// variable `x_j`. This mirrors the ergonomic `MPoly::new` constructor of the
/// reference `dessin_engine` and is the convenient way to hand-build systems
/// (and to emit them from an encoder).
///
/// # Panics
/// Panics if any exponent vector does not have length `nvars`.
pub fn poly_from_terms(nvars: usize, terms: &[(Vec<u32>, i64)]) -> MultivariatePolynomial<Integer> {
    let mut poly = MultivariatePolynomial::zero();
    for (exps, coeff) in terms {
        assert_eq!(exps.len(), nvars, "exponent vector arity");
        let mut map = BTreeMap::new();
        for (j, &e) in exps.iter().enumerate() {
            if e > 0 {
                map.insert(j, e);
            }
        }
        poly.add_term(Monomial::from_exponents(map), Integer::from(*coeff));
    }
    poly
}

/// A square-ish system `f_1 = … = f_eqs = 0` in `nvars` variables with integer
/// coefficients.
///
/// The equations are stored as
/// [`MultivariatePolynomial<Integer>`](crate::multivariate::MultivariatePolynomial),
/// reusing this crate's multivariate arithmetic (including
/// `partial_derivative`, used for the Jacobian).
#[derive(Debug, Clone)]
pub struct PolySystem {
    nvars: usize,
    polys: Vec<MultivariatePolynomial<Integer>>,
}

impl PolySystem {
    /// Create a system in `nvars` variables from a list of integer-coefficient
    /// polynomials.
    ///
    /// No arity assertion is imposed on the polynomials themselves (a
    /// `MultivariatePolynomial` carries no fixed arity); `nvars` records the
    /// ambient variable count and every evaluation point must have at least
    /// `nvars` entries.
    pub fn new(nvars: usize, polys: Vec<MultivariatePolynomial<Integer>>) -> Self {
        Self { nvars, polys }
    }

    /// Build a system directly from dense `(exponent vector, coefficient)` term
    /// lists — one list per equation. Convenience wrapper over
    /// [`poly_from_terms`].
    pub fn from_terms(nvars: usize, equations: &[Vec<(Vec<u32>, i64)>]) -> Self {
        let polys = equations
            .iter()
            .map(|terms| poly_from_terms(nvars, terms))
            .collect();
        Self { nvars, polys }
    }

    /// Number of ambient variables.
    pub fn num_variables(&self) -> usize {
        self.nvars
    }

    /// Number of equations.
    pub fn num_equations(&self) -> usize {
        self.polys.len()
    }

    /// Access the underlying equations.
    pub fn polynomials(&self) -> &[MultivariatePolynomial<Integer>] {
        &self.polys
    }

    // --- modular evaluation (Newton lift over m = p^k) -----------------------

    /// `f(point) mod m` for a single equation, reduced to `[0, m)`.
    ///
    /// The reduction is applied at every step so intermediate values never grow,
    /// mirroring the reference implementation's `eval_mod`.
    fn eval_poly_mod(
        poly: &MultivariatePolynomial<Integer>,
        point: &[Integer],
        m: &Integer,
    ) -> Integer {
        let mut acc = Integer::zero();
        for (mono, coeff) in poly.terms() {
            let mut term = coeff.modulo(m);
            for (&var, &exp) in mono.iter_exponents() {
                let pw = point[var]
                    .mod_pow(&Integer::from(exp as u64), m)
                    .expect("modulus must be non-zero")
                    .modulo(m);
                term = (term * pw).modulo(m);
            }
            acc = (acc + term).modulo(m);
        }
        acc
    }

    /// `F(point) mod m` — the residual vector reduced modulo `m` (each entry in
    /// `[0, m)`).
    ///
    /// # Panics
    /// Panics if `m` is zero or `point` has fewer than `nvars` entries.
    pub fn evaluate_mod(&self, point: &[Integer], m: &Integer) -> Vec<Integer> {
        assert!(!m.is_zero(), "modulus must be non-zero");
        assert!(point.len() >= self.nvars, "point arity too small");
        self.polys
            .iter()
            .map(|p| Self::eval_poly_mod(p, point, m))
            .collect()
    }

    /// Jacobian `J[i][k] = ∂f_i/∂x_k (point) mod m`, each entry in `[0, m)`.
    ///
    /// Partial derivatives are taken symbolically via the multivariate module's
    /// [`partial_derivative`](crate::multivariate::MultivariatePolynomial::partial_derivative),
    /// then evaluated modulo `m`.
    ///
    /// # Panics
    /// Panics if `m` is zero or `point` has fewer than `nvars` entries.
    pub fn jacobian_mod(&self, point: &[Integer], m: &Integer) -> Vec<Vec<Integer>> {
        assert!(!m.is_zero(), "modulus must be non-zero");
        assert!(point.len() >= self.nvars, "point arity too small");
        self.polys
            .iter()
            .map(|p| {
                (0..self.nvars)
                    .map(|k| Self::eval_poly_mod(&p.partial_derivative(k), point, m))
                    .collect()
            })
            .collect()
    }

    // --- exact rational evaluation (certification) ---------------------------

    /// Exact rational value of a single equation at `point`.
    fn eval_poly_rat(poly: &MultivariatePolynomial<Integer>, point: &[Rational]) -> Rational {
        let mut acc = Rational::from_i64(0);
        for (mono, coeff) in poly.terms() {
            let mut term = Rational::from_integer(coeff.clone());
            for (&var, &exp) in mono.iter_exponents() {
                term = term * Ring::pow(&point[var], exp);
            }
            acc = acc + term;
        }
        acc
    }

    /// Exact rational residual `F(point)` (the zero vector iff `point` is an
    /// exact solution).
    ///
    /// # Panics
    /// Panics if `point` has fewer than `nvars` entries.
    pub fn evaluate(&self, point: &[Rational]) -> Vec<Rational> {
        assert!(point.len() >= self.nvars, "point arity too small");
        self.polys
            .iter()
            .map(|p| Self::eval_poly_rat(p, point))
            .collect()
    }

    /// Exact zero check: `true` iff every equation vanishes exactly at `point`.
    pub fn is_exact_solution(&self, point: &[Rational]) -> bool {
        self.evaluate(point).iter().all(|r| r.is_zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ported from `dessin_engine/src/mpoly.rs::tests::eval_and_jacobian`.
    #[test]
    fn eval_and_jacobian() {
        // f = x^2 + y^2 - 1 (one eqn, two vars)
        let sys = PolySystem::from_terms(
            2,
            &[vec![(vec![2, 0], 1), (vec![0, 2], 1), (vec![0, 0], -1)]],
        );
        let m = Integer::from(1000);
        let pt = vec![Integer::from(3), Integer::from(4)];
        // 9 + 16 - 1 = 24
        assert_eq!(sys.evaluate_mod(&pt, &m), vec![Integer::from(24)]);
        // ∂/∂x = 2x = 6, ∂/∂y = 2y = 8
        assert_eq!(
            sys.jacobian_mod(&pt, &m),
            vec![vec![Integer::from(6), Integer::from(8)]]
        );
    }

    /// Ported from `dessin_engine/src/mpoly.rs::tests::exact_rational_solution`.
    #[test]
    fn exact_rational_solution() {
        // x^2 - 2 (one var); (3/2) is not a root, but the residual is exact.
        let sys = PolySystem::from_terms(1, &[vec![(vec![2], 1), (vec![0], -2)]]);
        let three_halves = Rational::new(3, 2).unwrap();
        // (3/2)^2 - 2 = 1/4
        assert_eq!(
            sys.evaluate(&[three_halves.clone()]),
            vec![Rational::new(1, 4).unwrap()]
        );
        assert!(!sys.is_exact_solution(&[three_halves]));
    }

    /// A hand example with a *known exact solution* verified against known
    /// values, plus a mod-m Jacobian cross-check.
    #[test]
    fn known_solution_and_mod_jacobian() {
        // System (2 vars, 2 eqns):
        //   f1 = x^2 + y^2 - 25
        //   f2 = x - y - 1
        // (x, y) = (4, 3) is an exact solution: 16+9-25=0, 4-3-1=0.
        let sys = PolySystem::from_terms(
            2,
            &[
                vec![(vec![2, 0], 1), (vec![0, 2], 1), (vec![0, 0], -25)],
                vec![(vec![1, 0], 1), (vec![0, 1], -1), (vec![0, 0], -1)],
            ],
        );

        // Exact solution check.
        let sol = vec![Rational::from_i64(4), Rational::from_i64(3)];
        assert!(sys.is_exact_solution(&sol));
        assert_eq!(
            sys.evaluate(&sol),
            vec![Rational::from_i64(0), Rational::from_i64(0)]
        );

        // A non-solution has a nonzero residual.
        let bad = vec![Rational::from_i64(0), Rational::from_i64(0)];
        assert!(!sys.is_exact_solution(&bad));
        assert_eq!(
            sys.evaluate(&bad),
            vec![Rational::from_i64(-25), Rational::from_i64(-1)]
        );

        // Jacobian:  [[2x, 2y], [1, -1]]  at (4,3) mod 7  ->  [[8,6],[1,-1]]
        //            reduced to [0,7): [[1, 6], [1, 6]]   (since -1 ≡ 6).
        let m = Integer::from(7);
        let pt = vec![Integer::from(4), Integer::from(3)];
        assert_eq!(
            sys.jacobian_mod(&pt, &m),
            vec![
                vec![Integer::from(1), Integer::from(6)],
                vec![Integer::from(1), Integer::from(6)],
            ]
        );

        // Residual mod 7 at the exact solution is the zero vector.
        assert_eq!(
            sys.evaluate_mod(&pt, &m),
            vec![Integer::from(0), Integer::from(0)]
        );
    }

    /// Negative coefficients / points reduce into `[0, m)` (matching the
    /// reference `modpos` behaviour).
    #[test]
    fn modular_reduction_is_nonnegative() {
        // f = x - 10, evaluated at x = 3 mod 7:  3 - 10 = -7 ≡ 0.
        let sys = PolySystem::from_terms(1, &[vec![(vec![1], 1), (vec![0], -10)]]);
        let m = Integer::from(7);
        assert_eq!(sys.evaluate_mod(&[Integer::from(3)], &m), vec![Integer::from(0)]);

        // g = x^3 at x = -2 mod 5: (-2)^3 = -8 ≡ 2 (mod 5), non-negative.
        let g = PolySystem::from_terms(1, &[vec![(vec![3], 1)]]);
        assert_eq!(
            g.evaluate_mod(&[Integer::from(-2)], &Integer::from(5)),
            vec![Integer::from(2)]
        );
    }
}
