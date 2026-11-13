//! MAGMA Handbook ch. 23 (Univariate Polynomial Rings), `IsIrreducible(f)` (§23.4.4 ring-element
//! predicates; §23.8.1 "true iff f is irreducible over R", by reduction to `Factorization`);
//! digest `rustmath-lava/docs/Chapter_23_Univariate_Polynomial_Rings.md`. Implementation:
//! `rustmath_polynomials::is_irreducible` (Z, Q, i64, i32 and a generic fallback) and
//! `rustmath_polynomials::berlekamp::is_irreducible` (GF(p), Berlekamp's criterion: Berlekamp
//! 1967; Knuth TAOCP vol. 2 §4.6.2).
//!
//! Reference values: PARI/GP 2.15.4 `polisirreducible`, which decides irreducibility over the
//! fraction field (constants give 0, the content is ignored: `polisirreducible(2*x+2) = 1`).
//! `is_irreducible` answers irreducibility as an element of R[x], where over Z the content
//! counts; its expected value in the Z table is the column `ring`, computed by gp as
//!
//! ```text
//! ring(f) = if(f == 0, 0, if(poldegree(f) == 0, isprime(abs(polcoef(f,0))),
//!                            content(f) == 1 && polisirreducible(f)))
//! ```
//!
//! (Gauss' lemma: a primitive polynomial is irreducible in Z[x] iff it is irreducible in Q[x];
//! a non-constant polynomial with content c != 1 is c times a non-unit; a constant is
//! irreducible in Z[x] iff it is a prime of Z). `is_irreducible_over_integers` follows PARI's
//! convention and is checked against the `pari` column. Rows:
//!
//! ```text
//! Z:     [Vecrev(f), polisirreducible(f), ring(f)]     45 named + 120 random (setrand(23))
//! Q:     [[numerator, denominator] of Vecrev(f), polisirreducible(f)]
//!                                                      23 named + 50 random (setrand(41))
//! GF(p): [Vecrev(f), p, polisirreducible(Mod(1,p)*f)]  f not reduced mod p; 64 named,
//!        27 ffinit(p,n) irreducibles and products, 96 random (setrand(57))
//! ```

use rustmath_core::{
    CommutativeRing, EuclideanDomain, Field, IntegralDomain, MathError, Result, Ring,
};
use rustmath_integers::Integer;
use rustmath_polynomials::{
    berlekamp, is_irreducible, is_irreducible_over_integers, UnivariatePolynomial,
};
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

fn ints(c: &[i64]) -> Vec<Integer> {
    c.iter().map(|&a| Integer::from(a)).collect()
}

fn z(c: &[i64]) -> UnivariatePolynomial<Integer> {
    UnivariatePolynomial::new(ints(c))
}

fn q(c: &[(i64, i64)]) -> UnivariatePolynomial<Rational> {
    UnivariatePolynomial::new(
        c.iter()
            .map(|&(n, d)| Rational::new(Integer::from(n), Integer::from(d)).unwrap())
            .collect(),
    )
}

/// `UnivariatePolynomial::new` wants at least one coefficient.
fn nonempty<T: Copy>(c: &[T], zero: T) -> Vec<T> {
    if c.is_empty() {
        vec![zero]
    } else {
        c.to_vec()
    }
}

/// `(f, polisirreducible(f), ring(f))` over Z (see the header).
#[rustfmt::skip]
const Z_ROWS: &[(&[i64], u8, u8)] = &[
    // named Z rows
    (&[-1, 0, 1], 0, 0),
    (&[1, 0, 1], 1, 1),
    (&[-2, 0, 0, 1], 1, 1),
    (&[1, -2, 1], 0, 0),
    (&[-1, 1, -1, 1], 0, 0),
    (&[-1, 0, 0, 1], 0, 0),
    (&[1, 1, 0, 1], 1, 1),
    (&[-1, -1, 2], 0, 0),
    (&[1, 5, 6], 0, 0),
    (&[1, 0, 4], 1, 1),
    (&[2, 2], 1, 0),
    (&[2, 0, 2], 1, 0),
    (&[-6, 0, 6], 0, 0),
    (&[-6, 0, 0, 3], 1, 0),
    (&[-1, -1], 1, 1),
    (&[-1, 0, -1], 1, 1),
    (&[0, 3], 1, 0),
    (&[0, 1], 1, 1),
    (&[0, -1], 1, 1),
    (&[0, 0, 1], 0, 0),
    (&[0, -1, 0, 1], 0, 0),
    (&[-4, 0, 9], 0, 0),
    (&[4, 0, 9], 1, 1),
    (&[1, 1, 1], 1, 1),
    (&[-1, -3, 0, 1], 1, 1),
    (&[1, 3, 3, 2], 0, 0),
    (&[4, 0, 0, 0, 1], 0, 0),
    (&[1, 0, 2, 0, 1], 0, 0),
    (&[6, 0, -5, 0, 1], 0, 0),
    (&[1, 0, -10, 0, 1], 1, 1),
    (&[1, 1, 1, 1, 1, 1, 1], 1, 1),
    (&[-1, -1, 0, 0, 0, 1], 1, 1),
    (&[1, -1, 0, 1, -1, 1, 0, -1, 1], 1, 1),
    (&[6, 0, 0, -5, 0, 0, 1], 0, 0),
    (&[1, 0, -4, 0, 4], 0, 0),
    (&[3, 0, 0, 0, 0, 5], 1, 1),
    (&[], 0, 0),
    (&[1], 0, 0),
    (&[-1], 0, 0),
    (&[2], 0, 1),
    (&[-7], 0, 1),
    (&[4], 0, 0),
    (&[6], 0, 0),
    (&[997], 0, 1),
    (&[-1001], 0, 0),
    // random Z rows
    (&[-3, 5, 7, -2, 1, -2, -7], 1, 1),
    (&[1, 5], 1, 1),
    (&[-3, 5, -7], 1, 1),
    (&[2], 0, 1),
    (&[1, -6, -4], 1, 1),
    (&[-5, -1, -5, -4, 1, -8, -6], 1, 1),
    (&[-3, 7, 4], 1, 1),
    (&[2, 9, 4, -5, 1], 1, 1),
    (&[8, 7, -8, 8, 8, 3, -2], 1, 1),
    (&[-7, 5], 1, 1),
    (&[6, -6], 1, 0),
    (&[2, -9, -8], 1, 1),
    (&[-3, -6], 1, 0),
    (&[-7, 7, -3], 1, 1),
    (&[3, 4, -8, -5, 3, -3], 1, 1),
    (&[-6, 1, -8, 8, 9, 5], 1, 1),
    (&[5], 0, 1),
    (&[-8, -9, 0, -6, 7, 1, 4], 1, 1),
    (&[-6, -6, 7, -9, -3], 1, 1),
    (&[-9, -8, 5, -3, -1, -6, -2, -4], 1, 1),
    (&[-2], 0, 1),
    (&[-4], 0, 0),
    (&[-2, 4, -3, 3], 1, 1),
    (&[-4], 0, 0),
    (&[3, -2, 2, -6, -6, -9, -1, 2], 1, 1),
    (&[9, 3, 4, 2], 1, 1),
    (&[2, -5], 1, 1),
    (&[9, 5, 8, -9, -6, 0, 9, 8], 1, 1),
    (&[-2, -3, 0, -6, 6, -3, -5, 5], 1, 1),
    (&[9, -4, 2, 0, -7, -5], 1, 1),
    (&[-7, 4], 1, 1),
    (&[9, -7, 3], 1, 1),
    (&[3], 0, 1),
    (&[-1, 5, 5, 8, -6, 6, 6, 2], 1, 1),
    (&[2, 8, 4, -5, 8, -3, -3, -5], 1, 1),
    (&[8, 3], 1, 1),
    (&[7, -4, -4, -4, -6, 9, 3], 1, 1),
    (&[-4, -8, 9, -7, -9, -4], 1, 1),
    (&[-1, 5, 6, 7], 1, 1),
    (&[9, 1, -1], 1, 1),
    (&[9, 4, 8], 1, 1),
    (&[-2, -8, -6, 8, -7], 1, 1),
    (&[3, -4, -8, -4, 6], 1, 1),
    (&[6, 9, -1, 5, -9, -4, -6], 0, 0),
    (&[-6, -5, 6, 5, -3], 0, 0),
    (&[4, -2, 5, -2, 1], 0, 0),
    (&[2, 8, -7, 0, -7, 3, -3, -9], 1, 1),
    (&[2, 1, -4, 1], 0, 0),
    (&[-3, -7, 9, -5, -2, 8], 0, 0),
    (&[-6, 9, -9, -9, -5, -7, 4, 9], 1, 1),
    (&[-3, -9, -9, -6, 4, 0, -9], 1, 1),
    (&[7, 1, -8, 5, 1, 0, 9], 1, 1),
    (&[5, 3, -5, -9], 1, 1),
    (&[3, 8, -1, -1, 8, -1, -7, 7], 1, 1),
    (&[-9], 0, 0),
    (&[7], 0, 1),
    (&[-3, -4, -8, -5, -6, -2], 1, 1),
    (&[4, 5, 3, 3], 1, 1),
    (&[2], 0, 1),
    (&[9, -7, 4, -9, 1, -3, 7, 9], 1, 1),
    (&[-15, 13, -13, -10, 1, -6], 0, 0),
    (&[0, -10, 26, -5, -11, 7, 2], 0, 0),
    (&[2, -5, 3, 3], 0, 0),
    (&[4, -12, 6, -8, 10], 0, 0),
    (&[-6, -9, -17, -25, -23, -20], 0, 0),
    (&[-20, 33, -31, 37, -10, 12], 0, 0),
    (&[12, -5, 5, -31, -24, -20], 0, 0),
    (&[16, 4, -14, -23, 7, 10], 0, 0),
    (&[4, -16, 15], 0, 0),
    (&[12, 11, -4, 3, 14, 12], 0, 0),
    (&[-12, 10, -2, -19, 19, 1, -6], 0, 0),
    (&[-10, 15, 5, -20], 0, 0),
    (&[0, 12, 19, 24, 9, 1], 0, 0),
    (&[1, -9, 19, 0, 25], 0, 0),
    (&[0, 4, -12, 11, -6], 0, 0),
    (&[15, -22, 8], 0, 0),
    (&[10, 2, 15, 6, 4, 8], 0, 0),
    (&[-8, -8, 14, -4], 0, 0),
    (&[-15, -5, 19, 10, 14, -20], 0, 0),
    (&[0, -6, -3, -7, 11, -3], 0, 0),
    (&[6, -10, 2, 4, -8], 0, 0),
    (&[0, -4, -10, 22, -24, 8], 0, 0),
    (&[-9, 18, 4, -3], 0, 0),
    (&[-12, 29, -23, -2, 6], 0, 0),
    (&[-10, -9, -27, 0, 4], 0, 0),
    (&[10, 21, -12, -5], 0, 0),
    (&[1, 4, 2, 9, 4], 0, 0),
    (&[4, -19, -8, 28, 16], 0, 0),
    (&[-12, -28, -31, -20], 0, 0),
    (&[8, 16, -14, -18, -20], 0, 0),
    (&[-1, 4, -8, 13, -19, 5], 0, 0),
    (&[-8, -14, -6, 10, 8, 0, -2], 0, 0),
    (&[-12, 14, 25, -8, 12, 9], 0, 0),
    (&[0, 4, 25, 25], 0, 0),
    (&[0, -1, -2, -6, -5], 0, 0),
    (&[10, -18, -8, 8], 0, 0),
    (&[20, -22, 31, 5, -12], 0, 0),
    (&[0, 4, 17, 12, 2], 0, 0),
    (&[0, -10, 35, -25], 0, 0),
    (&[0, 4, -16], 0, 0),
    (&[10, -8, 12, 10, -8], 1, 0),
    (&[-6, -2, 5, 2, -6, -4], 1, 1),
    (&[18, -3, -9, 3, -15, -3], 1, 0),
    (&[6, 0, 6, -2, -10], 0, 0),
    (&[-2, -1, 1, 6], 1, 1),
    (&[3, -5, -5, 4], 1, 1),
    (&[3, -18, 15], 0, 0),
    (&[6, -3, -2], 1, 1),
    (&[18, 6, 9], 1, 0),
    (&[0, 12, -18, 18], 0, 0),
    (&[-3, -5, -6, 1, 6], 1, 1),
    (&[-15, 18, 6, 0, -6], 1, 0),
    (&[0, -6, -5], 0, 0),
    (&[-3, -4, 4, -3, 6, -4], 1, 1),
    (&[-8, -2, 2, 4, -4, 8], 0, 0),
    (&[18, 15, 15, 15, 12], 1, 0),
    (&[-3, 0, -1], 1, 1),
    (&[-8, -10, -6, 2, 6], 0, 0),
    (&[-3, -6, 5, 6], 1, 1),
    (&[4, -6, -6, 12, -8], 1, 0),
];

/// `(f, polisirreducible(f))` over Q, coefficients as (numerator, denominator).
#[rustfmt::skip]
const Q_ROWS: &[(&[(i64, i64)], u8)] = &[
    // named Q rows
    (&[(1, 3), (1, 2)], 1),
    (&[(1, 1), (0, 1), (1, 4)], 1),
    (&[(-1, 4), (0, 1), (1, 1)], 0),
    (&[(-1, 2), (0, 1), (2, 1)], 0),
    (&[(-2, 27), (0, 1), (0, 1), (1, 1)], 1),
    (&[(-8, 27), (0, 1), (0, 1), (1, 1)], 0),
    (&[(1, 2)], 0),
    (&[], 0),
    (&[(-3, 1)], 0),
    (&[(1, 3), (0, 1), (1, 3)], 1),
    (&[(-3, 1), (0, 1), (1, 3)], 0),
    (&[(1, 7), (0, 1), (-10, 7), (0, 1), (1, 7)], 1),
    (&[(1, 16), (1, 2), (1, 1)], 0),
    (&[(2, 1), (2, 1)], 1),
    (&[(3, 1), (0, 1), (3, 1)], 1),
    (&[(-6, 1), (0, 1), (6, 1)], 0),
    (&[(-1, 1), (0, 1), (1, 1)], 0),
    (&[(1, 1), (0, 1), (1, 1)], 1),
    (&[(-2, 1), (0, 1), (0, 1), (1, 1)], 1),
    (&[(4, 1), (0, 1), (0, 1), (0, 1), (1, 1)], 0),
    (&[(-1, 5), (0, 1), (-1, 5)], 1),
    (&[(1, 1), (5, 1), (6, 1)], 0),
    (&[(1, 1), (5, 6), (1, 6)], 0),
    // random Q rows
    (&[(2, 1), (-6, 1), (2, 3)], 1),
    (&[(-3, 1)], 0),
    (&[(4, 3), (5, 4)], 1),
    (&[(1, 2), (5, 1), (5, 4), (-1, 3)], 1),
    (&[(-3, 4), (-1, 1), (-2, 1), (3, 4)], 1),
    (&[(5, 2), (2, 1), (-1, 2), (-5, 1), (-4, 3), (-2, 3)], 1),
    (&[(-1, 1), (1, 2)], 1),
    (&[(3, 2)], 0),
    (&[(4, 3), (0, 1), (-3, 1), (0, 1), (-1, 4)], 1),
    (&[(3, 1), (-1, 1), (-3, 1), (-3, 1), (-3, 2), (1, 2)], 1),
    (&[(-1, 1), (-4, 3), (-1, 1), (-5, 2), (5, 2), (-5, 4)], 1),
    (&[(1, 4), (-2, 1)], 1),
    (&[(4, 1), (-3, 2), (-1, 1), (0, 1), (2, 1), (2, 3)], 1),
    (&[(-5, 4), (0, 1), (-2, 1)], 1),
    (&[(0, 1), (-1, 1)], 1),
    (&[(2, 1), (-1, 1), (-1, 1), (6, 1), (-1, 4), (-5, 3)], 1),
    (&[(-1, 4), (3, 4), (3, 1)], 1),
    (&[(0, 1), (-4, 1), (-1, 3), (3, 2), (0, 1), (-2, 1)], 0),
    (&[(-2, 1), (1, 1), (2, 1), (-5, 4)], 1),
    (&[(-5, 1)], 0),
    (&[(4, 3)], 0),
    (&[(0, 1), (-2, 1), (1, 2), (4, 3), (5, 2), (1, 1)], 0),
    (&[(2, 1), (1, 1), (-1, 3), (1, 1), (-4, 3)], 1),
    (&[(-1, 1), (6, 1), (5, 3), (1, 1)], 1),
    (&[(-5, 3)], 0),
    (&[(5, 4)], 0),
    (&[(-5, 4), (-5, 1), (3, 1), (-3, 2), (-5, 4)], 1),
    (&[(-3, 1), (-3, 1), (2, 3), (-5, 2), (-1, 4), (2, 3)], 1),
    (&[(-5, 2)], 0),
    (&[(-1, 1), (0, 1), (-4, 1), (-3, 2), (4, 1), (-1, 2)], 1),
    (&[(3, 2), (-9, 4), (-5, 1), (25, 8)], 0),
    (&[(1, 2), (1, 4), (-1, 2), (-1, 4)], 0),
    (&[(1, 2), (-53, 18), (23, 18), (-133, 36), (1, 3), (-5, 2)], 0),
    (&[(0, 1), (0, 1), (-3, 1), (-30, 1)], 0),
    (&[(6, 1), (26, 1), (-4, 1), (-4, 1)], 0),
    (&[(-15, 8), (12, 1), (5, 24), (-5, 6)], 0),
    (&[(-4, 1), (-3, 1), (32, 3), (17, 3), (2, 3)], 0),
    (&[(1, 6), (-1, 8), (-1, 9), (1, 12)], 0),
    (&[(-1, 8), (1, 4), (1, 4), (1, 8), (1, 1), (-3, 4)], 0),
    (&[(1, 4), (1, 1), (-59, 4), (5, 2)], 0),
    (&[(0, 1), (-8, 1), (0, 1), (4, 1)], 0),
    (&[(0, 1), (-4, 3), (-10, 3), (1, 1), (8, 3)], 0),
    (&[(4, 1), (-17, 6), (19, 12), (-5, 3)], 0),
    (&[(-3, 2), (39, 2), (-83, 4), (1, 4), (-5, 2), (1, 2)], 0),
    (&[(0, 1), (-15, 2), (9, 4), (91, 12), (-5, 2), (5, 6)], 0),
    (&[(-6, 1), (1, 1), (17, 8), (35, 8), (19, 4), (3, 4)], 0),
    (&[(-5, 4), (-163, 8), (17, 4), (7, 1), (-2, 1)], 0),
    (&[(-3, 1), (-7, 2), (1, 2), (-3, 2)], 0),
    (&[(-8, 1), (-13, 2), (11, 4), (23, 16), (-3, 8)], 0),
    (&[(0, 1), (-1, 3), (-5, 6)], 0),
];

/// `(f, p, polisirreducible(Mod(1,p)*f))`, f not reduced modulo p.
#[rustfmt::skip]
const GFP_ROWS: &[(&[i64], i64, u8)] = &[
    // named GF(p) rows
    (&[1, 0, 1], 2, 0),
    (&[-1, 0, 1], 2, 0),
    (&[-2, 0, 0, 1], 2, 0),
    (&[1, 0, 0, 0, 1], 2, 0),
    (&[1, 0, 1], 3, 1),
    (&[-1, 0, 1], 3, 0),
    (&[-2, 0, 0, 1], 3, 0),
    (&[1, 0, 0, 0, 1], 3, 0),
    (&[1, 0, 1], 5, 0),
    (&[-1, 0, 1], 5, 0),
    (&[-2, 0, 0, 1], 5, 0),
    (&[1, 0, 0, 0, 1], 5, 0),
    (&[1, 0, 1], 7, 1),
    (&[-1, 0, 1], 7, 0),
    (&[-2, 0, 0, 1], 7, 1),
    (&[1, 0, 0, 0, 1], 7, 0),
    (&[1, 0, 1], 13, 0),
    (&[-1, 0, 1], 13, 0),
    (&[-2, 0, 0, 1], 13, 1),
    (&[1, 0, 0, 0, 1], 13, 0),
    (&[-1, -1, 1], 2, 1),
    (&[0, -1, 1], 2, 0),
    (&[1, 0, 0, 0, 1], 2, 0),
    (&[1, 2, 3, 2, 1], 2, 0),
    (&[-1, -1, 0, 1], 3, 1),
    (&[0, -1, 0, 1], 3, 0),
    (&[1, 0, 0, 0, 0, 0, 1], 3, 0),
    (&[1, 2, 3, 2, 1], 3, 0),
    (&[-1, -1, 0, 0, 0, 1], 5, 1),
    (&[0, -1, 0, 0, 0, 1], 5, 0),
    (&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 5, 0),
    (&[1, 2, 3, 2, 1], 5, 0),
    (&[-1, -1, 0, 0, 0, 0, 0, 1], 7, 1),
    (&[0, -1, 0, 0, 0, 0, 0, 1], 7, 0),
    (&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 7, 0),
    (&[1, 2, 3, 2, 1], 7, 0),
    (&[1, 1, 1], 2, 1),
    (&[1, 1, 0, 1], 2, 1),
    (&[1, 1, 0, 0, 1], 2, 1),
    (&[1, 0, 1, 0, 1], 2, 0),
    (&[1, 1, 0, 1, 1, 0, 0, 0, 1], 2, 1),
    (&[2, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 0),
    (&[1, 0, 0, 1, 0, 0, 1], 3, 0),
    (&[2, 0, 2], 3, 1),
    (&[-1, 0, -1], 3, 1),
    (&[1, 0, 1, 5], 5, 0),
    (&[2, 0, 1, 5], 5, 1),
    (&[3, 7], 7, 0),
    (&[], 5, 0),
    (&[2], 5, 0),
    (&[10], 5, 0),
    (&[3, 10], 5, 0),
    (&[3, 1], 5, 1),
    (&[-8, 11, -4], 7, 0),
    (&[0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1], 2, 0),
    (&[1, 1, 1, 0, 0, 1, 0, 0, 0, 1], 2, 0),
    (&[1, 0, 1], 2305843009213693951, 1),
    (&[-3, 0, 1], 2305843009213693951, 1),
    (&[-2, 0, 0, 1], 2147483647, 0),
    (&[-3, 0, 0, 1], 2147483647, 0),
    (&[-2, 0, 1], 65537, 0),
    (&[1, 0, 0, 0, 1], 65537, 0),
    (&[3, 1, 0, 0, 0, 1], 7, 1),
    (&[1, 1, 0, 1], 65537, 0),
    // ffinit and products
    (&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 2, 1),
    (&[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], 2, 0),
    (&[1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1], 2, 0),
    (&[1, 0, 2, 1, 2, 0, 1, 1], 3, 1),
    (&[1, 1, 2, 3, 3, 2, 1, 2, 1], 3, 0),
    (&[1, 1, 4, 4, 9, 6, 11, 7, 9, 6, 6, 4, 1, 2, 1], 3, 0),
    (&[4, 4, 0, 0, 0, 1], 5, 1),
    (&[4, 8, 4, 0, 0, 1, 1], 5, 0),
    (&[16, 16, 0, 0, 0, 8, 4, 0, 0, 0, 1], 5, 0),
    (&[1, 1, 1, 1, 1], 7, 1),
    (&[1, 2, 2, 2, 2, 1], 7, 0),
    (&[1, 3, 4, 5, 6, 5, 3, 2, 1], 7, 0),
    (&[10, 9, 1, 1], 11, 1),
    (&[10, 19, 10, 2, 1], 11, 0),
    (&[100, 190, 110, 39, 20, 2, 1], 11, 0),
    (&[100, 99, 1, 1], 101, 1),
    (&[100, 199, 100, 2, 1], 101, 0),
    (&[10000, 19900, 10100, 399, 200, 2, 1], 101, 0),
    (&[1, 1, 1], 65537, 1),
    (&[1, 2, 2, 1], 65537, 0),
    (&[1, 3, 4, 3, 1], 65537, 0),
    (&[1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1], 2, 1),
    (&[1, 2, 2, 1, 0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 2, 1, 1, 1], 2, 0),
    (&[1, 1, 2, 1, 1, 2, 3, 5, 3, 2, 1, 4, 6, 7, 6, 3, 6, 5, 8, 6, 4, 4, 3, 4, 3, 4, 3, 4, 3, 2, 2, 0, 1], 2, 0),
    (&[2, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 0, 1], 3, 1),
    (&[2, 2, 0, 1, 2, 1, 2, 2, 0, 2, 4, 2, 1, 1], 3, 0),
    (&[4, 2, 0, 4, 5, 1, 9, 4, 1, 12, 14, 2, 12, 9, 4, 10, 10, 0, 8, 8, 4, 4, 4, 0, 1], 3, 0),
    // random GF(p) rows
    (&[2, 0, -2, -1, 1], 2, 0),
    (&[1, -2, 3, -2, 1, 1, 0, -1, 1], 2, 0),
    (&[2, 0, -1, 1], 2, 0),
    (&[-1, 3, 1], 2, 1),
    (&[-2, 0, -2, 1], 2, 0),
    (&[-2, 1], 2, 1),
    (&[3, 1, 2, 3, 1, 3, 2, 1], 2, 0),
    (&[-2, -1, -2, 3, 0, -2, 1, 3, -1, 1], 2, 0),
    (&[-2, 3, 1], 2, 0),
    (&[3, 2, 2, -1, 1], 2, 1),
    (&[-1, 2, -1, 2, 1], 2, 0),
    (&[3, 1], 2, 1),
    (&[-2, 1, 2], 3, 1),
    (&[3, 1, 2, 1], 3, 0),
    (&[-1, 0, -2, 1], 3, 1),
    (&[1, -3, 2], 3, 0),
    (&[3, 0, 2], 3, 0),
    (&[2, 4, 4, 4, -2, -3, 5, 1], 3, 0),
    (&[-1, 5, -3, 0, 1], 3, 1),
    (&[1, 0, 4, 5, -1, 3, -1, 2], 3, 1),
    (&[5, 0, -3, 1], 3, 0),
    (&[4, -3, 4, -3, 2], 3, 1),
    (&[-1, 5, -2, -3, -3, -2, 1, -3, -1, 2], 3, 0),
    (&[4, -3, -3, -2, 1], 3, 0),
    (&[3, 1], 5, 1),
    (&[5, 8, 7, 6, -5, 5, -2, 3], 5, 0),
    (&[-2, -1, 1, 4, -5, 9, 0, 7, 1], 5, 1),
    (&[5, 6, 9, 9, 1], 5, 0),
    (&[2], 5, 0),
    (&[4, 4, 7, 6, 4, 3, -3, 2], 5, 1),
    (&[7, -2, 9, 0, 5, 0, -2, 4], 5, 0),
    (&[6, 0, -2, 4, -2, 5, 3], 5, 0),
    (&[-1, 1, -1, 4, 2, -5, 3, 7, 5, 4], 5, 0),
    (&[6, 1, 3, 7, 7, 0, -4, 3], 5, 1),
    (&[9, -3, 3, 7, -3, -3, 8, -1, 7, 2], 5, 0),
    (&[5, -2, 7, -3, -1, -5, -5, -2, 3, 4], 5, 0),
    (&[4, 13, 3, -3, 5], 7, 0),
    (&[-5, 8, -3, 3, 1, 12, 11, 11, 3, 2], 7, 0),
    (&[6], 7, 0),
    (&[10, 6, 11, 7, 12, -2, 0, 6], 7, 0),
    (&[-5, 1, 3, -5, 11, 12, 8, 3], 7, 0),
    (&[-3, -4, -7, 2, 7, 5, 1, -4, 6], 7, 0),
    (&[13, -3, -2, 8, 4, 12, 6], 7, 1),
    (&[-6, 7, 2, 5, 4, -2, 9, 1, 2, 3], 7, 0),
    (&[4, 7, 8, 12, -7, 1], 7, 1),
    (&[2, -2, 9, -5, -3, 1], 7, 0),
    (&[1, 5, 10, 11, 13, 0, 5, -4, 3], 7, 0),
    (&[-6, 0, 3, 7, 5], 7, 1),
    (&[9], 11, 0),
    (&[9, 6], 11, 1),
    (&[19, 14, 15, -4, -10, 19, -9, 3], 11, 0),
    (&[17, -8, -11, 19, 4, -10, -7, 2], 11, 0),
    (&[17, -8, 12, 17, 5, 13, 15, 9], 11, 0),
    (&[0, -3, 2], 11, 0),
    (&[13, 10, 18, 15, 2], 11, 0),
    (&[1, 14, 17, 4, 7], 11, 0),
    (&[6, -7, 13, 6, 11, -10, -2, 3], 11, 0),
    (&[7, -4, 7, 15, -7, 20, 2], 11, 0),
    (&[4, -2, 8, 7], 11, 0),
    (&[21, 20, 16, 11, 17, 16, -11, -2, 7], 11, 0),
    (&[19, 24, 10, 1, 10, -4, 4], 13, 0),
    (&[3, -2, 18, -8, 0, 2, 7], 13, 0),
    (&[12, 14, 1], 13, 1),
    (&[9, -13, 3], 13, 0),
    (&[0, 5, 9, 18, -8, 15, 2, 22, 20, 7], 13, 0),
    (&[18, 18, 18, 14, -3, 16, 0, 4], 13, 1),
    (&[-2, -6, -7, 21, 18, 4, 16, -13, 4, 6], 13, 0),
    (&[-10, -4, 4, 1], 13, 0),
    (&[20, 2, -2, 23, 12], 13, 1),
    (&[25, 1, -8, -2, 10, 2], 13, 0),
    (&[-9, 0, -8, 4, -12, 17, -12, -3, 10], 13, 0),
    (&[-12, 2, 8, 12], 13, 0),
    (&[-84, 112, -37, 62, -12, 12, -68, 80, 113, 61], 101, 0),
    (&[55], 101, 0),
    (&[-5, 30, 119, 93, 25, 37], 101, 0),
    (&[57], 101, 0),
    (&[91, -84, -40, -29, 19], 101, 1),
    (&[3, 18, 142, 54], 101, 0),
    (&[17, 70, 117, 48, 101, 41], 101, 0),
    (&[96, -16, -81, 183, 188, -44, 131, 55, 21], 101, 0),
    (&[17, -32, -30, 155, 75, -72, 42, 113, 28], 101, 0),
    (&[-52, -58, -86, 58], 101, 0),
    (&[139, 47, 17], 101, 1),
    (&[83, 6, -88, 41, -85, -43, 56], 101, 0),
    (&[85713, 33709, 24604], 65537, 1),
    (&[43542, -37276, 16747, 91235, 125743, 30203], 65537, 0),
    (&[-5096, 124210, 47013], 65537, 1),
    (&[-4667, 109566, 35810, 71219, 31459], 65537, 0),
    (&[49587, 75383, 28493], 65537, 0),
    (&[-61763, 113244, -63465, 91100, 109278, -5806, 99675, 3367], 65537, 1),
    (&[-13323, 54639], 65537, 1),
    (&[110655, 18390, 92457, 3625, 94543, -41730, 5886], 65537, 0),
    (&[80890, 28601, 112970, -39788, 53120], 65537, 0),
    (&[23963], 65537, 0),
    (&[-39885, 105549, 95554, -5477, 26880, -14350, 16587], 65537, 0),
    (&[55362, -17688, 86061, 105914, 96643, 51567], 65537, 0),
];

#[test]
fn z_table_matches_pari() {
    assert_eq!(Z_ROWS.len(), 165);
    for &(c, pari, ring) in Z_ROWS {
        let c = nonempty(c, 0);
        let f = z(&c);
        let want = ring == 1;
        assert_eq!(is_irreducible(&f).unwrap(), want, "Z: f = {c:?}");
        // Same ring through the machine-integer coefficient types.
        let f64_ = UnivariatePolynomial::new(c.clone());
        assert_eq!(is_irreducible(&f64_).unwrap(), want, "i64: f = {c:?}");
        if let Ok(c32) = c
            .iter()
            .map(|&a| i32::try_from(a))
            .collect::<std::result::Result<Vec<i32>, _>>()
        {
            assert_eq!(
                is_irreducible(&UnivariatePolynomial::new(c32)).unwrap(),
                want,
                "i32: f = {c:?}"
            );
        }
        // PARI's convention (over Q, content ignored).
        assert_eq!(
            is_irreducible_over_integers(&f).unwrap(),
            pari == 1,
            "over Q: f = {c:?}"
        );
        // The same polynomial in Q[x]: irreducibility over the field is PARI's answer.
        let fq = q(&c.iter().map(|&a| (a, 1)).collect::<Vec<_>>());
        assert_eq!(is_irreducible(&fq).unwrap(), pari == 1, "Q: f = {c:?}");
    }
}

/// The inputs on which the old code answered `Ok(true)` (degree 2-3 square-free was
/// "assumed irreducible", higher degree "irreducible unless not square-free").
#[test]
fn old_facade_inputs_are_decided() {
    // gp: factor(x^2-1) has two factors; polisirreducible((x^2-2)*(x^2-3)) = 0;
    // 2*x+2 = 2*(x+1) in Z[x]; factor(6*x^2+5*x+1) = (2x+1)(3x+1) (no integer root).
    assert!(!is_irreducible(&z(&[-1, 0, 1])).unwrap());
    assert!(!is_irreducible(&z(&[6, 0, -5, 0, 1])).unwrap());
    assert!(!is_irreducible(&z(&[2, 2])).unwrap());
    assert!(!is_irreducible(&z(&[1, 5, 6])).unwrap());
    assert!(!is_irreducible(&z(&[4, 0, 0, 0, 1])).unwrap());
    // gp: polisirreducible(x^2+1) = polisirreducible(x^3-2) = 1.
    assert!(is_irreducible(&z(&[1, 0, 1])).unwrap());
    assert!(is_irreducible(&z(&[-2, 0, 0, 1])).unwrap());
}

#[test]
fn q_table_matches_pari() {
    assert_eq!(Q_ROWS.len(), 73);
    for &(c, pari) in Q_ROWS {
        let c = nonempty(c, (0, 1));
        assert_eq!(is_irreducible(&q(&c)).unwrap(), pari == 1, "Q: f = {c:?}");
    }
}

#[test]
fn gf_p_table_matches_pari() {
    assert_eq!(GFP_ROWS.len(), 187);
    for &(c, p, pari) in GFP_ROWS {
        let pi = Integer::from(p);
        let f = ints(c);
        let got = berlekamp::is_irreducible(&f, &pi).unwrap();
        assert_eq!(got, pari == 1, "GF({p}): f = {c:?}");
        // Consistency with the factorization (an independent code path for the split).
        if let Ok((_, fac)) = berlekamp::factor(&f, &pi) {
            let single = fac.len() == 1 && fac[0].1 == 1;
            assert_eq!(got, single, "GF({p}): factor disagrees, f = {c:?}");
        }
    }
}

#[test]
fn large_constants_and_moduli() {
    let big = |s: &str| Integer::from_decimal_str(s).unwrap();
    let cz = |c: Integer| UnivariatePolynomial::new(vec![c]);
    // gp: isprime(2^61-1) = 1, below the proven bound.
    assert!(is_irreducible(&cz(big("2305843009213693951"))).unwrap());
    assert!(is_irreducible(&cz(big("-2305843009213693951"))).unwrap());
    // gp: isprime(2^89-1) = 1. Above 318665857834031151167461 the primality test is
    // Baillie-PSW plus an ECPP proof, so the constant is decided (before the proving
    // `is_prime` this was an honest `Err`; the proof now decides it).
    assert_eq!(
        is_irreducible(&cz(big("618970019642690137449562111"))),
        Ok(true)
    );
    // gp: factor(318665857834031151167461) = 399165290221 * 798330580441, a strong
    // pseudoprime to the bases 2..37: it must never be reported irreducible.
    assert!(!matches!(
        is_irreducible(&cz(big("318665857834031151167461"))),
        Ok(true)
    ));
    assert_eq!(
        is_irreducible(&cz(big("318665857834031151167461"))),
        Ok(false)
    );
    // gp: isprime(2^89+1) = 0 (a composite is always detected).
    assert!(!is_irreducible(&cz(big("618970019642690137449562113"))).unwrap());
    // A large prime content: (2^61-1)·x + 2 is primitive and linear (gp polisirreducible = 1);
    // (2^61-1)·(x + 1) has content 2^61-1.
    let m61 = big("2305843009213693951");
    let f = UnivariatePolynomial::new(vec![Integer::from(2), m61.clone()]);
    assert!(is_irreducible(&f).unwrap());
    let g = UnivariatePolynomial::new(vec![m61.clone(), m61]);
    assert!(!is_irreducible(&g).unwrap());
    // GF(2^89 - 1): gp polisirreducible(Mod(1,2^89-1)*(x^2+1)) = 1, (x^2-1) gives 0.
    let m89 = big("618970019642690137449562111");
    assert!(berlekamp::is_irreducible(&ints(&[1, 0, 1]), &m89).unwrap());
    assert!(!berlekamp::is_irreducible(&ints(&[-1, 0, 1]), &m89).unwrap());
}

#[test]
fn invalid_modulus_is_an_error() {
    for p in [0i64, 1, 4, 9, -7] {
        assert!(
            berlekamp::is_irreducible(&ints(&[1, 0, 1]), &Integer::from(p)).is_err(),
            "p = {p}"
        );
    }
}

// ---------------------------------------------------------------------------------------
// The generic fallback: coefficient rings the function does not recognise.

/// GF(P) for a prime P, a field the generic code knows nothing about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fp<const P: u64>(u64);

impl<const P: u64> Fp<P> {
    fn new(a: i64) -> Self {
        Fp(a.rem_euclid(P as i64) as u64)
    }
    fn inv(self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        // Fermat: a^(P-2)
        let (mut b, mut e, mut acc) = (self.0, P - 2, 1u64);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * b % P;
            }
            b = b * b % P;
            e >>= 1;
        }
        Some(Fp(acc))
    }
}
impl<const P: u64> fmt::Display for Fp<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl<const P: u64> Add for Fp<P> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Fp((self.0 + o.0) % P)
    }
}
impl<const P: u64> Sub for Fp<P> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Fp((self.0 + P - o.0) % P)
    }
}
impl<const P: u64> Mul for Fp<P> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Fp(self.0 * o.0 % P)
    }
}
impl<const P: u64> Neg for Fp<P> {
    type Output = Self;
    fn neg(self) -> Self {
        Fp((P - self.0) % P)
    }
}
// Division in F_p is multiplication by the inverse, which this lint cannot see.
#[allow(clippy::suspicious_arithmetic_impl)]
impl<const P: u64> Div for Fp<P> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        self * o.inv().expect("division by zero in F_p")
    }
}
impl<const P: u64> Ring for Fp<P> {
    fn zero() -> Self {
        Fp(0)
    }
    fn one() -> Self {
        Fp(1 % P)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
    fn is_one(&self) -> bool {
        self.0 == 1
    }
}
impl<const P: u64> CommutativeRing for Fp<P> {}
impl<const P: u64> IntegralDomain for Fp<P> {}
impl<const P: u64> EuclideanDomain for Fp<P> {
    fn norm(&self) -> u64 {
        u64::from(self.0 != 0)
    }
    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        match other.inv() {
            Some(i) => Ok((*self * i, Fp(0))),
            None => Err(MathError::DivisionByZero),
        }
    }
}
impl<const P: u64> Field for Fp<P> {
    fn inverse(&self) -> Result<Self> {
        self.inv().ok_or(MathError::DivisionByZero)
    }
}

fn fp<const P: u64>(c: &[i64]) -> UnivariatePolynomial<Fp<P>> {
    UnivariatePolynomial::new(c.iter().map(|&a| Fp::<P>::new(a)).collect())
}

#[test]
fn generic_field_decides_only_what_it_can() {
    // Decided: 0 and units are not irreducible, degree 1 is, a repeated factor is not.
    assert!(!is_irreducible(&fp::<5>(&[0])).unwrap());
    assert!(!is_irreducible(&fp::<5>(&[2])).unwrap());
    assert!(is_irreducible(&fp::<5>(&[1, 2])).unwrap());
    // gp: factormod((x+1)^2*(x^2+2), 5): gcd(f, f') = x + 1.
    assert!(!is_irreducible(&fp::<5>(&[2, 4, 3, 2, 1])).unwrap());
    // Undecided: degree 2 and 3 need a root test the generic field does not have.
    // gp: polisirreducible(Mod(1,5)*(x^2+1)) = 0, (x^2+2) gives 1,
    // polisirreducible(Mod(1,7)*(x^3-2)) = 1, polisirreducible(Mod(1,5)*(x^3+x+1)) = 1,
    // polisirreducible(Mod(1,5)*(x^3-1)) = 0.
    assert!(is_irreducible(&fp::<5>(&[1, 0, 1])).is_err());
    assert!(is_irreducible(&fp::<5>(&[2, 0, 1])).is_err());
    assert!(is_irreducible(&fp::<7>(&[-2, 0, 0, 1])).is_err());
    assert!(is_irreducible(&fp::<5>(&[1, 1, 0, 1])).is_err());
    assert!(is_irreducible(&fp::<5>(&[-1, 0, 0, 1])).is_err());
    // f' = 0: reducible over a perfect field, but the generic code cannot know that R is
    // perfect (over F_p(t), x^p - t is irreducible). gp: factormod(x^5-2, 5) = (x+3)^5.
    assert!(is_irreducible(&fp::<5>(&[-2, 0, 0, 0, 0, 1])).is_err());
    // The same inputs, decided over GF(p) by berlekamp::is_irreducible.
    let p5 = Integer::from(5);
    assert!(!berlekamp::is_irreducible(&ints(&[1, 0, 1]), &p5).unwrap());
    assert!(berlekamp::is_irreducible(&ints(&[2, 0, 1]), &p5).unwrap());
    assert!(berlekamp::is_irreducible(&ints(&[-2, 0, 0, 1]), &Integer::from(7)).unwrap());
    assert!(berlekamp::is_irreducible(&ints(&[1, 1, 0, 1]), &p5).unwrap());
    assert!(!berlekamp::is_irreducible(&ints(&[-1, 0, 0, 1]), &p5).unwrap());
    assert!(!berlekamp::is_irreducible(&ints(&[-2, 0, 0, 0, 0, 1]), &p5).unwrap());
}

/// Z under another name: a Euclidean domain that is not a field, with a decided
/// `unit_status`, that the function does not recognise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Zw(i64);

impl fmt::Display for Zw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Add for Zw {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Zw(self.0 + o.0)
    }
}
impl Sub for Zw {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Zw(self.0 - o.0)
    }
}
impl Mul for Zw {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Zw(self.0 * o.0)
    }
}
impl Neg for Zw {
    type Output = Self;
    fn neg(self) -> Self {
        Zw(-self.0)
    }
}
impl Ring for Zw {
    fn zero() -> Self {
        Zw(0)
    }
    fn one() -> Self {
        Zw(1)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
    fn is_one(&self) -> bool {
        self.0 == 1
    }
    fn unit_status(&self) -> Option<bool> {
        Some(self.0 == 1 || self.0 == -1)
    }
}
impl CommutativeRing for Zw {}
impl IntegralDomain for Zw {}
impl EuclideanDomain for Zw {
    fn norm(&self) -> u64 {
        self.0.unsigned_abs()
    }
    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.0 == 0 {
            return Err(MathError::DivisionByZero);
        }
        Ok((
            Zw(self.0.div_euclid(other.0)),
            Zw(self.0.rem_euclid(other.0)),
        ))
    }
    fn normalization_unit(&self) -> Option<Self> {
        Some(Zw(if self.0 < 0 { -1 } else { 1 }))
    }
}

fn zw(c: &[i64]) -> UnivariatePolynomial<Zw> {
    UnivariatePolynomial::new(c.iter().map(|&a| Zw(a)).collect())
}

#[test]
fn generic_euclidean_domain_decides_only_what_it_can() {
    // Decided: a non-unit content makes it reducible; a primitive linear polynomial is
    // irreducible; a repeated factor makes it reducible; 0 and units are not irreducible.
    assert!(!is_irreducible(&zw(&[2, 2])).unwrap());
    assert!(!is_irreducible(&zw(&[3, 0, 3])).unwrap());
    assert!(is_irreducible(&zw(&[-1, -1])).unwrap());
    assert!(is_irreducible(&zw(&[3, 2])).unwrap());
    assert!(!is_irreducible(&zw(&[1, 2, 1])).unwrap());
    assert!(!is_irreducible(&zw(&[0])).unwrap());
    assert!(!is_irreducible(&zw(&[-1])).unwrap());
    // Undecided: a non-unit constant (irreducible iff a prime of R), and square-free
    // polynomials of degree >= 2 (gp: polisirreducible(x^2-1) = 0, x^2+1 gives 1).
    assert!(is_irreducible(&zw(&[7])).is_err());
    assert!(is_irreducible(&zw(&[-1, 0, 1])).is_err());
    assert!(is_irreducible(&zw(&[1, 0, 1])).is_err());
}
