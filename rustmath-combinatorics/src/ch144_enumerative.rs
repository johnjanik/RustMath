//! Enumerative combinatorics — supplementary counting functions (MAGMA Chapter 144).
//!
//! Chapter 144 ("Enumerative Combinatorics") of the MAGMA Handbook is a catalogue of
//! classical counting functions. Most of them already live in this crate's `lib.rs`
//! (`factorial`, `binomial`, `multinomial`, `catalan`, `fibonacci`, `lucas`,
//! `stirling_first`, `stirling_second`, `bell_number`, `eulerian`), while set/subset
//! enumeration lives in `subset.rs`/`enumeration.rs`, necklaces in `binary_words.rs`
//! and Pólya enumeration in `group_action.rs`.
//!
//! This module fills the remaining Chapter-144 intrinsics:
//!
//! | MAGMA intrinsic                    | function here                     |
//! |------------------------------------|-----------------------------------|
//! | `NumberOfPermutations(n, k)`       | [`number_of_permutations`] (falling factorial, one balanced product) |
//! | `Fibonacci(n)` (negative index)    | [`fibonacci_signed`] (doubling, O(M(n)))  |
//! | `Lucas(n)` (negative index)        | [`lucas_signed`] (doubling)       |
//! | `GeneralizedFibonacciNumber(...)`  | [`generalized_fibonacci_number`] (G_n = g0 F_(n-1) + g1 F_n) |
//! | `BernoulliNumber(n)`               | [`bernoulli_number`] (re-export)  |
//! | `BernoulliPolynomial(n)`           | [`bernoulli_polynomial`]          |
//! | `HarmonicNumber(n)`                | [`harmonic_number`] (re-export)   |
//!
//! Bernoulli/harmonic numbers themselves are reused from
//! `rustmath-rationals::special_numbers` (no re-implementation).
//!
//! Reference: MAGMA Handbook, Chapter 144 §144.2.

use crate::binomial;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

pub use rustmath_rationals::{bernoulli as bernoulli_number, harmonic as harmonic_number};

/// `NumberOfPermutations(n, k)` — the number of permutations of `n` distinct objects
/// taken `k` at a time, i.e. the falling factorial `n!/(n-k)! = n·(n-1)···(n-k+1)`.
///
/// Returns `0` when `k > n`. The u32 form of [`crate::falling_factorial_integer`] (one
/// balanced product).
pub fn number_of_permutations(n: u32, k: u32) -> Integer {
    crate::falling_factorial_integer(&Integer::from(n), u64::from(k))
}

/// (F_m, F_(m+1)) by doubling. From the top bit of m down, with (a, b) = (F_j, F_(j+1)):
/// F_2j = a (2b - a), F_(2j+1) = a^2 + b^2, and on a set bit (F_(2j+1), F_2j + F_(2j+1)).
/// These are the cases m = n and m = n + 1 of the addition formula
/// F_(m+n) = F_m F_(n+1) + F_(m-1) F_n (Knuth, TAOCP vol. 1, §1.2.8; D. Takahashi, *A fast
/// algorithm for computing large Fibonacci numbers*, Inform. Process. Lett. 75 (2000)
/// 243-246). Three multiplications per bit, O(M(m)) overall.
pub(crate) fn fibonacci_pair(m: u64) -> (Integer, Integer) {
    let mut a = Integer::zero();
    let mut b = Integer::one();
    if m == 0 {
        return (a, b);
    }
    let top = 63 - m.leading_zeros();
    for bit in (0..=top).rev() {
        let two_b_minus_a = &(&b + &b) - &a;
        let c = &a * &two_b_minus_a; // F_2j
        let d = &(&a * &a) + &(&b * &b); // F_(2j+1)
        if (m >> bit) & 1 == 1 {
            b = &c + &d;
            a = d;
        } else {
            a = c;
            b = d;
        }
    }
    (a, b)
}

/// `GeneralizedFibonacciNumber(g0, g1, n)` — the `n`-th term of the two-term linear
/// recurrence `G_0 = g0`, `G_1 = g1`, `G_k = G_{k-1} + G_{k-2}`, for every integer `n`.
/// `Fibonacci` and `Lucas` are the special cases `(g0, g1) = (0, 1)` and `(2, 1)`.
///
/// Computed as `G_n = g0 F_(n-1) + g1 F_n` (it gives g0 at n = 0 since F_(-1) = 1, g1 at
/// n = 1, and satisfies the recurrence by linearity), with the Fibonacci pair by doubling
/// (`fibonacci_pair`) and `F_(-m) = (-1)^(m+1) F_m` for negative indices. O(M(|n|)) work.
pub fn generalized_fibonacci_number(g0: i64, g1: i64, n: i64) -> Integer {
    let (f_nm1, f_n) = if n >= 1 {
        // (F_(n-1), F_n)
        fibonacci_pair((n - 1) as u64)
    } else {
        // n = -m with m >= 0: F_n = F_(-m) = (-1)^(m+1) F_m and
        // F_(n-1) = F_(-(m+1)) = (-1)^m F_(m+1).
        let m = n.unsigned_abs();
        let (fm, fm1) = fibonacci_pair(m);
        let f_nm1 = if m.is_multiple_of(2) { fm1 } else { -fm1 };
        let f_n = if m % 2 == 1 { fm } else { -fm };
        (f_nm1, f_n)
    };
    &(&Integer::from(g0) * &f_nm1) + &(&Integer::from(g1) * &f_n)
}

/// `Fibonacci(n)` extended to all integers: `F_{-n} = (-1)^{n+1} F_n`.
pub fn fibonacci_signed(n: i64) -> Integer {
    generalized_fibonacci_number(0, 1, n)
}

/// `Lucas(n)` extended to all integers: `L_{-n} = (-1)^n L_n`.
pub fn lucas_signed(n: i64) -> Integer {
    generalized_fibonacci_number(2, 1, n)
}

/// `BernoulliPolynomial(n)` — the `n`-th Bernoulli polynomial
/// `B_n(x) = Σ_{k=0}^{n} C(n,k) B_k x^{n-k}`, returned as a coefficient vector in
/// ascending powers of `x` (index `j` is the coefficient of `x^j`).
///
/// Uses the standard convention `B_1 = -1/2` (from `rustmath-rationals`), so e.g.
/// `B_2(x) = x^2 - x + 1/6`.
pub fn bernoulli_polynomial(n: u32) -> Vec<Rational> {
    // coeff[power] = C(n, power) * B_{n-power}
    let mut coeffs = Vec::with_capacity((n + 1) as usize);
    for power in 0..=n {
        let k = n - power; // B_k contributes to x^{n-k} = x^{power}
        let binom = Rational::from_integer(binomial(n, power));
        let bk = bernoulli_number(k).expect("bernoulli number");
        coeffs.push(binom * bk);
    }
    coeffs
}

/// Evaluate a polynomial (ascending-power coefficient vector) at a rational point.
pub fn eval_poly(coeffs: &[Rational], x: &Rational) -> Rational {
    // Horner from the top.
    let mut acc = Rational::zero();
    for c in coeffs.iter().rev() {
        acc = acc * x.clone() + c.clone();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_of_permutations() {
        // P(5,2) = 20, P(5,0)=1, P(5,5)=120, P(5,6)=0
        assert_eq!(number_of_permutations(5, 2), Integer::from(20));
        assert_eq!(number_of_permutations(5, 0), Integer::from(1));
        assert_eq!(number_of_permutations(5, 5), Integer::from(120));
        assert_eq!(number_of_permutations(5, 6), Integer::from(0));
        // P(10,3) = 720
        assert_eq!(number_of_permutations(10, 3), Integer::from(720));
    }

    #[test]
    fn test_generalized_fibonacci_forward() {
        // Fibonacci: 0,1,1,2,3,5,8,13
        for (n, v) in [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5), (6, 8), (7, 13)] {
            assert_eq!(fibonacci_signed(n), Integer::from(v), "F_{}", n);
        }
        // Lucas: 2,1,3,4,7,11,18
        for (n, v) in [(0, 2), (1, 1), (2, 3), (3, 4), (4, 7), (5, 11), (6, 18)] {
            assert_eq!(lucas_signed(n), Integer::from(v), "L_{}", n);
        }
    }

    #[test]
    fn test_fibonacci_negative_index() {
        // F_{-n} = (-1)^{n+1} F_n:  F_{-1}=1, F_{-2}=-1, F_{-3}=2, F_{-4}=-3, F_{-5}=5
        assert_eq!(fibonacci_signed(-1), Integer::from(1));
        assert_eq!(fibonacci_signed(-2), Integer::from(-1));
        assert_eq!(fibonacci_signed(-3), Integer::from(2));
        assert_eq!(fibonacci_signed(-4), Integer::from(-3));
        assert_eq!(fibonacci_signed(-5), Integer::from(5));
    }

    #[test]
    fn test_lucas_negative_index() {
        // L_{-n} = (-1)^n L_n:  L_{-1}=-1, L_{-2}=3, L_{-3}=-4, L_{-4}=7
        assert_eq!(lucas_signed(-1), Integer::from(-1));
        assert_eq!(lucas_signed(-2), Integer::from(3));
        assert_eq!(lucas_signed(-3), Integer::from(-4));
        assert_eq!(lucas_signed(-4), Integer::from(7));
    }

    #[test]
    fn test_generalized_fibonacci_custom() {
        // g0=3, g1=5 -> 3,5,8,13,21,34
        assert_eq!(generalized_fibonacci_number(3, 5, 0), Integer::from(3));
        assert_eq!(generalized_fibonacci_number(3, 5, 1), Integer::from(5));
        assert_eq!(generalized_fibonacci_number(3, 5, 2), Integer::from(8));
        assert_eq!(generalized_fibonacci_number(3, 5, 5), Integer::from(34));
        // Backwards: G_{-1} = G_1 - G_0 = 5 - 3 = 2
        assert_eq!(generalized_fibonacci_number(3, 5, -1), Integer::from(2));
    }

    #[test]
    fn test_bernoulli_polynomial() {
        // B_0(x) = 1
        assert_eq!(bernoulli_polynomial(0), vec![Rational::from_integer(1)]);
        // B_1(x) = x - 1/2
        assert_eq!(
            bernoulli_polynomial(1),
            vec![Rational::new(-1, 2).unwrap(), Rational::from_integer(1)]
        );
        // B_2(x) = x^2 - x + 1/6
        assert_eq!(
            bernoulli_polynomial(2),
            vec![
                Rational::new(1, 6).unwrap(),
                Rational::from_integer(-1),
                Rational::from_integer(1)
            ]
        );
        // B_3(x) = x^3 - (3/2)x^2 + (1/2)x
        assert_eq!(
            bernoulli_polynomial(3),
            vec![
                Rational::from_integer(0),
                Rational::new(1, 2).unwrap(),
                Rational::new(-3, 2).unwrap(),
                Rational::from_integer(1),
            ]
        );
    }

    #[test]
    fn test_bernoulli_polynomial_value() {
        // B_n(0) = B_n. B_1(1) = 1/2.
        let b2 = bernoulli_polynomial(2);
        assert_eq!(
            eval_poly(&b2, &Rational::from_integer(0)),
            bernoulli_number(2).unwrap()
        );
        let b1 = bernoulli_polynomial(1);
        assert_eq!(eval_poly(&b1, &Rational::from_integer(1)), Rational::new(1, 2).unwrap());
    }
}
