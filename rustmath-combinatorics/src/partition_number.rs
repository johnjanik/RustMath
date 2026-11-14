//! The number of partitions p(n) as an exact integer.
//!
//! MAGMA Handbook ch. 145 §145.2 `NumberOfPartitions(n)` (digest
//! `rustmath-lava/docs/Chapter_145_Partitions_Words_and_Young_Tableaux.md:55`) and ch. 18 §18.8
//! `NumberOfPartitions(n)` (`Chapter_18_Ring_of_Integers.md:268`).
//!
//! Algorithms, as stated in F. Johansson, *Efficient implementation of the
//! Hardy-Ramanujan-Rademacher formula*, LMS J. Comput. Math. 15 (2012) 341-359
//! (arXiv:1205.5991):
//!
//! - n <= 1000: Euler's recurrence (1.2), from the pentagonal number theorem (1.1).
//! - n > 1000: the Hardy-Ramanujan-Rademacher series (1.4)-(1.5) (H. Rademacher, *On the
//!   partition function p(n)*, Proc. London Math. Soc. 43 (1938) 241-254), with the
//!   Kloosterman-type sums A_k(n) evaluated by Selberg's formula (2.1) (A. L. Whiteman, *A sum
//!   connected with the series for the partition function*, Pacific J. Math. 6 (1956)
//!   159-176) through Algorithm 1 of Johansson's paper, and the number of terms chosen by
//!   Rademacher's remainder bound (1.8).
//!
//! Certification. Every term is an enclosure computed with MPFR's directed rounding (each
//! operation rounded down for the lower end and up for the upper end; MPFR rounds `+ - * /`,
//! `sqrt`, `cos`, `cosh`, `sinh` and pi correctly in every direction). The sum of the
//! enclosures, widened by the computed upper bound Mb < 1/4 of |R(n, N)|, contains p(n). The
//! value is returned only when that interval contains exactly one integer. Otherwise the guard
//! precision doubles and the sum is recomputed. A value is never returned unproved.
//!
//! Cost: the search for the terms of Selberg's formula is about 0.12 n word steps (N is about
//! 0.34 n^(1/2) and each A_k needs 2k steps), so the time grows linearly in n. Johansson's
//! Algorithm 2 (Whiteman's factorisation of A_k, O(n^(1/2 + o(1)))) would remove that term; it
//! is a possible later speed-up.

use std::cmp::Ordering;

use num_bigint::{BigInt, Sign};
use rug::float::{Constant, Round};
use rug::ops::AssignRound;
use rug::Float;
use rustmath_integers::Integer;

/// The largest n that [`number_of_partitions`] accepts.
pub const NUMBER_OF_PARTITIONS_MAX_N: u64 = 10_000_000_000_000_000;

/// Euler's recurrence is used up to this n; the series above.
const EULER_MAX: u64 = 1000;

/// The guard precision doubles at most this many times before the run is declared a bug.
const MAX_ROUNDS: u32 = 10;

/// MAGMA `NumberOfPartitions(n)` (ch. 145 §145.2, ch. 18 §18.8): p(n), the number of
/// partitions of n. p(0) = 1. Exact for every n <= 10^16: Euler's recurrence for n <= 1000,
/// the Hardy-Ramanujan-Rademacher series above, returned only when a rigorous enclosure of the
/// series plus Rademacher's remainder bound contains exactly one integer.
///
/// Cost (release): 8 ms at 10^6, 0.15 s at 10^8, 0.94 s at 10^9, 5.4 s at 10^10. It grows
/// linearly in n (the search for the terms of Selberg's formula), so n must be small in
/// MAGMA's sense.
///
/// # Panics
///
/// If n > 10^16 ([`NUMBER_OF_PARTITIONS_MAX_N`]): near n = 8.4e16 the first term of the
/// series, about e^C with C = (pi/6) sqrt(24n - 1), overflows MPFR's exponent range (2^30 - 1
/// by default), and at 10^16 the run already takes at least 5e6 s. Also if the process-wide
/// MPFR exponent range has been narrowed so far that the first term does not fit. Both checks
/// run before any work.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::number_of_partitions;
/// use rustmath_integers::Integer;
///
/// // gp: numbpart(100) = 190569292, numbpart(1000) = 24061467864032622473692149727991.
/// assert_eq!(number_of_partitions(100), Integer::from(190569292u32));
/// assert_eq!(
///     number_of_partitions(1000),
///     Integer::from_decimal_str("24061467864032622473692149727991").unwrap()
/// );
/// ```
pub fn number_of_partitions(n: u64) -> Integer {
    assert!(
        n <= NUMBER_OF_PARTITIONS_MAX_N,
        "number_of_partitions: n = {n} is above 10^16, outside the supported domain (the series \
         terms would overflow the MPFR exponent range near 8.4e16; the run would take at least \
         5e6 s at 10^16)"
    );
    if n <= EULER_MAX {
        // n <= 1000 fits a usize on every target.
        return partitions_euler(n as usize);
    }
    let e1 = first_term_log2_bound(n);
    assert!(
        e1 < f64::from(rug::float::exp_max()),
        "number_of_partitions: the MPFR exponent range ({}) is too small for n = {n}",
        rug::float::exp_max()
    );
    partitions_hrr(n)
}

/// An upper bound of log2 of the first term of the series, plus 64 bits of margin:
/// C / ln 2 + 64 with C = (pi/6) sqrt(24n - 1). log2 of the k = 1 term is below C / ln 2, and
/// 64 bits cover the f64 rounding many times over.
fn first_term_log2_bound(n: u64) -> f64 {
    let c = std::f64::consts::PI / 6.0 * (24.0 * n as f64 - 1.0).sqrt();
    c / std::f64::consts::LN_2 + 64.0
}

/// p(0), ..., p(n) by Euler's recurrence (Johansson (1.2)):
/// p(m) = sum_{k>=1} (-1)^(k+1) [p(m - k(3k-1)/2) + p(m - k(3k+1)/2)].
/// O(n^1.5) additions of numbers of O(n^(1/2)) bits.
fn euler_list(n: usize) -> Vec<Integer> {
    let mut p: Vec<Integer> = Vec::with_capacity(n + 1);
    p.push(Integer::one());
    for m in 1..=n {
        let mut s = Integer::zero();
        let mut k = 1usize;
        loop {
            let g1 = k * (3 * k - 1) / 2;
            if g1 > m {
                break;
            }
            let g2 = g1 + k;
            let mut add = p[m - g1].clone();
            if g2 <= m {
                add = add + p[m - g2].clone();
            }
            s = if k % 2 == 1 { s + add } else { s - add };
            k += 1;
        }
        p.push(s);
    }
    p
}

/// p(n) by Euler's recurrence.
fn partitions_euler(n: usize) -> Integer {
    euler_list(n).swap_remove(n)
}

/// A closed interval [lo, hi] of reals with MPFR endpoints. Every operation rounds the lower
/// end down and the upper end up, so the result contains the exact value whenever the inputs
/// contain theirs.
#[derive(Clone, Debug)]
struct Enclosure {
    lo: Float,
    hi: Float,
}

fn down<T>(prec: u32, v: T) -> Float
where
    Float: AssignRound<T, Round = Round, Ordering = Ordering>,
{
    Float::with_val_round(prec, v, Round::Down).0
}

fn up<T>(prec: u32, v: T) -> Float
where
    Float: AssignRound<T, Round = Round, Ordering = Ordering>,
{
    Float::with_val_round(prec, v, Round::Up).0
}

impl Enclosure {
    fn zero(prec: u32) -> Enclosure {
        Enclosure {
            lo: Float::with_val(prec, 0),
            hi: Float::with_val(prec, 0),
        }
    }

    /// The point [x, x] (x is exact).
    #[cfg(test)]
    fn point(x: Float) -> Enclosure {
        Enclosure { lo: x.clone(), hi: x }
    }

    /// [pi down, pi up].
    fn pi(prec: u32) -> Enclosure {
        Enclosure {
            lo: down(prec, Constant::Pi),
            hi: up(prec, Constant::Pi),
        }
    }

    /// sqrt(v) of an exact u64 (loaded exactly at 128 bits).
    fn sqrt_u64(prec: u32, v: u64) -> Enclosure {
        let x = Float::with_val(128, v);
        Enclosure {
            lo: down(prec, x.sqrt_ref()),
            hi: up(prec, x.sqrt_ref()),
        }
    }

    fn add(&self, o: &Enclosure, prec: u32) -> Enclosure {
        Enclosure {
            lo: down(prec, &self.lo + &o.lo),
            hi: up(prec, &self.hi + &o.hi),
        }
    }

    fn sub(&self, o: &Enclosure, prec: u32) -> Enclosure {
        Enclosure {
            lo: down(prec, &self.lo - &o.hi),
            hi: up(prec, &self.hi - &o.lo),
        }
    }

    /// Negation is exact.
    fn neg(&self) -> Enclosure {
        Enclosure {
            lo: -self.hi.clone(),
            hi: -self.lo.clone(),
        }
    }

    /// The four end products, each rounded down for the minimum and up for the maximum.
    fn mul(&self, o: &Enclosure, prec: u32) -> Enclosure {
        let ends = [(&self.lo, &o.lo), (&self.lo, &o.hi), (&self.hi, &o.lo), (&self.hi, &o.hi)];
        let mut lo = down(prec, ends[0].0 * ends[0].1);
        let mut hi = up(prec, ends[0].0 * ends[0].1);
        for &(a, b) in &ends[1..] {
            let l = down(prec, a * b);
            if l < lo {
                lo = l;
            }
            let h = up(prec, a * b);
            if h > hi {
                hi = h;
            }
        }
        Enclosure { lo, hi }
    }

    /// Times an exact positive integer m.
    fn mul_u64(&self, m: u64, prec: u32) -> Enclosure {
        debug_assert!(m > 0);
        Enclosure {
            lo: down(prec, &self.lo * m),
            hi: up(prec, &self.hi * m),
        }
    }

    /// Divided by an exact positive integer d.
    fn div_u64(&self, d: u64, prec: u32) -> Enclosure {
        debug_assert!(d > 0);
        Enclosure {
            lo: down(prec, &self.lo / d),
            hi: up(prec, &self.hi / d),
        }
    }

    /// cos on [a, b]: |cos t - cos a| <= |t - a| <= b - a (|cos'| <= 1).
    fn cos(&self, prec: u32) -> Enclosure {
        let cl = down(prec, self.lo.cos_ref());
        let ch = up(prec, self.lo.cos_ref());
        let w = up(prec, &self.hi - &self.lo);
        Enclosure {
            lo: down(prec, &cl - &w),
            hi: up(prec, &ch + &w),
        }
    }

    /// U(x) = cosh x - sinh(x)/x on [a, b] with a > 0. U is increasing on x > 0, since
    /// U(x) = sum_{m>=1} 2m x^(2m) / (2m+1)! has positive coefficients, so
    /// U([a, b]) = [U(a), U(b)].
    fn u(&self, prec: u32) -> Enclosure {
        debug_assert!(self.lo > 0);
        let c = down(prec, self.lo.cosh_ref());
        let s = up(prec, self.lo.sinh_ref());
        let q = up(prec, &s / &self.lo);
        let lo = down(prec, &c - &q);
        let c = up(prec, self.hi.cosh_ref());
        let s = down(prec, self.hi.sinh_ref());
        let q = down(prec, &s / &self.hi);
        let hi = up(prec, &c - &q);
        Enclosure { lo, hi }
    }

    fn is_exact_zero(&self) -> bool {
        self.lo.is_zero() && self.hi.is_zero()
    }
}

/// Rademacher's bound (1.8) in f64, only to choose a first N (a heuristic; the rigorous bound
/// is [`remainder_bound_upper`]).
fn remainder_bound_f64(n: u64, terms: u64) -> f64 {
    let (n, nn) = (n as f64, terms as f64);
    let pi = std::f64::consts::PI;
    44.0 * pi * pi / (225.0 * 3f64.sqrt()) / nn.sqrt()
        + pi * 2f64.sqrt() / 75.0 * (nn / (n - 1.0)).sqrt() * (pi / nn * (2.0 * n / 3.0).sqrt()).sinh()
}

/// An upper bound of Rademacher's M(n, N) (Johansson (1.8)), so |R(n, N)| < M(n, N) <= the
/// result:
/// M(n, N) = 44 pi^2 / (225 sqrt 3) N^(-1/2) + pi sqrt 2 / 75 (N / (n - 1))^(1/2)
/// sinh(pi / N sqrt(2n / 3)).
/// Evaluated at 64 bits with every operation rounded up; pi and sqrt 2 are rounded up and
/// sqrt 3 and sqrt N (in denominators) are rounded down. The integers N, n - 1 and 2n are held
/// exactly in 128-bit floats, so N / (n - 1) and 2n / 3 are each one division rounded up.
/// Needs n >= 2.
fn remainder_bound_upper(n: u64, terms: u64) -> Float {
    let p = 64;
    let pi_hi = up(p, Constant::Pi);
    let sqrt3_lo = down(p, Float::with_val(p, 3).sqrt_ref());
    let nn_exact = Float::with_val(128, terms);
    let sqrt_nn_lo = down(p, nn_exact.sqrt_ref());
    // a = 44 pi^2 / (225 sqrt 3) / sqrt N
    let a = up(p, &pi_hi * &pi_hi);
    let a = up(p, &a * 44u32);
    let d = down(p, &sqrt3_lo * 225u32);
    let a = up(p, &a / &d);
    let a = up(p, &a / &sqrt_nn_lo);
    // b = pi sqrt 2 / 75 * sqrt(N / (n - 1)) * sinh(pi / N * sqrt(2n / 3))
    let sqrt2_hi = up(p, Float::with_val(p, 2).sqrt_ref());
    let b = up(p, &pi_hi * &sqrt2_hi);
    let b = up(p, &b / 75u32);
    let nm1_exact = Float::with_val(128, n - 1);
    let ratio = up(p, &nn_exact / &nm1_exact);
    let sr = up(p, ratio.sqrt_ref());
    let two_n_exact = Float::with_val(128, 2 * u128::from(n));
    let t = up(p, &two_n_exact / 3u32);
    let st = up(p, t.sqrt_ref());
    let arg = up(p, &pi_hi * &st);
    let arg = up(p, &arg / terms);
    let sh = up(p, arg.sinh_ref());
    let b = up(p, &b * &sr);
    let b = up(p, &b * &sh);
    up(p, &a + &b)
}

/// The number of terms N and the bound Mb >= M(n, N), with Mb < 1/4: the first N with the f64
/// estimate below 0.2, then the first N from there whose rigorous bound is below 1/4.
fn choose_terms(n: u64) -> (u64, Float) {
    let mut terms = 1u64;
    while remainder_bound_f64(n, terms) >= 0.2 {
        terms += 1;
    }
    let quarter = Float::with_val(64, 0.25);
    let mut mb = remainder_bound_upper(n, terms);
    while mb >= quarter {
        terms += 1;
        mb = remainder_bound_upper(n, terms);
    }
    (terms, mb)
}

/// S_k = sum over 0 <= l < 2k with (3l^2 + l)/2 = -n (mod k) of (-1)^l cos((6l+1) pi / (6k)),
/// so that A_k(n) = (k/3)^(1/2) S_k (Selberg's formula, Johansson (2.1)). Algorithm 1 of the
/// paper: m runs over n + (3l^2 + l)/2 mod k and r over 3l + 2 mod k, each updated with one
/// addition and at most one subtraction of k (m, r < k and k >= 3); a cosine is evaluated only
/// when m = 0.
fn selberg_sum(n: u64, k: u64, pi: &Enclosure, prec: u32) -> Enclosure {
    debug_assert!(k >= 3);
    let mut s = Enclosure::zero(prec);
    let (mut r, mut m) = (2u64, n % k);
    for l in 0..2 * k {
        if m == 0 {
            // The angle (6l + 1) pi / (6k); 6l + 1 < 12k + 1 fits a u64.
            let th = pi.mul_u64(6 * l + 1, prec).div_u64(6 * k, prec);
            let c = th.cos(prec);
            s = if l % 2 == 0 { s.add(&c, prec) } else { s.sub(&c, prec) };
        }
        m += r;
        if m >= k {
            m -= k;
        }
        r += 3;
        if r >= k {
            r -= k;
        }
    }
    s
}

/// The digits of a rug integer as an [`Integer`] (the inverse of the conversion in
/// rustmath-reals). Linear in the size, unlike a decimal string.
fn rug_to_integer(z: &rug::Integer) -> Integer {
    let mut bytes = vec![0u8; z.significant_digits::<u8>()];
    z.write_digits(&mut bytes, rug::integer::Order::Lsf);
    let sign = match z.cmp0() {
        Ordering::Less => Sign::Minus,
        Ordering::Equal => Sign::NoSign,
        Ordering::Greater => Sign::Plus,
    };
    Integer::new(BigInt::from_bytes_le(sign, &bytes))
}

/// p(n) for 2 <= n <= 10^16 by the certified Hardy-Ramanujan-Rademacher series (the public
/// function uses it above 1000; the unit tests compare it with Euler's recurrence from 2):
/// t_k = 4 U(C/k) S_k / (24n - 1), p(n) = sum_{k=1}^{N} t_k + R(n, N), |R(n, N)| < Mb < 1/4
/// (Johansson (1.4), (1.5), (2.1) and (1.8); the square roots of (1.4) and (2.1) cancel).
fn partitions_hrr(n: u64) -> Integer {
    debug_assert!((2..=NUMBER_OF_PARTITIONS_MAX_N).contains(&n));
    let (terms, mb) = choose_terms(n);
    // v = 24n - 1 < 2.4e17 < 2^64 in the domain.
    let v = 24 * n - 1;
    let ln2 = std::f64::consts::LN_2;
    let c_f64 = std::f64::consts::PI / 6.0 * (v as f64).sqrt();
    let log2_v = (v as f64).log2();
    let bitlen_terms = u64::BITS - terms.leading_zeros();
    let mut guard = 32u32;
    for _ in 0..=MAX_ROUNDS {
        // log2 |t_k| < C/(k ln 2) + log2 k + 4 - log2 v (U(x) < e^x / 2, |S_k| <= 2k). In the
        // domain this is at most about 3.7e8 + 33000 bits, below u32::MAX and
        // rug::float::prec_max().
        let bits = |k: u64| -> u32 {
            let b = c_f64 / (k as f64) / ln2 + (k as f64).log2() + 4.0 - log2_v;
            let b = b.max(0.0).ceil() as u32;
            (b + bitlen_terms + guard + 16).max(64)
        };
        let prec_acc = bits(1) + 16;
        let pi = Enclosure::pi(prec_acc);
        // C = pi sqrt(24n - 1) / 6.
        let c = pi.mul(&Enclosure::sqrt_u64(prec_acc, v), prec_acc).div_u64(6, prec_acc);
        let mut acc = Enclosure::zero(prec_acc);
        for k in 1..=terms {
            let p = bits(k);
            let s = match k {
                // S_1 = sqrt 3 (A_1 = 1).
                1 => Enclosure::sqrt_u64(p, 3),
                // S_2 = (-1)^n sqrt(3/2) = (-1)^n sqrt(6) / 2 (A_2 = (-1)^n).
                2 => {
                    let r = Enclosure::sqrt_u64(p, 6).div_u64(2, p);
                    if n.is_multiple_of(2) {
                        r
                    } else {
                        r.neg()
                    }
                }
                _ => selberg_sum(n, k, &pi, p),
            };
            if s.is_exact_zero() {
                // No l satisfies the congruence: A_k(n) = 0.
                continue;
            }
            let x = c.div_u64(k, p);
            let t = x.u(p).mul(&s, p).mul_u64(4, p).div_u64(v, p);
            acc = acc.add(&t, prec_acc);
        }
        let lo = down(prec_acc, &acc.lo - &mb);
        let hi = up(prec_acc, &acc.hi + &mb);
        let l = lo.ceil().to_integer();
        let h = hi.floor().to_integer();
        if let (Some(l), Some(h)) = (l, h) {
            if l == h {
                return rug_to_integer(&l);
            }
        }
        guard = guard.saturating_mul(2);
    }
    panic!("number_of_partitions: internal error: the enclosure of p({n}) did not narrow");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::NumericConversion;

    #[test]
    fn hrr_equals_euler_up_to_3000() {
        let p = euler_list(3000);
        for n in 2..=3000u64 {
            assert_eq!(partitions_hrr(n), p[n as usize], "p({n})");
        }
    }

    #[test]
    fn remainder_bound_is_below_a_quarter() {
        let quarter = Float::with_val(64, 0.25);
        for n in [1001u64, 10_000, 1_000_000, 1_000_000_000] {
            let (terms, mb) = choose_terms(n);
            assert!(mb < quarter);
            assert!(remainder_bound_upper(n, terms) < quarter);
            if n >= 10_000 {
                // Johansson's N = ceil(n^(1/2)) is for the weaker target |R| < 1/2.
                let root = (n as f64).sqrt().ceil() as u64;
                assert!(terms <= root, "n = {n}: N = {terms} > ceil(sqrt(n)) = {root}");
            }
        }
    }

    #[test]
    fn enclosure_operations_contain_the_value() {
        // gp: \p 70; Pi
        let d = "3.141592653589793238462643383279502884197169399375105820974944592307816";
        let pi = Enclosure::pi(200);
        assert!(pi.lo < pi.hi);
        let w = Float::with_val(200, &pi.hi - &pi.lo);
        assert!(w <= Float::with_val(200, Float::i_exp(1, -196)));
        let dv = Float::with_val(300, Float::parse(d).unwrap());
        let eps = Float::with_val(300, Float::parse("1e-68").unwrap());
        assert!(pi.lo <= Float::with_val(300, &dv + &eps));
        assert!(pi.hi >= Float::with_val(300, &dv - &eps));

        let a = Enclosure { lo: Float::with_val(64, -1), hi: Float::with_val(64, 2) };
        let b = Enclosure { lo: Float::with_val(64, -3), hi: Float::with_val(64, 4) };
        let m = a.mul(&b, 64);
        assert_eq!(m.lo, -6);
        assert_eq!(m.hi, 8);

        let c = Enclosure::point(Float::with_val(64, 0)).cos(64);
        assert_eq!(c.lo, 1);
        assert_eq!(c.hi, 1);

        let u = Enclosure::point(Float::with_val(64, 3)).u(64);
        let three = Float::with_val(300, 3);
        let cosh3 = Float::with_val(300, three.cosh_ref());
        let sinh3 = Float::with_val(300, three.sinh_ref());
        let exact = Float::with_val(300, &cosh3 - Float::with_val(300, &sinh3 / 3u32));
        assert!(u.lo <= exact && exact <= u.hi);
        let width = Float::with_val(64, &u.hi - &u.lo);
        assert!(width < Float::with_val(64, Float::i_exp(1, -55)));
    }

    /// A_k(n) from its definition (Johansson (1.6)) with the Dedekind sums (1.7) as exact
    /// rationals: A_k(n) = sum_{0 <= h < k, gcd(h, k) = 1} cos(pi (s(h, k) - 2hn/k)), where
    /// s(h, k) = sum_{i=1}^{k-1} (i/k) (hi/k - floor(hi/k) - 1/2).
    fn a_k_dedekind(k: u64, n: u64) -> f64 {
        fn gcd(a: u64, b: u64) -> u64 {
            if b == 0 {
                a
            } else {
                gcd(b, a % b)
            }
        }
        let mut total = 0.0;
        for h in 0..k {
            if gcd(h, k) != 1 {
                continue;
            }
            // s(h, k) = num / (2 k^2), exactly: each term is i (2 (hi mod k) - k) / (2 k^2).
            let mut num: i64 = 0;
            for i in 1..k {
                num += (i as i64) * (2 * ((h * i) % k) as i64 - k as i64);
            }
            let s = num as f64 / (2.0 * (k * k) as f64);
            // 2hn/k reduced mod 2 (cos has period 2 pi).
            let t = ((2 * h * n) % (2 * k)) as f64 / k as f64;
            total += (std::f64::consts::PI * (s - t)).cos();
        }
        total
    }

    #[test]
    fn selberg_sum_matches_the_dedekind_definition() {
        let prec = 128;
        let pi = Enclosure::pi(prec);
        for k in 3..=12u64 {
            for n in 0..=40u64 {
                let s = selberg_sum(n, k, &pi, prec);
                let mid = Float::with_val(prec, &s.lo + &s.hi) / 2u32;
                let a = mid.to_f64() * (k as f64 / 3.0).sqrt();
                let want = a_k_dedekind(k, n);
                assert!((a - want).abs() < 1e-9, "A_{k}({n}): Selberg {a}, Dedekind {want}");
            }
        }
    }

    #[test]
    fn rug_integer_converts_through_digits() {
        let cases: Vec<rug::Integer> = vec![
            rug::Integer::new(),
            rug::Integer::from(1),
            rug::Integer::from(-1),
            rug::Integer::from(rug::Integer::u_pow_u(2, 64)),
            -(rug::Integer::from(rug::Integer::u_pow_u(2, 64)) + 5u32),
            rug::Integer::from(rug::Integer::u_pow_u(3, 100_000)),
        ];
        for z in &cases {
            let want = Integer::from_decimal_str(&z.to_string()).unwrap();
            assert_eq!(rug_to_integer(z), want);
        }
        assert_eq!(rug_to_integer(&rug::Integer::from(1)).to_u64(), Some(1));
    }

    #[test]
    fn exponent_range_covers_the_domain() {
        let e1 = first_term_log2_bound(NUMBER_OF_PARTITIONS_MAX_N);
        assert!(e1 < f64::from(rug::float::exp_max()), "{e1}");
        let c = std::f64::consts::PI / 6.0 * (24.0 * NUMBER_OF_PARTITIONS_MAX_N as f64 - 1.0).sqrt();
        let x = Float::with_val(64, c);
        let ch = up(64, x.cosh_ref());
        assert!(ch.is_finite());
    }
}
