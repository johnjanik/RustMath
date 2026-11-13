//! Lehmer's gcd and extended gcd on `BigUint` (crate-private).
//!
//! MAGMA Handbook ch. 18 §18.6 `Gcd`, `Xgcd` (digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`): MAGMA uses Weber's accelerated gcd
//! [Web95] and Lehmer's XGCD [Knu97, pp. 345-348] in the classical range, and Schönhage's
//! half-gcd [Sch71] above about 32,768 bits. This module is Lehmer's algorithm; there is no
//! half-gcd (with num-bigint's Toom-3 products its crossover is estimated at 10^5 to 10^6 bits).
//!
//! Algorithm: D. E. Knuth, *The Art of Computer Programming* vol. 2, 3rd ed., §4.5.2,
//! Algorithm L (D. H. Lehmer, Amer. Math. Monthly 45 (1938) 227-233). The leading 63 bits of
//! the pair `(u, v)` (at the same shift) are run through single-precision Euclid while both
//! bounds `floor((û + A) / (v̂ + C))` and `floor((û + B) / (v̂ + D))` agree. Every quotient
//! accepted that way is a true quotient of Euclid's algorithm on `(u, v)`, so the cofactor
//! matrix `[[A, B], [C, D]]` maps `(u, v)` to two consecutive remainders of Euclid's sequence
//! in one multiprecision step. When no quotient is certain (`B = 0`) a plain division step is
//! taken. The gcd and the Euclid cofactor of `a` are therefore exactly those of Euclid's
//! algorithm; only the cost changes (about 10x to 60x faster from 1,000 to 10^6 bits).
//!
//! References:
//! - [Knu97] D. E. Knuth, TAOCP vol. 2, 3rd ed., Addison-Wesley 1997, §4.5.2 (Algorithm L).
//! - [Web95] K. Weber, *The accelerated integer GCD algorithm*, ACM TOMS 21 (1995) 111-122.
//! - [Sch71] A. Schönhage, *Schnelle Berechnung von Kettenbruchentwicklungen*, Acta Inform. 1
//!   (1971) 139-144.

use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};

/// `floor(x / 2^s) mod 2^64`; exact when `x < 2^(s + 64)`. Reads only two 64-bit digits.
fn shr_u64(x: &BigUint, s: u64) -> u64 {
    let w = usize::try_from(s / 64).expect("a digit index of a BigUint fits in usize");
    let r = (s % 64) as u32; // < 64
    let mut it = x.iter_u64_digits().skip(w);
    let lo = it.next().unwrap_or(0);
    let hi = it.next().unwrap_or(0);
    if r == 0 {
        lo
    } else {
        (lo >> r) | (hi << (64 - r))
    }
}

/// Knuth Algorithm L, steps L2-L3, on the leading parts `uh >= vh` (both `< 2^63`): the
/// cofactor matrix `[[A, B], [C, D]]` of the Euclid quotients that are certain, that is
/// `q = floor((uh + A) / (vh + C)) = floor((uh + B) / (vh + D))`. Starts at the identity.
///
/// The simulated pair `(û, v̂)` is Euclid's sequence on `(uh, vh)`, and
/// `(uh, vh) = (|D|·û + |B|·v̂, |C|·û + |A|·v̂)` (the inverse matrix has determinant ±1 and
/// the sign pattern of the continuants), so every entry is at most `uh < 2^63` in absolute
/// value. The checked conversion turns a broken invariant into a panic, never a wrong value.
fn lehmer_matrix(uh: u64, vh: u64) -> (i64, i64, i64, i64) {
    let (mut a, mut b, mut c, mut d): (i128, i128, i128, i128) = (1, 0, 0, 1);
    let (mut u, mut v) = (i128::from(uh), i128::from(vh));
    loop {
        if v + c == 0 || v + d == 0 {
            break;
        }
        let q = (u + a) / (v + c);
        if q != (u + b) / (v + d) {
            break;
        }
        (a, c) = (c, a - q * c);
        (b, d) = (d, b - q * d);
        (u, v) = (v, u - q * v);
    }
    let f = |x: i128| i64::try_from(x).expect("lehmer_matrix: |entry| <= uh < 2^63 (Knuth 4.5.2)");
    (f(a), f(b), f(c), f(d))
}

/// `p·x + q·y` for a pair `(p, q)` of opposite signs, or with a zero entry, whose value is
/// known to be `>= 0` (a remainder of Euclid's sequence). A negative value would be a broken
/// invariant and panics in the `BigUint` subtraction.
fn combine(x: &BigUint, p: i64, y: &BigUint, q: i64) -> BigUint {
    match (p >= 0, q >= 0) {
        (true, true) => x * p.unsigned_abs() + y * q.unsigned_abs(),
        (true, false) => x * p.unsigned_abs() - y * q.unsigned_abs(),
        (false, true) => y * q.unsigned_abs() - x * p.unsigned_abs(),
        (false, false) => unreachable!("Lehmer cofactor rows have opposite signs (Knuth 4.5.2)"),
    }
}

/// `x` as a `u64` when it fits.
fn small(x: &BigUint) -> Option<u64> {
    if x.bits() <= 64 {
        x.to_u64()
    } else {
        None
    }
}

/// Euclid on 64-bit words.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// `gcd(a, b) >= 0`; `gcd(0, 0) = 0`. Lehmer's algorithm (Knuth §4.5.2 Algorithm L) with
/// 63-bit leading parts, single-precision Euclid once the larger operand fits in 64 bits.
/// An unbalanced pair (a small `v`) costs one division, as with Euclid.
pub(crate) fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
    if let (Some(x), Some(y)) = (small(a), small(b)) {
        return BigUint::from(gcd_u64(x, y));
    }
    let (mut u, mut v) = if a >= b {
        (a.clone(), b.clone())
    } else {
        (b.clone(), a.clone())
    };
    let g = loop {
        if v.is_zero() {
            break u;
        }
        if let Some(x) = small(&u) {
            let y = v.to_u64().expect("v <= u < 2^64");
            break BigUint::from(gcd_u64(x, y));
        }
        let s = u.bits() - 63;
        let (aa, bb, cc, dd) = lehmer_matrix(shr_u64(&u, s), shr_u64(&v, s));
        if bb == 0 {
            let r = &u % &v;
            u = v;
            v = r;
        } else {
            let nu = combine(&u, aa, &v, bb);
            let nv = combine(&u, cc, &v, dd);
            u = nu;
            v = nv;
        }
    };
    // Checked only for large operands: below 2^128 the two remainders would cost as much as
    // the gcd itself, and every Rational normalisation runs through here.
    debug_assert!(
        a.bits() <= 128 || b.bits() <= 128 || ((a % &g).is_zero() && (b % &g).is_zero()),
        "Lehmer gcd does not divide its inputs"
    );
    g
}

/// `(g, s)` with `g = gcd(a, b)` and `s` the Euclid cofactor of `a`: `g = s·a + t·b` for an
/// integer `t`. The cofactor is that of Euclid's algorithm on `(a, b)` with floor quotients
/// (a quotient 0 first when `a < b`); Lehmer's matrix steps only group its quotients.
pub(crate) fn gcd_cofactor(a: &BigUint, b: &BigUint) -> (BigUint, BigInt) {
    let (mut u, mut v) = (a.clone(), b.clone());
    // u = su·a + (..)·b and v = sv·a + (..)·b throughout.
    let (mut su, mut sv) = (BigInt::one(), BigInt::zero());
    loop {
        if v.is_zero() {
            return (u, su);
        }
        let matrix = if u < v || u.bits() <= 64 {
            None
        } else {
            let s = u.bits() - 63;
            let m = lehmer_matrix(shr_u64(&u, s), shr_u64(&v, s));
            (m.1 != 0).then_some(m)
        };
        match matrix {
            None => {
                let (q, r) = num_integer::Integer::div_rem(&u, &v);
                let ns = &su - BigInt::from(q) * &sv;
                u = v;
                v = r;
                su = sv;
                sv = ns;
            }
            Some((aa, bb, cc, dd)) => {
                let nu = combine(&u, aa, &v, bb);
                let nv = combine(&u, cc, &v, dd);
                let nsu = &su * aa + &sv * bb;
                let nsv = &su * cc + &sv * dd;
                u = nu;
                v = nv;
                su = nsu;
                sv = nsv;
            }
        }
    }
}

/// Extended Euclid on `(a, b)`, `a, b < 2^64`, with floor quotients in i128 (the cofactors
/// are bounded by `max(a, b)`, so nothing overflows).
fn xgcd_u64(a: u64, b: u64) -> (u64, i128, i128) {
    let (mut r0, mut r1) = (i128::from(a), i128::from(b));
    let (mut s0, mut s1, mut t0, mut t1) = (1i128, 0i128, 0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (s0, s1) = (s1, s0 - q * s1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    let g = u64::try_from(r0).expect("gcd of two u64 values fits in u64");
    (g, s0, t0)
}

/// `(g, s, t)` with `g = gcd(a, b) = s·a + t·b` and `(s, t)` the cofactors of Euclid's
/// algorithm on the magnitudes `(a, b)` (floor quotients). `(0, 0)` gives `(0, 1, 0)`; the
/// caller handles `(0, 0)` first. `t = (g - s·a) / b` is an exact division (`t = 0` when
/// `b = 0`).
pub(crate) fn xgcd_magnitudes(a: &BigUint, b: &BigUint) -> (BigUint, BigInt, BigInt) {
    if let (Some(x), Some(y)) = (small(a), small(b)) {
        let (g, s, t) = xgcd_u64(x, y);
        return (BigUint::from(g), BigInt::from(s), BigInt::from(t));
    }
    let (g, s) = gcd_cofactor(a, b);
    let t = if b.is_zero() {
        BigInt::zero()
    } else {
        let num = BigInt::from(g.clone()) - &s * BigInt::from(a.clone());
        let (t, r) = num_integer::Integer::div_rem(&num, &BigInt::from(b.clone()));
        debug_assert!(r.is_zero(), "g - s·a is a multiple of b");
        t
    };
    (g, s, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Euclid's remainder sequence and cofactors on `(u0, v0)`: entries `(r_i, s_i, t_i)` with
    /// `r_i = s_i·u0 + t_i·v0`.
    fn euclid_table(u0: u64, v0: u64) -> Vec<(i128, i128, i128)> {
        let mut out = vec![(i128::from(u0), 1, 0), (i128::from(v0), 0, 1)];
        while out[out.len() - 1].0 != 0 {
            let (r0, s0, t0) = out[out.len() - 2];
            let (r1, s1, t1) = out[out.len() - 1];
            let q = r0 / r1;
            out.push((r0 - q * r1, s0 - q * s1, t0 - q * t1));
        }
        out
    }

    /// The matrix of `lehmer_matrix(uh, vh)` maps `(uh, vh)` to two consecutive entries of
    /// Euclid's sequence, and its rows are Euclid's cofactors there. Returns the index.
    fn check_matrix(uh: u64, vh: u64) -> usize {
        let (a, b, c, d) = lehmer_matrix(uh, vh);
        let table = euclid_table(uh, vh);
        let (a, b, c, d) = (i128::from(a), i128::from(b), i128::from(c), i128::from(d));
        let j = table
            .iter()
            .position(|&(_, s, t)| s == a && t == b)
            .expect("row (A, B) is a Euclid cofactor pair");
        assert_eq!((table[j + 1].1, table[j + 1].2), (c, d), "row (C, D)");
        assert_eq!(a * i128::from(uh) + b * i128::from(vh), table[j].0);
        assert_eq!(c * i128::from(uh) + d * i128::from(vh), table[j + 1].0);
        assert_eq!((a * d - b * c).abs(), 1, "determinant ±1");
        j
    }

    #[test]
    fn lehmer_matrix_follows_euclid() {
        // F_92 = 7540113804746346429, F_91 = 4660046610375530309 (gp: fibonacci(92),
        // fibonacci(91)): all quotients are 1, so the entries are ±Fibonacci numbers and
        // Lehmer accepts most of the 90 steps.
        let (f92, f91) = (7_540_113_804_746_346_429u64, 4_660_046_610_375_530_309u64);
        let j = check_matrix(f92, f91);
        assert!(j >= 40, "only {j} certain steps on a Fibonacci pair");
        let (a, b, c, d) = lehmer_matrix(f92, f91);
        let fib: Vec<i64> = {
            let mut f = vec![0i64, 1];
            while f.len() < 93 {
                let n = f[f.len() - 1] + f[f.len() - 2];
                f.push(n);
            }
            f
        };
        for e in [a, b, c, d] {
            assert!(fib.contains(&e.abs()), "{e} is not a Fibonacci number");
        }
        // (2^63 - 1, F_92) and generic pairs.
        check_matrix((1u64 << 63) - 1, f92);
        check_matrix((1u64 << 63) - 1, 1);
        check_matrix(1u64 << 62, (1u64 << 62) - 1);
        let mut st = 7u64;
        for _ in 0..2000 {
            st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u = (st >> 1) | (1 << 62);
            st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let v = (st >> 1) % u;
            check_matrix(u, v);
        }
        // vh = 0: no certain quotient, the identity.
        assert_eq!(lehmer_matrix(1 << 62, 0), (1, 0, 0, 1));
    }

    #[test]
    fn combine_sign_cases() {
        let (x, y) = (BigUint::from(100u32), BigUint::from(30u32));
        assert_eq!(combine(&x, 0, &y, 1), y);
        assert_eq!(combine(&x, 2, &y, 3), BigUint::from(290u32));
        assert_eq!(combine(&x, 1, &y, -3), BigUint::from(10u32));
        assert_eq!(combine(&x, -1, &y, 4), BigUint::from(20u32));
        assert_eq!(
            combine(&x, i64::MAX, &y, 0),
            BigUint::from(100u32) * i64::MAX.unsigned_abs()
        );
    }

    #[test]
    fn gcd_and_cofactor_small_and_structured() {
        // gp: gcd(2^200-1, 2^120-1) -> 2^40-1; gcd(0, 2^100) -> 2^100
        let two = BigUint::from(2u32);
        let m = |e: u32| two.pow(e) - 1u32;
        assert_eq!(gcd(&m(200), &m(120)), m(40));
        assert_eq!(gcd(&m(120), &m(200)), m(40));
        assert_eq!(gcd(&BigUint::zero(), &two.pow(100)), two.pow(100));
        assert_eq!(gcd(&two.pow(100), &BigUint::zero()), two.pow(100));
        assert_eq!(gcd(&BigUint::zero(), &BigUint::zero()), BigUint::zero());
        assert_eq!(
            gcd(&(two.pow(1000) + 1u32), &BigUint::one()),
            BigUint::one()
        );
        for (a, b) in [(m(200), m(120)), (m(120), m(200)), (two.pow(300), m(70))] {
            let (g, s, t) = xgcd_magnitudes(&a, &b);
            assert_eq!(
                BigInt::from(g.clone()),
                &s * BigInt::from(a.clone()) + &t * BigInt::from(b.clone())
            );
            assert_eq!(g, gcd(&a, &b));
        }
    }
}
