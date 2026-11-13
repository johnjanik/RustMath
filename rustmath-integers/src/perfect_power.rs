//! Exact k-th roots and perfect-power tests on `BigUint` (crate-private).
//!
//! MAGMA Handbook ch. 18 §18.4.5 `IsPower(n)` and `IsPower(n, k)` (digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`), used by [`crate::Integer::exact_root`],
//! [`crate::Integer::is_power`] and [`crate::Integer::is_perfect_power`].
//!
//! Exact odd roots go through the 2-adic root. Write `n = 2^v·m` with `m` odd. For odd
//! `k >= 3` the map `x -> x^k` is a bijection on the odd residues modulo `2^K` (their group
//! has order `2^(K-1)`, prime to `k`), so `m` has exactly one odd k-th root `r` modulo `2^K`.
//! Newton's iteration for `a^(-1/k)` doubles the 2-adic precision at every step. If
//! `m = c^k`, then `c` is odd with exactly `K = ceil(bits(m) / k)` bits, so `c = r`. Hence
//! `m` is a k-th power iff `r^k = m`. Every "no" below is a proof: `k ∤ v`, `bits(r) != K`,
//! a residue mismatch modulo one of two word primes (an equality `r^k = m` would make every
//! residue agree), or the exact power differs. No bounded search is involved.
//!
//! References:
//! - [Ber98] D. J. Bernstein, *Detecting perfect powers in essentially linear time*, Math.
//!   Comp. 67 (1998) 1253-1283, §1 (`n = b^k` with `b >= 2` forces `k <= log2 n`) and the
//!   2-adic root.
//! - [BS93] E. Bach, J. Sorenson, *Sieve algorithms for perfect power testing*, Algorithmica 9
//!   (1993) 313-328 (a q-th power is 0 or a q-th power residue modulo every prime p ≡ 1 mod q).
//! - [vzGG99] J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, Cambridge 1999, ch. 9
//!   (Newton iteration and p-adic lifting).
//! - D. E. Knuth, *The Art of Computer Programming* vol. 2, 3rd ed., §4.6.3 (powering).

use crate::fast_arith::pow_mod_u64;
use crate::primality::is_prime_u64;
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};

/// 2^61 - 1 (prime): the first fingerprint modulus.
const FP1: u64 = (1 << 61) - 1;
/// 2^64 - 59 = precprime(2^64): the second fingerprint modulus.
const FP2: u64 = 18_446_744_073_709_551_557;

/// Primes `p ≡ 1 (mod q)` that the power-residue sieve tries per exponent before the caller
/// falls back to the exact root.
pub(crate) const POWER_SIEVE_PRIMES: u32 = 4;
/// Candidates `p = 1 + j·lcm(2, q)` the sieve examines per exponent while looking for those
/// primes. With `q < 2^32` every candidate is below `2^43`.
pub(crate) const POWER_SIEVE_CANDIDATES: u64 = 1000;

/// Power-residue sieve for q-th powers ([BS93]). `false` is a proof that `n` is not a q-th
/// power: for a prime `p ≡ 1 (mod q)` with `p ∤ n`, `n = d^q` forces `p ∤ d` and
/// `n^((p-1)/q) = d^(p-1) ≡ 1 (mod p)` (Fermat), so any other residue refutes it. `true` only
/// says that none of the (at most [`POWER_SIEVE_PRIMES`]) tried primes refuted it; the caller
/// must still take the exact root. Primes dividing `n` give no information and count as tried.
pub(crate) fn may_be_qth_power(n: &BigUint, q: u64) -> bool {
    debug_assert!((2..1 << 32).contains(&q));
    let step = if q.is_multiple_of(2) { q } else { 2 * q }; // p = 1 + j·step is odd and ≡ 1 (mod q)
    let mut p = 1u64;
    let mut tried = 0;
    for _ in 0..POWER_SIEVE_CANDIDATES {
        p += step;
        if !is_prime_u64(p) {
            continue;
        }
        let r = (n % p).to_u64().expect("a residue modulo a u64 fits a u64");
        if r != 0 && pow_mod_u64(r, (p - 1) / q, p) != 1 {
            return false;
        }
        tried += 1;
        if tried == POWER_SIEVE_PRIMES {
            break;
        }
    }
    true
}

/// `n = 2^v · m` with `m` odd, and `m mod FP1`, `m mod FP2` (computed once per `n`, O(bits)).
pub(crate) struct OddPart {
    v: u64,
    m: BigUint,
    fp: [u64; 2],
}

/// The odd part of `n >= 1` with its fingerprints.
pub(crate) fn odd_part(n: &BigUint) -> OddPart {
    let v = n.trailing_zeros().expect("odd_part: n >= 1");
    let m = n >> v;
    let fp = [
        (&m % FP1)
            .to_u64()
            .expect("a residue modulo a u64 fits a u64"),
        (&m % FP2)
            .to_u64()
            .expect("a residue modulo a u64 fits a u64"),
    ];
    OddPart { v, m, fp }
}

/// `2^bits - 1`.
fn mask(bits: u64) -> BigUint {
    (BigUint::one() << bits) - 1u32
}

/// `x^e mod 2^t` (`t >= 1`), every product truncated to `t` bits.
fn pow_trunc(x: &BigUint, mut e: u64, t: u64) -> BigUint {
    let m = mask(t);
    let mut b = x & &m;
    let mut r = BigUint::one();
    while e > 0 {
        if e & 1 == 1 {
            r = (&r * &b) & &m;
        }
        e >>= 1;
        if e > 0 {
            b = (&b * &b) & &m;
        }
    }
    r
}

/// `x^e mod 2^64`.
fn pow_wrap(x: u64, mut e: u64) -> u64 {
    let (mut b, mut r) = (x, 1u64);
    while e > 0 {
        if e & 1 == 1 {
            r = r.wrapping_mul(b);
        }
        b = b.wrapping_mul(b);
        e >>= 1;
    }
    r
}

/// `k^(-1) mod 2^64` for odd `k`: `z <- z(2 - kz)` from `z = k`, which is correct modulo 8
/// (`k^2 ≡ 1 mod 8`); five steps reach 96 >= 64 bits.
fn inv_u64(k: u64) -> u64 {
    let mut z = k;
    for _ in 0..5 {
        z = z.wrapping_mul(2u64.wrapping_sub(k.wrapping_mul(z)));
    }
    z
}

/// The unique odd `r < 2^prec` with `r^k ≡ a (mod 2^prec)`, for odd `a`, odd `k >= 3` and
/// `prec >= 1`.
///
/// Newton's iteration for `y = a^(-1/k)`: `y <- y + y·(1 - a·y^k)·k^(-1)`. If
/// `a·y^k = 1 - e` with `e ≡ 0 (mod 2^j)`, then `y' = y(1 + e/k)` gives
/// `a·y'^k = (1 - e)(1 + e + O(e^2)) = 1 + O(e^2)`: the binomial terms `C(k, i)(e/k)^i`,
/// `i >= 2`, have 2-adic valuation `>= 2j` because `k` is a 2-adic unit. So the precision
/// doubles, also from `y = 1` (correct modulo 2). Six steps in wrapping u64 arithmetic give
/// 64 bits; above that `k^(-1)` and `y` are lifted with products truncated to the current
/// precision. Then `r = a·y^(k-1)`, so `r^k = a^k·a^(-(k-1)) = a`.
fn root_2adic(a: &BigUint, k: u64, prec: u64) -> BigUint {
    debug_assert!(k >= 3 && k % 2 == 1 && a.bit(0) && prec >= 1);
    let a0 = a.iter_u64_digits().next().unwrap_or(0);
    let kinv0 = inv_u64(k);
    let mut y0: u64 = 1;
    for _ in 0..6 {
        let e = 1u64.wrapping_sub(a0.wrapping_mul(pow_wrap(y0, k)));
        y0 = y0.wrapping_add(y0.wrapping_mul(e).wrapping_mul(kinv0));
    }
    let r = if prec <= 64 {
        let r = a0.wrapping_mul(pow_wrap(y0, k - 1));
        BigUint::from(if prec == 64 {
            r
        } else {
            r & ((1u64 << prec) - 1)
        })
    } else {
        let mut kinv = BigUint::from(kinv0);
        let mut cur = 64;
        while cur < prec {
            cur = (2 * cur).min(prec);
            let m = mask(cur);
            let kz = (&kinv * k) & &m;
            let two_minus = ((BigUint::one() << cur) + 2u32 - kz) & &m;
            kinv = (&kinv * two_minus) & &m;
        }
        let mut y = BigUint::from(y0);
        let mut cur = 64;
        while cur < prec {
            cur = (2 * cur).min(prec);
            let m = mask(cur);
            let ayk = ((a & &m) * pow_trunc(&y, k, cur)) & &m;
            // e = 1 - a·y^k mod 2^cur, formed as 2^cur + 1 - ayk (ayk < 2^cur).
            let e = ((&m + 2u32) - ayk) & &m;
            y = (&y + (((&y * e) & &m) * &kinv)) & &m;
        }
        let m = mask(prec);
        ((a & &m) * pow_trunc(&y, k - 1, prec)) & &m
    };
    debug_assert!(
        pow_trunc(&r, k, prec) == (a & &mask(prec)),
        "root_2adic: r^k != a mod 2^prec"
    );
    r
}

/// For `n = 2^v·m >= 2` (given by its [`OddPart`]) and odd `k >= 3`: `Some(b)` with
/// `b^k = n`, or `None`, which is a proof that `n` is not a k-th power (module docs).
pub(crate) fn exact_root_odd(o: &OddPart, k: u64) -> Option<BigUint> {
    debug_assert!(k >= 3 && k % 2 == 1);
    if !o.v.is_multiple_of(k) {
        return None;
    }
    if o.m.is_one() {
        return Some(BigUint::one() << (o.v / k));
    }
    let bits = o.m.bits();
    let kk = bits.div_ceil(k);
    if kk <= 1 {
        return None; // the only odd 1-bit root is 1, and m > 1
    }
    let r = root_2adic(&o.m, k, kk);
    if r.bits() != kk {
        return None;
    }
    for (i, p) in [FP1, FP2].into_iter().enumerate() {
        let rp = (&r % p)
            .to_u64()
            .expect("a residue modulo a u64 fits a u64");
        if pow_mod_u64(rp, k, p) != o.fp[i] {
            return None;
        }
    }
    let k32 = u32::try_from(k).expect("exact_root_odd: k < bits(n) fits in u32");
    if r.pow(k32) != o.m {
        return None;
    }
    Some(r << (o.v / k))
}

/// For `n >= 2` and `k >= 1`: `Some(b)` with `b^k = n` (`b >= 2`), or `None` when `n` is not a
/// k-th power. Write `k = 2^e·k'` with `k'` odd: the k'-th root by [`exact_root_odd`], then
/// `e` square roots, each preceded by the quadratic-residue sieve and checked by `s·s = x`.
pub(crate) fn exact_root_biguint(n: &BigUint, k: u32) -> Option<BigUint> {
    debug_assert!(n.bits() >= 2 && k >= 1);
    let e = k.trailing_zeros();
    let k_odd = u64::from(k >> e);
    let mut r = if k_odd > 1 {
        exact_root_odd(&odd_part(n), k_odd)?
    } else {
        n.clone()
    };
    for _ in 0..e {
        if !may_be_qth_power(&r, 2) {
            return None;
        }
        let s = r.sqrt();
        if &s * &s != r {
            return None;
        }
        r = s;
    }
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    #[test]
    fn power_residue_sieve_never_refutes_a_power() {
        // Soundness of the [BS93] sieve: d^q is never ruled out, including d divisible by the
        // sieve primes themselves (q = 2: 3, 5, 7, 11; q = 3: 7, 13, 19, 31; q = 5: 11, 31,
        // 41, 61), where the residue is 0 and gives no information.
        let mut ds: Vec<BigUint> = (1u32..=300).map(BigUint::from).collect();
        ds.push(BigUint::from(3u64 * 5 * 7 * 11 * 13 * 19 * 31 * 41 * 61));
        ds.push((BigUint::from(1u32) << 61u32) - 1u32);
        ds.push(BigUint::from(10u32).pow(40) + 7u32);
        for q in [2u32, 3, 5, 7, 11, 13, 31, 101, 1009] {
            for d in &ds {
                assert!(
                    may_be_qth_power(&d.pow(q), u64::from(q)),
                    "sieve refuted {d}^{q}"
                );
            }
        }
        // A refutation is a proof; spot-check that the sieve does refute: 2 is not a square
        // modulo 3, and 3 is not a cube modulo 7 (3^2 = 2 mod 7).
        assert!(!may_be_qth_power(&BigUint::from(2u32), 2));
        assert!(!may_be_qth_power(&BigUint::from(3u32), 3));
    }

    #[test]
    fn exact_root_odd_exhaustive_small() {
        // Every n in 2..=20000 and odd k in 3..=15 against the floor root of num-bigint.
        for n in 2u32..=20_000 {
            let nb = BigUint::from(n);
            let o = odd_part(&nb);
            for k in (3u32..=15).step_by(2) {
                let t = nb.nth_root(k);
                let want = (t.pow(k) == nb).then_some(t);
                assert_eq!(exact_root_odd(&o, u64::from(k)), want, "n = {n}, k = {k}");
            }
        }
    }

    #[test]
    fn root_2adic_is_a_root() {
        let mut st = 31u64;
        let mut next = || {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            st
        };
        for _ in 0..20 {
            let words: Vec<u64> = (0..17).map(|_| next()).collect();
            let mut a = BigUint::from_slice(
                &words
                    .iter()
                    .flat_map(|w| [*w as u32, (*w >> 32) as u32])
                    .collect::<Vec<_>>(),
            );
            a.set_bit(0, true);
            for k in [3u64, 5, 7, 63, 65, 1001] {
                for prec in [1u64, 2, 63, 64, 65, 128, 1000] {
                    let r = root_2adic(&a, k, prec);
                    assert!(r.bits() <= prec && r.bit(0), "odd residue below 2^prec");
                    assert_eq!(
                        pow_trunc(&r, k, prec),
                        &a & &mask(prec),
                        "k = {k}, prec = {prec}"
                    );
                }
            }
        }
    }

    #[test]
    fn exact_root_biguint_composite_exponents() {
        // 1234567^288 (288 = 2^5·9), 2^3000 with k = 3 and its odd neighbour.
        let b = BigUint::from(1_234_567u32);
        assert_eq!(exact_root_biguint(&b.pow(288), 288), Some(b.clone()));
        assert_eq!(exact_root_biguint(&b.pow(288), 96), Some(b.pow(3)));
        assert_eq!(exact_root_biguint(&(b.pow(288) + 1u32), 288), None);
        let two = BigUint::from(2u32);
        assert_eq!(exact_root_biguint(&two.pow(3000), 3), Some(two.pow(1000)));
        assert_eq!(exact_root_biguint(&two.pow(3001), 3), None);
        assert_eq!(exact_root_biguint(&two.pow(3000), 1), Some(two.pow(3000)));
        assert_eq!(exact_root_biguint(&BigUint::from(2u32), 2), None);
    }
}
