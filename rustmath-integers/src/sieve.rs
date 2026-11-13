//! Segmented sieve of Eratosthenes on 64-bit words (MAGMA Handbook ch. 18 §18.9.2 `PrimesUpTo`,
//! `PrimesInInterval`, `NthPrime`; digest `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`).
//!
//! Algorithm: C. Bays, R. H. Hudson, *The segmented sieve of Eratosthenes and primes in
//! arithmetic progressions to 10^12*, BIT 17 (1977) 121-127. Only odd numbers are stored, one
//! byte each, in segments of 2^15 odd numbers (32 KiB). The base primes go up to
//! L = min(isqrt(e), 2^20) for the interval [b, e]; each base prime p keeps its next odd multiple
//! (starting at max(b, p^2)) from one segment to the next. When isqrt(e) > 2^20 (that is
//! e > 2^40 roughly), a survivor above 2^40 can be a product of primes above 2^20, so those
//! survivors are confirmed with the deterministic [`crate::primality::is_prime_u64`]. Below 2^40
//! no survivor is tested.
//!
//! The upper bound for the n-th prime is J. B. Rosser, L. Schoenfeld, *Approximate formulas for
//! some functions of prime numbers*, Illinois J. Math. 6 (1962) 64-94, (3.13):
//! p_n < n (ln n + ln ln n) for n >= 6. It only chooses the end of the sieve; if the count falls
//! short (it cannot, up to float rounding, which a margin covers) the sieve goes on.
//!
//! All interval arithmetic is overflow-free up to `u64::MAX`: the internal form is the inclusive
//! interval [b, e], segment ends saturate, multiples are tracked in `u128`.

use std::ops::ControlFlow;
use std::sync::OnceLock;

/// Odd numbers per segment (one byte each).
const SEG: u64 = 1 << 15;

/// Largest base prime the sieve uses; survivors above `BASE_LIMIT^2` are confirmed.
const BASE_LIMIT: u64 = 1 << 20;

/// Floor of the square root of `n`, exact (float start, then integer correction).
pub(crate) fn isqrt_u64(n: u64) -> u64 {
    let mut r = (n as f64).sqrt() as u64;
    while u128::from(r) * u128::from(r) > u128::from(n) {
        r -= 1;
    }
    while u128::from(r + 1) * u128::from(r + 1) <= u128::from(n) {
        r += 1;
    }
    r
}

/// The primes up to `n` (`n` small) by the plain sieve of Eratosthenes on odd numbers.
fn plain_sieve(n: u64) -> Vec<u32> {
    let mut out = Vec::new();
    if n < 2 {
        return out;
    }
    out.push(2);
    let n = n as usize;
    // odd[i] stands for 2i + 1.
    let mut composite = vec![false; n / 2 + 1];
    let mut i = 1usize;
    while (2 * i + 1) * (2 * i + 1) <= n {
        if !composite[i] {
            let p = 2 * i + 1;
            let mut j = p * p;
            while j <= n {
                composite[j / 2] = true;
                j += 2 * p;
            }
        }
        i += 1;
    }
    for (i, &c) in composite.iter().enumerate().skip(1) {
        let v = 2 * i + 1;
        if v > n {
            break;
        }
        if !c {
            out.push(v as u32);
        }
    }
    out
}

/// The primes below 2^16 (6542 primes, 2 included), computed once per process. Used for trial
/// division and by the ECPP prover to strip small factors.
pub(crate) fn small_primes() -> &'static [u32] {
    static SMALL: OnceLock<Vec<u32>> = OnceLock::new();
    SMALL.get_or_init(|| plain_sieve((1 << 16) - 1))
}

/// The primes up to 2^20, computed once per process when a sieve needs base primes beyond 2^16.
fn base_primes() -> &'static [u32] {
    static BASE: OnceLock<Vec<u32>> = OnceLock::new();
    BASE.get_or_init(|| plain_sieve(BASE_LIMIT))
}

/// The odd base primes up to `limit` (at most 2^20).
fn odd_base_primes(limit: u64) -> &'static [u32] {
    let all: &'static [u32] = if limit < 1 << 16 {
        small_primes()
    } else {
        base_primes()
    };
    let end = all.partition_point(|&p| u64::from(p) <= limit);
    &all[1.min(end)..end]
}

/// Calls `f(p)` for each prime p in the inclusive interval [b, e] in ascending order, until `f`
/// returns `Break`. Returns `Break` iff `f` did.
fn for_each_prime_inclusive(
    b: u64,
    e: u64,
    f: &mut dyn FnMut(u64) -> ControlFlow<()>,
) -> ControlFlow<()> {
    if e < 2 || b > e {
        return ControlFlow::Continue(());
    }
    if b <= 2 {
        f(2)?;
    }
    // First odd number >= max(b, 3).
    let mut lo = b.max(3);
    if lo.is_multiple_of(2) {
        match lo.checked_add(1) {
            Some(v) => lo = v,
            None => return ControlFlow::Continue(()),
        }
    }
    if lo > e {
        return ControlFlow::Continue(());
    }
    let root = isqrt_u64(e);
    let limit = root.min(BASE_LIMIT);
    let confirm = root > BASE_LIMIT;
    let primes = odd_base_primes(limit);
    // next[i]: the next odd multiple of primes[i] to cross out (u128: may pass u64::MAX).
    let mut next: Vec<u128> = Vec::with_capacity(primes.len());
    let mut active = 0usize;
    let mut buf = vec![false; SEG as usize];
    let mut seg_lo = lo;
    loop {
        // Odd numbers seg_lo, seg_lo + 2, ..., seg_hi, with seg_hi <= e.
        let seg_hi = seg_lo.saturating_add(2 * (SEG - 1)).min(e);
        let seg_hi = if (seg_hi - seg_lo) % 2 == 1 {
            seg_hi - 1
        } else {
            seg_hi
        };
        let len = ((seg_hi - seg_lo) / 2 + 1) as usize;
        buf[..len].iter_mut().for_each(|x| *x = false);
        // Activate the base primes with p^2 <= seg_hi.
        while active < primes.len() {
            let p = u64::from(primes[active]);
            if p > seg_hi / p {
                break;
            }
            let p128 = u128::from(p);
            let start = (p128 * p128).max(u128::from(seg_lo));
            let mut m = start.div_ceil(p128) * p128;
            if m % 2 == 0 {
                m += p128;
            }
            next.push(m);
            active += 1;
        }
        let hi128 = u128::from(seg_hi);
        let lo128 = u128::from(seg_lo);
        for (i, nx) in next.iter_mut().enumerate() {
            let step = 2 * u128::from(primes[i]);
            let mut m = *nx;
            while m <= hi128 {
                buf[((m - lo128) / 2) as usize] = true;
                m += step;
            }
            *nx = m;
        }
        for (i, &crossed) in buf[..len].iter().enumerate() {
            if !crossed {
                let n = seg_lo + 2 * i as u64;
                if confirm && n > BASE_LIMIT * BASE_LIMIT && !crate::primality::is_prime_u64(n) {
                    continue;
                }
                f(n)?;
            }
        }
        if seg_hi >= e.saturating_sub(1) {
            // The last odd number <= e is in this segment.
            return ControlFlow::Continue(());
        }
        seg_lo = seg_hi + 2;
    }
}

/// Calls `f(p)` for each prime `lo <= p < hi` in ascending order until `f` returns
/// `ControlFlow::Break`. Memory is O(1) in the length of the range (one 32 KiB segment and the
/// base primes up to min(sqrt(hi), 2^20)).
pub fn for_each_prime(lo: u64, hi: u64, mut f: impl FnMut(u64) -> ControlFlow<()>) {
    if hi > lo {
        let _ = for_each_prime_inclusive(lo, hi - 1, &mut f);
    }
}

/// The primes p with `lo <= p < hi`, ascending.
pub fn primes_in_range(lo: u64, hi: u64) -> Vec<u64> {
    let mut out = Vec::new();
    for_each_prime(lo, hi, |p| {
        out.push(p);
        ControlFlow::Continue(())
    });
    out
}

/// MAGMA `PrimesUpTo(B)`: the primes p with 2 <= p <= b, ascending.
pub fn primes_up_to(b: u64) -> Vec<u64> {
    primes_in_interval(0, b)
}

/// MAGMA `PrimesInInterval(b, e)`: the primes p with b <= p <= e, ascending (e = `u64::MAX` is
/// allowed).
pub fn primes_in_interval(b: u64, e: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let _ = for_each_prime_inclusive(b, e, &mut |p| {
        out.push(p);
        ControlFlow::Continue(())
    });
    out
}

/// pi(x), the number of primes p <= x, by the segmented sieve. Cost: about 0.14 s for x = 10^8
/// in a release build (0.75 s in debug), linear in x.
pub fn prime_count(x: u64) -> u64 {
    let mut count = 0u64;
    let _ = for_each_prime_inclusive(0, x, &mut |_| {
        count += 1;
        ControlFlow::Continue(())
    });
    count
}

/// pi(2^64) = 425656284035217743 (OEIS A007053): no n-th prime with a larger n fits a `u64`.
pub const PRIME_PI_2_64: u64 = 425_656_284_035_217_743;

/// Calls `f(p)` for the first n primes in ascending order, or for all primes below 2^64 when
/// there are fewer (n > pi(2^64)). The sieve runs to the Rosser-Schoenfeld bound
/// n (ln n + ln ln n) (n >= 6), with a relative margin for the float evaluation; if the count
/// were short it would go on.
fn visit_first_n(n: u64, f: &mut dyn FnMut(u64)) {
    if n == 0 {
        return;
    }
    let bound = if n < 6 {
        15u64
    } else {
        let x = n as f64;
        let b = x * (x.ln() + x.ln().ln());
        let b = b * (1.0 + 1e-9) + 16.0;
        if b >= 1.8e19 {
            u64::MAX
        } else {
            b as u64
        }
    };
    let mut count = 0u64;
    let mut from = 0u64;
    let mut to = bound;
    loop {
        let flow = for_each_prime_inclusive(from, to, &mut |p| {
            count += 1;
            f(p);
            if count == n {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        if flow.is_break() || to == u64::MAX {
            return;
        }
        from = to + 1;
        to = to.saturating_mul(2);
    }
}

/// The first n primes, ascending (fewer when n > pi(2^64), which no machine reaches).
pub(crate) fn first_n_primes(n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    visit_first_n(n, &mut |p| out.push(p));
    out
}

/// MAGMA `NthPrime(n)`: the n-th prime (p_1 = 2), or `None` for n = 0 and for n > pi(2^64)
/// (p_n >= 2^64). Linear in p_n: `nth_prime_u64(10^7)` = 179424673 takes about 1.5 s in a
/// debug build.
pub fn nth_prime_u64(n: u64) -> Option<u64> {
    if n == 0 || n > PRIME_PI_2_64 {
        return None;
    }
    let mut count = 0u64;
    let mut last = None;
    visit_first_n(n, &mut |p| {
        count += 1;
        last = Some(p);
    });
    if count == n {
        last
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primality::is_prime_u64;

    fn filter(lo: u64, hi_inclusive: u64) -> Vec<u64> {
        (lo..=hi_inclusive).filter(|&n| is_prime_u64(n)).collect()
    }

    #[test]
    fn isqrt_edges() {
        assert_eq!(isqrt_u64(0), 0);
        assert_eq!(isqrt_u64(15), 3);
        assert_eq!(isqrt_u64(16), 4);
        assert_eq!(isqrt_u64(u64::MAX), u64::from(u32::MAX));
        assert_eq!(isqrt_u64((1u64 << 32) * (1u64 << 30)), 1u64 << 31);
    }

    #[test]
    fn small_primes_count() {
        // gp: primepi(2^16) = 6542, prime(6542) = 65521
        let s = small_primes();
        assert_eq!(s.len(), 6542);
        assert_eq!(s[0], 2);
        assert_eq!(*s.last().unwrap(), 65521);
    }

    /// The sieve against the deterministic word test (`is_prime_u64`, a proof for every u64).
    #[test]
    fn sieve_matches_is_prime_u64() {
        for lo in [0u64, 1_000_000_000, (1 << 32) - 5000, 1 << 40, 1 << 50] {
            assert_eq!(
                primes_in_range(lo, lo + 10_000),
                filter(lo, lo + 9_999),
                "lo = {lo}"
            );
        }
        // Near 2^64: lo + 10^4 = 2^64 does not fit, so use the inclusive form.
        // gp: precprime(2^64-1) = 18446744073709551557
        let lo = u64::MAX - 9_999;
        let got = primes_in_interval(lo, u64::MAX);
        assert_eq!(got, filter(lo, u64::MAX));
        assert_eq!(*got.last().unwrap(), 18_446_744_073_709_551_557);
        assert!(primes_in_interval(u64::MAX - 1, u64::MAX).is_empty());
        // gp: primes([2^64-101, 2^64-2]) = [18446744073709551521, 18446744073709551533,
        // 18446744073709551557]
        let mut seen = Vec::new();
        for_each_prime(u64::MAX - 100, u64::MAX, |p| {
            seen.push(p);
            ControlFlow::Continue(())
        });
        assert_eq!(
            seen,
            vec![
                18_446_744_073_709_551_521,
                18_446_744_073_709_551_533,
                18_446_744_073_709_551_557
            ]
        );
    }

    #[test]
    fn interval_edges_and_break() {
        assert_eq!(primes_up_to(1), Vec::<u64>::new());
        assert_eq!(primes_up_to(2), vec![2]);
        assert_eq!(primes_up_to(30), vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
        assert_eq!(primes_in_interval(3, 3), vec![3]);
        assert_eq!(primes_in_interval(4, 4), Vec::<u64>::new());
        assert_eq!(primes_in_interval(9, 2), Vec::<u64>::new());
        assert_eq!(primes_in_range(17, 18), vec![17]);
        assert_eq!(primes_in_range(18, 17), Vec::<u64>::new());
        // A range spanning several segments (2^15 odd numbers each).
        // gp: #primes([10^6, 13*10^5]) = 21523
        assert_eq!(primes_in_interval(1_000_000, 1_300_000).len(), 21523);
        let mut first = Vec::new();
        for_each_prime(0, 1000, |p| {
            first.push(p);
            if first.len() == 3 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(first, vec![2, 3, 5]);
    }

    #[test]
    fn counts_and_nth() {
        // gp: primepi(10^6) = 78498, prime(10^5) = 1299709, prime(1) .. prime(6)
        assert_eq!(prime_count(0), 0);
        assert_eq!(prime_count(2), 1);
        assert_eq!(prime_count(1_000_000), 78498);
        assert_eq!(nth_prime_u64(0), None);
        let first: Vec<u64> = (1..=6).map(|n| nth_prime_u64(n).unwrap()).collect();
        assert_eq!(first, vec![2, 3, 5, 7, 11, 13]);
        assert_eq!(nth_prime_u64(100_000), Some(1_299_709));
        assert_eq!(nth_prime_u64(PRIME_PI_2_64 + 1), None);
    }
}
