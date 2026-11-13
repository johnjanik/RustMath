//! Binomial coefficients C(n, k) by prime splitting of the window (crate-private).
//!
//! MAGMA Handbook ch. 18 §18.8 `Binomial(n, r)` (digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`), used by
//! [`crate::sage_wrapper::SageInteger::binomial`] after its sign and zero cases.
//!
//! Algorithm. `C(N, K) = Π_{i<K} W_i / K!` with the window `W_i = N - i`. Split each
//! `W_i = s_i·t_i`, where `s_i` is its K-smooth part (primes `<= K`) and `t_i` has no prime
//! factor `<= K`. `K!` is K-smooth and divides `Π W_i` (the quotient is an integer), and
//! `gcd(Π t_i, K!) = 1`, so `K! | Π s_i` and
//! `C(N, K) = Π t_i · Π_{p <= K} p^(c_p - e_p)` with `c_p = v_p(Π W_i)` (counted while the
//! primes are divided out) and `e_p = v_p(K!)` (Legendre: `Σ_j floor(K / p^j)`). Every
//! exponent is `>= 0`, so the product is exact and no large division happens. The multiples
//! of `p` in the window are the indices `i ≡ N (mod p)`.
//!
//! Cost: about `K·ln ln K` word divisions to remove the primes, plus `O(M(B) log K)` for the
//! balanced product trees, `B` the bit size of the result. The window is processed in blocks.
//! Memory: the sieve of `prime_range` (K bytes), three words per prime `<= K` (the prime,
//! `N mod p`, the signed count), one block of the window, and the block products (about the
//! size of the result).
//!
//! References:
//! - P. Goetgheluck, *Computing binomial coefficients*, Amer. Math. Monthly 94 (1987)
//!   360-365 (the prime-power factorisation of C(n, k)).
//! - A.-M. Legendre, *Essai sur la théorie des nombres* (1808): `v_p(K!)`.
//! - [vzGG99] J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, Cambridge 1999,
//!   §10.1 (balanced product trees).

use crate::fast_arith::prime_range;
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

/// Product of the factors by balanced binary splitting; `[]` gives 1.
fn product_tree(mut v: Vec<BigUint>) -> BigUint {
    if v.is_empty() {
        return BigUint::one();
    }
    while v.len() > 1 {
        let mut next = Vec::with_capacity(v.len().div_ceil(2));
        let mut it = v.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => next.push(a * b),
                None => next.push(a),
            }
        }
        v = next;
    }
    v.pop().expect("one factor left")
}

/// Packs word factors into products below `2^64`, then multiplies them with a product tree.
fn product_u64(factors: impl Iterator<Item = u64>) -> BigUint {
    let mut leaves: Vec<BigUint> = Vec::new();
    let mut acc: u64 = 1;
    for f in factors {
        if f == 1 {
            continue;
        }
        match acc.checked_mul(f) {
            Some(x) => acc = x,
            None => {
                leaves.push(BigUint::from(acc));
                acc = f;
            }
        }
    }
    if acc != 1 {
        leaves.push(BigUint::from(acc));
    }
    product_tree(leaves)
}

/// `C(n, kk)` for `0 <= kk <= n`. The caller passes `kk = min(k, n - k)`, so the window is
/// the shorter one; the result is correct for any `kk <= n`.
pub(crate) fn binomial_nonneg(n: &BigUint, kk: u64) -> BigUint {
    debug_assert!(BigUint::from(kk) <= *n);
    if kk == 0 {
        return BigUint::one();
    }
    if kk == 1 {
        return n.clone();
    }
    let primes = prime_range(2, kk + 1);
    // count[j] = c_p - e_p, starting at -e_p = -v_p(kk!) (Legendre).
    let mut count: Vec<i64> = primes
        .iter()
        .map(|&p| {
            let (mut e, mut t) = (0u64, kk);
            while t > 0 {
                t /= p;
                e += t;
            }
            -i64::try_from(e).expect("v_p(K!) < K <= 2^32 fits in i64")
        })
        .collect();
    let n_mod: Vec<u64> = primes
        .iter()
        .map(|&p| (n % p).to_u64().expect("a residue modulo a u64 fits a u64"))
        .collect();
    let small = if n.bits() <= 64 { n.to_u64() } else { None };
    let block: u64 = std::cmp::max(1 << 16, kk.div_ceil(64));
    let mut parts: Vec<BigUint> = Vec::new();
    let mut i0 = 0u64;
    while i0 < kk {
        let i1 = (i0 + block).min(kk);
        // The first index i >= i0 with p | n - i is i ≡ n (mod p).
        let first = |j: usize, p: u64| i0 + (n_mod[j] + p - i0 % p) % p;
        if let Some(nu) = small {
            let mut w: Vec<u64> = (i0..i1).map(|i| nu - i).collect();
            for (j, &p) in primes.iter().enumerate() {
                let mut i = first(j, p);
                while i < i1 {
                    let t = &mut w[usize::try_from(i - i0).expect("block index fits usize")];
                    while t.is_multiple_of(p) {
                        *t /= p;
                        count[j] += 1;
                    }
                    i += p;
                }
            }
            parts.push(product_u64(w.into_iter()));
        } else {
            let mut w: Vec<BigUint> = (i0..i1).map(|i| n - i).collect();
            for (j, &p) in primes.iter().enumerate() {
                let mut i = first(j, p);
                while i < i1 {
                    let t = &mut w[usize::try_from(i - i0).expect("block index fits usize")];
                    loop {
                        let (q, r) = num_integer::Integer::div_rem(&*t, &BigUint::from(p));
                        if !r.is_zero() {
                            break;
                        }
                        *t = q;
                        count[j] += 1;
                    }
                    i += p;
                }
            }
            parts.push(product_tree(w));
        }
        i0 = i1;
    }
    let mut prime_powers: Vec<u64> = Vec::new();
    for (&p, &c) in primes.iter().zip(&count) {
        let e = u64::try_from(c).expect("v_p(C(N, K)) >= 0: K! divides the window product");
        for _ in 0..e {
            prime_powers.push(p);
        }
    }
    parts.push(product_u64(prime_powers.into_iter()));
    product_tree(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The old K-step loop: result = C(n, i) times (n - i), divided exactly by i + 1.
    fn reference(n: &BigUint, kk: u64) -> BigUint {
        let mut r = BigUint::one();
        for i in 0..kk {
            r = r * (n - i) / (i + 1);
        }
        r
    }

    #[test]
    fn product_tree_edges() {
        assert_eq!(product_tree(Vec::new()), BigUint::one());
        assert_eq!(product_tree(vec![BigUint::from(7u32)]), BigUint::from(7u32));
        assert_eq!(product_u64([1u64, 1].into_iter()), BigUint::one());
        let big = product_u64([u64::MAX, u64::MAX, 3].into_iter());
        assert_eq!(big, BigUint::from(u64::MAX) * u64::MAX * 3u32);
    }

    #[test]
    fn binomial_nonneg_across_the_u64_boundary() {
        // The u64 window (n < 2^64) and the BigUint window (n >= 2^64) at the boundary.
        let two64 = BigUint::one() << 64u32;
        for n in [two64.clone() - 1u32, two64.clone(), two64 + 1u32] {
            for kk in [2u64, 3, 17, 1000] {
                assert_eq!(binomial_nonneg(&n, kk), reference(&n, kk), "C({n}, {kk})");
            }
        }
        for n in 0u64..=60 {
            for kk in 0..=n {
                let nb = BigUint::from(n);
                assert_eq!(binomial_nonneg(&nb, kk), reference(&nb, kk), "C({n}, {kk})");
            }
        }
    }
}
