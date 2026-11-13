//! Dense polynomials modulo N and one root of a split polynomial, for the ECPP prover (MAGMA
//! Handbook ch. 18 §18.9.1; digest `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`).
//!
//! Root finding is the Cantor-Zassenhaus splitting of H. Cohen, *A Course in Computational
//! Algebraic Number Theory*, GTM 138, §1.6.1 (roots mod p): for f of degree > 1 that splits into
//! distinct linear factors mod a prime N, g = gcd((x + delta)^((N-1)/2) - 1, f) is a proper
//! factor for about half of the random delta; recurse on the smaller factor. The power is
//! computed left to right with squarings mod f; multiplying by x + delta is linear.
//!
//! N is only a probable prime. A leading coefficient that is not invertible mod N gives a
//! proper divisor of N, which is returned as `Err(divisor)`.
//!
//! Lazy reduction: a product accumulates the unreduced sums and reduces each coefficient once;
//! the remainder by a monic polynomial subtracts multiples without reducing, and reduces only the
//! coefficient it divides by. `rustmath-polynomials` depends on this crate, so it cannot be used
//! here.

use num_bigint::BigInt;
use num_integer::Integer as _;
use num_traits::{One, Zero};

/// Coefficients low to high; no trailing zeros once trimmed.
type Poly = Vec<BigInt>;

fn trim(mut f: Poly) -> Poly {
    while f.last().is_some_and(|c| c.is_zero()) {
        f.pop();
    }
    f
}

fn reduce(f: Poly, n: &BigInt) -> Poly {
    trim(f.into_iter().map(|c| c.mod_floor(n)).collect())
}

/// The inverse of `a` modulo `n`, or `Err(gcd)` with 1 < gcd < n when it does not exist and
/// a is not 0 mod n.
fn inverse(a: &BigInt, n: &BigInt) -> Result<BigInt, BigInt> {
    let a = a.mod_floor(n);
    match a.modinv(n) {
        Some(v) => Ok(v),
        None => Err(a.gcd(n)),
    }
}

/// f g mod n, one reduction per coefficient.
fn mul(f: &[BigInt], g: &[BigInt], n: &BigInt) -> Poly {
    if f.is_empty() || g.is_empty() {
        return Vec::new();
    }
    let mut r = vec![BigInt::zero(); f.len() + g.len() - 1];
    for (i, a) in f.iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        for (j, b) in g.iter().enumerate() {
            r[i + j] += a * b;
        }
    }
    reduce(r, n)
}

/// f mod g for a monic g (coefficients of g reduced); the output is reduced.
fn rem_monic(f: Poly, g: &[BigInt], n: &BigInt) -> Poly {
    let dg = g.len() - 1;
    let mut f = f;
    while f.len() > dg {
        let top = f.len() - 1;
        let c = f[top].mod_floor(n);
        if !c.is_zero() {
            let sh = top - dg;
            for i in 0..dg {
                f[sh + i] -= &c * &g[i];
            }
        }
        f.pop();
    }
    reduce(f, n)
}

/// f mod g for any g of degree >= 0 with a unit leading coefficient.
fn rem(f: &[BigInt], g: &[BigInt], n: &BigInt) -> Result<Poly, BigInt> {
    let mut f = reduce(f.to_vec(), n);
    let li = inverse(g.last().expect("g is not zero"), n)?;
    while f.len() >= g.len() {
        let c = (f.last().expect("f is not empty") * &li).mod_floor(n);
        let sh = f.len() - g.len();
        for (i, b) in g.iter().enumerate() {
            f[sh + i] = (&f[sh + i] - &c * b).mod_floor(n);
        }
        f = trim(f);
    }
    Ok(f)
}

fn monic(f: &[BigInt], n: &BigInt) -> Result<Poly, BigInt> {
    let li = inverse(f.last().expect("f is not zero"), n)?;
    Ok(f.iter().map(|c| (c * &li).mod_floor(n)).collect())
}

/// The monic gcd of f and g (f not zero).
fn gcd(f: &[BigInt], g: &[BigInt], n: &BigInt) -> Result<Poly, BigInt> {
    let mut a = reduce(f.to_vec(), n);
    let mut b = reduce(g.to_vec(), n);
    while !b.is_empty() {
        let r = rem(&a, &b, n)?;
        a = b;
        b = r;
    }
    monic(&a, n)
}

/// f / g for g dividing f.
fn div_exact(f: &[BigInt], g: &[BigInt], n: &BigInt) -> Result<Poly, BigInt> {
    let mut f = f.to_vec();
    let li = inverse(g.last().expect("g is not zero"), n)?;
    let mut q = vec![BigInt::zero(); f.len() + 1 - g.len()];
    while f.len() >= g.len() {
        let c = (f.last().expect("f is not empty") * &li).mod_floor(n);
        let sh = f.len() - g.len();
        q[sh] = c.clone();
        for (i, b) in g.iter().enumerate() {
            f[sh + i] = (&f[sh + i] - &c * b).mod_floor(n);
        }
        f = trim(f);
    }
    Ok(trim(q))
}

/// (x + delta)^e mod f for a monic f of degree >= 1, left to right.
fn pow_linear(delta: &BigInt, e: &BigInt, f: &[BigInt], n: &BigInt) -> Poly {
    let mut r: Poly = vec![BigInt::one()];
    for i in (0..e.bits()).rev() {
        r = rem_monic(mul(&r, &r, n), f, n);
        if e.bit(i) {
            let mut s = vec![BigInt::zero(); r.len() + 1];
            for (k, c) in r.iter().enumerate() {
                s[k + 1] += c;
                s[k] += c * delta;
            }
            r = rem_monic(s, f, n);
        }
    }
    r
}

/// Number of random splitting attempts before [`find_root`] gives up.
const SPLIT_ATTEMPTS: u32 = 64;

/// One root of h (integer coefficients, low to high) modulo n, for h that splits into distinct
/// linear factors mod n. `Ok(None)` when h mod n has degree < 1 or no split was found within
/// 64 attempts (this says nothing about n), `Err(d)` with a proper divisor d of n when a
/// leading coefficient was not invertible.
pub(crate) fn find_root(
    h: &[BigInt],
    n: &BigInt,
    rng: &mut super::SplitMix64,
) -> Result<Option<BigInt>, BigInt> {
    let f = reduce(h.to_vec(), n);
    if f.len() < 2 {
        return Ok(None);
    }
    let mut f = monic(&f, n)?;
    let e: BigInt = (n - 1u32) >> 1u32;
    let mut attempts = 0;
    while f.len() > 2 {
        if attempts == SPLIT_ATTEMPTS {
            return Ok(None);
        }
        attempts += 1;
        let delta = rng.below(n);
        let mut g = pow_linear(&delta, &e, &f, n);
        if g.is_empty() {
            g.push(BigInt::zero());
        }
        g[0] = (&g[0] - 1u32).mod_floor(n);
        let g = trim(g);
        if g.is_empty() {
            continue; // (x + delta)^e = 1 mod f: no information
        }
        let d = gcd(&f, &g, n)?;
        if d.len() > 1 && d.len() < f.len() {
            let other = div_exact(&f, &d, n)?;
            f = if d.len() <= other.len() {
                d
            } else {
                monic(&other, n)?
            };
        }
    }
    Ok(Some((-&f[0]).mod_floor(n)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(v: &[i64]) -> Poly {
        v.iter().map(|&c| BigInt::from(c)).collect()
    }

    #[test]
    fn arithmetic_mod_n() {
        let n = BigInt::from(101u32);
        // (x + 1)(x + 2) = x^2 + 3x + 2
        assert_eq!(mul(&p(&[1, 1]), &p(&[2, 1]), &n), p(&[2, 3, 1]));
        // x^3 mod (x^2 + 3x + 2) = 7x + 6 (x^3 = (x - 3)(x^2 + 3x + 2) + 7x + 6)
        assert_eq!(rem_monic(p(&[0, 0, 0, 1]), &p(&[2, 3, 1]), &n), p(&[6, 7]));
        assert_eq!(gcd(&p(&[2, 3, 1]), &p(&[3, 4, 1]), &n).unwrap(), p(&[1, 1]));
        assert_eq!(
            div_exact(&p(&[2, 3, 1]), &p(&[1, 1]), &n).unwrap(),
            p(&[2, 1])
        );
        // A non-invertible leading coefficient mod 15 gives the divisor.
        assert_eq!(
            monic(&p(&[1, 3]), &BigInt::from(15u32)),
            Err(BigInt::from(3u32))
        );
    }

    #[test]
    fn root_of_split_polynomial() {
        // (x - 5)(x - 17)(x - 40) mod 101; gp: polrootsmod((x-5)*(x-17)*(x-40), 101)
        let n = BigInt::from(101u32);
        let f = mul(&mul(&p(&[-5, 1]), &p(&[-17, 1]), &n), &p(&[-40, 1]), &n);
        let mut rng = super::super::SplitMix64::new(&n);
        for _ in 0..10 {
            let r = find_root(&f, &n, &mut rng).unwrap().unwrap();
            assert!(
                [5u32, 17, 40].contains(&u32::try_from(&r).unwrap()),
                "r = {r}"
            );
        }
    }
}
