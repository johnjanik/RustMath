//! Real-root counting over `ℚ` via Sturm sequences — the native replacement for
//! the Sage real-root count in the Mestre pipeline (verify a specialization has
//! exactly `r = 20` real roots).
//!
//! For `f ∈ ℚ[x]`, the Sturm chain is `f₀ = f`, `f₁ = f'`,
//! `f_{i+1} = −rem(f_{i-1}, f_i)`. The number of **distinct** real roots in `(a, b]`
//! is `V(a) − V(b)`, where `V(x)` is the number of sign variations in the chain
//! evaluated at `x` (Sturm's theorem; correct for repeated roots too). The total
//! count over `ℝ` uses the signs at `±∞`, read off the leading coefficients.

use rustmath_integers::Integer;
use rustmath_rationals::Rational;

fn rzero() -> Rational {
    Rational::from(0i64)
}

fn deg(p: &[Rational]) -> i64 {
    let mut n = p.len();
    while n > 0 && p[n - 1] == rzero() {
        n -= 1;
    }
    n as i64 - 1
}

fn derivative(p: &[Rational]) -> Vec<Rational> {
    if p.len() <= 1 {
        return vec![rzero()];
    }
    (1..p.len()).map(|i| p[i].clone() * Rational::from(i as i64)).collect()
}

/// Remainder of `a` by `b` over `ℚ` (`b ≠ 0`).
fn rem(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let db = deg(b);
    if db < 0 {
        return vec![rzero()];
    }
    let lcb_inv = b[db as usize].reciprocal().expect("nonzero leading coeff");
    let mut r: Vec<Rational> = a.to_vec();
    while deg(&r) >= db && deg(&r) >= 0 {
        let dr = deg(&r) as usize;
        let coeff = r[dr].clone() * lcb_inv.clone();
        let shift = dr - db as usize;
        for j in 0..b.len() {
            r[j + shift] = r[j + shift].clone() - coeff.clone() * b[j].clone();
        }
        while r.len() > 1 && *r.last().unwrap() == rzero() {
            r.pop();
        }
        if deg(&r) < db {
            break;
        }
    }
    r
}

fn neg(p: &[Rational]) -> Vec<Rational> {
    p.iter().map(|c| -c.clone()).collect()
}

/// The Sturm chain of `f`: `f₀ = f`, `f₁ = f'`, `f_{i+1} = −rem(f_{i-1}, f_i)`.
fn sturm_chain(f: &[Rational]) -> Vec<Vec<Rational>> {
    let mut chain: Vec<Vec<Rational>> = Vec::new();
    if deg(f) < 0 {
        return chain;
    }
    chain.push(f.to_vec());
    if deg(f) == 0 {
        return chain;
    }
    chain.push(derivative(f));
    loop {
        let n = chain.len();
        if deg(&chain[n - 1]) < 0 {
            chain.pop();
            break;
        }
        let r = neg(&rem(&chain[n - 2], &chain[n - 1]));
        if deg(&r) < 0 {
            break;
        }
        chain.push(r);
    }
    chain
}

fn sign_of(x: &Rational) -> i32 {
    if *x > rzero() {
        1
    } else if *x < rzero() {
        -1
    } else {
        0
    }
}

fn variations(signs: &[i32]) -> usize {
    let mut v = 0;
    let mut prev = 0i32;
    for &s in signs {
        if s == 0 {
            continue;
        }
        if prev != 0 && s != prev {
            v += 1;
        }
        prev = s;
    }
    v
}

/// Leading coefficient sign of each chain member at `+∞` (= sign of leading coeff)
/// or `−∞` (= sign(lc)·(−1)^deg).
fn signs_at_infinity(chain: &[Vec<Rational>], positive: bool) -> Vec<i32> {
    chain
        .iter()
        .map(|p| {
            let d = deg(p);
            if d < 0 {
                return 0;
            }
            let s = sign_of(&p[d as usize]);
            if positive || d % 2 == 0 {
                s
            } else {
                -s
            }
        })
        .collect()
}

fn signs_at(chain: &[Vec<Rational>], x: &Rational) -> Vec<i32> {
    chain
        .iter()
        .map(|p| {
            let mut acc = rzero();
            for c in p.iter().rev() {
                acc = acc * x.clone() + c.clone();
            }
            sign_of(&acc)
        })
        .collect()
}

/// Number of distinct real roots of `f ∈ ℚ[x]` over all of `ℝ`.
pub fn count_real_roots(f: &[Rational]) -> usize {
    let chain = sturm_chain(f);
    if chain.is_empty() || deg(f) <= 0 {
        return 0;
    }
    let v_neg = variations(&signs_at_infinity(&chain, false));
    let v_pos = variations(&signs_at_infinity(&chain, true));
    v_neg - v_pos
}

/// Number of distinct real roots in the half-open interval `(a, b]`.
pub fn count_real_roots_in(f: &[Rational], a: &Rational, b: &Rational) -> usize {
    let chain = sturm_chain(f);
    if chain.is_empty() || deg(f) <= 0 {
        return 0;
    }
    let va = variations(&signs_at(&chain, a));
    let vb = variations(&signs_at(&chain, b));
    va - vb
}

/// Convenience: count distinct real roots of an integer polynomial.
pub fn count_real_roots_int(f: &[Integer]) -> usize {
    let fq: Vec<Rational> = f.iter().map(|c| Rational::from_integer(c.clone())).collect();
    count_real_roots(&fq)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> Rational {
        Rational::from(n)
    }
    fn qs(v: &[i64]) -> Vec<Rational> {
        v.iter().map(|&x| q(x)).collect()
    }

    #[test]
    fn quadratics() {
        assert_eq!(count_real_roots(&qs(&[-1, 0, 1])), 2); // x^2-1
        assert_eq!(count_real_roots(&qs(&[1, 0, 1])), 0); // x^2+1
        assert_eq!(count_real_roots(&qs(&[0, 0, 1])), 1); // x^2 (one distinct)
    }

    #[test]
    fn cubic_and_quartic() {
        // (x-1)(x-2)(x-3) = x^3 -6x^2 +11x -6 → 3 real
        assert_eq!(count_real_roots(&qs(&[-6, 11, -6, 1])), 3);
        // x^4+1 → 0 real
        assert_eq!(count_real_roots(&qs(&[1, 0, 0, 0, 1])), 0);
        // x^4 - 5x^2 + 4 = (x^2-1)(x^2-4) → 4 real (±1,±2)
        assert_eq!(count_real_roots(&qs(&[4, 0, -5, 0, 1])), 4);
    }

    #[test]
    fn interval_count() {
        // (x-1)(x-2)(x-3): roots in (0,2] are 1,2 → 2
        let f = qs(&[-6, 11, -6, 1]);
        assert_eq!(count_real_roots_in(&f, &q(0), &q(2)), 2);
        assert_eq!(count_real_roots_in(&f, &q(0), &q(10)), 3);
        assert_eq!(count_real_roots_in(&f, &q(4), &q(10)), 0);
    }

    #[test]
    fn mestre_seed_has_20_real_roots() {
        // L(X) = Π_{i=1}^{10}(X^2 - i^2): exactly 20 real roots ±1..±10
        let mut l = qs(&[1]); // start with 1
        for i in 1..=10i64 {
            // multiply by (X^2 - i^2)
            let factor = vec![q(-i * i), q(0), q(1)];
            let mut out = vec![rzero(); l.len() + 2];
            for (a, ca) in l.iter().enumerate() {
                for (b, cb) in factor.iter().enumerate() {
                    out[a + b] = out[a + b].clone() + ca.clone() * cb.clone();
                }
            }
            l = out;
        }
        assert_eq!(count_real_roots(&l), 20);
    }
}
