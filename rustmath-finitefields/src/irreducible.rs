//! Irreducibility testing and irreducible-polynomial generation over GF(p).
//!
//! Provides:
//! * [`is_irreducible`] — Rabin's irreducibility test for a polynomial over a
//!   finite field.
//! * [`find_irreducible`] — deterministic search for a monic irreducible
//!   polynomial of a given degree over GF(p) (lexicographic scan).
//! * [`random_irreducible`] — pseudo-random search, deterministic for a given
//!   seed (used so tests are reproducible).
//! * [`generate_gfpn_modulus`] — returns the integer coefficient vector of a
//!   monic irreducible of degree `n` over GF(p), suitable for constructing
//!   [`crate::ff_poly::Gfpn`] / [`crate::extension_field::ExtensionField`].
//!
//! Rabin's test: a monic polynomial `f` of degree `n` over GF(q) is irreducible
//! iff
//!   1. `x^(q^n) ≡ x (mod f)`, and
//!   2. for every prime `l` dividing `n`, `gcd(x^(q^(n/l)) - x, f) = 1`.

use crate::ff_poly::{FFPoly, FiniteFieldElement};
use crate::prime_field::PrimeField;
use rustmath_core::{EuclideanDomain, NumericConversion};
use rustmath_integers::Integer;

/// Distinct prime factors of `n` (small `n`; trial division).
fn prime_divisors(mut n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            out.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Rabin's irreducibility test for `f` over the finite field of its
/// coefficients. Returns `true` iff `f` is irreducible.
///
/// Constant polynomials and the zero polynomial are reported as not
/// irreducible. Degree-1 polynomials are always irreducible.
pub fn is_irreducible<F: FiniteFieldElement>(f: &FFPoly<F>) -> bool {
    let n = match f.degree() {
        None => return false,
        Some(0) => return false,
        Some(d) => d,
    };
    if n == 1 {
        return true;
    }

    let sample = f.sample().clone();
    let q = sample.order();
    let fm = f.make_monic();
    let x = FFPoly::x(sample.clone());

    // Condition 1: x^(q^n) ≡ x (mod f).
    // We compute h = x^(q^(n/l)) iteratively, reusing intermediate powers.
    // For each prime divisor l of n, check gcd(x^(q^(n/l)) - x, f) == 1.
    for l in prime_divisors(n) {
        let m = n / l;
        // exponent q^m
        let exp = q.pow(m as u32);
        let h = x.pow_mod(&exp, &fm).expect("monic divisor");
        let diff = h.sub(&x);
        if diff.is_zero() {
            // gcd would be fm itself -> reducible
            return false;
        }
        let g = fm.gcd(&diff);
        if g.degree() != Some(0) {
            return false;
        }
    }

    // Final condition: x^(q^n) ≡ x (mod f).
    let exp_n = q.pow(n as u32);
    let h = x.pow_mod(&exp_n, &fm).expect("monic divisor");
    h == x
}

/// Build a monic `FFPoly<PrimeField>` of degree `n` over GF(p) from the
/// integer encoding of its `n` low-degree coefficients (the leading 1 is
/// implicit). `idx` ranges over `0 .. p^n`; coefficient `i` is the `i`-th
/// base-`p` digit of `idx`.
fn poly_from_index(idx: &Integer, n: usize, p: &Integer) -> FFPoly<PrimeField> {
    let sample = PrimeField::new(Integer::zero(), p.clone()).unwrap();
    let mut coeffs: Vec<PrimeField> = Vec::with_capacity(n + 1);
    let mut rem = idx.clone();
    for _ in 0..n {
        let (q, r) = rem.div_rem(p).unwrap();
        coeffs.push(PrimeField::new(r, p.clone()).unwrap());
        rem = q;
    }
    coeffs.push(PrimeField::new(Integer::one(), p.clone()).unwrap()); // monic leading term
    FFPoly::new(coeffs, sample)
}

/// Deterministically find the lexicographically-first monic irreducible
/// polynomial of degree `n` over GF(p). Returns `None` only for invalid input
/// (`n == 0`).
///
/// The search is over all `p^n` monic polynomials of degree `n`, ordered by the
/// base-`p` value of their coefficient vector. An irreducible of every degree
/// exists, so for valid input this always succeeds.
pub fn find_irreducible(p: &Integer, n: usize) -> Option<FFPoly<PrimeField>> {
    if n == 0 {
        return None;
    }
    let count = p.pow(n as u32);
    let mut idx = Integer::zero();
    while idx < count {
        let f = poly_from_index(&idx, n, p);
        if is_irreducible(&f) {
            return Some(f);
        }
        idx = idx + Integer::one();
    }
    None
}

/// Simple deterministic PRNG (SplitMix64) so randomized search is reproducible.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// Pseudo-randomly search for a monic irreducible of degree `n` over GF(p),
/// seeded by `seed` for reproducibility. Falls back to a deterministic scan if
/// the random attempts are exhausted, so it never spuriously fails.
pub fn random_irreducible(p: &Integer, n: usize, seed: u64) -> Option<FFPoly<PrimeField>> {
    if n == 0 {
        return None;
    }
    let count = p.pow(n as u32);
    let count_u64 = count.to_usize().map(|u| u as u64);
    let mut rng = SplitMix64::new(seed);

    // Try a bounded number of random candidates first.
    let attempts = 64 * n.max(1);
    for _ in 0..attempts {
        let idx = match count_u64 {
            Some(c) if c > 0 => Integer::from((rng.next_u64() % c) as i64),
            _ => {
                // count too large for u64: build a random index digit by digit
                let mut v = Integer::zero();
                let mut place = Integer::one();
                for _ in 0..n {
                    let digit =
                        Integer::from((rng.next_u64() % p.to_usize().unwrap_or(2) as u64) as i64);
                    v = v + digit * place.clone();
                    place = place * p.clone();
                }
                v
            }
        };
        let f = poly_from_index(&idx, n, p);
        if is_irreducible(&f) {
            return Some(f);
        }
    }

    // Guaranteed fallback.
    find_irreducible(p, n)
}

/// Return the integer coefficient vector (low degree first, length `n + 1`,
/// monic) of an irreducible polynomial of degree `n` over GF(p). This is the
/// form expected by [`crate::ff_poly::Gfpn::new`] and
/// [`crate::extension_field::ExtensionField::new`] for constructing GF(p^n).
pub fn generate_gfpn_modulus(p: &Integer, n: usize) -> Option<Vec<Integer>> {
    let f = find_irreducible(p, n)?;
    Some(f.coeffs().iter().map(|c| c.value().clone()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff_poly::Gfpn;

    fn poly(vals: &[i64], p: i64) -> FFPoly<PrimeField> {
        let sample = PrimeField::new(Integer::zero(), Integer::from(p)).unwrap();
        FFPoly::new(
            vals.iter()
                .map(|&v| PrimeField::new(Integer::from(v), Integer::from(p)).unwrap())
                .collect(),
            sample,
        )
    }

    #[test]
    fn test_irreducible_known() {
        // x^2 + x + 1 is irreducible over GF(2)
        assert!(is_irreducible(&poly(&[1, 1, 1], 2)));
        // x^2 + 1 = (x+1)^2 over GF(2) -> reducible
        assert!(!is_irreducible(&poly(&[1, 0, 1], 2)));
        // x^2 - x = x(x-1) over GF(2) -> reducible
        assert!(!is_irreducible(&poly(&[0, 1, 1], 2)));
        // x^2 + 1 over GF(3) is irreducible (no root: 0,1,2 give 1,2,2)
        assert!(is_irreducible(&poly(&[1, 0, 1], 3)));
        // x^2 + x + 1 over GF(3): root at x=1 -> reducible
        assert!(!is_irreducible(&poly(&[1, 1, 1], 3)));
    }

    #[test]
    fn test_irreducible_degree3() {
        // x^3 + x + 1 is irreducible over GF(2)
        assert!(is_irreducible(&poly(&[1, 1, 0, 1], 2)));
        // x^3 + x^2 + 1 is irreducible over GF(2)
        assert!(is_irreducible(&poly(&[1, 0, 1, 1], 2)));
        // x^3 + 1 = (x+1)(x^2+x+1) -> reducible
        assert!(!is_irreducible(&poly(&[1, 0, 0, 1], 2)));
    }

    #[test]
    fn test_find_irreducible_deg4_gf2() {
        let f = find_irreducible(&Integer::from(2), 4).unwrap();
        assert_eq!(f.degree(), Some(4));
        assert!(is_irreducible(&f));
        // The lex-first degree-4 irreducible over GF(2) is x^4 + x + 1.
        assert_eq!(f, poly(&[1, 1, 0, 0, 1], 2));
    }

    #[test]
    fn test_random_irreducible_deterministic() {
        let a = random_irreducible(&Integer::from(2), 5, 12345).unwrap();
        let b = random_irreducible(&Integer::from(2), 5, 12345).unwrap();
        assert_eq!(a, b); // reproducible
        assert!(is_irreducible(&a));
        assert_eq!(a.degree(), Some(5));
    }

    #[test]
    fn test_generate_gfpn_modulus_builds_field() {
        let p = Integer::from(7);
        let modulus = generate_gfpn_modulus(&p, 3).unwrap();
        assert_eq!(modulus.len(), 4);
        // Build GF(7^3) and confirm a nonzero element is invertible.
        let elem = Gfpn::new(
            vec![Integer::from(2), Integer::from(3), Integer::from(1)],
            p.clone(),
            modulus.clone(),
        );
        let inv = elem.invert().unwrap();
        assert!(elem.mul(&inv).is_one());
        assert_eq!(elem.order(), Integer::from(343));
    }

    #[test]
    fn test_irreducible_over_gfpn() {
        // Build GF(2^2) and test irreducibility of a degree-2 poly over it.
        let p = Integer::from(2);
        let modu = vec![Integer::from(1), Integer::from(1), Integer::from(1)];
        let zero = Gfpn::new(vec![Integer::from(0)], p.clone(), modu.clone());
        let one = zero.one();
        let omega = Gfpn::new(vec![Integer::from(0), Integer::from(1)], p.clone(), modu.clone());
        // f(y) = y^2 + y + omega over GF(4). Check it has no root in GF(4):
        // if it had a root it'd be reducible. We just confirm the test runs and
        // returns a boolean consistent with a brute-force root check.
        let f = FFPoly::new(vec![omega.clone(), one.clone(), one.clone()], zero.clone());
        let irr = is_irreducible(&f);
        // brute force: any root among the 4 elements?
        let elems = [
            zero.clone(),
            one.clone(),
            omega.clone(),
            one.add(&omega),
        ];
        let has_root = elems.iter().any(|e| f.evaluate(e).is_zero());
        // degree 2: irreducible iff no root
        assert_eq!(irr, !has_root);
    }
}
