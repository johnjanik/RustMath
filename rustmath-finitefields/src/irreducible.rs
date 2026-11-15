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
//!
//! (\[Rab80\] M. O. Rabin, *Probabilistic algorithms in finite fields*, SIAM J.
//! Comput. 9 (1980) 273-280.) The generators need a prime `p`: they return
//! `None` when `p` is not proven prime (`rustmath_integers::prime::try_is_prime`
//! false, or an unresolved proof), and [`is_irreducible`] panics for a
//! coefficient ring of composite characteristic. [`random_irreducible`] is MAGMA
//! `RandomIrreduciblePolynomial` over GF(p) (ch. 21 §21.5: random polynomial with
//! irreducibility testing): every coefficient is uniform over GF(p), for every
//! size of `p`.

use crate::ff_poly::{check_prime_characteristic, uniform_below, FFPoly, FiniteFieldElement};
use crate::prime_field::PrimeField;
use rustmath_core::EuclideanDomain;
use rustmath_integers::Integer;

/// Distinct prime factors of `n` (small `n`; trial division).
pub(crate) fn prime_divisors(mut n: usize) -> Vec<usize> {
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
///
/// # Panics
/// When the characteristic of the coefficients is not a prime (a `bool` over a
/// ring that is not a field has no honest value), with
/// "irreducible::is_irreducible: characteristic {c} is not a prime"; also when
/// the primality proof of the characteristic is unresolved.
pub fn is_irreducible<F: FiniteFieldElement>(f: &FFPoly<F>) -> bool {
    if let Err(why) = check_prime_characteristic(f.sample()) {
        panic!("irreducible::is_irreducible: {why}");
    }
    rabin_irreducible(f)
}

/// [`is_irreducible`] without the characteristic test. Precondition: the
/// coefficients lie in a field (the caller has proven the characteristic prime).
fn rabin_irreducible<F: FiniteFieldElement>(f: &FFPoly<F>) -> bool {
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

/// Whether `p` is a proven prime (`try_is_prime`; an unresolved proof is not).
fn proven_prime(p: &Integer) -> bool {
    matches!(rustmath_integers::prime::try_is_prime(p), Ok(true))
}

/// Deterministically find the lexicographically-first monic irreducible
/// polynomial of degree `n` over GF(p). Returns `None` only for invalid input:
/// `n == 0`, or `p` not proven prime (composite, below 2, or an unresolved
/// primality proof).
///
/// The search is over all `p^n` monic polynomials of degree `n`, ordered by the
/// base-`p` value of their coefficient vector. An irreducible of every degree
/// exists, so for valid input this always succeeds.
pub fn find_irreducible(p: &Integer, n: usize) -> Option<FFPoly<PrimeField>> {
    if n == 0 || !proven_prime(p) {
        return None;
    }
    find_irreducible_unchecked(p, n)
}

/// [`find_irreducible`] for `n >= 1` and a prime `p` (the caller's duty).
fn find_irreducible_unchecked(p: &Integer, n: usize) -> Option<FFPoly<PrimeField>> {
    let count = p.pow(n as u32);
    let mut idx = Integer::zero();
    while idx < count {
        let f = poly_from_index(&idx, n, p);
        if rabin_irreducible(&f) {
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

/// Pseudo-randomly search for a monic irreducible of degree `n` over GF(p)
/// (MAGMA `RandomIrreduciblePolynomial`, ch. 21 §21.5), seeded by `seed` for
/// reproducibility. `None` for `n == 0` or when `p` is not proven prime
/// (composite, below 2, or an unresolved primality proof).
///
/// A candidate is `x^n + c_{n-1} x^(n-1) + ... + c_0` with `c_0, ..., c_{n-1}`
/// drawn in that order with the exact sampler `uniform_below(p, ..)` from a
/// SplitMix64 stream, so every candidate is uniform among the monic polynomials
/// of degree `n` and the first irreducible one is uniform among the monic
/// irreducibles (for every size of `p`). At most `64 n` candidates are tried;
/// then a deterministic scan ([`find_irreducible`]) answers, so this never
/// fails for valid input (the fallback is reached with probability about
/// `e^-64`).
pub fn random_irreducible(p: &Integer, n: usize, seed: u64) -> Option<FFPoly<PrimeField>> {
    if n == 0 || !proven_prime(p) {
        return None;
    }
    let sample = PrimeField::new(Integer::zero(), p.clone()).ok()?;
    let mut rng = SplitMix64::new(seed);
    let attempts = 64 * n;
    for _ in 0..attempts {
        let mut coeffs: Vec<PrimeField> = (0..n)
            .map(|_| {
                let c = uniform_below(p, &mut || rng.next_u64());
                PrimeField::new(c, p.clone()).expect("p >= 2")
            })
            .collect();
        coeffs.push(PrimeField::new(Integer::one(), p.clone()).expect("p >= 2"));
        let f = FFPoly::new(coeffs, sample.clone());
        if rabin_irreducible(&f) {
            return Some(f);
        }
    }

    // Guaranteed fallback.
    find_irreducible_unchecked(p, n)
}

/// Return the integer coefficient vector (low degree first, length `n + 1`,
/// monic) of an irreducible polynomial of degree `n` over GF(p). This is the
/// form expected by [`crate::ff_poly::Gfpn::new`] and
/// [`crate::extension_field::ExtensionField::new`] for constructing GF(p^n).
/// `None` when [`find_irreducible`] gives `None` (`n == 0`, or `p` not proven
/// prime).
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

    fn big(s: &str) -> Integer {
        Integer::from_decimal_str(s).unwrap()
    }

    fn values(f: &FFPoly<PrimeField>) -> Vec<Integer> {
        f.coeffs().iter().map(|c| c.value().clone()).collect()
    }

    /// Uniform draws over the whole of GF(p), also for p > 2^64 (HEAD drew only 0 and 1
    /// there, and went through `as i64` below 2^64). The pinned outputs come from the
    /// design prototype (`p3_randirr.rs`, same sampler and stream); gp
    /// polisirreducible(Mod(1,p)*f) = 1 for each:
    ///   p = 2^80+13: x^2 + 949416489080899775911262*x + 1116388955813207555071169,
    ///     x^3 + 826732766301133076639116*x^2 + 520708005305936630825557*x + 445155756734992886557959;
    ///   p = 2^64-59: x^2 + 13757245211066428519*x + 10451216379200822465.
    #[test]
    fn random_irreducible_draws_the_whole_field() {
        let p80 = big("1208925819614629174706189"); // 2^80 + 13, gp isprime = 1
        let p64 = big("18446744073709551557"); // 2^64 - 59, gp isprime = 1
        let f = random_irreducible(&p80, 2, 1).unwrap();
        assert_eq!(
            values(&f),
            vec![
                big("1116388955813207555071169"),
                big("949416489080899775911262"),
                Integer::one()
            ]
        );
        let f = random_irreducible(&p80, 3, 42).unwrap();
        assert_eq!(
            values(&f),
            vec![
                big("445155756734992886557959"),
                big("520708005305936630825557"),
                big("826732766301133076639116"),
                Integer::one()
            ]
        );
        let f = random_irreducible(&p64, 2, 1).unwrap();
        assert_eq!(
            values(&f),
            vec![big("10451216379200822465"), big("13757245211066428519"), Integer::one()]
        );
        // Range: over 64 seeds (n = 2), about half the coefficients mod 2^64 - 59 are
        // >= 2^63 (prototype: 67 of 128), and almost all mod 2^80 + 13 are >= 2^64
        // (prototype: 128 of 128).
        let two63 = big("9223372036854775808");
        let two64 = big("18446744073709551616");
        let (mut high64, mut high80) = (0, 0);
        for seed in 0..64u64 {
            let a = values(&random_irreducible(&p64, 2, seed).unwrap());
            let b = values(&random_irreducible(&p80, 2, seed).unwrap());
            high64 += a[..2].iter().filter(|c| **c >= two63).count();
            high80 += b[..2].iter().filter(|c| **c >= two64).count();
        }
        assert!((32..=96).contains(&high64), "{high64} of 128");
        assert_eq!(high80, 128);
    }

    /// HEAD: find_irreducible(15, 2) and generate_gfpn_modulus(15, 2) gave x^2 + 1 (a
    /// fabricated "irreducible over F_15"); random_irreducible(15, ..) and p = 0 panicked.
    #[test]
    fn generators_reject_non_prime_p() {
        let p15 = Integer::from(15);
        assert!(find_irreducible(&p15, 2).is_none());
        assert!(random_irreducible(&p15, 2, 7).is_none());
        assert!(generate_gfpn_modulus(&p15, 2).is_none());
        for p in [0i64, 1, 4, -7] {
            let p = Integer::from(p);
            assert!(find_irreducible(&p, 2).is_none());
            assert!(random_irreducible(&p, 2, 7).is_none());
            assert!(generate_gfpn_modulus(&p, 2).is_none());
        }
    }

    #[test]
    #[should_panic(expected = "irreducible::is_irreducible: characteristic 15 is not a prime")]
    fn is_irreducible_rejects_composite_characteristic() {
        is_irreducible(&poly(&[1, 0, 1], 15));
    }
}
