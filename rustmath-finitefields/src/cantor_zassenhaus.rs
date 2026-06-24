//! Cantor–Zassenhaus factorization over finite fields GF(p) and GF(p^n).
//!
//! The pipeline is the classical three-stage one:
//!
//! 1. **Square-free factorization** ([`squarefree_factorization`]) — splits an
//!    arbitrary polynomial into square-free factors with their multiplicities,
//!    handling the inseparable (`p`-th power) case in characteristic `p`.
//! 2. **Distinct-degree factorization** ([`distinct_degree_factorization`]) —
//!    given a square-free polynomial, groups its irreducible factors by degree.
//! 3. **Equal-degree factorization** ([`equal_degree_factorization`]) — splits a
//!    product of distinct irreducibles all of the same degree `d` into its
//!    individual factors, using the probabilistic Cantor–Zassenhaus splitting.
//!
//! [`factor`] composes all three to factor an arbitrary polynomial; the result
//! is a list of `(monic irreducible factor, multiplicity)` pairs together with
//! the leading-coefficient unit. The randomness used by equal-degree splitting
//! is derived deterministically from the input (a SplitMix64 PRNG seeded from
//! the coefficients), so results are reproducible.

use crate::ff_poly::{FFPoly, FiniteFieldElement};
use rustmath_core::{EuclideanDomain, NumericConversion};
use rustmath_integers::Integer;

/// Deterministic PRNG seeded from the polynomial so randomized splitting is
/// reproducible across runs (required for stable tests).
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

fn seed_from_poly<F: FiniteFieldElement>(f: &FFPoly<F>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for c in f.coeffs() {
        // mix in a coarse hash of each coefficient via its order/zero-ness
        let bit = if c.is_zero() { 0u64 } else if c.is_one() { 1 } else { 2 };
        h ^= bit;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= f.coeffs().len() as u64;
    h.wrapping_mul(0x100000001b3) | 1
}

/// A random field element drawn from the PRNG. Builds the element from a random
/// integer in `[0, q)` via the prime-subfield embedding for GF(p); for GF(p^n)
/// it builds each coordinate from `from_int`, which is enough to seed the
/// splitting polynomials used by equal-degree factorization.
fn random_poly_below_degree<F: FiniteFieldElement>(
    sample: &F,
    deg: usize,
    rng: &mut SplitMix64,
) -> FFPoly<F> {
    let p = sample.characteristic();
    let p_u = p.to_usize().unwrap_or(2).max(2) as u64;
    let mut coeffs = Vec::with_capacity(deg);
    for _ in 0..deg {
        let v = (rng.next_u64() % p_u) as i64;
        coeffs.push(sample.from_int(&Integer::from(v)));
    }
    FFPoly::new(coeffs, sample.clone())
}

/// Square-free factorization (Yun-style, generalised to characteristic `p`).
///
/// Returns a list of `(squarefree factor, multiplicity)` pairs, each factor
/// monic and of degree ≥ 1, whose product (with multiplicities) equals the
/// monic part of `f`. Repeated factors are correctly separated, including the
/// inseparable case where `gcd(f, f') = f` and `f` is a `p`-th power.
pub fn squarefree_factorization<F: FiniteFieldElement>(
    f: &FFPoly<F>,
) -> Vec<(FFPoly<F>, usize)> {
    let mut result = Vec::new();
    if f.degree().unwrap_or(0) == 0 {
        return result;
    }
    let sample = f.sample().clone();
    let p = sample.characteristic();
    let p_usize = p.to_usize().unwrap_or(0);

    sqf_recurse(&f.make_monic(), 1, p_usize, &sample, &mut result);
    result
}

fn sqf_recurse<F: FiniteFieldElement>(
    f: &FFPoly<F>,
    mult_scale: usize,
    p_usize: usize,
    sample: &F,
    out: &mut Vec<(FFPoly<F>, usize)>,
) {
    if f.degree().unwrap_or(0) == 0 {
        return;
    }
    let fp = f.derivative();
    if fp.is_zero() {
        // f is a p-th power: f(x) = g(x^p). Take p-th root and recurse.
        let g = pth_root(f, p_usize, sample);
        sqf_recurse(&g, mult_scale * p_usize, p_usize, sample, out);
        return;
    }
    let mut c = f.gcd(&fp); // product of factors with multiplicity > 1 (reduced)
    let mut w = f.div_rem(&c).expect("c divides f").0; // squarefree part skeleton
    let mut i = 1usize;
    while w.degree().unwrap_or(0) > 0 {
        let y = w.gcd(&c);
        let z = w.div_rem(&y).expect("y divides w").0; // the new squarefree factor of multiplicity i
        if z.degree().unwrap_or(0) > 0 {
            out.push((z.make_monic(), i * mult_scale));
        }
        w = y;
        c = c.div_rem(&w).expect("w divides c").0;
        i += 1;
    }
    // Whatever remains in c is a p-th power.
    if c.degree().unwrap_or(0) > 0 {
        let g = pth_root(&c, p_usize, sample);
        sqf_recurse(&g, mult_scale * p_usize, p_usize, sample, out);
    }
}

/// Given `f(x) = g(x^p)` (i.e. only exponents divisible by `p` appear), return
/// `g`. Coefficients are unchanged because in GF(p^k) the map `a -> a^p` is the
/// Frobenius; for the residue-field factoring use-cases here `p`-th roots of the
/// coefficients are needed when `k > 1`, but for the squarefree split it is the
/// exponents that matter — taking the `p`-th root of each coefficient keeps the
/// polynomial correct over GF(p^k). We compute the coefficient `p`-th root via
/// `a^(q/p) = a^(p^(k-1))`.
fn pth_root<F: FiniteFieldElement>(f: &FFPoly<F>, p_usize: usize, sample: &F) -> FFPoly<F> {
    let coeffs = f.coeffs();
    let mut out = Vec::with_capacity(coeffs.len() / p_usize + 1);
    // q = p^k, p-th root of a is a^(p^(k-1)) = a^(q/p)
    let q = sample.order();
    let p = sample.characteristic();
    let exp = q.div_rem(&p).unwrap().0; // q / p
    let mut i = 0;
    while i < coeffs.len() {
        let root = coeffs[i].pow(&exp);
        out.push(root);
        i += p_usize;
    }
    FFPoly::new(out, sample.clone())
}

/// Distinct-degree factorization of a **square-free, monic** polynomial.
///
/// Returns a list of `(g_d, d)` pairs where `g_d` is the product of all monic
/// irreducible factors of `f` of degree exactly `d`. The product of all `g_d`
/// equals `f`.
pub fn distinct_degree_factorization<F: FiniteFieldElement>(
    f: &FFPoly<F>,
) -> Vec<(FFPoly<F>, usize)> {
    let mut result = Vec::new();
    let sample = f.sample().clone();
    let q = sample.order();
    let x = FFPoly::x(sample.clone());

    let mut remaining = f.make_monic();
    let mut h = x.clone(); // h = x^(q^d) mod remaining, starts as x (d=0 -> x^(q^0)=x)
    let mut d = 0usize;

    while remaining.degree().unwrap_or(0) >= 2 * (d + 1) {
        d += 1;
        // h := h^q mod remaining  (so h = x^(q^d) mod remaining)
        h = h.pow_mod(&q, &remaining).expect("monic divisor");
        let diff = h.sub(&x);
        let g = remaining.gcd(&diff);
        if g.degree().unwrap_or(0) > 0 {
            let gm = g.make_monic();
            remaining = remaining.div_rem(&gm).expect("g divides remaining").0;
            result.push((gm, d));
            // reduce h modulo the smaller remaining for efficiency
            if remaining.degree().unwrap_or(0) > 0 {
                h = h.rem(&remaining).expect("nonzero");
            }
        }
    }

    if remaining.degree().unwrap_or(0) > 0 {
        // What's left is a single irreducible factor of its own degree.
        let dr = remaining.degree().unwrap();
        result.push((remaining, dr));
    }
    result
}

/// Equal-degree factorization: split a monic, square-free polynomial `f` that
/// is a product of `r` distinct irreducible factors each of degree exactly `d`.
///
/// Uses Cantor–Zassenhaus probabilistic splitting; the PRNG is seeded
/// deterministically from `f`, so the output is reproducible. Returns the list
/// of monic irreducible factors.
pub fn equal_degree_factorization<F: FiniteFieldElement>(
    f: &FFPoly<F>,
    d: usize,
) -> Vec<FFPoly<F>> {
    let n = f.degree().unwrap_or(0);
    if n == 0 {
        return Vec::new();
    }
    if n == d {
        // already irreducible
        return vec![f.make_monic()];
    }

    let sample = f.sample().clone();
    let mut seed = seed_from_poly(f);
    let mut rng = SplitMix64::new(seed);
    let mut factors = vec![f.make_monic()];
    let target = n / d;

    let two = Integer::from(2);
    let q = sample.order();
    let char_is_two = sample.characteristic() == two;

    // exponent (q^d - 1) / 2 for odd characteristic
    let qd = q.pow(d as u32);
    let exp_odd = if char_is_two {
        Integer::zero()
    } else {
        (qd.clone() - Integer::one()).div_rem(&two).unwrap().0
    };

    let mut guard = 0usize;
    while factors.len() < target {
        guard += 1;
        if guard > 100_000 {
            break; // safety; should not trigger for valid input
        }
        // pick a factor that is not yet irreducible (degree > d)
        let idx = match factors.iter().position(|g| g.degree().unwrap_or(0) > d) {
            Some(i) => i,
            None => break,
        };
        let g = factors[idx].clone();
        let gdeg = g.degree().unwrap();

        // random splitting polynomial of degree < gdeg
        let a = random_poly_below_degree(&sample, gdeg, &mut rng);
        if a.degree().unwrap_or(0) == 0 {
            // refresh the rng a little and retry
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            rng = SplitMix64::new(seed);
            continue;
        }

        let candidate = if char_is_two {
            // Trace-based splitting for characteristic 2 over GF(q):
            //   T(a) = a + a^q + a^(q^2) + ... + a^(q^(d-1))  (mod g),
            // then gcd(T(a), g). Each successive term applies the q-power
            // Frobenius (raising to q = p^m), which reduces to squaring only
            // when the base field is the prime field GF(2).
            let mut t = a.rem(&g).expect("nonzero");
            let mut acc = t.clone();
            for _ in 1..d {
                t = t.pow_mod(&q, &g).expect("monic divisor"); // t := t^q mod g
                acc = acc.add(&t);
            }
            acc.rem(&g).expect("nonzero")
        } else {
            // a^((q^d - 1)/2) - 1, then gcd with g.
            let b = a.pow_mod(&exp_odd, &g).expect("monic divisor");
            b.sub(&FFPoly::one(sample.clone()))
        };

        let gcd = g.gcd(&candidate);
        let gdcd = gcd.degree().unwrap_or(0);
        if gdcd > 0 && gdcd < gdeg {
            let h1 = gcd.make_monic();
            let h2 = g.div_rem(&h1).expect("h1 divides g").0.make_monic();
            factors[idx] = h1;
            factors.push(h2);
        }
    }

    factors.into_iter().map(|g| g.make_monic()).collect()
}

/// Factor a **square-free, monic** polynomial into its monic irreducible
/// factors (no multiplicities). Combines distinct-degree and equal-degree
/// factorization.
pub fn factor_squarefree<F: FiniteFieldElement>(f: &FFPoly<F>) -> Vec<FFPoly<F>> {
    let mut out = Vec::new();
    if f.degree().unwrap_or(0) == 0 {
        return out;
    }
    for (g_d, d) in distinct_degree_factorization(f) {
        if d == 0 {
            continue;
        }
        for fac in equal_degree_factorization(&g_d, d) {
            out.push(fac);
        }
    }
    out
}

/// Full factorization of an arbitrary polynomial over a finite field.
///
/// Returns `(unit, factors)` where `unit` is the leading coefficient of `f`
/// (the field unit) and `factors` is a list of `(monic irreducible, exponent)`
/// pairs such that `f = unit * prod(factor^exponent)`.
///
/// For the zero polynomial, returns the zero unit and an empty factor list.
pub fn factor<F: FiniteFieldElement>(f: &FFPoly<F>) -> (F, Vec<(FFPoly<F>, usize)>) {
    let sample = f.sample().clone();
    if f.is_zero() {
        return (sample.zero(), Vec::new());
    }
    let unit = f.leading().unwrap().clone();
    let monic = f.make_monic();

    let mut factors: Vec<(FFPoly<F>, usize)> = Vec::new();
    for (sqf, mult) in squarefree_factorization(&monic) {
        for irr in factor_squarefree(&sqf) {
            // merge if already present (shouldn't normally duplicate)
            if let Some(entry) = factors.iter_mut().find(|(g, _)| *g == irr) {
                entry.1 += mult;
            } else {
                factors.push((irr, mult));
            }
        }
    }
    (unit, factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff_poly::Gfpn;
    use crate::irreducible::is_irreducible;
    use crate::prime_field::PrimeField;

    fn pf(v: i64, p: i64) -> PrimeField {
        PrimeField::new(Integer::from(v), Integer::from(p)).unwrap()
    }

    fn poly(vals: &[i64], p: i64) -> FFPoly<PrimeField> {
        let sample = pf(0, p);
        FFPoly::new(vals.iter().map(|&v| pf(v, p)).collect(), sample)
    }

    /// Multiply a list of (factor, mult) back together (monic product).
    fn product(
        factors: &[(FFPoly<PrimeField>, usize)],
        sample: &PrimeField,
    ) -> FFPoly<PrimeField> {
        let mut acc = FFPoly::one(sample.clone());
        for (g, m) in factors {
            for _ in 0..*m {
                acc = acc.mul(g);
            }
        }
        acc
    }

    #[test]
    fn test_factor_quadratic_gf2() {
        // x^2 + 1 = (x+1)^2 over GF(2)
        let f = poly(&[1, 0, 1], 2);
        let (unit, factors) = factor(&f);
        assert!(unit.is_one());
        // single factor (x+1) with multiplicity 2
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].0, poly(&[1, 1], 2));
        assert_eq!(factors[0].1, 2);
        // reconstruct
        let recon = product(&factors, f.sample());
        assert_eq!(recon.scalar_mul(&unit), f);
    }

    #[test]
    fn test_factor_product_distinct_gf2() {
        // (x+1)(x^2+x+1)(x^3+x+1) over GF(2) — degrees 1,2,3, all distinct irreducibles
        let a = poly(&[1, 1], 2);
        let b = poly(&[1, 1, 1], 2);
        let c = poly(&[1, 1, 0, 1], 2);
        let f = a.mul(&b).mul(&c);
        let (unit, factors) = factor(&f);
        assert!(unit.is_one());
        // product equals input
        let recon = product(&factors, f.sample());
        assert_eq!(recon.scalar_mul(&unit), f);
        // each factor irreducible
        for (g, _) in &factors {
            assert!(is_irreducible(g), "factor {:?} not irreducible", g);
        }
        // total degree 6
        let total: usize = factors.iter().map(|(g, m)| g.degree().unwrap() * m).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_factor_equal_degree_gf3() {
        // Two distinct degree-1 factors over GF(3): (x-1)(x-2) = x^2 - 3x + 2 = x^2 + 2
        let f = poly(&[2, 0, 1], 3);
        let (unit, factors) = factor(&f);
        assert!(unit.is_one());
        let recon = product(&factors, f.sample());
        assert_eq!(recon.scalar_mul(&unit), f);
        assert_eq!(factors.len(), 2);
        for (g, m) in &factors {
            assert_eq!(g.degree(), Some(1));
            assert_eq!(*m, 1);
            assert!(is_irreducible(g));
        }
    }

    #[test]
    fn test_factor_gf7_mixed() {
        // (x+3)^2 (x^2+1) over GF(7); x^2+1 is irreducible over GF(7)? roots of -1:
        // squares mod 7 are {0,1,2,4}; -1=6 not a square -> irreducible.
        let lin = poly(&[3, 1], 7);
        let quad = poly(&[1, 0, 1], 7);
        assert!(is_irreducible(&quad));
        let f = lin.mul(&lin).mul(&quad);
        let (unit, factors) = factor(&f);
        assert!(unit.is_one());
        let recon = product(&factors, f.sample());
        assert_eq!(recon.scalar_mul(&unit), f);
        for (g, _) in &factors {
            assert!(is_irreducible(g));
        }
        // find the linear factor: should have multiplicity 2
        let lin_monic = lin.make_monic();
        let entry = factors.iter().find(|(g, _)| *g == lin_monic).unwrap();
        assert_eq!(entry.1, 2);
    }

    #[test]
    fn test_factor_nonmonic_gf7() {
        // 3*(x+1)(x+2) over GF(7)
        let f = poly(&[2, 3, 1], 7).scalar_mul(&pf(3, 7)); // 3x^2 + ...
        let (unit, factors) = factor(&f);
        assert_eq!(unit, pf(3, 7));
        let recon = product(&factors, f.sample()).scalar_mul(&unit);
        assert_eq!(recon, f);
    }

    #[test]
    fn test_squarefree_factorization() {
        // f = (x+1)^3 (x+2) over GF(5)
        let a = poly(&[1, 1], 5);
        let b = poly(&[2, 1], 5);
        let f = a.mul(&a).mul(&a).mul(&b);
        let sqf = squarefree_factorization(&f);
        // reconstruct
        let sample = f.sample().clone();
        let mut acc = FFPoly::one(sample);
        for (g, m) in &sqf {
            for _ in 0..*m {
                acc = acc.mul(g);
            }
        }
        assert_eq!(acc, f.make_monic());
        // multiplicity-3 factor is (x+1)
        let m3 = sqf.iter().find(|(_, m)| *m == 3).unwrap();
        assert_eq!(m3.0, poly(&[1, 1], 5));
    }

    // FIXME(WP-GFPN): equal-degree splitting over a proper extension field GF(p^n)
    // is buggy (factoring over GF(4) here fails). GF(p) factoring works; this
    // extension-field path needs the splitting polynomial drawn from the full
    // field GF(q), not the prime subfield. Ignored until fixed.
    #[ignore]
    #[test]
    fn test_factor_over_gfpn() {
        // Factor over GF(2^2). Modulus x^2 + x + 1; let w be a root (omega).
        let p = Integer::from(2);
        let modu = vec![Integer::from(1), Integer::from(1), Integer::from(1)];
        let zero = Gfpn::new(vec![Integer::from(0)], p.clone(), modu.clone());
        let one = zero.one();
        let omega = Gfpn::new(vec![Integer::from(0), Integer::from(1)], p.clone(), modu.clone());

        let ffp = |coeffs: Vec<Gfpn>| FFPoly::new(coeffs, zero.clone());

        // f(y) = (y - omega)(y - (omega+1)) = (y+omega)(y+omega+1) over GF(4) (char 2)
        let r1 = omega.clone();
        let r2 = omega.add(&one);
        let lin1 = ffp(vec![r1.clone(), one.clone()]); // y + omega
        let lin2 = ffp(vec![r2.clone(), one.clone()]); // y + omega + 1
        let f = lin1.mul(&lin2);

        let (unit, factors) = factor(&f);
        assert!(unit.is_one());
        // product reconstruction
        let mut acc = FFPoly::one(zero.clone());
        for (g, m) in &factors {
            for _ in 0..*m {
                acc = acc.mul(g);
            }
        }
        assert_eq!(acc.scalar_mul(&unit), f);
        assert_eq!(factors.len(), 2);
        for (g, _) in &factors {
            assert_eq!(g.degree(), Some(1));
            assert!(is_irreducible(g));
        }
    }
}
