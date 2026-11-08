//! Cantor–Zassenhaus factorization over finite fields GF(p) and GF(p^n).
//!
//! Specification: MAGMA Handbook ch. 23 *Univariate Polynomial Rings* §23.8.1
//! (`Factorization` over F_q, `SquarefreeFactorization`,
//! `DistinctDegreeFactorization`, `EqualDegreeFactorization(f, d, g)`; digest
//! `rustmath-lava/docs/Chapter_23_Univariate_Polynomial_Rings.md`) and ch. 21
//! *Finite Fields* §21.3.5 (`Roots`, `FactorizationOverSplittingField`; digest
//! `rustmath-lava/docs/Chapter_21_Finite_Fields.md`).
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
//!
//! Algorithms:
//! - [CZ81] D. G. Cantor, H. Zassenhaus, *A new algorithm for factoring
//!   polynomials over finite fields*, Math. Comp. 36 (1981) 587-592: random
//!   equal-degree splitting; the trace map in characteristic 2.
//! - [vzGG] J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, §14.3:
//!   Algorithm 14.8 (equal-degree splitting: a uniform `a` with
//!   `deg a < deg f`; return `gcd(a, f)` when it is proper (step 3); else
//!   `gcd(a^((Q-1)/2) - 1, f)`, `Q = q^d`, or in characteristic 2 the gcd with
//!   the absolute trace of `a`), Theorem 14.9 (a try fails with probability at
//!   most `2^(1-r) <= 1/2` on a product of `r >= 2` distinct irreducibles of
//!   degree `d`), Algorithm 14.10 (recursive equal-degree factorization).
//! - [Rab80] M. O. Rabin, *Probabilistic algorithms in finite fields*, SIAM J.
//!   Comput. 9 (1980) 273-280: `x^(q^d) - x` is the product of the monic
//!   irreducibles of degree dividing `d`; the equal-degree contract certificate
//!   is Rabin's criterion generalised to products.
//! - SplitMix64: G. L. Steele, D. Lea, C. H. Flood, *Fast splittable
//!   pseudorandom number generators*, OOPSLA 2014.
//!
//! **No unsplit block is ever returned.** The equal-degree loop draws uniform
//! splitting polynomials (`FiniteFieldElement::random_element`, exact over
//! every GF(q)); a block that resists `CERTIFY_AFTER` = 64 consecutive tries
//! is certified with the contract check (a violation panics with the reason),
//! and a certified block that resists `LAS_VEGAS_TRIES` = 256 further tries
//! panics (probability at most `2^-256` per block for a uniform generator).
//! [`equal_degree_factorization`] checks its contract eagerly and
//! [`factor_squarefree`] checks that its input is squarefree, so over a field
//! a contract violation of these entries panics instead of reporting a
//! reducible polynomial as irreducible.
//!
//! Preconditions: the coefficients must lie in a field. [`factor`],
//! [`factor_squarefree`] and [`equal_degree_factorization`] check that the
//! characteristic is a prime with `Integer::is_prime`: a `PrimeField` with
//! the modulus-0 sentinel or a composite modulus that the test detects panics
//! before any arithmetic. The test is a proof below
//! psi12 = 318665857834031151167461; above that it is a strong probable-prime
//! test to the bases 2..37, so a strong pseudoprime to all of them (psi12
//! itself, 399165290221 * 798330580441) passes and the results are then
//! unspecified. A [`crate::ff_poly::Gfpn`]
//! whose defining polynomial is reducible (not a field) is **not checked**;
//! the results are then unspecified (the lazy certificate, the cap or the
//! arithmetic may panic). [`squarefree_factorization`] and
//! [`distinct_degree_factorization`] check nothing.

use crate::ff_poly::{FFPoly, FiniteFieldElement};
use crate::irreducible::prime_divisors;
use rustmath_core::{EuclideanDomain, NumericConversion};
use rustmath_integers::Integer;

/// Consecutive failed splitting tries on one block after which the block is
/// certified with [`edf_contract`] (a valid block fails 64 uniform tries in a
/// row with probability at most `2^-64`).
const CERTIFY_AFTER: usize = 64;
/// Further uniform tries on a certified block before the loop gives up with a
/// panic (probability at most `2^-256` for a uniform generator over a field).
const LAS_VEGAS_TRIES: usize = 256;

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

/// A random polynomial of degree `< deg` whose coefficients are independent and
/// uniform over the **full** field GF(q) = GF(p^k) (if the words of `next` are),
/// so the polynomial is uniform among those of degree `< deg` ([vzGG] Alg. 14.8
/// step 1).
///
/// Coefficients must be arbitrary elements of GF(q), not just of the prime
/// subfield GF(p), and must cover all of GF(p) (the old draw gave only 0 or 1
/// for p > 2^64); `random_element` draws every prime-subfield coordinate by
/// exact rejection sampling.
fn random_poly_below_degree<F: FiniteFieldElement>(
    sample: &F,
    deg: usize,
    next: &mut dyn FnMut() -> u64,
) -> FFPoly<F> {
    let mut coeffs = Vec::with_capacity(deg);
    for _ in 0..deg {
        coeffs.push(sample.random_element(next));
    }
    FFPoly::new(coeffs, sample.clone())
}

/// `Ok(())` if `Integer::is_prime` accepts the characteristic of `sample`'s
/// field; otherwise the reason. Refuses the modulus-0 sentinel
/// (characteristic 0) and every composite modulus below psi12 =
/// 318665857834031151167461 (trial division below 1000, then strong
/// Miller-Rabin to the bases 2..37, a proof below psi12). Above psi12 a strong
/// pseudoprime to those bases, such as psi12 itself, is not detected.
fn check_prime_characteristic<F: FiniteFieldElement>(sample: &F) -> Result<(), String> {
    let c = sample.characteristic();
    if c.is_prime() {
        Ok(())
    } else {
        Err(format!("characteristic {c} is not a prime"))
    }
}

/// Contract certificate of the equal-degree stage ([Rab80], [vzGG] §14.2):
/// `Ok(())` iff `f` is a squarefree product of distinct monic irreducibles of
/// degree exactly `d` over GF(q), `q = f.sample().order()`, in a field of prime
/// characteristic; otherwise the first violated condition, in this order:
/// prime characteristic, `d >= 1`, `f != 0`, `deg f >= 1`, `d | deg f`,
/// squarefree (`f' != 0` and `gcd(f, f') = 1`), `x^(q^d) ≡ x (mod f)` (every
/// irreducible factor has degree dividing `d`), and `gcd(x^(q^(d/l)) - x, f) = 1`
/// for every prime `l | d` (no factor of degree dividing a proper divisor
/// `d/l`). Cost: `d` Frobenius powers `X -> X^q mod f` (about one odd-`q` try).
fn edf_contract<F: FiniteFieldElement>(f: &FFPoly<F>, d: usize) -> Result<(), String> {
    check_prime_characteristic(f.sample())?;
    block_contract(f, d)
}

/// [`edf_contract`] without its first (characteristic) test, for callers that
/// have run [`check_prime_characteristic`] already.
fn block_contract<F: FiniteFieldElement>(f: &FFPoly<F>, d: usize) -> Result<(), String> {
    if d < 1 {
        return Err(format!("d = {d} must be at least 1"));
    }
    let n = match f.degree() {
        None => return Err("f = 0 has no factorization".to_string()),
        Some(0) => return Err("f is constant".to_string()),
        Some(n) => n,
    };
    if n % d != 0 {
        return Err(format!("d = {d} does not divide deg f = {n}"));
    }
    let f = f.make_monic();
    let df = f.derivative();
    if df.is_zero() || f.gcd(&df).degree() != Some(0) {
        return Err("f is not squarefree".to_string());
    }
    // Frobenius ladder X_k = x^(q^k) mod f, k = 0..=d (d <= n here).
    let q = f.sample().order();
    let x0 = FFPoly::x(f.sample().clone()).rem(&f).expect("f != 0");
    let mut ladder = vec![x0.clone()];
    for k in 1..=d {
        let next = ladder[k - 1].pow_mod(&q, &f).expect("f != 0");
        ladder.push(next);
    }
    if ladder[d] != x0 {
        return Err(format!(
            "f has an irreducible factor whose degree does not divide {d}"
        ));
    }
    for l in prime_divisors(d) {
        let e = d / l;
        if f.gcd(&ladder[e].sub(&x0)).degree() != Some(0) {
            return Err(format!(
                "f has an irreducible factor of degree dividing {e}"
            ));
        }
    }
    Ok(())
}

/// Lazy certificate of the internal path: panics with the reason when the block
/// `g` violates the equal-degree contract.
fn certify_or_panic<F: FiniteFieldElement>(g: &FFPoly<F>, d: usize) {
    if let Err(why) = edf_contract(g, d) {
        panic!("cantor_zassenhaus: equal-degree block violates its contract ({why})");
    }
}

/// One try of [vzGG] Alg. 14.8 on the monic block `g` (degree `n > d`) with
/// the splitting polynomial `a` (`deg a < n`): step 3 (`gcd(a, g)` proper), then
/// in odd characteristic `gcd(a^((q^d-1)/2) - 1, g)` (`exp_odd`), in
/// characteristic 2 `gcd(T(a), g)` with `T` the absolute trace down to GF(2)
/// (`n_ext * d` terms). Returns a proper monic divisor of `g`, or `None`.
fn split_once<F: FiniteFieldElement>(
    g: &FFPoly<F>,
    a: &FFPoly<F>,
    d: usize,
    n_ext: usize,
    char_is_two: bool,
    exp_odd: &Integer,
) -> Option<FFPoly<F>> {
    let n = g.degree()?;
    let proper = |h: &FFPoly<F>| matches!(h.degree(), Some(k) if k > 0 && k < n);
    // Step 3: a shares a proper factor with g (also catches a = 0: gcd = g).
    let g0 = g.gcd(a);
    if proper(&g0) {
        return Some(g0);
    }
    let candidate = if char_is_two {
        // Trace-based splitting for characteristic 2. The roots of g live in
        // GF(2^(n_ext*d)) (n_ext = [GF(q):GF(2)], d the common factor degree),
        // so we use the absolute trace down to GF(2):
        //   Tr(a) = a + a^2 + a^4 + ... + a^(2^(n_ext*d - 1))  (mod g),
        // i.e. (n_ext*d) terms, accumulated with (n_ext*d - 1) squarings.
        // Tr is GF(2)-linear and onto at each root, so for a uniform a the
        // values Tr(a)(root) are independent and uniform in GF(2).
        // NOTE: squaring count is n_ext*d, NOT d — tracing only to GF(q) (d-1
        // q-power steps) does not map into GF(2) and fails to split.
        let squarings = n_ext * d;
        let mut t = a.rem(g).expect("nonzero");
        let mut acc = t.clone();
        for _ in 1..squarings {
            t = t.mul(&t).rem(g).expect("monic divisor"); // t := t^2 mod g
            acc = acc.add(&t);
        }
        acc.rem(g).expect("nonzero")
    } else {
        // a^((q^d - 1)/2) - 1, then gcd with g.
        let b = a.pow_mod(exp_odd, g).expect("monic divisor");
        b.sub(&FFPoly::one(g.sample().clone()))
    };
    let h = g.gcd(&candidate);
    if proper(&h) {
        Some(h)
    } else {
        None
    }
}

/// Equal-degree factorization of a monic `f` that the caller has checked (or
/// that comes from the distinct-degree split of a squarefree polynomial over a
/// field of prime characteristic) to be a product of distinct monic
/// irreducibles of degree `d`; seeded from `f`.
fn edf_unchecked<F: FiniteFieldElement>(f: &FFPoly<F>, d: usize) -> Vec<FFPoly<F>> {
    let mut rng = SplitMix64::new(seed_from_poly(f));
    edf_with_source(f, d, &mut || rng.next_u64())
}

/// The equal-degree loop of [`edf_unchecked`] with an explicit word source.
///
/// While some factor has degree `> d`, one uniform try ([`split_once`]) is
/// made on the first such factor; a proper divisor replaces it by its two
/// parts. `failures` counts the consecutive failed tries on that block (reset
/// on a split; a constant or zero `a` is an ordinary failed try): at
/// `CERTIFY_AFTER` the block is certified ([`certify_or_panic`]), at
/// `CERTIFY_AFTER + LAS_VEGAS_TRIES` the loop panics. At the end every factor
/// must have degree exactly `d` (else the input violated the contract: panic).
/// No path returns a block of degree `> d`.
fn edf_with_source<F: FiniteFieldElement>(
    f: &FFPoly<F>,
    d: usize,
    next: &mut dyn FnMut() -> u64,
) -> Vec<FFPoly<F>> {
    let n = f.degree().unwrap_or(0);
    if n == 0 {
        return Vec::new();
    }
    if n == d {
        // Sound only on the guarded paths (see `edf_unchecked`): the public
        // entry certifies first, and a distinct-degree block of a squarefree
        // polynomial of degree d is irreducible.
        return vec![f.make_monic()];
    }

    let sample = f.sample().clone();
    let mut factors = vec![f.make_monic()];

    let two = Integer::from(2);
    let q = sample.order();
    let char_is_two = sample.characteristic() == two;
    // Extension degree n of GF(q) = GF(p^n) over the prime field GF(p).
    let n_ext = sample.degree_over_prime();

    // exponent (q^d - 1) / 2 for odd characteristic, where q = p^n is the FULL
    // field order (NOT the characteristic p). This is essential over a proper
    // extension: using p here instead of q fails to split factors.
    let exp_odd = if char_is_two {
        Integer::zero()
    } else {
        (q.pow(d as u32) - Integer::one()).div_rem(&two).unwrap().0
    };

    let mut failures = 0usize;
    while let Some(idx) = factors.iter().position(|g| g.degree().unwrap_or(0) > d) {
        let g = factors[idx].clone();
        let gdeg = g.degree().unwrap();
        let a = random_poly_below_degree(&sample, gdeg, next);
        match split_once(&g, &a, d, n_ext, char_is_two, &exp_odd) {
            Some(h1) => {
                let h2 = g.div_rem(&h1).expect("h1 divides g").0.make_monic();
                factors[idx] = h1;
                factors.push(h2);
                failures = 0;
            }
            None => {
                failures += 1;
                if failures == CERTIFY_AFTER {
                    certify_or_panic(&g, d);
                }
                if failures >= CERTIFY_AFTER + LAS_VEGAS_TRIES {
                    panic!(
                        "cantor_zassenhaus: no split of a certified equal-degree block of \
                         degree {gdeg} (d = {d}) in {LAS_VEGAS_TRIES} uniform tries \
                         (probability <= 2^-{LAS_VEGAS_TRIES} over a field); the generator \
                         is broken or the coefficients do not lie in a field"
                    );
                }
            }
        }
    }
    if let Some(bad) = factors.iter().find(|g| g.degree() != Some(d)) {
        panic!(
            "cantor_zassenhaus: equal-degree block violates its contract (a factor of \
             degree {:?}, expected {d})",
            bad.degree()
        );
    }
    factors
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
///
/// The square-free precondition is **not checked** here: for a non-squarefree
/// input the blocks need not be products of distinct irreducibles of degree
/// `d` (the last one is not even tested). [`factor_squarefree`] checks it
/// before calling this.
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

/// Equal-degree factorization (MAGMA `EqualDegreeFactorization`, ch. 23
/// §23.8.1): split a square-free polynomial `f` that is a product of `r`
/// distinct irreducible factors each of degree exactly `d` into its monic
/// irreducible factors.
///
/// Uses Cantor–Zassenhaus splitting with uniform splitting polynomials
/// ([vzGG] Alg. 14.8/14.10); the PRNG is seeded deterministically from `f`, so
/// the output (in discovery order) is reproducible. The contract is checked
/// eagerly (MAGMA's result is "unpredictable" outside it; here it panics), so
/// a reducible block is never returned as a factor. `f` need not be monic.
///
/// A nonzero constant `f` gives the empty product `vec![]`.
///
/// # Panics
/// In this order: the characteristic is not a prime (before any arithmetic;
/// detected as described in the module docs, a proof below psi12);
/// `f = 0`; `d < 1`; `d` does not divide `deg f`; `f` is not squarefree; `f`
/// has an irreducible factor of degree other than `d` (including a reducible
/// `f` of degree `d`) — each with the reason. Also (probability at most
/// `2^-256` over a field) when a certified block resists 256 uniform tries.
pub fn equal_degree_factorization<F: FiniteFieldElement>(
    f: &FFPoly<F>,
    d: usize,
) -> Vec<FFPoly<F>> {
    const WHO: &str = "cantor_zassenhaus::equal_degree_factorization";
    if let Err(why) = check_prime_characteristic(f.sample()) {
        panic!("{WHO}: {why}");
    }
    if f.is_zero() {
        panic!("{WHO}: f = 0 has no factorization");
    }
    if f.degree() == Some(0) {
        return Vec::new();
    }
    let fm = f.make_monic();
    if let Err(why) = block_contract(&fm, d) {
        panic!("{WHO}: {why}");
    }
    edf_unchecked(&fm, d)
}

/// The body of [`factor_squarefree`] after the characteristic test: the
/// constant early return, the squarefree check (`Err` names the violation),
/// then the distinct-degree split and the equal-degree loop on every block.
fn factor_squarefree_prime_char<F: FiniteFieldElement>(
    f: &FFPoly<F>,
) -> Result<Vec<FFPoly<F>>, &'static str> {
    let mut out = Vec::new();
    if f.degree().unwrap_or(0) == 0 {
        return Ok(out);
    }
    let fm = f.make_monic();
    // Over GF(q) a nonconstant f with f' = 0 is a p-th power, so this test is
    // exact: f is squarefree iff f' != 0 and gcd(f, f') = 1.
    let df = fm.derivative();
    if df.is_zero() || fm.gcd(&df).degree() != Some(0) {
        return Err("f is not squarefree");
    }
    // With f squarefree over a field, every distinct-degree block is a product
    // of distinct irreducibles of its degree, so the equal-degree loop's
    // preconditions hold (it still certifies lazily).
    for (g_d, d) in distinct_degree_factorization(&fm) {
        if d == 0 {
            continue;
        }
        out.extend(edf_unchecked(&g_d, d));
    }
    Ok(out)
}

/// Factor a **square-free** polynomial into its monic irreducible factors (no
/// multiplicities; `f` need not be monic). Combines distinct-degree and
/// equal-degree factorization. A constant `f` (also `f = 0`) gives `vec![]`.
///
/// # Panics
/// If the characteristic is not a prime (checked first, a proof below psi12;
/// see the module docs), or if `f` is not
/// squarefree (`f' = 0` or `deg gcd(f, f') > 0`; without this check such an
/// input came back with a reducible "factor", e.g. `(x-1)^2` for `(x-1)^3`
/// over GF(5)). Also (probability at most `2^-256` over a field) when a certified
/// block resists 256 uniform tries.
pub fn factor_squarefree<F: FiniteFieldElement>(f: &FFPoly<F>) -> Vec<FFPoly<F>> {
    const WHO: &str = "cantor_zassenhaus::factor_squarefree";
    if let Err(why) = check_prime_characteristic(f.sample()) {
        panic!("{WHO}: {why}");
    }
    factor_squarefree_prime_char(f).unwrap_or_else(|why| panic!("{WHO}: {why}"))
}

/// Full factorization of an arbitrary polynomial over a finite field (MAGMA
/// `Factorization`, ch. 23 §23.8.1).
///
/// Returns `(unit, factors)` where `unit` is the leading coefficient of `f`
/// (the field unit) and `factors` is a list of `(monic irreducible, exponent)`
/// pairs such that `f = unit * prod(factor^exponent)`.
///
/// For the zero polynomial, returns the zero unit and an empty factor list.
///
/// # Panics
/// If the characteristic is not a prime (checked first, before any
/// arithmetic, a proof below psi12; see the module docs: a composite `PrimeField` modulus could otherwise make the
/// squarefree split return only constant pieces, i.e. an empty factor list for
/// a nonconstant `f`). Also (probability at most `2^-256` over a field) when a
/// certified equal-degree block resists 256 uniform tries.
pub fn factor<F: FiniteFieldElement>(f: &FFPoly<F>) -> (F, Vec<(FFPoly<F>, usize)>) {
    if let Err(why) = check_prime_characteristic(f.sample()) {
        panic!("cantor_zassenhaus::factor: {why}");
    }
    let sample = f.sample().clone();
    if f.is_zero() {
        return (sample.zero(), Vec::new());
    }
    let unit = f.leading().unwrap().clone();
    let monic = f.make_monic();

    let mut factors: Vec<(FFPoly<F>, usize)> = Vec::new();
    for (sqf, mult) in squarefree_factorization(&monic) {
        let irreducibles = factor_squarefree_prime_char(&sqf).unwrap_or_else(|why| {
            panic!(
                "cantor_zassenhaus::factor: a part returned by squarefree_factorization \
                 fails the squarefree check ({why}); this is a bug"
            )
        });
        for irr in irreducibles {
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

    // Helper: build a Gfpn element in GF(p^n) from low-degree-first coefficients.
    fn gfpn(coeffs: &[i64], p: i64, modu: &[i64]) -> Gfpn {
        Gfpn::new(
            coeffs.iter().map(|&v| Integer::from(v)).collect(),
            Integer::from(p),
            modu.iter().map(|&v| Integer::from(v)).collect(),
        )
    }

    /// Multiply a list of (factor, mult) pairs over an arbitrary finite field.
    fn product_gfpn(factors: &[(FFPoly<Gfpn>, usize)], sample: &Gfpn) -> FFPoly<Gfpn> {
        let mut acc = FFPoly::one(sample.clone());
        for (g, m) in factors {
            for _ in 0..*m {
                acc = acc.mul(g);
            }
        }
        acc
    }

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

    #[test]
    fn test_factor_over_gf9_odd() {
        // Odd-characteristic extension GF(3^2) = GF(9), modulus x^2 + 1.
        // Build f(y) = (y - a)(y - b) with two DISTINCT elements of GF(9):
        //   a = i (the root of x^2+1), b = 1 + i.
        let p = 3;
        let modu = [1, 0, 1]; // x^2 + 1
        let zero = gfpn(&[0], p, &modu);
        let one = zero.one();
        let i = gfpn(&[0, 1], p, &modu); // x  (a root of x^2+1)
        let b = i.add(&one); // 1 + i

        let ffp = |coeffs: Vec<Gfpn>| FFPoly::new(coeffs, zero.clone());
        let lin1 = ffp(vec![i.neg(), one.clone()]); // y - i
        let lin2 = ffp(vec![b.neg(), one.clone()]); // y - (1+i)
        let f = lin1.mul(&lin2);

        let (unit, factors) = factor(&f);
        assert!(unit.is_one());
        let recon = product_gfpn(&factors, &zero).scalar_mul(&unit);
        assert_eq!(recon, f);
        assert_eq!(factors.len(), 2);
        for (g, _) in &factors {
            assert_eq!(g.degree(), Some(1));
            assert!(is_irreducible(g));
        }
        // both roots recovered: f vanishes at i and at b
        assert!(f.evaluate(&i).is_zero());
        assert!(f.evaluate(&b).is_zero());
    }

    #[test]
    fn test_split_irreducible_over_extension_gf8() {
        // The "root of g in GF(p^M)" operation P2 needs: a cubic that is
        // irreducible over GF(2) splits into 3 distinct LINEAR factors over
        // GF(2^3) = GF(8) (since 3 | 3, the field contains all its roots).
        //
        // g(y) = y^3 + y + 1 is irreducible over GF(2). Over GF(8) it has 3
        // distinct roots (a primitive element and its Frobenius conjugates),
        // so it must factor into 3 linear factors.
        let p = 2;
        let modu = [1, 1, 0, 1]; // x^3 + x + 1, irreducible over GF(2) -> GF(8)
        let zero = gfpn(&[0], p, &modu);
        let one = zero.one();
        let ffp = |coeffs: Vec<Gfpn>| FFPoly::new(coeffs, zero.clone());

        // g(y) = y^3 + y + 1 with coefficients in GF(8) (all in the prime field)
        let g = ffp(vec![one.clone(), one.clone(), zero.clone(), one.clone()]);

        let (unit, factors) = factor(&g);
        assert!(unit.is_one());
        let recon = product_gfpn(&factors, &zero).scalar_mul(&unit);
        assert_eq!(recon, g);
        // exactly 3 distinct linear factors
        assert_eq!(factors.len(), 3, "expected 3 linear factors, got {:?}", factors);
        for (h, m) in &factors {
            assert_eq!(h.degree(), Some(1));
            assert_eq!(*m, 1);
            assert!(is_irreducible(h));
        }
        // the roots: a linear factor y - r means r = -(constant); each must be a
        // genuine root of g, and all three must be distinct.
        let mut roots: Vec<Gfpn> = Vec::new();
        for (h, _) in &factors {
            // h = c0 + c1*y, monic => c1 = 1, root r = -c0
            let r = h.coeffs()[0].neg();
            assert!(g.evaluate(&r).is_zero(), "{:?} is not a root of g", r);
            roots.push(r);
        }
        for a in 0..roots.len() {
            for b2 in (a + 1)..roots.len() {
                assert!(roots[a] != roots[b2], "roots not distinct");
            }
        }
    }

    // ---- X1 (equal-degree honesty) tests: CZ1, CZ4-CZ7 and the internal path ----
    // Reference values: PARI/GP 2.15 (`gp -q`), commands in each comment.

    fn big(s: &str) -> Integer {
        Integer::from_decimal_str(s).unwrap()
    }

    /// Polynomial over GF(p) (any size of p) from little-endian integer coefficients.
    fn poly_big(vals: &[Integer], p: &Integer) -> FFPoly<PrimeField> {
        let c = |v: &Integer| PrimeField::new(v.clone(), p.clone()).unwrap();
        FFPoly::new(vals.iter().map(c).collect(), c(&Integer::zero()))
    }

    /// Little-endian coefficient values of a polynomial over GF(p).
    fn vals(g: &FFPoly<PrimeField>) -> Vec<Integer> {
        g.coeffs().iter().map(|c| c.value().clone()).collect()
    }

    fn sorted_vals(fs: &[FFPoly<PrimeField>]) -> Vec<Vec<Integer>> {
        let mut out: Vec<Vec<Integer>> = fs.iter().map(vals).collect();
        out.sort();
        out
    }

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&c| Integer::from(c)).collect()
    }

    /// CZ1: (x-1)(x-4) over GF(p), p = 2^64 + 37 = 18446744073709551653, the first
    /// prime > 2^64 with (2/p) = (5/p) = -1. At HEAD the splitting coefficients were
    /// only 0 or 1 for p > 2^64, neither a = x nor a = x + 1 splits, and after
    /// 100,000 tries (about 105 s) `factor` returned the reducible quadratic
    /// x^2 + (p-5)x + 4 as one irreducible factor.
    /// gp: p = nextprime(2^64); while(kronecker(2,p) != -1 || kronecker(5,p) != -1,
    ///     p = nextprime(p+1)); factormod((x-1)*(x-4), p)
    ///     \\ [x + 18446744073709551649, 1; x + 18446744073709551652, 1]
    #[test]
    fn factor_over_prime_field_above_2_64() {
        let p = big("18446744073709551653");
        let f = poly_big(
            &[
                Integer::from(4),
                p.clone() - Integer::from(5),
                Integer::one(),
            ],
            &p,
        );
        let (unit, fac) = factor(&f);
        assert!(unit.is_one());
        let mut got: Vec<(Vec<Integer>, usize)> = fac.iter().map(|(g, m)| (vals(g), *m)).collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                (vec![big("18446744073709551649"), Integer::one()], 1),
                (vec![big("18446744073709551652"), Integer::one()], 1),
            ]
        );
        // the equal-degree entry on the same block
        let lin = equal_degree_factorization(&f, 1);
        assert_eq!(
            sorted_vals(&lin),
            vec![
                vec![big("18446744073709551649"), Integer::one()],
                vec![big("18446744073709551652"), Integer::one()],
            ]
        );
    }

    /// `y^e - y` over the field of `zero`.
    fn y_pow_minus_y(e: usize, zero: &Gfpn) -> FFPoly<Gfpn> {
        let mut c = vec![zero.clone(); e + 1];
        c[1] = zero.one().neg();
        c[e] = zero.one();
        FFPoly::new(c, zero.clone())
    }

    /// Factor `f` over GF(q) and check: unit 1, `n1` linear and `n2` quadratic
    /// factors and nothing else, all multiplicities 1, every factor irreducible
    /// (Rabin) and the product equal to `f`.
    fn check_split_into_lin_quad(f: &FFPoly<Gfpn>, zero: &Gfpn, n1: usize, n2: usize) {
        let (unit, fac) = factor(f);
        assert!(unit.is_one());
        assert_eq!(fac.len(), n1 + n2);
        assert_eq!(
            fac.iter().filter(|(g, _)| g.degree() == Some(1)).count(),
            n1
        );
        assert_eq!(
            fac.iter().filter(|(g, _)| g.degree() == Some(2)).count(),
            n2
        );
        assert!(fac.iter().all(|(_, m)| *m == 1));
        for (g, _) in &fac {
            assert!(g.is_monic());
            assert!(is_irreducible(g), "{:?} is not irreducible", g);
        }
        assert_eq!(product_gfpn(&fac, zero), *f);
    }

    /// CZ4: y^(q^2) - y over GF(q) is the product of the q monic linears and the
    /// (q^2 - q)/2 monic irreducible quadratics.
    /// gp: F = factorff(y^16 - y, 2, t^2 + t + 1)  \\ 10 factors, degrees [1,1,1,1,2,2,2,2,2,2]
    ///     F = factorff(y^81 - y, 3, t^2 + 1)      \\ 45 factors: 9 linear, 36 quadratic
    ///     F = factorff(y^64 - y, 2, t^3 + t + 1)  \\ 36 factors: 8 linear, 28 quadratic
    /// (all multiplicities 1).
    #[test]
    fn factor_full_splitting_over_extension_fields() {
        let gf4 = gfpn(&[0], 2, &[1, 1, 1]);
        check_split_into_lin_quad(&y_pow_minus_y(16, &gf4), &gf4, 4, 6);
        let gf9 = gfpn(&[0], 3, &[1, 0, 1]);
        check_split_into_lin_quad(&y_pow_minus_y(81, &gf9), &gf9, 9, 36);
        let gf8 = gfpn(&[0], 2, &[1, 1, 0, 1]);
        check_split_into_lin_quad(&y_pow_minus_y(64, &gf8), &gf8, 8, 28);
    }

    /// CZ5: x^4 + x^3 + 2 is irreducible over GF(3), so no product of quadratics;
    /// at HEAD `equal_degree_factorization(f, 2)` looped 100,000 times and returned
    /// [f]. gp: polisirreducible(Mod(1,3)*(x^4+x^3+2)) \\ 1
    #[test]
    #[should_panic(expected = "does not divide 2")]
    fn edf_rejects_irreducible_quartic_for_d_2() {
        let _ = equal_degree_factorization(&poly(&[2, 0, 0, 1, 1], 3), 2);
    }

    /// CZ5: (x+1)(x+2) over GF(5) with d = 2; at HEAD the `n == d` shortcut
    /// returned it at once as one irreducible quadratic.
    /// gp: factormod((x+1)*(x+2), 5) \\ two linear factors
    #[test]
    #[should_panic(expected = "degree dividing 1")]
    fn edf_rejects_reducible_block_of_degree_d() {
        let _ = equal_degree_factorization(&poly(&[2, 3, 1], 5), 2);
    }

    /// CZ5: a composite modulus is refused before any arithmetic over Z/15.
    /// gp: isprime(15) \\ 0
    #[test]
    #[should_panic(expected = "characteristic 15 is not a prime")]
    fn edf_rejects_composite_characteristic() {
        let _ = equal_degree_factorization(&poly(&[1, 0, 1], 15), 1);
    }

    #[test]
    #[should_panic(expected = "d = 0 must be at least 1")]
    fn edf_rejects_d_zero() {
        let _ = equal_degree_factorization(&poly(&[1, 0, 1], 5), 0);
    }

    #[test]
    #[should_panic(expected = "f = 0 has no factorization")]
    fn edf_rejects_zero_polynomial() {
        let _ = equal_degree_factorization(&poly(&[], 5), 1);
    }

    #[test]
    fn edf_of_a_constant_is_the_empty_product() {
        assert!(equal_degree_factorization(&poly(&[3], 5), 1).is_empty());
        // valid blocks, including the n == d case after the eager certificate
        // x^2 + 2 is irreducible over GF(5): gp polisirreducible(Mod(1,5)*(x^2+2)) = 1
        let quad = equal_degree_factorization(&poly(&[2, 0, 1], 5), 2);
        assert_eq!(sorted_vals(&quad), vec![ints(&[2, 0, 1])]);
        let lin = equal_degree_factorization(&poly(&[2, 3, 1], 5).scalar_mul(&pf(3, 5)), 1);
        assert_eq!(sorted_vals(&lin), vec![ints(&[1, 1]), ints(&[2, 1])]);
    }

    #[test]
    #[should_panic(expected = "cantor_zassenhaus::factor: characteristic 15 is not a prime")]
    fn factor_rejects_composite_characteristic() {
        let _ = factor(&poly(&[1, 0, 1], 15));
    }

    #[test]
    #[should_panic(
        expected = "cantor_zassenhaus::factor_squarefree: characteristic 15 is not a prime"
    )]
    fn factor_squarefree_rejects_composite_characteristic() {
        let _ = factor_squarefree(&poly(&[1, 0, 1], 15));
    }

    /// The modulus-0 sentinel (`Ring::zero()` of `PrimeField`) is no field either.
    #[test]
    #[should_panic(expected = "characteristic 0 is not a prime")]
    fn factor_rejects_the_modulus_zero_sentinel() {
        use rustmath_core::Ring;
        let z = <PrimeField as Ring>::zero();
        let o = <PrimeField as Ring>::one();
        let _ = factor(&FFPoly::new(vec![o.clone(), z.clone(), o], z));
    }

    /// CZ6: the fp_factor T1 rows A2, B2, C3, D2 over `PrimeField` through `factor`.
    /// gp: le(g) = Vecrev(lift(g)); F = factormod(Polrev(v), p);
    ///     vecsort(vector(#F~, i, le(F[i,1])))
    ///   A2 v = [1,1,1,1,1,1,1], p = 2           -> [[1,0,1,1],[1,1,0,1]]
    ///   B2 v = [1,0,1,0,1,0,1], p = 3           -> [[1,0,1],[2,1,1],[2,2,1]]
    ///   C3 v = [4,3,1,1,2,0,1,4,0,1], p = 5     -> [[1,1,0,1],[1,2,0,1],[4,1,0,1]]
    ///   D2 v = [8,0,14,0,7,0,1], p = 2147483647 -> [[1,0,1],[2,0,1],[4,0,1]]
    /// (all multiplicities 1). Run it only after the fix: on the old code row D2
    /// takes hours, because `PrimeField` had no `from_int` override and the O(p)
    /// default cost about 2^30 additions per random coefficient.
    #[test]
    fn factor_matches_pari_on_equal_degree_rows() {
        let rows: Vec<(i64, Vec<i64>, Vec<Vec<i64>>)> = vec![
            (
                2,
                vec![1, 1, 1, 1, 1, 1, 1],
                vec![vec![1, 0, 1, 1], vec![1, 1, 0, 1]],
            ),
            (
                3,
                vec![1, 0, 1, 0, 1, 0, 1],
                vec![vec![1, 0, 1], vec![2, 1, 1], vec![2, 2, 1]],
            ),
            (
                5,
                vec![4, 3, 1, 1, 2, 0, 1, 4, 0, 1],
                vec![vec![1, 1, 0, 1], vec![1, 2, 0, 1], vec![4, 1, 0, 1]],
            ),
            (
                2147483647,
                vec![8, 0, 14, 0, 7, 0, 1],
                vec![vec![1, 0, 1], vec![2, 0, 1], vec![4, 0, 1]],
            ),
        ];
        for (p, f, expected) in rows {
            let (unit, fac) = factor(&poly(&f, p));
            assert!(unit.is_one());
            assert!(fac.iter().all(|(_, m)| *m == 1), "p = {p}");
            let irr: Vec<FFPoly<PrimeField>> = fac.into_iter().map(|(g, _)| g).collect();
            let expected: Vec<Vec<Integer>> = expected.iter().map(|v| ints(v)).collect();
            assert_eq!(sorted_vals(&irr), expected, "p = {p}");
        }
    }

    /// CZ7 (review R1): (x-1)^3 = x^3 + 2x^2 + 3x + 4 over GF(5) is not squarefree;
    /// at HEAD `factor_squarefree` returned [x + 4, x^2 + 3x + 1] with
    /// x^2 + 3x + 1 = (x-1)^2 reported irreducible.
    /// gp: f = Mod(1,5)*(x-1)^3; Vecrev(lift(f)) \\ [4,3,2,1]; poldegree(gcd(f, f')) \\ 2
    #[test]
    #[should_panic(expected = "cantor_zassenhaus::factor_squarefree: f is not squarefree")]
    fn factor_squarefree_rejects_a_cube() {
        let _ = factor_squarefree(&poly(&[4, 3, 2, 1], 5));
    }

    /// CZ7, second variant of R1 (the count-based exit): (x+1)^2 (x-1)(x^2+2)(x^2+3)
    /// over GF(5); at HEAD `factor_squarefree` returned
    /// [x + 4, x + 1, x + 1, x^4 + 1] with x^4 + 1 = (x^2+2)(x^2+3) reducible.
    #[test]
    #[should_panic(expected = "f is not squarefree")]
    fn factor_squarefree_rejects_a_square_times_an_equal_degree_block() {
        let f = poly(&[1, 2, 1], 5).mul(&poly(&[4, 1, 0, 0, 4, 1], 5));
        let _ = factor_squarefree(&f);
    }

    /// CZ7 companions: `factor` of the cube still works (it passes squarefree parts),
    /// and a squarefree input still factors.
    /// gp: factormod((x-1)^3, 5) \\ [x + 4, 3]
    ///     g = (x-1)*(x^2+2)*(x^2+3); Vecrev(lift(Mod(1,5)*g)) \\ [4,1,0,0,4,1]
    ///     factormod(g, 5) \\ x + 4, x^2 + 2, x^2 + 3, each multiplicity 1
    #[test]
    fn factor_of_a_cube_and_factor_squarefree_of_a_squarefree_input() {
        let (unit, fac) = factor(&poly(&[4, 3, 2, 1], 5));
        assert!(unit.is_one());
        let got: Vec<(Vec<Integer>, usize)> = fac.iter().map(|(g, m)| (vals(g), *m)).collect();
        assert_eq!(got, vec![(ints(&[4, 1]), 3)]);
        let fs = factor_squarefree(&poly(&[4, 1, 0, 0, 4, 1], 5));
        assert_eq!(
            sorted_vals(&fs),
            vec![ints(&[2, 0, 1]), ints(&[3, 0, 1]), ints(&[4, 1])]
        );
        // not monic: 3 * that polynomial gives the same monic factors
        let fs3 = factor_squarefree(&poly(&[4, 1, 0, 0, 4, 1], 5).scalar_mul(&pf(3, 5)));
        assert_eq!(sorted_vals(&fs3), sorted_vals(&fs));
    }

    /// Internal path: a block that no try can split (x^4 + x^3 + 2 is irreducible
    /// over GF(3), so GF(3)[x]/(f) is a field and every gcd is 1 or f) reaches the
    /// lazy certificate after 64 failures and panics with the reason (HEAD: the
    /// block came back unsplit after 100,000 tries).
    #[test]
    #[should_panic(
        expected = "violates its contract (f has an irreducible factor whose degree does not divide 2)"
    )]
    fn edf_loop_certifies_a_block_it_cannot_split() {
        let _ = edf_unchecked(&poly(&[2, 0, 0, 1, 1], 3), 2);
    }

    /// Internal path, d = 0: every try fails (the exponent is 0), the certificate
    /// names the violation.
    #[test]
    #[should_panic(expected = "violates its contract (d = 0 must be at least 1)")]
    fn edf_loop_certifies_d_zero() {
        let _ = edf_unchecked(&poly(&[2, 3, 1], 5), 0);
    }

    /// The cap: with a stuck word source (every coefficient 0, so a = 0 on every
    /// try) a valid, certified block is never split, and the loop panics after
    /// 64 + 256 failures instead of returning it.
    #[test]
    #[should_panic(expected = "no split of a certified equal-degree block")]
    fn edf_loop_cap_panics_with_a_stuck_generator() {
        let _ = edf_with_source(&poly(&[4, 0, 1], 5), 1, &mut || 0u64);
    }

    /// The certificate itself, on valid and invalid blocks (gp facts as above;
    /// x^9 - x = x(x+1)(x+2)(x^2+1)(x^2+x+2)(x^2+2x+2) over GF(3), gp factormod).
    #[test]
    fn edf_contract_on_valid_and_invalid_blocks() {
        // valid: (x^9 - x)/(x^3 - x) over GF(3), d = 2 (three quadratics)
        assert_eq!(edf_contract(&poly(&[1, 0, 1, 0, 1, 0, 1], 3), 2), Ok(()));
        // valid: x^4 + x^3 + 2 with d = 4
        assert_eq!(edf_contract(&poly(&[2, 0, 0, 1, 1], 3), 4), Ok(()));
        // x^9 - x mixes degrees 1 and 2
        assert_eq!(
            edf_contract(&poly(&[0, 2, 0, 0, 0, 0, 0, 0, 0, 1], 3), 1),
            Err("f has an irreducible factor whose degree does not divide 1".to_string())
        );
        assert_eq!(
            edf_contract(&poly(&[0, 2, 0, 0, 0, 0, 0, 0, 0, 1], 3), 3),
            Err("f has an irreducible factor whose degree does not divide 3".to_string())
        );
        assert_eq!(
            edf_contract(&poly(&[1, 1], 3), 2),
            Err("d = 2 does not divide deg f = 1".to_string())
        );
        assert_eq!(
            edf_contract(&poly(&[1, 2, 1], 3), 1),
            Err("f is not squarefree".to_string())
        );
        assert_eq!(
            edf_contract(&poly(&[3], 5), 1),
            Err("f is constant".to_string())
        );
        assert_eq!(
            edf_contract(&poly(&[1, 0, 1], 15), 1),
            Err("characteristic 15 is not a prime".to_string())
        );
    }
}
