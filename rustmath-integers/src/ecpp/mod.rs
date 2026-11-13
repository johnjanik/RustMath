//! Elliptic curve primality proving (MAGMA Handbook ch. 18 §18.9.1 `IsPrime` with
//! `Proof := true`, `PrimalityCertificate`, `IsPrimeCertificate`; digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`).
//!
//! Method: the Atkin-Morain test [AM93] A. O. L. Atkin, F. Morain, *Elliptic curves and
//! primality proving*, Math. Comp. 61 (1993) 29-68, on the theorem of S. Goldwasser, J. Kilian,
//! *Primality testing using elliptic curves*, J. ACM 46 (1999) 450-472 (H. Cohen, *A Course in
//! Computational Algebraic Number Theory*, GTM 138, §9.2; R. Crandall, C. Pomerance, *Prime
//! Numbers: A Computational Perspective*, 2nd ed., §7.6).
//!
//! **Theorem.** Let N > 1 with gcd(N, 6) = 1, a, b with gcd(4a^3 + 27b^2, N) = 1, P = (x, y)
//! with y^2 = x^3 + a x + b (mod N), s >= 1 and a prime q > (N^(1/4) + 1)^2. If for every prime
//! p | N the point T_p = [s](P mod p) of E_p: y^2 = x^3 + a x + b over F_p is not O and
//! [q] T_p = O, then N is prime. (T_p has order q, so q <= #E_p(F_p) <= (sqrt p + 1)^2 by
//! Hasse; a composite N has a prime factor p <= sqrt N, which would give
//! q <= (N^(1/4) + 1)^2.)
//!
//! **Certificate.** A chain of steps in PARI/GP's `primecert` layout [N, t, s, a, [x, y]]: the
//! curve y^2 = x^3 + a x + b with b = y^2 - x^3 - a x, m = N + 1 - t, q = m / s, and q is the N
//! of the next step; the last q is below 2^64 and proven by the deterministic word test, as in
//! PARI ("q_ell <= 2^64 is prime"). So certificates are exchangeable with PARI/GP 2.15
//! (`primecert`, `primecertisvalid`).
//!
//! **Order check without inversions.** The verifier works in Jacobian coordinates
//! (X : Y : Z) = (X/Z^2, Y/Z^3), Z = 0 for O, with the doubling `dbl-1998-cmo-2` (H. Cohen,
//! A. Miyaji, T. Ono, *Efficient elliptic curve exponentiation using mixed coordinates*,
//! ASIACRYPT 1998, LNCS 1514, 51-65) and the mixed addition `madd-2004-hmv` (D. Hankerson,
//! A. Menezes, S. Vanstone, *Guide to Elliptic Curve Cryptography*, Springer 2004, §3.2). Over
//! F_p (p >= 5): DBL gives Z3 = 2 Y1 Z1, which is 0 exactly when the input is 2-torsion; MADD
//! gives Z3 = Z1 H, which is 0 exactly when the input is +-P2; and Z = 0 is absorbing. The
//! formulas are polynomials, so computing mod N and reducing mod p is computing over F_p. Hence
//! if the final Z of a double-and-add chain is a unit mod N, every intermediate Z was nonzero
//! mod every p | N, every step computed the true multiple, and the result is [k](P mod p) != O.
//! The check: T = [s]P with Z_T a unit, U = [q - 1]T with Z_U a unit and U = -T. Then
//! [q] T_p = O for every p | N, the hypothesis of the theorem. For a prime N and a point of the
//! right order the check always accepts (the accumulator before an addition is jT with
//! 2 <= j <= q - 2, never +-T, and nothing is 2-torsion).
//!
//! **Prover.** An iterative driver with an explicit stack of frames (one per level; no
//! recursion). A frame scans the discriminant list of [`hilbert`] (h ascending): (D/N) = 1, the
//! genus characters (p*/N) = 1, Cornacchia's 4N = u^2 + |D| v^2 (Cohen Algorithm 1.5.3), the
//! traces t, m = N + 1 - t = k q with k the part over the primes below 2^16, k >= 2,
//! q > (N^(1/4) + 1)^2 and q a probable prime. For a candidate it finds a root j of H_D mod N
//! ([`polymod`]), builds the curve with j-invariant j and its twists, and runs the order check
//! on random points. A frame that runs out of rows is popped and its parent tries its next
//! candidate. Every certificate is verified before `prove_primality` returns `Prime`.

pub(crate) mod hilbert;
pub(crate) mod polymod;

use crate::primality::{
    batches_below_1000, batches_below_2_16, bpsw, is_prime_u64, jacobi_small, probable_prime_big,
    strong_probable_prime, twelve_base_witness, NotPrimeWitness, PSI_12,
};
use crate::Integer;
use num_bigint::BigInt;
use num_integer::Integer as _;
use num_traits::{One, ToPrimitive, Zero};
use rustmath_core::{MathError, Result};
use std::collections::VecDeque;

/// One Atkin-Morain step in PARI's layout [N, t, s, a, [x, y]]: the curve y^2 = x^3 + a x + b
/// over Z/NZ with b = y^2 - x^3 - a x, m = N + 1 - t, q = m / s; q is the N of the next step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EcppStep {
    /// The number this step proves prime (given that q is prime).
    pub n: Integer,
    /// The trace: m = N + 1 - t is the curve order.
    pub t: Integer,
    /// The cofactor: q = m / s.
    pub s: Integer,
    /// The coefficient a of the curve.
    pub a: Integer,
    /// The x-coordinate of the point.
    pub x: Integer,
    /// The y-coordinate of the point.
    pub y: Integer,
}

/// An ECPP primality certificate: MAGMA's `PrimalityCertificate` (ch. 18 §18.9.1). Not related
/// to the trait `rustmath_constants::PrimalityCertificate` (Pratt certificates); import with an
/// alias if both are needed. A certificate of a prover run has no steps iff n < 2^64.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EcppCertificate {
    n: Integer,
    steps: Vec<EcppStep>,
}

/// Outcome of a primality proof (MAGMA `IsPrime` with `Proof := true`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Primality {
    /// A certificate that [`EcppCertificate::verify`] has accepted.
    Prime(EcppCertificate),
    /// A checkable reason why n is not prime.
    NotPrime(NotPrimeWitness),
    /// n passed BPSW, but the prover found no certificate; nothing is decided. The string says
    /// why.
    Unresolved(String),
}

// ------------------------------------------------------------------------------------------
// Jacobian arithmetic and the order check (shared by the prover and the verifier)

/// A point (X : Y : Z) in Jacobian coordinates, Z = 0 for O.
struct Jac {
    x: BigInt,
    y: BigInt,
    z: BigInt,
}

/// dbl-1998-cmo-2: Z3 = 2 Y1 Z1.
fn jdbl(p: &Jac, a: &BigInt, n: &BigInt) -> Jac {
    let xx = (&p.x * &p.x) % n;
    let yy = (&p.y * &p.y) % n;
    let zz = (&p.z * &p.z) % n;
    let s = (&p.x * &yy * 4u32) % n;
    let m = (&xx * 3u32 + a * ((&zz * &zz) % n)) % n;
    let x3 = (&m * &m - &s * 2u32).mod_floor(n);
    let y3 = (&m * (&s - &x3) - ((&yy * &yy) % n) * 8u32).mod_floor(n);
    let z3 = (&p.y * &p.z * 2u32) % n;
    Jac {
        x: x3,
        y: y3,
        z: z3,
    }
}

/// madd-2004-hmv (Z2 = 1): Z3 = Z1 H with H = x2 Z1^2 - X1.
fn jmadd(p: &Jac, x2: &BigInt, y2: &BigInt, n: &BigInt) -> Jac {
    let z1z1 = (&p.z * &p.z) % n;
    let u2 = (x2 * &z1z1) % n;
    let s2 = (y2 * &p.z % n * &z1z1) % n;
    let h = (&u2 - &p.x).mod_floor(n);
    let r = (&s2 - &p.y).mod_floor(n);
    let hh = (&h * &h) % n;
    let hhh = (&h * &hh) % n;
    let v = (&p.x * &hh) % n;
    let x3 = (&r * &r - &hhh - &v * 2u32).mod_floor(n);
    let y3 = (&r * (&v - &x3) - &p.y * &hhh).mod_floor(n);
    let z3 = (&p.z * &h) % n;
    Jac {
        x: x3,
        y: y3,
        z: z3,
    }
}

/// [k](x, y) for k >= 1 by left-to-right double-and-add over the bits of k below the top bit,
/// from (x : y : 1); k = 1 returns the base point. Inputs reduced mod n, 0 <= x, y < n.
fn jmul(k: &BigInt, x: &BigInt, y: &BigInt, a: &BigInt, n: &BigInt) -> Jac {
    let mut r = Jac {
        x: x.clone(),
        y: y.clone(),
        z: BigInt::one(),
    };
    for i in (0..k.bits() - 1).rev() {
        r = jdbl(&r, a, n);
        if k.bit(i) {
            r = jmadd(&r, x, y, n);
        }
    }
    r
}

/// Result of [`order_check`].
#[derive(Debug, PartialEq, Eq)]
enum OrderCheck {
    /// The hypothesis of the theorem holds for (s, q).
    Accept,
    /// [s]P = O mod N: try another point.
    Retry,
    /// The point does not have order q after the cofactor (for a prime N: the wrong twist).
    Reject,
    /// A proper divisor of N appeared.
    Factor(BigInt),
}

/// The order check of a step: T = [s]P, U = [q - 1]T, Z_T and Z_U units mod N and U = -T
/// (module documentation). N > 1 odd, 0 <= a, x, y < N, s >= 1, q >= 2.
fn order_check(
    n: &BigInt,
    a: &BigInt,
    x: &BigInt,
    y: &BigInt,
    s: &BigInt,
    q: &BigInt,
) -> OrderCheck {
    let t = jmul(s, x, y, a, n);
    let g = t.z.gcd(n);
    if g == *n {
        return OrderCheck::Retry;
    }
    if !g.is_one() {
        return OrderCheck::Factor(g);
    }
    let zi = t.z.modinv(n).expect("Z_T is a unit");
    let zi2 = (&zi * &zi) % n;
    let tx = (&t.x * &zi2) % n;
    let ty = (&t.y * &zi2 % n * &zi) % n;
    let u = jmul(&(q - 1u32), &tx, &ty, a, n);
    let g = u.z.gcd(n);
    if g == *n {
        return OrderCheck::Reject;
    }
    if !g.is_one() {
        return OrderCheck::Factor(g);
    }
    let uz2 = (&u.z * &u.z) % n;
    let uz3 = (&uz2 * &u.z) % n;
    if (&u.x - &tx * &uz2).mod_floor(n).is_zero() && (&u.y + &ty * &uz3).mod_floor(n).is_zero() {
        OrderCheck::Accept
    } else {
        OrderCheck::Reject
    }
}

/// The least r with r^4 >= n (n >= 0).
pub(crate) fn iroot4_ceil(n: &BigInt) -> BigInt {
    let mut r = n.sqrt().sqrt();
    if r.pow(4) < *n {
        r += 1u32;
    }
    r
}

/// The bound q must exceed: (iroot4_ceil(n) + 1)^2 >= (n^(1/4) + 1)^2.
pub(crate) fn q_bound(n: &BigInt) -> BigInt {
    let r = iroot4_ceil(n) + 1u32;
    &r * &r
}

// ------------------------------------------------------------------------------------------
// Certificates

impl EcppCertificate {
    /// An unchecked container; [`EcppCertificate::verify`] decides.
    pub fn from_steps(n: Integer, steps: Vec<EcppStep>) -> Self {
        EcppCertificate { n, steps }
    }

    /// The number this certificate is about. `verify` proves `n()` prime, so a caller that
    /// wants a proof for some m must also check `n() == m` (a certificate for q_1 is a valid
    /// certificate, of q_1).
    pub fn n(&self) -> &Integer {
        &self.n
    }

    /// The steps, from n down to the last q.
    pub fn steps(&self) -> &[EcppStep] {
        &self.steps
    }

    /// MAGMA `IsPrimeCertificate`: true iff the certificate proves `n()` prime. Each step must
    /// chain (its N is the previous q), have gcd(N, 6) = 1, t^2 < 4N, s >= 1 dividing
    /// m = N + 1 - t, q = m/s > (r + 1)^2 with r the least integer with r^4 >= N, a
    /// nonsingular curve, and pass the order check; the last q must be below 2^64 and prime
    /// (deterministic word test). It uses no randomness, no discriminant list and no class
    /// polynomial. Never panics; arbitrary data gives `false`. Iterative.
    pub fn verify(&self) -> bool {
        if self.steps.is_empty() {
            return self.n.as_bigint().to_u64().is_some_and(is_prime_u64);
        }
        let mut cur = self.n.as_bigint().clone();
        for st in &self.steps {
            if *st.n.as_bigint() != cur || cur <= BigInt::one() {
                return false;
            }
            if cur.is_even() || (&cur % 3u32).is_zero() {
                return false;
            }
            let t = st.t.as_bigint();
            if t * t >= &cur * 4u32 {
                return false;
            }
            let m: BigInt = &cur + 1u32 - t;
            let s = st.s.as_bigint();
            if *s < BigInt::one() || !(&m % s).is_zero() {
                return false;
            }
            let q = &m / s;
            if q <= q_bound(&cur) {
                return false;
            }
            let a = st.a.as_bigint().mod_floor(&cur);
            let x = st.x.as_bigint().mod_floor(&cur);
            let y = st.y.as_bigint().mod_floor(&cur);
            let b = (&y * &y - &x * &x * &x - &a * &x).mod_floor(&cur);
            let disc = (&a * &a * &a * 4u32 + &b * &b * 27u32).mod_floor(&cur);
            if !disc.gcd(&cur).is_one() {
                return false;
            }
            if order_check(&cur, &a, &x, &y, s, &q) != OrderCheck::Accept {
                return false;
            }
            cur = q;
        }
        cur.to_u64().is_some_and(is_prime_u64)
    }

    /// PARI/GP format: the decimal integer when there are no steps (n < 2^64), else
    /// "[[N, t, s, a, [x, y]], ...]".
    pub fn to_pari(&self) -> String {
        if self.steps.is_empty() {
            return self.n.to_string();
        }
        let v: Vec<String> = self
            .steps
            .iter()
            .map(|s| format!("[{}, {}, {}, {}, [{}, {}]]", s.n, s.t, s.s, s.a, s.x, s.y))
            .collect();
        format!("[{}]", v.join(", "))
    }

    /// Parses the PARI/GP format (the output of gp `primecert`): a decimal integer, or a list
    /// of steps [N, t, s, a, [x, y]]. All ASCII whitespace (spaces, tabs, line breaks, as in
    /// gp's wrapped output) is ignored; n is the first step's N. `Err(ParseError)` if
    /// malformed.
    ///
    /// ```
    /// use rustmath_integers::prime::EcppCertificate;
    /// use rustmath_integers::Integer;
    /// // gp: primecert(nextprime(2^64))
    /// let c = EcppCertificate::from_pari(
    ///     "[[18446744073709551629, -8423788454, 160388, 1, \
    ///       [10598342506117936052, 2225259013356795550]]]",
    /// )
    /// .unwrap();
    /// assert!(c.verify());
    /// assert_eq!(*c.n(), Integer::from_decimal_str("18446744073709551629").unwrap());
    /// ```
    pub fn from_pari(text: &str) -> Result<Self> {
        let s: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        let mut p = Parser { s: &s, i: 0 };
        let cert = if s.first() == Some(&b'[') {
            let mut steps = Vec::new();
            p.expect(b'[')?;
            loop {
                steps.push(p.step()?);
                if p.eat(b',') {
                    continue;
                }
                p.expect(b']')?;
                break;
            }
            let n = steps[0].n.clone();
            EcppCertificate { n, steps }
        } else {
            EcppCertificate {
                n: p.int()?,
                steps: Vec::new(),
            }
        };
        if p.i != s.len() {
            return Err(parse_error("trailing characters"));
        }
        Ok(cert)
    }
}

fn parse_error(why: &str) -> MathError {
    MathError::ParseError(format!("EcppCertificate::from_pari: {why}"))
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn eat(&mut self, c: u8) -> bool {
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(parse_error(&format!(
                "expected '{}' at position {}",
                c as char, self.i
            )))
        }
    }

    fn int(&mut self) -> Result<Integer> {
        let start = self.i;
        if self.s.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        let digits = self.i;
        while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
            self.i += 1;
        }
        if self.i == digits {
            return Err(parse_error(&format!(
                "expected an integer at position {start}"
            )));
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).expect("ASCII digits");
        Integer::from_decimal_str(text).ok_or_else(|| parse_error("bad integer"))
    }

    fn step(&mut self) -> Result<EcppStep> {
        self.expect(b'[')?;
        let n = self.int()?;
        self.expect(b',')?;
        let t = self.int()?;
        self.expect(b',')?;
        let s = self.int()?;
        self.expect(b',')?;
        let a = self.int()?;
        self.expect(b',')?;
        self.expect(b'[')?;
        let x = self.int()?;
        self.expect(b',')?;
        let y = self.int()?;
        self.expect(b']')?;
        self.expect(b']')?;
        Ok(EcppStep { n, t, s, a, x, y })
    }
}

// ------------------------------------------------------------------------------------------
// The prover

/// The prover's limits. `prove_primality` uses [`ProverParams::DEFAULT`]; tests pass others.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProverParams {
    /// Largest class number scanned by the top level and by frames above `h_inner_bits` bits.
    pub(crate) h_top: u32,
    /// Largest class number scanned by the other frames.
    pub(crate) h_inner: u32,
    /// Frames whose N has more bits than this scan to `h_top`.
    pub(crate) h_inner_bits: u64,
    /// After this many failed inner levels (in one call) the result is `Unresolved`.
    pub(crate) max_failed_subtrees: u32,
}

impl ProverParams {
    /// H_TOP = 100 (the whole list), H_INNER = 20, H_INNER_BITS = 2048,
    /// MAX_FAILED_SUBTREES = 64. Below 2048 bits a failing inner level is cheaper to replace
    /// (the parent tries its next candidate) than to extend (the rows with h in 21..100 are 23
    /// times as many and need slower root finding); the top level cannot backtrack. Above 2048
    /// bits an h <= 20 scan finds no candidate for half or more of the primes, so those frames
    /// scan to h = 100. Measured with the design prototype: at most 10 failed subtrees in one
    /// proof at 2048 bits.
    pub(crate) const DEFAULT: ProverParams = ProverParams {
        h_top: hilbert::MAX_H,
        h_inner: 20,
        h_inner_bits: 2048,
        max_failed_subtrees: 64,
    };
}

/// SplitMix64 (G. Steele, D. Lea, C. Flood, *Fast splittable pseudorandom number generators*,
/// OOPSLA 2014), seeded from N so that proofs are reproducible.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(n: &BigInt) -> Self {
        let low = n.iter_u64_digits().next().unwrap_or(0);
        SplitMix64(low ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A residue in [0, n): ceil(bits/64) + 1 random words reduced mod n.
    pub(crate) fn below(&mut self, n: &BigInt) -> BigInt {
        let words = n.bits().div_ceil(64) + 1;
        let mut x = BigInt::zero();
        for _ in 0..words {
            x = (x << 64u32) + BigInt::from(self.next_u64());
        }
        x.mod_floor(n)
    }
}

/// A candidate of a frame: m = N + 1 - t = k q.
struct Cand {
    q: BigInt,
    k: BigInt,
    t: BigInt,
    d: i64,
    h: u32,
}

/// One level of the proof: N and the scan state.
struct Frame {
    n: BigInt,
    bound: BigInt,
    hmax: u32,
    /// The next row: class number and index in its line.
    next_h: u32,
    next_i: usize,
    cands: VecDeque<Cand>,
    /// The least g >= 2 that is a non-square mod N; and the least non-square non-cube.
    nonsquare: Option<Option<BigInt>>,
    nonsquare_noncube: Option<Option<BigInt>>,
}

impl Frame {
    fn new(n: BigInt, hmax: u32) -> Self {
        let bound = q_bound(&n);
        Frame {
            n,
            bound,
            hmax,
            next_h: 1,
            next_i: 0,
            cands: VecDeque::new(),
            nonsquare: None,
            nonsquare_noncube: None,
        }
    }

    /// The next row of the list with h <= hmax, if any.
    fn next_row(&mut self) -> Option<&'static hilbert::Row> {
        while self.next_h <= self.hmax.min(hilbert::MAX_H) {
            let line = hilbert::rows(self.next_h);
            if self.next_i < line.len() {
                self.next_i += 1;
                return Some(&line[self.next_i - 1]);
            }
            self.next_h += 1;
            self.next_i = 0;
        }
        None
    }

    /// The least g in [2, 2^20] that is a non-square mod N (and, if asked, not a cube:
    /// g^((N-1)/3) != 1).
    fn twist_generator(&mut self, noncube: bool) -> Option<BigInt> {
        let slot = if noncube {
            &mut self.nonsquare_noncube
        } else {
            &mut self.nonsquare
        };
        if let Some(g) = slot {
            return g.clone();
        }
        let n = &self.n;
        let e3 = (n - 1u32) / 3u32;
        let mut found = None;
        for g in 2i64..=(1 << 20) {
            if jacobi_small(g, n) != -1 {
                continue;
            }
            if noncube && BigInt::from(g).modpow(&e3, n).is_one() {
                continue;
            }
            found = Some(BigInt::from(g));
            break;
        }
        *slot = Some(found.clone());
        found
    }
}

/// What one call of [`advance`] found.
enum Advance {
    /// An accepted step and its q (the N of the next level).
    Found(EcppStep, BigInt),
    /// N is composite: the witness, or `None` when the prover met a contradiction but found no
    /// strong-test witness among 64 bases.
    Composite(Option<NotPrimeWitness>),
    /// No row with h <= hmax gave a step.
    Exhausted,
}

/// Inside a candidate: go on, or stop the frame with a verdict.
enum CandOutcome {
    Found(EcppStep),
    Composite(Option<NotPrimeWitness>),
    Next,
}

/// N met an impossibility for a prime modulus (a square root failed, for example): search a
/// strong-test witness among the 64 odd primes 3, 5, 7, ... (at least 3/4 of all bases are
/// witnesses for an odd composite, M. O. Rabin, J. Number Theory 12 (1980) 128-138).
fn contradiction(n: &BigInt) -> Option<NotPrimeWitness> {
    crate::sieve::small_primes()[1..65]
        .iter()
        .find(|&&a| !strong_probable_prime(n, &BigInt::from(a)))
        .map(|&a| NotPrimeWitness::MillerRabin(a))
}

/// A probable prime on `BigInt` (exact below psi_12).
fn probable_prime(q: &BigInt) -> bool {
    match q.to_u64() {
        Some(v) => is_prime_u64(v),
        None => probable_prime_big(q).is_ok(),
    }
}

/// Cornacchia's algorithm in the form 4N = u^2 + |D| v^2 (Cohen GTM 138, Algorithm 1.5.3) for
/// a probable prime N with (D/N) = 1: x0 = sqrt(D) mod N with the parity of D, Euclid on
/// (2N, x0) until the remainder is at most floor(sqrt(4N)), b the last remainder; then
/// `Ok(Some((b, sqrt c)))` if c = (4N - b^2)/|D| is an integer square, `Ok(None)` if not (for a
/// prime N: N is not of this form, a normal miss). `Err(())` when the square root of D mod N
/// failed, which proves N composite.
fn cornacchia(d: i64, n: &BigInt) -> std::result::Result<Option<(BigInt, BigInt)>, ()> {
    let db = BigInt::from(d);
    let mut x0 = match crate::integer::sqrt_mod_prime_checked(&db, n) {
        Ok(Some(r)) => r,
        // (D/N) = 1 but no root, or the root computation failed: N is not prime.
        Ok(None) | Err(_) => return Err(()),
    };
    if (&x0 - &db).is_odd() {
        x0 = n - &x0;
    }
    let four_n: BigInt = n * 4u32;
    let l = four_n.sqrt();
    let mut a: BigInt = n * 2u32;
    let mut b = x0;
    while b > l {
        let r = &a % &b;
        a = b;
        b = r;
    }
    let rest = &four_n - &b * &b;
    let (c, r) = rest.div_rem(&BigInt::from(-d));
    if !r.is_zero() {
        return Ok(None);
    }
    let v = c.sqrt();
    if &v * &v != c {
        return Ok(None);
    }
    Ok(Some((b, v)))
}

/// Scan one row: fills `frame.cands` (sorted by q) or reports a composite N.
fn scan_row(frame: &mut Frame, row: &hilbert::Row) -> Option<Advance> {
    let n = &frame.n;
    let d = row.d;
    if jacobi_small(d, n) != 1 {
        return None;
    }
    if row.prime_discs.iter().any(|&p| jacobi_small(p, n) != 1) {
        return None;
    }
    let (u, v) = match cornacchia(d, n) {
        Ok(Some(uv)) => uv,
        Ok(None) => return None, // N is not represented: a normal miss
        Err(()) => return Some(Advance::Composite(contradiction(n))),
    };
    let traces: Vec<BigInt> = match d {
        -4 => vec![u.clone(), -&u, &v * 2u32, -(&v * 2u32)],
        -3 => {
            let p: BigInt = (&u + &v * 3u32) / 2u32;
            let m: BigInt = (&u - &v * 3u32) / 2u32;
            vec![u.clone(), -&u, p.clone(), -p, m.clone(), -m]
        }
        _ => vec![u.clone(), -&u],
    };
    let n1: BigInt = n + 1u32;
    let mut found = Vec::new();
    for t in traces {
        let m = &n1 - &t;
        let (k, q) = batches_below_2_16().strip(&m);
        if k < BigInt::from(2u32) || q <= frame.bound {
            continue;
        }
        if probable_prime(&q) {
            found.push(Cand {
                q,
                k,
                t,
                d,
                h: row.h,
            });
        }
    }
    found.sort_by(|x, y| x.q.cmp(&y.q));
    frame.cands.extend(found);
    None
}

/// Try one candidate of the frame: a root of H_D, the curves, the points, the order check.
fn try_candidate(frame: &mut Frame, cand: &Cand, rng: &mut SplitMix64) -> CandOutcome {
    let n = frame.n.clone();
    let n = &n;
    let curves: Vec<(BigInt, BigInt)> = match cand.d {
        -3 => {
            let Some(g) = frame.twist_generator(true) else {
                return CandOutcome::Next;
            };
            (0..6u32)
                .map(|i| (BigInt::zero(), g.modpow(&BigInt::from(i), n)))
                .collect()
        }
        -4 => {
            let Some(g) = frame.twist_generator(false) else {
                return CandOutcome::Next;
            };
            (0..4u32)
                .map(|i| (g.modpow(&BigInt::from(i), n), BigInt::zero()))
                .collect()
        }
        d => {
            let Some(hp) = hilbert::class_polynomial(d) else {
                return CandOutcome::Next;
            };
            if hp.len() != cand.h as usize + 1 {
                return CandOutcome::Next;
            }
            let j = match polymod::find_root(&hp, n, rng) {
                Ok(Some(j)) => j,
                Ok(None) => return CandOutcome::Next,
                Err(g) => {
                    return CandOutcome::Composite(Some(NotPrimeWitness::Divisor(Integer::new(g))))
                }
            };
            let Some(g) = frame.twist_generator(false) else {
                return CandOutcome::Next;
            };
            // j(y^2 = x^3 + 3k x + 2k) = j for k = j/(1728 - j); scaling by c = 1728 - j is
            // the quadratic twist by c, so the pair covers both twists.
            let c = (BigInt::from(1728u32) - &j).mod_floor(n);
            let a = (&j * &c * 3u32).mod_floor(n);
            let b = (&j * &c * &c * 2u32).mod_floor(n);
            let g2 = (&g * &g) % n;
            let g3 = (&g2 * &g) % n;
            let ta = (&a * &g2) % n;
            let tb = (&b * &g3) % n;
            vec![(a, b), (ta, tb)]
        }
    };
    'curves: for (a, b) in curves {
        let disc = (&a * &a * &a * 4u32 + &b * &b * 27u32).mod_floor(n);
        let e = disc.gcd(n);
        if e == *n {
            continue;
        }
        if !e.is_one() {
            return CandOutcome::Composite(Some(NotPrimeWitness::Divisor(Integer::new(e))));
        }
        for _point in 0..4 {
            let mut point = None;
            for _try in 0..64 {
                let x = rng.below(n);
                let f = (&x * &x * &x + &a * &x + &b).mod_floor(n);
                if f.is_zero() {
                    continue;
                }
                match crate::integer::kronecker_bigint(&f, n) {
                    0 => {
                        return CandOutcome::Composite(Some(NotPrimeWitness::Divisor(
                            Integer::new(f.gcd(n)),
                        )))
                    }
                    1 => {}
                    _ => continue,
                }
                match crate::integer::sqrt_mod_prime_checked(&f, n) {
                    Ok(Some(y)) => {
                        point = Some((x, y));
                        break;
                    }
                    Ok(None) | Err(_) => return CandOutcome::Composite(contradiction(n)),
                }
            }
            let Some((x, y)) = point else {
                continue 'curves;
            };
            match order_check(n, &a, &x, &y, &cand.k, &cand.q) {
                OrderCheck::Accept => {
                    return CandOutcome::Found(EcppStep {
                        n: Integer::new(n.clone()),
                        t: Integer::new(cand.t.clone()),
                        s: Integer::new(cand.k.clone()),
                        a: Integer::new(a),
                        x: Integer::new(x),
                        y: Integer::new(y),
                    })
                }
                OrderCheck::Retry => continue,
                OrderCheck::Reject => continue 'curves,
                OrderCheck::Factor(g) => {
                    return CandOutcome::Composite(Some(NotPrimeWitness::Divisor(Integer::new(g))))
                }
            }
        }
    }
    // No curve accepted. For a prime N and a correct H_D this does not happen; it is not a
    // proof of anything.
    CandOutcome::Next
}

/// Advance a frame until it has a step, a composite verdict, or no rows left.
fn advance(frame: &mut Frame, rng: &mut SplitMix64) -> Advance {
    loop {
        while frame.cands.is_empty() {
            let Some(row) = frame.next_row() else {
                return Advance::Exhausted;
            };
            if let Some(verdict) = scan_row(frame, row) {
                return verdict;
            }
        }
        let cand = frame.cands.pop_front().expect("not empty");
        match try_candidate(frame, &cand, rng) {
            CandOutcome::Found(step) => return Advance::Found(step, cand.q),
            CandOutcome::Composite(w) => return Advance::Composite(w),
            CandOutcome::Next => {}
        }
    }
}

/// Outcome of [`prove_chain`].
#[derive(Debug)]
pub(crate) enum ChainOutcome {
    /// The steps of a chain from N down to a q below 2^64 (not yet verified).
    Steps(Vec<EcppStep>),
    /// N is composite.
    Composite(NotPrimeWitness),
    /// No proof found; the string says why.
    Unresolved(String),
}

/// [`prove_chain`]'s outcome and statistics (the statistics are read by tests).
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct ChainResult {
    pub(crate) outcome: ChainOutcome,
    /// Inner levels that ran out of rows (their parent then tried its next candidate).
    pub(crate) failed_subtrees: u32,
    /// The largest stack depth reached (number of frames).
    pub(crate) max_depth: usize,
}

/// The iterative ECPP driver for a probable prime N >= 2^64 with gcd(N, 6) = 1. Invariant:
/// `steps[i]` is the accepted step of `stack[i]` and its q is `stack[i + 1].n`.
pub(crate) fn prove_chain(n: &BigInt, params: &ProverParams) -> ChainResult {
    let hmax = |q: &BigInt| {
        if q.bits() > params.h_inner_bits {
            params.h_top
        } else {
            params.h_inner
        }
    };
    let mut rng = SplitMix64::new(n);
    let mut stack = vec![Frame::new(n.clone(), params.h_top)];
    let mut steps: Vec<EcppStep> = Vec::new();
    let mut failed = 0u32;
    let mut max_depth = 1usize;
    let result = |outcome, failed, max_depth| ChainResult {
        outcome,
        failed_subtrees: failed,
        max_depth,
    };
    loop {
        let top = stack.last_mut().expect("the stack is never empty here");
        match advance(top, &mut rng) {
            Advance::Found(step, q) => {
                steps.push(step);
                if q.bits() <= 64 {
                    // q passed the probable-prime test, which is exact below 2^64.
                    return result(ChainOutcome::Steps(steps), failed, max_depth);
                }
                let h = hmax(&q);
                stack.push(Frame::new(q, h));
                max_depth = max_depth.max(stack.len());
            }
            Advance::Composite(w) => {
                if stack.len() == 1 {
                    let outcome = match w {
                        Some(w) => ChainOutcome::Composite(w),
                        None => ChainOutcome::Unresolved(
                            "composite (prover contradiction), no witness found".to_string(),
                        ),
                    };
                    return result(outcome, failed, max_depth);
                }
                // q was a BPSW pseudoprime: the parent goes on with its next candidate.
                stack.pop();
                steps.pop();
            }
            Advance::Exhausted => {
                if stack.len() == 1 {
                    return result(
                        ChainOutcome::Unresolved(format!(
                            "no curve found with class number <= {} for N",
                            params.h_top
                        )),
                        failed,
                        max_depth,
                    );
                }
                stack.pop();
                steps.pop();
                failed += 1;
                if failed >= params.max_failed_subtrees {
                    return result(
                        ChainOutcome::Unresolved(format!("{failed} failed subtrees")),
                        failed,
                        max_depth,
                    );
                }
            }
        }
    }
}

/// [`prove_primality`] with explicit prover limits.
pub(crate) fn prove_primality_with(n: &Integer, params: &ProverParams) -> Primality {
    let v = n.as_bigint();
    if *v < BigInt::from(2u32) {
        return Primality::NotPrime(NotPrimeWitness::BelowTwo);
    }
    let small_cert = || Primality::Prime(EcppCertificate::from_steps(n.clone(), Vec::new()));
    if let Some(w) = v.to_u64() {
        if w < 1000 {
            return if is_prime_u64(w) {
                small_cert()
            } else {
                let p = batches_below_1000()
                    .least_divisor(v)
                    .expect("a composite below 1000 has a prime factor below 1000");
                Primality::NotPrime(NotPrimeWitness::Divisor(Integer::from(p)))
            };
        }
    }
    if let Some(p) = batches_below_1000().least_divisor(v) {
        return Primality::NotPrime(NotPrimeWitness::Divisor(Integer::from(p)));
    }
    if let Some(w) = v.to_u64() {
        if is_prime_u64(w) {
            return small_cert();
        }
        let a = twelve_base_witness(v).expect("Sorenson-Webster: a base in 2..37 fails");
        return Primality::NotPrime(NotPrimeWitness::MillerRabin(a));
    }
    if *v < BigInt::from(PSI_12) {
        if let Some(a) = twelve_base_witness(v) {
            return Primality::NotPrime(NotPrimeWitness::MillerRabin(a));
        }
    } else if let Err(w) = bpsw(v) {
        return Primality::NotPrime(w);
    }
    match prove_chain(v, params).outcome {
        ChainOutcome::Steps(steps) => {
            let cert = EcppCertificate::from_steps(n.clone(), steps);
            if cert.verify() {
                Primality::Prime(cert)
            } else {
                debug_assert!(false, "the ECPP certificate of {n} failed verification");
                Primality::Unresolved(
                    "internal error: the ECPP certificate failed verification".to_string(),
                )
            }
        }
        ChainOutcome::Composite(w) => Primality::NotPrime(w),
        ChainOutcome::Unresolved(why) => Primality::Unresolved(why),
    }
}

/// MAGMA `IsPrime(n)` with `Proof := true` and `PrimalityCertificate(n)` (ch. 18 §18.9.1):
/// trial division below 1000, the deterministic word test below 2^64 (a certificate with no
/// steps), the twelve strong tests below psi_12 = 318665857834031151167461 or BPSW above it,
/// then the ECPP prover ([AM93]) for n >= 2^64. Every `Prime` carries a certificate that
/// [`EcppCertificate::verify`] accepted (the certificate is always verified before it is
/// returned, 2 to 7 % of the proof time). Every `NotPrime` carries a witness that
/// [`NotPrimeWitness::confirms`].
///
/// For 2^64 <= n < psi_12 the result can be `Unresolved` although the twelve strong tests prove
/// n prime (no ECPP certificate was found); [`crate::prime::try_is_prime`] and
/// [`crate::prime::is_prime`] decide those n by the strong tests and never use ECPP there.
///
/// `Unresolved` comes only from the prover: the top level found no candidate in the
/// discriminant list (h <= 100), or every top-level candidate's subtree failed, or 64 inner
/// levels failed in all. Measured with the design prototype (random primes; release build, a
/// debug build is 8 to 15 times slower): 128 bits 0.024 s, 256 bits 0.07 s, 512 bits 0.41 s,
/// 1024 bits 3.3 s, 1536 bits 21 s, 2048 bits 76 s, 3072 bits about 16 minutes. No prime up to
/// 2048 bits was unresolved in the experiments (61 full proofs, 180 coverage runs); at 3072 and
/// 4096 bits 2 of 32 primes had no top-level candidate with h <= 100 and are `Unresolved`.
/// No size limit is imposed; a caller that needs a fast answer uses
/// [`crate::prime::is_probable_prime`] or bounds the size itself.
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::{prove_primality, Primality};
/// let m127 = Integer::from(2).pow(127) - Integer::one();
/// match prove_primality(&m127) {
///     Primality::Prime(cert) => {
///         assert!(cert.verify());
///         assert_eq!(*cert.n(), m127);
///     }
///     other => panic!("{other:?}"),
/// }
/// ```
pub fn prove_primality(n: &Integer) -> Primality {
    prove_primality_with(n, &ProverParams::DEFAULT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Signed;

    fn big(s: &str) -> BigInt {
        s.parse().unwrap()
    }

    /// Affine reference arithmetic on y^2 = x^3 + a x + b over F_p (None = O).
    fn affine_add(
        p1: &Option<(BigInt, BigInt)>,
        p2: &Option<(BigInt, BigInt)>,
        a: &BigInt,
        p: &BigInt,
    ) -> Option<(BigInt, BigInt)> {
        let (Some((x1, y1)), Some((x2, y2))) = (p1, p2) else {
            return p1.clone().or(p2.clone());
        };
        let lambda = if x1 == x2 {
            if (y1 + y2).mod_floor(p).is_zero() {
                return None;
            }
            (x1 * x1 * 3u32 + a) * (y1 * 2u32).modinv(p).unwrap()
        } else {
            (y2 - y1) * (x2 - x1).mod_floor(p).modinv(p).unwrap()
        };
        let lambda = lambda.mod_floor(p);
        let x3 = (&lambda * &lambda - x1 - x2).mod_floor(p);
        let y3 = (&lambda * (x1 - &x3) - y1).mod_floor(p);
        Some((x3, y3))
    }

    fn affine_of(j: &Jac, p: &BigInt) -> Option<(BigInt, BigInt)> {
        if j.z.mod_floor(p).is_zero() {
            return None;
        }
        let zi = j.z.modinv(p).unwrap();
        let zi2 = (&zi * &zi) % p;
        Some(((&j.x * &zi2) % p, (&j.y * &zi2 % p * &zi) % p))
    }

    /// The order of P (affine reference).
    fn affine_order(base: &Option<(BigInt, BigInt)>, a: &BigInt, p: &BigInt) -> u32 {
        let mut acc = base.clone();
        let mut k = 1;
        while acc.is_some() {
            acc = affine_add(&acc, base, a, p);
            k += 1;
        }
        k
    }

    /// Jacobian `jmul` against affine arithmetic on y^2 = x^3 + 2x + 3 over F_97 (every point)
    /// and on a curve over F_(2^61 - 1). Soundness (facts 1-3 of the module documentation): a
    /// nonzero final Z always gives the true multiple. Exactness: for k < ord(P)/2 no
    /// exceptional case occurs, so Z is nonzero. gp: E = ellinit([2,3], 97); ellcard(E) = 100;
    /// ellorder and ellmul agree with the affine reference.
    #[test]
    fn ecpp_order_check_matches_affine() {
        let p = BigInt::from(97u32);
        let a = BigInt::from(2u32);
        let mut points = 0;
        for x in 0u32..97 {
            for y in 0u32..97 {
                let (x, y) = (BigInt::from(x), BigInt::from(y));
                if !(&y * &y - &x * &x * &x - &a * &x - 3u32)
                    .mod_floor(&p)
                    .is_zero()
                {
                    continue;
                }
                points += 1;
                let base = Some((x.clone(), y.clone()));
                let ord = affine_order(&base, &a, &p);
                let mut acc = base.clone();
                for k in 1u32..=200 {
                    let j = jmul(&BigInt::from(k), &x, &y, &a, &p);
                    let got = affine_of(&j, &p);
                    if got.is_some() {
                        assert_eq!(got, acc, "[{k}]P, P = ({x}, {y})");
                    }
                    if 2 * k < ord {
                        assert!(got.is_some(), "[{k}]P, P = ({x}, {y}) of order {ord}");
                    }
                    acc = affine_add(&acc, &base, &a, &p);
                }
            }
        }
        assert_eq!(points + 1, 100, "ellcard(ellinit([2,3],97)) = 100");
        // A curve over a 61-bit prime: affine and Jacobian agree for k = 1..200.
        let p = BigInt::from((1u64 << 61) - 1);
        let a = BigInt::from(5u32);
        let mut x = BigInt::from(11u32);
        let y = loop {
            let f = (&x * &x * &x + &a * &x + 7u32).mod_floor(&p);
            if let Some(y) = crate::integer::sqrt_mod_prime_checked(&f, &p).unwrap() {
                break y;
            }
            x += 1u32;
        };
        let base = Some((x.clone(), y.clone()));
        let mut acc = base.clone();
        for k in 1u32..=200 {
            let j = jmul(&BigInt::from(k), &x, &y, &a, &p);
            assert_eq!(affine_of(&j, &p), acc, "[{k}]P mod 2^61-1");
            acc = affine_add(&acc, &base, &a, &p);
        }
    }

    /// gp: E = ellinit([2,3], 97); ellcard(E) = 100; ellorder(E, [3,6]) = 5.
    #[test]
    fn order_check_accepts_and_rejects() {
        let n = BigInt::from(97u32);
        let a = BigInt::from(2u32);
        let (x, y) = (BigInt::from(3u32), BigInt::from(6u32));
        let one = BigInt::one();
        // [1]P has order 5: [4]P = -P.
        assert_eq!(
            order_check(&n, &a, &x, &y, &one, &BigInt::from(5u32)),
            OrderCheck::Accept
        );
        // q = 7 is not the order: [6]P = P, not -P.
        assert_eq!(
            order_check(&n, &a, &x, &y, &one, &BigInt::from(7u32)),
            OrderCheck::Reject
        );
        // [5]P = O: another point is needed.
        assert_eq!(
            order_check(&n, &a, &x, &y, &BigInt::from(5u32), &BigInt::from(5u32)),
            OrderCheck::Retry
        );
    }

    #[test]
    fn ecpp_q_bound_edges() {
        // 10^20 = (10^5)^4, exact integer arithmetic.
        let e20 = big("100000000000000000000");
        assert_eq!(iroot4_ceil(&e20), BigInt::from(100_000u32));
        assert_eq!(iroot4_ceil(&(&e20 + 1u32)), BigInt::from(100_001u32));
        assert_eq!(iroot4_ceil(&(&e20 - 1u32)), BigInt::from(100_000u32));
        // The bound test "q > (r + 1)^2" is false at q = (r + 1)^2, true one above it.
        let bound = q_bound(&e20);
        assert_eq!(bound, BigInt::from(100_001u64 * 100_001));
        let passes = |q: &BigInt| *q > q_bound(&e20);
        assert!(!passes(&bound));
        assert!(passes(&(&bound + 1u32)));
    }

    /// gp: qfbsolve(Qfb(1,1,6), 1000000000933) = [-1021181, 74772], so with u = 2x + y,
    /// 4N = u^2 + 23 v^2 for |u| = 1967590, |v| = 74772; qfbsolve(Qfb(1,1,1), 2^127-1) =
    /// [10527652944872033958, 4064494709036811419], so 4N = u^2 + 3 v^2 has a solution.
    #[test]
    fn ecpp_cornacchia_traces() {
        let n = BigInt::from(1_000_000_000_933u64);
        let (u, v) = cornacchia(-23, &n).unwrap().unwrap();
        assert_eq!(u.abs(), BigInt::from(1_967_590u32));
        assert_eq!(v.abs(), BigInt::from(74_772u32));
        let n = (BigInt::one() << 127u32) - 1u32;
        let (u, v) = cornacchia(-3, &n).unwrap().unwrap();
        assert_eq!(&u * &u + &v * &v * 3u32, &n * 4u32);
        // The six traces for D = -3 and the four for D = -4 all satisfy t^2 < 4N.
        let mut frame = Frame::new(n.clone(), 1);
        frame.bound = BigInt::zero();
        let row = hilbert::Row {
            d: -3,
            h: 1,
            prime_discs: vec![-3],
        };
        assert!(scan_row(&mut frame, &row).is_none());
        for c in &frame.cands {
            assert!(&c.t * &c.t < &n * 4u32);
            assert_eq!((&n + 1u32 - &c.t) % &c.q, BigInt::zero());
        }
        // 2^127 - 1 = 1 (mod 3) and is not represented with D = -23 iff (-23/N) != 1.
        assert_eq!(jacobi_small(-3, &n), 1);
    }

    // gp: primecert(2^127-1) (PARI/GP 2.15.4)
    const C127: &str = "[[170141183460469231731687303715884105727, 6463158235835222539, 37711126033, 0, [123106756171564943260470175830543720431, 75271964349985580273014815876123588329]], [4511697245836234712605197733, -103300761668182, 206599692, 0, [2028653280926922362829473336, 351806756186690237552618680]], [21837870144725762773, -277408667, 408201703, 0, [562706585515371056, 10173121340883365254]]]";

    #[test]
    fn verify_accepts_c127_and_rejects_x_plus_one() {
        let c = EcppCertificate::from_pari(C127).unwrap();
        assert!(c.verify());
        let mut steps = c.steps().to_vec();
        steps[0].x = &steps[0].x + &Integer::one();
        assert!(!EcppCertificate::from_steps(c.n().clone(), steps).verify());
    }

    /// `prove_chain` with H_INNER forced to 1: inner levels that have no h = 1 candidate fail,
    /// the driver pops them and the parent tries its next candidate. The prover is
    /// deterministic (seeded from N), so the outcome is pinned (review 2 n6): for
    /// nextprime(2^200) = 2^200 + 235 no inner level fails, for the next prime 2^200 + 697 one
    /// does, and the proof still ends with a verified certificate. (gp: nextprime(2^200) - 2^200
    /// = 235, nextprime(2^200 + 236) - 2^200 = 697.)
    #[test]
    fn ecpp_driver_backtracks() {
        let params = ProverParams {
            h_top: hilbert::MAX_H,
            h_inner: 1,
            h_inner_bits: 2048,
            max_failed_subtrees: 64,
        };
        for (d, want_failed) in [(235u32, 0u32), (697, 1)] {
            let n = (BigInt::one() << 200u32) + d;
            let r = prove_chain(&n, &params);
            let ChainOutcome::Steps(steps) = r.outcome else {
                panic!("2^200 + {d}: outcome {:?}", r.outcome);
            };
            assert_eq!(r.failed_subtrees, want_failed, "2^200 + {d}");
            assert!(r.max_depth >= steps.len(), "at least one frame per step");
            let cert = EcppCertificate::from_steps(Integer::new(n), steps);
            assert!(cert.verify(), "2^200 + {d}");
        }
        // With no rows for the inner levels at all, every inner level fails: the driver
        // gives up after max_failed_subtrees and never panics.
        let n = (BigInt::one() << 200u32) + 235u32;
        let starved = ProverParams {
            h_inner: 0,
            max_failed_subtrees: 3,
            ..params
        };
        let r = prove_chain(&n, &starved);
        assert!(
            matches!(r.outcome, ChainOutcome::Unresolved(_)),
            "{:?}",
            r.outcome
        );
        assert!(r.failed_subtrees <= 3);
    }

    /// The prover's contradiction path (section 2.3.6 of the design) on a composite past the
    /// probable-prime gate: psi_12 = 399165290221 * 798330580441 passes the twelve strong tests
    /// 2..37, a square root modulo it fails, and the witness search finds the first base among
    /// 3, 5, 7, ... to which it is not a strong probable prime. gp, with sp(n, a) the strong
    /// test: vector(13, i, sp(psi_12, prime(i))) = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], so 41
    /// is the first witness.
    #[test]
    fn prove_chain_on_strong_pseudoprime() {
        let psi = BigInt::from(PSI_12);
        let r = prove_chain(&psi, &ProverParams::DEFAULT);
        match r.outcome {
            ChainOutcome::Composite(w) => {
                assert_eq!(w, NotPrimeWitness::MillerRabin(41));
                assert!(w.confirms(&Integer::new(psi)));
            }
            other => panic!("psi_12: {other:?}"),
        }
    }
}
