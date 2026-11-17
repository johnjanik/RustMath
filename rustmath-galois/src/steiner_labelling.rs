//! The certified Steiner labelling of the roots of an M23 polynomial, the Golay invariants
//! `T5`, `J5`, `J7`, the degree 253 heptad and duad resolvents, and Frobenius signs.
//!
//! Sources:
//! - The paper `golay_m23.tex` of the owner's M23 repository (`johnjanik/M23` at `d05475b`):
//!   Section 2 (the two cyclic codes, Lemma `lem:twocodes`, Definition `def:sign`), Section 5
//!   (Proposition 5.1 `prop:labelling` with its proof, Theorem 5.2 `thm:invariants`, the
//!   Remark on Frobenius signs), Section 6 (the duad resolvent), Section 8 (`J5`, `J7`).
//!   Its reference implementation `m23golay/arith.py` (`root_bound`, `frobenius_lift`,
//!   `steiner_labelling`, `golay_invariants`, `heptad_resolvent`, `duad_resolvent`,
//!   `frobenius_sign`) is ported here as an algorithm.
//! - H. Fujiwara, *Ueber die obere Schranke des absoluten Betrages der Wurzeln einer
//!   algebraischen Gleichung*, Tohoku Math. J. 10 (1916) 167-171 (the root bound).
//! - E. Witt, *Die 5-fach transitiven Gruppen von Mathieu* and *Ueber Steinersche Systeme*, Abh.
//!   Math. Sem. Hamburg 12 (1938) (S(4,7,23) is unique, with automorphism group M23).
//! - R. P. Stauduhar, Math. Comp. 27 (1973) 981-996, and K. Geissler, J. Klueners, J. Symbolic
//!   Comput. 30 (2000) 653-674 (Frobenius-ordered `p`-adic roots, bounded integrality).
//! - MAGMA ch. 38 (`GaloisRoot`, `IsInt`, `Bound`, `Stauduhar`), ch. 47 (`GaloisImage`,
//!   `Trace`), ch. 48 (Galois rings); digests in `rustmath-lava/docs/`.
//! - Design: `docs/workstreams/m23_golay/` (Steiner labelling design SL and its review).
//!
//! # The decision (paper, Proposition 5.1)
//!
//! Let `f` be monic of degree 23 in `Z[x]` and `p` a prime with `f mod p` irreducible. In
//! `R = GR(p^N, f)` the roots are `alpha_i = phi^i(x)`, `i` in `Z/23`
//! ([`FrobeniusRoots`]). For a heptad `H` put `s_H = sum_{i in H} alpha_i`, and for each of the
//! two cyclic Golay systems `S+`, `S-` ([`GolaySign`]) put `T5^e = sum_{H in S^e} s_H^5`.
//!
//! An embedding of the splitting field into `Frac(R)` makes `Gal(f)` a subgroup of
//! `Sym(Z/23)` that contains the shift (the image of `Frob_p`). If `Gal(f)` is contained in a
//! conjugate of M23, that conjugate is `Aut(S)` for a shift-invariant S(4,7,23), so `S` is `S+`
//! or `S-` (Lemma `lem:twocodes`), and it preserves exactly one of them. For that sign `e`,
//! `T5^e` is a `Gal(f)`-invariant algebraic integer, hence a rational integer, with
//! `|T5^e| <= 253 (7M)^5 < B5` for the root bound `M` ([`RootBound`]). With `p^N > 2 B5` its
//! symmetric residue is `T5^e` itself, so the true sign always passes the test `|r| <= B5`.
//!
//! - Exactly one sign passes: **Decided**. That sign is `e`, `epsilon(Frob_p) = e`, and
//!   `T5(f)` is its residue.
//! - Both pass: **Unresolved**. A larger `N` may separate them (each try is a proof on its own;
//!   [`LabellingOptions::retries`]).
//! - Neither passes: [`LabellingStatus::NotInM23`], a proof without any hypothesis that `Gal(f)`
//!   lies in no conjugate of M23 (the argument above applies to every conjugate containing the
//!   shift).
//!
//! # The trusted input
//!
//! `Decided` rests on the hypothesis [`GaloisHypothesis::M23`], `Gal(f) = M23`, which is a
//! trusted input: it is recorded in every result and never checked (as the genus is in WP-F).
//! For HJLPPZ's f1 and f2 it is their theorem; for Le Lay's CC0 specialization it is his
//! certificate. The proof uses only the weaker "Gal(f) is contained in a conjugate of M23".
//! Under that weaker hypothesis a `Decided` status even proves `Gal(f) = M23`: a subgroup of M23
//! containing a 23-cycle is `C23`, `23:11` or M23 (ATLAS maximal subgroups), and the first two
//! preserve both cyclic systems, so they always give `Unresolved` (review 1 of design SL, 1.3).
//!
//! **A false hypothesis can give a false `Decided`.** Example: `x^23 - x - 1` has Galois group
//! S23 (Osada 1987). At `p = 157` with the minimal rigorous precision `N = 4` the rule returns
//! `Decided(-)` (residues `182056093`, `128168772`, `B5 = 136069473`; PARI/GP and `arith.py`
//! agree). With the default margin (`extra_digits = 30`, `N = 18`) neither candidate passes, which
//! refutes the hypothesis. `NotInM23` is only an occasional detection of a false input.
//!
//! # Precision and the other invariants
//!
//! [`precision_for`] gives the least `N` with `p^N > 2 B 10^e`. The labelling uses
//! `e = extra_digits` (default 30, the reference's margin). It changes nothing in the proof but
//! makes the wrong residue fall outside the bound with high probability. `J5`, `J7` and the
//! resolvents use `e = 0`, which is already rigorous: each value passes [`bounded_integer`]
//! against its proven bound, and a value outside it is `Err(BoundExceeded)`, never returned.
//! Sums over a shift-invariant family are sums of traces over the shift-orbit representatives;
//! products are products of orbit polynomials (design SL, 2.6). No stability loop is part of the
//! API; a recomputation at a higher `N` is only a cross-check.

use std::fmt;
use std::sync::OnceLock;

pub use rustmath_coding::cyclic_golay::GolaySign;
use rustmath_coding::cyclic_golay::{heptad_orbit_representatives, heptads, rotate, support};
use rustmath_combinatorics::is_steiner_system;
use rustmath_core::MathError;
use rustmath_finitefields::{
    orbit_polynomial, FrobeniusRoots, FrobeniusRootsError, GaloisRingElement,
};
use rustmath_integers::Integer;
use rustmath_polynomials::{fp_factor, zx};

use crate::integrality::{
    bounded_integer, integrality_dichotomy, symmetric_residue, Dichotomy, DichotomyGap,
    IntegralityError,
};

/// The degree 23.
pub const DEGREE: usize = 23;

/// The trusted input behind every `Decided` labelling and every invariant computed from one. It
/// is recorded, never checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GaloisHypothesis {
    /// `Gal(f) = M23` in its natural action on the 23 roots. The proof uses only "`Gal(f)` is
    /// contained in a conjugate of M23"; under that weaker form a `Decided` labelling proves
    /// `Gal(f) = M23` (module docs).
    M23,
}

/// What a certified value rests on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Certification {
    /// Symmetric in the roots (the duad resolvent): proven for every monic `f` in `Z[x]` with no
    /// hypothesis and no labelling.
    Symmetric,
    /// An invariant of the Steiner system on the roots: relative to the hypothesis and to the
    /// `Decided` labelling (sign at the labelling prime and precision).
    Labelling {
        /// The trusted input.
        hypothesis: GaloisHypothesis,
        /// The decided sign at `prime`.
        sign: GolaySign,
        /// The labelling prime.
        prime: Integer,
        /// The precision of the labelling.
        precision: u32,
    },
}

/// An integer computed from a residue mod `prime^precision` with `prime^precision > 2 bound`, and
/// `|value| <= bound` for the proven bound. Built only by this module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertifiedInteger {
    value: Integer,
    bound: Integer,
    prime: Integer,
    precision: u32,
    certification: Certification,
}

impl CertifiedInteger {
    /// The integer.
    pub fn value(&self) -> &Integer {
        &self.value
    }
    /// The proven bound `B` (`|value| <= B`).
    pub fn bound(&self) -> &Integer {
        &self.bound
    }
    /// The prime of the computation.
    pub fn prime(&self) -> &Integer {
        &self.prime
    }
    /// The precision `N` of the computation (`prime^N > 2 B`).
    pub fn precision(&self) -> u32 {
        self.precision
    }
    /// What the value rests on.
    pub fn certification(&self) -> &Certification {
        &self.certification
    }
}

/// `J5` and `J7` (paper, Section 8): `J7 = sum_H prod_{i in H} alpha_i` and
/// `J5 = sum_H e_5(alpha_H)` (each five-subset of a heptad lies in no other heptad).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GolayInvariants {
    /// `J5`.
    pub j5: CertifiedInteger,
    /// `J7`.
    pub j7: CertifiedInteger,
}

/// The coefficients of a degree 253 resolvent, ascending (length 254, leading 1), each the
/// symmetric residue mod `prime^precision > 2 bound` and within the proven bound. Built only by
/// this module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertifiedResolvent {
    coefficients: Vec<Integer>,
    bound: Integer,
    prime: Integer,
    precision: u32,
    certification: Certification,
}

impl CertifiedResolvent {
    /// The coefficients, ascending.
    pub fn coefficients(&self) -> &[Integer] {
        &self.coefficients
    }
    /// The proven coefficient bound.
    pub fn bound(&self) -> &Integer {
        &self.bound
    }
    /// The prime of the computation.
    pub fn prime(&self) -> &Integer {
        &self.prime
    }
    /// The precision `N` (`prime^N > 2 bound`).
    pub fn precision(&self) -> u32 {
        self.precision
    }
    /// What the coefficients rest on.
    pub fn certification(&self) -> &Certification {
        &self.certification
    }
}

/// Why a labelling routine refused or failed.
#[derive(Clone, Debug, PartialEq)]
pub enum SteinerError {
    /// `f` is not monic (use [`monic_associate`]).
    NotMonic,
    /// `deg f != 23` (or `deg f < 1` for [`RootBound::fujiwara`]).
    WrongDegree(usize),
    /// [`monic_associate`] of the zero polynomial.
    ZeroLeadingCoefficient,
    /// No inert prime in the search range `[3, below)` (no claim about larger primes).
    NoInertPrime {
        /// The end of the range searched.
        below: u64,
    },
    /// The Frobenius-ordered roots refused the prime or the precision.
    Frobenius(FrobeniusRootsError),
    /// A precision with `p^N <= 2B` (the only [`IntegralityError`] this module can cause).
    Integrality(IntegralityError),
    /// The invariant needs a `Decided` labelling.
    NotDecided(LabellingStatus),
    /// A computed value lies outside its proven bound. Impossible under the hypothesis; it
    /// refutes the hypothesis or reveals a bug. Never returned as a value.
    BoundExceeded {
        /// The quantity (`"J5"`, `"J7"`, `"heptad resolvent"`, `"duad resolvent"`).
        quantity: &'static str,
        /// The coefficient index for a resolvent.
        index: Option<usize>,
    },
    /// An orbit-polynomial coefficient outside `Z/p^N` (an arithmetic bug).
    NotInBaseRing {
        /// The quantity.
        quantity: &'static str,
    },
    /// The run-time check of the two Golay systems failed (a bug).
    GolaySystems(String),
}

impl fmt::Display for SteinerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMonic => write!(f, "f is not monic"),
            Self::WrongDegree(n) => write!(f, "deg f = {n}, expected {DEGREE}"),
            Self::ZeroLeadingCoefficient => write!(f, "the polynomial is zero"),
            Self::NoInertPrime { below } => write!(f, "no inert prime in [3, {below})"),
            Self::Frobenius(e) => write!(f, "Frobenius roots: {e}"),
            Self::Integrality(e) => write!(f, "{e}"),
            Self::NotDecided(s) => write!(f, "the labelling is not decided ({s:?})"),
            Self::BoundExceeded { quantity, index } => match index {
                Some(i) => write!(f, "{quantity}: coefficient {i} exceeds its proven bound"),
                None => write!(f, "{quantity} exceeds its proven bound"),
            },
            Self::NotInBaseRing { quantity } => {
                write!(f, "{quantity}: an orbit coefficient is not in Z/p^N")
            }
            Self::GolaySystems(why) => write!(f, "Golay systems check failed: {why}"),
        }
    }
}

impl std::error::Error for SteinerError {}

impl From<FrobeniusRootsError> for SteinerError {
    fn from(e: FrobeniusRootsError) -> Self {
        SteinerError::Frobenius(e)
    }
}

impl From<IntegralityError> for SteinerError {
    fn from(e: IntegralityError) -> Self {
        SteinerError::Integrality(e)
    }
}

// ---------------------------------------------------------------------------
// Root bound, bounds and precision
// ---------------------------------------------------------------------------

/// Fujiwara's root bound `M = best / 128` of a monic `f = x^n + a_{n-1} x^{n-1} + ... + a_0`:
/// `best = max_k r_k`, `r_k` the least integer with `r_k^k >= 256^k |a_{n-k}|`, so
/// `best / 256 >= mu = max_k |a_{n-k}|^{1/k}` and every complex root has `|z| <= 2 mu <= M`.
///
/// Proof (Fujiwara 1916): if `|z| > 2 mu` then `|a_{n-k} z^{n-k}| <= mu^k |z|^{n-k} <
/// 2^{-k} |z|^n`, so `|f(z)| > |z|^n (1 - sum_k 2^{-k}) > 0`. This is `arith.py` `root_bound`
/// (8 extra binary digits); the exact integer roots make it rigorous.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootBound {
    best: Integer,
}

/// The least `r >= 0` with `r^k >= a` (`a >= 0`).
fn ceil_root(a: &Integer, k: u32) -> Integer {
    if a <= &Integer::one() {
        return a.clone();
    }
    let r = a.nth_root(k).expect("k >= 1 and a >= 0");
    if &r.pow(k) >= a {
        r
    } else {
        r + Integer::one()
    }
}

impl RootBound {
    /// The bound of a monic `f` of degree `>= 1`.
    ///
    /// # Errors
    ///
    /// `NotMonic`; `WrongDegree(0)` for a constant.
    pub fn fujiwara(f: &[Integer]) -> Result<RootBound, SteinerError> {
        if f.last().map(|c| !c.is_one()).unwrap_or(true) {
            return Err(SteinerError::NotMonic);
        }
        let n = f.len() - 1;
        if n == 0 {
            return Err(SteinerError::WrongDegree(0));
        }
        let mut best = Integer::zero();
        for k in 1..=n {
            let a = f[n - k].abs();
            if a.is_zero() {
                continue;
            }
            let r = ceil_root(&(a * Integer::from(256u64).pow(k as u32)), k as u32);
            if r > best {
                best = r;
            }
        }
        Ok(RootBound { best })
    }

    /// `best` (the bound is `M = best / 128`).
    pub fn best(&self) -> &Integer {
        &self.best
    }

    /// `floor(count (scale M)^e) + 1`, exact.
    pub fn power_bound(&self, count: u64, scale: u64, e: u32) -> Integer {
        let top = Integer::from(count) * (Integer::from(scale) * self.best.clone()).pow(e);
        top / Integer::from(128u64).pow(e) + Integer::one()
    }

    /// `floor((1 + scale M)^e) + 1`, exact.
    pub fn binomial_bound(&self, scale: u64, e: u32) -> Integer {
        let top = (Integer::from(128u64) + Integer::from(scale) * self.best.clone()).pow(e);
        top / Integer::from(128u64).pow(e) + Integer::one()
    }

    /// `B5 = power_bound(253, 7, 5)`: `|s_H| <= 7M` for the 253 heptads.
    pub fn t5_bound(&self) -> Integer {
        self.power_bound(253, 7, 5)
    }

    /// `power_bound(5313, 1, 5)`: 5313 products of five roots.
    pub fn j5_bound(&self) -> Integer {
        self.power_bound(5313, 1, 5)
    }

    /// `power_bound(253, 1, 7)`: 253 products of seven roots.
    pub fn j7_bound(&self) -> Integer {
        self.power_bound(253, 1, 7)
    }

    /// `binomial_bound(7, 253)`: `|e_k(s)| <= C(253, k) (7M)^k <= (1 + 7M)^253`.
    pub fn heptad_resolvent_bound(&self) -> Integer {
        self.binomial_bound(7, 253)
    }

    /// `binomial_bound(2, 253)`: `|alpha_i + alpha_j| <= 2M`.
    pub fn duad_resolvent_bound(&self) -> Integer {
        self.binomial_bound(2, 253)
    }
}

/// The least `N >= 1` with `p^N > 2 bound 10^extra_digits`. Panics for `p < 2`.
pub fn precision_for(p: &Integer, bound: &Integer, extra_digits: u32) -> u32 {
    assert!(p > &Integer::one(), "precision_for: p = {p} < 2");
    let target = Integer::from(2u64) * bound.clone() * Integer::from(10u64).pow(extra_digits);
    let mut n = 1u32;
    let mut pn = p.clone();
    while pn <= target {
        n += 1;
        pn = pn * p.clone();
    }
    n
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// `g(x) = a^(n-1) f(x / a)` with `a` the leading coefficient of `f` (trailing zeros ignored):
/// monic in `Z[x]`, coefficient of `x^i` equal to `f_i a^(n-1-i)`, roots `a` times those of `f`.
/// Returns `(g, a)`.
///
/// # Errors
///
/// `ZeroLeadingCoefficient` for the zero polynomial.
pub fn monic_associate(f: &[Integer]) -> Result<(Vec<Integer>, Integer), SteinerError> {
    let n = match f.iter().rposition(|c| !c.is_zero()) {
        None => return Err(SteinerError::ZeroLeadingCoefficient),
        Some(n) => n,
    };
    let a = f[n].clone();
    let mut g = Vec::with_capacity(n + 1);
    for (i, c) in f.iter().enumerate().take(n) {
        g.push(c.clone() * a.pow((n - 1 - i) as u32));
    }
    g.push(Integer::one());
    Ok((g, a))
}

/// The first prime `p` in `[from, below)` (and `p <= fp_factor::MAX_P`) with `f mod p`
/// irreducible of degree `deg f`, by exact factorization mod `p`. A bounded search: `None` says
/// nothing about primes at or above `below`.
pub fn first_inert_prime(f: &[Integer], from: u64, below: u64) -> Option<u64> {
    let n = f.len().checked_sub(1)?;
    if n == 0 || f[n].is_zero() {
        return None;
    }
    let below = below.min(fp_factor::MAX_P as u64 + 1);
    (from.max(2)..below).find(|&p| {
        if !rustmath_integers::prime::is_prime_u64(p) {
            return false;
        }
        let pi = Integer::from(p);
        if f[n].modulo(&pi).is_zero() {
            return false;
        }
        let fp: Vec<i64> = f
            .iter()
            .map(|c| c.modulo(&pi).to_i64().expect("residue below MAX_P"))
            .collect();
        let fac = fp_factor::factor_with_multiplicity(&fp, p as i64);
        fac.len() == 1 && fac[0].1 == 1 && fac[0].0.len() == n + 1
    })
}

fn check_input(f: &[Integer]) -> Result<(), SteinerError> {
    if f.last().map(|c| !c.is_one()).unwrap_or(true) {
        return Err(SteinerError::NotMonic);
    }
    if f.len() != DEGREE + 1 {
        return Err(SteinerError::WrongDegree(f.len() - 1));
    }
    Ok(())
}

/// Refuse `p` unless it is a proven prime, with the error of [`FrobeniusRoots::new`] (the same
/// `try_is_prime`; about 10 us below `2^64`, and a proof above it is cached, so the second test
/// there is free). Called before [`precision_for`], which needs `p >= 2` (review 1 of SL, m1).
fn require_prime(p: &Integer) -> Result<(), SteinerError> {
    match rustmath_integers::prime::try_is_prime(p) {
        Ok(true) => Ok(()),
        Ok(false) => Err(SteinerError::Frobenius(FrobeniusRootsError::Prime(
            MathError::InvalidArgument(format!("{p} is not prime")),
        ))),
        Err(e) => Err(SteinerError::Frobenius(FrobeniusRootsError::Prime(e))),
    }
}

/// The facts about the two systems that the proof uses (module docs), checked once per process:
/// each is an S(4,7,23) (exact, `is_steiner_system`), shift-invariant, with 11 shift orbits, and
/// the two are distinct. About 1 ms in release.
fn golay_systems_checked() -> Result<(), SteinerError> {
    static CHECK: OnceLock<Result<(), String>> = OnceLock::new();
    CHECK
        .get_or_init(|| {
            for s in GolaySign::ALL {
                let hs = heptads(s);
                let blocks: Vec<Vec<usize>> = hs.iter().map(|&h| support(h)).collect();
                if blocks.len() != 253 || is_steiner_system(DEGREE, 4, &blocks) != Ok(true) {
                    return Err(format!("the {s} heptads are not an S(4,7,23)"));
                }
                if hs.iter().any(|&h| hs.binary_search(&rotate(h, 1)).is_err()) {
                    return Err(format!("the {s} system is not shift-invariant"));
                }
                if heptad_orbit_representatives(s).len() != 11 {
                    return Err(format!("the {s} system does not have 11 shift orbits"));
                }
            }
            if heptads(GolaySign::Plus) == heptads(GolaySign::Minus) {
                return Err("the two systems coincide".into());
            }
            Ok(())
        })
        .clone()
        .map_err(SteinerError::GolaySystems)
}

// ---------------------------------------------------------------------------
// Sums over heptads, by traces of orbit representatives
// ---------------------------------------------------------------------------

fn block_sum(roots: &FrobeniusRoots, mask: u32) -> GaloisRingElement {
    let r = roots.roots();
    support(mask)
        .into_iter()
        .fold(roots.ring().zero(), |acc, i| acc + r[i].clone())
}

/// `e_k` of the given elements (the usual recurrence).
fn elementary(
    values: &[GaloisRingElement],
    k: usize,
    zero: &GaloisRingElement,
    one: &GaloisRingElement,
) -> GaloisRingElement {
    let mut e = vec![one.clone()];
    e.extend((0..k).map(|_| zero.clone()));
    for (n, a) in values.iter().enumerate() {
        for j in (1..=k.min(n + 1)).rev() {
            e[j] = e[j].clone() + e[j - 1].clone() * a.clone();
        }
    }
    e[k].clone()
}

fn require_degree(roots: &FrobeniusRoots) -> Result<(), SteinerError> {
    let n = roots.roots().len();
    if n != DEGREE {
        return Err(SteinerError::WrongDegree(n));
    }
    Ok(())
}

/// The symmetric residues mod `p^N` of `T5^+` and `T5^-` for Frobenius-ordered roots (by traces
/// over the 11 shift-orbit representatives: 3 ring products each), at any precision. Without
/// `p^N > 2 B5` they decide nothing.
///
/// # Errors
///
/// `WrongDegree` when there are not 23 roots.
pub fn t5_candidates(roots: &FrobeniusRoots) -> Result<[(GolaySign, Integer); 2], SteinerError> {
    require_degree(roots)?;
    let fm = roots.frobenius();
    let m = roots.modulus();
    let value = |s: GolaySign| {
        let total = heptad_orbit_representatives(s)
            .iter()
            .fold(Integer::zero(), |acc, &rep| {
                let sh = block_sum(roots, rep);
                let s2 = sh.clone() * sh.clone();
                let s5 = s2.clone() * s2 * sh;
                acc + fm.trace(&s5)
            });
        symmetric_residue(&total, m)
    };
    Ok([
        (GolaySign::Plus, value(GolaySign::Plus)),
        (GolaySign::Minus, value(GolaySign::Minus)),
    ])
}

/// `sum_{H in S^sign} e_k(alpha_H)` mod `p^N`, symmetric residue.
fn heptad_elementary_sum(roots: &FrobeniusRoots, sign: GolaySign, k: usize) -> Integer {
    let fm = roots.frobenius();
    let r = roots.roots();
    let zero = roots.ring().zero();
    let one = roots.ring().one();
    let total = heptad_orbit_representatives(sign)
        .iter()
        .fold(Integer::zero(), |acc, &rep| {
            let vals: Vec<GaloisRingElement> =
                support(rep).into_iter().map(|i| r[i].clone()).collect();
            acc + fm.trace(&elementary(&vals, k, &zero, &one))
        });
    symmetric_residue(&total, roots.modulus())
}

/// `prod_{H} (Z - sum_{i in H} alpha_i)` over the shift orbits of the given representatives:
/// the product of their orbit polynomials, ascending, reduced mod `p^N` (not yet symmetric).
fn orbit_product(
    roots: &FrobeniusRoots,
    reps: &[u32],
    quantity: &'static str,
) -> Result<Vec<Integer>, SteinerError> {
    let m = roots.modulus();
    let mut total = vec![Integer::one()];
    for &rep in reps {
        let values: Vec<GaloisRingElement> = (0..DEGREE as u32)
            .map(|i| block_sum(roots, rotate(rep, i)))
            .collect();
        let o = orbit_polynomial(&values).map_err(|_| SteinerError::NotInBaseRing { quantity })?;
        total = zx::mul(&total, &o)
            .into_iter()
            .map(|c| c.modulo(m))
            .collect();
    }
    Ok(total)
}

fn certify_resolvent(
    raw: Vec<Integer>,
    roots: &FrobeniusRoots,
    bound: Integer,
    quantity: &'static str,
    certification: Certification,
) -> Result<CertifiedResolvent, SteinerError> {
    let m = roots.modulus();
    if raw.len() != 254 {
        return Err(SteinerError::NotInBaseRing { quantity });
    }
    let mut coefficients = Vec::with_capacity(raw.len());
    for (index, c) in raw.iter().enumerate() {
        match bounded_integer(c, m, &bound)? {
            Some(v) => coefficients.push(v),
            None => {
                return Err(SteinerError::BoundExceeded {
                    quantity,
                    index: Some(index),
                })
            }
        }
    }
    Ok(CertifiedResolvent {
        coefficients,
        bound,
        prime: roots.prime().clone(),
        precision: roots.precision(),
        certification,
    })
}

/// The representatives `{0, d}`, `d = 1..11`, of the shift orbits of the 253 duads.
fn duad_representatives() -> Vec<u32> {
    (1..=11u32).map(|d| 1 | (1 << d)).collect()
}

fn duad_resolvent_from(
    roots: &FrobeniusRoots,
    bound: Integer,
) -> Result<CertifiedResolvent, SteinerError> {
    let raw = orbit_product(roots, &duad_representatives(), "duad resolvent")?;
    certify_resolvent(
        raw,
        roots,
        bound,
        "duad resolvent",
        Certification::Symmetric,
    )
}

/// The duad resolvent `D(Z) = prod_{i<j} (Z - alpha_i - alpha_j)` of a monic `f` of degree 23
/// (paper, Section 6), from the Frobenius-ordered roots at the inert prime `p`. It is symmetric in
/// the roots, so it needs no labelling and no hypothesis ([`Certification::Symmetric`]). Each
/// coefficient is the symmetric residue mod `p^N`, `N = precision_for(p, (1 + 2M)^253, 0)`, and
/// lies within that bound.
///
/// Cost: `N` is about `253 log_p(2M)`, and the time grows faster than linearly in `N` (a lift
/// and 11 orbit polynomials over `GR(p^N, f)`). Measured in release at 29: f2 `N = 267` in
/// 1.3 s, f1 `N = 302` in 1.4 to 1.6 s. Le Lay's CC0 associate needs `N` about 12,400, about
/// 10 minutes by the estimate of design SL (Section 9; not run).
///
/// # Errors
///
/// `NotMonic`, `WrongDegree`, `Frobenius` (prime refused), `BoundExceeded` (a bug).
pub fn duad_resolvent(f: &[Integer], p: &Integer) -> Result<CertifiedResolvent, SteinerError> {
    check_input(f)?;
    require_prime(p)?;
    let bound = RootBound::fujiwara(f)?.duad_resolvent_bound();
    let n = precision_for(p, &bound, 0);
    let roots = FrobeniusRoots::new(f, p, n)?;
    duad_resolvent_from(&roots, bound)
}

// ---------------------------------------------------------------------------
// The labelling
// ---------------------------------------------------------------------------

/// Options of [`steiner_labelling`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabellingOptions {
    /// The labelling prime; `None`: the first inert prime in `[3, search_limit)`.
    pub prime: Option<u64>,
    /// End of the inert-prime search (default 10,000). Cost: one factorization of `f mod p` per
    /// prime of `[3, min(search_limit, fp_factor::MAX_P + 1))` until an inert one, so the whole
    /// range when `f` has none (1228 primes for the default, 0.6 s in release for a degree 23
    /// product of two factors).
    pub search_limit: u64,
    /// Margin of the precision: `p^N > 2 B5 10^extra_digits` (default 30, as the reference).
    /// Cost: each 10 digits add about `10 / log10 p` to `N` (7 at 29); f1, f2 at 29 with the
    /// default: `N = 31`, 30, a labelling in about 0.1 s in release.
    pub extra_digits: u32,
    /// An explicit precision `N`; refused (`PrecisionTooLow`) when `p^N <= 2 B5`. Cost: no upper
    /// limit; `p^N` is built at once and the lift runs at that `N`.
    pub precision: Option<u32>,
    /// After `Unresolved`, retry this many times, each 10 decimal digits more precise (default 0).
    /// Cost: each retry is a lift and a `T5` computation at the new `N`. An input that is always
    /// `Unresolved` (Galois group `C23`, such as `polsubcyclo(47, 23)`) spends every retry, so
    /// the precision and the time grow with `retries`.
    pub retries: u32,
}

impl Default for LabellingOptions {
    fn default() -> Self {
        LabellingOptions {
            prime: None,
            search_limit: 10_000,
            extra_digits: 30,
            precision: None,
            retries: 0,
        }
    }
}

/// The outcome of the decision rule (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LabellingStatus {
    /// Exactly one candidate passes: the roots carry the system of `sign`, and
    /// `epsilon(Frob_p) = sign`. Relative to [`GaloisHypothesis::M23`].
    Decided {
        /// The decided sign.
        sign: GolaySign,
    },
    /// Both candidates pass at every precision tried. Nothing is decided.
    Unresolved,
    /// Neither candidate passes: a proof (no hypothesis needed) that `Gal(f)` lies in no
    /// conjugate of M23. Not a labelling.
    NotInM23,
}

/// A Steiner labelling of the Frobenius-ordered roots of `f` at one prime ([`steiner_labelling`]).
#[derive(Clone, Debug)]
pub struct SteinerLabelling {
    status: LabellingStatus,
    root_bound: RootBound,
    t5_bound: Integer,
    residues: [(GolaySign, Integer); 2],
    attempts: Vec<u32>,
    roots: FrobeniusRoots,
}

/// Decide the Steiner system carried by the Frobenius-ordered roots of a monic `f` of degree 23
/// at an inert prime (paper, Proposition 5.1; module docs for the proof, the hypothesis and its
/// limits).
///
/// Steps: the root bound and `B5`; the prime; the precision (`precision_for(p, B5,
/// extra_digits)` or the given one); the roots; [`t5_candidates`]; [`integrality_dichotomy`]; on
/// `Unresolved`, up to `retries` further tries, each 10 digits more precise. Before the first
/// labelling the two Golay systems are checked once (they are what the proof uses).
///
/// # Errors
///
/// `NotMonic`, `WrongDegree`, `NoInertPrime`, `Frobenius` (prime not proven, not inert, or `f`
/// not squarefree mod `p`), `Integrality(PrecisionTooLow)` for a given precision with
/// `p^N <= 2 B5`, `GolaySystems` (a bug).
pub fn steiner_labelling(
    f: &[Integer],
    opts: &LabellingOptions,
) -> Result<SteinerLabelling, SteinerError> {
    check_input(f)?;
    golay_systems_checked()?;
    let root_bound = RootBound::fujiwara(f)?;
    let b5 = root_bound.t5_bound();
    let p = match opts.prime {
        Some(p) => p,
        None => first_inert_prime(f, 3, opts.search_limit).ok_or(SteinerError::NoInertPrime {
            below: opts.search_limit,
        })?,
    };
    let p = Integer::from(p);
    require_prime(&p)?;
    let n = match opts.precision {
        Some(n) => {
            let m = p.pow(n);
            if m <= Integer::from(2u64) * b5.clone() {
                return Err(SteinerError::Integrality(
                    IntegralityError::PrecisionTooLow {
                        modulus: m,
                        bound: b5,
                    },
                ));
            }
            n
        }
        None => precision_for(&p, &b5, opts.extra_digits),
    };
    let mut roots = FrobeniusRoots::new(f, &p, n)?;
    let mut attempts = vec![n];
    loop {
        let residues = t5_candidates(&roots)?;
        let cands: Vec<(GolaySign, Vec<Integer>)> = residues
            .iter()
            .map(|(s, r)| (*s, vec![r.clone()]))
            .collect();
        let status = match integrality_dichotomy(&cands, roots.modulus(), &b5)? {
            Dichotomy::Decided { label, .. } => LabellingStatus::Decided { sign: label },
            Dichotomy::Unresolved {
                gap: DichotomyGap::NoneWithinBound,
                ..
            } => LabellingStatus::NotInM23,
            Dichotomy::Unresolved {
                gap: DichotomyGap::SeveralWithinBound,
                ..
            } => LabellingStatus::Unresolved,
        };
        if status != LabellingStatus::Unresolved || attempts.len() > opts.retries as usize {
            return Ok(SteinerLabelling {
                status,
                root_bound,
                t5_bound: b5,
                residues,
                attempts,
                roots,
            });
        }
        // Ten more decimal digits.
        let target = roots.modulus().clone() * Integer::from(10u64).pow(10);
        let mut next = roots.precision();
        let mut pn = roots.modulus().clone();
        while pn < target {
            next += 1;
            pn = pn * p.clone();
        }
        roots = roots.with_precision(next)?;
        attempts.push(next);
    }
}

impl SteinerLabelling {
    /// The status.
    pub fn status(&self) -> &LabellingStatus {
        &self.status
    }

    /// The trusted input behind a `Decided` status.
    pub fn hypothesis(&self) -> GaloisHypothesis {
        GaloisHypothesis::M23
    }

    /// The decided sign, if any.
    pub fn sign(&self) -> Option<GolaySign> {
        match self.status {
            LabellingStatus::Decided { sign } => Some(sign),
            _ => None,
        }
    }

    /// `T5(f)`, the integer, when `Decided`.
    pub fn t5(&self) -> Option<&Integer> {
        self.sign().map(|s| self.t5_residue(s))
    }

    /// The residue of the rejected sign when `Decided` (an integer of absolute value above
    /// `B5` in the symmetric range).
    pub fn rejected_t5(&self) -> Option<&Integer> {
        self.sign().map(|s| self.t5_residue(s.opposite()))
    }

    /// The symmetric residue of `T5^sign` mod `p^N` at the final precision (always available).
    pub fn t5_residue(&self, sign: GolaySign) -> &Integer {
        &self
            .residues
            .iter()
            .find(|(s, _)| *s == sign)
            .expect("both signs")
            .1
    }

    /// `f`.
    pub fn polynomial(&self) -> &[Integer] {
        self.roots.polynomial()
    }

    /// The labelling prime.
    pub fn prime(&self) -> &Integer {
        self.roots.prime()
    }

    /// The final precision.
    pub fn precision(&self) -> u32 {
        self.roots.precision()
    }

    /// The precisions tried, in order (one unless `retries` was used).
    pub fn attempts(&self) -> &[u32] {
        &self.attempts
    }

    /// The root bound.
    pub fn root_bound(&self) -> &RootBound {
        &self.root_bound
    }

    /// `B5`.
    pub fn t5_bound(&self) -> &Integer {
        &self.t5_bound
    }

    /// The Frobenius-ordered roots at the final precision.
    pub fn roots(&self) -> &FrobeniusRoots {
        &self.roots
    }

    fn decided(&self) -> Result<(GolaySign, Certification), SteinerError> {
        match self.status {
            LabellingStatus::Decided { sign } => Ok((
                sign,
                Certification::Labelling {
                    hypothesis: GaloisHypothesis::M23,
                    sign,
                    prime: self.prime().clone(),
                    precision: self.precision(),
                },
            )),
            s => Err(SteinerError::NotDecided(s)),
        }
    }

    /// `J5` and `J7` of the decided system, at `N_J = max(N(B_J5, 0), N(B_J7, 0))` (the roots are
    /// projected or lifted to `N_J`), by traces over the 11 orbit representatives.
    ///
    /// Cost: `N_J` is about `7 log_p M`. At 29: f2 `N_J = 8`, f1 `N_J = 9` (milliseconds); Le
    /// Lay's CC0 associate `N_J = 344` (the test `cc0_golay_invariants`, with its labelling at
    /// `N = 270`, takes about 10 s in debug).
    ///
    /// # Errors
    ///
    /// `NotDecided`; `BoundExceeded` (impossible under the hypothesis).
    pub fn golay_invariants(&self) -> Result<GolayInvariants, SteinerError> {
        let (sign, certification) = self.decided()?;
        let p = self.prime();
        let b5 = self.root_bound.j5_bound();
        let b7 = self.root_bound.j7_bound();
        let nj = precision_for(p, &b5, 0).max(precision_for(p, &b7, 0));
        let roots = self.roots.with_precision(nj)?;
        let certify = |k: usize, bound: Integer, quantity: &'static str| {
            let r = heptad_elementary_sum(&roots, sign, k);
            match bounded_integer(&r, roots.modulus(), &bound)? {
                Some(value) => Ok(CertifiedInteger {
                    value,
                    bound,
                    prime: p.clone(),
                    precision: nj,
                    certification: certification.clone(),
                }),
                None => Err(SteinerError::BoundExceeded {
                    quantity,
                    index: None,
                }),
            }
        };
        let j5 = certify(5, b5, "J5")?;
        let j7 = certify(7, b7, "J7")?;
        Ok(GolayInvariants { j5, j7 })
    }

    /// The heptad resolvent `R(Z) = prod_{H in S} (Z - s_H)` of the decided system (paper,
    /// Theorem 5.2), at `N = precision_for(p, (1 + 7M)^253, 0)`: the product of the 11 orbit
    /// polynomials of degree 23.
    ///
    /// Cost: `N` is about `253 log_p(7M)`, and the time grows faster than linearly in `N`.
    /// Measured in release at 29: f2 `N = 360` in 1.9 s, f1 `N = 395` in 2.1 s. Le Lay's CC0
    /// associate needs `N` about 12,500, about 10 minutes by the estimate of design SL
    /// (Section 9; not run).
    ///
    /// # Errors
    ///
    /// `NotDecided`; `BoundExceeded`, `NotInBaseRing` (bugs or a false hypothesis).
    pub fn heptad_resolvent(&self) -> Result<CertifiedResolvent, SteinerError> {
        let (sign, certification) = self.decided()?;
        let bound = self.root_bound.heptad_resolvent_bound();
        let n = precision_for(self.prime(), &bound, 0);
        let roots = self.roots.with_precision(n)?;
        let raw = orbit_product(
            &roots,
            heptad_orbit_representatives(sign),
            "heptad resolvent",
        )?;
        certify_resolvent(raw, &roots, bound, "heptad resolvent", certification)
    }

    /// The duad resolvent (see [`duad_resolvent`]) at the labelling prime. It does not need the
    /// labelling: it works for every status.
    ///
    /// Cost: as [`duad_resolvent`] (f2 1.3 s, f1 1.4 to 1.6 s in release; CC0 about 10
    /// minutes, not run), but the lift continues from the labelling's roots.
    ///
    /// # Errors
    ///
    /// `BoundExceeded` (a bug).
    pub fn duad_resolvent(&self) -> Result<CertifiedResolvent, SteinerError> {
        let bound = self.root_bound.duad_resolvent_bound();
        let n = precision_for(self.prime(), &bound, 0);
        let roots = self.roots.with_precision(n)?;
        duad_resolvent_from(&roots, bound)
    }

    /// The Frobenius sign at another inert prime `q` from this labelling's certified `T5`
    /// ([`frobenius_sign_given_t5`]).
    ///
    /// # Errors
    ///
    /// `NotDecided`; those of [`frobenius_sign_given_t5`].
    pub fn frobenius_sign(
        &self,
        q: &Integer,
        precision: u32,
    ) -> Result<FrobeniusSign, SteinerError> {
        let t5 = self
            .t5()
            .ok_or(SteinerError::NotDecided(self.status))?
            .clone();
        frobenius_sign_given_t5(self.polynomial(), q, &t5, precision)
    }
}

/// The sign `epsilon(Frob_q)` (paper, Remark after Theorem 5.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrobeniusSign {
    /// Exactly one candidate matches `T5` mod `q^N`: the sign (relative to the hypothesis and to
    /// the value of `T5`).
    Decided(GolaySign),
    /// Both match (raise `N`) or neither matches (the given `T5` or the hypothesis is wrong).
    Unresolved {
        /// The signs whose candidate matches.
        matching: Vec<GolaySign>,
    },
}

/// The Frobenius sign at the inert prime `q` from a known `T5(f)`: compute `T5^+`, `T5^-` mod
/// `q^N` and compare with `t5`. The true sign `e` has `T5^e = T5(f)` exactly, so its residue
/// matches; if exactly one matches, it is `e`.
///
/// The given `t5` is a trusted input (as the hypothesis): a wrong `t5` that happens to match one
/// candidate gives a wrong sign. Prefer [`SteinerLabelling::frobenius_sign`], whose `T5` is
/// certified. Since `T5^+ - T5^- = 120 (J5^+ - J5^-)` (design SL, 2.7, review 1.7), the
/// candidates agree mod 8, 3 and 5, so `q = 2, 3, 5` with `N <= v_q(120)` never separates them.
///
/// # Errors
///
/// `NotMonic`, `WrongDegree`, `Frobenius` (`q` not inert, `N = 0`).
pub fn frobenius_sign_given_t5(
    f: &[Integer],
    q: &Integer,
    t5: &Integer,
    precision: u32,
) -> Result<FrobeniusSign, SteinerError> {
    check_input(f)?;
    golay_systems_checked()?;
    let roots = FrobeniusRoots::new(f, q, precision)?;
    let m = roots.modulus();
    let target = t5.modulo(m);
    let matching: Vec<GolaySign> = t5_candidates(&roots)?
        .iter()
        .filter(|(_, r)| r.modulo(m) == target)
        .map(|(s, _)| *s)
        .collect();
    Ok(match matching.as_slice() {
        [s] => FrobeniusSign::Decided(*s),
        _ => FrobeniusSign::Unresolved { matching },
    })
}
