//! # L-functions of elliptic curves over Q
//!
//! Two layers live here:
//!
//! 1. **The certified analytic layer** ([`CurveLSeries`], [`LValue`],
//!    [`AnalyticRank`]): the numeric Taylor coefficients L^{(r)}(E,1)/r! of
//!    every order r ([`CurveLSeries::l_derivative`]; r = 0 and r = 1 are the
//!    older [`CurveLSeries::l1`] / [`CurveLSeries::l1_derivative`]) over
//!    [`BigFloat`] with
//!    rigorous tail bounds and a documented rounding allowance, plus the
//!    honest analytic-rank lattice. All coefficients are EXACT (integer
//!    a_n from point counts / Tate reduction types, multiplicativity), the
//!    functional-equation sign is the EXACT global root number from Tate
//!    local data ([`crate::rootnumber`]), and the honesty contract is:
//!
//!    * a numeric value can certify NONZERO ([`LValue::certified_nonzero`])
//!      but can NEVER certify zero;
//!    * the only certified zeros are exact statements: ε = −1 forces
//!      L(E,1) = 0 via the functional equation, and externally supplied
//!      Manin–Birch winding certificates (see
//!      [`LFunction::analytic_rank_with_exact_l1`] and
//!      [`LFunction::analytic_rank_with_exact_vanishing`]);
//!    * uncertified cases are reported as [`AnalyticRank::Unresolved`],
//!      [`AnalyticRank::AtLeastTwoUnresolved`] or
//!      [`AnalyticRank::AtLeastNUnresolved`] — NEVER a bare integer.
//!
//!    A parity warning, because it is easy to get backwards: ε = +1 makes
//!    the COMPLETED Λ(1+u) an even function of u, so the Λ-coefficients of
//!    odd index vanish exactly — but L = c^{1+u}Λ(1+u)/Γ(1+u) is NOT even,
//!    and its odd derivatives at s = 1 are generally nonzero (e.g.
//!    ε(11a) = +1 yet L'(11a,1) = 0.30870853…, PARI-confirmed). What parity
//!    gives is that the ORDER OF VANISHING is even (ε = +1) or odd (ε = −1).
//!    That is the exact fact the rank lattice runs on.
//!
//!    The analytic continuation and functional equation
//!    Λ(s) = ε·Λ(2−s), Λ(s) = N^{s/2}(2π)^{−s}Γ(s)L(E,s), rest on the
//!    modularity theorem (Wiles, Taylor–Wiles, BCDT 2001; conductor and
//!    ε-factor match by Carayol). Given modularity, the formulas and
//!    bounds below are exactly those derived and independently verified
//!    (mpmath, split-point tests, brute integrals) for weight-2 rational
//!    newforms in `rustmath-modular::modsym::lseries` (stage 1):
//!
//!    * ε = +1:  L(E,1) = 2 Σ_{n≥1} (a_n/n) e^{−2πn/√N}, geometric tail;
//!    * ε = −1:  L(E,1) = 0 exactly, and
//!      L'(E,1) = 2 Σ_{n≥1} (a_n/n) E₁(2πn/√N), E₁(x) = ∫_x^∞ e^{−t}dt/t,
//!      with tail via E₁(x) ≤ e^{−x} for x ≥ 1.
//!
//!    Tail hypothesis: |a_n| ≤ d(n)√n (Deligne; the Hasse bound
//!    |a_p| ≤ 2√p is ALSO asserted at runtime on every good prime consumed,
//!    and |a_p| ∈ {0,1} at bad primes by the Tate reduction type), plus
//!    d(n) ≤ 2√n, so |a_n|/n ≤ 2 and with c = 2π/√N the tail after M terms
//!    is ≤ 4e^{−c(M+1)}/(1−e^{−c}) for both series (E₁ ≤ e^{−x} enforced by
//!    c(M+1) ≥ 1).
//!
//! 2. **The legacy f64 exploration layer** ([`ComplexNum`], the Euler
//!    factors and the raw Dirichlet partial sums in [`LFunction`]): NOT
//!    certified — the Dirichlet series only converges for Re(s) > 3/2. It
//!    is kept for interface exploration; nothing certified consumes it.
//!    Its coefficients are at least exact now (they come from
//!    [`l_series_coefficients`]).
//!
//! The Euler–Mascheroni constant and E₁ implementations are verbatim
//! ports of the stage-1 code in `rustmath-modular::modsym::lseries`
//! (kept crate-private here to avoid a lib dependency on rustmath-modular;
//! their gates — mpmath to 60 and 52 digits — are reproduced in reduced
//! form in this file's tests so the copies cannot rot silently).

use crate::curve::EllipticCurve;
use crate::rootnumber::global_root_number;
use rustmath_core::analytic::RealField;
use rustmath_core::ordering::OrderedRing;
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_reals::BigFloat;
use std::f64::consts::PI;
use std::fmt;

/// log2(10), rounded up a hair (used only to size working precisions).
const LOG2_10: f64 = 3.321928094887363;

// ---------------------------------------------------------------------------
// Exact L-series coefficients
// ---------------------------------------------------------------------------

/// Deterministic primality test by trial division (inputs are small).
fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// The exact L-series coefficients a_1..a_nmax of the curve (index 0
/// unused, set to 0): a_p = [`EllipticCurve::compute_a_p`] (point counts at
/// good primes, reduction type at bad primes — bad meaning positive
/// conductor exponent, so primes where the model is merely non-minimal are
/// handled correctly), a_{p^{k+1}} = a_p·a_{p^k} − p·a_{p^{k−1}} at good p,
/// a_{p^k} = a_p^k at bad p, and a_{mn} = a_m·a_n for coprime m, n
/// (weight 2, trivial character).
pub fn l_series_coefficients(curve: &EllipticCurve, nmax: usize) -> Vec<Integer> {
    let mut a = vec![Integer::zero(); nmax + 1];
    if nmax == 0 {
        return a;
    }
    a[1] = Integer::one();
    for p in 2..=nmax as u64 {
        if !is_prime_u64(p) {
            continue;
        }
        let p_int = Integer::from(p as i64);
        let ap = Integer::from(curve.compute_a_p(&p_int));
        // "bad" for the recurrence = positive conductor exponent (NOT just
        // p | disc of the given, possibly non-minimal, model).
        let bad = curve.is_bad_prime(&p_int) && curve.local_data(&p_int).conductor_exponent > 0;
        a[p as usize] = ap.clone();
        let mut pk = (p * p) as usize;
        while pk <= nmax {
            a[pk] = if bad {
                a[pk / p as usize].clone() * ap.clone()
            } else {
                ap.clone() * a[pk / p as usize].clone()
                    - p_int.clone() * a[pk / (p * p) as usize].clone()
            };
            pk *= p as usize;
        }
    }
    // composites: split off the full power of the smallest prime factor
    for m in 2..=nmax {
        let mut sp = 0usize;
        for q in 2..=m {
            if m % q == 0 {
                sp = q;
                break;
            }
        }
        if sp == m {
            continue; // prime (or prime power handled above when mm == 1)
        }
        let mut mm = m;
        let mut pk = 1usize;
        while mm % sp == 0 {
            mm /= sp;
            pk *= sp;
        }
        if mm > 1 {
            a[m] = a[pk].clone() * a[mm].clone();
        }
    }
    a
}

// ---------------------------------------------------------------------------
// LValue: a numeric value with its honesty envelope
// ---------------------------------------------------------------------------

/// A computed L-value (or derivative) with its honesty envelope.
/// Same contract as `rustmath_modular::LValue` (stage 1).
#[derive(Debug, Clone)]
pub struct LValue {
    /// The computed value (series truncated after finitely many terms).
    pub value: BigFloat,
    /// Rigorous bound on the omitted tail (see the module docs); exactly
    /// zero when the value is exact (the ε = −1 central zero).
    pub tail_bound: BigFloat,
    /// Documented allowance for BigFloat rounding across the summation
    /// (an engineering bound, spelled out in `rounding_note`; the working
    /// precision carries ≥ 64 guard bits past the requested digits).
    pub rounding_allowance: BigFloat,
    /// Human-readable statement of what was computed and what error model
    /// the two bounds follow.
    pub rounding_note: String,
}

impl LValue {
    /// True iff the value is certifiably nonzero: |value| exceeds the tail
    /// bound plus the documented rounding allowance. (A `false` return
    /// certifies NOTHING: numeric computation can never certify a zero.)
    pub fn certified_nonzero(&self) -> bool {
        let budget = self.tail_bound.clone() + self.rounding_allowance.clone();
        OrderedRing::abs(&self.value) > budget
    }

    /// The total certified error budget (tail bound + rounding allowance):
    /// the true value lies within this of `value`, modulo the documented
    /// rounding model.
    pub fn error_budget(&self) -> BigFloat {
        self.tail_bound.clone() + self.rounding_allowance.clone()
    }
}

// ---------------------------------------------------------------------------
// Euler-Mascheroni constant and E_1 (crate-private ports from
// rustmath-modular::modsym::lseries, stage 1; see the module docs)
// ---------------------------------------------------------------------------

/// 2^k as an Integer (k small).
pub(crate) fn pow2_integer(k: u64) -> Integer {
    let mut out = Integer::one();
    let two = Integer::from(2);
    for _ in 0..k {
        out = &out * &two;
    }
    out
}

/// The rational 2^-k as a BigFloat at the given precision.
pub(crate) fn pow2_neg(k: u64, prec: u64) -> BigFloat {
    let r = Rational::new(Integer::one(), pow2_integer(k)).expect("power of two is nonzero");
    BigFloat::from_rational(&r, prec)
}

/// Bernoulli numbers B_0..B_m as exact rationals, via the defining
/// recurrence sum_{k=0}^{n} C(n+1, k) B_k = 0 (B_0 = 1 seed; B_1 = -1/2
/// convention, irrelevant here since only even indices are consumed).
pub(crate) fn bernoulli_numbers(m: usize) -> Vec<Rational> {
    let mut b: Vec<Rational> = Vec::with_capacity(m + 1);
    b.push(Rational::one());
    let mut row = vec![Integer::one(), Integer::from(2), Integer::one()];
    for n in 1..=m {
        let mut acc = Rational::zero();
        for (k, bk) in b.iter().enumerate() {
            if !bk.is_zero() {
                acc = &acc + &(&Rational::from_integer(row[k].clone()) * bk);
            }
        }
        let cn = Rational::from_integer(row[n].clone());
        let bn = -&(&acc / &cn);
        b.push(bn);
        let mut next = vec![Integer::one()];
        for w in row.windows(2) {
            next.push(&w[0] + &w[1]);
        }
        next.push(Integer::one());
        row = next;
    }
    b
}

/// The Euler-Mascheroni constant at `prec` bits, absolute error
/// < 2^-(prec+8) (exact-rational Euler-Maclaurin core + one ln; stage-1
/// derivation, mpmath-gated there to 60 digits and re-gated below).
pub(crate) fn euler_gamma(prec: u64) -> BigFloat {
    let n = ((prec as f64 + 24.0) * 0.14) as u64 + 4;
    let threshold =
        Rational::new(Integer::one(), pow2_integer(prec + 24)).expect("power of two is nonzero");
    let n_int = Integer::from(n as i64);
    let mut core = Rational::zero();
    for k in 1..=n {
        core = &core + &Rational::new(Integer::one(), Integer::from(k as i64)).expect("k > 0");
    }
    core = &core - &Rational::new(Integer::one(), Integer::from(2 * n as i64)).expect("2n > 0");
    let jmax = 2 * n as usize + 8;
    let bern = bernoulli_numbers(2 * jmax + 2);
    let mut n2j = &n_int * &n_int;
    let mut reached = false;
    for j in 1..=jmax {
        let term = &bern[2 * j] / &Rational::from_integer(&Integer::from(2 * j as i64) * &n2j);
        let abs_term = if term < Rational::zero() {
            -&term
        } else {
            term.clone()
        };
        if abs_term < threshold {
            reached = true;
            break;
        }
        core = &core + &term;
        n2j = &(&n2j * &n_int) * &n_int;
    }
    assert!(
        reached,
        "euler_gamma: asymptotic series failed to reach threshold (n too small)"
    );
    let wp = prec + 32;
    let ln_n = RealField::ln(&BigFloat::from_integer(&n_int, wp));
    (BigFloat::from_rational(&core, wp) - ln_n).with_precision(prec)
}

/// Working precision for the E_1 alternating series at argument x.
pub(crate) fn e1_working_precision(xf: f64, prec: u64) -> u64 {
    prec + 48 + (std::f64::consts::LOG2_E * xf.max(0.0)).ceil() as u64
}

/// The exponential integral E_1(x) = ∫_x^∞ e^{−t}/t dt for x > 0, absolute
/// error < 2^-(prec+16): alternating series −γ − ln x + Σ (−1)^{k+1}
/// x^k/(k·k!) with a cancellation budget of ⌈x·log₂e⌉ + 48 guard bits
/// (stage-1 derivation, mpmath-gated there to 52+ digits and re-gated
/// below).
#[cfg_attr(not(test), allow(dead_code))] // batch callers use e1_with_gamma; the tests gate this wrapper
fn exp_integral_e1(x: &BigFloat, prec: u64) -> Result<BigFloat, String> {
    let wp = e1_working_precision(x.to_f64(), prec);
    let g = euler_gamma(wp);
    e1_with_gamma(x, prec, &g)
}

/// E_1 with a caller-supplied gamma (carried at ≥ the working precision
/// needed for this x), so batch callers do not recompute gamma per term.
fn e1_with_gamma(x: &BigFloat, prec: u64, gamma: &BigFloat) -> Result<BigFloat, String> {
    if x.sign() <= 0 {
        return Err("exp_integral_e1 requires x > 0".to_string());
    }
    let xf = x.to_f64();
    let wp = e1_working_precision(xf, prec);
    if RealField::precision(gamma) < wp {
        return Err(format!(
            "e1_with_gamma: gamma carries {} bits, need {wp}",
            RealField::precision(gamma)
        ));
    }
    let xw = x.with_precision(wp);
    let mut term = BigFloat::one_prec(wp);
    let mut sum = BigFloat::zero_prec(wp);
    let cutoff = pow2_neg(wp, wp);
    let mut k: i64 = 1;
    loop {
        term = term * xw.clone() / BigFloat::from_integer(&Integer::from(k), wp);
        let contrib = term.clone() / BigFloat::from_integer(&Integer::from(k), wp);
        sum = if k % 2 == 1 {
            sum + contrib
        } else {
            sum - contrib
        };
        if (k as f64) > xf && OrderedRing::abs(&term) < cutoff {
            break;
        }
        k += 1;
        if k > 10_000_000 {
            return Err("exp_integral_e1: series failed to converge".to_string());
        }
    }
    Ok((sum - gamma.with_precision(wp) - RealField::ln(&xw)).with_precision(prec))
}

// ---------------------------------------------------------------------------
// CurveLSeries: certified numeric L(E,1), L'(E,1)
// ---------------------------------------------------------------------------

/// The L-series of an elliptic curve over Q, with exact conductor (Tate)
/// and exact functional-equation sign (the global root number from Tate
/// local data at p ≥ 5 and the Kraus/Halberstadt tables at the wild
/// additive primes 2, 3; see [`crate::rootnumber`]). Construction now
/// succeeds for every nonsingular curve.
pub struct CurveLSeries {
    curve: EllipticCurve,
    conductor: Integer,
    epsilon: i8,
}

impl CurveLSeries {
    /// Attach to a curve: computes the exact conductor (Tate) and the
    /// exact global root number (complete at every prime, including the
    /// wild additive primes 2, 3 via the Kraus/Halberstadt tables). The
    /// `Err` arm now fires only for singular input; the `Result` is kept
    /// for API stability.
    pub fn new(curve: &EllipticCurve) -> Result<Self, String> {
        if curve.is_singular() {
            return Err("CurveLSeries: curve is singular".to_string());
        }
        let epsilon = global_root_number(curve)?;
        let conductor = curve.compute_conductor();
        Ok(CurveLSeries {
            curve: curve.clone(),
            conductor,
            epsilon,
        })
    }

    /// The exact conductor N (Tate's algorithm).
    pub fn conductor(&self) -> &Integer {
        &self.conductor
    }

    /// The exact functional-equation sign ε = w(E) (global root number).
    pub fn root_number(&self) -> i8 {
        self.epsilon
    }

    /// Exact a_1..a_nmax (index 0 unused), with runtime wrongness
    /// detectors: the Hasse bound is asserted inside
    /// [`EllipticCurve::compute_a_p`] at every good prime, and here
    /// |a_p| = 1 at multiplicative primes (p ‖ N) and a_p = 0 at additive
    /// primes (p² | N) are asserted against the conductor — these are also
    /// exactly the hypotheses of the tail bound at the consumed primes.
    pub fn coefficients(&self, nmax: usize) -> Vec<Integer> {
        let a = l_series_coefficients(&self.curve, nmax);
        for p in 2..=nmax as u64 {
            if !is_prime_u64(p) {
                continue;
            }
            let p_int = Integer::from(p as i64);
            let v = self.conductor.valuation(&p_int);
            let ap = a[p as usize].clone();
            if v == 1 {
                assert!(
                    ap.clone() * ap.clone() == Integer::one(),
                    "|a_{}| != 1 at a multiplicative prime: bug",
                    p
                );
            } else if v >= 2 {
                assert!(ap.is_zero(), "a_{} != 0 at an additive prime: bug", p);
            }
        }
        a
    }

    /// Truncation point M: smallest M with 4e^{−c(M+1)}/(1−e^{−c}) <
    /// 10^{−(digits+3)} (computed in f64, then the bound re-evaluated
    /// rigorously in BigFloat for the returned envelope).
    fn truncation_point(&self, digits: usize) -> usize {
        let n =
            <Integer as NumericConversion>::to_f64(&self.conductor).expect("conductor fits in f64");
        let c = 2.0 * PI / n.sqrt();
        let target = (digits as f64 + 3.0) * std::f64::consts::LN_10;
        let m_plus_1 = (target + 4.0f64.ln() - (1.0 - (-c).exp()).ln()) / c;
        (m_plus_1.ceil() as usize).max(10) + 2
    }

    /// The rigorous BigFloat tail bound 2·[4e^{−c(M+1)}/(1−e^{−c})] (the
    /// leading factor 2 absorbs the bound's own rounding).
    fn tail_bound(&self, m: usize, wp: u64) -> BigFloat {
        let c = self.decay_constant(wp);
        let e_c = RealField::exp(&(-c.clone()));
        let numer =
            RealField::exp(&(-(c * BigFloat::from_integer(&Integer::from((m + 1) as i64), wp))));
        let denom = BigFloat::one_prec(wp) - e_c;
        let eight = BigFloat::from_integer(&Integer::from(8), wp);
        eight * numer / denom
    }

    /// c = 2π/√N at wp bits.
    fn decay_constant(&self, wp: u64) -> BigFloat {
        let two = BigFloat::from_integer(&Integer::from(2), wp);
        let pi = <BigFloat as RealField>::pi(wp);
        let sqrt_n = RealField::sqrt(&BigFloat::from_integer(&self.conductor, wp));
        two * pi / sqrt_n
    }

    /// L(E,1) to about `digits` decimal digits.
    ///
    /// ε = +1: value = 2 Σ_{n≤M} (a_n/n) e^{−2πn/√N} with the rigorous
    /// geometric tail bound of the module docs.
    ///
    /// ε = −1: the EXACT zero (tail and allowance exactly zero; the zero
    /// comes from the exact functional-equation sign — the global root
    /// number derived from Tate local data — not from numerics).
    pub fn l1(&self, digits: usize) -> LValue {
        let wp = (digits as f64 * LOG2_10).ceil() as u64 + 64;
        if self.epsilon == -1 {
            return LValue {
                value: BigFloat::zero_prec(wp),
                tail_bound: BigFloat::zero_prec(wp),
                rounding_allowance: BigFloat::zero_prec(wp),
                rounding_note: "L(E,1) = 0 EXACTLY: epsilon = -1 in the functional \
                    equation Lambda(s) = epsilon Lambda(2-s), with epsilon the \
                    global root number computed exactly from Tate local data \
                    (crate::rootnumber; no numerics involved). Analytic \
                    continuation via modularity (Wiles/BCDT)."
                    .to_string(),
            };
        }
        let m = self.truncation_point(digits);
        let a = self.coefficients(m);
        let c = self.decay_constant(wp);
        let mut sum = BigFloat::zero_prec(wp);
        let mut abs_sum = BigFloat::zero_prec(wp);
        for (n, an) in a.iter().enumerate().skip(1) {
            if an.is_zero() {
                continue;
            }
            let n_bf = BigFloat::from_integer(&Integer::from(n as i64), wp);
            let coeff = BigFloat::from_integer(an, wp) / n_bf.clone();
            let term = coeff * RealField::exp(&(-(c.clone() * n_bf)));
            abs_sum = abs_sum + OrderedRing::abs(&term);
            sum = sum + term;
        }
        let two = BigFloat::from_integer(&Integer::from(2), wp);
        let value = two.clone() * sum;
        let tail = self.tail_bound(m, wp);
        let ops = BigFloat::from_integer(&Integer::from(16 * (m as i64 + 4)), wp);
        let allowance = two * (abs_sum + BigFloat::one_prec(wp)) * ops * pow2_neg(wp - 8, wp);
        LValue {
            value,
            tail_bound: tail,
            rounding_allowance: allowance,
            rounding_note: format!(
                "L(E,1) = 2 sum_(n=1..{m}) (a_n/n) exp(-2 pi n / sqrt({n})) for \
                 epsilon = +1, exact integer a_n (point counts + Tate reduction \
                 types), BigFloat arithmetic at {wp} bits. tail_bound = \
                 2 * [4 e^(-c(M+1))/(1-e^(-c))] from |a_n| <= d(n) sqrt(n) \
                 (Deligne; Hasse asserted at every consumed good prime) and \
                 d(n) <= 2 sqrt(n); rounding_allowance covers <= 16(M+4) \
                 operations of relative error 2^-({wp}-8) on the \
                 term-magnitude sum (engineering bound, 64 guard bits past \
                 the requested {digits} digits).",
                n = self.conductor,
            ),
        }
    }

    /// L'(E,1) to about `digits` decimal digits, for ε = −1 ONLY (the
    /// derivation of the E₁ series uses Λ(1) = 0; an honest error is
    /// returned otherwise): L'(E,1) = 2 Σ (a_n/n) E₁(2πn/√N), rigorous
    /// tail as in the module docs.
    pub fn l1_derivative(&self, digits: usize) -> Result<LValue, String> {
        if self.epsilon != -1 {
            return Err(
                "l1_derivative requires epsilon = -1 (the series derivation uses \
                 Lambda(1) = 0); for epsilon = +1 compute l1 instead"
                    .to_string(),
            );
        }
        let m = self.truncation_point(digits);
        let nf =
            <Integer as NumericConversion>::to_f64(&self.conductor).expect("conductor fits in f64");
        let cf64 = 2.0 * PI / nf.sqrt();
        if cf64 * ((m + 1) as f64) < 1.0 {
            return Err("internal: c(M+1) < 1, E_1 tail bound inapplicable".to_string());
        }
        let wp = (digits as f64 * LOG2_10).ceil() as u64 + 64;
        let a = self.coefficients(m);
        let c = self.decay_constant(wp);
        let gamma = euler_gamma(e1_working_precision(cf64 * (m as f64 + 2.0), wp));
        let mut sum = BigFloat::zero_prec(wp);
        let mut abs_sum = BigFloat::zero_prec(wp);
        for (n, an) in a.iter().enumerate().skip(1) {
            if an.is_zero() {
                continue;
            }
            let n_bf = BigFloat::from_integer(&Integer::from(n as i64), wp);
            let coeff = BigFloat::from_integer(an, wp) / n_bf.clone();
            let e1 = e1_with_gamma(&(c.clone() * n_bf), wp, &gamma)?;
            let term = coeff * e1;
            abs_sum = abs_sum + OrderedRing::abs(&term);
            sum = sum + term;
        }
        let two = BigFloat::from_integer(&Integer::from(2), wp);
        let value = two.clone() * sum;
        let tail = self.tail_bound(m, wp);
        let ops = BigFloat::from_integer(&Integer::from(16 * (m as i64 + 4)), wp);
        let allowance = two * (abs_sum + BigFloat::one_prec(wp)) * ops * pow2_neg(wp - 16, wp);
        Ok(LValue {
            value,
            tail_bound: tail,
            rounding_allowance: allowance,
            rounding_note: format!(
                "L'(E,1) = 2 sum_(n=1..{m}) (a_n/n) E_1(2 pi n / sqrt({n})) for \
                 epsilon = -1, exact integer a_n, BigFloat arithmetic at {wp} \
                 bits (E_1 carries its own cancellation budget). tail_bound = \
                 2 * [4 e^(-c(M+1))/(1-e^(-c))] using E_1(x) <= e^-x for x >= 1 \
                 and |a_n|/n <= 2 (Deligne + divisor pairing); \
                 rounding_allowance covers <= 16(M+4) operations of relative \
                 error 2^-({wp}-16) on the term-magnitude sum (engineering \
                 bound; E_1 itself carries absolute error < 2^-({wp}+16) by \
                 its own guard bits).",
                n = self.conductor,
            ),
        })
    }

    /// The r-th **Taylor coefficient** of L(E,s) at s = 1, i.e.
    /// L^{(r)}(E,1) / r!  (this is the normalisation the BSD formula wants;
    /// multiply by r! for the bare derivative). Unconditional: valid for
    /// every r ≥ 0 whatever the order of vanishing is.
    ///
    /// # The formula
    ///
    /// With c = 2π/√N, Λ(s) = N^{s/2}(2π)^{−s}Γ(s)L(E,s) = c^{−s}Γ(s)L(E,s)
    /// and Λ(s) = ε·Λ(2−s) (modularity: Wiles/BCDT, Carayol). Writing
    /// g(t) = Σ a_n e^{−cnt}, the substitution y = t/√N in
    /// Λ(s) = N^{s/2}∫_0^∞ f(iy) y^{s−1} dy gives Λ(s) = ∫_0^∞ g(t) t^{s−1} dt,
    /// and the Fricke relation g(1/t) = ε t² g(t) folds this to the entire
    ///
    /// ```text
    /// Lambda(1+u) = int_1^inf g(t) (t^u + eps t^-u) dt
    ///             = sum_{m>=0} u^m (1 + eps(-1)^m) sum_n a_n I_m(cn)/m!,
    /// I_m(x) = int_1^inf e^{-xt} (log t)^m dt.
    /// ```
    ///
    /// So Λ_m := [u^m]Λ(1+u) vanishes EXACTLY whenever m has the wrong
    /// parity for ε (this is the functional equation, an exact statement).
    /// Setting G_m(x) = x·I_m(x)/m! (see [`crate::ltaylor::g_kernels`];
    /// G_0(x) = e^{−x}, G_1(x) = E_1(x)) and S_m = Σ_n (a_n/n) G_m(cn),
    /// Λ_m = (1 + ε(−1)^m)·S_m/c. Since L(1+u) = c^{1+u}Λ(1+u)/Γ(1+u) and
    /// c^{1+u}/Γ(1+u) = c·exp(P(u)) with
    ///
    /// ```text
    /// P(u) = (ln c + gamma) u + sum_{k>=2} (-1)^(k+1) zeta(k) u^k / k
    /// ```
    ///
    /// (the Taylor series of −log Γ(1+u), plus u·ln c — note the sign against
    /// the +log Γ series inside [`crate::ltaylor::g_kernels`]), the factor c
    /// cancels and, with B_j := [u^j] exp(P(u)),
    ///
    /// ```text
    /// L^(r)(E,1)/r!  =  2 * sum_{m <= r, m == p (mod 2)}  B_(r-m) * S_m,
    /// p = 0 if eps = +1, p = 1 if eps = -1.
    /// ```
    ///
    /// Sanity: r = 0, ε = +1 gives 2·S_0 = 2Σ(a_n/n)e^{−cn} — exactly
    /// [`Self::l1`]; r = 1, ε = −1 gives 2·S_1 = 2Σ(a_n/n)E_1(cn) — exactly
    /// [`Self::l1_derivative`]. Both agreements are asserted in the tests.
    ///
    /// # Parity: what is and is not exactly zero
    ///
    /// The parity sum is EMPTY exactly when r = 0 and ε = −1, so that (and
    /// only that) case returns an exact zero `LValue` (tail and rounding
    /// allowance exactly 0). It is **not** true that ε = +1 forces every odd
    /// derivative of L to vanish: only the *completed* Λ is even in u, and
    /// L = c^{1+u}Λ/Γ(1+u) is not. For instance ε(11a) = +1 while
    /// L'(11a,1) = 2·B_1·S_0 = (ln c + γ)·L(11a,1) = 0.30870853…, which PARI
    /// confirms. What the functional equation *does* give is that the ORDER
    /// of vanishing has the parity of ε — that is what the analytic-rank
    /// lattice uses, and it is exact.
    ///
    /// # Tail bound (rigorous)
    ///
    /// log t ≤ t − 1 gives I_m(x) ≤ e^{−x} m!/x^{m+1}, hence
    /// 0 ≤ G_m(x) ≤ e^{−x}/x^m ≤ e^{−x} for x ≥ 1. With |a_n| ≤ d(n)√n
    /// (Deligne) and d(n) ≤ 2√n, |a_n/n| ≤ 2, so truncating every S_m at
    /// n ≤ M (with c(M+1) ≥ 1, enforced) omits at most
    /// 2·Σ_{n>M} e^{−cn} = 2e^{−c(M+1)}/(1−e^{−c}) from each S_m, and the
    /// omitted part of the answer is at most
    ///
    /// ```text
    /// 2 * [ sum_{m <= r, m == p (2)} |B_(r-m)| ] * 2 e^(-c(M+1)) / (1 - e^-c),
    /// ```
    ///
    /// which is what `tail_bound` reports (with the same factor-2 headroom as
    /// [`Self::l1`], to absorb the bound's own rounding). For r = 0 this is
    /// bit-for-bit the `l1` bound.
    ///
    /// `Err` only for the honest reasons: c(M+1) < 1 (bound inapplicable) or
    /// a kernel that failed to converge.
    pub fn l_derivative(&self, r: u32, digits: usize) -> Result<LValue, String> {
        let wp = (digits as f64 * LOG2_10).ceil() as u64 + 64;
        let parity: u32 = if self.epsilon == 1 { 0 } else { 1 };

        if r < parity {
            // r = 0 with epsilon = -1: the parity sum is empty.
            return Ok(LValue {
                value: BigFloat::zero_prec(wp),
                tail_bound: BigFloat::zero_prec(wp),
                rounding_allowance: BigFloat::zero_prec(wp),
                rounding_note: "L(E,1) = 0 EXACTLY: epsilon = -1 in the functional \
                    equation Lambda(s) = epsilon Lambda(2-s), with epsilon the \
                    global root number computed exactly from Tate local data \
                    (crate::rootnumber; no numerics involved). Analytic \
                    continuation via modularity (Wiles/BCDT)."
                    .to_string(),
            });
        }

        let nf =
            <Integer as NumericConversion>::to_f64(&self.conductor).expect("conductor fits in f64");
        let cf64 = 2.0 * PI / nf.sqrt();

        // B_0..B_r from P(u) = (ln c + gamma) u + sum_{k>=2} (-1)^k zeta(k) u^k / k.
        let cwp = wp + 64;
        let (gamma_b, zetas_b) = crate::ltaylor::taylor_constants(r, cwp);
        let c_b = self.decay_constant(cwp);
        let mut p = vec![BigFloat::zero_prec(cwp); r as usize + 1];
        if r >= 1 {
            p[1] = RealField::ln(&c_b) + gamma_b.clone();
        }
        for (k, pk) in p.iter_mut().enumerate().skip(2) {
            // -log Gamma(1+u) = gamma u + sum_{k>=2} (-1)^(k+1) zeta(k) u^k / k
            // (note the sign flip against g_kernels' Q, which expands +log Gamma).
            let z = zetas_b[k].clone() / BigFloat::from_integer(&Integer::from(k as i64), cwp);
            *pk = if k % 2 == 0 { -z } else { z };
        }
        let b = crate::ltaylor::exp_series_coeffs(&p, cwp);

        // sum of |B_{r-m}| over the parity class: the tail multiplier.
        let ms: Vec<usize> = (parity..=r).step_by(2).map(|m| m as usize).collect();
        let mut bsum = BigFloat::zero_prec(cwp);
        for &m in &ms {
            bsum = bsum + OrderedRing::abs(&b[r as usize - m]);
        }
        let bsum_f64 = bsum.to_f64().max(1.0);

        let m_terms = self.truncation_point_scaled(digits, bsum_f64);
        if cf64 * ((m_terms + 1) as f64) < 1.0 {
            return Err(
                "internal: c(M+1) < 1, the G_m(x) <= e^-x tail bound is inapplicable".to_string(),
            );
        }

        // gamma / zeta at the precision the largest kernel argument needs.
        let xmax = cf64 * (m_terms as f64 + 2.0);
        let gwp = crate::ltaylor::g_working_precision(xmax, wp);
        let (gamma, zetas) = crate::ltaylor::taylor_constants(r, gwp);

        let a = self.coefficients(m_terms);
        let c = self.decay_constant(wp);
        let mut sums = vec![BigFloat::zero_prec(wp); r as usize + 1];
        let mut abs_sums = vec![BigFloat::zero_prec(wp); r as usize + 1];
        for (n, an) in a.iter().enumerate().skip(1) {
            if an.is_zero() {
                continue;
            }
            let n_bf = BigFloat::from_integer(&Integer::from(n as i64), wp);
            let coeff = BigFloat::from_integer(an, wp) / n_bf.clone();
            let g = crate::ltaylor::g_kernels(&(c.clone() * n_bf), r, wp, &gamma, &zetas)?;
            for &m in &ms {
                let term = coeff.clone() * g[m].clone();
                abs_sums[m] = abs_sums[m].clone() + OrderedRing::abs(&term);
                sums[m] = sums[m].clone() + term;
            }
        }

        let two = BigFloat::from_integer(&Integer::from(2), wp);
        let mut value = BigFloat::zero_prec(wp);
        let mut weighted_abs = BigFloat::zero_prec(wp);
        for &m in &ms {
            let bj = b[r as usize - m].with_precision(wp);
            value = value + bj.clone() * sums[m].clone();
            weighted_abs = weighted_abs + OrderedRing::abs(&bj) * abs_sums[m].clone();
        }
        let value = two.clone() * value;

        let tail = self.tail_bound(m_terms, wp) * bsum.with_precision(wp);
        let ops = BigFloat::from_integer(
            &Integer::from(16 * (m_terms as i64 + 4) * (r as i64 + 2)),
            wp,
        );
        let allowance = two * (weighted_abs + BigFloat::one_prec(wp)) * ops * pow2_neg(wp - 8, wp);

        Ok(LValue {
            value,
            tail_bound: tail,
            rounding_allowance: allowance,
            rounding_note: format!(
                "L^({r})(E,1)/{r}! = 2 sum_(m <= {r}, m = {parity} mod 2) B_({r}-m) \
                 sum_(n=1..{m_terms}) (a_n/n) G_m(2 pi n / sqrt({n})), with \
                 epsilon = {eps} (exact global root number, Tate local data), \
                 exact integer a_n, G_m(x) = (x/m!) int_1^inf e^(-xt)(log t)^m dt \
                 (G_0 = e^-x, G_1 = E_1), and B_j = [u^j] exp((ln c + gamma) u + \
                 sum_(k>=2) (-1)^(k+1) zeta(k) u^k / k) = [u^j] c^u / Gamma(1+u). \
                 BigFloat arithmetic at {wp} bits (each G_m carries its own \
                 cancellation budget). tail_bound = (sum_m |B_(r-m)|) * \
                 2 * [4 e^(-c(M+1))/(1-e^(-c))] from 0 <= G_m(x) <= e^-x for \
                 x >= 1 (log t <= t-1) and |a_n|/n <= 2 (Deligne + d(n) <= \
                 2 sqrt(n)); rounding_allowance covers <= 16(M+4)({r}+2) \
                 operations of relative error 2^-({wp}-8) on the \
                 B-weighted term-magnitude sum (engineering bound, 64 guard \
                 bits past the requested {digits} digits).",
                n = self.conductor,
                eps = self.epsilon,
            ),
        })
    }

    /// Truncation point M for a series whose tail is inflated by `factor`:
    /// smallest M with factor·4e^{−c(M+1)}/(1−e^{−c}) < 10^{−(digits+3)}
    /// (computed in f64; the bound itself is re-evaluated rigorously in
    /// BigFloat for the returned envelope, so this only affects quality).
    fn truncation_point_scaled(&self, digits: usize, factor: f64) -> usize {
        let n =
            <Integer as NumericConversion>::to_f64(&self.conductor).expect("conductor fits in f64");
        let c = 2.0 * PI / n.sqrt();
        let target = (digits as f64 + 3.0) * std::f64::consts::LN_10;
        let m_plus_1 = (target + (4.0 * factor).ln() - (1.0 - (-c).exp()).ln()) / c;
        (m_plus_1.ceil() as usize).max(10) + 2
    }
}

// ---------------------------------------------------------------------------
// The honest analytic-rank lattice
// ---------------------------------------------------------------------------

/// Parity of the analytic rank, from the exact functional-equation sign:
/// ε = +1 → even order of vanishing, ε = −1 → odd.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankParity {
    Even,
    Odd,
}

/// The analytic rank of E, in the honest lattice: certified values carry
/// their certificates, uncertified cases say exactly what is and is not
/// known. NEVER a bare fabricated integer.
#[derive(Debug, Clone)]
pub enum AnalyticRank {
    /// ord_{s=1} L(E,s) = 0, certified: either |L(1)| exceeds its rigorous
    /// error envelope (numeric certificate, `l1` present), or an exact
    /// external nonvanishing certificate was supplied (Manin–Birch winding
    /// projection; `l1` empty).
    ZeroCertified {
        /// The numeric evidence, when the certificate is numeric.
        l1: Option<LValue>,
        /// What certified it.
        evidence: String,
    },
    /// ord_{s=1} L(E,s) = 1, certified modulo the documented rounding
    /// model: L(1) = 0 EXACTLY (ε = −1 functional equation, and/or an
    /// exact winding certificate) and L'(1) is certified nonzero within
    /// its rigorous-tail + documented-rounding envelope.
    OneCertifiedModuloRounding {
        l1_derivative: LValue,
        evidence: String,
    },
    /// ord_{s=1} L(E,s) = `rank` ≥ 2, certified modulo the documented
    /// rounding model: every Taylor coefficient below `rank` is known to
    /// vanish EXACTLY (ε-parity from the functional equation, plus exact
    /// external certificates for the same-parity ones — see
    /// [`LFunction::analytic_rank_with_exact_vanishing`]), and
    /// L^{(rank)}(1)/rank! is certified nonzero inside its rigorous-tail +
    /// documented-rounding envelope.
    ///
    /// The showpiece: 389a1 has ε = +1 (so the order is even) and
    /// L(389a,1) = 0 exactly by the Manin–Birch winding element, hence
    /// ord ≥ 2; and L''(389a,1)/2! = 0.7593165… is certified nonzero. So
    /// ord = 2 exactly — PROVED, not conjectural.
    RankCertifiedModuloRounding {
        rank: u32,
        /// L^{(rank)}(1)/rank!, the certified-nonzero leading coefficient.
        leading_coefficient: LValue,
        evidence: String,
    },
    /// ord_{s=1} L(E,s) ≥ 2 is certified (an exact L(1) = 0 certificate
    /// combined with even parity), but the exact order is NOT resolved.
    AtLeastTwoUnresolved {
        /// Parity of the order of vanishing (from the exact ε).
        parity: RankParity,
        /// Certified lower bound on ord_{s=1} L(E,s).
        known_vanishing: u32,
        evidence: String,
    },
    /// ord_{s=1} L(E,s) ≥ `known_vanishing` ≥ 3 is certified (exact
    /// vanishing certificates combined with the exact ε-parity), but the
    /// exact order is NOT resolved: the candidate leading coefficient was
    /// not certified nonzero, and numerics can never certify it zero.
    /// (`AtLeastTwoUnresolved` is the `known_vanishing == 2` case, kept
    /// separate for API stability.)
    AtLeastNUnresolved {
        parity: RankParity,
        known_vanishing: u32,
        evidence: String,
    },
    /// Nothing certified either way; the reason records what is known.
    Unresolved { reason: String },
}

impl AnalyticRank {
    /// The certified analytic rank, when there is one; `None` for the
    /// unresolved variants. With the general Taylor machinery this is no
    /// longer capped at 1.
    pub fn certified_value(&self) -> Option<u32> {
        match self {
            AnalyticRank::ZeroCertified { .. } => Some(0),
            AnalyticRank::OneCertifiedModuloRounding { .. } => Some(1),
            AnalyticRank::RankCertifiedModuloRounding { rank, .. } => Some(*rank),
            _ => None,
        }
    }
}

impl fmt::Display for AnalyticRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalyticRank::ZeroCertified { .. } => write!(f, "0 (certified: L(E,1) != 0)"),
            AnalyticRank::OneCertifiedModuloRounding { .. } => write!(
                f,
                "1 (certified modulo documented rounding: L(1) = 0 exactly, L'(1) != 0)"
            ),
            AnalyticRank::RankCertifiedModuloRounding { rank, .. } => write!(
                f,
                "{} (certified modulo documented rounding: L^(k)(1) = 0 exactly for \
                 k < {}, L^({})(1) != 0)",
                rank, rank, rank
            ),
            AnalyticRank::AtLeastTwoUnresolved {
                parity,
                known_vanishing,
                ..
            }
            | AnalyticRank::AtLeastNUnresolved {
                parity,
                known_vanishing,
                ..
            } => write!(
                f,
                ">= {} (unresolved beyond that; parity {:?})",
                known_vanishing, parity
            ),
            AnalyticRank::Unresolved { reason } => write!(f, "unresolved: {}", reason),
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy f64 exploration layer + the wired LFunction interface
// ---------------------------------------------------------------------------

/// Complex number for the (uncertified, legacy) f64 L-function exploration
/// helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexNum {
    pub re: f64,
    pub im: f64,
}

impl ComplexNum {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    pub fn norm(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Add for ComplexNum {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl std::ops::Mul for ComplexNum {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl std::ops::Div for ComplexNum {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        Self {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }
}

impl std::ops::Mul<ComplexNum> for f64 {
    type Output = ComplexNum;

    fn mul(self, other: ComplexNum) -> ComplexNum {
        ComplexNum {
            re: self * other.re,
            im: self * other.im,
        }
    }
}

/// The Hasse-Weil L-function of an elliptic curve.
///
/// The certified entry points are [`LFunction::analytic_rank`],
/// [`LFunction::analytic_rank_with_exact_l1`] and
/// [`LFunction::root_number`], wired to the exact Tate-data root number
/// and the rigorous [`CurveLSeries`] numerics. The f64 helpers
/// ([`LFunction::evaluate`] etc.) are uncertified legacy exploration
/// tools (see the module docs).
pub struct LFunction {
    curve: EllipticCurve,
    conductor: Integer,
}

impl LFunction {
    /// Create a new L-function for the given curve
    pub fn new(curve: EllipticCurve) -> Self {
        let conductor = curve
            .conductor
            .clone()
            .unwrap_or_else(|| Self::compute_conductor(&curve));

        Self { curve, conductor }
    }

    /// The conductor of the curve, N = prod p^{f_p}, with every local
    /// exponent computed by Tate's algorithm (see `crate::tate`). This
    /// replaces the old squarefree "product of bad primes" semistable
    /// approximation, which is kept below as
    /// [`Self::compute_conductor_semistable_approx`] for callers that
    /// explicitly want the cheap approximation.
    ///
    /// Cost note: this factors the discriminant (trial division), which is
    /// fine for moderate discriminants but can be slow when the
    /// discriminant has large prime factors.
    pub(crate) fn compute_conductor(curve: &EllipticCurve) -> Integer {
        curve.compute_conductor()
    }

    /// DOCUMENTED FALLBACK (approximation, not Tate's algorithm):
    /// approximate the conductor of the curve as the product of its bad
    /// primes (checked only for p in 2..=31), each raised to the first
    /// power.
    ///
    /// The product-of-bad-primes value equals the true conductor only when
    /// the curve has multiplicative (semistable) reduction at every bad
    /// prime and the given model is minimal; it silently undercounts for
    /// curves with additive reduction or wild ramification at 2 or 3
    /// (where the true exponent can exceed 1), and overcounts at primes
    /// where a non-minimal model hides good reduction. Prefer
    /// [`Self::compute_conductor`] (exact, via Tate's algorithm); this
    /// remains only as a cheap factoring-free approximation for semistable
    /// small-bad-prime curves.
    #[allow(dead_code)]
    pub(crate) fn compute_conductor_semistable_approx(curve: &EllipticCurve) -> Integer {
        let mut conductor = Integer::one();

        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            let p_big = Integer::from(p);
            if curve.is_bad_prime(&p_big) {
                conductor = conductor * p_big;
            }
        }

        if conductor.is_one() {
            // No bad prime <= 31 was found. Every elliptic curve over Q has
            // bad reduction somewhere, so this means either the curve's bad
            // primes lie outside the scanned range, or the semistable
            // assumption above does not hold. This approximation refuses to
            // guess in that case; use `compute_conductor` (Tate) instead.
            unimplemented!(
                "the bad-prime-product approximation cannot handle curves with no bad prime \
                 <= 31; use the exact Tate-based compute_conductor instead"
            );
        }

        conductor
    }

    /// Compute the Euler factor at a prime p (legacy f64 helper; the a_p
    /// are exact, the arithmetic is not).
    pub fn euler_factor(&self, p: &Integer, s: ComplexNum) -> ComplexNum {
        if self.curve.is_bad_prime(p) {
            self.bad_euler_factor(p, s)
        } else {
            self.good_euler_factor(p, s)
        }
    }

    /// Compute Euler factor at a good prime (legacy f64 helper)
    fn good_euler_factor(&self, p: &Integer, s: ComplexNum) -> ComplexNum {
        let a_p = self.curve.compute_a_p(p);
        let p_f = p.to_f64().unwrap_or(2.0);

        // L_p(s) = 1 / (1 - a_p p^{-s} + p^{1-2s})
        let p_to_s =
            p_f.powf(s.re) * ComplexNum::new((s.im * p_f.ln()).cos(), (s.im * p_f.ln()).sin());

        let p_to_1_minus_2s = p_f.powf(1.0 - 2.0 * s.re)
            * ComplexNum::new(
                (-(1.0 - 2.0 * s.re) * p_f.ln() * s.im).cos(),
                (-(1.0 - 2.0 * s.re) * p_f.ln() * s.im).sin(),
            );

        let numerator = ComplexNum::real(1.0);
        let denominator =
            ComplexNum::real(1.0) + ComplexNum::real(-a_p as f64) / p_to_s + p_to_1_minus_2s;

        numerator / denominator
    }

    /// Euler factor at a bad prime (legacy f64 helper): 1/(1 − a_p p^{−s})
    /// with the exact a_p ∈ {0, ±1} from the reduction type (a_p = 0 makes
    /// the additive factor 1, as it should be).
    fn bad_euler_factor(&self, p: &Integer, s: ComplexNum) -> ComplexNum {
        let a_p = self.curve.compute_a_p(p);
        let p_f = p.to_f64().unwrap_or(2.0);
        let p_to_minus_s =
            p_f.powf(-s.re) * ComplexNum::new((-s.im * p_f.ln()).cos(), (-s.im * p_f.ln()).sin());
        ComplexNum::real(1.0)
            / (ComplexNum::real(1.0) + ComplexNum::real(-(a_p as f64)) * p_to_minus_s)
    }

    /// Evaluate the raw Dirichlet partial sum Σ_{n≤max_terms} a_n n^{−s}
    /// (legacy f64 helper; the coefficients are exact via
    /// [`l_series_coefficients`], but a truncated Dirichlet series only
    /// approximates L(E,s) for Re(s) > 3/2 — it is NOT an analytic
    /// continuation and certifies nothing near s = 1).
    pub fn evaluate(&self, s: ComplexNum, max_terms: usize) -> ComplexNum {
        let coeffs = l_series_coefficients(&self.curve, max_terms);
        let mut sum = ComplexNum::real(0.0);

        for (n, an) in coeffs.iter().enumerate().skip(1) {
            if an.is_zero() {
                continue;
            }
            let a_n = <Integer as NumericConversion>::to_f64(an).unwrap_or(0.0);
            let n_f = n as f64;
            let log_n = n_f.ln();
            let n_to_minus_s = ComplexNum::new(
                (-s.re * log_n).exp() * (-s.im * log_n).cos(),
                (-s.re * log_n).exp() * (-s.im * log_n).sin(),
            );
            sum = sum + ComplexNum::real(a_n) * n_to_minus_s;
        }

        sum
    }

    /// Compute the completed L-function Λ(s) (legacy f64 helper; see
    /// [`Self::evaluate`] for why this is uncertified).
    pub fn complete_l_function(&self, s: ComplexNum) -> ComplexNum {
        // Λ(s) = N^{s/2} * (2π)^{-s} * Γ(s) * L(s)
        let n = self.conductor.to_f64().unwrap_or(1.0);

        let gamma_factor = self.gamma_factor(s);
        let l_value = self.evaluate(s, 1000);

        let n_to_s_half = n.powf(s.re / 2.0)
            * ComplexNum::new(((s.im / 2.0) * n.ln()).cos(), ((s.im / 2.0) * n.ln()).sin());

        let two_pi_to_minus_s = (2.0 * PI).powf(-s.re)
            * ComplexNum::new(
                (-s.im * (2.0 * PI).ln()).cos(),
                (-s.im * (2.0 * PI).ln()).sin(),
            );

        n_to_s_half * two_pi_to_minus_s * gamma_factor * l_value
    }

    /// Compute Γ(s) (simplified for real s; legacy f64 helper)
    fn gamma_factor(&self, s: ComplexNum) -> ComplexNum {
        ComplexNum::real(self.gamma(s.re))
    }

    /// Real gamma function (Stirling approximation; legacy f64 helper)
    fn gamma(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }
        if x == 1.0 || x == 2.0 {
            return 1.0;
        }
        if x < 1.0 {
            return self.gamma(x + 1.0) / x;
        }

        // Stirling's approximation
        let two_pi = 2.0 * PI;
        (two_pi / x).sqrt() * (x / std::f64::consts::E).powf(x)
    }

    /// The analytic rank in the honest lattice, using only this crate's
    /// own machinery (Tate-data root number + certified numerics):
    ///
    /// * ε = +1 and L(1) certified nonzero → [`AnalyticRank::ZeroCertified`];
    /// * ε = −1 (so L(1) = 0 EXACTLY) and L'(1) certified nonzero →
    ///   [`AnalyticRank::OneCertifiedModuloRounding`];
    /// * anything else → [`AnalyticRank::Unresolved`] with the reason
    ///   (a value too small to certify nonzero at this precision —
    ///   numerics can NEVER certify a zero; the root number itself is now
    ///   complete at every prime).
    ///
    /// `digits` controls the working precision of the numeric layer.
    pub fn analytic_rank(&self, digits: usize) -> AnalyticRank {
        self.analytic_rank_with_exact_l1(digits, None)
    }

    /// The analytic rank, optionally consuming an EXACT external
    /// certificate about L(E,1).
    ///
    /// `exact_l1_is_zero`:
    /// * `None` — no external knowledge; same as [`Self::analytic_rank`].
    /// * `Some(false)` — L(E,1) ≠ 0 is known EXACTLY. The only accepted
    ///   provenance is an exact statement (e.g. a nonzero Manin–Birch
    ///   winding projection of the attached newform, as computed by
    ///   `rustmath-modular::ModularSymbolsGamma0::l1_vanishes`); do NOT
    ///   pass numeric guesses. Yields `ZeroCertified` with no numeric
    ///   evidence needed.
    /// * `Some(true)` — L(E,1) = 0 is known EXACTLY (same provenance
    ///   requirement). With ε = +1 the order of vanishing is even and ≥ 1,
    ///   hence ≥ 2: `AtLeastTwoUnresolved`. With ε = −1 the derivative
    ///   path decides between `OneCertifiedModuloRounding` and
    ///   `Unresolved`.
    ///
    /// # Panics
    ///
    /// Panics if the external certificate contradicts the exact root
    /// number (`Some(false)` with ε = −1): both are exact statements, so a
    /// contradiction is a bug in one of the two exact pipelines, never a
    /// rounding issue.
    pub fn analytic_rank_with_exact_l1(
        &self,
        digits: usize,
        exact_l1_is_zero: Option<bool>,
    ) -> AnalyticRank {
        if let Some(false) = exact_l1_is_zero {
            if global_root_number(&self.curve) == Ok(-1) {
                panic!(
                    "contradictory exact certificates: external L(1) != 0 vs epsilon = -1 \
                     (which forces L(1) = 0); bug in an exact pipeline"
                );
            }
            return AnalyticRank::ZeroCertified {
                l1: None,
                evidence: "external exact certificate: L(E,1) != 0 (e.g. nonzero \
                    Manin-Birch winding projection); no numerics needed"
                    .to_string(),
            };
        }
        match exact_l1_is_zero {
            Some(true) => self.analytic_rank_with_exact_vanishing(
                digits,
                1,
                "external exact certificate: L(E,1) = 0 (e.g. a vanishing Manin-Birch \
                 winding projection of the attached newform)",
            ),
            _ => self.analytic_rank_with_exact_vanishing(digits, 0, ""),
        }
    }

    /// The analytic rank, given EXACT (non-numeric) knowledge that
    /// L^{(k)}(E,1) = 0 for every k < `exact_vanishing_below`.
    ///
    /// # The decision procedure
    ///
    /// Λ(s) = c^{−s}Γ(s)L(E,s) has the same order of vanishing at s = 1 as L
    /// (Γ(1) = 1 ≠ 0), and Λ(1+u) = ε·Λ(1−u), so the Taylor coefficients Λ_m
    /// vanish EXACTLY for every m of the wrong parity for ε. Because
    /// L(1+u) = c·exp(P(u))·Λ(1+u) with a unit leading factor,
    /// "L^{(k)}(1) = 0 for all k < v" is equivalent to "Λ_m = 0 for all
    /// m < v". Hence:
    ///
    /// * the candidate order is r = the least integer ≥ v whose parity
    ///   matches ε (even for ε = +1, odd for ε = −1), and ord ≥ r is then an
    ///   EXACT statement;
    /// * if L^{(r)}(1)/r! ([`CurveLSeries::l_derivative`]) is certified
    ///   nonzero, then ord = r exactly — [`AnalyticRank::ZeroCertified`]
    ///   (r = 0), [`AnalyticRank::OneCertifiedModuloRounding`] (r = 1), or
    ///   [`AnalyticRank::RankCertifiedModuloRounding`] (r ≥ 2);
    /// * otherwise nothing more is decided (numerics can never certify a
    ///   zero): `AtLeastTwoUnresolved` / `AtLeastNUnresolved` when r ≥ 2 is
    ///   itself certified, plain `Unresolved` when r ≤ 1.
    ///
    /// # The provenance requirement (read this)
    ///
    /// `exact_vanishing_below` must be backed by an EXACT algebraic
    /// certificate, never by "the numeric value looked small". The only such
    /// source wired into this workspace is the Manin–Birch winding element
    /// (`rustmath-modular::ModularSymbolsGamma0::l1_vanishes`, exact rational
    /// linear algebra), and it only certifies L(E,1) = 0 — i.e. v = 1.
    ///
    /// **There is no exact certificate for L'(E,1) = 0 anywhere in this
    /// workspace.** So for a rank-3 curve such as 5077a1 (ε = −1, hence
    /// L(1) = 0 by parity) the candidate order from v = 0 is r = 1; L'(1) is
    /// truly zero and therefore cannot be certified nonzero; and the honest
    /// answer is `Unresolved`. This machinery CANNOT certify analytic rank 3
    /// for 5077a1 and does not pretend to. Rank 2 (389a1) is reachable
    /// precisely because the one coefficient below it that parity does not
    /// kill — L(1) itself — IS exactly certifiable.
    ///
    /// `provenance` is copied into the evidence string.
    ///
    /// # Panics
    ///
    /// Panics if `provenance` is empty while `exact_vanishing_below ≥ 1`: an
    /// unattributed exact certificate is exactly what this crate refuses to
    /// launder.
    pub fn analytic_rank_with_exact_vanishing(
        &self,
        digits: usize,
        exact_vanishing_below: u32,
        provenance: &str,
    ) -> AnalyticRank {
        assert!(
            exact_vanishing_below == 0 || !provenance.is_empty(),
            "analytic_rank_with_exact_vanishing: an exact vanishing claim needs a stated \
             provenance (an EXACT algebraic certificate, never a small numeric value)"
        );
        let eps = match global_root_number(&self.curve) {
            Ok(e) => e,
            Err(reason) => {
                return AnalyticRank::Unresolved {
                    reason: format!(
                        "functional-equation sign unresolved, so neither the L(1) series \
                         nor the exact epsilon = -1 vanishing applies: {}",
                        reason
                    ),
                }
            }
        };
        let parity = if eps == 1 {
            RankParity::Even
        } else {
            RankParity::Odd
        };
        let p: u32 = if eps == 1 { 0 } else { 1 };
        let r = if exact_vanishing_below <= p {
            p
        } else if (exact_vanishing_below - p) % 2 == 0 {
            exact_vanishing_below
        } else {
            exact_vanishing_below + 1
        };

        let ls = match CurveLSeries::new(&self.curve) {
            Ok(ls) => ls,
            Err(reason) => return AnalyticRank::Unresolved { reason },
        };
        let lv = match ls.l_derivative(r, digits) {
            Ok(lv) => lv,
            Err(reason) => return AnalyticRank::Unresolved { reason },
        };

        let vanishing_evidence = || -> String {
            let mut s = format!(
                "L^(k)(E,1) = 0 EXACTLY for every k < {r}: the order of vanishing has the \
                 parity of epsilon = {eps} (functional equation Lambda(1+u) = epsilon \
                 Lambda(1-u), epsilon exact from Tate local data)"
            );
            if exact_vanishing_below >= 1 {
                s.push_str(&format!(
                    ", and the same-parity coefficients below {r} vanish by [{provenance}]"
                ));
            }
            s
        };

        if lv.certified_nonzero() {
            let evidence = format!(
                "{}; |L^({r})(1)/{r}!| = {} exceeds its certified error budget {} \
                 ({digits} requested digits)",
                vanishing_evidence(),
                lv.value.to_decimal_string(12),
                lv.error_budget().to_decimal_string(6),
            );
            return match r {
                0 => AnalyticRank::ZeroCertified {
                    l1: Some(lv),
                    evidence,
                },
                1 => AnalyticRank::OneCertifiedModuloRounding {
                    l1_derivative: lv,
                    evidence,
                },
                _ => AnalyticRank::RankCertifiedModuloRounding {
                    rank: r,
                    leading_coefficient: lv,
                    evidence,
                },
            };
        }

        match r {
            0 => AnalyticRank::Unresolved {
                reason: format!(
                    "epsilon = +1 and L(1) = {} is NOT certified nonzero at {} digits \
                     (budget {}); numerics can never certify a zero, so the order is \
                     0 (tiny value) or >= 2 (even parity) — unresolved",
                    lv.value.to_decimal_string(12),
                    digits,
                    lv.error_budget().to_decimal_string(6)
                ),
            },
            1 => AnalyticRank::Unresolved {
                reason: format!(
                    "L(1) = 0 exactly (epsilon = -1) but L'(1) = {} is NOT \
                     certified nonzero at {} digits (budget {}); order is odd and >= 1, \
                     could be 1 or >= 3 — unresolved (no exact certificate for \
                     L'(1) = 0 exists in this workspace, so a rank >= 3 curve such as \
                     5077a1 CANNOT be resolved here)",
                    lv.value.to_decimal_string(12),
                    digits,
                    lv.error_budget().to_decimal_string(6)
                ),
            },
            _ => {
                let evidence = format!(
                    "{}; but L^({r})(1)/{r}! = {} is NOT certified nonzero at {digits} \
                     digits (budget {}); numerics can never certify a zero, so the order \
                     is {r} (tiny leading coefficient) or >= {r2} — unresolved",
                    vanishing_evidence(),
                    lv.value.to_decimal_string(12),
                    lv.error_budget().to_decimal_string(6),
                    r2 = r + 2,
                );
                if r == 2 {
                    AnalyticRank::AtLeastTwoUnresolved {
                        parity,
                        known_vanishing: 2,
                        evidence,
                    }
                } else {
                    AnalyticRank::AtLeastNUnresolved {
                        parity,
                        known_vanishing: r,
                        evidence,
                    }
                }
            }
        }
    }

    /// Compute special values of the L-function (legacy f64 helper; see
    /// [`Self::evaluate`]).
    pub fn special_value(&self, s: f64) -> ComplexNum {
        self.evaluate(ComplexNum::real(s), 1000)
    }

    /// The global root number (sign of the functional equation), as the
    /// product of local root numbers from Tate local data — see
    /// [`crate::rootnumber`] for the derivation, citations, and the
    /// independent split-point validation battery. `Err` = honest refusal
    /// (additive reduction at 2 or 3), never a guess.
    pub fn root_number(&self) -> Result<i8, String> {
        global_root_number(&self.curve)
    }

    /// Check functional equation: Λ(s) ≈ w·Λ(2−s) using the EXACT root
    /// number but the legacy f64 Λ evaluation (uncertified; loose
    /// tolerance). `Err` when the root number is unresolved.
    pub fn check_functional_equation(&self, s: f64) -> Result<bool, String> {
        let s_complex = ComplexNum::real(s);
        let two_minus_s = ComplexNum::real(2.0 - s);

        let lambda_s = self.complete_l_function(s_complex);
        let lambda_2_minus_s = self.complete_l_function(two_minus_s);

        let w = self.root_number()? as f64;
        let expected = ComplexNum::real(w) * lambda_2_minus_s;

        let diff = (lambda_s.re - expected.re).abs() + (lambda_s.im - expected.im).abs();
        Ok(diff < 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rank::RankBoundResult;

    fn curve(a1: i64, a2: i64, a3: i64, a4: i64, a6: i64) -> EllipticCurve {
        EllipticCurve::new(
            Integer::from(a1),
            Integer::from(a2),
            Integer::from(a3),
            Integer::from(a4),
            Integer::from(a6),
        )
    }

    /// |a - decimal| < 10^-k.
    fn close_to(a: &BigFloat, decimal: &str, k: usize) -> bool {
        let prec = RealField::precision(a).max(256);
        let b = BigFloat::from_decimal_str(decimal, prec).unwrap();
        let tol_str = format!("0.{}1", "0".repeat(k - 1));
        let tol = BigFloat::from_decimal_str(&tol_str, prec).unwrap();
        OrderedRing::abs(&(a.clone() - b)) < tol
    }

    #[test]
    fn test_complex_arithmetic() {
        let z1 = ComplexNum::new(1.0, 2.0);
        let z2 = ComplexNum::new(3.0, 4.0);

        let sum = z1 + z2;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        let prod = z1 * z2;
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_l_function_creation() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(1));

        let l_func = LFunction::new(curve);
        // y² = x³ - x + 1 has conductor 92 = 2²·23 (Tate: type IV with
        // f=2 at 2, I1 at 23; PARI/GP ellglobalred-verified). The old
        // semistable approximation would have given 2·23 = 46 here.
        assert_eq!(l_func.conductor, Integer::from(92));
    }

    #[test]
    fn test_euler_factor() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(1));

        let l_func = LFunction::new(curve);
        let p = Integer::from(5);
        let s = ComplexNum::real(2.0);

        let factor = l_func.euler_factor(&p, s);
        assert!(factor.norm() > 0.0);
    }

    #[test]
    fn test_l_series_evaluation() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(0), Integer::from(-1));

        let l_func = LFunction::new(curve);
        let s = ComplexNum::real(2.0);

        let value = l_func.evaluate(s, 100);
        assert!(value.norm() > 0.0);
    }

    /// The exact-coefficient generator reproduces the point-counted 11a1
    /// table through composites (a_1..a_14; the prime entries were derived
    /// independently in Python, the composites follow by multiplicativity:
    /// a_4 = a_2^2 - 2, a_6 = a_2 a_3, a_9 = a_3^2 - 3, a_10 = a_2 a_5,
    /// a_12 = a_4 a_3, a_14 = a_2 a_7).
    #[test]
    fn test_l_series_coefficients_11a() {
        let e = curve(0, -1, 1, -10, -20);
        let a = l_series_coefficients(&e, 14);
        let expected: [i64; 14] = [1, -2, -1, 2, 1, 2, -2, 0, -2, -2, 1, -2, 4, 4];
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(a[i + 1], Integer::from(v), "a_{} of 11a1", i + 1);
        }
    }

    /// Reduced-precision re-gates of the crate-private gamma / E_1 ports
    /// (full 60/52-digit mpmath gates live in rustmath-modular stage 1;
    /// these keep the copies from rotting).
    #[test]
    fn test_gamma_and_e1_ports() {
        let g = euler_gamma(180);
        assert!(
            close_to(
                &g,
                "0.57721566490153286060651209008240243104215933593992",
                45
            ),
            "gamma to 45 digits; got {}",
            g.to_decimal_string(50)
        );
        let x1 = BigFloat::from_integer(&Integer::from(1), 180);
        let e1 = exp_integral_e1(&x1, 180).unwrap();
        assert!(
            close_to(&e1, "0.2193839343955202736771637754601216490310472934", 40),
            "E_1(1) to 40 digits; got {}",
            e1.to_decimal_string(45)
        );
        assert!(exp_integral_e1(&BigFloat::zero_prec(64), 64).is_err());
    }

    /// NUMERIC GATES, rank 0: L(E,1) certified nonzero and matching the
    /// independently derived mpmath truth (Python: point-counted a_n,
    /// epsilon pinned by split-point independence; all values derived
    /// BEFORE this test) to 25+ digits, with the truth inside the
    /// certified envelope.
    #[test]
    fn test_l1_rank0_gates() {
        let gates: [(&str, [i64; 5], &str); 5] = [
            (
                "11a1",
                [0, -1, 1, -10, -20],
                "0.253841860855910684337758923350909461043898",
            ),
            (
                "14a1",
                [1, 0, 1, 4, -6],
                "0.330223659344480539028261946122834877540452",
            ),
            (
                "15a1",
                [1, 1, 1, -10, -10],
                "0.350150760583150505795045209202429651153492",
            ),
            (
                "37b1",
                [0, 1, 1, -23, -50],
                "0.725681061936152782336205541026396548736760",
            ),
            (
                "49a1",
                [1, -1, 0, -2, -1],
                "0.966655852808405773366538419514906865525071",
            ),
        ];
        for (label, a, truth) in &gates {
            let e = curve(a[0], a[1], a[2], a[3], a[4]);
            let ls = CurveLSeries::new(&e).unwrap();
            assert_eq!(ls.root_number(), 1, "epsilon({}) = +1", label);
            let lv = ls.l1(32);
            assert!(
                close_to(&lv.value, truth, 27),
                "L({},1) to 27 digits; got {}",
                label,
                lv.value.to_decimal_string(35)
            );
            assert!(lv.certified_nonzero(), "L({},1) certified nonzero", label);
            let t = BigFloat::from_decimal_str(truth, RealField::precision(&lv.value).max(256))
                .unwrap();
            assert!(
                OrderedRing::abs(&(lv.value.clone() - t)) < lv.error_budget(),
                "L({},1): independently derived truth inside the certified envelope",
                label
            );
        }
    }

    /// NUMERIC GATE, 37a1: the exact functional-equation zero at s = 1
    /// (epsilon = -1 from Tate data) and L'(37a,1) matching the
    /// independent mpmath value 0.305999773834052301820483683321676474
    /// (E_1 series AND brute log-weighted integral, stage-1 + re-derived
    /// here) to 20 digits, certified nonzero.
    #[test]
    fn test_l1_37a_exact_zero_and_derivative() {
        let e = curve(0, 0, 1, -1, 0);
        let ls = CurveLSeries::new(&e).unwrap();
        assert_eq!(ls.root_number(), -1, "epsilon(37a) = -1");
        let lv = ls.l1(30);
        assert!(Ring::is_zero(&lv.value));
        assert!(Ring::is_zero(&lv.tail_bound));
        assert!(!lv.certified_nonzero(), "a zero is never certified nonzero");
        assert!(lv.rounding_note.contains("EXACTLY"));
        let dv = ls.l1_derivative(26).unwrap();
        assert!(
            close_to(
                &dv.value,
                "0.305999773834052301820483683321676474452638",
                20
            ),
            "L'(37a,1) to 20 digits; got {}",
            dv.value.to_decimal_string(30)
        );
        assert!(dv.certified_nonzero(), "L'(37a,1) certified nonzero");
        let truth = BigFloat::from_decimal_str(
            "0.305999773834052301820483683321676474452638",
            RealField::precision(&dv.value).max(256),
        )
        .unwrap();
        assert!(
            OrderedRing::abs(&(dv.value.clone() - truth)) < dv.error_budget(),
            "L'(37a,1) within its certified envelope"
        );
        // the derivative formula honestly refuses epsilon = +1 curves
        let e11 = curve(0, -1, 1, -10, -20);
        assert!(CurveLSeries::new(&e11).unwrap().l1_derivative(12).is_err());
    }

    /// NUMERIC GATE, 65a1 (rank 1, both bad primes non-split): epsilon =
    /// -1 and L'(65a,1) = 0.50533434230685977745953442460958 (independent
    /// mpmath derivation) certified nonzero.
    #[test]
    fn test_l1_derivative_65a() {
        let e = curve(1, 0, 0, -1, 0);
        let ls = CurveLSeries::new(&e).unwrap();
        assert_eq!(ls.conductor(), &Integer::from(65));
        assert_eq!(ls.root_number(), -1);
        let dv = ls.l1_derivative(24).unwrap();
        assert!(
            close_to(&dv.value, "0.50533434230685977745953442460958", 20),
            "L'(65a,1) to 20 digits; got {}",
            dv.value.to_decimal_string(28)
        );
        assert!(dv.certified_nonzero());
    }

    /// THE ANALYTIC-RANK GATES (task 4): 11a/14a/15a/37b (and 49a) →
    /// ZeroCertified; 37a (and 65a) → OneCertifiedModuloRounding; 389a
    /// (true analytic rank 2) → honestly Unresolved from numerics alone;
    /// and the former wild-additive Unresolved (y² = x³ − x) is now
    /// ZeroCertified via the Kraus/Halberstadt root number.
    #[test]
    fn test_analytic_rank_gates() {
        for (label, a) in [
            ("11a1", [0i64, -1, 1, -10, -20]),
            ("14a1", [1, 0, 1, 4, -6]),
            ("15a1", [1, 1, 1, -10, -10]),
            ("37b1", [0, 1, 1, -23, -50]),
            ("49a1", [1, -1, 0, -2, -1]),
        ] {
            let lf = LFunction::new(curve(a[0], a[1], a[2], a[3], a[4]));
            let r = lf.analytic_rank(28);
            assert!(
                matches!(r, AnalyticRank::ZeroCertified { l1: Some(_), .. }),
                "{}: expected ZeroCertified, got {}",
                label,
                r
            );
            assert_eq!(r.certified_value(), Some(0));
        }
        for (label, a) in [("37a1", [0i64, 0, 1, -1, 0]), ("65a1", [1, 0, 0, -1, 0])] {
            let lf = LFunction::new(curve(a[0], a[1], a[2], a[3], a[4]));
            let r = lf.analytic_rank(24);
            assert!(
                matches!(r, AnalyticRank::OneCertifiedModuloRounding { .. }),
                "{}: expected OneCertifiedModuloRounding, got {}",
                label,
                r
            );
            assert_eq!(r.certified_value(), Some(1));
        }
        // 389a1: epsilon = +1 but L(1) is (truly) ~ 0; numerics must NOT
        // fabricate an answer in either direction.
        let lf = LFunction::new(curve(0, 1, 1, -2, 0));
        let r = lf.analytic_rank(20);
        assert!(
            matches!(r, AnalyticRank::Unresolved { .. }),
            "389a1: expected Unresolved, got {}",
            r
        );
        assert_eq!(r.certified_value(), None);
        if let AnalyticRank::Unresolved { reason } = &r {
            assert!(
                reason.contains("NOT certified nonzero"),
                "reason: {}",
                reason
            );
        }
        // y² = x³ − x (N = 32, additive at 2): MOVED from Unresolved to
        // decided by the Kraus/Halberstadt tables — w₂ = −1, ε = +1, and
        // L(1) = 0.65551438857… ≠ 0 (PARI + independent mpmath series,
        // gated to 25 digits in test_l1_wild_additive_movers): certified
        // analytic rank 0.
        let lf = LFunction::new(curve(0, 0, 0, -1, 0));
        let r = lf.analytic_rank(20);
        assert!(
            matches!(r, AnalyticRank::ZeroCertified { l1: Some(_), .. }),
            "y²=x³−x: expected ZeroCertified now, got {}",
            r
        );
        assert_eq!(r.certified_value(), Some(0));
    }

    /// THE WILD-ADDITIVE MOVERS: curves whose analytic rank was an honest
    /// Unresolved (root number blocked at 2 or 3) and is now CERTIFIED,
    /// with L(E,1) matching the independently derived truths (PARI ellL1
    /// AND a from-scratch mpmath series from point-counted a_n, agreeing
    /// to 40+ digits BEFORE this test) to 25 digits, inside the certified
    /// envelope.
    #[test]
    fn test_l1_wild_additive_movers() {
        let gates: [(&str, [i64; 5], &str); 5] = [
            (
                "x3-x(N=32)",
                [0, 0, 0, -1, 0],
                "0.655514388573029952616209897472779853420689",
            ),
            (
                "x3+1(N=36)",
                [0, 0, 0, 0, 1],
                "0.701091052662727130587509539525147067731511",
            ),
            (
                "27a1",
                [0, 0, 1, 0, -7],
                "0.588879583428483319104563166549479567523956",
            ),
            (
                "x3-1(N=144)",
                [0, 0, 0, 0, -1],
                "1.214325323943790805909970844890465624277517",
            ),
            (
                "20a(N=20)",
                [0, 1, 0, -1, 0],
                "0.470729190326518966580631591507238020566902",
            ),
        ];
        for (label, a, truth) in &gates {
            let e = curve(a[0], a[1], a[2], a[3], a[4]);
            let ls = CurveLSeries::new(&e)
                .unwrap_or_else(|err| panic!("{}: CurveLSeries {}", label, err));
            assert_eq!(ls.root_number(), 1, "epsilon({}) = +1", label);
            let lv = ls.l1(30);
            assert!(
                close_to(&lv.value, truth, 25),
                "L({},1) to 25 digits; got {}",
                label,
                lv.value.to_decimal_string(35)
            );
            assert!(lv.certified_nonzero(), "L({},1) certified nonzero", label);
            let t = BigFloat::from_decimal_str(truth, RealField::precision(&lv.value).max(256))
                .unwrap();
            assert!(
                OrderedRing::abs(&(lv.value.clone() - t)) < lv.error_budget(),
                "L({},1): truth inside the certified envelope",
                label
            );
        }
    }

    /// The exact-certificate entry point: an external Manin–Birch
    /// statement upgrades the lattice without numerics.
    #[test]
    fn test_analytic_rank_with_exact_certificates() {
        // 11a1 + exact "L(1) != 0": ZeroCertified with no numeric leg.
        let lf = LFunction::new(curve(0, -1, 1, -10, -20));
        let r = lf.analytic_rank_with_exact_l1(20, Some(false));
        assert!(matches!(r, AnalyticRank::ZeroCertified { l1: None, .. }));
        // 37a1 + exact "L(1) = 0": corroborates epsilon = -1, still rank 1
        // via the certified derivative.
        let lf = LFunction::new(curve(0, 0, 1, -1, 0));
        let r = lf.analytic_rank_with_exact_l1(22, Some(true));
        assert!(matches!(r, AnalyticRank::OneCertifiedModuloRounding { .. }));
        // 389a1 + exact "L(1) = 0": STRENGTHENED. This used to be the best
        // the lattice could do (>= 2, unresolved beyond), because there was
        // no L''(1). There is now: the exact zero plus even parity gives
        // ord >= 2, and L''(1)/2! = 0.7593165... is certified nonzero, so
        // the order is EXACTLY 2. (The full proof, with the winding
        // certificate actually computed rather than asserted, is
        // test_389a_analytic_rank_two_certified.)
        let lf = LFunction::new(curve(0, 1, 1, -2, 0));
        let r = lf.analytic_rank_with_exact_l1(16, Some(true));
        match &r {
            AnalyticRank::RankCertifiedModuloRounding { rank, .. } => assert_eq!(*rank, 2),
            other => panic!("389a1 with exact zero: expected certified rank 2, got {other}"),
        }
        assert_eq!(r.certified_value(), Some(2));
    }

    /// THE RANK-4 REFUSAL — 234446a1 = [1,-1,0,-79,289] is the smallest
    /// conductor of analytic rank 4 (ε = +1). It exercises the
    /// `AtLeastTwoUnresolved` arm honestly: given the exact L(1) = 0
    /// certificate, even parity forces ord ≥ 2, but L''(1)/2! is a TRUE zero,
    /// so nothing beyond 2 is decided — and the machinery must say exactly
    /// that rather than invent a 2 or a 4.
    ///
    /// It is also a PARI gate at r = 4: L''''(1)/4! = 8.943847395900889…
    ///
    /// PROVENANCE NOTE, read it: the L(234446a,1) = 0 certificate is a
    /// Cremona/LMFDB table fact supplied BY THIS TEST to drive the decision
    /// lattice. It is NOT computed in-crate — the winding element at level
    /// 234446 is far out of reach — and the code never fabricates it: without
    /// the certificate, `analytic_rank` on this curve is honestly
    /// `Unresolved`, which is also asserted below.
    #[test]
    fn test_rank_four_curve_honest_refusal() {
        let e = curve(1, -1, 0, -79, 289);
        let ls = CurveLSeries::new(&e).unwrap();
        assert_eq!(ls.conductor(), &Integer::from(234446));
        assert_eq!(ls.root_number(), 1, "epsilon(234446a) = +1");

        // r = 4 against PARI (lfun(...,1,4)/4!, realprecision 30).
        let lv4 = ls.l_derivative(4, 6).unwrap();
        assert!(
            close_to(&lv4.value, "8.94384739590088904641759168347", 4),
            "L''''(234446a,1)/4!; got {}",
            lv4.value.to_decimal_string(14)
        );
        assert!(lv4.certified_nonzero());

        // r = 2 is a TRUE zero: never certified nonzero.
        let lv2 = ls.l_derivative(2, 6).unwrap();
        assert!(
            !lv2.certified_nonzero(),
            "L''(234446a,1)/2! is a true zero; it must not be 'certified nonzero'"
        );

        let lf = LFunction::new(e);
        // Without an exact certificate: honestly Unresolved (L(1) is a true
        // zero and numerics can never certify a zero).
        assert!(matches!(
            lf.analytic_rank(6),
            AnalyticRank::Unresolved { .. }
        ));

        // With the (externally supplied, clearly labelled) exact certificate:
        // ord >= 2 certified, exact order NOT resolved.
        let r = lf.analytic_rank_with_exact_vanishing(
            6,
            1,
            "L(234446a1, 1) = 0: Cremona/LMFDB table fact, supplied by this test to \
             drive the decision lattice; NOT computed in-crate",
        );
        match &r {
            AnalyticRank::AtLeastTwoUnresolved {
                parity,
                known_vanishing,
                ..
            } => {
                assert_eq!(*parity, RankParity::Even);
                assert_eq!(*known_vanishing, 2);
            }
            other => panic!("234446a1: expected AtLeastTwoUnresolved, got {other}"),
        }
        assert_eq!(r.certified_value(), None);
    }

    /// Contradictory exact certificates must panic (both sides are exact;
    /// a mismatch is a bug, not a rounding issue): 37a has epsilon = -1,
    /// so an external "L(1) != 0" is inconsistent.
    #[test]
    #[should_panic(expected = "contradictory exact certificates")]
    fn test_exact_certificate_contradiction_panics() {
        let lf = LFunction::new(curve(0, 0, 1, -1, 0));
        let _ = lf.analytic_rank_with_exact_l1(12, Some(false));
    }

    /// CONSISTENCY WITH 2-DESCENT (task 4): on every battery curve where
    /// the descent interval collapses, the certified analytic rank (when
    /// there is one) equals the algebraic rank, and the exact root number
    /// matches (-1)^rank; and NO curve with descent upper bound 0 gets a
    /// certified analytic rank >= 1.
    #[test]
    fn test_descent_analytic_consistency() {
        // (label, model, descent-certified rank if the interval collapses)
        let cases: [(&str, [i64; 5], Option<u32>); 6] = [
            ("y2=x3-x", [0, 0, 0, -1, 0], Some(0)), // N=32, wild at 2
            ("y2=x3+1", [0, 0, 0, 0, 1], Some(0)),  // N=36, wild at 2,3
            ("14a1", [1, 0, 1, 4, -6], Some(0)),
            ("15a1", [1, 1, 1, -10, -10], Some(0)),
            ("49a1", [1, -1, 0, -2, -1], Some(0)),
            ("65a1", [1, 0, 0, -1, 0], Some(1)),
        ];
        for (label, a, expect_rank) in &cases {
            let e = curve(a[0], a[1], a[2], a[3], a[4]);
            let alg = match e.rank_bounds() {
                RankBoundResult::Bounds(b) if b.lower == b.upper => Some(b.lower),
                _ => None,
            };
            assert_eq!(alg, *expect_rank, "{}: descent interval", label);
            let lf = LFunction::new(e.clone());
            let an = lf.analytic_rank(24);
            if let (Some(r), Some(v)) = (alg, an.certified_value()) {
                assert_eq!(v, r, "{}: certified analytic rank vs descent", label);
            }
            // descent-certified rank 0 must never coexist with a certified
            // analytic rank >= 1
            if alg == Some(0) {
                assert!(
                    an.certified_value().is_none_or(|v| v == 0),
                    "{}: descent rank 0 but analytic {}",
                    label,
                    an
                );
            }
            // parity: where both the root number and the descent rank are
            // decided, (-1)^rank == epsilon (each instance independently
            // verified numerically in Python before this test).
            if let (Ok(eps), Some(r)) = (e.root_number(), alg) {
                let parity = if r % 2 == 0 { 1i8 } else { -1i8 };
                assert_eq!(eps, parity, "{}: rank parity vs root number", label);
            }
        }
        // 37a1: descent is honestly Unresolved (no rational 2-torsion),
        // analytic rank 1 is certified — no conflict by construction.
        let e = curve(0, 0, 1, -1, 0);
        assert!(matches!(
            e.rank_bounds(),
            RankBoundResult::Unresolved { .. }
        ));
    }

    /// Twice re-pointed facade test: y² = x³ − x went from an
    /// unimplemented!() facade, to an honest Unresolved (wild root number
    /// blocked), to a CERTIFIED analytic rank 0 now that the
    /// Kraus/Halberstadt tables decide w₂ (L(1) = 0.65551438857… ≠ 0,
    /// independently gated in test_l1_wild_additive_movers).
    #[test]
    fn test_analytic_rank() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        let l_func = LFunction::new(curve);
        let rank = l_func.analytic_rank(20);
        assert!(matches!(rank, AnalyticRank::ZeroCertified { .. }));
        assert_eq!(rank.certified_value(), Some(0));
    }

    /// The general Taylor machinery must REPRODUCE the two PARI-validated
    /// special cases it generalises — they are independent derivations
    /// (l1: the plain e^{-x} kernel; l1_derivative: the E_1 kernel; both
    /// with no Gamma-factor correction at all) and the general one goes
    /// through the B_j = [u^j] c^u/Gamma(1+u) expansion and the G_m family.
    /// Agreement to the full requested precision is a strong cross-check of
    /// both.
    #[test]
    fn test_l_derivative_reproduces_l1_and_l1_derivative() {
        for (label, a) in [
            ("11a1", [0i64, -1, 1, -10, -20]),
            ("14a1", [1, 0, 1, 4, -6]),
            ("15a1", [1, 1, 1, -10, -10]),
            ("37a1", [0, 0, 1, -1, 0]),
            ("65a1", [1, 0, 0, -1, 0]),
        ] {
            let e = curve(a[0], a[1], a[2], a[3], a[4]);
            let ls = CurveLSeries::new(&e).unwrap();
            let old = ls.l1(24);
            let new = ls.l_derivative(0, 24).unwrap();
            let budget = old.error_budget() + new.error_budget();
            assert!(
                OrderedRing::abs(&(old.value.clone() - new.value.clone())) <= budget,
                "{}: l1 = {} vs l_derivative(0) = {}",
                label,
                old.value.to_decimal_string(30),
                new.value.to_decimal_string(30)
            );
            if ls.root_number() == -1 {
                let old = ls.l1_derivative(24).unwrap();
                let new = ls.l_derivative(1, 24).unwrap();
                let budget = old.error_budget() + new.error_budget();
                assert!(
                    OrderedRing::abs(&(old.value.clone() - new.value.clone())) <= budget,
                    "{}: l1_derivative = {} vs l_derivative(1) = {}",
                    label,
                    old.value.to_decimal_string(30),
                    new.value.to_decimal_string(30)
                );
            }
        }
    }

    /// THE PARI CROSS-CHECK TABLE for L^(r)(E,1)/r!, r = 0..3, over curves of
    /// analytic rank 0, 1, 2 and 3. Every expected value was produced by
    /// PARI/GP (`lfun(lfuncreate(ellinit(m)), 1, r)/r!`, realprecision 45)
    /// BEFORE this test was written; none was read out of the code under
    /// test. Truths are also required to lie inside the certified envelope.
    ///
    /// Note what this table shows about parity: 11a/14a/15a have ε = +1 and
    /// nonzero ODD derivatives (L'(11a,1) = 0.3087085…). Only the order of
    /// vanishing is parity-constrained, not the individual coefficients.
    #[test]
    fn test_l_derivative_pari_table() {
        // (label, model, digits, [L^(r)(1)/r! for r = 0..3], each None when
        // it is a true zero PARI reports as ~1e-61)
        let table: [(&str, [i64; 5], usize, [Option<&str>; 4]); 8] = [
            (
                "11a1",
                [0, -1, 1, -10, -20],
                25,
                [
                    Some("0.253841860855910684337758923350909461043898448"),
                    Some("0.308708533963172285620043118807185401969897135"),
                    Some("0.0113280542179201936918730233038576078627378839"),
                    Some("-0.0367068776204353155007383441280255754574937739"),
                ],
            ),
            (
                "14a1",
                [1, 0, 1, 4, -6],
                25,
                [
                    Some("0.330223659344480539028261946122834877540452341"),
                    Some("0.361781175087022732524319291172809243498589902"),
                    Some("-0.0250570050130642908232144428961488992859162723"),
                    Some("-0.0398795775038863436206425633312843506560127165"),
                ],
            ),
            (
                "15a1",
                [1, 1, 1, -10, -10],
                25,
                [
                    Some("0.350150760583150505795045209202429651153491832"),
                    Some("0.371533637940696127893913187766225954453710322"),
                    Some("-0.0372265714473849082388727875661810327310985936"),
                    Some("-0.0386317102806347212091353094131204337653256611"),
                ],
            ),
            (
                "37a1",
                [0, 0, 1, -1, 0],
                25,
                [
                    None,
                    Some("0.305999773834052301820483683321676474452637775"),
                    Some("0.186547797268161964173817368779507591454082445"),
                    Some("-0.136791463097187666302582216428158570769194708"),
                ],
            ),
            (
                "43a1",
                [0, 1, 1, 0, 0],
                25,
                [
                    None,
                    Some("0.343523974618478230618071163921737442803975861"),
                    Some("0.183611047592843076217012896804098371078584327"),
                    Some("-0.162711973044460744670877003552142554207633283"),
                ],
            ),
            (
                "53a1",
                [1, -1, 1, 0, 0],
                25,
                [
                    None,
                    Some("0.435863824177857162053863132073440756333621248"),
                    Some("0.187398245341680458068415226645847178619386032"),
                    Some("-0.221676918786663789132217152904784937784434811"),
                ],
            ),
            (
                // rank 2: L(1) and L'(1) are TRUE zeros, so no expectation is
                // asserted for them beyond "inside the envelope"; r = 2, 3 are
                // pinned to PARI.
                "389a1",
                [0, 1, 1, -2, 0],
                20,
                [
                    None,
                    None,
                    Some("0.759316500288426770230192607894722019078097516"),
                    Some("-0.430302337583361999290351775060044236190415547"),
                ],
            ),
            (
                "433a1",
                [1, 0, 0, 0, 1],
                18,
                [
                    None,
                    None,
                    Some("0.947020780865814533489726400980751253204348358"),
                    Some("-0.587414387532858546773380277086852900757089980"),
                ],
            ),
        ];
        for (label, m, digits, expect) in &table {
            let e = curve(m[0], m[1], m[2], m[3], m[4]);
            let ls = CurveLSeries::new(&e).unwrap();
            for (r, want) in expect.iter().enumerate() {
                let lv = ls.l_derivative(r as u32, *digits).unwrap();
                match want {
                    Some(truth) => {
                        let prec = RealField::precision(&lv.value).max(320);
                        let t = BigFloat::from_decimal_str(truth, prec).unwrap();
                        assert!(
                            OrderedRing::abs(&(lv.value.clone() - t)) < lv.error_budget(),
                            "{} r={}: PARI {} outside the certified envelope of {} \
                             (budget {})",
                            label,
                            r,
                            truth,
                            lv.value.to_decimal_string(30),
                            lv.error_budget().to_decimal_string(6)
                        );
                        assert!(
                            close_to(&lv.value, truth, digits - 4),
                            "{} r={}: want {}, got {}",
                            label,
                            r,
                            truth,
                            lv.value.to_decimal_string(digits + 6)
                        );
                        assert!(
                            lv.certified_nonzero(),
                            "{} r={}: should be certified nonzero",
                            label,
                            r
                        );
                    }
                    None => {
                        // a TRUE zero: numerics must not certify it nonzero,
                        // and the value must sit inside its own error budget.
                        assert!(
                            !lv.certified_nonzero(),
                            "{} r={}: a true zero was 'certified nonzero' — bug",
                            label,
                            r
                        );
                        assert!(
                            OrderedRing::abs(&lv.value) < lv.error_budget()
                                || Ring::is_zero(&lv.value),
                            "{} r={}: |{}| should be within the budget {}",
                            label,
                            r,
                            lv.value.to_decimal_string(30),
                            lv.error_budget().to_decimal_string(6)
                        );
                    }
                }
            }
        }
    }

    /// 5077a1 (analytic rank 3): L'''(1)/3! = 1.7318499001193006898 (PARI)
    /// is reproduced by the general formula — but see
    /// `test_5077a_rank_three_is_NOT_certifiable` for why that value alone
    /// does NOT certify the rank.
    #[test]
    fn test_l_derivative_5077a_rank3() {
        let e = curve(0, 0, 1, -7, 6);
        let ls = CurveLSeries::new(&e).unwrap();
        assert_eq!(ls.conductor(), &Integer::from(5077));
        assert_eq!(ls.root_number(), -1);
        let lv = ls.l_derivative(3, 12).unwrap();
        assert!(
            close_to(
                &lv.value,
                "1.73184990011930068979197508506015284495439273",
                8
            ),
            "L'''(5077a,1)/3! to 8 digits; got {}",
            lv.value.to_decimal_string(20)
        );
        assert!(lv.certified_nonzero());
        let truth = BigFloat::from_decimal_str(
            "1.73184990011930068979197508506015284495439273",
            RealField::precision(&lv.value).max(320),
        )
        .unwrap();
        assert!(
            OrderedRing::abs(&(lv.value.clone() - truth)) < lv.error_budget(),
            "PARI truth inside the certified envelope"
        );
        // L'(1) is a TRUE zero here; numerics must refuse to certify it.
        let d1 = ls.l_derivative(1, 12).unwrap();
        assert!(!d1.certified_nonzero());
    }

    /// THE SHOWPIECE — 389a1 has analytic rank EXACTLY 2, fully proved:
    ///
    /// * ε(389a) = +1, exactly, from Tate local data (`crate::rootnumber`),
    ///   so ord_{s=1} L is EVEN (functional equation, modularity);
    /// * L(389a,1) = 0 EXACTLY, by the Manin–Birch winding element: the
    ///   winding projection π_f(e) of the attached weight-2 newform vanishes,
    ///   which is exact rational linear algebra in the modular-symbol space
    ///   (`rustmath_modular::…::l1_vanishes`), NOT a small numeric value.
    ///   Hence ord ≥ 1, hence ord ≥ 2 by parity;
    /// * L''(389a,1)/2! = 0.759316500288… is certified NONZERO (|value|
    ///   exceeds tail bound + rounding allowance), so ord ≤ 2.
    ///
    /// ⇒ ord = 2. Nothing here assumes BSD or any other conjecture (beyond
    /// modularity, which is a theorem).
    #[test]
    fn test_389a_analytic_rank_two_certified() {
        use rustmath_modular::modsym::decomposition::{HeckeEigenvalue, SummandHeckeAction};
        use rustmath_modular::modsym::ModularSymbolsGamma0;
        use rustmath_rationals::Rational;

        let e = curve(0, 1, 1, -2, 0);
        let ls = CurveLSeries::new(&e).unwrap();
        assert_eq!(ls.conductor(), &Integer::from(389));
        assert_eq!(ls.root_number(), 1, "epsilon(389a) = +1");

        // The EXACT external certificate: the winding element kills the 389a
        // newform's Hecke summand <=> L(389a,1) = 0. The summand is pinned by
        // its Hecke eigenvalues, which must equal the curve's own exact a_p
        // (Eichler-Shimura) -- a_2 = -2, a_3 = -2, a_5 = -3 for 389a.
        let ms = ModularSymbolsGamma0::new(389);
        let dec = ms
            .cuspidal_hecke_decomposition()
            .expect("cuspidal Hecke decomposition at level 389");
        let want = |p: i64| Rational::from_integer(Integer::from(e.compute_a_p(&Integer::from(p))));
        let mut vanishes = None;
        for w in dec.summands() {
            let matches = [2i64, 3, 5].iter().all(|&p| {
                matches!(
                    ms.hecke_action_on_summand(w, p as u64),
                    Ok(SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(ref a)))
                        if *a == want(p)
                )
            });
            if matches {
                assert!(vanishes.is_none(), "389a's summand must be unique");
                vanishes = Some(ms.l1_vanishes(w).expect("winding projection"));
            }
        }
        let vanishes = vanishes.expect("the 389a newform must appear in the decomposition");
        assert!(
            vanishes,
            "Manin-Birch: the winding projection of the 389a newform must vanish"
        );

        // Now the lattice, fed the EXACT certificate.
        let lf = LFunction::new(e.clone());
        let r = lf.analytic_rank_with_exact_vanishing(
            20,
            1,
            "Manin-Birch winding element: pi_f(e) = 0 in the level-389 modular \
             symbol space (exact rational linear algebra, rustmath-modular)",
        );
        match &r {
            AnalyticRank::RankCertifiedModuloRounding {
                rank,
                leading_coefficient,
                ..
            } => {
                assert_eq!(*rank, 2);
                assert!(leading_coefficient.certified_nonzero());
                assert!(
                    close_to(
                        &leading_coefficient.value,
                        "0.759316500288426770230192607894722019078097516",
                        16
                    ),
                    "L''(389a,1)/2! ; got {}",
                    leading_coefficient.value.to_decimal_string(26)
                );
            }
            other => panic!("389a1: expected certified rank 2, got {other}"),
        }
        assert_eq!(r.certified_value(), Some(2));
        // and the same certificate reached through the older seam
        assert_eq!(
            lf.analytic_rank_with_exact_l1(20, Some(true))
                .certified_value(),
            Some(2)
        );
        // WITHOUT the exact certificate nothing is certified: numerics alone
        // can never establish L(389a,1) = 0.
        assert_eq!(lf.analytic_rank(20).certified_value(), None);
    }

    /// THE HONEST NON-RESULT — 5077a1 has analytic rank 3, and this machinery
    /// CANNOT certify that, because the chain of exact zeros breaks:
    /// ε = −1 gives L(1) = 0 for free (odd order), but the next coefficient
    /// that must vanish, L'(1), is of the RIGHT parity — parity says nothing
    /// about it — and there is no exact certificate for L'(E,1) = 0 anywhere
    /// in this workspace (the winding element only certifies L(1)). Numerics
    /// can never certify the zero. So the answer is `Unresolved`, and it
    /// stays `Unresolved` no matter how many digits are requested.
    #[test]
    #[allow(non_snake_case)]
    fn test_5077a_rank_three_is_NOT_certifiable() {
        let lf = LFunction::new(curve(0, 0, 1, -7, 6));
        for digits in [12, 20] {
            let r = lf.analytic_rank(digits);
            assert!(
                matches!(r, AnalyticRank::Unresolved { .. }),
                "5077a1 at {} digits: expected Unresolved, got {}",
                digits,
                r
            );
            assert_eq!(r.certified_value(), None);
            if let AnalyticRank::Unresolved { reason } = &r {
                assert!(
                    reason.contains("CANNOT be resolved here"),
                    "reason: {}",
                    reason
                );
            }
        }
    }

    /// The AtLeastN arm: fed an exact "L(1) = L'(1) = L''(1) = 0" claim, a
    /// curve of true analytic rank 3 with ε = −1 lands on the certified
    /// lower bound 3 and, if the leading coefficient is certified nonzero,
    /// on rank 3. (The certificate is supplied here to exercise the DECISION
    /// lattice; the crate has no exact source for it, which is exactly what
    /// `test_5077a_rank_three_is_NOT_certifiable` records.)
    #[test]
    fn test_rank_three_lattice_given_an_exact_certificate() {
        let lf = LFunction::new(curve(0, 0, 1, -7, 6));
        let r = lf.analytic_rank_with_exact_vanishing(
            12,
            3,
            "HYPOTHETICAL certificate, supplied by this test only to exercise the \
             decision lattice; no such exact source exists in the workspace",
        );
        match &r {
            AnalyticRank::RankCertifiedModuloRounding { rank, .. } => assert_eq!(*rank, 3),
            other => panic!("expected certified rank 3 from the lattice, got {other}"),
        }
        assert_eq!(r.certified_value(), Some(3));
    }

    /// An unattributed exact-vanishing claim is refused.
    #[test]
    #[should_panic(expected = "needs a stated provenance")]
    fn test_exact_vanishing_needs_provenance() {
        let lf = LFunction::new(curve(0, 1, 1, -2, 0));
        let _ = lf.analytic_rank_with_exact_vanishing(12, 1, "");
    }

    #[test]
    fn test_root_number_wired() {
        // 11a1: +1; 37a1: -1; y²=x³-x: +1 now that the wild additive
        // tables are in (w₂ = −1 · w_∞ = −1; PARI-verified).
        let lf = LFunction::new(curve(0, -1, 1, -10, -20));
        assert_eq!(lf.root_number(), Ok(1));
        let lf = LFunction::new(curve(0, 0, 1, -1, 0));
        assert_eq!(lf.root_number(), Ok(-1));
        let lf = LFunction::new(curve(0, 0, 0, -1, 0));
        assert_eq!(lf.root_number(), Ok(1));
    }
}
