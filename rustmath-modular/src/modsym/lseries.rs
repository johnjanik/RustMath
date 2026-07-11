//! # Numeric L(f, 1) and L'(f, 1) for weight-2 rational newforms
//!
//! Exponentially convergent series for the central L-value and its first
//! derivative, over [`BigFloat`], with RIGOROUS tail bounds and a documented
//! rounding allowance.  The honesty lattice implemented here:
//!
//! * a numeric value can certify NONZERO ([`LValue::certified_nonzero`]:
//!   magnitude exceeds tail bound plus the documented rounding allowance);
//! * a numeric value can NEVER certify zero.  Certified zeros come from the
//!   two EXACT sources only: the winding projection
//!   ([`super::winding`], Manin--Birch) and the exact functional-equation
//!   sign (epsilon = -1 forces L(f, 1) = 0; epsilon is computed from the
//!   exact rational Atkin-Lehner matrix, not numerically).
//!
//! ## Derivation of the formulas (verified against brute mpmath integrals
//! and a split-point-independence test BEFORE implementation)
//!
//! Let f = sum a_n q^n be a weight-2 newform of level N with Fricke
//! eigenvalue f|_2 W_N = w_N f, and epsilon = -w_N.  The completed
//! L-function Lambda(s) = N^{s/2} (2 pi)^{-s} Gamma(s) L(f, s)
//! = N^{s/2} int_0^oo f(iy) y^{s-1} dy.  Splitting the integral at
//! y_0 = 1/sqrt(N) and substituting y -> 1/(N y) in the lower half (which
//! by the Fricke relation f(i/(N y)) = -w_N N y^2 f(iy) maps it to the
//! upper half) gives, EXACTLY for every s:
//!
//!   Lambda(s) = int_{1/sqrt N}^oo f(iy) [ N^{s/2} y^{s-1}
//!                 + epsilon N^{1 - s/2} y^{1-s} ] dy .
//!
//! * At s = 1 with q-expansion integrated termwise
//!   (int_{y0}^oo e^{-2 pi n y} dy = e^{-2 pi n y0}/(2 pi n)):
//!
//!     L(f, 1) = (1 + epsilon) * sum_{n >= 1} (a_n / n) e^{-2 pi n / sqrt N}.
//!
//!   For epsilon = -1 this is 0 = 0 (the exact zero); for epsilon = +1 the
//!   series converges geometrically with ratio e^{-2 pi / sqrt N}.
//!
//! * Differentiating the split representation at s = 1 when epsilon = -1
//!   (so Lambda(1) = 0): d/ds [N^{s/2} y^{s-1} - N^{1-s/2} y^{1-s}] at
//!   s = 1 equals 2 sqrt(N) ln(sqrt(N) y), hence
//!   Lambda'(1) = 2 sqrt(N) int_{1/sqrt N}^oo f(iy) ln(sqrt(N) y) dy, and
//!   termwise (u = sqrt(N) y, then int_1^oo e^{-au} ln u du = E_1(a)/a by
//!   parts, E_1(x) = int_x^oo e^{-t} dt / t):
//!
//!     L'(f, 1) = 2 * sum_{n >= 1} (a_n / n) E_1(2 pi n / sqrt N)
//!
//!   using Lambda'(1) = (sqrt(N)/(2 pi)) L'(f, 1) (all other product-rule
//!   terms carry the factor L(f, 1) = 0).
//!
//! ## Rigorous tails
//!
//! Coefficient bound: |a_n| <= d(n) sqrt(n) for a weight-2 newform
//! (Deligne, *La conjecture de Weil I*, Publ. IHES 43 (1974): the
//! Ramanujan-Petersson bound |a_p| <= 2 sqrt(p) at good primes; at bad
//! primes |a_p| in {0, 1} -- both facts are ALSO asserted at runtime on
//! every prime eigenvalue consumed, so the tail bound does not rest on the
//! citation alone for the primes actually used).  Combined with the
//! elementary d(n) <= 2 sqrt(n) (divisors pair d <-> n/d with
//! min(d, n/d) <= sqrt(n)), this gives |a_n|/n <= 2, so with
//! c = 2 pi / sqrt(N):
//!
//!   |L-tail after M terms|  <= 4 sum_{n > M} e^{-cn}
//!                            = 4 e^{-c(M+1)} / (1 - e^{-c}),
//!   |L'-tail after M terms| <= 4 sum_{n > M} E_1(cn)
//!                           <= 4 e^{-c(M+1)} / (1 - e^{-c})
//!
//! (for the second: E_1(x) <= e^{-x}/x <= e^{-x} for x >= 1, enforced by
//! requiring c(M+1) >= 1).  The returned [`LValue::tail_bound`] is the
//! BigFloat evaluation of this bound times an extra factor 2 absorbing its
//! own rounding.
//!
//! ## E_1 and the Euler-Mascheroni constant
//!
//! E_1(x) = -gamma - ln x + sum_{k>=1} (-1)^{k+1} x^k / (k * k!)
//! (Abramowitz-Stegun 5.1.11), evaluated with ceil(x log2 e) + 64 extra
//! working bits to absorb the alternating cancellation (largest term
//! ~ e^x / x).  gamma is computed from scratch at any precision by
//! Euler-Maclaurin: gamma = H_n - ln n - 1/(2n) + sum_{j=1}^{J}
//! B_{2j}/(2j n^{2j}) + R, with H_n and the Bernoulli numbers B_{2j} EXACT
//! rationals (so the only rounding before the final ln is one rational ->
//! BigFloat conversion), and |R| <= first omitted term (valid because
//! 1/x is completely monotone; Abramowitz-Stegun 23.1.5 / Graham-Knuth-
//! Patashnik 9.5).  Verified against mpmath to 60+ digits in the tests.
//!
//! Corresponds to `sage.modular.modform` L-series functionality
//! (`ModularForm.lseries`, Dokchitser-style exponential sums specialized
//! to weight 2) and the MAGMA handbook chapter "L-functions" (`LSeries`,
//! `Evaluate`, `CentralValue`), restricted to rational newform summands.

use super::decomposition::{HeckeEigenvalue, HeckeSummand, SummandHeckeAction};
use super::gamma0::ModularSymbolsGamma0;
use super::involutions::InvolutionAction;
use rustmath_core::analytic::RealField;
use rustmath_core::ordering::OrderedRing;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_reals::BigFloat;

/// log2(10), rounded up a hair (used only to size working precisions).
const LOG2_10: f64 = 3.321928094887363;

/// A computed L-value (or derivative) with its honesty envelope.
#[derive(Debug, Clone)]
pub struct LValue {
    /// The computed value (series truncated after finitely many terms).
    pub value: BigFloat,
    /// Rigorous bound on the omitted tail (see the module docs); exactly
    /// zero when the value is exact (the epsilon = -1 central zero).
    pub tail_bound: BigFloat,
    /// Documented allowance for BigFloat rounding across the summation
    /// (an engineering bound, spelled out in `rounding_note`; the working
    /// precision carries >= 64 guard bits past the requested digits).
    pub rounding_allowance: BigFloat,
    /// Human-readable statement of what was computed and what error model
    /// the two bounds follow.
    pub rounding_note: String,
}

impl LValue {
    /// True iff the value is certifiably nonzero: |value| exceeds the tail
    /// bound plus the documented rounding allowance.  (A `false` return
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

/// 2^k as an Integer (k small).
fn pow2_integer(k: u64) -> Integer {
    let mut out = Integer::one();
    let two = Integer::from(2);
    for _ in 0..k {
        out = &out * &two;
    }
    out
}

/// The rational 2^-k as a BigFloat at the given precision.
fn pow2_neg(k: u64, prec: u64) -> BigFloat {
    let r = Rational::new(Integer::one(), pow2_integer(k)).expect("power of two is nonzero");
    BigFloat::from_rational(&r, prec)
}

/// Bernoulli numbers B_0..B_m as exact rationals, via the defining
/// recurrence sum_{k=0}^{n} C(n+1, k) B_k = 0 (self-contained; B_1 = -1/2
/// convention, irrelevant here since only even indices are consumed).
/// The Pascal row for n+1 is maintained incrementally (O(m^2) total).
fn bernoulli_numbers(m: usize) -> Vec<Rational> {
    let mut b: Vec<Rational> = Vec::with_capacity(m + 1);
    // seed: B_0 = 1 (the recurrence below defines B_n only for n >= 1)
    b.push(Rational::one());
    // row = binomials C(n+1, k), k = 0..=n+1; starts at n = 1 -> C(2, .)
    let mut row = vec![Integer::one(), Integer::from(2), Integer::one()];
    for n in 1..=m {
        // sum_{k=0}^{n-1} C(n+1, k) B_k + C(n+1, n) B_n = 0
        let mut acc = Rational::zero();
        for (k, bk) in b.iter().enumerate() {
            if !bk.is_zero() {
                acc = &acc + &(&Rational::from_integer(row[k].clone()) * bk);
            }
        }
        let cn = Rational::from_integer(row[n].clone());
        let bn = -&(&acc / &cn);
        b.push(bn);
        // advance the Pascal row from C(n+1, .) to C(n+2, .)
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
/// < 2^-(prec+8) under the documented model (exact-rational Euler-Maclaurin
/// core + one ln evaluation; see the module docs).
pub fn euler_gamma(prec: u64) -> BigFloat {
    // The asymptotic term |B_2j|/(2j n^{2j}) at the stopping index j = 2n
    // is ~ e^{-5.8 n}; n = 0.14 (prec + 24) + 4 leaves a comfortable margin
    // over the required 2^-(prec+24) = e^{-0.694 (prec+24)} (and the exact
    // rational threshold test below is the actual guarantee).
    let n = ((prec as f64 + 24.0) * 0.14) as u64 + 4;
    let threshold =
        Rational::new(Integer::one(), pow2_integer(prec + 24)).expect("power of two is nonzero");
    // H_n - 1/(2n), exact
    let n_int = Integer::from(n as i64);
    let mut core = Rational::zero();
    for k in 1..=n {
        core = &core + &Rational::new(Integer::one(), Integer::from(k as i64)).expect("k > 0");
    }
    core = &core - &Rational::new(Integer::one(), Integer::from(2 * n as i64)).expect("2n > 0");
    // + sum_j B_2j / (2j n^{2j}) until the (exact) term drops below threshold
    let jmax = 2 * n as usize + 8; // past the asymptotic minimum ~ pi n
    let bern = bernoulli_numbers(2 * jmax + 2);
    let mut n2j = &n_int * &n_int; // n^{2j}, starting j = 1
    let mut reached = false;
    for j in 1..=jmax {
        let term = &bern[2 * j] / &Rational::from_integer(&Integer::from(2 * j as i64) * &n2j);
        let abs_term = if term < Rational::zero() {
            -&term
        } else {
            term.clone()
        };
        if abs_term < threshold {
            // |remainder| <= first omitted term < 2^-(prec+24)
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

/// Working precision for the E_1 alternating series at argument x: the
/// requested bits plus the cancellation budget (largest term ~ e^x/x).
fn e1_working_precision(xf: f64, prec: u64) -> u64 {
    prec + 48 + (std::f64::consts::LOG2_E * xf.max(0.0)).ceil() as u64
}

/// The exponential integral E_1(x) = int_x^oo e^{-t}/t dt for x > 0, with
/// absolute error < 2^-(prec+16) under the documented model: alternating
/// series E_1(x) = -gamma - ln x + sum (-1)^{k+1} x^k/(k k!) evaluated with
/// ceil(x log2 e) + 48 guard bits (the cancellation budget), truncated when
/// the terms (eventually decreasing) drop below 2^-(wp).
pub fn exp_integral_e1(x: &BigFloat, prec: u64) -> Result<BigFloat, String> {
    let wp = e1_working_precision(x.to_f64(), prec);
    let g = euler_gamma(wp);
    e1_with_gamma(x, prec, &g)
}

/// E_1 as above with a caller-supplied Euler-Mascheroni constant (carried
/// at >= the working precision needed for this x), so batch callers do not
/// recompute gamma per term.
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
    let mut term = BigFloat::one_prec(wp); // x^k / k! at k = 0
    let mut sum = BigFloat::zero_prec(wp);
    let cutoff = pow2_neg(wp, wp);
    let mut k: i64 = 1;
    loop {
        // term <- x^k / k!
        term = term * xw.clone() / BigFloat::from_integer(&Integer::from(k), wp);
        let contrib = term.clone() / BigFloat::from_integer(&Integer::from(k), wp);
        sum = if k % 2 == 1 {
            sum + contrib
        } else {
            sum - contrib
        };
        // terms decrease once k > x; then first omitted < cutoff bounds the rest
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

/// The L-series of a RATIONAL weight-2 newform, tied to a 2-dimensional
/// summand of the sign-0 cuspidal Hecke decomposition.
///
/// The 2-dimensionality is exactly the rational-newform certificate: an
/// old block has >= 2 degeneracy images (dimension >= 4 in sign 0) and a
/// Galois orbit of degree d has dimension 2d, so dimension 2 forces a
/// newform of THIS level with rational eigenvalues.  As a cross-check the
/// constructor also demands that the Fricke involution W_N act as a SCALAR
/// on the summand (old blocks always split under W_N because it swaps the
/// degeneracy images f(dz) <-> f((N/(M d)) z)).
pub struct RationalNewformLSeries<'a> {
    space: &'a ModularSymbolsGamma0,
    summand: &'a HeckeSummand,
    fricke_sign: i8,
}

impl<'a> RationalNewformLSeries<'a> {
    /// Attach to a rational-newform summand.  Errors (honestly) on old
    /// blocks, Galois orbits of degree > 1, and anything else that is not
    /// a 2-dimensional sign-0 summand with a scalar Fricke action.
    pub fn new(space: &'a ModularSymbolsGamma0, summand: &'a HeckeSummand) -> Result<Self, String> {
        if summand.dimension() != 2 {
            return Err(format!(
                "RationalNewformLSeries needs a 2-dimensional sign-0 summand \
                 (rational newform); got dimension {}",
                summand.dimension()
            ));
        }
        let n = space.level();
        let fricke_sign = match space.atkin_lehner_on_summand(summand, n)? {
            InvolutionAction::Scalar(s) => s,
            other => {
                return Err(format!(
                    "Fricke involution is not scalar on the summand ({other:?}): \
                     not a newform summand"
                ))
            }
        };
        Ok(RationalNewformLSeries {
            space,
            summand,
            fricke_sign,
        })
    }

    /// The level N of the newform (= the level of the ambient space).
    pub fn level(&self) -> u64 {
        self.space.level()
    }

    /// The Fricke eigenvalue w_N (f|_2 W_N = w_N f), from the exact
    /// rational Atkin-Lehner matrix.
    pub fn fricke_sign(&self) -> i8 {
        self.fricke_sign
    }

    /// The functional-equation sign epsilon = -w_N
    /// (Lambda(s) = epsilon Lambda(2 - s); derived in the module docs).
    pub fn root_number(&self) -> i8 {
        -self.fricke_sign
    }

    /// The exact Hecke eigenvalues a_1..a_nmax as rationals (in fact
    /// integers; integrality and the Ramanujan / bad-prime bounds are
    /// asserted on every prime consumed): a_p from the modular-symbol
    /// eigensystem, a_{p^{k+1}} = a_p a_{p^k} - p a_{p^{k-1}} for p
    /// coprime to N, a_{p^k} = a_p^k for p | N, and a_{mn} = a_m a_n for
    /// coprime m, n (weight 2, trivial character).
    ///
    /// Index 0 of the returned vector is unused (set to 0).
    pub fn coefficients(&self, nmax: usize) -> Result<Vec<Rational>, String> {
        let n_level = self.space.level();
        let mut a = vec![Rational::zero(); nmax + 1];
        if nmax == 0 {
            return Ok(a);
        }
        a[1] = Rational::one();
        for p in 2..=nmax as u64 {
            if !is_prime_u64(p) {
                continue;
            }
            let ap = match self.space.hecke_action_on_summand(self.summand, p)? {
                SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(v)) => v,
                other => {
                    return Err(format!(
                        "T_{p} does not act by a rational scalar on the summand \
                         ({other:?}): not a rational newform"
                    ))
                }
            };
            if !ap.denominator().is_one() {
                return Err(format!("a_{p} = {ap:?} is not an integer"));
            }
            // wrongness detectors that also certify the tail-bound
            // hypotheses (|a_n| <= d(n) sqrt(n)) at the consumed primes
            let ai = ap.numerator().to_i64();
            if n_level.is_multiple_of(p) {
                let exact = !(n_level / p).is_multiple_of(p);
                if exact && ai * ai != 1 {
                    return Err(format!("|a_{p}| != 1 at a multiplicative prime: {ai}"));
                }
                if !exact && ai != 0 {
                    return Err(format!("a_{p} != 0 at an additive prime: {ai}"));
                }
            } else if (ai * ai) as u64 > 4 * p {
                return Err(format!("Ramanujan bound violated: a_{p} = {ai}"));
            }
            a[p as usize] = ap.clone();
            // prime powers
            let mut pk = (p * p) as usize;
            while pk <= nmax {
                a[pk] = if n_level.is_multiple_of(p) {
                    &a[pk / p as usize] * &ap
                } else {
                    &(&ap * &a[pk / p as usize])
                        - &(&Rational::from_integer(Integer::from(p as i64))
                            * &a[pk / (p * p) as usize])
                };
                pk *= p as usize;
            }
        }
        // composites: split off the full power of the smallest prime factor
        for m in 2..=nmax {
            let mut sp = 0usize;
            let mut mm = m;
            for q in 2..=m {
                if mm % q == 0 {
                    sp = q;
                    break;
                }
            }
            if sp == 0 {
                continue;
            }
            let mut pk = 1usize;
            while mm % sp == 0 {
                mm /= sp;
                pk *= sp;
            }
            if mm > 1 {
                a[m] = &a[pk] * &a[mm];
            }
        }
        Ok(a)
    }

    /// Truncation point M and the geometric tail bound machinery shared by
    /// [`Self::l1`] and [`Self::l1_derivative`]: smallest M with
    /// 4 e^{-c(M+1)} / (1 - e^{-c}) < 10^{-(digits+3)}, computed in f64 and
    /// then re-evaluated rigorously in BigFloat for the returned bound.
    fn truncation_point(&self, digits: usize) -> usize {
        let n = self.space.level() as f64;
        let c = 2.0 * std::f64::consts::PI / n.sqrt();
        // 4 e^{-c(M+1)} / (1 - e^{-c}) < 10^{-(digits+3)}  <=>
        // M + 1 > [ (digits+3) ln 10 + ln 4 - ln(1 - e^{-c}) ] / c
        let target = (digits as f64 + 3.0) * std::f64::consts::LN_10;
        let m_plus_1 = (target + 4.0f64.ln() - (1.0 - (-c).exp()).ln()) / c;
        (m_plus_1.ceil() as usize).max(10) + 2
    }

    /// The rigorous BigFloat tail bound 2 * 4 e^{-c(M+1)} / (1 - e^{-c})
    /// (the leading factor 2 absorbs the bound's own rounding).
    fn tail_bound(&self, m: usize, wp: u64) -> BigFloat {
        let c = self.decay_constant(wp);
        let e_c = RealField::exp(&(-c.clone()));
        let numer =
            RealField::exp(&(-(c * BigFloat::from_integer(&Integer::from((m + 1) as i64), wp))));
        let denom = BigFloat::one_prec(wp) - e_c;
        let eight = BigFloat::from_integer(&Integer::from(8), wp); // 2 (safety) * 4
        eight * numer / denom
    }

    /// c = 2 pi / sqrt(N) at wp bits.
    fn decay_constant(&self, wp: u64) -> BigFloat {
        let two = BigFloat::from_integer(&Integer::from(2), wp);
        let pi = <BigFloat as RealField>::pi(wp);
        let sqrt_n = RealField::sqrt(&BigFloat::from_integer(
            &Integer::from(self.space.level() as i64),
            wp,
        ));
        two * pi / sqrt_n
    }

    /// L(f, 1) to about `digits` decimal digits.
    ///
    /// epsilon = +1: value = 2 sum_{n<=M} (a_n/n) e^{-2 pi n / sqrt N} with
    /// the rigorous geometric tail bound of the module docs.
    ///
    /// epsilon = -1: the EXACT zero (tail and allowance are exactly zero;
    /// the zero comes from the exact functional-equation sign, not from
    /// numerics -- see the module docs).
    pub fn l1(&self, digits: usize) -> Result<LValue, String> {
        let wp = (digits as f64 * LOG2_10).ceil() as u64 + 64;
        if self.root_number() == -1 {
            return Ok(LValue {
                value: BigFloat::zero_prec(wp),
                tail_bound: BigFloat::zero_prec(wp),
                rounding_allowance: BigFloat::zero_prec(wp),
                rounding_note: "L(f,1) = 0 EXACTLY: epsilon = -1 in the functional \
                    equation Lambda(s) = epsilon Lambda(2-s), with epsilon = -w_N \
                    taken from the exact rational Atkin-Lehner matrix W_N (no \
                    numerics involved).  Cross-certified by the exact winding \
                    projection (Manin--Birch)."
                    .to_string(),
            });
        }
        let m = self.truncation_point(digits);
        let a = self.coefficients(m)?;
        let c = self.decay_constant(wp);
        let mut sum = BigFloat::zero_prec(wp);
        let mut abs_sum = BigFloat::zero_prec(wp);
        for (n, an) in a.iter().enumerate().skip(1) {
            if an.is_zero() {
                continue;
            }
            let n_bf = BigFloat::from_integer(&Integer::from(n as i64), wp);
            let coeff = BigFloat::from_rational(an, wp) / n_bf.clone();
            let term = coeff * RealField::exp(&(-(c.clone() * n_bf)));
            abs_sum = abs_sum + OrderedRing::abs(&term);
            sum = sum + term;
        }
        let two = BigFloat::from_integer(&Integer::from(2), wp);
        let value = two.clone() * sum;
        let tail = self.tail_bound(m, wp);
        // rounding model: <= ~8 BigFloat ops per term (exp counts once),
        // each with relative error <= 2^-(wp-2); allowance =
        // 2 * (abs_sum + 1) * 16 (M + 4) * 2^-(wp - 8).
        let ops = BigFloat::from_integer(&Integer::from(16 * (m as i64 + 4)), wp);
        let allowance = two * (abs_sum + BigFloat::one_prec(wp)) * ops * pow2_neg(wp - 8, wp);
        Ok(LValue {
            value,
            tail_bound: tail,
            rounding_allowance: allowance,
            rounding_note: format!(
                "L(f,1) = 2 sum_(n=1..{m}) (a_n/n) exp(-2 pi n / sqrt({n})) for \
                 epsilon = +1, exact rational a_n, BigFloat arithmetic at {wp} \
                 bits.  tail_bound = 2 * [4 e^(-c(M+1))/(1-e^(-c))] from \
                 |a_n| <= d(n) sqrt(n) (Deligne) and d(n) <= 2 sqrt(n); \
                 rounding_allowance covers <= 16(M+4) operations of relative \
                 error 2^-({wp}-8) on the term-magnitude sum (engineering \
                 bound, 64 guard bits past the requested {digits} digits).",
                n = self.space.level(),
            ),
        })
    }

    /// L'(f, 1) to about `digits` decimal digits, for epsilon = -1 ONLY
    /// (the derivation of the E_1 series uses Lambda(1) = 0; an honest
    /// error is returned otherwise):
    /// L'(f,1) = 2 sum (a_n/n) E_1(2 pi n / sqrt N), rigorous tail as in
    /// the module docs.
    pub fn l1_derivative(&self, digits: usize) -> Result<LValue, String> {
        if self.root_number() != -1 {
            return Err(
                "l1_derivative requires epsilon = -1 (the series derivation uses \
                 Lambda(1) = 0); for epsilon = +1 compute l1 instead"
                    .to_string(),
            );
        }
        let m = self.truncation_point(digits);
        let cf64 = 2.0 * std::f64::consts::PI / (self.space.level() as f64).sqrt();
        if cf64 * ((m + 1) as f64) < 1.0 {
            return Err("internal: c(M+1) < 1, E_1 tail bound inapplicable".to_string());
        }
        // E_1 handles its own cancellation budget internally; the outer sum
        // only needs the standard guard bits
        let wp = (digits as f64 * LOG2_10).ceil() as u64 + 64;
        let a = self.coefficients(m)?;
        let c = self.decay_constant(wp);
        // gamma once, at the working precision of the LARGEST E_1 argument
        let gamma = euler_gamma(e1_working_precision(cf64 * (m as f64 + 2.0), wp));
        let mut sum = BigFloat::zero_prec(wp);
        let mut abs_sum = BigFloat::zero_prec(wp);
        for (n, an) in a.iter().enumerate().skip(1) {
            if an.is_zero() {
                continue;
            }
            let n_bf = BigFloat::from_integer(&Integer::from(n as i64), wp);
            let coeff = BigFloat::from_rational(an, wp) / n_bf.clone();
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
                "L'(f,1) = 2 sum_(n=1..{m}) (a_n/n) E_1(2 pi n / sqrt({n})) for \
                 epsilon = -1, exact rational a_n, BigFloat arithmetic at {wp} \
                 bits (E_1 carries its own cancellation budget).  tail_bound = \
                 2 * [4 e^(-c(M+1))/(1-e^(-c))] using E_1(x) <= e^-x for x >= 1 \
                 and |a_n|/n <= 2 (Deligne + divisor pairing); \
                 rounding_allowance covers <= 16(M+4) operations of relative \
                 error 2^-({wp}-16) on the term-magnitude sum (engineering \
                 bound; E_1 itself carries absolute error < 2^-({wp}+16) by its \
                 own guard bits).",
                n = self.space.level(),
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64, d: i64) -> Rational {
        Rational::new(Integer::from(n), Integer::from(d)).unwrap()
    }

    /// |a - b| < 10^-k, all in exact BigFloat comparisons.
    fn close_to(a: &BigFloat, decimal: &str, k: usize) -> bool {
        let prec = RealField::precision(a).max(256);
        let b = BigFloat::from_decimal_str(decimal, prec).unwrap();
        let tol_str = format!("0.{}1", "0".repeat(k - 1));
        let tol = BigFloat::from_decimal_str(&tol_str, prec).unwrap();
        OrderedRing::abs(&(a.clone() - b)) < tol
    }

    #[test]
    fn test_bernoulli_numbers_classical_values() {
        // classical table (independently: B_12 = -691/2730 is the famous
        // irregular-prime witness), odd indices > 1 vanish
        let b = bernoulli_numbers(12);
        assert_eq!(b[0], rat(1, 1));
        assert_eq!(b[1], rat(-1, 2));
        assert_eq!(b[2], rat(1, 6));
        assert_eq!(b[4], rat(-1, 30));
        assert_eq!(b[6], rat(1, 42));
        assert_eq!(b[8], rat(-1, 30));
        assert_eq!(b[10], rat(5, 66));
        assert_eq!(b[12], rat(-691, 2730));
        for k in [3usize, 5, 7, 9, 11] {
            assert!(b[k].is_zero(), "B_{k} = 0");
        }
    }

    /// gamma pinned against mpmath (65 digits, derived independently
    /// before this test):
    /// 0.57721566490153286060651209008240243104215933593992359880576723488
    #[test]
    fn test_euler_gamma_matches_mpmath() {
        let g = euler_gamma(220);
        assert!(
            close_to(
                &g,
                "0.57721566490153286060651209008240243104215933593992359880576723488",
                60
            ),
            "gamma to 60 digits; got {}",
            g.to_decimal_string(65)
        );
        // internal consistency across two precisions
        let g_low = euler_gamma(96);
        assert!(
            OrderedRing::abs(&(g_low - g.with_precision(96))) < pow2_neg(90, 96),
            "gamma(96) vs gamma(220)"
        );
    }

    /// E_1 pinned against mpmath (independently derived):
    ///   E_1(1)   = 0.2193839343955202736771637754601216490310472934069082076
    ///   E_1(1/2) = 0.5597735947761608117467959393150852352268468903163535152
    ///   E_1(10)  = 4.156968929685324277402859810278180384346290082419533133e-6
    #[test]
    fn test_exp_integral_e1_matches_mpmath() {
        let prec = 200u64;
        let x1 = BigFloat::from_integer(&Integer::from(1), prec);
        let e1 = exp_integral_e1(&x1, prec).unwrap();
        assert!(
            close_to(
                &e1,
                "0.2193839343955202736771637754601216490310472934069082076",
                52
            ),
            "E_1(1); got {}",
            e1.to_decimal_string(55)
        );
        let xh = BigFloat::from_rational(&rat(1, 2), prec);
        let eh = exp_integral_e1(&xh, prec).unwrap();
        assert!(
            close_to(
                &eh,
                "0.5597735947761608117467959393150852352268468903163535152",
                52
            ),
            "E_1(1/2); got {}",
            eh.to_decimal_string(55)
        );
        let x10 = BigFloat::from_integer(&Integer::from(10), prec);
        let e10 = exp_integral_e1(&x10, prec).unwrap();
        assert!(
            close_to(
                &e10,
                "0.000004156968929685324277402859810278180384346290082419533133",
                55
            ),
            "E_1(10); got {}",
            e10.to_decimal_string(55)
        );
        // domain errors
        assert!(exp_integral_e1(&BigFloat::zero_prec(64), 64).is_err());
        assert!(
            exp_integral_e1(&(-BigFloat::one_prec(64)), 64).is_err(),
            "negative argument refused"
        );
    }

    /// Construction certificates: rational newform summands are accepted
    /// with the pinned Fricke signs; old blocks and quadratic orbits are
    /// honestly refused.
    #[test]
    fn test_lseries_construction_and_root_numbers() {
        let m11 = ModularSymbolsGamma0::new(11);
        let d11 = m11.cuspidal_hecke_decomposition().unwrap();
        let ls = RationalNewformLSeries::new(&m11, &d11.summands()[0]).unwrap();
        assert_eq!(ls.level(), 11);
        assert_eq!(ls.fricke_sign(), -1, "eta-derived w_11 = -1");
        assert_eq!(ls.root_number(), 1, "epsilon(11a) = +1");
        // 22: the old block must be refused (dimension 4)
        let m22 = ModularSymbolsGamma0::new(22);
        let d22 = m22.cuspidal_hecke_decomposition().unwrap();
        assert!(RationalNewformLSeries::new(&m22, &d22.summands()[0]).is_err());
        // 23: the quadratic orbit must be refused (dimension 4)
        let m23 = ModularSymbolsGamma0::new(23);
        let d23 = m23.cuspidal_hecke_decomposition().unwrap();
        assert!(RationalNewformLSeries::new(&m23, &d23.summands()[0]).is_err());
    }

    /// Coefficients from the eigensystem reproduce the point-counted /
    /// eta-certified 11a table a_1..a_14 (same table as the involutions
    /// tests, derived independently in stage 1).
    #[test]
    fn test_coefficients_match_11a_table() {
        let m = ModularSymbolsGamma0::new(11);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        let ls = RationalNewformLSeries::new(&m, &dec.summands()[0]).unwrap();
        let a = ls.coefficients(14).unwrap();
        let table11: [i64; 14] = [1, -2, -1, 2, 1, 2, -2, 0, -2, -2, 1, -2, 4, 4];
        for (i, &v) in table11.iter().enumerate() {
            assert_eq!(a[i + 1], rat(v, 1), "a_{} of 11a", i + 1);
        }
    }

    /// NUMERIC GATE: L(11a, 1) to 30+ digits.  Expected value derived
    /// independently (python mpmath, point-counted a_p, epsilon pinned by
    /// split-point independence, confirmed by a brute integral):
    /// 0.253841860855910684337758923350909461043898448 (45 digits).
    #[test]
    fn test_l1_11a_thirty_plus_digits() {
        let m = ModularSymbolsGamma0::new(11);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        let ls = RationalNewformLSeries::new(&m, &dec.summands()[0]).unwrap();
        let lv = ls.l1(36).unwrap();
        assert!(
            close_to(
                &lv.value,
                "0.253841860855910684337758923350909461043898448",
                33
            ),
            "L(11a,1) to 33 digits; got {}",
            lv.value.to_decimal_string(40)
        );
        assert!(lv.certified_nonzero(), "L(11a,1) certified nonzero");
        // the certified envelope really is at the requested scale
        let budget = lv.error_budget();
        let cap = BigFloat::from_decimal_str(
            &format!("0.{}1", "0".repeat(29)),
            RealField::precision(&budget),
        )
        .unwrap();
        assert!(budget < cap, "error budget below 1e-30");
        // and the true (independently derived) value sits INSIDE the
        // envelope: |value - truth| <= budget
        let truth = BigFloat::from_decimal_str(
            "0.253841860855910684337758923350909461043898448",
            RealField::precision(&lv.value).max(256),
        )
        .unwrap();
        assert!(
            OrderedRing::abs(&(lv.value.clone() - truth)) < budget,
            "computed value within its own certified envelope of the truth"
        );
    }

    /// NUMERIC GATE: L(37b, 1) nonzero, value pinned to 28 digits
    /// (independent derivation as above):
    /// 0.725681061936152782336205541026396548736760336
    #[test]
    fn test_l1_37b_nonzero() {
        let m = ModularSymbolsGamma0::new(37);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        let w37b = dec
            .summands()
            .iter()
            .find(|w| {
                matches!(
                    m.hecke_action_on_summand(w, 2).unwrap(),
                    SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(ref a))
                        if a.is_zero()
                )
            })
            .expect("37b has a_2 = 0");
        let ls = RationalNewformLSeries::new(&m, w37b).unwrap();
        assert_eq!(ls.root_number(), 1, "epsilon(37b) = +1");
        let lv = ls.l1(30).unwrap();
        assert!(
            close_to(
                &lv.value,
                "0.725681061936152782336205541026396548736760336",
                28
            ),
            "L(37b,1); got {}",
            lv.value.to_decimal_string(35)
        );
        assert!(lv.certified_nonzero());
    }

    /// NUMERIC GATE: 37a.  L(37a, 1) is the EXACT functional-equation zero
    /// (epsilon = -1 from the exact W_37 matrix), and L'(37a, 1) to 20+
    /// digits, certified nonzero.  Expected derivative derived
    /// independently (python mpmath: the E_1 series AND the brute
    /// log-weighted integral 2 sqrt(N) int f(iy) ln(sqrt(N) y) dy agree to
    /// 40 digits): 0.3059997738340523018204836833216764744526
    #[test]
    fn test_l1_37a_exact_zero_and_derivative() {
        let m = ModularSymbolsGamma0::new(37);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        let w37a = dec
            .summands()
            .iter()
            .find(|w| {
                matches!(
                    m.hecke_action_on_summand(w, 2).unwrap(),
                    SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(ref a))
                        if *a == rat(-2, 1)
                )
            })
            .expect("37a has a_2 = -2");
        let ls = RationalNewformLSeries::new(&m, w37a).unwrap();
        assert_eq!(ls.fricke_sign(), 1);
        assert_eq!(ls.root_number(), -1, "epsilon(37a) = -1");
        // the central value is the exact zero
        let lv = ls.l1(30).unwrap();
        assert!(Ring::is_zero(&lv.value));
        assert!(Ring::is_zero(&lv.tail_bound));
        assert!(!lv.certified_nonzero(), "a zero is never certified nonzero");
        assert!(lv.rounding_note.contains("EXACTLY"));
        // the derivative, certified nonzero to 20+ digits
        let dv = ls.l1_derivative(24).unwrap();
        assert!(
            close_to(&dv.value, "0.3059997738340523018204836833216764744526", 21),
            "L'(37a,1) to 21 digits; got {}",
            dv.value.to_decimal_string(28)
        );
        assert!(dv.certified_nonzero(), "L'(37a,1) certified nonzero");
        let truth = BigFloat::from_decimal_str(
            "0.3059997738340523018204836833216764744526",
            RealField::precision(&dv.value).max(256),
        )
        .unwrap();
        assert!(
            OrderedRing::abs(&(dv.value.clone() - truth)) < dv.error_budget(),
            "L' within its certified envelope"
        );
        // derivative formula honestly refuses epsilon = +1 summands
        let m11 = ModularSymbolsGamma0::new(11);
        let d11 = m11.cuspidal_hecke_decomposition().unwrap();
        let ls11 = RationalNewformLSeries::new(&m11, &d11.summands()[0]).unwrap();
        assert!(ls11.l1_derivative(12).is_err());
    }

    /// CROSS-GATE: the exact vanishing lattice (winding projection) agrees
    /// with the numeric certified-nonzero lattice on every rational
    /// newform summand computed in this chunk:
    ///   epsilon = +1  =>  !l1_vanishes  and  l1 certified nonzero;
    ///   epsilon = -1  =>   l1_vanishes  and  the exact zero (and at 37,
    ///                      L' certified nonzero: analytic rank exactly 1).
    #[test]
    fn test_cross_gate_exact_vs_numeric() {
        for n in [11u64, 14, 15, 17, 19, 20, 21, 24, 37] {
            let m = ModularSymbolsGamma0::new(n);
            let dec = m.cuspidal_hecke_decomposition().unwrap();
            for w in dec.summands() {
                if w.dimension() != 2 {
                    continue;
                }
                let ls = RationalNewformLSeries::new(&m, w).unwrap();
                let vanish = m.l1_vanishes(w).unwrap();
                if ls.root_number() == 1 {
                    assert!(!vanish, "epsilon = +1, rank 0 at level {n}");
                    assert!(
                        ls.l1(15).unwrap().certified_nonzero(),
                        "numeric nonzero at level {n}"
                    );
                } else {
                    assert!(vanish, "epsilon = -1 at level {n}");
                    assert!(Ring::is_zero(&ls.l1(15).unwrap().value));
                    assert!(
                        ls.l1_derivative(12).unwrap().certified_nonzero(),
                        "L' != 0 at level {n}"
                    );
                }
            }
        }
    }
}
