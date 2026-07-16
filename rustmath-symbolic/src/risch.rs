//! Rational-function integration as a decision procedure.
//!
//! This module integrates any expression that normalizes to a univariate
//! rational function over ℚ, via the classical exact pipeline:
//!
//! 1. **Normalizer** ([`as_rational_function`]): `Expr` → reduced `num/den`
//!    over `ℚ[x]` with `gcd(num, den) = 1` and `den` monic. Anything that is
//!    not exactly such a rational function is refused with `None`.
//! 2. **Hermite reduction** (`hermite_reduce`): peels off the derivative of a
//!    rational function, leaving a proper integrand with squarefree
//!    denominator. Self-gated: the identity
//!    `input = d/dx(rational part) + poly part + log part` is re-checked
//!    exactly (assert, on in release).
//! 3. **Rothstein–Trager** (`rothstein_trager`): for the remaining
//!    `A₁/D*` (squarefree `D*`), the log coefficients are the roots of
//!    `R(t) = Res_X(A₁ − t·D*', D*)`, computed with the fast
//!    evaluation–interpolation resultant from `rustmath-polynomials`
//!    (`bivariate::resultant_in_t`). Rational roots give `c·log(v)` terms;
//!    conjugate pairs `a ± bi` with rational `a, b` give the real form
//!    `a·log(P² + Q²) − 2b·arctan(Q/P)`; every other irreducible factor is
//!    emitted as an honest symbolic `root_sum` (see below). The sum of the
//!    exact derivatives of all emitted terms is re-checked against `A₁/D*`
//!    (assert, on in release) — `root_sum` derivatives are computed exactly
//!    via traces in `ℚ(x)[t]/(f)`, no numerics.
//!
//! Every rational function has an elementary antiderivative; the split
//! between [`RischResult::Elementary`] and [`RischResult::WithRootSum`] only
//! records whether this module could *express* all log terms with rational
//! data. Nothing is ever dropped or approximated.
//!
//! # The `root_sum` symbolic object
//!
//! `Expr::Function("root_sum", [f, g])` denotes `Σ_{c ∈ ℂ : f(c) = 0} g|_{τ=c}`,
//! where `f` is a polynomial in the reserved symbol `τ` (named `"_t"`, or
//! `"_t0"` if the integration variable is itself named `"_t"`), irreducible
//! over ℚ (so its roots are simple), and `g` is an expression in the
//! integration variable and `τ`. Here `g` is always `τ·log(v(x, τ))` with `v`
//! monic in `x` and coefficients reduced mod `f`. The generic
//! `Expr::differentiate` knows exactly one rule for `root_sum`:
//! `d/dx root_sum(f, g) = root_sum(f, ∂g/∂x)` — sound because the sum ranges
//! over the roots of `f` in the bound variable τ, which do not involve `x`.
//! Independently of that rule, the correctness of every emitted `root_sum`
//! term is certified by this module's internal exact gate.
//!
//! # Resource budgets
//!
//! Hermite reduction and Rothstein–Trager both suffer super-exponential
//! coefficient swell on inputs of modest syntactic size (high multiplicities
//! in the denominator; high-degree irreducible factors of the resultant).
//! [`integrate_rational_risch`] therefore runs under a [`RischBudget`]:
//! structural pre-caps decided from the shape of the input, plus a
//! coefficient-bit-size ceiling checked on every iteration of every loop
//! whose coefficients can swell, so nothing can spin regardless of shape. A
//! tripped budget returns [`RischResult::BudgetExceeded`] — a labeled
//! **resource** refusal, deliberately distinct from the **mathematical**
//! refusal [`RischResult::NotRational`] (the input *is* rational and its
//! elementary antiderivative *does* exist; we merely declined to pay for
//! it). Callers who want to pay more use
//! [`integrate_rational_risch_with_budget`].
//!
//! # Honest-refusal surface
//!
//! [`as_rational_function`] returns `None` (and hence
//! [`integrate_rational_risch`] returns [`RischResult::NotRational`]) for:
//! floating-point constants (`Expr::Real` is not exact), any free symbol
//! other than `var` (multivariate is out of scope), `var` inside any
//! non-polynomial context (`sin(x)`, `sqrt(x)`, `x^(1/2)`, `x^y`, `x % y`,
//! unknown functions), non-integer or oversized (`|k| > 4096`) exponents,
//! opaque constant subexpressions that are not `Integer`/`Rational` literals
//! combined by `+ − × ÷ ^` (e.g. `sin(2)`, `2^(1/2)`), and division by an
//! expression that is identically zero.
//!
//! One deliberate non-refusal: a product with a factor that normalizes to the
//! zero rational function annihilates the whole product (`0·log(x) = 0`),
//! provided the discarded factor has no identically-zero denominator. This is
//! the generic-identity convention every step of rational integration already
//! uses (`1/x` is integrated despite its pole), and it is what lets
//! `differentiate`'s unsimplified product-rule residues (`0·log(v) + c·v'/v`)
//! normalize during the differentiate-back gate.

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_core::{EuclideanDomain, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::bivariate::resultant_in_t;
use rustmath_polynomials::factorization::factor_over_integers;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;
use std::sync::Arc;

type QP = UnivariatePolynomial<Rational>;
/// A fraction of polynomials over ℚ in canonical form: `gcd(num, den) = 1`,
/// `den` monic (hence nonzero). The zero fraction is `(0, 1)`. Canonical form
/// makes equality of rational functions plain structural equality.
pub(crate) type QFrac = (QP, QP);

fn qr(n: i64) -> Rational {
    Rational::from_i64(n)
}

/// Coefficient of `x^i`, zero when `i` exceeds the degree.
/// (`UnivariatePolynomial::coeff` falls back to `coeffs[0]` out of range,
/// which is wrong for this use — do not use it here.)
fn coeff_at(p: &QP, i: usize) -> Rational {
    p.coefficients().get(i).cloned().unwrap_or_else(Rational::zero)
}

fn monic(p: &QP) -> QP {
    match p.leading_coefficient() {
        Some(lc) if !lc.is_one() => {
            let inv = lc.reciprocal().expect("monic: nonzero leading coefficient");
            p.scalar_mul(&inv)
        }
        _ => p.clone(),
    }
}

/// Exact division in ℚ[x]; panics if the division leaves a remainder
/// (all call sites divide by a known factor).
fn exact_div(a: &QP, b: &QP) -> QP {
    let (quo, rem) = a.quo_rem(b);
    assert!(rem.is_zero(), "exact_div: {} not divisible by {}", a, b);
    quo
}

fn poly_pow(p: &QP, k: usize) -> QP {
    let mut out = QP::one();
    for _ in 0..k {
        out = out * p.clone();
    }
    out
}

/// Reduce `num/den` to canonical form. `None` iff `den` is the zero polynomial.
pub(crate) fn reduce(num: QP, den: QP) -> Option<QFrac> {
    if den.is_zero() {
        return None;
    }
    if num.is_zero() {
        return Some((QP::zero(), QP::one()));
    }
    let g = num.gcd(&den);
    let (num, den) = if g.degree().unwrap_or(0) >= 1 {
        (exact_div(&num, &g), exact_div(&den, &g))
    } else {
        (num, den)
    };
    let inv = den
        .leading_coefficient()
        .expect("reduce: nonzero denominator")
        .reciprocal()
        .expect("reduce: nonzero leading coefficient");
    Some((num.scalar_mul(&inv), den.scalar_mul(&inv)))
}

pub(crate) fn frac_zero() -> QFrac {
    (QP::zero(), QP::one())
}

pub(crate) fn frac_one() -> QFrac {
    (QP::one(), QP::one())
}

pub(crate) fn frac_add(a: &QFrac, b: &QFrac) -> QFrac {
    let num = a.0.clone() * b.1.clone() + b.0.clone() * a.1.clone();
    let den = a.1.clone() * b.1.clone();
    reduce(num, den).expect("frac_add: denominators nonzero")
}

pub(crate) fn frac_mul(a: &QFrac, b: &QFrac) -> QFrac {
    reduce(a.0.clone() * b.0.clone(), a.1.clone() * b.1.clone())
        .expect("frac_mul: denominators nonzero")
}

fn frac_inv(a: &QFrac) -> QFrac {
    assert!(!a.0.is_zero(), "frac_inv: division by zero rational function");
    reduce(a.1.clone(), a.0.clone()).expect("frac_inv: numerator nonzero")
}

pub(crate) fn frac_scale(a: &QFrac, s: &Rational) -> QFrac {
    reduce(a.0.scalar_mul(s), a.1.clone()).expect("frac_scale: denominator nonzero")
}

/// Exact derivative of a rational function, in canonical form.
fn frac_derivative(a: &QFrac) -> QFrac {
    let num = a.0.derivative() * a.1.clone() - a.0.clone() * a.1.derivative();
    let den = a.1.clone() * a.1.clone();
    reduce(num, den).expect("frac_derivative: denominator nonzero")
}

/// Equality of canonical fractions is structural equality.
fn frac_eq(a: &QFrac, b: &QFrac) -> bool {
    a.0 == b.0 && a.1 == b.1
}

// ------------------------------------------------------------------------ //
// Resource budgets
// ------------------------------------------------------------------------ //

/// Resource budget for [`integrate_rational_risch_with_budget`].
///
/// Two kinds of caps: **structural pre-caps**, decided from the shape of the
/// input before a stage starts, and a **coefficient-bit-size ceiling**,
/// checked on every iteration of every loop whose coefficients can swell.
/// The loops themselves all terminate structurally (degrees or
/// multiplicities strictly decrease), so bounding the size of every working
/// value bounds total time. Exceeding any cap aborts the stage with
/// [`RischResult::BudgetExceeded`]; nothing partial is ever emitted.
///
/// Defaults were chosen from measured debug-build timings (recorded on the
/// field docs below) so that everything they admit completes in about a
/// second or less, while the audited blowups — which took 63 s to
/// non-terminating (kills at 150–550 s) pre-budget — are refused in
/// milliseconds. All fields are public: construct a bigger budget to pay
/// more.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RischBudget {
    /// Hermite structural pre-cap: the work measure
    /// `(Σ_{(V,m): m ≥ 2} (m − 1) · deg V) · deg(den)` over the squarefree
    /// decomposition of the denominator — an upper-bound proxy for
    /// (reduction-loop iterations) × (per-iteration operand degree).
    /// Measured on `1/((x−1)^m (x+2))` (debug build): work 63 ≈ 28 ms,
    /// 143 ≈ 0.32 s, 195 ≈ 0.94 s, 288 ≈ 3.5 s, 399 ≈ 14 s, 899 ≈ 63 s.
    pub max_hermite_work: usize,
    /// Rothstein–Trager structural pre-cap on `deg D*` (the squarefree
    /// denominator of the log integrand): bounds the resultant and its
    /// Zassenhaus factorization, and the `x`-degree of every `K[x]` gcd.
    /// Measured on `1/(x^d+x+1)` up to the factorization (debug build):
    /// d = 15 ≈ 31 ms, 20 ≈ 0.15 s, 24 ≈ 0.6 s, 40 ≈ 48 s.
    pub max_log_degree: usize,
    /// Structural pre-cap on the degree of an irreducible factor `f` of the
    /// Rothstein–Trager resultant for which arithmetic in `K = ℚ[t]/(f)`
    /// (the `root_sum` gcd and its exact trace gate; degree 2 also covers
    /// the arctan branch) is attempted. Measured on `1/(x^d+x+1)` with
    /// `deg f = d` (debug build): d = 5 ≈ 19 ms, 6 ≈ 0.33 s, 7 ≈ 1.2 s,
    /// 10 ≈ 3.5 s, 15 > 500 s — super-exponential swell.
    pub max_root_sum_degree: usize,
    /// Ceiling on the total coefficient bit size (numerators plus
    /// denominators, summed over all coefficients) of any single working
    /// value inside the iterative kernels: the Hermite loop state, the
    /// `K[x]` remainder sequences, and the trace machinery over
    /// `ℚ(x)[t]/(f)`. Checked every loop iteration.
    pub max_coeff_bits: u64,
}

impl Default for RischBudget {
    fn default() -> Self {
        RischBudget {
            max_hermite_work: 200,
            max_log_degree: 24,
            max_root_sum_degree: 6,
            max_coeff_bits: 1 << 16,
        }
    }
}

/// Internal marker for a tripped budget: which stage refused, and why.
#[derive(Debug, Clone)]
struct BudgetTrip {
    stage: &'static str,
    reason: String,
}

fn rat_bits(r: &Rational) -> u64 {
    r.numerator().bit_length() + r.denominator().bit_length()
}

fn qp_bits(p: &QP) -> u64 {
    p.coefficients().iter().map(rat_bits).sum()
}

fn frac_bits(f: &QFrac) -> u64 {
    qp_bits(&f.0) + qp_bits(&f.1)
}

/// The per-iteration ceiling check shared by all swell-prone loops.
fn check_bits(
    bits: u64,
    budget: &RischBudget,
    stage: &'static str,
    what: &str,
) -> Result<(), BudgetTrip> {
    if bits > budget.max_coeff_bits {
        Err(BudgetTrip {
            stage,
            reason: format!(
                "{} reached {} coefficient bits (max_coeff_bits {})",
                what, bits, budget.max_coeff_bits
            ),
        })
    } else {
        Ok(())
    }
}

/// Extended Euclid in ℚ[x]: returns `(g, s, t)` with `s·a + t·b = g`,
/// `g` a greatest common divisor (not normalized).
fn ext_gcd(a: &QP, b: &QP) -> (QP, QP, QP) {
    let (mut r0, mut r1) = (a.clone(), b.clone());
    let (mut s0, mut s1) = (QP::one(), QP::zero());
    let (mut t0, mut t1) = (QP::zero(), QP::one());
    while !r1.is_zero() {
        let (quo, rem) = r0.quo_rem(&r1);
        let s2 = s0 - quo.clone() * s1.clone();
        let t2 = t0 - quo * t1.clone();
        r0 = r1;
        r1 = rem;
        s0 = s1;
        s1 = s2;
        t0 = t1;
        t1 = t2;
    }
    (r0, s0, t0)
}

// ------------------------------------------------------------------------ //
// Part A: the normalizer
// ------------------------------------------------------------------------ //

/// Conservative well-definedness check backing the `0·f = 0` annihilation
/// rule of the normalizer: `false` if any denominator (right side of `Div`,
/// or base of a negative integer power) is a rational function of `var` that
/// is identically zero. Denominators we cannot decide (e.g. `sin(x)`) are
/// accepted under the generic-identity convention: rational-function
/// integration works up to equality on a dense open set, exactly like `1/x`.
pub(crate) fn generically_defined(e: &Expr, var: &Symbol) -> bool {
    match e {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_) => true,
        Expr::Unary(_, inner) => generically_defined(inner, var),
        Expr::Function(_, args) => args.iter().all(|a| generically_defined(a, var)),
        Expr::Binary(op, l, r) => {
            if !generically_defined(l, var) || !generically_defined(r, var) {
                return false;
            }
            match op {
                BinaryOp::Div => {
                    !matches!(as_rational_function(r, var), Some((n, _)) if n.is_zero())
                }
                BinaryOp::Pow => match as_exact_integer(r) {
                    Some(k) if k < Integer::from(0) => {
                        !matches!(as_rational_function(l, var), Some((n, _)) if n.is_zero())
                    }
                    _ => true,
                },
                _ => true,
            }
        }
    }
}

/// Exact rational value of a constant expression built from
/// `Integer`/`Rational` literals with `+ − × ÷ ^` and unary negation.
/// Anything else (floats, symbols, sin(2), 2^(1/2), ...) is `None`.
/// Needed in exponent position: `differentiate` emits unevaluated exponents
/// like `3 - 1`, and the normalizer must fold them exactly.
pub(crate) fn as_exact_rational(e: &Expr) -> Option<Rational> {
    match e {
        Expr::Integer(n) => Some(Rational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        Expr::Unary(UnaryOp::Neg, inner) => as_exact_rational(inner).map(|r| -r),
        Expr::Binary(op, l, r) => {
            let a = as_exact_rational(l)?;
            match op {
                BinaryOp::Add => Some(a + as_exact_rational(r)?),
                BinaryOp::Sub => Some(a - as_exact_rational(r)?),
                BinaryOp::Mul => Some(a * as_exact_rational(r)?),
                BinaryOp::Div => {
                    let b = as_exact_rational(r)?;
                    if b.is_zero() {
                        None
                    } else {
                        Some(a * b.reciprocal().expect("nonzero"))
                    }
                }
                BinaryOp::Pow => {
                    let k = as_exact_integer(r)?;
                    if k.abs() > Integer::from(4096) {
                        return None;
                    }
                    let ku = k.abs().to_i64() as usize;
                    let mut out = Rational::one();
                    for _ in 0..ku {
                        out = out * a.clone();
                    }
                    if k < Integer::from(0) {
                        out.reciprocal().ok()
                    } else {
                        Some(out)
                    }
                }
                BinaryOp::Mod => None,
            }
        }
        _ => None,
    }
}

/// Exact integer value of a constant expression; `None` if not an integer.
pub(crate) fn as_exact_integer(e: &Expr) -> Option<Integer> {
    let r = as_exact_rational(e)?;
    if r.is_integer() {
        Some(r.numerator().clone())
    } else {
        None
    }
}

/// Normalize an expression to a rational function `num/den` over ℚ in `var`.
///
/// # Contract
///
/// On success the pair is **canonical**: `gcd(num, den) = 1`, `den` monic
/// (hence nonzero); the zero function is `(0, 1)`. Two expressions denote the
/// same rational function iff their canonical pairs are equal. Negative
/// integer powers of `var` land in the denominator.
///
/// # Refusals (`None`)
///
/// See the module docs: `Expr::Real` constants, free symbols other than
/// `var`, `var` under any non-polynomial operation, non-integer or oversized
/// exponents (`|k| > 4096`), opaque non-rational constants (`sin(2)`,
/// `2^(1/2)`), and division by an identically-zero expression. Exception:
/// `0·f` normalizes to zero for non-rational but generically defined `f`
/// (see the module docs on the generic-identity convention).
pub fn as_rational_function(e: &Expr, var: &Symbol) -> Option<QFrac> {
    match e {
        Expr::Integer(n) => Some((QP::constant(Rational::from_integer(n.clone())), QP::one())),
        Expr::Rational(r) => Some((QP::constant(r.clone()), QP::one())),
        // f64 is not exact arithmetic: refuse rather than guess.
        Expr::Real(_) => None,
        Expr::Symbol(s) if s == var => Some((QP::var(), QP::one())),
        // Another free symbol: multivariate is out of scope.
        Expr::Symbol(_) => None,
        Expr::Unary(UnaryOp::Neg, inner) => {
            let (n, d) = as_rational_function(inner, var)?;
            Some((-n, d))
        }
        // sin, sqrt, log, ...: not rational, even on constant arguments
        // (sin(2) is a transcendental constant, not an element of ℚ).
        Expr::Unary(_, _) | Expr::Function(_, _) => None,
        Expr::Binary(op, l, r) => match op {
            BinaryOp::Add | BinaryOp::Sub => {
                let (nl, dl) = as_rational_function(l, var)?;
                let (nr, dr) = as_rational_function(r, var)?;
                let nr = if *op == BinaryOp::Sub { -nr } else { nr };
                reduce(nl * dr.clone() + nr * dl.clone(), dl * dr)
            }
            BinaryOp::Mul => {
                let left = as_rational_function(l, var);
                let right = as_rational_function(r, var);
                match (left, right) {
                    (Some((nl, dl)), Some((nr, dr))) => reduce(nl * nr, dl * dr),
                    // 0·f = 0 under the generic-identity convention, even when
                    // f itself is not rational (e.g. 0·log(x), which is what
                    // `differentiate` leaves behind for d/dx(c·log(v))). Only
                    // allowed when f is generically defined: a factor with an
                    // identically-zero denominator is refused, not annihilated.
                    (Some((nl, _)), None) if nl.is_zero() && generically_defined(r, var) => {
                        Some((QP::zero(), QP::one()))
                    }
                    (None, Some((nr, _))) if nr.is_zero() && generically_defined(l, var) => {
                        Some((QP::zero(), QP::one()))
                    }
                    _ => None,
                }
            }
            BinaryOp::Div => {
                let (nl, dl) = as_rational_function(l, var)?;
                let (nr, dr) = as_rational_function(r, var)?;
                if nr.is_zero() {
                    // division by the identically-zero rational function
                    return None;
                }
                reduce(nl * dr, dl * nr)
            }
            BinaryOp::Pow => {
                let k = as_exact_integer(r)?;
                // Resource guard: refuse absurd exponents instead of building
                // a polynomial with millions of coefficients.
                if k.abs() > Integer::from(4096) {
                    return None;
                }
                let (nb, db) = as_rational_function(l, var)?;
                let ku = k.abs().to_i64() as usize;
                if k < Integer::from(0) {
                    if nb.is_zero() {
                        return None; // 0^(negative)
                    }
                    reduce(poly_pow(&db, ku), poly_pow(&nb, ku))
                } else {
                    reduce(poly_pow(&nb, ku), poly_pow(&db, ku))
                }
            }
            BinaryOp::Mod => None,
        },
    }
}

// ------------------------------------------------------------------------ //
// Part B: Hermite reduction
// ------------------------------------------------------------------------ //

/// Outcome of Hermite reduction of `num/den`.
struct HermiteParts {
    /// Polynomial part of the integrand (still to be integrated termwise).
    poly_part: QP,
    /// Already-integrated rational part `g` of the antiderivative.
    rational_part: QFrac,
    /// Remaining log integrand `A₁/D*`: proper, canonical, `D*` squarefree.
    /// `A₁ = 0` iff there is no logarithmic part.
    log_part: QFrac,
}

/// Hermite reduction: `num/den = poly_part + d/dx(rational_part) + log_part`
/// with `log_part` proper over a squarefree denominator.
///
/// Contract: `(num, den)` canonical (as produced by [`as_rational_function`]).
/// The defining identity is re-verified exactly before returning (assert, on
/// in release). `Err` is a budget trip (see [`RischBudget`]): the structural
/// work measure `Σ (m − 1)·deg(den)` is checked up front, and the loop state
/// is checked against the coefficient-bit ceiling every iteration.
fn hermite_reduce(num: &QP, den: &QP, budget: &RischBudget) -> Result<HermiteParts, BudgetTrip> {
    // Structural pre-cap: the loop below runs at most Σ (m − 1) iterations
    // (each pass drops the multiplicity of the chosen repeated factor by
    // one), on operands of degree ≤ deg(den) whose coefficients swell fast.
    let work = den
        .squarefree_decomposition()
        .iter()
        .filter(|(f, m)| *m >= 2 && f.degree().unwrap_or(0) >= 1)
        .map(|(f, m)| (m - 1) * f.degree().unwrap_or(0))
        .sum::<usize>()
        * den.degree().unwrap_or(0);
    if work > budget.max_hermite_work {
        return Err(BudgetTrip {
            stage: "hermite_reduce",
            reason: format!(
                "repeated-factor work measure {} exceeds max_hermite_work {}",
                work, budget.max_hermite_work
            ),
        });
    }

    let (poly_part, rem) = num.quo_rem(den);
    let (mut a, mut d) = reduce(rem, den.clone()).expect("hermite: den nonzero");
    let mut g = frac_zero();

    while !a.is_zero() {
        check_bits(
            qp_bits(&a) + qp_bits(&d) + frac_bits(&g),
            budget,
            "hermite_reduce",
            "Hermite loop state",
        )?;
        // Locate a repeated factor V^n (n ≥ 2) of D; done if none remain.
        let repeated = d
            .squarefree_decomposition()
            .into_iter()
            .filter(|(f, m)| *m >= 2 && f.degree().unwrap_or(0) >= 1)
            .max_by_key(|(_, m)| *m);
        let Some((v, n)) = repeated else { break };
        let v = monic(&v);
        let u = exact_div(&d, &poly_pow(&v, n));

        // Bezout split: A = B·(U·V') + C·V with deg B < deg V, using
        // gcd(U·V', V) = 1 (V squarefree and coprime to U).
        let w = u.clone() * v.derivative();
        let (g0, s, _) = ext_gcd(&w, &v);
        assert_eq!(g0.degree(), Some(0), "hermite: U·V' and V must be coprime");
        let g0inv = g0.coefficients()[0]
            .reciprocal()
            .expect("hermite: nonzero gcd");
        let (_, b) = (a.clone() * s.scalar_mul(&g0inv)).quo_rem(&v);
        let c = exact_div(&(a.clone() - b.clone() * w.clone()), &v);

        // B·V'/V^n = d/dx(−B/((n−1)·V^{n−1})) + B'/((n−1)·V^{n−1}), so
        //   A/D  =  d/dx(−B/((n−1)·V^{n−1}))  +  (B'·U/(n−1) + C)/(U·V^{n−1}).
        let inv_nm1 = qr((n - 1) as i64).reciprocal().expect("n ≥ 2");
        let vnm1 = poly_pow(&v, n - 1);
        let term = reduce((-b.clone()).scalar_mul(&inv_nm1), vnm1.clone())
            .expect("hermite: V^{n-1} nonzero");
        g = frac_add(&g, &term);

        let new_a = (b.derivative() * u.clone()).scalar_mul(&inv_nm1) + c;
        let new_d = u * vnm1;
        let (ra, rd) = reduce(new_a, new_d).expect("hermite: denominator nonzero");
        a = ra;
        d = rd;
    }

    let log_part = reduce(a, d).expect("hermite: denominator nonzero");

    // THE HERMITE GATE (on in release): the decomposition must reproduce the
    // input exactly as a rational function.
    let mut lhs = frac_derivative(&g);
    lhs = frac_add(&lhs, &(poly_part.clone(), QP::one()));
    lhs = frac_add(&lhs, &log_part);
    let rhs = reduce(num.clone(), den.clone()).expect("hermite: den nonzero");
    assert!(
        frac_eq(&lhs, &rhs),
        "Hermite gate failed: d/dx(g) + poly + log integrand != input \
         (got {}/{}, want {}/{})",
        lhs.0,
        lhs.1,
        rhs.0,
        rhs.1
    );

    Ok(HermiteParts {
        poly_part,
        rational_part: g,
        log_part,
    })
}

// ------------------------------------------------------------------------ //
// Part C: Rothstein–Trager
// ------------------------------------------------------------------------ //

/// One term of the logarithmic part of the antiderivative.
#[derive(Debug, Clone, PartialEq)]
enum LogTerm {
    /// `c · log(v(x))`, `c ∈ ℚ`, `v` monic.
    RatLog { c: Rational, v: QP },
    /// `a·log(P² + Q²) − 2b·arctan(Q/P)`: the real form of
    /// `(a+bi)·log(P+iQ) + (a−bi)·log(P−iQ)` for a conjugate residue pair
    /// `a ± bi` with `a, b ∈ ℚ`, `b > 0`. `P` is monic (it carries the
    /// leading coefficient of the monic gcd), `Q ≠ 0`.
    AtanLog { a: Rational, b: Rational, p: QP, q: QP },
    /// `Σ_{c : f(c)=0} c · log(v(x, c))`: `f ∈ ℤ[t]` primitive with positive
    /// leading coefficient, irreducible over ℚ, `deg f ≥ 2`; `v[i]` is the
    /// coefficient of `x^i` as a polynomial in `t` reduced mod `f`; `v` is
    /// monic in `x`.
    RootSum { f: QP, v: Vec<QP> },
}

/// Scale a ℚ[t] polynomial to a primitive ℤ[t] polynomial with positive
/// leading coefficient (roots unchanged).
fn to_integer_primitive(p: &QP) -> UnivariatePolynomial<Integer> {
    let mut denom_lcm = Integer::from(1);
    for c in p.coefficients() {
        denom_lcm = denom_lcm.lcm(c.denominator());
    }
    let ints: Vec<Integer> = p
        .coefficients()
        .iter()
        .map(|c| {
            let scaled = c.clone() * Rational::from_integer(denom_lcm.clone());
            assert!(scaled.is_integer(), "clearing denominators must be exact");
            scaled.numerator().clone()
        })
        .collect();
    let mut content = Integer::from(0);
    for c in &ints {
        content = content.gcd(c);
    }
    let lc_negative = ints.last().map(|c| *c < Integer::from(0)).unwrap_or(false);
    if lc_negative {
        content = -content;
    }
    UnivariatePolynomial::new(
        ints.into_iter()
            .map(|c| {
                let (quo, rem) = c.div_rem(&content).expect("content nonzero");
                assert!(rem.is_zero());
                quo
            })
            .collect(),
    )
}

fn int_to_qpoly(p: &UnivariatePolynomial<Integer>) -> QP {
    QP::new(
        p.coefficients()
            .iter()
            .map(|c| Rational::from_integer(c.clone()))
            .collect(),
    )
}

// --- Arithmetic in K = ℚ[t]/(f), f monic irreducible; elements are ℚ[t]
// --- polynomials of degree < deg f.

fn k_red(a: QP, fm: &QP) -> QP {
    a.quo_rem(fm).1
}

fn k_mul(a: &QP, b: &QP, fm: &QP) -> QP {
    k_red(a.clone() * b.clone(), fm)
}

fn k_inv(a: &QP, fm: &QP) -> QP {
    let (g, s, _) = ext_gcd(a, fm);
    assert_eq!(
        g.degree(),
        Some(0),
        "k_inv: f must be irreducible and a nonzero mod f"
    );
    let ginv = g.coefficients()[0].reciprocal().expect("nonzero gcd");
    k_red(s.scalar_mul(&ginv), fm)
}

// --- K[x] polynomials as Vec<QP> (index = x-degree, entry = element of K).

fn kp_trim(mut v: Vec<QP>) -> Vec<QP> {
    while v.last().is_some_and(|c| c.is_zero()) {
        v.pop();
    }
    v
}

fn kp_rem(a: &[QP], b: &[QP], fm: &QP, budget: &RischBudget) -> Result<Vec<QP>, BudgetTrip> {
    let db = b.len() - 1;
    let lead_inv = k_inv(&b[db], fm);
    let mut r = kp_trim(a.to_vec());
    while r.len() > db {
        check_bits(
            r.iter().map(qp_bits).sum(),
            budget,
            "rothstein_trager",
            "K[x] division remainder",
        )?;
        let dr = r.len() - 1;
        let coef = k_mul(&r[dr], &lead_inv, fm);
        for j in 0..=db {
            r[dr - db + j] = k_red(
                r[dr - db + j].clone() - k_mul(&coef, &b[j], fm),
                fm,
            );
        }
        r = kp_trim(r);
    }
    Ok(r)
}

/// `gcd(A₁ − t·D*', D*)` in `(ℚ[t]/(fm))[x]`, monic in `x`.
fn kx_gcd(
    a1: &QP,
    dd: &QP,
    dstar: &QP,
    fm: &QP,
    budget: &RischBudget,
) -> Result<Vec<QP>, BudgetTrip> {
    let dmax = a1.degree().unwrap_or(0).max(dd.degree().unwrap_or(0));
    let first: Vec<QP> = (0..=dmax)
        .map(|i| QP::new(vec![coeff_at(a1, i), -coeff_at(dd, i)]))
        .collect();
    let second: Vec<QP> = dstar
        .coefficients()
        .iter()
        .map(|c| QP::constant(c.clone()))
        .collect();
    let mut r0 = kp_trim(first);
    let mut r1 = kp_trim(second);
    while !r1.is_empty() {
        let r2 = kp_rem(&r0, &r1, fm, budget)?;
        r0 = r1;
        r1 = r2;
    }
    assert!(!r0.is_empty(), "kx_gcd: gcd of nonzero polynomials");
    let lead_inv = k_inv(r0.last().unwrap(), fm);
    Ok(r0.iter().map(|c| k_mul(c, &lead_inv, fm)).collect())
}

/// Rothstein–Trager on a proper canonical fraction `a1/dstar` with `dstar`
/// squarefree of degree ≥ 1 and `a1 ≠ 0`. Returns the log terms whose exact
/// derivatives sum to `a1/dstar` — an identity that is re-verified before
/// returning (assert, on in release; `root_sum` terms are differentiated
/// exactly via traces in ℚ(x)[t]/(f)). `Err` is a budget trip (see
/// [`RischBudget`]): `deg D*` is pre-capped before the resultant is
/// computed, every irreducible resultant factor entering the `ℚ[t]/(f)`
/// tower is pre-capped by its degree, and all remainder-sequence/trace loops
/// run under the coefficient-bit ceiling.
fn rothstein_trager(a1: &QP, dstar: &QP, budget: &RischBudget) -> Result<Vec<LogTerm>, BudgetTrip> {
    let n = dstar.degree().expect("dstar nonzero");
    if n > budget.max_log_degree {
        return Err(BudgetTrip {
            stage: "rothstein_trager",
            reason: format!(
                "squarefree log-part denominator has degree {}, exceeding max_log_degree {}",
                n, budget.max_log_degree
            ),
        });
    }
    let dd = dstar.derivative();

    // R(t) = Res_X(A₁(X) − t·D*'(X), D*(X)) via the evaluation–interpolation
    // resultant. Grids are X-major: grid[i][j] = coefficient of X^i t^j.
    let dmax = a1.degree().unwrap_or(0).max(dd.degree().unwrap_or(0));
    let f_grid: Vec<Vec<Rational>> = (0..=dmax)
        .map(|i| vec![coeff_at(a1, i), -coeff_at(&dd, i)])
        .collect();
    let g_grid: Vec<Vec<Rational>> = dstar
        .coefficients()
        .iter()
        .map(|c| vec![c.clone()])
        .collect();
    let rt = QP::new(resultant_in_t(&f_grid, &g_grid));
    // With D* monic, R(t) = ±Π_{D*(β)=0}(A₁(β) − t·D*'(β)); its degree is
    // exactly deg D* because D* is squarefree (no D*'(β) vanishes).
    assert_eq!(
        rt.degree(),
        Some(n),
        "Rothstein–Trager: resultant degree must equal deg D*"
    );

    let rsf = monic(&exact_div(&rt, &rt.gcd(&rt.derivative())));
    let rz = to_integer_primitive(&rsf);
    let factors = factor_over_integers(&rz).expect("Zassenhaus factorization");

    let mut terms = Vec::new();
    for (fac, mult) in factors {
        let Some(fdeg) = fac.degree() else { continue };
        if fdeg == 0 {
            continue; // integer content
        }
        assert_eq!(mult, 1, "squarefree R must factor with multiplicity 1");
        if fdeg == 1 {
            // root c = -γ/α of α·t + γ
            let c = Rational::new(-fac.coefficients()[0].clone(), fac.coefficients()[1].clone())
                .expect("linear factor has nonzero leading coefficient");
            let cand = a1.clone() - dd.scalar_mul(&c);
            let v = monic(&cand.gcd(dstar));
            assert!(
                v.degree().unwrap_or(0) >= 1,
                "root of R must yield a nontrivial log argument"
            );
            terms.push(LogTerm::RatLog { c, v });
            continue;
        }

        // Structural pre-cap on the ℚ[t]/(f) tower before any arithmetic in
        // it: the K[x] gcd and the trace gate both swell super-exponentially
        // in deg f (measured: deg 5 ≈ 2 ms, deg 10 ≈ 3.5 s, deg 15 > 500 s).
        if fdeg > budget.max_root_sum_degree {
            return Err(BudgetTrip {
                stage: "rothstein_trager",
                reason: format!(
                    "irreducible resultant factor of degree {} exceeds max_root_sum_degree {}",
                    fdeg, budget.max_root_sum_degree
                ),
            });
        }

        // v(x, t) = gcd(A₁ − t·D*', D*) over ℚ[t]/(f), monic in x.
        let fm = monic(&int_to_qpoly(&fac));
        let v = kx_gcd(a1, &dd, dstar, &fm, budget)?;
        assert!(v.len() >= 2, "root of R must yield a nontrivial log argument");

        if fdeg == 2 {
            // α·t² + β·t + γ with conjugate roots a ± bi, a,b ∈ ℚ, iff the
            // discriminant is negative and 4αγ − β² is a perfect square.
            let gamma = fac.coefficients()[0].clone();
            let beta = fac.coefficients()[1].clone();
            let alpha = fac.coefficients()[2].clone();
            let disc = beta.clone() * beta.clone()
                - Integer::from(4) * alpha.clone() * gamma.clone();
            if disc < Integer::from(0) {
                let s = -disc;
                let r = s.sqrt().expect("s > 0");
                if r.clone() * r.clone() == s {
                    let two_alpha = Integer::from(2) * alpha.clone();
                    let a = Rational::new(-beta.clone(), two_alpha.clone())
                        .expect("alpha nonzero");
                    let b = Rational::new(r, two_alpha).expect("alpha nonzero").abs();
                    // Evaluate v at t = a + bi. Coefficients of v are linear
                    // in t (reduced mod a quadratic): u₀ + u₁·t evaluates to
                    // (u₀ + u₁·a) + (u₁·b)·i.
                    let p = QP::new(
                        v.iter()
                            .map(|ci| coeff_at(ci, 0) + coeff_at(ci, 1) * a.clone())
                            .collect(),
                    );
                    let q = QP::new(
                        v.iter().map(|ci| coeff_at(ci, 1) * b.clone()).collect(),
                    );
                    // v is monic in x, so P is monic; Q ≡ 0 would force the
                    // coprime gcds at c and c̄ to coincide — impossible.
                    assert!(!q.is_zero(), "conjugate residues: imaginary part of v");
                    terms.push(LogTerm::AtanLog { a, b, p, q });
                    continue;
                }
            }
        }

        terms.push(LogTerm::RootSum {
            f: int_to_qpoly(&fac),
            v,
        });
    }

    // THE ROTHSTEIN–TRAGER GATE (on in release): the exact derivatives of the
    // emitted terms must sum to the integrand. The gate itself runs under the
    // budget (the trace machinery can swell too); a trip here refuses the
    // whole computation — an uncertified answer is never emitted.
    let mut sum = frac_zero();
    for term in &terms {
        sum = frac_add(&sum, &log_term_derivative(term, budget)?);
    }
    let want = reduce(a1.clone(), dstar.clone()).expect("dstar nonzero");
    assert!(
        frac_eq(&sum, &want),
        "Rothstein–Trager gate failed: Σ d/dx(term) = {}/{}, want {}/{}",
        sum.0,
        sum.1,
        want.0,
        want.1
    );

    Ok(terms)
}

/// Exact derivative of a log term as a rational function of `x`. `Err` is a
/// budget trip from the `root_sum` trace machinery.
fn log_term_derivative(term: &LogTerm, budget: &RischBudget) -> Result<QFrac, BudgetTrip> {
    match term {
        // d/dx c·log(v) = c·v'/v
        LogTerm::RatLog { c, v } => {
            Ok(reduce(v.derivative().scalar_mul(c), v.clone()).expect("v nonzero"))
        }
        // d/dx [a·log(P²+Q²) − 2b·atan(Q/P)]
        //   = [2a(PP'+QQ') − 2b(Q'P − QP')] / (P²+Q²)
        LogTerm::AtanLog { a, b, p, q } => {
            let two_a = qr(2) * a.clone();
            let two_b = qr(2) * b.clone();
            let num = (p.clone() * p.derivative() + q.clone() * q.derivative())
                .scalar_mul(&two_a)
                - (q.derivative() * p.clone() - q.clone() * p.derivative()).scalar_mul(&two_b);
            let den = p.clone() * p.clone() + q.clone() * q.clone();
            Ok(reduce(num, den).expect("P²+Q² nonzero"))
        }
        LogTerm::RootSum { f, v } => root_sum_derivative(&monic(f), v, budget),
    }
}

// --- Exact differentiation of a root_sum via traces in ℚ(x)[t]/(f).
//
// d/dx Σ_c c·log(v(x,c)) = Σ_c c·v_x(x,c)/v(x,c) = Tr_{ℚ(x)[t]/(f) / ℚ(x)}(α)
// with α = t·v_x·v⁻¹ mod f: the summands are exactly the conjugates of α
// under t ↦ c, and the trace is Σ_k w_k(x)·p_k where α = Σ_k w_k t^k and
// p_k = Tr(t^k) are the power sums of the roots of f (Newton's identities).

/// Polynomials in `t` over the field ℚ(x); index = t-degree.
type TPoly = Vec<QFrac>;

fn tp_trim(mut v: TPoly) -> TPoly {
    while v.last().is_some_and(|c| c.0.is_zero()) {
        v.pop();
    }
    v
}

fn tp_mul(a: &[QFrac], b: &[QFrac]) -> TPoly {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![frac_zero(); a.len() + b.len() - 1];
    for (i, ca) in a.iter().enumerate() {
        if ca.0.is_zero() {
            continue;
        }
        for (j, cb) in b.iter().enumerate() {
            out[i + j] = frac_add(&out[i + j], &frac_mul(ca, cb));
        }
    }
    tp_trim(out)
}

fn tp_sub(a: &[QFrac], b: &[QFrac]) -> TPoly {
    let mut out = a.to_vec();
    if out.len() < b.len() {
        out.resize(b.len(), frac_zero());
    }
    for (j, cb) in b.iter().enumerate() {
        out[j] = frac_add(&out[j], &frac_scale(cb, &qr(-1)));
    }
    tp_trim(out)
}

fn tp_divrem(
    a: &[QFrac],
    b: &[QFrac],
    budget: &RischBudget,
) -> Result<(TPoly, TPoly), BudgetTrip> {
    let db = b.len() - 1;
    let lead_inv = frac_inv(&b[db]);
    let mut r = tp_trim(a.to_vec());
    let mut quo = vec![frac_zero(); r.len().saturating_sub(db)];
    while r.len() > db {
        check_bits(
            r.iter().map(frac_bits).sum(),
            budget,
            "rothstein_trager",
            "ℚ(x)[t] division remainder",
        )?;
        let dr = r.len() - 1;
        let coef = frac_mul(&r[dr], &lead_inv);
        quo[dr - db] = coef.clone();
        for j in 0..=db {
            let sub = frac_mul(&coef, &b[j]);
            r[dr - db + j] = frac_add(&r[dr - db + j], &frac_scale(&sub, &qr(-1)));
        }
        r = tp_trim(r);
    }
    Ok((tp_trim(quo), r))
}

/// Inverse of `v` in ℚ(x)[t]/(fm) (fm irreducible over ℚ, hence over ℚ(x)).
fn tp_invmod(v: &[QFrac], fm: &[QFrac], budget: &RischBudget) -> Result<TPoly, BudgetTrip> {
    let (mut r0, mut r1) = (fm.to_vec(), tp_trim(v.to_vec()));
    let (mut t0, mut t1) = (Vec::new(), vec![frac_one()]);
    while r1.len() > 1 {
        check_bits(
            r1.iter().chain(t1.iter()).map(frac_bits).sum(),
            budget,
            "rothstein_trager",
            "ℚ(x)[t]/(f) extended-Euclid state",
        )?;
        let (quo, r2) = tp_divrem(&r0, &r1, budget)?;
        let t2 = tp_sub(&t0, &tp_mul(&quo, &t1));
        r0 = r1;
        r1 = r2;
        t0 = t1;
        t1 = t2;
    }
    assert_eq!(r1.len(), 1, "tp_invmod: f irreducible, v nonzero mod f");
    let scale = frac_inv(&r1[0]);
    let scaled: TPoly = t1.iter().map(|c| frac_mul(c, &scale)).collect();
    Ok(tp_divrem(&scaled, fm, budget)?.1)
}

/// Power sums `p_0..p_upto` of the roots of monic `fm`, via Newton's
/// identities: `p_k = −k·a_{d−k} − Σ_{j=1}^{k−1} a_{d−j}·p_{k−j}` (k ≤ d).
fn power_sums(fm: &QP, upto: usize) -> Vec<Rational> {
    let d = fm.degree().expect("fm nonzero");
    assert!(upto <= d);
    let mut p = vec![qr(d as i64)];
    for k in 1..=upto {
        let mut val = qr(-(k as i64)) * coeff_at(fm, d - k);
        for j in 1..k {
            val = val - coeff_at(fm, d - j) * p[k - j].clone();
        }
        p.push(val);
    }
    p
}

/// Exact derivative `Σ_{c: fm(c)=0} c·v_x(x,c)/v(x,c)` as a rational
/// function. `Err` is a budget trip from the ℚ(x)[t]/(f) arithmetic.
fn root_sum_derivative(fm: &QP, v: &[QP], budget: &RischBudget) -> Result<QFrac, BudgetTrip> {
    let d = fm.degree().expect("fm nonzero");
    // Transpose v (x-major, coefficients in ℚ[t]) into a t-major polynomial
    // with ℚ(x) coefficients; likewise its x-derivative.
    let vt: TPoly = tp_trim(
        (0..d)
            .map(|k| {
                let coeffs: Vec<Rational> = v.iter().map(|ci| coeff_at(ci, k)).collect();
                (QP::new(coeffs), QP::one())
            })
            .collect(),
    );
    let vx: TPoly = tp_trim(
        (0..d)
            .map(|k| {
                let coeffs: Vec<Rational> = v
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(i, ci)| coeff_at(ci, k) * qr(i as i64))
                    .collect();
                (QP::new(coeffs), QP::one())
            })
            .collect(),
    );
    let fmt: TPoly = fm
        .coefficients()
        .iter()
        .map(|c| (QP::constant(c.clone()), QP::one()))
        .collect();
    let inv = tp_invmod(&vt, &fmt, budget)?;
    let tvar: TPoly = vec![frac_zero(), frac_one()];
    let prod = tp_mul(&tp_mul(&vx, &inv), &tvar);
    check_bits(
        prod.iter().map(frac_bits).sum(),
        budget,
        "rothstein_trager",
        "trace numerator t·v_x·v⁻¹",
    )?;
    let alpha = tp_divrem(&prod, &fmt, budget)?.1;
    let psums = power_sums(fm, d.saturating_sub(1));
    let mut out = frac_zero();
    for (k, w) in alpha.iter().enumerate() {
        out = frac_add(&out, &frac_scale(w, &psums[k]));
    }
    Ok(out)
}

// ------------------------------------------------------------------------ //
// Assembly into Expr
// ------------------------------------------------------------------------ //

pub(crate) fn rational_to_expr(r: &Rational) -> Expr {
    if r.is_integer() {
        Expr::Integer(r.numerator().clone())
    } else {
        Expr::Rational(r.clone())
    }
}

pub(crate) fn qpoly_to_expr(p: &QP, var: &Symbol) -> Expr {
    if p.is_zero() {
        return Expr::from(0);
    }
    let x = Expr::Symbol(var.clone());
    let mut acc: Option<Expr> = None;
    for (k, c) in p.coefficients().iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let monom = if k == 0 {
            rational_to_expr(c)
        } else {
            let xp = if k == 1 {
                x.clone()
            } else {
                x.clone().pow(Expr::from(k as i64))
            };
            if c.is_one() {
                xp
            } else {
                rational_to_expr(c) * xp
            }
        };
        acc = Some(match acc {
            None => monom,
            Some(prev) => prev + monom,
        });
    }
    acc.expect("nonzero polynomial has a nonzero coefficient")
}

/// Termwise integral of a polynomial (no constant of integration).
fn integrate_qpoly(p: &QP) -> QP {
    let mut coeffs = vec![Rational::zero()];
    for (k, c) in p.coefficients().iter().enumerate() {
        let inv = Rational::new(1i64, (k + 1) as i64).expect("k+1 > 0");
        coeffs.push(c.clone() * inv);
    }
    QP::new(coeffs)
}

/// The reserved root_sum bound symbol, guaranteed distinct from `var`.
fn root_sum_symbol(var: &Symbol) -> Symbol {
    if var.name() == "_t" {
        Symbol::new("_t0")
    } else {
        Symbol::new("_t")
    }
}

fn log_term_to_expr(term: &LogTerm, var: &Symbol) -> Expr {
    match term {
        LogTerm::RatLog { c, v } => {
            let logv = qpoly_to_expr(v, var).log();
            if c.is_one() {
                logv
            } else {
                rational_to_expr(c) * logv
            }
        }
        LogTerm::AtanLog { a, b, p, q } => {
            let w = p.clone() * p.clone() + q.clone() * q.clone();
            let atan_term = rational_to_expr(&(qr(2) * b.clone()))
                * (qpoly_to_expr(q, var) / qpoly_to_expr(p, var)).arctan();
            if a.is_zero() {
                -atan_term
            } else {
                rational_to_expr(a) * qpoly_to_expr(&w, var).log() - atan_term
            }
        }
        LogTerm::RootSum { f, v } => {
            let tau = root_sum_symbol(var);
            let x = Expr::Symbol(var.clone());
            let mut vexpr: Option<Expr> = None;
            for (i, ci) in v.iter().enumerate() {
                if ci.is_zero() {
                    continue;
                }
                let coeff = qpoly_to_expr(ci, &tau);
                let monom = if i == 0 {
                    coeff
                } else {
                    let xp = if i == 1 {
                        x.clone()
                    } else {
                        x.clone().pow(Expr::from(i as i64))
                    };
                    if ci.is_one() {
                        xp
                    } else {
                        coeff * xp
                    }
                };
                vexpr = Some(match vexpr {
                    None => monom,
                    Some(prev) => prev + monom,
                });
            }
            let summand =
                Expr::Symbol(tau.clone()) * vexpr.expect("v monic, hence nonzero").log();
            Expr::Function(
                "root_sum".to_string(),
                vec![Arc::new(qpoly_to_expr(f, &tau)), Arc::new(summand)],
            )
        }
    }
}

// ------------------------------------------------------------------------ //
// The decision procedure
// ------------------------------------------------------------------------ //

/// Result of [`integrate_rational_risch`].
#[derive(Debug, Clone, PartialEq)]
pub enum RischResult {
    /// The antiderivative, fully expressed with rational functions, `log`,
    /// and `arctan` of rational functions.
    Elementary(Expr),
    /// The antiderivative contains `root_sum` terms (log coefficients that
    /// are irrational algebraic numbers — see the module docs for the exact
    /// semantics of `Function("root_sum", ..)`). The antiderivative is still
    /// exact and complete; it is elementary as a mathematical object, we
    /// merely cannot express its constants in ℚ.
    WithRootSum(Expr),
    /// The input is not a univariate rational function of `var` over ℚ
    /// (see the honest-refusal surface in the module docs).
    NotRational,
    /// The input **is** a rational function of `var`, but integrating it
    /// would exceed the [`RischBudget`] in force (a structural pre-cap or
    /// the coefficient-bit ceiling tripped). This is a labeled *resource*
    /// refusal: the elementary antiderivative exists — we declined to pay
    /// for it, and we never mislabel that as the *mathematical* refusal
    /// [`RischResult::NotRational`]. Retry through
    /// [`integrate_rational_risch_with_budget`] with a bigger budget to pay
    /// more.
    BudgetExceeded {
        /// The pipeline stage that refused: `"hermite_reduce"` or
        /// `"rothstein_trager"`.
        stage: &'static str,
        /// Which cap tripped, with the measured value and the ceiling.
        reason: String,
    },
}

/// Decision procedure for integrating rational functions: normalize, Hermite
/// reduction, Rothstein–Trager. Both algebraic stages re-verify their
/// defining identities exactly before returning (asserts, on in release).
///
/// Runs under [`RischBudget::default`]; a tripped cap returns the labeled
/// resource refusal [`RischResult::BudgetExceeded`] (never a hang, and never
/// a mislabeled [`RischResult::NotRational`]). Use
/// [`integrate_rational_risch_with_budget`] to pay more.
pub fn integrate_rational_risch(e: &Expr, var: &Symbol) -> RischResult {
    integrate_rational_risch_with_budget(e, var, &RischBudget::default())
}

/// [`integrate_rational_risch`] under a caller-chosen [`RischBudget`].
///
/// The mathematics is identical; only the refusal threshold moves. Within
/// the budget the result is exactly what the default entry point would
/// produce; beyond it, [`RischResult::BudgetExceeded`] reports the stage and
/// the cap that tripped.
pub fn integrate_rational_risch_with_budget(
    e: &Expr,
    var: &Symbol,
    budget: &RischBudget,
) -> RischResult {
    let Some((num, den)) = as_rational_function(e, var) else {
        return RischResult::NotRational;
    };
    let parts = match hermite_reduce(&num, &den, budget) {
        Ok(parts) => parts,
        Err(trip) => {
            return RischResult::BudgetExceeded {
                stage: trip.stage,
                reason: trip.reason,
            }
        }
    };
    let (a1, dstar) = parts.log_part.clone();
    let terms = if a1.is_zero() {
        Vec::new()
    } else {
        match rothstein_trager(&a1, &dstar, budget) {
            Ok(terms) => terms,
            Err(trip) => {
                return RischResult::BudgetExceeded {
                    stage: trip.stage,
                    reason: trip.reason,
                }
            }
        }
    };

    let mut acc: Vec<Expr> = Vec::new();
    let poly_integral = integrate_qpoly(&parts.poly_part);
    if !poly_integral.is_zero() {
        acc.push(qpoly_to_expr(&poly_integral, var));
    }
    if !parts.rational_part.0.is_zero() {
        acc.push(
            qpoly_to_expr(&parts.rational_part.0, var)
                / qpoly_to_expr(&parts.rational_part.1, var),
        );
    }
    let has_root_sum = terms.iter().any(|t| matches!(t, LogTerm::RootSum { .. }));
    for term in &terms {
        acc.push(log_term_to_expr(term, var));
    }
    let expr = acc
        .into_iter()
        .reduce(|a, b| a + b)
        .unwrap_or_else(|| Expr::from(0));
    if has_root_sum {
        RischResult::WithRootSum(expr)
    } else {
        RischResult::Elementary(expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xsym() -> Symbol {
        Symbol::new("x")
    }

    fn xexpr() -> Expr {
        Expr::Symbol(xsym())
    }

    fn qp(coeffs: &[(i64, i64)]) -> QP {
        QP::new(
            coeffs
                .iter()
                .map(|(n, d)| Rational::new(*n, *d).unwrap())
                .collect(),
        )
    }

    fn qpz(coeffs: &[i64]) -> QP {
        QP::new(coeffs.iter().map(|n| qr(*n)).collect())
    }

    // The battery below predates the budget plumbing and must stay literally
    // unchanged; these default-budget wrappers shadow the budgeted internals
    // (explicit items win over the `use super::*` glob). Every battery input
    // fitting comfortably inside the default budget is itself one of the
    // gates: the budget must never trigger on the correctness battery.
    fn hermite_reduce(num: &QP, den: &QP) -> HermiteParts {
        super::hermite_reduce(num, den, &RischBudget::default())
            .expect("battery input must fit the default budget")
    }

    fn rothstein_trager(a1: &QP, dstar: &QP) -> Vec<LogTerm> {
        super::rothstein_trager(a1, dstar, &RischBudget::default())
            .expect("battery input must fit the default budget")
    }

    fn root_sum_derivative(fm: &QP, v: &[QP]) -> QFrac {
        super::root_sum_derivative(fm, v, &RischBudget::default())
            .expect("battery input must fit the default budget")
    }

    /// THE GATE: differentiate the answer with the existing differentiate(),
    /// subtract the input, and normalize through as_rational_function — the
    /// difference must be exactly (0, 1). No floating point anywhere.
    fn assert_exact_antiderivative(input: &Expr, antiderivative: &Expr, var: &Symbol) {
        let diff = antiderivative.differentiate(var) - input.clone();
        let (n, d) = as_rational_function(&diff, var)
            .expect("gate: F' - f must normalize as a rational function");
        assert!(
            n.is_zero() && d == QP::one(),
            "gate failed: F' - f = {} / {}, expected 0",
            n,
            d
        );
    }

    fn integrate_elementary_gated(input: &Expr) -> Expr {
        match integrate_rational_risch(input, &xsym()) {
            RischResult::Elementary(f) => {
                assert_exact_antiderivative(input, &f, &xsym());
                f
            }
            other => panic!("expected Elementary, got {:?}", other),
        }
    }

    // ---------------- normalizer ----------------

    #[test]
    fn normalizer_reduces_and_makes_denominator_monic() {
        let x = xexpr();
        // (x^2 - 1) / (x - 1) => x + 1
        let e = (x.clone().pow(Expr::from(2)) - Expr::from(1)) / (x.clone() - Expr::from(1));
        let (n, d) = as_rational_function(&e, &xsym()).unwrap();
        assert_eq!(n, qpz(&[1, 1]));
        assert_eq!(d, QP::one());

        // 1 / (2x + 2) => (1/2) / (x + 1)
        let e = Expr::from(1) / (Expr::from(2) * x.clone() + Expr::from(2));
        let (n, d) = as_rational_function(&e, &xsym()).unwrap();
        assert_eq!(n, qp(&[(1, 2)]));
        assert_eq!(d, qpz(&[1, 1]));

        // x^(-2) => 1 / x^2
        let e = x.clone().pow(Expr::from(-2));
        let (n, d) = as_rational_function(&e, &xsym()).unwrap();
        assert_eq!(n, QP::one());
        assert_eq!(d, qpz(&[0, 0, 1]));

        // (1/2)*x + 3, exact rational constants accepted
        let e = Expr::Rational(Rational::new(1, 2).unwrap()) * x.clone() + Expr::from(3);
        let (n, d) = as_rational_function(&e, &xsym()).unwrap();
        assert_eq!(n, qp(&[(3, 1), (1, 2)]));
        assert_eq!(d, QP::one());

        // nested quotient: (x/(x+1)) / (x/(x-1)) => (x-1)/(x+1)
        let e = (x.clone() / (x.clone() + Expr::from(1)))
            / (x.clone() / (x.clone() - Expr::from(1)));
        let (n, d) = as_rational_function(&e, &xsym()).unwrap();
        assert_eq!(n, qpz(&[-1, 1]));
        assert_eq!(d, qpz(&[1, 1]));
    }

    #[test]
    fn normalizer_refuses_non_rational_inputs() {
        let x = xexpr();
        let y = Expr::symbol("y");
        let half = Expr::Rational(Rational::new(1, 2).unwrap());
        let refusals: Vec<Expr> = vec![
            x.clone().sin(),                       // var under sin
            x.clone().sqrt(),                      // var under sqrt
            x.clone().pow(half.clone()),           // fractional power of var
            x.clone().pow(y.clone()),              // symbolic exponent
            x.clone() + y.clone(),                 // second free symbol
            Expr::Real(1.5) * x.clone(),           // f64 is not exact
            Expr::from(1) / (x.clone() - x.clone()), // division by identically zero
            Expr::from(2).pow(half),               // irrational constant
            Expr::from(2).sin(),                   // transcendental constant
            Expr::Binary(
                BinaryOp::Mod,
                Arc::new(x.clone()),
                Arc::new(Expr::from(3)),
            ),
        ];
        for e in refusals {
            assert!(
                as_rational_function(&e, &xsym()).is_none(),
                "should refuse {:?}",
                e
            );
        }
    }

    // ---------------- power sums / trace machinery units ----------------

    #[test]
    fn power_sums_match_hand_computation() {
        // (t-2)(t-3) = t^2 - 5t + 6: p0 = 2, p1 = 2+3 = 5
        let fm = qpz(&[6, -5, 1]);
        assert_eq!(power_sums(&fm, 1), vec![qr(2), qr(5)]);
    }

    #[test]
    fn root_sum_derivative_case_sqrt2() {
        // f = t^2 - 1/8 (monic form of 8t^2 - 1), v = x - 4t:
        // sum over roots c = ±1/(2√2) of c/(x - 4c) = 1/(x^2 - 2).
        // Derived by hand and confirmed by sympy (see battery test below).
        let fm = qp(&[(-1, 8), (0, 1), (1, 1)]);
        let v = vec![qpz(&[0, -4]), QP::one()];
        let deriv = root_sum_derivative(&fm, &v);
        assert_eq!(deriv.0, QP::one());
        assert_eq!(deriv.1, qpz(&[-2, 0, 1]));
    }

    // ---------------- battery (expected values derived with sympy first) ----------------

    // sympy: integrate(1/(x**2+1)) = atan(x)
    #[test]
    fn battery_1_inverse_quadratic_atan() {
        let x = xexpr();
        let input = Expr::from(1) / (x.clone().pow(Expr::from(2)) + Expr::from(1));
        let f = integrate_elementary_gated(&input);
        // structure: no poly part, no rational part, single conjugate pair
        let (num, den) = as_rational_function(&input, &xsym()).unwrap();
        let parts = hermite_reduce(&num, &den);
        assert!(parts.poly_part.is_zero());
        assert!(parts.rational_part.0.is_zero());
        let terms = rothstein_trager(&parts.log_part.0, &parts.log_part.1);
        assert_eq!(terms.len(), 1);
        match &terms[0] {
            LogTerm::AtanLog { a, b, p, q } => {
                assert!(a.is_zero());
                assert_eq!(*b, Rational::new(1, 2).unwrap());
                assert_eq!(*p, qpz(&[0, 1]));
                assert_eq!(*q, QP::one());
            }
            other => panic!("expected AtanLog, got {:?}", other),
        }
        // the assembled expression contains an arctan node
        fn contains_arctan(e: &Expr) -> bool {
            match e {
                Expr::Unary(UnaryOp::Arctan, _) => true,
                Expr::Unary(_, inner) => contains_arctan(inner),
                Expr::Binary(_, l, r) => contains_arctan(l) || contains_arctan(r),
                Expr::Function(_, args) => args.iter().any(|a| contains_arctan(a)),
                _ => false,
            }
        }
        assert!(contains_arctan(&f));
    }

    // sympy: integrate(x/(x**2+1)) = log(x**2 + 1)/2
    #[test]
    fn battery_2_log_derivative() {
        let x = xexpr();
        let input = x.clone() / (x.clone().pow(Expr::from(2)) + Expr::from(1));
        integrate_elementary_gated(&input);
        let (num, den) = as_rational_function(&input, &xsym()).unwrap();
        let parts = hermite_reduce(&num, &den);
        let terms = rothstein_trager(&parts.log_part.0, &parts.log_part.1);
        assert_eq!(
            terms,
            vec![LogTerm::RatLog {
                c: Rational::new(1, 2).unwrap(),
                v: qpz(&[1, 0, 1]),
            }]
        );
    }

    // sympy: integrate(1/(x**2-2)) = sqrt(2)*log(x-sqrt(2))/4 - sqrt(2)*log(x+sqrt(2))/4.
    // The residues ±√2/4 are irrational, so the honest answer is a root_sum:
    // R(t) = 1 - 8t^2 (sympy: resultant confirms), primitive positive-lc factor
    // f = 8t^2 - 1, v = x - 4t (hand-checked: 4t ≡ 1/(2t) mod 8t^2-1).
    #[test]
    fn battery_3_sqrt2_root_sum() {
        let x = xexpr();
        let input = Expr::from(1) / (x.clone().pow(Expr::from(2)) - Expr::from(2));
        match integrate_rational_risch(&input, &xsym()) {
            RischResult::WithRootSum(_) => {}
            other => panic!("expected WithRootSum, got {:?}", other),
        }
        let (num, den) = as_rational_function(&input, &xsym()).unwrap();
        let parts = hermite_reduce(&num, &den);
        let terms = rothstein_trager(&parts.log_part.0, &parts.log_part.1);
        assert_eq!(
            terms,
            vec![LogTerm::RootSum {
                f: qpz(&[-1, 0, 8]),
                v: vec![qpz(&[0, -4]), QP::one()],
            }]
        );
        // The Rothstein–Trager gate inside rothstein_trager() already verified
        // exactly (via the trace in ℚ(x)[t]/(f)) that the term differentiates
        // back to 1/(x²-2).
    }

    // sympy: integrate(1/(x**3-x)) = -log(x) + log(x**2-1)/2
    #[test]
    fn battery_4_three_rational_residues() {
        let x = xexpr();
        let input = Expr::from(1) / (x.clone().pow(Expr::from(3)) - x.clone());
        integrate_elementary_gated(&input);
        let (num, den) = as_rational_function(&input, &xsym()).unwrap();
        let parts = hermite_reduce(&num, &den);
        let mut terms = rothstein_trager(&parts.log_part.0, &parts.log_part.1);
        terms.sort_by(|s, t| match (s, t) {
            (LogTerm::RatLog { c: c1, .. }, LogTerm::RatLog { c: c2, .. }) => c1.cmp(c2),
            _ => panic!("expected only RatLog terms"),
        });
        assert_eq!(
            terms,
            vec![
                LogTerm::RatLog {
                    c: qr(-1),
                    v: qpz(&[0, 1]),
                },
                LogTerm::RatLog {
                    c: Rational::new(1, 2).unwrap(),
                    v: qpz(&[-1, 0, 1]),
                },
            ]
        );
    }

    // sympy: integrate((x**4+1)/(x**2+1)) = x**3/3 - x + 2*atan(x)
    #[test]
    fn battery_5_polynomial_part_plus_atan() {
        let x = xexpr();
        let input = (x.clone().pow(Expr::from(4)) + Expr::from(1))
            / (x.clone().pow(Expr::from(2)) + Expr::from(1));
        integrate_elementary_gated(&input);
        let (num, den) = as_rational_function(&input, &xsym()).unwrap();
        let parts = hermite_reduce(&num, &den);
        assert_eq!(parts.poly_part, qpz(&[-1, 0, 1])); // x^2 - 1
        assert!(parts.rational_part.0.is_zero());
        let terms = rothstein_trager(&parts.log_part.0, &parts.log_part.1);
        assert_eq!(terms.len(), 1);
        match &terms[0] {
            LogTerm::AtanLog { a, b, .. } => {
                assert!(a.is_zero());
                assert_eq!(*b, qr(1)); // -2b·atan(1/x) with b=1 ⇒ derivative 2/(x²+1)
            }
            other => panic!("expected AtanLog, got {:?}", other),
        }
    }

    // sympy: integrate(1/(x**2+1)**2) = x/(2*x**2+2) + atan(x)/2
    // Hermite rational part (canonical): (x/2)/(x^2+1)
    #[test]
    fn battery_6_hermite_rational_part() {
        let x = xexpr();
        let input =
            Expr::from(1) / (x.clone().pow(Expr::from(2)) + Expr::from(1)).pow(Expr::from(2));
        integrate_elementary_gated(&input);
        let (num, den) = as_rational_function(&input, &xsym()).unwrap();
        let parts = hermite_reduce(&num, &den);
        assert_eq!(parts.rational_part, (qp(&[(0, 1), (1, 2)]), qpz(&[1, 0, 1])));
        // remaining log integrand (1/2)/(x^2+1)
        assert_eq!(parts.log_part, (qp(&[(1, 2)]), qpz(&[1, 0, 1])));
    }

    // sympy: integrate((3*x**2+1)/(x**3+x+1)) = log(x**3+x+1)
    #[test]
    fn battery_7_pure_log_derivative() {
        let x = xexpr();
        let input = (Expr::from(3) * x.clone().pow(Expr::from(2)) + Expr::from(1))
            / (x.clone().pow(Expr::from(3)) + x.clone() + Expr::from(1));
        integrate_elementary_gated(&input);
        let (num, den) = as_rational_function(&input, &xsym()).unwrap();
        let parts = hermite_reduce(&num, &den);
        let terms = rothstein_trager(&parts.log_part.0, &parts.log_part.1);
        assert_eq!(
            terms,
            vec![LogTerm::RatLog {
                c: qr(1),
                v: qpz(&[1, 1, 0, 1]),
            }]
        );
    }

    // sympy: integrate(1/(x**5-x-1)) =
    //   RootSum(2869*t**5 + 160*t**3 - 80*t**2 + 15*t - 1,
    //           Lambda(t, t*log(x - 183616*t**4/625 - 45904*t**3/625
    //                             - 21716*t**2/625 - 309*t/625 - 256/625)))
    // (independently re-derived: R(t) = Res_X(1 - t(5X^4-1), X^5-X-1) is
    //  -(2869 t^5 + 160 t^3 - 80 t^2 + 15 t - 1), irreducible over Q, and the
    //  gcd v(x,t) above satisfies D*(x)|_{x=root} = 0 for every root of R.)
    #[test]
    fn battery_8_quintic_root_sum() {
        let x = xexpr();
        let input =
            Expr::from(1) / (x.clone().pow(Expr::from(5)) - x.clone() - Expr::from(1));
        match integrate_rational_risch(&input, &xsym()) {
            RischResult::WithRootSum(_) => {}
            other => panic!("expected WithRootSum, got {:?}", other),
        }
        let (num, den) = as_rational_function(&input, &xsym()).unwrap();
        let parts = hermite_reduce(&num, &den);
        let terms = rothstein_trager(&parts.log_part.0, &parts.log_part.1);
        let expected_v0 = qp(&[
            (-256, 625),
            (-309, 625),
            (-21716, 625),
            (-45904, 625),
            (-183616, 625),
        ]);
        assert_eq!(
            terms,
            vec![LogTerm::RootSum {
                f: qpz(&[-1, 15, -80, 160, 0, 2869]),
                v: vec![expected_v0, QP::one()],
            }]
        );

        // Supplementary WEAKER check (numeric, f64): find the roots of f by
        // Durand–Kerner and verify Σ c/(x0 - w(c)) ≈ 1/(x0^5 - x0 - 1) at
        // sample points. The exact certification is the trace-based
        // Rothstein–Trager gate that already ran inside rothstein_trager().
        let f = [-1.0, 15.0, -80.0, 160.0, 0.0, 2869.0];
        let mut roots = [(0.4f64, 0.9f64); 5];
        for (k, r) in roots.iter_mut().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) / 5.0 + 0.5;
            *r = (0.5 * angle.cos(), 0.5 * angle.sin());
        }
        let cmul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
        let cdiv = |a: (f64, f64), b: (f64, f64)| {
            let m = b.0 * b.0 + b.1 * b.1;
            ((a.0 * b.0 + a.1 * b.1) / m, (a.1 * b.0 - a.0 * b.1) / m)
        };
        let peval = |z: (f64, f64)| {
            let mut acc = (0.0, 0.0);
            for c in f.iter().rev() {
                acc = cmul(acc, z);
                acc = (acc.0 + c, acc.1);
            }
            acc
        };
        for _ in 0..200 {
            for i in 0..5 {
                let mut denom = (f[5], 0.0);
                for j in 0..5 {
                    if i != j {
                        denom = cmul(
                            denom,
                            (roots[i].0 - roots[j].0, roots[i].1 - roots[j].1),
                        );
                    }
                }
                let delta = cdiv(peval(roots[i]), denom);
                roots[i] = (roots[i].0 - delta.0, roots[i].1 - delta.1);
            }
        }
        let w = [-256.0 / 625.0, -309.0 / 625.0, -21716.0 / 625.0, -45904.0 / 625.0,
            -183616.0 / 625.0];
        for x0 in [0.7f64, 1.3, -2.1] {
            let mut sum = (0.0, 0.0);
            for c in roots {
                let mut wc = (0.0, 0.0);
                for cf in w.iter().rev() {
                    wc = cmul(wc, c);
                    wc = (wc.0 + cf, wc.1);
                }
                // summand c / (x0 + w(c))  [v(x,c) = x + w(c)]
                let term = cdiv(c, (x0 + wc.0, wc.1));
                sum = (sum.0 + term.0, sum.1 + term.1);
            }
            let want = 1.0 / (x0.powi(5) - x0 - 1.0);
            assert!(
                (sum.0 - want).abs() < 1e-8 && sum.1.abs() < 1e-8,
                "numeric root_sum check failed at x0={}: got {:?}, want {}",
                x0,
                sum,
                want
            );
        }
    }

    // sympy: integrate(x**7/(x**4+2)) = x**4/4 - log(x**4+2)/2
    #[test]
    fn battery_9_poly_part_and_log() {
        let x = xexpr();
        let input = x.clone().pow(Expr::from(7)) / (x.clone().pow(Expr::from(4)) + Expr::from(2));
        integrate_elementary_gated(&input);
        let (num, den) = as_rational_function(&input, &xsym()).unwrap();
        let parts = hermite_reduce(&num, &den);
        assert_eq!(parts.poly_part, qpz(&[0, 0, 0, 1])); // x^3
        let terms = rothstein_trager(&parts.log_part.0, &parts.log_part.1);
        assert_eq!(
            terms,
            vec![LogTerm::RatLog {
                c: Rational::new(-1, 2).unwrap(),
                v: qpz(&[2, 0, 0, 0, 1]),
            }]
        );
    }

    // sympy: integrate(1/((x-1)**3*(x+2)**2)) =
    //   (2*x**2 - x - 4)/(18*x**3 - 54*x + 36) + log(x-1)/27 - log(x+2)/27
    // canonical rational part: ((2x^2 - x - 4)/18) / (x^3 - 3x + 2);
    // apart() gives residues 1/27 at (x-1) and -1/27 at (x+2), i.e. the log
    // integrand (1/9)/(x^2 + x - 2).
    #[test]
    fn battery_10_heavy_hermite() {
        let x = xexpr();
        let input = Expr::from(1)
            / ((x.clone() - Expr::from(1)).pow(Expr::from(3))
                * (x.clone() + Expr::from(2)).pow(Expr::from(2)));
        integrate_elementary_gated(&input);
        let (num, den) = as_rational_function(&input, &xsym()).unwrap();
        let parts = hermite_reduce(&num, &den);
        assert!(parts.poly_part.is_zero());
        assert_eq!(
            parts.rational_part,
            (qp(&[(-2, 9), (-1, 18), (1, 9)]), qpz(&[2, -3, 0, 1]))
        );
        assert_eq!(parts.log_part, (qp(&[(1, 9)]), qpz(&[-2, 1, 1])));
        let mut terms = rothstein_trager(&parts.log_part.0, &parts.log_part.1);
        terms.sort_by(|s, t| match (s, t) {
            (LogTerm::RatLog { c: c1, .. }, LogTerm::RatLog { c: c2, .. }) => c1.cmp(c2),
            _ => panic!("expected only RatLog terms"),
        });
        assert_eq!(
            terms,
            vec![
                LogTerm::RatLog {
                    c: Rational::new(-1, 27).unwrap(),
                    v: qpz(&[2, 1]),
                },
                LogTerm::RatLog {
                    c: Rational::new(1, 27).unwrap(),
                    v: qpz(&[-1, 1]),
                },
            ]
        );
    }

    // ---------------- simple and refused integrands ----------------

    #[test]
    fn integrates_polynomials_and_constants() {
        let x = xexpr();
        // sympy: integrate(x**2) = x**3/3
        let f = integrate_elementary_gated(&x.clone().pow(Expr::from(2)));
        let (n, d) = as_rational_function(&f, &xsym()).unwrap();
        assert_eq!(n, qp(&[(0, 1), (0, 1), (0, 1), (1, 3)]));
        assert_eq!(d, QP::one());
        // constants and zero
        integrate_elementary_gated(&Expr::from(5));
        assert_eq!(
            integrate_rational_risch(&Expr::from(0), &xsym()),
            RischResult::Elementary(Expr::from(0))
        );
    }

    #[test]
    fn integrates_negative_power() {
        let x = xexpr();
        // sympy: integrate(x**-2) = -1/x
        integrate_elementary_gated(&x.clone().pow(Expr::from(-2)));
    }

    #[test]
    fn refuses_non_rational_integrands() {
        let x = xexpr();
        let y = Expr::symbol("y");
        let inputs = vec![
            x.clone().sin() / x.clone(),
            x.clone().sqrt(),
            x.clone() + y.clone(),
            Expr::Real(1.5) * x.clone(),
        ];
        for e in inputs {
            assert_eq!(
                integrate_rational_risch(&e, &xsym()),
                RischResult::NotRational,
                "should refuse {:?}",
                e
            );
        }
    }

    // ---------------- resource budgets: labeled refusal, never a hang ----------------

    /// Expect a budget trip at the given stage; the refusal must be labeled
    /// BudgetExceeded — mislabeling it NotRational (the input IS rational)
    /// would be a lie, and completing would mean the cap is not working.
    fn assert_budget_refusal(input: &Expr, want_stage: &str) {
        match integrate_rational_risch(input, &xsym()) {
            RischResult::BudgetExceeded { stage, reason } => {
                assert_eq!(stage, want_stage, "wrong stage for {:?} ({})", input, reason);
            }
            other => panic!(
                "expected BudgetExceeded at {} for {:?}, got {:?}",
                want_stage, input, other
            ),
        }
    }

    // Pre-budget (adversarial audit, debug build): 1/(x^15+x+1) killed at
    // 150 s; 1/(x^40+x+1) killed at 240 s and 550 s. The Rothstein–Trager
    // caps must turn both into labeled refusals within milliseconds
    // (time-asserted generously for slow CI).
    #[test]
    fn budget_refuses_rothstein_trager_blowups_promptly() {
        let x = xexpr();
        let t0 = std::time::Instant::now();
        // deg 15: the resultant factors into an irreducible of degree > cap,
        // so the ℚ[t]/(f) tower is refused before any arithmetic in it.
        let input =
            Expr::from(1) / (x.clone().pow(Expr::from(15)) + x.clone() + Expr::from(1));
        assert_budget_refusal(&input, "rothstein_trager");
        // deg 40: refused by the deg D* pre-cap before the resultant and its
        // Zassenhaus factorization are even attempted.
        let input =
            Expr::from(1) / (x.clone().pow(Expr::from(40)) + x.clone() + Expr::from(1));
        assert_budget_refusal(&input, "rothstein_trager");
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(30),
            "budget refusal must be prompt, took {:?}",
            t0.elapsed()
        );
    }

    // Pre-budget (adversarial audit, debug build): 1/((x-1)^30(x+2)) took
    // 63 s; (x^3+1)/((x^2+1)^10(x-2)^20) was killed at 300 s. The Hermite
    // structural pre-cap must refuse both before the reduction loop starts.
    #[test]
    fn budget_refuses_hermite_blowups_promptly() {
        let x = xexpr();
        let t0 = std::time::Instant::now();
        let input = Expr::from(1)
            / ((x.clone() - Expr::from(1)).pow(Expr::from(30))
                * (x.clone() + Expr::from(2)));
        assert_budget_refusal(&input, "hermite_reduce");
        let input = (x.clone().pow(Expr::from(3)) + Expr::from(1))
            / ((x.clone().pow(Expr::from(2)) + Expr::from(1)).pow(Expr::from(10))
                * (x.clone() - Expr::from(2)).pow(Expr::from(20)));
        assert_budget_refusal(&input, "hermite_reduce");
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(30),
            "budget refusal must be prompt, took {:?}",
            t0.elapsed()
        );
    }

    #[test]
    fn with_budget_pays_more_and_agrees_with_default_inside_it() {
        let x = xexpr();
        // (a) Inside the default budget the two entry points agree exactly.
        let small = Expr::from(1) / (x.clone().pow(Expr::from(2)) + Expr::from(1));
        assert_eq!(
            integrate_rational_risch(&small, &xsym()),
            integrate_rational_risch_with_budget(&small, &xsym(), &RischBudget::default())
        );
        // (b) An input the default budget refuses succeeds — and is still
        // gated exactly — when the caller pays more. Hermite work measure
        // for 1/((x-1)^17(x+2)) is 16·18 = 288 > 200 (measured ≈ 3.5 s in
        // debug under the paid budget — affordable for one test).
        let input = Expr::from(1)
            / ((x.clone() - Expr::from(1)).pow(Expr::from(17))
                * (x.clone() + Expr::from(2)));
        assert_budget_refusal(&input, "hermite_reduce");
        let paid = RischBudget {
            max_hermite_work: 512,
            ..RischBudget::default()
        };
        match integrate_rational_risch_with_budget(&input, &xsym(), &paid) {
            RischResult::Elementary(f) => assert_exact_antiderivative(&input, &f, &xsym()),
            other => panic!("expected Elementary under the paid budget, got {:?}", other),
        }
    }

    // ---------------- the root_sum differentiation rule ----------------

    // d/dx root_sum(f, g) = root_sum(f, ∂g/∂x): the WithRootSum results that
    // escape through integrate() must NOT differentiate to 0 through the
    // generic unknown-function fallback (that was a silently wrong
    // derivative; see differentiate.rs).
    #[test]
    fn root_sum_differentiates_through_the_summand_not_to_zero() {
        let x = xexpr();
        let input = Expr::from(1) / (x.clone().pow(Expr::from(2)) - Expr::from(2));
        let f = match integrate_rational_risch(&input, &xsym()) {
            RischResult::WithRootSum(f) => f,
            other => panic!("expected WithRootSum, got {:?}", other),
        };
        let Expr::Function(name, args) = &f else {
            panic!("expected a bare root_sum, got {}", f)
        };
        assert_eq!(name, "root_sum");

        let df = f.differentiate(&xsym());
        assert_ne!(df, Expr::from(0), "root_sum derivative must not collapse to 0");
        let Expr::Function(dname, dargs) = &df else {
            panic!("expected a root_sum derivative, got {}", df)
        };
        assert_eq!(dname, "root_sum");
        assert_eq!(
            *dargs[0], *args[0],
            "the root polynomial f must pass through unchanged"
        );
        assert_eq!(
            *dargs[1],
            args[1].differentiate(&xsym()),
            "the summand must be differentiated in place"
        );
    }

    // The bound variable τ is bound: the value Σ_{c: f(c)=0} g(x,c) has no
    // free τ, so d/dτ is exactly 0 (and NOT root_sum(f, ∂g/∂τ)).
    #[test]
    fn root_sum_derivative_wrt_bound_variable_is_zero() {
        let x = xexpr();
        let input = Expr::from(1) / (x.clone().pow(Expr::from(2)) - Expr::from(2));
        let f = match integrate_rational_risch(&input, &xsym()) {
            RischResult::WithRootSum(f) => f,
            other => panic!("expected WithRootSum, got {:?}", other),
        };
        assert_eq!(f.differentiate(&Symbol::new("_t")), Expr::from(0));
    }
}
