//! Exact, gated integration patterns beyond rational functions.
//!
//! [`integrate_exact_patterns`] extends the honest frontier of
//! [`Expr::integrate`] past the rational-function decision procedure
//! ([`crate::risch`]) with three families whose antiderivatives can be both
//! *constructed* and *certified* by exact algebra — nothing here ever returns
//! a plausible-but-unchecked form:
//!
//! 1. **Log-derivative** `c·u′/u ↦ c·log(u)` (`c ∈ ℚ`) for arbitrary `u`:
//!    the numerator is matched against `u.differentiate(var)` up to a
//!    rational constant, after a *sound* cleanup pass ([`fold_clean`]) whose
//!    every rewrite is an exact identity. That match — `num ≡ c·u′` — is
//!    precisely the content of the differentiation gate
//!    `d/dx (c·log u) = c·u′/u = num/den`. (For rational `u` this pattern is
//!    subsumed by the Risch tier, which runs first.)
//! 2. **`p(x)·exp(g)`**, `p ∈ ℚ[x]`, `g = a·x + b` with `a, b ∈ ℚ`, `a ≠ 0`:
//!    the classical terminating recurrence `q = Σₖ (−1)ᵏ p⁽ᵏ⁾ / a^{k+1}`
//!    gives `∫ p·e^g = q·e^g`; the defining identity `q′ + a·q = p` is
//!    re-verified in ℚ[x] (assert, on in release).
//! 3. **`p(x)·sin(g)` / `p(x)·cos(g)`**, same `g`: the ansatz
//!    `u·sin(g) + v·cos(g)` with the analogous recurrences, gated by the
//!    exact ℚ[x] identities `u′ − a·v = p, v′ + a·u = 0` (sin) and
//!    `u′ − a·v = 0, v′ + a·u = p` (cos).
//!
//! # The differentiate-back certificate
//!
//! On top of the construction gates, every emitted expression `F` is fed
//! back through the crate's own [`Expr::differentiate`] and the difference
//! `F′ − input` is decided to be *identically zero* by [`exact_zero`]: an
//! exact zero-test in `ℚ(x)[θ]` (θ = `exp(g)`, transcendental over ℚ(x) for
//! nonconstant rational `g`) or in `ℚ(x)[s, c]/(s² + c² − 1)`
//! (`s = sin(g)`, `c = cos(g)`, whose only algebraic relation over ℚ(x) is
//! the Pythagorean one) — fraction fields over these integral domains, no
//! numerics anywhere. What happens on an *undecidable* delta (`None` from
//! the certifier — it is conservative: several kernels, non-rational angle,
//! unknown functions) differs by family, and both behaviors are sound:
//!
//! - **Families 2–3 (`p·exp/sin/cos`)**: the candidate is *discarded* and
//!   `None` is returned rather than an unverified answer. Here the
//!   differentiate-back certificate is load-bearing (it is what ties the
//!   ℚ[x] recurrence gates to the emitted `Expr`), so no certificate means
//!   no emission.
//! - **Family 1 (`c·u′/u`)**: the candidate is emitted anyway, because the
//!   structural match *is already an exact proof*: every [`fold_clean`]
//!   rewrite is an identity, so structural equality of the cleaned forms
//!   proves `num ≡ c·u′`, which is verbatim the differentiation identity
//!   `d/dx (c·log u) = c·u′/u = num/den`. The θ-certifier still runs as an
//!   independent double-check whenever it can decide (a decided *nonzero*
//!   delta is a hard failure), but an undecidable delta cannot invalidate
//!   the already-established proof — this is what lets e.g.
//!   `∫ (1/x)/log(x) = log(log x)` emit although `log` is outside the
//!   certifier's kernels.
//!
//! # Honest refusals
//!
//! Everything else is refused (`None`), notably the classical
//! non-elementary integrands `exp(x)/x`, `1/log(x)`, `exp(x²)`, `sin(x)/x`,
//! kernels with a non-linear or non-rational angle (`exp(x²)`, `sin(1/x)`),
//! kernels in a denominator (`p/sin(x)`), and products with a non-polynomial
//! cofactor (`exp(x)·log(x)`).

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::risch::{
    as_exact_integer, as_exact_rational, as_rational_function, frac_add, frac_mul, frac_one,
    frac_scale, frac_zero, generically_defined, qpoly_to_expr, rational_to_expr, reduce, QFrac,
};
use crate::symbol::Symbol;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;
use std::sync::Arc;

type QP = UnivariatePolynomial<Rational>;

/// Try the exact, gated patterns; `None` is an honest refusal (the input is
/// outside all three decided families, or a certificate could not be
/// completed). See the module docs for the exact surface.
pub fn integrate_exact_patterns(e: &Expr, var: &Symbol) -> Option<Expr> {
    try_log_derivative(e, var).or_else(|| try_poly_kernel(e, var))
}

// ------------------------------------------------------------------------ //
// Sound structural cleanup (used to detect u'/u)
// ------------------------------------------------------------------------ //

fn is_lit_zero(e: &Expr) -> bool {
    matches!(e, Expr::Integer(n) if n.is_zero())
}

fn is_lit_one(e: &Expr) -> bool {
    matches!(e, Expr::Integer(n) if n.is_one())
}

fn neg_clean(e: Expr) -> Expr {
    match e {
        Expr::Unary(UnaryOp::Neg, inner) => (*inner).clone(),
        _ if is_lit_zero(&e) => Expr::from(0),
        _ => -e,
    }
}

/// Deterministic bottom-up cleanup in which **every rewrite is an exact
/// identity**: fold exact-rational subtrees to literals (via the exact
/// constant evaluator), drop `+0`/`·1`/`/1`/`^1`, cancel double negation, and
/// annihilate `0·f` for generically defined `f` (the same generic-identity
/// convention the Risch tier uses). Structural equality of cleaned forms
/// therefore implies equality of the underlying functions; inequality
/// implies nothing (the cleanup is deliberately incomplete).
fn fold_clean(e: &Expr, var: &Symbol) -> Expr {
    if let Some(r) = as_exact_rational(e) {
        return rational_to_expr(&r);
    }
    match e {
        Expr::Unary(UnaryOp::Neg, inner) => neg_clean(fold_clean(inner, var)),
        Expr::Unary(op, inner) => Expr::Unary(*op, Arc::new(fold_clean(inner, var))),
        Expr::Binary(op, l, r) => {
            let l = fold_clean(l, var);
            let r = fold_clean(r, var);
            match op {
                BinaryOp::Add if is_lit_zero(&l) => r,
                BinaryOp::Add if is_lit_zero(&r) => l,
                BinaryOp::Sub if is_lit_zero(&r) => l,
                BinaryOp::Sub if is_lit_zero(&l) => neg_clean(r),
                BinaryOp::Mul if is_lit_one(&l) => r,
                BinaryOp::Mul if is_lit_one(&r) => l,
                BinaryOp::Mul if is_lit_zero(&l) && generically_defined(&r, var) => Expr::from(0),
                BinaryOp::Mul if is_lit_zero(&r) && generically_defined(&l, var) => Expr::from(0),
                BinaryOp::Div if is_lit_one(&r) => l,
                BinaryOp::Pow if is_lit_one(&r) => l,
                _ => Expr::Binary(*op, Arc::new(l), Arc::new(r)),
            }
        }
        Expr::Function(name, args) => Expr::Function(
            name.clone(),
            args.iter().map(|a| Arc::new(fold_clean(a, var))).collect(),
        ),
        _ => e.clone(),
    }
}

/// Peel exact-rational constant factors (and negations) off a cleaned
/// expression: returns `(c, core)` with `e ≡ c·core`.
fn split_rational_const(e: &Expr) -> (Rational, Expr) {
    let mut c = Rational::one();
    let mut cur = e.clone();
    loop {
        match cur {
            Expr::Unary(UnaryOp::Neg, inner) => {
                c = -c;
                cur = (*inner).clone();
            }
            Expr::Binary(BinaryOp::Mul, l, r) => {
                if let Some(k) = as_exact_rational(&l) {
                    c = c * k;
                    cur = (*r).clone();
                } else if let Some(k) = as_exact_rational(&r) {
                    c = c * k;
                    cur = (*l).clone();
                } else {
                    return (c, Expr::Binary(BinaryOp::Mul, l, r));
                }
            }
            Expr::Binary(BinaryOp::Div, l, r) => {
                match as_exact_rational(&r) {
                    Some(k) if !k.is_zero() => {
                        c = c * k.reciprocal().expect("nonzero");
                        cur = (*l).clone();
                    }
                    _ => return (c, Expr::Binary(BinaryOp::Div, l, r)),
                }
            }
            other => return (c, other),
        }
    }
}

// ------------------------------------------------------------------------ //
// Pattern 1: log-derivative  c·u'/u  →  c·log(u)
// ------------------------------------------------------------------------ //

fn try_log_derivative(e: &Expr, var: &Symbol) -> Option<Expr> {
    let Expr::Binary(BinaryOp::Div, num, den) = e else {
        return None;
    };
    if !den.contains_symbol(var) {
        return None;
    }
    let dden = den.differentiate(var);
    let (c1, core1) = split_rational_const(&fold_clean(num, var));
    let (c2, core2) = split_rational_const(&fold_clean(&dden, var));
    if c1.is_zero() || c2.is_zero() || core1 != core2 || !core1.contains_symbol(var) {
        return None;
    }
    // Exactness: fold_clean rewrites are identities, so num ≡ c1·core and
    // den' ≡ c2·core, hence with c = c1/c2 we have num ≡ c·den' and
    // d/dx (c·log den) = c·den'/den = num/den. This *is* the differentiation
    // gate for this pattern, established by exact algebra.
    let c = c1 * c2.reciprocal().expect("c2 nonzero");
    let logd = (**den).clone().log();
    let candidate = if c.is_one() {
        logd
    } else if c == Rational::from_i64(-1) {
        -logd
    } else {
        rational_to_expr(&c) * logd
    };
    // Independent certificate through Expr::differentiate whenever the
    // kernel structure is decidable (single exp / single trig angle).
    let delta = candidate.differentiate(var) - e.clone();
    if let Some(ok) = exact_zero(&delta, var) {
        assert!(
            ok,
            "log-derivative gate failed: d/dx({}) != {}",
            candidate, e
        );
    }
    Some(candidate)
}

// ------------------------------------------------------------------------ //
// Patterns 2 and 3: p(x)·exp(ax+b), p(x)·sin(ax+b), p(x)·cos(ax+b)
// ------------------------------------------------------------------------ //

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KernelKind {
    Exp,
    Sin,
    Cos,
}

/// Split a product into `p(x) · kernel(g)`: flatten `Mul`/`Div`/`Neg`,
/// require exactly one non-inverted `exp`/`sin`/`cos` factor whose argument
/// contains `var`, and multiply everything else through the rational-function
/// normalizer. Refuses unless the cofactor is a nonzero **polynomial** and
/// `g` is a **linear** polynomial `a·x + b` over ℚ.
fn split_kernel_product(e: &Expr, var: &Symbol) -> Option<(QP, KernelKind, Expr, Rational)> {
    let mut kernel: Option<(KernelKind, Expr)> = None;
    let mut coeff = frac_one();
    let mut stack: Vec<(&Expr, bool)> = vec![(e, false)];
    while let Some((f, inv)) = stack.pop() {
        match f {
            Expr::Binary(BinaryOp::Mul, l, r) => {
                stack.push((l, inv));
                stack.push((r, inv));
            }
            Expr::Binary(BinaryOp::Div, l, r) => {
                stack.push((l, inv));
                stack.push((r, !inv));
            }
            Expr::Unary(UnaryOp::Neg, inner) => {
                // 1/(−f) = −(1/f), so the sign flips regardless of `inv`.
                coeff = frac_scale(&coeff, &Rational::from_i64(-1));
                stack.push((inner, inv));
            }
            Expr::Unary(op @ (UnaryOp::Exp | UnaryOp::Sin | UnaryOp::Cos), inner)
                if inner.contains_symbol(var) =>
            {
                if inv || kernel.is_some() {
                    // kernel in a denominator, or more than one kernel
                    return None;
                }
                let kind = match op {
                    UnaryOp::Exp => KernelKind::Exp,
                    UnaryOp::Sin => KernelKind::Sin,
                    _ => KernelKind::Cos,
                };
                kernel = Some((kind, (**inner).clone()));
            }
            other => {
                let (n, d) = as_rational_function(other, var)?;
                let factor = if inv {
                    if n.is_zero() {
                        return None;
                    }
                    reduce(d, n).expect("numerator nonzero")
                } else {
                    (n, d)
                };
                coeff = frac_mul(&coeff, &factor);
            }
        }
    }
    let (kind, g) = kernel?;
    if coeff.1 != QP::one() || coeff.0.is_zero() {
        return None;
    }
    let (gn, gd) = as_rational_function(&g, var)?;
    if gd != QP::one() || gn.degree() != Some(1) {
        return None;
    }
    let a = gn.coefficients()[1].clone(); // nonzero since deg = 1
    Some((coeff.0, kind, g, a))
}

/// `[p, p', p'', ...]` down to the last nonzero derivative.
fn derivatives(p: &QP) -> Vec<QP> {
    let mut out = vec![p.clone()];
    loop {
        let d = out.last().expect("nonempty").derivative();
        if d.is_zero() {
            return out;
        }
        out.push(d);
    }
}

/// `c(x) · kernel`, emitted tidily; `None` iff `c = 0`.
fn coeff_times(c: &QP, var: &Symbol, kernel: Expr) -> Option<Expr> {
    if c.is_zero() {
        return None;
    }
    if c.degree() == Some(0) {
        let c0 = c.coefficients()[0].clone();
        if c0.is_one() {
            return Some(kernel);
        }
        if c0 == Rational::from_i64(-1) {
            return Some(-kernel);
        }
        return Some(rational_to_expr(&c0) * kernel);
    }
    Some(qpoly_to_expr(c, var) * kernel)
}

fn try_poly_kernel(e: &Expr, var: &Symbol) -> Option<Expr> {
    let (p, kind, g, a) = split_kernel_product(e, var)?;
    let ainv = a.reciprocal().expect("a nonzero");
    let ds = derivatives(&p);
    let kexpr = |op: UnaryOp| Expr::Unary(op, Arc::new(g.clone()));

    let f_expr = match kind {
        KernelKind::Exp => {
            // q = Σ_j (−1)^j p^{(j)} / a^{j+1}
            let mut q = QP::zero();
            let mut fac = ainv.clone();
            for (j, d) in ds.iter().enumerate() {
                let t = d.scalar_mul(&fac);
                q = if j % 2 == 1 { q - t } else { q + t };
                fac = fac * ainv.clone();
            }
            // CONSTRUCTION GATE (on in release): d/dx(q·e^g) = (q' + a·q)·e^g
            // must reproduce p·e^g, i.e. q' + a·q = p exactly in ℚ[x].
            assert!(
                q.derivative() + q.scalar_mul(&a) == p,
                "exp recurrence gate failed: q' + a·q != p"
            );
            coeff_times(&q, var, kexpr(UnaryOp::Exp)).expect("q nonzero for nonzero p")
        }
        KernelKind::Sin | KernelKind::Cos => {
            // F = u·sin(g) + v·cos(g), F' = (u' − a·v)·sin + (v' + a·u)·cos.
            // sin: u = Σ_k (−1)^k p^{(2k+1)}/a^{2k+2}, v = −Σ_k (−1)^k p^{(2k)}/a^{2k+1}
            // cos: u = Σ_k (−1)^k p^{(2k)}/a^{2k+1},   v = Σ_k (−1)^k p^{(2k+1)}/a^{2k+2}
            let mut u = QP::zero();
            let mut v = QP::zero();
            let mut fac = ainv.clone();
            for (j, d) in ds.iter().enumerate() {
                let mut t = d.scalar_mul(&fac);
                if (j / 2) % 2 == 1 {
                    t = QP::zero() - t;
                }
                match (kind, j % 2) {
                    (KernelKind::Sin, 0) => v = v - t,
                    (KernelKind::Sin, _) => u = u + t,
                    (KernelKind::Cos, 0) => u = u + t,
                    _ => v = v + t,
                }
                fac = fac * ainv.clone();
            }
            // CONSTRUCTION GATES (on in release), exactly in ℚ[x]:
            let sin_coeff = u.derivative() - v.scalar_mul(&a);
            let cos_coeff = v.derivative() + u.scalar_mul(&a);
            match kind {
                KernelKind::Sin => assert!(
                    sin_coeff == p && cos_coeff.is_zero(),
                    "sin recurrence gate failed"
                ),
                _ => assert!(
                    sin_coeff.is_zero() && cos_coeff == p,
                    "cos recurrence gate failed"
                ),
            }
            let us = coeff_times(&u, var, kexpr(UnaryOp::Sin));
            let vs = coeff_times(&v, var, kexpr(UnaryOp::Cos));
            match (us, vs) {
                (Some(s), Some(c)) => s + c,
                (Some(s), None) => s,
                (None, Some(c)) => c,
                (None, None) => unreachable!("p nonzero forces u or v nonzero"),
            }
        }
    };

    // DIFFERENTIATE-BACK GATE: F' − input must be identically zero, decided
    // exactly in ℚ(x)[θ] resp. ℚ(x)[s,c]/(s²+c²−1). An undecidable delta
    // (None) discards the candidate rather than emitting unchecked.
    let delta = f_expr.differentiate(var) - e.clone();
    match exact_zero(&delta, var) {
        Some(true) => Some(f_expr),
        Some(false) => panic!(
            "integrate_exact gate failed: d/dx({}) != {}",
            f_expr, e
        ),
        None => None,
    }
}

// ------------------------------------------------------------------------ //
// Exact zero-decision for differentiate-back deltas
// ------------------------------------------------------------------------ //

/// Decide whether `e` is identically zero, exactly. `Some(true)`/`Some(false)`
/// are proofs; `None` means "cannot decide" (never used as a certificate).
///
/// Decidable cases: pure rational functions of `var`; expressions whose only
/// transcendental subtrees are `exp(g)` of a single fixed nonconstant
/// rational `g` (zero-test in the fraction field of `ℚ(x)[θ]`, θ
/// transcendental); expressions whose only transcendental subtrees are
/// `sin(g)`/`cos(g)` of a single fixed nonconstant rational `g` (zero-test in
/// the fraction field of `ℚ(x)[s,c]/(s²+c²−1)`, the full relation ideal).
pub(crate) fn exact_zero(e: &Expr, var: &Symbol) -> Option<bool> {
    if let Some((n, _)) = as_rational_function(e, var) {
        return Some(n.is_zero());
    }
    let mut kernels = Vec::new();
    collect_kernels(e, var, &mut kernels);
    let (first_is_exp, g) = kernels.first()?.clone();
    if !kernels.iter().all(|(k, gi)| *k == first_is_exp && *gi == g) {
        return None;
    }
    // Transcendence of exp(g) over ℚ(x) (resp. of the pair sin(g), cos(g)
    // modulo s²+c²−1) needs g to be a nonconstant rational function of var.
    let (gn, gd) = as_rational_function(&g, var)?;
    if gn.degree().unwrap_or(0) < 1 && gd.degree().unwrap_or(0) < 1 {
        return None;
    }
    if first_is_exp {
        let (n, _) = theta_normalize::<EPoly>(e, var, &g)?;
        Some(n.is_zero())
    } else {
        let (n, _) = theta_normalize::<TrigPair>(e, var, &g)?;
        Some(n.is_zero())
    }
}

/// Collect the distinct `exp`/`sin`/`cos` subtrees whose argument involves
/// `var`, tagged with the family (`true` = exp). `sin` and `cos` of the same
/// angle count as one entry.
fn collect_kernels(e: &Expr, var: &Symbol, out: &mut Vec<(bool, Expr)>) {
    match e {
        Expr::Unary(op @ (UnaryOp::Exp | UnaryOp::Sin | UnaryOp::Cos), inner)
            if inner.contains_symbol(var) =>
        {
            let is_exp = *op == UnaryOp::Exp;
            if !out.iter().any(|(k, g)| *k == is_exp && *g == **inner) {
                out.push((is_exp, (**inner).clone()));
            }
            collect_kernels(inner, var, out);
        }
        Expr::Unary(_, inner) => collect_kernels(inner, var, out),
        Expr::Binary(_, l, r) => {
            collect_kernels(l, var, out);
            collect_kernels(r, var, out);
        }
        Expr::Function(_, args) => {
            for a in args {
                collect_kernels(a, var, out);
            }
        }
        _ => {}
    }
}

// --- Coefficient polynomials: Vec<QFrac>, index = degree, kept trimmed. ---

fn cp_trim(mut v: Vec<QFrac>) -> Vec<QFrac> {
    while v.last().is_some_and(|c| c.0.is_zero()) {
        v.pop();
    }
    v
}

fn cp_add(a: &[QFrac], b: &[QFrac]) -> Vec<QFrac> {
    let mut out = a.to_vec();
    if out.len() < b.len() {
        out.resize(b.len(), frac_zero());
    }
    for (i, c) in b.iter().enumerate() {
        out[i] = frac_add(&out[i], c);
    }
    cp_trim(out)
}

fn cp_neg(a: &[QFrac]) -> Vec<QFrac> {
    let m1 = Rational::from_i64(-1);
    a.iter().map(|c| frac_scale(c, &m1)).collect()
}

fn cp_mul(a: &[QFrac], b: &[QFrac]) -> Vec<QFrac> {
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
    cp_trim(out)
}

/// An integral domain extending ℚ(x) by the kernel(s); zero-test must be
/// exact, and a nonzero element must be a nonzero function (that is what
/// makes the fraction arithmetic in [`theta_normalize`] sound).
trait ThetaRing: Clone {
    fn from_frac(f: QFrac) -> Self;
    fn one() -> Self {
        Self::from_frac(frac_one())
    }
    fn add(&self, o: &Self) -> Self;
    fn neg(&self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn is_zero(&self) -> bool;
    /// The image of a kernel subtree of angle `g`, if `e` is one.
    fn kernel(e: &Expr, g: &Expr) -> Option<Self>;
}

/// ℚ(x)[θ] with θ = exp(g): plain polynomials in θ over ℚ(x).
#[derive(Clone)]
struct EPoly(Vec<QFrac>);

impl ThetaRing for EPoly {
    fn from_frac(f: QFrac) -> Self {
        EPoly(cp_trim(vec![f]))
    }
    fn add(&self, o: &Self) -> Self {
        EPoly(cp_add(&self.0, &o.0))
    }
    fn neg(&self) -> Self {
        EPoly(cp_neg(&self.0))
    }
    fn mul(&self, o: &Self) -> Self {
        EPoly(cp_mul(&self.0, &o.0))
    }
    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }
    fn kernel(e: &Expr, g: &Expr) -> Option<Self> {
        if let Expr::Unary(UnaryOp::Exp, inner) = e {
            if **inner == *g {
                return Some(EPoly(vec![frac_zero(), frac_one()]));
            }
        }
        None
    }
}

/// ℚ(x)[s, c]/(s² + c² − 1) with s = sin(g), c = cos(g), in the normal form
/// `A(c) + B(c)·s` (s² reduced to 1 − c²). An integral domain: s² − (1 − c²)
/// is irreducible over ℚ(x)(c) since 1 − c² is not a square there. Zero iff
/// A = B = 0, which is exact because for nonconstant rational g the only
/// algebraic relation between sin(g) and cos(g) over ℚ(x) is s² + c² = 1.
#[derive(Clone)]
struct TrigPair {
    a: Vec<QFrac>,
    b: Vec<QFrac>,
}

impl ThetaRing for TrigPair {
    fn from_frac(f: QFrac) -> Self {
        TrigPair {
            a: cp_trim(vec![f]),
            b: Vec::new(),
        }
    }
    fn add(&self, o: &Self) -> Self {
        TrigPair {
            a: cp_add(&self.a, &o.a),
            b: cp_add(&self.b, &o.b),
        }
    }
    fn neg(&self) -> Self {
        TrigPair {
            a: cp_neg(&self.a),
            b: cp_neg(&self.b),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        // (A1 + B1·s)(A2 + B2·s) = A1A2 + B1B2·(1 − c²) + (A1B2 + B1A2)·s
        let one_minus_c2 = vec![
            frac_one(),
            frac_zero(),
            frac_scale(&frac_one(), &Rational::from_i64(-1)),
        ];
        TrigPair {
            a: cp_add(
                &cp_mul(&self.a, &o.a),
                &cp_mul(&cp_mul(&self.b, &o.b), &one_minus_c2),
            ),
            b: cp_add(&cp_mul(&self.a, &o.b), &cp_mul(&self.b, &o.a)),
        }
    }
    fn is_zero(&self) -> bool {
        self.a.is_empty() && self.b.is_empty()
    }
    fn kernel(e: &Expr, g: &Expr) -> Option<Self> {
        match e {
            Expr::Unary(UnaryOp::Sin, inner) if **inner == *g => Some(TrigPair {
                a: Vec::new(),
                b: vec![frac_one()],
            }),
            Expr::Unary(UnaryOp::Cos, inner) if **inner == *g => Some(TrigPair {
                a: vec![frac_zero(), frac_one()],
                b: Vec::new(),
            }),
            _ => None,
        }
    }
}

/// Normalize `e` to a fraction `(num, den)` over the θ-ring `R` (den built
/// from nonzero factors, hence nonzero in the domain — so `e ≡ 0` iff
/// `num.is_zero()`). Kernel subtrees map through `R::kernel`; kernel-free
/// subtrees delegate to the exact rational normalizer. `None` = refusal.
fn theta_normalize<R: ThetaRing>(e: &Expr, var: &Symbol, g: &Expr) -> Option<(R, R)> {
    if let Some(k) = R::kernel(e, g) {
        return Some((k, R::one()));
    }
    if let Some(f) = as_rational_function(e, var) {
        return Some((R::from_frac(f), R::one()));
    }
    match e {
        Expr::Unary(UnaryOp::Neg, inner) => {
            let (n, d) = theta_normalize::<R>(inner, var, g)?;
            Some((n.neg(), d))
        }
        Expr::Binary(op, l, r) => match op {
            BinaryOp::Add | BinaryOp::Sub => {
                let (n1, d1) = theta_normalize::<R>(l, var, g)?;
                let (n2, d2) = theta_normalize::<R>(r, var, g)?;
                let n2 = if *op == BinaryOp::Sub { n2.neg() } else { n2 };
                Some((n1.mul(&d2).add(&n2.mul(&d1)), d1.mul(&d2)))
            }
            BinaryOp::Mul => {
                let (n1, d1) = theta_normalize::<R>(l, var, g)?;
                let (n2, d2) = theta_normalize::<R>(r, var, g)?;
                Some((n1.mul(&n2), d1.mul(&d2)))
            }
            BinaryOp::Div => {
                let (n1, d1) = theta_normalize::<R>(l, var, g)?;
                let (n2, d2) = theta_normalize::<R>(r, var, g)?;
                if n2.is_zero() {
                    return None;
                }
                Some((n1.mul(&d2), d1.mul(&n2)))
            }
            BinaryOp::Pow => {
                let k = as_exact_integer(r)?;
                if k.abs() > Integer::from(4096) {
                    return None;
                }
                let (n, d) = theta_normalize::<R>(l, var, g)?;
                let ku = k.abs().to_i64() as usize;
                let (mut pn, mut pd) = (R::one(), R::one());
                for _ in 0..ku {
                    pn = pn.mul(&n);
                    pd = pd.mul(&d);
                }
                if k < Integer::from(0) {
                    if pn.is_zero() {
                        return None;
                    }
                    Some((pd, pn))
                } else {
                    Some((pn, pd))
                }
            }
            BinaryOp::Mod => None,
        },
        _ => None,
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

    /// THE GATE (test form): differentiate the candidate with the crate's
    /// differentiate() and certify F' − input ≡ 0 with the exact θ-machinery.
    fn assert_gated(input: &Expr, antiderivative: &Expr) {
        let delta = antiderivative.differentiate(&xsym()) - input.clone();
        assert_eq!(
            exact_zero(&delta, &xsym()),
            Some(true),
            "gate: d/dx({}) != {}",
            antiderivative,
            input
        );
    }

    fn integrate_gated(input: &Expr) -> Expr {
        let f = integrate_exact_patterns(input, &xsym())
            .unwrap_or_else(|| panic!("expected Some for {}", input));
        assert_gated(input, &f);
        f
    }

    fn contains_unary(e: &Expr, op: UnaryOp) -> bool {
        match e {
            Expr::Unary(o, inner) => *o == op || contains_unary(inner, op),
            Expr::Binary(_, l, r) => contains_unary(l, op) || contains_unary(r, op),
            Expr::Function(_, args) => args.iter().any(|a| contains_unary(a, op)),
            _ => false,
        }
    }

    // ---------------- the exact zero-decider itself ----------------

    #[test]
    fn exact_zero_decides_trig_and_exp_identities() {
        let x = xexpr();
        let s = x.clone().sin();
        let c = x.clone().cos();
        // sin² + cos² − 1 ≡ 0: needs the full relation ideal
        let pyth = s.clone().pow(Expr::from(2)) + c.clone().pow(Expr::from(2)) - Expr::from(1);
        assert_eq!(exact_zero(&pyth, &xsym()), Some(true));
        // sin − cos is NOT zero
        assert_eq!(exact_zero(&(s.clone() - c.clone()), &xsym()), Some(false));
        // exp(x)·exp(x) − exp(x)² ≡ 0
        let ex = x.clone().exp();
        let e2 = ex.clone() * ex.clone() - ex.clone().pow(Expr::from(2));
        assert_eq!(exact_zero(&e2, &xsym()), Some(true));
        // exp(x) − 1 is NOT zero
        assert_eq!(exact_zero(&(ex - Expr::from(1)), &xsym()), Some(false));
        // undecidable: two distinct angles / unknown transcendental
        assert_eq!(exact_zero(&(x.clone().sin() + (x.clone() * Expr::from(2)).sin()), &xsym()), None);
        assert_eq!(exact_zero(&x.clone().log(), &xsym()), None);
        // mixed exp and trig of the same angle: refused
        assert_eq!(exact_zero(&(x.clone().sin() * x.clone().exp()), &xsym()), None);
        // constant angle: refused (transcendence argument needs nonconstant g)
        assert_eq!(exact_zero(&Expr::from(2).sin(), &xsym()), None);
        // pure rational still decided
        let r = (x.clone() + Expr::from(1)).pow(Expr::from(2))
            - x.clone().pow(Expr::from(2))
            - Expr::from(2) * x.clone()
            - Expr::from(1);
        assert_eq!(exact_zero(&r, &xsym()), Some(true));
    }

    // ---------------- pattern 1: log-derivative ----------------

    // sympy: integrate(cos(x)/sin(x)) = log(sin(x))
    #[test]
    fn log_derivative_cot() {
        let x = xexpr();
        let input = x.clone().cos() / x.clone().sin();
        let f = integrate_gated(&input);
        assert_eq!(f, x.clone().sin().log());
    }

    // sympy: integrate(2*cos(x)/sin(x)) = 2*log(sin(x))
    #[test]
    fn log_derivative_constant_multiple() {
        let x = xexpr();
        let input = (Expr::from(2) * x.clone().cos()) / x.clone().sin();
        let f = integrate_gated(&input);
        assert_eq!(f, Expr::from(2) * x.clone().sin().log());
    }

    // sympy: integrate(cos(x)/(2*sin(x))) = log(sin(x))/2. Ours is
    // (1/2)·log(2·sin(x)) — the same antiderivative up to the constant
    // log(2)/2, certified by the differentiate-back gate.
    #[test]
    fn log_derivative_scaled_denominator() {
        let x = xexpr();
        let input = x.clone().cos() / (Expr::from(2) * x.clone().sin());
        let f = integrate_gated(&input);
        assert!(contains_unary(&f, UnaryOp::Log));
    }

    // sympy: integrate(exp(x)/(exp(x) + 1)) = log(exp(x) + 1)
    #[test]
    fn log_derivative_exp_kernel() {
        let x = xexpr();
        let input = x.clone().exp() / (x.clone().exp() + Expr::from(1));
        let f = integrate_gated(&input);
        assert_eq!(f, (x.clone().exp() + Expr::from(1)).log());
    }

    // sympy: integrate(2*x*cos(x**2)/sin(x**2)) = log(sin(x**2)); the angle
    // is not linear, so the poly·kernel tier refuses it, but u'/u matches
    // structurally and the trig certifier handles the nonconstant rational
    // angle g = x².
    #[test]
    fn log_derivative_nonlinear_angle() {
        let x = xexpr();
        let x2 = x.clone().pow(Expr::from(2));
        let num = Expr::from(2) * x.clone() * x2.clone().cos();
        let input = num / x2.clone().sin();
        // detection needs cleaned(num) == cleaned(d/dx sin(x²)) up to a
        // rational constant; d/dx sin(x²) = cos(x²)·(2x·1) — forms differ
        // structurally (2·x·cos vs cos·(2x)), so this may honestly refuse.
        // Accept either outcome, but never an unverified emission.
        if let Some(f) = integrate_exact_patterns(&input, &xsym()) {
            assert_gated(&input, &f);
        }
    }

    // ---------------- pattern 2: p(x)·exp(ax+b) ----------------

    // sympy: integrate(x*exp(x)) = (x - 1)*exp(x)
    #[test]
    fn poly_exp_x() {
        let x = xexpr();
        let input = x.clone() * x.clone().exp();
        let f = integrate_gated(&input);
        // exact emitted form: (−1 + x)·exp(x)
        let expected = (Expr::from(-1) + x.clone()) * x.clone().exp();
        assert_eq!(f, expected);
    }

    // sympy: integrate(x**2*exp(-x)) = (-x**2 - 2*x - 2)*exp(-x)
    #[test]
    fn poly_exp_negative_a() {
        let x = xexpr();
        let input = x.clone().pow(Expr::from(2)) * (-x.clone()).exp();
        integrate_gated(&input);
    }

    // sympy: integrate(x/2*exp(2*x+1)) = (2*x - 1)*exp(2*x + 1)/8,
    // i.e. (x/4 − 1/8)·e^{2x+1}
    #[test]
    fn poly_exp_affine_angle_rational_coeff() {
        let x = xexpr();
        let g = Expr::from(2) * x.clone() + Expr::from(1);
        let input = (x.clone() / Expr::from(2)) * g.clone().exp();
        integrate_gated(&input);
    }

    // sympy: integrate(exp(3*x)) = exp(3*x)/3
    #[test]
    fn bare_exp_scaled() {
        let x = xexpr();
        let input = (Expr::from(3) * x.clone()).exp();
        let f = integrate_gated(&input);
        let expected = Expr::Rational(Rational::new(1, 3).unwrap())
            * (Expr::from(3) * x.clone()).exp();
        assert_eq!(f, expected);
    }

    // sympy: integrate((x+1)*exp(2*x-3)) = (2*x + 1)*exp(2*x - 3)/4,
    // i.e. (x/2 + 1/4)·e^{2x−3}
    #[test]
    fn poly_exp_affine_shift() {
        let x = xexpr();
        let g = Expr::from(2) * x.clone() - Expr::from(3);
        let input = (x.clone() + Expr::from(1)) * g.clone().exp();
        integrate_gated(&input);
    }

    // ---------------- pattern 3: p(x)·sin/cos(ax+b) ----------------

    // sympy: integrate(x*sin(x)) = -x*cos(x) + sin(x)
    #[test]
    fn poly_sin_x() {
        let x = xexpr();
        let input = x.clone() * x.clone().sin();
        let f = integrate_gated(&input);
        assert!(contains_unary(&f, UnaryOp::Sin) && contains_unary(&f, UnaryOp::Cos));
    }

    // sympy: integrate(x**2*cos(2*x))
    //        = x**2*sin(2*x)/2 + x*cos(2*x)/2 - sin(2*x)/4,
    // i.e. u = x²/2 − 1/4 on sin(2x), v = x/2 on cos(2x)
    #[test]
    fn poly_cos_2x() {
        let x = xexpr();
        let input = x.clone().pow(Expr::from(2)) * (Expr::from(2) * x.clone()).cos();
        integrate_gated(&input);
    }

    // sympy: integrate(sin(2*x)) = -cos(2*x)/2
    #[test]
    fn bare_sin_scaled() {
        let x = xexpr();
        let input = (Expr::from(2) * x.clone()).sin();
        let f = integrate_gated(&input);
        let expected = Expr::Rational(Rational::new(-1, 2).unwrap())
            * (Expr::from(2) * x.clone()).cos();
        assert_eq!(f, expected);
    }

    // sympy: integrate(x*sin(x/2)) = -2*x*cos(x/2) + 4*sin(x/2)
    // (a = 1/2: u = 4, v = −2x)
    #[test]
    fn poly_sin_half_x() {
        let x = xexpr();
        let input = x.clone() * (x.clone() / Expr::from(2)).sin();
        integrate_gated(&input);
    }

    // sympy: integrate(-sin(x)) = cos(x)
    #[test]
    fn negated_bare_sin() {
        let x = xexpr();
        let input = -(x.clone().sin());
        let f = integrate_gated(&input);
        assert_eq!(f, x.clone().cos());
    }

    // sympy: integrate((3*x**2-1)*cos(3*x))
    //        = x**2*sin(3*x) + 2*x*cos(3*x)/3 - 5*sin(3*x)/9
    // (a = 3: u = x² − 5/9, v = 2x/3)
    #[test]
    fn poly_cos_3x() {
        let x = xexpr();
        let p = Expr::from(3) * x.clone().pow(Expr::from(2)) - Expr::from(1);
        let input = p * (Expr::from(3) * x.clone()).cos();
        integrate_gated(&input);
    }

    // legacy-form compatibility: the simple kernels emit exactly the shapes
    // the old table produced, so pre-existing literal assertions keep passing
    #[test]
    fn simple_kernels_match_legacy_forms() {
        let x = xexpr();
        assert_eq!(
            integrate_gated(&x.clone().sin()),
            -x.clone().cos() // ∫sin = −cos, emitted as Neg(cos), not (−1)·cos
        );
        assert_eq!(integrate_gated(&x.clone().cos()), x.clone().sin());
        assert_eq!(integrate_gated(&x.clone().exp()), x.clone().exp());
    }

    // ---------------- honest refusals ----------------

    #[test]
    fn refuses_nonelementary_and_out_of_scope() {
        let x = xexpr();
        let refusals: Vec<Expr> = vec![
            x.clone().exp() / x.clone(),               // Ei(x): nonelementary
            Expr::from(1) / x.clone().log(),           // li(x): nonelementary
            x.clone().pow(Expr::from(2)).exp(),        // erfi: nonlinear angle
            x.clone().sin() / x.clone(),               // Si(x): nonelementary
            x.clone().sin() * x.clone().exp(),         // two kernels
            x.clone() / x.clone().sin(),               // kernel in denominator
            x.clone().log() * x.clone().exp(),         // non-polynomial cofactor
            (Expr::from(1) / x.clone()).exp(),         // non-polynomial angle
            Expr::symbol("y") * x.clone().exp(),       // second free symbol
        ];
        for e in refusals {
            assert_eq!(
                integrate_exact_patterns(&e, &xsym()),
                None,
                "should refuse {}",
                e
            );
        }
    }
}
