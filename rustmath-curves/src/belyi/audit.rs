//! The independent **monodromy audit** — the decisive, non-provisional gate.
//!
//! Once the `[2,12,5]` cover is solved (its 25-coefficient tuple
//! `a₀..a₇, b₀..b₇, r₀..r₂, s₀..s₃, λ, c` exactified), this module answers the
//! only question that makes a verdict trustworthy rather than *provisional*:
//! **is this really the M24 cover, and — over `Q` — does it realize M23?**
//!
//! The Belyi map at a solution is
//!
//! ```text
//!     φ(x) = A(x)²·B(x) / (λ·R(x)⁵·S(x))
//! ```
//!
//! with the exact identity `A²B − λR⁵S = c·x¹²` (so `φ = 0` at type `2⁸1⁸`,
//! `φ = ∞` at `5⁴1⁴`, `φ = 1` at `x¹²`). `A,B` are monic degree 8;
//! `R = (x−1)·(monic cubic)` degree 4; `S` monic degree 4.
//!
//! The audit is *Frobenius-statistical* and honest: it classifies the Galois
//! group of the specialized fibre polynomials over `Q` by their cycle types at
//! several primes, matched against the Wave-1 transitive-group classifiers
//! ([`rustmath_groups::transitive24`] / [`rustmath_groups::transitive23`]). It
//! **only** asserts a group when the evidence *forces* it; otherwise it is
//! [`GroupVerdict::Unresolved`].
//!
//! * [`audit_m24`] — for a few generic rational `t₀`, form the degree-24
//!   numerator `f_{t₀}(x) = A²B − t₀·λ·R⁵S` (the numerator of `φ − t₀`) over `Q`,
//!   collect Frobenius cycle types, and require them to force the monodromy group
//!   to be **M24** (`24T24680`). This is the decisive gate: a parasitic/degenerate
//!   solution (whose group is a proper subgroup, or the full `S₂₄`) fails it.
//! * [`audit_m23_residual`] — for a rational source point `x₀`, set `t₀ = φ(x₀)`,
//!   split off the linear factor `(x − x₀)` from `f_{t₀}` (the Belyi `1+23`
//!   split), require the degree-23 residual `g(x)` irreducible, and classify it as
//!   **M23**. When it succeeds, `g` **is** the witnessing M23/`Q` polynomial.
//!
//! Cycle types alone cannot separate `M24 ⊂ A24 ⊂ S24` (all share the same
//! *supersets*), so M24 is forced via the **blind class** — the set of degree-24
//! transitive groups whose cycle-type support is *exactly* the observed set. Once
//! the prime sample is Chebotarev-saturated on the M24 fibre, that blind class is
//! the single group `24T24680`.
//!
//! NOTE (before publication): an exact OSCAR `galois_group` cross-check of both
//! the degree-24 fibre polynomial and the degree-23 residual should follow this
//! native statistical audit — this module is the fast, dependency-free gate, not
//! a substitute for the computer-algebra certificate.

use rustmath_groups::transitive23::{self, Group23};
use rustmath_groups::transitive24::CycleTypeSupport;
use rustmath_integers::Integer;
use rustmath_polynomials::disc::discriminant;
use rustmath_polynomials::padic_factor::cycle_type;
use rustmath_polynomials::zassenhaus;
use rustmath_polynomials::zx;
use rustmath_rationals::Rational;
use std::collections::BTreeSet;

/// The LMFDB `t`-index of the Mathieu group `M₂₄` among the degree-24 transitive
/// groups (`24T24680`).
pub const M24_T: usize = 24680;

/// Which group an audit confirmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKind {
    /// The Mathieu group `M₂₄` (`24T24680`) — the geometric monodromy group.
    M24,
    /// The Mathieu group `M₂₃` — the point stabilizer, realized over `Q`.
    M23,
}

/// The verdict of a monodromy audit — honest about the strength of the evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupVerdict {
    /// The observed evidence **forces** the named group (no smaller/other candidate).
    Confirmed(GroupKind),
    /// The evidence is *consistent* with the target group but does not force it
    /// (more primes needed, or the sample is not yet Chebotarev-saturated).
    SubgroupOnly,
    /// The evidence does not resolve the group, or excludes the target
    /// (parasitic/degenerate solution, contradictory parity, etc.).
    Unresolved,
}

// ---------------------------------------------------------------------------
// Coefficient access — the exactified 25-vector.
// ---------------------------------------------------------------------------

/// The four Belyi factors as dense **ascending** rational coefficient vectors,
/// plus `λ`. Layout of `coeffs` (length ≥ 24): `a₀..a₇` (0..8), `b₀..b₇` (8..16),
/// `r₀..r₂` (16..19), `s₀..s₃` (19..23), `λ` (23); `c` (24) is not needed here.
struct Factors {
    a: Vec<Rational>, // A = x⁸ + a₇x⁷ + … + a₀   (length 9)
    b: Vec<Rational>, // B = x⁸ + …                (length 9)
    r: Vec<Rational>, // R = (x−1)(x³+r₂x²+r₁x+r₀) (length 5)
    s: Vec<Rational>, // S = x⁴ + …                (length 5)
    lambda: Rational,
}

fn extract(coeffs: &[Rational]) -> Factors {
    assert!(
        coeffs.len() >= 24,
        "expected the 25-vector a0..a7,b0..b7,r0..r2,s0..s3,lambda,c (got {})",
        coeffs.len()
    );
    let one = Rational::from_i64(1);

    let mut a = coeffs[0..8].to_vec();
    a.push(one.clone()); // + x⁸
    let mut b = coeffs[8..16].to_vec();
    b.push(one.clone());
    let mut s = coeffs[19..23].to_vec();
    s.push(one.clone());

    // cubic = x³ + r₂x² + r₁x + r₀ ; R = (x−1)·cubic
    let cubic = vec![
        coeffs[16].clone(),
        coeffs[17].clone(),
        coeffs[18].clone(),
        one.clone(),
    ];
    let x_minus_1 = vec![Rational::from_i64(-1), one];
    let r = rat_mul(&x_minus_1, &cubic);

    Factors {
        a,
        b,
        r,
        s,
        lambda: coeffs[23].clone(),
    }
}

// ---------------------------------------------------------------------------
// Dense rational polynomial helpers (ascending coefficients).
// ---------------------------------------------------------------------------

fn rat_mul(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut c = vec![Rational::from_i64(0); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            c[i + j] = c[i + j].clone() + ai.clone() * bj.clone();
        }
    }
    c
}

fn rat_pow(a: &[Rational], e: u32) -> Vec<Rational> {
    let mut acc = vec![Rational::from_i64(1)];
    for _ in 0..e {
        acc = rat_mul(&acc, a);
    }
    acc
}

fn rat_scale(a: &[Rational], k: &Rational) -> Vec<Rational> {
    a.iter().map(|c| c.clone() * k.clone()).collect()
}

fn rat_sub(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let n = a.len().max(b.len());
    let z = Rational::from_i64(0);
    (0..n)
        .map(|i| a.get(i).unwrap_or(&z).clone() - b.get(i).unwrap_or(&z).clone())
        .collect()
}

/// Horner evaluation of an ascending rational polynomial at a rational point.
fn eval_rat(poly: &[Rational], x0: &Rational) -> Rational {
    let mut acc = Rational::from_i64(0);
    for c in poly.iter().rev() {
        acc = acc * x0.clone() + c.clone();
    }
    acc
}

/// Synthetic division of `poly` (ascending) by `(x − x0)`; returns
/// `(quotient, remainder)`.
fn div_linear(poly: &[Rational], x0: &Rational) -> (Vec<Rational>, Rational) {
    let n = poly.len();
    if n == 0 {
        return (Vec::new(), Rational::from_i64(0));
    }
    let mut q = vec![Rational::from_i64(0); n - 1];
    let mut carry = poly[n - 1].clone();
    for k in (0..n - 1).rev() {
        q[k] = carry.clone();
        carry = poly[k].clone() + x0.clone() * carry;
    }
    (q, carry)
}

/// Clear an ascending rational polynomial to its **primitive integer** form
/// (positive leading coefficient, content 1, trailing zeros trimmed).
fn clear_primitive(f: &[Rational]) -> Vec<Integer> {
    // Least common multiple of the denominators.
    let mut den = Integer::one();
    for c in f {
        den = den.lcm(c.denominator());
    }
    let ints: Vec<Integer> = f
        .iter()
        .map(|c| {
            let scale = den.clone() / c.denominator().clone(); // exact: den is a multiple
            c.numerator().clone() * scale
        })
        .collect();
    zx::normalize(&ints)
}

// ---------------------------------------------------------------------------
// The specialized fibre polynomial and φ evaluation.
// ---------------------------------------------------------------------------

/// The degree-24 numerator of `φ − t₀`:
/// `f_{t₀}(x) = A(x)²B(x) − t₀·λ·R(x)⁵·S(x)`, cleared to primitive integer form.
///
/// For a generic rational `t₀` the leading coefficient is `1 − t₀·λ ≠ 0`, so
/// `f_{t₀}` has degree 24. (If `t₀ = 1/λ` the degree drops; callers screen this
/// by requiring the cycle type to sum to 24.)
pub fn specialize_numerator(coeffs: &[Rational], t0: &Rational) -> Vec<Integer> {
    let f = extract(coeffs);
    let a2b = rat_mul(&rat_pow(&f.a, 2), &f.b); // degree 24
    let r5s = rat_mul(&rat_pow(&f.r, 5), &f.s); // degree 24
    let scalar = t0.clone() * f.lambda.clone(); // t₀·λ
    let num = rat_sub(&a2b, &rat_scale(&r5s, &scalar));
    clear_primitive(&num)
}

/// Evaluate the Belyi map `φ = A²B / (λR⁵S)` at a rational source point `x₀`.
/// `None` when the denominator vanishes (a pole).
pub fn phi_at(coeffs: &[Rational], x0: &Rational) -> Option<Rational> {
    let f = extract(coeffs);
    let a = eval_rat(&f.a, x0);
    let b = eval_rat(&f.b, x0);
    let r = eval_rat(&f.r, x0);
    let s = eval_rat(&f.s, x0);
    let num = a.clone() * a * b;
    let den = f.lambda.clone() * rat_pow_scalar(&r, 5) * s;
    if den == Rational::from_i64(0) {
        return None;
    }
    Some(num * rat_inv(&den)?)
}

fn rat_pow_scalar(x: &Rational, e: u32) -> Rational {
    let mut acc = Rational::from_i64(1);
    for _ in 0..e {
        acc = acc * x.clone();
    }
    acc
}

fn rat_inv(x: &Rational) -> Option<Rational> {
    if *x == Rational::from_i64(0) {
        return None;
    }
    Rational::new(x.denominator().clone(), x.numerator().clone()).ok()
}

// ---------------------------------------------------------------------------
// Frobenius cycle-type collection + classification.
// ---------------------------------------------------------------------------

/// The set of Frobenius cycle types of the integer polynomial `f` at the given
/// primes, keeping only unramified primes whose cycle type sums to `deg` (so a
/// degree drop / bad reduction is discarded). Returned sorted & de-duplicated.
pub fn frobenius_types(f: &[Integer], primes: &[i64], deg: usize) -> Vec<Vec<usize>> {
    let mut set: BTreeSet<Vec<usize>> = BTreeSet::new();
    for &p in primes {
        if let Some(ct) = cycle_type(f, p) {
            if ct.iter().sum::<usize>() == deg {
                set.insert(ct);
            }
        }
    }
    set.into_iter().collect()
}

/// Classify a degree-24 observed cycle-type set against the degree-24 transitive
/// groups, forcing M24 via the **blind class** (exact-support match).
///
/// * `Confirmed(M24)` iff the blind class is exactly `{24T24680}` — no other
///   degree-24 transitive group has this precise cycle-type support, so once the
///   sample is Chebotarev-saturated on the M24 fibre the group is forced.
/// * `SubgroupOnly` iff M24 is merely *consistent* (in the sound candidate set)
///   but not forced — more primes are needed.
/// * `Unresolved` iff M24 is excluded (parasitic/degenerate) or the support is
///   unavailable.
pub fn classify_m24(observed: &[Vec<usize>]) -> GroupVerdict {
    if observed.is_empty() {
        return GroupVerdict::Unresolved;
    }
    let support = match CycleTypeSupport::load_default() {
        Ok(s) => s,
        Err(_) => return GroupVerdict::Unresolved,
    };
    let blind = support.blind_class(observed); // exact-support match
    if blind == [M24_T] {
        return GroupVerdict::Confirmed(GroupKind::M24);
    }
    let candidates = support.candidates(observed); // sound superset
    if candidates.contains(&M24_T) {
        GroupVerdict::SubgroupOnly
    } else {
        GroupVerdict::Unresolved
    }
}

/// A short, "generic" list of rational `t₀` values at which to sample the fibre.
fn generic_t0s() -> Vec<Rational> {
    let mut v: Vec<Rational> = [2, 3, 5, 7, -1, -2, -3, 11, 13]
        .iter()
        .map(|&n| Rational::from_i64(n))
        .collect();
    v.push(Rational::new(1, 2).unwrap());
    v.push(Rational::new(3, 2).unwrap());
    v.push(Rational::new(-1, 3).unwrap());
    v
}

/// The decisive M24 gate: for several generic rational `t₀`, build the degree-24
/// fibre polynomial `f_{t₀}` over `Q`, collect Frobenius cycle types over
/// `primes`, and require them to **force** the monodromy group to be `M24`.
///
/// Statistical & honest: returns [`GroupVerdict::Confirmed`] only when the
/// combined observed cycle-type set is exactly the M24 blind class; otherwise
/// [`GroupVerdict::SubgroupOnly`] (consistent, unforced) or
/// [`GroupVerdict::Unresolved`] (excluded/degenerate).
pub fn audit_m24(coeffs: &[Rational], primes: &[i64]) -> GroupVerdict {
    let mut observed: BTreeSet<Vec<usize>> = BTreeSet::new();
    for t0 in generic_t0s() {
        let f = specialize_numerator(coeffs, &t0);
        if zx::degree(&f) != 24 {
            continue; // degenerate specialization (e.g. t₀ = 1/λ)
        }
        for ct in frobenius_types(&f, primes, 24) {
            observed.insert(ct);
        }
    }
    let obs: Vec<Vec<usize>> = observed.into_iter().collect();
    classify_m24(&obs)
}

// ---------------------------------------------------------------------------
// The degree-23 residual — the M23/Q realization.
// ---------------------------------------------------------------------------

/// The witness produced by [`audit_m23_residual`].
#[derive(Debug, Clone)]
pub struct M23Witness {
    /// The audit verdict (`Confirmed(M23)` is the payoff).
    pub verdict: GroupVerdict,
    /// The classified degree-23 group (when a residual was formed & classified).
    pub group23: Option<Group23>,
    /// The witnessing degree-23 **primitive integer** polynomial (ascending),
    /// present exactly when a `1+23` split was formed and irreducible.
    pub residual: Option<Vec<Integer>>,
    /// The source point `x₀` that produced the residual.
    pub x0: Option<Rational>,
    /// `t₀ = φ(x₀)`.
    pub t0: Option<Rational>,
    /// Whether `disc(residual)` is a perfect square (the `Gal ⊆ A₂₃` bit).
    pub disc_is_square: bool,
    pub notes: Vec<String>,
}

impl M23Witness {
    fn unresolved(note: impl Into<String>) -> Self {
        M23Witness {
            verdict: GroupVerdict::Unresolved,
            group23: None,
            residual: None,
            x0: None,
            t0: None,
            disc_is_square: false,
            notes: vec![note.into()],
        }
    }
    /// The payoff: M23 confirmed over `Q` with the witnessing polynomial in hand.
    pub fn is_m23_realized(&self) -> bool {
        matches!(self.verdict, GroupVerdict::Confirmed(GroupKind::M23))
    }
}

/// Is the degree-23 integer polynomial `g` irreducible over `Q`? Sound complete
/// test via full Zassenhaus factorization (irreducible iff a single degree-23
/// factor of multiplicity one).
fn is_irreducible_deg(g: &[Integer], deg: usize) -> bool {
    match zassenhaus::factor(g) {
        Ok((_, factors)) => {
            factors.len() == 1
                && factors[0].1 == 1
                && zx::degree(&factors[0].0) == deg as i64
        }
        Err(_) => false,
    }
}

/// The `1+23` split: for each rational source point `x₀`, set `t₀ = φ(x₀)`, split
/// the linear factor `(x − x₀)` off the degree-24 numerator `f_{t₀}`, and — when
/// the degree-23 residual `g` is irreducible — classify its Galois group. Returns
/// on the **first** `x₀` that yields an M23 witness; otherwise the last
/// informative attempt (or an `Unresolved` witness if none formed).
///
/// The disc-square bit `Gal ⊆ A₂₃` is computed exactly from `disc(g)`; M23 is
/// confirmed only when the observed cycle types force it in the even chain
/// (`C23 ⊂ F23 ⊂ M23 ⊂ A23`).
pub fn audit_m23_residual(
    coeffs: &[Rational],
    x0_candidates: &[Rational],
    primes: &[i64],
) -> M23Witness {
    let mut last: M23Witness = M23Witness::unresolved("no rational source point produced a residual");

    for x0 in x0_candidates {
        let t0 = match phi_at(coeffs, x0) {
            Some(t) => t,
            None => continue, // x₀ is a pole
        };
        // f_{t0} has x₀ as a root by construction (φ(x₀) = t₀).
        let f = specialize_numerator(coeffs, &t0);
        if zx::degree(&f) != 24 {
            continue;
        }
        // Split (x − x₀) off over Q, then clear to primitive integer form.
        let f_rat: Vec<Rational> = f
            .iter()
            .map(|c| Rational::from_integer(c.clone()))
            .collect();
        let (quot, rem) = div_linear(&f_rat, x0);
        if rem != Rational::from_i64(0) {
            continue; // not actually a root (numerical drift upstream)
        }
        let g = clear_primitive(&quot);
        if zx::degree(&g) != 23 {
            continue;
        }
        if !is_irreducible_deg(&g, 23) {
            last = M23Witness {
                verdict: GroupVerdict::Unresolved,
                group23: None,
                residual: Some(g),
                x0: Some(x0.clone()),
                t0: Some(t0.clone()),
                disc_is_square: false,
                notes: vec!["degree-23 residual is reducible (not a 1+23 M23 split)".into()],
            };
            continue;
        }

        // Exact disc-square bit → parity class; Frobenius cycle types → the group.
        let disc = discriminant(&g);
        let disc_is_square = disc.is_perfect_square();
        let observed = frobenius_types(&g, primes, 23);
        let group = transitive23::classify(&observed, disc_is_square);

        let verdict = match group {
            Some(Group23::M23) => GroupVerdict::Confirmed(GroupKind::M23),
            Some(_) => GroupVerdict::SubgroupOnly,
            None => GroupVerdict::Unresolved,
        };
        let witness = M23Witness {
            verdict: verdict.clone(),
            group23: group,
            residual: Some(g),
            x0: Some(x0.clone()),
            t0: Some(t0),
            disc_is_square,
            notes: vec![format!(
                "irreducible degree-23 residual; disc_is_square={disc_is_square}; classify={:?}",
                group
            )],
        };
        if witness.is_m23_realized() {
            return witness;
        }
        last = witness;
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ri(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    /// A concrete (non-solution) coefficient vector for the pinned form, used to
    /// exercise the algebraic plumbing of `specialize_numerator` / `phi_at`.
    fn sample_coeffs() -> Vec<Rational> {
        let mut v: Vec<Rational> = Vec::new();
        v.extend([1, -2, 3, 0, -1, 2, 1, -3].iter().map(|&n| ri(n))); // a0..a7
        v.extend([2, 1, -1, 3, 0, -2, 1, 1].iter().map(|&n| ri(n))); // b0..b7
        v.extend([-1, 2, 1].iter().map(|&n| ri(n))); // r0..r2
        v.extend([3, -2, 1, 2].iter().map(|&n| ri(n))); // s0..s3
        v.push(Rational::new(3, 2).unwrap()); // lambda
        v.push(ri(1)); // c
        v
    }

    // ------------------------------------------------------------------
    // specialize_numerator: algebraic self-consistency.
    // ------------------------------------------------------------------
    #[test]
    fn specialize_numerator_matches_direct_evaluation() {
        let coeffs = sample_coeffs();
        let t0 = ri(3);
        let f_int = specialize_numerator(&coeffs, &t0);
        assert_eq!(zx::degree(&f_int), 24, "generic t0 gives a degree-24 numerator");

        // The primitive integer polynomial equals a positive rational multiple of
        // A²B − t0·λ·R⁵S; verify proportionality at several sample points.
        let fac = extract(&coeffs);
        let a2b = rat_mul(&rat_pow(&fac.a, 2), &fac.b);
        let r5s = rat_mul(&rat_pow(&fac.r, 5), &fac.s);
        let scalar = t0.clone() * fac.lambda.clone();
        let num = rat_sub(&a2b, &rat_scale(&r5s, &scalar));

        let f_rat: Vec<Rational> = f_int
            .iter()
            .map(|c| Rational::from_integer(c.clone()))
            .collect();

        // f_int(x) = k · num(x) for a fixed positive rational k, so the ratio is
        // constant across evaluation points.
        let mut ratio: Option<Rational> = None;
        for xv in [ri(2), ri(-1), ri(3), ri(5)] {
            let a = eval_rat(&num, &xv);
            let b = eval_rat(&f_rat, &xv);
            assert!(a != ri(0), "test point should not be a root");
            let k = b * rat_inv(&a).unwrap();
            match &ratio {
                None => ratio = Some(k),
                Some(prev) => assert_eq!(*prev, k, "primitive form is a constant multiple"),
            }
        }
    }

    // ------------------------------------------------------------------
    // phi_at makes x0 a root of the specialized numerator.
    // ------------------------------------------------------------------
    #[test]
    fn phi_at_gives_a_root_of_the_fibre() {
        let coeffs = sample_coeffs();
        let x0 = ri(2);
        let t0 = phi_at(&coeffs, &x0).expect("x0=2 is not a pole");
        let f = specialize_numerator(&coeffs, &t0);
        let f_rat: Vec<Rational> = f.iter().map(|c| Rational::from_integer(c.clone())).collect();
        assert_eq!(eval_rat(&f_rat, &x0), ri(0), "x0 is a root of f_{{t0}}");
        // and (x - x0) divides it exactly
        let (_q, rem) = div_linear(&f_rat, &x0);
        assert_eq!(rem, ri(0));
    }

    // ------------------------------------------------------------------
    // (a) transitive24 classify: synthetic M24 cycle-type set -> Confirmed(M24).
    // ------------------------------------------------------------------
    #[test]
    fn classify_m24_confirms_on_full_m24_support() {
        // Feed M24's exact cycle-type support (the Chebotarev-saturated sample):
        // the blind class is exactly {24T24680} -> Confirmed(M24).
        let support = match CycleTypeSupport::load_default() {
            Ok(s) => s,
            Err(_) => return, // data file absent in this checkout: skip
        };
        let m24: Vec<Vec<usize>> = support
            .by_t
            .get(&M24_T)
            .expect("M24 present in support")
            .iter()
            .cloned()
            .collect();
        assert_eq!(classify_m24(&m24), GroupVerdict::Confirmed(GroupKind::M24));
    }

    #[test]
    fn classify_m24_is_honest_on_partial_or_wrong_evidence() {
        let support = match CycleTypeSupport::load_default() {
            Ok(s) => s,
            Err(_) => return,
        };
        // A single 24-cycle EXCLUDES M24 — M24 has no order-24 element — so the
        // gate honestly refuses (Unresolved), never Confirmed.
        let cycle24 = vec![vec![24usize]];
        assert_eq!(classify_m24(&cycle24), GroupVerdict::Unresolved);

        // A type M24 *does* contain (a 23-cycle fixing one point) is consistent
        // with M24 but does not force it -> SubgroupOnly, never Confirmed.
        let m23_type = vec![vec![1usize, 23]];
        let v = classify_m24(&m23_type);
        assert!(
            v == GroupVerdict::SubgroupOnly || v == GroupVerdict::Unresolved,
            "a single M24 cycle type must not force M24, got {v:?}"
        );
        assert_ne!(v, GroupVerdict::Confirmed(GroupKind::M24));

        // The empty sample is Unresolved.
        assert_eq!(classify_m24(&[]), GroupVerdict::Unresolved);

        // A partial M24 sample (drop one type) is not the exact blind class, so it
        // must NOT be Confirmed(M24).
        let mut partial: Vec<Vec<usize>> = support
            .by_t
            .get(&M24_T)
            .unwrap()
            .iter()
            .cloned()
            .collect();
        partial.sort();
        partial.pop();
        assert_ne!(
            classify_m24(&partial),
            GroupVerdict::Confirmed(GroupKind::M24),
            "a strict subset of M24's support must not force M24"
        );
    }

    // ------------------------------------------------------------------
    // (b) specialize + Frobenius plumbing on a CONSTRUCTED degree-24 example
    //     whose group is controlled: an intransitive (reducible) polynomial must
    //     NOT be classified as M24 -> the plumbing distinguishes M24 from junk.
    // ------------------------------------------------------------------
    #[test]
    fn frobenius_plumbing_rejects_non_m24_polynomial() {
        // f = (x^12 - 2)(x^12 - 3): reducible over Q, so every Frobenius cycle
        // type splits into the two degree-12 blocks — a single 24-cycle can never
        // occur, so M24 (which contains 24-cycles) is excluded.
        let mut f12a = vec![Integer::from(0); 13];
        f12a[0] = Integer::from(-2);
        f12a[12] = Integer::from(1); // x^12 - 2
        let mut f12b = vec![Integer::from(0); 13];
        f12b[0] = Integer::from(-3);
        f12b[12] = Integer::from(1); // x^12 - 3
        let prod = zx_mul(&f12a, &f12b);
        assert_eq!(zx::degree(&prod), 24);

        let primes = [7i64, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73];
        let observed = frobenius_types(&prod, &primes, 24);
        assert!(!observed.is_empty(), "collected some Frobenius cycle types");
        // Every observed type must have all parts dividing into two 12-blocks
        // (in particular no part of size > 12), so it is NOT confirmed as M24.
        assert_ne!(
            classify_m24(&observed),
            GroupVerdict::Confirmed(GroupKind::M24),
            "a reducible degree-24 polynomial is not the M24 cover"
        );
    }

    fn zx_mul(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
        let mut c = vec![Integer::from(0); a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                c[i + j] = c[i + j].clone() + ai.clone() * bj.clone();
            }
        }
        c
    }

    // ------------------------------------------------------------------
    // (c) audit_m23_residual correctly forms + irreducibility-checks a 1+23
    //     split, and (d) is honest when it cannot force M23.
    // ------------------------------------------------------------------
    #[test]
    fn residual_split_forms_degree_23_and_is_honest() {
        // On the sample (non-solution) coefficients the residual is generic, not
        // an M23 polynomial — the audit must NOT falsely claim M23.
        let coeffs = sample_coeffs();
        let x0s = [ri(2), ri(-1), ri(3), Rational::new(1, 2).unwrap()];
        let primes = [7i64, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
        let w = audit_m23_residual(&coeffs, &x0s, &primes);
        assert!(
            !w.is_m23_realized(),
            "a non-solution cover must not be certified M23/Q"
        );
        // But when a residual formed, it is genuinely degree 23.
        if let Some(g) = &w.residual {
            assert_eq!(zx::degree(g), 23, "residual is the degree-23 1+23 split");
        }
    }

    // ------------------------------------------------------------------
    // transitive23 classify reconfirmed through this path (Wave-1 classifier).
    // ------------------------------------------------------------------
    #[test]
    fn transitive23_classifies_synthetic_m23() {
        // M23's full documented cycle-type fingerprint, disc a square -> M23.
        let full: Vec<Vec<usize>> = Group23::M23
            .type_set()
            .expect("M23 has a finite fingerprint")
            .into_iter()
            .collect();
        assert_eq!(transitive23::classify(&full, true), Some(Group23::M23));
        // The same fingerprint with disc NOT a square cannot be M23 (M23 ⊆ A23).
        assert_ne!(transitive23::classify(&full, false), Some(Group23::M23));
    }
}
