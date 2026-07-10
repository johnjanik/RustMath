//! Rank bounds for E(Q) via genuine 2-descent.
//!
//! Two classical descent methods are implemented, both fully exact:
//!
//! **1. Descent via 2-isogeny** (Silverman AEC X.4.9; Silverman–Tate,
//! *Rational Points on Elliptic Curves*, III.4–III.6). For a curve with a
//! rational 2-torsion point moved to the origin,
//! E: y² = x³ + ax² + bx, the 2-isogenous curve is
//! E′: y² = x³ − 2ax² + (a² − 4b)x, and the connecting homomorphism
//! α : E(Q) → Q\*/(Q\*)² (O ↦ 1, (0,0) ↦ b, (x,y) ↦ x) satisfies the exact
//! rank formula
//!
//! ```text
//!     |Im α| · |Im α′| = 2^(rank + 2).
//! ```
//!
//! A square-free class b₁ (with b₂ = b/b₁) lies in Im α iff the torsor
//!
//! ```text
//!     N² = b₁M⁴ + aM²e² + b₂e⁴
//! ```
//!
//! has a rational point with (M, e) ≠ (0, 0). Since Im α ⊆ Sel ⊆
//! {everywhere-locally-solvable classes}, bounded global search gives a
//! certified **lower** bound and local solvability a certified **upper**
//! bound on the rank.
//!
//! **2. Full 2-descent** when E[2] ⊆ E(Q), i.e. y² = (x−e₁)(x−e₂)(x−e₃)
//! with eᵢ ∈ Z: the descent map P ↦ (class(x−e₁), class(x−e₂)) embeds
//! E(Q)/2E(Q) into (Q\*/(Q\*)²)², and |E(Q)/2E(Q)| = 2^(rank+2). The
//! (b₁, b₂)-torsor is the intersection of two quadrics; with d₂ = e₂−e₁,
//! d₃ = e₃−e₁ and homogeneous coordinates (z₀ : z₁) it is solvable iff
//! b₂(b₁z₁² − d₂z₀²) and b₁b₂(b₁z₁² − d₃z₀²) are simultaneously squares.
//!
//! # Local solvability (exact, certified)
//!
//! Membership of a class in the Selmer set is decided place by place, at ∞
//! and at every prime p | 2·b·(a²−4b) (all other places are automatically
//! solvable: the torsor is a smooth genus-1 curve with good reduction there,
//! has F_p-points by Hasse–Weil, and they lift by Hensel).
//!
//! Q_p-solvability is decided by an exact recursion over residue discs
//! r + p^k Z_p:
//! * the center value g(r) is an exact integer, so "g(r) is a square in
//!   Q_p" is exactly decidable (even valuation + unit part a square:
//!   Legendre/Euler for odd p, ≡ 1 mod 8 for p = 2);
//! * if v = v_p(g(r)) satisfies k ≥ v + 1 (odd p) or k ≥ v + 3 (p = 2),
//!   the square-class of g is *constant on the whole disc* (g(x) ≡ g(r)
//!   mod p^k for x in the disc), so the disc can be certified insolvable;
//! * if v_p(g(r)) > 2·v_p(g′(r)) and v_p(g(r)) − v_p(g′(r)) ≥ k, Newton
//!   iteration converges to a root of g inside the disc (Hensel), so g
//!   attains 0 (a torsor point with N = 0) — solvable;
//! * otherwise the disc splits into its p sub-discs.
//!
//! Termination: an infinite recursion path would converge to θ ∈ Z_p with
//! g(θ) = 0 and v_p(g′(θ)) unbounded, i.e. a double root, contradicting
//! disc(g) ≠ 0 (the torsor quartic has discriminant 16b(a² − 4b)² ≠ 0 for
//! a nonsingular curve). The recursion depth is capped and the cap treated
//! as an internal bug (panic), never as an answer.
//!
//! This decision procedure was cross-checked against a brute-force p-adic
//! oracle on thousands of random (b₁, a, b₂, p) instances (single quartics
//! and quadric pairs) in an independent Python reference implementation
//! before being ported here.
//!
//! # Honesty contract
//!
//! * Results are **intervals** `[lower, upper]`, never fabricated integers.
//!   `lower` counts only classes *witnessed by explicit rational points*
//!   (each re-verified on the curve); `upper` counts everywhere-locally-
//!   solvable classes.
//! * An everywhere-locally-solvable torsor with no global point found by
//!   the bounded search is an **unresolved** bit (a candidate element of
//!   Sha); it keeps the interval open. The interval is *never* collapsed by
//!   fiat. (Example: y² = x³ − 17²x, the classical non-congruent-number
//!   curve with Sha(E)[2] ≅ (Z/2)², stays at [0, 2] here.)
//! * Curves with no rational 2-torsion (irreducible 2-division polynomial,
//!   e.g. 37a1, 389a1) are honestly reported as
//!   [`RankBoundResult::Unresolved`]: descent for them lives over a number
//!   field, which is out of scope.
//! * Every intermediate model change is a certified
//!   [`WeierstrassIsomorphism`] (all five transformation equations checked
//!   exactly); every witness point is checked on-curve at every stage; the
//!   found and Selmer class sets are certified to be subgroups of
//!   Q\*/(Q\*)² with the found set contained in the Selmer set.
//!
//! Factorization uses `rustmath_integers::prime::factor` (trial division),
//! which is fine for the coefficient sizes handled here; huge-discriminant
//! curves inherit that cost profile.
//!
//! Every expected value asserted in this module's tests was derived and
//! verified with the independent Python reference implementation (including
//! the brute-force local-solvability oracle) before being written down.

use crate::curve::{EllipticCurve, Point};
use crate::minimal::WeierstrassIsomorphism;
use crate::torsion::integer_cubic_roots;
use rustmath_core::Ring;
use rustmath_integers::prime::factor;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// Default bounded-search radius for torsor points (numerators and
/// denominators of the parametrizing coordinate up to this size).
pub const DEFAULT_SEARCH_BOUND: u64 = 40;

/// Result of a rank computation: an honest interval, or an honest refusal.
#[derive(Debug, Clone)]
pub enum RankBoundResult {
    /// Certified bounds `lower ≤ rank E(Q) ≤ upper`.
    Bounds(RankBounds),
    /// This machinery cannot bound the rank (currently: the 2-division
    /// polynomial is irreducible over Q, so 2-descent lives over a number
    /// field, which is out of scope here). Never a guess.
    Unresolved { reason: String },
}

/// Certified rank bounds from 2-descent.
#[derive(Debug, Clone)]
pub struct RankBounds {
    /// Certified lower bound: witnessed by explicit rational points.
    pub lower: u32,
    /// Certified upper bound: from everywhere-local solvability (Selmer).
    pub upper: u32,
    /// All rational points found on the torsors, mapped back to (and
    /// re-verified on) the *input* model. May include torsion points.
    pub points: Vec<Point>,
    /// Number of everywhere-locally-solvable descent classes for which the
    /// bounded search found no global point, taken from the descent method
    /// that achieved `upper` — candidate Sha elements. Set to 0 when
    /// `lower == upper` (the gap is closed, so no class blocks the rank).
    pub unresolved_classes: u32,
}

impl RankBounds {
    /// The subset of [`Self::points`] of infinite order on `e` (the same
    /// input curve the bounds were computed for). These are genuine
    /// witnesses of positive rank: each has canonical height > 0.
    pub fn infinite_order_points(&self, e: &EllipticCurve) -> Vec<Point> {
        self.points
            .iter()
            .filter(|p| e.point_order(p).is_none())
            .cloned()
            .collect()
    }
}

impl EllipticCurve {
    /// Certified rank bounds via 2-descent with the default search bound.
    /// See the module docs of [`crate::rank`] for the exact semantics.
    pub fn rank_bounds(&self) -> RankBoundResult {
        self.rank_bounds_with_search(DEFAULT_SEARCH_BOUND)
    }

    /// Certified rank bounds via 2-descent, searching torsors with the
    /// given bound. A larger bound can only raise `lower` (find more
    /// points); `upper` does not depend on it.
    ///
    /// # Panics
    ///
    /// Panics if the curve is singular, or if an internal certification
    /// step fails (which indicates a bug, never a wrong answer).
    pub fn rank_bounds_with_search(&self, search_bound: u64) -> RankBoundResult {
        rank_bounds_impl(self, search_bound)
    }
}

// ---------------------------------------------------------------------------
// orchestrator
// ---------------------------------------------------------------------------

/// The model E_s with a1 = a3 = 0 used for descent, the certified
/// isomorphism input → E_s, and the integer x-coordinates of the rational
/// 2-torsion points of E_s.
pub(crate) struct ShortModel {
    pub eshort: EllipticCurve,
    pub iso_input_to_short: WeierstrassIsomorphism,
    pub roots: Vec<Integer>,
}

/// Compute a descent-ready model: minimal model, then (only if a1 or a3 is
/// nonzero) the (36, 108)-scaled model y² = x³ − 27c₄x − 54c₆. Keeping the
/// minimal model when it already has a1 = a3 = 0 keeps torsor coefficients
/// small, which matters for the bounded search.
pub(crate) fn short_model(e: &EllipticCurve) -> ShortModel {
    let (emin, iso_to_min) = e.minimal_model();
    let (eshort, iso_min_to_short) = if emin.a1.is_zero() && emin.a3.is_zero() {
        (emin.clone(), WeierstrassIsomorphism::identity())
    } else {
        let (b2, _, _, _) = emin.b_invariants();
        let (c4, c6) = emin.c_invariants();
        let es =
            EllipticCurve::from_short_weierstrass(Integer::from(-27) * c4, Integer::from(-54) * c6);
        let q = Rational::from_integer;
        let iso = WeierstrassIsomorphism {
            u: Rational::new(Integer::one(), Integer::from(6)).expect("1/6"),
            r: -(q(b2.clone()) / Rational::from_i64(12)),
            s: -(q(emin.a1.clone()) / Rational::from_i64(2)),
            t: q(emin.a1.clone()) * q(b2) / Rational::from_i64(24)
                - q(emin.a3.clone()) / Rational::from_i64(2),
        };
        assert!(
            iso.is_isomorphism(&emin, &es),
            "rank: minimal → short transformation failed certification"
        );
        (es, iso)
    };

    // Integer roots of x³ + a2x² + a4x + a6 via the depressed cubic
    // X³ − 27c4X − 54c6 (X = 36x + 3b2): rational roots of a monic integer
    // cubic are integers, and the substitution is a bijection on roots.
    let (b2s, _, _, _) = eshort.b_invariants();
    let (c4s, c6s) = eshort.c_invariants();
    let mut roots = Vec::new();
    for bigroot in integer_cubic_roots(&(Integer::from(-27) * c4s), &(Integer::from(-54) * c6s)) {
        let num = bigroot - Integer::from(3) * b2s.clone();
        let r = &num / &Integer::from(36);
        assert!(
            &r * &Integer::from(36) == num,
            "rank: depressed-cubic root does not map to an integer root"
        );
        // verify exactly on the a1 = a3 = 0 model
        let val = r.pow(3) + &eshort.a2 * &(&r * &r) + &eshort.a4 * &r + eshort.a6.clone();
        assert!(val.is_zero(), "rank: mapped-back root is not a root");
        roots.push(r);
    }
    roots.sort();

    ShortModel {
        eshort,
        iso_input_to_short: iso_to_min.compose(&iso_min_to_short),
        roots,
    }
}

fn rank_bounds_impl(e: &EllipticCurve, h: u64) -> RankBoundResult {
    assert!(
        !e.is_singular(),
        "rank_bounds: curve is singular (discriminant 0)"
    );
    let sm = short_model(e);
    if sm.roots.is_empty() {
        return RankBoundResult::Unresolved {
            reason: "no rational 2-torsion: the 2-division polynomial is irreducible over Q, \
                     so 2-descent requires working over a number field (out of scope); \
                     no rank bound is fabricated"
                .to_string(),
        };
    }
    let back = sm.iso_input_to_short.inverse();

    let mut lower = 0u32;
    let mut upper = u32::MAX;
    let mut unresolved_at_upper = u32::MAX;
    let mut points: Vec<Point> = Vec::new();

    let mut merge = |out: &DescentOutcome, lower: &mut u32, upper: &mut u32| {
        *lower = (*lower).max(out.lower);
        if out.upper < *upper {
            *upper = out.upper;
            unresolved_at_upper = out.unresolved;
        } else if out.upper == *upper {
            unresolved_at_upper = unresolved_at_upper.min(out.unresolved);
        }
    };

    // Descent via 2-isogeny from every rational 2-torsion point.
    for x0 in &sm.roots {
        // substitute x → x + x0 into x³ + a2x² + a4x + a6 (a root at x0):
        // y² = x³ + a x² + b x with a = a2 + 3x0, b = 3x0² + 2a2x0 + a4.
        let a = sm.eshort.a2.clone() + Integer::from(3) * x0.clone();
        let b = Integer::from(3) * x0.clone() * x0.clone()
            + Integer::from(2) * sm.eshort.a2.clone() * x0.clone()
            + sm.eshort.a4.clone();
        let eshift = curve_ab(&a, &b);
        let iso_short_to_shift = WeierstrassIsomorphism {
            u: Rational::one(),
            r: Rational::from_integer(x0.clone()),
            s: Rational::zero(),
            t: Rational::zero(),
        };
        assert!(
            iso_short_to_shift.is_isomorphism(&sm.eshort, &eshift),
            "rank: short → shifted transformation failed certification"
        );
        let out = isogeny_descent(&a, &b, h);
        let shift_back = iso_short_to_shift.inverse();
        for p in &out.points {
            let q = back.map_point(&shift_back.map_point(p));
            assert!(
                e.is_on_curve(&q),
                "rank: mapped-back torsor point not on the input curve"
            );
            points.push(q);
        }
        merge(&out, &mut lower, &mut upper);
    }

    // Full 2-descent when all of E[2] is rational.
    if sm.roots.len() == 3 {
        let out = full_two_descent(&sm.roots[0], &sm.roots[1], &sm.roots[2], h, &sm.eshort);
        for p in &out.points {
            let q = back.map_point(p);
            assert!(
                e.is_on_curve(&q),
                "rank: mapped-back 2-descent point not on the input curve"
            );
            points.push(q);
        }
        merge(&out, &mut lower, &mut upper);
    }

    assert!(
        lower <= upper,
        "rank: descent methods produced contradictory bounds [{}, {}]: bug",
        lower,
        upper
    );
    points.sort_by(|p1, p2| (&p1.x, &p1.y).cmp(&(&p2.x, &p2.y)));
    points.dedup();
    let unresolved_classes = if lower == upper {
        0
    } else {
        unresolved_at_upper
    };
    RankBoundResult::Bounds(RankBounds {
        lower,
        upper,
        points,
        unresolved_classes,
    })
}

/// y² = x³ + ax² + bx as an [`EllipticCurve`].
fn curve_ab(a: &Integer, b: &Integer) -> EllipticCurve {
    EllipticCurve::new(
        Integer::zero(),
        a.clone(),
        Integer::zero(),
        b.clone(),
        Integer::zero(),
    )
}

// ---------------------------------------------------------------------------
// descent via 2-isogeny
// ---------------------------------------------------------------------------

/// Outcome of one descent method: certified bounds, witness points on the
/// model the method ran on, and the count of locally-solvable-but-unwitnessed
/// classes.
struct DescentOutcome {
    lower: u32,
    upper: u32,
    points: Vec<Point>,
    unresolved: u32,
}

/// One α-side of the 2-isogeny descent, for E: y² = x³ + ax² + bx.
pub(crate) struct AlphaSide {
    /// Subgroup of Q*/(Q*)² generated by point-witnessed classes.
    pub found: Vec<Integer>,
    /// Everywhere-locally-solvable classes (contains `found`; certified to
    /// be a subgroup).
    pub selmer: Vec<Integer>,
    /// Torsor-derived points on y² = x³ + ax² + bx (verified).
    pub points: Vec<Point>,
    /// Locally-solvable non-automatic classes without a global point found.
    pub unresolved: u32,
}

/// Compute bounds on Im(α) for E: y² = x³ + ax² + bx (b(a²−4b) ≠ 0).
pub(crate) fn alpha_side(a: &Integer, b: &Integer, h: u64) -> AlphaSide {
    let e_ab = curve_ab(a, b);
    let bclass = squarefree_part(b);
    let disc_q = a * a - Integer::from(4) * b.clone();
    assert!(
        !b.is_zero() && !disc_q.is_zero(),
        "alpha_side: degenerate (singular) model"
    );
    let bad = bad_primes(&(Integer::from(2) * b.clone() * disc_q.clone()));

    let mut found: Vec<Integer> = vec![Integer::one(), bclass.clone()];
    let mut selmer: Vec<Integer> = Vec::new();
    let mut points: Vec<Point> = Vec::new();
    let mut unresolved = 0u32;

    for b1 in sf_divisors(b) {
        let b2 = exact_div(b, &b1);
        let auto = b1.is_one() || b1 == bclass;
        let loc = real_soluble_quartic(&b1, a, &b2)
            && bad.iter().all(|p| qp_soluble_quartic(&b1, a, &b2, p));
        if auto {
            // 1 = α(O) and class(b) = α((0,0)) have global points, so a
            // local obstruction for them means the local solver is broken.
            assert!(
                loc,
                "alpha_side: automatic class {} locally insolvable: bug",
                b1
            );
        }
        if !loc {
            continue;
        }
        selmer.push(b1.clone());
        // Bounded global search on N² = b1·M⁴ + a·M²e² + b2·e⁴.
        if let Some((m, e, n)) = torsor_search(&b1, a, &b2, h) {
            let x = Rational::new(&b1 * &(&m * &m), &e * &e).expect("e nonzero");
            let y = Rational::new(&b1 * &(&m * &n), e.pow(3)).expect("e nonzero");
            let pt = Point::new(x, y);
            assert!(
                e_ab.is_on_curve(&pt),
                "alpha_side: torsor solution maps off-curve: bug"
            );
            points.push(pt);
            if !auto && !found.contains(&b1) {
                found.push(b1);
            }
        } else if !auto {
            unresolved += 1;
        }
    }

    let found = group_closure_1(&found);
    assert!(
        group_closed_1(&selmer),
        "alpha_side: Selmer class set is not a subgroup of Q*/(Q*)²: bug"
    );
    assert!(
        found.iter().all(|c| selmer.contains(c)),
        "alpha_side: point-witnessed class fails local solvability: bug"
    );
    selmer.sort_by(class_order);
    AlphaSide {
        found,
        selmer,
        points,
        unresolved,
    }
}

/// Full descent via the 2-isogeny φ: E → E′ and its dual, for
/// E: y² = x³ + ax² + bx. Returns bounds and points on the (a, b) model.
fn isogeny_descent(a: &Integer, b: &Integer, h: u64) -> DescentOutcome {
    let bp = a * a - Integer::from(4) * b.clone();
    let ap = Integer::from(-2) * a.clone();
    let side_e = alpha_side(a, b, h);
    let side_ep = alpha_side(&ap, &bp, h);

    let dims_found = dim2(side_e.found.len()) + dim2(side_ep.found.len());
    let dims_selmer = dim2(side_e.selmer.len()) + dim2(side_ep.selmer.len());
    assert!(
        dims_selmer >= 2,
        "isogeny_descent: |Sel φ|·|Sel φ'| < 4 contradicts |Im α|·|Im α'| = 2^(r+2): bug"
    );
    let lower = dims_found.saturating_sub(2);
    let upper = dims_selmer - 2;
    assert!(lower <= upper);

    // E-side points live on E already; E′-side points are pushed to E
    // through the dual isogeny φ̂: (X, Y) ↦ (Y²/(4X²), Y(X²−b′)/(8X²)).
    let e_ab = curve_ab(a, b);
    let mut points = side_e.points;
    for p in &side_ep.points {
        if p.x.is_zero() {
            continue; // (0,0) generates ker φ̂, maps to O
        }
        let x2 = &p.x * &p.x;
        let four_x2 = Rational::from_i64(4) * x2.clone();
        let x = (&p.y * &p.y) / four_x2;
        let y = p.y.clone() * (x2 - Rational::from_integer(bp.clone()))
            / (Rational::from_i64(8) * p.x.clone() * p.x.clone());
        let q = Point::new(x, y);
        assert!(
            e_ab.is_on_curve(&q),
            "isogeny_descent: dual-isogeny image off-curve: bug"
        );
        points.push(q);
    }

    DescentOutcome {
        lower,
        upper,
        points,
        unresolved: side_e.unresolved + side_ep.unresolved,
    }
}

/// Bounded search for N² = b1·M⁴ + a·M²e² + b2·e⁴ with gcd(M, e) = 1,
/// 1 ≤ M, e ≤ h. (Solutions with M = 0 or e = 0 only witness the automatic
/// classes, which need no witness.)
fn torsor_search(
    b1: &Integer,
    a: &Integer,
    b2: &Integer,
    h: u64,
) -> Option<(Integer, Integer, Integer)> {
    for e in 1..=h {
        for m in 1..=h {
            if gcd_u64(m, e) != 1 {
                continue;
            }
            let mi = Integer::from(m as i64);
            let ei = Integer::from(e as i64);
            let m2 = &mi * &mi;
            let e2 = &ei * &ei;
            let rhs = b1 * &(&m2 * &m2) + a * &(&m2 * &e2) + b2 * &(&e2 * &e2);
            if rhs.signum() >= 0 && rhs.is_perfect_square() {
                let n = rhs.sqrt().expect("nonnegative");
                return Some((mi, ei, n));
            }
        }
    }
    None
}

/// Real solvability of N² = b1M⁴ + aM²e² + b2e⁴, (M, e) ≠ (0, 0): with
/// u = (M/e)² ≥ 0 the quadratic q(u) = b1u² + au + b2 must reach ≥ 0 on
/// u ∈ [0, ∞]. u = ∞ gives b1 > 0, u = 0 gives b2 > 0; otherwise (both
/// negative) the maximum of the downward parabola on u ≥ 0 is at
/// u* = a/(2|b1|), which is ≥ 0 iff a > 0, with value (a²−4b1b2)/(4|b1|).
fn real_soluble_quartic(b1: &Integer, a: &Integer, b2: &Integer) -> bool {
    if b1.signum() > 0 || b2.signum() > 0 {
        return true;
    }
    a.signum() > 0 && (a * a - Integer::from(4) * (b1 * b2)).signum() >= 0
}

/// Q_p-solvability of N² = b1M⁴ + aM²e² + b2e⁴ on P¹: chart t = M/e ∈ Z_p
/// plus chart s = e/M ∈ pZ_p (the reversed quartic), which together cover
/// all of P¹(Q_p).
fn qp_soluble_quartic(b1: &Integer, a: &Integer, b2: &Integer, p: &Integer) -> bool {
    let f = biquadratic(b1, a, b2);
    let g = biquadratic(b2, a, b1);
    let zero = Integer::zero();
    soluble_disc(&f, &f.derivative(), p, &zero, 0, 0)
        || soluble_disc(&g, &g.derivative(), p, &zero, 1, 0)
}

// ---------------------------------------------------------------------------
// full 2-descent (E[2] ⊆ E(Q))
// ---------------------------------------------------------------------------

/// Full 2-descent on y² = (x−e1)(x−e2)(x−e3) (distinct integers, given as
/// the roots of `eshort`'s cubic). Returns bounds and points on `eshort`.
fn full_two_descent(
    e1: &Integer,
    e2: &Integer,
    e3: &Integer,
    h: u64,
    eshort: &EllipticCurve,
) -> DescentOutcome {
    // certify that eshort really is y² = (x−e1)(x−e2)(x−e3)
    assert!(eshort.a1.is_zero() && eshort.a3.is_zero());
    assert!(e1.clone() + e2.clone() + e3.clone() == -eshort.a2.clone());
    assert!(e1 * e2 + e1 * e3 + e2 * e3 == eshort.a4);
    assert!(-((e1 * e2) * e3.clone()) == eshort.a6);

    let d2 = e2.clone() - e1.clone();
    let d3 = e3.clone() - e1.clone();
    let d23 = e3.clone() - e2.clone();
    assert!(!d2.is_zero() && !d3.is_zero() && !d23.is_zero());

    let cand1 = sf_divisors(&(&d2 * &d3));
    let cand2 = sf_divisors(&(&d2 * &(&d2 - &d3))); // (e2−e1)(e2−e3)
    let bad = bad_primes(&(Integer::from(2) * (&d2 * &d3) * d23.clone()));
    let supp3 = &d3 * &d23; // (e3−e1)(e3−e2), support of class(x−e3)

    // Automatic image classes: O and the three 2-torsion points, with the
    // standard conventions of the descent map at 2-torsion.
    let autos: Vec<(Integer, Integer)> = {
        let mut v: Vec<(Integer, Integer)> = Vec::new();
        for c in [
            (Integer::one(), Integer::one()),
            (squarefree_part(&(&d2 * &d3)), squarefree_part(&-d2.clone())),
            (squarefree_part(&d2), squarefree_part(&(&d2 * &(&d2 - &d3)))),
            (squarefree_part(&d3), squarefree_part(&(&d3 - &d2))),
        ] {
            if !v.contains(&c) {
                v.push(c);
            }
        }
        v
    };

    let mut found: Vec<(Integer, Integer)> = autos.clone();
    let mut selmer: Vec<(Integer, Integer)> = Vec::new();
    let mut points: Vec<Point> = Vec::new();
    let mut unresolved = 0u32;

    for b1 in &cand1 {
        'pair: for b2 in &cand2 {
            // support filter: class(x−e3) = sf(b1·b2) must be supported on
            // the primes of (e3−e1)(e3−e2)
            let b3 = squarefree_part(&(b1 * b2));
            for (q, _) in factor(&b3.abs()) {
                if !(&supp3 % &q).is_zero() {
                    assert!(
                        !autos.contains(&(b1.clone(), b2.clone())),
                        "full_two_descent: automatic class filtered by support: bug"
                    );
                    continue 'pair;
                }
            }
            // torsor: b2(b1z1² − d2z0²) and b1b2(b1z1² − d3z0²) both squares
            let a1c = b1 * b2;
            let c1c = -(b2 * &d2);
            let a2c = &(b1 * b1) * b2;
            let c2c = -(&(b1 * b2) * &d3);
            let auto = autos.contains(&(b1.clone(), b2.clone()));
            let loc = real_soluble_pair(&a1c, &c1c, &a2c, &c2c)
                && bad
                    .iter()
                    .all(|p| qp_soluble_pair(&a1c, &c1c, &a2c, &c2c, p));
            if auto {
                assert!(
                    loc,
                    "full_two_descent: automatic class ({}, {}) locally insolvable: bug",
                    b1, b2
                );
            }
            if !loc {
                continue;
            }
            selmer.push((b1.clone(), b2.clone()));
            if auto {
                continue;
            }
            // bounded global search: (z1 : z0) = (s : t), gcd(s, t) = 1
            let mut witness = None;
            's_loop: for t in 0..=h {
                for s in 0..=h {
                    if gcd_u64(s, t) != 1 {
                        continue; // also skips (0, 0)
                    }
                    let si = Integer::from(s as i64);
                    let ti = Integer::from(t as i64);
                    let s2 = &si * &si;
                    let t2 = &ti * &ti;
                    let k1 = b1 * &s2 - &d2 * &t2;
                    let k2 = b1 * &s2 - &d3 * &t2;
                    if (b2 * &k1).is_perfect_square() && (&(b1 * b2) * &k2).is_perfect_square() {
                        witness = Some((si, ti));
                        break 's_loop;
                    }
                }
            }
            match witness {
                Some((_, t)) if t.is_zero() => {
                    // (1 : 0) forces b1, b2 to be squares — the class (1,1),
                    // which is automatic; nothing new. (Reachable only for
                    // the trivial class, which never gets here.)
                    unreachable!("full_two_descent: (s:t)=(1:0) witnesses only the trivial class")
                }
                Some((s, t)) => {
                    let x = Rational::new(b1 * &(&s * &s), &t * &t).expect("t nonzero")
                        + Rational::from_integer(e1.clone());
                    let yy = (x.clone() - Rational::from_integer(e1.clone()))
                        * (x.clone() - Rational::from_integer(e2.clone()))
                        * (x.clone() - Rational::from_integer(e3.clone()));
                    let y = rational_sqrt(&yy)
                        .expect("full_two_descent: torsor solution gives non-square y²: bug");
                    let pt = Point::new(x, y);
                    assert!(
                        eshort.is_on_curve(&pt),
                        "full_two_descent: torsor point off-curve: bug"
                    );
                    points.push(pt);
                    found.push((b1.clone(), b2.clone()));
                }
                None => unresolved += 1,
            }
        }
    }

    let found = group_closure_2(&found);
    assert!(
        group_closed_2(&selmer),
        "full_two_descent: Selmer set is not a subgroup: bug"
    );
    assert!(
        found.iter().all(|c| selmer.contains(c)),
        "full_two_descent: point-witnessed class fails local solvability: bug"
    );
    let dim_selmer = dim2(selmer.len());
    assert!(
        dim_selmer >= 2,
        "full_two_descent: |Sel₂| < 4 contradicts |E(Q)/2E(Q)| = 2^(r+2): bug"
    );
    let lower = dim2(found.len()).saturating_sub(2);
    let upper = dim_selmer - 2;
    assert!(lower <= upper);

    DescentOutcome {
        lower,
        upper,
        points,
        unresolved,
    }
}

/// Real solvability of the pair A1z1² + C1z0² ≥ 0 ∧ A2z1² + C2z0² ≥ 0 with
/// (z0, z1) ≠ 0: writing u = (z1/z0)² ∈ [0, ∞], the closed feasible set is
/// an intersection of two closed half-lines in u, so it is nonempty iff one
/// of the four candidate boundary values u ∈ {0, ∞, −C1/A1, −C2/A2} is
/// feasible. (A1, A2 ≠ 0 always: they are ±products of the nonzero bᵢ.)
fn real_soluble_pair(a1: &Integer, c1: &Integer, a2: &Integer, c2: &Integer) -> bool {
    assert!(!a1.is_zero() && !a2.is_zero());
    if c1.signum() >= 0 && c2.signum() >= 0 {
        return true; // u = 0
    }
    if a1.signum() > 0 && a2.signum() > 0 {
        return true; // u = ∞
    }
    let q = |n: &Integer, d: &Integer| {
        Rational::new(n.clone(), d.clone()).expect("nonzero denominator")
    };
    let zero = Rational::zero();
    let u1 = q(&-c1.clone(), a1);
    if u1 >= zero
        && Rational::from_integer(a2.clone()) * u1.clone() + Rational::from_integer(c2.clone())
            >= zero
    {
        return true;
    }
    let u2 = q(&-c2.clone(), a2);
    if u2 >= zero
        && Rational::from_integer(a1.clone()) * u2.clone() + Rational::from_integer(c1.clone())
            >= zero
    {
        return true;
    }
    false
}

/// Q_p-solvability of "A1x² + C1 and A2x² + C2 simultaneously squares" for
/// x ∈ P¹(Q_p) (chart x ∈ Z_p plus reversed chart in pZ_p).
fn qp_soluble_pair(a1: &Integer, c1: &Integer, a2: &Integer, c2: &Integer, p: &Integer) -> bool {
    let f = quadratic(a1, c1);
    let g = quadratic(a2, c2);
    let fr = quadratic(c1, a1);
    let gr = quadratic(c2, a2);
    let zero = Integer::zero();
    soluble_disc_pair(&f, &f.derivative(), &g, &g.derivative(), p, &zero, 0, 0)
        || soluble_disc_pair(&fr, &fr.derivative(), &gr, &gr.derivative(), p, &zero, 1, 0)
}

// ---------------------------------------------------------------------------
// exact p-adic solvability primitives
// ---------------------------------------------------------------------------

/// Hard cap on the residue-disc recursion. Mathematically the recursion
/// terminates for squarefree polynomials (an infinite path would produce a
/// p-adic double root); exceeding the cap is treated as a bug, never as an
/// answer.
const MAX_DISC_DEPTH: u32 = 96;

/// Dense integer polynomial (c[i] = coefficient of x^i).
#[derive(Clone)]
struct IntPoly {
    c: Vec<Integer>,
}

impl IntPoly {
    fn eval(&self, x: &Integer) -> Integer {
        let mut acc = Integer::zero();
        for c in self.c.iter().rev() {
            acc = acc * x.clone() + c.clone();
        }
        acc
    }

    fn derivative(&self) -> IntPoly {
        let mut c = Vec::with_capacity(self.c.len().saturating_sub(1));
        for (i, ci) in self.c.iter().enumerate().skip(1) {
            c.push(Integer::from(i as i64) * ci.clone());
        }
        IntPoly { c }
    }
}

/// b1x⁴ + ax² + b2.
fn biquadratic(b1: &Integer, a: &Integer, b2: &Integer) -> IntPoly {
    IntPoly {
        c: vec![
            b2.clone(),
            Integer::zero(),
            a.clone(),
            Integer::zero(),
            b1.clone(),
        ],
    }
}

/// ax² + c.
fn quadratic(a: &Integer, c: &Integer) -> IntPoly {
    IntPoly {
        c: vec![c.clone(), Integer::zero(), a.clone()],
    }
}

/// (v_p(n), unit part of n); n ≠ 0.
fn vp_unit(n: &Integer, p: &Integer) -> (u32, Integer) {
    debug_assert!(!n.is_zero());
    let v = n.valuation(p);
    let mut u = n.clone();
    for _ in 0..v {
        u = u / p.clone();
    }
    (v, u)
}

/// Is the p-adic unit u a square in Z_p? (u ≡ 1 mod 8 for p = 2; Euler's
/// criterion for odd p.)
fn unit_is_square(u: &Integer, p: &Integer) -> bool {
    if *p == Integer::from(2) {
        u.modulo(&Integer::from(8)).is_one()
    } else {
        let e = (p.clone() - Integer::one()) / Integer::from(2);
        u.modulo(p)
            .modpow(&e, p)
            .expect("modpow with odd prime modulus")
            .is_one()
    }
}

/// Is the nonzero integer n a square in Q_p?
fn is_qp_square(n: &Integer, p: &Integer) -> bool {
    let (v, u) = vp_unit(n, p);
    v % 2 == 0 && unit_is_square(&u, p)
}

/// Does there exist x ∈ Z_p with x ≡ r (mod p^k) such that f(x) is a square
/// in Q_p (including 0)? See the module docs for the case analysis and the
/// termination argument.
fn soluble_disc(f: &IntPoly, fp: &IntPoly, p: &Integer, r: &Integer, k: u32, depth: u32) -> bool {
    assert!(
        depth < MAX_DISC_DEPTH,
        "soluble_disc: recursion depth exceeded (degenerate quartic?): bug"
    );
    let d = f.eval(r);
    if d.is_zero() {
        return true; // exact root: N = 0
    }
    let (v, _) = vp_unit(&d, p);
    if is_qp_square(&d, p) {
        return true; // x = r itself works
    }
    let dp = fp.eval(r);
    if !dp.is_zero() {
        let (w, _) = vp_unit(&dp, p);
        if v > 2 * w && v - w >= k {
            return true; // Hensel root of f inside the disc: f attains 0
        }
    }
    let stab = if *p == Integer::from(2) { v + 3 } else { v + 1 };
    if k >= stab {
        return false; // square-class constant (nonsquare) on the whole disc
    }
    let pk = p.pow(k);
    let mut t = Integer::zero();
    while &t < p {
        let r2 = r.clone() + &t * &pk;
        if soluble_disc(f, fp, p, &r2, k + 1, depth + 1) {
            return true;
        }
        t = t + Integer::one();
    }
    false
}

/// Status of a single polynomial on a residue disc, for the pair recursion.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DiscStatus {
    /// Attains 0 inside the disc (certified via Hensel or exactly at r).
    Root,
    /// Square in Q_p for every x in the disc.
    All,
    /// Nonsquare for every x in the disc.
    Never,
    /// Not yet determined at this radius.
    Undecided,
}

fn disc_status(d: &Integer, dpoly: &IntPoly, p: &Integer, r: &Integer, k: u32) -> DiscStatus {
    if d.is_zero() {
        return DiscStatus::Root;
    }
    let (v, u) = vp_unit(d, p);
    let stab = if *p == Integer::from(2) { v + 3 } else { v + 1 };
    if k >= stab {
        return if v % 2 == 0 && unit_is_square(&u, p) {
            DiscStatus::All
        } else {
            DiscStatus::Never
        };
    }
    let dp = dpoly.eval(r);
    if !dp.is_zero() {
        let (w, _) = vp_unit(&dp, p);
        if v > 2 * w && v - w >= k {
            return DiscStatus::Root;
        }
    }
    DiscStatus::Undecided
}

/// Does there exist x ∈ Z_p, x ≡ r (mod p^k), with f(x) AND g(x) both
/// squares in Q_p (including 0)?
#[allow(clippy::too_many_arguments)]
fn soluble_disc_pair(
    f: &IntPoly,
    fp: &IntPoly,
    g: &IntPoly,
    gp: &IntPoly,
    p: &Integer,
    r: &Integer,
    k: u32,
    depth: u32,
) -> bool {
    assert!(
        depth < MAX_DISC_DEPTH,
        "soluble_disc_pair: recursion depth exceeded (degenerate pair?): bug"
    );
    let df = f.eval(r);
    let dg = g.eval(r);
    let sf = disc_status(&df, fp, p, r, k);
    let sg = disc_status(&dg, gp, p, r, k);
    use DiscStatus::*;
    if sf == Never || sg == Never {
        return false;
    }
    if (sf == All && (sg == All || sg == Root)) || (sf == Root && sg == All) {
        return true;
    }
    // exact center test: both squares at x = r itself
    let okf = df.is_zero() || is_qp_square(&df, p);
    let okg = dg.is_zero() || is_qp_square(&dg, p);
    if okf && okg {
        return true;
    }
    let pk = p.pow(k);
    let mut t = Integer::zero();
    while &t < p {
        let r2 = r.clone() + &t * &pk;
        if soluble_disc_pair(f, fp, g, gp, p, &r2, k + 1, depth + 1) {
            return true;
        }
        t = t + Integer::one();
    }
    false
}

// ---------------------------------------------------------------------------
// square-class arithmetic in Q*/(Q*)²
// ---------------------------------------------------------------------------

/// The square-free part of n (sign preserved); n ≠ 0.
pub(crate) fn squarefree_part(n: &Integer) -> Integer {
    assert!(!n.is_zero());
    let mut s = Integer::one();
    for (q, e) in factor(&n.abs()) {
        if e % 2 == 1 {
            s = s * q;
        }
    }
    if n.signum() < 0 {
        -s
    } else {
        s
    }
}

/// All square-free divisors of n, positive and negative (candidate classes
/// for the descent images).
pub(crate) fn sf_divisors(n: &Integer) -> Vec<Integer> {
    assert!(!n.is_zero());
    let primes: Vec<Integer> = factor(&n.abs()).into_iter().map(|(q, _)| q).collect();
    let mut pos = vec![Integer::one()];
    for q in &primes {
        let mut next = pos.clone();
        for d in &pos {
            next.push(d.clone() * q.clone());
        }
        pos = next;
    }
    let mut all = Vec::with_capacity(pos.len() * 2);
    for d in pos {
        all.push(d.clone());
        all.push(-d);
    }
    all.sort_by(class_order);
    all
}

/// Deterministic ordering of square-free classes: by |·|, positives first.
fn class_order(x: &Integer, y: &Integer) -> std::cmp::Ordering {
    (x.abs(), x.signum() < 0).cmp(&(y.abs(), y.signum() < 0))
}

fn class_mul(x: &Integer, y: &Integer) -> Integer {
    squarefree_part(&(x * y))
}

/// Subgroup of Q*/(Q*)² generated by the given classes.
fn group_closure_1(gens: &[Integer]) -> Vec<Integer> {
    let mut g: Vec<Integer> = vec![Integer::one()];
    let mut changed = true;
    while changed {
        changed = false;
        let cur = g.clone();
        for x in &cur {
            for y in gens.iter().chain(cur.iter()) {
                let z = class_mul(x, y);
                if !g.contains(&z) {
                    g.push(z);
                    changed = true;
                }
            }
        }
    }
    g.sort_by(class_order);
    g
}

fn group_closed_1(s: &[Integer]) -> bool {
    s.iter()
        .all(|x| s.iter().all(|y| s.contains(&class_mul(x, y))))
}

fn class_mul_2(x: &(Integer, Integer), y: &(Integer, Integer)) -> (Integer, Integer) {
    (class_mul(&x.0, &y.0), class_mul(&x.1, &y.1))
}

fn group_closure_2(gens: &[(Integer, Integer)]) -> Vec<(Integer, Integer)> {
    let mut g = vec![(Integer::one(), Integer::one())];
    let mut changed = true;
    while changed {
        changed = false;
        let cur = g.clone();
        for x in &cur {
            for y in gens.iter().chain(cur.iter()) {
                let z = class_mul_2(x, y);
                if !g.contains(&z) {
                    g.push(z);
                    changed = true;
                }
            }
        }
    }
    g
}

fn group_closed_2(s: &[(Integer, Integer)]) -> bool {
    s.iter()
        .all(|x| s.iter().all(|y| s.contains(&class_mul_2(x, y))))
}

/// log₂ of a power of two (asserted).
pub(crate) fn dim2(n: usize) -> u32 {
    assert!(
        n > 0 && n & (n - 1) == 0,
        "dim2: class-set size {} is not a power of two: bug",
        n
    );
    n.trailing_zeros()
}

/// The odd primes dividing n, plus 2 if 2 | n (n is always even here since
/// we pass 2·… — kept general anyway).
fn bad_primes(n: &Integer) -> Vec<Integer> {
    factor(&n.abs()).into_iter().map(|(q, _)| q).collect()
}

fn exact_div(n: &Integer, d: &Integer) -> Integer {
    let q = n / d;
    assert!(&(&q * d) == n, "exact_div: {} does not divide {}", d, n);
    q
}

fn gcd_u64(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd_u64(b, a % b)
    }
}

/// Exact square root of a nonnegative rational, if it is a square.
pub(crate) fn rational_sqrt(q: &Rational) -> Option<Rational> {
    if q < &Rational::zero() {
        return None;
    }
    let n = q.numerator();
    let d = q.denominator();
    if n.is_perfect_square() && d.is_perfect_square() {
        let sn = n.sqrt().expect("nonnegative");
        let sd = d.sqrt().expect("positive");
        Some(Rational::new(sn, sd).expect("nonzero denominator"))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(a1: i64, a2: i64, a3: i64, a4: i64, a6: i64) -> EllipticCurve {
        EllipticCurve::new(
            Integer::from(a1),
            Integer::from(a2),
            Integer::from(a3),
            Integer::from(a4),
            Integer::from(a6),
        )
    }

    fn bounds(e: &EllipticCurve) -> RankBounds {
        match e.rank_bounds() {
            RankBoundResult::Bounds(b) => b,
            RankBoundResult::Unresolved { reason } => {
                panic!("expected bounds for {}, got Unresolved: {}", e, reason)
            }
        }
    }

    fn has_point(pts: &[Point], x: i64, y: i64) -> bool {
        pts.iter()
            .any(|p| !p.infinity && p.x == Rational::from_i64(x) && p.y == Rational::from_i64(y))
    }

    fn assert_hhat_between(e: &EllipticCurve, p: &Point, lo: f64, hi: f64) {
        assert!(e.is_on_curve(p));
        assert_eq!(e.point_order(p), None, "point unexpectedly of finite order");
        let h = e.canonical_height(p, 128).to_f64();
        assert!(
            h > lo && h < hi,
            "hhat = {} out of the Python-verified window ({}, {})",
            h,
            lo,
            hi
        );
    }

    // Every interval, point and height window asserted below was derived
    // and verified with the independent Python reference implementation
    // (exact descent + brute-force local-solvability oracle + exact
    // duplication-limit heights) before being written down here.

    #[test]
    fn gate_x3_minus_x_rank_0() {
        // y² = x³ − x: full 2-torsion; three isogeny descents + full
        // 2-descent all collapse to [0, 0].
        let b = bounds(&curve(0, 0, 0, -1, 0));
        assert_eq!((b.lower, b.upper), (0, 0));
        assert_eq!(b.unresolved_classes, 0);
        assert!(b.infinite_order_points(&curve(0, 0, 0, -1, 0)).is_empty());
    }

    #[test]
    fn gate_x3_minus_25x_rank_1() {
        // y² = x³ − 25x (congruent number n = 5): rank exactly 1, witnessed
        // by (−4, ±6) among others; hhat(−4, 6) ≈ 1.89948.
        let e = curve(0, 0, 0, -25, 0);
        let b = bounds(&e);
        assert_eq!((b.lower, b.upper), (1, 1));
        assert_eq!(b.unresolved_classes, 0);
        assert!(has_point(&b.points, -4, 6) || has_point(&b.points, -4, -6));
        let p = Point::from_integers(-4, 6);
        assert_hhat_between(&e, &p, 1.5, 2.3);
        assert!(!b.infinite_order_points(&e).is_empty());
    }

    #[test]
    fn gate_x3_minus_4x_rank_0() {
        let b = bounds(&curve(0, 0, 0, -4, 0));
        assert_eq!((b.lower, b.upper), (0, 0));
        assert_eq!(b.unresolved_classes, 0);
    }

    #[test]
    fn gate_x3_plus_x_rank_0() {
        let b = bounds(&curve(0, 0, 0, 1, 0));
        assert_eq!((b.lower, b.upper), (0, 0));
        assert_eq!(b.unresolved_classes, 0);
    }

    #[test]
    fn gate_x3_minus_36x_rank_1() {
        // y² = x³ − 36x (congruent number n = 6): rank exactly 1;
        // hhat(−3, 9) ≈ 0.88862 (equal for (18, 72) and (−2, 8), which
        // differ from ±(−3, 9) by 2-torsion).
        let e = curve(0, 0, 0, -36, 0);
        let b = bounds(&e);
        assert_eq!((b.lower, b.upper), (1, 1));
        assert_eq!(b.unresolved_classes, 0);
        assert!(has_point(&b.points, -3, 9) || has_point(&b.points, -3, -9));
        assert!(has_point(&b.points, 18, 72) || has_point(&b.points, 18, -72));
        let p = Point::from_integers(-3, 9);
        assert_hhat_between(&e, &p, 0.5, 1.3);
    }

    #[test]
    fn rank_2_collapses_with_independent_points() {
        // y² = x³ − 6x² − 60x: 2-isogeny descent alone collapses to [2, 2].
        // (16, 40) and (12, 12) are independent: hhat ≈ 2.22090 / 1.42512,
        // regulator ≈ 3.00018 (verified twice in Python via the exact
        // duplication limit).
        let e = curve(0, -6, 0, -60, 0);
        let b = bounds(&e);
        assert_eq!((b.lower, b.upper), (2, 2));
        assert_eq!(b.unresolved_classes, 0);
        assert!(has_point(&b.points, 16, 40) || has_point(&b.points, 16, -40));
        assert!(has_point(&b.points, 12, 12) || has_point(&b.points, 12, -12));
        let p = Point::from_integers(16, 40);
        let q = Point::from_integers(12, 12);
        assert_hhat_between(&e, &p, 1.9, 2.6);
        assert_hhat_between(&e, &q, 1.1, 1.8);
        // Stage-1 integration: nonzero regulator certifies independence.
        let reg = e.regulator(&[p, q], 128).to_f64();
        assert!(
            reg > 2.9 && reg < 3.1,
            "regulator {} outside the Python-verified window (2.9, 3.1)",
            reg
        );
    }

    #[test]
    fn honest_unresolved_interval_n17() {
        // y² = x³ − 17²x: the classical non-congruent number 17 with
        // Sha(E)[2] ≅ (Z/2)². Everywhere-locally-solvable torsors with no
        // rational points keep the interval honestly open at [0, 2]; it is
        // NEVER collapsed by fiat.
        let e = curve(0, 0, 0, -289, 0);
        let b = bounds(&e);
        assert_eq!((b.lower, b.upper), (0, 2));
        assert!(b.unresolved_classes > 0);
        // no infinite-order point exists among the found (torsion) points
        assert!(b.infinite_order_points(&e).is_empty());
    }

    #[test]
    fn full_two_descent_beats_isogeny_descent_n42() {
        // y² = x³ − 42²x: descent via 2-isogeny from the (0,0) torsion
        // point alone leaves [0, 2] (unresolved classes), but the combined
        // machinery (isogeny descent from the other 2-torsion points and
        // full 2-descent, both of which collapse) certifies rank 0.
        let b = bounds(&curve(0, 0, 0, -1764, 0));
        assert_eq!((b.lower, b.upper), (0, 0));
        assert_eq!(b.unresolved_classes, 0);
    }

    #[test]
    fn general_model_14a1_rank_0() {
        // 14a1: y² + xy + y = x³ + 4x − 6 (torsion Z/6): exercises the
        // minimal → (36,108)-short path; single 2-torsion root.
        let b = bounds(&curve(1, 0, 1, 4, -6));
        assert_eq!((b.lower, b.upper), (0, 0));
        assert_eq!(b.unresolved_classes, 0);
    }

    #[test]
    fn general_model_15a1_rank_0() {
        // 15a1: y² + xy + y = x³ + x² − 10x − 10: full 2-torsion on the
        // scaled model (three roots) + full 2-descent; still rank 0.
        let b = bounds(&curve(1, 1, 1, -10, -10));
        assert_eq!((b.lower, b.upper), (0, 0));
        assert_eq!(b.unresolved_classes, 0);
    }

    #[test]
    fn general_model_65a1_rank_1() {
        // 65a1: y² + xy = x³ − x, rank 1: general model through the scaled
        // path, with a genuine infinite-order witness point mapped back to
        // the input model.
        let e = curve(1, 0, 0, -1, 0);
        let b = bounds(&e);
        assert_eq!((b.lower, b.upper), (1, 1));
        assert_eq!(b.unresolved_classes, 0);
        let inf = b.infinite_order_points(&e);
        assert!(!inf.is_empty());
        for p in &inf {
            let h = e.canonical_height(p, 128).to_f64();
            assert!(
                h > 1e-3,
                "infinite-order witness with implausibly small height {}",
                h
            );
        }
    }

    #[test]
    fn x3_plus_1_rank_0() {
        // y² = x³ + 1 (torsion Z/6): single 2-torsion root at x = −1 on the
        // unscaled path.
        let b = bounds(&curve(0, 0, 0, 0, 1));
        assert_eq!((b.lower, b.upper), (0, 0));
        assert_eq!(b.unresolved_classes, 0);
    }

    #[test]
    fn no_rational_two_torsion_is_unresolved() {
        // 11a1, 37a1, 389a1: irreducible 2-division polynomial → honest
        // Unresolved (number-field descent is out of scope), never a guess.
        for e in [
            curve(0, -1, 1, -10, -20),
            curve(0, 0, 1, -1, 0),
            curve(0, 1, 1, -2, 0),
        ] {
            match e.rank_bounds() {
                RankBoundResult::Unresolved { reason } => {
                    assert!(reason.contains("2-division polynomial"));
                    assert!(reason.contains("number field"));
                }
                RankBoundResult::Bounds(b) => {
                    panic!(
                        "expected Unresolved for {}, got [{}, {}]",
                        e, b.lower, b.upper
                    )
                }
            }
        }
    }

    // ---- unit tests of the exact primitives -----------------------------

    #[test]
    fn qp_soluble_quartic_spot_table() {
        // Cross-checked against the Python reference (which was itself
        // cross-checked against a brute-force p-adic oracle).
        let cases: [(i64, i64, i64, i64, bool); 12] = [
            (-1, 0, 25, 2, true),
            (-1, 0, 25, 5, true),
            (17, 0, -17, 2, true),
            (17, 0, -17, 17, true),
            (2, 0, -882, 3, true),
            (2, 0, -882, 7, true),
            (-2, 0, 882, 2, true),
            (3, 0, -108, 2, true),
            (3, 0, -108, 3, false),
            (-1, 6, -1, 2, true),
            (1, 0, 1, 2, true),
            (2, 6, 2, 3, true),
        ];
        for (b1, a, b2, p, expected) in cases {
            assert_eq!(
                qp_soluble_quartic(
                    &Integer::from(b1),
                    &Integer::from(a),
                    &Integer::from(b2),
                    &Integer::from(p)
                ),
                expected,
                "qp_soluble_quartic({}, {}, {}, {})",
                b1,
                a,
                b2,
                p
            );
        }
    }

    #[test]
    fn real_soluble_quartic_spot_table() {
        let cases: [(i64, i64, i64, bool); 5] = [
            (-1, 0, 25, true),
            (-1, 0, -25, false),
            (-1, -6, -1, false),
            (-1, 6, -1, true),
            (-2, 3, -1, true),
        ];
        for (b1, a, b2, expected) in cases {
            assert_eq!(
                real_soluble_quartic(&Integer::from(b1), &Integer::from(a), &Integer::from(b2)),
                expected,
                "real_soluble_quartic({}, {}, {})",
                b1,
                a,
                b2
            );
        }
    }

    #[test]
    fn square_class_arithmetic() {
        assert_eq!(squarefree_part(&Integer::from(-144)), Integer::from(-1));
        assert_eq!(squarefree_part(&Integer::from(72)), Integer::from(2));
        assert_eq!(squarefree_part(&Integer::from(1)), Integer::from(1));
        let d = sf_divisors(&Integer::from(-12));
        let expect: Vec<Integer> = [1, -1, 2, -2, 3, -3, 6, -6]
            .iter()
            .map(|&k| Integer::from(k))
            .collect();
        assert_eq!(d, expect);
        // closure of {2, 3} is {1, 2, 3, 6}
        let g = group_closure_1(&[Integer::from(2), Integer::from(3)]);
        assert_eq!(g.len(), 4);
        assert!(g.contains(&Integer::from(6)));
        assert!(group_closed_1(&g));
        assert_eq!(dim2(g.len()), 2);
    }

    #[test]
    fn larger_search_bound_only_helps() {
        // For an already-collapsed case a larger bound must not change the
        // certified interval.
        let e = curve(0, 0, 0, -36, 0);
        match e.rank_bounds_with_search(60) {
            RankBoundResult::Bounds(b) => assert_eq!((b.lower, b.upper), (1, 1)),
            _ => panic!("expected bounds"),
        }
    }
}
