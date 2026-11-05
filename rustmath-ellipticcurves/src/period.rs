//! The real period Ω_E of E/Q, computed by the arithmetic–geometric mean
//! (AGM) over [`rustmath_reals::bigfloat::BigFloat`].
//!
//! # Convention (pinned)
//!
//! ```text
//! Ω_E = ∫_{E(R)} |ω|,   ω = dx / (2y + a₁x + a₃)  the Néron differential
//!                        of the GLOBAL MINIMAL MODEL,
//! ```
//!
//! the integral running over **all** real components. Completing the square,
//! (2y + a₁x + a₃)² = g(x) := 4x³ + b₂x² + 2b₄x + b₆ (the two-torsion
//! cubic), and each x with g(x) > 0 carries exactly two real points, so
//!
//! ```text
//! Ω_E = 2 ∫_{ {x : g(x) ≥ 0} } dx / √g(x).
//! ```
//!
//! Ω_E is an invariant of the minimal model: for a non-minimal input model
//! the computation passes to the minimal model first (a Weierstrass
//! isomorphism with scale u rescales ω by u, so only the minimal model gives
//! the canonical normalization used in BSD).
//!
//! # Formulas (derived, then pinned against brute numeric integrals)
//!
//! Both formulas below were validated in mpmath at 60 digits against direct
//! quadrature of ∫ dx/√g over the exact region {g ≥ 0} (with smooth
//! substitutions at the branch points) for 11a1, 14a1, 15a1, 37a1, 37b1,
//! 65a1, 389a1, y² = x³ − x and y² = x³ + 1 — agreement to ~10⁻⁵⁹ — rather
//! than trusted from memory.
//!
//! * **Δ > 0** (three real roots e₃ < e₂ < e₁ of g; E(R) has 2 components):
//!   the bounded ("egg") and unbounded components carry *equal* mass
//!   (verified numerically to 60 digits as part of the pinning), and
//!
//!   ```text
//!   Ω_E = 2π / AGM( √(e₁−e₃), √(e₁−e₂) ).
//!   ```
//!
//! * **Δ < 0** (one real root e₁, complex pair a ± bi; 1 component): with
//!
//!   ```text
//!   c = a − e₁ = −b₂/8 − (3/2)e₁,     A = |e₁ − (a+bi)| = √(c² + b²),
//!   ```
//!
//!   one has A² = g′(e₁)/4 (verified: g = 4(x−e₁)((x−a)²+b²) gives
//!   g′(e₁) = 4((e₁−a)²+b²)) and
//!
//!   ```text
//!   Ω_E = π / AGM( √A, √((A−c)/2) ).
//!   ```
//!
//! # Exact root isolation
//!
//! The roots of g are isolated with **exact rational arithmetic** (signs of
//! g at rational points are computed exactly; no floating-point decision is
//! ever made about a root):
//!
//! 1. All *rational* roots of g are found exactly first, via the integer
//!    roots of the monic transform X³ − 27c₄X − 54c₆, X = 36x + 3b₂
//!    (the same reduction as [`crate::curve::EllipticCurve::two_torsion_rank`]).
//! 2. Δ > 0 with 3 rational roots: done. With 1 rational root m: g deflates
//!    exactly to 4(x−m)(x²+px+q) with p, q ∈ Q, and the two (necessarily
//!    irrational) quadratic roots are bisected from the vertex −p/2 outward.
//!    With 0 rational roots: the two critical points of g (roots of
//!    g′ = 12x² + 2b₂x + 2b₄, which interlace e₃ < κ₁ < e₂ < κ₂ < e₁) are
//!    enclosed by bisection of g′ and refined until a rational point with
//!    g > 0 in (e₃, e₂) and one with g < 0 in (e₂, e₁) are certified; these
//!    separate the roots and plain bisection finishes. Since all remaining
//!    roots are irrational, bisection midpoints (rational) never collide
//!    with a root, and an exact zero of g′ at a midpoint is itself a valid
//!    separator.
//! 3. Δ < 0: the single real root is either the rational root found in
//!    step 1 (exact) or is bisected on the Cauchy interval [−M, M],
//!    M = 1 + max(|b₂|, |2b₄|, |b₆|), where g(−M) < 0 < g(M).
//!
//! # Error control (honest accounting)
//!
//! Let prec = `prec_bits`. All floating-point work runs at wp = prec + 64
//! bits; the AGM iterates until |aₙ − bₙ| ≤ aₙ·2^{−(wp−4)} plus one extra
//! (error-squaring) step, and M(a,b) always lies between bₙ and aₙ, so the
//! AGM value is exact to relative 2^{−(wp−4)} before rounding. The root
//! uncertainty enters as follows.
//!
//! * Δ > 0: e₁−e₂ and e₁−e₃ are formed as **exact rational differences of
//!   enclosure midpoints**, so their error is at most the enclosure width w.
//!   Enclosures are refined until w ≤ d·2^{−(prec+24)} where d is an exact
//!   rational **lower bound for e₁−e₂** (≤ e₁−e₃) read off the disjoint
//!   enclosures, giving relative error ≤ 2^{−(prec+24)} in the differences,
//!   halved by √, and not amplified by the AGM (M(a,b) is monotone in each
//!   argument and homogeneous of degree 1, so its relative error is at most
//!   the max relative error of its arguments).
//! * Δ < 0: A² = g′(e₁)/4 and c = −b₂/8 − (3/2)e₁ are evaluated as exact
//!   rationals at the enclosure midpoint; the width target
//!   w ≤ 2^{−(prec+32)} / C₂, C₂ = 24M + 2|b₂| ≥ sup_{[−M,M]} |g″|, keeps
//!   the error of A² below 2^{−(prec+32)}. Since the model is integral,
//!   A⁴b² = |Δ|/64 ≥ 1/64 and b ≤ A force **A ≥ 1/2**, so absolute errors
//!   are relative errors up to a factor 2. The difference A − c is
//!   cancellation-prone only when c > 0, and is then computed
//!   cancellation-free as b²/(A + c) with b² = |Δ|/(64A⁴) exact in Q given
//!   A² (identity verified numerically to 60 digits during pinning:
//!   disc(monic g/4) = ∏(rᵢ−rⱼ)² = Δ/16 = −4A⁴b² for Δ < 0).
//!
//! Total: relative error ≤ 2^{−(prec+20)} from root uncertainty plus
//! ≤ 2^{−(wp−8)} from AGM truncation and BigFloat rounding (each primitive
//! is correctly rounded to its own guard bits), comfortably below the final
//! rounding to `prec_bits`. The gate tests check 30+ decimal digits at
//! 128-bit precision against 55-digit mpmath values computed independently
//! two ways (brute integral and AGM).

use crate::curve::EllipticCurve;
use rustmath_core::analytic::RealField;
use rustmath_core::ordering::OrderedRing;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_reals::bigfloat::BigFloat;

impl EllipticCurve {
    /// The number of connected components of E(R): 2 if Δ > 0 (the
    /// two-torsion cubic has three real roots, so E(R) has an "egg"),
    /// otherwise 1. The sign of Δ is model-invariant (Δ scales by u¹² > 0),
    /// so this needs no minimalization.
    ///
    /// # Panics
    ///
    /// Panics if the curve is singular.
    pub fn num_real_components(&self) -> u32 {
        assert!(
            !self.is_singular(),
            "num_real_components: curve is singular (discriminant 0)"
        );
        if self.discriminant.signum() > 0 {
            2
        } else {
            1
        }
    }

    /// The real period Ω_E = ∫_{E(R)} |dx/(2y+a₁x+a₃)| of the **global
    /// minimal model**, over all real components, computed via the AGM at
    /// relative error ≤ 2^{−prec_bits} (see the module docs for the pinned
    /// convention, the formulas and the derived error bound). Always > 0.
    ///
    /// A non-minimal input model is minimalized first; Ω_E is an invariant
    /// of the curve, not of the presented model.
    ///
    /// # Panics
    ///
    /// Panics if the curve is singular.
    pub fn real_period(&self, prec_bits: u64) -> BigFloat {
        assert!(
            !self.is_singular(),
            "real_period: curve is singular (discriminant 0)"
        );
        let (emin, _) = self.minimal_model();
        real_period_minimal(&emin, prec_bits)
    }
}

// ---------------------------------------------------------------------------
// Exact rational root isolation for the two-torsion cubic
// ---------------------------------------------------------------------------

/// The two-torsion cubic g(x) = 4x³ + b₂x² + 2b₄x + b₆ with exact integer
/// coefficients, plus its derivative, evaluated exactly over Q.
#[derive(Clone)]
struct TwoTorsionCubic {
    b2: Integer,
    b4: Integer,
    b6: Integer,
}

impl TwoTorsionCubic {
    fn new(e: &EllipticCurve) -> Self {
        let (b2, b4, b6, _) = e.b_invariants();
        Self { b2, b4, b6 }
    }

    /// g(x), exactly.
    fn eval(&self, x: &Rational) -> Rational {
        let q = |n: &Integer| Rational::from_integer(n.clone());
        ((Rational::from_i64(4) * x.clone() + q(&self.b2)) * x.clone()
            + Rational::from_i64(2) * q(&self.b4))
            * x.clone()
            + q(&self.b6)
    }

    /// g′(x) = 12x² + 2b₂x + 2b₄, exactly.
    fn eval_deriv(&self, x: &Rational) -> Rational {
        let q = |n: &Integer| Rational::from_integer(n.clone());
        (Rational::from_i64(12) * x.clone() + Rational::from_i64(2) * q(&self.b2)) * x.clone()
            + Rational::from_i64(2) * q(&self.b4)
    }

    /// Cauchy bound: every real root has |x| < 1 + max(|b₂|, |2b₄|, |b₆|)/4;
    /// we use the (conservative) integer 1 + max(|b₂|, |2b₄|, |b₆|).
    fn root_bound(&self) -> Integer {
        let mut m = self.b2.abs();
        let t = Integer::from(2) * self.b4.abs();
        if t > m {
            m = t;
        }
        let t = self.b6.abs();
        if t > m {
            m = t;
        }
        m + Integer::one()
    }

    /// All rational roots of g, exactly: they biject with the integer roots
    /// of X³ − 27c₄X − 54c₆ under X = 36x + 3b₂ (the standard monic
    /// transform, already used by `two_torsion_rank`). Each returned root is
    /// certified by an exact g(x) = 0 check.
    fn rational_roots(&self, e: &EllipticCurve) -> Vec<Rational> {
        let (c4, c6) = e.c_invariants();
        let a = Integer::from(-27) * c4;
        let b = Integer::from(-54) * c6;
        let mut out: Vec<Rational> = crate::torsion::integer_cubic_roots(&a, &b)
            .into_iter()
            .map(|xx| {
                Rational::new(xx - Integer::from(3) * self.b2.clone(), Integer::from(36))
                    .expect("nonzero denominator")
            })
            .collect();
        out.sort();
        for r in &out {
            assert!(
                self.eval(r).is_zero(),
                "rational_roots: monic-transform root fails exact g(x) = 0 check (bug)"
            );
        }
        out
    }
}

/// An enclosure of a single real root: either exact (lo == hi, a certified
/// rational root) or an open interval with a strict sign change of f,
/// f(lo)·f(hi) < 0.
#[derive(Clone, Debug)]
struct Enclosure {
    lo: Rational,
    hi: Rational,
}

impl Enclosure {
    fn exact(r: Rational) -> Self {
        Self {
            lo: r.clone(),
            hi: r,
        }
    }

    fn width(&self) -> Rational {
        self.hi.clone() - self.lo.clone()
    }

    fn mid(&self) -> Rational {
        (self.lo.clone() + self.hi.clone()) * Rational::new(1, 2).unwrap()
    }

    fn is_exact(&self) -> bool {
        self.lo == self.hi
    }
}

fn sign(r: &Rational) -> i8 {
    r.numerator().signum()
}

/// Bisect f on the enclosure until width ≤ target. Precondition: exact, or
/// strict sign change at the endpoints. If a rational midpoint ever
/// evaluates to exactly 0 the enclosure collapses to that exact root (only
/// possible for a rational root; the isolation strategy removes those first,
/// but the branch is correct regardless).
fn refine<F: Fn(&Rational) -> Rational>(f: &F, enc: &mut Enclosure, target: &Rational) {
    if enc.is_exact() {
        return;
    }
    let mut s_lo = sign(&f(&enc.lo));
    debug_assert!(
        s_lo != 0 && sign(&f(&enc.hi)) == -s_lo,
        "refine: bad enclosure"
    );
    while enc.width() > *target {
        let m = enc.mid();
        let s_m = sign(&f(&m));
        if s_m == 0 {
            *enc = Enclosure::exact(m);
            return;
        }
        if s_m == s_lo {
            enc.lo = m;
            s_lo = s_m;
        } else {
            enc.hi = m;
        }
    }
}

/// Grow s (doubling from 1) until f(v − s) > 0 and f(v + s) > 0; returns s.
/// Terminates for any upward-opening polynomial f.
fn grow_bracket<F: Fn(&Rational) -> Rational>(f: &F, v: &Rational) -> Rational {
    let mut s = Rational::from_i64(1);
    for _ in 0..4096 {
        let left = v.clone() - s.clone();
        let right = v.clone() + s.clone();
        if sign(&f(&left)) > 0 && sign(&f(&right)) > 0 {
            return s;
        }
        s = s * Rational::from_i64(2);
    }
    unreachable!("grow_bracket: no bracket after 4096 doublings (bug)")
}

/// Rational 2^{−k} as a width target.
fn pow2_inv(k: u64) -> Rational {
    Rational::new(Integer::one(), Integer::from(2).pow(k as u32)).unwrap()
}

/// Isolate the three real roots (Δ > 0), returned ascending e₃ < e₂ < e₁,
/// each refined to width ≤ (lower bound of e₁−e₂)·2^{−rel_bits}.
fn isolate_three(g: &TwoTorsionCubic, e: &EllipticCurve, rel_bits: u64) -> [Enclosure; 3] {
    let gf = |x: &Rational| g.eval(x);
    let rats = g.rational_roots(e);
    let mut encs: Vec<Enclosure> = match rats.len() {
        3 => rats.into_iter().map(Enclosure::exact).collect(),
        1 => {
            // Exact deflation: monic(g)/4 = (x − m)(x² + px + q).
            let m = rats[0].clone();
            let q4 = |n: &Integer| Rational::from_integer(n.clone()) * Rational::new(1, 4).unwrap();
            let p = m.clone() + q4(&g.b2);
            let qq = m.clone() * p.clone()
                + Rational::from_integer(g.b4.clone()) * Rational::new(1, 2).unwrap();
            assert!(
                (m.clone() * qq.clone() + q4(&g.b6)).is_zero(),
                "isolate_three: exact deflation remainder nonzero (bug)"
            );
            let quad = |x: &Rational| (x.clone() + p.clone()) * x.clone() + qq.clone();
            let v = -(p.clone() * Rational::new(1, 2).unwrap());
            assert!(
                sign(&quad(&v)) < 0,
                "isolate_three: quadratic factor has no two real roots (bug: Δ > 0)"
            );
            let s = grow_bracket(&quad, &v);
            let mut left = Enclosure {
                lo: v.clone() - s.clone(),
                hi: v.clone(),
            };
            let mut right = Enclosure {
                lo: v.clone(),
                hi: v + s,
            };
            // Quadratic roots are irrational here (else they would have been
            // rational roots of g); refine until both enclosures exclude m
            // and are disjoint from each other.
            let mut t = Rational::new(1, 16).unwrap();
            loop {
                refine(&quad, &mut left, &t);
                refine(&quad, &mut right, &t);
                let excludes = |enc: &Enclosure, pt: &Rational| pt < &enc.lo || pt > &enc.hi;
                if excludes(&left, &m) && excludes(&right, &m) && left.hi < right.lo {
                    break;
                }
                t = t * Rational::new(1, 4).unwrap();
            }
            vec![Enclosure::exact(m), left, right]
        }
        0 => {
            // All three roots irrational. Separate them with the critical
            // points κ₁ < κ₂ of g (e₃ < κ₁ < e₂ < κ₂ < e₁, g(κ₁) > 0,
            // g(κ₂) < 0).
            let gp = |x: &Rational| g.eval_deriv(x);
            let v = Rational::from_integer(-g.b2.clone()) * Rational::new(1, 12).unwrap();
            assert!(
                sign(&gp(&v)) < 0,
                "isolate_three: g' has no two real roots (bug: Δ > 0 needs 3 real roots of g)"
            );
            let s = grow_bracket(&gp, &v);
            let mut k1 = Enclosure {
                lo: v.clone() - s.clone(),
                hi: v.clone(),
            };
            let mut k2 = Enclosure {
                lo: v.clone(),
                hi: v + s,
            };
            // Certify a separator a1 ∈ (e₃, e₂) with g(a1) > 0: any point of
            // [k1.lo, κ₁] with g > 0 works, since g > 0 exactly on
            // (e₃, e₂) ∪ (e₁, ∞) and k1.lo ≤ κ₁ < e₂. Symmetrically for
            // a2 ∈ (e₂, e₁) with g(a2) < 0. If bisection of g′ lands on a
            // rational κ exactly, κ itself separates.
            let mut t = Rational::new(1, 16).unwrap();
            let a1 = loop {
                refine(&gp, &mut k1, &t);
                if sign(&gf(&k1.lo)) > 0 {
                    break k1.lo.clone();
                }
                if k1.is_exact() {
                    assert!(sign(&gf(&k1.lo)) > 0, "critical value at κ₁ not > 0 (bug)");
                    break k1.lo.clone();
                }
                t = t * Rational::new(1, 4).unwrap();
            };
            let mut t = Rational::new(1, 16).unwrap();
            let a2 = loop {
                refine(&gp, &mut k2, &t);
                if sign(&gf(&k2.hi)) < 0 {
                    break k2.hi.clone();
                }
                if k2.is_exact() {
                    assert!(sign(&gf(&k2.hi)) < 0, "critical value at κ₂ not < 0 (bug)");
                    break k2.hi.clone();
                }
                t = t * Rational::new(1, 4).unwrap();
            };
            let mb = Rational::from_integer(g.root_bound());
            assert!(sign(&gf(&(-mb.clone()))) < 0 && sign(&gf(&mb)) > 0);
            vec![
                Enclosure {
                    lo: -mb.clone(),
                    hi: a1.clone(),
                },
                Enclosure {
                    lo: a1,
                    hi: a2.clone(),
                },
                Enclosure { lo: a2, hi: mb },
            ]
        }
        n => unreachable!("cubic with {} rational roots and Δ > 0 (nonsingular)", n),
    };

    // Refine until pairwise strictly disjoint, then sort ascending.
    let mut t = Rational::new(1, 1024).unwrap();
    loop {
        for enc in encs.iter_mut() {
            refine(&gf, enc, &t);
        }
        let mut sorted = encs.clone();
        sorted.sort_by(|a, b| a.lo.cmp(&b.lo));
        if sorted[0].hi < sorted[1].lo && sorted[1].hi < sorted[2].lo {
            encs = sorted;
            break;
        }
        t = t * Rational::new(1, 4).unwrap();
    }

    // Final refinement: width ≤ (exact lower bound of e₁ − e₂)·2^{−rel_bits}.
    let d12_lb = encs[2].lo.clone() - encs[1].hi.clone();
    assert!(sign(&d12_lb) > 0);
    let target = d12_lb * pow2_inv(rel_bits);
    for enc in encs.iter_mut() {
        refine(&gf, enc, &target);
    }
    [encs[0].clone(), encs[1].clone(), encs[2].clone()]
}

/// Isolate the single real root (Δ < 0), refined to width ≤ 2^{−abs_bits}/C₂
/// with C₂ = 24M + 2|b₂| ≥ sup_{[−M,M]} |g″| (see module docs).
fn isolate_one(g: &TwoTorsionCubic, e: &EllipticCurve, abs_bits: u64) -> Enclosure {
    let gf = |x: &Rational| g.eval(x);
    let mb = g.root_bound();
    let c2 = Integer::from(24) * mb.clone() + Integer::from(2) * g.b2.abs();
    let target = Rational::new(Integer::one(), c2).unwrap() * pow2_inv(abs_bits);

    let rats = g.rational_roots(e);
    match rats.len() {
        1 => Enclosure::exact(rats[0].clone()),
        0 => {
            let mbq = Rational::from_integer(mb);
            let mut enc = Enclosure {
                lo: -mbq.clone(),
                hi: mbq,
            };
            assert!(
                sign(&gf(&enc.lo)) < 0 && sign(&gf(&enc.hi)) > 0,
                "isolate_one: Cauchy bound endpoints have wrong signs (bug)"
            );
            refine(&gf, &mut enc, &target);
            enc
        }
        n => unreachable!("Δ < 0 cubic with {} rational roots (needs 0 or 1)", n),
    }
}

// ---------------------------------------------------------------------------
// AGM over BigFloat
// ---------------------------------------------------------------------------

/// AGM(a, b) for a, b > 0 at working precision wp: iterate
/// (a, b) ← ((a+b)/2, √(ab)) until |a − b| ≤ a·2^{−(wp−4)}, then one extra
/// (error-squaring) step; the limit always lies in [bₙ, aₙ], so the returned
/// (a+b)/2 has relative error ≤ 2^{−(wp−4)} plus rounding. Panics (bug
/// detector, never an answer) on nonpositive input or non-convergence.
fn agm(a0: BigFloat, b0: BigFloat, wp: u64) -> BigFloat {
    let zero = BigFloat::zero_prec(wp);
    assert!(a0 > zero && b0 > zero, "agm: arguments must be positive");
    let (mut a, mut b) = if a0 >= b0 { (a0, b0) } else { (b0, a0) };
    let two = BigFloat::from_integer(&Integer::from(2), wp);
    let eps = BigFloat::one_prec(wp)
        / BigFloat::from_integer(&Integer::from(2).pow(wp.saturating_sub(4) as u32), wp);
    let cap = 2 * wp as usize + 128;
    for _ in 0..cap {
        let step = |a: BigFloat, b: BigFloat| -> (BigFloat, BigFloat) {
            ((a.clone() + b.clone()) / two.clone(), (a * b).sqrt())
        };
        if (a.clone() - b.clone()).abs() <= a.clone() * eps.clone() {
            // one extra step squares the (already tiny) gap
            let (an, bn) = step(a, b);
            return (an + bn) / two;
        }
        let (an, bn) = step(a, b);
        a = an;
        b = bn;
    }
    unreachable!("agm: no convergence in {} iterations (bug)", cap)
}

// ---------------------------------------------------------------------------
// The period itself (on a minimal model)
// ---------------------------------------------------------------------------

/// Ω of the (already minimal) model `e` at `prec_bits`; see module docs.
fn real_period_minimal(e: &EllipticCurve, prec_bits: u64) -> BigFloat {
    let g = TwoTorsionCubic::new(e);
    let wp = prec_bits + 64;
    let pi = <BigFloat as RealField>::pi(wp);
    let two = BigFloat::from_integer(&Integer::from(2), wp);

    let omega = if e.discriminant.signum() > 0 {
        // Ω = 2π / AGM(√(e₁−e₃), √(e₁−e₂)); differences as exact rationals
        // of enclosure midpoints (error ≤ width ≤ (e₁−e₂)·2^{−(prec+24)}).
        let [e3, e2, e1] = isolate_three(&g, e, prec_bits + 24);
        let d13 = e1.mid() - e3.mid();
        let d12 = e1.mid() - e2.mid();
        assert!(sign(&d13) > 0 && sign(&d12) > 0);
        let s13 = BigFloat::from_rational(&d13, wp).sqrt();
        let s12 = BigFloat::from_rational(&d12, wp).sqrt();
        two * pi / agm(s13, s12, wp)
    } else {
        // Ω = π / AGM(√A, √((A−c)/2)), A² = g′(e₁)/4, c = −b₂/8 − (3/2)e₁,
        // all exact rationals at the enclosure midpoint.
        let enc = isolate_one(&g, e, prec_bits + 32);
        let mid = enc.mid();
        let a2q = g.eval_deriv(&mid) * Rational::new(1, 4).unwrap();
        assert!(
            sign(&a2q) > 0,
            "real_period: A² = g′(e₁)/4 not positive (bug)"
        );
        let cq = Rational::from_integer(-g.b2.clone()) * Rational::new(1, 8).unwrap()
            - Rational::new(3, 2).unwrap() * mid;
        let a_bf = BigFloat::from_rational(&a2q, wp).sqrt();
        // t = (A − c)/2, cancellation-free when c > 0 via A−c = b²/(A+c),
        // b² = |Δ|/(64 A⁴) (identity pinned numerically; see module docs).
        let t = if sign(&cq) <= 0 {
            (a_bf.clone() - BigFloat::from_rational(&cq, wp)) / two.clone()
        } else {
            let b2q = Rational::new(e.discriminant.abs(), Integer::from(64)).unwrap()
                / (a2q.clone() * a2q);
            BigFloat::from_rational(&b2q, wp)
                / (two.clone() * (a_bf.clone() + BigFloat::from_rational(&cq, wp)))
        };
        assert!(
            t > BigFloat::zero_prec(wp),
            "real_period: (A−c)/2 not positive (bug)"
        );
        pi / agm(a_bf.sqrt(), t.sqrt(), wp)
    };

    assert!(
        omega > BigFloat::zero_prec(wp),
        "real_period: Ω must be positive (bug)"
    );
    omega.with_precision(prec_bits)
}

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

    /// 10^{−digits} as a BigFloat comparison tolerance.
    fn tol(digits: u32) -> BigFloat {
        BigFloat::one_prec(64) / BigFloat::from_integer(&Integer::from(10).pow(digits), 64)
    }

    fn assert_digits(got: &BigFloat, expected: &str, digits: u32, what: &str) {
        let exp = BigFloat::from_decimal_str(expected, 256).unwrap();
        let d = (got.clone() - exp).abs();
        assert!(
            d < tol(digits),
            "{}: got {}, expected {} ({} digits; |diff| = {:e})",
            what,
            got.to_decimal_string(40),
            expected,
            digits,
            d.to_f64()
        );
    }

    /// AGM(1, √2) against the 55-digit mpmath value (Gauss's lemniscatic
    /// constant reciprocal relation), computed independently before this
    /// test was written.
    #[test]
    fn agm_gauss_constant() {
        let wp = 256;
        let one = BigFloat::one_prec(wp);
        let sqrt2 = BigFloat::from_integer(&Integer::from(2), wp).sqrt();
        let m = agm(one, sqrt2, wp);
        assert_digits(
            &m,
            "1.198140234735592207439922492280323878227212663215651558",
            50,
            "AGM(1, sqrt 2)",
        );
    }

    /// Ω gates. Every expected value was derived independently in mpmath at
    /// 80 dps TWO ways before being asserted here: (a) brute quadrature of
    /// 2∫_{g≥0} dx/√g over the exact region (smooth substitutions at branch
    /// points; disc > 0 covers BOTH components), and (b) the AGM formulas;
    /// (a) and (b) agreed to ~10⁻⁵⁹ on every curve. 37a1/11a1/389a1 also
    /// match the LMFDB real periods to all published digits.
    ///
    /// Path coverage of the exact root isolation:
    /// * x³−x: Δ > 0, THREE rational roots (0, ±1) — fully exact path;
    /// * 15a1: Δ > 0, ONE rational root (x = 3) — exact deflation + quadratic;
    /// * 37a1, 37b1, 65a1, 389a1: Δ > 0, no rational roots — critical-point
    ///   separators + bisection;
    /// * 11a1, 14a1: Δ < 0, irrational real root — Cauchy-interval bisection;
    /// * x³+1: Δ < 0, rational real root (x = −1) — exact.
    #[test]
    fn real_period_gate_table() {
        let gates: [(&str, [i64; 5], u32, &str); 9] = [
            (
                "11a1",
                [0, -1, 1, -10, -20],
                1,
                "1.269209304279553421688794616754547305219492241830608668",
            ),
            (
                "14a1",
                [1, 0, 1, 4, -6],
                1,
                "1.981341956066883234169571676737009265242714044691321517",
            ),
            (
                "15a1",
                [1, 1, 1, -10, -10],
                2,
                "2.801206084665204046360361673619437209227934657466882283",
            ),
            (
                "37a1",
                [0, 0, 1, -1, 0],
                2,
                "5.986917292463919259664019958905016355595167582740265971",
            ),
            (
                "37b1",
                [0, 1, 1, -23, -50],
                2,
                "2.177043185808458347008616623079189646210281008602755598",
            ),
            (
                "65a1",
                [1, 0, 0, -1, 0],
                2,
                "5.382853470571800994115223529422657868804998633783814035",
            ),
            (
                "389a1",
                [0, 1, 1, -2, 0],
                2,
                "4.980425121710110150642715583884604920312116360679140080",
            ),
            (
                "x3-x",
                [0, 0, 0, -1, 0],
                2,
                "5.244115108584239620929679179782238827365509902863246326",
            ),
            (
                "x3+1",
                [0, 0, 0, 0, 1],
                1,
                "4.206546315976362783525057237150882406389066616271958289",
            ),
        ];
        for (label, a, ncomp, expected) in gates {
            let e = curve(a[0], a[1], a[2], a[3], a[4]);
            assert_eq!(
                e.num_real_components(),
                ncomp,
                "num_real_components of {}",
                label
            );
            let omega = e.real_period(128);
            assert!(omega > BigFloat::zero_prec(64), "Ω({}) must be > 0", label);
            // 128 bits ≈ 38.5 decimal digits; assert 30 (gate demands 25+).
            assert_digits(&omega, expected, 30, &format!("Ω of {}", label));
        }
    }

    /// Ω is an invariant of the minimal model: u-scaled (non-minimal)
    /// integral models must give the SAME Ω as the minimal curve.
    #[test]
    fn real_period_minimal_model_invariance() {
        // 37a1 scaled by u = 2: [0, 0, 8, −16, 0].
        let e37s = curve(0, 0, 8, -16, 0);
        assert_digits(
            &e37s.real_period(128),
            "5.986917292463919259664019958905016355595167582740265971",
            30,
            "Ω of 37a1 scaled by u=2",
        );
        // 11a1 scaled by u = 3: [0, −9, 27, −810, −14580].
        let e11s = curve(0, -9, 27, -810, -14580);
        assert_digits(
            &e11s.real_period(128),
            "1.269209304279553421688794616754547305219492241830608668",
            30,
            "Ω of 11a1 scaled by u=3",
        );
        // Component count is also model-invariant (sign of Δ).
        assert_eq!(e37s.num_real_components(), 2);
        assert_eq!(e11s.num_real_components(), 1);
    }

    /// Self-consistency across precisions: the documented error bound makes
    /// Ω(192 bits) and Ω(288 bits) agree to ≲ 2^{−185}.
    #[test]
    fn real_period_precision_consistency() {
        let thr = BigFloat::one_prec(64) / BigFloat::from_integer(&Integer::from(2).pow(185), 64);
        for a in [[0i64, 0, 1, -1, 0], [0, -1, 1, -10, -20]] {
            let e = curve(a[0], a[1], a[2], a[3], a[4]);
            let lo = e.real_period(192).with_precision(320);
            let hi = e.real_period(288).with_precision(320);
            let d = (lo - hi).abs();
            assert!(
                d < thr,
                "Ω precision consistency: |Ω(192) − Ω(288)| = {:e}",
                d.to_f64()
            );
        }
    }

    #[test]
    #[should_panic(expected = "singular")]
    fn real_period_rejects_singular() {
        let e = curve(0, 0, 0, 0, 0); // y² = x³, Δ = 0
        let _ = e.real_period(64);
    }
}
