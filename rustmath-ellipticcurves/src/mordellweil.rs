//! Subgroups of E(Q): certified independence, and p-saturation via
//! [`EllipticCurve::divide_point`].
//!
//! # What this module DOES prove
//!
//! * **Independence.** A set of points P_1 … P_r is independent in
//!   E(Q) ⊗ Q iff the Néron–Tate height-pairing Gram matrix ⟨P_i, P_j⟩ is
//!   nonsingular. [`EllipticCurve::regulator_checked`] decides that with an
//!   explicit interval: it returns the determinant *and* a rigorous error
//!   bound, and it refuses (honest `Err`) unless |det| > bound. It never
//!   compares a float to 0. Independence gives the unconditional lower bound
//!   rank E(Q) ≥ r.
//! * **p-saturation.** Λ = ⟨P_1 … P_r⟩ is p-saturated in E(Q)/tors iff no
//!   nonzero class (c_1 … c_r) ∈ (Z/p)^r has Σ c_i P_i ∈ p·E(Q) + E(Q)_tors —
//!   i.e. iff, for every such class and every torsion point T,
//!   Σ c_i P_i + T is **not** divisible by p in E(Q). Divisibility is decided
//!   EXACTLY by [`EllipticCurve::divide_point`] (rational roots of a
//!   degree-p² division polynomial, then a re-multiplication gate), so a
//!   completed run is a proof, not a search that gave up.
//!
//!   Divisibility is invariant under scaling the class by a unit of Z/p
//!   (λ·(pQ) = p·(λQ), and λT is again torsion), so it is enough to sweep the
//!   (p^r − 1)/(p − 1) *projective* classes — normalized so the first nonzero
//!   coefficient is 1, the rest reduced to (−p/2, p/2] to keep the heights
//!   small. When a class IS divisible, the pivot generator is replaced by the
//!   divisor Q; the new lattice contains the old with index exactly p, which
//!   the code re-checks by requiring Reg_new · p² = Reg_old inside the
//!   certified interval.
//!
//! # What this module does NOT prove — read this before believing a number
//!
//! `saturated_up_to: Some(B)` means, and only means: **p-saturated for every
//! prime p ≤ B**. It is NOT a proof that the generators are a basis of
//! E(Q)/tors. A basis proof additionally needs
//!
//! * a bound on the index [E(Q)/tors : Λ] — e.g. from the difference between
//!   the naive and canonical heights (Silverman / Cremona–Prickett–Siksek),
//!   which bounds the primes p that could possibly divide the index — and
//! * saturation at every prime up to that bound.
//!
//! Neither is implemented here. So a `MordellWeilSubgroup` is a *subgroup*,
//! its `provenance` says so, and every consumer (notably
//! [`EllipticCurve::analytic_sha_rank_r`]) must carry the assumption forward.

use crate::curve::{EllipticCurve, Point};
use crate::torsion::TorsionSubgroup;
use rustmath_core::ordering::OrderedRing;
use rustmath_integers::Integer;
use rustmath_reals::bigfloat::BigFloat;
use std::cmp::Ordering;

/// Fail-closed strict comparison (a partially-ordered type: incomparable is
/// NOT certified).
fn certified_less(a: &BigFloat, b: &BigFloat) -> bool {
    matches!(a.partial_cmp(b), Some(Ordering::Less))
}

fn pow2_neg(k: u64, wp: u64) -> BigFloat {
    BigFloat::one_prec(wp) / BigFloat::from_integer(&Integer::from(2).pow(k as u32), wp)
}

/// The regulator det⟨P_i, P_j⟩ **with a rigorous error bound**, so that
/// nonsingularity (= independence) is a certified statement and not a float
/// compared against zero.
#[derive(Debug, Clone)]
pub struct CertifiedRegulator {
    /// The numeric determinant of the height-pairing Gram matrix.
    pub value: BigFloat,
    /// CERTIFIED: |true regulator − `value`| < `error_bound`.
    pub error_bound: BigFloat,
    /// r = the number of points.
    pub rank: usize,
}

impl CertifiedRegulator {
    /// |value| > error_bound: the Gram matrix is CERTIFIED nonsingular, hence
    /// the points are independent. (The converse is never claimed — a value
    /// inside the bound means "undecided at this precision", which for a
    /// genuinely dependent set it always will be, since numerics can never
    /// certify a zero.)
    pub fn certified_nonzero(&self) -> bool {
        certified_less(&self.error_bound, &OrderedRing::abs(&self.value))
    }
}

impl EllipticCurve {
    /// The regulator of `points` together with a rigorous error bound; `Err`
    /// unless the Gram determinant is certified nonzero (i.e. unless the
    /// points are certified independent).
    ///
    /// Error model. Each canonical height carries absolute truncation error
    /// ≤ 2^{−wp} ([`crate::height`]) and each pairing combines three of them
    /// over 2, so every Gram entry is within ε := 2^{−(wp−2)} of the truth.
    /// Expanding det(A + E) multilinearly in the rows,
    /// |det(A+E) − det(A)| ≤ r!·[(M+ε)^r − M^r] with M = max|A_ij|; a ×2
    /// factor absorbs the O(2^{−wp}) rounding of the elimination itself
    /// (wp ≫ the size of ε·r!·M^{r−1} in every use here). To that is added the
    /// rounding of the returned value down to `prec_bits`, |det|·2^{−prec_bits+1},
    /// which is the DOMINANT term whenever prec_bits < wp — omitting it was a
    /// bound that the code's own saturation self-check immediately falsified.
    ///
    /// The empty set has regulator 1 with bound 0 (the rank-0 convention).
    pub fn regulator_checked(
        &self,
        points: &[Point],
        prec_bits: u64,
    ) -> Result<CertifiedRegulator, String> {
        let r = points.len();
        if r == 0 {
            return Ok(CertifiedRegulator {
                value: BigFloat::one_prec(prec_bits),
                error_bound: BigFloat::zero_prec(prec_bits),
                rank: 0,
            });
        }
        let wp = prec_bits + 32;
        for p in points {
            if !self.is_on_curve(p) {
                return Err("regulator_checked: a supplied point is not on the curve".to_string());
            }
            if self.point_order(p).is_some() {
                return Err(
                    "regulator_checked: a supplied point is torsion (its canonical height \
                     is 0, so the Gram matrix is singular by construction)"
                        .to_string(),
                );
            }
        }

        let mut g = vec![vec![BigFloat::zero_prec(wp); r]; r];
        let mut m = BigFloat::zero_prec(wp);
        for i in 0..r {
            for j in i..r {
                let v = if i == j {
                    self.canonical_height(&points[i], wp)
                } else {
                    self.height_pairing(&points[i], &points[j], wp)
                };
                let av = OrderedRing::abs(&v);
                if av > m {
                    m = av;
                }
                g[i][j] = v.clone();
                g[j][i] = v;
            }
        }
        let det = crate::height::det_bigfloat(g, wp);

        let eps = pow2_neg(wp - 2, wp);
        let rfact = {
            let mut f = Integer::one();
            for k in 2..=r {
                f = f * Integer::from(k as i64);
            }
            BigFloat::from_integer(&f, wp)
        };
        let pow = |b: &BigFloat, e: usize| {
            let mut acc = BigFloat::one_prec(wp);
            for _ in 0..e {
                acc = acc * b.clone();
            }
            acc
        };
        let spread = pow(&(m.clone() + eps.clone()), r) - pow(&m, r);
        let two = BigFloat::from_integer(&Integer::from(2), wp);
        // …plus the rounding of the ANSWER itself down to prec_bits, which for
        // prec_bits well below wp is the dominant term (|det|·2^{−prec_bits+1}).
        let repr = OrderedRing::abs(&det) * pow2_neg(prec_bits - 1, wp);
        let error_bound = two * rfact * spread + repr;

        let out = CertifiedRegulator {
            value: det.with_precision(prec_bits),
            error_bound: error_bound.with_precision(prec_bits),
            rank: r,
        };
        if !out.certified_nonzero() {
            return Err(format!(
                "regulator_checked: the height-pairing Gram determinant {} is NOT certified \
                 nonzero (error bound {}) — the {} points are dependent, or the precision is \
                 too low to tell. Numerics can never certify a zero, so this is a refusal, \
                 not a proof of dependence.",
                out.value.to_decimal_string(20),
                out.error_bound.to_decimal_string(6),
                r
            ));
        }
        Ok(out)
    }
}

/// A finite-index subgroup of E(Q) presented by independent generators of a
/// subgroup of E(Q)/tors, plus the full torsion.
///
/// **This is a subgroup, not (in general) the Mordell–Weil group.** See the
/// module docs: `saturated_up_to = Some(B)` only means "p-saturated for every
/// prime p ≤ B". No index bound is computed anywhere in this crate, so no
/// basis of E(Q) is ever claimed.
#[derive(Debug, Clone)]
pub struct MordellWeilSubgroup {
    /// Independent points of infinite order (certified independent at
    /// construction, and re-certified after every saturation step).
    pub generators: Vec<Point>,
    /// The exact torsion subgroup E(Q)_tors.
    pub torsion: TorsionSubgroup,
    /// `Some(B)`: p-saturated for EVERY prime p ≤ B. `None`: no saturation
    /// has been proved. Never means "this is a basis of E(Q)/tors".
    pub saturated_up_to: Option<u32>,
    /// Where the generators came from and exactly what has been proved.
    pub provenance: String,
}

/// One enlargement performed by the saturator.
#[derive(Debug, Clone)]
pub struct SaturationStep {
    /// The prime p at which the lattice was not saturated.
    pub prime: u32,
    /// The class (c_1 … c_r), pivot coefficient 1, rest in (−p/2, p/2].
    pub class: Vec<i64>,
    /// The torsion point T for which Σ c_i P_i + T was divisible by p.
    pub torsion_shift: Point,
    /// The generator that was replaced (the pivot index, where c_j = 1).
    pub replaced_index: usize,
    /// Q with [p]Q = Σ c_i P_i + T.
    pub new_point: Point,
    /// Regulator before and after; the ratio is certified to be p².
    pub regulator_before: BigFloat,
    pub regulator_after: BigFloat,
}

/// What a saturation run actually did — so a test can assert that it ran, and
/// what it found, rather than trusting a bare flag.
#[derive(Debug, Clone, Default)]
pub struct SaturationReport {
    /// Every prime swept.
    pub primes_tested: Vec<u32>,
    /// Every (class, torsion) pair for which divisibility was decided.
    pub classes_tested: usize,
    /// The enlargements found (empty ⇒ the input was already saturated).
    pub steps: Vec<SaturationStep>,
}

/// Cap on enlargements at one prime; each divides the regulator by p², so a
/// long run means a bug, not mathematics.
const MAX_ENLARGEMENTS: usize = 24;

/// Cap on the class sweep at one prime. (p^r − 1)/(p − 1) grows exponentially in
/// the rank, so an unguarded sweep is an unbounded allocation.
const MAX_CLASSES: usize = 200_000;

/// The projective classes of (Z/p)^r: first nonzero coefficient normalized to
/// 1, the remaining coefficients reduced to the symmetric range (−p/2, p/2]
/// so that Σ c_i P_i stays as small as possible. Returns (class, pivot index).
fn projective_classes(r: usize, p: u32) -> Result<Vec<(Vec<i64>, usize)>, String> {
    let pi = p as i64;
    // (p^r − 1)/(p − 1), computed without overflowing and without building anything.
    let mut count: usize = 0;
    for pivot in 0..r {
        let free = (r - pivot - 1) as u32;
        let block = (p as usize)
            .checked_pow(free)
            .filter(|b| count.checked_add(*b).is_some_and(|c| c <= MAX_CLASSES))
            .ok_or_else(|| {
                format!(
                    "saturate: the projective class sweep at p = {p} for rank {r} exceeds \
                     {MAX_CLASSES} classes — refusing (guard)"
                )
            })?;
        count += block;
    }
    let mut out = Vec::with_capacity(count);
    for pivot in 0..r {
        // c_pivot = 1, c_i = 0 for i < pivot, c_i free for i > pivot
        let free = r - pivot - 1;
        let total = pi.pow(free as u32);
        for mask in 0..total {
            let mut class = vec![0i64; r];
            class[pivot] = 1;
            let mut m = mask;
            for c in class.iter_mut().skip(pivot + 1) {
                let mut v = m % pi;
                m /= pi;
                if v > pi / 2 {
                    v -= pi;
                }
                *c = v;
            }
            out.push((class, pivot));
        }
    }
    Ok(out)
}

impl MordellWeilSubgroup {
    /// Build from points that must be on the curve, of infinite order, and
    /// CERTIFIED independent (nonsingular Gram, decided with the interval
    /// model in [`EllipticCurve::regulator_checked`]).
    pub fn new(
        curve: &EllipticCurve,
        generators: &[Point],
        prec_bits: u64,
        provenance: &str,
    ) -> Result<Self, String> {
        assert!(
            !provenance.trim().is_empty(),
            "MordellWeilSubgroup::new: the generators need a stated provenance"
        );
        let reg = curve.regulator_checked(generators, prec_bits)?;
        Ok(MordellWeilSubgroup {
            generators: generators.to_vec(),
            torsion: curve.torsion_subgroup(),
            saturated_up_to: None,
            provenance: format!(
                "{} independent point(s) of infinite order (CERTIFIED: Gram determinant {} > \
                 its error bound {}); source: {}. NO saturation proved yet; this is a \
                 SUBGROUP of E(Q), not a proved Mordell-Weil basis.",
                generators.len(),
                reg.value.to_decimal_string(20),
                reg.error_bound.to_decimal_string(6),
                provenance
            ),
        })
    }

    /// The rank of the *subgroup* — an unconditional LOWER bound for rank E(Q).
    pub fn rank(&self) -> usize {
        self.generators.len()
    }

    /// The certified regulator of the current generators.
    pub fn regulator(
        &self,
        curve: &EllipticCurve,
        prec_bits: u64,
    ) -> Result<CertifiedRegulator, String> {
        curve.regulator_checked(&self.generators, prec_bits)
    }

    /// p-saturate the subgroup for every prime p ≤ `primes_up_to`, enlarging
    /// the generators whenever a class turns out to be divisible.
    ///
    /// On success `saturated_up_to` becomes `Some(primes_up_to)` — which means
    /// exactly "p-saturated for every prime p ≤ primes_up_to" and NOTHING
    /// about being a basis of E(Q)/tors (see the module docs).
    ///
    /// `Err` (never a silent pass) if any divisibility test cannot be decided,
    /// if the enlarged set fails to be independent, or if the regulator does
    /// not drop by exactly p² (which would be an internal inconsistency).
    pub fn saturate(
        &mut self,
        curve: &EllipticCurve,
        primes_up_to: u32,
        prec_bits: u64,
    ) -> Result<SaturationReport, String> {
        let mut report = SaturationReport::default();
        if self.generators.is_empty() {
            self.saturated_up_to = Some(primes_up_to);
            return Ok(report);
        }

        let mut torsion_pts = vec![Point::infinity()];
        for (t, _) in &self.torsion.points {
            torsion_pts.push(t.clone());
        }

        for p in 2..=primes_up_to {
            if !is_prime_small(p) {
                continue;
            }
            report.primes_tested.push(p);
            let mut enlargements = 0usize;
            'restart: loop {
                let r = self.generators.len();
                for (class, pivot) in projective_classes(r, p)? {
                    for t in &torsion_pts {
                        report.classes_tested += 1;
                        let target = combine(curve, &self.generators, &class, t);
                        if target.infinity {
                            return Err(format!(
                                "saturate: the class {:?} plus a torsion point is O — the \
                                 generators are NOT independent (this contradicts the \
                                 construction-time certificate: bug)",
                                class
                            ));
                        }
                        let divisors = curve.try_divide_point(&target, p)?;
                        let Some(q) = divisors.first().cloned() else {
                            continue;
                        };
                        enlargements += 1;
                        if enlargements > MAX_ENLARGEMENTS {
                            return Err(format!(
                                "saturate: more than {} enlargements at p = {} — each divides \
                                 the regulator by p², so this is a bug, not mathematics",
                                MAX_ENLARGEMENTS, p
                            ));
                        }
                        let before = curve.regulator_checked(&self.generators, prec_bits)?;
                        let mut next = self.generators.clone();
                        next[pivot] = q.clone();
                        let after = curve.regulator_checked(&next, prec_bits)?;

                        // Index exactly p: Reg_after · p² must equal Reg_before.
                        let wp = prec_bits + 32;
                        let p2 =
                            BigFloat::from_integer(&Integer::from((p as i64) * (p as i64)), wp);
                        let lhs = after.value.clone().with_precision(wp) * p2.clone();
                        let diff =
                            OrderedRing::abs(&(lhs - before.value.clone().with_precision(wp)));
                        let tol = (after.error_bound.clone().with_precision(wp) * p2
                            + before.error_bound.clone().with_precision(wp))
                            * BigFloat::from_integer(&Integer::from(2), wp);
                        if !certified_less(&diff, &tol) {
                            return Err(format!(
                                "saturate: after replacing generator {} by a p-divisor at \
                                 p = {}, Reg went {} -> {}, but p²·Reg_new differs from \
                                 Reg_old by {} > tolerance {} — the index is not p (bug)",
                                pivot,
                                p,
                                before.value.to_decimal_string(20),
                                after.value.to_decimal_string(20),
                                diff.to_decimal_string(6),
                                tol.to_decimal_string(6)
                            ));
                        }

                        report.steps.push(SaturationStep {
                            prime: p,
                            class: class.clone(),
                            torsion_shift: t.clone(),
                            replaced_index: pivot,
                            new_point: q,
                            regulator_before: before.value,
                            regulator_after: after.value,
                        });
                        self.generators = next;
                        continue 'restart;
                    }
                }
                break;
            }
        }

        self.saturated_up_to = Some(primes_up_to);
        let reg = curve.regulator_checked(&self.generators, prec_bits)?;
        self.provenance = format!(
            "{} independent point(s) of infinite order (CERTIFIED: Gram determinant {} > its \
             error bound {}), p-SATURATED for every prime p <= {} ({} (class, torsion) \
             divisibility tests decided exactly via division polynomials; {} enlargement(s) \
             found). This does NOT prove a Mordell-Weil basis: no height-difference bound and \
             no index bound is computed anywhere in this crate, so primes above {} are \
             untested. {}",
            self.generators.len(),
            reg.value.to_decimal_string(20),
            reg.error_bound.to_decimal_string(6),
            primes_up_to,
            report.classes_tested,
            report.steps.len(),
            primes_up_to,
            self.provenance
        );
        Ok(report)
    }
}

/// Σ c_i P_i + T.
fn combine(curve: &EllipticCurve, gens: &[Point], class: &[i64], t: &Point) -> Point {
    let mut acc = t.clone();
    for (c, g) in class.iter().zip(gens) {
        if *c == 0 {
            continue;
        }
        acc = curve.add_points(&acc, &curve.scalar_mul(&Integer::from(*c), g));
    }
    acc
}

fn is_prime_small(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2u32;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descent::TwoDescent;
    use rustmath_core::analytic::RealField;

    fn curve(a: [i64; 5]) -> EllipticCurve {
        EllipticCurve::new(
            Integer::from(a[0]),
            Integer::from(a[1]),
            Integer::from(a[2]),
            Integer::from(a[3]),
            Integer::from(a[4]),
        )
    }

    fn close_to(a: &BigFloat, decimal: &str, k: usize) -> bool {
        let prec = RealField::precision(a).max(256);
        let b = BigFloat::from_decimal_str(decimal, prec).unwrap();
        let tol_str = format!("0.{}1", "0".repeat(k - 1));
        let tol = BigFloat::from_decimal_str(&tol_str, prec).unwrap();
        OrderedRing::abs(&(a.clone() - b)) < tol
    }

    /// THE INDEPENDENCE GATE, and it CATCHES A REAL DEPENDENCY.
    ///
    /// 5077a1 has rank 3, and (−1,3), (0,2), (2,0) all lie on it — but they are
    /// NOT independent: (−1,3) + (0,2) + (2,0) = O exactly (verified below by
    /// the group law, and independently in PARI: `elladd` of the three is [0]).
    /// The old `regulator()` would happily return a float near 0 for them;
    /// `regulator_checked` refuses. A true basis is (1,0), (2,0), (0,2)
    /// (PARI `ellgenerators`), whose regulator is 0.41714355875838397 —
    /// derived from PARI's `matdet(ellheightmatrix(...))` BEFORE this test.
    #[test]
    fn test_regulator_checked_catches_dependence_5077a() {
        let e = curve([0, 0, 1, -7, 6]);
        let dep = [
            Point::from_integers(-1, 3),
            Point::from_integers(0, 2),
            Point::from_integers(2, 0),
        ];
        for p in &dep {
            assert!(e.is_on_curve(p));
        }
        // the exact relation
        let s = e.add_points(&e.add_points(&dep[0], &dep[1]), &dep[2]);
        assert!(s.infinity, "(-1,3) + (0,2) + (2,0) = O");

        let r = e.regulator_checked(&dep, 128);
        assert!(
            r.is_err(),
            "a dependent triple must be refused, got {:?}",
            r.map(|c| c.value.to_decimal_string(12))
        );
        assert!(r.unwrap_err().contains("NOT certified nonzero"));

        let indep = [
            Point::from_integers(1, 0),
            Point::from_integers(2, 0),
            Point::from_integers(0, 2),
        ];
        let reg = e.regulator_checked(&indep, 128).expect("independent");
        assert!(reg.certified_nonzero());
        assert!(
            close_to(&reg.value, "0.4171435587583839698171195446180933967", 15),
            "Reg(5077a) = {}",
            reg.value.to_decimal_string(25)
        );
    }

    /// 389a1's regulator on the pair used everywhere below: 0.15246017794314375
    /// (PARI `matdet(ellheightmatrix(E,[[0,0],[-1,1]]))`, derived first).
    /// (0,0) and (−1,1) are NOT the pair PARI's `ellgenerators` returns
    /// ([[0,0],[1,0]]), but they span the same lattice — same regulator.
    #[test]
    fn test_regulator_checked_389a() {
        let e = curve([0, 1, 1, -2, 0]);
        let pts = [Point::from_integers(0, 0), Point::from_integers(-1, 1)];
        let reg = e.regulator_checked(&pts, 128).expect("independent");
        assert!(
            close_to(&reg.value, "0.15246017794314375162432475704945582", 15),
            "Reg(389a) = {}",
            reg.value.to_decimal_string(25)
        );
        // and a dependent pair (P, 2P) is refused
        let bad = [pts[0].clone(), e.double_point(&pts[0])];
        assert!(e.regulator_checked(&bad, 128).is_err());
    }

    /// SATURATION WITH A KNOWN NON-SATURATED INPUT (the demanded gate).
    /// 37a1 has rank 1 with generator P = (0,0) (PARI `ellgenerators`,
    /// ĥ(P) = 0.05111140823996884). Feed the saturator 2·P = (1,0) instead —
    /// an index-2 subgroup, ĥ(2P) = 4·ĥ(P) = 0.20444563295987537. The 2-class
    /// {2P} must be found divisible and the generator replaced by a point of
    /// height ĥ(P); the regulator drops by exactly 4.
    #[test]
    fn test_saturation_recovers_the_generator_37a() {
        let e = curve([0, 0, 1, -1, 0]);
        let p = Point::from_integers(0, 0);
        let two_p = e.double_point(&p);
        assert_eq!(two_p, Point::from_integers(1, 0));

        let mut mw = MordellWeilSubgroup::new(
            &e,
            std::slice::from_ref(&two_p),
            128,
            "deliberately un-saturated: 2·(0,0), an index-2 subgroup",
        )
        .expect("2P has infinite order");
        assert!(mw.saturated_up_to.is_none());
        let before = mw.regulator(&e, 128).unwrap().value;
        // PARI ellheight(E,ellmul(E,[0,0],2)) = 0.20444563295987536094…
        assert!(
            close_to(&before, "0.2044456329598753609435443990277680864", 16),
            "h(2P) = {}",
            before.to_decimal_string(20)
        );

        let rep = mw.saturate(&e, 5, 128).expect("saturation runs");
        assert_eq!(mw.saturated_up_to, Some(5));
        assert_eq!(rep.primes_tested, vec![2, 3, 5]);
        assert_eq!(rep.steps.len(), 1, "exactly one enlargement, at p = 2");
        assert_eq!(rep.steps[0].prime, 2);
        assert!(rep.steps[0].torsion_shift.infinity, "|T(37a)| = 1");

        let after = mw.regulator(&e, 128).unwrap().value;
        // PARI ellheight(E,[0,0]) = 0.05111140823996884023588609975694202161
        assert!(
            close_to(&after, "0.0511114082399688402358860997569420216", 16),
            "hhat(P) = {}",
            after.to_decimal_string(20)
        );
        // the recovered point really is ±P (trivial torsion, so nothing else)
        let g = &mw.generators[0];
        assert!(*g == p || *g == e.negate_point(&p), "recovered {:?}", g);
        assert!(mw
            .provenance
            .contains("does NOT prove a Mordell-Weil basis"));
    }

    /// The same one level deeper: feed 3·P = (−1, −1) (PARI
    /// `ellmul(ellinit([0,0,1,-1,0]),[0,0],3)` = [-1,-1]), an index-3
    /// subgroup. The saturator must fire at p = 3, not p = 2, and land back
    /// on ĥ(P).
    #[test]
    fn test_saturation_recovers_the_generator_37a_index_3() {
        let e = curve([0, 0, 1, -1, 0]);
        let p = Point::from_integers(0, 0);
        let three_p = e.scalar_mul(&Integer::from(3), &p);
        assert_eq!(three_p, Point::from_integers(-1, -1));
        let mut mw = MordellWeilSubgroup::new(
            &e,
            &[three_p],
            128,
            "deliberately un-saturated: 3·(0,0), an index-3 subgroup",
        )
        .unwrap();
        let rep = mw.saturate(&e, 5, 128).unwrap();
        assert_eq!(rep.steps.len(), 1);
        assert_eq!(rep.steps[0].prime, 3);
        let after = mw.regulator(&e, 128).unwrap().value;
        assert!(close_to(
            &after,
            "0.0511114082399688402358860997569420216",
            16
        ));
        let g = &mw.generators[0];
        assert!(*g == p || *g == e.negate_point(&p));
    }

    /// Saturation with NONTRIVIAL torsion, where the torsion sweep matters:
    /// 65a1 has |T| = 2 and generator (1,0). Feeding 2·(1,0) + T (T the
    /// 2-torsion point (0,0)) is still an index-2 subgroup of E(Q)/tors, and
    /// it is p=2-divisible only ONCE the torsion coset is swept — the class
    /// {c=1} with T ≠ O is what fires. Derived from the crate's own exact
    /// torsion, cross-checked in PARI (`elltors(ellinit([1,0,0,-1,0]))` = 2).
    #[test]
    fn test_saturation_sweeps_the_torsion_coset_65a() {
        let e = curve([1, 0, 0, -1, 0]);
        let t = e.torsion_subgroup();
        assert_eq!(t.order, 2);
        let tors = t.points[0].0.clone();
        let p = Point::from_integers(1, 0);
        let target = e.add_points(&e.double_point(&p), &tors);

        let mut mw = MordellWeilSubgroup::new(&e, &[target], 128, "2·(1,0) + T on 65a1").unwrap();
        let rep = mw.saturate(&e, 3, 128).unwrap();
        assert_eq!(rep.steps.len(), 1);
        assert_eq!(rep.steps[0].prime, 2);
        assert!(
            !rep.steps[0].torsion_shift.infinity,
            "the divisibility is only visible after adding T"
        );
        let after = mw.regulator(&e, 128).unwrap().value;
        // PARI ellheight(ellinit([1,0,0,-1,0]),[1,0]) = 0.375514098661266321804…
        assert!(
            close_to(&after, "0.3755140986612663218044728768245783177", 16),
            "got {}",
            after.to_decimal_string(20)
        );
    }

    /// 389a1: the generators are FOUND by naive search (descent.rs), not
    /// seeded, then certified independent and p-saturated for p ≤ 5. The
    /// saturation genuinely runs (3 + 4 + 6 = 13 projective classes over a
    /// trivial torsion group) and finds NOTHING — the pair is already
    /// 2-, 3- and 5-saturated, which is the expected answer (PARI's
    /// `ellgenerators` spans the same lattice: same regulator 0.15246017794…).
    #[test]
    fn test_389a_generators_found_and_saturated() {
        let e = curve([0, 1, 1, -2, 0]);
        let found = TwoDescent::new(&e).find_rational_points(4);
        let indep: Vec<Point> = {
            // pick the two smallest-height independent affine points found
            let mut affine: Vec<Point> = found
                .into_iter()
                .filter(|p| !p.infinity && e.point_order(p).is_none())
                .collect();
            affine.sort_by(|a, b| {
                EllipticCurve::naive_height(a, 64)
                    .partial_cmp(&EllipticCurve::naive_height(b, 64))
                    .unwrap()
            });
            let mut chosen: Vec<Point> = Vec::new();
            for p in affine {
                let mut trial = chosen.clone();
                trial.push(p.clone());
                if e.regulator_checked(&trial, 96).is_ok() {
                    chosen.push(p);
                }
                if chosen.len() == 2 {
                    break;
                }
            }
            chosen
        };
        assert_eq!(indep.len(), 2, "search must find 2 independent points");

        let mut mw = MordellWeilSubgroup::new(
            &e,
            &indep,
            128,
            "found by naive x = m/n search (descent::find_rational_points, bound 4)",
        )
        .unwrap();
        let rep = mw.saturate(&e, 5, 128).expect("saturation of 389a");
        assert_eq!(rep.primes_tested, vec![2, 3, 5]);
        assert_eq!(
            rep.classes_tested, 13,
            "3 + 4 + 6 projective classes, trivial torsion"
        );
        assert!(
            rep.steps.is_empty(),
            "389a's found pair is already saturated at 2, 3, 5; got {:?}",
            rep.steps.iter().map(|s| s.prime).collect::<Vec<_>>()
        );
        assert_eq!(mw.saturated_up_to, Some(5));

        let reg = mw.regulator(&e, 128).unwrap();
        assert!(
            close_to(&reg.value, "0.15246017794314375162432475704945582", 15),
            "Reg(389a) = {}",
            reg.value.to_decimal_string(25)
        );
    }
}
