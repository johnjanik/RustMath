//! Exact computation of the torsion subgroup E(Q)_tors.
//!
//! Algorithm (all arithmetic exact):
//!
//! 1. Replace E by its globally minimal reduced model E_min ([`crate::minimal`]).
//! 2. Bound |E(Q)_tors| by gcd of #Ē(F_p) over several odd primes of good
//!    reduction (reduction mod p is injective on torsion for odd good p).
//! 3. Find every torsion point by the (strong) Lutz–Nagell theorem applied
//!    to the integral short model E_s : Y² = X³ + AX + B obtained from E_min
//!    via (x, y) ↦ (36x + 3b₂, 108(2y + a₁x + a₃)), where A = −27c₄,
//!    B = −54c₆: any affine torsion point of E_s has X, Y ∈ Z and either
//!    Y = 0 or Y² | 4A³ + 27B². Note 4A³ + 27B² = −2⁸·3¹²·Δ_min (identity
//!    checked at runtime). Candidates are screened by Mazur's theorem
//!    (torsion point orders over Q lie in {1,…,10, 12}).
//! 4. Certify the result: the found set (plus O) must be closed under
//!    addition, its order must divide the reduction bound, and the group
//!    structure must be one of Mazur's fifteen groups.
//! 5. Map the points back to the *original* model through the inverse
//!    Weierstrass isomorphisms and re-verify each point and its order there.
//!
//! Every expected value asserted in the tests below was verified
//! independently with an exact brute Lutz–Nagell reference implementation in
//! Python (fractions + sympy) before being written down here.

use crate::curve::{EllipticCurve, Point};
use crate::minimal::WeierstrassIsomorphism;
use rustmath_integers::prime::factor;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;

/// The abstract group structure of E(Q)_tors, per Mazur's classification:
/// Z/nZ for n ∈ {1,…,10, 12} or Z/2Z × Z/2nZ for n ∈ {1, 2, 3, 4}.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorsionStructure {
    /// Cyclic of order n (n = 1 is the trivial group).
    Cyclic(u32),
    /// Z/2Z × Z/2nZ (order 4n), n ∈ {1, 2, 3, 4}.
    TwoByTwoN(u32),
}

impl TorsionStructure {
    /// The order of the group.
    pub fn order(&self) -> u32 {
        match self {
            TorsionStructure::Cyclic(n) => *n,
            TorsionStructure::TwoByTwoN(n) => 4 * n,
        }
    }

    /// Invariant factors, e.g. [6] for Z/6Z, [2, 6] for Z/2Z × Z/6Z.
    /// Empty for the trivial group.
    pub fn invariants(&self) -> Vec<u32> {
        match self {
            TorsionStructure::Cyclic(1) => vec![],
            TorsionStructure::Cyclic(n) => vec![*n],
            TorsionStructure::TwoByTwoN(n) => vec![2, 2 * n],
        }
    }
}

impl fmt::Display for TorsionStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TorsionStructure::Cyclic(1) => write!(f, "trivial"),
            TorsionStructure::Cyclic(n) => write!(f, "Z/{}Z", n),
            TorsionStructure::TwoByTwoN(n) => write!(f, "Z/2Z x Z/{}Z", 2 * n),
        }
    }
}

/// The torsion subgroup E(Q)_tors of an elliptic curve over Q.
#[derive(Debug, Clone)]
pub struct TorsionSubgroup {
    /// |E(Q)_tors| (including the point at infinity).
    pub order: u32,
    /// Group structure per Mazur's classification.
    pub structure: TorsionStructure,
    /// All affine torsion points *on the original input model*, with their
    /// exact orders, sorted by (x, y). Length = order − 1.
    pub points: Vec<(Point, u32)>,
}

impl EllipticCurve {
    /// The exact torsion subgroup E(Q)_tors, computed by minimal model +
    /// reduction bound + Lutz–Nagell; see the module docs of
    /// [`crate::torsion`]. The input model may be non-minimal; returned
    /// points are in the coordinates of the input model.
    ///
    /// # Panics
    ///
    /// Panics if the curve is singular, or if an internal certification
    /// step fails (which would indicate a bug, never a wrong answer).
    pub fn torsion_subgroup(&self) -> TorsionSubgroup {
        torsion_subgroup(self)
    }

    /// |E(Q)_tors|. Convenience wrapper around [`Self::torsion_subgroup`].
    pub fn torsion_order(&self) -> u32 {
        self.torsion_subgroup().order
    }

    /// The exact order of a point, or `None` if it has infinite order.
    ///
    /// By Mazur's theorem a rational torsion point has order in
    /// {1,…,10, 12}, so only 12 multiples need to be computed (exactly).
    ///
    /// # Panics
    ///
    /// Panics if `p` is not on the curve.
    pub fn point_order(&self, p: &Point) -> Option<u32> {
        assert!(self.is_on_curve(p), "point_order: point not on curve");
        if p.infinity {
            return Some(1);
        }
        // q runs through [n]P for n = 2, …, 12; by Mazur a torsion point
        // has order ≤ 12, so if no multiple vanishes P has infinite order.
        let mut q = p.clone();
        for n in 2..=12u32 {
            q = self.add_points(&q, p);
            if q.infinity {
                return Some(n);
            }
        }
        None
    }
}

/// All integer roots of the monic cubic X³ + aX + c, found by exact binary
/// search on the (at most three) monotone integer intervals inside the
/// Cauchy bound |X| ≤ 1 + max(|a|, |c|). f'(X) = 3X² + a is ≥ 0 for
/// |X| ≥ m + 2 and ≤ 0 for |X| ≤ m, where m = isqrt(⌊−a/3⌋) when a < 0.
pub(crate) fn integer_cubic_roots(a: &Integer, c: &Integer) -> Vec<Integer> {
    let f = |x: &Integer| -> Integer { x.pow(3) + a.clone() * x.clone() + c.clone() };
    let bound = Integer::one() + if a.abs() > c.abs() { a.abs() } else { c.abs() };
    let mut roots: Vec<Integer> = Vec::new();

    // Monotone integer intervals [lo, hi] with direction (+1 increasing,
    // −1 decreasing), plus a few individually-checked integers near the
    // critical points.
    let mut intervals: Vec<(Integer, Integer, i32)> = Vec::new();
    let mut singles: Vec<Integer> = Vec::new();
    if a.signum() >= 0 {
        intervals.push((-bound.clone(), bound.clone(), 1));
    } else {
        let m = (-a.clone() / Integer::from(3)).sqrt().expect("nonneg sqrt");
        let m2 = m.clone() + Integer::from(2);
        intervals.push((-bound.clone(), -m2.clone(), 1));
        intervals.push((-m.clone(), m.clone(), -1));
        intervals.push((m2.clone(), bound.clone(), 1));
        for k in [
            -m2.clone(),
            -(m.clone() + Integer::one()),
            m.clone() + Integer::one(),
            m2,
        ] {
            singles.push(k);
        }
    }
    for x in singles {
        if x.abs() <= bound && f(&x).is_zero() {
            roots.push(x);
        }
    }
    for (lo, hi, dir) in intervals {
        if lo > hi {
            continue;
        }
        // g = dir * f is non-decreasing on the integer points of [lo, hi].
        let g = |x: &Integer| -> Integer {
            let v = f(x);
            if dir < 0 {
                -v
            } else {
                v
            }
        };
        let (mut lo, mut hi) = (lo, hi);
        if g(&lo).signum() > 0 || g(&hi).signum() < 0 {
            if f(&lo).is_zero() {
                roots.push(lo.clone());
            }
            if f(&hi).is_zero() {
                roots.push(hi.clone());
            }
            continue;
        }
        // Invariant: g(lo) <= 0 <= g(hi). Find the least x with g(x) >= 0.
        while lo < hi {
            // floor((lo + hi) / 2); BigInt division truncates toward zero.
            let s = lo.clone() + hi.clone();
            let mut mid = s.clone() / Integer::from(2);
            if s.signum() < 0 && mid.clone() * Integer::from(2) != s {
                mid = mid - Integer::one();
            }
            if g(&mid).signum() < 0 {
                lo = mid + Integer::one();
            } else {
                hi = mid;
            }
        }
        if f(&lo).is_zero() {
            roots.push(lo);
        }
    }
    roots.sort();
    roots.dedup();
    roots
}

/// gcd of #Ē(F_p) over the first `count` odd primes of good reduction for
/// the (assumed minimal) model `e`. |E(Q)_tors| divides the result.
fn reduction_bound(e: &EllipticCurve, count: usize) -> u64 {
    let (b2, b4, b6, _) = e.b_invariants();
    let mut bound: u64 = 0;
    let mut used = 0;
    let mut p: u64 = 3;
    while used < count {
        if is_small_prime(p) && !(&e.discriminant % &Integer::from(p as i64)).is_zero() {
            let pi = p as i128;
            let bb2 = reduce_mod(&b2, p) as i128;
            let bb4 = reduce_mod(&b4, p) as i128;
            let bb6 = reduce_mod(&b6, p) as i128;
            // #Ē(F_p) = 1 + Σ_x (1 + χ(4x³ + b2x² + 2b4x + b6))
            let mut n: u64 = 1;
            for x in 0..pi {
                let v = (((4 * x % pi) * x % pi * x % pi)
                    + bb2 * x % pi * x % pi
                    + 2 * bb4 * x % pi
                    + bb6)
                    .rem_euclid(pi) as u64;
                if v == 0 {
                    n += 1;
                } else if pow_mod(v, (p - 1) / 2, p) == 1 {
                    n += 2;
                }
            }
            bound = gcd_u64(bound, n);
            used += 1;
        }
        p += 2;
    }
    bound
}

fn is_small_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

fn reduce_mod(n: &Integer, p: u64) -> u64 {
    n.modulo(&Integer::from(p as i64)).to_i64() as u64
}

fn pow_mod(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mut r: u64 = 1;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = (r as u128 * b as u128 % m as u128) as u64;
        }
        b = (b as u128 * b as u128 % m as u128) as u64;
        e >>= 1;
    }
    r
}

fn gcd_u64(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd_u64(b, a % b)
    }
}

/// All y ≥ 0 with y² dividing n (given the factorization of n):
/// products Π p^{k_p} with k_p ≤ ⌊e_p/2⌋.
fn square_divisor_roots(fac: &[(Integer, u32)]) -> Vec<Integer> {
    let mut ys = vec![Integer::one()];
    for (p, e) in fac {
        let kmax = e / 2;
        let mut next = Vec::with_capacity(ys.len() * (kmax as usize + 1));
        for y in &ys {
            let mut pk = Integer::one();
            for _ in 0..=kmax {
                next.push(y.clone() * pk.clone());
                pk = pk * p.clone();
            }
        }
        ys = next;
    }
    ys.sort();
    ys.dedup();
    ys
}

/// See [`EllipticCurve::torsion_subgroup`].
pub fn torsion_subgroup(e: &EllipticCurve) -> TorsionSubgroup {
    assert!(
        !e.is_singular(),
        "torsion_subgroup: curve is singular (discriminant 0)"
    );

    // 1. Minimal model and the isomorphism e → e_min.
    let (emin, iso_to_min) = e.minimal_model();

    // 2. Reduction bound (uses 12 odd good primes).
    let bound = reduction_bound(&emin, 12);

    // 3. Short model E_s : Y² = X³ + AX + B with A = −27c4, B = −54c6,
    //    reached from e_min by (u, r, s, t) = (1/6, −b2/12, −a1/2, a1b2/24 − a3/2).
    let (b2, _, _, _) = emin.b_invariants();
    let (c4, c6) = emin.c_invariants();
    let a_s = Integer::from(-27) * c4;
    let b_s = Integer::from(-54) * c6;
    let eshort = EllipticCurve::from_short_weierstrass(a_s.clone(), b_s.clone());
    let iso_min_to_short = {
        let q = Rational::from_integer;
        let u = Rational::new(Integer::one(), Integer::from(6)).expect("1/6");
        let r = -(q(b2.clone()) / Rational::from_i64(12));
        let s = -(q(emin.a1.clone()) / Rational::from_i64(2));
        let t = q(emin.a1.clone()) * q(b2.clone()) / Rational::from_i64(24)
            - q(emin.a3.clone()) / Rational::from_i64(2);
        let iso = WeierstrassIsomorphism { u, r, s, t };
        assert!(
            iso.is_isomorphism(&emin, &eshort),
            "torsion_subgroup: minimal → short transformation failed certification"
        );
        iso
    };

    // Lutz–Nagell discriminant quantity: 4A³ + 27B² = −2⁸·3¹²·Δ_min.
    let d_ln = Integer::from(4) * a_s.pow(3) + Integer::from(27) * b_s.clone() * b_s.clone();
    assert!(
        d_ln == -(Integer::from(2).pow(8) * Integer::from(3).pow(12) * emin.discriminant.clone()),
        "torsion_subgroup: Lutz–Nagell discriminant identity failed"
    );

    // Candidate |Y| values: 0 and every y > 0 with y² | |D_LN|.
    let mut fac = factor(&emin.discriminant.abs());
    // merge in the explicit 2⁸·3¹² factor
    let mut merged: Vec<(Integer, u32)> = Vec::new();
    let mut e2 = 8u32;
    let mut e3 = 12u32;
    for (p, k) in fac.drain(..) {
        if p == Integer::from(2) {
            e2 += k;
        } else if p == Integer::from(3) {
            e3 += k;
        } else {
            merged.push((p, k));
        }
    }
    merged.push((Integer::from(2), e2));
    merged.push((Integer::from(3), e3));
    let ys = square_divisor_roots(&merged);

    // 4. Collect torsion points of E_s (with orders), screened by Mazur.
    let mut short_points: Vec<(Point, u32)> = Vec::new();
    let push_if_torsion = |x: &Integer, y: &Integer, out: &mut Vec<(Point, u32)>| {
        let pt = Point::new(
            Rational::from_integer(x.clone()),
            Rational::from_integer(y.clone()),
        );
        debug_assert!(eshort.is_on_curve(&pt));
        if let Some(order) = eshort.point_order(&pt) {
            out.push((pt, order));
        }
    };
    // Y = 0 (2-torsion): roots of X³ + AX + B.
    for x in integer_cubic_roots(&a_s, &b_s) {
        push_if_torsion(&x, &Integer::zero(), &mut short_points);
    }
    // Y = ±y, y > 0 with y² | D_LN: roots of X³ + AX + (B − y²).
    for y in ys.iter() {
        let c = b_s.clone() - y.clone() * y.clone();
        for x in integer_cubic_roots(&a_s, &c) {
            // check Y² = X³ + AX + B exactly (it does by construction)
            push_if_torsion(&x, y, &mut short_points);
            push_if_torsion(&x, &(-y.clone()), &mut short_points);
        }
    }
    short_points.sort_by(|(p1, _), (p2, _)| (&p1.x, &p1.y).cmp(&(&p2.x, &p2.y)));
    short_points.dedup_by(|(p1, _), (p2, _)| p1 == p2);

    let order = short_points.len() as u32 + 1;

    // 5. Certification.
    // (a) |T| must divide the reduction bound (theorem: injectivity of
    //     reduction on torsion at odd good primes).
    assert!(
        bound.is_multiple_of(order as u64),
        "torsion_subgroup: |T| = {} does not divide reduction bound {}",
        order,
        bound
    );
    // (b) Closure under the group law.
    for (p1, _) in &short_points {
        for (p2, _) in &short_points {
            let s = eshort.add_points(p1, p2);
            assert!(
                s.infinity || short_points.iter().any(|(q, _)| *q == s),
                "torsion_subgroup: found set not closed under addition"
            );
        }
    }
    // (c) Structure per Mazur.
    let two_torsion = 1 + short_points.iter().filter(|(_, o)| *o == 2).count() as u32;
    let max_order = short_points.iter().map(|(_, o)| *o).max().unwrap_or(1);
    let structure = match two_torsion {
        1 | 2 => {
            assert!(
                max_order == order,
                "torsion_subgroup: cyclic case but max order {} != |T| = {}",
                max_order,
                order
            );
            assert!(
                (1..=10).contains(&order) || order == 12,
                "torsion_subgroup: cyclic order {} violates Mazur",
                order
            );
            TorsionStructure::Cyclic(order)
        }
        4 => {
            assert!(
                order.is_multiple_of(4),
                "torsion_subgroup: full 2-torsion but 4 ∤ |T|"
            );
            let n = order / 4;
            assert!(
                max_order == order / 2 || (order == 4 && max_order == 2),
                "torsion_subgroup: Z/2×Z/2n case but max order {} for |T| = {}",
                max_order,
                order
            );
            assert!(
                (1..=4).contains(&n),
                "torsion_subgroup: Z/2 x Z/{} violates Mazur",
                2 * n
            );
            TorsionStructure::TwoByTwoN(n)
        }
        k => panic!("torsion_subgroup: impossible 2-torsion count {}", k),
    };
    assert_eq!(structure.order(), order);

    // 6. Map back to the original model and re-verify points and orders.
    let iso_orig_to_short = iso_to_min.compose(&iso_min_to_short);
    let back = iso_orig_to_short.inverse();
    let mut points: Vec<(Point, u32)> = short_points
        .iter()
        .map(|(p, o)| {
            let q = back.map_point(p);
            assert!(
                e.is_on_curve(&q),
                "torsion_subgroup: mapped-back torsion point not on original curve"
            );
            assert_eq!(
                e.point_order(&q),
                Some(*o),
                "torsion_subgroup: order changed under isomorphism"
            );
            (q, *o)
        })
        .collect();
    points.sort_by(|(p1, _), (p2, _)| (&p1.x, &p1.y).cmp(&(&p2.x, &p2.y)));

    TorsionSubgroup {
        order,
        structure,
        points,
    }
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

    fn assert_torsion(
        e: &EllipticCurve,
        expected: TorsionStructure,
        sample_points: &[(i64, i64, u32)],
    ) {
        let t = e.torsion_subgroup();
        assert_eq!(t.structure, expected, "curve {}", e);
        assert_eq!(t.order, expected.order());
        assert_eq!(t.points.len() as u32 + 1, t.order);
        for &(x, y, o) in sample_points {
            let p = Point::from_integers(x, y);
            assert!(
                t.points.iter().any(|(q, oo)| *q == p && *oo == o),
                "expected torsion point ({}, {}) of order {} on {}",
                x,
                y,
                o,
                e
            );
        }
    }

    // Every expected structure and point below was verified with an exact
    // brute Lutz–Nagell reference computation in Python (fractions + sympy)
    // before being asserted here; curve labels are informational only.

    #[test]
    fn torsion_11a1_z5() {
        // 11a1: y² + y = x³ − x² − 10x − 20; Z/5Z with points
        // (5, ±…), (16, ±…).
        let e = curve(0, -1, 1, -10, -20);
        assert_torsion(
            &e,
            TorsionStructure::Cyclic(5),
            &[(5, 5, 5), (5, -6, 5), (16, 60, 5), (16, -61, 5)],
        );
    }

    #[test]
    fn torsion_37a1_trivial() {
        // 37a1: y² + y = x³ − x; trivial torsion (rank-1 curve).
        let e = curve(0, 0, 1, -1, 0);
        assert_torsion(&e, TorsionStructure::Cyclic(1), &[]);
    }

    #[test]
    fn torsion_y2_x3_minus_x_z2xz2() {
        // y² = x³ − x: full 2-torsion Z/2 × Z/2 at x ∈ {−1, 0, 1}.
        let e = curve(0, 0, 0, -1, 0);
        assert_torsion(
            &e,
            TorsionStructure::TwoByTwoN(1),
            &[(-1, 0, 2), (0, 0, 2), (1, 0, 2)],
        );
    }

    #[test]
    fn torsion_14a1_z6() {
        // 14a1: y² + xy + y = x³ + 4x − 6; Z/6Z; (9, 23) has order 6.
        let e = curve(1, 0, 1, 4, -6);
        assert_torsion(
            &e,
            TorsionStructure::Cyclic(6),
            &[(1, -1, 2), (2, 2, 3), (9, 23, 6)],
        );
    }

    #[test]
    fn torsion_small_orders() {
        // y² = x³ + x: Z/2 ((0,0)); reduction bound is 4 here, so this also
        // exercises |T| < bound.
        assert_torsion(
            &curve(0, 0, 0, 1, 0),
            TorsionStructure::Cyclic(2),
            &[(0, 0, 2)],
        );
        // y² = x³ + 4: Z/3 with (0, ±2).
        assert_torsion(
            &curve(0, 0, 0, 0, 4),
            TorsionStructure::Cyclic(3),
            &[(0, 2, 3), (0, -2, 3)],
        );
        // y² + xy + y = x³ + x²: Z/4 ((0,0) of order 4).
        assert_torsion(
            &curve(1, 1, 1, 0, 0),
            TorsionStructure::Cyclic(4),
            &[(-1, 0, 2), (0, 0, 4)],
        );
    }

    #[test]
    fn torsion_kubert_z7_z8() {
        // Kubert/Tate normal form curves; torsion verified in Python.
        // y² − xy − 4y = x³ − 4x²: Z/7 with (0,0) of order 7.
        assert_torsion(
            &curve(-1, -4, -4, 0, 0),
            TorsionStructure::Cyclic(7),
            &[(0, 0, 7), (4, 0, 7), (2, 2, 7)],
        );
        // y² − xy − 24y = x³ − 12x²: Z/8 with (0,0) of order 8.
        assert_torsion(
            &curve(-1, -12, -24, 0, 0),
            TorsionStructure::Cyclic(8),
            &[(0, 0, 8), (8, 16, 2), (12, 0, 4)],
        );
    }

    #[test]
    fn torsion_z9_z10_z12() {
        // y² + xy + y = x³ − x² − 14x + 29: Z/9 ((3,1) of order 9).
        assert_torsion(
            &curve(1, -1, 1, -14, 29),
            TorsionStructure::Cyclic(9),
            &[(3, 1, 9), (1, 3, 3)],
        );
        // y² + 5xy − 18y = x³ − 6x²: Z/10 ((0,0) of order 10). This model is
        // minimal but not reduced, so it also exercises the (u = 1) r,s,t
        // normalization path and point mapping.
        assert_torsion(
            &curve(5, -6, -18, 0, 0),
            TorsionStructure::Cyclic(10),
            &[(0, 0, 10), (2, 4, 2), (6, 0, 5)],
        );
        // y² − 3xy − 90y = x³ − 30x²: Z/12 ((0,0) of order 12).
        assert_torsion(
            &curve(-3, -30, -90, 0, 0),
            TorsionStructure::Cyclic(12),
            &[(0, 0, 12), (-6, 36, 2), (30, 0, 6)],
        );
    }

    #[test]
    fn torsion_z2_products() {
        // y² = x(x+1)(x+4) = x³ + 5x² + 4x: Z/2 × Z/4.
        assert_torsion(
            &curve(0, 5, 0, 4, 0),
            TorsionStructure::TwoByTwoN(2),
            &[(-4, 0, 2), (0, 0, 2), (2, 6, 4), (-2, 2, 4)],
        );
        // y² = x(x−32)(x−27) → model (0, −59, 0, 864, 0): Z/2 × Z/6.
        assert_torsion(
            &curve(0, -59, 0, 864, 0),
            TorsionStructure::TwoByTwoN(3),
            &[(0, 0, 2), (27, 0, 2), (32, 0, 2), (36, 36, 3), (12, 60, 6)],
        );
        // y² + xy = x³ − 1070x + 7812: Z/2 × Z/8 (verified in Python).
        assert_torsion(
            &curve(1, 0, 0, -1070, 7812),
            TorsionStructure::TwoByTwoN(4),
            &[],
        );
    }

    #[test]
    fn torsion_of_nonminimal_model() {
        // 11a1 scaled by λ = 2: y² + 8y = x³ − 4x² − 160x − 1280 is a
        // non-minimal integral model; torsion must still be Z/5Z, with the
        // points in the *scaled* coordinates (verified in Python).
        let e = curve(0, -4, 8, -160, -1280);
        assert_torsion(
            &e,
            TorsionStructure::Cyclic(5),
            &[(20, 40, 5), (20, -48, 5), (64, 480, 5), (64, -488, 5)],
        );
    }

    #[test]
    fn torsion_rank_positive_curves_trivial() {
        // 389a1 (rank 2): trivial torsion.
        assert_torsion(&curve(0, 1, 1, -2, 0), TorsionStructure::Cyclic(1), &[]);
        // y² = x³ − 2 (Mordell, rank 1): trivial torsion.
        assert_torsion(&curve(0, 0, 0, 0, -2), TorsionStructure::Cyclic(1), &[]);
    }

    #[test]
    fn torsion_order_convenience() {
        assert_eq!(curve(0, -1, 1, -10, -20).torsion_order(), 5);
        assert_eq!(curve(0, 0, 1, -1, 0).torsion_order(), 1);
    }

    #[test]
    fn point_order_detects_infinite_order() {
        // (0,0) on 37a1 is the rank generator: infinite order.
        let e = curve(0, 0, 1, -1, 0);
        let p = Point::from_integers(0, 0);
        assert_eq!(e.point_order(&p), None);
        // (5,5) on 11a1 has order 5.
        let e11 = curve(0, -1, 1, -10, -20);
        assert_eq!(e11.point_order(&Point::from_integers(5, 5)), Some(5));
        assert_eq!(e11.point_order(&Point::infinity()), Some(1));
    }

    #[test]
    fn integer_cubic_roots_exact() {
        // (X−1)(X−2)(X+3) = X³ − 7X + 6
        let r = integer_cubic_roots(&Integer::from(-7), &Integer::from(6));
        assert_eq!(
            r,
            vec![Integer::from(-3), Integer::from(1), Integer::from(2)]
        );
        // X³ + X + 1: no integer roots
        assert!(integer_cubic_roots(&Integer::from(1), &Integer::from(1)).is_empty());
        // X³ − 27·496·... use X³ − 4X = X(X−2)(X+2)
        let r = integer_cubic_roots(&Integer::from(-4), &Integer::from(0));
        assert_eq!(
            r,
            vec![Integer::from(-2), Integer::from(0), Integer::from(2)]
        );
    }

    #[test]
    fn two_torsion_rank_real() {
        assert_eq!(curve(0, 0, 0, -1, 0).two_torsion_rank(), 2); // y²=x³−x
        assert_eq!(curve(0, 0, 0, 1, 0).two_torsion_rank(), 1); // y²=x³+x
        assert_eq!(curve(0, 0, 1, -1, 0).two_torsion_rank(), 0); // 37a1
        assert_eq!(curve(0, -1, 1, -10, -20).two_torsion_rank(), 0); // 11a1
        assert_eq!(curve(1, 0, 1, 4, -6).two_torsion_rank(), 1); // 14a1 (Z/6)
    }
}
