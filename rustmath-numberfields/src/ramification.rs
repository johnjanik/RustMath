//! Higher ramification groups: the lower/upper-numbering filtration calculus for a
//! Galois extension of local fields (Module 9, the wild endgame).
//!
//! For a Galois extension `L/K` of local fields with group `G = Gal(L/K)`, the
//! **lower-numbering ramification filtration** is `G_{-1}=G ⊇ G_0 ⊇ G_1 ⊇ …` with
//! `G_0` the inertia group and `G_i = {σ : v_L(σπ − π) ≥ i+1}`. It satisfies:
//! `G_0/G_1` is cyclic of order prime to `p` (the *tame* quotient) and each
//! `G_i/G_{i+1}` (`i ≥ 1`) is an elementary abelian `p`-group (*wild* inertia, the
//! `p`-Sylow `G_1`). The data is the sequence of orders `gᵢ = |G_i|`.
//!
//! This module records that sequence and computes the invariants built on it:
//! - **Hilbert's different formula** `d(L/K) = Σ_{i≥0} (|G_i| − 1)`;
//! - the **Herbrand transition** functions `φ_{L/K}` and `ψ_{L/K} = φ⁻¹` converting
//!   lower ↔ upper numbering;
//! - the **upper-numbering breaks** `φ(b)` of the lower breaks `b`, which by the
//!   **Hasse–Arf theorem** are integers when `G` is abelian.
//!
//! Validated against PARI/GP `idealval(K.diff, P)`: the Hilbert different of the
//! reconstructed cyclotomic filtrations matches the actual different exponent, e.g.
//! `ℚ₂(i)`→2, `ℚ₂(√2)`→3, `ℚ₃(ζ₉)`→9, `ℚ₂(ζ₈)`→8, `ℚ₃(ζ₂₇)`→45; and the upper
//! breaks reproduce the textbook cyclotomic breaks (integers, per Hasse–Arf).

use crate::local_field::{eisenstein_different_exponent, EisensteinElement};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

fn rq(n: i64) -> Rational {
    Rational::from_i64(n)
}

fn binom_int(n: usize, k: usize) -> Integer {
    if k > n {
        return Integer::zero();
    }
    let k = k.min(n - k);
    let mut num = Integer::one();
    let mut den = Integer::one();
    for i in 0..k {
        num = num * Integer::from((n - i) as i64);
        den = den * Integer::from((i + 1) as i64);
    }
    num / den
}

/// Lower convex hull of integer points sorted by `x` (monotone chain): keep the
/// vertices of the Newton polygon.
fn lower_hull(pts: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut h: Vec<(i64, i64)> = Vec::new();
    for &p in pts {
        while h.len() >= 2 {
            let (x1, y1) = h[h.len() - 2];
            let (x2, y2) = h[h.len() - 1];
            let (x3, y3) = p;
            // cross product of edges (p1→p2) and (p1→p3); pop while not a left turn.
            let cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
            if cross <= 0 {
                h.pop();
            } else {
                break;
            }
        }
        h.push(p);
    }
    h
}

/// The **ramification polygon** of an Eisenstein polynomial `g` (degree `e`,
/// uniformizer `π`): the Newton polygon (over `K = ℚ_p[π]/(g)`, `v_K(π)=1`) of
/// `ρ(x) = g(πx + π)`. Its `e−1` roots are `(βᵢ − π)/π` for the conjugates `βᵢ ≠ π`,
/// so the slopes give the conjugate valuations `mᵢ = v_K(βᵢ − π)`. Returns the
/// multiset `{mᵢ}` (length `e−1`), computed **directly from `g`** with no
/// root-finding. `Σ mᵢ = v_K(g'(π)) =` the different exponent.
pub fn ramification_polygon(g: &[Integer], p: i64) -> Vec<Rational> {
    let e = g.len() - 1;
    if e <= 1 {
        return Vec::new();
    }
    // Precision in ℤ_p high enough to see v_K up to ~ different + e (≤ e² + e).
    let prec = (4 * e + 24) as u32;
    let gv = g.to_vec();
    let pi = EisensteinElement::uniformizer(p, prec, gv.clone());
    // π^i for i = 0..=e.
    let mut pip = vec![EisensteinElement::one(p, prec, gv.clone())];
    for _ in 1..=e {
        pip.push(pip.last().unwrap().mul(&pi));
    }
    // ρ_k = Σ_{i=k}^e g_i·C(i,k)·π^i (coefficient of x^k in g(πx+π) = Σ_i g_i π^i (x+1)^i).
    let mut pts: Vec<(i64, i64)> = Vec::new();
    for k in 1..=e {
        let mut rho = EisensteinElement::zero(p, prec, gv.clone());
        for i in k..=e {
            let scalar = g[i].clone() * binom_int(i, k);
            if scalar.is_zero() {
                continue;
            }
            let term =
                EisensteinElement::from_int(p, prec, gv.clone(), scalar).mul(&pip[i]);
            rho = rho.add(&term);
        }
        if let Some(v) = rho.valuation() {
            pts.push((k as i64, v));
        }
    }
    // Slopes of the lower hull → conjugate valuations m = −slope + 1, with
    // multiplicity = horizontal length of the segment.
    let hull = lower_hull(&pts);
    let mut ms: Vec<Rational> = Vec::new();
    for w in hull.windows(2) {
        let (k1, v1) = w[0];
        let (k2, v2) = w[1];
        let length = (k2 - k1) as usize;
        // m = −slope + 1 = (v1 − v2)/(k2 − k1) + 1.
        let m = Rational::new(Integer::from(v1 - v2), Integer::from(k2 - k1)).unwrap() + rq(1);
        for _ in 0..length {
            ms.push(m.clone());
        }
    }
    ms
}

/// The lower-numbering ramification filtration built from the ramification polygon
/// of an Eisenstein `g`, via `|G_i| = 1 + #{ conjugates β : v_K(β−π) ≥ i+1 }`.
///
/// **This is the true `Gal(K/ℚ_p)` filtration exactly when `K/ℚ_p` is Galois** (then
/// the `e−1` conjugates are the non-trivial automorphisms). For a non-Galois `K` the
/// conjugates are not automorphisms, so the returned object is the *ramification-
/// polygon shape*, not a literal group filtration — e.g. `x⁴−2` (Galois group `D₄`)
/// yields `[4,4,4,2,2]`, the shape of a `C₄`, because its closure is not seen here.
/// The caller must supply Galois-ness (e.g. via the residual/segmental data or the
/// splitting field). Its different always equals `eisenstein_different_exponent(g,p)`.
///
/// Returns `None` when the polygon has a non-integer slope — which *does* prove
/// `K/ℚ_p` is **not Galois** (a conjugate valuation `v_K(β−π) ∉ ℤ` forces `β ∉ K`).
pub fn wild_filtration_from_eisenstein(g: &[Integer], p: i64) -> Option<RamificationFiltration> {
    let ms = ramification_polygon(g, p);
    if ms.is_empty() || !ms.iter().all(|m| m.is_integer()) {
        return None;
    }
    let mvals: Vec<i64> = ms.iter().map(|m| m.numerator().to_i64()).collect();
    let maxm = *mvals.iter().max().unwrap();
    let mut orders = Vec::new();
    for i in 0..maxm {
        let gi = 1 + mvals.iter().filter(|&&m| m >= i + 1).count();
        orders.push(gi);
    }
    Some(RamificationFiltration::new(orders))
}

/// The lower-numbering ramification filtration of a Galois local extension, as the
/// orders `orders[i] = |G_i|` for `i = 0, 1, 2, …`. Beyond the stored length the
/// order is `1` (trivial). `orders[0] = |G_0|` is the inertia order (`= e` for a
/// totally ramified extension); the sequence must be nonincreasing with each entry
/// dividing the previous.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RamificationFiltration {
    pub orders: Vec<usize>,
}

impl RamificationFiltration {
    /// Build from the orders `|G_0|, |G_1|, …` (trailing `1`s trimmed).
    pub fn new(mut orders: Vec<usize>) -> Self {
        while orders.last() == Some(&1) {
            orders.pop();
        }
        RamificationFiltration { orders }
    }

    /// `|G_i|` (`= 1` past the stored length).
    pub fn order_at(&self, i: usize) -> usize {
        self.orders.get(i).copied().unwrap_or(1)
    }

    /// `|G_0|`, the inertia order (`= e` when totally ramified).
    pub fn inertia_order(&self) -> usize {
        self.order_at(0)
    }

    /// `|G_1|`, the wild inertia (`p`-Sylow of `G_0`).
    pub fn wild_order(&self) -> usize {
        self.order_at(1)
    }

    /// Tame quotient order `|G_0/G_1| = |G_0|/|G_1|`.
    pub fn tame_order(&self) -> usize {
        self.inertia_order() / self.wild_order().max(1)
    }

    /// **Hilbert's formula**: the different exponent `d(L/K) = Σ_{i≥0} (|G_i| − 1)`.
    pub fn different_exponent(&self) -> i64 {
        self.orders.iter().map(|&g| g as i64 - 1).sum()
    }

    /// The lower-numbering breaks: indices `i ≥ 0` with `|G_i| > |G_{i+1}|`.
    pub fn lower_breaks(&self) -> Vec<usize> {
        (0..self.orders.len())
            .filter(|&i| self.order_at(i) > self.order_at(i + 1))
            .collect()
    }

    /// The Herbrand function `φ_{L/K}(u)` for `u ≥ −1`. `φ(u) = u` on `[−1, 0]`, and
    /// for `u ≥ 0`, `φ(u) = (1/g₀)·(Σ_{i=1}^{m} gᵢ + (u−m)·g_{m+1})` with `m = ⌊u⌋`.
    pub fn phi(&self, u: &Rational) -> Rational {
        let zero = rq(0);
        if *u <= zero {
            return u.clone(); // φ(u) = u on [−1, 0]
        }
        let g0 = rq(self.inertia_order() as i64);
        let m = floor_usize(u);
        let mut sum = rq(0);
        for i in 1..=m {
            sum = sum + rq(self.order_at(i) as i64);
        }
        let frac = u.clone() - rq(m as i64);
        let tail = frac * rq(self.order_at(m + 1) as i64);
        (sum + tail) / g0
    }

    /// The inverse Herbrand function `ψ_{L/K} = φ⁻¹`. `ψ(v) = v` on `[−1, 0]`.
    pub fn psi(&self, v: &Rational) -> Rational {
        let zero = rq(0);
        if *v <= zero {
            return v.clone();
        }
        // φ is piecewise linear with breakpoints at integers; locate the segment.
        let g0 = self.inertia_order() as i64;
        let mut m = 0usize;
        loop {
            let phi_m = self.phi(&rq(m as i64));
            let phi_m1 = self.phi(&rq((m + 1) as i64));
            if *v <= phi_m1 || self.order_at(m + 1) == 0 {
                // u = m + (v − φ(m))·g₀/g_{m+1}
                let slope_inv = Rational::new(
                    Integer::from(g0),
                    Integer::from(self.order_at(m + 1).max(1) as i64),
                )
                .unwrap();
                let _ = phi_m1;
                return rq(m as i64) + (v.clone() - phi_m) * slope_inv;
            }
            m += 1;
            if m > 1_000_000 {
                return v.clone(); // safety
            }
        }
    }

    /// The upper-numbering breaks `φ(b)` of the lower breaks `b`. By **Hasse–Arf**
    /// these are integers when `G` is abelian.
    pub fn upper_breaks(&self) -> Vec<Rational> {
        self.lower_breaks().into_iter().map(|b| self.phi(&rq(b as i64))).collect()
    }

    /// Whether every upper break is an integer (the Hasse–Arf property; necessary for
    /// `G` abelian).
    pub fn hasse_arf_integral(&self) -> bool {
        self.upper_breaks().iter().all(|v| v.is_integer())
    }
}

fn floor_usize(r: &Rational) -> usize {
    // r ≥ 0: truncated division of numerator by denominator is the floor.
    let q = r.numerator().clone() / r.denominator().clone();
    q.to_i64().max(0) as usize
}

/// The filtration of a **tamely** totally ramified extension of degree `e` (`p ∤ e`):
/// `G_0` cyclic of order `e`, `G_1 = 1`. Different exponent `e − 1`.
pub fn tame_filtration(e: usize) -> RamificationFiltration {
    RamificationFiltration::new(vec![e])
}

/// The filtration of a **cyclic degree-`p`** totally (wildly) ramified extension with
/// a single lower break `b`: `G_0 = … = G_b = C_p`, `G_{b+1} = 1`. Its different
/// exponent is `(b+1)(p−1)`. Conversely, [`cyclic_p_break_from_different`] recovers
/// `b` from `d`.
pub fn cyclic_p_filtration(p: usize, b: usize) -> RamificationFiltration {
    RamificationFiltration::new(vec![p; b + 1])
}

/// The single lower break `b` of a cyclic degree-`p` totally ramified extension with
/// different exponent `d`: `d = (b+1)(p−1)` ⇒ `b = d/(p−1) − 1`. Returns `None` if
/// `d` is not of that form.
pub fn cyclic_p_break_from_different(p: usize, d: i64) -> Option<usize> {
    let pm1 = (p - 1) as i64;
    if d <= 0 || d % pm1 != 0 {
        return None;
    }
    let b = d / pm1 - 1;
    if b < 0 {
        None
    } else {
        Some(b as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filt(v: &[usize]) -> RamificationFiltration {
        RamificationFiltration::new(v.to_vec())
    }

    #[test]
    fn hilbert_different_matches_pari_gp() {
        // Lower-numbering filtrations reconstructed from cyclotomic theory; the
        // Hilbert different equals the PARI/GP idealval(K.diff, P).
        assert_eq!(filt(&[2, 2]).different_exponent(), 2); // ℚ₂(i)
        assert_eq!(filt(&[2, 2, 2]).different_exponent(), 3); // ℚ₂(√2)
        assert_eq!(filt(&[6, 3, 3]).different_exponent(), 9); // ℚ₃(ζ₉), C₆
        assert_eq!(filt(&[4, 4, 2, 2]).different_exponent(), 8); // ℚ₂(ζ₈), V₄
        assert_eq!(
            filt(&[18, 9, 9, 3, 3, 3, 3, 3, 3]).different_exponent(),
            45 // ℚ₃(ζ₂₇), C₁₈
        );
    }

    #[test]
    fn upper_breaks_and_hasse_arf() {
        // ℚ₂(i): single lower break at 1 → upper break 1.
        let q2i = filt(&[2, 2]);
        assert_eq!(q2i.lower_breaks(), vec![1]);
        assert_eq!(q2i.upper_breaks(), vec![rq(1)]);
        assert!(q2i.hasse_arf_integral());

        // ℚ₂(√2): lower break at 2 → upper break 2.
        assert_eq!(filt(&[2, 2, 2]).upper_breaks(), vec![rq(2)]);

        // ℚ₃(ζ₉) = C₆: tame break (lower 0 → upper 0) and wild break (lower 2 → 1).
        let z9 = filt(&[6, 3, 3]);
        assert_eq!(z9.lower_breaks(), vec![0, 2]);
        assert_eq!(z9.upper_breaks(), vec![rq(0), rq(1)]);
        assert!(z9.hasse_arf_integral());

        // ℚ₂(ζ₈) = V₄: lower breaks 1, 3 → upper breaks 1, 2.
        let z8 = filt(&[4, 4, 2, 2]);
        assert_eq!(z8.lower_breaks(), vec![1, 3]);
        assert_eq!(z8.upper_breaks(), vec![rq(1), rq(2)]);
        assert!(z8.hasse_arf_integral());

        // ℚ₃(ζ₂₇) = C₁₈: lower breaks 0, 2, 8 → upper breaks 0, 1, 2.
        let z27 = filt(&[18, 9, 9, 3, 3, 3, 3, 3, 3]);
        assert_eq!(z27.lower_breaks(), vec![0, 2, 8]);
        assert_eq!(z27.upper_breaks(), vec![rq(0), rq(1), rq(2)]);
        assert!(z27.hasse_arf_integral());
    }

    #[test]
    fn herbrand_phi_psi_are_inverse() {
        let f = filt(&[18, 9, 9, 3, 3, 3, 3, 3, 3]);
        for (n, d) in [(0, 1), (1, 1), (3, 2), (5, 1), (7, 3), (17, 2)] {
            let u = Rational::new(Integer::from(n), Integer::from(d)).unwrap();
            assert_eq!(f.psi(&f.phi(&u)), u, "ψ∘φ ≠ id at {n}/{d}");
        }
        // φ(0)=0 and φ at the inertia jump.
        assert_eq!(f.phi(&rq(0)), rq(0));
    }

    #[test]
    fn tame_and_cyclic_constructors() {
        // Tame degree e: different e−1, single break? G_0 cyclic, G_1=1.
        let t = tame_filtration(5);
        assert_eq!(t.different_exponent(), 4);
        assert_eq!(t.lower_breaks(), vec![0]); // tame break at lower 0
        assert!(t.wild_order() == 1);

        // Cyclic C_p reconstruction from the different.
        assert_eq!(cyclic_p_break_from_different(2, 2), Some(1)); // ℚ₂(i)
        assert_eq!(cyclic_p_break_from_different(2, 3), Some(2)); // ℚ₂(√2)
        assert_eq!(cyclic_p_filtration(2, 1).different_exponent(), 2);
        assert_eq!(cyclic_p_filtration(2, 2).different_exponent(), 3);
        // C_3 wild: a break-1 extension has different (1+1)(3−1)=4.
        assert_eq!(cyclic_p_filtration(3, 1).different_exponent(), 4);
        assert_eq!(cyclic_p_break_from_different(3, 4), Some(1));
        assert_eq!(cyclic_p_break_from_different(2, 5), Some(4));
    }

    #[test]
    fn ramification_polygon_sum_is_different() {
        // Σ m_i = different exponent (the product formula g'(π) = ∏(π − β)).
        let cases: &[(&[i64], i64)] = &[
            (&[2, -2, 1], 2),                  // Q_2(i), d=2
            (&[-2, 0, 1], 2),                  // Q_2(√2), d=3
            (&[2, 4, 6, 4, 1], 2),             // Φ_8(x+1), Q_2(ζ_8), d=8
            (&[-2, 0, 0, 0, 1], 2),            // x⁴−2, d=11 (NOT Galois)
            (&[3, 9, 18, 21, 15, 6, 1], 3),    // Φ_9(x+1), Q_3(ζ_9), d=9
        ];
        for &(g, p) in cases {
            let gz: Vec<Integer> = g.iter().map(|&x| Integer::from(x)).collect();
            let ms = ramification_polygon(&gz, p);
            assert_eq!(ms.len(), g.len() - 2, "e−1 conjugate valuations for {g:?}");
            let sum: Rational = ms.iter().fold(rq(0), |a, m| a + m.clone());
            assert_eq!(
                sum,
                rq(eisenstein_different_exponent(&gz, p)),
                "Σ m_i = different for {g:?}"
            );
        }
    }

    #[test]
    fn wild_filtration_from_polynomial_matches_known() {
        let g = |v: &[i64]| -> Vec<Integer> { v.iter().map(|&x| Integer::from(x)).collect() };
        // Q_2(i): C_2, filtration [2,2].
        assert_eq!(
            wild_filtration_from_eisenstein(&g(&[2, -2, 1]), 2).unwrap(),
            filt(&[2, 2])
        );
        // Q_2(√2): C_2, filtration [2,2,2].
        assert_eq!(
            wild_filtration_from_eisenstein(&g(&[-2, 0, 1]), 2).unwrap(),
            filt(&[2, 2, 2])
        );
        // Q_2(ζ_8): V_4, filtration [4,4,2,2] — computed straight from Φ_8(x+1).
        let z8 = wild_filtration_from_eisenstein(&g(&[2, 4, 6, 4, 1]), 2).unwrap();
        assert_eq!(z8, filt(&[4, 4, 2, 2]));
        assert_eq!(z8.different_exponent(), 8);
        assert_eq!(z8.upper_breaks(), vec![rq(1), rq(2)]);
        // Q_3(ζ_9): C_6, filtration [6,3,3] from Φ_9(x+1).
        let z9 = wild_filtration_from_eisenstein(&g(&[3, 9, 18, 21, 15, 6, 1]), 3).unwrap();
        assert_eq!(z9, filt(&[6, 3, 3]));
        assert_eq!(z9.different_exponent(), 9);
    }

    #[test]
    fn ramification_polygon_non_galois_x4_minus_2() {
        // x⁴−2 (Galois group D₄, not Galois as a degree-4 ext over ℚ_2 — needs i).
        // Its ramification polygon still gives integer conjugate valuations {3,3,5}
        // (Σ = 11 = different); integer slopes do NOT imply Galois, so the
        // "filtration" [4,4,4,2,2] is the polygon shape (of a C₄), not Gal(D₄).
        let g: Vec<Integer> = [-2, 0, 0, 0, 1].iter().map(|&x| Integer::from(x)).collect();
        let mut ms: Vec<i64> =
            ramification_polygon(&g, 2).iter().map(|m| m.numerator().to_i64()).collect();
        ms.sort_unstable();
        assert_eq!(ms, vec![3, 3, 5]);
        let shape = wild_filtration_from_eisenstein(&g, 2).unwrap();
        assert_eq!(shape, filt(&[4, 4, 4, 2, 2]));
        assert_eq!(shape.different_exponent(), 11);
    }

    #[test]
    fn cross_check_with_eisenstein_different() {
        // The filtration different of the cyclic C₂ extensions equals the intrinsic
        // Eisenstein different v_π(g'(π)) from local_field.
        use crate::local_field::eisenstein_different_exponent;
        let g_sqrt2: Vec<Integer> = [-2, 0, 1].iter().map(|&x| Integer::from(x)).collect();
        assert_eq!(
            cyclic_p_filtration(2, 2).different_exponent(),
            eisenstein_different_exponent(&g_sqrt2, 2)
        );
        let g_i: Vec<Integer> = [2, -2, 1].iter().map(|&x| Integer::from(x)).collect();
        assert_eq!(
            cyclic_p_filtration(2, 1).different_exponent(),
            eisenstein_different_exponent(&g_i, 2)
        );
    }
}
