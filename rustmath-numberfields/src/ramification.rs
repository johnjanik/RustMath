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

use rustmath_integers::Integer;
use rustmath_rationals::Rational;

fn rq(n: i64) -> Rational {
    Rational::from_i64(n)
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
