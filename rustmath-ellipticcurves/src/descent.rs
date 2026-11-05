//! 2-descent interface: honest rank bounds and φ-Selmer class sets.
//!
//! The actual descent machinery (descent via 2-isogeny and full 2-descent,
//! with exact local solvability and certified `[lower, upper]` rank
//! intervals) lives in [`crate::rank`]; this module keeps the historical
//! `TwoDescent` / `SelmerGroup` names as a thin interface over it.
//!
//! The previous contents of this module were facades: a hardcoded torsor
//! list, a "Selmer group" whose `rank_bound` field was never computed, and
//! an `unimplemented!` `rank_bound`. They have been replaced by the real
//! thing; see the honesty contract in the [`crate::rank`] module docs
//! (results are certified intervals, never fabricated integers; curves
//! without rational 2-torsion are an honest refusal).

use crate::curve::{EllipticCurve, Point};
use crate::rank::{self, RankBoundResult};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// The φ- and φ̂-Selmer class sets of the descent via 2-isogeny at the
/// first (smallest-x) rational 2-torsion point of the curve.
///
/// Classes are square-free integers representing elements of Q\*/(Q\*)²;
/// both sets are certified subgroups containing the point-witnessed
/// classes.
#[derive(Debug, Clone)]
pub struct SelmerGroup {
    /// (a, b) of the model y² = x³ + ax² + bx the isogeny descent ran on
    /// (the curve translated so the chosen 2-torsion point is (0, 0)).
    pub isogeny_model: (Integer, Integer),
    /// Everywhere-locally-solvable classes for φ : E → E′.
    pub phi_classes: Vec<Integer>,
    /// Everywhere-locally-solvable classes for the dual isogeny φ̂ : E′ → E.
    pub phi_prime_classes: Vec<Integer>,
    /// dim₂ Sel_φ + dim₂ Sel_φ̂ − 2: a certified upper bound on rank E(Q)
    /// from this single isogeny. [`TwoDescent::rank_bound`], which combines
    /// the descents at every rational 2-torsion point (and full 2-descent
    /// when E[2] ⊆ E(Q)), may be tighter.
    pub rank_upper_bound: u32,
}

/// 2-descent driver for rank bounds. See [`crate::rank`] for semantics.
pub struct TwoDescent<'a> {
    curve: &'a EllipticCurve,
}

impl<'a> TwoDescent<'a> {
    pub fn new(curve: &'a EllipticCurve) -> Self {
        Self { curve }
    }

    /// Certified rank bounds; identical to [`EllipticCurve::rank_bounds`].
    pub fn rank_bounds(&self) -> RankBoundResult {
        self.curve.rank_bounds()
    }

    /// A certified **upper** bound on rank E(Q), from everywhere-local
    /// solvability of the descent torsors (never from a point search, so it
    /// cannot be an underestimate).
    ///
    /// # Panics
    ///
    /// Panics (honest refusal, never a guess) if the curve has no rational
    /// 2-torsion; use [`Self::rank_bounds`] for the non-panicking form.
    pub fn rank_bound(&self) -> i32 {
        match self.curve.rank_bounds() {
            RankBoundResult::Bounds(b) => b.upper as i32,
            RankBoundResult::Unresolved { reason } => panic!("rank_bound: {}", reason),
        }
    }

    /// The φ/φ̂-Selmer class sets for the 2-isogeny at the first rational
    /// 2-torsion point.
    ///
    /// # Panics
    ///
    /// Panics (honest refusal) if the curve has no rational 2-torsion.
    pub fn compute_selmer_group(&self) -> SelmerGroup {
        let sm = rank::short_model(self.curve);
        let x0 = sm.roots.first().unwrap_or_else(|| {
            panic!(
                "compute_selmer_group: no rational 2-torsion, so the 2-isogeny descent \
                 does not apply over Q (see EllipticCurve::rank_bounds for the honest \
                 Unresolved form)"
            )
        });
        let a = sm.eshort.a2.clone() + Integer::from(3) * x0.clone();
        let b = Integer::from(3) * x0.clone() * x0.clone()
            + Integer::from(2) * sm.eshort.a2.clone() * x0.clone()
            + sm.eshort.a4.clone();
        let ap = Integer::from(-2) * a.clone();
        let bp = a.clone() * a.clone() - Integer::from(4) * b.clone();
        let side = rank::alpha_side(&a, &b, rank::DEFAULT_SEARCH_BOUND);
        let side_p = rank::alpha_side(&ap, &bp, rank::DEFAULT_SEARCH_BOUND);
        let dims = rank::dim2(side.selmer.len()) + rank::dim2(side_p.selmer.len());
        assert!(dims >= 2, "Selmer product below 4: bug");
        SelmerGroup {
            isogeny_model: (a, b),
            phi_classes: side.selmer,
            phi_prime_classes: side_p.selmer,
            rank_upper_bound: dims - 2,
        }
    }

    /// Naive bounded search for rational points with x = m/n,
    /// |m| ≤ `height_bound`, 1 ≤ n ≤ `height_bound`, on the **full
    /// generalized Weierstrass** model (the previous version silently
    /// assumed a₁ = a₂ = a₃ = 0 and missed points on general models).
    /// Includes the point at infinity.
    pub fn find_rational_points(&self, height_bound: i64) -> Vec<Point> {
        let mut points = vec![Point::infinity()];
        let e = self.curve;
        let q = Rational::from_integer;
        let (a1, a2, a3, a4, a6) = (
            q(e.a1.clone()),
            q(e.a2.clone()),
            q(e.a3.clone()),
            q(e.a4.clone()),
            q(e.a6.clone()),
        );
        let two = Rational::from_i64(2);
        for x_num in -height_bound..=height_bound {
            for x_den in 1..=height_bound {
                if gcd_i64(x_num, x_den) != 1 {
                    continue;
                }
                let x = Rational::new(Integer::from(x_num), Integer::from(x_den))
                    .expect("x_den >= 1 is nonzero");
                // y² + (a₁x + a₃)y − (x³ + a₂x² + a₄x + a₆) = 0
                let s = a1.clone() * x.clone() + a3.clone();
                let f = x.clone() * x.clone() * x.clone()
                    + a2.clone() * x.clone() * x.clone()
                    + a4.clone() * x.clone()
                    + a6.clone();
                let disc = s.clone() * s.clone() + Rational::from_i64(4) * f;
                if let Some(root) = rank::rational_sqrt(&disc) {
                    for y in [
                        (-s.clone() + root.clone()) / two.clone(),
                        (-s.clone() - root.clone()) / two.clone(),
                    ] {
                        let p = Point::new(x.clone(), y);
                        debug_assert!(e.is_on_curve(&p));
                        if !points.contains(&p) {
                            points.push(p);
                        }
                    }
                }
            }
        }
        points
    }
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values below were derived with the independent Python
    // reference implementation of the descent before being asserted.

    #[test]
    fn test_selmer_group() {
        // y² = x³ − x: first 2-torsion root is x = −1 → model (a, b) = (−3, 2);
        // Sel_φ = {1, 2}, Sel_φ̂ = {1, −1}, upper bound 0 (the curve has
        // rank 0).
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        let descent = TwoDescent::new(&curve);
        let selmer = descent.compute_selmer_group();

        assert_eq!(selmer.isogeny_model, (Integer::from(-3), Integer::from(2)));
        assert_eq!(selmer.phi_classes, vec![Integer::from(1), Integer::from(2)]);
        assert_eq!(
            selmer.phi_prime_classes,
            vec![Integer::from(1), Integer::from(-1)]
        );
        assert_eq!(selmer.rank_upper_bound, 0);
    }

    #[test]
    fn test_rank_bound() {
        // Real now (was an `unimplemented!` facade): y² = x³ − x has
        // certified rank upper bound 0 (and rank exactly 0).
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        let descent = TwoDescent::new(&curve);
        assert_eq!(descent.rank_bound(), 0);
    }

    #[test]
    #[should_panic(expected = "2-division polynomial")]
    fn test_rank_bound_refuses_without_two_torsion() {
        // y² = x³ − x + 1 has an irreducible 2-division polynomial: the
        // honest behavior is refusal, not a fabricated bound.
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(1));
        let descent = TwoDescent::new(&curve);
        let _ = descent.rank_bound();
    }

    #[test]
    fn test_find_rational_points() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        let descent = TwoDescent::new(&curve);
        let points = descent.find_rational_points(10);

        // O and the three 2-torsion points (−1, 0), (0, 0), (1, 0).
        assert!(points.iter().any(|p| p.infinity));
        for x in [-1i64, 0, 1] {
            let p = Point::from_integers(x, 0);
            assert!(points.contains(&p), "missing 2-torsion point ({}, 0)", x);
        }
    }

    #[test]
    fn test_find_rational_points_general_model() {
        // 11a1: y² + y = x³ − x² − 10x − 20 (a₃ = 1): the search must use
        // the full Weierstrass equation; (5, 5) and (5, −6) are 5-torsion.
        let e = EllipticCurve::new(
            Integer::from(0),
            Integer::from(-1),
            Integer::from(1),
            Integer::from(-10),
            Integer::from(-20),
        );
        let descent = TwoDescent::new(&e);
        let points = descent.find_rational_points(16);
        assert!(points.contains(&Point::from_integers(5, 5)));
        assert!(points.contains(&Point::from_integers(5, -6)));
        assert!(points.contains(&Point::from_integers(16, 60)));
    }
}
