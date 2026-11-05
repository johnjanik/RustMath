//! Global minimal models and Weierstrass isomorphisms for E/Q.
//!
//! A Weierstrass isomorphism over Q between two models E and E' is given by
//! a quadruple (u, r, s, t) with u ∈ Q^×, r, s, t ∈ Q acting by
//!
//! ```text
//! x = u² x' + r,        y = u³ y' + s u² x' + t,
//! ```
//!
//! under which the a-invariants transform as (Silverman, AEC III Table 3.1):
//!
//! ```text
//! u  a1' = a1 + 2s
//! u² a2' = a2 − s·a1 + 3r − s²
//! u³ a3' = a3 + r·a1 + 2t
//! u⁴ a4' = a4 − s·a3 + 2r·a2 − (t + rs)·a1 + 3r² − 2st
//! u⁶ a6' = a6 + r·a4 + r²·a2 + r³ − t·a3 − t² − rt·a1
//! ```
//!
//! and c4' = c4/u⁴, c6' = c6/u⁶, Δ' = Δ/u¹².
//!
//! The global minimal model is computed by the Laska–Kraus–Connell method:
//! the scale factor u = ∏ p^{(v_p(Δ) − v_p(Δ_min))/12} is read off from
//! Tate's algorithm (which is already Cremona-verified in [`crate::tate`]),
//! and the reduced minimal model is reconstructed from (c4/u⁴, c6/u⁶) with
//! the standard normalization a1, a3 ∈ {0, 1}, a2 ∈ {−1, 0, 1}.
//!
//! Every step is self-certifying: the reconstructed model is checked to have
//! exactly the predicted (c4, c6, Δ), and the solved isomorphism is checked
//! against all five a-invariant transformation equations. The reconstruction
//! formulas were additionally verified against exact Python reference
//! computations (scaled models of 11a1, 37a1, 14a1, 389a1, … recover their
//! reduced minimal models).

use crate::curve::{EllipticCurve, Point};
use crate::tate::tate_local_data;
use rustmath_core::Ring;
use rustmath_integers::prime::factor;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

impl EllipticCurve {
    /// The b-invariants (b2, b4, b6, b8) of this Weierstrass model.
    pub fn b_invariants(&self) -> (Integer, Integer, Integer, Integer) {
        let b2 = &self.a1 * &self.a1 + Integer::from(4) * self.a2.clone();
        let b4 = Integer::from(2) * self.a4.clone() + &self.a1 * &self.a3;
        let b6 = &self.a3 * &self.a3 + Integer::from(4) * self.a6.clone();
        let b8 = self.a1.clone() * self.a1.clone() * self.a6.clone()
            + Integer::from(4) * self.a2.clone() * self.a6.clone()
            - self.a1.clone() * self.a3.clone() * self.a4.clone()
            + self.a2.clone() * self.a3.clone() * self.a3.clone()
            - self.a4.clone() * self.a4.clone();
        (b2, b4, b6, b8)
    }

    /// The c-invariants (c4, c6) of this Weierstrass model.
    pub fn c_invariants(&self) -> (Integer, Integer) {
        let (b2, b4, b6, _) = self.b_invariants();
        let c4 = &b2 * &b2 - Integer::from(24) * b4.clone();
        let c6 = -(b2.clone() * b2.clone() * b2.clone()) + Integer::from(36) * b2 * b4
            - Integer::from(216) * b6;
        (c4, c6)
    }

    /// The globally minimal reduced model of this curve together with the
    /// isomorphism `self → minimal` (Laska–Kraus–Connell; the p-minimality
    /// data comes from Tate's algorithm).
    ///
    /// The returned model is the standard reduced minimal model with
    /// a1, a3 ∈ {0, 1} and a2 ∈ {−1, 0, 1} (the LMFDB/Cremona normal form).
    ///
    /// # Panics
    ///
    /// Panics if the curve is singular.
    pub fn minimal_model(&self) -> (EllipticCurve, WeierstrassIsomorphism) {
        minimal_model(self)
    }
}

/// A Weierstrass isomorphism (u, r, s, t) over Q; see the module docs for
/// the coordinate action and the a-invariant transformation laws.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeierstrassIsomorphism {
    pub u: Rational,
    pub r: Rational,
    pub s: Rational,
    pub t: Rational,
}

impl WeierstrassIsomorphism {
    /// The identity isomorphism.
    pub fn identity() -> Self {
        Self {
            u: Rational::one(),
            r: Rational::zero(),
            s: Rational::zero(),
            t: Rational::zero(),
        }
    }

    /// Map a point of E to the corresponding point of E':
    /// x' = (x − r)/u², y' = (y − s(x − r) − t)/u³.
    pub fn map_point(&self, p: &Point) -> Point {
        if p.infinity {
            return Point::infinity();
        }
        let u2 = self.u.clone() * self.u.clone();
        let u3 = u2.clone() * self.u.clone();
        let dx = p.x.clone() - self.r.clone();
        let xp = dx.clone() / u2;
        let yp = (p.y.clone() - self.s.clone() * dx - self.t.clone()) / u3;
        Point::new(xp, yp)
    }

    /// The inverse isomorphism E' → E.
    pub fn inverse(&self) -> Self {
        let u2 = self.u.clone() * self.u.clone();
        let u3 = u2.clone() * self.u.clone();
        Self {
            u: Rational::one() / self.u.clone(),
            r: -(self.r.clone() / u2.clone()),
            s: -(self.s.clone() / self.u.clone()),
            t: (self.r.clone() * self.s.clone() - self.t.clone()) / u3,
        }
    }

    /// Composition: if `self : E → E'` and `then : E' → E''`, the result maps
    /// E → E''.
    pub fn compose(&self, then: &Self) -> Self {
        let (u1, r1, s1, t1) = (&self.u, &self.r, &self.s, &self.t);
        let (u2, r2, s2, t2) = (&then.u, &then.r, &then.s, &then.t);
        let u1sq = u1.clone() * u1.clone();
        let u1cu = u1sq.clone() * u1.clone();
        Self {
            u: u1.clone() * u2.clone(),
            r: u1sq.clone() * r2.clone() + r1.clone(),
            s: u1.clone() * s2.clone() + s1.clone(),
            t: u1cu * t2.clone() + s1.clone() * u1sq * r2.clone() + t1.clone(),
        }
    }

    /// Check that this quadruple really is an isomorphism from `e` to `ep`,
    /// i.e. that all five a-invariant transformation equations hold exactly.
    pub fn is_isomorphism(&self, e: &EllipticCurve, ep: &EllipticCurve) -> bool {
        let q = Rational::from_integer;
        let (a1, a2, a3, a4, a6) = (
            q(e.a1.clone()),
            q(e.a2.clone()),
            q(e.a3.clone()),
            q(e.a4.clone()),
            q(e.a6.clone()),
        );
        let (b1, b2n, b3, b4n, b6n) = (
            q(ep.a1.clone()),
            q(ep.a2.clone()),
            q(ep.a3.clone()),
            q(ep.a4.clone()),
            q(ep.a6.clone()),
        );
        let (u, r, s, t) = (&self.u, &self.r, &self.s, &self.t);
        let two = Rational::from_i64(2);
        let three = Rational::from_i64(3);
        let u2 = u.clone() * u.clone();
        let u3 = u2.clone() * u.clone();
        let u4 = u2.clone() * u2.clone();
        let u6 = u3.clone() * u3.clone();

        let eq1 = u.clone() * b1 == a1.clone() + two.clone() * s.clone();
        let eq2 = u2 * b2n
            == a2.clone() - s.clone() * a1.clone() + three.clone() * r.clone()
                - s.clone() * s.clone();
        let eq3 = u3 * b3 == a3.clone() + r.clone() * a1.clone() + two.clone() * t.clone();
        let eq4 = u4 * b4n
            == a4.clone() - s.clone() * a3.clone() + two.clone() * r.clone() * a2.clone()
                - (t.clone() + r.clone() * s.clone()) * a1.clone()
                + three * r.clone() * r.clone()
                - two.clone() * s.clone() * t.clone();
        let eq6 = u6 * b6n
            == a6.clone()
                + r.clone() * a4
                + r.clone() * r.clone() * a2
                + r.clone() * r.clone() * r.clone()
                - t.clone() * a3
                - t.clone() * t.clone()
                - r.clone() * t.clone() * a1;
        eq1 && eq2 && eq3 && eq4 && eq6
    }
}

/// Exact division helper: n / d, panicking if it does not divide.
fn exact_div(n: &Integer, d: &Integer) -> Integer {
    let q = n / d;
    assert!((&q * d) == *n, "exact_div: {} does not divide {}", d, n);
    q
}

/// Reconstruct the reduced Weierstrass model with the given c-invariants
/// (Laska–Kraus–Connell / Connell's "c4c6 model"). Panics if (c4, c6) do not
/// arise from an integral model — for inputs produced by `minimal_model`
/// this cannot happen, because a minimal model with these invariants exists.
fn model_from_c4c6(c4: &Integer, c6: &Integer) -> EllipticCurve {
    let twelve = Integer::from(12);
    // b2 ≡ −c6 (mod 12), normalized to (−6, 6].
    let mut b2 = (-c6.clone()).modulo(&twelve);
    if b2 > Integer::from(6) {
        b2 = b2 - twelve;
    }
    let b4 = exact_div(&(&b2 * &b2 - c4.clone()), &Integer::from(24));
    let b6 = exact_div(
        &(-(b2.clone() * b2.clone() * b2.clone()) + Integer::from(36) * b2.clone() * b4.clone()
            - c6.clone()),
        &Integer::from(216),
    );
    let two = Integer::from(2);
    let four = Integer::from(4);
    let a1 = b2.modulo(&two);
    let a2 = exact_div(&(&b2 - &a1), &four);
    let a3 = b6.modulo(&two);
    let a6 = exact_div(&(&b6 - &a3), &four);
    let a4 = exact_div(&(&b4 - &(&a1 * &a3)), &two);
    let e = EllipticCurve::new(a1, a2, a3, a4, a6);
    let (cc4, cc6) = e.c_invariants();
    assert!(
        cc4 == *c4 && cc6 == *c6,
        "model_from_c4c6: reconstruction mismatch"
    );
    e
}

/// Compute the globally minimal reduced model of `e` and the isomorphism
/// `e → minimal`. See [`EllipticCurve::minimal_model`].
pub fn minimal_model(e: &EllipticCurve) -> (EllipticCurve, WeierstrassIsomorphism) {
    assert!(
        !e.is_singular(),
        "minimal_model: curve is singular (discriminant 0)"
    );
    let (c4, c6) = e.c_invariants();

    // Scale factor u = prod p^{(v_p(Δ) − v_p(Δ_min))/12} from Tate's algorithm.
    // Only primes with v_p(Δ) >= 12 can contribute.
    let mut u = Integer::one();
    for (p, vp) in factor(&e.discriminant.abs()) {
        if vp >= 12 {
            let ld = tate_local_data(e, &p);
            let diff = vp - ld.minimal_disc_valuation;
            assert!(
                diff.is_multiple_of(12),
                "minimal_model: v_p(Δ) − v_p(Δ_min) = {} not divisible by 12 at p = {}",
                diff,
                p
            );
            u = u * p.pow(diff / 12);
        }
    }

    let u4 = u.pow(4);
    let u6 = u.pow(6);
    let c4m = exact_div(&c4, &u4);
    let c6m = exact_div(&c6, &u6);
    let emin = model_from_c4c6(&c4m, &c6m);
    assert!(
        emin.discriminant.clone() * u.pow(12) == e.discriminant,
        "minimal_model: discriminant mismatch"
    );

    // Solve the isomorphism e → emin (scale factor u, then r, s, t from the
    // first three transformation equations) and certify it on all five.
    let uq = Rational::from_integer(u);
    let q = Rational::from_integer;
    let two = Rational::from_i64(2);
    let three = Rational::from_i64(3);
    let s = (uq.clone() * q(emin.a1.clone()) - q(e.a1.clone())) / two.clone();
    let r = (uq.clone() * uq.clone() * q(emin.a2.clone()) - q(e.a2.clone())
        + s.clone() * q(e.a1.clone())
        + s.clone() * s.clone())
        / three;
    let t = (uq.clone() * uq.clone() * uq.clone() * q(emin.a3.clone())
        - q(e.a3.clone())
        - r.clone() * q(e.a1.clone()))
        / two;
    let iso = WeierstrassIsomorphism { u: uq, r, s, t };
    assert!(
        iso.is_isomorphism(e, &emin),
        "minimal_model: solved transformation failed certification"
    );
    (emin, iso)
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

    /// u-scaled (non-minimal) copy of a model: a_i ↦ a_i λ^i, Δ ↦ Δ λ^12.
    fn scaled(e: &EllipticCurve, lam: i64) -> EllipticCurve {
        let l = Integer::from(lam);
        EllipticCurve::new(
            e.a1.clone() * l.clone(),
            e.a2.clone() * l.pow(2),
            e.a3.clone() * l.pow(3),
            e.a4.clone() * l.pow(4),
            e.a6.clone() * l.pow(6),
        )
    }

    #[test]
    fn minimal_model_of_minimal_reduced_curve_is_itself() {
        // Already-reduced minimal models must be returned unchanged with the
        // identity isomorphism. (Reduced minimal models per LMFDB/Cremona;
        // minimality re-verified exactly by the Python reference run.)
        for e in [
            curve(0, -1, 1, -10, -20), // 11a1
            curve(0, 0, 1, -1, 0),     // 37a1
            curve(1, 0, 1, 4, -6),     // 14a1
            curve(0, 1, 1, -2, 0),     // 389a1
            curve(0, 0, 0, -1, 0),     // y² = x³ − x
        ] {
            let (emin, iso) = e.minimal_model();
            assert_eq!(emin, e);
            assert_eq!(iso, WeierstrassIsomorphism::identity());
        }
    }

    #[test]
    fn minimal_model_recovers_from_scaling() {
        // Scaling a minimal model by λ = 2, 3, 6 produces a non-minimal
        // integral model; minimal_model must recover the reduced original.
        // (Verified against the exact Python reference: LKC reconstruction
        // on u-scaled models of these curves returns the reduced model.)
        for e in [
            curve(0, -1, 1, -10, -20),
            curve(0, 0, 1, -1, 0),
            curve(1, 0, 1, 4, -6),
        ] {
            for lam in [2i64, 3, 6] {
                let es = scaled(&e, lam);
                let (emin, iso) = es.minimal_model();
                assert_eq!(emin, e, "λ = {}", lam);
                assert!(iso.is_isomorphism(&es, &emin));
                // Point mapping round-trip: iso ∘ iso⁻¹ = id on a sample point.
                let p = Point::new(
                    Rational::from_i64(5 * lam * lam),
                    Rational::from_i64(-6 * lam * lam * lam),
                );
                let back = iso.inverse().map_point(&iso.map_point(&p));
                assert_eq!(back, p);
            }
        }
    }

    #[test]
    fn minimal_model_of_unreduced_minimal_curve() {
        // (5, -6, -18, 0, 0) (torsion Z/10 test curve) is minimal but not
        // reduced; Python reference gives reduced form (1, 0, 0, -45, 81).
        let e = curve(5, -6, -18, 0, 0);
        let (emin, iso) = e.minimal_model();
        assert_eq!(emin, curve(1, 0, 0, -45, 81));
        assert!(iso.is_isomorphism(&e, &emin));
        // The order-10 torsion point (0,0) must map to a point on emin.
        let p = Point::new(Rational::zero(), Rational::zero());
        let q = iso.map_point(&p);
        assert!(emin.is_on_curve(&q));
    }

    #[test]
    fn compose_and_inverse_are_consistent() {
        let e = curve(0, -1, 1, -10, -20);
        let es = scaled(&e, 6);
        let (emin, iso) = es.minimal_model();
        assert_eq!(emin, e);
        let idlike = iso.compose(&iso.inverse());
        assert_eq!(idlike, WeierstrassIsomorphism::identity());
        assert!(iso.inverse().is_isomorphism(&emin, &es));
    }
}
