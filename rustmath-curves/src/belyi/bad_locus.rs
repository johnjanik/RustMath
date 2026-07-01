//! The bad locus `Z_C` predicate — a conservative, exact classifier.
//!
//! Ported from `dessin_engine/src/bad_locus.rs` in
//! `/home/john/inverse_galois/M23/dessin_engine` (S4, D4′ completeness). The
//! reference implementation's private `Rat` is replaced by
//! [`rustmath_rationals::Rational`].
//!
//! A rational point on the source conic realizes `M23/Q` only **off** `Z_C` —
//! branch, cusp, singular, and monodromy-drop points. This predicate operates on
//! the factorized genus-0 forms `P=A²B, Q=R⁵S, P−Q=c·U¹²`: it flags branch/pinned
//! points and degree-drops, and otherwise returns [`BadLocusStatus::MonodromyDropUnknown`]
//! — it **never** emits [`BadLocusStatus::Clear`]. Affirmative clearance must be
//! certified elsewhere (monodromy of the residual specialization).

use rustmath_core::Ring;
use rustmath_rationals::Rational;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BadLocusStatus {
    /// Off every detectable component of `Z_C`. This predicate alone never emits
    /// `Clear`; only an independent monodromy check may upgrade a point.
    Clear,
    BranchPoint,
    DegreeDrop,
    /// Not on a detectable component, but monodromy-drop is invisible locally —
    /// defer to the monodromy certifier.
    MonodromyDropUnknown,
}

/// Projective point `[Y:Z]` on the normalized source `P^1` (split-conic chart).
#[derive(Debug, Clone)]
pub struct P1PointQ {
    pub y: Rational,
    pub z: Rational,
}

/// A binary homogeneous form over `Q`, `coeff[i]` on `Y^i Z^{d-i}`.
#[derive(Debug, Clone)]
pub struct BinaryFormQ {
    pub coeff: Vec<Rational>,
}

impl BinaryFormQ {
    pub fn degree(&self) -> usize {
        self.coeff.len().saturating_sub(1)
    }

    pub fn eval(&self, p: &P1PointQ) -> Rational {
        let d = self.degree();
        let mut acc = Rational::from_i64(0);
        for (i, a) in self.coeff.iter().enumerate() {
            let term = a.clone() * Ring::pow(&p.y, i as u32) * Ring::pow(&p.z, (d - i) as u32);
            acc = acc + term;
        }
        acc
    }
}

/// Factorized genus-0 Belyi data in the split chart.
#[derive(Debug, Clone)]
pub struct GenusZeroBelyiFactorizationQ {
    pub a: BinaryFormQ,
    pub b: BinaryFormQ,
    pub r: BinaryFormQ,
    pub s: BinaryFormQ,
    pub u: BinaryFormQ,
    pub c: Rational,
}

impl GenusZeroBelyiFactorizationQ {
    /// `p` lies over one of the three branch values `0,1,∞` (a ramification or
    /// pinned point).
    pub fn is_branch_or_pinned_point(&self, p: &P1PointQ) -> bool {
        self.a.eval(p).is_zero()
            || self.b.eval(p).is_zero()
            || self.r.eval(p).is_zero()
            || self.s.eval(p).is_zero()
            || self.u.eval(p).is_zero()
    }

    /// Degree drop: `P` and `Q` both vanish (the map is locally indeterminate).
    pub fn degree_drop_at(&self, p: &P1PointQ) -> bool {
        let p_zero = self.a.eval(p).is_zero() || self.b.eval(p).is_zero();
        let q_zero = self.r.eval(p).is_zero() || self.s.eval(p).is_zero();
        p_zero && q_zero
    }

    /// Conservative `Z_C` classification. Never emits `Clear`; a clean point is
    /// `MonodromyDropUnknown`, to be settled by the monodromy certifier.
    pub fn classify(&self, p: &P1PointQ) -> BadLocusStatus {
        if self.degree_drop_at(p) {
            return BadLocusStatus::DegreeDrop;
        }
        if self.is_branch_or_pinned_point(p) {
            return BadLocusStatus::BranchPoint;
        }
        BadLocusStatus::MonodromyDropUnknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ri(n: i64) -> Rational {
        Rational::from_i64(n)
    }
    fn lin_root_at(root_y_over_z: i64) -> BinaryFormQ {
        // Y - root·Z : coeff[0] = -root (on Z), coeff[1] = 1 (on Y)
        BinaryFormQ {
            coeff: vec![ri(-root_y_over_z), ri(1)],
        }
    }

    #[test]
    fn branch_point_is_not_clear() {
        // A = Y (root at [0:1]); the point [0:1] is a branch/pinned point.
        let f = GenusZeroBelyiFactorizationQ {
            a: BinaryFormQ { coeff: vec![ri(0), ri(1)] }, // Y
            b: lin_root_at(2),
            r: BinaryFormQ { coeff: vec![ri(1), ri(0)] }, // Z
            s: lin_root_at(3),
            u: lin_root_at(1),
            c: ri(-1),
        };
        let at_zero = P1PointQ { y: ri(0), z: ri(1) };
        assert_eq!(f.classify(&at_zero), BadLocusStatus::BranchPoint);

        // a generic point hits no detectable component ⇒ deferred
        let generic = P1PointQ { y: ri(5), z: ri(1) };
        assert_eq!(f.classify(&generic), BadLocusStatus::MonodromyDropUnknown);
    }

    #[test]
    fn degree_drop_detected() {
        // A = Y (P=0 at [0:1]) and R = Y (Q=0 at [0:1]) ⇒ both vanish ⇒ drop.
        let f = GenusZeroBelyiFactorizationQ {
            a: BinaryFormQ { coeff: vec![ri(0), ri(1)] }, // Y
            b: lin_root_at(2),
            r: BinaryFormQ { coeff: vec![ri(0), ri(1)] }, // Y
            s: lin_root_at(3),
            u: lin_root_at(1),
            c: ri(-1),
        };
        let at_zero = P1PointQ { y: ri(0), z: ri(1) };
        assert_eq!(f.classify(&at_zero), BadLocusStatus::DegreeDrop);
    }

    #[test]
    fn classify_never_emits_clear() {
        // The predicate must never, on any point, return Clear.
        let f = GenusZeroBelyiFactorizationQ {
            a: BinaryFormQ { coeff: vec![ri(0), ri(1)] },
            b: lin_root_at(2),
            r: BinaryFormQ { coeff: vec![ri(1), ri(0)] },
            s: lin_root_at(3),
            u: lin_root_at(1),
            c: ri(-1),
        };
        for k in -5..=5 {
            let p = P1PointQ { y: ri(k), z: ri(1) };
            assert_ne!(f.classify(&p), BadLocusStatus::Clear);
        }
    }
}
