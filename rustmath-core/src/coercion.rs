//! Coercion and pushouts on top of the [`Parent`] layer.
//!
//! MAGMA source: Handbook §17.3 (`R ! a` coercion, automatic coercion into a
//! common overstructure).
//!
//! Coercion is the act of moving an element of one parent into another parent
//! (`R ! a` in MAGMA syntax). Two parents that both embed into a common
//! overstructure have a *pushout*; mixed-parent arithmetic first coerces both
//! operands into that pushout. This module gives the minimal, extensible
//! vocabulary for both, deliberately routing through [`Parent`]/[`Element`]
//! rather than ad-hoc `From` impls.
//!
//! Purely additive.

use crate::parent::Parent;

/// A target parent `Self` into which elements of `Source` can be coerced.
///
/// This models MAGMA's `R ! a`: given `a` in `Source`, produce the corresponding
/// element of `Self` (the ring `R`). Coercion may fail (e.g. reducing a rational
/// that is not `p`-integral into `Z_(p)`), hence the [`Option`] return.
pub trait Coercible<Source: Parent>: Parent {
    /// Coerce `element` (a member of `source`) into `self`, if possible.
    fn coerce(&self, source: &Source, element: &Source::Element) -> Option<Self::Element>;

    /// Whether there is a canonical coercion `source -> self`.
    fn has_coercion_from(&self, source: &Source) -> bool {
        let _ = source;
        true
    }
}

/// A pushout (common overstructure) of two parents.
///
/// The associated [`Common`](Pushout::Common) parent is one that both `Self` and
/// `Other` coerce into. Implementors typically pick the "larger" of the two
/// structures (e.g. `pushout(Z, Q) = Q`).
pub trait Pushout<Other: Parent>: Parent {
    /// The common parent both operands embed into.
    type Common: Parent;

    /// Construct the common overstructure of `self` and `other`, if one exists.
    fn pushout(&self, other: &Other) -> Option<Self::Common>;
}

/// Coerce `element` of `source` into `target`, a thin free-function wrapper over
/// [`Coercible::coerce`] for ergonomic call sites.
pub fn coerce_into<Target, Source>(
    target: &Target,
    source: &Source,
    element: &Source::Element,
) -> Option<Target::Element>
where
    Source: Parent,
    Target: Coercible<Source>,
{
    target.coerce(source, element)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Toy parents: the integers and the "rationals" (represented as pairs).
    #[derive(Debug, Clone, PartialEq)]
    struct Zed;
    #[derive(Debug, Clone, PartialEq)]
    struct Que;

    impl Parent for Zed {
        type Element = i64;
        fn contains(&self, _e: &i64) -> bool {
            true
        }
    }
    impl Parent for Que {
        type Element = (i64, i64); // (numerator, denominator)
        fn contains(&self, e: &(i64, i64)) -> bool {
            e.1 != 0
        }
    }

    // Q can coerce Z (n |-> n/1).
    impl Coercible<Zed> for Que {
        fn coerce(&self, _s: &Zed, e: &i64) -> Option<(i64, i64)> {
            Some((*e, 1))
        }
    }

    // Z coerces from itself trivially.
    impl Coercible<Zed> for Zed {
        fn coerce(&self, _s: &Zed, e: &i64) -> Option<i64> {
            Some(*e)
        }
    }

    // The pushout of Z and Q is Q.
    impl Pushout<Que> for Zed {
        type Common = Que;
        fn pushout(&self, _other: &Que) -> Option<Que> {
            Some(Que)
        }
    }

    #[test]
    fn test_coerce_z_into_q() {
        let q = Que;
        let z = Zed;
        assert_eq!(coerce_into(&q, &z, &7), Some((7, 1)));
        assert!(q.has_coercion_from(&z));
    }

    #[test]
    fn test_pushout_z_q_is_q() {
        let z = Zed;
        let q = Que;
        let common = z.pushout(&q).unwrap();
        // In the pushout Q, the integer 3 becomes 3/1.
        assert_eq!(common.coerce(&z, &3), Some((3, 1)));
    }
}
