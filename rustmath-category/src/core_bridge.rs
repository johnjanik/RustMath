//! Bridge from the category-level coercion graph to `rustmath-core`'s
//! `Parent`/`Coercible`/`Pushout` layer.
//!
//! # Decision record (P2-G, "wire category into core coercion")
//!
//! The Phase-2 plan item asked whether `rustmath-category`'s coercion system
//! could drive `rustmath_core::coercion` **additively and object-safely**. The
//! answer is **yes**, with this module as the wiring, because all three
//! constraints hold in the category → core direction:
//!
//! 1. **No cyclic dependency.** The bridge lives *here*, in `rustmath-category`,
//!    which already depends on `rustmath-core`. Core itself gains no knowledge
//!    of (and no dependency on) this crate; its coercion module stays a pure
//!    trait vocabulary.
//! 2. **Object safety.** `rustmath_core::Ring` is not dyn-safe, but the bridge
//!    never needs `dyn Ring`: pushout resolution is statically dispatched via
//!    [`Pushout`]/[`Coercible`], and where type erasure is genuinely needed
//!    (storing many coercions homogeneously) we lower into core's *object-safe*
//!    [`rustmath_core::morphism::Morphism`] layer via [`coercion_as_morphism`].
//! 3. **Purely additive.** Nothing in core or in the pre-existing
//!    [`crate::coercion`] module changes; this module only consumes them.
//!
//! # What the bridge does
//!
//! The category-side [`CoercionMap`] is a *runtime graph* of coercion edges
//! keyed by `std::any::type_name`. Core's [`Coercible`]/[`Pushout`] impls are
//! *compile-time witnesses* that a coercion/pushout actually exists and can
//! produce elements. The bridge connects the two:
//!
//! - [`register_parent_coercion`] records a statically-witnessed
//!   `Coercible<Source>` impl as an edge `Source -> Target` in the graph
//!   (keyed by the **parent** types, matching MAGMA's view of §17.3 coercion
//!   as a relation between structures, not element types).
//! - [`graph_pushout`] resolves the common overstructure of two parents *only
//!   if* the graph contains the edges into `Pushout::Common` (identity edges
//!   are implicit). The graph genuinely drives the decision: an unregistered
//!   edge means `None` even though the static impl exists.
//! - [`coerce_pair_via_graph`] is the MAGMA mixed-arithmetic step: coerce both
//!   operands into the graph-approved pushout parent.
//! - [`coercion_as_morphism`] erases a parent-level coercion into core's
//!   boxed object-safe morphism layer.
//!
//! The canonical example (exercised in `tests/core_coercion_bridge.rs` with
//! the real `rustmath_integers::Integer` / `rustmath_rationals::Rational`
//! types): `pushout(Z, Q) = Q`, with `3 ∈ Z` landing as `3/1 ∈ Q`.
//!
//! # Caveat
//!
//! Like [`CoercionMap`] itself, edge identity is `std::any::type_name`-based;
//! two distinct parents of the same Rust type (e.g. `Z/5` and `Z/7` as one
//! `Zmod(n)` struct) share edges. The graph models coercibility of *types of
//! parents*; value-level failure is still reported by [`Coercible::coerce`]
//! returning `None`.

use crate::coercion::CoercionMap;
use rustmath_core::coercion::{Coercible, Pushout};
use rustmath_core::morphism::{boxed, BoxedMorphism, FnMorphism};
use rustmath_core::parent::Parent;

/// Record the statically-witnessed coercion `Source -> Target` (a core
/// [`Coercible`] impl) as an edge in the runtime coercion graph `map`.
///
/// The edge is keyed by the *parent* types `Source` and `Target`.
pub fn register_parent_coercion<Source, Target>(map: &mut CoercionMap)
where
    Source: Parent + 'static,
    Target: Coercible<Source> + 'static,
{
    map.register::<Source, Target>(format!(
        "core::Coercible parent coercion: {} -> {}",
        std::any::type_name::<Source>(),
        std::any::type_name::<Target>(),
    ));
}

/// Whether the graph knows the edge `S -> T`: either the trivial identity
/// edge (`S` and `T` are the same parent type) or a registered coercion.
fn graph_has_edge<S: 'static, T: 'static>(map: &CoercionMap) -> bool {
    std::any::type_name::<S>() == std::any::type_name::<T>() || map.has_coercion::<S, T>()
}

/// Resolve the pushout (common overstructure) of `a` and `b`, *driven by the
/// coercion graph*: the statically-known [`Pushout`] impl is consulted only
/// if `map` contains edges `A -> A::Common` and `B -> A::Common` (identity
/// edges are implicit).
///
/// Returns `None` if either edge is missing from the graph or if the static
/// [`Pushout::pushout`] itself declines.
pub fn graph_pushout<A, B>(map: &CoercionMap, a: &A, b: &B) -> Option<A::Common>
where
    A: Pushout<B> + 'static,
    B: Parent + 'static,
    A::Common: Parent + 'static,
{
    if graph_has_edge::<A, A::Common>(map) && graph_has_edge::<B, A::Common>(map) {
        a.pushout(b)
    } else {
        None
    }
}

/// Coerce a mixed pair of elements into the graph-approved pushout parent.
///
/// This is the element-level step behind MAGMA's automatic coercion for
/// mixed-parent arithmetic (§17.3): find the common overstructure via
/// [`graph_pushout`], then move both operands into it. Returns the common
/// parent together with both coerced elements.
#[allow(clippy::type_complexity)]
pub fn coerce_pair_via_graph<A, B>(
    map: &CoercionMap,
    a: &A,
    b: &B,
    x: &A::Element,
    y: &B::Element,
) -> Option<(
    A::Common,
    <A::Common as Parent>::Element,
    <A::Common as Parent>::Element,
)>
where
    A: Pushout<B> + 'static,
    B: Parent + 'static,
    A::Common: Coercible<A> + Coercible<B> + 'static,
{
    let common = graph_pushout(map, a, b)?;
    let cx = common.coerce(a, x)?;
    let cy = common.coerce(b, y)?;
    Some((common, cx, cy))
}

/// Erase a parent-level coercion into core's object-safe morphism layer.
///
/// `rustmath_core::Ring` is not dyn-safe, so coercions cannot be stored as
/// `Box<dyn Ring>`-flavoured objects; core's [`rustmath_core::morphism`]
/// module exists precisely as the object-safe alternative. This adapter turns
/// a `Target: Coercible<Source>` witness into a boxed
/// `Morphism<Domain = Source::Element, Codomain = Option<Target::Element>>`
/// (partiality of coercion is kept in the codomain).
pub fn coercion_as_morphism<Source, Target>(
    target: Target,
    source: Source,
) -> BoxedMorphism<Source::Element, Option<Target::Element>>
where
    Source: Parent + 'static,
    Target: Coercible<Source> + 'static,
{
    boxed(FnMorphism::new(move |x: &Source::Element| {
        target.coerce(&source, x)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::morphism::Morphism as _;

    // Toy parents mirroring core's own coercion tests: Z-like and Q-like.
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

    impl Coercible<Zed> for Que {
        fn coerce(&self, _s: &Zed, e: &i64) -> Option<(i64, i64)> {
            Some((*e, 1))
        }
    }
    impl Coercible<Que> for Que {
        fn coerce(&self, _s: &Que, e: &(i64, i64)) -> Option<(i64, i64)> {
            Some(*e)
        }
    }

    impl Pushout<Que> for Zed {
        type Common = Que;
        fn pushout(&self, _other: &Que) -> Option<Que> {
            Some(Que)
        }
    }

    #[test]
    fn graph_pushout_requires_registered_edge() {
        let empty = CoercionMap::new();
        // The static Pushout impl exists, but the graph has no Zed -> Que
        // edge, so the graph-driven resolution must decline.
        assert!(graph_pushout(&empty, &Zed, &Que).is_none());

        let mut map = CoercionMap::new();
        register_parent_coercion::<Zed, Que>(&mut map);
        // Zed -> Que registered; Que -> Que is the implicit identity edge.
        assert_eq!(graph_pushout(&map, &Zed, &Que), Some(Que));
    }

    #[test]
    fn coerce_pair_lands_in_common_parent() {
        let mut map = CoercionMap::new();
        register_parent_coercion::<Zed, Que>(&mut map);

        let (common, cx, cy) =
            coerce_pair_via_graph(&map, &Zed, &Que, &3, &(1, 2)).expect("pushout resolves");
        assert_eq!(common, Que);
        assert_eq!(cx, (3, 1));
        assert_eq!(cy, (1, 2));
        assert!(common.contains(&cx));
    }

    #[test]
    fn coercion_erases_into_core_morphism_layer() {
        let m = coercion_as_morphism(Que, Zed);
        assert_eq!(m.apply(&7), Some((7, 1)));

        // Object safety: homogeneous storage of erased coercions.
        let maps: Vec<BoxedMorphism<i64, Option<(i64, i64)>>> =
            vec![coercion_as_morphism(Que, Zed), coercion_as_morphism(Que, Zed)];
        assert!(maps.iter().all(|m| m.apply(&1) == Some((1, 1))));
    }
}
