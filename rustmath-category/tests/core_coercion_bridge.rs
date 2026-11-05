//! Integration test for the category → core coercion bridge (P2-G).
//!
//! Exercises `rustmath_category::core_bridge` against the *real*
//! `rustmath_integers::Integer` and `rustmath_rationals::Rational` types:
//! the category-level coercion graph drives core's `Pushout`/`Coercible`
//! resolution, and the canonical pushout of the Integer and Rational parents
//! resolves to the Rational parent (`pushout(Z, Q) = Q`).

use rustmath_category::coercion::CoercionMap;
use rustmath_category::core_bridge::{
    coerce_pair_via_graph, coercion_as_morphism, graph_pushout, register_parent_coercion,
};
use rustmath_core::coercion::{Coercible, Pushout};
use rustmath_core::morphism::Morphism as _;
use rustmath_core::parent::Parent;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// The ring of integers Z as a core `Parent` over the real `Integer` type.
#[derive(Debug, Clone, PartialEq)]
struct IntegerRing;

/// The field of rationals Q as a core `Parent` over the real `Rational` type.
#[derive(Debug, Clone, PartialEq)]
struct RationalField;

impl Parent for IntegerRing {
    type Element = Integer;

    fn contains(&self, _e: &Integer) -> bool {
        true
    }

    fn zero(&self) -> Option<Integer> {
        Some(Integer::from(0))
    }

    fn one(&self) -> Option<Integer> {
        Some(Integer::from(1))
    }
}

impl Parent for RationalField {
    type Element = Rational;

    fn contains(&self, _e: &Rational) -> bool {
        true
    }

    fn zero(&self) -> Option<Rational> {
        Some(Rational::from_integer(0))
    }

    fn one(&self) -> Option<Rational> {
        Some(Rational::from_integer(1))
    }
}

// The canonical embedding Z -> Q, n |-> n/1.
impl Coercible<IntegerRing> for RationalField {
    fn coerce(&self, _source: &IntegerRing, element: &Integer) -> Option<Rational> {
        Some(Rational::from_integer(element.clone()))
    }
}

// Q coerces from itself (identity).
impl Coercible<RationalField> for RationalField {
    fn coerce(&self, _source: &RationalField, element: &Rational) -> Option<Rational> {
        Some(element.clone())
    }
}

// The pushout of Z and Q is Q.
impl Pushout<RationalField> for IntegerRing {
    type Common = RationalField;

    fn pushout(&self, _other: &RationalField) -> Option<RationalField> {
        Some(RationalField)
    }
}

fn graph_with_z_to_q() -> CoercionMap {
    let mut map = CoercionMap::new();
    register_parent_coercion::<IntegerRing, RationalField>(&mut map);
    map
}

#[test]
fn pushout_of_integer_and_rational_parents_is_rational() {
    let map = graph_with_z_to_q();
    let common = graph_pushout(&map, &IntegerRing, &RationalField);
    assert_eq!(common, Some(RationalField));
}

#[test]
fn pushout_declines_without_graph_edge() {
    // The static Pushout impl exists, but the coercion graph is what drives
    // resolution: with no registered Z -> Q edge, the pushout must decline.
    let empty = CoercionMap::new();
    assert_eq!(graph_pushout(&empty, &IntegerRing, &RationalField), None);
}

#[test]
fn integer_lands_in_rational_pushout_as_n_over_1() {
    let map = graph_with_z_to_q();

    let x = Integer::from(3);
    let y = Rational::new(1, 2).unwrap();

    let (common, cx, cy) =
        coerce_pair_via_graph(&map, &IntegerRing, &RationalField, &x, &y).expect("Z, Q -> Q");

    assert_eq!(common, RationalField);
    // 3 in Z becomes 3/1 in Q.
    assert_eq!(cx, Rational::from_integer(3));
    assert_eq!(cx.numerator(), &Integer::from(3));
    assert_eq!(cx.denominator(), &Integer::from(1));
    // 1/2 passes through the identity coercion unchanged.
    assert_eq!(cy, Rational::new(1, 2).unwrap());
    assert!(common.contains(&cx) && common.contains(&cy));

    // Mixed arithmetic in the pushout: 3 + 1/2 = 7/2.
    assert_eq!(cx + cy, Rational::new(7, 2).unwrap());
}

#[test]
fn embedding_erases_into_core_object_safe_morphism() {
    // The Z -> Q embedding as a boxed object-safe core morphism (the layer
    // that exists because `Ring` itself is not dyn-safe).
    let embed = coercion_as_morphism(RationalField, IntegerRing);
    assert_eq!(embed.apply(&Integer::from(42)), Some(Rational::from_integer(42)));
}
