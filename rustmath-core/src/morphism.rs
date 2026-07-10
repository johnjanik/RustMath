//! Object-safe morphisms and an erased/boxed helper layer.
//!
//! MAGMA source: Handbook §17.3 (ring maps / coercion), and the Hom/End,
//! chain-map and scheme-morphism machinery throughout the handbook.
//!
//! The core [`Ring`](crate::Ring) trait is **not** `dyn`-safe (it requires
//! `PartialEq` and by-value arithmetic operators), so anything that needs to
//! store or compose maps polymorphically — homomorphism sets, endomorphism
//! rings, chain maps, coercions — cannot use `Box<dyn Ring>`. This module
//! provides an object-safe [`Morphism`] trait plus a boxed/erased helper so that
//! downstream code can build those layers without fighting the trait tower.
//!
//! Purely additive.

use std::marker::PhantomData;

/// An object-safe map from `Domain` to `Codomain`.
///
/// Unlike the algebraic-structure traits, this trait *is* object-safe: it has no
/// generic methods, takes `&self`, and returns owned values. `dyn Morphism<...>`
/// is therefore a valid type, which is exactly what the erased helpers below use.
pub trait Morphism {
    /// The source type.
    type Domain;
    /// The target type.
    type Codomain;

    /// Apply the morphism to an element of the domain.
    fn apply(&self, x: &Self::Domain) -> Self::Codomain;
}

/// A boxed, type-erased morphism between fixed domain and codomain types.
pub type BoxedMorphism<D, C> = Box<dyn Morphism<Domain = D, Codomain = C>>;

/// Box up any morphism into a [`BoxedMorphism`].
pub fn boxed<M>(m: M) -> BoxedMorphism<M::Domain, M::Codomain>
where
    M: Morphism + 'static,
{
    Box::new(m)
}

impl<D, C> Morphism for BoxedMorphism<D, C> {
    type Domain = D;
    type Codomain = C;

    fn apply(&self, x: &D) -> C {
        (**self).apply(x)
    }
}

/// The identity morphism on a `Clone` type.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityMorphism<T>(PhantomData<T>);

impl<T> IdentityMorphism<T> {
    /// Construct the identity morphism.
    pub fn new() -> Self {
        IdentityMorphism(PhantomData)
    }
}

impl<T: Clone> Morphism for IdentityMorphism<T> {
    type Domain = T;
    type Codomain = T;

    fn apply(&self, x: &T) -> T {
        x.clone()
    }
}

/// A morphism backed by an arbitrary closure `Fn(&D) -> C`.
///
/// This is the workhorse adapter: any coercion or ad-hoc map can be lifted into
/// the [`Morphism`] world (and then boxed) via [`FnMorphism::new`].
pub struct FnMorphism<D, C, F: Fn(&D) -> C> {
    f: F,
    _pd: PhantomData<(D, C)>,
}

impl<D, C, F: Fn(&D) -> C> FnMorphism<D, C, F> {
    /// Wrap a closure as a morphism.
    pub fn new(f: F) -> Self {
        FnMorphism {
            f,
            _pd: PhantomData,
        }
    }
}

impl<D, C, F: Fn(&D) -> C> Morphism for FnMorphism<D, C, F> {
    type Domain = D;
    type Codomain = C;

    fn apply(&self, x: &D) -> C {
        (self.f)(x)
    }
}

/// The composition `second ∘ first` of two morphisms.
///
/// Well-typed exactly when `first.Codomain == second.Domain`.
pub struct Composed<M1, M2> {
    first: M1,
    second: M2,
}

impl<M1, M2> Morphism for Composed<M1, M2>
where
    M1: Morphism,
    M2: Morphism<Domain = M1::Codomain>,
{
    type Domain = M1::Domain;
    type Codomain = M2::Codomain;

    fn apply(&self, x: &Self::Domain) -> Self::Codomain {
        self.second.apply(&self.first.apply(x))
    }
}

/// Compose two morphisms into `second ∘ first`.
pub fn compose<M1, M2>(first: M1, second: M2) -> Composed<M1, M2>
where
    M1: Morphism,
    M2: Morphism<Domain = M1::Codomain>,
{
    Composed { first, second }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_morphism_and_boxing() {
        let double = FnMorphism::new(|x: &i64| x * 2);
        assert_eq!(double.apply(&21), 42);

        let erased: BoxedMorphism<i64, i64> = boxed(double);
        assert_eq!(erased.apply(&21), 42);

        // Object safety: a homogeneous collection of erased maps.
        let maps: Vec<BoxedMorphism<i64, i64>> = vec![
            boxed(FnMorphism::new(|x: &i64| x + 1)),
            boxed(FnMorphism::new(|x: &i64| x * 10)),
        ];
        let out: Vec<i64> = maps.iter().map(|m| m.apply(&5)).collect();
        assert_eq!(out, vec![6, 50]);
    }

    #[test]
    fn test_identity_and_compose() {
        let id = IdentityMorphism::<i64>::new();
        assert_eq!(id.apply(&7), 7);

        // (x -> x+1) then (x -> x*3): 5 -> 6 -> 18
        let f = FnMorphism::new(|x: &i64| x + 1);
        let g = FnMorphism::new(|x: &i64| x * 3);
        let gf = compose(f, g);
        assert_eq!(gf.apply(&5), 18);
    }

    #[test]
    fn test_cross_type_compose() {
        // i64 -> String -> usize
        let show = FnMorphism::new(|x: &i64| format!("{x:+}"));
        let len = FnMorphism::new(|s: &String| s.len());
        let m = compose(show, len);
        assert_eq!(m.apply(&42), 3); // "+42" has length 3
    }
}
