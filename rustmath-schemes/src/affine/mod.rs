//! Affine Schemes
//!
//! This module provides comprehensive support for affine schemes in algebraic geometry.
//!
//! # Overview
//!
//! An affine scheme is a scheme of the form Spec(R) for some commutative ring R.
//! The spectrum Spec(R) is the set of all prime ideals of R, equipped with the
//! Zariski topology and a structure sheaf.
//!
//! Key concepts:
//! - **Spec Construction**: Building Spec(R) from a ring R
//! - **Prime Ideals**: Points correspond to prime ideals
//! - **Zariski Topology**: Closed sets are V(I) = {p ∈ Spec(R) | I ⊆ p}
//! - **Structure Sheaf**: Regular functions on open sets
//! - **Distinguished Opens**: D(f) = {p ∈ Spec(R) | f ∉ p}
//!
//! # Examples
//!
//! ## Creating Affine Schemes
//!
//! ```rust
//! use rustmath_schemes::affine::AffineScheme;
//! use rustmath_integers::Integer;
//!
//! // Spec(ℤ) - the prime spectrum of the integers
//! // let spec_z = AffineScheme::spec_integers();
//! // assert!(spec_z.is_affine());
//! // assert_eq!(spec_z.dimension(), Some(1));
//! ```
//!
//! ## Affine Space
//!
//! ```rust
//! use rustmath_schemes::affine::{AffineSpace, AffinePoint};
//!
//! // Create 𝔸² (affine plane)
//! // let a2: AffineSpace<i32> = AffineSpace::new(2);
//! // assert_eq!(a2.dimension(), Some(2));
//!
//! // Create a point (1, 2) in 𝔸²
//! // let point = AffinePoint::new(vec![1, 2]);
//! // assert!(a2.contains_point(&point));
//! ```
//!
//! ## Closed Subschemes
//!
//! ```rust
//! use rustmath_schemes::affine::{AffineScheme, ClosedSubscheme};
//! use rustmath_polynomials::Polynomial;
//!
//! // Define V(x² + y² - 1) ⊆ 𝔸² (a circle)
//! // let a2 = AffineSpace::new(2);
//! // let circle = ClosedSubscheme::from_ideal(a2, ideal);
//! // assert_eq!(circle.dimension(), Some(1)); // 1-dimensional curve
//! ```

use rustmath_core::{Field, MathError, Result, Ring};
use crate::generic::{Scheme, SchemeMorphism, SchemePoint, DimensionTheory, Separated};
use crate::singularity;
use rustmath_polynomials::elimination::krull_dimension;
use rustmath_polynomials::groebner::GroebnerBudget;
use rustmath_polynomials::multivariate::MultivariatePolynomial;
use std::fmt;
use std::marker::PhantomData;

/// Affine scheme Spec(R)
///
/// Represents the prime spectrum of a commutative ring R. The points are
/// prime ideals of R, with the Zariski topology and structure sheaf.
///
/// # Type Parameters
///
/// - `R`: The coordinate ring
#[derive(Debug, Clone)]
pub struct AffineScheme<R: Ring> {
    /// The base ring R
    coordinate_ring: R,
    /// Number of polynomial variables adjoined to R.
    ///
    /// The scheme is Spec(R[x₀, …, x_{num_vars−1}]); `num_vars == 0` is plain Spec(R).
    /// Variables are indexed `0..num_vars`, matching the `usize` variable indices of
    /// [`MultivariatePolynomial`].
    num_vars: usize,
    /// Krull dimension, when it is actually known.
    dimension: Option<usize>,
    /// Whether Spec(R[x]) is irreducible — when that has been *established*.
    ///
    /// `None` means "not known", not "no". See [`AffineScheme::known_irreducible`].
    irreducible: Option<bool>,
    /// Whether Spec(R[x]) is reduced — when that has been *established*. `None` = unknown.
    reduced: Option<bool>,
    /// Whether R[x] is Noetherian — when that has been *established*. `None` = unknown.
    noetherian: Option<bool>,
}

impl<R: Ring> AffineScheme<R> {
    /// Create Spec(R) for a given ring R
    pub fn new(ring: R) -> Self {
        AffineScheme {
            coordinate_ring: ring,
            num_vars: 0,
            dimension: None,
            irreducible: None,
            reduced: None,
            noetherian: None,
        }
    }

    /// Is this scheme known to be irreducible? `None` = not established.
    ///
    /// Irreducibility of Spec(R[x₀,…,x_{n−1}]) is equivalent to R[x] having a unique
    /// minimal prime, which for a polynomial ring comes down to R having one. The
    /// [`Ring`] trait exposes nothing about R beyond `+ − × 0 1` — not commutativity, not
    /// domain-ness, not even reducedness — so for a general `R` there is nothing to
    /// compute from and the honest answer is "unknown".
    ///
    /// It *is* established when the scheme was built over a field
    /// ([`AffineScheme::affine_space_over_field`]): `k` is a domain, `R` a domain implies
    /// `R[x]` a domain (the leading coefficients of a product multiply), so by induction
    /// `k[x₀,…,x_{n−1}]` is a domain and its Spec is irreducible.
    pub fn known_irreducible(&self) -> Option<bool> {
        self.irreducible
    }

    /// Is this scheme known to be reduced? `None` = not established.
    ///
    /// Same situation as [`AffineScheme::known_irreducible`]: `Spec(R[x])` is reduced iff
    /// `R` is reduced, and nothing in [`Ring`] says whether `R` has nilpotents. It is
    /// emphatically not safe to guess: `𝔸ⁿ` over `ℤ/4` is *not* reduced.
    pub fn known_reduced(&self) -> Option<bool> {
        self.reduced
    }

    /// Is the coordinate ring known to be Noetherian? `None` = not established.
    ///
    /// By Hilbert's basis theorem, `R` Noetherian implies `R[x₀,…,x_{n−1}]` Noetherian.
    /// But "R is Noetherian" is not something the [`Ring`] trait tells us, and it is not
    /// true of every ring one could plug in, so the old `true` here ("assume the ring is
    /// Noetherian — true for most rings we use") was an unfounded assumption dressed up as
    /// a fact. Over a field it is established: a field is Noetherian (its only ideals are
    /// `(0)` and `(1)`), hence so is `k[x₀,…,x_{n−1}]`.
    pub fn known_noetherian(&self) -> Option<bool> {
        self.noetherian
    }

    /// Get the base ring
    pub fn coordinate_ring(&self) -> &R {
        &self.coordinate_ring
    }

    /// The number of polynomial variables adjoined: this is Spec(R[x₀,…,x_{n−1}]).
    pub fn num_variables(&self) -> usize {
        self.num_vars
    }

    /// Create affine n-space over R, i.e. Spec(R[x₀, …, x_{n−1}]).
    ///
    /// The dimension is left unknown: `dim R[x₁,…,xₙ] = dim R + n` for Noetherian `R`, and
    /// nothing in the [`Ring`] trait exposes `dim R` (it is 0 for a field but 1 for ℤ). Use
    /// [`AffineScheme::affine_space_over_field`] when the base is a field, where the answer
    /// is exactly `n`.
    pub fn affine_space(n: usize, base_ring: R) -> Self {
        AffineScheme {
            coordinate_ring: base_ring,
            num_vars: n,
            dimension: None,
            irreducible: None,
            reduced: None,
            noetherian: None,
        }
    }
}

impl<R: Field> AffineScheme<R> {
    /// Affine n-space 𝔸ⁿ over a field k: Spec(k[x₀, …, x_{n−1}]), of Krull dimension `n`.
    ///
    /// This is the one case where everything is known without computing anything:
    ///
    /// - **dimension `n`**: `dim k = 0`, and `dim k[x₁,…,xₙ] = dim k + n` for a Noetherian
    ///   base;
    /// - **irreducible** and **reduced**: `k` is an integral domain, and `R` a domain
    ///   implies `R[x]` a domain, so `k[x₀,…,x_{n−1}]` is a domain — its Spec is integral;
    /// - **Noetherian**: a field is Noetherian, so `k[x₀,…,x_{n−1}]` is too, by Hilbert's
    ///   basis theorem.
    ///
    /// `n = 0` gives Spec(k), the one-point scheme.
    pub fn affine_space_over_field(n: usize, base_field: R) -> Self {
        AffineScheme {
            coordinate_ring: base_field,
            num_vars: n,
            dimension: Some(n),
            irreducible: Some(true),
            reduced: Some(true),
            noetherian: Some(true),
        }
    }
}

impl<R: Ring> Scheme for AffineScheme<R> {
    type BaseRing = R;

    fn base_ring(&self) -> &Self::BaseRing {
        &self.coordinate_ring
    }

    fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    fn is_affine(&self) -> bool {
        true
    }

    /// # Panics
    ///
    /// Panics unless irreducibility has been *established* for this scheme — i.e. unless it
    /// was built by [`AffineScheme::affine_space_over_field`]. See
    /// [`AffineScheme::known_irreducible`] for a non-panicking `Option<bool>` and for why
    /// a general `R: Ring` admits no answer: the trait exposes nothing about R, and
    /// guessing is not free — `false` would claim that 𝔸ⁿ over a field is *reducible*,
    /// which is flatly wrong.
    fn is_irreducible(&self) -> bool {
        self.irreducible.unwrap_or_else(|| {
            unimplemented!(
                "AffineScheme::is_irreducible: Spec(R[x₀..x_{{{}}}]) is irreducible iff R has a \
                 unique minimal prime, and the Ring trait exposes nothing about R (not \
                 commutativity, not domain-ness). Build the scheme with \
                 AffineScheme::affine_space_over_field to get the answer for a field, or call \
                 known_irreducible() for an Option<bool> instead of a fabricated bool",
                self.num_vars.saturating_sub(1)
            )
        })
    }

    /// # Panics
    ///
    /// Panics unless reducedness has been established (see
    /// [`AffineScheme::known_reduced`]). 𝔸ⁿ over ℤ/4 is not reduced, so no default is
    /// sound.
    fn is_reduced(&self) -> bool {
        self.reduced.unwrap_or_else(|| {
            unimplemented!(
                "AffineScheme::is_reduced: Spec(R[x]) is reduced iff R has no nilpotents, and \
                 the Ring trait says nothing about the nilradical of R (𝔸ⁿ over ℤ/4 is NOT \
                 reduced, so `true` is not a safe default, and `false` would deny that 𝔸ⁿ over \
                 a field is reduced). Build with AffineScheme::affine_space_over_field, or call \
                 known_reduced() for an Option<bool>"
            )
        })
    }

    /// # Panics
    ///
    /// Panics unless Noetherianity has been established (see
    /// [`AffineScheme::known_noetherian`]). Hilbert's basis theorem needs `R` Noetherian as
    /// a *hypothesis*; the old code simply assumed it.
    fn is_noetherian(&self) -> bool {
        self.noetherian.unwrap_or_else(|| {
            unimplemented!(
                "AffineScheme::is_noetherian: R[x₀,…] is Noetherian iff R is (Hilbert's basis \
                 theorem), and the Ring trait does not tell us whether R is. Build with \
                 AffineScheme::affine_space_over_field, or call known_noetherian() for an \
                 Option<bool>"
            )
        })
    }

    /// Sound for every `R`: Spec(R[x₀,…,x_{n−1}]) is Spec of a finitely generated
    /// R-algebra (generated by the `n` variables), which is what "of finite type over the
    /// base" means. No assumption about R is needed.
    fn is_finite_type(&self) -> bool {
        true
    }
}

impl<R: Ring> Separated for AffineScheme<R> {
    fn is_separated(&self) -> bool {
        // All affine schemes are separated
        true
    }
}

/// Affine space 𝔸ⁿ
///
/// Represents n-dimensional affine space over a ring R.
/// This is the scheme Spec(R[x₁, ..., xₙ]).
#[derive(Debug, Clone)]
pub struct AffineSpace<R: Ring> {
    /// Dimension of the space
    dimension: usize,
    /// Base ring
    base_ring: R,
}

impl<R: Ring> AffineSpace<R> {
    /// Create n-dimensional affine space
    pub fn new(n: usize, base_ring: R) -> Self {
        AffineSpace {
            dimension: n,
            base_ring,
        }
    }

    /// Get the dimension
    pub fn dim(&self) -> usize {
        self.dimension
    }

    /// Convert to the underlying affine scheme
    pub fn as_scheme(&self) -> AffineScheme<R> {
        AffineScheme::affine_space(self.dimension, self.base_ring.clone())
    }
}

impl<R: Ring> Scheme for AffineSpace<R> {
    type BaseRing = R;

    fn base_ring(&self) -> &Self::BaseRing {
        &self.base_ring
    }

    fn dimension(&self) -> Option<usize> {
        Some(self.dimension)
    }

    fn is_affine(&self) -> bool {
        true
    }

    fn is_irreducible(&self) -> bool {
        true // Affine space is irreducible
    }

    fn is_reduced(&self) -> bool {
        true // Affine space is reduced
    }

    fn is_noetherian(&self) -> bool {
        true
    }

    fn is_finite_type(&self) -> bool {
        true
    }
}

/// A point in affine space
///
/// Represents a point in 𝔸ⁿ with coordinates in the base ring.
#[derive(Debug, Clone, PartialEq)]
pub struct AffinePoint<R: Ring> {
    /// Coordinates of the point
    coordinates: Vec<R>,
}

impl<R: Ring> AffinePoint<R> {
    /// Create a new affine point from coordinates
    pub fn new(coordinates: Vec<R>) -> Result<Self> {
        if coordinates.is_empty() {
            return Err(MathError::InvalidArgument(
                "Point must have at least one coordinate".to_string(),
            ));
        }
        Ok(AffinePoint { coordinates })
    }

    /// Get the coordinates
    pub fn coordinates(&self) -> &[R] {
        &self.coordinates
    }

    /// Get the dimension (number of coordinates)
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }
}

impl<R: Ring> SchemePoint for AffinePoint<R> {
    type Parent = AffineSpace<R>;

    fn parent(&self) -> &Self::Parent {
        // This is a simplified implementation
        // In practice, we'd store a reference to the parent
        unimplemented!("AffinePoint::parent requires lifetime management")
    }

    fn is_closed(&self) -> bool {
        // In affine space over an algebraically closed field, all points are closed
        true
    }

    // Commented out: Ring is not dyn compatible
    // fn residue_field(&self) -> Result<Box<dyn Ring>> {
    //     // The residue field at a closed point over k is k itself
    //     unimplemented!("Residue field computation requires field theory")
    // }
}

/// Morphism between affine schemes
///
/// A morphism Spec(S) → Spec(R) is induced by a ring homomorphism R → S.
#[derive(Debug, Clone)]
pub struct AffineSchemeMorphism<R: Ring, S: Ring> {
    source: AffineScheme<S>,
    target: AffineScheme<R>,
    // In a full implementation, this would store the ring homomorphism
    _phantom: PhantomData<(R, S)>,
}

impl<R: Ring, S: Ring> AffineSchemeMorphism<R, S> {
    /// Create a new morphism from a ring homomorphism
    ///
    /// A ring homomorphism φ: R → S induces a morphism Spec(S) → Spec(R)
    pub fn new(source: AffineScheme<S>, target: AffineScheme<R>) -> Self {
        AffineSchemeMorphism {
            source,
            target,
            _phantom: PhantomData,
        }
    }
}

impl<R: Ring, S: Ring> SchemeMorphism for AffineSchemeMorphism<R, S> {
    type Source = AffineScheme<S>;
    type Target = AffineScheme<R>;

    fn source(&self) -> &Self::Source {
        &self.source
    }

    fn target(&self) -> &Self::Target {
        &self.target
    }

    fn is_proper(&self) -> bool {
        false // Most affine morphisms are not proper
    }

    fn is_finite(&self) -> bool {
        // A morphism of affine schemes is finite iff the ring map makes S
        // a finitely generated R-module
        false
    }

    fn is_finite_type(&self) -> bool {
        true // Most morphisms we construct are of finite type
    }

    fn is_closed_embedding(&self) -> bool {
        false
    }

    fn is_open_embedding(&self) -> bool {
        false
    }
}

/// Closed subscheme V(I) ⊆ 𝔸ⁿ_R = Spec(R[x₀,…,x_{n−1}]), i.e. Spec(R[x]/I).
///
/// It genuinely carries its defining ideal: `I` is the ideal generated by [`Self::ideal`].
/// Nothing here guesses — every question that needs a Gröbner basis returns a `Result` and
/// reports an honest error when the budget trips.
#[derive(Debug, Clone)]
pub struct ClosedSubscheme<R: Ring> {
    ambient: AffineScheme<R>,
    /// Generators of the defining ideal I ⊆ R[x₀,…,x_{n−1}].
    ideal: Vec<MultivariatePolynomial<R>>,
}

impl<R: Ring> ClosedSubscheme<R> {
    /// Create the closed subscheme V(I) of `ambient`, where I is generated by `ideal`.
    ///
    /// Errors if a generator mentions a variable outside the ambient's variable range —
    /// that would silently place the subscheme in a bigger space than advertised.
    pub fn new(ambient: AffineScheme<R>, ideal: Vec<MultivariatePolynomial<R>>) -> Result<Self> {
        let n = ambient.num_variables();
        for g in &ideal {
            if let Some(v) = g.max_variable() {
                if v >= n {
                    return Err(MathError::InvalidArgument(format!(
                        "generator {} uses variable x{} but the ambient is 𝔸^{}",
                        g, v, n
                    )));
                }
            }
        }
        Ok(ClosedSubscheme {
            ambient,
            ideal: ideal.into_iter().filter(|g| !g.is_zero()).collect(),
        })
    }

    /// Get the ambient scheme
    pub fn ambient(&self) -> &AffineScheme<R> {
        &self.ambient
    }

    /// Generators of the defining ideal I.
    pub fn ideal(&self) -> &[MultivariatePolynomial<R>] {
        &self.ideal
    }

    /// The dimension `n` of the ambient affine space 𝔸ⁿ.
    pub fn ambient_dimension(&self) -> usize {
        self.ambient.num_variables()
    }
}

impl<R: Field> ClosedSubscheme<R> {
    /// The Krull dimension of the coordinate ring k[x₀,…,x_{n−1}]/I.
    ///
    /// This is the honest dimension of the subscheme, computed from a Gröbner basis (see
    /// [`krull_dimension`]) — not a cached guess. The empty subscheme (I = (1)) has
    /// dimension `-1`, which is why this returns `isize` rather than `usize`.
    pub fn dimension(&self) -> std::result::Result<isize, String> {
        self.dimension_with_budget(&GroebnerBudget::default())
    }

    /// [`Self::dimension`] with an explicit Gröbner budget.
    pub fn dimension_with_budget(
        &self,
        budget: &GroebnerBudget,
    ) -> std::result::Result<isize, String> {
        krull_dimension(self.ideal.clone(), self.ambient.num_variables(), budget)
    }

    /// Is V(I) empty over the algebraic closure? Equivalently, is `1 ∈ I`?
    ///
    /// See [`singularity::is_unit_ideal`] for exactly what this decides and why the answer
    /// computed over k is also the answer over k̄.
    pub fn is_empty(&self) -> std::result::Result<bool, String> {
        singularity::is_unit_ideal(self.ideal.clone(), &GroebnerBudget::default())
    }
}

/// Distinguished open subset D(f) ⊆ Spec(R[x]) — the locus where `f` does not vanish.
///
/// This is naturally the affine scheme Spec(R[x]_f).
#[derive(Debug, Clone)]
pub struct DistinguishedOpen<R: Ring> {
    ambient: AffineScheme<R>,
    /// The element f being inverted.
    f: MultivariatePolynomial<R>,
}

impl<R: Ring> DistinguishedOpen<R> {
    /// Create the distinguished open D(f).
    pub fn new(ambient: AffineScheme<R>, f: MultivariatePolynomial<R>) -> Result<Self> {
        let n = ambient.num_variables();
        if let Some(v) = f.max_variable() {
            if v >= n {
                return Err(MathError::InvalidArgument(format!(
                    "f = {} uses variable x{} but the ambient is 𝔸^{}",
                    f, v, n
                )));
            }
        }
        Ok(DistinguishedOpen { ambient, f })
    }

    /// Get the ambient scheme
    pub fn ambient(&self) -> &AffineScheme<R> {
        &self.ambient
    }

    /// The element f that is inverted on this open set.
    pub fn f(&self) -> &MultivariatePolynomial<R> {
        &self.f
    }
}

impl<R: Ring> Scheme for DistinguishedOpen<R> {
    type BaseRing = R;

    fn base_ring(&self) -> &Self::BaseRing {
        self.ambient.base_ring()
    }

    fn dimension(&self) -> Option<usize> {
        self.ambient.dimension()
    }

    fn is_affine(&self) -> bool {
        true // Distinguished opens of affine schemes are affine
    }

    /// D(f) ⊆ X is a dense open of X, so it is irreducible exactly when X is (for f ≠ 0).
    ///
    /// # Panics
    ///
    /// Inherits the honest refusal of [`AffineScheme::is_irreducible`] when the ambient
    /// scheme's irreducibility has not been established.
    fn is_irreducible(&self) -> bool {
        self.ambient.is_irreducible()
    }

    /// An open subscheme of a reduced scheme is reduced (localisation of a reduced ring is
    /// reduced).
    ///
    /// # Panics
    ///
    /// Inherits the honest refusal of [`AffineScheme::is_reduced`].
    fn is_reduced(&self) -> bool {
        self.ambient.is_reduced()
    }

    /// A localisation of a Noetherian ring is Noetherian.
    ///
    /// # Panics
    ///
    /// Inherits the honest refusal of [`AffineScheme::is_noetherian`].
    fn is_noetherian(&self) -> bool {
        self.ambient.is_noetherian()
    }

    fn is_finite_type(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_affine_scheme_creation() {
        // Test basic construction (would use concrete rings in practice)
        // let ring = /* some ring */;
        // let spec = AffineScheme::new(ring);
        // assert!(spec.is_affine());
    }

    #[test]
    fn test_affine_space() {
        // Would test with concrete rings
        // let a2 = AffineSpace::new(2, base_ring);
        // assert_eq!(a2.dim(), 2);
        // assert!(a2.is_affine());
    }

    #[test]
    fn test_affine_point() {
        // Would test with concrete coordinates
        // let point = AffinePoint::new(vec![1, 2]);
        // assert_eq!(point.dimension(), 2);
    }

    /// 𝔸² over ℚ: everything is established, and it is established *because* the base is a
    /// field — k[x,y] is a domain (so irreducible + reduced) and Noetherian (Hilbert).
    #[test]
    fn affine_space_over_a_field_knows_it_is_integral() {
        let a2: AffineScheme<Rational> = AffineScheme::affine_space_over_field(2, Rational::zero());

        assert_eq!(a2.known_irreducible(), Some(true));
        assert_eq!(a2.known_reduced(), Some(true));
        assert_eq!(a2.known_noetherian(), Some(true));

        assert!(a2.is_irreducible());
        assert!(a2.is_reduced());
        assert!(a2.is_integral());
        assert!(a2.is_noetherian());
        assert_eq!(a2.dimension(), Some(2));

        // Spec(k) itself: the one-point scheme.
        let pt: AffineScheme<Rational> = AffineScheme::affine_space_over_field(0, Rational::zero());
        assert_eq!(pt.dimension(), Some(0));
        assert!(pt.is_integral());
    }

    /// Over an unspecified `R: Ring` nothing is established — and the honest report of that
    /// is `None`, not a fabricated `false`.
    ///
    /// The old code answered `is_irreducible() = false` ("a conservative placeholder"),
    /// which is not conservative at all: it asserts that 𝔸ⁿ is reducible, which is wrong
    /// over every field. And `is_noetherian() = true` ("assume the ring is Noetherian")
    /// was an assumption, not a computation.
    #[test]
    fn spec_of_an_unknown_ring_establishes_nothing() {
        let x: AffineScheme<Rational> = AffineScheme::affine_space(2, Rational::zero());

        assert_eq!(x.known_irreducible(), None);
        assert_eq!(x.known_reduced(), None);
        assert_eq!(x.known_noetherian(), None);

        // What IS sound without knowing anything about R:
        assert!(x.is_affine());
        assert!(x.is_finite_type());
    }

    #[test]
    #[should_panic(expected = "known_irreducible")]
    fn is_irreducible_refuses_rather_than_guessing() {
        let x: AffineScheme<Rational> = AffineScheme::affine_space(2, Rational::zero());
        let _ = x.is_irreducible();
    }

    #[test]
    #[should_panic(expected = "known_noetherian")]
    fn is_noetherian_refuses_rather_than_assuming() {
        let x: AffineScheme<Rational> = AffineScheme::affine_space(2, Rational::zero());
        let _ = x.is_noetherian();
    }
}
