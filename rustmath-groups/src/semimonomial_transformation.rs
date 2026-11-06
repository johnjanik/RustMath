//! Semimonomial Transformations
//!
//! This module implements semimonomial transformations, which are elements of a group
//! that combines:
//! - A vector φ in (R×)^n (units of the ring)
//! - A permutation π in S_n
//! - A ring automorphism α
//!
//! # Mathematical Structure
//!
//! The semimonomial transformation group over a ring R is the semidirect product
//! of the monomial group and the automorphism group of R.
//!
//! # Multiplication
//!
//! For (φ, π, α) and (ψ, σ, β):
//! ```text
//! (φ, π, α)(ψ, σ, β) = (φ · ψ^(π,α), πσ, α ∘ β)
//! ```
//!
//! where ψ^(π,α)_i = α(ψ_{π(i)}) and the permutation product πσ applies π
//! first: (πσ)(i) = σ(π(i)), as in GAP and Sage. A transformation acts on
//! vectors from the left by (φ, π, α)·v = (φ_i⁻¹ α(v_{π(i)}))_i.
//!
//! Automorphisms are identified by a `usize` id; the caller supplies the maps
//! (apply α, multiply, invert, compose ids) to the free functions below.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::semimonomial_transformation::{
//!     action_on_vector, compose, SemimonomialTransformation,
//! };
//! use rustmath_combinatorics::Permutation;
//!
//! // Over Z the units are ±1 and Aut(Z) is trivial (automorphism id 0).
//! let id_aut = |x: &i64| *x;
//! let mul = |a: &i64, b: &i64| a * b;
//! let inv = |u: &i64| *u; // ±1 is its own inverse
//!
//! // g = ([1, -1, 1], (0 1), id): a signed permutation of Z^3
//! let perm = Permutation::from_vec(vec![1, 0, 2]).unwrap();
//! let g = SemimonomialTransformation::new(vec![1i64, -1, 1], perm, 0);
//! assert_eq!(g.degree(), 3);
//!
//! // (g·v)_i = φ_i⁻¹ v_{π(i)}
//! let v = vec![10i64, 20, 30];
//! let gv = action_on_vector(&g, &v, id_aut, mul, inv);
//! assert_eq!(gv, vec![20, -10, 30]);
//!
//! // g² = ([-1, -1, 1], identity, id), and it acts as g·(g·v)
//! let g2 = compose(&g, &g, id_aut, mul, |_, _| 0);
//! assert_eq!(g2.units(), &[-1, -1, 1]);
//! assert_eq!(g2.permutation(), &Permutation::identity(3));
//! assert_eq!(
//!     action_on_vector(&g2, &v, id_aut, mul, inv),
//!     action_on_vector(&g, &gv, id_aut, mul, inv)
//! );
//! ```

use rustmath_combinatorics::Permutation;
use std::fmt;

/// A ring automorphism represented as a function
///
/// For simplicity, we represent automorphisms as boxed functions.
/// In practice, this could be extended to support symbolic automorphisms.
pub type Automorphism<T> = Box<dyn Fn(&T) -> T>;

/// A semimonomial transformation over a ring
///
/// Consists of:
/// - A vector of ring units (invertible elements)
/// - A permutation
/// - A ring automorphism
#[derive(Clone)]
pub struct SemimonomialTransformation<T: Clone> {
    /// Vector component φ in (R×)^n
    units: Vec<T>,
    /// Permutation component π in S_n
    permutation: Permutation,
    /// Automorphism component α (stored as function pointer hash for cloning)
    /// Note: For practical use, this should be replaced with a symbolic representation
    automorphism_id: usize,
}

impl<T: Clone> SemimonomialTransformation<T> {
    /// Create a new semimonomial transformation
    ///
    /// # Arguments
    /// * `units` - Vector of ring units (must be invertible)
    /// * `permutation` - The permutation component
    /// * `automorphism_id` - ID identifying the automorphism
    ///
    /// # Panics
    /// Panics if the length of units doesn't match the permutation degree
    pub fn new(units: Vec<T>, permutation: Permutation, automorphism_id: usize) -> Self {
        assert_eq!(
            units.len(),
            permutation.size(),
            "Units vector length must match permutation size"
        );

        Self {
            units,
            permutation,
            automorphism_id,
        }
    }

    /// Create the identity transformation
    ///
    /// # Arguments
    /// * `degree` - The degree n
    /// * `one` - The multiplicative identity of the ring
    pub fn identity(degree: usize, one: T) -> Self {
        Self {
            units: vec![one; degree],
            permutation: Permutation::identity(degree),
            automorphism_id: 0, // Identity automorphism
        }
    }

    /// Get the degree (size of the transformation)
    pub fn degree(&self) -> usize {
        self.units.len()
    }

    /// Get the units vector
    pub fn units(&self) -> &[T] {
        &self.units
    }

    /// Get the permutation
    pub fn permutation(&self) -> &Permutation {
        &self.permutation
    }

    /// Get the automorphism ID
    pub fn automorphism_id(&self) -> usize {
        self.automorphism_id
    }

    /// Compute the inverse
    ///
    /// For a transformation (φ, π, α), the inverse is
    /// (φ', π⁻¹, α⁻¹) with φ'_j = α⁻¹(φ_{π⁻¹(j)})⁻¹,
    /// so that g·g⁻¹ = g⁻¹·g = identity under [`compose`].
    ///
    /// # Arguments
    /// * `apply_inverse_automorphism` - The map u ↦ α⁻¹(u) on R
    /// * `invert_unit` - The map u ↦ u⁻¹ on the units of R
    /// * `invert_automorphism_id` - ID identifying α⁻¹
    pub fn inverse_with<A, F>(
        &self,
        apply_inverse_automorphism: A,
        invert_unit: F,
        invert_automorphism_id: usize,
    ) -> Self
    where
        A: Fn(&T) -> T,
        F: Fn(&T) -> T,
    {
        let inv_perm = self.permutation.inverse();
        let n = self.degree();

        // φ'_j = α⁻¹(φ_{π⁻¹(j)})⁻¹
        let mut inv_units = Vec::with_capacity(n);
        for j in 0..n {
            let pi_inv_j = inv_perm.apply(j).unwrap();
            let inv_u = invert_unit(&apply_inverse_automorphism(&self.units[pi_inv_j]));
            inv_units.push(inv_u);
        }

        Self {
            units: inv_units,
            permutation: inv_perm,
            automorphism_id: invert_automorphism_id,
        }
    }
}

impl<T: Clone + fmt::Debug> fmt::Debug for SemimonomialTransformation<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemimonomialTransformation")
            .field("units", &self.units)
            .field("permutation", &self.permutation)
            .field("automorphism_id", &self.automorphism_id)
            .finish()
    }
}

impl<T: Clone + fmt::Display> fmt::Display for SemimonomialTransformation<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        write!(f, "[")?;
        for (i, u) in self.units.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", u)?;
        }
        write!(f, "], ")?;
        write!(f, "{:?}, ", self.permutation)?;
        write!(f, "α_{})", self.automorphism_id)
    }
}

/// Action of semimonomial transformation on a vector
///
/// For a transformation (φ, π, α) acting on vector v:
/// Result[i] = φ[i]^(-1) * α(v[π(i)])
pub fn action_on_vector<T, F>(
    transform: &SemimonomialTransformation<T>,
    vector: &[T],
    apply_automorphism: F,
    multiply: impl Fn(&T, &T) -> T,
    invert: impl Fn(&T) -> T,
) -> Vec<T>
where
    T: Clone,
    F: Fn(&T) -> T,
{
    let n = transform.degree();
    assert_eq!(vector.len(), n, "Vector length must match transformation degree");

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let pi_i = transform.permutation.apply(i).unwrap();
        let transformed = apply_automorphism(&vector[pi_i]);
        let inv_unit = invert(&transform.units[i]);
        result.push(multiply(&inv_unit, &transformed));
    }

    result
}

/// Action of semimonomial transformation on a matrix (acts on each row)
pub fn action_on_matrix<T, F>(
    transform: &SemimonomialTransformation<T>,
    matrix: &[Vec<T>],
    apply_automorphism: F,
    multiply: impl Fn(&T, &T) -> T,
    invert: impl Fn(&T) -> T,
) -> Vec<Vec<T>>
where
    T: Clone,
    F: Fn(&T) -> T + Clone,
{
    matrix
        .iter()
        .map(|row| {
            action_on_vector(
                transform,
                row,
                apply_automorphism.clone(),
                &multiply,
                &invert,
            )
        })
        .collect()
}

/// Compose two semimonomial transformations
///
/// (φ, π, α)(ψ, σ, β) = (φ · ψ^(π,α), πσ, α ∘ β)
///
/// with ψ^(π,α)_i = α(ψ_{π(i)}) and πσ the product applying π first,
/// (πσ)(i) = σ(π(i)) (the GAP/Sage convention). With this product the law is
/// associative and [`action_on_vector`] is a left action: g·(h·v) = (gh)·v.
pub fn compose<T>(
    first: &SemimonomialTransformation<T>,
    second: &SemimonomialTransformation<T>,
    apply_first_automorphism: impl Fn(&T) -> T,
    multiply: impl Fn(&T, &T) -> T,
    compose_automorphisms: impl Fn(usize, usize) -> usize,
) -> SemimonomialTransformation<T>
where
    T: Clone,
{
    assert_eq!(
        first.degree(),
        second.degree(),
        "Transformations must have same degree"
    );

    let n = first.degree();

    // πσ: i ↦ σ(π(i)). `Permutation::compose(a, b)` is a ∘ b (apply b first),
    // so this is σ ∘ π.
    let composed_perm = second.permutation.compose(&first.permutation).unwrap();

    // Compute new units: φ · ψ^(π,α)
    let mut new_units = Vec::with_capacity(n);
    for i in 0..n {
        let pi_i = first.permutation.apply(i).unwrap();
        let transformed = apply_first_automorphism(&second.units[pi_i]);
        let product = multiply(&first.units[i], &transformed);
        new_units.push(product);
    }

    // Compose automorphisms
    let new_automorphism_id = compose_automorphisms(first.automorphism_id, second.automorphism_id);

    SemimonomialTransformation::new(new_units, composed_perm, new_automorphism_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id: SemimonomialTransformation<i32> =
            SemimonomialTransformation::identity(3, 1);

        assert_eq!(id.degree(), 3);
        assert_eq!(id.units(), &[1, 1, 1]);
        assert_eq!(id.permutation(), &Permutation::identity(3));
        assert_eq!(id.automorphism_id(), 0);
    }

    #[test]
    fn test_new() {
        let perm = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        let transform = SemimonomialTransformation::new(vec![1, -1, 1], perm, 0);

        assert_eq!(transform.degree(), 3);
        assert_eq!(transform.units(), &[1, -1, 1]);
    }

    #[test]
    fn test_action_on_vector() {
        let perm = Permutation::identity(3);
        let transform = SemimonomialTransformation::new(vec![2, 3, 4], perm, 0);
        let vector = vec![1, 2, 3];

        let result = action_on_vector(
            &transform,
            &vector,
            |x| *x, // Identity automorphism
            |a, b| a * b,
            |x| 1 / x, // Note: integer division won't work properly; this is just for testing
        );

        // For proper testing with integers, we'd need a different approach
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_compose() {
        let perm1 = Permutation::identity(3);
        let perm2 = Permutation::identity(3);

        let t1 = SemimonomialTransformation::new(vec![1, 1, 1], perm1, 0);
        let t2 = SemimonomialTransformation::new(vec![2, 2, 2], perm2, 0);

        let composed = compose(
            &t1,
            &t2,
            |x| *x, // Identity automorphism
            |a, b| a * b,
            |a, b| a + b, // Just for testing
        );

        assert_eq!(composed.degree(), 3);
        assert_eq!(composed.units(), &[2, 2, 2]);
    }

    // GF(4) = F_2(w), w^2 = w + 1. A unit w^k is stored as its exponent
    // k in Z/3; Aut(GF(4)) = {id, Frobenius x -> x^2} acts on exponents as
    // k -> 2^a k (a = automorphism id, 0 or 1), and composes by a + b mod 2.
    fn gf4_aut(a: usize) -> impl Fn(&u8) -> u8 {
        move |k: &u8| if a % 2 == 0 { *k } else { (2 * *k) % 3 }
    }
    fn gf4_mul(x: &u8, y: &u8) -> u8 {
        (x + y) % 3
    }
    fn gf4_inv(x: &u8) -> u8 {
        (3 - x) % 3
    }
    fn gf4_compose(
        g: &SemimonomialTransformation<u8>,
        h: &SemimonomialTransformation<u8>,
    ) -> SemimonomialTransformation<u8> {
        compose(g, h, gf4_aut(g.automorphism_id()), gf4_mul, |a, b| (a + b) % 2)
    }
    fn gf4_act(g: &SemimonomialTransformation<u8>, v: &[u8]) -> Vec<u8> {
        action_on_vector(g, v, gf4_aut(g.automorphism_id()), gf4_mul, gf4_inv)
    }
    fn gf4_inverse(g: &SemimonomialTransformation<u8>) -> SemimonomialTransformation<u8> {
        // Frobenius is an involution, so alpha^{-1} = alpha.
        g.inverse_with(gf4_aut(g.automorphism_id()), gf4_inv, g.automorphism_id())
    }
    fn gf4_samples() -> Vec<SemimonomialTransformation<u8>> {
        let perms = [vec![1, 2, 0], vec![1, 0, 2], vec![0, 2, 1], vec![2, 0, 1]];
        let units = [vec![0, 1, 2], vec![2, 2, 1], vec![1, 0, 0], vec![0, 0, 1]];
        let mut out = Vec::new();
        for (i, p) in perms.iter().enumerate() {
            for a in 0..2 {
                out.push(SemimonomialTransformation::new(
                    units[(i + a) % units.len()].clone(),
                    Permutation::from_vec(p.clone()).unwrap(),
                    a,
                ));
            }
        }
        out
    }
    fn same(a: &SemimonomialTransformation<u8>, b: &SemimonomialTransformation<u8>) -> bool {
        a.units() == b.units()
            && a.permutation() == b.permutation()
            && a.automorphism_id() == b.automorphism_id()
    }

    #[test]
    fn test_compose_is_associative_over_gf4() {
        // The semimonomial group (GF(4)^*)^3 x| (S_3 x Aut GF(4)) is a group:
        // (gh)k = g(hk) for all sampled g, h, k (including non-commuting
        // permutations and the Frobenius automorphism).
        let xs = gf4_samples();
        for g in &xs {
            for h in &xs {
                for k in &xs {
                    let left = gf4_compose(&gf4_compose(g, h), k);
                    let right = gf4_compose(g, &gf4_compose(h, k));
                    assert!(same(&left, &right), "(gh)k != g(hk) for {:?} {:?} {:?}", g, h, k);
                }
            }
        }
    }

    #[test]
    fn test_action_on_vector_is_a_left_action_over_gf4() {
        // g.(h.v) = (gh).v for the documented action and composition.
        let xs = gf4_samples();
        let vectors = [vec![0u8, 1, 2], vec![2, 2, 0], vec![1, 0, 1]];
        for g in &xs {
            for h in &xs {
                for v in &vectors {
                    assert_eq!(gf4_act(g, &gf4_act(h, v)), gf4_act(&gf4_compose(g, h), v));
                }
            }
        }
    }

    #[test]
    fn test_inverse_over_gf4() {
        // g g^{-1} = g^{-1} g = identity, also when alpha is the Frobenius and
        // pi is a 3-cycle (where pi(i) and pi^{-1}(i) differ).
        let id = SemimonomialTransformation::identity(3, 0u8); // w^0 = 1
        for g in &gf4_samples() {
            let g_inv = gf4_inverse(g);
            assert!(same(&gf4_compose(g, &g_inv), &id), "g g^-1 != 1 for {:?}", g);
            assert!(same(&gf4_compose(&g_inv, g), &id), "g^-1 g != 1 for {:?}", g);
        }
    }

    #[test]
    fn test_display() {
        let perm = Permutation::identity(2);
        let transform = SemimonomialTransformation::new(vec![1, -1], perm, 0);
        let display = format!("{}", transform);

        assert!(display.contains("[1, -1]"));
        assert!(display.contains("α_0"));
    }
}
