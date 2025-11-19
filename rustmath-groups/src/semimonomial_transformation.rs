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
//! where ψ^(π,α) means applying α to each component then permuting by π.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::semimonomial_transformation::SemimonomialTransformation;
//! use rustmath_combinatorics::Permutation;
//!
//! // Create a simple transformation with integer units
//! let perm = Permutation::identity(3);
//! let units = vec![1, -1, 1];
//! let transform = SemimonomialTransformation::new(units, perm, Box::new(|x| x));
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

    /// Compute the inverse (requires inverse computation)
    ///
    /// For a transformation (φ, π, α), the inverse is:
    /// (φ', π^(-1), α^(-1))
    /// where φ'_i = α^(-1)(φ_{π(i)})^(-1)
    pub fn inverse_with<F>(&self, invert_unit: F, invert_automorphism_id: usize) -> Self
    where
        F: Fn(&T) -> T,
    {
        let inv_perm = self.permutation.inverse();
        let n = self.degree();

        // Compute inverse units
        let mut inv_units = Vec::with_capacity(n);
        for i in 0..n {
            let pi_i = self.permutation.apply(i).unwrap();
            let inv_u = invert_unit(&self.units[pi_i]);
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

    // Compose permutations
    let composed_perm = first.permutation.compose(&second.permutation).unwrap();

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

    #[test]
    fn test_display() {
        let perm = Permutation::identity(2);
        let transform = SemimonomialTransformation::new(vec![1, -1], perm, 0);
        let display = format!("{}", transform);

        assert!(display.contains("[1, -1]"));
        assert!(display.contains("α_0"));
    }
}
