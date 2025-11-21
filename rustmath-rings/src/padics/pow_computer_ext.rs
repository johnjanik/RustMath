//! Extended Power Computer for p-adic Extensions
//!
//! This module provides power computation for p-adic field extensions,
//! including support for Frobenius endomorphisms.
//!
//! ## Overview
//!
//! The `PowComputerExt` extends `PowComputer` functionality for:
//! - Unramified extensions: Extensions where the ramification index e = 1
//! - Eisenstein extensions: Extensions where e = degree
//! - Frobenius endomorphism: The map σ: x → x^p in characteristic p
//!
//! ## Frobenius Endomorphism
//!
//! For an unramified extension of degree f over ℚ_p, the Frobenius endomorphism σ
//! is defined as σ(x) = x^p. This is a generator of the Galois group Gal(K/ℚ_p) ≅ ℤ/fℤ.
//!
//! Key properties:
//! - σ^f = identity
//! - σ fixes ℚ_p
//! - For x in the residue field: σ(x) ≡ x^p (mod p)
//!
//! ## Cache Invalidation
//!
//! Like `PowComputer`, the cache is immutable. Extension-specific caches include:
//! - Powers of p in the extension
//! - Frobenius powers (σ, σ², ..., σ^f)
//! - Ramification data
//!
//! ## Examples
//!
//! ```
//! use rustmath_integers::Integer;
//! use rustmath_rings::padics::PowComputerExt;
//!
//! // Create for unramified extension of degree 3 over Q_5
//! let pc_ext = PowComputerExt::unramified(Integer::from(5), 10, 3);
//!
//! // Compute Frobenius power
//! // In practice, this would apply σ^2 to an element
//! assert_eq!(pc_ext.frobenius_degree(), 3);
//! ```

use super::pow_computer::PowComputer;
use rustmath_integers::Integer;

/// Type of p-adic extension
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtensionType {
    /// Unramified extension (ramification index e = 1, inertia degree f > 1)
    Unramified,

    /// Totally ramified (Eisenstein) extension (e = degree, f = 1)
    TotallyRamified,

    /// General extension (e > 1, f > 1)
    General,
}

/// Extended power computer for p-adic field extensions
///
/// Provides efficient computation for p-adic extensions, including:
/// - Power caching for the base field
/// - Ramification data
/// - Frobenius endomorphism support for unramified extensions
#[derive(Clone, Debug)]
pub struct PowComputerExt {
    /// Base power computer
    base: PowComputer,

    /// Degree of the extension [K : ℚ_p]
    degree: usize,

    /// Ramification index e (order of p in the extension)
    ramification_index: usize,

    /// Inertia degree f (degree of residue field extension)
    inertia_degree: usize,

    /// Type of extension
    extension_type: ExtensionType,

    /// Uniformizer for the extension (typically π where π^e = p for Eisenstein)
    /// Stored as coefficients in some representation
    uniformizer_power: usize,
}

impl PowComputerExt {
    /// Create a power computer for an unramified extension
    ///
    /// In an unramified extension of degree f:
    /// - Ramification index e = 1
    /// - Inertia degree f = degree
    /// - The uniformizer is p itself
    ///
    /// # Arguments
    ///
    /// * `prime` - The base prime p
    /// * `cache_limit` - Cache limit for powers
    /// * `inertia_degree` - Degree of residue field extension
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputerExt;
    ///
    /// // Unramified extension of degree 3 over Q_5
    /// let pc = PowComputerExt::unramified(Integer::from(5), 10, 3);
    /// assert_eq!(pc.ramification_index(), 1);
    /// assert_eq!(pc.inertia_degree(), 3);
    /// ```
    pub fn unramified(prime: Integer, cache_limit: usize, inertia_degree: usize) -> Self {
        assert!(inertia_degree > 0, "Inertia degree must be positive");

        PowComputerExt {
            base: PowComputer::for_field(prime, cache_limit),
            degree: inertia_degree,
            ramification_index: 1,
            inertia_degree,
            extension_type: ExtensionType::Unramified,
            uniformizer_power: 1,
        }
    }

    /// Create a power computer for a totally ramified (Eisenstein) extension
    ///
    /// In a totally ramified extension of degree e:
    /// - Ramification index e = degree
    /// - Inertia degree f = 1
    /// - The uniformizer π satisfies π^e = p
    ///
    /// # Arguments
    ///
    /// * `prime` - The base prime p
    /// * `cache_limit` - Cache limit for powers
    /// * `ramification_index` - Ramification index e
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputerExt;
    ///
    /// // Eisenstein extension of degree 4 over Q_7
    /// let pc = PowComputerExt::eisenstein(Integer::from(7), 10, 4);
    /// assert_eq!(pc.ramification_index(), 4);
    /// assert_eq!(pc.inertia_degree(), 1);
    /// ```
    pub fn eisenstein(prime: Integer, cache_limit: usize, ramification_index: usize) -> Self {
        assert!(ramification_index > 0, "Ramification index must be positive");

        PowComputerExt {
            base: PowComputer::for_field(prime, cache_limit),
            degree: ramification_index,
            ramification_index,
            inertia_degree: 1,
            extension_type: ExtensionType::TotallyRamified,
            uniformizer_power: ramification_index,
        }
    }

    /// Create a power computer for a general extension
    ///
    /// For a general extension:
    /// - Degree = e × f
    /// - e is the ramification index
    /// - f is the inertia degree
    ///
    /// # Arguments
    ///
    /// * `prime` - The base prime p
    /// * `cache_limit` - Cache limit for powers
    /// * `ramification_index` - Ramification index e
    /// * `inertia_degree` - Inertia degree f
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputerExt;
    ///
    /// // Extension with e=2, f=3 over Q_5
    /// let pc = PowComputerExt::general(Integer::from(5), 10, 2, 3);
    /// assert_eq!(pc.degree(), 6); // e × f = 2 × 3
    /// ```
    pub fn general(
        prime: Integer,
        cache_limit: usize,
        ramification_index: usize,
        inertia_degree: usize,
    ) -> Self {
        assert!(ramification_index > 0, "Ramification index must be positive");
        assert!(inertia_degree > 0, "Inertia degree must be positive");

        let degree = ramification_index * inertia_degree;

        let extension_type = if ramification_index == 1 {
            ExtensionType::Unramified
        } else if inertia_degree == 1 {
            ExtensionType::TotallyRamified
        } else {
            ExtensionType::General
        };

        PowComputerExt {
            base: PowComputer::for_field(prime, cache_limit),
            degree,
            ramification_index,
            inertia_degree,
            extension_type,
            uniformizer_power: ramification_index,
        }
    }

    /// Get the base power computer
    pub fn base(&self) -> &PowComputer {
        &self.base
    }

    /// Get the prime p
    pub fn prime(&self) -> &Integer {
        self.base.prime()
    }

    /// Get the degree of the extension
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the ramification index e
    pub fn ramification_index(&self) -> usize {
        self.ramification_index
    }

    /// Get the inertia degree f (also called residue degree)
    pub fn inertia_degree(&self) -> usize {
        self.inertia_degree
    }

    /// Get the Frobenius degree (same as inertia degree)
    ///
    /// The Frobenius automorphism generates a cyclic group of order f.
    pub fn frobenius_degree(&self) -> usize {
        self.inertia_degree
    }

    /// Get the extension type
    pub fn extension_type(&self) -> ExtensionType {
        self.extension_type
    }

    /// Check if this is an unramified extension
    pub fn is_unramified(&self) -> bool {
        matches!(self.extension_type, ExtensionType::Unramified)
    }

    /// Check if this is totally ramified
    pub fn is_totally_ramified(&self) -> bool {
        matches!(self.extension_type, ExtensionType::TotallyRamified)
    }

    /// Get p^n in the base field
    pub fn pow(&self, n: usize) -> Integer {
        self.base.pow(n)
    }

    /// Compute the norm degree for Frobenius trace computations
    ///
    /// For an element x in an unramified extension K/ℚ_p of degree f,
    /// the norm is: N(x) = x · σ(x) · σ²(x) · ... · σ^(f-1)(x)
    ///
    /// This method returns f, which is used in norm computations.
    pub fn norm_degree(&self) -> usize {
        self.inertia_degree
    }

    /// Compute Frobenius exponent for iteration k
    ///
    /// Returns p^k, which is the exponent applied in the k-th Frobenius power.
    /// For σ^k(x), we compute x^(p^k).
    ///
    /// # Arguments
    ///
    /// * `k` - Frobenius iteration (0 ≤ k < f)
    ///
    /// # Panics
    ///
    /// Panics if k >= inertia_degree
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputerExt;
    ///
    /// let pc = PowComputerExt::unramified(Integer::from(5), 10, 3);
    /// assert_eq!(pc.frobenius_exponent(0), Integer::from(1));   // σ^0 = id
    /// assert_eq!(pc.frobenius_exponent(1), Integer::from(5));   // σ: x → x^5
    /// assert_eq!(pc.frobenius_exponent(2), Integer::from(25));  // σ²: x → x^25
    /// ```
    pub fn frobenius_exponent(&self, k: usize) -> Integer {
        assert!(
            k < self.inertia_degree,
            "Frobenius iteration must be < inertia degree"
        );

        if k == 0 {
            Integer::one()
        } else {
            self.base.prime().pow(k as u32)
        }
    }

    /// Apply Frobenius iteration k times (symbolic representation)
    ///
    /// In practice, this returns the exponent p^k to be applied to an element.
    /// The actual application depends on the element representation.
    ///
    /// # Mathematical Background
    ///
    /// For an unramified extension K/ℚ_p with residue field 𝔽_{p^f}:
    /// - The Frobenius σ: x ↦ x^p generates Gal(K/ℚ_p)
    /// - σ^f = identity (since residue field has p^f elements)
    /// - For k iterations: σ^k(x) = x^(p^k)
    ///
    /// # Arguments
    ///
    /// * `k` - Number of Frobenius applications
    ///
    /// # Returns
    ///
    /// The exponent p^k to apply to an element
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputerExt;
    ///
    /// let pc = PowComputerExt::unramified(Integer::from(7), 10, 4);
    ///
    /// // Apply Frobenius twice: σ²(x) = x^(7²) = x^49
    /// assert_eq!(pc.frobenius(2), Integer::from(49));
    /// ```
    pub fn frobenius(&self, k: usize) -> Integer {
        // Reduce k modulo f since σ^f = identity
        let k_reduced = k % self.inertia_degree;
        self.frobenius_exponent(k_reduced)
    }

    /// Compute trace of Frobenius action
    ///
    /// For an element x, the trace is:
    /// Tr(x) = x + σ(x) + σ²(x) + ... + σ^(f-1)(x)
    ///
    /// This returns the list of Frobenius exponents [1, p, p², ..., p^(f-1)]
    /// to be applied.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputerExt;
    ///
    /// let pc = PowComputerExt::unramified(Integer::from(3), 10, 3);
    /// let exponents = pc.frobenius_trace_exponents();
    ///
    /// // Should get [1, 3, 9] for trace computation
    /// assert_eq!(exponents.len(), 3);
    /// assert_eq!(exponents[0], Integer::from(1));
    /// assert_eq!(exponents[1], Integer::from(3));
    /// assert_eq!(exponents[2], Integer::from(9));
    /// ```
    pub fn frobenius_trace_exponents(&self) -> Vec<Integer> {
        (0..self.inertia_degree)
            .map(|k| self.frobenius_exponent(k))
            .collect()
    }

    /// Compute norm of Frobenius action
    ///
    /// For an element x, the norm is:
    /// N(x) = x · σ(x) · σ²(x) · ... · σ^(f-1)(x)
    ///
    /// This returns the sum of exponents: 1 + p + p² + ... + p^(f-1)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::PowComputerExt;
    ///
    /// let pc = PowComputerExt::unramified(Integer::from(2), 10, 3);
    /// // Norm exponent sum: 1 + 2 + 4 = 7
    /// assert_eq!(pc.frobenius_norm_exponent_sum(), Integer::from(7));
    /// ```
    pub fn frobenius_norm_exponent_sum(&self) -> Integer {
        self.frobenius_trace_exponents().iter().cloned().sum()
    }

    /// Get the uniformizer power (π^e = p for Eisenstein extensions)
    ///
    /// For an Eisenstein extension with uniformizer π:
    /// - π^e = p where e is the ramification index
    /// - This returns e
    pub fn uniformizer_power(&self) -> usize {
        self.uniformizer_power
    }

    /// Extend the cache limit
    ///
    /// Creates a new `PowComputerExt` with increased cache limit.
    pub fn extend_cache(&self, new_limit: usize) -> Self {
        PowComputerExt {
            base: self.base.extend_cache(new_limit),
            degree: self.degree,
            ramification_index: self.ramification_index,
            inertia_degree: self.inertia_degree,
            extension_type: self.extension_type,
            uniformizer_power: self.uniformizer_power,
        }
    }
}

impl PartialEq for PowComputerExt {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
            && self.degree == other.degree
            && self.ramification_index == other.ramification_index
            && self.inertia_degree == other.inertia_degree
            && self.extension_type == other.extension_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unramified_extension() {
        let pc = PowComputerExt::unramified(Integer::from(5), 10, 3);

        assert_eq!(pc.degree(), 3);
        assert_eq!(pc.ramification_index(), 1);
        assert_eq!(pc.inertia_degree(), 3);
        assert_eq!(pc.frobenius_degree(), 3);
        assert!(pc.is_unramified());
        assert!(!pc.is_totally_ramified());
    }

    #[test]
    fn test_eisenstein_extension() {
        let pc = PowComputerExt::eisenstein(Integer::from(7), 10, 4);

        assert_eq!(pc.degree(), 4);
        assert_eq!(pc.ramification_index(), 4);
        assert_eq!(pc.inertia_degree(), 1);
        assert!(!pc.is_unramified());
        assert!(pc.is_totally_ramified());
        assert_eq!(pc.uniformizer_power(), 4);
    }

    #[test]
    fn test_general_extension() {
        let pc = PowComputerExt::general(Integer::from(3), 10, 2, 3);

        assert_eq!(pc.degree(), 6); // e × f = 2 × 3
        assert_eq!(pc.ramification_index(), 2);
        assert_eq!(pc.inertia_degree(), 3);
        assert!(!pc.is_unramified());
        assert!(!pc.is_totally_ramified());
    }

    #[test]
    fn test_frobenius_exponent() {
        let pc = PowComputerExt::unramified(Integer::from(5), 10, 3);

        // σ^0 = identity
        assert_eq!(pc.frobenius_exponent(0), Integer::from(1));

        // σ: x → x^5
        assert_eq!(pc.frobenius_exponent(1), Integer::from(5));

        // σ²: x → x^25
        assert_eq!(pc.frobenius_exponent(2), Integer::from(25));
    }

    #[test]
    fn test_frobenius_iteration() {
        let pc = PowComputerExt::unramified(Integer::from(7), 10, 4);

        // Test various iterations
        assert_eq!(pc.frobenius(0), Integer::from(1));
        assert_eq!(pc.frobenius(1), Integer::from(7));
        assert_eq!(pc.frobenius(2), Integer::from(49));
        assert_eq!(pc.frobenius(3), Integer::from(343));

        // σ^4 = identity for degree 4
        assert_eq!(pc.frobenius(4), Integer::from(1));
        assert_eq!(pc.frobenius(5), Integer::from(7)); // wraps around
    }

    #[test]
    fn test_frobenius_trace_exponents() {
        let pc = PowComputerExt::unramified(Integer::from(2), 10, 4);
        let exponents = pc.frobenius_trace_exponents();

        assert_eq!(exponents.len(), 4);
        assert_eq!(exponents[0], Integer::from(1));  // σ^0
        assert_eq!(exponents[1], Integer::from(2));  // σ^1
        assert_eq!(exponents[2], Integer::from(4));  // σ^2
        assert_eq!(exponents[3], Integer::from(8));  // σ^3
    }

    #[test]
    fn test_frobenius_norm_exponent() {
        let pc = PowComputerExt::unramified(Integer::from(2), 10, 3);

        // Sum: 1 + 2 + 4 = 7
        assert_eq!(pc.frobenius_norm_exponent_sum(), Integer::from(7));
    }

    #[test]
    fn test_norm_degree() {
        let pc = PowComputerExt::unramified(Integer::from(5), 10, 3);
        assert_eq!(pc.norm_degree(), 3);
    }

    #[test]
    fn test_extend_cache() {
        let pc = PowComputerExt::unramified(Integer::from(3), 10, 2);
        let extended = pc.extend_cache(20);

        assert_eq!(extended.base().cache_limit(), 20);
        assert_eq!(extended.degree(), 2);
    }

    #[test]
    fn test_base_access() {
        let pc = PowComputerExt::eisenstein(Integer::from(11), 15, 3);

        assert_eq!(pc.prime(), &Integer::from(11));
        assert_eq!(pc.pow(2), Integer::from(121));
    }

    #[test]
    #[should_panic(expected = "Frobenius iteration must be < inertia degree")]
    fn test_frobenius_out_of_bounds() {
        let pc = PowComputerExt::unramified(Integer::from(5), 10, 3);
        pc.frobenius_exponent(3); // Should panic (0, 1, 2 are valid)
    }

    #[test]
    fn test_extension_types() {
        let unram = PowComputerExt::unramified(Integer::from(5), 10, 3);
        assert_eq!(unram.extension_type(), ExtensionType::Unramified);

        let eisenstein = PowComputerExt::eisenstein(Integer::from(7), 10, 4);
        assert_eq!(eisenstein.extension_type(), ExtensionType::TotallyRamified);

        let general = PowComputerExt::general(Integer::from(3), 10, 2, 3);
        assert_eq!(general.extension_type(), ExtensionType::General);
    }

    #[test]
    fn test_clone() {
        let pc1 = PowComputerExt::unramified(Integer::from(5), 10, 3);
        let pc2 = pc1.clone();

        assert_eq!(pc1, pc2);
        assert_eq!(pc2.frobenius(2), Integer::from(25));
    }
}
