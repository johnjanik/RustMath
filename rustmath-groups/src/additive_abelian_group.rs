//! Additive Abelian Groups
//!
//! This module implements finitely generated additive abelian groups.
//! These are groups where the operation is addition (not multiplication)
//! and all elements commute.
//!
//! By the fundamental theorem, every finitely generated additive abelian group
//! is isomorphic to Z^r ⊕ Z/n₁Z ⊕ Z/n₂Z ⊕ ... ⊕ Z/nₖZ
//! where n₁ | n₂ | ... | nₖ (invariant factors)

use std::fmt;
use std::collections::HashMap;

/// Create an additive abelian group from invariants
///
/// # Arguments
/// * `invariants` - List of invariants (n₁, n₂, ..., nₖ) where n₁ | n₂ | ... | nₖ
///
/// # Returns
/// An `AdditiveAbelianGroup` with the specified structure
///
/// # Examples
/// ```
/// use rustmath_groups::additive_abelian_group::additive_abelian_group;
///
/// // Create Z/2Z ⊕ Z/4Z
/// let g = additive_abelian_group(vec![2, 4]).unwrap();
/// assert_eq!(g.order(), Some(8));
/// ```
pub fn additive_abelian_group(invariants: Vec<usize>) -> Result<AdditiveAbelianGroup, String> {
    AdditiveAbelianGroup::new(0, invariants)
}

/// An element of an additive abelian group
///
/// Represents an element as a vector of coordinates with respect to
/// the group's generators
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdditiveAbelianGroupElement {
    /// Coordinates with respect to generators
    coordinates: Vec<i64>,
    /// Reference to the parent group
    parent: AdditiveAbelianGroup,
}

impl AdditiveAbelianGroupElement {
    /// Create a new element from coordinates
    pub fn new(coordinates: Vec<i64>, parent: AdditiveAbelianGroup) -> Result<Self, String> {
        if coordinates.len() != parent.rank() {
            return Err(format!(
                "Expected {} coordinates, got {}",
                parent.rank(),
                coordinates.len()
            ));
        }

        // Reduce coordinates modulo invariants
        let mut reduced = coordinates.clone();
        let offset = parent.free_rank();
        for (i, &inv) in parent.invariant_factors().iter().enumerate() {
            if inv > 0 {
                reduced[offset + i] = reduced[offset + i].rem_euclid(inv as i64);
            }
        }

        Ok(AdditiveAbelianGroupElement {
            coordinates: reduced,
            parent,
        })
    }

    /// Get the coordinates
    pub fn coordinates(&self) -> &[i64] {
        &self.coordinates
    }

    /// Get the parent group
    pub fn parent(&self) -> &AdditiveAbelianGroup {
        &self.parent
    }

    /// Add two elements
    pub fn add(&self, other: &AdditiveAbelianGroupElement) -> Result<AdditiveAbelianGroupElement, String> {
        if self.parent != other.parent {
            return Err("Cannot add elements from different groups".to_string());
        }

        let sum: Vec<i64> = self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| a + b)
            .collect();

        AdditiveAbelianGroupElement::new(sum, self.parent.clone())
    }

    /// Negate an element
    pub fn negate(&self) -> AdditiveAbelianGroupElement {
        let neg: Vec<i64> = self.coordinates.iter().map(|x| -x).collect();
        AdditiveAbelianGroupElement::new(neg, self.parent.clone()).unwrap()
    }

    /// Scalar multiplication (n * element)
    pub fn scalar_mul(&self, n: i64) -> AdditiveAbelianGroupElement {
        let scaled: Vec<i64> = self.coordinates.iter().map(|x| n * x).collect();
        AdditiveAbelianGroupElement::new(scaled, self.parent.clone()).unwrap()
    }

    /// Check if this is the identity element
    pub fn is_identity(&self) -> bool {
        self.coordinates.iter().all(|&x| x == 0)
    }

    /// Compute the order of this element
    ///
    /// Returns None if the element has infinite order
    pub fn order(&self) -> Option<usize> {
        if self.is_identity() {
            return Some(1);
        }

        // Check free part
        let free_rank = self.parent.free_rank();
        for i in 0..free_rank {
            if self.coordinates[i] != 0 {
                return None; // Infinite order
            }
        }

        // Compute LCM of orders in torsion part
        let mut lcm = 1usize;
        for (i, &inv) in self.parent.invariant_factors().iter().enumerate() {
            if inv == 0 {
                continue;
            }

            let coord = self.coordinates[free_rank + i].rem_euclid(inv as i64) as usize;
            if coord != 0 {
                let gcd = gcd_usize(coord, inv);
                let order = inv / gcd;
                lcm = lcm_usize(lcm, order);
            }
        }

        Some(lcm)
    }
}

impl fmt::Display for AdditiveAbelianGroupElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, coord) in self.coordinates.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coord)?;
        }
        write!(f, ")")
    }
}

/// A finitely generated additive abelian group
///
/// Represents groups of the form Z^r ⊕ Z/n₁Z ⊕ Z/n₂Z ⊕ ... ⊕ Z/nₖZ
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdditiveAbelianGroup {
    /// Free rank (number of Z factors)
    free_rank: usize,
    /// Invariant factors (torsion part)
    invariant_factors: Vec<usize>,
}

impl AdditiveAbelianGroup {
    /// Create a new additive abelian group
    ///
    /// # Arguments
    /// * `free_rank` - Number of free Z factors
    /// * `invariant_factors` - Torsion part, must satisfy n₁ | n₂ | ... | nₖ
    pub fn new(free_rank: usize, invariant_factors: Vec<usize>) -> Result<Self, String> {
        // Validate divisibility condition
        for i in 1..invariant_factors.len() {
            if invariant_factors[i] > 0 && invariant_factors[i - 1] > 0 {
                if invariant_factors[i] % invariant_factors[i - 1] != 0 {
                    return Err(format!(
                        "Invariant factors must be divisible: {} does not divide {}",
                        invariant_factors[i - 1],
                        invariant_factors[i]
                    ));
                }
            }
        }

        Ok(AdditiveAbelianGroup {
            free_rank,
            invariant_factors,
        })
    }

    /// Get the free rank
    pub fn free_rank(&self) -> usize {
        self.free_rank
    }

    /// Get the invariant factors
    pub fn invariant_factors(&self) -> &[usize] {
        &self.invariant_factors
    }

    /// Get the torsion rank
    pub fn torsion_rank(&self) -> usize {
        self.invariant_factors.len()
    }

    /// Get the total rank
    pub fn rank(&self) -> usize {
        self.free_rank + self.torsion_rank()
    }

    /// Get the order of the group
    ///
    /// Returns None if infinite, Some(n) if finite
    pub fn order(&self) -> Option<usize> {
        if self.free_rank > 0 {
            return None;
        }

        let mut order = 1usize;
        for &n in &self.invariant_factors {
            if n == 0 {
                return None;
            }
            order = order.checked_mul(n)?;
        }

        Some(order)
    }

    /// Get the exponent of the group (LCM of all element orders)
    pub fn exponent(&self) -> Option<usize> {
        if self.free_rank > 0 {
            return None;
        }

        if self.invariant_factors.is_empty() {
            return Some(1);
        }

        // The exponent is the largest invariant factor
        self.invariant_factors.last().copied()
    }

    /// Check if the group is finite
    pub fn is_finite(&self) -> bool {
        self.free_rank == 0 && !self.invariant_factors.iter().any(|&n| n == 0)
    }

    /// Check if the group is cyclic
    pub fn is_cyclic(&self) -> bool {
        self.free_rank + self.invariant_factors.len() <= 1
    }

    /// Check if this is a multiplicative group (always false for additive groups)
    pub fn is_multiplicative(&self) -> bool {
        false
    }

    /// Create the identity element
    pub fn identity(&self) -> AdditiveAbelianGroupElement {
        AdditiveAbelianGroupElement::new(vec![0; self.rank()], self.clone()).unwrap()
    }

    /// Create a generator
    pub fn gen(&self, i: usize) -> Result<AdditiveAbelianGroupElement, String> {
        if i >= self.rank() {
            return Err(format!("Generator index {} out of range", i));
        }

        let mut coords = vec![0; self.rank()];
        coords[i] = 1;

        AdditiveAbelianGroupElement::new(coords, self.clone())
    }

    /// Get all generators
    pub fn gens(&self) -> Vec<AdditiveAbelianGroupElement> {
        (0..self.rank())
            .map(|i| self.gen(i).unwrap())
            .collect()
    }

    /// Get a short descriptive name
    pub fn short_name(&self) -> String {
        self.structure_string()
    }

    /// Get a string representation following the structure theorem
    pub fn structure_string(&self) -> String {
        let mut parts = Vec::new();

        // Add free part
        if self.free_rank > 0 {
            if self.free_rank == 1 {
                parts.push("Z".to_string());
            } else {
                parts.push(format!("Z^{}", self.free_rank));
            }
        }

        // Add torsion part
        for &n in &self.invariant_factors {
            if n > 0 {
                parts.push(format!("Z/{}", n));
            }
        }

        if parts.is_empty() {
            "{0}".to_string()
        } else {
            parts.join(" + ")
        }
    }
}

impl fmt::Display for AdditiveAbelianGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Additive abelian group isomorphic to {}", self.structure_string())
    }
}

/// A variant of AdditiveAbelianGroup with fixed (custom) generators
///
/// This allows specifying generators that are not in Smith normal form
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdditiveAbelianGroupFixedGens {
    /// The underlying group
    base: AdditiveAbelianGroup,
    /// Custom generators (as elements of the base group)
    generators: Vec<AdditiveAbelianGroupElement>,
}

impl AdditiveAbelianGroupFixedGens {
    /// Create a new group with fixed generators
    pub fn new(
        base: AdditiveAbelianGroup,
        generators: Vec<AdditiveAbelianGroupElement>,
    ) -> Result<Self, String> {
        if generators.len() != base.rank() {
            return Err("Number of generators must match rank".to_string());
        }

        Ok(AdditiveAbelianGroupFixedGens { base, generators })
    }

    /// Get the generators
    pub fn gens(&self) -> &[AdditiveAbelianGroupElement] {
        &self.generators
    }

    /// Get the underlying group
    pub fn base_group(&self) -> &AdditiveAbelianGroup {
        &self.base
    }
}

/// Utility function to compute cover and relations from invariants
///
/// This function is used internally to convert invariants into the
/// appropriate module structure
pub fn cover_and_relations_from_invariants(
    invariants: &[usize],
) -> (Vec<usize>, Vec<Vec<i64>>) {
    let n = invariants.len();

    // Cover is Z^n (free module of rank n)
    let cover = vec![0; n]; // 0 means infinite/free

    // Relations are given by the invariants on the diagonal
    let mut relations = Vec::new();
    for (i, &inv) in invariants.iter().enumerate() {
        if inv > 0 {
            let mut rel = vec![0; n];
            rel[i] = inv as i64;
            relations.push(rel);
        }
    }

    (cover, relations)
}

// Helper functions

fn gcd_usize(a: usize, b: usize) -> usize {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

fn lcm_usize(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        return 0;
    }
    a * b / gcd_usize(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_additive_abelian_group_creation() {
        // Z/2Z ⊕ Z/4Z
        let g = additive_abelian_group(vec![2, 4]).unwrap();
        assert_eq!(g.free_rank(), 0);
        assert_eq!(g.torsion_rank(), 2);
        assert_eq!(g.rank(), 2);
        assert_eq!(g.order(), Some(8));
        assert!(g.is_finite());
        assert!(!g.is_cyclic());
    }

    #[test]
    fn test_cyclic_group() {
        let z6 = additive_abelian_group(vec![6]).unwrap();
        assert_eq!(z6.order(), Some(6));
        assert!(z6.is_cyclic());
        assert!(z6.is_finite());
    }

    #[test]
    fn test_free_group() {
        // Z ⊕ Z
        let g = AdditiveAbelianGroup::new(2, vec![]).unwrap();
        assert_eq!(g.free_rank(), 2);
        assert_eq!(g.torsion_rank(), 0);
        assert!(g.order().is_none());
        assert!(!g.is_finite());
    }

    #[test]
    fn test_element_operations() {
        let g = additive_abelian_group(vec![4]).unwrap();
        let e1 = AdditiveAbelianGroupElement::new(vec![1], g.clone()).unwrap();
        let e2 = AdditiveAbelianGroupElement::new(vec![2], g.clone()).unwrap();

        // Addition
        let sum = e1.add(&e2).unwrap();
        assert_eq!(sum.coordinates(), &[3]);

        // Negation
        let neg = e1.negate();
        assert_eq!(neg.coordinates(), &[3]); // -1 ≡ 3 (mod 4)

        // Scalar multiplication
        let triple = e1.scalar_mul(3);
        assert_eq!(triple.coordinates(), &[3]);
    }

    #[test]
    fn test_element_order() {
        let g = additive_abelian_group(vec![12]).unwrap();

        // Element of order 1 (identity)
        let e0 = AdditiveAbelianGroupElement::new(vec![0], g.clone()).unwrap();
        assert_eq!(e0.order(), Some(1));
        assert!(e0.is_identity());

        // Element of order 12
        let e1 = AdditiveAbelianGroupElement::new(vec![1], g.clone()).unwrap();
        assert_eq!(e1.order(), Some(12));

        // Element of order 6
        let e2 = AdditiveAbelianGroupElement::new(vec![2], g.clone()).unwrap();
        assert_eq!(e2.order(), Some(6));

        // Element of order 4
        let e3 = AdditiveAbelianGroupElement::new(vec![3], g.clone()).unwrap();
        assert_eq!(e3.order(), Some(4));
    }

    #[test]
    fn test_identity() {
        let g = additive_abelian_group(vec![2, 4]).unwrap();
        let id = g.identity();
        assert!(id.is_identity());
        assert_eq!(id.coordinates(), &[0, 0]);
    }

    #[test]
    fn test_generators() {
        let g = additive_abelian_group(vec![3, 6]).unwrap();
        let gens = g.gens();
        assert_eq!(gens.len(), 2);
        assert_eq!(gens[0].coordinates(), &[1, 0]);
        assert_eq!(gens[1].coordinates(), &[0, 1]);
    }

    #[test]
    fn test_exponent() {
        let g1 = additive_abelian_group(vec![2, 4]).unwrap();
        assert_eq!(g1.exponent(), Some(4));

        let g2 = additive_abelian_group(vec![3, 9]).unwrap();
        assert_eq!(g2.exponent(), Some(9));

        let g3 = AdditiveAbelianGroup::new(1, vec![]).unwrap();
        assert_eq!(g3.exponent(), None); // Infinite group
    }

    #[test]
    fn test_structure_string() {
        let g1 = additive_abelian_group(vec![6]).unwrap();
        assert_eq!(g1.structure_string(), "Z/6");

        let g2 = AdditiveAbelianGroup::new(2, vec![]).unwrap();
        assert_eq!(g2.structure_string(), "Z^2");

        let g3 = AdditiveAbelianGroup::new(1, vec![2, 4]).unwrap();
        assert_eq!(g3.structure_string(), "Z + Z/2 + Z/4");
    }

    #[test]
    fn test_short_name() {
        let g = additive_abelian_group(vec![2, 4]).unwrap();
        assert_eq!(g.short_name(), "Z/2 + Z/4");
    }

    #[test]
    fn test_is_multiplicative() {
        let g = additive_abelian_group(vec![5]).unwrap();
        assert!(!g.is_multiplicative());
    }

    #[test]
    fn test_cover_and_relations() {
        let (cover, relations) = cover_and_relations_from_invariants(&[2, 4, 6]);
        assert_eq!(cover.len(), 3);
        assert_eq!(relations.len(), 3);
        assert_eq!(relations[0], vec![2, 0, 0]);
        assert_eq!(relations[1], vec![0, 4, 0]);
        assert_eq!(relations[2], vec![0, 0, 6]);
    }

    #[test]
    fn test_fixed_generators() {
        let g = additive_abelian_group(vec![6]).unwrap();
        let gen = AdditiveAbelianGroupElement::new(vec![2], g.clone()).unwrap();

        let fixed = AdditiveAbelianGroupFixedGens::new(g.clone(), vec![gen.clone()]).unwrap();
        assert_eq!(fixed.gens().len(), 1);
        assert_eq!(fixed.gens()[0], gen);
    }

    #[test]
    fn test_invalid_invariants() {
        // Invariants must be divisible
        let result = AdditiveAbelianGroup::new(0, vec![4, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_element_modular_reduction() {
        let g = additive_abelian_group(vec![5]).unwrap();

        // 7 should be reduced to 2 (mod 5)
        let e = AdditiveAbelianGroupElement::new(vec![7], g.clone()).unwrap();
        assert_eq!(e.coordinates(), &[2]);

        // -3 should be reduced to 2 (mod 5)
        let e2 = AdditiveAbelianGroupElement::new(vec![-3], g.clone()).unwrap();
        assert_eq!(e2.coordinates(), &[2]);
    }
}
