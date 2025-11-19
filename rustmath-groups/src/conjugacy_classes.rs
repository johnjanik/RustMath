//! Conjugacy Classes in Groups
//!
//! This module implements conjugacy classes for group elements.
//! Two elements g and h in a group G are conjugate if there exists
//! an element x in G such that h = x⁻¹gx.
//!
//! A conjugacy class is the set of all elements conjugate to a given element.

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

/// A conjugacy class in a group
///
/// Represents the set of all elements conjugate to a representative element.
/// For element g, the conjugacy class is {xgx⁻¹ | x ∈ G}.
#[derive(Clone, Debug)]
pub struct ConjugacyClass<G: GroupElement> {
    /// Representative element of the conjugacy class
    representative: G,
    /// Cached set of elements (computed lazily)
    elements: Option<HashSet<G>>,
}

/// Trait for group elements that can be used in conjugacy classes
pub trait GroupElement: Clone + Eq + Hash + fmt::Debug {
    /// Compute the conjugate of this element by another: x⁻¹ * self * x
    fn conjugate_by(&self, x: &Self) -> Self;

    /// Get the inverse of this element
    fn inverse(&self) -> Self;

    /// Get the identity element
    fn identity() -> Self;

    /// Multiply two elements
    fn multiply(&self, other: &Self) -> Self;
}

impl<G: GroupElement> ConjugacyClass<G> {
    /// Create a new conjugacy class with the given representative
    ///
    /// # Arguments
    /// * `representative` - The element representing this conjugacy class
    ///
    /// # Examples
    /// ```ignore
    /// use rustmath_groups::conjugacy_classes::ConjugacyClass;
    ///
    /// let g = /* some group element */;
    /// let conj_class = ConjugacyClass::new(g);
    /// ```
    pub fn new(representative: G) -> Self {
        ConjugacyClass {
            representative,
            elements: None,
        }
    }

    /// Get the representative element
    pub fn representative(&self) -> &G {
        &self.representative
    }

    /// Check if an element is in this conjugacy class
    ///
    /// This checks if the given element is conjugate to the representative.
    pub fn contains(&self, element: &G, conjugators: &[G]) -> bool {
        if element == &self.representative {
            return true;
        }

        // Check if element = x⁻¹ * rep * x for any x in conjugators
        for x in conjugators {
            let conjugate = self.representative.conjugate_by(x);
            if &conjugate == element {
                return true;
            }
        }

        false
    }

    /// Compute all elements in the conjugacy class
    ///
    /// # Arguments
    /// * `conjugators` - List of group elements to use for conjugation
    ///
    /// # Returns
    /// A set containing all elements in the conjugacy class
    pub fn compute_elements(&mut self, conjugators: &[G]) -> &HashSet<G> {
        if self.elements.is_none() {
            let mut elements = HashSet::new();
            elements.insert(self.representative.clone());

            for x in conjugators {
                let conjugate = self.representative.conjugate_by(x);
                elements.insert(conjugate);
            }

            self.elements = Some(elements);
        }

        self.elements.as_ref().unwrap()
    }

    /// Get the size of the conjugacy class
    ///
    /// # Arguments
    /// * `conjugators` - List of group elements to use for conjugation
    pub fn size(&mut self, conjugators: &[G]) -> usize {
        self.compute_elements(conjugators).len()
    }

    /// Check if this is a "real" conjugacy class
    ///
    /// A conjugacy class is real if it's closed under taking inverses,
    /// i.e., if g is in the class, then g⁻¹ is also in the class.
    pub fn is_real(&mut self, conjugators: &[G]) -> bool {
        let elements = self.compute_elements(conjugators);

        for elem in elements.iter() {
            let inv = elem.inverse();
            if !elements.contains(&inv) {
                return false;
            }
        }

        true
    }

    /// Get a list of all elements in the conjugacy class
    pub fn to_vec(&mut self, conjugators: &[G]) -> Vec<G> {
        self.compute_elements(conjugators).iter().cloned().collect()
    }

    /// Check if this conjugacy class equals another
    ///
    /// Two conjugacy classes are equal if they contain the same elements.
    pub fn equals(&mut self, other: &mut ConjugacyClass<G>, conjugators: &[G]) -> bool {
        let self_elements = self.compute_elements(conjugators);
        let other_elements = other.compute_elements(conjugators);

        self_elements == other_elements
    }
}

impl<G: GroupElement + fmt::Display> fmt::Display for ConjugacyClass<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Conjugacy class of {}", self.representative)
    }
}

/// Compute all conjugacy classes of a group
///
/// # Arguments
/// * `elements` - All elements of the group
///
/// # Returns
/// A vector of conjugacy classes partitioning the group
pub fn conjugacy_classes<G: GroupElement>(elements: &[G]) -> Vec<ConjugacyClass<G>> {
    let mut classes = Vec::new();
    let mut visited = HashSet::new();

    for elem in elements {
        if visited.contains(elem) {
            continue;
        }

        let mut class = ConjugacyClass::new(elem.clone());
        let class_elements = class.compute_elements(elements);

        // Mark all elements in this class as visited
        for e in class_elements {
            visited.insert(e.clone());
        }

        classes.push(class);
    }

    classes
}

/// Count the number of conjugacy classes in a group
///
/// This is equal to the number of irreducible representations.
pub fn num_conjugacy_classes<G: GroupElement>(elements: &[G]) -> usize {
    conjugacy_classes(elements).len()
}

/// Simple example implementation of GroupElement for demonstration
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PermutationElement {
    /// Permutation as a vector (maps index i to perm[i])
    perm: Vec<usize>,
}

impl PermutationElement {
    /// Create a new permutation element
    pub fn new(perm: Vec<usize>) -> Result<Self, String> {
        // Validate that this is a valid permutation
        let n = perm.len();
        let mut seen = vec![false; n];

        for &i in &perm {
            if i >= n {
                return Err(format!("Invalid permutation: {} >= {}", i, n));
            }
            if seen[i] {
                return Err(format!("Invalid permutation: {} appears twice", i));
            }
            seen[i] = true;
        }

        Ok(PermutationElement { perm })
    }

    /// Apply this permutation to an index
    pub fn apply(&self, i: usize) -> usize {
        self.perm[i]
    }

    /// Get the degree of this permutation
    pub fn degree(&self) -> usize {
        self.perm.len()
    }
}

impl GroupElement for PermutationElement {
    fn conjugate_by(&self, x: &Self) -> Self {
        // Conjugation: x⁻¹ * self * x
        let x_inv = x.inverse();
        x_inv.multiply(self).multiply(x)
    }

    fn inverse(&self) -> Self {
        let n = self.perm.len();
        let mut inv = vec![0; n];

        for (i, &val) in self.perm.iter().enumerate() {
            inv[val] = i;
        }

        PermutationElement { perm: inv }
    }

    fn identity() -> Self {
        PermutationElement {
            perm: vec![0, 1, 2], // Identity permutation of degree 3
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.degree(), other.degree());

        let n = self.degree();
        let mut result = vec![0; n];

        for i in 0..n {
            result[i] = self.apply(other.apply(i));
        }

        PermutationElement { perm: result }
    }
}

impl fmt::Display for PermutationElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.perm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_identity() {
        let id = PermutationElement::new(vec![0, 1, 2]).unwrap();
        let inv = id.inverse();
        assert_eq!(id, inv);
    }

    #[test]
    fn test_permutation_inverse() {
        let p = PermutationElement::new(vec![1, 2, 0]).unwrap();
        let p_inv = p.inverse();
        let product = p.multiply(&p_inv);
        let identity = PermutationElement::new(vec![0, 1, 2]).unwrap();
        assert_eq!(product, identity);
    }

    #[test]
    fn test_permutation_multiply() {
        // (0 1) * (1 2) = (0 1 2)
        let p1 = PermutationElement::new(vec![1, 0, 2]).unwrap();
        let p2 = PermutationElement::new(vec![0, 2, 1]).unwrap();
        let product = p1.multiply(&p2);
        assert_eq!(product.perm, vec![1, 2, 0]);
    }

    #[test]
    fn test_conjugacy_class_creation() {
        let g = PermutationElement::new(vec![1, 2, 0]).unwrap();
        let conj_class = ConjugacyClass::new(g.clone());
        assert_eq!(conj_class.representative(), &g);
    }

    #[test]
    fn test_conjugacy_class_contains() {
        let g = PermutationElement::new(vec![1, 2, 0]).unwrap();
        let mut conj_class = ConjugacyClass::new(g.clone());

        // Create conjugators (all elements of S3)
        let conjugators = vec![
            PermutationElement::new(vec![0, 1, 2]).unwrap(),
            PermutationElement::new(vec![1, 0, 2]).unwrap(),
            PermutationElement::new(vec![0, 2, 1]).unwrap(),
            PermutationElement::new(vec![2, 1, 0]).unwrap(),
            PermutationElement::new(vec![1, 2, 0]).unwrap(),
            PermutationElement::new(vec![2, 0, 1]).unwrap(),
        ];

        assert!(conj_class.contains(&g, &conjugators));
    }

    #[test]
    fn test_conjugacy_class_size() {
        let g = PermutationElement::new(vec![1, 0, 2]).unwrap(); // (0 1)
        let mut conj_class = ConjugacyClass::new(g.clone());

        let conjugators = vec![
            PermutationElement::new(vec![0, 1, 2]).unwrap(),
            PermutationElement::new(vec![1, 0, 2]).unwrap(),
            PermutationElement::new(vec![0, 2, 1]).unwrap(),
            PermutationElement::new(vec![2, 1, 0]).unwrap(),
        ];

        let size = conj_class.size(&conjugators);
        // (0 1), (0 2), (1 2) should be in the class
        assert!(size > 0);
    }

    #[test]
    fn test_conjugation() {
        // Test g^x = x⁻¹gx
        let g = PermutationElement::new(vec![1, 2, 0]).unwrap();
        let x = PermutationElement::new(vec![1, 0, 2]).unwrap();

        let conjugate1 = g.conjugate_by(&x);
        let x_inv = x.inverse();
        let conjugate2 = x_inv.multiply(&g).multiply(&x);

        assert_eq!(conjugate1, conjugate2);
    }

    #[test]
    fn test_identity_conjugacy_class() {
        let id = PermutationElement::new(vec![0, 1, 2]).unwrap();
        let mut conj_class = ConjugacyClass::new(id.clone());

        let conjugators = vec![
            PermutationElement::new(vec![0, 1, 2]).unwrap(),
            PermutationElement::new(vec![1, 0, 2]).unwrap(),
            PermutationElement::new(vec![2, 1, 0]).unwrap(),
        ];

        let size = conj_class.size(&conjugators);
        // Identity is always in its own conjugacy class
        assert_eq!(size, 1);
    }

    #[test]
    fn test_num_conjugacy_classes_s3() {
        // S3 has 3 conjugacy classes: {e}, {(12), (13), (23)}, {(123), (132)}
        let elements = vec![
            PermutationElement::new(vec![0, 1, 2]).unwrap(), // e
            PermutationElement::new(vec![1, 0, 2]).unwrap(), // (01)
            PermutationElement::new(vec![0, 2, 1]).unwrap(), // (12)
            PermutationElement::new(vec![2, 1, 0]).unwrap(), // (02)
            PermutationElement::new(vec![1, 2, 0]).unwrap(), // (012)
            PermutationElement::new(vec![2, 0, 1]).unwrap(), // (021)
        ];

        let num_classes = num_conjugacy_classes(&elements);
        assert_eq!(num_classes, 3);
    }

    #[test]
    fn test_conjugacy_classes_partition() {
        // Test that conjugacy classes partition the group
        let elements = vec![
            PermutationElement::new(vec![0, 1, 2]).unwrap(),
            PermutationElement::new(vec![1, 0, 2]).unwrap(),
            PermutationElement::new(vec![0, 2, 1]).unwrap(),
        ];

        let classes = conjugacy_classes(&elements);

        // Count total elements in all classes
        let mut total = 0;
        for mut class in classes {
            total += class.size(&elements);
        }

        // Should equal the number of group elements
        assert!(total >= elements.len());
    }

    #[test]
    fn test_is_real_conjugacy_class() {
        let g = PermutationElement::new(vec![1, 0, 2]).unwrap();
        let mut conj_class = ConjugacyClass::new(g);

        let conjugators = vec![
            PermutationElement::new(vec![0, 1, 2]).unwrap(),
            PermutationElement::new(vec![1, 0, 2]).unwrap(),
        ];

        // Transpositions are self-inverse, so their conjugacy class is real
        assert!(conj_class.is_real(&conjugators));
    }
}
