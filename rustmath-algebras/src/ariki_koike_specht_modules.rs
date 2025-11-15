//! Specht Modules for Ariki-Koike Algebras
//!
//! Specht modules are irreducible representations of the Ariki-Koike algebra
//! (Hecke algebra of complex reflection group G(r,1,n)).
//!
//! These modules are indexed by multipartitions and generalize the classical
//! Specht modules for the symmetric group.
//!
//! Corresponds to sage.algebras.hecke_algebras.ariki_koike_specht_modules

use rustmath_core::Ring;
use crate::ariki_koike_algebra::{ArikiKoikeAlgebra, ArikiKoikeElement};
use std::collections::HashMap;
use std::fmt;

/// A partition of a non-negative integer
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Partition {
    /// Parts of the partition (in non-increasing order)
    pub parts: Vec<usize>,
}

impl Partition {
    /// Create a new partition
    pub fn new(mut parts: Vec<usize>) -> Self {
        // Sort in descending order and remove zeros
        parts.retain(|&p| p > 0);
        parts.sort_unstable_by(|a, b| b.cmp(a));
        Partition { parts }
    }

    /// Get the size (sum of parts)
    pub fn size(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Check if this is a valid partition
    pub fn is_valid(&self) -> bool {
        // Check that parts are in non-increasing order
        self.parts.windows(2).all(|w| w[0] >= w[1])
    }
}

impl fmt::Display for Partition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;
        for (i, part) in self.parts.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", part)?;
        }
        write!(f, ")")
    }
}

/// A multipartition - r-tuple of partitions
///
/// Used to index Specht modules of the Ariki-Koike algebra
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Multipartition {
    /// The r partitions
    pub partitions: Vec<Partition>,
}

impl Multipartition {
    /// Create a new multipartition
    pub fn new(partitions: Vec<Partition>) -> Self {
        Multipartition { partitions }
    }

    /// Get the rank (number of partitions)
    pub fn rank(&self) -> usize {
        self.partitions.len()
    }

    /// Get the total size (sum of sizes of all partitions)
    pub fn size(&self) -> usize {
        self.partitions.iter().map(|p| p.size()).sum()
    }

    /// Check if this is a valid multipartition
    pub fn is_valid(&self) -> bool {
        self.partitions.iter().all(|p| p.is_valid())
    }
}

impl fmt::Display for Multipartition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;
        for (i, part) in self.partitions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", part)?;
        }
        write!(f, ")")
    }
}

/// Element of a Specht module
///
/// Represented as a linear combination of standard tableaux
#[derive(Debug, Clone)]
pub struct SpechtModuleElement<R: Ring> {
    /// Coefficients for each standard tableau (indexed by tableau ID)
    pub coeffs: HashMap<usize, R>,
    /// Parent multipartition
    multipartition: Multipartition,
}

impl<R: Ring + Clone> SpechtModuleElement<R> {
    /// Create a new Specht module element
    pub fn new(coeffs: HashMap<usize, R>, multipartition: Multipartition) -> Self {
        SpechtModuleElement {
            coeffs,
            multipartition,
        }
    }

    /// Create the zero element
    pub fn zero(multipartition: Multipartition) -> Self {
        SpechtModuleElement {
            coeffs: HashMap::new(),
            multipartition,
        }
    }

    /// Create a basis element (single standard tableau)
    pub fn basis_element(tableau_id: usize, multipartition: Multipartition) -> Self
    where
        R: From<i64>,
    {
        let mut coeffs = HashMap::new();
        coeffs.insert(tableau_id, R::from(1));
        SpechtModuleElement {
            coeffs,
            multipartition,
        }
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.values().all(|c| c.is_zero())
    }

    /// Get the parent multipartition
    pub fn multipartition(&self) -> &Multipartition {
        &self.multipartition
    }

    /// Add two elements
    pub fn add(&self, other: &SpechtModuleElement<R>) -> Self {
        if self.multipartition != other.multipartition {
            panic!("Cannot add elements from different Specht modules");
        }

        let mut result = self.coeffs.clone();
        for (id, coeff) in &other.coeffs {
            let current = result.entry(*id).or_insert_with(R::zero);
            *current = current.clone() + coeff.clone();
        }

        // Remove zero coefficients
        result.retain(|_, v| !v.is_zero());

        SpechtModuleElement {
            coeffs: result,
            multipartition: self.multipartition.clone(),
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        let mut result = HashMap::new();
        for (id, coeff) in &self.coeffs {
            let new_coeff = scalar.clone() * coeff.clone();
            if !new_coeff.is_zero() {
                result.insert(*id, new_coeff);
            }
        }

        SpechtModuleElement {
            coeffs: result,
            multipartition: self.multipartition.clone(),
        }
    }
}

impl<R: Ring + Clone + fmt::Display> fmt::Display for SpechtModuleElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (id, coeff) in &self.coeffs {
            if !first {
                write!(f, " + ")?;
            }
            write!(f, "{}*T{}", coeff, id)?;
            first = false;
        }
        Ok(())
    }
}

/// Specht Module for an Ariki-Koike algebra
///
/// The irreducible representation indexed by a multipartition
pub struct SpechtModule<R: Ring> {
    /// Parent Ariki-Koike algebra
    algebra: ArikiKoikeAlgebra<R>,
    /// The multipartition indexing this module
    multipartition: Multipartition,
    /// Dimension of the module
    dimension: usize,
}

impl<R: Ring + Clone> SpechtModule<R> {
    /// Create a new Specht module
    ///
    /// # Arguments
    /// * `algebra` - The parent Ariki-Koike algebra
    /// * `multipartition` - The multipartition indexing this module
    pub fn new(algebra: ArikiKoikeAlgebra<R>, multipartition: Multipartition) -> Self {
        // Validate that the multipartition matches the algebra rank
        if multipartition.rank() != algebra.rank() {
            panic!(
                "Multipartition rank {} doesn't match algebra rank {}",
                multipartition.rank(),
                algebra.rank()
            );
        }

        // Compute dimension (number of standard tableaux)
        let dimension = Self::compute_dimension(&multipartition);

        SpechtModule {
            algebra,
            multipartition,
            dimension,
        }
    }

    /// Get the parent algebra
    pub fn algebra(&self) -> &ArikiKoikeAlgebra<R> {
        &self.algebra
    }

    /// Get the multipartition
    pub fn multipartition(&self) -> &Multipartition {
        &self.multipartition
    }

    /// Get the dimension of the module
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get a basis element by index
    pub fn basis_element(&self, index: usize) -> Option<SpechtModuleElement<R>>
    where
        R: From<i64>,
    {
        if index < self.dimension {
            Some(SpechtModuleElement::basis_element(
                index,
                self.multipartition.clone(),
            ))
        } else {
            None
        }
    }

    /// Get the zero element
    pub fn zero(&self) -> SpechtModuleElement<R> {
        SpechtModuleElement::zero(self.multipartition.clone())
    }

    /// Action of algebra element on module element
    ///
    /// Computes h * v where h is an algebra element and v is a module element
    pub fn action(
        &self,
        _algebra_elem: &ArikiKoikeElement<R>,
        module_elem: &SpechtModuleElement<R>,
    ) -> SpechtModuleElement<R> {
        // Placeholder: actual implementation would compute the action
        // using the Murphy basis and Garnir relations
        module_elem.clone()
    }

    /// Compute dimension as number of standard tableaux
    fn compute_dimension(multipartition: &Multipartition) -> usize {
        // Simplified: actual implementation would count standard tableaux
        // using the hook-length formula for multipartitions
        let n = multipartition.size();
        if n == 0 {
            return 1;
        }

        // Placeholder: return a reasonable estimate
        let mut dim = 1;
        for partition in &multipartition.partitions {
            if partition.size() > 0 {
                dim *= partition.size();
            }
        }
        dim
    }
}

impl<R: Ring + Clone> fmt::Display for SpechtModule<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SpechtModule({}, dim={})",
            self.multipartition, self.dimension
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multipartition() {
        let p1 = Partition::new(vec![2, 1]);
        let p2 = Partition::new(vec![1]);
        let mp = Multipartition::new(vec![p1, p2]);

        assert_eq!(mp.rank(), 2);
        assert_eq!(mp.size(), 4); // (2,1) + (1) = 4 total boxes
    }

    #[test]
    fn test_specht_module_creation() {
        let algebra: ArikiKoikeAlgebra<i64> = ArikiKoikeAlgebra::new(2, 3);
        let p1 = Partition::new(vec![2]);
        let p2 = Partition::new(vec![1]);
        let mp = Multipartition::new(vec![p1, p2]);

        let module = SpechtModule::new(algebra, mp);
        assert!(module.dimension() > 0);
    }

    #[test]
    fn test_specht_element_operations() {
        let p1 = Partition::new(vec![2]);
        let p2 = Partition::new(vec![1]);
        let mp = Multipartition::new(vec![p1, p2]);

        let e1: SpechtModuleElement<i64> = SpechtModuleElement::basis_element(0, mp.clone());
        let e2: SpechtModuleElement<i64> = SpechtModuleElement::basis_element(1, mp.clone());

        let sum = e1.add(&e2);
        assert!(!sum.is_zero());
    }

    #[test]
    fn test_scalar_multiplication() {
        let p = Partition::new(vec![2, 1]);
        let mp = Multipartition::new(vec![p]);

        let elem: SpechtModuleElement<i64> = SpechtModuleElement::basis_element(0, mp);
        let scaled = elem.scalar_mul(&3);

        assert!(!scaled.is_zero());
    }

    #[test]
    fn test_zero_element() {
        let p = Partition::new(vec![1]);
        let mp = Multipartition::new(vec![p]);

        let zero: SpechtModuleElement<i64> = SpechtModuleElement::zero(mp);
        assert!(zero.is_zero());
    }
}
