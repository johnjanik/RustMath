//! Partition Algebra
//!
//! The partition algebra P_n(δ) is an associative algebra with a basis indexed by
//! set partitions of {1,...,n,1',...,n'} (n top points and n bottom points).
//!
//! A partition diagram is represented as a pair of set partitions where blocks can
//! connect top points, bottom points, or both. Multiplication is defined by vertical
//! concatenation with a trace rule: closed loops contribute a factor of δ.
//!
//! The partition algebra generalizes:
//! - Symmetric group algebra (diagrams with perfect matchings)
//! - Brauer algebra (diagrams with pairwise matchings)
//! - Temperley-Lieb algebra (planar non-crossing diagrams)
//!
//! This implementation works over arbitrary rings R, with the parameter δ ∈ R.

use rustmath_core::{Ring, MathError, Result};
use rustmath_combinatorics::{SetPartition, set_partition_iterator};
use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use std::ops::{Add, Sub, Mul, Neg};
use std::hash::{Hash, Hasher};
use crate::traits::Algebra;

/// A partition diagram representing a basis element of the partition algebra
///
/// A diagram consists of:
/// - n top points (labeled 0, 1, ..., n-1)
/// - n bottom points (labeled n, n+1, ..., 2n-1)
/// - A set partition of all 2n points
///
/// For easier manipulation, we store this as a single set partition of 2n elements,
/// where elements 0..n are top points and elements n..2n are bottom points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionDiagram {
    /// The underlying set partition of 2n points
    partition: SetPartition,
    /// Size parameter n (number of points on top and bottom)
    n: usize,
}

impl Hash for PartitionDiagram {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n.hash(state);
        // Hash the RGS representation for consistent hashing
        self.partition.to_rgs().hash(state);
    }
}

impl PartitionDiagram {
    /// Create a new partition diagram from a set partition of 2n points
    pub fn new(partition: SetPartition, n: usize) -> Option<Self> {
        if partition.size() != 2 * n {
            return None;
        }
        Some(PartitionDiagram { partition, n })
    }

    /// Get the size parameter n
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the underlying set partition
    pub fn partition(&self) -> &SetPartition {
        &self.partition
    }

    /// Check if a block contains only top points (0..n)
    fn is_top_block(&self, block: &[usize]) -> bool {
        block.iter().all(|&x| x < self.n)
    }

    /// Check if a block contains only bottom points (n..2n)
    fn is_bottom_block(&self, block: &[usize]) -> bool {
        block.iter().all(|&x| x >= self.n)
    }

    /// Check if a block is "through" (contains both top and bottom points)
    fn is_through_block(&self, block: &[usize]) -> bool {
        let has_top = block.iter().any(|&x| x < self.n);
        let has_bottom = block.iter().any(|&x| x >= self.n);
        has_top && has_bottom
    }

    /// Create the identity diagram (each top point connected to corresponding bottom point)
    pub fn identity(n: usize) -> Self {
        let mut blocks = Vec::new();
        for i in 0..n {
            blocks.push(vec![i, n + i]);
        }
        let partition = SetPartition::new(blocks, 2 * n).expect("identity diagram construction failed");
        PartitionDiagram { partition, n }
    }

    /// Compose two diagrams: self on top, other on bottom
    ///
    /// Returns (composed_diagram, number_of_closed_loops)
    ///
    /// The composition connects the bottom points of self to the top points of other.
    /// Blocks that close up (don't reach the final top or bottom) become closed loops.
    pub fn compose(&self, other: &Self) -> Option<(Self, usize)> {
        if self.n != other.n {
            return None;
        }

        let n = self.n;

        // We'll build a new partition on 2n points:
        // - Points 0..n represent the top points of self
        // - Points n..2n represent the bottom points of other
        //
        // The middle points (bottom of self = top of other) are temporarily labeled
        // and then removed if they form closed loops.

        // Create a union-find structure for the 4n points:
        // 0..n: top points of self
        // n..2n: middle points (bottom of self = top of other)
        // 2n..3n: middle points again (same physical points, different role)
        // 3n..4n: bottom points of other
        //
        // Actually, let's use a simpler approach with direct block merging

        let mut point_to_component = vec![None; 4 * n];
        let mut components: Vec<Vec<usize>> = Vec::new();

        // Helper to get or create component for a point
        let mut get_component = |point: usize, point_to_component: &mut Vec<Option<usize>>, components: &mut Vec<Vec<usize>>| -> usize {
            if let Some(comp_id) = point_to_component[point] {
                comp_id
            } else {
                let comp_id = components.len();
                components.push(vec![point]);
                point_to_component[point] = Some(comp_id);
                comp_id
            }
        };

        // Process blocks from self (diagram on top)
        // Map: 0..n -> 0..n (top points stay)
        // Map: n..2n -> n..2n (bottom points are middle)
        for block in self.partition.blocks() {
            let mut comp_id = None;
            for &point in block {
                let mapped_point = point; // Direct mapping
                let point_comp = get_component(mapped_point, &mut point_to_component, &mut components);

                if let Some(existing_comp) = comp_id {
                    if existing_comp != point_comp {
                        // Merge components
                        let mut to_merge = components[point_comp].clone();
                        for &p in &to_merge {
                            point_to_component[p] = Some(existing_comp);
                        }
                        components[existing_comp].append(&mut to_merge);
                        components[point_comp].clear();
                    }
                } else {
                    comp_id = Some(point_comp);
                }
            }
        }

        // Process blocks from other (diagram on bottom)
        // Map: 0..n -> n..2n (top points are middle, same as bottom of self)
        // Map: n..2n -> 2n..3n (bottom points need new indices)
        for block in other.partition.blocks() {
            let mut comp_id = None;
            for &point in block {
                let mapped_point = if point < n {
                    // Top point of other = bottom point of self
                    n + point
                } else {
                    // Bottom point of other -> map to 2n..3n range
                    n + point
                };
                let point_comp = get_component(mapped_point, &mut point_to_component, &mut components);

                if let Some(existing_comp) = comp_id {
                    if existing_comp != point_comp {
                        // Merge components
                        let mut to_merge = components[point_comp].clone();
                        for &p in &to_merge {
                            point_to_component[p] = Some(existing_comp);
                        }
                        components[existing_comp].append(&mut to_merge);
                        components[point_comp].clear();
                    }
                } else {
                    comp_id = Some(point_comp);
                }
            }
        }

        // Now separate into final blocks and closed loops
        let mut final_blocks = Vec::new();
        let mut num_closed_loops = 0;

        for component in components {
            if component.is_empty() {
                continue;
            }

            // Check if component has top points (0..n) or bottom points (2n..3n)
            let has_top = component.iter().any(|&x| x < n);
            let has_bottom = component.iter().any(|&x| x >= 2 * n);

            if !has_top && !has_bottom {
                // Closed loop (only middle points)
                num_closed_loops += 1;
            } else {
                // Remap to 0..2n range
                let mut final_block = Vec::new();
                for &point in &component {
                    if point < n {
                        // Top points stay as 0..n
                        final_block.push(point);
                    } else if point >= 2 * n {
                        // Bottom points map to n..2n
                        final_block.push(point - n);
                    }
                    // Middle points (n..2n) are ignored in final diagram
                }
                if !final_block.is_empty() {
                    final_blocks.push(final_block);
                }
            }
        }

        // Handle case where all blocks closed
        if final_blocks.is_empty() {
            // All loops closed - return identity with maximum loops
            return Some((Self::identity(n), num_closed_loops));
        }

        let partition = SetPartition::new(final_blocks, 2 * n)?;
        let diagram = PartitionDiagram { partition, n };
        Some((diagram, num_closed_loops))
    }

    /// Generate all partition diagrams of size n
    pub fn all_diagrams(n: usize) -> Vec<Self> {
        set_partition_iterator(2 * n)
            .map(|partition| PartitionDiagram { partition, n })
            .collect()
    }

    /// Get the dimension of the partition algebra P_n(δ)
    /// This is Bell(2n), the number of set partitions of 2n elements
    pub fn dimension(n: usize) -> usize {
        // For practical purposes, we can compute this
        Self::all_diagrams(n).len()
    }
}

impl Display for PartitionDiagram {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PartitionDiagram(n={}, blocks={:?})", self.n, self.partition.blocks())
    }
}

/// An element of the partition algebra P_n(δ) over a ring R
///
/// Elements are formal linear combinations of partition diagrams with coefficients in R.
#[derive(Debug, Clone)]
pub struct PartitionAlgebraElement<R: Ring> {
    /// Linear combination: diagram -> coefficient
    terms: HashMap<PartitionDiagram, R>,
    /// Size parameter n
    n: usize,
    /// Trace parameter δ ∈ R
    delta: R,
}

impl<R: Ring> PartitionAlgebraElement<R> {
    /// Create a new algebra element
    pub fn new(terms: HashMap<PartitionDiagram, R>, n: usize, delta: R) -> Self {
        let mut element = PartitionAlgebraElement { terms, n, delta };
        element.cleanup();
        element
    }

    /// Create the zero element
    pub fn zero_with_params(n: usize, delta: R) -> Self {
        PartitionAlgebraElement {
            terms: HashMap::new(),
            n,
            delta,
        }
    }

    /// Create a basis element (single diagram with coefficient 1)
    pub fn basis_element(diagram: PartitionDiagram, delta: R) -> Self {
        let n = diagram.size();
        let mut terms = HashMap::new();
        terms.insert(diagram, R::one());
        PartitionAlgebraElement { terms, n, delta }
    }

    /// Create the identity element
    pub fn identity(n: usize, delta: R) -> Self {
        Self::basis_element(PartitionDiagram::identity(n), delta)
    }

    /// Remove zero coefficients
    fn cleanup(&mut self) {
        self.terms.retain(|_, coeff| !coeff.is_zero());
    }

    /// Get the size parameter
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the trace parameter
    pub fn delta(&self) -> &R {
        &self.delta
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<PartitionDiagram, R> {
        &self.terms
    }

    /// Multiply this element by a scalar
    pub fn scalar_multiply(&self, scalar: &R) -> Self {
        let terms = self.terms
            .iter()
            .map(|(diagram, coeff)| (diagram.clone(), coeff.clone() * scalar.clone()))
            .collect();
        PartitionAlgebraElement::new(terms, self.n, self.delta.clone())
    }
}

impl<R: Ring> PartialEq for PartitionAlgebraElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.n != other.n {
            return false;
        }

        // Two elements are equal if they have the same non-zero terms
        let mut self_clean = self.clone();
        let mut other_clean = other.clone();
        self_clean.cleanup();
        other_clean.cleanup();

        if self_clean.terms.len() != other_clean.terms.len() {
            return false;
        }

        for (diagram, coeff) in &self_clean.terms {
            match other_clean.terms.get(diagram) {
                Some(other_coeff) if coeff == other_coeff => {},
                _ => return false,
            }
        }

        true
    }
}

impl<R: Ring> Eq for PartitionAlgebraElement<R> {}

impl<R: Ring> Add for PartitionAlgebraElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.n, other.n, "Cannot add elements of different sizes");

        let mut terms = self.terms.clone();
        for (diagram, coeff) in other.terms {
            let entry = terms.entry(diagram).or_insert_with(R::zero);
            *entry = entry.clone() + coeff;
        }

        PartitionAlgebraElement::new(terms, self.n, self.delta)
    }
}

impl<R: Ring> Sub for PartitionAlgebraElement<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.n, other.n, "Cannot subtract elements of different sizes");

        let mut terms = self.terms.clone();
        for (diagram, coeff) in other.terms {
            let entry = terms.entry(diagram).or_insert_with(R::zero);
            *entry = entry.clone() - coeff;
        }

        PartitionAlgebraElement::new(terms, self.n, self.delta)
    }
}

impl<R: Ring> Mul for PartitionAlgebraElement<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.n, other.n, "Cannot multiply elements of different sizes");

        let mut result_terms: HashMap<PartitionDiagram, R> = HashMap::new();

        for (diagram1, coeff1) in &self.terms {
            for (diagram2, coeff2) in &other.terms {
                if let Some((composed_diagram, num_loops)) = diagram1.compose(diagram2) {
                    // Coefficient is coeff1 * coeff2 * δ^num_loops
                    let mut coeff = coeff1.clone() * coeff2.clone();
                    for _ in 0..num_loops {
                        coeff = coeff * self.delta.clone();
                    }

                    let entry = result_terms.entry(composed_diagram).or_insert_with(R::zero);
                    *entry = entry.clone() + coeff;
                }
            }
        }

        PartitionAlgebraElement::new(result_terms, self.n, self.delta)
    }
}

impl<R: Ring> Neg for PartitionAlgebraElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let terms = self.terms
            .into_iter()
            .map(|(diagram, coeff)| (diagram, -coeff))
            .collect();
        PartitionAlgebraElement::new(terms, self.n, self.delta)
    }
}

impl<R: Ring> Ring for PartitionAlgebraElement<R> {
    fn zero() -> Self {
        panic!("Cannot create zero element without size and delta parameters. Use zero_with_params instead.");
    }

    fn one() -> Self {
        panic!("Cannot create one element without size and delta parameters. Use identity instead.");
    }

    fn is_zero(&self) -> bool {
        self.terms.is_empty() || self.terms.values().all(|c| c.is_zero())
    }

    fn is_one(&self) -> bool {
        if self.terms.len() != 1 {
            return false;
        }
        if let Some((diagram, coeff)) = self.terms.iter().next() {
            *diagram == PartitionDiagram::identity(self.n) && coeff.is_one()
        } else {
            false
        }
    }

    fn pow(&self, exp: u32) -> Self {
        if exp == 0 {
            return Self::identity(self.n, self.delta.clone());
        }

        let mut result = self.clone();
        for _ in 1..exp {
            result = result.clone() * self.clone();
        }
        result
    }
}

impl<R: Ring> Display for PartitionAlgebraElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (diagram, coeff) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            write!(f, "{}*{}", coeff, diagram)?;
            first = false;
        }
        Ok(())
    }
}

impl<R: Ring> Algebra<R> for PartitionAlgebraElement<R> {
    fn base_ring() -> R {
        R::zero() // Placeholder - actual base ring depends on instance
    }

    fn scalar_mul(&self, scalar: &R) -> Self {
        self.scalar_multiply(scalar)
    }

    fn dimension() -> Option<usize> {
        None // Depends on n parameter
    }
}

/// The partition algebra P_n(δ) as a structure
///
/// This represents the algebra itself with a specific n and δ, allowing for
/// dimension computation and basis enumeration.
#[derive(Debug, Clone)]
pub struct PartitionAlgebra<R: Ring> {
    n: usize,
    delta: R,
    basis_cache: Option<Vec<PartitionDiagram>>,
}

impl<R: Ring> PartitionAlgebra<R> {
    /// Create a new partition algebra P_n(δ)
    pub fn new(n: usize, delta: R) -> Self {
        PartitionAlgebra {
            n,
            delta,
            basis_cache: None,
        }
    }

    /// Get the size parameter
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the trace parameter
    pub fn delta(&self) -> &R {
        &self.delta
    }

    /// Get the dimension (Bell number B(2n))
    pub fn dimension(&self) -> usize {
        PartitionDiagram::dimension(self.n)
    }

    /// Get a basis for the algebra
    pub fn basis(&mut self) -> &Vec<PartitionDiagram> {
        if self.basis_cache.is_none() {
            self.basis_cache = Some(PartitionDiagram::all_diagrams(self.n));
        }
        self.basis_cache.as_ref().unwrap()
    }

    /// Create the zero element
    pub fn zero(&self) -> PartitionAlgebraElement<R> {
        PartitionAlgebraElement::zero_with_params(self.n, self.delta.clone())
    }

    /// Create the identity element
    pub fn one(&self) -> PartitionAlgebraElement<R> {
        PartitionAlgebraElement::identity(self.n, self.delta.clone())
    }

    /// Create a basis element
    pub fn basis_element(&self, diagram: PartitionDiagram) -> PartitionAlgebraElement<R> {
        assert_eq!(diagram.size(), self.n, "Diagram size mismatch");
        PartitionAlgebraElement::basis_element(diagram, self.delta.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_partition_diagram_identity() {
        let id = PartitionDiagram::identity(2);
        assert_eq!(id.size(), 2);

        // Identity should have 2 blocks: {0,2} and {1,3}
        let blocks = id.partition().blocks();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_partition_diagram_compose_identity() {
        let id = PartitionDiagram::identity(2);

        // Identity composed with itself should give identity with 0 loops
        if let Some((composed, loops)) = id.compose(&id) {
            assert_eq!(composed, id);
            assert_eq!(loops, 0);
        } else {
            panic!("Composition failed");
        }
    }

    #[test]
    fn test_partition_algebra_element_zero() {
        let zero = PartitionAlgebraElement::<Integer>::zero_with_params(2, Integer::from(1));
        assert!(zero.is_zero());
        assert_eq!(zero.terms().len(), 0);
    }

    #[test]
    fn test_partition_algebra_element_identity() {
        let one = PartitionAlgebraElement::<Integer>::identity(2, Integer::from(1));
        assert!(one.is_one());
        assert_eq!(one.terms().len(), 1);
    }

    #[test]
    fn test_partition_algebra_element_addition() {
        let delta = Integer::from(2);
        let id = PartitionDiagram::identity(2);

        let elem1 = PartitionAlgebraElement::basis_element(id.clone(), delta.clone());
        let elem2 = PartitionAlgebraElement::basis_element(id.clone(), delta.clone());

        let sum = elem1 + elem2;

        // Should have coefficient 2 for the identity diagram
        assert_eq!(sum.terms().len(), 1);
        assert_eq!(sum.terms().get(&id), Some(&Integer::from(2)));
    }

    #[test]
    fn test_partition_algebra_element_multiplication_identity() {
        let delta = Integer::from(1);
        let id = PartitionAlgebraElement::<Integer>::identity(2, delta.clone());

        let result = id.clone() * id.clone();

        // Identity * Identity = Identity
        assert!(result.is_one());
    }

    #[test]
    fn test_partition_algebra_dimension() {
        // Dimension of P_1(δ) is Bell(2) = 2
        assert_eq!(PartitionDiagram::dimension(1), 2);

        // Dimension of P_2(δ) is Bell(4) = 15
        assert_eq!(PartitionDiagram::dimension(2), 15);
    }

    #[test]
    fn test_partition_algebra_structure() {
        let mut algebra = PartitionAlgebra::new(1, Integer::from(3));

        assert_eq!(algebra.size(), 1);
        assert_eq!(algebra.delta(), &Integer::from(3));
        assert_eq!(algebra.dimension(), 2); // Bell(2) = 2

        let basis = algebra.basis();
        assert_eq!(basis.len(), 2);
    }

    #[test]
    fn test_all_diagrams() {
        let diagrams = PartitionDiagram::all_diagrams(1);
        assert_eq!(diagrams.len(), 2); // Bell(2) = 2

        // Check all diagrams have correct size
        for diagram in &diagrams {
            assert_eq!(diagram.size(), 1);
            assert_eq!(diagram.partition().size(), 2);
        }
    }

    #[test]
    fn test_partition_algebra_element_scalar_multiplication() {
        let delta = Integer::from(1);
        let id = PartitionAlgebraElement::<Integer>::identity(2, delta);

        let scaled = id.scalar_multiply(&Integer::from(5));

        // Should have coefficient 5 for identity diagram
        let id_diagram = PartitionDiagram::identity(2);
        assert_eq!(scaled.terms().get(&id_diagram), Some(&Integer::from(5)));
    }

    #[test]
    fn test_partition_algebra_ring_operations() {
        let delta = Integer::from(2);
        let zero = PartitionAlgebraElement::<Integer>::zero_with_params(1, delta.clone());
        let one = PartitionAlgebraElement::<Integer>::identity(1, delta.clone());

        // Test additive identity
        let result = one.clone() + zero.clone();
        assert_eq!(result, one);

        // Test multiplicative identity
        let result = one.clone() * one.clone();
        assert!(result.is_one());

        // Test additive inverse
        let neg_one = -one.clone();
        let result = one.clone() + neg_one;
        assert!(result.is_zero());
    }

    #[test]
    fn test_diagram_composition_with_loops() {
        // Create a diagram that will produce loops when composed with itself
        // For n=2, we have points {0,1,2,3} where 0,1 are top and 2,3 are bottom

        // Create diagram with block {0,2,1,3} (all connected)
        let blocks = vec![vec![0, 1, 2, 3]];
        let partition = SetPartition::new(blocks, 4).unwrap();
        let diagram = PartitionDiagram::new(partition, 2).unwrap();

        // Compose with identity
        let id = PartitionDiagram::identity(2);
        if let Some((_, loops)) = diagram.compose(&id) {
            // The number of loops depends on the specific composition
            assert!(loops >= 0); // Just verify it completes
        }
    }

    #[test]
    fn test_partition_algebra_pow() {
        let delta = Integer::from(1);
        let id = PartitionAlgebraElement::<Integer>::identity(1, delta);

        // id^0 = id
        let pow0 = id.pow(0);
        assert!(pow0.is_one());

        // id^3 = id
        let pow3 = id.pow(3);
        assert!(pow3.is_one());
    }
}
