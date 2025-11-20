//! Partition Algebra
//!
//! The partition algebra P_n(δ) is a diagram algebra where:
//! - Elements are linear combinations of all partition diagrams with n top and n bottom vertices
//! - Multiplication is diagram composition
//! - Closed loops evaluate to a parameter δ
//!
//! The partition algebra contains the Brauer algebra as a subalgebra (those partitions
//! where each block has size at most 2).
//!
//! Applications:
//! - Representation theory of symmetric groups
//! - Invariant theory
//! - Statistical mechanics
//! - Schur-Weyl duality generalizations
//!
//! References:
//! - Martin, P. "The structure of the partition algebras" (1996)
//! - Halverson, T. and Ram, A. "Partition algebras" (2005)

use rustmath_core::{Ring, MathError, Result};
use rustmath_modules::CombinatorialFreeModuleElement;
use rustmath_combinatorics::SetPartition;
use crate::diagram::PartitionDiagram;
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::collections::BTreeSet;

/// The partition algebra P_n(δ) over a ring R
///
/// # Type Parameters
///
/// * `R` - The base ring (coefficient ring)
///
/// # Examples
///
/// ```
/// use rustmath_algebras::partition_algebra::PartitionAlgebra;
/// use rustmath_integers::Integer;
///
/// // Create the partition algebra P_3(2) over integers
/// let algebra = PartitionAlgebra::<Integer>::new(3, Integer::from(2));
/// ```
#[derive(Clone, Debug)]
pub struct PartitionAlgebra<R: Ring> {
    /// Order of the algebra (number of vertices per side)
    order: usize,
    /// The parameter δ (value of closed loops)
    delta: R,
    /// Phantom data for ring type
    _phantom: PhantomData<R>,
}

impl<R: Ring> PartitionAlgebra<R> {
    /// Create a new partition algebra
    ///
    /// # Arguments
    ///
    /// * `order` - Number of vertices on each side (must be positive)
    /// * `delta` - The loop parameter δ
    ///
    /// # Returns
    ///
    /// A new partition algebra P_n(δ)
    pub fn new(order: usize, delta: R) -> Self {
        assert!(order > 0, "Order must be positive");
        PartitionAlgebra {
            order,
            delta,
            _phantom: PhantomData,
        }
    }

    /// Get the order of the algebra
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the delta parameter
    pub fn delta(&self) -> &R {
        &self.delta
    }

    /// Get the identity element
    pub fn one(&self) -> PartitionElement<R> {
        let identity_diagram = PartitionDiagram::identity(self.order);
        PartitionElement {
            module_element: CombinatorialFreeModuleElement::from_basis_index(identity_diagram),
        }
    }

    /// Get the zero element
    pub fn zero(&self) -> PartitionElement<R> {
        PartitionElement {
            module_element: CombinatorialFreeModuleElement::zero(),
        }
    }

    /// Create an element from a single partition diagram
    ///
    /// # Arguments
    ///
    /// * `diagram` - The partition diagram
    /// * `coefficient` - The coefficient (usually 1)
    pub fn from_diagram(&self, diagram: PartitionDiagram, coefficient: R) -> PartitionElement<R> {
        PartitionElement {
            module_element: CombinatorialFreeModuleElement::monomial(diagram, coefficient),
        }
    }

    /// Create a partition diagram from a set partition
    ///
    /// The set partition represents how vertices {0, ..., 2n-1} are partitioned.
    ///
    /// # Arguments
    ///
    /// * `partition` - Set partition of 2n elements
    pub fn diagram_from_set_partition(&self, partition: &SetPartition) -> Result<PartitionDiagram> {
        let n = self.order;

        if partition.num_elements() != 2 * n {
            return Err(MathError::InvalidArgument(
                format!("Set partition must have {} elements, got {}",
                    2 * n, partition.num_elements())
            ));
        }

        let blocks: Vec<BTreeSet<usize>> = partition.blocks().iter()
            .map(|block| block.iter().copied().collect())
            .collect();

        PartitionDiagram::new(n, blocks)
    }

    /// Multiply two basis elements (diagrams) and evaluate loops
    ///
    /// # Arguments
    ///
    /// * `diagram1` - First diagram (on top)
    /// * `diagram2` - Second diagram (on bottom)
    ///
    /// # Returns
    ///
    /// The product as a partition algebra element
    pub fn product_on_basis(&self, diagram1: &PartitionDiagram, diagram2: &PartitionDiagram) -> PartitionElement<R> {
        // Compose diagrams
        let composed = match diagram1.compose(diagram2) {
            Ok(d) => d,
            Err(_) => return self.zero(),
        };

        // Count loops
        let loops_count = self.count_loops_in_composition(diagram1, diagram2);

        // Compute coefficient with δ^loops_count
        let mut coefficient = R::one();
        for _ in 0..loops_count {
            coefficient = coefficient * self.delta.clone();
        }

        PartitionElement {
            module_element: CombinatorialFreeModuleElement::monomial(composed, coefficient),
        }
    }

    /// Count closed loops in composition
    fn count_loops_in_composition(&self, diagram1: &PartitionDiagram, diagram2: &PartitionDiagram) -> usize {
        let n = self.order;
        let mut loops = 0;

        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
            let root_x = find(parent, x);
            let root_y = find(parent, y);
            if root_x != root_y {
                parent[root_x] = root_y;
            }
        }

        // Process diagram1 bottom vertices
        for block in diagram1.blocks() {
            let bottom_vertices: Vec<usize> = block.iter()
                .filter(|&&v| v >= n && v < 2 * n)
                .map(|&v| v - n)
                .collect();

            for i in 1..bottom_vertices.len() {
                union(&mut parent, bottom_vertices[0], bottom_vertices[i]);
            }
        }

        // Process diagram2 top vertices
        for block in diagram2.blocks() {
            let top_vertices: Vec<usize> = block.iter()
                .filter(|&&v| v < n)
                .copied()
                .collect();

            for i in 1..top_vertices.len() {
                union(&mut parent, top_vertices[0], top_vertices[i]);
            }
        }

        // Check which components touch outside
        let mut component_touches_outside: std::collections::HashMap<usize, bool> =
            std::collections::HashMap::new();

        for block in diagram1.blocks() {
            let has_top = block.iter().any(|&v| v < n);
            let bottom_vertices: Vec<usize> = block.iter()
                .filter(|&&v| v >= n && v < 2 * n)
                .map(|&v| v - n)
                .collect();

            if has_top {
                for &v in &bottom_vertices {
                    let root = find(&mut parent, v);
                    component_touches_outside.insert(root, true);
                }
            }
        }

        for block in diagram2.blocks() {
            let has_bottom = block.iter().any(|&v| v >= n);
            let top_vertices: Vec<usize> = block.iter()
                .filter(|&&v| v < n)
                .copied()
                .collect();

            if has_bottom {
                for &v in &top_vertices {
                    let root = find(&mut parent, v);
                    component_touches_outside.insert(root, true);
                }
            }
        }

        // Count loops
        let mut seen_roots = std::collections::HashSet::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            if !seen_roots.contains(&root) {
                seen_roots.insert(root);
                if !component_touches_outside.get(&root).unwrap_or(&false) {
                    loops += 1;
                }
            }
        }

        loops
    }

    /// Get the dimension of the algebra
    ///
    /// The dimension is the Bell number B_{2n} (number of partitions of a 2n-element set).
    pub fn dimension_upper_bound(&self) -> usize {
        // This is the exact dimension for generic δ
        // Computing large Bell numbers is expensive, so we provide an upper bound
        bell_number_upper_bound(2 * self.order)
    }
}

/// Compute an upper bound for the nth Bell number
///
/// Bell numbers grow very quickly: B_0=1, B_1=1, B_2=2, B_3=5, B_4=15, B_5=52, ...
/// Exact formula: B_n = sum over k of S(n,k) where S(n,k) is Stirling number of second kind
fn bell_number_upper_bound(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 1;
    }
    if n == 2 {
        return 2;
    }
    if n == 3 {
        return 5;
    }
    if n == 4 {
        return 15;
    }
    if n == 5 {
        return 52;
    }
    if n == 6 {
        return 203;
    }

    // For larger n, use the upper bound: B_n < n^n
    // More accurate would be to compute via dynamic programming, but that's expensive
    n.saturating_pow(n as u32)
}

/// An element of the partition algebra
///
/// Represented as a linear combination of partition diagrams
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionElement<R: Ring> {
    /// The underlying combinatorial free module element
    pub module_element: CombinatorialFreeModuleElement<R, PartitionDiagram>,
}

impl<R: Ring> PartitionElement<R> {
    /// Create the zero element
    pub fn zero() -> Self {
        PartitionElement {
            module_element: CombinatorialFreeModuleElement::zero(),
        }
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.module_element.is_zero()
    }

    /// Add two elements
    pub fn add(&self, other: &PartitionElement<R>) -> PartitionElement<R> {
        PartitionElement {
            module_element: self.module_element.clone() + other.module_element.clone(),
        }
    }

    /// Subtract two elements
    pub fn sub(&self, other: &PartitionElement<R>) -> PartitionElement<R> {
        PartitionElement {
            module_element: self.module_element.clone() - other.module_element.clone(),
        }
    }

    /// Negate this element
    pub fn neg(&self) -> PartitionElement<R> {
        PartitionElement {
            module_element: -self.module_element.clone(),
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> PartitionElement<R> {
        PartitionElement {
            module_element: self.module_element.scalar_mul(scalar),
        }
    }

    /// Multiply two algebra elements
    pub fn multiply(&self, other: &PartitionElement<R>, algebra: &PartitionAlgebra<R>) -> PartitionElement<R> {
        let mut result = PartitionElement::zero();

        for (diagram1, coeff1) in self.module_element.iter() {
            for (diagram2, coeff2) in other.module_element.iter() {
                let basis_product = algebra.product_on_basis(diagram1, diagram2);
                let coeff_product = coeff1.clone() * coeff2.clone();
                let term = basis_product.scalar_mul(&coeff_product);

                result = result.add(&term);
            }
        }

        result
    }

    /// Get the number of terms in the element
    pub fn num_terms(&self) -> usize {
        self.module_element.iter().count()
    }
}

impl<R: Ring + Display> Display for PartitionElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let terms: Vec<_> = self.module_element.iter().collect();

        for (i, (diagram, coeff)) in terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }

            if coeff.is_one() {
                write!(f, "{}", diagram)?;
            } else {
                write!(f, "{}*{}", coeff, diagram)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_partition_algebra_creation() {
        let algebra = PartitionAlgebra::<Integer>::new(3, Integer::from(2));
        assert_eq!(algebra.order(), 3);
        assert_eq!(*algebra.delta(), Integer::from(2));
    }

    #[test]
    fn test_identity_element() {
        let algebra = PartitionAlgebra::<Integer>::new(3, Integer::from(2));
        let one = algebra.one();
        assert!(!one.is_zero());
        assert_eq!(one.num_terms(), 1);
    }

    #[test]
    fn test_zero_element() {
        let algebra = PartitionAlgebra::<Integer>::new(3, Integer::from(2));
        let zero = algebra.zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_identity_multiplication() {
        let algebra = PartitionAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();

        let result = one.multiply(&one, &algebra);
        assert_eq!(result, one);
    }

    #[test]
    fn test_diagram_element() {
        let algebra = PartitionAlgebra::<Integer>::new(2, Integer::from(2));
        let diagram = PartitionDiagram::identity(2);
        let element = algebra.from_diagram(diagram, Integer::from(1));

        assert!(!element.is_zero());
        assert_eq!(element.num_terms(), 1);
    }

    #[test]
    fn test_addition() {
        let algebra = PartitionAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();
        let zero = algebra.zero();

        let sum = one.add(&zero);
        assert_eq!(sum, one);
    }

    #[test]
    fn test_scalar_multiplication() {
        let algebra = PartitionAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();
        let scalar = Integer::from(3);

        let scaled = one.scalar_mul(&scalar);
        assert!(!scaled.is_zero());
    }

    #[test]
    fn test_negation() {
        let algebra = PartitionAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();
        let neg_one = one.neg();

        let sum = one.add(&neg_one);
        assert!(sum.is_zero());
    }

    #[test]
    fn test_bell_numbers() {
        assert_eq!(bell_number_upper_bound(0), 1);
        assert_eq!(bell_number_upper_bound(1), 1);
        assert_eq!(bell_number_upper_bound(2), 2);
        assert_eq!(bell_number_upper_bound(3), 5);
        assert_eq!(bell_number_upper_bound(4), 15);
        assert_eq!(bell_number_upper_bound(5), 52);
    }

    #[test]
    fn test_dimension_bounds() {
        let p1 = PartitionAlgebra::<Integer>::new(1, Integer::from(2));
        assert!(p1.dimension_upper_bound() >= 2); // B_2 = 2

        let p2 = PartitionAlgebra::<Integer>::new(2, Integer::from(2));
        assert!(p2.dimension_upper_bound() >= 15); // B_4 = 15
    }
}
