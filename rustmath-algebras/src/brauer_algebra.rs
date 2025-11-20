//! Brauer Algebra
//!
//! The Brauer algebra B_n(δ) is a diagram algebra where:
//! - Elements are linear combinations of partition diagrams with n top and n bottom vertices
//! - Multiplication is diagram composition
//! - Closed loops evaluate to a parameter δ
//!
//! The Brauer algebra is fundamental in representation theory and has applications
//! in invariant theory, knot theory, and statistical mechanics.
//!
//! References:
//! - Brauer, R. "On algebras which are connected with the semisimple continuous groups" (1937)
//! - Martin, P. "The structure of the partition algebras" (1996)

use rustmath_core::{Ring, MathError, Result};
use rustmath_modules::CombinatorialFreeModuleElement;
use crate::diagram::{PartitionDiagram, BrauerDiagram};
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// The Brauer algebra B_n(δ) over a ring R
///
/// # Type Parameters
///
/// * `R` - The base ring (coefficient ring)
///
/// # Examples
///
/// ```
/// use rustmath_algebras::brauer_algebra::BrauerAlgebra;
/// use rustmath_integers::Integer;
///
/// // Create the Brauer algebra B_3(2) over integers
/// let algebra = BrauerAlgebra::<Integer>::new(3, Integer::from(2));
/// ```
#[derive(Clone, Debug)]
pub struct BrauerAlgebra<R: Ring> {
    /// Order of the algebra (number of vertices per side)
    order: usize,
    /// The parameter δ (value of closed loops)
    delta: R,
    /// Phantom data for ring type
    _phantom: PhantomData<R>,
}

impl<R: Ring> BrauerAlgebra<R> {
    /// Create a new Brauer algebra
    ///
    /// # Arguments
    ///
    /// * `order` - Number of vertices on each side (must be positive)
    /// * `delta` - The loop parameter δ
    ///
    /// # Returns
    ///
    /// A new Brauer algebra B_n(δ)
    pub fn new(order: usize, delta: R) -> Self {
        assert!(order > 0, "Order must be positive");
        BrauerAlgebra {
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
    pub fn one(&self) -> BrauerElement<R> {
        let identity_diagram = PartitionDiagram::identity(self.order);
        BrauerElement {
            module_element: CombinatorialFreeModuleElement::from_basis_index(identity_diagram),
        }
    }

    /// Get the zero element
    pub fn zero(&self) -> BrauerElement<R> {
        BrauerElement {
            module_element: CombinatorialFreeModuleElement::zero(),
        }
    }

    /// Create an element from a single diagram
    ///
    /// # Arguments
    ///
    /// * `diagram` - The partition diagram
    /// * `coefficient` - The coefficient (usually 1)
    pub fn from_diagram(&self, diagram: PartitionDiagram, coefficient: R) -> BrauerElement<R> {
        BrauerElement {
            module_element: CombinatorialFreeModuleElement::monomial(diagram, coefficient),
        }
    }

    /// Multiply two basis elements (diagrams) and evaluate loops
    ///
    /// When composing diagrams, closed loops in the middle are removed
    /// and contribute a factor of δ for each loop.
    ///
    /// # Arguments
    ///
    /// * `diagram1` - First diagram (on top)
    /// * `diagram2` - Second diagram (on bottom)
    ///
    /// # Returns
    ///
    /// The product as a Brauer algebra element
    pub fn product_on_basis(&self, diagram1: &BrauerDiagram, diagram2: &BrauerDiagram) -> BrauerElement<R> {
        // Compose diagrams (this may create loops in the middle)
        let composed = match diagram1.compose(diagram2) {
            Ok(d) => d,
            Err(_) => return self.zero(), // Composition failed
        };

        // Count loops created in the middle
        let loops_count = self.count_loops_in_composition(diagram1, diagram2);

        // Each loop contributes a factor of δ
        let mut coefficient = R::one();
        for _ in 0..loops_count {
            coefficient = coefficient * self.delta.clone();
        }

        BrauerElement {
            module_element: CombinatorialFreeModuleElement::monomial(composed, coefficient),
        }
    }

    /// Count the number of closed loops created in the middle when composing diagrams
    ///
    /// When we compose two diagrams, the bottom vertices of diagram1 are identified
    /// with the top vertices of diagram2. Any blocks that only touch these middle
    /// vertices form closed loops.
    fn count_loops_in_composition(&self, diagram1: &BrauerDiagram, diagram2: &BrauerDiagram) -> usize {
        let n = self.order;
        let mut loops = 0;

        // We need to track which middle vertices are connected
        // Middle vertices are: bottom of diagram1 (n..2n-1) = top of diagram2 (0..n-1)

        // Create a union-find structure for middle vertices
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

        // Process diagram1: find which bottom vertices are connected
        for block in diagram1.blocks() {
            let bottom_vertices: Vec<usize> = block.iter()
                .filter(|&&v| v >= n && v < 2 * n)
                .map(|&v| v - n)
                .collect();

            for i in 1..bottom_vertices.len() {
                union(&mut parent, bottom_vertices[0], bottom_vertices[i]);
            }
        }

        // Process diagram2: find which top vertices are connected
        for block in diagram2.blocks() {
            let top_vertices: Vec<usize> = block.iter()
                .filter(|&&v| v < n)
                .copied()
                .collect();

            for i in 1..top_vertices.len() {
                union(&mut parent, top_vertices[0], top_vertices[i]);
            }
        }

        // Check each connected component: if it doesn't connect to
        // top of diagram1 or bottom of diagram2, it's a loop
        let mut component_touches_outside: std::collections::HashMap<usize, bool> =
            std::collections::HashMap::new();

        // Check if components touch top vertices of diagram1
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

        // Check if components touch bottom vertices of diagram2
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

        // Count components that don't touch outside (these are loops)
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

    /// Get the dimension of the algebra over the base ring
    ///
    /// The dimension is the number of partition diagrams on 2n vertices,
    /// which is the Bell number B_{2n}.
    pub fn dimension_upper_bound(&self) -> usize {
        // This is an upper bound; exact dimension depends on δ
        // For generic δ, dimension is the (2n-1)!! (double factorial)
        let n = self.order;
        if n == 0 {
            return 1;
        }

        // (2n)! / (2^n * n!) = (2n-1)!! for number of perfect matchings on 2n vertices
        // But Brauer algebra can have more general partitions
        // Exact dimension is sum of partition diagrams with k propagating strands
        // for k = 0, 2, 4, ..., 2*min(n,n) with specific counting

        // Simplified: return an upper bound
        // Exact formula involves Catalan numbers and binomial coefficients
        2_usize.pow((2 * n) as u32) // Very rough upper bound
    }
}

/// An element of the Brauer algebra
///
/// Represented as a linear combination of Brauer diagrams
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrauerElement<R: Ring> {
    /// The underlying combinatorial free module element
    pub module_element: CombinatorialFreeModuleElement<R, BrauerDiagram>,
}

impl<R: Ring> BrauerElement<R> {
    /// Create the zero element
    pub fn zero() -> Self {
        BrauerElement {
            module_element: CombinatorialFreeModuleElement::zero(),
        }
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.module_element.is_zero()
    }

    /// Add two elements
    pub fn add(&self, other: &BrauerElement<R>) -> BrauerElement<R> {
        BrauerElement {
            module_element: self.module_element.clone() + other.module_element.clone(),
        }
    }

    /// Subtract two elements
    pub fn sub(&self, other: &BrauerElement<R>) -> BrauerElement<R> {
        BrauerElement {
            module_element: self.module_element.clone() - other.module_element.clone(),
        }
    }

    /// Negate this element
    pub fn neg(&self) -> BrauerElement<R> {
        BrauerElement {
            module_element: -self.module_element.clone(),
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> BrauerElement<R> {
        BrauerElement {
            module_element: self.module_element.scalar_mul(scalar),
        }
    }

    /// Multiply two algebra elements
    ///
    /// Uses bilinearity: (Σ a_i d_i) * (Σ b_j d_j) = Σ a_i b_j (d_i * d_j)
    pub fn multiply(&self, other: &BrauerElement<R>, algebra: &BrauerAlgebra<R>) -> BrauerElement<R> {
        let mut result = BrauerElement::zero();

        for (diagram1, coeff1) in self.module_element.iter() {
            for (diagram2, coeff2) in other.module_element.iter() {
                // Compute product of basis elements
                let basis_product = algebra.product_on_basis(diagram1, diagram2);

                // Multiply by coefficients
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

impl<R: Ring + Display> Display for BrauerElement<R> {
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
    use crate::diagram::PartitionDiagram;

    #[test]
    fn test_brauer_algebra_creation() {
        let algebra = BrauerAlgebra::<Integer>::new(3, Integer::from(2));
        assert_eq!(algebra.order(), 3);
        assert_eq!(*algebra.delta(), Integer::from(2));
    }

    #[test]
    fn test_identity_element() {
        let algebra = BrauerAlgebra::<Integer>::new(3, Integer::from(2));
        let one = algebra.one();
        assert!(!one.is_zero());
        assert_eq!(one.num_terms(), 1);
    }

    #[test]
    fn test_zero_element() {
        let algebra = BrauerAlgebra::<Integer>::new(3, Integer::from(2));
        let zero = algebra.zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_identity_multiplication() {
        let algebra = BrauerAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();

        let result = one.multiply(&one, &algebra);
        assert_eq!(result, one);
    }

    #[test]
    fn test_diagram_element() {
        let algebra = BrauerAlgebra::<Integer>::new(2, Integer::from(2));
        let diagram = PartitionDiagram::identity(2);
        let element = algebra.from_diagram(diagram, Integer::from(1));

        assert!(!element.is_zero());
        assert_eq!(element.num_terms(), 1);
    }

    #[test]
    fn test_addition() {
        let algebra = BrauerAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();
        let zero = algebra.zero();

        let sum = one.add(&zero);
        assert_eq!(sum, one);
    }

    #[test]
    fn test_scalar_multiplication() {
        let algebra = BrauerAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();
        let scalar = Integer::from(3);

        let scaled = one.scalar_mul(&scalar);
        assert!(!scaled.is_zero());
    }

    #[test]
    fn test_product_with_loops() {
        // Create a diagram that will produce loops when composed with itself
        let algebra = BrauerAlgebra::<Integer>::new(2, Integer::from(2));

        // Create a diagram with a "cap" on top: (0,1), (2), (3)
        let cap_diagram = PartitionDiagram::from_matching(2, vec![(0, 1)]).unwrap();

        // Create a diagram with a "cup" on bottom: (0), (1), (2,3)
        let cup_diagram = PartitionDiagram::from_matching(2, vec![(2, 3)]).unwrap();

        let cap = algebra.from_diagram(cap_diagram, Integer::from(1));
        let cup = algebra.from_diagram(cup_diagram, Integer::from(1));

        // When we compose cap on top of cup, we should get loops
        let product = cap.multiply(&cup, &algebra);

        // The result should have δ factors from loops
        assert!(!product.is_zero());
    }

    #[test]
    fn test_negation() {
        let algebra = BrauerAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();
        let neg_one = one.neg();

        let sum = one.add(&neg_one);
        assert!(sum.is_zero());
    }
}
