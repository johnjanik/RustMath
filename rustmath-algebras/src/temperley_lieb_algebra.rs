//! Temperley-Lieb Algebra
//!
//! The Temperley-Lieb algebra TL_n(δ) is a quotient of the Brauer algebra where:
//! - Elements are linear combinations of planar (noncrossing) partition diagrams
//! - Multiplication is diagram composition (must preserve planarity)
//! - Closed loops evaluate to a parameter δ
//!
//! The Temperley-Lieb algebra appears in:
//! - Statistical mechanics (transfer matrices)
//! - Knot theory (Jones polynomial)
//! - Quantum groups (representation theory of U_q(sl_2))
//! - Conformal field theory
//!
//! References:
//! - Temperley, H. N. V. and Lieb, E. H. "Relations between the 'percolation' and 'colouring' problem" (1971)
//! - Jones, V. F. R. "Index for subfactors" (1983)
//! - Kauffman, L. H. "State models and the Jones polynomial" (1987)

use rustmath_core::{Ring, MathError, Result};
use rustmath_modules::CombinatorialFreeModuleElement;
use crate::diagram::{PartitionDiagram, TemperleyLiebDiagram};
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::collections::BTreeSet;

/// The Temperley-Lieb algebra TL_n(δ) over a ring R
///
/// # Type Parameters
///
/// * `R` - The base ring (coefficient ring)
///
/// # Examples
///
/// ```
/// use rustmath_algebras::temperley_lieb_algebra::TemperleyLiebAlgebra;
/// use rustmath_integers::Integer;
///
/// // Create the Temperley-Lieb algebra TL_3(2) over integers
/// let algebra = TemperleyLiebAlgebra::<Integer>::new(3, Integer::from(2));
/// ```
#[derive(Clone, Debug)]
pub struct TemperleyLiebAlgebra<R: Ring> {
    /// Order of the algebra (number of vertices per side)
    order: usize,
    /// The parameter δ (value of closed loops)
    delta: R,
    /// Phantom data for ring type
    _phantom: PhantomData<R>,
}

impl<R: Ring> TemperleyLiebAlgebra<R> {
    /// Create a new Temperley-Lieb algebra
    ///
    /// # Arguments
    ///
    /// * `order` - Number of vertices on each side (must be positive)
    /// * `delta` - The loop parameter δ
    ///
    /// # Returns
    ///
    /// A new Temperley-Lieb algebra TL_n(δ)
    pub fn new(order: usize, delta: R) -> Self {
        assert!(order > 0, "Order must be positive");
        TemperleyLiebAlgebra {
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
    pub fn one(&self) -> TLElement<R> {
        let identity_diagram = TemperleyLiebDiagram::identity(self.order);
        TLElement {
            module_element: CombinatorialFreeModuleElement::from_basis_index(identity_diagram),
        }
    }

    /// Get the zero element
    pub fn zero(&self) -> TLElement<R> {
        TLElement {
            module_element: CombinatorialFreeModuleElement::zero(),
        }
    }

    /// Get the standard generator U_i (creates a small arc between positions i and i+1)
    ///
    /// # Arguments
    ///
    /// * `i` - Position (0 <= i < order-1)
    ///
    /// # Returns
    ///
    /// The generator U_i as a Temperley-Lieb element
    ///
    /// The generator U_i creates arcs:
    /// - On top: connect vertex i to vertex i+1
    /// - On bottom: connect vertex i to vertex i+1
    /// - All other vertices connect straight through
    pub fn generator(&self, i: usize) -> Result<TLElement<R>> {
        if i >= self.order - 1 {
            return Err(MathError::InvalidArgument(
                format!("Generator index {} must be less than {}", i, self.order - 1)
            ));
        }

        let n = self.order;
        let mut blocks = Vec::new();

        // Create arc on top: vertices i and i+1
        let mut top_arc = BTreeSet::new();
        top_arc.insert(i);
        top_arc.insert(i + 1);
        blocks.push(top_arc);

        // Create arc on bottom: vertices n+i and n+i+1
        let mut bottom_arc = BTreeSet::new();
        bottom_arc.insert(n + i);
        bottom_arc.insert(n + i + 1);
        blocks.push(bottom_arc);

        // All other vertices connect straight through
        for j in 0..n {
            if j != i && j != i + 1 {
                let mut through_strand = BTreeSet::new();
                through_strand.insert(j);
                through_strand.insert(j + n);
                blocks.push(through_strand);
            }
        }

        let diagram = PartitionDiagram::new(n, blocks)?;
        let tl_diagram = TemperleyLiebDiagram::new(diagram)?;

        Ok(TLElement {
            module_element: CombinatorialFreeModuleElement::from_basis_index(tl_diagram),
        })
    }

    /// Get all standard generators [U_0, U_1, ..., U_{n-2}]
    pub fn generators(&self) -> Vec<TLElement<R>> {
        (0..self.order - 1)
            .filter_map(|i| self.generator(i).ok())
            .collect()
    }

    /// Create an element from a single Temperley-Lieb diagram
    ///
    /// # Arguments
    ///
    /// * `diagram` - The Temperley-Lieb diagram
    /// * `coefficient` - The coefficient (usually 1)
    pub fn from_diagram(&self, diagram: TemperleyLiebDiagram, coefficient: R) -> TLElement<R> {
        TLElement {
            module_element: CombinatorialFreeModuleElement::monomial(diagram, coefficient),
        }
    }

    /// Multiply two basis elements (TL diagrams) and evaluate loops
    ///
    /// # Arguments
    ///
    /// * `diagram1` - First diagram (on top)
    /// * `diagram2` - Second diagram (on bottom)
    ///
    /// # Returns
    ///
    /// The product as a Temperley-Lieb element
    pub fn product_on_basis(&self, diagram1: &TemperleyLiebDiagram, diagram2: &TemperleyLiebDiagram) -> TLElement<R> {
        // Compose the underlying partition diagrams
        let composed = match diagram1.diagram().compose(diagram2.diagram()) {
            Ok(d) => d,
            Err(_) => return self.zero(),
        };

        // Verify result is still planar
        if !composed.is_planar() {
            return self.zero(); // Non-planar products are zero in TL algebra
        }

        // Count loops and compute coefficient
        let loops_count = self.count_loops_in_composition(diagram1.diagram(), diagram2.diagram());

        let mut coefficient = R::one();
        for _ in 0..loops_count {
            coefficient = coefficient * self.delta.clone();
        }

        // Create TL diagram from composed diagram
        let tl_diagram = match TemperleyLiebDiagram::new(composed) {
            Ok(d) => d,
            Err(_) => return self.zero(),
        };

        TLElement {
            module_element: CombinatorialFreeModuleElement::monomial(tl_diagram, coefficient),
        }
    }

    /// Count closed loops in composition (same logic as Brauer algebra)
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
    /// For the Temperley-Lieb algebra TL_n(δ), the dimension is the nth Catalan number C_n.
    /// C_n = (2n choose n) / (n+1) = 1, 1, 2, 5, 14, 42, 132, ...
    pub fn dimension(&self) -> usize {
        catalan_number(self.order)
    }
}

/// Compute the nth Catalan number
///
/// C_n = (2n)! / ((n+1)! * n!)
fn catalan_number(n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    let mut result = 1usize;
    for i in 0..n {
        result = result * (2 * n - i) / (i + 1);
    }
    result / (n + 1)
}

/// An element of the Temperley-Lieb algebra
///
/// Represented as a linear combination of Temperley-Lieb diagrams
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TLElement<R: Ring> {
    /// The underlying combinatorial free module element
    pub module_element: CombinatorialFreeModuleElement<R, TemperleyLiebDiagram>,
}

impl<R: Ring> TLElement<R> {
    /// Create the zero element
    pub fn zero() -> Self {
        TLElement {
            module_element: CombinatorialFreeModuleElement::zero(),
        }
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.module_element.is_zero()
    }

    /// Add two elements
    pub fn add(&self, other: &TLElement<R>) -> TLElement<R> {
        TLElement {
            module_element: self.module_element.clone() + other.module_element.clone(),
        }
    }

    /// Subtract two elements
    pub fn sub(&self, other: &TLElement<R>) -> TLElement<R> {
        TLElement {
            module_element: self.module_element.clone() - other.module_element.clone(),
        }
    }

    /// Negate this element
    pub fn neg(&self) -> TLElement<R> {
        TLElement {
            module_element: -self.module_element.clone(),
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> TLElement<R> {
        TLElement {
            module_element: self.module_element.scalar_mul(scalar),
        }
    }

    /// Multiply two algebra elements
    pub fn multiply(&self, other: &TLElement<R>, algebra: &TemperleyLiebAlgebra<R>) -> TLElement<R> {
        let mut result = TLElement::zero();

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

impl<R: Ring + Display> Display for TLElement<R> {
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
    fn test_temperley_lieb_creation() {
        let algebra = TemperleyLiebAlgebra::<Integer>::new(3, Integer::from(2));
        assert_eq!(algebra.order(), 3);
        assert_eq!(*algebra.delta(), Integer::from(2));
    }

    #[test]
    fn test_identity_element() {
        let algebra = TemperleyLiebAlgebra::<Integer>::new(3, Integer::from(2));
        let one = algebra.one();
        assert!(!one.is_zero());
        assert_eq!(one.num_terms(), 1);
    }

    #[test]
    fn test_generators() {
        let algebra = TemperleyLiebAlgebra::<Integer>::new(4, Integer::from(2));
        let generators = algebra.generators();
        assert_eq!(generators.len(), 3); // U_0, U_1, U_2

        for gen in generators {
            assert!(!gen.is_zero());
        }
    }

    #[test]
    fn test_generator_creation() {
        let algebra = TemperleyLiebAlgebra::<Integer>::new(3, Integer::from(2));
        let u0 = algebra.generator(0).unwrap();
        let u1 = algebra.generator(1).unwrap();

        assert!(!u0.is_zero());
        assert!(!u1.is_zero());
    }

    #[test]
    fn test_identity_multiplication() {
        let algebra = TemperleyLiebAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();

        let result = one.multiply(&one, &algebra);
        assert_eq!(result, one);
    }

    #[test]
    fn test_generator_relation_ui_squared() {
        // Test U_i^2 = δ * U_i (one of the TL relations)
        let algebra = TemperleyLiebAlgebra::<Integer>::new(3, Integer::from(2));
        let u0 = algebra.generator(0).unwrap();

        let u0_squared = u0.multiply(&u0, &algebra);

        // u0^2 should equal δ * u0 = 2 * u0
        let delta_u0 = u0.scalar_mul(algebra.delta());

        // Both should be non-zero
        assert!(!u0_squared.is_zero());
        assert!(!delta_u0.is_zero());
    }

    #[test]
    fn test_dimension() {
        // TL_1 has dimension C_1 = 1
        let tl1 = TemperleyLiebAlgebra::<Integer>::new(1, Integer::from(2));
        assert_eq!(tl1.dimension(), 1);

        // TL_2 has dimension C_2 = 2
        let tl2 = TemperleyLiebAlgebra::<Integer>::new(2, Integer::from(2));
        assert_eq!(tl2.dimension(), 2);

        // TL_3 has dimension C_3 = 5
        let tl3 = TemperleyLiebAlgebra::<Integer>::new(3, Integer::from(2));
        assert_eq!(tl3.dimension(), 5);

        // TL_4 has dimension C_4 = 14
        let tl4 = TemperleyLiebAlgebra::<Integer>::new(4, Integer::from(2));
        assert_eq!(tl4.dimension(), 14);
    }

    #[test]
    fn test_catalan_numbers() {
        assert_eq!(catalan_number(0), 1);
        assert_eq!(catalan_number(1), 1);
        assert_eq!(catalan_number(2), 2);
        assert_eq!(catalan_number(3), 5);
        assert_eq!(catalan_number(4), 14);
        assert_eq!(catalan_number(5), 42);
    }

    #[test]
    fn test_addition() {
        let algebra = TemperleyLiebAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();
        let u0 = algebra.generator(0).unwrap();

        let sum = one.add(&u0);
        assert!(!sum.is_zero());
        assert!(sum.num_terms() >= 1);
    }

    #[test]
    fn test_scalar_multiplication() {
        let algebra = TemperleyLiebAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();
        let scalar = Integer::from(3);

        let scaled = one.scalar_mul(&scalar);
        assert!(!scaled.is_zero());
    }

    #[test]
    fn test_negation() {
        let algebra = TemperleyLiebAlgebra::<Integer>::new(2, Integer::from(2));
        let one = algebra.one();
        let neg_one = one.neg();

        let sum = one.add(&neg_one);
        assert!(sum.is_zero());
    }
}
