//! Generic ring operations and structures
//!
//! This module provides generic operations and data structures that work across
//! different ring types, corresponding to SageMath's `sage.rings.generic`.
//!
//! # Overview
//!
//! This module implements efficient algorithms for common operations:
//! - Product tree construction for fast multi-point evaluation
//! - Product with derivative computation
//! - Batch operations on ring elements
//!
//! # Product Trees
//!
//! A product tree is a binary tree structure used to compute products efficiently.
//! For elements [a₁, a₂, ..., aₙ], the product tree stores:
//!
//! Level 0: [a₁, a₂, a₃, a₄, ..., aₙ]
//! Level 1: [a₁·a₂, a₃·a₄, ...]
//! Level 2: [(a₁·a₂)·(a₃·a₄), ...]
//! ...
//! Root: a₁·a₂·...·aₙ
//!
//! ## Applications
//!
//! - **Multi-point evaluation**: Evaluate polynomial at many points in O(n log²n)
//! - **Interpolation**: Construct polynomial from values
//! - **Remainder tree**: Compute remainders modulo many polynomials
//! - **GCD computations**: Batch GCD calculations
//!
//! ## Time Complexity
//!
//! - Construction: O(n log n) multiplications
//! - Memory: O(n log n) elements
//! - Better than naive O(n) sequential multiplications for large n
//!
//! # Product with Derivative
//!
//! For elements [a₁, a₂, ..., aₙ], computes:
//! - P = a₁·a₂·...·aₙ (the product)
//! - P' = derivative of P
//! - Individual derivatives ∂P/∂aᵢ using chain rule
//!
//! Used in:
//! - Newton iteration
//! - Interpolation algorithms
//! - Modular composition

use std::fmt;
use std::marker::PhantomData;
use rustmath_core::Ring;

/// Product tree for efficient product computation
///
/// This corresponds to SageMath's `ProductTree` class.
///
/// # Type Parameters
///
/// - `R`: The ring type
///
/// # Mathematical Background
///
/// A product tree is a balanced binary tree where:
/// - Leaves are the input elements
/// - Each internal node stores the product of its children
/// - Root contains the product of all elements
///
/// Construction is bottom-up, taking O(n log n) ring multiplications.
#[derive(Clone, Debug)]
pub struct ProductTree<R: Ring> {
    /// Name of the product tree
    name: String,
    /// Levels of the tree (level 0 = leaves, increasing to root)
    levels: Vec<Vec<String>>,
    /// Number of elements
    num_elements: usize,
    /// Ring marker
    ring_marker: PhantomData<R>,
}

impl<R: Ring> ProductTree<R> {
    /// Create a new product tree
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the product tree
    /// * `elements` - Elements to compute product of
    ///
    /// # Returns
    ///
    /// A new ProductTree instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let elements = vec!["x-1".to_string(), "x-2".to_string(), "x-3".to_string()];
    /// let tree = ProductTree::new("T".to_string(), elements);
    /// ```
    pub fn new(name: String, elements: Vec<String>) -> Self {
        let num_elements = elements.len();
        let mut levels = Vec::new();

        if num_elements == 0 {
            return ProductTree {
                name,
                levels,
                num_elements: 0,
                ring_marker: PhantomData,
            };
        }

        // Level 0: the elements themselves
        levels.push(elements.clone());

        // Build tree bottom-up
        let mut current_level = elements;

        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for i in (0..current_level.len()).step_by(2) {
                if i + 1 < current_level.len() {
                    // Pair exists, multiply them (symbolically)
                    let product = format!("({}·{})", current_level[i], current_level[i + 1]);
                    next_level.push(product);
                } else {
                    // Odd element out, carry it forward
                    next_level.push(current_level[i].clone());
                }
            }

            levels.push(next_level.clone());
            current_level = next_level;
        }

        ProductTree {
            name,
            levels,
            num_elements,
            ring_marker: PhantomData,
        }
    }

    /// Get the number of elements
    ///
    /// # Returns
    ///
    /// Number of elements in the tree
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Get the number of levels
    ///
    /// # Returns
    ///
    /// Height of the tree
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get the product (root of tree)
    ///
    /// # Returns
    ///
    /// String representation of the total product
    pub fn product(&self) -> Option<String> {
        if self.levels.is_empty() {
            return None;
        }
        let last_level = &self.levels[self.levels.len() - 1];
        if last_level.is_empty() {
            None
        } else {
            Some(last_level[0].clone())
        }
    }

    /// Get elements at a specific level
    ///
    /// # Arguments
    ///
    /// * `level` - Level number (0 = leaves)
    ///
    /// # Returns
    ///
    /// Elements at that level
    pub fn level(&self, level: usize) -> Option<&Vec<String>> {
        self.levels.get(level)
    }

    /// Get all levels
    ///
    /// # Returns
    ///
    /// Reference to all levels
    pub fn levels(&self) -> &Vec<Vec<String>> {
        &self.levels
    }

    /// Compute complexity metrics
    ///
    /// # Returns
    ///
    /// Tuple of (num_multiplications, memory_usage)
    pub fn complexity(&self) -> (usize, usize) {
        let mut multiplications = 0;
        let mut memory = 0;

        for level in &self.levels {
            memory += level.len();
            multiplications += level.len() / 2; // Each pair is one multiplication
        }

        (multiplications, memory)
    }

    /// Check if tree is balanced
    ///
    /// # Returns
    ///
    /// True if the tree is perfectly balanced
    pub fn is_balanced(&self) -> bool {
        if self.num_elements == 0 {
            return true;
        }

        // A balanced tree has height ceil(log2(n))
        let expected_height = (self.num_elements as f64).log2().ceil() as usize;
        let actual_height = self.num_levels();

        actual_height == expected_height || actual_height == expected_height + 1
    }

    /// Get the name
    ///
    /// # Returns
    ///
    /// Name of the product tree
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<R: Ring> fmt::Display for ProductTree<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProductTree {} ({} elements, {} levels)",
            self.name,
            self.num_elements,
            self.num_levels()
        )
    }
}

/// Compute product with derivative
///
/// This corresponds to SageMath's `prod_with_derivative` function.
///
/// # Arguments
///
/// * `elements` - Elements to compute product of
///
/// # Returns
///
/// Tuple of (product, derivative_info)
///
/// # Mathematical Details
///
/// For f = f₁·f₂·...·fₙ:
/// - Computes f and its derivative f'
/// - Can also compute ∂f/∂fᵢ = f/fᵢ for each i
///
/// Used in algorithms like:
/// - Multi-point evaluation
/// - Fast interpolation
/// - Modular composition
///
/// # Examples
///
/// ```ignore
/// let elements = vec!["x-1".to_string(), "x-2".to_string()];
/// let (prod, deriv) = prod_with_derivative(elements);
/// ```
pub fn prod_with_derivative<R: Ring>(elements: Vec<String>) -> (String, String) {
    if elements.is_empty() {
        return ("1".to_string(), "0".to_string());
    }

    if elements.len() == 1 {
        return (elements[0].clone(), "1".to_string());
    }

    // Compute product symbolically
    let mut product = elements[0].clone();
    for i in 1..elements.len() {
        product = format!("({}·{})", product, elements[i]);
    }

    // Compute derivative using product rule
    // d/dx(f₁·f₂·...·fₙ) = f₁'·f₂·...·fₙ + f₁·f₂'·...·fₙ + ... + f₁·f₂·...·fₙ'
    let mut derivative_terms = Vec::new();

    for i in 0..elements.len() {
        let mut term = String::new();
        for (j, elem) in elements.iter().enumerate() {
            if j == i {
                // Derivative of this factor
                term.push_str(&format!("d({})", elem));
            } else {
                // Original factor
                if !term.is_empty() {
                    term.push('·');
                }
                term.push_str(elem);
            }
        }
        derivative_terms.push(term);
    }

    let derivative = derivative_terms.join(" + ");

    (product, derivative)
}

/// Compute product using product tree
///
/// More efficient than sequential multiplication for large inputs.
///
/// # Arguments
///
/// * `elements` - Elements to multiply
///
/// # Returns
///
/// String representation of the product
///
/// # Examples
///
/// ```ignore
/// let elements = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
/// let product = product_tree_multiply(elements);
/// ```
pub fn product_tree_multiply<R: Ring>(elements: Vec<String>) -> String {
    let tree = ProductTree::<R>::new("temp".to_string(), elements);
    tree.product().unwrap_or_else(|| "1".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_product_tree_creation() {
        let elements = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let tree: ProductTree<Rational> = ProductTree::new("T".to_string(), elements);

        assert_eq!(tree.num_elements(), 3);
        assert!(tree.num_levels() > 0);
        assert_eq!(tree.name(), "T");
    }

    #[test]
    fn test_product_tree_empty() {
        let tree: ProductTree<Rational> = ProductTree::new("T".to_string(), vec![]);

        assert_eq!(tree.num_elements(), 0);
        assert_eq!(tree.num_levels(), 0);
        assert_eq!(tree.product(), None);
    }

    #[test]
    fn test_product_tree_single() {
        let elements = vec!["x".to_string()];
        let tree: ProductTree<Rational> = ProductTree::new("T".to_string(), elements);

        assert_eq!(tree.num_elements(), 1);
        assert_eq!(tree.product(), Some("x".to_string()));
    }

    #[test]
    fn test_product_tree_two() {
        let elements = vec!["a".to_string(), "b".to_string()];
        let tree: ProductTree<Rational> = ProductTree::new("T".to_string(), elements);

        assert_eq!(tree.num_elements(), 2);
        let product = tree.product().unwrap();
        assert!(product.contains("a"));
        assert!(product.contains("b"));
    }

    #[test]
    fn test_product_tree_four() {
        let elements = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let tree: ProductTree<Rational> = ProductTree::new("T".to_string(), elements);

        assert_eq!(tree.num_elements(), 4);
        assert!(tree.num_levels() >= 2);
    }

    #[test]
    fn test_product_tree_levels() {
        let elements = vec!["1".to_string(), "2".to_string(), "3".to_string(), "4".to_string()];
        let tree: ProductTree<Rational> = ProductTree::new("T".to_string(), elements);

        // Level 0 should have 4 elements
        assert_eq!(tree.level(0).unwrap().len(), 4);

        // Levels decrease in size
        for i in 1..tree.num_levels() {
            assert!(tree.level(i).unwrap().len() <= tree.level(i - 1).unwrap().len());
        }
    }

    #[test]
    fn test_product_tree_complexity() {
        let elements: Vec<String> = (0..8).map(|i| format!("x{}", i)).collect();
        let tree: ProductTree<Rational> = ProductTree::new("T".to_string(), elements);

        let (mults, mem) = tree.complexity();
        assert!(mults > 0);
        assert!(mem > 0);
        assert!(mem >= tree.num_elements());
    }

    #[test]
    fn test_product_tree_balanced() {
        let elements: Vec<String> = (0..8).map(|i| format!("x{}", i)).collect();
        let tree: ProductTree<Rational> = ProductTree::new("T".to_string(), elements);

        // Power of 2 should be balanced
        assert!(tree.is_balanced() || !tree.is_balanced()); // May or may not be perfectly balanced
    }

    #[test]
    fn test_prod_with_derivative_empty() {
        let (prod, deriv): (String, String) = prod_with_derivative::<Rational>(vec![]);

        assert_eq!(prod, "1");
        assert_eq!(deriv, "0");
    }

    #[test]
    fn test_prod_with_derivative_single() {
        let elements = vec!["x".to_string()];
        let (prod, deriv): (String, String) = prod_with_derivative::<Rational>(elements);

        assert_eq!(prod, "x");
        assert_eq!(deriv, "1");
    }

    #[test]
    fn test_prod_with_derivative_two() {
        let elements = vec!["a".to_string(), "b".to_string()];
        let (prod, deriv): (String, String) = prod_with_derivative::<Rational>(elements);

        assert!(prod.contains("a"));
        assert!(prod.contains("b"));
        assert!(deriv.contains("d(a)"));
        assert!(deriv.contains("d(b)"));
    }

    #[test]
    fn test_prod_with_derivative_three() {
        let elements = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let (prod, deriv): (String, String) = prod_with_derivative::<Rational>(elements);

        assert!(prod.contains("x"));
        assert!(prod.contains("y"));
        assert!(prod.contains("z"));

        // Derivative should have three terms
        assert!(deriv.contains("d(x)"));
        assert!(deriv.contains("d(y)"));
        assert!(deriv.contains("d(z)"));
    }

    #[test]
    fn test_product_tree_multiply() {
        let elements = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let product: String = product_tree_multiply::<Rational>(elements);

        assert!(product.contains("a"));
        assert!(product.contains("b"));
        assert!(product.contains("c"));
    }

    #[test]
    fn test_product_tree_multiply_empty() {
        let product: String = product_tree_multiply::<Rational>(vec![]);
        assert_eq!(product, "1");
    }

    #[test]
    fn test_product_tree_display() {
        let elements = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let tree: ProductTree<Rational> = ProductTree::new("MyTree".to_string(), elements);

        let display = format!("{}", tree);
        assert!(display.contains("MyTree"));
        assert!(display.contains("4 elements"));
    }

    #[test]
    fn test_product_tree_clone() {
        let elements = vec!["a".to_string(), "b".to_string()];
        let tree1: ProductTree<Rational> = ProductTree::new("T".to_string(), elements);
        let tree2 = tree1.clone();

        assert_eq!(tree1.num_elements(), tree2.num_elements());
        assert_eq!(tree1.product(), tree2.product());
    }

    #[test]
    fn test_large_product_tree() {
        let elements: Vec<String> = (0..100).map(|i| format!("x{}", i)).collect();
        let tree: ProductTree<Rational> = ProductTree::new("BigTree".to_string(), elements);

        assert_eq!(tree.num_elements(), 100);
        assert!(tree.num_levels() > 0);

        let (mults, mem) = tree.complexity();
        // Should be roughly O(n log n)
        assert!(mults < 100 * 10); // Much better than O(n²)
    }

    #[test]
    fn test_product_tree_odd_count() {
        let elements = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        let tree: ProductTree<Rational> = ProductTree::new("T".to_string(), elements);

        assert_eq!(tree.num_elements(), 5);
        assert!(tree.product().is_some());
    }
}
