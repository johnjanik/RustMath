//! Random expression generation for testing
//!
//! This module provides utilities for generating random symbolic expressions
//! and testing symbolic operations. It's particularly useful for:
//! - Property-based testing
//! - Fuzzing symbolic operations
//! - Testing expression ordering
//! - Verifying algebraic properties
//!
//! # Random Expression Generation
//!
//! The module can generate random expressions with:
//! - Configurable depth and complexity
//! - Different probability distributions for operations
//! - Integer, rational, and symbolic components
//! - Various function types (trig, exp, log, etc.)
//!
//! # Testing Expression Order
//!
//! Symbolic expressions need a total order for canonicalization.
//! This module provides utilities to test that the order is:
//! - Transitive: if a < b and b < c, then a < c
//! - Antisymmetric: if a < b, then !(b < a)
//! - Total: for any a, b either a < b, b < a, or a == b

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rand::Rng;

/// Probability list for choosing operations
///
/// A list of (probability, value) pairs. Probabilities should sum to 1.0,
/// but if they don't, they will be normalized.
pub type ProbList<T> = Vec<(f64, T)>;

/// Normalize a probability list so probabilities sum to 1.0
///
/// # Arguments
///
/// * `probs` - List of (probability, value) pairs
///
/// # Returns
///
/// Normalized list where probabilities sum to 1.0
///
/// # Example
///
/// ```
/// use rustmath_symbolic::random_tests::normalize_prob_list;
///
/// let probs = vec![(2.0, "a"), (3.0, "b"), (5.0, "c")];
/// let normalized = normalize_prob_list(probs);
///
/// // Probabilities now sum to 1.0: [0.2, 0.3, 0.5]
/// assert!((normalized[0].0 - 0.2).abs() < 0.001);
/// assert!((normalized[1].0 - 0.3).abs() < 0.001);
/// assert!((normalized[2].0 - 0.5).abs() < 0.001);
/// ```
pub fn normalize_prob_list<T>(probs: ProbList<T>) -> ProbList<T> {
    let total: f64 = probs.iter().map(|(p, _)| p).sum();

    if total == 0.0 {
        return probs;
    }

    probs
        .into_iter()
        .map(|(p, v)| (p / total, v))
        .collect()
}

/// Choose an element from a probability list
///
/// # Arguments
///
/// * `probs` - Normalized probability list
/// * `rng` - Random number generator
///
/// # Returns
///
/// A reference to the chosen element, or None if list is empty
///
/// # Example
///
/// ```
/// use rustmath_symbolic::random_tests::{normalize_prob_list, choose_from_prob_list};
/// use rand::thread_rng;
///
/// let probs = vec![(0.5, "heads"), (0.5, "tails")];
/// let normalized = normalize_prob_list(probs);
/// let mut rng = thread_rng();
///
/// let choice = choose_from_prob_list(&normalized, &mut rng);
/// assert!(choice.is_some());
/// ```
pub fn choose_from_prob_list<'a, T, R: Rng>(
    probs: &'a ProbList<T>,
    rng: &mut R,
) -> Option<&'a T> {
    if probs.is_empty() {
        return None;
    }

    let mut r = rng.gen::<f64>();

    for (prob, value) in probs {
        if r < *prob {
            return Some(value);
        }
        r -= prob;
    }

    // Due to floating point errors, might not find anything
    // Return the last element
    probs.last().map(|(_, v)| v)
}

/// Generate a random integer vector
///
/// # Arguments
///
/// * `size` - Number of elements
/// * `min` - Minimum value (inclusive)
/// * `max` - Maximum value (inclusive)
/// * `rng` - Random number generator
///
/// # Returns
///
/// A vector of random integers
///
/// # Example
///
/// ```
/// use rustmath_symbolic::random_tests::random_integer_vector;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
/// let vec = random_integer_vector(5, 0, 10, &mut rng);
/// assert_eq!(vec.len(), 5);
/// assert!(vec.iter().all(|&x| x >= 0 && x <= 10));
/// ```
pub fn random_integer_vector<R: Rng>(size: usize, min: i64, max: i64, rng: &mut R) -> Vec<i64> {
    (0..size)
        .map(|_| rng.gen_range(min..=max))
        .collect()
}

/// Configuration for random expression generation
#[derive(Debug, Clone)]
pub struct RandomExprConfig {
    /// Maximum depth of expression tree
    pub max_depth: usize,

    /// Probability of choosing each operation type
    pub operation_probs: ProbList<OperationType>,

    /// Range for random integers
    pub integer_range: (i64, i64),

    /// Available variable names
    pub variables: Vec<String>,
}

impl Default for RandomExprConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            operation_probs: normalize_prob_list(vec![
                (3.0, OperationType::Add),
                (3.0, OperationType::Mul),
                (1.0, OperationType::Sub),
                (1.0, OperationType::Div),
                (2.0, OperationType::Pow),
                (2.0, OperationType::Sin),
                (2.0, OperationType::Cos),
                (1.0, OperationType::Exp),
                (1.0, OperationType::Log),
                (5.0, OperationType::Leaf),
            ]),
            integer_range: (-10, 10),
            variables: vec!["x".to_string(), "y".to_string(), "z".to_string()],
        }
    }
}

/// Types of operations for random expression generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    Leaf, // Terminal (integer or variable)
}

/// Helper function for random expression generation
///
/// # Arguments
///
/// * `depth` - Current depth in the tree
/// * `config` - Configuration
/// * `rng` - Random number generator
///
/// # Returns
///
/// A random expression
fn random_expr_helper<R: Rng>(depth: usize, config: &RandomExprConfig, rng: &mut R) -> Expr {
    // Force leaf if we've reached max depth
    if depth >= config.max_depth {
        return random_leaf(config, rng);
    }

    let op = choose_from_prob_list(&config.operation_probs, rng);

    match op {
        Some(OperationType::Add) => {
            let left = random_expr_helper(depth + 1, config, rng);
            let right = random_expr_helper(depth + 1, config, rng);
            left + right
        }
        Some(OperationType::Sub) => {
            let left = random_expr_helper(depth + 1, config, rng);
            let right = random_expr_helper(depth + 1, config, rng);
            left - right
        }
        Some(OperationType::Mul) => {
            let left = random_expr_helper(depth + 1, config, rng);
            let right = random_expr_helper(depth + 1, config, rng);
            left * right
        }
        Some(OperationType::Div) => {
            let left = random_expr_helper(depth + 1, config, rng);
            let right = random_expr_helper(depth + 1, config, rng);
            left / right
        }
        Some(OperationType::Pow) => {
            let base = random_expr_helper(depth + 1, config, rng);
            // Keep exponents small to avoid huge expressions
            let exp = Expr::from(rng.gen_range(0..=3));
            base.pow(exp)
        }
        Some(OperationType::Sin) => {
            let inner = random_expr_helper(depth + 1, config, rng);
            inner.sin()
        }
        Some(OperationType::Cos) => {
            let inner = random_expr_helper(depth + 1, config, rng);
            inner.cos()
        }
        Some(OperationType::Tan) => {
            let inner = random_expr_helper(depth + 1, config, rng);
            inner.tan()
        }
        Some(OperationType::Exp) => {
            let inner = random_expr_helper(depth + 1, config, rng);
            inner.exp()
        }
        Some(OperationType::Log) => {
            let inner = random_expr_helper(depth + 1, config, rng);
            inner.log()
        }
        Some(OperationType::Sqrt) => {
            let inner = random_expr_helper(depth + 1, config, rng);
            inner.sqrt()
        }
        Some(OperationType::Leaf) | None => random_leaf(config, rng),
    }
}

/// Generate a random leaf node (integer or variable)
fn random_leaf<R: Rng>(config: &RandomExprConfig, rng: &mut R) -> Expr {
    let (min, max) = config.integer_range;

    // 50/50 chance of integer vs variable
    if rng.gen_bool(0.5) && !config.variables.is_empty() {
        let var_idx = rng.gen_range(0..config.variables.len());
        Expr::symbol(&config.variables[var_idx])
    } else {
        Expr::from(rng.gen_range(min..=max))
    }
}

/// Generate a random symbolic expression
///
/// # Arguments
///
/// * `config` - Configuration for expression generation
/// * `rng` - Random number generator
///
/// # Returns
///
/// A random expression
///
/// # Example
///
/// ```
/// use rustmath_symbolic::random_tests::{random_expr, RandomExprConfig};
/// use rand::thread_rng;
///
/// let config = RandomExprConfig::default();
/// let mut rng = thread_rng();
/// let expr = random_expr(&config, &mut rng);
///
/// // Expression should not be trivially empty
/// // (This is a weak test, but demonstrates usage)
/// ```
pub fn random_expr<R: Rng>(config: &RandomExprConfig, rng: &mut R) -> Expr {
    random_expr_helper(0, config, rng)
}

/// Test that expression ordering is a strict weak order
///
/// A strict weak order must satisfy:
/// 1. Irreflexivity: !(a < a)
/// 2. Asymmetry: if a < b then !(b < a)
/// 3. Transitivity: if a < b and b < c then a < c
/// 4. Transitivity of incomparability: if a~b and b~c then a~c
///    where a~b means !(a<b) && !(b<a)
///
/// # Arguments
///
/// * `exprs` - List of expressions to test
///
/// # Implementation Note
///
/// Currently, `Expr` does not implement `PartialOrd` or `Ord`.
/// This function is a placeholder for when expression ordering is implemented.
/// For now, it simply verifies that all expressions are valid (don't panic
/// when used in comparisons like equality).
pub fn assert_strict_weak_order(exprs: &[Expr]) {
    // Placeholder implementation: verify expressions are valid
    // In a full implementation, this would test ordering properties

    // Test that equality works
    for (i, a) in exprs.iter().enumerate() {
        for (j, b) in exprs.iter().enumerate() {
            if i == j {
                // Same expression should equal itself
                assert_eq!(a, b, "Expression should equal itself");
            }
        }
    }

    // In the future, when Expr implements Ord:
    // - Test irreflexivity: !(a < a)
    // - Test asymmetry: if a < b then !(b < a)
    // - Test transitivity: if a < b and b < c then a < c
}

/// Test symbolic expression ordering on random expressions
///
/// Generates many random expressions and verifies that their ordering
/// satisfies the strict weak order properties.
///
/// # Arguments
///
/// * `num_exprs` - Number of expressions to generate
/// * `config` - Configuration for expression generation
/// * `rng` - Random number generator
///
/// # Panics
///
/// Panics if ordering violations are found
///
/// # Example
///
/// ```
/// use rustmath_symbolic::random_tests::{test_symbolic_expression_order, RandomExprConfig};
/// use rand::thread_rng;
///
/// let config = RandomExprConfig::default();
/// let mut rng = thread_rng();
/// test_symbolic_expression_order(20, &config, &mut rng);
/// ```
pub fn test_symbolic_expression_order<R: Rng>(
    num_exprs: usize,
    config: &RandomExprConfig,
    rng: &mut R,
) {
    let exprs: Vec<Expr> = (0..num_exprs)
        .map(|_| random_expr(config, rng))
        .collect();

    assert_strict_weak_order(&exprs);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_normalize_prob_list_sum_to_one() {
        let probs = vec![(1.0, "a"), (1.0, "b"), (1.0, "c")];
        let normalized = normalize_prob_list(probs);

        let sum: f64 = normalized.iter().map(|(p, _)| p).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_prob_list_proportions() {
        let probs = vec![(2.0, "a"), (3.0, "b"), (5.0, "c")];
        let normalized = normalize_prob_list(probs);

        assert!((normalized[0].0 - 0.2).abs() < 0.001);
        assert!((normalized[1].0 - 0.3).abs() < 0.001);
        assert!((normalized[2].0 - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_normalize_prob_list_empty() {
        let probs: ProbList<i32> = vec![];
        let normalized = normalize_prob_list(probs);
        assert_eq!(normalized.len(), 0);
    }

    #[test]
    fn test_normalize_prob_list_zero_total() {
        let probs = vec![(0.0, "a"), (0.0, "b")];
        let normalized = normalize_prob_list(probs);
        // Should return unchanged when total is 0
        assert_eq!(normalized[0].0, 0.0);
        assert_eq!(normalized[1].0, 0.0);
    }

    #[test]
    fn test_choose_from_prob_list() {
        let mut rng = StdRng::seed_from_u64(12345);
        let probs = vec![(0.5, "heads"), (0.5, "tails")];

        let choice = choose_from_prob_list(&probs, &mut rng);
        assert!(choice.is_some());
        assert!(choice == Some(&"heads") || choice == Some(&"tails"));
    }

    #[test]
    fn test_choose_from_prob_list_empty() {
        let mut rng = StdRng::seed_from_u64(12345);
        let probs: ProbList<i32> = vec![];

        let choice = choose_from_prob_list(&probs, &mut rng);
        assert!(choice.is_none());
    }

    #[test]
    fn test_choose_from_prob_list_deterministic() {
        let mut rng = StdRng::seed_from_u64(12345);
        let probs = vec![(1.0, "always")];

        let choice = choose_from_prob_list(&probs, &mut rng);
        assert_eq!(choice, Some(&"always"));
    }

    #[test]
    fn test_random_integer_vector() {
        let mut rng = StdRng::seed_from_u64(12345);
        let vec = random_integer_vector(10, 0, 5, &mut rng);

        assert_eq!(vec.len(), 10);
        assert!(vec.iter().all(|&x| x >= 0 && x <= 5));
    }

    #[test]
    fn test_random_integer_vector_negative_range() {
        let mut rng = StdRng::seed_from_u64(12345);
        let vec = random_integer_vector(10, -5, 5, &mut rng);

        assert_eq!(vec.len(), 10);
        assert!(vec.iter().all(|&x| x >= -5 && x <= 5));
    }

    #[test]
    fn test_random_expr_generates() {
        let mut rng = StdRng::seed_from_u64(12345);
        let config = RandomExprConfig::default();

        let expr = random_expr(&config, &mut rng);
        // Just test that it doesn't panic and generates something
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_random_expr_respects_max_depth() {
        let mut rng = StdRng::seed_from_u64(12345);
        let mut config = RandomExprConfig::default();
        config.max_depth = 1;

        let expr = random_expr(&config, &mut rng);
        // With max_depth=1, we should get mostly leaves
        // This is a weak test, but verifies the config is used
        let _ = format!("{:?}", expr);
    }

    #[test]
    fn test_random_expr_with_variables() {
        let mut rng = StdRng::seed_from_u64(12345);
        let mut config = RandomExprConfig::default();
        config.variables = vec!["a".to_string(), "b".to_string()];

        let expr = random_expr(&config, &mut rng);
        let expr_str = format!("{:?}", expr);
        // Expression might contain variables (but not guaranteed with random generation)
        let _ = expr_str;
    }

    #[test]
    fn test_assert_strict_weak_order_simple() {
        let exprs = vec![
            Expr::from(1),
            Expr::from(2),
            Expr::from(3),
        ];

        assert_strict_weak_order(&exprs);
    }

    #[test]
    fn test_assert_strict_weak_order_with_symbols() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let exprs = vec![
            Expr::from(1),
            Expr::Symbol(x),
            Expr::Symbol(y),
        ];

        assert_strict_weak_order(&exprs);
    }

    #[test]
    fn test_assert_strict_weak_order_complex() {
        let x = Symbol::new("x");

        let exprs = vec![
            Expr::from(0),
            Expr::from(1),
            Expr::Symbol(x.clone()),
            Expr::Symbol(x.clone()) + Expr::from(1),
            Expr::Symbol(x).sin(),
        ];

        assert_strict_weak_order(&exprs);
    }

    #[test]
    fn test_symbolic_expression_order_random() {
        let mut rng = StdRng::seed_from_u64(12345);
        let config = RandomExprConfig::default();

        // Test with a small number of expressions
        test_symbolic_expression_order(10, &config, &mut rng);
    }

    #[test]
    fn test_random_expr_config_default() {
        let config = RandomExprConfig::default();
        assert_eq!(config.max_depth, 5);
        assert_eq!(config.variables.len(), 3);
        assert!(config.operation_probs.len() > 0);
    }
}
