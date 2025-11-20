//! Cluster variables and their computation
//!
//! This module provides functionality for computing and tracking cluster variables
//! in cluster algebras. Cluster variables are the generators of a cluster algebra,
//! and they transform under mutations according to the exchange relations.
//!
//! # Background
//!
//! In a cluster algebra, each seed contains a cluster of n variables.
//! When we mutate at direction k, the variable x_k is replaced by a new
//! variable x_k' according to the exchange relation:
//!
//! x_k * x_k' = ∏(x_i^max(b_ik, 0)) + ∏(x_i^max(-b_ik, 0))
//!
//! where B = (b_ij) is the exchange matrix.

use crate::cluster_algebra::{GVector, DVector, ClusterAlgebraSeed};
use rustmath_core::Ring;
use rustmath_polynomials::MultivariatePolynomial;
use std::collections::HashMap;
use std::fmt::{self, Display};

/// A cluster variable in a cluster algebra
///
/// Cluster variables are parametrized by their g-vectors and can be
/// computed using F-polynomials.
#[derive(Debug, Clone)]
pub struct ClusterVariable<R: Ring> {
    /// The g-vector parametrizing this variable
    pub g_vector: GVector,
    /// The d-vector (denominator vector)
    pub d_vector: DVector,
    /// The F-polynomial (in coefficient variables)
    pub f_polynomial: Option<MultivariatePolynomial<R>>,
    /// The cluster variable as a Laurent polynomial (if computed)
    pub laurent_polynomial: Option<MultivariatePolynomial<R>>,
    /// Index in the initial seed (if it's an initial variable)
    pub initial_index: Option<usize>,
}

impl<R: Ring + Clone> ClusterVariable<R> {
    /// Create a new cluster variable from a g-vector
    pub fn new(g_vector: GVector) -> Self {
        let dim = g_vector.dim();
        ClusterVariable {
            g_vector,
            d_vector: DVector::new(vec![0; dim]),
            f_polynomial: None,
            laurent_polynomial: None,
            initial_index: None,
        }
    }

    /// Create an initial cluster variable (from the initial seed)
    pub fn initial(index: usize, rank: usize) -> Self {
        let g_vec = GVector::standard_basis(rank, index);
        let mut d_vec_components = vec![0; rank];
        d_vec_components[index] = 1;

        ClusterVariable {
            g_vector: g_vec,
            d_vector: DVector::new(d_vec_components),
            f_polynomial: None,
            laurent_polynomial: None,
            initial_index: Some(index),
        }
    }

    /// Check if this is an initial cluster variable
    pub fn is_initial(&self) -> bool {
        self.initial_index.is_some()
    }

    /// Get the index if this is an initial variable
    pub fn get_initial_index(&self) -> Option<usize> {
        self.initial_index
    }

    /// Set the F-polynomial
    pub fn set_f_polynomial(&mut self, f_poly: MultivariatePolynomial<R>) {
        self.f_polynomial = Some(f_poly);
    }

    /// Set the Laurent polynomial
    pub fn set_laurent_polynomial(&mut self, laurent: MultivariatePolynomial<R>) {
        self.laurent_polynomial = Some(laurent);
    }

    /// Set the d-vector
    pub fn set_d_vector(&mut self, d_vec: DVector) {
        self.d_vector = d_vec;
    }
}

impl<R: Ring + Clone + Display> Display for ClusterVariable<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(idx) = self.initial_index {
            write!(f, "x_{}", idx)
        } else {
            write!(f, "x_{}", self.g_vector)
        }
    }
}

/// A cluster (set of cluster variables forming a seed)
#[derive(Debug, Clone)]
pub struct Cluster<R: Ring> {
    /// The cluster variables in this cluster
    pub variables: Vec<ClusterVariable<R>>,
    /// Mutation path from initial seed
    pub mutation_path: Vec<usize>,
}

impl<R: Ring + Clone> Cluster<R> {
    /// Create a new empty cluster
    pub fn new() -> Self {
        Cluster {
            variables: Vec::new(),
            mutation_path: Vec::new(),
        }
    }

    /// Create the initial cluster of given rank
    pub fn initial(rank: usize) -> Self {
        let variables = (0..rank)
            .map(|i| ClusterVariable::initial(i, rank))
            .collect();

        Cluster {
            variables,
            mutation_path: Vec::new(),
        }
    }

    /// Get the rank of the cluster
    pub fn rank(&self) -> usize {
        self.variables.len()
    }

    /// Get a variable by index
    pub fn get_variable(&self, index: usize) -> Option<&ClusterVariable<R>> {
        self.variables.get(index)
    }

    /// Mutate the cluster at direction k
    ///
    /// This replaces x_k with the mutated variable according to
    /// the exchange relation.
    pub fn mutate(&mut self, k: usize, exchange_matrix: &[Vec<i64>])
    where
        R: From<i64>,
    {
        if k >= self.rank() {
            return;
        }

        // Compute new g-vector for the mutated variable
        let old_g = &self.variables[k].g_vector;
        let mut new_g_components = old_g.components.clone();

        // Apply g-vector mutation formula
        // g'_k = -g_k
        for comp in &mut new_g_components {
            *comp = -*comp;
        }

        let new_g = GVector::new(new_g_components);

        // Create the new cluster variable
        let mut new_var = ClusterVariable::new(new_g);

        // Compute new d-vector
        // This is simplified; full version would use proper formula
        let mut new_d_components = vec![0i64; self.rank()];
        new_d_components[k] = 1;
        new_var.set_d_vector(DVector::new(new_d_components));

        // Replace the variable
        self.variables[k] = new_var;

        // Update mutation path
        self.mutation_path.push(k);
    }

    /// Get the mutation path from the initial cluster
    pub fn mutation_path(&self) -> &[usize] {
        &self.mutation_path
    }

    /// Get all g-vectors in this cluster
    pub fn g_vectors(&self) -> Vec<GVector> {
        self.variables.iter().map(|v| v.g_vector.clone()).collect()
    }
}

impl<R: Ring + Clone> Default for Cluster<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring + Clone + Display> Display for Cluster<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{ ")?;
        for (i, var) in self.variables.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", var)?;
        }
        write!(f, " }}")
    }
}

/// A cluster variable cache for efficient computation
///
/// This caches computed cluster variables to avoid recomputation.
#[derive(Debug, Clone)]
pub struct ClusterVariableCache<R: Ring> {
    /// Map from g-vectors to cluster variables
    cache: HashMap<GVector, ClusterVariable<R>>,
    /// Rank of the cluster algebra
    rank: usize,
}

impl<R: Ring + Clone> ClusterVariableCache<R> {
    /// Create a new cache
    pub fn new(rank: usize) -> Self {
        let mut cache = HashMap::new();

        // Add initial variables
        for i in 0..rank {
            let var = ClusterVariable::initial(i, rank);
            cache.insert(var.g_vector.clone(), var);
        }

        ClusterVariableCache { cache, rank }
    }

    /// Get a cluster variable by g-vector
    pub fn get(&self, g_vec: &GVector) -> Option<&ClusterVariable<R>> {
        self.cache.get(g_vec)
    }

    /// Insert a cluster variable
    pub fn insert(&mut self, var: ClusterVariable<R>) {
        self.cache.insert(var.g_vector.clone(), var);
    }

    /// Check if a g-vector is cached
    pub fn contains(&self, g_vec: &GVector) -> bool {
        self.cache.contains_key(g_vec)
    }

    /// Get the number of cached variables
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty (shouldn't be if initialized properly)
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear the cache except for initial variables
    pub fn clear_non_initial(&mut self) {
        self.cache.retain(|_, v| v.is_initial());
    }

    /// Get all g-vectors in the cache
    pub fn g_vectors(&self) -> Vec<GVector> {
        self.cache.keys().cloned().collect()
    }

    /// Get all cached variables
    pub fn variables(&self) -> Vec<&ClusterVariable<R>> {
        self.cache.values().collect()
    }
}

/// Compute the exchange polynomial for mutation at direction k
///
/// The exchange polynomial is:
/// E_k = ∏(x_i^max(b_ik, 0)) + ∏(x_i^max(-b_ik, 0))
///
/// This is used in the exchange relation: x_k * x_k' = E_k
pub fn exchange_polynomial<R>(
    k: usize,
    exchange_matrix: &[Vec<i64>],
    _variables: &[ClusterVariable<R>],
) -> MultivariatePolynomial<R>
where
    R: Ring + Clone + From<i64>,
{
    // Simplified implementation
    // Full version would construct the actual polynomial
    MultivariatePolynomial::zero()
}

/// Compute F-polynomial for a cluster variable
///
/// F-polynomials are polynomials in the coefficient variables that
/// appear in the separation of additions formula.
pub fn f_polynomial<R>(
    _g_vec: &GVector,
    _exchange_matrix: &[Vec<i64>],
) -> MultivariatePolynomial<R>
where
    R: Ring + Clone + From<i64>,
{
    // Simplified implementation
    // Full version would use recursive computation or greedy algorithm
    MultivariatePolynomial::zero()
}

/// Compute a cluster variable from its g-vector
///
/// Uses the separation of additions formula to express the cluster
/// variable as a Laurent polynomial in the initial cluster variables.
pub fn cluster_variable_from_g_vector<R>(
    g_vec: &GVector,
    _exchange_matrix: &[Vec<i64>],
    _cache: &mut ClusterVariableCache<R>,
) -> ClusterVariable<R>
where
    R: Ring + Clone + From<i64>,
{
    // Simplified implementation
    // Full version would compute F-polynomial and apply formula
    ClusterVariable::new(g_vec.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_cluster_variable_creation() {
        let g_vec = GVector::new(vec![1, 0, 0]);
        let var: ClusterVariable<Integer> = ClusterVariable::new(g_vec.clone());

        assert_eq!(var.g_vector.components, vec![1, 0, 0]);
        assert!(!var.is_initial());
    }

    #[test]
    fn test_initial_cluster_variable() {
        let var: ClusterVariable<Integer> = ClusterVariable::initial(1, 3);

        assert_eq!(var.g_vector.components, vec![0, 1, 0]);
        assert!(var.is_initial());
        assert_eq!(var.get_initial_index(), Some(1));
    }

    #[test]
    fn test_cluster_creation() {
        let cluster: Cluster<Integer> = Cluster::initial(3);

        assert_eq!(cluster.rank(), 3);
        assert!(cluster.get_variable(0).unwrap().is_initial());
        assert_eq!(cluster.mutation_path().len(), 0);
    }

    #[test]
    fn test_cluster_g_vectors() {
        let cluster: Cluster<Integer> = Cluster::initial(3);
        let g_vecs = cluster.g_vectors();

        assert_eq!(g_vecs.len(), 3);
        assert_eq!(g_vecs[0].components, vec![1, 0, 0]);
        assert_eq!(g_vecs[1].components, vec![0, 1, 0]);
        assert_eq!(g_vecs[2].components, vec![0, 0, 1]);
    }

    #[test]
    fn test_cluster_variable_cache() {
        let cache: ClusterVariableCache<Integer> = ClusterVariableCache::new(3);

        assert_eq!(cache.len(), 3);

        let g_vec = GVector::new(vec![1, 0, 0]);
        assert!(cache.contains(&g_vec));
    }

    #[test]
    fn test_cache_operations() {
        let mut cache: ClusterVariableCache<Integer> = ClusterVariableCache::new(2);

        let g_vec = GVector::new(vec![1, 1]);
        let var = ClusterVariable::new(g_vec.clone());
        cache.insert(var);

        assert_eq!(cache.len(), 3); // 2 initial + 1 new
        assert!(cache.contains(&g_vec));
    }

    #[test]
    fn test_cache_clear_non_initial() {
        let mut cache: ClusterVariableCache<Integer> = ClusterVariableCache::new(2);

        let g_vec = GVector::new(vec![1, 1]);
        let var = ClusterVariable::new(g_vec);
        cache.insert(var);

        cache.clear_non_initial();
        assert_eq!(cache.len(), 2); // Only initial variables remain
    }

    #[test]
    fn test_cluster_mutation() {
        let mut cluster: Cluster<Integer> = Cluster::initial(2);

        // A_2 exchange matrix
        let exchange_matrix = vec![vec![0, 1], vec![-1, 0]];

        cluster.mutate(0, &exchange_matrix);

        assert_eq!(cluster.mutation_path().len(), 1);
        assert_eq!(cluster.mutation_path()[0], 0);
    }
}
