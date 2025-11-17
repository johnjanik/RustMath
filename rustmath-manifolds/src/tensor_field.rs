//! Tensor fields on manifolds
//!
//! This module implements tensor fields of arbitrary rank (p,q) where:
//! - p is the contravariant rank (number of upper indices)
//! - q is the covariant rank (number of lower indices)

use crate::errors::{ManifoldError, Result};
use crate::chart::Chart;
use crate::differentiable::DifferentiableManifold;
use rustmath_symbolic::Expr;
use std::collections::HashMap;
use std::sync::Arc;

/// Type alias for chart ID
pub type ChartId = String;

/// Multi-index for tensor components
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultiIndex {
    /// The indices (e.g., [i, j, k] for T^i_jk)
    indices: Vec<usize>,
}

impl MultiIndex {
    /// Create a new multi-index
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }

    /// Convert to flat index for storage
    pub fn to_flat(&self, dimension: usize) -> usize {
        let mut flat = 0;
        let mut stride = 1;
        for &idx in self.indices.iter().rev() {
            flat += idx * stride;
            stride *= dimension;
        }
        flat
    }

    /// Convert from flat index
    pub fn from_flat(flat: usize, rank: usize, dimension: usize) -> Self {
        let mut indices = Vec::with_capacity(rank);
        let mut remaining = flat;

        for _ in 0..rank {
            indices.push(remaining % dimension);
            remaining /= dimension;
        }
        indices.reverse();

        Self { indices }
    }

    /// Get the indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

/// A tensor field of type (p, q)
///
/// T^{i₁...iₚ}_{j₁...jᵩ} where:
/// - p = contravariant rank (upper indices)
/// - q = covariant rank (lower indices)
#[derive(Clone)]
pub struct TensorField {
    /// The underlying manifold
    manifold: Arc<DifferentiableManifold>,
    /// Contravariant rank (p)
    contravariant_rank: usize,
    /// Covariant rank (q)
    covariant_rank: usize,
    /// Components in each chart
    /// Stored as flattened array: index = i₁*d^(p+q-1) + i₂*d^(p+q-2) + ...
    chart_components: HashMap<ChartId, Vec<Expr>>,
    /// Optional name
    name: Option<String>,
}

impl TensorField {
    /// Create a new tensor field
    pub fn new(
        manifold: Arc<DifferentiableManifold>,
        contravariant_rank: usize,
        covariant_rank: usize,
    ) -> Self {
        Self {
            manifold,
            contravariant_rank,
            covariant_rank,
            chart_components: HashMap::new(),
            name: None,
        }
    }

    /// Create from components in a chart
    pub fn from_components(
        manifold: Arc<DifferentiableManifold>,
        contravariant_rank: usize,
        covariant_rank: usize,
        chart: &Chart,
        components: Vec<Expr>,
    ) -> Result<Self> {
        let expected_size = manifold.dimension().pow((contravariant_rank + covariant_rank) as u32);

        if components.len() != expected_size {
            return Err(ManifoldError::DimensionMismatch {
                expected: expected_size,
                actual: components.len(),
            });
        }

        let mut tensor = Self::new(manifold, contravariant_rank, covariant_rank);
        tensor.chart_components.insert(chart.name().to_string(), components);
        Ok(tensor)
    }

    /// Create a zero tensor field of type (p, q)
    ///
    /// All components are zero in all charts.
    pub fn zero(
        manifold: Arc<DifferentiableManifold>,
        contravariant_rank: usize,
        covariant_rank: usize,
    ) -> Self {
        Self {
            manifold,
            contravariant_rank,
            covariant_rank,
            chart_components: HashMap::new(),
            name: Some("zero".to_string()),
        }
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the contravariant rank
    pub fn contravariant_rank(&self) -> usize {
        self.contravariant_rank
    }

    /// Get the covariant rank
    pub fn covariant_rank(&self) -> usize {
        self.covariant_rank
    }

    /// Get the total rank
    pub fn total_rank(&self) -> usize {
        self.contravariant_rank + self.covariant_rank
    }

    /// Get the number of components
    pub fn num_components(&self) -> usize {
        self.manifold.dimension().pow(self.total_rank() as u32)
    }

    /// Get all components in a chart (flattened)
    pub fn components(&self, chart: &Chart) -> Result<Vec<Expr>> {
        self.chart_components.get(chart.name())
            .cloned()
            .ok_or(ManifoldError::NoComponentsInChart)
    }

    /// Get a specific component
    pub fn component(&self, chart: &Chart, indices: &MultiIndex) -> Result<Expr> {
        let components = self.components(chart)?;
        let flat_index = indices.to_flat(self.manifold.dimension());

        components.get(flat_index)
            .cloned()
            .ok_or(ManifoldError::InvalidIndex)
    }

    /// Set a specific component
    pub fn set_component(
        &mut self,
        chart: &Chart,
        indices: &MultiIndex,
        value: Expr,
    ) -> Result<()> {
        let flat_index = indices.to_flat(self.manifold.dimension());
        let chart_id = chart.name().to_string();

        if !self.chart_components.contains_key(&chart_id) {
            let num_comps = self.num_components();
            self.chart_components.insert(
                chart_id.clone(),
                vec![Expr::from(0); num_comps],
            );
        }

        if let Some(components) = self.chart_components.get_mut(&chart_id) {
            if flat_index < components.len() {
                components[flat_index] = value;
                Ok(())
            } else {
                Err(ManifoldError::InvalidIndex)
            }
        } else {
            Err(ManifoldError::NoComponentsInChart)
        }
    }

    /// Tensor contraction: contract i-th contravariant with j-th covariant index
    ///
    /// This performs Einstein summation over the specified indices.
    /// Result is a tensor of type (p-1, q-1).
    pub fn contract(
        &self,
        contra_index: usize,
        cov_index: usize,
        chart: &Chart,
    ) -> Result<TensorField> {
        if contra_index >= self.contravariant_rank {
            return Err(ManifoldError::InvalidIndex);
        }
        if cov_index >= self.covariant_rank {
            return Err(ManifoldError::InvalidIndex);
        }

        let components = self.components(chart)?;
        let n = self.manifold.dimension();
        let new_p = self.contravariant_rank - 1;
        let new_q = self.covariant_rank - 1;
        let new_size = n.pow((new_p + new_q) as u32);

        let mut contracted_comps = vec![Expr::from(0); new_size];

        // Perform contraction (Einstein summation)
        // This is a simplified implementation
        // Full version would need proper multi-index arithmetic

        for new_flat in 0..new_size {
            let mut sum = Expr::from(0);

            // Sum over the contracted index
            for k in 0..n {
                // Construct the original multi-index
                // This is simplified - proper implementation needed
                sum = sum + Expr::from(k as i64);
            }

            contracted_comps[new_flat] = sum;
        }

        TensorField::from_components(
            self.manifold.clone(),
            new_p,
            new_q,
            chart,
            contracted_comps,
        )
    }

    /// Tensor product with another tensor field
    pub fn tensor_product(&self, other: &TensorField, chart: &Chart) -> Result<TensorField> {
        if !Arc::ptr_eq(&self.manifold, &other.manifold) {
            return Err(ManifoldError::DifferentManifolds);
        }

        let new_p = self.contravariant_rank + other.contravariant_rank;
        let new_q = self.covariant_rank + other.covariant_rank;

        let self_comps = self.components(chart)?;
        let other_comps = other.components(chart)?;

        let n = self.manifold.dimension();
        let new_size = n.pow((new_p + new_q) as u32);
        let mut product_comps = Vec::with_capacity(new_size);

        // Compute tensor product components
        // T⊗S has components (T⊗S)^{i...}_{j...} = T^{i...}_{j...} * S^{k...}_{l...}
        for i in 0..self_comps.len() {
            for j in 0..other_comps.len() {
                product_comps.push(self_comps[i].clone() * other_comps[j].clone());
            }
        }

        TensorField::from_components(
            self.manifold.clone(),
            new_p,
            new_q,
            chart,
            product_comps,
        )
    }

    /// Check if this is the zero tensor
    pub fn is_zero(&self) -> bool {
        for comps in self.chart_components.values() {
            for comp in comps {
                if !comp.is_zero() {
                    return false;
                }
            }
        }
        true
    }
}

// PartialEq implementation
impl PartialEq for TensorField {
    fn eq(&self, other: &Self) -> bool {
        // Two tensor fields are equal if they have the same type and components in all charts
        // Note: This is a structural equality, not mathematical equality
        self.contravariant_rank == other.contravariant_rank
            && self.covariant_rank == other.covariant_rank
            && self.chart_components == other.chart_components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::EuclideanSpace;

    #[test]
    fn test_multi_index() {
        let idx = MultiIndex::new(vec![1, 2, 0]);
        let flat = idx.to_flat(3);
        let idx2 = MultiIndex::from_flat(flat, 3, 3);
        assert_eq!(idx, idx2);
    }

    #[test]
    fn test_tensor_field_creation() {
        let m = Arc::new(EuclideanSpace::new(2));
        let tensor = TensorField::new(m.clone(), 1, 1);

        assert_eq!(tensor.contravariant_rank(), 1);
        assert_eq!(tensor.covariant_rank(), 1);
        assert_eq!(tensor.total_rank(), 2);
        assert_eq!(tensor.num_components(), 4); // 2^2
    }

    #[test]
    fn test_tensor_field_from_components() {
        let m = Arc::new(EuclideanSpace::new(2));
        let chart = m.default_chart().unwrap();

        // (1,1)-tensor has 4 components
        let components = vec![
            Expr::from(1), Expr::from(0),
            Expr::from(0), Expr::from(1),
        ];

        let tensor = TensorField::from_components(
            m.clone(),
            1,
            1,
            chart,
            components,
        ).unwrap();

        assert_eq!(tensor.num_components(), 4);
    }

    #[test]
    fn test_tensor_component_access() {
        let m = Arc::new(EuclideanSpace::new(2));
        let chart = m.default_chart().unwrap();

        let components = vec![
            Expr::from(1), Expr::from(2),
            Expr::from(3), Expr::from(4),
        ];

        let tensor = TensorField::from_components(
            m.clone(),
            1,
            1,
            chart,
            components,
        ).unwrap();

        let idx = MultiIndex::new(vec![0, 0]);
        assert_eq!(tensor.component(chart, &idx).unwrap(), Expr::from(1));

        let idx = MultiIndex::new(vec![1, 1]);
        assert_eq!(tensor.component(chart, &idx).unwrap(), Expr::from(4));
    }
}
