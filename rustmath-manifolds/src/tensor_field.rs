//! Tensor fields on manifolds
//!
//! This module implements tensor fields of arbitrary rank (p,q) where:
//! - p is the contravariant rank (number of upper indices)
//! - q is the covariant rank (number of lower indices)
//!
//! Components are stored densely per chart: n^(p+q) slots, the contravariant indices first,
//! the first index most significant ([`MultiIndex::to_flat`]). Every count n^k is checked
//! ([`component_count`]), and every component vector is allocated fallibly.
//!
//! References: J. M. Lee, *Introduction to Smooth Manifolds*, 2nd ed., GTM 218, Springer 2013,
//! ch. 12 "Tensors" (tensor product, contraction). MAGMA has no smooth-manifold chapter. The
//! SageMath module `sage.manifolds.differentiable.tensorfield` is the API reference.

use crate::errors::{ManifoldError, Result};
use crate::chart::Chart;
use crate::differentiable::DifferentiableManifold;
use crate::expr_zero::sum_signed;
use rustmath_symbolic::Expr;
use std::collections::HashMap;
use std::sync::Arc;

/// n^rank, the number of components of a rank-`rank` tensor on an n-dimensional manifold.
///
/// `Err(ComponentCountOverflow)` when n^rank does not fit in `usize`, or when its storage
/// (n^rank times `size_of::<Expr>()` bytes) exceeds `isize::MAX` bytes, the largest
/// allocation Rust allows. This is not a policy limit: callers such as the REPL keep their own.
///
/// ```
/// use rustmath_manifolds::tensor_field::component_count;
///
/// assert_eq!(component_count(1000, 2).unwrap(), 1_000_000);
/// assert_eq!(component_count(0, 0).unwrap(), 1);
/// assert!(component_count(usize::MAX, 2).is_err());
/// ```
pub fn component_count(dimension: usize, rank: usize) -> Result<usize> {
    let overflow = || ManifoldError::ComponentCountOverflow { dimension, rank };
    let count = match dimension {
        0 => {
            if rank == 0 {
                1
            } else {
                0
            }
        }
        1 => 1,
        n => {
            let r = u32::try_from(rank).map_err(|_| overflow())?;
            n.checked_pow(r).ok_or_else(overflow)?
        }
    };
    match count.checked_mul(std::mem::size_of::<Expr>()) {
        Some(bytes) if bytes <= isize::MAX as usize => Ok(count),
        _ => Err(overflow()),
    }
}

/// The flat index of `idx` in dimension `n` (first index most significant). The caller
/// guarantees that every index is below `n` and that n^len fits (the slot exists).
pub(crate) fn flat_index(idx: &[usize], n: usize) -> usize {
    idx.iter().fold(0, |acc, &i| acc * n + i)
}

/// A vector of `count` literal zeros. `Err(ComponentAllocationFailed)` when the allocator
/// refuses (`try_reserve_exact`).
pub(crate) fn zeroed_components(count: usize) -> Result<Vec<Expr>> {
    let mut v = Vec::new();
    v.try_reserve_exact(count)
        .map_err(|_| ManifoldError::ComponentAllocationFailed { count })?;
    v.resize(count, Expr::from(0));
    Ok(v)
}

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

    /// Convert to flat index for storage (first index most significant).
    ///
    /// Horner evaluation with checked arithmetic. `Err(InvalidIndex)` when an index is
    /// `>= dimension` (it would alias another slot) or the value overflows `usize`.
    pub fn to_flat(&self, dimension: usize) -> Result<usize> {
        let mut flat: usize = 0;
        for (pos, &idx) in self.indices.iter().enumerate() {
            if idx >= dimension {
                return Err(ManifoldError::InvalidIndex(format!(
                    "Index {} at position {} out of range for dimension {}",
                    idx, pos, dimension
                )));
            }
            flat = flat
                .checked_mul(dimension)
                .and_then(|f| f.checked_add(idx))
                .ok_or_else(|| ManifoldError::InvalidIndex(format!(
                    "The flat index of {:?} in dimension {} does not fit in a machine word",
                    self.indices, dimension
                )))?;
        }
        Ok(flat)
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
#[derive(Clone, Debug)]
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
        let rank = contravariant_rank.checked_add(covariant_rank).ok_or(
            ManifoldError::DegreeOverflow { left: contravariant_rank, right: covariant_rank },
        )?;
        let expected_size = component_count(manifold.dimension(), rank)?;

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

    /// Get the total rank p + q. `Err(DegreeOverflow)` when the sum overflows `usize`.
    pub fn total_rank(&self) -> Result<usize> {
        self.contravariant_rank.checked_add(self.covariant_rank).ok_or(
            ManifoldError::DegreeOverflow { left: self.contravariant_rank, right: self.covariant_rank },
        )
    }

    /// Get the number of components n^(p+q) ([`component_count`]).
    pub fn num_components(&self) -> Result<usize> {
        component_count(self.manifold.dimension(), self.total_rank()?)
    }

    /// Get all components in a chart (flattened)
    pub fn components(&self, chart: &Chart) -> Result<Vec<Expr>> {
        self.chart_components.get(chart.name())
            .cloned()
            .ok_or(ManifoldError::NoComponentsInChart)
    }

    /// The components in a chart, borrowed (no clone). `Err(NoComponentsInChart)`.
    pub(crate) fn components_ref(&self, chart: &Chart) -> Result<&[Expr]> {
        self.chart_components
            .get(chart.name())
            .map(|v| v.as_slice())
            .ok_or(ManifoldError::NoComponentsInChart)
    }

    /// A tensor field with the given dense components in one chart. The caller guarantees
    /// the length n^(p+q) (checked in debug builds only).
    pub(crate) fn from_dense_unchecked(
        manifold: Arc<DifferentiableManifold>,
        contravariant_rank: usize,
        covariant_rank: usize,
        chart: &Chart,
        components: Vec<Expr>,
    ) -> TensorField {
        debug_assert_eq!(
            contravariant_rank
                .checked_add(covariant_rank)
                .and_then(|r| component_count(manifold.dimension(), r).ok()),
            Some(components.len())
        );
        let mut tensor = Self::new(manifold, contravariant_rank, covariant_rank);
        tensor.chart_components.insert(chart.name().to_string(), components);
        tensor
    }

    /// The components in every chart.
    pub(crate) fn chart_components(&self) -> &HashMap<ChartId, Vec<Expr>> {
        &self.chart_components
    }

    /// The components in every chart, mutably. The caller keeps each length.
    pub(crate) fn chart_components_mut(&mut self) -> &mut HashMap<ChartId, Vec<Expr>> {
        &mut self.chart_components
    }

    /// A tensor field with the given components per chart. The caller guarantees each length.
    pub(crate) fn from_chart_components(
        manifold: Arc<DifferentiableManifold>,
        contravariant_rank: usize,
        covariant_rank: usize,
        map: HashMap<ChartId, Vec<Expr>>,
    ) -> TensorField {
        Self {
            manifold,
            contravariant_rank,
            covariant_rank,
            chart_components: map,
            name: None,
        }
    }

    /// `Err(InvalidIndex)` unless the multi-index has p + q entries.
    fn check_index_rank(&self, indices: &MultiIndex) -> Result<()> {
        let rank = self.total_rank()?;
        if indices.indices().len() != rank {
            return Err(ManifoldError::InvalidIndex(format!(
                "A rank-{} tensor needs {} indices, got {}",
                rank,
                rank,
                indices.indices().len()
            )));
        }
        Ok(())
    }

    /// Get a specific component
    pub fn component(&self, chart: &Chart, indices: &MultiIndex) -> Result<Expr> {
        let components = self.components_ref(chart)?;
        self.check_index_rank(indices)?;
        let flat_index = indices.to_flat(self.manifold.dimension())?;

        components.get(flat_index)
            .cloned()
            .ok_or_else(|| ManifoldError::InvalidIndex(
                format!("Flat index {} out of range for {} components", flat_index, components.len())
            ))
    }

    /// Set a specific component
    pub fn set_component(
        &mut self,
        chart: &Chart,
        indices: &MultiIndex,
        value: Expr,
    ) -> Result<()> {
        self.check_index_rank(indices)?;
        let flat_index = indices.to_flat(self.manifold.dimension())?;
        let chart_id = chart.name().to_string();

        if !self.chart_components.contains_key(&chart_id) {
            let zeros = zeroed_components(self.num_components()?)?;
            self.chart_components.insert(chart_id.clone(), zeros);
        }

        if let Some(components) = self.chart_components.get_mut(&chart_id) {
            if flat_index < components.len() {
                components[flat_index] = value;
                Ok(())
            } else {
                Err(ManifoldError::InvalidIndex(
                    format!("Flat index {} out of range for {} components", flat_index, components.len())
                ))
            }
        } else {
            Err(ManifoldError::NoComponentsInChart)
        }
    }

    /// Tensor contraction: contract the `contra_index`-th contravariant index with the
    /// `cov_index`-th covariant index (Einstein summation, Lee 2013, ch. 12).
    ///
    /// The result is a tensor of type (p-1, q-1) with
    /// (C T)^{..}_{..} = Σ_k T^{..k..}_{..k..}, the remaining indices in their order.
    pub fn contract(
        &self,
        contra_index: usize,
        cov_index: usize,
        chart: &Chart,
    ) -> Result<TensorField> {
        if contra_index >= self.contravariant_rank {
            return Err(ManifoldError::InvalidIndex(
                format!("Contravariant index {} out of range for rank {}", contra_index, self.contravariant_rank)
            ));
        }
        if cov_index >= self.covariant_rank {
            return Err(ManifoldError::InvalidIndex(
                format!("Covariant index {} out of range for rank {}", cov_index, self.covariant_rank)
            ));
        }

        let components = self.components_ref(chart)?;
        let n = self.manifold.dimension();
        let p = self.contravariant_rank;
        let rank = self.total_rank()?;
        let new_p = p - 1;
        let new_q = self.covariant_rank - 1;
        let out_rank = rank - 2;
        let count = component_count(n, out_rank)?;
        let mut out = zeroed_components(count)?;

        // positions of the two summed indices in the input index list
        let (pa, pb) = (contra_index, p + cov_index);
        let mut digits = vec![0usize; out_rank];
        let mut idx = vec![0usize; rank];
        for (o, slot) in out.iter_mut().enumerate() {
            // decode o into out_rank digits (first most significant); n >= 1 here, since
            // count is 0 for n = 0 and out_rank >= 1
            let mut rem = o;
            for d in digits.iter_mut().rev() {
                *d = rem % n;
                rem /= n;
            }
            let mut terms: Vec<(Expr, bool)> = Vec::new();
            for k in 0..n {
                let mut it = digits.iter();
                for (pos, i) in idx.iter_mut().enumerate() {
                    *i = if pos == pa || pos == pb { k } else { *it.next().unwrap_or(&0) };
                }
                let v = &components[flat_index(&idx, n)];
                if !v.is_zero() {
                    terms.push((v.clone(), false));
                }
            }
            if let Some(sum) = sum_signed(terms) {
                *slot = sum.simplify();
            }
        }

        Ok(TensorField::from_dense_unchecked(self.manifold.clone(), new_p, new_q, chart, out))
    }

    /// Tensor product with another tensor field
    ///
    /// For T of type (p1, q1) and S of type (p2, q2), T ⊗ S has type (p1 + p2, q1 + q2) and
    /// (T⊗S)^{I J}_{K L} = T^I_K S^J_L, stored in the layout
    /// [contra(T), contra(S), cov(T), cov(S)] of its type (Lee 2013, ch. 12).
    pub fn tensor_product(&self, other: &TensorField, chart: &Chart) -> Result<TensorField> {
        if !Arc::ptr_eq(&self.manifold, &other.manifold) {
            return Err(ManifoldError::DifferentManifolds);
        }

        let (p1, q1) = (self.contravariant_rank, self.covariant_rank);
        let (p2, q2) = (other.contravariant_rank, other.covariant_rank);
        let new_p = p1.checked_add(p2).ok_or(ManifoldError::DegreeOverflow { left: p1, right: p2 })?;
        let new_q = q1.checked_add(q2).ok_or(ManifoldError::DegreeOverflow { left: q1, right: q2 })?;
        let rank = new_p.checked_add(new_q).ok_or(ManifoldError::DegreeOverflow { left: new_p, right: new_q })?;

        let self_comps = self.components_ref(chart)?;
        let other_comps = other.components_ref(chart)?;

        let n = self.manifold.dimension();
        let count = component_count(n, rank)?;
        let nq1 = component_count(n, q1)?;
        let nq2 = component_count(n, q2)?;
        let np2 = component_count(n, p2)?;
        let mut out = zeroed_components(count)?;

        // the loops below run only when n >= 1 or the ranks are 0, so nq1, nq2 >= 1
        for (ti, tv) in self_comps.iter().enumerate() {
            if tv.is_zero() {
                continue;
            }
            let (ut, lt) = (ti / nq1, ti % nq1);
            for (si, sv) in other_comps.iter().enumerate() {
                if sv.is_zero() {
                    continue;
                }
                let (us, ls) = (si / nq2, si % nq2);
                // fits: o < count
                let o = ((ut * np2 + us) * nq1 + lt) * nq2 + ls;
                out[o] = (tv.clone() * sv.clone()).simplify();
            }
        }

        Ok(TensorField::from_dense_unchecked(self.manifold.clone(), new_p, new_q, chart, out))
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

    /// Check if all components are approximately zero
    ///
    /// Returns `true` if all tensor components have absolute value less than epsilon.
    /// Uses epsilon = 1e-10 as the tolerance threshold.
    ///
    /// # Returns
    ///
    /// `true` if all components are within epsilon of zero, `false` otherwise.
    /// If a component cannot be evaluated to a numeric value, returns `false`.
    pub fn is_approximately_zero(&self) -> bool {
        const EPSILON: f64 = 1e-10;

        for comps in self.chart_components.values() {
            for comp in comps {
                // Try to evaluate the expression to f64
                let val = match comp {
                    Expr::Integer(ref n) => n.to_f64().unwrap_or(f64::INFINITY),
                    Expr::Rational(ref r) => r.to_f64().unwrap_or(f64::INFINITY),
                    _ => return false, // If we can't evaluate to a number, assume not approximately zero
                };

                if val.abs() >= EPSILON {
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
        let flat = idx.to_flat(3).unwrap();
        let idx2 = MultiIndex::from_flat(flat, 3, 3);
        assert_eq!(idx, idx2);
    }

    #[test]
    fn test_tensor_field_creation() {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let tensor = TensorField::new(m.clone(), 1, 1);

        assert_eq!(tensor.contravariant_rank(), 1);
        assert_eq!(tensor.covariant_rank(), 1);
        assert_eq!(tensor.total_rank().unwrap(), 2);
        assert_eq!(tensor.num_components().unwrap(), 4); // 2^2
    }

    #[test]
    fn test_tensor_field_from_components() {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
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

        assert_eq!(tensor.num_components().unwrap(), 4);
    }

    #[test]
    fn test_tensor_component_access() {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
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

    fn r2() -> Arc<DifferentiableManifold> {
        Arc::new(EuclideanSpace::new(2).manifold().clone())
    }

    #[test]
    fn test_component_count_checked() {
        assert!(matches!(
            component_count(1 << 33, 2),
            Err(ManifoldError::ComponentCountOverflow { dimension, rank: 2 }) if dimension == 1 << 33
        ));
        // rank 2^32 + 1 must not be truncated to 1
        assert!(component_count(2, (1 << 32) + 1).is_err());
        assert_eq!(component_count(0, 0).unwrap(), 1);
        assert_eq!(component_count(0, 5).unwrap(), 0);
        assert_eq!(component_count(1, usize::MAX).unwrap(), 1);
        assert_eq!(component_count(1000, 2).unwrap(), 1_000_000);
        // the byte boundary, derived from the size of Expr (64 bytes today: k = 56)
        let max = isize::MAX as usize / std::mem::size_of::<Expr>();
        let k = max.ilog2() as usize;
        assert_eq!(component_count(2, k).unwrap(), 1 << k);
        assert!(matches!(
            component_count(2, k + 1),
            Err(ManifoldError::ComponentCountOverflow { dimension: 2, .. })
        ));
    }

    #[test]
    fn test_num_components_checked() {
        let t = TensorField::new(r2(), 0, (1 << 32) + 1);
        assert!(matches!(t.num_components(), Err(ManifoldError::ComponentCountOverflow { .. })));
        let big = Arc::new(DifferentiableManifold::new("M", 1 << 33));
        let t = TensorField::new(big, 0, 2);
        assert!(matches!(t.num_components(), Err(ManifoldError::ComponentCountOverflow { .. })));
        let t = TensorField::new(r2(), usize::MAX, 1);
        assert_eq!(
            t.total_rank(),
            Err(ManifoldError::DegreeOverflow { left: usize::MAX, right: 1 })
        );
    }

    #[test]
    fn test_from_components_huge_rank() {
        let big = Arc::new(DifferentiableManifold::new("M", 1 << 33));
        let c = Chart::new("c", 2, vec!["x", "y"]).unwrap();
        let r = TensorField::from_components(big, 0, 2, &c, vec![]);
        assert!(matches!(r, Err(ManifoldError::ComponentCountOverflow { .. })));
        let r = TensorField::from_components(r2(), usize::MAX, 1, &c, vec![]);
        assert!(matches!(r, Err(ManifoldError::DegreeOverflow { .. })));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn test_zeroed_components_allocation_failure() {
        // about 2^63 bytes, beyond any 64-bit address space
        let max = isize::MAX as usize / std::mem::size_of::<Expr>();
        assert_eq!(
            zeroed_components(max).err(),
            Some(ManifoldError::ComponentAllocationFailed { count: max })
        );
        assert_eq!(zeroed_components(3).unwrap(), vec![Expr::from(0); 3]);
    }

    #[test]
    fn test_multi_index_checked() {
        // at HEAD (0, 3) with n = 3 gave 3, the slot of (1, 0)
        assert!(matches!(MultiIndex::new(vec![0, 3]).to_flat(3), Err(ManifoldError::InvalidIndex(_))));
        assert_eq!(MultiIndex::new(vec![1, 0]).to_flat(3).unwrap(), 3);
        assert!(matches!(
            MultiIndex::new(vec![1, 1, 1]).to_flat(1 << 40),
            Err(ManifoldError::InvalidIndex(_))
        ));
        let m = r2();
        let c = m.default_chart().unwrap().clone();
        let mut t = TensorField::from_components(m.clone(), 0, 2, &c, vec![Expr::from(0); 4]).unwrap();
        assert!(matches!(
            t.set_component(&c, &MultiIndex::new(vec![0, 3]), Expr::from(7)),
            Err(ManifoldError::InvalidIndex(_))
        ));
        assert_eq!(t.components(&c).unwrap(), vec![Expr::from(0); 4]);
        assert!(matches!(t.component(&c, &MultiIndex::new(vec![1])), Err(ManifoldError::InvalidIndex(_))));
        assert!(matches!(
            t.set_component(&c, &MultiIndex::new(vec![1, 0, 0]), Expr::from(7)),
            Err(ManifoldError::InvalidIndex(_))
        ));
        t.set_component(&c, &MultiIndex::new(vec![1, 0]), Expr::from(7)).unwrap();
        assert_eq!(t.component(&c, &MultiIndex::new(vec![1, 0])).unwrap(), Expr::from(7));
        // a chart without components gets checked zeros first
        let mut u = TensorField::new(m, 1, 1);
        u.set_component(&c, &MultiIndex::new(vec![0, 1]), Expr::from(5)).unwrap();
        assert_eq!(
            u.components(&c).unwrap(),
            vec![Expr::from(0), Expr::from(5), Expr::from(0), Expr::from(0)]
        );
    }

    #[test]
    fn test_contract_trace() {
        // gp: trace(matdiagonal([5,7])) = 12; at HEAD every slot was n(n-1)/2 = 1
        let m = r2();
        let c = m.default_chart().unwrap().clone();
        let t = TensorField::from_components(
            m.clone(), 1, 1, &c,
            vec![Expr::from(5), Expr::from(0), Expr::from(0), Expr::from(7)],
        ).unwrap();
        let tr = t.contract(0, 0, &c).unwrap();
        assert_eq!(tr.contravariant_rank(), 0);
        assert_eq!(tr.covariant_rank(), 0);
        assert_eq!(tr.components(&c).unwrap(), vec![Expr::from(12)]);

        // T^i_{jk} = δ^i_j a_k on R^3: contract(0, 0) = 3 a_k, contract(0, 1) = a_j
        let m3 = Arc::new(EuclideanSpace::new(3).manifold().clone());
        let c3 = m3.default_chart().unwrap().clone();
        let a: Vec<Expr> = ["a0", "a1", "a2"].iter().map(|s| Expr::symbol(*s)).collect();
        let mut comps = vec![Expr::from(0); 27];
        for i in 0..3 {
            for k in 0..3 {
                comps[(i * 3 + i) * 3 + k] = a[k].clone();
            }
        }
        let t = TensorField::from_components(m3.clone(), 1, 2, &c3, comps).unwrap();
        let c00 = t.contract(0, 0, &c3).unwrap();
        assert_eq!((c00.contravariant_rank(), c00.covariant_rank()), (0, 1));
        let v = c00.components(&c3).unwrap();
        for k in 0..3 {
            let diff = v[k].clone() - Expr::from(3) * a[k].clone();
            assert_eq!(crate::expr_zero::zero_test(&diff), crate::expr_zero::ZeroTest::Zero, "{}", v[k]);
        }
        let c01 = t.contract(0, 1, &c3).unwrap();
        let v = c01.components(&c3).unwrap();
        for j in 0..3 {
            assert_eq!(v[j], a[j]);
        }
        assert!(matches!(t.contract(1, 0, &c3), Err(ManifoldError::InvalidIndex(_))));
        assert!(matches!(t.contract(0, 2, &c3), Err(ManifoldError::InvalidIndex(_))));
    }

    #[test]
    fn test_tensor_product_index_order() {
        // (0,1) ⊗ (1,0), t = (3, 4), s = (1, 2): (t⊗s)^i_j = s^i t_j = [3, 4, 6, 8]
        // (HEAD wrote the layout [cov(T), contra(S)]: [3, 6, 4, 8])
        let m = r2();
        let c = m.default_chart().unwrap().clone();
        let t = TensorField::from_components(m.clone(), 0, 1, &c, vec![Expr::from(3), Expr::from(4)]).unwrap();
        let s = TensorField::from_components(m.clone(), 1, 0, &c, vec![Expr::from(1), Expr::from(2)]).unwrap();
        let ts = t.tensor_product(&s, &c).unwrap();
        assert_eq!((ts.contravariant_rank(), ts.covariant_rank()), (1, 1));
        assert_eq!(
            ts.components(&c).unwrap(),
            vec![Expr::from(3), Expr::from(4), Expr::from(6), Expr::from(8)]
        );
        // s ⊗ t has the same type and the same layout here: s^i t_j
        let st = s.tensor_product(&t, &c).unwrap();
        assert_eq!(st.components(&c).unwrap(), ts.components(&c).unwrap());
        // (1,1) ⊗ (0,1): (T⊗w)^i_{jk} = T^i_j w_k
        let tt = TensorField::from_components(
            m.clone(), 1, 1, &c,
            vec![Expr::from(1), Expr::from(2), Expr::from(3), Expr::from(5)],
        ).unwrap();
        let p = tt.tensor_product(&t, &c).unwrap();
        let v = p.components(&c).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let tij = [1i64, 2, 3, 5][i * 2 + j];
                    let wk = [3i64, 4][k];
                    assert_eq!(v[(i * 2 + j) * 2 + k], Expr::from(tij * wk));
                }
            }
        }
    }
}
