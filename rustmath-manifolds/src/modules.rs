//! Module structures for manifold fields
//!
//! This module implements the algebraic structures (Parents) that contain
//! various types of fields on manifolds:
//! - DiffScalarFieldAlgebra: The algebra C^∞(M) of smooth scalar fields
//! - VectorFieldModule: The module 𝔛(M) of vector fields over C^∞(M)
//! - VectorFieldFreeModule: Free module for parallelizable manifolds
//! - TensorFieldModule: The module T^(p,q)(M) of tensor fields
//! - DiffFormModule: The module Ω^p(M) of differential forms

use crate::scalar_field::ScalarFieldEnhanced;
use crate::vector_field::VectorField;
use crate::tensor_field::TensorField;
use crate::diff_form::DiffForm;
use crate::differentiable::DifferentiableManifold;
use crate::errors::{ManifoldError, Result};
use crate::traits::{
    DiffScalarFieldAlgebraTrait, ScalarFieldAlgebraTrait,
    VectorFieldModuleTrait, VectorFieldFreeModuleTrait,
    TensorFieldModuleTrait, DifferentiableManifoldTrait,
};
use rustmath_core::{Parent, ParentWithBasis, Ring};
use std::sync::Arc;
use std::collections::HashMap;
use std::fmt;

// ============================================================================
// FINITENESS (MAGMA Handbook Ch. 17 §17.4.3 `IsFinite`)
// ============================================================================
//
// Every parent below is a module of sections Γ(E) of a real vector bundle E over M
// (C^∞(M) = Γ(M × R), 𝔛(M) = Γ(TM), T^(p,q)(M), Ω^p(M) = Γ(Λ^p T*M)), so it is a module
// over C^∞(M).
//
// * M is nonempty: the crate models every manifold as nonempty
//   (`TopologicalManifold::point` accepts every coordinate vector of length dim M).
// * C^∞(M) contains the constant functions c ∈ R, pairwise distinct on a nonempty M, so a
//   copy of R (characteristic 0): it is infinite.
// * If rank E >= 1, pick p ∈ M and v ≠ 0 in the fibre E_p. The extension lemma for vector
//   bundles gives a global section s with s(p) = v: a local section through v in a
//   trivialisation near p, times a bump function supported there (J. M. Lee, *Introduction
//   to Smooth Manifolds*, 2nd ed. (2013), Ch. 8 and Ch. 10, the extension lemmas for vector
//   fields and for vector bundles). The sections c·s, c ∈ R, are pairwise distinct (evaluate
//   at p): Γ(E) is infinite. The argument needs no differentiability: continuous bump
//   functions exist on every topological manifold (max(0, 1 − |φ|) in a chart φ centred at p
//   whose image contains the closed unit ball, extended by 0).
// * TM, T*M and their tensor and exterior powers exist only for a differentiability class
//   >= 1 (for a C^k manifold, k >= 1, TM is a C^(k−1) bundle). On a C^0 manifold the modules
//   built from them are not defined, so no claim is made. The trivial bundle M × R exists for
//   every class; its module of sections is the scalar algebra (C^∞(M) in this crate's naming;
//   C^k(M) on a C^k manifold), which is also T^(0,0) and Ω^0, so it is infinite on every
//   manifold.
// * If rank E = 0, Γ(E) = {0}; the element types here do not enforce that, so no claim.

/// `Parent::is_finite` of a module of sections of a vector bundle built from TM over
/// `manifold`; `rank_is_nonzero` states whether the bundle has rank >= 1 (see the proof
/// above).
fn sections_is_finite(manifold: &DifferentiableManifold, rank_is_nonzero: bool) -> Option<bool> {
    // TM, T*M and their tensor and exterior powers need class >= 1; on a C^0 manifold these
    // modules are not defined, so no claim. Callers handle the trivial bundle M × R first.
    if rank_is_nonzero && manifold.differentiability_class() != Some(0) {
        Some(false)
    } else {
        None
    }
}

// ============================================================================
// SCALAR FIELD ALGEBRA
// ============================================================================

/// The algebra C^∞(M) of smooth scalar fields on a differentiable manifold
///
/// This structure implements the Parent trait with ScalarFieldEnhanced as elements.
/// It provides the algebraic structure for scalar fields including:
/// - Addition and multiplication of scalar fields
/// - Zero element (constant 0 function)
/// - One element (constant 1 function)
///
/// # Examples
///
/// ```
/// use rustmath_manifolds::{DiffScalarFieldAlgebra, EuclideanSpace};
/// use rustmath_core::Parent;
/// use std::sync::Arc;
///
/// let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
/// let algebra = DiffScalarFieldAlgebra::new(m.clone());
///
/// let zero = algebra.zero().unwrap();
/// let one = algebra.one().unwrap();
/// ```
#[derive(Clone)]
pub struct DiffScalarFieldAlgebra {
    /// The underlying differentiable manifold
    manifold: Arc<DifferentiableManifold>,
    /// Cache for commonly used scalar fields
    cache: HashMap<String, ScalarFieldEnhanced>,
}

impl DiffScalarFieldAlgebra {
    /// Create a new scalar field algebra for a manifold
    ///
    /// # Arguments
    ///
    /// * `manifold` - The differentiable manifold M
    ///
    /// # Returns
    ///
    /// The algebra C^∞(M) of smooth functions on M
    pub fn new(manifold: Arc<DifferentiableManifold>) -> Self {
        Self {
            manifold,
            cache: HashMap::new(),
        }
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Create a scalar field from a constant value
    pub fn constant(&self, value: f64) -> ScalarFieldEnhanced {
        ScalarFieldEnhanced::constant(self.manifold.clone(), value)
    }

    /// Get a coordinate function
    ///
    /// Returns the i-th coordinate function from the default chart
    pub fn coordinate_function(&self, index: usize) -> Result<ScalarFieldEnhanced> {
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        if index >= self.manifold.dimension() {
            return Err(ManifoldError::InvalidIndex(
                format!("Index {} out of range for manifold dimension {}", index, self.manifold.dimension())
            ));
        }

        let symbol = chart.coordinate_symbols()[index].clone();
        let expr = rustmath_symbolic::Expr::Symbol(symbol);

        Ok(ScalarFieldEnhanced::from_expr(self.manifold.clone(), chart, expr))
    }
}

impl fmt::Debug for DiffScalarFieldAlgebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiffScalarFieldAlgebra")
            .field("manifold", &self.manifold.name())
            .field("dimension", &self.manifold.dimension())
            .finish()
    }
}

impl fmt::Display for DiffScalarFieldAlgebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C^∞({})", self.manifold.name())
    }
}

impl Parent for DiffScalarFieldAlgebra {
    type Element = ScalarFieldEnhanced;

    fn contains(&self, element: &Self::Element) -> bool {
        // Check if the scalar field is defined on this manifold
        Arc::ptr_eq(element.manifold(), &self.manifold)
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(ScalarFieldEnhanced::constant(self.manifold.clone(), 0.0))
    }

    fn one(&self) -> Option<Self::Element> {
        Some(ScalarFieldEnhanced::constant(self.manifold.clone(), 1.0))
    }

    fn cardinality(&self) -> Option<usize> {
        None // Infinite (uncountably many smooth functions)
    }

    /// `Some(false)`: the constant functions give a copy of R in C^∞(M) (M is nonempty;
    /// see the finiteness notes at the top of this module). Constants are continuous, so
    /// this holds for every differentiability class.
    fn is_finite(&self) -> Option<bool> {
        Some(false)
    }

    fn name(&self) -> String {
        format!("C^∞({})", self.manifold.name())
    }
}

/// The ring predicates of C^∞(M) (MAGMA Ch. 17 §17.4.2-3), for the nonempty manifold the
/// crate models (see the finiteness notes at the top of this module).
impl rustmath_category::RingParentMethods for DiffScalarFieldAlgebra {
    /// `Some(0)`: the constant functions give a copy of R in C^∞(M).
    fn characteristic(&self) -> Option<u64> {
        Some(0)
    }

    /// The product of real-valued functions is pointwise, so commutative.
    fn is_commutative(&self) -> Option<bool> {
        Some(true)
    }

    /// For dim M >= 1 not a domain: two smooth bump functions with disjoint nonempty
    /// supports are nonzero and multiply to 0. For dim M = 0, C^∞(M) = R^(#points), a field
    /// iff M is one point; the point count is not known here, so `None`.
    fn is_integral_domain(&self) -> Option<bool> {
        if self.manifold.dimension() >= 1 {
            Some(false)
        } else {
            None
        }
    }

    /// A non-domain is not a field; for dim M = 0 see `is_integral_domain`.
    fn is_field(&self) -> Option<bool> {
        rustmath_category::RingParentMethods::is_integral_domain(self)
    }
}

impl rustmath_category::CommutativeRingParentMethods for DiffScalarFieldAlgebra {}

impl ScalarFieldAlgebraTrait for DiffScalarFieldAlgebra {
    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait> {
        // We need to convert Arc<DifferentiableManifold> to Arc<dyn DifferentiableManifoldTrait>
        // For now, this is a limitation - we'll need to adjust the design
        // to use Arc<dyn DifferentiableManifoldTrait> throughout
        unimplemented!("Trait object conversion needed")
    }
}

impl DiffScalarFieldAlgebraTrait for DiffScalarFieldAlgebra {
    // All scalar fields are smooth by definition
}

// ============================================================================
// VECTOR FIELD MODULE
// ============================================================================

/// The module 𝔛(M) of vector fields on a differentiable manifold
///
/// Vector fields form a module over the ring C^∞(M) of smooth scalar fields.
/// This implements the Parent trait with VectorField as elements.
///
/// # Examples
///
/// ```
/// use rustmath_manifolds::{VectorFieldModule, EuclideanSpace};
/// use rustmath_core::Parent;
/// use std::sync::Arc;
///
/// let m = Arc::new(EuclideanSpace::new(3).manifold().clone());
/// let module = VectorFieldModule::new(m.clone());
///
/// let zero_field = module.zero().unwrap();
/// ```
#[derive(Clone)]
pub struct VectorFieldModule {
    /// The underlying differentiable manifold
    manifold: Arc<DifferentiableManifold>,
    /// The scalar ring C^∞(M)
    scalar_ring: Arc<DiffScalarFieldAlgebra>,
}

impl VectorFieldModule {
    /// Create a new vector field module for a manifold
    ///
    /// # Arguments
    ///
    /// * `manifold` - The differentiable manifold M
    ///
    /// # Returns
    ///
    /// The module 𝔛(M) of vector fields on M
    pub fn new(manifold: Arc<DifferentiableManifold>) -> Self {
        let scalar_ring = Arc::new(DiffScalarFieldAlgebra::new(manifold.clone()));
        Self {
            manifold,
            scalar_ring,
        }
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the scalar ring C^∞(M)
    pub fn scalar_ring(&self) -> &Arc<DiffScalarFieldAlgebra> {
        &self.scalar_ring
    }

    /// Create a vector field from components in the default chart
    pub fn from_components(&self, components: Vec<rustmath_symbolic::Expr>) -> Result<VectorField> {
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        VectorField::from_components(self.manifold.clone(), chart, components)
    }

    /// Get the contravariant rank (always 1 for vector fields)
    pub fn contravariant_rank(&self) -> usize {
        1
    }

    /// Get the covariant rank (always 0 for vector fields)
    pub fn covariant_rank(&self) -> usize {
        0
    }
}

impl fmt::Debug for VectorFieldModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorFieldModule")
            .field("manifold", &self.manifold.name())
            .field("dimension", &self.manifold.dimension())
            .finish()
    }
}

impl fmt::Display for VectorFieldModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "𝔛({})", self.manifold.name())
    }
}

impl Parent for VectorFieldModule {
    type Element = VectorField;

    fn contains(&self, element: &Self::Element) -> bool {
        Arc::ptr_eq(element.manifold(), &self.manifold)
    }

    fn zero(&self) -> Option<Self::Element> {
        let chart = self.manifold.default_chart()?;
        let zero_comps = vec![rustmath_symbolic::Expr::from(0); self.manifold.dimension()];
        VectorField::from_components(self.manifold.clone(), chart, zero_comps).ok()
    }

    fn cardinality(&self) -> Option<usize> {
        None // Infinite
    }

    /// `Some(false)` for dim M >= 1 on a manifold of class >= 1 (TM has rank dim M; see the
    /// finiteness notes at the top of this module); `None` for dim M = 0 (𝔛(M) = 0) or a C^0
    /// manifold (no TM).
    fn is_finite(&self) -> Option<bool> {
        sections_is_finite(&self.manifold, self.manifold.dimension() >= 1)
    }

    fn name(&self) -> String {
        format!("𝔛({})", self.manifold.name())
    }
}

impl VectorFieldModuleTrait for VectorFieldModule {
    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait> {
        unimplemented!("Trait object conversion needed")
    }

    fn rank(&self) -> (usize, usize) {
        (1, 0) // Vector fields are (1,0) tensors
    }
}

// ============================================================================
// FREE VECTOR FIELD MODULE (Parallelizable case)
// ============================================================================

/// Free module of vector fields on a parallelizable manifold
///
/// When a manifold admits a global frame (is parallelizable), vector fields
/// form a free module with a finite basis.
///
/// # Examples
///
/// ```
/// use rustmath_manifolds::{VectorFieldFreeModule, EuclideanSpace};
/// use rustmath_manifolds::traits::VectorFieldFreeModuleTrait;
/// use std::sync::Arc;
///
/// let m = Arc::new(EuclideanSpace::new(3).manifold().clone());
/// let module = VectorFieldFreeModule::new(m.clone()).unwrap();
///
/// assert_eq!(module.rank_value(), 3);
/// ```
#[derive(Clone)]
pub struct VectorFieldFreeModule {
    /// The underlying vector field module
    base_module: VectorFieldModule,
    /// The global frame (basis vector fields)
    frame: Vec<VectorField>,
}

impl VectorFieldFreeModule {
    /// Create a new free vector field module with a given frame
    ///
    /// # Arguments
    ///
    /// * `manifold` - The parallelizable manifold
    /// * `frame` - The global frame (basis vector fields)
    ///
    /// # Returns
    ///
    /// The free module with the given basis
    pub fn new_with_frame(
        manifold: Arc<DifferentiableManifold>,
        frame: Vec<VectorField>,
    ) -> Result<Self> {
        if frame.len() != manifold.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: manifold.dimension(),
                actual: frame.len(),
            });
        }

        let base_module = VectorFieldModule::new(manifold);

        Ok(Self {
            base_module,
            frame,
        })
    }

    /// Create a free module with the coordinate frame from the default chart
    pub fn new(manifold: Arc<DifferentiableManifold>) -> Result<Self> {
        let chart = manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        // Create coordinate vector fields ∂/∂x^i
        let mut frame = Vec::with_capacity(manifold.dimension());
        for i in 0..manifold.dimension() {
            let mut components = vec![rustmath_symbolic::Expr::from(0); manifold.dimension()];
            components[i] = rustmath_symbolic::Expr::from(1);

            let field = VectorField::from_components(manifold.clone(), chart, components)?;
            frame.push(field);
        }

        Self::new_with_frame(manifold, frame)
    }

    /// Get the global frame
    pub fn frame(&self) -> &[VectorField] {
        &self.frame
    }

    /// Get the i-th basis vector field
    pub fn basis_vector_field(&self, index: usize) -> Option<&VectorField> {
        self.frame.get(index)
    }
}

impl fmt::Debug for VectorFieldFreeModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorFieldFreeModule")
            .field("manifold", &self.base_module.manifold.name())
            .field("rank", &self.frame.len())
            .finish()
    }
}

impl fmt::Display for VectorFieldFreeModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "𝔛_free({})", self.base_module.manifold.name())
    }
}

impl Parent for VectorFieldFreeModule {
    type Element = VectorField;

    fn contains(&self, element: &Self::Element) -> bool {
        self.base_module.contains(element)
    }

    fn zero(&self) -> Option<Self::Element> {
        self.base_module.zero()
    }

    fn cardinality(&self) -> Option<usize> {
        None
    }

    /// As for [`VectorFieldModule`]: `Some(false)` for a frame of length dim M >= 1 on a
    /// manifold of class >= 1.
    fn is_finite(&self) -> Option<bool> {
        self.base_module.is_finite()
    }

    fn name(&self) -> String {
        format!("𝔛_free({})", self.base_module.manifold.name())
    }
}

impl VectorFieldModuleTrait for VectorFieldFreeModule {
    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait> {
        unimplemented!("Trait object conversion needed")
    }

    fn rank(&self) -> (usize, usize) {
        (1, 0)
    }
}

impl VectorFieldFreeModuleTrait for VectorFieldFreeModule {
    fn rank_value(&self) -> usize {
        self.frame.len()
    }
}

impl ParentWithBasis for VectorFieldFreeModule {
    type BasisIndex = usize;

    fn dimension(&self) -> Option<usize> {
        Some(self.frame.len())
    }

    fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element> {
        self.frame.get(*index).cloned()
    }

    fn basis_indices(&self) -> Vec<Self::BasisIndex> {
        (0..self.frame.len()).collect()
    }
}

// ============================================================================
// TENSOR FIELD MODULE
// ============================================================================

/// The module T^(p,q)(M) of tensor fields of type (p,q)
///
/// Tensor fields of type (p,q) have p contravariant indices and q covariant indices.
/// They form a module over C^∞(M).
///
/// # Examples
///
/// ```
/// use rustmath_manifolds::{TensorFieldModule, EuclideanSpace};
/// use std::sync::Arc;
///
/// let m = Arc::new(EuclideanSpace::new(3).manifold().clone());
///
/// // Module of (1,1) tensor fields
/// let module_1_1 = TensorFieldModule::new(m.clone(), 1, 1);
///
/// assert_eq!(module_1_1.contravariant_rank(), 1);
/// assert_eq!(module_1_1.covariant_rank(), 1);
/// assert_eq!(module_1_1.total_rank(), 2);
/// ```
#[derive(Clone)]
pub struct TensorFieldModule {
    /// The underlying manifold
    manifold: Arc<DifferentiableManifold>,
    /// Contravariant rank p
    contravariant_rank: usize,
    /// Covariant rank q
    covariant_rank: usize,
    /// The scalar ring C^∞(M)
    scalar_ring: Arc<DiffScalarFieldAlgebra>,
}

impl TensorFieldModule {
    /// Create a new tensor field module of type (p,q)
    ///
    /// # Arguments
    ///
    /// * `manifold` - The differentiable manifold M
    /// * `p` - Contravariant rank
    /// * `q` - Covariant rank
    ///
    /// # Returns
    ///
    /// The module T^(p,q)(M)
    pub fn new(manifold: Arc<DifferentiableManifold>, p: usize, q: usize) -> Self {
        let scalar_ring = Arc::new(DiffScalarFieldAlgebra::new(manifold.clone()));
        Self {
            manifold,
            contravariant_rank: p,
            covariant_rank: q,
            scalar_ring,
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

    /// Get the scalar ring
    pub fn scalar_ring(&self) -> &Arc<DiffScalarFieldAlgebra> {
        &self.scalar_ring
    }
}

impl fmt::Debug for TensorFieldModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorFieldModule")
            .field("manifold", &self.manifold.name())
            .field("type", &format!("({},{})", self.contravariant_rank, self.covariant_rank))
            .finish()
    }
}

impl fmt::Display for TensorFieldModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T^({},{})({})",
            self.contravariant_rank,
            self.covariant_rank,
            self.manifold.name())
    }
}

impl Parent for TensorFieldModule {
    type Element = TensorField;

    fn contains(&self, element: &Self::Element) -> bool {
        Arc::ptr_eq(element.manifold(), &self.manifold)
            && element.contravariant_rank() == self.contravariant_rank
            && element.covariant_rank() == self.covariant_rank
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(TensorField::zero(
            self.manifold.clone(),
            self.contravariant_rank,
            self.covariant_rank,
        ))
    }

    fn cardinality(&self) -> Option<usize> {
        None
    }

    /// `Some(false)` for p + q = 0 (T^(0,0)(M) = Γ(M × R) contains the constants, for every
    /// class) and for p + q >= 1 when n = dim M >= 1 and the class is >= 1 (T^(p,q)M has rank
    /// n^(p+q) >= 1). `None` for p + q >= 1 when n = 0 (the zero bundle) or on a C^0 manifold
    /// (no TM). See the finiteness notes at the top of this module.
    fn is_finite(&self) -> Option<bool> {
        // T^(0,0)M = M × R: its sections contain the constants for every class (as C^∞(M)).
        if self.total_rank() == 0 {
            return Some(false);
        }
        sections_is_finite(&self.manifold, self.manifold.dimension() >= 1)
    }

    fn name(&self) -> String {
        format!("T^({},{})({})",
            self.contravariant_rank,
            self.covariant_rank,
            self.manifold.name())
    }
}

impl TensorFieldModuleTrait for TensorFieldModule {
    fn contravariant_rank(&self) -> usize {
        self.contravariant_rank
    }

    fn covariant_rank(&self) -> usize {
        self.covariant_rank
    }
}

// ============================================================================
// DIFFERENTIAL FORM MODULE
// ============================================================================

/// The module Ω^p(M) of differential p-forms
///
/// Differential forms are totally antisymmetric covariant tensor fields.
/// They form the de Rham complex with the exterior derivative.
///
/// # Examples
///
/// ```
/// use rustmath_manifolds::{DiffFormModule, EuclideanSpace};
/// use std::sync::Arc;
///
/// let m = Arc::new(EuclideanSpace::new(3).manifold().clone());
///
/// // Module of 1-forms
/// let omega_1 = DiffFormModule::new(m.clone(), 1);
/// assert_eq!(omega_1.degree(), 1);
///
/// // Module of 2-forms
/// let omega_2 = DiffFormModule::new(m.clone(), 2);
/// assert_eq!(omega_2.degree(), 2);
/// ```
#[derive(Clone)]
pub struct DiffFormModule {
    /// The underlying manifold
    manifold: Arc<DifferentiableManifold>,
    /// The degree p of the forms
    degree: usize,
    /// The scalar ring C^∞(M)
    scalar_ring: Arc<DiffScalarFieldAlgebra>,
}

impl DiffFormModule {
    /// Create a new differential form module of degree p
    ///
    /// # Arguments
    ///
    /// * `manifold` - The differentiable manifold M
    /// * `p` - The degree of the forms
    ///
    /// # Returns
    ///
    /// The module Ω^p(M)
    pub fn new(manifold: Arc<DifferentiableManifold>, degree: usize) -> Self {
        let scalar_ring = Arc::new(DiffScalarFieldAlgebra::new(manifold.clone()));
        Self {
            manifold,
            degree,
            scalar_ring,
        }
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the degree of forms in this module
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the scalar ring
    pub fn scalar_ring(&self) -> &Arc<DiffScalarFieldAlgebra> {
        &self.scalar_ring
    }
}

impl fmt::Debug for DiffFormModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiffFormModule")
            .field("manifold", &self.manifold.name())
            .field("degree", &self.degree)
            .finish()
    }
}

impl fmt::Display for DiffFormModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ω^{}({})", self.degree, self.manifold.name())
    }
}

impl Parent for DiffFormModule {
    type Element = DiffForm;

    fn contains(&self, element: &Self::Element) -> bool {
        Arc::ptr_eq(element.manifold(), &self.manifold)
            && element.degree() == self.degree
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(DiffForm::zero(self.manifold.clone(), self.degree))
    }

    fn cardinality(&self) -> Option<usize> {
        None
    }

    /// `Some(false)` for p = 0 (Ω^0(M) = Γ(M × R) contains the constants, for every class) and
    /// for 1 <= p <= n = dim M when the class is >= 1 (Λ^p T*M has rank C(n, p) >= 1). `None`
    /// for p > n (Ω^p(M) = 0) and for p >= 1 on a C^0 manifold (no T*M). See the finiteness
    /// notes at the top of this module.
    fn is_finite(&self) -> Option<bool> {
        // Ω^0(M) = Γ(M × R) ⊇ constants, for every class.
        if self.degree == 0 {
            return Some(false);
        }
        sections_is_finite(&self.manifold, self.degree <= self.manifold.dimension())
    }

    fn name(&self) -> String {
        format!("Ω^{}({})", self.degree, self.manifold.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::EuclideanSpace;

    // C^∞(R^2): characteristic 0 (constants), commutative (pointwise product), not a domain
    // (two bump functions with disjoint supports multiply to 0), so not a field, PID, UFD.
    // At c79a582 no ring-predicate impl existed (R1 hand-off F.2, review 2 N4).
    #[test]
    fn scalar_field_algebra_ring_predicates() {
        use rustmath_category::{CommutativeRingParentMethods, RingParentMethods};
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let algebra = DiffScalarFieldAlgebra::new(m);
        assert_eq!(algebra.characteristic(), Some(0));
        assert_eq!(algebra.is_commutative(), Some(true));
        assert_eq!(algebra.is_integral_domain(), Some(false));
        assert_eq!(algebra.is_field(), Some(false));
        assert_eq!(algebra.is_pid(), Some(false));
        let point = Arc::new(DifferentiableManifold::new("P", 0));
        let a0 = DiffScalarFieldAlgebra::new(point);
        assert_eq!(a0.is_integral_domain(), None);
        assert_eq!(a0.is_field(), None);
    }

    #[test]
    fn test_scalar_field_algebra_creation() {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let algebra = DiffScalarFieldAlgebra::new(m.clone());

        assert_eq!(algebra.manifold().dimension(), 2);
        assert_eq!(algebra.name(), "C^∞(ℝ^2)");
    }

    #[test]
    fn test_scalar_field_algebra_zero_one() {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let algebra = DiffScalarFieldAlgebra::new(m.clone());

        let zero = algebra.zero().unwrap();
        let one = algebra.one().unwrap();

        assert!(algebra.contains(&zero));
        assert!(algebra.contains(&one));
    }

    #[test]
    fn test_vector_field_module_creation() {
        let m = Arc::new(EuclideanSpace::new(3).manifold().clone());
        let module = VectorFieldModule::new(m.clone());

        assert_eq!(module.manifold().dimension(), 3);
        assert_eq!(module.contravariant_rank(), 1);
        assert_eq!(module.covariant_rank(), 0);
    }

    #[test]
    fn test_vector_field_module_zero() {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let module = VectorFieldModule::new(m.clone());

        let zero = module.zero().unwrap();
        assert!(module.contains(&zero));
    }

    #[test]
    fn test_vector_field_free_module() {
        let m = Arc::new(EuclideanSpace::new(3).manifold().clone());
        let module = VectorFieldFreeModule::new(m.clone()).unwrap();

        assert_eq!(module.rank_value(), 3);
        assert_eq!(module.dimension(), Some(3));
        assert_eq!(module.frame().len(), 3);
    }

    #[test]
    fn test_vector_field_free_module_basis() {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let module = VectorFieldFreeModule::new(m.clone()).unwrap();

        let e0 = module.basis_element(&0).unwrap();
        let e1 = module.basis_element(&1).unwrap();

        assert!(module.contains(&e0));
        assert!(module.contains(&e1));
    }

    #[test]
    fn test_tensor_field_module_creation() {
        let m = Arc::new(EuclideanSpace::new(3).manifold().clone());

        let module_1_0 = TensorFieldModule::new(m.clone(), 1, 0);
        assert_eq!(module_1_0.contravariant_rank(), 1);
        assert_eq!(module_1_0.covariant_rank(), 0);
        assert_eq!(module_1_0.total_rank(), 1);

        let module_2_1 = TensorFieldModule::new(m.clone(), 2, 1);
        assert_eq!(module_2_1.contravariant_rank(), 2);
        assert_eq!(module_2_1.covariant_rank(), 1);
        assert_eq!(module_2_1.total_rank(), 3);
    }

    #[test]
    fn test_diff_form_module_creation() {
        let m = Arc::new(EuclideanSpace::new(3).manifold().clone());

        let omega_0 = DiffFormModule::new(m.clone(), 0);
        assert_eq!(omega_0.degree(), 0);

        let omega_1 = DiffFormModule::new(m.clone(), 1);
        assert_eq!(omega_1.degree(), 1);

        let omega_2 = DiffFormModule::new(m.clone(), 2);
        assert_eq!(omega_2.degree(), 2);

        let omega_3 = DiffFormModule::new(m.clone(), 3);
        assert_eq!(omega_3.degree(), 3);
    }

    #[test]
    fn test_diff_form_module_zero() {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let module = DiffFormModule::new(m.clone(), 1);

        let zero = module.zero().unwrap();
        assert!(module.contains(&zero));
    }

    // MAGMA Ch. 17 §17.4.3 IsFinite. A module of sections of a bundle of rank >= 1 over a
    // nonempty manifold contains c·s for c ∈ R, a copy of R (GAP: IsFinite(Rationals) ->
    // false, and Q ⊂ R); a rank-0 bundle gives the zero module, where no claim is made.
    #[test]
    fn test_section_modules_are_infinite() {
        let m = Arc::new(EuclideanSpace::new(3).manifold().clone());
        assert_eq!(DiffScalarFieldAlgebra::new(m.clone()).is_finite(), Some(false));
        assert_eq!(VectorFieldModule::new(m.clone()).is_finite(), Some(false));
        assert_eq!(VectorFieldFreeModule::new(m.clone()).unwrap().is_finite(), Some(false));
        for (p, q) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)] {
            assert_eq!(TensorFieldModule::new(m.clone(), p, q).is_finite(), Some(false));
        }
        for p in 0..=3 {
            assert_eq!(DiffFormModule::new(m.clone(), p).is_finite(), Some(false), "Ω^{p}");
        }
        // Ω^4 of a 3-manifold is the zero module (Λ^4 of a rank-3 bundle is 0).
        assert_eq!(DiffFormModule::new(m.clone(), 4).is_finite(), None);
        // `cardinality` still makes no claim.
        assert_eq!(VectorFieldModule::new(m.clone()).cardinality(), None);
    }

    #[test]
    fn test_section_modules_over_zero_dimensional_manifold() {
        // dim 0: TM = 0, so 𝔛(M) = 0 and T^(p,q)(M) = 0 for p + q >= 1; C^∞(M) = Ω^0(M)
        // still contains the constants.
        let m = Arc::new(DifferentiableManifold::new("P", 0));
        assert_eq!(DiffScalarFieldAlgebra::new(m.clone()).is_finite(), Some(false));
        assert_eq!(VectorFieldModule::new(m.clone()).is_finite(), None);
        assert_eq!(TensorFieldModule::new(m.clone(), 0, 0).is_finite(), Some(false));
        assert_eq!(TensorFieldModule::new(m.clone(), 1, 0).is_finite(), None);
        assert_eq!(TensorFieldModule::new(m.clone(), 0, 2).is_finite(), None);
        assert_eq!(DiffFormModule::new(m.clone(), 0).is_finite(), Some(false));
        assert_eq!(DiffFormModule::new(m.clone(), 1).is_finite(), None);
    }

    #[test]
    fn test_section_modules_over_c0_manifold() {
        // A C^0 manifold: the constants lie in C^0(M) (the trivial bundle); no claim for the
        // modules built from TM.
        let m = Arc::new(DifferentiableManifold::with_class("M", 2, 0));
        assert_eq!(DiffScalarFieldAlgebra::new(m.clone()).is_finite(), Some(false));
        assert_eq!(VectorFieldModule::new(m.clone()).is_finite(), None);
        assert_eq!(DiffFormModule::new(m.clone(), 1).is_finite(), None);
        // T^(0,0)(M) and Ω^0(M) are C^0(M) on a C^0 manifold: the constants (GAP
        // IsFinite(Rationals) -> false, Q ⊂ R), as for the scalar algebra above. At 029f7ab
        // both gave `None` while C^0(M) itself gave `Some(false)`.
        assert_eq!(
            TensorFieldModule::new(m.clone(), 0, 0).is_finite(),
            Some(false)
        );
        assert_eq!(DiffFormModule::new(m.clone(), 0).is_finite(), Some(false));
        // T^(1,0) and T^(0,2) still make no claim on a C^0 manifold (no TM).
        assert_eq!(TensorFieldModule::new(m.clone(), 1, 0).is_finite(), None);
        assert_eq!(TensorFieldModule::new(m.clone(), 0, 2).is_finite(), None);
        // C^1 is enough for TM to exist (a C^0 bundle).
        let m1 = Arc::new(DifferentiableManifold::with_class("M", 2, 1));
        assert_eq!(VectorFieldModule::new(m1.clone()).is_finite(), Some(false));
        assert_eq!(
            TensorFieldModule::new(m1.clone(), 1, 1).is_finite(),
            Some(false)
        );
        assert_eq!(DiffFormModule::new(m1.clone(), 2).is_finite(), Some(false));
        // Ω^3 of a 2-manifold is the zero module for every class.
        assert_eq!(DiffFormModule::new(m1.clone(), 3).is_finite(), None);
    }
}
