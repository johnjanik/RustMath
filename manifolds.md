# Manifolds Implementation Blueprint

## Executive Summary

This document provides a comprehensive implementation strategy for extending RustMath's manifolds system to support differential geometry capabilities comparable to SageMath's manifolds package. The implementation is organized into three phases: infrastructure extensions, core differential structures, and comprehensive testing.

**Estimated Complexity:** ~8-12 weeks of development
**Dependencies:** rustmath-symbolic, rustmath-core, rustmath-matrix
**Target Coverage:** ~80% of SageMath manifolds functionality

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Infrastructure Extensions](#phase-1-infrastructure-extensions)
3. [Phase 2: Core Differential Structures](#phase-2-core-differential-structures)
4. [Phase 3: Testing Strategy](#phase-3-testing-strategy)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Appendices](#appendices)

---

## Architecture Overview

### Design Philosophy: From Inheritance to Traits

SageMath uses Python's multiple inheritance to compose capabilities. Rust uses **trait composition** instead:

```rust
// SageMath Python pattern:
class VectorFieldParal(FiniteRankFreeModuleElement, MultivectorFieldParal, VectorField):
    pass

// Rust equivalent:
struct VectorFieldParal<M: DifferentiableManifold> {
    // ... fields
}

impl<M: DifferentiableManifold> FiniteRankFreeModuleElement for VectorFieldParal<M> { }
impl<M: DifferentiableManifold> MultivectorFieldTrait for VectorFieldParal<M> { }
impl<M: DifferentiableManifold> VectorFieldTrait for VectorFieldParal<M> { }
```

### Core Design Patterns

#### 1. Parent-Element Pattern (already implemented in rustmath-core)

```rust
pub trait Parent {
    type Element: Clone + PartialEq;
    fn contains(&self, element: &Self::Element) -> bool;
    fn zero(&self) -> Option<Self::Element>;
    fn one(&self) -> Option<Self::Element>;
}
```

**Example usage:**
- `DiffScalarFieldAlgebra` is a `Parent` with `Element = DiffScalarField`
- `VectorFieldModule` is a `Parent` with `Element = VectorField`

#### 2. UniqueRepresentation Pattern (already in rustmath-core)

Prevents duplicate creation of mathematically identical structures:

```rust
pub trait UniqueRepresentation: Sized {
    type Key: Hash + Eq + Clone;
    fn key(&self) -> Self::Key;
}

pub struct UniqueCache<T: UniqueRepresentation> {
    cache: HashMap<T::Key, Arc<T>>,
}
```

**Application:** Manifolds, algebras, and modules should use this to ensure `M1 == M2` when they represent the same mathematical object.

#### 3. Generic Over Coefficient Rings

```rust
// Modules are generic over their base ring
pub struct VectorFieldModule<R: Ring> {
    manifold: Arc<dyn DifferentiableManifold>,
    base_ring: R,
}

// Tensor fields carry type parameters for rank
pub struct TensorField<R: Ring, const P: usize, const Q: usize> {
    // P = contravariant rank, Q = covariant rank
}
```

---

## Phase 1: Infrastructure Extensions

### 1.1 Expression Tree Walker System

**Status:** New infrastructure needed
**Location:** `rustmath-symbolic/src/walker.rs`
**Priority:** CRITICAL (foundation for differential geometry)

#### Visitor Pattern for Expression Trees

```rust
/// Trait for walking expression trees
pub trait ExprVisitor {
    type Output;

    /// Visit an integer constant
    fn visit_integer(&mut self, value: &Integer) -> Self::Output;

    /// Visit a rational constant
    fn visit_rational(&mut self, value: &Rational) -> Self::Output;

    /// Visit a symbol
    fn visit_symbol(&mut self, symbol: &Symbol) -> Self::Output;

    /// Visit a binary operation
    fn visit_binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr) -> Self::Output;

    /// Visit a unary operation
    fn visit_unary(&mut self, op: UnaryOp, inner: &Expr) -> Self::Output;

    /// Visit a function call
    fn visit_function(&mut self, name: &str, args: &[Arc<Expr>]) -> Self::Output;
}

/// Mutable walker for transforming expressions
pub trait ExprMutator {
    /// Transform an expression (default: traverse children)
    fn mutate(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Symbol(_) => expr.clone(),
            Expr::Binary(op, left, right) => {
                let new_left = self.mutate(left);
                let new_right = self.mutate(right);
                Expr::Binary(*op, Arc::new(new_left), Arc::new(new_right))
            }
            Expr::Unary(op, inner) => {
                let new_inner = self.mutate(inner);
                Expr::Unary(*op, Arc::new(new_inner))
            }
            Expr::Function(name, args) => {
                let new_args = args.iter().map(|a| Arc::new(self.mutate(a))).collect();
                Expr::Function(name.clone(), new_args)
            }
        }
    }
}
```

#### Concrete Walker Implementations

```rust
/// Collect all free symbols in an expression
pub struct SymbolCollector {
    symbols: HashSet<Symbol>,
}

impl ExprVisitor for SymbolCollector {
    type Output = ();

    fn visit_symbol(&mut self, symbol: &Symbol) -> Self::Output {
        self.symbols.insert(symbol.clone());
    }
    // ... other visits do nothing or recurse
}

/// Substitute symbols with expressions
pub struct Substituter {
    replacements: HashMap<Symbol, Expr>,
}

impl ExprMutator for Substituter {
    fn mutate(&mut self, expr: &Expr) -> Expr {
        if let Expr::Symbol(s) = expr {
            if let Some(replacement) = self.replacements.get(s) {
                return replacement.clone();
            }
        }
        // Default behavior: recurse
        self.mutate_default(expr)
    }
}

/// Partial differentiation with respect to coordinate functions
pub struct CoordinateDerivativeWalker {
    coordinate_symbol: Symbol,
    chart: Arc<Chart>,
}

impl ExprMutator for CoordinateDerivativeWalker {
    // Implements chain rule for coordinate transformations
}
```

**Key Use Cases:**
1. Computing pullbacks of scalar fields under chart transitions
2. Computing Lie derivatives and exterior derivatives
3. Simplifying coordinate expressions
4. Converting between chart representations

---

### 1.2 Chart Transition Function System

**Status:** Partially implemented, needs extension
**Location:** `rustmath-manifolds/src/chart.rs`

#### Current Implementation Gap

The existing `Chart` struct has basic coordinate functions but lacks:
- Transition function computation
- Jacobian calculation
- Compatibility verification

#### Extended Chart Implementation

```rust
pub struct Chart {
    name: String,
    dimension: usize,
    coordinate_names: Vec<String>,
    domain: ManifoldSubset,
    coordinate_functions: Vec<CoordinateFunction>,
    // NEW: Store expression-based coordinate functions for differentiation
    coordinate_expressions: Vec<Option<Expr>>,
    // NEW: Cache for transition functions to other charts
    transitions: Arc<RwLock<HashMap<ChartId, TransitionFunction>>>,
}

pub struct TransitionFunction {
    /// Map from coordinates in source chart to target chart
    coordinate_maps: Vec<Expr>,
    /// Jacobian matrix (∂y^i/∂x^j)
    jacobian: Option<Matrix<Expr>>,
    /// Domain where transition is valid
    domain: ManifoldSubset,
}

impl Chart {
    /// Compute transition function to another chart
    pub fn transition_to(&self, target: &Chart) -> Result<TransitionFunction> {
        // 1. Check if charts are compatible (overlap)
        // 2. Compose inverse of self with target
        // 3. Compute Jacobian using symbolic differentiation
    }

    /// Evaluate the Jacobian at a point
    pub fn jacobian_at(&self, point: &ManifoldPoint, target: &Chart) -> Result<Matrix<f64>> {
        // Uses symbolic Jacobian + numerical evaluation
    }

    /// Pull back a scalar field expression through this chart
    pub fn pullback_scalar(&self, expr: &Expr, target_chart: &Chart) -> Result<Expr> {
        // Transform expression from one coordinate system to another
        let transition = self.transition_to(target_chart)?;
        let mut substituter = Substituter::new();
        for (i, coord_expr) in transition.coordinate_maps.iter().enumerate() {
            let target_symbol = target_chart.coordinate_symbol(i);
            substituter.add_replacement(target_symbol, coord_expr.clone());
        }
        Ok(substituter.mutate(expr))
    }
}
```

---

### 1.3 Symbolic Registry System

**Status:** New infrastructure needed
**Location:** `rustmath-symbolic/src/registry.rs`

Charts and coordinate functions create new symbolic expressions dynamically. We need a registry to track coordinate symbols and their relationships.

```rust
/// Global registry for coordinate symbols and their charts
pub struct CoordinateRegistry {
    /// Map from symbol to the chart it belongs to
    symbol_to_chart: HashMap<Symbol, Weak<Chart>>,
    /// Map from chart to its coordinate symbols
    chart_to_symbols: HashMap<ChartId, Vec<Symbol>>,
}

impl CoordinateRegistry {
    pub fn register_chart(&mut self, chart: Arc<Chart>) {
        // Associate coordinate symbols with chart
    }

    pub fn get_chart_for_symbol(&self, symbol: &Symbol) -> Option<Arc<Chart>> {
        // Reverse lookup
    }
}
```

**Thread Safety:** Use `Arc<RwLock<CoordinateRegistry>>` as a global static.

---

## Phase 2: Core Differential Structures

### 2.1 Trait Hierarchy for Manifolds

Map SageMath's class hierarchy to Rust traits:

```rust
// ============================================================================
// DOMAIN TRAITS (manifolds, subsets, points)
// ============================================================================

/// Base trait for any subset of a manifold
pub trait ManifoldSubsetTrait: Parent<Element = ManifoldPoint> + UniqueRepresentation {
    fn dimension(&self) -> usize;
    fn ambient_manifold(&self) -> Option<Arc<dyn TopologicalManifoldTrait>>;
    fn is_open(&self) -> bool;
    fn is_closed(&self) -> bool;
}

/// Topological manifold
pub trait TopologicalManifoldTrait: ManifoldSubsetTrait {
    fn atlas(&self) -> &[Chart];
    fn default_chart(&self) -> Option<&Chart>;
    fn add_chart(&mut self, chart: Chart) -> Result<()>;
}

/// Differentiable (smooth) manifold
pub trait DifferentiableManifoldTrait: TopologicalManifoldTrait {
    /// Check that atlas is C^∞-compatible
    fn verify_smoothness(&self) -> Result<()>;

    /// Get the scalar field algebra C^∞(M)
    fn scalar_field_algebra(&self) -> Arc<DiffScalarFieldAlgebra>;

    /// Get the vector field module 𝔛(M)
    fn vector_field_module(&self) -> Arc<VectorFieldModule>;

    /// Get the tangent bundle TM
    fn tangent_bundle(&self) -> Arc<TangentBundle>;
}

/// Parallelizable manifold (has global frame)
pub trait ParallelizableManifoldTrait: DifferentiableManifoldTrait {
    /// Get the global frame (basis of vector fields)
    fn global_frame(&self) -> &[VectorFieldParal];

    /// Vector fields are free module elements
    fn vector_field_free_module(&self) -> Arc<VectorFieldFreeModule>;
}

// ============================================================================
// SCALAR FIELD TRAITS
// ============================================================================

/// Algebra of scalar fields C^∞(M)
pub trait ScalarFieldAlgebraTrait: Parent<Element = ScalarField> + Ring {
    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait>;
    fn base_field(&self) -> BaseField; // Usually ℝ or ℂ
}

/// Differentiable scalar field algebra (specialization)
pub trait DiffScalarFieldAlgebraTrait: ScalarFieldAlgebraTrait {
    // All fields are C^∞
}

/// A scalar field f: M → ℝ
pub trait ScalarFieldTrait: Clone {
    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait>;

    /// Express field in coordinates of a chart
    fn expr(&self, chart: &Chart) -> Result<Expr>;

    /// Evaluate at a point
    fn eval_at(&self, point: &ManifoldPoint) -> Result<f64>;

    /// Differential (covector field)
    fn differential(&self) -> DiffForm;
}

// ============================================================================
// MODULE TRAITS (tensor fields, vector fields)
// ============================================================================

/// Module of vector fields over C^∞(M)
pub trait VectorFieldModuleTrait: Parent<Element = VectorField> {
    type ScalarRing: DiffScalarFieldAlgebraTrait;

    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait>;
    fn scalar_ring(&self) -> &Self::ScalarRing;
    fn rank(&self) -> (usize, usize); // (contravariant, covariant)
}

/// Free module of vector fields (parallelizable case)
pub trait VectorFieldFreeModuleTrait: VectorFieldModuleTrait + ParentWithBasis {
    fn frame(&self) -> &[VectorFieldParal];
}

/// Tensor field module T^(p,q)(M)
pub trait TensorFieldModuleTrait: Parent<Element = TensorField> {
    fn contravariant_rank(&self) -> usize; // p
    fn covariant_rank(&self) -> usize;     // q
    fn total_rank(&self) -> usize {
        self.contravariant_rank() + self.covariant_rank()
    }
}

/// Tangent space at a point T_p(M)
pub trait TangentSpaceTrait: VectorFieldFreeModuleTrait {
    fn base_point(&self) -> &ManifoldPoint;
}

// ============================================================================
// ELEMENT TRAITS (vector fields, tensor fields)
// ============================================================================

/// A vector field X ∈ 𝔛(M)
pub trait VectorFieldTrait: Clone {
    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait>;

    /// Components in a chart
    fn components(&self, chart: &Chart) -> Result<Vec<Expr>>;

    /// Act on a scalar field: X(f)
    fn apply_to_scalar(&self, field: &ScalarField) -> Result<ScalarField>;

    /// Lie bracket [X, Y]
    fn lie_bracket(&self, other: &Self) -> Self;

    /// Evaluate at a point to get tangent vector
    fn at_point(&self, point: &ManifoldPoint) -> Result<TangentVector>;
}

/// Vector field on parallelizable manifold (free module element)
pub trait VectorFieldParalTrait: VectorFieldTrait {
    /// Components with respect to the global frame
    fn frame_components(&self) -> Vec<ScalarField>;
}

/// Tensor field of type (p, q)
pub trait TensorFieldTrait: Clone {
    fn contravariant_rank(&self) -> usize;
    fn covariant_rank(&self) -> usize;

    /// Components in a chart
    fn components(&self, chart: &Chart) -> Result<Vec<Expr>>;

    /// Tensor contraction
    fn contract(&self, i: usize, j: usize) -> Result<TensorField>;
}

/// Tangent vector at a point
pub trait TangentVectorTrait: Clone {
    fn base_point(&self) -> &ManifoldPoint;
    fn components(&self, chart: &Chart) -> Result<Vec<f64>>;
}

/// Differential form (totally antisymmetric covariant tensor)
pub trait DiffFormTrait: TensorFieldTrait {
    fn degree(&self) -> usize;

    /// Exterior derivative
    fn exterior_derivative(&self) -> DiffForm;

    /// Wedge product
    fn wedge(&self, other: &DiffForm) -> DiffForm;

    /// Interior product with vector field
    fn interior_product(&self, vector: &VectorField) -> DiffForm;

    /// Lie derivative
    fn lie_derivative(&self, vector: &VectorField) -> DiffForm;
}
```

---

### 2.2 Concrete Type Implementations

#### 2.2.1 Scalar Fields

```rust
/// A smooth scalar field f: M → ℝ
pub struct ScalarField {
    manifold: Arc<dyn DifferentiableManifoldTrait>,
    /// Expression in each chart
    chart_expressions: HashMap<ChartId, Expr>,
    /// Optional name
    name: Option<String>,
}

impl ScalarField {
    pub fn new(manifold: Arc<dyn DifferentiableManifoldTrait>) -> Self {
        Self {
            manifold,
            chart_expressions: HashMap::new(),
            name: None,
        }
    }

    pub fn from_expr(
        manifold: Arc<dyn DifferentiableManifoldTrait>,
        chart: &Chart,
        expr: Expr,
    ) -> Self {
        let mut field = Self::new(manifold);
        field.set_expr(chart, expr);
        field
    }

    pub fn set_expr(&mut self, chart: &Chart, expr: Expr) {
        self.chart_expressions.insert(chart.id(), expr);
    }

    /// Get expression in a specific chart
    pub fn expr(&self, chart: &Chart) -> Result<Expr> {
        if let Some(expr) = self.chart_expressions.get(&chart.id()) {
            return Ok(expr.clone());
        }

        // Try to compute via chart transition from a known chart
        for (other_chart_id, other_expr) in &self.chart_expressions {
            if let Some(other_chart) = self.manifold.get_chart(other_chart_id) {
                let transition = other_chart.transition_to(chart)?;
                let transformed = transition.pullback(other_expr)?;
                return Ok(transformed);
            }
        }

        Err(ManifoldError::NoExpressionInChart)
    }
}

impl Add for ScalarField {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        // Add expressions in each chart
    }
}

impl Mul for ScalarField {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        // Multiply expressions in each chart
    }
}
```

#### 2.2.2 Algebra of Scalar Fields

```rust
/// The algebra C^∞(M) of smooth functions on M
pub struct DiffScalarFieldAlgebra {
    manifold: Arc<dyn DifferentiableManifoldTrait>,
    cache: UniqueCache<ScalarField>,
}

impl Parent for DiffScalarFieldAlgebra {
    type Element = ScalarField;

    fn contains(&self, element: &Self::Element) -> bool {
        Arc::ptr_eq(&element.manifold, &self.manifold)
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(ScalarField::constant(self.manifold.clone(), 0.0))
    }

    fn one(&self) -> Option<Self::Element> {
        Some(ScalarField::constant(self.manifold.clone(), 1.0))
    }
}

impl Ring for DiffScalarFieldAlgebra {
    // Implement ring operations that delegate to ScalarField operations
}
```

#### 2.2.3 Vector Fields

```rust
/// A vector field X ∈ 𝔛(M)
pub struct VectorField {
    manifold: Arc<dyn DifferentiableManifoldTrait>,
    /// Components in each chart (contravariant components)
    chart_components: HashMap<ChartId, Vec<Expr>>,
    name: Option<String>,
}

impl VectorField {
    /// Create vector field from components in a chart
    pub fn from_components(
        manifold: Arc<dyn DifferentiableManifoldTrait>,
        chart: &Chart,
        components: Vec<Expr>,
    ) -> Result<Self> {
        if components.len() != manifold.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: manifold.dimension(),
                actual: components.len(),
            });
        }

        let mut field = Self {
            manifold,
            chart_components: HashMap::new(),
            name: None,
        };
        field.chart_components.insert(chart.id(), components);
        Ok(field)
    }

    /// Get components in a specific chart
    pub fn components(&self, chart: &Chart) -> Result<Vec<Expr>> {
        if let Some(comps) = self.chart_components.get(&chart.id()) {
            return Ok(comps.clone());
        }

        // Transform from another chart using Jacobian
        for (other_chart_id, other_comps) in &self.chart_components {
            if let Some(other_chart) = self.manifold.get_chart(other_chart_id) {
                let transition = other_chart.transition_to(chart)?;
                let jacobian = transition.jacobian()?;
                let transformed = jacobian.apply_to_vector(other_comps)?;
                return Ok(transformed);
            }
        }

        Err(ManifoldError::NoComponentsInChart)
    }

    /// Apply vector field to scalar field: X(f)
    pub fn apply_to_scalar(&self, field: &ScalarField) -> Result<ScalarField> {
        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        let f_expr = field.expr(chart)?;
        let x_comps = self.components(chart)?;

        // X(f) = Σ X^i ∂f/∂x^i
        let mut result_expr = Expr::from(0);
        for i in 0..self.manifold.dimension() {
            let coord_symbol = chart.coordinate_symbol(i);
            let df_dxi = f_expr.differentiate(&coord_symbol);
            result_expr = result_expr + x_comps[i].clone() * df_dxi;
        }

        Ok(ScalarField::from_expr(self.manifold.clone(), chart, result_expr))
    }

    /// Lie bracket [X, Y]
    pub fn lie_bracket(&self, other: &VectorField) -> Result<VectorField> {
        // [X, Y](f) = X(Y(f)) - Y(X(f))
        // In coordinates: [X, Y]^k = X^i ∂Y^k/∂x^i - Y^i ∂X^k/∂x^i

        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        let x_comps = self.components(chart)?;
        let y_comps = other.components(chart)?;
        let n = self.manifold.dimension();

        let mut bracket_comps = Vec::with_capacity(n);
        for k in 0..n {
            let mut term = Expr::from(0);
            for i in 0..n {
                let coord = chart.coordinate_symbol(i);

                // X^i ∂Y^k/∂x^i
                let dy_k = y_comps[k].differentiate(&coord);
                let term1 = x_comps[i].clone() * dy_k;

                // Y^i ∂X^k/∂x^i
                let dx_k = x_comps[k].differentiate(&coord);
                let term2 = y_comps[i].clone() * dx_k;

                term = term + term1 - term2;
            }
            bracket_comps.push(term);
        }

        VectorField::from_components(self.manifold.clone(), chart, bracket_comps)
    }
}
```

#### 2.2.4 Module of Vector Fields

```rust
/// The module 𝔛(M) of vector fields over C^∞(M)
pub struct VectorFieldModule {
    manifold: Arc<dyn DifferentiableManifoldTrait>,
    scalar_ring: Arc<DiffScalarFieldAlgebra>,
}

impl Parent for VectorFieldModule {
    type Element = VectorField;

    fn contains(&self, element: &Self::Element) -> bool {
        Arc::ptr_eq(&element.manifold, &self.manifold)
    }

    fn zero(&self) -> Option<Self::Element> {
        // Zero vector field (all components = 0)
        let chart = self.manifold.default_chart()?;
        let zero_comps = vec![Expr::from(0); self.manifold.dimension()];
        VectorField::from_components(self.manifold.clone(), chart, zero_comps).ok()
    }
}

/// Module multiplication: scalar field × vector field → vector field
impl Mul<VectorField> for ScalarField {
    type Output = VectorField;

    fn mul(self, rhs: VectorField) -> Self::Output {
        // Multiply each component by the scalar field
        let chart = rhs.manifold.default_chart().unwrap();
        let f_expr = self.expr(chart).unwrap();
        let x_comps = rhs.components(chart).unwrap();

        let new_comps: Vec<Expr> = x_comps.iter()
            .map(|comp| f_expr.clone() * comp.clone())
            .collect();

        VectorField::from_components(rhs.manifold.clone(), chart, new_comps).unwrap()
    }
}
```

#### 2.2.5 Tensor Fields

```rust
/// Tensor field of type (p, q)
pub struct TensorField {
    manifold: Arc<dyn DifferentiableManifoldTrait>,
    contravariant_rank: usize, // p
    covariant_rank: usize,     // q
    /// Components T^{i₁...iₚ}_{j₁...jᵩ}
    /// Stored as flattened multi-index array
    chart_components: HashMap<ChartId, Vec<Expr>>,
}

impl TensorField {
    pub fn total_rank(&self) -> usize {
        self.contravariant_rank + self.covariant_rank
    }

    pub fn num_components(&self) -> usize {
        self.manifold.dimension().pow(self.total_rank() as u32)
    }

    /// Contract indices: i-th contravariant with j-th covariant
    pub fn contract(&self, i: usize, j: usize) -> Result<TensorField> {
        if i >= self.contravariant_rank || j >= self.covariant_rank {
            return Err(ManifoldError::InvalidIndex);
        }

        // Perform Einstein summation over the specified indices
        // Result is a (p-1, q-1) tensor

        let chart = self.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;
        let comps = self.chart_components.get(&chart.id())
            .ok_or(ManifoldError::NoComponentsInChart)?;

        let n = self.manifold.dimension();
        let new_p = self.contravariant_rank - 1;
        let new_q = self.covariant_rank - 1;

        // Build contracted components (Einstein summation)
        let mut contracted_comps = vec![Expr::from(0); n.pow((new_p + new_q) as u32)];

        // ... (complex multi-index computation)

        Ok(TensorField {
            manifold: self.manifold.clone(),
            contravariant_rank: new_p,
            covariant_rank: new_q,
            chart_components: [(chart.id(), contracted_comps)].into_iter().collect(),
        })
    }
}
```

#### 2.2.6 Differential Forms

```rust
/// A differential p-form
pub struct DiffForm {
    /// Base tensor field (totally antisymmetric covariant tensor)
    tensor: TensorField,
    degree: usize, // p
}

impl DiffForm {
    /// Exterior derivative d: Ωᵖ(M) → Ωᵖ⁺¹(M)
    pub fn exterior_derivative(&self) -> Result<DiffForm> {
        let chart = self.tensor.manifold.default_chart()
            .ok_or(ManifoldError::NoChart)?;

        // For a p-form ω, dω has components:
        // (dω)_{i₀...iₚ} = Σⱼ ∂ω_{ji₁...iₚ}/∂xʲ (alternating sum)

        let n = self.tensor.manifold.dimension();
        let p = self.degree;

        // ... (antisymmetrization + exterior differentiation)

        Ok(DiffForm {
            tensor: new_tensor,
            degree: p + 1,
        })
    }

    /// Wedge product: ωᵖ ∧ ηᵍ → ω∧η ∈ Ωᵖ⁺ᵍ(M)
    pub fn wedge(&self, other: &DiffForm) -> Result<DiffForm> {
        // Antisymmetrize the tensor product
        let p = self.degree;
        let q = other.degree;

        // ... (complex antisymmetrization)

        Ok(DiffForm {
            tensor: wedge_tensor,
            degree: p + q,
        })
    }

    /// Interior product (contraction) with vector field: iₓω
    pub fn interior_product(&self, vector: &VectorField) -> Result<DiffForm> {
        // Contract first index with vector field
        // Reduces degree by 1

        Ok(DiffForm {
            tensor: contracted,
            degree: self.degree - 1,
        })
    }

    /// Lie derivative along vector field: ℒₓω
    pub fn lie_derivative(&self, vector: &VectorField) -> Result<DiffForm> {
        // Cartan's formula: ℒₓω = iₓ(dω) + d(iₓω)
        let d_omega = self.exterior_derivative()?;
        let ix_d_omega = d_omega.interior_product(vector)?;

        let ix_omega = self.interior_product(vector)?;
        let d_ix_omega = ix_omega.exterior_derivative()?;

        ix_d_omega.add(&d_ix_omega)
    }
}
```

---

### 2.3 Bundle Structures

#### 2.3.1 Tangent Bundle

```rust
/// The tangent bundle TM
pub struct TangentBundle {
    base_manifold: Arc<dyn DifferentiableManifoldTrait>,
    total_space_dimension: usize, // 2 * base dimension
    /// Projection π: TM → M
    projection: Box<dyn Fn(&BundlePoint) -> ManifoldPoint>,
}

impl TangentBundle {
    pub fn new(base: Arc<dyn DifferentiableManifoldTrait>) -> Self {
        let dim = base.dimension();
        Self {
            base_manifold: base,
            total_space_dimension: 2 * dim,
            projection: Box::new(|bp| bp.base_point.clone()),
        }
    }

    /// Get the tangent space at a point
    pub fn fiber_at(&self, point: &ManifoldPoint) -> TangentSpace {
        TangentSpace {
            base_point: point.clone(),
            manifold: self.base_manifold.clone(),
        }
    }

    /// Get a section (vector field)
    pub fn section(&self) -> VectorField {
        VectorField::new(self.base_manifold.clone())
    }
}

/// Point in the tangent bundle: (p, v) where p ∈ M, v ∈ TₚM
pub struct BundlePoint {
    base_point: ManifoldPoint,
    tangent_vector: TangentVector,
}

/// The tangent space TₚM at a point p
pub struct TangentSpace {
    base_point: ManifoldPoint,
    manifold: Arc<dyn DifferentiableManifoldTrait>,
}

impl TangentSpace {
    /// Basis of tangent space induced by chart
    pub fn coordinate_basis(&self, chart: &Chart) -> Vec<TangentVector> {
        // Basis vectors ∂/∂x^i
        (0..self.manifold.dimension())
            .map(|i| self.coordinate_vector(chart, i))
            .collect()
    }

    fn coordinate_vector(&self, chart: &Chart, index: usize) -> TangentVector {
        // The vector ∂/∂x^i at this point
        let mut components = vec![0.0; self.manifold.dimension()];
        components[index] = 1.0;

        TangentVector {
            base_point: self.base_point.clone(),
            chart_components: [(chart.id(), components)].into_iter().collect(),
        }
    }
}

impl ParentWithBasis for TangentSpace {
    type BasisIndex = usize;

    fn dimension(&self) -> Option<usize> {
        Some(self.manifold.dimension())
    }

    fn basis_element(&self, index: &Self::BasisIndex) -> Option<TangentVector> {
        let chart = self.manifold.default_chart()?;
        Some(self.coordinate_vector(&chart, *index))
    }

    fn basis_indices(&self) -> Vec<Self::BasisIndex> {
        (0..self.manifold.dimension()).collect()
    }
}
```

#### 2.3.2 Cotangent Bundle and 1-Forms

```rust
/// The cotangent bundle T*M
pub struct CotangentBundle {
    base_manifold: Arc<dyn DifferentiableManifoldTrait>,
}

impl CotangentBundle {
    pub fn fiber_at(&self, point: &ManifoldPoint) -> CotangentSpace {
        CotangentSpace {
            base_point: point.clone(),
            manifold: self.base_manifold.clone(),
        }
    }

    /// A section is a 1-form
    pub fn section(&self) -> DiffForm {
        DiffForm::new(self.base_manifold.clone(), 1)
    }
}

/// The cotangent space Tₚ*M (dual to TₚM)
pub struct CotangentSpace {
    base_point: ManifoldPoint,
    manifold: Arc<dyn DifferentiableManifoldTrait>,
}

impl CotangentSpace {
    /// Coordinate basis 1-forms dx^i
    pub fn coordinate_basis(&self, chart: &Chart) -> Vec<Covector> {
        (0..self.manifold.dimension())
            .map(|i| self.coordinate_covector(chart, i))
            .collect()
    }

    fn coordinate_covector(&self, chart: &Chart, index: usize) -> Covector {
        // The covector dx^i at this point
        Covector {
            base_point: self.base_point.clone(),
            components: vec![if i == index { 1.0 } else { 0.0 }
                             for i in 0..self.manifold.dimension()],
        }
    }
}

/// A covector (element of cotangent space)
pub struct Covector {
    base_point: ManifoldPoint,
    components: Vec<f64>,
}

impl Covector {
    /// Apply covector to tangent vector (dual pairing)
    pub fn apply(&self, vector: &TangentVector) -> Result<f64> {
        if !self.base_point.eq(&vector.base_point) {
            return Err(ManifoldError::DifferentBasePoints);
        }

        let v_comps = vector.components_in_default_chart()?;
        Ok(self.components.iter()
            .zip(v_comps.iter())
            .map(|(a, b)| a * b)
            .sum())
    }
}
```

---

### 2.4 Example Manifolds

Provide concrete implementations for testing and reference:

```rust
/// The real line ℝ
pub struct RealLine {
    manifold: DifferentiableManifold,
}

impl RealLine {
    pub fn new() -> Self {
        let mut manifold = DifferentiableManifold::new("ℝ", 1);

        // Standard chart (identity map)
        let chart = Chart::new("standard", 1)
            .with_coordinate_names(vec!["x".to_string()])
            .with_coordinate_function(0, Box::new(|p| p.coordinates()[0]));

        manifold.add_chart(chart).unwrap();

        Self { manifold }
    }
}

impl Deref for RealLine {
    type Target = DifferentiableManifold;
    fn deref(&self) -> &Self::Target {
        &self.manifold
    }
}

/// The circle S¹ = ℝ/2πℤ
pub struct Circle {
    manifold: DifferentiableManifold,
}

impl Circle {
    pub fn new() -> Self {
        let mut manifold = DifferentiableManifold::new("S¹", 1);

        // Chart 1: U₁ = S¹ \ {(0,1)}, φ₁(θ) = θ ∈ (-π, π)
        let chart1 = Chart::new("U1", 1)
            .with_coordinate_names(vec!["θ".to_string()])
            .with_domain_condition(Box::new(|p| {
                let theta = p.coordinates()[0];
                -std::f64::consts::PI < theta && theta < std::f64::consts::PI
            }));

        // Chart 2: U₂ = S¹ \ {(0,-1)}, φ₂(θ) = θ ∈ (0, 2π)
        let chart2 = Chart::new("U2", 1)
            .with_coordinate_names(vec!["θ".to_string()])
            .with_domain_condition(Box::new(|p| {
                let theta = p.coordinates()[0];
                0.0 < theta && theta < 2.0 * std::f64::consts::PI
            }));

        manifold.add_chart(chart1).unwrap();
        manifold.add_chart(chart2).unwrap();

        Self { manifold }
    }
}

/// The 2-sphere S²
pub struct Sphere2 {
    manifold: DifferentiableManifold,
}

impl Sphere2 {
    pub fn new() -> Self {
        let mut manifold = DifferentiableManifold::new("S²", 2);

        // Stereographic projection from north pole
        let chart_north = Chart::new("north_stereo", 2)
            .with_coordinate_names(vec!["u".to_string(), "v".to_string()])
            .with_coordinate_function(0, Box::new(|p| {
                let (x, y, z) = (p.coordinates()[0], p.coordinates()[1], p.coordinates()[2]);
                x / (1.0 - z)
            }))
            .with_coordinate_function(1, Box::new(|p| {
                let (x, y, z) = (p.coordinates()[0], p.coordinates()[1], p.coordinates()[2]);
                y / (1.0 - z)
            }));

        // Stereographic projection from south pole
        let chart_south = Chart::new("south_stereo", 2)
            .with_coordinate_names(vec!["u".to_string(), "v".to_string()])
            .with_coordinate_function(0, Box::new(|p| {
                let (x, y, z) = (p.coordinates()[0], p.coordinates()[1], p.coordinates()[2]);
                x / (1.0 + z)
            }))
            .with_coordinate_function(1, Box::new(|p| {
                let (x, y, z) = (p.coordinates()[0], p.coordinates()[1], p.coordinates()[2]);
                y / (1.0 + z)
            }));

        manifold.add_chart(chart_north).unwrap();
        manifold.add_chart(chart_south).unwrap();

        Self { manifold }
    }
}

/// The 2-torus T² = S¹ × S¹
pub struct Torus2 {
    manifold: DifferentiableManifold,
}

impl Torus2 {
    pub fn new() -> Self {
        let mut manifold = DifferentiableManifold::new("T²", 2);

        // Standard chart with angular coordinates
        let chart = Chart::new("angular", 2)
            .with_coordinate_names(vec!["φ".to_string(), "ψ".to_string()])
            .with_periodic_coordinates(vec![
                (0, 2.0 * std::f64::consts::PI),
                (1, 2.0 * std::f64::consts::PI),
            ]);

        manifold.add_chart(chart).unwrap();

        Self { manifold }
    }
}
```

---

## Phase 3: Testing Strategy

### 3.1 Unit Tests

Each module should have comprehensive unit tests:

#### Chart Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chart_creation() {
        let chart = Chart::new("test", 2)
            .with_coordinate_names(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(chart.dimension(), 2);
        assert_eq!(chart.coordinate_names(), &["x", "y"]);
    }

    #[test]
    fn test_chart_transition_identity() {
        let chart = Chart::identity("id", 2);
        let transition = chart.transition_to(&chart).unwrap();

        // Should be identity map
        let point = vec![1.0, 2.0];
        let mapped = transition.apply(&point).unwrap();
        assert_eq!(mapped, point);
    }

    #[test]
    fn test_jacobian_polar_to_cartesian() {
        // (r, θ) → (x, y) where x = r cos θ, y = r sin θ
        // Jacobian should be:
        // | cos θ   -r sin θ |
        // | sin θ    r cos θ |

        let polar_chart = Chart::polar_coordinates();
        let cartesian_chart = Chart::cartesian_coordinates(2);

        let transition = polar_chart.transition_to(&cartesian_chart).unwrap();
        let jacobian = transition.jacobian().unwrap();

        // Test at r=1, θ=π/4
        // ... (specific numerical checks)
    }
}
```

#### Scalar Field Tests
```rust
#[test]
fn test_scalar_field_addition() {
    let m = RealLine::new();
    let x = m.coordinate_function(0);

    let f = ScalarField::from_expr(m.clone(), &m.default_chart().unwrap(), x.clone());
    let g = ScalarField::from_expr(m.clone(), &m.default_chart().unwrap(), x.clone() + Expr::from(1));

    let h = f + g; // Should be 2x + 1

    let p = ManifoldPoint::from_coords(m.clone(), vec![2.0]);
    assert_eq!(h.eval_at(&p).unwrap(), 5.0); // 2*2 + 1
}

#[test]
fn test_scalar_field_multiplication() {
    let m = RealLine::new();
    let x = m.coordinate_function(0);

    let f = ScalarField::from_expr(m.clone(), &m.default_chart().unwrap(), x.clone());
    let g = f.clone() * f; // x²

    let p = ManifoldPoint::from_coords(m.clone(), vec![3.0]);
    assert_eq!(g.eval_at(&p).unwrap(), 9.0);
}

#[test]
fn test_scalar_field_differential() {
    let m = RealLine::new();
    let x = m.coordinate_function(0);

    // f(x) = x²
    let f = ScalarField::from_expr(m.clone(), &m.default_chart().unwrap(),
                                    x.clone().pow(Expr::from(2)));

    let df = f.differential(); // Should be 2x dx

    // Evaluate at x=3: df should give 6
    let p = ManifoldPoint::from_coords(m.clone(), vec![3.0]);
    let basis_vector = m.tangent_space_at(&p).coordinate_vector(0);
    assert_eq!(df.apply(&basis_vector).unwrap(), 6.0);
}
```

#### Vector Field Tests
```rust
#[test]
fn test_vector_field_application() {
    let m = RealLine::new();
    let chart = m.default_chart().unwrap();
    let x = m.coordinate_function(0);

    // Vector field X = d/dx
    let x_field = VectorField::from_components(
        m.clone(),
        &chart,
        vec![Expr::from(1)]
    ).unwrap();

    // Scalar field f(x) = x²
    let f = ScalarField::from_expr(m.clone(), &chart, x.clone().pow(Expr::from(2)));

    // X(f) should be 2x
    let xf = x_field.apply_to_scalar(&f).unwrap();

    let p = ManifoldPoint::from_coords(m.clone(), vec![3.0]);
    assert_eq!(xf.eval_at(&p).unwrap(), 6.0);
}

#[test]
fn test_lie_bracket() {
    let m = EuclideanSpace::new(2);
    let chart = m.default_chart().unwrap();

    // X = ∂/∂x
    let x_field = VectorField::from_components(
        m.clone(),
        &chart,
        vec![Expr::from(1), Expr::from(0)]
    ).unwrap();

    // Y = ∂/∂y
    let y_field = VectorField::from_components(
        m.clone(),
        &chart,
        vec![Expr::from(0), Expr::from(1)]
    ).unwrap();

    // [X, Y] = 0 (coordinate vector fields commute)
    let bracket = x_field.lie_bracket(&y_field).unwrap();

    assert!(bracket.is_zero());
}

#[test]
fn test_lie_bracket_nonzero() {
    let m = EuclideanSpace::new(2);
    let chart = m.default_chart().unwrap();
    let (x, y) = (chart.coordinate_symbol(0), chart.coordinate_symbol(1));

    // X = y ∂/∂x
    let x_field = VectorField::from_components(
        m.clone(),
        &chart,
        vec![Expr::Symbol(y.clone()), Expr::from(0)]
    ).unwrap();

    // Y = ∂/∂y
    let y_field = VectorField::from_components(
        m.clone(),
        &chart,
        vec![Expr::from(0), Expr::from(1)]
    ).unwrap();

    // [X, Y] = -∂/∂x
    let bracket = x_field.lie_bracket(&y_field).unwrap();
    let expected_comps = vec![Expr::from(-1), Expr::from(0)];

    assert_eq!(bracket.components(&chart).unwrap(), expected_comps);
}
```

#### Differential Form Tests
```rust
#[test]
fn test_exterior_derivative_0form() {
    let m = EuclideanSpace::new(2);
    let chart = m.default_chart().unwrap();
    let (x, y) = (chart.coordinate_symbol(0), chart.coordinate_symbol(1));

    // f = x² + y²
    let f = ScalarField::from_expr(
        m.clone(),
        &chart,
        Expr::Symbol(x.clone()).pow(Expr::from(2)) +
        Expr::Symbol(y.clone()).pow(Expr::from(2))
    );

    // df = 2x dx + 2y dy
    let df = f.differential();

    assert_eq!(df.degree(), 1);
    let comps = df.components(&chart).unwrap();
    assert_eq!(comps[0], Expr::from(2) * Expr::Symbol(x));
    assert_eq!(comps[1], Expr::from(2) * Expr::Symbol(y));
}

#[test]
fn test_wedge_product() {
    let m = EuclideanSpace::new(2);
    let chart = m.default_chart().unwrap();

    // dx and dy (coordinate 1-forms)
    let dx = DiffForm::coordinate_form(&m, &chart, 0);
    let dy = DiffForm::coordinate_form(&m, &chart, 1);

    // dx ∧ dy
    let dx_dy = dx.wedge(&dy).unwrap();

    assert_eq!(dx_dy.degree(), 2);

    // dy ∧ dx = -dx ∧ dy
    let dy_dx = dy.wedge(&dx).unwrap();
    assert_eq!(dy_dx, -dx_dy.clone());

    // dx ∧ dx = 0
    let dx_dx = dx.wedge(&dx).unwrap();
    assert!(dx_dx.is_zero());
}

#[test]
fn test_exterior_derivative_1form() {
    let m = EuclideanSpace::new(2);
    let chart = m.default_chart().unwrap();
    let (x, y) = (chart.coordinate_symbol(0), chart.coordinate_symbol(1));

    // ω = x dy
    let omega = DiffForm::from_components(
        m.clone(),
        &chart,
        1,
        vec![Expr::from(0), Expr::Symbol(x.clone())]
    );

    // dω = dx ∧ dy
    let d_omega = omega.exterior_derivative().unwrap();

    assert_eq!(d_omega.degree(), 2);
    // Component should be 1 (coefficient of dx ∧ dy)
}

#[test]
fn test_exterior_derivative_closed_form() {
    let m = EuclideanSpace::new(2);
    let chart = m.default_chart().unwrap();
    let (x, y) = (chart.coordinate_symbol(0), chart.coordinate_symbol(1));

    // Exact form: ω = d(xy) = y dx + x dy
    let f = ScalarField::from_expr(
        m.clone(),
        &chart,
        Expr::Symbol(x.clone()) * Expr::Symbol(y.clone())
    );
    let omega = f.differential();

    // dω = 0 (d² = 0)
    let d_omega = omega.exterior_derivative().unwrap();
    assert!(d_omega.is_zero());
}
```

---

### 3.2 Integration Tests

Test interactions between components:

```rust
#[test]
fn test_sphere_vector_field() {
    let sphere = Sphere2::new();

    // Define a vector field on S² using stereographic coordinates
    let north_chart = sphere.get_chart_by_name("north_stereo").unwrap();

    // Rotational vector field around z-axis: X = -v ∂/∂u + u ∂/∂v
    let (u, v) = (north_chart.coordinate_symbol(0), north_chart.coordinate_symbol(1));
    let x_field = VectorField::from_components(
        Arc::new(sphere.clone()),
        &north_chart,
        vec![
            -Expr::Symbol(v.clone()),
            Expr::Symbol(u.clone()),
        ]
    ).unwrap();

    // This field should have closed orbits (circles around z-axis)
    // Test: Lie derivative of itself should give 0 (Killing vector)
    let lie_deriv = x_field.lie_bracket(&x_field).unwrap();
    assert!(lie_deriv.is_zero());
}

#[test]
fn test_torus_volume_form() {
    let torus = Torus2::new();
    let chart = torus.default_chart().unwrap();

    // Volume form on T²: dφ ∧ dψ
    let dphi = DiffForm::coordinate_form(&torus, &chart, 0);
    let dpsi = DiffForm::coordinate_form(&torus, &chart, 1);
    let vol_form = dphi.wedge(&dpsi).unwrap();

    // Integrate over torus: ∫∫ dφ dψ = (2π)(2π) = 4π²
    let volume = torus.integrate_form(&vol_form).unwrap();
    let expected = 4.0 * std::f64::consts::PI * std::f64::consts::PI;
    assert!((volume - expected).abs() < 1e-6);
}

#[test]
fn test_tangent_bundle_projection() {
    let m = RealLine::new();
    let tm = m.tangent_bundle();

    // Create a point in TM: (x=2, v=3)
    let base_point = ManifoldPoint::from_coords(m.clone(), vec![2.0]);
    let tangent_vec = TangentVector::from_components(
        base_point.clone(),
        vec![3.0]
    );
    let bundle_point = BundlePoint::new(base_point.clone(), tangent_vec);

    // Project back to M
    let projected = tm.project(&bundle_point);
    assert_eq!(projected, base_point);
}
```

---

### 3.3 Property-Based Tests

Use `proptest` or `quickcheck` for algebraic properties:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn scalar_field_addition_commutative(a: f64, b: f64) {
        let m = RealLine::new();
        let f = ScalarField::constant(m.clone(), a);
        let g = ScalarField::constant(m.clone(), b);

        let fg = f.clone() + g.clone();
        let gf = g + f;

        prop_assert_eq!(fg, gf);
    }

    #[test]
    fn lie_bracket_antisymmetric(
        x_comp: f64,
        y_comp: f64,
    ) {
        let m = EuclideanSpace::new(2);
        let chart = m.default_chart().unwrap();

        let x = VectorField::from_components(
            m.clone(),
            &chart,
            vec![Expr::from(x_comp), Expr::from(0)]
        ).unwrap();

        let y = VectorField::from_components(
            m.clone(),
            &chart,
            vec![Expr::from(0), Expr::from(y_comp)]
        ).unwrap();

        let xy = x.lie_bracket(&y).unwrap();
        let yx = y.lie_bracket(&x).unwrap();

        prop_assert_eq!(xy, -yx);
    }

    #[test]
    fn exterior_derivative_nilpotent(x_coef: f64, y_coef: f64) {
        let m = EuclideanSpace::new(2);
        let chart = m.default_chart().unwrap();

        let omega = DiffForm::from_components(
            m.clone(),
            &chart,
            1,
            vec![Expr::from(x_coef), Expr::from(y_coef)]
        );

        // d(dω) = 0
        let d_omega = omega.exterior_derivative().unwrap();
        let dd_omega = d_omega.exterior_derivative().unwrap();

        prop_assert!(dd_omega.is_zero());
    }
}
```

---

### 3.4 Benchmark Tests

Performance tests for critical operations:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_chart_transition(c: &mut Criterion) {
    let m = Sphere2::new();
    let north = m.get_chart_by_name("north_stereo").unwrap();
    let south = m.get_chart_by_name("south_stereo").unwrap();

    c.bench_function("sphere_chart_transition", |b| {
        b.iter(|| {
            north.transition_to(black_box(&south))
        })
    });
}

fn bench_lie_bracket(c: &mut Criterion) {
    let m = EuclideanSpace::new(3);
    let chart = m.default_chart().unwrap();

    let x = VectorField::random(m.clone(), &chart);
    let y = VectorField::random(m.clone(), &chart);

    c.bench_function("lie_bracket_3d", |b| {
        b.iter(|| {
            black_box(&x).lie_bracket(black_box(&y))
        })
    });
}

fn bench_exterior_derivative(c: &mut Criterion) {
    let m = EuclideanSpace::new(3);
    let chart = m.default_chart().unwrap();

    let omega = DiffForm::random(m.clone(), &chart, 1);

    c.bench_function("exterior_derivative_1form", |b| {
        b.iter(|| {
            black_box(&omega).exterior_derivative()
        })
    });
}

criterion_group!(benches, bench_chart_transition, bench_lie_bracket, bench_exterior_derivative);
criterion_main!(benches);
```

---

### 3.5 Correctness Tests (Mathematical Identities)

Verify fundamental theorems of differential geometry:

```rust
#[test]
fn test_stokes_theorem_circle() {
    // ∫_∂D ω = ∫_D dω
    let disk = Disk2::new();
    let boundary = Circle::new();

    // ω = x dy (1-form on disk)
    let chart = disk.default_chart().unwrap();
    let x = chart.coordinate_symbol(0);
    let omega = DiffForm::from_components(
        disk.clone(),
        &chart,
        1,
        vec![Expr::from(0), Expr::Symbol(x.clone())]
    );

    // Left side: ∫_∂D ω
    let boundary_integral = boundary.integrate_form(&omega).unwrap();

    // Right side: ∫_D dω
    let d_omega = omega.exterior_derivative().unwrap();
    let disk_integral = disk.integrate_form(&d_omega).unwrap();

    assert!((boundary_integral - disk_integral).abs() < 1e-6);
}

#[test]
fn test_jacobi_identity() {
    // [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0
    let m = EuclideanSpace::new(3);
    let chart = m.default_chart().unwrap();

    let x = VectorField::random(m.clone(), &chart);
    let y = VectorField::random(m.clone(), &chart);
    let z = VectorField::random(m.clone(), &chart);

    let term1 = x.lie_bracket(&y.lie_bracket(&z).unwrap()).unwrap();
    let term2 = y.lie_bracket(&z.lie_bracket(&x).unwrap()).unwrap();
    let term3 = z.lie_bracket(&x.lie_bracket(&y).unwrap()).unwrap();

    let result = term1 + term2 + term3;
    assert!(result.is_zero());
}

#[test]
fn test_cartan_formula() {
    // ℒ_X ω = i_X(dω) + d(i_X ω)
    let m = EuclideanSpace::new(3);
    let chart = m.default_chart().unwrap();

    let x = VectorField::random(m.clone(), &chart);
    let omega = DiffForm::random(m.clone(), &chart, 1);

    // Left side
    let lie_deriv = omega.lie_derivative(&x).unwrap();

    // Right side
    let d_omega = omega.exterior_derivative().unwrap();
    let ix_d_omega = d_omega.interior_product(&x).unwrap();
    let ix_omega = omega.interior_product(&x).unwrap();
    let d_ix_omega = ix_omega.exterior_derivative().unwrap();
    let rhs = ix_d_omega + d_ix_omega;

    assert_eq!(lie_deriv, rhs);
}
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

**Week 1: Expression Infrastructure**
- [ ] Implement `ExprVisitor` and `ExprMutator` traits
- [ ] Implement `SymbolCollector`, `Substituter` walkers
- [ ] Add symbolic Jacobian computation to differentiation module
- [ ] Create `CoordinateRegistry` for tracking chart symbols
- [ ] Write unit tests for all walkers

**Week 2: Chart Extensions**
- [ ] Extend `Chart` with transition function computation
- [ ] Implement symbolic Jacobian storage and caching
- [ ] Add `pullback_scalar` method for coordinate transformations
- [ ] Create `TransitionFunction` struct with validation
- [ ] Write tests for chart transitions (polar ↔ cartesian, stereographic)

**Week 3: Trait Hierarchy**
- [ ] Define all manifold-related traits
- [ ] Define scalar field and module traits
- [ ] Define element traits (vector fields, tensor fields)
- [ ] Document trait relationships and design patterns
- [ ] Create example trait implementations

---

### Phase 2: Core Structures (Weeks 4-8)

**Week 4: Scalar Fields**
- [ ] Implement `ScalarField` struct with multi-chart support
- [ ] Implement `DiffScalarFieldAlgebra` as `Parent`
- [ ] Add arithmetic operations (+, -, *, /)
- [ ] Implement `differential()` method
- [ ] Write comprehensive tests

**Week 5: Vector Fields**
- [ ] Implement `VectorField` struct
- [ ] Implement `VectorFieldModule` as `Parent`
- [ ] Add `apply_to_scalar` method
- [ ] Implement `lie_bracket`
- [ ] Add module multiplication (scalar field × vector field)
- [ ] Write tests for all operations

**Week 6: Tensor Fields Part 1**
- [ ] Implement `TensorField` struct with multi-index storage
- [ ] Add index conversion utilities (multi-index ↔ flat index)
- [ ] Implement tensor contraction
- [ ] Add coordinate transformation logic
- [ ] Write tests for rank (1,0), (0,1), (1,1) tensors

**Week 7: Tensor Fields Part 2 & Differential Forms**
- [ ] Implement `DiffForm` struct
- [ ] Implement `exterior_derivative`
- [ ] Implement `wedge` product
- [ ] Implement `interior_product` and `lie_derivative`
- [ ] Write tests for all operations

**Week 8: Tangent and Cotangent Bundles**
- [ ] Implement `TangentBundle` and `TangentSpace`
- [ ] Implement `CotangentBundle` and `CotangentSpace`
- [ ] Add `TangentVector` and `Covector` types
- [ ] Implement coordinate basis generation
- [ ] Write tests for fiber structures

---

### Phase 3: Integration & Testing (Weeks 9-12)

**Week 9: Example Manifolds**
- [ ] Implement `RealLine` with symbolic coordinates
- [ ] Implement `Circle` with two charts
- [ ] Implement `Sphere2` with stereographic charts
- [ ] Implement `Torus2` with periodic coordinates
- [ ] Add validation tests for each

**Week 10: Advanced Features**
- [ ] Implement parallelizable manifold detection
- [ ] Add `VectorFieldFreeModule` for parallelizable case
- [ ] Implement metric tensor structure (for Riemannian manifolds)
- [ ] Add connection and covariant derivative (basic version)
- [ ] Write integration tests

**Week 11: Integration Tests & Mathematical Identities**
- [ ] Test Stokes' theorem on various manifolds
- [ ] Test Jacobi identity for Lie brackets
- [ ] Test Cartan formula for Lie derivatives
- [ ] Test coordinate transformation consistency
- [ ] Add property-based tests

**Week 12: Documentation, Benchmarks & Polish**
- [ ] Write comprehensive module documentation
- [ ] Add examples to all public APIs
- [ ] Create tutorial notebooks/examples
- [ ] Run benchmarks and optimize critical paths
- [ ] Code review and refactoring

---

## Appendices

### Appendix A: Trait-to-Class Mapping

Quick reference for converting SageMath classes to Rust traits:

| SageMath Class | Rust Trait/Struct | Notes |
|---|---|---|
| `UniqueRepresentation` | `UniqueRepresentation` | Already in rustmath-core |
| `Parent` | `Parent` | Already in rustmath-core |
| `Element` | (no trait, just Clone + PartialEq) | Generic constraint |
| `ManifoldSubset` | `ManifoldSubsetTrait` | Base trait for subsets |
| `TopologicalManifold` | `TopologicalManifoldTrait` | Adds atlas |
| `DifferentiableManifold` | `DifferentiableManifoldTrait` | Adds C^∞ structure |
| `ScalarFieldAlgebra` | `ScalarFieldAlgebraTrait` | Parent for scalar fields |
| `DiffScalarFieldAlgebra` | `DiffScalarFieldAlgebraTrait` | Specialization |
| `VectorFieldModule` | `VectorFieldModuleTrait` | Module over C^∞(M) |
| `TensorFieldModule` | `TensorFieldModuleTrait` | Higher-rank tensors |
| `FiniteRankFreeModule` | `ParentWithBasis` | Already in rustmath-core |
| `TangentSpace` | `TangentSpaceTrait` | Fiber of TM |
| `ScalarField` | `ScalarField` (struct) | Implements ScalarFieldTrait |
| `VectorField` | `VectorField` (struct) | Implements VectorFieldTrait |
| `TensorField` | `TensorField` (struct) | Implements TensorFieldTrait |
| `TangentVector` | `TangentVector` (struct) | Element of T_pM |

---

### Appendix B: Critical Design Decisions

**1. Why Arc instead of Rc?**
- Allows multithreaded computation (parallel tensor operations)
- Slightly higher overhead, but worth it for future parallelism

**2. Why HashMap for chart storage instead of Vec?**
- Allows efficient lookup by `ChartId`
- Prevents duplicate chart registration
- Easier to cache transition functions

**3. Why symbolic expressions for components?**
- Enables automatic differentiation (Lie derivatives, exterior derivatives)
- Allows coordinate transformations without numerical approximation
- Cleaner than storing closures + derivatives separately

**4. Why not implement multiple trait inheritance directly?**
- Rust doesn't support it
- Use trait composition instead: implement all relevant traits separately
- Example: `VectorFieldParal` implements both `VectorFieldTrait` and `FiniteRankFreeModuleElement`

**5. Caching strategy**
- Transition functions: cache in `Arc<RwLock<HashMap<...>>>`
- Symbolic expressions: store directly (cheap to clone Arc)
- Numerical values: recompute (less memory, still fast)

---

### Appendix C: Performance Considerations

**Bottlenecks:**
1. **Symbolic differentiation**: Can be slow for large expressions
   - *Mitigation*: Cache derivatives, simplify eagerly
2. **Chart transitions**: Involve substitution + simplification
   - *Mitigation*: Cache computed transitions, use memoization
3. **Tensor operations**: Exponential in rank
   - *Mitigation*: Use sparse representations for high ranks

**Optimization Opportunities:**
1. **Parallel tensor contraction**: Use rayon for large tensors
2. **Expression simplification**: Use rustmath-symbolic's simplify infrastructure
3. **Numerical evaluation**: SIMD for batch point evaluation

---

### Appendix D: Future Extensions

**Not in initial implementation, but planned:**

1. **Riemannian Geometry**
   - Metric tensors
   - Christoffel symbols
   - Covariant derivatives
   - Curvature tensors (Riemann, Ricci, scalar curvature)
   - Geodesics

2. **Lie Groups and Lie Algebras**
   - Lie group manifold structure
   - Left/right-invariant vector fields
   - Maurer-Cartan form
   - Exponential map

3. **Fiber Bundles (General)**
   - Principal bundles
   - Associated bundles
   - Connection forms

4. **Symplectic Geometry**
   - Symplectic manifolds
   - Hamiltonian vector fields
   - Poisson brackets

5. **Integration on Manifolds**
   - Numerical integration of forms
   - Stokes' theorem verification
   - Volume computation

---

### Appendix E: References

**SageMath Documentation:**
- [SageMath Manifolds Tutorial](https://doc.sagemath.org/html/en/reference/manifolds/)
- [arXiv:1804.07346](https://arxiv.org/pdf/1804.07346) - "Symbolic tensor calculus on manifolds: a SageMath implementation"

**Mathematical References:**
- Lee, *Introduction to Smooth Manifolds* (2nd ed.)
- Spivak, *Calculus on Manifolds*
- Kobayashi & Nomizu, *Foundations of Differential Geometry*

**Rust Resources:**
- [Trait objects vs generics](https://doc.rust-lang.org/book/ch17-02-trait-objects.html)
- [Arc and RwLock](https://doc.rust-lang.org/std/sync/struct.Arc.html)
- [Visitor pattern in Rust](https://rust-unofficial.github.io/patterns/patterns/behavioural/visitor.html)

---

## Summary

This blueprint provides a complete roadmap for implementing differential manifolds in RustMath. The architecture leverages Rust's trait system to achieve the flexibility of SageMath's class hierarchy while maintaining type safety and performance.

**Key Takeaways:**
1. **Traits replace inheritance**: Composition of traits achieves multiple inheritance
2. **Symbolic infrastructure is critical**: Expression walkers enable differential geometry
3. **Incremental implementation**: Start with scalar fields, build up to tensor fields
4. **Test-driven development**: Mathematical identities serve as correctness tests
5. **~12 week timeline**: Aggressive but achievable with focused effort

The resulting system will support research-grade differential geometry computations while maintaining RustMath's commitment to exact arithmetic and zero unsafe code.
