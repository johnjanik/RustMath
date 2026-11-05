# Complex Dependencies and TODOs

This document lists complex dependencies and implementation issues encountered during development that need to be addressed in future sessions.

## 1. Matrix<Expr> - Expression Matrices

**Issue**: `Matrix<R: Ring>` requires `R` to implement the `Ring` trait, but `rustmath_symbolic::Expr` does not implement `Ring`.

**Location**:
- `rustmath-manifolds/src/maps.rs` (Jacobian computation)
- Any code that needs symbolic matrix operations

**Current Workaround**: Using `Vec<Vec<Expr>>` instead of `Matrix<Expr>`

**Future Fix**: Either:
- Implement `Ring` trait for `Expr` in `rustmath-symbolic`
- Create a separate `SymbolicMatrix` type that doesn't require `Ring`
- Use a different representation for symbolic Jacobians

**Impact**: Moderate - affects differential geometry computations

---

## 2. TopologicalVectorBundle Generic Parameters

**Issue**: `TopologicalVectorBundle<M, F>` requires two generic parameters (manifold type and field type), making it difficult to use with trait objects and `Arc<DifferentiableManifold>`.

**Location**:
- `rustmath-manifolds/src/topology.rs` (ChernClass, PontryaginClass, EulerClass)
- Any code working with vector bundles and characteristic classes

**Current Workaround**: Store manifold and rank separately in characteristic class structs instead of storing the bundle

**Future Fix**: Consider:
- Using type erasure for bundles
- Making characteristic classes generic over bundle type
- Creating a non-generic wrapper type for bundles

**Impact**: Moderate - limits integration between topology and bundle theory

---

## 3. TangentVector Construction API

**Issue**: Unclear/inconsistent API for creating `TangentVector` instances. Methods like `from_components_simple()` or `new()` may not exist or have different signatures than expected.

**Location**:
- `rustmath-manifolds/src/maps.rs` (PushForward implementation)
- `rustmath-manifolds/src/tangent_space.rs`

**Current Workaround**: Simplified placeholder implementation in push forward

**Future Fix**:
- Audit and standardize TangentVector creation API
- Document all construction methods
- Add helper methods for common use cases

**Impact**: Low to Moderate - affects differential map implementations

---

## 4. ScalarField Expression API

**Issue**: Multiple ways to create scalar fields from expressions, unclear which methods exist and what their signatures are.

**Location**:
- `rustmath-manifolds/src/scalar_field.rs`
- `rustmath-manifolds/src/maps.rs` (PullBack)
- `rustmath-manifolds/src/catalog.rs`

**Current Workaround**: Using `ScalarField::from_expr()` which may or may not be the correct API

**Future Fix**:
- Standardize scalar field creation API
- Add clear documentation for all construction methods
- Consider builder pattern for complex scenarios

**Impact**: Low - mostly affects convenience

---

## 5. RiemannianMetric Creation from Components

**Issue**: Creating metrics from tensor fields or component expressions has unclear/undocumented API.

**Location**:
- `rustmath-manifolds/src/catalog.rs` (Minkowski, Schwarzschild, Kerr)
- `rustmath-manifolds/src/riemannian.rs`

**Current Workaround**: Using assumed methods like `from_tensor()` and `euclidean()`/`round_sphere()`/`hyperbolic()`

**Future Fix**:
- Document all RiemannianMetric construction methods
- Add examples for common metrics
- Verify that helper methods (euclidean, round_sphere, etc.) actually exist

**Impact**: Moderate - affects all Riemannian geometry code

---

## 6. Partial Eq Implementation for DiffForm

**Issue**: `DiffForm` doesn't implement `PartialEq`, but `Parent` trait requires `Element: PartialEq`.

**Location**:
- `rustmath-manifolds/src/diff_form.rs`
- `rustmath-manifolds/src/modules.rs` (DiffFormModule)

**Current Workaround**: None yet - this causes compilation errors

**Future Fix**:
- Implement `PartialEq` for `DiffForm` (compare by components)
- Consider if exact equality or approximate equality is needed
- May need to propagate to `TensorField` as well

**Impact**: High - blocks module/parent implementations

---

## 7. Symbolic Expression Simplification

**Issue**: No clear API for simplifying complex symbolic expressions produced by tensor operations.

**Location**:
- Throughout `rustmath-manifolds` wherever expressions are manipulated
- `rustmath-symbolic` simplification infrastructure

**Current Workaround**: No simplification - expressions may grow large

**Future Fix**:
- Implement comprehensive simplification for `Expr`
- Add automatic simplification hooks
- Consider caching simplified forms

**Impact**: Low initially, High for complex computations

---

## 8. LieGroup and ComplexManifold Dependencies

**Issue**: `catalog.rs` imports `LieGroup` and `ComplexManifold` which may have their own dependency issues.

**Location**:
- `rustmath-manifolds/src/catalog.rs` (SpecialOrthogonalGroup, SpecialUnitaryGroup, ComplexProjectiveSpace)

**Current Workaround**: Assuming these exist and work correctly

**Future Fix**:
- Verify LieGroup and ComplexManifold implementations are complete
- Test integration between catalog and these advanced structures

**Impact**: Unknown - depends on state of those modules

---

## 9. Chart Domain Bounds

**Issue**: Charts need domain information for integration, but this isn't systematically tracked.

**Location**:
- `rustmath-manifolds/src/chart.rs`
- `rustmath-manifolds/src/integration.rs`

**Current Workaround**: Added `get_domain_bounds()` with default implementation

**Future Fix**:
- Make domain tracking systematic
- Allow charts to specify their actual domain
- Handle charts with non-rectangular domains

**Impact**: Moderate - affects numerical integration accuracy

---

## 10. Numerical Evaluation of Symbolic Expressions

**Issue**: Need robust way to evaluate symbolic expressions at numerical points, handling all function types.

**Location**:
- `rustmath-manifolds/src/integration.rs` (`evaluate_to_float()`)
- Anywhere symbolic→numerical conversion is needed

**Current Workaround**: Basic implementation supporting common functions

**Future Fix**:
- Comprehensive function evaluation
- Error handling for undefined operations
- Type safety for evaluation context

**Impact**: High - affects all numerical computations

---

## Priority Ranking

**High Priority** (blocks major functionality):
1. PartialEq for DiffForm (#6)
2. Numerical evaluation (#10)
3. TopologicalVectorBundle generics (#2)

**Medium Priority** (limits features):
4. Matrix<Expr> (#1)
5. RiemannianMetric API (#5)
6. Chart domains (#9)

**Low Priority** (conveniences):
7. TangentVector API (#3)
8. ScalarField API (#4)
9. Simplification (#7)
10. Advanced structure integration (#8)

---

## Notes for Future Sessions

- Consider creating integration tests that exercise these dependencies
- May want to refactor some core traits to be less restrictive
- Documentation is key - many issues stem from unclear APIs
- Some workarounds (like Vec<Vec<Expr>>) are acceptable long-term if documented

Last Updated: 2025-11-17
