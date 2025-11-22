# Build Failure Investigation & Fix Strategy
## RustMath: rustmath-schemes, rustmath-groups, rustmath-liealgebras

**Date**: 2025-11-22
**Session**: claude/investigate-rust-build-failures-01WALLXFZts2HCufTSfza6pE
**Total Errors**: ~99 errors across 3 crates

---

## Executive Summary

This document outlines a comprehensive strategy to fix build failures in three RustMath crates. These crates implement advanced mathematical structures and are experiencing compilation errors likely due to:

1. **Trait bound issues** - Complex generic constraints not being satisfied
2. **Missing trait implementations** - Required traits not implemented for types
3. **Type inference failures** - Compiler unable to infer generic parameters
4. **Import/module errors** - Missing or incorrect use statements
5. **Lifetime annotation issues** - Complex borrowing patterns in generic code

---

## Crate Overview

### 1. rustmath-schemes (17 files)
**Purpose**: Algebraic schemes, elliptic curves, projective geometry
**Key Modules**:
- `affine/` - Affine schemes
- `elliptic_curves/` - Elliptic curve implementations (generic, rational, isogeny, Heegner)
- `projective/` - Projective schemes and morphisms
- `graded_ring.rs`, `line_bundle.rs`, `veronese.rs`, `segre.rs`

**Expected Error Categories**:
- Generic ring/field trait bounds
- Morphism type constraints
- Point arithmetic trait requirements

### 2. rustmath-groups (37 files)
**Purpose**: Group theory implementations
**Key Modules**:
- Permutation groups, matrix groups, free groups
- Braid groups (Artin, cubic, cactus)
- Finitely presented groups
- GAP interface bindings (`libgap_*`)
- Representations and conjugacy classes

**Expected Error Categories**:
- Group element trait implementations
- External library bindings (GAP)
- Generic group operations
- Subgroup and quotient constructions

### 3. rustmath-liealgebras (38 files)
**Purpose**: Lie algebras and representation theory
**Key Modules**:
- Classical Lie algebras, exceptional Lie algebras
- Cartan types, Dynkin diagrams, root systems
- Representations (Verma modules, BGG resolution)
- Universal enveloping algebras (Poincaré-Birkhoff-Witt)
- Affine and Kac-Moody algebras

**Expected Error Categories**:
- Lie bracket trait constraints
- Module and representation trait bounds
- Root system generic parameters
- Cartan matrix type requirements

---

## Error Categories & Fix Strategies

### Category 1: Missing Trait Bounds
**Symptoms**: "the trait bound `X: Trait` is not satisfied"
**Fix Strategy**:
1. Identify required traits from error messages
2. Add trait bounds to generic parameters
3. Propagate bounds through type hierarchies
4. Use `where` clauses for complex constraints

**Example Fix**:
```rust
// Before
impl<R> MyType<R> { ... }

// After
impl<R: Ring + Clone> MyType<R> where R: EuclideanDomain { ... }
```

### Category 2: Unimplemented Traits
**Symptoms**: "method `foo` not found", "no method named `bar`"
**Fix Strategy**:
1. Implement missing trait for type
2. Derive standard traits where possible (Clone, Debug, PartialEq)
3. Add blanket implementations for common cases

### Category 3: Type Inference Failures
**Symptoms**: "cannot infer type for `T`", "type annotations needed"
**Fix Strategy**:
1. Add explicit type annotations
2. Use turbofish syntax `::<Type>`
3. Refactor to provide more type context

### Category 4: Import/Module Errors
**Symptoms**: "unresolved import", "cannot find type `X` in this scope"
**Fix Strategy**:
1. Add missing `use` statements
2. Fix module visibility (`pub`)
3. Verify dependency versions in Cargo.toml

### Category 5: Lifetime Issues
**Symptoms**: "lifetime `'a` does not live long enough"
**Fix Strategy**:
1. Add explicit lifetime parameters
2. Use lifetime bounds (`'a: 'b`)
3. Consider using `'static` or owned types

---

## Implementation Plan

### Phase 1: Dependency Resolution (Critical Path)
**Priority**: Highest
**Reason**: Some errors cascade from failed dependencies

1. **Check rustmath-core**: Ensure all base traits compile
2. **Verify rustmath-polynomials**: Groups/Schemes/Lie algebras depend on polynomials
3. **Confirm rustmath-matrix**: Used extensively in groups and Lie algebras
4. **Validate rustmath-finitefields**: Required for elliptic curves

### Phase 2: Parallel Error Fixes
**Priority**: High
**Approach**: Fix errors in parallel across crates

#### Sub-plan 2A: rustmath-schemes
1. Fix base scheme traits and implementations
2. Fix elliptic curve generic implementations
3. Fix projective space constructions
4. Fix morphisms and line bundles

#### Sub-plan 2B: rustmath-groups
1. Fix core group trait definitions
2. Fix permutation and matrix group implementations
3. Fix GAP interface bindings (may need conditional compilation)
4. Fix representation and conjugacy class code

#### Sub-plan 2C: rustmath-liealgebras
1. Fix base Lie algebra trait
2. Fix classical Lie algebra implementations
3. Fix root system and Cartan matrix code
4. Fix representation theory modules

### Phase 3: Integration Testing
**Priority**: Medium
**Actions**:
1. Run `cargo check` on each crate
2. Run `cargo test` on each crate
3. Fix any integration errors
4. Verify examples compile

### Phase 4: Code Quality
**Priority**: Low
**Actions**:
1. Fix all warnings (unused variables, dead code, etc.)
2. Run `cargo clippy` and address lints
3. Format code with `cargo fmt`
4. Update documentation

---

## Common Patterns & Solutions

### Pattern 1: Ring/Field Generic Parameters
**Problem**: Elliptic curves, schemes need correct ring bounds
**Solution**:
```rust
// Ensure Field: Ring trait hierarchy is respected
pub struct EllipticCurve<F: Field> where F: Clone { ... }
```

### Pattern 2: GAP Integration
**Problem**: External library bindings may fail if GAP not available
**Solution**:
```rust
// Use conditional compilation
#[cfg(feature = "gap")]
mod libgap_wrapper;
```

### Pattern 3: Trait Object Constraints
**Problem**: "trait cannot be made into an object"
**Solution**:
```rust
// Add + Sized or use Box<dyn Trait>
trait MyTrait: Sized { ... }
```

### Pattern 4: Associated Type Constraints
**Problem**: Associated types need bounds
**Solution**:
```rust
trait MyTrait {
    type Output: Ring + Clone;
}
```

---

## Risk Assessment

### High Risk:
- **External dependencies**: GAP library bindings may need system libraries
- **Complex generics**: Deep trait hierarchies may have circular constraints
- **Type inference**: May need significant refactoring

### Medium Risk:
- **Missing implementations**: Time-consuming but straightforward
- **Import errors**: Easy to fix but numerous

### Low Risk:
- **Warnings**: Don't block compilation
- **Documentation**: Can be deferred

---

## Success Criteria

✓ All 3 crates pass `cargo check`
✓ All 3 crates pass `cargo test`
✓ No compilation errors
✓ All warnings addressed
✓ Code passes `cargo clippy`

---

## Next Steps

1. **Complete error collection**: Wait for `cargo check` to finish
2. **Categorize all errors**: Group by type and priority
3. **Execute parallel fixes**: Use one-liner prompts below
4. **Test incrementally**: Verify fixes don't introduce new errors
5. **Commit and push**: Save progress to feature branch

---

## One-Liner Fix Prompts (For Parallel Execution)

The following prompts can be executed in parallel to fix errors across all three crates:

### For rustmath-schemes:
1. "Fix all missing trait bound errors in rustmath-schemes by adding appropriate Ring/Field constraints"
2. "Implement missing Clone, Debug, and PartialEq traits for all scheme types in rustmath-schemes"
3. "Fix all type inference errors in rustmath-schemes elliptic curve implementations by adding explicit type annotations"
4. "Resolve all import errors in rustmath-schemes by adding missing use statements"
5. "Fix all lifetime annotation errors in rustmath-schemes projective morphisms"

### For rustmath-groups:
1. "Fix all missing trait bound errors in rustmath-groups by adding appropriate Group trait constraints"
2. "Implement missing Clone, Debug, and PartialEq traits for all group types in rustmath-groups"
3. "Fix all type inference errors in rustmath-groups permutation and matrix group implementations"
4. "Add conditional compilation (#[cfg(feature = \"gap\")]) to all GAP interface code in rustmath-groups"
5. "Resolve all import errors in rustmath-groups by adding missing use statements"
6. "Fix all lifetime annotation errors in rustmath-groups representation code"

### For rustmath-liealgebras:
1. "Fix all missing trait bound errors in rustmath-liealgebras by adding appropriate Lie algebra trait constraints"
2. "Implement missing Clone, Debug, and PartialEq traits for all Lie algebra types in rustmath-liealgebras"
3. "Fix all type inference errors in rustmath-liealgebras root system and Cartan matrix code"
4. "Resolve all import errors in rustmath-liealgebras by adding missing use statements"
5. "Fix all lifetime annotation errors in rustmath-liealgebras representation theory modules"
6. "Fix all associated type constraint errors in rustmath-liealgebras Verma module and BGG resolution code"

### Cross-cutting:
1. "Fix all unused variable warnings across rustmath-schemes, rustmath-groups, and rustmath-liealgebras"
2. "Fix all dead code warnings across rustmath-schemes, rustmath-groups, and rustmath-liealgebras"
3. "Add missing documentation comments to all public APIs in rustmath-schemes, rustmath-groups, and rustmath-liealgebras"

---

## Estimated Effort

- **Error Collection**: 10-15 minutes (cargo check compilation)
- **Error Analysis**: 30-45 minutes
- **Implementation**: 4-6 hours (parallel execution can reduce this)
- **Testing & Validation**: 1-2 hours
- **Total**: ~6-9 hours

---

## Notes

- Compilation is currently blocked on `gmp-mpfr-sys` build (C library linking)
- Once complete, full error list will be appended to this document
- Errors should be tackled in priority order: dependencies first, then trait bounds, then implementations
- Consider using `cargo fix` for automatic fixes where applicable

