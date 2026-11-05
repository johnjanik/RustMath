# ModulesWithBasis Implementation - Quick Reference Guide

## What Should I Implement?

The **ModulesWithBasis** category for RustMath is a mathematical structure that combines:
1. **Modules** (generalizations of vector spaces) from `rustmath-modules`
2. **Distinguished Basis** infrastructure from `rustmath-core`s `ParentWithBasis`
3. **Category-theoretic Morphisms** from `rustmath-category`

This represents modules where elements are stored using sparse coordinates in a fixed basis.

---

## Quick Start: Where to Implement

**Primary Target Location**: `/home/user/RustMath/rustmath-modules/src/with_basis/`

**Files to Create/Modify**:
```
with_basis/
├── lib.rs              ← Main exports (MODIFY)
├── parent.rs           ← NEW: ModuleWithBasis trait
├── element.rs          ← NEW: ModuleWithBasisElement<I, R>
├── morphism.rs         ← EXPAND: ModuleWithBasisMorphism
├── indexed_element.rs  ← EXPAND: IndexedElement specialization
└── examples.rs         ← NEW: Tests and examples
```

---

## Core Types You'll Implement

### 1. ModuleWithBasis Trait (parent.rs)
```rust
pub trait ModuleWithBasis: ParentWithBasis + Module {
    // Basis operations
    fn basis_keys(&self) -> Vec<Self::BasisIndex>;
    fn basis_matrix(&self) -> Matrix<Self::BaseRing>;
    
    // Category operations
    fn direct_sum(&self, other: &Self) -> Self;
    fn tensor_product(&self, other: &Self) -> Self;
}
```

### 2. ModuleWithBasisElement Type (element.rs)
```rust
pub struct ModuleWithBasisElement<I: Ord + Clone, R: Ring> {
    coefficients: BTreeMap<I, R>,  // Sparse representation
}

impl<I: Ord + Clone, R: Ring> ModuleWithBasisElement<I, R> {
    pub fn new(coefficients: BTreeMap<I, R>) -> Self { ... }
    pub fn coefficient(&self, index: &I) -> Option<&R> { ... }
    pub fn support(&self) -> Vec<I> { ... }
    pub fn items(&self) -> Vec<(I, R)> { ... }
    // Arithmetic operations: add, negate, scalar_mul
}
```

### 3. ModuleWithBasisMorphism Type (morphism.rs)
```rust
pub struct ModuleWithBasisMorphism<I: Ord + Clone, R: Ring> {
    source: Box<dyn ModuleWithBasis<I, R>>,
    target: Box<dyn ModuleWithBasis<I, R>>,
    basis_action: BTreeMap<I, ModuleWithBasisElement<I, R>>,
}

impl<I: Ord + Clone, R: Ring> Morphism for ModuleWithBasisMorphism<I, R> {
    // Implement source(), target(), compose(), on_basis()
}
```

---

## Key Infrastructure You'll Use

### From rustmath-core
- `ParentWithBasis` trait (already defined)
  - `dimension()` → basis size
  - `basis_element(index)` → get nth basis element
  - `basis_indices()` → all indices

### From rustmath-category
- `Morphism` trait
  - `source()` / `target()` → domain/codomain
  - `compose()` → morphism composition

### From rustmath-modules
- `Module` trait
  - `add()`, `negate()`, `scalar_mul()`
  - `zero()`, `is_zero()`

---

## Element Representation (Why BTreeMap?)

**Sparse Representation**:
```rust
// Instead of: vec![1, 0, 0, 5, 0, 0, 3]  (dense)
// Use: BTreeMap from [0 → 1, 3 → 5, 6 → 3]  (sparse)

BTreeMap<I, R> {
    0 → 1,    // coefficient at basis index 0
    3 → 5,    // coefficient at basis index 3
    6 → 3,    // coefficient at basis index 6
    // Rest are implicitly zero
}
```

**Advantages**:
- Space efficient for sparse vectors (common in modules)
- Natural for symbolic basis indexing (not just integers)
- Ordered iteration preserves basis order
- Easy to implement: `coefficient(index)`, `support()`, `items()`

---

## Trait Integration Pattern

```
┌─────────────────────────────────────────────┐
│ ParentWithBasis (from rustmath-core)       │
│ ├─ BasisIndex type                         │
│ ├─ dimension()                             │
│ ├─ basis_element(index)                    │
│ └─ basis_indices()                         │
└─────────────────────────────────────────────┘
                    ▲
                    │ extends
                    │
┌─────────────────────────────────────────────┐
│ ModuleWithBasis (NEW - you implement)      │
│ ├─ basis_keys() → basis indices            │
│ ├─ basis_matrix() → matrix form            │
│ ├─ direct_sum()                            │
│ ├─ tensor_product()                        │
│ └─ quotient_module()                       │
└─────────────────────────────────────────────┘
                    ▲
                    │ composes with
                    ▼
┌─────────────────────────────────────────────┐
│ Module (from rustmath-modules)             │
│ ├─ add(), negate()                         │
│ ├─ scalar_mul()                            │
│ ├─ zero(), is_zero()                       │
│ └─ base_ring()                             │
└─────────────────────────────────────────────┘
```

---

## Implementation Checklist (Phase 1)

### Core Infrastructure:
- [ ] Define `ModuleWithBasis` trait extending `ParentWithBasis`
- [ ] Implement `ModuleWithBasisElement<I, R>` struct
- [ ] Implement `element.coefficient(index)`
- [ ] Implement `element.support()`
- [ ] Implement `element.items()`
- [ ] Implement arithmetic: `add()`, `negate()`, `scalar_mul()`
- [ ] Write element tests

### Basic Morphisms:
- [ ] Define `ModuleWithBasisMorphism<I, R>` struct
- [ ] Implement `Morphism` trait for morphisms
- [ ] Implement `morphism.on_basis(index)`
- [ ] Implement morphism composition
- [ ] Write morphism tests

---

## Where Does This Fit in SageMath?

SageMath's structure:
```
sage/categories/
├── modules_with_basis.py  ← You're implementing this!
├── modules.py             ← Already have rustmath-modules
├── categories.py          ← Already have rustmath-category
└── ...

sage/modules/
├── with_basis/
│   ├── indexed_element.py     ← You'll expand this
│   ├── cell_module.py         ← You'll expand this
│   ├── morphism.py            ← You'll expand this
│   └── subquotient.py         ← You'll expand this
└── ...
```

Your implementation is **the bridge** between abstract modules and concrete basis-indexed representations.

---

## Files You Can Reference

1. **For Parent pattern**: `/home/user/RustMath/rustmath-core/src/parent.rs` (lines 78-96)
   - Look at `ParentWithBasis` trait

2. **For Module pattern**: `/home/user/RustMath/rustmath-modules/src/free_module.rs` (lines 1-91)
   - See how `FreeModule` implements `Module` trait

3. **For Morphism pattern**: `/home/user/RustMath/rustmath-category/src/morphism.rs` (lines 24-42)
   - See `Morphism` trait and composition pattern

4. **For element operations**: `/home/user/RustMath/rustmath-modules/src/free_module_element.rs`
   - See how elements implement operations

5. **For tensor infrastructure**: `/home/user/RustMath/rustmath-modules/src/tensor/free_module_basis.rs`
   - See basis operations on tensor elements

---

## Expected Completion

**Phase 1 (Core)**: ~2-3 hours
- Element type and basic operations
- ModuleWithBasis trait definition
- Basic tests

**Phase 2 (Morphisms)**: ~2-3 hours
- Morphism type and composition
- Kernel/image operations
- Integration tests

**Phase 3 (Implementations)**: ~2-3 hours
- FreeModuleWithBasis concrete type
- IndexedElement specialization
- CellModule example

**Total**: ~6-9 hours for a solid Phase 1-2 implementation

---

## Document References

- **Full Overview**: `/home/user/RustMath/MODULES_WITH_BASIS_OVERVIEW.md`
- **Architecture Diagram**: `/home/user/RustMath/MODULESWITHBASIS_ARCHITECTURE.txt`
- **This Quick Reference**: You're reading it!

---

## Questions to Guide Your Implementation

1. **Element Storage**: Should I use `BTreeMap<I, R>` or `Vec<(I, R)>`?
   - Answer: BTreeMap for fast lookup and sparse support

2. **Index Flexibility**: Should I support any `Ord` type as basis index?
   - Answer: Yes! Generic `I: Ord + Clone` gives maximum flexibility

3. **Morphism Definition**: How do I define a morphism compactly?
   - Answer: Store action on basis elements, extend linearly

4. **Category Methods**: Should morphisms compose efficiently?
   - Answer: Lazy evaluation - compose representations, not concrete matrices

5. **Inheritance vs Composition**: Trait bounds or concrete fields?
   - Answer: Trait bounds for flexibility (implement Morphism, not contain it)

---

**Next Steps**: Read `/home/user/RustMath/MODULES_WITH_BASIS_OVERVIEW.md` for detailed architecture!

