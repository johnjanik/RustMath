# ModulesWithBasis Category Implementation Overview

## Project Context
- **Repository**: RustMath (Rust rewrite of SageMath)
- **Current Branch**: `claude/implement-modules-with-basis-01LSiUQyhqv7BDVyPLaxuqq6`
- **Status**: Feature branch for implementing ModulesWithBasis category

---

## 1. EXISTING INFRASTRUCTURE

### 1.1 Category Framework (rustmath-category)
Location: `/home/user/RustMath/rustmath-category/src/`

**Components**:
- **morphism.rs**: Core trait `Morphism` with source/target/compose operations
  - Implements: `IdentityMorphism`, `SetMorphism`, `SetIsomorphism`, `FormalCoercionMorphism`
- **functor.rs**: `Functor` trait for structure-preserving maps between categories
  - Implements: `IdentityFunctor`, `ForgetfulFunctor`
- **natural_transformation.rs**: Natural transformations between functors
- **lib.rs**: Main exports

**Key Design**:
```rust
pub trait Morphism: Clone {
    type Object;
    fn source(&self) -> &Self::Object;
    fn target(&self) -> &Self::Object;
    fn compose(&self, other: &Self) -> Option<Self>;
    fn is_identity(&self) -> bool;
}
```

### 1.2 Core Algebraic Traits (rustmath-core)
Location: `/home/user/RustMath/rustmath-core/src/`

**Key Traits**:
- `Ring`: Basic ring operations (add, mul, neg)
- `Field`: Ring with multiplicative inverses
- `EuclideanDomain`: Ring with division algorithm
- **`Parent`**: Container for elements (sage.structure.parent.Parent equivalent)
- **`ParentWithBasis`**: Parent with distinguished basis
  ```rust
  pub trait ParentWithBasis: Parent {
      type BasisIndex: Clone + PartialEq;
      fn dimension(&self) -> Option<usize>;
      fn basis_element(&self, index: &Self::BasisIndex) -> Option<Self::Element>;
      fn basis_indices(&self) -> Vec<Self::BasisIndex>;
  }
  ```
- `ParentWithGenerators`: Parents with generators (groups, algebras)
- `RingParent`: Parent with ring structure

### 1.3 Module Framework (rustmath-modules)
Location: `/home/user/RustMath/rustmath-modules/src/`

**Existing Module Components**:
- **module.rs**: Base `Module` trait
  ```rust
  pub trait Module: Clone {
      type BaseRing: Ring;
      type Element: Clone + Debug;
      fn base_ring(&self) -> &Self::BaseRing;
      fn rank(&self) -> usize;
      fn zero(&self) -> Self::Element;
      fn is_zero(&self, elem: &Self::Element) -> bool;
      fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
      fn negate(&self, a: &Self::Element) -> Self::Element;
      fn scalar_mul(&self, scalar: &Self::BaseRing, elem: &Self::Element) -> Self::Element;
  }
  ```

- **free_module.rs**: Free modules over rings with basis operations
- **free_module_element.rs**: Elements in free modules
- **free_module_morphism.rs**: Linear maps between modules
- **tensor/**: Comprehensive tensor algebra infrastructure
- **with_basis/**: Stub directory (currently mostly empty)
  - `all.rs`: Empty `All` struct
  - `indexed_element.rs`: Empty `IndexedElement` struct
  - `morphism.rs`: Empty `ModuleMorphismWithBasis` struct
  - `cell_module.rs`: Empty `CellModule` struct
  - `representation.rs`: Empty `Representation` struct
  - `invariant.rs`: (needs checking)
  - `subquotient.rs`: (needs checking)

**Submodules**:
- `fp_graded`: Finitely generated graded modules (partially implemented)
- `fg_pid`: Finitely generated modules over PIDs (partially implemented)

---

## 2. WHAT NEEDS TO BE IMPLEMENTED

### 2.1 ModulesWithBasis Category Structure

Based on SageMath's architecture, ModulesWithBasis should define:

**Parent Category**:
- Inherits from `Modules` category
- Defines modules with a distinguished basis
- When base ring is a field, becomes `VectorSpacesWithBasis`

**Key Methods on Parent**:
1. **Basis Operations**:
   - `basis()` or `basis_keys()` - returns basis indices
   - `basis_element(index)` - gets basis element by index
   - `dimension()` / `rank()` - size of module

2. **Element Operations**:
   - `coefficients()` - coefficients in the distinguished basis
   - `support()` - indices with non-zero coefficients
   - `coefficient(index)` - single coefficient

3. **Category Methods**:
   - `basis_from_elements(elements)` - extract basis from spanning set
   - Direct sum construction
   - Tensor products (preserve basis structure)
   - Exterior/symmetric powers

### 2.2 Element Operations

**Key Methods on ModulesWithBasis.Element**:
1. Scalar multiplication by basis elements
2. Coefficient retrieval: `element.coefficient(index)`
3. Support: `element.support()` or `element.items()`
4. Conversion to/from dense representation

### 2.3 Morphism Operations

**ModuleMorphismWithBasis should support**:
1. Definition by action on basis: `morphism.on_basis(index)`
2. Extend linearly to all elements
3. Matrix representation w.r.t. basis
4. Kernel and image operations

### 2.4 Subcategories to Support

1. **FiniteDimensionalModulesWithBasis**: Modules with finite dimension
2. **GradedModulesWithBasis**: Modules with grading that respects basis
3. **FieldModulesWithBasis**: Vector spaces with basis (when base ring is field)

---

## 3. ARCHITECTURE RECOMMENDATIONS

### 3.1 Design Pattern

Follow RustMath's trait-based generics pattern:

```rust
// Parent type for modules with basis
pub trait ModuleWithBasis: ParentWithBasis {
    type BaseRing: Ring;
    // Additional methods specific to modules with basis
}

// Element type that represents coefficients
pub struct ModuleWithBasisElement<R: Ring> {
    // Sparse representation: map from basis index to coefficient
    coefficients: BTreeMap<usize, R>,
}

// Morphisms
pub trait ModuleWithBasisMorphism: Morphism {
    // Actions defined on basis elements
    fn on_basis(&self, index: usize) -> Self::Element;
}
```

### 3.2 Element Representation

**Recommended Representation**:
```rust
pub struct ModuleWithBasisElement<I, R> {
    coefficients: BTreeMap<I, R>,  // Sparse representation
    // or for dense: Vec<(I, R)>
}
```

**Why BTreeMap**:
- Efficient for sparse vectors (common in basis-indexed modules)
- Ordered iteration by index
- Natural support for symbolic indices

### 3.3 File Organization

```
rustmath-modules/src/with_basis/
├── lib.rs                      # Main category definition
├── parent.rs                   # ModuleWithBasis trait
├── element.rs                  # Element type & operations
├── morphism.rs                 # Morphism type & operations
├── indexed_element.rs          # Specialized element for integer indices
├── cell_module.rs              # Cell modules (combinatorial structure)
├── representation.rs           # Rep theory - module as representation
├── subquotient.rs              # Quotient modules with basis
├── invariant.rs                # Invariant submodules
├── examples.rs                 # Examples and tests
└── tensor.rs                   # Tensor products with basis
```

---

## 4. INTEGRATION POINTS

### 4.1 With Existing Code

1. **Inherit from `ParentWithBasis`**: Use existing parent infrastructure
2. **Compose with `Module` trait**: Add basis structure to modules
3. **Implement `Morphism` for basis morphisms**: Use category framework
4. **Use tensor module infrastructure**: Build on `/tensor/` crate

### 4.2 With rustmath-core

- Extend `Parent` trait if needed for basis-specific operations
- Ensure compatibility with `ParentWithBasis` trait definition

### 4.3 With rustmath-category

- Implement `Morphism` for `ModuleWithBasisMorphism`
- Create functors between categories (e.g., forget basis structure)

---

## 5. IMPLEMENTATION SEQUENCE

### Phase 1: Core Infrastructure (Priority: CRITICAL)
1. [ ] Define `ModuleWithBasis` trait extending `ParentWithBasis`
2. [ ] Implement `ModuleWithBasisElement<I, R>` generic type
3. [ ] Add basic operations: access coefficients, support
4. [ ] Write tests for element operations

### Phase 2: Morphisms & Category Structure (Priority: HIGH)
1. [ ] Define `ModuleWithBasisMorphism` trait
2. [ ] Implement morphism composition
3. [ ] Define category methods (kernel, image, etc.)
4. [ ] Write tests for morphisms

### Phase 3: Specific Implementations (Priority: MEDIUM)
1. [ ] Free modules with basis (`FreeModuleWithBasis`)
2. [ ] Indexed modules with integer indices (`IndexedFreeModule`)
3. [ ] Cell modules for combinatorial structures
4. [ ] Graded modules with basis

### Phase 4: Advanced Features (Priority: LOW)
1. [ ] Tensor products with basis
2. [ ] Exterior/symmetric powers with basis
3. [ ] Subquotient modules
4. [ ] Morphism matrix representations

---

## 6. COMPARISON WITH SAGEMATH

### SageMath Structure
```python
# sage/categories/modules_with_basis.py
class ModulesWithBasis(Category_over_base_ring):
    def _call_(self, element):
        # Element constructor
        
    class ParentMethods:
        def basis(self):
            """Return the basis of the module"""
        def basis_keys(self):
            """Return indices for basis elements"""
        def basis_matrix(self):
            """Return basis as matrix"""
        
    class ElementMethods:
        def coefficient(self, index):
            """Get coefficient of basis element"""
        def items(self):
            """Iterate over (index, coefficient) pairs"""
```

### RustMath Equivalent
```rust
// rustmath-modules/src/with_basis/lib.rs
pub trait ModuleWithBasis: ParentWithBasis + Module {
    fn basis_keys(&self) -> Vec<Self::BasisIndex>;
    fn basis_matrix(&self) -> Matrix<Self::BaseRing>;
}

pub trait ModuleWithBasisElement: Clone {
    type Index;
    fn coefficient(&self, index: &Self::Index) -> Option<Self::BaseRing>;
    fn items(&self) -> Vec<(Self::Index, Self::BaseRing)>;
}
```

---

## 7. KEY REFERENCES IN CODEBASE

1. `/home/user/RustMath/rustmath-core/src/parent.rs` - ParentWithBasis trait
2. `/home/user/RustMath/rustmath-modules/src/module.rs` - Module trait
3. `/home/user/RustMath/rustmath-modules/src/free_module.rs` - Example module
4. `/home/user/RustMath/rustmath-category/src/morphism.rs` - Morphism pattern
5. `/home/user/RustMath/rustmath-modules/src/tensor/` - Tensor algebra structure

---

## 8. KNOWN LIMITATIONS & NOTES

1. **Stub Files**: The `with_basis/` directory currently contains only stub implementations
2. **Sparse vs Dense**: Need to decide representation strategy (sparse recommended)
3. **Index Type**: Need to support flexible basis indexing (integers, tuples, custom types)
4. **Grading**: Integration with graded modules needs careful design
5. **Morphism Representation**: Need way to represent morphisms compactly (matrix w.r.t. basis)

