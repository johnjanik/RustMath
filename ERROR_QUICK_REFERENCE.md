# Quick Reference for Parallel Fix Prompts

## High Priority Errors (Foundation)

### Group 1: E0432/E0433/E0412/E0422 - Missing Types and Exports
**Priority:** HIGHEST - These block everything else
**Total:** 128 errors

#### Fix Prompt Template 1: Export Missing Types from rustmath-rings/function_field
**File:** `/home/user/RustMath/rustmath-rings/src/function_field/mod.rs`

Missing exports needed:
```
function_field_polymod::{FunctionField_polymod, FunctionField_simple, FunctionField_char_zero,
                         FunctionField_integral, FunctionField_char_zero_integral,
                         FunctionField_global, FunctionField_global_integral}
function_field_rational::{RationalFunctionField_char_zero, RationalFunctionField_global}
ideal_rational::{FunctionFieldIdeal_rational, FunctionFieldIdealInfinite_rational}
order_basis::{FunctionFieldOrder_basis, FunctionFieldOrderInfinite_basis}
order_rational::{FunctionFieldMaximalOrder_rational, FunctionFieldMaximalOrderInfinite_rational}
valuation::{FunctionFieldValuation_base, ClassicalFunctionFieldValuation_base,
            RationalFunctionFieldValuation_base}
maps::{FunctionFieldMorphism_polymod, FunctionFieldMorphism_rational}
place_rational::FunctionFieldPlace_rational
```

#### Fix Prompt Template 2: Fix Drinfeld Module Exports
**Files:**
- `/home/user/RustMath/rustmath-rings/src/function_field/drinfeld_modules/mod.rs`
- `/home/user/RustMath/rustmath-rings/src/function_field/drinfeld_modules/charzero_drinfeld_module.rs`

Missing exports:
```
finite_drinfeld_module::DrinfeldModule_finite
charzero_drinfeld_module::{DrinfeldModule_charzero, DrinfeldModule_rational}
place_polymod::FunctionFieldPlace_polymod
```

#### Fix Prompt Template 3: Fix Noncommutative Ideals
**Files:**
- `/home/user/RustMath/rustmath-rings/src/noncommutative_ideals.rs`
- `/home/user/RustMath/rustmath-rings/src/function_field_element_rational.rs`

Missing:
```
Ideal_nc (type definition and export)
IdealMonoid_nc (type definition and export)
```

---

## Medium Priority Errors (Type/Trait Issues)

### Group 2: E0308 - Type Mismatches (79 errors)
**Priority:** HIGH - Fix after foundation types
**Affected Files (25 files):** Mostly in rustmath-groups and rustmath-rings

Major clusters:
- **rustmath-groups/src/finitely_presented_named.rs** (17 mismatches)
- **rustmath-groups/src/free_group.rs** (6 mismatches)
- **rustmath-groups/src/braid.rs** (4 mismatches)
- **rustmath-rings/src/function_field/** directory (20+ mismatches)

#### Fix Prompt Template 4: Function Field Type Mismatches
Focus on these files with most errors:
- `/home/user/RustMath/rustmath-rings/src/function_field/function_field_polymod.rs` (5 errors)
- `/home/user/RustMath/rustmath-rings/src/function_field/ideal_rational.rs` (6 errors)
- `/home/user/RustMath/rustmath-rings/src/function_field/place_polymod.rs` (4 errors)

#### Fix Prompt Template 5: Groups Type Mismatches
Focus on:
- `/home/user/RustMath/rustmath-groups/src/finitely_presented_named.rs` (17 errors)
- `/home/user/RustMath/rustmath-groups/src/free_group.rs` (6 errors)

### Group 3: E0277 - Trait Bounds Not Satisfied (12 errors)
**Priority:** HIGH
**Key Files:**
- `/home/user/RustMath/rustmath-groups/src/additive_abelian_wrapper.rs` (6 errors)
  - Missing: `Add<T>`, `Mul<i64>`, `Default` trait implementations
- `/home/user/RustMath/rustmath-groups/src/indexed_free_group.rs` (2 errors)
  - Missing: `Hash`, `Display` implementations
- `/home/user/RustMath/rustmath-groups/src/group_exp.rs` (1 error)

#### Fix Prompt Template 6: Fix AdditiveAbelianGroupWrapper Traits
**File:** `/home/user/RustMath/rustmath-groups/src/additive_abelian_wrapper.rs`

Required implementations:
- `Add` trait for generic type `T`
- `Mul<i64>` trait for generic type `T`
- `Default` trait for generic type `T`
- `PartialEq` for the wrapper itself

---

### Group 4: E0599 - Missing Methods (5 errors)
**Priority:** MEDIUM
**Files:**
- `/home/user/RustMath/rustmath-groups/src/group_exp.rs` (2 errors)
  - Missing: `is_zero()`, `scalar_multiply()`
- `/home/user/RustMath/rustmath-rings/src/function_field_element_polymod.rs` (1 error)
  - Missing: `hash()` method
- `/home/user/RustMath/rustmath-groups/src/affine_group.rs` (1 error)
  - Matrix.inverse() trait bounds not satisfied

#### Fix Prompt Template 7: Fix AdditiveAbelianGroupElement Missing Methods
**File:** `/home/user/RustMath/rustmath-groups/src/group_exp.rs`

Add methods to `AdditiveAbelianGroupElement`:
- `is_zero() -> bool`
- `scalar_multiply(&self, scalar: i32) -> Self`

---

### Group 5: E0369 - Binary Operations Not Supported (3 errors)
**Priority:** MEDIUM
**Files:**
- `/home/user/RustMath/rustmath-groups/src/additive_abelian_wrapper.rs`
  - Implement `PartialEq` for `AdditiveAbelianGroupWrapper<T>`
- `/home/user/RustMath/rustmath-groups/src/finitely_presented_named.rs`
  - Implement `Mul` for generic type `G`
- `/home/user/RustMath/rustmath-groups/src/semidirect_product.rs`
  - Implement `Mul` for generic type `H`

---

## Low Priority Errors (Edge Cases)

### Group 6: E0573/E0574 - Function vs Type (6 errors)
**Priority:** LOW
**File:** `/home/user/RustMath/rustmath-rings/src/real_lazy.rs`

Issue: `RealLazyField` and `ComplexLazyField` are functions but used as types

Solution: Either:
1. Convert to struct types with factory functions
2. Create type aliases
3. Rename to factory function pattern

---

### Group 7: E0107 - Wrong Generic Argument Count (1 error)
**Priority:** LOW
**File:** `/home/user/RustMath/rustmath-plot-core/src/bbox.rs`

Issue: Struct takes 3 generic arguments but 1 provided

---

### Group 8: E0061 - Wrong Function Argument Count (1 error)
**Priority:** LOW
**File:** `/home/user/RustMath/rustmath-groups/src/group_exp.rs`

Issue: Function called with 1 argument but takes 2

---

## Error Summary Matrix

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║ ERROR CODE │ COUNT │ PRIORITY │ PRIMARY ISSUE                                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ E0432      │  11   │ CRITICAL │ Unresolved imports (missing module exports)  ║
║ E0433      │  22   │ CRITICAL │ Undeclared types (not defined)                ║
║ E0412      │  75   │ CRITICAL │ Type not found in scope (same as above)       ║
║ E0422      │  29   │ CRITICAL │ Struct/variant not found (same as above)      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ E0308      │  79   │ HIGH     │ Type mismatches in assignments/calls          ║
║ E0277      │  12   │ HIGH     │ Missing trait implementations                 ║
║ E0599      │   5   │ MEDIUM   │ Missing methods on types                      ║
║ E0369      │   3   │ MEDIUM   │ Operator traits not implemented               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ E0573      │   6   │ LOW      │ Function used where type expected              ║
║ E0574      │   2   │ LOW      │ Function used where struct expected            ║
║ E0107      │   1   │ LOW      │ Wrong generic argument count                   ║
║ E0061      │   1   │ LOW      │ Wrong function argument count                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## Parallelizable Fix Groups

These groups can be fixed in parallel (mostly independent):

### Parallel Set 1: Module Exports (can start immediately)
```
1. Fix rustmath-rings/src/function_field/mod.rs exports
2. Fix rustmath-rings/src/function_field/drinfeld_modules/mod.rs exports
3. Fix rustmath-rings/src/noncommutative_ideals.rs definitions
4. Fix rustmath-rings/src/real_lazy.rs (RealLazyField / ComplexLazyField)
```

### Parallel Set 2: Groups Fixes (depends on Parallel Set 1)
```
1. Fix rustmath-groups/src/additive_abelian_wrapper.rs traits and methods
2. Fix rustmath-groups/src/finitely_presented_named.rs type mismatches
3. Fix rustmath-groups/src/free_group.rs type mismatches
4. Fix rustmath-groups/src/group_exp.rs missing methods
5. Fix rustmath-groups/src/braid.rs type mismatches
```

### Parallel Set 3: Function Field Fixes (depends on Parallel Set 1)
```
1. Fix rustmath-rings/src/function_field/function_field_polymod.rs
2. Fix rustmath-rings/src/function_field/ideal_rational.rs
3. Fix rustmath-rings/src/function_field/place_polymod.rs
4. Fix rustmath-rings/src/function_field/place_rational.rs
5. Fix rustmath-rings/src/function_field_element_polymod.rs
6. Fix rustmath-rings/src/function_field_element_rational.rs
```

### Parallel Set 4: Edge Cases (can be parallel or sequential)
```
1. Fix rustmath-plot-core/src/bbox.rs generic arguments
2. Fix rustmath-liealgebras/src/weight_lattice.rs type mismatches
3. Fix remaining affine_group.rs and semidirect_product.rs issues
```

---

## File-to-File Dependency Graph

```
FOUNDATION LAYER (Fix first):
├── rustmath-rings/src/function_field/mod.rs (needs exports)
├── rustmath-rings/src/function_field/drinfeld_modules/mod.rs (needs exports)
├── rustmath-rings/src/noncommutative_ideals.rs (needs Ideal_nc definition)
└── rustmath-rings/src/real_lazy.rs (needs type definition)

SECONDARY LAYER (Fix after foundation):
├── rustmath-rings/src/function_field/*.rs (all files - use exported types)
├── rustmath-rings/src/function_field_element_*.rs (use exported types)
└── rustmath-groups/src/additive_abelian_wrapper.rs (add trait impls)

TERTIARY LAYER (Fix after secondary):
├── rustmath-groups/src/free_group.rs
├── rustmath-groups/src/finitely_presented*.rs
├── rustmath-groups/src/braid.rs
├── rustmath-groups/src/group_exp.rs
└── rustmath-groups/src/affine_group.rs
```

