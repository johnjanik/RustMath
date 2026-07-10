# Rust Build Error Analysis - Complete Categorization

**Build Date:** 2025-11-21
**Total Errors:** 246
**Total Error Types:** 12 unique error codes

---

## Quick Summary Table

| Error Code | Count | Primary Issue | Affected Crates |
|----------|-------|---|---|
| E0308 | 79 | Mismatched types / Incorrect arguments | rustmath-groups, rustmath-rings, rustmath-liealgebras |
| E0412 | 75 | Cannot find type in scope | rustmath-rings, rustmath-groups |
| E0422 | 29 | Cannot find struct/variant/union type | rustmath-rings, rustmath-groups |
| E0433 | 22 | Failed to resolve undeclared type | rustmath-rings |
| E0277 | 12 | Trait bound not satisfied | rustmath-groups, rustmath-rings |
| E0432 | 11 | Unresolved imports | rustmath-groups, rustmath-rings |
| E0573 | 6 | Expected type, found function | rustmath-rings |
| E0599 | 5 | Missing method | rustmath-groups, rustmath-rings |
| E0369 | 3 | Binary operation not supported | rustmath-groups |
| E0574 | 2 | Expected struct, found function | rustmath-rings |
| E0107 | 1 | Wrong number of generic arguments | rustmath-plot-core |
| E0061 | 1 | Wrong number of function arguments | rustmath-groups |

---

## Error Category 1: E0432 - Unresolved Imports (11 errors)

**Description:** Code is trying to import items that don't exist or aren't exported.

### Affected Files and Missing Imports

#### 1. `/home/user/RustMath/rustmath-rings/src/function_field/mod.rs` (8 unresolved imports)
```rust
// Missing imports:
- function_field_polymod::FunctionField_polymod
- function_field_polymod::FunctionField_simple
- function_field_polymod::FunctionField_char_zero
- function_field_polymod::FunctionField_integral
- function_field_polymod::FunctionField_char_zero_integral
- function_field_polymod::FunctionField_global
- function_field_polymod::FunctionField_global_integral
- function_field_rational::RationalFunctionField_char_zero
- function_field_rational::RationalFunctionField_global
- ideal_rational::FunctionFieldIdeal_rational
- ideal_rational::FunctionFieldIdealInfinite_rational
- order_basis::FunctionFieldOrder_basis
- order_basis::FunctionFieldOrderInfinite_basis
- order_rational::FunctionFieldMaximalOrder_rational
- order_rational::FunctionFieldMaximalOrderInfinite_rational
- valuation::FunctionFieldValuation_base
- valuation::ClassicalFunctionFieldValuation_base
- valuation::RationalFunctionFieldValuation_base
- maps::FunctionFieldMorphism_polymod
- maps::FunctionFieldMorphism_rational
- place_rational::FunctionFieldPlace_rational
```

#### 2. `/home/user/RustMath/rustmath-rings/src/function_field/drinfeld_modules/mod.rs` (2 unresolved imports)
```rust
- finite_drinfeld_module::DrinfeldModule_finite
- place_polymod::FunctionFieldPlace_polymod
```

#### 3. `/home/user/RustMath/rustmath-groups/src/indexed_free_group.rs` (1 unresolved import)
```rust
- charzero_drinfeld_module::DrinfeldModule_charzero
- charzero_drinfeld_module::DrinfeldModule_rational
```

**Root Cause:** These module items are either not defined or not properly exported from their respective modules.

---

## Error Category 2: E0308 - Type Mismatches (79 errors)

**Description:** Function arguments don't match parameter types, or assigned values don't match expected types.

### Breakdown by Sub-Type

#### 2A. "Mismatched types" - 76 occurrences

**Affected Files (25 files):**
- `/home/user/RustMath/rustmath-groups/src/argument_groups.rs` (3x)
- `/home/user/RustMath/rustmath-groups/src/artin.rs` (2x)
- `/home/user/RustMath/rustmath-groups/src/braid.rs` (4x)
- `/home/user/RustMath/rustmath-groups/src/class_function.rs` (3x)
- `/home/user/RustMath/rustmath-groups/src/cubic_braid.rs` (2x)
- `/home/user/RustMath/rustmath-groups/src/finitely_presented.rs` (3x)
- `/home/user/RustMath/rustmath-groups/src/finitely_presented_named.rs` (6x)
- `/home/user/RustMath/rustmath-groups/src/free_group.rs` (6x)
- `/home/user/RustMath/rustmath-groups/src/additive_abelian_wrapper.rs` (1x)
- `/home/user/RustMath/rustmath-groups/src/affine_group.rs` (1x)
- `/home/user/RustMath/rustmath-groups/src/semidirect_product.rs` (1x)
- `/home/user/RustMath/rustmath-groups/src/group_exp.rs` (1x)
- `/home/user/RustMath/rustmath-liealgebras/src/weight_lattice.rs` (1x)
- `/home/user/RustMath/rustmath-rings/src/function_field/drinfeld_modules/charzero_drinfeld_module.rs` (3x)
- `/home/user/RustMath/rustmath-rings/src/function_field/function_field_polymod.rs` (5x)
- `/home/user/RustMath/rustmath-rings/src/function_field/function_field_rational.rs` (2x)
- `/home/user/RustMath/rustmath-rings/src/function_field/ideal_rational.rs` (6x)
- `/home/user/RustMath/rustmath-rings/src/function_field/order_rational.rs` (1x)
- `/home/user/RustMath/rustmath-rings/src/function_field/place_polymod.rs` (4x)
- `/home/user/RustMath/rustmath-rings/src/function_field/place_rational.rs` (3x)
- `/home/user/RustMath/rustmath-rings/src/function_field/valuation.rs` (1x)
- `/home/user/RustMath/rustmath-rings/src/function_field_element_polymod.rs` (3x)
- `/home/user/RustMath/rustmath-rings/src/function_field_element_rational.rs` (2x)
- `/home/user/RustMath/rustmath-rings/src/noncommutative_ideals.rs` (1x)
- `/home/user/RustMath/rustmath-rings/src/real_lazy.rs` (1x)

#### 2B. "Arguments to this function are incorrect" - 3 occurrences

**Affected Files (3 files):**
- `/home/user/RustMath/rustmath-groups/src/argument_groups.rs`
- `/home/user/RustMath/rustmath-groups/src/class_function.rs`
- `/home/user/RustMath/rustmath-groups/src/finitely_presented_named.rs` (11x)

**Root Cause:** Function calls don't provide correct number or types of arguments. Often caused by renamed functions or changed signatures.

---

## Error Category 3: E0412 - Cannot Find Type in Scope (75 errors)

**Description:** A type name is used but not found or imported in the current scope.

### Breakdown by Missing Type

| Missing Type | Count | Affected Files |
|---|---|---|
| `FunctionFieldElement_polymod` | 8 | rustmath-rings/src/function_field_element_polymod.rs (5x), rustmath-groups/src/group_exp.rs (2x), rustmath-groups/src/indexed_free_group.rs |
| `FunctionFieldElement_rational` | 4 | rustmath-rings/src/function_field_element_rational.rs (3x), rustmath-rings/src/function_field_element_polymod.rs |
| `RationalFunctionFieldValuation_base` | 8 | rustmath-rings/src/function_field/valuation.rs (7x), rustmath-rings/src/function_field/place_rational.rs |
| `FunctionFieldValuation_base` | 9 | rustmath-rings/src/function_field/valuation.rs (7x), rustmath-rings/src/function_field/place_rational.rs, rustmath-rings/src/function_field/function_field_polymod.rs |
| `FunctionField_simple` | 5 | rustmath-rings/src/function_field/function_field_polymod.rs (4x), rustmath-rings/src/function_field/valuation.rs |
| `FunctionField_integral` | 4 | rustmath-rings/src/function_field/function_field_polymod.rs (3x), rustmath-rings/src/function_field/function_field_rational.rs |
| `FunctionField_global` | 3 | rustmath-rings/src/function_field/function_field_polymod.rs (2x), rustmath-rings/src/function_field/function_field_rational.rs |
| `FunctionField_char_zero` | 3 | rustmath-rings/src/function_field/function_field_polymod.rs (2x), rustmath-rings/src/function_field/function_field_rational.rs |
| `FunctionField_char_zero_integral` | 1 | rustmath-rings/src/function_field/function_field_polymod.rs |
| `FunctionField_polymod` | 2 | rustmath-rings/src/function_field/function_field_polymod.rs, rustmath-rings/src/function_field/valuation.rs |
| `FunctionField_global_integral` | 1 | rustmath-rings/src/function_field/function_field_polymod.rs |
| `DrinfeldModule_charzero` | 4 | rustmath-rings/src/function_field/drinfeld_modules/charzero_drinfeld_module.rs (3x), rustmath-rings/src/noncommutative_ideals.rs |
| `DrinfeldModule_rational` | 3 | rustmath-rings/src/function_field/drinfeld_modules/charzero_drinfeld_module.rs (3x) |
| `DrinfeldModule_finite` | 2 | rustmath-rings/src/function_field/drinfeld_modules/charzero_drinfeld_module.rs, rustmath-rings/src/function_field/drinfeld_modules/finite_drinfeld_module.rs |
| `FunctionFieldPlace_polymod` | 2 | rustmath-rings/src/function_field/drinfeld_modules/finite_drinfeld_module.rs, rustmath-rings/src/function_field/place_polymod.rs |
| `FunctionFieldPlace_rational` | 2 | rustmath-rings/src/function_field/place_polymod.rs, rustmath-rings/src/function_field/place_rational.rs |
| `FunctionFieldIdeal_rational` | 1 | rustmath-rings/src/function_field/function_field_rational.rs |
| `FunctionFieldIdealInfinite_rational` | 2 | rustmath-rings/src/function_field/ideal_rational.rs (2x) |
| `FunctionFieldMaximalOrder_rational` | 1 | rustmath-rings/src/function_field/order_basis.rs |
| `FunctionFieldMaximalOrderInfinite_rational` | 1 | rustmath-rings/src/function_field/order_rational.rs |
| `FunctionFieldOrder_basis` | 1 | rustmath-rings/src/function_field/maps.rs |
| `FunctionFieldOrderInfinite_basis` | 1 | rustmath-rings/src/function_field/order_basis.rs |
| `FunctionFieldMorphism_polymod` | 1 | rustmath-rings/src/function_field/ideal_rational.rs |
| `FunctionFieldMorphism_rational` | 1 | rustmath-rings/src/function_field/maps.rs |
| `ClassicalFunctionFieldValuation_base` | 3 | rustmath-rings/src/function_field/valuation.rs (3x) |
| `Ideal_nc` | 4 | rustmath-rings/src/noncommutative_ideals.rs (4x) |
| `IdealMonoid_nc` | 2 | rustmath-rings/src/function_field_element_rational.rs, rustmath-rings/src/noncommutative_ideals.rs |

**Root Cause:** Types are referenced but either not defined or not in scope. Often related to:
- Unexported types from modules
- Typos in type names
- Missing module imports
- Types in unfinished modules

---

## Error Category 4: E0422 - Cannot Find Struct/Variant/Union Type (29 errors)

**Description:** Similar to E0412, but specifically for struct, variant, or union types.

### Breakdown by Missing Type

| Missing Type | Count | Affected Files |
|---|---|---|
| `FunctionFieldElement_polymod` | 8 | rustmath-rings/src/function_field_element_polymod.rs (6x), rustmath-groups/src/group_exp.rs |
| `FunctionFieldElement_rational` | 7 | rustmath-rings/src/function_field_element_rational.rs (5x), rustmath-groups/src/indexed_free_group.rs |
| `DrinfeldModule_charzero` | 1 | rustmath-rings/src/function_field/drinfeld_modules/charzero_drinfeld_module.rs |
| `DrinfeldModule_finite` | 2 | rustmath-rings/src/function_field/drinfeld_modules/finite_drinfeld_module.rs (2x) |
| `DrinfeldModule_rational` | 2 | rustmath-rings/src/function_field/drinfeld_modules/charzero_drinfeld_module.rs, rustmath-liealgebras/src/chevalley_basis.rs |
| `FunctionFieldPlace_polymod` | 1 | rustmath-rings/src/function_field/place_polymod.rs |
| `FunctionFieldPlace_rational` | 2 | rustmath-rings/src/function_field/place_rational.rs (2x) |
| `FunctionFieldValuation_base` | 1 | rustmath-rings/src/function_field/valuation.rs |
| `ClassicalFunctionFieldValuation_base` | 1 | rustmath-rings/src/function_field/valuation.rs |
| `RationalFunctionFieldValuation_base` | 2 | rustmath-rings/src/function_field/valuation.rs (2x) |
| `Ideal_nc` | 3 | rustmath-rings/src/noncommutative_ideals.rs (3x) |
| `IdealMonoid_nc` | 1 | rustmath-rings/src/noncommutative_ideals.rs |

**Root Cause:** Same as E0412 but compiler is specifically looking for these as struct/variant/union types.

---

## Error Category 5: E0433 - Failed to Resolve Undeclared Type (22 errors)

**Description:** Type is used in a context where compiler can't determine if it's a struct, enum, or other construct.

### Breakdown by Undeclared Type

| Undeclared Type | Count | Affected Files |
|---|---|---|
| `FunctionField_simple` | 3 | rustmath-rings/src/function_field/function_field_polymod.rs (3x) |
| `FunctionField_integral` | 2 | rustmath-rings/src/function_field/function_field_polymod.rs (2x) |
| `FunctionField_polymod` | 2 | rustmath-rings/src/function_field/function_field_polymod.rs (2x) |
| `FunctionField_char_zero` | 1 | rustmath-rings/src/function_field/function_field_polymod.rs |
| `FunctionField_global` | 1 | rustmath-rings/src/function_field/function_field_polymod.rs |
| `FunctionFieldValuation_base` | 3 | rustmath-rings/src/function_field/valuation.rs (3x) |
| `FunctionFieldValuation_base` | 1 | rustmath-rings/src/function_field/place_rational.rs |
| `RationalFunctionFieldValuation_base` | 2 | rustmath-rings/src/function_field/valuation.rs (2x) |
| `FunctionFieldPlace_rational` | 1 | rustmath-rings/src/function_field/place_polymod.rs |
| `DrinfeldModule_charzero` | 2 | rustmath-rings/src/function_field/drinfeld_modules/charzero_drinfeld_module.rs (2x) |
| `Ideal_nc` | 2 | rustmath-rings/src/noncommutative_ideals.rs, rustmath-rings/src/function_field_element_rational.rs |
| `RealLazyField` | 1 | rustmath-rings/src/function_field/order_rational.rs |
| `ComplexLazyField` | 1 | rustmath-rings/src/real_lazy.rs |

**Root Cause:** These types are used as type constructors but are not properly declared.

---

## Error Category 6: E0277 - Trait Bound Not Satisfied (12 errors)

**Description:** A generic type doesn't have the required trait implementations.

### Breakdown by Missing Trait

| Trait Issue | Count | Affected Files |
|---|---|---|
| Cannot add `T` to `T` | 2 | rustmath-groups/src/additive_abelian_wrapper.rs, rustmath-rings/src/function_field/mod.rs |
| Cannot multiply `T` by `i64` | 2 | rustmath-groups/src/additive_abelian_wrapper.rs (2x) |
| `T: Default` not satisfied | 2 | rustmath-groups/src/additive_abelian_wrapper.rs (2x) |
| `I` doesn't implement `Display` | 2 | rustmath-groups/src/cactus_group.rs, rustmath-groups/src/indexed_free_group.rs |
| `HashMap<I, i32>: Hash` not satisfied | 1 | rustmath-groups/src/indexed_free_group.rs |
| `IndexedFreeGroup<I>: Hash` not satisfied | 1 | rustmath-rings/src/function_field_element_polymod.rs |
| `IndexedFreeAbelianGroup<I>: Hash` not satisfied | 1 | rustmath-rings/src/function_field_element_rational.rs |
| `?` couldn't convert error to `String` | 1 | rustmath-groups/src/affine_group.rs |

**Root Cause:** Generic parameters need to derive or implement certain traits. Common fixes:
- Add `#[derive(Hash, Eq, PartialEq)]` or similar
- Add trait bounds to generic parameters
- Implement missing trait methods

---

## Error Category 7: E0599 - Missing Method (5 errors)

**Description:** A method is called on a struct but doesn't exist or trait bounds aren't satisfied.

| Method | Struct | File | Issue |
|---|---|---|---|
| `inverse` | `Matrix<R>` | rustmath-groups/src/affine_group.rs | Trait bounds not satisfied |
| `is_zero` | `AdditiveAbelianGroupElement` | rustmath-groups/src/group_exp.rs | Method doesn't exist |
| `scalar_multiply` | `AdditiveAbelianGroupElement` | rustmath-groups/src/group_exp.rs | Method doesn't exist |
| `zero` | `Result<T, E>` | rustmath-rings/src/function_field_element_polymod.rs | Calling on wrong type |
| `hash` | `AdditiveAbelianGroupElement` | rustmath-rings/src/function_field_element_polymod.rs | Method doesn't exist |

**Root Cause:**
- Methods were removed or renamed
- Struct doesn't implement required trait
- Method is being called on wrong type

---

## Error Category 8: E0369 - Binary Operation Not Supported (3 errors)

**Description:** Binary operations like `+`, `*`, `!=` aren't implemented for the given types.

| Operation | Type | File | Count |
|---|---|---|---|
| `!=` | `AdditiveAbelianGroupWrapper<T>` | rustmath-groups/src/additive_abelian_wrapper.rs | 1 |
| Cannot multiply `G` by `G` | Generic `G` | rustmath-groups/src/finitely_presented_named.rs | 1 |
| Cannot multiply `H` by `H` | Generic `H` | rustmath-groups/src/semidirect_product.rs | 1 |

**Root Cause:** Types need to implement `PartialEq`, `Mul`, or other operator traits.

---

## Error Category 9: E0573 - Expected Type, Found Function (6 errors)

**Description:** A function is used where a type is expected (usually in type annotations).

| Function | File | Count | Context |
|---|---|---|---|
| `RealLazyField` | rustmath-rings/src/real_lazy.rs | 2 | Expected struct/type annotation |
| `ComplexLazyField` | rustmath-rings/src/real_lazy.rs | 3 | Expected struct/type annotation |
| `RealLazyField` | rustmath-rings/src/function_field/order_rational.rs | 1 | Expected struct/type annotation |

**Root Cause:**
- `RealLazyField` and `ComplexLazyField` are likely functions, not types
- Should be renamed to PascalCase type names or wrapped in a struct

---

## Error Category 10: E0574 - Expected Struct, Found Function (2 errors)

**Description:** A function is used where a struct type is expected in pattern matching or type construction.

| Function | File | Context |
|---|---|---|
| `RealLazyField` | rustmath-rings/src/real_lazy.rs | Pattern matching or construction |
| `ComplexLazyField` | rustmath-rings/src/real_lazy.rs | Pattern matching or construction |

**Root Cause:** Same as E0573 - these are functions but used as type constructors.

---

## Error Category 11: E0107 - Wrong Number of Generic Arguments (1 error)

**Description:** A struct is given wrong number of generic parameters.

| File | Issue |
|---|---|
| `/home/user/RustMath/rustmath-plot-core/src/bbox.rs` | Struct takes 3 generic arguments but 1 was supplied |

**Root Cause:** Recent change to generic parameters of a struct.

---

## Error Category 12: E0061 - Wrong Number of Function Arguments (1 error)

**Description:** Function call has wrong number of arguments.

| File | Issue |
|---|---|
| `/home/user/RustMath/rustmath-groups/src/group_exp.rs` | Function takes 2 arguments but 1 was supplied |

**Root Cause:** Function signature was changed or function was renamed.

---

## Summary by Affected Files (Top Issues)

### Most Problematic Files (by error count)

1. **rustmath-rings/src/function_field_element_polymod.rs** - 21 errors
   - E0412 (5x), E0422 (6x), E0599 (1x), E0277 (1x), E0308 (3x), E0573, E0574

2. **rustmath-rings/src/function_field/valuation.rs** - 20 errors
   - E0412 (9x), E0422 (1x), E0433 (4x), E0277 (1x), E0308 (1x), E0412 (7x)

3. **rustmath-rings/src/function_field/function_field_polymod.rs** - 19 errors
   - E0412 (7x), E0433 (8x), E0308 (5x)

4. **rustmath-groups/src/finitely_presented_named.rs** - 19 errors
   - E0308 (17x)

5. **rustmath-rings/src/function_field_element_rational.rs** - 16 errors
   - E0422 (5x), E0412 (3x), E0433 (1x), E0277 (2x), E0308 (2x), E0433 (1x)

6. **rustmath-groups/src/free_group.rs** - 15 errors
   - E0308 (6x), E0308 (6x)

7. **rustmath-rings/src/function_field/ideal_rational.rs** - 14 errors
   - E0308 (6x), E0412 (2x), E0412 (1x), E0412 (1x), E0308 (2x)

8. **rustmath-groups/src/finitely_presented.rs** - 14 errors
   - E0308 (3x), E0308 (8x)

9. **rustmath-rings/src/function_field/place_polymod.rs** - 13 errors
   - E0308 (4x), E0412 (1x), E0412 (1x), E0433 (1x), E0308 (4x)

10. **rustmath-groups/src/braid.rs** - 12 errors
    - E0308 (4x), E0308 (4x)

---

## Recommended Fix Strategy

### Phase 1: Foundation Types (E0412, E0422, E0432, E0433)
These errors prevent compilation of dependent code. Fix in order:
1. **E0432 Unresolved Imports** - Export missing types from modules
2. **E0433 Undeclared Types** - Define missing types (likely in function_field modules)
3. **E0412/E0422 Type Not Found** - These should resolve after #1 and #2

**Key files:**
- rustmath-rings/src/function_field/mod.rs
- rustmath-rings/src/function_field_element_*.rs
- rustmath-rings/src/noncommutative_ideals.rs

### Phase 2: Trait Issues (E0277, E0599, E0369)
After foundation types are fixed, address trait bounds:
1. **E0599 Missing Methods** - Check if methods were renamed or removed
2. **E0277 Trait Bounds** - Add required trait implementations
3. **E0369 Binary Operations** - Implement operator traits

**Key files:**
- rustmath-groups/src/additive_abelian_wrapper.rs
- rustmath-groups/src/group_exp.rs
- rustmath-groups/src/affine_group.rs

### Phase 3: Type Mismatches (E0308)
Once types are properly defined, fix the 79 type mismatch errors:
- Most are in groups and function field modules
- Usually require parameter type adjustments or return type changes

**Key files:**
- rustmath-groups/src/braid.rs
- rustmath-groups/src/free_group.rs
- rustmath-rings/src/function_field/place_*.rs

### Phase 4: Type Annotations (E0573, E0574)
Fix function vs type issues:
- Rename `RealLazyField` and `ComplexLazyField` or convert to types
- Update all usages

**Files:**
- rustmath-rings/src/real_lazy.rs

### Phase 5: Edge Cases (E0107, E0061, E0107)
Fix remaining edge cases:
- Generic argument counts
- Function argument counts

