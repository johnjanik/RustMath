# Compilation Error Examples and Root Causes

## E0432 Examples: Unresolved Imports

### Example 1: Missing Function Field Type Exports
**File:** `rustmath-rings/src/function_field/mod.rs`
```rust
// ERROR:
error[E0432]: unresolved imports `function_field_polymod::FunctionField_polymod`, ...
  --> rustmath-rings/src/function_field/mod.rs

// CAUSE: 
The mod.rs file is trying to re-export types from submodules that either:
1. Don't exist (not defined in those files)
2. Aren't public (defined but not `pub`)

// FIX:
1. Check that function_field_polymod.rs defines pub struct FunctionField_polymod
2. Add pub use statements in mod.rs:
   pub use self::function_field_polymod::FunctionField_polymod;
```

### Example 2: Missing Drinfeld Module Exports
**File:** `rustmath-rings/src/function_field/drinfeld_modules/mod.rs`
```rust
// ERROR:
error[E0432]: unresolved import `finite_drinfeld_module::DrinfeldModule_finite`

// CAUSE:
The submodule finite_drinfeld_module.rs either doesn't exist or doesn't define
DrinfeldModule_finite as a public type.

// FIX:
1. Verify finite_drinfeld_module.rs exists
2. Ensure it contains: pub struct DrinfeldModule_finite { ... }
3. Add pub use statement in mod.rs
```

---

## E0412 / E0422 Examples: Type Not Found in Scope

### Example 1: FunctionFieldElement_polymod
**File:** `rustmath-rings/src/function_field_element_polymod.rs`
```rust
// ERROR:
error[E0412]: cannot find type `FunctionFieldElement_polymod` in this scope
  --> rustmath-rings/src/function_field_element_polymod.rs:45:13

// CAUSE:
File is defining or using FunctionFieldElement_polymod but the type doesn't exist.
This is likely:
1. File is incomplete/under development
2. Type was deleted
3. File itself is not being compiled

// FIX:
Define the type in this file:
pub struct FunctionFieldElement_polymod {
    // fields...
}
```

### Example 2: RationalFunctionFieldValuation_base in valuation.rs
**File:** `rustmath-rings/src/function_field/valuation.rs`
```rust
// ERROR:
error[E0412]: cannot find type `RationalFunctionFieldValuation_base` in this scope
  --> rustmath-rings/src/function_field/valuation.rs:120:15

// CAUSE:
Multiple references to a base type that should be defined in valuation.rs
but isn't present.

// FIX:
Define the base type first, then the variant types:
pub struct RationalFunctionFieldValuation_base {
    // base fields
}

pub struct RationalFunctionFieldValuation<Base> {
    base: Base,
}
```

---

## E0308 Examples: Type Mismatches

### Example 1: Argument Type Mismatch
**Files:** `rustmath-groups/src/finitely_presented_named.rs`
```rust
// ERROR:
error[E0308]: mismatched types
  --> rustmath-groups/src/finitely_presented_named.rs:150:20
   |
150 |    let result = some_function(group_element);
    |                                ^^^^^^^^^^^^^ 
    |                                expected `&GroupElement`
    |                                found `GroupElement`

// CAUSE:
Function signature changed or implementation is passing wrong reference type.
Common in generic functions where:
- Function expects `&T` but gets `T`
- Function expects `T` but gets `&T`
- Type changed from one struct to another

// FIX:
Check function signature and either:
1. Pass reference: some_function(&group_element)
2. Dereference: some_function(*group_element)
3. Clone if needed: some_function(group_element.clone())
```

### Example 2: Return Type Mismatch
**File:** `rustmath-rings/src/function_field/function_field_polymod.rs`
```rust
// ERROR:
error[E0308]: mismatched types
  --> rustmath-rings/src/function_field/function_field_polymod.rs:75:9
   |
   |     return FunctionField_polymod { ... };
   |            ^^^^^^^^^^^^^^^^^^^ 
   |            expected different type

// CAUSE:
Function is supposed to return `FunctionField_char_zero` but code returns
`FunctionField_polymod`.

// FIX:
Either:
1. Change return type annotation to match actual return
2. Wrap/convert the return value to expected type
```

---

## E0277 Examples: Missing Trait Implementations

### Example 1: Missing Add Trait
**File:** `rustmath-groups/src/additive_abelian_wrapper.rs`
```rust
// ERROR:
error[E0277]: cannot add `T` to `T`
  --> rustmath-groups/src/additive_abelian_wrapper.rs:95:25
   |
   |     let sum = a + b;
   |               ^ cannot add
   |
   = note: the trait bound `T: Add` is not satisfied

// CAUSE:
Generic parameter T used in addition but doesn't implement Add trait.

// FIX:
Add trait bound to generic parameter:
impl<T: Add<Output = T> + Clone> AdditiveAbelianGroupWrapper<T> {
    fn add(&self, other: &Self) -> Self {
        let sum = self.value.clone() + other.value.clone();
        Self { value: sum }
    }
}
```

### Example 2: Missing Default Trait
**File:** `rustmath-groups/src/additive_abelian_wrapper.rs`
```rust
// ERROR:
error[E0277]: the trait bound `T: Default` is not satisfied
  --> rustmath-groups/src/additive_abelian_wrapper.rs:110:20
   |
   |     let zero = T::default();
   |                ^ Default is not implemented

// CAUSE:
Code assumes generic T has a default/zero value.

// FIX:
Either:
1. Add Default trait bound: impl<T: Default> ...
2. Pass zero explicitly as parameter
3. Define custom zero method
```

### Example 3: Missing Hash Trait
**File:** `rustmath-groups/src/indexed_free_group.rs`
```rust
// ERROR:
error[E0277]: the trait bound `HashMap<I, i32>: Hash` is not satisfied
  --> rustmath-groups/src/indexed_free_group.rs:45:15

// CAUSE:
HashMap<K, V> doesn't implement Hash. Only K must implement Hash.
Type is being used as a hash key when it shouldn't be.

// FIX:
Don't use HashMap directly as a hash key:
1. Wrap in a newtype
2. Use a different collection type
3. Or derive Hash for the struct containing it
```

---

## E0599 Examples: Missing Methods

### Example 1: Method Doesn't Exist
**File:** `rustmath-groups/src/group_exp.rs`
```rust
// ERROR:
error[E0599]: no method named `is_zero` found for struct `AdditiveAbelianGroupElement`
  --> rustmath-groups/src/group_exp.rs:120:25
   |
   |     if element.is_zero() {
   |                 ^^^^^^^ method not found

// CAUSE:
Method was removed, renamed, or never implemented on this type.

// FIX:
Either:
1. Implement the method:
   impl AdditiveAbelianGroupElement {
       fn is_zero(&self) -> bool {
           self.value == 0
       }
   }
2. Use different method if it was renamed
3. Call a function instead: is_zero(&element)
```

### Example 2: Trait Bounds Not Satisfied for Method Call
**File:** `rustmath-groups/src/affine_group.rs`
```rust
// ERROR:
error[E0599]: the method `inverse` exists for struct `Matrix<R>`, 
              but its trait bounds were not satisfied
  --> rustmath-groups/src/affine_group.rs:180:35

// CAUSE:
Method exists but requires specific trait bounds on generic type R.
For example: impl<R: Field> Matrix<R> { fn inverse(...) }
But R is not guaranteed to be a Field in the current context.

// FIX:
Add required trait bound to function:
impl<R: Field> AffineGroup<R> {  // Add trait bound here
    fn some_method(&self) -> Result<...> {
        self.matrix.inverse()  // Now this works
    }
}
```

---

## E0369 Examples: Missing Operator Traits

### Example 1: Missing PartialEq
**File:** `rustmath-groups/src/additive_abelian_wrapper.rs`
```rust
// ERROR:
error[E0369]: binary operation `!=` cannot be applied to type `AdditiveAbelianGroupWrapper<T>`
  --> rustmath-groups/src/additive_abelian_wrapper.rs:75:20
   |
   |     if wrapper1 != wrapper2 {
   |        ^^^^^^^ ^^^^^^^ AdditiveAbelianGroupWrapper<T>

// CAUSE:
Type doesn't derive or implement PartialEq trait needed for != operator.

// FIX:
Either:
1. Add derive: #[derive(PartialEq)]
2. Implement manually:
   impl<T: PartialEq> PartialEq for AdditiveAbelianGroupWrapper<T> {
       fn eq(&self, other: &Self) -> bool {
           self.value == other.value
       }
   }
```

### Example 2: Missing Mul for Generic Type
**File:** `rustmath-groups/src/finitely_presented_named.rs`
```rust
// ERROR:
error[E0369]: cannot multiply `G` by `G`
  --> rustmath-groups/src/finitely_presented_named.rs:95:25
   |
   |     let product = g1 * g2;
   |                   ^^ * element of type G

// CAUSE:
Generic type G doesn't implement multiplication.

// FIX:
Add trait bound:
impl<G: Mul<Output = G> + Clone> SomeType<G> {
    fn multiply(&self, a: G, b: G) -> G {
        a * b
    }
}
```

---

## E0573 / E0574 Examples: Function vs Type Confusion

### Example: Function Used as Type
**File:** `rustmath-rings/src/real_lazy.rs`
```rust
// ERROR:
error[E0573]: expected type, found function `RealLazyField`
  --> rustmath-rings/src/real_lazy.rs:45:25
   |
   |     let value: RealLazyField = compute_value();
   |               ^^^^^^^^^^^^^^ this is a function, not a type

// ERROR also shows:
error[E0574]: expected struct, variant or union type, found function
  --> rustmath-rings/src/real_lazy.rs:50:15

// CAUSE:
RealLazyField is defined as a function (factory method):
pub fn RealLazyField() -> LazyField { ... }

But code tries to use it as a type annotation.

// FIX 1: Create actual struct type:
pub struct RealLazyField {
    // fields...
}

impl RealLazyField {
    pub fn new() -> Self { ... }  // factory method
}

// FIX 2: Use factory correctly:
let value = RealLazyField();  // Call function, not type annotation

// FIX 3: Create type alias:
pub type RealLazyField = LazyField;
```

---

## E0107 Examples: Wrong Generic Argument Count

### Example:
**File:** `rustmath-plot-core/src/bbox.rs`
```rust
// ERROR:
error[E0107]: struct takes 3 generic arguments but 1 generic argument was supplied
  --> rustmath-plot-core/src/bbox.rs:220:15
   |
   |     let bbox: BoundingBox<f64> = ...
   |               ^^^^^^^^^^^^^^^ 
   |               expected 3 generic arguments

// CAUSE:
BoundingBox struct definition was changed to require 3 type parameters:
pub struct BoundingBox<T, U, V> { ... }

But code only provides 1: BoundingBox<f64>

// FIX:
Supply all 3 generic arguments:
let bbox: BoundingBox<f64, f64, f64> = ...

OR: Check if the struct definition should use fewer generics
```

---

## E0061 Examples: Wrong Function Argument Count

### Example:
**File:** `rustmath-groups/src/group_exp.rs`
```rust
// ERROR:
error[E0061]: this function takes 2 arguments but 1 argument was supplied
  --> rustmath-groups/src/group_exp.rs:130:15
   |
   |     let result = some_function(element);
   |                  ^^^^^^^^^^^^^ 
   |                  expected 2 arguments

// CAUSE:
Function signature changed. Was: fn some_function(x: T)
Now is: fn some_function(x: T, y: T)

// FIX:
Either:
1. Provide the second argument:
   let result = some_function(element, other_element);

2. Find overload with 1 argument if available:
   let result = some_function_single(element);

3. Update function signature if change was mistake
```

---

## Common Root Causes Summary

| Error Type | Root Cause | Typical Fix |
|---|---|---|
| E0432 | Missing pub use statements | Add pub use in mod.rs |
| E0412/E0422 | Type not defined or not public | Define type or make pub |
| E0308 | Type mismatch in arguments/returns | Adjust references, conversions |
| E0277 | Missing trait implementation | Add trait bounds or impl blocks |
| E0599 | Method doesn't exist or trait bounds | Add method or trait bound |
| E0369 | Operator trait missing | Derive or implement trait |
| E0573/E0574 | Function used as type | Create type wrapper for function |
| E0107 | Wrong generic argument count | Add/remove generic parameters |
| E0061 | Wrong function argument count | Add/remove function arguments |

