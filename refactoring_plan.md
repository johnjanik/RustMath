# RustMath Refactoring Plan

## Executive Summary

**Current Status**: 76 compilation errors across 2 crates
- rustmath-groups: 74 errors
- rustmath-liealgebras: 2 errors

**Root Causes Identified**:
1. Missing trait implementations (GroupElement)
2. Architectural mismatch in type system (Field vs Ring methods)
3. Missing method implementations on core types
4. Type mismatches from API evolution
5. Missing trait bounds and constraints

---

## Error Categorization

### By Error Code
- **E0753** (123): ✅ FIXED - Misplaced doc comments
- **E0599** (32): Method/function not found - **HIGH PRIORITY**
- **E0308** (44): Type mismatches - **MEDIUM PRIORITY**
- **E0277** (19): Unsatisfied trait bounds - **HIGH PRIORITY**
- **E0609** (7): ✅ FIXED - Field access errors
- **E0107** (7): ✅ FIXED - Missing generic parameters
- **E0369** (4): Binary operation not supported - **MEDIUM PRIORITY**
- **E0061** (3): Incorrect function arguments - **LOW PRIORITY**
- **E0038** (3): ✅ FIXED - Trait not dyn compatible

### By Root Cause

#### 1. GroupElement Trait Implementation Issues (7 types)
**Impact**: ~15-20 errors
**Affected Types**:
- ArtinGroupElement
- Braid
- CactusGroupElement
- CubicBraidElement
- GroupExpElement
- IndexedFreeGroupElement<I>
- IndexedFreeAbelianGroupElement<I>

**Problem**: These types are used as `Group::Element` but don't implement `GroupElement` trait.

#### 2. Field/Ring Method Confusion (15+ errors)
**Impact**: ~15 errors
**Problem**: Code calls `zero()`, `one()` as methods on type parameters `F: Field`, but these are likely associated functions, not methods.

**Examples**:
```rust
// Current (incorrect):
base_field.zero()  // Error: no method named `zero` found

// Should be:
F::zero() // or
<F as Field>::zero()
```

#### 3. Missing AdditiveAbelianGroup Methods (5 errors)
**Impact**: 5 errors
**Missing methods**:
- `zero()` - return identity element
- `generators()` - return group generators
- `contains()` - membership test

#### 4. Missing AdditiveAbelianGroupElement Methods (3 errors)
**Impact**: 3 errors
**Missing methods**:
- `is_zero()` - check if identity
- `scalar_multiply()` - scalar multiplication
- `hash()` - compute hash

#### 5. Missing IndexedFreeGroup Methods (2 errors)
**Impact**: 2 errors
**Missing methods**:
- `ngens()` - number of generators

#### 6. Missing PermutationGroup Methods (2 errors)
**Impact**: 2 errors
**Missing methods**:
- `symmetric(n)` - constructor for symmetric group

#### 7. Missing FreeGroupElement Methods (3 errors)
**Impact**: 3 errors
**Missing methods**:
- `letters()` - get constituent letters
- `exponent_sum()` - sum of exponents

#### 8. Type Mismatches (44 errors)
**Impact**: 44 errors
**Problem**: Function signatures changed but callers not updated
**Common patterns**:
- Wrong number of arguments
- Wrong parameter types
- Missing or extra type parameters

#### 9. Missing Trait Implementations
**Impact**: ~10 errors
**Missing traits**:
- `Eq` on `FinitelyPresentedGroup`
- `Default` on various additive group elements
- `Mul` for binary operations
- `Display` for various group elements

---

## Proposed Refactoring Strategy

### Phase 1: Core Trait Architecture (HIGH PRIORITY)

#### 1.1 Fix Field/Ring Method Calls
**Scope**: rustmath-schemes, rustmath-matrix
**Effort**: 2-3 hours
**Files**: ~10 files

**Changes**:
```rust
// Pattern 1: Type parameter with Field bound
impl<F: Field> SomeStruct<F> {
    fn method(&self) {
        // OLD: self.field_value.zero()
        // NEW: F::zero()
        let zero = F::zero();
    }
}

// Pattern 2: Use Zero/One traits from num_traits
use num_traits::{Zero, One};

impl<F: Field + Zero + One> SomeStruct<F> {
    fn method(&self) {
        let zero = F::zero();  // Now available as associated function
        let one = F::one();
    }
}
```

**Action Items**:
- Add `num_traits` dependency to affected crates
- Replace all `field.zero()` with `F::zero()`
- Replace all `field.one()` with `F::one()`
- Add `where F: Zero + One` bounds where needed

#### 1.2 Implement GroupElement for All Group Element Types
**Scope**: rustmath-groups
**Effort**: 4-5 hours
**Files**: 7 files

**Template**:
```rust
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArtinGroupElement {
    parent: ArtinGroup,
    word: FreeGroupElement,
}

impl fmt::Display for ArtinGroupElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.word)
    }
}

impl GroupElement for ArtinGroupElement {
    fn identity() -> Self {
        // Implementation
    }

    fn inverse(&self) -> Self {
        // Implementation
    }

    fn op(&self, other: &Self) -> Self {
        // Implementation
    }
}
```

**Files to modify**:
1. `rustmath-groups/src/artin.rs` - ArtinGroupElement
2. `rustmath-groups/src/braid.rs` - Braid
3. `rustmath-groups/src/cactus_group.rs` - CactusGroupElement
4. `rustmath-groups/src/cubic_braid.rs` - CubicBraidElement
5. `rustmath-groups/src/group_exp.rs` - GroupExpElement
6. `rustmath-groups/src/indexed_free_group.rs` - IndexedFreeGroupElement
7. `rustmath-groups/src/indexed_free_abelian_group.rs` - IndexedFreeAbelianGroupElement

### Phase 2: Add Missing Methods (MEDIUM PRIORITY)

#### 2.1 AdditiveAbelianGroup Methods
**File**: `rustmath-groups/src/additive_abelian_group.rs`
**Effort**: 1 hour

```rust
impl AdditiveAbelianGroup {
    pub fn zero(&self) -> AdditiveAbelianGroupElement {
        AdditiveAbelianGroupElement {
            parent: self.clone(),
            coordinates: vec![0; self.rank()],
        }
    }

    pub fn generators(&self) -> Vec<AdditiveAbelianGroupElement> {
        // Return standard basis elements
    }

    pub fn contains(&self, element: &AdditiveAbelianGroupElement) -> bool {
        // Check if element is in this group
        element.parent == *self
    }
}
```

#### 2.2 AdditiveAbelianGroupElement Methods
**File**: `rustmath-groups/src/additive_abelian_group.rs`
**Effort**: 1 hour

```rust
impl AdditiveAbelianGroupElement {
    pub fn is_zero(&self) -> bool {
        self.coordinates.iter().all(|&x| x == 0)
    }

    pub fn scalar_multiply(&self, n: i64) -> Self {
        Self {
            parent: self.parent.clone(),
            coordinates: self.coordinates.iter().map(|&x| x * n).collect(),
        }
    }
}

// Implement Hash if needed
impl Hash for AdditiveAbelianGroupElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coordinates.hash(state);
    }
}
```

#### 2.3 IndexedFreeGroup Methods
**File**: `rustmath-groups/src/indexed_free_group.rs`
**Effort**: 30 minutes

```rust
impl<I> IndexedFreeGroup<I> {
    pub fn ngens(&self) -> usize {
        self.generators.len()
    }
}
```

#### 2.4 PermutationGroup Methods
**File**: `rustmath-groups/src/permutation_group.rs`
**Effort**: 1 hour

```rust
impl PermutationGroup {
    pub fn symmetric(n: usize) -> Self {
        // Construct S_n
    }

    pub fn cyclic(n: usize) -> Self {
        // Construct cyclic group C_n
    }
}
```

#### 2.5 FreeGroupElement Methods
**File**: `rustmath-groups/src/free_group.rs`
**Effort**: 1 hour

```rust
impl FreeGroupElement {
    pub fn letters(&self) -> &[(i32, i32)] {
        &self.word
    }

    pub fn exponent_sum(&self, generator: i32) -> i32 {
        self.word.iter()
            .filter(|(g, _)| *g == generator)
            .map(|(_, e)| e)
            .sum()
    }
}
```

#### 2.6 Character Methods
**File**: `rustmath-groups/src/character.rs`
**Effort**: 30 minutes

```rust
impl Character {
    pub fn values(&self) -> &HashMap<GroupElement, Complex> {
        &self.values
    }
}
```

### Phase 3: Fix Type Mismatches (MEDIUM PRIORITY)

#### 3.1 Function Signature Audits
**Effort**: 2-3 hours
**Approach**: Systematic review of E0308 errors

**Common patterns**:
1. Functions expecting 2 args but receiving 1
2. Type parameter mismatches
3. Wrong concrete types

**Action**: For each E0308 error:
1. Check function signature
2. Check call site
3. Determine which should change
4. Update consistently

#### 3.2 Matrix Method Fixes
**File**: `rustmath-matrix/src/lib.rs`
**Effort**: 1-2 hours

```rust
impl<R: Ring> Matrix<R> {
    // Add missing methods:
    pub fn zero(rows: usize, cols: usize) -> Self {
        // Return zero matrix
    }

    pub fn mul(&self, other: &Matrix<R>) -> Result<Matrix<R>> {
        // Matrix multiplication (if not already impl via Mul trait)
    }
}
```

### Phase 4: Add Missing Trait Implementations (LOW PRIORITY)

#### 4.1 Eq for FinitelyPresentedGroup
**File**: `rustmath-groups/src/finitely_presented.rs`

```rust
impl PartialEq for FinitelyPresentedGroup {
    fn eq(&self, other: &Self) -> bool {
        // Implement structural equality
    }
}

impl Eq for FinitelyPresentedGroup {}
```

#### 4.2 Default for Group Elements
**Files**: Various in rustmath-groups

```rust
impl Default for SomeGroupElement {
    fn default() -> Self {
        Self::identity()
    }
}
```

#### 4.3 Mul traits for Binary Operations
**Files**: Various

```rust
use std::ops::Mul;

impl<G> Mul<G> for G where G: GroupElement {
    type Output = G;

    fn mul(self, rhs: G) -> G {
        self.op(&rhs)
    }
}
```

---

## Implementation Plan - Parallel Execution

### Prerequisites
1. Create feature branch: `refactor/fix-remaining-errors`
2. Ensure all tests are documented
3. Backup current state

### Parallel Workstreams

#### Stream 1: Field/Ring Method Refactoring
**Owner**: Could be automated with regex
**Duration**: 2-3 hours
**Files**: ~10-15 files across rustmath-schemes, rustmath-matrix

**Script**:
```bash
# Find all instances
rg "\.zero\(\)" --type rust
rg "\.one\(\)" --type rust

# Pattern: field_var.zero() -> F::zero()
# Pattern: field_var.one() -> F::one()
```

#### Stream 2: GroupElement Implementations
**Owner**: Requires understanding group theory
**Duration**: 4-5 hours
**Files**: 7 files in rustmath-groups

**Order**:
1. Start with simplest: CubicBraidElement, GroupExpElement
2. Then: ArtinGroupElement, Braid, CactusGroupElement
3. Finally: IndexedFreeGroupElement (generic), IndexedFreeAbelianGroupElement (generic)

#### Stream 3: Add Missing Methods
**Owner**: Can be done independently
**Duration**: 4-5 hours
**Files**: 6 files

**Order** (by dependency):
1. AdditiveAbelianGroup + AdditiveAbelianGroupElement (foundational)
2. FreeGroupElement (used by many)
3. IndexedFreeGroup
4. PermutationGroup
5. Character

#### Stream 4: Fix Type Mismatches
**Owner**: Requires careful analysis
**Duration**: 3-4 hours
**Approach**: Case-by-case review

---

## Testing Strategy

### Unit Tests
- Add tests for each new method
- Verify GroupElement implementations
- Test edge cases

### Integration Tests
- Run `cargo test` for each crate after changes
- Verify no regressions

### Build Verification
- `cargo build` after each phase
- Track error count reduction

---

## Success Metrics

### Phase 1 Complete
- ✅ All Field/Ring method calls fixed
- ✅ All 7 GroupElement traits implemented
- ✅ Error count < 40

### Phase 2 Complete
- ✅ All missing methods implemented
- ✅ Error count < 20

### Phase 3 Complete
- ✅ All type mismatches resolved
- ✅ Error count < 5

### Phase 4 Complete
- ✅ All trait implementations added
- ✅ **Zero compilation errors**

---

## Risk Assessment

### High Risk
- **GroupElement implementations**: Complex group theory, easy to get wrong
  - Mitigation: Start with simpler groups, add comprehensive tests

### Medium Risk
- **Field/Ring refactoring**: Large surface area, many files
  - Mitigation: Automated search/replace, careful review

### Low Risk
- **Adding methods**: Straightforward implementations
  - Mitigation: Good documentation, unit tests

---

## Rollback Plan

1. Each phase committed separately
2. Git tags at phase boundaries
3. Can revert individual phases if needed
4. Keep detailed changelog

---

## Long-Term Architectural Improvements

### 1. Trait Coherence
- Define clear trait hierarchy
- Ensure Ring/Field have consistent method access
- Consider blanket implementations

### 2. Type Safety
- Use newtypes to distinguish mathematical objects
- Add marker traits for mathematical properties
- Leverage Rust's type system more

### 3. API Consistency
- Standardize method naming
- Consistent use of `new()` constructors
- Uniform error handling

### 4. Documentation
- Add module-level docs explaining architecture
- Document trait relationships
- Add "how to implement" guides

---

## Estimated Total Effort

- **Phase 1**: 6-8 hours (parallelizable)
- **Phase 2**: 4-5 hours (partially parallelizable)
- **Phase 3**: 3-4 hours (sequential)
- **Phase 4**: 2-3 hours (parallelizable)

**Total**: 15-20 hours of focused development time
**With parallelization**: Could complete in 2-3 full work days

---

## One-Liner Prompts for Parallel Execution

See section below for specific prompts.

---

## ONE-LINER PROMPTS FOR PARALLEL REFACTORING

### Stream 1A: Fix Field Method Calls in rustmath-schemes
```
In rustmath-schemes, replace all instances of 'field_value.zero()' and 'field_value.one()' with 'F::zero()' and 'F::one()' respectively, adding 'use num_traits::{Zero, One};' imports and 'where F: Zero + One' bounds where F is a type parameter
```

### Stream 1B: Fix Field Method Calls in rustmath-matrix
```
In rustmath-matrix, replace all field method calls like 'self.field.zero()' and 'base_field.one()' with associated function calls 'F::zero()' and 'F::one()', adding necessary num_traits imports and trait bounds
```

### Stream 2A: Implement GroupElement for Simple Group Types
```
Add Clone, Eq, Hash, Debug derives and implement GroupElement trait with identity(), inverse(), and op() methods for CubicBraidElement, GroupExpElement, and CactusGroupElement in rustmath-groups
```

### Stream 2B: Implement GroupElement for Complex Group Types
```
Add Clone, Eq, Hash, Debug derives and implement GroupElement trait with identity(), inverse(), and op() methods for ArtinGroupElement, Braid in rustmath-groups, ensuring Display trait is also implemented
```

### Stream 2C: Implement GroupElement for Generic Group Types
```
Add Clone, Eq, Hash, Debug derives and implement GroupElement trait for IndexedFreeGroupElement<I> and IndexedFreeAbelianGroupElement<I> in rustmath-groups with proper generic bounds on I: Clone + Eq + Hash + Debug + Display
```

### Stream 3A: Add AdditiveAbelianGroup Methods
```
In rustmath-groups/src/additive_abelian_group.rs, implement methods zero() returning identity element, generators() returning basis elements, and contains() for membership testing on AdditiveAbelianGroup
```

### Stream 3B: Add AdditiveAbelianGroupElement Methods
```
In rustmath-groups/src/additive_abelian_group.rs, implement is_zero(), scalar_multiply(n: i64), and Hash trait for AdditiveAbelianGroupElement
```

### Stream 3C: Add FreeGroupElement and IndexedFreeGroup Methods
```
In rustmath-groups/src/free_group.rs add letters() and exponent_sum(generator) methods to FreeGroupElement; in indexed_free_group.rs add ngens() method to IndexedFreeGroup<I>
```

### Stream 3D: Add PermutationGroup Constructors
```
In rustmath-groups/src/permutation_group.rs, implement symmetric(n: usize) and cyclic(n: usize) static constructor methods for PermutationGroup
```

### Stream 3E: Add Character Methods
```
In rustmath-groups/src/character.rs, implement values() method returning reference to internal HashMap for Character type
```

### Stream 4A: Fix Matrix Methods
```
In rustmath-matrix/src/lib.rs, add zero(rows, cols) constructor for zero matrix and ensure mul() method exists for matrix multiplication, or implement Mul trait if missing
```

### Stream 4B: Add Eq and Default Traits
```
In rustmath-groups/src/finitely_presented.rs add PartialEq and Eq derives/implementations for FinitelyPresentedGroup; add Default implementations returning identity for all group element types
```

### Stream 4C: Implement Mul Traits for Group Operations
```
Add std::ops::Mul trait implementations for group element types where self.op(other) provides multiplication, ensuring generic bounds are satisfied
```

### Stream 5: Fix Type Mismatches (requires analysis)
```
Review all E0308 type mismatch errors, identify whether function signature or call site needs updating, and fix consistently across the codebase ensuring API compatibility
```

---

## PRIORITY ORDER FOR EXECUTION

### Immediate (Do These First)
1. **Stream 1A + 1B**: Field/Ring method refactoring - BLOCKS MANY ERRORS
2. **Stream 3A + 3B**: AdditiveAbelianGroup methods - BLOCKS GROUP OPERATIONS

### High Priority (Do Next)
3. **Stream 2A**: Simple GroupElement implementations
4. **Stream 3C**: FreeGroupElement methods
5. **Stream 2B**: Complex GroupElement implementations

### Medium Priority (Then)
6. **Stream 3D**: PermutationGroup constructors
7. **Stream 2C**: Generic GroupElement implementations
8. **Stream 4A**: Matrix methods

### Low Priority (Finally)
9. **Stream 3E**: Character methods
10. **Stream 4B + 4C**: Trait implementations
11. **Stream 5**: Type mismatches (case-by-case)

---

## AUTOMATED REFACTORING SCRIPTS

### Script 1: Find and Replace Field Methods
```bash
#!/bin/bash
# field_method_fix.sh

# Find files with problematic patterns
FILES=$(rg -l "\.zero\(\)|\.one\(\)" --type rust rustmath-schemes rustmath-matrix)

for file in $FILES; do
    echo "Processing: $file"
    
    # Backup
    cp "$file" "$file.bak"
    
    # Replace patterns (careful with false positives)
    # This is a starting point - needs manual review
    sed -i 's/self\.base_field\.zero()/F::zero()/g' "$file"
    sed -i 's/self\.base_field\.one()/F::one()/g' "$file"
    sed -i 's/base_field\.zero()/F::zero()/g' "$file"
    sed -i 's/base_field\.one()/F::one()/g' "$file"
    
    # Check if num_traits import exists
    if ! grep -q "use num_traits::{Zero, One}" "$file"; then
        # Add import after other uses
        sed -i '/^use/a use num_traits::{Zero, One};' "$file"
    fi
done

echo "Done! Review changes and run: cargo build"
```

### Script 2: Generate GroupElement Implementation Templates
```bash
#!/bin/bash
# generate_group_element_impls.sh

TYPES=(
    "ArtinGroupElement:artin"
    "Braid:braid"
    "CactusGroupElement:cactus_group"
    "CubicBraidElement:cubic_braid"
    "GroupExpElement:group_exp"
)

for entry in "${TYPES[@]}"; do
    TYPE="${entry%%:*}"
    FILE="${entry##*:}"
    
    echo "// Add to rustmath-groups/src/${FILE}.rs"
    echo ""
    echo "impl GroupElement for ${TYPE} {"
    echo "    fn identity() -> Self {"
    echo "        // TODO: Implement identity element"
    echo "        unimplemented!()"
    echo "    }"
    echo ""
    echo "    fn inverse(&self) -> Self {"
    echo "        // TODO: Implement inverse"
    echo "        unimplemented!()"
    echo "    }"
    echo ""
    echo "    fn op(&self, other: &Self) -> Self {"
    echo "        // TODO: Implement group operation"
    echo "        unimplemented!()"
    echo "    }"
    echo "}"
    echo ""
    echo "impl fmt::Display for ${TYPE} {"
    echo "    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {"
    echo "        // TODO: Implement display"
    echo "        write!(f, \"{}(...)\", stringify!(${TYPE}))"
    echo "    }"
    echo "}"
    echo ""
    echo "---"
    echo ""
done
```

---

## VERIFICATION CHECKLIST

After each stream completes:

- [ ] Code compiles without errors
- [ ] All new methods have unit tests
- [ ] Documentation updated
- [ ] No clippy warnings introduced
- [ ] Error count decreased
- [ ] Git commit with descriptive message

Final verification:
- [ ] `cargo build` succeeds with 0 errors
- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` shows no new warnings
- [ ] Documentation builds: `cargo doc --no-deps`
- [ ] All changes committed and pushed

---

## NOTES

### Important Considerations

1. **GroupElement trait**: Requires understanding of group theory. The identity, inverse, and operation must satisfy group axioms.

2. **Field methods**: The transition from `field.zero()` to `F::zero()` is architectural. It reflects that these are type-level properties, not instance properties.

3. **Type mismatches**: Many of these may resolve automatically once foundational issues (field methods, missing methods) are fixed.

4. **Generic bounds**: When implementing for `IndexedFreeGroupElement<I>`, ensure `I` has appropriate bounds (Clone, Eq, Hash, Debug, Display).

5. **Testing**: Each new method should have at least a smoke test. Group operations should verify axioms.

### Post-Refactoring Tasks

1. **Performance review**: Check if any new implementations are inefficient
2. **API review**: Ensure consistency across the codebase
3. **Documentation pass**: Update module docs, add examples
4. **Clippy pass**: Address any warnings
5. **Format pass**: Run `cargo fmt` on all modified files

