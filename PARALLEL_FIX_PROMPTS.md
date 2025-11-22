# RustMath Parallel Fix Prompts

This document contains specific, actionable prompts to fix all build errors and warnings in the RustMath project. Each prompt is designed to be executed independently and in parallel where possible.

---

## CRITICAL FIX (Must be done FIRST)

### Fix 1: Add Ring and CommutativeRing trait implementations to UnivariatePolynomial
**Priority:** CRITICAL - Blocks 35 crates
**File:** `rustmath-polynomials/src/univariate.rs`
**Location:** After line 772, before line 773
**Estimated Time:** 2 minutes

**Prompt:**
```
In rustmath-polynomials/src/univariate.rs, add Ring and CommutativeRing trait implementations for UnivariatePolynomial. Insert the following code after line 772 (before the IntegralDomain impl on line 773):

// Ring implementation for polynomials
impl<R: Ring> Ring for UnivariatePolynomial<R> {
    fn zero() -> Self {
        UnivariatePolynomial::new(vec![R::zero()])
    }

    fn one() -> Self {
        UnivariatePolynomial::new(vec![R::one()])
    }

    fn is_zero(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_zero()
    }

    fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_one()
    }
}

// CommutativeRing implementation (polynomials are commutative)
impl<R: CommutativeRing> CommutativeRing for UnivariatePolynomial<R> {}

Then verify with: cargo build -p rustmath-polynomials && cargo build --all
```

---

## HIGH PRIORITY WARNINGS (Should be done SECOND)

### Fix 2: Replace mutable static with thread-safe Lazy in rustmath-numbertheory
**Priority:** HIGH - Undefined behavior and Rust 2024 incompatibility
**File:** `rustmath-numbertheory/src/bernoulli.rs`
**Estimated Time:** 10 minutes

**Prompt:**
```
In rustmath-numbertheory/src/bernoulli.rs, replace the mutable static CACHE with a thread-safe Lazy<Mutex<>> to fix undefined behavior. Replace the static declaration and all usages (lines 86, 90, 123). Add 'use once_cell::sync::Lazy;' and 'use std::sync::Mutex;' at the top, change 'static mut CACHE' to use Lazy<Mutex<HashMap>>, and update all access points to use lock(). Verify with: cargo build -p rustmath-numbertheory
```

---

## PARALLEL WARNING FIXES (Can be done in any order, in parallel)

### Fix 3: Remove unused variables in rustmath-special-functions
**Priority:** LOW
**Files:** `rustmath-special-functions/src/bessel.rs`, `rustmath-special-functions/src/error.rs`
**Estimated Time:** 3 minutes

**Prompt:**
```
In rustmath-special-functions, prefix unused variables with underscore: change 'mu' to '_mu' at bessel.rs:64, 'k_f' to '_k_f' at bessel.rs:120, and 'x2' to '_x2' at error.rs:69. Verify with: cargo build -p rustmath-special-functions
```

### Fix 4: Remove unused imports in rustmath-special-functions
**Priority:** LOW
**File:** `rustmath-special-functions/src/lib.rs`
**Estimated Time:** 2 minutes

**Prompt:**
```
In rustmath-special-functions/src/lib.rs, remove unused imports: delete 'Float, One, Zero' from line 12 and 'E, PI' from line 13. Verify with: cargo build -p rustmath-special-functions
```

### Fix 5: Fix unused assignment in rustmath-special-functions
**Priority:** LOW
**File:** `rustmath-special-functions/src/error.rs`
**Estimated Time:** 2 minutes

**Prompt:**
```
In rustmath-special-functions/src/error.rs:93, either remove the unused assignment 'let mut a = 1.0;' or fix the logic to use the variable. Verify with: cargo build -p rustmath-special-functions
```

### Fix 6: Remove unused variables in rustmath-stats
**Priority:** LOW
**File:** `rustmath-stats/src/lib.rs`
**Estimated Time:** 3 minutes

**Prompt:**
```
In rustmath-stats, prefix or remove unused variables 'alpha' and 'n_features', and remove unnecessary 'mut' keyword. Verify with: cargo build -p rustmath-stats
```

### Fix 7: Remove unused imports in rustmath-stats
**Priority:** LOW
**File:** `rustmath-stats/src/lib.rs`
**Estimated Time:** 2 minutes

**Prompt:**
```
In rustmath-stats/src/lib.rs, remove unused imports: 'std::f64::consts::PI' and 'variance'. Verify with: cargo build -p rustmath-stats
```

### Fix 8: Remove unused variables in rustmath-numerical
**Priority:** LOW
**File:** `rustmath-numerical/src/lib.rs`
**Estimated Time:** 3 minutes

**Prompt:**
```
In rustmath-numerical, prefix unused variables with underscore or remove: '_x', '_b', '_max_iter', and remove unnecessary 'mut' keyword and fix unused assignment to 'fb'. Verify with: cargo build -p rustmath-numerical
```

### Fix 9: Fix unused assignments in rustmath-dynamics
**Priority:** LOW
**File:** `rustmath-dynamics/src/lib.rs`
**Estimated Time:** 3 minutes

**Prompt:**
```
In rustmath-dynamics, fix unused assignments to 'x_prev' (2 occurrences) - either remove the assignments or fix the logic to use the values. Verify with: cargo build -p rustmath-dynamics
```

### Fix 10: Fix unused assignment in rustmath-numbertheory
**Priority:** LOW
**File:** `rustmath-numbertheory/src/quadratic_forms.rs`
**Estimated Time:** 2 minutes

**Prompt:**
```
In rustmath-numbertheory/src/quadratic_forms.rs:432, fix unused assignment to 'count' - either remove the assignment or fix the logic to use the value. Verify with: cargo build -p rustmath-numbertheory
```

### Fix 11: Remove unused struct fields in rustmath-databases
**Priority:** LOW
**File:** `rustmath-databases/src/lib.rs`
**Estimated Time:** 5 minutes

**Prompt:**
```
In rustmath-databases, remove unused struct fields: 'count', 'data_path', 'base_url', 'client', 'cache' or add them to constructor/methods if intended for future use. Verify with: cargo build -p rustmath-databases
```

### Fix 12: Remove unused struct field in rustmath-monoids
**Priority:** LOW
**File:** `rustmath-monoids/src/lib.rs`
**Estimated Time:** 2 minutes

**Prompt:**
```
In rustmath-monoids, remove unused struct field 'mult_table' or use it in implementation. Verify with: cargo build -p rustmath-monoids
```

### Fix 13: Remove unused struct field in rustmath-misc
**Priority:** LOW
**File:** `rustmath-misc/src/lib.rs`
**Estimated Time:** 2 minutes

**Prompt:**
```
In rustmath-misc, remove unused struct field 'headers' or use it in implementation. Verify with: cargo build -p rustmath-misc
```

### Fix 14: Remove unused functions in rustmath-plot
**Priority:** LOW
**File:** `rustmath-plot/src/lib.rs`
**Estimated Time:** 3 minutes

**Prompt:**
```
In rustmath-plot, either remove unused functions 'plot_multiple', 'parametric_plot_adaptive', 'list_plot_y' or make them public if intended for external use. Verify with: cargo build -p rustmath-plot
```

### Fix 15: Remove unused associated items in rustmath-plot-core
**Priority:** LOW
**File:** `rustmath-plot-core/src/lib.rs`
**Estimated Time:** 3 minutes

**Prompt:**
```
In rustmath-plot-core, either remove unused associated items 'new', 'from_points', 'center', 'volume', 'contains' or make them public if intended for external use. Verify with: cargo build -p rustmath-plot-core
```

---

## EXECUTION STRATEGY

### Sequential Execution (Recommended)
```bash
# STEP 1: Fix critical error (MUST BE FIRST)
# Execute Fix 1
cargo build --all  # Should succeed with warnings

# STEP 2: Fix high-priority warning
# Execute Fix 2
cargo build -p rustmath-numbertheory

# STEP 3: Fix all low-priority warnings (can be done in parallel)
# Execute Fixes 3-15 in any order
```

### Parallel Execution (Advanced)
```bash
# STEP 1: Fix critical error (MUST BE FIRST)
# Execute Fix 1
cargo build --all

# STEP 2: Create feature branches and fix in parallel
git checkout -b fix/numbertheory-mutable-static
# Execute Fix 2
# ... commit and push

# In separate terminals/processes:
git checkout -b fix/special-functions-warnings
# Execute Fixes 3-5
# ... commit and push

git checkout -b fix/stats-warnings
# Execute Fixes 6-7
# ... commit and push

git checkout -b fix/numerical-warnings
# Execute Fix 8
# ... commit and push

# ... etc for remaining fixes
```

---

## VERIFICATION COMMANDS

### After Critical Fix (Fix 1):
```bash
cargo build --all
# Expected: 0 errors, ~40 warnings (from 10 crates)
```

### After High Priority Fix (Fix 2):
```bash
cargo build -p rustmath-numbertheory
# Expected: 0 errors, 1 warning (down from 4)
```

### After All Fixes:
```bash
cargo build --all
# Expected: 0 errors, 0 warnings

cargo clippy --all
# Expected: Clean or minimal clippy warnings

cargo test --all
# Expected: All tests pass (same as before fixes)
```

---

## COPY-PASTE PROMPTS FOR CLAUDE CODE

Below are the exact prompts you can copy-paste to Claude Code to execute all fixes in parallel:

### 🔴 CRITICAL (Execute First, Alone)
```
Fix the missing Ring and CommutativeRing trait implementations in rustmath-polynomials/src/univariate.rs at line 772 by adding the trait impls before the IntegralDomain impl, then verify the fix resolves all 35 cascading build failures.
```

### 🟡 HIGH PRIORITY (Execute Second, Alone)
```
Replace the unsafe mutable static CACHE in rustmath-numbertheory/src/bernoulli.rs with a thread-safe Lazy<Mutex<HashMap>> to fix undefined behavior and Rust 2024 compatibility issues.
```

### 🟢 LOW PRIORITY (Can Execute All in Parallel)

**Batch 1: rustmath-special-functions**
```
Fix all warnings in rustmath-special-functions by: (1) prefixing unused variables mu, k_f, x2 with underscore, (2) removing unused imports Float, One, Zero, E, PI, (3) fixing or removing unused assignment to variable 'a'.
```

**Batch 2: rustmath-stats**
```
Fix all warnings in rustmath-stats by removing unused imports and variables, and removing unnecessary mutability.
```

**Batch 3: rustmath-numerical**
```
Fix all warnings in rustmath-numerical by prefixing unused variables with underscore, removing unnecessary mutability, and fixing unused assignment to 'fb'.
```

**Batch 4: rustmath-dynamics**
```
Fix unused assignment warnings in rustmath-dynamics by fixing or removing the two unused assignments to 'x_prev'.
```

**Batch 5: rustmath-numbertheory (additional)**
```
Fix the unused assignment warning in rustmath-numbertheory/src/quadratic_forms.rs:432 for variable 'count'.
```

**Batch 6: rustmath-databases**
```
Remove unused struct fields in rustmath-databases: count, data_path, base_url, client, cache.
```

**Batch 7: rustmath-monoids**
```
Remove unused struct field 'mult_table' in rustmath-monoids or add it to methods if intended for use.
```

**Batch 8: rustmath-misc**
```
Remove unused struct field 'headers' in rustmath-misc or use it in implementation.
```

**Batch 9: rustmath-plot**
```
Remove unused functions in rustmath-plot: plot_multiple, parametric_plot_adaptive, list_plot_y.
```

**Batch 10: rustmath-plot-core**
```
Remove unused associated items in rustmath-plot-core: new, from_points, center, volume, contains.
```

---

## DEPENDENCIES BETWEEN FIXES

```
Fix 1 (polynomials)
  ↓
  No dependencies, can proceed with any other fix

Fix 2 (numbertheory)
  ↓
  Independent of all other fixes

Fixes 3-15 (all warnings)
  ↓
  All independent of each other
  Can be executed in any order
  Can be executed in parallel
```

---

## ESTIMATED TOTAL TIME

- **Critical Fix:** 5 minutes
- **High Priority Fix:** 15 minutes
- **All Low Priority Fixes (parallel):** 30 minutes
- **Verification:** 10 minutes

**Total Sequential:** ~60 minutes
**Total Parallel:** ~30 minutes (if all warning fixes done simultaneously)

---

## SUCCESS CRITERIA

✅ All 59 crates build successfully
✅ Zero compilation errors
✅ Zero warnings
✅ All existing tests pass
✅ Clippy returns clean or minimal warnings
✅ No unsafe code introduced
✅ No breaking API changes
