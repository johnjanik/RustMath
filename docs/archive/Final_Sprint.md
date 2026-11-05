# Final Sprint: Strategic Module Completion Plan
## Analysis of Tracker Parts 10-14

**Analysis Date:** 2025-11-20
**Goal:** Maximize module completion with optimal effort-to-impact ratio

---

## Executive Summary

Analysis of tracker parts 10-14 reveals **3,151 partial implementations** across 17 major modules. Strategic focus on **nearly-complete modules** and **high-value quick wins** can deliver maximum completion percentage with minimal effort.

### Key Findings
- **3 modules already at 100%:** sage.quadratic_forms, sage.structure, sage.symbolic (373 items) ✅
- **1 module in active progress:** sage.rings at 31% complete (506/1,634 items)
- **12 modules not yet started:** 1,813 items at 0% completion
- **Total across parts 10-14:** 3,846 items (22.9% complete, 77.1% partial)

### Strategic Recommendation
**Focus on finishing nearly-complete submodules** within sage.rings and **quick-win standalone modules** to achieve maximum completion metrics with minimal effort investment.

---

## Tier 1: Immediate Wins (1-2 Weeks)
### Nearly Complete Modules - Finishing Touches

These modules are **85-95% complete** and require minimal effort to mark as "Done":

| Module | Items | Status | Effort | Impact |
|--------|-------|--------|--------|--------|
| **sage.rings.sum_of_squares** | 5 | 90%+ | 2-4 hrs | ⭐⭐⭐⭐⭐ |
| **sage.rings.complex_interval** | 5 | 95% | 2-4 hrs | ⭐⭐⭐⭐⭐ |
| **sage.rings.factorint** | 5 | 85% | 4-8 hrs | ⭐⭐⭐⭐⭐ |

**Implementation Status:**
- ✅ **sum_of_squares** already exists at `rustmath-rings/src/sum_of_squares.rs` (336 lines, comprehensive tests)
- ✅ **complex_interval** already exists at `rustmath-complex/src/complex_interval.rs` (524 lines)
- ✅ **factorint** already exists at `rustmath-integers/src/factorint.rs` (464 lines)

**Required Work:**
1. Review existing implementations against SageMath API
2. Add any missing methods/functions
3. Verify comprehensive test coverage
4. Update documentation to match SageMath docs
5. Mark as complete in tracker

**Total Time:** 8-16 hours (1-2 days)
**Total Items Completed:** 15 items
**Completion Boost:** High visibility, showcases Rust's numeric computation strength

---

## Tier 2: Quick Wins - Standalone Modules (2-4 Weeks)
### Small, Self-Contained, High-Value Modules

These modules are **5-30 items**, self-contained, and perfect for Rust implementation:

### Week 1 - Easy Wins (5-6 days)

#### 1. **sage.monoids.string_ops** (5 items, 1-2 days)
**What:** Cryptographic text analysis functions
- `coincidence_index()`: N-gram probability analysis
- `coincidence_discriminant()`: Character pair frequency
- `frequency_distribution()`: N-gram frequency counting
- `strip_encoding()`: String normalization

**Why Quick:** Pure functions, no complex dependencies, perfect for Rust iterators
**Crate:** `rustmath-monoids/src/string_ops.rs`

#### 2. **sage.misc.temporary_file** (6 items, 1-2 days)
**What:** Temporary file/directory management utilities
- `tmp_filename()`, `tmp_dir()`: Temp file generation
- `atomic_write`, `atomic_dir`: Atomic file operations

**Why Quick:** Can wrap Rust's `tempfile` crate, pure utility code
**Crate:** `rustmath-misc/src/temporary_file.rs`

#### 3. **sage.typeset.unicode_art** (5 items, 1-2 days)
**What:** Unicode art rendering for mathematical objects
- `unicode_art()`: Render as unicode art
- `unicode_subscript()`, `unicode_superscript()`: Unicode formatting

**Why Quick:** String manipulation, character mapping, enhances UX
**Crate:** `rustmath-typeset/src/unicode_art.rs`

**Week 1 Total:** 16 items, 3 modules completed

---

### Week 2 - Core Algorithms (5-6 days)

#### 4. **sage.sets.disjoint_set** (5 items, 2-3 days)
**What:** Union-find (disjoint-set) data structure
- `DisjointSet_of_integers`: Array-based implementation
- `DisjointSet_of_hashables`: HashMap-based implementation
- Path compression and union-by-rank optimizations

**Why Valuable:** Classic CS algorithm, perfect for Rust's ownership model
**Crate:** `rustmath-sets/src/disjoint_set.rs`

```rust
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl DisjointSet {
    pub fn find(&mut self, x: usize) -> usize { /* path compression */ }
    pub fn union(&mut self, x: usize, y: usize) { /* union by rank */ }
}
```

#### 5. **sage.stats.basic_stats** (7 items, 2-3 days)
**What:** Essential statistical functions
- `mean()`, `median()`, `mode()`: Central tendency
- `std()`, `variance()`: Dispersion measures
- `moving_average()`: Time series analysis

**Why Valuable:** High-value user functionality, generic over Ring trait
**Crate:** `rustmath-stats/src/basic_stats.rs`

**Week 2 Total:** 12 items, 2 modules completed

---

### Week 3 - Algebraic Structures (5-6 days)

#### 6. **sage.monoids.free_abelian_monoid** (5 items, 2-3 days)
**What:** Free abelian monoid with finite generators
- Elements as integer exponent vectors
- Natural extension of existing rustmath-monoids

**Why Valuable:** Fits existing architecture, extends algebraic capabilities
**Crate:** `rustmath-monoids/src/free_abelian_monoid.rs`

#### 7. **sage.sets.integer_range** (7 items, 2-3 days)
**What:** Mathematical integer ranges with set operations
- Extends Rust's Range types with set algebra
- Union, intersection, complement operations

**Why Valuable:** Useful utility for many other modules, clean Rust mapping
**Crate:** `rustmath-sets/src/integer_range.rs`

**Week 3 Total:** 12 items, 2 modules completed

---

### Week 4 - Numerical Methods (8-11 days)

#### 8. **sage.numerical.gauss_legendre** (6 items, 3-5 days)
**What:** Gauss-Legendre quadrature for numerical integration
- `nodes()`: Cached integration nodes and weights
- `integrate_vector()`: Adaptive integration
- `estimate_error()`: Error estimation

**Why Valuable:** Well-understood numerical methods, showcases Rust performance
**Crate:** `rustmath-numerical/src/gauss_legendre.rs`

#### 9. **sage.numerical.optimize** (8 items, 4-6 days)
**What:** Optimization and root-finding functions
- `find_root()`: Bisection, Newton-Raphson, Brent's method
- `find_local_minimum/maximum()`: Optimization
- `find_fit()`: Curve fitting

**Why Valuable:** High-value functionality, can leverage existing Rust crates (argmin)
**Crate:** `rustmath-numerical/src/optimize.rs`

**Week 4 Total:** 14 items, 2 modules completed

---

## Tier 3: Strategic Targets (4-8 Weeks)
### Large Module Areas with High Impact

These are **major module areas** that represent significant implementation effort but high completion impact:

### sage.rings Submodules (Partial Progress: 31%)

**Priority Submodules:**

1. **sage.rings.number_field** (201 partials, 2-4 weeks)
   - Core algebraic number theory functionality
   - S-unit solver, orders, ideals, morphisms
   - High mathematical value

2. **sage.rings.padics** (195 partials, 2-4 weeks)
   - p-adic numbers (capped relative, fixed mod, absolute)
   - Factory/construction infrastructure
   - Extension mechanics, power computers

3. **sage.rings.function_field** (161 partials, 3-5 weeks)
   - Function fields over finite fields
   - Places, divisors, valuations
   - Advanced algebraic geometry

4. **sage.rings.real_mpfi** (7 items, 1-2 days) ⭐ QUICK WIN
   - Arbitrary precision real intervals
   - Low effort, high value

5. **sage.rings.integer** (8 items, 2-3 days) ⭐ QUICK WIN
   - Integer wrapper functionality
   - Extends existing rustmath-integers

6. **sage.rings.rational** (9 items, 2-3 days) ⭐ QUICK WIN
   - Rational number wrapper
   - Extends existing rustmath-rationals

**sage.rings Quick Wins Total:** 24 items in 5-8 days (real_mpfi + integer + rational)

---

### NOT STARTED Major Modules (0% Complete)

These modules are **entirely partial** but represent major functionality areas:

| Module | Partials | Estimated Effort | Priority |
|--------|----------|------------------|----------|
| **sage.schemes** | 677 | 8-16 weeks | HIGH |
| **sage.plot** | 291 | 6-12 weeks | MEDIUM |
| **sage.modules** | 266 | 4-8 weeks | HIGH |
| **sage.topology** | 111 | 3-6 weeks | MEDIUM |
| **sage.sets** | 95 | 2-4 weeks | HIGH |
| **sage.misc** | 95 | 2-4 weeks | MEDIUM |
| **sage.numerical** | 81 | 2-4 weeks | HIGH |
| **sage.tensor** | 58 | 2-4 weeks | MEDIUM |
| **sage.monoids** | 54 | 1-3 weeks | MEDIUM |
| **sage.stats** | 43 | 1-2 weeks | MEDIUM |

**Note:** These represent major undertakings. Focus on completing Tier 1 and Tier 2 first to maximize completion metrics.

---

## Implementation Roadmap

### Phase 1: Sprint Foundations (Week 1-2, 28 items)
**Goal:** Build momentum with quick wins

✅ **Already Complete (from prior work):**
- sum_of_squares (5 items) - Review & finalize
- complex_interval (5 items) - Review & finalize
- factorint (5 items) - Review & finalize

🚀 **New Implementations:**
- string_ops (5 items)
- temporary_file (6 items)
- unicode_art (5 items)
- disjoint_set (5 items)
- basic_stats (7 items)

**Deliverable:** 8 completed modules, 43 items marked complete

---

### Phase 2: Consolidation (Week 3-4, 26 items)
**Goal:** Strengthen algebraic and numerical foundations

- free_abelian_monoid (5 items)
- integer_range (7 items)
- gauss_legendre (6 items)
- optimize (8 items)

**Deliverable:** 4 additional modules, total 69 items complete

---

### Phase 3: Strategic Expansion (Week 5-8, 24 items)
**Goal:** Finish remaining sage.rings quick wins

- real_mpfi (7 items)
- integer (8 items)
- rational (9 items)

**Deliverable:** 3 additional modules, total 93 items complete

---

### Phase 4: Major Modules (Week 9-24+)
**Goal:** Tackle large module areas

- Begin sage.rings.number_field (201 items) - Prioritize core functionality
- Begin sage.rings.padics (195 items) - Start with element types
- Evaluate sage.modules (266 items) - High mathematical value

---

## Metrics and Success Criteria

### Completion Targets

**After Phase 1 (2 weeks):**
- ✅ 8 modules complete (5 new + 3 finalized)
- ✅ 43 total items marked complete
- ✅ ~1.4% boost in overall completion
- ✅ Momentum established with visible progress

**After Phase 2 (4 weeks):**
- ✅ 12 modules complete
- ✅ 69 total items marked complete
- ✅ ~2.3% boost in overall completion
- ✅ Strong foundation in numerical methods and data structures

**After Phase 3 (8 weeks):**
- ✅ 15 modules complete
- ✅ 93 total items marked complete
- ✅ ~3.1% boost in overall completion
- ✅ sage.rings nearing 40% completion milestone

**After Phase 4 (24 weeks):**
- ✅ sage.rings at 50%+ completion
- ✅ Major submodules (number_field, padics) substantially complete
- ✅ 300+ additional items marked complete
- ✅ ~12% boost in overall completion

---

## Risk Analysis and Mitigation

### Implementation Risks

1. **Dependency Complexity**
   - Risk: Some "quick win" modules may have hidden dependencies
   - Mitigation: Start with truly standalone modules (string_ops, temporary_file)
   - Contingency: Skip to next module if dependencies are complex

2. **API Surface Misalignment**
   - Risk: RustMath API may differ significantly from SageMath
   - Mitigation: Focus on mathematical correctness over exact API match
   - Contingency: Document differences, provide migration guide

3. **Testing Coverage Gaps**
   - Risk: Existing implementations may lack comprehensive tests
   - Mitigation: Add tests as part of "completion" criteria
   - Contingency: Use SageMath test suite as reference

4. **Scope Creep on Large Modules**
   - Risk: number_field and padics are massive undertakings
   - Mitigation: Break into smaller milestones, prioritize core functionality
   - Contingency: Mark submodules as complete incrementally

---

## Prioritization Matrix

### Effort vs Impact Analysis

```
High Impact, Low Effort (DO FIRST):
  ├─ sum_of_squares (REVIEW ONLY)
  ├─ complex_interval (REVIEW ONLY)
  ├─ factorint (REVIEW ONLY)
  ├─ string_ops
  ├─ temporary_file
  ├─ unicode_art
  ├─ real_mpfi
  ├─ integer
  └─ rational

High Impact, Medium Effort (DO SECOND):
  ├─ disjoint_set
  ├─ basic_stats
  ├─ free_abelian_monoid
  ├─ integer_range
  └─ gauss_legendre

High Impact, High Effort (DO LATER):
  ├─ optimize
  ├─ sage.rings.number_field
  ├─ sage.rings.padics
  └─ sage.schemes

Medium Impact (DEPRIORITIZE):
  ├─ sage.plot (visualization)
  ├─ sage.topology
  └─ sage.tensor
```

---

## Resource Allocation

### Recommended Team Structure

**Solo Developer:** Follow Phases 1-4 sequentially, 3-6 months timeline

**Small Team (2-3 developers):**
- Developer 1: Quick wins (Tier 1 & 2)
- Developer 2: sage.rings submodules (Tier 3)
- Developer 3: Testing, documentation, integration

**Larger Team (4+ developers):**
- Team A: Quick wins + numerical methods
- Team B: sage.rings major submodules
- Team C: sage.schemes, sage.modules
- Team D: Testing, CI/CD, documentation

---

## Technical Debt Considerations

### Quality Over Quantity

While maximizing completion percentage is the goal, **never sacrifice correctness for speed**:

✅ **DO:**
- Write comprehensive tests for every function
- Document assumptions and limitations
- Use safe Rust (no unsafe blocks)
- Maintain exact arithmetic where possible
- Follow existing RustMath architectural patterns

❌ **DON'T:**
- Mark items complete without thorough testing
- Skip edge cases or error handling
- Introduce unsafe code for performance
- Break existing APIs without strong justification
- Create circular dependencies between crates

---

## Success Stories: Already Complete Modules

### Modules at 100% (from Parts 10-14)

These modules serve as **proof of concept** that comprehensive SageMath→RustMath migration is achievable:

1. **sage.quadratic_forms** (75 items) ✅
   - Comprehensive quadratic form arithmetic
   - Binary, ternary forms with proper invariants
   - Perfect example of trait-based generic implementation

2. **sage.structure** (87 items) ✅
   - Core algebraic structure traits
   - Parent/element framework
   - Foundation for all other modules

3. **sage.symbolic** (211 items) ✅
   - Expression trees with differentiation
   - Assumption propagation system
   - Simplification engine

**Total:** 373 items at 100% completion demonstrates feasibility of approach.

---

## Conclusion

**Optimal Strategy:** Focus on **Tier 1 (15 items in 1-2 days)** and **Tier 2 (54 items in 4 weeks)** to achieve **69 completed items** in one month. This delivers maximum completion metrics with manageable effort.

**Long-term Vision:** Continue with sage.rings submodules (number_field, padics) to reach 50% completion of the largest module area, then expand into sage.schemes and sage.modules for major functionality breadth.

**Key Insight:** The 3,151 partial implementations represent a **70% complete RustMath** when combined with other tracker parts. Strategic finishing of nearly-complete modules and high-value quick wins maximizes visible progress and user impact.

---

## Next Steps

1. ✅ Review and finalize sum_of_squares, complex_interval, factorint (1-2 days)
2. 🚀 Implement string_ops, temporary_file, unicode_art (3-4 days)
3. 🚀 Implement disjoint_set, basic_stats (4-5 days)
4. 🚀 Implement free_abelian_monoid, integer_range (4-5 days)
5. 📊 Reassess and begin numerical methods (gauss_legendre, optimize)
6. 📈 Plan sage.rings major submodule implementation strategy

**First PR Target:** Sum of squares, complex_interval, factorint finalization + string_ops implementation (Week 1)

---

## Parallel Execution Prompts

### One-Liner Prompts for Maximum Parallelization

The following prompts can be executed **simultaneously in parallel** to maximize development velocity. Each prompt is independent and can be assigned to different developers or AI agents.

---

### Phase 1: Sprint Foundations (Week 1-2)

**Parallel Execution Group 1A - Review & Finalization (Can run simultaneously):**

```
Prompt 1A-1: Review rustmath-rings/src/sum_of_squares.rs against SageMath's sage.rings.sum_of_squares API, add any missing methods, verify comprehensive test coverage matches SageMath functionality, and update documentation to align with SageMath docs.

Prompt 1A-2: Review rustmath-complex/src/complex_interval.rs against SageMath's sage.rings.complex_interval API, add any missing interval arithmetic methods, verify test coverage for edge cases (zero-width intervals, infinity handling), and update documentation.

Prompt 1A-3: Review rustmath-integers/src/factorint.rs against SageMath's sage.rings.factorint API, verify factorization algorithms (Pollard's rho, trial division) match SageMath behavior, add comprehensive tests for edge cases, and update documentation.
```

**Parallel Execution Group 1B - New Implementations Week 1 (Can run simultaneously):**

```
Prompt 1B-1: Implement sage.monoids.string_ops in rustmath-monoids/src/string_ops.rs with functions: coincidence_index(s, n), coincidence_discriminant(s, n), frequency_distribution(s, n), strip_encoding(s). Use HashMap for frequency counting, add comprehensive tests, and document cryptographic text analysis use cases.

Prompt 1B-2: Implement sage.misc.temporary_file in rustmath-misc/src/temporary_file.rs wrapping Rust's tempfile crate with functions: tmp_filename(), tmp_dir(), atomic_write, atomic_dir. Follow Rust RAII patterns, add tests for cleanup, and document thread-safety guarantees.

Prompt 1B-3: Implement sage.typeset.unicode_art in rustmath-typeset/src/unicode_art.rs with functions: unicode_art(obj), unicode_subscript(text), unicode_superscript(text). Create character mapping tables for mathematical symbols, add tests for all supported characters, and document Unicode ranges used.
```

**Parallel Execution Group 1C - New Implementations Week 2 (Can run simultaneously):**

```
Prompt 1C-1: Implement sage.sets.disjoint_set in rustmath-sets/src/disjoint_set.rs with DisjointSet_of_integers (Vec-based) and DisjointSet_of_hashables (HashMap-based) using path compression and union-by-rank optimizations. Add comprehensive tests including performance benchmarks, and document time complexity.

Prompt 1C-2: Implement sage.stats.basic_stats in rustmath-stats/src/basic_stats.rs with functions: mean(v), median(v), mode(v), std(v, bias), variance(v, bias), moving_average(v, n). Make generic over Ring trait, add tests for exact (Rational) and approximate (f64) arithmetic, and document bias parameter behavior.
```

---

### Phase 2: Consolidation (Week 3-4)

**Parallel Execution Group 2A - Algebraic Structures (Can run simultaneously):**

```
Prompt 2A-1: Implement sage.monoids.free_abelian_monoid in rustmath-monoids/src/free_abelian_monoid.rs with FreeAbelianMonoid_class and FreeAbelianMonoidElement using Vec<Integer> for exponent representation. Implement monoid operations (composition, identity), add comprehensive tests including generator creation, and document relationship to existing rustmath-monoids architecture.

Prompt 2A-2: Implement sage.sets.integer_range in rustmath-sets/src/integer_range.rs extending Rust's Range types with mathematical set operations (union, intersection, complement, difference). Make generic over Integer types, add tests for range arithmetic, and document how it extends std::ops::Range.
```

**Parallel Execution Group 2B - Numerical Methods (Can run simultaneously):**

```
Prompt 2B-1: Implement sage.numerical.gauss_legendre in rustmath-numerical/src/gauss_legendre.rs with functions: nodes(degree, prec), nodes_uncached(degree, prec), estimate_error(results, prec, epsilon), integrate_vector(f, prec, epsilon), integrate_vector_N(f, prec, N). Use Legendre polynomial recurrence relations, implement node caching with lazy_static, add comprehensive tests against known integrals, and document error estimation approach.

Prompt 2B-2: Implement sage.numerical.optimize in rustmath-numerical/src/optimize.rs with functions: find_root(f, a, b) using bisection/Brent's method, find_local_minimum(f, a, b) using golden section search, find_local_maximum(f, a, b), find_fit(data, model) using least squares. Consider wrapping argmin crate for advanced methods, add tests with known solutions, and document convergence criteria.
```

---

### Phase 3: Strategic Expansion (Week 5-8)

**Parallel Execution Group 3A - sage.rings Quick Wins (Can run simultaneously):**

```
Prompt 3A-1: Implement sage.rings.real_mpfi in rustmath-rings/src/real_mpfi.rs for arbitrary precision real interval arithmetic wrapping MPFI library (or pure Rust implementation). Implement interval operations (addition, multiplication, division with proper rounding), add tests for interval containment, and document precision guarantees.

Prompt 3A-2: Implement sage.rings.integer wrapper in rustmath-integers/src/sage_wrapper.rs providing SageMath-compatible API for Integer type. Add missing SageMath methods (nth_root, is_prime_power, factorial, binomial), comprehensive tests comparing to SageMath behavior, and document API differences.

Prompt 3A-3: Implement sage.rings.rational wrapper in rustmath-rationals/src/sage_wrapper.rs providing SageMath-compatible API for Rational type. Add missing SageMath methods (continued_fraction, convergents, denominator, numerator), comprehensive tests, and document exact arithmetic guarantees.
```

---

### Phase 4: Major Modules (Week 9-24+)

**Parallel Execution Group 4A - sage.rings.number_field Core (Can run simultaneously):**

```
Prompt 4A-1: Implement sage.rings.number_field core number field class in rustmath-rings/src/number_field/mod.rs with NumberField struct, defining polynomial, basis computation, and field arithmetic. Start with absolute number fields over Q, add comprehensive tests for quadratic and cyclotomic fields, and document algebraic number theory foundations.

Prompt 4A-2: Implement sage.rings.number_field.order in rustmath-rings/src/number_field/order.rs with Order class representing rings of integers in number fields. Implement order basis computation, conductor calculation, and ideal factorization. Add tests for maximal orders, and document relationship to Dedekind domains.

Prompt 4A-3: Implement sage.rings.number_field.S_unit_solver in rustmath-rings/src/number_field/s_unit_solver.rs for solving S-unit equations using algorithm from Smart's paper. Implement fundamental units computation, regulator calculation, and S-unit group structure. Add tests with known examples from literature, and document algorithm complexity.

Prompt 4A-4: Implement sage.rings.number_field.morphisms in rustmath-rings/src/number_field/morphisms.rs for number field homomorphisms. Support embedding computations, Galois group actions, and automorphism groups. Add tests for splitting fields and normal extensions, and document Galois theory implementation approach.
```

**Parallel Execution Group 4B - sage.rings.padics Core (Can run simultaneously):**

```
Prompt 4B-1: Implement sage.rings.padics.factory in rustmath-rings/src/padics/factory.rs with Zp, Qp, Zq, Qq constructors for p-adic rings and fields. Support capped relative, capped absolute, and fixed modulus precision models. Add tests for different precision types, and document precision model trade-offs.

Prompt 4B-2: Implement sage.rings.padics.padic_capped_relative_element in rustmath-rings/src/padics/capped_relative.rs for p-adic elements with capped relative precision. Implement arithmetic operations (add, sub, mul, div) with proper precision tracking, Teichmüller lifts, and unit splitting. Add comprehensive tests including precision loss scenarios, and document precision semantics.

Prompt 4B-3: Implement sage.rings.padics.padic_extension in rustmath-rings/src/padics/extension.rs for unramified and Eisenstein extensions of p-adic fields. Support extension arithmetic, norm/trace computations, and embedding into algebraic closures. Add tests for Galois extensions, and document ramification theory implementation.

Prompt 4B-4: Implement sage.rings.padics.pow_computer in rustmath-rings/src/padics/pow_computer.rs for efficient p-power caching and modular reduction. Implement Frobenius endomorphism computations for extensions. Add performance benchmarks, and document cache invalidation strategy.
```

**Parallel Execution Group 4C - sage.schemes.elliptic_curves Foundation (Can run simultaneously):**

```
Prompt 4C-1: Implement sage.schemes.elliptic_curves.ell_generic in rustmath-schemes/src/elliptic_curves/generic.rs with EllipticCurve_generic base class supporting Weierstrass equations over arbitrary fields. Implement point addition, doubling, and scalar multiplication. Add tests for curves over Q, finite fields, and p-adics, and document arithmetic implementation.

Prompt 4C-2: Implement sage.schemes.elliptic_curves.ell_rational_field in rustmath-schemes/src/elliptic_curves/rational.rs for elliptic curves over Q. Support conductor computation, minimal models, Cremona database integration, and torsion group structure. Add tests with known curves from LMFDB, and document BSD conjecture computations.

Prompt 4C-3: Implement sage.schemes.elliptic_curves.isogeny in rustmath-schemes/src/elliptic_curves/isogeny.rs for isogeny computations using Vélu's formulas. Support isogeny degree computation, kernel polynomial calculation, and isogeny graph construction. Add tests for ℓ-isogenies, and document algorithm complexity.

Prompt 4C-4: Implement sage.schemes.elliptic_curves.heegner in rustmath-schemes/src/elliptic_curves/heegner.rs for Heegner point computations. Support complex multiplication, Gross-Zagier formula, and height pairings for BSD conjecture applications. Add tests with examples from literature, and document CM theory foundations.
```

**Parallel Execution Group 4D - Supporting Infrastructure (Can run simultaneously):**

```
Prompt 4D-1: Create rustmath-schemes crate with initial structure: Cargo.toml, src/lib.rs, src/elliptic_curves/mod.rs, src/generic/mod.rs, src/projective/mod.rs, src/affine/mod.rs. Set up dependencies on rustmath-core, rustmath-rings, and rustmath-polynomials. Add basic scheme trait definitions, and document architectural decisions.

Prompt 4D-2: Create rustmath-numerical crate with initial structure: Cargo.toml, src/lib.rs, src/integration/mod.rs, src/optimization/mod.rs, src/root_finding/mod.rs. Set up dependencies and feature flags for optional libraries (BLAS, LAPACK). Add basic numerical trait definitions, and document precision guarantees.

Prompt 4D-3: Create rustmath-stats crate with initial structure: Cargo.toml, src/lib.rs, src/basic_stats/mod.rs, src/distributions/mod.rs. Set up dependencies on rustmath-core for generic statistics over Ring trait. Add statistical trait definitions, and document exact vs approximate arithmetic modes.

Prompt 4D-4: Create rustmath-sets crate with initial structure: Cargo.toml, src/lib.rs, src/disjoint_set.rs, src/integer_range.rs, src/finite_set_maps.rs. Set up dependencies on rustmath-core. Add set theory trait definitions, and document relationship to Rust's std collections.
```

---

### Usage Instructions

**For Solo Development:**
1. Execute prompts sequentially within each group (1A → 1B → 1C → 2A → ...)
2. Run prompts within a group in any order (all are independent)
3. Complete all prompts in a phase before moving to next phase

**For Team Development (Recommended):**
1. Assign each prompt in a group to a different developer/agent
2. Execute all prompts in a group simultaneously
3. Merge results at end of each group
4. Move to next group once all prompts in current group complete

**For Maximum Parallelization:**
1. Phase 1 can run 8 prompts in parallel (1A-1 through 1C-2)
2. Phase 2 can run 4 prompts in parallel (2A-1 through 2B-2)
3. Phase 3 can run 3 prompts in parallel (3A-1 through 3A-3)
4. Phase 4 can run up to 16 prompts in parallel (4A-1 through 4D-4)

**Estimated Timeline with Full Parallelization:**
- Phase 1: 2 weeks → 3-4 days (with 8 parallel agents)
- Phase 2: 2 weeks → 4-6 days (with 4 parallel agents)
- Phase 3: 4 weeks → 1 week (with 3 parallel agents)
- Phase 4: 16 weeks → 2-4 weeks (with 16 parallel agents)

**Total: 24 weeks → 5-7 weeks with maximum parallelization**

---

*Analysis based on comprehensive review of sagemath_to_rustmath_tracker_part_10-14.csv files containing 3,846 total items across 17 major modules.*
