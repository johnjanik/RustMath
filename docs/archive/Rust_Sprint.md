# RustMath Final Sprint Plan
**SageMath to Rust Conversion - Final Push**

## Executive Summary

This document outlines the plan to complete the remaining SageMath to Rust conversion. Based on analysis of trackers 01-14, we have **4,670 items** remaining across 15 major modules. The work is organized into **42 parallel sessions** that can be executed concurrently across **5 dependency layers**.

## Module Implementation Status

| Module | Trackers | Total Items | Implemented | Remaining | Progress |
|--------|----------|-------------|-------------|-----------|----------|
| Categories | 01, 02 | 1,053 | 52 | 1,001 | 4.9% |
| Combinatorics | 02-05 | 2,768 | 373 | 2,395 | 13.5% |
| Crypto | 05 | 107 | 28 | 79 | 26.2% |
| Databases | 05, 06 | 151 | 22 | 129 | 14.6% |
| Features | 06 | 180 | 0 | 180 | 0.0% |
| Graphs | 06, 07 | 749 | 458 | 291 | 61.1% |
| Interfaces (GAP) | 08 | 20 | 0 | 20 | 0.0% |
| Knots | 08 | 23 | 0 | 23 | 0.0% |
| L-Functions | 08 | 23 | 0 | 23 | 0.0% |
| Libraries | 08 | 228 | 1 | 227 | 0.4% |
| Manifolds | 08, 09 | 220 | 201 | 19 | 91.4% |
| Modules | 10 | 265 | 265 | 0 | ✅ 100% |
| Plot | 11 | 290 | 90 | 200 | 31.0% |
| Qqbar | 13 | 39 | 11 | 28 | 28.2% |
| Schemes | 13, 14 | 676 | 271 | 405 | 40.1% |
| Typeset | 14 | 17 | 0 | 17 | 0.0% |
| **TOTAL** | | **5,809** | **1,772** | **4,037** | **30.5%** |

## Dependency Layers & Parallelization Strategy

The modules are organized into 5 dependency layers. **All sessions within a layer can run in parallel**. Each layer must complete before the next layer begins.

### Layer 0: Core Infrastructure (2 sessions)
**Dependencies**: None
**Parallelization**: Run both sessions concurrently

| Session | Module | Command | Estimated Items |
|---------|--------|---------|-----------------|
| S01 | Categories-Core | Implement core category traits (Ring, Field, EuclideanDomain) from trackers 01-02 | 1,001 |
| S02 | Features | Implement feature detection system from tracker 06 | 180 |

### Layer 1: Fundamental Math Structures (6 sessions)
**Dependencies**: Categories-Core
**Parallelization**: Run all 6 sessions concurrently

| Session | Module | Command | Estimated Items |
|---------|--------|---------|-----------------|
| S03 | Combinatorics-Core | Implement permutations, combinations, partitions from tracker 02 | ~35 |
| S04 | Crypto-Classical | Implement classical ciphers and basic RSA from tracker 05 | 79 |
| S05 | Databases | Implement OEIS, Cunningham tables, Cremona database from trackers 05-06 | 129 |
| S06 | Qqbar | Implement algebraic number field (QQbar) from tracker 13 | 28 |
| S07 | Typeset | Implement LaTeX/ASCII art output system from tracker 14 | 17 |
| S08 | Knots | Implement knot theory basics (Jones polynomial, etc.) from tracker 08 | 23 |

### Layer 2: Advanced Combinatorics & Geometry (8 sessions)
**Dependencies**: Combinatorics-Core, Categories-Core
**Parallelization**: Run all 8 sessions concurrently

| Session | Module | Command | Estimated Items |
|---------|--------|---------|-----------------|
| S09 | Combinatorics-Posets | Implement posets, Hasse diagrams, Möbius functions from tracker 03 | ~500 |
| S10 | Combinatorics-Words | Implement word combinatorics, pattern avoidance from tracker 03 | ~500 |
| S11 | Combinatorics-Crystals | Implement crystal bases, root systems from tracker 04 | ~500 |
| S12 | Combinatorics-SymFunc | Implement symmetric functions, Schur polynomials from tracker 04 | ~500 |
| S13 | Combinatorics-Species | Implement species, generating functions from tracker 05 | ~400 |
| S14 | Combinatorics-Designs | Implement combinatorial designs, block designs from tracker 05 | ~333 |
| S15 | Manifolds | Complete remaining differential manifolds from trackers 08-09 | 19 |
| S16 | Libraries-Math | Implement mathematical constant libraries from tracker 08 | 227 |

### Layer 3: Graph Theory & Visualization (5 sessions)
**Dependencies**: Combinatorics-Core
**Parallelization**: Run all 5 sessions concurrently

| Session | Module | Command | Estimated Items |
|---------|--------|---------|-----------------|
| S17 | Graphs-Core | Complete graph construction, basic algorithms from tracker 06 | ~150 |
| S18 | Graphs-Advanced | Implement graph homomorphisms, automorphisms from tracker 07 | ~141 |
| S19 | Plot-2D | Implement 2D plotting (line, scatter, contour) from tracker 11 | ~100 |
| S20 | Plot-3D | Implement 3D plotting (surface, parametric) from tracker 11 | ~100 |
| S21 | Interfaces-GAP | Implement GAP (Groups, Algorithms, Programming) interface from tracker 08 | 20 |

### Layer 4: Algebraic Geometry & Analysis (6 sessions)
**Dependencies**: Categories-Core, Qqbar
**Parallelization**: Run all 6 sessions concurrently

| Session | Module | Command | Estimated Items |
|---------|--------|---------|-----------------|
| S22 | Schemes-Affine | Implement affine schemes and morphisms from tracker 13 | ~200 |
| S23 | Schemes-Projective | Implement projective schemes and varieties from tracker 13 | ~200 |
| S24 | Schemes-Toric | Implement toric varieties from tracker 14 | ~94 |
| S25 | Schemes-Curves | Implement algebraic curves (elliptic, hyperelliptic) from tracker 13 | ~105 |
| S26 | L-Functions | Implement L-functions and Dirichlet series from tracker 08 | 23 |
| S27 | Schemes-Jacobians | Complete Jacobian varieties and divisors from tracker 14 | ~87 |

---

## Detailed Session Instructions

Each session below is a **one-line instruction** that Claude can execute in a separate session.

### LAYER 0: Core Infrastructure

#### S01: Categories-Core
```
Implement core category theory infrastructure from trackers 01-02: trait hierarchy (Category, Functor, NaturalTransformation), axioms (Associativity, Commutativity, Unity), parent/element framework, coercion system, and morphism composition for algebraic structures
```

**Key files to create/modify:**
- `rustmath-category/src/axioms.rs` - Implement categorical axioms
- `rustmath-category/src/parent_element.rs` - Parent/element framework
- `rustmath-category/src/coercion.rs` - Type coercion system
- `rustmath-core/src/traits.rs` - Update trait hierarchy

**Estimated complexity**: High (foundational)
**Estimated time**: 3-4 days
**Priority**: Critical (blocks many other modules)

#### S02: Features
```
Implement feature detection system from tracker 06: optional dependency management, runtime feature checking, fallback implementations, and integration with Cargo feature flags
```

**Key files to create/modify:**
- `rustmath-features/` - New crate
- `rustmath-features/src/lib.rs` - Feature detection API
- `Cargo.toml` - Feature flag configuration

**Estimated complexity**: Medium
**Estimated time**: 1-2 days
**Priority**: Medium

---

### LAYER 1: Fundamental Math Structures

#### S03: Combinatorics-Core
```
Implement basic combinatorics from tracker 02: Partition variations (restricted, bounded), RankingTable for enumeration, SetSystem for set families, Binary word operations, and enumeration utilities
```

**Key files to create/modify:**
- `rustmath-combinatorics/src/partitions.rs` - Add restricted/bounded partitions
- `rustmath-combinatorics/src/set_systems.rs` - New module
- `rustmath-combinatorics/src/words.rs` - Binary words

**Estimated complexity**: Medium
**Estimated time**: 2 days
**Priority**: High (blocks Layer 2)

#### S04: Crypto-Classical
```
Implement remaining cryptography from tracker 05: stream ciphers (RC4, ChaCha20), authenticated encryption (GCM), key derivation (PBKDF2, Argon2), digital signatures (EdDSA), and cryptographic utilities
```

**Key files to create/modify:**
- `rustmath-crypto/src/stream_cipher.rs` - Stream ciphers
- `rustmath-crypto/src/aead.rs` - Authenticated encryption
- `rustmath-crypto/src/kdf.rs` - Key derivation
- `rustmath-crypto/src/signatures.rs` - Add EdDSA

**Estimated complexity**: Medium-High (security critical)
**Estimated time**: 2-3 days
**Priority**: Medium

#### S05: Databases
```
Implement mathematical databases from trackers 05-06: extend OEIS interface with sequence analysis, complete Cunningham tables for all bases, extend Cremona database with isogeny graphs, add LMFDB integration, and implement caching layer
```

**Key files to create/modify:**
- `rustmath-databases/src/oeis.rs` - Extend with analysis tools
- `rustmath-databases/src/cunningham.rs` - Complete all bases
- `rustmath-databases/src/cremona.rs` - Add isogeny graphs
- `rustmath-databases/src/lmfdb.rs` - New module
- `rustmath-databases/src/cache.rs` - Caching layer

**Estimated complexity**: Medium
**Estimated time**: 2 days
**Priority**: Low (mostly data integration)

#### S06: Qqbar
```
Implement algebraic numbers (QQbar) from tracker 13: algebraic closure of Q, minimal polynomials, algebraic operations with radicals, complex embeddings, Galois conjugates, and exact comparisons
```

**Key files to create/modify:**
- `rustmath-qqbar/` - New crate
- `rustmath-qqbar/src/lib.rs` - Algebraic number field
- `rustmath-qqbar/src/minimal_poly.rs` - Minimal polynomial computation
- `rustmath-qqbar/src/embeddings.rs` - Complex embeddings

**Estimated complexity**: High (advanced number theory)
**Estimated time**: 3-4 days
**Priority**: High (needed for schemes)

#### S07: Typeset
```
Implement typesetting system from tracker 14: LaTeX output for all mathematical objects, ASCII art rendering, Unicode math symbols, HTML/MathML generation, and customizable output formats
```

**Key files to create/modify:**
- `rustmath-typeset/` - New crate
- `rustmath-typeset/src/latex.rs` - LaTeX generator
- `rustmath-typeset/src/ascii.rs` - ASCII art renderer
- `rustmath-typeset/src/unicode.rs` - Unicode symbols
- `rustmath-typeset/src/html.rs` - HTML/MathML output

**Estimated complexity**: Low-Medium
**Estimated time**: 1-2 days
**Priority**: Low (cosmetic)

#### S08: Knots
```
Implement knot theory from tracker 08: knot representations (Gauss codes, PD codes, braid words), Reidemeister moves, Jones/HOMFLY polynomials, knot invariants (crossing number, unknotting number), and link operations
```

**Key files to create/modify:**
- `rustmath-knots/` - New crate
- `rustmath-knots/src/knot.rs` - Knot representation
- `rustmath-knots/src/invariants.rs` - Jones, HOMFLY polynomials
- `rustmath-knots/src/moves.rs` - Reidemeister moves

**Estimated complexity**: Medium-High
**Estimated time**: 2-3 days
**Priority**: Low (specialized area)

---

### LAYER 2: Advanced Combinatorics & Geometry

#### S09: Combinatorics-Posets
```
Implement posets from tracker 03: lattice operations, order ideals, order filters, distributive lattices, modular lattices, Birkhoff's representation, chain partitions, and antichain counting
```

**Key files to create/modify:**
- `rustmath-combinatorics/src/posets.rs` - Extend with lattice operations
- `rustmath-combinatorics/src/ideals.rs` - Order ideals/filters
- `rustmath-combinatorics/src/lattices.rs` - Distributive/modular lattices

**Estimated complexity**: Medium-High
**Estimated time**: 2-3 days
**Priority**: Medium

#### S10: Combinatorics-Words
```
Implement word combinatorics from tracker 03: Lyndon words, Christoffel words, Sturmian words, word morphisms, pattern matching, abelian complexity, and automatic sequences
```

**Key files to create/modify:**
- `rustmath-combinatorics/src/words.rs` - Extend with specialized words
- `rustmath-combinatorics/src/morphisms.rs` - Word morphisms
- `rustmath-combinatorics/src/patterns.rs` - Pattern matching

**Estimated complexity**: Medium
**Estimated time**: 2 days
**Priority**: Low

#### S11: Combinatorics-Crystals
```
Implement crystal bases from tracker 04: crystal operators, Kashiwara crystals, tensor products, character formulas, affine crystals, highest weight crystals, and Littelmann paths
```

**Key files to create/modify:**
- `rustmath-combinatorics/src/crystals/` - New module
- `rustmath-combinatorics/src/crystals/kashiwara.rs` - Kashiwara operators
- `rustmath-combinatorics/src/crystals/tensor.rs` - Tensor products
- `rustmath-combinatorics/src/root_systems.rs` - Root systems integration

**Estimated complexity**: Very High (requires Lie theory)
**Estimated time**: 4-5 days
**Priority**: Low (specialized)

#### S12: Combinatorics-SymFunc
```
Implement symmetric functions from tracker 04: Schur functions, monomial/elementary/power sum bases, Kostka numbers, plethysm, inner product, Jacobi-Trudi formula, and ribbon tableaux
```

**Key files to create/modify:**
- `rustmath-combinatorics/src/symmetric/` - New module
- `rustmath-combinatorics/src/symmetric/schur.rs` - Schur functions
- `rustmath-combinatorics/src/symmetric/bases.rs` - Multiple bases
- `rustmath-combinatorics/src/symmetric/plethysm.rs` - Plethysm operations

**Estimated complexity**: High
**Estimated time**: 3-4 days
**Priority**: Medium

#### S13: Combinatorics-Species
```
Implement combinatorial species from tracker 05: species operations (sum, product, composition), molecular decomposition, species generation, weighted species, and recursive species
```

**Key files to create/modify:**
- `rustmath-combinatorics/src/species.rs` - Species framework
- `rustmath-combinatorics/src/species_ops.rs` - Operations
- `rustmath-combinatorics/src/generating_functions.rs` - Link to power series

**Estimated complexity**: High (abstract concept)
**Estimated time**: 3 days
**Priority**: Low

#### S14: Combinatorics-Designs
```
Implement combinatorial designs from tracker 05: block designs (BIBD, Steiner systems), Latin squares, orthogonal arrays, difference sets, Hadamard matrices, and design automorphisms
```

**Key files to create/modify:**
- `rustmath-combinatorics/src/designs/` - New module
- `rustmath-combinatorics/src/designs/bibd.rs` - Balanced incomplete block designs
- `rustmath-combinatorics/src/designs/steiner.rs` - Steiner systems
- `rustmath-combinatorics/src/designs/latin.rs` - Extend Latin squares
- `rustmath-combinatorics/src/designs/hadamard.rs` - Hadamard matrices

**Estimated complexity**: Medium
**Estimated time**: 2-3 days
**Priority**: Low

#### S15: Manifolds
```
Complete differential manifolds from trackers 08-09: remaining chart operations, Lie derivatives, pullback/pushforward for general tensors, and integration on manifolds
```

**Key files to create/modify:**
- `rustmath-manifolds/src/charts.rs` - Complete chart atlas
- `rustmath-manifolds/src/tensor_calculus.rs` - Tensor operations
- `rustmath-manifolds/src/integration.rs` - Integration on manifolds

**Estimated complexity**: Medium (mostly completion)
**Estimated time**: 1 day
**Priority**: Low (91% complete)

#### S16: Libraries-Math
```
Implement mathematical libraries from tracker 08: special constants library (extended π, e, φ digits), mathematical tables, lookup tables for common sequences, precomputed factorizations, and primality certificates
```

**Key files to create/modify:**
- `rustmath-libs/` - New crate
- `rustmath-libs/src/constants.rs` - Mathematical constants
- `rustmath-libs/src/tables.rs` - Lookup tables
- `rustmath-libs/src/primes.rs` - Prime tables and certificates

**Estimated complexity**: Low (mostly data)
**Estimated time**: 1-2 days
**Priority**: Low

---

### LAYER 3: Graph Theory & Visualization

#### S17: Graphs-Core
```
Complete basic graph theory from tracker 06: graph products (Cartesian, tensor, strong), graph powers, line graphs, graph minors, graph invariants (girth, diameter, radius), and perfect graphs
```

**Key files to create/modify:**
- `rustmath-graphs/src/products.rs` - Graph products
- `rustmath-graphs/src/operations.rs` - Line graph, minors
- `rustmath-graphs/src/invariants.rs` - Extend invariants

**Estimated complexity**: Medium
**Estimated time**: 2 days
**Priority**: Medium

#### S18: Graphs-Advanced
```
Implement advanced graph theory from tracker 07: graph homomorphisms, graph automorphisms (using nauty algorithm), Cayley graphs, Tutte polynomial, graph spectra, strongly regular graphs, and Ramsey theory
```

**Key files to create/modify:**
- `rustmath-graphs/src/homomorphisms.rs` - Graph homomorphisms
- `rustmath-graphs/src/automorphisms.rs` - Automorphism group (nauty)
- `rustmath-graphs/src/cayley.rs` - Cayley graphs
- `rustmath-graphs/src/polynomials.rs` - Tutte polynomial
- `rustmath-graphs/src/spectra.rs` - Graph spectra

**Estimated complexity**: High
**Estimated time**: 3-4 days
**Priority**: Medium

#### S19: Plot-2D
```
Implement 2D plotting from tracker 11: line plots, scatter plots, bar charts, histograms, contour plots, density plots, parametric curves, implicit plots, and SVG/PNG export
```

**Key files to create/modify:**
- `rustmath-plot/` - New crate
- `rustmath-plot/src/plot2d.rs` - 2D plotting engine
- `rustmath-plot/src/contour.rs` - Contour plots
- `rustmath-plot/src/export.rs` - Export to SVG/PNG

**Estimated complexity**: Medium (can use plotters crate)
**Estimated time**: 2-3 days
**Priority**: Medium

#### S20: Plot-3D
```
Implement 3D plotting from tracker 11: surface plots, wireframe plots, parametric surfaces, 3D scatter, implicit surfaces, vector fields, and 3D export formats (OBJ, STL)
```

**Key files to create/modify:**
- `rustmath-plot/src/plot3d.rs` - 3D plotting engine
- `rustmath-plot/src/surfaces.rs` - Surface rendering
- `rustmath-plot/src/export3d.rs` - 3D export formats

**Estimated complexity**: Medium-High
**Estimated time**: 3 days
**Priority**: Low (visualization)

#### S21: Interfaces-GAP
```
Implement GAP interface from tracker 08: GAP process management, GAP command translation, group operations via GAP, permutation group algorithms, and result parsing from GAP output
```

**Key files to create/modify:**
- `rustmath-interfaces/` - New crate
- `rustmath-interfaces/src/gap.rs` - GAP interface
- `rustmath-interfaces/src/gap_parser.rs` - Parse GAP output

**Estimated complexity**: Medium (external dependency)
**Estimated time**: 2 days
**Priority**: Low (requires GAP installation)

---

### LAYER 4: Algebraic Geometry & Analysis

#### S22: Schemes-Affine
```
Implement affine schemes from tracker 13: affine scheme structure sheaves, morphisms of affine schemes, fiber products, base change, dimension theory, and affine varieties over general fields
```

**Key files to create/modify:**
- `rustmath-schemes/` - New crate
- `rustmath-schemes/src/affine.rs` - Affine schemes
- `rustmath-schemes/src/sheaves.rs` - Structure sheaves
- `rustmath-schemes/src/morphisms.rs` - Scheme morphisms

**Estimated complexity**: Very High (algebraic geometry)
**Estimated time**: 4-5 days
**Priority**: High (core AG)

#### S23: Schemes-Projective
```
Implement projective schemes from tracker 13: projective space, homogeneous coordinates, Veronese/Segre embeddings, graded rings, Proj construction, projective morphisms, and ample line bundles
```

**Key files to create/modify:**
- `rustmath-schemes/src/projective.rs` - Projective schemes
- `rustmath-schemes/src/graded.rs` - Graded rings
- `rustmath-schemes/src/embeddings.rs` - Veronese/Segre

**Estimated complexity**: Very High
**Estimated time**: 4-5 days
**Priority**: High

#### S24: Schemes-Toric
```
Implement toric varieties from tracker 14: complete toric variety construction, toric divisors, Chow groups, toric morphisms, moment maps, and fan subdivisions
```

**Key files to create/modify:**
- `rustmath-schemes/src/toric/` - Toric varieties (extend existing in geometry)
- `rustmath-schemes/src/toric/divisors.rs` - Toric divisors
- `rustmath-schemes/src/toric/chow.rs` - Chow groups

**Estimated complexity**: Very High
**Estimated time**: 4 days
**Priority**: Medium

#### S25: Schemes-Curves
```
Implement algebraic curves from tracker 13: plane curves, curve singularities, genus computation, hyperelliptic curves, curve parameterization, and Weierstrass form transformations
```

**Key files to create/modify:**
- `rustmath-schemes/src/curves/` - New module
- `rustmath-schemes/src/curves/plane.rs` - Plane curves
- `rustmath-schemes/src/curves/hyperelliptic.rs` - Hyperelliptic curves
- `rustmath-schemes/src/curves/genus.rs` - Genus computation

**Estimated complexity**: Very High
**Estimated time**: 4 days
**Priority**: Medium

#### S26: L-Functions
```
Implement L-functions from tracker 08: Dirichlet L-functions, functional equations, special values, Euler products, approximate functional equation, and critical line evaluation
```

**Key files to create/modify:**
- `rustmath-lfunctions/` - New crate
- `rustmath-lfunctions/src/dirichlet.rs` - Dirichlet L-functions
- `rustmath-lfunctions/src/euler_product.rs` - Euler product expansions
- `rustmath-lfunctions/src/functional_equation.rs` - Functional equations

**Estimated complexity**: Very High (analytic number theory)
**Estimated time**: 3-4 days
**Priority**: Low (specialized)

#### S27: Schemes-Jacobians
```
Complete Jacobian varieties from tracker 14: Picard groups, divisor class groups, Abel-Jacobi maps, theta functions, and Jacobian of hyperelliptic curves
```

**Key files to create/modify:**
- `rustmath-schemes/src/jacobians.rs` - Jacobian varieties
- `rustmath-schemes/src/picard.rs` - Picard groups
- `rustmath-schemes/src/theta.rs` - Theta functions

**Estimated complexity**: Very High
**Estimated time**: 4 days
**Priority**: Low

---

## Execution Strategy

### Parallel Execution Plan

**Maximum Concurrency**: You can run **8 sessions in parallel per layer**

#### Phase 1: Layer 0 (Critical Path - 2 concurrent sessions)
```
Session S01 (Categories-Core) + Session S02 (Features)
Estimated total time: 3-4 days
```

#### Phase 2: Layer 1 (Foundation - 6 concurrent sessions)
```
Sessions S03, S04, S05, S06, S07, S08 in parallel
Estimated total time: 3-4 days (limited by S06 Qqbar)
```

#### Phase 3: Layer 2 (Advanced - 8 concurrent sessions)
```
Sessions S09, S10, S11, S12, S13, S14, S15, S16 in parallel
Estimated total time: 4-5 days (limited by S11 Crystals)
```

#### Phase 4: Layer 3 (Graphs & Viz - 5 concurrent sessions)
```
Sessions S17, S18, S19, S20, S21 in parallel
Estimated total time: 3-4 days (limited by S18)
```

#### Phase 5: Layer 4 (Algebraic Geometry - 6 concurrent sessions)
```
Sessions S22, S23, S24, S25, S26, S27 in parallel
Estimated total time: 4-5 days (limited by multiple VH complexity items)
```

### Total Timeline Estimate
- **Minimum (with 8 parallel workers)**: ~17-22 days
- **Realistic (with 4 parallel workers)**: ~25-35 days
- **Conservative (with 2 parallel workers)**: ~40-50 days

---

## Session Command Format

Each session can be started with a command like:

```bash
# Start session for Categories-Core
claude --session "categories-core" --instruction "Implement core category theory infrastructure from trackers 01-02: trait hierarchy (Category, Functor, NaturalTransformation), axioms (Associativity, Commutativity, Unity), parent/element framework, coercion system, and morphism composition for algebraic structures"
```

---

## Priority Rankings

### Critical (Must Have)
1. S01: Categories-Core - Blocks everything
2. S06: Qqbar - Needed for schemes
3. S22: Schemes-Affine - Core algebraic geometry
4. S23: Schemes-Projective - Core algebraic geometry

### High (Should Have)
1. S03: Combinatorics-Core - Blocks Layer 2
2. S12: Combinatorics-SymFunc - Used in many areas
3. S17: Graphs-Core - Complete existing work
4. S18: Graphs-Advanced - High-value algorithms

### Medium (Nice to Have)
1. S04: Crypto - Security features
2. S09: Combinatorics-Posets - Extends existing
3. S19: Plot-2D - User-facing features
4. S24: Schemes-Toric - Specialized AG

### Low (Optional)
1. S02: Features - Infrastructure
2. S05: Databases - External data
3. S07: Typeset - Cosmetic
4. S08: Knots - Specialized
5. S10-S16: Advanced combinatorics - Specialized
6. S20-S21: 3D Plot, GAP - External dependencies
7. S25-S27: Advanced schemes - Very specialized

---

## Testing Strategy

Each session should include:
1. **Unit tests** for individual functions
2. **Integration tests** against SageMath output
3. **Property tests** using quickcheck
4. **Documentation examples** that run as tests

Example test pattern:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_against_sagemath() {
        // Compare RustMath output to known SageMath results
    }

    quickcheck! {
        fn prop_identity(x: i64) -> bool {
            // Property-based test
        }
    }
}
```

---

## Success Metrics

- [ ] All 4,037 remaining items implemented
- [ ] Test coverage ≥ 80% for new code
- [ ] All tests passing
- [ ] Documentation complete for public APIs
- [ ] Performance within 2x of SageMath on standard benchmarks
- [ ] Zero unsafe code added

---

## Rollout Plan

### Week 1-2: Layer 0 + Layer 1
Focus on foundation and critical dependencies

### Week 3-4: Layer 2
Advanced combinatorics (parallel execution)

### Week 5: Layer 3
Graph theory and visualization

### Week 6-7: Layer 4
Algebraic geometry (most complex)

### Week 8: Integration & Polish
- Full test suite run
- Documentation review
- Performance optimization
- Bug fixes

---

## Notes for Implementers

### For Categories (S01)
- Study SageMath's coercion system carefully - it's complex
- Parent/Element pattern is fundamental to SageMath design
- Focus on Ring/Field/Module first, other categories later

### For Combinatorics (S03-S14)
- Many algorithms have well-known implementations - reference Knuth Vol. 4
- Performance matters - these are often used in tight loops
- Iterator-based APIs work well in Rust

### For Schemes (S22-S27)
- This is the hardest part - algebraic geometry is deep
- Rely heavily on Gröbner basis code already implemented
- Study Hartshorne's "Algebraic Geometry" for theory
- May need computer algebra expertise

### For Plotting (S19-S20)
- Consider using `plotters` or `resvg` crates as backends
- SVG output is easier than raster
- Focus on mathematical accuracy over aesthetics

### For Interfaces (S21)
- GAP integration can be optional feature
- Use process spawning, not FFI
- Parse GAP output carefully - it's designed for humans

---

## Quick Reference: Session Assignment Table

| Layer | Sessions | Concurrency | Est. Days | Critical? |
|-------|----------|-------------|-----------|-----------|
| 0 | S01-S02 | 2 | 3-4 | Yes |
| 1 | S03-S08 | 6 | 3-4 | Partial |
| 2 | S09-S16 | 8 | 4-5 | No |
| 3 | S17-S21 | 5 | 3-4 | Partial |
| 4 | S22-S27 | 6 | 4-5 | Yes |

**Total**: 27 sessions across 5 layers, 17-22 days with full parallelization

---

## Appendix: Tracker File Details

### Tracker 01: Categories (first half)
- Focus: Basic category axioms, magmas, semigroups
- Status: 0% implemented
- Items: 281

### Tracker 02: Categories (second half) + Combinatorics (basic)
- Focus: Advanced categories, sets, basic combinatorics
- Status: 4.9% categories, 13.5% combinatorics
- Items: 772 categories, 35 combinatorics

### Tracker 03: Combinatorics (posets, words)
- Focus: Posets, words, tableaux
- Status: 13.5%
- Items: 1,000

### Tracker 04: Combinatorics (crystals, symmetric functions)
- Focus: Crystal bases, symmetric functions
- Status: 13.5%
- Items: 1,000

### Tracker 05: Combinatorics (species) + Crypto + Databases
- Focus: Species, designs, crypto, databases
- Status: 13.5% combinatorics, 26.2% crypto, 14.6% databases
- Items: 733 + 107 + 74

### Tracker 06: Databases + Features + Graphs (basic)
- Focus: Databases, feature system, basic graphs
- Status: 14.6% databases, 0% features, 61.1% graphs
- Items: 77 + 180 + 22

### Tracker 07: Graphs (advanced)
- Focus: Advanced graph algorithms
- Status: 61.1%
- Items: 727

### Tracker 08: Interfaces + Knots + L-Functions + Libraries + Manifolds
- Focus: GAP interface, knots, analysis, libraries
- Status: 0% interfaces, 0% knots, 0% lfunctions, 0.4% libs, 91.4% manifolds
- Items: 355 + 23 + 23 + 228 + 201

### Tracker 09: Manifolds (completion)
- Focus: Final manifold operations
- Status: 91.4%
- Items: 19

### Tracker 10: Modules
- Focus: Module theory
- Status: ✅ 100% COMPLETE
- Items: 265

### Tracker 11: Plot
- Focus: 2D/3D plotting
- Status: 31.0%
- Items: 290

### Tracker 13: Qqbar + Rings + Schemes (first half)
- Focus: Algebraic numbers, schemes
- Status: 28.2% qqbar, 40.1% schemes
- Items: 40 + 582

### Tracker 14: Schemes (second half) + Typeset
- Focus: Advanced schemes, LaTeX output
- Status: 40.1% schemes, 0% typeset
- Items: 94 + 17

---

**Document Version**: 1.0
**Last Updated**: 2025-11-19
**Total Remaining Work**: 4,037 items across 15 modules
**Estimated Completion**: 17-50 days depending on parallelization
