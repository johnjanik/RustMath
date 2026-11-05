# RustMath Combinatorial Sprint Build Plan

## Executive Summary

**Total SageMath Combinatorial Modules:** 147
**Currently Implemented:** 14 modules (9.5%)
**To Be Implemented:** 133 modules (90.5%)

**Estimated Phases:** 6 priority levels
**Maximum Concurrent Builds:** Up to 20+ modules per phase

---

## Current Implementation Status

### ✅ Already Implemented in RustMath
1. **binary_words** - Binary words, Lyndon words, necklaces
2. **combinations** - Combination generation and manipulation
3. **designs** - Block designs, Latin squares, Hadamard matrices
4. **enumeration** - Cartesian products, Gray codes, compositions
5. **partitions** - Integer partitions with various constraints
6. **permutations** - Permutation generation and operations
7. **posets** - Partially ordered sets (basic)
8. **ranking** - Ranking and unranking algorithms
9. **set_system** - Set system operations
10. **species** - Combinatorial species
11. **tableaux** - Young tableaux, Robinson-Schensted
12. **word/words** - Word operations, morphisms, sequences
13. **lib.rs** - Catalan, Bell, Stirling numbers, compositions, set partitions, Dyck words, perfect matchings, Latin squares
14. **fibonacci** (in lib.rs) - Fibonacci and Lucas numbers

---

## Analysis of Tracker Line 966 (Part 01)

**Line 966:** `sage.categories.dual.DualObjectsCategory`

This is a **category theory** module, not a combinatorics module. It's part of SageMath's category framework for algebraic structures. This would belong in a separate `rustmath-categories` crate for category theory infrastructure.

**Recommendation:** Skip for combinatorial sprint; handle in separate category theory implementation.

---

## Build Order by Priority Level

### 🎯 PRIORITY 1: Foundation Modules (10 modules)
**Dependencies:** None
**Concurrency:** All can be built in parallel

#### Core Combinatorial Structures (5 modules)
- **composition** ✅ (Already in lib.rs - extract to module)
- **composition_signed** - Signed compositions
- **subset** - Subset operations
- **tuple** - Tuple combinatorics
- **cartesian_product** ✅ (Already in enumeration.rs - enhance)

#### Number Sequences (5 modules)
- **binary_recurrence_sequences** - General binary recurrence (extract Fibonacci as special case)
- **counting** - General counting functions
- **q_analogues** - q-analogs of binomial coefficients, factorials
- **q_bernoulli** - q-Bernoulli numbers
- ~~**fibonacci_sequence**~~ ✅ (Already in lib.rs)

---

### 🎯 PRIORITY 2: First Wave Extensions (35 modules)
**Dependencies:** Priority 1 complete
**Concurrency:** All 35 can be built in parallel after Priority 1

#### Trees and Words (6 modules)
- **abstract_tree** - Abstract tree interface
- **binary_tree** - Binary trees (basic version in lib.rs - enhance)
- **rooted_tree** - Rooted tree structures
- **ordered_tree** - Ordered trees
- **dyck_word** ✅ (Already in lib.rs - extract to module)
- **nu_dyck_word** - ν-Dyck words (generalization)

#### Permutation Extensions (4 modules)
- **affine_permutation** - Affine permutations (type A, B, C, D, G)
- **colored_permutations** - Colored permutation groups
- **permutation_cython** - High-performance permutation algorithms
- **derangements** - Permutations with no fixed points

#### Partition Extensions (7 modules)
- **skew_partition** - Skew partitions (λ/μ)
- **partition_tuple** - Tuples of partitions
- **partition_kleshchev** - Kleshchev partitions
- **core** - t-cores and t-quotients
- **superpartition** - Super partitions
- **plane_partition** - Plane partitions
- **vector_partition** ✅ (Basic version exists - enhance)

#### Matching and Lattices (5 modules)
- **perfect_matching** ✅ (Already in lib.rs - extract to module)
- **fully_packed_loop** - Fully packed loop configurations
- **alternating_sign_matrix** - ASM enumeration and operations
- **tamari_lattices** - Tamari lattice structures
- **nu_tamari_lattice** - ν-Tamari lattices

#### Set Partitions (4 modules)
- **set_partition** ✅ (Already in lib.rs - extract to module)
- **set_partition_iterator** - Efficient set partition iteration
- **set_partition_ordered** - Ordered set partitions
- **multiset_partition_into_sets_ordered** - Multiset partitions

#### Integer Lists and Vectors (5 modules)
- **integer_lists** - Integer list generation with constraints
- **integer_vector** - Integer vectors with constraints
- **integer_vector_weighted** - Weighted integer vectors
- **integer_matrices** - Integer matrix enumeration
- **fast_vector_partitions** - Optimized vector partition algorithms

#### Degree Sequences (4 modules from misc categories)
- **degree_sequences** - Degree sequences for graphs
- **restricted_growth** - Restricted growth strings
- **sidon_sets** - Sidon set construction
- **subsets_hereditary** - Hereditary set systems

---

### 🎯 PRIORITY 3: Second Wave Extensions (41 modules)
**Dependencies:** Priority 1-2 complete
**Concurrency:** ~35 modules can be built in parallel (some have internal dependencies)

#### Advanced Words (3 modules)
- **necklace** - Necklaces and Lyndon words (enhance existing)
- **debruijn_sequence** - De Bruijn sequence generation
- **lyndon_words** - Lyndon word operations (enhance existing)

#### Tableaux Extensions (10 modules)
- **skew_tableau** - Skew tableaux
- **ribbon_tableau** - Ribbon tableaux
- **ribbon_shaped_tableau** - Ribbon-shaped tableaux
- **k_tableau** - k-tableaux
- **composition_tableau** - Composition tableaux
- **super_tableau** - Super tableaux
- **tableau_tuple** - Tuples of tableaux
- **shifted_primed_tableau** - Shifted primed tableaux
- **path_tableaux** - Path tableaux
- **tableau_residues** - Tableau residue operations

#### Poset Extensions (2 modules)
- **interval_posets** - Interval posets
- **shard_order** - Shard order for permutations

#### Parking Functions (2 modules)
- **parking_functions** - Parking function generation
- **non_decreasing_parking_function** - Non-decreasing parking functions

#### Finite State Machines (4 modules)
- **finite_state_machine** - FSM construction and operations
- **finite_state_machine_generators** - FSM generators
- **recognizable_series** - Recognizable series operations
- **regular_sequence** - Regular sequences

#### Graphs and Paths (2 modules)
- **graph_path** - Graph path enumeration
- **yang_baxter_graph** - Yang-Baxter graph construction

#### Polyominoes and Tilings (2 modules)
- **parallelogram_polyomino** - Parallelogram polyominoes
- **tiling** - Tiling enumeration

#### Combinatorial Maps (2 modules)
- **combinatorial_map** - Combinatorial map framework
- **bijectionist** - Automated bijection discovery

#### Enumeration Tools (6 modules)
- **backtrack** - Generic backtracking framework
- **dlx** - Dancing Links (Algorithm X)
- **enumeration_mod_permgroup** - Enumeration modulo group action
- **gray_codes** - Gray code generation (enhance existing)
- **integer_vectors_mod_permgroup** - Integer vectors mod symmetry
- **ranker** ✅ (Already exists - enhance)

#### Sequences and Special Numbers (10 modules)
- **expnums** - Exponential numbers
- **t_sequences** - t-sequences
- **sloane_functions** - OEIS sequence generators (145 functions!)
- **cyclic_sieving_phenomenon** - CSP verification
- **degree_sequences** - Graph degree sequences
- **growth** - Growth diagram computation
- **subsets_pairwise** - Pairwise disjoint sets
- ~~**restricted_growth**~~ (moved to Priority 2)
- ~~**sidon_sets**~~ (moved to Priority 2)
- ~~**subsets_hereditary**~~ (moved to Priority 2)

---

### 🎯 PRIORITY 4: Advanced Structures (16 modules)
**Dependencies:** Priority 1-3 complete
**Concurrency:** ~14 modules can be built in parallel

#### Robinson-Schensted-Knuth (2 modules)
- **rsk** ✅ (Basic version exists - enhance with full bijection)
- **hillman_grassl** - Hillman-Grassl algorithm

#### Polynomials and Keys (3 modules)
- **schubert_polynomial** - Schubert polynomial computation
- **key_polynomial** - Key polynomial construction
- **hall_polynomial** - Hall polynomial computation

#### Symmetric Functions (4 modules - LARGE!)
- **sf** - Symmetric functions (137 entities!)
- **ncsf_qsym** - Noncommutative symmetric functions & quasi-symmetric functions (62 entities)
- **ncsym** - Noncommutative symmetric functions (34 entities)
- **fqsym** - Free quasi-symmetric functions (10 entities)

#### Root Systems (1 module - VERY LARGE!)
- **root_system** - Root system infrastructure (227 entities!)

#### Geometric Structures (4 modules)
- **gelfand_tsetlin_patterns** - Gelfand-Tsetlin pattern enumeration
- **knutson_tao_puzzles** - Knutson-Tao puzzles
- **six_vertex_model** - Six-vertex model configurations
- **triangles_FHM** - FHM triangle construction

#### Baxter and Fully Commutative (2 modules)
- **baxter_permutations** - Baxter permutation enumeration
- **fully_commutative_elements** - Fully commutative elements in Coxeter groups

---

### 🎯 PRIORITY 5: Algebraic Structures (11 modules)
**Dependencies:** Priority 1-4 complete
**Concurrency:** All 11 can be built in parallel

#### Algebras (9 modules)
- **symmetric_group_algebra** - Symmetric group algebra over Z
- **descent_algebra** - Descent algebra of the symmetric group
- **partition_algebra** - Partition algebra (49 entities)
- **diagram_algebras** - General diagram algebras (42 entities)
- **blob_algebra** - Blob algebra construction
- **free_dendriform_algebra** - Free dendriform algebra
- **free_prelie_algebra** - Free pre-Lie algebra
- **grossman_larson_algebras** - Grossman-Larson Hopf algebras
- **partition_shifting_algebras** - Partition shifting algebras

#### Symmetric Group Representations (2 modules)
- **symmetric_group_representations** - Representation theory (17 entities)
- **specht_module** - Specht module construction (13 entities)

---

### 🎯 PRIORITY 6: Expert-Level Structures (12 modules)
**Dependencies:** Priority 1-5 complete
**Concurrency:** ~10 modules can be built in parallel

#### Crystals (2 modules - VERY LARGE!)
- **crystals** - Crystal base theory (198 entities!)
- **rigged_configurations** - Rigged configurations for crystals (94 entities)

#### Cluster Algebras (2 modules)
- **cluster_algebra_quiver** - Cluster algebra quiver operations (23 entities)
- **cluster_complex** - Cluster complex construction (3 entities)

#### Advanced Structures (8 modules)
- **kazhdan_lusztig** - Kazhdan-Lusztig polynomials
- **sine_gordon** - Sine-Gordon Y-systems
- **constellation** - Constellation construction (13 entities)
- **e_one_star** - E_1^* structure
- **similarity_class_type** - Similarity class types (19 entities)
- **shuffle** - Shuffle product operations
- **subword** - Subword operations
- **subword_complex** - Subword complexes

---

## One-Line Build Prompts

### Phase 1A: Extract and Organize Existing Code
```
Extract composition, dyck_word, set_partition, and perfect_matching from lib.rs into dedicated modules, keeping backward compatibility through re-exports
```

### Phase 1B: Priority 1 - Foundation (10 modules, fully parallel)

**Batch 1.1: Core Structures (3 new modules)**
```
Implement composition_signed for signed integer compositions with sign tracking and reversal operations
```
```
Implement subset module with efficient subset generation, rank/unrank, and successor algorithms
```
```
Implement tuple module for fixed-length tuple generation with lexicographic ordering
```

**Batch 1.2: Number Sequences (4 new modules)**
```
Implement binary_recurrence_sequences for general linear recurrences with characteristic polynomial solving
```
```
Implement q_analogues with q-binomial coefficients, q-factorials, and Gaussian polynomials
```
```
Implement q_bernoulli for q-Bernoulli numbers using the Carlitz q-analog
```
```
Implement counting module with general combinatorial counting functions including Eulerian numbers and Narayana numbers
```

**Enhancement:**
```
Enhance cartesian_product in enumeration.rs with iterator-based lazy evaluation and infinite product support
```

---

### Phase 2: Priority 2 - First Wave (35 modules, fully parallel after Phase 1)

**Batch 2.1: Trees (4 new modules)**
```
Implement abstract_tree with trait-based tree interface supporting traversal, serialization, and cloneable tree operations
```
```
Implement rooted_tree with parent pointers, depth tracking, and subtree operations
```
```
Implement ordered_tree for ordered plane trees with left-child right-sibling representation
```
```
Implement nu_dyck_word for ν-Dyck words with generalized bounce paths and area sequences
```

**Batch 2.2: Permutations (4 modules)**
```
Implement affine_permutation for all Coxeter types (A,B,C,D,G) with window notation and support for infinite permutations
```
```
Implement colored_permutations for wreath products Z_r ≀ S_n with cycle type computation
```
```
Implement permutation_cython with SIMD-optimized permutation multiplication, inversion, and cycle decomposition
```
```
Implement derangements with inclusion-exclusion counting and efficient generation via recursive construction
```

**Batch 2.3: Partitions (6 modules)**
```
Implement skew_partition for skew shapes λ/μ with ribbon decomposition and straightening
```
```
Implement partition_tuple for multipartitions with dominance ordering and conjugation
```
```
Implement partition_kleshchev for Kleshchev partitions over Hecke algebras with residue sequences
```
```
Implement core for t-cores, t-quotients, and abacus representation with core towers
```
```
Implement superpartition for super partitions with circled/uncircled parts and sign rules
```
```
Implement plane_partition for 3D partitions with MacMahon's enumeration and symmetry classes
```

**Batch 2.4: Matching and Lattices (4 modules)**
```
Implement fully_packed_loop for FPL configurations with link patterns and ASM bijection
```
```
Implement alternating_sign_matrix with Razumov-Stroganov correspondence and enumeration formulas
```
```
Implement tamari_lattices for Catalan lattices with rotation operations and meet/join
```
```
Implement nu_tamari_lattice for rational Tamari lattices indexed by Dyck paths
```

**Batch 2.5: Set Partitions (3 modules)**
```
Implement set_partition_iterator with restricted growth string generation and optimized Bell number iteration
```
```
Implement set_partition_ordered for ordered set partitions (compositions of set partitions)
```
```
Implement multiset_partition_into_sets_ordered for multiset partitions with frequency tracking
```

**Batch 2.6: Integer Vectors (5 modules)**
```
Implement integer_lists with constraint propagation for sum/min/max/length constraints
```
```
Implement integer_vector with weighted enumeration and lattice point counting
```
```
Implement integer_vector_weighted for weighted compositions with priority queues
```
```
Implement integer_matrices for matrix enumeration with row/column sum constraints
```
```
Implement fast_vector_partitions with memoization and dynamic programming for vector partition counting
```

**Batch 2.7: Miscellaneous Structures (4 modules)**
```
Implement degree_sequences with Erdős-Gallai theorem verification and graph realization
```
```
Implement restricted_growth for RG strings with surjection bijection to set partitions
```
```
Implement sidon_sets for B_h sequences with greedy and backtracking construction
```
```
Implement subsets_hereditary for hereditary set systems with shadow/shade operations
```

**Enhancements:**
```
Enhance binary_tree module with threaded trees, AVL operations, and full/complete tree generation
```
```
Extract dyck_word from lib.rs to dedicated module with peak/valley detection and bounce statistics
```
```
Extract perfect_matching from lib.rs with chord diagram representation and noncrossing matchings
```
```
Extract set_partition from lib.rs with refinement partial order and lattice operations
```

---

### Phase 3: Priority 3 - Second Wave (41 modules, mostly parallel)

**Batch 3.1: Advanced Words (3 modules)**
```
Implement debruijn_sequence with FKM algorithm and universal cycle construction
```
```
Enhance necklace module with Duval's algorithm for Lyndon factorization and fixed-density necklaces
```
```
Enhance lyndon_words with Chen-Fox-Lyndon theorem and standard factorization
```

**Batch 3.2: Tableaux Extensions Wave 1 (5 modules)**
```
Implement skew_tableau for skew semistandard tableaux with Schützenberger's jeu de taquin
```
```
Implement ribbon_tableau for ribbon tableaux with spin and fermionic formula
```
```
Implement ribbon_shaped_tableau for ribbons as special skew shapes with height function
```
```
Implement k_tableau for k-tableaux with weak tableaux and increasing tableaux variants
```
```
Implement composition_tableau for composition tableaux with descent set operations
```

**Batch 3.3: Tableaux Extensions Wave 2 (5 modules)**
```
Implement super_tableau for super tableaux with circled entries and super Schur functions
```
```
Implement tableau_tuple for tuples of tableaux with row-strict and column-strict variants
```
```
Implement shifted_primed_tableau for shifted primed tableaux with primed/unprimed entries
```
```
Implement path_tableaux for path tableaux counting lattice paths
```
```
Implement tableau_residues for residue computations in modular representation theory
```

**Batch 3.4: Posets and Parking (4 modules)**
```
Implement interval_posets for interval posets with Fishburn's theorem and interval orders
```
```
Implement shard_order for shard order on permutations with forcing relation
```
```
Implement parking_functions for parking functions with bijections to labeled trees
```
```
Implement non_decreasing_parking_function for non-decreasing parking functions with area statistics
```

**Batch 3.5: Finite State Machines (4 modules)**
```
Implement finite_state_machine with DFA/NFA/Moore/Mealy support, minimization, and composition
```
```
Implement finite_state_machine_generators for predefined FSM generators (all ones, counting, etc.)
```
```
Implement recognizable_series for power series recognized by weighted automata
```
```
Implement regular_sequence for automatic sequences with Cobham's theorem
```

**Batch 3.6: Graphs and Polyominoes (4 modules)**
```
Implement graph_path for directed graph paths with generating functions
```
```
Implement yang_baxter_graph for Yang-Baxter graphs from quantum groups
```
```
Implement parallelogram_polyomino for parallelogram polyominoes with bounce paths
```
```
Implement tiling for domino and polyomino tilings with transfer matrix method
```

**Batch 3.7: Combinatorial Tools (8 modules)**
```
Implement combinatorial_map with decorator-based map registration and bidirectional bijections
```
```
Implement bijectionist with automated bijection discovery via Burnside's lemma
```
```
Implement backtrack with generic constraint satisfaction and pruning callbacks
```
```
Implement dlx with Knuth's Dancing Links Algorithm X for exact cover problems
```
```
Implement enumeration_mod_permgroup with Burnside's lemma and Pólya enumeration
```
```
Implement integer_vectors_mod_permgroup for orbit representatives under group action
```
```
Enhance gray_codes in enumeration.rs with binary reflected Gray code and combinatorial Gray codes
```
```
Enhance ranker module with additional ranking algorithms for trees and graphs
```

**Batch 3.8: Special Sequences (7 modules)**
```
Implement expnums for exponential numbers with exponential generating functions
```
```
Implement t_sequences for t-analogs of integer sequences
```
```
Implement sloane_functions with 145 OEIS sequence generators including A000001-A000999
```
```
Implement cyclic_sieving_phenomenon with CSP triple verification and orbit polynomial computation
```
```
Implement growth for growth diagrams in the Robinson-Schensted correspondence
```
```
Implement subsets_pairwise for pairwise disjoint set families with sunflower lemma
```

---

### Phase 4: Priority 4 - Advanced Structures (16 modules, mostly parallel)

**Batch 4.1: RSK and Polynomials (5 modules)**
```
Enhance rsk module with dual RSK, mixed insertion, and Hecke insertion variants
```
```
Implement hillman_grassl for Hillman-Grassl algorithm relating matrices to tableaux pairs
```
```
Implement schubert_polynomial with divided difference operators and Monk's rule
```
```
Implement key_polynomial for Demazure characters and key polynomial multiplication
```
```
Implement hall_polynomial for Hall polynomials counting submodules with quotients
```

**Batch 4.2: Symmetric Functions (4 LARGE modules - sequential recommended)**
```
Implement sf module with complete/elementary/power/monomial/Schur bases and Littlewood-Richardson coefficients (137 entities)
```
```
Implement ncsf_qsym for noncommutative/quasi-symmetric functions with Hopf algebra structure (62 entities)
```
```
Implement ncsym for noncommutative symmetric functions with ribbon Schur functions (34 entities)
```
```
Implement fqsym for free quasi-symmetric functions with F-basis and shuffle product (10 entities)
```

**Batch 4.3: Root Systems (1 VERY LARGE module)**
```
Implement root_system with Cartan matrices, Weyl groups, weight lattices, and Dynkin diagrams for all types (A-G, affine) (227 entities)
```

**Batch 4.4: Geometric and Baxter (6 modules)**
```
Implement gelfand_tsetlin_patterns for GT patterns with semistandard tableaux bijection
```
```
Implement knutson_tao_puzzles for K-T puzzles computing Littlewood-Richardson coefficients
```
```
Implement six_vertex_model for ice-type models with domain wall boundary conditions
```
```
Implement triangles_FHM for Fomin-Harrington-Meszaros triangles
```
```
Implement baxter_permutations for Baxter permutations avoiding vincular patterns
```
```
Implement fully_commutative_elements for FC elements in Coxeter groups with heap posets
```

---

### Phase 5: Priority 5 - Algebraic Structures (11 modules, fully parallel)

**Batch 5.1: Group Algebras (3 modules)**
```
Implement symmetric_group_algebra for QSₙ and ZSₙ with multiplication via Coxeter relations
```
```
Implement descent_algebra for Solomon's descent algebra with product formulas
```
```
Implement grossman_larson_algebras for Grossman-Larson Hopf algebra of rooted trees
```

**Batch 5.2: Partition and Diagram Algebras (4 modules)**
```
Implement partition_algebra for partition algebra over arbitrary rings with diagram basis (49 entities)
```
```
Implement diagram_algebras for Brauer/Temperley-Lieb/Planar algebras with diagram multiplication (42 entities)
```
```
Implement blob_algebra for blob algebra with propagating number parameter
```
```
Implement partition_shifting_algebras for partition shifting algebra construction
```

**Batch 5.3: Free Algebras (2 modules)**
```
Implement free_dendriform_algebra for free dendriform algebra with planar binary trees
```
```
Implement free_prelie_algebra for free pre-Lie algebra with rooted trees
```

**Batch 5.4: Representations (2 modules)**
```
Implement symmetric_group_representations with irreducible representations indexed by partitions (17 entities)
```
```
Implement specht_module for Specht modules with polytabloid basis and Garnir relations (13 entities)
```

---

### Phase 6: Priority 6 - Expert Structures (12 modules, mostly parallel)

**Batch 6.1: Crystals (2 VERY LARGE modules - sequential recommended)**
```
Implement crystals module with Kashiwara operators, tensor products, and highest weight crystals for all types (198 entities)
```
```
Implement rigged_configurations for rigged configurations with bijection to tensor product crystals (94 entities)
```

**Batch 6.2: Cluster Algebras (2 modules)**
```
Implement cluster_algebra_quiver with mutation sequences, cluster variables, and mutation types (23 entities)
```
```
Implement cluster_complex for cluster complexes with generalized associahedra
```

**Batch 6.3: Advanced Specialized Structures (8 modules)**
```
Implement kazhdan_lusztig for Kazhdan-Lusztig polynomials with R-polynomials and Bruhat order
```
```
Implement sine_gordon for sine-Gordon Y-systems from quantum field theory
```
```
Implement constellation for constellations (genus 0 maps on surfaces) with encoding (13 entities)
```
```
Implement e_one_star for E_1^* Hopf algebra structure
```
```
Implement similarity_class_type for similarity class types in matrix conjugacy (19 entities)
```
```
Implement shuffle for shuffle product on words and algebras
```
```
Implement subword for subword order and subword complexes
```
```
Implement subword_complex for Knutson-Miller subword complexes and pipe dreams
```

---

## Optimization Strategies

### Module Extraction Pattern
Many simple structures are already in `lib.rs`:
1. Extract to dedicated module
2. Add comprehensive tests
3. Add iterators and lazy evaluation
4. Keep re-exports in lib.rs for backward compatibility

### Concurrent Build Strategy
1. **Priority 1-2:** Can build ~45 modules in parallel (foundation + first wave)
2. **Priority 3:** Can build ~35 modules in parallel (some internal dependencies)
3. **Priority 4-6:** Mix of large sequential builds (sf, root_system, crystals) and small parallel builds

### Large Module Approach
For modules with 100+ entities:
- **sf (137), sloane_functions (145), crystals (198), root_system (227), designs (290)**
- Break into sub-modules by functionality
- Implement incrementally with feature flags
- Use trait-based abstractions for extensibility

### Testing Strategy
1. Property-based testing with `proptest` for enumeration correctness
2. Cross-check counts with OEIS sequences
3. Bijection verification (round-trip tests)
4. Performance benchmarks against known algorithms

### Integration with Existing Crates
- **rustmath-core:** Ring/Field traits for algebraic structures
- **rustmath-polynomials:** For symmetric functions and polynomials
- **rustmath-matrix:** For matrix-based algorithms
- **rustmath-graphs:** For graph path enumeration
- **rustmath-integers:** For number-theoretic sequences

---

## Quick Reference: Maximum Parallelism Per Phase

| Phase | Priority | Total Modules | Max Parallel | Sequential | Notes |
|-------|----------|---------------|--------------|------------|-------|
| 1A | Extract | 4 | 1 | 4 | Refactoring existing code |
| 1B | 1 | 7 | 7 | 0 | All independent |
| 2 | 2 | 35 | 35 | 0 | All depend only on P1 |
| 3 | 3 | 41 | 35 | 6 | Some tableau dependencies |
| 4 | 4 | 16 | 12 | 4 | Large sf/root_system sequential |
| 5 | 5 | 11 | 11 | 0 | All depend on P1-4 |
| 6 | 6 | 12 | 10 | 2 | Crystals sequential |
| **Total** | | **133** | **111** | **22** | |

**Estimated Timeline:**
- **Phase 1:** 1-2 weeks (foundation)
- **Phase 2:** 2-3 weeks (35 modules in parallel)
- **Phase 3:** 2-3 weeks (41 modules, some dependencies)
- **Phase 4:** 3-4 weeks (large modules like sf, root_system)
- **Phase 5:** 2 weeks (algebras)
- **Phase 6:** 3-4 weeks (crystals + cluster algebras)

**Total:** ~13-18 weeks for complete implementation

---

## Dependencies Not in Combinatorics

Some prompts reference:
- **rustmath-categories:** For category theory infrastructure (if implementing categorical methods)
- **rustmath-lie-algebras:** For Lie algebra support (crystals, root systems)
- **rustmath-quantum:** For quantum groups (some crystals, Yang-Baxter)

These are out of scope for the pure combinatorics sprint but should be tracked separately.

---

## Conclusion

This sprint plan provides:
1. ✅ **Complete coverage** of 147 SageMath combinatorial modules
2. ✅ **Efficient build order** with 6 priority levels
3. ✅ **Maximum parallelism** - up to 35 modules can be built concurrently in Phase 2
4. ✅ **One-line prompts** for each module group
5. ✅ **Clear dependencies** and sequencing constraints

**Next Steps:**
1. Begin Phase 1A: Extract existing code from lib.rs
2. Launch Phase 1B: Build all 7 Priority 1 modules in parallel
3. Proceed through phases sequentially, maximizing parallelism within each phase

---

**Generated:** 2025-11-20
**Tracker Analysis:** Parts 01 (line 966), 02-05 (complete)
**Total Entities Analyzed:** ~2768 combinatorial entities from SageMath
