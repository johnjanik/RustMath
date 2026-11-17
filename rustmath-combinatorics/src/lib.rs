//! RustMath Combinatorics - Combinatorial objects and algorithms
//!
//! This crate provides combinatorial structures like permutations, combinations,
//! partitions, and algorithms for generating and manipulating them.

#[macro_use]
extern crate lazy_static;

pub mod affine_permutations;
pub mod alternating_sign_matrix;
pub mod baxter_permutations;
pub mod bijectionist;
pub mod backtrack;
pub mod combinatorial_map;
pub mod combinatorial_map_examples;
pub mod binary_words;
pub mod cluster_complex;
pub mod combinations;
pub mod composition;
pub mod dendriform_algebra;
pub mod cyclic_sieving;
pub mod composition_tableau;
pub mod constellation;
pub mod debruijn;
pub mod derangements;
pub mod descent_algebra;
pub mod designs;
pub mod multiset_partition_into_sets_ordered;
pub mod dlx;
pub mod dyck_word;
pub mod enumeration;
pub mod interval_posets;
pub mod fully_packed_loop;
pub mod gelfand_tsetlin;
pub mod fully_commutative_elements;
pub mod free_prelie_algebra;
pub mod grossman_larson;
pub mod growth_diagram;
pub mod hall_polynomial;
pub mod expnums;
pub mod group_action;
pub mod finite_state_machine_generators;
pub mod integer_vectors;
pub mod integer_matrices;
pub mod kazhdan_lusztig;
pub mod knutson_tao;
pub mod integer_lists;
pub mod ordered_tree;
pub mod parking_function;
pub mod partitions;
pub mod partition_shifting_algebras;
pub mod path_tableaux;
pub mod perfect_matching;
pub mod permutations;
pub mod plane_partition;
pub mod parallelogram_polyomino;
pub mod permutation_simd;
pub mod pipe_dream;
pub mod posets;
pub mod q_analogue;
pub mod ranking;
pub mod t_sequences;
pub mod recurrence_sequences;
pub mod regular_sequences;
pub mod ribbon_tableau;
pub mod restricted_growth;
pub mod schubert;
pub mod set_partition;
pub mod set_system;
pub mod sidon_sets;
pub mod similarity_class_type;
pub mod six_vertex_model;
pub mod skew_partition;
pub mod specht_module;
pub mod species;
pub mod subset;
pub mod subword;
pub mod subword_complex;
pub mod superpartitions;
pub mod symmetric_group_representations;
pub mod super_tableaux;
pub mod tableaux;
pub mod tableau_tuple;
pub mod tamari;
pub mod triangles_fhm;
pub mod tamari_lattice;
pub mod tiling;
pub mod tuple;
pub mod vector_partition;
pub mod word;
pub mod words;
pub mod wreath_product;
pub mod ncsym;
pub mod key_polynomial;

// MAGMA port (Wave 1): Chapters 92, 144, 145, 147, 148.
pub mod ch144_enumerative;
pub mod plactic_monoid;
pub mod rsk;
pub mod young_representations;
pub mod incidence_structure;
pub mod hadamard;
pub mod factorials;
pub mod partition_number;
mod machine_word;

pub use affine_permutations::{AffinePermutation, CoxeterType};
pub use baxter_permutations::{baxter_permutations, count_baxter_permutations, is_baxter};
pub use bijectionist::{
    all_k_subsets, all_ordered_pairs, all_unordered_pairs, burnside_count, BijectionFinder,
    BijectiveCorrespondence, GroupAction, OrbitStructure, PermutationActionOnMultiset,
    PermutationActionOnPairs, PermutationActionOnSequence, PermutationActionOnSet,
};
pub use backtrack::{BacktrackFn, BacktrackProblem, Backtracker};
pub use combinations::{combinations, Combination};
pub use fully_commutative_elements::{
    count_fc_elements_by_length, enumerate_fc_elements, CoxeterMatrix, FullyCommutativeElement,
    HeapPoset, ReducedWord as FCReducedWord, SimpleCoxeterMatrix,
};
pub use partitions::{
    count_partitions_with_max_parts, number_of_standard_tableaux, partition_count, partitions,
    partitions_with_distinct_parts, partitions_with_even_parts, partitions_with_exact_parts,
    partitions_with_max_part,
    partitions_with_max_parts, partitions_with_min_part, partitions_with_odd_parts, Partition,
    PartitionTuple,
};
pub use permutations::{all_permutations, Permutation};
pub use permutation_simd::{
    batch_compose_simd, compose_simd, cycles_simd, inverse_simd, power_simd, simd_available,
    simd_info,
};
pub use posets::Poset;
pub use specht_module::{
    GarnirElement, GarnirSet, Polytabloid, PolytabloidSum, SpechtModule, Tabloid,
};
pub use tableaux::{
    dual_robinson_schensted, hecke_insertion, mixed_insertion,
    robinson_schensted, rs_insert, standard_tableaux, Tableau,
};
pub use super_tableaux::{
    standard_super_tableaux, super_semistandard_tableaux, SuperTableau, SuperTableauEntry,
};
pub use tuple::{tuples as tuple_tuples, Tuple, TupleIterator};
pub use growth_diagram::{
    growth_sequence, growth_to_p_tableau, growth_to_q_tableau, GrowthDiagram,
};

// Re-export new modules
pub use binary_words::{
    all_binary_words, binary_words_with_weight, lyndon_words, lyndon_words_with_weight,
    necklaces, necklaces_with_weight, BinaryWord,
};
pub use composition::{
    compositions, compositions_k, integer_vectors_weighted, integer_vectors_weighted_dp,
    signed_compositions, signed_compositions_k, Composition, SignedComposition,
    WeightedIntegerVector,
};
pub use composition_tableau::{
    count_tableaux_with_descent_set, standard_composition_tableaux, CompositionTableau,
};
pub use designs::{
    are_latin_squares_orthogonal, is_steiner_system, mutually_orthogonal_latin_squares,
    t_design_index, BlockDesign, DesignAutomorphism, DesignCheckError, DifferenceSet,
    HadamardMatrix, OrthogonalArray, SteinerSystem, DESIGN_CHECK_LIMIT,
};
pub use dlx::DancingLinks;
pub use dyck_word::{dyck_words, nu_dyck_words, BounceStats, DyckWord, NuDyckWord};
pub use enumeration::{
    binary_to_gray, cartesian_product, gray_code_change_bit, gray_code_rank, gray_code_sequence,
    gray_code_unrank, gray_to_binary, stars_and_bars, tuples, weak_compositions,
    BinaryGrayCodeIterator, CartesianProduct, CombinationGrayCodeIterator, CompositionIterator, Enumerable, GrayCodeIterator,
    InfiniteCartesianProduct, LazyEnumerator, PartitionIterator, PermutationGrayCodeIterator,
    RevolvingDoorIterator,
};
pub use finite_state_machine_generators::{
    Automaton, AutomatonGenerators, FSMState, FSMTransition, Transducer, TransducerGenerators,
};
pub use gelfand_tsetlin::{
    gelfand_tsetlin_patterns, GelfandTsetlinIterator, GelfandTsetlinPattern,
};
pub use group_action::{
    burnside_lemma, count_fixed_colorings, cycle_index, enumerate_distinct_colorings,
    polya_enumeration, PermutationGroup,
};
pub use free_prelie_algebra::{PreLieAlgebra, RootedTree as PreLieRootedTree};
pub use integer_matrices::{
    count_integer_matrices, integer_matrices, integer_matrices_bounded, IntegerMatrix,
};
pub use perfect_matching::{perfect_matchings, PerfectMatching};
pub use pipe_dream::{PipeDream as PipeDreamType, PipeDreamBuilder};
pub use ranking::{CombinationRank, PermutationRank, Rankable, RankingTable};
pub use recurrence_sequences::{
    solve_binary_recurrence, BinaryRecurrence, LinearRecurrence, RecurrenceSequence,
};
pub use regular_sequences::{
    baum_sweet, characteristic_sequence, detect_periodicity, fibonacci_word,
    multiplicatively_independent, paperfolding, rudin_shapiro, sequence_product, sequence_sum,
    thue_morse, verify_cobham, AutomaticSequenceWrapper, CobhamResult, MorphicSequence,
    RegularSequence,
};
pub use set_partition::{
    bell_number_optimized, bell_numbers_up_to, set_partition_iterator, set_partitions,
    SetPartition, SetPartitionIterator,
};
pub use set_system::SetSystem;
// TODO: Fix missing module
// pub use multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
pub use word::{
    abelian_complexity, boyer_moore_search, christoffel_word, factor_complexity,
    from_cfl_factorization, from_standard_factorization, is_cfl_factorization, kmp_search,
    lyndon_factorization, lyndon_words as general_lyndon_words, lyndon_words_up_to,
    standard_lyndon_factorization, sturmian_word, AutomaticSequence, Morphism, Word,
};
pub use subset::{
    all_subsets, count_k_subsets, k_subset_iterator, k_subsets,
    max_pairwise_disjoint_k_subsets, pairwise_disjoint_family_iterator,
    pairwise_disjoint_k_subsets, subset_iterator, KSubsetIterator,
    PairwiseDisjointFamilyIterator, Subset, SubsetIterator,
};
pub use subword_complex::{ReducedWord as SubwordReducedWord, SubwordComplex as SubwordComplexType};
pub use q_analogue::{
    gaussian_polynomial, q_binomial, q_binomial_eval, q_factorial, q_integer, q_multinomial,
};
pub use hall_polynomial::{
    count_submodules, hall_littlewood_p, hall_polynomial, kostka_foulkes,
};
pub use t_sequences::{
    t_bell, t_binomial, t_binomial_eval, t_catalan, t_eulerian, t_factorial, t_fibonacci,
    t_integer, t_lucas, t_multinomial, t_stirling_second,
};
pub use wreath_product::{all_colored_permutations, ColoredPermutation};
pub use schubert::{
    schubert_polynomial, monk_rule, monk_rule_expansion, DividedDifference,
};
pub use cyclic_sieving::{CyclicAction, CyclicSievingTriple, FunctionAction};
pub use plane_partition::{
    count_cyclically_symmetric_plane_partitions, count_plane_partitions_in_box,
    count_self_complementary_plane_partitions, count_totally_symmetric_plane_partitions,
    count_transpose_complement_plane_partitions, plane_partitions, plane_partitions_in_box,
    PlanePartition,
};
pub use parallelogram_polyomino::{
    count_parallelogram_polyominoes, parallelogram_polyominoes,
    parallelogram_polyominoes_with_bounce_statistic, BouncePath, ParallelogramPolyomino, Point,
};
pub use ordered_tree::{OrderedTree, OrderedTreeNode, PreorderIterator};
pub use grossman_larson::{
    all_rooted_trees, GrossmanLarsonElement, RootedTree as GLRootedTree,
};
pub use derangements::{
    all_derangements, count_derangements, count_derangements_recurrence, is_derangement,
};
pub use descent_algebra::{
    count_permutations_with_descent_set, DescentAlgebraElement, DescentSet,
};
pub use superpartitions::{
    count_superpartitions, strict_superpartitions, superpartitions, superpartitions_with_k_parts,
    superpartitions_with_m_circled, SuperPart, SuperPartition,
};
pub use integer_vectors::{
    count_integer_vectors_in_box, count_integer_vectors_with_sum, integer_vector_sum_iter,
    integer_vectors_in_box, integer_vectors_with_l1_norm_bounded, integer_vectors_with_sum,
    integer_vectors_with_weighted_sum, IntegerVector, IntegerVectorSumIter,
};
pub use vector_partition::{
    count_vector_partitions_exact_parts, count_vector_partitions_max_parts,
    fast_vector_partitions, fast_vector_partitions_with_max_part, vector_partitions,
    vector_partitions_with_max_part, VectorPartition,
};
pub use partition_shifting_algebras::{
    ShiftingSequence, ShiftingSequenceSpace, ShiftingOperatorAlgebra, ShiftingOperatorElement,
};
pub use dendriform_algebra::{
    dimension as dendriform_dimension, free_dendriform_basis, DendriFormElement,
};
pub use tamari_lattice::{all_binary_trees, BinaryTree, TamariLattice};
pub use symmetric_group_representations::{
    all_irreducible_representations, conjugacy_class_size, decompose_character,
    murnaghan_nakayama, permutation_cycle_type, IrreducibleRepresentation,
    SymmetricGroupCharacterTable,
};
pub use alternating_sign_matrix::{
    all_asms, asm_count, asm_count_by_inversions, asm_q_enumeration, razumov_stroganov_polynomial,
    razumov_stroganov_refined_count, AlternatingSignMatrix,
};
pub use fully_packed_loop::{
    all_asms as all_fpls, count_asms, AlternatingSignMatrix as FPLAlternatingSignMatrix,
    FullyPackedLoop, LinkPattern,
};
pub use six_vertex_model::{
    all_dwbc_configurations, count_dwbc_configurations, from_asm as six_vertex_from_asm,
    partition_function as six_vertex_partition_function,
    weighted_partition_function as six_vertex_weighted_partition_function, SixVertexConfiguration,
    VertexType,
};
pub use ncsym::{
    coarse_powersum, deformed_coarse_powersum, elementary, homogeneous, matchings, monomial,
    nesting, powersum, ribbon_schur, supercharacter, x_basis, NCSymBasis, NCSymDual,
    NCSymFunction, RibbonSchurFunction,
};
pub use triangles_fhm::{Triangle, MTriangle, HTriangle, FTriangle, GammaTriangle};
pub use key_polynomial::{
    key_polynomial, KeyMonomial, KeyPolynomial, WeakComposition,
};
pub use expnums::{EGF, ExpNum};
pub use tiling::{
    build_transfer_matrix, count_domino_tilings, count_monomino_tilings,
    count_polyomino_tilings, count_tilings, ColumnState, Polyomino,
};
pub use debruijn::{
    debruijn_graph_path, debruijn_prefer_one, debruijn_prefer_zero, debruijn_sequence,
    debruijn_sequence_binary, universal_cycle_permutations, universal_cycle_subsets,
    DeBruijnSequence,
};
pub use path_tableaux::{
    ballot_sequences, count_dyck_paths, count_lattice_paths_ne, dyck_path_to_partition,
    dyck_paths, lattice_paths_ne, partition_to_dyck_path, LatticePath, PathTableau, Step,
};
pub use interval_posets::{
    interval_representation, is_interval_order, is_semiorder, Interval, IntervalPoset,
};
pub use ribbon_tableau::{
    fermionic_formula, kostka_foulkes_ribbon, ribbon_tableaux, RibbonDecomposition, RibbonTableau as RibbonTableauType,
};
pub use subword::{
    all_subwords, count_distinct_subwords, is_subword, reduced_pipe_dreams, subword_order,
    subword_positions, PipeDream as SubwordPipeDream, SubwordComplex as SubwordComplexFromSubword,
};
pub use skew_partition::{
    ribbon_shaped_tableaux, RibbonTableau as SkewRibbonTableau, SkewPartition, SkewTableau,
};
pub use similarity_class_type::{
    centralizer_algebra_dim, centralizer_group_cardinality, dictionary_from_generator,
    ext_orbit_centralizers, ext_orbits, fq, input_parsing,
    invariant_subspace_generating_function, matrix_centralizer_cardinalities,
    matrix_centralizer_cardinalities_length_two, matrix_similarity_classes,
    matrix_similarity_classes_length_two, order_of_general_linear_group, primitives,
    PrimarySimilarityClassType, PrimarySimilarityClassTypes, SimilarityClassType,
    SimilarityClassTypes,
};
pub use kazhdan_lusztig::{
    bruhat_interval, bruhat_le, bruhat_poset, kazhdan_lusztig_polynomial, length, r_polynomial,
};
pub use constellation::{
    cube_constellation, tetrahedron_constellation, Constellation, ConstellationEncoding,
};
pub use parking_function::{
    count_non_decreasing_parking_functions, count_parking_functions,
    non_decreasing_parking_functions, non_decreasing_parking_functions_with_area,
    parking_functions, NonDecreasingParkingFunction, ParkingFunction,
};
pub use cluster_complex::{
    associahedron, cluster_complex_type_a, AlmostPositiveRoot, Cluster, ClusterCartanType,
    ClusterComplex, GeneralizedAssociahedron,
};

// MAGMA port (Wave 1) re-exports.
pub use ch144_enumerative::{
    bernoulli_polynomial, eval_poly, fibonacci_signed, generalized_fibonacci_number,
    lucas_signed, number_of_permutations,
};
pub use plactic_monoid::{
    elementary_knuth_moves, is_knuth_equivalent, knuth_class, PlacticElement, PlacticIntegerMonoid,
};
pub use rsk::{inverse_rsk, matrix_to_biword, rsk, rsk_biword};
pub use young_representations::{
    hook_length, hook_length_dimension, seminormal_representation, seminormal_transposition,
    symmetric_character_value,
};
pub use incidence_structure::{DesignParameters, IncidenceStructure};
pub use factorials::{
    falling_factorial_in, falling_factorial_integer, rising_factorial_in, rising_factorial_integer,
};
pub use partition_number::{number_of_partitions, NUMBER_OF_PARTITIONS_MAX_N};
pub use hadamard::{
    hadamard_3_design, hadamard_invariant, is_hadamard, is_hadamard_equivalent, paley_type_i,
    paley_type_ii,
};

// Core combinatorial functions (factorials, Stirling numbers, etc.) defined in this module.
// MAGMA Handbook ch. 144 §144.2 (digest rustmath-lava/docs/Chapter_144_Enumerative_Combinatorics.md:34-53)
// and ch. 18 §18.8 (Chapter_18_Ring_of_Integers.md:261-275): Factorial, Binomial, Multinomial,
// Catalan, Fibonacci, Lucas, StirlingFirst, StirlingSecond, Bell, EulerianNumber. The Motzkin,
// Schroder and Narayana numbers are not MAGMA intrinsics (OEIS A001006, A006318, A001263).

use rustmath_integers::{Integer, SageInteger};
use std::collections::VecDeque;

/// MAGMA `Factorial(n)` (ch. 144 §144.2, ch. 18 §18.8): n!.
///
/// A balanced product of the factors 2..=n, packed into u64 words first (von zur Gathen and
/// Gerhard, *Modern Computer Algebra*, §10.1; [`factorials`]).
pub fn factorial(n: u32) -> Integer {
    factorials::product_range_u64(2, u64::from(n))
}

/// MAGMA `Binomial(n, r)` (ch. 144 §144.2, ch. 18 §18.8): C(n, k) = n! / (k! (n-k)!), and 0
/// for k > n.
///
/// Delegates to `SageInteger::binomial` (Goetgheluck's prime-power splitting in
/// rustmath-integers).
pub fn binomial(n: u32, k: u32) -> Integer {
    <Integer as SageInteger>::binomial(&Integer::from(n), &Integer::from(k))
}

/// MAGMA `Multinomial(n, [k_1, ..., k_m])` (ch. 144 §144.2, ch. 18 §18.8):
/// n! / (k_1! k_2! ... k_m!) when k_1 + ... + k_m = n.
///
/// Returns 0 when the parts do not sum to n (the coefficient of a monomial of another
/// degree). The sum is formed in u128, so it never wraps. The value is computed as the product
/// of binomials prod_i C(s_i, k_i), s_i = k_(1) + ... + k_(i), with the largest k first, so
/// each binomial runs over its lower index and the total window is n - max k.
///
/// # Panics
///
/// Panics (in `SageInteger::binomial`) if a factor has more than about 2^32 bits.
pub fn multinomial(n: u32, ks: &[u32]) -> Integer {
    let sum: u128 = ks.iter().map(|&k| u128::from(k)).sum();
    if sum != u128::from(n) {
        return Integer::zero();
    }
    let mut sorted: Vec<u64> = ks.iter().map(|&k| u64::from(k)).collect();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let mut s = 0u64;
    let parts: Vec<Integer> = sorted
        .into_iter()
        .map(|k| {
            s += k;
            <Integer as SageInteger>::binomial(&Integer::from(s), &Integer::from(k))
        })
        .collect();
    factorials::product_tree(parts)
}

/// MAGMA `Catalan(n)` (ch. 144 §144.2): the n-th Catalan number
/// C_n = C(2n, n) / (n + 1), which satisfies MAGMA's C_(n+1) = C_n (4n + 2) / (n + 2).
///
/// 2n and n + 1 are formed in u64, so they cannot overflow for any u32 n.
///
/// # Panics
///
/// Panics (in `SageInteger::binomial`) if the result has more than about 2^32 bits.
pub fn catalan(n: u32) -> Integer {
    let n = u64::from(n);
    <Integer as SageInteger>::binomial(&Integer::from(2 * n), &Integer::from(n)) / Integer::from(n + 1)
}

/// MAGMA `Fibonacci(n)` (ch. 144 §144.2) for n >= 0: F_0 = 0, F_1 = 1.
///
/// By doubling, F_2k = F_k (2 F_(k+1) - F_k) and F_(2k+1) = F_k^2 + F_(k+1)^2 (Knuth, TAOCP
/// vol. 1, §1.2.8): O(M(n)) work.
pub fn fibonacci(n: u32) -> Integer {
    ch144_enumerative::fibonacci_pair(u64::from(n)).0
}

/// MAGMA `Lucas(n)` (ch. 144 §144.2) for n >= 0: L_0 = 2, L_1 = 1, L_n = L_(n-1) + L_(n-2).
///
/// L_n = 2 F_(n+1) - F_n, with the Fibonacci pair by doubling.
pub fn lucas(n: u32) -> Integer {
    let (f, f1) = ch144_enumerative::fibonacci_pair(u64::from(n));
    f1.clone() + f1 - f
}

/// The falling factorial (n)_k = n (n-1) (n-2) ... (n-k+1), and 0 when k > n.
///
/// The u32 form of [`falling_factorial_integer`], which takes any integer n.
pub fn falling_factorial(n: u32, k: u32) -> Integer {
    falling_factorial_integer(&Integer::from(n), u64::from(k))
}

/// The rising factorial n (n+1) (n+2) ... (n+k-1): the Pochhammer symbol (n)_k of MAGMA ch. 49
/// (`HypergeometricSeries`) and Sage `rising_factorial`.
///
/// The u32 form of [`rising_factorial_integer`], which takes any integer n. Nothing is added
/// in u32, so n + k - 1 may exceed u32::MAX.
pub fn rising_factorial(n: u32, k: u32) -> Integer {
    rising_factorial_integer(&Integer::from(n), u64::from(k))
}

/// Which row recurrence [`band_rows`] runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowKind {
    /// [i j] = (i-1) [i-1 j] + [i-1 j-1]
    StirlingFirst,
    /// {i j} = j {i-1 j} + {i-1 j-1}
    StirlingSecond,
    /// <i j> = (j+1) <i-1 j> + (i-j) <i-1 j-1>
    Eulerian,
}

/// The row recurrence of [`stirling_first`], [`stirling_second`] and [`eulerian`] (Graham,
/// Knuth, Patashnik, *Concrete Mathematics*, 2nd ed., §6.1, §6.2), kept in a buffer of at most
/// (band width + 1) entries. Returns the (n, k) entry and the largest buffer length (the
/// length is for the unit test that pins the memory bound).
///
/// Needs 1 <= k < n (Stirling) or 1 <= k <= n - 1 - k (Eulerian). (n, k) depends on (i, j)
/// only if k - (n - i) <= j <= k, so row i needs only j in lo(i)..=hi(i) with
/// lo(i) = max(1, k + i - n) and hi(i) = min(i, k) (Eulerian: min(i - 1, k)); the recurrence
/// at j = lo(i) also reads the entry j - 1 of row i - 1. `d[t]` holds the row value at
/// j = base + t; positions past `d.len()` stand for 0. The entries are updated for descending
/// j, so every entry read still holds row i - 1. Memory: at most (w + 1) entries with
/// w = min(k, n - k + 1), each at most the final value; work O(n w) big operations.
fn band_rows(n: u64, k: u64, kind: RowKind) -> (Integer, usize) {
    // Stirling starts from row 0 = [1] (j = 0); Eulerian from row 1, A(1, 0) = 1.
    let first_row = if kind == RowKind::Eulerian { 1 } else { 0 };
    let mut d: VecDeque<Integer> = VecDeque::new();
    d.push_back(Integer::one());
    let mut base = 0u64;
    let mut max_len = d.len();
    for i in first_row + 1..=n {
        let lo = (k + i).saturating_sub(n).max(1);
        let hi = if kind == RowKind::Eulerian { (i - 1).min(k) } else { i.min(k) };
        while base + 1 < lo {
            d.pop_front();
            base += 1;
        }
        while base + (d.len() as u64) <= hi {
            d.push_back(Integer::zero());
        }
        max_len = max_len.max(d.len());
        let mut j = hi;
        while j >= lo {
            let t = (j - base) as usize;
            let v = match kind {
                RowKind::StirlingFirst => &(&Integer::from(i - 1) * &d[t]) + &d[t - 1],
                RowKind::StirlingSecond => &(&Integer::from(j) * &d[t]) + &d[t - 1],
                RowKind::Eulerian => {
                    &(&Integer::from(j + 1) * &d[t]) + &(&Integer::from(i - j) * &d[t - 1])
                }
            };
            d[t] = v;
            j -= 1;
        }
        // [i 0] = {i 0} = 0 for i >= 1; the Eulerian A(i, 0) = 1 is never rewritten.
        if base == 0 && kind != RowKind::Eulerian {
            d[0] = Integer::zero();
        }
    }
    (d[(k - base) as usize].clone(), max_len)
}

/// MAGMA `StirlingFirst(n, k)` (ch. 144 §144.2, ch. 18 §18.8): the unsigned Stirling number of
/// the first kind [n k], the number of permutations of n elements with exactly k cycles.
///
/// The row recurrence [i j] = (i-1) [i-1 j] + [i-1 j-1] over the band of entries that reach
/// (n, k), in a buffer of at most min(k, n - k + 1) + 1 entries.
pub fn stirling_first(n: u32, k: u32) -> Integer {
    if k > n {
        return Integer::zero();
    }
    if k == n {
        return Integer::one();
    }
    if k == 0 {
        return Integer::zero();
    }
    band_rows(u64::from(n), u64::from(k), RowKind::StirlingFirst).0
}

/// MAGMA `StirlingSecond(n, k)` (ch. 144 §144.2, ch. 18 §18.8): the Stirling number of the
/// second kind {n k}, the number of ways to partition n elements into k non-empty subsets.
///
/// The row recurrence {i j} = j {i-1 j} + {i-1 j-1} over the band of entries that reach
/// (n, k), in a buffer of at most min(k, n - k + 1) + 1 entries.
pub fn stirling_second(n: u32, k: u32) -> Integer {
    if k > n {
        return Integer::zero();
    }
    if k == n {
        return Integer::one();
    }
    if k == 0 {
        return Integer::zero();
    }
    band_rows(u64::from(n), u64::from(k), RowKind::StirlingSecond).0
}

/// MAGMA `Bell(n)` (ch. 144 §144.2, ch. 18 §18.8): the number of partitions of a set of n
/// elements, B(n) = sum_k StirlingSecond(n, k).
///
/// By the Bell triangle (A. C. Aitken, *A problem in combinations*, Math. Notes Edinburgh 28
/// (1933) xviii-xxiii; OEIS A011971): row i starts with the last entry of row i - 1, and each
/// next entry is its left neighbour plus the entry above it; B(n) is the first entry of row n.
/// One vector updated in place: O(n^2) additions of numbers of at most n log2 n bits.
pub fn bell_number(n: u32) -> Integer {
    let mut row = vec![Integer::one()];
    for _ in 0..n {
        bell_triangle_step(&mut row);
    }
    row.swap_remove(0)
}

/// One row of the Bell triangle, in place: row i - 1 becomes row i.
pub(crate) fn bell_triangle_step(row: &mut Vec<Integer>) {
    let mut cur = row[row.len() - 1].clone();
    for entry in row.iter_mut() {
        let next = &cur + &*entry;
        *entry = std::mem::replace(&mut cur, next);
    }
    row.push(cur);
}

/// MAGMA `EulerianNumber(n, k)` (ch. 144 §144.2): A(n, k), the number of permutations of
/// {1, 2, ..., n} with exactly k descents (equivalently, k ascents).
///
/// The recurrence A(n, k) = (k+1) A(n-1, k) + (n-k) A(n-1, k-1) (Concrete Mathematics §6.2)
/// over the band of entries that reach (n, k), after the symmetry A(n, k) = A(n, n-1-k).
///
/// Properties:
/// - A(n, 0) = 1 (identity permutation has 0 descents)
/// - A(n, k) = 0 for k >= n
/// - Sum over k of A(n, k) = n!
pub fn eulerian(n: u32, k: u32) -> Integer {
    if n == 0 {
        return if k == 0 { Integer::one() } else { Integer::zero() };
    }
    if k >= n {
        return Integer::zero();
    }
    let k = k.min(n - 1 - k);
    if k == 0 {
        return Integer::one();
    }
    band_rows(u64::from(n), u64::from(k), RowKind::Eulerian).0
}

/// Compute Narayana number N(n, k)
///
/// Number of expressions containing n pairs of parentheses which are correctly matched
/// and which contain k distinct nestings. Equivalently, the number of Dyck paths of
/// length 2n with exactly k peaks.
///
/// The Narayana numbers are given by:
/// N(n, k) = (1/n) * C(n, k) * C(n, k-1)
///
/// Properties:
/// - N(n, 0) = 0 for n > 0
/// - N(n, 1) = 1 for n >= 1
/// - Sum over k of N(n, k) = C_n (nth Catalan number)
pub fn narayana(n: u32, k: u32) -> Integer {
    if n == 0 {
        return if k == 0 { Integer::one() } else { Integer::zero() };
    }
    if k == 0 || k > n {
        return Integer::zero();
    }
    if k == 1 {
        return Integer::one();
    }

    // N(n, k) = (1/n) * C(n, k) * C(n, k-1)
    let binom_n_k = binomial(n, k);
    let binom_n_k_minus_1 = binomial(n, k - 1);

    (binom_n_k * binom_n_k_minus_1) / Integer::from(n)
}

/// Compute central Delannoy number D(n)
///
/// Number of paths from (0, 0) to (n, n) using steps (1, 0), (0, 1), and (1, 1).
/// Also the number of ways to place n non-attacking rooks on an n×n chessboard
/// with some squares removed.
///
/// The central Delannoy numbers satisfy:
/// D(n) = Sum_{k=0}^{n} C(n, k)^2 * 2^k
///
/// First few values: 1, 3, 13, 63, 321, 1683, ...
pub fn delannoy_central(n: u32) -> Integer {
    let mut sum = Integer::zero();

    for k in 0..=n {
        let binom_sq = binomial(n, k).clone();
        let term = binom_sq.clone() * binom_sq * Integer::from(2).pow(k);
        sum = sum + term;
    }

    sum
}

/// Compute Delannoy number D(m, n)
///
/// Number of paths from (0, 0) to (m, n) using steps (1, 0), (0, 1), and (1, 1).
///
/// The Delannoy numbers satisfy:
/// D(m, n) = Sum_{k=0}^{min(m,n)} C(m, k) * C(n, k) * 2^k
pub fn delannoy(m: u32, n: u32) -> Integer {
    let min_mn = m.min(n);
    let mut sum = Integer::zero();

    for k in 0..=min_mn {
        let term = binomial(m, k) * binomial(n, k) * Integer::from(2).pow(k);
        sum = sum + term;
    }

    sum
}

/// Compute nth Motzkin number M(n)
///
/// Number of ways to draw non-intersecting chords between n points on a circle,
/// where not all points need to be paired. Also the number of lattice paths from
/// (0, 0) to (n, 0) using steps (1, 1), (1, -1), and (1, 0) that never go below y=0.
///
/// The Motzkin numbers satisfy the convolution M(n) = M(n-1) + Sum_{k=0}^{n-2} M(k) M(n-2-k).
/// They are computed by the three-term recurrence (i+2) M_i = (2i+1) M_(i-1) + 3(i-1) M_(i-2),
/// M_0 = M_1 = 1 (R. Donaghey, L. W. Shapiro, *Motzkin numbers*, J. Combin. Theory Ser. A 23
/// (1977) 291-301; OEIS A001006): two rolling values, each step two one-word
/// multiplications, an addition and an exact one-word division.
///
/// First few values: 1, 1, 2, 4, 9, 21, 51, 127, ...
pub fn motzkin(n: u32) -> Integer {
    let (mut prev, mut cur) = (Integer::one(), Integer::one()); // M_0, M_1
    if n == 0 {
        return prev;
    }
    for i in 2..=u64::from(n) {
        let next = (&(&Integer::from(2 * i + 1) * &cur) + &(&Integer::from(3 * (i - 1)) * &prev))
            / Integer::from(i + 2);
        prev = std::mem::replace(&mut cur, next);
    }
    cur
}

/// (3(2i - 1), i - 2, i + 1): the multipliers of (i+1) S_i = 3(2i-1) S_(i-1) - (i-2) S_(i-2),
/// i >= 2, in u64 (3(2i - 1) < 2^35 for i <= 2^32).
fn schroder_step_factors(i: u64) -> (u64, u64, u64) {
    (3 * (2 * i - 1), i - 2, i + 1)
}

/// Compute nth large Schröder number S(n)
///
/// Number of lattice paths from (0, 0) to (2n, 0) using steps (1, 1), (1, -1),
/// and (2, 0) that never go below the x-axis. Also counts the number of ways
/// to parenthesize a string of n+1 factors.
///
/// The large Schröder numbers satisfy the recurrence:
/// (n+1)*S(n) = 3*(2n-1)*S(n-1) - (n-2)*S(n-2),
/// run with two rolling values and u64 multipliers (OEIS A006318).
///
/// First few values: 1, 2, 6, 22, 90, 394, 1806, ...
pub fn schroder_large(n: u32) -> Integer {
    let (mut prev, mut cur) = (Integer::one(), Integer::from(2u32)); // S_0, S_1
    if n == 0 {
        return prev;
    }
    for i in 2..=u64::from(n) {
        let (a, b, c) = schroder_step_factors(i);
        let next = (&(&Integer::from(a) * &cur) - &(&Integer::from(b) * &prev)) / Integer::from(c);
        prev = std::mem::replace(&mut cur, next);
    }
    cur
}

/// Compute nth small Schröder number (or Schröder-Hipparchus number) s(n)
///
/// Related to large Schröder numbers by: s(n) = S(n) / 2 for n >= 1, and s(0) = 1
/// The small Schröder numbers count lattice paths and bracket structures.
///
/// First few values: 1, 1, 3, 11, 45, 197, 903, ...
pub fn schroder_small(n: u32) -> Integer {
    if n == 0 {
        return Integer::one();
    }

    // s(n) = S(n) / 2 for n >= 1
    schroder_large(n) / Integer::from(2)
}

/// A Latin square of order n
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatinSquare {
    /// The square as a 2D grid
    grid: Vec<Vec<usize>>,
    n: usize,
}

impl LatinSquare {
    /// Create a new Latin square
    pub fn new(grid: Vec<Vec<usize>>) -> Option<Self> {
        let n = grid.len();

        if n == 0 {
            return Some(LatinSquare { grid, n: 0 });
        }

        // Check dimensions
        for row in &grid {
            if row.len() != n {
                return None;
            }
        }

        // Check that each row and column contains each symbol exactly once
        for i in 0..n {
            // Check row i
            let mut row_symbols = vec![false; n];
            for j in 0..n {
                if grid[i][j] >= n {
                    return None;
                }
                if row_symbols[grid[i][j]] {
                    return None;
                }
                row_symbols[grid[i][j]] = true;
            }

            // Check column i
            let mut col_symbols = vec![false; n];
            for j in 0..n {
                if col_symbols[grid[j][i]] {
                    return None;
                }
                col_symbols[grid[j][i]] = true;
            }
        }

        Some(LatinSquare { grid, n })
    }

    /// Get the grid
    pub fn grid(&self) -> &[Vec<usize>] {
        &self.grid
    }

    /// Get the order of the square
    pub fn order(&self) -> usize {
        self.n
    }

    /// Get the element at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<usize> {
        self.grid.get(i)?.get(j).copied()
    }
}

/// Generate all Latin squares of order n (warning: grows very quickly!)
pub fn latin_squares(n: usize) -> Vec<LatinSquare> {
    if n == 0 {
        return vec![LatinSquare {
            grid: vec![],
            n: 0,
        }];
    }

    let mut result = Vec::new();
    let mut grid = vec![vec![0; n]; n];

    generate_latin_squares(&mut grid, 0, 0, n, &mut result);

    result
}

fn generate_latin_squares(
    grid: &mut Vec<Vec<usize>>,
    row: usize,
    col: usize,
    n: usize,
    result: &mut Vec<LatinSquare>,
) {
    if row == n {
        result.push(LatinSquare {
            grid: grid.clone(),
            n,
        });
        return;
    }

    let (next_row, next_col) = if col + 1 < n {
        (row, col + 1)
    } else {
        (row + 1, 0)
    };

    // Try each symbol
    for symbol in 0..n {
        // Check if symbol can be placed at (row, col)
        if can_place_latin(grid, row, col, symbol, n) {
            grid[row][col] = symbol;
            generate_latin_squares(grid, next_row, next_col, n, result);
            grid[row][col] = 0; // Reset for backtracking
        }
    }
}

fn can_place_latin(grid: &[Vec<usize>], row: usize, col: usize, symbol: usize, _n: usize) -> bool {
    // Check row
    for j in 0..col {
        if grid[row][j] == symbol {
            return false;
        }
    }

    // Check column
    for i in 0..row {
        if grid[i][col] == symbol {
            return false;
        }
    }

    // Also check the rest of the current row and column for future consistency
    // (This is optional but helps prune search space)
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schroder_step_factors_do_not_wrap() {
        // HEAD formed 3 * (2i - 1) as u32: 5 at i = 715827884 and 3 at i = 2147483649.
        assert_eq!(schroder_step_factors(715827884).0, 4294967301);
        assert_eq!(schroder_step_factors(2147483649).0, 12884901891);
        assert_eq!(schroder_step_factors(2), (9, 0, 3));
        assert_eq!(schroder_step_factors(4294967296), (25769803773, 4294967294, 4294967297));
    }

    #[test]
    fn band_rows_stay_band_sized() {
        for (n, k) in [(100_000u64, 99_999u64), (2000, 1990), (500, 250), (500, 1)] {
            let w = k.min(n - k + 1) as usize;
            let (_, len1) = band_rows(n, k, RowKind::StirlingFirst);
            let (_, len2) = band_rows(n, k, RowKind::StirlingSecond);
            assert!(len1 <= w + 1, "stirling_first({n}, {k}): {len1} > {}", w + 1);
            assert!(len2 <= w + 1, "stirling_second({n}, {k}): {len2} > {}", w + 1);
            // Eulerian runs on k' = min(k, n - 1 - k) (the symmetry); k' = 0 needs no rows.
            let ke = k.min(n - 1 - k);
            if ke >= 1 {
                let we = ke.min(n - ke) as usize;
                let (_, len3) = band_rows(n, ke, RowKind::Eulerian);
                assert!(len3 <= we + 1, "eulerian({n}, {ke}): {len3} > {}", we + 1);
            }
        }
        // Both are C(n, 2) at k = n - 1: gp binomial(10^5, 2) = 4999950000.
        assert_eq!(stirling_second(100_000, 99_999), Integer::from(4999950000u64));
        assert_eq!(stirling_first(100_000, 99_999), Integer::from(4999950000u64));
        // A(n, n - 2) = A(n, 1) = 2^n - n - 1, which has n bits for n >= 3.
        let e = eulerian(100_000, 99_998);
        assert_eq!(e.bit_length(), 100_000);
        assert_eq!(e, Integer::from(2).pow(100_000) - Integer::from(100_001u32));
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), Integer::from(1));
        assert_eq!(factorial(5), Integer::from(120));
        assert_eq!(factorial(10), Integer::from(3628800));
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2), Integer::from(10));
        assert_eq!(binomial(10, 5), Integer::from(252));
        assert_eq!(binomial(5, 0), Integer::from(1));
        assert_eq!(binomial(5, 5), Integer::from(1));
        assert_eq!(binomial(5, 6), Integer::from(0));
    }

    #[test]
    fn test_multinomial() {
        // 3!/(1!*1!*1!) = 6
        assert_eq!(multinomial(3, &[1, 1, 1]), Integer::from(6));

        // 5!/(2!*2!*1!) = 30
        assert_eq!(multinomial(5, &[2, 2, 1]), Integer::from(30));

        // Should be 0 if sum doesn't equal n
        assert_eq!(multinomial(5, &[2, 2]), Integer::from(0));
    }

    #[test]
    fn test_catalan() {
        // First few Catalan numbers: 1, 1, 2, 5, 14, 42, ...
        assert_eq!(catalan(0), Integer::from(1));
        assert_eq!(catalan(1), Integer::from(1));
        assert_eq!(catalan(2), Integer::from(2));
        assert_eq!(catalan(3), Integer::from(5));
        assert_eq!(catalan(4), Integer::from(14));
        assert_eq!(catalan(5), Integer::from(42));
    }

    #[test]
    fn test_fibonacci() {
        // First few Fibonacci numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
        assert_eq!(fibonacci(0), Integer::from(0));
        assert_eq!(fibonacci(1), Integer::from(1));
        assert_eq!(fibonacci(2), Integer::from(1));
        assert_eq!(fibonacci(3), Integer::from(2));
        assert_eq!(fibonacci(4), Integer::from(3));
        assert_eq!(fibonacci(5), Integer::from(5));
        assert_eq!(fibonacci(8), Integer::from(21));
        assert_eq!(fibonacci(10), Integer::from(55));
    }

    #[test]
    fn test_lucas() {
        // First few Lucas numbers: 2, 1, 3, 4, 7, 11, 18, ...
        assert_eq!(lucas(0), Integer::from(2));
        assert_eq!(lucas(1), Integer::from(1));
        assert_eq!(lucas(2), Integer::from(3));
        assert_eq!(lucas(3), Integer::from(4));
        assert_eq!(lucas(4), Integer::from(7));
        assert_eq!(lucas(5), Integer::from(11));
        assert_eq!(lucas(6), Integer::from(18));
    }

    #[test]
    fn test_falling_factorial() {
        // 5!/(5-3)! = 5*4*3 = 60
        assert_eq!(falling_factorial(5, 3), Integer::from(60));
        assert_eq!(falling_factorial(10, 2), Integer::from(90));
        assert_eq!(falling_factorial(5, 0), Integer::from(1));
        assert_eq!(falling_factorial(5, 6), Integer::from(0));
    }

    #[test]
    fn test_rising_factorial() {
        // 5 * 6 * 7 = 210
        assert_eq!(rising_factorial(5, 3), Integer::from(210));
        assert_eq!(rising_factorial(10, 2), Integer::from(110));
        assert_eq!(rising_factorial(5, 0), Integer::from(1));
    }

    #[test]
    fn test_stirling_second() {
        // S(n, 0) = 0 for n > 0
        assert_eq!(stirling_second(5, 0), Integer::from(0));

        // S(n, 1) = 1
        assert_eq!(stirling_second(5, 1), Integer::from(1));

        // S(n, n) = 1
        assert_eq!(stirling_second(5, 5), Integer::from(1));

        // S(5, 2) = 15
        assert_eq!(stirling_second(5, 2), Integer::from(15));

        // S(5, 3) = 25
        assert_eq!(stirling_second(5, 3), Integer::from(25));
    }

    #[test]
    fn test_bell_number() {
        // First few Bell numbers: 1, 1, 2, 5, 15, 52, ...
        assert_eq!(bell_number(0), Integer::from(1));
        assert_eq!(bell_number(1), Integer::from(1));
        assert_eq!(bell_number(2), Integer::from(2));
        assert_eq!(bell_number(3), Integer::from(5));
        assert_eq!(bell_number(4), Integer::from(15));
        assert_eq!(bell_number(5), Integer::from(52));
    }

    #[test]
    fn test_stirling_first() {
        // s(n, 0) = 0 for n > 0
        assert_eq!(stirling_first(5, 0), Integer::from(0));

        // s(n, n) = 1
        assert_eq!(stirling_first(5, 5), Integer::from(1));

        // s(4, 2) = 11
        assert_eq!(stirling_first(4, 2), Integer::from(11));

        // s(5, 2) = 50
        assert_eq!(stirling_first(5, 2), Integer::from(50));

        // s(5, 3) = 35
        assert_eq!(stirling_first(5, 3), Integer::from(35));
    }

    #[test]
    fn test_latin_squares() {
        // Latin squares of order 1
        let squares1 = latin_squares(1);
        assert_eq!(squares1.len(), 1);

        // Latin squares of order 2: should be 2 squares
        // [[0,1],[1,0]] and [[1,0],[0,1]]
        let squares2 = latin_squares(2);
        assert_eq!(squares2.len(), 2);

        // Verify each is valid
        for square in &squares2 {
            assert_eq!(square.order(), 2);
        }
    }

    #[test]
    fn test_latin_square_validation() {
        // Valid 3x3 Latin square
        let valid = LatinSquare::new(vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]]);
        assert!(valid.is_some());

        // Invalid - repeated element in row
        let invalid = LatinSquare::new(vec![vec![0, 0, 1], vec![1, 2, 0], vec![2, 1, 2]]);
        assert!(invalid.is_none());

        // Invalid - repeated element in column
        let invalid2 = LatinSquare::new(vec![vec![0, 1, 2], vec![0, 2, 1], vec![2, 0, 1]]);
        assert!(invalid2.is_none());
    }

    #[test]
    fn test_eulerian() {
        // A(n, k) = 0 for k >= n
        assert_eq!(eulerian(5, 5), Integer::from(0));
        assert_eq!(eulerian(5, 6), Integer::from(0));

        // A(n, 0) = 1 (identity permutation has 0 descents)
        assert_eq!(eulerian(0, 0), Integer::from(1));
        assert_eq!(eulerian(5, 0), Integer::from(1));
        assert_eq!(eulerian(10, 0), Integer::from(1));

        // Known values for small n
        // A(3, 0) = 1, A(3, 1) = 4, A(3, 2) = 1
        assert_eq!(eulerian(3, 0), Integer::from(1));
        assert_eq!(eulerian(3, 1), Integer::from(4));
        assert_eq!(eulerian(3, 2), Integer::from(1));

        // A(4, 0) = 1, A(4, 1) = 11, A(4, 2) = 11, A(4, 3) = 1
        assert_eq!(eulerian(4, 0), Integer::from(1));
        assert_eq!(eulerian(4, 1), Integer::from(11));
        assert_eq!(eulerian(4, 2), Integer::from(11));
        assert_eq!(eulerian(4, 3), Integer::from(1));

        // A(5, 1) = 26, A(5, 2) = 66
        assert_eq!(eulerian(5, 1), Integer::from(26));
        assert_eq!(eulerian(5, 2), Integer::from(66));

        // Verify that sum of A(n, k) over k equals n!
        let mut sum = Integer::zero();
        for k in 0..5 {
            sum = sum + eulerian(5, k);
        }
        assert_eq!(sum, factorial(5));
    }

    #[test]
    fn test_narayana() {
        // N(0, 0) = 1
        assert_eq!(narayana(0, 0), Integer::from(1));

        // N(n, 0) = 0 for n > 0
        assert_eq!(narayana(1, 0), Integer::from(0));
        assert_eq!(narayana(5, 0), Integer::from(0));

        // N(n, k) = 0 for k > n
        assert_eq!(narayana(3, 5), Integer::from(0));

        // N(n, 1) = 1 for n >= 1
        assert_eq!(narayana(1, 1), Integer::from(1));
        assert_eq!(narayana(5, 1), Integer::from(1));
        assert_eq!(narayana(10, 1), Integer::from(1));

        // Known values
        // N(3, 1) = 1, N(3, 2) = 3, N(3, 3) = 1
        assert_eq!(narayana(3, 1), Integer::from(1));
        assert_eq!(narayana(3, 2), Integer::from(3));
        assert_eq!(narayana(3, 3), Integer::from(1));

        // N(4, 2) = 6, N(4, 3) = 6
        assert_eq!(narayana(4, 2), Integer::from(6));
        assert_eq!(narayana(4, 3), Integer::from(6));

        // Verify that sum of N(n, k) equals C_n (Catalan number)
        let mut sum = Integer::zero();
        for k in 1..=5 {
            sum = sum + narayana(5, k);
        }
        assert_eq!(sum, catalan(5));

        // Another verification for n=4
        let mut sum4 = Integer::zero();
        for k in 1..=4 {
            sum4 = sum4 + narayana(4, k);
        }
        assert_eq!(sum4, catalan(4));
    }

    #[test]
    fn test_delannoy_central() {
        // First few central Delannoy numbers: 1, 3, 13, 63, 321, 1683
        assert_eq!(delannoy_central(0), Integer::from(1));
        assert_eq!(delannoy_central(1), Integer::from(3));
        assert_eq!(delannoy_central(2), Integer::from(13));
        assert_eq!(delannoy_central(3), Integer::from(63));
        assert_eq!(delannoy_central(4), Integer::from(321));
        assert_eq!(delannoy_central(5), Integer::from(1683));
    }

    #[test]
    fn test_delannoy() {
        // D(m, n) should be symmetric
        assert_eq!(delannoy(3, 5), delannoy(5, 3));
        assert_eq!(delannoy(2, 4), delannoy(4, 2));

        // D(n, n) should equal central Delannoy number
        assert_eq!(delannoy(0, 0), delannoy_central(0));
        assert_eq!(delannoy(3, 3), delannoy_central(3));
        assert_eq!(delannoy(5, 5), delannoy_central(5));

        // D(0, n) = 1 (only one path along the edge)
        assert_eq!(delannoy(0, 0), Integer::from(1));
        assert_eq!(delannoy(0, 5), Integer::from(1));
        assert_eq!(delannoy(5, 0), Integer::from(1));

        // Known values
        assert_eq!(delannoy(2, 3), Integer::from(25));
        assert_eq!(delannoy(3, 4), Integer::from(129));
    }

    #[test]
    fn test_motzkin() {
        // First few Motzkin numbers: 1, 1, 2, 4, 9, 21, 51, 127, 323
        assert_eq!(motzkin(0), Integer::from(1));
        assert_eq!(motzkin(1), Integer::from(1));
        assert_eq!(motzkin(2), Integer::from(2));
        assert_eq!(motzkin(3), Integer::from(4));
        assert_eq!(motzkin(4), Integer::from(9));
        assert_eq!(motzkin(5), Integer::from(21));
        assert_eq!(motzkin(6), Integer::from(51));
        assert_eq!(motzkin(7), Integer::from(127));
        assert_eq!(motzkin(8), Integer::from(323));
    }

    #[test]
    fn test_schroder_large() {
        // First few large Schröder numbers: 1, 2, 6, 22, 90, 394, 1806
        assert_eq!(schroder_large(0), Integer::from(1));
        assert_eq!(schroder_large(1), Integer::from(2));
        assert_eq!(schroder_large(2), Integer::from(6));
        assert_eq!(schroder_large(3), Integer::from(22));
        assert_eq!(schroder_large(4), Integer::from(90));
        assert_eq!(schroder_large(5), Integer::from(394));
        assert_eq!(schroder_large(6), Integer::from(1806));
    }

    #[test]
    fn test_schroder_small() {
        // First few small Schröder numbers: 1, 1, 3, 11, 45, 197, 903
        assert_eq!(schroder_small(0), Integer::from(1));
        assert_eq!(schroder_small(1), Integer::from(1));
        assert_eq!(schroder_small(2), Integer::from(3));
        assert_eq!(schroder_small(3), Integer::from(11));
        assert_eq!(schroder_small(4), Integer::from(45));
        assert_eq!(schroder_small(5), Integer::from(197));
        assert_eq!(schroder_small(6), Integer::from(903));

        // Verify relationship: S(n) = 2*s(n) for n > 0
        for n in 1..=6 {
            assert_eq!(schroder_large(n), schroder_small(n) * Integer::from(2));
        }
    }

    #[test]
    fn test_counting_edge_cases() {
        // Test edge cases for all new functions

        // Eulerian with n=0
        assert_eq!(eulerian(0, 0), Integer::from(1));
        assert_eq!(eulerian(0, 1), Integer::from(0));

        // Narayana edge cases
        assert_eq!(narayana(1, 1), Integer::from(1));
        assert_eq!(narayana(1, 2), Integer::from(0));

        // Delannoy with zero
        assert_eq!(delannoy(0, 0), Integer::from(1));
        assert_eq!(delannoy_central(0), Integer::from(1));

        // Motzkin base cases
        assert_eq!(motzkin(0), Integer::from(1));
        assert_eq!(motzkin(1), Integer::from(1));

        // Schröder base cases
        assert_eq!(schroder_large(0), Integer::from(1));
        assert_eq!(schroder_small(0), Integer::from(1));
    }
}
