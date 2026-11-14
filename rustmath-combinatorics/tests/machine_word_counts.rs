//! Machine-word counts, ranks and loop bounds (design C1 §1.6, the sweep of the crate).
//!
//! MAGMA Handbook ch. 144 §144.2 and §144.3 (counts, subset ranks and successors; digest
//! `rustmath-lava/docs/Chapter_144_Enumerative_Combinatorics.md`), ch. 145 §145.4.9
//! `NumberOfStandardTableaux` (`Chapter_145_Partitions_Words_and_Young_Tableaux.md:299`),
//! ch. 58 `Order(g)` (`Chapter_58_Permutation_Groups.md:204`).
//!
//! Each public function here returns a machine word (or a value built from one). A value that
//! fits must be exact, and one that does not must be a documented panic, never a wrapped,
//! saturated or truncated value. Reference values are from PARI/GP 2.15; the gp command of
//! each is quoted next to it. Tests marked 64-bit run only where usize has 64 bits.

use rustmath_combinatorics::dyck_word::DyckWord;
use rustmath_combinatorics::grossman_larson::RootedTree;
use rustmath_combinatorics::integer_lists::{integer_lists, IntegerListConstraints};
use rustmath_combinatorics::multiset_partition_into_sets_ordered::OrderedMultisetPartitionIntoSets;
use rustmath_combinatorics::*;
use rustmath_integers::Integer;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn int(s: &str) -> Integer {
    Integer::from_decimal_str(s).unwrap()
}

// ---------------------------------------------------------------- revision 1 (items 7-23)

#[test]
#[cfg(target_pointer_width = "64")]
fn integer_vectors_with_sum_do_not_truncate() {
    // gp binomial(2^32+2, 2) = 9223372043297226753 (HEAD truncated to C(2, 2) = 1).
    assert_eq!(count_integer_vectors_with_sum(3, 1 << 32), 9223372043297226753);
    assert_eq!(count_integer_vectors_with_sum(3, 4), 15);
}

#[test]
#[should_panic(expected = "exceeds usize::MAX")]
fn integer_vectors_with_sum_overflow_panics() {
    // C(1999, 999) has 1994 bits (gp logint(binomial(1999,999),2)+1); HEAD returned 0.
    count_integer_vectors_with_sum(1000, 1000);
}

#[test]
fn integer_vectors_in_box_values() {
    assert_eq!(count_integer_vectors_in_box(&[0, 0], &[2, 3]), 12);
    // An empty side wins before any product.
    assert_eq!(count_integer_vectors_in_box(&[0, 5], &[(1 << 32) - 1, 4]), 0);
    #[cfg(target_pointer_width = "64")]
    {
        // gp 2^64 - 1 = 18446744073709551615
        assert_eq!(count_integer_vectors_in_box(&[i64::MIN + 1], &[i64::MAX]), usize::MAX);
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn integer_vectors_in_box_overflow_panics() {
    // gp 2^96 = 79228162514264337593543950336 (HEAD returned 0).
    count_integer_vectors_in_box(&[0; 3], &[(1 << 32) - 1; 3]);
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn integer_vectors_in_full_i64_box_overflow_panics() {
    // gp 2^64 = 18446744073709551616 (HEAD returned 0).
    count_integer_vectors_in_box(&[i64::MIN], &[i64::MAX]);
}

#[test]
fn six_vertex_partition_function_keeps_small_values() {
    // A(5) = 429 alternating sign matrices (OEIS A005130).
    assert_eq!(six_vertex_partition_function(5), Integer::from(429));
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds u32::MAX")]
fn six_vertex_partition_function_rejects_huge_n() {
    // HEAD truncated 2^32 + 3 to 3 and returned A(3) = 7.
    six_vertex_partition_function((1 << 32) + 3);
}

#[test]
fn parking_function_counts() {
    assert_eq!(count_non_decreasing_parking_functions(5), 42);
    #[cfg(target_pointer_width = "64")]
    {
        // gp 17^15 = 2862423051509815793; binomial(72,36)/37 = 11959798385860453492
        assert_eq!(count_parking_functions(16), 2862423051509815793);
        assert_eq!(count_non_decreasing_parking_functions(36), 11959798385860453492);
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn parking_function_counts_overflow_panic() {
    // gp 18^16 = 121439531096594251776 (HEAD wrapped to 10759066654336942080).
    count_parking_functions(17);
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn non_decreasing_parking_count_overflow_panics() {
    // gp binomial(74,37)/38 = 45950804324621742364 (HEAD wrapped to 9057316177202639132).
    count_non_decreasing_parking_functions(37);
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn non_decreasing_parking_count_refuses_huge_n() {
    // Must panic at once through the usize::BITS bound (HEAD allocated 2^40 + 1 words).
    count_non_decreasing_parking_functions(1 << 40);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn tuple_counts_are_checked() {
    // gp 2^63 = 9223372036854775808
    assert_eq!(TupleIterator::count(&TupleIterator::new(2, 63)), 1 << 63);
    // 0^(2^32) = 0: HEAD truncated k to 0 and gave 1.
    assert_eq!(TupleIterator::count(&TupleIterator::new(0, 1 << 32)), 0);
    // 2^64 does not fit: the hint is (0, None) at the start and after two calls.
    let mut it = TupleIterator::new(2, 64);
    assert_eq!(it.size_hint(), (0, None));
    it.next();
    it.next();
    assert_eq!(it.size_hint(), (0, None));
    // gp 2^63 - 2 = 9223372036854775806
    let mut it = TupleIterator::new(2, 63);
    it.next();
    it.next();
    assert_eq!(it.size_hint(), ((1 << 63) - 2, Some((1 << 63) - 2)));
    // gp 2^31 + 1 = 2147483649
    assert_eq!(Tuple::from_vec(vec![1, 1], 1 << 31).unwrap().rank(), (1 << 31) + 1);
    // (2^40)^3 does not fit, so every rank is valid (HEAD returned None).
    assert_eq!(Tuple::unrank(5, 1 << 40, 3).unwrap().indices(), &[0, 0, 5]);
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn tuple_count_overflow_panics() {
    // gp 2^64 (HEAD saturated to 18446744073709551615).
    TupleIterator::count(&TupleIterator::new(2, 64));
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn tuple_rank_overflow_panics() {
    // gp 2^80 + 2^40 + 1 = 1208925819615728686333953 (HEAD gave 1099511627777).
    Tuple::from_vec(vec![1, 1, 1], 1 << 40).unwrap().rank();
}

/// HEAD's `hook_length_dimension` body (young_representations.rs:52 at dc367a2): n! divided
/// by the product of the hook lengths, in Integer. An independent reference (review 4 m1).
fn hook_formula(p: &Partition) -> Integer {
    let n = p.sum();
    let mut numerator = Integer::one();
    for k in 2..=n {
        numerator = numerator * Integer::from(k as u64);
    }
    let mut denominator = Integer::one();
    for row in p.hook_lengths() {
        for h in row {
            denominator = denominator * Integer::from(h as u64);
        }
    }
    numerator / denominator
}

#[test]
fn standard_tableaux_counts() {
    let nst = |v: Vec<usize>| number_of_standard_tableaux(&Partition::new(v));
    // H145E13: the shape (3, 2) has 5 standard tableaux. The other values are gp hookdim
    // (n! / prod of hook lengths) of the design's rev1/rev1.gp and rev1/maxf.gp.
    assert_eq!(nst(vec![3, 2]), Integer::from(5));
    assert_eq!(nst(vec![11, 10]), Integer::from(58786));
    assert_eq!(nst(vec![20, 1]), Integer::from(20));
    assert_eq!(nst(vec![5, 5, 5, 5, 5]), Integer::from(701149020));
    assert_eq!(nst(vec![9, 7, 6, 4, 3, 3, 2, 1, 1]), int("40971642983700000000"));
    assert_eq!(nst((1..=10).rev().collect()), int("44261486084874072183645699204710400"));
    assert_eq!(nst(vec![7; 7]), int("475073684264389879228560"));
    assert_eq!(nst(vec![1_000_000_000]), Integer::one());
    assert_eq!(nst(vec![1; 30]), Integer::one());
    assert_eq!(nst(vec![]), Integer::one()); // one tableau of the empty shape
    // Hook-branch shapes (r(r-1)/2 > n): f(m, 1^(m-1)) = binomial(2m-2, m-1).
    // gp binomial(10,5) = 252; v = binomial(4000,2000): lift(Mod(v, 2^127-1)), logint(v,2)+1.
    let mut fat = vec![6];
    fat.extend(std::iter::repeat_n(1, 5));
    assert_eq!(nst(fat), Integer::from(252));
    let mut fat = vec![2001];
    fat.extend(std::iter::repeat_n(1, 2000));
    let v = nst(fat);
    let m = Integer::from(2).pow(127) - Integer::one();
    assert_eq!((v.clone() % m).to_string(), "85248900849712891484894804346834018665");
    assert_eq!(v.bit_length(), 3994);
    // An independent enumeration for n <= 9 (tableaux.rs standard_tableaux).
    for n in 0..=9 {
        for p in partitions(n) {
            assert_eq!(
                number_of_standard_tableaux(&p),
                Integer::from(standard_tableaux(&p).len() as u64),
                "{:?}",
                p.parts()
            );
        }
    }
    // Partition::dimension is the usize form.
    assert_eq!(Partition::new(vec![11, 10]).dimension(), 58786);
    assert_eq!(Partition::new(vec![20, 1]).dimension(), 20);
    assert_eq!(Partition::new(vec![5; 5]).dimension(), 701149020);
    assert_eq!(Partition::new(vec![]).dimension(), 1);
    #[cfg(target_pointer_width = "64")]
    assert_eq!(Partition::new(vec![9, 7, 6, 4, 3, 2, 2, 1, 1]).dimension(), 7821859115070000000);
}

#[test]
fn number_of_standard_tableaux_matches_the_hook_formula() {
    // Review 4 m1: against a test-local copy of HEAD's n!/prod(hooks), for every partition of
    // n <= 20 (2714 partitions; 311 of them take the hook branch).
    for n in 0..=20 {
        for p in partitions(n) {
            assert_eq!(number_of_standard_tableaux(&p), hook_formula(&p), "{:?}", p.parts());
        }
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn partition_dimension_overflow_panics() {
    // gp hookdim([8,7,6,5,4,3,2,1]) = 29258366996258488320 > 2^64 - 1 (HEAD returned 2).
    Partition::new(vec![8, 7, 6, 5, 4, 3, 2, 1]).dimension();
}

#[test]
fn hook_length_dimension_is_number_of_standard_tableaux() {
    for n in 0..=12 {
        for p in partitions(n) {
            assert_eq!(hook_length_dimension(&p), number_of_standard_tableaux(&p));
        }
    }
    // gp hookdim([11,10]) = 58786
    assert_eq!(hook_length_dimension(&Partition::new(vec![11, 10])), Integer::from(58786));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn colorings_keep_the_colour_count() {
    // gp 2^32 + 1 = 4294967297 and (2^32+2)^2 = 18446744090889420804 (HEAD: 1 and 4).
    assert_eq!(
        burnside_lemma(&PermutationGroup::cyclic(1), (1 << 32) + 1),
        Integer::from(4294967297u64)
    );
    assert_eq!(
        count_fixed_colorings(&Permutation::identity(2), (1 << 32) + 2),
        int("18446744090889420804")
    );
    assert_eq!(burnside_lemma(&PermutationGroup::cyclic(4), 2), Integer::from(6));
}

#[test]
fn tilings_keep_small_values() {
    // gp fibonacci(11) = 89; the 3 x 4 domino tilings are 11 (OEIS A001835).
    assert_eq!(count_tilings(2, 10, &Polyomino::dominoes()), Integer::from(89));
    assert_eq!(count_tilings(3, 4, &Polyomino::dominoes()), Integer::from(11));
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds u32::MAX")]
fn tilings_reject_n_above_u32() {
    // HEAD truncated 2^32 + 1 to 1 and returned count_tilings(2, 1) = 1.
    count_tilings(2, (1 << 32) + 1, &Polyomino::dominoes());
}

#[test]
#[cfg(target_pointer_width = "64")]
fn subset_ranks_are_exact() {
    // gp binomial(5*10^9, 2) = 12499999997500000000 (HEAD saturated to 2^63 - 1).
    let s = Subset::from_elements(vec![1, 2, 3], 5_000_000_001).unwrap();
    assert_eq!(s.rank_k_subset(3), Some(12499999997500000000));
    let u = Subset::unrank_k_subset(12499999997500000000, 5_000_000_001, 3).unwrap();
    assert_eq!(u.elements(), &[1, 2, 3]);
    // HEAD's rank_binary returned 0 for every subset when n > 64.
    assert_eq!(Subset::from_elements(vec![0], 100).unwrap().rank_binary(), 1);
    assert_eq!(Subset::from_elements(vec![63], 64).unwrap().rank_binary(), 1 << 63);
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn subset_k_rank_overflow_panics() {
    // gp binomial(10^7,3)-1 = 166666616666669999999 > 2^64 - 1.
    Subset::from_elements(vec![9_999_997, 9_999_998, 9_999_999], 10_000_000)
        .unwrap()
        .rank_k_subset(3);
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn subset_binary_rank_overflow_panics() {
    // The rank 2^64 of {64} does not fit (HEAD returned 0).
    Subset::from_elements(vec![64], 100).unwrap().rank_binary();
}

#[test]
fn sequence_validations_check_the_length() {
    // 2^64 wrapped to 0 = the length at HEAD, which accepted these.
    assert!(DeBruijnSequence::new(vec![], 2, 64).is_none());
    #[cfg(target_pointer_width = "64")]
    assert!(DeBruijnSequence::new(vec![0], 2, 1 << 32).is_none());
    assert!(OrthogonalArray::new(64, 1, 2, vec![]).is_none());
    let db = DeBruijnSequence::new(debruijn_sequence(2, 3), 2, 3).unwrap();
    assert!(db.is_valid());
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn universal_cycles_refuse_lengths_above_usize() {
    // 70! > 2^64 (gp 70! > 2^64 is 1); HEAD returned Some([0]) at once.
    universal_cycle_permutations(70, 70);
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn universal_cycle_subsets_refuse_lengths_above_usize() {
    // gp binomial(70,35) = 112186277816662845432 > 2^64 - 1; HEAD ran out of memory.
    universal_cycle_subsets(70, 35);
}

#[test]
fn sunflower_bound_is_checked() {
    assert_eq!(SetSystem::sunflower_bound(3, 3), 48); // 3! 2^3
    #[cfg(target_pointer_width = "64")]
    assert_eq!(SetSystem::sunflower_bound(20, 2), 2432902008176640000); // gp 20!
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn sunflower_bound_overflow_panics() {
    // gp 30!*2^30 = 284813089515958324736640819941867520000000 (HEAD saturated).
    SetSystem::sunflower_bound(30, 3);
}

// ---------------------------------------------------------------- revision 2 (items 24-36)

#[test]
#[cfg(target_pointer_width = "64")]
fn combination_ranks_are_exact() {
    // gp binomial(5*10^9, 2) = 12499999997500000000 (HEAD: 3276627960645224192).
    let c = Combination::from_vec(vec![1, 2, 3], 5_000_000_001).unwrap();
    assert_eq!(c.rank(), 12499999997500000000);
    let u = Combination::unrank(12499999997500000000, 5_000_000_001, 3).unwrap();
    assert_eq!(u.indices(), &[1, 2, 3]);
    for n in 0..=8 {
        for k in 0..=n {
            for (r, c) in combinations(n, k).iter().enumerate() {
                assert_eq!(c.rank(), r, "n = {n}, k = {k}");
                assert_eq!(Combination::unrank(r, n, k).unwrap(), *c);
            }
        }
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn combination_rank_overflow_panics() {
    // gp binomial(10^7,3)-1 = 166666616666669999999 > 2^64 - 1.
    Combination::from_vec(vec![9_999_997, 9_999_998, 9_999_999], 10_000_000)
        .unwrap()
        .rank();
}

#[test]
#[cfg(target_pointer_width = "64")]
fn permutation_gray_code_size_hint_is_an_upper_bound() {
    // gp 20! = 2432902008176640000; 21! = 51090942171709440000 > 2^64 - 1.
    assert_eq!(
        PermutationGrayCodeIterator::new(20).size_hint(),
        (0, Some(2432902008176640000))
    );
    assert_eq!(PermutationGrayCodeIterator::new(21).size_hint(), (0, None));
}

#[test]
fn superpartition_counts_are_checked() {
    for n in 0..=12 {
        assert_eq!(count_superpartitions(n), superpartitions(n).len(), "n = {n}");
    }
    // gp polcoef(prod(i=1,62,1/(1-2*x^i))+O(x^63),62) = 15969100102326684686
    #[cfg(target_pointer_width = "64")]
    assert_eq!(count_superpartitions(62), 15969100102326684686);
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn superpartition_count_overflow_panics() {
    // gp polcoef(prod(i=1,63,1/(1-2*x^i))+O(x^64),63) = 31938200243029484454 > 2^64 - 1.
    count_superpartitions(63);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn binary_words_respect_usize_bits() {
    let mut bits = vec![false; 100];
    bits[63] = true;
    assert_eq!(BinaryWord::new(bits).to_usize(), 1 << 63);
    assert_eq!(BinaryWord::from_usize(usize::MAX, 64).to_usize(), usize::MAX);
    let w = BinaryWord::from_usize(1, 65);
    let set: Vec<usize> = (0..65).filter(|&i| w.bits()[i]).collect();
    assert_eq!(set, vec![0]); // HEAD also set bit 64 (the shift by 64 wrapped to 0)
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "the value exceeds usize::MAX")]
fn binary_word_to_usize_overflow_panics() {
    let mut bits = vec![false; 65];
    bits[0] = true;
    bits[64] = true;
    BinaryWord::new(bits).to_usize(); // HEAD gave 2
}

#[test]
fn subset_binary_order_at_the_word_size() {
    let e = |s: Option<Subset>| s.map(|t| t.elements().to_vec());
    assert_eq!(e(Subset::empty(64).next_binary()), Some(vec![0]));
    assert_eq!(e(Subset::full(64).next_binary()), None);
    assert_eq!(Subset::from_bitvector(1, 65).elements(), &[0]);
    assert_eq!(Subset::unrank_binary(5, 100).elements(), &[0, 2]);
    assert_eq!(e(Subset::empty(63).next_binary()), Some(vec![0]));
}

#[test]
fn subset_binary_successors_for_any_n() {
    let e = |s: Option<Subset>| s.map(|t| t.elements().to_vec());
    let low: Vec<usize> = (0..64).collect();
    assert_eq!(e(Subset::empty(65).next_binary()), Some(vec![0]));
    assert_eq!(e(Subset::from_elements(vec![0], 65).unwrap().prev_binary()), Some(vec![]));
    assert_eq!(e(Subset::from_elements(low.clone(), 65).unwrap().next_binary()), Some(vec![64]));
    assert_eq!(e(Subset::from_elements(vec![64], 65).unwrap().prev_binary()), Some(low));
    assert_eq!(e(Subset::full(65).next_binary()), None);
    assert_eq!(e(Subset::empty(65).prev_binary()), None);
    for n in 0..=10usize {
        for bits in 0usize..(1 << n) {
            let s = Subset::from_bitvector(bits, n);
            assert_eq!(s.to_bitvector(), Some(bits));
            let next = if bits + 1 == 1 << n { None } else { Some(Subset::from_bitvector(bits + 1, n)) };
            assert_eq!(s.next_binary(), next, "n = {n}, bits = {bits}");
            let prev = if bits == 0 { None } else { Some(Subset::from_bitvector(bits - 1, n)) };
            assert_eq!(s.prev_binary(), prev, "n = {n}, bits = {bits}");
        }
    }
}

/// A permutation with one cycle (i, i+1, ..., i+l-1) of each length l.
fn with_cycles(lengths: &[usize]) -> Permutation {
    let total: usize = lengths.iter().sum();
    let mut perm = vec![0; total];
    let mut start = 0;
    for &l in lengths {
        for i in 0..l {
            perm[start + i] = start + (i + 1) % l;
        }
        start += l;
    }
    Permutation::from_vec(perm).unwrap()
}

const PRIMES_TO_53: [usize; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

#[test]
fn permutation_order_is_exact() {
    // gp lcm(concat(primes(15),[94])) = 614889782588491410 (422 points; HEAD 26163907895633379)
    let mut l = PRIMES_TO_53[..15].to_vec();
    l.push(94);
    let p = with_cycles(&l);
    assert_eq!(p.order_integer(), Integer::from(614889782588491410u64));
    #[cfg(target_pointer_width = "64")]
    assert_eq!(p.order(), 614889782588491410);
    // gp lcm(primes(16)) = 32589158477190044730 (381 points)
    assert_eq!(with_cycles(&PRIMES_TO_53).order_integer(), int("32589158477190044730"));
    assert_eq!(Permutation::identity(5).order_integer(), Integer::one());
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "use order_integer")]
fn permutation_order_overflow_panics() {
    // gp lcm(primes(16)) = 32589158477190044730 > 2^64 - 1 (HEAD 14142414403480493114).
    with_cycles(&PRIMES_TO_53).order();
}

/// The message of the panic of `f`, which must panic.
fn panic_message<F: FnOnce() -> R, R>(f: F) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).err().expect("expected a panic");
    if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&str>() {
        s.to_string()
    } else {
        String::from("<non-string panic>")
    }
}

fn chain(n: usize) -> Poset {
    Poset::new((0..n).collect(), (1..n).map(|i| (i - 1, i)).collect())
}

fn path_tree(vertices: usize) -> RootedTree {
    let mut t = RootedTree::from_children(vec![]);
    for _ in 1..vertices {
        t = RootedTree::from_children(vec![t]);
    }
    t
}

#[test]
#[cfg(target_pointer_width = "64")]
fn subset_mask_loops_refuse_word_size_sets() {
    const MSG: &str = "subsets exceed usize::MAX";
    let check = |m: String| assert!(m.contains(MSG), "unexpected panic: {m}");
    check(panic_message(|| all_binary_words(64)));
    check(panic_message(|| chain(64).all_order_ideals()));
    check(panic_message(|| chain(64).all_antichains()));
    check(panic_message(|| all_subwords(&[0u8; 64])));
    check(panic_message(|| {
        rustmath_combinatorics::subword::SubwordComplex::new(vec![0u8; 64], vec![0u8])
    }));
    check(panic_message(|| {
        SetSystem::from_sets(64, vec![(0..64).collect()]).unwrap().shadow()
    }));
    check(panic_message(|| SetSystem::from_sets(64, vec![vec![]]).unwrap().shade()));
    check(panic_message(|| path_tree(65).admissible_cuts()));
    check(panic_message(|| {
        OrderedMultisetPartitionIntoSets::new(vec![(0..64u32).collect()]).finer()
    }));
    check(panic_message(|| integer_vectors_with_l1_norm_bounded(64, 0)));
    // Small values stay.
    assert_eq!(chain(5).all_order_ideals().len(), 6);
    assert_eq!(chain(5).all_antichains().len(), 6);
    assert_eq!(path_tree(5).admissible_cuts().len(), 5);
    assert_eq!(all_binary_words(3).len(), 8);
}

#[test]
fn signed_compositions_need_no_sign_loop() {
    let parts = |v: Vec<SignedComposition>| -> Vec<Vec<isize>> {
        v.iter().map(|c| c.parts().to_vec()).collect()
    };
    // HEAD's i32 mask made these empty.
    assert_eq!(parts(signed_compositions_k(31, 31)), vec![vec![1isize; 31]]);
    assert_eq!(parts(signed_compositions_k(64, 64)), vec![vec![1isize; 64]]);
    assert_eq!(parts(signed_compositions_k(-3, 2)), vec![vec![-1, -2], vec![-2, -1]]);
    let pos = parts(signed_compositions(4));
    assert_eq!(pos.len(), 8); // the 2^3 compositions of 4, all positive
    assert!(pos.iter().all(|c| c.iter().all(|&x| x > 0)));
    let neg: Vec<Vec<isize>> = parts(signed_compositions(-4));
    let negated: Vec<Vec<isize>> = pos.iter().map(|c| c.iter().map(|&x| -x).collect()).collect();
    assert_eq!(neg, negated);
}

#[test]
fn q_and_t_multinomials_check_the_sum() {
    // The u32 sum 2^32 + 1 wrapped to 1 = n at HEAD, which then built q_factorial(2^32 - 1).
    assert!(q_multinomial(1, &[u32::MAX, 2]).is_zero());
    assert!(t_multinomial(1, &[u32::MAX, 2]).is_zero());
    // gp qf(n) = prod(i=1,n,(1-q^i)/(1-q)); Vecrev(qf(5)/(qf(2)^2*qf(1)))
    let want: Vec<Integer> = [1, 2, 4, 5, 6, 5, 4, 2, 1].iter().map(|&c| Integer::from(c)).collect();
    assert_eq!(q_multinomial(5, &[2, 2, 1]).coefficients(), &want[..]);
    assert_eq!(t_multinomial(5, &[2, 2, 1]).coefficients(), &want[..]);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn integer_vectors_keep_values_above_i64() {
    // gp 2^63 = 9223372036854775808 (HEAD: -9223372036854775808).
    let two63 = int("9223372036854775808");
    let v = integer_vectors_with_sum(1, 1 << 63);
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].components(), std::slice::from_ref(&two63));
    let first = integer_vector_sum_iter(2, 1 << 63).next().unwrap();
    assert_eq!(first.components(), &[Integer::zero(), two63]);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn integer_lists_prune_without_wrapping() {
    // The only list: three parts of at least 2^62 with sum 3 * 2^62 (HEAD: no list).
    let c = IntegerListConstraints::new()
        .sum(3 << 62)
        .length(3)
        .min_part(1 << 62)
        .max_part(1 << 63);
    let lists: Vec<Vec<usize>> = integer_lists(&c).iter().map(|l| l.parts().to_vec()).collect();
    assert_eq!(lists, vec![vec![1 << 62; 3]]);
    let small = IntegerListConstraints::new().sum(3).length(2);
    assert_eq!(integer_lists(&small).len(), 4);
}

#[test]
fn vector_partition_counts_are_exact() {
    // gp numbpart(100) = 190569292
    assert_eq!(fast_vector_partitions(&[100]), 190569292);
    assert_eq!(count_vector_partitions_max_parts(&[100], 100), 190569292);
    // gp polcoef(1/(1-x)+O(x^1003),1001) = 1: the partition 1^1001 (HEAD's cap of 1000 copies
    // dropped it).
    assert_eq!(fast_vector_partitions_with_max_part(&[1001], Some(&[1])), 1);
    assert_eq!(vector_partitions_with_max_part(&[1001], Some(&[1])).len(), 1);
    // The coefficient of x^1001 y in 1/((1-x)(1-y)(1-xy)) is 2 (design rev2/rev2.gp.extra:
    // f = (1/(1-x+O(x^1003)))*(1+y)*(1+x*y); polcoef(polcoef(f,1,y),1001,x)); HEAD gave 1.
    assert_eq!(fast_vector_partitions_with_max_part(&[1001, 1], Some(&[1, 1])), 2);
}

#[test]
#[should_panic(
    expected = "fast_vector_partitions_with_max_part: max_part must have as many components as the vector"
)]
fn vector_partition_count_refuses_a_longer_max_part() {
    // The part [0, 1] would be the zero vector on [5], so the copy loop would never end.
    // HEAD panicked here with an index out of bounds.
    fast_vector_partitions_with_max_part(&[5], Some(&[1, 1]));
}

#[test]
#[should_panic(
    expected = "vector_partitions_with_max_part: max_part must have as many components as the vector"
)]
fn vector_partition_list_refuses_a_longer_max_part() {
    vector_partitions_with_max_part(&[5], Some(&[1, 1]));
}

#[test]
#[should_panic(
    expected = "fast_vector_partitions_with_max_part: max_part must have as many components as the vector"
)]
fn vector_partition_count_refuses_a_shorter_max_part() {
    // HEAD returned 0 here: the second component could never reach 0.
    fast_vector_partitions_with_max_part(&[5, 5], Some(&[1]));
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "exceeds usize::MAX")]
fn vector_partition_count_overflow_panics() {
    // gp numbpart(417) = 18987964267331664557 > 2^64 - 1 (HEAD wrapped to 541220193622112941).
    fast_vector_partitions(&[417]);
}

#[test]
fn replication_number_is_exact() {
    // The Fano plane, a 2-(7, 3, 1) design: r = 3.
    let fano = vec![
        vec![0, 1, 2],
        vec![0, 3, 4],
        vec![0, 5, 6],
        vec![1, 3, 5],
        vec![1, 4, 6],
        vec![2, 3, 6],
        vec![2, 4, 5],
    ];
    assert_eq!(BlockDesign::new(7, 3, 2, 1, fano).unwrap().replication_number(), Some(3));
    #[cfg(target_pointer_width = "64")]
    {
        // gp 2^32*(2^33-1)/2 = 18446744071562067968 (HEAD 9223372034707292160).
        let d = BlockDesign::new(1 << 33, 3, 2, 1 << 32, vec![]).unwrap();
        assert_eq!(d.replication_number(), Some(18446744071562067968));
    }
}

fn mountain(semilength: usize) -> DyckWord {
    let mut w = vec![true; semilength];
    w.extend(std::iter::repeat_n(false, semilength));
    DyckWord::new(w).unwrap()
}

#[test]
fn dyck_area_does_not_wrap() {
    // gp 65536*65535/2 = 2147450880
    assert_eq!(mountain(65536).area(), 2147450880);
}

#[test]
#[should_panic(expected = "leaves the i32 range")]
fn dyck_area_overflow_panics() {
    // gp 65537*65536/2 = 2147516416 > 2^31 - 1 (HEAD: -2147450880).
    mountain(65537).area();
}
