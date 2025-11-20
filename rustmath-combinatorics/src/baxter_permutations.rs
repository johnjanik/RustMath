//! Baxter permutations - permutations avoiding vincular patterns 2-41-3 and 3-14-2
//!
//! A Baxter permutation is a permutation that avoids two specific vincular patterns:
//! - Pattern 2-41-3: where elements with relative order (2nd smallest, largest, smallest, 3rd smallest)
//!   have the largest and smallest elements consecutive in position
//! - Pattern 3-14-2: where elements with relative order (3rd smallest, smallest, largest, 2nd smallest)
//!   have the smallest and largest elements consecutive in position
//!
//! Baxter permutations are counted by the Baxter numbers, related to Catalan numbers.
//! The first few counts are: 1, 1, 2, 6, 22, 92, 422, 2074, 10754, ...
//! This is OEIS sequence A001181.

use crate::Permutation;

/// Check if a permutation is a Baxter permutation
///
/// A Baxter permutation avoids the vincular patterns 2-41-3 and 3-14-2
pub fn is_baxter(perm: &Permutation) -> bool {
    !contains_vincular_2413(perm) && !contains_vincular_3142(perm)
}

/// Check if a permutation contains the vincular pattern 2-41-3
///
/// This pattern consists of four positions i < j < k < l where:
/// - Positions j and k are consecutive (k = j + 1)
/// - The values have relative order: 2nd smallest, largest, smallest, 3rd smallest
/// - In other words: p[j] < p[l] < p[i] < p[k]
fn contains_vincular_2413(perm: &Permutation) -> bool {
    let p = perm.as_slice();
    let n = p.len();

    if n < 4 {
        return false;
    }

    // For each pair of consecutive positions (j, k) where k = j + 1
    for j in 0..n - 1 {
        let k = j + 1;
        let val_j = p[j];
        let val_k = p[k];

        // Try all positions i < j for the first element
        for i in 0..j {
            let val_i = p[i];

            // Try all positions l > k for the last element
            for l in (k + 1)..n {
                let val_l = p[l];

                // Check if these four values form the pattern 2-41-3
                // Pattern 2-41-3 at positions i, j, k, l means:
                // val_k < val_i < val_l < val_j
                // (smallest at k, 2nd smallest at i, 3rd smallest at l, largest at j)
                if val_k < val_i && val_i < val_l && val_l < val_j {
                    return true;
                }
            }
        }
    }

    false
}

/// Check if a permutation contains the vincular pattern 3-14-2
///
/// This pattern consists of four positions i < j < k < l where:
/// - Positions j and k are consecutive (k = j + 1)
/// - The values have relative order: 3rd smallest, smallest, largest, 2nd smallest
/// - In other words: p[j] < p[l] < p[i] < p[k]
fn contains_vincular_3142(perm: &Permutation) -> bool {
    let p = perm.as_slice();
    let n = p.len();

    if n < 4 {
        return false;
    }

    // For each pair of consecutive positions (j, k) where k = j + 1
    for j in 0..n - 1 {
        let k = j + 1;
        let val_j = p[j];
        let val_k = p[k];

        // Try all positions i < j for the first element
        for i in 0..j {
            let val_i = p[i];

            // Try all positions l > k for the last element
            for l in (k + 1)..n {
                let val_l = p[l];

                // Check if these four values form the pattern 3-14-2
                // Pattern 3-14-2 at positions i, j, k, l means:
                // val_j < val_l < val_i < val_k
                // (smallest at j, 2nd smallest at l, 3rd smallest at i, largest at k)
                if val_j < val_l && val_l < val_i && val_i < val_k {
                    return true;
                }
            }
        }
    }

    false
}

/// Generate all Baxter permutations of size n
///
/// Returns a vector of all permutations of {0, 1, ..., n-1} that avoid
/// the vincular patterns 2-41-3 and 3-14-2
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::baxter_permutations;
///
/// let baxters = baxter_permutations(3);
/// assert_eq!(baxters.len(), 6); // All permutations of size 3 are Baxter
/// ```
pub fn baxter_permutations(n: usize) -> Vec<Permutation> {
    use crate::all_permutations;

    all_permutations(n)
        .into_iter()
        .filter(|p| is_baxter(p))
        .collect()
}

/// Count the number of Baxter permutations of size n
///
/// Returns the nth Baxter number, which counts the number of
/// Baxter permutations of size n.
///
/// The sequence begins: 1, 1, 2, 6, 22, 92, 422, 2074, 10754, ...
/// (OEIS A001181)
pub fn count_baxter_permutations(n: usize) -> usize {
    baxter_permutations(n).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_baxter_small() {
        // All permutations of size 0, 1, 2, 3 are Baxter
        let id0 = Permutation::from_vec(vec![]).unwrap();
        assert!(is_baxter(&id0));

        let id1 = Permutation::from_vec(vec![0]).unwrap();
        assert!(is_baxter(&id1));

        let p21 = Permutation::from_vec(vec![1, 0]).unwrap();
        assert!(is_baxter(&p21));

        // All permutations of size 3 are Baxter
        let perms3 = crate::all_permutations(3);
        for perm in &perms3 {
            assert!(is_baxter(perm), "Permutation {:?} should be Baxter", perm.as_slice());
        }
    }

    #[test]
    fn test_baxter_count_n3() {
        // All 6 permutations of size 3 are Baxter
        let baxters = baxter_permutations(3);
        assert_eq!(baxters.len(), 6);
    }

    #[test]
    fn test_baxter_count_n4() {
        // For n=4, there should be 22 Baxter permutations
        let count = count_baxter_permutations(4);
        assert_eq!(count, 22);
    }

    #[test]
    fn test_baxter_counts() {
        // Test the first few Baxter numbers (OEIS A001181)
        // 1, 1, 2, 6, 22, 92, 422, ...
        assert_eq!(count_baxter_permutations(0), 1);
        assert_eq!(count_baxter_permutations(1), 1);
        assert_eq!(count_baxter_permutations(2), 2);
        assert_eq!(count_baxter_permutations(3), 6);
        assert_eq!(count_baxter_permutations(4), 22);
        // Note: n=5 would be 92 but might be slow for testing
    }

    #[test]
    fn test_baxter_examples() {
        // These should be Baxter permutations
        // Identity is always Baxter
        let id = Permutation::identity(4);
        assert!(is_baxter(&id));

        // All permutations of size <= 3 are Baxter
        let id3 = Permutation::identity(3);
        assert!(is_baxter(&id3));

        let p3 = Permutation::from_vec(vec![2, 1, 0]).unwrap();
        assert!(is_baxter(&p3));
    }

    #[test]
    fn test_baxter_permutations_n2() {
        let baxters = baxter_permutations(2);
        assert_eq!(baxters.len(), 2);

        // Both should be present: [0,1] and [1,0]
        let id = Permutation::from_vec(vec![0, 1]).unwrap();
        let swap = Permutation::from_vec(vec![1, 0]).unwrap();
        assert!(baxters.contains(&id));
        assert!(baxters.contains(&swap));
    }

    #[test]
    fn test_non_baxter_count_n4() {
        // For n=4, there are 24 total permutations and 22 are Baxter
        // So exactly 2 permutations of size 4 should NOT be Baxter
        let all_perms = crate::all_permutations(4);
        let baxter_perms = baxter_permutations(4);

        assert_eq!(all_perms.len(), 24);
        assert_eq!(baxter_perms.len(), 22);

        let non_baxter_count = all_perms.iter().filter(|p| !is_baxter(p)).count();
        assert_eq!(non_baxter_count, 2);
    }
}
