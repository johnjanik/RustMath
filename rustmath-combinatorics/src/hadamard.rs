//! Hadamard matrices: constructions and equivalence (MAGMA Chapter 148).
//!
//! A *Hadamard matrix* of order `n` is an `n × n` matrix of `±1` with `H Hᵀ = n·I`. This
//! module complements the `HadamardMatrix` type in `designs.rs` (which already provides the
//! Sylvester construction and normalization) with:
//!
//! | MAGMA intrinsic            | function here                          |
//! |----------------------------|----------------------------------------|
//! | `IsHadamard(H)`            | [`is_hadamard`]                        |
//! | Paley type I / II          | [`paley_type_i`], [`paley_type_ii`]   |
//! | `HadamardNormalize(H)`     | `HadamardMatrix::normalize` (reused)   |
//! | `HadamardInvariant(H)`     | [`hadamard_invariant`]                 |
//! | `IsHadamardEquivalent(H,J)`| [`is_hadamard_equivalent`]            |
//! | `HadamardRowDesign` (3-design) | [`hadamard_3_design`]             |
//!
//! * The **4-profile** [`hadamard_invariant`] is the sorted multiset of `|Σ_k Π_{r∈S} H[r][k]|`
//!   over all 4-subsets `S` of rows; it is invariant under row/column permutations and
//!   negations, so different profiles prove *inequivalence* cheaply.
//! * [`is_hadamard_equivalent`] performs the *complete* search over signed row permutations
//!   `P` (checking column permutation+sign equivalence to the target). It is exact for small
//!   orders and returns `None` (unresolved) once the search space exceeds a fixed budget —
//!   a bounded search that finds nothing is never reported as a decision.
//! * Only prime `q` is supported for the Paley constructions (Legendre symbol over `GF(q)`);
//!   prime-power `q` (needing `GF(q)`) is deferred.
//!
//! Reference: MAGMA Handbook, Chapter 148; Paley (1933).

use crate::designs::HadamardMatrix;
use crate::incidence_structure::IncidenceStructure;

/// Verify the Hadamard property `H Hᵀ = n·I` for a matrix of `±1` (any order).
pub fn is_hadamard(h: &HadamardMatrix) -> bool {
    let n = h.order();
    let m = h.matrix();
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0i64;
            for k in 0..n {
                dot += (m[i][k] as i64) * (m[j][k] as i64);
            }
            let expected = if i == j { n as i64 } else { 0 };
            if dot != expected {
                return false;
            }
        }
    }
    true
}

// --- Legendre symbol / primality -----------------------------------------------------

fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

/// Legendre symbol `(a | p)` for an odd prime `p`: `0` if `p | a`, `+1` if `a` is a nonzero
/// quadratic residue, `-1` otherwise.
fn legendre(a: i64, p: usize) -> i32 {
    let p_i = p as i64;
    let mut a = a % p_i;
    if a < 0 {
        a += p_i;
    }
    if a == 0 {
        return 0;
    }
    // a^((p-1)/2) mod p via modular exponentiation (u128 to avoid overflow).
    let mut result: u128 = 1;
    let mut base = a as u128 % p as u128;
    let mut exp = (p - 1) / 2;
    let modulus = p as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exp >>= 1;
    }
    if result == 1 {
        1
    } else {
        -1
    }
}

/// Paley construction of the **first** kind: for a prime `q ≡ 3 (mod 4)`, a (skew) Hadamard
/// matrix of order `q + 1`. Returns `None` if `q` is not a prime `≡ 3 (mod 4)`.
pub fn paley_type_i(q: usize) -> Option<HadamardMatrix> {
    if !is_prime(q) || q % 4 != 3 {
        return None;
    }
    let n = q + 1;
    let mut m = vec![vec![0i32; n]; n];
    // Border.
    m[0][0] = 1;
    for j in 1..n {
        m[0][j] = 1;
        m[j][0] = -1;
    }
    // Inner q×q block: H = I + Jacobsthal, indices a = i-1, b = j-1 in GF(q).
    for i in 1..n {
        for j in 1..n {
            if i == j {
                m[i][j] = 1;
            } else {
                let a = (i - 1) as i64;
                let b = (j - 1) as i64;
                m[i][j] = legendre(a - b, q);
            }
        }
    }
    HadamardMatrix::new(m)
}

/// Paley construction of the **second** kind: for a prime `q ≡ 1 (mod 4)`, a Hadamard matrix
/// of order `2(q + 1)`. Returns `None` if `q` is not a prime `≡ 1 (mod 4)`.
pub fn paley_type_ii(q: usize) -> Option<HadamardMatrix> {
    if !is_prime(q) || q % 4 != 1 {
        return None;
    }
    let m0 = q + 1; // order of the symmetric conference matrix
                    // Symmetric conference matrix S (0 diagonal, ±1 off-diagonal).
    let mut s = vec![vec![0i32; m0]; m0];
    for j in 1..m0 {
        s[0][j] = 1;
        s[j][0] = 1;
    }
    for i in 1..m0 {
        for j in 1..m0 {
            if i == j {
                s[i][j] = 0;
            } else {
                s[i][j] = legendre((i as i64) - (j as i64), q);
            }
        }
    }
    // Expand each entry into a 2×2 block:
    //   0  -> [[1,-1],[-1,-1]]
    //  +1  -> [[1, 1],[ 1,-1]]
    //  -1  -> [[-1,-1],[-1, 1]]
    let n = 2 * m0;
    let mut h = vec![vec![0i32; n]; n];
    for i in 0..m0 {
        for j in 0..m0 {
            let (a, b, c, d) = match s[i][j] {
                0 => (1, -1, -1, -1),
                1 => (1, 1, 1, -1),
                _ => (-1, -1, -1, 1),
            };
            h[2 * i][2 * j] = a;
            h[2 * i][2 * j + 1] = b;
            h[2 * i + 1][2 * j] = c;
            h[2 * i + 1][2 * j + 1] = d;
        }
    }
    HadamardMatrix::new(h)
}

// --- 4-profile invariant -------------------------------------------------------------

/// `HadamardInvariant(H)` — the 4-profile: the sorted multiset of `|Σ_k Π_{r∈S} H[r][k]|`
/// over all 4-subsets `S` of the rows. Invariant under Hadamard equivalence; equal profiles
/// do *not* imply equivalence, but distinct profiles imply inequivalence.
pub fn hadamard_invariant(h: &HadamardMatrix) -> Vec<i64> {
    let n = h.order();
    let m = h.matrix();
    let mut out = Vec::new();
    if n < 4 {
        return out;
    }
    for a in 0..n {
        for b in (a + 1)..n {
            for c in (b + 1)..n {
                for d in (c + 1)..n {
                    let mut s = 0i64;
                    for k in 0..n {
                        s += (m[a][k] as i64)
                            * (m[b][k] as i64)
                            * (m[c][k] as i64)
                            * (m[d][k] as i64);
                    }
                    out.push(s.abs());
                }
            }
        }
    }
    out.sort_unstable();
    out
}

// --- Equivalence ---------------------------------------------------------------------

/// Sign-canonicalise the columns (flip each column so its first entry is `+1`) and sort them.
fn canonical_columns(rows: &[Vec<i32>], n: usize) -> Vec<Vec<i32>> {
    let mut cols = Vec::with_capacity(n);
    for j in 0..n {
        let mut col: Vec<i32> = (0..n).map(|i| rows[i][j]).collect();
        if col[0] == -1 {
            for x in col.iter_mut() {
                *x = -*x;
            }
        }
        cols.push(col);
    }
    cols.sort();
    cols
}

/// Enumerate permutations of `0..n` via Heap's algorithm, calling `f` on each; stop early if
/// `f` returns `true`. Returns `true` if some call returned `true`.
fn for_each_permutation<F: FnMut(&[usize]) -> bool>(n: usize, mut f: F) -> bool {
    let mut a: Vec<usize> = (0..n).collect();
    let mut c = vec![0usize; n];
    if f(&a) {
        return true;
    }
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i % 2 == 0 {
                a.swap(0, i);
            } else {
                a.swap(c[i], i);
            }
            if f(&a) {
                return true;
            }
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    false
}

/// `IsHadamardEquivalent(H, J)` — decide whether `H` and `J` are Hadamard-equivalent
/// (related by row/column permutations and negations).
///
/// Returns `Some(true)`/`Some(false)` when decided, and `None` when the (complete) search is
/// abandoned because its size exceeds the internal budget (currently exact for orders `≤ 8`).
/// Different orders or different [`hadamard_invariant`]s always yield `Some(false)`.
pub fn is_hadamard_equivalent(h: &HadamardMatrix, j: &HadamardMatrix) -> Option<bool> {
    let n = h.order();
    if n != j.order() {
        return Some(false);
    }
    if n == 0 {
        return Some(true);
    }
    if hadamard_invariant(h) != hadamard_invariant(j) {
        return Some(false);
    }

    // Total signed-row-permutation search space = n! · 2^n. Refuse if over budget.
    const BUDGET: u128 = 20_000_000;
    let mut fact: u128 = 1;
    for k in 1..=n as u128 {
        fact = fact.saturating_mul(k);
    }
    let cost = fact.saturating_mul(1u128 << (n.min(63)));
    if cost > BUDGET {
        return None;
    }

    let hrows: Vec<Vec<i32>> = h.matrix().iter().cloned().collect();
    let jrows: Vec<Vec<i32>> = j.matrix().iter().cloned().collect();
    let jcanon = canonical_columns(&jrows, n);

    let found = for_each_permutation(n, |perm| {
        for mask in 0u32..(1u32 << n) {
            let mut a = vec![vec![0i32; n]; n];
            for i in 0..n {
                let sign = if (mask >> i) & 1 == 1 { -1 } else { 1 };
                let src = &hrows[perm[i]];
                for k in 0..n {
                    a[i][k] = sign * src[k];
                }
            }
            if canonical_columns(&a, n) == jcanon {
                return true;
            }
        }
        false
    });

    Some(found)
}

// --- Associated 3-design -------------------------------------------------------------

/// The Hadamard 3-design of a Hadamard matrix `H` of order `n ≥ 8` with `n ≡ 0 (mod 4)`:
/// normalise `H`, then for each non-constant row take the set of `+1` columns and the set of
/// `-1` columns as blocks. The result is a `3-(n, n/2, n/4 − 1)` design on the `n` columns
/// (`2(n-1)` blocks). Returns `None` for `n < 8` or `n` not divisible by 4.
pub fn hadamard_3_design(h: &HadamardMatrix) -> Option<IncidenceStructure> {
    let n = h.order();
    if n < 8 || n % 4 != 0 {
        return None;
    }
    let hn = h.normalize();
    let m = hn.matrix();
    let mut blocks = Vec::new();
    // Skip row 0 (all +1 after normalization).
    for i in 1..n {
        let plus: Vec<usize> = (0..n).filter(|&k| m[i][k] == 1).collect();
        let minus: Vec<usize> = (0..n).filter(|&k| m[i][k] == -1).collect();
        blocks.push(plus);
        blocks.push(minus);
    }
    IncidenceStructure::new(n, blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legendre_and_prime() {
        assert!(is_prime(7));
        assert!(!is_prime(9));
        // QRs mod 7 are {1,2,4}; non-residues {3,5,6}.
        assert_eq!(legendre(1, 7), 1);
        assert_eq!(legendre(2, 7), 1);
        assert_eq!(legendre(4, 7), 1);
        assert_eq!(legendre(3, 7), -1);
        assert_eq!(legendre(0, 7), 0);
        assert_eq!(legendre(-1, 7), -1); // 7 ≡ 3 mod 4 -> χ(-1) = -1
    }

    #[test]
    fn test_paley_type_i() {
        // q = 3 -> order 4.
        let h = paley_type_i(3).expect("paley I q=3");
        assert_eq!(h.order(), 4);
        assert!(is_hadamard(&h));
        // q = 7 -> order 8.
        let h8 = paley_type_i(7).expect("paley I q=7");
        assert_eq!(h8.order(), 8);
        assert!(is_hadamard(&h8));
        // q = 5 is ≡ 1 mod 4, not valid for type I.
        assert!(paley_type_i(5).is_none());
        // q = 9 is not prime.
        assert!(paley_type_i(9).is_none());
    }

    #[test]
    fn test_paley_type_ii() {
        // q = 5 -> order 2*(5+1) = 12.
        let h = paley_type_ii(5).expect("paley II q=5");
        assert_eq!(h.order(), 12);
        assert!(is_hadamard(&h));
        // q = 13 -> order 28.
        let h28 = paley_type_ii(13).expect("paley II q=13");
        assert_eq!(h28.order(), 28);
        assert!(is_hadamard(&h28));
        // q = 3 is ≡ 3 mod 4, not valid for type II.
        assert!(paley_type_ii(3).is_none());
    }

    #[test]
    fn test_sylvester_is_hadamard() {
        for k in 0..=4 {
            let h = HadamardMatrix::sylvester(k);
            assert!(is_hadamard(&h), "sylvester {} not hadamard", k);
        }
    }

    #[test]
    fn test_invariant_equal_under_equivalence() {
        // A matrix and a row-permuted, row-negated version share the 4-profile.
        let h = HadamardMatrix::sylvester(3); // order 8
        let m = h.matrix();
        // Build a scrambled equivalent: reverse row order and negate row 0.
        let n = h.order();
        let mut scr = vec![vec![0i32; n]; n];
        for i in 0..n {
            let src = &m[n - 1 - i];
            let sign = if i == 0 { -1 } else { 1 };
            for k in 0..n {
                scr[i][k] = sign * src[k];
            }
        }
        let h2 = HadamardMatrix::new(scr).unwrap();
        assert_eq!(hadamard_invariant(&h), hadamard_invariant(&h2));
    }

    #[test]
    fn test_equivalence_small() {
        // Order-4 Hadamard matrices are all equivalent.
        let h = HadamardMatrix::sylvester(2); // order 4
        assert_eq!(is_hadamard_equivalent(&h, &h), Some(true));
        // Equivalent scramble: permute columns and negate one column.
        let m = h.matrix();
        let n = h.order();
        let colperm = [2usize, 0, 3, 1];
        let mut scr = vec![vec![0i32; n]; n];
        for i in 0..n {
            for k in 0..n {
                let sign = if colperm[k] == 3 { -1 } else { 1 };
                scr[i][k] = sign * m[i][colperm[k]];
            }
        }
        let h2 = HadamardMatrix::new(scr).unwrap();
        assert_eq!(is_hadamard_equivalent(&h, &h2), Some(true));
    }

    #[test]
    fn test_equivalence_different_order() {
        let h4 = HadamardMatrix::sylvester(2);
        let h2 = HadamardMatrix::order_2();
        assert_eq!(is_hadamard_equivalent(&h4, &h2), Some(false));
    }

    #[test]
    fn test_hadamard_3_design() {
        // Order-8 Hadamard matrix -> unique 3-(8,4,1) design with 14 blocks.
        let h = HadamardMatrix::sylvester(3);
        let d = hadamard_3_design(&h).expect("3-design");
        assert_eq!(d.num_points(), 8);
        assert_eq!(d.num_blocks(), 14);
        assert_eq!(d.is_uniform(), Some(4));
        assert_eq!(d.is_balanced(3), Some(1));
        assert_eq!(d.is_design(3), Some((4, 1)));
        // Order 4 is too small (degenerate).
        assert!(hadamard_3_design(&HadamardMatrix::sylvester(2)).is_none());
    }
}
