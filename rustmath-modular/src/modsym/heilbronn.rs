//! # Merel's Heilbronn-type matrices (determinant n) for Hecke operators
//!
//! For n >= 1, Merel's family
//!     M_n = { [[a, b], [c, d]] in M_2(Z) : ad - bc = n, a > b >= 0, d > c >= 0 }
//! computes the Hecke operator T_n on weight-2 Manin symbols of any level N:
//!     T_n [(c : d)] = sum_{h in M_n} [(c : d) h],
//! where terms with (c : d) h not a point of P^1(Z/NZ) are omitted.  Such
//! terms occur only when gcd(n, N) > 1, and with the omission the formula
//! computes the operator U_p for primes p | N.
//!
//! Corresponds to `sage.modular.modsym.heilbronn` (`HeilbronnMerel`) and the
//! MAGMA handbook chapter "Modular Symbols" (`HeckeOperator`).  References:
//! L. Merel, *Universal Fourier expansions of modular forms* (Springer LNM
//! 1585, 1994); Cremona, *Algorithms for Modular Elliptic Curves*, section
//! 2.4; Stein, *Modular Forms: A Computational Approach*, section 8.3.
//!
//! VERIFIED before implementation (python brute force, exact `Fraction`
//! arithmetic, no external tables): on the Manin-symbol quotient of
//! M_2(Gamma0(N)) the matrix of [x] -> sum_h [x h] over this family equals,
//! entry for entry, the matrix of the double-coset definition
//!     T_p {a, b} = sum_{r=0}^{p-1} {(a+r)/p, (b+r)/p} + {pa, pb}
//! (last term omitted when p | N, giving U_p) computed independently via
//! continued fractions, for
//!   T_p at (N, p) in {(11,2),(11,3),(11,5),(14,3),(14,5),(15,2),(24,5),
//!                     (33,2),(37,2),(37,3)},
//!   U_p at (N, p) in {(11,11),(14,2),(14,7),(15,3),(15,5),(24,2),(24,3),
//!                     (33,3),(33,11)}.
//! Moreover the summed image kills the two- and three-term Manin relations
//! at every generator (well-definedness on the quotient) for N in
//! {11,14,15,24,33,37} and p in {2,3,5}, and for composite n the family
//! reproduces the Hecke recursion (T_4 = T_2^2 - 2, T_6 = T_2 T_3,
//! T_9 = T_3^2 - 3) exactly on the ambient space at N in {11,14,15}.

/// All integer matrices [[a, b], [c, d]] with determinant `n`, `a > b >= 0`
/// and `d > c >= 0` (Merel's family M_n), in lexicographic (a, d, b, c)
/// order.
///
/// Completeness of the loop bounds: the constraints force
/// `bc = ad - n` with `0 <= b <= a - 1` and `0 <= c <= d - 1`, hence
/// `0 <= ad - n <= (a - 1)(d - 1)`, which rearranges to `a + d <= n + 1`
/// (and `ad >= n`); in particular `1 <= a <= n` and `1 <= d <= n + 1 - a`.
pub fn merel_matrices(n: u64) -> Vec<[[i64; 2]; 2]> {
    assert!(n >= 1, "determinant must be positive");
    assert!(n <= i64::MAX as u64, "determinant too large");
    let n = n as i64;
    let mut out = Vec::new();
    for a in 1..=n {
        for d in 1..=(n + 1 - a) {
            let k = a * d - n;
            if k < 0 {
                continue;
            }
            for b in 0..a {
                for c in 0..d {
                    if b * c == k {
                        out.push([[a, b], [c, d]]);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_family_sizes_match_python_bruteforce() {
        // Sizes recomputed independently in python by the same brute force
        // that verified the family against the double-coset definition.
        let expected: [(u64, usize); 12] = [
            (2, 4),
            (3, 7),
            (4, 13),
            (5, 15),
            (6, 26),
            (7, 25),
            (9, 40),
            (11, 49),
            (13, 63),
            (17, 93),
            (19, 109),
            (23, 143),
        ];
        for (n, size) in expected {
            assert_eq!(merel_matrices(n).len(), size, "|M_{n}|");
        }
    }

    #[test]
    fn test_defining_constraints() {
        for n in 1..=30u64 {
            let mats = merel_matrices(n);
            for m in &mats {
                let [[a, b], [c, d]] = *m;
                assert_eq!(a * d - b * c, n as i64, "determinant {n}");
                assert!(a > b && b >= 0, "a > b >= 0 in M_{n}");
                assert!(d > c && c >= 0, "d > c >= 0 in M_{n}");
            }
            // no duplicates
            let mut sorted = mats.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), mats.len(), "M_{n} has no duplicates");
        }
    }

    #[test]
    fn test_explicit_family_for_p_2() {
        // The classical determinant-2 Heilbronn set.
        let mut mats = merel_matrices(2);
        mats.sort();
        assert_eq!(
            mats,
            vec![
                [[1, 0], [0, 2]],
                [[1, 0], [1, 2]],
                [[2, 0], [0, 1]],
                [[2, 1], [0, 1]],
            ]
        );
    }

    #[test]
    fn test_identity_family_for_n_1() {
        assert_eq!(merel_matrices(1), vec![[[1, 0], [0, 1]]]);
    }
}
