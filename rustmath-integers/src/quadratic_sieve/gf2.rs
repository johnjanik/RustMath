//! Left nullspace of a dense matrix over GF(2), the linear algebra step of the quadratic sieve.
//!
//! MAGMA Handbook ch. 18 §18.10.3 `MPQS` (digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`).
//! R. Crandall, C. Pomerance, *Prime Numbers: A Computational Perspective*, 2nd ed., Springer
//! 2005, §6.1.3 (the matrix step): Gaussian elimination on bit-packed rows with an appended
//! identity ("history").
//!
//! For each column c, the first row that is not yet a pivot and has bit c becomes the pivot of
//! c, and it is added (XOR, the matrix part from word c/64 on, plus the history) to every other
//! non-pivot row with bit c. A non-pivot row only ever receives pivot rows, so after column c
//! every non-pivot row has bit c clear, and later pivots have bit c clear too: at the end the
//! matrix part of a non-pivot row is 0 and its history is a set of rows that sums to 0. A pivot
//! row's history holds only its own bit and the bits of earlier pivots, so each returned
//! history holds its own row and no other returned row: the vectors are independent. There are
//! rows - rank of them, so they are a basis of the left nullspace.
//!
//! Memory: rows x (ncols + rows) bits. Time: O(rows^2 (ncols + rows) / 64) word operations.

/// A basis of the left nullspace of the GF(2) matrix whose rows are `rows` (bit j of row r is
/// word j / 64, bit j % 64; every row has `ncols.div_ceil(64)` words). Each vector is the list
/// of row indices whose sum is 0, ascending. The number of vectors is rows - rank.
///
/// # Panics
///
/// If a row does not have `ncols.div_ceil(64)` words (an internal invariant of the sieve).
pub(super) fn left_nullspace(rows: &[Vec<u64>], ncols: usize) -> Vec<Vec<usize>> {
    let nrows = rows.len();
    let wc = ncols.div_ceil(64);
    let wr = nrows.div_ceil(64);
    let mut mat: Vec<Vec<u64>> = rows.to_vec();
    for row in &mat {
        assert_eq!(
            row.len(),
            wc,
            "left_nullspace: a row has the wrong number of words"
        );
    }
    let mut hist: Vec<Vec<u64>> = (0..nrows)
        .map(|r| {
            let mut h = vec![0u64; wr];
            h[r / 64] |= 1u64 << (r % 64);
            h
        })
        .collect();
    let mut pivot = vec![false; nrows];
    for c in 0..ncols {
        let (w, bit) = (c / 64, 1u64 << (c % 64));
        let Some(piv) = (0..nrows).find(|&r| !pivot[r] && mat[r][w] & bit != 0) else {
            continue;
        };
        pivot[piv] = true;
        let prow = mat[piv][w..].to_vec();
        let phist = hist[piv].clone();
        for r in 0..nrows {
            if !pivot[r] && mat[r][w] & bit != 0 {
                for (x, y) in mat[r][w..].iter_mut().zip(&prow) {
                    *x ^= *y;
                }
                for (x, y) in hist[r].iter_mut().zip(&phist) {
                    *x ^= *y;
                }
            }
        }
    }
    (0..nrows)
        .filter(|&r| !pivot[r])
        .map(|r| {
            debug_assert!(mat[r].iter().all(|&x| x == 0));
            (0..nrows)
                .filter(|&rr| hist[r][rr / 64] & (1u64 << (rr % 64)) != 0)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows from lists of set columns.
    fn rows_of(cols: &[&[usize]], ncols: usize) -> Vec<Vec<u64>> {
        cols.iter()
            .map(|set| {
                let mut row = vec![0u64; ncols.div_ceil(64)];
                for &c in *set {
                    row[c / 64] ^= 1u64 << (c % 64);
                }
                row
            })
            .collect()
    }

    fn sum_of(rows: &[Vec<u64>], dep: &[usize]) -> Vec<u64> {
        let mut acc = vec![0u64; rows[0].len()];
        for &r in dep {
            for (a, b) in acc.iter_mut().zip(&rows[r]) {
                *a ^= *b;
            }
        }
        acc
    }

    /// Port of HEAD's `test_gaussian_elimination`, with the vectors checked. HEAD's
    /// `gaussian_elimination_gf2` returned `[r0]` for the 3 x 2 all-ones matrix (row sum
    /// [1, 1], not a dependency); the left nullspace there is {r0, r1}, {r0, r2} (dimension 2).
    #[test]
    fn gf2_left_nullspace_hand() {
        // [1,0,1], [0,1,1], [1,1,0]: rank 2, the only dependency is r0 + r1 + r2.
        let rows = rows_of(&[&[0, 2], &[1, 2], &[0, 1]], 3);
        let deps = left_nullspace(&rows, 3);
        assert_eq!(deps, vec![vec![0, 1, 2]]);
        // 3 x 2 all ones: rank 1, two independent dependencies.
        let rows = rows_of(&[&[0, 1], &[0, 1], &[0, 1]], 2);
        let deps = left_nullspace(&rows, 2);
        assert_eq!(deps.len(), 2);
        for dep in &deps {
            assert!(sum_of(&rows, dep).iter().all(|&w| w == 0), "{dep:?}");
        }
        assert_ne!(deps[0], deps[1]);
        // An empty matrix and a zero row.
        assert!(left_nullspace(&[], 5).is_empty());
        let rows = rows_of(&[&[], &[3]], 4);
        assert_eq!(left_nullspace(&rows, 4), vec![vec![0]]);
    }

    /// splitmix64, for seeded test matrices.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Rank by an independent elimination: column echelon form of the transpose, on `bool`s.
    fn rank(rows: &[Vec<u64>], ncols: usize) -> usize {
        let nrows = rows.len();
        // Transpose: one vector of length nrows per column.
        let mut t: Vec<Vec<bool>> = (0..ncols)
            .map(|c| {
                (0..nrows)
                    .map(|r| rows[r][c / 64] >> (c % 64) & 1 == 1)
                    .collect()
            })
            .collect();
        let mut rank = 0;
        for r in 0..nrows {
            let Some(p) = (rank..ncols).find(|&c| t[c][r]) else {
                continue;
            };
            t.swap(rank, p);
            for c in 0..ncols {
                if c != rank && t[c][r] {
                    let (a, b) = if c < rank {
                        let (lo, hi) = t.split_at_mut(rank);
                        (&mut lo[c], &hi[0])
                    } else {
                        let (lo, hi) = t.split_at_mut(c);
                        (&mut hi[0], &lo[rank])
                    };
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x ^= *y;
                    }
                }
            }
            rank += 1;
        }
        rank
    }

    #[test]
    fn gf2_left_nullspace_random() {
        let mut state = 7u64;
        for (nrows, ncols, density_percent) in
            [(64usize, 64usize, 50u64), (120, 100, 5), (200, 150, 50)]
        {
            let rows: Vec<Vec<u64>> = (0..nrows)
                .map(|_| {
                    let mut row = vec![0u64; ncols.div_ceil(64)];
                    for c in 0..ncols {
                        if next(&mut state) % 100 < density_percent {
                            row[c / 64] |= 1u64 << (c % 64);
                        }
                    }
                    row
                })
                .collect();
            let deps = left_nullspace(&rows, ncols);
            for dep in &deps {
                assert!(!dep.is_empty());
                assert!(sum_of(&rows, dep).iter().all(|&w| w == 0), "{dep:?}");
            }
            assert_eq!(deps.len(), nrows - rank(&rows, ncols), "{nrows} x {ncols}");
        }
    }
}
