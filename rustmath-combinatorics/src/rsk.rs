//! The full RSK (Robinson–Schensted–Knuth) correspondence on nonnegative-integer
//! matrices.
//!
//! Knuth's generalisation of the Robinson–Schensted correspondence is a bijection
//!
//! ```text
//!   { p x q matrices with nonnegative integer entries }
//!       <-->  { (P, Q) : P, Q semistandard Young tableaux of the same shape,
//!               entries of P in {1..q}, entries of Q in {1..p} }
//! ```
//!
//! A matrix `A = (a_ij)` is encoded as its *biword* (two-line array): the pairs
//! `(i, j)` repeated `a_ij` times, listed in lexicographic order. Row-inserting the
//! bottom letters `j` (Schensted bumping) builds the insertion tableau `P`, while the
//! top letters `i` are recorded in `Q` at the cell created by each insertion. Under
//! this bijection the column sums of `A` give the weight (content) of `P` and the row
//! sums give the weight of `Q`; permutation matrices recover the classical
//! Robinson–Schensted correspondence.
//!
//! References: Knuth, *Permutations, matrices, and generalized Young tableaux*
//! (Pacific J. Math. 34, 1970); Stanley, *Enumerative Combinatorics* vol. 2, §7.11;
//! Fulton, *Young Tableaux*, §4.

use crate::tableaux::Tableau;

/// The biword (two-line array) of a nonnegative-integer matrix, in lexicographic
/// order: for each row `i` (1-based) and column `j` (1-based), the pair `(i, j)` is
/// repeated `a_ij` times.
///
/// # Examples
/// ```
/// use rustmath_combinatorics::rsk::matrix_to_biword;
///
/// let m = vec![vec![1, 0, 2], vec![0, 2, 0], vec![1, 1, 0]];
/// assert_eq!(
///     matrix_to_biword(&m),
///     vec![(1, 1), (1, 3), (1, 3), (2, 2), (2, 2), (3, 1), (3, 2)]
/// );
/// ```
pub fn matrix_to_biword(matrix: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut biword = Vec::new();
    for (i, row) in matrix.iter().enumerate() {
        for (j, &mult) in row.iter().enumerate() {
            for _ in 0..mult {
                biword.push((i + 1, j + 1));
            }
        }
    }
    biword
}

/// Schensted row insertion of `value` into rows-in-place; returns the index of the
/// row where a new cell was created (the new cell is always at the end of that row).
fn row_insert(rows: &mut Vec<Vec<usize>>, value: usize) -> usize {
    let mut cur = value;
    for (idx, row) in rows.iter_mut().enumerate() {
        // Bump the leftmost entry strictly greater than the incoming letter.
        match row.iter().position(|&x| x > cur) {
            Some(pos) => std::mem::swap(&mut cur, &mut row[pos]),
            None => {
                row.push(cur);
                return idx;
            }
        }
    }
    rows.push(vec![cur]);
    rows.len() - 1
}

/// RSK insertion of a biword (two-line array).
///
/// The biword must consist of pairs of positive integers and be weakly increasing in
/// lexicographic order (as produced by [`matrix_to_biword`]); otherwise `None` is
/// returned. Returns the pair `(P, Q)`: `P` is the insertion tableau of the bottom
/// letters, `Q` records the top letter at each newly created cell. Both are
/// semistandard of the same shape.
pub fn rsk_biword(biword: &[(usize, usize)]) -> Option<(Tableau, Tableau)> {
    for &(i, j) in biword {
        if i == 0 || j == 0 {
            return None;
        }
    }
    for w in biword.windows(2) {
        if w[1] < w[0] {
            return None;
        }
    }

    let mut p_rows: Vec<Vec<usize>> = Vec::new();
    let mut q_rows: Vec<Vec<usize>> = Vec::new();
    for &(i, j) in biword {
        let r = row_insert(&mut p_rows, j);
        if r == q_rows.len() {
            q_rows.push(vec![i]);
        } else {
            q_rows[r].push(i);
        }
    }
    Some((Tableau::new(p_rows)?, Tableau::new(q_rows)?))
}

/// The RSK correspondence: nonnegative-integer matrix to a pair `(P, Q)` of
/// semistandard Young tableaux of the same shape.
///
/// `P` has weight equal to the column sums of the matrix, `Q` weight equal to the
/// row sums. Inverted by [`inverse_rsk`].
///
/// # Examples
/// ```
/// use rustmath_combinatorics::rsk::rsk;
///
/// // Stanley, EC2 §7.11 example.
/// let m = vec![vec![1, 0, 2], vec![0, 2, 0], vec![1, 1, 0]];
/// let (p, q) = rsk(&m);
/// assert_eq!(p.rows(), &[vec![1, 1, 2, 2], vec![2, 3], vec![3]]);
/// assert_eq!(q.rows(), &[vec![1, 1, 1, 3], vec![2, 2], vec![3]]);
/// ```
pub fn rsk(matrix: &[Vec<usize>]) -> (Tableau, Tableau) {
    rsk_biword(&matrix_to_biword(matrix))
        .expect("the biword of a nonnegative matrix is always lexicographically sorted")
}

/// The inverse RSK correspondence: a pair `(P, Q)` of semistandard Young tableaux of
/// the same shape back to the unique `nrows x ncols` nonnegative-integer matrix
/// mapping to it under [`rsk`].
///
/// Returns `None` if the tableaux do not form a valid same-shape semistandard pair,
/// or if their entries exceed `nrows` (for `Q`) / `ncols` (for `P`).
pub fn inverse_rsk(p: &Tableau, q: &Tableau, nrows: usize, ncols: usize) -> Option<Vec<Vec<usize>>> {
    if p.shape() != q.shape() {
        return None;
    }
    if p.size() > 0 && (!p.is_semistandard() || !q.is_semistandard()) {
        return None;
    }

    let mut p_rows: Vec<Vec<usize>> = p.rows().to_vec();
    let mut q_rows: Vec<Vec<usize>> = q.rows().to_vec();
    let mut matrix = vec![vec![0usize; ncols]; nrows];

    while q_rows.iter().any(|r| !r.is_empty()) {
        // The last-inserted biword pair has the maximal top letter, and among equal
        // top letters the last-inserted cell is the rightmost (equal recording
        // letters form a horizontal strip, added left to right). In a semistandard
        // Q every occurrence of the maximum ends its row, so scan the row ends.
        let mx = *q_rows.iter().flat_map(|r| r.iter()).max().unwrap();
        let mut best: Option<(usize, usize)> = None;
        for (r, row) in q_rows.iter().enumerate() {
            if row.last() == Some(&mx) {
                let c = row.len() - 1;
                if best.map_or(true, |(_, bc)| c > bc) {
                    best = Some((r, c));
                }
            }
        }
        let (r, _c) = best?;

        // Remove the corner cell and reverse-bump its P entry up through the rows:
        // in each row above, the incoming value displaces the rightmost entry
        // strictly smaller than it.
        q_rows[r].pop();
        let mut v = p_rows[r].pop()?;
        for rr in (0..r).rev() {
            let row = &mut p_rows[rr];
            let k = row.iter().rposition(|&x| x < v)?;
            std::mem::swap(&mut row[k], &mut v);
        }

        if mx > nrows || v > ncols {
            return None;
        }
        matrix[mx - 1][v - 1] += 1;
    }

    Some(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitions::partitions;
    use crate::tableaux::robinson_schensted;
    use std::collections::{HashMap, HashSet};

    /// All semistandard Young tableaux of the given shape with entries in 1..=maxval.
    fn ssyt_of_shape(shape: &[usize], maxval: usize) -> Vec<Vec<Vec<usize>>> {
        let cells: Vec<(usize, usize)> = shape
            .iter()
            .enumerate()
            .flat_map(|(r, &len)| (0..len).map(move |c| (r, c)))
            .collect();
        let mut filled: Vec<Vec<usize>> = shape.iter().map(|&len| vec![0; len]).collect();
        let mut out = Vec::new();
        fn rec(
            cells: &[(usize, usize)],
            idx: usize,
            maxval: usize,
            filled: &mut Vec<Vec<usize>>,
            out: &mut Vec<Vec<Vec<usize>>>,
        ) {
            if idx == cells.len() {
                out.push(filled.clone());
                return;
            }
            let (r, c) = cells[idx];
            let mut lo = 1;
            if c > 0 {
                lo = lo.max(filled[r][c - 1]); // weakly increasing rows
            }
            if r > 0 {
                lo = lo.max(filled[r - 1][c] + 1); // strictly increasing columns
            }
            for v in lo..=maxval {
                filled[r][c] = v;
                rec(cells, idx + 1, maxval, filled, out);
            }
            filled[r][c] = 0;
        }
        rec(&cells, 0, maxval, &mut filled, &mut out);
        out
    }

    fn weight(rows: &[Vec<usize>], letter: usize) -> usize {
        rows.iter().flat_map(|r| r.iter()).filter(|&&x| x == letter).count()
    }

    #[test]
    fn test_rsk_stanley_example() {
        // Stanley EC2 §7.11: verified independently by brute force (python3).
        let m = vec![vec![1, 0, 2], vec![0, 2, 0], vec![1, 1, 0]];
        assert_eq!(
            matrix_to_biword(&m),
            vec![(1, 1), (1, 3), (1, 3), (2, 2), (2, 2), (3, 1), (3, 2)]
        );
        let (p, q) = rsk(&m);
        assert_eq!(p.rows(), &[vec![1, 1, 2, 2], vec![2, 3], vec![3]]);
        assert_eq!(q.rows(), &[vec![1, 1, 1, 3], vec![2, 2], vec![3]]);
        assert_eq!(inverse_rsk(&p, &q, 3, 3).unwrap(), m);
    }

    #[test]
    fn test_rsk_empty_and_zero_matrix() {
        let (p, q) = rsk(&[]);
        assert_eq!(p.size(), 0);
        assert_eq!(q.size(), 0);

        let zero = vec![vec![0, 0], vec![0, 0]];
        let (p, q) = rsk(&zero);
        assert_eq!(p.size(), 0);
        assert_eq!(q.size(), 0);
        assert_eq!(inverse_rsk(&p, &q, 2, 2).unwrap(), zero);
    }

    #[test]
    fn test_rsk_biword_rejects_invalid() {
        // Not lexicographically sorted.
        assert!(rsk_biword(&[(2, 1), (1, 1)]).is_none());
        assert!(rsk_biword(&[(1, 2), (1, 1)]).is_none());
        // Letters must be positive.
        assert!(rsk_biword(&[(0, 1)]).is_none());
        assert!(rsk_biword(&[(1, 0)]).is_none());
    }

    #[test]
    fn test_rsk_permutation_matrix_recovers_robinson_schensted() {
        // On permutation matrices, RSK is the classical Robinson–Schensted map.
        fn perms(n: usize) -> Vec<Vec<usize>> {
            if n == 0 {
                return vec![vec![]];
            }
            let mut out = Vec::new();
            for p in perms(n - 1) {
                for pos in 0..=p.len() {
                    let mut q = p.clone();
                    q.insert(pos, n);
                    out.push(q);
                }
            }
            out
        }
        for n in 1..=4 {
            for perm in perms(n) {
                let mut m = vec![vec![0usize; n]; n];
                for (i, &v) in perm.iter().enumerate() {
                    m[i][v - 1] = 1;
                }
                let (p, q) = rsk(&m);
                let (rp, rq) = robinson_schensted(&perm);
                assert_eq!(p, rp, "P mismatch for {:?}", perm);
                assert_eq!(q, rq, "Q mismatch for {:?}", perm);
                assert_eq!(inverse_rsk(&p, &q, n, n).unwrap(), m);
            }
        }
    }

    #[test]
    fn test_rsk_correspondence_exhaustive_2x2_entries_le_2() {
        // RSK correspondence theorem, exhaustively on all 81 2x2 matrices with
        // entries <= 2: image is a same-shape semistandard pair, weight(P) = column
        // sums, weight(Q) = row sums, the map is injective, and inverse_rsk
        // recovers the matrix.
        let mut seen: HashMap<(Vec<Vec<usize>>, Vec<Vec<usize>>), Vec<Vec<usize>>> = HashMap::new();
        for a in 0..=2usize {
            for b in 0..=2usize {
                for c in 0..=2usize {
                    for d in 0..=2usize {
                        let m = vec![vec![a, b], vec![c, d]];
                        let (p, q) = rsk(&m);
                        assert_eq!(p.shape(), q.shape(), "shapes differ for {:?}", m);
                        assert_eq!(p.size(), a + b + c + d);
                        if p.size() > 0 {
                            assert!(p.is_semistandard(), "P not SSYT for {:?}", m);
                            assert!(q.is_semistandard(), "Q not SSYT for {:?}", m);
                        }
                        // Weights: columns of the matrix for P, rows for Q.
                        assert_eq!(weight(p.rows(), 1), a + c);
                        assert_eq!(weight(p.rows(), 2), b + d);
                        assert_eq!(weight(q.rows(), 1), a + b);
                        assert_eq!(weight(q.rows(), 2), c + d);
                        // Injectivity.
                        let key = (p.rows().to_vec(), q.rows().to_vec());
                        if let Some(other) = seen.insert(key, m.clone()) {
                            panic!("RSK collision: {:?} and {:?}", other, m);
                        }
                        // Roundtrip.
                        assert_eq!(inverse_rsk(&p, &q, 2, 2).unwrap(), m, "roundtrip failed");
                    }
                }
            }
        }
        assert_eq!(seen.len(), 81);
    }

    #[test]
    fn test_rsk_bijection_onto_same_shape_ssyt_pairs() {
        // Full bijection statement, sliced by total sum n = 0..4: RSK maps the set
        // of 2x2 matrices with entry sum n bijectively ONTO the set of pairs (P, Q)
        // of same-shape semistandard tableaux with |shape| = n and entries <= 2.
        // (Counts verified independently by brute force in python3: 1, 4, 10, 20, 35.)
        let expected_counts = [1usize, 4, 10, 20, 35];
        for n in 0..=4usize {
            // All same-shape SSYT pairs of size n with entries <= 2.
            let mut pairs: HashSet<(Vec<Vec<usize>>, Vec<Vec<usize>>)> = HashSet::new();
            for shape in partitions(n) {
                let fillings = ssyt_of_shape(shape.parts(), 2);
                for p in &fillings {
                    for q in &fillings {
                        pairs.insert((p.clone(), q.clone()));
                    }
                }
            }
            // Images of all 2x2 matrices with total sum n.
            let mut images: HashSet<(Vec<Vec<usize>>, Vec<Vec<usize>>)> = HashSet::new();
            let mut num_matrices = 0usize;
            for a in 0..=n {
                for b in 0..=n - a {
                    for c in 0..=n - a - b {
                        let d = n - a - b - c;
                        let m = vec![vec![a, b], vec![c, d]];
                        let (p, q) = rsk(&m);
                        let key = (p.rows().to_vec(), q.rows().to_vec());
                        assert!(pairs.contains(&key), "image not a bounded pair: {:?}", m);
                        assert!(images.insert(key), "not injective at {:?}", m);
                        num_matrices += 1;
                    }
                }
            }
            assert_eq!(images, pairs, "RSK not surjective for n = {}", n);
            assert_eq!(num_matrices, expected_counts[n]);
            assert_eq!(pairs.len(), expected_counts[n]);
        }
    }

    #[test]
    fn test_inverse_rsk_rejects_invalid_pairs() {
        // Different shapes.
        let p = Tableau::new(vec![vec![1, 2]]).unwrap();
        let q = Tableau::new(vec![vec![1], vec![2]]).unwrap();
        assert!(inverse_rsk(&p, &q, 2, 2).is_none());
        // Not semistandard.
        let bad = Tableau::new(vec![vec![2, 1]]).unwrap();
        let ok = Tableau::new(vec![vec![1, 1]]).unwrap();
        assert!(inverse_rsk(&bad, &ok, 2, 2).is_none());
        // Entries exceeding the requested matrix dimensions.
        let p = Tableau::new(vec![vec![1, 3]]).unwrap();
        let q = Tableau::new(vec![vec![1, 1]]).unwrap();
        assert!(inverse_rsk(&p, &q, 2, 2).is_none());
        assert_eq!(
            inverse_rsk(&p, &q, 1, 3).unwrap(),
            vec![vec![1, 0, 1]]
        );
    }
}
