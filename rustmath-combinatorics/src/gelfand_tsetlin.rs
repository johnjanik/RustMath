//! Gelfand-Tsetlin patterns and their bijection with semistandard tableaux
//!
//! A Gelfand-Tsetlin (GT) pattern is a triangular array of integers satisfying
//! certain interlacing conditions. They arise in representation theory and have
//! a natural bijection with semistandard Young tableaux (SSYT).
//!
//! # Structure
//!
//! A GT pattern is a triangular array:
//! ```text
//! a_{n,1}  a_{n,2}  ...  a_{n,n}
//!   a_{n-1,1}  a_{n-1,2}  ...  a_{n-1,n-1}
//!     ...
//!       a_{1,1}
//! ```
//!
//! satisfying the interlacing conditions:
//! - a_{i,j} ≥ a_{i-1,j} ≥ a_{i,j+1} for all valid i, j
//!
//! # Bijection with SSYT
//!
//! Given a GT pattern with top row (a₁, a₂, ..., aₙ) and bottom row (b₁, b₂, ..., bₘ),
//! the corresponding semistandard tableau has:
//! - Shape given by the bottom row differences
//! - Content determined by the differences between consecutive rows
//!
//! # References
//!
//! - Stanley, R. P. (1999). Enumerative Combinatorics, Volume 2
//! - Fulton, W. (1997). Young Tableaux

use crate::partitions::Partition;
use crate::tableaux::Tableau;

/// A Gelfand-Tsetlin pattern - a triangular array with interlacing conditions
///
/// The pattern is stored as rows from top (longest) to bottom (shortest).
/// Row i has length n - i + 1 for a pattern with n rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GelfandTsetlinPattern {
    /// Rows of the pattern, from top (longest) to bottom (shortest)
    rows: Vec<Vec<i64>>,
}

impl GelfandTsetlinPattern {
    /// Create a new Gelfand-Tsetlin pattern from rows
    ///
    /// Returns None if the rows don't satisfy the interlacing conditions or
    /// don't form a proper triangular shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::gelfand_tsetlin::GelfandTsetlinPattern;
    ///
    /// // Valid GT pattern
    /// let pattern = GelfandTsetlinPattern::new(vec![
    ///     vec![3, 2, 1],
    ///     vec![2, 1],
    ///     vec![1],
    /// ]);
    /// assert!(pattern.is_some());
    /// ```
    pub fn new(rows: Vec<Vec<i64>>) -> Option<Self> {
        if rows.is_empty() {
            return Some(GelfandTsetlinPattern { rows: vec![] });
        }

        let n = rows[0].len();

        // Check triangular shape: row i should have length n - i
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n - i {
                return None;
            }
        }

        // Check that each row is weakly decreasing
        for row in &rows {
            for i in 1..row.len() {
                if row[i] > row[i - 1] {
                    return None;
                }
            }
        }

        // Check interlacing conditions. With `rows[0]` the top (longest) row and
        // `rows[i]` of length n - i, the upper row `rows[i-1]` must interlace the
        // lower row `rows[i]`:  upper[j] >= lower[j] >= upper[j+1].
        for i in 1..rows.len() {
            for j in 0..rows[i].len() {
                // upper[j] >= lower[j]
                if rows[i - 1][j] < rows[i][j] {
                    return None;
                }
                // lower[j] >= upper[j+1]
                if rows[i][j] < rows[i - 1][j + 1] {
                    return None;
                }
            }
        }

        Some(GelfandTsetlinPattern { rows })
    }

    /// Get the rows of the pattern
    pub fn rows(&self) -> &[Vec<i64>] {
        &self.rows
    }

    /// Get the number of rows in the pattern
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Get the top row (longest row)
    pub fn top_row(&self) -> Option<&[i64]> {
        self.rows.first().map(|r| r.as_slice())
    }

    /// Get the bottom row (shortest row, typically a single element)
    pub fn bottom_row(&self) -> Option<&[i64]> {
        self.rows.last().map(|r| r.as_slice())
    }

    /// Check if the pattern is valid (satisfies interlacing conditions)
    ///
    /// This is automatically checked during construction, but can be useful
    /// for verification.
    pub fn is_valid(&self) -> bool {
        if self.rows.is_empty() {
            return true;
        }

        let n = self.rows[0].len();

        // Check shape
        for (i, row) in self.rows.iter().enumerate() {
            if row.len() != n - i {
                return false;
            }
        }

        // Check that each row is weakly decreasing
        for row in &self.rows {
            for i in 1..row.len() {
                if row[i] > row[i - 1] {
                    return false;
                }
            }
        }

        // Check interlacing conditions: upper[j] >= lower[j] >= upper[j+1].
        for i in 1..self.rows.len() {
            for j in 0..self.rows[i].len() {
                if self.rows[i - 1][j] < self.rows[i][j] || self.rows[i][j] < self.rows[i - 1][j + 1] {
                    return false;
                }
            }
        }

        true
    }

    /// Convert the GT pattern to a semistandard Young tableau (SSYT)
    ///
    /// The bijection works as follows:
    /// 1. The shape of the tableau is determined by the differences between
    ///    consecutive values in the bottom row (extended appropriately)
    /// 2. The filling is constructed by reading the differences between rows
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::gelfand_tsetlin::GelfandTsetlinPattern;
    ///
    /// let pattern = GelfandTsetlinPattern::new(vec![
    ///     vec![3, 2, 1],
    ///     vec![2, 1],
    ///     vec![1],
    /// ]).unwrap();
    ///
    /// let tableau = pattern.to_tableau().unwrap();
    /// let rows: Vec<Vec<usize>> = tableau.rows().to_vec();
    /// assert_eq!(rows, vec![vec![1, 2, 3], vec![2, 3], vec![3]]);
    /// ```
    pub fn to_tableau(&self) -> Option<Tableau> {
        if self.rows.is_empty() {
            return Tableau::new(vec![]);
        }

        // `rows[0]` is the top (longest) row, and `rows[i]` has length n - i, so the
        // pattern is only a full triangle when it has exactly n rows.
        let n = self.rows[0].len();
        if self.rows.len() != n {
            return None;
        }

        // Write lambda^(k) for the k-th row counted from the bottom, i.e.
        // lambda^(k) = rows[n - k], which has length k. Under the GT <-> SSYT
        // bijection, row i of the tableau holds exactly
        //     lambda^(k)_i - lambda^(k-1)_i
        // entries equal to k. Interlacing makes each such difference non-negative,
        // and the length of row i telescopes to lambda^(n)_i, the correct shape.
        let part = |k: usize, i: usize| -> i64 {
            if k == 0 || i > k {
                0
            } else {
                self.rows[n - k].get(i - 1).copied().unwrap_or(0)
            }
        };

        let mut tableau_rows: Vec<Vec<usize>> = Vec::new();
        for i in 1..=n {
            let mut row: Vec<usize> = Vec::new();
            for k in 1..=n {
                // A malformed pattern can make this negative; converting rather than
                // casting keeps it from wrapping to a near-usize::MAX repeat count.
                let mult = usize::try_from(part(k, i) - part(k - 1, i)).ok()?;
                row.extend(std::iter::repeat(k).take(mult));
            }
            // lambda^(n) is weakly decreasing, so the first empty row ends the shape.
            if row.is_empty() {
                break;
            }
            tableau_rows.push(row);
        }

        Tableau::new(tableau_rows)
    }

    /// Create a GT pattern from a semistandard Young tableau
    ///
    /// This is the inverse of `to_tableau()`. Given an SSYT, construct the
    /// corresponding GT pattern.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::gelfand_tsetlin::GelfandTsetlinPattern;
    /// use rustmath_combinatorics::Tableau;
    ///
    /// let tableau = Tableau::new(vec![
    ///     vec![1, 1, 2],
    ///     vec![2, 3],
    /// ]).unwrap();
    ///
    /// let pattern = GelfandTsetlinPattern::from_tableau(&tableau).unwrap();
    /// // `from_tableau` is the exact inverse of `to_tableau`.
    /// assert_eq!(pattern.to_tableau().unwrap().rows(), tableau.rows());
    /// ```
    pub fn from_tableau(tableau: &Tableau) -> Option<Self> {
        if tableau.size() == 0 {
            return Some(GelfandTsetlinPattern { rows: vec![] });
        }

        // `n` is the largest entry, hence the number of rows in the GT pattern.
        let n = tableau.content().into_iter().max()?;

        // For the k-th row from the bottom (k = 1..=n) the pattern records
        // lambda^(k), the shape of the sub-tableau of cells with entry <= k:
        //     lambda^(k)_i = (# entries <= k in row i of the tableau).
        // This is exactly the shape whose row-content differences `to_tableau`
        // reads back, so the two maps are inverse. `Self::new` then validates the
        // interlacing, returning `None` for any tableau that is not semistandard.
        let mut gt_rows: Vec<Vec<i64>> = vec![Vec::new(); n];
        for k in 1..=n {
            let mut lambda_k = vec![0i64; k];
            for (i, count) in lambda_k.iter_mut().enumerate() {
                if let Some(row) = tableau.rows().get(i) {
                    *count = row.iter().filter(|&&entry| entry <= k).count() as i64;
                }
            }
            // lambda^(k) is stored as `rows[n - k]` (length k).
            gt_rows[n - k] = lambda_k;
        }

        // Validate interlacing (and reject non-semistandard input) and return.
        Self::new(gt_rows)
    }

    /// Generate all Gelfand-Tsetlin patterns with a given top row
    ///
    /// This produces all patterns that satisfy the interlacing conditions
    /// for the given top row.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::gelfand_tsetlin::GelfandTsetlinPattern;
    ///
    /// let patterns = GelfandTsetlinPattern::all_with_top_row(vec![3, 2, 1]);
    /// assert!(patterns.len() > 0);
    /// ```
    pub fn all_with_top_row(top_row: Vec<i64>) -> Vec<Self> {
        let mut result = Vec::new();

        if top_row.is_empty() {
            return vec![GelfandTsetlinPattern { rows: vec![] }];
        }

        let n = top_row.len();
        let mut current_rows = vec![top_row.clone()];

        generate_gt_patterns_recursive(&mut current_rows, n - 1, &mut result);

        result
    }

    /// Display the pattern as a string with proper formatting
    pub fn to_string(&self) -> String {
        let mut result = String::new();

        for (i, row) in self.rows.iter().enumerate() {
            // Add indentation
            for _ in 0..i {
                result.push_str("  ");
            }

            // Add row elements
            let row_str: Vec<String> = row.iter().map(|x| x.to_string()).collect();
            result.push_str(&row_str.join("  "));
            result.push('\n');
        }

        result
    }
}

/// Helper function to recursively generate all GT patterns
fn generate_gt_patterns_recursive(
    current_rows: &mut Vec<Vec<i64>>,
    remaining: usize,
    result: &mut Vec<GelfandTsetlinPattern>,
) {
    if remaining == 0 {
        // Base case: we've built the complete pattern
        result.push(GelfandTsetlinPattern {
            rows: current_rows.clone(),
        });
        return;
    }

    // We need to generate the next row (of length 'remaining')
    let prev_row = current_rows.last().unwrap().clone(); // Clone to avoid borrow issues
    let new_row_len = remaining;

    // Generate all valid next rows
    let mut new_row = vec![0i64; new_row_len];

    generate_next_row_recursive(
        &prev_row,
        &mut new_row,
        0,
        current_rows,
        remaining,
        result,
    );
}

/// Helper to generate all valid values for a new row
fn generate_next_row_recursive(
    prev_row: &[i64],
    new_row: &mut [i64],
    pos: usize,
    current_rows: &mut Vec<Vec<i64>>,
    remaining: usize,
    result: &mut Vec<GelfandTsetlinPattern>,
) {
    if pos == new_row.len() {
        // We've filled the entire new row
        current_rows.push(new_row.to_vec());
        generate_gt_patterns_recursive(current_rows, remaining - 1, result);
        current_rows.pop();
        return;
    }

    // Determine the valid range for new_row[pos].
    // Interlacing condition: prev_row[pos] >= new_row[pos] >= prev_row[pos + 1].
    // Because prev_row is weakly decreasing, this also forces new_row itself to be
    // weakly decreasing (new_row[pos-1] >= prev_row[pos] >= new_row[pos]), so no
    // extra left constraint is needed.
    let max_val = prev_row[pos];
    let min_val = prev_row[pos + 1];

    // Try all valid values
    for val in min_val..=max_val {
        new_row[pos] = val;
        generate_next_row_recursive(
            prev_row,
            new_row,
            pos + 1,
            current_rows,
            remaining,
            result,
        );
    }
}

/// Iterator over all Gelfand-Tsetlin patterns with a given top row
pub struct GelfandTsetlinIterator {
    patterns: Vec<GelfandTsetlinPattern>,
    index: usize,
}

impl GelfandTsetlinIterator {
    /// Create a new iterator for patterns with the given top row
    pub fn new(top_row: Vec<i64>) -> Self {
        let patterns = GelfandTsetlinPattern::all_with_top_row(top_row);
        GelfandTsetlinIterator { patterns, index: 0 }
    }
}

impl Iterator for GelfandTsetlinIterator {
    type Item = GelfandTsetlinPattern;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.patterns.len() {
            let pattern = self.patterns[self.index].clone();
            self.index += 1;
            Some(pattern)
        } else {
            None
        }
    }
}

/// Generate all GT patterns with a given top row
pub fn gelfand_tsetlin_patterns(top_row: Vec<i64>) -> GelfandTsetlinIterator {
    GelfandTsetlinIterator::new(top_row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gt_pattern_creation() {
        // Valid GT pattern
        let pattern = GelfandTsetlinPattern::new(vec![
            vec![3, 2, 1],
            vec![2, 1],
            vec![1],
        ]);
        assert!(pattern.is_some());

        let p = pattern.unwrap();
        assert_eq!(p.num_rows(), 3);
        assert_eq!(p.top_row(), Some(&[3, 2, 1][..]));
        assert_eq!(p.bottom_row(), Some(&[1][..]));
    }

    #[test]
    fn test_invalid_shape() {
        // Invalid - not triangular
        let pattern = GelfandTsetlinPattern::new(vec![
            vec![3, 2, 1],
            vec![2, 1, 0], // Should have length 2, not 3
            vec![1],
        ]);
        assert!(pattern.is_none());
    }

    #[test]
    fn test_invalid_interlacing() {
        // Invalid - violates interlacing: for i=1, j=1 we need
        // lower[1] >= upper[2], i.e. 0 >= 1, which fails.
        let pattern = GelfandTsetlinPattern::new(vec![
            vec![4, 2, 1],
            vec![3, 0], // 0 >= upper[2]=1 fails
            vec![1],
        ]);
        assert!(pattern.is_none());
    }

    #[test]
    fn test_empty_pattern() {
        let pattern = GelfandTsetlinPattern::new(vec![]);
        assert!(pattern.is_some());
        assert_eq!(pattern.unwrap().num_rows(), 0);
    }

    #[test]
    fn test_single_element_pattern() {
        let pattern = GelfandTsetlinPattern::new(vec![vec![5]]);
        assert!(pattern.is_some());

        let p = pattern.unwrap();
        assert_eq!(p.num_rows(), 1);
        assert_eq!(p.top_row(), Some(&[5][..]));
        assert_eq!(p.bottom_row(), Some(&[5][..]));
    }

    #[test]
    fn test_generate_patterns_small() {
        // Generate all patterns with top row [2, 1]
        let patterns = GelfandTsetlinPattern::all_with_top_row(vec![2, 1]);

        // Should have some patterns
        assert!(patterns.len() > 0);

        // All should be valid
        for (i, pattern) in patterns.iter().enumerate() {
            if !pattern.is_valid() {
                eprintln!("Invalid pattern {}:", i);
                eprintln!("{}", pattern.to_string());
                eprintln!("Rows: {:?}", pattern.rows());
            }
            assert!(pattern.is_valid(), "Pattern {} is invalid", i);
            assert_eq!(pattern.top_row(), Some(&[2, 1][..]));
        }
    }

    #[test]
    fn test_generate_patterns_counts() {
        // For top row [a, b, ...], the number of patterns depends on the
        // specific values and the interlacing conditions

        // Simple case: [1, 0]
        // The next row's single entry x must satisfy 1 >= x >= 0, so x in {0, 1}:
        // two patterns (matching the two SSYT of shape (1) filled with 1 or 2).
        let patterns1 = GelfandTsetlinPattern::all_with_top_row(vec![1, 0]);
        assert_eq!(patterns1.len(), 2);
        assert_eq!(patterns1[0].rows(), &[vec![1, 0], vec![0]]);
        assert_eq!(patterns1[1].rows(), &[vec![1, 0], vec![1]]);

        // [2, 1, 0] - more complex, multiple valid patterns
        let patterns2 = GelfandTsetlinPattern::all_with_top_row(vec![2, 1, 0]);
        // Each pattern should be valid
        for p in &patterns2 {
            assert!(p.is_valid(), "Pattern should be valid: {:?}", p);
        }
        // The exact count depends on all valid interlacing sequences
        assert!(patterns2.len() > 0);

        // [2, 1] - should have 2 patterns
        let patterns3 = GelfandTsetlinPattern::all_with_top_row(vec![2, 1]);
        assert!(patterns3.len() > 0);
        for p in &patterns3 {
            assert!(p.is_valid());
        }
    }

    #[test]
    fn test_iterator() {
        let iter = gelfand_tsetlin_patterns(vec![2, 1, 0]);
        let patterns: Vec<_> = iter.collect();

        assert!(patterns.len() > 0);

        for pattern in patterns {
            assert_eq!(pattern.top_row(), Some(&[2, 1, 0][..]));
            assert!(pattern.is_valid());
        }
    }

    #[test]
    fn test_to_string_formatting() {
        let pattern = GelfandTsetlinPattern::new(vec![
            vec![3, 2, 1],
            vec![2, 1],
            vec![1],
        ]).unwrap();

        let s = pattern.to_string();
        assert!(s.contains("3"));
        assert!(s.contains("2"));
        assert!(s.contains("1"));
    }

    #[test]
    fn test_tableau_bijection_simple() {
        // Create a simple SSYT
        let tableau = Tableau::new(vec![
            vec![1, 2],
            vec![2],
        ]).unwrap();

        assert!(tableau.is_semistandard());

        // `from_tableau` and `to_tableau` are exact inverses.
        let pattern = GelfandTsetlinPattern::from_tableau(&tableau).unwrap();
        assert!(pattern.is_valid());
        let tableau2 = pattern.to_tableau().unwrap();
        assert_eq!(tableau2.rows(), tableau.rows());
    }

    #[test]
    fn test_gt_ssyt_round_trip() {
        // Valid GT patterns (including patterns the old backwards interlacing
        // check wrongly rejected) whose SSYT uses every value 1..=n, so
        // `from_tableau . to_tableau` is the identity on them.
        let cases = vec![
            vec![vec![3, 2, 1], vec![2, 1], vec![1]],
            vec![vec![4, 2, 1, 0], vec![3, 2, 0], vec![2, 1], vec![1]],
        ];
        for rows in cases {
            let p = GelfandTsetlinPattern::new(rows.clone())
                .unwrap_or_else(|| panic!("should be a valid GT pattern: {:?}", rows));
            let t = p.to_tableau().expect("valid pattern converts to a tableau");
            let back = GelfandTsetlinPattern::from_tableau(&t)
                .expect("tableau converts back to a GT pattern");
            assert_eq!(back, p, "round trip failed for {:?}", rows);
        }
    }

    #[test]
    fn test_previously_rejected_patterns_now_valid() {
        // Regression: the interlacing check in `new()` was backwards and rejected
        // these valid Gelfand-Tsetlin patterns.
        assert!(GelfandTsetlinPattern::new(vec![vec![2, 0], vec![1]]).is_some());
        assert!(GelfandTsetlinPattern::new(vec![
            vec![4, 2, 1, 0],
            vec![3, 2, 0],
            vec![2, 1],
            vec![1],
        ])
        .is_some());
    }

    #[test]
    fn test_specific_interlacing_conditions() {
        // Test specific interlacing: a[i-1][j] >= a[i][j] and a[i-1][j+1] >= a[i][j]

        // Simple valid pattern
        let pattern = GelfandTsetlinPattern::new(vec![
            vec![3, 2, 1],
            vec![2, 1],
            vec![1],
        ]);
        assert!(pattern.is_some());

        // Another valid pattern
        let pattern2 = GelfandTsetlinPattern::new(vec![
            vec![4, 3, 2, 1],
            vec![3, 2, 1],
            vec![2, 1],
            vec![1],
        ]);
        assert!(pattern2.is_some());

        // Test that a violation is caught - decreasing values in a row
        let bad1 = GelfandTsetlinPattern::new(vec![
            vec![5, 3, 4, 0], // Not weakly decreasing!
            vec![3, 2, 1],
            vec![2, 1],
            vec![1],
        ]);
        assert!(bad1.is_none());

        // Test interlacing violation: for i=1, j=0 we need lower[0] >= upper[1],
        // i.e. 2 >= 3, which fails.
        let bad2 = GelfandTsetlinPattern::new(vec![
            vec![5, 3, 2, 0],
            vec![2, 2, 1], // lower[0]=2 < upper[1]=3
            vec![2, 1],
            vec![1],
        ]);
        assert!(bad2.is_none());
    }
}
