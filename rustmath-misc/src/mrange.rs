///! Multi-dimensional range iterators and Cartesian products
///!
///! This module provides iterators for multi-dimensional ranges and Cartesian products,
///! corresponding to SageMath's sage.misc.mrange module.

use std::vec::Vec;

/// Multi-dimensional range iterator that generates n-tuples with entries from 0 to sizes[i]-1
///
/// This generates all possible combinations of indices where each index i ranges from
/// 0 to sizes[i]-1, in lexicographic order.
///
/// # Arguments
/// * `sizes` - Vector of upper bounds for each dimension
///
/// # Examples
/// ```
/// use rustmath_misc::mrange::xmrange;
///
/// let result: Vec<Vec<usize>> = xmrange(vec![3, 2]).collect();
/// assert_eq!(result, vec![
///     vec![0, 0], vec![0, 1],
///     vec![1, 0], vec![1, 1],
///     vec![2, 0], vec![2, 1],
/// ]);
/// ```
pub fn xmrange(sizes: Vec<usize>) -> impl Iterator<Item = Vec<usize>> {
    XMRange {
        sizes,
        current: None,
        exhausted: false,
    }
}

struct XMRange {
    sizes: Vec<usize>,
    current: Option<Vec<usize>>,
    exhausted: bool,
}

impl Iterator for XMRange {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if any size is 0 - if so, return empty iterator
        if self.sizes.iter().any(|&s| s == 0) {
            return None;
        }

        if self.exhausted {
            return None;
        }

        // Initialize on first call
        if self.current.is_none() {
            self.current = Some(vec![0; self.sizes.len()]);
            return self.current.clone();
        }

        let mut current = self.current.as_mut().unwrap();

        // Increment the indices (rightmost first)
        let mut i = current.len();
        while i > 0 {
            i -= 1;
            current[i] += 1;
            if current[i] < self.sizes[i] {
                return self.current.clone();
            }
            current[i] = 0;
        }

        // If we've rolled over all indices, we're done
        self.exhausted = true;
        None
    }
}

/// Cartesian product iterator over arbitrary iterables
///
/// Creates an iterator that yields all possible combinations by taking one element
/// from each input iterable.
///
/// # Arguments
/// * `iterables` - Vector of vectors to compute the Cartesian product over
///
/// # Examples
/// ```
/// use rustmath_misc::mrange::cartesian_product_iterator;
///
/// let result: Vec<Vec<i32>> = cartesian_product_iterator(vec![
///     vec![1, 2],
///     vec![10, 20],
/// ]).collect();
///
/// assert_eq!(result, vec![
///     vec![1, 10], vec![1, 20],
///     vec![2, 10], vec![2, 20],
/// ]);
/// ```
pub fn cartesian_product_iterator<T: Clone>(
    iterables: Vec<Vec<T>>,
) -> impl Iterator<Item = Vec<T>> {
    XMRangeIter {
        iterables,
        current_indices: None,
        exhausted: false,
    }
}

struct XMRangeIter<T: Clone> {
    iterables: Vec<Vec<T>>,
    current_indices: Option<Vec<usize>>,
    exhausted: bool,
}

impl<T: Clone> Iterator for XMRangeIter<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if any iterable is empty - if so, return empty iterator
        if self.iterables.iter().any(|v| v.is_empty()) {
            return None;
        }

        if self.exhausted {
            return None;
        }

        // Initialize on first call
        if self.current_indices.is_none() {
            self.current_indices = Some(vec![0; self.iterables.len()]);
            let result: Vec<T> = self
                .iterables
                .iter()
                .map(|v| v[0].clone())
                .collect();
            return Some(result);
        }

        let mut indices = self.current_indices.as_mut().unwrap();

        // Increment the indices (rightmost first)
        let mut i = indices.len();
        while i > 0 {
            i -= 1;
            indices[i] += 1;
            if indices[i] < self.iterables[i].len() {
                let result: Vec<T> = indices
                    .iter()
                    .enumerate()
                    .map(|(idx, &pos)| self.iterables[idx][pos].clone())
                    .collect();
                return Some(result);
            }
            indices[i] = 0;
        }

        // If we've rolled over all indices, we're done
        self.exhausted = true;
        None
    }
}

/// Cantor diagonal product iterator
///
/// Generates the Cartesian product of input iterables but in diagonal (Cantor) enumeration order.
/// This is useful for infinite iterables as it ensures every element will eventually be reached.
///
/// # Arguments
/// * `iterables` - Vector of vectors to compute the diagonal product over
///
/// # Examples
/// ```
/// use rustmath_misc::mrange::cantor_product;
///
/// let result: Vec<Vec<i32>> = cantor_product(vec![
///     vec![1, 2, 3],
///     vec![10, 20],
/// ]).collect();
///
/// // Elements are enumerated in diagonal order
/// assert_eq!(result.len(), 6);
/// assert!(result.contains(&vec![1, 10]));
/// assert!(result.contains(&vec![2, 20]));
/// ```
pub fn cantor_product<T: Clone>(iterables: Vec<Vec<T>>) -> impl Iterator<Item = Vec<T>> {
    CantorProduct {
        iterables,
        diagonal: 0,
        position_in_diagonal: 0,
    }
}

struct CantorProduct<T: Clone> {
    iterables: Vec<Vec<T>>,
    diagonal: usize,
    position_in_diagonal: usize,
}

impl<T: Clone> Iterator for CantorProduct<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if any iterable is empty
        if self.iterables.iter().any(|v| v.is_empty()) {
            return None;
        }

        let n = self.iterables.len();

        loop {
            // Try to generate an element at the current diagonal position
            if let Some(indices) = self.diagonal_indices(self.diagonal, self.position_in_diagonal, n) {
                // Check if all indices are within bounds
                let all_in_bounds = indices
                    .iter()
                    .enumerate()
                    .all(|(i, &idx)| idx < self.iterables[i].len());

                if all_in_bounds {
                    let result: Vec<T> = indices
                        .iter()
                        .enumerate()
                        .map(|(i, &idx)| self.iterables[i][idx].clone())
                        .collect();

                    self.position_in_diagonal += 1;
                    return Some(result);
                }
            }

            // Move to next position in diagonal
            self.position_in_diagonal += 1;

            // Check if we've exhausted this diagonal
            let diagonal_size = self.diagonal_size(self.diagonal, n);
            if self.position_in_diagonal >= diagonal_size {
                self.diagonal += 1;
                self.position_in_diagonal = 0;

                // Check if we've enumerated all possible elements
                let max_sum: usize = self.iterables.iter().map(|v| v.len() - 1).sum();
                if self.diagonal > max_sum {
                    return None;
                }
            }
        }
    }
}

impl<T: Clone> CantorProduct<T> {
    /// Calculate the size of a diagonal (number of partitions of `diagonal` into `n` parts)
    fn diagonal_size(&self, diagonal: usize, n: usize) -> usize {
        // This is the number of ways to write diagonal as a sum of n non-negative integers
        // which is C(diagonal + n - 1, n - 1)
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }
        binomial(diagonal + n - 1, n - 1)
    }

    /// Get the indices for the k-th element in the diagonal
    fn diagonal_indices(&self, diagonal: usize, position: usize, n: usize) -> Option<Vec<usize>> {
        if n == 0 {
            return None;
        }

        // Generate all partitions of `diagonal` into `n` parts
        let partitions = partitions_of_sum(diagonal, n);
        if position < partitions.len() {
            Some(partitions[position].clone())
        } else {
            None
        }
    }
}

/// Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k); // Take advantage of symmetry
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Generate all partitions of `sum` into exactly `parts` non-negative integers
fn partitions_of_sum(sum: usize, parts: usize) -> Vec<Vec<usize>> {
    if parts == 0 {
        return vec![];
    }
    if parts == 1 {
        return vec![vec![sum]];
    }

    let mut result = Vec::new();
    partition_recursive(sum, parts, vec![], &mut result);
    result
}

fn partition_recursive(
    remaining: usize,
    parts_left: usize,
    current: Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if parts_left == 1 {
        let mut partition = current.clone();
        partition.push(remaining);
        result.push(partition);
        return;
    }

    for i in 0..=remaining {
        let mut next = current.clone();
        next.push(i);
        partition_recursive(remaining - i, parts_left - 1, next, result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xmrange() {
        let result: Vec<Vec<usize>> = xmrange(vec![3, 2]).collect();
        assert_eq!(
            result,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![1, 0],
                vec![1, 1],
                vec![2, 0],
                vec![2, 1],
            ]
        );
    }

    #[test]
    fn test_xmrange_empty() {
        let result: Vec<Vec<usize>> = xmrange(vec![3, 0, 2]).collect();
        let expected: Vec<Vec<usize>> = vec![];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_xmrange_single() {
        let result: Vec<Vec<usize>> = xmrange(vec![4]).collect();
        assert_eq!(result, vec![vec![0], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_cartesian_product_iterator() {
        let result: Vec<Vec<i32>> = cartesian_product_iterator(vec![vec![1, 2], vec![10, 20]]).collect();
        assert_eq!(
            result,
            vec![vec![1, 10], vec![1, 20], vec![2, 10], vec![2, 20],]
        );
    }

    #[test]
    fn test_cartesian_product_empty() {
        let result: Vec<Vec<i32>> =
            cartesian_product_iterator(vec![vec![1, 2], vec![], vec![100]]).collect();
        let expected: Vec<Vec<i32>> = vec![];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cartesian_product_single() {
        let result: Vec<Vec<i32>> = cartesian_product_iterator(vec![vec![1, 2, 3]]).collect();
        assert_eq!(result, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_cantor_product() {
        let result: Vec<Vec<i32>> = cantor_product(vec![vec![1, 2], vec![10, 20]]).collect();
        assert_eq!(result.len(), 4);
        assert!(result.contains(&vec![1, 10]));
        assert!(result.contains(&vec![1, 20]));
        assert!(result.contains(&vec![2, 10]));
        assert!(result.contains(&vec![2, 20]));
    }

    #[test]
    fn test_cantor_product_three() {
        let result: Vec<Vec<i32>> =
            cantor_product(vec![vec![1, 2], vec![10, 20], vec![100, 200]]).collect();
        assert_eq!(result.len(), 8); // 2 * 2 * 2
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(6, 3), 20);
        assert_eq!(binomial(10, 0), 1);
        assert_eq!(binomial(10, 10), 1);
        assert_eq!(binomial(5, 10), 0);
    }

    #[test]
    fn test_partitions_of_sum() {
        let partitions = partitions_of_sum(3, 2);
        assert_eq!(partitions.len(), 4);
        assert!(partitions.contains(&vec![0, 3]));
        assert!(partitions.contains(&vec![1, 2]));
        assert!(partitions.contains(&vec![2, 1]));
        assert!(partitions.contains(&vec![3, 0]));
    }
}
