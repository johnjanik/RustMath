//! Perfect matchings and chord diagrams
//!
//! A perfect matching is a partition of vertices into pairs, where each vertex
//! appears in exactly one pair. When vertices are arranged on a circle, the
//! matching can be visualized as a chord diagram, where each pair is represented
//! as a chord connecting two points.
//!
//! A matching is noncrossing if no two chords intersect (except at endpoints).
//! The number of noncrossing perfect matchings on 2n points equals the nth
//! Catalan number.

/// A perfect matching on 2n vertices
///
/// Vertices are labeled 0, 1, 2, ..., 2n-1. In the chord diagram representation,
/// these vertices are arranged clockwise around a circle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerfectMatching {
    /// The matching as pairs of vertices
    pairs: Vec<(usize, usize)>,
    n: usize,
}

impl PerfectMatching {
    /// Create a new perfect matching
    pub fn new(pairs: Vec<(usize, usize)>, n: usize) -> Option<Self> {
        // Verify it's a valid perfect matching
        let mut seen = vec![false; 2 * n];

        for &(a, b) in &pairs {
            if a >= 2 * n || b >= 2 * n || seen[a] || seen[b] {
                return None;
            }
            seen[a] = true;
            seen[b] = true;
        }

        // Check all vertices are matched
        if !seen.iter().all(|&x| x) {
            return None;
        }

        Some(PerfectMatching { pairs, n })
    }

    /// Get the pairs in the matching
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Get the number of pairs
    pub fn size(&self) -> usize {
        self.n
    }

    /// Check if two chords (arcs) cross in the chord diagram representation
    ///
    /// Two chords (a,b) and (c,d) cross if the points appear in one of these
    /// cyclic orders around the circle: a < c < b < d or c < a < d < b
    fn chords_cross(a: usize, b: usize, c: usize, d: usize) -> bool {
        // Normalize so that a < b and c < d
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        let (c, d) = if c < d { (c, d) } else { (d, c) };

        // Check if they cross: one of {c, d} is between a and b, but not both
        (a < c && c < b && (d < a || d > b)) || (a < d && d < b && (c < a || c > b))
    }

    /// Check if this matching is noncrossing in the chord diagram representation
    ///
    /// A matching is noncrossing if no two chords (pairs) cross when vertices
    /// are arranged on a circle in order 0, 1, 2, ..., 2n-1.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::PerfectMatching;
    ///
    /// // Noncrossing matching: {(0,1), (2,3)}
    /// let m1 = PerfectMatching::new(vec![(0,1), (2,3)], 2).unwrap();
    /// assert!(m1.is_noncrossing());
    ///
    /// // Crossing matching: {(0,2), (1,3)}
    /// let m2 = PerfectMatching::new(vec![(0,2), (1,3)], 2).unwrap();
    /// assert!(!m2.is_noncrossing());
    /// ```
    pub fn is_noncrossing(&self) -> bool {
        // Check all pairs of chords
        for i in 0..self.pairs.len() {
            for j in (i + 1)..self.pairs.len() {
                let (a, b) = self.pairs[i];
                let (c, d) = self.pairs[j];
                if Self::chords_cross(a, b, c, d) {
                    return false;
                }
            }
        }
        true
    }

    /// Convert to chord diagram representation
    ///
    /// Returns a vector of length 2n where each element indicates which vertex
    /// it is paired with in the matching. This is useful for visualizing the
    /// chord diagram.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_combinatorics::PerfectMatching;
    ///
    /// let m = PerfectMatching::new(vec![(0,3), (1,2)], 2).unwrap();
    /// assert_eq!(m.to_chord_diagram(), vec![3, 2, 1, 0]);
    /// ```
    pub fn to_chord_diagram(&self) -> Vec<usize> {
        let mut diagram = vec![0; 2 * self.n];
        for &(a, b) in &self.pairs {
            diagram[a] = b;
            diagram[b] = a;
        }
        diagram
    }

    /// Create a perfect matching from a chord diagram representation
    ///
    /// The chord diagram is a vector where diagram[i] indicates which vertex
    /// vertex i is paired with.
    pub fn from_chord_diagram(diagram: Vec<usize>) -> Option<Self> {
        let len = diagram.len();
        if len % 2 != 0 {
            return None;
        }

        let n = len / 2;
        let mut pairs = Vec::new();
        let mut seen = vec![false; len];

        for i in 0..len {
            if seen[i] {
                continue;
            }

            let j = diagram[i];
            if j >= len || seen[j] || diagram[j] != i {
                return None;
            }

            pairs.push((i, j));
            seen[i] = true;
            seen[j] = true;
        }

        Self::new(pairs, n)
    }
}

/// Generate all perfect matchings on 2n vertices
pub fn perfect_matchings(n: usize) -> Vec<PerfectMatching> {
    if n == 0 {
        return vec![PerfectMatching {
            pairs: vec![],
            n: 0,
        }];
    }

    let mut result = Vec::new();
    let mut current_pairs = Vec::new();
    let mut available: Vec<usize> = (0..2 * n).collect();

    generate_perfect_matchings(&mut current_pairs, &mut available, n, &mut result);

    result
}

fn generate_perfect_matchings(
    current: &mut Vec<(usize, usize)>,
    available: &mut Vec<usize>,
    n: usize,
    result: &mut Vec<PerfectMatching>,
) {
    if available.is_empty() {
        result.push(PerfectMatching {
            pairs: current.clone(),
            n,
        });
        return;
    }

    // Take the first available vertex and try matching it with all others
    let first = available[0];

    for i in 1..available.len() {
        let second = available[i];

        // Create the pair
        current.push((first, second));

        // Remove both from available
        let mut new_available = available.clone();
        new_available.retain(|&x| x != first && x != second);

        generate_perfect_matchings(current, &mut new_available, n, result);

        current.pop();
    }
}

/// Generate all noncrossing perfect matchings on 2n vertices
///
/// A noncrossing matching is one where, when vertices are arranged on a circle
/// labeled 0, 1, 2, ..., 2n-1, no two chords (pairs) cross.
///
/// The number of noncrossing perfect matchings on 2n vertices equals the nth
/// Catalan number: C_n = (1/(n+1)) * C(2n, n).
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::noncrossing_perfect_matchings;
///
/// // For n=2, there are 2 noncrossing matchings (C_2 = 2):
/// // {(0,1), (2,3)} and {(0,3), (1,2)}
/// let matchings = noncrossing_perfect_matchings(2);
/// assert_eq!(matchings.len(), 2);
///
/// // For n=3, there are 5 noncrossing matchings (C_3 = 5)
/// let matchings3 = noncrossing_perfect_matchings(3);
/// assert_eq!(matchings3.len(), 5);
/// ```
pub fn noncrossing_perfect_matchings(n: usize) -> Vec<PerfectMatching> {
    if n == 0 {
        return vec![PerfectMatching {
            pairs: vec![],
            n: 0,
        }];
    }

    let mut result = Vec::new();
    let mut current_pairs = Vec::new();

    generate_noncrossing_matchings(&mut current_pairs, 0, 2 * n, n, &mut result);

    result
}

/// Recursive helper to generate noncrossing matchings
///
/// The algorithm works by always matching vertex 0 with some vertex k,
/// which splits the circle into two independent regions:
/// - Vertices 1, 2, ..., k-1 (must form a noncrossing matching)
/// - Vertices k+1, k+2, ..., 2n-1 (must form a noncrossing matching)
///
/// This gives the Catalan number recurrence:
/// C_n = sum_{i=0}^{n-1} C_i * C_{n-1-i}
fn generate_noncrossing_matchings(
    current: &mut Vec<(usize, usize)>,
    offset: usize,
    size: usize,
    n: usize,
    result: &mut Vec<PerfectMatching>,
) {
    if size == 0 {
        if current.len() == n {
            result.push(PerfectMatching {
                pairs: current.clone(),
                n,
            });
        }
        return;
    }

    // Match the first vertex (offset) with each possible partner
    // For noncrossing, we can only match with vertices at odd distances
    // (to ensure both sides have an even number of vertices)
    let first = offset;

    for i in 1..size {
        if i % 2 == 1 {
            // Only match at odd positions to maintain even-sized regions
            let second = offset + i;

            // Create the pair
            current.push((first, second));

            // Left region: offset+1 to second-1
            let left_offset = offset + 1;
            let left_size = i - 1;

            // Recursively solve left region
            let mut left_result = Vec::new();
            if left_size > 0 {
                generate_noncrossing_matchings_region(
                    left_offset,
                    left_size,
                    n,
                    &mut left_result,
                );
            } else {
                left_result.push(Vec::new());
            }

            // Right region: second+1 to offset+size-1
            let right_offset = second + 1;
            let right_size = size - i - 1;

            let mut right_result = Vec::new();
            if right_size > 0 {
                generate_noncrossing_matchings_region(
                    right_offset,
                    right_size,
                    n,
                    &mut right_result,
                );
            } else {
                right_result.push(Vec::new());
            }

            // Combine left and right results
            for left_pairs in &left_result {
                for right_pairs in &right_result {
                    let mut combined = current.clone();
                    combined.extend(left_pairs);
                    combined.extend(right_pairs);

                    if combined.len() == n {
                        result.push(PerfectMatching {
                            pairs: combined,
                            n,
                        });
                    }
                }
            }

            current.pop();
        }
    }
}

/// Helper function to generate noncrossing matchings for a region
fn generate_noncrossing_matchings_region(
    offset: usize,
    size: usize,
    _n: usize,
    result: &mut Vec<Vec<(usize, usize)>>,
) {
    if size == 0 {
        result.push(Vec::new());
        return;
    }

    if size % 2 != 0 {
        return; // Can't have a perfect matching on odd number of vertices
    }

    let first = offset;

    for i in 1..size {
        if i % 2 == 1 {
            let second = offset + i;

            // Left region
            let left_offset = offset + 1;
            let left_size = i - 1;

            let mut left_result = Vec::new();
            generate_noncrossing_matchings_region(left_offset, left_size, _n, &mut left_result);

            // Right region
            let right_offset = second + 1;
            let right_size = size - i - 1;

            let mut right_result = Vec::new();
            generate_noncrossing_matchings_region(right_offset, right_size, _n, &mut right_result);

            // Combine
            for left_pairs in &left_result {
                for right_pairs in &right_result {
                    let mut combined = vec![(first, second)];
                    combined.extend(left_pairs);
                    combined.extend(right_pairs);
                    result.push(combined);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_matchings() {
        // Perfect matchings on 2 vertices: just one matching {(0,1)}
        let matchings1 = perfect_matchings(1);
        assert_eq!(matchings1.len(), 1);

        // Perfect matchings on 4 vertices: should be 3 matchings
        // {(0,1),(2,3)}, {(0,2),(1,3)}, {(0,3),(1,2)}
        let matchings2 = perfect_matchings(2);
        assert_eq!(matchings2.len(), 3);

        // Verify each is a valid matching
        for matching in &matchings2 {
            assert_eq!(matching.pairs().len(), 2);
        }
    }

    #[test]
    fn test_perfect_matching_validation() {
        // Valid matching on 4 vertices
        let matching = PerfectMatching::new(vec![(0, 1), (2, 3)], 2);
        assert!(matching.is_some());

        // Invalid - duplicate vertex
        let invalid = PerfectMatching::new(vec![(0, 1), (1, 2)], 2);
        assert!(invalid.is_none());

        // Invalid - missing vertex
        let invalid2 = PerfectMatching::new(vec![(0, 1)], 2);
        assert!(invalid2.is_none());
    }

    #[test]
    fn test_noncrossing_detection() {
        // Noncrossing: {(0,1), (2,3)} - adjacent pairs
        let m1 = PerfectMatching::new(vec![(0, 1), (2, 3)], 2).unwrap();
        assert!(m1.is_noncrossing());

        // Crossing: {(0,2), (1,3)} - chords cross
        let m2 = PerfectMatching::new(vec![(0, 2), (1, 3)], 2).unwrap();
        assert!(!m2.is_noncrossing());

        // Noncrossing: {(0,3), (1,2)} - nested pairs
        let m3 = PerfectMatching::new(vec![(0, 3), (1, 2)], 2).unwrap();
        assert!(m3.is_noncrossing());

        // Test with 6 vertices
        // Noncrossing: {(0,5), (1,2), (3,4)}
        let m4 = PerfectMatching::new(vec![(0, 5), (1, 2), (3, 4)], 3).unwrap();
        assert!(m4.is_noncrossing());

        // Crossing: {(0,3), (1,4), (2,5)}
        let m5 = PerfectMatching::new(vec![(0, 3), (1, 4), (2, 5)], 3).unwrap();
        assert!(!m5.is_noncrossing());
    }

    #[test]
    fn test_chord_diagram_conversion() {
        // Test to_chord_diagram
        let m = PerfectMatching::new(vec![(0, 3), (1, 2)], 2).unwrap();
        let diagram = m.to_chord_diagram();
        assert_eq!(diagram, vec![3, 2, 1, 0]);

        // Test from_chord_diagram
        let m2 = PerfectMatching::from_chord_diagram(vec![3, 2, 1, 0]).unwrap();
        assert_eq!(m2.pairs().len(), 2);

        // Test round-trip
        let original = PerfectMatching::new(vec![(0, 1), (2, 5), (3, 4)], 3).unwrap();
        let diagram = original.to_chord_diagram();
        let reconstructed = PerfectMatching::from_chord_diagram(diagram).unwrap();
        // Check that both have the same chord diagram representation
        assert_eq!(original.to_chord_diagram(), reconstructed.to_chord_diagram());
    }

    #[test]
    fn test_chord_diagram_invalid() {
        // Odd length - invalid
        assert!(PerfectMatching::from_chord_diagram(vec![1, 0, 2]).is_none());

        // Inconsistent pairing - invalid
        assert!(PerfectMatching::from_chord_diagram(vec![1, 2, 1, 0]).is_none());

        // Out of bounds - invalid
        assert!(PerfectMatching::from_chord_diagram(vec![5, 1, 1, 0]).is_none());
    }

    #[test]
    fn test_noncrossing_perfect_matchings() {
        // n=1: C_1 = 1 noncrossing matching
        let matchings1 = noncrossing_perfect_matchings(1);
        assert_eq!(matchings1.len(), 1);
        assert!(matchings1[0].is_noncrossing());

        // n=2: C_2 = 2 noncrossing matchings
        let matchings2 = noncrossing_perfect_matchings(2);
        assert_eq!(matchings2.len(), 2);
        for m in &matchings2 {
            assert!(m.is_noncrossing());
        }

        // n=3: C_3 = 5 noncrossing matchings
        let matchings3 = noncrossing_perfect_matchings(3);
        assert_eq!(matchings3.len(), 5);
        for m in &matchings3 {
            assert!(m.is_noncrossing());
        }

        // n=4: C_4 = 14 noncrossing matchings
        let matchings4 = noncrossing_perfect_matchings(4);
        assert_eq!(matchings4.len(), 14);
        for m in &matchings4 {
            assert!(m.is_noncrossing());
        }
    }

    #[test]
    fn test_noncrossing_vs_all_matchings() {
        // For n=2, out of 3 total matchings, 2 are noncrossing
        let all = perfect_matchings(2);
        let noncrossing = noncrossing_perfect_matchings(2);

        assert_eq!(all.len(), 3);
        assert_eq!(noncrossing.len(), 2);

        // Count how many of the all matchings are noncrossing
        let count_noncrossing = all.iter().filter(|m| m.is_noncrossing()).count();
        assert_eq!(count_noncrossing, 2);
    }

    #[test]
    fn test_catalan_correspondence() {
        // The number of noncrossing matchings should equal Catalan numbers
        // C_0 = 1, C_1 = 1, C_2 = 2, C_3 = 5, C_4 = 14, C_5 = 42

        assert_eq!(noncrossing_perfect_matchings(0).len(), 1); // C_0
        assert_eq!(noncrossing_perfect_matchings(1).len(), 1); // C_1
        assert_eq!(noncrossing_perfect_matchings(2).len(), 2); // C_2
        assert_eq!(noncrossing_perfect_matchings(3).len(), 5); // C_3
        assert_eq!(noncrossing_perfect_matchings(4).len(), 14); // C_4
    }

    #[test]
    fn test_specific_noncrossing_matching() {
        // Test a specific known noncrossing matching
        let m = PerfectMatching::new(vec![(0, 5), (1, 4), (2, 3)], 3).unwrap();
        assert!(m.is_noncrossing());

        // Visualize: points 0,1,2,3,4,5 on a circle
        // Chord 0-5 wraps around
        // Chord 1-4 is inside
        // Chord 2-3 is inside
        // None of these cross
    }

    #[test]
    fn test_crossing_chords() {
        // Test the chords_cross helper function directly
        assert!(PerfectMatching::chords_cross(0, 2, 1, 3)); // (0,2) and (1,3) cross
        assert!(!PerfectMatching::chords_cross(0, 1, 2, 3)); // (0,1) and (2,3) don't cross
        assert!(!PerfectMatching::chords_cross(0, 3, 1, 2)); // (0,3) and (1,2) don't cross (nested)
    }
}
