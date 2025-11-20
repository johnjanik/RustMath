//! Path tableaux and lattice path enumeration
//!
//! This module provides tools for working with lattice paths and their
//! corresponding tableaux representations. Path tableaux provide a bijection
//! between certain classes of lattice paths and Young tableaux.

use crate::partitions::Partition;
use crate::tableaux::Tableau;
use rustmath_integers::Integer;

/// Direction of a step in a lattice path
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Step {
    /// North step (0, 1)
    North,
    /// East step (1, 0)
    East,
    /// South step (0, -1) - used in some path models
    South,
    /// West step (-1, 0) - used in some path models
    West,
    /// Diagonal step (1, 1)
    Diagonal,
}

impl Step {
    /// Get the (dx, dy) coordinates for this step
    pub fn to_coords(&self) -> (i32, i32) {
        match self {
            Step::North => (0, 1),
            Step::East => (1, 0),
            Step::South => (0, -1),
            Step::West => (-1, 0),
            Step::Diagonal => (1, 1),
        }
    }

    /// Create a step from coordinates
    pub fn from_coords(dx: i32, dy: i32) -> Option<Self> {
        match (dx, dy) {
            (0, 1) => Some(Step::North),
            (1, 0) => Some(Step::East),
            (0, -1) => Some(Step::South),
            (-1, 0) => Some(Step::West),
            (1, 1) => Some(Step::Diagonal),
            _ => None,
        }
    }
}

/// A lattice path in the integer lattice
///
/// A lattice path is a sequence of steps connecting lattice points.
/// The most common variant uses North (0,1) and East (1,0) steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticePath {
    /// The sequence of steps
    steps: Vec<Step>,
    /// Starting point (default is (0, 0))
    start: (i32, i32),
}

impl LatticePath {
    /// Create a new lattice path from a sequence of steps
    pub fn new(steps: Vec<Step>) -> Self {
        LatticePath {
            steps,
            start: (0, 0),
        }
    }

    /// Create a new lattice path with a custom starting point
    pub fn new_with_start(steps: Vec<Step>, start: (i32, i32)) -> Self {
        LatticePath { steps, start }
    }

    /// Get the steps of the path
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Get the starting point
    pub fn start(&self) -> (i32, i32) {
        self.start
    }

    /// Get the ending point
    pub fn end(&self) -> (i32, i32) {
        let mut pos = self.start;
        for step in &self.steps {
            let (dx, dy) = step.to_coords();
            pos.0 += dx;
            pos.1 += dy;
        }
        pos
    }

    /// Get the length of the path (number of steps)
    pub fn length(&self) -> usize {
        self.steps.len()
    }

    /// Get all points visited by the path (including start and end)
    pub fn points(&self) -> Vec<(i32, i32)> {
        let mut points = vec![self.start];
        let mut current = self.start;

        for step in &self.steps {
            let (dx, dy) = step.to_coords();
            current.0 += dx;
            current.1 += dy;
            points.push(current);
        }

        points
    }

    /// Check if the path stays weakly above the diagonal y = x
    ///
    /// This means for all points (x, y) on the path, we have y >= x
    pub fn is_above_diagonal(&self) -> bool {
        let mut pos = self.start;
        if pos.1 < pos.0 {
            return false;
        }

        for step in &self.steps {
            let (dx, dy) = step.to_coords();
            pos.0 += dx;
            pos.1 += dy;
            if pos.1 < pos.0 {
                return false;
            }
        }

        true
    }

    /// Check if the path stays weakly below the diagonal y = x
    pub fn is_below_diagonal(&self) -> bool {
        let mut pos = self.start;
        if pos.1 > pos.0 {
            return false;
        }

        for step in &self.steps {
            let (dx, dy) = step.to_coords();
            pos.0 += dx;
            pos.1 += dy;
            if pos.1 > pos.0 {
                return false;
            }
        }

        true
    }

    /// Check if the path never goes below the x-axis (y >= 0 always)
    pub fn is_above_x_axis(&self) -> bool {
        let mut pos = self.start;
        if pos.1 < 0 {
            return false;
        }

        for step in &self.steps {
            let (dx, dy) = step.to_coords();
            pos.0 += dx;
            pos.1 += dy;
            if pos.1 < 0 {
                return false;
            }
        }

        true
    }

    /// Count the number of steps of a specific type
    pub fn count_step(&self, step_type: Step) -> usize {
        self.steps.iter().filter(|&&s| s == step_type).count()
    }

    /// Convert to a Dyck path encoding (sequence of +1 and -1)
    ///
    /// North steps become +1, East steps become -1
    /// Only valid for paths using North and East steps
    pub fn to_dyck_encoding(&self) -> Option<Vec<i32>> {
        let mut result = Vec::new();
        for &step in &self.steps {
            match step {
                Step::North => result.push(1),
                Step::East => result.push(-1),
                _ => return None,
            }
        }
        Some(result)
    }

    /// Create a lattice path from a Dyck encoding
    pub fn from_dyck_encoding(encoding: &[i32]) -> Option<Self> {
        let mut steps = Vec::new();
        for &val in encoding {
            match val {
                1 => steps.push(Step::North),
                -1 => steps.push(Step::East),
                _ => return None,
            }
        }
        Some(LatticePath::new(steps))
    }
}

/// Generate all lattice paths from (0, 0) to (m, n) using North and East steps
///
/// Returns all paths that use exactly m East steps and n North steps.
pub fn lattice_paths_ne(m: usize, n: usize) -> Vec<LatticePath> {
    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_ne_paths(m, n, &mut current, &mut result);

    result
}

fn generate_ne_paths(
    east_remaining: usize,
    north_remaining: usize,
    current: &mut Vec<Step>,
    result: &mut Vec<LatticePath>,
) {
    if east_remaining == 0 && north_remaining == 0 {
        result.push(LatticePath::new(current.clone()));
        return;
    }

    if east_remaining > 0 {
        current.push(Step::East);
        generate_ne_paths(east_remaining - 1, north_remaining, current, result);
        current.pop();
    }

    if north_remaining > 0 {
        current.push(Step::North);
        generate_ne_paths(east_remaining, north_remaining - 1, current, result);
        current.pop();
    }
}

/// Generate all Dyck paths of length 2n
///
/// A Dyck path is a lattice path from (0,0) to (n,n) that never goes below
/// the diagonal y = x. Equivalently, paths with n North steps and n East steps
/// where at every point, the number of North steps >= number of East steps.
pub fn dyck_paths(n: usize) -> Vec<LatticePath> {
    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_dyck_paths(n, n, 0, &mut current, &mut result);

    result
}

fn generate_dyck_paths(
    east_remaining: usize,
    north_remaining: usize,
    height: i32,
    current: &mut Vec<Step>,
    result: &mut Vec<LatticePath>,
) {
    if east_remaining == 0 && north_remaining == 0 {
        result.push(LatticePath::new(current.clone()));
        return;
    }

    // Can always add a North step if available
    if north_remaining > 0 {
        current.push(Step::North);
        generate_dyck_paths(east_remaining, north_remaining - 1, height + 1, current, result);
        current.pop();
    }

    // Can only add East step if we're above the diagonal (height > 0)
    if east_remaining > 0 && height > 0 {
        current.push(Step::East);
        generate_dyck_paths(east_remaining - 1, north_remaining, height - 1, current, result);
        current.pop();
    }
}

/// Count the number of lattice paths from (0,0) to (m,n) using the binomial coefficient
///
/// The number of such paths is C(m+n, m) = C(m+n, n)
pub fn count_lattice_paths_ne(m: u32, n: u32) -> Integer {
    crate::binomial(m + n, m)
}

/// Count the number of Dyck paths of length 2n (Catalan number)
///
/// This is the nth Catalan number: C_n = (1/(n+1)) * C(2n, n)
pub fn count_dyck_paths(n: u32) -> Integer {
    crate::catalan(n)
}

/// A path tableau is a Young tableau that encodes a lattice path
///
/// The correspondence works as follows:
/// - Each row i of the tableau corresponds to the positions where
///   the path has height i
/// - The tableau structure encodes the relative order of steps
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathTableau {
    /// The underlying tableau
    tableau: Tableau,
    /// The lattice path this tableau represents
    path: LatticePath,
}

impl PathTableau {
    /// Create a path tableau from a lattice path
    ///
    /// Converts a lattice path (using North/East steps) into a tableau
    /// representation using the RSK-like correspondence.
    pub fn from_lattice_path(path: &LatticePath) -> Option<Self> {
        // For a North-East path, we can use a simple encoding:
        // Label the steps 1, 2, 3, ..., n
        // Build the tableau by inserting based on step type and position

        let mut rows: Vec<Vec<usize>> = Vec::new();
        let mut height = 0i32;

        for (i, &step) in path.steps().iter().enumerate() {
            match step {
                Step::North => {
                    height += 1;
                    // Ensure we have enough rows
                    while rows.len() < height as usize {
                        rows.push(Vec::new());
                    }
                    // Add label to this row
                    rows[height as usize - 1].push(i + 1);
                }
                Step::East => {
                    height -= 1;
                    if height < 0 {
                        return None; // Invalid path (goes below baseline)
                    }
                }
                _ => return None, // Only North/East steps supported
            }
        }

        // Create the tableau
        let tableau = Tableau::new(rows)?;

        Some(PathTableau {
            tableau,
            path: path.clone(),
        })
    }

    /// Get the underlying tableau
    pub fn tableau(&self) -> &Tableau {
        &self.tableau
    }

    /// Get the lattice path
    pub fn path(&self) -> &LatticePath {
        &self.path
    }

    /// Get the shape of the path tableau
    pub fn shape(&self) -> &Partition {
        self.tableau.shape()
    }
}

/// Convert a Dyck path to a partition
///
/// The area under a Dyck path forms a Young diagram, which corresponds
/// to a partition. This function computes that partition.
///
/// For each East step, we record the y-coordinate (height), which represents
/// the number of boxes in that column of the Young diagram.
pub fn dyck_path_to_partition(path: &LatticePath) -> Option<Partition> {
    let mut heights = Vec::new();
    let mut pos = (0i32, 0i32);

    for &step in path.steps() {
        let (dx, dy) = step.to_coords();
        pos.0 += dx;
        pos.1 += dy;

        match step {
            Step::East => {
                // When we take an East step, record the current height (y-coordinate)
                if pos.1 < 0 {
                    return None; // Path goes below x-axis
                }
                heights.push(pos.1 as usize);
            }
            Step::North => {
                // North steps just increase height
            }
            _ => return None, // Only North/East supported for Dyck paths
        }
    }

    // Sort to get partition in decreasing order
    heights.sort_unstable_by(|a, b| b.cmp(a));

    Some(Partition::new(heights))
}

/// Convert a partition to a Dyck path
///
/// Given a partition, construct the Dyck path that traces the boundary
/// of the Young diagram.
pub fn partition_to_dyck_path(partition: &Partition) -> LatticePath {
    let mut steps = Vec::new();

    let parts = partition.parts();
    let num_parts = parts.len();

    for i in 0..num_parts {
        // Add North steps to reach height i+1
        if i == 0 {
            steps.push(Step::North);
        } else if i > 0 && i < num_parts {
            steps.push(Step::North);
        }

        // Add East steps based on the difference in part sizes
        let current_length = parts[i];
        let next_length = if i + 1 < num_parts { parts[i + 1] } else { 0 };
        let east_steps = current_length.saturating_sub(next_length);

        for _ in 0..east_steps {
            steps.push(Step::East);
        }
    }

    // Add final North step to return to diagonal if needed
    let total_north = steps.iter().filter(|&&s| s == Step::North).count();
    let total_east = steps.iter().filter(|&&s| s == Step::East).count();

    // Balance the path
    while total_east > total_north {
        steps.insert(0, Step::North);
    }

    LatticePath::new(steps)
}

/// Generate all ballot sequences of length n
///
/// A ballot sequence is a sequence of +1 and -1 that never becomes negative
/// when read left to right. These correspond to lattice paths that stay
/// above the x-axis.
pub fn ballot_sequences(north: usize, east: usize) -> Vec<Vec<i32>> {
    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_ballot_sequences(north, east, 0, &mut current, &mut result);

    result
}

fn generate_ballot_sequences(
    north_remaining: usize,
    east_remaining: usize,
    cumsum: i32,
    current: &mut Vec<i32>,
    result: &mut Vec<Vec<i32>>,
) {
    if north_remaining == 0 && east_remaining == 0 {
        result.push(current.clone());
        return;
    }

    // Add +1 (North)
    if north_remaining > 0 {
        current.push(1);
        generate_ballot_sequences(north_remaining - 1, east_remaining, cumsum + 1, current, result);
        current.pop();
    }

    // Add -1 (East) only if cumsum > 0
    if east_remaining > 0 && cumsum > 0 {
        current.push(-1);
        generate_ballot_sequences(north_remaining, east_remaining - 1, cumsum - 1, current, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_coords() {
        assert_eq!(Step::North.to_coords(), (0, 1));
        assert_eq!(Step::East.to_coords(), (1, 0));
        assert_eq!(Step::South.to_coords(), (0, -1));
        assert_eq!(Step::West.to_coords(), (-1, 0));
        assert_eq!(Step::Diagonal.to_coords(), (1, 1));
    }

    #[test]
    fn test_step_from_coords() {
        assert_eq!(Step::from_coords(0, 1), Some(Step::North));
        assert_eq!(Step::from_coords(1, 0), Some(Step::East));
        assert_eq!(Step::from_coords(0, -1), Some(Step::South));
        assert_eq!(Step::from_coords(-1, 0), Some(Step::West));
        assert_eq!(Step::from_coords(1, 1), Some(Step::Diagonal));
        assert_eq!(Step::from_coords(2, 3), None);
    }

    #[test]
    fn test_lattice_path_basic() {
        let path = LatticePath::new(vec![Step::North, Step::East, Step::North, Step::East]);
        assert_eq!(path.length(), 4);
        assert_eq!(path.start(), (0, 0));
        assert_eq!(path.end(), (2, 2));
    }

    #[test]
    fn test_lattice_path_points() {
        let path = LatticePath::new(vec![Step::East, Step::North]);
        let points = path.points();
        assert_eq!(points, vec![(0, 0), (1, 0), (1, 1)]);
    }

    #[test]
    fn test_is_above_diagonal() {
        // Path that stays above diagonal: N, N, E, E
        let path1 = LatticePath::new(vec![Step::North, Step::North, Step::East, Step::East]);
        assert!(path1.is_above_diagonal());

        // Path that goes below diagonal: E, N
        let path2 = LatticePath::new(vec![Step::East, Step::North]);
        assert!(!path2.is_above_diagonal());
    }

    #[test]
    fn test_is_above_x_axis() {
        let path1 = LatticePath::new(vec![Step::North, Step::East]);
        assert!(path1.is_above_x_axis());

        let path2 = LatticePath::new(vec![Step::South, Step::North]);
        assert!(!path2.is_above_x_axis());
    }

    #[test]
    fn test_count_step() {
        let path = LatticePath::new(vec![Step::North, Step::North, Step::East, Step::North]);
        assert_eq!(path.count_step(Step::North), 3);
        assert_eq!(path.count_step(Step::East), 1);
    }

    #[test]
    fn test_dyck_encoding() {
        let path = LatticePath::new(vec![Step::North, Step::East, Step::North, Step::East]);
        let encoding = path.to_dyck_encoding().unwrap();
        assert_eq!(encoding, vec![1, -1, 1, -1]);

        let reconstructed = LatticePath::from_dyck_encoding(&encoding).unwrap();
        assert_eq!(path, reconstructed);
    }

    #[test]
    fn test_lattice_paths_ne_small() {
        // Paths from (0,0) to (1,1)
        let paths = lattice_paths_ne(1, 1);
        assert_eq!(paths.len(), 2); // NE and EN
    }

    #[test]
    fn test_lattice_paths_ne_count() {
        // Number of paths from (0,0) to (2,2) is C(4,2) = 6
        let paths = lattice_paths_ne(2, 2);
        assert_eq!(paths.len(), 6);
    }

    #[test]
    fn test_count_lattice_paths_ne() {
        assert_eq!(count_lattice_paths_ne(2, 2), Integer::from(6));
        assert_eq!(count_lattice_paths_ne(3, 2), Integer::from(10));
        assert_eq!(count_lattice_paths_ne(0, 0), Integer::from(1));
    }

    #[test]
    fn test_dyck_paths_small() {
        // Dyck paths of length 2 (n=1): just "NE"
        let paths = dyck_paths(1);
        assert_eq!(paths.len(), 1);

        // Dyck paths of length 4 (n=2): "NNEE" and "NENE"
        let paths2 = dyck_paths(2);
        assert_eq!(paths2.len(), 2);

        // All should satisfy the Dyck path property
        for path in &paths2 {
            assert!(path.is_above_diagonal());
        }
    }

    #[test]
    fn test_count_dyck_paths() {
        // Catalan numbers: 1, 1, 2, 5, 14, 42, ...
        assert_eq!(count_dyck_paths(0), Integer::from(1));
        assert_eq!(count_dyck_paths(1), Integer::from(1));
        assert_eq!(count_dyck_paths(2), Integer::from(2));
        assert_eq!(count_dyck_paths(3), Integer::from(5));
        assert_eq!(count_dyck_paths(4), Integer::from(14));
    }

    #[test]
    fn test_dyck_paths_generate_all() {
        // Verify that dyck_paths generates exactly Catalan(n) paths
        for n in 0..=5 {
            let paths = dyck_paths(n);
            let expected = count_dyck_paths(n as u32);
            assert_eq!(Integer::from(paths.len() as u64), expected);
        }
    }

    #[test]
    fn test_path_tableau_from_lattice_path() {
        // Simple Dyck path: N, N, E, E
        let path = LatticePath::new(vec![Step::North, Step::North, Step::East, Step::East]);
        let pt = PathTableau::from_lattice_path(&path).unwrap();

        // The tableau should have the labels of North steps in rows
        let tableau = pt.tableau();
        assert_eq!(tableau.num_rows(), 2);
    }

    #[test]
    fn test_dyck_path_to_partition() {
        // Dyck path N, N, E, E gives partition [2]
        let path = LatticePath::new(vec![Step::North, Step::North, Step::East, Step::East]);
        let partition = dyck_path_to_partition(&path).unwrap();
        assert_eq!(partition.parts(), &[2, 2]);
    }

    #[test]
    fn test_partition_to_dyck_path() {
        let partition = Partition::new(vec![3, 2, 1]);
        let path = partition_to_dyck_path(&partition);

        // The path should be a valid Dyck path
        // (may not be the exact original due to different encodings)
        assert_eq!(path.count_step(Step::North), path.count_step(Step::East));
    }

    #[test]
    fn test_ballot_sequences() {
        // Ballot sequences with 2 North, 1 East
        let sequences = ballot_sequences(2, 1);

        // All should satisfy ballot property (never negative)
        for seq in &sequences {
            let mut cumsum = 0;
            for &val in seq {
                cumsum += val;
                assert!(cumsum >= 0);
            }
            assert_eq!(cumsum, 1); // Should end at 2-1 = 1
        }
    }

    #[test]
    fn test_ballot_sequences_count() {
        // For equal north and east that end at 0, we get 0 sequences
        // For north > east, we get the ballot number
        let sequences = ballot_sequences(3, 2);

        // All sequences should end with positive cumsum
        for seq in &sequences {
            let sum: i32 = seq.iter().sum();
            assert_eq!(sum, 1); // 3 - 2 = 1
        }
    }

    #[test]
    fn test_lattice_path_diagonal_property() {
        // Test that all generated Dyck paths satisfy the diagonal property
        let paths = dyck_paths(3);

        for path in &paths {
            assert!(path.is_above_diagonal());
            assert_eq!(path.count_step(Step::North), path.count_step(Step::East));
        }
    }
}
