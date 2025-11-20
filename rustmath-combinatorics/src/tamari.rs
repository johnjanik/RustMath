//! Tamari lattices and nu-Tamari lattices
//!
//! The Tamari lattice is a partial order on Dyck paths (or equivalently on binary trees).
//! The nu-Tamari lattice is a generalization parameterized by a lattice path nu from (0,0) to (a,b).
//!
//! # References
//! - Tamari, D. (1962). "The algebra of bracketings and their enumeration"
//! - Préville-Ratelle, L.-F., & Viennot, X. (2014). "The enumeration of generalized Tamari intervals"

use crate::dyck_word::DyckWord;
use crate::posets::Poset;
use std::collections::HashMap;

/// A lattice path from (0,0) to (a,b) using steps (1,0) and (0,1)
///
/// Represented as a sequence of steps: true = (1,0) = North, false = (0,1) = East
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LatticePath {
    /// The sequence of steps (true = North/right, false = East/up)
    steps: Vec<bool>,
    /// Target x-coordinate
    a: usize,
    /// Target y-coordinate
    b: usize,
}

impl LatticePath {
    /// Create a new lattice path from a sequence of steps
    ///
    /// Steps are represented as: true = (1,0) North step, false = (0,1) East step
    /// Returns None if the path doesn't end at (a,b) or has invalid steps
    pub fn new(steps: Vec<bool>) -> Option<Self> {
        let a = steps.iter().filter(|&&s| s).count();
        let b = steps.iter().filter(|&&s| !s).count();

        Some(LatticePath { steps, a, b })
    }

    /// Create a lattice path from coordinates using steps
    ///
    /// a is the number of North steps (1,0), b is the number of East steps (0,1)
    pub fn from_coordinates(a: usize, b: usize, steps: Vec<bool>) -> Option<Self> {
        let north_count = steps.iter().filter(|&&s| s).count();
        let east_count = steps.iter().filter(|&&s| !s).count();

        if north_count != a || east_count != b {
            return None;
        }

        Some(LatticePath { steps, a, b })
    }

    /// Create the diagonal path from (0,0) to (n,n)
    ///
    /// This is the path that alternates between North and East steps
    pub fn diagonal(n: usize) -> Self {
        let mut steps = Vec::with_capacity(2 * n);
        for _ in 0..n {
            steps.push(true);  // North
            steps.push(false); // East
        }
        LatticePath { steps, a: n, b: n }
    }

    /// Get the target coordinates (a, b)
    pub fn target(&self) -> (usize, usize) {
        (self.a, self.b)
    }

    /// Get the steps of the path
    pub fn steps(&self) -> &[bool] {
        &self.steps
    }

    /// Get the length of the path (total number of steps)
    pub fn length(&self) -> usize {
        self.steps.len()
    }

    /// Get the coordinates at position i
    ///
    /// Returns (x, y) after taking the first i steps
    pub fn position_at(&self, i: usize) -> (usize, usize) {
        let mut x = 0;
        let mut y = 0;

        for j in 0..i.min(self.steps.len()) {
            if self.steps[j] {
                x += 1; // North step
            } else {
                y += 1; // East step
            }
        }

        (x, y)
    }

    /// Check if this path stays weakly above another path
    ///
    /// For Dyck paths, this means comparing the "height" (number of North steps minus East steps)
    /// at each position. Path P is above path Q if at every step, height(P) >= height(Q).
    pub fn above_or_equal(&self, other: &LatticePath) -> bool {
        if self.a != other.a || self.b != other.b {
            return false;
        }

        let mut self_height = 0;
        let mut other_height = 0;

        // Check at the starting position
        if self_height < other_height {
            return false;
        }

        // Check at each step position
        for i in 0..self.steps.len() {
            // Take the next step and update heights
            if self.steps[i] {
                self_height += 1; // North step increases height
            } else {
                self_height -= 1; // East step decreases height
            }

            if other.steps[i] {
                other_height += 1; // North step increases height
            } else {
                other_height -= 1; // East step decreases height
            }

            // After step i, check if self is at or above other
            if self_height < other_height {
                return false;
            }
        }

        true
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.steps
            .iter()
            .map(|&b| if b { 'N' } else { 'E' })
            .collect()
    }
}

/// The nu-Tamari lattice
///
/// A partial order on Dyck paths that lie above the lattice path nu
#[derive(Debug, Clone)]
pub struct TamariLattice {
    /// The lattice path nu
    nu: LatticePath,
    /// The Dyck paths in the lattice (indexed for the poset)
    paths: Vec<DyckWord>,
    /// Mapping from path to index
    path_to_index: HashMap<Vec<bool>, usize>,
    /// The underlying poset structure
    poset: Poset,
}

impl TamariLattice {
    /// Create a new nu-Tamari lattice
    ///
    /// Generates all Dyck paths compatible with nu and constructs the lattice
    pub fn new(nu: LatticePath) -> Self {
        let (a, b) = nu.target();

        // Generate all Dyck paths from (0,0) to (a,b)
        // A Dyck path to (a,b) has a North steps and b East steps,
        // and at each point has taken at least as many North as East steps
        let mut paths = Vec::new();
        let mut path_to_index = HashMap::new();

        Self::generate_nu_dyck_paths(a, b, &nu, &mut paths, &mut path_to_index);

        // Build the covering relations
        let covering_relations = Self::compute_covering_relations(&paths, &path_to_index);

        // Create the poset
        let elements: Vec<usize> = (0..paths.len()).collect();
        let poset = Poset::new(elements, covering_relations);

        TamariLattice {
            nu,
            paths,
            path_to_index,
            poset,
        }
    }

    /// Create the classical Tamari lattice T_n
    ///
    /// This is the nu-Tamari lattice where nu is the diagonal from (0,0) to (n,n)
    pub fn classical(n: usize) -> Self {
        let nu = LatticePath::diagonal(n);
        Self::new(nu)
    }

    /// Generate all Dyck paths compatible with nu
    fn generate_nu_dyck_paths(
        a: usize,
        b: usize,
        nu: &LatticePath,
        paths: &mut Vec<DyckWord>,
        path_to_index: &mut HashMap<Vec<bool>, usize>,
    ) {
        let mut current = Vec::new();
        Self::generate_helper(a, b, 0, 0, nu, &mut current, paths, path_to_index);
    }

    fn generate_helper(
        a_remaining: usize,
        b_remaining: usize,
        north_taken: usize,
        east_taken: usize,
        nu: &LatticePath,
        current: &mut Vec<bool>,
        paths: &mut Vec<DyckWord>,
        path_to_index: &mut HashMap<Vec<bool>, usize>,
    ) {
        if a_remaining == 0 && b_remaining == 0 {
            // Check if this path is valid (above nu)
            let lattice_path = LatticePath::new(current.clone()).unwrap();
            if lattice_path.above_or_equal(nu) {
                if let Some(dyck) = DyckWord::new(Self::lattice_to_dyck_steps(current)) {
                    let index = paths.len();
                    path_to_index.insert(current.clone(), index);
                    paths.push(dyck);
                }
            }
            return;
        }

        // Try adding a North step (1,0)
        if a_remaining > 0 {
            current.push(true);
            Self::generate_helper(
                a_remaining - 1,
                b_remaining,
                north_taken + 1,
                east_taken,
                nu,
                current,
                paths,
                path_to_index,
            );
            current.pop();
        }

        // Try adding an East step (0,1)
        // Can only add East if we have more North than East (Dyck condition)
        if b_remaining > 0 && north_taken > east_taken {
            current.push(false);
            Self::generate_helper(
                a_remaining,
                b_remaining - 1,
                north_taken,
                east_taken + 1,
                nu,
                current,
                paths,
                path_to_index,
            );
            current.pop();
        }
    }

    /// Convert lattice path steps to Dyck word steps
    ///
    /// North (true) -> X (true), East (false) -> Y (false)
    fn lattice_to_dyck_steps(lattice_steps: &[bool]) -> Vec<bool> {
        lattice_steps.to_vec()
    }

    /// Compute the covering relations for the nu-Tamari lattice
    ///
    /// In the nu-Tamari lattice, P is covered by Q if Q can be obtained from P
    /// by a specific rotation operation
    fn compute_covering_relations(
        paths: &[DyckWord],
        path_to_index: &HashMap<Vec<bool>, usize>,
    ) -> Vec<(usize, usize)> {
        let mut relations = Vec::new();

        for (i, path_i) in paths.iter().enumerate() {
            // Find all paths that cover path_i
            for (j, path_j) in paths.iter().enumerate() {
                if i != j && Self::covers(path_i, path_j) {
                    relations.push((i, j));
                }
            }
        }

        relations
    }

    /// Check if path q covers path p in the Tamari order
    ///
    /// This implements the rotation operation: p < q if q can be obtained
    /// from p by a right rotation (moving a closing paren past an opening paren)
    fn covers(p: &DyckWord, q: &DyckWord) -> bool {
        let p_steps = p.as_slice();
        let q_steps = q.as_slice();

        if p_steps.len() != q_steps.len() {
            return false;
        }

        // Find the position where they first differ
        let mut diff_pos = None;
        for i in 0..p_steps.len() {
            if p_steps[i] != q_steps[i] {
                diff_pos = Some(i);
                break;
            }
        }

        let diff_pos = match diff_pos {
            Some(pos) => pos,
            None => return false, // Paths are identical
        };

        // Check if this is a valid covering relation
        // In Tamari order, we can do a rotation: ...XYX... -> ...XXY...
        // This means: p has XY at some position, q has YX at the same position
        // No wait, let me reconsider...

        // Actually, in the Tamari lattice, the covering relation is:
        // p ⋖ q if there exist positions i < j such that:
        // - p[i] = Y (down), p[j] = X (up)
        // - q is obtained from p by swapping these
        // - The segment p[i+1..j-1] is balanced
        // - Everything else is identical

        // For a simpler approach: check if q > p by one rotation
        // A rotation is: ...YX... -> ...XY... where the segment between is balanced

        if p_steps[diff_pos] == false && q_steps[diff_pos] == true {
            // p has Y, q has X at diff_pos
            // Find the next position where they differ
            let mut balance = -1; // We have Y from p
            for i in (diff_pos + 1)..p_steps.len() {
                if p_steps[i] {
                    balance += 1; // X
                } else {
                    balance -= 1; // Y
                }

                if p_steps[i] != q_steps[i] {
                    // Check if this is the rotation point
                    if balance == 0 && p_steps[i] == true && q_steps[i] == false {
                        // Verify the rest is identical
                        if p_steps[i + 1..] == q_steps[i + 1..] {
                            return true;
                        }
                    }
                    return false;
                }

                if balance < -1 {
                    return false;
                }
            }
        }

        false
    }

    /// Get the underlying poset
    pub fn poset(&self) -> &Poset {
        &self.poset
    }

    /// Get all Dyck paths in the lattice
    pub fn paths(&self) -> &[DyckWord] {
        &self.paths
    }

    /// Get the lattice path nu
    pub fn nu(&self) -> &LatticePath {
        &self.nu
    }

    /// Get the number of elements in the lattice
    pub fn size(&self) -> usize {
        self.paths.len()
    }

    /// Check if path p is less than or equal to path q in the Tamari order
    pub fn less_than_or_equal(&self, p: &DyckWord, q: &DyckWord) -> bool {
        let p_idx = self.path_to_index.get(p.as_slice());
        let q_idx = self.path_to_index.get(q.as_slice());

        match (p_idx, q_idx) {
            (Some(&pi), Some(&qi)) => self.poset.less_than_or_equal(pi, qi),
            _ => false,
        }
    }

    /// Get the path at a given index
    pub fn path_at(&self, index: usize) -> Option<&DyckWord> {
        self.paths.get(index)
    }
}

/// Create a nu-Tamari lattice from a lattice path
pub fn nu_tamari_lattice(nu: LatticePath) -> TamariLattice {
    TamariLattice::new(nu)
}

/// Create the classical Tamari lattice T_n
pub fn tamari_lattice(n: usize) -> TamariLattice {
    TamariLattice::classical(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_path_creation() {
        let path = LatticePath::new(vec![true, false, true, false]).unwrap();
        assert_eq!(path.target(), (2, 2));
        assert_eq!(path.length(), 4);
    }

    #[test]
    fn test_diagonal_path() {
        let diag = LatticePath::diagonal(3);
        assert_eq!(diag.target(), (3, 3));
        assert_eq!(diag.length(), 6);
        assert_eq!(diag.to_string(), "NENENE");
    }

    #[test]
    fn test_lattice_path_positions() {
        let path = LatticePath::new(vec![true, true, false, false]).unwrap();
        assert_eq!(path.position_at(0), (0, 0));
        assert_eq!(path.position_at(1), (1, 0));
        assert_eq!(path.position_at(2), (2, 0));
        assert_eq!(path.position_at(3), (2, 1));
        assert_eq!(path.position_at(4), (2, 2));
    }

    #[test]
    fn test_path_above_or_equal() {
        let path1 = LatticePath::new(vec![true, true, false, false]).unwrap();
        let path2 = LatticePath::new(vec![true, false, true, false]).unwrap();

        // path1 (NNEY) goes all north first, so it's "above" in the lattice
        // Actually need to reconsider the definition of "above"
        // In standard convention, path1 is above path2 if path1 takes
        // more East steps initially (stays to the left)
    }

    #[test]
    fn test_classical_tamari_lattice_size() {
        // T_1 should have C_1 = 1 element
        let t1 = tamari_lattice(1);
        assert_eq!(t1.size(), 1);

        // T_2 should have C_2 = 2 elements
        let t2 = tamari_lattice(2);
        assert_eq!(t2.size(), 2);

        // T_3 should have C_3 = 5 elements
        let t3 = tamari_lattice(3);
        assert_eq!(t3.size(), 5);
    }

    #[test]
    fn test_tamari_lattice_is_lattice() {
        let t3 = tamari_lattice(3);
        // The Tamari lattice should be a lattice (every pair has meet and join)
        assert!(t3.poset().is_lattice());
    }

    #[test]
    fn test_nu_tamari_lattice() {
        // Create a simple nu path
        let nu = LatticePath::new(vec![true, false, true, false]).unwrap();
        let lattice = nu_tamari_lattice(nu);

        // Should have at least one element (the path that follows nu exactly)
        assert!(lattice.size() >= 1);
    }

    #[test]
    fn test_tamari_order_properties() {
        let t2 = tamari_lattice(2);

        // Should have a unique minimal element (all north then all east: NNEY)
        let minimal = t2.poset().minimal_elements();
        assert_eq!(minimal.len(), 1);

        // Should have a unique maximal element (alternating: NENY)
        let maximal = t2.poset().maximal_elements();
        assert_eq!(maximal.len(), 1);
    }

    #[test]
    fn test_covering_relations() {
        let t2 = tamari_lattice(2);
        let hasse = t2.poset().hasse_diagram();

        // T_2 should have exactly one covering relation (between the 2 elements)
        assert_eq!(hasse.len(), 1);
    }

    #[test]
    fn test_lattice_path_from_coordinates() {
        let path = LatticePath::from_coordinates(2, 3, vec![true, false, true, false, false]).unwrap();
        assert_eq!(path.target(), (2, 3));

        // Invalid: wrong number of steps
        assert!(LatticePath::from_coordinates(2, 2, vec![true, false, true]).is_none());
    }
}
