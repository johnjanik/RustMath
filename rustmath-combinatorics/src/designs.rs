//! Combinatorial Designs
//!
//! This module implements various combinatorial design structures:
//! - Block designs (BIBD - Balanced Incomplete Block Designs)
//! - Steiner systems
//! - Orthogonal arrays
//! - Difference sets
//! - Hadamard matrices
//! - Design automorphisms

use std::collections::HashSet;

/// A block design consists of a set of points and a collection of blocks (subsets of points)
///
/// A t-(v,k,λ) design is a collection of k-element subsets (blocks) of a v-element set
/// such that every t-element subset is contained in exactly λ blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockDesign {
    /// Number of points
    v: usize,
    /// Block size
    k: usize,
    /// t-parameter (number of points in common)
    t: usize,
    /// lambda parameter (number of blocks containing each t-subset)
    lambda: usize,
    /// The blocks as sets of point indices
    blocks: Vec<Vec<usize>>,
}

impl BlockDesign {
    /// Create a new block design
    pub fn new(v: usize, k: usize, t: usize, lambda: usize, blocks: Vec<Vec<usize>>) -> Option<Self> {
        // Validate blocks
        for block in &blocks {
            if block.len() != k {
                return None;
            }
            for &point in block {
                if point >= v {
                    return None;
                }
            }
            // Check for duplicates in block
            let mut sorted_block = block.clone();
            sorted_block.sort_unstable();
            sorted_block.dedup();
            if sorted_block.len() != block.len() {
                return None;
            }
        }

        let design = BlockDesign { v, k, t, lambda, blocks };

        // Verify the design property if it's feasible to check
        if t <= 2 && v <= 10 {
            if !design.verify_design_property() {
                return None;
            }
        }

        Some(design)
    }

    /// Create a trivial design (all points in one block)
    pub fn trivial(v: usize) -> Self {
        BlockDesign {
            v,
            k: v,
            t: v,
            lambda: 1,
            blocks: vec![(0..v).collect()],
        }
    }

    /// Verify the design property: each t-subset appears in exactly lambda blocks
    fn verify_design_property(&self) -> bool {
        if self.t == 0 || self.t > self.k || self.t > self.v {
            return false;
        }

        // Generate all t-subsets of the point set
        let points: Vec<usize> = (0..self.v).collect();
        let t_subsets = Self::k_subsets(&points, self.t);

        // Count how many blocks contain each t-subset
        for t_subset in t_subsets {
            let mut count = 0;
            for block in &self.blocks {
                if Self::contains_subset(block, &t_subset) {
                    count += 1;
                }
            }
            if count != self.lambda {
                return false;
            }
        }

        true
    }

    /// Check if a block contains a subset
    fn contains_subset(block: &[usize], subset: &[usize]) -> bool {
        subset.iter().all(|&x| block.contains(&x))
    }

    /// Generate all k-subsets of a set
    fn k_subsets(set: &[usize], k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        if k > set.len() {
            return vec![];
        }

        let mut result = Vec::new();
        Self::k_subsets_helper(set, k, 0, &mut vec![], &mut result);
        result
    }

    fn k_subsets_helper(
        set: &[usize],
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }

        for i in start..set.len() {
            current.push(set[i]);
            Self::k_subsets_helper(set, k, i + 1, current, result);
            current.pop();
        }
    }

    /// Get the number of points
    pub fn v(&self) -> usize {
        self.v
    }

    /// Get the block size
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get the t parameter
    pub fn t(&self) -> usize {
        self.t
    }

    /// Get the lambda parameter
    pub fn lambda(&self) -> usize {
        self.lambda
    }

    /// Get the blocks
    pub fn blocks(&self) -> &[Vec<usize>] {
        &self.blocks
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Check if this is a BIBD (Balanced Incomplete Block Design)
    /// A BIBD is a 2-(v,k,λ) design
    pub fn is_bibd(&self) -> bool {
        self.t == 2
    }

    /// Get the replication number (number of blocks containing each point)
    /// For a 2-design, this is r = λ(v-1)/(k-1)
    pub fn replication_number(&self) -> Option<usize> {
        if self.t != 2 {
            return None;
        }

        // Calculate r = λ(v-1)/(k-1)
        if self.k == 1 {
            return None;
        }

        let numerator = self.lambda * (self.v - 1);
        let denominator = self.k - 1;

        if numerator % denominator != 0 {
            return None;
        }

        Some(numerator / denominator)
    }
}

/// A Steiner system S(t,k,v) is a t-(v,k,1) design (λ = 1)
///
/// Special cases:
/// - S(2,3,v): Steiner triple system
/// - S(3,4,v): Steiner quadruple system
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SteinerSystem {
    design: BlockDesign,
}

impl SteinerSystem {
    /// Create a Steiner system S(t,k,v)
    pub fn new(t: usize, k: usize, v: usize, blocks: Vec<Vec<usize>>) -> Option<Self> {
        let design = BlockDesign::new(v, k, t, 1, blocks)?;
        Some(SteinerSystem { design })
    }

    /// Create a Steiner triple system S(2,3,v) if one exists
    /// These exist if and only if v ≡ 1 or 3 (mod 6)
    pub fn steiner_triple_system(v: usize) -> Option<Self> {
        if v % 6 != 1 && v % 6 != 3 {
            return None; // Necessary condition
        }

        // For small values, construct explicitly
        match v {
            3 => {
                // Trivial case: one block {0,1,2}
                Self::new(2, 3, 3, vec![vec![0, 1, 2]])
            }
            7 => {
                // Fano plane: 7 points, 7 blocks
                Self::new(
                    2,
                    3,
                    7,
                    vec![
                        vec![0, 1, 2],
                        vec![0, 3, 4],
                        vec![0, 5, 6],
                        vec![1, 3, 5],
                        vec![1, 4, 6],
                        vec![2, 3, 6],
                        vec![2, 4, 5],
                    ],
                )
            }
            9 => {
                // Affine plane AG(2,3): 9 points, 12 blocks
                Self::new(
                    2,
                    3,
                    9,
                    vec![
                        vec![0, 1, 2],
                        vec![3, 4, 5],
                        vec![6, 7, 8],
                        vec![0, 3, 6],
                        vec![1, 4, 7],
                        vec![2, 5, 8],
                        vec![0, 4, 8],
                        vec![1, 5, 6],
                        vec![2, 3, 7],
                        vec![0, 5, 7],
                        vec![1, 3, 8],
                        vec![2, 4, 6],
                    ],
                )
            }
            _ => None, // General construction not implemented
        }
    }

    /// Get the underlying block design
    pub fn design(&self) -> &BlockDesign {
        &self.design
    }

    /// Get the parameters (t, k, v)
    pub fn parameters(&self) -> (usize, usize, usize) {
        (self.design.t, self.design.k, self.design.v)
    }
}

/// An orthogonal array OA(t, k, v) is an array with v^t rows and k columns
/// over an alphabet of size v, such that in any t columns, every t-tuple
/// appears exactly once
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrthogonalArray {
    /// Strength (t parameter)
    t: usize,
    /// Number of columns
    k: usize,
    /// Alphabet size
    v: usize,
    /// The array (rows x columns)
    rows: Vec<Vec<usize>>,
}

impl OrthogonalArray {
    /// Create a new orthogonal array
    pub fn new(t: usize, k: usize, v: usize, rows: Vec<Vec<usize>>) -> Option<Self> {
        // Validate dimensions
        if rows.len() != v.pow(t as u32) {
            return None;
        }

        for row in &rows {
            if row.len() != k {
                return None;
            }
            for &val in row {
                if val >= v {
                    return None;
                }
            }
        }

        let array = OrthogonalArray { t, k, v, rows };

        // Verify orthogonality for small cases
        if t <= 2 && k <= 4 && v <= 3 {
            if !array.verify_orthogonality() {
                return None;
            }
        }

        Some(array)
    }

    /// Verify the orthogonality property
    fn verify_orthogonality(&self) -> bool {
        // Check all t-column combinations
        let columns: Vec<usize> = (0..self.k).collect();
        let column_combinations = BlockDesign::k_subsets(&columns, self.t);

        for col_subset in column_combinations {
            // Extract the t columns
            let mut tuples = HashSet::new();
            for row in &self.rows {
                let tuple: Vec<usize> = col_subset.iter().map(|&c| row[c]).collect();
                tuples.insert(tuple);
            }

            // Should have exactly v^t distinct tuples
            if tuples.len() != self.v.pow(self.t as u32) {
                return false;
            }
        }

        true
    }

    /// Create an OA(2, k, v) from k-2 mutually orthogonal Latin squares
    pub fn from_mols(squares: &[Vec<Vec<usize>>]) -> Option<Self> {
        if squares.is_empty() {
            return None;
        }

        let v = squares[0].len();
        let k = squares.len() + 2;

        // Verify all squares are v x v Latin squares
        for square in squares {
            if square.len() != v || square.iter().any(|row| row.len() != v) {
                return None;
            }
        }

        // Build the orthogonal array
        let mut rows = Vec::new();
        for i in 0..v {
            for j in 0..v {
                let mut row = vec![i, j];
                for square in squares {
                    row.push(square[i][j]);
                }
                rows.push(row);
            }
        }

        Self::new(2, k, v, rows)
    }

    /// Get the parameters (t, k, v)
    pub fn parameters(&self) -> (usize, usize, usize) {
        (self.t, self.k, self.v)
    }

    /// Get the rows
    pub fn rows(&self) -> &[Vec<usize>] {
        &self.rows
    }

    /// Get a specific entry
    pub fn get(&self, row: usize, col: usize) -> Option<usize> {
        self.rows.get(row)?.get(col).copied()
    }
}

/// A (v,k,λ) difference set is a k-element subset D of a group G of order v
/// such that every non-identity element of G can be expressed as a difference
/// d1 - d2 (with d1, d2 ∈ D, d1 ≠ d2) in exactly λ ways
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DifferenceSet {
    /// Group order
    v: usize,
    /// Size of difference set
    k: usize,
    /// Lambda parameter
    lambda: usize,
    /// The difference set elements (indices in Z_v)
    elements: Vec<usize>,
}

impl DifferenceSet {
    /// Create a new difference set
    pub fn new(v: usize, k: usize, lambda: usize, elements: Vec<usize>) -> Option<Self> {
        if elements.len() != k {
            return None;
        }

        for &elem in &elements {
            if elem >= v {
                return None;
            }
        }

        let diffset = DifferenceSet {
            v,
            k,
            lambda,
            elements,
        };

        // Verify for small cases
        if v <= 20 {
            if !diffset.verify_difference_property() {
                return None;
            }
        }

        Some(diffset)
    }

    /// Verify the difference property
    fn verify_difference_property(&self) -> bool {
        // Count differences (mod v)
        let mut diff_count = vec![0; self.v];

        for &d1 in &self.elements {
            for &d2 in &self.elements {
                if d1 != d2 {
                    let diff = (d1 + self.v - d2) % self.v;
                    diff_count[diff] += 1;
                }
            }
        }

        // Every non-zero element should appear exactly lambda times
        for i in 1..self.v {
            if diff_count[i] != self.lambda {
                return false;
            }
        }

        true
    }

    /// Create a (7,3,1) difference set (Fano plane)
    pub fn fano_plane() -> Self {
        DifferenceSet {
            v: 7,
            k: 3,
            lambda: 1,
            elements: vec![0, 1, 3],
        }
    }

    /// Create a (v,v-1,v-2) trivial difference set (all non-zero elements)
    pub fn trivial(v: usize) -> Self {
        DifferenceSet {
            v,
            k: v - 1,
            lambda: v - 2,
            elements: (1..v).collect(),
        }
    }

    /// Get the parameters (v, k, λ)
    pub fn parameters(&self) -> (usize, usize, usize) {
        (self.v, self.k, self.lambda)
    }

    /// Get the elements
    pub fn elements(&self) -> &[usize] {
        &self.elements
    }

    /// Develop the difference set into a block design
    /// Each block is obtained by adding i (mod v) to all elements
    pub fn develop(&self) -> BlockDesign {
        let mut blocks = Vec::new();

        for i in 0..self.v {
            let block: Vec<usize> = self
                .elements
                .iter()
                .map(|&d| (d + i) % self.v)
                .collect();
            blocks.push(block);
        }

        BlockDesign::new(self.v, self.k, 2, self.lambda, blocks).unwrap()
    }
}

/// A Hadamard matrix is an n×n matrix with entries ±1 such that HH^T = nI
/// Hadamard matrices can only exist when n = 1, 2, or n ≡ 0 (mod 4)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HadamardMatrix {
    /// Size of the matrix
    n: usize,
    /// The matrix entries (+1 or -1)
    matrix: Vec<Vec<i32>>,
}

impl HadamardMatrix {
    /// Create a new Hadamard matrix
    pub fn new(matrix: Vec<Vec<i32>>) -> Option<Self> {
        let n = matrix.len();

        if n == 0 {
            return Some(HadamardMatrix { n: 0, matrix: vec![] });
        }

        // Check dimensions
        for row in &matrix {
            if row.len() != n {
                return None;
            }
        }

        // Check entries are ±1
        for row in &matrix {
            for &val in row {
                if val != 1 && val != -1 {
                    return None;
                }
            }
        }

        let h = HadamardMatrix { n, matrix };

        // Verify Hadamard property for small matrices
        if n <= 8 {
            if !h.verify_hadamard_property() {
                return None;
            }
        }

        Some(h)
    }

    /// Verify that HH^T = nI
    fn verify_hadamard_property(&self) -> bool {
        for i in 0..self.n {
            for j in 0..self.n {
                let mut dot_product = 0;
                for k in 0..self.n {
                    dot_product += self.matrix[i][k] * self.matrix[j][k];
                }

                let expected = if i == j { self.n as i32 } else { 0 };
                if dot_product != expected {
                    return false;
                }
            }
        }

        true
    }

    /// Create a Hadamard matrix of order 1
    pub fn order_1() -> Self {
        HadamardMatrix {
            n: 1,
            matrix: vec![vec![1]],
        }
    }

    /// Create a Hadamard matrix of order 2
    pub fn order_2() -> Self {
        HadamardMatrix {
            n: 2,
            matrix: vec![vec![1, 1], vec![1, -1]],
        }
    }

    /// Create a Hadamard matrix using Sylvester's construction
    /// H_{2n} = H_2 ⊗ H_n (Kronecker product)
    pub fn sylvester(k: usize) -> Self {
        let n = 2_usize.pow(k as u32);

        if k == 0 {
            return Self::order_1();
        }

        let h_prev = Self::sylvester(k - 1);
        let h2 = Self::order_2();

        // Kronecker product
        let mut matrix = vec![vec![0; n]; n];
        let n_prev = h_prev.n;

        for i in 0..n {
            for j in 0..n {
                let i1 = i / n_prev;
                let i2 = i % n_prev;
                let j1 = j / n_prev;
                let j2 = j % n_prev;

                matrix[i][j] = h2.matrix[i1][j1] * h_prev.matrix[i2][j2];
            }
        }

        HadamardMatrix { n, matrix }
    }

    /// Create a Hadamard matrix of order 4 (base case for Sylvester construction)
    pub fn order_4() -> Self {
        HadamardMatrix {
            n: 4,
            matrix: vec![
                vec![1, 1, 1, 1],
                vec![1, -1, 1, -1],
                vec![1, 1, -1, -1],
                vec![1, -1, -1, 1],
            ],
        }
    }

    /// Get the order
    pub fn order(&self) -> usize {
        self.n
    }

    /// Get the matrix
    pub fn matrix(&self) -> &[Vec<i32>] {
        &self.matrix
    }

    /// Get an entry
    pub fn get(&self, i: usize, j: usize) -> Option<i32> {
        self.matrix.get(i)?.get(j).copied()
    }

    /// Normalize the Hadamard matrix (make first row and column all +1)
    pub fn normalize(&self) -> Self {
        if self.n == 0 {
            return self.clone();
        }

        let mut matrix = self.matrix.clone();

        // Multiply rows by first column entries to make first column all +1
        for i in 0..self.n {
            if matrix[i][0] == -1 {
                for j in 0..self.n {
                    matrix[i][j] *= -1;
                }
            }
        }

        // Multiply columns by first row entries to make first row all +1
        for j in 0..self.n {
            if matrix[0][j] == -1 {
                for i in 0..self.n {
                    matrix[i][j] *= -1;
                }
            }
        }

        HadamardMatrix { n: self.n, matrix }
    }
}

/// Design automorphisms - permutations that preserve the design structure
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesignAutomorphism {
    /// The permutation of points
    permutation: Vec<usize>,
    /// The design this automorphism acts on
    v: usize,
}

impl DesignAutomorphism {
    /// Create a new design automorphism
    pub fn new(permutation: Vec<usize>) -> Option<Self> {
        let v = permutation.len();

        // Verify it's a permutation
        let mut seen = vec![false; v];
        for &p in &permutation {
            if p >= v || seen[p] {
                return None;
            }
            seen[p] = true;
        }

        Some(DesignAutomorphism { permutation, v })
    }

    /// Identity automorphism
    pub fn identity(v: usize) -> Self {
        DesignAutomorphism {
            permutation: (0..v).collect(),
            v,
        }
    }

    /// Apply the automorphism to a point
    pub fn apply(&self, point: usize) -> Option<usize> {
        if point >= self.v {
            return None;
        }
        Some(self.permutation[point])
    }

    /// Apply the automorphism to a block
    pub fn apply_to_block(&self, block: &[usize]) -> Vec<usize> {
        block.iter().filter_map(|&p| self.apply(p)).collect()
    }

    /// Apply the automorphism to a design
    pub fn apply_to_design(&self, design: &BlockDesign) -> Option<BlockDesign> {
        if design.v != self.v {
            return None;
        }

        let new_blocks: Vec<Vec<usize>> = design
            .blocks
            .iter()
            .map(|block| self.apply_to_block(block))
            .collect();

        BlockDesign::new(design.v, design.k, design.t, design.lambda, new_blocks)
    }

    /// Compose two automorphisms
    pub fn compose(&self, other: &DesignAutomorphism) -> Option<Self> {
        if self.v != other.v {
            return None;
        }

        let permutation: Vec<usize> = (0..self.v)
            .map(|i| self.permutation[other.permutation[i]])
            .collect();

        Some(DesignAutomorphism { permutation, v: self.v })
    }

    /// Compute the inverse automorphism
    pub fn inverse(&self) -> Self {
        let mut inv_perm = vec![0; self.v];
        for i in 0..self.v {
            inv_perm[self.permutation[i]] = i;
        }

        DesignAutomorphism {
            permutation: inv_perm,
            v: self.v,
        }
    }

    /// Get the permutation
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// Check if this automorphism preserves a design
    pub fn preserves_design(&self, design: &BlockDesign) -> bool {
        if design.v != self.v {
            return false;
        }

        let transformed_design = match self.apply_to_design(design) {
            Some(d) => d,
            None => return false,
        };

        // Check if the block multisets are the same
        let mut original_blocks = design.blocks.clone();
        let mut transformed_blocks = transformed_design.blocks.clone();

        for block in &mut original_blocks {
            block.sort_unstable();
        }
        for block in &mut transformed_blocks {
            block.sort_unstable();
        }

        original_blocks.sort();
        transformed_blocks.sort();

        original_blocks == transformed_blocks
    }

    /// Find all automorphisms of a design (brute force for small designs)
    pub fn find_automorphisms(design: &BlockDesign) -> Vec<DesignAutomorphism> {
        if design.v > 8 {
            // Too large for brute force
            return vec![DesignAutomorphism::identity(design.v)];
        }

        let mut automorphisms = Vec::new();
        let points: Vec<usize> = (0..design.v).collect();

        Self::find_automorphisms_helper(design, &points, &mut vec![], &mut automorphisms);

        automorphisms
    }

    fn find_automorphisms_helper(
        design: &BlockDesign,
        remaining: &[usize],
        current: &mut Vec<usize>,
        result: &mut Vec<DesignAutomorphism>,
    ) {
        if remaining.is_empty() {
            let auto = DesignAutomorphism {
                permutation: current.clone(),
                v: design.v,
            };
            if auto.preserves_design(design) {
                result.push(auto);
            }
            return;
        }

        for (i, &point) in remaining.iter().enumerate() {
            current.push(point);
            let mut new_remaining = remaining.to_vec();
            new_remaining.remove(i);
            Self::find_automorphisms_helper(design, &new_remaining, current, result);
            current.pop();
        }
    }
}

/// Check if two Latin squares are orthogonal
pub fn are_latin_squares_orthogonal(square1: &[Vec<usize>], square2: &[Vec<usize>]) -> bool {
    let n = square1.len();

    if n == 0 || square2.len() != n {
        return false;
    }

    // Check dimensions
    for row in square1.iter().chain(square2.iter()) {
        if row.len() != n {
            return false;
        }
    }

    // Check that when superimposed, all n² ordered pairs appear
    let mut pairs = HashSet::new();
    for i in 0..n {
        for j in 0..n {
            pairs.insert((square1[i][j], square2[i][j]));
        }
    }

    pairs.len() == n * n
}

/// Generate a pair of mutually orthogonal Latin squares (MOLS) of order n
/// Returns None if no such pair is known to exist
pub fn mutually_orthogonal_latin_squares(n: usize) -> Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    if n <= 1 || n == 2 || n == 6 {
        return None; // No MOLS exist for these orders
    }

    // For prime powers, we can construct MOLS
    // Simple construction for n = 3, 4, 5, 7
    match n {
        3 => Some((
            vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]],
            vec![vec![0, 1, 2], vec![2, 0, 1], vec![1, 2, 0]],
        )),
        4 => Some((
            vec![
                vec![0, 1, 2, 3],
                vec![1, 0, 3, 2],
                vec![2, 3, 0, 1],
                vec![3, 2, 1, 0],
            ],
            vec![
                vec![0, 1, 2, 3],
                vec![2, 3, 0, 1],
                vec![3, 2, 1, 0],
                vec![1, 0, 3, 2],
            ],
        )),
        5 => Some((
            vec![
                vec![0, 1, 2, 3, 4],
                vec![1, 2, 3, 4, 0],
                vec![2, 3, 4, 0, 1],
                vec![3, 4, 0, 1, 2],
                vec![4, 0, 1, 2, 3],
            ],
            vec![
                vec![0, 1, 2, 3, 4],
                vec![2, 3, 4, 0, 1],
                vec![4, 0, 1, 2, 3],
                vec![1, 2, 3, 4, 0],
                vec![3, 4, 0, 1, 2],
            ],
        )),
        7 => Some((
            vec![
                vec![0, 1, 2, 3, 4, 5, 6],
                vec![1, 2, 3, 4, 5, 6, 0],
                vec![2, 3, 4, 5, 6, 0, 1],
                vec![3, 4, 5, 6, 0, 1, 2],
                vec![4, 5, 6, 0, 1, 2, 3],
                vec![5, 6, 0, 1, 2, 3, 4],
                vec![6, 0, 1, 2, 3, 4, 5],
            ],
            vec![
                vec![0, 1, 2, 3, 4, 5, 6],
                vec![2, 3, 4, 5, 6, 0, 1],
                vec![4, 5, 6, 0, 1, 2, 3],
                vec![6, 0, 1, 2, 3, 4, 5],
                vec![1, 2, 3, 4, 5, 6, 0],
                vec![3, 4, 5, 6, 0, 1, 2],
                vec![5, 6, 0, 1, 2, 3, 4],
            ],
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_design_basic() {
        // A simple 2-(4,3,2) design
        let design = BlockDesign::new(
            4,
            3,
            2,
            2,
            vec![
                vec![0, 1, 2],
                vec![0, 1, 3],
                vec![0, 2, 3],
                vec![1, 2, 3],
            ],
        );
        assert!(design.is_some());
        let design = design.unwrap();
        assert_eq!(design.num_blocks(), 4);
        assert!(design.is_bibd());
    }

    #[test]
    fn test_steiner_triple_system() {
        // Fano plane S(2,3,7)
        let sts = SteinerSystem::steiner_triple_system(7);
        assert!(sts.is_some());
        let sts = sts.unwrap();
        assert_eq!(sts.parameters(), (2, 3, 7));
        assert_eq!(sts.design().num_blocks(), 7);
    }

    #[test]
    fn test_steiner_triple_system_invalid() {
        // v = 5 doesn't satisfy v ≡ 1 or 3 (mod 6)
        assert!(SteinerSystem::steiner_triple_system(5).is_none());
    }

    #[test]
    fn test_difference_set_fano() {
        let diffset = DifferenceSet::fano_plane();
        assert_eq!(diffset.parameters(), (7, 3, 1));
        assert_eq!(diffset.elements(), &[0, 1, 3]);

        // Develop into a design
        let design = diffset.develop();
        assert_eq!(design.v(), 7);
        assert_eq!(design.num_blocks(), 7);
    }

    #[test]
    fn test_hadamard_matrix_order_2() {
        let h = HadamardMatrix::order_2();
        assert_eq!(h.order(), 2);
        assert!(h.verify_hadamard_property());
    }

    #[test]
    fn test_hadamard_matrix_sylvester() {
        // Create H_4 using Sylvester construction
        let h4 = HadamardMatrix::sylvester(2);
        assert_eq!(h4.order(), 4);
        assert!(h4.verify_hadamard_property());

        // Create H_8
        let h8 = HadamardMatrix::sylvester(3);
        assert_eq!(h8.order(), 8);
        assert!(h8.verify_hadamard_property());
    }

    #[test]
    fn test_hadamard_normalize() {
        let h = HadamardMatrix::order_4();
        let normalized = h.normalize();

        // First row and column should be all +1
        for j in 0..4 {
            assert_eq!(normalized.get(0, j), Some(1));
        }
        for i in 0..4 {
            assert_eq!(normalized.get(i, 0), Some(1));
        }
    }

    #[test]
    fn test_design_automorphism_identity() {
        let auto = DesignAutomorphism::identity(5);
        assert_eq!(auto.apply(3), Some(3));

        let block = vec![0, 1, 2];
        assert_eq!(auto.apply_to_block(&block), block);
    }

    #[test]
    fn test_design_automorphism_compose() {
        // Create two simple permutations
        let auto1 = DesignAutomorphism::new(vec![1, 0, 2]).unwrap(); // Swap 0 and 1
        let auto2 = DesignAutomorphism::new(vec![0, 2, 1]).unwrap(); // Swap 1 and 2

        let composed = auto1.compose(&auto2).unwrap();
        // auto2 first: 0->0, 1->2, 2->1
        // then auto1: 0->1, 2->2, 1->0
        // Result: 0->1, 1->2, 2->0
        assert_eq!(composed.permutation(), &[1, 2, 0]);
    }

    #[test]
    fn test_design_automorphism_inverse() {
        let auto = DesignAutomorphism::new(vec![2, 0, 1]).unwrap();
        let inv = auto.inverse();

        let identity = auto.compose(&inv).unwrap();
        assert_eq!(identity.permutation(), &[0, 1, 2]);
    }

    #[test]
    fn test_orthogonal_latin_squares() {
        let (l1, l2) = mutually_orthogonal_latin_squares(3).unwrap();
        assert!(are_latin_squares_orthogonal(&l1, &l2));
    }

    #[test]
    fn test_orthogonal_array_from_mols() {
        let (l1, l2) = mutually_orthogonal_latin_squares(3).unwrap();
        let oa = OrthogonalArray::from_mols(&[l1, l2]);
        assert!(oa.is_some());

        let oa = oa.unwrap();
        assert_eq!(oa.parameters(), (2, 4, 3)); // OA(2, 4, 3)
        assert_eq!(oa.rows().len(), 9); // 3^2 = 9 rows
    }

    #[test]
    fn test_no_mols_for_order_2() {
        assert!(mutually_orthogonal_latin_squares(2).is_none());
    }

    #[test]
    fn test_no_mols_for_order_6() {
        // Famous result: no MOLS of order 6 exist (Euler's conjecture, proved by Tarry)
        assert!(mutually_orthogonal_latin_squares(6).is_none());
    }
}
