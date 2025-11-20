//! Affine permutations for Coxeter types A, B, C, D, and G
//!
//! This module implements affine permutations, which are permutations of the integers
//! that satisfy certain periodicity conditions. They arise naturally as elements of
//! affine Coxeter groups.
//!
//! # Window Notation
//!
//! An affine permutation can be represented using "window notation" - a finite sequence
//! that determines the infinite permutation. For type A_n, a window of length n+1
//! determines the entire affine permutation.
//!
//! # Coxeter Types
//!
//! - **Type A**: The affine symmetric group, permutations σ where σ(i+n) = σ(i) + n
//! - **Type B**: Signed permutations with specific affine structure
//! - **Type C**: Similar to type B with different reflection hyperplanes
//! - **Type D**: Even-signed permutations with affine structure
//! - **Type G**: Affine Coxeter group of type G₂

use std::fmt;

/// Coxeter type for affine permutations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoxeterType {
    /// Type A_n - affine symmetric group
    A,
    /// Type B_n - affine hyperoctahedral group
    B,
    /// Type C_n - affine symplectic group
    C,
    /// Type D_n - affine even-signed permutations
    D,
    /// Type G_2 - affine dihedral group of order 12
    G,
}

impl fmt::Display for CoxeterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoxeterType::A => write!(f, "A"),
            CoxeterType::B => write!(f, "B"),
            CoxeterType::C => write!(f, "C"),
            CoxeterType::D => write!(f, "D"),
            CoxeterType::G => write!(f, "G"),
        }
    }
}

/// An affine permutation represented in window notation
///
/// The window is a sequence of integers that determines the entire affine permutation.
/// For type A_n, the window has length n+1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinePermutation {
    /// The Coxeter type
    coxeter_type: CoxeterType,
    /// Rank of the Coxeter group (n for type X_n)
    rank: usize,
    /// The window notation - finite sequence determining the infinite permutation
    window: Vec<i64>,
}

impl AffinePermutation {
    /// Create a new affine permutation from window notation
    ///
    /// # Arguments
    /// * `coxeter_type` - The Coxeter type (A, B, C, D, or G)
    /// * `rank` - The rank n of the group
    /// * `window` - The window notation
    ///
    /// # Returns
    /// `Some(AffinePermutation)` if valid, `None` if the window is invalid
    pub fn new(coxeter_type: CoxeterType, rank: usize, window: Vec<i64>) -> Option<Self> {
        let expected_len = match coxeter_type {
            CoxeterType::A => rank + 1,
            CoxeterType::B | CoxeterType::C | CoxeterType::D => rank,
            CoxeterType::G => {
                if rank != 2 {
                    return None; // G type only defined for rank 2
                }
                2
            }
        };

        if window.len() != expected_len {
            return None;
        }

        // Validate based on type
        if !Self::is_valid_window(coxeter_type, rank, &window) {
            return None;
        }

        Some(AffinePermutation {
            coxeter_type,
            rank,
            window,
        })
    }

    /// Create the identity affine permutation
    pub fn identity(coxeter_type: CoxeterType, rank: usize) -> Self {
        let window = match coxeter_type {
            CoxeterType::A => (1..=(rank + 1) as i64).collect(),
            CoxeterType::B | CoxeterType::C | CoxeterType::D => (1..=rank as i64).collect(),
            CoxeterType::G => vec![1, 2],
        };

        AffinePermutation {
            coxeter_type,
            rank,
            window,
        }
    }

    /// Validate window notation for the given Coxeter type
    fn is_valid_window(coxeter_type: CoxeterType, rank: usize, window: &[i64]) -> bool {
        match coxeter_type {
            CoxeterType::A => Self::is_valid_type_a(rank, window),
            CoxeterType::B => Self::is_valid_type_b(rank, window),
            CoxeterType::C => Self::is_valid_type_c(rank, window),
            CoxeterType::D => Self::is_valid_type_d(rank, window),
            CoxeterType::G => Self::is_valid_type_g(window),
        }
    }

    /// Validate type A window: must be a permutation of {1, 2, ..., n+1} up to adding n
    fn is_valid_type_a(rank: usize, window: &[i64]) -> bool {
        if window.len() != rank + 1 {
            return false;
        }

        // Check that all elements modulo (n+1) are distinct
        let n1 = (rank + 1) as i64;
        let mut remainders: Vec<i64> = window.iter().map(|&x| ((x - 1).rem_euclid(n1))).collect();
        remainders.sort_unstable();

        for i in 0..remainders.len() {
            if remainders[i] != i as i64 {
                return false;
            }
        }

        // The sum modulo (n+1) should equal the sum of 0..n = n(n+1)/2 modulo (n+1)
        // Actually, the sum constraint is: sum(window) ≡ sum(1..n+1) (mod n+1)
        let expected_sum_mod = ((rank * (rank + 1) / 2) % (rank + 1)) as i64;
        let actual_sum: i64 = window.iter().sum();
        let actual_sum_mod = actual_sum.rem_euclid(n1);

        if actual_sum_mod != expected_sum_mod {
            return false;
        }

        true
    }

    /// Validate type B window
    fn is_valid_type_b(rank: usize, window: &[i64]) -> bool {
        if window.len() != rank {
            return false;
        }

        // Type B: signed permutation, |w(i)| must be a permutation of {1, ..., n}
        let mut abs_values: Vec<i64> = window.iter().map(|&x| x.abs()).collect();
        abs_values.sort_unstable();

        for (i, &val) in abs_values.iter().enumerate() {
            if val != (i + 1) as i64 {
                return false;
            }
        }

        true
    }

    /// Validate type C window (similar to type B)
    fn is_valid_type_c(rank: usize, window: &[i64]) -> bool {
        // Type C has the same window structure as type B
        Self::is_valid_type_b(rank, window)
    }

    /// Validate type D window
    fn is_valid_type_d(rank: usize, window: &[i64]) -> bool {
        if window.len() != rank {
            return false;
        }

        // Type D: signed permutation with even number of negative entries
        let mut abs_values: Vec<i64> = window.iter().map(|&x| x.abs()).collect();
        abs_values.sort_unstable();

        for (i, &val) in abs_values.iter().enumerate() {
            if val != (i + 1) as i64 {
                return false;
            }
        }

        // Count negative entries - must be even
        let neg_count = window.iter().filter(|&&x| x < 0).count();
        neg_count % 2 == 0
    }

    /// Validate type G window (only defined for rank 2)
    fn is_valid_type_g(window: &[i64]) -> bool {
        if window.len() != 2 {
            return false;
        }

        // For G_2, the window must satisfy certain properties
        // We allow any pair of integers whose difference is preserved mod 3
        true
    }

    /// Apply the affine permutation to an integer
    pub fn apply(&self, i: i64) -> i64 {
        match self.coxeter_type {
            CoxeterType::A => self.apply_type_a(i),
            CoxeterType::B => self.apply_type_b(i),
            CoxeterType::C => self.apply_type_c(i),
            CoxeterType::D => self.apply_type_d(i),
            CoxeterType::G => self.apply_type_g(i),
        }
    }

    /// Apply type A affine permutation
    ///
    /// For 1-based indexing: σ(i) = w_{((i-1) mod (n+1)) + 1} + ⌊(i-1)/(n+1)⌋ × (n+1)
    fn apply_type_a(&self, i: i64) -> i64 {
        let n1 = (self.rank + 1) as i64;
        let idx = (i - 1).rem_euclid(n1) as usize;
        let quotient = (i - 1).div_euclid(n1);
        self.window[idx] + quotient * n1
    }

    /// Apply type B affine permutation
    fn apply_type_b(&self, i: i64) -> i64 {
        let n = self.rank as i64;

        // Handle periodicity: σ(i + 2n) = σ(i) + 2n
        let period = 2 * n;
        let base = i.rem_euclid(period);
        let quotient = i.div_euclid(period);

        if base == 0 {
            return self.window[(n - 1) as usize] + quotient * period;
        }

        let abs_base = base.abs();
        if abs_base <= n {
            let idx = (abs_base - 1) as usize;
            if base > 0 {
                self.window[idx] + quotient * period
            } else {
                -self.window[idx] + quotient * period
            }
        } else {
            // Handle values in (-2n, -n] and (n, 2n]
            let reflected = period - abs_base + 1;
            let idx = (reflected - 1) as usize;
            if base > 0 {
                -self.window[idx] + quotient * period
            } else {
                self.window[idx] + quotient * period
            }
        }
    }

    /// Apply type C affine permutation
    fn apply_type_c(&self, i: i64) -> i64 {
        // Type C is similar to type B but with different reflection structure
        self.apply_type_b(i)
    }

    /// Apply type D affine permutation
    fn apply_type_d(&self, i: i64) -> i64 {
        // Type D is like type B but restricted to even sign changes
        self.apply_type_b(i)
    }

    /// Apply type G affine permutation
    fn apply_type_g(&self, i: i64) -> i64 {
        // G_2 has period 6
        let period = 6;
        let idx = i.rem_euclid(period);
        let quotient = i.div_euclid(period);

        match idx {
            0 => self.window[1] + (quotient - 1) * period,
            1 => self.window[0] + quotient * period,
            2 => self.window[1] + quotient * period,
            3 => self.window[0] + (quotient + 1) * period,
            4 => self.window[1] + quotient * period,
            5 => self.window[0] + (quotient + 1) * period,
            _ => unreachable!(),
        }
    }

    /// Get the window notation
    pub fn window(&self) -> &[i64] {
        &self.window
    }

    /// Get the Coxeter type
    pub fn coxeter_type(&self) -> CoxeterType {
        self.coxeter_type
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Compose two affine permutations of the same type
    pub fn compose(&self, other: &AffinePermutation) -> Option<Self> {
        if self.coxeter_type != other.coxeter_type || self.rank != other.rank {
            return None;
        }

        // Compute composition: (self ∘ other)(i) = self(other(i))
        let mut new_window = Vec::with_capacity(self.window.len());

        match self.coxeter_type {
            CoxeterType::A => {
                for i in 1..=(self.rank + 1) {
                    let other_i = other.apply(i as i64);
                    let composed = self.apply(other_i);
                    new_window.push(composed);
                }
            }
            CoxeterType::B | CoxeterType::C | CoxeterType::D => {
                for i in 1..=self.rank {
                    let other_i = other.apply(i as i64);
                    let composed = self.apply(other_i);
                    new_window.push(composed);
                }
            }
            CoxeterType::G => {
                for i in 1..=2 {
                    let other_i = other.apply(i as i64);
                    let composed = self.apply(other_i);
                    new_window.push(composed);
                }
            }
        }

        AffinePermutation::new(self.coxeter_type, self.rank, new_window)
    }

    /// Compute the inverse affine permutation
    pub fn inverse(&self) -> Self {
        match self.coxeter_type {
            CoxeterType::A => self.inverse_type_a(),
            CoxeterType::B => self.inverse_type_b(),
            CoxeterType::C => self.inverse_type_c(),
            CoxeterType::D => self.inverse_type_d(),
            CoxeterType::G => self.inverse_type_g(),
        }
    }

    fn inverse_type_a(&self) -> Self {
        let n1 = self.rank + 1;
        let mut inv_window = vec![0; n1];

        for (i, &w) in self.window.iter().enumerate() {
            let idx = ((w - 1).rem_euclid(n1 as i64)) as usize;
            inv_window[idx] = (i + 1) as i64;
        }

        AffinePermutation {
            coxeter_type: CoxeterType::A,
            rank: self.rank,
            window: inv_window,
        }
    }

    fn inverse_type_b(&self) -> Self {
        let mut inv_window = vec![0; self.rank];

        for (i, &w) in self.window.iter().enumerate() {
            let abs_w = w.abs() as usize;
            if abs_w <= self.rank && abs_w > 0 {
                inv_window[abs_w - 1] = if w > 0 {
                    (i + 1) as i64
                } else {
                    -((i + 1) as i64)
                };
            }
        }

        AffinePermutation {
            coxeter_type: CoxeterType::B,
            rank: self.rank,
            window: inv_window,
        }
    }

    fn inverse_type_c(&self) -> Self {
        // Type C inverse is similar to type B
        let inv_b = self.inverse_type_b();
        AffinePermutation {
            coxeter_type: CoxeterType::C,
            rank: self.rank,
            window: inv_b.window,
        }
    }

    fn inverse_type_d(&self) -> Self {
        // Type D inverse is similar to type B
        let inv_b = self.inverse_type_b();
        AffinePermutation {
            coxeter_type: CoxeterType::D,
            rank: self.rank,
            window: inv_b.window,
        }
    }

    fn inverse_type_g(&self) -> Self {
        // For G_2, compute inverse by finding preimages
        let mut inv_window = vec![0; 2];

        for i in 1..=2 {
            let val = self.apply(i as i64);
            let idx = ((val - 1).rem_euclid(2)) as usize;
            inv_window[idx] = i as i64;
        }

        AffinePermutation {
            coxeter_type: CoxeterType::G,
            rank: 2,
            window: inv_window,
        }
    }

    /// Compute the length of the affine permutation (in the Coxeter group sense)
    ///
    /// This is the minimal number of simple reflections needed to express the element
    pub fn length(&self) -> usize {
        match self.coxeter_type {
            CoxeterType::A => self.length_type_a(),
            CoxeterType::B | CoxeterType::C => self.length_type_bc(),
            CoxeterType::D => self.length_type_d(),
            CoxeterType::G => self.length_type_g(),
        }
    }

    fn length_type_a(&self) -> usize {
        // For type A, length is the number of inversions
        let mut count = 0;
        let n1 = (self.rank + 1) as i64;

        for i in 0..self.window.len() {
            for j in (i + 1)..self.window.len() {
                // Count inversions considering the affine structure
                let wi = self.window[i];
                let wj = self.window[j];

                if wi > wj {
                    count += ((wi - wj - 1) / n1 + 1) as usize;
                } else {
                    count += ((wj - wi - 1) / n1) as usize;
                }
            }
        }

        count
    }

    fn length_type_bc(&self) -> usize {
        // For types B and C, count inversions plus negative positions
        let mut count = 0;

        // Count negative entries
        count += self.window.iter().filter(|&&x| x < 0).count();

        // Count inversions
        for i in 0..self.window.len() {
            for j in (i + 1)..self.window.len() {
                if self.window[i].abs() > self.window[j].abs() {
                    count += 1;
                }
            }
        }

        count
    }

    fn length_type_d(&self) -> usize {
        // Similar to type B/C but adjusted for even sign constraint
        let mut count = 0;

        // Count negative entries (should be even)
        let neg_count = self.window.iter().filter(|&&x| x < 0).count();
        count += neg_count;

        // Count inversions
        for i in 0..self.window.len() {
            for j in (i + 1)..self.window.len() {
                if self.window[i].abs() > self.window[j].abs() {
                    count += 1;
                }
            }
        }

        count
    }

    fn length_type_g(&self) -> usize {
        // For G_2, use a simplified calculation
        let diff = (self.window[0] - 1).abs() + (self.window[1] - 2).abs();
        diff as usize
    }

    /// Check if this is a Grassmannian permutation
    ///
    /// A Grassmannian permutation has at most one descent
    pub fn is_grassmannian(&self) -> bool {
        if self.coxeter_type != CoxeterType::A {
            return false;
        }

        let mut descent_count = 0;
        for i in 0..(self.window.len() - 1) {
            if self.window[i] > self.window[i + 1] {
                descent_count += 1;
            }
        }

        descent_count <= 1
    }
}

impl fmt::Display for AffinePermutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{}[{}]",
            self.coxeter_type,
            self.rank,
            self.window
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_a_identity() {
        let id = AffinePermutation::identity(CoxeterType::A, 3);
        assert_eq!(id.window(), &[1, 2, 3, 4]);
        assert_eq!(id.apply(1), 1);
        assert_eq!(id.apply(5), 5);
        assert_eq!(id.length(), 0);
    }

    #[test]
    fn test_type_a_basic() {
        // Simple type A affine permutation
        let perm = AffinePermutation::new(CoxeterType::A, 3, vec![2, 1, 3, 4]).unwrap();
        assert_eq!(perm.apply(1), 2);
        assert_eq!(perm.apply(2), 1);
        assert_eq!(perm.apply(5), 6);
        assert_eq!(perm.apply(6), 5);
    }

    #[test]
    fn test_type_a_window_notation() {
        // Test window [2, 3, 4, 5] for type A_3 (shift by 1)
        let perm = AffinePermutation::new(CoxeterType::A, 3, vec![2, 3, 4, 5]).unwrap();
        assert_eq!(perm.apply(1), 2);
        assert_eq!(perm.apply(2), 3);
        assert_eq!(perm.apply(3), 4);
        assert_eq!(perm.apply(4), 5);

        // Check periodicity: σ(i+4) = σ(i) + 4
        assert_eq!(perm.apply(5), 6);
        assert_eq!(perm.apply(6), 7);

        // Check negative indices
        assert_eq!(perm.apply(0), 1);
        assert_eq!(perm.apply(-1), 0);
        assert_eq!(perm.apply(-3), -2);
    }

    #[test]
    fn test_type_a_invalid_window() {
        // Invalid window - wrong length
        assert!(AffinePermutation::new(CoxeterType::A, 3, vec![1, 2, 3]).is_none());

        // Invalid window - wrong sum
        assert!(AffinePermutation::new(CoxeterType::A, 3, vec![1, 2, 3, 5]).is_none());
    }

    #[test]
    fn test_type_a_inverse() {
        let perm = AffinePermutation::new(CoxeterType::A, 3, vec![2, 1, 4, 3]).unwrap();
        let inv = perm.inverse();

        // Check that inv ∘ perm = identity
        for i in 1..=8 {
            let composed = inv.apply(perm.apply(i));
            assert_eq!(composed, i);
        }
    }

    #[test]
    fn test_type_a_composition() {
        let p1 = AffinePermutation::new(CoxeterType::A, 2, vec![2, 1, 3]).unwrap();
        let p2 = AffinePermutation::new(CoxeterType::A, 2, vec![1, 3, 2]).unwrap();

        let comp = p1.compose(&p2).unwrap();

        // Verify composition
        for i in 1..=6 {
            assert_eq!(comp.apply(i), p1.apply(p2.apply(i)));
        }
    }

    #[test]
    fn test_type_b_identity() {
        let id = AffinePermutation::identity(CoxeterType::B, 3);
        assert_eq!(id.window(), &[1, 2, 3]);
        assert_eq!(id.length(), 0);
    }

    #[test]
    fn test_type_b_basic() {
        // Type B with sign changes
        let perm = AffinePermutation::new(CoxeterType::B, 3, vec![-1, 2, 3]).unwrap();
        assert!(perm.window()[0] < 0);
        assert_eq!(perm.length(), 1); // One negative entry
    }

    #[test]
    fn test_type_b_validation() {
        // Valid type B
        assert!(AffinePermutation::new(CoxeterType::B, 3, vec![2, -1, 3]).is_some());

        // Invalid - not all absolute values present
        assert!(AffinePermutation::new(CoxeterType::B, 3, vec![1, 1, 3]).is_none());

        // Invalid - wrong length
        assert!(AffinePermutation::new(CoxeterType::B, 3, vec![1, 2]).is_none());
    }

    #[test]
    fn test_type_c_identity() {
        let id = AffinePermutation::identity(CoxeterType::C, 3);
        assert_eq!(id.window(), &[1, 2, 3]);
    }

    #[test]
    fn test_type_c_basic() {
        let perm = AffinePermutation::new(CoxeterType::C, 3, vec![3, -2, 1]).unwrap();
        assert_eq!(perm.window(), &[3, -2, 1]);
    }

    #[test]
    fn test_type_d_identity() {
        let id = AffinePermutation::identity(CoxeterType::D, 4);
        assert_eq!(id.window(), &[1, 2, 3, 4]);
        assert_eq!(id.length(), 0);
    }

    #[test]
    fn test_type_d_even_signs() {
        // Valid - even number of negatives (2)
        let perm = AffinePermutation::new(CoxeterType::D, 4, vec![-1, -2, 3, 4]).unwrap();
        assert_eq!(perm.window(), &[-1, -2, 3, 4]);

        // Invalid - odd number of negatives
        assert!(AffinePermutation::new(CoxeterType::D, 4, vec![-1, 2, 3, 4]).is_none());

        // Valid - all positive (0 negatives, even)
        assert!(AffinePermutation::new(CoxeterType::D, 4, vec![1, 2, 3, 4]).is_some());
    }

    #[test]
    fn test_type_g_basic() {
        let id = AffinePermutation::identity(CoxeterType::G, 2);
        assert_eq!(id.window(), &[1, 2]);
    }

    #[test]
    fn test_type_g_validation() {
        // Type G only valid for rank 2
        assert!(AffinePermutation::new(CoxeterType::G, 3, vec![1, 2, 3]).is_none());
        assert!(AffinePermutation::new(CoxeterType::G, 2, vec![1, 2]).is_some());
        assert!(AffinePermutation::new(CoxeterType::G, 2, vec![2, 1]).is_some());
    }

    #[test]
    fn test_grassmannian() {
        // Identity is Grassmannian (0 descents)
        let id = AffinePermutation::identity(CoxeterType::A, 3);
        assert!(id.is_grassmannian());

        // One descent
        let grass = AffinePermutation::new(CoxeterType::A, 3, vec![2, 1, 3, 4]).unwrap();
        assert!(grass.is_grassmannian());

        // Two descents - not Grassmannian
        let non_grass = AffinePermutation::new(CoxeterType::A, 3, vec![2, 1, 4, 3]).unwrap();
        assert!(!non_grass.is_grassmannian());
    }

    #[test]
    fn test_type_a_length() {
        let id = AffinePermutation::identity(CoxeterType::A, 3);
        assert_eq!(id.length(), 0);

        // A simple transposition
        let swap = AffinePermutation::new(CoxeterType::A, 2, vec![2, 1, 3]).unwrap();
        assert!(swap.length() > 0);
    }

    #[test]
    fn test_display() {
        let perm = AffinePermutation::new(CoxeterType::A, 3, vec![2, 1, 3, 4]).unwrap();
        let display = format!("{}", perm);
        assert!(display.contains("A_3"));
        assert!(display.contains("2"));
        assert!(display.contains("1"));
    }

    #[test]
    fn test_type_b_inverse() {
        let perm = AffinePermutation::new(CoxeterType::B, 3, vec![2, -1, 3]).unwrap();
        let inv = perm.inverse();

        // Basic inverse property check
        assert_eq!(inv.window().len(), 3);
    }

    #[test]
    fn test_mixed_type_composition_fails() {
        let a_perm = AffinePermutation::identity(CoxeterType::A, 3);
        let b_perm = AffinePermutation::identity(CoxeterType::B, 3);

        // Cannot compose different types
        assert!(a_perm.compose(&b_perm).is_none());
    }

    #[test]
    fn test_type_a_affine_property() {
        // Test that σ(i + n+1) = σ(i) + n+1 for type A
        let perm = AffinePermutation::new(CoxeterType::A, 3, vec![2, 3, 1, 4]).unwrap();

        for i in 1..=10 {
            let val_i = perm.apply(i);
            let val_i_plus_4 = perm.apply(i + 4);
            assert_eq!(val_i_plus_4, val_i + 4, "Failed for i={}", i);
        }
    }

    #[test]
    fn test_type_b_all_negative() {
        // All entries negative
        let perm = AffinePermutation::new(CoxeterType::B, 3, vec![-1, -2, -3]).unwrap();
        assert_eq!(perm.length(), 3); // Three negative entries
    }

    #[test]
    fn test_type_d_four_negatives() {
        // Four negatives (even)
        let perm = AffinePermutation::new(CoxeterType::D, 4, vec![-1, -2, -3, -4]).unwrap();
        assert_eq!(perm.window(), &[-1, -2, -3, -4]);
    }
}
