//! Cartan Matrices
//!
//! The Cartan matrix encodes the structure of a root system through the inner products
//! of simple roots. For simple roots α_i and α_j, the Cartan matrix entry is:
//! A_ij = 2⟨α_i, α_j⟩ / ⟨α_j, α_j⟩
//!
//! The Cartan matrix determines the Dynkin diagram, root system, and Weyl group.
//!
//! Corresponds to sage.combinat.root_system.cartan_matrix

use crate::cartan_type::{CartanType, CartanLetter, Affinity};
use rustmath_core::Ring;
use rustmath_rationals::Rational;
use rustmath_integers::Integer;
use std::fmt::{self, Display};

/// A Cartan matrix for a root system
///
/// The Cartan matrix is an n×n integer matrix where n is the rank of the root system.
/// For finite types, it is:
/// - Indecomposable (cannot be block-diagonalized)
/// - Has 2's on the diagonal
/// - Non-diagonal entries are non-positive integers
/// - A_ij = 0 if and only if A_ji = 0
/// - A_ij * A_ji ∈ {0, 1, 2, 3} determines edge multiplicity in Dynkin diagram
#[derive(Clone, Debug, PartialEq)]
pub struct CartanMatrix {
    /// The Cartan type
    pub cartan_type: CartanType,
    /// The matrix entries (row-major order)
    pub matrix: Vec<Vec<Integer>>,
    /// The rank (dimension)
    pub rank: usize,
}

impl CartanMatrix {
    /// Create a Cartan matrix for a given Cartan type
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
    /// use rustmath_liealgebras::cartan_matrix::CartanMatrix;
    ///
    /// let ct = CartanType::new(CartanLetter::A, 3).unwrap();
    /// let cm = CartanMatrix::new(ct);
    /// assert_eq!(cm.rank(), 3);
    /// ```
    pub fn new(cartan_type: CartanType) -> Self {
        let rank = cartan_type.rank;
        let matrix = Self::construct_matrix(&cartan_type);

        CartanMatrix {
            cartan_type,
            matrix,
            rank,
        }
    }

    /// Construct the Cartan matrix for a given type
    fn construct_matrix(ct: &CartanType) -> Vec<Vec<Integer>> {
        match ct.affinity {
            Affinity::Finite => Self::construct_finite_matrix(ct),
            Affinity::Affine(_) => Self::construct_affine_matrix(ct),
        }
    }

    /// Construct a finite Cartan matrix
    fn construct_finite_matrix(ct: &CartanType) -> Vec<Vec<Integer>> {
        match ct.letter {
            CartanLetter::A => Self::cartan_matrix_a(ct.rank),
            CartanLetter::B => Self::cartan_matrix_b(ct.rank),
            CartanLetter::C => Self::cartan_matrix_c(ct.rank),
            CartanLetter::D => Self::cartan_matrix_d(ct.rank),
            CartanLetter::E => Self::cartan_matrix_e(ct.rank),
            CartanLetter::F => Self::cartan_matrix_f(),
            CartanLetter::G => Self::cartan_matrix_g(),
        }
    }

    /// Construct an affine Cartan matrix
    fn construct_affine_matrix(ct: &CartanType) -> Vec<Vec<Integer>> {
        // For affine types, we extend the finite Cartan matrix by one dimension
        match ct.letter {
            CartanLetter::A => Self::cartan_matrix_a_affine(ct.rank),
            CartanLetter::B => Self::cartan_matrix_b_affine(ct.rank),
            CartanLetter::C => Self::cartan_matrix_c_affine(ct.rank),
            CartanLetter::D => Self::cartan_matrix_d_affine(ct.rank),
            CartanLetter::E => Self::cartan_matrix_e_affine(ct.rank),
            CartanLetter::F => Self::cartan_matrix_f_affine(),
            CartanLetter::G => Self::cartan_matrix_g_affine(),
        }
    }

    /// Type A_n Cartan matrix
    /// A_ij = 2 if i=j, -1 if |i-j|=1, 0 otherwise
    fn cartan_matrix_a(n: usize) -> Vec<Vec<Integer>> {
        let mut matrix = vec![vec![Integer::zero(); n]; n];
        for i in 0..n {
            matrix[i][i] = Integer::from(2);
            if i > 0 {
                matrix[i][i - 1] = Integer::from(-1);
            }
            if i < n - 1 {
                matrix[i][i + 1] = Integer::from(-1);
            }
        }
        matrix
    }

    /// Type B_n Cartan matrix
    fn cartan_matrix_b(n: usize) -> Vec<Vec<Integer>> {
        let mut matrix = vec![vec![Integer::zero(); n]; n];
        for i in 0..n {
            matrix[i][i] = Integer::from(2);
            if i > 0 {
                matrix[i][i - 1] = Integer::from(-1);
            }
            if i < n - 1 {
                matrix[i][i + 1] = Integer::from(-1);
            }
        }
        // B_n has a double bond at the end: α_{n-1} ← α_n
        // A_{n-1,n} = -1, A_{n,n-1} = -2
        if n >= 2 {
            matrix[n - 1][n - 2] = Integer::from(-2);
        }
        matrix
    }

    /// Type C_n Cartan matrix
    fn cartan_matrix_c(n: usize) -> Vec<Vec<Integer>> {
        let mut matrix = vec![vec![Integer::zero(); n]; n];
        for i in 0..n {
            matrix[i][i] = Integer::from(2);
            if i > 0 {
                matrix[i][i - 1] = Integer::from(-1);
            }
            if i < n - 1 {
                matrix[i][i + 1] = Integer::from(-1);
            }
        }
        // C_n has a double bond at the end: α_{n-1} → α_n
        // A_{n-1,n} = -2, A_{n,n-1} = -1
        if n >= 2 {
            matrix[n - 2][n - 1] = Integer::from(-2);
        }
        matrix
    }

    /// Type D_n Cartan matrix
    fn cartan_matrix_d(n: usize) -> Vec<Vec<Integer>> {
        let mut matrix = vec![vec![Integer::zero(); n]; n];
        for i in 0..n {
            matrix[i][i] = Integer::from(2);
        }

        // Linear chain up to n-3
        for i in 0..n - 2 {
            if i > 0 {
                matrix[i][i - 1] = Integer::from(-1);
            }
            matrix[i][i + 1] = Integer::from(-1);
        }

        // D_n has a fork at the end
        // α_{n-3} connects to both α_{n-2} and α_{n-1}
        if n >= 3 {
            matrix[n - 3][n - 2] = Integer::from(-1);
            matrix[n - 2][n - 3] = Integer::from(-1);
            matrix[n - 3][n - 1] = Integer::from(-1);
            matrix[n - 1][n - 3] = Integer::from(-1);
        }

        matrix
    }

    /// Type E_n Cartan matrix (n = 6, 7, 8)
    fn cartan_matrix_e(n: usize) -> Vec<Vec<Integer>> {
        let mut matrix = vec![vec![Integer::zero(); n]; n];

        // Diagonal
        for i in 0..n {
            matrix[i][i] = Integer::from(2);
        }

        // E_n has a linear chain with a branch at position 2 (0-indexed)
        // Linear chain: 0-1-2-3-4-...
        //                   |
        //                   5 (branch)

        // Main chain
        for i in 0..n - 1 {
            if i == 2 {
                // Position 2 connects to both 1, 3, and the branch
                matrix[2][1] = Integer::from(-1);
                matrix[1][2] = Integer::from(-1);
                matrix[2][3] = Integer::from(-1);
                matrix[3][2] = Integer::from(-1);
            } else if i < n - 2 {
                matrix[i][i + 1] = Integer::from(-1);
                matrix[i + 1][i] = Integer::from(-1);
            }
        }

        // Branch at position 2
        // For E_6: nodes 0,1,2,3,4,5 where 5 branches from 2
        // For E_7: add node 6 to the chain
        // For E_8: add node 6 and 7 to the chain

        if n >= 6 {
            // Connect branch node (last node) to position 2
            matrix[2][n - 1] = Integer::from(-1);
            matrix[n - 1][2] = Integer::from(-1);
        }

        // For E_7 and E_8, extend the main chain
        if n >= 7 {
            matrix[4][5] = Integer::from(-1);
            matrix[5][4] = Integer::from(-1);
        }
        if n >= 8 {
            matrix[5][6] = Integer::from(-1);
            matrix[6][5] = Integer::from(-1);
        }

        matrix
    }

    /// Type F_4 Cartan matrix
    fn cartan_matrix_f() -> Vec<Vec<Integer>> {
        vec![
            vec![Integer::from(2), Integer::from(-1), Integer::from(0), Integer::from(0)],
            vec![Integer::from(-1), Integer::from(2), Integer::from(-2), Integer::from(0)],
            vec![Integer::from(0), Integer::from(-1), Integer::from(2), Integer::from(-1)],
            vec![Integer::from(0), Integer::from(0), Integer::from(-1), Integer::from(2)],
        ]
    }

    /// Type G_2 Cartan matrix
    fn cartan_matrix_g() -> Vec<Vec<Integer>> {
        vec![
            vec![Integer::from(2), Integer::from(-1)],
            vec![Integer::from(-3), Integer::from(2)],
        ]
    }

    /// Type A_n^(1) affine Cartan matrix
    fn cartan_matrix_a_affine(n: usize) -> Vec<Vec<Integer>> {
        let size = n + 1;
        let mut matrix = vec![vec![Integer::zero(); size]; size];

        // Circular chain
        for i in 0..size {
            matrix[i][i] = Integer::from(2);
            matrix[i][(i + 1) % size] = Integer::from(-1);
            matrix[(i + 1) % size][i] = Integer::from(-1);
        }

        matrix
    }

    /// Type B_n^(1) affine Cartan matrix
    fn cartan_matrix_b_affine(n: usize) -> Vec<Vec<Integer>> {
        let size = n + 1;
        let mut matrix = vec![vec![Integer::zero(); size]; size];

        // Similar to finite B_n but with additional affine node
        for i in 0..n {
            matrix[i][i] = Integer::from(2);
            if i > 0 {
                matrix[i][i - 1] = Integer::from(-1);
            }
            if i < n - 1 {
                matrix[i][i + 1] = Integer::from(-1);
            }
        }

        // Affine node connections
        matrix[n][n] = Integer::from(2);
        matrix[n][0] = Integer::from(-1);
        matrix[0][n] = Integer::from(-1);

        if n >= 2 {
            matrix[n - 1][n - 2] = Integer::from(-2);
        }

        matrix
    }

    /// Type C_n^(1) affine Cartan matrix
    fn cartan_matrix_c_affine(n: usize) -> Vec<Vec<Integer>> {
        let size = n + 1;
        let mut matrix = Self::cartan_matrix_c(n);

        // Add affine node
        matrix.push(vec![Integer::zero(); n]);
        for row in &mut matrix[0..n] {
            row.push(Integer::zero());
        }

        matrix[n][n] = Integer::from(2);
        matrix[n][0] = Integer::from(-1);
        matrix[0][n] = Integer::from(-2);

        matrix
    }

    /// Type D_n^(1) affine Cartan matrix
    fn cartan_matrix_d_affine(n: usize) -> Vec<Vec<Integer>> {
        let size = n + 1;
        let mut matrix = Self::cartan_matrix_d(n);

        // Add affine node
        matrix.push(vec![Integer::zero(); n]);
        for row in &mut matrix[0..n] {
            row.push(Integer::zero());
        }

        matrix[n][n] = Integer::from(2);
        matrix[n][1] = Integer::from(-1);
        matrix[1][n] = Integer::from(-1);

        matrix
    }

    /// Type E_n^(1) affine Cartan matrix
    fn cartan_matrix_e_affine(n: usize) -> Vec<Vec<Integer>> {
        let size = n + 1;
        let mut matrix = Self::cartan_matrix_e(n);

        // Add affine node
        matrix.push(vec![Integer::zero(); n]);
        for row in &mut matrix[0..n] {
            row.push(Integer::zero());
        }

        matrix[n][n] = Integer::from(2);

        // Connection depends on E type
        match n {
            6 => {
                matrix[n][5] = Integer::from(-1);
                matrix[5][n] = Integer::from(-1);
            }
            7 => {
                matrix[n][0] = Integer::from(-1);
                matrix[0][n] = Integer::from(-1);
            }
            8 => {
                matrix[n][7] = Integer::from(-1);
                matrix[7][n] = Integer::from(-1);
            }
            _ => {}
        }

        matrix
    }

    /// Type F_4^(1) affine Cartan matrix
    fn cartan_matrix_f_affine() -> Vec<Vec<Integer>> {
        vec![
            vec![Integer::from(2), Integer::from(-1), Integer::from(0), Integer::from(0), Integer::from(0)],
            vec![Integer::from(-1), Integer::from(2), Integer::from(-2), Integer::from(0), Integer::from(0)],
            vec![Integer::from(0), Integer::from(-1), Integer::from(2), Integer::from(-1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(0), Integer::from(-1), Integer::from(2), Integer::from(-1)],
            vec![Integer::from(0), Integer::from(0), Integer::from(0), Integer::from(-1), Integer::from(2)],
        ]
    }

    /// Type G_2^(1) affine Cartan matrix
    fn cartan_matrix_g_affine() -> Vec<Vec<Integer>> {
        vec![
            vec![Integer::from(2), Integer::from(-1), Integer::from(0)],
            vec![Integer::from(-3), Integer::from(2), Integer::from(-1)],
            vec![Integer::from(0), Integer::from(-1), Integer::from(2)],
        ]
    }

    /// Get the rank of the Cartan matrix
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get a matrix entry A_ij
    pub fn entry(&self, i: usize, j: usize) -> &Integer {
        &self.matrix[i][j]
    }

    /// Get the determinant of the Cartan matrix
    ///
    /// For finite types, this is positive. For affine types, it is 0.
    pub fn determinant(&self) -> Integer {
        if self.rank == 0 {
            return Integer::one();
        }

        // For affine types, determinant is 0
        if self.cartan_type.is_affine() {
            return Integer::zero();
        }

        // Simple determinant calculation for small matrices
        // Full implementation would use proper matrix determinant algorithm
        match self.rank {
            1 => self.matrix[0][0].clone(),
            2 => {
                let a = &self.matrix[0][0];
                let b = &self.matrix[0][1];
                let c = &self.matrix[1][0];
                let d = &self.matrix[1][1];
                a * d - b * c
            }
            _ => {
                // For larger matrices, return known values
                match self.cartan_type.letter {
                    CartanLetter::A => Integer::from((self.rank + 1) as i64),
                    CartanLetter::B | CartanLetter::C => Integer::from(2),
                    CartanLetter::D => Integer::from(4),
                    CartanLetter::E => match self.rank {
                        6 => Integer::from(3),
                        7 => Integer::from(2),
                        8 => Integer::from(1),
                        _ => Integer::one(),
                    },
                    CartanLetter::F => Integer::from(1),
                    CartanLetter::G => Integer::from(1),
                }
            }
        }
    }

    /// Check if the Cartan matrix is symmetric
    ///
    /// A Cartan matrix is symmetric if and only if the root system is simply-laced
    /// (types A, D, E)
    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.rank {
            for j in i + 1..self.rank {
                if self.matrix[i][j] != self.matrix[j][i] {
                    return false;
                }
            }
        }
        true
    }

    /// Get the symmetrized Cartan matrix (Gram matrix)
    ///
    /// The symmetrized matrix is D*A where D is a diagonal matrix making it symmetric
    pub fn symmetrize(&self) -> Vec<Vec<Rational>> {
        let mut result = vec![vec![Rational::zero(); self.rank]; self.rank];

        // For simply-laced types, the matrix is already symmetric
        if self.is_symmetric() {
            for i in 0..self.rank {
                for j in 0..self.rank {
                    result[i][j] = Rational::from_integer(self.matrix[i][j].clone());
                }
            }
            return result;
        }

        // For non-simply-laced types, compute symmetrization
        for i in 0..self.rank {
            for j in 0..self.rank {
                let a_ij = Rational::from_integer(self.matrix[i][j].clone());
                let a_ji = Rational::from_integer(self.matrix[j][i].clone());
                result[i][j] = a_ij * a_ji;
            }
        }

        result
    }

    /// Get the dual Cartan matrix
    ///
    /// The dual is obtained by transposing the matrix, which swaps B ↔ C
    pub fn dual(&self) -> Self {
        let mut dual_matrix = vec![vec![Integer::zero(); self.rank]; self.rank];
        for i in 0..self.rank {
            for j in 0..self.rank {
                dual_matrix[i][j] = self.matrix[j][i].clone();
            }
        }

        CartanMatrix {
            cartan_type: self.cartan_type.dual(),
            matrix: dual_matrix,
            rank: self.rank,
        }
    }
}

impl Display for CartanMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Cartan matrix of type {}:", self.cartan_type)?;
        for row in &self.matrix {
            write!(f, "[")?;
            for (j, entry) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:3}", entry)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartan_matrix_a2() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let cm = CartanMatrix::new(ct);

        assert_eq!(cm.rank(), 2);
        assert_eq!(*cm.entry(0, 0), Integer::from(2));
        assert_eq!(*cm.entry(0, 1), Integer::from(-1));
        assert_eq!(*cm.entry(1, 0), Integer::from(-1));
        assert_eq!(*cm.entry(1, 1), Integer::from(2));

        assert!(cm.is_symmetric());
    }

    #[test]
    fn test_cartan_matrix_b3() {
        let ct = CartanType::new(CartanLetter::B, 3).unwrap();
        let cm = CartanMatrix::new(ct);

        assert_eq!(cm.rank(), 3);
        assert_eq!(*cm.entry(2, 1), Integer::from(-2));
        assert_eq!(*cm.entry(1, 2), Integer::from(-1));

        assert!(!cm.is_symmetric());
    }

    #[test]
    fn test_cartan_matrix_g2() {
        let ct = CartanType::new(CartanLetter::G, 2).unwrap();
        let cm = CartanMatrix::new(ct);

        assert_eq!(cm.rank(), 2);
        assert_eq!(*cm.entry(1, 0), Integer::from(-3));
        assert_eq!(*cm.entry(0, 1), Integer::from(-1));
    }

    #[test]
    fn test_cartan_matrix_dual() {
        let b3 = CartanType::new(CartanLetter::B, 3).unwrap();
        let c3 = CartanType::new(CartanLetter::C, 3).unwrap();

        let cm_b = CartanMatrix::new(b3);
        let cm_c = CartanMatrix::new(c3);
        let cm_b_dual = cm_b.dual();

        assert_eq!(cm_b_dual.cartan_type, c3);
        assert_eq!(cm_b_dual.matrix, cm_c.matrix);
    }

    #[test]
    fn test_affine_cartan_matrix() {
        let a2_aff = CartanType::new_affine(CartanLetter::A, 2, 1).unwrap();
        let cm = CartanMatrix::new(a2_aff);

        assert_eq!(cm.rank(), 3);
        assert_eq!(cm.determinant(), Integer::zero());
    }

    #[test]
    fn test_cartan_matrix_determinants() {
        let a3 = CartanMatrix::new(CartanType::new(CartanLetter::A, 3).unwrap());
        assert_eq!(a3.determinant(), Integer::from(4));

        let b3 = CartanMatrix::new(CartanType::new(CartanLetter::B, 3).unwrap());
        assert_eq!(b3.determinant(), Integer::from(2));
    }

    #[test]
    fn test_all_finite_types() {
        // Test that all standard finite types can be constructed
        for n in 1..=5 {
            let _ = CartanMatrix::new(CartanType::new(CartanLetter::A, n).unwrap());
        }

        for n in 2..=5 {
            let _ = CartanMatrix::new(CartanType::new(CartanLetter::B, n).unwrap());
            let _ = CartanMatrix::new(CartanType::new(CartanLetter::C, n).unwrap());
        }

        for n in 3..=5 {
            let _ = CartanMatrix::new(CartanType::new(CartanLetter::D, n).unwrap());
        }

        let _ = CartanMatrix::new(CartanType::new(CartanLetter::E, 6).unwrap());
        let _ = CartanMatrix::new(CartanType::new(CartanLetter::E, 7).unwrap());
        let _ = CartanMatrix::new(CartanType::new(CartanLetter::E, 8).unwrap());
        let _ = CartanMatrix::new(CartanType::new(CartanLetter::F, 4).unwrap());
        let _ = CartanMatrix::new(CartanType::new(CartanLetter::G, 2).unwrap());
    }

    #[test]
    fn test_all_affine_types() {
        // Test that all standard affine types can be constructed
        for n in 1..=5 {
            let _ = CartanMatrix::new(CartanType::new_affine(CartanLetter::A, n, 1).unwrap());
        }

        for n in 2..=5 {
            let _ = CartanMatrix::new(CartanType::new_affine(CartanLetter::B, n, 1).unwrap());
            let _ = CartanMatrix::new(CartanType::new_affine(CartanLetter::C, n, 1).unwrap());
        }

        for n in 3..=5 {
            let _ = CartanMatrix::new(CartanType::new_affine(CartanLetter::D, n, 1).unwrap());
        }

        let _ = CartanMatrix::new(CartanType::new_affine(CartanLetter::E, 6, 1).unwrap());
        let _ = CartanMatrix::new(CartanType::new_affine(CartanLetter::E, 7, 1).unwrap());
        let _ = CartanMatrix::new(CartanType::new_affine(CartanLetter::E, 8, 1).unwrap());
        let _ = CartanMatrix::new(CartanType::new_affine(CartanLetter::F, 4, 1).unwrap());
        let _ = CartanMatrix::new(CartanType::new_affine(CartanLetter::G, 2, 1).unwrap());
    }
}
