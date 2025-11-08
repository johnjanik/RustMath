//! Cochain complexes and cohomology computation
//!
//! A cochain complex is the dual of a chain complex, with coboundary maps
//! going up in degree rather than down.

use rustmath_groups::AbelianGroup;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use std::collections::HashMap;
use std::fmt;

/// A cochain complex of free abelian groups
///
/// Represents: ... ← C^{n-1} <--δ^{n-1}-- C^n <--δ^n-- C^{n+1} ← ...
///
/// Where each C^n is a free abelian group Z^{r_n} and δ^n are coboundary maps
/// satisfying δ^{n+1} ∘ δ^n = 0
#[derive(Clone, Debug)]
pub struct CochainComplex {
    /// Dimensions of cochain groups at each degree
    ranks: HashMap<i32, usize>,

    /// Coboundary maps δ^n: C^n → C^{n+1}
    coboundary_maps: HashMap<i32, Matrix<Integer>>,

    /// Minimum and maximum non-zero degrees
    min_degree: i32,
    max_degree: i32,
}

impl CochainComplex {
    /// Create a new cochain complex
    ///
    /// # Arguments
    /// * `ranks` - Dimensions of each cochain group
    /// * `coboundary_maps` - Coboundary maps between consecutive groups
    pub fn new(
        ranks: HashMap<i32, usize>,
        coboundary_maps: HashMap<i32, Matrix<Integer>>,
    ) -> Result<Self, String> {
        // Find min and max degrees
        let mut min_degree = i32::MAX;
        let mut max_degree = i32::MIN;

        for &deg in ranks.keys() {
            min_degree = min_degree.min(deg);
            max_degree = max_degree.max(deg);
        }

        if ranks.is_empty() {
            return Err("Cochain complex must have at least one cochain group".to_string());
        }

        // Validate coboundary maps
        for (&deg, map) in &coboundary_maps {
            let current_rank = ranks.get(&deg).unwrap_or(&0);
            let next_rank = ranks.get(&(deg + 1)).unwrap_or(&0);

            if map.rows() != *next_rank {
                return Err(format!(
                    "Coboundary map δ^{} has {} rows, expected {}",
                    deg,
                    map.rows(),
                    next_rank
                ));
            }

            if map.cols() != *current_rank {
                return Err(format!(
                    "Coboundary map δ^{} has {} columns, expected {}",
                    deg,
                    map.cols(),
                    current_rank
                ));
            }
        }

        // Verify δ^{n+1} ∘ δ^n = 0 where possible
        for deg in min_degree..(max_degree - 1) {
            if let (Some(d_n), Some(d_next)) = (
                coboundary_maps.get(&deg),
                coboundary_maps.get(&(deg + 1)),
            ) {
                // Compute δ^{n+1} * δ^n
                let product = (d_next.clone() * d_n.clone())
                    .map_err(|e| format!("Failed to compose coboundary maps: {:?}", e))?;

                // Check if it's the zero matrix
                for i in 0..product.rows() {
                    for j in 0..product.cols() {
                        if let Ok(elem) = product.get(i, j) {
                            if *elem != Integer::from(0) {
                                return Err(format!(
                                    "Coboundary maps don't compose to zero at degree {}",
                                    deg
                                ));
                            }
                        }
                    }
                }
            }
        }

        Ok(CochainComplex {
            ranks,
            coboundary_maps,
            min_degree,
            max_degree,
        })
    }

    /// Create a trivial cochain complex (all zero groups)
    pub fn trivial() -> Self {
        CochainComplex {
            ranks: HashMap::new(),
            coboundary_maps: HashMap::new(),
            min_degree: 0,
            max_degree: 0,
        }
    }

    /// Get the rank of the cochain group at degree n
    pub fn rank(&self, n: i32) -> usize {
        *self.ranks.get(&n).unwrap_or(&0)
    }

    /// Get the coboundary map δ^n: C^n → C^{n+1}
    pub fn coboundary_map(&self, n: i32) -> Option<&Matrix<Integer>> {
        self.coboundary_maps.get(&n)
    }

    /// Get the minimum non-zero degree
    pub fn min_degree(&self) -> i32 {
        self.min_degree
    }

    /// Get the maximum non-zero degree
    pub fn max_degree(&self) -> i32 {
        self.max_degree
    }

    /// Compute the n-th cohomology group H^n = ker(δ^n) / im(δ^{n-1})
    pub fn cohomology(&self, n: i32) -> CohomologyGroup {
        // Get δ^n and δ^{n-1}
        let rank_n = self.rank(n);

        if rank_n == 0 {
            // If C^n is trivial, H^n is trivial
            return CohomologyGroup {
                degree: n,
                free_rank: 0,
                torsion: vec![],
            };
        }

        // Compute kernel of δ^n
        let kernel_gens = if let Some(d_n) = self.coboundary_map(n) {
            compute_kernel(d_n)
        } else {
            // If δ^n doesn't exist, ker(δ^n) = C^n (all of it)
            rank_n
        };

        // Compute image of δ^{n-1}
        let image_rank = if let Some(d_prev) = self.coboundary_map(n - 1) {
            compute_rank(d_prev)
        } else {
            0
        };

        // H^n has free rank = dim(ker(δ^n)) - dim(im(δ^{n-1}))
        let free_rank = if kernel_gens >= image_rank {
            kernel_gens - image_rank
        } else {
            0
        };

        CohomologyGroup {
            degree: n,
            free_rank,
            torsion: vec![], // Simplified - full torsion computation requires Smith normal form
        }
    }

    /// Compute all cohomology groups
    pub fn all_cohomology(&self) -> Vec<CohomologyGroup> {
        let mut result = Vec::new();

        for deg in self.min_degree..=self.max_degree {
            result.push(self.cohomology(deg));
        }

        result
    }

    /// Compute the Euler characteristic χ = Σ (-1)^n rank(C^n)
    pub fn euler_characteristic(&self) -> i64 {
        let mut chi = 0i64;

        for deg in self.min_degree..=self.max_degree {
            let rank = self.rank(deg) as i64;
            if deg % 2 == 0 {
                chi += rank;
            } else {
                chi -= rank;
            }
        }

        chi
    }
}

impl fmt::Display for CochainComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cochain Complex:")?;
        for deg in self.min_degree..=self.max_degree {
            let rank = self.rank(deg);
            write!(f, "  C^{} = Z^{}", deg, rank)?;
            if let Some(d) = self.coboundary_map(deg) {
                writeln!(f, " --δ^{}--> ({}×{} matrix)", deg, d.rows(), d.cols())?;
            } else {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// A cohomology group H^n
///
/// Represented as H^n ≅ Z^r ⊕ Z/t₁Z ⊕ ... ⊕ Z/tₖZ
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CohomologyGroup {
    /// Degree n
    pub degree: i32,
    /// Free rank r (number of Z factors)
    pub free_rank: usize,
    /// Torsion coefficients [t₁, t₂, ..., tₖ]
    pub torsion: Vec<usize>,
}

impl CohomologyGroup {
    /// Check if the cohomology group is trivial (zero)
    pub fn is_trivial(&self) -> bool {
        self.free_rank == 0 && self.torsion.is_empty()
    }

    /// Check if the cohomology group is free (no torsion)
    pub fn is_free(&self) -> bool {
        self.torsion.is_empty()
    }

    /// Get the rank (number of Z factors)
    pub fn rank(&self) -> usize {
        self.free_rank
    }

    /// Convert to an AbelianGroup
    pub fn to_abelian_group(&self) -> AbelianGroup {
        AbelianGroup::new(self.free_rank, self.torsion.clone())
            .unwrap_or_else(|_| AbelianGroup::free(self.free_rank))
    }

    /// Get a string representation
    pub fn structure_string(&self) -> String {
        if self.is_trivial() {
            return "0".to_string();
        }

        let mut parts = Vec::new();

        if self.free_rank > 0 {
            if self.free_rank == 1 {
                parts.push("Z".to_string());
            } else {
                parts.push(format!("Z^{}", self.free_rank));
            }
        }

        for &t in &self.torsion {
            parts.push(format!("Z/{}", t));
        }

        parts.join(" ⊕ ")
    }
}

impl fmt::Display for CohomologyGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H^{} = {}", self.degree, self.structure_string())
    }
}

/// Compute the rank (dimension of column space) of a matrix
fn compute_rank(mat: &Matrix<Integer>) -> usize {
    let rows = mat.rows();
    let cols = mat.cols();

    if rows == 0 || cols == 0 {
        return 0;
    }

    // Create a mutable copy for row reduction
    let mut temp = Vec::new();
    for i in 0..rows {
        let mut row = Vec::new();
        for j in 0..cols {
            row.push(mat.get(i, j).unwrap().clone());
        }
        temp.push(row);
    }

    let mut rank = 0;
    let mut col = 0;

    while rank < rows && col < cols {
        // Find pivot
        let mut pivot_row = rank;
        for r in (rank + 1)..rows {
            if temp[r][col].abs() > temp[pivot_row][col].abs() {
                pivot_row = r;
            }
        }

        if temp[pivot_row][col] == Integer::from(0) {
            col += 1;
            continue;
        }

        // Swap rows
        temp.swap(rank, pivot_row);

        // Eliminate below
        for r in (rank + 1)..rows {
            if temp[r][col] != Integer::from(0) {
                let factor = temp[r][col].clone();
                let pivot = temp[rank][col].clone();

                for c in col..cols {
                    let new_val = temp[r][c].clone() * pivot.clone()
                        - temp[rank][c].clone() * factor.clone();
                    temp[r][c] = new_val;
                }
            }
        }

        rank += 1;
        col += 1;
    }

    rank
}

/// Compute the dimension of the kernel (nullspace) of a matrix
fn compute_kernel(mat: &Matrix<Integer>) -> usize {
    let cols = mat.cols();
    let rank = compute_rank(mat);

    // By rank-nullity theorem: dim(ker) = dim(domain) - rank
    if cols >= rank {
        cols - rank
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_cochain_complex() {
        let complex = CochainComplex::trivial();
        assert_eq!(complex.rank(0), 0);
        assert_eq!(complex.rank(1), 0);
    }

    #[test]
    fn test_simple_cochain_complex() {
        // Create: 0 ← Z^3 <--δ0-- Z^2 ← 0
        // where δ0 = transpose of d1 from chain complex

        let mut ranks = HashMap::new();
        ranks.insert(0, 2);
        ranks.insert(1, 3);

        let mut coboundary_maps = HashMap::new();
        let d0 = Matrix::from_vec(
            3,
            2,
            vec![
                Integer::from(1),
                Integer::from(0),
                Integer::from(0),
                Integer::from(1),
                Integer::from(-1),
                Integer::from(-1),
            ],
        )
        .unwrap();
        coboundary_maps.insert(0, d0);

        let complex = CochainComplex::new(ranks, coboundary_maps);
        assert!(complex.is_ok());

        let complex = complex.unwrap();
        assert_eq!(complex.rank(0), 2);
        assert_eq!(complex.rank(1), 3);
    }

    #[test]
    fn test_cohomology_of_circle() {
        // Dual of circle chain complex
        // C^0 = Z, C^1 = Z, δ^0 = [0] (transpose of boundary map)

        let mut ranks = HashMap::new();
        ranks.insert(0, 1);
        ranks.insert(1, 1);

        let mut coboundary_maps = HashMap::new();
        let d0 = Matrix::from_vec(1, 1, vec![Integer::from(0)]).unwrap();
        coboundary_maps.insert(0, d0);

        let complex = CochainComplex::new(ranks, coboundary_maps).unwrap();

        // H^0 should be Z (connected)
        let h0 = complex.cohomology(0);
        assert_eq!(h0.free_rank, 1);

        // H^1 should be Z (one loop)
        let h1 = complex.cohomology(1);
        assert_eq!(h1.free_rank, 1);
    }

    #[test]
    fn test_cohomology_group_trivial() {
        let h = CohomologyGroup {
            degree: 0,
            free_rank: 0,
            torsion: vec![],
        };

        assert!(h.is_trivial());
        assert!(h.is_free());
        assert_eq!(h.structure_string(), "0");
    }

    #[test]
    fn test_cohomology_group_free() {
        let h = CohomologyGroup {
            degree: 1,
            free_rank: 2,
            torsion: vec![],
        };

        assert!(!h.is_trivial());
        assert!(h.is_free());
        assert_eq!(h.rank(), 2);
        assert_eq!(h.structure_string(), "Z^2");
    }

    #[test]
    fn test_cohomology_group_with_torsion() {
        let h = CohomologyGroup {
            degree: 2,
            free_rank: 1,
            torsion: vec![2, 3],
        };

        assert!(!h.is_trivial());
        assert!(!h.is_free());
        assert_eq!(h.structure_string(), "Z ⊕ Z/2 ⊕ Z/3");
    }

    #[test]
    fn test_euler_characteristic() {
        // Point: C^0 = Z
        let mut ranks = HashMap::new();
        ranks.insert(0, 1);
        let coboundary_maps = HashMap::new();

        let complex = CochainComplex::new(ranks, coboundary_maps).unwrap();
        assert_eq!(complex.euler_characteristic(), 1);
    }

    #[test]
    fn test_cohomology_to_abelian_group() {
        let h = CohomologyGroup {
            degree: 1,
            free_rank: 2,
            torsion: vec![],
        };

        let g = h.to_abelian_group();
        assert_eq!(g.free_rank(), 2);
        assert_eq!(g.torsion_rank(), 0);
    }

    #[test]
    fn test_display_cochain_complex() {
        let mut ranks = HashMap::new();
        ranks.insert(0, 2);
        ranks.insert(1, 1);

        let coboundary_maps = HashMap::new();
        let complex = CochainComplex::new(ranks, coboundary_maps).unwrap();

        let display = format!("{}", complex);
        assert!(display.contains("Cochain Complex"));
        assert!(display.contains("C^0"));
    }

    #[test]
    fn test_coboundary_composition_validation() {
        // Create an invalid complex where δ^{n+1} ∘ δ^n ≠ 0
        let mut ranks = HashMap::new();
        ranks.insert(0, 1);
        ranks.insert(1, 2);
        ranks.insert(2, 2);

        let mut coboundary_maps = HashMap::new();

        // δ^0: Z → Z^2, [1; 1]
        let d0 = Matrix::from_vec(2, 1, vec![Integer::from(1), Integer::from(1)]).unwrap();
        coboundary_maps.insert(0, d0);

        // δ^1: Z^2 → Z^2, identity
        let d1 = Matrix::from_vec(
            2,
            2,
            vec![
                Integer::from(1),
                Integer::from(0),
                Integer::from(0),
                Integer::from(1),
            ],
        )
        .unwrap();
        coboundary_maps.insert(1, d1);

        // δ^1 ∘ δ^0 = [1; 1], which is not zero
        // This should fail validation
        let complex = CochainComplex::new(ranks, coboundary_maps);
        assert!(complex.is_err());
    }
}
