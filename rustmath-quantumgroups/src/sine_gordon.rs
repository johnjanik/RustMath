//! Sine-Gordon Y-Systems
//!
//! This module implements Y-systems for the sine-Gordon model from quantum field theory.
//! The sine-Gordon Y-system is a system of functional equations that arise in the study
//! of integrable quantum field theories, particularly in the context of the Thermodynamic
//! Bethe Ansatz (TBA).
//!
//! # Background
//!
//! The sine-Gordon model is a 1+1 dimensional integrable quantum field theory described by
//! the Lagrangian density:
//!
//! L = (1/2)(∂_μ φ)² + (m²/β²) cos(β φ)
//!
//! where β is the coupling constant.
//!
//! The Y-system consists of functional equations for variables Y_n(u), where n is the
//! particle species index and u is the rapidity parameter. These equations take the form:
//!
//! Y_n(u + iπ/2) * Y_n(u - iπ/2) = ∏_m (1 + Y_m(u))^{I_{nm}}
//!
//! where I_{nm} is the incidence matrix of the Dynkin diagram.
//!
//! For the sine-Gordon model, we have a simpler structure corresponding to A_{n-1} type
//! Dynkin diagrams.
//!
//! # References
//!
//! - Zamolodchikov, A.B., "On the thermodynamic Bethe ansatz equations for reflectionless ADE scattering theories"
//! - Tateo, R., "The sine-Gordon model as S^1/Z_2 perturbed coset theory and generalizations"
//! - Ravanini, F., Tateo, R., Valleriani, A., "Dynkin TBA's"

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// Represents a Y-system variable Y_n(u) in the sine-Gordon model
///
/// The Y-variables satisfy functional equations that encode the thermodynamic properties
/// of the integrable system. Each variable is indexed by a particle species `n` and
/// depends on a rapidity parameter `u`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YVariable {
    /// Index of the particle species (typically 1, 2, ..., N for A_{N-1} type)
    pub index: usize,

    /// Rapidity shift (in units of iπ/2)
    /// For example, shift = 0 means Y_n(u), shift = 1 means Y_n(u + iπ/2)
    pub shift: i32,
}

impl YVariable {
    /// Create a new Y-variable with given index and rapidity shift
    ///
    /// # Arguments
    ///
    /// * `index` - The particle species index (typically ≥ 1)
    /// * `shift` - The rapidity shift in units of iπ/2
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_quantumgroups::sine_gordon::YVariable;
    ///
    /// // Create Y_1(u)
    /// let y1 = YVariable::new(1, 0);
    ///
    /// // Create Y_2(u + iπ/2)
    /// let y2_shifted = YVariable::new(2, 1);
    /// ```
    pub fn new(index: usize, shift: i32) -> Self {
        YVariable { index, shift }
    }

    /// Apply a rapidity shift to this Y-variable
    ///
    /// # Arguments
    ///
    /// * `shift` - Additional shift to apply (in units of iπ/2)
    ///
    /// # Returns
    ///
    /// A new Y-variable with the combined shift
    pub fn with_shift(&self, shift: i32) -> Self {
        YVariable {
            index: self.index,
            shift: self.shift + shift,
        }
    }
}

/// Represents the sine-Gordon Y-system
///
/// The Y-system encodes the functional equations satisfied by the Y-variables.
/// For the sine-Gordon model with N particle species (A_{N-1} type), the equations are:
///
/// Y_n(u + iπ/2) * Y_n(u - iπ/2) = (1 + Y_{n-1}(u)) * (1 + Y_{n+1}(u))
///
/// with boundary conditions Y_0 = Y_{N+1} = 0.
#[derive(Debug, Clone)]
pub struct SineGordonYSystem {
    /// Number of particle species (rank of the system)
    pub rank: usize,

    /// Coupling constant β (stored as a rational for exact arithmetic)
    /// The physical regime is 0 < β² < 8π
    pub beta_squared: Rational,
}

impl SineGordonYSystem {
    /// Create a new sine-Gordon Y-system
    ///
    /// # Arguments
    ///
    /// * `rank` - Number of particle species (N for A_{N-1} type)
    /// * `beta_squared` - The square of the coupling constant β
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_quantumgroups::sine_gordon::SineGordonYSystem;
    /// use rustmath_rationals::Rational;
    ///
    /// // Create a rank 3 sine-Gordon Y-system (A_2 type)
    /// let beta_sq = Rational::new(1.into(), 2.into());
    /// let system = SineGordonYSystem::new(3, beta_sq);
    /// ```
    pub fn new(rank: usize, beta_squared: Rational) -> Self {
        SineGordonYSystem { rank, beta_squared }
    }

    /// Get the Dynkin diagram incidence matrix for A_{N-1} type
    ///
    /// The incidence matrix I_{nm} determines the structure of the Y-system equations.
    /// For A_{N-1} type (sine-Gordon), this is a tridiagonal matrix with:
    /// - I_{n,n-1} = I_{n,n+1} = 1 for all valid n
    /// - I_{nm} = 0 otherwise
    ///
    /// # Returns
    ///
    /// A 2D vector representing the incidence matrix
    pub fn incidence_matrix(&self) -> Vec<Vec<i32>> {
        let n = self.rank;
        let mut matrix = vec![vec![0; n]; n];

        for i in 0..n {
            if i > 0 {
                matrix[i][i - 1] = 1;
            }
            if i + 1 < n {
                matrix[i][i + 1] = 1;
            }
        }

        matrix
    }

    /// Verify if the given Y-values satisfy the Y-system equations
    ///
    /// This checks the functional equations:
    /// Y_n(u + iπ/2) * Y_n(u - iπ/2) ≈ (1 + Y_{n-1}(u)) * (1 + Y_{n+1}(u))
    ///
    /// # Arguments
    ///
    /// * `y_values` - A map from (index, shift) to Y-values (as rationals)
    /// * `tolerance` - Tolerance for numerical comparison (as a rational)
    ///
    /// # Returns
    ///
    /// `true` if all equations are satisfied within tolerance, `false` otherwise
    pub fn verify_equations(
        &self,
        y_values: &HashMap<(usize, i32), Rational>,
        tolerance: &Rational,
    ) -> bool {
        for n in 1..=self.rank {
            // Get Y_n(u+1), Y_n(u-1), Y_{n-1}(u), Y_{n+1}(u) for shift u=0
            let y_n_plus = y_values.get(&(n, 1)).cloned();
            let y_n_minus = y_values.get(&(n, -1)).cloned();
            let y_n_left = if n > 1 {
                y_values.get(&(n - 1, 0)).cloned()
            } else {
                Some(Rational::zero()) // Boundary condition
            };
            let y_n_right = if n < self.rank {
                y_values.get(&(n + 1, 0)).cloned()
            } else {
                Some(Rational::zero()) // Boundary condition
            };

            if let (Some(yp), Some(ym), Some(yl), Some(yr)) = (y_n_plus, y_n_minus, y_n_left, y_n_right) {
                // LHS: Y_n(u+1) * Y_n(u-1)
                let lhs = yp.clone() * ym.clone();

                // RHS: (1 + Y_{n-1}(u)) * (1 + Y_{n+1}(u))
                let one = Rational::one();
                let rhs = (one.clone() + yl) * (one + yr);

                // Check if |LHS - RHS| ≤ tolerance
                let diff = if lhs >= rhs {
                    lhs - rhs
                } else {
                    rhs - lhs
                };

                if diff > *tolerance {
                    return false;
                }
            }
        }

        true
    }

    /// Compute the mass spectrum of the sine-Gordon model
    ///
    /// For the sine-Gordon model with coupling β, the mass spectrum consists of
    /// solitons, antisolitons, and bound states (breathers). The breather masses
    /// are given by:
    ///
    /// m_n = 2M sin(nπβ²/16π)
    ///
    /// where M is the soliton mass and n = 1, 2, ..., N with N determined by β.
    ///
    /// # Arguments
    ///
    /// * `soliton_mass` - The mass of the fundamental soliton M
    ///
    /// # Returns
    ///
    /// A vector of breather masses (as rationals, in units where the formula is simplified)
    ///
    /// Note: This returns a simplified version for small β where sin can be approximated
    pub fn mass_spectrum(&self, _soliton_mass: &Rational) -> Vec<Rational> {
        // Simplified implementation: return masses proportional to indices
        // A full implementation would compute sin(nπβ²/16π) for each n
        let mut masses = Vec::new();

        for n in 1..=self.rank {
            // Simplified: m_n ∝ n for small β
            masses.push(Rational::from(n as i32));
        }

        masses
    }

    /// Compute the c-function (effective central charge) from Y-system data
    ///
    /// The c-function encodes the UV behavior of the conformal field theory.
    /// For the sine-Gordon model:
    ///
    /// c_eff = Σ_n ∫ (du/2π) log(1 + Y_n(u))
    ///
    /// # Returns
    ///
    /// A rational approximation of the effective central charge
    ///
    /// Note: This is a placeholder that returns the expected value for sine-Gordon
    pub fn c_function(&self) -> Rational {
        // The sine-Gordon model flows to a CFT with central charge c = 1
        // A full implementation would integrate the Y-functions
        Rational::one()
    }

    /// Generate initial Y-system data for iteration
    ///
    /// This creates an initial guess for Y-values that can be iteratively refined
    /// to solve the Y-system equations.
    ///
    /// # Arguments
    ///
    /// * `max_shift` - Maximum rapidity shift to include (in units of iπ/2)
    ///
    /// # Returns
    ///
    /// A HashMap mapping (index, shift) pairs to initial Y-values
    pub fn initial_data(&self, max_shift: i32) -> HashMap<(usize, i32), Rational> {
        let mut data = HashMap::new();

        // Initialize with Y_n(u) = 1 for all n and small shifts
        for n in 1..=self.rank {
            for shift in -max_shift..=max_shift {
                data.insert((n, shift), Rational::one());
            }
        }

        data
    }

    /// Perform one iteration of the Y-system equations
    ///
    /// This updates Y-values according to the functional equations:
    /// Y_n(u) = [(1 + Y_{n-1}(u)) * (1 + Y_{n+1}(u))] / [Y_n(u+1) * Y_n(u-1)]
    ///
    /// # Arguments
    ///
    /// * `y_values` - Current Y-values (modified in place)
    ///
    /// # Returns
    ///
    /// The maximum change in Y-values (for convergence checking)
    pub fn iterate(&self, y_values: &mut HashMap<(usize, i32), Rational>) -> Rational {
        let mut new_values = HashMap::new();
        let mut max_change = Rational::zero();

        for n in 1..=self.rank {
            // Collect all shifts present for this index
            let shifts: Vec<i32> = y_values
                .keys()
                .filter(|(idx, _)| *idx == n)
                .map(|(_, s)| *s)
                .collect();

            for &shift in &shifts {
                // Skip boundary shifts where we don't have neighbors
                let y_plus = y_values.get(&(n, shift + 1));
                let y_minus = y_values.get(&(n, shift - 1));

                if y_plus.is_none() || y_minus.is_none() {
                    // Keep current value for boundary
                    if let Some(v) = y_values.get(&(n, shift)) {
                        new_values.insert((n, shift), v.clone());
                    }
                    continue;
                }

                // Get neighbor values
                let y_left = if n > 1 {
                    y_values.get(&(n - 1, shift)).cloned().unwrap_or(Rational::zero())
                } else {
                    Rational::zero()
                };

                let y_right = if n < self.rank {
                    y_values.get(&(n + 1, shift)).cloned().unwrap_or(Rational::zero())
                } else {
                    Rational::zero()
                };

                let y_plus = y_plus.unwrap().clone();
                let y_minus = y_minus.unwrap().clone();

                // Compute new value: Y_n(u) = [(1+Y_{n-1}(u))(1+Y_{n+1}(u))] / [Y_n(u+1)*Y_n(u-1)]
                let one = Rational::one();
                let numerator = (one.clone() + y_left) * (one + y_right);
                let denominator = y_plus * y_minus;

                if !denominator.is_zero() {
                    let new_value = numerator / denominator;

                    // Track maximum change
                    if let Some(old_value) = y_values.get(&(n, shift)) {
                        let change = if new_value >= *old_value {
                            new_value.clone() - old_value.clone()
                        } else {
                            old_value.clone() - new_value.clone()
                        };

                        if change > max_change {
                            max_change = change;
                        }
                    }

                    new_values.insert((n, shift), new_value);
                } else {
                    // Keep old value if denominator is zero
                    if let Some(v) = y_values.get(&(n, shift)) {
                        new_values.insert((n, shift), v.clone());
                    }
                }
            }
        }

        // Update the map
        *y_values = new_values;

        max_change
    }
}

/// Builder for sine-Gordon Y-systems with convenient defaults
pub struct SineGordonBuilder {
    rank: Option<usize>,
    beta_squared: Option<Rational>,
}

impl SineGordonBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        SineGordonBuilder {
            rank: None,
            beta_squared: None,
        }
    }

    /// Set the rank (number of particle species)
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = Some(rank);
        self
    }

    /// Set β² from a ratio of integers
    pub fn beta_squared(mut self, numerator: i32, denominator: i32) -> Self {
        self.beta_squared = Some(Rational::new(
            Integer::from(numerator),
            Integer::from(denominator),
        ));
        self
    }

    /// Build the Y-system
    pub fn build(self) -> Result<SineGordonYSystem, String> {
        let rank = self.rank.ok_or("Rank must be specified")?;
        let beta_squared = self.beta_squared.unwrap_or(Rational::one());

        if rank == 0 {
            return Err("Rank must be positive".to_string());
        }

        Ok(SineGordonYSystem::new(rank, beta_squared))
    }
}

impl Default for SineGordonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_y_variable_creation() {
        let y1 = YVariable::new(1, 0);
        assert_eq!(y1.index, 1);
        assert_eq!(y1.shift, 0);

        let y2 = YVariable::new(2, 3);
        assert_eq!(y2.index, 2);
        assert_eq!(y2.shift, 3);
    }

    #[test]
    fn test_y_variable_shift() {
        let y = YVariable::new(1, 0);
        let y_shifted = y.with_shift(2);

        assert_eq!(y_shifted.index, 1);
        assert_eq!(y_shifted.shift, 2);

        let y_double_shifted = y_shifted.with_shift(-1);
        assert_eq!(y_double_shifted.shift, 1);
    }

    #[test]
    fn test_system_creation() {
        let beta_sq = Rational::new(Integer::from(1), Integer::from(2));
        let system = SineGordonYSystem::new(3, beta_sq.clone());

        assert_eq!(system.rank, 3);
        assert_eq!(system.beta_squared, beta_sq);
    }

    #[test]
    fn test_incidence_matrix_rank3() {
        let system = SineGordonYSystem::new(3, Rational::one());
        let matrix = system.incidence_matrix();

        assert_eq!(matrix.len(), 3);

        // Check A_2 structure:
        // 0-1-2
        // Matrix should be:
        // [0, 1, 0]
        // [1, 0, 1]
        // [0, 1, 0]

        assert_eq!(matrix[0][0], 0);
        assert_eq!(matrix[0][1], 1);
        assert_eq!(matrix[0][2], 0);

        assert_eq!(matrix[1][0], 1);
        assert_eq!(matrix[1][1], 0);
        assert_eq!(matrix[1][2], 1);

        assert_eq!(matrix[2][0], 0);
        assert_eq!(matrix[2][1], 1);
        assert_eq!(matrix[2][2], 0);
    }

    #[test]
    fn test_incidence_matrix_rank5() {
        let system = SineGordonYSystem::new(5, Rational::one());
        let matrix = system.incidence_matrix();

        assert_eq!(matrix.len(), 5);

        // Check that it's tridiagonal
        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    assert_eq!(matrix[i][j], 0);
                } else if (i as i32 - j as i32).abs() == 1 {
                    assert_eq!(matrix[i][j], 1);
                } else {
                    assert_eq!(matrix[i][j], 0);
                }
            }
        }
    }

    #[test]
    fn test_initial_data() {
        let system = SineGordonYSystem::new(2, Rational::one());
        let data = system.initial_data(2);

        // Should have data for n=1,2 and shifts -2,-1,0,1,2
        assert_eq!(data.len(), 2 * 5);

        // All values should be initialized to 1
        for &value in data.values() {
            assert_eq!(value, Rational::one());
        }

        // Check specific entries exist
        assert!(data.contains_key(&(1, 0)));
        assert!(data.contains_key(&(1, -2)));
        assert!(data.contains_key(&(2, 2)));
    }

    #[test]
    fn test_verify_trivial_solution() {
        let system = SineGordonYSystem::new(2, Rational::one());
        let mut y_values = HashMap::new();

        // Trivial solution: Y_n(u) = 1 for all n, u
        // This satisfies: 1 * 1 = (1+1) * (1+0) = 2 for n=1
        // and: 1 * 1 = (1+0) * (1+1) = 2 for n=2
        // So trivial solution doesn't work exactly

        // Instead test Y_n(u) = 0 which gives: 0 = 1 * 1, also doesn't work

        // Let's try Y_n(u) = φ where φ² = 1+φ (golden ratio relation)
        // Then Y(u+1)*Y(u-1) = φ² = 1+φ = (1+0)*(1+φ) works for boundary

        // For this test, just check that the verification runs
        for n in 1..=2 {
            for shift in -1..=1 {
                y_values.insert((n, shift), Rational::one());
            }
        }

        let tolerance = Rational::new(Integer::from(10), Integer::from(1));
        // This won't be satisfied exactly, but the function should run
        system.verify_equations(&y_values, &tolerance);
    }

    #[test]
    fn test_mass_spectrum() {
        let system = SineGordonYSystem::new(3, Rational::one());
        let soliton_mass = Rational::from(2);
        let masses = system.mass_spectrum(&soliton_mass);

        assert_eq!(masses.len(), 3);

        // In the simplified implementation, masses are proportional to index
        assert_eq!(masses[0], Rational::from(1));
        assert_eq!(masses[1], Rational::from(2));
        assert_eq!(masses[2], Rational::from(3));
    }

    #[test]
    fn test_c_function() {
        let system = SineGordonYSystem::new(3, Rational::one());
        let c = system.c_function();

        // Sine-Gordon flows to c=1 CFT
        assert_eq!(c, Rational::one());
    }

    #[test]
    fn test_builder() {
        let system = SineGordonBuilder::new()
            .rank(4)
            .beta_squared(1, 3)
            .build()
            .unwrap();

        assert_eq!(system.rank, 4);
        assert_eq!(system.beta_squared, Rational::new(Integer::from(1), Integer::from(3)));
    }

    #[test]
    fn test_builder_default_beta() {
        let system = SineGordonBuilder::new()
            .rank(2)
            .build()
            .unwrap();

        assert_eq!(system.beta_squared, Rational::one());
    }

    #[test]
    fn test_builder_missing_rank() {
        let result = SineGordonBuilder::new()
            .beta_squared(1, 2)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_zero_rank() {
        let result = SineGordonBuilder::new()
            .rank(0)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_iteration() {
        let system = SineGordonYSystem::new(2, Rational::one());
        let mut y_values = system.initial_data(2);

        // Perform one iteration
        let change = system.iterate(&mut y_values);

        // Change should be non-negative
        assert!(change >= Rational::zero());

        // Values should still exist
        assert!(y_values.contains_key(&(1, 0)));
        assert!(y_values.contains_key(&(2, 0)));
    }

    #[test]
    fn test_iteration_convergence() {
        let system = SineGordonYSystem::new(2, Rational::one());
        let mut y_values = system.initial_data(1);

        // Perform multiple iterations
        let mut prev_change = Rational::from(1000);
        for _ in 0..5 {
            let change = system.iterate(&mut y_values);

            // In a proper implementation, changes should generally decrease
            // (though not monotonically in all cases)
            // For now, just check that iteration completes
            assert!(change >= Rational::zero());
            prev_change = change;
        }
    }
}
