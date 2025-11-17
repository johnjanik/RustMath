//! Lie algebras - tangent spaces of Lie groups at the identity
//!
//! A Lie algebra is a vector space 𝔤 equipped with a bilinear operation [·,·]: 𝔤 × 𝔤 → 𝔤
//! called the Lie bracket, which is:
//! - Antisymmetric: [X, Y] = -[Y, X]
//! - Satisfies the Jacobi identity: [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0

use crate::errors::{ManifoldError, Result};
use crate::lie_group::LieGroup;
use crate::tangent_space::{TangentVector, TangentSpace};
use crate::point::ManifoldPoint;
use crate::chart::Chart;
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;
use rustmath_symbolic::Expr;
use std::sync::Arc;
use std::collections::HashMap;

/// A Lie algebra - tangent space of a Lie group at the identity
///
/// The Lie algebra 𝔤 = T_e G is the tangent space at the identity element,
/// equipped with the Lie bracket operation.
///
/// # Examples
///
/// ```
/// // so(3) - Lie algebra of SO(3) (3x3 skew-symmetric matrices)
/// // su(2) - Lie algebra of SU(2) (2x2 skew-Hermitian traceless matrices)
/// // gl(n) - Lie algebra of GL(n) (all n×n matrices)
/// ```
#[derive(Clone)]
pub struct LieAlgebra {
    /// The associated Lie group (if known)
    group: Option<Arc<LieGroup>>,

    /// Dimension of the Lie algebra
    dimension: usize,

    /// Basis elements of the Lie algebra
    /// Each element is a tangent vector at the identity
    basis: Vec<TangentVector>,

    /// Structure constants: [e_i, e_j] = Σ_k c^k_{ij} e_k
    /// Stored as structure_constants[i][j][k] = c^k_{ij}
    structure_constants: Vec<Vec<Vec<f64>>>,

    /// Name of the Lie algebra
    name: String,
}

impl LieAlgebra {
    /// Create a new Lie algebra from structure constants
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of the algebra
    /// * `structure_constants` - The structure constants c^k_{ij}
    /// * `name` - Name of the algebra (e.g., "so(3)", "su(2)")
    pub fn new(
        dimension: usize,
        structure_constants: Vec<Vec<Vec<f64>>>,
        name: String,
    ) -> Result<Self> {
        // Verify structure constants have correct dimensions
        if structure_constants.len() != dimension {
            return Err(ManifoldError::InvalidDimension(format!(
                "Structure constants outer dimension {} does not match Lie algebra dimension {}",
                structure_constants.len(), dimension
            )));
        }

        for i in 0..dimension {
            if structure_constants[i].len() != dimension {
                return Err(ManifoldError::InvalidDimension(format!(
                    "Structure constants middle dimension {} does not match Lie algebra dimension {}",
                    structure_constants[i].len(), dimension
                )));
            }
            for j in 0..dimension {
                if structure_constants[i][j].len() != dimension {
                    return Err(ManifoldError::InvalidDimension(format!(
                        "Structure constants inner dimension {} does not match Lie algebra dimension {}",
                        structure_constants[i][j].len(), dimension
                    )));
                }
            }
        }

        // Verify antisymmetry: c^k_{ij} = -c^k_{ji}
        for i in 0..dimension {
            for j in 0..dimension {
                for k in 0..dimension {
                    let cijk = structure_constants[i][j][k];
                    let cjik = structure_constants[j][i][k];
                    if (cijk + cjik).abs() > 1e-10 {
                        return Err(ManifoldError::InvalidStructure(
                            "Structure constants are not antisymmetric".to_string()
                        ));
                    }
                }
            }
        }

        // For now, create placeholder basis vectors
        // In a full implementation, these would be actual tangent vectors
        let basis = vec![]; // Will be populated when associated with a group

        Ok(Self {
            group: None,
            dimension,
            basis,
            structure_constants,
            name,
        })
    }

    /// Create a Lie algebra from a Lie group
    ///
    /// The Lie algebra is T_e G with the Lie bracket
    pub fn from_group(group: Arc<LieGroup>) -> Result<Self> {
        let dimension = group.dimension();

        // In a full implementation, we'd compute structure constants from the group
        // For now, create a placeholder with zero structure constants
        let structure_constants = vec![vec![vec![0.0; dimension]; dimension]; dimension];

        Ok(Self {
            group: Some(group.clone()),
            dimension,
            basis: vec![],
            structure_constants,
            name: "Lie algebra".to_string(),
        })
    }

    /// Get the dimension of the Lie algebra
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the name of the Lie algebra
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Lie bracket: [X, Y]
    ///
    /// Given elements X = Σ x^i e_i and Y = Σ y^j e_j,
    /// compute [X, Y] = Σ_k (Σ_{i,j} c^k_{ij} x^i y^j) e_k
    pub fn bracket(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.dimension || y.len() != self.dimension {
            return Err(ManifoldError::InvalidDimension(format!("Bracket dimension mismatch: expected {}, got x:{} y:{}", self.dimension, x.len(), y.len())));
        }

        let mut result = vec![0.0; self.dimension];

        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    result[k] += self.structure_constants[i][j][k] * x[i] * y[j];
                }
            }
        }

        Ok(result)
    }

    /// Verify Jacobi identity: [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0
    pub fn verify_jacobi_identity(&self) -> bool {
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for l in 0..self.dimension {
                    let mut sum = 0.0;

                    // [e_i, [e_j, e_l]] + [e_j, [e_l, e_i]] + [e_l, [e_i, e_j]]
                    for k in 0..self.dimension {
                        for m in 0..self.dimension {
                            sum += self.structure_constants[i][j][k]
                                * self.structure_constants[k][l][m];

                            sum += self.structure_constants[j][l][k]
                                * self.structure_constants[k][i][m];

                            sum += self.structure_constants[l][i][k]
                                * self.structure_constants[k][j][m];
                        }
                    }

                    if sum.abs() > 1e-10 {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Get structure constant c^k_{ij}
    pub fn structure_constant(&self, i: usize, j: usize, k: usize) -> Result<f64> {
        if i >= self.dimension || j >= self.dimension || k >= self.dimension {
            return Err(ManifoldError::InvalidIndex(
                format!("Structure constant indices ({},{},{}) out of range for dimension {}",
                        i, j, k, self.dimension)
            ));
        }

        Ok(self.structure_constants[i][j][k])
    }

    /// Compute the Killing form: B(X, Y) = tr(ad_X ∘ ad_Y)
    ///
    /// where ad_X(Y) = [X, Y] is the adjoint representation
    pub fn killing_form(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != self.dimension || y.len() != self.dimension {
            return Err(ManifoldError::InvalidDimension(format!("Killing form dimension mismatch: expected {}, got x:{} y:{}", self.dimension, x.len(), y.len())));
        }

        // Compute matrices of ad_X and ad_Y
        let mut ad_x = vec![vec![0.0; self.dimension]; self.dimension];
        let mut ad_y = vec![vec![0.0; self.dimension]; self.dimension];

        for i in 0..self.dimension {
            for j in 0..self.dimension {
                // (ad_X)^k_j = Σ_i c^k_{ij} x^i
                for k in 0..self.dimension {
                    ad_x[k][j] += self.structure_constants[i][j][k] * x[i];
                    ad_y[k][j] += self.structure_constants[i][j][k] * y[i];
                }
            }
        }

        // Compute trace of ad_X ∘ ad_Y
        let mut trace = 0.0;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                trace += ad_x[i][j] * ad_y[j][i];
            }
        }

        Ok(trace)
    }

    /// Check if the Lie algebra is abelian (all brackets vanish)
    pub fn is_abelian(&self) -> bool {
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    if self.structure_constants[i][j][k].abs() > 1e-10 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Get the associated Lie group (if known)
    pub fn group(&self) -> Option<&Arc<LieGroup>> {
        self.group.as_ref()
    }
}

/// Exponential map: exp: 𝔤 → G
///
/// Maps elements of the Lie algebra to the Lie group via the exponential map
/// For matrix Lie groups: exp(X) = e^X = Σ_{n=0}^∞ X^n/n!
pub struct ExponentialMap {
    /// The Lie group
    group: Arc<LieGroup>,

    /// The Lie algebra
    algebra: Arc<LieAlgebra>,

    /// Implementation of the exponential map
    exp_impl: Arc<dyn Fn(&[f64]) -> Result<ManifoldPoint> + Send + Sync>,
}

impl ExponentialMap {
    /// Create an exponential map
    ///
    /// # Arguments
    ///
    /// * `group` - The Lie group
    /// * `algebra` - The Lie algebra
    /// * `exp_impl` - Function that computes exp(X) for X ∈ 𝔤
    pub fn new(
        group: Arc<LieGroup>,
        algebra: Arc<LieAlgebra>,
        exp_impl: Arc<dyn Fn(&[f64]) -> Result<ManifoldPoint> + Send + Sync>,
    ) -> Self {
        Self {
            group,
            algebra,
            exp_impl,
        }
    }

    /// Apply the exponential map: exp: 𝔤 → G
    ///
    /// # Arguments
    ///
    /// * `x` - Element of the Lie algebra (as coefficients in basis)
    ///
    /// # Returns
    ///
    /// The corresponding group element exp(X)
    pub fn exp(&self, x: &[f64]) -> Result<ManifoldPoint> {
        if x.len() != self.algebra.dimension() {
            return Err(ManifoldError::InvalidDimension(format!("Exp dimension mismatch: expected {}, got {}", self.algebra.dimension(), x.len())));
        }

        (self.exp_impl)(x)
    }

    /// Compute a one-parameter subgroup: γ(t) = exp(tX)
    ///
    /// # Arguments
    ///
    /// * `x` - Element of the Lie algebra
    /// * `t` - Parameter value
    pub fn one_parameter_subgroup(&self, x: &[f64], t: f64) -> Result<ManifoldPoint> {
        let tx: Vec<f64> = x.iter().map(|xi| xi * t).collect();
        self.exp(&tx)
    }

    /// Get the Lie group
    pub fn group(&self) -> &Arc<LieGroup> {
        &self.group
    }

    /// Get the Lie algebra
    pub fn algebra(&self) -> &Arc<LieAlgebra> {
        &self.algebra
    }
}

/// Pre-defined Lie algebras

impl LieAlgebra {
    /// Create the abelian Lie algebra of dimension n
    ///
    /// All structure constants are zero: [e_i, e_j] = 0
    pub fn abelian(n: usize) -> Self {
        let structure_constants = vec![vec![vec![0.0; n]; n]; n];

        Self {
            group: None,
            dimension: n,
            basis: vec![],
            structure_constants,
            name: format!("ℝ^{}", n),
        }
    }

    /// Create so(3) - Lie algebra of SO(3) rotations
    ///
    /// Basis: { L_x, L_y, L_z } with [L_i, L_j] = ε_{ijk} L_k
    pub fn so3() -> Self {
        let mut structure_constants = vec![vec![vec![0.0; 3]; 3]; 3];

        // [L_x, L_y] = L_z (structure constant c^2_{01} = 1)
        structure_constants[0][1][2] = 1.0;
        structure_constants[1][0][2] = -1.0;

        // [L_y, L_z] = L_x
        structure_constants[1][2][0] = 1.0;
        structure_constants[2][1][0] = -1.0;

        // [L_z, L_x] = L_y
        structure_constants[2][0][1] = 1.0;
        structure_constants[0][2][1] = -1.0;

        Self {
            group: None,
            dimension: 3,
            basis: vec![],
            structure_constants,
            name: "so(3)".to_string(),
        }
    }

    /// Create su(2) - Lie algebra of SU(2)
    ///
    /// Isomorphic to so(3) with different normalization
    pub fn su2() -> Self {
        let mut algebra = Self::so3();
        algebra.name = "su(2)".to_string();

        // Scale structure constants by factor of 2
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    algebra.structure_constants[i][j][k] *= 2.0;
                }
            }
        }

        algebra
    }

    /// Create sl(2, ℝ) - Lie algebra of SL(2, ℝ)
    ///
    /// Basis: {H, X, Y} with [H,X] = 2X, [H,Y] = -2Y, [X,Y] = H
    pub fn sl2_real() -> Self {
        let mut structure_constants = vec![vec![vec![0.0; 3]; 3]; 3];

        // [H, X] = 2X (indices: 0, 1 -> 1)
        structure_constants[0][1][1] = 2.0;
        structure_constants[1][0][1] = -2.0;

        // [H, Y] = -2Y (indices: 0, 2 -> 2)
        structure_constants[0][2][2] = -2.0;
        structure_constants[2][0][2] = 2.0;

        // [X, Y] = H (indices: 1, 2 -> 0)
        structure_constants[1][2][0] = 1.0;
        structure_constants[2][1][0] = -1.0;

        Self {
            group: None,
            dimension: 3,
            basis: vec![],
            structure_constants,
            name: "sl(2,ℝ)".to_string(),
        }
    }

    /// Create the Heisenberg Lie algebra
    ///
    /// 3-dimensional nilpotent algebra with basis {X, Y, Z}
    /// where [X, Y] = Z and all other brackets vanish
    pub fn heisenberg() -> Self {
        let mut structure_constants = vec![vec![vec![0.0; 3]; 3]; 3];

        // [X, Y] = Z (indices: 0, 1 -> 2)
        structure_constants[0][1][2] = 1.0;
        structure_constants[1][0][2] = -1.0;

        Self {
            group: None,
            dimension: 3,
            basis: vec![],
            structure_constants,
            name: "Heisenberg".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abelian_algebra() {
        let algebra = LieAlgebra::abelian(3);

        assert_eq!(algebra.dimension(), 3);
        assert!(algebra.is_abelian());
        assert!(algebra.verify_jacobi_identity());

        // All brackets should be zero
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let bracket = algebra.bracket(&x, &y).unwrap();
        for i in 0..3 {
            assert_eq!(bracket[i], 0.0);
        }
    }

    #[test]
    fn test_so3_algebra() {
        let algebra = LieAlgebra::so3();

        assert_eq!(algebra.dimension(), 3);
        assert_eq!(algebra.name(), "so(3)");
        assert!(!algebra.is_abelian());
        assert!(algebra.verify_jacobi_identity());

        // Test [L_x, L_y] = L_z
        let l_x = vec![1.0, 0.0, 0.0];
        let l_y = vec![0.0, 1.0, 0.0];

        let bracket = algebra.bracket(&l_x, &l_y).unwrap();
        assert_eq!(bracket[0], 0.0);
        assert_eq!(bracket[1], 0.0);
        assert_eq!(bracket[2], 1.0); // Should be L_z
    }

    #[test]
    fn test_sl2_algebra() {
        let algebra = LieAlgebra::sl2_real();

        assert_eq!(algebra.dimension(), 3);
        assert!(algebra.verify_jacobi_identity());

        // Test [H, X] = 2X
        let h = vec![1.0, 0.0, 0.0];
        let x = vec![0.0, 1.0, 0.0];

        let bracket = algebra.bracket(&h, &x).unwrap();
        assert_eq!(bracket[0], 0.0);
        assert_eq!(bracket[1], 2.0); // 2X
        assert_eq!(bracket[2], 0.0);

        // Test [X, Y] = H
        let y = vec![0.0, 0.0, 1.0];
        let bracket_xy = algebra.bracket(&x, &y).unwrap();
        assert_eq!(bracket_xy[0], 1.0); // H
        assert_eq!(bracket_xy[1], 0.0);
        assert_eq!(bracket_xy[2], 0.0);
    }

    #[test]
    fn test_heisenberg_algebra() {
        let algebra = LieAlgebra::heisenberg();

        assert_eq!(algebra.dimension(), 3);
        assert!(algebra.verify_jacobi_identity());

        // Test [X, Y] = Z
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0, 1.0, 0.0];

        let bracket = algebra.bracket(&x, &y).unwrap();
        assert_eq!(bracket[0], 0.0);
        assert_eq!(bracket[1], 0.0);
        assert_eq!(bracket[2], 1.0); // Z

        // Test that Z is central: [Z, X] = [Z, Y] = 0
        let z = vec![0.0, 0.0, 1.0];
        let bracket_zx = algebra.bracket(&z, &x).unwrap();
        assert_eq!(bracket_zx[0], 0.0);
        assert_eq!(bracket_zx[1], 0.0);
        assert_eq!(bracket_zx[2], 0.0);
    }

    #[test]
    fn test_killing_form_so3() {
        let algebra = LieAlgebra::so3();

        // For so(3), the Killing form is B(X,Y) = 2 tr(XY)
        let l_x = vec![1.0, 0.0, 0.0];
        let l_y = vec![0.0, 1.0, 0.0];

        let b_xx = algebra.killing_form(&l_x, &l_x).unwrap();
        let b_xy = algebra.killing_form(&l_x, &l_y).unwrap();

        // Different basis elements should be orthogonal
        assert!(b_xy.abs() < 1e-10);
    }

    #[test]
    fn test_antisymmetry() {
        let algebra = LieAlgebra::so3();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let bracket_xy = algebra.bracket(&x, &y).unwrap();
        let bracket_yx = algebra.bracket(&y, &x).unwrap();

        // [X, Y] = -[Y, X]
        for i in 0..3 {
            assert!((bracket_xy[i] + bracket_yx[i]).abs() < 1e-10);
        }
    }
}
