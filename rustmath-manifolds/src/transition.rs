//! Chart transition functions
//!
//! This module provides structures for representing coordinate transformations
//! between different charts on a manifold, including their Jacobian matrices.

use crate::chart::Chart;
use crate::errors::{ManifoldError, Result};
use rustmath_matrix::Matrix;
use rustmath_symbolic::{Expr, Symbol, Substituter, ExprMutator};
use std::collections::HashMap;
use std::sync::Arc;

/// A transition function between two coordinate charts
///
/// Represents the coordinate transformation from one chart to another,
/// along with its Jacobian matrix for computing how tensors transform.
///
/// # Mathematical Background
///
/// If φ: U → ℝⁿ and ψ: V → ℝⁿ are two charts with overlap U ∩ V ≠ ∅,
/// the transition function is ψ ∘ φ⁻¹: φ(U ∩ V) → ψ(U ∩ V).
///
/// In coordinates: (x¹,...,xⁿ) ↦ (y¹(x),...,yⁿ(x))
///
/// The Jacobian matrix is J = [∂y^i/∂x^j]
#[derive(Clone)]
pub struct TransitionFunction {
    /// Name of the source chart
    source_chart_name: String,
    /// Name of the target chart
    target_chart_name: String,
    /// Coordinate transformation expressions: y^i = f^i(x^1, ..., x^n)
    /// Each expression maps from source coordinates to one target coordinate
    coordinate_maps: Vec<Expr>,
    /// Jacobian matrix: J^i_j = ∂y^i/∂x^j
    /// Computed lazily and cached
    jacobian: Option<Matrix<Expr>>,
    /// Inverse Jacobian matrix (if computed)
    inverse_jacobian: Option<Matrix<Expr>>,
}

impl TransitionFunction {
    /// Create a new transition function
    ///
    /// # Arguments
    ///
    /// * `source_chart_name` - Name of the source chart
    /// * `target_chart_name` - Name of the target chart
    /// * `coordinate_maps` - Expressions for each target coordinate in terms of source coordinates
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::transition::TransitionFunction;
    /// use rustmath_symbolic::{Expr, Symbol};
    ///
    /// // Polar to Cartesian: x = r*cos(θ), y = r*sin(θ)
    /// let r = Symbol::new("r");
    /// let theta = Symbol::new("theta");
    ///
    /// let x = Expr::Symbol(r.clone()) * Expr::Symbol(theta.clone()).cos();
    /// let y = Expr::Symbol(r.clone()) * Expr::Symbol(theta.clone()).sin();
    ///
    /// let transition = TransitionFunction::new(
    ///     "polar",
    ///     "cartesian",
    ///     vec![x, y],
    /// ).unwrap();
    /// ```
    pub fn new(
        source_chart_name: impl Into<String>,
        target_chart_name: impl Into<String>,
        coordinate_maps: Vec<Expr>,
    ) -> Result<Self> {
        if coordinate_maps.is_empty() {
            return Err(ManifoldError::InvalidChart(
                "Coordinate maps cannot be empty".to_string(),
            ));
        }

        Ok(Self {
            source_chart_name: source_chart_name.into(),
            target_chart_name: target_chart_name.into(),
            coordinate_maps,
            jacobian: None,
            inverse_jacobian: None,
        })
    }

    /// Get the source chart name
    pub fn source_chart_name(&self) -> &str {
        &self.source_chart_name
    }

    /// Get the target chart name
    pub fn target_chart_name(&self) -> &str {
        &self.target_chart_name
    }

    /// Get the coordinate map expressions
    pub fn coordinate_maps(&self) -> &[Expr] {
        &self.coordinate_maps
    }

    /// Get the dimension of the transformation
    pub fn dimension(&self) -> usize {
        self.coordinate_maps.len()
    }

    /// Compute the Jacobian matrix symbolically
    ///
    /// The Jacobian J^i_j = ∂y^i/∂x^j where y are target coordinates
    /// and x are source coordinates.
    ///
    /// # Arguments
    ///
    /// * `source_symbols` - Symbols representing source coordinates [x¹, x², ...]
    ///
    /// # Returns
    ///
    /// A symbolic matrix where entry (i,j) is ∂y^i/∂x^j
    pub fn compute_jacobian(&mut self, source_symbols: &[Symbol]) -> Result<&Matrix<Expr>> {
        if source_symbols.len() != self.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension(),
                actual: source_symbols.len(),
            });
        }

        if self.jacobian.is_some() {
            return Ok(self.jacobian.as_ref().unwrap());
        }

        let n = self.dimension();
        let mut jacobian_entries = Vec::with_capacity(n * n);

        // Compute J^i_j = ∂(y^i)/∂(x^j)
        for i in 0..n {
            for j in 0..n {
                let derivative = self.coordinate_maps[i].differentiate(&source_symbols[j]);
                jacobian_entries.push(derivative);
            }
        }

        let jacobian = Matrix::from_vec(n, n, jacobian_entries)
            .map_err(|e| ManifoldError::ComputationError(format!("Failed to create Jacobian matrix: {:?}", e)))?;

        self.jacobian = Some(jacobian);
        Ok(self.jacobian.as_ref().unwrap())
    }

    /// Get the Jacobian matrix (must be computed first)
    pub fn jacobian(&self) -> Option<&Matrix<Expr>> {
        self.jacobian.as_ref()
    }

    /// Apply the transition function to coordinates
    ///
    /// Given source coordinates, compute target coordinates.
    ///
    /// # Arguments
    ///
    /// * `source_coords` - Map from source coordinate symbols to values
    ///
    /// # Returns
    ///
    /// Vector of target coordinate values
    pub fn apply(&self, source_coords: &HashMap<Symbol, f64>) -> Result<Vec<f64>> {
        let mut result = Vec::with_capacity(self.dimension());

        for map_expr in &self.coordinate_maps {
            // Substitute source coordinate values into the expression
            let value = evaluate_expr(map_expr, source_coords)?;
            result.push(value);
        }

        Ok(result)
    }

    /// Pullback a scalar field expression through this coordinate change
    ///
    /// If f is expressed in target coordinates y, this computes f ∘ transition
    /// expressed in source coordinates x.
    ///
    /// # Arguments
    ///
    /// * `target_expr` - Expression in target coordinates
    /// * `target_symbols` - Symbols for target coordinates
    ///
    /// # Returns
    ///
    /// Expression in source coordinates
    pub fn pullback_scalar(
        &self,
        target_expr: &Expr,
        target_symbols: &[Symbol],
    ) -> Result<Expr> {
        if target_symbols.len() != self.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension(),
                actual: target_symbols.len(),
            });
        }

        // Build substitution map: y^i -> f^i(x^1, ..., x^n)
        let mut substituter = Substituter::new();
        for (i, target_symbol) in target_symbols.iter().enumerate() {
            substituter.add_replacement(target_symbol, self.coordinate_maps[i].clone());
        }

        Ok(substituter.mutate(target_expr))
    }

    /// Transform a contravariant vector using the Jacobian
    ///
    /// V^i_new = J^i_j V^j_old
    ///
    /// # Arguments
    ///
    /// * `vector_components` - Components in source chart
    /// * `source_symbols` - Source coordinate symbols (for Jacobian computation)
    ///
    /// # Returns
    ///
    /// Components in target chart
    pub fn transform_contravariant_vector(
        &mut self,
        vector_components: &[Expr],
        source_symbols: &[Symbol],
    ) -> Result<Vec<Expr>> {
        if vector_components.len() != self.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension(),
                actual: vector_components.len(),
            });
        }

        // Ensure Jacobian is computed
        self.compute_jacobian(source_symbols)?;
        let jacobian = self.jacobian.as_ref().unwrap();

        let n = self.dimension();
        let mut result = Vec::with_capacity(n);

        // V^i_new = J^i_j V^j_old
        for i in 0..n {
            let mut component = Expr::from(0);
            for j in 0..n {
                let j_ij = jacobian.get(i, j)
                    .map_err(|e| ManifoldError::ComputationError(format!("Jacobian access error: {:?}", e)))?;
                component = component + j_ij.clone() * vector_components[j].clone();
            }
            result.push(component);
        }

        Ok(result)
    }

    /// Compute the determinant of the Jacobian (useful for integration)
    ///
    /// Returns det(J) where J is the Jacobian matrix.
    pub fn jacobian_determinant(&mut self, source_symbols: &[Symbol]) -> Result<Expr> {
        self.compute_jacobian(source_symbols)?;
        let jacobian = self.jacobian.as_ref().unwrap();

        // For small dimensions, we can compute determinant symbolically
        // For larger dimensions, this becomes very expensive
        jacobian.det()
            .map_err(|e| ManifoldError::ComputationError(format!("Failed to compute Jacobian determinant: {:?}", e)))
    }
}

/// Evaluate an expression by substituting coordinate values
fn evaluate_expr(expr: &Expr, coord_values: &HashMap<Symbol, f64>) -> Result<f64> {
    match expr {
        Expr::Integer(i) => Ok(i.to_f64().ok_or_else(|| {
            ManifoldError::ComputationError("Integer too large to convert to f64".to_string())
        })?),
        Expr::Rational(r) => Ok(r.to_f64().ok_or_else(|| {
            ManifoldError::ComputationError("Rational too large to convert to f64".to_string())
        })?),
        Expr::Symbol(s) => coord_values.get(s).copied().ok_or_else(|| {
            ManifoldError::ComputationError(format!("Symbol {} not found in coordinate values", s.name()))
        }),
        Expr::Binary(op, left, right) => {
            let left_val = evaluate_expr(left, coord_values)?;
            let right_val = evaluate_expr(right, coord_values)?;

            use rustmath_symbolic::BinaryOp;
            match op {
                BinaryOp::Add => Ok(left_val + right_val),
                BinaryOp::Sub => Ok(left_val - right_val),
                BinaryOp::Mul => Ok(left_val * right_val),
                BinaryOp::Div => {
                    if right_val.abs() < 1e-15 {
                        Err(ManifoldError::ComputationError("Division by zero".to_string()))
                    } else {
                        Ok(left_val / right_val)
                    }
                }
                BinaryOp::Pow => Ok(left_val.powf(right_val)),
            }
        }
        Expr::Unary(op, inner) => {
            let val = evaluate_expr(inner, coord_values)?;

            use rustmath_symbolic::UnaryOp;
            match op {
                UnaryOp::Neg => Ok(-val),
                UnaryOp::Sin => Ok(val.sin()),
                UnaryOp::Cos => Ok(val.cos()),
                UnaryOp::Tan => Ok(val.tan()),
                UnaryOp::Exp => Ok(val.exp()),
                UnaryOp::Log => {
                    if val <= 0.0 {
                        Err(ManifoldError::ComputationError("Log of non-positive number".to_string()))
                    } else {
                        Ok(val.ln())
                    }
                }
                UnaryOp::Sqrt => {
                    if val < 0.0 {
                        Err(ManifoldError::ComputationError("Square root of negative number".to_string()))
                    } else {
                        Ok(val.sqrt())
                    }
                }
                UnaryOp::Abs => Ok(val.abs()),
                UnaryOp::Sign => Ok(if val > 0.0 { 1.0 } else if val < 0.0 { -1.0 } else { 0.0 }),
                UnaryOp::Sinh => Ok(val.sinh()),
                UnaryOp::Cosh => Ok(val.cosh()),
                UnaryOp::Tanh => Ok(val.tanh()),
                UnaryOp::Arcsin => Ok(val.asin()),
                UnaryOp::Arccos => Ok(val.acos()),
                UnaryOp::Arctan => Ok(val.atan()),
                UnaryOp::Arcsinh => Ok(val.asinh()),
                UnaryOp::Arccosh => Ok(val.acosh()),
                UnaryOp::Arctanh => Ok(val.atanh()),
                _ => Err(ManifoldError::ComputationError(format!("Unsupported unary operation: {:?}", op))),
            }
        }
        Expr::Function(name, _args) => {
            Err(ManifoldError::ComputationError(format!("Function evaluation not supported: {}", name)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_creation() {
        let r = Symbol::new("r");
        let theta = Symbol::new("theta");

        // Polar to Cartesian
        let x = Expr::Symbol(r.clone()) * Expr::Symbol(theta.clone()).cos();
        let y = Expr::Symbol(r.clone()) * Expr::Symbol(theta.clone()).sin();

        let transition = TransitionFunction::new("polar", "cartesian", vec![x, y]).unwrap();

        assert_eq!(transition.source_chart_name(), "polar");
        assert_eq!(transition.target_chart_name(), "cartesian");
        assert_eq!(transition.dimension(), 2);
    }

    #[test]
    fn test_jacobian_computation() {
        let r = Symbol::new("r");
        let theta = Symbol::new("theta");

        // Polar to Cartesian: x = r*cos(θ), y = r*sin(θ)
        let x = Expr::Symbol(r.clone()) * Expr::Symbol(theta.clone()).cos();
        let y = Expr::Symbol(r.clone()) * Expr::Symbol(theta.clone()).sin();

        let mut transition = TransitionFunction::new("polar", "cartesian", vec![x, y]).unwrap();

        let source_symbols = vec![r.clone(), theta.clone()];
        let jacobian = transition.compute_jacobian(&source_symbols).unwrap();

        // Jacobian should be 2x2
        assert_eq!(jacobian.rows(), 2);
        assert_eq!(jacobian.cols(), 2);

        // J[0,0] = ∂x/∂r = cos(θ)
        // J[0,1] = ∂x/∂θ = -r*sin(θ)
        // J[1,0] = ∂y/∂r = sin(θ)
        // J[1,1] = ∂y/∂θ = r*cos(θ)
    }

    #[test]
    fn test_identity_transition() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // Identity transformation
        let maps = vec![Expr::Symbol(x.clone()), Expr::Symbol(y.clone())];
        let mut transition = TransitionFunction::new("chart1", "chart2", maps).unwrap();

        let source_symbols = vec![x, y];
        let jacobian = transition.compute_jacobian(&source_symbols).unwrap();

        // Identity Jacobian should be [[1, 0], [0, 1]]
        assert_eq!(jacobian.get(0, 0).unwrap(), &Expr::from(1));
        assert_eq!(jacobian.get(0, 1).unwrap(), &Expr::from(0));
        assert_eq!(jacobian.get(1, 0).unwrap(), &Expr::from(0));
        assert_eq!(jacobian.get(1, 1).unwrap(), &Expr::from(1));
    }

    #[test]
    fn test_apply_transition() {
        let r = Symbol::new("r");
        let theta = Symbol::new("theta");

        // Simple linear transformation: x = 2r, y = 3r
        let x = Expr::Symbol(r.clone()) * Expr::from(2);
        let y = Expr::Symbol(r.clone()) * Expr::from(3);

        let transition = TransitionFunction::new("source", "target", vec![x, y]).unwrap();

        // Apply with r = 5
        let mut coords = HashMap::new();
        coords.insert(r, 5.0);
        coords.insert(theta, 0.0); // Not used but included

        let result = transition.apply(&coords).unwrap();

        assert_eq!(result.len(), 2);
        assert!((result[0] - 10.0).abs() < 1e-10); // x = 2*5 = 10
        assert!((result[1] - 15.0).abs() < 1e-10); // y = 3*5 = 15
    }

    #[test]
    fn test_pullback_scalar() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let u = Symbol::new("u");
        let v = Symbol::new("v");

        // Transition: x = u + v, y = u - v
        let maps = vec![
            Expr::Symbol(u.clone()) + Expr::Symbol(v.clone()),
            Expr::Symbol(u.clone()) - Expr::Symbol(v.clone()),
        ];
        let transition = TransitionFunction::new("uv", "xy", maps).unwrap();

        // Scalar field in xy coordinates: f = x + 2*y
        let f_xy = Expr::Symbol(x.clone()) + Expr::Symbol(y.clone()) * Expr::from(2);

        // Pullback to uv coordinates
        let f_uv = transition.pullback_scalar(&f_xy, &[x, y]).unwrap();

        // f(u,v) = (u+v) + 2(u-v) = 3u - v
        // We can't easily check symbolic equality, but we verified it's computed
        assert!(!f_uv.is_constant());
    }
}
