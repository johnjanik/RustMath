//! # Tropical Varieties
//!
//! This module implements tropical varieties - the zero loci of tropical polynomials.
//!
//! ## Overview
//!
//! A tropical variety is the set of points where a tropical polynomial
//! attains its minimum/maximum at least twice (i.e., where the piecewise linear
//! function is not smooth).
//!
//! ## Theory
//!
//! - **Tropical Curve**: 1-dimensional tropical variety (in 2D space)
//! - **Tropical Surface**: 2-dimensional tropical variety
//! - **Tropical Hypersurface**: (n-1)-dimensional variety in n-dimensional space
//!
//! Tropical varieties are:
//! - Balanced polyhedral complexes
//! - Dual to subdivisions of Newton polytopes
//! - Piecewise-linear analogues of algebraic varieties
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::semirings::tropical_variety::TropicalVariety;
//! use rustmath_rings::semirings::tropical_semiring::TropicalType;
//!
//! let variety = TropicalVariety::new(2, TropicalType::Min);
//! ```

use super::tropical_mpolynomial::TropicalMPolynomial;
use super::tropical_semiring::TropicalType;
use std::fmt;

/// A tropical variety
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalVariety {
    /// Ambient dimension
    dimension: usize,
    /// Tropical type
    tropical_type: TropicalType,
    /// Defining polynomials
    polynomials: Vec<TropicalMPolynomial>,
}

impl TropicalVariety {
    /// Create a new tropical variety
    pub fn new(dimension: usize, tropical_type: TropicalType) -> Self {
        Self {
            dimension,
            tropical_type,
            polynomials: Vec::new(),
        }
    }

    /// Add a defining polynomial
    pub fn add_polynomial(&mut self, poly: TropicalMPolynomial) {
        if poly.nvars() != self.dimension {
            panic!("Polynomial dimension must match variety dimension");
        }
        self.polynomials.push(poly);
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the defining polynomials
    pub fn polynomials(&self) -> &[TropicalMPolynomial] {
        &self.polynomials
    }

    /// Check if a point is on the variety
    /// A point is on the variety if for each polynomial, the minimum is attained
    /// by at least two terms
    pub fn contains_point(&self, point: &[f64]) -> bool {
        if point.len() != self.dimension {
            return false;
        }

        for poly in &self.polynomials {
            if !self.is_tropical_zero(poly, point) {
                return false;
            }
        }

        true
    }

    /// Check if a polynomial evaluates to a "tropical zero" at a point
    /// (i.e., minimum attained by at least 2 terms)
    fn is_tropical_zero(&self, poly: &TropicalMPolynomial, point: &[f64]) -> bool {
        let mut min_value = f64::INFINITY;
        let mut min_count = 0;

        for (exp, coeff) in poly.terms() {
            let mut value = *coeff;
            for (i, e) in exp.iter().enumerate() {
                value += (*e as f64) * point[i];
            }

            if (value - min_value).abs() < 1e-10 {
                min_count += 1;
            } else if value < min_value {
                min_value = value;
                min_count = 1;
            }
        }

        min_count >= 2
    }
}

impl fmt::Display for TropicalVariety {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tropical Variety in dimension {} ({:?})",
            self.dimension, self.tropical_type
        )
    }
}

/// A tropical curve (1-dimensional tropical variety in 2D)
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalCurve {
    /// Base variety
    variety: TropicalVariety,
}

impl TropicalCurve {
    /// Create a new tropical curve
    pub fn new(tropical_type: TropicalType) -> Self {
        Self {
            variety: TropicalVariety::new(2, tropical_type),
        }
    }

    /// Add a defining polynomial
    pub fn add_polynomial(&mut self, poly: TropicalMPolynomial) {
        self.variety.add_polynomial(poly);
    }

    /// Get the underlying variety
    pub fn variety(&self) -> &TropicalVariety {
        &self.variety
    }

    /// Get genus (simplified implementation)
    pub fn genus(&self) -> usize {
        // Actual genus computation requires analyzing the graph structure
        // This is a placeholder
        0
    }
}

/// A tropical surface (2-dimensional tropical variety)
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalSurface {
    /// Base variety
    variety: TropicalVariety,
}

impl TropicalSurface {
    /// Create a new tropical surface
    pub fn new(dimension: usize, tropical_type: TropicalType) -> Self {
        Self {
            variety: TropicalVariety::new(dimension, tropical_type),
        }
    }

    /// Add a defining polynomial
    pub fn add_polynomial(&mut self, poly: TropicalMPolynomial) {
        self.variety.add_polynomial(poly);
    }

    /// Get the underlying variety
    pub fn variety(&self) -> &TropicalVariety {
        &self.variety
    }
}

impl fmt::Display for TropicalCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tropical Curve")
    }
}

impl fmt::Display for TropicalSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tropical Surface in dimension {}",
            self.variety.dimension()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_tropical_variety() {
        let variety = TropicalVariety::new(2, TropicalType::Min);
        assert_eq!(variety.dimension(), 2);
        assert_eq!(variety.polynomials().len(), 0);
    }

    #[test]
    fn test_add_polynomial() {
        let mut variety = TropicalVariety::new(2, TropicalType::Min);

        let mut terms = BTreeMap::new();
        terms.insert(vec![0, 0], 0.0);
        terms.insert(vec![1, 0], 1.0);

        let poly = TropicalMPolynomial::new(terms, TropicalType::Min, 2);
        variety.add_polynomial(poly);

        assert_eq!(variety.polynomials().len(), 1);
    }

    #[test]
    #[should_panic(expected = "Polynomial dimension must match variety dimension")]
    fn test_dimension_mismatch() {
        let mut variety = TropicalVariety::new(2, TropicalType::Min);

        let mut terms = BTreeMap::new();
        terms.insert(vec![1, 0, 0], 1.0); // 3D polynomial

        let poly = TropicalMPolynomial::new(terms, TropicalType::Min, 3);
        variety.add_polynomial(poly);
    }

    #[test]
    fn test_tropical_curve() {
        let curve = TropicalCurve::new(TropicalType::Min);
        assert_eq!(curve.variety().dimension(), 2);
        assert_eq!(curve.genus(), 0);
    }

    #[test]
    fn test_tropical_surface() {
        let surface = TropicalSurface::new(3, TropicalType::Min);
        assert_eq!(surface.variety().dimension(), 3);
    }

    #[test]
    fn test_contains_point() {
        let mut variety = TropicalVariety::new(2, TropicalType::Min);

        let mut terms = BTreeMap::new();
        terms.insert(vec![0, 0], 0.0); // constant
        terms.insert(vec![1, 0], 0.0); // x
        terms.insert(vec![0, 1], 0.0); // y

        let poly = TropicalMPolynomial::new(terms, TropicalType::Min, 2);
        variety.add_polynomial(poly);

        // At (0, 0), all three terms give value 0, so it's on the variety
        assert!(variety.contains_point(&[0.0, 0.0]));
    }

    #[test]
    fn test_display() {
        let variety = TropicalVariety::new(3, TropicalType::Min);
        let display = format!("{}", variety);
        assert!(display.contains("dimension 3"));

        let curve = TropicalCurve::new(TropicalType::Min);
        assert_eq!(format!("{}", curve), "Tropical Curve");

        let surface = TropicalSurface::new(3, TropicalType::Max);
        let display = format!("{}", surface);
        assert!(display.contains("dimension 3"));
    }
}
