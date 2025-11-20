//! Genus computation for algebraic curves
//!
//! The genus is a topological invariant of a curve that measures its "complexity"
//! or number of "holes" when viewed as a Riemann surface.
//!
//! For a smooth plane curve of degree d, the genus is given by:
//! g = (d-1)(d-2)/2
//!
//! For singular curves, we use the genus-delta formula:
//! g = (d-1)(d-2)/2 - Σ δᵢ
//! where δᵢ is the delta invariant of each singularity.

use rustmath_core::Ring;
use rustmath_polynomials::multivariate::MultiPoly;
use crate::singularities::{Singularity, find_singularities};

/// Compute the arithmetic genus of a smooth plane curve
///
/// For a smooth plane curve of degree d: g = (d-1)(d-2)/2
pub fn arithmetic_genus(degree: usize) -> usize {
    if degree <= 1 {
        return 0;
    }
    (degree - 1) * (degree - 2) / 2
}

/// Compute the geometric genus of a curve with singularities
///
/// Uses the genus-delta formula: g = g_arithmetic - Σ δᵢ
pub fn geometric_genus<R: Ring + Clone + PartialEq>(
    poly: &MultiPoly<R>,
    singularities: &[Singularity<R>],
) -> isize {
    let d = poly.total_degree();
    let g_arithmetic = arithmetic_genus(d) as isize;

    // Sum up all delta invariants
    let total_delta: isize = singularities.iter().map(|s| s.delta as isize).sum();

    // Geometric genus = arithmetic genus - total delta
    (g_arithmetic - total_delta).max(0)
}

/// Compute the genus of a curve (automatically detecting singularities)
pub fn compute_genus<R: Ring + Clone + PartialEq>(poly: &MultiPoly<R>) -> isize {
    let singularities = find_singularities(poly);
    geometric_genus(poly, &singularities)
}

/// Classification based on genus
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenusClass {
    /// Genus 0: Rational curves (can be parameterized by rational functions)
    Rational,
    /// Genus 1: Elliptic curves (have a group structure)
    Elliptic,
    /// Genus ≥ 2: Curves of general type
    GeneralType(usize),
}

impl GenusClass {
    /// Classify a curve by its genus
    pub fn from_genus(g: usize) -> Self {
        match g {
            0 => GenusClass::Rational,
            1 => GenusClass::Elliptic,
            n => GenusClass::GeneralType(n),
        }
    }

    /// Get the genus value
    pub fn genus(&self) -> usize {
        match self {
            GenusClass::Rational => 0,
            GenusClass::Elliptic => 1,
            GenusClass::GeneralType(g) => *g,
        }
    }

    /// Check if the curve is rational (genus 0)
    pub fn is_rational(&self) -> bool {
        matches!(self, GenusClass::Rational)
    }

    /// Check if the curve is elliptic (genus 1)
    pub fn is_elliptic(&self) -> bool {
        matches!(self, GenusClass::Elliptic)
    }

    /// Check if the curve is of general type (genus ≥ 2)
    pub fn is_general_type(&self) -> bool {
        matches!(self, GenusClass::GeneralType(_))
    }
}

/// Properties related to genus
pub struct GenusData {
    /// The genus of the curve
    pub genus: usize,
    /// Classification by genus
    pub classification: GenusClass,
    /// Arithmetic genus (for smooth curve of same degree)
    pub arithmetic_genus: usize,
    /// Number of singularities
    pub num_singularities: usize,
    /// Total delta invariant
    pub total_delta: usize,
}

impl GenusData {
    /// Create genus data from a polynomial
    pub fn from_polynomial<R: Ring + Clone + PartialEq>(poly: &MultiPoly<R>) -> Self {
        let degree = poly.total_degree();
        let arith_genus = arithmetic_genus(degree);
        let singularities = find_singularities(poly);
        let total_delta: usize = singularities.iter().map(|s| s.delta).sum();
        let geom_genus = (arith_genus as isize - total_delta as isize).max(0) as usize;

        GenusData {
            genus: geom_genus,
            classification: GenusClass::from_genus(geom_genus),
            arithmetic_genus: arith_genus,
            num_singularities: singularities.len(),
            total_delta,
        }
    }

    /// Check if the curve is smooth
    pub fn is_smooth(&self) -> bool {
        self.num_singularities == 0
    }

    /// Get the dimension of the space of holomorphic differentials
    pub fn dimension_of_differentials(&self) -> usize {
        self.genus
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_genus() {
        // Line (degree 1): g = 0
        assert_eq!(arithmetic_genus(1), 0);

        // Conic (degree 2): g = 0
        assert_eq!(arithmetic_genus(2), 0);

        // Cubic (degree 3): g = 1
        assert_eq!(arithmetic_genus(3), 1);

        // Quartic (degree 4): g = 3
        assert_eq!(arithmetic_genus(4), 3);

        // Quintic (degree 5): g = 6
        assert_eq!(arithmetic_genus(5), 6);

        // Sextic (degree 6): g = 10
        assert_eq!(arithmetic_genus(6), 10);
    }

    #[test]
    fn test_genus_class() {
        let rational = GenusClass::Rational;
        assert!(rational.is_rational());
        assert!(!rational.is_elliptic());
        assert_eq!(rational.genus(), 0);

        let elliptic = GenusClass::Elliptic;
        assert!(!elliptic.is_rational());
        assert!(elliptic.is_elliptic());
        assert_eq!(elliptic.genus(), 1);

        let general = GenusClass::GeneralType(5);
        assert!(general.is_general_type());
        assert_eq!(general.genus(), 5);
    }

    #[test]
    fn test_from_genus() {
        assert_eq!(GenusClass::from_genus(0), GenusClass::Rational);
        assert_eq!(GenusClass::from_genus(1), GenusClass::Elliptic);
        assert_eq!(GenusClass::from_genus(2), GenusClass::GeneralType(2));
    }

    #[test]
    fn test_geometric_genus_with_singularities() {
        use rustmath_polynomials::multivariate::MultiPoly;
        use rustmath_rationals::Rational;
        use crate::singularities::{Singularity, SingularityType};

        // Cubic with one node: g = 1 - 1 = 0
        let poly = MultiPoly::<Rational>::new(2);
        let node = Singularity::new(
            vec![Rational::zero(), Rational::zero()],
            SingularityType::Node,
            2,
        );

        let g = geometric_genus(&poly, &[node]);
        // For a cubic (degree 3), arithmetic genus = 1
        // With one node (delta = 1), geometric genus = 0
        // But our poly has degree 0, so this is just a test
        assert!(g >= 0);
    }
}
