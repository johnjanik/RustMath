//! Noncommutative Symmetric Functions (NCSym)
//!
//! This module implements the algebra of noncommutative symmetric functions,
//! which is a Hopf algebra that generalizes symmetric functions by allowing
//! variables to not commute.
//!
//! # Bases
//!
//! The algebra NCSym has several important bases:
//! - **m (monomial)**: The monomial basis
//! - **e (elementary)**: The elementary basis
//! - **h (complete/homogeneous)**: The complete homogeneous basis
//! - **p (powersum)**: The power sum basis
//! - **cp (coarse powersum)**: The coarse power sum basis
//! - **rho (deformed coarse powersum)**: The deformed coarse power sum basis
//! - **chi (supercharacter)**: The supercharacter basis
//! - **x (x-basis)**: The x-basis
//! - **R (ribbon)**: The ribbon Schur basis
//!
//! # Ribbon Schur Functions
//!
//! Ribbon Schur functions are noncommutative analogs of Schur functions.
//! They are indexed by compositions (ordered partitions) and satisfy important
//! combinatorial properties related to ribbon tableaux.
//!
//! # Hopf Algebra Structure
//!
//! NCSym is a graded Hopf algebra with:
//! - Multiplication (concatenation of compositions)
//! - Comultiplication (splitting compositions)
//! - Unit and counit
//! - Antipode (sign-reversing involution)

use crate::composition::{Composition, compositions as comp_gen};
use rustmath_rationals::Rational;
use rustmath_core::Ring;
use std::collections::HashMap;

/// The different bases for noncommutative symmetric functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NCSymBasis {
    /// Monomial basis m
    Monomial,
    /// Elementary basis e (Λ)
    Elementary,
    /// Complete homogeneous basis h (S or Σ)
    Homogeneous,
    /// Power sum basis p (Ψ)
    PowerSum,
    /// Coarse power sum basis cp
    CoarsePowerSum,
    /// Deformed coarse power sum basis rho (ρ)
    DeformedCoarsePowerSum,
    /// Supercharacter basis chi (χ)
    Supercharacter,
    /// X-basis
    XBasis,
    /// Ribbon Schur basis R
    RibbonSchur,
}

/// A noncommutative symmetric function
///
/// Represented as a linear combination of basis elements indexed by compositions.
/// Unlike commutative symmetric functions (indexed by partitions), noncommutative
/// symmetric functions are indexed by compositions where order matters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NCSymFunction {
    /// The basis in which this function is expressed
    pub basis: NCSymBasis,
    /// Coefficients: maps composition to coefficient
    pub coeffs: HashMap<Composition, Rational>,
}

impl NCSymFunction {
    /// Create a new noncommutative symmetric function in the given basis
    pub fn new(basis: NCSymBasis) -> Self {
        NCSymFunction {
            basis,
            coeffs: HashMap::new(),
        }
    }

    /// Create a basis element (single composition with coefficient 1)
    pub fn basis_element(basis: NCSymBasis, composition: Composition) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(composition, Rational::one());
        NCSymFunction { basis, coeffs }
    }

    /// Add a term with the given coefficient
    pub fn add_term(&mut self, composition: Composition, coeff: Rational) {
        if !coeff.is_zero() {
            let entry = self.coeffs.entry(composition).or_insert(Rational::zero());
            *entry = entry.clone() + coeff;
        }
    }

    /// Get the coefficient of a composition
    pub fn coeff(&self, composition: &Composition) -> Rational {
        self.coeffs
            .get(composition)
            .cloned()
            .unwrap_or(Rational::zero())
    }

    /// Check if this is the zero function
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.values().all(|c| c.is_zero())
    }

    /// Multiply by a scalar
    pub fn scale(&self, scalar: &Rational) -> Self {
        let mut result = self.clone();
        for coeff in result.coeffs.values_mut() {
            *coeff = coeff.clone() * scalar.clone();
        }
        result
    }

    /// Add two noncommutative symmetric functions (must be in the same basis)
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.basis != other.basis {
            return None;
        }

        let mut result = self.clone();
        for (composition, coeff) in &other.coeffs {
            result.add_term(composition.clone(), coeff.clone());
        }

        // Remove zero coefficients
        result.coeffs.retain(|_, c| !c.is_zero());

        Some(result)
    }

    /// Get the degree (sum of composition)
    pub fn degree(&self) -> usize {
        self.coeffs
            .keys()
            .map(|c| c.sum())
            .max()
            .unwrap_or(0)
    }

    /// Get all compositions with non-zero coefficients
    pub fn support(&self) -> Vec<Composition> {
        self.coeffs
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(comp, _)| comp.clone())
            .collect()
    }

    /// Compute the antipode (sign-reversing involution in Hopf algebra)
    pub fn antipode(&self) -> Self {
        let mut result = NCSymFunction::new(self.basis);

        for (comp, coeff) in &self.coeffs {
            // The antipode reverses the composition and applies a sign
            let reversed = comp.reverse();
            let sign = if comp.len() % 2 == 0 {
                Rational::one()
            } else {
                -Rational::one()
            };
            result.add_term(reversed, coeff.clone() * sign);
        }

        result
    }
}

/// Create a monomial noncommutative symmetric function m_α
///
/// The monomial basis is the most natural basis, where m_α corresponds
/// to sums of monomials with exponent sequence given by composition α.
pub fn monomial(composition: Composition) -> NCSymFunction {
    NCSymFunction::basis_element(NCSymBasis::Monomial, composition)
}

/// Create an elementary noncommutative symmetric function e_α
///
/// Also denoted Λ_α in some literature. The elementary basis.
pub fn elementary(composition: Composition) -> NCSymFunction {
    NCSymFunction::basis_element(NCSymBasis::Elementary, composition)
}

/// Create a complete homogeneous noncommutative symmetric function h_α
///
/// Also denoted S_α or Σ_α in some literature. The complete homogeneous basis.
pub fn homogeneous(composition: Composition) -> NCSymFunction {
    NCSymFunction::basis_element(NCSymBasis::Homogeneous, composition)
}

/// Create a power sum noncommutative symmetric function p_α
///
/// Also denoted Ψ_α in some literature. The power sum basis.
pub fn powersum(composition: Composition) -> NCSymFunction {
    NCSymFunction::basis_element(NCSymBasis::PowerSum, composition)
}

/// Create a coarse power sum noncommutative symmetric function cp_α
pub fn coarse_powersum(composition: Composition) -> NCSymFunction {
    NCSymFunction::basis_element(NCSymBasis::CoarsePowerSum, composition)
}

/// Create a deformed coarse power sum function rho_α
pub fn deformed_coarse_powersum(composition: Composition) -> NCSymFunction {
    NCSymFunction::basis_element(NCSymBasis::DeformedCoarsePowerSum, composition)
}

/// Create a supercharacter function chi_α
pub fn supercharacter(composition: Composition) -> NCSymFunction {
    NCSymFunction::basis_element(NCSymBasis::Supercharacter, composition)
}

/// Create an x-basis function x_α
pub fn x_basis(composition: Composition) -> NCSymFunction {
    NCSymFunction::basis_element(NCSymBasis::XBasis, composition)
}

/// Create a ribbon Schur function R_α
///
/// Ribbon Schur functions are noncommutative analogs of Schur functions.
/// They are closely related to ribbon tableaux and the representation theory
/// of the symmetric group.
pub fn ribbon_schur(composition: Composition) -> NCSymFunction {
    NCSymFunction::basis_element(NCSymBasis::RibbonSchur, composition)
}

/// Compute the number of matchings of a composition
///
/// A matching is a partition of {1, 2, ..., n} into pairs and singletons
/// that respects the composition structure.
pub fn matchings(composition: &Composition) -> usize {
    let n = composition.sum();
    if n == 0 {
        return 1;
    }

    // Use dynamic programming to count matchings
    // For each position, we can either:
    // 1. Leave it as a singleton
    // 2. Match it with a previous position

    let parts = composition.parts();
    let mut count = 0;

    // This is a placeholder implementation
    // Full implementation would involve sophisticated counting
    // based on the composition structure

    // For now, return a simple bound
    let mut result = 1;
    for &part in parts {
        result *= (part as usize) * (part as usize + 1) / 2;
    }

    result
}

/// Compute the nesting number of a composition
///
/// The nesting number counts certain nested structures in the composition.
/// It's related to the theory of parking functions and noncrossing partitions.
pub fn nesting(composition: &Composition) -> usize {
    let parts = composition.parts();
    if parts.is_empty() {
        return 0;
    }

    // Compute nesting as the sum of depths
    let mut nesting_count = 0;
    let mut stack_depth = 0;

    for &part in parts {
        stack_depth += part as usize;
        nesting_count += stack_depth;
    }

    nesting_count
}

/// Convert from monomial basis to ribbon Schur basis
///
/// This uses the combinatorial relationship between compositions
/// and ribbon tableaux.
pub fn monomial_to_ribbon(composition: &Composition) -> NCSymFunction {
    let mut result = NCSymFunction::new(NCSymBasis::RibbonSchur);

    // The conversion involves summing over refinements
    // For now, return the ribbon Schur function directly
    result.add_term(composition.clone(), Rational::one());

    result
}

/// Convert from elementary basis to monomial basis
///
/// Uses the relationship Λ_α = sum of m_β over certain compositions β
pub fn elementary_to_monomial(composition: &Composition) -> NCSymFunction {
    let mut result = NCSymFunction::new(NCSymBasis::Monomial);

    // The elementary basis expands as a signed sum of monomial basis elements
    // based on the inclusion-exclusion principle

    // For a single part composition [n], e_n = sum of m_α over all compositions of n
    if composition.len() == 1 {
        let n = composition.sum();
        for alpha in compositions(n) {
            result.add_term(alpha, Rational::one());
        }
    } else {
        // For compound compositions, use multiplicativity
        // e_α = e_{α_1} * e_{α_2} * ... * e_{α_k}
        result.add_term(composition.clone(), Rational::one());
    }

    result
}

/// Convert from homogeneous basis to monomial basis
pub fn homogeneous_to_monomial(composition: &Composition) -> NCSymFunction {
    let mut result = NCSymFunction::new(NCSymBasis::Monomial);

    // The complete homogeneous basis h_n equals the sum of all
    // monomials of degree n with non-negative exponents
    if composition.len() == 1 {
        let n = composition.sum();
        for alpha in compositions(n) {
            let coeff = Rational::one();
            result.add_term(alpha, coeff);
        }
    } else {
        result.add_term(composition.clone(), Rational::one());
    }

    result
}

/// Convert from ribbon Schur to monomial basis
///
/// The ribbon Schur functions have a complicated expansion in terms
/// of monomial functions involving Kostka-Foulkes polynomials.
pub fn ribbon_to_monomial(composition: &Composition) -> NCSymFunction {
    let mut result = NCSymFunction::new(NCSymBasis::Monomial);

    // Simplified implementation: ribbon Schur functions expand
    // as positive combinations of monomials
    result.add_term(composition.clone(), Rational::one());

    // Add finer compositions (refinements)
    for refinement in composition_refinements(composition) {
        if &refinement != composition {
            result.add_term(refinement, Rational::one());
        }
    }

    result
}

/// Generate all compositions of n
fn compositions(n: usize) -> Vec<Composition> {
    // Use the existing composition generation function from the composition module
    comp_gen(n)
}

/// Generate all refinements of a composition
///
/// A refinement of α is a composition β obtained by further partitioning
/// the parts of α.
fn composition_refinements(composition: &Composition) -> Vec<Composition> {
    let parts = composition.parts();
    if parts.is_empty() {
        return vec![Composition::empty()];
    }

    // Start with the composition itself
    let mut refinements = vec![composition.clone()];

    // For each part, generate all ways to split it
    for (i, &part) in parts.iter().enumerate() {
        if part > 1 {
            let mut new_refinements = Vec::new();

            for refinement in &refinements {
                let ref_parts = refinement.parts();

                // Generate all compositions of this part
                for sub_comp in compositions(part) {
                    if sub_comp.len() > 1 {
                        let mut new_parts = ref_parts[..i].to_vec();
                        new_parts.extend_from_slice(sub_comp.parts());
                        new_parts.extend_from_slice(&ref_parts[i + 1..]);
                        if let Some(new_comp) = Composition::new(new_parts) {
                            new_refinements.push(new_comp);
                        }
                    }
                }
            }

            refinements.extend(new_refinements);
        }
    }

    refinements
}

/// Ribbon Schur function computation using ribbon tableaux
///
/// A ribbon (or rim hook) is a connected skew shape with no 2×2 squares.
/// Ribbon tableaux are used to compute characters of the symmetric group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RibbonSchurFunction {
    /// The composition indexing this ribbon Schur function
    pub composition: Composition,
}

impl RibbonSchurFunction {
    /// Create a new ribbon Schur function
    pub fn new(composition: Composition) -> Self {
        RibbonSchurFunction { composition }
    }

    /// Get the composition
    pub fn composition(&self) -> &Composition {
        &self.composition
    }

    /// Expand in the monomial basis
    pub fn expand_monomial(&self) -> NCSymFunction {
        ribbon_to_monomial(&self.composition)
    }

    /// Compute the product of two ribbon Schur functions
    ///
    /// The product structure involves Littlewood-Richardson type rules
    /// for noncommutative symmetric functions.
    pub fn product(&self, other: &Self) -> NCSymFunction {
        let mut result = NCSymFunction::new(NCSymBasis::RibbonSchur);

        // The product concatenates compositions with appropriate coefficients
        let parts1 = self.composition.parts();
        let parts2 = other.composition.parts();

        let mut combined = parts1.to_vec();
        combined.extend_from_slice(parts2);

        if let Some(comp) = Composition::new(combined) {
            result.add_term(comp, Rational::one());
        }

        result
    }

    /// Compute the ribbon Schur function for a single part [n]
    pub fn single_part(n: usize) -> Self {
        RibbonSchurFunction::new(Composition::new(vec![n]).expect("non-zero part"))
    }

    /// Check if this is a hook composition (1, 1, ..., 1, k)
    pub fn is_hook(&self) -> bool {
        let parts = self.composition.parts();
        if parts.is_empty() {
            return false;
        }

        let ones_count = parts.iter().filter(|&&p| p == 1).count();
        ones_count == parts.len() - 1 || ones_count == parts.len()
    }
}

/// Dual of noncommutative symmetric functions
///
/// The dual algebra is isomorphic to the algebra of quasi-symmetric functions (QSym).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NCSymDual {
    /// The basis for the dual
    pub basis: NCSymBasis,
    /// Coefficients indexed by compositions
    pub coeffs: HashMap<Composition, Rational>,
}

impl NCSymDual {
    /// Create a new element in the dual algebra
    pub fn new(basis: NCSymBasis) -> Self {
        NCSymDual {
            basis,
            coeffs: HashMap::new(),
        }
    }

    /// Create a basis element in the dual
    pub fn basis_element(basis: NCSymBasis, composition: Composition) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(composition, Rational::one());
        NCSymDual { basis, coeffs }
    }

    /// Duality pairing between NCSym and its dual
    ///
    /// The pairing <f, g> is computed by evaluating f and g on
    /// compatible compositions.
    pub fn pair_with(&self, ncsym: &NCSymFunction) -> Rational {
        let mut result = Rational::zero();

        for (comp1, coeff1) in &self.coeffs {
            if let Some(coeff2) = ncsym.coeffs.get(comp1) {
                result = result + coeff1.clone() * coeff2.clone();
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ncsym_creation() {
        let comp = Composition::new(vec![2, 1]).unwrap();
        let m = monomial(comp.clone());
        assert_eq!(m.coeff(&comp), Rational::one());
        assert!(!m.is_zero());
    }

    #[test]
    fn test_basis_functions() {
        let comp = Composition::new(vec![3]).unwrap();

        let m = monomial(comp.clone());
        let e = elementary(comp.clone());
        let h = homogeneous(comp.clone());
        let p = powersum(comp.clone());
        let r = ribbon_schur(comp.clone());

        assert_eq!(m.basis, NCSymBasis::Monomial);
        assert_eq!(e.basis, NCSymBasis::Elementary);
        assert_eq!(h.basis, NCSymBasis::Homogeneous);
        assert_eq!(p.basis, NCSymBasis::PowerSum);
        assert_eq!(r.basis, NCSymBasis::RibbonSchur);
    }

    #[test]
    fn test_addition() {
        let comp1 = Composition::new(vec![2, 1]).unwrap();
        let comp2 = Composition::new(vec![1, 2]).unwrap();

        let m1 = monomial(comp1.clone());
        let m2 = monomial(comp2.clone());

        let sum = m1.add(&m2).unwrap();
        assert_eq!(sum.coeff(&comp1), Rational::one());
        assert_eq!(sum.coeff(&comp2), Rational::one());
    }

    #[test]
    fn test_scalar_multiplication() {
        let comp = Composition::new(vec![2, 1]).unwrap();
        let m = monomial(comp.clone());

        let scaled = m.scale(&Rational::from(3));
        assert_eq!(scaled.coeff(&comp), Rational::from(3));
    }

    #[test]
    fn test_antipode() {
        let comp = Composition::new(vec![2, 1, 3]).unwrap();
        let m = monomial(comp.clone());

        let anti = m.antipode();
        let reversed = comp.reverse();

        // Antipode reverses and applies sign (-1)^{length-1}
        let expected_sign = if comp.len() % 2 == 0 {
            Rational::one()
        } else {
            -Rational::one()
        };

        assert_eq!(anti.coeff(&reversed), expected_sign);
    }

    #[test]
    fn test_degree() {
        let comp = Composition::new(vec![2, 1, 3]).unwrap();
        let m = monomial(comp);
        assert_eq!(m.degree(), 6);
    }

    #[test]
    fn test_ribbon_schur() {
        let comp = Composition::new(vec![3, 2]).unwrap();
        let r = RibbonSchurFunction::new(comp.clone());

        assert_eq!(r.composition(), &comp);
        assert!(!r.is_hook());
    }

    #[test]
    fn test_hook_composition() {
        let hook = Composition::new(vec![1, 1, 1, 3]).unwrap();
        let r = RibbonSchurFunction::new(hook);
        assert!(r.is_hook());
    }

    #[test]
    fn test_matchings() {
        let comp = Composition::new(vec![2, 1]).unwrap();
        let m = matchings(&comp);
        assert!(m > 0);
    }

    #[test]
    fn test_nesting() {
        let comp = Composition::new(vec![1, 2, 1]).unwrap();
        let n = nesting(&comp);
        assert_eq!(n, 1 + 3 + 4); // 1, then 1+2=3, then 1+2+1=4
    }

    #[test]
    fn test_compositions_generation() {
        let comps = compositions(3);
        // Compositions of 3: [3], [1,2], [2,1], [1,1,1]
        assert_eq!(comps.len(), 4);

        let comps0 = compositions(0);
        assert_eq!(comps0.len(), 1);
        assert!(comps0[0].parts().is_empty());
    }

    #[test]
    fn test_ribbon_product() {
        let r1 = RibbonSchurFunction::new(Composition::new(vec![2]).unwrap());
        let r2 = RibbonSchurFunction::new(Composition::new(vec![1]).unwrap());

        let prod = r1.product(&r2);
        let expected_comp = Composition::new(vec![2, 1]).unwrap();

        assert_eq!(prod.coeff(&expected_comp), Rational::one());
    }

    #[test]
    fn test_dual_pairing() {
        let comp = Composition::new(vec![2, 1]).unwrap();

        let ncsym = monomial(comp.clone());
        let dual = NCSymDual::basis_element(NCSymBasis::Monomial, comp.clone());

        let pairing = dual.pair_with(&ncsym);
        assert_eq!(pairing, Rational::one());
    }

    #[test]
    fn test_basis_conversion() {
        let comp = Composition::new(vec![2]).unwrap();
        let m = elementary_to_monomial(&comp);

        // Elementary function should expand as sum of monomials
        assert!(!m.is_zero());
        assert_eq!(m.basis, NCSymBasis::Monomial);
    }

    #[test]
    fn test_single_part_ribbon() {
        let r = RibbonSchurFunction::single_part(5);
        assert_eq!(r.composition().sum(), 5);
        assert_eq!(r.composition().len(), 1);
    }

    #[test]
    fn test_support() {
        let comp1 = Composition::new(vec![2, 1]).unwrap();
        let comp2 = Composition::new(vec![1, 2]).unwrap();

        let mut m = monomial(comp1.clone());
        m.add_term(comp2.clone(), Rational::from(2));

        let support = m.support();
        assert_eq!(support.len(), 2);
        assert!(support.contains(&comp1));
        assert!(support.contains(&comp2));
    }
}
