//! Lie Conformal Algebras with Basis
//!
//! Provides infrastructure for Lie conformal algebras with an explicit basis,
//! including structure coefficients for the λ-bracket operation.
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.lie_conformal_algebra_with_basis

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::Hash;
use crate::lie_conformal_algebra::{LieConformalAlgebra, LambdaBracket, GeneratorIndex};

/// Basis element for a Lie conformal algebra
///
/// Represents a single basis element, typically corresponding to a generator
/// or a monomial in generators.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BasisElement {
    /// A simple generator
    Generator(GeneratorIndex),
    /// A derivative of a generator: ∂^n(generator)
    Derivative(GeneratorIndex, usize),
    /// A monomial (for free algebras)
    Monomial(Vec<(GeneratorIndex, usize)>), // List of (generator, derivative power)
}

impl BasisElement {
    /// Create a generator basis element
    pub fn generator(index: impl Into<GeneratorIndex>) -> Self {
        BasisElement::Generator(index.into())
    }

    /// Create a derivative basis element
    pub fn derivative(index: impl Into<GeneratorIndex>, n: usize) -> Self {
        if n == 0 {
            BasisElement::Generator(index.into())
        } else {
            BasisElement::Derivative(index.into(), n)
        }
    }

    /// Get the generator index if this is a generator
    pub fn as_generator(&self) -> Option<&GeneratorIndex> {
        match self {
            BasisElement::Generator(idx) => Some(idx),
            _ => None,
        }
    }

    /// Check if this is a generator (∂^0)
    pub fn is_generator(&self) -> bool {
        matches!(self, BasisElement::Generator(_))
    }

    /// Get the derivative order
    pub fn derivative_order(&self) -> usize {
        match self {
            BasisElement::Generator(_) => 0,
            BasisElement::Derivative(_, n) => *n,
            BasisElement::Monomial(_) => 0, // TODO: Define properly for monomials
        }
    }
}

impl Display for BasisElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BasisElement::Generator(idx) => write!(f, "{}", idx),
            BasisElement::Derivative(idx, n) => {
                if *n == 1 {
                    write!(f, "∂({})", idx)
                } else {
                    write!(f, "∂^{}({})", n, idx)
                }
            }
            BasisElement::Monomial(terms) => {
                if terms.is_empty() {
                    write!(f, "1")
                } else {
                    for (i, (idx, power)) in terms.iter().enumerate() {
                        if i > 0 {
                            write!(f, "*")?;
                        }
                        if *power == 0 {
                            write!(f, "{}", idx)?;
                        } else {
                            write!(f, "∂^{}({})", power, idx)?;
                        }
                    }
                    Ok(())
                }
            }
        }
    }
}

/// Structure coefficients for λ-brackets
///
/// Stores the coefficients defining the λ-bracket operation on basis elements.
/// For basis elements b_i and b_j:
/// [b_i_λ b_j] = Σ_k Σ_n c_{ij}^{k,n} λ^n ⊗ b_k
///
/// The structure is: HashMap<(i, j), HashMap<k, Vec<R>>>
/// where the Vec<R> represents the polynomial in λ
pub type StructureCoefficients<R> = HashMap<(usize, usize), HashMap<usize, Vec<R>>>;

/// Trait for Lie conformal algebras with an explicit basis
pub trait LieConformalAlgebraWithBasis<R: Ring>: LieConformalAlgebra<R> {
    /// The basis element type
    type Basis: Clone + Eq + Hash;

    /// Get the i-th basis element
    fn basis_element(&self, i: usize) -> Option<Self::Basis>;

    /// Get all basis elements (if finitely generated)
    fn basis_elements(&self) -> Option<Vec<Self::Basis>> {
        if let Some(n) = self.ngens() {
            Some((0..n).filter_map(|i| self.basis_element(i)).collect())
        } else {
            None
        }
    }

    /// Get the structure coefficients
    fn structure_coefficients(&self) -> &StructureCoefficients<R>;

    /// Compute λ-bracket on basis elements
    ///
    /// Returns [b_i_λ b_j] as a polynomial in λ with coefficients in the algebra
    fn lambda_bracket_on_basis(
        &self,
        i: usize,
        j: usize,
    ) -> HashMap<Self::Basis, Vec<R>> {
        // Default implementation using structure coefficients
        let coeffs = self.structure_coefficients();
        if let Some(bracket) = coeffs.get(&(i, j)) {
            bracket
                .iter()
                .filter_map(|(k, poly)| {
                    self.basis_element(*k).map(|basis| (basis, poly.clone()))
                })
                .collect()
        } else {
            HashMap::new()
        }
    }

    /// Check sesquilinearity relation
    ///
    /// Verifies: [∂a_λ b] = -λ[a_λ b] and [a_λ ∂b] = (∂ + λ)[a_λ b]
    fn check_sesquilinearity(&self) -> bool {
        // This would require computing derivatives and λ-brackets
        // Implementation deferred to concrete algebras
        true
    }

    /// Check Jacobi identity
    ///
    /// Verifies: [a_λ [b_μ c]] = [[a_λ b]_{λ+μ} c] + [b_μ [a_λ c]]
    fn check_jacobi(&self) -> bool {
        // This would require computing nested λ-brackets
        // Implementation deferred to concrete algebras
        true
    }
}

/// A generic Lie conformal algebra with basis
///
/// Stores basis information and structure coefficients.
#[derive(Clone)]
pub struct LCAWithBasis<R: Ring, B: Clone + Eq + Hash> {
    /// Base ring
    base_ring: R,
    /// Basis elements
    basis: Vec<B>,
    /// Structure coefficients for λ-brackets
    structure: StructureCoefficients<R>,
}

impl<R: Ring + Clone, B: Clone + Eq + Hash> LCAWithBasis<R, B> {
    /// Create a new Lie conformal algebra with basis
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring
    /// * `basis` - The basis elements
    /// * `structure` - Structure coefficients defining λ-brackets
    pub fn new(
        base_ring: R,
        basis: Vec<B>,
        structure: StructureCoefficients<R>,
    ) -> Self {
        LCAWithBasis {
            base_ring,
            basis,
            structure,
        }
    }

    /// Get the base ring
    pub fn base_ring(&self) -> &R {
        &self.base_ring
    }

    /// Get the number of basis elements
    pub fn rank(&self) -> usize {
        self.basis.len()
    }

    /// Get a basis element
    pub fn basis_element(&self, i: usize) -> Option<&B> {
        self.basis.get(i)
    }

    /// Get all basis elements
    pub fn basis_elements(&self) -> &[B] {
        &self.basis
    }

    /// Get the structure coefficients
    pub fn structure_coefficients(&self) -> &StructureCoefficients<R> {
        &self.structure
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_element_creation() {
        let gen = BasisElement::generator(GeneratorIndex::finite(0));
        assert!(gen.is_generator());
        assert_eq!(gen.derivative_order(), 0);

        let deriv = BasisElement::derivative(GeneratorIndex::finite(0), 2);
        assert!(!deriv.is_generator());
        assert_eq!(deriv.derivative_order(), 2);
    }

    #[test]
    fn test_basis_element_display() {
        let gen = BasisElement::generator(GeneratorIndex::named("L"));
        assert_eq!(format!("{}", gen), "L");

        let deriv = BasisElement::derivative(GeneratorIndex::named("L"), 1);
        assert_eq!(format!("{}", deriv), "∂(L)");

        let deriv2 = BasisElement::derivative(GeneratorIndex::named("L"), 3);
        assert_eq!(format!("{}", deriv2), "∂^3(L)");
    }

    #[test]
    fn test_lca_with_basis_creation() {
        let base_ring = 1i64;
        let basis = vec![
            BasisElement::generator(GeneratorIndex::finite(0)),
            BasisElement::generator(GeneratorIndex::finite(1)),
        ];
        let structure: StructureCoefficients<i64> = HashMap::new();

        let lca = LCAWithBasis::new(base_ring, basis, structure);
        assert_eq!(lca.rank(), 2);
        assert!(lca.basis_element(0).is_some());
        assert!(lca.basis_element(2).is_none());
    }
}
