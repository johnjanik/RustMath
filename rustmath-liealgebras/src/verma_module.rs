//! Verma Modules for Lie Algebras
//!
//! A Verma module M_λ is the fundamental object in the representation theory
//! of Kac-Moody Lie algebras. For a weight λ, it is defined as:
//!
//!     M_λ := U(g) ⊗_{U(b)} F_λ
//!
//! where U(g) is the universal enveloping algebra, b is a Borel subalgebra,
//! and F_λ is a one-dimensional U(b)-module.
//!
//! Verma modules have a canonical highest weight vector v_λ that is annihilated
//! by all positive root elements and satisfies h·v_λ = λ(h)v_λ for h in the
//! Cartan subalgebra.
//!
//! Corresponds to sage.algebras.lie_algebras.verma_module

use rustmath_core::{Ring, Field, MathError, Result};
use std::collections::{HashMap, BTreeMap};
use std::fmt::{self, Display};
use std::ops::{Add, Mul, Neg, Sub};
use crate::poincare_birkhoff_witt::{PBWElement, PBWMonomial, PoincareBirkhoffWittBasis};
use crate::lie_algebra::{LieAlgebraBase, FinitelyGeneratedLieAlgebra};
use crate::root_system::{Root, RootSystem};
use crate::weyl_group::{WeylGroup, WeylGroupElement};

/// A weight for a Lie algebra representation
///
/// Weights are linear functionals on the Cartan subalgebra,
/// represented as vectors in the dual space
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Weight<F: Field> {
    /// Coefficients in the basis dual to the simple roots
    coords: Vec<F>,
}

impl<F: Field> Weight<F> {
    /// Create a new weight from coordinates
    pub fn new(coords: Vec<F>) -> Self {
        Self { coords }
    }

    /// Zero weight
    pub fn zero(rank: usize) -> Self {
        Self {
            coords: vec![F::zero(); rank],
        }
    }

    /// Rank (dimension of the weight space)
    pub fn rank(&self) -> usize {
        self.coords.len()
    }

    /// Get the i-th coordinate
    pub fn coord(&self, i: usize) -> Result<&F> {
        self.coords.get(i).ok_or(MathError::IndexOutOfBounds)
    }

    /// Add two weights
    pub fn add_weight(&self, other: &Self) -> Result<Self> {
        if self.rank() != other.rank() {
            return Err(MathError::DimensionMismatch);
        }
        let coords = self.coords.iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        Ok(Self { coords })
    }

    /// Subtract two weights
    pub fn sub_weight(&self, other: &Self) -> Result<Self> {
        if self.rank() != other.rank() {
            return Err(MathError::DimensionMismatch);
        }
        let coords = self.coords.iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();
        Ok(Self { coords })
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: &F) -> Self {
        Self {
            coords: self.coords.iter().map(|x| x.clone() * scalar.clone()).collect(),
        }
    }

    /// Inner product with a root (assumes standard pairing)
    pub fn pair_with_root(&self, root: &Root) -> F {
        // This is a simplified version; actual implementation would use
        // the Cartan matrix and coroots
        let mut result = F::zero();
        for (i, coeff) in root.coordinates().iter().enumerate() {
            if i < self.coords.len() {
                result = result + (self.coords[i].clone() * F::from_i64(*coeff as i64).unwrap());
            }
        }
        result
    }
}

impl<F: Field> Display for Weight<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;
        for (i, coord) in self.coords.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coord)?;
        }
        write!(f, ")")
    }
}

/// Basis element in a Verma module
///
/// Elements are indexed by PBW monomials in U(g⁻), the negative part
/// of the universal enveloping algebra
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VermaModuleBasisElement {
    /// The PBW monomial from U(g⁻)
    monomial: PBWMonomial,
}

impl VermaModuleBasisElement {
    /// Create a new basis element
    pub fn new(monomial: PBWMonomial) -> Self {
        Self { monomial }
    }

    /// The identity element (highest weight vector)
    pub fn identity() -> Self {
        Self {
            monomial: PBWMonomial::identity(),
        }
    }

    /// Get the underlying PBW monomial
    pub fn monomial(&self) -> &PBWMonomial {
        &self.monomial
    }

    /// Degree of this basis element
    pub fn degree(&self) -> usize {
        self.monomial.degree()
    }
}

impl Display for VermaModuleBasisElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.monomial.is_identity() {
            write!(f, "v")
        } else {
            write!(f, "{}·v", self.monomial)
        }
    }
}

/// Element of a Verma module
///
/// Represented as a linear combination of basis elements
#[derive(Clone, Debug)]
pub struct VermaModuleElement<F: Field> {
    /// Coefficients for each basis element
    terms: BTreeMap<VermaModuleBasisElement, F>,
    /// The highest weight of this module
    highest_weight: Weight<F>,
}

impl<F: Field> VermaModuleElement<F> {
    /// Create zero element
    pub fn zero(highest_weight: Weight<F>) -> Self {
        Self {
            terms: BTreeMap::new(),
            highest_weight,
        }
    }

    /// Create from a single term
    pub fn from_term(basis: VermaModuleBasisElement, coeff: F, highest_weight: Weight<F>) -> Self {
        let mut terms = BTreeMap::new();
        if !coeff.is_zero() {
            terms.insert(basis, coeff);
        }
        Self { terms, highest_weight }
    }

    /// The highest weight vector v_λ
    pub fn highest_weight_vector(weight: Weight<F>) -> Self {
        Self::from_term(
            VermaModuleBasisElement::identity(),
            F::one(),
            weight,
        )
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get the highest weight
    pub fn highest_weight(&self) -> &Weight<F> {
        &self.highest_weight
    }

    /// Number of terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Iterator over terms
    pub fn terms(&self) -> impl Iterator<Item = (&VermaModuleBasisElement, &F)> {
        self.terms.iter()
    }

    /// Get coefficient of a basis element
    pub fn coeff(&self, basis: &VermaModuleBasisElement) -> F {
        self.terms.get(basis).cloned().unwrap_or_else(F::zero)
    }

    /// Add coefficient to a basis element
    fn add_term(&mut self, basis: VermaModuleBasisElement, coeff: F) {
        if !coeff.is_zero() {
            *self.terms.entry(basis).or_insert_with(F::zero) =
                self.terms.get(&basis).cloned().unwrap_or_else(F::zero) + coeff;
            // Clean up zero coefficients
            if self.terms.get(&basis).unwrap().is_zero() {
                self.terms.remove(&basis);
            }
        }
    }

    /// Compute the weight of this element (assuming homogeneous)
    /// Returns the weight relative to the highest weight
    pub fn weight(&self) -> Result<Weight<F>> {
        if self.is_zero() {
            return Err(MathError::InvalidOperation("Cannot compute weight of zero element".into()));
        }
        // All terms should have the same weight in a homogeneous element
        // The weight is highest_weight - (sum of root heights)
        // This is simplified; actual implementation would track roots properly
        Ok(self.highest_weight.clone())
    }
}

impl<F: Field> Add for VermaModuleElement<F> {
    type Output = Result<Self>;

    fn add(self, other: Self) -> Self::Output {
        if self.highest_weight != other.highest_weight {
            return Err(MathError::InvalidOperation(
                "Cannot add elements from different Verma modules".into()
            ));
        }
        let mut result = self.clone();
        for (basis, coeff) in other.terms {
            result.add_term(basis, coeff);
        }
        Ok(result)
    }
}

impl<F: Field> Neg for VermaModuleElement<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            terms: self.terms.into_iter()
                .map(|(basis, coeff)| (basis, -coeff))
                .collect(),
            highest_weight: self.highest_weight,
        }
    }
}

impl<F: Field> Display for VermaModuleElement<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        for (i, (basis, coeff)) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{} {}", coeff, basis)?;
        }
        Ok(())
    }
}

/// A Verma module M(λ) for a Lie algebra
///
/// The Verma module is the induced module from a highest weight representation
/// of the Borel subalgebra. It is a free U(g⁻)-module generated by a highest
/// weight vector v_λ.
pub struct VermaModule<F: Field> {
    /// The highest weight λ
    highest_weight: Weight<F>,
    /// Rank of the Lie algebra
    rank: usize,
    /// Root system of the Lie algebra
    root_system: Option<RootSystem>,
    /// PBW basis for the universal enveloping algebra
    pbw_basis: Option<PoincareBirkhoffWittBasis<F>>,
}

impl<F: Field> VermaModule<F> {
    /// Create a new Verma module with highest weight λ
    pub fn new(highest_weight: Weight<F>, rank: usize) -> Self {
        Self {
            highest_weight,
            rank,
            root_system: None,
            pbw_basis: None,
        }
    }

    /// Create a Verma module with a root system
    pub fn with_root_system(highest_weight: Weight<F>, root_system: RootSystem) -> Self {
        let rank = root_system.rank();
        Self {
            highest_weight,
            rank,
            root_system: Some(root_system),
            pbw_basis: None,
        }
    }

    /// Get the highest weight
    pub fn highest_weight(&self) -> &Weight<F> {
        &self.highest_weight
    }

    /// Rank of the underlying Lie algebra
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The highest weight vector v_λ
    pub fn highest_weight_vector(&self) -> VermaModuleElement<F> {
        VermaModuleElement::highest_weight_vector(self.highest_weight.clone())
    }

    /// Zero element
    pub fn zero(&self) -> VermaModuleElement<F> {
        VermaModuleElement::zero(self.highest_weight.clone())
    }

    /// Construct a homogeneous component basis at a given depth
    ///
    /// Returns basis elements obtained by applying depth-d monomials
    /// in U(g⁻) to the highest weight vector
    pub fn homogeneous_component_basis(&self, depth: usize) -> Vec<VermaModuleBasisElement> {
        if depth == 0 {
            return vec![VermaModuleBasisElement::identity()];
        }

        // Generate all PBW monomials of the given degree in negative roots
        // This is a simplified version; actual implementation would enumerate
        // all monomials of total degree = depth
        let mut basis = Vec::new();

        // For now, we create basis elements corresponding to products of
        // simple root vectors. In a full implementation, this would enumerate
        // all partitions and compositions.
        if depth == 1 {
            for i in 0..self.rank {
                let monomial = PBWMonomial::from_generator(i);
                basis.push(VermaModuleBasisElement::new(monomial));
            }
        }
        // For higher depths, we would recursively build up the basis

        basis
    }

    /// Check if the module is simple (irreducible)
    ///
    /// A Verma module M(λ) is simple if and only if λ is antidominant,
    /// meaning ⟨λ+ρ, α^∨⟩ ∉ ℤ_{>0} for all positive roots α
    pub fn is_simple(&self) -> bool {
        // This requires checking the weight against the positive roots
        // For now, return false as a placeholder
        // Full implementation would check the antidominance condition
        false
    }

    /// Check if the module is projective in category O
    ///
    /// M(λ) is projective if λ is dominant (Verma dominant)
    pub fn is_projective(&self) -> bool {
        // Requires checking dominance condition
        // Placeholder for now
        false
    }

    /// Check if the weight is singular
    ///
    /// A weight is singular if it has no dominant element in its dot orbit
    pub fn is_singular(&self) -> bool {
        // Requires Weyl group action
        // Placeholder for now
        false
    }

    /// Compute the dimension of the homogeneous component at depth d
    ///
    /// This is the Kostant partition function p(d) counting the number
    /// of ways to write d as a sum of positive roots
    pub fn dimension_at_depth(&self, depth: usize) -> usize {
        // This is the Kostant partition function
        // For now, use a simple formula; full implementation would
        // use generating functions or recursion
        if depth == 0 {
            1
        } else {
            // Approximate with binomial coefficient for simplicity
            // Actual formula involves the Kostant partition function
            let n = self.rank;
            if depth <= n {
                (1..=depth).product::<usize>()
            } else {
                // Very rough approximation
                depth.pow(n as u32)
            }
        }
    }
}

/// Morphism between Verma modules
///
/// A morphism M(w·λ) → M(w'·λ') exists (and is unique) if and only if
/// λ = λ' and w' ≤ w in the Bruhat order of the Weyl group
pub struct VermaModuleHomomorphism<F: Field> {
    /// Source Verma module weight
    source_weight: Weight<F>,
    /// Target Verma module weight
    target_weight: Weight<F>,
    /// Weyl group element relating the weights (if applicable)
    weyl_element: Option<WeylGroupElement>,
}

impl<F: Field> VermaModuleHomomorphism<F> {
    /// Create a morphism between Verma modules
    ///
    /// Returns None if no morphism exists (weights incomparable)
    pub fn new(
        source_weight: Weight<F>,
        target_weight: Weight<F>,
        weyl_element: Option<WeylGroupElement>,
    ) -> Option<Self> {
        // Check if morphism exists based on Bruhat order
        // For now, always return Some; full implementation would check
        Some(Self {
            source_weight,
            target_weight,
            weyl_element,
        })
    }

    /// Check if this is a zero morphism
    pub fn is_zero(&self) -> bool {
        // Morphism is zero if weights are incomparable in Bruhat order
        false
    }

    /// Apply the morphism to an element
    pub fn apply(&self, element: &VermaModuleElement<F>) -> Result<VermaModuleElement<F>> {
        if element.highest_weight() != &self.source_weight {
            return Err(MathError::InvalidOperation(
                "Element not from source module".into()
            ));
        }

        // The morphism sends the highest weight vector to a singular vector
        // in the target module. For now, just create zero.
        Ok(VermaModuleElement::zero(self.target_weight.clone()))
    }

    /// Compute the dimension of the homomorphism space
    ///
    /// Returns 1 if a non-zero morphism exists, 0 otherwise
    pub fn dimension() -> usize {
        // dim Hom(M(w·λ), M(w'·λ)) = 1 if w' ≤ w in Bruhat order, 0 otherwise
        // For now, return 1 as placeholder
        1
    }
}

/// Homset of morphisms between Verma modules
///
/// For weights λ, λ', the homset Hom(M(λ), M(λ')) has dimension 0 or 1
pub struct VermaModuleHomset<F: Field> {
    source_weight: Weight<F>,
    target_weight: Weight<F>,
}

impl<F: Field> VermaModuleHomset<F> {
    /// Create a new homset
    pub fn new(source_weight: Weight<F>, target_weight: Weight<F>) -> Self {
        Self {
            source_weight,
            target_weight,
        }
    }

    /// Dimension of the homset
    pub fn dimension(&self) -> usize {
        // Check Bruhat order condition
        // For now, placeholder
        0
    }

    /// Check if the homset is trivial (contains only zero map)
    pub fn is_trivial(&self) -> bool {
        self.dimension() == 0
    }

    /// Get the canonical morphism (if it exists)
    pub fn canonical_morphism(&self) -> Option<VermaModuleHomomorphism<F>> {
        if self.is_trivial() {
            None
        } else {
            VermaModuleHomomorphism::new(
                self.source_weight.clone(),
                self.target_weight.clone(),
                None,
            )
        }
    }
}

/// Contravariant form on a Verma module
///
/// The contravariant form is a symmetric bilinear pairing on M(λ)
/// defined by ⟨v_λ, v_λ⟩ = 1 and ⟨x·u, v⟩ = ⟨u, ω(x)·v⟩
/// where ω is the Cartan involution
pub struct ContravariantForm<F: Field> {
    highest_weight: Weight<F>,
}

impl<F: Field> ContravariantForm<F> {
    /// Create a contravariant form for a Verma module
    pub fn new(highest_weight: Weight<F>) -> Self {
        Self { highest_weight }
    }

    /// Evaluate the form on two elements
    pub fn eval(
        &self,
        x: &VermaModuleElement<F>,
        y: &VermaModuleElement<F>,
    ) -> Result<F> {
        if x.highest_weight() != &self.highest_weight || y.highest_weight() != &self.highest_weight {
            return Err(MathError::InvalidOperation(
                "Elements must be from the same Verma module".into()
            ));
        }

        // The form is computed using the PBW basis
        // For now, return zero as placeholder
        Ok(F::zero())
    }

    /// Check if the form is degenerate
    ///
    /// The form is degenerate if and only if the module is reducible
    pub fn is_degenerate(&self) -> bool {
        // Related to whether the module is simple
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    type Q = Rational;

    #[test]
    fn test_weight_creation() {
        let w = Weight::<Q>::new(vec![
            Q::from(1),
            Q::from(2),
            Q::from(3),
        ]);
        assert_eq!(w.rank(), 3);
        assert_eq!(*w.coord(0).unwrap(), Q::from(1));
        assert_eq!(*w.coord(1).unwrap(), Q::from(2));
        assert_eq!(*w.coord(2).unwrap(), Q::from(3));
    }

    #[test]
    fn test_weight_zero() {
        let w = Weight::<Q>::zero(3);
        assert_eq!(w.rank(), 3);
        for i in 0..3 {
            assert_eq!(*w.coord(i).unwrap(), Q::zero());
        }
    }

    #[test]
    fn test_weight_addition() {
        let w1 = Weight::<Q>::new(vec![Q::from(1), Q::from(2)]);
        let w2 = Weight::<Q>::new(vec![Q::from(3), Q::from(4)]);
        let sum = w1.add_weight(&w2).unwrap();
        assert_eq!(*sum.coord(0).unwrap(), Q::from(4));
        assert_eq!(*sum.coord(1).unwrap(), Q::from(6));
    }

    #[test]
    fn test_weight_scaling() {
        let w = Weight::<Q>::new(vec![Q::from(2), Q::from(4)]);
        let scaled = w.scale(&Q::from(3));
        assert_eq!(*scaled.coord(0).unwrap(), Q::from(6));
        assert_eq!(*scaled.coord(1).unwrap(), Q::from(12));
    }

    #[test]
    fn test_verma_module_creation() {
        let weight = Weight::<Q>::new(vec![Q::from(1), Q::from(2), Q::from(3)]);
        let verma = VermaModule::new(weight.clone(), 3);
        assert_eq!(verma.rank(), 3);
        assert_eq!(verma.highest_weight(), &weight);
    }

    #[test]
    fn test_highest_weight_vector() {
        let weight = Weight::<Q>::new(vec![Q::from(1), Q::from(2)]);
        let verma = VermaModule::new(weight.clone(), 2);
        let v = verma.highest_weight_vector();
        assert!(!v.is_zero());
        assert_eq!(v.num_terms(), 1);
        assert_eq!(v.highest_weight(), &weight);
    }

    #[test]
    fn test_zero_element() {
        let weight = Weight::<Q>::new(vec![Q::from(1)]);
        let verma = VermaModule::new(weight, 1);
        let zero = verma.zero();
        assert!(zero.is_zero());
        assert_eq!(zero.num_terms(), 0);
    }

    #[test]
    fn test_element_addition() {
        let weight = Weight::<Q>::new(vec![Q::from(1)]);
        let v1 = VermaModuleElement::from_term(
            VermaModuleBasisElement::identity(),
            Q::from(2),
            weight.clone(),
        );
        let v2 = VermaModuleElement::from_term(
            VermaModuleBasisElement::identity(),
            Q::from(3),
            weight,
        );
        let sum = (v1 + v2).unwrap();
        assert_eq!(sum.coeff(&VermaModuleBasisElement::identity()), Q::from(5));
    }

    #[test]
    fn test_element_negation() {
        let weight = Weight::<Q>::new(vec![Q::from(1)]);
        let v = VermaModuleElement::from_term(
            VermaModuleBasisElement::identity(),
            Q::from(5),
            weight,
        );
        let neg_v = -v;
        assert_eq!(neg_v.coeff(&VermaModuleBasisElement::identity()), Q::from(-5));
    }

    #[test]
    fn test_homogeneous_basis_depth_0() {
        let weight = Weight::<Q>::new(vec![Q::from(1), Q::from(2)]);
        let verma = VermaModule::new(weight, 2);
        let basis = verma.homogeneous_component_basis(0);
        assert_eq!(basis.len(), 1);
        assert!(basis[0].monomial().is_identity());
    }

    #[test]
    fn test_homogeneous_basis_depth_1() {
        let weight = Weight::<Q>::new(vec![Q::from(1), Q::from(2)]);
        let verma = VermaModule::new(weight, 2);
        let basis = verma.homogeneous_component_basis(1);
        assert_eq!(basis.len(), 2); // Two simple roots
    }

    #[test]
    fn test_dimension_at_depth() {
        let weight = Weight::<Q>::new(vec![Q::from(1)]);
        let verma = VermaModule::new(weight, 1);
        assert_eq!(verma.dimension_at_depth(0), 1);
        assert!(verma.dimension_at_depth(1) > 0);
    }

    #[test]
    fn test_contravariant_form_creation() {
        let weight = Weight::<Q>::new(vec![Q::from(1), Q::from(2)]);
        let form = ContravariantForm::new(weight.clone());
        let v = VermaModuleElement::highest_weight_vector(weight);
        let result = form.eval(&v, &v);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verma_homomorphism_creation() {
        let w1 = Weight::<Q>::new(vec![Q::from(1)]);
        let w2 = Weight::<Q>::new(vec![Q::from(2)]);
        let hom = VermaModuleHomomorphism::new(w1, w2, None);
        assert!(hom.is_some());
    }

    #[test]
    fn test_homset_creation() {
        let w1 = Weight::<Q>::new(vec![Q::from(1)]);
        let w2 = Weight::<Q>::new(vec![Q::from(2)]);
        let homset = VermaModuleHomset::new(w1, w2);
        assert_eq!(homset.dimension(), 0); // Placeholder behavior
    }

    #[test]
    fn test_basis_element_display() {
        let basis = VermaModuleBasisElement::identity();
        let display = format!("{}", basis);
        assert!(display.contains("v"));
    }

    #[test]
    fn test_weight_display() {
        let w = Weight::<Q>::new(vec![Q::from(1), Q::from(2)]);
        let display = format!("{}", w);
        assert!(display.contains("1"));
        assert!(display.contains("2"));
    }
}
