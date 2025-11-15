//! Center of Universal Enveloping Algebra
//!
//! The center of the universal enveloping algebra U(g) consists of all elements
//! that commute with every element of the Lie algebra g. For a Lie algebra g,
//! the center Z(U(g)) is:
//!
//!     Z(U(g)) = {z ∈ U(g) | [z, x] = 0 for all x ∈ g}
//!
//! The center is a commutative graded algebra that plays a fundamental role
//! in representation theory, as central elements act as scalars in irreducible
//! representations.
//!
//! Corresponds to sage.algebras.lie_algebras.center_uea

use rustmath_core::{Ring, Field, MathError, Result};
use std::collections::{HashMap, BTreeMap, BTreeSet};
use std::fmt::{self, Display};
use std::ops::{Add, Mul};
use crate::poincare_birkhoff_witt::{PBWElement, PBWMonomial, PoincareBirkhoffWittBasis};
use crate::lie_algebra::{LieAlgebraBase, FinitelyGeneratedLieAlgebra};

/// Index for basis elements in the center of UEA
///
/// Indices are monomials in the central generators, represented as
/// exponent vectors
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CenterIndex {
    /// Exponents for each central generator
    /// Example: [2, 1, 0] means z_0^2 * z_1^1 * z_2^0
    exponents: Vec<usize>,
}

impl CenterIndex {
    /// Create a new index from exponents
    pub fn new(exponents: Vec<usize>) -> Self {
        Self { exponents }
    }

    /// The identity index (all zeros)
    pub fn identity(num_generators: usize) -> Self {
        Self {
            exponents: vec![0; num_generators],
        }
    }

    /// Total degree of this monomial
    pub fn degree(&self) -> usize {
        self.exponents.iter().sum()
    }

    /// Get the i-th exponent
    pub fn exponent(&self, i: usize) -> usize {
        self.exponents.get(i).copied().unwrap_or(0)
    }

    /// Number of generators
    pub fn num_generators(&self) -> usize {
        self.exponents.len()
    }

    /// Multiply two indices (add exponents)
    pub fn mul_index(&self, other: &Self) -> Result<Self> {
        if self.num_generators() != other.num_generators() {
            return Err(MathError::DimensionMismatch);
        }
        let exponents = self.exponents.iter()
            .zip(other.exponents.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self { exponents })
    }
}

impl Display for CenterIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.degree() == 0 {
            write!(f, "1")
        } else {
            let mut first = true;
            for (i, &exp) in self.exponents.iter().enumerate() {
                if exp > 0 {
                    if !first {
                        write!(f, "*")?;
                    }
                    write!(f, "z{}", i)?;
                    if exp > 1 {
                        write!(f, "^{}", exp)?;
                    }
                    first = false;
                }
            }
            Ok(())
        }
    }
}

/// Element of the center of the universal enveloping algebra
///
/// Represented as a linear combination of monomials in central generators
#[derive(Clone, Debug)]
pub struct CenterElement<F: Field> {
    /// Coefficients for each basis monomial
    terms: BTreeMap<CenterIndex, F>,
    /// Number of central generators available
    num_generators: usize,
}

impl<F: Field> CenterElement<F> {
    /// Create zero element
    pub fn zero(num_generators: usize) -> Self {
        Self {
            terms: BTreeMap::new(),
            num_generators,
        }
    }

    /// Create from a single term
    pub fn from_term(index: CenterIndex, coeff: F) -> Self {
        let num_generators = index.num_generators();
        let mut terms = BTreeMap::new();
        if !coeff.is_zero() {
            terms.insert(index, coeff);
        }
        Self { terms, num_generators }
    }

    /// Create the identity (scalar 1)
    pub fn one(num_generators: usize) -> Self {
        Self::from_term(CenterIndex::identity(num_generators), F::one())
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Check if this is the identity
    pub fn is_one(&self) -> bool {
        if self.terms.len() != 1 {
            return false;
        }
        let (idx, coeff) = self.terms.iter().next().unwrap();
        idx.degree() == 0 && *coeff == F::one()
    }

    /// Number of terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Iterator over terms
    pub fn terms(&self) -> impl Iterator<Item = (&CenterIndex, &F)> {
        self.terms.iter()
    }

    /// Get coefficient of a monomial
    pub fn coeff(&self, index: &CenterIndex) -> F {
        self.terms.get(index).cloned().unwrap_or_else(F::zero)
    }

    /// Add a term
    fn add_term(&mut self, index: CenterIndex, coeff: F) {
        if !coeff.is_zero() {
            *self.terms.entry(index).or_insert_with(F::zero) =
                self.terms.get(&index).cloned().unwrap_or_else(F::zero) + coeff;
            // Clean up zero coefficients
            if self.terms.get(&index).unwrap().is_zero() {
                self.terms.remove(&index);
            }
        }
    }

    /// Total degree (maximum degree of any term)
    pub fn degree(&self) -> usize {
        self.terms.keys().map(|idx| idx.degree()).max().unwrap_or(0)
    }
}

impl<F: Field> Add for CenterElement<F> {
    type Output = Result<Self>;

    fn add(self, other: Self) -> Self::Output {
        if self.num_generators != other.num_generators {
            return Err(MathError::InvalidOperation(
                "Cannot add center elements with different numbers of generators".into()
            ));
        }
        let mut result = self.clone();
        for (index, coeff) in other.terms {
            result.add_term(index, coeff);
        }
        Ok(result)
    }
}

impl<F: Field> Mul for CenterElement<F> {
    type Output = Result<Self>;

    fn mul(self, other: Self) -> Self::Output {
        if self.num_generators != other.num_generators {
            return Err(MathError::InvalidOperation(
                "Cannot multiply center elements with different numbers of generators".into()
            ));
        }
        let mut result = Self::zero(self.num_generators);
        for (idx1, coeff1) in &self.terms {
            for (idx2, coeff2) in &other.terms {
                let new_idx = idx1.mul_index(idx2)?;
                let new_coeff = coeff1.clone() * coeff2.clone();
                result.add_term(new_idx, new_coeff);
            }
        }
        Ok(result)
    }
}

impl<F: Field> Display for CenterElement<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        for (i, (index, coeff)) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            if index.degree() == 0 {
                write!(f, "{}", coeff)?;
            } else {
                write!(f, "{} {}", coeff, index)?;
            }
        }
        Ok(())
    }
}

/// The center of the universal enveloping algebra as a graded algebra
///
/// The center Z(U(g)) is a commutative algebra with basis elements
/// constructed incrementally by degree
pub struct CenterUEA<F: Field> {
    /// Dimension of the Lie algebra
    lie_algebra_dim: usize,
    /// Central generators and their degrees
    generators: Vec<(CenterIndex, usize)>,
    /// Lift map: center basis element -> UEA element
    /// Maps center indices to their representation in the UEA
    lift_map: HashMap<CenterIndex, PBWElement<F>>,
    /// Maximum degree computed so far
    max_degree_computed: usize,
}

impl<F: Field> CenterUEA<F> {
    /// Create a new center UEA
    pub fn new(lie_algebra_dim: usize) -> Self {
        Self {
            lie_algebra_dim,
            generators: Vec::new(),
            lift_map: HashMap::new(),
            max_degree_computed: 0,
        }
    }

    /// Get the Lie algebra dimension
    pub fn lie_algebra_dim(&self) -> usize {
        self.lie_algebra_dim
    }

    /// Number of central generators found so far
    pub fn num_generators(&self) -> usize {
        self.generators.len()
    }

    /// Get generator degrees
    pub fn generator_degrees(&self) -> Vec<usize> {
        self.generators.iter().map(|(_, deg)| *deg).collect()
    }

    /// The zero element
    pub fn zero(&self) -> CenterElement<F> {
        CenterElement::zero(self.num_generators())
    }

    /// The identity element (scalar 1)
    pub fn one(&self) -> CenterElement<F> {
        CenterElement::one(self.num_generators())
    }

    /// Construct central elements up to a given degree
    ///
    /// This incrementally builds the center by finding elements that
    /// commute with all Lie algebra generators
    pub fn construct_to_degree(&mut self, max_degree: usize) -> Result<()> {
        for degree in (self.max_degree_computed + 1)..=max_degree {
            self.construct_degree(degree)?;
        }
        self.max_degree_computed = max_degree;
        Ok(())
    }

    /// Construct central elements at a specific degree
    fn construct_degree(&mut self, degree: usize) -> Result<()> {
        // This is a simplified version. Full implementation would:
        // 1. Generate all PBW basis elements at this degree
        // 2. For each element z, compute [g_i, z] for all Lie generators g_i
        // 3. Find kernel elements (those that commute with all generators)
        // 4. Reduce by previously found central elements
        // 5. Add new central generators to the basis

        // For now, we just add a placeholder generator at each degree
        if self.generators.len() < degree {
            let new_gen_idx = CenterIndex::new(
                (0..degree).map(|i| if i == self.generators.len() { 1 } else { 0 }).collect()
            );
            self.generators.push((new_gen_idx, degree));
        }

        Ok(())
    }

    /// Lift a center element to the universal enveloping algebra
    pub fn lift(&self, element: &CenterElement<F>) -> Result<PBWElement<F>> {
        let mut result = PBWElement::zero();

        for (index, coeff) in element.terms() {
            // Look up the PBW representation of this basis element
            if let Some(pbw_elem) = self.lift_map.get(index) {
                let scaled = pbw_elem.clone().scale(coeff.clone());
                result = (result + scaled)?;
            } else {
                // If not in lift map, construct it
                // For now, return error; full implementation would build it
                return Err(MathError::InvalidOperation(
                    "Center basis element not yet constructed".into()
                ));
            }
        }

        Ok(result)
    }

    /// Attempt to express a UEA element as a center element (retraction)
    pub fn retract(&self, pbw_element: &PBWElement<F>) -> Option<CenterElement<F>> {
        // This would check if the element is central and express it in the center basis
        // For now, return None; full implementation would perform centrality test
        None
    }

    /// Check if a UEA element is central
    pub fn is_central(&self, pbw_element: &PBWElement<F>) -> bool {
        // Would need to compute commutators with all Lie generators
        // For now, return false as placeholder
        false
    }

    /// Get a central generator by index
    pub fn generator(&self, i: usize) -> Option<CenterElement<F>> {
        if i >= self.generators.len() {
            return None;
        }
        let (idx, _) = &self.generators[i];
        Some(CenterElement::from_term(idx.clone(), F::one()))
    }

    /// Get all generators as center elements
    pub fn algebra_generators(&self) -> Vec<CenterElement<F>> {
        self.generators.iter()
            .map(|(idx, _)| CenterElement::from_term(idx.clone(), F::one()))
            .collect()
    }
}

/// Specialized center for simple Lie algebras
///
/// For finite-dimensional simple Lie algebras in characteristic zero,
/// the center degrees are known from the Coxeter group structure
pub struct SimpleLieCenter<F: Field> {
    /// The base center
    center: CenterUEA<F>,
    /// Coxeter group degrees (determines generator degrees)
    coxeter_degrees: Vec<usize>,
}

impl<F: Field> SimpleLieCenter<F> {
    /// Create a center for a simple Lie algebra
    ///
    /// The coxeter_degrees determine the degrees of the central generators.
    /// For type A_n, these are [2, 3, ..., n+1]
    /// For type B_n, these are [2, 4, 6, ..., 2n]
    /// etc.
    pub fn new(coxeter_degrees: Vec<usize>) -> Self {
        let lie_dim = coxeter_degrees.len();
        let mut center = CenterUEA::new(lie_dim);

        // Pre-populate generators at known degrees
        for (i, &deg) in coxeter_degrees.iter().enumerate() {
            let mut exponents = vec![0; coxeter_degrees.len()];
            exponents[i] = 1;
            let idx = CenterIndex::new(exponents);
            center.generators.push((idx, deg));
        }

        Self {
            center,
            coxeter_degrees,
        }
    }

    /// Get the underlying center
    pub fn center(&self) -> &CenterUEA<F> {
        &self.center
    }

    /// Get a generator by index
    pub fn generator(&self, i: usize) -> Option<CenterElement<F>> {
        self.center.generator(i)
    }

    /// Coxeter number (sum of exponents + rank)
    pub fn coxeter_number(&self) -> usize {
        self.coxeter_degrees.iter().sum::<usize>() / self.coxeter_degrees.len()
    }

    /// All generator degrees
    pub fn generator_degrees(&self) -> &[usize] {
        &self.coxeter_degrees
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    type Q = Rational;

    #[test]
    fn test_center_index_creation() {
        let idx = CenterIndex::new(vec![2, 1, 0]);
        assert_eq!(idx.degree(), 3);
        assert_eq!(idx.exponent(0), 2);
        assert_eq!(idx.exponent(1), 1);
        assert_eq!(idx.exponent(2), 0);
        assert_eq!(idx.num_generators(), 3);
    }

    #[test]
    fn test_center_index_identity() {
        let idx = CenterIndex::identity(3);
        assert_eq!(idx.degree(), 0);
        assert_eq!(idx.num_generators(), 3);
    }

    #[test]
    fn test_center_index_multiplication() {
        let idx1 = CenterIndex::new(vec![2, 1]);
        let idx2 = CenterIndex::new(vec![1, 3]);
        let prod = idx1.mul_index(&idx2).unwrap();
        assert_eq!(prod.exponent(0), 3);
        assert_eq!(prod.exponent(1), 4);
        assert_eq!(prod.degree(), 7);
    }

    #[test]
    fn test_center_element_zero() {
        let z: CenterElement<Q> = CenterElement::zero(3);
        assert!(z.is_zero());
        assert_eq!(z.num_terms(), 0);
    }

    #[test]
    fn test_center_element_one() {
        let one: CenterElement<Q> = CenterElement::one(3);
        assert!(one.is_one());
        assert_eq!(one.num_terms(), 1);
        assert_eq!(one.degree(), 0);
    }

    #[test]
    fn test_center_element_from_term() {
        let idx = CenterIndex::new(vec![2, 1]);
        let elem: CenterElement<Q> = CenterElement::from_term(idx.clone(), Q::from(3));
        assert_eq!(elem.num_terms(), 1);
        assert_eq!(elem.coeff(&idx), Q::from(3));
        assert_eq!(elem.degree(), 3);
    }

    #[test]
    fn test_center_element_addition() {
        let idx1 = CenterIndex::new(vec![1, 0]);
        let idx2 = CenterIndex::new(vec![0, 1]);
        let e1: CenterElement<Q> = CenterElement::from_term(idx1.clone(), Q::from(2));
        let e2: CenterElement<Q> = CenterElement::from_term(idx2.clone(), Q::from(3));
        let sum = (e1 + e2).unwrap();
        assert_eq!(sum.num_terms(), 2);
        assert_eq!(sum.coeff(&idx1), Q::from(2));
        assert_eq!(sum.coeff(&idx2), Q::from(3));
    }

    #[test]
    fn test_center_element_multiplication() {
        let idx1 = CenterIndex::new(vec![1, 0]);
        let idx2 = CenterIndex::new(vec![0, 1]);
        let e1: CenterElement<Q> = CenterElement::from_term(idx1, Q::from(2));
        let e2: CenterElement<Q> = CenterElement::from_term(idx2, Q::from(3));
        let prod = (e1 * e2).unwrap();
        assert_eq!(prod.num_terms(), 1);
        let expected_idx = CenterIndex::new(vec![1, 1]);
        assert_eq!(prod.coeff(&expected_idx), Q::from(6));
    }

    #[test]
    fn test_center_uea_creation() {
        let center: CenterUEA<Q> = CenterUEA::new(3);
        assert_eq!(center.lie_algebra_dim(), 3);
        assert_eq!(center.num_generators(), 0);
    }

    #[test]
    fn test_center_uea_zero_one() {
        let center: CenterUEA<Q> = CenterUEA::new(3);
        let zero = center.zero();
        assert!(zero.is_zero());
        let one = center.one();
        assert!(one.is_one());
    }

    #[test]
    fn test_center_uea_construct() {
        let mut center: CenterUEA<Q> = CenterUEA::new(3);
        center.construct_to_degree(2).unwrap();
        assert_eq!(center.max_degree_computed, 2);
        assert!(center.num_generators() >= 1);
    }

    #[test]
    fn test_simple_lie_center_type_a() {
        // Type A_2 has Coxeter degrees [2, 3]
        let center: SimpleLieCenter<Q> = SimpleLieCenter::new(vec![2, 3]);
        assert_eq!(center.generator_degrees(), &[2, 3]);
        assert_eq!(center.coxeter_number(), 2); // (2+3)/2 = 2 (rounded)
    }

    #[test]
    fn test_simple_lie_center_generators() {
        let center: SimpleLieCenter<Q> = SimpleLieCenter::new(vec![2, 3, 4]);
        let gen0 = center.generator(0).unwrap();
        let gen1 = center.generator(1).unwrap();
        assert_eq!(gen0.num_terms(), 1);
        assert_eq!(gen1.num_terms(), 1);
    }

    #[test]
    fn test_center_index_display() {
        let idx = CenterIndex::new(vec![2, 1, 0]);
        let display = format!("{}", idx);
        assert!(display.contains("z0") && display.contains("z1"));
    }

    #[test]
    fn test_center_element_display() {
        let idx = CenterIndex::new(vec![1, 1]);
        let elem: CenterElement<Q> = CenterElement::from_term(idx, Q::from(5));
        let display = format!("{}", elem);
        assert!(display.contains("5"));
    }

    #[test]
    fn test_center_index_identity_display() {
        let idx = CenterIndex::identity(3);
        let display = format!("{}", idx);
        assert_eq!(display, "1");
    }

    #[test]
    fn test_center_element_zero_display() {
        let elem: CenterElement<Q> = CenterElement::zero(3);
        let display = format!("{}", elem);
        assert_eq!(display, "0");
    }
}
