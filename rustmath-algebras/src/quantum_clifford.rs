//! Quantum Clifford Algebra
//!
//! Implements quantum Clifford algebras (q-Clifford algebras) of rank n and twist k
//! over a field F, with generators ψₐ, ψₐ†, ωₐ for a = 1,...,n.
//!
//! The algebra satisfies these fundamental relations:
//! - ωₐ ωᵦ = ωᵦ ωₐ (commutation)
//! - ψₐψᵦ + ψᵦψₐ = 0 (anticommutation)
//! - ωₐ⁴ᵏ = (1 + q⁻²ᵏ)ωₐ²ᵏ - q⁻²ᵏ (polynomial relation)
//!
//! Corresponds to sage.algebras.quantum_clifford

use rustmath_core::{Field, Ring};
use std::collections::HashMap;
use std::fmt::{self, Display};

/// Generator type in the quantum Clifford algebra
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CliffordGenerator {
    /// Fermionic generator ψₐ
    Psi(usize),
    /// Conjugate fermionic generator ψₐ†
    PsiDagger(usize),
    /// Bosonic generator ωₐ
    Omega(usize),
}

impl Display for CliffordGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CliffordGenerator::Psi(a) => write!(f, "ψ_{}", a),
            CliffordGenerator::PsiDagger(a) => write!(f, "ψ†_{}", a),
            CliffordGenerator::Omega(a) => write!(f, "ω_{}", a),
        }
    }
}

/// Fermionic index type
///
/// Represents the state of fermionic operators:
/// - MinusOne: ψₐ
/// - Zero: 1 (identity)
/// - One: ψₐ†
/// - Two: ψₐψₐ† (only in root-unity case)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FermionIndex {
    /// Represents ψₐ
    MinusOne,
    /// Represents identity
    Zero,
    /// Represents ψₐ†
    One,
    /// Represents ψₐψₐ† (root-unity only)
    Two,
}

impl FermionIndex {
    /// Convert to integer representation
    pub fn to_int(&self) -> i32 {
        match self {
            FermionIndex::MinusOne => -1,
            FermionIndex::Zero => 0,
            FermionIndex::One => 1,
            FermionIndex::Two => 2,
        }
    }

    /// Create from integer representation
    pub fn from_int(i: i32) -> Option<Self> {
        match i {
            -1 => Some(FermionIndex::MinusOne),
            0 => Some(FermionIndex::Zero),
            1 => Some(FermionIndex::One),
            2 => Some(FermionIndex::Two),
            _ => None,
        }
    }
}

/// An index in the quantum Clifford algebra basis
///
/// Represents a basis element as a product of fermionic and bosonic generators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CliffordIndex {
    /// Fermionic indices for each rank (one per generator)
    pub fermion_indices: Vec<FermionIndex>,
    /// Omega exponents for each rank
    pub omega_exponents: Vec<usize>,
}

impl CliffordIndex {
    /// Create a new Clifford index
    pub fn new(fermion_indices: Vec<FermionIndex>, omega_exponents: Vec<usize>) -> Self {
        CliffordIndex {
            fermion_indices,
            omega_exponents,
        }
    }

    /// Create the identity index for given rank
    pub fn identity(rank: usize) -> Self {
        CliffordIndex {
            fermion_indices: vec![FermionIndex::Zero; rank],
            omega_exponents: vec![0; rank],
        }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.fermion_indices.len()
    }

    /// Check if this is the identity
    pub fn is_identity(&self) -> bool {
        self.fermion_indices.iter().all(|&f| f == FermionIndex::Zero)
            && self.omega_exponents.iter().all(|&e| e == 0)
    }
}

impl Display for CliffordIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "1");
        }

        let mut parts = Vec::new();

        for (i, (&ferm, &omega)) in self
            .fermion_indices
            .iter()
            .zip(self.omega_exponents.iter())
            .enumerate()
        {
            match ferm {
                FermionIndex::MinusOne => parts.push(format!("ψ_{}", i + 1)),
                FermionIndex::One => parts.push(format!("ψ†_{}", i + 1)),
                FermionIndex::Two => parts.push(format!("(ψψ†)_{}", i + 1)),
                FermionIndex::Zero => {}
            }

            if omega > 0 {
                if omega == 1 {
                    parts.push(format!("ω_{}", i + 1));
                } else {
                    parts.push(format!("ω_{}^{}", i + 1, omega));
                }
            }
        }

        if parts.is_empty() {
            write!(f, "1")
        } else {
            write!(f, "{}", parts.join("·"))
        }
    }
}

/// An element of the quantum Clifford algebra
///
/// Represented as a linear combination of basis elements
#[derive(Debug, Clone)]
pub struct CliffordElement<F: Field> {
    /// Map from basis index to coefficient
    terms: HashMap<CliffordIndex, F>,
}

impl<F: Field> CliffordElement<F> {
    /// Create the zero element
    pub fn zero() -> Self {
        CliffordElement {
            terms: HashMap::new(),
        }
    }

    /// Create the one element (multiplicative identity)
    pub fn one(rank: usize) -> Self
    where
        F: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(CliffordIndex::identity(rank), F::from(1));
        CliffordElement { terms }
    }

    /// Create an element from a single basis index with coefficient
    pub fn from_basis(index: CliffordIndex, coeff: F) -> Self {
        if coeff.is_zero() {
            return CliffordElement::zero();
        }
        let mut terms = HashMap::new();
        terms.insert(index, coeff);
        CliffordElement { terms }
    }

    /// Add two elements
    pub fn add(&self, other: &CliffordElement<F>) -> CliffordElement<F> {
        let mut result = self.terms.clone();
        for (index, coeff) in &other.terms {
            let entry = result.entry(index.clone()).or_insert_with(F::zero);
            *entry = entry.add(coeff);
        }
        // Remove zero coefficients
        result.retain(|_, v| !v.is_zero());
        CliffordElement { terms: result }
    }

    /// Multiply by a scalar
    pub fn scale(&self, scalar: &F) -> CliffordElement<F> {
        if scalar.is_zero() {
            return CliffordElement::zero();
        }
        let terms: HashMap<_, _> = self
            .terms
            .iter()
            .map(|(idx, c)| (idx.clone(), c.mul(scalar)))
            .collect();
        CliffordElement { terms }
    }

    /// Negate the element
    pub fn negate(&self) -> CliffordElement<F> {
        let terms: HashMap<_, _> = self
            .terms
            .iter()
            .map(|(idx, c)| (idx.clone(), c.neg()))
            .collect();
        CliffordElement { terms }
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get the number of terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }
}

impl<F: Field + Display> Display for CliffordElement<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut items: Vec<_> = self.terms.iter().collect();
        items.sort_by(|(i1, _), (i2, _)| {
            // Sort by some canonical ordering
            i1.fermion_indices
                .cmp(&i2.fermion_indices)
                .then(i1.omega_exponents.cmp(&i2.omega_exponents))
        });

        for (i, (index, coeff)) in items.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            if index.is_identity() {
                write!(f, "{}", coeff)?;
            } else {
                write!(f, "{}*{}", coeff, index)?;
            }
        }
        Ok(())
    }
}

/// Quantum Clifford Algebra
///
/// A q-Clifford algebra of rank n and twist k over a field F,
/// with generators ψₐ, ψₐ†, ωₐ for a = 1,...,n.
///
/// # Type Parameters
///
/// * `F` - The base field (coefficients)
///
/// # Examples
///
/// ```
/// use rustmath_algebras::quantum_clifford::QuantumCliffordAlgebra;
/// use rustmath_rationals::Rational;
///
/// let qc: QuantumCliffordAlgebra<Rational> = QuantumCliffordAlgebra::new(2, 1);
/// ```
#[derive(Debug, Clone)]
pub struct QuantumCliffordAlgebra<F: Field> {
    /// The rank n (number of generators)
    rank: usize,
    /// The twist parameter k
    twist: usize,
    /// The q parameter
    q: F,
    /// Whether q^(2k) = 1 (root of unity case)
    is_root_unity: bool,
}

impl<F: Field + From<i64>> QuantumCliffordAlgebra<F> {
    /// Create a new quantum Clifford algebra
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank n (number of generators)
    /// * `twist` - The twist parameter k
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::quantum_clifford::QuantumCliffordAlgebra;
    /// use rustmath_rationals::Rational;
    ///
    /// let qc: QuantumCliffordAlgebra<Rational> = QuantumCliffordAlgebra::new(2, 1);
    /// ```
    pub fn new(rank: usize, twist: usize) -> Self {
        QuantumCliffordAlgebra {
            rank,
            twist,
            q: F::from(1),
            is_root_unity: false,
        }
    }

    /// Create a quantum Clifford algebra with a specific q parameter
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank n
    /// * `twist` - The twist parameter k
    /// * `q` - The q parameter
    pub fn with_q(rank: usize, twist: usize, q: F) -> Self {
        // TODO: Check if q^(2k) = 1
        QuantumCliffordAlgebra {
            rank,
            twist,
            q,
            is_root_unity: false,
        }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the twist parameter
    pub fn twist(&self) -> usize {
        self.twist
    }

    /// Get the q parameter
    pub fn q(&self) -> &F {
        &self.q
    }

    /// Check if this is the root-unity case
    pub fn is_root_unity(&self) -> bool {
        self.is_root_unity
    }

    /// Get the dimension of the algebra
    ///
    /// Returns (8k)^n for generic case
    pub fn dimension(&self) -> usize {
        let base = 8 * self.twist;
        base.pow(self.rank as u32)
    }

    /// Get a Psi generator
    pub fn psi(&self, index: usize) -> CliffordElement<F> {
        if index == 0 || index > self.rank {
            return CliffordElement::zero();
        }

        let mut fermion_indices = vec![FermionIndex::Zero; self.rank];
        fermion_indices[index - 1] = FermionIndex::MinusOne;
        let omega_exponents = vec![0; self.rank];

        CliffordElement::from_basis(
            CliffordIndex::new(fermion_indices, omega_exponents),
            F::from(1),
        )
    }

    /// Get a Psi-dagger generator
    pub fn psi_dagger(&self, index: usize) -> CliffordElement<F> {
        if index == 0 || index > self.rank {
            return CliffordElement::zero();
        }

        let mut fermion_indices = vec![FermionIndex::Zero; self.rank];
        fermion_indices[index - 1] = FermionIndex::One;
        let omega_exponents = vec![0; self.rank];

        CliffordElement::from_basis(
            CliffordIndex::new(fermion_indices, omega_exponents),
            F::from(1),
        )
    }

    /// Get an Omega generator
    pub fn omega(&self, index: usize) -> CliffordElement<F> {
        if index == 0 || index > self.rank {
            return CliffordElement::zero();
        }

        let fermion_indices = vec![FermionIndex::Zero; self.rank];
        let mut omega_exponents = vec![0; self.rank];
        omega_exponents[index - 1] = 1;

        CliffordElement::from_basis(
            CliffordIndex::new(fermion_indices, omega_exponents),
            F::from(1),
        )
    }

    /// Get all generators as a vector
    ///
    /// Returns [ψ₁, ..., ψₙ, ψ†₁, ..., ψ†ₙ, ω₁, ..., ωₙ]
    pub fn generators(&self) -> Vec<CliffordElement<F>> {
        let mut gens = Vec::new();

        // Psi generators
        for i in 1..=self.rank {
            gens.push(self.psi(i));
        }

        // Psi-dagger generators
        for i in 1..=self.rank {
            gens.push(self.psi_dagger(i));
        }

        // Omega generators
        for i in 1..=self.rank {
            gens.push(self.omega(i));
        }

        gens
    }

    /// Get the one element (identity)
    pub fn one(&self) -> CliffordElement<F> {
        CliffordElement::one(self.rank)
    }

    /// Get the zero element
    pub fn zero(&self) -> CliffordElement<F> {
        CliffordElement::zero()
    }
}

impl<F: Field + Display> Display for QuantumCliffordAlgebra<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Quantum Clifford algebra of rank {} and twist {} with q = {}",
            self.rank, self.twist, self.q
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_fermion_index() {
        assert_eq!(FermionIndex::Zero.to_int(), 0);
        assert_eq!(FermionIndex::One.to_int(), 1);
        assert_eq!(FermionIndex::MinusOne.to_int(), -1);
        assert_eq!(FermionIndex::Two.to_int(), 2);

        assert_eq!(FermionIndex::from_int(0), Some(FermionIndex::Zero));
        assert_eq!(FermionIndex::from_int(1), Some(FermionIndex::One));
        assert_eq!(FermionIndex::from_int(-1), Some(FermionIndex::MinusOne));
        assert_eq!(FermionIndex::from_int(3), None);
    }

    #[test]
    fn test_clifford_index() {
        let idx = CliffordIndex::identity(2);
        assert!(idx.is_identity());
        assert_eq!(idx.rank(), 2);

        let idx2 = CliffordIndex::new(
            vec![FermionIndex::One, FermionIndex::Zero],
            vec![1, 0],
        );
        assert!(!idx2.is_identity());
        assert_eq!(idx2.rank(), 2);
    }

    #[test]
    fn test_clifford_element() {
        let idx = CliffordIndex::identity(2);
        let elem: CliffordElement<Rational> = CliffordElement::from_basis(idx, Rational::from(1));
        assert_eq!(elem.num_terms(), 1);

        let zero: CliffordElement<Rational> = CliffordElement::zero();
        assert!(zero.is_zero());

        let one: CliffordElement<Rational> = CliffordElement::one(2);
        assert_eq!(one.num_terms(), 1);
    }

    #[test]
    fn test_quantum_clifford_algebra() {
        let qc: QuantumCliffordAlgebra<Rational> = QuantumCliffordAlgebra::new(2, 1);
        assert_eq!(qc.rank(), 2);
        assert_eq!(qc.twist(), 1);
        assert_eq!(qc.dimension(), 64); // (8*1)^2 = 64

        let psi1 = qc.psi(1);
        assert_eq!(psi1.num_terms(), 1);

        let psi_dag1 = qc.psi_dagger(1);
        assert_eq!(psi_dag1.num_terms(), 1);

        let omega1 = qc.omega(1);
        assert_eq!(omega1.num_terms(), 1);
    }

    #[test]
    fn test_generators() {
        let qc: QuantumCliffordAlgebra<Rational> = QuantumCliffordAlgebra::new(2, 1);
        let gens = qc.generators();
        assert_eq!(gens.len(), 6); // 2 psi + 2 psi† + 2 omega = 6
    }

    #[test]
    fn test_element_arithmetic() {
        let qc: QuantumCliffordAlgebra<Rational> = QuantumCliffordAlgebra::new(2, 1);
        let psi1 = qc.psi(1);
        let psi2 = qc.psi(2);

        let sum = psi1.add(&psi2);
        assert_eq!(sum.num_terms(), 2);

        let scaled = psi1.scale(&Rational::from(3));
        assert_eq!(scaled.num_terms(), 1);

        let neg = psi1.negate();
        assert_eq!(neg.num_terms(), 1);
    }

    #[test]
    fn test_invalid_indices() {
        let qc: QuantumCliffordAlgebra<Rational> = QuantumCliffordAlgebra::new(2, 1);

        let psi0 = qc.psi(0);
        assert!(psi0.is_zero());

        let psi3 = qc.psi(3);
        assert!(psi3.is_zero());
    }

    #[test]
    fn test_algebra_with_q() {
        let q = Rational::from(2);
        let qc = QuantumCliffordAlgebra::with_q(2, 1, q.clone());
        assert_eq!(qc.q(), &q);
        assert_eq!(qc.rank(), 2);
        assert_eq!(qc.twist(), 1);
    }

    #[test]
    fn test_generator_display() {
        let gen = CliffordGenerator::Psi(1);
        let display = format!("{}", gen);
        assert!(display.contains("ψ"));

        let gen_dag = CliffordGenerator::PsiDagger(2);
        let display_dag = format!("{}", gen_dag);
        assert!(display_dag.contains("ψ†"));

        let gen_omega = CliffordGenerator::Omega(3);
        let display_omega = format!("{}", gen_omega);
        assert!(display_omega.contains("ω"));
    }

    #[test]
    fn test_index_display() {
        let idx = CliffordIndex::identity(2);
        assert_eq!(format!("{}", idx), "1");

        let idx2 = CliffordIndex::new(
            vec![FermionIndex::One, FermionIndex::MinusOne],
            vec![0, 1],
        );
        let display = format!("{}", idx2);
        assert!(display.contains("ψ"));
    }

    #[test]
    fn test_different_ranks() {
        let qc1: QuantumCliffordAlgebra<Rational> = QuantumCliffordAlgebra::new(1, 1);
        assert_eq!(qc1.dimension(), 8);

        let qc3: QuantumCliffordAlgebra<Rational> = QuantumCliffordAlgebra::new(3, 1);
        assert_eq!(qc3.dimension(), 512); // 8^3
    }

    #[test]
    fn test_different_twists() {
        let qc_k1: QuantumCliffordAlgebra<Rational> = QuantumCliffordAlgebra::new(2, 1);
        assert_eq!(qc_k1.dimension(), 64); // (8*1)^2

        let qc_k2: QuantumCliffordAlgebra<Rational> = QuantumCliffordAlgebra::new(2, 2);
        assert_eq!(qc_k2.dimension(), 256); // (8*2)^2
    }
}
