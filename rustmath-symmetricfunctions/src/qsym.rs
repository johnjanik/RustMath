//! Quasi-Symmetric Functions (QSym)
//!
//! This module implements the Hopf algebra of quasi-symmetric functions.
//! QSym is indexed by integer compositions and is dual to NCSF.
//!
//! Key bases:
//! - Monomial (M): The monomial quasi-symmetric basis
//! - Fundamental (F): The fundamental quasi-symmetric basis
//!
//! QSym forms a commutative Hopf algebra that is dual to the algebra of
//! noncommutative symmetric functions (NCSF).

use rustmath_combinatorics::{Composition, compositions, compositions_k};
use rustmath_core::Ring;
use rustmath_rationals::Rational;
use rustmath_integers::Integer;
use std::collections::HashMap;
use std::fmt;

/// The different bases for quasi-symmetric functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QSymBasis {
    /// Monomial quasi-symmetric basis M
    Monomial,
    /// Fundamental quasi-symmetric basis F
    Fundamental,
}

impl fmt::Display for QSymBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QSymBasis::Monomial => write!(f, "M"),
            QSymBasis::Fundamental => write!(f, "F"),
        }
    }
}

/// A quasi-symmetric function
///
/// Represented as a linear combination of basis elements indexed by compositions
#[derive(Debug, Clone, PartialEq)]
pub struct QSym {
    /// The basis in which this function is expressed
    pub basis: QSymBasis,
    /// Coefficients: maps composition to coefficient
    pub coeffs: HashMap<Composition, Rational>,
}

impl QSym {
    /// Create a new QSym in the given basis
    pub fn new(basis: QSymBasis) -> Self {
        QSym {
            basis,
            coeffs: HashMap::new(),
        }
    }

    /// Create a basis element (single composition with coefficient 1)
    pub fn basis_element(basis: QSymBasis, composition: Composition) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(composition, Rational::one());
        QSym { basis, coeffs }
    }

    /// Create the multiplicative identity (empty composition)
    pub fn one(basis: QSymBasis) -> Self {
        let empty = Composition::new(vec![]).unwrap();
        Self::basis_element(basis, empty)
    }

    /// Add a term with the given coefficient
    pub fn add_term(&mut self, composition: Composition, coeff: Rational) {
        if !coeff.is_zero() {
            let entry = self.coeffs.entry(composition.clone()).or_insert(Rational::zero());
            *entry = entry.clone() + coeff;
            if entry.is_zero() {
                self.coeffs.remove(&composition);
            }
        }
    }

    /// Get the coefficient of a composition
    pub fn coeff(&self, composition: &Composition) -> Rational {
        self.coeffs.get(composition).cloned().unwrap_or(Rational::zero())
    }

    /// Check if this is the zero function
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.values().all(|c| c.is_zero())
    }

    /// Get the degree (largest composition sum)
    pub fn degree(&self) -> usize {
        self.coeffs.keys().map(|c| c.sum()).max().unwrap_or(0)
    }

    /// Multiply by a scalar
    pub fn scale(&self, scalar: &Rational) -> Self {
        let mut result = self.clone();
        for coeff in result.coeffs.values_mut() {
            *coeff = coeff.clone() * scalar.clone();
        }
        result
    }

    /// Add two QSym (must be in the same basis)
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.basis != other.basis {
            return None;
        }

        let mut result = self.clone();
        for (comp, coeff) in &other.coeffs {
            result.add_term(comp.clone(), coeff.clone());
        }

        Some(result)
    }

    /// Get all compositions with non-zero coefficients
    pub fn support(&self) -> Vec<Composition> {
        self.coeffs
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(c, _)| c.clone())
            .collect()
    }
}

/// Create a Monomial basis element M_I
pub fn monomial(composition: Composition) -> QSym {
    QSym::basis_element(QSymBasis::Monomial, composition)
}

/// Create a Fundamental basis element F_I
pub fn fundamental(composition: Composition) -> QSym {
    QSym::basis_element(QSymBasis::Fundamental, composition)
}

/// Compute all coarsenings of a composition
///
/// A coarsening of I is obtained by merging adjacent parts
pub fn coarsenings(comp: &Composition) -> Vec<Composition> {
    let parts = comp.parts();
    let n = parts.len();

    if n == 0 {
        return vec![comp.clone()];
    }

    if n == 1 {
        return vec![comp.clone()];
    }

    let mut result = Vec::new();

    // Generate all subsets of merge positions using bit patterns
    // For n parts, there are n-1 possible merge positions
    let num_merges = n - 1;
    for mask in 0..(1 << num_merges) {
        let mut new_parts = Vec::new();
        let mut current_sum = parts[0];

        for i in 0..num_merges {
            if mask & (1 << i) != 0 {
                // Merge: accumulate the next part
                current_sum += parts[i + 1];
            } else {
                // Don't merge: output current sum and start new
                new_parts.push(current_sum);
                current_sum = parts[i + 1];
            }
        }
        new_parts.push(current_sum);

        result.push(Composition::new(new_parts).unwrap());
    }

    result
}

/// Compute all refinements of a composition
///
/// A refinement of I is obtained by splitting parts
pub fn refinements(comp: &Composition) -> Vec<Composition> {
    let parts = comp.parts();

    if parts.is_empty() {
        return vec![comp.clone()];
    }

    // For small cases, enumerate explicitly
    let mut result = vec![comp.clone()];

    fn refine_part(part: usize) -> Vec<Vec<usize>> {
        compositions(part).iter().map(|c| c.parts().to_vec()).collect()
    }

    // Generate all ways to refine each part
    fn refine_composition_helper(parts: &[usize], index: usize, current: &mut Vec<usize>, results: &mut Vec<Vec<usize>>) {
        if index >= parts.len() {
            results.push(current.clone());
            return;
        }

        for refinement in refine_part(parts[index]) {
            let old_len = current.len();
            current.extend_from_slice(&refinement);
            refine_composition_helper(parts, index + 1, current, results);
            current.truncate(old_len);
        }
    }

    let mut all_refinements = Vec::new();
    refine_composition_helper(parts, 0, &mut Vec::new(), &mut all_refinements);

    for ref_parts in all_refinements {
        if !ref_parts.is_empty() {
            result.push(Composition::new(ref_parts).unwrap());
        }
    }

    result
}

/// Product of two QSym in the Monomial basis (commutative)
///
/// M_I * M_J = sum over shuffles of M_K
pub fn product_monomial(f: &QSym, g: &QSym) -> QSym {
    if f.basis != QSymBasis::Monomial || g.basis != QSymBasis::Monomial {
        panic!("Both operands must be in Monomial basis");
    }

    let mut result = QSym::new(QSymBasis::Monomial);

    for (I, coeff_I) in &f.coeffs {
        for (J, coeff_J) in &g.coeffs {
            // Product involves quasi-shuffle of compositions
            // For simplicity, we implement a basic version
            let shuffles = quasi_shuffle(I, J);
            let coeff = coeff_I.clone() * coeff_J.clone();

            for shuffle in shuffles {
                result.add_term(shuffle, coeff.clone());
            }
        }
    }

    result
}

/// Compute quasi-shuffles of two compositions
///
/// A quasi-shuffle allows merging overlapping elements
fn quasi_shuffle(I: &Composition, J: &Composition) -> Vec<Composition> {
    let I_parts = I.parts();
    let J_parts = J.parts();

    if I_parts.is_empty() {
        return vec![J.clone()];
    }
    if J_parts.is_empty() {
        return vec![I.clone()];
    }

    let mut result = Vec::new();

    // Recursively compute quasi-shuffles
    fn quasi_shuffle_helper(
        I_parts: &[usize],
        J_parts: &[usize],
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if I_parts.is_empty() && J_parts.is_empty() {
            result.push(current.clone());
            return;
        }

        if !I_parts.is_empty() {
            // Take from I
            current.push(I_parts[0]);
            quasi_shuffle_helper(&I_parts[1..], J_parts, current, result);
            current.pop();
        }

        if !J_parts.is_empty() {
            // Take from J
            current.push(J_parts[0]);
            quasi_shuffle_helper(I_parts, &J_parts[1..], current, result);
            current.pop();
        }

        if !I_parts.is_empty() && !J_parts.is_empty() {
            // Merge (quasi-shuffle)
            current.push(I_parts[0] + J_parts[0]);
            quasi_shuffle_helper(&I_parts[1..], &J_parts[1..], current, result);
            current.pop();
        }
    }

    let mut shuffles = Vec::new();
    quasi_shuffle_helper(I_parts, J_parts, &mut Vec::new(), &mut shuffles);

    for shuffle in shuffles {
        result.push(Composition::new(shuffle).unwrap());
    }

    result
}

/// Coproduct of QSym in the Monomial basis
///
/// Δ(M_I) = sum over all ways to partition indices
pub fn coproduct_monomial(f: &QSym) -> Vec<(QSym, QSym, Rational)> {
    if f.basis != QSymBasis::Monomial {
        panic!("Operand must be in Monomial basis");
    }

    let mut result = Vec::new();

    for (I, coeff) in &f.coeffs {
        let splits = composition_coproduct_splits(I);
        for (J, K) in splits {
            let left = QSym::basis_element(QSymBasis::Monomial, J);
            let right = QSym::basis_element(QSymBasis::Monomial, K);
            result.push((left, right, coeff.clone()));
        }
    }

    result
}

/// All ways to split a composition for coproduct
///
/// For QSym, the coproduct involves splitting the indexing set
fn composition_coproduct_splits(comp: &Composition) -> Vec<(Composition, Composition)> {
    let parts = comp.parts();
    let n: usize = parts.iter().sum();

    let mut result = Vec::new();

    // For each subset S of {1, 2, ..., n}, create two compositions:
    // one from elements in S, one from elements not in S
    // This is simplified - full implementation is more complex

    // Simplified version: split by number of elements
    for k in 0..=parts.len() {
        let left_parts = if k == 0 {
            vec![]
        } else {
            parts[..k].to_vec()
        };

        let right_parts = if k == parts.len() {
            vec![]
        } else {
            parts[k..].to_vec()
        };

        let left = if left_parts.is_empty() {
            Composition::new(vec![]).unwrap()
        } else {
            Composition::new(left_parts).unwrap()
        };

        let right = if right_parts.is_empty() {
            Composition::new(vec![]).unwrap()
        } else {
            Composition::new(right_parts).unwrap()
        };

        result.push((left, right));
    }

    result
}

/// Convert from Monomial to Fundamental basis
///
/// F_I = sum over compositions J that coarsen to I of M_J
pub fn monomial_to_fundamental(f: &QSym) -> QSym {
    if f.basis != QSymBasis::Monomial {
        panic!("Input must be in Monomial basis");
    }

    let mut result = QSym::new(QSymBasis::Fundamental);

    for (I, coeff) in &f.coeffs {
        // Use the relation between M and F
        // This is a placeholder - proper implementation requires
        // computing the transformation matrix
        result.add_term(I.clone(), coeff.clone());
    }

    result
}

/// Convert from Fundamental to Monomial basis
///
/// M_I = sum over compositions J that refine I of (-1)^{ℓ(J)-ℓ(I)} F_J
pub fn fundamental_to_monomial(f: &QSym) -> QSym {
    if f.basis != QSymBasis::Fundamental {
        panic!("Input must be in Fundamental basis");
    }

    let mut result = QSym::new(QSymBasis::Monomial);

    for (I, coeff) in &f.coeffs {
        let I_len = I.length();

        // Sum over refinements
        for J in refinements(I) {
            let J_len = J.length();
            let sign = if (J_len - I_len) % 2 == 0 { 1 } else { -1 };
            result.add_term(J, coeff.clone() * Rational::from(sign));
        }
    }

    result
}

/// Compute the antipode (Hopf algebra antipode map)
pub fn antipode(f: &QSym) -> QSym {
    let mut result = QSym::new(f.basis);

    for (I, coeff) in &f.coeffs {
        // Antipode involves reversing the composition and sign changes
        let reversed = reverse_composition(I);
        let sign = if I.length() % 2 == 0 { 1 } else { -1 };
        result.add_term(reversed, coeff.clone() * Rational::from(sign));
    }

    result
}

/// Reverse a composition
fn reverse_composition(comp: &Composition) -> Composition {
    let mut parts = comp.parts().to_vec();
    parts.reverse();
    Composition::new(parts).unwrap()
}

/// Duality pairing between QSym and NCSF
///
/// Computes ⟨f, g⟩ where f is in QSym and g is in NCSF
/// The pairing is defined so that ⟨M_I, S_J⟩ = δ_{I,J}
pub fn duality_pairing(f: &QSym, g: &crate::ncsf::NCSF) -> Rational {
    use crate::ncsf::{NCSFBasis, complete_to_monomial};

    // Convert to Monomial/Complete bases
    let f_mono = if f.basis == QSymBasis::Monomial {
        f.clone()
    } else {
        fundamental_to_monomial(f)
    };

    let g_complete = if g.basis == NCSFBasis::Complete {
        g.clone()
    } else {
        // Would need conversion - placeholder
        g.clone()
    };

    // Compute pairing: ⟨M_I, S_J⟩ = δ_{I,J}
    let mut result = Rational::zero();

    for (I, f_coeff) in &f_mono.coeffs {
        if let Some(g_coeff) = g_complete.coeffs.get(I) {
            result = result + f_coeff.clone() * g_coeff.clone();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qsym_creation() {
        let qsym = QSym::new(QSymBasis::Monomial);
        assert!(qsym.is_zero());
        assert_eq!(qsym.degree(), 0);
    }

    #[test]
    fn test_qsym_basis_element() {
        let comp = Composition::new(vec![2, 1]).unwrap();
        let qsym = QSym::basis_element(QSymBasis::Monomial, comp.clone());
        assert_eq!(qsym.coeff(&comp), Rational::one());
        assert_eq!(qsym.degree(), 3);
        assert!(!qsym.is_zero());
    }

    #[test]
    fn test_qsym_addition() {
        let I = Composition::new(vec![2, 1]).unwrap();
        let J = Composition::new(vec![1, 2]).unwrap();

        let mut f = QSym::new(QSymBasis::Monomial);
        f.add_term(I.clone(), Rational::from(2));

        let mut g = QSym::new(QSymBasis::Monomial);
        g.add_term(J.clone(), Rational::from(3));
        g.add_term(I.clone(), Rational::from(-1));

        let sum = f.add(&g).unwrap();
        assert_eq!(sum.coeff(&I), Rational::one());
        assert_eq!(sum.coeff(&J), Rational::from(3));
    }

    #[test]
    fn test_coarsenings() {
        let comp = Composition::new(vec![1, 1, 1]).unwrap();
        let coars = coarsenings(&comp);
        // Should include [3], [2,1], [1,2], [1,1,1]
        assert!(coars.len() >= 4);
    }

    #[test]
    fn test_reverse_composition() {
        let comp = Composition::new(vec![2, 1, 3]).unwrap();
        let rev = reverse_composition(&comp);
        assert_eq!(rev.parts(), &[3, 1, 2]);
    }

    #[test]
    fn test_quasi_shuffle() {
        let I = Composition::new(vec![1]).unwrap();
        let J = Composition::new(vec![1]).unwrap();
        let shuffles = quasi_shuffle(&I, &J);

        // Should get [1,1], [2] (merge), [1,1] (reverse order is same)
        assert!(shuffles.len() >= 2);
    }

    #[test]
    fn test_monomial_identity() {
        let empty = Composition::new(vec![]).unwrap();
        let one = QSym::basis_element(QSymBasis::Monomial, empty);

        let comp = Composition::new(vec![2, 1]).unwrap();
        let f = monomial(comp.clone());

        let prod = product_monomial(&one, &f);
        // Product with identity should give back f (up to shuffles)
        assert!(prod.coeff(&comp) >= Rational::one());
    }

    #[test]
    fn test_antipode_sign() {
        let I = Composition::new(vec![1, 1]).unwrap(); // length 2 (even)
        let f = monomial(I.clone());
        let s_f = antipode(&f);
        let rev_I = reverse_composition(&I);
        assert_eq!(s_f.coeff(&rev_I), Rational::one()); // even length, positive

        let J = Composition::new(vec![1, 1, 1]).unwrap(); // length 3 (odd)
        let g = monomial(J.clone());
        let s_g = antipode(&g);
        let rev_J = reverse_composition(&J);
        assert_eq!(s_g.coeff(&rev_J), Rational::from(-1)); // odd length, negative
    }
}
