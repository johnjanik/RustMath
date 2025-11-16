//! Wigner Symbols and Related Functions
//!
//! This module implements Wigner 3-j, 6-j, and 9-j symbols, Clebsch-Gordan coefficients,
//! and related functions from quantum mechanics and representation theory.
//!
//! Corresponds to sage.functions.wigner
//!
//! # Functions
//!
//! - `wigner_3j(j1, j2, j3, m1, m2, m3)`: Wigner 3-j symbol
//! - `wigner_6j(j1, j2, j3, j4, j5, j6)`: Wigner 6-j symbol
//! - `wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)`: Wigner 9-j symbol
//! - `clebsch_gordan(j1, j2, j3, m1, m2, m3)`: Clebsch-Gordan coefficient
//! - `racah(a, b, c, d, e, f)`: Racah W coefficient
//! - `gaunt(l1, l2, l3, m1, m2, m3)`: Gaunt coefficient
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::wigner::*;
//! use rustmath_symbolic::Expr;
//!
//! let w3j = wigner_3j(&Expr::from(1), &Expr::from(1), &Expr::from(0),
//!                     &Expr::from(0), &Expr::from(0), &Expr::from(0));
//! ```
//!
//! # Mathematical Background
//!
//! Wigner symbols arise in the quantum theory of angular momentum when coupling
//! angular momenta. They encode the selection rules and coupling coefficients for
//! combining quantum states.
//!
//! The 3-j symbol is related to the Clebsch-Gordan coefficient by:
//! ⟨j₁ m₁ j₂ m₂|j₃ m₃⟩ = (-1)^(j₁-j₂+m₃) √(2j₃+1) ( j₁  j₂  j₃ )
//!                                                    ( m₁ m₂ -m₃ )

use crate::expression::Expr;
use std::sync::Arc;

/// Wigner 3-j symbol
///
/// The 3-j symbol represents the coupling of two angular momenta j₁ and j₂
/// to form a third angular momentum j₃.
///
/// # Arguments
///
/// * `j1`, `j2`, `j3` - Angular momentum quantum numbers (non-negative integers or half-integers)
/// * `m1`, `m2`, `m3` - Magnetic quantum numbers (|mᵢ| ≤ jᵢ)
///
/// # Returns
///
/// The Wigner 3-j symbol:
/// ```text
/// ( j₁  j₂  j₃ )
/// ( m₁ m₂ m₃ )
/// ```
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::wigner::wigner_3j;
/// use rustmath_symbolic::Expr;
///
/// let w = wigner_3j(&Expr::from(1), &Expr::from(1), &Expr::from(0),
///                   &Expr::from(0), &Expr::from(0), &Expr::from(0));
/// ```
///
/// # Properties
///
/// - Selection rules: m₁ + m₂ + m₃ = 0, |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂
/// - Symmetric under even permutations of columns
/// - Changes sign under odd permutations (with appropriate phase factor)
/// - Orthogonality relations
pub fn wigner_3j(j1: &Expr, j2: &Expr, j3: &Expr, m1: &Expr, m2: &Expr, m3: &Expr) -> Expr {
    Expr::Function(
        "wigner_3j".to_string(),
        vec![
            Arc::new(j1.clone()),
            Arc::new(j2.clone()),
            Arc::new(j3.clone()),
            Arc::new(m1.clone()),
            Arc::new(m2.clone()),
            Arc::new(m3.clone()),
        ],
    )
}

/// Wigner 6-j symbol
///
/// The 6-j symbol represents the recoupling of three angular momenta.
/// It appears in the expansion of products of four spherical harmonics.
///
/// # Arguments
///
/// * `j1` through `j6` - Angular momentum quantum numbers
///
/// # Returns
///
/// The Wigner 6-j symbol:
/// ```text
/// { j₁  j₂  j₃ }
/// { j₄  j₅  j₆ }
/// ```
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::wigner::wigner_6j;
/// use rustmath_symbolic::Expr;
///
/// let w = wigner_6j(&Expr::from(1), &Expr::from(1), &Expr::from(0),
///                   &Expr::from(1), &Expr::from(1), &Expr::from(1));
/// ```
///
/// # Properties
///
/// - Tetrahedral symmetry: invariant under certain permutations
/// - Selection rules for triads (j₁, j₂, j₃), (j₁, j₅, j₆), (j₄, j₂, j₆), (j₄, j₅, j₃)
/// - Orthogonality relations
///
/// # Relation to Racah W Coefficient
///
/// W(a,b,c,d;e,f) = { a  b  e }
///                   { d  c  f }
pub fn wigner_6j(j1: &Expr, j2: &Expr, j3: &Expr, j4: &Expr, j5: &Expr, j6: &Expr) -> Expr {
    Expr::Function(
        "wigner_6j".to_string(),
        vec![
            Arc::new(j1.clone()),
            Arc::new(j2.clone()),
            Arc::new(j3.clone()),
            Arc::new(j4.clone()),
            Arc::new(j5.clone()),
            Arc::new(j6.clone()),
        ],
    )
}

/// Wigner 9-j symbol
///
/// The 9-j symbol represents the coupling of four angular momenta and appears
/// in the reduction of direct products of representations.
///
/// # Arguments
///
/// * `j1` through `j9` - Angular momentum quantum numbers
///
/// # Returns
///
/// The Wigner 9-j symbol:
/// ```text
/// { j₁  j₂  j₃ }
/// { j₄  j₅  j₆ }
/// { j₇  j₈  j₉ }
/// ```
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::wigner::wigner_9j;
/// use rustmath_symbolic::Expr;
///
/// let w = wigner_9j(&Expr::from(1), &Expr::from(1), &Expr::from(0),
///                   &Expr::from(1), &Expr::from(1), &Expr::from(0),
///                   &Expr::from(1), &Expr::from(1), &Expr::from(0));
/// ```
///
/// # Properties
///
/// - Selection rules for rows and columns
/// - Symmetry under reflections and certain permutations
/// - Can be expressed as a sum of products of 6-j symbols
/// - Orthogonality relations
pub fn wigner_9j(
    j1: &Expr,
    j2: &Expr,
    j3: &Expr,
    j4: &Expr,
    j5: &Expr,
    j6: &Expr,
    j7: &Expr,
    j8: &Expr,
    j9: &Expr,
) -> Expr {
    Expr::Function(
        "wigner_9j".to_string(),
        vec![
            Arc::new(j1.clone()),
            Arc::new(j2.clone()),
            Arc::new(j3.clone()),
            Arc::new(j4.clone()),
            Arc::new(j5.clone()),
            Arc::new(j6.clone()),
            Arc::new(j7.clone()),
            Arc::new(j8.clone()),
            Arc::new(j9.clone()),
        ],
    )
}

/// Clebsch-Gordan coefficient
///
/// The Clebsch-Gordan coefficient represents the decomposition of the tensor
/// product of two irreducible representations into irreducible representations.
///
/// # Arguments
///
/// * `j1`, `j2`, `j3` - Angular momentum quantum numbers
/// * `m1`, `m2`, `m3` - Magnetic quantum numbers
///
/// # Returns
///
/// The Clebsch-Gordan coefficient ⟨j₁ m₁ j₂ m₂|j₃ m₃⟩
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::wigner::clebsch_gordan;
/// use rustmath_symbolic::Expr;
///
/// let cg = clebsch_gordan(&Expr::from(1), &Expr::from(1), &Expr::from(2),
///                         &Expr::from(0), &Expr::from(0), &Expr::from(0));
/// ```
///
/// # Properties
///
/// - Selection rules: m₁ + m₂ = m₃, |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂
/// - Orthogonality: Σ_{m₁,m₂} ⟨j₁m₁j₂m₂|jm⟩⟨j₁m₁j₂m₂|j'm'⟩ = δⱼⱼ′δₘₘ′
/// - Related to 3-j symbol by phase factor
///
/// # Relation to 3-j Symbol
///
/// ⟨j₁m₁j₂m₂|j₃m₃⟩ = (-1)^(j₁-j₂+m₃) √(2j₃+1) ( j₁  j₂  j₃ )
///                                              ( m₁ m₂ -m₃ )
///
/// # Applications
///
/// - Addition of angular momenta in quantum mechanics
/// - Atomic and nuclear spectroscopy
/// - Coupling of spin and orbital angular momentum
pub fn clebsch_gordan(j1: &Expr, j2: &Expr, j3: &Expr, m1: &Expr, m2: &Expr, m3: &Expr) -> Expr {
    Expr::Function(
        "clebsch_gordan".to_string(),
        vec![
            Arc::new(j1.clone()),
            Arc::new(j2.clone()),
            Arc::new(j3.clone()),
            Arc::new(m1.clone()),
            Arc::new(m2.clone()),
            Arc::new(m3.clone()),
        ],
    )
}

/// Racah W coefficient
///
/// The Racah W coefficient is equivalent to the Wigner 6-j symbol up to a
/// permutation of arguments. It's named after Giulio Racah who studied
/// these coefficients in the context of atomic spectroscopy.
///
/// # Arguments
///
/// * `a`, `b`, `c`, `d`, `e`, `f` - Angular momentum quantum numbers
///
/// # Returns
///
/// W(a,b,c,d;e,f)
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::wigner::racah;
/// use rustmath_symbolic::Expr;
///
/// let w = racah(&Expr::from(1), &Expr::from(1), &Expr::from(1),
///               &Expr::from(1), &Expr::from(1), &Expr::from(0));
/// ```
///
/// # Relation to 6-j Symbol
///
/// W(a,b,c,d;e,f) = { a  b  e }
///                   { d  c  f }
///
/// # Applications
///
/// - Atomic spectroscopy (original application)
/// - Recoupling of angular momenta
/// - Matrix elements of operators in coupled basis
pub fn racah(a: &Expr, b: &Expr, c: &Expr, d: &Expr, e: &Expr, f: &Expr) -> Expr {
    Expr::Function(
        "racah".to_string(),
        vec![
            Arc::new(a.clone()),
            Arc::new(b.clone()),
            Arc::new(c.clone()),
            Arc::new(d.clone()),
            Arc::new(e.clone()),
            Arc::new(f.clone()),
        ],
    )
}

/// Gaunt coefficient
///
/// The Gaunt coefficient is the integral of three spherical harmonics.
/// It appears in the theory of multipole expansions and selection rules.
///
/// # Arguments
///
/// * `l1`, `l2`, `l3` - Orbital angular momentum quantum numbers
/// * `m1`, `m2`, `m3` - Magnetic quantum numbers
///
/// # Returns
///
/// ∫ Y_{l₁}^{m₁}(θ,φ) Y_{l₂}^{m₂}(θ,φ) Y_{l₃}^{m₃}(θ,φ) dΩ
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::wigner::gaunt;
/// use rustmath_symbolic::Expr;
///
/// let g = gaunt(&Expr::from(1), &Expr::from(1), &Expr::from(0),
///               &Expr::from(0), &Expr::from(0), &Expr::from(0));
/// ```
///
/// # Properties
///
/// - Selection rules: m₁ + m₂ + m₃ = 0, triangle inequality for (l₁,l₂,l₃)
/// - Parity selection: l₁ + l₂ + l₃ must be even
/// - Can be expressed in terms of 3-j symbols and factorials
///
/// # Relation to 3-j Symbols
///
/// The Gaunt coefficient can be expressed as a product of two 3-j symbols
/// with appropriate normalization factors.
///
/// # Applications
///
/// - Multipole expansions in electromagnetism
/// - Selection rules for radiative transitions
/// - Molecular orbital theory
/// - Gravitational radiation
pub fn gaunt(l1: &Expr, l2: &Expr, l3: &Expr, m1: &Expr, m2: &Expr, m3: &Expr) -> Expr {
    Expr::Function(
        "gaunt".to_string(),
        vec![
            Arc::new(l1.clone()),
            Arc::new(l2.clone()),
            Arc::new(l3.clone()),
            Arc::new(m1.clone()),
            Arc::new(m2.clone()),
            Arc::new(m3.clone()),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_wigner_3j_symbolic() {
        let w = wigner_3j(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
        );
        assert!(matches!(w, Expr::Function(name, args)
            if name == "wigner_3j" && args.len() == 6));
    }

    #[test]
    fn test_wigner_6j_symbolic() {
        let w = wigner_6j(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(1),
        );
        assert!(matches!(w, Expr::Function(name, args)
            if name == "wigner_6j" && args.len() == 6));
    }

    #[test]
    fn test_wigner_9j_symbolic() {
        let w = wigner_9j(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
        );
        assert!(matches!(w, Expr::Function(name, args)
            if name == "wigner_9j" && args.len() == 9));
    }

    #[test]
    fn test_clebsch_gordan_symbolic() {
        let cg = clebsch_gordan(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(2),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
        );
        assert!(matches!(cg, Expr::Function(name, args)
            if name == "clebsch_gordan" && args.len() == 6));
    }

    #[test]
    fn test_racah_symbolic() {
        let w = racah(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
        );
        assert!(matches!(w, Expr::Function(name, args)
            if name == "racah" && args.len() == 6));
    }

    #[test]
    fn test_gaunt_symbolic() {
        let g = gaunt(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
        );
        assert!(matches!(g, Expr::Function(name, args)
            if name == "gaunt" && args.len() == 6));
    }

    #[test]
    fn test_wigner_3j_vs_clebsch_gordan() {
        let w3j = wigner_3j(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
        );
        let cg = clebsch_gordan(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
        );

        // Different functions even though they're related
        assert_ne!(w3j, cg);
    }

    #[test]
    fn test_wigner_6j_vs_racah() {
        let w6j = wigner_6j(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(1),
        );
        let w_racah = racah(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
        );

        // Different functions (related by permutation)
        assert_ne!(w6j, w_racah);
    }

    #[test]
    fn test_wigner_different_j() {
        let w1 = wigner_3j(
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
        );
        let w2 = wigner_3j(
            &Expr::from(2),
            &Expr::from(1),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(0),
            &Expr::from(0),
        );

        assert_ne!(w1, w2);
    }
}
