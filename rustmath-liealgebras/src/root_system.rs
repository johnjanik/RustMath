//! Root Systems
//!
//! A root system is a configuration of vectors (roots) in a Euclidean space
//! satisfying certain geometric properties. Root systems classify semisimple
//! Lie algebras and are fundamental to representation theory.
//!
//! Corresponds to sage.combinat.root_system.root_system

use crate::cartan_type::{CartanType, CartanLetter};
use rustmath_core::Ring;
use rustmath_rationals::Rational;
use std::collections::HashSet;

/// A root in a root system
///
/// Represented as a vector of rational coordinates
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Root {
    /// Coordinates of the root vector
    pub coordinates: Vec<Rational>,
}

impl Root {
    /// Create a new root from coordinates
    pub fn new(coordinates: Vec<Rational>) -> Self {
        Root { coordinates }
    }

    /// Get the dimension of this root
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }

    /// Compute the inner product with another root
    pub fn inner_product(&self, other: &Root) -> Rational {
        assert_eq!(self.dimension(), other.dimension());
        self.coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| a.clone() * b.clone())
            .fold(Rational::zero(), |acc, x| acc + x)
    }

    /// Negate this root
    pub fn negate(&self) -> Root {
        Root {
            coordinates: self.coordinates.iter().map(|x| -x.clone()).collect(),
        }
    }
}

/// A root system associated with a Cartan type
pub struct RootSystem {
    /// The Cartan type of this root system
    pub cartan_type: CartanType,
    /// The simple roots (generators of all positive roots)
    pub simple_roots: Vec<Root>,
    /// All positive roots
    pub positive_roots: Vec<Root>,
    /// The ambient dimension
    pub ambient_dim: usize,
}

impl RootSystem {
    /// Create a root system for a given Cartan type
    ///
    /// This constructs the simple roots in the standard realization
    pub fn new(cartan_type: CartanType) -> Self {
        let (simple_roots, ambient_dim) = Self::construct_simple_roots(&cartan_type);
        let positive_roots = Self::compute_positive_roots(&simple_roots);

        RootSystem {
            cartan_type,
            simple_roots,
            positive_roots,
            ambient_dim,
        }
    }

    /// Construct the simple roots for a given Cartan type
    ///
    /// Returns (simple_roots, ambient_dimension)
    fn construct_simple_roots(ct: &CartanType) -> (Vec<Root>, usize) {
        match ct.letter {
            CartanLetter::A => Self::simple_roots_type_a(ct.rank),
            CartanLetter::B => Self::simple_roots_type_b(ct.rank),
            CartanLetter::C => Self::simple_roots_type_c(ct.rank),
            CartanLetter::D => Self::simple_roots_type_d(ct.rank),
            CartanLetter::E => Self::simple_roots_type_e(ct.rank),
            CartanLetter::F => Self::simple_roots_type_f(),
            CartanLetter::G => Self::simple_roots_type_g(),
        }
    }

    /// Simple roots for type A_n in R^{n+1}
    fn simple_roots_type_a(n: usize) -> (Vec<Root>, usize) {
        let mut roots = Vec::new();
        let dim = n + 1;

        for i in 0..n {
            let mut coords = vec![Rational::zero(); dim];
            coords[i] = Rational::one();
            coords[i + 1] = -Rational::one();
            roots.push(Root::new(coords));
        }

        (roots, dim)
    }

    /// Simple roots for type B_n in R^n
    fn simple_roots_type_b(n: usize) -> (Vec<Root>, usize) {
        let mut roots = Vec::new();

        // α_i = e_i - e_{i+1} for i = 1, ..., n-1
        for i in 0..n - 1 {
            let mut coords = vec![Rational::zero(); n];
            coords[i] = Rational::one();
            coords[i + 1] = -Rational::one();
            roots.push(Root::new(coords));
        }

        // α_n = e_n
        let mut coords = vec![Rational::zero(); n];
        coords[n - 1] = Rational::one();
        roots.push(Root::new(coords));

        (roots, n)
    }

    /// Simple roots for type C_n in R^n
    fn simple_roots_type_c(n: usize) -> (Vec<Root>, usize) {
        let mut roots = Vec::new();

        // α_i = e_i - e_{i+1} for i = 1, ..., n-1
        for i in 0..n - 1 {
            let mut coords = vec![Rational::zero(); n];
            coords[i] = Rational::one();
            coords[i + 1] = -Rational::one();
            roots.push(Root::new(coords));
        }

        // α_n = 2e_n
        let mut coords = vec![Rational::zero(); n];
        coords[n - 1] = Rational::from_integer(2);
        roots.push(Root::new(coords));

        (roots, n)
    }

    /// Simple roots for type D_n in R^n
    fn simple_roots_type_d(n: usize) -> (Vec<Root>, usize) {
        let mut roots = Vec::new();

        // α_i = e_i - e_{i+1} for i = 1, ..., n-1
        for i in 0..n - 1 {
            let mut coords = vec![Rational::zero(); n];
            coords[i] = Rational::one();
            coords[i + 1] = -Rational::one();
            roots.push(Root::new(coords));
        }

        // α_n = e_{n-1} + e_n
        let mut coords = vec![Rational::zero(); n];
        coords[n - 2] = Rational::one();
        coords[n - 1] = Rational::one();
        roots.push(Root::new(coords));

        (roots, n)
    }

    /// Simple roots for type E (E_6, E_7, E_8)
    fn simple_roots_type_e(n: usize) -> (Vec<Root>, usize) {
        // E_n embeds in R^{n} for n = 6, 7, 8
        // Using standard realizations
        match n {
            6 | 7 | 8 => {
                // Simplified: return placeholder (full implementation would use proper embedding)
                let mut roots = Vec::new();
                let dim = 8;

                // α_1 = (1/2, -1/2, -1/2, -1/2, -1/2, -1/2, -1/2, 1/2)
                // This is a simplified version - proper E type roots require careful construction
                for i in 0..n {
                    let mut coords = vec![Rational::zero(); dim];
                    if i < dim - 1 {
                        coords[i] = Rational::one();
                        coords[i + 1] = -Rational::one();
                    } else {
                        coords[i] = Rational::one();
                    }
                    roots.push(Root::new(coords));
                }

                (roots, dim)
            }
            _ => panic!("Invalid E type rank"),
        }
    }

    /// Simple roots for type F_4 in R^4
    fn simple_roots_type_f() -> (Vec<Root>, usize) {
        let roots = vec![
            Root::new(vec![
                Rational::zero(),
                Rational::one(),
                -Rational::one(),
                Rational::zero(),
            ]),
            Root::new(vec![
                Rational::zero(),
                Rational::zero(),
                Rational::one(),
                -Rational::one(),
            ]),
            Root::new(vec![
                Rational::zero(),
                Rational::zero(),
                Rational::zero(),
                Rational::one(),
            ]),
            Root::new(vec![
                Rational::new(1, 2).unwrap(),
                -Rational::new(1, 2).unwrap(),
                -Rational::new(1, 2).unwrap(),
                -Rational::new(1, 2).unwrap(),
            ]),
        ];

        (roots, 4)
    }

    /// Simple roots for type G_2 in R^3
    fn simple_roots_type_g() -> (Vec<Root>, usize) {
        let roots = vec![
            Root::new(vec![Rational::one(), -Rational::one(), Rational::zero()]),
            Root::new(vec![
                -Rational::from_integer(2),
                Rational::one(),
                Rational::one(),
            ]),
        ];

        (roots, 3)
    }

    /// Compute all positive roots from simple roots
    ///
    /// This is a simplified version - full implementation would use reflection formulas
    fn compute_positive_roots(simple_roots: &[Root]) -> Vec<Root> {
        // For now, just return the simple roots
        // Full implementation would generate all positive roots using the root system axioms
        simple_roots.to_vec()
    }

    /// Get the rank of this root system
    pub fn rank(&self) -> usize {
        self.cartan_type.rank
    }

    /// Get all roots (positive and negative)
    pub fn all_roots(&self) -> Vec<Root> {
        let mut roots = self.positive_roots.clone();
        roots.extend(self.positive_roots.iter().map(|r| r.negate()));
        roots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_inner_product() {
        let r1 = Root::new(vec![Rational::one(), Rational::zero()]);
        let r2 = Root::new(vec![Rational::zero(), Rational::one()]);

        assert_eq!(r1.inner_product(&r2), Rational::zero());
        assert_eq!(r1.inner_product(&r1), Rational::one());
    }

    #[test]
    fn test_root_system_type_a() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let rs = RootSystem::new(ct);

        assert_eq!(rs.simple_roots.len(), 2);
        assert_eq!(rs.rank(), 2);
        assert_eq!(rs.ambient_dim, 3);
    }

    #[test]
    fn test_root_system_type_b() {
        let ct = CartanType::new(CartanLetter::B, 3).unwrap();
        let rs = RootSystem::new(ct);

        assert_eq!(rs.simple_roots.len(), 3);
        assert_eq!(rs.ambient_dim, 3);
    }

    #[test]
    fn test_simple_roots_orthogonality() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let rs = RootSystem::new(ct);

        // α_1 and α_2 for type A_2 should have specific inner product
        let alpha1 = &rs.simple_roots[0];
        let alpha2 = &rs.simple_roots[1];

        // For type A, adjacent simple roots have inner product -1
        assert_eq!(alpha1.inner_product(alpha2), -Rational::one());
    }
}
