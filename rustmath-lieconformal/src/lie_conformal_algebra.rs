//! Base Lie Conformal Algebra Module
//!
//! Defines the fundamental traits and structures for Lie conformal algebras.

use rustmath_core::Ring;
use std::fmt::{self, Display};
use std::collections::HashMap;

/// Trait for Lie conformal algebras
pub trait LieConformalAlgebra<R: Ring> {
    type Element;
    fn base_ring(&self) -> &R;
    fn ngens(&self) -> Option<usize>;
    fn generator(fn generator(&self, i: usize) -> Option<Self::Element>;self, i: usize) -> Option<Self::Element> where R: From<i64>;
    fn generators(&self) -> Vec<Self::Element> {
        if let Some(n) = self.ngens() {
            (0..n).filter_map(|i| self.generator(i)).collect()
        } else {
            vec![]
        }
    }
    fn is_abelian(&self) -> bool {
        false
    }
    fn central_charge(&self) -> Option<R> {
        None
    }
    fn zero(&self) -> Self::Element;
    fn one(&self) -> Option<Self::Element> {
        None
    }
}

/// λ-bracket operation for Lie conformal algebras
pub trait LambdaBracket<R: Ring, E: Clone> {
    fn lambda_bracket(&self, a: &E, b: &E) -> HashMap<usize, E>;
    fn n_product(&self, a: &E, b: &E, n: usize) -> E {
        self.lambda_bracket(a, b).get(&n).cloned()
            .unwrap_or_else(|| panic!("No term for power {}", n))
    }
}

/// Derivation operator ∂ for Lie conformal algebras
pub trait Derivation<E: Clone> {
    fn apply_derivation(&self, element: &E) -> E;
    fn apply_derivation_n(&self, element: &E, n: usize) -> E {
        let mut result = element.clone();
        for _ in 0..n {
            result = self.apply_derivation(&result);
        }
        result
    }
}

/// Index set for generators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GeneratorIndex {
    Finite(usize),
    Named(String),
    Composite(Vec<Box<GeneratorIndex>>),
}

impl Display for GeneratorIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GeneratorIndex::Finite(i) => write!(f, "{}", i),
            GeneratorIndex::Named(s) => write!(f, "{}", s),
            GeneratorIndex::Composite(indices) => {
                write!(f, "(")?;
                for (i, idx) in indices.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "{}", idx)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl GeneratorIndex {
    pub fn finite(i: usize) -> Self {
        GeneratorIndex::Finite(i)
    }
    pub fn named(s: impl Into<String>) -> Self {
        GeneratorIndex::Named(s.into())
    }
    pub fn is_finite(&self) -> bool {
        matches!(self, GeneratorIndex::Finite(_))
    }
    pub fn as_finite(&self) -> Option<usize> {
        match self {
            GeneratorIndex::Finite(i) => Some(*i),
            _ => None,
        }
    }
}
