//! # Free Module Elements
//!
//! This module provides elements of finite rank free modules,
//! corresponding to SageMath's `sage.tensor.modules.free_module_element`.

use std::fmt;
use std::ops::{Add, Mul, Neg};

/// An element of a finite rank free module
///
/// Represented as a linear combination of basis vectors
pub struct FiniteRankFreeModuleElement<R> {
    components: Vec<R>,
}

impl<R: Clone> FiniteRankFreeModuleElement<R> {
    pub fn new(components: Vec<R>) -> Self {
        Self { components }
    }

    pub fn rank(&self) -> usize {
        self.components.len()
    }

    pub fn components(&self) -> &[R] {
        &self.components
    }

    pub fn component(&self, i: usize) -> &R {
        &self.components[i]
    }
}

impl<R: Clone + Add<Output = R>> Add for FiniteRankFreeModuleElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.rank(), other.rank());

        let components = self
            .components
            .iter()
            .zip(other.components.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Self::new(components)
    }
}

impl<R: Clone + Neg<Output = R>> Neg for FiniteRankFreeModuleElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let components = self.components.iter().map(|x| -x.clone()).collect();
        Self::new(components)
    }
}

impl<R: Clone + Mul<Output = R>> Mul<R> for FiniteRankFreeModuleElement<R> {
    type Output = Self;

    fn mul(self, scalar: R) -> Self {
        let components = self
            .components
            .iter()
            .map(|x| x.clone() * scalar.clone())
            .collect();
        Self::new(components)
    }
}

impl<R: Clone + fmt::Display> fmt::Display for FiniteRankFreeModuleElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, comp) in self.components.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", comp)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_creation() {
        let elem = FiniteRankFreeModuleElement::new(vec![1, 2, 3]);
        assert_eq!(elem.rank(), 3);
        assert_eq!(elem.component(1), &2);
    }

    #[test]
    fn test_element_addition() {
        let e1 = FiniteRankFreeModuleElement::new(vec![1, 2, 3]);
        let e2 = FiniteRankFreeModuleElement::new(vec![4, 5, 6]);

        let sum = e1 + e2;
        assert_eq!(sum.components(), &[5, 7, 9]);
    }

    #[test]
    fn test_element_negation() {
        let elem = FiniteRankFreeModuleElement::new(vec![1, -2, 3]);
        let neg = -elem;

        assert_eq!(neg.components(), &[-1, 2, -3]);
    }

    #[test]
    fn test_scalar_multiplication() {
        let elem = FiniteRankFreeModuleElement::new(vec![1, 2, 3]);
        let scaled = elem * 2;

        assert_eq!(scaled.components(), &[2, 4, 6]);
    }
}
