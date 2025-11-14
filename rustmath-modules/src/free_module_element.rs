//! Elements of free modules

use rustmath_core::Ring;

/// An element of a free module (vector)
#[derive(Clone, Debug, PartialEq)]
pub struct FreeModuleElement<R: Ring> {
    coordinates: Vec<R>,
}

impl<R: Ring> FreeModuleElement<R> {
    /// Create a new element from coordinates
    pub fn new(coordinates: Vec<R>) -> Self {
        Self { coordinates }
    }

    /// Create the zero element of given dimension
    pub fn zero(dim: usize) -> Self {
        Self {
            coordinates: vec![R::zero(); dim],
        }
    }

    /// Get the coordinates
    pub fn coordinates(&self) -> &[R] {
        &self.coordinates
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }

    /// Check if this is the zero element
    pub fn is_zero(&self, _ring: &R) -> bool {
        self.coordinates.iter().all(|x| x.is_zero())
    }

    /// Add two elements
    pub fn add(&self, other: &Self, _ring: &R) -> Self {
        assert_eq!(self.dimension(), other.dimension());
        let coords: Vec<R> = self
            .coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        Self::new(coords)
    }

    /// Negate this element
    pub fn negate(&self, _ring: &R) -> Self {
        let coords: Vec<R> = self.coordinates.iter().map(|x| -x.clone()).collect();
        Self::new(coords)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R, _ring: &R) -> Self {
        let coords: Vec<R> = self
            .coordinates
            .iter()
            .map(|x| scalar.clone() * x.clone())
            .collect();
        Self::new(coords)
    }

    /// Dot product with another element
    pub fn dot(&self, other: &Self, _ring: &R) -> R {
        assert_eq!(self.dimension(), other.dimension());
        self.coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| a.clone() * b.clone())
            .fold(R::zero(), |acc, x| acc + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_traits::{Zero, One};

    #[test]
    fn test_element_creation() {
        let elem = FreeModuleElement::new(vec![
            BigInt::from(1),
            BigInt::from(2),
            BigInt::from(3),
        ]);
        assert_eq!(elem.dimension(), 3);
    }

    #[test]
    fn test_zero() {
        let zero: FreeModuleElement<BigInt> = FreeModuleElement::zero(3);
        assert_eq!(zero.dimension(), 3);
    }
}
