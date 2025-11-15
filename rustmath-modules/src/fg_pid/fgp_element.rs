//! Elements of FGP modules

use rustmath_core::Ring;

/// An element of a finitely generated module over a PID
#[derive(Clone, Debug, PartialEq)]
pub struct FGPElement<R: Ring> {
    /// Coordinates in the Smith normal form decomposition
    coordinates: Vec<R>,
}

impl<R: Ring> FGPElement<R> {
    pub fn new(coordinates: Vec<R>) -> Self {
        Self { coordinates }
    }

    pub fn zero(rank: usize) -> Self {
        Self {
            coordinates: vec![R::zero(); rank],
        }
    }

    pub fn coordinates(&self) -> &[R] {
        &self.coordinates
    }

    pub fn is_zero(&self) -> bool {
        self.coordinates.iter().all(|x| x.is_zero())
    }
}
