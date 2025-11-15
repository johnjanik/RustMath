//! Morphisms between FGP modules

use rustmath_core::Ring;
use super::fgp_module::FGPModule;
use super::fgp_element::FGPElement;

/// A morphism between FGP modules
#[derive(Clone, Debug)]
pub struct FGPMorphism<R: Ring> {
    matrix: Vec<Vec<R>>,
}

impl<R: Ring> FGPMorphism<R> {
    pub fn new(matrix: Vec<Vec<R>>) -> Self {
        Self { matrix }
    }

    pub fn apply(&self, element: &FGPElement<R>) -> FGPElement<R> {
        let coords = element.coordinates();
        let mut result = vec![R::zero(); self.matrix.len()];

        for (i, row) in self.matrix.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                if j < coords.len() {
                    result[i] = result[i].clone() + val.clone() * coords[j].clone();
                }
            }
        }

        FGPElement::new(result)
    }
}
