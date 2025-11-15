//! Submodules

use rustmath_core::Ring;
use crate::free_module_element::FreeModuleElement;

/// A submodule of a free module, given by generators
#[derive(Clone, Debug)]
pub struct Submodule<R: Ring> {
    ambient_rank: usize,
    generators: Vec<FreeModuleElement<R>>,
}

impl<R: Ring> Submodule<R> {
    pub fn new(ambient_rank: usize, generators: Vec<FreeModuleElement<R>>) -> Self {
        for gen in &generators {
            assert_eq!(gen.dimension(), ambient_rank);
        }
        Self {
            ambient_rank,
            generators,
        }
    }

    pub fn zero(ambient_rank: usize) -> Self {
        Self {
            ambient_rank,
            generators: Vec::new(),
        }
    }

    pub fn ambient_rank(&self) -> usize {
        self.ambient_rank
    }

    pub fn generators(&self) -> &[FreeModuleElement<R>] {
        &self.generators
    }

    pub fn rank(&self) -> usize {
        self.generators.len()
    }

    pub fn is_zero(&self) -> bool {
        self.generators.is_empty()
    }

    /// Add a generator
    pub fn add_generator(&mut self, gen: FreeModuleElement<R>) {
        assert_eq!(gen.dimension(), self.ambient_rank);
        self.generators.push(gen);
    }
}
