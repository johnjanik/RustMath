//! Free module morphisms

use rustmath_core::Ring;

#[derive(Clone, Debug)]
pub struct FreeModuleMorphism<R: Ring> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> FreeModuleMorphism<R> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}
