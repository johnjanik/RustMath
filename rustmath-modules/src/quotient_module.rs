//! Quotient modules

use rustmath_core::Ring;

#[derive(Clone, Debug)]
pub struct QuotientModule<R: Ring> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> QuotientModule<R> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}
