//! Finitely Generated modules over Principal Ideal Domains

use rustmath_core::Ring;

/// A finitely generated module over a PID
/// Represented in Smith normal form: M ≅ R^r ⊕ R/(d₁) ⊕ ... ⊕ R/(dₖ)
#[derive(Clone, Debug)]
pub struct FGPModule<R: Ring> {
    free_rank: usize,
    /// Invariant factors (divisors d₁, d₂, ..., dₖ where d₁|d₂|...|dₖ)
    invariant_factors: Vec<R>,
}

impl<R: Ring> FGPModule<R> {
    /// Create a new FGP module with given free rank and invariant factors
    pub fn new(free_rank: usize, invariant_factors: Vec<R>) -> Self {
        Self {
            free_rank,
            invariant_factors,
        }
    }

    /// Create a free module of given rank
    pub fn free(rank: usize) -> Self {
        Self {
            free_rank: rank,
            invariant_factors: Vec::new(),
        }
    }

    /// Create a finite module with given invariant factors
    pub fn finite(invariant_factors: Vec<R>) -> Self {
        Self {
            free_rank: 0,
            invariant_factors,
        }
    }

    pub fn free_rank(&self) -> usize {
        self.free_rank
    }

    pub fn invariant_factors(&self) -> &[R] {
        &self.invariant_factors
    }

    pub fn is_free(&self) -> bool {
        self.invariant_factors.is_empty()
    }

    pub fn is_finite(&self) -> bool {
        self.free_rank == 0
    }

    pub fn is_torsion_free(&self) -> bool {
        self.is_free()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn test_free_module() {
        let m: FGPModule<BigInt> = FGPModule::free(3);
        assert_eq!(m.free_rank(), 3);
        assert!(m.is_free());
    }

    #[test]
    fn test_finite_module() {
        let m = FGPModule::finite(vec![BigInt::from(2), BigInt::from(6)]);
        assert!(m.is_finite());
        assert_eq!(m.invariant_factors().len(), 2);
    }
}
