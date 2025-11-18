//! # Exterior Powers of Free Modules
//!
//! This module provides exterior power constructions for free modules,
//! corresponding to SageMath's `sage.tensor.modules.ext_pow_free_module`.

use std::marker::PhantomData;

/// Exterior power of a free module
///
/// Λ^p(M) is the p-th exterior power of module M
pub struct ExtPowerFreeModule<R> {
    degree: usize,
    base_rank: usize,
    ring: PhantomData<R>,
}

impl<R> ExtPowerFreeModule<R> {
    pub fn new(degree: usize, base_rank: usize) -> Self {
        Self {
            degree,
            base_rank,
            ring: PhantomData,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn base_rank(&self) -> usize {
        self.base_rank
    }

    /// Rank of the exterior power
    pub fn rank(&self) -> usize {
        binomial(self.base_rank, self.degree)
    }
}

/// Exterior power of the dual of a free module
pub struct ExtPowerDualFreeModule<R> {
    degree: usize,
    base_rank: usize,
    ring: PhantomData<R>,
}

impl<R> ExtPowerDualFreeModule<R> {
    pub fn new(degree: usize, base_rank: usize) -> Self {
        Self {
            degree,
            base_rank,
            ring: PhantomData,
        }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn rank(&self) -> usize {
        binomial(self.base_rank, self.degree)
    }
}

/// Compute binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(4, 0), 1);
        assert_eq!(binomial(4, 1), 4);
        assert_eq!(binomial(4, 2), 6);
        assert_eq!(binomial(4, 3), 4);
        assert_eq!(binomial(4, 4), 1);
    }

    #[test]
    fn test_ext_power_module() {
        let ext: ExtPowerFreeModule<i32> = ExtPowerFreeModule::new(2, 4);
        assert_eq!(ext.degree(), 2);
        assert_eq!(ext.base_rank(), 4);
        assert_eq!(ext.rank(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_ext_power_dual() {
        let ext: ExtPowerDualFreeModule<i32> = ExtPowerDualFreeModule::new(3, 5);
        assert_eq!(ext.degree(), 3);
        assert_eq!(ext.rank(), 10); // C(5,3) = 10
    }
}
