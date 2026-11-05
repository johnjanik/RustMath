//! Multiple zeta values ζ(s₁, s₂, ..., sₖ)

/// Multiple zeta value
#[derive(Clone, Debug)]
pub struct MultipleZeta {
    indices: Vec<u32>,
}

impl MultipleZeta {
    /// Create ζ(s₁, s₂, ..., sₖ)
    pub fn new(indices: Vec<u32>) -> Self {
        assert!(!indices.is_empty());
        assert!(indices[indices.len() - 1] >= 2, "Last index must be ≥ 2");
        Self { indices }
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn weight(&self) -> u32 {
        self.indices.iter().sum()
    }

    pub fn depth(&self) -> usize {
        self.indices.len()
    }

    /// The numerical value of the multiple zeta value.
    ///
    /// NOT IMPLEMENTED.  A multiple zeta value
    /// `zeta(s_1, ..., s_k) = sum_{n_1 > ... > n_k >= 1} prod n_i^{-s_i}`
    /// converges far too slowly to sum directly; an actual implementation needs
    /// an accelerated evaluation (an iterated-integral / Euler-Maclaurin scheme,
    /// or the standard telescoping of the nested sums) together with a rigorous
    /// error bound to honour the requested precision.  Nothing here computes
    /// either.
    ///
    /// This used to return 0.0 for every index tuple, which is not merely
    /// imprecise -- no MZV is zero.
    pub fn numerical_value(&self, precision: usize) -> f64 {
        unimplemented!(
            "MultipleZeta::numerical_value(precision = {precision}) for {:?}: not \
             implemented. The nested sum converges far too slowly to evaluate directly, \
             and no accelerated scheme with a rigorous error bound is implemented here. \
             Previously returned 0.0 for every index tuple -- and no MZV is zero.",
            self.indices
        )
    }

    /// Check if this is a Riemann zeta value ζ(n)
    pub fn is_riemann_zeta(&self) -> bool {
        self.indices.len() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mzv() {
        let zeta = MultipleZeta::new(vec![2, 3]);
        assert_eq!(zeta.weight(), 5);
        assert_eq!(zeta.depth(), 2);
        assert!(!zeta.is_riemann_zeta());
        assert!(MultipleZeta::new(vec![3]).is_riemann_zeta());
    }

    /// The value is refused, not faked.  zeta(2) = pi^2/6 is not 0, and the
    /// placeholder returned 0.0 for every index tuple.
    #[test]
    #[should_panic(expected = "MultipleZeta::numerical_value")]
    fn test_numerical_value_is_refused_not_faked() {
        let _ = MultipleZeta::new(vec![2]).numerical_value(53);
    }
}
