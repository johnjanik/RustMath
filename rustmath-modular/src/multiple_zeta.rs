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

    /// Compute numerical value (approximate)
    pub fn numerical_value(&self, precision: usize) -> f64 {
        // Placeholder - would need actual MZV computation
        0.0
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
    }
}
