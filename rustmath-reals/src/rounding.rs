//! Rounding modes for real number arithmetic

/// Rounding modes for real number operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round toward nearest, ties to even
    Nearest,
    /// Round toward zero (truncate)
    TowardZero,
    /// Round toward positive infinity (ceiling)
    TowardPositive,
    /// Round toward negative infinity (floor)
    TowardNegative,
    /// Round away from zero
    AwayFromZero,
}

impl Default for RoundingMode {
    fn default() -> Self {
        RoundingMode::Nearest
    }
}

impl RoundingMode {
    /// Apply rounding mode to a value
    pub fn round(&self, value: f64) -> f64 {
        match self {
            RoundingMode::Nearest => value.round(),
            RoundingMode::TowardZero => value.trunc(),
            RoundingMode::TowardPositive => value.ceil(),
            RoundingMode::TowardNegative => value.floor(),
            RoundingMode::AwayFromZero => {
                if value >= 0.0 {
                    value.ceil()
                } else {
                    value.floor()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounding_modes() {
        let value = 3.7;

        assert_eq!(RoundingMode::Nearest.round(value), 4.0);
        assert_eq!(RoundingMode::TowardZero.round(value), 3.0);
        assert_eq!(RoundingMode::TowardPositive.round(value), 4.0);
        assert_eq!(RoundingMode::TowardNegative.round(value), 3.0);
        assert_eq!(RoundingMode::AwayFromZero.round(value), 4.0);

        let neg_value = -3.7;
        assert_eq!(RoundingMode::Nearest.round(neg_value), -4.0);
        assert_eq!(RoundingMode::TowardZero.round(neg_value), -3.0);
        assert_eq!(RoundingMode::TowardPositive.round(neg_value), -3.0);
        assert_eq!(RoundingMode::TowardNegative.round(neg_value), -4.0);
        assert_eq!(RoundingMode::AwayFromZero.round(neg_value), -4.0);
    }
}
