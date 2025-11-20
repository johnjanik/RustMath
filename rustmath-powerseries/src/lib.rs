//! Power series over rings
//!
//! Provides truncated power series arithmetic for formal power series.
//! A power series is an infinite series of the form: Σ(n=0 to ∞) aₙxⁿ
//! We represent truncated series keeping coefficients up to some precision.
//!
//! Also provides weighted automata for recognizable (rational) series.

pub mod automaton;
pub mod series;

pub use automaton::WeightedAutomaton;
pub use series::PowerSeries;

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn basic_series() {
        let coeffs = vec![Integer::from(1), Integer::from(2), Integer::from(3)];
        let series = PowerSeries::new(coeffs, 5);

        assert_eq!(series.coeff(0), &Integer::from(1));
        assert_eq!(series.coeff(1), &Integer::from(2));
        assert_eq!(series.coeff(2), &Integer::from(3));
    }
}
