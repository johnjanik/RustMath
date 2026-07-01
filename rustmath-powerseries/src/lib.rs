//! Power series over rings
//!
//! Provides truncated power series arithmetic for formal power series.
//! A power series is an infinite series of the form: Σ(n=0 to ∞) aₙxⁿ
//! We represent truncated series keeping coefficients up to some precision.
//!
//! Also provides weighted automata for recognizable (rational) series.
//!
//! # MAGMA handbook coverage (Chapters 49, 50, 52)
//!
//! - Chapter 49 (Power/Laurent/Puiseux series): [`PowerSeries`] now implements
//!   the `rustmath-core` `Ring`/`CommutativeRing`/`IntegralDomain` tower (see
//!   `series_core_impls`), transcendental functions live in `transcendental`,
//!   and [`LaurentSeries`] (integral valuation, a **field** over a field) and
//!   [`PuiseuxSeries`] (rational exponents) are new.  A [`Precision`] regime and
//!   a lightweight [`SeriesRing`] parent model free vs fixed precision.
//! - Chapter 50 (Lazy power series): [`LazyPowerSeries`] — memoised on-demand
//!   coefficients (`Rc<RefCell>` cache), concrete-generic so it implements
//!   `Ring`.
//! - Chapter 52 (Algebraic power series): [`implicit_function`] and
//!   [`newton_puiseux`] compute series roots of a [`BivariatePoly`].

pub mod algebraic;
pub mod automaton;
pub mod laurent;
pub mod lazy;
pub mod precision;
pub mod puiseux;
pub mod series;
pub mod series_core_impls;
pub mod transcendental;

pub use algebraic::{
    implicit_function, newton_puiseux, prime_field_roots, rational_roots, BivariatePoly,
    NewtonPuiseuxResult,
};
pub use automaton::WeightedAutomaton;
pub use laurent::{LaurentSeries, LaurentValuation};
pub use lazy::LazyPowerSeries;
pub use precision::{Precision, SeriesKind, SeriesRing, DEFAULT_PRECISION};
pub use puiseux::PuiseuxSeries;
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
