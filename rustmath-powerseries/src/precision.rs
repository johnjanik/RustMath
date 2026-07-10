//! Precision model and lightweight series-ring parents.
//!
//! MAGMA source: Handbook Chapter 49 "Power, Laurent and Puiseux Series",
//! §49.1.4 (Precision), §49.1.5 (Free and Fixed Precision), §49.2.1 (creation
//! of the series-ring structures) and §49.3 (structure operations).
//!
//! A MAGMA series ring is either *free* (each element carries its own precision,
//! new elements default to the ring's `DefaultPrecision`, default 20) or
//! *fixed-precision* (every element is capped at a fixed absolute/relative
//! precision `n`). We model that distinction with the [`Precision`] enum and a
//! small value-type parent [`SeriesRing`] carrying the flavour, precision and
//! generator name. This is deliberately a lightweight companion to the concrete
//! element types (`PowerSeries`, `LaurentSeries`, `PuiseuxSeries`) rather than a
//! full `Parent`/`Element` framework.

use std::fmt;

/// The MAGMA default precision for a *free* series ring (Chapter 49.1.5).
pub const DEFAULT_PRECISION: usize = 20;

/// Precision regime of a series ring or of the environment used to build a
/// series (Chapter 49.1.5).
///
/// * `Free(default)` — elements carry their own precision; freshly created
///   elements (e.g. the generator) use `default` terms.  This mirrors a MAGMA
///   free series ring with `DefaultPrecision = default`.
/// * `Fixed(n)` — every element is truncated to `n` terms (absolute precision
///   for power series, maximum relative precision for Laurent/Puiseux rings),
///   mirroring `PowerSeriesRing(R, n)` and friends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Free precision with the given default term count.
    Free(usize),
    /// Fixed precision of exactly `n` terms.
    Fixed(usize),
}

impl Precision {
    /// A free precision with the MAGMA default of 20 terms.
    pub fn free_default() -> Self {
        Precision::Free(DEFAULT_PRECISION)
    }

    /// The number of terms a freshly created element should carry.
    pub fn default_terms(&self) -> usize {
        match self {
            Precision::Free(n) | Precision::Fixed(n) => *n,
        }
    }

    /// Whether this is a fixed-precision regime.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Precision::Fixed(_))
    }

    /// Cap a requested term count against this precision regime: fixed
    /// precision never exceeds its bound, free precision honours the request.
    pub fn cap(&self, requested: usize) -> usize {
        match self {
            Precision::Free(_) => requested,
            Precision::Fixed(n) => requested.min(*n),
        }
    }
}

impl Default for Precision {
    fn default() -> Self {
        Precision::free_default()
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::Free(n) => write!(f, "free (default {n})"),
            Precision::Fixed(n) => write!(f, "fixed {n}"),
        }
    }
}

/// The flavour of a series ring (Chapter 49.1.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeriesKind {
    /// `R[[x]]` — non-negative integral exponents.
    Power,
    /// `R((x))` — integral exponents, possibly negative.
    Laurent,
    /// `R⟨⟨x⟩⟩` — rational exponents.
    Puiseux,
}

impl fmt::Display for SeriesKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeriesKind::Power => write!(f, "Power series ring"),
            SeriesKind::Laurent => write!(f, "Laurent series ring"),
            SeriesKind::Puiseux => write!(f, "Puiseux series ring"),
        }
    }
}

/// A lightweight, value-typed series-ring parent (Chapter 49.2.1 / 49.3).
///
/// It records the kind of ring, its precision regime and the printed name of
/// the generator. It does not carry the coefficient ring as data (the element
/// types are generic over the coefficient ring `R`); instead it is a convenience
/// for driving element construction with a consistent default precision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeriesRing {
    kind: SeriesKind,
    precision: Precision,
    var_name: String,
}

impl SeriesRing {
    /// `PowerSeriesRing(R)` / `PowerSeriesRing(R, p)`.
    pub fn power(precision: Precision) -> Self {
        SeriesRing {
            kind: SeriesKind::Power,
            precision,
            var_name: "x".to_string(),
        }
    }

    /// `LaurentSeriesRing(R)` / `LaurentSeriesRing(R, p)`.
    pub fn laurent(precision: Precision) -> Self {
        SeriesRing {
            kind: SeriesKind::Laurent,
            precision,
            var_name: "x".to_string(),
        }
    }

    /// `PuiseuxSeriesRing(R)` / `PuiseuxSeriesRing(R, p)`.
    pub fn puiseux(precision: Precision) -> Self {
        SeriesRing {
            kind: SeriesKind::Puiseux,
            precision,
            var_name: "x".to_string(),
        }
    }

    /// Set the generator name (`AssignNames`).
    pub fn with_var_name(mut self, name: impl Into<String>) -> Self {
        self.var_name = name.into();
        self
    }

    /// The kind of ring.
    pub fn kind(&self) -> SeriesKind {
        self.kind
    }

    /// The precision regime.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// The generator name.
    pub fn var_name(&self) -> &str {
        &self.var_name
    }

    /// `ChangePrecision(R, r)` — a ring identical to this one but with a new
    /// (fixed) precision.
    pub fn change_precision(&self, r: usize) -> Self {
        SeriesRing {
            kind: self.kind,
            precision: Precision::Fixed(r),
            var_name: self.var_name.clone(),
        }
    }

    /// The number of terms a freshly created element should carry.
    pub fn default_terms(&self) -> usize {
        self.precision.default_terms()
    }
}

impl fmt::Display for SeriesRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} in {} ({})",
            self.kind, self.var_name, self.precision
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_cap_and_defaults() {
        assert_eq!(Precision::free_default(), Precision::Free(DEFAULT_PRECISION));
        assert_eq!(Precision::Free(30).cap(100), 100);
        assert_eq!(Precision::Fixed(10).cap(100), 10);
        assert_eq!(Precision::Fixed(10).cap(3), 3);
        assert!(Precision::Fixed(5).is_fixed());
        assert!(!Precision::Free(5).is_fixed());
    }

    #[test]
    fn series_ring_parent() {
        let r = SeriesRing::power(Precision::free_default()).with_var_name("t");
        assert_eq!(r.kind(), SeriesKind::Power);
        assert_eq!(r.var_name(), "t");
        assert_eq!(r.default_terms(), DEFAULT_PRECISION);

        let fixed = r.change_precision(7);
        assert_eq!(fixed.precision(), Precision::Fixed(7));
        assert_eq!(fixed.default_terms(), 7);

        let laur = SeriesRing::laurent(Precision::Fixed(12));
        assert_eq!(laur.kind(), SeriesKind::Laurent);
        assert_eq!(laur.default_terms(), 12);
    }
}
