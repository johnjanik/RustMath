//! Discrete valuations and places.
//!
//! MAGMA source: Handbook chapters 45 (valuation rings), 47/51 (p-adics),
//! 41–43 (function fields). A place is a normalized discrete valuation; for
//! global fields the set of places splits into *finite* places (attached to a
//! prime element / maximal ideal, carrying a residue degree) and *infinite* /
//! *degree* places (the archimedean places of a number field, or the degree
//! place at infinity of a function field).
//!
//! This module is purely additive.

/// A discrete valuation `v : R -> Z ∪ {∞}` on a ring or field `R`.
///
/// `valuation` returns the (finite) valuation of a non-zero element; the
/// valuation of zero is conventionally `+∞` and is represented by the sentinel
/// [`DiscreteValuation::INFINITY`].
pub trait DiscreteValuation<R> {
    /// Sentinel value used to represent `v(0) = +∞`.
    const INFINITY: i64 = i64::MAX;

    /// The valuation `v(x)`; returns [`DiscreteValuation::INFINITY`] for the
    /// zero element.
    fn valuation(&self, x: &R) -> i64;

    /// A uniformizer: an element `π` with `v(π) = 1`.
    fn uniformizer(&self) -> R;

    /// Whether `x` lies in the valuation ring, i.e. `v(x) >= 0`.
    fn is_integral(&self, x: &R) -> bool {
        self.valuation(x) >= 0
    }

    /// Whether `x` is a unit of the valuation ring, i.e. `v(x) = 0`.
    fn is_unit(&self, x: &R) -> bool {
        self.valuation(x) == 0
    }
}

/// A place (a normalized valuation) of a global field.
///
/// A place is *sign-aware*: an infinite/archimedean place carries a sign
/// (`+1` / `-1` for the two real embeddings of a conjugate pair, `0` for a
/// genuinely complex place), while a finite place or a function-field degree
/// place carries a residue/place degree. `P` is the type of the prime element
/// (or ideal generator) underlying a finite place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place<P> {
    /// `Some(prime)` for a finite place, `None` for an infinite/degree place.
    prime: Option<P>,
    /// Residue degree (finite places) or degree of the place (function-field
    /// infinite place). Always `1` for the archimedean places of a number field.
    degree: u64,
    /// Orientation sign: meaningful for archimedean places (`+1`/`-1`/`0`),
    /// `+1` by convention for finite and degree places.
    sign: i32,
}

impl<P> Place<P> {
    /// Construct a finite place from a prime element / ideal generator and its
    /// residue degree (`degree >= 1`).
    pub fn finite(prime: P, degree: u64) -> Self {
        Place {
            prime: Some(prime),
            degree: degree.max(1),
            sign: 1,
        }
    }

    /// Construct an archimedean (infinite) place with the given orientation
    /// sign (`+1`/`-1` for real embeddings, `0` for a complex place).
    pub fn infinite(sign: i32) -> Self {
        Place {
            prime: None,
            degree: 1,
            sign,
        }
    }

    /// Construct the degree (infinite) place of a function field, carrying its
    /// degree.
    pub fn infinite_degree(degree: u64) -> Self {
        Place {
            prime: None,
            degree: degree.max(1),
            sign: 1,
        }
    }

    /// Whether this is a finite place.
    pub fn is_finite(&self) -> bool {
        self.prime.is_some()
    }

    /// Whether this is an infinite / degree place.
    pub fn is_infinite(&self) -> bool {
        self.prime.is_none()
    }

    /// The residue/place degree.
    pub fn degree(&self) -> u64 {
        self.degree
    }

    /// The orientation sign of the place.
    pub fn sign(&self) -> i32 {
        self.sign
    }

    /// The prime element underlying a finite place, if any.
    pub fn prime(&self) -> Option<&P> {
        self.prime.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `p`-adic valuation on `i64` for a small prime `p`, used to exercise the
    /// [`DiscreteValuation`] contract without pulling in `rustmath-integers`
    /// (which would create a dependency cycle for `rustmath-core`).
    struct PAdic {
        p: i64,
    }

    impl DiscreteValuation<i64> for PAdic {
        fn valuation(&self, x: &i64) -> i64 {
            if *x == 0 {
                return Self::INFINITY;
            }
            let mut n = x.abs();
            let mut v = 0;
            while n % self.p == 0 {
                n /= self.p;
                v += 1;
            }
            v
        }

        fn uniformizer(&self) -> i64 {
            self.p
        }
    }

    #[test]
    fn test_padic_valuation() {
        let v2 = PAdic { p: 2 };
        assert_eq!(v2.valuation(&40), 3); // 40 = 2^3 * 5
        assert_eq!(v2.valuation(&5), 0);
        assert_eq!(v2.valuation(&0), i64::MAX);
        assert_eq!(v2.uniformizer(), 2);
        assert!(v2.is_unit(&5));
        assert!(!v2.is_unit(&4));
        assert!(v2.is_integral(&4));
    }

    #[test]
    fn test_places() {
        let finite = Place::finite(7i64, 2);
        assert!(finite.is_finite());
        assert_eq!(finite.degree(), 2);
        assert_eq!(finite.prime(), Some(&7));
        assert_eq!(finite.sign(), 1);

        let real_plus = Place::<i64>::infinite(1);
        let real_minus = Place::<i64>::infinite(-1);
        assert!(real_plus.is_infinite());
        assert_ne!(real_plus, real_minus);
        assert_eq!(real_minus.sign(), -1);

        let deg = Place::<i64>::infinite_degree(3);
        assert!(deg.is_infinite());
        assert_eq!(deg.degree(), 3);
    }
}
