//! Places of `ℚ(t)` and their valuations — the base for ramification / genus.
//!
//! A place of the rational function field `ℚ(t)` is either
//! - **finite**: a monic irreducible `q(t) ∈ ℚ[t]` (residue field `ℚ[t]/(q)`,
//!   degree `deg q`), with valuation `v_q(p) = mult_q(p)`; or
//! - **infinite**: the place at `t = ∞` (uniformizer `1/t`, residue field `ℚ`,
//!   degree `1`), with valuation `v_∞(p/r) = deg r − deg p`.
//!
//! This is **Layer 0 / A1** of the function-field stack
//! (`M23/rustmath_function_field_plan.md`): the substrate the Montes /
//! Newton-polygon ramification computation runs on.

use crate::ratfunc::{QtPoly, RationalFunction};

/// A place of `ℚ(t)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Place {
    /// A finite place: a monic irreducible `q(t) ∈ ℚ[t]`.
    Finite(QtPoly),
    /// The infinite place `t = ∞`.
    Infinite,
}

impl Place {
    /// The residue degree `deg q` (finite) or `1` (infinite) — the `deg(P)`
    /// weight in the different/genus sum.
    pub fn degree(&self) -> usize {
        match self {
            Place::Finite(q) => q.degree().unwrap_or(0),
            Place::Infinite => 1,
        }
    }

    /// `v_P` of a rational function. Returns `i64::MAX` for the zero element.
    pub fn valuation(&self, r: &RationalFunction) -> i64 {
        if r.numerator().is_zero() {
            return i64::MAX;
        }
        match self {
            Place::Finite(q) => {
                mult_of(q, r.numerator()) - mult_of(q, r.denominator())
            }
            Place::Infinite => {
                deg(r.denominator()) - deg(r.numerator())
            }
        }
    }
}

/// `deg` as an `i64` (the zero polynomial reports `-1`, never used here).
fn deg(p: &QtPoly) -> i64 {
    p.degree().map(|d| d as i64).unwrap_or(-1)
}

/// Multiplicity of a (nonzero) factor `q` in `p`: largest `k` with `q^k | p`.
/// `q` need not be irreducible; for genus use it is.
pub fn mult_of(q: &QtPoly, p: &QtPoly) -> i64 {
    if p.is_zero() || q.degree().map_or(true, |d| d == 0) {
        return 0;
    }
    let mut cur = p.clone();
    let mut k = 0i64;
    loop {
        let (quo, rem) = cur.quo_rem(q);
        if rem.is_zero() {
            cur = quo;
            k += 1;
            if cur.degree().map_or(true, |d| d == 0) {
                break;
            }
        } else {
            break;
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn qt(coeffs: &[i64]) -> QtPoly {
        QtPoly::new(coeffs.iter().map(|&c| Rational::from_i64(c)).collect())
    }

    #[test]
    fn finite_place_valuation() {
        // q = t + 1, p = (t+1)^2 (t-3): v_q(p) = 2.
        let q = qt(&[1, 1]);
        let p = qt(&[1, 1]) // (t+1)
            .clone();
        let p2 = p.clone() * p.clone() * qt(&[-3, 1]); // (t+1)^2 (t-3)
        let r = RationalFunction::new(p2, QtPoly::one()).unwrap();
        let place = Place::Finite(q);
        assert_eq!(place.valuation(&r), 2);
        assert_eq!(place.degree(), 1);
    }

    #[test]
    fn infinite_place_valuation() {
        // v_∞(t^3 + 1) = -3 ; v_∞(1/(t^2)) = +2.
        let p = qt(&[1, 0, 0, 1]); // t^3 + 1
        let r = RationalFunction::new(p, QtPoly::one()).unwrap();
        assert_eq!(Place::Infinite.valuation(&r), -3);
        let inv = RationalFunction::new(QtPoly::one(), qt(&[0, 0, 1])).unwrap();
        assert_eq!(Place::Infinite.valuation(&inv), 2);
    }
}
