//! Cusps of modular curves
//!
//! A cusp is a rational number p/q (including infinity) that represents
//! a point on the boundary of the upper half-plane.

use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;

/// A cusp of a modular curve, represented as p/q in lowest terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cusp {
    /// Rational cusp p/q
    Rational(Integer, Integer),
    /// The cusp at infinity
    Infinity,
}

impl Cusp {
    /// Create a new cusp from numerator and denominator
    pub fn new(p: Integer, q: Integer) -> Self {
        if q.is_zero() {
            Cusp::Infinity
        } else {
            // Reduce to lowest terms
            let g = p.gcd(&q);
            let mut p_reduced = &p / &g;
            let mut q_reduced = &q / &g;

            // Ensure denominator is positive
            if q_reduced.signum() < 0 {
                p_reduced = -p_reduced;
                q_reduced = -q_reduced;
            }

            Cusp::Rational(p_reduced, q_reduced)
        }
    }

    /// Create cusp from i64 values
    pub fn from_i64(p: i64, q: i64) -> Self {
        Cusp::new(Integer::from(p), Integer::from(q))
    }

    /// Create the cusp at 0
    pub fn zero() -> Self {
        Cusp::Rational(Integer::zero(), Integer::one())
    }

    /// Create the cusp at infinity
    pub fn infinity() -> Self {
        Cusp::Infinity
    }

    /// Convert to a rational number (None for infinity)
    pub fn to_rational(&self) -> Option<Rational> {
        match self {
            Cusp::Rational(p, q) => Some(
                Rational::new(p.clone(), q.clone())
                    .expect("Rational cusp has nonzero denominator"),
            ),
            Cusp::Infinity => None,
        }
    }

    /// Get numerator (None for infinity)
    pub fn numerator(&self) -> Option<&Integer> {
        match self {
            Cusp::Rational(p, _) => Some(p),
            Cusp::Infinity => None,
        }
    }

    /// Get denominator (None for infinity)
    pub fn denominator(&self) -> Option<&Integer> {
        match self {
            Cusp::Rational(_, q) => Some(q),
            Cusp::Infinity => None,
        }
    }

    /// Check if this is the cusp at infinity
    pub fn is_infinity(&self) -> bool {
        matches!(self, Cusp::Infinity)
    }

    /// Apply a matrix transformation to the cusp
    /// If [[a,b],[c,d]] acts on p/q, result is (ap+bq)/(cp+dq)
    pub fn apply_matrix(&self, a: &Integer, b: &Integer, c: &Integer, d: &Integer) -> Self {
        match self {
            Cusp::Rational(p, q) => {
                let new_p = a * p + b * q;
                let new_q = c * p + d * q;
                Cusp::new(new_p, new_q)
            }
            Cusp::Infinity => {
                // Infinity maps to a/c
                Cusp::new(a.clone(), c.clone())
            }
        }
    }

    /// Are the two cusps equivalent under SL(2, Z)?  ALWAYS TRUE.
    ///
    /// This is a theorem, not a shrug: SL(2, Z) acts TRANSITIVELY on
    /// `P^1(Q)`.  Given `p/q` in lowest terms, `gcd(p, q) = 1` gives `d`, `b` with
    /// `pd - qb = 1`, and then `[[p, b], [q, d]]` is in SL(2, Z) and carries
    /// `infinity = 1/0` to `p/q`.  So every cusp is SL(2,Z)-equivalent to
    /// `infinity`, and hence to every other cusp -- `X(1) = SL(2,Z) \ H*` has
    /// exactly ONE cusp.
    ///
    /// (This used to return "the two cusps differ by an integer", which is the
    /// equivalence under the TRANSLATION subgroup `<T>`, not under SL(2, Z): it
    /// reported the equivalent cusps 0 and infinity, and 1/2 and 1/3, as
    /// inequivalent.  That predicate is still available, correctly named, as
    /// [`Self::is_equivalent_translation`].  For the equivalence that actually
    /// varies from cusp to cusp, see the cusps of `Gamma_0(N)` --
    /// [`crate::etaproducts::all_cusps`].)
    pub fn is_equivalent_sl2z(&self, _other: &Cusp) -> bool {
        true
    }

    /// Are the two cusps equivalent under the translation subgroup `<T>` of
    /// SL(2, Z), i.e. do they differ by an integer?  (`infinity` only to itself.)
    ///
    /// This is the predicate [`Self::is_equivalent_sl2z`] used to compute under
    /// the wrong name.
    pub fn is_equivalent_translation(&self, other: &Cusp) -> bool {
        match (self, other) {
            (Cusp::Infinity, Cusp::Infinity) => true,
            (Cusp::Rational(p1, q1), Cusp::Rational(p2, q2)) => {
                q1 == q2 && (&(p1 - p2) % q1).is_zero()
            }
            _ => false,
        }
    }

    /// The width of this cusp in `Gamma_0(N)`: the smallest `h > 0` such that
    /// `sigma [[1, h], [0, 1]] sigma^{-1}` lies in `Gamma_0(N)`, where
    /// `sigma(infinity) = ` this cusp.
    ///
    /// For a cusp `a/c` in lowest terms (and `c = 0` for `infinity`),
    ///
    /// ```text
    ///     h = N / gcd(c^2, N).
    /// ```
    ///
    /// So `infinity` (`c = 0`, `gcd(0, N) = N`) has width 1, and the cusp
    /// `0 = 0/1` has width `N` -- and the widths sum to the index
    /// `[SL(2,Z) : Gamma_0(N)]`, which is what the tests check against the
    /// certified `dims::gamma0_invariants`.
    ///
    /// (This used to compute `N / gcd(c, N)`, which is a different number as soon
    /// as `gcd(c^2, N) != gcd(c, N)`: it gave width 2 for the cusp 1/2 of
    /// `Gamma_0(4)`, whose width is 1, and it gave width `N` for the cusp
    /// `infinity`, whose width is always 1 -- it had `infinity` and the cusp 0
    /// backwards.)
    pub fn width_gamma0(&self, level: u64) -> u64 {
        fn gcd_u64(a: u64, b: u64) -> u64 {
            if b == 0 { a } else { gcd_u64(b, a % b) }
        }
        let c: u64 = match self {
            Cusp::Infinity => 0,
            Cusp::Rational(_, q) => q
                .to_string()
                .parse::<u64>()
                .expect("a reduced cusp has a positive denominator"),
        };
        let c_squared_mod = ((c as u128 * c as u128) % (level as u128)) as u64;
        // gcd(c^2, N) = gcd(c^2 mod N, N), which avoids overflowing on c^2
        level / gcd_u64(level, c_squared_mod)
    }
}

impl fmt::Display for Cusp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cusp::Infinity => write!(f, "∞"),
            Cusp::Rational(p, q) => {
                if q.is_one() {
                    write!(f, "{}", p)
                } else {
                    write!(f, "{}/{}", p, q)
                }
            }
        }
    }
}

impl From<Rational> for Cusp {
    fn from(r: Rational) -> Self {
        Cusp::new(r.numerator().clone(), r.denominator().clone())
    }
}

impl From<i64> for Cusp {
    fn from(n: i64) -> Self {
        Cusp::from_i64(n, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cusp_creation() {
        let c1 = Cusp::from_i64(1, 2);
        assert_eq!(c1.numerator(), Some(&Integer::from(1)));
        assert_eq!(c1.denominator(), Some(&Integer::from(2)));

        let c2 = Cusp::from_i64(2, 4); // Should reduce to 1/2
        assert_eq!(c2.numerator(), Some(&Integer::from(1)));
        assert_eq!(c2.denominator(), Some(&Integer::from(2)));

        let inf = Cusp::infinity();
        assert!(inf.is_infinity());
        assert_eq!(inf.numerator(), None);
    }

    #[test]
    fn test_cusp_zero() {
        let c = Cusp::zero();
        assert_eq!(c.numerator(), Some(&Integer::zero()));
        assert_eq!(c.denominator(), Some(&Integer::one()));
    }

    #[test]
    fn test_cusp_zero_over_q_is_not_infinity() {
        // 0/q is the cusp 0, not the cusp at infinity, for every nonzero q.
        // (Previously `Cusp::new` special-cased a reduced form of
        // `Rational(0, 1)` and mapped it to `Infinity`, which conflated the
        // rational cusp 0 with the point at infinity.)
        for q in [1i64, 2, -2, 3, -3, 100] {
            let c = Cusp::from_i64(0, q);
            assert!(!c.is_infinity(), "0/{q} must not be Infinity");
            assert_eq!(c, Cusp::zero(), "0/{q} must reduce to the cusp 0");
            assert_eq!(c.numerator(), Some(&Integer::zero()));
            assert_eq!(c.denominator(), Some(&Integer::one()));
        }
    }

    #[test]
    fn test_cusp_zero_infinity_and_half_are_pairwise_distinct() {
        let zero = Cusp::zero();
        let infinity = Cusp::infinity();
        let half = Cusp::from_i64(1, 2);

        assert_ne!(zero, infinity);
        assert_ne!(zero, half);
        assert_ne!(infinity, half);

        assert!(!zero.is_infinity());
        assert!(infinity.is_infinity());
        assert!(!half.is_infinity());

        // NOTE: these three cusps are pairwise DISTINCT as points of P^1(Q) but
        // pairwise EQUIVALENT under SL(2, Z) (which is transitive on P^1(Q) --
        // X(1) has one cusp).  The old assertions here were
        // `!zero.is_equivalent_sl2z(&infinity)` etc., which pinned the old,
        // wrong predicate: S = [[0, -1], [1, 0]] carries 0 to infinity.
        assert!(zero.is_equivalent_sl2z(&infinity));
        assert!(zero.is_equivalent_sl2z(&half));
        assert!(infinity.is_equivalent_sl2z(&half));
    }

    /// SL(2, Z) is transitive on P^1(Q): for every cusp p/q there is an EXPLICIT
    /// matrix in SL(2, Z) carrying infinity to it.  This exhibits the witness, so
    /// `is_equivalent_sl2z == true` is checked, not just asserted.
    #[test]
    fn test_sl2z_is_transitive_on_cusps() {
        fn ext_gcd(a: i64, b: i64) -> (i64, i64, i64) {
            if b == 0 {
                (a, 1, 0)
            } else {
                let (g, x, y) = ext_gcd(b, a % b);
                (g, y, x - (a / b) * y)
            }
        }

        for q in 1i64..=12 {
            for p in -12i64..=12 {
                let (g, d, b) = ext_gcd(p, q); // p*d + q*b = g
                if g != 1 {
                    continue;
                }
                // [[p, -b], [q, d]] has determinant p*d + b*q = 1
                let m = (p, -b, q, d);
                assert_eq!(m.0 * m.3 - m.1 * m.2, 1, "witness must be in SL(2, Z)");

                // it carries infinity to p/q
                let image = Cusp::infinity().apply_matrix(
                    &Integer::from(m.0),
                    &Integer::from(m.1),
                    &Integer::from(m.2),
                    &Integer::from(m.3),
                );
                assert_eq!(image, Cusp::from_i64(p, q), "matrix must send oo to {p}/{q}");
                assert!(Cusp::infinity().is_equivalent_sl2z(&Cusp::from_i64(p, q)));
            }
        }
    }

    /// GATE: the cusp widths of Gamma_0(N) must sum to the index
    /// [SL(2,Z) : Gamma_0(N)], over the certified list of cusp representatives.
    /// `width_gamma0` used to return N/gcd(c, N), which fails this badly (it had
    /// the widths of the cusps 0 and infinity swapped).
    #[test]
    fn test_width_gamma0_sums_to_the_index() {
        for n in 1..=40u64 {
            let inv = crate::dims::gamma0_invariants(n).unwrap();
            let mut total = 0u128;
            for (a, d) in crate::etaproducts::all_cusps(Integer::from(n)) {
                let cusp = Cusp::new(a, d);
                total += cusp.width_gamma0(n) as u128;
            }
            assert_eq!(
                total, inv.index,
                "sum of cusp widths of Gamma_0({n}) must be the index"
            );
        }

        // Gamma_0(4): infinity has width 1, 0 has width 4, 1/2 has width 1.
        assert_eq!(Cusp::infinity().width_gamma0(4), 1);
        assert_eq!(Cusp::zero().width_gamma0(4), 4);
        assert_eq!(Cusp::from_i64(1, 2).width_gamma0(4), 1);
        // and the cusp infinity ALWAYS has width 1
        for n in 1..=20u64 {
            assert_eq!(Cusp::infinity().width_gamma0(n), 1, "width of oo in Gamma_0({n})");
        }
    }

    #[test]
    fn test_cusp_only_zero_denominator_is_infinity() {
        // The *only* route to `Cusp::Infinity` is a literal zero
        // denominator; a zero numerator with nonzero denominator must not
        // take that path.
        assert_eq!(Cusp::new(Integer::from(5), Integer::zero()), Cusp::Infinity);
        assert_eq!(Cusp::new(Integer::zero(), Integer::zero()), Cusp::Infinity);
        assert_ne!(
            Cusp::new(Integer::zero(), Integer::from(7)),
            Cusp::Infinity
        );
    }

    #[test]
    fn test_cusp_matrix_action() {
        // Apply [[1,1],[0,1]] (translation by 1) to 0
        let c = Cusp::zero();
        let result = c.apply_matrix(
            &Integer::one(),
            &Integer::one(),
            &Integer::zero(),
            &Integer::one(),
        );
        assert_eq!(result.numerator(), Some(&Integer::one()));
        assert_eq!(result.denominator(), Some(&Integer::one()));

        // Apply to infinity
        let inf = Cusp::infinity();
        let result_inf = inf.apply_matrix(
            &Integer::from(2),
            &Integer::from(3),
            &Integer::from(4),
            &Integer::from(5),
        );
        assert_eq!(result_inf.numerator(), Some(&Integer::from(1))); // 2/4 = 1/2
        assert_eq!(result_inf.denominator(), Some(&Integer::from(2)));
    }

    /// The TRANSLATION equivalence (differ by an integer) -- which is what the old
    /// `is_equivalent_sl2z` actually computed.  1/3 and 4/3 differ by 1; 1/3 and
    /// 1/2 do not.  (Under SL(2, Z) all three are equivalent; see
    /// `test_sl2z_is_transitive_on_cusps`.)
    #[test]
    fn test_cusp_equivalence() {
        let c1 = Cusp::from_i64(1, 3);
        let c2 = Cusp::from_i64(4, 3); // Differs by 1
        assert!(c1.is_equivalent_translation(&c2));

        let c3 = Cusp::from_i64(1, 2);
        assert!(!c1.is_equivalent_translation(&c3));

        // infinity is translation-equivalent only to itself
        assert!(Cusp::infinity().is_equivalent_translation(&Cusp::infinity()));
        assert!(!Cusp::infinity().is_equivalent_translation(&c1));

        // but SL(2, Z) identifies them all
        assert!(c1.is_equivalent_sl2z(&c3));
    }

    #[test]
    fn test_cusp_display() {
        assert_eq!(format!("{}", Cusp::from_i64(1, 2)), "1/2");
        assert_eq!(format!("{}", Cusp::from_i64(3, 1)), "3");
        assert_eq!(format!("{}", Cusp::infinity()), "∞");
    }
}
