//! Newton polygons over a discrete valuation.
//!
//! For a polynomial `f(x) = sum a_i x^i` over a field with valuation `v`
//! (e.g. `v = v_p` on `Q_p`), the **Newton polygon** is the lower convex hull
//! of the finite points `(i, v(a_i))` (coefficients with `a_i = 0`, i.e.
//! `v(a_i) = +infinity`, are simply omitted).
//!
//! # Sign convention (pinned)
//!
//! We take the **lower** convex hull, scanned left to right, so the segment
//! slopes are **weakly increasing**. A segment of slope `s` and horizontal
//! length `l` certifies that `f` has exactly `l` roots (in an algebraic
//! closure, counted with multiplicity) of valuation `m = -s`.
//!
//! Example: `x^2 - 2` over `Q_2` has points `(0, 1)` and `(2, 0)`; the single
//! hull segment has slope `-1/2` and length `2`, i.e. two roots of valuation
//! `+1/2` (indeed `±sqrt(2)` have `v_2 = 1/2`). Confusing this with slope
//! `+1/2` is the classic inversion error; the orientation is pinned by
//! explicit tests below (`test_orientation_*`).
//!
//! # What the polygon does NOT decide
//!
//! A slope-0 segment (roots of valuation 0) says nothing about how the unit
//! part factors. `x^2 - 17` over `Q_2` has the polygon `{slope 0, length 2}`;
//! it happens to split (17 = 1 mod 8 is a square in `Z_2`), while `x^2 - 3`
//! has the *same* polygon but is irreducible over `Q_2`. Unit-part splitting
//! requires residual factorization, not the polygon — that is exactly what
//! [`crate::padics::om_factorization::om_factorization`] decides (the full
//! MacLane/OM tree with residue-field towers).
//!
//! # Zero roots
//!
//! If `x^k` exactly divides `f`, the `k` zero roots ("valuation +infinity")
//! are not part of any segment; they are reported by
//! [`NewtonPolygon::num_infinite_slope_roots`].

use rustmath_core::{MathError, Result, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;

/// One segment of a Newton polygon.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewtonSlope {
    /// Slope of the hull segment (weakly increasing left to right).
    pub slope: Rational,
    /// Horizontal length of the segment (= number of certified roots).
    pub length: u64,
}

/// The Newton polygon (lower convex hull) of a set of valuation points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewtonPolygon {
    /// Hull vertices, abscissae strictly increasing.
    vertices: Vec<(i64, Rational)>,
    /// Number of omitted "infinite valuation" indices below the first finite
    /// point (for polynomials: multiplicity of the root 0).
    infinite_prefix: u64,
}

impl NewtonPolygon {
    /// Build the polygon from points `(i, v)` where `v = None` means
    /// `v = +infinity` (the point is omitted).
    ///
    /// Duplicate abscissae keep the lowest valuation (only that one can lie
    /// on the lower hull). At least one finite point is required.
    pub fn from_points<I>(points: I) -> Result<Self>
    where
        I: IntoIterator<Item = (i64, Option<Rational>)>,
    {
        let mut finite: Vec<(i64, Rational)> = vec![];
        let mut infinite: Vec<i64> = vec![];
        for (i, v) in points {
            match v {
                Some(v) => finite.push((i, v)),
                None => infinite.push(i),
            }
        }
        if finite.is_empty() {
            return Err(MathError::InvalidArgument(
                "NewtonPolygon: need at least one finite-valuation point".to_string(),
            ));
        }
        finite.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        finite.dedup_by(|later, earlier| {
            // after the sort, `earlier` has the smaller valuation at equal x
            later.0 == earlier.0
        });
        let first_finite = finite[0].0;
        let infinite_prefix = infinite.iter().filter(|&&i| i < first_finite).count() as u64;

        // Monotone-chain lower hull. Orientation (the classic trap): with
        // cross(a,b,c) = (bx-ax)(cy-ay) - (by-ay)(cx-ax), we DISCARD the
        // middle point b while cross <= 0, i.e. while b lies on or above the
        // segment a-c. This keeps the LOWER hull; pinned by tests.
        let mut hull: Vec<(i64, Rational)> = vec![];
        for c in finite {
            while hull.len() >= 2 {
                let a = &hull[hull.len() - 2];
                let b = &hull[hull.len() - 1];
                let dx_ab = Rational::from(b.0 - a.0);
                let dy_ac = c.1.clone() - a.1.clone();
                let dy_ab = b.1.clone() - a.1.clone();
                let dx_ac = Rational::from(c.0 - a.0);
                let cross = dx_ab * dy_ac - dy_ab * dx_ac;
                if cross <= Rational::from(0) {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(c);
        }
        Ok(NewtonPolygon {
            vertices: hull,
            infinite_prefix,
        })
    }

    /// Polygon of `f = sum a_i x^i` over `Q_p` from exact rational
    /// coefficients (little-endian). `a_i = 0` contributes no point.
    pub fn of_rational_polynomial(coeffs: &[Rational], p: &Integer) -> Result<Self> {
        if p <= &Integer::one() || !p.is_prime() {
            return Err(MathError::InvalidArgument(
                "NewtonPolygon: p must be prime".to_string(),
            ));
        }
        let points = coeffs.iter().enumerate().map(|(i, c)| {
            let v = if c.is_zero() {
                None
            } else {
                let vnum = c.numerator().valuation(p) as i64;
                let vden = c.denominator().valuation(p) as i64;
                Some(Rational::from(vnum - vden))
            };
            (i as i64, v)
        });
        Self::from_points(points)
    }

    /// Polygon of an integer polynomial over `Q_p`.
    pub fn of_integer_polynomial(
        poly: &UnivariatePolynomial<Integer>,
        p: &Integer,
    ) -> Result<Self> {
        let coeffs: Vec<Rational> = poly
            .coefficients()
            .iter()
            .map(|c| Rational::new(c.clone(), Integer::one()).expect("denominator 1"))
            .collect();
        Self::of_rational_polynomial(&coeffs, p)
    }

    /// Hull vertices `(i, v)`, abscissae strictly increasing.
    pub fn vertices(&self) -> &[(i64, Rational)] {
        &self.vertices
    }

    /// Segment slopes with horizontal lengths, left to right
    /// (slopes weakly increasing by convexity; strictly increasing since
    /// collinear points are merged into one segment).
    pub fn slopes(&self) -> Vec<NewtonSlope> {
        self.vertices
            .windows(2)
            .map(|w| {
                let (x0, y0) = (&w[0].0, &w[0].1);
                let (x1, y1) = (&w[1].0, &w[1].1);
                let dx = Rational::from(x1 - x0);
                NewtonSlope {
                    slope: (y1.clone() - y0.clone()) / dx,
                    length: (x1 - x0) as u64,
                }
            })
            .collect()
    }

    /// Certified root valuations `(m, l)`: `l` roots of valuation `m = -slope`,
    /// ordered by **decreasing** valuation (mirror of the increasing slopes).
    ///
    /// For `f` over `Q_p` this counts roots in an algebraic closure with
    /// multiplicity; roots equal to 0 (valuation `+infinity`) are *not*
    /// listed — see [`Self::num_infinite_slope_roots`].
    pub fn root_valuations(&self) -> Vec<(Rational, u64)> {
        self.slopes()
            .into_iter()
            .map(|s| (-s.slope, s.length))
            .collect()
    }

    /// Number of roots of valuation `+infinity` (for a polynomial: the
    /// multiplicity of `x = 0`, i.e. the largest `k` with `x^k | f`).
    pub fn num_infinite_slope_roots(&self) -> u64 {
        self.infinite_prefix
    }

    /// Total number of finite-valuation roots certified (sum of lengths;
    /// equals `deg f - ord_0 f` for a polynomial).
    pub fn num_finite_roots(&self) -> u64 {
        self.slopes().iter().map(|s| s.length).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64, d: i64) -> Rational {
        Rational::new(Integer::from(n), Integer::from(d)).unwrap()
    }

    fn poly(coeffs: &[i64]) -> UnivariatePolynomial<Integer> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| Integer::from(c)).collect())
    }

    fn root_vals(coeffs: &[i64], p: i64) -> Vec<(Rational, u64)> {
        NewtonPolygon::of_integer_polynomial(&poly(coeffs), &Integer::from(p))
            .unwrap()
            .root_valuations()
    }

    /// Pin the hull orientation: a dip must be kept...
    #[test]
    fn test_orientation_dip_kept() {
        let np = NewtonPolygon::from_points(vec![
            (0, Some(rat(0, 1))),
            (1, Some(rat(-1, 1))),
            (2, Some(rat(0, 1))),
        ])
        .unwrap();
        assert_eq!(
            np.vertices(),
            &[(0, rat(0, 1)), (1, rat(-1, 1)), (2, rat(0, 1))]
        );
    }

    /// ...and a bump must be skipped (the classic inverted-hull bug).
    #[test]
    fn test_orientation_bump_skipped() {
        let np = NewtonPolygon::from_points(vec![
            (0, Some(rat(0, 1))),
            (1, Some(rat(1, 1))),
            (2, Some(rat(0, 1))),
        ])
        .unwrap();
        assert_eq!(np.vertices(), &[(0, rat(0, 1)), (2, rat(0, 1))]);
    }

    /// Gate (sympy-verified): x^2 - 2 over Q_2 -> single slope -1/2, length 2,
    /// i.e. two roots of valuation +1/2. Also pins the SIGN: the slope is
    /// -1/2 and the root valuation is +1/2, not the other way round.
    #[test]
    fn test_x2_minus_2_over_q2() {
        let np = NewtonPolygon::of_integer_polynomial(&poly(&[-2, 0, 1]), &Integer::from(2))
            .unwrap();
        let slopes = np.slopes();
        assert_eq!(slopes.len(), 1);
        assert_eq!(slopes[0].slope, rat(-1, 2));
        assert_eq!(slopes[0].length, 2);
        assert_eq!(np.root_valuations(), vec![(rat(1, 2), 2)]);
        assert_eq!(np.num_infinite_slope_roots(), 0);
        assert_eq!(np.num_finite_roots(), 2);
    }

    /// Gate (sympy-verified): x^2 - 17 over Q_2 -> slope 0, length 2. The
    /// polygon does NOT decide the unit-part splitting (here 17 = 1 mod 8 is
    /// in fact a square in Z_2, while x^2 - 3 with the same polygon is
    /// irreducible) — that requires residual factorization.
    #[test]
    fn test_x2_minus_17_over_q2() {
        assert_eq!(root_vals(&[-17, 0, 1], 2), vec![(rat(0, 1), 2)]);
        // same polygon for the inert x^2 - 3: polygon can't tell them apart
        assert_eq!(root_vals(&[-3, 0, 1], 2), vec![(rat(0, 1), 2)]);
    }

    /// Gate (sympy-verified): x^3 + 2x + 2 IS Eisenstein at 2 (a_0 has v = 1
    /// exactly, a_1 has v = 1 >= 1, monic) -> single slope -1/3, length 3:
    /// three roots of valuation 1/3. Hand-derived: points (0,1),(1,1),(3,0);
    /// the point (1,1) lies above the segment (0,1)-(3,0) (which passes
    /// through y = 2/3 at x = 1), so it is not a vertex.
    #[test]
    fn test_x3_plus_2x_plus_2_over_q2() {
        let np = NewtonPolygon::of_integer_polynomial(&poly(&[2, 2, 0, 1]), &Integer::from(2))
            .unwrap();
        assert_eq!(np.vertices(), &[(0, rat(1, 1)), (3, rat(0, 1))]);
        assert_eq!(np.root_valuations(), vec![(rat(1, 3), 3)]);
    }

    /// Mixed slopes (sympy-verified): (x-1)(x-2)(x-4) = x^3 - 7x^2 + 14x - 8
    /// over Q_2 has roots of valuation 0, 1, 2 — three segments.
    #[test]
    fn test_mixed_slopes_split_cubic() {
        assert_eq!(
            root_vals(&[-8, 14, -7, 1], 2),
            vec![(rat(2, 1), 1), (rat(1, 1), 1), (rat(0, 1), 1)]
        );
    }

    /// Mixed slopes (sympy-verified): 2x^2 + x + 4 over Q_2 -> roots of
    /// valuation 2 and -1 (product of roots = 2 has v = 1 = 2 + (-1); sum
    /// -1/2 has v = -1: consistent). Exercises a NEGATIVE root valuation
    /// (positive hull slope).
    #[test]
    fn test_mixed_slopes_nonmonic() {
        assert_eq!(
            root_vals(&[4, 1, 2], 2),
            vec![(rat(2, 1), 1), (rat(-1, 1), 1)]
        );
    }

    /// Mixed slopes (sympy-verified): (x^2-2)(x-4) over Q_2 -> one root of
    /// valuation 2 and two of valuation 1/2 (multiplicity on a segment).
    #[test]
    fn test_mixed_slopes_with_multiplicity() {
        assert_eq!(
            root_vals(&[8, -2, -4, 1], 2),
            vec![(rat(2, 1), 1), (rat(1, 2), 2)]
        );
    }

    /// Rational coefficients (sympy-verified): (x - 1/2)(x - 2) over Q_2 ->
    /// valuations 1 and -1.
    #[test]
    fn test_rational_coefficients() {
        let coeffs = vec![rat(1, 1), rat(-5, 2), rat(1, 1)];
        let np = NewtonPolygon::of_rational_polynomial(&coeffs, &Integer::from(2)).unwrap();
        assert_eq!(
            np.root_valuations(),
            vec![(rat(1, 1), 1), (rat(-1, 1), 1)]
        );
    }

    /// p = 3 (sympy-verified): (x-3)(x-9)(x-1) -> valuations 0, 1, 2 over Q_3.
    #[test]
    fn test_p3_split_cubic() {
        let f = poly(&[-27, 39, -13, 1]); // expand((x-3)(x-9)(x-1))
        let np = NewtonPolygon::of_integer_polynomial(&f, &Integer::from(3)).unwrap();
        assert_eq!(
            np.root_valuations(),
            vec![(rat(2, 1), 1), (rat(1, 1), 1), (rat(0, 1), 1)]
        );
    }

    /// x^k | f: the zero roots are reported separately, not as a segment.
    #[test]
    fn test_infinite_prefix() {
        // x^2 (x^2 - 2) = x^4 - 2x^2 over Q_2
        let np = NewtonPolygon::of_integer_polynomial(&poly(&[0, 0, -2, 0, 1]), &Integer::from(2))
            .unwrap();
        assert_eq!(np.num_infinite_slope_roots(), 2);
        assert_eq!(np.root_valuations(), vec![(rat(1, 2), 2)]);
        assert_eq!(np.num_finite_roots(), 2);
    }

    #[test]
    fn test_zero_polynomial_rejected() {
        assert!(NewtonPolygon::of_rational_polynomial(&[], &Integer::from(2)).is_err());
        assert!(NewtonPolygon::from_points(vec![(0, None), (3, None)]).is_err());
    }

    /// Collinear points merge into a single segment (no spurious vertices).
    #[test]
    fn test_collinear_merged() {
        let np = NewtonPolygon::from_points(vec![
            (0, Some(rat(2, 1))),
            (1, Some(rat(1, 1))),
            (2, Some(rat(0, 1))),
        ])
        .unwrap();
        assert_eq!(np.vertices(), &[(0, rat(2, 1)), (2, rat(0, 1))]);
        assert_eq!(np.root_valuations(), vec![(rat(1, 1), 2)]);
    }
}
