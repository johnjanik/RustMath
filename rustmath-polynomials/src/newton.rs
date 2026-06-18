//! Newton polygon of `f(x) = Σ aᵢ xⁱ ∈ ℤ[x]` at a finite prime `p`.
//!
//! The polygon is the lower convex hull of the points `(i, v_p(aᵢ))` (skipping
//! `aᵢ = 0`). The slope of a hull segment is the negative of the `p`-adic
//! valuation of the roots it accounts for; the segment's horizontal length is the
//! number of such roots. This is the entry point for local analysis: it reads off
//! root valuations and the ramification structure feeding [`crate::padic_factor`].
//!
//! Ported from the p-adelic engine's `NewtonPolygon.swift`.

use rustmath_integers::Integer;

/// A hull point in `(coefficient index, valuation)` coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vertex {
    pub i: usize,
    pub v: i64,
}

/// A lower-hull segment, slope in lowest terms (denominator `> 0`, sign on the
/// numerator). `length = i_to − i_from` is the number of roots of valuation
/// `−slope` it accounts for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Segment {
    pub from: Vertex,
    pub to: Vertex,
    pub slope_num: i64, // signed
    pub slope_den: i64, // > 0
    pub length: usize,  // i_to − i_from
}

/// A Newton polygon: hull vertices left→right and the segments between them.
#[derive(Debug, Clone)]
pub struct NewtonPolygon {
    pub vertices: Vec<Vertex>,
    pub segments: Vec<Segment>,
}

/// `p`-adic valuation of a nonzero integer; `i64::MAX` for zero (treated as ∞).
pub fn int_valuation(n: &Integer, p: i64) -> i64 {
    if n.is_zero() {
        return i64::MAX;
    }
    let pi = Integer::from(p);
    let mut x = n.abs();
    let mut v = 0i64;
    while (x.clone() % pi.clone()).is_zero() {
        x = x / pi.clone();
        v += 1;
    }
    v
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut x, mut y) = (a.abs(), b.abs());
    while y != 0 {
        let t = x % y;
        x = y;
        y = t;
    }
    x
}

/// Cross-product test for the lower hull: pop the middle point `b` when the turn
/// `a→b→c` is clockwise or collinear (`cross ≤ 0`), i.e. `b` lies on or above the
/// chord `a→c`. (Andrew's monotone chain, lower hull.)
fn should_pop(a: Vertex, b: Vertex, c: Vertex) -> bool {
    let cross = (b.i as i64 - a.i as i64) * (c.v - a.v) - (b.v - a.v) * (c.i as i64 - a.i as i64);
    cross <= 0
}

/// Build the Newton polygon of `f` at `p`. Returns `None` for the zero polynomial.
pub fn newton_polygon(f: &[Integer], p: i64) -> Option<NewtonPolygon> {
    let points: Vec<Vertex> = f
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(i, c)| Vertex { i, v: int_valuation(c, p) })
        .collect();
    if points.is_empty() {
        return None;
    }

    // Andrew's monotone chain on points already sorted by i ascending.
    let mut hull: Vec<Vertex> = Vec::new();
    for &pt in &points {
        while hull.len() >= 2 {
            let q1 = hull[hull.len() - 1];
            let q2 = hull[hull.len() - 2];
            if should_pop(q2, q1, pt) {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(pt);
    }

    let mut segments = Vec::new();
    for k in 1..hull.len() {
        let a = hull[k - 1];
        let b = hull[k];
        let dx = b.i as i64 - a.i as i64;
        let dy = b.v - a.v;
        let g = gcd_i64(dy, dx).max(1);
        segments.push(Segment {
            from: a,
            to: b,
            slope_num: dy / g,
            slope_den: (dx / g).abs(),
            length: dx as usize,
        });
    }

    Some(NewtonPolygon { vertices: hull, segments })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(cs: &[i64]) -> Vec<Integer> {
        cs.iter().map(|&c| Integer::from(c)).collect()
    }

    #[test]
    fn test_valuation() {
        assert_eq!(int_valuation(&Integer::from(72), 3), 2); // 72 = 8·9
        assert_eq!(int_valuation(&Integer::from(5), 3), 0);
        assert_eq!(int_valuation(&Integer::from(-54), 3), 3); // 54 = 2·27
    }

    #[test]
    fn test_unramified_flat_polygon() {
        // x^2 + x + 1 at p=5: v(a_0)=v(a_1)=v(a_2)=0 → single slope-0 segment
        let np = newton_polygon(&p(&[1, 1, 1]), 5).unwrap();
        assert_eq!(np.segments.len(), 1);
        assert_eq!(np.segments[0].slope_num, 0);
        assert_eq!(np.segments[0].length, 2);
    }

    #[test]
    fn test_eisenstein_polygon() {
        // x^3 + 3x + 3 at p=3: Eisenstein. Points (0,1),(1,1),(3,0).
        // Lower hull (0,1)→(3,0): single segment slope -1/3, length 3.
        let np = newton_polygon(&p(&[3, 3, 0, 1]), 3).unwrap();
        assert_eq!(np.segments.len(), 1);
        assert_eq!(np.segments[0].slope_num, -1);
        assert_eq!(np.segments[0].slope_den, 3);
        assert_eq!(np.segments[0].length, 3);
    }

    #[test]
    fn test_mixed_two_segments() {
        // f = x·(x^2 + 3) roots: 0 (val 0) and ±√-3 (val 1/2) at p=3.
        // f = x^3 + 3x. Coeffs [0,3,0,1] but a_0=0 skipped; points (1,1),(3,0).
        // That's one segment. Use (x^2+3)(x-1) = x^3 - x^2 + 3x - 3:
        // coeffs [-3,3,-1,1]; vals at p=3: (0,1),(1,1),(2,0),(3,0).
        // Lower hull: (0,1)→(2,0)→(3,0): slope -1/2 (len 2) then slope 0 (len 1).
        let np = newton_polygon(&p(&[-3, 3, -1, 1]), 3).unwrap();
        assert_eq!(np.segments.len(), 2);
        assert_eq!((np.segments[0].slope_num, np.segments[0].slope_den), (-1, 2));
        assert_eq!(np.segments[0].length, 2);
        assert_eq!(np.segments[1].slope_num, 0);
        assert_eq!(np.segments[1].length, 1);
    }
}
