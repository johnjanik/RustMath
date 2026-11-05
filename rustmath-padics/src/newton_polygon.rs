//! Newton polygon of `f(x) = Σ aᵢ xⁱ ∈ ℤ[x]` over `ℚ_p`.
//!
//! The polygon is the lower convex hull of the points `(i, v_p(aᵢ))` (skipping
//! `aᵢ = 0`). Each hull *face* (segment) has a rational slope `−λ`; the negative
//! slope `λ = h/e` (lowest terms, `e > 0`) is the common `p`-adic valuation of the
//! `length` roots that face accounts for, and `e` is their ramification index. This
//! is the input to the Montes/Ore factorization in [`crate::montes`].
//!
//! Kept self-contained inside the `rustmath-padics` crate (only depends on
//! `rustmath-integers`) so the local-analysis stack lives with the p-adics without a
//! polynomials dependency. It mirrors `rustmath-polynomials::newton`.

use rustmath_integers::Integer;

/// A hull point in `(coefficient index, valuation)` coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vertex {
    pub i: usize,
    pub v: i64,
}

/// A lower-hull face (segment). The negative slope `−slope = h/e` is written as
/// `neg_slope_num/neg_slope_den` in lowest terms with `neg_slope_den > 0`; it is the
/// common valuation `h/e` of the roots this face accounts for, with ramification
/// index `e = neg_slope_den`. `length = i_to − i_from` is the number of such roots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Face {
    pub from: Vertex,
    pub to: Vertex,
    /// `h` in `−slope = h/e` (`≥ 0` for the lower hull of a polynomial).
    pub neg_slope_num: i64,
    /// `e` in `−slope = h/e` (`> 0`).
    pub neg_slope_den: i64,
    /// Horizontal length `i_to − i_from` (number of roots on this face).
    pub length: usize,
}

impl Face {
    /// The (true, signed) slope of the face as a rational `slope_num/slope_den`.
    pub fn slope(&self) -> (i64, i64) {
        (-self.neg_slope_num, self.neg_slope_den)
    }

    /// Ramification index contributed by roots on this face: `e = neg_slope_den`.
    pub fn ramification(&self) -> usize {
        self.neg_slope_den as usize
    }

    /// Degree of the face's Ore residual polynomial: `t = length / e`.
    pub fn residual_degree(&self) -> usize {
        self.length / self.ramification()
    }
}

/// A Newton polygon: hull vertices left→right and the faces between them.
#[derive(Debug, Clone)]
pub struct NewtonPolygon {
    pub vertices: Vec<Vertex>,
    pub faces: Vec<Face>,
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

/// Build the Newton polygon of `f` (little-endian coefficients) at `p`. Returns
/// `None` for the zero polynomial.
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

    let mut faces = Vec::new();
    for k in 1..hull.len() {
        let a = hull[k - 1];
        let b = hull[k];
        let dx = b.i as i64 - a.i as i64; // > 0
        let dy = b.v - a.v; // slope = dy/dx ≤ 0 on the lower hull
        let neg_dy = -dy; // h ≥ 0
        let g = gcd_i64(neg_dy, dx).max(1);
        faces.push(Face {
            from: a,
            to: b,
            neg_slope_num: neg_dy / g,
            neg_slope_den: (dx / g).abs(),
            length: dx as usize,
        });
    }

    Some(NewtonPolygon { vertices: hull, faces })
}

/// The "principal" part of the polygon — the faces with strictly positive `h`
/// (i.e. roots of positive valuation). The slope-0 face (unramified part) is the
/// complement. Convenience for ramification analysis.
pub fn principal_faces(np: &NewtonPolygon) -> Vec<Face> {
    np.faces.iter().copied().filter(|f| f.neg_slope_num > 0).collect()
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
    fn test_flat_polygon() {
        // x^2 + x + 1 at p=5: a single negative-slope-0 face, length 2.
        let np = newton_polygon(&p(&[1, 1, 1]), 5).unwrap();
        assert_eq!(np.faces.len(), 1);
        assert_eq!(np.faces[0].neg_slope_num, 0);
        assert_eq!(np.faces[0].neg_slope_den, 1);
        assert_eq!(np.faces[0].length, 2);
        assert_eq!(np.faces[0].slope(), (0, 1));
    }

    #[test]
    fn test_eisenstein_polygon() {
        // x^3 + 3x + 3 at p=3: Eisenstein. Points (0,1),(1,1),(3,0).
        // Single face (0,1)→(3,0): slope −1/3, h=1, e=3, length 3.
        let np = newton_polygon(&p(&[3, 3, 0, 1]), 3).unwrap();
        assert_eq!(np.faces.len(), 1);
        assert_eq!(np.faces[0].neg_slope_num, 1);
        assert_eq!(np.faces[0].neg_slope_den, 3);
        assert_eq!(np.faces[0].length, 3);
        assert_eq!(np.faces[0].ramification(), 3);
        assert_eq!(np.faces[0].residual_degree(), 1);
        assert_eq!(np.faces[0].slope(), (-1, 3));
    }

    #[test]
    fn test_mixed_two_faces() {
        // (x^2+3)(x-1) = x^3 - x^2 + 3x - 3 at p=3.
        // vals: (0,1),(1,1),(2,0),(3,0). Hull: (0,1)→(2,0)→(3,0).
        let np = newton_polygon(&p(&[-3, 3, -1, 1]), 3).unwrap();
        assert_eq!(np.faces.len(), 2);
        assert_eq!(np.faces[0].slope(), (-1, 2)); // h=1,e=2, len 2
        assert_eq!(np.faces[0].length, 2);
        assert_eq!(np.faces[1].slope(), (0, 1)); // unramified, len 1
        assert_eq!(np.faces[1].length, 1);
        assert_eq!(principal_faces(&np).len(), 1);
    }
}
