//! # The projective line P^1(Z/NZ)
//!
//! Enumeration and canonical normalization of the projective line over
//! Z/NZ, indexing the right cosets of Gamma0(N) in SL(2, Z) via the bottom
//! row (c : d) of a coset representative.  This is the combinatorial
//! backbone of the Manin-symbol presentation of modular symbols.
//!
//! Corresponds to `sage.modular.modsym.p1list` and the MAGMA handbook
//! chapter "Modular Symbols" (creation of modular symbol spaces / Manin
//! symbols).  The canonical form follows Cremona, *Algorithms for Modular
//! Elliptic Curves*, section 2.2 (see also Stein, *Modular Forms: A
//! Computational Approach*, chapter 8): the representative of the orbit of
//! (c : d) under scaling by units of Z/NZ is the lexicographically smallest
//! unit multiple, whose first coordinate is gcd(c, N).
//!
//! Size: |P^1(Z/NZ)| = N * prod_{p | N} (1 + 1/p) = [SL2(Z) : Gamma0(N)].

use std::collections::HashMap;

/// gcd for u64.
fn gcd_u64(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd_u64(b, a % b)
    }
}

/// Extended gcd on i64: returns (g, x, y) with x*a + y*b = g and g = gcd(a,b) >= 0.
fn xgcd_i64(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1i64, 0i64);
    let (mut old_t, mut t) = (0i64, 1i64);
    while r != 0 {
        let q = old_r / r; // truncated division keeps the Bezout invariant
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
        (old_t, t) = (t, old_t - q * t);
    }
    if old_r < 0 {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    }
}

/// The projective line P^1(Z/NZ): canonical representatives (c : d) with an
/// O(1) index lookup for arbitrary pairs.
///
/// For N = 1 the projective line is the single point represented as (0, 0),
/// matching the SageMath convention.
#[derive(Debug, Clone)]
pub struct P1List {
    n: u64,
    list: Vec<(u64, u64)>,
    /// Every valid residue pair (c, d) in [0,N)^2 -> index of its orbit.
    index: HashMap<(u64, u64), usize>,
}

impl P1List {
    /// Build the projective line over Z/NZ.  N must be positive.
    pub fn new(n: u64) -> Self {
        assert!(n > 0, "level must be positive");
        if n == 1 {
            let mut index = HashMap::new();
            index.insert((0, 0), 0);
            return P1List {
                n,
                list: vec![(0, 0)],
                index,
            };
        }
        let units: Vec<u64> = (1..n).filter(|&u| gcd_u64(u, n) == 1).collect();
        let mut list = Vec::new();
        let mut index = HashMap::new();
        let mut canon_index: HashMap<(u64, u64), usize> = HashMap::new();
        for c in 0..n {
            for d in 0..n {
                if gcd_u64(gcd_u64(c, d), n) != 1 {
                    continue;
                }
                // canonical representative: lexicographically smallest unit
                // multiple (equivalently Cremona's (gcd(c,N), v_min) form)
                let mut best = (u64::MAX, u64::MAX);
                for &u in &units {
                    let cand = (u * c % n, u * d % n);
                    if cand < best {
                        best = cand;
                    }
                }
                let idx = match canon_index.get(&best) {
                    Some(&i) => i,
                    None => {
                        let i = list.len();
                        list.push(best);
                        canon_index.insert(best, i);
                        i
                    }
                };
                index.insert((c, d), idx);
            }
        }
        P1List { n, list, index }
    }

    /// The level N.
    pub fn level(&self) -> u64 {
        self.n
    }

    /// Number of points, |P^1(Z/NZ)| = N * prod_{p|N} (1 + 1/p).
    pub fn len(&self) -> usize {
        self.list.len()
    }

    /// True iff the list is empty (never, for a valid level).
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// The canonical representatives, in enumeration order.
    pub fn list(&self) -> &[(u64, u64)] {
        &self.list
    }

    /// Canonical representative of (c : d), or None if gcd(c, d, N) > 1
    /// (i.e. the pair does not define a point of P^1(Z/NZ)).
    pub fn normalize(&self, c: i64, d: i64) -> Option<(u64, u64)> {
        self.index_of(c, d).map(|i| self.list[i])
    }

    /// Index of the orbit of (c : d), or None if the pair is not a valid
    /// point (gcd(c, d, N) > 1).
    pub fn index_of(&self, c: i64, d: i64) -> Option<usize> {
        let n = self.n as i64;
        let key = (c.rem_euclid(n) as u64, d.rem_euclid(n) as u64);
        self.index.get(&key).copied()
    }

    /// Right action of S = [[0,-1],[1,0]]: (c : d) -> (d : -c).
    pub fn apply_s(&self, i: usize) -> usize {
        let (c, d) = self.list[i];
        self.index_of(d as i64, -(c as i64))
            .expect("S preserves P^1")
    }

    /// Right action of T = [[0,-1],[1,-1]] (order 3 in PSL2):
    /// (c : d) -> (d : -c-d).
    pub fn apply_t(&self, i: usize) -> usize {
        let (c, d) = self.list[i];
        self.index_of(d as i64, -(c as i64) - (d as i64))
            .expect("T preserves P^1")
    }

    /// Right action of an arbitrary integer matrix m = [[a,b],[c,d]] with
    /// determinant coprime to N: (u : v) -> (u*a + v*c : u*b + v*d).
    pub fn apply_right(&self, i: usize, m: [[i64; 2]; 2]) -> Option<usize> {
        let (u, v) = self.list[i];
        let (u, v) = (u as i64, v as i64);
        self.index_of(u * m[0][0] + v * m[1][0], u * m[0][1] + v * m[1][1])
    }

    /// Lift the i-th point (c : d) to a matrix [[a, b], [c', d']] in SL2(Z)
    /// whose bottom row is congruent to (c, d) mod N.  Returned row-major as
    /// [[a, b], [c', d']] with a*d' - b*c' = 1.
    pub fn lift_to_sl2z(&self, i: usize) -> [[i64; 2]; 2] {
        let (c, d) = self.list[i];
        let n = self.n as i64;
        let (mut c, mut d) = (c as i64, d as i64);
        if self.n == 1 || (c, d) == (0, 1) {
            return [[1, 0], [0, 1]];
        }
        if c == 0 {
            // (0 : d) with d a unit; canonical rep is (0, 1), but be general:
            // replace c by N so gcd search below applies.
            c = n;
        }
        // find k with gcd(c, d + k*N) = 1; exists since gcd(c, d, N) = 1
        let mut k = 0;
        loop {
            let (g, _, _) = xgcd_i64(c, d + k * n);
            if g == 1 {
                d += k * n;
                break;
            }
            let (g2, _, _) = xgcd_i64(c, d - k * n);
            if g2 == 1 {
                d -= k * n;
                break;
            }
            k += 1;
            assert!(
                k <= c.abs() + 1,
                "no coprime lift found for ({c}:{d}) mod {n}: invalid P^1 element"
            );
        }
        // a*d - b*c = 1  <=>  x*d + y*c = 1 with a = x, b = -y
        let (g, x, y) = xgcd_i64(d, c);
        debug_assert_eq!(g, 1);
        let (a, b) = (x, -y);
        debug_assert_eq!(a * d - b * c, 1);
        [[a, b], [c, d]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// psi(N) = N * prod_{p|N} (1 + 1/p), computed independently of P1List.
    fn psi(n: u64) -> u64 {
        let mut result = n;
        let mut m = n;
        let mut p = 2;
        while p * p <= m {
            if m.is_multiple_of(p) {
                result += result / p;
                while m.is_multiple_of(p) {
                    m /= p;
                }
            }
            p += 1;
        }
        if m > 1 {
            result += result / m;
        }
        result
    }

    #[test]
    fn test_p1_sizes_against_formula_and_python_table() {
        // Expected sizes recomputed independently in python (sympy-free,
        // brute-force orbit count agreed with the formula for every N):
        let expected: [(u64, usize); 16] = [
            (1, 1),
            (2, 3),
            (3, 4),
            (4, 6),
            (5, 6),
            (6, 12),
            (7, 8),
            (8, 12),
            (10, 18),
            (11, 12),
            (12, 24),
            (13, 14),
            (15, 24),
            (24, 48),
            (25, 30),
            (49, 56),
        ];
        for (n, size) in expected {
            let p1 = P1List::new(n);
            assert_eq!(p1.len(), size, "P^1(Z/{n}) size vs python table");
            assert_eq!(p1.len(), psi(n) as usize, "P^1(Z/{n}) size vs formula");
        }
    }

    #[test]
    fn test_normalization_is_canonical_under_unit_scaling() {
        for n in [1u64, 6, 11, 12, 24, 49] {
            let p1 = P1List::new(n);
            for c in 0..n.max(1) {
                for d in 0..n.max(1) {
                    let idx = p1.index_of(c as i64, d as i64);
                    if n == 1 {
                        assert_eq!(idx, Some(0));
                        continue;
                    }
                    if gcd_u64(gcd_u64(c, d), n) != 1 {
                        assert_eq!(idx, None, "({c}:{d}) mod {n} should be invalid");
                        continue;
                    }
                    let idx = idx.unwrap();
                    // every unit multiple normalizes to the same index
                    for u in 1..n {
                        if gcd_u64(u, n) != 1 {
                            continue;
                        }
                        assert_eq!(
                            p1.index_of((u * c % n) as i64, (u * d % n) as i64),
                            Some(idx)
                        );
                    }
                    // the stored representative is a fixed point of normalize
                    let rep = p1.list()[idx];
                    assert_eq!(p1.normalize(rep.0 as i64, rep.1 as i64), Some(rep));
                }
            }
        }
    }

    #[test]
    fn test_s_and_t_orders() {
        for n in [1u64, 2, 3, 11, 12, 25, 30] {
            let p1 = P1List::new(n);
            for i in 0..p1.len() {
                // S^2 = -I acts trivially on P^1
                assert_eq!(p1.apply_s(p1.apply_s(i)), i, "S^2 = id on P^1(Z/{n})");
                // T^3 = I
                assert_eq!(
                    p1.apply_t(p1.apply_t(p1.apply_t(i))),
                    i,
                    "T^3 = id on P^1(Z/{n})"
                );
                // apply_right with the explicit matrices agrees
                assert_eq!(p1.apply_right(i, [[0, -1], [1, 0]]), Some(p1.apply_s(i)));
                assert_eq!(p1.apply_right(i, [[0, -1], [1, -1]]), Some(p1.apply_t(i)));
            }
        }
    }

    #[test]
    fn test_lift_to_sl2z() {
        for n in [1u64, 2, 4, 11, 12, 24, 25, 49] {
            let p1 = P1List::new(n);
            for i in 0..p1.len() {
                let m = p1.lift_to_sl2z(i);
                assert_eq!(
                    m[0][0] * m[1][1] - m[0][1] * m[1][0],
                    1,
                    "det 1 for lift of point {i} of P^1(Z/{n})"
                );
                assert_eq!(
                    p1.index_of(m[1][0], m[1][1]),
                    Some(i),
                    "bottom row of lift represents the point"
                );
            }
        }
    }

    #[test]
    fn test_xgcd_i64() {
        for a in -30i64..=30 {
            for b in -30i64..=30 {
                let (g, x, y) = xgcd_i64(a, b);
                assert!(g >= 0);
                assert_eq!(x * a + y * b, g);
                if a != 0 || b != 0 {
                    assert_eq!(g, gcd_u64(a.unsigned_abs(), b.unsigned_abs()) as i64);
                }
            }
        }
    }
}
