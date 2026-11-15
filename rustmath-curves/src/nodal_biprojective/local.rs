//! Exact local classification of a singular point of a plane curve over a finite field of odd
//! characteristic, with its delta invariant and the Frobenius orbits of its branches.
//!
//! MAGMA ch. 114.4.3: `IsNode`, `IsCusp`, `IsOrdinarySingularity`, `TangentCone`,
//! `IsAnalyticallyIrreducible`, `Multiplicity`. Mathematics (WP-F design, Section 4.4):
//!
//! - Hessian rank 2 (`b^2 - ac != 0` for the quadratic part `(a x^2 + 2b xy + c y^2)/2`): a node;
//!   its two branches are defined over the residue field iff `b^2 - ac` is a square.
//! - Hessian rank 1: after a linear change `H_uu(0) = a != 0`; the polar curve `H_u = 0` is a
//!   smooth branch `u = phi(y)` (Newton iteration in `kappa[[y]]`), and `g(y) = H(phi(y), y)` has
//!   order `k + 1`: the point is `A_k`. By Weierstrass preparation `H = W ((u + beta_1/2)^2 -
//!   Delta(y))` (characteristic not 2) and `g = -W(0) Delta (1 + O(Delta))`, so
//!   `ord Delta = k + 1` and `lead Delta = -2 lead(g) / a` (G.-M. Greuel, C. Lossen,
//!   E. Shustin, *Introduction to singularities and deformations*, I.2). `delta = floor((k+1)/2)`
//!   (W. Fulton, *Algebraic curves*, Noether's formula). `k` even: one branch; `k` odd: two
//!   branches, defined over `kappa` iff `-2 a lead(g)` is a square.
//! - Hessian rank 0: if the lowest homogeneous part `T_m` (degree `m >= 3`) is squarefree as a
//!   binary form, the point is an ordinary m-fold point: `m` smooth branches with distinct
//!   tangents, `delta = m(m-1)/2`, and the branches correspond to the roots of `T_m` in `P^1`.
//! - Anything else is `Unclassified`. The power-series precision is capped at `y^128`; a point
//!   whose `g` vanishes to that order is `Unclassified` (its delta would exceed 63 anyway).

use super::residue::{factor_degrees, ResField};

/// The analytic type of a singular point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalType {
    /// An ordinary double point (`A_1`; MAGMA `IsNode`).
    Node,
    /// `A_k`, `k >= 2` (`A_2` is an ordinary cusp, MAGMA `IsCusp`; `A_3` a tacnode).
    A(usize),
    /// An ordinary `m`-fold point, `m >= 3` (MAGMA `IsOrdinarySingularity`).
    OrdinaryMultiple(usize),
    /// Not classified; the reason.
    Unclassified(String),
}

impl LocalType {
    /// The delta invariant over the algebraic closure.
    pub fn delta(&self) -> Option<usize> {
        match self {
            LocalType::Node => Some(1),
            LocalType::A(k) => Some(k.div_ceil(2)),
            LocalType::OrdinaryMultiple(m) => Some(m * (m - 1) / 2),
            LocalType::Unclassified(_) => None,
        }
    }
}

impl std::fmt::Display for LocalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalType::Node => write!(f, "node"),
            LocalType::A(k) => write!(f, "A{k}"),
            LocalType::OrdinaryMultiple(m) => write!(f, "ordinary {m}-fold point"),
            LocalType::Unclassified(s) => write!(f, "unclassified ({s})"),
        }
    }
}

/// The local data of a singular point: type, and the multiset of the sizes of the Frobenius
/// orbits (over the residue field) of its branches.
#[derive(Clone, Debug)]
pub(crate) struct LocalData {
    pub(crate) local: LocalType,
    pub(crate) branch_orbits: Option<Vec<usize>>,
}

fn unclassified(s: &str) -> LocalData {
    LocalData {
        local: LocalType::Unclassified(s.into()),
        branch_orbits: None,
    }
}

/// `binom(i, l) mod p` for `l = 0..=i`.
fn binomials(i: usize, p: i64) -> Vec<i64> {
    let mut row = vec![1i64];
    for _ in 0..i {
        let mut nxt = vec![1i64; row.len() + 1];
        for l in 1..row.len() {
            nxt[l] = (row[l - 1] + row[l]) % p;
        }
        row = nxt;
    }
    row
}

type Bi<E> = Vec<Vec<E>>;

/// `G(x, y) = F(x + x0, y + y0)`; `cols[i]` = the polynomial in `Y` (over `F_p`) of `X^i`.
fn shifted<F: ResField>(f: &F, cols: &[Vec<i64>], x0: &F::E, y0: &F::E) -> Bi<F::E> {
    let p = f.p();
    let nx = cols.len();
    let ny = cols.iter().map(|c| c.len()).max().unwrap_or(1);
    // Taylor shift in y of each column: r(y) <- r(y) (y + y0) + c_j
    let ys: Vec<Vec<F::E>> = cols
        .iter()
        .map(|c| {
            let mut r: Vec<F::E> = vec![f.zero(); ny];
            for &cj in c.iter().rev() {
                let mut nr = vec![f.zero(); ny];
                for k in 0..ny {
                    // coefficient of y^k in r(y) (y + y0)
                    let mut t = f.mul(&r[k], y0);
                    if k >= 1 {
                        t = f.add(&t, &r[k - 1]);
                    }
                    nr[k] = t;
                }
                nr[0] = f.add(&nr[0], &f.constant(cj));
                r = nr;
            }
            r
        })
        .collect();
    let mut out: Bi<F::E> = vec![vec![f.zero(); ny]; nx];
    // x0^e
    let mut x0p = vec![f.one()];
    for e in 1..nx {
        x0p.push(f.mul(&x0p[e - 1], x0));
    }
    for (i, ysi) in ys.iter().enumerate() {
        let bin = binomials(i, p);
        for (l, &bl) in bin.iter().enumerate() {
            if bl == 0 {
                continue;
            }
            let coef = f.scale(&x0p[i - l], bl);
            if f.is_zero(&coef) {
                continue;
            }
            for (j, yij) in ysi.iter().enumerate() {
                if !f.is_zero(yij) {
                    out[l][j] = f.add(&out[l][j], &f.mul(&coef, yij));
                }
            }
        }
    }
    out
}

fn get<F: ResField>(f: &F, g: &Bi<F::E>, i: usize, j: usize) -> F::E {
    g.get(i)
        .and_then(|r| r.get(j))
        .cloned()
        .unwrap_or_else(|| f.zero())
}

// Truncated power series over F, length n.
fn smul<F: ResField>(f: &F, a: &[F::E], b: &[F::E], n: usize) -> Vec<F::E> {
    let mut r = vec![f.zero(); n];
    for (i, ai) in a.iter().enumerate().take(n) {
        if f.is_zero(ai) {
            continue;
        }
        for (j, bj) in b.iter().enumerate().take(n - i) {
            r[i + j] = f.add(&r[i + j], &f.mul(ai, bj));
        }
    }
    r
}

fn sinv<F: ResField>(f: &F, a: &[F::E], n: usize) -> Vec<F::E> {
    let a0inv = f.inv(&a[0]).expect("unit series");
    let mut r = vec![f.zero(); n];
    r[0] = a0inv.clone();
    for k in 1..n {
        let mut s = f.zero();
        for i in 1..=k.min(a.len() - 1) {
            s = f.add(&s, &f.mul(&a[i], &r[k - i]));
        }
        r[k] = f.neg(&f.mul(&s, &a0inv));
    }
    r
}

/// `H(phi(y), y) mod y^n`.
fn beval<F: ResField>(f: &F, h: &Bi<F::E>, phi: &[F::E], n: usize) -> Vec<F::E> {
    let mut r = vec![f.zero(); n];
    for row in h.iter().rev() {
        r = smul(f, &r, phi, n);
        for (j, c) in row.iter().enumerate().take(n) {
            r[j] = f.add(&r[j], c);
        }
    }
    r
}

fn du<F: ResField>(f: &F, h: &Bi<F::E>) -> Bi<F::E> {
    h.iter()
        .enumerate()
        .skip(1)
        .map(|(i, row)| row.iter().map(|c| f.scale(c, i as i64)).collect())
        .collect()
}

/// Classify the point `(x0, y0)` (coordinates in the field `f`) of `V(F)`, `cols[i]` the
/// polynomial in `Y` of `X^i` with coefficients in `[0, p)`.
pub(crate) fn classify<F: ResField>(f: &F, cols: &[Vec<i64>], x0: &F::E, y0: &F::E) -> LocalData {
    let g = shifted(f, cols, x0, y0);
    let z = |i, j| get(f, &g, i, j);
    if !f.is_zero(&z(0, 0)) || !f.is_zero(&z(1, 0)) || !f.is_zero(&z(0, 1)) {
        return unclassified("not a singular point");
    }
    let a = f.scale(&z(2, 0), 2);
    let b = z(1, 1);
    let c = f.scale(&z(0, 2), 2);
    let hd = f.sub(&f.mul(&b, &b), &f.mul(&a, &c));
    if !f.is_zero(&hd) {
        let orbits = if f.is_square(&hd) {
            vec![1, 1]
        } else {
            vec![2]
        };
        return LocalData {
            local: LocalType::Node,
            branch_orbits: Some(orbits),
        };
    }
    let nx = g.len();
    let ny = g.first().map(|r| r.len()).unwrap_or(0);
    if f.is_zero(&a) && f.is_zero(&b) && f.is_zero(&c) {
        // Hessian rank 0: the lowest homogeneous part.
        for m in 3..(nx + ny) {
            let t: Vec<F::E> = (0..=m)
                .map(|i| if i <= m { z(i, m - i) } else { f.zero() })
                .collect();
            if t.iter().all(|x| f.is_zero(x)) {
                continue;
            }
            let mut d = m as isize;
            while d >= 0 && f.is_zero(&t[d as usize]) {
                d -= 1;
            }
            let drop = m - d as usize;
            if drop >= 2 {
                return unclassified(&format!(
                    "non-ordinary {m}-fold point (repeated tangent y = 0)"
                ));
            }
            return match factor_degrees(f, &t[..=(d as usize)]) {
                None => unclassified(&format!("non-ordinary {m}-fold point")),
                Some(mut degs) => {
                    if drop == 1 {
                        degs.push(1);
                    }
                    degs.sort();
                    LocalData {
                        local: LocalType::OrdinaryMultiple(m),
                        branch_orbits: Some(degs),
                    }
                }
            };
        }
        return unclassified("the local equation vanishes to all orders");
    }
    // Hessian rank 1: make the u^2 coefficient nonzero, then kill the cross term.
    let (gg, lam): (Bi<F::E>, F::E) = if !f.is_zero(&a) {
        (g.clone(), f.mul(&b, &f.inv(&a).expect("a != 0")))
    } else {
        // a = 0 forces b = 0 and c != 0: exchange x and y.
        let mut s: Bi<F::E> = vec![vec![f.zero(); nx]; ny];
        for (i, row) in g.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                s[j][i] = v.clone();
            }
        }
        (s, f.zero())
    };
    let p = f.p();
    let nu = gg.len();
    let nyy = gg.first().map(|r| r.len()).unwrap_or(0) + nu;
    // H(u, y) = gg(u - lam y, y)
    let mlam = f.neg(&lam);
    let mut mlp = vec![f.one()];
    for e in 1..nu {
        mlp.push(f.mul(&mlp[e - 1], &mlam));
    }
    let mut h: Bi<F::E> = vec![vec![f.zero(); nyy]; nu];
    for (i, row) in gg.iter().enumerate() {
        let bin = binomials(i, p);
        for (l, &bl) in bin.iter().enumerate() {
            let e = i - l;
            let coef = f.scale(&mlp[e], bl);
            if f.is_zero(&coef) {
                continue;
            }
            for (j, v) in row.iter().enumerate() {
                if !f.is_zero(v) {
                    h[l][j + e] = f.add(&h[l][j + e], &f.mul(&coef, v));
                }
            }
        }
    }
    let aa = f.scale(&get(f, &h, 2, 0), 2);
    if f.is_zero(&aa) {
        return unclassified("rank-one Hessian without a square term (internal)");
    }
    let hu = du(f, &h);
    let huu = du(f, &hu);
    let mut n = 16usize;
    while n <= 128 {
        let mut phi = vec![f.zero(); n];
        // Newton converges quadratically from phi = 0 (phi(0) = 0 since H_u(0, 0) = 0).
        let mut iters = 0;
        loop {
            let num = beval(f, &hu, &phi, n);
            if num.iter().all(|x| f.is_zero(x)) {
                break;
            }
            let den = beval(f, &huu, &phi, n);
            let corr = smul(f, &num, &sinv(f, &den, n), n);
            for j in 0..n {
                phi[j] = f.sub(&phi[j], &corr[j]);
            }
            iters += 1;
            if iters > 2 * 8 + 4 {
                return unclassified("Newton iteration did not converge (internal)");
            }
        }
        let gy = beval(f, &h, &phi, n);
        if let Some(k1) = gy.iter().position(|x| !f.is_zero(x)) {
            if k1 < 3 {
                return unclassified("inconsistent order of the polar restriction (internal)");
            }
            let k = k1 - 1;
            let orbits = if k % 2 == 0 {
                vec![1]
            } else {
                let w = f.mul(&f.scale(&aa, -2), &gy[k1]);
                if f.is_square(&w) {
                    vec![1, 1]
                } else {
                    vec![2]
                }
            };
            return LocalData {
                local: LocalType::A(k),
                branch_orbits: Some(orbits),
            };
        }
        n *= 2;
    }
    unclassified("A_k with k >= 127, or a non-reduced point")
}
