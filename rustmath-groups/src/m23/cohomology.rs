//! Parabolic cohomology over `F_2` of the Golay modules of M23 for a triple (WP-P, HANDOFF
//! 4.3).
//!
//! MAGMA: ch. 68 (cohomology of finite groups: `CohomologyModule`, `CohomologyGroup`,
//! cocycles and coboundaries); digest `rustmath-lava/docs/Chapter_68_Cohomology_and_Extensions.md`.
//! RustMath has no group cohomology (`rustmath-homology` gives free ranks over Z only); this
//! module computes the one case needed here.
//!
//! Sources: the owner's paper (`johnjanik/M23` at `2d11038`, `paper/golay_m23.tex`), Section 4
//! (Theorem 4.1: `W = H^1(P^1, j_* G0)` is a Lagrangian of `Jac(X)[2]`), and its
//! `m23golay/cohomology.py` and `f2.py`, ported as algorithms.
//!
//! The modules (right action `(v.g)[x^g] = v[x]` on 23-bit masks): `G0`, the even Golay code
//! (dimension 11); `Q = P0 / G0` (dimension 11); `P0`, the even-weight vectors (dimension 22). A
//! module is a subquotient with an ordered basis (quotient basis first); coordinates come from
//! the reduction in an echelon span with combination masks.
//!
//! For a triple `(g1, g2, g3)`, `g1 g2 g3 = 1`: cocycles are `(c1, c2)` in `M^2` with
//! `c3 = c1 (g2 g3) + c2 g3`; coboundaries `(a (1 + g1), a (1 + g2))`; parabolic iff `c_i` lies
//! in `M (1 + g_i)` for `i = 1, 2, 3`; orbifold iff `c_i N_i = 0` with
//! `N_i = 1 + g_i + ... + g_i^(e_i - 1)`. `H^1_par = Z_par / B`. Complex conjugation with the
//! real structure `c0` acts on parabolic classes by `z'(gamma_1) = c1 g1^-1 c0`,
//! `z'(gamma_2) = c3 g3^-1 c0`; it must be an involution.
//!
//! Bit masks (the specialised fast path; `Matrix<PrimeField>` takes 12 ms per 44 x 66 kernel,
//! too slow for the 6,127 triples, and is used as a cross-check in the tests). A cocycle packs
//! `(c1, c2, c3)` into `3 dim` bits and the combination masks of [`F2Span`] use one bit per
//! added vector, so every constructor refuses `3 dim > 128` with
//! `Err(F2Error::DimensionTooLarge)`, and [`F2Span::add`] refuses a 129th vector with
//! `Err(F2Error::CapacityExceeded)`. No shift by 128 or more is executed.

use std::fmt;

use super::group::{identity23, inverse, mask_image, order, rmul, Perm23, M23};
use super::nielsen::Triple;

/// Errors of the `F_2` layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum F2Error {
    /// A module of this dimension does not fit the 128-bit packing (`3 dim > 128`).
    DimensionTooLarge {
        /// The dimension of the module.
        dim: usize,
    },
    /// A span already holds 128 added vectors.
    CapacityExceeded,
    /// A vector is not in the module, or a permutation does not preserve it.
    NotInModule,
    /// `c0` is not a real structure of the triple: `c0^2 != 1`, `c0 g1 c0 != g1^-1` or
    /// `c0 g2 c0 != g3^-1`.
    NotARealStructure,
    /// A coboundary is not a parabolic (or orbifold) cocycle (impossible for a triple).
    CoboundaryNotCocycle,
    /// The image of a parabolic cocycle under conjugation is not parabolic (impossible for a
    /// real structure).
    NotParabolic,
    /// The action of complex conjugation is not an involution.
    NotAnInvolution,
    /// The triple does not have product 1.
    ProductNotOne,
}

impl fmt::Display for F2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            F2Error::DimensionTooLarge { dim } => {
                write!(f, "module of dimension {dim}: 3 dim > 128")
            }
            F2Error::CapacityExceeded => write!(f, "a span holds at most 128 added vectors"),
            F2Error::NotInModule => write!(f, "the vector is not in the module"),
            F2Error::NotARealStructure => write!(f, "c0 is not a real structure of the triple"),
            F2Error::CoboundaryNotCocycle => write!(f, "a coboundary is not a cocycle"),
            F2Error::NotParabolic => {
                write!(f, "the conjugate of a parabolic cocycle is not parabolic")
            }
            F2Error::NotAnInvolution => write!(f, "conjugation is not an involution"),
            F2Error::ProductNotOne => write!(f, "the triple does not have product 1"),
        }
    }
}

impl std::error::Error for F2Error {}

/// A subspace of `F_2^128` in echelon form (pivot = highest bit), with the combination of the
/// added vectors that gives each pivot row.
#[derive(Clone, Debug, Default)]
pub struct F2Span {
    /// (pivot bit, vector, combination), pivots descending
    piv: Vec<(u32, u128, u128)>,
    n_added: usize,
}

impl F2Span {
    /// The zero space.
    pub fn new() -> F2Span {
        F2Span::default()
    }

    fn reduce_full(&self, mut v: u128, mut combo: u128) -> (u128, u128) {
        for &(hb, pv, pc) in &self.piv {
            if (v >> hb) & 1 == 1 {
                v ^= pv;
                combo ^= pc;
            }
        }
        (v, combo)
    }

    fn insert(&mut self, r: u128, combo: u128) {
        let hb = 127 - r.leading_zeros();
        let pos = self
            .piv
            .iter()
            .position(|&(b, _, _)| b < hb)
            .unwrap_or(self.piv.len());
        self.piv.insert(pos, (hb, r, combo));
    }

    /// Adds `v`; `Ok(true)` if it was independent. The `k`-th added vector (from 0) has
    /// combination bit `k`. `Err(CapacityExceeded)` once 128 vectors have been added.
    pub fn add(&mut self, v: u128) -> Result<bool, F2Error> {
        if self.n_added >= 128 {
            return Err(F2Error::CapacityExceeded);
        }
        let idx = self.n_added;
        self.n_added += 1;
        let (r, c) = self.reduce_full(v, 1u128 << idx);
        if r == 0 {
            return Ok(false);
        }
        self.insert(r, c);
        Ok(true)
    }

    /// The dimension.
    pub fn dim(&self) -> usize {
        self.piv.len()
    }

    /// The number of vectors added so far (independent or not).
    pub fn num_added(&self) -> usize {
        self.n_added
    }

    /// Whether `v` lies in the span.
    pub fn contains(&self, v: u128) -> bool {
        self.reduce_full(v, 0).0 == 0
    }

    /// The reduction of `v` modulo the span (0 iff `v` lies in it).
    pub fn reduce(&self, v: u128) -> u128 {
        self.reduce_full(v, 0).0
    }

    /// The combination of the added vectors equal to `v`, or `None` if `v` is not in the span.
    pub fn solve(&self, v: u128) -> Option<u128> {
        let (r, c) = self.reduce_full(v, 0);
        if r == 0 {
            Some(c)
        } else {
            None
        }
    }

    /// An echelon basis (the pivot rows).
    pub fn basis(&self) -> Vec<u128> {
        self.piv.iter().map(|&(_, v, _)| v).collect()
    }
}

/// The kernel of the map with the given images of the unit vectors: a basis of the
/// combinations (bit `i` for unit vector `i`) that map to 0. At most 128 images.
fn kernel(images: &[u128]) -> Result<Vec<u128>, F2Error> {
    if images.len() > 128 {
        return Err(F2Error::CapacityExceeded);
    }
    let mut s = F2Span::new();
    let mut out = Vec::new();
    for (i, &v) in images.iter().enumerate() {
        let (r, c) = s.reduce_full(v, 1u128 << i);
        if r == 0 {
            out.push(c);
        } else {
            s.insert(r, c);
        }
    }
    Ok(out)
}

/// `x M` for a row vector `x` and a matrix `M` given by its rows.
fn apply(x: u128, m: &[u128]) -> u128 {
    let mut out = 0;
    let mut v = x;
    while v != 0 {
        let i = v.trailing_zeros() as usize;
        out ^= m[i];
        v &= v - 1;
    }
    out
}

fn mat_mul(a: &[u128], b: &[u128]) -> Vec<u128> {
    a.iter().map(|&r| apply(r, b)).collect()
}

fn mat_add_id(a: &[u128]) -> Vec<u128> {
    a.iter()
        .enumerate()
        .map(|(i, &r)| r ^ (1u128 << i))
        .collect()
}

/// The dimension of the fixed space `ker(M + 1)`.
fn fixed_dim(m: &[u128]) -> Result<usize, F2Error> {
    Ok(kernel(&mat_add_id(m))?.len())
}

fn check_dim(dim: usize) -> Result<(), F2Error> {
    if 3 * dim > 128 {
        Err(F2Error::DimensionTooLarge { dim })
    } else {
        Ok(())
    }
}

/// A subquotient `sub / quo` of `F_2^23` with the right permutation action, and an ordered
/// basis (the quotient basis first, then the vectors that extend it to `sub`).
#[derive(Clone, Debug)]
pub struct F2Module {
    quo_dim: usize,
    ordered: Vec<u128>,
    /// `span(sub)`, in echelon form over the ordered basis
    span: F2Span,
    /// `span(quo)`, to check that a permutation preserves it
    quo_span: F2Span,
    dim: usize,
}

impl F2Module {
    /// `span(sub) / span(quo)`. `Err(NotInModule)` if `quo` does not lie in `span(sub)`;
    /// `Err(DimensionTooLarge)` when `3 dim > 128`.
    pub fn subquotient(sub: &[u32], quo: &[u32]) -> Result<F2Module, F2Error> {
        let mut sub_span = F2Span::new();
        for &v in sub {
            sub_span.add(v as u128)?;
        }
        let mut qs = F2Span::new();
        for &v in quo {
            if !sub_span.contains(v as u128) {
                return Err(F2Error::NotInModule);
            }
            qs.add(v as u128)?;
        }
        let quo_dim = qs.dim();
        let mut ordered = qs.basis();
        let mut full = F2Span::new();
        for &v in &ordered {
            full.add(v)?;
        }
        for &v in sub {
            if full.add(v as u128)? {
                ordered.push(v as u128);
            }
        }
        let dim = ordered.len() - quo_dim;
        check_dim(dim)?;
        let mut span = F2Span::new();
        for &v in &ordered {
            span.add(v)?;
        }
        Ok(F2Module {
            quo_dim,
            ordered,
            span,
            quo_span: qs,
            dim,
        })
    }

    /// `span(sub)` as a module.
    pub fn new(sub: &[u32]) -> Result<F2Module, F2Error> {
        F2Module::subquotient(sub, &[])
    }

    /// The dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The coordinates of `v` in the module basis (bit `i` for basis vector `i`).
    pub fn coords(&self, v: u32) -> Result<u128, F2Error> {
        let c = self.span.solve(v as u128).ok_or(F2Error::NotInModule)?;
        Ok(c >> self.quo_dim)
    }

    /// The matrix of `g` (row `i` = coordinates of the image of basis vector `i`).
    /// `Err(NotInModule)` if `g` does not preserve the module: the image of a vector of `sub`
    /// leaves `span(sub)`, or the image of a vector of `quo` leaves `span(quo)`.
    pub fn matrix(&self, g: &Perm23) -> Result<Vec<u128>, F2Error> {
        for &q in &self.ordered[..self.quo_dim] {
            if !self.quo_span.contains(mask_image(q as u32, g) as u128) {
                return Err(F2Error::NotInModule);
            }
        }
        (0..self.dim)
            .map(|i| self.coords(mask_image(self.ordered[self.quo_dim + i] as u32, g)))
            .collect()
    }
}

/// The three modules of HANDOFF 4.3: `G0` (the even Golay code, dimension 11),
/// `Q = P0 / G0` (dimension 11), `P0` (even-weight vectors, dimension 22).
pub fn standard_modules(g: &M23) -> Result<[F2Module; 3], F2Error> {
    let g0: Vec<u32> = g.even_golay_basis().iter().map(|&(v, _)| v).collect();
    let p0: Vec<u32> = (1..23).map(|i| 1 | (1 << i)).collect();
    Ok([
        F2Module::new(&g0)?,
        F2Module::subquotient(&p0, &g0)?,
        F2Module::new(&p0)?,
    ])
}

/// The parabolic and orbifold cohomology of a module for one triple.
pub struct TripleCohomology<'a> {
    m: &'a F2Module,
    t: Triple,
    d: usize,
    mats: [Vec<u128>; 3],
    g2g3: Vec<u128>,
    b: F2Span,
    zpar: F2Span,
    zorb: F2Span,
}

impl<'a> TripleCohomology<'a> {
    /// Cocycles, coboundaries, parabolic and orbifold cocycles of `m` for `t`.
    /// `Err(ProductNotOne)`, `Err(DimensionTooLarge)` when `3 dim > 128`, `Err(NotInModule)`
    /// if an entry does not preserve the module, `Err(CoboundaryNotCocycle)` if `B` is not in
    /// `Z_par` or `Z_orb`.
    pub fn new(m: &'a F2Module, t: &Triple) -> Result<TripleCohomology<'a>, F2Error> {
        if rmul(&rmul(&t.0, &t.1), &t.2) != identity23() {
            return Err(F2Error::ProductNotOne);
        }
        let d = m.dim();
        check_dim(d)?;
        let mats = [m.matrix(&t.0)?, m.matrix(&t.1)?, m.matrix(&t.2)?];
        let one_plus: Vec<Vec<u128>> = mats.iter().map(|x| mat_add_id(x)).collect();
        let ords = [order(&t.0), order(&t.1), order(&t.2)];
        let id: Vec<u128> = (0..d).map(|k| 1u128 << k).collect();
        let norms: Vec<Vec<u128>> = (0..3)
            .map(|i| {
                let mut acc = id.clone();
                let mut pw = id.clone();
                for _ in 1..ords[i] {
                    pw = mat_mul(&pw, &mats[i]);
                    acc = acc.iter().zip(&pw).map(|(x, y)| x ^ y).collect();
                }
                acc
            })
            .collect();
        let g2g3 = mat_mul(&mats[1], &mats[2]);
        let mut b = F2Span::new();
        for (r1, r2) in one_plus[0].iter().zip(&one_plus[1]) {
            b.add(r1 | (r2 << d))?;
        }
        let mut images: Vec<F2Span> = Vec::new();
        for rows in &one_plus {
            let mut s = F2Span::new();
            for &r in rows {
                s.add(r)?;
            }
            images.push(s);
        }
        let mask = if d == 0 { 0 } else { (1u128 << d) - 1 };
        let unpack = |z: u128| -> (u128, u128, u128) {
            let c1 = z & mask;
            let c2 = z >> d;
            (c1, c2, apply(c1, &g2g3) ^ apply(c2, &mats[2]))
        };
        let solve = |f: &dyn Fn((u128, u128, u128)) -> [u128; 3]| -> Result<F2Span, F2Error> {
            let imgs: Vec<u128> = (0..2 * d)
                .map(|i| {
                    let v = f(unpack(1u128 << i));
                    v[0] | (v[1] << d) | (v[2] << (2 * d))
                })
                .collect();
            let mut s = F2Span::new();
            for k in kernel(&imgs)? {
                s.add(k)?;
            }
            Ok(s)
        };
        let zpar = solve(&|c| {
            [
                images[0].reduce(c.0),
                images[1].reduce(c.1),
                images[2].reduce(c.2),
            ]
        })?;
        let zorb = solve(&|c| {
            [
                apply(c.0, &norms[0]),
                apply(c.1, &norms[1]),
                apply(c.2, &norms[2]),
            ]
        })?;
        if !b
            .basis()
            .iter()
            .all(|&v| zpar.contains(v) && zorb.contains(v))
        {
            return Err(F2Error::CoboundaryNotCocycle);
        }
        Ok(TripleCohomology {
            m,
            t: *t,
            d,
            mats,
            g2g3,
            b,
            zpar,
            zorb,
        })
    }

    /// `dim H^1_par = dim Z_par - dim B`.
    pub fn dim_parabolic(&self) -> usize {
        self.zpar.dim() - self.b.dim()
    }

    /// `dim H^1_orb = dim Z_orb - dim B`.
    pub fn dim_orbifold(&self) -> usize {
        self.zorb.dim() - self.b.dim()
    }

    /// The dimension of the fixed space of complex conjugation (real structure `c0`) on
    /// `H^1_par`. `Err(NotARealStructure)` unless `c0^2 = 1`, `c0 g1 c0 = g1^-1` and
    /// `c0 g2 c0 = g3^-1`; `Err(NotParabolic)` if the image of a parabolic cocycle is not
    /// parabolic, `Err(NotAnInvolution)` if the action is not an involution.
    pub fn conjugation_fixed_dim(&self, c0: &Perm23) -> Result<usize, F2Error> {
        let (g1, g2, g3) = &self.t;
        if rmul(c0, c0) != identity23()
            || rmul(&rmul(c0, g1), c0) != inverse(g1)
            || rmul(&rmul(c0, g2), c0) != inverse(g3)
        {
            return Err(F2Error::NotARealStructure);
        }
        let d = self.d;
        let mask = if d == 0 { 0 } else { (1u128 << d) - 1 };
        let mc0 = self.m.matrix(c0)?;
        let m1 = mat_mul(&self.m.matrix(&inverse(&self.t.0))?, &mc0);
        let m3 = mat_mul(&self.m.matrix(&inverse(&self.t.2))?, &mc0);
        // a complement of B in Z_par
        let mut probe = F2Span::new();
        for v in self.b.basis() {
            probe.add(v)?;
        }
        let mut basis = Vec::new();
        for v in self.zpar.basis() {
            if probe.add(v)? {
                basis.push(v);
            }
        }
        // B basis, then the complement: combination bits nb.. index the complement
        let mut s = F2Span::new();
        for v in self.b.basis() {
            s.add(v)?;
        }
        let nb = s.num_added();
        for &v in &basis {
            s.add(v)?;
        }
        let mut rows = Vec::with_capacity(basis.len());
        for &v in &basis {
            let c1 = v & mask;
            let c2 = v >> d;
            let c3 = apply(c1, &self.g2g3) ^ apply(c2, &self.mats[2]);
            let img = apply(c1, &m1) | (apply(c3, &m3) << d);
            if !self.zpar.contains(img) {
                return Err(F2Error::NotParabolic);
            }
            let combo = s.solve(img).ok_or(F2Error::NotParabolic)?;
            rows.push(combo >> nb);
        }
        let id: Vec<u128> = (0..rows.len()).map(|i| 1u128 << i).collect();
        if mat_mul(&rows, &rows) != id {
            return Err(F2Error::NotAnInvolution);
        }
        fixed_dim(&rows)
    }
}
