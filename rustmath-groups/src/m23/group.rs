//! M23 on the published generators of HJLPPZ, its elements, its Steiner system and its even
//! Golay code.
//!
//! MAGMA: ch. 58 (permutation groups: `PermutationGroup`, `Order` by Schreier-Sims, `in` by
//! sifting, iteration over the elements); ch. 152 (the binary Golay code and its even subcode).
//! Digests: `rustmath-lava/docs/Chapter_58_Permutation_Groups.md`,
//! `Chapter_152_Linear_Codes_over_Finite_Fields.md`.
//!
//! Mathematics and sources:
//! - Huang, Jackson, Lee, Poonen, Pries, Zhang (HJLPPZ), arXiv:2608.08538v2, Section 3.1: the
//!   generators `g1`, `g2`, `g3` (right action, `g1 g2 g3 = 1`).
//! - C. C. Sims, Computational methods in the study of permutation groups (1970); Holt, Eick,
//!   O'Brien, *Handbook of Computational Group Theory* (2005), 4.4: the stabilizer chain and the
//!   factorization of each element through the transversals ([`crate::bsgs`]).
//! - E. Witt, Ueber Steinersche Systeme, Abh. Math. Sem. Hamburg 12 (1938): the S(4,7,23).
//! - MacWilliams, Sloane, *The Theory of Error-Correcting Codes* (1977), ch. 20: the Golay code,
//!   spanned by its heptads; the even subcode `G0` is spanned by the sums of two heptads.
//! - The owner's M23 repository (`johnjanik/M23` at `2d11038`): `m23golay/m23.py` (`G1`, `G2`,
//!   `G3`, `steiner`), `golay.py` (`even_subcode`), ported as algorithms.
//!
//! Conventions (HANDOFF Section 1): permutations act on the right, `x^(gh) = (x^g)^h`, so a
//! product is read left to right. A [`Perm23`] is a 0-based image list: `p[x]` is `x^p`. The
//! published cycles are 1-based.

use std::fmt;

use rustmath_combinatorics::designs::is_steiner_system;

use crate::bsgs::StabilizerChain;
use crate::transitive23::{identity, perm_from_cycles};

/// A permutation of `{0, ..., 22}` as an image list (`p[x] = x^p`); the existing
/// `transitive23::Perm` type.
pub type Perm23 = crate::transitive23::Perm;

/// The order of M23 (ATLAS, p. 71).
pub const M23_ORDER: u64 = 10_200_960;

/// All 23 points as a bit mask.
pub(crate) const FULL_MASK: u32 = (1 << 23) - 1;

/// `g1` of HJLPPZ 3.1, 1-based cycles.
const G1_CYCLES: [[u8; 2]; 8] = [
    [1, 11],
    [2, 23],
    [3, 8],
    [4, 16],
    [5, 21],
    [7, 20],
    [15, 19],
    [18, 22],
];
/// `g2` of HJLPPZ 3.1, a 23-cycle (1-based).
const G2_CYCLE: [u8; 23] = [
    1, 2, 3, 4, 10, 11, 12, 7, 19, 18, 8, 6, 9, 16, 17, 21, 22, 5, 14, 20, 13, 15, 23,
];
/// `g3` of HJLPPZ 3.1, a 23-cycle (1-based); `g1 g2 g3 = 1` is checked at construction.
const G3_CYCLE: [u8; 23] = [
    1, 2, 11, 10, 16, 9, 6, 3, 23, 19, 20, 14, 21, 17, 4, 8, 22, 5, 18, 15, 13, 7, 12,
];

/// Errors of the M23 layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum M23Error {
    /// The input is not a permutation of 23 points (with the reason).
    NotAPermutation(String),
    /// The permutation is not an element of M23.
    NotInM23,
    /// A verification at construction failed. This is a defect, never expected.
    Construction(String),
    /// A search would exceed its budget. `needed` is the size the search needs (for an orbit,
    /// a lower bound), `budget` the caller's limit.
    BudgetExceeded {
        /// The size needed (a lower bound for an orbit search).
        needed: u64,
        /// The budget.
        budget: u64,
    },
    /// A split-class invariant took a value that belongs to neither class of its pair. Only
    /// `heptad_family_11` and `cyclic_steiner_sign` return it; inside M23 it cannot happen.
    InvariantFailed(String),
    /// A file could not be read.
    Io(String),
    /// A file could not be parsed.
    Parse(String),
}

impl fmt::Display for M23Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            M23Error::NotAPermutation(s) => write!(f, "not a permutation of 23 points: {s}"),
            M23Error::NotInM23 => write!(f, "not an element of M23"),
            M23Error::Construction(s) => write!(f, "M23 construction check failed: {s}"),
            M23Error::BudgetExceeded { needed, budget } => {
                write!(f, "search needs {needed} candidates, budget {budget}")
            }
            M23Error::InvariantFailed(s) => write!(f, "split-class invariant failed: {s}"),
            M23Error::Io(s) => write!(f, "I/O error: {s}"),
            M23Error::Parse(s) => write!(f, "parse error: {s}"),
        }
    }
}

impl std::error::Error for M23Error {}

/// `p q`: first `p`, then `q` (right action): `x^(pq) = (x^p)^q`, so `rmul(p, q)[x] = q[p[x]]`.
#[inline]
pub fn rmul(p: &Perm23, q: &Perm23) -> Perm23 {
    let mut r = [0u8; 23];
    for (out, &px) in r.iter_mut().zip(p) {
        *out = q[px as usize];
    }
    r
}

/// The inverse permutation.
#[inline]
pub fn inverse(p: &Perm23) -> Perm23 {
    let mut r = [0u8; 23];
    for (i, &pi) in p.iter().enumerate() {
        r[pi as usize] = i as u8;
    }
    r
}

/// `x^h = h^-1 x h` (right action): first `h^-1`, then `x`, then `h`.
#[inline]
pub fn conjugate(x: &Perm23, h: &Perm23) -> Perm23 {
    rmul(&rmul(&inverse(h), x), h)
}

/// The identity of `Sym(23)`.
pub fn identity23() -> Perm23 {
    identity()
}

/// The cycles of `p` (each starting at its least point, the cycles ordered by that point).
pub fn cycles(p: &Perm23) -> Vec<Vec<u8>> {
    let mut seen = [false; 23];
    let mut out = Vec::new();
    for i in 0..23 {
        if !seen[i] {
            let mut c = Vec::new();
            let mut j = i;
            while !seen[j] {
                seen[j] = true;
                c.push(j as u8);
                j = p[j] as usize;
            }
            out.push(c);
        }
    }
    out
}

/// The number of cycles of `p`, fixed points included.
pub fn num_cycles(p: &Perm23) -> usize {
    cycles(p).len()
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// The order of `p` (the lcm of its cycle lengths).
pub fn order(p: &Perm23) -> u32 {
    cycles(p)
        .iter()
        .fold(1u32, |l, c| l / gcd(l, c.len() as u32) * c.len() as u32)
}

/// `p^k` for any integer `k` (negative powers are powers of the inverse).
pub fn power(p: &Perm23, k: i64) -> Perm23 {
    let o = order(p) as i64;
    let k = k.rem_euclid(o);
    let mut r = identity();
    for _ in 0..k {
        r = rmul(&r, p);
    }
    r
}

/// The image of a point set (bit mask) under `g`.
#[inline]
pub fn mask_image(m: u32, g: &Perm23) -> u32 {
    let mut out = 0;
    let mut v = m;
    while v != 0 {
        let i = v.trailing_zeros() as usize;
        out |= 1 << g[i];
        v &= v - 1;
    }
    out
}

/// The fixed points of `p` as a bit mask.
#[inline]
pub fn fixed_mask(p: &Perm23) -> u32 {
    let mut m = 0;
    for (i, &pi) in p.iter().enumerate() {
        if pi as usize == i {
            m |= 1 << i;
        }
    }
    m
}

/// Whether `<a, b>` is transitive on the 23 points: the closure of `{0}` under `a` and `b`,
/// on bit masks (at most 23 rounds).
#[inline]
pub fn generates_transitive(a: &Perm23, b: &Perm23) -> bool {
    let mut reach: u32 = 1;
    loop {
        let mut new = reach;
        let mut v = reach;
        while v != 0 {
            let i = v.trailing_zeros() as usize;
            new |= (1 << a[i]) | (1 << b[i]);
            v &= v - 1;
        }
        if new == reach {
            return reach == FULL_MASK;
        }
        reach = new;
    }
}

/// A permutation of the 23 points from disjoint cycles in 1-based points. `Err` on a point
/// outside `1..=23` or a repeated point. Cycles of length 0 or 1 are allowed.
pub fn from_cycles_1based<C: AsRef<[u8]>>(cycles: &[C]) -> Result<Perm23, M23Error> {
    let mut used = 0u32;
    for c in cycles {
        for &x in c.as_ref() {
            if !(1..=23).contains(&x) {
                return Err(M23Error::NotAPermutation(format!(
                    "point {x} outside 1..23"
                )));
            }
            if (used >> (x - 1)) & 1 == 1 {
                return Err(M23Error::NotAPermutation(format!("point {x} repeated")));
            }
            used |= 1 << (x - 1);
        }
    }
    let owned: Vec<Vec<u8>> = cycles.iter().map(|c| c.as_ref().to_vec()).collect();
    Ok(perm_from_cycles(&owned))
}

/// A permutation from a 0-based image list of length 23. `Err` if it is not a bijection.
pub fn from_image_list(v: &[usize]) -> Result<Perm23, M23Error> {
    if v.len() != 23 {
        return Err(M23Error::NotAPermutation(format!(
            "length {} instead of 23",
            v.len()
        )));
    }
    let mut p = [0u8; 23];
    let mut used = 0u32;
    for (i, &x) in v.iter().enumerate() {
        if x >= 23 || (used >> x) & 1 == 1 {
            return Err(M23Error::NotAPermutation(format!("image {x} of {i}")));
        }
        used |= 1 << x;
        p[i] = x as u8;
    }
    Ok(p)
}

/// The image list as `usize` entries.
pub fn to_image_list(p: &Perm23) -> [usize; 23] {
    std::array::from_fn(|i| p[i] as usize)
}

/// Reduced echelon form over `F_2` of 23-bit vectors: the pivot of a row is its highest bit,
/// and every row is 0 at the pivots of the others. Returns `(row, pivot)` in insertion order.
pub(crate) fn echelon_f2(vs: impl IntoIterator<Item = u32>) -> Vec<(u32, u8)> {
    let mut rows: Vec<u32> = Vec::new();
    for mut v in vs {
        for &r in &rows {
            let p = 31 - r.leading_zeros();
            if (v >> p) & 1 == 1 {
                v ^= r;
            }
        }
        if v != 0 {
            let p = 31 - v.leading_zeros();
            for r in rows.iter_mut() {
                if (*r >> p) & 1 == 1 {
                    *r ^= v;
                }
            }
            rows.push(v);
        }
    }
    rows.into_iter()
        .map(|r| (r, (31 - r.leading_zeros()) as u8))
        .collect()
}

/// M23 as `<g1, g2>` on 23 points, with its stabilizer chain, its Steiner system S(4,7,23)
/// (253 heptads) and the reduced echelon basis of its even Golay code `G0` (dimension 11).
pub struct M23 {
    g1: Perm23,
    g2: Perm23,
    g3: Perm23,
    chain: StabilizerChain,
    heptads: Vec<u32>,
    golay: Vec<(u32, u8)>,
}

impl fmt::Debug for M23 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("M23")
            .field("g1", &self.g1)
            .field("g2", &self.g2)
            .field("base", &self.chain.base())
            .finish()
    }
}

impl M23 {
    /// `<g1, g2>` of HJLPPZ 3.1. Verified at construction: `g1 g2 g3 = 1` with the published
    /// `g3`; the Schreier-Sims order is 10,200,960; the orbit of `Fix(g1)` under `<g1, g2>` has
    /// 253 blocks and is an S(4,7,23) (`is_steiner_system`, exact); the sums `h0 + h` of
    /// heptads span a code of dimension 11. A failure is `Err(Construction)`, a defect.
    pub fn published() -> Result<M23, M23Error> {
        let g1 = from_cycles_1based(&G1_CYCLES)?;
        let g2 = from_cycles_1based(&[G2_CYCLE])?;
        let g3 = from_cycles_1based(&[G3_CYCLE])?;
        if rmul(&rmul(&g1, &g2), &g3) != identity() {
            return Err(M23Error::Construction("g1 g2 g3 is not 1".into()));
        }
        let chain = StabilizerChain::new(
            &[to_image_list(&g1).to_vec(), to_image_list(&g2).to_vec()],
            23,
        )
        .map_err(M23Error::Construction)?;
        if chain.order().to_string() != M23_ORDER.to_string() {
            return Err(M23Error::Construction(format!(
                "Schreier-Sims order {} instead of {M23_ORDER}",
                chain.order()
            )));
        }
        // The Steiner system: the orbit of the heptad Fix(g1) under <g1, g2>.
        let start = fixed_mask(&g1);
        if start.count_ones() != 7 {
            return Err(M23Error::Construction("g1 does not fix 7 points".into()));
        }
        let mut seen = std::collections::HashSet::from([start]);
        let mut stack = vec![start];
        while let Some(h) = stack.pop() {
            for g in [&g1, &g2] {
                let k = mask_image(h, g);
                if seen.insert(k) {
                    stack.push(k);
                }
            }
        }
        let mut heptads: Vec<u32> = seen.into_iter().collect();
        heptads.sort_unstable();
        if heptads.len() != 253 {
            return Err(M23Error::Construction(format!(
                "{} heptads instead of 253",
                heptads.len()
            )));
        }
        let blocks: Vec<Vec<usize>> = heptads
            .iter()
            .map(|&h| (0..23).filter(|&i| (h >> i) & 1 == 1).collect())
            .collect();
        match is_steiner_system(23, 4, &blocks) {
            Ok(true) => {}
            other => {
                return Err(M23Error::Construction(format!(
                    "the heptads are not an S(4,7,23): {other:?}"
                )))
            }
        }
        let h0 = heptads[0];
        let golay = echelon_f2(heptads.iter().map(|&h| h ^ h0));
        if golay.len() != 11 {
            return Err(M23Error::Construction(format!(
                "the even Golay code has dimension {}",
                golay.len()
            )));
        }
        Ok(M23 {
            g1,
            g2,
            g3,
            chain,
            heptads,
            golay,
        })
    }

    /// The published involution `g1` (class 2A).
    pub fn g1(&self) -> Perm23 {
        self.g1
    }

    /// The published 23-cycle `g2`; its class is 23A by definition (HANDOFF Section 1).
    pub fn g2(&self) -> Perm23 {
        self.g2
    }

    /// The published 23-cycle `g3 = (g1 g2)^-1` (class 23B).
    pub fn g3(&self) -> Perm23 {
        self.g3
    }

    /// The stabilizer chain of `<g1, g2>` (deterministic Schreier-Sims).
    pub fn chain(&self) -> &StabilizerChain {
        &self.chain
    }

    /// The lengths of the basic orbits, `[23, 22, 21, 20, 3, 16]` for these generators.
    pub fn base_orbit_lengths(&self) -> Vec<usize> {
        (0..self.chain.num_levels())
            .map(|i| self.chain.orbit(i).len())
            .collect()
    }

    /// Membership by sifting.
    pub fn contains(&self, p: &Perm23) -> bool {
        self.chain.contains(&to_image_list(p))
    }

    /// Calls `f` on each of the 10,200,960 elements once, as a [`Perm23`]. It wraps
    /// [`StabilizerChain::for_each_element`] and converts each image list in one reused buffer.
    pub fn for_each_element(&self, mut f: impl FnMut(&Perm23)) {
        let mut p = [0u8; 23];
        self.chain.for_each_element(|g| {
            for (out, &x) in p.iter_mut().zip(g) {
                *out = x as u8;
            }
            f(&p);
        });
    }

    /// The 253 heptads (blocks of the S(4,7,23)) as bit masks, sorted.
    pub fn heptads(&self) -> &[u32] {
        &self.heptads
    }

    /// The reduced echelon basis `(v_k, i_k)` of the even Golay code `G0` (11 rows): `v_k` is a
    /// 23-bit word, `i_k` its pivot (highest bit); `v_k` is 0 at the other pivots.
    pub fn even_golay_basis(&self) -> &[(u32, u8)] {
        &self.golay
    }
}
